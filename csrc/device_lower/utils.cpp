// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#include <device_lower/utils.h>

#include <ATen/cuda/CUDAContext.h>
#include <device_lower/analysis/thread_predicate.h>
#include <device_lower/lower2device.h>
#include <device_lower/utils.h>
#include <id_model/utils.h>
#include <ir/iostream.h>
#include <ir/utils.h>
#include <iter_visitor.h>
#include <kernel_ir_dispatch.h>
#include <logical_domain_map.h>
#include <ops/arith.h>
#include <val_graph_visitor.h>

#include <expr_simplifier.h>
#include <algorithm>
#include <deque>
#include <memory>

// TODO: refactor this file (one per namespace)

namespace nvfuser {

namespace scope_utils {

//! Create an **empty** Forloop and copy the metadata.
ForLoop* cloneForLoop(ForLoop* for_loop) {
  return IrBuilder::create<ForLoop>(for_loop);
}

//! Create an **empty** IfThenElse and copy the metadata.
kir::IfThenElse* cloneIfThenElse(kir::IfThenElse* ite) {
  return IrBuilder::create<kir::IfThenElse>(ite->predicate());
}

} // namespace scope_utils

namespace ir_utils {

std::vector<IterDomain*> iterDomainInputsOf(
    const std::vector<IterDomain*>& input_ids,
    const std::vector<IterDomain*>& all_inputs) {
  auto inputs = IterVisitor::getInputsTo(
      {input_ids.begin(), input_ids.end()},
      {all_inputs.begin(), all_inputs.end()});
  std::vector<IterDomain*> id_inputs(
      ir_utils::filterByType<IterDomain>(inputs).begin(),
      ir_utils::filterByType<IterDomain>(inputs).end());
  return id_inputs;
}

std::vector<IterDomain*> iterDomainInputsOfOrderedAs(
    const std::vector<IterDomain*>& of,
    const std::vector<IterDomain*>& order) {
  auto inputs_vec = iterDomainInputsOf(of, order);

  std::unordered_set<IterDomain*> inputs_set(
      inputs_vec.begin(), inputs_vec.end());

  std::vector<IterDomain*> ordered_inputs;
  std::copy_if(
      order.begin(),
      order.end(),
      std::back_inserter(ordered_inputs),
      [&inputs_set](const auto& id) {
        return inputs_set.find(id) != inputs_set.end();
      });

  return ordered_inputs;
}

bool isTV(const Val* val) {
  return val->getValType().value() == ValType::TensorView ||
      val->getValType().value() == ValType::TensorIndex;
}

// Check if we're a TensorView op that we can generate code for.
bool isTvOp(const Expr* expr) {
  if (std::ranges::any_of(expr->outputs(), [](Val* v) { return isTV(v); }) &&
      (expr->isOneOf<
          ArgsortOp,
          GroupedMmaOp,
          ScaledMmaOp,
          CutlassNvfp4GroupedMmaOp,
          TopKOp,
          UnaryOp,
          BinaryOp,
          TernaryOp,
          TensorConstruct,
          SelectOp,
          IndexSelectOp,
          IndexPutAccumulateOp,
          GatherOp,
          ScatterOp,
          RNGOp,
          FullOp,
          IotaOp,
          EyeOp,
          ReductionOp,
          GroupedReductionOp,
          WelfordOp,
          GroupedWelfordOp,
          LoadStoreOp,
          MatmulOp,
          MmaOp,
          LinearOp,
          SdpaFwdOp,
          SdpaBwdOp,
          EmbeddingFwdOp,
          BroadcastOp,
          SqueezeOp,
          ExpandOp,
          RepeatOp,
          ViewAsScalar,
          ReshapeOp,
          PadOp,
          SliceOp,
          CatOp,
          ScanOp,
          kir::AllocTMem,
          kir::GridReduction,
          kir::GroupedGridReduction,
          kir::GridBroadcast,
          kir::GridWelford,
          kir::GroupedGridWelford,
          kir::VectorizedWelfordOp,
          kir::RNGOp>())) {
    return true;
  }
  return false;
}

bool isLdMatrixOp(const Expr* expr) {
  if (auto ldst = dynamic_cast<const LoadStoreOp*>(expr)) {
    return ldst->opType() == LoadStoreOpType::LdMatrix;
  }
  return false;
}

bool isStMatrixOp(const Expr* expr) {
  if (auto ldst = dynamic_cast<const LoadStoreOp*>(expr)) {
    return ldst->opType() == LoadStoreOpType::StMatrix;
  }
  return false;
}

bool isCpAsyncOp(const Expr* expr) {
  if (auto ldst = dynamic_cast<const LoadStoreOp*>(expr)) {
    return ldst->opType() == LoadStoreOpType::CpAsync;
  }
  return false;
}

namespace {

enum class CpAsyncBulkMode { G2S, S2G, NotACpAsyncBulk };

inline CpAsyncBulkMode getCpAsyncBulkMode(const Expr* expr) {
  // Attempt to cast to LoadStoreOp
  if (auto ldst = dynamic_cast<const LoadStoreOp*>(expr)) {
    // Check if opType is either CpAsyncBulk or CpAsyncBulkTensorTile
    auto op_type = ldst->opType();
    if (op_type == LoadStoreOpType::CpAsyncBulk ||
        op_type == LoadStoreOpType::CpAsyncBulkTensorTile) {
      // Check memory types
      auto in_mem = getTv(ldst->in())->getMemoryType();
      auto out_mem = getTv(ldst->out())->getMemoryType();
      if (in_mem == MemoryType::Global && out_mem == MemoryType::Shared) {
        return CpAsyncBulkMode::G2S;
      } else if (
          in_mem == MemoryType::Shared && out_mem == MemoryType::Global) {
        return CpAsyncBulkMode::S2G;
      } else {
        NVF_THROW("Invalid memory types for CpAsyncBulk or CpAsyncBulkTile");
      }
    }
  }
  return CpAsyncBulkMode::NotACpAsyncBulk;
}

} // namespace

bool isCpAsyncBulk(const Expr* expr) {
  return getCpAsyncBulkMode(expr) != CpAsyncBulkMode::NotACpAsyncBulk;
}

bool isCpAsyncBulkLoad(const Expr* expr) {
  return getCpAsyncBulkMode(expr) == CpAsyncBulkMode::G2S;
}

bool isCpAsyncBulkStore(const Expr* expr) {
  return getCpAsyncBulkMode(expr) == CpAsyncBulkMode::S2G;
}

// return true if expr is nD TMA load or store.
// nD TMA ops handles out of bound accesses automatically in hardware, no need
// to predicate it.
bool isCpAsyncBulkTensorTile(const Expr* expr) {
  return isCpAsyncBulk(expr) &&
      expr->as<LoadStoreOp>()->opType() ==
      LoadStoreOpType::CpAsyncBulkTensorTile;
}
bool isCpAsyncBulkTensorTileLoad(const Expr* expr) {
  return isCpAsyncBulkLoad(expr) &&
      expr->as<LoadStoreOp>()->opType() ==
      LoadStoreOpType::CpAsyncBulkTensorTile;
}
bool isCpAsyncBulkTensorTileStore(const Expr* expr) {
  return isCpAsyncBulkStore(expr) &&
      expr->as<LoadStoreOp>()->opType() ==
      LoadStoreOpType::CpAsyncBulkTensorTile;
}

bool isCpAsyncBulk1D(const Expr* expr) {
  return isCpAsyncBulk(expr) &&
      expr->as<LoadStoreOp>()->opType() == LoadStoreOpType::CpAsyncBulk;
}
bool isCpAsyncBulk1DLoad(const Expr* expr) {
  return isCpAsyncBulkLoad(expr) &&
      expr->as<LoadStoreOp>()->opType() == LoadStoreOpType::CpAsyncBulk;
}
bool isCpAsyncBulk1DStore(const Expr* expr) {
  return isCpAsyncBulkStore(expr) &&
      expr->as<LoadStoreOp>()->opType() == LoadStoreOpType::CpAsyncBulk;
}

bool isLdStTMem(const Expr* expr) {
  if (auto ldst = dynamic_cast<const LoadStoreOp*>(expr)) {
    return ldst->opType() == LoadStoreOpType::LdTMem ||
        ldst->opType() == LoadStoreOpType::StTMem;
  }
  return false;
}

bool isTensorScalarFillOp(const Expr* expr) {
  // Check that the input is a single scalar.
  if (expr->inputs().size() == 1 && expr->input(0)->isScalar()) {
    // All load store op with a single scalar input
    //  should be a scalar filling op. Semantically
    //  it literally means `Store`'ing a scalar
    //  into a tensor.
    if (expr->isA<LoadStoreOp>()) {
      return true;
    }
  }
  // Ideally any scalar expression that outputs
  //  to a tensor should be considered in this function
  //  but since we currently only limit scope to
  //  initialization patterns so other scalar expr's
  //  are low priority and are excluded here to avoid confusion.
  return false;
}

TensorView* getTv(Val* val) {
  // NOLINTNEXTLINE(cppcoreguidelines-pro-type-const-cast)
  return const_cast<TensorView*>(getTv(const_cast<const Val*>(val)));
}

const TensorView* getTv(const Val* val) {
  if (val->isA<TensorView>()) {
    return val->as<TensorView>();
  } else if (val->isA<kir::TensorIndex>()) {
    return val->as<kir::TensorIndex>()->view();
  }
  return nullptr;
}

std::vector<TensorView*> getTvs(const std::vector<Val*>& vals) {
  std::vector<TensorView*> tvs;
  for (auto val : vals) {
    auto tv = ir_utils::getTv(val);
    if (tv) {
      tvs.emplace_back(tv);
    }
  }
  return tvs;
}

bool isScalarOp(const Expr* expr) {
  for (auto out : expr->outputs()) {
    if (!out->isScalar()) {
      return false;
    }
  }
  return true;
}

bool isIterDomainOp(const Expr* expr) {
  return expr->isOneOf<Split, Merge, Swizzle, Swizzle2D, Resize>();
}

std::optional<std::pair<IterDomain*, IterDomain*>> getMaybeWarpReductionDim(
    const Val* output,
    const Val* input) {
  auto tv_out = getTv(output);
  if (tv_out == nullptr) {
    return std::nullopt;
  }

  auto tv_in = getTv(input);
  // only support reducing to registers for now.
  if (tv_in->getMemoryType() != MemoryType::Local ||
      tv_out->getMemoryType() != MemoryType::Local) {
    return std::nullopt;
  }

  IterDomain* reduction_on_xdim = nullptr;
  IterDomain* reduction_on_ydim = nullptr;
  IterDomain* reduction_on_zdim = nullptr;
  for (auto id : tv_out->getLoopDomain()) {
    // Currently warp reduction only allows:
    // (1) block.x parallel reductions
    // (2) block.x and block.y parallel reductions
    if (id->isReduction() && id->isParallelized()) {
      if (id->getParallelType() == ParallelType::TIDx) {
        reduction_on_xdim = id;
      } else if (id->getParallelType() == ParallelType::TIDy) {
        reduction_on_ydim = id;
      } else if (id->getParallelType() == ParallelType::TIDz) {
        reduction_on_zdim = id;
      }
    }
  }
  if (!reduction_on_xdim) {
    return std::nullopt;
  }

  if (!reduction_on_xdim->start()->isZeroInt()) {
    return std::nullopt;
  }

  // reduction only in xdim.
  if (!reduction_on_ydim && !reduction_on_zdim) {
    if (reduction_on_xdim->hasPaddingToMultipleOfWarp()) {
      return std::make_pair(reduction_on_xdim, nullptr);
    }

    if (reduction_on_xdim->extent()->isConstInt()) {
      auto extent_value = reduction_on_xdim->extent()->evaluate();
      if (extent_value % at::cuda::warp_size() == 0) {
        return std::make_pair(reduction_on_xdim, nullptr);
      }
    }
  } else if (reduction_on_xdim && reduction_on_ydim && reduction_on_zdim) {
    // special case used in innerOuter scheduler where bdimx and bdimy are
    // constants bdimz is always 1.
    if (reduction_on_xdim->extent()->isConstInt() &&
        reduction_on_ydim->extent()->isConstInt()) {
      auto extent_x_value = reduction_on_xdim->extent()->evaluate();
      auto extent_y_value = reduction_on_ydim->extent()->evaluate();
      if ((extent_x_value * extent_y_value) % at::cuda::warp_size() == 0) {
        return std::make_pair(reduction_on_xdim, reduction_on_ydim);
      }
    }
  }
  return std::nullopt;
}

std::unordered_map<ParallelType, IterDomain*> getParallelDomains(
    const Val* val) {
  const TensorView* tv = nullptr;
  if (val->isA<TensorView>()) {
    tv = val->as<TensorView>();
  } else if (val->isA<kir::TensorIndex>()) {
    tv = val->as<kir::TensorIndex>()->view();
  } else {
    NVF_THROW("Provided val is not TensorIndex or TensorView.");
  }

  std::unordered_map<ParallelType, IterDomain*> parallel_domains;
  for (auto d : tv->getLoopDomain()) {
    if (d->isThread()) {
      parallel_domains.insert(std::make_pair(d->getParallelType(), d));
    }
  }
  return parallel_domains;
}

bool isCpAsyncInit(const Expr* expr) {
  return isTensorScalarFillOp(expr) &&
      // FIXME:
      //  We'd need to add a flag to all the init
      //   exprs so we could robustly detect initialization
      //   in all cases.
      isCpAsyncOp(getTvOutput(expr)->definition());
}

std::optional<Expr*> getMaybePredicatedSingleton(Expr* expr) {
  if (auto ite = dynamic_cast<kir::IfThenElse*>(expr)) {
    if (ite->elseBody().empty()) {
      if (ite->thenBody().size() == 1) {
        return ite->thenBody().exprs()[0];
      }
    }
  }
  return std::nullopt;
}

//! Short-cut for checking if the expression loads from global memory.
bool isGlobalLoad(const Expr* expr) {
  if (expr->isA<LoadStoreOp>()) {
    if (auto in_tv = getTv(expr->input(0))) {
      return in_tv->getMemoryType() == MemoryType::Global;
    }
  }
  return false;
}

//! Short-cut for checking if the given expression initializes buffers
//!  for global memory load.
bool isGlobalLoadInit(const Expr* expr) {
  if (auto uop = dynamic_cast<const UnaryOp*>(expr)) {
    if (uop->in()->isScalar()) {
      // FIXME:
      //  We'd need to add a flag to all the init
      //   exprs so we could robustly detect initialization
      //   in all cases.
      if (isGlobalLoad(getTvOutput(uop)->definition())) {
        return true;
      }
    }
  }
  return false;
}

namespace {

class ExprFlattener : private kir::IrVisitor {
 private:
  using kir::IrVisitor::handle;

  void dispatch(Expr* expr) final {
    if (expr->isA<ForLoop>() || expr->isA<kir::IfThenElse>()) {
      kir::IrVisitor::dispatch(expr);
    } else {
      flat_exprs_.push_back(expr);
    }
  }

 private:
  std::vector<Expr*> flat_exprs_;

 public:
  //! Flattens scopes extracting out a single ordered list of exprs.
  static std::vector<Expr*> flatten(const std::vector<Expr*>& loop_nests) {
    ExprFlattener flattener;
    for (auto expr : loop_nests) {
      flattener.dispatch(expr);
    }
    return flattener.flat_exprs_;
  }
};

} // namespace

std::vector<Expr*> flattenScopedExprs(const std::vector<Expr*>& loop_nests) {
  return ExprFlattener::flatten(loop_nests);
}

namespace {

class ReplaceExprInput : private kir::ExprMutator {
 public:
  static std::vector<Expr*> replace(
      const std::vector<Expr*>& exprs,
      const std::unordered_map<Val*, Val*>& replacement_map) {
    ReplaceExprInput replacer(replacement_map);
    replacer.traverseAndInsert(exprs);
    return replacer.exprs_;
  }

 private:
  ReplaceExprInput(const std::unordered_map<Val*, Val*>& replacement_map)
      : replacement_map_(replacement_map) {}

  using kir::ExprMutator::handle;

  std::optional<std::unordered_map<Val*, Val*>> getMaybeInputReplacementMap(
      Expr* expr) {
    bool need_replacement = false;

    std::unordered_map<Val*, Val*> replaced_val;
    for (auto in : expr->inputs()) {
      auto replace_it = replacement_map_.find(in);
      if (replace_it != replacement_map_.end()) {
        need_replacement = true;
        replaced_val[in] = replace_it->second;
      } else {
        replaced_val[in] = in;
      }
    }
    if (need_replacement) {
      return std::optional<std::unordered_map<Val*, Val*>>(replaced_val);
    } else {
      return std::nullopt;
    }
  }

  // Copy predicates and register expression replacement
  void registerReplaceWithPredicate(Expr* old_expr, Expr* new_expr) {
    new_expr = new_expr->withPredicate(old_expr->predicate())
                   ->withWritePredicate(old_expr->writePredicate());
    registerReplace(old_expr, new_expr);
  }

  void handle(UnaryOp* node) final {
    auto replaced_inputs = getMaybeInputReplacementMap(node);
    if (replaced_inputs.has_value()) {
      auto replacement = IrBuilder::create<UnaryOp>(
          node->getUnaryOpType(), node->out(), replaced_inputs->at(node->in()));
      registerReplaceWithPredicate(node, replacement);
    }
  }

  void handle(BinaryOp* node) final {
    auto replaced_inputs = getMaybeInputReplacementMap(node);
    if (replaced_inputs.has_value()) {
      auto replacement = IrBuilder::create<BinaryOp>(
          node->getBinaryOpType(),
          node->out(),
          replaced_inputs->at(node->lhs()),
          replaced_inputs->at(node->rhs()));
      registerReplaceWithPredicate(node, replacement);
    }
  }

  void handle(TernaryOp* node) final {
    auto replaced_inputs = getMaybeInputReplacementMap(node);
    if (replaced_inputs.has_value()) {
      auto replacement = IrBuilder::create<TernaryOp>(
          node->getTernaryOpType(),
          node->out(),
          replaced_inputs->at(node->in1()),
          replaced_inputs->at(node->in2()),
          replaced_inputs->at(node->in3()));
      registerReplaceWithPredicate(node, replacement);
    }
  }

  void handle(RNGOp* node) final {
    // RNGOp has no input
    return;
  }

  void handle(kir::RNGOp* node) final {
    auto replaced_inputs = getMaybeInputReplacementMap(node);
    if (replaced_inputs.has_value()) {
      kir::RNGOp* replacement = nullptr;
      if (node->inputs().size() == 4) {
        replacement = IrBuilder::create<kir::RNGOp>(
            node->output(0),
            replaced_inputs->at(node->input(0)),
            replaced_inputs->at(node->input(1)),
            node->dtype(),
            node->getRNGOpType(),
            std::vector<Val*>{
                replaced_inputs->at(node->input(2)),
                replaced_inputs->at(node->input(3))});
      } else {
        replacement = IrBuilder::create<kir::RNGOp>(
            node->output(0),
            replaced_inputs->at(node->input(0)),
            replaced_inputs->at(node->input(1)),
            node->dtype(),
            node->getRNGOpType());
      }
      registerReplaceWithPredicate(node, replacement);
    }
  }

  void handle(ReductionOp* node) final {
    auto replaced_inputs = getMaybeInputReplacementMap(node);
    if (replaced_inputs.has_value()) {
      auto replacement = IrBuilder::create<ReductionOp>(
          node->getReductionOpType(),
          node->init(),
          node->out(),
          replaced_inputs->at(node->in()),
          node->isAllreduce());
      registerReplaceWithPredicate(node, replacement);
    }
  }

  void handle(GroupedReductionOp* node) final {
    auto replaced_inputs = getMaybeInputReplacementMap(node);
    if (replaced_inputs.has_value()) {
      const auto& map = replaced_inputs.value();
      auto inputs = node->inputs();
      for (auto& input : inputs) {
        auto it = map.find(input);
        if (it != map.end()) {
          input = it->second;
        }
      }
      auto replacement = IrBuilder::create<GroupedReductionOp>(
          node->getReductionOpTypes(),
          node->initVals(),
          node->outputs(),
          inputs,
          node->isAllreduce());
      registerReplaceWithPredicate(node, replacement);
    }
  }
  void handle(BroadcastOp* node) final {
    auto replaced_inputs = getMaybeInputReplacementMap(node);
    if (replaced_inputs.has_value()) {
      auto replacement = IrBuilder::create<BroadcastOp>(
          node->out(),
          replaced_inputs->at(node->in()),
          node->getBroadcastDimFlags());
      registerReplaceWithPredicate(node, replacement);
    }
  }

  void handle(WelfordOp* node) final {
    auto replaced_inputs = getMaybeInputReplacementMap(node);
    if (replaced_inputs.has_value()) {
      auto replacement = IrBuilder::create<WelfordOp>(
          node->outAvg(),
          node->outVar(),
          node->outN(),
          node->initAvg(),
          node->initVar(),
          node->initN(),
          replaced_inputs->at(node->inAvg()),
          replaced_inputs->at(node->inVar()),
          replaced_inputs->at(node->inN()));
      registerReplaceWithPredicate(node, replacement);
    }
  }

  void handle(MmaOp* node) final {
    auto replaced_inputs = getMaybeInputReplacementMap(node);
    if (replaced_inputs.has_value()) {
      auto replacement = IrBuilder::create<MmaOp>(
          node->out(),
          replaced_inputs->at(node->inA()),
          replaced_inputs->at(node->inB()),
          node->init(),
          node->macro());
      registerReplaceWithPredicate(node, replacement);
    }
  }

  void handle(LoadStoreOp* node) final {
    auto replaced_inputs = getMaybeInputReplacementMap(node);
    if (replaced_inputs.has_value()) {
      auto replacement = IrBuilder::create<LoadStoreOp>(
          node->opType(), node->out(), node->in(), node->cacheOp());
      registerReplaceWithPredicate(node, replacement);
    }
  }

 private:
  const std::unordered_map<Val*, Val*>& replacement_map_;
};

} // namespace

std::vector<Expr*> replaceInputsInExpr(
    const std::vector<Expr*>& exprs,
    const std::unordered_map<Val*, Val*>& replacement_map) {
  return ReplaceExprInput::replace(exprs, replacement_map);
}

std::vector<Expr*> getAllSwizzlesBetween(
    std::vector<IterDomain*> from,
    std::vector<IterDomain*> to) {
  auto all_expr = DependencyCheck::getAllExprsBetween(
      {from.begin(), from.end()}, {to.begin(), to.end()});

  std::vector<Expr*> all_swizzles;

  std::copy_if(
      all_expr.begin(),
      all_expr.end(),
      std::back_inserter(all_swizzles),
      [](Expr* expr) { return expr->isA<Swizzle2D>(); });

  return all_swizzles;
}

bool isTMAOrMMASmemTv(TensorView* tv) {
  return tv->getMemoryType() == MemoryType::Shared &&
      (ir_utils::isCpAsyncBulkLoad(tv->definition()) ||
       std::ranges::any_of(tv->uses(), [](Expr* e) {
         return e->isA<MmaOp>() || ir_utils::isCpAsyncBulkStore(e);
       }));
}

MmaInputSmemSwizzle getSwizzleMode(TensorView* tv) {
  // Output of TMA load
  if (ir_utils::isCpAsyncBulkLoad(tv->definition())) {
    return GpuLower::current()->consumerToTMAInfo().at(tv).swizzle();
  }
  for (auto use : tv->uses()) {
    // Input of TMA store
    if (ir_utils::isCpAsyncBulkStore(use)) {
      TensorView* consumer_tv = ir_utils::getTvOutput(use);
      return GpuLower::current()->consumerToTMAInfo().at(consumer_tv).swizzle();
    }

    // Input of MmaOp
    if (use->isA<MmaOp>()) {
      TensorView* consumer_tv = ir_utils::getTvOutput(use);
      auto id_graph = GpuLower::current()->tensorIndexer().traversalGraph();
      const auto& to_domain = id_graph.toGroups(tv->getMaybeAllocationDomain());
      const auto& from_domain = id_graph.toGroups(consumer_tv->getLoopDomain());
      auto exprs = ValGraphBFS::getExprGroupsBetween(
                       id_graph,
                       {from_domain.begin(), from_domain.end()},
                       {to_domain.begin(), to_domain.end()},
                       false)
                       .first;
      for (const auto& [eg, dir] : exprs) {
        auto expr = eg->front();
        if (Swizzle* swizzle = dynamic_cast<Swizzle*>(expr)) {
          NVF_ERROR(
              swizzle->swizzleType() == SwizzleType::XOR, "expect xor swizzle");
          return getSwizzleFromBytes(
              swizzle->inX()->extent()->evaluate().as<int64_t>() * 16);
        }
      }
    }
  }
  return MmaInputSmemSwizzle::None;
}

std::optional<int64_t> getStageSlicePosition(const TensorView* tv) {
  NVF_ERROR(tv != nullptr);

  bool is_warp_specialized =
      std::holds_alternative<WarpSpecialized>(tv->circularBufferOptions().type);
  if (!is_warp_specialized) {
    return std::nullopt;
  }

  const auto& warp_specialized =
      std::get<WarpSpecialized>(tv->circularBufferOptions().type);
  if (!warp_specialized.stage_slice_position.has_value()) {
    return std::nullopt;
  }

  return warp_specialized.stage_slice_position.value();
}

// Returns true if the for_loops contain a loop with the given
// CircularBufferLoopStage.
bool containsCircularBufferStage(
    const std::vector<ForLoop*>& for_loops,
    CircularBufferLoopStage stage_type) {
  return std::any_of(
      for_loops.begin(), for_loops.end(), [stage_type](const ForLoop* fl) {
        return fl->circularBufferLoopStage() == stage_type;
      });
}

} // namespace ir_utils

namespace lower_utils {

bool hasBlockSync(const Expr* expr, const ThreadPredicateMap& pred_map) {
  if (expr->isA<kir::BlockSync>() || expr->isA<kir::GridSync>() ||
      expr->isA<kir::BlockSerializeWait>() ||
      expr->isA<kir::BlockSerializeRelease>()) {
    return true;
  }

  if (!ir_utils::isTvOp(expr)) {
    return false;
  }

  if (auto gr = dynamic_cast<const kir::GridReduction*>(expr);
      gr && gr->isSerial()) {
    // Serial GridReductions do not have a block sync. Instead, they sync in
    // separate nodes surrounding their loop nest.
    return false;
  }

  // GroupedReductionOp can have multiple output TVs, but they must be
  // parallelized in the same way, so just checking one of them is enough.
  auto tv = ir_utils::getTvOutput(expr);

  if (ir_utils::isReductionOp(expr) &&
      (tv->hasBlockReduction() || tv->hasGridReduction())) {
    return true;
  }

  if ((expr->isA<BroadcastOp>() &&
       GpuLower::current()
           ->info()
           .threadPredicateMap()
           .getParallelBroadcastDomains(tv)
           .any()) ||
      expr->isA<kir::GridBroadcast>()) {
    return true;
  }

  // These ops currently use CUB, which uses syncthreads internally
  if (expr->isOneOf<ArgsortOp, ScanOp, TopKOp>()) {
    return true;
  }

  return false;
}

kir::Allocate* allocGlobalBufferForGridComm(
    Val* buffer_size,
    DataType dtype,
    bool zero_init,
    bool resets_to_zero) {
  const std::vector<IterDomain*> new_buffer_ids = {
      IrBuilder::create<IterDomain>(IterDomainBuilder(
          GpuLower::current()->kernel()->zeroVal(),
          SimplifyingIrBuilder::maybeCastExpr(DataType::Index, buffer_size)))};
  const auto buffer_domain = IrBuilder::create<TensorDomain>(new_buffer_ids);
  const auto buffer_tv =
      IrBuilder::create<TensorView>(buffer_domain, dtype, MemoryType::Global);
  return IrBuilder::create<kir::Allocate>(
      buffer_tv,
      buffer_tv->getMemoryType(),
      nullptr,
      zero_init,
      resets_to_zero);
}

AllocPosInfo getAllocPosInfo(
    const TensorView* tv,
    const std::vector<ForLoop*>& for_loops,
    const std::unordered_map<IterDomain*, IterDomain*>& id_map,
    bool use_id_map) {
  DEBUG_PRINT_SCOPE(tv);
  AllocPosInfo info;
  auto gpu_lower = GpuLower::current();

  bool outer_alloc_found = false;

  // Use stage_slice_position if it exists for TensorView. Otherwise, fallback
  // to compute_at_position.
  int64_t compute_position =
      ir_utils::getStageSlicePosition(tv).value_or(tv->getComputeAtPosition());

  for (auto fl : for_loops) {
    if (info.alloc_pos == compute_position) {
      DEBUG_LOG("Break at info.alloc_pos = ", info.alloc_pos);
      break;
    }

    if (tv->axis(info.alloc_pos)->isReduction()) {
      const auto outputs = FusionGuard::getCurFusion()->getTerminatingOutputs();
      NVF_ERROR(
          std::find(outputs.begin(), outputs.end(), tv) != outputs.end(),
          "Invalid computeAt of T",
          tv->name(),
          ". A reducation axis is detected outside computeAt point even though "
          "it is not an output tensor.");
      DEBUG_LOG("Break at info.alloc_pos = ", info.alloc_pos);
      break;
    }

    auto fl_id = fl->iter_domain();

    if (fl_id->getParallelType() == ParallelType::Unroll) {
      DEBUG_LOG("Break at info.alloc_pos = ", info.alloc_pos);
      break;
    }

    // Shared memory must be allocated outside of unswitched
    // domains. See issue #1133.
    if (fl_id->getParallelType() == ParallelType::Unswitch &&
        tv->getMemoryType() == MemoryType::Shared) {
      outer_alloc_found = true;
    }

    // Assume global memory is allocated at outer most scope.
    if (tv->getMemoryType() == MemoryType::Global) {
      outer_alloc_found = true;
    }

    // Allocation of a circular buffered tensor is placed outside its
    // circular buffer axis.
    if (tv->isCircularBuffered() &&
        tv->axis(info.alloc_pos) ==
            gpu_lower->circularBufferInfo().getCircularBufferAxis(tv)) {
      outer_alloc_found = true;
    }

    auto local_id = tv->axis(info.alloc_pos);

    if (use_id_map) {
      auto id_it = id_map.find(local_id);
      if (id_it != id_map.end()) {
        local_id = id_it->second;
      }
    }

    if (lower_utils::getConcreteLoopID(local_id) ==
        lower_utils::getConcreteLoopID(fl_id)) {
      info.alloc_pos++;
    }

    info.init_for_loop = fl;

    if (!outer_alloc_found) {
      info.alloc_for_loop = fl;
    }
  }

  return info;
}

//! Implementing this in here to avoid including too many headers
//!  in type.cpp. Conceptually this should be a generic definition
//!  rather than a util.
bool supportInlinePredicate(Expr* expr) {
  if (ir_utils::isCpAsyncOp(expr)) {
    return true;
  }
  // TODO: build out support.
  return false;
}

bool isScalarExpr(Expr* expr) {
  if (expr->inputs().empty() || expr->outputs().empty()) {
    // For expressions that does not have input/output, they are usually lowered
    // expressions like AsyncWait. We don't consider these as scalar
    // expressions.
    return false;
  }
  for (auto inp : expr->inputs()) {
    if (!inp->isScalar()) {
      return false;
    }
  }
  for (auto out : expr->outputs()) {
    if (!out->isScalar()) {
      return false;
    }
  }
  return true;
}

bool isExtentEqualToMaxParallelTypeExtent(
    const IterDomain* id,
    bool in_compute_warp) {
  const auto& parallel_dim_map =
      GpuLower::current()->info().parallelDimensionMap();
  Val* pdm_max_extent = nullptr;
  if (in_compute_warp) {
    pdm_max_extent = parallel_dim_map.getRawCompute(id->getParallelType());
  } else {
    pdm_max_extent = parallel_dim_map.getRaw(id->getParallelType());
  }
  if (nullptr == pdm_max_extent) {
    return false;
  }
  auto* is_exact_val = IrBuilder::eqExpr(id->extent(), pdm_max_extent);
  return simplifyExpr(is_exact_val)->isTrue();
}

Val* u32IndexScalarSmemTv(TensorView* smem_tv) {
  auto u32addr = IrBuilder::create<Val>(DataType::SMemAddress);
  IrBuilder::create<UnaryOp>(
      UnaryOpType::ToUnsignedSmemAddr,
      u32addr,
      IrBuilder::metadataExpr(smem_tv));
  return u32addr;
}

Val* u32IndexScalarSmemTv(kir::TensorIndex* index) {
  auto ptr_address = IrBuilder::addressExpr(index);
  auto u32addr = IrBuilder::create<Val>(DataType::SMemAddress);
  IrBuilder::create<UnaryOp>(
      UnaryOpType::ToUnsignedSmemAddr, u32addr, ptr_address);
  return u32addr;
}

Val* getGridSyncBufferSize(const ParallelTypeBitmap& ptb) {
  // See the comment above for getGridCommWorkBufferSize.
  NVF_ERROR(
      ptb.hasBID(),
      "Detected  needing a grid sync but no grid bits set in bitmap.");
  Val* buffer_size = GpuLower::current()->kernel()->oneVal();
  for (auto pt : kParallelTypeBIDs) {
    // Synchronized within pt, so all blocks of this PT use the same
    // sync buffer location, and thus no need to expand the sync
    // buffer size.
    if (ptb.get(pt)) {
      continue;
    }
    auto pt_dim = GpuLower::current()->info().parallelDimensionMap().get(pt);
    if (pt_dim == nullptr || pt_dim->isOneInt()) {
      continue;
    }
    buffer_size = SimplifyingIrBuilder::mulExpr(buffer_size, pt_dim);
  }
  return buffer_size;
}

std::vector<Val*> getFusionOutputsRequiringCodegen(Fusion* fusion) {
  std::vector<Val*> outs_requiring_codegen;
  outs_requiring_codegen.reserve(fusion->outputs().size());
  std::copy_if(
      fusion->outputs().begin(),
      fusion->outputs().end(),
      std::back_inserter(outs_requiring_codegen),
      [&fusion](Val* out) {
        return (fusion->getOutputAlias(out).type != AllocationType::Evaluate);
      });
  return outs_requiring_codegen;
}

Val* getNumThreadsInTensorView(TensorView* tv) {
  Val* num_threads = tv->fusion()->oneVal();
  for (auto id : tv->getLoopDomain()) {
    if (id->isThreadDim()) {
      num_threads = SimplifyingIrBuilder::mulExpr(num_threads, id->extent());
    }
  }
  return num_threads;
}

std::array<UnitDim, 2> getMmaLayout(const MmaOp* expr) {
  if (isAmpere(expr->macro()) || isTuring(expr->macro())) {
    return {UnitDim::K, UnitDim::K};
  }

  // NOLINTNEXTLINE(cppcoreguidelines-pro-type-member-init)
  std::array<UnitDim, 2> layout;

  auto out_tv = ir_utils::getTv(expr->out());
  IterDomain* reduction_id = nullptr;
  // For hopper matmuls, the mma_result logical domain is reordered as [M, N, K]
  // using commitLeafToLogical. In the split-k case, use the root domain for the
  // mma layout because the k dimension is divided into two iterDomains in the
  // logical domain.
  for (auto id : out_tv->getMaybeRootDomain()) {
    if (id->isReduction()) {
      reduction_id = id;
      break;
    }
  }
  NVF_ERROR(reduction_id != nullptr);

  std::array<TensorView*, 2> inputs = {
      ir_utils::getTv(expr->inA()), ir_utils::getTv(expr->inB())};
  for (auto i : arange(2)) {
    auto in_tv = inputs.at(i);
    if (in_tv->getMemoryType() == MemoryType::Local) {
      layout[i] = UnitDim::K;
      continue;
    }
    NVF_ERROR(in_tv->getMemoryType() == MemoryType::Shared);
    auto out2in =
        PairwiseLogicalDomainMap(in_tv, out_tv).mapConsumerToProducer();
    auto reduction_id_in = out2in.at(reduction_id);
    auto inner_id = in_tv->getMaybeAllocationDomain().back();
    while (inner_id != reduction_id_in && inner_id->definition() != nullptr) {
      inner_id = inner_id->definition()->inputs().back()->as<IterDomain>();
    }
    layout[i] = inner_id == reduction_id_in ? UnitDim::K : UnitDim::M_or_N;
  }

  return layout;
}

bool isReductionInitExpr(const Expr* expr) {
  // False if its output isn't a TensorView
  if (!ir_utils::isTvOp(expr)) {
    return false;
  }
  // False if it doesn't have any reduction axis
  const auto out_tv = ir_utils::getTvOutput(expr);
  if (!out_tv->domain()->hasReduction()) {
    return false;
  }
  // False if it has TensorView inputs as initialization should
  // never use TensorViews
  const auto tv_filter_inp_view =
      ir_utils::filterByType<TensorView>(expr->inputs());
  if (tv_filter_inp_view.begin() != tv_filter_inp_view.end()) {
    return false;
  }
  return true;
}

bool predicateAtEnd(ForLoop* loop) {
  auto loop_id = loop->iter_domain();
  auto split = dynamic_cast<Split*>(loop_id->definition());
  if (split == nullptr) {
    return false;
  }

  bool is_divisible = GpuLower::current()->divisibleSplitSet().count(split) > 0;

  if (!is_divisible) {
    return false;
  }

  // Find the other output of the split
  auto other_out_id =
      split->inner() == loop_id ? split->outer() : split->inner();

  // If the other output is mapped with a vectorized IterDomain,
  // this IterDomain needs to be predicated at each iteration point.
  const auto& other_id_exact_set = FusionInfoGuard::current()
                                       ->idModel()
                                       .idGraph(IdMappingMode::EXACT)
                                       .toGroup(other_out_id);

  if (std::any_of(
          other_id_exact_set->begin(), other_id_exact_set->end(), [](Val* val) {
            return val->as<IterDomain>()->getParallelType() ==
                ParallelType::Vectorize;
          })) {
    return false;
  }

  // Now it is either loop_id is mapped with a vectorized IterDomain
  // or it's an output of view transformations.
  return true;
}

// Implementation of:
//   Val* proveLinearAndGetStride(
//       const ValGraph& id_graph,
//       const ValGroup& linear_g,
//       const ValGroups& domain);
//
// The idea is similar to the vectorization helper in vectorize_helper.h:
// We propagate from linear_g to domain, and figure out how linear_g project to
// domain. From the projection, we will know if linear_g is linear and the
// stride. For example, let's consider the following two schedules (same example
// as the NVFuserTest.ProveLinearAndGetStride test):
//
// v1:
//        I0         I1
//       /  \       /  \.
//          128        128
//          / \        / \.
//         /   \      /   \.
//        /     \    /     \.
//       /       \  /       \.
//      /         \/        64.
//     /          /\       /  \.
//    /          /  \     /    \.
//   16        [2]   8   8      8
//                    \ /
//                    xor
//                    / \.
//                   8   8
//
// v3:
//        I0         I1
//       /  \       /  \.
//          32         256
//          / \        / \.
//         /   \      /   \.
//        /     \    /     \.
//       /       \  /       \.
//      /         \/        64.
//     /          /\       /  \.
//    /          /  \     /    \.
//   4          4    8   8      8
//                    \ /
//                    xor
//                   /   \.
//                  8     8
//
// Suppose that the [2] in v1 is linear_g, and the leaves in v3 are domain.
// Domain is in the order [I0o, I1o, 4, 4, 8, 8, 8]. To figure out if linear_g
// is linear in domain, we start from propagating linear_g back in v1. The first
// step we see is the split of 128 into [2] and 64. From this split, we know
// that when [2] projects to the 128, it projects to the 2 after a 64 on its
// inner. The next step is the split of I1 by 128 in v1. Similarly, we know that
// when [2] projects to I1, it projects to the 2 after a 64 on its inner.
// Because the I1 in v1 and v3 are mapped, then we will continue propagation in
// v3. The next step is to process the split of I1 into I1o and 256 in v3. We
// already know that when [2] projects to I1, it projects to the 2 after a 64 on
// its inner. Because 2 * 64 = 128, which is a factor of 256, we know that the
// 256 is able to fully cover the [2] in I1. Therefore, I1o is unrelated to [2],
// and we only need to focus on the 256. And when the [2] projects to this 256,
// it projects to the 2 after a 64 on its inner. The next step is to process the
// split of 256 into 4 and 64. Because the 64 happens to be the extent of the
// inner of [2] in the 256, we know that the [2] will be fully covered by the
// inner 2 of the 4 of the output of the split. So, we know that, when the [2]
// projects to the 4, it projects to the inner 2 of the 4. Now we have finished
// propagation and reached domain. Because [2] is the inner of the 4, and the 4
// is linear in domain, so we have proved that [2] is linear in domain. In
// domain, the domains on the right of the 4 are 8, 8, 8, so the stride is 8*8*8
// = 512.
namespace {

// From the above example, we can see that how linear_g lives in domain could be
// complicated. It can be, for example:
//   1. linear_g is equivalent to a single ValGroup in domain. For example,
//      linear_g itself is inside domain.
//   2. linear_g is the inner of a ValGroup in domain. For example, something
//      like:
//        x  linear_g
//         \ /
//          g
//      where g is a ValGroup in domain.
//   3. linear_g is the outer of a ValGroup in domain. For example, something
//      like:
//        linear_g   x
//                \ /
//                 g
//      where g is a ValGroup in domain.
//   4. linear_g is the middle of a ValGroup in domain, where on the right,
//      there is a 2. For example, something like:
//        linear_g   2
//                \ /
//            x   g1
//             \ /
//              g
//      where g is a ValGroup in domain.
//   5. linear_g is projected as g1, g2, g3 in domain. For example, something
//      like:
//          linear_g
//            /   \.
//          g1    g23
//                / \.
//               g2  g3
//      where g1, g2, g3 are ValGroups in domain.
//   6. linear_g is projected as the inner 2 of g1, g2, and the outer 4 of g3.
//      For example, something like:
//             linear_g
//              /   \.
//        x    2     4    8
//         \  /       \  /
//          ga         g3
//         /  \.
//        g1   g2
//      where g1, g2, g3 are ValGroups in domain.
//
// We use a dynamic type called Projection to represent structures like above.
// Because dynamic type can be recursive, it is very expressive and can
// represent all the cases above. It is helpful to think of the dynamic type
// Projection as a formal language to describe how linear_g is projected in
// domain. Different types in the dynamic type Projection are different types of
// abstract syntax tree nodes for this language. Note that neither the above
// examples nor the dynamic type Projection is exhaustive. For example,
// "linear_g is projected as the merge of the inner of g with the outer of g in
// reverse order" is not covered. For the case where we can not represent using
// the language of the dynamic type Projection, we will use the std::monostate
// to denote "unknown". In the future, if we need more expressive power, we can
// extend the dynamic type Projection with more types of abstract syntax tree
// nodes. Because the dynamic type Projection is recursive, the theoretical
// upper limit of the expressive power of this design is as high as the world
// that a formal language can describe.

// selected = PartOf(what, inner_extent, selected_extent) represents that the
// selected node is part of `what`. This projection usually comes from merge.
// For example, if we have
//   selected    2
//           \  /
//            g1
// then
//   selected = PartOf(g1, 2, extent_of_selected).
template <typename Projection>
struct PartOf {
  // Part of what?
  std::shared_ptr<Projection> what;
  // The structure of `what` is shown below:
  //   .--------------------------.
  //   | outer | selected | inner |
  //   '--------------------------'
  // `inner_extent` refers to the extent of `inner`, which can also be
  // understood as the stride of `selected` in `what`. If `selected` is the
  // innermost of `what`, then there is nothing on the inner of `selected`. For
  // this case, we say the inner_extent is one, and assign nullptr here.
  Val* inner_extent = nullptr;
  // The extent of the `selected`. This value is just carried over and never
  // changed.
  Val* selected_extent = nullptr;

  bool operator==(const PartOf& other) const {
    return what->type() == other.what->type() && *what == *other.what &&
        inner_extent == other.inner_extent &&
        selected_extent == other.selected_extent;
  }
  bool operator!=(const PartOf& other) const {
    return !(*this == other);
  }
};

// selected = Composition{g1, g2, g3} represents that the `selected` is a
// composition of g1, g2, and g3. This projection usually comes from split. For
// example, if we have
//    selected
//     /   \.
//   g1     g2
// then:
//   selected = Composition{g1, g2}.
template <typename... Args>
using Composition = std::deque<Args...>;

using Projection = dynamic_type::
    DynamicType<dynamic_type::Containers<Composition, PartOf>, ValGroup>;

// Utilities to print the entire abstract syntax tree of a projection.
std::string print(const Projection& proj);

std::string print(const ValGroup& group) {
  return group->toString();
}

std::string print(const PartOf<Projection>& part) {
  auto str_or_null = [](Val* val) {
    return val == nullptr ? "nullptr" : val->toInlineString();
  };
  return "PartOf(what=" + print(*part.what) +
      ", inner_extent=" + str_or_null(part.inner_extent) +
      ", selected_extent=" + str_or_null(part.selected_extent) + ")";
}

std::string print(const Composition<Projection>& vec) {
  std::stringstream ss;
  ss << "[";
  for (const auto& g : vec) {
    ss << print(g) << ", ";
  }
  ss << "]";
  return ss.str();
}

std::string print(const std::monostate&) {
  return "std::monostate";
}

std::string print(const Projection& proj) {
  return Projection::dispatch(
      [&](const auto& proj) { return print(proj); }, proj);
}

// Utilities to check if a ValGroup is contained in proj or its subtree.
bool related(const Projection& proj, const ValGroup& to);

bool related(const ValGroup& group, const ValGroup& to) {
  return group == to;
}

bool related(const PartOf<Projection>& part, const ValGroup& to) {
  return related(*part.what, to);
}

bool related(const Composition<Projection>& comp, const ValGroup& to) {
  return std::any_of(
      comp.begin(), comp.end(), [&](const auto& g) { return related(g, to); });
}

bool related(const std::monostate&, const ValGroup& to) {
  return false;
}

bool related(const Projection& proj, const ValGroup& to) {
  return Projection::dispatch(
      [&](const auto& proj) { return related(proj, to); }, proj);
}

// Utilities to get the extent of a projection.
Val* extent(const Projection& proj);

Val* extent(const ValGroup& group) {
  return group->front()->as<IterDomain>()->extent();
}

Val* extent(const PartOf<Projection>& part) {
  return part.selected_extent;
}

Val* extent(const Composition<Projection>& comp) {
  return std::accumulate(
      comp.begin(),
      comp.end(),
      FusionGuard::getCurFusion()->oneVal(),
      [](Val* acc, const auto& g) {
        return SimplifyingIrBuilder::mulExpr(acc, extent(g));
      });
}

Val* extent(const std::monostate&) {
  NVF_THROW("Cannot get extent of std::monostate");
}

Val* extent(const Projection& proj) {
  return Projection::dispatch(
      [&](const auto& proj) { return extent(proj); }, proj);
}

// Simplify the abstract syntax tree so that it is easier to be pattern
// matched. Defined below.
Projection simplify(Projection proj);

// Given an expression on the traversal path and its direction, get the from
// and to groups.
auto fromGroups(
    const ValGraph& id_graph,
    const ExprGroup& eg,
    Direction direction) {
  return direction == Direction::Backward ? id_graph.outputGroups(eg)
                                          : id_graph.inputGroups(eg);
}

auto toGroups(
    const ValGraph& id_graph,
    const ExprGroup& eg,
    Direction direction) {
  return direction == Direction::Backward ? id_graph.inputGroups(eg)
                                          : id_graph.outputGroups(eg);
}

// Do the propagation to project linear_g on domain through the given
// expression, build out and simplify the abstract syntax tree on the fly by
// substituting equivalent items. For example, if we have
//   2   [2]  3
//    \    \ /
//     \    6
//      \  /
//       12   2
//        \  /
//         24
//        /  \.
//       4    6
// and the linear_g is [2], when we propagate from [2] to 24, we will build out
// the abstract syntax tree with the following steps:
//
// First, we will traverse the expression 6 = merge(2, 3). We will build out
//   linear_g = PartOf{what=6, inner_extent=3, selected_extent=2}
//
// Second, we will traverse the expression 12 = merge(2, 6). From this
// expression, we know that
//   6 = PartOf{what=12, inner_extent=nullptr, selected_extent=6}
// Substituting definition of 6, in the above definition of linear_g, we get
//   linear_g = PartOf{
//     what=PartOf{what=12, inner_extent=nullptr, selected_extent=6},
//     inner_extent=3,
//     selected_extent=2
//   }
//
// Third, we will traverse the expression 24 = merge(12, 2). From this
// expression, we know that
//   12 = PartOf{what=24, inner_extent=2, selected_extent=12}
// Substituting definition of 12, in the above definition of linear_g, we get
//   linear_g = PartOf{
//     what=PartOf{
//        what=PartOf{what=24, inner_extent=2, selected_extent=12},
//        inner_extent=nullptr,
//        selected_extent=6
//     },
//     inner_extent=3,
//     selected_extent=2
//   }
//
// Finally, we will traverse the expression 4, 6 = split(24). From this
// expression, we know that
//   24 = Composition{4, 6}
// Substituting definition of 24, in the above definition of linear_g, we get
//   linear_g = PartOf{
//     what=PartOf{
//       what=PartOf{
//         what=Composition{4, 6},
//         inner_extent=2,
//         selected_extent=12
//       },
//       inner_extent=nullptr,
//       selected_extent=6
//     },
//     inner_extent=3,
//     selected_extent=2
//   }
//
// Note that the dynamic type Projection has limited expressiveness, we may
// encounter cases where the projection can not be represented in the language
// of the dynamic type Projection. For such cases, we will just use
// std::monostate to denote "unknown".
Projection propagate(
    const Projection& proj,
    const ValGraph& id_graph,
    const ExprGroup& eg,
    Direction direction);

Projection propagate(
    const ValGroup& group,
    const ValGraph& id_graph,
    const ExprGroup& eg,
    Direction direction) {
  auto from = fromGroups(id_graph, eg, direction);
  auto to = toGroups(id_graph, eg, direction);
  if (from.size() == 1 && to.size() == 2) {
    // If we have
    //    group
    //    /   \.
    //   g1   g2
    // and the split is divisible, then build the following abstract syntax
    // tree:
    //   group = Composition{g1, g2}
    // If the split is not divisible, then build the following abstract syntax
    // tree:
    //   group = PartOf{what=Composition{g1, g2},
    //                  inner_extent=nullptr,
    //                  selected_extent=extent(group)}
    NVF_ERROR(eg->front()->isA<Split>() || eg->front()->isA<Merge>());
    if (from.front() != group) {
      return group;
    }
    auto comp = Composition<Projection>{to.front(), to.back()};
    bool may_be_indivisible_split = eg->front()->isA<Split>() &&
        !simplifyExpr(eg->front()->as<Split>()->isDivisible())->isTrue();
    if (may_be_indivisible_split) {
      return PartOf<Projection>{
          std::make_shared<Projection>(comp),
          /*inner_extent=*/nullptr,
          /*selected_extent=*/extent(group)};
    }
    return comp;
  } else if (from.size() == 2 && to.size() == 1) {
    // If we have
    //   group    g1
    //        \  /
    //         g2
    // then build the following abstract syntax tree
    //   group = PartOf{what=g2,
    //                  inner_extent=extent(g1),
    //                  selected_extent=extent(group)}
    //
    // If we have
    //   g1   group
    //     \  /
    //      g2
    // then build the following abstract syntax tree
    //   group = PartOf{what=g2,
    //                  inner_extent=nullptr,
    //                  selected_extent=extent(group)}
    NVF_ERROR(eg->front()->isA<Split>() || eg->front()->isA<Merge>());
    if (from.front() != group && from.back() != group) {
      return group;
    }
    return PartOf<Projection>{
        std::make_shared<Projection>(to.front()),
        /*inner_extent=*/from.front() == group ? extent(from.back()) : nullptr,
        /*selected_extent=*/
        simplifyExpr(extent(group))};
  }
  if (std::none_of(from.begin(), from.end(), [&](const auto& g) {
        return g == group;
      })) {
    return group;
  }
  // Not representable (or don't know how to represent) by the language of the
  // dynamic type Projection.
  return {};
}

Projection propagate(
    const PartOf<Projection>& part,
    const ValGraph& id_graph,
    const ExprGroup& eg,
    Direction direction) {
  // Just recursively propagate subtree.
  auto propagated = propagate(*part.what, id_graph, eg, direction);
  if (!propagated.hasValue()) {
    return {};
  }
  auto result = PartOf<Projection>{
      std::make_shared<Projection>(propagated),
      part.inner_extent,
      part.selected_extent};
  return simplify(result);
}

template <typename Container1, typename Container2>
auto search(Container1& from, Container2& substr) {
  return std::search(
      from.begin(),
      from.end(),
      substr.begin(),
      substr.end(),
      [](const Projection& a, const Projection& b) {
        return a.type() == b.type() && a == b;
      });
}

Projection propagate(
    const Composition<Projection>& comp,
    const ValGraph& id_graph,
    const ExprGroup& eg,
    Direction direction) {
  auto from = fromGroups(id_graph, eg, direction);
  auto to = toGroups(id_graph, eg, direction);
  int64_t num_related_components =
      std::count_if(comp.begin(), comp.end(), [&](const auto& proj) {
        return std::any_of(from.begin(), from.end(), [&](const auto& g) {
          return related(proj, g);
        });
      });

  // Not related at all. No-op.
  if (num_related_components == 0) {
    return comp;
  }

  // If only one item in `comp` is related to from, then we can just treat `eg`
  // as a "unary op" and recursively propagate subtrees.
  if (num_related_components == 1) {
    Composition<Projection> result;
    for (const auto& proj : comp) {
      auto propagated = propagate(proj, id_graph, eg, direction);
      if (!propagated.hasValue()) {
        return {};
      }
      result.emplace_back(std::move(propagated));
    }
    return simplify(result);
  }

  // If more than one group in `from` is related to comp, this is more complex.
  // We need to check if these involved multiple groups in comp are transformed
  // by `eg` in a compatible way, in the sense that after the transformation,
  // the result is still representable by language of the dynamic type
  // Projection.
  if (from.size() == 2 && to.size() == 1) {
    NVF_ERROR(eg->front()->isA<Split>() || eg->front()->isA<Merge>());
    // If merging two contiguous components, replace them with the merged
    // ValGroup.
    Composition<Projection> result = comp;
    auto it = search(result, from);
    if (it != comp.end()) {
      result.erase(it + 1, it + (int64_t)from.size());
      *it = to.front();
      return simplify(result);
    }
  }

  // Not representable (or don't know how to represent) by the language of the
  // dynamic type Projection.
  return {};
}

Projection propagate(
    const std::monostate&,
    const ValGraph& id_graph,
    const ExprGroup& eg,
    Direction direction) {
  NVF_THROW("Should not reach here.");
}

Projection propagate(
    const Projection& proj,
    const ValGraph& id_graph,
    const ExprGroup& eg,
    Direction direction) {
  return Projection::dispatch(
      [&](const auto& proj) {
        return propagate(proj, id_graph, eg, direction);
      },
      proj);
}

// After propagation, we should have the information about how linear_g lives
// in domain. Parse this information to check if linear_g is linear in domain,
// and if it is, compute the stride.
Val* proveLinearAndGetStrideAfterPropagation(
    const Projection& proj,
    const ValGroups& domain);

Val* proveLinearAndGetStrideAfterPropagation(
    const ValGroup& group,
    const ValGroups& domain) {
  Val* stride = group->front()->fusion()->oneVal();
  for (auto it = domain.rbegin(); it != domain.rend(); ++it) {
    if (*it == group) {
      return stride;
    }
    stride = SimplifyingIrBuilder::mulExpr(stride, extent(*it));
  }
  return nullptr;
}

Val* proveLinearAndGetStrideAfterPropagation(
    const PartOf<Projection>& part,
    const ValGroups& domain) {
  auto inner_stride =
      proveLinearAndGetStrideAfterPropagation(*part.what, domain);
  if (inner_stride == nullptr) {
    return nullptr;
  }
  return SimplifyingIrBuilder::mulExpr(inner_stride, part.inner_extent);
}

Val* proveLinearAndGetStrideAfterPropagation(
    const Composition<Projection>& comp,
    const ValGroups& domain) {
  if (comp.empty()) {
    return FusionGuard::getCurFusion()->zeroVal();
  }
  auto it = search(domain, comp);
  if (it == domain.end()) {
    return nullptr;
  }
  return proveLinearAndGetStrideAfterPropagation(comp.back(), domain);
}

Val* proveLinearAndGetStrideAfterPropagation(
    const std::monostate&,
    const ValGroups& domain) {
  NVF_THROW("Should not reach here.");
  return nullptr;
}

Val* proveLinearAndGetStrideAfterPropagation(
    const Projection& proj,
    const ValGroups& domain) {
  return Projection::dispatch(
      [&](const auto& proj) {
        return proveLinearAndGetStrideAfterPropagation(proj, domain);
      },
      proj);
}

// Simplify the abstract syntax tree so that it is easier to be pattern
// matched.
Projection simplify(const ValGroup& group) {
  return group;
}

PartOf<Projection> cancelCommonFactors(const PartOf<Projection>& part) {
  // If `what` is a composition and inner_extent is a multiple of the extent of
  // the last items in `what`, we can simplify the inner_extent and `what` by
  // canceling the last items in `what`.
  //
  // Example:
  // PartOf{what=[5,3,2], inner_extent=42} => PartOf{what=[5], inner_extent=7}
  //
  // Proof of correctness:
  // Suppose we have an IterDomain I0, and I0 is split in two different ways:
  //           I0
  //          /  \.
  //         /    \.
  //     split    split
  //     /   \    /   \.
  //   I1  I2{m} I3   I4{n}
  // Assuming I0 is divisible by m, then we have:
  //   I3 = PartOf{what=I0, inner_extent=n} ............ (1)
  // Let g = gcd(m, n), and m = g*m', n = g*n'. Then according to Theorem 2.1 in
  // doc/reading/iterdomain.md, the above transformation is mathematically
  // equivalent to the following transformation:
  //               I0
  //               |
  //             split
  //             /   \.
  //           Io   Ig{g}
  //          /  \.
  //         /    \.
  //        /      \.
  //    split      split
  //    /   \      /   \.
  //   I1 I2'{m'} I3 I4'{n'}
  // where
  //   I2 = merge(I2', Ig), I4 = merge(I4', Ig)
  // Note that
  //   I0 = Composition{I1, I2', Ig}
  // substitute to the above equation (1), we get:
  //   I3 = PartOf{what=[I1, I2', Ig], inner_extent=n} ............ (2)
  // Because I0 is divisible by m, Io is also divisible by m'. So we have:
  //   I3 = PartOf{what=Io, inner_extent=n'}
  // and
  //   Io = [I1, I2']
  // That is,
  //   I3 = PartOf{what=[I1, I2'], inner_extent=n'} ............ (3)
  // Comparing equation (2) and (3), we have:
  //   PartOf{what=[I1, I2', Ig], inner_extent=n} =
  //     PartOf{what=[I1, I2'], inner_extent=n'}
  // That is, we can cancel the common factor of `what` and inner_extent.
  if (!part.what->is<Composition>()) {
    return part;
  }
  auto dq = part.what->as<Composition>();

  Val* new_inner_extent = part.inner_extent;
  if (new_inner_extent == nullptr) {
    return part;
  }

  while (!dq.empty() &&
         simplifyExpr(
             IrBuilder::isDivisibleExpr(new_inner_extent, extent(dq.back())))
             ->isTrue()) {
    new_inner_extent =
        simplifyExpr(IrBuilder::divExpr(new_inner_extent, extent(dq.back())));
    dq.pop_back();
  }
  if (new_inner_extent->isOne()) {
    new_inner_extent = nullptr;
  }
  if (dq.size() == 1) {
    return PartOf<Projection>{
        std::make_shared<Projection>(dq.front()),
        new_inner_extent,
        part.selected_extent};
  }
  return PartOf<Projection>{
      std::make_shared<Projection>(std::move(dq)),
      new_inner_extent,
      part.selected_extent};
}

PartOf<Projection> trimRedundant(const PartOf<Projection>& part) {
  // If part.what is a composition then we only keep the minimum number of
  // items in `what` that is sufficient to represent the selected_extent *
  // inner_extent.
  //
  // Example:
  // PartOf{what=[7, 3, 5, 2], inner_extent=3, selected_extent=5} =>
  //   PartOf{what=[3, 5, 2], inner_extent=3, selected_extent=5}
  //
  // Proof of correctness:
  // Suppose we have an IterDomain I0, and I0 is split in two different ways:
  //             I0
  //            /  \.
  //           /    \.
  //       split    split
  //       /   \    /   \.
  //     I1  I2{m} I3   I4{n}
  //               |
  //             split
  //              /  \.
  //             I5  I6{k}
  // Assuming I0 is divisible by m, then we have:
  //   I6 = PartOf{what=I0, inner_extent=n, selected_extent=k}
  // and
  //   I0 = Composition{I1, I2}
  // substitute to the above equation, we get:
  //   I6 = PartOf{what=[I1, I2], inner_extent=n, selected_extent=k}
  //                                                       ............ (1)
  // For the case where m is a multiple of n*k, according to Theorem 2.1 in
  // doc/reading/iterdomain.md, the above transformation is mathematically
  // equivalent to the following transformation:
  //               I0
  //               |
  //             split
  //             /   \.
  //            I1   I2{m}
  //                   |
  //                 split
  //                 /   \.
  //                I7   I4{n}
  //                 |
  //               split
  //               /   \.
  //              I8   I6{k}
  // where
  //   I5 = merge(I1, I8)
  // In the above transformation, we have:
  //   I6 = PartOf{what=I2, inner_extent=n, selected_extent=k}
  // Comparing equation (1) and the above equation, we have:
  //   PartOf{what=[I1, I2], inner_extent=n, selected_extent=k} =
  //     PartOf{what=I2, inner_extent=n, selected_extent=k}
  // Note that the condition of Theorem 2.1 requires that the extent of I2 is
  // a multiple of n*k, which is the same as the condition of this
  // simplification.
  if (!part.what->is<Composition>()) {
    return part;
  }
  auto dq = part.what->as<Composition>();

  Val* required_extent =
      SimplifyingIrBuilder::mulExpr(part.selected_extent, part.inner_extent);

  Val* what_extent = nullptr;
  int64_t count = 0;
  while (count < (int64_t)dq.size()) {
    count++;
    const auto& item = dq.at(dq.size() - count);
    what_extent = SimplifyingIrBuilder::mulExpr(what_extent, extent(item));
    if (simplifyExpr(IrBuilder::isDivisibleExpr(what_extent, required_extent))
            ->isTrue()) {
      break;
    }
  }
  while (count < (int64_t)dq.size()) {
    dq.pop_front();
  }
  if (dq.size() == 1) {
    return PartOf<Projection>{
        std::make_shared<Projection>(dq.front()),
        part.inner_extent,
        part.selected_extent};
  }
  return PartOf<Projection>{
      std::make_shared<Projection>(std::move(dq)),
      part.inner_extent,
      part.selected_extent};
}

PartOf<Projection> mergeParts(const PartOf<Projection>& part) {
  // Combine PartOf(what=PartOf(what=x), ...) into PartOf(what=x, ...).
  //
  // Example:
  // PartOf{
  //   what=PartOf{what=24, inner_extent=2, selected_extent=12},
  //   inner_extent=3,
  //   selected_extent=2}
  // =>
  // PartOf{what=24, inner_extent=6, selected_extent=2}
  if (!part.what->is<PartOf>()) {
    return part;
  }
  const auto& what = part.what->as<PartOf>();

  return PartOf<Projection>{
      what.what,
      SimplifyingIrBuilder::mulExpr(part.inner_extent, what.inner_extent),
      part.selected_extent};
}

Projection eliminateTrivialPartOf(const PartOf<Projection>& part) {
  // If part.what has the same extent as the selected extent, and there is no
  // inner extent in part, then the full `what` is identical to the selected
  // part.
  //
  // Example:
  // PartOf{what=5, inner_extent=nullptr, selected_extent=5} => 5
  if (part.inner_extent == nullptr &&
      simplifyExpr(IrBuilder::eqExpr(extent(*part.what), part.selected_extent))
          ->isTrue()) {
    return *part.what;
  }
  return part;
}

Projection simplify(const PartOf<Projection>& part) {
  // Recursively simplify subtree.
  auto simplified = PartOf<Projection>{
      std::make_shared<Projection>(simplify(*part.what)),
      part.inner_extent,
      part.selected_extent};
  // Run simplification rules.
  simplified = cancelCommonFactors(simplified);
  simplified = trimRedundant(simplified);
  simplified = mergeParts(simplified);
  return eliminateTrivialPartOf(simplified);
}

Composition<Projection> flattenCompositions(
    const Composition<Projection>& comp) {
  // Flatten the composition into a single level.
  //
  // Example:
  // Composition{Composition{5, 3}, 2} => Composition{5, 3, 2}
  Composition<Projection> result;
  for (const auto& proj : comp) {
    if (proj.is<Composition>()) {
      const auto& flat = proj.as<Composition>();
      result.insert(result.end(), flat.begin(), flat.end());
    } else {
      result.push_back(proj);
    }
  }
  return result;
}

Projection eliminateTrivialComposition(const Composition<Projection>& comp) {
  // If the composition has only one element, then the composition is the same
  // as the element.
  //
  // Example:
  // Composition{5} => 5
  if (comp.size() == 1) {
    return comp.front();
  }
  return comp;
}

Projection simplify(const Composition<Projection>& comp) {
  // Recursively simplify subtrees.
  Composition<Projection> simplified;
  for (const auto& proj : comp) {
    simplified.push_back(simplify(proj));
  }

  // Run simplification rules.
  simplified = flattenCompositions(simplified);
  return eliminateTrivialComposition(simplified);
}

Projection simplify(const std::monostate& null) {
  return null;
}

Projection simplify(Projection projection) {
  // Run simplifications until convergence.
  auto simplified = projection;
  do {
    projection = simplified;
    simplified = Projection::dispatch(
        [&](const auto& projection) { return simplify(projection); },
        projection);
  } while (simplified.type() != projection.type() || simplified != projection);
  return projection;
}

} // namespace

Val* proveLinearAndGetStride(
    const ValGraph& id_graph,
    const ValGroup& linear_g,
    const ValGroups& domain) {
  FusionGuard fg(linear_g->front()->fusion());
  // This function uses simplifyExpr extensively. If we have disable expression
  // simplification in order to help inspect generated kernels then we will get
  // incorrect results here. Instead, we ensure it is enabled using this guard.
  DisableOptionsGuard dog;
  DisableOptionsGuard::getCurOptions().unset(DisableOption::ExprSimplify);
  if (simplifyExpr(extent(linear_g))->isOne()) {
    // If the extent of the linear group is 1, we always consider it as linear,
    // regardless of its relationship with domain. For this case, we use stride
    // zero as a placeholder, as "stride" is really meaningless for a dimension
    // of size one.
    return linear_g->front()->fusion()->zeroVal();
  }
  // Propagate from linear_g to domain. Use frontier to keep track of the
  // how linear_g lives in the current propagation front. Note that linear_g may
  // not contain full dependency of domain.
  Projection frontier = linear_g;
  auto path =
      ValGraphPermissiveBFS::getExprGroupsBetween(
          id_graph, {linear_g}, domain, /*require_all_to_visited=*/false)
          .first;
  // Propagate along the path from linear_g to domain. Note that we do not
  // always propagate all the way through the path. Instead, early stopping
  // is necessary to be functionally correct. For example, if we have the
  // following ValGroups:
  //   4   2
  //    \ /
  //     8
  //    / \.
  //   4'  2'
  // and we are asking: is 2' linear in [4, 2']? The answer is trivially
  // yes by eyeballing, because 2' is the inner of [4, 2']. However, we must be
  // careful in propagation to algorithmically get it right. Although we can
  // directly tell the answer for this example without any progagation, because
  // ValGraphPermissiveBFS has no information about the underlying problem we
  // are solving, it always generate a path that visits `domain` as much as
  // possible, regardless of whether the underlying problem want it or not.
  // For this case, although the 4 in `domain` is unrelated to the answer,
  // ValGraphPermissiveBFS will still visit it. Therefore, it will generate a
  // path that include the merge of 4 and 2, and the split of 8. If we
  // mindlessly propagate along this path without early stopping, we will
  // propagate linear_g into frontier = 2, which leads to a conclusion that
  // "linear_g is the 2, and domain is [4, 2'], linear_g is not in domain, so I
  // can not prove linearity", which is not the answer we want. Note that
  // patterns like this can appear anywhere in the path, so we need to check for
  // early stopping at each step of the propagation.
  Val* stride = proveLinearAndGetStrideAfterPropagation(frontier, domain);
  if (stride != nullptr) {
    return stride;
  }
  for (const auto& [eg, direction] : path) {
    frontier = propagate(frontier, id_graph, eg, direction);
    if (!frontier.hasValue()) {
      // Not representable (or don't know how to represent) by the language of
      // the dynamic type Projection.
      return nullptr;
    }
    // Check for early stopping.
    Val* stride = proveLinearAndGetStrideAfterPropagation(frontier, domain);
    if (stride != nullptr) {
      return stride;
    }
  }
  return nullptr;
}

IterDomain* getConcreteLoopID(IterDomain* id) {
  // FusionInfo with ComputeAtMap is required
  NVF_ERROR(FusionInfoGuard::hasCurrent());
  NVF_ERROR(FusionInfoGuard::current()->hasComputeAtMap());

  // Currently, the concrete loop ID uses the IdModel loop
  // promotion only when opted in.
  if ((GpuLower::hasCurrent() &&
       GpuLower::current()->idModelOptions().loop()) ||
      (!GpuLower::hasCurrent() && FusionInfoGuard::current()->hasIdModel() &&
       FusionInfoGuard::current()->idModel().hasIdGraph(IdMappingMode::LOOP))) {
    // If enabled, the concret ID should be basically just the
    // promotion ID itself. However, just to reduce literacl changes
    // of generated kernels so that the CI diff check could report
    // smaller number of errors, we try to see if the concrete ID by
    // ComputeAtMap could be used as a substitute. If yes, that ID is
    // returned instead of the promotion ID.

    const auto& loop_graph =
        FusionInfoGuard::current()->idModel().idGraph(IdMappingMode::LOOP);
    const auto& exact_graph =
        FusionInfoGuard::current()->idModel().idGraph(IdMappingMode::EXACT);
    auto promotion =
        getLoopPromotion(id, FusionInfoGuard::current()->idModel());
    const auto& ca_map = FusionInfoGuard::current()->caMap();
    const auto& loop_group = loop_graph.toGroup(id);

    // Try to see if the CA concrete domain can be used instead
    for (auto loop_val : *loop_group) {
      IterDomain* loop_id = loop_val->as<IterDomain>();
      if (ca_map.idExistsInMap(loop_id, IdMappingMode::LOOP)) {
        auto ca_map_concrete =
            ca_map.getConcreteMappedID(loop_id, IdMappingMode::LOOP);
        if (loop_graph.disjointValSets().strictAreMapped(
                ca_map_concrete, promotion) &&
            exact_graph.disjointValSets().strictAreMapped(
                ca_map_concrete, promotion)) {
          return ca_map_concrete;
        }
      }
    }

    // The CAMap concrete ID is not a valid concrete ID. Use the
    // promotion ID instead.
    return promotion;
  } else {
    const auto& ca_map = FusionInfoGuard::current()->caMap();
    return ca_map.getConcreteMappedID(id, IdMappingMode::LOOP);
  }
}

bool allMmaInputsGuardedByMBarrier(const MmaOp* mma) {
  return ir_utils::isCpAsyncBulkLoad(
             ir_utils::getTv(mma->inA())->definition()) &&
      ir_utils::isCpAsyncBulkLoad(ir_utils::getTv(mma->inB())->definition());
}

bool isWarpSpecializedLoop(ForLoop* loop) {
  return std::holds_alternative<WarpSpecialized>(
      GpuLower::current()
          ->circularBufferInfo()
          .getCircularBufferOptionsFor(loop->iter_domain())
          .type);
}

bool isCopyOnly(Expr* expr) {
  return expr->isOneOf<
      LoadStoreOp,
      BroadcastOp,
      SqueezeOp,
      SliceOp,
      PadOp,
      ReshapeOp>();
}

bool isCopyOnly(Val* val) {
  if (val->definition() != nullptr && !isCopyOnly(val->definition())) {
    return false;
  }
  for (auto use : val->uses()) {
    if (!isCopyOnly(use)) {
      return false;
    }
  }
  return true;
}

} // namespace lower_utils

} // namespace nvfuser
