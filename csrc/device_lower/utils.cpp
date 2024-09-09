// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#include <device_lower/utils.h>

#include <ATen/cuda/CUDAContext.h>
#include <c10/util/irange.h>
#include <device_lower/analysis/thread_predicate.h>
#include <device_lower/lower2device.h>
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

TVDomainGuard::TVDomainGuard(TensorView* tv, TensorDomain* td)
    : tv_(tv), prev_domain_(tv_->domain()) {
  tv_->setDomain(td);
}

TVDomainGuard::TVDomainGuard(TVDomainGuard&& guard)
    : tv_(nullptr), prev_domain_(guard.prev_domain_) {
  std::swap(tv_, guard.tv_);
}

TVDomainGuard::~TVDomainGuard() {
  if (tv_ != nullptr) {
    tv_->setDomain(prev_domain_);
  }
}

ir_utils::TVDomainGuard overrideContiguityGuard(
    TensorView* tv,
    bool contiguity) {
  // Use domain guard to ignore the contiguity of the given tv.
  TensorDomain* domain_with_specified_contiguity =
      IrBuilder::create<TensorDomain>(
          tv->getRootDomain(),
          tv->getLogicalDomain(),
          tv->getAllocationDomain(),
          tv->getLoopDomain(),
          TensorDomain::getContiguityFilledWith(
              tv->getMaybeAllocationDomain(), contiguity));

  return ir_utils::TVDomainGuard(tv, domain_with_specified_contiguity);
}

ir_utils::TVDomainGuard allocateToLogicalDomainGuard(
    TensorView* tv,
    bool contiguity) {
  // Use domain guard to ignore the contiguity of the given tv.
  TensorDomain* domain_with_specified_contiguity =
      IrBuilder::create<TensorDomain>(
          tv->getRootDomain(),
          tv->getLogicalDomain(),
          tv->getLoopDomain(),
          TensorDomain::getContiguityFilledWith(
              tv->getLogicalDomain(), contiguity));

  return ir_utils::TVDomainGuard(tv, domain_with_specified_contiguity);
}

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
  if (std::any_of(
          expr->outputs().begin(),
          expr->outputs().end(),
          [](Val* v) { return isTV(v); }) &&
      (expr->isOneOf<
          UnaryOp,
          BinaryOp,
          TernaryOp,
          TensorConstruct,
          SelectOp,
          IndexSelectOp,
          TorchGatherOp,
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
          BroadcastOp,
          SqueezeOp,
          ExpandOp,
          ViewAsScalar,
          ViewOp,
          PadOp,
          SliceOp,
          CatOp,
          kir::GridReduction,
          kir::GroupedGridReduction,
          kir::GridBroadcast,
          kir::GridWelford,
          kir::GroupedGridWelford,
          kir::VectorizedWelfordOp>())) {
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

enum class CpAsyncBulkTileType { G2S, S2G, NotACpAsyncBulkTile };

inline CpAsyncBulkTileType getCpAsyncBulkTileType(const Expr* expr) {
  if (auto ldst = dynamic_cast<const LoadStoreOp*>(expr)) {
    if (ldst->opType() == LoadStoreOpType::CpAsyncBulkTensorTile) {
      if (getTv(ldst->in())->getMemoryType() == MemoryType::Global &&
          getTv(ldst->out())->getMemoryType() == MemoryType::Shared) {
        return CpAsyncBulkTileType::G2S;
      } else if (
          getTv(ldst->in())->getMemoryType() == MemoryType::Shared &&
          getTv(ldst->out())->getMemoryType() == MemoryType::Global) {
        return CpAsyncBulkTileType::S2G;
      } else {
        NVF_ERROR(false, "Invalid CpAsyncBulkTileType");
      }
    }
  }
  return CpAsyncBulkTileType::NotACpAsyncBulkTile;
}

} // namespace

bool isCpAsyncBulk(const Expr* expr) {
  return getCpAsyncBulkTileType(expr) !=
      CpAsyncBulkTileType::NotACpAsyncBulkTile;
}

bool isCpAsyncBulkLoad(const Expr* expr) {
  return getCpAsyncBulkTileType(expr) == CpAsyncBulkTileType::G2S;
}

bool isCpAsyncBulkStore(const Expr* expr) {
  return getCpAsyncBulkTileType(expr) == CpAsyncBulkTileType::S2G;
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

TensorView* getTvOutput(const Expr* expr) {
  for (auto out : expr->outputs()) {
    if (auto tv = getTv(out)) {
      return tv;
    }
  }
  return nullptr;
}

TensorView* getTvInput(const Expr* expr) {
  for (auto inp : expr->inputs()) {
    if (auto tv = getTv(inp)) {
      return tv;
    }
  }
  return nullptr;
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

std::optional<IterDomain*> getMaybeWarpReductionDim(
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
  for (auto id : tv_out->getLoopDomain()) {
    // Currently warp reduction only allows
    //  serial and block.x parallel reductions
    if (id->isReduction() && id->isParallelized()) {
      if (id->getParallelType() == ParallelType::TIDx) {
        reduction_on_xdim = id;
      } else if (id->isThread()) {
        return std::nullopt;
      }
    }
  }
  if (!reduction_on_xdim) {
    return std::nullopt;
  }

  if (!reduction_on_xdim->start()->isZeroInt()) {
    return std::nullopt;
  }

  if (reduction_on_xdim->hasPaddingToMultipleOfWarp()) {
    return std::optional<IterDomain*>(reduction_on_xdim);
  }

  if (reduction_on_xdim->extent()->isConstInt()) {
    auto extent_value = reduction_on_xdim->extent()->evaluate();
    if (extent_value % at::cuda::warp_size() == 0) {
      return std::optional<IterDomain*>(reduction_on_xdim);
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
    NVF_ERROR(false, "Provided val is not TensorIndex or TensorView.");
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

  if (!(ir_utils::isReductionOp(expr) || expr->isA<BroadcastOp>() ||
        expr->isA<kir::GridBroadcast>())) {
    return false;
  }

  // GroupedReductionOp can have multiple output TVs, but they must be
  // parallelized in the same way, so just checking one of them is enough.
  auto tv = ir_utils::getTvOutput(expr);

  if (tv->hasBlockReduction() || tv->hasGridReduction()) {
    return true;
  } else if (expr->isA<BroadcastOp>()) {
    const ParallelTypeBitmap pt_map =
        GpuLower::current()->threadPredMap().getParallelBroadcastDomains(tv);
    return pt_map.any();
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

BasicAllocInfo getAllocInformation(
    const TensorView* tv,
    const std::vector<ForLoop*>& for_loops,
    const std::unordered_map<IterDomain*, IterDomain*>& id_map,
    bool use_id_map) {
  DEBUG_PRINT_SCOPE(tv);
  BasicAllocInfo info;
  auto gpu_lower = GpuLower::current();

  bool outer_alloc_found = false;

  for (auto fl : for_loops) {
    if (info.alloc_pos == tv->getComputeAtPosition()) {
      DEBUG_LOG("Break at info.alloc_pos = ", info.alloc_pos);
      break;
    }

    if (tv->axis(info.alloc_pos)->isReduction()) {
      const auto outputs = FusionGuard::getCurFusion()->getTerminatingOutputs();
      NVF_ERROR(
          std::find(outputs.begin(), outputs.end(), tv) != outputs.end(),
          "Invalid computeAt of T",
          tv->name(),
          ". A reducation axis is detected outside computeAt point even though it is not an output tensor.");
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

    if (GpuLower::current()->caMap()->areMapped(
            local_id, fl_id, IdMappingMode::PERMISSIVE)) {
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

bool isExtentEqualToMaxParallelTypeExtent(const IterDomain* id) {
  const auto& parallel_dim_map = GpuLower::current()->parallelDimensionMap();
  auto* pdm_max_extent = parallel_dim_map.getRaw(id->getParallelType());
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
    auto pt_dim = GpuLower::current()->parallelDimensionMap().get(pt);
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
  NVF_ERROR(isHopper(expr->macro()));

  // NOLINTNEXTLINE(cppcoreguidelines-pro-type-member-init)
  std::array<UnitDim, 2> layout;

  auto out_tv = ir_utils::getTv(expr->out());
  IterDomain* reduction_id = nullptr;
  for (auto id : out_tv->getLogicalDomain()) {
    if (id->isReduction()) {
      reduction_id = id;
      break;
    }
  }
  NVF_ERROR(reduction_id != nullptr);

  std::array<TensorView*, 2> inputs = {
      ir_utils::getTv(expr->inA()), ir_utils::getTv(expr->inB())};
  for (auto i : c10::irange(2)) {
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

bool hasRootToLoopLinearTransformations(const TensorView* tv) {
  auto root = tv->getMaybeRootDomain();
  auto loop = tv->getLoopDomain();
  std::vector<Val*> loop_val(loop.begin(), loop.end());
  auto all_ids_vec =
      DependencyCheck::getAllValsBetween({root.begin(), root.end()}, loop_val);
  std::unordered_set<Val*> all_ids_set(all_ids_vec.begin(), all_ids_vec.end());
  auto alloc = tv->getMaybeAllocationDomain();
  auto logical = tv->getLogicalDomain();
  bool all_alloc_id_on_path = std::all_of(
      alloc.begin(), alloc.end(), [&](Val* v) { return all_ids_set.count(v); });
  bool all_logical_id_on_path =
      std::all_of(logical.begin(), logical.end(), [&](Val* v) {
        return all_ids_set.count(v);
      });
  return all_alloc_id_on_path && all_logical_id_on_path;
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
  const auto& other_id_exact_set = GpuLower::current()
                                       ->caMap()
                                       ->getIdSets(IdMappingMode::EXACT)
                                       .getDisjointSetOf(other_out_id);

  if (std::any_of(
          other_id_exact_set.begin(), other_id_exact_set.end(), [](auto id) {
            return id->getParallelType() == ParallelType::Vectorize;
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
// The idea is to propagate from linear_g to domain, keep track of how linear_g
// lives in domain. For example, let's consider the following two schedules
// (same example as the NVFuserTest.ProveLinearAndGetStride test):
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
// that [2] is linear in 128, and there is a 64 on the inner of [2] in 128. The
// next step is the split of I1 by 128 in v1. Similarly, we know that [2] is
// linear in I1, and there is a 64 on the inner of [2] in I1. Because the I1 in
// v1 and v3 are mapped, then we will continue propagation in v3. The next step
// is to process the split of I1 into I1o and 256 in v3. We already know that
// [2] is linear in I1, and on the inner of [2], there is a 64. Because 2 * 64 =
// 128, which is a factor of 256, we know that the 256 is able to fully cover
// the [2] in I1. Therefore, I1o is unrelated to [2], and we only need to focus
// on the 256. And in this 256, there is a 64 on the inner of [2]. The next step
// is to process the split of 256 into 4 and 64. Because the 64 happens to be
// the size of the inner of [2], we know that the [2] will be fully covered by
// the inner 2 of the 4 of the output of the split. Now we have finished
// propagation and reached domain. We already proved that [2] is linear in
// domain. In domain, the domains on the right of the 4 are 8, 8, 8, so the
// stride is 8*8*8 = 512.
namespace {

// From the above example, we can see that how linear_g lives in domain could be
// complicated. It can be, for example:
//   1. linear_g is equivalent to a single ValGroup in domain.
//   2. linear_g is the inner of a ValGroup in domain.
//   3. linear_g is the outer of a ValGroup in domain.
//   4. linear_g is the middle of a ValGroup in domain, where on the right,
//      there is a 2.
//   5. linear_g is split as g1, g2, g3 in domain.
//   6. linear_g is split as the inner 2 of g1, g2, and the outer 4 of g3.
//
// We use a dynamic type to be able to represent all these cases.
template <typename ValGroupRegion>
struct PartOf {
  std::shared_ptr<ValGroupRegion> group;
  Val* inner_extent = nullptr;
  Val* selected_extent = nullptr;
};

using ValGroupRegion = dynamic_type::DynamicType<
    dynamic_type::Containers<
        std::deque, // a composition of ValGroupRegions
        PartOf // part of a ValGroupRegion
        >,
    ValGroup // a whole ValGroup
    >;

// Utilities to print ValGroupRegion.
std::string print(const ValGroupRegion& group);

std::string print(const ValGroup& group) {
  return group->toString();
}

std::string print(const PartOf<ValGroupRegion>& part) {
  auto str_or_null = [](Val* val) {
    return val == nullptr ? "nullptr" : val->toInlineString();
  };
  return "PartOf(group=" + toString(*part.group) +
      ", inner_extent=" + str_or_null(part.inner_extent) +
      ", selected_extent=" + str_or_null(part.selected_extent) + ")";
}

std::string print(const std::deque<ValGroupRegion>& vec) {
  std::stringstream ss;
  ss << "[";
  for (const auto& g : vec) {
    ss << toString(g) << ", ";
  }
  ss << "]";
  return ss.str();
}

std::string print(const std::monostate&) {
  return "std::monostate";
}

std::string print(const ValGroupRegion& group) {
  return ValGroupRegion::dispatch(
      [&](const auto& group) { return toString(group); }, group);
}

// Utilities to check if ValGroupRegion is related to ValGroup.
bool related(const ValGroupRegion& current, const ValGroup& to);

bool related(const ValGroup& current, const ValGroup& to) {
  return current == to;
}

bool related(const PartOf<ValGroupRegion>& current, const ValGroup& to) {
  return related(*current.group, to);
}

bool related(const std::deque<ValGroupRegion>& current, const ValGroup& to) {
  return std::any_of(current.begin(), current.end(), [&](const auto& g) {
    return related(g, to);
  });
}

bool related(const std::monostate& current, const ValGroup& to) {
  return false;
}

bool related(const ValGroupRegion& current, const ValGroup& to) {
  return ValGroupRegion::dispatch(
      [&](const auto& current) { return related(current, to); }, current);
}

// Given an expression on the traversal path and its direction, get the from and
// to groups. Note that the traversal path is obtained by running BFS from
// domain to linear_g, so the direction is flipped with respect to how we
// propagate from linear_g to domain.
auto fromGroups(
    const ValGraph& id_graph,
    const ExprGroup& eg,
    Direction direction) {
  return direction == Direction::Forward ? id_graph.outputGroups(eg)
                                         : id_graph.inputGroups(eg);
}

auto toGroups(
    const ValGraph& id_graph,
    const ExprGroup& eg,
    Direction direction) {
  return direction == Direction::Forward ? id_graph.inputGroups(eg)
                                         : id_graph.outputGroups(eg);
}

// Do the propagation from linear_g to domain.
ValGroupRegion propagate(
    const ValGroupRegion& current,
    const ValGraph& id_graph,
    const ExprGroup& eg,
    Direction direction);

ValGroupRegion propagate(
    const ValGroup& current,
    const ValGraph& id_graph,
    const ExprGroup& eg,
    Direction direction) {
  auto from = fromGroups(id_graph, eg, direction);
  auto to = toGroups(id_graph, eg, direction);
  if (from.size() == 1) {
    // If linear_g is equivalent to from, and from is split into two groups,
    // then these two groups together represents linear_g.
    NVF_ERROR(to.size() == 2);
    NVF_ERROR(from.front() == current);
    return std::deque<ValGroupRegion>{to.front(), to.back()};
  } else {
    // If linear_g is merged with another group, then part of the merged group
    // represents linear_g.
    NVF_ERROR(from.size() == 2);
    NVF_ERROR(to.size() == 1);
    NVF_ERROR(from.front() == current || from.back() == current);
    return PartOf<ValGroupRegion>{
        std::make_shared<ValGroupRegion>(to.front()),
        /*inner_extent=*/from.front() == current
            ? from.back()->front()->as<IterDomain>()->extent()
            : nullptr,
        /*selected_extent=*/
        simplifyExpr(current->front()->as<IterDomain>()->extent())};
  }
}

ValGroupRegion propagate(
    const PartOf<ValGroupRegion>& current,
    const ValGraph& id_graph,
    const ExprGroup& eg,
    Direction direction) {
  auto from = fromGroups(id_graph, eg, direction);
  auto to = toGroups(id_graph, eg, direction);
  if (from.size() == 1) {
    NVF_ERROR(to.size() == 2);
    NVF_ERROR(related(*current.group, from.front()));
    auto group = propagate(*current.group, id_graph, eg, direction);
    if (!group.hasValue()) {
      return {};
    }
    // The result is actually just the
    //   PartOf{group, current.inner_extent, current.selected_extent}
    // However, we want to simplify the result. For example, if the inner_extent
    // is a multiple of the extent of the last item in group, we can simplify
    // the inner_extent and group by canceling the last item in group.
    Val* new_inner_extent = current.inner_extent;
    Val* group_extent = nullptr;
    // Now we will simplify inner_extent and `group`, for example, by removing
    // common factors. The simplified group will be updated in the variable
    // `group`, the extent of the group after simplification is stored in
    // `group_extent`, and the new inner_extent after simplification is stored
    // in `new_inner_extent`.
    if (group.is<std::deque>()) {
      auto dq = group.as<std::deque>();
      // If the new_inner_extent is divisible by the extent of the last item
      // in dq, we can simplify new_inner_extent and dq by canceling dq's last
      // items. For example, if new_inner_extent is 6 and dq is [5, 3], we can
      // simplify them to 2 and [5].
      if (new_inner_extent != nullptr) {
        for (auto* back = &dq.back(); back->is<ValGroup>() &&
             simplifyExpr(
                 IrBuilder::isDivisibleExpr(
                     new_inner_extent,
                     back->as<ValGroup>()->front()->as<IterDomain>()->extent()))
                 ->isTrue();
             dq.pop_back(), back = &dq.back()) {
          new_inner_extent = simplifyExpr(IrBuilder::divExpr(
              current.inner_extent,
              back->as<ValGroup>()->front()->as<IterDomain>()->extent()));
        }
        if (new_inner_extent->isOne()) {
          new_inner_extent = nullptr;
        }
      }
      // We only keep the minimum number of items that is sufficient to
      // represent the selected_extent * new_inner_extent. For example, if
      // selected_extent is 5 and new_inner_extent is 3, and dq is [7, 3, 5, 2],
      // we can simplify dq to [3, 5, 2].
      Val* required_extent = SimplifyingIrBuilder::mulExpr(
          current.selected_extent, new_inner_extent);
      int64_t count = 0;
      bool has_unknown = false;
      while (count < dq.size()) {
        count++;
        const auto& item = dq.at(dq.size() - count);
        if (item.is<ValGroup>()) {
          group_extent = SimplifyingIrBuilder::mulExpr(
              group_extent,
              item.as<ValGroup>()->front()->as<IterDomain>()->extent());
          if (simplifyExpr(
                  IrBuilder::isDivisibleExpr(group_extent, required_extent))
                  ->isTrue()) {
            break;
          }
        } else {
          // We can not accurately compute the extent of simplified group. The
          // group_extent just stores a factor of the actual extent.
          has_unknown = true;
        }
      }
      if (has_unknown) {
        group_extent = nullptr;
      }
      while (count < dq.size()) {
        dq.pop_front();
      }
      // If dq has only one item, we can simplify it to the item itself.
      if (dq.size() == 1) {
        group = dq.front();
      } else {
        group = std::move(dq);
      }
    }
    // If group has the same extent as linear_g, and there is no inner extent in
    // group, the the full group represents linear_g.
    if (new_inner_extent == nullptr && group_extent != nullptr &&
        simplifyExpr(IrBuilder::eqExpr(group_extent, current.selected_extent))
            ->isTrue()) {
      return group;
    }
    return PartOf<ValGroupRegion>{
        std::make_shared<ValGroupRegion>(group),
        new_inner_extent,
        current.selected_extent};
  } else {
    NVF_ERROR(from.size() == 2);
    NVF_ERROR(to.size() == 1);
    NVF_ERROR(
        related(*current.group, from.front()) ||
        related(*current.group, from.back()));
    // Adding more extent to the inner.
    if (current.group->is<ValGroup>() &&
        current.group->as<ValGroup>() == from.front()) {
      return PartOf<ValGroupRegion>{
          std::make_shared<ValGroupRegion>(to.front()),
          SimplifyingIrBuilder::mulExpr(
              current.inner_extent,
              from.back()->front()->as<IterDomain>()->extent()),
          current.selected_extent};
    }
    // Adding more extent to the outer.
    if (current.group->is<ValGroup>() &&
        current.group->as<ValGroup>() == from.back()) {
      return PartOf<ValGroupRegion>{
          std::make_shared<ValGroupRegion>(to.front()),
          current.inner_extent,
          current.selected_extent};
    }
    // Other cases are not implemented yet. Just return std::monostate,
    // which will make proveLinearAndGetStride stop the propagation and return
    // "can not prove linear". In the future, we can implement these cases if it
    // turns out to be useful.
    return {};
  }
}

ValGroupRegion propagate(
    const std::deque<ValGroupRegion>& current,
    const ValGraph& id_graph,
    const ExprGroup& eg,
    Direction direction) {
  auto from = fromGroups(id_graph, eg, direction);
  auto to = toGroups(id_graph, eg, direction);
  if (from.size() == 1) {
    NVF_ERROR(to.size() == 2);
    NVF_ERROR(std::any_of(current.begin(), current.end(), [&](const auto& g) {
      return related(g, from.front());
    }));
    // Split one group in current into two groups.
    bool may_be_indivisible_split = eg->front()->isA<Split>() &&
        !simplifyExpr(eg->front()->as<Split>()->isDivisible())->isTrue();
    std::deque<ValGroupRegion> result;
    bool first = true;
    for (const auto& g : current) {
      if (g.is<ValGroup>() && g.as<ValGroup>() == from.front()) {
        if (may_be_indivisible_split && !first) {
          // indivisible split will make dims not contiguous, so not linear.
          return {};
        }
        result.emplace_back(to.front());
        result.emplace_back(to.back());
      } else if (related(g, from.front())) {
        if (may_be_indivisible_split && !first) {
          // indivisible split will make dims not contiguous, so not linear.
          return {};
        }
        result.emplace_back(propagate(g, id_graph, eg, direction));
      } else {
        result.emplace_back(g);
      }
      first = false;
    }
    return result;
  } else {
    NVF_ERROR(from.size() == 2);
    NVF_ERROR(to.size() == 1);
    // If merging two contiguous groups, we need to update the result by
    // replacing them into the merged group. If merging two non-contiguous
    // groups, just fail the proof because it is not important for us yet.
    auto outer_it =
        std::find_if(current.begin(), current.end(), [&](const auto& g) {
          return g.template is<ValGroup>() &&
              g.template as<ValGroup>() == from.front();
        });
    if (outer_it != current.end()) {
      auto inner_it = std::next(outer_it);
      if (inner_it != current.end() && inner_it->is<ValGroup>() &&
          inner_it->as<ValGroup>() == from.back()) {
        std::deque<ValGroupRegion> result = current;
        result.erase(result.begin() + std::distance(current.begin(), inner_it));
        result.at(std::distance(current.begin(), outer_it)) = to.front();
        if (result.size() == 1) {
          return result.front();
        }
        return result;
      }
    }
    // Other cases are not implemented yet. Just return std::monostate,
    // which will make proveLinearAndGetStride stop the propagation and return
    // "can not prove linear". In the future, we can implement these cases if it
    // turns out to be useful.
    return {};
  }
}

ValGroupRegion propagate(
    const std::monostate& current,
    const ValGraph& id_graph,
    const ExprGroup& eg,
    Direction direction) {
  NVF_ERROR(false, "Should not reach here.");
}

ValGroupRegion propagate(
    const ValGroupRegion& current,
    const ValGraph& id_graph,
    const ExprGroup& eg,
    Direction direction) {
  return ValGroupRegion::dispatch(
      [&](const auto& current) {
        return propagate(current, id_graph, eg, direction);
      },
      current);
}

// After propagation, we should have the information about how linear_g lives in
// domain. Parse this information to check if linear_g is linear in domain, and
// if it is, compute the stride.
Val* proveLinearAndGetStrideAfterPropagation(
    const ValGroupRegion& g_in_domain,
    const ValGroups& domain);

Val* proveLinearAndGetStrideAfterPropagation(
    const ValGroup& g_in_domain,
    const ValGroups& domain) {
  Val* stride = g_in_domain->front()->fusion()->oneVal();
  for (auto it = domain.rbegin(); it != domain.rend(); ++it) {
    if (*it == g_in_domain) {
      return stride;
    }
    stride = SimplifyingIrBuilder::mulExpr(
        stride, (*it)->front()->as<IterDomain>()->extent());
  }
  return nullptr;
}

Val* proveLinearAndGetStrideAfterPropagation(
    const PartOf<ValGroupRegion>& g_in_domain,
    const ValGroups& domain) {
  auto inner_stride =
      proveLinearAndGetStrideAfterPropagation(*g_in_domain.group, domain);
  if (inner_stride == nullptr) {
    return nullptr;
  }
  return SimplifyingIrBuilder::mulExpr(inner_stride, g_in_domain.inner_extent);
}

Val* proveLinearAndGetStrideAfterPropagation(
    const std::deque<ValGroupRegion>& g_in_domain,
    const ValGroups& domain) {
  // The idea is like: given a string domain, find the substring g_in_domain.
  if (!std::all_of(g_in_domain.begin(), g_in_domain.end(), [&](const auto& g) {
        return g.template is<ValGroup>();
      })) {
    // Not implemented yet, just fail the proof.
    return nullptr;
  }
  auto first_it = std::find(
      domain.begin(), domain.end(), g_in_domain.front().as<ValGroup>());
  if (first_it == domain.end()) {
    return nullptr;
  }
  for (auto it = g_in_domain.begin(); it != g_in_domain.end();
       ++it, ++first_it) {
    if (it->as<ValGroup>() != *first_it) {
      return nullptr;
    }
  }
  return proveLinearAndGetStrideAfterPropagation(
      g_in_domain.back().as<ValGroup>(), domain);
}

Val* proveLinearAndGetStrideAfterPropagation(
    const std::monostate& g_in_domain,
    const ValGroups& domain) {
  NVF_ERROR(false, "Should not reach here.");
  return nullptr;
}

Val* proveLinearAndGetStrideAfterPropagation(
    const ValGroupRegion& g_in_domain,
    const ValGroups& domain) {
  return ValGroupRegion::dispatch(
      [&](const auto& g_in_domain) {
        return proveLinearAndGetStrideAfterPropagation(g_in_domain, domain);
      },
      g_in_domain);
}

} // namespace

Val* proveLinearAndGetStride(
    const ValGraph& id_graph,
    const ValGroup& linear_g,
    const ValGroups& domain) {
  if (simplifyExpr(linear_g->front()->as<IterDomain>()->extent())->isOne()) {
    // If the extent of the linear group is 1, we always consider it as linear,
    // regardless of its relationship with domain. For this case, we use stride
    // zero as a placeholder, as "stride" is really meaningless for a dimension
    // of size one.
    return linear_g->front()->fusion()->zeroVal();
  }
  // Propagate from linear_g to domain. Use frontier to keep track of the
  // how linear_g lives in the current propagation front.
  ValGroupRegion frontier = linear_g;
  auto path = ValGraphBFS::getExprsBetween(id_graph, domain, {linear_g});
  while (!path.empty()) {
    const auto& [eg, direction] = path.back();
    path.pop_back();
    auto from = fromGroups(id_graph, eg, direction);
    if (!std::any_of(from.begin(), from.end(), [&](const auto& g) {
          return related(frontier, g);
        })) {
      // Not all expressions in the path are related to linear_g. For example,
      // in the above example, the split of 64 is not interesting for us because
      // the 64 does not carry anything on [2], neither in part nor in full.
      continue;
    }
    if (!eg->front()->isOneOf<Split, Merge>()) {
      // Other expressions are not supported yet. Just fail the proof.
      return nullptr;
    }
    frontier = propagate(frontier, id_graph, eg, direction);
    if (!frontier.hasValue()) {
      // std::monostate is a special value that indicates that the propagation
      // has failed.
      return nullptr;
    }
  }
  // After propagation, we should have the information about how linear_g lives
  // in domain. Parse this information to check if linear_g is linear in domain.
  return proveLinearAndGetStrideAfterPropagation(frontier, domain);
}

} // namespace lower_utils

} // namespace nvfuser
