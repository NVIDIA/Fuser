// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#include <device_lower/lower2device.h>
#include <ir/interface_nodes.h>
#include <ir/utils.h>
#include <iter_visitor.h>
#include <kernel_ir_dispatch.h>

#include <device_lower/analysis/fused_reduction.h>

#include <algorithm>

namespace nvfuser {

namespace {

//! An instance of reduction patterns to fuse
class FusedReductionBroadcastInfo : public PolymorphicBase {
 public:
  FusedReductionBroadcastInfo(ReductionOp* reduction, bool with_broadcast)
      : reductions_({reduction}), with_broadcast_({with_broadcast}) {}

  FusedReductionBroadcastInfo(WelfordOp* welford, bool with_broadcast)
      : reductions_({welford}), with_broadcast_({with_broadcast}) {}

  FusedReductionBroadcastInfo(
      GroupedReductionOp* grouped_rop,
      bool with_broadcast)
      : reductions_({grouped_rop}), with_broadcast_({with_broadcast}) {}

  const std::vector<Expr*>& reductions() const {
    return reductions_;
  }

  const std::vector<bool>& withBroadcast() const {
    return with_broadcast_;
  }

 private:
  // Holds ReductionOp, WelfordOp or GroupedReductionOp.
  std::vector<Expr*> reductions_;
  // True each reduction also broadcasts
  std::vector<bool> with_broadcast_;
};

//! Inspect a fusion to detect eligible sequences of expressions to
//! use the fused reduction kernel
class FusionInspector : private IterVisitor {
 public:
  static std::vector<FusedReductionBroadcastInfo> run(Fusion* fusion) {
    FusionInspector inspector(fusion);
    return inspector.fusion_list_;
  }

 private:
  FusionInspector(Fusion* fusion)
      : has_warp_specialization_(checkWarpSpecialization(fusion)) {
    traverse(fusion);
    if (cluster_reduction_count_ > 0) {
      GpuLower::current()->setClusterReductionCount(cluster_reduction_count_);
    }
  }

  static bool checkWarpSpecialization(Fusion* fusion) {
    auto all_tvs = fusion->allTvs();
    return std::any_of(all_tvs.begin(), all_tvs.end(), [](TensorView* tv) {
      return tv->isCircularBuffered() &&
          std::holds_alternative<WarpSpecialized>(
                 tv->circularBufferOptions().type);
    });
  }

  using IterVisitor::handle;

  void handle(ReductionOp* rop) final {
    // If it's a grid reduction or has grouped Id, keep track of tensors that
    // depend on this reduction. Only consider when out is on register as that
    // is assumed in the fused reduction kernel.
    auto out = ir_utils::getTvOutput(rop);
    // Check if this reduction can use staticWarpAllReduceTIDX optimization.
    // Ensure there is only one reduction domain and it is parallelized with
    // TIDx and its size is a multiple of warp size (32).
    auto is_static_warp_reduction = [](TensorView* out,
                                       bool has_warp_specialization) {
      if (!has_warp_specialization) {
        return false;
      }

      constexpr int64_t kThreadsPerWarp = 32L;
      int reduction_count = 0;
      bool has_valid_tidx_reduction = false;
      for (auto ld : out->getLoopDomain()) {
        if (ld->isReduction()) {
          reduction_count++;
          if (ld->getParallelType() == ParallelType::TIDx &&
              ld->extent()->isConst() &&
              ld->extent()->value().as<int64_t>() % kThreadsPerWarp == 0) {
            has_valid_tidx_reduction = true;
          }
        }
      }

      return reduction_count == 1 && has_valid_tidx_reduction;
    };
    bool is_cluster_reduction = out->domain()->hasClusterReduction();
    if (out->getMemoryType() == MemoryType::Local &&
        (is_static_warp_reduction(out, has_warp_specialization_) ||
         out->domain()->hasGridReduction() || is_cluster_reduction ||
         std::any_of(
             out->getLoopDomain().begin(),
             out->getLoopDomain().end(),
             [](IterDomain* id) {
               return id->getParallelType() == ParallelType::Group;
             }))) {
      reduction_dep_[out].insert(rop);
    }

    if (is_cluster_reduction) {
      cluster_reduction_count_++;
    }
  }
  void handle(WelfordOp* wop) final {
    // If it's a grid welford, keep track of tensors that depend on
    // this reduction.

    // Skip if any of the outputs is not a Local tensor.
    auto out_tvs = ir_utils::filterByType<TensorView>(wop->outputs());
    if (std::any_of(out_tvs.begin(), out_tvs.end(), [](auto out_tv) {
          return out_tv->getMemoryType() != MemoryType::Local;
        })) {
      return;
    }

    // Keep track of all output TVs if grid parallelized
    auto out = ir_utils::getTvOutput(wop);
    if (out->domain()->hasGridReduction()) {
      for (auto out : out_tvs) {
        reduction_dep_[out].insert(wop);
      }
    }
  }

  void handle(GroupedReductionOp* grouped_rop) final {
    // Skip if any of the outputs is not a Local tensor.
    auto out_tvs = ir_utils::filterByType<TensorView>(grouped_rop->outputs());
    if (std::any_of(out_tvs.begin(), out_tvs.end(), [](auto out_tv) {
          return out_tv->getMemoryType() != MemoryType::Local;
        })) {
      return;
    }

    // Keep track of all output TVs if grid parallelized
    auto out = ir_utils::getTvOutput(grouped_rop);
    if (out->domain()->hasGridReduction()) {
      for (auto out : out_tvs) {
        reduction_dep_[out].insert(grouped_rop);
      }
    }
  }

  void dispatch(Expr* expr) final {
    IterVisitor::dispatch(expr);
    for (auto in_tv : ir_utils::filterByType<TensorView>(expr->inputs())) {
      for (auto reduction_op : reduction_dep_[in_tv]) {
        if (fused_exprs_.find(reduction_op) != fused_exprs_.end()) {
          continue;
        }
        for (auto out_tv :
             ir_utils::filterByType<TensorView>(expr->outputs())) {
          reduction_dep_[out_tv].insert(reduction_op);
        }
      }
    }
  }

  // In the case of welford, use the fused broadcast reduction when at
  // least one of the outputs is broadcast.
  void handle(BroadcastOp* bop) final {
    // Detect a pattern where a reduction is followed by a broadcast
    auto bop_out = bop->out()->as<TensorView>();
    auto bop_in = bop->in()->as<TensorView>();

    for (Expr* preceding_expr : reduction_dep_[bop_in]) {
      auto parallel_reduction_axes =
          getReductionParallelTypeStates(preceding_expr);

      // If not matching, propagate the reduction further down to
      // subsequent expressions
      if (!isBroadcastFuseable(bop_out, parallel_reduction_axes)) {
        continue;
      }

      if (fused_exprs_.find(preceding_expr) != fused_exprs_.end()) {
        // Already added to the fusion list. This can happen with
        // welford as there can be multiple broadcast consumer
        // expressions.
        continue;
      }

      if (preceding_expr->isA<ReductionOp>()) {
        fusion_list_.emplace_back(preceding_expr->as<ReductionOp>(), true);
      } else if (preceding_expr->isA<GroupedReductionOp>()) {
        fusion_list_.emplace_back(
            preceding_expr->as<GroupedReductionOp>(), true);
      } else if (preceding_expr->isA<WelfordOp>()) {
        fusion_list_.emplace_back(preceding_expr->as<WelfordOp>(), true);
      } else {
        NVF_THROW("Invalid preceding expr: ", preceding_expr->toString());
      }

      fused_exprs_.insert(preceding_expr);
    }
  }

  ParallelTypeBitmap getReductionParallelTypeStates(Expr* expr) {
    ParallelTypeBitmap parallel_reduction_axes;

    for (auto id : ir_utils::getTvOutput(expr)->getLoopDomain()) {
      auto pt = id->getParallelType();
      if (id->isReduction() && isParallelTypeThread(pt)) {
        parallel_reduction_axes.set(pt);
      }
    }

    return parallel_reduction_axes;
  }

  // Requires reduction parallel dimensions to exactly match parallel
  // broadcast dimensions
  bool isBroadcastFuseable(
      TensorView* broadcast_out,
      const ParallelTypeBitmap& parallel_reduction_axes) {
    const auto broadcast_parallel_types =
        GpuLower::current()
            ->info()
            .threadPredicateMap()
            .getParallelBroadcastDomains(broadcast_out);

    // If no parallel broadcast, nothing to fuse
    if (broadcast_parallel_types.none()) {
      return false;
    }

    // Make sure the broadcast parallel types are the types reduced by
    // the preceding reduction op
    for (auto id : broadcast_out->getLoopDomain()) {
      auto pt = id->getParallelType();
      if (!isParallelTypeThread(pt)) {
        continue;
      }
      // Parallel broadcast must be included in reduction_states
      if (id->isBroadcast() && broadcast_parallel_types.get(pt)) {
        if (!parallel_reduction_axes.get(pt)) {
          return false;
        }
      }
    }

    return true;
  }

 private:
  //! List of expression sequences to fuse
  std::vector<FusedReductionBroadcastInfo> fusion_list_;
  //! Keep track of fused reduction/welford exprs to avoid duplication
  std::unordered_set<Expr*> fused_exprs_;
  //! Keep track of ReductionOp/WelfordOp expressions that are
  //! (indirectly) input to a tensor
  std::unordered_map<TensorView*, std::unordered_set<Expr*>> reduction_dep_;
  //! Whether this fusion has warp specialization enabled
  const bool has_warp_specialization_;
  //! Track number of cluster reductions, used for mbarrier allocation
  //! as for each cluster reduction, we need to allocate a mbarrier
  int64_t cluster_reduction_count_ = 0;
};

//! Transform a fusion to use the fused reduction kernel.
class FusionTransformer {
 public:
  static FusedReductionInfo run(
      Fusion* fusion,
      const std::vector<FusedReductionBroadcastInfo>& fusion_list) {
    FusionTransformer transformer(fusion, fusion_list);
    return transformer.info_;
  }

 private:
  FusionTransformer(
      Fusion* fusion,
      const std::vector<FusedReductionBroadcastInfo>& fusion_list)
      : fusion_(fusion), fusion_list_(fusion_list) {
    transform();
  }

  void transform() {
    for (const auto& info : fusion_list_) {
      transform(info);
    }
  }

  void transform(const FusedReductionBroadcastInfo& info) {
    NVF_ERROR(
        info.reductions().size() == 1, "Horizontal fusion not supported yet");

    for (const auto i : arange(info.reductions().size())) {
      const auto expr = info.reductions().at(i);
      const auto with_broadcast = info.withBroadcast().at(i);
      Expr* fused_expr = nullptr;

      if (auto reduction = dynamic_cast<ReductionOp*>(expr)) {
        NVF_ERROR(!reduction->isAllreduce());

        auto red_op_type = reduction->getReductionOpType();
        auto init = reduction->init();
        auto out = reduction->out();
        auto in = reduction->in();

        fusion_->removeExpr(reduction);

        fused_expr =
            IrBuilder::create<ReductionOp>(red_op_type, init, out, in, true);
      } else if (auto welford = dynamic_cast<WelfordOp*>(expr)) {
        NVF_ERROR(!welford->isAllreduce());

        auto out_avg = welford->outAvg();
        auto out_var = welford->outVar();
        auto out_n = welford->outN();
        auto init_avg = welford->initAvg();
        auto init_var = welford->initVar();
        auto init_n = welford->initN();
        auto in_avg = welford->inAvg();
        auto in_var = welford->inVar();
        auto in_n = welford->inN();

        fusion_->removeExpr(welford);

        fused_expr = IrBuilder::create<WelfordOp>(
            WelfordTriplet{out_avg, out_var, out_n},
            WelfordTriplet{in_avg, in_var, in_n},
            WelfordTriplet{init_avg, init_var, init_n},
            true);
      } else if (auto grouped_rop = dynamic_cast<GroupedReductionOp*>(expr)) {
        NVF_ERROR(!grouped_rop->isAllreduce());

        auto op_types = grouped_rop->getReductionOpTypes();
        auto init_vals = grouped_rop->initVals();
        auto outputs = grouped_rop->outputs();
        auto inputs = grouped_rop->inputs();

        fusion_->removeExpr(grouped_rop);

        fused_expr = IrBuilder::create<GroupedReductionOp>(
            op_types, init_vals, outputs, inputs, true);
      } else {
        NVF_ERROR(expr != nullptr);
        NVF_THROW("Invalid expr: ", expr->toString());
      }

      NVF_ERROR(fused_expr != nullptr);

      // Do not just remove the broadcast but just reset the thread
      // predicate of the broadcast op. Since fusion is applied only
      // when all parallel broadcast domains are to be parallel
      // reduction, all parallel types can be reset.
      if (with_broadcast) {
        // It may be just fine to remove the broadcast expr, but
        // technically speaking that would violate the root domain mapping
        // as broadcast domains would appear in the consumer of the
        // broadcast output tensor without a broadcast expression.
        for (auto reduction_out :
             ir_utils::filterByType<TensorView>(fused_expr->outputs())) {
          for (auto id : reduction_out->getLoopDomain()) {
            if (id->isReduction()) {
              info_.markAsAllreduce(id);
            }
          }
        }
      }
    }
  }

 private:
  Fusion* fusion_ = nullptr;
  const std::vector<FusedReductionBroadcastInfo>& fusion_list_;
  FusedReductionInfo info_;
};

} // namespace

FusedReductionInfo fuseReductionsAndBroadcasts(Fusion* fusion) {
  auto fusion_list = FusionInspector::run(fusion);
  return FusionTransformer::run(fusion, fusion_list);
}

void FusedReductionInfo::markAsAllreduce(IterDomain* id) {
  allreduce_ids_.insert(id);
}

bool FusedReductionInfo::isAllreduce(IterDomain* id) const {
  return allreduce_ids_.find(id) != allreduce_ids_.end();
}

} // namespace nvfuser
