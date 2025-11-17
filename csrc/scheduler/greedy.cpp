// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on

#include <debug.h>
#include <device_lower/analysis/fusion_info.h>
#include <device_lower/analysis/sync_information.h>
#include <device_lower/utils.h>
#include <exceptions.h>
#include <id_model/id_model.h>
#include <id_model/to_string.h>
#include <instrumentation.h>
#include <ir/utils.h>
#include <iter_visitor.h>
#include <ops/all_ops.h>
#include <options.h>
#include <scheduler/debug_utils.h>
#include <scheduler/greedy.h>
#include <scheduler/mark_aliases.h>
#include <scheduler/runtime_info.h>
#include <scheduler/tools/cub_utils.h>
#include <scheduler/tools/inlining.h>
#include <scheduler/tools/loop_domain_scheduler.h>
#include <scheduler/tools/maxinfo_propagator.h>
#include <scheduler/utils.h>
#include <transform_replay.h>
#include <val_graph_visitor.h>

#include <ATen/cuda/CUDAContext.h>

#include <functional>
#include <ranges>
#include <vector>

namespace nvfuser {

GreedyParams::GreedyParams() : HeuristicParams(SchedulerType::Greedy) {
  tag = "Greedy heuristics";
}

bool GreedyParams::sameAs(const HeuristicParams* other_base) const {
  auto other = dynamic_cast<const GreedyParams*>(other_base);
  if (other == nullptr) {
    return false;
  }
  bool attr_equal = consumer_to_params_ == other->consumer_to_params_ &&
      producer_to_params_ == other->producer_to_params_;
  return attr_equal;
}

std::string GreedyParams::toString() const {
  std::stringstream ss;
  ss << "\n========= Greedy Parameters ========\n";
  for (const auto& [tv_name, params] : consumer_to_params_) {
    ss << "t" << tv_name << " (consumer) -> " << params.toString() << "\n";
  }
  for (const auto& [producer_consumer_pair, params] : producer_to_params_) {
    ss << "t" << producer_consumer_pair.first << " (producer) for "
       << "t" << producer_consumer_pair.second << " (consumer) -> "
       << params.toString() << "\n";
  }
  ss << "====================================\n";
  return ss.str();
}

size_t GreedyParams::hash() const {
  size_t x = 0;
  for (const auto& [tv, size] : consumer_to_params_) {
    x = x ^ std::hash<int64_t>()(size.batch_size);
  }
  for (const auto& [producer_consumer_pair, size] : producer_to_params_) {
    x = x ^ std::hash<int64_t>()(size.batch_size);
  }
  return x;
}

std::unique_ptr<HeuristicParams> GreedyParams::clone() const {
  return std::make_unique<GreedyParams>(*this);
}

bool GreedyParams::hasConsumerParams(TensorView* consumer_tv) {
  return consumer_to_params_.contains(consumer_tv->name());
}

bool GreedyParams::hasProducerParams(
    TensorView* producer_tv,
    TensorView* consumer_tv) const {
  return producer_to_params_.contains(
      std::make_pair(producer_tv->name(), consumer_tv->name()));
}

void GreedyParams::transferConsumerParams(
    TensorView* old_tv,
    TensorView* new_tv) {
  NVF_ERROR(old_tv != nullptr);
  NVF_ERROR(new_tv != nullptr);
  auto it = consumer_to_params_.find(old_tv->name());
  if (it == consumer_to_params_.end()) {
    return;
  }
  NVF_ERROR(
      setConsumerParams(new_tv, it->second),
      "Duplicated setting for ",
      new_tv->toString());
  // Remove the old entry
  consumer_to_params_.erase(old_tv->name());
}

void GreedyParams::transferProducerParams(
    TensorView* old_producer_tv,
    TensorView* old_consumer_tv,
    TensorView* new_producer_tv,
    TensorView* new_consumer_tv) {
  NVF_ERROR(old_producer_tv != nullptr);
  NVF_ERROR(old_consumer_tv != nullptr);
  NVF_ERROR(new_producer_tv != nullptr);
  NVF_ERROR(new_consumer_tv != nullptr);
  auto it = producer_to_params_.find(
      std::make_pair(old_producer_tv->name(), old_consumer_tv->name()));
  if (it == producer_to_params_.end()) {
    return;
  }
  NVF_ERROR(
      setProducerParams(new_producer_tv, new_consumer_tv, it->second),
      "Duplicated setting for ",
      new_producer_tv->toString(),
      ", ",
      new_consumer_tv->toString());
  // Remove the old entry
  producer_to_params_.erase(
      std::make_pair(old_producer_tv->name(), old_producer_tv->name()));
}

namespace {

// Utility function to get the total size of the given IDs if all
// extents are statically known
std::optional<int64_t> getMaybeStaticSize(const std::vector<IterDomain*>& ids) {
  NVF_ERROR(!ids.empty());

  bool all_static_ids = true;
  int64_t static_size = 1;

  for (const auto& id : ids) {
    if (id->getMaybeExpandedExtent()->isConstInt()) {
      auto extent_int = id->getMaybeExpandedExtent()->evaluate().as<int64_t>();
      static_size *= extent_int;
    } else {
      all_static_ids = false;
      break;
    }
  }

  if (all_static_ids) {
    return static_size;
  } else {
    return std::nullopt;
  }
}

// These are the current supported constrained ops.
bool isConstrainedOp(Expr* expr) {
  return expr != nullptr &&
      expr->isOneOf<ArgsortOp, PadOp, ScanOp, ScatterOp, TopKOp>();
}

std::vector<Expr*> getAllConstrainedOps(Fusion* fusion) {
  std::vector<Expr*> ops;
  std::ranges::copy_if(
      fusion->exprs(), std::back_inserter(ops), isConstrainedOp);
  return ops;
}

std::vector<TensorView*> getAllConstrainedTvs(Fusion* fusion) {
  const auto constrained_exprs = getAllConstrainedOps(fusion);

  std::vector<TensorView*> constrained_tvs;

  // All outputs of constrained ops are considered constrained
  constrained_tvs.reserve(constrained_exprs.size());
  std::ranges::transform(
      constrained_exprs,
      std::back_inserter(constrained_tvs),
      [](const Expr* expr) { return ir_utils::getTvOutput(expr); });

  // Grab additional constrained tensors
  for (auto expr : constrained_exprs) {
    if (auto scatter = dynamic_cast<ScatterOp*>(expr)) {
      // ScatterOp's inputs are also considered constrained unless it's
      // a fusion input. Fusion inputs don't need to be scheduled, so they
      // shouldn't impose any constraint.
      for (auto inp : scatter->inputs()) {
        if (!inp->isFusionInput() && inp->isA<TensorView>()) {
          constrained_tvs.push_back(inp->as<TensorView>());
        }
      }
      for (auto use : scatter->out()->uses()) {
        auto use_tv_out = ir_utils::getTvOutput(use);
        if (use_tv_out == nullptr) {
          continue;
        }
        if (std::ranges::find(constrained_tvs, use_tv_out) ==
            constrained_tvs.end()) {
          constrained_tvs.push_back(use_tv_out);
        }
      }
    } else if (auto topk = dynamic_cast<TopKOp*>(expr)) {
      // Similar to ScatterOp, TopKOp inputs are also considered
      // constrained since there's a resize between the output and
      // input tensors.
      if (!topk->in()->isFusionInput()) {
        constrained_tvs.push_back(topk->in()->as<TensorView>());
      }
    }
  }

  return constrained_tvs;
}

// Given offsets of logical IDs, return corresponding loop ID offsets
std::vector<int64_t> getDependentLoopIds(
    TensorView* tv,
    const std::vector<int64_t>& logical_id_offsets) {
  std::vector<Val*> logical_ids;
  logical_ids.reserve(logical_id_offsets.size());
  std::ranges::transform(
      logical_id_offsets,
      std::back_inserter(logical_ids),
      [tv](int64_t logical_id_offset) {
        return tv->getLogicalDomain().at(logical_id_offset);
      });

  const auto logical_loop_all_ids = DependencyCheck::getAllValsBetween(
      {logical_ids.begin(), logical_ids.end()},
      {tv->getLoopDomain().begin(), tv->getLoopDomain().end()});
  const std::unordered_set<Val*> logical_loop_all_id_set{
      logical_loop_all_ids.begin(), logical_loop_all_ids.end()};

  std::vector<int64_t> loop_id_offsets;
  for (const auto [i, loop_id] : enumerate(tv->getLoopDomain())) {
    if (logical_loop_all_id_set.contains(loop_id)) {
      loop_id_offsets.push_back(i);
    }
  }

  return loop_id_offsets;
}

class CompileTimeChecker : private IterVisitor {
 public:
  static bool run(Fusion* fusion, const ValGraph& exact_graph) {
    CompileTimeChecker checker(fusion, exact_graph);
    if (!checker.can_schedule_ && !checker.reject_reason_.empty()) {
      scheduler_debug_utils::canScheduleRejectReason(
          SchedulerType::Greedy, checker.reject_reason_);
    }
    return checker.can_schedule_;
  }

 private:
  CompileTimeChecker(Fusion* fusion, const ValGraph& exact_graph)
      : exact_graph_(exact_graph) {
    checkConflictingReshape();
    if (!can_schedule_) {
      return;
    }

    traverse(fusion);
    if (!can_schedule_) {
      return;
    }

    if (needs_all_tid_participation_ &&
        (!has_largest_constrained_size_ ||
         std::ranges::any_of(
             all_exact_constrained_sizes_, [&](int64_t constrained_size) {
               return constrained_size < largest_constrained_size_;
             }))) {
      reject(
          "Found constrained ops for which all threads must participate "
          "without predication but not guaranteed");
    }

    // Make sure constrained and unconstrained ids are
    // disjoint. Because of the requirement that all ID groups must be
    // used uniquely (no multiple distinctive use Expr groups), it is
    // suffient to look at reachable graph nodes from each of the
    // groups by a forward traversal and see if there's any common
    // nodes. Because there's no ID group that has multiple uses, it
    // is not necessary to traverse backward.
    if (unique_unconstrained_domain_.has_value()) {
      auto reachable_vals_from_unconstrained_domain =
          getReachableValsFrom<ValGraphPermissiveBFS>(
              unique_unconstrained_domain_.value().vector(),
              exact_graph_.disjointValSets().disjointSets(),
              Direction::Forward,
              exact_graph_);
      auto common_reachable_ids = getReachableValsFrom<ValGraphPermissiveBFS>(
          all_constrained_domain_.vector(),
          reachable_vals_from_unconstrained_domain,
          Direction::Forward,
          exact_graph_);
      if (!common_reachable_ids.empty()) {
        reject(
            "Constrained and unconstrained IDs are merged at: ",
            nvfuser::toString(common_reachable_ids));
      }
    }
  }

  void dispatch(Expr* expr) override {
    if (!can_schedule_) {
      return;
    }

    // These are the ops that are currently allowed to exist in the
    // given fusion. Notably, BroadcastOp, ReductionOp and ReshapeOp
    // are still missing.
    can_schedule_ = can_schedule_ &&
        expr->isOneOf<
            LoadStoreOp,
            UnaryOp,
            BinaryOp,
            TernaryOp,
            FullOp,
            ReshapeOp,
            IotaOp,
            BroadcastOp,
            SqueezeOp,
            ArgsortOp,
            ScanOp,
            PadOp,
            ScatterOp,
            TopKOp>();
    if (!can_schedule_) {
      reject("Unsupported operation: ", expr->toString());
      return;
    }
    IterVisitor::dispatch(expr);
  }

  void handle(ArgsortOp* argsort) override {
    auto out_tv = ir_utils::getTvOutput(argsort);
    checkDomainConstraints(out_tv->getLogicalDomain(), {argsort->dim()});
    if (!can_schedule_) {
      return;
    }

    // Only static dim supported for now. See also
    // CudaKernelGenerator::handle(ArgsortOp*)
    auto sorted_id = out_tv->getLogicalDomain().at(argsort->dim());
    if (!sorted_id->extent()->isConstInt()) {
      reject("Symbolic dimension not supported yet: ", argsort->toString());
      return;
    }
  }

  void handle(ScanOp* scan) override {
    auto out_tv = ir_utils::getTvOutput(scan);
    checkDomainConstraints(out_tv->getLogicalDomain(), {scan->dim()});

    // Only static dim supported for now. See also
    // CudaKernelGenerator::handle(ScanOp*)
    auto scan_id = out_tv->getLogicalDomain().at(scan->dim());
    if (!scan_id->extent()->isConstInt()) {
      reject("Symbolic dimension not supported yet: ", scan->toString());
      return;
    }
  }

  void handle(ScatterOp* scatter) override {
    auto inp = scatter->in()->as<TensorView>();
    auto out = scatter->out()->as<TensorView>();

    if (!scatter->exactSizes()) {
      reject("Non-exact scatter is not yet supported");
      return;
    }

    // Scatter input tensor is only allowed to be used by this scatter
    // op itself due to the input-output aliasing
    if (inp->uses().size() != 1) {
      reject("Scatter input can only be used by the scatter op itself");
      return;
    }

    // If allocation domains already exist for the input and
    // output, they must match since the input and output need to
    // share the same memory buffer. This condition does not matter if
    // no explicit allocation domain exists since we can set the
    // allocation domain as required for this scheduler.
    if (inp->hasAllocation() && out->hasAllocation() &&
        (exact_graph_.toGroups(inp->getAllocationDomain()) !=
         exact_graph_.toGroups(out->getAllocationDomain()))) {
      reject("Scatter input and output do not have the same allocation domain");
      return;
    }

    // This is just a temporary shortcut for simplicity: The allocation domain
    // must stay the same as the logical domain. This condition should
    // be lifted.
    if (inp->hasAllocation() &&
        inp->getAllocationDomain() != inp->getLogicalDomain()) {
      reject(
          "Scatter input has an allocation domain that is not the same as the "
          "logical domain: ",
          inp->toString(),
          ", allocation: ",
          toDelimitedString(inp->getAllocationDomain()),
          ", logical: ",
          toDelimitedString(inp->getLogicalDomain()));
      return;
    }
    if (out->hasAllocation() &&
        out->getAllocationDomain() != out->getLogicalDomain()) {
      reject(
          "Scatter out has an allocation domain that is not the same as the "
          "logical domain: ",
          out->toString(),
          ", allocation: ",
          toDelimitedString(out->getAllocationDomain()),
          ", logical: ",
          toDelimitedString(out->getLogicalDomain()));
      return;
    }

    // In the case of scatter, the scatter dimension doesn't need to
    // be parallelized with TID, but we need to make sure it isn't
    // parallelized with BID. In that sense, categorizing it as a
    // constrained ID may be too restrictive.
    // TODO: Consider introducing another group of IDs that are semi
    // constrained.
    auto constrained_out_logical_dim = scatter->dim();

    // For the scatter in and out tensors, the scatter dimension must
    // not be parallelized with TID. Note that if the input is not
    // produced within this fusion and the output is not further used,
    // it doesn't matter. In that case, the scatter op should not be
    // considered constrained from the beginning.
    // TODO: Exclude such scatter ops from constrained ops
    auto out_tv = ir_utils::getTvOutput(scatter);
    checkDomainConstraints(
        out_tv->domain()->logical(), {constrained_out_logical_dim});

    // In addition, the index and src tensors are not allowed to use
    // BID with the scatter dim. Their logical domains are not mapped
    // with the logical domains of the input and output tensors, so
    // they need to be checked separately.
    checkDomainConstraints(
        TensorDomain::noReductions(
            scatter->index()->as<TensorView>()->getLogicalDomain()),
        {constrained_out_logical_dim});
    // Index and src tensors are mapped, so just checking index should
    // be sufficient.
  }

  void handle(PadOp* pad) override {
    checkDomainConstraints(
        ir_utils::getTvOutput(pad)->getLogicalDomain(), pad->getPaddedAxes());

    for (const auto& logical_id :
         ir_utils::getTvOutput(pad)->getLogicalDomain()) {
      auto resize = dynamic_cast<Resize*>(logical_id->definition());
      if (resize == nullptr) {
        continue;
      }
      // Resize to broadcast not supported yet. Have not looked at
      // details but getLoopPromotion fails at csrc/id_model/utils.h:105 (e.g.,
      // ResizeTest.ResizePadToBroadcastStatic), likely because
      // broadcast IDs are introduced without BroadcastOp.
      // This is also the case with the resize scheduler.
      if (resize->out()->isBroadcast()) {
        reject("Resize to a broadcast ID is not allowed: ", pad->toString());
        return;
      }
    }
  }

  void handle(TopKOp* topk) override {
    auto in_tv = ir_utils::getTvInput(topk);
    auto out_tv = ir_utils::getTvOutput(topk);

    // Like ScatterOp, the input defines the scheduling, so check the
    // input logical domain
    checkDomainConstraints(in_tv->getLogicalDomain(), {topk->dim()});

    // Only static dim supported for now.
    auto topk_id = out_tv->getLogicalDomain().at(topk->dim());
    if (!topk_id->extent()->isConstInt()) {
      reject("Symbolic dimension not supported yet: ", topk->toString());
      return;
    }
  }

  // Check if the logical IDs of the given constrained tv can be
  // acceptable.
  //
  // When require_exact_constrained_ids is true, the aggregated size
  // of the constrained IDs must be the largest among all the
  // constrained tensors.
  void checkDomainConstraints(
      const std::vector<IterDomain*>& domain_to_check,
      const std::vector<int64_t>& constrained_id_offsets,
      bool require_exact_constrained_ids = false) {
    const std::unordered_set<int64_t> constrained_id_offset_set(
        constrained_id_offsets.begin(), constrained_id_offsets.end());

    ValGroups constrained_domain;
    ValGroups unconstrained_domain;
    std::vector<IterDomain*> constrained_ids;
    for (const auto& [i, id] : enumerate(domain_to_check)) {
      if (constrained_id_offset_set.contains(i)) {
        constrained_ids.push_back(id);
        const auto& id_group = exact_graph_.toGroup(id);
        constrained_domain.pushBack(id_group);
        // Keep track of all constrained IDs as well for reshape analysis
        all_constrained_domain_.pushBack(id_group);
      } else {
        // Broadcast should not matter for scheduling
        if (id->isBroadcast()) {
          continue;
        }
        unconstrained_domain.pushBack(exact_graph_.toGroup(id));
      }
    }

    // All the unconstrained iter domains would be flattened and
    // parallelized with BIDx. The BIDx parallelized iter
    // domain must be mapped across the fusion to avoid the grid
    // synchronization. For the mapping, the exact graph is used for
    // now since broadcast IDs can be ignored for this analysis.

    if (unique_unconstrained_domain_.has_value()) {
      if (unique_unconstrained_domain_->set() != unconstrained_domain.set()) {
        reject(
            "Mismatched unconstrained IDs detected with ",
            toDelimitedString(domain_to_check),
            ": ",
            nvfuser::toString(unconstrained_domain),
            ". Ref: ",
            nvfuser::toString(*unique_unconstrained_domain_));
        unique_unconstrained_domain_.reset();
      }
    } else {
      unique_unconstrained_domain_ = unconstrained_domain;
    }

    // Keep track of the largest size of the constrained IDs if statically
    // known
    if (has_largest_constrained_size_) {
      auto static_size = getMaybeStaticSize(constrained_ids);
      if (static_size.has_value()) {
        largest_constrained_size_ =
            std::max(largest_constrained_size_, static_size.value());
        if (require_exact_constrained_ids) {
          all_exact_constrained_sizes_.insert(static_size.value());
        }
      } else {
        has_largest_constrained_size_ = false;
      }
    }
  }

  // In order to ensure no conflicting reshape exists, fusions are
  // only allowed to have one use ExprGroup for each ID group. This
  // condition is not strictly necessary, but it makes the
  // can-schedule analysis fairly simple as seen below.
  void checkConflictingReshape() {
    for (const ValGroup& val_group :
         exact_graph_.disjointValSets().disjointSets()) {
      const auto& use_groups = exact_graph_.getUses(val_group);
      // Root-to-logical exprs may include Resize ops too, but they
      // can be ignored for this analysis since transformations are
      // simply propagated along Resize ops
      int num_reshape_exprs = 0;
      for (const auto& use_group : use_groups) {
        if (use_group->front()->isA<Merge>() ||
            use_group->front()->isA<Split>()) {
          ++num_reshape_exprs;
        }
      }
      if (num_reshape_exprs > 1) {
        reject(
            "Potentially conflicting reshape found for ",
            nvfuser::toString(val_group));
        return;
      }
    }
  }

  template <typename... Args>
  void reject(Args&&... args) {
    can_schedule_ = false;
    // Only keeps the first reason
    if (reject_reason_.empty()) {
      std::stringstream reason;
      ((reason << args << " "), ...);
      reject_reason_ = reason.str();
    }
  }

 private:
  const ValGraph& exact_graph_;

  bool can_schedule_ = true;
  std::string reject_reason_;

  std::optional<ValGroups> unique_unconstrained_domain_;

  ValGroups all_constrained_domain_;

  // True if all threads need to participate without predicates. This
  // was previously used for ops like TopKOp.
  bool needs_all_tid_participation_ = false;
  bool has_largest_constrained_size_ = true;
  int64_t largest_constrained_size_ = -1;
  // Sizes of iter domains where all threads must participate without predicate
  std::unordered_set<int64_t> all_exact_constrained_sizes_;
};

class RunTimeChecker : private IterVisitor {
 public:
  static bool run(Fusion* fusion, SchedulerRuntimeInfo& runtime_info) {
    RunTimeChecker checker(fusion, runtime_info);
    if (!checker.can_schedule_ && !checker.reject_reason_.empty()) {
      scheduler_debug_utils::canScheduleRejectReason(
          SchedulerType::Greedy, checker.reject_reason_);
    }
    return checker.can_schedule_;
  }

 private:
  RunTimeChecker(Fusion* fusion, SchedulerRuntimeInfo& runtime_info)
      : runtime_info_(runtime_info),
        max_threads_per_block_(
            at::cuda::getCurrentDeviceProperties()->maxThreadsPerBlock) {
    traverse(fusion);

    checkSharedMemoryBufferUsage();
  }

  void dispatch(Expr* expr) override {
    if (!can_schedule_) {
      return;
    }
    IterVisitor::dispatch(expr);
  }

  void handle(ArgsortOp* argsort) override {
    int64_t size_of_constrained_ids = checkDomainConstraints(
        ir_utils::getTvOutput(argsort)->getLogicalDomain(),
        {argsort->dim()},
        dataTypeSizeByte(ir_utils::getTvOutput(argsort)->dtype()),
        /*support_batching=*/true);

    int64_t batch_size =
        ceilDiv(size_of_constrained_ids, max_threads_per_block_);
    int64_t bdimx = std::min(size_of_constrained_ids, max_threads_per_block_);
    cub_shmem_buffer_.registerArgsort(
        bdimx, batch_size, ir_utils::getTvInput(argsort)->dtype());
  }

  void handle(PadOp* pad) override {
    checkDomainConstraints(
        ir_utils::getTvOutput(pad)->getLogicalDomain(),
        pad->getPaddedAxes(),
        dataTypeSizeByte(ir_utils::getTvOutput(pad)->dtype()));
  }

  void handle(ScanOp* scan) override {
    checkDomainConstraints(
        ir_utils::getTvOutput(scan)->getLogicalDomain(),
        {scan->dim()},
        dataTypeSizeByte(ir_utils::getTvOutput(scan)->dtype()),
        /*support_batching=*/true);
  }

  void handle(TopKOp* topk) override {
    // TopKOp produces two outputs: one has the same type as the input
    // and another is an integer index tensor
    int64_t size_of_constrained_ids = checkDomainConstraints(
        TensorDomain::noReductions(
            ir_utils::getTvInput(topk)->getLogicalDomain()),
        {topk->dim()},
        dataTypeSizeByte(ir_utils::getTvInput(topk)->dtype()) +
            dataTypeSizeByte(DataType::Int),
        /*support_batching=*/true);

    int64_t batch_size =
        ceilDiv(size_of_constrained_ids, max_threads_per_block_);
    int64_t bdimx = std::min(size_of_constrained_ids, max_threads_per_block_);
    cub_shmem_buffer_.registerTopK(
        bdimx, batch_size, ir_utils::getTvInput(topk)->dtype());
  }

  void handle(ScatterOp* scatter) override {
    auto out = ir_utils::getTvOutput(scatter);
    auto index = scatter->index()->as<TensorView>();

    // TODO: If the input and output is a fusion input and output,
    // there will be no computation for the shape of the logical
    // domain, so this check is not necessary.
    checkDomainConstraints(
        out->getLogicalDomain(),
        {scatter->dim()},
        dataTypeSizeByte(out->dtype()),
        /*support_batching=*/true);

    int64_t index_bytes = dataTypeSizeByte(index->dtype());
    // If it's scalar, ignore the contribution
    int64_t src_bytes = scatter->src()->isA<TensorView>()
        ? dataTypeSizeByte(scatter->src()->dtype())
        : 0;

    checkDomainConstraints(
        TensorDomain::noReductions(index->getLogicalDomain()),
        {scatter->dim()},
        index_bytes + src_bytes,
        /*support_batching=*/true);
  }

  // Check the constraints on the given domain. bytes_per_element
  // indicates the size of data required to hold one work item, which
  // may correspond to multiple tensor elements. For example, in the
  // case of TopKOp, two outputs are produced, so the size should
  // cover both of them.
  //
  // Returns the size of the constrained IDs in bytes
  int64_t checkDomainConstraints(
      const std::vector<IterDomain*>& domain,
      const std::vector<int64_t>& constrained_id_offsets,
      int64_t bytes_per_element,
      bool support_batching = false) {
    int64_t size_of_constrained_ids = 1;
    for (const auto i : constrained_id_offsets) {
      auto constrained_id = domain.at(i);
      auto extent_val = runtime_info_.expressionEvaluator().evaluate(
          constrained_id->extent());
      NVF_ERROR(
          extent_val.hasValue(),
          "Cannot infer the extent of a constrained ID: ",
          constrained_id->toString());
      size_of_constrained_ids *= extent_val.as<int64_t>();
    }

    const int64_t threads_per_block = max_threads_per_block_;

    // At this moment, not all constrained ops supports batching. If
    // batching is not supported, the limit is simply set as the
    // maximum number of threads per thread block. This is likely
    // a sufficient condition even for shared memory, although not
    // guaranteed.
    if (!support_batching) {
      if (size_of_constrained_ids > threads_per_block) {
        reject(
            "Extent of constrained logical IDs, ",
            size_of_constrained_ids,
            ", exceeds the number of threads per thread block: ",
            threads_per_block);
      }
    }

    // The maximum supported size depends on several factors. The hard
    // limit is the shared memory capacity since the kernel launch
    // would just fail if the shared memory usage exceeds the
    // available size. It is checked at the end of the RunTimeChecker
    // constructor.
    //
    // The next important limit would be the register usage as we
    // would not want to have excessive register spilling. The
    // register usage would be linearly correlated with the batching
    // factor. For now, just put a simple upper limit to avoid
    // disastrous regressions. Fine tuning would be necessary.
    const int64_t register_count_per_thread =
        ceilDiv(size_of_constrained_ids, threads_per_block) *
        bytes_per_element / 4;
    const int64_t available_register_count_per_thread =
        at::cuda::getCurrentDeviceProperties()->regsPerBlock /
        threads_per_block;
    // Make sure at least 20 registers are always available
    const int64_t reserved_register_count_per_thread = 20;
    if (register_count_per_thread + reserved_register_count_per_thread >
        available_register_count_per_thread) {
      reject(
          "Expected register usage, ",
          register_count_per_thread,
          ", exceeds the available count, ",
          available_register_count_per_thread);
    }

    return size_of_constrained_ids;
  }

  void checkSharedMemoryBufferUsage() {
    // TODO: Use the constant and util functions added in #5272
    auto aligned_size = [](int64_t x) { return (x + 127) / 128 * 128; };

    const int64_t cub_buffer_size =
        aligned_size(cub_shmem_buffer_.getTotalSizeInBytes());

    // TODO: Shared memory may be also used for resolving mismatched
    // parallelization of constrained.

    const auto total_required_size = cub_buffer_size;

    const auto available_size = static_cast<int64_t>(
        at::cuda::getCurrentDeviceProperties()->sharedMemPerBlock);

    if (total_required_size > available_size) {
      reject(
          "Not enough shared memory. Required size for CUB: ",
          cub_buffer_size,
          ". Total required size: ",
          total_required_size,
          ". Available: ",
          available_size);
    }
  }

  template <typename... Args>
  void reject(Args&&... args) {
    can_schedule_ = false;
    // Only keeps the first reason
    if (reject_reason_.empty()) {
      std::stringstream reason;
      ((reason << args << " "), ...);
      reject_reason_ = reason.str();
    }
  }

 private:
  SchedulerRuntimeInfo& runtime_info_;
  int64_t max_threads_per_block_ = 0;
  scheduler_tools::CubSharedMemoryBuffer cub_shmem_buffer_;

  bool can_schedule_ = true;
  std::string reject_reason_;
};

class HeuristicsBuilder : private IterVisitor {
 public:
  static std::unique_ptr<GreedyParams> run(
      Fusion* fusion,
      SchedulerRuntimeInfo& runtime_info) {
    HeuristicsBuilder builder(fusion, runtime_info);
    return std::move(builder.params_);
  }

 private:
  HeuristicsBuilder(Fusion* fusion, SchedulerRuntimeInfo& runtime_info)
      : runtime_info_(runtime_info) {
    params_ = std::make_unique<GreedyParams>();
    params_->cparams.index_type = runtime_info.getIndexType();

    traverse(fusion);
  }

  void handle(ArgsortOp* argsort) override {
    addHeuristicsFor(
        ir_utils::getTvOutput(argsort),
        ir_utils::getTvOutput(argsort)->getLogicalDomain(),
        {argsort->dim()});
  }

  // TODO: Support batching
  void handle(PadOp* pad) override {
    setDefaultParameters(ir_utils::getTvOutput(pad));
  }

  void handle(ScanOp* scan) override {
    addHeuristicsFor(
        ir_utils::getTvOutput(scan),
        ir_utils::getTvOutput(scan)->getLogicalDomain(),
        {scan->dim()});
  }

  void handle(ScatterOp* scatter) override {
    // Need to have two sets of parameters: one for the logical domain
    // of the index tensor and another for the logical domain of the
    // input tensor. For the former, we store its parameters in the
    // scatter output tensor. For the latter, it's stored as a
    // producer parameter for the input tensor. See
    // ConstrainedOpScheduler::handle(ScatterOp*).
    auto out_tv = ir_utils::getTvOutput(scatter);
    auto inp_tv = ir_utils::getTvInput(scatter);
    addHeuristicsFor(out_tv, out_tv->domain()->initialLoop(), {scatter->dim()});
    addHeuristicsFor(
        inp_tv,
        TensorDomain::noReductions(inp_tv->getLogicalDomain()),
        {scatter->dim()},
        out_tv);
  }

  void handle(TopKOp* topk) override {
    // Batching factor is determined by the dimension of the input
    // topk ID, so add a heuristics parameter based on the topk input
    // tensor
    auto inp_tv = ir_utils::getTvInput(topk);
    auto out_tv = ir_utils::getTvOutput(topk);
    addHeuristicsFor(
        inp_tv,
        TensorDomain::noReductions(inp_tv->getLogicalDomain()),
        {topk->dim()},
        out_tv);
  }

  // Make sure a given tensor has some heuristics parameters
  void setDefaultParameters(TensorView* tv) {
    NVF_ERROR(
        params_->setConsumerParams(tv, {1}),
        "Duplicated setting of item per thread factor for ",
        tv->toString());
  }

  // Register heuristics parameters for constrained_tv. When
  // consumer is non-null, constrained_tv is considered a producer of
  // consumer and a producer parameter is added. Otherwise, a consumer
  // parameter is added for constrained_tv.
  void addHeuristicsFor(
      TensorView* constrained_tv,
      const std::vector<IterDomain*>& domain_to_schedule,
      const std::vector<int64_t>& constrained_id_offsets,
      TensorView* consumer = nullptr) {
    const bool as_consumer = consumer == nullptr;

    int64_t size_of_constrained_ids = 1;
    for (const auto i : constrained_id_offsets) {
      auto logical_id = domain_to_schedule.at(i);
      auto extent_val =
          runtime_info_.expressionEvaluator().evaluate(logical_id->extent());
      NVF_ERROR(
          extent_val.hasValue(),
          "Cannot infer the extent of a constrained logical ID: ",
          logical_id->toString());
      size_of_constrained_ids *= extent_val.as<int64_t>();
    }

    // TODO: The maximum allowed number of threads are launched even
    // when batching is supported. This should be revisited for
    // performance optimization.
    const int64_t bdim =
        at::cuda::getCurrentDeviceProperties()->maxThreadsPerBlock;

    auto batch_size = ceilDiv(size_of_constrained_ids, bdim);

    if (as_consumer) {
      NVF_ERROR(
          params_->setConsumerParams(constrained_tv, {batch_size}),
          "Duplicated setting of item per thread factor for ",
          constrained_tv->toString());
    } else {
      NVF_ERROR(
          params_->setProducerParams(constrained_tv, consumer, {batch_size}),
          "Duplicated setting of item per thread factor for ",
          constrained_tv->toString());
    }
  }

 private:
  SchedulerRuntimeInfo& runtime_info_;

  std::unique_ptr<GreedyParams> params_;
};

// Propagate all reshape transformations throughout the fusion.
void propagateReshape(Fusion* fusion) {
  const auto reshape_ops = ir_utils::getOpsOfType<ReshapeOp>(fusion);
  const auto all_tvs = fusion->allTvs();

  // Visit al reshape ops in a topological order. Propagate the merge
  // and split ops to all tensors as long as they have matching input
  // IDs. Propagation should work consistently as all reshapes are
  // guaranteed to have no conflicting transformations. A single ID
  // group may get propagated multiple times if there are multiple
  // reshapes, but they are guaranteed to have the same
  // transformations.
  for (auto reshape : reshape_ops) {
    auto reshape_exprs = DependencyCheck::getAllExprsBetween(
        {reshape->out()->getRootDomain().begin(),
         reshape->out()->getRootDomain().end()},
        {reshape->out()->getLogicalDomain().begin(),
         reshape->out()->getLogicalDomain().end()});
    scheduler_tools::scheduleLoopDomainsBy(
        all_tvs, reshape_exprs, Direction::Forward);
  }
}

// Scatter: For each scatter output, if there's a use of the output,
// insert a copy between the output and the use (i.e.,
// cacheAfter). This intermediate copy is used to simplify the
// propagation of scheduling from the scatter output tensor. Similarly,
// since scatter inputs need to be scheduled in a particular way, they
// are considered constrained for the scatter op, but they may be also
// produced by another constrained op, which may have different
// scheduling constraints. In order to avoid sheduling the same tensor
// for two different constrained ops, insert a copy for such inputs as
// well.
//
// ArgsortOp, ScanOp, TopKOp: To avoid predicating the output, use a
// new Local tensor as the output if the original output is not a
// Local tensor, and insert a copy from the Local tensor to the
// original output.
//
// ScatterOp, ArgsortOp, ScanOp, TopKOp: Insert a new Local tensor as
// a copy of the input when the input is also produced by another constrained
// op so that the input can be scheduled without any conflict with any other
// constrained ops. For example, in the case of scatter, since its inputs
// need to be scheduled in a particular way, they
// are considered constrained for the scatter op, but they may be also
// produced by another constrained op, which may have different
// scheduling constraints.
void insertCopies(Fusion* fusion, GreedyParams& greedy_params) {
  for (auto expr : getAllConstrainedOps(fusion)) {
    if (expr->isA<ScatterOp>()) {
      auto inp_tv = ir_utils::getTvInput(expr);
      auto out_tv = ir_utils::getTvOutput(expr);
      if (!out_tv->uses().empty()) {
        auto cache = out_tv->cacheAfter(
            LoadStoreOpType::Set,
            CacheOp::Unspecified,
            /*propagate_allocation_domain=*/false);
        // Since the cache, which is a copy of the scatter output,
        // needs to be scheduled as a constrained tensor, add its
        // consumer parameter. That tensor should be scheduled in the
        // same way as the input tensor of the scatter, so set its
        // parameter using the heuristics for the input.
        greedy_params.setConsumerParams(
            cache, greedy_params.getProducerParams(inp_tv, out_tv));
      }
    } else if (expr->isOneOf<ArgsortOp, ScanOp, TopKOp>()) {
      auto outputs = expr->outputs();
      for (auto out_tv : ir_utils::filterByType<TensorView>(outputs)) {
        if (out_tv->getMemoryType() == MemoryType::Local) {
          continue;
        }
        auto cache = out_tv->cacheBefore(LoadStoreOpType::Set);
        // cache is the new output of this op
        expr = cache->definition();
        // cache is the new constraint tv. Transfer heuristic params
        // if any
        greedy_params.transferConsumerParams(out_tv, cache);
      }
    }

    // If an input is produced by a constrained op, make sure it can
    // be scheduled independently from the another constrained op
    // by creating a copy
    if (expr->isOneOf<ArgsortOp, ScanOp, ScatterOp, TopKOp>()) {
      for (const auto inp_tv :
           ir_utils::filterByType<TensorView>(expr->inputs())) {
        if (isConstrainedOp(inp_tv->definition()) ||
            inp_tv->uses().size() > 1) {
          // Insert an exclusive copy
          auto inp_copy = set(inp_tv);
          expr = ir_utils::replaceValInExprInputs(expr, inp_tv, inp_copy);
          greedy_params.transferProducerParams(
              inp_tv,
              ir_utils::getTvOutput(expr),
              inp_copy,
              ir_utils::getTvOutput(expr));
        }
      }
    }
  }
}

class ConstrainedOpScheduler : public OptOutDispatch {
 public:
  static void run(
      Fusion* fusion,
      const ValGraph& exact_graph,
      std::unordered_set<IterDomain*>& uninlinable_ids,
      const GreedyParams* params) {
    ConstrainedOpScheduler scheduler(
        fusion, exact_graph, uninlinable_ids, params);
  }

 private:
  ConstrainedOpScheduler(
      Fusion* fusion,
      const ValGraph& exact_graph,
      std::unordered_set<IterDomain*>& uninlinable_ids,
      const GreedyParams* params)
      : exact_graph_(exact_graph),
        uninlinable_ids_(uninlinable_ids),
        params_(params) {
    for (auto expr : fusion->exprs()) {
      dispatch(expr);
    }
  }

  void handle(ArgsortOp* argsort) override {
    auto out_tv = ir_utils::getTvOutput(argsort);
    auto dim = argsort->dim();
    scheduleConstrainedTv(
        out_tv,
        {dim},
        params_->getConsumerParams(out_tv),
        /*support_grouping=*/true);
  }

  void handle(PadOp* pad) override {
    auto out_tv = ir_utils::getTvOutput(pad);
    scheduleConstrainedTv(
        out_tv, pad->getPaddedAxes(), params_->getConsumerParams(out_tv));
  }

  void handle(ScanOp* scan) override {
    auto scan_dim = scan->dim();
    auto out_tv = ir_utils::getTvOutput(scan);
    scheduleConstrainedTv(
        out_tv,
        {scan_dim},
        params_->getConsumerParams(out_tv),
        /*support_grouping=*/true);
  }

  void handle(ScatterOp* scatter) override {
    auto scatter_dim = scatter->dim();
    auto in_tv = ir_utils::getTvInput(scatter);
    auto index_tv = scatter->index()->as<TensorView>();
    auto out_tv = ir_utils::getTvOutput(scatter);

    // Effectively there are two scheduling patterns around a
    // scatter. One for the scatter loop domain, which is equivalent
    // to the logical domain of the index tensor. Another is for the
    // scatter input tensor and also the consumers of the scatter
    // output. The former is stored as the parameter for the scatter
    // output. The latter is stored as a producer parameter of the
    // input tensor.
    const auto& params_for_index = params_->getConsumerParams(out_tv);
    const auto& params_for_input = params_->getProducerParams(in_tv, out_tv);

    // Schedule the output and the index tensors with the index parameters
    scheduleConstrainedTv(out_tv, {scatter_dim}, params_for_index);
    scheduleConstrainedTv(index_tv, {scatter_dim}, params_for_index);
    if (scatter->src()->isA<TensorView>()) {
      auto src_tv = scatter->src()->as<TensorView>();
      scheduleConstrainedTv(src_tv, {scatter_dim}, params_for_index);
    }

    // Schedule the input. Note that it is guaranteed that the input
    // is exclusively used by this scatter op and is not a constrained
    // tensor for another constrained op.
    scheduleConstrainedTv(in_tv, {scatter_dim}, params_for_input);

    // If there's a use of the scatter output, it is the copy op
    // inserted by insertCopies. It is not automatically
    // scheduled as the propagation from the scatter output won't
    // happen because the loop domain of the scatter output is not mapped
    // with its logical domain.
    if (!out_tv->uses().empty()) {
      NVF_ERROR_EQ(out_tv->uses().size(), 1);
      auto use_of_out = out_tv->uses().at(0);
      NVF_ERROR(use_of_out->isA<LoadStoreOp>());
      auto out_of_use_of_out = ir_utils::getTvOutput(use_of_out);
      scheduleConstrainedTv(out_of_use_of_out, {scatter_dim}, params_for_input);
    }

    // Setting the memory type.
    // If either of the input and output needs to be a global memory
    // tensor, both tensors should use global. Otherwise, use shared.
    // Note that the in_tv tensor should never be produced by another
    // scatter since a copy must have been inserted by
    // insertCopies.
    NVF_ERROR(dynamic_cast<ScatterOp*>(in_tv->definition()) == nullptr);
    if (in_tv->isFusionInput() || in_tv->isFusionOutput() ||
        out_tv->isFusionOutput()) {
      in_tv->setMemoryType(MemoryType::Global);
      out_tv->setMemoryType(MemoryType::Global);
    } else {
      in_tv->setMemoryType(MemoryType::Shared);
      out_tv->setMemoryType(MemoryType::Shared);
    }

    scheduleScatterAllocationDomains(scatter);
  }

  // Scatter-specific allocation domain scheduling
  void scheduleScatterAllocationDomains(ScatterOp* scatter) {
    auto in_tv = ir_utils::getTvInput(scatter);
    auto out_tv = ir_utils::getTvOutput(scatter);

    // For now, just use the logical domain.
    if (!in_tv->hasAllocation()) {
      in_tv->setAllocationDomain(in_tv->getLogicalDomain(), true);
    }
    if (!out_tv->hasAllocation()) {
      out_tv->setAllocationDomain(out_tv->getLogicalDomain(), true);
    }
  }

  // In TopKOp, both input and output are constrained tensors
  void handle(TopKOp* topk) override {
    auto topk_dim = topk->dim();
    auto inp_tv = ir_utils::getTvInput(topk);
    auto out_tv = ir_utils::getTvOutput(topk);

    const auto& params_for_input = params_->getProducerParams(inp_tv, out_tv);

    scheduleConstrainedTv(inp_tv, {topk_dim}, params_for_input);

    // The heuristics parameter for the input is also used to schedule
    // the output so that both inputs and outputs have the same number
    // of items per thread
    scheduleConstrainedTv(
        out_tv,
        {topk_dim},
        params_for_input,
        /*support_grouping=*/true);
  }

  void scheduleConstrainedTv(
      TensorView* tv,
      const std::vector<int64_t>& constrained_logical_id_offsets,
      const GreedyParams::TvParams& heuristic_params,
      bool support_grouping = false) {
    NVF_ERROR(!constrained_logical_id_offsets.empty());

    const auto& constrained_loop_id_offsets =
        getDependentLoopIds(tv, constrained_logical_id_offsets);

    // Move the constrained_logical_ids innermost
    std::unordered_map<int64_t, int64_t> old2new;
    for (const auto [i, offset] : enumerate(constrained_loop_id_offsets)) {
      old2new.emplace(offset, i - std::ssize(constrained_loop_id_offsets));
    }
    tv->reorder(old2new);

    // Flatten the constrained ids
    if (constrained_loop_id_offsets.size() > 1) {
      tv->flatten(-std::ssize(constrained_loop_id_offsets), -1);
    }

    const bool has_unconstrained_ids = tv->getLoopDomain().size() > 1;
    int64_t num_constrained_loop_ids = 1;

    const int64_t batch_size = heuristic_params.batch_size;

    if (batch_size > 1) {
      tv->split(-1, batch_size);
      ++num_constrained_loop_ids;
      tv->axis(-2)->parallelize(ParallelType::TIDx);
      // The batch dimension is grouped. This is convenient as grouped
      // iter domains do not manifest as for-loops, and we do not want
      // for-loops for batch dimensions. However, if the batch
      // dimension is a broadcast, it does not make any difference as
      // broadcast IDs do not create for-loops.
      if (support_grouping && !tv->axis(-1)->isBroadcast()) {
        tv->axis(-1)->parallelize(ParallelType::Group);
      }
    } else {
      tv->axis(-1)->parallelize(ParallelType::TIDx);
    }

    // All done if there's no unconstrained ID
    if (!has_unconstrained_ids) {
      return;
    }

    // Scheduling of the unconstrained IDs with BIDx. Currently all
    // tensors are assumed to have exact-mapped IDs for BID in order to
    // avoid grid sync. Reordering is allowed, though TransposeOp is
    // not yet enabled.

    // Accommodate reordered unconstrained IDs
    if (ref_unconstrained_domain_.empty()) {
      ref_unconstrained_domain_ =
          exact_graph_.toGroups(std::vector<IterDomain*>{
              tv->getLoopDomain().begin(),
              tv->getLoopDomain().end() - num_constrained_loop_ids});
    } else {
      std::vector<int64_t> permutation;
      permutation.reserve(ref_unconstrained_domain_.size());
      for (const auto i :
           arange(tv->getLoopDomain().size() - num_constrained_loop_ids)) {
        auto id = tv->getLoopDomain().at(i);
        auto ref_it = std::ranges::find_if(
            ref_unconstrained_domain_,
            [&](const ValGroup& id_group) { return id_group->has(id); });
        NVF_ERROR(
            ref_it != ref_unconstrained_domain_.end(),
            "Failed find matching ID group: ",
            id->toString());
        permutation.push_back(
            std::distance(ref_unconstrained_domain_.begin(), ref_it));
      }
      tv->reorder(permutation);
    }

    tv->flatten(
        0, std::ssize(tv->getLoopDomain()) - 1 - num_constrained_loop_ids);
    tv->axis(0)->parallelize(ParallelType::BIDx);

    // Don't inline constrained IDs. For example, like reduction IDs,
    // argsort'ed IDs should never be inlined into its consumers.
    std::unordered_set<Val*> constrained_logical;
    for (const auto constrained_logical_id_offset :
         constrained_logical_id_offsets) {
      constrained_logical.insert(
          tv->getLogicalDomain().at(constrained_logical_id_offset));
    }

    auto all_constrained_ids = DependencyCheck::getAllValsBetween(
        constrained_logical,
        {tv->getLoopDomain().begin(), tv->getLoopDomain().end()});
    for (const auto loop_id : tv->getLoopDomain()) {
      if (std::ranges::find(all_constrained_ids, loop_id) !=
          all_constrained_ids.end()) {
        uninlinable_ids_.insert(loop_id);
      }
    }
  }

 private:
  const ValGraph& exact_graph_;
  std::unordered_set<IterDomain*>& uninlinable_ids_;
  const GreedyParams* params_ = nullptr;
  ValGroups ref_unconstrained_domain_;
};

// Partition all tensors in a given fusion to disjoint sets using
// constrained tensors as references. Returns a map
// from each tensor to its assigned reference.
//
// The partitioning proceeds bottom-up, traversing from
// constrained tensors to all other tensors. When a tensor is
// reached that hasn't been grouped yet, it is assigned into the
// reference's subset. If the tensor is already part of a group, its
// original assignment remains unchanged.
//
// The traversal occurs both backward and forward directions, with a
// preference for backward. Currently, this doesn't make any
// difference since reshape is not allowed. However, backward schedule
// propagation can trivially work across reshapes, whereas forward
// propagation requires a reshape to be cancelled.
std::unordered_map<TensorView*, TensorView*> partitionFusion(
    Fusion* fusion,
    const std::unordered_set<TensorView*>& constrained_tvs) {
  FusionGuard fg(fusion);

  const auto all_exprs = fusion->exprs();
  const auto all_tvs = fusion->allTvs();

  std::unordered_map<TensorView*, TensorView*> tv_to_constrained_tv_map;

  // Register self and sibling mappings for constrained
  // tensors.
  for (auto tv : constrained_tvs) {
    NVF_ERROR(
        tv_to_constrained_tv_map.emplace(tv, tv).second,
        "Already mapped: ",
        tv->toString());

    if (auto def = tv->definition();
        def != nullptr && def->outputs().size() > 1) {
      for (const auto out_tv :
           ir_utils::filterByType<TensorView>(def->outputs())) {
        if (out_tv == tv) {
          continue;
        }
        NVF_ERROR(
            tv_to_constrained_tv_map.emplace(out_tv, tv).second,
            "Already mapped: ",
            tv->toString());
      }
    }
  }

  // The inputs of a constrained op may need to be grouped together for
  // consistent scheduling.
  for (auto expr : all_exprs) {
    // Put all inputs of these ops together with the output. This is
    // not strictly required unless grouped but enforced for
    // simplicity. Inputs of ScatterOp and TopKOp are scheduled
    // separately from the outputs, so they are not included here.
    if (expr->isOneOf<ArgsortOp, ScanOp>()) {
      auto out_tv = ir_utils::getTvOutput(expr);
      NVF_ERROR(
          tv_to_constrained_tv_map.contains(out_tv),
          "Expected to be included in the map but not found: ",
          out_tv->toString());
      for (auto inp_tv : ir_utils::filterByType<TensorView>(expr->inputs())) {
        NVF_ERROR(
            tv_to_constrained_tv_map.emplace(inp_tv, out_tv).second,
            "Already mapped: ",
            inp_tv->toString());
      }
    }
  }

  // Propagate source reference through a given expr. Returns true if
  // propagation is indeed done.
  auto propagateThroughExpr = [&](Expr* expr, Direction dir) -> bool {
    if (!ir_utils::isTvOp(expr)) {
      return false;
    }

    // Find a reference to propagate. If dir is Forward, the reference
    // of the first producer tensor with a reference is used as the
    // reference of this expr. Similarly, if dir is Backward, the reference
    // of the first consumer tensor with a reference is used.
    //
    // When multiple producers or consumers have different
    // references, the reference of the first producer or consumer is
    // propagated.
    TensorView* ref_to_propagate = nullptr;
    const auto& src_vals =
        dir == Direction::Forward ? expr->inputs() : expr->outputs();
    const auto& dst_vals =
        dir == Direction::Forward ? expr->outputs() : expr->inputs();

    auto src_with_ref_it = std::ranges::find_if(src_vals, [&](Val* src) {
      return src->isA<TensorView>() &&
          tv_to_constrained_tv_map.contains(src->as<TensorView>());
    });
    if (src_with_ref_it != src_vals.end()) {
      ref_to_propagate =
          tv_to_constrained_tv_map.at((*src_with_ref_it)->as<TensorView>());
    }

    // No reference to propagate is found
    if (ref_to_propagate == nullptr) {
      return false;
    }

    bool updated = false;

    for (auto dst_tv : ir_utils::filterByType<TensorView>(dst_vals)) {
      // If already set, don't overwrite. If not, propagate the output
      // reference if found.
      if (tv_to_constrained_tv_map.contains(dst_tv)) {
        continue;
      } else {
        NVF_ERROR(
            tv_to_constrained_tv_map.emplace(dst_tv, ref_to_propagate).second,
            "Trying to propagate reference multiple times to: ",
            dst_tv->toString());
        updated = true;
      }
    }

    return updated;
  };

  // Backward propagation across the fusion. Repeat until all
  // expressions are visited.
  auto propagate_backward = [&]() -> bool {
    bool updated = false;
    for (auto expr : all_exprs | std::views::reverse) {
      if (tv_to_constrained_tv_map.size() == all_tvs.size()) {
        return updated;
      }
      if (propagateThroughExpr(expr, Direction::Backward)) {
        updated = true;
      }
    }
    return updated;
  };

  // Forward propagation across the fusion. Unlike the backward prop,
  // immediately terminate once a propagation is done. This is for
  // prioritizing backward propagation.
  auto propagate_forward = [&]() -> bool {
    if (tv_to_constrained_tv_map.size() == all_tvs.size()) {
      return false;
    }
    for (auto expr : all_exprs) {
      if (propagateThroughExpr(expr, Direction::Forward)) {
        return true;
      }
    }
    return false;
  };

  while (tv_to_constrained_tv_map.size() != all_tvs.size()) {
    // Prioritize backprop
    if (propagate_backward()) {
      continue;
    }

    if (propagate_forward()) {
      continue;
    }

    // No progress made
    break;
  }

  if (tv_to_constrained_tv_map.size() != all_tvs.size()) {
    std::vector<TensorView*> ungrouped_tvs;
    std::ranges::copy_if(
        all_tvs, std::back_inserter(ungrouped_tvs), [&](auto tv) {
          return !tv_to_constrained_tv_map.contains(tv);
        });
    NVF_THROW(
        "Fail to group the following tensors: ",
        toDelimitedString(ungrouped_tvs));
  }

  return tv_to_constrained_tv_map;
}

SyncMap buildSyncMap(Fusion* fusion) {
  FusionInfo fusion_info;
  FusionInfoGuard info_guard(&fusion_info);
  fusion_info.set(std::make_unique<ConcretizedBroadcastDomains>(fusion));
  fusion_info.set(std::make_unique<PaddedParallelDimensions>(
      collectPaddedParallelDims(fusion)));
  fusion_info.set(std::make_unique<IdModel>(fusion, /*build_graphs=*/true));
  fusion_info.set(std::make_unique<ComputeAtMap>(fusion));
  fusion_info.set(std::make_unique<ParallelDimensionMap>(fusion));
  fusion_info.set(std::make_unique<ThreadPredicateMap>(fusion));
  return SyncMap(fusion, /*error_on_failure=*/false);
}

} // namespace

bool GreedyScheduler::canScheduleCompileTime(Fusion* fusion) {
  if (!isOptionEnabled(EnableOption::GreedyScheduler)) {
    scheduler_debug_utils::canScheduleRejectReason(
        SchedulerType::Greedy, "Not enabled");
    return false;
  }

  auto constrained_ops = getAllConstrainedOps(fusion);
  if (constrained_ops.empty()) {
    scheduler_debug_utils::canScheduleRejectReason(
        SchedulerType::Greedy, "No constrained op found");
    return false;
  }

  IdModel id_model(fusion);
  const auto& exact_graph = id_model.buildExactGraph();

  return CompileTimeChecker::run(fusion, exact_graph);
}

bool GreedyScheduler::canScheduleRunTime(
    Fusion* fusion,
    SchedulerRuntimeInfo& runtime_info,
    HeuristicDataCache* data_cache) {
  return RunTimeChecker::run(fusion, runtime_info);
}

std::unique_ptr<HeuristicParams> GreedyScheduler::computeHeuristics(
    Fusion* fusion,
    SchedulerRuntimeInfo& runtime_info,
    HeuristicDataCache* data_cache) {
  FUSER_PERF_SCOPE("GreedyScheduler::computeHeuristics");

  // TODO: use data_cache

  auto params = HeuristicsBuilder::run(fusion, runtime_info);

  if (isDebugDumpEnabled(DebugDumpOption::SchedulerDebug)) {
    debug() << params->toString() << std::endl;
  }

  return params;
}

void GreedyScheduler::schedule(Fusion* fusion, const HeuristicParams* params) {
  FUSER_PERF_SCOPE("GreedyScheduler::schedule");
  FusionGuard fg(fusion);
  // Heuristics are copied as they may need to be updated
  GreedyParams greedy_params = *dynamic_cast<const GreedyParams*>(params);

  scheduler_utils::clearMemorySpace(fusion);

  // Cache inputs
  auto cached_inputs = scheduler_utils::cacheInputs(fusion, true);
  // Transfer the heuristics parameters for the inputs to their caches
  for (const auto& [cache, original] : cached_inputs) {
    for (const auto& consumer : ir_utils::consumerTvsOf(cache)) {
      greedy_params.transferProducerParams(
          fusion->inputs().at(original)->as<TensorView>(),
          consumer,
          cache,
          consumer);
    }
  }
  // Cache and fork outputs
  auto cached_outputs = scheduler_utils::cacheAndForkOutputs(fusion, true);
  for (const auto& [cache, original] : cached_outputs) {
    greedy_params.transferConsumerParams(
        fusion->outputs().at(original)->as<TensorView>(), cache);
    for (const auto& producer : ir_utils::producerTvsOf(cache)) {
      greedy_params.transferProducerParams(
          producer,
          fusion->outputs().at(original)->as<TensorView>(),
          producer,
          cache);
    }
  }

  propagateReshape(fusion);

  insertCopies(fusion, greedy_params);

  std::vector<TensorView*> constrained_tvs = getAllConstrainedTvs(fusion);

  IdModel id_model(fusion);
  const auto& exact_graph = id_model.buildExactGraph();

  std::unordered_set<IterDomain*> uninlinable_ids;

  // Schedule all constrained tensors
  ConstrainedOpScheduler::run(
      fusion, exact_graph, uninlinable_ids, &greedy_params);

  // Need to fetch constrained tensors again as cacheAfter/Before may be used
  constrained_tvs = getAllConstrainedTvs(fusion);

  // Partition the fusion
  auto tv_to_ref_map =
      partitionFusion(fusion, {constrained_tvs.begin(), constrained_tvs.end()});
  if (isDebugDumpEnabled(DebugDumpOption::SchedulerVerbose)) {
    std::unordered_map<TensorView*, std::unordered_set<TensorView*>> ref_to_tvs;
    for (const auto& [tv, ref] : tv_to_ref_map) {
      ref_to_tvs[ref].insert(tv);
    }
    scheduler_debug_utils::log("[Greedy scheduler] partitioned fusion:");
    for (const auto& [ref, tvs] : ref_to_tvs) {
      scheduler_debug_utils::log(
          "Ref: ", ref->toString(), ": ", toDelimitedString(tvs));
    }
  }

  // Propagate the schedule of each constrained tv to its disjoint set
  for (auto constrained_tv : constrained_tvs) {
    std::unordered_set<TensorView*> tvs_to_transform;
    for (const auto& [tv, ref] : tv_to_ref_map) {
      if (ref == constrained_tv) {
        tvs_to_transform.insert(tv);
      }
    }

    SetSelector selector(tvs_to_transform);
    MaxLogicalDomainInfoSpanningTree tree(constrained_tv, &selector);
    TransformPropagator tp(constrained_tv);
    tree.traverse(&tp);

    scheduler_utils::parallelizeAllLike(
        constrained_tv,
        -1,
        {tvs_to_transform.begin(), tvs_to_transform.end()},
        {ParallelType::BIDx, ParallelType::TIDx});
  }

  inlineMost(uninlinable_ids);

  // Resolve conflicts. Find conflicting producer-consumer pairs
  // and insert memory promotion

  VectorOfUniqueEntries<TensorView*> tvs_to_upload_to_smem;
  const auto sync_map = buildSyncMap(fusion);

  for (const auto& [tv, pt_map] : sync_map.map()) {
    NVF_ERROR(
        !pt_map.hasBID(),
        "Grid sync not expected: ",
        tv->toString(),
        ", ",
        pt_map.toString());
    // Nothing to do if already in Global or Shared
    if (tv->getMemoryType() == MemoryType::Global ||
        tv->getMemoryType() == MemoryType::Shared) {
      continue;
    }
    tvs_to_upload_to_smem.pushBack(tv);
  }

  for (const auto& tv : tvs_to_upload_to_smem) {
    // Create a copy of this tensor on shared memory
    auto no_reduction_logical_domain =
        TensorDomain::noReductions(tv->getLogicalDomain());
    std::vector<IterDomain*> new_logical_domain;
    new_logical_domain.reserve(no_reduction_logical_domain.size());
    for (const auto& dom : no_reduction_logical_domain) {
      new_logical_domain.push_back(dom->cloneWithoutRFactor());
    }
    auto copy = IrBuilder::create<TensorView>(
        IrBuilder::create<TensorDomain>(
            new_logical_domain,
            TensorDomain::getContiguityFilledWith(new_logical_domain, true)),
        tv->dtype());
    TransformReplay::selfReplay(tv->domain(), copy->domain());
    copy->setMemoryType(MemoryType::Shared);

    // Insert a copy op
    IrBuilder::create<LoadStoreOp>(LoadStoreOpType::Set, copy, tv);

    // Replace use of this tv if it has a conflicting consumer
    const auto& tv_sync_map = sync_map.producerConsumerRawSync().at(tv);
    std::vector<Expr*> uses_to_update;
    std::ranges::copy_if(
        tv->uses(), std::back_inserter(uses_to_update), [&](Expr* use) {
          return std::ranges::any_of(use->outputs(), [&](Val* out) {
            auto* out_tv = dynamic_cast<TensorView*>(out);
            if (out_tv == nullptr) {
              return false;
            }
            auto it = tv_sync_map.find(out_tv);
            return it != tv_sync_map.end() && it->second.hasTID();
          });
        });
    NVF_ERROR(!uses_to_update.empty());

    for (auto tv_use : uses_to_update) {
      ir_utils::replaceValInExprInputs(tv_use, tv, copy);
    }
  }

  // If a new copy op is inserted, inlining positions need to be
  // reset. We could probably fix up only tensors around those newly
  // inserted ones, but here's just a quick approach
  if (!tvs_to_upload_to_smem.empty()) {
    resetInlining(fusion);
    inlineMost(uninlinable_ids);
  }
}

} // namespace nvfuser
