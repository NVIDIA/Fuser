// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#include <fusion_segmenter.h>
#include <host_ir/container.h>
#include <host_ir/host_ir.h>
#include <host_ir/lower_to_communication.h>
#include <host_ir/lowering.h>
#include <ir/utils.h>
#include <multidevice/propagation.h>
#include <multidevice/resharding.h>
#include <multidevice/utils.h>
#include <ops/utils.h>
#include <runtime/executor_abstract.h>
#include <transform_replay.h>

namespace nvfuser {

namespace {

struct LoopInfo {
  hir::ForLoop* loop = nullptr;

  // The Scope that owns `loop`. It's one level outer than `loop`'s body scope.
  Scope* parent_scope = nullptr;

  // The iterator that points to `loop`. This way, we can insert instructions,
  // e.g. Allocate, right before the loop.
  Scope::Iterator parent_insertion_point;
};

std::ostream& operator<<(std::ostream& os, const LoopInfo& loop_info) {
  if (loop_info.loop == nullptr) {
    os << "<null>";
  } else {
    os << loop_info.loop->toInlineString();
  }
  return os;
}

class LoopNest {
 public:
  LoopNest(Scope& top_level) : top_level_(top_level) {}

  int64_t size() const {
    return std::ssize(loop_infos_);
  }

  bool empty() const {
    return loop_infos_.empty();
  }

  void closeLoop() {
    NVF_ERROR(!empty());
    loop_infos_.pop_back();
  }

  const LoopInfo& innermost() const {
    NVF_ERROR(!empty());
    return loop_infos_.back();
  }

  // Returns the scope of the innermost for-loop or the top-level scope if the
  // loop nest is empty.
  Scope& innermostScope() const {
    return empty() ? top_level_ : innermost().loop->body();
  }

  hir::ForLoop* openLoop(IterDomain* id) {
    Scope& parent_scope = innermostScope();
    auto* for_loop = hir::ForLoop::createFromIterDomain(id);
    loop_infos_.push_back(
        {for_loop, &parent_scope, parent_scope.push_back(for_loop)});
    return for_loop;
  }

  friend std::ostream& operator<<(std::ostream& os, const LoopNest& loop_nest);

 private:
  std::vector<LoopInfo> loop_infos_;
  Scope& top_level_;
};

std::ostream& operator<<(std::ostream& os, const LoopNest& loop_nest) {
  os << "LoopNest:" << std::endl;
  for (const auto& loop_info : loop_nest.loop_infos_) {
    indent(os, 1) << loop_info << std::endl;
  }
  return os;
}

// Finds the TensorView in the group whose loop domain has the most parallel
// types and returns its loop domain.
const std::vector<IterDomain*>& findMostParallelLoopDomain(
    const SegmentedGroup& group) {
  TensorView* reference = nullptr;
  int max_parallel_count = -1;
  for (Expr* expr : group.exprs()) {
    TensorView* tv = findMostParallelTensorView(
        ir_utils::filterByType<TensorView>(expr->outputs()));
    if (tv == nullptr) {
      continue;
    }
    auto parallel_count = numParallelIterDomains(tv);
    if (parallel_count > max_parallel_count) {
      max_parallel_count = parallel_count;
      reference = tv;
    }
  }
  NVF_ERROR(reference != nullptr, "Can't find any TensorView in ", &group);
  return reference->getLoopDomain();
}

// Returns a new Expr with the inputs and outputs replaced by the replacement
// map. If none of the inputs or outputs are replaced, returns the original
// Expr.
Expr* cloneWithNewOperands(
    Expr* e,
    const std::unordered_map<Val*, Val*>& replacement_map) {
  auto maybe_replace = [&](Val*& x) -> bool {
    Val* new_x = getOrDefault(replacement_map, x);
    if (new_x == nullptr) {
      return false;
    }
    x = new_x;
    return true;
  };

  int64_t replaced = 0;

  std::vector<Val*> new_ins = e->inputs();
  replaced += std::ranges::count_if(new_ins, maybe_replace);

  std::vector<Val*> new_outs = e->outputs();
  replaced += std::ranges::count_if(new_outs, maybe_replace);

  if (replaced == 0) {
    return e;
  }

  return e->newObjectFunc()(e->container(), new_ins, new_outs, e->attributes());
}

void lowerSegment(
    const SegmentedGroup& group,
    const AliasInfoMap& aliases,
    const LaunchParams& launch_params,
    hir::HostIrContainer& hic,
    LoopNest& loop_nest,
    IrCloner& ir_cloner) {
  Scope& innermost_scope = loop_nest.innermostScope();
  LoopInfo innermost;
  if (!loop_nest.empty()) {
    innermost = loop_nest.innermost();
  }

  switch (group.schedulerType()) {
    case SchedulerType::Communication: {
      // We can probably unify the processing of a Communication segment and
      // that of an ExprEval segment. A Communication can only have one output
      // and that output is always pre-allocated, simplifying the logic a bit.
      auto device_id = Communicator::getInstance().deviceId();
      NVF_ERROR_EQ(
          group.exprs().size(),
          1,
          "Communication segments must contain only one Expr.");
      // If a value is already cloned, IrCloner::clone returns the cloned value
      // without cloning the value again.
      Expr* e = ir_cloner.clone(group.exprs().front());

      // TODO: `replacement_map` should be associated with the scope so
      // ShardByStream across segments in the same for-loop can be reused.
      std::unordered_map<Val*, Val*> replacement_map;
      for (Expr* c : convertSingleOpToCommunication(e, device_id)) {
        NVF_ERROR(
            c->isA<Communication>(),
            "Exprs in a Communication group should be Communication: ",
            c);
        auto* communication = c->as<Communication>();
        TensorView* in = communication->in();
        TensorView* out = communication->out();
        if (haveDifferentShardings(
                in,
                DomainType::kAllocation,
                out,
                DomainType::kLoop,
                {ParallelType::Stream})) {
          auto [i, inserted] = replacement_map.try_emplace(
              in,
              hir::shardByStream(in, innermost.loop->index(), communication));
          if (inserted) {
            innermost_scope.push_back(i->second->definition());
          }
        }

        // Allocate the recv buffers of communications
        auto* allocate =
            IrBuilder::create<kir::Allocate>(out, MemoryType::Global);
        if (getShardedIterDomain(
                out, ParallelType::Stream, DomainType::kLoop) != nullptr &&
            getShardedIterDomain(
                out, ParallelType::Stream, DomainType::kAllocation) ==
                nullptr) {
          innermost.parent_scope->insert(
              innermost.parent_insertion_point, allocate);
          auto [i, inserted] = replacement_map.try_emplace(
              out,
              hir::shardByStream(out, innermost.loop->index(), communication));
          NVF_ERROR(inserted, "The input segmented fusion should be SSA.");
          innermost_scope.push_back(i->second->definition());
        } else {
          innermost_scope.push_back(allocate);
        }

        Expr* new_c = cloneWithNewOperands(c, replacement_map);
        innermost_scope.push_back(new_c);

        auto* wait = IrBuilder::create<hir::Wait>(new_c);
        innermost_scope.push_back(wait);
      }
      break;
    }
    case SchedulerType::ExprEval: {
      // Pseudocode:
      // clang-format off
      // ```
      // if this segment isn't inside a loop:
      //   append the list to the top level
      //   return
      //
      // for each expression in the segment:
      //   for each input TensorView of that expression:
      //     if it's allocated outside the loop:
      //       shard it by stream
      //   for each output TensorView of that expression:
      //     if it needs to be allocated outside the loop:
      //       create an Allocate before the for loop
      //       shard it by stream
      //   add the expression to the loop body with the maybe-sharded inputs and outputs
      // ```
      // clang-format on
      const std::vector<Expr*>& exprs =
          ir_cloner.clone(group.stablyOrderedExprs());

      // All expressions in the group are expected to be stream parallelized in
      // the same way. So it's safe to find the stream IterDomain from any of
      // them.  Ideally, loop domains should be tied to expressions not
      // TensorViews.
      if (loop_nest.empty()) {
        for (Expr* e : exprs) {
          innermost_scope.push_back(e);
        }
        break;
      }

      std::unordered_map<Val*, Val*> replacement_map;
      for (Expr* e : exprs) {
        // A loop domain should go with an Expr rather than each individual
        // output TensorView. Before this is fixed, pick the most parallel
        // output TensorView as a proxy.
        TensorView* ref_out = findMostParallelTensorView(
            ir_utils::filterByType<TensorView>(e->outputs()));
        NVF_ERROR(ref_out != nullptr);

        for (auto* in : ir_utils::filterByType<TensorView>(e->inputs())) {
          if (replacement_map.contains(in)) {
            continue;
          }

          // Check whether in's **allocation** and out's loop are sharded on
          // ParallelType::Stream consistently. If not, insert a ShardByStream.
          //
          // Consider the following example:
          // ```
          // in: [m, k]    w: [k, n]   # logical/allocation
          //            |
          //            | matmul
          //            v
          //      out: [m, n]     logical
          //           / \.
          //          s  m/s      loop
          // ```
          // `in` needs to be sharded by stream regardless of its loop domain.
          if (haveDifferentShardings(
                  in,
                  DomainType::kAllocation,
                  ref_out,
                  DomainType::kLoop,
                  {ParallelType::Stream})) {
            TensorView* sharded_in =
                hir::shardByStream(in, innermost.loop->index(), e);
            replacement_map[in] = sharded_in;
            innermost_scope.push_back(sharded_in->definition());
          }
        }

        for (auto* out : ir_utils::filterByType<TensorView>(e->outputs())) {
          NVF_ERROR(
              !replacement_map.contains(out),
              "The input segmented fusion should be SSA.");
          if (getShardedIterDomain(
                  out, ParallelType::Stream, DomainType::kAllocation) ==
              nullptr) {
            auto* allocate =
                IrBuilder::create<kir::Allocate>(out, MemoryType::Global);
            innermost.parent_scope->insert(
                innermost.parent_insertion_point, allocate);
            // Loop is stream parallelized but allocation is not. Therefore,
            // `out` should be allocated outside the loop.
            //
            // I use try_emplace here so shardByStream is called only when `out`
            // is missing.
            TensorView* sharded_out =
                hir::shardByStream(out, innermost.loop->index(), e);
            replacement_map[out] = sharded_out;
            innermost_scope.push_back(sharded_out->definition());
          }
        }

        Expr* new_e = cloneWithNewOperands(e, replacement_map);
        innermost_scope.push_back(new_e);
      }
      break;
    }
    default: {
      std::vector<Val*> ins = ir_cloner.clone(group.inputs());
      std::vector<Val*> outs = ir_cloner.clone(group.outputs());

      // Allocate the output TensorViews.
      for (auto* out : outs) {
        NVF_ERROR(
            out->isA<TensorView>(),
            "Output must be a TensorView but got ",
            out);
        const AliasInfo& alias = aliases.get(out);
        NVF_ERROR_EQ(
            alias.type,
            AllocationType::New,
            "Output ",
            out->toString(),
            " must not be an alias, got ",
            alias);
        auto* allocate =
            IrBuilder::create<kir::Allocate>(out, MemoryType::Global);
        innermost_scope.push_back(allocate);
      }

      // Add the LaunchKernel instruction.
      const int group_id = group.groupId();
      KernelExecutor& ke = hic.getKernelExecutor(group_id);
      // Needed for KernelExecutor. Should be removed once #4927 is fixed.
      auto* cache_id =
          IrBuilder::create<NamedScalar>("cacheId", DataType::UInt64);
      auto launch_kernel = IrBuilder::create<hir::LaunchKernel>(
          group_id,
          launch_params,
          ke.compiledKernel()->compileParams(),
          ins,
          outs,
          cache_id);
      innermost_scope.push_back(launch_kernel);
    }
  } // switch
} // lowerSegment

int64_t computeInlinePosition(
    const std::vector<IterDomain*>& prev_ref_loop,
    const std::vector<IterDomain*>& curr_ref_loop,
    const IdModel& id_model) {
  const auto& exact_graph = id_model.idGraph(IdMappingMode::EXACT);
  int64_t inline_position = 0;
  for (auto [prev_id, curr_id] : zip(prev_ref_loop, curr_ref_loop)) {
    if (prev_id->getParallelType() != curr_id->getParallelType()) {
      break;
    }

    if (!exact_graph.disjointValSets().strictAreMapped(prev_id, curr_id)) {
      break;
    }

    inline_position++;
  }

  return inline_position;
}
} // namespace

std::unique_ptr<hir::HostIrContainer> lowerSegmentedFusionToHostIr(
    const SegmentedFusion& segmented_fusion,
    const std::vector<LaunchParams>& launch_params_per_segment,
    std::vector<std::unique_ptr<ExecutorAbstract>>& executors) {
  auto hic = std::make_unique<hir::HostIrContainer>();
  IrCloner ir_cloner =
      Fusion::copy(segmented_fusion.completeFusion(), hic.get());

  FusionGuard fg(hic.get());

  for (auto& executor : executors) {
    if (executor == nullptr) {
      continue;
    }
    auto* ke = executor.release()->as<KernelExecutor>();
    hic->addKernelExecutor(std::unique_ptr<KernelExecutor>(ke));
  }

  LoopNest loop_nest(hic->topLevel());

  IdModel id_model(segmented_fusion.completeFusion(), /*build_graphs=*/false);
  id_model.buildExactGraph();

  std::vector<IterDomain*> prev_ref_loop;
  for (SegmentedGroup* group :
       prepareRuntimeOrder(segmented_fusion).group_run_order) {
    const std::vector<IterDomain*>& curr_ref_loop =
        findMostParallelLoopDomain(*group);
    const int64_t inline_position =
        computeInlinePosition(prev_ref_loop, curr_ref_loop, id_model);
    while (loop_nest.size() > inline_position) {
      loop_nest.closeLoop();
    }

    while (loop_nest.size() < std::ssize(curr_ref_loop)) {
      IterDomain* ref_loop_id = curr_ref_loop.at(loop_nest.size());
      if (!ref_loop_id->isStream()) {
        break;
      }

      auto* stream_id = ir_cloner.clone(ref_loop_id);
      loop_nest.openLoop(stream_id);
    }

    // TODO(#5524): Consider making a class HostIrLowering so many parameters
    // can be made class members instead of having to be passed around.
    lowerSegment(
        *group,
        segmented_fusion.completeFusion()->getOutputAliases(),
        launch_params_per_segment.at(group->groupId()),
        *hic,
        loop_nest,
        ir_cloner);

    prev_ref_loop = std::move(curr_ref_loop);
  }

  return hic;
}

} // namespace nvfuser
