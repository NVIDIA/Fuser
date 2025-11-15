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
#include <host_ir/pass/insert_deallocations.h>
#include <multidevice/utils.h>
#include <runtime/executor_abstract.h>

namespace nvfuser {

namespace {

struct LoopInfo {
  hir::ForLoop* loop;

  // The Scope that owns `loop`. It's one level outer than `loop`'s body scope.
  Scope* parent_scope;

  // The iterator that points to `loop`. This way, we can insert instructions,
  // e.g. Allocate, right before the loop.
  Scope::Iterator parent_insertion_point;
};

std::ostream& operator<<(std::ostream& os, const LoopInfo& loop_info) {
  os << loop_info.loop->toInlineString();
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
const std::vector<IterDomain*>& findReferenceLoopDomain(
    const SegmentedGroup& group) {
  TensorView* reference_tv = nullptr;
  int max_parallel_count = -1;
  for (auto* expr : group.exprs()) {
    for (auto* tv : ir_utils::filterByType<TensorView>(expr->outputs())) {
      auto loop_domain = tv->getLoopDomain();
      int parallel_count = 0;
      for (auto* id : loop_domain) {
        if (id->isParallelized()) {
          parallel_count++;
        }
      }
      if (parallel_count > max_parallel_count) {
        max_parallel_count = parallel_count;
        reference_tv = tv;
      }
    }
  }
  NVF_ERROR(reference_tv != nullptr);
  return reference_tv->getLoopDomain();
}

void lowerSegment(
    const SegmentedGroup& group,
    const AliasInfoMap& aliases,
    const LaunchParams& launch_params,
    hir::HostIrContainer& hic,
    LoopNest& loop_nest,
    IrCloner& ir_cloner) {
  switch (group.schedulerType()) {
    case SchedulerType::Communication: {
      auto device_id = Communicator::getInstance().deviceId();
      NVF_ERROR_EQ(
          group.exprs().size(),
          1,
          "Communication segments must contain only one Expr.");
      // If a value is already cloned, IrCloner::clone returns the cloned value
      // without cloning the value again.
      Expr* e = ir_cloner.clone(group.exprs().front());

      for (auto* c : convertSingleOpToCommunication(e, device_id)) {
        NVF_ERROR(
            c->isA<Communication>(),
            "Exprs in a Communication group should be Communication: ",
            c);
        // Allocate the recv buffers of communications
        auto* communication = c->as<Communication>();
        TensorView* tv = communication->out();
        if (tv->getDeviceMesh().has(device_id)) {
          auto* allocate =
              IrBuilder::create<kir::Allocate>(tv, MemoryType::Global);
          // TODO: allocation may have to go to the top level. See how
          // SchedulerType::ExprEval handles allocations.
          loop_nest.innermostScope().push_back(allocate);
        }
        loop_nest.innermostScope().push_back(communication);
        auto wait = IrBuilder::create<hir::Wait>(communication);
        loop_nest.innermostScope().push_back(wait);
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
          loop_nest.innermostScope().push_back(e);
        }
        break;
      }

      auto [for_loop, parent_scope, parent_insertion_point] =
          loop_nest.innermost();

      std::unordered_map<Val*, Val*> replacement_map;
      for (Expr* e : exprs) {
        for (auto* in : ir_utils::filterByType<TensorView>(e->inputs())) {
          if (getShardedIterDomain(
                  in, ParallelType::Stream, DomainType::kLoop) != nullptr &&
              getShardedIterDomain(
                  in, ParallelType::Stream, DomainType::kAllocation) ==
                  nullptr) {
            auto [i, inserted] = replacement_map.try_emplace(
                in, hir::shardByStream(in, for_loop->index()));
            if (inserted) {
              for_loop->body().push_back(i->second->definition());
            }
          }
        }

        bool output_is_preallocated = false;
        for (auto* out : ir_utils::filterByType<TensorView>(e->outputs())) {
          if (getShardedIterDomain(
                  out, ParallelType::Stream, DomainType::kAllocation) ==
              nullptr) {
            auto* allocate =
                IrBuilder::create<kir::Allocate>(out, MemoryType::Global);
            output_is_preallocated = true;
            parent_scope->insert(parent_insertion_point, allocate);
            // Loop is stream parallelized but allocation is not. Therefore,
            // `out` should be allocated outside the loop.
            //
            // I use try_emplace here so shardByStream is called only when `out`
            // is missing.
            auto [i, inserted] = replacement_map.try_emplace(
                out, hir::shardByStream(out, for_loop->index()));
            NVF_ERROR(inserted);
            for_loop->body().push_back(i->second->definition());
          }
        }

        std::vector<Val*> new_inputs;
        std::transform(
            e->inputs().begin(),
            e->inputs().end(),
            std::back_inserter(new_inputs),
            [&replacement_map](Val* input) {
              return getOrDefault(replacement_map, input, input);
            });
        std::vector<Val*> new_outputs;
        std::transform(
            e->outputs().begin(),
            e->outputs().end(),
            std::back_inserter(new_outputs),
            [&replacement_map](Val* output) {
              return getOrDefault(replacement_map, output, output);
            });
        Expr* new_e = e->newObjectFunc()(
            e->container(), new_inputs, new_outputs, e->attributes());
        if (output_is_preallocated) {
          new_e = new_e->withOutputPreallocated();
        }
        for_loop->body().push_back(new_e);
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
        auto* tv = out->as<TensorView>();
        auto* allocate =
            IrBuilder::create<kir::Allocate>(tv, MemoryType::Global);
        loop_nest.innermostScope().push_back(allocate);
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
      loop_nest.innermostScope().push_back(launch_kernel);
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
        findReferenceLoopDomain(*group);
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

  hir_pass::InsertDeallocations().runPass(hic.get());

  return hic;
}

} // namespace nvfuser
