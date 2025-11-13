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
// Ideally, recomputation should be done automatically in TensorView's cloner.
// But I'm hitting #4849 when trying that.
void recomputeTv(const TensorView* tv, IrCloner& ir_cloner) {
  for (Expr* e : StmtSort::getExprsTo(
           {tv->getLoopDomain().begin(), tv->getLoopDomain().end()})) {
    ir_cloner.clone(e);
  }
  for (IterDomain* id : tv->getLoopDomain()) {
    for (Expr* e : StmtSort::getExprsTo({id->extent()})) {
      ir_cloner.clone(e);
    }
  }
}

void recomputeOutputTvs(Expr* e, IrCloner& ir_cloner) {
  for (auto* out : ir_utils::filterByType<TensorView>(e->outputs())) {
    recomputeTv(out, ir_cloner);
  }
}

// Finds the stream-parallelized IterDomain in the loop domain of a TensorView,
// or nullptr if not found.  This is different from `getShardedIterDomain(tv,
// ParallelType::Stream)`, which searches the allocation domain.  Consider
// unifying them into one function with an extra DomainType parameter.
IterDomain* findStreamIterDomain(TensorView* tv) {
  const std::vector<IterDomain*>& loop = tv->getLoopDomain();
  // FinalizeMultideviceDomains pass puts the stream IterDomain to the
  // front.
  if (!loop.empty() && loop.front()->isStream()) {
    return loop.front();
  }
  return nullptr;
}

// Finds the stream IterDomain in the outputs of a segment.
IterDomain* findStreamIterDomain(const std::vector<Val*>& outs) {
  for (auto* out : ir_utils::filterByType<TensorView>(outs)) {
    if (auto* stream_id = findStreamIterDomain(out)) {
      return stream_id;
    }
  }
  return nullptr;
}

void lowerSegment(
    const SegmentedGroup& group,
    const AliasInfoMap& aliases,
    const LaunchParams& launch_params,
    hir::HostIrContainer& hic,
    IrCloner& ir_cloner) {
  switch (group.schedulerType()) {
    case SchedulerType::Communication: {
      auto device_id = Communicator::getInstance().deviceId();
      NVF_ERROR_EQ(
          group.exprs().size(),
          1,
          "Communication segments must contain only one Expr.");
      Expr* e = group.exprs().front();
      Expr* e_clone = ir_cloner.clone(e);
      recomputeOutputTvs(e, ir_cloner);

      for (auto* c : convertSingleOpToCommunication(e_clone, device_id)) {
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
          hic.pushBackTopLevelExprs(allocate);
        }
        hic.pushBackTopLevelExprs(communication);
        auto wait = IrBuilder::create<hir::Wait>(communication);
        hic.pushBackTopLevelExprs(wait);
      }
      break;
    }
    case SchedulerType::ExprEval: {
      // Pseudocode:
      // clang-format off
      // ```
      // clone all expressions and store the copies to a list
      //
      // if no expressions are stream parallelized:
      //   append the list to the top level
      //   return
      //
      // create a new, empty for loop
      // for each cloned expression:
      //   for each input TensorView of that expression:
      //     if it's allocated outside the loop:
      //       shard it by stream
      //   for each output TensorView of that expression:
      //     if it needs to be allocated outside the loop:
      //       create an Allocate before the for loop
      //       shard it by stream
      //   add the cloned expression to the loop body with the maybe-sharded inputs and outputs
      // ```
      // clang-format on
      std::vector<Expr*> cloned_exprs;
      cloned_exprs.reserve(group.exprs().size());
      for (Expr* e : group.stablyOrderedExprs()) {
        auto* e_clone = ir_cloner.clone(e);
        recomputeOutputTvs(e, ir_cloner);
        cloned_exprs.push_back(e_clone);
      }

      std::vector<Val*> cloned_outs = ir_cloner.clone(group.outputs());
      // All expressions in the group are expected to be stream parallelized in
      // the same way. So it's safe to find the stream IterDomain from any of
      // them.  Ideally, loop domains should be tied to expressions not
      // TensorViews.
      IterDomain* stream_id = findStreamIterDomain(cloned_outs);
      if (stream_id == nullptr) {
        for (Expr* e : cloned_exprs) {
          hic.pushBackTopLevelExprs(e);
        }
        break;
      }

      auto* for_loop = hir::ForLoop::createFromIterDomain(stream_id);
      auto top_level_insertion_point = hic.pushBackTopLevelExprs(for_loop);

      std::unordered_map<Val*, Val*> replacement_map;
      for (Expr* e : cloned_exprs) {
        for (auto* in : ir_utils::filterByType<TensorView>(e->inputs())) {
          if (findStreamIterDomain(in) != nullptr &&
              getShardedIterDomain(in, ParallelType::Stream) == nullptr) {
            auto [i, inserted] = replacement_map.try_emplace(
                in, hir::shardByStream(in, for_loop->index()));
            if (inserted) {
              for_loop->body().push_back(i->second->definition());
            }
          }
        }

        for (auto* out : ir_utils::filterByType<TensorView>(e->outputs())) {
          if (getShardedIterDomain(out, ParallelType::Stream) == nullptr) {
            auto* allocate =
                IrBuilder::create<kir::Allocate>(out, MemoryType::Global);
            hic.insertExprBefore(top_level_insertion_point, allocate);
            // Loop is stream parallelized but allocation is not. Therefore,
            // `out` should be allocated outside the loop.
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
        for_loop->body().push_back(new_e);
      }
      break;
    }
    default: {
      const int group_id = group.groupId();

      // Copy the input/output TensorViews to the container.
      auto cloned_ins = ir_cloner.clone(group.inputs());
      auto cloned_outs = ir_cloner.clone(group.outputs());
      for (auto* out : ir_utils::filterByType<TensorView>(group.outputs())) {
        recomputeTv(out, ir_cloner);
      }

      // Allocate the output TensorViews.
      for (auto* out : cloned_outs) {
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
        hic.pushBackTopLevelExprs(allocate);
      }

      // Add the LaunchKernel instruction.
      KernelExecutor& ke = hic.getKernelExecutor(group_id);
      // Needed for KernelExecutor. Should be removed once #4927 is fixed.
      auto* cache_id =
          IrBuilder::create<NamedScalar>("cacheId", DataType::UInt64);
      auto launch_kernel = IrBuilder::create<hir::LaunchKernel>(
          group_id,
          launch_params,
          ke.compiledKernel()->compileParams(),
          cloned_ins,
          cloned_outs,
          cache_id);
      hic.pushBackTopLevelExprs(launch_kernel);
    }
  } // switch
} // lowerSegment
} // namespace

std::unique_ptr<hir::HostIrContainer> lowerSegmentedFusionToHostIr(
    const SegmentedFusion& segmented_fusion,
    const std::vector<LaunchParams>& launch_params_per_segment,
    std::vector<std::unique_ptr<ExecutorAbstract>>& executors) {
  auto hic = std::make_unique<hir::HostIrContainer>();

  IrCloner ir_cloner(hic.get());
  FusionGuard::setCurFusion(hic.get());

  for (const Val* in : segmented_fusion.inputs()) {
    hic->addInput(ir_cloner.clone(in));
    if (auto* tv = in->as<TensorView>()) {
      recomputeTv(tv, ir_cloner);
    }
  }

  for (auto& executor : executors) {
    if (executor == nullptr) {
      continue;
    }
    auto* ke = executor.release()->as<KernelExecutor>();
    hic->addKernelExecutor(std::unique_ptr<KernelExecutor>(ke));
  }

  for (SegmentedGroup* group :
       prepareRuntimeOrder(segmented_fusion).group_run_order) {
    lowerSegment(
        *group,
        segmented_fusion.completeFusion()->getOutputAliases(),
        launch_params_per_segment.at(group->groupId()),
        *hic,
        ir_cloner);
  }

  for (const Val* out : segmented_fusion.outputs()) {
    hic->addOutput(ir_cloner.clone(out));
  }

  hir_pass::InsertDeallocations().runPass(hic.get());

  return hic;
}

} // namespace nvfuser
