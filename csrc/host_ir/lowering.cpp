// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on

#include <fusion_segmenter.h>
#include <host_ir/container.h>
#include <host_ir/lower_to_communication.h>
#include <host_ir/lowering.h>
#include <host_ir/pass/insert_deallocations.h>
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
      Expr* e = group.exprs().at(0);
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
    } break;
    case SchedulerType::ExprEval: {
      // push back segment's exprs into the container as top level
      // expressions
      for (auto* e : group.stablyOrderedExprs()) {
        auto* e_clone = ir_cloner.clone(e);
        recomputeOutputTvs(e, ir_cloner);
        hic.pushBackTopLevelExprs(e_clone);
      }
    } break;
    default:
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
      IterDomain* stream_id = nullptr;
      for (auto* out : cloned_outs) {
        auto* tv = out->as<TensorView>();
        auto i = std::find_if(
            tv->getLoopDomain().begin(),
            tv->getLoopDomain().end(),
            [](IterDomain* id) {
              return id->getParallelType() == ParallelType::Stream;
            });
        if (i == tv->getLoopDomain().end()) {
          continue;
        }
        stream_id = *i;
      }

      KernelExecutor& ke = hic.getKernelExecutor(group_id);
      // Needed for KernelExecutor. Should be removed once #4927 is fixed.
      auto* cache_id =
          IrBuilder::create<NamedScalar>("cacheId", DataType::UInt64);
      if (stream_id == nullptr) {
        auto launch_kernel = IrBuilder::create<hir::LaunchKernel>(
            group_id,
            launch_params,
            ke.compiledKernel()->compileParams(),
            cloned_ins,
            cloned_outs,
            cache_id);
        hic.pushBackTopLevelExprs(launch_kernel);
      } else {
        auto* stream_index = IrBuilder::create<Val>(DataType::Index);
        auto* for_loop =
            hir::createForLoopFromIterDomain(stream_index, stream_id);
        cloned_ins.push_back(stream_index);
        auto launch_kernel = IrBuilder::create<hir::LaunchKernel>(
            group_id,
            launch_params,
            ke.compiledKernel()->compileParams(),
            cloned_ins,
            cloned_outs,
            cache_id);
        for_loop->body().push_back(launch_kernel);
        hic.pushBackTopLevelExprs(for_loop);
      }
  }
}
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
