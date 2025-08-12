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
} // namespace

std::unique_ptr<hir::HostIrContainer> lowerSegmentedFusionToHostIr(
    const SegmentedFusion& segmented_fusion,
    const std::vector<SegmentedGroup*>& group_run_order,
    const std::vector<LaunchParams>& launch_params_per_segment,
    std::vector<std::unique_ptr<ExecutorAbstract>>& executors) {
  auto hic = std::make_unique<hir::HostIrContainer>(group_run_order.size());

  IrCloner ir_cloner(hic.get());
  FusionGuard::setCurFusion(hic.get());
  auto* cache_id = IrBuilder::create<NamedScalar>("cacheId", DataType::UInt64);

  for (const Val* in : segmented_fusion.inputs()) {
    hic->addInput(ir_cloner.clone(in));
    if (auto* tv = in->as<TensorView>()) {
      recomputeTv(tv, ir_cloner);
    }
  }

  for (SegmentedGroup* group : group_run_order) {
    switch (group->schedulerType()) {
      case SchedulerType::Communication: {
        auto deviceid = Communicator::getInstance().deviceId();
        NVF_ERROR_EQ(
            group->exprs().size(),
            1,
            "Communication segments must contain only one Expr.");
        Expr* e = group->exprs().at(0);
        Expr* e_clone = ir_cloner.clone(e);
        recomputeOutputTvs(e, ir_cloner);

        for (auto* c : convertSingleOpToCommunication(e_clone, deviceid)) {
          NVF_ERROR(
              c->isA<Communication>(),
              "Exprs in a Communication group should be Communication: ",
              c);
          // Allocate the recv buffers of communications
          auto* communication = c->as<Communication>();
          TensorView* tv = communication->out();
          if (tv->getDeviceMesh().has(deviceid)) {
            auto* allocate =
                IrBuilder::create<kir::Allocate>(tv, MemoryType::Global);
            hic->pushBackTopLevelExprs(allocate);
          }
          hic->pushBackTopLevelExprs(communication);
          auto wait = IrBuilder::create<hir::Wait>(communication);
          hic->pushBackTopLevelExprs(wait);
        }
      } break;
      case SchedulerType::ExprEval: {
        // push back segment's exprs into the container as top level
        // expressions
        for (auto* e : group->stablyOrderedExprs()) {
          auto* e_clone = ir_cloner.clone(e);
          recomputeOutputTvs(e, ir_cloner);
          hic->pushBackTopLevelExprs(e_clone);
        }
      } break;
      default:
        const int group_id = group->groupId();
        // Add the kernel executor to the container.
        auto* ke = executors.at(group_id).release()->as<KernelExecutor>();
        hic->addKernelExecutor(std::unique_ptr<KernelExecutor>(ke));
        NVF_ERROR_EQ(
            ke->groupId(),
            group_id,
            "The group ID of the kernel executor doesn't match the group ID of "
            "the segment.");
        // Add a LaunchKernel instruction to the container.
        auto in_clone = ir_cloner.clone(group->inputs());
        auto out_clone = ir_cloner.clone(group->outputs());
        for (auto* out : ir_utils::filterByType<TensorView>(group->outputs())) {
          recomputeTv(out, ir_cloner);
        }
        auto launch_kernel = IrBuilder::create<hir::LaunchKernel>(
            group_id,
            launch_params_per_segment.at(group_id),
            ke->compiledKernel()->compileParams(),
            in_clone,
            out_clone,
            cache_id);
        for (auto* val : out_clone) {
          NVF_ERROR(
              val->isA<TensorView>(),
              "Output must be a TensorView but got ",
              val);
          const AliasInfo& alias_info =
              segmented_fusion.completeFusion()->getOutputAlias(val);
          NVF_ERROR(
              alias_info.type == AllocationType::New,
              "Output ",
              val->toString(),
              " must not be an alias, got ",
              alias_info.toString());
          auto* tv = val->as<TensorView>();
          auto* allocate =
              IrBuilder::create<kir::Allocate>(tv, MemoryType::Global);
          hic->pushBackTopLevelExprs(allocate);
        }
        hic->pushBackTopLevelExprs(launch_kernel);
    }
  }

  for (const Val* out : segmented_fusion.outputs()) {
    hic->addOutput(ir_cloner.clone(out));
  }

  hir_pass::InsertDeallocations().runPass(hic.get());

  return hic;
}

} // namespace nvfuser
