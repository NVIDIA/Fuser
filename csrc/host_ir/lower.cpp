// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#include <device_lower/utils.h>
#include <host_ir/lower.h>
#include <host_ir/lower_to_communication.h>
#include <host_ir/pass/convert_op_to_communication.h>
#include <host_ir/pass/stream_parallel_type.h>
#include <ir/all_nodes.h>
#include <ir/builder.h>
#include <ir/interface_nodes.h>
#include <ir/iostream.h>
#include <multidevice/device_mesh.h>
#include <multidevice/utils.h>
#include <ops/all_ops.h>
#include <ops/utils.h>
#include <preseg_passes/insert_reshardings.h>
#include <preseg_passes/make_resharding_contiguous.h>
#include <preseg_passes/propagate_shardings.h>
#include <preseg_passes/reorder_sharded_axis.h>
#include <runtime/fusion_kernel_runtime.h>
#include <limits>

namespace nvfuser {

bool HostIrLower::canLower(Expr* expr, bool ignore_inner_resharding) {
  if (!isResharding(expr)) {
    return true;
  }
  if (!ir_utils::isTvOp(expr)) {
    return false;
  }
  if (auto* reduction = dynamic_cast<ReductionOp*>(expr)) {
    if (!ignore_inner_resharding && isInnerResharding(expr)) {
      return false;
    }
    auto in = reduction->in()->as<TensorView>();
    auto out = reduction->out()->as<TensorView>();
    // get the reduced axis
    std::vector<IterDomain*> reduction_axis;
    std::copy_if(
        out->getLogicalDomain().begin(),
        out->getLogicalDomain().end(),
        std::back_inserter(reduction_axis),
        [](IterDomain* id) { return id->isReduction(); });
    // check whether the reduction involves only one axis
    if (reduction_axis.size() != 1) {
      return false;
    }
    // We check whether the reduced axis is sharded on the input
    const auto c2p_map =
        PairwiseLogicalDomainMap(in, out).mapConsumerToProducer();
    auto c2p_map_it = c2p_map.find(reduction_axis.at(0));
    return c2p_map_it != c2p_map.end() && c2p_map_it->second->isDeviceDim();
  } else if (auto* ldst = dynamic_cast<LoadStoreOp*>(expr)) {
    if (!ignore_inner_resharding && isInnerResharding(expr)) {
      return false;
    }
    return ldst->as<LoadStoreOp>()->opType() == LoadStoreOpType::Set;
  } else if (auto* matmul = dynamic_cast<MatmulOp*>(expr)) {
    // For now we only support out = matmul(a,b) when b, out are fully
    // replicated, a is sharded on axis 1, and out i stream-parallelized on axis
    // 0.
    return !isSharded(matmul->inB()) && !isSharded(matmul->out()) &&
        matmul->inA()->axis(0)->getParallelType() == ParallelType::Serial &&
        getShardedLogicalAxis(matmul->inA(), ParallelType::DIDx) == 1 &&
        matmul->out()->axis(0)->getParallelType() == ParallelType::Stream;
  } else if (auto* linear = dynamic_cast<LinearOp*>(expr)) {
    // For now we only support out = linear(a, b, bias) when b, bias, and out
    // are fully replicated, a is sharded on axis 1, and out i
    // stream-parallelized on axis 0.
    auto* a = linear->inA()->as<TensorView>();
    auto* b = linear->inB()->as<TensorView>();
    auto* bias =
        (linear->has_bias() ? linear->bias()->as<TensorView>() : nullptr);
    auto* out = linear->out()->as<TensorView>();
    return !isSharded(b) && !(linear->has_bias() && isSharded(bias)) &&
        !isSharded(out) &&
        a->axis(0)->getParallelType() == ParallelType::Serial &&
        getShardedLogicalAxis(a, ParallelType::DIDx) == 1 &&
        out->axis(0)->getParallelType() == ParallelType::Stream;
  }
  return false;
}

bool HostIrLower::isLowerableAsStandaloneHostOp(Expr* expr) {
  if (expr->isOneOf<
          MatmulOp,
          SliceOp,
          SelectOp,
          LinearOp,
          BinaryOp,
          ReductionOp,
          Communication,
          P2PCommunication>()) {
    return true;
  }

  // Lower as standalone op "set" ops, i.e., LoadStoreOp of "Set" type with no
  // permute
  if (expr->isA<LoadStoreOp>()) {
    auto* load_store = expr->as<LoadStoreOp>();
    if (load_store->opType() == LoadStoreOpType::Set &&
        load_store->out()->isA<TensorView>()) {
      auto* tv = load_store->out()->as<TensorView>();
      // If the output tensor has no root, it means it has no permute
      if (!tv->hasRoot()) {
        return true;
      }
    }
  }

  return false;
}

bool HostIrLower::shouldMergeSegmentedGroups(
    SegmentedGroup* group1,
    SegmentedGroup* group2) {
  for (auto group : {group1, group2}) {
    for (Expr* expr : group->exprs()) {
      if (isLowerableAsStandaloneHostOp(expr)) {
        return false;
      }
    }
  }
  return true;
}

std::unique_ptr<hir::HostIrContainer> HostIrLower::lower(
    std::unique_ptr<Fusion> fusion,
    DeviceIdxType my_device_index) {
  // Sharding PreSegmenter passes.
  // Note: passes run before PreSegmenter optimization passes.
  preseg_passes::OptimizationPass<
      preseg_passes::PropagateShardingsPass>::runPass(fusion.get());
  preseg_passes::OptimizationPass<
      preseg_passes::InsertReshardingsPass>::runPass(fusion.get());
  preseg_passes::OptimizationPass<
      preseg_passes::ReorderShardedAxisPass>::runPass(fusion.get());
  preseg_passes::OptimizationPass<
      preseg_passes::MakeReshardingContiguousPass>::runPass(fusion.get());

  // Performs segmentation at the inter-device communications
  // Each SegmentedGroup represents a pipeline's stage, and can be either
  // 1) a Fusion which doesn't involve inter-device communication
  // 2) a Fusion comprised of one Expr, representing inter-device communication
  SegmentCandidateFinderOptions options{
      .run_translate_welford = false,
      .run_combine_reductions = false,
      .run_herrmann_merge = true,
      .run_final_merge = true,
      .custom_should_merge_groups = &shouldMergeSegmentedGroups};
  std::unique_ptr<SegmentedFusion> staged_fusion =
      SegmentCandidateFinder::segment(
          std::move(fusion), KernelArgumentHolder(), options, true);
  // Infer a topologically ordered traversal of the segmented fusion to
  // determine the order for launching the kernels/comms
  RuntimeWorkSpace workspace;
  prepareRuntimeOrder(staged_fusion.get(), workspace);
  // Create the HostIrContainer representing the host program. Each segment of
  // the segmented fusion will be translated to a HostIR
  auto hic = std::make_unique<hir::HostIrContainer>();
  FusionGuard fg(hic.get());
  IrCloner ir_cloner(hic.get());
  auto clone =
      [&ir_cloner](const std::vector<Val*>& vals) -> std::vector<Val*> {
    std::vector<Val*> cloned_vals(vals.size());
    std::transform(
        vals.begin(), vals.end(), cloned_vals.begin(), [&ir_cloner](Val* val) {
          return ir_cloner.clone(val);
        });
    return cloned_vals;
  };

  for (auto group : workspace.group_run_order) {
    NVF_ERROR(!group->exprs().empty(), "invalid segmentation");
    if (involvedDevices(group->exprs().at(0)).count(my_device_index) == 0) {
      continue;
    }
    // we decide whether to insert the Expr as a standalone op in the
    // HostIRContainer, which will result in using ATen Op to evaluate it --
    // or, alternatively, to wrap them into a PostOnStream(HostUnit(.)) which
    // will result in a kernel code generation.
    if (std::all_of(
            group->exprs().begin(),
            group->exprs().end(),
            isLowerableAsStandaloneHostOp)) {
      NVF_ERROR(
          group->exprs().size() == 1,
          "Expr executed as a standalone op cannot be fused");
      hic->pushBackTopLevelExprs(ir_cloner.clone(group->exprs().at(0)));
    } else {
      auto host_unit = IrBuilder::create<hir::HostUnit>(
          staged_fusion->makeFusion(group).second);
      auto post_on_stream = IrBuilder::create<hir::PostOnStream>(
          host_unit, clone(group->inputs()), clone(group->outputs()));
      hic->pushBackTopLevelExprs(post_on_stream);
    }
  }
  for (auto input : staged_fusion->inputs()) {
    hic->addInput(ir_cloner.clone(input));
  }
  for (auto output : staged_fusion->outputs()) {
    hic->addOutput(ir_cloner.clone(output));
  }

  for (auto tv : hic->allTvs()) {
    // set all host tensors to global memory type. This must be the case by
    // definition of a host tensor, and setting the memory type to global is
    // also required to avoid Allocate HIR nodes to throw
    tv->setMemoryType(MemoryType::Global);
  }

  hir_pass::StreamParallelType().runPass(hic.get());

  hir_pass::ConvertOpToCommunication(params_).runPass(hic.get());

  return hic;
}

} // namespace nvfuser
