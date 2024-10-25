// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#include <ATen/cuda/CUDAContext.h>
#include <device_lower/utils.h>
#include <fusion_segmenter.h>
#include <host_ir/container.h>
#include <host_ir/host_ir.h>
#include <ir/builder.h>
#include <ir/utils.h>
#include <multidevice/device_mesh.h>
#include <multidevice/executor.h>
#include <multidevice/lower_communication.h>
#include <multidevice/utils.h>
#include <preseg_passes/insert_reshardings.h>
#include <preseg_passes/make_resharding_contiguous.h>
#include <preseg_passes/propagate_shardings.h>
#include <preseg_passes/reorder_sharded_axis.h>
#include <runtime/allocations.h>
#include <runtime/fusion_kernel_runtime.h>

namespace nvfuser {

namespace {

// returns a copied fusion where the original outputs have been replaced by
// the ones given as argument
std::unique_ptr<Fusion> copyFusionAndChangeOutputs(
    Fusion* fusion,
    const std::vector<Val*>& outputs) {
  std::unique_ptr<Fusion> fusion_copy = std::make_unique<Fusion>();
  std::unordered_map<Val*, Val*> copy_to_original_map;
  auto original_to_copy_cloner = Fusion::copy(fusion, fusion_copy.get());

  auto original_outputs = fusion_copy->outputs();

  // Remove original outputs
  std::for_each(
      original_outputs.begin(), original_outputs.end(), [&](auto& output) {
        fusion_copy->removeOutput(output);
      });

  // Add new outputs
  std::for_each(outputs.begin(), outputs.end(), [&](Val* const& output) {
    fusion_copy->addOutput(original_to_copy_cloner.clone(output));
  });

  return fusion_copy;
}

} // namespace

MultiDeviceExecutor::MultiDeviceExecutor(
    std::unique_ptr<Fusion> fusion,
    Communicator& comm,
    hir::HostIrEvaluatorParams params)
    : comm_(comm), complete_fusion_(std::move(fusion)) {
  // Sharding PreSegmenter passes.
  // Note: passes run before PreSegmenter optimization passes.
  preseg_passes::OptimizationPass<
      preseg_passes::PropagateShardingsPass>::runPass(complete_fusion_.get());
  preseg_passes::OptimizationPass<
      preseg_passes::InsertReshardingsPass>::runPass(complete_fusion_.get());
  preseg_passes::OptimizationPass<
      preseg_passes::ReorderShardedAxisPass>::runPass(complete_fusion_.get());
  preseg_passes::OptimizationPass<preseg_passes::MakeReshardingContiguousPass>::
      runPass(complete_fusion_.get());

  // Performs segmentation at the inter-device communications
  // Each SegmentedGroup represents a pipeline's stage, and can be either
  // 1) a Fusion which doesn't involve inter-device communication
  // 2) a Fusion comprised of one Expr, representing inter-device communication
  SegmentCandidateFinderOptions options{
      .run_translate_welford = false,
      .run_combine_reductions = false,
      .run_herrmann_merge = true,
      .run_final_merge = true,
      .only_segment_resharding_exprs = true};
  std::unique_ptr<SegmentedFusion> staged_fusion =
      SegmentCandidateFinder::segment(
          std::make_unique<Fusion>(*complete_fusion_), nullptr, options);
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
    std::vector<Expr*> host_exprs;
    NVF_ERROR(!group->exprs().empty(), "invalid segmentation");
    if (involvedDevices(group->exprs().at(0)).count(comm_.deviceId()) == 0) {
      continue;
    }
    const bool is_resharding = std::any_of(
        group->exprs().begin(), group->exprs().end(), [](auto expr) {
          return isResharding(expr);
        });
    if (!is_resharding) {
      auto host_unit = IrBuilder::create<hir::HostUnit>(
          staged_fusion->makeFusion(group).second);
      auto post_on_stream = IrBuilder::create<hir::PostOnStream>(
          host_unit, clone(group->inputs()), clone(group->outputs()));
      hic->pushBackTopLevelExprs(post_on_stream);
    } else {
      NVF_ERROR(
          group->exprs().size() == 1,
          "Communication segments must contain only one Expr");
      std::vector<Communication*> communications =
          lowerCommunication(ir_cloner.clone(group->exprs().at(0)));
      for (Communication* communication : communications) {
        auto wait = IrBuilder::create<hir::Wait>(communication);
        hic->pushBackTopLevelExprs(communication);
        hic->pushBackTopLevelExprs(wait);
      }
    }
  }
  for (auto input : staged_fusion->inputs()) {
    hic->addInput(ir_cloner.clone(input));
  }
  for (auto output : staged_fusion->outputs()) {
    hic->addOutput(ir_cloner.clone(output));
  }

  // Create the HostIrEvaluator representing the host program
  host_ir_executor_ =
      std::make_unique<hir::HostIrEvaluator>(std::move(hic), &comm, params);

  // Allocator setup
  // vals_to_allocate_ stores the tensors that need to be allocated at runtime,
  // which correspond to the destination buffers of interdevice communications.
  // TODO: reuse allocated buffers and support inplace collectives
  // TODO: handle allocation as Host Ir
  for (SegmentedGroup* group : staged_fusion->groups()) {
    if (isResharding(group->exprs().at(0))) {
      NVF_ERROR(group->exprs().at(0)->outputs().size() == 1);
      auto val = group->exprs().at(0)->outputs().at(0);
      NVF_ERROR(val->isA<TensorView>());
      auto tv = val->as<TensorView>();
      NVF_ERROR(tv->hasDeviceMesh());
      if (tv->getDeviceMesh().has(comm_.deviceId())) {
        vals_to_allocate_.push_back(val);
      }
    }
  }
  allocator_fusion_ = copyFusionAndChangeOutputs(
      staged_fusion->completeFusion(), vals_to_allocate_);
  vals_to_allocate_ = clone(vals_to_allocate_);
}

std::vector<at::Tensor> MultiDeviceExecutor::runWithInput(
    const std::vector<c10::IValue>& inputs) {
  // make sure the communicator can run the Fusion (e.g. there is enough GPUs,
  // etc)
  auto error_msg = validate();
  NVF_ERROR(error_msg.empty(), error_msg);

  // Stores concrete computed values,
  std::unordered_map<Val*, c10::IValue> val_to_IValue;

  // Make sure inputs align at global boundary.
  NVF_ERROR(
      inputs.size() == host_ir_executor_->inputs().size(),
      "Wrong number of inputs");
  // process input values:
  for (auto input_idx : c10::irange(inputs.size())) {
    val_to_IValue[host_ir_executor_->inputs().at(input_idx)] =
        inputs.at(input_idx);
  }

  auto allocations =
      allocOutputSpace(inputs, allocator_fusion_.get(), comm()->device());
  NVF_ERROR(vals_to_allocate_.size() == allocations.size());
  for (auto i : c10::irange(allocations.size())) {
    val_to_IValue[vals_to_allocate_.at(i)] = allocations.at(i);
  }

  return host_ir_executor_->runWithInput(val_to_IValue);
}

std::string MultiDeviceExecutor::validate() const {
  if (!comm_.is_available()) {
    return "distributed configuration required";
  }

  if (requestedNumberOfDevices(completeFusion()) > comm_.size()) {
    return "the pipeline requests " +
        std::to_string(requestedNumberOfDevices(completeFusion())) +
        " GPUs to run, but there are only " + std::to_string(comm_.size()) +
        " ranks in the communicator";
  }

  if (comm_.size() > at::cuda::getNumGPUs()) {
    return std::to_string(comm_.local_size()) +
        " processes are spawn on the node but only " +
        std::to_string(at::cuda::getNumGPUs()) + " GPUs are available";
  }

  return "";
}

std::ostream& MultiDeviceExecutor::print(std::ostream& os) {
  return host_ir_executor_->print(os);
}

} // namespace nvfuser
