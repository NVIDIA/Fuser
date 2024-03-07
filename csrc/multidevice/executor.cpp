// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#ifdef NVFUSER_DISTRIBUTED
#include <ATen/cuda/CUDAContext.h>
#include <device_lower/utils.h>
#include <fusion_segmenter.h>
#include <ir/utils.h>
#include <multidevice/device_mesh.h>
#include <multidevice/executor.h>
#include <multidevice/lower_communication.h>
#include <multidevice/utils.h>

#include <chrono>

namespace nvfuser {

namespace {

// returned a copied fusion where the original outputs have been replaced by
// the ones given as argument
std::unique_ptr<Fusion> copyFusionAndChangeOutputs(
    Fusion* fusion,
    const std::vector<Val*>& outputs) {
  std::unique_ptr<Fusion> fusion_copy = std::make_unique<Fusion>();
  std::unordered_map<Val*, Val*> copy_to_original_map;
  auto original_to_copy_cloner = Fusion::copy(fusion, fusion_copy.get());

  auto original_inputs = fusion_copy->inputs();
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

  // for (auto tv : ir_utils::filterByType<TensorView>(fusion_copy->vals())) {
  //   tv->setMemoryType(MemoryType::Global);
  //   for (auto i : c10::irange(tv->domain()->nDims())) {
  //     if (!tv->axis(static_cast<int>(i))->isDeviceDim()) {
  //       tv->axis(static_cast<int>(i))->parallelize(ParallelType::Serial);
  //     }
  //   }
  // }
  return fusion_copy;
}

} // namespace

// TODO: use native allocator instead.
// TODO: reimplement. The implementation here is very naive and wasteful since
// we entirely copy the fusion, change the outputs to be the Vals we want to
// allocate, and call allocOutputSpace which effectively compile and run the
// Fusion. This function creates a potentially important overhead, it needs to
// be reimplemented
void MultiDeviceExecutor::allocateBuffers(
    std::vector<c10::IValue> global_inputs_IValues) {
  if (vals_to_allocate_.empty()) {
    return;
  }

  auto buffers = allocator_->allocOutputSpace(global_inputs_IValues);

  NVF_ERROR(
      buffers.size() == vals_to_allocate_.size(),
      "something went wrong with multidevice allocator");
  for (auto i : c10::irange(buffers.size())) {
    val_to_IValue_[vals_to_allocate_.at(i)] = buffers.at(i);
  }
}

MultiDeviceExecutor::MultiDeviceExecutor(
    std::unique_ptr<Fusion> fusion,
    Communicator& comm,
    bool auto_schedule)
    : comm_(comm), auto_schedule_(auto_schedule) {
  insertReshardings(fusion.get());
  SegmentCandidateFinderOptions options{
      .run_translate_welford = false,
      .run_combine_reductions = false,
      .run_herrmann_merge = true,
      .run_final_merge = true,
      .only_segment_resharding_exprs = true};

  staged_fusion_ =
      SegmentCandidateFinder::segment(std::move(fusion), nullptr, options);

  for (auto group : staged_fusion_->groups()) {
    NVF_ERROR(!group->exprs().empty(), "invalid segmentation");
    is_resharding_[group] = std::any_of(
        group->exprs().begin(), group->exprs().end(), [](auto expr) {
          return isResharding(expr);
        });
    NVF_ERROR(
        !is_resharding_[group] || group->exprs().size() == 1,
        "Communications cannot be fused");
    auto expr = group->exprs().at(0);
    should_run_[group] = involvedDevices(expr).count(comm_.deviceId());
  }
  // prepare the order in which to launch the kernels/comms
  prepareRuntimeOrder(staged_fusion_.get(), workspace);

  // Allocator setup
  // First, figure out what Vals need allocation at runtime
  for (auto group : staged_fusion_->groups()) {
    if (is_resharding_[group]) {
      NVF_ERROR(group->exprs().size() == 1);
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
  // Then, instantiate the allocator
  if (!vals_to_allocate_.empty()) {
    allocator_ = std::make_unique<FusionExecutorCache>(
        copyFusionAndChangeOutputs(completeFusion(), vals_to_allocate_),
        0,
        false);
  }
}

void MultiDeviceExecutor::postKernel(SegmentedGroup* group) {
  if (!should_run_.at(group)) {
    return;
  }
  // get the IValues corresponding to the group's input
  std::vector<c10::IValue> group_input_IValues;
  for (auto& input : group->inputs()) {
    NVF_ERROR(
        val_to_IValue_.find(input) != val_to_IValue_.end(),
        "Device ",
        comm_.deviceId(),
        " has no buffer associated with Val ",
        input,
        " for handling group ",
        toString(group));
    NVF_ERROR(val_to_IValue_.at(input).isTensor());
    group_input_IValues.push_back(val_to_IValue_.at(input));
  }

  // placeholder for storing the group's outputs
  std::vector<at::Tensor> outputs;

  // Compile the group and execute it with FusionExecutor
  // Check if the executor has been cached. If not, create and cache it
  if (fec_.find(group) == fec_.end()) {
    fec_.emplace(
        group,
        std::make_unique<FusionExecutorCache>(
            staged_fusion_->makeFusion(group), 0, auto_schedule_));
  }
  outputs = fec_[group]->runFusionWithInputs(group_input_IValues);

  // Store the outputs in the context
  for (auto output_idx : c10::irange(outputs.size())) {
    val_to_IValue_[group->outputs().at(output_idx)] = outputs.at(output_idx);
  }
}

void MultiDeviceExecutor::postCommunication(SegmentedGroup* group) {
  // Lower the group into a vector of Communications
  NVF_ERROR(
      group->exprs().size() == 1,
      "Communication segments must contain only one Expr");
  auto expr = group->exprs().at(0);
  NVF_ERROR(
      expr->inputs().size() == 1, "Communication must have exactly one input");
  NVF_ERROR(
      expr->outputs().size() == 1,
      "Communication must have exactly one output");
  auto input_val = expr->inputs().at(0);
  auto output_val = expr->outputs().at(0);
  at::Tensor input_tensor, output_tensor;
  if (val_to_IValue_.find(input_val) != val_to_IValue_.end()) {
    input_tensor = val_to_IValue_.at(input_val).toTensor();
  }
  if (val_to_IValue_.find(output_val) != val_to_IValue_.end()) {
    output_tensor = val_to_IValue_.at(output_val).toTensor();
  }

  auto communications =
      lowerCommunication(comm_.deviceId(), expr, input_tensor, output_tensor);

  // post and wait communications
  for (auto& communication : communications) {
    auto work = communication->post(comm_);
    if (work) {
      work->wait();
    }
  }
}

std::vector<at::Tensor> MultiDeviceExecutor::runWithInput(
    const std::vector<c10::IValue>& inputs) {
  // make sure the communicator can run the Fusion (e.g. there is enough GPUs,
  // etc)
  auto error_msg = validate();
  NVF_ERROR(error_msg.empty(), error_msg);

  // Make sure inputs align at global boundary.
  NVF_ERROR(
      inputs.size() == staged_fusion_->inputs().size(),
      "Wrong number of inputs");

  auto start = std::chrono::high_resolution_clock::now();
  allocateBuffers(inputs);
  auto end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> duration = end - start;
  std::cout << "Allocate recv buffers: " << duration.count() << "seconds" << std::endl;
  
  // process input values:
  for (auto input_idx : c10::irange(inputs.size())) {
    val_to_IValue_[staged_fusion_->inputs().at(input_idx)] =
        inputs.at(input_idx);
  }

  // Run through the groups to launch kernels and comms
  for (auto group : workspace.group_run_order) {
    if (!is_resharding_.at(group)) {
      start = std::chrono::high_resolution_clock::now();
      postKernel(group);
      end = std::chrono::high_resolution_clock::now();
      duration = end - start;
      std::cout << "Kernel call: " << duration.count() << "seconds" << std::endl;
    } else {
      start = std::chrono::high_resolution_clock::now();
      postCommunication(group);
      end = std::chrono::high_resolution_clock::now();
      duration = end - start;
      std::cout << "Comms call: " << duration.count() << "seconds" << std::endl;
    }
  }

  // Collect global outputs from context
  std::vector<at::Tensor> outputs;
  for (auto output_val : staged_fusion_->outputs()) {
    auto output = (val_to_IValue_.find(output_val) != val_to_IValue_.end())
        ? val_to_IValue_.at(output_val).toTensor()
        : at::Tensor();
    outputs.push_back(output);
  }

  return outputs;
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

std::ostream& MultiDeviceExecutor::print() {
  int compute_segment_counter = 0;
  int communication_counter = 0;
  for (auto group : workspace.group_run_order) {
    if (is_resharding_[group]) {
      debug() << "Communication " << communication_counter << ":{\n";
      for (const auto& comm :
           lowerCommunication(comm_.deviceId(), group->exprs().at(0), {}, {})) {
        debug() << comm->toString(2) << "\n";
      }
      debug() << "}\n";
      communication_counter++;
    } else {
      debug() << "Compute segment " << compute_segment_counter << ":{\n";
      auto fusion = staged_fusion_->makeFusion(group);
      fusion->print();
      debug() << "}\n";
      compute_segment_counter++;
    }
  }
  return debug();
}

} // namespace nvfuser

#endif
