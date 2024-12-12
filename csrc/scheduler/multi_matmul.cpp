// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on

#include <ATen/cuda/CUDAContext.h>
#include <multidevice/utils.h>
#include <scheduler/ampere_multi_matmul.h>
#include <scheduler/hopper_multi_matmul.h>

namespace nvfuser {

void MultipleMatmulScheduler::findPatterns() {
  patterns_ = mma_utils::findMatmulPatterns(fusion_);
  NVF_ERROR(!patterns_.empty(), "No matmul patterns were found");
}

void MultipleMatmulScheduler::translatePatterns() {
  mma_results_.reserve(patterns_.size());
  for (mma_utils::MatmulPattern& pattern : patterns_) {
    MmaOp* mma = pattern.translateToMmaOp();
    mma_results_.push_back(mma->out()->as<TensorView>());
  }

  // Build IdModel graphs now since translateToMmaOp creates new TVs. Before
  // this point the graphs are not yet built.
  updateIdModel();
}

// Get tensor roles and id roles
// When there are multiple matmul patterns, we can have conflicting roles.
// For now we throw an error if this is the case.
// TODO: This should be checked in canScheduleCompileTime
void MultipleMatmulScheduler::findRoles() {
  const auto roles_opt = mma_utils::allPatternRoles(id_model_, patterns_);
  NVF_ERROR(
      roles_opt.has_value(),
      "Incompatible roles found between matmul patterns");
  std::tie(id_roles_, tensor_roles_) = roles_opt.value();

  mma_utils::MatmulOperandInnerDimsOpt inner_dims_opt =
      mma_utils::getOperandInnerDims(id_model_, id_roles_, tensor_roles_);
  NVF_ERROR(inner_dims_opt.isValid(), inner_dims_opt.getErrorMsg());
  inner_dims_ = inner_dims_opt.getData();

  as_ = tensor_roles_.at(MatmulTensorRole::OPERAND_A);
  bs_ = tensor_roles_.at(MatmulTensorRole::OPERAND_B);

  countDims();
}

void MultipleMatmulScheduler::countDims() {
  NVF_ERROR(!patterns_.empty());
  TensorView* mma_result = patterns_.front().output;
  num_device_dims_ = numDeviceDims(mma_result);
  for (const auto& it : id_roles_) {
    if (it.second == MatmulDimRole::Batch &&
        // Skip device dims
        !std::any_of(it.first->begin(), it.first->end(), [](Val* v) {
          return v->as<IterDomain>()->isDeviceDim();
        })) {
      // All batch dims will be merged into one, if any exist
      num_local_batch_dims_ = 1;
    }
  }
  num_splitk_dims_ = params_->splitk_factor > 1 ? 1 : 0;
  // Subtract 6 for the [Mo, No, Ko, Mi, Ni, Ki]
  num_device_and_batch_dims_ = num_device_dims_ + num_local_batch_dims_;
}

//! Rebuilds IdModel, then updates all ValGroups in abstract tensors to refer
//! to the new IdModel. This is necessary whenever we perform an operation
//! that creates a new TensorView, such as caching or rFactor
void MultipleMatmulScheduler::updateIdModel() {
  // Build new IdModel
  IdModel new_id_model(fusion_, /*build_graphs=*/false);
  new_id_model.buildBroadcastGraph();

  // Get new broadcast graph
  ValGraph& new_graph = new_id_model.idGraph(IdMappingMode::BROADCAST);

  if (!id_roles_.empty()) {
    // Update id_roles_ to have keys corresponding to ValGroups in the new
    // IdModel
    std::unordered_map<ValGroup, MatmulDimRole> new_id_roles;
    for (auto& [k, v] : id_roles_) {
      const ValGroup& new_group = new_graph.toGroup(k->front());
      new_id_roles.emplace(new_group, v);
    }
    id_roles_ = new_id_roles;
  }

  graph_ = &new_id_model.idGraph(IdMappingMode::BROADCAST);

  // Set id_model_ after we are done using the old one
  id_model_ = std::move(new_id_model);
}

void scheduleMultipleMatmuls(Fusion* fusion, const MatmulParams* params) {
  FusionGuard fg(fusion);

  // NOTE: In the future we should be able to simply check the generation of
  // the macro instead of looking at the device properties here. However,
  // until we have Hopper mma ready, we will be using Ampere macros on Hopper
  // machines for testing. This means in order to trigger Hopper code, we need
  // to look at the device instead of the macro for now. See commented
  // conditions below.
  const auto device_prop = at::cuda::getCurrentDeviceProperties();
  const int cc = device_prop->major * 10 + device_prop->minor;
  // AmpereMultipleMatmulScheduler(fusion, params).run();
  if (cc >= 75 && cc < 90) {
    AmpereMultipleMatmulScheduler(fusion, params).run();
  } else if (cc >= 90 && cc < 100) {
    HopperMultipleMatmulScheduler(fusion, params).run();
  } else {
    NVF_THROW(
        "The matrix multiplication scheduler is unavailable for this device: ",
        device_prop->major,
        ".",
        device_prop->minor);
  }
}

} // namespace nvfuser
