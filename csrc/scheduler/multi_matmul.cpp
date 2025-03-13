// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on

#include <ATen/cuda/CUDAContext.h>
#include <c10/util/irange.h>
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
    // TODO: properly handle all mul+sum patterns for Hopper. For now, these
    // should work fine as long as the inner dimensions are the ones being
    // reduced.
    if (!isAmpere(params_->mma_macro) && !isTuring(params_->mma_macro) &&
        pattern.output->definition()->isA<ReductionOp>()) {
      bool found_reduction = false;
      for (size_t dim : c10::irange((size_t)pattern.output->nDims())) {
        NVF_ERROR(
            !found_reduction ||
                !pattern.output->axis((int64_t)dim)->isReduction(),
            "Mul+Sum patterns can only be translated on Hopper if the reduction dim is innermost");
      }
    }

    mma_utils::MatmulPattern::TranslationResult res = pattern.translateToMmaOp(
        /*avoid_intermediates=*/!isAmpere(params_->mma_macro) &&
        !isTuring(params_->mma_macro));
    mma_results_.push_back(res.mma->out()->as<TensorView>());

    // During MatmulPattern translation, we might replace some tensors in the
    // fusion. If those replaced tensors were themselves the A or B members of
    // another MatmulPattern, we should update the pattern to point to the
    // replacement.
    for (mma_utils::MatmulPattern& other_pattern : patterns_) {
      if (&other_pattern == &pattern) {
        continue;
      }
      if (auto it = res.replacements.find(other_pattern.A);
          it != res.replacements.end()) {
        other_pattern.A = it->second;
      }
      if (auto it = res.replacements.find(other_pattern.B);
          it != res.replacements.end()) {
        other_pattern.B = it->second;
      }
    }
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
  // When translating MatmulOp or LinearOp with avoid_intermediates, we
  // introduce some intermediate Global tensors which will be ignored during
  // lowering. We update as_ and bs_ to point at the last of these tensors
  // before their next consumer is in non-global memory.
  auto find_last_global_consumer = [](TensorView* tv) -> TensorView* {
    // Example: Suppose we start out with:
    //
    //   Inputs:
    //     tv0_g
    //     tv1_g
    //
    //   tv2_l = matmul(tv0_g, tv1_g)
    //
    // Earlier in scheduling we replace the operands to produce something like:
    //
    //   Inputs:
    //     tv0_g
    //     tv1_g
    //
    //   tv3_g = broadcast(tv0_g)
    //   tv4_g = broadcast(tv1_g)
    //   tv5_g = permute(tv4_g)
    //   tv2 = matmul(tv3, tv5)
    //
    // We start out with:
    //
    //   tensor_roles_[A] = {tv0_g}
    //   tensor_roles_[B] = {tv1_g}
    //
    // Here we update that to:
    //
    //   tensor_roles_[A] = {tv3_g}
    //   tensor_roles_[B] = {tv5_g}
    while (tv != nullptr) {
      if (tv->uses().size() != 1) {
        break;
      }
      Expr* use = tv->uses().front();

      // TODO: support ViewOp
      if (!use->isOneOf<BroadcastOp, SqueezeOp, LoadStoreOp>()) {
        break;
      }
      TensorView* consumer = ir_utils::getTvOutput(use);
      if (consumer == nullptr ||
          consumer->getMemoryType() != MemoryType::Global) {
        break;
      }

      // Traverse down consumers
      tv = consumer;
    }
    return tv;
  };

  // Apply in-place transformation
  std::transform(
      as_.cbegin(), as_.cend(), as_.begin(), find_last_global_consumer);
  std::transform(
      bs_.cbegin(), bs_.cend(), bs_.begin(), find_last_global_consumer);

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
