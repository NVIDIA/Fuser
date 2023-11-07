// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#ifdef USE_DISTRIBUTED
#include <executor.h>
#include <fusion.h>
#include <ir/cloner.h>
#include <ir/utils.h>
#include <multidevice/allocator.h>

namespace nvfuser {

namespace {

// fully copy a fusion and replace the original outputs by the one specified as
// arguments returns a tuple composed of a pointer to the copied fusion and a
// "copy to original" Val map
std::pair<std::unique_ptr<Fusion>, std::unordered_map<Val*, Val*>>
copyFusionAndChangeOutputs(Fusion* fusion, std::unordered_set<Val*> outputs) {
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
    copy_to_original_map[original_to_copy_cloner.clone(output)] = output;
  });

  for (auto tv : ir_utils::filterByType<TensorView>(fusion_copy->vals())) {
    tv->setMemoryType(MemoryType::Global);
    for (auto i : c10::irange(tv->domain()->nDims())) {
      tv->axis(i)->parallelize(ParallelType::Serial);
    }
  }

  return std::
      make_pair<std::unique_ptr<Fusion>, std::unordered_map<Val*, Val*>>(
          std::move(fusion_copy), std::move(copy_to_original_map));
}

} // namespace

std::unordered_map<Val*, c10::IValue> allocatePipelineIntermediateBuffers(
    Pipeline* pipeline,
    DeviceIdxType my_device_index,
    std::vector<c10::IValue> global_inputs_IValues) {
  // Stores the Vals that needs to be allocated
  std::unordered_set<Val*> vals_to_allocate;
  // Add all the input of stages run by the current device
  const auto& exprs = pipeline->exprs();
  for (auto stage : ir_utils::filterByType<PipelineStage>(exprs)) {
    if (stage->descriptor()->mesh.has(my_device_index)) {
      for (auto input : stage->inputs()) {
        auto input_val = input->as<PipelineVal>()->getOriginalVal();
        vals_to_allocate.insert(input_val);
      }
    }
  }
  // Add all the global outputs
  for (auto global_output : pipeline->originalFusion()->outputs()) {
    vals_to_allocate.insert(global_output);
  }
  // Remove any global inputs that have been added
  for (auto global_input : pipeline->originalFusion()->inputs()) {
    vals_to_allocate.erase(global_input);
  }

  // We copy the original fusion and set the outputs to be the tensors to be
  // allocated This way we can directly use FusionExecutor::allocOutputSpace
  auto [fusion_copy, copy_to_original_map] =
      copyFusionAndChangeOutputs(pipeline->originalFusion(), vals_to_allocate);
  if (fusion_copy->outputs().empty()) {
    return {};
  }
  FusionExecutor fe;
  fe.compileFusion(fusion_copy.get(), global_inputs_IValues);
  auto allocated_tensors = fe.allocOutputSpace(global_inputs_IValues);

  // Stores the returned result
  std::unordered_map<Val*, c10::IValue> allocations;
  // Map each symbolic tensor to its allocated buffer
  for (auto i : c10::irange(allocated_tensors.size())) {
    allocations.emplace(
        copy_to_original_map[fusion_copy->outputs().at(i)],
        allocated_tensors.at(i));
  }

  return allocations;
}

} // namespace nvfuser

#endif
