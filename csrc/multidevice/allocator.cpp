// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#ifdef USE_DISTRIBUTED
#include <ir/utils.h>
#include <multidevice/allocator.h>
#include <fusion.h>
#include <executor.h>
#include <ir/cloner.h>

namespace nvfuser {

std::pair<std::unique_ptr<Fusion>, std::unordered_map<Val*, Val*>> copyFusionAndChangeOutputs(Fusion* fusion, std::unordered_set<Val*> outputs) {
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
    std::for_each(
        outputs.begin(),
        outputs.end(),
        [&](Val* const& output) {
        fusion_copy->addOutput(original_to_copy_cloner.clone(output));
        copy_to_original_map[original_to_copy_cloner.clone(output)] = output;
        });

    for (auto tv : ir_utils::filterByType<TensorView>(fusion_copy->vals())) {
      tv->setMemoryType(MemoryType::Global);
    }

    return std::make_pair<std::unique_ptr<Fusion>, std::unordered_map<Val*, Val*>>(std::move(fusion_copy), std::move(copy_to_original_map));
}


std::unordered_map<Val*, c10::IValue> allocatePipelineIntermediateBuffers(Pipeline* pipeline, DeviceIdxType dId, std::vector<c10::IValue> global_inputs_IValues) {
    std::unordered_set<Val*> vals_to_allocate;
    std::unordered_set<Val*> vals_to_not_allocate;
    const auto& exprs = pipeline->exprs();
    for (auto stage: ir_utils::filterByType<PipelineStage>(exprs)) {
        if (stage->descriptor()->mesh.has(dId)) {
            for (auto input: stage->inputs()) {
                auto input_val = input->as<PipelineVal>()->getOriginalVal();
                vals_to_allocate.insert(input_val);
            }
            // for (auto output: stage->outputs()) {
            //     auto output_val = output->as<PipelineVal>()->getOriginalVal();
            //     vals_to_not_allocate.insert(output_val);
            // }
        }
    }
    for (auto global_output: pipeline->originalFusion()->outputs()) {
        vals_to_allocate.insert(global_output);
    }
    for (auto global_input: pipeline->originalFusion()->inputs()) {
        vals_to_not_allocate.insert(global_input);
    }

    for (auto val_to_not_allocate: vals_to_not_allocate){
        vals_to_allocate.erase(val_to_not_allocate);
    }
    // vals_to_allocate.erase(vals_to_not_allocate.begin(), vals_to_not_allocate.end());
    auto [fusion_copy, copy_to_original_map] = copyFusionAndChangeOutputs(pipeline->originalFusion(), vals_to_allocate);
    if (fusion_copy->outputs().empty()) {
        return {};
    }
    FusionExecutor fe;
    fe.compileFusion(fusion_copy.get(), global_inputs_IValues);
    auto buffers = fe.allocOutputSpace(global_inputs_IValues);

    std::unordered_map<Val*, c10::IValue> allocations;
    for (auto i: c10::irange(buffers.size())) {
        // allocations.emplace(vals_to_allocate.at(i), buffers.at(i));
        allocations.emplace(copy_to_original_map[fusion_copy->outputs().at(i)], buffers.at(i));
    }

    return allocations;
}


} // namespace nvfuser

#endif
