// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#ifdef USE_DISTRIBUTED
#include <multidevice/lower_resharding_expr.h>
#include <exceptions.h>
#include <ir/utils.h>
#include <scheduler/utils.h>
#include <ops/all_ops.h>
#include <multidevice/lower_communication.h>


namespace nvfuser{

namespace {

void shardAllLike(TensorView* ref, std::vector<TensorView*> tvs) {
    for (auto tv: tvs) {
        tv->setDeviceMesh(ref->getDeviceMesh());
    }
    scheduler_utils::parallelizeAllLike(ref, tvs, {ParallelType::DIDx});
}

void reshardBefore(Expr* expr, Fusion* fusion) {
    NVF_ERROR(expr->outputs().size() == 1, "multi-output expressions are not supported");
    NVF_ERROR(expr->outputs().at(0)->isA<TensorView>(), "the expression's output is not a TensorView");
    TensorView* output = expr->outputs().at(0)->as<TensorView>();
    std::unordered_set<TensorView*> inputs;
    std::transform(expr->inputs().begin(),
                   expr->inputs().end(),
                   std::inserter(inputs, inputs.end()),
                   [](Val* val) {
                     NVF_ERROR(val->isA<TensorView>(), "the expression's input is not a TensorView");
                     return val->as<TensorView>();});
    std::vector<TensorView*> new_inputs;
    for (auto input: ir_utils::haveDifferentSharding(output, inputs)) {
        TensorView* new_input = set(input);
        expr = ir_utils::replaceValInExprInputs(expr, input, new_input);
        new_inputs.push_back(new_input);
        // add something like that ? :
        // auto replayed_consumer_pair = TransformReplay::replayCasP(
        // new_input, input, -1, TransformReplayOptions().replayAllocation());
        // new_input->setDomain(replayed_consumer_pair.first);
    }
    if (!new_inputs.empty()) {
        shardAllLike(output, new_inputs);
    }
}

} //namespace

void insertReshardings(Fusion* fusion) {
    auto exprs = fusion->exprs();
    for (auto expr: exprs) {
        if (!isLowerableToCommunication(expr)) {
            reshardBefore(expr, fusion);
        }
    }
}

} //namespace nvfuser

#endif
