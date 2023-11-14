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
#include <ops/all_ops.h>
#include <multidevice/lower_communication.h>


namespace nvfuser{

namespace {

bool needsSetInsertion(Expr* expr) {
    return ir_utils::isResharding(expr) && !isLowerableToCommunication(expr);
}

bool areSameExceptMaybeParallelType(IterDomain* id1, IterDomain* id2) {
    return id1->start()->sameAs(id2->start()) &&
      id1->extent()->sameAs(id2->extent()) &&
      id1->hasExpandedExtent() == id2->hasExpandedExtent() &&
      (!id1->hasExpandedExtent() ||
       id1->expandedExtent()->sameAs(id2->expandedExtent())) &&
      id1->stopOffset()->sameAs(id2->stopOffset()) &&
      id1->getIterType() == id2->getIterType() &&
      id1->hasPaddingToMultipleOfWarp() == id2->hasPaddingToMultipleOfWarp() &&
      id1->getMaybeSizeAfterPadding() == id2->getMaybeSizeAfterPadding() &&
      id1->isMmaSwizzled() == id2->isMmaSwizzled();
}

void setSameSharding(TensorView* tv, TensorView* ref) {
    tv->setDeviceMesh(ref->getDeviceMesh());

    auto domain = tv->getLeafDomain();
    auto domain_ref = ref->getLeafDomain();
    NVF_ERROR(domain.size() == domain_ref.size(), "tensors ", tv, " and ",
        ref, " have mismatching leaf domains");
    for (auto i: c10::irange(domain_ref.size())) {
        auto id = domain.at(i);
        auto id_ref = domain_ref.at(i);
        if (id_ref->isDevice() || id->isDevice()) {
            NVF_ERROR(areSameExceptMaybeParallelType(id, id_ref), "IterDomains ", id, " and ",
                id_ref, " are mismatching");
            id->parallelize(id_ref->getParallelType());
        }
    }
}

void insertSetBefore(Expr* expr, Fusion* fusion) {
    NVF_ERROR(expr->outputs().size() == 1, "multi-output expressions are not supported");
    NVF_ERROR(expr->outputs().at(0)->isA<TensorView>(), "the expression's output is not a TensorView");
    TensorView* output = expr->outputs().at(0)->as<TensorView>();
    auto inputs = expr->inputs();
    for (auto input: inputs) {
        NVF_ERROR(input->isA<TensorView>(), "the expression's input is not a TensorView");
        auto input_tv = input->as<TensorView>();
        if (!ir_utils::haveSameSharding(input_tv, output)) {
            TensorView* new_input = set(input_tv);
            setSameSharding(new_input, output);
            expr = ir_utils::replaceValInExprInputs(expr, input_tv, new_input);
        }
    }
}

} //namespace

void insertSetBeforeReshardingExpr(Fusion* fusion) {
    auto exprs = fusion->exprs();
    for (auto expr: exprs) {
        if (needsSetInsertion(expr)) {
            insertSetBefore(expr, fusion);
        }
    }
}

} //namespace nvfuser

#endif
