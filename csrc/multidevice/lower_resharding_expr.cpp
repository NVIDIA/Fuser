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
    return expr->isResharding() && !isLowerableToCommunication(expr);
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
    NVF_ERROR(domain.size() == domain_ref.size());
    for (auto i: c10::irange(domain_ref.size())) {
        auto id = domain.at(i);
        auto id_ref = domain_ref.at(i);
        if (id_ref->isDevice()) {
            NVF_ERROR(areSameExceptMaybeParallelType(id, id_ref));
            id->parallelize(id_ref->getParallelType());
        }
    }
}

void insertSetBefore(Expr* expr, Fusion* fusion) {
    NVF_ERROR(expr->outputs().size() == 1);
    NVF_ERROR(expr->outputs().at(0)->isA<TensorView>());
    TensorView* output = expr->outputs().at(0)->as<TensorView>();
    for (auto input: expr->inputs()) {
        NVF_ERROR(input->isA<TensorView>());
        auto input_tv = input->as<TensorView>();
        TensorView* new_input = set(input_tv);
        ir_utils::replaceValInExprInputs(expr, input_tv, new_input);
        setSameSharding(new_input, output);
    }
}

} //namespace

void insertSetBeforeReshardingExpr(Fusion* fusion) {
    for (auto expr: fusion->exprs()) {
        if (needsSetInsertion(expr)) {
            insertSetBefore(expr, fusion);
        }
    }
}

} //namespace nvfuser

#endif
