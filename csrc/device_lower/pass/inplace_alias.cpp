// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on

#include <device_lower/lower2device.h>
#include <device_lower/pass/inplace_alias.h>
#include <device_lower/utils.h>
#include <ir/builder.h>
#include <ir/utils.h>
#include <kernel_ir.h>
#include <kernel_ir_dispatch.h>

namespace nvfuser {

namespace {

// Properties gathered from a given Kernel
struct InplaceAliasInfo {
  // Map to find the Allocate node for each tensor
  std::unordered_map<TensorView*, kir::Allocate*> alloc_map;
  // Disjoint sets to group aliased tensors
  DisjointSets<TensorView*> aliased_tvs;
  // Unique tensor for each tensor alias groups representing its real allocation
  std::unordered_map<DisjointSets<TensorView*>::DisjointSet, TensorView*>
      real_alloc_map;
};

class InplaceAliasInfoBuilder : public kir::IrVisitor {
 public:
  const InplaceAliasInfo& info() {
    return info_;
  }

  using IrVisitor::handle;

  void handle(ScatterOp* sop) final {
    auto in_tv = sop->in()->as<TensorView>();
    auto out_tv = sop->out()->as<TensorView>();

    // Note that in_tv and out_tv are already validated to be safe to
    // alias each other by validateScatter

    NVF_ERROR(
        info_.alloc_map.find(in_tv) != info_.alloc_map.end(),
        "No allocation mapping found for scatter input: ",
        in_tv->toString());
    NVF_ERROR(
        info_.alloc_map.find(out_tv) != info_.alloc_map.end(),
        "No allocation mapping found for scatter output: ",
        out_tv->toString());

    auto in_tv_alias_it = info_.aliased_tvs.find(in_tv);
    if (in_tv_alias_it != info_.aliased_tvs.end()) {
      info_.aliased_tvs.appendToSet(out_tv, in_tv_alias_it->second);
    } else {
      info_.aliased_tvs.mapEntries(in_tv, out_tv);
      in_tv_alias_it = info_.aliased_tvs.find(in_tv);
      // Pick the input as the actual allocation of this tensor group
      info_.real_alloc_map.emplace(in_tv_alias_it->second, in_tv);
    }

    // If the output is also a fusion output, use it as the real
    // allocation.
    if (out_tv->isFusionOutput()) {
      info_.real_alloc_map[in_tv_alias_it->second] = out_tv;
    }
  }

  void handle(kir::Allocate* alloc) final {
    // Keep track of tensor allocations. Do not bother if already
    // aliasing another
    if (auto alloc_tv = dynamic_cast<TensorView*>(alloc->buffer());
        alloc_tv != nullptr && alloc->alias() == nullptr) {
      NVF_ERROR(info_.alloc_map.emplace(alloc_tv, alloc).second);
    }
  }

 private:
  InplaceAliasInfo info_;
};

class InplaceAliasMutator : public kir::ExprMutator {
 public:
  InplaceAliasMutator(const InplaceAliasInfo& info) : info_(info) {}

 protected:
  using ExprMutator::handle;

  void handle(kir::Allocate* alloc) final {
    auto tv = dynamic_cast<TensorView*>(alloc->buffer());
    if (tv == nullptr) {
      // Ignore non-tensor allocation
      return;
    }

    auto alias_it = info_.aliased_tvs.find(tv);
    if (alias_it == info_.aliased_tvs.end()) {
      // Not aliased
      return;
    }

    auto real_alloc_tv = info_.real_alloc_map.at(alias_it->second);
    if (tv == real_alloc_tv) {
      // This tensor is the actual allocation
      return;
    }

    auto real_alloc = info_.alloc_map.at(real_alloc_tv);

    auto new_alloc = IrBuilder::create<kir::Allocate>(
        alloc->buffer(),
        alloc->memoryType(),
        alloc->shape(),
        alloc->zeroInit(),
        /*reset_to_zero=*/false,
        real_alloc);

    registerReplace(alloc, new_alloc);
  }

 private:
  const InplaceAliasInfo& info_;
};

} // namespace

std::vector<Expr*> setInplaceAlias(const std::vector<Expr*>& exprs) {
  InplaceAliasInfoBuilder builder;
  builder.handle(exprs);
  return InplaceAliasMutator(builder.info()).traverseAndInsert(exprs);
}

} // namespace nvfuser
