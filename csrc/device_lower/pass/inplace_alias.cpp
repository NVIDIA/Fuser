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

#include <sstream>

namespace nvfuser {

namespace {

struct InplaceAliasInfo {
  std::unordered_map<TensorView*, kir::Allocate*> alloc_map;
  DisjointSets<TensorView*> aliased_tvs;
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

    auto in_tv_alias_it = info_.aliased_tvs.find(in_tv);
    if (in_tv_alias_it != info_.aliased_tvs.end()) {
      info_.aliased_tvs.appendToSet(out_tv, in_tv_alias_it->second);
    } else {
      info_.aliased_tvs.mapEntries(in_tv, out_tv);
      in_tv_alias_it = info_.aliased_tvs.find(in_tv);
      info_.real_alloc_map.emplace(in_tv_alias_it->second, in_tv);
    }

    if (out_tv->isFusionOutput()) {
      std::cerr << "Aliased to output: " << sop->toString();
      info_.real_alloc_map[in_tv_alias_it->second] = out_tv;
    }
  }

  void handle(kir::Allocate* alloc) final {
    // Keep track of tensor allocations
    if (auto alloc_tv = dynamic_cast<TensorView*>(alloc->buffer())) {
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
      return;
    }

    auto alias_it = info_.aliased_tvs.find(tv);
    if (alias_it == info_.aliased_tvs.end()) {
      return;
    }

    std::cerr << "Possibly Alias: " << alloc->toString();

    auto real_alloc_tv = info_.real_alloc_map.at(alias_it->second);
    if (tv == real_alloc_tv) {
      return;
    }

    std::cerr << "Real alloc: " << real_alloc_tv->toString() << "\n";

    auto real_alloc = info_.alloc_map.at(real_alloc_tv);

    auto new_alloc = IrBuilder::create<kir::Allocate>(
        alloc->buffer(),
        alloc->memoryType(),
        alloc->shape(),
        alloc->zeroInit(),
        /*reset_to_zero=*/false,
        real_alloc);

    std::cerr << "Mutating " << alloc->toString();
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
