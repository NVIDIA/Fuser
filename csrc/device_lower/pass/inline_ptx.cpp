// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on

#include <device_lower/pass/inline_ptx.h>
#include <ir/builder.h>
#include <kernel_ir_dispatch.h>

namespace nvfuser {

class LowerToInlinePtx : public kir::ExprMutator {
 protected:
  using ExprMutator::handle;

  void handle(kir::CpAsyncCommit* commit) override {
    registerReplace(
        commit,
        IrBuilder::create<kir::Asm>(
            "cp.async.commit_group;",
            std::vector<Val*>{},
            std::vector<Val*>{},
            kir::Asm::Options{true}));
  }

  void handle(kir::CpAsyncWait* wait) override {
    auto stages = wait->keepStages();
    Expr* replace = nullptr;
    if (stages > 0) {
      replace = IrBuilder::create<kir::Asm>(
          "cp.async.wait_group %0;",
          std::vector<Val*>{},
          std::vector<Val*>{IrBuilder::create<Val>(stages)},
          kir::Asm::Options{true});
    } else {
      replace = IrBuilder::create<kir::Asm>(
          "cp.async.wait_all;",
          std::vector<Val*>{},
          std::vector<Val*>{},
          kir::Asm::Options{true});
    }

    registerReplace(wait, replace);
  }

  void handle(kir::CpAsyncBulkS2GCommit* commit) override {
    registerReplace(
        commit,
        IrBuilder::create<kir::Asm>(
            "cp.async.bulk.commit_group;",
            std::vector<Val*>{},
            std::vector<Val*>{},
            kir::Asm::Options{true}));
  }

  void handle(kir::CpAsyncBulkS2GWait* wait) override {
    auto stages = wait->keepStages();
    registerReplace(
        wait,
        IrBuilder::create<kir::Asm>(
            "cp.async.bulk.wait_group.read %0;",
            std::vector<Val*>{},
            std::vector<Val*>{IrBuilder::create<Val>(stages)},
            kir::Asm::Options{true, true}));
  }
};

std::vector<Expr*> lowerToInlinePtx(const std::vector<Expr*>& exprs) {
  return LowerToInlinePtx{}.traverseAndInsert(exprs);
}

} // namespace nvfuser
