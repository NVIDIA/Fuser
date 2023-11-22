// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on

#include <device_lower/pass/inline_ptx.h>
#include <device_lower/utils.h>
#include <ir/builder.h>
#include <ir/utils.h>
#include <kernel_ir_dispatch.h>

#include <sstream>

namespace nvfuser {

class LowerToInlinePtx : public kir::ExprMutator {
 protected:
  using ExprMutator::handle;

  void handle(kir::CpAsyncCommit* commit) override {
    registerReplace(
        commit,
        IrBuilder::create<kir::Asm>(
            "cp.async.commit_group",
            std::vector<Val*>{},
            std::vector<Val*>{},
            kir::Asm::Options{true}));
  }

  void handle(kir::CpAsyncWait* wait) override {
    auto stages = wait->keepStages();
    Expr* replace = nullptr;
    if (stages > 0) {
      replace = IrBuilder::create<kir::Asm>(
          "cp.async.wait_group",
          std::vector<Val*>{},
          std::vector<Val*>{IrBuilder::create<Val>(stages)},
          kir::Asm::Options{true});
    } else {
      replace = IrBuilder::create<kir::Asm>(
          "cp.async.wait_all",
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
            "cp.async.bulk.commit_group",
            std::vector<Val*>{},
            std::vector<Val*>{},
            kir::Asm::Options{true}));
  }

  void handle(kir::CpAsyncBulkS2GWait* wait) override {
    auto stages = wait->keepStages();
    registerReplace(
        wait,
        IrBuilder::create<kir::Asm>(
            "cp.async.bulk.wait_group.read",
            std::vector<Val*>{},
            std::vector<Val*>{IrBuilder::create<Val>(stages)},
            kir::Asm::Options{true, true}));
  }

  void handle(LoadStoreOp* ldst) override {
    if (ir_utils::isLdMatrixOp(ldst)) {
      auto op = ldst->opType();
      std::stringstream ss;
      ss << "ldmatrix.sync.aligned.x"
         << std::get<ArrayType>(ldst->out()->dtype().type).size;
      if (op == LoadStoreOpType::LdMatrixTranspose) {
        ss << ".trans";
      }
      ss << ".m8n8.shared.b16";
      registerReplace(
          ldst,
          IrBuilder::create<kir::Asm>(
              ss.str(),
              std::vector<Val*>{ldst->out()},
              std::vector<Val*>{ldst->in()},
              kir::Asm::Options{true}));
      return;
    } else if (ir_utils::isCpAsyncOp(ldst)) {
      auto out_tv = ldst->out()->as<kir::TensorIndex>()->view();
      auto vec_size =
          ir_utils::getVectorizeSize(out_tv) * dataTypeSize(out_tv->dtype());
      std::stringstream ss;
      ss << "cp.async.";
      if (ldst->cacheOp() == CacheOp::AllLevels) {
        ss << "ca";
      } else {
        ss << "cg";
        NVF_ERROR(vec_size == 16, "cp.async.cg only support vectorize 16 bytes");
      }
      ss << ".shared.global";
      registerReplace(
          ldst,
          IrBuilder::create<kir::Asm>(
              ss.str(),
              std::vector<Val*>{},
              std::vector<Val*>{
                  ldst->out(),
                  ldst->in(),
                  IrBuilder::create<Val>(vec_size),
                  ldst->predicate()},
              kir::Asm::Options{true}));
    }
  }
};

std::vector<Expr*> lowerToInlinePtx(const std::vector<Expr*>& exprs) {
  return LowerToInlinePtx{}.traverseAndInsert(exprs);
}

} // namespace nvfuser
