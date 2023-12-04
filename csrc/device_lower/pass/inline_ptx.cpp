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
            kir::Asm::Options{/*volatile=*/true}));
  }

  void handle(kir::CpAsyncWait* wait) override {
    auto stages = wait->keepStages();
    Expr* replace = nullptr;
    if (stages > 0) {
      replace = IrBuilder::create<kir::Asm>(
          "cp.async.wait_group",
          std::vector<Val*>{},
          std::vector<Val*>{IrBuilder::create<Val>(stages)},
          kir::Asm::Options{/*volatile=*/true});
    } else {
      replace = IrBuilder::create<kir::Asm>(
          "cp.async.wait_all",
          std::vector<Val*>{},
          std::vector<Val*>{},
          kir::Asm::Options{/*volatile=*/true});
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
            kir::Asm::Options{/*volatile=*/true}));
  }

  void handle(kir::CpAsyncBulkS2GWait* wait) override {
    auto stages = wait->keepStages();
    registerReplace(
        wait,
        IrBuilder::create<kir::Asm>(
            "cp.async.bulk.wait_group.read",
            std::vector<Val*>{},
            std::vector<Val*>{IrBuilder::create<Val>(stages)},
            kir::Asm::Options{/*volatile=*/true, /*memory=*/true}));
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
              kir::Asm::Options{/*volatile=*/true}));
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
        NVF_ERROR(
            vec_size == 16, "cp.async.cg only support vectorize 16 bytes");
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
              kir::Asm::Options{/*volatile=*/true}));
    }
  }

  void handleTuringOrAmpereMma(MmaOp* mma) {
    // Constants definitions based on MMA PTX instruction documentation:
    // https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#multiply-and-accumulate-instruction-mma
    const int m = 16;
    const int n = 8;
    const int k = mma->isAmpere() ? 16 : 8;

    std::string op;
    {
      std::stringstream op_ss;
      op_ss << "mma.sync.aligned.m" << m << "n" << n << "k" << k
            << ".row.col.f32";
      if (mma->inA()->as<kir::TensorIndex>()->view()->getDataType().value() ==
          DataType::BFloat16) {
        op_ss << ".bf16.bf16";
      } else {
        op_ss << ".f16.f16";
      }
      op_ss << ".f32";
      op = op_ss.str();
    }

    int64_t split_n = mma->n() / n;
    int64_t split_k = mma->k() / k;

    // If factor == 1, then do nothing, otherwise, view array<T, n> as
    // array<array<T, n / factor>, factor>
    auto maybe_outer_split = [](DataType dtype, int64_t factor) -> DataType {
      if (factor == 1) {
        return dtype;
      }
      const auto& array = std::get<ArrayType>(dtype.type);
      return ArrayType{
          std::make_shared<DataType>(
              ArrayType{array.type, array.size / (size_t)factor}),
          (size_t)factor};
    };

    DataType accumulator_type = maybe_outer_split(mma->out()->dtype(), split_n);
    DataType a_type = maybe_outer_split(mma->inA()->dtype(), split_k);
    DataType b_type = maybe_outer_split(mma->inB()->dtype(), split_n);
    if (split_n > 1) {
      // array<array<array<T, n / split_n / split_k>, split_k>, split_n>
      auto& item_type = *std::get<ArrayType>(b_type.type).type;
      item_type = maybe_outer_split(item_type, split_k);
    } else {
      // array<array<T, n / split_k>, split_k>
      b_type = maybe_outer_split(b_type, split_k);
    }

    auto accumulator =
        IrBuilder::maybeRefCastExpr(accumulator_type, mma->out());
    auto a = IrBuilder::maybeRefCastExpr(a_type, mma->inA());
    auto b = IrBuilder::maybeRefCastExpr(b_type, mma->inB());

    for (auto in : c10::irange(split_n)) {
      auto acc =
          split_n == 1 ? accumulator : IrBuilder::getItemExpr(accumulator, in);
      auto bb = split_n == 1 ? b : IrBuilder::getItemExpr(b, in);
      for (auto ik : c10::irange(split_k)) {
        auto aa = split_k == 1 ? a : IrBuilder::getItemExpr(a, ik);
        auto bbb = split_k == 1 ? bb : IrBuilder::getItemExpr(bb, ik);
        auto mma_asm = IrBuilder::create<kir::Asm>(
            op, std::vector<Val*>{acc}, std::vector<Val*>{aa, bbb, acc});
        registerInsertBefore(mma, mma_asm);
      }
    }
    registerRemove(mma);
  }

  void handleHopperMma(MmaOp* mma) {
    NVF_ERROR(false, "Hopper MMA not supported yet");
  }

  void handle(MmaOp* mma) override {
    if (mma->isTuring() || mma->isAmpere()) {
      handleTuringOrAmpereMma(mma);
    } else if (mma->isHopper()) {
      handleHopperMma(mma);
    } else {
      NVF_ERROR(false, "Unsupported MMA architecture");
    }
  }
};

std::vector<Expr*> lowerToInlinePtx(const std::vector<Expr*>& exprs) {
  return LowerToInlinePtx{}.traverseAndInsert(exprs);
}

} // namespace nvfuser
