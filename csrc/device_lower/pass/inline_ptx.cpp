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
#include <kernel_ir.h>
#include <kernel_ir_dispatch.h>
#include <scheduler/mma_utils.h>

#include <sstream>

namespace nvfuser {

class LowerToInlinePtx : public kir::ExprMutator {
 private:
  // create a new predicate with the inverted value, used for cpAsync
  kir::Predicate* invertedPredicate(const kir::Predicate* predicate) {
    auto pred = predicate->value();
    Val* invert = SimplifyingIrBuilder::logicalNotExpr(pred);
    return IrBuilder::create<kir::Predicate>(invert);
  }

 protected:
  using ExprMutator::handle;

  void handle(kir::AsyncCommit* commit) final {
    registerReplace(
        commit,
        IrBuilder::create<kir::Asm>(
            commit->ptx(),
            std::vector<Val*>{},
            std::vector<Val*>{},
            kir::Asm::Options{/*volatile=*/true}));
  }

  void handle(kir::AsyncWait* wait) final {
    if (wait->asyncOpType() == AsyncOpType::CpAsync &&
        wait->keepStages() == 0) {
      // cp.async uses wait_all for zero keep stages, other instructions uses a
      // unified interface for all keep stages.
      registerReplace(
          wait,
          IrBuilder::create<kir::Asm>(
              wait->ptx(),
              std::vector<Val*>{},
              std::vector<Val*>{},
              kir::Asm::Options{/*volatile=*/true}));
    } else {
      registerReplace(
          wait,
          IrBuilder::create<kir::Asm>(
              wait->ptx(),
              std::vector<Val*>{},
              std::vector<Val*>{IrBuilder::create<Val>(wait->keepStages())},
              kir::Asm::Options{/*volatile=*/true, /*memory=*/wait->memory()}));
    }
  }

  void handle(LoadStoreOp* ldst) final {
    if (ir_utils::isLdMatrixOp(ldst)) {
      std::stringstream ss;
      ss << "ldmatrix.sync.aligned.x"
         << std::get<ArrayType>(ldst->out()->dtype().type).size;
      if (mma_utils::isLdMatrixTranspose(ldst)) {
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
    } else if (ir_utils::isStMatrixOp(ldst)) {
      std::stringstream ss;
      ss << "stmatrix.sync.aligned.x"
         << std::get<ArrayType>(ldst->in()->dtype().type).size;
      ss << ".m8n8.shared.b16";
      registerReplace(
          ldst,
          // stmatrix has no output.
          IrBuilder::create<kir::Asm>(
              ss.str(),
              std::vector<Val*>{},
              std::vector<Val*>{ldst->out(), ldst->in()},
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
                  invertedPredicate(ldst->predicate())},
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
    // Reference:
    // https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#asynchronous-warpgroup-level-matrix-multiply-accumulate-instructions

    // Do MMA
    std::stringstream inst_ss;
    inst_ss << "wgmma.mma_async.sync.aligned.m" << mma->m() << "n" << mma->n()
            << "k" << mma->k() << ".f32";
    if (mma->inA()->as<kir::TensorIndex>()->view()->getDataType().value() ==
        DataType::BFloat16) {
      inst_ss << ".bf16.bf16";
    } else {
      inst_ss << ".f16.f16";
    }
    bool a_on_smem =
        mma->inA()->as<kir::TensorIndex>()->view()->getMemoryType() ==
        MemoryType::Shared;
    std::vector<Val*> inputs{
        a_on_smem ? mma->inA()->as<kir::TensorIndex>()->index() : mma->inA(),
        mma->inB()->as<kir::TensorIndex>()->index(),
        /*scaleD=*/IrBuilder::create<Val>(true),
        /*scaleA=*/IrBuilder::create<Val>(1, DataType::Int32),
        /*scaleB=*/IrBuilder::create<Val>(1, DataType::Int32)};
    auto layout = lower_utils::getMmaLayout(mma);
    if (a_on_smem) {
      // tnspA
      if (layout[0] == UnitDim::K) {
        inputs.push_back(IrBuilder::create<Val>(0, DataType::Int32));
      } else {
        inputs.push_back(IrBuilder::create<Val>(1, DataType::Int32));
      }
    }
    // tnspB
    if (layout[1] == UnitDim::K) {
      inputs.push_back(IrBuilder::create<Val>(0, DataType::Int32));
    } else {
      inputs.push_back(IrBuilder::create<Val>(1, DataType::Int32));
    }
    registerInsertBefore(
        mma,
        IrBuilder::create<kir::Asm>(
            inst_ss.str(),
            std::vector<Val*>{mma->out()},
            inputs,
            kir::Asm::Options{
                /*volatile=*/true,
                /*memory=*/false,
                /*readable_outputs=*/{0}}));

    // The above call is asynchronous, so we need to wait to prevent a data race
    // TODO: Why is it safe to not always use zero here?
    CircularBufferOptions cb_opts =
        mma->inA()->as<kir::TensorIndex>()->view()->circularBufferOptions();
    auto* commit = IrBuilder::create<kir::AsyncCommit>(AsyncOpType::WgMma);
    auto* wait = IrBuilder::create<kir::AsyncWait>(
        AsyncOpType::WgMma,
        // If cb_opts.stage - cb_opts.prefetch == 0, then keep_stages will be
        // -1, which is invalid.
        std::max(0L, /*keep_stages=*/cb_opts.stage - cb_opts.prefetch - 1L));

    registerInsertBefore(mma, commit);
    registerInsertBefore(mma, wait);
    registerRemove(mma);

    // These are needed for actually converting the nodes above into kir::Asm
    // nodes properly
    handle(commit);
    handle(wait);
  }

  void handle(MmaOp* mma) final {
    if (mma->isTuring() || mma->isAmpere()) {
      handleTuringOrAmpereMma(mma);
    } else if (mma->isHopper()) {
      handleHopperMma(mma);
    } else {
      NVF_THROW("Unsupported MMA architecture");
    }
  }

  void handle(kir::FenceAsyncProxy* fence) final {
    registerReplace(
        fence,
        IrBuilder::create<kir::Asm>(
            "fence.proxy.async",
            std::vector<Val*>{},
            std::vector<Val*>{},
            kir::Asm::Options{/*volatile=*/true}));
  }

  void handle(kir::WgMmaFence* fence) final {
    registerReplace(
        fence,
        IrBuilder::create<kir::Asm>(
            "wgmma.fence.sync.aligned",
            std::vector<Val*>{},
            std::vector<Val*>{},
            kir::Asm::Options{/*volatile=*/true}));
  }

  void handle(kir::SetMaxNReg* maxnreg) final {
    std::string ptx = (maxnreg->increaseRegisters())
        ? "setmaxnreg.inc.sync.aligned.u32"
        : "setmaxnreg.dec.sync.aligned.u32";
    registerReplace(
        maxnreg,
        IrBuilder::create<kir::Asm>(
            ptx,
            std::vector<Val*>{},
            std::vector<Val*>{maxnreg->numberOfRegisters()},
            kir::Asm::Options{/*volatile=*/true}));
  }
};

std::vector<Expr*> lowerToInlinePtx(const std::vector<Expr*>& exprs) {
  return LowerToInlinePtx{}.traverseAndInsert(exprs);
}

} // namespace nvfuser
