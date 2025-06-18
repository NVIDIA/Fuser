// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on

#include <device_lower/lower2device.h>
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
              kir::Asm::Options{
                  /*volatile=*/true,
                  /*memory=*/wait->memory(),
                  /*readable_outputs=*/{},
                  /*immediate_inputs=*/{0}}));
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
      auto vec_size = ir_utils::getVectorizeSize(out_tv) *
          dataTypeSizeByte(out_tv->dtype());
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
              kir::Asm::Options{
                  /*volatile=*/true,
                  /*memory=*/false,
                  /*readable_outputs=*/{},
                  /*immediate_inputs=*/{2}}));
    } else if (ldst->opType() == LoadStoreOpType::LdTMem) {
      const auto& tmem_info = GpuLower::current()->tmemInfo();
      std::stringstream ptx_ss;
      ptx_ss << "tcgen05.ld.sync.aligned."
             << tmem_info.load_data_path.at(ir_utils::getTvInput(ldst)) << ".x"
             << ir_utils::getTMemLdStVectorizeSize(ir_utils::getTvOutput(ldst))
             << ".b32";
      registerReplace(
          ldst,
          IrBuilder::create<kir::Asm>(
              ptx_ss.str(),
              std::vector<Val*>{ldst->out()},
              std::vector<Val*>{ldst->in()}));
      auto wait_ptx = "tcgen05.wait::ld.sync.aligned";
      registerInsertAfter(
          ldst,
          IrBuilder::create<kir::Asm>(
              wait_ptx,
              std::vector<Val*>{},
              std::vector<Val*>{},
              kir::Asm::Options{/*volatile=*/true}));
    } else if (ldst->opType() == LoadStoreOpType::StTMem) {
      NVF_ERROR(
          ir_utils::getTvInput(ldst)->getMemoryType() == MemoryType::Local,
          "StTMem requires write from register to tmem, ldst: ",
          ldst->toString());

      const auto& tmem_info = GpuLower::current()->tmemInfo();
      std::stringstream ptx_ss;
      ptx_ss << "tcgen05.st.sync.aligned."
             << tmem_info.store_data_path.at(ir_utils::getTvOutput(ldst))
             << ".x"
             << ir_utils::getTMemLdStVectorizeSize(ir_utils::getTvOutput(ldst))
             << ".b32";
      registerReplace(
          ldst,
          IrBuilder::create<kir::Asm>(
              ptx_ss.str(),
              std::vector<Val*>{},
              std::vector<Val*>{ldst->out(), ldst->in()},
              kir::Asm::Options{/*volatile=*/true}));
      auto wait_ptx = "tcgen05.wait::st.sync.aligned";
      registerInsertAfter(
          ldst,
          IrBuilder::create<kir::Asm>(
              wait_ptx,
              std::vector<Val*>{},
              std::vector<Val*>{},
              kir::Asm::Options{/*volatile=*/true}));
    } else if (ldst->opType() == LoadStoreOpType::SmemToTmem) {
      // Copy from smem to tmem using tcgen05.cp 
      
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

    for (auto in : arange(split_n)) {
      auto acc =
          split_n == 1 ? accumulator : IrBuilder::getItemExpr(accumulator, in);
      auto bb = split_n == 1 ? b : IrBuilder::getItemExpr(b, in);
      for (auto ik : arange(split_k)) {
        auto aa = split_k == 1 ? a : IrBuilder::getItemExpr(a, ik);
        auto bbb = split_k == 1 ? bb : IrBuilder::getItemExpr(bb, ik);
        auto mma_asm = IrBuilder::create<kir::Asm>(
            op, std::vector<Val*>{acc}, std::vector<Val*>{aa, bbb, acc});
        registerInsertBefore(mma, mma_asm);
      }
    }
    registerRemove(mma);
  }

  // Determine if we want to do D = D + A * B or D = A * B.
  // We want to do the latter for the first iteration, and the former for the
  // rest.
  Val* getUseInputAcc(MmaOp* mma) {
    Val* dont_use_input_acc = mma->fusion()->trueVal();
    std::vector<IterDomain*> reduction_ids;
    for (IterDomain* id : ir_utils::getTvOutput(mma)->getLoopDomain()) {
      if (id->isReduction()) {
        reduction_ids.push_back(id);
      }
    }
    for (auto fl : for_loops_) {
      // Skip non-reduction loops.
      if (!std::ranges::any_of(reduction_ids, [fl](IterDomain* id) {
            return GpuLower::current()
                ->idModel()
                .idGraph(IdMappingMode::LOOP)
                .disjointValSets()
                .strictAreMapped(fl->iter_domain(), id);
          })) {
        continue;
      }
      // The Epilogue loop is never the first iteration.
      if (fl->circularBufferLoopStage() == CircularBufferLoopStage::Epilog) {
        dont_use_input_acc = mma->fusion()->falseVal();
        break;
      }
      // Skip trivial loops as they are always the first iteration.
      if (fl->isTrivial()) {
        continue;
      }
      Val* loop_index = GpuLower::current()->tensorIndexer().getLoopIndex(
          fl->iter_domain(), for_loops_);
      dont_use_input_acc = SimplifyingIrBuilder::logicalAndExpr(
          dont_use_input_acc,
          SimplifyingIrBuilder::eqExpr(loop_index, fl->start()));
    }
    return SimplifyingIrBuilder::logicalNotExpr(dont_use_input_acc);
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
        /*scaleD=*/getUseInputAcc(mma),
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
                /*readable_outputs=*/{0},
                /*immediate_inputs=*/{3, 4, 5, 6}}));
    registerRemove(mma);
  }

  void handleBlackwellMma(MmaOp* mma) {
    // Reference:
    // https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#tensorcore-5th-generation-of-mma-instructions

    NVF_ERROR(
        mma->isBlackwell1CTA(), "Currently only supports 1 CTA Blackwell MMA");

    // Create instruction descriptor
    // https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#tcgen05-instruction-descriptor
    Val* accumulator_dtype =
        IrBuilder::create<Val>(1LL << 4LL, DataType::UInt32);
    Val* a_dtype = IrBuilder::create<Val>(
        (mma->inA()->dtype() == DataType::BFloat16 ? 1LL : 0LL) << 7LL,
        DataType::UInt32);
    Val* b_dtype = IrBuilder::create<Val>(
        (mma->inB()->dtype() == DataType::BFloat16 ? 1LL : 0LL) << 10LL,
        DataType::UInt32);

    auto layout = lower_utils::getMmaLayout(mma);
    Val* tnspA = IrBuilder::create<Val>(
        (layout[0] == UnitDim::K ? 0LL : 1LL) << 15LL, DataType::UInt32);
    Val* tnspB = IrBuilder::create<Val>(
        (layout[1] == UnitDim::K ? 0LL : 1LL) << 16LL, DataType::UInt32);

    Val* n =
        IrBuilder::create<Val>((mma->n() >> 3LL) << 17LL, DataType::UInt32);
    Val* m =
        IrBuilder::create<Val>((mma->m() >> 4LL) << 24LL, DataType::UInt32);

    Val* idesc = SimplifyingIrBuilder::bitwiseOrExpr(
        SimplifyingIrBuilder::bitwiseOrExpr(
            accumulator_dtype,
            SimplifyingIrBuilder::bitwiseOrExpr(a_dtype, b_dtype)),
        SimplifyingIrBuilder::bitwiseOrExpr(
            SimplifyingIrBuilder::bitwiseOrExpr(tnspA, tnspB),
            SimplifyingIrBuilder::bitwiseOrExpr(n, m)));

    // Switch between C = A * B and C = A * B + C.
    Val* enable_input_d = getUseInputAcc(mma);

    // Do MMA
    registerReplace(
        mma,
        IrBuilder::create<kir::Asm>(
            "tcgen05.mma.cta_group::1.kind::f16",
            std::vector<Val*>{},
            std::vector<Val*>{
                mma->out(),
                mma->inA()->as<kir::TensorIndex>()->index(),
                mma->inB()->as<kir::TensorIndex>()->index(),
                idesc,
                enable_input_d,
            },
            kir::Asm::Options{
                /*volatile=*/true,
                /*memory=*/false,
                /*readable_outputs=*/{0}}));
  }

  void handle(MmaOp* mma) final {
    if (mma->isTuring() || mma->isAmpere()) {
      handleTuringOrAmpereMma(mma);
    } else if (mma->isHopper()) {
      handleHopperMma(mma);
    } else if (mma->isBlackwell()) {
      handleBlackwellMma(mma);
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
            kir::Asm::Options{
                /*volatile=*/true,
                /*memory=*/false,
                /*readable_outputs=*/{},
                /*immediate_inputs=*/{0}}));
  }

  void handle(kir::AllocTMem* alloc) final {
    registerReplace(
        alloc,
        IrBuilder::create<kir::Asm>(
            "tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32",
            std::vector<Val*>{},
            std::vector<Val*>{alloc->address(), alloc->numColumns()},
            kir::Asm::Options{/*volatile=*/true}));
  }
};

std::vector<Expr*> lowerToInlinePtx(const std::vector<Expr*>& exprs) {
  return LowerToInlinePtx{}.traverseAndInsert(exprs);
}

} // namespace nvfuser
