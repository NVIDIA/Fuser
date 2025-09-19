// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#include <device_lower/pass/staging.h>

#include <device_lower/lower2device.h>
#include <device_lower/utils.h>
#include <dispatch.h>
#include <fusion.h>
#include <instrumentation.h>
#include <ir/utils.h>
#include <ops/all_ops.h>
#include <transform_replay.h>

namespace nvfuser {

namespace {

IterDomain* getGroupedLoopID(TensorView* tv) {
  auto it = std::ranges::find_if(tv->getLoopDomain(),
                                 [] (IterDomain* id) {
                                   return id->getParallelType() == ParallelType::Group;
                                 });
  if (it != tv->getLoopDomain().end()) {
    return *it;
  } else {
    return nullptr;
  }
}

void clearGroupParallelType(TensorView* tv) {
  for (auto loop_id: tv->getLoopDomain()) {
    if (loop_id->getParallelType() == ParallelType::Group) {
      loop_id->parallelize(ParallelType::Serial);
    }
  }
}

class InputAndOutputStaging: public OptOutDispatch {
 public:
  static void run(Fusion* fusion) {
    FusionGuard fg(fusion);
    InputAndOutputStaging staging(fusion);
  }

 private:
  InputAndOutputStaging(Fusion* fusion) {
    // Traverse tvs rather than exprs since exprs may be replaced, but
    // tvs should stay the same
    auto all_tvs = fusion->allTvs();
    for (auto tv: all_tvs) {
      OptOutDispatch::dispatch(tv);
    }
  }

  using OptOutDispatch::handle;

  void handle(TensorView* tv) final {
    if (auto def = tv->definition()) {
      OptOutDispatch::dispatch(def);
    }
  }

  void handle(ArgsortOp* aop) final {
    std::cerr << aop->toString();

    auto out_tv = ir_utils::getTvOutput(aop);

    validateGrouping(out_tv);

    auto innermost_id = out_tv->getLoopDomain().back();
    bool is_grouped = innermost_id->getParallelType() == ParallelType::Group;

    Expr* replaced_aop = aop;

    // Staging inputs into registers
    for (auto inp_tv: ir_utils::filterByType<TensorView>(aop->inputs())) {

      auto copy = set(inp_tv);
      replaced_aop = ir_utils::replaceValInExprInputs(replaced_aop, inp_tv, copy);
      out_tv->setDefinition(replaced_aop);

      TransformReplay::selfReplay(out_tv->domain(), copy->domain());

      // Clear the Group type
      clearGroupParallelType(copy);

      // Fully inline except for the grouped ID
      int64_t inlining_pos = is_grouped ? -2 : -1;
      copy->inlineAt(inlining_pos);
    }

    // Stageing outputs out of registers
    for (auto out_tv : ir_utils::filterByType<TensorView>(aop->outputs())) {
      auto copy = IrBuilder::create<TensorView>(
          IrBuilder::create<TensorDomain>(out_tv->domain()),
          out_tv->dtype());
      TransformReplay::selfReplay(out_tv->domain(), copy->domain());
      
      // Use the copy instead
      replaced_aop = ir_utils::transferDefinitionToNewOutputs(replaced_aop, {copy});
      // Copy the intermedaite to the original output
      IrBuilder::create<LoadStoreOp>(LoadStoreOpType::Set, out_tv, copy);
    
      // out_tv should no longer have ParallelType::Group
      clearGroupParallelType(out_tv);
    }

    if (replaced_aop != aop) {
      // Since Fusion is actually a Kernel, the old expr is not
      // automatically removed
      aop->fusion()->removeExpr(aop);
    }
  }

  void validateGrouping(TensorView* tv) const {
    for (const auto& id : tv->getLoopDomain()) {
      if (id->getParallelType() == ParallelType::Group) {
        NVF_ERROR_EQ(id, tv->getLoopDomain().back(),
                     "ParallelType::Group must be only used at the innermost position: ",
                     tv->toString());
      }
    }
  }
};

} // namespace

void stageInputsAndOutputsOfExternFuncs(Fusion* fusion) {
  InputAndOutputStaging::run(fusion);
}

#if 0
namespace {

// Assumes:
// - For-loop structure is already built
// - Indexing is not yet done. Creates a new for-loop where indexing
// should be done.
class PrepareInputsForGroupedFuncs : public kir::ExprMutator {
 public:
  static std::vector<Expr*> run(const std::vector<Expr*>& exprs) {
    PrepareInputsForGroupedFuncs inserter(exprs);
    return inserter.exprs_;
  }

 private:
  PrepareInputsForGroupedFuncs(const std::vector<Expr*>& exprs) {
    kir::ExprMutator::traverseAndInsert(exprs);
  }

  using kir::ExprMutator::handle;

  void handle(ArgsortOp* aop) final {
    auto out_tv = ir_utils::getTvOutput(aop);
    auto inp_tv = ir_utils::getTvInput(aop);

    

    auto group_size = grouped_id->extent()->evaluate().as<int64_t>();
    std::cerr << "Grouped op: " << aop->toString() << ", group size: " << group_size << std::endl;

    auto local_inp = createCopyOfInput(inp_tv, grouped_id);
    std::cerr << "Input copy: " << local_inp->toString() << std::endl;

    auto local_inp_alloc = IrBuilder::create<kir::Allocate>(
        local_inp, local_inp->getMemoryType());
    std::cerr << "Alloc: " << local_inp_alloc->toString() << std::endl;

    auto copy_loop = createLoadLoop(grouped_id);

    auto load_op = createLoad(inp_tv, local_inp);
    copy_loop->body().push_back(load_op);

    std::cerr << "Copy loop: " << copy_loop->toString();

    registerInsertBefore(aop, copy_loop);
  }

  TensorView* createCopyOfInput(TensorView* original_input, IterDomain* grouped_id) {
    return IrBuilder::create<TensorView>(
        IrBuilder::create<TensorDomain>(std::vector<IterDomain*>{grouped_id}), original_input->dtype(), MemoryType::Local);
  }

  kir::ForLoop* createLoadLoop(IterDomain* grouped_id) {
    return IrBuilder::create<kir::ForLoop>(grouped_id);
  }

  Expr* createLoad(TensorView* original_input, TensorView* local_copy) {
    return IrBuilder::create<LoadStoreOp>(
        LoadStoreOpType::Set, local_copy, original_input);
  }

};

} // namespace

std::vector<Expr*> prepareInputsForGroupedFuncs(const std::vector<Expr*>& exprs) {
  FUSER_PERF_SCOPE("GpuLower::Lower::prepareInputsForGroupedFuncs");
  return PrepareInputsForGroupedFuncs::run(exprs);
}
#endif
} // namespace nvfuser
