// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#include <device_lower/pass/magic_zero.h>

#include <device_lower/analysis/index_compute.h>
#include <device_lower/lower2device.h>
#include <dispatch.h>
#include <instrumentation.h>
#include <ir/utils.h>
#include <kernel_ir_dispatch.h>

namespace nvfuser {

namespace {

class RNGInserter : public kir::ExprMutator {
 public:
  static std::vector<Expr*> insert(const std::vector<Expr*>& exprs) {
    RNGInserter inserter(exprs);
    return inserter.exprs_;
  }

 private:
  Val* rng_subseq = nullptr;
  Val* rng_offset = nullptr;
  const std::vector<Expr*>& exprs;

  struct InsertionInfo {
    Scope* scope = nullptr;
    ForLoop* fl = nullptr;
  };

  RNGInserter(const std::vector<Expr*>& _exprs) : exprs(_exprs) {
    kir::ExprMutator::traverseAndInsert(exprs);
  }

  void handle(RNGOp* rng_op) final {
    if (rng_subseq == nullptr) {
      NVF_ERROR(!exprs.empty());
      auto neg_1 = IrBuilder::create<Val>(-1, DataType::Index);
      rng_subseq =
          IrBuilder::create<NamedScalar>("rng_subseq", DataType::Index);
      rng_offset =
          IrBuilder::create<NamedScalar>("rng_offset", DataType::Index);
      // std::cout << rng_subseq->name() << std::endl;
      // std::cout << rng_offset->name() << std::endl;

      auto alloc0 = IrBuilder::create<kir::Allocate>(
          rng_subseq,
          MemoryType::Local,
          GpuLower::current()->kernel()->oneVal());
      auto alloc1 = IrBuilder::create<kir::Allocate>(
          rng_offset,
          MemoryType::Local,
          GpuLower::current()->kernel()->oneVal());
      auto expr0 = IrBuilder::create<LoadStoreOp>(
          LoadStoreOpType::Set, rng_subseq, neg_1);
      auto expr1 = IrBuilder::create<LoadStoreOp>(
          LoadStoreOpType::Set, rng_offset, neg_1);
      // std::cout << expr0->toString() << std::endl;
      // std::cout << expr1->toString() << std::endl;
      // std::cout << exprs.front()->toString() << std::endl;
      kir::ExprMutator::registerInsertBefore(exprs.front(), alloc0, nullptr);
      kir::ExprMutator::registerInsertBefore(exprs.front(), expr0, nullptr);
      kir::ExprMutator::registerInsertBefore(exprs.front(), alloc1, nullptr);
      kir::ExprMutator::registerInsertBefore(exprs.front(), expr1, nullptr);
    }
    // std::cout << rng_op->toString() << std::endl;
    // auto linear_index = rng_op->getPhiloxIndex();
    // auto multiple =  rng_op->getPhiloxMultiple();
    // auto rng_subseq = SimplifyingIrBuilder::div(linear_index, multiple);
    // auto rng_component = SimplifyingIrBuilder::mod(linear_index, multiple);
    // auto rng_offset = rng_op->getRNGOffsetVal();

    //  nvfuser_index_t rng_offset215 = (((ptr2 == nullptr) ? i3 : ((*ptr2) +
    //  i3)) / 4LL);
    //   if (rng_subseq != rng_subseq215 || rng_offset != rng_offset215) {
    //     rng_result = philox(((ptr0 == nullptr) ? i1 : (*ptr0)),
    //     rng_subseq215, rng_offset215); rng_subseq = rng_subseq215; rng_offset
    //     = rng_offset215;
    //   }
    //   T1[i5] = rng_uniformf(rng_result, rng_component215);
    // }

    // if (fl->isUnrolled()) {
    //   if (scope_.empty()) {
    //     kir::ExprMutator::registerInsertAfter(
    //         fl, IrBuilder::create<kir::UpdateMagicZero>());
    //   } else {
    //     NVF_ERROR(
    //         !scope_.back()->exprs().empty(), "Not expecting an empty loop.");
    //     kir::ExprMutator::registerInsertAfter(
    //         fl, IrBuilder::create<kir::UpdateMagicZero>(), scope_.back());
    //   }
    // } else {
    //   kir::ExprMutator::handle(fl);
    // }
    // NVF_THROW("TEST");
  }

  std::vector<InsertionInfo> insertion_list_;
};

} // namespace

std::vector<Expr*> addRNG(const std::vector<Expr*>& exprs) {
  FUSER_PERF_SCOPE("GpuLower::Lower::addRNG");
  // Check if magic zero was even used, if not we don't have to define it or
  // update it.
  const auto gpu_lower = GpuLower::current();
  auto kernel = gpu_lower->kernel();
  const bool has_rng = std::any_of(
      kernel->exprs().begin(), kernel->exprs().end(), [](Expr* expr) {
        return expr->isA<RNGOp>();
      });

  if (!has_rng) {
    return exprs;
  }
  auto exprs_ = RNGInserter::insert(exprs);
  std::cout << "====================" << std::endl;
  for (auto expr : exprs_) {
    std::cout << expr->toString() << std::endl;
  }
  std::cout << "====================" << std::endl;
  // NVF_THROW("throw");
  return exprs_;
}

} // namespace nvfuser
