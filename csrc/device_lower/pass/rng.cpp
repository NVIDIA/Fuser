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
  Val* rng_subseq;
  Val* rng_offset;
  struct InsertionInfo {
    Scope* scope = nullptr;
    ForLoop* fl = nullptr;
  };

  RNGInserter(const std::vector<Expr*>& exprs) {
    NVF_ERROR(!exprs.empty());
    auto neg_1 = IrBuilder::create<Val>(-1, DataType::Index);
    auto rng_subseq =
        IrBuilder::create<NamedScalar>("rng_subseq", DataType::Index);
    auto rng_offset =
        IrBuilder::create<NamedScalar>("rng_offset", DataType::Index);
    kir::ExprMutator::registerInsertBefore(
        exprs.front(),
        IrBuilder::create<LoadStoreOp>(
            LoadStoreOpType::Set, rng_subseq, neg_1));
    kir::ExprMutator::registerInsertBefore(
        exprs.front(),
        IrBuilder::create<LoadStoreOp>(
            LoadStoreOpType::Set, rng_offset, neg_1));
    kir::ExprMutator::traverseAndInsert(exprs);
  }

  void handle(RNGOp* rng_op) final {
    std::cout << rng_op->toString() << std::endl;
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

  return RNGInserter::insert(exprs);
}

} // namespace nvfuser
