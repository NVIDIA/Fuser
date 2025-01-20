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

std::tuple<Val*, Expr*> createAndAllocNS(
    std::string name,
    DataType dtype = DataType::Index) {
  Val* val = IrBuilder::create<NamedScalar>(name, dtype);
  auto alloc = IrBuilder::create<kir::Allocate>(
      val, MemoryType::Local, GpuLower::current()->kernel()->oneVal());
  return std::make_tuple(val, alloc);
}

class RNGInserter : public kir::ExprMutator {
 public:
  static std::vector<Expr*> insert(const std::vector<Expr*>& exprs) {
    RNGInserter inserter(exprs);
    return inserter.exprs_;
  }

 private:
  Val* rng_subseq = nullptr;
  Val* rng_offset = nullptr;
  TensorView* rng_result = nullptr;
  const std::vector<Expr*>& exprs;

  struct InsertionInfo {
    Scope* scope = nullptr;
    ForLoop* fl = nullptr;
  };

  RNGInserter(const std::vector<Expr*>& _exprs) : exprs(_exprs) {
    kir::ExprMutator::traverseAndInsert(exprs);
  }

  void handle(RNGOp* rop) final {
    // Set prologue if not already set
    if (rng_subseq == nullptr) {
      NVF_ERROR(!exprs.empty());
      auto neg_1 = IrBuilder::create<Val>(-1, DataType::Index);
      auto subseq_tuple = createAndAllocNS("rng_subseq");
      kir::ExprMutator::registerInsertBefore(
          exprs.front(), std::get<1>(subseq_tuple), nullptr);
      kir::ExprMutator::registerInsertBefore(
          exprs.front(),
          IrBuilder::create<LoadStoreOp>(
              LoadStoreOpType::Set, std::get<0>(subseq_tuple), neg_1),
          nullptr);

      rng_subseq = std::get<0>(subseq_tuple);

      auto offset_tuple = createAndAllocNS("rng_offset");
      kir::ExprMutator::registerInsertBefore(
          exprs.front(), std::get<1>(offset_tuple), nullptr);
      kir::ExprMutator::registerInsertBefore(
          exprs.front(),
          IrBuilder::create<LoadStoreOp>(
              LoadStoreOpType::Set, std::get<0>(offset_tuple), neg_1),
          nullptr);

      rng_offset = std::get<0>(offset_tuple);

      rng_result = TensorViewBuilder()
                       .shape(std::vector<int64_t>{4})
                       .dtype(DataType::UInt64)
                       .contiguity(true)
                       .build();
      rng_result->setMemoryType(MemoryType::Local);

      auto rng_result_alloc =
          IrBuilder::create<kir::Allocate>(rng_result, MemoryType::Local);
      kir::ExprMutator::registerInsertBefore(
          exprs.front(), rng_result_alloc, nullptr);
    }

    auto index_tuple =
        createAndAllocNS("liner_index" + std::to_string(rop->name()));
    kir::ExprMutator::registerInsertBefore(rop, std::get<1>(index_tuple));
    kir::ExprMutator::registerInsertBefore(
        rop,
        IrBuilder::create<LoadStoreOp>(
            LoadStoreOpType::Set,
            std::get<0>(index_tuple),
            rop->getPhiloxIndex()));

    auto multiple =
        IrBuilder::create<Val>(rop->getPhiloxMultiple(), DataType::Index);

    auto rop_subseq_tuple =
        createAndAllocNS("rng_subseq" + std::to_string(rop->name()));
    kir::ExprMutator::registerInsertBefore(rop, std::get<1>(rop_subseq_tuple));
    kir::ExprMutator::registerInsertBefore(
        rop,
        IrBuilder::create<BinaryOp>(
            BinaryOpType::Div,
            std::get<0>(rop_subseq_tuple),
            std::get<0>(index_tuple),
            multiple));

    auto rop_component_tuple =
        createAndAllocNS("rng_component" + std::to_string(rop->name()));
    kir::ExprMutator::registerInsertBefore(
        rop, std::get<1>(rop_component_tuple));
    kir::ExprMutator::registerInsertBefore(
        rop,
        IrBuilder::create<BinaryOp>(
            BinaryOpType::Mod,
            std::get<0>(rop_component_tuple),
            std::get<0>(index_tuple),
            multiple));

    auto rop_offset_tuple =
        createAndAllocNS("rng_offset" + std::to_string(rop->name()));
    kir::ExprMutator::registerInsertBefore(rop, std::get<1>(rop_offset_tuple));
    kir::ExprMutator::registerInsertBefore(
        rop,
        IrBuilder::create<LoadStoreOp>(
            LoadStoreOpType::Set,
            std::get<0>(rop_offset_tuple),
            rop->getRNGOffsetVal()));

    kir::IfThenElse* ite = IrBuilder::create<kir::IfThenElse>(
        IrBuilder::create<kir::Predicate>(SimplifyingIrBuilder::logicalOrExpr(
            SimplifyingIrBuilder::neExpr(
                rng_subseq, std::get<0>(rop_subseq_tuple)),
            SimplifyingIrBuilder::neExpr(
                rng_offset, std::get<0>(rop_offset_tuple)))));
    ite->thenBody().push_back(IrBuilder::create<LoadStoreOp>(
        LoadStoreOpType::Set, rng_subseq, std::get<0>(rop_subseq_tuple)));

    ite->thenBody().push_back(IrBuilder::create<LoadStoreOp>(
        LoadStoreOpType::Set, rng_offset, std::get<0>(rop_offset_tuple)));

    kir::ExprMutator::registerInsertBefore(rop, ite);

    kir::ExprMutator::registerInsertBefore(
        rop,
        IrBuilder::create<TernaryOp>(
            TernaryOpType::Philox,
            IrBuilder::create<NamedScalar>("rng_result", DataType::Index),
            rop->getRNGSeedVal(),
            rng_subseq,
            rng_offset));

    // auto rop_component =
    //     createAndAllocNS("rng_component" + std::to_string(rop->name()));

    // auto rng_subseq = SimplifyingIrBuilder::div(linear_index, multiple);
    // auto rng_component = SimplifyingIrBuilder::mod(linear_index, multiple);
    // auto rng_offset = rop->getRNGOffsetVal();

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
