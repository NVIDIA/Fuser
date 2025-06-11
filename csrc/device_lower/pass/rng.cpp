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
  using kir::ExprMutator::handle;
  static std::vector<Expr*> insert(const std::vector<Expr*>& exprs) {
    RNGInserter inserter(exprs);
    return inserter.exprs_;
  }

 private:
  Val* rng_subseq_ = nullptr;
  Val* rng_offset_ = nullptr;
  TensorView* rng_result_ = nullptr;
  const std::vector<Expr*>& exprs;

  struct InsertionInfo {
    Scope* scope = nullptr;
    ForLoop* fl = nullptr;
  };

  RNGInserter(const std::vector<Expr*>& _exprs) : exprs(_exprs) {
    kir::ExprMutator::traverseAndInsert(exprs);
  }

  void handle(RNGOp* rop) final {
    NVF_ERROR(!exprs.empty());

    // Set prologue if not already set
    if (rng_result_ == nullptr) {
      rng_result_ = TensorViewBuilder()
                        .shape(std::vector<int64_t>{4})
                        .dtype(DataType::UInt32)
                        .contiguity(true)
                        .build();
      rng_result_->setMemoryType(MemoryType::Local);

      auto rng_result_alloc =
          IrBuilder::create<kir::Allocate>(rng_result_, MemoryType::Local);
      kir::ExprMutator::registerInsertBefore(
          exprs.front(), rng_result_alloc, nullptr);

      auto neg_1 = IrBuilder::create<Val>(-1, DataType::Index);
      auto subseq_tuple = createAndAllocNS("rng_subseq");
      kir::ExprMutator::registerInsertBefore(
          exprs.front(), std::get<1>(subseq_tuple), nullptr);
      kir::ExprMutator::registerInsertBefore(
          exprs.front(),
          IrBuilder::create<LoadStoreOp>(
              LoadStoreOpType::Set, std::get<0>(subseq_tuple), neg_1),
          nullptr);

      rng_subseq_ = std::get<0>(subseq_tuple);

      auto offset_tuple = createAndAllocNS("rng_offset");
      kir::ExprMutator::registerInsertBefore(
          exprs.front(), std::get<1>(offset_tuple), nullptr);
      kir::ExprMutator::registerInsertBefore(
          exprs.front(),
          IrBuilder::create<LoadStoreOp>(
              LoadStoreOpType::Set, std::get<0>(offset_tuple), neg_1),
          nullptr);

      rng_offset_ = std::get<0>(offset_tuple);
    }

    auto index_tuple =
        createAndAllocNS("linear_index" + std::to_string(rop->name()));
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
                rng_subseq_, std::get<0>(rop_subseq_tuple)),
            SimplifyingIrBuilder::neExpr(
                rng_offset_, std::get<0>(rop_offset_tuple)))));

    ite->thenBody().push_back(IrBuilder::create<TernaryOp>(
        TernaryOpType::Philox,
        rng_result_,
        rop->getRNGSeedVal(),
        std::get<0>(rop_subseq_tuple),
        std::get<0>(rop_offset_tuple)));

    ite->thenBody().push_back(IrBuilder::create<LoadStoreOp>(
        LoadStoreOpType::Set, rng_subseq_, std::get<0>(rop_subseq_tuple)));

    ite->thenBody().push_back(IrBuilder::create<LoadStoreOp>(
        LoadStoreOpType::Set, rng_offset_, std::get<0>(rop_offset_tuple)));

    kir::ExprMutator::registerInsertBefore(rop, ite);
    if (rop->inputs().size() == 4) {
      auto new_rng_op = IrBuilder::create<kir::RNGOp>(
          rop->output(0),
          rng_result_,
          std::get<0>(rop_component_tuple),
          rop->dtype(),
          rop->getRNGOpType(),
          std::vector<Val*>{rop->getParameters()[0], rop->getParameters()[1]});
      kir::ExprMutator::registerReplace(rop, new_rng_op);
    } else if (rop->inputs().size() == 2) {
      auto new_rng_op = IrBuilder::create<kir::RNGOp>(
          rop->output(0),
          rng_result_,
          std::get<0>(rop_component_tuple),
          rop->dtype(),
          rop->getRNGOpType());
      kir::ExprMutator::registerReplace(rop, new_rng_op);
    } else {
      NVF_THROW(
          "Unexpected number of inputs: ",
          rop->inputs().size(),
          " for RNG operation.");
    }
  }

  std::vector<InsertionInfo> insertion_list_;
};

} // namespace

std::vector<Expr*> addRNG(const std::vector<Expr*>& exprs) {
  FUSER_PERF_SCOPE("GpuLower::Lower::addRNG");
  // Check if magic zero was even used, if not we don't have to define it or
  // update it.
  auto kernel_exprs = GpuLower::current()->kernel()->exprs();
  const bool has_rng =
      std::any_of(kernel_exprs.begin(), kernel_exprs.end(), [](Expr* expr) {
        return expr->isA<RNGOp>();
      });

  if (!has_rng) {
    return exprs;
  }

  return RNGInserter::insert(exprs);
}

} // namespace nvfuser
