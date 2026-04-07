// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#include <device_lower/pass/magic_zero.h>

#include <device_lower/lower2device.h>
#include <dispatch.h>
#include <instrumentation.h>
#include <ir/utils.h>
#include <kernel_ir_dispatch.h>

namespace nvfuser {

namespace {

class MagicZeroInserter : public kir::ExprMutator {
 public:
  static std::vector<Expr*> insert(const std::vector<Expr*>& exprs) {
    MagicZeroInserter inserter(exprs);
    return inserter.exprs_;
  }

 private:
  struct InsertionInfo {
    Scope* scope = nullptr;
    kir::ForLoop* fl = nullptr;
  };

  MagicZeroInserter(const std::vector<Expr*>& exprs) {
    NVF_ERROR(!exprs.empty());
    kir::ExprMutator::registerInsertBefore(
        exprs.front(), IrBuilder::create<kir::InitMagicZero>(), nullptr);
    kir::ExprMutator::traverseAndInsert(exprs);
  }

  void handle(kir::ForLoop* fl) final {
    if (fl->isUnrolled()) {
      if (scope_.empty()) {
        kir::ExprMutator::registerInsertAfter(
            fl, IrBuilder::create<kir::UpdateMagicZero>());
      } else {
        NVF_ERROR(
            !scope_.back()->exprs().empty(), "Not expecting an empty loop.");
        kir::ExprMutator::registerInsertAfter(
            fl, IrBuilder::create<kir::UpdateMagicZero>(), scope_.back());
      }
    } else {
      kir::ExprMutator::handle(fl);
    }
  }

  std::vector<InsertionInfo> insertion_list_;
};

} // namespace

std::vector<Expr*> insertMagicZero(const std::vector<Expr*>& exprs) {
  FUSER_PERF_SCOPE("GpuLower::Lower::insertMagicZero");
  if (!GpuLower::current()->isNvFuserZeroEnabled()) {
    return exprs;
  }
  // Check if magic zero was even used, if not we don't have to define it or
  // update it.
  const auto gpu_lower = GpuLower::current();
  auto kernel = gpu_lower->kernel();
  const bool has_magic_zero =
      std::any_of(kernel->vals().begin(), kernel->vals().end(), [](Val* val) {
        return isMagicZero(val);
      });

  if (!has_magic_zero) {
    return exprs;
  }

  return MagicZeroInserter::insert(exprs);
}

bool isMagicZero(const Val* val) {
  if (!val->isA<NamedScalar>()) {
    return false;
  }
  auto ns = val->as<NamedScalar>();
  return ns->dtype() == DataType::Index &&
      ns->name() == std::string(kMagicZeroName);
}

bool isProtectedWithMagicZero(const Val* val) {
  if (val->definition() == nullptr || !val->definition()->isA<BinaryOp>()) {
    return false;
  }
  auto bop = val->definition()->as<BinaryOp>();
  return bop->getBinaryOpType() == BinaryOpType::Add && isMagicZero(bop->rhs());
}

Val* maybeUnwrapMagicZero(Val* val) {
  if (isProtectedWithMagicZero(val)) {
    return val->definition()->as<BinaryOp>()->lhs();
  } else {
    return val;
  }
}

bool needsMagicZero(
    kir::ForLoop* loop,
    IterDomain* reference_domain,
    Val* ind) {
  if (!GpuLower::current()->isNvFuserZeroEnabled()) {
    return false;
  }

  NVF_ERROR(ind != nullptr);
  NVF_ERROR(reference_domain != nullptr);

  if (ind->isConstScalar()) {
    return false;
  }

  bool ref_dom_simple = reference_domain->definition() != nullptr;
  bool ind_simple = ind->definition() != nullptr && !ind->isZeroInt();

  return loop->isUnrolled() && (!ref_dom_simple || !ind_simple);
}

} // namespace nvfuser
