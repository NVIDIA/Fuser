// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#include <device_lower/analysis/bank_conflict.h>

#include <dynamic_type.h>
#include <expr_evaluator.h>
#include <ir/utils.h>
#include <kernel_ir.h>
#include <kernel_ir_dispatch.h>
#include <type.h>

#include <unordered_set>

namespace nvfuser {

namespace {

bool isSmemTensorIndex(Val* value) {
  return value->isA<kir::TensorIndex>() &&
      value->as<kir::TensorIndex>()->view()->getMemoryType() ==
      MemoryType::Shared;
}

int64_t getVectorizeSize(kir::TensorIndex* ti) {
  for (auto id : ti->view()->getLeafDomain()) {
    if (!isParallelTypeVectorize(id->getParallelType())) {
      continue;
    }

    TORCH_INTERNAL_ASSERT(
        id->extent()->isConstInt(),
        "Could not evaluate constant value bound to vectorized dim.");

    return id->extent()->evaluateInt();
  }
  return 1;
}

// Sometimes, the index can have pointer type like:
//   BaseAddress(T7) + 16 * threadIdx.x + 8192 * i487 + 2048 * i481
// For this case, we need to replace BaseAddress(T7) with 0 because expression
// evaluator can not handle BaseAddress(T7)
Val* replaceBaseAddrWithZero(Val* index) {
  std::unordered_map<Val*, Val*> replacement_map;
  std::vector<Val*> to_visit{index};
  Val* zero = index->container()->zeroVal();
  while (!to_visit.empty()) {
    auto back = to_visit.back();
    to_visit.pop_back();
    auto def = back->definition();
    if (def == nullptr) {
      continue;
    }
    if (def->isA<kir::BaseAddress>()) {
      replacement_map.emplace(back, zero);
      continue;
    }
    to_visit.insert(to_visit.end(), def->inputs().begin(), def->inputs().end());
  }
  if (replacement_map.empty()) {
    return index;
  }
  FusionGuard guard(index->fusion());
  return ir_utils::replaceValInIndexVal(index, replacement_map);
}

inline int64_t getPhaseSize(int64_t word_size_bytes) {
  if (word_size_bytes == 16) {
    return 8;
  }
  if (word_size_bytes == 8) {
    return 16;
  }
  return 32;
}

// Doc for ldmatrix can be found at:
// https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#warp-level-matrix-load-instruction-ldmatrix
//
// According to this doc, ldmatrix always do vectorize 8 load, which might be
// different from the consumer vectorization factor. The vectorization factor of
// the ldmatrix consumer is not the load vector size (which is always 8),
// instead, it is the number of items each thread will eventually get. This
// vectorization factor determines the .num modifier of ldmatrix, which further
// determines the number of addresses used by ldmatrix. For ldmatrix.x1, there
// are 8 addresses. For ldmatrix.x2, there are 16 addresses. For ldmatrix.x4,
// there are 32 addresses.
int64_t getLdMatrixNumThreads(int64_t word_size) {
  switch (word_size) {
    case 2:
      // If the consumer has vector 2, then each thread of the warp get 2 items.
      // So there are in total 32*2 = 64 items for the warp. Each vector has 8
      // elements, so there are 64/8 = 8 vectors, i.e. this is a .x1 ldmatrix
      // and the first 8 threads contain useful addresses.
      return 8;
    case 4:
      // If the consumer has vector 4, then each thread of the warp get 4 items.
      // So there are in total 32*4 = 128 items for the warp. Each vector has 8
      // elements, so there are 128/8 = 16 vectors, i.e. this is a .x2 ldmatrix
      // and the first 16 threads contain useful addresses.
      return 16;
    case 8:
      // If the consumer has vector 8, then each thread of the warp get 8 items.
      // So there are in total 32*8 = 256 items for the warp. Each vector has 8
      // elements, so there are 256/8 = 32 vectors, i.e. this is a .x4 ldmatrix
      // and all the 32 threads contain useful addresses.
      return 32;
    default:
      TORCH_INTERNAL_ASSERT(false, "Invalid word size for ldmatrix");
  }
}

std::vector<int64_t> evaluateAddressesOnFirstPhase(
    const std::vector<kir::ForLoop*>& for_loops,
    ExpressionEvaluator expr_eval_common,
    LoadStoreOp* ldst,
    bool is_producer) {
  auto bdimx = expr_eval_common.evaluate(ParallelType::TIDx);
  auto bdimy = expr_eval_common.evaluate(ParallelType::TIDy);
  auto bdimz = expr_eval_common.evaluate(ParallelType::TIDz);

  std::vector<int64_t> addresses;
  auto consumer = ldst->output(0)->as<kir::TensorIndex>();
  auto ti = (is_producer ? ldst->input(0)->as<kir::TensorIndex>() : consumer);
  int64_t word_size = -1;
  int64_t num_threads = -1;
  if (ir_utils::isLdMatrixOp(ldst)) {
    // See the comment of getLdMatrixNumThreads for why ldmatrix is handled
    // differently.
    word_size = 8;
    num_threads = getLdMatrixNumThreads(getVectorizeSize(consumer));
  } else {
    word_size = getVectorizeSize(consumer);
    num_threads = (bdimx ? bdimx->as<int64_t>() : 1) *
        (bdimy ? bdimy->as<int64_t>() : 1) * (bdimz ? bdimz->as<int64_t>() : 1);
  }
  int64_t dtype_size = (int64_t)dataTypeSize(*(ti->getDataType()));
  int64_t word_size_bytes = dtype_size * word_size;
  int64_t phase_size =
      std::min(num_threads, getPhaseSize((int64_t)word_size_bytes));

  for (int64_t linear_tidx : c10::irange(phase_size)) {
    int64_t tidx = linear_tidx;
    int64_t tidy = 0;
    int64_t tidz = 0;
    if (bdimx.has_value()) {
      tidy = tidx / bdimx->as<int64_t>();
      tidx = tidx % bdimx->as<int64_t>();
    }
    if (bdimy.has_value()) {
      tidz = tidy / bdimy->as<int64_t>();
      tidy = tidy % bdimy->as<int64_t>();
    }
    // make a copy of the expression evaluator
    ExpressionEvaluator expr_eval = expr_eval_common;
    expr_eval.bind("threadIdx.x", tidx);
    expr_eval.bind("threadIdx.y", tidy);
    expr_eval.bind("threadIdx.z", tidz);
    for (auto fl : for_loops) {
      if (fl->index()->isA<NamedScalar>()) {
        auto ns = fl->index()->as<NamedScalar>();
        TORCH_INTERNAL_ASSERT(
            ns->isThreadIdx() || ns->isBlockIdx(), "unknow loop index");
      } else {
        auto start = expr_eval.evaluate(fl->start())->as<int64_t>();
        expr_eval.bind(fl->index(), start);
      }
    }
    auto index_val = replaceBaseAddrWithZero(ti->index());
    int64_t index = expr_eval.evaluate(index_val)->as<int64_t>();
    if (ir_utils::isLdMatrixOp(ldst) || ir_utils::isCpAsyncOp(ldst)) {
      addresses.emplace_back(index);
    } else {
      addresses.emplace_back(index * dtype_size);
    }
  }
  return addresses;
}

int getConflictWays(const std::vector<int64_t>& addresses) {
  using long_set = std::unordered_set<int64_t>;
  std::array<long_set, 32> words_by_bank;
  for (auto addr : addresses) {
    int64_t word = addr / 4;
    int64_t bank = word % 32;
    words_by_bank.at(bank).insert(word);
  }
  int conflict = 1;
  for (const auto& words : words_by_bank) {
    conflict = std::max<int>(conflict, (int)words.size());
  }
  return conflict;
}

class BankConflictInfo : public kir::IrVisitor {
 public:
  static std::unordered_map<const Expr*, std::pair<int, int>> get(
      const kir::Kernel* kernel,
      LaunchParams launch_params,
      const std::unordered_map<Val*, EvaluatorValue>& known_values) {
    if (kernel->topLevelExprs().empty()) {
      return {};
    }
    return BankConflictInfo(kernel, launch_params, known_values)
        .bank_conflict_info_;
  }

 private:
  BankConflictInfo(
      const kir::Kernel* kernel,
      LaunchParams launch_params,
      const std::unordered_map<Val*, EvaluatorValue>& known_values) {
    bindValues(launch_params, known_values);
    inferLaunchParams(kernel);
    handle(kernel->topLevelExprs());
  }

  void bindValues(
      LaunchParams launch_params,
      const std::unordered_map<Val*, EvaluatorValue>& known_values) {
    expr_eval_.bind("blockIdx.x", 0);
    expr_eval_.bind("blockIdx.y", 0);
    expr_eval_.bind("blockIdx.z", 0);
    if (launch_params.bdimx() != LaunchParams::UNINITIALIZED_VAL) {
      expr_eval_.bind(ParallelType::TIDx, launch_params.bdimx());
    }
    if (launch_params.bdimy() != LaunchParams::UNINITIALIZED_VAL) {
      expr_eval_.bind(ParallelType::TIDy, launch_params.bdimy());
    }
    if (launch_params.bdimz() != LaunchParams::UNINITIALIZED_VAL) {
      expr_eval_.bind(ParallelType::TIDz, launch_params.bdimz());
    }
    if (launch_params.gdimx() != LaunchParams::UNINITIALIZED_VAL) {
      expr_eval_.bind(ParallelType::BIDx, launch_params.gdimx());
    }
    if (launch_params.gdimy() != LaunchParams::UNINITIALIZED_VAL) {
      expr_eval_.bind(ParallelType::BIDy, launch_params.gdimy());
    }
    if (launch_params.gdimz() != LaunchParams::UNINITIALIZED_VAL) {
      expr_eval_.bind(ParallelType::BIDz, launch_params.gdimz());
    }
    for (const auto& pair : known_values) {
      expr_eval_.bind(pair.first, pair.second);
    }
  }

  void inferLaunchParams(const kir::Kernel* kernel) {
    const auto& parallel_dimension_map =
        kernel->summary().parallel_dimension_map_.getMap();
    for (const auto& [p, v] : parallel_dimension_map) {
      auto inferred_parallel_dim = expr_eval_.evaluate(v);
      if (inferred_parallel_dim.has_value()) {
        expr_eval_.bind(p, inferred_parallel_dim->as<int64_t>());
      }
    }
  }

  using kir::IrVisitor::handle;

  void handle(Expr* expr) final {
    if (expr->isA<kir::ForLoop>() || expr->isA<kir::IfThenElse>()) {
      kir::IrVisitor::handle(expr);
      return;
    }

    if (expr->isA<LoadStoreOp>()) {
      auto ldst = expr->as<LoadStoreOp>();
      std::pair<int, int> conflict_ways{0, 0};
      if (isSmemTensorIndex(ldst->in())) {
        conflict_ways.first = getConflictWays(
            evaluateAddressesOnFirstPhase(for_loops_, expr_eval_, ldst, true));
      }
      if (isSmemTensorIndex(ldst->out())) {
        conflict_ways.second = getConflictWays(
            evaluateAddressesOnFirstPhase(for_loops_, expr_eval_, ldst, false));
      }
      if (conflict_ways.first > 1 || conflict_ways.second > 1) {
        bank_conflict_info_[expr] = conflict_ways;
      }
    }
  }

  std::unordered_map<const Expr*, std::pair<int, int>> bank_conflict_info_;
  ExpressionEvaluator expr_eval_;
};

} // namespace

std::unordered_map<const Expr*, std::pair<int, int>> getBankConflictInfo(
    const kir::Kernel* kernel,
    LaunchParams launch_params,
    const std::unordered_map<Val*, EvaluatorValue>& known_values) {
  for (const auto& pair : known_values) {
    if (auto ns = dynamic_cast<NamedScalar*>(pair.first)) {
      TORCH_CHECK(
          !ns->isThreadIdx(),
          "threadIdx.{x,y,z} should be computed instead of provided");
      TORCH_CHECK(
          !ns->isBlockIdx(),
          "blockIdx.{x,y,z} should not be provided (they are always zero)");
      TORCH_CHECK(
          !ns->isBlockDim(),
          "blockDim.{x,y,z} should be provided by launch_params");
      TORCH_CHECK(
          !ns->isGridDim(),
          "gridDim.{x,y,z} should be provided by launch_params");
    }
  }
  return BankConflictInfo::get(kernel, launch_params, known_values);
}

} // namespace nvfuser
