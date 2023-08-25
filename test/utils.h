// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#pragma once

#include <codegen.h>
#include <device_lower/lower2device.h>
#include <device_lower/pass/magic_zero.h>
#include <executor.h>
#include <expr_evaluator.h>
#include <ir/all_nodes.h>
#include <kernel_cache.h>
#include <kernel_ir_dispatch.h>
#include <transform_replay.h>

#include <ATen/Context.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDACachingAllocator.h>
#include <torch/torch.h>

#include <gtest/gtest.h>

#include <chrono>
#include <cstddef>
#include <string>
#include <unordered_map>
#include <vector>

namespace nvfuser {

// Make s Stack used for TorchScript execution
inline torch::jit::Stack createStack(std::vector<at::Tensor>&& list) {
  return torch::jit::Stack(
      std::make_move_iterator(list.begin()),
      std::make_move_iterator(list.end()));
}

// Make a tensor that is known to be fully contiguous of dimensionality=ndims,
// but unknown sizes
inline TensorView* makeContigTensor(
    size_t ndims,
    DataType dtype = DataType::Float) {
  return TensorViewBuilder().ndims(ndims).dtype(dtype).contiguity(true).build();
}

// Make a tensor that is known to be non-contiguous of dimensionality=ndims,
// but unknown sizes
inline TensorView* makeSymbolicTensor(
    size_t ndims,
    DataType dtype = DataType::Float) {
  return TensorViewBuilder().ndims(ndims).dtype(dtype).build();
}

// Similar to the other overload but uses shape only to create
// broadcast IterDomains for size-1 axes. The extents of other axes
// remain symbolic.
inline TensorView* makeSymbolicTensor(
    std::vector<int64_t> shape,
    DataType dtype = DataType::Float) {
  for (auto& s : shape) {
    if (s != 1) {
      s = -1;
    }
  }
  return TensorViewBuilder().shape(shape).dtype(dtype).build();
}

// Make a non-contiguous tensor of compile-time known sizes
inline TensorView* makeConcreteTensor(
    std::vector<int64_t> shape,
    DataType dtype = DataType::Float) {
  return TensorViewBuilder().shape(shape).dtype(dtype).build();
}

inline TensorView* makeContigConcreteTensor(
    std::vector<int64_t> shape,
    DataType dtype = DataType::Float) {
  return TensorViewBuilder().shape(shape).dtype(dtype).contiguity(true).build();
}

int64_t prime_number(int64_t i);

inline bool deviceMajorMinorCheck(int major, int minor = 0) {
  auto dev_prop = at::cuda::getCurrentDeviceProperties();
  if (dev_prop->major < major ||
      (dev_prop->major == major && dev_prop->minor < minor)) {
    return false;
  }
  return true;
}

inline int deviceSMCount() {
  int sm_count = at::cuda::getCurrentDeviceProperties()->multiProcessorCount;
  return sm_count;
}

inline void clearL2Cache() {
  torch::NoGradGuard no_grad;
  auto l2_cache_size = at::cuda::getCurrentDeviceProperties()->l2CacheSize;
  auto options =
      torch::TensorOptions().dtype(torch::kFloat32).device(at::kCUDA, 0);

  auto l2_elems = l2_cache_size / 4;
  torch::Tensor t0 = torch::empty(l2_elems, options);
  torch::Tensor t1 = torch::clone(t0);
};

inline TensorView* loweredTv(TensorView* tv, kir::Kernel* kernel) {
  auto used_tvs = ir_utils::allTvs(kernel);
  TensorView* matching_tv = nullptr;
  for (auto lowered_tv : used_tvs) {
    if (lowered_tv->name() == tv->name()) {
      matching_tv = lowered_tv;
    }
  }
  TORCH_INTERNAL_ASSERT(matching_tv != nullptr);
  return matching_tv;
}

inline TensorView* loweredTv(TensorView* tv, GpuLower& gpulw) {
  return loweredTv(tv, gpulw.kernel());
}

class PredicatedChecker : public kir::IrVisitor {
 public:
  // Checks if the provided tv is written to within a non-trivial conditional
  static bool isPredicated(StmtNameType tv_name, GpuLower& gpulw) {
    PredicatedChecker checker(tv_name, gpulw.kernel()->topLevelExprs());
    return checker.is_predicated_;
  }

  static bool isPredicated(StmtNameType tv_name, kir::Kernel* kernel) {
    PredicatedChecker checker(tv_name, kernel->topLevelExprs());
    return checker.is_predicated_;
  }

  static bool isPredicated(TensorView* tv, GpuLower& gpulw) {
    return isPredicated(tv->name(), gpulw);
  }

  static bool isPredicated(TensorView* tv, kir::Kernel* kernel) {
    return isPredicated(tv->name(), kernel);
  }

 private:
  PredicatedChecker() = delete;

  PredicatedChecker(StmtNameType tv_name, std::vector<Expr*> exprs)
      : tv_name_(tv_name) {
    kir::IrVisitor::handle(exprs);
  }

  using kir::IrVisitor::handle;
  bool is_predicated_ = false;
  bool predicated_ite_ = false;
  StmtNameType tv_name_ = 0;

  void handle(kir::IfThenElse* ite) final {
    auto prev_ite = predicated_ite_;
    predicated_ite_ = !ite->predicate()->value()->isConstScalar();
    kir::IrVisitor::handle(ite);
    predicated_ite_ = prev_ite;
  }

  void dispatch(Expr* expr) final {
    if (expr->outputs().size() && expr->outputs()[0]->isA<kir::TensorIndex>()) {
      auto ti = expr->outputs()[0]->as<kir::TensorIndex>();
      if (ti->view()->name() == tv_name_) {
        is_predicated_ = is_predicated_ | predicated_ite_;
        if (expr->predicate() != nullptr &&
            !expr->predicate()->value()->isConst()) {
          is_predicated_ = true;
        }
      }
    }
    kir::IrVisitor::dispatch(expr);
  }
};

class UnswitchInElseChecker : public kir::IrVisitor {
 public:
  // Checks if there are any unswitched for loops within an else clause
  static bool check(GpuLower& gpulw) {
    UnswitchInElseChecker checker(gpulw.kernel()->topLevelExprs());
    return checker.found_in_else_;
  }

 private:
  UnswitchInElseChecker() = delete;
  UnswitchInElseChecker(std::vector<Expr*> exprs) {
    kir::IrVisitor::handle(exprs);
  }

  using kir::IrVisitor::handle;
  bool within_else_ = false;
  bool found_in_else_ = false;

  void handle(kir::IfThenElse* ite) final {
    auto prev_within_else = within_else_;
    within_else_ = true;
    kir::IrVisitor::handle(ite->elseBody().exprs());
    within_else_ = prev_within_else;
  }

  void handle(kir::ForLoop* for_loop) final {
    if (for_loop->iter_domain()->getParallelType() == ParallelType::Unswitch) {
      found_in_else_ = found_in_else_ || within_else_;
    }
    kir::IrVisitor::handle(for_loop);
  }
};

class PredicateMagicZeroChecker : public kir::IrVisitor {
 public:
  // Checks if all predicated domains of the provided tv are protected with
  // magic zero
  static bool isProtected(TensorView* tv, GpuLower& gpulw) {
    PredicateMagicZeroChecker checker(
        loweredTv(tv, gpulw), gpulw.kernel()->topLevelExprs());
    return checker.is_protected_;
  }

 private:
  using kir::IrVisitor::dispatch;
  using kir::IrVisitor::handle;

  PredicateMagicZeroChecker(TensorView* tv, std::vector<Expr*> exprs)
      : tv_(tv) {
    handle(exprs);
  }

  void handle(kir::IfThenElse* ite) final {
    auto prev_predicate = predicate_;
    predicate_ = ite->predicate()->value();
    kir::IrVisitor::handle(ite);
    predicate_ = prev_predicate;
  }

  void dispatch(Expr* expr) final {
    if (expr->outputs().size() && expr->outputs()[0]->isA<kir::TensorIndex>()) {
      auto ti = expr->outputs()[0]->as<kir::TensorIndex>();
      if (ti->view() == tv_) {
        is_protected_ = checkPredicateOfTensor(predicate_);
        return;
      }
    }

    if (expr->isA<kir::ForLoop>()) {
      handle(expr->as<kir::ForLoop>());
    } else if (expr->isA<kir::IfThenElse>()) {
      handle(expr->as<kir::IfThenElse>());
    } else {
      for (auto input : expr->inputs()) {
        dispatch(input);
      }
    }
  }

  // Return true If all predicated domains are protected
  bool checkPredicateOfTensor(Val* predicate) {
    auto id_predicates = decomposeCompoundPredicate(predicate);
    for (auto id_predicate : id_predicates) {
      // Just check if nvfuser_zero is used. Not perfect but probably
      // good enough.
      is_magic_zero_found_ = false;
      dispatch(id_predicate);
      if (!is_magic_zero_found_) {
        return false;
      }
    }
    return true;
  }

  // Decompose "X && Y" to a vector of {X, Y}.
  std::vector<Val*> decomposeCompoundPredicate(Val* predicate) {
    if (auto binary_op = dynamic_cast<BinaryOp*>(predicate->definition())) {
      if (binary_op->getBinaryOpType() == BinaryOpType::LogicalAnd) {
        auto pred = decomposeCompoundPredicate(binary_op->lhs());
        auto rhs_pred = decomposeCompoundPredicate(binary_op->rhs());
        pred.insert(pred.end(), rhs_pred.begin(), rhs_pred.end());
        return pred;
      }
    }

    return {predicate};
  }

  void dispatch(Val* val) final {
    if (isMagicZero(val)) {
      is_magic_zero_found_ = true;
      return;
    }

    auto def = val->definition();
    if (def != nullptr) {
      dispatch(def);
    }
  }

 private:
  bool is_protected_ = false;
  Val* predicate_ = nullptr;
  TensorView* tv_ = nullptr;
  bool is_magic_zero_found_ = false;
};

// Basically just TransformPropagator, except that it checks the consistency
// replayPasC with getMatchedLeafPosWithoutReplayPasC, replayCasP with
// getMatchedLeafPosWithoutReplayCasP, and fullSelfReplay with fullSelfMatching:
// - After replayPasC, getMatchedLeafPosWithoutReplayPasC should return the same
//   replayed position
// - After replayCasP, getMatchedLeafPosWithoutReplayCasP should return the same
//   replayed position
// - After fullSelfReplay, fullSelfMatching should return true
struct TransformPropagatorWithCheck : public TransformPropagator {
 public:
  virtual void propagateC2P(TensorView* from, TensorView* to) override {
    TransformPropagator::propagateC2P(from, to);
    auto from_pos = replayed_pos_.at(from);
    auto to_pos = replayed_pos_.at(to);
    TORCH_CHECK(
        TransformReplay::getMatchedLeafPosWithoutReplayPasC(
            to, from, from_pos) == (int)to_pos);
  }
  virtual void propagateP2C(TensorView* from, TensorView* to) override {
    TransformPropagator::propagateP2C(from, to);
    auto from_pos = replayed_pos_.at(from);
    auto to_pos = replayed_pos_.at(to);
    TORCH_CHECK(
        TransformReplay::getMatchedLeafPosWithoutReplayCasP(
            to, from, from_pos) == (int)to_pos);
  }
  virtual void propagateSibling(TensorView* from, TensorView* to) override {
    TransformPropagator::propagateSibling(from, to);
    auto from_pos = replayed_pos_.at(from);
    auto to_pos = replayed_pos_.at(to);
    TORCH_CHECK(from_pos == to_pos);
    TORCH_CHECK(TransformReplay::fullSelfMatching(from, to));
  }
  using TransformPropagator::TransformPropagator;
};

class KernelExprVisitor : private kir::IrVisitor {
 public:
  static std::vector<Expr*> getAllExprs(const kir::Kernel* kernel) {
    KernelExprVisitor visitor(kernel);
    return visitor.all_exprs_;
  }

 private:
  KernelExprVisitor(const kir::Kernel* kernel) {
    handle(kernel->topLevelExprs());
  }

  using kir::IrVisitor::handle;

  void dispatch(Expr* expr) final {
    all_exprs_.push_back(expr);
    kir::IrVisitor::dispatch(expr);
  }

 private:
  std::vector<Expr*> all_exprs_;
};

class ContextCudnnTF32Disabled {
 public:
  ContextCudnnTF32Disabled() {
    flag_ = at::globalContext().allowTF32CuDNN();
    at::globalContext().setAllowTF32CuDNN(false);
  }

  ~ContextCudnnTF32Disabled() {
    at::globalContext().setAllowTF32CuDNN(flag_);
  }

 private:
  bool flag_;
};

inline bool maybeClearAllocator(int64_t max_bytes = ((int64_t)1 << 32)) {
  // check used memory and empty allocator cache if above a set threshold
  auto allocator = c10::cuda::CUDACachingAllocator::get();
  if (allocator->initialized()) {
    int device = 0;
#define TORCH_VERSION_GREATER(major, minor, patch)                    \
  TORCH_VERSION_MAJOR > major ||                                      \
      (TORCH_VERSION_MAJOR == major && TORCH_VERSION_MINOR > minor || \
       (TORCH_VERSION_MINOR == minor && TORCH_VERSION_PATCH > patch))
#if TORCH_VERSION_GREATER(2, 0, 1)
    // GetDevice was introduced in https://github.com/pytorch/pytorch/pull/94864
    // in order to properly handle new CUDA 112 behavior
    c10::cuda::GetDevice(&device);
#else
    cudaGetDevice(&device);
#endif

    auto device_stats = allocator->getDeviceStats(0);
    // allocated_bytes[] holds multiple statistics but the first is sum across
    // both small and large blocks
    if (device_stats.reserved_bytes[0].current > max_bytes) {
      allocator->emptyCache();
      return true;
    }
  }
  return false;
}

//! Get the random seed to use for each test. If aten=true, returns the ATen
//! seed. Otherwise returns the seed for std::rand()
size_t getRandomSeed(bool aten = false);

// Fixture class must be uniquely identified, i.e., can't be in an
// anonymous namespace
class NVFuserTest : public ::testing::Test {
 protected:
  void SetUp() override {
    // requires PASCAL or newer
    if (!deviceMajorMinorCheck(6)) {
      GTEST_SKIP() << "skipping tests on pre-PASCAL GPUs";
    }
    setFillAllocationWithNan(true);

    maybeClearAllocator();

    // If NVFUSER_TEST_RANDOM_SEED is provided, use that for the C random seed.
    // Otherwise, use system time. If a test fails, this seed will be printed.
    at::manual_seed(getRandomSeed(false));

    // If NVFUSER_TEST_ATEN_RANDOM_SEED is provided, use that for the ATen
    // random seed. Otherwise, use zero. If a test fails, this seed will be
    // printed.
    std::srand(getRandomSeed(true));
  }

  void TearDown() override {
    if (::testing::Test::HasFailure()) {
      std::cerr
          << "To reproduce, use environment variables NVFUSER_TEST_RANDOM_SEED="
          << getRandomSeed(false)
          << " NVFUSER_TEST_ATEN_RANDOM_SEED=" << getRandomSeed(true)
          << std::endl;
    }
  }
};

// assert that the given fusion lowers to the given CUDA kernel
void assertCUDAKernel(Fusion* fusion, const std::string& expected_kernel);

namespace sass {

// For SASS instruction definitions, see:
// https://docs.nvidia.com/cuda/cuda-binary-utilities/index.html#instruction-set-reference

struct Instruction {
  std::string str;
  size_t address;

  std::string predicate();
  std::string action(); // The part of the string that is not predicate
  std::string op(); // Some thing like: LDGSTS.E.128
  std::string opCode(); // Something like LDGSTS
  std::vector<std::string> modifiers(); // Something like {E, 128}
  std::vector<std::string> args(); // Something like {[R217+0x1800], [R202.64]}
};

struct Label {
  std::string name;
};

struct Container {
  std::unordered_map<std::string, std::string> attributes;
  std::vector<std::variant<Instruction, Label>> code;

  std::string toString();
};

Container parse(const std::string& nvdisasm_output);

} // namespace sass

inline bool cudaArchGuardShouldSkip(int required_major, int required_minor) {
  int capability_major = at::cuda::getCurrentDeviceProperties()->major;
  int capability_minor = at::cuda::getCurrentDeviceProperties()->minor;

  if (capability_major < required_major ||
      (capability_major == required_major &&
       capability_minor < required_minor)) {
    return true;
  }
  return false;
}

inline bool cudaArchGuardShouldSkip(
    int lower_major, // inclusive
    int lower_minor, // inclusive
    int upper_major, // exclusive
    int upper_minor // exclusive
) {
  int capability_major = at::cuda::getCurrentDeviceProperties()->major;
  int capability_minor = at::cuda::getCurrentDeviceProperties()->minor;

  if (capability_major < lower_major ||
      (capability_major == lower_major && capability_minor < lower_minor)) {
    return true;
  }
  if (capability_major > upper_major ||
      (capability_major == upper_major && capability_minor >= upper_minor)) {
    return true;
  }
  return false;
}

#define NVFUSER_TEST_CUDA_ARCH_GUARD(REQUIRED_MAJOR, REQUIRED_MINOR)          \
  if (cudaArchGuardShouldSkip(REQUIRED_MAJOR, REQUIRED_MINOR)) {              \
    GTEST_SKIP() << "Requires GPU capability above " << REQUIRED_MAJOR << "." \
                 << REQUIRED_MINOR << " to run.\n";                           \
  }

#define NVFUSER_TEST_CUDA_ARCH_RANGE_GUARD(                             \
    LOWER_MAJOR, LOWER_MINOR, UPPER_MAJOR, UPPER_MINOR)                 \
  if (cudaArchGuardShouldSkip(                                          \
          LOWER_MAJOR, LOWER_MINOR, UPPER_MAJOR, UPPER_MINOR)) {        \
    GTEST_SKIP() << "Requires GPU capability >= " << LOWER_MAJOR << "." \
                 << LOWER_MINOR << " and < " << UPPER_MAJOR << "."      \
                 << UPPER_MINOR << " to run.\n";                        \
  }

#define NVFUSER_TEST_CUDA_ARCH_COMPILE_CHECK(                                \
    REQUIRED_MAJOR, REQUIRED_MINOR, COMPILE_FUSION)                          \
  if (cudaArchGuardShouldSkip(REQUIRED_MAJOR, REQUIRED_MINOR)) {             \
    ASSERT_ANY_THROW(COMPILE_FUSION);                                        \
    GTEST_SKIP() << "(Lowered Only) Requires GPU capability above "          \
                 << REQUIRED_MAJOR << "." << REQUIRED_MINOR << " to run.\n"; \
  } else {                                                                   \
    COMPILE_FUSION;                                                          \
  }

// util to track support matmul operand layout.
using MatmulLayout = MmaOptions::MmaLayout;

static constexpr std::array<MatmulLayout, 4> kAllSupportedMatmulLayout = {
    MatmulLayout::TT,
    MatmulLayout::NT,
    MatmulLayout::TN,
    MatmulLayout::NN};

// Generic interface to get matmul op with the given layout.
TensorView* matmul(
    TensorView* a,
    TensorView* b,
    MatmulLayout layout,
    bool turing_or_later // TODO: This is a temporary solution. Remove this!
);

// Generic interface to get splitK-like batched matmul op with the given layout.
// For splitK like batched matmul, there is only one batch dimension, and that
// dimension should be right before the K dimension. This function currently
// assume Ampere or Turing.
TensorView* splitkLikeBatchedMatmul(
    TensorView* a,
    TensorView* b,
    MatmulLayout layout);

// Utility to generate matmul input tensors based on given layout
at::Tensor atMatmul(at::Tensor a, at::Tensor b, MatmulLayout layout);

// Utility to generate matmul input tensors based on given layout
at::Tensor splitkLikeAtMatmul(at::Tensor a, at::Tensor b, MatmulLayout layout);

// Utility to generate inputs based on given layout
std::pair<at::Tensor, at::Tensor> matmulAtInput(
    int M,
    int N,
    int K,
    MatmulLayout layout,
    c10::ScalarType dtype = at::kHalf);

// Labels to describe tensor position in matmul:
// A, B - input
// C - input if beta is provided, shape must be the same as output (D)
// Bias - input vector, shape is equal to D rows
// D - output
enum class TensorMatmulPos { A, B, C, D, Bias };

// Utility to generate buffers based on given problem, layout and tensor
//  position in matmul with support for matmul and strided batch matmul
at::Tensor matmulAtInput(
    const MatmulLayout layout,
    const TensorMatmulPos tensor,
    const c10::ScalarType dtype,
    const int M,
    const int N,
    const int K,
    const int B = 0,
    const int device = 0);

#define REQUIRE_DEVICE_SMEM_SIZE(required_size, device_idx)                 \
  if (at::cuda::getDeviceProperties(device_idx)->sharedMemPerBlockOptin <   \
      required_size) {                                                      \
    GTEST_SKIP() << "not enough shared memory space on device to run test"; \
  }

// Utility to check if for given kernel the expected scheduler has
// been used
bool isSchedulerInUse(
    nvfuser::FusionKernelRuntime* kernel_rt,
    const ScheduleHeuristic& scheduler);

// Disable magic zero
constexpr CompileParams matmul_cparams{DataType::Int32, 255, false};

// Validate that the fusion is segmented with desired scheduler, currently only
// supporting two segments
void validateSegmentation(
    FusionKernelRuntime* runtime,
    const std::vector<ScheduleHeuristic>& expected_heuristics);

// Utility to generate tensor with bias applied on the input tensor
TensorView* biasEpilogue(TensorView* tensor, TensorView* bias);

// Utility to generate tensor with bias applied on the input tensor,
// to be used to caldulate reference data
at::Tensor atBiasEpilogue(const at::Tensor& tensor, const at::Tensor& bias);
} // namespace nvfuser
