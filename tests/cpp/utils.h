// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#pragma once

#include <codegen.h>
#include <csrc/exceptions.h>
#include <device_lower/lower2device.h>
#include <device_lower/pass/magic_zero.h>
#include <expr_evaluator.h>
#include <id_model/id_model.h>
#include <ir/all_nodes.h>
#include <kernel_ir_dispatch.h>
#include <runtime/allocations.h>
#include <runtime/executor.h>
#include <runtime/fusion_executor_cache.h>
#include <runtime/fusion_kernel_runtime.h>
#include <scheduler/registry.h>
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

struct CGResultsPackage {
  KernelArgumentHolder outputs;
  std::unique_ptr<HeuristicParams> heuristic_params;
  std::unique_ptr<KernelExecutor> kernel_executor;
};

// Returns the only executor in the most recent runtime.
const KernelExecutor* onlyKernelExecutorInMostRecentRuntime(
    const FusionExecutorCache& executor_cache);

// Grabs heuristics and schedules with the provided scheduler type, compiles and
// runs with Fuion executor, returns a struct containing the outputs,
// heuristic_params, and KernelExecutor. These structures are for convenience in
// testing. If validate_scheduler is set to false the scheduler check will still
// be run but it will be ignored. Otherwise canScheduler returning false will
// throw.
CGResultsPackage scheduleAndRun(
    Fusion* fusion,
    SchedulerType scheduler_type,
    const KernelArgumentHolder& runtime_inputs,
    bool validate_scheduler = true);

// Make s Stack used for TorchScript execution
inline torch::jit::Stack createStack(std::vector<at::Tensor>&& list) {
  return torch::jit::Stack(
      std::make_move_iterator(list.begin()),
      std::make_move_iterator(list.end()));
}

void runAndValidate(
    Fusion* fusion,
    SchedulerType scheduler_type,
    const KernelArgumentHolder& runtime_inputs,
    int line_number,
    const char* file_name);

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
  auto used_tvs = kernel->allTvs();
  TensorView* matching_tv = nullptr;
  for (auto lowered_tv : used_tvs) {
    if (lowered_tv->name() == tv->name()) {
      matching_tv = lowered_tv;
    }
  }
  NVF_ERROR(matching_tv != nullptr);
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

  static bool isPredicatedByIfThenElse(
      StmtNameType tv_name,
      kir::Kernel* kernel) {
    PredicatedChecker checker(tv_name, kernel->topLevelExprs());
    return checker.predicated_ite_;
  }

  // If CpAsync from gmem to smem, then loaded from smem to registers using
  // ldmatrix, then it is used in mma and should not use if-then-else predicate.
  // If just CpAsync from gmem to smem, without further copy to register, then
  // it is not used in mma and can use if-then-else predicate.
  static bool isCpAsyncMmaPredicatedByIfThenElse(kir::Kernel* kernel) {
    for (auto tv : kernel->allTvs()) {
      if (tv->definition() != nullptr &&
          ir_utils::isCpAsyncOp(tv->definition())) {
        const auto& consumers = ir_utils::consumerTvsOf(tv);
        if (std::any_of(
                consumers.begin(), consumers.end(), [&](TensorView* tv) {
                  return ir_utils::isLdMatrixOp(tv->definition());
                })) {
          return isPredicatedByIfThenElse(tv->name(), kernel);
        }
      }
    }
    return false;
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

  void handle(ForLoop* for_loop) final {
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

    if (expr->isA<ForLoop>()) {
      handle(expr->as<ForLoop>());
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
    NVF_CHECK(
        TransformReplay::getMatchedLeafPosWithoutReplayPasC(
            to, from, from_pos) == to_pos);
  }
  virtual void propagateP2C(TensorView* from, TensorView* to) override {
    TransformPropagator::propagateP2C(from, to);
    auto from_pos = replayed_pos_.at(from);
    auto to_pos = replayed_pos_.at(to);
    NVF_CHECK(
        TransformReplay::getMatchedLeafPosWithoutReplayCasP(
            to, from, from_pos) == to_pos);
  }
  virtual void propagateSibling(TensorView* from, TensorView* to) override {
    TransformPropagator::propagateSibling(from, to);
    auto from_pos = replayed_pos_.at(from);
    auto to_pos = replayed_pos_.at(to);
    NVF_CHECK(from_pos == to_pos);
    NVF_CHECK(TransformReplay::fullSelfMatching(from, to));
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
#if NVF_TORCH_VERSION_NO_LESS(2, 3, 0)
    // c10::cuda uses DeviceIndex instead of int
    // https://github.com/pytorch/pytorch/pull/119142
    c10::DeviceIndex device_index;
    c10::cuda::GetDevice(&device_index);
    device = static_cast<int>(device_index);
#elif NVF_TORCH_VERSION_GREATER(2, 0, 1)
    // GetDevice was introduced in https://github.com/pytorch/pytorch/pull/94864
    // in order to properly handle new CUDA 112 behavior
    c10::cuda::GetDevice(&device);
#else
    cudaGetDevice(&device);
#endif

    auto device_stats = allocator->getDeviceStats(device);
    // allocated_bytes[] holds multiple statistics but the first is sum across
    // both small and large blocks
    if (uint64_t(device_stats.reserved_bytes[0].current) >
        uint64_t(max_bytes)) {
      allocator->emptyCache();
      return true;
    }
  }
  return false;
}

//! Returns the seed for std::rand() used for every test.
size_t getCRandomSeed();

//! Returns the seed for ATen functions like at::randn() used for every test.
size_t getATenRandomSeed();

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

// Fixture class must be uniquely identified, i.e., can't be in an
// anonymous namespace
class NVFuserTest : public ::testing::Test {
 protected:
  NVFuserTest();
  ~NVFuserTest() override;
  void SetUp() override;

  // Start capturing of stdout if not already started
  void captureStdout();
  // Stop capturing of stdout if being captured
  void ensureStopCaptureStdout();
  // Get capturing stdout
  std::string getCapturedStdout();

 protected:
  EnableOptionsGuard enable_options_guard_;
  DisableOptionsGuard disable_options_guard_;

 private:
  bool capturing_ = false;
};

class HopperBase : public NVFuserTest {
 protected:
  void SetUp() override {
    if (cudaArchGuardShouldSkip(9, 0, 10, 0)) {
      GTEST_SKIP() << "skipping tests on non-Hopper GPUs";
    }
    NVFuserTest::SetUp();
  }
};

class BlackwellBase : public NVFuserTest {
 protected:
  void SetUp() override {
    if (cudaArchGuardShouldSkip(10, 0)) {
      GTEST_SKIP() << "skipping tests on non-Blackwell GPUs";
    }
    NVFuserTest::SetUp();
    EnableOptionsGuard::getCurOptions().set(EnableOption::IdModel, {"all"});
  }
};

// TMA is supported on Hopper and newer GPUs
class TmaBase : public NVFuserTest {
 protected:
  void SetUp() override {
    if (cudaArchGuardShouldSkip(9, 0)) {
      GTEST_SKIP() << "skipping tests on pre-Hopper GPUs";
    }
    NVFuserTest::SetUp();
    EnableOptionsGuard::getCurOptions().set(EnableOption::IdModel, {"all"});
  }
};

// Fixture with param class must be uniquely identified, i.e., can't be in an
// anonymous namespace
template <typename tParam>
class NVFuserFixtureParamTest : public NVFuserTest,
                                public ::testing::WithParamInterface<tParam> {};

// assert that the given fusion lowers to the given CUDA kernel
void assertCUDAKernel(Fusion* fusion, const std::string& expected_kernel);

namespace sass {

// For SASS instruction definitions, see:
// https://docs.nvidia.com/cuda/cuda-binary-utilities/index.html#instruction-set-reference

struct Instruction {
  std::string str;
  size_t address;

  std::string predicate() const;
  std::string action() const; // The part of the string that is not predicate
  std::string op() const; // Some thing like: LDGSTS.E.128
  std::string opCode() const; // Something like LDGSTS
  std::vector<std::string> modifiers() const; // Something like {E, 128}
  std::vector<std::string> args()
      const; // Something like {[R217+0x1800], [R202.64]}
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

static auto kAllSupportedMmaLayout = std::vector<MmaLayout>{
    MmaLayout::TT,
    MmaLayout::TN,
    MmaLayout::NT,
    MmaLayout::NN};

inline std::string mmaLayoutName(
    const testing::TestParamInfo<MmaLayout>& info) {
  return toString(info.param);
}

static auto kAllSmemSwizzleModes = std::vector<MmaInputSmemSwizzle>{
    MmaInputSmemSwizzle::None,
    MmaInputSmemSwizzle::B128,
    MmaInputSmemSwizzle::B64,
    MmaInputSmemSwizzle::B32};

static auto kAllHopperMacros = std::vector<MmaMacro>{
    MmaMacro::Hopper_64_8_16,   MmaMacro::Hopper_64_16_16,
    MmaMacro::Hopper_64_24_16,  MmaMacro::Hopper_64_32_16,
    MmaMacro::Hopper_64_40_16,  MmaMacro::Hopper_64_48_16,
    MmaMacro::Hopper_64_56_16,  MmaMacro::Hopper_64_64_16,
    MmaMacro::Hopper_64_72_16,  MmaMacro::Hopper_64_80_16,
    MmaMacro::Hopper_64_88_16,  MmaMacro::Hopper_64_96_16,
    MmaMacro::Hopper_64_104_16, MmaMacro::Hopper_64_112_16,
    MmaMacro::Hopper_64_120_16, MmaMacro::Hopper_64_128_16,
    MmaMacro::Hopper_64_136_16, MmaMacro::Hopper_64_144_16,
    MmaMacro::Hopper_64_152_16, MmaMacro::Hopper_64_160_16,
    MmaMacro::Hopper_64_168_16, MmaMacro::Hopper_64_176_16,
    MmaMacro::Hopper_64_184_16, MmaMacro::Hopper_64_192_16,
    MmaMacro::Hopper_64_200_16, MmaMacro::Hopper_64_208_16,
    MmaMacro::Hopper_64_216_16, MmaMacro::Hopper_64_224_16,
    MmaMacro::Hopper_64_232_16, MmaMacro::Hopper_64_240_16,
    MmaMacro::Hopper_64_248_16, MmaMacro::Hopper_64_256_16};

static auto kAllBlackwell1CTAM64Macros = std::vector<MmaMacro>{
    MmaMacro::Blackwell1CTA_64_8_16,   MmaMacro::Blackwell1CTA_64_16_16,
    MmaMacro::Blackwell1CTA_64_24_16,  MmaMacro::Blackwell1CTA_64_32_16,
    MmaMacro::Blackwell1CTA_64_40_16,  MmaMacro::Blackwell1CTA_64_48_16,
    MmaMacro::Blackwell1CTA_64_56_16,  MmaMacro::Blackwell1CTA_64_64_16,
    MmaMacro::Blackwell1CTA_64_72_16,  MmaMacro::Blackwell1CTA_64_80_16,
    MmaMacro::Blackwell1CTA_64_88_16,  MmaMacro::Blackwell1CTA_64_96_16,
    MmaMacro::Blackwell1CTA_64_104_16, MmaMacro::Blackwell1CTA_64_112_16,
    MmaMacro::Blackwell1CTA_64_120_16, MmaMacro::Blackwell1CTA_64_128_16,
    MmaMacro::Blackwell1CTA_64_136_16, MmaMacro::Blackwell1CTA_64_144_16,
    MmaMacro::Blackwell1CTA_64_152_16, MmaMacro::Blackwell1CTA_64_160_16,
    MmaMacro::Blackwell1CTA_64_168_16, MmaMacro::Blackwell1CTA_64_176_16,
    MmaMacro::Blackwell1CTA_64_184_16, MmaMacro::Blackwell1CTA_64_192_16,
    MmaMacro::Blackwell1CTA_64_200_16, MmaMacro::Blackwell1CTA_64_208_16,
    MmaMacro::Blackwell1CTA_64_216_16, MmaMacro::Blackwell1CTA_64_224_16,
    MmaMacro::Blackwell1CTA_64_232_16, MmaMacro::Blackwell1CTA_64_240_16,
    MmaMacro::Blackwell1CTA_64_248_16, MmaMacro::Blackwell1CTA_64_256_16};

static auto kAllBlackwell1CTAM128Macros = std::vector<MmaMacro>{
    MmaMacro::Blackwell1CTA_128_16_16,
    MmaMacro::Blackwell1CTA_128_32_16,
    MmaMacro::Blackwell1CTA_128_48_16,
    MmaMacro::Blackwell1CTA_128_64_16,
    MmaMacro::Blackwell1CTA_128_80_16,
    MmaMacro::Blackwell1CTA_128_96_16,
    MmaMacro::Blackwell1CTA_128_112_16,
    MmaMacro::Blackwell1CTA_128_128_16,
    MmaMacro::Blackwell1CTA_128_144_16,
    MmaMacro::Blackwell1CTA_128_160_16,
    MmaMacro::Blackwell1CTA_128_176_16,
    MmaMacro::Blackwell1CTA_128_192_16,
    MmaMacro::Blackwell1CTA_128_208_16,
    MmaMacro::Blackwell1CTA_128_224_16,
    MmaMacro::Blackwell1CTA_128_240_16,
    MmaMacro::Blackwell1CTA_128_256_16};

static auto kAllBlackwell2CTAM128Macros = std::vector<MmaMacro>{
    MmaMacro::Blackwell2CTA_128_32_16,
    MmaMacro::Blackwell2CTA_128_64_16,
    MmaMacro::Blackwell2CTA_128_96_16,
    MmaMacro::Blackwell2CTA_128_128_16,
    MmaMacro::Blackwell2CTA_128_160_16,
    MmaMacro::Blackwell2CTA_128_192_16,
    MmaMacro::Blackwell2CTA_128_224_16,
    MmaMacro::Blackwell2CTA_128_256_16};

static auto kAllBlackwell2CTAM256Macros = std::vector<MmaMacro>{
    MmaMacro::Blackwell2CTA_256_32_16,
    MmaMacro::Blackwell2CTA_256_64_16,
    MmaMacro::Blackwell2CTA_256_96_16,
    MmaMacro::Blackwell2CTA_256_128_16,
    MmaMacro::Blackwell2CTA_256_160_16,
    MmaMacro::Blackwell2CTA_256_192_16,
    MmaMacro::Blackwell2CTA_256_224_16,
    MmaMacro::Blackwell2CTA_256_256_16};

std::string macroToString(const MmaMacro macro);

// Utility to generate matmul input tensors based on given layout
at::Tensor atMatmul(at::Tensor a, at::Tensor b, MmaLayout layout);

// Utility to generate matmul input tensors based on given layout
at::Tensor splitkLikeAtMatmul(at::Tensor a, at::Tensor b, MmaLayout layout);

// Utility to generate inputs based on given layout
std::pair<at::Tensor, at::Tensor> matmulAtInput2D(
    int64_t M,
    int64_t N,
    int64_t K,
    MmaLayout layout,
    c10::ScalarType dtype = at::kHalf);

// Utility to generate input shapes based on given layout
std::pair<std::vector<int64_t>, std::vector<int64_t>> matmulAtInputShape3DTuring(
    int64_t M,
    int64_t N,
    int64_t K,
    MmaLayout layout);

// Utility to generate inputs based on given layout
std::pair<at::Tensor, at::Tensor> matmulAtInput3DTuring(
    int64_t M,
    int64_t N,
    int64_t K,
    MmaLayout layout,
    c10::ScalarType dtype = at::kHalf);

// Labels to describe tensor position in matmul:
// A, B - input
// C - input if beta is provided, shape must be the same as output (D)
// Bias - input vector, shape is equal to D rows
// D - output
enum class TensorMatmulPos { A, B, C, D, Bias };

// Utility to generate buffers based on given problem, layout and tensor
//  position in matmul with support for matmul and strided batch matmul
at::Tensor matmulAtInput2D(
    const MmaLayout layout,
    const TensorMatmulPos tensor,
    const c10::ScalarType dtype,
    const int64_t M,
    const int64_t N,
    const int64_t K,
    const int64_t B = 0,
    const int64_t device = 0);

inline std::pair<std::vector<int64_t>, std::vector<int64_t>>
matmulAtInputShape3DHopperRS(int M, int N, int K, MmaLayout layout) {
  switch (layout) {
    case MmaLayout::TT:
      return {{M, K, 1}, {1, K, N}};
    case MmaLayout::TN:
      return {{M, 1, K}, {1, N, K}};
    default:
      NVF_CHECK(false, "unsupported data layout.");
  }
}

inline std::pair<at::Tensor, at::Tensor> matmulAtInput3DHopperRS(
    int M,
    int N,
    int K,
    MmaLayout layout,
    c10::ScalarType dtype) {
  auto options = at::TensorOptions().dtype(dtype).device(at::kCUDA, 0);
  auto shapes = matmulAtInputShape3DHopperRS(M, N, K, layout);
  return std::make_pair(
      at::randn(shapes.first, options), at::randn(shapes.second, options));
}

inline std::pair<std::vector<int64_t>, std::vector<int64_t>>
matmulAtInputShape3DSS(int M, int N, int K, MmaLayout layout) {
  switch (layout) {
    case MmaLayout::TT:
      return {{M, K, 1}, {1, K, N}};
    case MmaLayout::TN:
      return {{M, 1, K}, {1, N, K}};
    case MmaLayout::NT:
      return {{K, M, 1}, {K, 1, N}};
    case MmaLayout::NN:
      return {{1, K, M}, {N, K, 1}};
    default:
      NVF_CHECK(false, "unsupported data layout.");
  }
}

inline std::pair<at::Tensor, at::Tensor> matmulAtInput3DSS(
    int M,
    int N,
    int K,
    MmaLayout layout,
    c10::ScalarType dtype) {
  auto options = at::TensorOptions().dtype(dtype).device(at::kCUDA, 0);
  auto shapes = matmulAtInputShape3DSS(M, N, K, layout);
  return std::make_pair(
      at::randn(shapes.first, options), at::randn(shapes.second, options));
}

// Given a tensor view created by matmulAtInput2D or matmulAtInput3DTuring,
// insert permute/BroadcastOp as needed to make it [B, M, N, K]. The returned
// tensor view can be directly used as input to fusedMultiplySum.
TensorView* canonicalizeInputToBMNK(
    TensorView* tv,
    MmaLayout layout,
    MmaOperand operand);

#define REQUIRE_DEVICE_SMEM_SIZE(required_size, device_idx)                 \
  if (at::cuda::getDeviceProperties(device_idx)->sharedMemPerBlockOptin <   \
      required_size) {                                                      \
    GTEST_SKIP() << "not enough shared memory space on device to run test"; \
  }

// Utility to check if for given kernel the expected scheduler has
// been used
bool isSchedulerInUse(
    const nvfuser::FusionKernelRuntime* kernel_rt,
    const SchedulerType& scheduler_type);

// Disable magic zero
const CompileParams matmul_cparams{DataType::Int32, 255, false};

// Utility to generate tensor with bias applied on the input tensor
TensorView* biasEpilogue(TensorView* tensor, TensorView* bias);

// Utility to generate tensor with bias applied on the input tensor,
// to be used to caldulate reference data
at::Tensor atBiasEpilogue(const at::Tensor& tensor, const at::Tensor& bias);

// Get the number of SMs on the current device
int64_t getNumSMs();

bool checkMapped(const ValGraph& vg, IterDomain* x, IterDomain* y);

// This uses mma_utils::getOperandInnerDims(fusion) to get the inner allocation
// dimensions of fusion operands and translate that into one of the MmaOp
// layouts TT, TN, NT, or NN.
MmaLayout getMatmulProblemLayout(Fusion* fusion);

// Get floating data types including half, float, double, complex_float,
// complex_double, and bfloat16 if supported by the device.
std::vector<DataType> getFloatingDataTypes(bool include_complex = true);

// gtest requires test name contains only alphanumeric characters and
// underscores. Sanitize name e.g. std::complex<float> -> std_complex_float
std::string sanitizeTestName(const std::string& name);

// values frequently used in tests
constexpr std::array<int64_t, 21> Pow2Vals1to1Million = {
    1,    2,    4,    8,     16,    32,    64,     128,    256,    512,    1024,
    2048, 4096, 8192, 16384, 32768, 65536, 131072, 262144, 524288, 1048576};

bool isVectorized(TensorView* tv);

// Get philox seed and offset tensorviews or random tensors for SDPA based on
// torch version.
std::pair<TensorView*, TensorView*> createSdpaRngTvs();
std::pair<at::Tensor, at::Tensor> createSdpaRngTensors();

// C++ implementation of torch.cuda.reset_peak_memory_stats. Note that this
// resets peak to current, not zero.
void resetPeakMemoryStats(c10::DeviceIndex device);

// C++ implementation of torch.cuda.max_memory_allocated
int64_t maxMemoryAllocated(const c10::DeviceIndex device);

// C++ implementation of torch.cuda.memory_allocated
int64_t memoryAllocated(const c10::DeviceIndex device);

} // namespace nvfuser
