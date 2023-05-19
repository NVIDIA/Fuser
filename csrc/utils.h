// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#pragma once

#include <ATen/ATen.h>
#include <c10/util/Exception.h>
#include <torch/csrc/jit/ir/ir.h>
#include <type.h>

#include <deque>
#include <sstream>
#include <string>
#include <type_traits>
#include <typeinfo>
#include <vector>

//! IR header hierarchy
//! 1. ** utils.h ** - PolymorphicBase and NonCopyable
//! 2. ir/base_nodes.h - Statement, Expr, and Val
//! 3. ir/internal_base_nodes.h - IterDomain and TensorDomain
//! 4. ir/interface_nodes.h - TensorView and Scalar
//! 5. ir/internal_nodes.h ** - Any internal-only IR nodes

namespace nvfuser {

void debugPrint(const c10::TensorTypePtr& type);

bool is_zero_dim_tensor(const std::shared_ptr<c10::TensorType>& tensor_type);
bool is_zero_sized_tensor(const std::shared_ptr<c10::TensorType>& tensor_type);

bool is_cpu_scalar(const at::Tensor& tensor);
bool is_cpu_scalar(const c10::TensorType& tensor_type);

// TODO: merge these two
// check if input is compatible with 32b index mode
int8_t getCommonDeviceCUDA(const at::ArrayRef<c10::IValue>& inputs);

//! Types of debug print-outs
//!
//! These can be set through the `PYTORCH_NVFUSER_DUMP` environment variable
//!
enum class DebugDumpOption {
  FusionIr, //!< Dump the Fusion IR before lowering
  FusionIrMath, //!< Dump just the compute (math) part of the Fusion IR
  FusionIrPresched, //!< Dump the Fusion IR before it is scheduled.
  KernelIr, //!< Dump the compiler Kernel IR
  ComputeAtMap, //!< Dump the computeAt map
  CudaKernel, //!< Dump the generated CUDA C++ kernel code
  CudaFull, //!< Dump the complete CUDA C++ code
  CudaToFile, //!< Dump CUDA Strings to File
  DebugInfo, //!< Embed line info and debug info to compiled kernel, and dump
             //!< the full CUDA C++ code
  AssertMemoryViolation, //!< Assert in the kernel when accessing global tensor
                         //!< out of bound. This might hurt performance.
  LaunchParam, //!< Dump the Launch parameters of kernel
  FusionSegments, //!< Dump Segmented Fusion Graph
  FusionSegmenterLog, //!< Dump Detailed Segmenter Logging
  FusionArgs, //!< Print the runtime fusion arguments
  KernelArgs, //!< Print the runtime kernel arguments when launching kernels
  EffectiveBandwidth, //! Measure kernel performance and print effective
                      //! bandwidth
  FusionSegmentsDrawing, //!< Dump Segmented Fusion Graph
  PrintPtxasLog, //!< Print the ptxas verbose log including register usage
  BufferReuseInfo, //!< Dump the analysis details of local/shared buffer re-use
  SchedulerDebug, //! Dump scheduler heuristic parameters
  SchedulerVerbose, //! Dump detailed scheduler logging
  ParallelDimensions, //!< Dump known parallel dimensions
  Halo, //! Halo information of tensors
  PerfDebugVerbose, //! When running kernels, print verbose information
                    //! associated with what's running
  PythonDefinition, //! Python Frontend Fusion Definition.
  PythonFrontendDebug, //! Python Frontend debug information.
  TransformPropagator, //! When running TransformPropagator, print propagation
                       //! path and replay result
  Cubin, //! Dump compiled CUBIN
  Sass, // Dump disassembled SASS
  Ptx, //! Dump compiled PTX
  BankConflictInfo, //! Dump bank confliction info
  SyncMap, //! RAW dependency info
  LowerVerbose, //! Print all passes' transform in GpuLower::lower
  ExprSimplification, //! Print all passes' transform in simplifyExpr
  ExprSort, //! Print merging decisions on expression sorting
  LoopRotation, //! Print loop rotation log
  MatmulChecks, //! Print logs from tools around matmul scheduler used in
                //! segmenter
  IndexType, //! Print the index type of the launched kernel
  EndOfOption //! Placeholder for counting the number of elements
};

TORCH_CUDA_CU_API bool isDebugDumpEnabled(DebugDumpOption option);
TORCH_CUDA_CU_API const std::vector<std::string>& getDebugDumpArguments(
    DebugDumpOption option);

//! Types of features to disable
//!
//! These can be set through the `PYTORCH_NVFUSER_DISABLE` environment variable
//!
enum class DisableOption {
  CompileToSass, //! Disable direct compilation to sass so the ptx can be
                 //! examined
  Fallback, //! Disable fallback
  Fma, //! Disable FMA instructions
  GroupedGridWelfordOuterOpt, //! Disable use of outer-optimized
                              //! grouped grid welford kernel
  IndexHoist, //! Disable index hoisting
  ExprSimplify, //! Disable expression simplifier
  Nvtx, //! Disable NVTX instrumentation
  PredicateElimination, //! Disable predicate elimination
  WelfordVectorization, //! Disable vectorizaton of Welford ops
  MagicZero, //! Disable nvfuser_zero
  EndOfOption //! Placeholder for counting the number of elements
};

// used only for testing/debugging
class TORCH_CUDA_CU_API ThreadLocalFmaDisableOverwrite {
 public:
  ThreadLocalFmaDisableOverwrite(bool flag = true);
  ~ThreadLocalFmaDisableOverwrite();

 private:
  bool old_flag_;
};

TORCH_CUDA_CU_API bool isOptionDisabled(DisableOption option);
TORCH_CUDA_CU_API const std::vector<std::string>& getDisableOptionArguments(
    DisableOption option);

//! Types of features to enable
//!
//! These can be set through the `PYTORCH_NVFUSER_ENABLE` environment variable
//!
enum class EnableOption {
  Complex, //! Enable complex support on python
  KernelProfile, //! Enable intra-kernel performance profiling
  LinearDecomposition, //! Enable linear-bias decomposition
  ConvDecomposition, //! Enable conv-bias decomposition
  GraphOp, //! Enable graphOps(index_select/gather/scatter)
  KernelDb, //! Enable Kernel Database
  WarnRegisterSpill, //! Enable warnings of register spill
  EndOfOption //! Placeholder for counting the number of elements
};

TORCH_CUDA_CU_API bool isOptionEnabled(EnableOption option);
TORCH_CUDA_CU_API const std::vector<std::string>& getEnableOptionArguments(
    EnableOption option);
TORCH_CUDA_CU_API int64_t
getRegPerThreadGivenThreadsPerSM(int64_t threads_per_sm);

TORCH_CUDA_CU_API int64_t
getThreadsPerSMGivenRegPerThread(int64_t reg_per_thread);

// Check if fallback path should be used which will dispatch to eager mode if
// any errors are encountered. Helpful for debugging.
bool useFallback();

//! Ceil integer division
constexpr int64_t ceilDiv(int64_t a, int64_t b) {
  return (a + b - 1) / b;
}

//! Simple mixin for suppressing copy & move operations, ex:
//!
//!  class Foo : public NonCopyable {
//!   ...
//!  };
//!
class NonCopyable {
 public:
  NonCopyable() = default;

  // No copy/move semantics
  NonCopyable(const NonCopyable&) = delete;
  NonCopyable& operator=(const NonCopyable&) = delete;
};

//! A generic root for a hierarchy of polymorphic classes:
//! - It ensures virtual destructors
//! - Provides the base->as<Derived>() and node->isA<T>() notation
class PolymorphicBase {
 public:
  virtual ~PolymorphicBase() = default;

  // Replacement for static_cast<T*>(ptr): ptr->as<T>()
  // (checked in DEBUG builds)
  template <class T>
  T* as() {
#ifdef NDEBUG
    auto downcast_ptr = static_cast<T*>(this);
#else
    auto downcast_ptr = dynamic_cast<T*>(this);
    TORCH_INTERNAL_ASSERT(downcast_ptr != nullptr);
#endif
    return downcast_ptr;
  }

  template <class T>
  const T* as() const {
#ifdef NDEBUG
    auto downcast_ptr = static_cast<const T*>(this);
#else
    auto downcast_ptr = dynamic_cast<const T*>(this);
    TORCH_INTERNAL_ASSERT(downcast_ptr != nullptr);
#endif
    return downcast_ptr;
  }

  //! Check if the runtime type is T (or derived from T)
  //!
  //! \note Don't use this for conditional casts. Instead, use:
  //!
  //!  if (auto t = dynamic_cast<T>(p)) { ... }
  //!
  //! instead of:
  //!
  //!  if (p->isA<T>()) { auto t = p->as<T>(); ... }
  //!
  template <class T>
  bool isA() const {
    return dynamic_cast<const T*>(this) != nullptr;
  }

  //! Check if the runtime type is strictly T. Returns false for classes
  //! derived from T
  template <class T>
  bool isStrictlyA() const {
    return typeid(*this) == typeid(T);
  }

 private:
  template <int> // unused template argument
  bool isOneOf() const {
    return false;
  }
  template <int, class T1, class... T>
  bool isOneOf() const {
    return isA<T1>() || isOneOf<0, T...>();
  }
  template <int> // unused template argument
  bool isStrictlyOneOf() const {
    return false;
  }
  template <int, class T1, class... T>
  bool isStrictlyOneOf() const {
    return isStrictlyA<T1>() || isStrictlyOneOf<0, T...>();
  }

 public:
  //! Check if the runtime type is one of the given types (or derived from
  //! one of the given types)
  template <class... T>
  bool isOneOf() const {
    return isOneOf<0, T...>();
  }

  //! Check if the runtime type is strictly one of the given types. Derived
  //! types not in the given list does not count.
  template <class... T>
  bool isStrictlyOneOf() const {
    return isStrictlyOneOf<0, T...>();
  }
};

template <class T, std::enable_if_t<std::is_enum<T>::value, bool> = true>
constexpr unsigned int switch_pair(T t1, T t2) {
  constexpr unsigned int _WORD_SHIFT = 16;
  return ((unsigned int)t1 << _WORD_SHIFT) + (unsigned int)t2;
}

std::vector<int64_t> getTensorSizes(at::TensorTypePtr const& tensor_type);

//! Return a sorted list of keys of an unordered map so that it can be
//! iterated deterministically
template <typename KeyType, typename ValueType, typename Cmp>
std::vector<KeyType> getSortedKeys(
    const std::unordered_map<KeyType, ValueType>& map,
    Cmp cmp) {
  std::vector<KeyType> keys(map.size());
  auto keys_it = keys.begin();
  for (const auto& kv : map) {
    *keys_it = kv.first;
    ++keys_it;
  }
  std::sort(keys.begin(), keys.end(), cmp);
  return keys;
}

// Based on https://stackoverflow.com/a/9154394
template <typename T>
static auto hasToStringHelper(int) -> decltype(
    std::declval<typename std::remove_pointer<T>::type>().toString(),
    std::true_type{});

template <typename>
static auto hasToStringHelper(long) -> std::false_type;

template <class T>
struct hasToString : decltype(hasToStringHelper<T>(0)) {};

// If T::toString() is defined, use the toString() to get its
// string. If std::stringstream << is defined for T, then use <<.
// otherwise, just returns a "<attr>"

template <typename T>
struct Printer {
  static std::string toString(const T& value) {
    if constexpr (hasToString<T>()) {
      if constexpr (std::is_pointer<T>::value) {
        return value->toString();
      } else {
        return value.toString();
      }
    } else {
      return "<attr>";
    }
  }
};

#if 0

// Waiting for C++20....

#include <concepts>

template<typename T>
concept Printable = requires(T a)
{
  { std::stringstream{} << a } -> std::convertible_to<std::stringstream>;
};

template <Printable T>
struct Printer<T> {
  static std::string toString(const T& value) {
    std::stringstream ss;
    ss << value;
    return ss.str();
  }
};

#else

#define SPECIALIZE_PRINTER(T)                     \
  template <>                                     \
  struct Printer<T> {                             \
    static std::string toString(const T& value) { \
      std::stringstream ss;                       \
      ss << value;                                \
      return ss.str();                            \
    }                                             \
  }

SPECIALIZE_PRINTER(bool);
SPECIALIZE_PRINTER(int);
SPECIALIZE_PRINTER(std::string);
SPECIALIZE_PRINTER(int64_t);
SPECIALIZE_PRINTER(DataType);
SPECIALIZE_PRINTER(MemoryType);
SPECIALIZE_PRINTER(UnaryOpType);
SPECIALIZE_PRINTER(BinaryOpType);
SPECIALIZE_PRINTER(TernaryOpType);
SPECIALIZE_PRINTER(LoadStoreOpType);
SPECIALIZE_PRINTER(DoubleBufferLoopStage);
SPECIALIZE_PRINTER(Swizzle2DType);
SPECIALIZE_PRINTER(SwizzleMode);
SPECIALIZE_PRINTER(std::vector<int>);
SPECIALIZE_PRINTER(std::vector<int64_t>);

#undef SPECIALIZE_PRINTER

#endif // if 0

// Stringification with delimiter
template <typename Iterator>
std::string toDelimitedString(
    Iterator first,
    Iterator last,
    std::string delim = ", ") {
  std::stringstream ss;
  bool first_val = true;
  for (auto it = first; it != last; ++it) {
    if (!first_val) {
      ss << delim;
    }
    ss << Printer<typename Iterator::value_type>::toString(*it);
    first_val = false;
  }
  return ss.str();
}

template <typename Printable>
std::string toDelimitedString(
    const std::vector<Printable>& vec,
    std::string delim = ", ") {
  return toDelimitedString(vec.begin(), vec.end(), delim);
}

template <typename Printable>
std::string toDelimitedString(
    const std::deque<Printable>& dq,
    std::string delim = ", ") {
  return toDelimitedString(dq.begin(), dq.end(), delim);
}

template <typename Printable>
std::string toDelimitedString(
    const std::unordered_set<Printable>& set,
    std::string delim = ", ") {
  return toDelimitedString(set.begin(), set.end(), delim);
}

template <int64_t index, int64_t stop, int64_t step, typename func_t>
void unrolled_for(func_t fun) {
  if constexpr (index < stop) {
    fun(std::integral_constant<int64_t, index>());
    unrolled_for<index + step, stop>(fun);
  }
}

template <int64_t index, int64_t stop, typename func_t>
void unrolled_for(func_t fun) {
  unrolled_for<index, stop, 1>(fun);
}

template <int64_t stop, typename func_t>
void unrolled_for(func_t fun) {
  unrolled_for<0, stop>(fun);
}

template <typename... Args>
std::string toDelimitedString(
    const std::tuple<Args...>& args,
    std::string delim = ", ") {
  std::stringstream ss;
  bool first_val = true;
  unrolled_for<sizeof...(Args)>([&](auto i) {
    if (!first_val) {
      ss << delim;
    }
    auto item = std::get<decltype(i)::value>(args);
    ss << Printer<decltype(item)>::toString(item);
    first_val = false;
  });
  return ss.str();
}

template <typename... Args>
class DebugPrintScope {
 public:
  DebugPrintScope(std::string name, Args... args) : name_(std::move(name)) {
    std::cout << "Entering " << name_ << "("
              << toDelimitedString(std::forward_as_tuple(args...)) << ")"
              << std::endl;
  }
  ~DebugPrintScope() {
    std::cout << "Leaving " << name_ << std::endl;
  }

 private:
  std::string name_;
};

// Note: ##__VA_ARGS__ is not C++ stardard, but it should work on gcc and clang.
// Compared to __VA_ARGS__, ##__VA_ARGS__ automatically remove the preceding
// comma when empty, allowing empty variadic parameters. If using other
// compiler, please use DebugPrintScope directly without this macro.
#define DEBUG_PRINT_SCOPE(...) \
  DebugPrintScope _debug_print_scope(__func__, ##__VA_ARGS__)

// Computes the index type required.
// Made into a class w/ state to allow reuse with
// different tensors and without needing to pass an allocated
// vector of size+stride
class KernelIndexTypeCompute {
  // Save 1 more bit besides the sign bit to be conservative
  static constexpr int64_t most_positive_int32_index =
      std::numeric_limits<int>::max() / 2;

 public:
  // Updates counters and returns current reqd mode
  inline PrimDataType addDim(int64_t size, int64_t stride) {
    if (size > 1) {
      TORCH_INTERNAL_ASSERT(
          stride >= 0, "Negative stride is not supported: ", stride);
      if (stride > 0) {
        // Accumulate positive stride
        tensor_most_positive_index_ += (size - 1) * stride;
      }
    }
    return getType();
  }

  inline PrimDataType getType() const {
    if (tensor_most_positive_index_ > most_positive_int32_index) {
      return PrimDataType::Int;
    } else {
      return PrimDataType::Int32;
    }
  }

 private:
  int64_t tensor_most_positive_index_ = 0;
};

} // namespace nvfuser
