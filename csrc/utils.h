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

#include <debug.h>
#include <type.h>

#include <deque>
#include <optional>
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

//! Find common device among tensor inputs. If no tensor inputs are found and
//! the selected_device argument is omitted, a default value of 0 is returned.
//! If no tensor inputs are found and selected_device is provided,
//! selected_device will be returned. If tensor inputs are found their devices
//! must match one another, and if selected_device is given they must match it
//! as well, otherwise -1 is returned.
int8_t getCommonDeviceCUDA(
    const at::ArrayRef<c10::IValue>& inputs,
    std::optional<int8_t> selected_device = std::nullopt);

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
static auto hasToStringHelper(int)
    -> decltype(std::declval<typename std::remove_pointer<T>::type>().toString(), std::true_type{});

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

template <typename ContainerOfStatement>
std::string toDelimitedInlineString(
    const ContainerOfStatement& container,
    std::string delim = ", ") {
  std::stringstream ss;
  bool first_val = true;
  for (const auto& item : container) {
    if (!first_val) {
      ss << delim;
    }
    ss << item->toInlineString();
    first_val = false;
  }
  return ss.str();
}

template <typename... Args>
class DebugPrintScope {
 public:
  DebugPrintScope(std::string name, Args... args) : name_(std::move(name)) {
    debug() << "Entering " << name_ << "("
            << toDelimitedString(std::forward_as_tuple(args...)) << ")"
            << std::endl;
  }
  ~DebugPrintScope() {
    debug() << "Leaving " << name_ << std::endl;
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

template <typename>
struct is_std_vector : std::false_type {};

template <typename T, typename A>
struct is_std_vector<std::vector<T, A>> : std::true_type {};

template <typename T>
constexpr auto is_std_vector_v = is_std_vector<T>::value;

//! Alter an existing hash in order to combine it with a new hash in a way that
//! is order-dependent and spreads bits over the entire range of a size_t.
//! Inspired by boost::hash_combine. See https://stackoverflow.com/q/35985960
inline void hashCombine(size_t& hash, size_t new_hash) {
  hash ^= new_hash + 0x9e3779b9 + (hash << 6) + (hash >> 2);
}

//! A wrapper to std::getenv. env_name is prepended with NVFUSER_.
char* getNvFuserEnv(const char* env_name);

} // namespace nvfuser
