// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#pragma once

#include <ATen/ATen.h>
#include <exceptions.h>
#include <torch/csrc/jit/ir/ir.h>
#include <torch/torch.h>
#include <visibility.h>

#include <debug.h>
#include <mma_type.h>
#include <tma.h>
#include <type.h>

#include <c10/core/thread_pool.h>
#include <deque>
#include <iterator>
#include <memory>
#include <optional>
#include <ranges>
#include <regex>
#include <sstream>
#include <string>
#include <type_traits>
#include <typeinfo>
#include <unordered_map>
#include <vector>

#define NVF_TORCH_VERSION_GREATER(major, minor, patch)                \
  TORCH_VERSION_MAJOR > major ||                                      \
      (TORCH_VERSION_MAJOR == major && TORCH_VERSION_MINOR > minor || \
       (TORCH_VERSION_MINOR == minor && TORCH_VERSION_PATCH > patch))

#define NVF_TORCH_VERSION_NO_LESS(major, minor, patch)                \
  TORCH_VERSION_MAJOR > major ||                                      \
      (TORCH_VERSION_MAJOR == major && TORCH_VERSION_MINOR > minor || \
       (TORCH_VERSION_MINOR == minor && TORCH_VERSION_PATCH >= patch))

//! IR header hierarchy
//! 1. ** utils.h ** - PolymorphicBase and NonCopyable
//! 2. ir/base_nodes.h - Statement, Expr, and Val
//! 3. ir/internal_base_nodes.h - IterDomain and TensorDomain
//! 4. ir/interface_nodes.h - TensorView and Scalar
//! 5. ir/internal_nodes.h ** - Any internal-only IR nodes

namespace nvfuser {

class KernelArgumentHolder;

int getNumThreads();
c10::ThreadPool* getThreadPool();

std::string debug_str(const at::Tensor& tensor);

bool is_cpu_scalar(const at::Tensor& tensor);

bool is_meta_scalar(const at::Tensor& tensor);

//! Find common device among tensor inputs. If no tensor inputs are found and
//! the selected_device argument is omitted, a default value of 0 is returned.
//! If no tensor inputs are found and selected_device is provided,
//! selected_device will be returned. If tensor inputs are found their devices
//! must match one another, and if selected_device is given they must match it
//! as well, otherwise -1 is returned.
int8_t NVF_API getCommonDeviceCUDA(
    const KernelArgumentHolder& inputs,
    std::optional<int8_t> selected_device = std::nullopt);

int64_t getRegPerThreadGivenThreadsPerSM(int64_t threads_per_sm);

int64_t getThreadsPerSMGivenRegPerThread(int64_t reg_per_thread);

// Check if fallback path should be used which will dispatch to eager mode if
// any errors are encountered. Helpful for debugging.
bool useFallback();

//! Ceil integer division
constexpr int64_t ceilDiv(int64_t dividend, int64_t divisor) {
  return (dividend + divisor - 1) / divisor;
}

constexpr int64_t roundUpToMultiple(int64_t dividend, int64_t divisor) {
  return ceilDiv(dividend, divisor) * divisor;
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
#if defined(NDEBUG) && !defined(NVFUSER_EXPLICIT_ERROR_CHECK)
    auto downcast_ptr = static_cast<T*>(this);
#else
    auto downcast_ptr = dynamic_cast<T*>(this);
    NVF_ERROR(downcast_ptr != nullptr);
#endif // defined(NDEBUG) && !defined(NVFUSER_EXPLICIT_ERROR_CHECK)
    return downcast_ptr;
  }

  template <class T>
  const T* as() const {
#if defined(NDEBUG) && !defined(NVFUSER_EXPLICIT_ERROR_CHECK)
    auto downcast_ptr = static_cast<const T*>(this);
#else
    auto downcast_ptr = dynamic_cast<const T*>(this);
    NVF_ERROR(downcast_ptr != nullptr);
#endif // defined(NDEBUG) && !defined(NVFUSER_EXPLICIT_ERROR_CHECK)
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
using ConstCharStar = const char*;
SPECIALIZE_PRINTER(ConstCharStar);
using VoidStar = void*;
SPECIALIZE_PRINTER(VoidStar);
SPECIALIZE_PRINTER(uint32_t);
SPECIALIZE_PRINTER(int64_t);
SPECIALIZE_PRINTER(uint64_t);
SPECIALIZE_PRINTER(DataType);
SPECIALIZE_PRINTER(MemoryType);
SPECIALIZE_PRINTER(UnaryOpType);
SPECIALIZE_PRINTER(BinaryOpType);
SPECIALIZE_PRINTER(TernaryOpType);
SPECIALIZE_PRINTER(LoadStoreOpType);
SPECIALIZE_PRINTER(CircularBufferLoopStage);
SPECIALIZE_PRINTER(tma::TensorMapInterleave);
SPECIALIZE_PRINTER(tma::TensorMapL2Promotion);
SPECIALIZE_PRINTER(tma::TensorMapFloatOOBFill);
SPECIALIZE_PRINTER(MmaInputSmemSwizzle);
SPECIALIZE_PRINTER(SwizzleType);
SPECIALIZE_PRINTER(Swizzle2DType);
SPECIALIZE_PRINTER(SwizzleMode);
SPECIALIZE_PRINTER(std::vector<int>);
SPECIALIZE_PRINTER(std::vector<uint32_t>);
SPECIALIZE_PRINTER(std::vector<int64_t>);
SPECIALIZE_PRINTER(std::vector<uint64_t>);
SPECIALIZE_PRINTER(std::optional<bool>);

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
    std::initializer_list<Printable> list,
    std::string delim = ", ") {
  // toDelimitedString(list.begin(), list.end(), delim) doesn't work out of the
  // box, because list.begin() returns a Printable* not an iterator.
  return toDelimitedString(std::vector<Printable>(list), delim);
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

class DebugPrintScope {
 public:
  template <typename... Args>
  DebugPrintScope(std::string name, Args... args) : name_(std::move(name)) {
    debug() << "Entering " << name_ << "("
            << toDelimitedString(std::forward_as_tuple(args...)) << ")"
            << std::endl;
  }

  ~DebugPrintScope() {
    debug() << "Leaving " << name_;
    if (!return_.empty()) {
      debug() << " returning " << return_;
    }
    if (!file_.empty()) {
      debug() << " at " << file_;
    }
    if (line_ >= 0) {
      debug() << ":" << line_;
    }
    debug() << std::endl;
  }

  template <typename T>
  void setReturn(const T& ret, std::string file = "", int64_t line = -1) {
    return_ = Printer<std::decay_t<T>>::toString(ret);
    file_ = std::move(file);
    line_ = line;
  }

 private:
  // The name of the scope, as specified as the first argument of
  // DEBUG_PRINT_SCOPE_NAME. If using DEBUG_PRINT_SCOPE, then this is __func__.
  std::string name_;

  // Return value and location of the return statement.
  // Note that the recording of the return value is not automatic. The function
  // needs to be manually instrumented to replace `return XXX;` with
  // `RECORD_AND_RETURN(XXX)` to record the return value.
  std::string return_;
  std::string file_;
  int64_t line_ = -1;
};

// Debug printing the entering and leaving of a function. The given arguments
// will be printed when entering the function.
//
// Note: ##__VA_ARGS__ is not C++ stardard, but it should work on gcc and clang.
// Compared to __VA_ARGS__, ##__VA_ARGS__ automatically remove the preceding
// comma when empty, allowing empty variadic parameters. If using other
// compiler, please use DebugPrintScope directly without this macro.
#define DEBUG_PRINT_SCOPE_NAME(name, ...)                                 \
  std::unique_ptr<DebugPrintScope> _debug_print_scope;                    \
  if (isDebugDumpEnabled(DebugDumpOption::FunctionTrace)) {               \
    auto enabled = getDebugDumpArguments(DebugDumpOption::FunctionTrace); \
    for (auto pattern : enabled) {                                        \
      std::regex re(pattern);                                             \
      if (std::regex_match(name, re)) {                                   \
        _debug_print_scope =                                              \
            std::make_unique<DebugPrintScope>(name, ##__VA_ARGS__);       \
        break;                                                            \
      }                                                                   \
    }                                                                     \
  }

#define DEBUG_PRINT_SCOPE(...) DEBUG_PRINT_SCOPE_NAME(__func__, ##__VA_ARGS__)

#define DEBUG_LOG(...)                                    \
  if (_debug_print_scope) {                               \
    debug() << "[" << __FILE__ << ":" << __LINE__ << "] " \
            << to_str("", ##__VA_ARGS__) << std::endl;    \
  }

// Record the return value and return it.
#define RECORD_AND_RETURN(ret)                              \
  if (_debug_print_scope) {                                 \
    _debug_print_scope->setReturn(ret, __FILE__, __LINE__); \
  }                                                         \
  return ret

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
      NVF_ERROR(stride >= 0, "Negative stride is not supported: ", stride);
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
NVF_API char* getNvFuserEnv(const char* env_name);

// Returns the mapped value or the default.
template <typename K, typename V>
const V& getOrDefault(
    const std::unordered_map<K, V>& map,
    const K& key,
    const V& default_value = V()) {
  const auto i = map.find(key);
  return i == map.end() ? default_value : i->second;
}

size_t deviceAvailableSharedMemoryBytes();

inline int64_t wrapDim(int64_t dim, int64_t ndim) {
  if (dim < 0) {
    dim += ndim;
  }
  NVF_CHECK(
      dim >= 0 && dim < ndim,
      "Tried to access out of boundary index ",
      dim,
      ". total index: ",
      ndim);
  return dim;
}

// This is the same as the pow utility included in runtime/helpers.cu. It is
// included here to facilitate matching host-side computation.
template <typename T>
T pow(T a, T b) {
  if (b < 0) {
    if (a == 1) {
      return 1;
    } else if (a == -1) {
      auto negative = (-b) % static_cast<T>(2);
      return negative ? -1 : 1;
    } else {
      return 0;
    }
  } else {
    T result = 1;
    while (b) {
      if (b & 1) {
        result *= a;
      }
      b /= 2;
      a *= a;
    }
    return result;
  }
}

// Returns true if given number is power of 2
constexpr bool isPowOf2(int64_t x) {
  return x > 1 && (x & (x - 1)) == 0;
}

template <typename T>
using MaybeUniqueOwningPtr = dynamic_type::
    DynamicType<dynamic_type::NoContainers, T*, std::unique_ptr<T>>;

template <typename T>
void checkAllEqual(std::initializer_list<T> elements) {
  for (const auto& element : elements) {
    NVF_CHECK(
        element == *elements.begin(),
        "Expected all elements to be equal, but found ",
        element,
        " and ",
        *elements.begin(),
        " in [",
        toDelimitedString(elements),
        "]");
  }
}

#if __cplusplus >= 202302L

using std::views::zip;

#else

namespace views {

template <std::ranges::input_range... Rs>
class zip_view : public std::ranges::view_interface<zip_view<Rs...>> {
 private:
  std::tuple<Rs...> bases;

  // Iterator for begin()
  struct begin_iterator {
    std::tuple<std::ranges::iterator_t<Rs>...> iterators;

    using value_type = std::tuple<std::ranges::range_value_t<Rs>...>;
    using reference = std::tuple<std::ranges::range_reference_t<Rs>...>;
    using difference_type = std::ptrdiff_t;

    begin_iterator& operator++() {
      std::apply([](auto&... it) { ((++it), ...); }, iterators);
      return *this;
    }

    reference operator*() const {
      return std::apply(
          [](auto&... it) -> reference { return {*it...}; }, iterators);
    }

    bool operator==(const begin_iterator& other) const {
      return iterators == other.iterators;
    }
  };

  // Sentinel for end()
  struct end_iterator {
    std::tuple<std::ranges::sentinel_t<Rs>...> sentinels;

    bool operator==(const begin_iterator& it) const {
      return compare(it, std::make_index_sequence<sizeof...(Rs)>{});
    }

   private:
    template <std::size_t... I>
    bool compare(const begin_iterator& it, std::index_sequence<I...>) const {
      return ((std::get<I>(it.iterators) == std::get<I>(sentinels)) || ...);
    }
  };

 public:
  explicit zip_view(Rs&&... ranges)
      : bases{std::forward<Rs>(ranges)...} {} // Ensure perfect forwarding

  auto begin() {
    return begin_iterator{std::apply(
        [](auto&... r) { return std::tuple{std::ranges::begin(r)...}; },
        bases)};
  }

  auto end() {
    return end_iterator{std::apply(
        [](auto&... r) { return std::tuple{std::ranges::end(r)...}; }, bases)};
  }
};

template <std::ranges::input_range... Rs>
zip_view(Rs&&...) -> zip_view<Rs...>;

template <std::ranges::input_range... Rs>
auto zip(Rs&&... rs) {
  return zip_view{std::forward<Rs>(rs)...};
}
} // namespace views
using views::zip;

#endif // C++23

auto enumerate(auto&& range) {
  return zip(
      std::views::iota((int64_t)0), std::forward<decltype(range)>(range));
}

} // namespace nvfuser
