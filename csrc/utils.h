// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#pragma once

#include <concepts>
#include <coroutine>
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

#include <ATen/ATen.h>
#include <c10/core/thread_pool.h>

#include <debug.h>
#include <exceptions.h>
#include <mma_type.h>
#include <options.h>
#include <tma.h>
#include <type.h>
#include <visibility.h>
#include <C++23/utility>

//! IR header hierarchy
//! 1. ** utils.h ** - PolymorphicBase and NonCopyable
//! 2. ir/base_nodes.h - Statement, Expr, and Val
//! 3. ir/internal_base_nodes.h - IterDomain and TensorDomain
//! 4. ir/interface_nodes.h - TensorView and Scalar
//! 5. ir/internal_nodes.h ** - Any internal-only IR nodes

namespace nvfuser {

//! Warp specialization padded threads count
constexpr int64_t kWarpSpecializationPaddedThreads = 128;

//! TMA hardware limit: maximum elements per dimension in a TMA box
constexpr int64_t kMaxElementsPerTmaBoxDim = 256;

//! In general TMA terminology, "box" is the dense rectangular region loaded,
//! while "tile" is a potentially strided subset. The pointwise scheduler uses
//! dense tiles (tile = box), so we use "tile" terminology for consistency.
constexpr int64_t kMaxElementsPerTmaTileDim = kMaxElementsPerTmaBoxDim;

//! shared memory alignment in bytes
//! TMA requires 128 bytes alignment, other usage doesn't have such requirement,
//! but still align to 128 bytes for simplicity and robustness.
//! When shared memory swizzling is used, up to 1024 bytes alignment can be
//! required for swizzling and they are handled by:
//! getSharedMemoryByteAlignment(MmaInputSmemSwizzle swizzle).
constexpr int64_t kSharedMemoryAlignmentBytes = 128;

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

// Get the maximum vectorization size in bits for the current CUDA device
int64_t getMaxVectorizationSizeInBit();

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

constexpr int64_t alignSharedMemoryBits(int64_t unaligned_bits) {
  constexpr int64_t alignment = kSharedMemoryAlignmentBytes * 8;
  return (unaligned_bits + (alignment - 1)) & (~(alignment - 1));
}

constexpr int64_t alignSharedMemoryBytes(int64_t unaligned_bytes) {
  constexpr int64_t alignment = kSharedMemoryAlignmentBytes;
  return (unaligned_bytes + (alignment - 1)) & (~(alignment - 1));
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
NVF_API const char* getNvFuserEnv(
    const std::string& env_name,
    const char* default_value = nullptr);

// Returns the mapped value or the default.
template <
    typename MapKey,
    typename Value,
    typename Key,
    typename = std::enable_if_t<std::is_convertible_v<Key, MapKey>>>
Value getOrDefault(
    const std::unordered_map<MapKey, Value>& map,
    const Key& key,
    const Value& default_value = Value()) {
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

// Returns a range of integers [start, end)
auto arange(auto start, auto end) {
  static_assert(std::is_integral<decltype(start)>());
  static_assert(std::is_integral<decltype(end)>());
  // If start and end are the same type, use the range directly
  if constexpr (std::is_same_v<decltype(start), decltype(end)>) {
    return std::ranges::iota_view(start, end);
  }
  return std::ranges::iota_view(decltype(end)(start), end);
}

// Returns a range of integers [0, end)
auto arange(auto end) {
  static_assert(std::is_integral<decltype(end)>());
  return std::ranges::iota_view(decltype(end)(0), end);
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

using std::views::enumerate;
using std::views::zip;

#else

namespace views {
template <std::ranges::view... Rs>
class zip_view : public std::ranges::view_interface<zip_view<Rs...>> {
 private:
  std::tuple<Rs...> bases;

  static constexpr bool is_bidirectional =
      (std::ranges::bidirectional_range<Rs> && ...);

  struct iterator {
    std::tuple<std::ranges::iterator_t<Rs>...> iterators;

    using value_type = std::tuple<std::ranges::range_value_t<Rs>...>;
    using reference = std::tuple<std::ranges::range_reference_t<Rs>...>;
    using difference_type = std::ptrdiff_t;
    using iterator_category = std::conditional_t<
        is_bidirectional,
        std::bidirectional_iterator_tag,
        std::input_iterator_tag>;

    iterator& operator++() {
      std::apply([](auto&... it) { ((++it), ...); }, iterators);
      return *this;
    }

    iterator operator++(int) {
      iterator temp = *this;
      ++(*this);
      return temp;
    }

    // Enable bidirectional iteration only if `is_bidirectional` is true
    iterator& operator--() requires is_bidirectional {
      std::apply([](auto&... it) { ((--it), ...); }, iterators);
      return *this;
    }

    iterator operator--(int) requires is_bidirectional {
      iterator temp = *this;
      --(*this);
      return temp;
    }

    reference operator*() const {
      return std::apply(
          [](auto&... it) -> reference { return {*it...}; }, iterators);
    }

    bool operator==(const iterator& other) const = default;
  };

  struct sentinel {
    std::tuple<std::ranges::sentinel_t<Rs>...> sentinels;

    bool operator==(const iterator& it) const {
      return compare(it, std::make_index_sequence<sizeof...(Rs)>{});
    }

   private:
    template <std::size_t... I>
    bool compare(const iterator& it, std::index_sequence<I...>) const {
      return ((std::get<I>(it.iterators) == std::get<I>(sentinels)) || ...);
    }
  };

 public:
  zip_view() = default;
  explicit zip_view(Rs... ranges) : bases{std::move(ranges)...} {}

  auto begin() {
    return iterator{std::apply(
        [](auto&... r) { return std::tuple{std::ranges::begin(r)...}; },
        bases)};
  }

  auto end() {
    return sentinel{std::apply(
        [](auto&... r) { return std::tuple{std::ranges::end(r)...}; }, bases)};
  }
};

// Deduction guide
template <std::ranges::viewable_range... Rs>
zip_view(Rs&&...) -> zip_view<std::views::all_t<Rs>...>;

// Helper function
template <std::ranges::viewable_range... Rs>
auto zip(Rs&&... rs) {
  return zip_view{std::forward<Rs>(rs)...};
}

template <std::ranges::view V>
class enumerate_view : public std::ranges::view_interface<enumerate_view<V>> {
 private:
  V base_;

  // Base iterator
  template <typename BaseIterator, bool IsBidirectional>
  struct iterator_base {
    using base_iterator = BaseIterator;
    using value_type =
        std::pair<std::size_t, std::ranges::range_reference_t<V>>;
    using reference = std::pair<std::size_t, std::ranges::range_reference_t<V>>;
    using difference_type = std::ranges::range_difference_t<V>;
    using iterator_category = std::conditional_t<
        IsBidirectional,
        std::bidirectional_iterator_tag,
        std::forward_iterator_tag>;

    base_iterator current_;
    int64_t index_;

    iterator_base() = default;
    iterator_base(base_iterator current, std::size_t index)
        : current_(current), index_(index) {}

    reference operator*() const {
      return {index_, *current_};
    }

    iterator_base& operator++() {
      ++current_;
      ++index_;
      return *this;
    }

    iterator_base operator++(int) {
      iterator_base tmp = *this;
      ++(*this);
      return tmp;
    }

    // Enable bidirectional iteration only if IsBidirectional == true
    iterator_base& operator--() requires IsBidirectional {
      --current_;
      --index_;
      return *this;
    }

    iterator_base operator--(int) requires IsBidirectional {
      iterator_base tmp = *this;
      --(*this);
      return tmp;
    }

    bool operator==(const iterator_base& other) const = default;
  };

  struct sentinel {
    std::ranges::sentinel_t<V> end_;

    bool operator==(const auto& it) const {
      return it.current_ == end_;
    }
  };

  using base_iterator = std::ranges::iterator_t<V>;
  static constexpr bool is_bidirectional = std::ranges::bidirectional_range<V>;

  using iterator_type = iterator_base<base_iterator, is_bidirectional>;

 public:
  enumerate_view() = default;
  explicit enumerate_view(V base) : base_(std::move(base)) {}

  auto begin() {
    return iterator_type{std::ranges::begin(base_), 0};
  }
  auto end() {
    return sentinel{std::ranges::end(base_)};
  } // Use sentinel for forward iterators

  V base() const& {
    return base_;
  }
  V base() && {
    return std::move(base_);
  }
};

// Deduction guide
template <std::ranges::viewable_range R>
enumerate_view(R&&) -> enumerate_view<std::views::all_t<R>>;

// Helper function
auto enumerate(std::ranges::viewable_range auto&& r) {
  return enumerate_view{std::forward<decltype(r)>(r)};
}

} // namespace views

using views::enumerate;
using views::zip;

#endif // C++23

// Helper: turn T into reference_wrapper<U> if T is reference
template <typename T>
using Yielded = std::conditional_t<
    std::is_reference_v<T>,
    std::reference_wrapper<std::remove_reference_t<T>>,
    T>;

// Writing yield in C++20 just like Python:
// See NVFuserTest.Generator[1-5] for usage examples
template <typename T>
class Generator : public std::ranges::view_interface<Generator<T>> {
 public:
  struct promise_type;
  using handle_type = std::coroutine_handle<promise_type>;
  using stored_type = Yielded<T>;

  Generator(handle_type h) : coroutine_(h) {}
  Generator(Generator&& other) noexcept : coroutine_(other.coroutine_) {
    other.coroutine_ = nullptr;
  }
  Generator& operator=(Generator&& other) noexcept {
    if (this != &other) {
      if (coroutine_) {
        coroutine_.destroy();
      }
      coroutine_ = other.coroutine_;
      other.coroutine_ = nullptr;
    }
    return *this;
  }
  ~Generator() {
    if (coroutine_) {
      coroutine_.destroy();
    }
  }
  Generator(const Generator&) = delete;
  Generator& operator=(const Generator&) = delete;

  struct iterator {
    using value_type = std::remove_reference_t<T>;
    using reference = T;
    using difference_type = std::ptrdiff_t;
    using iterator_category = std::input_iterator_tag;

    iterator() = default;
    explicit iterator(handle_type h) : coroutine(h) {
      ++(*this);
    }

    reference operator*() const {
      if constexpr (std::is_reference_v<T>) {
        return value->get(); // unwrap reference_wrapper<T>
      } else {
        return *value;
      }
    }

    iterator& operator++() {
      coroutine.resume();
      if (coroutine.done()) {
        if (coroutine.promise().exception) {
          std::rethrow_exception(coroutine.promise().exception);
        }
        value.reset();
      } else {
        value = std::ref(coroutine.promise().current_value);
      }
      return *this;
    }

    iterator operator++(int) {
      auto tmp = *this;
      ++(*this);
      return tmp;
    }
    bool operator==(std::default_sentinel_t) const {
      return !value.has_value();
    }
    bool operator!=(std::default_sentinel_t) const {
      return value.has_value();
    }
    friend bool operator==(std::default_sentinel_t s, const iterator& it) {
      return it == s;
    }
    friend bool operator!=(std::default_sentinel_t s, const iterator& it) {
      return it != s;
    }

    handle_type coroutine = nullptr;
    std::optional<stored_type> value;
  };

  iterator begin() const {
    return iterator{coroutine_};
  }
  std::default_sentinel_t end() const {
    return {};
  }

 private:
  handle_type coroutine_;

 public:
  struct promise_type {
    std::optional<stored_type> current_value;
    std::exception_ptr exception;

    auto get_return_object() {
      return Generator{handle_type::from_promise(*this)};
    }
    std::suspend_always initial_suspend() {
      return {};
    }
    std::suspend_always final_suspend() noexcept {
      return {};
    }
    std::suspend_always yield_value(T value) {
      if constexpr (std::is_reference_v<T>) {
        current_value = std::ref(value); // wraps T& as reference_wrapper
      } else {
        current_value = std::move(value);
      }
      return {};
    }

    void return_void() {}
    void unhandled_exception() {
      exception = std::current_exception();
    }
  };
};

} // namespace nvfuser
