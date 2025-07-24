// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
// This is a refactor of the NVF_ERROR and NVF_CHECK macros
// from PyTorch for implementing NVFuser specific macros.

#pragma once

#include <array>
#include <cstdint>
#include <deque>
#include <exception>
#include <iosfwd>
#include <optional>
#include <sstream>
#include <string>
#include <vector>

#include <visibility.h>

namespace nvfuser {

// This function will demangle the mangled function name into a more human
// readable format, e.g. _Z1gv -> g().
// More information:
// https://github.com/gcc-mirror/gcc/blob/master/libstdc%2B%2B-v3/libsupc%2B%2B/cxxabi.h
// NOTE: `__cxa_demangle` returns a malloc'd string that we have to free
// ourselves.
std::string demangle(const char* name);

std::string _get_backtrace(
    size_t frames_to_skip = 0,
    size_t maximum_number_of_frames = 64,
    bool skip_python_frames = true);

struct SourceLocation {
  const char* function;
  const char* file;
  uint32_t line;
};

std::ostream& operator<<(std::ostream& out, const SourceLocation& loc);

struct CompileTimeEmptyString {
  operator const std::string&() const {
    static const std::string empty_string_literal;
    return empty_string_literal;
  }
  operator const char*() const {
    return "";
  }
};

template <typename T>
struct CanonicalizeStrTypes {
  using type = const T&;
};

template <size_t N>
struct CanonicalizeStrTypes<std::array<char, N>> {
  using type = const char*;
};

inline std::ostream& _to_str(std::ostream& ss) {
  return ss;
}

template <typename T>
inline std::ostream& _to_str(std::ostream& ss, const T& t) {
  // NOLINTNEXTLINE(clang-analyzer-core.CallAndMessage)
  ss << t;
  return ss;
}

template <>
inline std::ostream& _to_str<CompileTimeEmptyString>(
    std::ostream& ss,
    const CompileTimeEmptyString&) {
  return ss;
}

template <typename T, typename... Args>
inline std::ostream& _to_str(
    std::ostream& ss,
    const T& t,
    const Args&... args) {
  return _to_str(_to_str(ss, t), args...);
}

template <typename... Args>
struct _str_wrapper final {
  static std::string call(const Args&... args) {
    std::ostringstream ss;
    _to_str(ss, args...);
    return ss.str();
  }
};

// Specializations for already-a-string types.
template <>
struct _str_wrapper<std::string> final {
  // return by reference to avoid the binary size of a string copy
  static const std::string& call(const std::string& str) {
    return str;
  }
};

template <>
struct _str_wrapper<const char*> final {
  static const char* call(const char* str) {
    return str;
  }
};

// For nvfuser::to_str() with an empty argument list (which is common in our
// assert macros), we don't want to pay the binary size for constructing and
// destructing a stringstream or even constructing a string.
template <>
struct _str_wrapper<> final {
  static CompileTimeEmptyString call() {
    return CompileTimeEmptyString();
  }
};

// Convert a list of string-like arguments into a single string.
template <typename... Args>
inline decltype(auto) to_str(const Args&... args) {
  return _str_wrapper<typename CanonicalizeStrTypes<Args>::type...>::call(
      args...);
}

class NVF_API nvfError : public std::exception {
  // The actual error message.
  std::string msg_;
  // Context for the message (in order of decreasing specificity).  Context will
  // be automatically formatted appropriately, so it is not necessary to add
  // extra leading/trailing newlines to strings inside this vector
  std::vector<std::string> context_;
  // The C++ backtrace at the point when this exception was raised.  This
  // may be empty if there is no valid backtrace.  (We don't use optional
  // here to reduce the dependencies this file has.)
  std::string backtrace_;
  // These two are derived fields from msg_stack_ and backtrace_, but we need
  // fields for the strings so that we can return a const char* (as the
  // signature of std::exception requires).  Currently, the invariant
  // is that these fields are ALWAYS populated consistently with respect
  // to msg_stack_ and backtrace_.
  std::string what_;
  std::string what_without_backtrace_;
  // This is a little debugging trick: you can stash a relevant pointer
  // in caller, and then when you catch the exception, you can compare
  // against pointers you have on hand to get more information about
  // where the exception came from.  In Caffe2, this is used to figure
  // out which operator raised an exception.
  const void* caller_;

 public:
  // PyTorch-style Error constructor.
  nvfError(SourceLocation source_location, std::string msg);
  // Base constructor
  nvfError(
      std::string msg,
      std::string backtrace,
      const void* caller = nullptr);
  // Add some new context to the message stack.  The last added context
  // will be formatted at the end of the context list upon printing.
  // WARNING: This method is O(n) in the size of the stack, so don't go
  // wild adding a ridiculous amount of context to error messages.
  void add_context(std::string msg);
  const std::string& msg() const {
    return msg_;
  }
  const std::vector<std::string>& context() const {
    return context_;
  }
  const std::string& backtrace() const {
    return backtrace_;
  }
  /// Returns the complete error message, including the source location.
  /// The returned pointer is invalidated if you call add_context() on
  /// this object.
  const char* what() const noexcept override {
    return what_.c_str();
  }
  const void* caller() const noexcept {
    return caller_;
  }
  /// Returns only the error message string, without source location.
  /// The returned pointer is invalidated if you call add_context() on
  /// this object.
  const char* what_without_backtrace() const noexcept {
    return what_without_backtrace_.c_str();
  }

 private:
  void refresh_what();
  std::string compute_what(bool include_backtrace) const;
};

[[noreturn]] NVF_API void nvfCheckFail(
    const char* func,
    const char* file,
    uint32_t line,
    const std::string& msg);

[[noreturn]] NVF_API void nvfCheckFail(
    const char* func,
    const char* file,
    uint32_t line,
    const char* msg);

[[noreturn]] void nvfErrorFail(
    const char* func,
    const char* file,
    uint32_t line,
    const char* condMsg,
    const char* userMsg);

[[noreturn]] inline void nvfErrorFail(
    const char* func,
    const char* file,
    uint32_t line,
    const char* condMsg,
    CompileTimeEmptyString /*userMsg*/) {
  nvfCheckFail(func, file, line, condMsg);
}

[[noreturn]] NVF_API void nvfErrorFail(
    const char* func,
    const char* file,
    uint32_t line,
    const char* condMsg,
    const std::string& userMsg);

template <typename... Args>
decltype(auto) nvfCheckMsgImpl(const char* /*msg*/, const Args&... args) {
  return to_str(args...);
}
inline const char* nvfCheckMsgImpl(const char* msg) {
  return msg;
}
// If there is just 1 user-provided C-string argument, use it.
inline const char* nvfCheckMsgImpl(const char* /*msg*/, const char* args) {
  return args;
}

} // namespace nvfuser

#define STRINGIZE_IMPL(x) #x
#define STRINGIZE(x) STRINGIZE_IMPL(x)

#define NVF_THROW(...) \
  nvfuser::nvfErrorFail(                                    \
        __FUNCTION__,                                       \
        __FILE__,                                           \
        static_cast<uint32_t>(__LINE__),                    \
        " INTERNAL ASSERT FAILED at "                       \
        STRINGIZE(__FILE__) ":" STRINGIZE(__LINE__)         \
        ", please report a bug with repro script to NVFuser at " \
        "https://github.com/NVIDIA/Fuser/issues. ",         \
        nvfuser::to_str(__VA_ARGS__));

#define NVF_ERROR(cond, ...) \
  if ((!(cond))) {           \
    NVF_THROW(__VA_ARGS__)   \
  }

#define NVF_COMPARISON_ERROR_MESSAGE(lhs, op, rhs) \
  "Expected " #lhs " " #op " " #rhs ", but found ", (lhs), " vs ", (rhs), ". "

#define NVF_ERROR_EQ(lhs, rhs, ...)               \
  NVF_ERROR(                                      \
      (lhs) == (rhs),                             \
      NVF_COMPARISON_ERROR_MESSAGE(lhs, ==, rhs), \
      ##__VA_ARGS__)

#define NVF_CHECK_MSG(cond, type, ...) \
  (nvfuser::nvfCheckMsgImpl(           \
      "Expected " #cond " to be true, but got false.  ", ##__VA_ARGS__))

#define NVF_CHECK(cond, ...)                     \
  if ((!(cond))) {                               \
    nvfuser::nvfCheckFail(                       \
        __func__,                                \
        __FILE__,                                \
        static_cast<uint32_t>(__LINE__),         \
        NVF_CHECK_MSG(cond, "", ##__VA_ARGS__)); \
  }

#define NVF_CHECK_EQ(lhs, rhs, ...)               \
  NVF_CHECK(                                      \
      (lhs) == (rhs),                             \
      NVF_COMPARISON_ERROR_MESSAGE(lhs, ==, rhs), \
      ##__VA_ARGS__)
