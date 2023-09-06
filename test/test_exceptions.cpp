// This is a refactor of the tests used for PyTorch macros --
// NVF_ERROR and NVF_CHECK.

#include <csrc/exceptions.h>
#include <gtest/gtest.h>
#include <stdexcept>

#include <test/utils.h>
#include <test/validator.h>

namespace nvfuser {

namespace {

template <class Functor>
inline void expectThrowsEq(Functor&& functor, const char* expectedMessage) {
  try {
    std::forward<Functor>(functor)();
  } catch (const nvfError& e) {
    EXPECT_STREQ(e.what_without_backtrace(), expectedMessage);
    return;
  }
  ADD_FAILURE() << "Expected to throw exception with message \""
                << expectedMessage << "\" but didn't throw";
}
} // namespace

TEST_F(NVFuserTest, ErrorFormatting) {
  expectThrowsEq(
      []() { NVF_CHECK(false, "This is invalid"); }, "This is invalid");
}

static int assertionArgumentCounter = 0;

namespace {
int getAssertionArgument() {
  return ++assertionArgumentCounter;
}

void failCheck() {
  NVF_CHECK(false, "message ", getAssertionArgument());
}

void failError() {
  NVF_ERROR(false, "message ", getAssertionArgument());
}
} // namespace

TEST_F(NVFuserTest, MultipleArgCalls) {
  assertionArgumentCounter = 0;
  // NOLINTNEXTLINE(cppcoreguidelines-avoid-goto,hicpp-avoid-goto)
  EXPECT_ANY_THROW(failCheck());
  EXPECT_EQ(assertionArgumentCounter, 1) << "NVF_CHECK called argument twice";

  // NOLINTNEXTLINE(cppcoreguidelines-avoid-goto,hicpp-avoid-goto)
  EXPECT_ANY_THROW(failError());
  EXPECT_EQ(assertionArgumentCounter, 2) << "NVF_ERROR called argument twice";
}

} // namespace nvfuser
