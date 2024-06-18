#include <gtest/gtest.h>

namespace nvfuser {
TEST(FooTest, Bar) {}
} // namespace nvfuser

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
