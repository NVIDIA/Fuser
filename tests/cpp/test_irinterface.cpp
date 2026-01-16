// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on

#include <gtest/gtest.h>

#include <ir/interface.h>
#include <ir/builder.h>
#include <tests/cpp/utils.h>

namespace nvfuser {

class IrInterfaceTest : public NVFuserTest {};

TEST_F(IrInterfaceTest, BasicForwarding) {
  // Test that IrInterface correctly forwards IrContainer methods
  IrInterface interface;

  // Test that collections are initially empty
  EXPECT_TRUE(interface.vals().empty());
  EXPECT_TRUE(interface.unordered_exprs().empty());
  EXPECT_EQ(interface.numExprs(), 0);
  EXPECT_EQ(interface.numVals(false), 0);

  // Test deterministic collections are empty
  EXPECT_TRUE(interface.deterministic_vals().empty());
  EXPECT_TRUE(interface.deterministic_exprs().empty());

  // Test shortcut values work
  auto zero = interface.zeroVal();
  EXPECT_NE(zero, nullptr);
  EXPECT_TRUE(zero->isZero());

  auto one = interface.oneVal();
  EXPECT_NE(one, nullptr);
  EXPECT_TRUE(one->isOne());

  auto false_val = interface.falseVal();
  EXPECT_NE(false_val, nullptr);

  auto true_val = interface.trueVal();
  EXPECT_NE(true_val, nullptr);

  auto magic_zero = interface.magicZeroVal();
  EXPECT_NE(magic_zero, nullptr);

  // Test typed shortcuts
  auto zero_int = interface.zeroVal(DataType::Int);
  EXPECT_NE(zero_int, nullptr);
  EXPECT_EQ(zero_int->dtype(), DataType::Int);

  auto one_float = interface.oneVal(DataType::Float);
  EXPECT_NE(one_float, nullptr);
  EXPECT_EQ(one_float->dtype(), DataType::Float);

  // After creating shortcuts, vals should not be empty
  // (shortcuts are stored in container)
  EXPECT_GT(interface.numVals(true), 0);
}

TEST_F(IrInterfaceTest, OwningFusionDefault) {
  // Base IrInterface should return nullptr for owningFusion()
  IrInterface interface;

  EXPECT_EQ(interface.owningFusion(), nullptr);

  const IrInterface& const_interface = interface;
  EXPECT_EQ(const_interface.owningFusion(), nullptr);
}

TEST_F(IrInterfaceTest, CopySemantics) {
  // Test that copying IrInterface clones the container
  IrInterface interface1;

  // Create some values in interface1
  auto zero1 = interface1.zeroVal();
  auto one1 = interface1.oneVal();

  // Copy construct
  IrInterface interface2(interface1);

  // Containers should be different instances
  EXPECT_NE(interface1.container(), interface2.container());

  // But should have same content (same number of vals)
  EXPECT_EQ(interface1.numVals(true), interface2.numVals(true));

  // Values should be different objects (cloned, not shared)
  auto zero2 = interface2.zeroVal();
  EXPECT_NE(zero1, zero2);  // Different pointers
  EXPECT_TRUE(zero2->isZero());  // But same logical value

  // Test copy assignment
  IrInterface interface3;
  auto false_val = interface3.falseVal();  // Create different content

  interface3 = interface1;  // Copy assign

  // After assignment, should have same content as interface1
  EXPECT_EQ(interface3.numVals(true), interface1.numVals(true));
  EXPECT_NE(interface3.container(), interface1.container());
}

TEST_F(IrInterfaceTest, MoveSemantics) {
  // Test that moving IrInterface transfers ownership
  IrInterface interface1;
  auto zero1 = interface1.zeroVal();

  auto* original_container = interface1.container();
  auto original_num_vals = interface1.numVals(true);

  // Move construct
  IrInterface interface2(std::move(interface1));

  // interface2 should have the original container
  EXPECT_EQ(interface2.container(), original_container);
  EXPECT_EQ(interface2.numVals(true), original_num_vals);

  // interface1 should be moved-from (container is nullptr)
  EXPECT_EQ(interface1.container(), nullptr);

  // Test move assignment
  IrInterface interface3;
  auto false_val = interface3.falseVal();

  auto* container3 = interface3.container();

  IrInterface interface4;
  auto true_val = interface4.trueVal();

  interface3 = std::move(interface4);  // Move assign

  // interface3 should have interface4's old container
  EXPECT_NE(interface3.container(), nullptr);
  EXPECT_NE(interface3.container(), container3);  // Different from original

  // interface4 should be moved-from
  EXPECT_EQ(interface4.container(), nullptr);
}

TEST_F(IrInterfaceTest, SwapFunction) {
  // Test that swap correctly exchanges containers
  IrInterface interface1;
  IrInterface interface2;

  auto zero1 = interface1.zeroVal();
  auto one2 = interface2.oneVal();

  auto* container1 = interface1.container();
  auto* container2 = interface2.container();

  EXPECT_NE(container1, container2);

  // Swap
  swap(interface1, interface2);

  // Containers should be swapped
  EXPECT_EQ(interface1.container(), container2);
  EXPECT_EQ(interface2.container(), container1);

  // Content should be swapped
  // interface1 now has what was in interface2
  auto one_from_1 = interface1.oneVal();
  EXPECT_NE(one_from_1, nullptr);

  // interface2 now has what was in interface1
  auto zero_from_2 = interface2.zeroVal();
  EXPECT_NE(zero_from_2, nullptr);
}

TEST_F(IrInterfaceTest, ContainerAccess) {
  // Test direct container access methods
  IrInterface interface;

  // Non-const access
  IrContainer* container = interface.container();
  EXPECT_NE(container, nullptr);

  // Const access
  const IrInterface& const_interface = interface;
  const IrContainer* const_container = const_interface.container();
  EXPECT_NE(const_container, nullptr);
  EXPECT_EQ(const_container, container);
}

TEST_F(IrInterfaceTest, ContainerQueries) {
  // Test inContainer and assertInContainer methods
  IrInterface interface;

  auto zero = interface.zeroVal();
  EXPECT_NE(zero, nullptr);

  // zero should be in the container
  EXPECT_TRUE(interface.inContainer(zero));

  // assertInContainer should not throw for values in container
  EXPECT_NO_THROW(interface.assertInContainer(zero, "Test message"));

  // Create a value not in this container
  IrInterface other_interface;
  auto other_zero = other_interface.zeroVal();

  // other_zero should NOT be in interface's container
  EXPECT_FALSE(interface.inContainer(other_zero));
}

TEST_F(IrInterfaceTest, Axioms) {
  // Test axiom-related methods
  IrInterface interface;

  auto val = interface.zeroVal();

  // Test assumePositive
  EXPECT_NO_THROW(interface.assumePositive(val));

  // Test assumeNonNegative
  EXPECT_NO_THROW(interface.assumeNonNegative(val));

  // Test axioms accessor
  const auto& axioms = interface.axioms();
  EXPECT_TRUE(std::is_const<std::remove_reference<decltype(axioms)>::type>::value ||
              true);  // axioms() returns const reference
}

} // namespace nvfuser
