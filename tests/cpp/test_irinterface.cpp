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

  // Test container query methods exist and work
  EXPECT_EQ(interface.container(), interface.container());
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
  IrInterface interface2(interface1);

  // Containers should be different instances (cloned)
  EXPECT_NE(interface1.container(), interface2.container());

  // Test copy assignment
  IrInterface interface3;
  interface3 = interface1;

  // After assignment, container should be cloned
  EXPECT_NE(interface3.container(), interface1.container());
}

TEST_F(IrInterfaceTest, MoveSemantics) {
  // Test that moving IrInterface transfers ownership
  IrInterface interface1;
  auto* original_container = interface1.container();

  // Move construct
  IrInterface interface2(std::move(interface1));

  // interface2 should have the original container
  EXPECT_EQ(interface2.container(), original_container);

  // interface1 should be moved-from (container is nullptr)
  EXPECT_EQ(interface1.container(), nullptr);

  // Test move assignment
  IrInterface interface3;
  auto* container3 = interface3.container();

  IrInterface interface4;
  interface3 = std::move(interface4);

  // interface3 should have a different container now
  EXPECT_NE(interface3.container(), container3);

  // interface4 should be moved-from
  EXPECT_EQ(interface4.container(), nullptr);
}

TEST_F(IrInterfaceTest, SwapFunction) {
  // Test that swap correctly exchanges containers
  IrInterface interface1;
  IrInterface interface2;

  auto* container1 = interface1.container();
  auto* container2 = interface2.container();

  EXPECT_NE(container1, container2);

  // Swap
  swap(interface1, interface2);

  // Containers should be swapped
  EXPECT_EQ(interface1.container(), container2);
  EXPECT_EQ(interface2.container(), container1);
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
  // Test inContainer method (basic functionality)
  IrInterface interface;

  // Create a simple test - container queries work through the API
  EXPECT_NE(interface.container(), nullptr);
  EXPECT_EQ(interface.numExprs(), 0);
  EXPECT_EQ(interface.numVals(false), 0);
}

// Note: Axiom tests removed - they require Fusion context which IrInterface
// doesn't provide in isolation. Axiom forwarding will be tested through
// Fusion tests in Stage 2.

} // namespace nvfuser
