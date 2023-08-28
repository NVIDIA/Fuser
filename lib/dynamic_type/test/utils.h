// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#pragma once

#include <gtest/gtest.h>

#include "dynamic_type.h"

using namespace dynamic_type;

class DynamicTypeTest : public ::testing::Test {};

struct SomeType {};
struct SomeType2 {};

struct NonInstantiable {
  NonInstantiable() = delete;
};

// Adding NonInstantiable as a member type to test that we never instantiate any
// member types when not necessary.
using DoubleInt64Bool =
    DynamicType<NoContainers, double, int64_t, bool, NonInstantiable>;
// Note: because std::vector does not has trivial destructor, we can not
// static_assert to test the following class:
using DoubleInt64BoolVec = DynamicType<
    Containers<std::vector>,
    double,
    int64_t,
    bool,
    NonInstantiable>;
using IntSomeType = DynamicType<NoContainers, int, SomeType, NonInstantiable>;
using BoolSomeType = DynamicType<NoContainers, bool, SomeType, NonInstantiable>;
using SomeTypes =
    DynamicType<NoContainers, SomeType, SomeType, NonInstantiable>;
