// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on

#include "dynamic_type/dynamic_type.h"

using namespace dynamic_type;

using DoubleInt64Bool = DynamicType<NoContainers, double, int64_t, bool>;

constexpr DoubleInt64Bool a, b;
static_assert(a.isNull());
static_assert(!a.hasValue());
static_assert(b.isNull());
static_assert(!b.hasValue());
static_assert(a == b);
static_assert(b == a);
static_assert(!(a != b));
static_assert(!(b != a));
static_assert(!(a < b));
static_assert(!(b < a));
static_assert(!(a > b));
static_assert(!(b > a));
static_assert(a <= b);
static_assert(b <= a);
static_assert(a >= b);
static_assert(b >= a);
static_assert(a == std::monostate{});
static_assert(std::monostate{} == a);
static_assert(!(a != std::monostate{}));
static_assert(!(std::monostate{} != a));
static_assert(!(a < std::monostate{}));
static_assert(!(std::monostate{} < a));
static_assert(!(a > std::monostate{}));
static_assert(!(std::monostate{} > a));
static_assert(a <= std::monostate{});
static_assert(std::monostate{} <= a);
static_assert(a >= std::monostate{});
static_assert(std::monostate{} >= a);
