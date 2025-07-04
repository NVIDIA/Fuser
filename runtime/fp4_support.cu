// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on

// __e2m1 is just a placeholder for the fp4 type.
// Because its size can not be represented as a whole byte, we can not really
// implement a single fp4 number.
struct __e2m1 {
  uint8_t data;
};

static_assert(sizeof(__e2m1) == 1, "__e2m1 must be 1 byte");

struct __e2m1_ptr {
  __e2m1* raw_ptr;
  __e2m1_ptr(void* ptr) : raw_ptr((__e2m1*)ptr) {}
  __e2m1_ptr(const __e2m1_ptr& other) = default;
  __e2m1_ptr& operator=(const __e2m1_ptr& other) = default;
  __e2m1 operator[](int64_t index) const {
    // For performance reason, we do not check the index is even, but we assume
    // it. assert(index % 2 == 0);
    return raw_ptr[index / 2];
  }
  __e2m1& operator[](int64_t index) {
    // For performance reason, we do not check the index is even, but we assume
    // it. assert(index % 2 == 0);
    return raw_ptr[index / 2];
  }
  __e2m1_ptr operator+(int64_t offset) const {
    // For performance reason, we do not check the offset is even, but we assume
    // it. assert(offset % 2 == 0);
    return __e2m1_ptr(raw_ptr + offset / 2);
  }
  __e2m1_ptr operator-(int64_t offset) const {
    // For performance reason, we do not check the offset is even, but we assume
    // it. assert(offset % 2 == 0);
    return __e2m1_ptr(raw_ptr - offset / 2);
  }
  __e2m1 operator*() const {
    return *raw_ptr;
  }
  __e2m1& operator*() {
    return *raw_ptr;
  }
};
