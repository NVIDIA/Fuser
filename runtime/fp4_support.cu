// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on

// e2m1 is just a placeholder for the fp4 type.
// Because its size can not be represented as a whole byte, we can not really
// implement a single fp4 number.
struct e2m1 {};

static_assert(sizeof(e2m1) == 1, "e2m1 must be 1 byte");

struct e2m1_ptr {
  e2m1* raw_ptr;
  e2m1_ptr(void* ptr) : raw_ptr((e2m1*)ptr) {}
  e2m1_ptr(const e2m1_ptr& other) = default;
  e2m1_ptr& operator=(const e2m1_ptr& other) = default;
  e2m1 operator[](int64_t index) const {
    // For performance reason, we do not check the index is even, but we assume
    // it. assert(index % 2 == 0);
    return raw_ptr[index / 2];
  }
  e2m1& operator[](int64_t index) {
    // For performance reason, we do not check the index is even, but we assume
    // it. assert(index % 2 == 0);
    return raw_ptr[index / 2];
  }
  e2m1_ptr operator+(int64_t offset) const {
    // For performance reason, we do not check the offset is even, but we assume
    // it. assert(offset % 2 == 0);
    return e2m1_ptr(raw_ptr + offset / 2);
  }
  e2m1_ptr operator-(int64_t offset) const {
    // For performance reason, we do not check the offset is even, but we assume
    // it. assert(offset % 2 == 0);
    return e2m1_ptr(raw_ptr - offset / 2);
  }
  e2m1 operator*() const {
    return *raw_ptr;
  }
  e2m1& operator*() {
    return *raw_ptr;
  }
};
