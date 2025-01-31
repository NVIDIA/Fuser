// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on

// TMemTensor is a wrapper around a uint32_t that provides a convenient way to
// manipulate tensor memory addresses. Example usage:
//  TMemTensor T0(0x12345678):
//    -> address (lane=0x1234, col=0x5678):
//  TMemTensor T1 = T0 + {64, 64}:
//    -> address (lane=T0.lane+64, col=T0.col+64)
struct TMemTensor {
  uint32_t raw_address;

 public:
  uint32_t static add(uint32_t base, Array<uint16_t, 2> offset) {
    return base + *reinterpret_cast<const uint32_t*>(&offset);
  }

  TMemTensor(uint32_t raw_address) : raw_address(raw_address) {}

  operator uint32_t() const {
    return raw_address;
  }

  uint32_t operator+(Array<uint16_t, 2> offset) const {
    return add(raw_address, offset);
  }
};

static_assert(
    sizeof(TMemTensor) == sizeof(uint32_t),
    "TMemTensor must be a uint32_t");
