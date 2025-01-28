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
//  TMemTensor T1(0x12345678, 32, 32):
//    -> address (lane=0x1234+32, col=0x5678+32)
//  TMemTensor T2 = T1 + {64, 64}:
//    -> address (lane=T1.lane+64, col=T1.col+64)
struct TMemTensor {
  uint32_t raw_address;

 public:
  uint32_t static add(
      uint32_t base,
      uint16_t lane_offset,
      uint16_t col_offset) {
    uint32_t base_lane = base >> 16;
    uint32_t base_col = base & 0xFFFF;
    uint32_t lane = base_lane + lane_offset;
    uint32_t col = base_col + col_offset;
    return (lane << 16) | col;
  }

  TMemTensor(uint32_t raw_address) : raw_address(raw_address) {}

  TMemTensor(uint32_t base_address, uint16_t lane_offset, uint16_t col_offset)
      : raw_address(add(base_address, lane_offset, col_offset)) {}

  operator uint32_t() const {
    return raw_address;
  }

  uint32_t operator+(Array<uint16_t, 2> offset) const {
    return add(raw_address, offset[0], offset[1]);
  }

  uint16_t lane() const {
    return raw_address >> 16;
  }

  uint16_t col() const {
    return raw_address & 0xFFFF;
  }
};

static_assert(
    sizeof(TMemTensor) == sizeof(uint32_t),
    "TMemTensor must be a uint32_t");
