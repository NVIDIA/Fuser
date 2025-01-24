// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#pragma once

#include <vector>

namespace nvfuser {

class Val;
class TensorView;
class Fusion;

// TODO: mention hackability

// Each entry correspond to a tcgen05.alloc and tcgen05.dealloc.
struct TMemAllocationEntry {
  // Tensor memory is allocated by PTX instruction tcgen05.alloc, which stores
  // the address of the allocated memory in shared memory. In nvFuser, we use
  // a TensorView with MemoryType::Shared to store this address.
  TensorView* allocation_address;
  // The TMem tensors covered by this entry. Currently, collectTMemInfo will
  // assign each TensorView one tcgen05.alloc and tcgen05.dealloc, so this
  // vector will always have size 1. But the representation here is designed to
  // be hackable. For prototyping, just put manual allocation information here.
  std::vector<TensorView*> tensors_to_allocate;
  // The number of columns to allocate. Must be >= 32 and a power of two.
  Val* num_columns;
};

struct TensorMemoryInfo {
  // A vector of TMemAllocationEntry. Each entry corresponds to a tcgen05.alloc
  // and a tcgen05.dealloc.
  std::vector<TMemAllocationEntry> allocations;
};

TensorMemoryInfo collectTMemInfo(Fusion* fusion);

} // namespace nvfuser
