// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on

#include <device_lower/analysis/tensor_memory.h>
#include <fusion.h>
#include <ir/all_nodes.h>
#include <type.h>

namespace nvfuser {

TensorMemoryInfo collectTMemInfo(Fusion* fusion) {
  TensorMemoryInfo result;
  for (auto tv : fusion->allTvs()) {
    if (tv->getMemoryType() == MemoryType::Tensor) {
      result.allocations.emplace_back();
      result.allocations.back().tensors_to_allocate.push_back(tv);
      result.allocations.back().allocation_address =
          TensorViewBuilder()
              .shape(std::vector<Val*>{})
              .dtype(DataType::UInt32)
              .build();
      result.allocations.back().allocation_address->setMemoryType(
          MemoryType::Shared);
      // TODO: right now we hardcode the number of columns to be 32. This is
      // definitely not correct.
      result.allocations.back().num_columns =
          IrBuilder::create<Val>(32, DataType::UInt32);
    }
  }
  return result;
}

} // namespace nvfuser
