// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#pragma once

#include <ir/all_nodes.h>
#include <kernel_ir.h>

#include <string>
#include <unordered_map>
#include <unordered_set>

namespace nvfuser {

//! Maps TID/BID to its dimension.
class TORCH_CUDA_CU_API ParallelDimensionMap {
 public:
  void build(Fusion* fusion);

  //! Returns the dimension of a ParallelType. nullptr is returned if
  //! a ParallelType is unused. If a dimension is not a constant, return
  //! blockDim/gridDim instead.
  Val* get(ParallelType pt) const;

  //! Returns the raw dimension of a ParallelType. nullptr is returned if
  //! a ParallelType is unused.
  Val* getRaw(ParallelType pt) const;

  //! True if the dimension of a ParallelType is known to be exact
  bool isExact(ParallelType pt) const;

  std::string toString() const;

  const std::unordered_map<ParallelType, Val*>& getMap() const {
    return dim_map_;
  }

 private:
  //! TIDx may need to be marked as non-exact as it may be padded to a
  //! multiple of the warp size.
  void adjustMappingsForWarpPadding();

 private:
  //! Maps from parallel types to dimensions, which are constant if
  //! a unique value is found.
  std::unordered_map<ParallelType, Val*> dim_map_;
  //! Set of parallel types whose dimensions are identified to be
  //! exactly the same as extents of mapped domains.
  std::unordered_set<ParallelType> exact_types_;
};

} // namespace nvfuser
