// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#pragma once

#include <exceptions.h>
#include <ir/all_nodes.h>
#include <kernel_ir.h>
#include <visibility.h>

#include <string>
#include <unordered_map>
#include <unordered_set>

namespace nvfuser {

//! Maps TID/BID to its dimension.
class ParallelDimensionMap {
 public:
  void build(Fusion* fusion);

  //! Returns the dimension of a ParallelType. nullptr is returned if
  //! a ParallelType is unused. If a dimension is not a constant, return
  //! blockDim/gridDim instead.
  NVF_API Val* get(ParallelType pt) const;

  //! Returns the raw dimension of a ParallelType. nullptr is returned if
  //! a ParallelType is unused.
  Val* getRaw(ParallelType pt) const;

  //! True if the dimension of a ParallelType is known to be exact
  NVF_API bool isExact(ParallelType pt) const;

  std::string toString() const;

  const std::unordered_map<ParallelType, Val*>& getMap() const {
    return dim_map_;
  }

  //! Get the "compute" parallel dimension on the given ParallelType. In case
  //! of no warp specialization, this is the same as getRaw(pt). If we are doing
  //! warp specialization on pt without register sharing, the result is
  //! getRaw(pt) - padded, because the last of pt is used for loading circular
  //! buffer tensors. If register sharing is also used, difference padded
  //! threads are required for different cta shapes.
  Val* getRawCompute(ParallelType pt) const;

  //! Get the "load" parallel dimension on the given ParallelType. In case
  //! of without warp specialization, this is the same as getRaw(pt). For warp
  //! warp specialization on pt, the result is padded_value. The last part of
  //! pt is used for AsyncWarp warp group.
  Val* getRawAsync(ParallelType pt) const;

  //! The padded val ensures that CTA has 128 threads for the AsyncWarp. This
  //! function returns the padded val for the warp specialized ParallelType.
  int64_t getWarpSpecializationPaddedVal(ParallelType pt) const;

  //! Get the number of threads per each CTA used for computation. When there is
  //! no warp specialization, the result is trivial: it is just the product of
  //! parallel dimensions of TIDx, TIDy and TIDz. If we do have warp
  //! specialization, this returns the number of threads used for computing. For
  //! example, if we have a simple kernel warp specialized on TIDy and all the
  //! TIDx parallelized IterDomains have extent 32, and all the TIDy
  //! parallelized IterDomains have extent 16, and there is no TIDz
  //! parallelization, then we will have:
  //!   blockDim = (x=32, y=17, z=1)
  //! And this function will return (32 * 16) because the extra one for TIDy is
  //! introduced by warp specialization and only used for loading circular
  //! buffer tensors.
  Val* getNumComputeThreadsEachBlock() const;

  //! Assign linear index to each thread of CTA. Assume (TDZ, TDY, TDX) order.
  Val* getLinearThreadIndexAsync() const;

  //! Get if the kernel uses warp specialization
  bool hasWarpSpecialization() const {
    return warp_specialized_parallel_type_.has_value();
  }

  //! Check if ParallelType is WarpSpecialized parallel type.
  bool isWarpSpecialized(ParallelType pt) const {
    return warp_specialized_parallel_type_.value_or(ParallelType::Serial) == pt;
  }

  bool has(ParallelType pt) const {
    return dim_map_.count(pt) > 0;
  }

  // If warp specialized on TIDx and padded value is less than 32 threads, then
  // elect-sync cannot be used.
  bool canUseElectSyncInAsyncWarp() const;

 private:
  //! Get number of threads for ParallelType axis
  //! Not used: 1, Const: n, Dynamic: -1
  int64_t getThreadCountInDim(ParallelType pt);

  //! TIDx may need to be marked as non-exact as it may be padded to a
  //! multiple of the warp size.
  void adjustMappingsForWarpPadding();

  //! If we are doing warp specialization on pt, then we need to increase
  //! the parallel dimension size of pt by one, where the extra one is used
  //! as the async warp. In this case, pt becomes non-exact.
  void adjustMappingsForWarpSpecialization();

 private:
  //! Maps from parallel types to dimensions, which are constant if
  //! a unique value is found.
  std::unordered_map<ParallelType, Val*> dim_map_;
  //! Set of parallel types whose dimensions are identified to be
  //! exactly the same as extents of mapped domains.
  std::unordered_set<ParallelType> exact_types_;

  //! Keep track of warp specialized parallel type and padding value
  std::optional<ParallelType> warp_specialized_parallel_type_;
  std::optional<int64_t> warp_specialized_padding_value_;
};
} // namespace nvfuser
