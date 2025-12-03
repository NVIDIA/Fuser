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
#include <utils.h>
#include <visibility.h>

#include <string>
#include <unordered_map>
#include <unordered_set>

namespace nvfuser {

//! Maps TID/BID to its dimension.
class ParallelDimensionMap {
 public:
  ParallelDimensionMap() = default;

  ParallelDimensionMap(Fusion* fusion);

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

  Val* getEvalExtent(ParallelDim* pdim) const {
    return mapOrDefault(dim_eval_extent_map_, pdim, /*default=*/(Val*)nullptr);
  }

  Val* getCodegenExtent(ParallelDim* pdim) const {
    return mapOrDefault(
        dim_codegen_extent_map_, pdim, /*default=*/(Val*)nullptr);
  }

  Val* getIndex(ParallelDim* pdim) const {
    return mapOrDefault(dim_index_map_, pdim, /*default=*/(Val*)nullptr);
  }

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

  //! Infers the extents of ParallelDims based on bound dims and algebraic
  //! operations and creates scalar Vals that can be evaluated on the host. For
  //! instance, if TIDx is split into two new dims and they are bound to sizes 8
  //! and 32, then TIDx is inferred to be of size 8*32=256.
  void inferEvalExtents(Fusion* fusion);

  //! Infers the extents of ParallelDims just like inferEvalExtents, but
  //! generates scalar Vals that are appropriate for evaluation in device code
  //! instead of on the host. FOr instance, instead of deriving a complicated
  //! expression for the extent of a dimension based on input tensor's shape, we
  //! can often use special values like blockIdx.x.
  void inferCodegenExtents(Fusion* fusion);

  //! Infers indices to use for each parallel Dim. For example, TIDx dims will
  //! have an index which is a NamedScalar with value "threadIdx.x". We traverse
  //! the ParallelDim graph to assign indices for all dims. For instance, if
  //! TIDx is split into two dims and the inner dim has extent 32, then we
  //! assign threadIdx.x / 32 and threadIdx.x % 32 for the indices of those two
  //! dims.
  //!
  //! Note that non-thread parallel dimensions are excluded from this analysis.
  void inferIndices(Fusion* fusion);

 private:
  //! Maps from parallel types to dimensions, which are constant if
  //! a unique value is found.
  std::unordered_map<ParallelType, Val*> dim_map_;
  //! Set of parallel types whose dimensions are identified to be
  //! exactly the same as extents of mapped domains.
  std::unordered_set<ParallelType> exact_types_;

  //! Holds the extent of each parallel dimension in a form that can be easily
  //! evaluated by ExpressionEvaluator. NamedScalars like blockDim.x will be
  //! avoided in favor of expressions like ceilDiv(i0, 128) where i0 is the
  //! extent of an input TensorView.
  std::unordered_map<ParallelDim*, Val*> dim_eval_extent_map_;
  //! Holds the extent of each parallel dimension in a form suitable for placing
  //! in a generated CUDA kernel.
  //! For example these expressions will contain things like blockDim.x
  std::unordered_map<ParallelDim*, Val*> dim_codegen_extent_map_;

  //! Holds the index of a parallel dimension for use in a generated CUDA
  //! kernel. Dims corresponding to ParallelTypes will have indices like
  //! threadIdx.x, but these will be more complicated if there are other
  //! expressions in the ParallelDim graph. For example a ParallelDimSplit can
  //! result in div and mod expressions.
  std::unordered_map<ParallelDim*, Val*> dim_index_map_;

  //! Keep track of warp specialized parallel type and padding value
  std::optional<ParallelType> warp_specialized_parallel_type_;
  std::optional<int64_t> warp_specialized_padding_value_;
};
} // namespace nvfuser
