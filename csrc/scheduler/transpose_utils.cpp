// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on

#include "scheduler/transpose_utils.h"

#include "scheduler/transpose.h"

namespace nvfuser {
namespace transpose {
namespace utils {

bool hasSmallTransposeDimensions(
    const std::unique_ptr<TransposeParams>& params) {
  return !params->split_before_tiling.empty() ||
      !params->dims_merged_with_1.empty() ||
      !params->dims_merged_with_2.empty();
}

// Note: [Supporting small transpose dimensions]
// We prefer to make tiles of size 32x32 if there are enough elements to achieve
// good occupancy, otherwise, we will use tile size 8x8. In both cases, it is
// possible that the inner dimension of group 1 and/or group 2 are smaller than
// the desired tile size. If this happens, part of the threads of a block will
// be wasted, leading to bad performance. To prevent this from happening, if the
// size of the inner-most dim is smaller than the tile size, we merge other
// dimensions with the inner-most dimension to create larger "virtual inner-most
// dimension". The algorithm that we create these virtual inner-most dimensions
// is as follows:
//
// For example, if we have
//   T0[I0{2}, I1{1024*1024}, I2{2}, I3{2}, I4{2}, I5{2}, I6{2}] input
//   T1 = transpose(T0, 4, 6)
// We first try to merge each inner-most dim with the dimensions on its left:
//   T0[I0{2}, I1*I2*I3*I4{1024*1024*8}, I5*I6{4}]
// If there is/are still unsatisfied innermost dim(s) after this step (I5*I6 in
// this example), we find other dims that is not merged yet to satisfy it/them:
//   T0[I0*I5*I6{8}, I1*I2*I3*I4{1024*1024*8}]
// If after merging all the dims, there is still one of them not satisfied, this
// usually means there is one large dim that is consumed by the satisfied one.
// We will split that dim and large dim and and use the splitted ones to satisfy
// both of them:
//   T0[I0*I1o*I5*I6{1024*1024/4*8}, I1i*I2*I3*I4{32}]
void maybeBuildVirtualInnerDims(
    TransposeParams* tparams,
    int64_t device_multiprocessor_count,
    int64_t n_elems,
    const std::vector<int64_t>& shape_in_ref1,
    int64_t inner_most1,
    int64_t inner_most2) {
  int64_t merged_size1 = shape_in_ref1[inner_most1];
  int64_t merged_size2 = shape_in_ref1[inner_most2];

  int64_t actual_tile_size1 =
      std::min<int64_t>(merged_size1, (int64_t)tparams->tile_size1);
  int64_t actual_tile_size2 =
      std::min<int64_t>(merged_size2, (int64_t)tparams->tile_size2);
  int64_t wave_elements =
      device_multiprocessor_count * actual_tile_size1 * actual_tile_size2;

  if (wave_elements >= n_elems) {
    // if one full wave can handle all elements, don't create virtual inner dims
    return;
  }

  // merge inner_most1 and inner_most2 left until we are done or we can no
  // longer do so
  int64_t dim = inner_most1 - 1;
  while (dim >= 0 && dim != inner_most2 &&
         merged_size1 < (int64_t)tparams->tile_size1) {
    tparams->dims_merged_with_1.push_back(dim);
    merged_size1 *= shape_in_ref1[dim];
    dim--;
  }
  dim = inner_most2 - 1;
  while (dim >= 0 && dim != inner_most1 &&
         merged_size2 < (int64_t)tparams->tile_size2) {
    tparams->dims_merged_with_2.push_back(dim);
    merged_size2 *= shape_in_ref1[dim];
    dim--;
  }
  // If any of them are unsatisfied, then find other dims to merge
  std::unordered_set<int64_t> unavailable_dims;
  unavailable_dims.reserve(
      2 + tparams->dims_merged_with_1.size() +
      tparams->dims_merged_with_2.size());
  unavailable_dims.insert(inner_most1);
  unavailable_dims.insert(inner_most2);
  for (auto i : tparams->dims_merged_with_1) {
    unavailable_dims.insert((int64_t)i);
  }
  for (auto i : tparams->dims_merged_with_2) {
    unavailable_dims.insert((int64_t)i);
  }
  dim = (int64_t)shape_in_ref1.size() - 1;
  while (dim >= 0 && merged_size1 < (int64_t)tparams->tile_size1) {
    if (unavailable_dims.count(dim) == 0) {
      tparams->dims_merged_with_1.push_back(dim);
      merged_size1 *= shape_in_ref1[dim];
      unavailable_dims.insert(dim);
    }
    dim--;
  }
  dim = (int64_t)shape_in_ref1.size() - 1;
  while (dim >= 0 && merged_size2 < (int64_t)tparams->tile_size2) {
    if (unavailable_dims.count(dim) == 0) {
      tparams->dims_merged_with_2.push_back(dim);
      merged_size2 *= shape_in_ref1[dim];
      unavailable_dims.insert(dim);
    }
    dim--;
  }
  // If both are satisfied, then we are done. If neither are satisfied, then it
  // is impossible to satisfy both of them, also done.
  if ((merged_size1 < (int64_t)tparams->tile_size1) ==
      (merged_size2 < (int64_t)tparams->tile_size2)) {
    return; // no need to split
  }
  // If one of them are not satisfied, there might be two cases:
  // 1. The satisfied one just merged in a large dim. If this is the case, We
  //    split this large dim, so that now we have two available dims to satisfy
  //    both virtual innermost dim.
  // 2. The satisfied one did not merge in anything. For example,
  //    T0[I0{1024*1024}, I1{2}]
  //    If this is the case, this means that we need to split the large
  //    inner-most dimension to satisfy the small innermost dimension
  int64_t large_dim = -1;
  int64_t split_factor = -1;
  bool split_inner_most = false;
  if (merged_size1 < (int64_t)tparams->tile_size1) {
    if (tparams->dims_merged_with_2.empty()) {
#if SUPPORT_SPLITTING_INNERMOST_DIM
      // https://github.com/csarofeen/pytorch/issues/1964
      // case 2
      split_inner_most = true;
      large_dim = inner_most2;
      split_factor = tparams->tile_size2;
#else
      // disabled due to indexing error
      return;
#endif
    } else {
      // case 1
      split_inner_most = false;
      large_dim = (int64_t)tparams->dims_merged_with_2.back();
      auto prev_merged_size2 = merged_size2 / shape_in_ref1[large_dim];
      split_factor = ceilDiv((int64_t)tparams->tile_size2, prev_merged_size2);
    }
  } else {
    if (tparams->dims_merged_with_1.empty()) {
#if SUPPORT_SPLITTING_INNERMOST_DIM
      // https://github.com/csarofeen/pytorch/issues/1964
      // case 2
      split_inner_most = true;
      large_dim = inner_most1;
      split_factor = tparams->tile_size1;
#else
      // disabled due to indexing error
      return;
#endif
    } else {
      // case 1
      split_inner_most = false;
      large_dim = (int64_t)tparams->dims_merged_with_1.back();
      auto prev_merged_size1 = merged_size1 / shape_in_ref1[large_dim];
      split_factor = ceilDiv((int64_t)tparams->tile_size1, prev_merged_size1);
    }
  }
  tparams->split_before_tiling.emplace_back(large_dim, split_factor);
  // adjust all dims to after-split
  for (auto& i : tparams->dims_merged_with_1) {
    if ((int64_t)i > large_dim) {
      i++;
    }
  }
  for (auto& i : tparams->dims_merged_with_2) {
    if ((int64_t)i > large_dim) {
      i++;
    }
  }
  // Give the split-out dim to the unsatisfied one, so that both are satisfied.
  if (merged_size1 < (int64_t)tparams->tile_size1) {
    if (!split_inner_most) {
      tparams->dims_merged_with_2.pop_back();
      tparams->dims_merged_with_2.push_back(large_dim + 1);
    }
    tparams->dims_merged_with_1.push_back(large_dim);
  } else {
    if (!split_inner_most) {
      tparams->dims_merged_with_1.pop_back();
      tparams->dims_merged_with_1.push_back(large_dim + 1);
    }
    tparams->dims_merged_with_2.push_back(large_dim);
  }
}

} // namespace utils
} // namespace transpose
} // namespace nvfuser
