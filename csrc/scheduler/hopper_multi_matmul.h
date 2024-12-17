// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#pragma once

#include <ATen/cuda/CUDAContext.h>
#include <scheduler/multi_matmul.h>

namespace nvfuser {

// MmaOps in the scheduled tensor. Each one outputs a TensorView* which we call
// an mma_result. Each MmaOp will also have two input TensorViews which we call
// "ab" and "bb" since they are the immediate A and B operands and they contain
// broadcast dimensions. Again there can be multiple abs and multiple bbs in
// one fusion. These TensorViews are loaded from global memory tensors that we
// call "a" and "b" into shared memory tensors called acw_smem and bcw_smem.
// They are loaded from shared memory to register buffers we call "acr" and
// "bcr" ("cr" meaning "cache read" in this context).
//
// Putting this all together we have the following order for a simple matmul
//
//   a -> acw_smem -> acr -> ... -> ab
//                                    \                                      .
//                                      mma_result ->  ... -> dc -> d
//                                    /
//   b -> bcw_smem -> bcr -> ... -> bb
//
// The ... indicate that there might be other tensors involved in a prologue or
// epilogue section at that location.
//
// In this example there are two matmuls both using the same "a" operand:
//
//   b1 -> bcw_smem1 -> bcr1 -> ... -> bb1
//                                        \                                  .
//                                          mma_result1
//                                        /             \                    .
//       a -> acw_smem -> acr -> ... -> ab                ... -> dc -> d
//                                        \             /
//                                          mma_result2
//                                        /
//   b2 -> bcw_smem2 -> bcr2 -> ... -> bb2
//
// Note that there can be more than one output d and each one will have its own
// register cache dc.
//
// Split-K and smem epilogue unswizzling add two additional tensors for each
// mma in the fusion: splitk_sum and smem_epilogue.
//
//   // No split-K, no smem epilogue unswizzling:
//     mma_result ->  ... -> dc -> d
//   // split-K, no smem epilogue unswizzling:
//     mma_result -> splitk_sum -> ... -> dc -> d
//   // smem epilogue unswizzling, no split-K:
//     mma_result -> smem_epilogue -> ... -> dc -> d
//   // split-K and smem epilogue unswizzling:
//     mma_result -> smem_epilogue -> splitk_sum -> ... -> dc -> d
//
// These additional tensors are added to each mma_result in the fusion.
//
// Each of the named tensors above is scheduled differently. We schedule them
// by building AbstractTensors for each tensor category; these are held in
// HopperMultipleMatmulScheduler::schedules_.
// TODO: Inherit from SchedulerEntry
class HopperMultipleMatmulScheduler : public MultipleMatmulScheduler {
 public:
  HopperMultipleMatmulScheduler(Fusion* fusion, const MatmulParams* params)
      : MultipleMatmulScheduler(fusion, params) {
    const auto device_prop = at::cuda::getCurrentDeviceProperties();
    const int cc = device_prop->major * 10 + device_prop->minor;
    NVF_ERROR(
        cc >= 90 && cc < 100, "This matmul scheduler is restricted to Hopper.");
  }

  void run() final;

 private:
  void cacheInputsAndOutputs();

  // Including current tensor naming convention for reference,
  //  this is very temporary and will change over time and
  //  in fact the whole body of this function will
  //  eventually be a set of utility functions for different
  //  sections of matmul(fusion) kernels, with
  //  each having its own build out to do.
  //
  // Current naming convention is based on the following formula:
  //
  //  d = alpha * (a x b) + beta * c
  //
  // and is defined in the following way:
  //
  //  operands assumed in global memory : a, b, c
  //
  //  registers staging global load : ar, br (short for a/b read)
  //
  //  shared mem cache of operands : acw_smem, bcw_smem (short for a/b
  //  cache_write smem)
  //
  //  registers at shared memory load output : acr, bcr (short for a/b cache
  //  read)
  //
  //  register tensor input to the actual mma op: ab, bb (short for a/b
  //  broadcasted)
  //
  //  accumulator register: mma_result
  //   - mma_result is MmaOp output if there is epilogue
  //   - mma_result is dc (short for d cache) if there is no epilogue
  //
  //  result in global memory: d

  // Currently the support is for a, b, c and d as fusion inputs/outputs
  //  aka. no prolog fusion yet.
  void defineOperandCaches();

  void cacheOperandsToSmem(
      const std::vector<TensorView*>& operands,
      std::vector<TensorView*>& smem_operands);

  //! Swizzle the M and N outer dimensions after makeTile has been called.
  //! This updates outer_dim_roles if we introduce a new dimension, which can
  //! happen if tv is missing a merged axis, in which case we skip merging after
  //! the split. This is analogous to forwarding during transform propagation.
  void swizzleBlockTiles(
      TensorView* tv,
      std::vector<MatmulDimRole>& outer_dim_roles);

  //! This calls orig->cacheAfter() and also updates the broadcast graph to
  //! reflect the new IterDomain mappings
  TensorView* cacheAfter(
      TensorView* orig,
      LoadStoreOpType op_type = LoadStoreOpType::Set,
      CacheOp cache_op = CacheOp::AllLevels,
      bool propagate_allocation_domain = false);

  //! Do block tiling for a collection of TensorViews. The tensors should be
  //! unscheduled before this method is called.
  //!   1) Axes will be ordered according to canonicalDimOrdering, and then axes
  //! with the same role will be merged.
  //!   2) After that, we perform splits according to
  //!   params_->tile_sizes.cta_tile, e.g. [M, K] -> [Mo, Ko, Mi, Ki].
  //!   3) Depending on the value of params_->grid_swizzle_factor, if the TV has
  //! both M and N dimensions, we perform a 2D swizzle of the outer dimensions
  //! Mo and No.
  //!   4) Finally, we do a split-K split if the splitk_factor is not 1
  std::vector<std::vector<MatmulDimRole>> blockTileTensors(
      const std::vector<TensorView*>& tvs);

  //! Specifies the CGA dimensions by setting "cluster_dims" as fusion-managed
  //! data
  void setCGADims() const {
    if (params_->cluster_dims != std::tuple<int, int, int>{1, 1, 1}) {
      fusion_->manage("cluster_dims", params_->cluster_dims);
    }
  }

  //! Schedule the loads of all operands from global memory to shared memory.
  //! Starting from the basic tiled schedule, we swizzle the operand memory.
  //! Note that the cache op and LoadStoreOpType are already set during
  //! defineOperandCaches().
  void scheduleOperands();

  //! Check that there is no computation in the prologues, since we do not
  //! support that (yet)
  void inspectPrologues() const;

  void parallelizeBlocks(const std::vector<TensorView*>& tvs) const;

  void scheduleMmaResults();

  void scheduleEpilogue();

  void scheduleSplitKSum();

  void setUpInlining();

  void setUpCircularBuffering();

  // Map TensorView's iterDomain to its ValGroup.
  // Then, find the MatmulDimRole for the ValGroup.
  // Return MatmulDimRole for IterDomain
  MatmulDimRole findMatmulDimRole(IterDomain* id);

  // Schedule a block-tiled TensorView like mma output.
  // Why? WGMMA has a unique output format. TensorViews after the mma-result in
  // registers must respect this format for correctness.
  void transformLikeMmaOutput(TensorView* tv, bool is_mma_result);

 private:
  std::vector<ValGroup> canonical_dim_ordering_;

  std::vector<TensorView*> acw_smems_, bcw_smems_, acrs_, bcrs_, abs_, bbs_,
      splitk_sums_, smem_epilogues_;
};

} // namespace nvfuser
