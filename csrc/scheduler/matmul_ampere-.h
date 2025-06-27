// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#pragma once

#include <ATen/cuda/CUDAContext.h>
#include <scheduler/matmul.h>

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
// AmpereMinusMultipleMatmulScheduler::schedules_.

namespace schedule_matmul {

class AmpereMinus : public Common {
 public:
  AmpereMinus(Fusion* fusion, const MatmulParams* params)
      : Common(fusion, params) {
    validate();
  }

  void run() final;

 private:
  void validate() const;

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
      std::vector<TensorView*>& smem_operands,
      int64_t vec_size);

  // We add two LoadStore operators to the inputs of our fusions. The first
  // one is for a read from global memory and the second one (below) is for a
  // cache read. As an optimizaton, we avoid adding an operator if there's an
  // existing LoadStoreOp present. Please note that for the second LoadStore
  // we don't propagate the allocation domain, since the scheduler sets the
  // allocation domain in the registers.
  void cacheOperandsToRegisters(
      const std::vector<TensorView*>& tv_smems,
      std::vector<TensorView*>& tv_rs);

  //! Swizzle the M and N outer dimensions after makeTile has been called.
  //! This updates outer_dim_roles if we introduce a new dimension, which can
  //! happen if tv is missing a merged axis, in which case we skip merging after
  //! the split. This is analogous to forwarding during transform propagation.
  void reorderBlockTileTraversal(
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
  //!   3) Depending on the value of params_->grid_traversal_factor, if the TV
  //!   has
  //! both M and N dimensions, we perform a 2D swizzle of the outer dimensions
  //! Mo and No.
  //!   4) Finally, we do a split-K split if the splitk_factor is not 1
  std::vector<std::vector<MatmulDimRole>> blockTileTensors(
      const std::vector<TensorView*>& tvs);

  //! Schedule the loads of all operands from global memory to shared memory.
  //! Starting from the basic tiled schedule, we swizzle the operand memory.
  //! Note that the cache op and LoadStoreOpType are already set during
  //! defineOperandCaches().
  void scheduleOperandSmemStores();

  void scheduleMmaOperands(
      std::vector<TensorView*>& tvs,
      const std::optional<MmaOperand> operand_type);

  // MmaOperand contains only A and B. If tvs are outputs (i.e. not operands),
  // then operand_type should be std::nullopt.
  void scheduleMmaResults();

  void schedulePrologues();

  void scheduleOutputTensor(TensorView* c);

  void scheduleEpilogue();

  //! Propagates transformations from fusion output to fusion tv inputs that are
  //!  producers in the epilogue. Transformations' propagation aims at input tvs
  //!  which are not assigned to core roles, that is, are not MMA inputs.
  void scheduleFusionInputsForEpilogue();

  void scheduleSplitKSum();

  void setUpInlining();

  // NOTE: this should be called after acw_smem, acr, ..., ab, and mma_result
  // transforms have been applied and inlining
  void setUpCircularBuffering();

  void setOperandSmemLoadAndCacheOps(TensorView* operand, int64_t vec_size)
      final;

 private:
  // Tensors used for loading operands from smem to registers, and the
  // broadcasted mma op inputs (abs_, bbs_)
  std::vector<TensorView*> acrs_, bcrs_, abs_, bbs_;
};

} // namespace schedule_matmul
} // namespace nvfuser
