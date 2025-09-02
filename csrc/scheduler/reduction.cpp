// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#include <ATen/cuda/CUDAContext.h>
#include <debug.h>
#include <instrumentation.h>
#include <multidevice/utils.h>
#include <scheduler/debug_utils.h>
#include <scheduler/mark_aliases.h>
#include <scheduler/reduction.h>
#include <scheduler/reduction_utils.h>
#include <scheduler/registry_utils.h>
#include <scheduler/runtime_info.h>
#include <scheduler/tools/domain_map.h>
#include <scheduler/tools/inlining.h>
#include <scheduler/utils.h>
#include <scheduler/vectorize_helper.h>

namespace nvfuser {

namespace {

// Rounds x up to a power of 2 or a multiple of multiple
int64_t roundUpPow2OrMultipleOf(const int64_t x, const int64_t multiple) {
  auto round_up_pow2 = scheduler_utils::lastPow2(x);
  if (round_up_pow2 < x) {
    round_up_pow2 *= 2;
  }
  auto round_up_multiple =
      x % multiple == 0 ? x : x + (multiple - x % multiple);
  return std::max(std::min(round_up_multiple, round_up_pow2), (int64_t)1);
}

// Rounds x down to a power of 2 or a multiple of multiple, whichever is bigger
int64_t roundDownPow2OrMultipleOf(const int64_t x, const int64_t multiple) {
  auto round_down_pow2 = scheduler_utils::lastPow2(x);

  auto round_down_multiple = x % multiple == 0 ? x : x - x % multiple;
  return std::max(std::max(round_down_multiple, round_down_pow2), (int64_t)1);
}

int64_t clamp(const int64_t val, const int64_t min_val, const int64_t max_val) {
  return std::min(std::max(val, min_val), max_val);
}

// Reduce x, y, z until it's product is less than max value, reduce round robin
// starting with x
void reduceProductTo(int64_t& z, int64_t& y, int64_t& x, const int64_t max) {
  NVF_ERROR(max > 1);
  if (z * y * x > max) {
    z = scheduler_utils::safeDiv(z, 2);
  }
  if (z * y * x > max) {
    y = scheduler_utils::safeDiv(y, 2);
  }
  if (z * y * x > max) {
    x = scheduler_utils::safeDiv(x, 2);
  }
  if (z * y * x > max) {
    reduceProductTo(x, y, z, max);
  }
}

// The returned value is the product of vectorization_factor and
// reduction_unroll_factor for 2d inner reduction heuristics. The estimation is
// based on properties of the fusion and hardware memory bandwidth.
int64_t getVectUnroll(
    const int64_t max_dtype_size_bit_for_vectorization,
    const int64_t max_vectorize_factor,
    const int64_t n_tensor_inputs,
    const int64_t target_threads_per_sm,
    const bool has_mufu_computation) {
  // empirical value, derived from A100 & H100
  int64_t vect_factor = ceilDiv(
      // Available unrolling based on size of data type
      (int64_t)128 / max_dtype_size_bit_for_vectorization,
      // Reduce unrolling if we have many inputs, start reduction at 4 inputs
      scheduler_utils::lastPow2(std::max(n_tensor_inputs >> 2, (int64_t)1)));

  // If has computation uses mufu units, thread local computation is already
  // expensive, don't need further unroll. This is opposite to pointwise
  // scheduler where extra unroll is beneficial if we have expensive ops. Why?
  // Probably because the reduction after pointwise ops is already expensive
  // enough to hide the memory access latency of other blocks.
  if (has_mufu_computation) {
    return vect_factor;
  }

  int64_t required_bits_in_flight = scheduler_utils::getRequiredBitsInFlight();
  int64_t required_bits_per_thread =
      ceilDiv(required_bits_in_flight, target_threads_per_sm);
  int64_t bits_per_element =
      max_dtype_size_bit_for_vectorization * n_tensor_inputs;
  int64_t unroll_vect = ceilDiv(required_bits_per_thread, bits_per_element);

  // prioritize vectorization over unrolling
  vect_factor = std::min(vect_factor, scheduler_utils::lastPow2(unroll_vect));

  // When fully vectorized, unroll by at least 2 to provide some
  // instruction level parallelism. This is for A100-40G whose bandwidth is much
  // lower and won't need unroll if only based on bytes in flight.
  int64_t unroll_factor = 1;
  if (vect_factor == max_vectorize_factor) {
    unroll_factor = std::max(2L, ceilDiv(unroll_vect, vect_factor));
  }
  return unroll_factor * vect_factor;
}

int64_t getL1L2WarpSize(
    const int64_t total_reduction_numel,
    const int64_t total_iteration_numel,
    const int64_t n_tensor_inputs,
    const int64_t max_dtype_size_bit_for_vectorization) {
  const int64_t n_elems = total_reduction_numel * total_iteration_numel;
  // Conservative value, could be set to larger based on arch if necessary.
  constexpr int64_t l1_cache_bit = (int64_t)32 * 1024 * 8;
  // Could change per generation, but for l1 we want to consider active threads,
  // not resident
  constexpr int64_t active_threads = 1024;

  // if data fits in l2 and we need more parallelization in the reduction dim,
  // we can use a smaller warp size. While thread local data fits in l1, and
  // reduction dim is really small, we can use <32 threads per warp.
  const bool fits_in_l2 =
      n_elems * max_dtype_size_bit_for_vectorization * n_tensor_inputs <
      at::cuda::getCurrentDeviceProperties()->l2CacheSize * 8;

  // If it fits in l2, we just want to make sure each warp uses 256 bits. Set
  // minimum warp as 16 threads instead of 32 as if we have a small reduction
  // dim going a bit smaller than 32 usually helps.
  const int64_t warp_size_based_on_l2 =
      fits_in_l2 ? (int64_t)256 / max_dtype_size_bit_for_vectorization : 16;

  // Check how many elements it would take per thread to start thrashing l1
  // set that to minimum number we want to reduce per thread.
  const int64_t warp_size_based_on_l1 = std::min(
      ceilDiv(
          total_reduction_numel,
          std::max(
              l1_cache_bit /
                  (n_tensor_inputs * max_dtype_size_bit_for_vectorization *
                   active_threads),
              (int64_t)1)),
      (int64_t)16);
  return std::min(warp_size_based_on_l1, warp_size_based_on_l2);
}

// Parallelization strategy:
// [] indicates optional
// Reduction dim: Serial, [BIDx or unroll], TIDx, Vect
// Iteration dim: Serial, BIDy, [TIDy], [unroll]
std::unique_ptr<ReductionParams> inner2dReductionHeuristic(
    const int64_t total_reduction_numel,
    const int64_t total_iteration_numel,
    const int64_t n_tensor_inputs,
    const int64_t max_dtype_size_bit_for_vectorization,
    const int64_t max_vectorize_factor,
    const bool has_mufu_computation) {
  // Get device properties
  auto dev_prop = at::cuda::getCurrentDeviceProperties();
  const int64_t threads_per_warp = dev_prop->warpSize;
  const int64_t max_threads_per_block = dev_prop->maxThreadsPerBlock;
  const int64_t max_threads_per_sm = dev_prop->maxThreadsPerMultiProcessor;
  const int64_t sm_count = dev_prop->multiProcessorCount;
  const int64_t target_threads_per_sm = max_threads_per_sm / 2;

  const int64_t min_warp_size = getL1L2WarpSize(
      total_reduction_numel,
      total_iteration_numel,
      n_tensor_inputs,
      max_dtype_size_bit_for_vectorization);

  // Set target_vect_unroll
  auto target_vect_unroll = getVectUnroll(
      max_dtype_size_bit_for_vectorization,
      max_vectorize_factor,
      n_tensor_inputs,
      target_threads_per_sm,
      has_mufu_computation);

  // redu = [i-remainder, i-Unroll, TIDx, Vect]
  int64_t inner_unroll = 1, bdimx = 1, vect_factor = 1;
  auto getInnerRemainder = [&]() {
    return ceilDiv(
        ceilDiv(total_reduction_numel / vect_factor, bdimx), inner_unroll);
  };

  // Max vectorization factor
  vect_factor = std::min(
      scheduler_utils::lastPow2(target_vect_unroll),
      (int64_t)max_vectorize_factor);
  const int64_t four_warps = 4 * threads_per_warp;
  int64_t target_inner_unroll = target_vect_unroll / vect_factor;
  int64_t after_vect = total_reduction_numel / vect_factor;

  // Empirical CTA size, both values allow thread uses 255 registers, so we
  // don't need to set maxrregcount which may affect occupancy since it is
  // derived by assuming one block per sm. This is due to the fact that when
  // maxrregcount is set, the compiler are more aggressive in register
  // allocation. When has expensive ops, use a smaller CTA size for higher
  // opportunity to reach higher occupancy.
  int64_t target_threads_per_block = has_mufu_computation ? 128 : 256;

  // When iteration dim is small, the number of blocks is small since each block
  // reduces one row (if iteration dim is extremely small, may do grid reduction
  // where multiple blocks are used for one rwo). Then, we need a large CTA size
  // to saturate SMs. But don't go over max_threads_per_block / 2 to reduce
  // communication cost and allow more than 1 block per SM.
  int64_t max_blocks = total_iteration_numel;
  int64_t min_threads_per_block = scheduler_utils::roundUpPow2(
      ceilDiv(target_threads_per_sm * sm_count, max_blocks));
  min_threads_per_block =
      std::min(min_threads_per_block, max_threads_per_block / 2);
  target_threads_per_block =
      std::max(target_threads_per_block, min_threads_per_block);

  // Put inner remainder to bdimx, but don't go over 4 warps
  // to leave some room for unrolling.
  bdimx = std::max(getInnerRemainder(), min_warp_size);
  bdimx = std::min(bdimx, four_warps);

  // Put inner remainder to unroll, but don't go over target_inner_unroll
  inner_unroll = std::min(getInnerRemainder(), target_inner_unroll);

  // Put inner remainder to bdimx, but don't go over target_threads_per_block
  bdimx = std::min(ceilDiv(after_vect, inner_unroll), target_threads_per_block);

  // if unroll can't cover the whole reduction, adjust bdimx and unroll to
  // remove quantization effect. e.g. at 20736, vect = 8, bdimx = 512, unroll =
  // 5, then iteration = 1.01, the 2nd iteration only has 1% effective threads.
  // Check all possible bdimx and unroll, choose the one with smallest remainder
  if (getInnerRemainder() > 1) {
    int64_t after_vect_bdimx = ceilDiv(after_vect, bdimx);
    int64_t current_remainder = after_vect_bdimx % inner_unroll;
    for (auto tmp_bdimx = four_warps; tmp_bdimx <= target_threads_per_block;
         tmp_bdimx += min_warp_size) {
      after_vect_bdimx = ceilDiv(after_vect, tmp_bdimx);
      for (auto factor = target_inner_unroll; factor > 1; factor--) {
        int64_t remainder = after_vect_bdimx % factor;
        if (remainder == 0) {
          inner_unroll = factor;
          bdimx = tmp_bdimx;
          break;
        }
        if (remainder < current_remainder) {
          current_remainder = remainder;
          inner_unroll = factor;
          bdimx = tmp_bdimx;
        }
      }
    }
  }

  // Set iteration dims, iter = [BIDy, o-Unroll, TIDy]
  // outer unroll is not used since test shows lower performance compared with
  // inner unroll because the lack of a iteration domain grouped warp
  // reduction. So each block needs to call warp reduction [o-Unroll] times.
  int64_t bdimy = 1, outer_unroll = 1, godim = 1;

  // fill bdimy, but don't go over target_threads_per_block
  bdimy = std::min(total_iteration_numel, target_threads_per_block / bdimx);

  // Put what is left to godim
  godim = ceilDiv(ceilDiv(total_iteration_numel, bdimy), outer_unroll);

  // When iteration dim is small, may have unused SMs, to increase SM usage
  // needs to shift from block reduction to grid reduction.
  int64_t grdim = 1;
  while (godim * grdim * 2 <= sm_count && getInnerRemainder() / grdim >= 2) {
    grdim *= 2;
  }

  // For grid reduction, try to do some cleanup of ragged waves on device
  if (grdim > 1 && // If we have less than 8 waves of blocks
      grdim * godim < sm_count * 8 &&
      // And we don't have an even divisible number of blocks
      (grdim * godim) % sm_count != 0 &&
      // And we have more than one wave
      grdim * godim > sm_count) {
    // round waves down
    auto waves = std::max((godim * grdim) / sm_count, (int64_t)1);
    auto new_grdim = std::max((waves * sm_count) / godim, (int64_t)1);
    if (
        // If difference is less than 25% of the original grdim
        (new_grdim - grdim) * 4 < grdim &&
        // and difference is less than 25% of the original number of blocks
        ((new_grdim * godim) - (grdim * godim)) * 4 < grdim * godim) {
      grdim = new_grdim;
    }
  }

  if (grdim > 1) {
    // Grid reductions do not support unrolling iteration dimension, revert if
    // set. Recalculate godim.
    if (outer_unroll) {
      outer_unroll = 1;
      godim = ceilDiv(total_iteration_numel, bdimy * outer_unroll);
    }
    // This could mess up parallelization which could be redone, but that would
    // require iterating over this entire function.
  }

  auto rparams = std::make_unique<ReductionParams>();
  rparams->schedule_3D = false;
  rparams->fastest_dim = true;
  rparams->cross_block_inner_reduction = true;
  rparams->block_dim_inner_reduction = ParallelType::TIDx;
  rparams->cross_grid_inner_reduction = grdim > 1;
  rparams->multiple_reds_per_blk = bdimy > 1;
  bool pad_bdimx = bdimx > 16 && bdimx * bdimy < max_threads_per_block;
  rparams->pad_inner_reduction_to_warp = pad_bdimx;

  if (rparams->pad_inner_reduction_to_warp) {
    // Adjust bdimx based on padding
    auto min_warp_size =
        (int64_t)at::cuda::getCurrentDeviceProperties()->warpSize;
    bdimx = bdimx % min_warp_size == 0
        ? bdimx
        : bdimx + min_warp_size - bdimx % min_warp_size;
  }
  rparams->unroll_factor_inner_reduction = vect_factor;
  rparams->unroll_factor_top_of_vectorization = inner_unroll;
  rparams->vectorize_inner_reduction = vect_factor > 1;
  if (rparams->multiple_reds_per_blk) {
    rparams->block_dim_iter_dom = ParallelType::TIDy;
  }

  rparams->unroll_factor_iter_dom = outer_unroll;

  int64_t gdimx = LaunchParams::UNINITIALIZED_VAL;
  int64_t gdimy = LaunchParams::UNINITIALIZED_VAL;

  // If we have a cross grid case we want to have gdimy assigned to godim and
  // gdimx assigned to grdim. Otherwise it's helpful to pull godim into gdimx in
  // case it's larger than gdimy can hold, as not doing so can thrash the cache.

  if (rparams->cross_grid_inner_reduction) {
    rparams->grid_dim_inner_reduction = ParallelType::BIDx;
    rparams->split_grid_dim_inner_reduction = true;
    gdimx = std::min(grdim, scheduler_utils::x_grid_limit);

    rparams->grid_dim_iter_dom = ParallelType::BIDy;
    if (godim > scheduler_utils::y_grid_limit) {
      NVF_ERROR(
          false,
          "Shouldn't do grid reduction when the iteration dimension is larger "
          "than y_grid_limit. iteration dim: ",
          total_iteration_numel);
      rparams->split_grid_dim_iter_dom_outer = true;
      gdimy = scheduler_utils::y_grid_limit;
    }

  } else {
    rparams->grid_dim_iter_dom = ParallelType::BIDx;
    if (godim > scheduler_utils::x_grid_limit) {
      rparams->split_grid_dim_iter_dom_outer = true;
      gdimx = scheduler_utils::x_grid_limit;
    }
  }

  rparams->lparams = LaunchParams(
      gdimx,
      gdimy,
      LaunchParams::UNINITIALIZED_VAL,
      bdimx,
      bdimy > 1 ? bdimy : LaunchParams::UNINITIALIZED_VAL,
      LaunchParams::UNINITIALIZED_VAL);

  if (isDebugDumpEnabled(DebugDumpOption::SchedulerDebug)) {
    debug() << "\n===== Inner 2D Reduction Stats ========\n"
            << "total_reduction_numel: " << total_reduction_numel << "\n"
            << "total_iteration_numel: " << total_iteration_numel << "\n"
            << "target_vect_unroll: " << target_vect_unroll << "\n"
            << "vectorize_factor: " << vect_factor << "\n"
            << "inner_unroll: " << inner_unroll << "\n"
            << "outer_unroll: " << outer_unroll << "\n"
            << "inner iteration: " << getInnerRemainder() << "\n"
            << "n_tensor_inputs: " << n_tensor_inputs << "\n"
            << "max_dtype_size_bit_for_vectorization: "
            << max_dtype_size_bit_for_vectorization << "\n"
            << "block(" << bdimx << ", " << bdimy << ", " << 1 << ")"
            << std::endl;
    debug() << rparams->toString() << std::endl;
  }
  return rparams;
}

std::unique_ptr<ReductionParams> inner3dReductionHeuristic(
    const int64_t total_reduction_numel,
    const int64_t total_iteration_numel,
    const int64_t inner_most_dimension_numel,
    const int64_t n_tensor_inputs,
    const int64_t max_dtype_size_bit_for_vectorization,
    const size_t vectorize_factor,
    const bool has_mufu_computation) {
  // Set some targets for parallelization

  const int64_t n_elems = total_reduction_numel * total_iteration_numel;

  // WARNING: At some point we may want to generate heuristics for another
  // device that is not the current device.
  const int64_t max_threads_per_sm =
      (int64_t)at::cuda::getCurrentDeviceProperties()
          ->maxThreadsPerMultiProcessor;

  const int64_t device_multiprocessor_count =
      (int64_t)at::cuda::getCurrentDeviceProperties()->multiProcessorCount;

  auto const max_unroll = ceilDiv(
      // Available unrolling based on size of data type
      (int64_t)128 / (int64_t)max_dtype_size_bit_for_vectorization,
      // Reduce unrolling if we have many inputs, start reduction at 4 inputs
      scheduler_utils::lastPow2(
          std::max((int64_t)n_tensor_inputs >> 2, (int64_t)1)));

  // Take the smaller
  const int64_t min_warp_size = getL1L2WarpSize(
      total_reduction_numel,
      total_iteration_numel,
      n_tensor_inputs,
      max_dtype_size_bit_for_vectorization);

  // Initialization
  int64_t target_blocks = 1;
  int64_t target_unroll = 1;
  int64_t target_iterations = 1;

  // Try to set a minmum amount of work for each thread, as cross thread
  // communication is slow so it shouldn't be done for every element in the
  // reduction.
  int64_t min_target_iterations = std::max(
      (int64_t)256 / (int64_t)max_dtype_size_bit_for_vectorization, (int64_t)1);

  // Start trying to break parallelization up across threads,
  // unrolling/iterations, and blocks.

  // target_threads_in_block is the cap on a thread block, the minimum is based
  // on min_warp_size
  int64_t target_threads_in_block = std::max(
      min_warp_size, ceilDiv(total_reduction_numel, min_target_iterations));

  // If we have one warp per block, check if that's enough to saturate the SMs
  target_blocks = ceilDiv(n_elems, min_warp_size);

  // If we have more than a wave of blocks, put parallelism into unrolling and
  // target iterations
  if (target_blocks > device_multiprocessor_count) {
    auto available_unroll = std::max(
        n_elems / (min_warp_size * device_multiprocessor_count), (int64_t)1);

    // Spread across unrolling and iterations, want a balance of the two so flip
    // back and forth to alternate adding to them.
    bool flip = true;

    while (available_unroll > 1 &&
           (target_unroll < max_unroll ||
            // Prefer unrolling
            target_iterations < max_unroll)) {
      if (target_unroll * 2 <= max_unroll && flip) {
        target_unroll *= 2;
      }

      if (target_iterations * 2 <= max_unroll && !flip) {
        target_iterations *= 2;
      }

      available_unroll = std::max(
          n_elems /
              (min_warp_size * device_multiprocessor_count * target_unroll *
               target_iterations),
          (int64_t)1);

      flip = !flip;
    }

    // Recompute target blocks
    target_blocks =
        ceilDiv(n_elems, min_warp_size * target_unroll * target_iterations);
  }

  // Cap target blocks to 4 waves
  target_blocks = std::min(target_blocks, device_multiprocessor_count * 4);

  if (target_blocks * target_unroll * target_iterations < n_elems) {
    // targetting 4 waves, so try to use a quarter of available threads
    target_threads_in_block = std::min(
        ceilDiv(n_elems, target_blocks * target_unroll),
        ceilDiv(max_threads_per_sm, (int64_t)4));
  }

  // Round up to nearest warp.
  if (target_threads_in_block % min_warp_size != 0) {
    target_threads_in_block +=
        min_warp_size - target_threads_in_block % min_warp_size;
  }

  // To get to target threads:
  // Prioritize
  // (1) x dim in reduction
  // (2) unrolling in reduction
  // (3) y in output
  // To get target blocks:
  // Prioritize
  // (1) x dim in multiple outputs
  // (2) y dim in multiple reductions

  // Cross grid inner reduction, number of blocks to cross-grid on
  int64_t gridim = 1;
  // Cross grid outer reduction, number of blocks to cross-grid on
  int64_t grodim = 1;
  // Blocks for outputs
  int64_t godim = 1;

  // Threads for reduction
  int64_t bdimx = 1;
  // Threads for outputs
  int64_t bdimy = 1;
  // Threads for outer reduction dimension
  int64_t bdimz = 1;

  // Unroll amount
  int64_t inner_reduction_unroll_factor = 1;
  int64_t outer_reduction_unroll_factor = 1;
  int64_t iter_unroll_factor = 1;

  inner_reduction_unroll_factor =
      vectorize_factor > 1 ? (int64_t)vectorize_factor : 1;

  // Grab what we can out of reduction domain, but don't go over a warp size yet
  bdimx = std::min(
      std::max(
          ceilDiv(inner_most_dimension_numel, inner_reduction_unroll_factor),
          (int64_t)min_warp_size),
      target_threads_in_block);

  // If we're not just barely covering the dimension, round to a more friendly
  // number
  if (bdimx * inner_reduction_unroll_factor != inner_most_dimension_numel) {
    // Round bdimx down to multiple of warp size or power 2
    if (bdimx < min_warp_size) {
      bdimx = scheduler_utils::lastPow2(bdimx);
    } else {
      bdimx = bdimx - bdimx % min_warp_size;
    }
  }

  // Put everything else in bdimy for now
  bdimy = std::max(min_warp_size / bdimx, (int64_t)1);

  // If 3D fill the rest of the threads into bdimz
  bdimz = std::min(
      std::min(
          std::max(target_threads_in_block / (bdimx * bdimy), (int64_t)1),
          ceilDiv(total_reduction_numel, inner_most_dimension_numel)),
      scheduler_utils::z_block_limit);

  // If 3D doesn't fill out the threads, adjust to add to bdimy
  bdimy = std::max(target_threads_in_block / (bdimx * bdimz), (int64_t)1);

  // If we don't have a full warp and have an unroll factor, move unroll into
  // bdimx
  if (bdimx * bdimy * bdimz < min_warp_size &&
      inner_reduction_unroll_factor > 1) {
    bdimx = std::min(
        std::max(inner_most_dimension_numel, min_warp_size),
        target_threads_in_block);

    inner_reduction_unroll_factor =
        std::min(ceilDiv(inner_most_dimension_numel, bdimx), max_unroll);

    // Readjust bdimy and bdimz
    bdimy = std::max(min_warp_size / bdimx, (int64_t)1);

    bdimz = std::min(
        std::max(target_threads_in_block / (bdimx * bdimy), (int64_t)1),
        ceilDiv(total_reduction_numel, inner_most_dimension_numel));

    bdimy = std::max(target_threads_in_block / (bdimx * bdimz), (int64_t)1);
  }

  godim = ceilDiv(total_iteration_numel, bdimy);

  bool vectorize = false;

  // Move unrolling factor into vectorization upto vectorization limit.
  if (vectorize_factor > 1 && inner_reduction_unroll_factor > 1) {
    vectorize = true;
    inner_reduction_unroll_factor = std::min(
        scheduler_utils::lastPow2(inner_reduction_unroll_factor),
        (int64_t)vectorize_factor);
  }

  // Attempt to put some unrolling into the outer reduction if inner hasn't
  // taken the max unrolling
  if (inner_reduction_unroll_factor < max_unroll) {
    outer_reduction_unroll_factor = std::min(
        ceilDiv(max_unroll, inner_reduction_unroll_factor),
        ceilDiv(
            ceilDiv(total_reduction_numel, inner_most_dimension_numel), bdimz));
  }

  int64_t remainder_in_reduction = ceilDiv(
      total_reduction_numel,
      bdimx * inner_reduction_unroll_factor * bdimz *
          outer_reduction_unroll_factor * target_iterations);

  int64_t remainder_in_inner_dim = ceilDiv(
      inner_most_dimension_numel,
      bdimx * inner_reduction_unroll_factor * target_iterations);

  // If we haven't gotten to the max_unroll case, try to take it out of the
  // iteration domain
  if (inner_reduction_unroll_factor * outer_reduction_unroll_factor <
      max_unroll) {
    // Don't go over a combined inner/outer unroll of max_unroll
    auto unroll_available = ceilDiv(
        max_unroll,
        inner_reduction_unroll_factor * outer_reduction_unroll_factor);

    if (unroll_available > 1 && godim > 2 * device_multiprocessor_count) {
      unroll_available = std::min(
          unroll_available, ceilDiv(godim, 2 * device_multiprocessor_count));
      iter_unroll_factor = unroll_available;
    }
  }

  godim = ceilDiv(total_iteration_numel, bdimy * iter_unroll_factor);

  // Clang tidy
  constexpr int64_t kEight = 8;
  // Cross grid reduction if we haven't hit our target blocks, and we have manyr
  // reduction elements.
  if ((godim < target_blocks && remainder_in_reduction >= 0) ||
      (remainder_in_reduction >= kEight)) {
    auto grdim = std::min(remainder_in_reduction, bdimx * bdimy * kEight);

    gridim = remainder_in_inner_dim;
    grodim = std::max(grdim / gridim, (int64_t)1);
    grodim = std::max(
        std::min(remainder_in_reduction / remainder_in_inner_dim, grodim),
        (int64_t)1);
  }

  // Try to do some cleanup of ragged waves on device, don't do this if we're
  // trying to do a 3D schedule. godim is a remainder of a split, so can only
  // control gridim
  if (grodim == 1 &&
      // If we have less than 8 waves of blocks
      gridim * godim < device_multiprocessor_count * kEight &&
      // And we don't have an even divisible number of blocks
      (gridim * godim) % device_multiprocessor_count != 0 &&
      // And we have more than one wave
      gridim * godim > device_multiprocessor_count) {
    // round waves down
    auto waves =
        std::max((godim * gridim) / device_multiprocessor_count, (int64_t)1);
    auto new_gridim =
        std::max((waves * device_multiprocessor_count) / godim, (int64_t)1);
    if (
        // If difference is less than 25% of the original gridim
        (new_gridim - gridim) * 4 < gridim &&
        // and difference is less than 25% of the original number of blocks
        ((new_gridim * godim) - (gridim * godim)) * 4 < gridim * godim) {
      gridim = new_gridim;
    }
  }

  if (grodim > 1 || gridim > 1) {
    // Grid reductions do not support unrolling iteration dimension, revert if
    // set. Recalculate godim.
    if (iter_unroll_factor) {
      iter_unroll_factor = 1;
      godim = ceilDiv(total_iteration_numel, bdimy * iter_unroll_factor);
    }
    // This could mess up parallelization which could be redone, but that would
    // require iterating over this entire function.
  }

  auto rparams = std::make_unique<ReductionParams>();
  rparams->fastest_dim = true;
  rparams->cross_block_inner_reduction = true;
  rparams->block_dim_inner_reduction = ParallelType::TIDx;
  rparams->cross_grid_inner_reduction = gridim > 1;
  rparams->multiple_reds_per_blk = bdimy > 1;
  bool pad_bdimx = bdimx > 16 &&
      bdimx * bdimy <
          (int64_t)at::cuda::getCurrentDeviceProperties()->maxThreadsPerBlock;
  rparams->pad_inner_reduction_to_warp = pad_bdimx;

  if (rparams->pad_inner_reduction_to_warp) {
    // Adjust bdimx based on padding
    auto min_warp_size =
        (int64_t)at::cuda::getCurrentDeviceProperties()->warpSize;
    bdimx = bdimx % min_warp_size == 0
        ? bdimx
        : bdimx + min_warp_size - bdimx % min_warp_size;
  }

  rparams->unroll_factor_inner_reduction = inner_reduction_unroll_factor;
  rparams->vectorize_inner_reduction = vectorize;

  if (rparams->multiple_reds_per_blk) {
    rparams->block_dim_iter_dom = ParallelType::TIDy;
  }

  rparams->unroll_factor_iter_dom = iter_unroll_factor;

  rparams->schedule_3D = total_reduction_numel != inner_most_dimension_numel;
  // Outer reduction domain
  if (rparams->schedule_3D) {
    rparams->cross_grid_outer_reduction = grodim > 1;
    if (bdimz > 1) {
      rparams->block_dim_outer_reduction = ParallelType::TIDz;
      rparams->cross_block_outer_reduction = true;
    }
    rparams->unroll_factor_outer_reduction = outer_reduction_unroll_factor;
  }

  int64_t gdimx = LaunchParams::UNINITIALIZED_VAL;
  int64_t gdimy = LaunchParams::UNINITIALIZED_VAL;
  int64_t gdimz = LaunchParams::UNINITIALIZED_VAL;

  // If we have a cross grid case we want to have gdimy assigned to godim and
  // gdimx assigned to grdim. Otherwise it's helpful to pull godim into gdimx in
  // case it's larger than gdimy can hold, as not doing so can thrash the cache.

  if (rparams->cross_grid_inner_reduction) {
    rparams->grid_dim_inner_reduction = ParallelType::BIDx;
    rparams->split_grid_dim_inner_reduction = true;
    gdimx = std::min(gridim, scheduler_utils::x_grid_limit);

    rparams->grid_dim_iter_dom = ParallelType::BIDy;
    if (godim > scheduler_utils::y_grid_limit) {
      rparams->split_grid_dim_iter_dom_outer = true;
      gdimy = scheduler_utils::y_grid_limit;
    }

  } else {
    rparams->grid_dim_iter_dom = ParallelType::BIDx;
    if (godim > scheduler_utils::x_grid_limit) {
      rparams->split_grid_dim_iter_dom_outer = true;
      gdimx = scheduler_utils::x_grid_limit;
    }
  }

  if (rparams->cross_grid_outer_reduction) {
    if (rparams->cross_block_inner_reduction) {
      rparams->grid_dim_outer_reduction = ParallelType::BIDz;
      gdimz = std::min(grodim, scheduler_utils::z_grid_limit);
      rparams->split_grid_dim_outer_reduction = true;
    } else {
      rparams->grid_dim_outer_reduction = ParallelType::BIDy;
      gdimy = std::min(grodim, scheduler_utils::y_grid_limit);
      rparams->split_grid_dim_outer_reduction = true;
    }
  }

  rparams->lparams = LaunchParams(
      gdimx,
      gdimy,
      gdimz,
      bdimx,
      bdimy > 1 ? bdimy : LaunchParams::UNINITIALIZED_VAL,
      bdimz > 1 ? bdimz : LaunchParams::UNINITIALIZED_VAL);

  if (isDebugDumpEnabled(DebugDumpOption::SchedulerDebug)) {
    debug() << "\n===== Inner 3D Reduction Stats ========\n"
            << "total_reduction_numel: "
            << total_reduction_numel / inner_most_dimension_numel << " * "
            << inner_most_dimension_numel << "\n"
            << "total_iteration_numel: " << total_iteration_numel << "\n"
            << "vectorize_factor: " << vectorize_factor << "\n"
            << "n_tensor_inputs: " << n_tensor_inputs << "\n"
            << "max_dtype_size_bit_for_vectorization: "
            << max_dtype_size_bit_for_vectorization << "\n"
            << "block(" << bdimx << ", " << bdimy << ", " << bdimz << ")"
            << std::endl;
    debug() << rparams->toString() << std::endl;
  }

  // If 3d, check if it's supported by the scheduler, otherwise force 1D
  // schedule
  if (rparams->schedule_3D) {
    if (rparams->multiple_reds_per_blk &&
        (rparams->cross_grid_inner_reduction ||
         rparams->cross_grid_outer_reduction)) {
      if (isDebugDumpEnabled(DebugDumpOption::SchedulerDebug)) {
        debug() << "\n===== UNSUPPORTED REDUCTION HEURISTIC ========\n";
        debug() << rparams->multiple_reds_per_blk << ", "
                << (rparams->unroll_factor_inner_reduction > 1) << ", "
                << rparams->cross_grid_inner_reduction << std::endl;
      }
      return inner2dReductionHeuristic(
          total_reduction_numel,
          total_iteration_numel,
          (int64_t)n_tensor_inputs,
          (int64_t)max_dtype_size_bit_for_vectorization,
          (int64_t)vectorize_factor,
          has_mufu_computation);
    }
  }

  return rparams;
}

struct OuterReductionParams {
  OuterReductionParams(
      int64_t total_iteration_numel,
      int64_t total_reduction_numel)
      : total_iteration_numel(total_iteration_numel),
        total_reduction_numel(total_reduction_numel) {}
  // iteration dim paras
  // iteration elements = iter_unroll * bdimx * gidim
  int64_t iter_unroll_factor = 1;
  int64_t bdimx = 1;
  int64_t gidim = 1;
  // reduction dim paras
  // reduction elments = redu_unroll * bdimy * grdim * redu_serial
  int64_t redu_unroll_factor = 1;
  int64_t bdimy = 1;
  int64_t grdim = 1;
  int64_t redu_serial = 1;

  // iteration and reduction dim elements
  int64_t total_iteration_numel = -1;
  int64_t total_reduction_numel = -1;

  // Helper to figure out how much is left in the iter or reduction dim
  int64_t iDimAvail() const {
    return ceilDiv(total_iteration_numel, gidim * bdimx * iter_unroll_factor);
  }
  int64_t rDimAvail() const {
    return ceilDiv(total_reduction_numel, grdim * bdimy * redu_unroll_factor);
  };

  std::string toString() const {
    std::stringstream ss;
    ss << "\n===== Outer Reduction Stats ========\n"
       << "total_reduction_numel: " << total_reduction_numel << "\n"
       << "total_iteration_numel: " << total_iteration_numel << "\n"
       << "vectorize_factor: " << iter_unroll_factor << "\n"
       << "redu_unroll_factor: " << redu_unroll_factor << "\n"
       << "grid(" << gidim << ", " << grdim << ", 1)"
       << "\n"
       << "block(" << bdimx << ", " << bdimy << ", 1)" << std::endl;
    return ss.str();
  }
};
// compare block reduction with grid reduction
bool isBetterThan(
    const OuterReductionParams& block_params,
    const OuterReductionParams& grid_params,
    int64_t sm_count) {
  NVF_ERROR(
      block_params.grdim == 1,
      "Only support compare block reduction heuristic with grid reduction not "
      "vice versa");

  // use block reduction if its SM usage >= 90% and its iter_unroll_factor is
  // equal or larger than grid reduction. These two conditions ensure high SM
  // usage and efficient global memory access. The corresponding block reduction
  // avoids the overhead of inter-block data exchange through global memory.
  // It is faster than grid reduction even not all SMs are used.
  // TODO: if we know the fusion is memory bound (e.g. pure reduction), we can
  // use a lower threshold. For computation bound (e.g. gelu bwd), relaxing
  // the threshold leads to regression.
  float f_wave = (float)block_params.gidim / (float)sm_count;
  float sm_efficiency = f_wave / std::ceil(f_wave);
  if (sm_efficiency >= 0.9f &&
      block_params.iter_unroll_factor >= grid_params.iter_unroll_factor) {
    return true;
  }

  // prefer block reduction if it uses same or more blocks than grid
  // reduction. This may happen when input size is very small, e.g. 512 x 128.
  // Current grid reduction heuristic start bdimx from 16 and prioritize
  // vectorization. It may not be able to fully utilize all the SMs.
  // This is really a tiny problem. The performance impact is most likely in a
  // range of a few us in <10us kernels
  // This condition is a WAR. Ideally, the grid reduction heuristics should be
  // improved, but given that the impact is likely negligible, we decided to do
  // this quick adjustment.
  if (block_params.gidim * block_params.grdim >=
      grid_params.gidim * grid_params.grdim) {
    return true;
  }

  // use grid reduction
  return false;
}

std::unique_ptr<ReductionParams> heuristicParaToSchedulerPara(
    const OuterReductionParams& params) {
  int64_t gdimx = LaunchParams::UNINITIALIZED_VAL;
  int64_t gdimy = LaunchParams::UNINITIALIZED_VAL;

  // In these instances latency of the cleanup may be significant so flip gdimx
  // and gdimy to try and prevent all cleanup from happening at the
  // same time
  // Always disabled for now.
  // bool flip_grid = gidim > 1 && gidim < 8;
  const bool flip_grid = false;
  auto rparams = std::make_unique<ReductionParams>();

  // Configure register count to achieve 50% occupancy with 1024 threads per SM.
  // Typical thread configurations are 512 or 1024 threads per block, both
  // evenly divisible. This setup ensures each thread is limited to 64
  // registers, preventing compilers from allocating more registers per thread,
  // which could reduce occupancy and degrade performance.
  int64_t target_threads_per_sm =
      (int64_t)at::cuda::getCurrentDeviceProperties()
          ->maxThreadsPerMultiProcessor /
      2;
  int64_t threads_per_block = params.bdimx * params.bdimy;
  int64_t threads_per_sm =
      scheduler_utils::safeDiv(target_threads_per_sm, threads_per_block) *
      threads_per_block;
  int64_t reg_per_thread = getRegPerThreadGivenThreadsPerSM(threads_per_sm);
  rparams->cparams.maxrregcount = reg_per_thread;

  // cross grid implies cross block
  rparams->cross_block_inner_reduction = params.bdimy > 1 || params.grdim > 1;
  rparams->cross_grid_inner_reduction = params.grdim > 1;
  if (rparams->cross_grid_inner_reduction) {
    rparams->split_grid_dim_inner_reduction = true;
    rparams->grid_dim_inner_reduction =
        flip_grid ? ParallelType::BIDx : ParallelType::BIDy;
    if (flip_grid) {
      gdimx = std::min(params.grdim, scheduler_utils::x_grid_limit);
    } else {
      gdimy = std::min(params.grdim, scheduler_utils::y_grid_limit);
    }
  }
  rparams->multiple_reds_per_blk =
      params.bdimx > 1 || params.iter_unroll_factor > 1;

  if (rparams->multiple_reds_per_blk) {
    rparams->block_dim_iter_dom = ParallelType::TIDx;
  }

  rparams->grid_dim_iter_dom =
      flip_grid ? ParallelType::BIDy : ParallelType::BIDx;
  if (params.gidim > (flip_grid ? scheduler_utils::y_grid_limit
                                : scheduler_utils::x_grid_limit)) {
    rparams->split_grid_dim_iter_dom_outer = true;
    if (flip_grid) {
      gdimy = scheduler_utils::y_grid_limit;
    } else {
      gdimx = scheduler_utils::x_grid_limit;
    }
  }

  rparams->flip_grid = flip_grid;

  if (rparams->cross_block_inner_reduction) {
    if (rparams->block_dim_iter_dom == ParallelType::TIDx) {
      rparams->block_dim_inner_reduction = ParallelType::TIDy;
    } else {
      rparams->block_dim_inner_reduction = ParallelType::TIDx;
    }
  }

  rparams->unroll_factor_inner_reduction = params.redu_unroll_factor;

  rparams->unroll_factor_iter_dom = params.iter_unroll_factor;
  rparams->vectorize_iter_dom = params.iter_unroll_factor > 1;

  rparams->lparams = LaunchParams(
      gdimx,
      gdimy,
      LaunchParams::UNINITIALIZED_VAL,
      rparams->multiple_reds_per_blk ? params.bdimx : params.bdimy,
      rparams->multiple_reds_per_blk ? params.bdimy
                                     : LaunchParams::UNINITIALIZED_VAL,
      LaunchParams::UNINITIALIZED_VAL);

  if (isDebugDumpEnabled(DebugDumpOption::SchedulerDebug)) {
    debug() << params.toString() << std::endl;
    debug() << rparams->toString() << std::endl;
  }
  return rparams;
}

OuterReductionParams getBlockOuterReduction(
    int64_t total_reduction_numel,
    int64_t total_iteration_numel,
    int64_t vectorize_factor,
    int64_t max_unroll,
    int64_t sm_count,
    int64_t max_threads_per_block) {
  OuterReductionParams params(total_iteration_numel, total_reduction_numel);

  int64_t sm_count_pow2 = scheduler_utils::lastPow2(sm_count);
  // Step-1, set iteration dim
  // (1) start with bdimx = 8, gidim = 32, iter_unroll = 1.
  // starts bdimx from 8, ensures each warp spans at most into 4 different rows.
  // when each thread reads 16 bytes, each warp does 4 transactions each
  // with 128 bytes. This is the maximum global memory transaction size
  // for each warp.
  // starts gidim from 32 and iter_unroll from 1, defers fully vectorization
  // to SM usage is high enough.
  params.bdimx = std::min(8L, params.iDimAvail());
  params.gidim = std::min(std::min(32L, sm_count_pow2), params.iDimAvail());
  params.iter_unroll_factor = 1;

  // (2) increase iter_unroll to its maximum following two rules:
  // (2.1) ensure divisible split
  // (2.2) leave enough blocks to saturate the device.
  // For example on any GPU with more than 32 SMs
  // bdimx-vect-gidim = 8-1-32  = 256
  // bdimx-vect-gidim = 8-2-32  = 512
  // bdimx-vect-gidim = 8-2-64  = 1024
  // bdimx-vect-gidim = 8-4-64  = 2048
  // bdimx-vect-gidim = 8-4-128 = 4096
  // bdimx-vect-gidim = 8-8-128 = 8192
  int64_t max_iter_unroll = vectorize_factor;
  while (params.iDimAvail() > 1) {
    if (params.iDimAvail() % 2 == 0 &&
        params.iter_unroll_factor * 2 <= max_iter_unroll) {
      params.iter_unroll_factor *= 2;
    }
    if (params.iDimAvail() > 1) {
      params.gidim *= 2;
    }
    if (params.iter_unroll_factor == max_iter_unroll) {
      break;
    }
  }

  // (3) reset gidim, ensures enough blocks to saturate the
  // device but doesn't use more SMs than available.
  params.gidim = std::min(
      ceilDiv(total_iteration_numel, params.bdimx * params.iter_unroll_factor),
      sm_count);

  // (4) increase bdimx to its maximum
  params.bdimx =
      ceilDiv(total_iteration_numel, params.gidim * params.iter_unroll_factor);
  params.bdimx = std::min(
      scheduler_utils::roundUpPow2(params.bdimx),
      scheduler_utils::roundUpToN(params.bdimx, 32));
  params.bdimx = std::min(params.bdimx, max_threads_per_block);

  // (5) re-calculate gidim after bdimx to fix round up differences. Also
  // handles extreme cases where iter dim is larger than iter_unroll_factor x
  // max_threads_per_block x sm count
  params.gidim =
      ceilDiv(total_iteration_numel, params.bdimx * params.iter_unroll_factor);

  // Step-2, set Reduction dim
  // (1) reduction unroll takes what is left by iter unroll
  params.redu_unroll_factor = std::min(
      params.rDimAvail(),
      scheduler_utils::safeDiv(max_unroll, params.iter_unroll_factor));

  // (2) bdimy takes what is left by bdimx.
  params.bdimy =
      std::min(params.rDimAvail(), max_threads_per_block / params.bdimx);

  // Step-3, final check
  // (1) revisit bdimx just in case bdimy doesn't take all the left threads
  while (params.bdimy * params.bdimx * 2 <= max_threads_per_block &&
         params.gidim / 2 >= sm_count_pow2) {
    params.bdimx *= 2;
    params.gidim /= 2;
  }
  return params;
}

OuterReductionParams getGridOuterReduction(
    const int64_t total_reduction_numel,
    const int64_t total_iteration_numel,
    const int64_t n_tensor_inputs,
    const int64_t max_dtype_size_bit_for_vectorization,
    const int64_t vectorize_factor,
    const int64_t max_unroll,
    const int64_t sm_count,
    const int64_t max_threads_per_sm) {
  // grid or block reduction
  const int64_t n_elems = total_reduction_numel * total_iteration_numel;
  // Try to use 4 * SM blocks to reduce communication cost. But still
  // use 8 * SM blocks if the problem size is large, it increased requested
  // waves, helps hiding memory latency, if still use 4 * SM blocks, about 5%
  // regression for compute bound kernels. Not much change for memory bound.
  const int64_t n_waves = n_elems >= (int64_t)64 * 1024 * 1024 ? 8 : 4;

  // if data fits in l2 and we need more parallelization in the iter dim,
  // we can use a smaller warp size. While thread local data fits in l1, and
  // iter dim is really small, we can use <32 threads per warp.
  // TODO: Could get a much more accurate estimation of it the problem fits in
  // L2
  const bool fits_in_l2 =
      n_elems * max_dtype_size_bit_for_vectorization * n_tensor_inputs <
      at::cuda::getCurrentDeviceProperties()->l2CacheSize * 8;

  const int64_t min_warp_size = fits_in_l2 ? 16 : 32;

  // Set some targets for parallelization
  int64_t target_threads_in_block = min_warp_size;
  // Start target blocks at roughly a quarter wave if available
  int64_t target_blocks =
      std::min(ceilDiv(sm_count, (int64_t)4), ceilDiv(n_elems, min_warp_size));
  int64_t target_unroll = 1;

  auto available_parallelism =
      [&n_elems, &target_threads_in_block, &target_blocks, &target_unroll]() {
        return ceilDiv(
            n_elems, target_threads_in_block * target_blocks * target_unroll);
      };

  // If there's available parallelism, divide it between threads, blocks, and
  // vectorization
  // Threads are currently at a warp (16 or 32)
  // Blocks are currently at a quarter wave
  // Unroll is currently at 1
  while (
      // and there's parallelism left
      available_parallelism() > 1 &&
      (
          //  There's a place to put it in the block
          target_threads_in_block < ceilDiv(max_threads_per_sm, (int64_t)4)
          // There's a place to put it in the device
          || target_blocks < sm_count * n_waves
          // There's a place to put it in unrolling
          || target_unroll < max_unroll)) {
    if (target_threads_in_block < ceilDiv(max_threads_per_sm, (int64_t)4)) {
      target_threads_in_block *= 2;
    }

    if (target_blocks < sm_count * n_waves && available_parallelism() > 1) {
      target_blocks *= 2;
    }

    // Delay increasing unroll until we have more than one block per SM.
    // Assuming each SM can take more than one block.
    if (target_blocks > sm_count && target_unroll < max_unroll &&
        available_parallelism() > 1) {
      target_unroll *= 2;
    }
  }

  // Fill out unrolling if possible
  if (target_unroll < max_unroll && available_parallelism() > 1) {
    target_unroll = std::min(available_parallelism(), max_unroll);
  }

  target_unroll = scheduler_utils::lastPow2(target_unroll);

  // To get to target threads:
  // Prioritize
  // (1) x dim in iter domain
  // (2) unrolling in iter domain
  // (3) y in reduction domain
  // To get target blocks:
  // Prioritize
  // (1) x dim in multiple outputs
  // (2) y dim in multiple reductions - need to flip unrolling to reduction
  // domain for this

  // Blocks for reductions
  int64_t grdim = 1;
  // Blocks for outputs
  int64_t gidim = 1;

  // Threads for reduction
  int64_t bdimy = 1;
  // Threads for output
  int64_t bdimx = 1;

  // Unroll amount
  int64_t inner_reduction_unroll_factor = 1;
  int64_t iter_unroll_factor = 1;

  // Helper lambda's to figure out how much is left in the iter or reduction dim
  auto iDimAvail = [&]() {
    return ceilDiv(total_iteration_numel, gidim * bdimx * iter_unroll_factor);
  };

  auto rDimAvail = [&]() {
    return ceilDiv(
        total_reduction_numel, grdim * bdimy * inner_reduction_unroll_factor);
  };

  // Start bdimx as a warp
  bdimx = std::min(min_warp_size, total_iteration_numel);

  if (total_iteration_numel > bdimx && total_iteration_numel < bdimx * 2) {
    // If rounding up would require less than 3/4 of the warp
    if ((total_iteration_numel % bdimx) * 4 < bdimx * 3) {
      // Round up to avoid nasty edge effects
      bdimx = total_iteration_numel;
    }
  }

  // gradually increased iter_unroll_factor from 1 to 2, 4, 8, ensure the
  // split is divisible. This improves performance when iteration dim is not
  // power of 2, e.g. 1600 and 4800.
  int64_t max_iter_unroll_factor =
      std::min(vectorize_factor, std::min(iDimAvail(), target_unroll));
  while (total_iteration_numel % (bdimx * iter_unroll_factor * 2) == 0 &&
         iter_unroll_factor * 2 <= max_iter_unroll_factor) {
    iter_unroll_factor *= 2;
  }

  // If iteration numel is not something huge like 64k we probably shouldn't do
  // this, maybe it could be 2 * device_multi_count to make sure iter dim is
  if (iDimAvail() > sm_count) {
    // Put more into bdimx
    bdimx = std::min(
        // Leave 2x a full wave of blocks
        ceilDiv(total_iteration_numel, iter_unroll_factor * sm_count),
        // Don't exceed max thread count
        target_threads_in_block);
  }

  // Round bdimx to pow2 since target threads per block is pow2
  bdimx = scheduler_utils::roundUpPow2(bdimx);

  // Fill bdimy with left over threads
  bdimy = std::min(
      scheduler_utils::safeDiv(target_threads_in_block, bdimx),
      total_reduction_numel);

  bdimy = roundDownPow2OrMultipleOf(bdimy, 8);

  // Move parallelization into unrolling the reduction dimension if
  // parallelizing iteration dimension didn't take the available unroll factor.
  if (iter_unroll_factor < max_unroll && rDimAvail() > 2) {
    inner_reduction_unroll_factor = std::min(
        rDimAvail(), scheduler_utils::safeDiv(max_unroll, iter_unroll_factor));

    inner_reduction_unroll_factor =
        scheduler_utils::lastPow2(inner_reduction_unroll_factor);
  }

  gidim = iDimAvail();

  // Try to hit a wave by going cross reduction
  grdim = std::min(rDimAvail(), ceilDiv(sm_count, gidim));
  // Extend to go to target blocks
  if (gidim * grdim < target_blocks) {
    // What should we use out of the reduction factor to hit target blocks? Make
    // sure we have 2 reductions per thread beyond what's already set as we
    // consider expanding to target block
    grdim = std::min(
        // At least 2 iterations of the reduction per thread on top of unroll
        ceilDiv(rDimAvail() * grdim, 2),
        // Expand to target blocks
        ceilDiv(target_blocks, gidim));
  }

  // If there isn't a lot of available parallelism from the iteration dimension,
  // expand across the reduction dimension. This has to be done carefully.
  // expand further
  if (rDimAvail() > 16 &&
      ceilDiv(total_iteration_numel, min_warp_size) < sm_count * 2) {
    // Find minimum we want to parallelize by, we don't want blocks striding
    // across too many elements: In the parallel scheme [rBIDy, remainder,
    // iBIDx, rTIDy, i_unroll, r_unroll] figure out how many bytes iterations
    // across remainder stride
    int64_t bits_stride_remainder = max_dtype_size_bit_for_vectorization *
        bdimx * bdimy * iter_unroll_factor * inner_reduction_unroll_factor;
    // Empiercally found stride shouldn't exceed 256kiB boundaries in a block
    int64_t kMaxStrideBit = 128l * 1024l * 8;

    int64_t max_remainder_size =
        scheduler_utils::safeDiv(kMaxStrideBit, bits_stride_remainder);

    int64_t grdim_for_stride = ceilDiv(
        total_reduction_numel,
        max_remainder_size * bdimy * inner_reduction_unroll_factor);

    grdim = grdim_for_stride;
  }

  // Try to do some cleanup of ragged waves on device
  if (
      // If we have less than 8 waves of blocks
      grdim * gidim < sm_count * 16 &&
      // And we don't have an even divisible number of blocks
      (grdim * gidim) % sm_count != 0 &&
      // And we have more than one wave
      grdim * gidim > sm_count) {
    // round waves down
    auto waves = std::max((gidim * grdim) / sm_count, (int64_t)1);
    auto new_grdim = std::max((waves * sm_count) / gidim, (int64_t)1);
    if ((grdim - new_grdim) * 4 <= grdim &&
        new_grdim * gidim % sm_count > grdim * gidim % sm_count) {
      grdim = new_grdim;
    }
  }

  OuterReductionParams params(total_iteration_numel, total_reduction_numel);
  params.bdimx = bdimx;
  params.bdimy = bdimy;
  params.grdim = grdim;
  params.gidim = gidim;
  params.iter_unroll_factor = iter_unroll_factor;
  params.redu_unroll_factor = inner_reduction_unroll_factor;
  return params;
}

std::unique_ptr<ReductionParams> outerReductionHeuristic(
    const int64_t total_reduction_numel,
    const int64_t total_iteration_numel,
    const int64_t n_tensor_inputs,
    const int64_t max_dtype_size_bit_for_vectorization,
    const size_t vectorize_factor,
    const bool has_mufu_computation) {
  // WARNING: Current device for codegen may not be the target device
  auto dev_prop = at::cuda::getCurrentDeviceProperties();
  const int64_t sm_count = (int64_t)dev_prop->multiProcessorCount;
  const int64_t max_threads_per_block = (int64_t)dev_prop->maxThreadsPerBlock;
  const int64_t max_threads_per_sm =
      (int64_t)dev_prop->maxThreadsPerMultiProcessor;
  // Set register used to store vectorized and unrolled data loaded from gmem.
  // A large value allows more unroll and vectorization, which is beneficial
  // for memory-bound kernels. However, it increases register pressure and may
  // lead to lower occupancy which is bad for compute-bound kernels. In most
  // cases, the scheduler uses 512 threads and to reach an occupancy of 50%,
  // each thread can use up to 64 registers, here only 8 or 32 registers are
  // reserved for unroll and vectorization. The fused ops can have 56 or 32
  // registers for other purposes. Test shows it leads to 50% occupancy for
  // outer reduction without fused ops and 50% occupancy for gelu backward which
  // fused 21 ops including the expensive tanh op.
  int64_t buffer_reg_count = 0L, input_factor = 0L;
  if (has_mufu_computation) {
    // when we have expensive ops, computation cost of each thread is already
    // high, prioritize thread level parallelism, 8 registers are reserved for
    // unroll and vectorization, the corresponding unroll factor is 2 if fully
    // vectorized by 16 bytes. when there are more than 4 inputs, reduce to 1.
    buffer_reg_count = 8L;
    input_factor = (int64_t)n_tensor_inputs >> 2;
  } else {
    // when there is no expensive op, prioritize instruction level parallelism,
    // 32 registers are reserved for unroll and vectorization, the corresponding
    // unroll factor is 8 if fully vectorized by 16 bytes. Gradually reduce
    // unroll factor if we have many inputs, e.g. 2 inputs -> unroll by 4; 3 or
    // 4 inputs -> unroll by 2; 5, 6, 7 or 8 inputs -> unroll by 1.
    buffer_reg_count = 32L;
    input_factor = n_tensor_inputs;
  }
  auto const max_unroll = ceilDiv(
      // Available unrolling based on size of data type
      buffer_reg_count * scheduler_utils::bits_per_register /
          (int64_t)max_dtype_size_bit_for_vectorization,
      // Reduce unrolling if we have many inputs
      scheduler_utils::lastPow2(std::max(input_factor, (int64_t)1)));

  // block or grid reduction heuristic
  auto grid_params = getGridOuterReduction(
      total_reduction_numel,
      total_iteration_numel,
      n_tensor_inputs,
      max_dtype_size_bit_for_vectorization,
      (int64_t)vectorize_factor,
      max_unroll,
      sm_count,
      max_threads_per_sm);

  // block reduction heuristic
  auto block_params = getBlockOuterReduction(
      total_reduction_numel,
      total_iteration_numel,
      (int64_t)vectorize_factor,
      max_unroll,
      sm_count,
      max_threads_per_block);

  // pick the better heuristic
  if (isBetterThan(block_params, grid_params, sm_count)) {
    return heuristicParaToSchedulerPara(block_params);
  } else {
    return heuristicParaToSchedulerPara(grid_params);
  }
}

std::unique_ptr<ReductionParams> reductionHeuristic(
    const int64_t total_reduction_numel,
    const int64_t total_iteration_numel,
    const int64_t inner_most_dimension_numel,
    const bool fastest_dim_reduction,
    const int64_t n_tensor_inputs,
    const int64_t max_dtype_size_bit_for_vectorization,
    const size_t vectorize_factor,
    const bool has_mufu_computation) {
  if (fastest_dim_reduction) {
    if (total_reduction_numel == inner_most_dimension_numel) {
      return inner2dReductionHeuristic(
          total_reduction_numel,
          total_iteration_numel,
          (int64_t)n_tensor_inputs,
          (int64_t)max_dtype_size_bit_for_vectorization,
          (int64_t)vectorize_factor,
          has_mufu_computation);
    } else {
      return inner3dReductionHeuristic(
          total_reduction_numel,
          total_iteration_numel,
          inner_most_dimension_numel,
          (int64_t)n_tensor_inputs,
          (int64_t)max_dtype_size_bit_for_vectorization,
          vectorize_factor,
          has_mufu_computation);
    }

  } else {
    // 3D schedules not enabled for outer reductions
    return outerReductionHeuristic(
        total_reduction_numel,
        total_iteration_numel,
        (int64_t)n_tensor_inputs,
        (int64_t)max_dtype_size_bit_for_vectorization,
        vectorize_factor,
        has_mufu_computation);
  }
}

std::unique_ptr<ReductionParams> getReductionHeuristics(
    Fusion* fusion,
    SchedulerRuntimeInfo& runtime_info,
    HeuristicDataCache* data_cache) {
  FusionGuard fg(fusion);

  auto reduction_tv_entry =
      HeuristicDataCacheEntry<HeuristicCompileTime::ReductionTVs>(
          data_cache, [&fusion]() {
            return std::make_unique<std::vector<TensorView*>>(
                scheduler_utils::getReductionTvs(fusion));
          });

  auto& reduction_tvs = reduction_tv_entry.get();

  NVF_ERROR(!reduction_tvs.empty(), "Need reduction tensor views to schedule.");

  auto reduction_tv = reduction_tvs[0];

  NVF_ERROR(
      reduction_tv->hasReduction(), "TensorView doesn't have a reduction.");

  const auto red_expr = reduction_tv->definition();

  NVF_ERROR(
      ir_utils::isReductionOp(red_expr),
      "TensorView doesn't have a reduction.");

  auto properties = scheduler_utils::getReductionProperties(
      fusion, runtime_info, reduction_tv);

  auto tv_inps = ir_utils::filterByType<TensorView>(fusion->inputs());
  NVF_ERROR(
      !tv_inps.empty(),
      "Tried to schedule a fusion with no tensor inputs, currently not "
      "supported.");

  auto reduced_tv = ir_utils::getSoleProducerTv(reduction_tv);

  auto unrollable_inputs_outputs_entry =
      HeuristicDataCacheEntry<HeuristicCompileTime::UnrollableInputsAndOutputs>(
          data_cache, [&reduced_tv]() {
            return std::make_unique<std::vector<TensorView*>>(
                scheduler_utils::getInputsOutputsWithInnerDim(
                    reduced_tv, false, false));
          });

  auto& unrollable_inputs_outputs = unrollable_inputs_outputs_entry.get();

  // Although properties contains runtime information
  // "inner_most_dimension_ndims" is a compile time value
  auto vec_break_point = HeuristicDataCacheEntry<
      HeuristicCompileTime::VectorizationBreakPointOfReductionProducer>(
      data_cache, [&reduction_tv, &reduced_tv, &properties]() {
        return std::make_unique<int64_t>(
            vectorize_helper::getVectorizationBreakPointOfReductionProducer(
                reduction_tv,
                reduced_tv,
                properties.inner_most_dimension_ndims));
      });

  // `getVectorizationFactor` makes comparisons assuming the break point
  // is wrt to the logical domain size. Schedulers such as reduction
  // compute it based on loop domain size, whereas pointwise computes it
  // based on non-device/non-reduction domain size and accounts for device
  // dimensions during scheduling.
  // TODO (priya): We should make this consistent across all schedulers
  int64_t num_device_dims = numDeviceDims(reduced_tv);
  int64_t no_device_break_point = vec_break_point.get() - num_device_dims;
  const auto vectorize_factor = vectorize_helper::getVectorizationFactor(
      runtime_info, reduced_tv, data_cache, no_device_break_point);

  // Base max dtype and n_tensor_inputs on tensors that are vectorizable (i.e.
  // share inner dimension with data pattern we're looking at).
  int64_t max_dtype_size_bit_for_vectorization = 0;

  // TODO: This might be better if it was the larger of input or outputs. Would
  // be even better if we had better analysis as not all unrolled elements have
  // to be alive at the same time.
  int64_t n_tensor_inputs = 0;
  for (auto tv : unrollable_inputs_outputs) {
    max_dtype_size_bit_for_vectorization = std::max(
        max_dtype_size_bit_for_vectorization,
        static_cast<int64_t>(dataTypeSizeBit(
            tv->getDataType().value(), runtime_info.getIndexType())));
    if (!tv->isFusionInput()) {
      continue;
    }
    n_tensor_inputs++;
  }

  // If max_dtype_size_bit_for_vectorization is 0, it means there
  // is no vectorizable input/output. For this case, we set it to 8
  // as a default value to prevent having a too large vectorization factor.
  // TODO: run a benchmark and see if there is a better default value.
  if (max_dtype_size_bit_for_vectorization == 0) {
    max_dtype_size_bit_for_vectorization = 8;
  }

  // Protect heuristics div by 0:
  n_tensor_inputs = std::max(n_tensor_inputs, 1l);

  bool has_mufu_computation = scheduler_utils::hasExpensiveMUFUops(fusion);

  auto heuristic = reductionHeuristic(
      properties.total_reduction_numel,
      properties.total_iteration_numel,
      properties.inner_most_dimension_numel,
      properties.fastest_dim_reduction,
      n_tensor_inputs,
      max_dtype_size_bit_for_vectorization,
      vectorize_factor,
      has_mufu_computation);
  heuristic->cparams.index_type = runtime_info.getIndexType();
  return heuristic;
}

// fusion is the input IR that will be modified by this function
void scheduleReduction(Fusion* fusion, const ReductionParams* rparams) {
  FusionGuard fg(fusion);

  bool unroll = rparams->isUnrolled();

  // Cache inputs if unrolled
  auto cached_inputs = scheduler_utils::cacheInputs(fusion, unroll);

  // Cache and fork outputs
  auto cached_outputs = scheduler_utils::cacheAndForkOutputs(fusion, unroll);

  // Make sure we don't have global memory set on intermediate tensors from
  // fusion segmentation
  scheduler_utils::clearMemorySpace(fusion);

  scheduler_utils::prepareForMemoryTypePromotion(fusion);

  auto reduction_tvs = scheduler_utils::getReductionTvs(fusion);

  NVF_ERROR(!reduction_tvs.empty());

  // Registry assumes the reference tv is the first reduction_tv, if this
  // changes registry needs to change.
  auto reduction_tv = reduction_tvs[0];

  if (!ir_utils::getReshapeOps(fusion).empty()) {
    ComputeAtMap ca_map(fusion);
    // Propagate reshape transforms through the graph, expecially the reference.
    scheduler_utils::propagateReshapeTransforms(fusion, ca_map);

    // Reorder reference_tv after propagating the view operation. This will
    // reorder for better merging.
    reduction_tv->reorder(
        scheduler_utils::domainReorderAsLogicalMap(reduction_tv));
  }

  NVF_ERROR(
      !(rparams->schedule_3D &&
        getShardedIterDomain(reduction_tv, ParallelType::DIDx) != nullptr),
      "Multidevice nvFuser does not support 3D reduction schedules");

  auto dim_analysis = scheduler_utils::canonicalDimReduction(
      fusion, reduction_tv, rparams->fastest_dim && rparams->schedule_3D);

  bool has_iter_axis = dim_analysis.first;
  bool has_red_axis = dim_analysis.second;

  NVF_ERROR(
      has_red_axis,
      "Could not find reduction axis in tensor used for reduction scheduler.");

  if (!has_iter_axis) {
    NVF_ERROR(
        rparams->fastest_dim,
        "If all dims are reduction, should be sending it to fastest dim "
        "scheduler.");
  }

  TensorView* reference_tv = reduction_scheduler_utils::scheduleReductionTV(
      rparams, reduction_tv, has_iter_axis);

  // Reduction tensor views and rfactor tensor views are setup. Let's finish off
  // the scheduling, particularly inlining and unrolling.
  NVF_ERROR(
      reference_tv != nullptr && reduction_tv != nullptr,
      "Need these two tensor views to finish the scheduling.");
  const bool is_vectorize =
      rparams->vectorize_inner_reduction || rparams->vectorize_iter_dom;

  // allow iter domain grouped reduction for block and grid outer reductions.
  // TODO: the var name is confusing, should rename
  // [cross_grid/block_inner_reduction] to [cross_grid/block_reduction], see
  // https://github.com/NVIDIA/Fuser/issues/1863
  // grouped welford is only enabled for grid persistent.
  // see validateAndConvertIterDomainGrouping
  const bool has_welford = ir_utils::hasOpsOfType<WelfordOp>(fusion);
  const bool use_iter_grouped_reduction = !rparams->fastest_dim &&
      (has_welford
           ? rparams->cross_grid_inner_reduction && rparams->persistent_kernel
           : rparams->cross_block_inner_reduction);

  scheduler_utils::moveNonConcretizedBroadcastInnermost(fusion, {reference_tv});

  // Propagate transformations before we rfactor the other reductions
  reduction_scheduler_utils::propagateTransformation(reference_tv);
  // If reduction_tv is rfactored, rfactor all reductions.
  if (reference_tv != reduction_tv) {
    reduction_scheduler_utils::propagateRFactor(
        reference_tv, reduction_tv, reduction_tvs);
  }

  const auto& unroll_vectorizable_cached_tvs =
      reduction_scheduler_utils::getCachedTvsToUnrollOrVectorize(
          reference_tv, is_vectorize, cached_inputs, cached_outputs);

  reduction_scheduler_utils::propagateParallelization(
      reduction_tv,
      reference_tv,
      unroll,
      use_iter_grouped_reduction,
      reduction_tvs,
      unroll_vectorizable_cached_tvs);

  // Inline the schedule
  inlineMost();

  scheduler_utils::promoteProducerMemoryTypes(fusion, cached_inputs);

  // TODO(#1401): We could let segmentation split a partially alias-producing
  // fusion into an alias-only segment and the rest. This way, the rest of the
  // fusion (which has fewer expressions) can potentially find a better
  // scheduler and we need to call markAliases only in NoOpScheduler.
  markAliases(fusion);
}

} // namespace

//! Check if the reduction heuristics apply in given fusion
bool ReductionScheduler::canScheduleCompileTime(Fusion* fusion) {
  FUSER_PERF_SCOPE("ReductionScheduler::canScheduleCompileTime");

  for (auto tv : fusion->allTvs()) {
    if (tv->dtype() != DataType::Index &&
        dataTypeSizeBit(tv->dtype()) % 8 != 0) {
      scheduler_debug_utils::canScheduleRejectReason(
          schedulerType(), "Does not support sub-byte data types.");
      return false;
    }
  }

  if (scheduler_utils::isResharding(fusion)) {
    scheduler_debug_utils::canScheduleRejectReason(
        schedulerType(), "Fusion is resharding.");
    return false;
  }

  // Needs at least one reduction to consider.
  if (!ir_utils::hasAnyReductionOps(fusion)) {
    scheduler_debug_utils::canScheduleRejectReason(
        schedulerType(), "No reduction op to schedule");
    return false;
  }

  if (ir_utils::filterByType<TensorView>(fusion->inputs()).empty()) {
    scheduler_debug_utils::canScheduleRejectReason(
        schedulerType(), "Scheduling not supported with no input");
    return false;
  }

  // Check that inputs of all select/gather-like ops are fusion inputs
  if (registry_utils::rejectScheduleForMemoryPromotion(
          fusion, schedulerType())) {
    return false;
  }

  auto reduction_tvs = scheduler_utils::getReductionTvs(fusion);

  if (reduction_tvs.empty()) {
    // Use pointwise logic
    return false;
  }

  // Reject when output IDs are not covered by reference tv. Assuming reduction
  // scheduler simply uses reduction_tvs[0] as the reference, if that changes,
  // this needs to be changed. see issue
  // https://github.com/NVIDIA/Fuser/issues/3811
  scheduler_tools::DomainMap domain_map(fusion);
  if (!domain_map.isValidReference(reduction_tvs[0], /*check_inputs=*/true)) {
    scheduler_debug_utils::canScheduleRejectReason(
        schedulerType(),
        "Output contains ID that's not scheduled by reference tv.");
    return false;
  }

  if (registry_utils::hasNonUniqueBcast(fusion)) {
    scheduler_debug_utils::canScheduleRejectReason(
        schedulerType(),
        "Broadcasting dimension might be broadcasting to multiple sizes.");
    return false;
  }

  if (!ir_utils::getReshapeOps(fusion).empty()) {
    ComputeAtMap ca_map(fusion);
    if (registry_utils::requiresForwardViewReplay(fusion, ca_map)) {
      scheduler_debug_utils::canScheduleRejectReason(
          schedulerType(), "Fusion requires view being reversible.");
      return false;
    }

    // Reduction scheduler simply uses reduction_tvs[0] as the reference, if
    // that changes, this needs to be changed.
    if (registry_utils::reductionInterferingView(
            fusion, ca_map, reduction_tvs[0])) {
      scheduler_debug_utils::canScheduleRejectReason(
          schedulerType(), "View may interfere with reduction scheduling.");
      return false;
    }
  }

  // Make sure reduction axes are consistent through the fusion
  auto reduction_ops = ir_utils::getAllTypesOfReductionOps(fusion);
  if (reduction_ops.size() > 1) {
    // Before examining the reduction axes want to quickly
    //   check the reductions have the same axis width
    //   to avoid building root domain map in easier cases
    bool valid_axis_count = false;
    size_t axis_count = 0;
    auto reduction_root_size = [](TensorView* red_tv) {
      size_t count = 0;
      for (auto id : red_tv->getMaybeRootDomain()) {
        if (!id->isBroadcast()) {
          count++;
        }
      }
      return count;
    };

    for (auto red : reduction_tvs) {
      if (!valid_axis_count) {
        valid_axis_count = true;
        axis_count = reduction_root_size(red);
      } else {
        if (reduction_root_size(red) != axis_count) {
          scheduler_debug_utils::canScheduleRejectReason(
              schedulerType(),
              "Inconsistent reduction root size: ",
              red->toString(),
              ", expected: ",
              axis_count);
          return false;
        }
      }
    }

    // Use root domain map to check the reduction ops have the same axes
    FusionGuard fg(fusion);
    ComputeAtLogicalDomainMap logical_map;
    logical_map.build(true);

    // red_ops.size()>1 checked before
    for (size_t it = 1; it < reduction_tvs.size(); it++) {
      if (!registry_utils::checkPatternEquivalence(
              reduction_tvs[it - 1], reduction_tvs[it], logical_map)) {
        scheduler_debug_utils::canScheduleRejectReason(
            schedulerType(),
            "Un-mapped multi-reduction: ",
            reduction_tvs[it - 1]->toString(),
            " and ",
            reduction_tvs[it]->toString());
        return false;
      }
    }
  }

  // Doesn't allow persistent kernels in this scheduler
  auto persistent_buffer_info = scheduler_utils::persistentBuffers(fusion);
  if (!persistent_buffer_info.persistent_buffers.empty()) {
    scheduler_debug_utils::canScheduleRejectReason(
        schedulerType(),
        "need persistent buffers that reduction scheduler doesn't handle");
    return false;
  }

  if (!registry_utils::SchedulerTopologyChecker::supportedPostReductionFusion(
          fusion, reduction_tvs) ||
      registry_utils::SchedulerTopologyChecker::hasPostReductionBCast(fusion)) {
    scheduler_debug_utils::canScheduleRejectReason(
        schedulerType(), "has unsupported post reduction fusion");
    return false;
  }

  if (registry_utils::SchedulerTopologyChecker::
          hasGatherToBroadcastBeforeReduction(fusion, reduction_tvs)) {
    scheduler_debug_utils::canScheduleRejectReason(
        schedulerType(), "has unsupported gather-like ops before reduction");
    return false;
  }

  return true;
}

bool ReductionScheduler::canScheduleRunTime(
    Fusion* fusion,
    SchedulerRuntimeInfo& runtime_info,
    HeuristicDataCache* data_cache) {
  FUSER_PERF_SCOPE("ReductionScheduler::canScheduleRunTime");
  return true;
}

std::unique_ptr<HeuristicParams> ReductionScheduler::computeHeuristics(
    Fusion* fusion,
    SchedulerRuntimeInfo& runtime_info,
    HeuristicDataCache* data_cache) {
  FUSER_PERF_SCOPE("ReductionScheduler::computeHeuristics");
  auto rparams = getReductionHeuristics(fusion, runtime_info, data_cache);
  NVF_ERROR(rparams != nullptr);
  return rparams;
}

void ReductionScheduler::schedule(
    Fusion* fusion,
    const HeuristicParams* params) {
  FUSER_PERF_SCOPE("ReductionScheduler::schedule");
  auto rparams = dynamic_cast<const ReductionParams*>(params);
  NVF_ERROR(
      rparams != nullptr,
      "Incorrect parameters sent to ReductionScheduler::schedule",
      params);
  scheduleReduction(fusion, rparams);
}
} // namespace nvfuser
