// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#pragma once
#include <ir/interface_nodes.h>
#include <scheduler/heuristic.h>

#include <sstream>

namespace nvfuser {

class TensorView;

// Parameters of the reduction heuristic to describe the optimal schedule.
// Warning: equal operator is intended for use in caching the kernel associated
// with these reduction parameters. It does not check if the launch parameters
// are equivelent!
class ReductionParams : public HeuristicParams {
 public:
  // Note that heuristictype can be different from SchedulerType::Reduction
  // since ReductionParams is also used by, e.g., normalization schedulers.
  ReductionParams(SchedulerType scheduler_type = SchedulerType::Reduction)
      : HeuristicParams(scheduler_type) {};
  // Reducing inner most dimension?
  bool fastest_dim = false;

  // Store input in shared memory or registers to reduce global memory reads
  bool persistent_kernel = false;

  // Project persistent buffers back to inputs to reduce persistent buffer size
  bool project_persistent_buffers = false;

  // Are we treating the scheduling as 3 dimensional, can be useful for patterns
  // like [reduction, iteration, reduction].
  bool schedule_3D = false;

  // For outer reductions we may want to swap the gdimx and gdimy bindings to
  // amortize the cost of the final cleanup in grid reductions.
  bool flip_grid = false;

  // Inner Reduction Domain:

  // Reduce across the block?
  bool cross_block_inner_reduction = false;
  // Reduce across the grid?
  bool cross_grid_inner_reduction = false;
  // Unrolling/Vectorization factor for inner reduction dimension
  int64_t unroll_factor_inner_reduction = 1;

  // Extra unroll on top of vectorization
  int64_t unroll_factor_top_of_vectorization = 1;

  // vectorize instead of unroll
  bool vectorize_inner_reduction = false;
  // Split grid dim for iteration axis in case it's too large for cuda
  bool split_grid_dim_inner_reduction = false;
  // Pad inner dimension to nearest warp
  bool pad_inner_reduction_to_warp = false;
  // Register persistent buffer size in inner dimension
  int64_t batches_per_block_inner_reduction = 1;

  // Which block parallel dimension should be used for the inner reduction.
  // !!WARNING!! Convenience method, this be unique based on non-parallel type
  // parameters, not used for equivalence/hashing.
  ParallelType block_dim_inner_reduction = ParallelType::Serial;
  // Which grid parallel dimension should be used for the inner reduction.
  // !!WARNING!! Convenience method, this be unique based on non-parallel type
  // parameters, not used for equivalence/hashing.
  ParallelType grid_dim_inner_reduction = ParallelType::Serial;

  // Iteration Domain:

  // Perform multiple reductions per block?
  bool multiple_reds_per_blk = false;
  // Unrolling/Vectorization factor for iteration dimension
  int64_t unroll_factor_iter_dom = 1;
  // vectorize instead of unroll
  bool vectorize_iter_dom = false;
  // Inner split grid dim for iteration axis in case it's too large for cuda
  bool split_grid_dim_iter_dom_inner = false;
  // Outer split grid dim for iteration axis in case it's too large for cuda
  bool split_grid_dim_iter_dom_outer = false;

  // Which block parallel dimension should be used for the iter domain.
  // !!WARNING!! Convenience method, this be unique based on non-parallel type
  // parameters, not used for equivalence/hashing.
  ParallelType block_dim_iter_dom = ParallelType::Serial;
  // Which grid parallel dimension should be used for the iter domain.
  // !!WARNING!! Convenience method, this be unique based on non-parallel type
  // parameters, not used for equivalence/hashing.
  ParallelType grid_dim_iter_dom = ParallelType::Serial;

  // Outer Reduction Domain if 3D Scheduled:

  // Reduce across the block?
  bool cross_block_outer_reduction = false;
  // Reduce across the grid?
  bool cross_grid_outer_reduction = false;
  // Split grid dim for iteration axis in case it's too large for cuda
  bool split_grid_dim_outer_reduction = false;
  // Register persistent buffer size in outer dimension
  int64_t batches_per_block_outer_reduction = 1;
  // Unrolling/Vectorization factor for outer reduction factor
  int64_t unroll_factor_outer_reduction = 1;

  // Which block parallel dimension should be used for the outer reduction.
  // !!WARNING!! Convenience method, this be unique based on non-parallel type
  // parameters, not used for equivalence/hashing.
  ParallelType block_dim_outer_reduction = ParallelType::Serial;
  // Which grid parallel dimension should be used for the outer reduction.
  // !!WARNING!! Convenience method, this be unique based on non-parallel type
  // parameters, not used for equivalence/hashing.
  ParallelType grid_dim_outer_reduction = ParallelType::Serial;

  // Use computeWith to persistent buffers
  bool compute_persistent_buffer_with_first_consumer = false;

  bool static_bdimx = false;
  bool static_bdimy = false;

  bool isUnrolled() const {
    return unroll_factor_inner_reduction > 1 || unroll_factor_iter_dom > 1 ||
        unroll_factor_outer_reduction > 1;
  }

  // specific to combined inner and outer reduction
  bool combined_inner_outer = false;
  // use TIDx for out reduction axis
  bool tidx_for_outer_reduction = false;
  // pad outer reduction to warp
  bool pad_outer_reduction_to_warp = false;
  // in outer reduction part of inner-outer persistent scheduler, may further
  // split inner dim by grid
  bool combined_split_grid_inner_dim = false;

  // TMA warp specialized, only used in inner-outer persistent scheduler
  bool tma_warp_specialized = false;

  // Circular buffer used in tma warp specialized normalization
  CircularBufferOptions circular_buffer_options;

  // partial result of outer reduction is written to gmem then read back in a
  // different parallel pattern set the vectorization factor of its read and
  // write
  int64_t vectorization_factor_outer = 1;
  int64_t vectorization_factor_tmp_gmem_write = 1;
  // inner reduction axis is parallelized by block_dim_inner_reduction (usually
  // TIDx) the remaining part is further parallelized by
  // block_dim_inner_reduction_extra (usually TIDy)
  ParallelType block_dim_inner_reduction_extra = ParallelType::Serial;

  // vector stores buffer should be moved to shared memory.
  // TODO: For innerOuterPersistentHeuristic, only the persistent tensors in the
  // original fusion definition may be moved to shared memory, the intermediate
  // persistent tensors which are creased by the scheduler to store the partial
  // outer reduction results are always stored in registers because they are
  // frequently accessed by both read and write. The code can be extended to
  // allow the move of these intermediate persistent tensors to shared memory
  // when the shared memory is much larger than the register file.
  std::vector<TensorView*> smem_persistent_buffers;

 public:
  using HeuristicParams::HeuristicParams;

  // Warning: Does not check launch parameters!
  bool sameAs(const HeuristicParams* other_base) const override {
    auto other = dynamic_cast<const ReductionParams*>(other_base);
    if (other == nullptr) {
      return false;
    }

    bool attr_equal = other->cparams == cparams &&
        other->fastest_dim == fastest_dim &&
        other->persistent_kernel == persistent_kernel &&
        other->project_persistent_buffers == project_persistent_buffers &&
        other->schedule_3D == schedule_3D && other->flip_grid == flip_grid &&
        other->cross_block_inner_reduction == cross_block_inner_reduction &&
        other->cross_grid_inner_reduction == cross_grid_inner_reduction &&
        other->unroll_factor_inner_reduction == unroll_factor_inner_reduction &&
        other->vectorize_inner_reduction == vectorize_inner_reduction &&
        other->split_grid_dim_inner_reduction ==
            split_grid_dim_inner_reduction &&
        other->pad_inner_reduction_to_warp == pad_inner_reduction_to_warp &&
        other->batches_per_block_inner_reduction ==
            batches_per_block_inner_reduction &&
        other->multiple_reds_per_blk == multiple_reds_per_blk &&
        other->unroll_factor_iter_dom == unroll_factor_iter_dom &&
        other->vectorize_iter_dom == vectorize_iter_dom &&
        other->split_grid_dim_iter_dom_inner == split_grid_dim_iter_dom_inner &&
        other->split_grid_dim_iter_dom_outer == split_grid_dim_iter_dom_outer &&
        other->cross_block_outer_reduction == cross_block_outer_reduction &&
        other->cross_grid_outer_reduction == cross_grid_outer_reduction &&
        other->unroll_factor_outer_reduction == unroll_factor_outer_reduction &&
        other->split_grid_dim_outer_reduction ==
            split_grid_dim_outer_reduction &&
        other->batches_per_block_outer_reduction ==
            batches_per_block_outer_reduction &&
        other->compute_persistent_buffer_with_first_consumer ==
            compute_persistent_buffer_with_first_consumer &&
        other->combined_inner_outer == combined_inner_outer &&
        other->tidx_for_outer_reduction == tidx_for_outer_reduction &&
        other->pad_outer_reduction_to_warp == pad_outer_reduction_to_warp &&
        other->vectorization_factor_outer == vectorization_factor_outer &&
        other->combined_split_grid_inner_dim == combined_split_grid_inner_dim &&
        other->unroll_factor_top_of_vectorization ==
            unroll_factor_top_of_vectorization &&
        other->vectorization_factor_tmp_gmem_write ==
            vectorization_factor_tmp_gmem_write;

    if (other->static_bdimy || static_bdimy) {
      attr_equal = attr_equal && other->lparams.bdimy() == lparams.bdimy();
    }
    if (other->static_bdimx || static_bdimx) {
      attr_equal = attr_equal && other->lparams.bdimx() == lparams.bdimx();
    }
    return attr_equal;
  }

  std::string toString() const override {
    std::stringstream ss;
    ss << "\n===== Reduction Parameters ========\n"
       << (tag.empty() ? "" : "Tag: ") << tag << "\n"
       << (fastest_dim ? "Red On Fastest Dim\n" : "Red On Slow Dim\n")
       << (persistent_kernel ? "Persistent Kernel\n" : "")
       << (project_persistent_buffers ? "Project Persistent Buffers\n" : "");
    if (batches_per_block_inner_reduction > 1 || persistent_kernel) {
      ss << "Batches per block: " << batches_per_block_inner_reduction << "\n";
    }
    if (circular_buffer_options.isEnable()) {
      ss << circular_buffer_options;
    } else {
      ss << "Circular buffer: not used\n";
    }
    if (schedule_3D) {
      ss << "3D Schedule\n"
         << "Outer Reduction: ";
      if (cross_block_outer_reduction) {
        ss << "cross block - " << block_dim_outer_reduction << " / ";
      }
      if (cross_grid_outer_reduction) {
        ss << "cross grid - " << grid_dim_outer_reduction << " / ";
        ss << (split_grid_dim_outer_reduction ? "split grid dim / " : "");
      }

      ss << (unroll_factor_outer_reduction > 1 ? "unroll / " : "");
      if (unroll_factor_outer_reduction > 1) {
        ss << "factor " << unroll_factor_outer_reduction << " ";
      }

      if (batches_per_block_outer_reduction > 1 || persistent_kernel) {
        ss << "persistent batch - " << batches_per_block_outer_reduction;
      }
    }

    ss << "\nIteration Domain: ";

    if (grid_dim_iter_dom != ParallelType::Serial) {
      ss << grid_dim_iter_dom << " / ";
      if (split_grid_dim_iter_dom_outer) {
        ss << "split grid dimension outer / ";
      } else if (split_grid_dim_iter_dom_inner) {
        ss << "split grid dimension inner / ";
      }
    }
    if (block_dim_iter_dom != ParallelType::Serial) {
      ss << block_dim_iter_dom << " / ";
    }
    ss << (multiple_reds_per_blk ? "multiple reductions per block / " : "")
       << (vectorize_iter_dom ? "vectorize / " : "")
       << (unroll_factor_iter_dom > 1 && !vectorize_iter_dom ? "unroll / "
                                                             : "");
    if (unroll_factor_iter_dom > 1) {
      ss << "factor " << unroll_factor_iter_dom;
    }

    ss << "\nInner Reduction Domain: ";

    if (cross_block_inner_reduction) {
      ss << "cross block - " << block_dim_inner_reduction << " / ";
      ss << (pad_inner_reduction_to_warp ? " pad to warp / " : "");
    }
    if (cross_grid_inner_reduction) {
      ss << "cross grid - " << grid_dim_inner_reduction << " / ";
      ss << (split_grid_dim_inner_reduction ? "split grid dim / " : "");
    }
    if (batches_per_block_inner_reduction > 1 || persistent_kernel) {
      ss << "persistent batch - " << batches_per_block_inner_reduction << " / ";
    }
    ss << (cross_grid_inner_reduction && split_grid_dim_inner_reduction
               ? "split grid dimension / "
               : "")
       << (vectorize_inner_reduction ? "vectorize / " : "")
       << (unroll_factor_inner_reduction > 1 && !vectorize_inner_reduction
               ? "unroll / "
               : "");
    if (unroll_factor_inner_reduction > 1) {
      ss << "factor " << unroll_factor_inner_reduction;
    }

    if (compute_persistent_buffer_with_first_consumer) {
      ss << "\ncomputeWith persistent buffers";
    }

    ss << "\n" << lparams.toString();
    ss << cparams.toString() << "\n";
    ss << "====================================\n";
    return ss.str();
  }

  // Warning: Hash is not based on launch parameters!
  size_t hash() const override {
    constexpr size_t bits = sizeof(std::size_t) * 8;
    size_t attr_hash = static_cast<size_t>(fastest_dim) << (bits - 1) ^
        static_cast<size_t>(persistent_kernel) << (bits - 2) ^
        static_cast<size_t>(project_persistent_buffers) << (bits - 3) ^
        static_cast<size_t>(schedule_3D) << (bits - 4) ^
        static_cast<size_t>(flip_grid) << (bits - 5) ^
        static_cast<size_t>(cross_block_inner_reduction) << (bits - 6) ^
        static_cast<size_t>(cross_grid_inner_reduction) << (bits - 7) ^
        static_cast<size_t>(unroll_factor_inner_reduction) << (bits - 8) ^
        static_cast<size_t>(vectorize_inner_reduction) << (bits - 9) ^
        static_cast<size_t>(split_grid_dim_inner_reduction) << (bits - 10) ^
        static_cast<size_t>(pad_inner_reduction_to_warp) << (bits - 11) ^
        static_cast<size_t>(batches_per_block_inner_reduction) << (bits - 12) ^
        static_cast<size_t>(multiple_reds_per_blk) << (bits - 13) ^
        static_cast<size_t>(unroll_factor_iter_dom) << (bits - 14) ^
        static_cast<size_t>(vectorize_iter_dom) << (bits - 15) ^
        static_cast<size_t>(split_grid_dim_iter_dom_outer) << (bits - 16) ^
        static_cast<size_t>(split_grid_dim_iter_dom_inner) << (bits - 17) ^
        static_cast<size_t>(cross_block_outer_reduction) << (bits - 18) ^
        static_cast<size_t>(cross_grid_outer_reduction) << (bits - 19) ^
        static_cast<size_t>(split_grid_dim_outer_reduction) << (bits - 20) ^
        static_cast<size_t>(batches_per_block_outer_reduction) << (bits - 21) ^
        static_cast<size_t>(unroll_factor_outer_reduction) << (bits - 22) ^
        static_cast<size_t>(compute_persistent_buffer_with_first_consumer)
            << (bits - 23) ^
        static_cast<size_t>(unroll_factor_top_of_vectorization) << (bits - 24);
    return attr_hash;
  }

  std::unique_ptr<HeuristicParams> clone() const override {
    return std::make_unique<ReductionParams>(*this);
  }
};

} // namespace nvfuser
