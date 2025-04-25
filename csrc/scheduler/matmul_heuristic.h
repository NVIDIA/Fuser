// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#pragma once

#include <c10/util/hash.h>
#include <mma_type.h>
#include <scheduler/heuristic.h>
#include <utils.h>
#include <functional>

#include <sstream>
#include "type.h"

namespace nvfuser {

// Parameters of the matmul heuristic to describe the optimial schedule.
class MatmulParams : public HeuristicParams {
 public:
  MatmulParams()
      : HeuristicParams(SchedulerType::Matmul), supported_vec_size() {};
  //! A list of possible strategies used to define along which axis
  //!  parallelization will be done.
  enum class TileRasterizationOrder { RowMajor = 0, ColumnMajor = 1 };

  //! A wrapper for circular buffering config pieces
  struct CircularBufferOptions {
    bool circular_buffer_smem_write = false;
    bool circular_buffer_smem_read = false;
    // This parameter controls the number of circular
    // buffering stages to use when loading operands a and b.
    //
    // If this value is greater than two then it indicates circular buffering,
    // in which case async_gmem_load_operands must also be true.
    //
    // Note that whenever circular_buffer_smem_write is true, this value must be
    // greater than one. Otherwise it is ignored.
    int smem_circular_buffer_stage = 2;

    // The circular buffering prefetch distance will be set to
    //   smem_circular_buffer_stage - smem_circular_buffer_prefetch_gap
    // This value must be positive since the prefetch distance must be strictly
    // less than the number of stages.
    int smem_circular_buffer_prefetch_gap = 1;

    bool operator==(const CircularBufferOptions& other) const {
      return other.circular_buffer_smem_write == circular_buffer_smem_write &&
          other.circular_buffer_smem_read == circular_buffer_smem_read &&
          other.smem_circular_buffer_stage == smem_circular_buffer_stage &&
          other.smem_circular_buffer_prefetch_gap ==
          smem_circular_buffer_prefetch_gap;
    }

    std::string toString() const {
      std::stringstream ss;
      ss << "CircularBufferOptions:\n"
         << "  circular_buffer_smem_write: "
         << (circular_buffer_smem_write ? "true" : "false") << "\n"
         << "  circular_buffer_smem_read: "
         << (circular_buffer_smem_read ? "true" : "false") << "\n"
         << "  smem_circular_buffer_stage: " << smem_circular_buffer_stage
         << "\n"
         << "  smem_circular_buffer_prefetch_gap: "
         << smem_circular_buffer_prefetch_gap;
      return ss.str();
    }

    size_t hash() const {
      return std::hash<size_t>{}(
                 (static_cast<size_t>(smem_circular_buffer_prefetch_gap) << 3) |
                 (static_cast<size_t>(smem_circular_buffer_stage) << 2) |
                 (static_cast<size_t>(circular_buffer_smem_write)) << 1) |
          (static_cast<size_t>(circular_buffer_smem_read));
    }
  };

  //! This is the maximum vectorization supported by the inputs and outputs.
  //! This refers to the number of data elements loaded simultaneously, not the
  //! number of bytes.
  struct SupportedVectorization {
    // Each operand load from global to shared memory is vectorized along its
    // inner-most allocation dimension as long as that is an M, N, or K
    // dimension. For example, if the innermost dimension is a batch dimension
    // then we will not vectorize that operand's loads from global to shared
    // memory. If there are multiple dimensions in a given role, such as
    // multiple K dimensions, then we can only vectorize those inner dimensions
    // that are consistent with the canonical dimension ordering shared by all
    // tensors in the Fusion.
    int64_t a;
    int64_t b;

    // The epilogue is handled in a separate loop from the main loop/operand
    // loads. We inline the epilogue expressions as much as possible, and we
    // vectorize all tensors with the same factor for better memory coalescence;
    // i.e. we parallelize the epilogue like [ ... TIDx V ] so we do not
    // introduce any loops between the TIDx and V dimensions. If we used
    // different vectorization for each output or epilogue input, then we would
    // need an unrolled loop between TIDx and V which would interfere with
    // memory coalescence. We assume the decrease in indexing arithmetic from
    // vectorization is not worth the slowdown from non-coalesced accesses, so
    // we prefer to use a smaller vectorization instead.
    //
    // To determine the epilogue vectorization we do the following steps:
    //  - Look at each output, then each epilogue input and find the first
    //    tensor with a non-batch dimension as its innermost allocation
    //    dimension. We will use that as the innermost loop dimension and will
    //    vectorize that dimension. If there are multiple such innermost
    //    dimensions with the same role and full contiguity then we consider all
    //    those dimensions as the merged vectorized dimension. For example if
    //    we have an output whose allocation domain is [ B1 M1 N1 M2 M3 ] then
    //    (M2*M3) will be the vectorized dimension. On the other hand, we would
    //    skip a tensor that had allocation domain [ M1 M2 M3 N1 B1 ] since the
    //    batch dimension is innermost.
    //  - Then we pass over all epilogue inputs and outputs. For each tensor, we
    //    consider all innermost dimensions in order. For example if we have
    //    determined that we will vectorize along M1*M2*M3 and a tensor has
    //    allocation [ B1 M1 N1 M2 M3 ] then we consider dimension M2*M3 (along
    //    with all other strides) to find supported vectorization. If another
    //    tensor has allocation [ B1 M1 M2 M3 N1 ] then we skip it since its
    //    innermost dimension is not an N role dimension so its access will not
    //    be vectorized.
    //  - We store the minimum of all the maximum supported vectorizations
    //    across all epilogue input and output tensors that were not skipped.
    //    That is the value below. If no vectorization is possible, this will be
    //    set to 1.
    int64_t epilogue;

    bool operator==(const SupportedVectorization& other) const {
      return other.a == a && other.b == b && other.epilogue == epilogue;
    }

    std::string toString() const {
      std::stringstream ss;
      ss << "SupportedVectorization:\n"
         << "  a: " << a << "\n"
         << "  b: " << b << "\n"
         << "  epilogue: " << epilogue;
      return ss.str();
    }

    size_t hash() const {
      return std::hash<size_t>{}(
                 (static_cast<size_t>(a) << 8) |
                 (static_cast<size_t>(b)) << 4) |
          (static_cast<size_t>(epilogue));
    }
  } supported_vec_size;

  //! (Ampere+) Use cp.async to load operands.
  bool async_gmem_load_operands = false;

  //! Specifies the tiling hierarchy on block and warp levels.
  MatMulTileOptions tile_sizes = {};

  //! Specify the type of MMA op to be used in generated kernel.
  MmaMacro mma_macro = MmaMacro::NoMMA;

  // [Basic Matmul Configuration]
  // We compute matrix products by decomposing the output into tiles. The size
  // of these tiles is specified by the M and N dimensions of the CTA tile. The
  // K dimension of the CTA tile must equal that of the warp tile, and will be
  // discussed later.
  //
  // Normally, each output tile is processed by a single CTA (for exceptions,
  // see notes about split-K and stream-K below). That CTA contains threads
  // organized into warps of 32 threads and (on Hopper+) warpgroups consisting
  // of 4 warps. The warp tile's M and N dimensions indicate the subtile of the
  // CTA tile that each warp or warpgroup is responsible for computing.
  //
  // The K dimension of the warp tile, which must match that of the CTA tile,
  // indicates the K dimension of the operand tiles that are processed in the
  // "K loop", a serial loop in the generated kernel used to accumulate
  // contributions to the mma result via summation in a register buffer in each
  // thread.
  //
  // The MmaMacro determines the actual PTX instruction used to compute a small
  // matrix-matrix product on the device's tensor cores. These macros determine
  // an "instruction tile" which can be computed in a single instruction. The
  // number of instruction tiles that make up a single warp tile translate to
  // loops in the generated kernel inside of the K loop, allowing each thread
  // to compute a warp tile result that is larger than the specific
  // instruction. Importantly, the warp tile determines the amount of data that
  // must be loaded before performing the loop to issue mma instructions, so the
  // warp tile provides a lower bound on the size of each loading or circular
  // buffering stage.
  //
  // [Detailed Matmul Configuration]
  // One simple way to compute the output tiles is to assign each CTA tile to
  // an individual CTA, launching a 2D grid that matches the tiling of the
  // output matrix. Each of those CTAs can then compute a single K loop,
  // loading one CTA tile at each iteration, in order to accumulate the result
  // for a single output tile. This might require multiple waves of CTAs to be
  // launched, and each one will need to compute a prologue consisting of some
  // indexing expressions. Furthermore, the epilogue computation must complete
  // before each SM can launch the next CTA to which it is assigned.
  //
  // Alternatively, we could launch exactly one CTA per SM on the device. This
  // allows us to compute some of the prologue once, then loop over a set of
  // output tiles. For each output tile we then compute a K loop and epilogue.
  // However, along with warp specialization and other approaches, we can
  // sometimes begin loading data for the next tile before the epilogue is
  // complete (see below). We call such an approach a "persistent kernel".
  //
  // Within each iteration of the K loop, two distinct things need to happen.
  // First, we need to load data from the operands to the SM in either shared
  // memory or registers. Then we need to perform a set of mma instructions to
  // compute the contribution of a warp tile to the final result. Waiting for
  // the data to load before computing the mma instructions would mean leaving
  // the tensor cores idle, hurting performance. Instead, we commonly employ
  // circular buffering, wherein at each iteration of the K loop we launch an
  // asynchronous load of data for a future iteration. This way each thread
  // only needs to launch an asynchronous load, then wait for a previous load
  // to complete before computing mma instructions. This is called the
  // "pipelined" strategy wherein we leave a number of asynchronous transfers
  // in flight at all points of the K loop.
  //
  // The load instructions inside each K loop iteration can also be avoided by
  // moving them to a separate thread. This is done via "warp specialization":
  // we launch one additional warp group called the "dma warp group" whose only
  // responsibility is to monitor the circular buffer and issue asynchronous
  // load instructions. The mma instructions are left to the other warp groups,
  // which we call "math warp groups".
  //
  // [Split-K and Stream-K]
  // When the M, N are much smaller than the K dimension, distributing separate
  // output tiles across the grid will not fully-occupy all compute resources on
  // the GPU. An alternative is to parallelize work along the K dimension and
  // then have a single CTA aggregate results for an output tile.
  //
  // Split-K divides the K dimension by constant factor. For example, when the
  // split-k factor is 4, the k dimension is split across 4 CTAs. Each CTA
  // accumulates a (CTA-M, CTA-N, K/4) output tile. A grid reductions is then
  // performed on the K dimension to get the complete (CTA-M, CTA-N) tile. When
  // the split-k factor is 1, it is equivalent to the data-parallel approach.
  //
  // The Steam-K approach combines the persistent grid strategy, which launches
  // a single wave of CTAs, and k dimension parallelization. The core idea is to
  // have each SM complete a fixed unit of work per stage, utilizing M, N, and K
  // dimension parallelization. Each CTA computes a fixed (CTA-M, CTA-N, CTA-K)
  // tile per stage. CTA-K dimension may split across multiple (CTA-M, CTA-N)
  // output tiles. Once all partial tiles are completed, a grid sum accumulates
  // all partial tiles. The advantage of stream-k over split-k is finding the
  // optimal split-k factor to avoid wave quantization is non-trivial.
  //
  // When (CTA-K == K), then stream-k is equivalent to the persistent
  // data-parallel strategy. When K dimension is evenly divided among CTAs (K %
  // CTA-K == 0), then stream-k is equivalent to persistent split-k strategy.

  //! Specify whether to use a 1-1 mapping from output tile to CTA or to launch
  //! one CTA per SM then loop over a subset of output tiles within the kernel
  //! (persistent).
  enum class TilingStrategy {
    OneTilePerCTA, // Map each output tile to a single CTA and launch as many as
                   // are needed to cover the tile grid. This is also commonly
                   // referred to as the (data-parallel) strategy.
    DistributeTilesAcrossSMs, // Use persistent kernels to compute entire output
                              // tiles
    DistributeStagesAcrossSMs // Use persistent kernels to compute whole and
                              // partial output tiles (stream-K)
  } tiling_strategy = TilingStrategy::OneTilePerCTA;

  //! Configure circular buffering loops
  enum class BufferingLoopLevel {
    CTATiles, // Warp groups cooperatively compute whole CTA tiles in each
              // K iteration. If splitk_factor > 1, all math warp groups
              // cooperate, but only for a portion of the whole K loop.
              // splitk_factor > 1 requires a grid reduction to combine the
              // contributions from each portion. Also called split-K.
    WarpTiles // All warp tiles in a K loop for each math warp group are
              // iterated over then the next math warp group's warp tile is
              // processed. Also called ping-pong or alternating stratgy.
  } buffering_loop_level = BufferingLoopLevel::CTATiles;

  //! Whether to do regular circular buffering (pipelined) or warp
  //! specialization using an additional dma warp group
  enum class CircularBufferingStrategy {
    Pipelined,
    WarpSpecialized
  } circular_buffering_strategy = CircularBufferingStrategy::Pipelined;

  //! Specify CTA rastrization order.
  TileRasterizationOrder cta_order = TileRasterizationOrder::RowMajor;

  //! Specify which tensor we circular buffer.
  CircularBufferOptions circular_buffer_options = {};

  //! Swizzle factor is used to increase L2 hit rate.
  //!  It horizontally squeezes the grid so that gridDim.x is larger and
  //!  gridDim.y is smaller.
  //!  We rely on the observation that the CTAs are scheduled by the GPU by
  //!  iterating on gridDim.x first. As a result, as blocks are launched, they
  //!  will more likely be forming sub-tiles of the C matrix. This will increase
  //!  L2 hit rate/data reuse of A and B.
  //!
  //! Eg for grid_traversal_factor = {2, 1}:
  //!    A1 A2 B1 B2 -->   A1 A2 A3 A4 B1 B2 B3 B4
  //!    A3 A4 B3 B4       C1 C2 C3 C4 D1 D2 D3 D4
  //!    C1 C2 D1 D2
  //!    C3 C4 D3 D4
  std::pair<int, int> grid_traversal_factor = {1, 1};

  //! Unswizzle MMA results in shared memory to get
  //!  coalesced write to global memory
  bool use_smem_epilogue = false;

  //! Promote reuse of prologue shared memory
  bool promote_prologue_smem_reuse = false;

  //! If use_smem_epilogue==false, this has no effect. Otherwise, it enables
  //! storing the mma result to shared memory using stmatrix and loading
  //! epilogue inputs to registers using ldmatrix instructions. Note that
  //! stmatrix nor ldmatrix are never used on TensorViews whose dtype is fp16
  //! or bf16.
  bool use_ldst_matrix = true;

  //! Whether to do single-kernel split-K. If this is >1, we will rfactor the K
  //! axis and perform a grid reduction before the epilogue.
  int splitk_factor = 1;

  //! This is the CGA size on Hopper+ devices. This parameter is ignored on
  //! Ampere and Turing.
  struct ClusterDims {
    int64_t x = 1;
    int64_t y = 1;
    int64_t z = 1;

    bool operator==(const ClusterDims& other) const {
      return x == other.x && y == other.y && z == other.z;
    }

    bool operator!=(const ClusterDims& other) const {
      return !(*this == other);
    }

    std::string toString() const {
      std::stringstream ss;
      ss << "__cluster_dims__(" << x << ", " << y << ", " << z << ")";
      return ss.str();
    }

    size_t hash() const {
      return std::hash<size_t>{}(
                 (static_cast<size_t>(x) << 32) |
                 (static_cast<size_t>(y)) << 16) |
          (static_cast<size_t>(z));
    }
  } cluster_dims;

  std::string toString() const override {
    std::stringstream ss;
    ss << "\n===== Matmul Parameters ========\n"
       << (tag.empty() ? "" : "Tag: ") << tag << "\n"
       << "MMA macro: " << nvfuser::toString(mma_macro) << "\n"
       << circular_buffer_options.toString() << "\n"
       << supported_vec_size.toString() << "\n"
       << nvfuser::toString(tile_sizes) << "\n"
       << "Async global mem load: "
       << (async_gmem_load_operands ? "true" : "false") << "\n"
       << "Indexing mode: "
       << (cparams.index_type.has_value()
               ? (cparams.index_type.value() == PrimDataType::Int ? "int64_t"
                                                                  : "int32_t")
               : "unavailable")
       << "\n"
       << "Tile rasterization order: "
       << ((cta_order == TileRasterizationOrder::RowMajor) ? "row-major"
                                                           : "column-major")
       << "\n"
       << "Grid swizzle factor: " << grid_traversal_factor << "\n";
    ss << "Tiling strategy: ";
    switch (tiling_strategy) {
      case TilingStrategy::OneTilePerCTA:
        ss << "OneTilePerCTA";
        break;
      case TilingStrategy::DistributeTilesAcrossSMs:
        ss << "DistributeTilesAcrossSMs";
        break;
      case TilingStrategy::DistributeStagesAcrossSMs:
        ss << "DistributeStagesAcrossSMs";
        break;
    }
    ss << "\n";
    ss << "Buffering loop level: ";
    switch (buffering_loop_level) {
      case BufferingLoopLevel::CTATiles:
        ss << "CTATiles";
        break;
      case BufferingLoopLevel::WarpTiles:
        ss << "WarpTiles";
        break;
    }
    ss << "\n";
    ss << "Circular buffering strategy: ";
    switch (circular_buffering_strategy) {
      case CircularBufferingStrategy::Pipelined:
        ss << "Pipelined";
        break;
      case CircularBufferingStrategy::WarpSpecialized:
        ss << "WarpSpecialized";
        break;
    }
    ss << "\n";
    ss << cluster_dims.toString() << "\n"
       << "Use shared memory epilogue: " << use_smem_epilogue << "\n"
       << "Promote re-use of prologue shared memory: "
       << promote_prologue_smem_reuse << "\n"
       << "Split-K factor: " << splitk_factor << "\n"
       << "====================================\n";
    return ss.str();
  }

  size_t hash() const override {
    // combine boolean flags for hashing
    size_t attr_hash = (static_cast<size_t>(promote_prologue_smem_reuse) << 2) |
        (static_cast<size_t>(use_smem_epilogue) << 1) |
        (static_cast<size_t>(async_gmem_load_operands));

    // combined hash
    attr_hash = std::hash<size_t>{}(attr_hash) ^
        (nvfuser::hash(mma_macro) << 1) ^
        (circular_buffer_options.hash() << 2) ^
        (nvfuser::hash(tile_sizes) << 3) ^
        (std::hash<size_t>{}(static_cast<size_t>(cta_order)) << 4) ^
        (std::hash<size_t>{}(grid_traversal_factor.first) << 5) ^
        (std::hash<size_t>{}(grid_traversal_factor.second) << 6) ^
        (std::hash<size_t>{}(splitk_factor) << 7);
    return attr_hash;
  }

  bool sameAs(const HeuristicParams* other_base) const override {
    auto other = dynamic_cast<const MatmulParams*>(other_base);
    if (other == nullptr) {
      return false;
    }

    return other->cparams == cparams && other->mma_macro == mma_macro &&
        other->async_gmem_load_operands == async_gmem_load_operands &&
        other->tile_sizes == tile_sizes &&
        other->circular_buffer_options == circular_buffer_options &&
        other->supported_vec_size == supported_vec_size &&
        other->cta_order == cta_order &&
        other->tiling_strategy == tiling_strategy &&
        other->buffering_loop_level == buffering_loop_level &&
        other->circular_buffering_strategy == circular_buffering_strategy &&
        other->use_ldst_matrix == use_ldst_matrix &&
        other->grid_traversal_factor == grid_traversal_factor &&
        other->use_smem_epilogue == use_smem_epilogue &&
        other->promote_prologue_smem_reuse == promote_prologue_smem_reuse &&
        other->cluster_dims == cluster_dims &&
        other->splitk_factor == splitk_factor;
  }

  std::unique_ptr<HeuristicParams> clone() const override {
    return std::make_unique<MatmulParams>(*this);
  }
};

} // namespace nvfuser
