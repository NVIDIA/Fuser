// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#include <scheduler/matmul_heuristic.h>
#include <scheduler/matmul_heuristic_plugin.h>
#include <scheduler/matmul_utils.h>
#include <scheduler/mma_utils.h>
#include <scheduler/registry.h>
#include <scheduler/runtime_info.h>

// NOTE: included to avoid compilation error caused by missing destructor in
// 'SchedulerRuntimeInfo'
#include <cuda_utils.h>
#include <debug.h>
#include <id_model/id_model.h>
#include <ir/base_nodes.h>
#include <ir/interface_nodes.h>
#include <ir/internal_nodes.h>
#include <ir/utils.h>
#include <mma_type.h>
#include <options.h>
#include <runtime/executor_utils.h>
#include <scheduler/matmul_heuristic.h>
#include <scheduler/mma_utils.h>
#include <type.h>
#include <utils.h>
#include <val_graph.h>

#include <algorithm>
#include <deque>
#include <iostream>
#include <iterator>
#include <limits>
#include <memory>
#include <sstream>
#include <type_traits>
#include <utility>
#include <variant>

#include <ATen/cuda/CUDAContext.h>

namespace nvfuser {
namespace matmul_utils {
namespace {

//! Access to the structure should be done with labels defined in MatmulDimRole.
using ProblemShape = std::array<int64_t, 4>;

//! A helper for deciding the type of MMA op for given fusion and problem shape.
inline std::optional<MmaMacro> getMmaOp(
    const int dev_version,
    const ProblemShape& problem) {
  const int64_t n_extent = problem[(size_t)MatmulDimRole::N];

  MmaMacroEncode macro_encode{MmaMacroEncode::Arch::NoMma, 16, 8, 16};

  switch (dev_version) {
    case 75:
      macro_encode.arch = MmaMacroEncode::Arch::Turing;
      if ((n_extent % 16) == 0) {
        macro_encode.n = 16;
      }
      break;
    case 80:
    case 86:
    case 89:
      macro_encode.arch = MmaMacroEncode::Arch::Ampere;
      if ((n_extent % 16) == 0) {
        macro_encode.n = 16;
      }
      break;
    case 90: {
      macro_encode.arch = MmaMacroEncode::Arch::Hopper;
      macro_encode.m = 64;
      // TODO: guess here whether it is advantageous to double M or double N in
      // order to form the CTA tile. Currently we assume M is doubled (coopA)
      const auto score_macro = [&n_extent](int64_t macro_n) {
        // We prefer a macro_n that divides n_extent most evenly, since a small
        // remainder means we will have a row of tiles with a lot of wasted
        // work. The best case though is when there is no remainder as in that
        // case no extra tile row is needed.
        //
        // However, it is not worth creating new rows of tiles in the output
        // just to avoid a remainder, so the largest contribution to the score
        // is 256 times the number of tile rows needed.
        int64_t score = ceilDiv(n_extent, macro_n) * 256L;
        int64_t remainder = n_extent % macro_n;
        if (remainder != 0) {
          score += macro_n - remainder;
        }
        return score;
      };
      // Scan through all possible macros to find the one with the lowest score
      uint16_t best_macro_n = 8;
      int64_t best_macro_score = score_macro(8);
      for (uint16_t macro_n = 16; macro_n <= 256; macro_n += 8) {
        int64_t score = score_macro((int64_t)macro_n);
        if (score < best_macro_score) {
          best_macro_n = macro_n;
          best_macro_score = score;
        }
      }
      macro_encode.n = best_macro_n;
      break;
    }
    default:
      return std::nullopt;
  }
  return macro_encode;
}

//! Find the number of circular buffer stages for shared memory operands, so
//! that the entire pipeline is filled given problem and heuristics.
void limitCircularBufferingSmemOperands(
    MatmulParams* mparams,
    const ProblemShape& problem_shape) {
  // The axes of the mma tensorviews are permuted to [B, M, N, K],
  // so K / cta_tile_k is the circular buffer axis for both operands.
  int64_t numerator =
      ceilDiv(problem_shape[(size_t)MatmulDimRole::K], mparams->splitk_factor);
  int64_t k_stages_per_tile =
      ceilDiv(numerator, mparams->tile_sizes.cta_tile.k);
  int64_t num_tiles_per_cta = 1;
  if (mparams->tiling_strategy ==
      MatmulParams::TilingStrategy::DistributeTilesAcrossSMs) {
    // Each SM will compute multiple tiles and the circular buffer can wrap
    // around
    const int64_t tiles_m = ceilDiv(
        problem_shape[(size_t)MatmulDimRole::M],
        mparams->tile_sizes.cta_tile.m);
    const int64_t tiles_n = ceilDiv(
        problem_shape[(size_t)MatmulDimRole::N],
        mparams->tile_sizes.cta_tile.n);
    const int64_t num_sms =
        at::cuda::getCurrentDeviceProperties()->multiProcessorCount;
    // NOTE: we don't account for swizzle here which might increase this number
    // in rare cases
    num_tiles_per_cta = ceilDiv(tiles_m * tiles_n, num_sms);
  }
  int64_t stages = std::min(
      (int64_t)mparams->circular_buffer_options.smem_circular_buffer_stage,
      k_stages_per_tile * num_tiles_per_cta);

  // Limit to 8 stages, even though in some cases we could use more. This is
  // needed because mbarrier initialization has a penalty and we do not expect
  // performance gains beyond a certain point.
  stages = std::min(stages, 8L);

  mparams->circular_buffer_options.circular_buffer_smem_write = (stages != 1);
  mparams->circular_buffer_options.smem_circular_buffer_stage = (int)stages;
}

namespace {

bool fillDefaultAmpereHeuristic(
    MatmulParams* mparams,
    const ProblemShape& problem_shape,
    const mma_utils::TensorRolesMap& tensor_roles,
    const size_t num_problems) {
  const GemmTile instruction_tile = getMmaOpShape(mparams->mma_macro);
  GemmTile warp_tile = {-1, -1, -1};
  GemmTile cta_tile = {-1, -1, -1};

  using DimType = decltype(GemmTile::m);

  // warp tile shape
  {
    // Initial target:
    // - 1 MMA ops per thread in a warp (32 threads), warp tile should be
    //   then 32x bigger than instruction tile,
    // - start with [4, 4, 2] shape, later it should depend on problem
    //   shape and have bigger impact on CTA tile shape

    const DimType m_ratio = 4 / (DimType)num_problems;
    const DimType n_ratio = 4;
    const DimType k_ratio = 2;

    warp_tile = {
        instruction_tile.m * m_ratio,
        instruction_tile.n * n_ratio,
        instruction_tile.k * k_ratio};
  }

  // cta tile shape
  {
    // Initial target:
    // - 4 warp tiles per CTA
    // - CTA k-dim should be same as warp tile k-dim

    DimType m_ratio = 2;
    DimType n_ratio = 2;

    const auto mn_ratio = (double)problem_shape[(size_t)MatmulDimRole::M] /
        (double)problem_shape[(size_t)MatmulDimRole::N];
    if (mn_ratio < 0.5) {
      m_ratio = 1;
      n_ratio = 4;
    } else if (mn_ratio > 2) {
      m_ratio = 4;
      n_ratio = 1;
    }

    cta_tile = {warp_tile.m * m_ratio, warp_tile.n * n_ratio, warp_tile.k};
  }

  mparams->tile_sizes = {cta_tile, warp_tile};

  // stages and async mem copy
  {
    // NOTE: compilation errors when async is enabled on Turing devices
    if (isAmpere(mparams->mma_macro)) {
      constexpr int stages = 3;

      mparams->circular_buffer_options.circular_buffer_smem_write = true;
      mparams->circular_buffer_options.circular_buffer_smem_read = true;
      mparams->circular_buffer_options.smem_circular_buffer_stage = stages;
    }
  }

  const auto roleMinDtypeSize =
      [&tensor_roles](MatmulTensorRole role) -> int64_t {
    const auto op_it = tensor_roles.find(role);
    NVF_ERROR(op_it != tensor_roles.end());
    int64_t min_size_bytes = 128LL;
    for (const TensorView* operand : op_it->second) {
      min_size_bytes =
          std::min(min_size_bytes, dataTypeSizeByte(operand->dtype()));
    }
    return min_size_bytes;
  };
  // Use cp.async on Ampere if possible
  mparams->async_gmem_load_operands = isCpAsyncOperandLoadSupported(
      mparams,
      std::min(
          roleMinDtypeSize(MatmulTensorRole::OPERAND_A),
          roleMinDtypeSize(MatmulTensorRole::OPERAND_B)));

  if (!mparams->async_gmem_load_operands) {
    // Circular buffering requires async load. If we cannot use async load due
    // to unsupported vectorization width, then we can only circular buffer at
    // most.
    mparams->circular_buffer_options.smem_circular_buffer_stage = std::min(
        2, mparams->circular_buffer_options.smem_circular_buffer_stage);
  }
  return true;
}

// This function sets cta_tile, warp_tile, and the mma instruction for Hopper
// problems. Our strategy is as follows:
// 1) We use warp_tile.k=cta_tile.k=64 usually, but we also check 16, 32,
// and 48. 2) We always use 1 or 2 math groups. For 2 math groups, the warp tile
// is
//    doubled in either the M or N dimension, called coopA and coopB.
// 3) We check the three scenarios separately for 1 math group, coopA, and
//    coopB.
// 4) For each scenario we check all possible macros. For each possible
//    configuration, we score it for the given problem size. The score indicates
//    the amount of data loaded in total, including OOB accesses as if they were
//    really loaded. This considers that data is loaded multiple times for
//    different tiles.
void fillOptimalHopperTileSizes(
    MatmulParams* mparams,
    const ProblemShape& problem_shape) {
  const int64_t m = problem_shape[(size_t)MatmulDimRole::M];
  const int64_t n = problem_shape[(size_t)MatmulDimRole::N];
  const int64_t k = problem_shape[(size_t)MatmulDimRole::K];

  struct TileConfig {
    GemmTile cta;
    GemmTile warp;
    GemmTile instr;
  };

  const int64_t num_sms =
      at::cuda::getCurrentDeviceProperties()->multiProcessorCount;

  // Compute how many total bytes need to be computed, summed over all tiles in
  // the output
  const auto bytes_transferred = [m, n, k, num_sms](const GemmTile& cta) {
    const int64_t tiles_m = ceilDiv(m, cta.m);
    const int64_t tiles_n = ceilDiv(n, cta.n);
    const int64_t tiles_k = ceilDiv(k, cta.k);

    // This is the number of bytes that get loaded per circular buffering stage
    // in a single tile.
    const int64_t bytes_per_output_tile = tiles_k * (cta.m + cta.n) * cta.k;

    // Now we model wave quantization of the MN tile grid. For example if there
    // are 132 SMs, then we round up the number of tiles to a multiple of 132
    // and then multiply the bytes per output tile by that number. The last wave
    // might run faster than we model it here because it is not really loading
    // that full amount of data, but it is still wasted computation and so this
    // is how we model it. Note that we also do not model L2 locality here at
    // all, so this number is meant as a very rough estimate of compute time for
    // memory-bound problems.
    const int64_t num_waves = ceilDiv(tiles_m * tiles_n, num_sms);
    return num_waves * num_sms * bytes_per_output_tile;
  };

  TileConfig best_config{
      /*cta=*/{64, 64, 64}, /*warp=*/{64, 64, 64}, /*instr=*/{64, 64, 16}};
  int64_t best_bytes_tx = std::numeric_limits<int64_t>::max();
  int64_t best_stages = std::numeric_limits<int64_t>::max();

  // If two sizes result in the same number of total byte, prefer the larger CTA
  // K. To do this we iterate backwards here.
  for (int64_t cta_k = 256; cta_k > 0; cta_k -= 64) {
    for (const auto& [m_ratio, n_ratio] :
         std::vector<std::pair<int64_t, int64_t>>{
             {2L, 1L}, // coopA
             {1L, 2L}, // coopB
             {1L, 1L}, // single math group
         }) {
      for (int64_t instr_n = 256; instr_n > 8; instr_n -= 64) {
        // for (int64_t instr_n = 64; instr_n <= 64; instr_n += 64) {
        const int64_t cta_m = m_ratio * 64;
        const int64_t cta_n = n_ratio * instr_n;

        // TODO: inspect output tensors for epilogue smem dtype
        const int64_t epilogue_bytes = cta_m * cta_n * 2;
        // operands plus outputs plus mbarriers
        const int64_t bytes_per_stage = (cta_m + cta_n) * cta_k * 2 + 8 * 2;
        // Don't consider cases where we would need fewer than 3 load stages due
        // to smem constraint
        const int64_t smem_bytes =
            at::cuda::getCurrentDeviceProperties()->sharedMemPerBlockOptin;
        int64_t min_stages = 3L;

        // There might be fewer than min_stages stages needed to compute the
        // whole problem.
        const int64_t tiles_m = ceilDiv(m, cta_m);
        const int64_t tiles_n = ceilDiv(n, cta_n);
        const int64_t stages_per_tile = ceilDiv(k, cta_k);
        const int64_t tiles_per_sm = ceilDiv(tiles_m * tiles_n, num_sms);
        const int64_t stages_per_sm = stages_per_tile * tiles_per_sm;
        min_stages = std::max(1L, std::min(min_stages, stages_per_sm));

        if (bytes_per_stage * min_stages + epilogue_bytes > smem_bytes) {
          continue;
        }

        if (cta_m > 256 || cta_n > 256 || cta_k > 256) {
          // CTA tile is limited to 256 is each dim
          continue;
        }
        const TileConfig cfg{
            .cta = {cta_m, cta_n, cta_k},
            .warp = {64L, instr_n, cta_k},
            .instr = {64L, instr_n, 16L},
        };
        const int64_t bytes_tx = bytes_transferred(cfg.cta);
        bool new_best = bytes_tx < best_bytes_tx;
        if (bytes_tx == best_bytes_tx) {
          // Break ties by preferring more stages, up to a point. We don't
          // want to choose a non-circular buffered solution (best_stages==1)
          // over a circular buffered solution like stages_per_sm = 5, but it
          // is also not ideal to use so low a cta_k that we need tons of
          // stages
          if (best_stages < 10) {
            if (stages_per_sm > best_stages) {
              new_best = true;
            }
          } else {
            // best_stages is already at least 10. From here prefer fewer but
            // larger stages
            new_best = stages_per_sm >= 10 && stages_per_sm < best_stages;
          }
        }
        if (new_best) {
          best_bytes_tx = bytes_tx;
          best_config = cfg;
          best_stages = stages_per_sm;
        }
      }
    }
  }

  mparams->tile_sizes.cta_tile = best_config.cta;
  mparams->tile_sizes.warp_tile = best_config.warp;
  mparams->mma_macro = MmaMacroEncode{
      MmaMacroEncode::Arch::Hopper,
      (uint16_t)64,
      (uint16_t)best_config.instr.n,
      (uint16_t)16};
}

void maximizeHopperOperandStages(
    MatmulParams* mparams,
    const mma_utils::TensorRolesMap& tensor_roles,
    const size_t num_problems) {
  const GemmTile& cta_tile = mparams->tile_sizes.cta_tile;

  // TODO: We should take the main loop structure into account here to get a
  // more accurate estimate in case of horizontal fusion
  int64_t operand_smem_per_stage =
      (int64_t)num_problems * 2 * (cta_tile.m + cta_tile.n) * cta_tile.k;
  // warp specialized kernels require two mbarriers per stage
  if (mparams->circular_buffering_strategy ==
      MatmulParams::CircularBufferingStrategy::WarpSpecialized) {
    // https://docs.nvidia.com/cuda/parallel-thread-execution/#parallel-synchronization-and-communication-instructions-mbarrier-size-alignment
    constexpr int64_t mbarrier_bytes = 8;
    operand_smem_per_stage += 2 * mbarrier_bytes;
  }

  // We leave a bit of space for semaphores.
  int64_t max_operand_smem =
      (int64_t)at::cuda::getCurrentDeviceProperties()->sharedMemPerBlockOptin;
  // TODO: subtract additional space such as mbarriers used to synchronize the
  // epilogue in ping-pong
  if (mparams->tiling_strategy != MatmulParams::TilingStrategy::OneTilePerCTA) {
    // We cannot reuse memory for smem epilogue in persistent kernels.
    for (TensorView* out : tensor_roles.at(MatmulTensorRole::OUTPUT)) {
      max_operand_smem -=
          dataTypeSizeByte(out->dtype()) * cta_tile.m * cta_tile.n;
    }
  }

  // Maximize stages such that stages*operand_smem_per_stage fits into
  // max_operand_smem
  mparams->circular_buffer_options.smem_circular_buffer_stage =
      max_operand_smem / operand_smem_per_stage;

  mparams->circular_buffer_options.circular_buffer_smem_write =
      mparams->circular_buffer_options.smem_circular_buffer_stage > 1;
}

int64_t computeHopperBIDxTiles(
    MatmulParams* mparams,
    const ProblemShape& problem_shape) {
  const GemmTile& cta_tile = mparams->tile_sizes.cta_tile;
  int64_t Mtiles = ceilDiv(problem_shape[(size_t)MatmulDimRole::M], cta_tile.m);
  int64_t Ntiles = ceilDiv(problem_shape[(size_t)MatmulDimRole::N], cta_tile.n);

  // When we choose the rasterization direction, that becomes the fast
  // dimension. However, when we swizzle, we pull groups of a fixed size from
  // the _other_ dimension to create a new inner dimension. We find the swizzle
  // factor that is largest and has the least quantization when we divide that
  // other dimension by the swizzle factor.
  mparams->cta_order = Mtiles <= Ntiles
      ? MatmulParams::TileRasterizationOrder::ColumnMajor
      : MatmulParams::TileRasterizationOrder::RowMajor;

  int64_t BIDx_tiles =
      mparams->cta_order == MatmulParams::TileRasterizationOrder::ColumnMajor
      ? Mtiles
      : Ntiles;
  if (mparams->grid_traversal_factor.first != 1) {
    BIDx_tiles = BIDx_tiles * mparams->grid_traversal_factor.first;
  }
  return BIDx_tiles;
}

bool fillDefaultHopperHeuristic(
    MatmulParams* mparams,
    const ProblemShape& problem_shape,
    const mma_utils::TensorRolesMap& tensor_roles,
    const size_t num_problems) {
  // Use non-persistent kernel
  mparams->tiling_strategy =
      MatmulParams::TilingStrategy::DistributeTilesAcrossSMs;

  fillOptimalHopperTileSizes(mparams, problem_shape);

  maximizeHopperOperandStages(mparams, tensor_roles, num_problems);

  // Use warp specialization on hopper by default
  mparams->circular_buffering_strategy =
      MatmulParams::CircularBufferingStrategy::WarpSpecialized;

  // Always use TMA on Hopper
  mparams->async_gmem_load_operands = true;

  // See here for more information:
  // https://research.colfax-intl.com/cutlass-tutorial-wgmma-hopper/

  // We count the number of tiles in each dimension to determine the
  // rasterization order. The fast rasterization axis is the shortest axis, to
  // encourage L2 hits by looping over the same rows or cols more frequently.
  const GemmTile& cta_tile = mparams->tile_sizes.cta_tile;
  int64_t Mtiles = ceilDiv(problem_shape[(size_t)MatmulDimRole::M], cta_tile.m);
  int64_t Ntiles = ceilDiv(problem_shape[(size_t)MatmulDimRole::N], cta_tile.n);

  // When we choose the rasterization direction, that becomes the fast
  // dimension. However, when we swizzle, we pull groups of a fixed size from
  // the _other_ dimension to create a new inner dimension. We find the swizzle
  // factor that is largest and has the least quantization when we divide that
  // other dimension by the swizzle factor.
  mparams->cta_order = Mtiles <= Ntiles
      ? MatmulParams::TileRasterizationOrder::ColumnMajor
      : MatmulParams::TileRasterizationOrder::RowMajor;

  // TODO: swizzle is computed at the level of CGA so we first set CGA size
  // then set swizzle factor.
  // For CGA size we pick the largest CGA up to size 4 such that the problem
  // size is divisible by CGA
  if (mparams->tiling_strategy != MatmulParams::TilingStrategy::OneTilePerCTA ||
      computeHopperBIDxTiles(mparams, problem_shape) % 2 == 0) {
    const int64_t num_sms =
        at::cuda::getCurrentDeviceProperties()->multiProcessorCount;

    // TODO: When this does not divide num_sms evenly, we should penalize using
    // large CGAs because they will leave some SMs idle.
    int64_t largest_cga = -1L;
    MatmulParams::ClusterDims largest_cga_cfg{1, 1, 1};
    // Allow CGAs larger than 2 only for small problems
    const int64_t max_cga_size =
        Mtiles * Ntiles > ((double)num_sms * .75) ? 2L : 4L;
    for (int64_t cx : arange(1L, 9L)) {
      for (int64_t cy : arange(1L, 9L)) {
        int64_t cga_size = cx * cy;
        if (Mtiles % cx != 0 || Ntiles % cy != 0 || cga_size > max_cga_size) {
          continue;
        }
        if (largest_cga == -1L || cga_size > largest_cga) {
          largest_cga = cga_size;
          largest_cga_cfg = {cx, cy, 1};
        }
      }
    }

    mparams->cluster_dims = largest_cga_cfg;
  }

  // This is the size of the non-fast dimension before swizzling
  int64_t swizzled_tiles =
      mparams->cta_order == MatmulParams::TileRasterizationOrder::RowMajor
      ? Mtiles
      : Ntiles;

  // We swizzle the tiles as much as possible up to 16 tiles. Like choosing
  // the rasterization order, this is used to increase L2 locality. We start at
  // swizzled_tiles-1 because if we set the swizzle factor equal to the size of
  // the non-swizzled dimension that is equivalent to simply switching
  // cta_order. The largest possible divisor of an integer i is i/2, so we
  // start our search there (or 16, whichever is smaller).
  int grid_traversal_factor = std::max(1L, std::min(swizzled_tiles / 2L, 16L));
  while (grid_traversal_factor > 1 &&
         swizzled_tiles % grid_traversal_factor != 0) {
    // Decrease the swizzle factor if it would result in nondivisible splits,
    // since this would unnecessarily increase the grid size.
    grid_traversal_factor--;
  }
  // TODO: Use only 1D grid traversal factor for now
  grid_traversal_factor = 1;
  mparams->grid_traversal_factor = {grid_traversal_factor, 1};

  return true;
}

} // namespace

//! A wrapper for core heuristics initialization.
//! We should have already set mparams->mma_macro before calling this function.
inline bool initCoreHeuristics(
    MatmulParams* mparams,
    const ProblemShape& problem_shape,
    const mma_utils::TensorRolesMap& tensor_roles,
    const size_t num_problems) {
  if (isHopper(mparams->mma_macro)) {
    return fillDefaultHopperHeuristic(
        mparams, problem_shape, tensor_roles, num_problems);
  } else if (isAmpere(mparams->mma_macro) || isTuring(mparams->mma_macro)) {
    return fillDefaultAmpereHeuristic(
        mparams, problem_shape, tensor_roles, num_problems);
  }
  // Unsupported arch
  return false;
}

//! A helper for getting problem shape from fusion and runtime info.
//!
//! For a given domain, try to find the size by evaluating the extent of an
//! IterDomain in each group of that domain type. For example, if there are
//! multiple Batch dimensions, we find all ValGroups that are mapped as
//! MatmulDimRole::Batch, we evaluate the extent of each, then we multiply those
//! dimensions together to get the overall batch size.
ProblemShape getProblemShape(
    const mma_utils::DimRolesMap& dim_roles,
    SchedulerRuntimeInfo& runtime_info) {
  ProblemShape shape{1, 1, 1, 1};
  for (const auto& [g, dom] : dim_roles) {
    NVF_ERROR(!g->empty());
    IterDomain* id = g->front()->as<IterDomain>();
    const PolymorphicValue extent =
        runtime_info.expressionEvaluator().evaluate(id->extent());
    NVF_ERROR(
        extent.hasValue(), "Could not evaluate extent of ", id->toString());
    shape[(size_t)dom] *= extent.as<int64_t>();
  }
  return shape;
}

// Checks that this pattern:
//   - is a GEMM or batch GEMM
//   - has at least two inputs i.e. not A @ A.T
//   - has a single A and a single B operand i.e not A @ (B1 * B2)
//   - has a fusion output with OUTPUT_D role i.e. that has M, N dims
//   - includes all fusion inputs/outputs in its tensor roles
//   - has no fusion inputs with non-trivial allocation domain
std::string isMatmulFusionDefinitionSupported(
    Fusion* fusion,
    const mma_utils::MatmulPattern& pattern,
    const mma_utils::TensorRolesMap& tensor_roles,
    const mma_utils::DimRolesMap& id_roles,
    const ValGraph& broadcast_graph) {
  const auto& fusion_inputs = fusion->inputs();
  const auto& fusion_outputs = fusion->outputs();
  std::vector<TensorView*> mma_inputs = {pattern.A, pattern.B};
  const auto mma_output = pattern.output;

  const auto fusion_inputs_tvs =
      ir_utils::filterByType<TensorView>(fusion_inputs).vector();
  const auto fusion_outputs_tvs =
      ir_utils::filterByType<TensorView>(fusion_outputs).vector();

  constexpr size_t minimal_number_of_inputs = 2;

  // Quick checks
  {
    if (!mma_output->hasReduction()) {
      return "MMA output TV has no reduction domain";
    }

    // Check that there is a single K dimension
    if (std::count_if(
            mma_output->getLogicalDomain().begin(),
            mma_output->getLogicalDomain().end(),
            [](IterDomain* id) { return id->isReduction(); }) != 1) {
      return "MMA output TV must have exactly one reduction (K) dimension";
    }

    // Fusion should contain at least two inputs (for now)
    if (minimal_number_of_inputs > fusion_inputs.size()) {
      return "Fusion inputs contain at least one non-TensorView object";
    }
  }

  // Fusion topology check
  {
    // Track TensorViews with assigned roles so we can check that all inputs and
    // outputs have recognized roles
    std::set<TensorView*> tvs_with_roles;

    {
      for (MatmulTensorRole role :
           {MatmulTensorRole::OPERAND_A, MatmulTensorRole::OPERAND_B}) {
        auto entry = tensor_roles.find(role);
        if (entry != tensor_roles.end()) {
          if (isOptionEnabled(EnableOption::FuseMultipleMatmuls) ||
              1 == entry->second.size()) {
            tvs_with_roles.insert(entry->second.begin(), entry->second.end());
          } else {
            return "There is more than one fusion input that can be MMA "
                   "operand (enable fuse_multiple_matmuls)";
          }
        } else {
          return "No candidate in fusion inputs for MMA operand";
        }
      }
    }

    auto entry = tensor_roles.find(MatmulTensorRole::OUTPUT);
    if (entry != tensor_roles.end()) {
      tvs_with_roles.insert(entry->second.begin(), entry->second.end());
    } else {
      return "No candidate in fusion outputs MMA output";
    }

    // Non-core input roles are optional, no requirements for definitions
    entry = tensor_roles.find(MatmulTensorRole::EPILOGUE_INPUT);
    if (entry != tensor_roles.end()) {
      tvs_with_roles.insert(entry->second.begin(), entry->second.end());
    }

    const auto in_out_tvs_count =
        fusion_inputs_tvs.size() + fusion_outputs_tvs.size();
    if (in_out_tvs_count != tvs_with_roles.size()) {
      return "Detected input/output TVs without assigned roles";
    }
  }

  // Check that canonical dim order is (B)MNK
  // TODO: Remove this check once we are confident that non-standard orders are
  // properly handled
  {
    std::vector<ValGroup> dim_ordering = mma_utils::canonicalDimOrdering(
        tensor_roles, id_roles, broadcast_graph);
    VectorOfUniqueEntries<MatmulDimRole> role_order;
    for (const ValGroup& g : dim_ordering) {
      const auto it = id_roles.find(g);
      NVF_ERROR(it != id_roles.end());
      role_order.pushBack(it->second);
    }
    if (role_order.size() != 3 && role_order.size() != 4) {
      std::stringstream ss;
      ss << "Expected either {B,M,N,K} roles or {M,N,K} but role_order.size()="
         << role_order.size();
      return ss.str();
    }
    if (role_order.back() != MatmulDimRole::K) {
      return "Canonical dim order must be BMNK";
    }
    if (role_order.at(role_order.size() - 2) != MatmulDimRole::N) {
      return "Canonical dim order must be BMNK";
    }
    if (role_order.at(role_order.size() - 3) != MatmulDimRole::M) {
      return "Canonical dim order must be BMNK";
    }

    // Also check that dims within each role are consecutive with one another
    // for this pattern.
    // TODO: Lift this requirement by modifying the definition or setting
    // allocation domains to support this setting in MmaOp
    NVF_ERROR(pattern.output->definition() != nullptr);
    if (pattern.output->definition()->isA<MatmulOp>()) {
      if (TensorDomain::noReductions(
              TensorDomain::noDevices(pattern.B->getLogicalDomain()))
              .size() >
          TensorDomain::noReductions(
              TensorDomain::noDevices(pattern.A->getLogicalDomain()))
              .size()) {
        return "Implicit broadcast in MatmulOp causes new non-consecutive N "
               "dimension";
      }
    }
  }

  // TODO: Lift this requirement once we properly handle output allocation
  // domain
  for (Val* outp : fusion->outputs()) {
    if (auto tv = dynamic_cast<TensorView*>(outp);
        tv && !ir_utils::hasTrivialAllocationDomain(tv)) {
      return "detected output TV with non-trivial allocation domain";
    }
  }
  return "";
}

class VectorizationCalculator {
 public:
  VectorizationCalculator(
      const mma_utils::TensorRolesMap& tensor_roles,
      const mma_utils::DimRolesMap& dim_roles,
      const ValGraph& broadcast_graph,
      SchedulerRuntimeInfo& runtime_info)
      : runtime_info_(runtime_info),
        tensor_roles_(tensor_roles),
        dim_roles_(dim_roles),
        broadcast_graph_(broadcast_graph),
        dim_ordering_(mma_utils::canonicalDimOrdering(
            tensor_roles,
            dim_roles_,
            broadcast_graph_)) {}

  MatmulParams::SupportedVectorization compute() {
    const std::vector<int64_t> a_vecs =
        operandVectorizations(MatmulTensorRole::OPERAND_A);
    NVF_ERROR(
        isOptionEnabled(EnableOption::FuseMultipleMatmuls) ||
            a_vecs.size() == 1,
        "Expected exactly one A operand");
    const std::vector<int64_t> b_vecs =
        operandVectorizations(MatmulTensorRole::OPERAND_B);
    NVF_ERROR(
        isOptionEnabled(EnableOption::FuseMultipleMatmuls) ||
            b_vecs.size() == 1,
        "Expected exactly one B operand");
    return {a_vecs[0], b_vecs[0], epilogueVectorization()};
  }

 private:
  std::vector<int64_t> operandVectorizations(MatmulTensorRole role) {
    std::vector<int64_t> vec_sizes;
    const auto op_it = tensor_roles_.find(role);
    if (op_it != tensor_roles_.end()) {
      for (TensorView* tv : op_it->second) {
        vec_sizes.push_back(operandVectorization(tv));
      }
    }
    return vec_sizes;
  }

  MatmulDimRole dimRole(const ValGroup& g) const {
    auto dim_role_it = dim_roles_.find(g);
    NVF_ERROR(
        dim_role_it != dim_roles_.end(), "Found ValGroup with unknown role");
    return dim_role_it->second;
  }

  int64_t ptrAndDTypeVec(TensorView* tv) const {
    // TODO: ptrOf returns a fully aligned value of 16 for non-inputs.
    // However, we might be provided an output tensor. We should verify once
    // preallocated outputs are fully plumbed in that misaligned pointers are
    // respected in this calculation.
    const int64_t data_ptr_bit = (int64_t)runtime_info_.ptrBitOf(tv);
    int64_t vec_size_bit = scheduler_utils::maxVectorizationWidth(data_ptr_bit);
    vec_size_bit = std::min(vec_size_bit, 128l);
    int64_t vec_size = vec_size_bit / dataTypeSizeBit(tv->dtype());
    vec_size = std::max(vec_size, 1l);
    return vec_size;
  }

  //! To analyze vectorization, we need to know pointer alignment, sizes, and
  //! strides. SchedulerRuntimeInfo contains all this info about fusion
  //! inputs, but fusion outputs are allocated by KernelExecutor so they are
  //! absent from SchedulerRuntimeInfo.
  //!
  //! This function just extracts sizes and strides from runtime_info_ when
  //! the argument is a fusion input. When the input is a fusion output, we
  //! respect the contiguity marked in the allocation domain. For
  //! discontiguous dimensions, we return a stride that has been padded to an
  //! odd value, which is the worst case scenario for vectorization.
  //!
  //! Note that this function is non-const because we use
  //! runtime_info_.expressionEvaluator() which caches intermediate values
  std::pair<std::vector<int64_t>, std::vector<int64_t>> getSizesAndStrides(
      TensorView* tv) {
    if (tv->isFusionInput()) {
      return {
          runtime_info_.getInputAllocationSizes(tv),
          runtime_info_.getInputAllocationStrides(tv)};
    }
    NVF_ERROR(
        tv->isFusionOutput(),
        "getSizesAndStrides should only be called with fusion inputs or "
        "outputs. Found ",
        tv->toString());
    // For outputs, compute sizes using ExpressionEvaluator, then compute
    // strides based on allocation domain, assuming contiguity as marked in
    // the TensorView. For discontiguous dimensions, we compute a stride that
    // is least favorable to vectorization, by padding to an odd value.
    std::vector<int64_t> sizes, strides;
    std::vector<bool> concrete_contig;
    for (auto&& [i, id] : enumerate(tv->getMaybeAllocationDomain())) {
      if (id->isBroadcast()) {
        sizes.push_back(1);
        concrete_contig.push_back(false);
        continue;
      }
      if (id->isReduction()) {
        continue;
      }
      // Record contiguity of concrete dimensions
      std::optional<bool> contig_opt = tv->getContiguity().at(i);
      NVF_ERROR(contig_opt.has_value());
      concrete_contig.push_back(contig_opt.value());

      PolymorphicValue ext =
          runtime_info_.expressionEvaluator().evaluate(id->extent());
      NVF_ERROR(ext.hasValue());
      sizes.push_back(ext.as<int64_t>());
    }

    strides.resize(sizes.size(), 0l);
    int64_t stride = 1;
    for (int64_t i :
         std::views::iota(0, std::ssize(sizes)) | std::views::reverse) {
      strides[i] = sizes[i] == 1 ? 0 : stride;
      stride *= sizes[i];
      if (!concrete_contig.at(i)) {
        // pad non-concrete dims to next odd value
        stride |= 1l;
      }
    }
    return {sizes, strides};
  }

  // Given a TensorView and a vector of dimension ValGroups find
  // vectorization. The vector of dimensions indicates how the tensor will be
  // scheduled; dimensions in tv will be reordered if needed then the vector
  // of dimensions will be merged. We check the allocation domain of tv to
  // tell how the resulting merged TV can be vectorized. If the tensor does
  // not have any inner_dims, then it cannot be vectorized. In that case we
  // return 0 so that this tensor can be ignored in later computation.
  int64_t innerDimsVectorization(
      TensorView* tv,
      const std::vector<ValGroup>& inner_dims) {
    const auto& [sizes, strides] = getSizesAndStrides(tv);
    NVF_ERROR(sizes.size() == strides.size());

    // Position of the outermost vectorizable dimension, in allocation domain
    size_t inner_dim_pos = tv->getMaybeAllocationDomain().size();
    // Product of sizes of all vectorizable dims; i.e. the number of elements
    // in the merged vectorized dimension.
    int64_t inner_dims_numel = 1;
    std::vector<ValGroup> remaining_inner_dims(inner_dims);
    for (int64_t i = (int64_t)tv->getMaybeAllocationDomain().size() - 1; i >= 0;
         --i) {
      IterDomain* id = tv->getMaybeAllocationDomain()[i];
      if (id->isDeviceDim() || id->isReduction() || id->isBroadcast()) {
        continue;
      }

      ValGroup g = broadcast_graph_.toGroup(id);
      // Exit when this does not match the given ordered inner dimension
      if (remaining_inner_dims.empty() || g != remaining_inner_dims.back()) {
        break;
      }
      remaining_inner_dims.pop_back();

      std::optional<bool> c = tv->getContiguity().at(i);
      NVF_ERROR(c.has_value());
      if (!c.value()) {
        // axis is marked discontiguous; can't vectorize
        break;
      } else {
        NVF_CHECK(
            strides.at(i) == inner_dims_numel,
            "TensorView ",
            tv->toString(),
            " has marked contiguous inner dimension ",
            id->toString(),
            " but provided tensor has stride ",
            strides.at(i),
            " in that dimension.");
        inner_dim_pos = i;
        inner_dims_numel *= sizes.at(i);
      }
    }

    if (remaining_inner_dims.size() == inner_dims.size()) {
      // We didn't match any inner dims, so this tensor is not vectorizable.
      return 0l;
    }

    if (inner_dims_numel == 1l) {
      return 1l;
    }

    // Since this is unpredicated vectorization, the size of the innermost
    // dimension must be a multiple of the vectorization factor.
    int64_t vec_size = scheduler_utils::maxVectorizationWidth(inner_dims_numel);

    // Account for misaligned rows due to outer strides
    for (size_t i : arange(inner_dim_pos)) {
      if (sizes.at(i) == 1) {
        // outer size-1 dimensions don't affect vectorizability
        continue;
      }
      vec_size = std::min(
          vec_size, scheduler_utils::maxVectorizationWidth(strides.at(i)));
    }

    return vec_size;
  }

  // Inspect the allocation domain of an operand input TensorView to determine
  // vectorization width.
  //
  // We canonicalize dimensions by reordering them with the given ordering
  // before merging all dimensions that have the same role. For a given
  // operand, this might mean that the inner-most dimension gets reordered to
  // be outer, even if it has the same role as the innermost dimension in the
  // canonical ordering.
  int64_t operandVectorization(TensorView* tv) {
    // Check data pointer alignment
    int64_t vec_size = ptrAndDTypeVec(tv);
    if (vec_size == 1l) {
      return vec_size;
    }

    // Find the inner-most non-batch role for this tensor, and collect all
    // ValGroups in that role, in the canonical ordering.
    std::optional<MatmulDimRole> vec_dim_role = std::nullopt;
    for (int64_t i = (int64_t)(tv->getMaybeAllocationDomain().size()) - 1;
         i >= 0;
         --i) {
      IterDomain* id = tv->getMaybeAllocationDomain()[i];
      if (id->isDeviceDim() || id->isReduction() || id->isBroadcast()) {
        continue;
      }

      ValGroup g = broadcast_graph_.toGroup(id);
      MatmulDimRole dim_role = dimRole(g);
      if (dim_role == MatmulDimRole::Batch) {
        // We cannot vectorize in batch dimensions
        break;
      }
      if (!vec_dim_role.has_value()) {
        vec_dim_role = dim_role;
        break;
      }
    }
    if (!vec_dim_role.has_value()) {
      // Didn't find any dimensions to vectorize
      return 1l;
    }

    // Extract dims with this role in the canonical ordering
    std::vector<ValGroup> ordered_inner_dims;
    for (const ValGroup& og : dim_ordering_) {
      if (dimRole(og) == vec_dim_role.value()) {
        ordered_inner_dims.push_back(og);
      }
    }

    return std::min(vec_size, innerDimsVectorization(tv, ordered_inner_dims));
  }

  int64_t epilogueVectorization() {
    // This is a vector of non-K dimensions sorted from inner to outer
    std::vector<ValGroup> inner_nonk_dims;
    std::optional<MatmulDimRole> inner_nonk_role = std::nullopt;
    for (auto g_it = dim_ordering_.rbegin(); g_it != dim_ordering_.rend();
         ++g_it) {
      const ValGroup& g = *g_it;

      MatmulDimRole dim_role = dimRole(g);
      if (dim_role == MatmulDimRole::K) {
        // Skip K dims since they won't appear in epilogue loop nest
        continue;
      }
      if (!inner_nonk_role.has_value()) {
        inner_nonk_role = dim_role;
      }
      if (dim_role != inner_nonk_role.value()) {
        break;
      }
      inner_nonk_dims.push_back(g);
    }

    if (!inner_nonk_role.has_value() ||
        inner_nonk_role.value() == MatmulDimRole::Batch) {
      // If the innermost non-K dimension is a batch dimension, then we cannot
      // vectorize the outputs since we parallelize batch dimensions across
      // the grid.
      return 1l;
    }

    // Match the innermost dimensions above to contiguous innermost dims in tv
    // from inner to outer. Determine supported vectorization based on product
    // of matching sizes along with all outer strides.
    const auto innerMostVec = [&](TensorView* tv) {
      return std::min(
          ptrAndDTypeVec(tv), innerDimsVectorization(tv, inner_nonk_dims));
    };

    const auto d_it = tensor_roles_.find(MatmulTensorRole::OUTPUT);
    NVF_ERROR(
        d_it != tensor_roles_.end(), "Could not find any output D tensors");
    int64_t vec_size = 16l;
    for (TensorView* tv : d_it->second) {
      int64_t v = innerMostVec(tv);
      if (v == 0) {
        continue;
      }
      vec_size = std::min(vec_size, v);
    }
    if (const auto c_it = tensor_roles_.find(MatmulTensorRole::EPILOGUE_INPUT);
        c_it != tensor_roles_.end()) {
      for (TensorView* tv : c_it->second) {
        int64_t v = innerMostVec(tv);
        if (v == 0) {
          continue;
        }
        vec_size = std::min(vec_size, v);
      }
    }
    return vec_size;
  }

 private:
  SchedulerRuntimeInfo& runtime_info_;
  const mma_utils::TensorRolesMap& tensor_roles_;
  const mma_utils::DimRolesMap& dim_roles_;
  const ValGraph& broadcast_graph_;
  std::vector<ValGroup> dim_ordering_;
};

MatmulParams::SupportedVectorization getSupportedVectorization(
    const mma_utils::TensorRolesMap& tensor_roles,
    const mma_utils::DimRolesMap& dim_roles,
    const ValGraph& broadcast_graph,
    SchedulerRuntimeInfo& runtime_info) {
  VectorizationCalculator calc(
      tensor_roles, dim_roles, broadcast_graph, runtime_info);
  return calc.compute();
}

const char* noopPtx = R"(
.version 8.0
.target sm_90
.address_size 64

.entry noopKernel()
{
  ret;
}

)";

} // anonymous namespace

//! Determine how many CGAs can launch in a single wave with the given cluster
//! dimensions
int64_t getMaxActiveClusters(const MatmulParams::ClusterDims& cluster_dims) {
  // I don't think we'd ever use a cluster size larger than 8, but we can make
  // space for 8 just to future-proof this
  thread_local std::array<int64_t, 16> cached_results;

  const int64_t cluster_size = cluster_dims.x * cluster_dims.y * cluster_dims.z;
  if (cached_results.at(cluster_size) != 0L) {
    return cached_results.at(cluster_size);
  }

  CUmodule mod;
  NVFUSER_CUDA_SAFE_CALL(cuModuleLoadData(&mod, noopPtx));
  CUfunction func;
  NVFUSER_CUDA_SAFE_CALL(cuModuleGetFunction(&func, mod, "noopKernel"));

  int max_smem_opt_in;
  NVFUSER_CUDA_SAFE_CALL(cuDeviceGetAttribute(
      &max_smem_opt_in,
      CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN,
      /*device=*/0));
  NVFUSER_CUDA_SAFE_CALL(cuFuncSetAttribute(
      func, CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, max_smem_opt_in));
  NVFUSER_CUDA_SAFE_CALL(cuFuncSetAttribute(
      func, CU_FUNC_ATTRIBUTE_NON_PORTABLE_CLUSTER_SIZE_ALLOWED, 1));

  size_t maxDynamicSmemSize;
  NVFUSER_CUDA_RT_SAFE_CALL(cudaOccupancyAvailableDynamicSMemPerBlock(
      &maxDynamicSmemSize, func, /*numBlocks*/ 1, /*blockSize*/ 1));

  int32_t max_active_blocks;
  NVFUSER_CUDA_RT_SAFE_CALL(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      &max_active_blocks, func, /*blockSize=*/1, maxDynamicSmemSize));

  cudaLaunchConfig_t config{0};
  cudaLaunchAttribute attribute[1];
  attribute[0].id = cudaLaunchAttributeClusterDimension;
  attribute[0].val.clusterDim.x = (uint32_t)cluster_size;
  attribute[0].val.clusterDim.y = 1;
  attribute[0].val.clusterDim.z = 1;

  config.numAttrs = 1;
  config.attrs = attribute;
  config.blockDim.x = 128;
  config.blockDim.y = 1;
  config.blockDim.z = 1;

  config.dynamicSmemBytes = maxDynamicSmemSize;

  config.gridDim.x = (unsigned int)cluster_size;
  config.gridDim.y = (unsigned int)max_active_blocks;
  config.gridDim.z = 1;

  int num_clusters;
  NVFUSER_CUDA_RT_SAFE_CALL(
      cudaOccupancyMaxActiveClusters(&num_clusters, (CUfunction)func, &config));

  NVFUSER_CUDA_SAFE_CALL(cuModuleUnload(mod));

  cached_results.at(cluster_size) = (int64_t)num_clusters;
  return cached_results.at(cluster_size);
}

std::unique_ptr<MatmulParams> getMatmulHeuristics(
    Fusion* fusion,
    SchedulerRuntimeInfo& runtime_info,
    HeuristicDataCache* data_cache) {
  FusionGuard fg(fusion);
  (void)data_cache;
  auto mparams = std::make_unique<MatmulParams>();

  // Set kernel index mode
  mparams->cparams.index_type = runtime_info.getIndexType();

  // Check initial conditions
  std::vector<mma_utils::MatmulPattern> patterns =
      mma_utils::findMatmulPatterns(fusion);
  NVF_ERROR(!patterns.empty(), "No matmul patterns were found");
  NVF_ERROR(
      isOptionEnabled(EnableOption::FuseMultipleMatmuls) ||
          patterns.size() == 1,
      "Only a single matmul pattern can currently be fused ",
      "unless the fuse_multiple_matmuls option is enabled");
  mma_utils::MatmulPattern& pattern = patterns.front();

  // IdModel is used to analyze problem shape & layout
  IdModel id_model(fusion, /*build_graphs=*/false);
  id_model.maybeBuildGraph(IdMappingMode::BROADCAST);

  const mma_utils::DimRolesMap id_roles = pattern.getDimRoles(id_model);

  const auto problem_shape = getProblemShape(id_roles, runtime_info);

  const auto device_prop = at::cuda::getCurrentDeviceProperties();
  const auto mma_op =
      getMmaOp(device_prop->major * 10 + device_prop->minor, problem_shape);
  NVF_ERROR(
      mma_op.has_value(), "Failed to determine a MMA op for given problem.");
  mparams->mma_macro = mma_op.value();

  const auto& tensor_roles_opt =
      mma_utils::getTensorRoles(fusion, id_model, id_roles);
  NVF_ERROR(
      tensor_roles_opt.isValid(), "Tensor roles map in mma is not valid.");
  const auto tensor_roles = tensor_roles_opt.getData();

  mparams->supported_vec_size = getSupportedVectorization(
      tensor_roles,
      id_roles,
      id_model.idGraph(IdMappingMode::BROADCAST),
      runtime_info);

  // Populate heuristic details
  auto status = initCoreHeuristics(
      mparams.get(),
      problem_shape,
      tensor_roles,
      // TODO: this assumes all patterns will lie in the same main loop, which
      // might be false
      /*num_problems=*/patterns.size());
  NVF_ERROR(status, "Initialization of core part of heuristics failed.");

  if (matmul_heuristic_plugin::hasPlugin()) {
    const mma_utils::MatmulOperandInnerDimsOpt inner_dims_opt =
        mma_utils::getOperandInnerDims(id_model, id_roles, tensor_roles);
    NVF_ERROR(inner_dims_opt.isValid(), inner_dims_opt.getErrorMsg());
    const mma_utils::MatmulOperandInnerDims inner_dims =
        inner_dims_opt.getData();

    // Fill in proper values using plugin
    matmul_heuristic_plugin::updateMatmulParams(
        mparams.get(),
        problem_shape[(size_t)MatmulDimRole::M],
        problem_shape[(size_t)MatmulDimRole::N],
        problem_shape[(size_t)MatmulDimRole::K],
        problem_shape[(size_t)MatmulDimRole::Batch],
        inner_dims,
        tensor_roles);

    if (mparams->splitk_factor > 1) {
      // Disable persistent for split-K
      mparams->tiling_strategy = MatmulParams::TilingStrategy::OneTilePerCTA;

      // Set cluster dims automatically, ignoring the value set by the plugin
      // It is necessary to do this because we might switch between persistent
      // and non-persistent after the plugin runs and this changes the launch
      // grid.
      // TODO: respect the cluster size given by plugin
      if (mparams->tiling_strategy !=
              MatmulParams::TilingStrategy::OneTilePerCTA ||
          computeHopperBIDxTiles(mparams.get(), problem_shape) % 2 == 0) {
        mparams->cluster_dims = {2, 1, 1};
      } else {
        mparams->cluster_dims = {1, 1, 1};
      }
    }

    // TODO: more sophisticated handling of multiple matmuls when using plugin
    mparams->tile_sizes.cta_tile.m /= (int64_t)patterns.size();
  } else {
    TORCH_WARN_ONCE(
        "Scheduling a matmul without heuristic plugin. "
        "Specify plugin location like this: "
        "NVFUSER_MATMUL_HEURISTIC_PLUGIN=/path/to/libmatmulheuristic.so");
  }

  if (isHopper(mparams->mma_macro)) {
    // Always maximize stages on hopper, for both default heuristic and plugin
    maximizeHopperOperandStages(
        mparams.get(), tensor_roles, /*num_problems=*/(size_t)patterns.size());
  }

  // Ensure that entire pipeline is filled for shared memory operands given
  // problem and heuristics.
  limitCircularBufferingSmemOperands(mparams.get(), problem_shape);

  mparams->circular_buffer_options.circular_buffer_smem_write =
      mparams->circular_buffer_options.smem_circular_buffer_stage > 1;
  mparams->circular_buffer_options.circular_buffer_smem_read = false;
  if (isHopper(mparams->mma_macro) &&
      mparams->circular_buffer_options.smem_circular_buffer_stage > 1) {
    mparams->circular_buffering_strategy =
        MatmulParams::CircularBufferingStrategy::WarpSpecialized;
  } else {
    mparams->circular_buffering_strategy =
        MatmulParams::CircularBufferingStrategy::Pipelined;
  }

  // Disable magic zero for matmul kernels
  mparams->cparams.enable_magic_zero = false;

  // Set whether to use shared memory for epilogue
  std::tie(mparams->use_smem_epilogue, mparams->promote_prologue_smem_reuse) =
      mma_utils::generateSharedMemoryEpilogueHeuristics(
          mparams->tile_sizes,
          mparams->circular_buffer_options.smem_circular_buffer_stage,
          tensor_roles,
          /*ignore_occupancy_drop=*/true);

  // We currently hit a misaligned read for TMA store when warp tile is not a
  // multiple of 64 and there are multiple warp tiles
  // TODO: enable smem epilogue for warp tile sizes that are not multiples of 64
  // See https://github.com/NVIDIA/Fuser/issues/3602
  if (isHopper(mparams->mma_macro) &&
      mparams->tile_sizes.warp_tile.n != mparams->tile_sizes.cta_tile.n &&
      mparams->tile_sizes.warp_tile.n % 64 != 0) {
    mparams->use_smem_epilogue = false;
    mparams->promote_prologue_smem_reuse = false;
  }

  if (isHopper(mparams->mma_macro) && mparams->use_smem_epilogue) {
    // Disable stmatrix unless the problem size is divisible by 16x16, which is
    // the hardcoded stmatrix size we use currently. See
    // See https://github.com/NVIDIA/Fuser/issues/3963
    mparams->use_ldst_matrix =
        problem_shape[(size_t)MatmulDimRole::M] % 16L == 0L &&
        problem_shape[(size_t)MatmulDimRole::N] % 16L == 0L;

    // Disable stmatrix when warp N is less than 32 bytes since this indicates
    // we will not be swizzling.
    mparams->use_ldst_matrix =
        mparams->use_ldst_matrix && mparams->tile_sizes.warp_tile.n >= 16;

    // Always promote smem reuse for Hopper. This is needed because we use TMA
    // which has higher alignment requirements, so it's important that we place
    // our TMA buffers at an offset that's a multiple of 64 (like 0) if
    // possible.
    mparams->promote_prologue_smem_reuse = true;

    // TMA allows us to nearly always avoid linear indexing.
    // verify here that we will be able to use Int32 indexing. If not,
    // then disable use_smem_epilogue.
    NVF_CHECK(
        problem_shape[(size_t)MatmulDimRole::M] <=
                std::numeric_limits<int32_t>::max() &&
            problem_shape[(size_t)MatmulDimRole::N] <=
                std::numeric_limits<int32_t>::max() &&
            problem_shape[(size_t)MatmulDimRole::K] <=
                std::numeric_limits<int32_t>::max(),
        "Cannot safely use TMA because one of the M, N, or K dimensions "
        "overflows Int32");
    // TODO: Setting index type like this causes launch failure
    // mparams->cparams.index_type = PrimDataType::Int32;
  }

  if (isDebugDumpEnabled(DebugDumpOption::SchedulerDebug)) {
    debug() << mparams->toString() << std::endl;
  }

  return mparams;
}

// The analysis is based on mul-sum pair pattern, detected in the provided
//  fusion definition. If detected and properties of an instance of such
//  pattern are valid then it will be later replaced with MmaOp in
//  fusion definition.
// For the time being a direct instance of MmaOp is also accepted and handled
//  by the analysis.
std::string getMatmulCompileTimeRejectReason(Fusion* fusion) {
  // The plan:
  // 0. Check if the current CUDA device is supported
  // 1. Check if there is exactly one matmul pattern defined in the fusion.
  // 2. Check if fusion of MatmulOp and LinearOp is enabled, if applicable
  // 3. Check if inputs to the matmul pattern match any of
  // supported inputs layout
  // 4. Check if fusion represents expressions that are recognized by matmul
  // 5. Check if the input layout for the matmul pattern can be determined
  // scheduler.
  // 6. Check if the fusion is resharding.

  const auto device_prop = at::cuda::getCurrentDeviceProperties();

  // #0
  {
    // Use a dummy problem shape to determine whether this is a supported
    // device.
    const auto mma_op = getMmaOp(
        device_prop->major * 10 + device_prop->minor, {128, 128, 128, 1});
    if (!mma_op.has_value()) {
      return "Unsupported device compute capability";
    }
  }

  // #1
  // Find matmul patterns
  std::vector<mma_utils::MatmulPattern> patterns =
      mma_utils::findMatmulPatterns(fusion);
  if (patterns.empty()) {
    return "No matmul patterns were found";
  }

  // #2
  {
    for (const mma_utils::MatmulPattern& pattern : patterns) {
      Expr* op = pattern.output->definition();
      if (device_prop->major >= 9) {
        for (TensorView* operand : {pattern.A, pattern.B}) {
          // Check that the prologue is trivial
          TensorView* tv = operand;
          while (!tv->isFusionInput()) {
            Expr* def = tv->definition();
            NVF_ERROR(
                def != nullptr,
                "Unexpected undefined tensor ",
                tv->toString(),
                " which is not a fusion input");
            if (!def->isOneOf<LoadStoreOp, BroadcastOp, SqueezeOp>()) {
              return "Operand " + operand->toString() +
                  " must have only trivial prologue ops (set, broadcast, "
                  "squeeze) but found " +
                  def->toString();
            }
            tv = ir_utils::getTvInput(def);
          }
        }
        if (op->isA<ReductionOp>()) {
          bool found_reduction = false;
          for (size_t dim : arange((size_t)pattern.output->nDims())) {
            if (found_reduction &&
                !pattern.output->axis((int64_t)dim)->isReduction()) {
              return "Mul+Sum patterns can only be translated to MmaOp "
                     "on Hopper if the reduction dim is innermost";
            }
          }
        }
      }
      if (op->isA<MatmulOp>() || op->isA<LinearOp>()) {
        if (!isOptionEnabled(EnableOption::FuseMatmul)) {
          // Check for MatmulOp or LinearOp. If found, then only fuse if option
          // is specified
          return "MatmulOp and LinearOp fusion is disabled by default. "
                 "Enable it using NVFUSER_ENABLE=fuse_matmul";
        }
        // Refuse patterns containing 1D inputs since these are mat-vec as
        // opposed to mat-mat products.
        if (pattern.A->nDims() < 2 || pattern.B->nDims() < 2) {
          return "Cannot fuse matrix-vector products";
        }
        for (TensorView* operand : {pattern.A, pattern.B}) {
          if (operand->dtype() != DataType::Half &&
              operand->dtype() != DataType::BFloat16) {
            return "Unsupported operand type. Operands must be fp16 or bf16";
          }
        }
      }
    }
  }

  if (!isOptionEnabled(EnableOption::FuseMultipleMatmuls) &&
      patterns.size() > 1) {
    return "Only a single matmul pattern can currently be fused";
  }

  // #3
  // Prepare an IdModel which will be reused to check remaining conditions
  IdModel id_model(fusion, /*build_graphs=*/false);
  const auto id_roles = patterns.front().getDimRoles(id_model);
  const mma_utils::TensorRolesMapOpt tensor_roles_opt =
      mma_utils::getTensorRoles(fusion, id_model, id_roles);
  if (!tensor_roles_opt.isValid()) {
    return {tensor_roles_opt.getErrorMsg()};
  }
  mma_utils::TensorRolesMap tensor_roles = tensor_roles_opt.getData();

  // #4
  {
    auto support_status = isMatmulFusionDefinitionSupported(
        fusion,
        patterns.front(),
        tensor_roles,
        id_roles,
        id_model.idGraph(IdMappingMode::BROADCAST));
    if (!support_status.empty()) {
      return support_status;
    }
  }

  // #5
  const auto input_layout_opt =
      mma_utils::getOperandInnerDims(id_model, id_roles, tensor_roles);
  if (!input_layout_opt.isValid()) {
    return input_layout_opt.getErrorMsg();
  }

  // #6
  if (scheduler_utils::isResharding(fusion)) {
    return "Fusion is resharding.";
  }

  return "";
}

std::string getMatmulRunTimeRejectReason(
    Fusion* fusion,
    HeuristicDataCache* data_cache,
    SchedulerRuntimeInfo& runtime_info) {
  // On Hopper, we use TMA to load operands. Since TMA requires each coordinate
  // of the input to be represented with a 32-bit signed int, we will encounter
  // overflow if any dimension of an operand is larger than that.
  const auto device_prop = at::cuda::getCurrentDeviceProperties();
  if (device_prop->major == 9) {
    for (Val* inp : fusion->inputs()) {
      if (auto* tv = dynamic_cast<TensorView*>(inp)) {
        for (int64_t extent : runtime_info.getInputAllocationSizes(tv)) {
          if (extent >= (1L << 31)) {
            std::stringstream ss;
            ss << "Cannot schedule Hopper matmul with dims larger than 2^31-1, "
                  "but found "
               << extent;
            return ss.str();
          }
        }
      }
    }
  }
  return "";
}

bool isCpAsyncOperandLoadSupported(
    const MatmulParams* mparams,
    int64_t min_dtype_size) {
  if (!isAmpere(mparams->mma_macro)) {
    return false;
  }
  // Use cp.async for loading operands if vec size is compatible
  const auto& validCpAsyncVecSize = [](int64_t dtype_size,
                                       int64_t vec_size) -> bool {
    int64_t cp_bytes = dtype_size * vec_size;
    return cp_bytes == 16 || cp_bytes == 8 || cp_bytes == 4;
  };
  // TODO: We should compute validCpAsyncVecSize for all the operand
  // dtype/vec_size pairs and AND them together
  return mparams->circular_buffer_options.smem_circular_buffer_stage > 1 &&
      validCpAsyncVecSize(
             min_dtype_size,
             std::min(
                 mparams->supported_vec_size.a, mparams->supported_vec_size.b));
}

void moveInnerBroadcastLeft(TensorView* tv, int64_t number_of_inner_pos) {
  NVF_ERROR(tv->nDims() >= number_of_inner_pos);
  std::vector<int64_t> broadcast_pos;
  std::vector<int64_t> nonbroadcast_pos;

  for (auto i : arange(number_of_inner_pos)) {
    auto axis_idx = i - number_of_inner_pos;
    auto id = tv->axis(axis_idx);
    if (id->isBroadcast()) {
      broadcast_pos.push_back(axis_idx);
    } else {
      nonbroadcast_pos.push_back(axis_idx);
    }
  }

  auto combined_pos_vec = broadcast_pos;
  combined_pos_vec.insert(
      combined_pos_vec.end(), nonbroadcast_pos.begin(), nonbroadcast_pos.end());

  std::unordered_map<int64_t, int64_t> order_map;
  for (auto i : arange(number_of_inner_pos)) {
    order_map[combined_pos_vec.at(i)] = i - number_of_inner_pos;
  }

  // Apply ordering.
  tv->reorder(order_map);
}
} // namespace matmul_utils
} // namespace nvfuser
