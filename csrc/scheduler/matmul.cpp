// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#include <scheduler/matmul.h>
#include <scheduler/mma_utils.h>
#include <scheduler/utils.h>

namespace nvfuser {

namespace {

// Returns true if given number is power of 2
bool isPowOf2(int x) {
  return x > 1 && (x & (x - 1)) == 0;
}

// Move the broadcast axes to the left on the specified number of inner
// dimensions e.g.  (when number_of_inner_pos == 3):
//      [... I0, B, I1] -> [... B, I0, I1]
//  should probably be only used to order innermost mnk axes.
void moveInnerBroadcastLeft(TensorView* tv, int number_of_inner_pos = 3) {
  TORCH_INTERNAL_ASSERT(int(tv->nDims()) >= number_of_inner_pos);
  std::vector<int> broadcast_pos;
  std::vector<int> nonbroadcast_pos;

  for (auto i : c10::irange(number_of_inner_pos)) {
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

  std::unordered_map<int, int> order_map;
  for (auto i : c10::irange(number_of_inner_pos)) {
    order_map[combined_pos_vec.at(i)] = i - number_of_inner_pos;
  }

  // Apply ordering.
  tv->reorder(order_map);
}

//! Automatically generates the shared memory swizzled data layout
//!  for matmul mainloop.
//! The shared mem datalayout is always 2D currently, and this utility
//!  function assumes that the innermost 2 dimensions on shared_mem_tv
//!  are the ones begin swizzled.
void prologSwizzle(TensorView* shared_mem_tv, const MatmulParam& params) {
  // Check that the innermost 2 dimensions are concrete and static
  //  sized so that the swizzle function can be defined.

  // Utility to check concrete static size:
  auto check_concrete_static_dim = [](IterDomain* id) {
    TORCH_INTERNAL_ASSERT(
        !id->isBroadcast() && !id->isReduction(),
        "no support on reduction or broadcast dims, but get ",
        id->toString());
    TORCH_INTERNAL_ASSERT(
        id->extent()->isConstInt(),
        "swizzled dimensions need to be statically, but get ",
        id->toString());
  };

  TORCH_INTERNAL_ASSERT(
      shared_mem_tv->nDims() >= 2,
      "At least 2D input needed for swizzling, but get ",
      shared_mem_tv->toString());
  check_concrete_static_dim(shared_mem_tv->axis(-2));
  check_concrete_static_dim(shared_mem_tv->axis(-1));

  auto mma_config = params.mma_builder.build();

  // Extract the constant sizes of the swizzled tile
  const auto tile_size_x = shared_mem_tv->axis(-2)->extent()->evaluateInt();
  const auto tile_size_y = shared_mem_tv->axis(-1)->extent()->evaluateInt();

  // TODO: add support for tf32(different macro) and fp32(ffma)
  if (isTuring(mma_config.macro) || isAmpere(mma_config.macro)) {
    // Dimension of each inner unit of swizzled indices.
    // Turing and Ampere case, ldmatrix access assumed (see TODO above)
    // Each ldmatrix access is 8x8
    int row_unit = 8;
    int col_unit = 8;

    // Column size of the tile needs to be multiples of 8 for ldmatrix to work.
    TORCH_INTERNAL_ASSERT(
        tile_size_x >= row_unit && tile_size_x % row_unit == 0 &&
            tile_size_y >= col_unit && tile_size_y % col_unit == 0,
        "Prolog swizzle for ldmatrix, illegal tile size for prolog swizzle",
        tile_size_x,
        "x",
        tile_size_y);

    int units_per_row = tile_size_y / col_unit;

    // Number of column units that can fit in a conflict free shared mem wave
    //  with memory width = 128 Byte assumed.
    const int units_per_memory_row =
        128 / dataTypeSize(DataType::Half) / col_unit;

    // Calculate swizzle period:
    int residue_unit_count = units_per_row % units_per_memory_row;

    // In the case where tile row is a multiple of memory row, the whole memory
    // row
    //  is the repeated pattern of swizzle. In the case where tile row is not
    //  divisible, the residule part is the repeated pattern.
    int repeated_pattern_size_in_units =
        residue_unit_count == 0 ? units_per_memory_row : residue_unit_count;

    // Calculate row multiplier, which is defined as minimum number of rows
    //  to look down from an element until the same bank index is observed.
    c10::optional<int> maybe_row_multiplier = c10::nullopt;

    if (units_per_memory_row % repeated_pattern_size_in_units == 0) {
      maybe_row_multiplier =
          units_per_memory_row / repeated_pattern_size_in_units;
    } else if (
        units_per_memory_row > repeated_pattern_size_in_units &&
        units_per_memory_row %
                (units_per_memory_row - repeated_pattern_size_in_units) ==
            0) {
      maybe_row_multiplier = units_per_memory_row /
          (units_per_memory_row - repeated_pattern_size_in_units);
    }

    // The case where the row multiplier cannot be an integer would be where
    //  fractional tiling support is needed. Would gradually build out support
    //  on this one.
    if (!maybe_row_multiplier.has_value()) {
      // calculate effective row_period = lcm(row_period, repeated_pattern) /
      // repeated_pattern_size which is the same as below
      int row_period = units_per_memory_row /
          std::gcd(units_per_memory_row, repeated_pattern_size_in_units);

      if (row_period < row_unit) {
        TORCH_WARN_ONCE(
            "Fractional pattern not yet implemented for swizzling memory row of size :",
            units_per_memory_row,
            " and tile row of size: ",
            repeated_pattern_size_in_units);
        // This would not lead to functional issue but just perf regression, so
        // just do not swizzle anything yet.
        //  TODO: add support for swizzles with different row and col periods to
        //  enable this case.
        return;
      } else {
        // This case would not need swizzling at all as the period of
        //   memory bank index over the row is wider than the access window.
        return;
      }
    } else if (maybe_row_multiplier.value() >= row_unit) {
      // No need to swizzle in this case.
      return;
    }

    // Calculate swizzle period, only equal row/col periods at the moment:
    //  TODO: aperiodic swizzle could also be supported in a follow up:
    int max_swizzle_period = repeated_pattern_size_in_units;

    int swizzle_period = max_swizzle_period;

    // Do not have to use the max_swizzle period if we already had
    //  enough swizzle to permute a row_unit. This would encourage
    //  usage of power of 2 swizzle periods.
    if (row_unit % maybe_row_multiplier.value() == 0) {
      swizzle_period =
          std::min(swizzle_period, row_unit / maybe_row_multiplier.value());
    }

    int row_multiplier = maybe_row_multiplier.value();

    TORCH_INTERNAL_ASSERT(
        tile_size_x % (swizzle_period * row_multiplier) == 0 &&
            tile_size_y % (swizzle_period * col_unit) == 0,
        "need aperiodic swizzle config for tile size ",
        tile_size_x,
        "x",
        tile_size_y,
        "with units ",
        row_unit,
        "x",
        col_unit);

    // add the swizzling op:
    shared_mem_tv->split(-2, row_multiplier * swizzle_period);
    shared_mem_tv->split(-2, row_multiplier);

    shared_mem_tv->split(-1, col_unit * swizzle_period);
    shared_mem_tv->split(-1, col_unit);

    //        -6        -5           -4              -3        -2       -1
    // [..., Irow_o, Irow_period, Irow_multiplier, Icol_o, Icol_period,
    // Icol_unit]
    if (isPowOf2(swizzle_period)) {
      shared_mem_tv->swizzle(Swizzle2DType::XOR, -5, -2);
    } else {
      shared_mem_tv->swizzle(Swizzle2DType::CyclicShift, -5, -2);
    }

    // Merge back the tile for subsequent vectorization scheduling
    //  TODO: could potentially simplify away the merges
    shared_mem_tv->merge(-6);
    shared_mem_tv->merge(-5);
    shared_mem_tv->merge(-3);
    shared_mem_tv->merge(-2);
  } else if (isVolta(mma_config.macro)) {
    // TODO: Volta is slightly more complex, and a fixed recipe would
    //  not scale. In a follow up this would be inferred from the mma
    //  macro layout themselves as we already have them registered in
    //  the utils.
    return;
  } else {
    TORCH_INTERNAL_ASSERT(false, "Prolog swizzle: unsupported mma macro");
  }
}

//! Generates the prolog schedule on the shared memory buffer
//!  tensor. The scheduling performs two steps:
//!
//! 1. Swizzled the shared mem data layout.
//! 2. Coalesce and vectorize the read write schedule.
void scheduleProlog(TensorView* shared_mem_tv, const MatmulParam& params) {
  // Swizzle the shared memory data layout
  prologSwizzle(shared_mem_tv, params);

  // Assuming we are always vectorizing smem write by 128b at the moment:
  //   TODO: would need a data-type and alignment dependent interface
  //    to support non-vectorizable shapes.
  //   The vectorizable width logic would be in a separate PR as the
  //    current effort tries to focus on generating swizzles.
  shared_mem_tv->merge(-2);
  scheduler_utils::matmul_utils::scheduleContiguousVectorLoad(
      shared_mem_tv, params.tile_sizes, 8, false);
}

} // namespace

void scheduleMatmul(
    TensorView* c,
    TensorView* a,
    TensorView* b,
    MatmulParam& params) {
  // Unpack from params.
  auto& mma_builder = params.mma_builder;
  auto& gemm_tile = params.tile_sizes;

  // Including current tensor naming convention for reference,
  //  this is very temporary and will change over time and
  //  in fact the whole body of this function will
  //  eventually be a set of utility functions for different
  //  sections of matmul(fusion) kernels, with
  //  each having its own build out to do.
  //
  // Current naming convention:
  //
  //  operands assumed in global memory : a, b
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
  //  accumulator register: cc (short for c cache)
  //
  //  result in global memory: c

  // Currently only support a, b, c as fusion inputs/outputs
  //  aka. no prolog and epilog fusion yet.
  TORCH_CHECK(
      c->isFusionOutput() && a->isFusionInput() && b->isFusionInput(),
      "not supporting matmul fusion yet");
  TORCH_CHECK(c->definition() && c->definition()->isA<MmaOp>());

  mma_builder.configureMma(c);

  // TODO:
  // Beyond this point, mma_builder really just becomes a populated
  //  list of parameters to describes the mma swizzles that should
  //  be annotated on the tensor domain. Conceptually the mma builder
  //  object should be separated to 2 parts, one as scheduler utility
  //  and the other as matmul heuristic parameters, which we are
  //  starting to build out.

  // Setup register and shared memory stages:
  //   TODO: this section goes to a separate matmul util,
  //   and needs more configurability.

  // Setup accumulator register.
  auto cc = c->cacheBefore();

  // Get the input to the mma op.
  auto mma = dynamic_cast<MmaOp*>(cc->definition());
  TORCH_INTERNAL_ASSERT(mma != nullptr);
  auto ab = mma->inA()->as<TensorView>();
  auto bb = mma->inB()->as<TensorView>();

  // Get exact configurations from mma builder.
  mma_builder.accumulatorTv(cc);
  auto mma_options = mma_builder.build();

  // Staging register for global memory load
  TensorView *ar = a, *br = b;

  if (!params.async_gmem_load_operands) {
    ar = a->cacheAfter();
    br = b->cacheAfter();
  }

  // TODO:
  //  Significant build out needed here
  //   for more flexibility and data type support.
  // Shared memory
  TensorView* acw_smem = nullptr;
  TensorView* bcw_smem = nullptr;
  // Shared memory read
  TensorView* acr = nullptr;
  TensorView* bcr = nullptr;

  // Different paths because Volta swizzle needs to
  //  involve the broadcast dimensions that are concretized
  //  at mma, while Ampere ones should be done before
  //  the broadcast op to be able to use cp.async.
  // TODO:
  // Also a few additional parameters should be introduced
  // to control this stage of scheduling.
  if (isVolta(mma_options.macro)) {
    acw_smem = ab->cacheAfter();
    bcw_smem = bb->cacheAfter();
    // Cache again to be able to vectorize.
    acw_smem = acw_smem->cacheAfter();
    bcw_smem = bcw_smem->cacheAfter();

    acr = acw_smem->cacheAfter();
    bcr = bcw_smem->cacheAfter();
    if (params.double_buffer_options.double_buffer_smem_read) {
      // Provide another copy op between the double buffered
      //  smem load register and the actual mma ops to avoid
      //  complication in double buffered fragment iteration.
      ab = acr->cacheAfter();
      bb = bcr->cacheAfter();
    } else {
      ab = acr;
      bb = bcr;
    }

  } else {
    // Use cp.async as requested in scheduler params.
    c10::optional<LoadStoreOpType> load_op = c10::nullopt;
    if (params.async_gmem_load_operands) {
      load_op = LoadStoreOpType::CpAsyncCg;
    }

    acw_smem = ar->cacheAfter(load_op);
    bcw_smem = br->cacheAfter(load_op);
    acr = acw_smem->cacheAfter(
        mma_builder.operand(MmaOptions::Operand::A).ldMatrix());
    bcr = bcw_smem->cacheAfter(
        mma_builder.operand(MmaOptions::Operand::B).ldMatrix());
  }

  // Make a CTA tile
  // ------------------------------------------------------------------
  scheduler_utils::matmul_utils::canonicalizeMmaTvOrdering(cc);
  // [... M,N,K]
  scheduler_utils::matmul_utils::makeTile(cc, gemm_tile.cta_tile.toVector());

  // Applies swizzle factor on C
  if (params.grid_swizzle_factor != 1) {
    int factor = std::max(1, params.grid_swizzle_factor); // must be >=1
    if (params.rasterization_order ==
        MatmulParam::TileRasterizationOrder::RowMajor) {
      cc->split(1, factor);
      // [I1, I2/factor, factor]
      cc->reorder({{1, 2}});
      // [I1, factor, I2/factor]
      cc->merge(0);
      // [I1*factor, I2/factor]
    } else if (
        params.rasterization_order ==
        MatmulParam::TileRasterizationOrder::ColumnMajor) {
      cc->split(0, factor);
      // [I1/factor, factor, I2]
      cc->reorder({{1, 2}});
      // [I1/factor, I2, factor]
      cc->merge(1);
      // [I1/factor, I2*factor]
    }
  }

  // [Mo, No, Ko, Mi, Ni, Ki]
  // Propagate tiling globally
  scheduler_utils::transformPropagateToAllFrom(cc, -1);

  // Schedule warp tile
  scheduler_utils::matmul_utils::scheduleWarpTileWithReduction(cc, gemm_tile);

  // Propagate warp tile to main loop and epilog/output tvs
  scheduler_utils::BoundedDirectionalTransformPropagator::bothWays(
      cc, -1, {acw_smem, bcw_smem}, {c});

  // Schedule prolog:
  //   TODO: this section goes to a separate matmul util,
  //   and needs more configurability.
  // ------------------------------------------------------------------
  scheduler_utils::matmul_utils::orderTiledConcreteIdAsRoot(acw_smem);
  // [... M, K]
  scheduleProlog(acw_smem, params);

  scheduler_utils::matmul_utils::orderTiledConcreteIdAsRoot(bcw_smem);
  // [... N, K]
  scheduleProlog(bcw_smem, params);

  // Propagate prolog tensors
  //  propagate up the DAG, and propagate parallel type.
  scheduler_utils::BoundedDirectionalTransformPropagator::backward(
      acw_smem,
      -1,
      {a},
      scheduler_utils::BoundedDirectionalTransformPropagator::Options()
          .propagateParallelType());
  scheduler_utils::BoundedDirectionalTransformPropagator::backward(
      bcw_smem,
      -1,
      {b},
      scheduler_utils::BoundedDirectionalTransformPropagator::Options()
          .propagateParallelType());

  // Set computeAt, setup the loop nesting structure on the kernel.
  //   TODO: this section goes to a separate matmul util,
  //   and needs more configurability.
  // ------------------------------------------------------------------
  // CTA tile:

  // Swizzle block tiles:
  // c->swizzle(Swizzle2DType::ZShape, 0, 1, SwizzleMode::Loop);

  a->computeAt(c, 2);
  b->computeAt(c, 2);

  // Prolog:
  a->computeAt(cc, 3);
  b->computeAt(cc, 3);

  // Main Loop:
  acr->computeAt(cc, -6);
  bcr->computeAt(cc, -6);

  // Add mma swizzle:
  //   TODO: this section goes to a separate matmul util,
  //   and needs more configurability.
  // ------------------------------------------------------------------
  if (isTuring(mma_options.macro) || isAmpere(mma_options.macro)) {
    moveInnerBroadcastLeft(ab);
    moveInnerBroadcastLeft(bb);
  }

  ab->applyMmaSwizzle(mma_builder.operand(MmaOptions::Operand::A).build());
  bb->applyMmaSwizzle(mma_builder.operand(MmaOptions::Operand::B).build());

  // Propagate mma input swizzle up the DAG
  //  to all the tensors before mma op and after shared mem read.
  scheduler_utils::BoundedDirectionalTransformPropagator::backward(
      ab,
      -1,
      {acw_smem},
      scheduler_utils::BoundedDirectionalTransformPropagator::Options()
          .propagateParallelType());
  scheduler_utils::BoundedDirectionalTransformPropagator::backward(
      bb,
      -1,
      {bcw_smem},
      scheduler_utils::BoundedDirectionalTransformPropagator::Options()
          .propagateParallelType());

  cc->applyMmaSwizzle(
      mma_builder.operand(MmaOptions::Operand::Accumulator).build());

  // Set memory type:
  acw_smem->setMemoryType(MemoryType::Shared);
  bcw_smem->setMemoryType(MemoryType::Shared);

  // Set parallelization:
  //   TODO: this section goes to a separate matmul util,
  //   and needs more configurability.
  // ------------------------------------------------------------------

  // Vectorize smem stores/loads:
  acw_smem->axis(-1)->parallelize(ParallelType::Vectorize);
  bcw_smem->axis(-1)->parallelize(ParallelType::Vectorize);

  acr->axis(-1)->parallelize(ParallelType::Vectorize);
  bcr->axis(-1)->parallelize(ParallelType::Vectorize);

  //  0   1  2  3   4   5   6  7  8  9  10
  // [Mo No Ko Kwo Mwo Nwo Mw Nw (Mi Ni Ki)]
  if (params.rasterization_order ==
      MatmulParam::TileRasterizationOrder::RowMajor) {
    cc->axis(0)->parallelize(ParallelType::BIDx);
    cc->axis(1)->parallelize(ParallelType::BIDy);
  } else if (
      params.rasterization_order ==
      MatmulParam::TileRasterizationOrder::ColumnMajor) {
    cc->axis(0)->parallelize(ParallelType::BIDy);
    cc->axis(1)->parallelize(ParallelType::BIDx);
  } else {
    TORCH_CHECK(
        false, "Invalid TileRasterizationOrder passed to Matmul scheduler");
  }

  cc->axis(4)->parallelize(ParallelType::TIDz);
  cc->axis(5)->parallelize(ParallelType::TIDy);

  // Propagate mma output swizzle and parallelization down the DAG
  if (params.double_buffer_options.double_buffer_smem_write) {
    TORCH_CHECK(
        params.double_buffer_options.smem_double_buffer_stage > 1,
        "Invalid buffer stage config")
    if (params.double_buffer_options.smem_double_buffer_stage > 2) {
      TORCH_CHECK(
          params.async_gmem_load_operands,
          "Circular buffer only supports async load");
    }

    acw_smem->circularBuffer(
        params.double_buffer_options.smem_double_buffer_stage);
    bcw_smem->circularBuffer(
        params.double_buffer_options.smem_double_buffer_stage);
  }

  if (params.double_buffer_options.double_buffer_smem_read) {
    acr->doubleBuffer();
    bcr->doubleBuffer();
  }

  scheduler_utils::BoundedDirectionalTransformPropagator::forward(
      cc,
      -1,
      {c},
      scheduler_utils::BoundedDirectionalTransformPropagator::Options()
          .propagateParallelType()
          .propagateToBoundary());

  if (params.double_buffer_options.double_buffer_smem_read &&
      params.double_buffer_options.double_buffer_smem_write) {
    scheduler_utils::rotateLoop(cc, 2, {acr, bcr});
  }
}

} // namespace nvfuser
