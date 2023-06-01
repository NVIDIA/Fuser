// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#include <inlining.h>
#include <scheduler/matmul.h>
#include <scheduler/mma_utils.h>
#include <scheduler/registry.h>
#include <scheduler/utils.h>

// NOTE: included to avoid compilation error caused by missing destructor in
// 'SchedulerRuntimeInfo'
#include <executor_utils.h>
#include "mma_type.h"

namespace nvfuser {

namespace {

// Returns true if given number is power of 2
constexpr bool isPowOf2(int64_t x) {
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
void prologSwizzle(TensorView* shared_mem_tv, const MatmulParams& params) {
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

  // Extract the constant sizes of the swizzled tile
  const auto tile_size_x = shared_mem_tv->axis(-2)->extent()->evaluateInt();
  const auto tile_size_y = shared_mem_tv->axis(-1)->extent()->evaluateInt();

  if (isTuring(params.mma_macro) || isAmpere(params.mma_macro)) {
    // TODO: right now, we are assuming ldmatrix access, which only supports
    // sizeof(T) == 16bit (i.e. half/bfloat16) load according to offical doc:
    // https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#warp-level-matrix-load-instruction-ldmatrix
    // In the future, when we start adding support for tf32(different macro),
    // fp32(ffma), double, int8, fp8, etc. we need to update this function.
    TORCH_INTERNAL_ASSERT(dataTypeSize(*shared_mem_tv->getDataType()) == 2);

    // ldmatrix loads a ldmatrix_rows x ldmatrix_cols = 8 x 8 matrix each time,
    constexpr int64_t ldmatrix_rows = 8;
    constexpr int64_t ldmatrix_cols = 8;

    // Column size of the tile needs to be multiples of 8 for ldmatrix to work.
    TORCH_INTERNAL_ASSERT(
        tile_size_x >= ldmatrix_rows && tile_size_x % ldmatrix_rows == 0 &&
            tile_size_y >= ldmatrix_cols && tile_size_y % ldmatrix_cols == 0,
        "Prolog swizzle for ldmatrix, illegal tile size for prolog swizzle",
        tile_size_x,
        "x",
        tile_size_y);

    /* Note [How to remove bank conflict for ldmatrix?]
     *
     * **This note is interleaved with code, I suggest reading this note like
     *   reading a jupyter notebook**
     *
     * Our task is to make sure different rows does not fall into the same
     * bank of shared memory.
     *
     * Introduction to bank conflict can be found at page 54-72 of:
     * https://on-demand.gputechconf.com/gtc/2018/presentation/s81006-volta-architecture-and-performance-optimization.pdf
     *
     * When we talk about bank conflict removal, we are talking about the
     * following task:
     *   "there are 32 banks, and each bank contains one 4-byte word, we want to
     *    make sure different lanes in a warp does not access different word
     *    addresses in the same bank"
     * For example, if thread 0 is accessing word address 1, and thread 1 is
     * accessing word address 33, then these two threads will have a bank
     * conflict because they are accessing different word addresses in the same
     * bank. However, if thread 0 is accessing byte address 4 and thread 1 is
     * accessing byte address 6 then there will be no bank conflict because 4
     * and 6 both belong to word 1.
     */

    constexpr int64_t smem_bytes_per_word = 4;
    constexpr int64_t smem_banks = 32;

    /* but here, for our convenience, because ldmatrix always use vectorized
     * access of 8 items = 16 bytes = 4 words, we further group words into
     * units: we consider each 4 words as a "unit", and each 4 banks as a
     * "megabank". So we can rephrase our task as:
     *   "there are 8 megabanks, and each megabanks contains one 4-word unit, we
     *    want to make sure different lanes in a warp does not access different
     *    unit addresses in the same megabank"
     * In this terminology, matrices are in the row major format, each matrix
     * has 8 rows, and each row has exactly one unit.
     */

    constexpr int64_t items_per_unit = ldmatrix_cols;
    constexpr int64_t bytes_per_unit =
        items_per_unit * primDataTypeSize(DataType::Half);
    constexpr int64_t words_per_unit = bytes_per_unit / smem_bytes_per_word;
    constexpr int64_t num_megabanks = smem_banks / words_per_unit;

    /* In the following example, each CTA tile contains 2 rows and 3 colums of
     * matrices, each 8x8 size:
     *   +----------+----------+----------+
     *   | matrix 0 | matrix 1 | matrix 2 |
     *   +----------+----------+----------+
     *   | matrix 3 | matrix 4 | matrix 5 |
     *   +----------+----------+----------+
     * The addresses of different rows in the same matrix are offset by 3 units.
     * In this perspective, loading a matrix is a strided memory access with the
     * following stride (in units):
     */

    // number of units per row
    int64_t row_stride = tile_size_y / items_per_unit;

    /* So the bank conflicting problem is now converted to the following game:
     *   I have a clock that has one pointer and `num_megabanks` ticks. I start
     *   my game by making my pointer pointing to somewhere, and turn forward
     *   the pointer `ldmatrix_rows` times, each time by `row_stride` ticks.
     * This problem can be well modeled by modular arithmetic in number theory
     * using the concept "integers modulo n" a.k.a. "Z/nZ"[1].
     * Take n = 6 as an example, Z/6Z only has 6 elements: 0, 1, 2, 3, 4, 5.
     * Additions and multiplications are defined in a cyclic manner:
     *   5 + 1 = 0, 5 + 2 = 1, 5 + 3 = 2, 5 + 4 = 3, ...
     *   2 * 1 = 2, 2 * 2 = 4, 2 * 3 = 0, 2 * 4 = 2, ...
     * With this definition, Z is mapped to Z/nZ naturally by i -> i % n [2]
     *
     * It worth mention that Z/nZ is a "commutative ring", that is, we can use
     * addition and multiplication rules just like using normal integers:
     *   a + b = b + a, a * (b + c) = a * b + a * c, ...
     * In short, we can reason about Z/nZ just like we are reasoning about
     * integers, except that every number is automatically "% n".
     *
     * Reference:
     * [1] https://en.wikipedia.org/wiki/Modular_arithmetic#Integers_modulo_n
     * [2] The % is under Euclidean definition, that is -1 % 6 is 5 instead of
     *     -1, see [The Mathematics of Integer Arithmetic] for more detail. But
     *     we are only interested in non-negative numbers here, so there is no
     *     need to worry about this problem
     */

    // row_stride in Z/nZ, where n is num_megabanks:
    // assert(row_stride >= 0);
    // assert(num_megabanks >= 0);
    int64_t row_stride_znz = row_stride % num_megabanks;

    /* Consider the following function in Z/nZ:
     *   f(i; init) = init + i * stride
     * where init is the initial position of the pointer in the clock when we
     * start the game, and stride is the number of ticks we move forward each
     * time, and i is the number of times we move forward. For a fixed init, we
     * abbrivate f(i; init) as f(i).
     *
     * In our problem, f(i) is the megabank of the `i`th row of the matrix, and
     * `init` is the megabank of the 0th row of the matrix.
     *
     * One very important property of f(i) is:
     * - if f(i1) == f(i2), then for every j, f(i1 + j) = f(i2 + j)
     * This property is true because:
     *   f(i1 + j) = f(i1) + j * stride = f(i2) + j * stride = f(i2 + j)
     *
     * The above property tells us, as we turn the clock forward:
     * - initially, we will go to a never-visited tick in each turn, but,
     * - at some point, we will return back to our original position, and,
     * - after we return, we start repeat the pervious pattern again and again.
     *
     * As an example, consider f(i) where init = 0, stride = 6, under Z/8Z:
     *     i  0 1 2 3 4 5 6 7
     *   f(i) 0 6 4 2 0 6 4 2
     * We can see that f(i) is repeating a pattern of four unique numbers
     * "0 6 4 2" twice. In our bank conflict problem, this means we are using 4
     * different megabanks, and we have a 2-way conflict.
     *
     * The question of interest is, does the above observation generalize? That
     * is, does f(i) always repeat a pattern of p unique numbers q times? Note
     * that p and q must satisfy p * q = n.
     *
     * The answer to the above question is: yes! Consider the following
     * equation:
     *    f(i1 + j) == f(i1)
     * We want to know what is the smallest positive number j that makes the
     * above equation true. Because this tells us in how many steps we will see
     * repeat. This equation can be simplified as:
     *   f(i1 + j) == f(i1) + j * stride == f(i1)
     *   ==> j * stride == 0
     *
     * An important tool to study this equation is multiplicative inverse:
     * https://en.wikipedia.org/wiki/Modular_multiplicative_inverse
     * A number i has multiplicative inverse `minv(i)` in Z/nZ if and only if it
     * coprime with n. `minv(i)` is the number that `i * minv(i) == 1`. So in
     * Z/nZ, the equation `ax = b` has solution `x = minv(a)*b` if a has
     * multiplicative inverse. For example, in Z/15Z, `minv(2) = 8` because
     *   (2 * 8) % 15 = 1
     *
     * stride has an multiplicative inverse if and only if stride coprime with
     * n, that is, g := gcd(stride, n) == 1. In such case, the solution to our
     * equation j * stride == 0 is j = minv(stride) * 0 = 0, that is: f(i) does
     * not repeat, that is: there is no bank conflict.
     */

    int64_t g = std::gcd(num_megabanks, row_stride_znz);
    if (g == 1) {
      return; // No need to swizzle in this case.
    }

    /* For the case where stride does not coprime with n, we note that
     * j * stride == 0 in Z/nZ is equivalent to (j * stride) % n = 0 in Z. We
     * can write stride and n as:
     *   stride = s * g, n = m * g
     * According to Theorem 4.13 in [The Mathematics of Integer Arithmetic], we
     * have:
     *   (j * stride) % n = 0
     *   ==> (j * s) % m * g = 0
     *   ==> (j * s) % m = 0
     * which is equivalent to j * s == 0 in Z/mZ. Because s coprime with m, we
     * further get:
     *   j == 0 (in Z/mZ)
     * That is, j is a multiple of m in Z. So the smallest positive j that make
     * the equation hold is n / g.
     *
     * That is: f(i) always repeat a pattern of n/g unique numbers g times.
     * In other word: we are using n/g megabanks, and we have a g-way bank
     * conflict.
     *
     * Let's use the word "pattern" to refer to the set of values of `f` at
     * different `i`, that is:
     *   pattern k = { f(i; init=k) | i in Z/nZ }
     * For the example of stride = 6 under Z/8Z, we have the following patterns
     *        f(i): 01234567
     *   pattern 0: x_x_x_x_
     *   pattern 1: _x_x_x_x
     *   (x => occupied, _ => unoccupied)
     */

    int64_t repeated_pattern_size = num_megabanks / g;

    if (repeated_pattern_size >= ldmatrix_rows) {
      return; // No need to swizzle in this case.
    }

    /* Now we know that we have a g-way bank conflict. How do we remove this
     * bank conflict? The answer is to mix the storage of different matrices.
     * We first split the matrices along the row axis into g pieces, each piece
     * has n/g rows. With this split, each piece occupies exactly one pattern.
     * We want to use some non-traditional storage to let different pieces of
     * the same matrix to occupy different patterns.
     *
     * Because Z/nZ has n items, each pattern has n/g different items, so we
     * have in total g different patterns. We want to find the corresponding
     * `init` values of these g different patterns.
     *
     * Consider two different init values `init1` and `init2`. When do they
     * represent the same pattern? They represent the same pattern if and only
     * if `f(0; init2)` falls on the pattern of `init1`, that is, there exist an
     * i such that
     *   f(i; init1) == f(0; init2)
     * which simplifies to
     *   init1 + i * stride == init2
     *   ==> init2 - init1 == i * stride
     * What values can `i * stride` be? It can be an arbitrary multiple of g:
     * i * stride in Z/nZ is (i * stride) % n in Z. Let m = n/g, according to
     * Theorem 4.13 in [The Mathematics of Integer Arithmetic]
     *   (i * stride) % n = (i * s) % m * g
     * Because s coprime with m, we know that for an arbitrary value `j` in
     * Z/mZ, we can take `i = minv(s) * j` to make `i * s == j`.
     *
     * That said, for init values that are off by a multiple of g they
     * correspond to the same pattern, otherwise they belongs to different
     * patterns. So, we can use
     *   init = 0, 1, ..., g - 1
     * to canonically represent g patterns. Let's call the above
     * `init` values "pattern id".
     *
     * Now we have the idea about how to remove bank conflict: We can do an
     * inner split of our row dimension by `repeated_pattern_size` to get
     * (repeat, pattern), then different indices of the "repeat" dimension will
     * be using the same megabank, and different indices of the "pattern"
     * dimension will be using different megabank. We don't need to touch the
     * "pattern" dimension, but we need to play with the "repeat" dimension to
     * interleave it with matrice ids so that each matrix is distributed across
     * different banks.
     *
     * For example, if we have repeated_pattern_size = 4, we would want to do
     * something like below:
     *    +----------+----------+
     *   0|          |          |
     *   1| matrix 0 | matrix 1 |
     *   2|          |          |
     *   3|          |          |
     *    +----------+----------+
     *   4|          |          |
     *   5| matrix 1 | matrix 0 |
     *   6|          |          |
     *   7|          |          |
     *    +----------+----------+
     */

    //   -2   -1
    // [row, col]
    TORCH_INTERNAL_ASSERT(
        tile_size_x % ldmatrix_rows == 0, "Partial matrices not supported");
    shared_mem_tv->split(-2, ldmatrix_rows);
    TORCH_INTERNAL_ASSERT(
        tile_size_y % ldmatrix_cols == 0, "Partial matrices not supported");
    shared_mem_tv->split(-1, ldmatrix_cols);
    //     -4        -3      -2         -1
    // [matrix id, matrix, matrix id, matrix]
    TORCH_INTERNAL_ASSERT(
        ldmatrix_rows % repeated_pattern_size == 0,
        "ldmatrix_rows is assumed to be a multiple of repeated_pattern_size");
    shared_mem_tv->split(-3, repeated_pattern_size);
    //     -5        -4      -3       -2         -1
    // [matrix id, repeat, pattern, matrix id, matrix]
    int64_t swizzle_period = ldmatrix_rows / repeated_pattern_size;
    TORCH_INTERNAL_ASSERT(
        tile_size_y % (swizzle_period * ldmatrix_cols) == 0,
        "need aperiodic swizzle config for tile size ",
        tile_size_x,
        "x",
        tile_size_y,
        "with units ",
        ldmatrix_rows,
        "x",
        ldmatrix_cols);
    shared_mem_tv->split(-2, swizzle_period);
    //     -6        -5      -4            -3           -2         -1
    // [matrix id, repeat, pattern, matrix id outer, pattern id, matrix]
    // swizzle repeat with pattern id to make repeat no longer repeat
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
  } else if (isVolta(params.mma_macro)) {
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
void scheduleProlog(TensorView* shared_mem_tv, const MatmulParams& params) {
  shared_mem_tv->setMemoryType(MemoryType::Shared);

  mma_utils::orderTiledConcreteIdAsRoot(shared_mem_tv);

  // Swizzle the shared memory data layout
  prologSwizzle(shared_mem_tv, params);

  // Assuming we are always vectorizing smem write by 128b at the moment:
  //   TODO: would need a data-type and alignment dependent interface
  //    to support non-vectorizable shapes.
  //   The vectorizable width logic would be in a separate PR as the
  //    current effort tries to focus on generating swizzles.
  shared_mem_tv->merge(-2);
  mma_utils::scheduleContiguousVectorLoad(
      shared_mem_tv, params.tile_sizes, 8, true);

  // Propagate prolog tensors
  //  propagate up the DAG, and propagate parallel type.
  scheduler_utils::BoundedDirectionalTransformPropagator::backward(
      shared_mem_tv,
      -1,
      {},
      scheduler_utils::BoundedDirectionalTransformPropagator::Options()
          .propagateParallelType());
}

} // namespace

void scheduleMatmul(Fusion* fusion, const MatmulParams& params) {
  const auto& roles_map_opt = mma_utils::getTensorsRoles(fusion);

  // NOTE: the contents of roles_map have been already validated during
  //  compute-time checks
  TORCH_INTERNAL_ASSERT(roles_map_opt.isValid(), roles_map_opt.getErrorMsg());
  const auto roles_map = roles_map_opt.getData();

  auto mma_ops = ir_utils::getMmaOps(fusion);
  TORCH_INTERNAL_ASSERT(
      mma_ops.size() == 1,
      "scheduleMatmul supports fusion with single mma op in definition, got ",
      mma_ops.size());
  TORCH_INTERNAL_ASSERT(
      mma_ops.front()->layout().has_value(),
      "fusion mma op has undefined input layout");

  TensorView* a = roles_map.at(MatmulRole::MMA_INPUT_A);
  TensorView* b = roles_map.at(MatmulRole::MMA_INPUT_B);
  TensorView* c = roles_map.at(MatmulRole::MMA_OUTPUT);

  // Collect mma swizzle info
  auto mma = mma_ops.front();
  const auto mma_layout_opt = mma->layout();
  TORCH_INTERNAL_ASSERT(
      mma_layout_opt.has_value(), "fusion mma op has undefined input layout");
  const auto mma_layout = mma_layout_opt.value();
  const auto fusion_layout = mma_utils::getMatmulLayout(fusion);
  TORCH_INTERNAL_ASSERT(fusion_layout.isValid(), fusion_layout.getErrorMsg());

  auto mma_builder =
      MmaBuilder(params.mma_macro, params.tile_sizes).layout(mma_layout);
  const auto& gemm_tile = params.tile_sizes;

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

  mma_builder.configureMma(c);

  // TODO:
  // Beyond this point, mma_builder really just becomes a populated
  //  list of parameters to describe the mma swizzles that should
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
  mma = cc->definition()->as<MmaOp>();
  auto ab = mma->inA()->as<TensorView>();
  auto bb = mma->inB()->as<TensorView>();

  // Set accumulation tv for mma op.
  mma_builder.accumulatorTv(cc);

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
  if (isVolta(params.mma_macro)) {
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
    LoadStoreOpType load_op = LoadStoreOpType::Set;
    if (params.async_gmem_load_operands) {
      load_op = LoadStoreOpType::CpAsyncCg;
    }

    acw_smem = ar->cacheAfter(load_op);
    bcw_smem = br->cacheAfter(load_op);
    TORCH_INTERNAL_ASSERT(acw_smem->uses().size() == 1);
    TORCH_INTERNAL_ASSERT(bcw_smem->uses().size() == 1);
    if (auto ldst = dynamic_cast<LoadStoreOp*>(acw_smem->uses().at(0));
        ldst != nullptr && ldst->hasTranspose()) {
      acr = ldst->out()->as<TensorView>();
      ldst->setOpType(LoadStoreOpType::LdMatrixTranspose);
    } else {
      acr = acw_smem->cacheAfter(LoadStoreOpType::LdMatrix);
    }
    if (auto ldst = dynamic_cast<LoadStoreOp*>(bcw_smem->uses().at(0));
        ldst != nullptr && ldst->hasTranspose()) {
      bcr = ldst->out()->as<TensorView>();
      ldst->setOpType(LoadStoreOpType::LdMatrixTranspose);
    } else {
      bcr = bcw_smem->cacheAfter(LoadStoreOpType::LdMatrix);
    }

    // For Turing and Ampere, the layout of the MmaOp is always TN
    TORCH_INTERNAL_ASSERT(
        mma_layout == MmaOptions::MmaLayout::TN,
        "MMAs in Turing and Ampere are TN only, transpose is handled either "
        "via ldmatrix.trans for fp16 or explicitly for other types.");
    mma_builder.layout(fusion_layout.getData());
  }

  // Make a CTA tile
  // ------------------------------------------------------------------
  mma_utils::canonicalizeMmaTvOrdering(cc);
  // [... M,N,K]
  mma_utils::makeTile(cc, gemm_tile.cta_tile.toVector());

  // Swizzle block tiles:
  if (params.grid_swizzle_factor != 1) {
    int factor = std::max(1, params.grid_swizzle_factor); // must be >=1
    if (params.cta_order == MatmulParams::TileRasterizationOrder::RowMajor) {
      cc->split(1, factor);
      // [I1, I2/factor, factor]
      cc->reorder({{1, 2}});
      // [I1, factor, I2/factor]
      cc->merge(0);
      // [I1*factor, I2/factor]
    } else if (
        params.cta_order == MatmulParams::TileRasterizationOrder::ColumnMajor) {
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
  mma_utils::scheduleWarpTileWithReduction(cc, gemm_tile);

  // Propagate warp tile to main loop and epilog/output tvs
  scheduler_utils::BoundedDirectionalTransformPropagator::bothWays(
      cc, -1, {acw_smem, bcw_smem}, {c});

  // Schedule prolog:
  //   TODO: this section needs more configurability.
  // ------------------------------------------------------------------
  scheduleProlog(acw_smem, params);
  scheduleProlog(bcw_smem, params);

  // Add mma swizzle:
  //   TODO: this section goes to a separate matmul util,
  //   and needs more configurability.
  // ------------------------------------------------------------------
  if (isTuring(params.mma_macro) || isAmpere(params.mma_macro)) {
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

  // Set parallelization:
  //   TODO: this section goes to a separate matmul util,
  //   and needs more configurability.
  // ------------------------------------------------------------------

  // Vectorize smem stores/loads:
  acr->axis(-1)->parallelize(ParallelType::Vectorize);
  bcr->axis(-1)->parallelize(ParallelType::Vectorize);

  //  0   1  2  3   4   5   6  7  8  9  10
  // [Mo No Ko Kwo Mwo Nwo Mw Nw (Mi Ni Ki)]
  switch (params.cta_order) {
    case MatmulParams::TileRasterizationOrder::RowMajor:
      cc->axis(0)->parallelize(ParallelType::BIDx);
      cc->axis(1)->parallelize(ParallelType::BIDy);
      break;
    case MatmulParams::TileRasterizationOrder::ColumnMajor:
      cc->axis(0)->parallelize(ParallelType::BIDy);
      cc->axis(1)->parallelize(ParallelType::BIDx);
      break;
    default:
      TORCH_INTERNAL_ASSERT(
          false, "Invalid TileRasterizationOrder passed to Matmul scheduler");
  }

  cc->axis(4)->parallelize(ParallelType::TIDz);
  cc->axis(5)->parallelize(ParallelType::TIDy);

  scheduler_utils::parallelizeAllLike(
      cc,
      -1,
      {acr, bcr, ab, bb, a, b},
      {ParallelType::TIDy, ParallelType::TIDz});

  // auto inline for all tensors except register tensors and output tensor
  inlineMost(ir_utils::allTvsExcept(fusion, {acr, bcr, ab, bb, c}));

  // if auto inline, will inline to position-7, leads to performance regression
  inlineSelectedAt({acr, bcr, ab, bb}, cc, 6);

  // Propagate mma output swizzle and parallelization down the DAG
  if (params.double_buffer_options.double_buffer_smem_write) {
    TORCH_INTERNAL_ASSERT(
        params.double_buffer_options.smem_double_buffer_stage > 1,
        "Invalid buffer stage config")
    if (params.double_buffer_options.smem_double_buffer_stage > 2) {
      TORCH_INTERNAL_ASSERT(
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

  c->axis(-1)->parallelize(ParallelType::Vectorize);

  if (params.double_buffer_options.double_buffer_smem_read &&
      params.double_buffer_options.double_buffer_smem_write) {
    scheduler_utils::rotateLoop(cc, 2, {acr, bcr});
  }
}

} // namespace nvfuser
