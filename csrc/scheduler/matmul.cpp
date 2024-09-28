// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#include <abstract_tensor.h>
#include <device_lower/analysis/circular_buffer.h>
#include <inlining.h>
#include <instrumentation.h>
#include <multidevice/utils.h>
#include <scheduler/debug_utils.h>
#include <scheduler/matmul.h>
#include <scheduler/matmul_utils.h>
#include <scheduler/mma_utils.h>
#include <scheduler/multi_matmul.h>
#include <scheduler/utils.h>

// NOTE: included to avoid compilation error caused by missing destructor in
// 'SchedulerRuntimeInfo'
#include <runtime/executor_utils.h>
#include "mma_type.h"

namespace nvfuser {
namespace {

// Returns true if given number is power of 2
constexpr bool isPowOf2(int64_t x) {
  return x > 1 && (x & (x - 1)) == 0;
}

// Utility to check concrete static size:
inline void checkConcreteStaticDim(IterDomain* id) {
  NVF_ERROR(
      !id->isBroadcast() && !id->isReduction(),
      "no support for reduction or broadcast domains, but got ",
      id->toString());
  NVF_ERROR(
      id->extent()->isConstInt(),
      "swizzled dimension's extend must be known during scheduling, got ",
      id->toString());
}

//! Automatically generates the shared memory swizzled data layout
//!  for matmul mainloop and epilogue.
//! The shared mem data layout is always 2D currently, and this utility
//!  function assumes that the shared_mem_tv has the following structure:
//!  [tile_row, tile_col]
//! Returns the domain with swizzle. For the case of legacy swizzle, this
//! domain must be set as loop domain. For the case of new swizzle, this domain
//! must be set as allocation domain.
template <bool legacy = true>
AbstractTensor swizzleSharedMemory(TensorView* shared_mem_tv) {
  NVF_ERROR(shared_mem_tv->getMemoryType() == MemoryType::Shared);
  AbstractTensor swizzle_domain(shared_mem_tv->getLoopDomain());

  // Check that the innermost 2 dimensions are concrete and static
  //  sized so that the swizzle function can be defined.
  NVF_ERROR(
      (int64_t)swizzle_domain.size() >= 2,
      "At least 2D input (excluding consecutive reduction domains starting from the innermost dim) needed for swizzling, but get ",
      shared_mem_tv->toString());
  checkConcreteStaticDim(swizzle_domain[-2].as<IterDomain*>());
  checkConcreteStaticDim(swizzle_domain[-1].as<IterDomain*>());

  // Extract the constant sizes of the swizzled tile
  const int64_t tile_size_x =
      swizzle_domain[-2]->extent()->evaluate().as<int64_t>();
  const int64_t tile_size_y =
      swizzle_domain[-1]->extent()->evaluate().as<int64_t>();

  // Only tested for (1) ldmatrix access with sizeof(T) == 16bit (i.e.
  // half/bfloat16) and (2) epilogue general access with sizeof(T) == 32bit
  // (i.e. float)
  const int64_t data_type_size = dataTypeSize(*shared_mem_tv->getDataType());
  NVF_ERROR(data_type_size == 2 || data_type_size == 4);

  // For main loop, ldmatrix loads a n_rows x n_cols = 8 x 8 matrix each time.
  // For epilogue, threads in a warp is organized as 8 rows x 4 columns.
  // Each thread vectorized write 2 items, so 8 items per row.
  //--0--1--2--3
  //--4--5--6--7
  //--8--9--10-11
  //--12-13-14-15
  //--16-17-18-19
  //--20-21-22-23
  //--24-25-26-27
  //--28-29-30-31
  constexpr int64_t n_rows = 8;
  constexpr int64_t n_cols = 8;

  // Column size of the tile needs to be multiples of 8 for ldmatrix to work.
  NVF_ERROR(
      tile_size_x >= n_rows && tile_size_x % n_rows == 0 &&
          tile_size_y >= n_cols && tile_size_y % n_cols == 0,
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

  constexpr int64_t items_per_unit = n_cols;
  const int64_t bytes_per_unit = items_per_unit * data_type_size;
  const int64_t words_per_unit = bytes_per_unit / smem_bytes_per_word;
  const int64_t num_megabanks = smem_banks / words_per_unit;

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
   *   the pointer `n_rows` times, each time by `row_stride` ticks.
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
    return swizzle_domain; // No need to swizzle in this case.
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

  if (repeated_pattern_size >= n_rows) {
    return swizzle_domain; // No need to swizzle in this case.
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
   *
   * We can consider each repeated_pattern_size rows as a gigarow, and each
   * repeated_pattern_size megabanks as a gigabank. Note that megabank is a
   * contiguous chunk of banks, but gigabank is not contiguous. Indeed,
   * nearby megabanks in a gigabank has a distance of `g` megabanks
   */

  NVF_ERROR(
      n_rows % repeated_pattern_size == 0,
      "Can not partition matrix into megarows");
  int64_t num_gigarows = n_rows / repeated_pattern_size;
  int64_t num_gigabanks = g; // also = num_megabanks / repeated_pattern_size

  //   -2   -1
  // [row, col]
  if (repeated_pattern_size > 1) {
    swizzle_domain.split(-2, repeated_pattern_size);
  }
  swizzle_domain.split(-1, n_cols);
  //      -4         -3       -2        -1
  // [gigarow id, gigarow, matrix id, matrix]
  swizzle_domain.split(-2, num_gigabanks);
  //      -5        -4        -3        -2         -1
  // [gigarow id, gigarow, y outer, gigabank id, matrix]
  // Note that megabanks inside a gigabank are not contiguous, so the gigabank
  // id is -2 instead of -3

  /* We want to evenly distribute gigarows across gigabanks, for example, if
   * we have 7 gigarows and 3 gigabanks, then we might distribute them as:
   *  +---+
   *  |x  |
   *  | x |
   *  |  x|
   *  |x  |
   *  | x |
   *  |  x|
   *  |x  |
   *  +---+
   * considering all matrices, this is a swizzle function like:
   *  +---+
   *  |012|
   *  |201|
   *  |120|
   *  |012|
   *  |201|
   *  |120|
   *  |012|
   *  +---+
   * which is a cyclic shift.
   *
   * Note that because num_gigabanks (a.k.a. g) divide num_megabanks and
   * row_stride_znz (which is row_stride % num_megabanks), g should also
   * divide row_stride, because according to the fundamental
   * division-with-remainder property (see doc/math/integer-division.md):
   *   row_stride = q * num_megabanks + row_stride_znz
   * which means, we can just consider each num_gigabanks matrices as a group,
   * and we always have complete groups (i.e. no group has less than
   * num_gigabanks matrices). Interleaving the memory of matrices within each
   * group should be enough to fully remove bank conflict.
   */

  /* To further simplify the problem, if we assume: */
  NVF_ERROR(
      num_gigarows % num_gigabanks == 0,
      "Requires non-square swizzle, which is not supported yet");
  /* Then we can partition gigarows into full waves, each wave has
   * num_gigabanks gigarows. This partition creates square dimensions, making
   * the swizzle implementation easier */

  //      -5        -4        -3        -2         -1
  // [gigarow id, gigarow, y outer, gigabank id, matrix]
  int axis_of_gigarow_id = repeated_pattern_size > 1 ? -5 : -4;
  swizzle_domain.split(axis_of_gigarow_id, num_gigabanks);
  //     -6     -5     -4       -3        -2         -1
  // [wave id, wave, gigarow, y outer, gigabank id, matrix]

  // swizzle wave with gigabank id to make threads in a wave access different
  // gigabank. Apply swizzle only when shared_mem_tv is stored in shared
  // memory.
  // TODO: This is a temporary workaround for the following issue:
  // For the mma output, we have the following schedule:
  // rFactor: [...., X, Y] -> mma-swizzle transformations -> loop
  // For epilogue smem tensor, the schedule is
  // rFactor: [...., X, Y] -> split -> [...., X1, X2, X3, Y1, Y2, Y3]
  //   -> swizzle X2, Y2 -> [...., X1, X2', X3, Y1, Y2', Y3]
  //   -> merge back -> [...., X', Y']
  //   -> mma-swizzle transformations -> loop
  // The mma-swizzle transformations for the mma output and epilogue smem
  // tensor are the same. In indexing, we do require {X, X'} and {Y, Y'} to be
  // mapped in CA map, however, we currently can not handle that. So we have
  // to do the same split and merge to the mma output without actually
  // applying the swizzle, and this check is to detect and handle this
  // specific case. We should remove this special handling when we fix our CA
  // mapping.
  using SwizzleTypeMaybeLegacy =
      std::conditional_t<legacy, Swizzle2DType, SwizzleType>;
  if (isPowOf2(num_gigabanks)) {
    swizzle_domain.swizzle(SwizzleTypeMaybeLegacy::XOR, axis_of_gigarow_id, -2);
  } else {
    swizzle_domain.swizzle(
        SwizzleTypeMaybeLegacy::CyclicShift, axis_of_gigarow_id, -2);
  }

  if (legacy) {
    if (repeated_pattern_size > 1) {
      swizzle_domain.merge(-6);
    }
    swizzle_domain.merge(-5);

    // merge back tile_size_y
    swizzle_domain.merge(-3);
    swizzle_domain.merge(-2);
  }

  return swizzle_domain;
}

//! Generates the prolog schedule on the shared memory buffer
//!  tensor. The scheduling performs two steps:
//!
//! 1. Swizzled the shared mem data layout.
//! 2. Coalesce and vectorize the read write schedule.
void scheduleProlog(
    TensorView* shared_mem_tv,
    int64_t vec_size,
    const MatmulParams* mparams) {
  shared_mem_tv->setMemoryType(MemoryType::Shared);

  // The following line allows us to reclaim the memory allocated to
  // shared_mem_tv and reuse it for the epilogue, introducing one block sync if
  // needed. This is not done by default as we do not insert new syncs unless
  // requested to do so. If smem is not used for the epilogue, this call will
  // have no effect.
  if (mparams->promote_prologue_smem_reuse) {
    shared_mem_tv->promoteReuse();
  }

  mma_utils::orderTiledConcreteIdAsMaybeAllocationDomain(shared_mem_tv);

  // Swizzle the shared memory data layout
  auto swizzled_dom = swizzleSharedMemory(shared_mem_tv);
  shared_mem_tv->setLoopDomain(swizzled_dom.as<IterDomain*>());
  shared_mem_tv->setHasSwizzleOp();
  // Assuming we are always vectorizing smem write by 128b at the moment:
  //   TODO: would need a data-type and alignment dependent interface
  //    to support non-vectorizable shapes.
  //   The vectorizable width logic would be in a separate PR as the
  //    current effort tries to focus on generating swizzles.
  shared_mem_tv->merge(-2);
  mma_utils::scheduleContiguousVectorLoad(
      shared_mem_tv, mparams->tile_sizes, vec_size, /*vectorize=*/vec_size > 1);

  // Propagate prolog tensors
  //  propagate up the DAG, and propagate parallel type.
  scheduler_utils::BoundedDirectionalTransformPropagator::backward(
      shared_mem_tv,
      -1,
      {},
      scheduler_utils::BoundedDirectionalTransformPropagator::Options()
          .propagateParallelType());
}

void scheduleOutputTensor(
    TensorView* mma_result,
    TensorView* c,
    const MatMulTileOptions& gemm_tile,
    int64_t vectorization_factor) {
  // input tensor is in the form of [Mo,No,cta_tile_m,cta_tile_n]
  checkConcreteStaticDim(c->axis(-2));
  checkConcreteStaticDim(c->axis(-1));
  const int64_t tile_size_m = c->axis(-2)->extent()->evaluate().as<int64_t>();
  const int64_t tile_size_n = c->axis(-1)->extent()->evaluate().as<int64_t>();
  NVF_ERROR(
      tile_size_m == gemm_tile.cta_tile.m,
      "Actual tile size at axis(-2) in output tensor is different from CTA tile size! Expected: ",
      gemm_tile.cta_tile.m,
      ", actual: ",
      tile_size_m);
  NVF_ERROR(
      tile_size_n == gemm_tile.cta_tile.n,
      "Actual tile size at axis(-1) in output tensor is different from CTA tile size! Expected: ",
      gemm_tile.cta_tile.n,
      ", actual: ",
      tile_size_n);
  const int64_t tot_elements = tile_size_m * tile_size_n;
  constexpr int64_t warp_size = 32l;
  const int64_t tidx = warp_size;
  const int64_t tidy = gemm_tile.cta_tile.n / gemm_tile.warp_tile.n;
  const int64_t tidz = gemm_tile.cta_tile.m / gemm_tile.warp_tile.m;
  // step-1, merge last 2 dims
  c->merge(-2);
  // [Mo, No, m*n]

  // step-2, set vectorization to maximum
  // We have fixed tidx, tidy, and tidz, so we need to make sure that the output
  // tensor is divisible by tidx * tidy * tidz * vectorization_factor
  NVF_ERROR(
      tot_elements % (tidx * tidy * tidz * vectorization_factor) == 0,
      "Output tensor cannot be fully vectorized! tot_elements:",
      tot_elements,
      ", tidx: ",
      tidx,
      ", tidy: ",
      tidy,
      ", tidz: ",
      tidz,
      ", vectorization_factor: ",
      vectorization_factor);
  c->split(-1, vectorization_factor);
  c->axis(-1)->parallelize(ParallelType::Vectorize);
  // [Mo, No, m*n/vect, vect]

  // step-3, Split out a warp for TIDx
  c->split(-2, tidx);
  c->axis(-2)->parallelize(ParallelType::TIDx);
  // [Mo, No, m*n/vect/TIDx, TIDx, vect]

  // step-4, Split out for TIDy and TIDz
  // TIDy = cta_tile_n/warp_tile_n
  // TIDz = cta_tile_m/warp_tile_m
  c->split(-3, tidy);
  c->axis(-3)->parallelize(ParallelType::TIDy);

  c->split(-4, tidz);
  c->axis(-4)->parallelize(ParallelType::TIDz);
  // [Mo, No, m*n/vect/TIDx/TIDy/TIDz, TIDz, TIDy, TIDx, vect]

  // step-5, Parallel first 2 dims same as mma_result
  scheduler_utils::parallelizeAllLike(
      mma_result,
      2,
      {c},
      {ParallelType::BIDx, ParallelType::BIDy, ParallelType::BIDz});
}
//! Propagates transformations from fusion output to fusion tv inputs that are
//!  producers in the epilogue. Transformations' propagation aims at input tvs
//!  which are not assigned to core roles, that is, are not MMA inputs.
void scheduleFusionInputsForEpilogue(
    const mma_utils::TensorRolesMap& tensor_roles,
    bool with_smem_epilogue) {
  std::vector<TensorView*> cached_tvs;

  // Handling transformations in fusion input tvs with assigned EPILOGUE_INPUT
  //  role by propagating fusion output transformations through cached views of
  //  EPILOGUE_INPUT fusion input tvs and by setting vectorization of the inner
  //  most iterdomain of these cached views
  if (tensor_roles.count(MatmulTensorRole::EPILOGUE_INPUT)) {
    auto& c_tvs = tensor_roles.at(MatmulTensorRole::EPILOGUE_INPUT);

    // The system supports only scenario where there is only one fusion output
    //  with assigned OUTPUT role, this condition is already verified so there
    //  is no need for an additional checks here
    auto output_d = tensor_roles.at(MatmulTensorRole::OUTPUT).front();
    for (auto* c : c_tvs) {
      cached_tvs.push_back(c->cacheAfter());
    }

    scheduler_utils::BoundedDirectionalTransformPropagator::backward(
        output_d, -1, c_tvs);

    std::unordered_set<ParallelType> parallel_types = {};
    if (with_smem_epilogue) {
      //! In cases where smem epilogue feature is enabled, the vectorization of
      //!  domains will be propagated to fusion inputs that are epilogue inputs,
      //!  this may result in unaligned memory reads. Vectorization is
      //!  explicitly excluded form parallelization types to avoid this issue.
      //! This should be changed when vectorization analysis is available and
      //!  enabled for matmul scheduler.
      parallel_types = allParallelTypesExcept({ParallelType::Vectorize});
    }
    scheduler_utils::parallelizeAllLike(
        output_d, -1, cached_tvs, parallel_types);

    // The cached EPILOGUE_INPUT tvs are not needed anymore
    cached_tvs.clear();
  }
}

void scheduleSplitKSum(
    TensorView* splitk_sum,
    const int64_t num_device_and_batch_dims, // TODO: this should not be needed
    bool use_smem_epilogue) {
  if (splitk_sum == nullptr) {
    // This indicates no split-K was used
    return;
  }

  // Always use serial grid reduction for split-K sum
  splitk_sum->definition()->as<ReductionOp>()->requestSerialGridReduction();

  if (use_smem_epilogue) {
    // Now that transforms are propagated backward to smem_epilogue, which is
    // before splitk_sum, we can vectorize the inner-most non-trivial
    // dimension of splitk_sum
    //
    // Note that the split-K reduction is the inner-most dimension.
    Val* vec_ext = splitk_sum->axis(-2)->extent();
    NVF_ERROR(vec_ext->isConstInt());
    int64_t vec_ext_int = vec_ext->evaluate().as<int64_t>();
    splitk_sum->axis(-1)->parallelize(ParallelType::BIDz);
    splitk_sum->axis(-3)->parallelize(ParallelType::TIDx);
    if (vec_ext_int * dataTypeSize(splitk_sum->dtype()) > 16) {
      // NOTE: We might encounter an illegal vectorization size if we are
      // using Float for this reduction and Half for output. So here we first
      // check whether the vectorize size is at most 16 bytes. If not, then we
      // split into an unrolled loop that will do multiple vectorized
      // reads/writes instead. Note that we reorder such that the axes are in
      // order UR TIDx V.
      splitk_sum->split(
          -2, 16 / dataTypeSize(splitk_sum->dtype()), /*inner_split=*/true);
      splitk_sum->axis(-3)->parallelize(ParallelType::Unroll);
      splitk_sum->reorder({{-4, -3}});
      // In this case, we have [... iUR iTx rBz iS]
    }
    splitk_sum->reorder({{-2, -1}});
  } else { // no smem epilogue
    // Reorder to place the split-K reduction next to innermost [... rBz iS]
    splitk_sum->reorder({{-9, -2}});
  }
  // Vectorize inner-most dimension [... (iUR iTx) rBz iV]
  splitk_sum->axis(-1)->parallelize(ParallelType::Vectorize);
}

void scheduleMatmul(Fusion* fusion, const MatmulParams* mparams) {
  if (isOptionEnabled(EnableOption::FuseMultipleMatmuls)) {
    scheduleMultipleMatmuls(fusion, mparams);
    return;
  }

  FusionGuard fg(fusion);

  // Make sure we don't have global memory set on intermediate tensors from
  // fusion segmentation
  scheduler_utils::clearMemorySpace(fusion);

  // Cache inputs
  scheduler_utils::cacheInputs(fusion, true);

  // Cache and fork outputs
  auto cached_outputs = scheduler_utils::cacheAndForkOutputs(fusion, true);

  std::vector<mma_utils::MatmulPattern> patterns =
      mma_utils::findMatmulPatterns(fusion);
  NVF_ERROR(!patterns.empty(), "No matmul patterns were found");
  NVF_ERROR(
      patterns.size() == 1,
      "Only a single matmul pattern can currently be fused");
  std::vector<MmaOp*> mma_ops;
  mma_ops.reserve(patterns.size());
  for (mma_utils::MatmulPattern& pattern : patterns) {
    mma_ops.push_back(pattern.translateToMmaOp());
  }

  IdModel id_model(fusion);
  mma_utils::DimRolesMap id_roles = patterns.front().getDimRoles(id_model);
  const auto& tensor_roles_opt =
      mma_utils::getTensorRoles(fusion, id_model, id_roles);

  // NOTE: the contents of tensor_roles have been already validated during
  //  compute-time checks
  NVF_ERROR(tensor_roles_opt.isValid(), tensor_roles_opt.getErrorMsg());
  const auto tensor_roles = tensor_roles_opt.getData();

  const mma_utils::MatmulOperandInnerDimsOpt inner_dims =
      mma_utils::getOperandInnerDims(id_model, id_roles, tensor_roles);
  NVF_ERROR(inner_dims.isValid(), inner_dims.getErrorMsg());

  // Core roles: there can be only one... TV with assigned core role
  const std::vector<TensorView*>& a_operands =
      tensor_roles.at(MatmulTensorRole::OPERAND_A);
  NVF_ERROR(
      a_operands.size() == 1, "We currently require exactly one A operand");
  TensorView* a = a_operands.front();
  const std::vector<TensorView*>& b_operands =
      tensor_roles.at(MatmulTensorRole::OPERAND_B);
  NVF_ERROR(
      b_operands.size() == 1, "We currently require exactly one B operand");
  TensorView* b = b_operands.back();

  const auto& gemm_tile = mparams->tile_sizes;

  // Collect mma swizzle info
  auto mma = mma_ops.front();
  const bool has_epilogue = !mma->out()->isFusionOutput();

  const bool has_fusion_c_roles =
      (0 != tensor_roles.count(MatmulTensorRole::EPILOGUE_INPUT));
  const bool has_non_mma_input_tvs = has_epilogue && has_fusion_c_roles;

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

  mma->setMacro(mparams->mma_macro);

  // Setup register and shared memory stages:
  //   TODO: this section goes to a separate matmul util,
  //   and needs more configurability.

  // Setup accumulator register.
  auto mma_result = mma->out()->as<TensorView>();

  // TODO:
  //  Significant build out needed here
  //   for more flexibility and data type support.
  // Shared memory
  TensorView* acw_smem = nullptr;
  TensorView* bcw_smem = nullptr;
  // Shared memory read
  TensorView* acr = nullptr;
  TensorView* bcr = nullptr;

  // Use cp.async as requested in scheduler mparams->
  LoadStoreOpType load_op = LoadStoreOpType::Set;
  CacheOp cache_op_a = CacheOp::Unspecified;
  CacheOp cache_op_b = CacheOp::Unspecified;
  if (mparams->async_gmem_load_operands) {
    load_op = LoadStoreOpType::CpAsync;
    auto getCacheOp = [](int64_t vec_size, TensorView* operand) -> CacheOp {
      int64_t vec_bytes = vec_size * dataTypeSize(operand->dtype());
      NVF_CHECK(
          vec_bytes == 4LL || vec_bytes == 8LL || vec_bytes == 16LL,
          "Unsupported async vectorization size ",
          vec_size,
          " = ",
          vec_bytes,
          " bytes for operand ",
          operand->toString(),
          " which has data type ",
          operand->dtype(),
          ". Size must be 4, 8, or 16 bytes. ",
          "MatmulParams::async_gmem_load_operands should be set to false in this case.");
      return vec_bytes == 16LL ? CacheOp::Global : CacheOp::AllLevels;
    };
    cache_op_a = getCacheOp(mparams->supported_vec_size.a, a);
    cache_op_b = getCacheOp(mparams->supported_vec_size.b, b);
  }

  NVF_ERROR(a->uses().size() == 1);
  NVF_ERROR(b->uses().size() == 1);
  acw_smem = ir_utils::consumerTvsOf(a).at(0);
  acw_smem->definition()->as<LoadStoreOp>()->setOpType(load_op);
  acw_smem->definition()->as<LoadStoreOp>()->setCacheOp(cache_op_a);
  bcw_smem = ir_utils::consumerTvsOf(b).at(0);
  bcw_smem->definition()->as<LoadStoreOp>()->setOpType(load_op);
  bcw_smem->definition()->as<LoadStoreOp>()->setCacheOp(cache_op_b);
  NVF_ERROR(acw_smem->uses().size() == 1);
  NVF_ERROR(bcw_smem->uses().size() == 1);

  // We add two LoadStore operators to the inputs of our fusions. The first one
  // is for a read from global memory and the second one (below) is for
  // a cache read. As an optimizaton, we avoid adding an operator if there's an
  // existing LoadStoreOp present. Please note that for the second LoadStore we
  // don't propagte the allocation domain, since the scheduler sets the
  // allocation domain in the registers.
  auto addSetForCacheRead = [](TensorView* tv_smem, TensorView** tv_r) {
    if (auto ldst = dynamic_cast<LoadStoreOp*>(tv_smem->uses().at(0))) {
      *tv_r = ldst->out()->as<TensorView>();
      ldst->setOpType(LoadStoreOpType::LdMatrix);
    } else {
      *tv_r = tv_smem->cacheAfter(
          LoadStoreOpType::LdMatrix,
          CacheOp::Unspecified,
          /*propagate_allocation_domain=*/false);
    }
  };

  addSetForCacheRead(acw_smem, &acr);
  addSetForCacheRead(bcw_smem, &bcr);

  const std::vector<ValGroup> ordering = mma_utils::canonicalDimOrdering(
      tensor_roles, id_roles, id_model.idGraph(IdMappingMode::PERMISSIVE));

  // Make a CTA tile
  // ------------------------------------------------------------------
  // Dimensions ordered as: [ (device dims), (batch dims), M, N, K ]
  mma_utils::canonicalizeMmaTvOrdering(
      mma_result,
      id_model.idGraph(IdMappingMode::PERMISSIVE),
      id_roles,
      ordering);
  const int64_t num_local_dims =
      (int64_t)TensorDomain::noDevices(mma_result->getLoopDomain()).size();
  NVF_ERROR(
      num_local_dims == 3 || num_local_dims == 4,
      "Currently, we only support B, M, N and K being a single dimension.",
      " More general tensor contraction is not supported yet.");
  const int64_t num_device_dims = numDeviceDims(mma_result);
  const int64_t num_local_batch_dims =
      mma_result->nDims() - num_device_dims - 3;
  const int64_t num_device_and_batch_dims =
      num_device_dims + num_local_batch_dims;

  // [... M,N,K]
  mma_utils::makeTile(mma_result, gemm_tile.cta_tile.toVector());
  // [..., Mo, No, Ko, Mi, Ni, Ki]

  // Unswizzle mma result in shared memory
  // Note that if we are using split-K, we will set up this buffer after
  // rfactoring the matmul, between the MmaOp and the ReductionOp, in order to
  // take advantage of unswizzling during the grid reduction
  TensorView* smem_epilogue = mma_result;

  // Swizzle block tiles:
  if (mparams->grid_swizzle_factor != 1) {
    int factor = std::max(1, mparams->grid_swizzle_factor); // must be >=1
    if (mparams->cta_order == MatmulParams::TileRasterizationOrder::RowMajor) {
      mma_result->split(num_device_and_batch_dims + 1, factor);
      // [I1, I2/factor, factor]
      mma_result->reorder(
          {{num_device_and_batch_dims + 1, num_device_and_batch_dims + 2}});
      // [I1, factor, I2/factor]
      mma_result->merge(num_device_and_batch_dims);
      // [I1*factor, I2/factor]
    } else if (
        mparams->cta_order ==
        MatmulParams::TileRasterizationOrder::ColumnMajor) {
      mma_result->split(num_device_and_batch_dims, factor);
      // [I1/factor, factor, I2]
      mma_result->reorder(
          {{num_device_and_batch_dims + 1, num_device_and_batch_dims + 2}});
      // [I1/factor, I2, factor]
      mma_result->merge(num_device_and_batch_dims + 1);
      // [I1/factor, I2*factor]
    }
  }

  // [..., iMo, iNo, rKo, iMi, iNi, rKi]
  int num_splitk_dims = 0;
  TensorView* splitk_sum = nullptr;
  if (mparams->splitk_factor != 1) {
    // Split Ko -> [rKf, rKg]
    mma_result->split(-4, mparams->splitk_factor, /*inner*/ false);
    // After split [..., iMo, iNo, rKf, rKg, iMi, iNi, rKi]
    // rFactor converts
    //   mma_result = mma(A, B, {/*Kf*/-5, /*Kg*/-4, /*Ki*/-1});
    // to
    //   intermediate = mma(A, B, {-4, -1});
    //   final_sum = sum(intermediate, {/*Kf*/-3});
    // and the method returns "intermediate". We need mma_result to refer to
    // the actual MmaOp output, so here we reassign that to the intermediate.
    splitk_sum = mma_result;
    mma_result = splitk_sum->rFactor({-4, -1});

    num_splitk_dims = 1;
  }

  // At this point we have the following schedule:
  //   No split-K
  //     mma_result      [..., iMo, iNo, rKo, iMi, iNi, rKi]
  //   Split-K
  //     mma_result      [..., iMo, iNo, iKf, rKg, iMi, iNi, rKi]
  //     splitk_sum      [..., iMo, iNo, rKf, iMi, iNi]

  if (mparams->use_smem_epilogue) {
    // Note that for split-K
    //   splitk_sum = sum(mma_result)
    // becomes
    //   smem_epilogue = set(mma_result)
    //   splitk_sum = sum(smem_epilogue)
    smem_epilogue = mma_result->cacheAfter();
    // smem_epilogue = [..., iMo, iNo, iKf, iMi, iNi]
  }

  // Propagate tiling globally
  scheduler_utils::transformPropagateToAllFrom(mma_result, -1);

  if (mparams->use_smem_epilogue && mparams->splitk_factor != 1) {
    // TODO:
    // This is a workaround for a problem that different dimensions in the loop
    // domain are mapped in the loop graph of IdModel due to the mapping of
    // compliment IDs. We should remove forwarding completely, and remove this
    // workaround.
    mma_result->split(-2, 1);
    mma_result->merge(-3);
  }

  // Schedule warp tile
  // Incoming mma_result = [... iMo iNo (iKf) rKg iMi iNi rKi]
  mma_utils::scheduleWarpTileWithReduction(mma_result, gemm_tile);
  // After scheduling warp tile, the last three dimensions are split and
  // rearranged:
  //        -3 -2 -1
  //   [...  M  N  K]
  // maps to
  //         -8  -7 -6  -5 -4 -3 -2 -1
  //   [... Kwo Mwo Nwo Mw Nw Mi Ni Ki]
  // so now
  //                   -12 -11  -10   -9   -8   -7   -6  -5  -4   -3   -2   -1
  // mma_result = [... iMo iNo (iKf) rKg rKwo iMwo iNwo iMw iNw iMin iNin rKin]
  // splitk_sum = [... iMo iNo  rKf  iMi  iNi]

  // Propagate warp tile to main loop and epilog/output tvs
  scheduler_utils::BoundedDirectionalTransformPropagator::bothWays(
      mma_result, -1, {acw_smem, bcw_smem}, {smem_epilogue});

  // No (cross-CTA) split-K
  //   mma_result      [..., iMo iNo rKo rKwo iMwo iNwo iMw iNw iMin iNin rKin]
  //   smem_epilogue   (unscheduled, same as original or current mma_result)
  //   splitk_sum      (nullptr)
  //
  // With split-K
  //   mma_result   [... iMo iNo iKf  rKg rKwo iMwo iNwo iMw iNw iMin iNin rKin]
  //   splitk_sum   [... iMo iNo rKf  iMi  iNi]

  // Schedule prolog:
  //   TODO: this section needs more configurability.
  // ------------------------------------------------------------------
  scheduleProlog(acw_smem, mparams->supported_vec_size.a, mparams);
  scheduleProlog(bcw_smem, mparams->supported_vec_size.b, mparams);

  // Get the input to the mma op.
  mma = mma_result->definition()->as<MmaOp>();
  auto ab = mma->inA()->as<TensorView>();
  auto bb = mma->inB()->as<TensorView>();

  // Add mma swizzle:
  //   TODO: this section goes to a separate matmul util,
  //   and needs more configurability.
  // ------------------------------------------------------------------
  if (isTuring(mparams->mma_macro) || isAmpere(mparams->mma_macro)) {
    matmul_utils::moveInnerBroadcastLeft(ab);
    matmul_utils::moveInnerBroadcastLeft(bb);
  }

  ab->applyMmaSwizzle(MmaOperand::A);
  bb->applyMmaSwizzle(MmaOperand::B);

  // Propagate mma input swizzle up the DAG
  //  to all the tensors before mma op and after shared mem read.
  auto propagate_mma_input_schedule_to = [&](TensorView* a_boundary,
                                             TensorView* b_boundary) {
    if (a_boundary != nullptr) {
      scheduler_utils::BoundedDirectionalTransformPropagator::backward(
          ab,
          -1,
          {a_boundary},
          scheduler_utils::BoundedDirectionalTransformPropagator::Options()
              .propagateParallelType());
    }
    if (b_boundary != nullptr) {
      scheduler_utils::BoundedDirectionalTransformPropagator::backward(
          bb,
          -1,
          {b_boundary},
          scheduler_utils::BoundedDirectionalTransformPropagator::Options()
              .propagateParallelType());
    }
  };
  propagate_mma_input_schedule_to(acw_smem, bcw_smem);

  // This does a split-reorder-merge swizzle of the last two M and N dimensions
  // (and a possible final reduction dim).
  // eg. [M64, N24, R]  -> [WarpGroup128, N3, M2, N2, Ro, R4, R2]
  // Before
  //   mma_result  [... iMo iNo (iKf) rKg rKwo iMwo iNwo iMw iNw iMin iNin rKin]
  // After
  //   mma_result  [... iMo iNo (iKf) rKg rKwo iMwo iNwo iMw
  //                              iNw iMino iNino iMin2 iNin2 rKino rKin4 rKin2]
  {
    auto s = mma_utils::MmaSwizzler::scheduleMmaOutputAllocation(
        mma_result->getLoopDomain());
    mma_result->setLoopDomain(s.as<IterDomain*>());
    mma_result->setAllocationDomain(s.as<IterDomain*>(), true);
  }

  // Set parallelization:
  //   TODO: this section goes to a separate matmul util,
  //   and needs more configurability.
  // ------------------------------------------------------------------

  if (acr != ab) {
    //  -5  -4   -3   -2   -1
    //[8mi, 4k, 2ko, 2mo, 2ki]
    acr->setAllocationDomain(acr->getLoopDomain(), true);
    mma_utils::MmaSwizzler::scheduleLdMatrix(acr, MmaOperand::A);
    ab->merge(-5);
    ab->axis(-4)->parallelize(ParallelType::TIDx);
    propagate_mma_input_schedule_to(acr, nullptr);
  }
  if (bcr != bb) {
    //   -5  -4   -3   -2   -1
    // [8ni, 4k, 2ko, 1no, 2ki]
    bcr->setAllocationDomain(bcr->getLoopDomain(), true);
    mma_utils::MmaSwizzler::scheduleLdMatrix(bcr, MmaOperand::B);
    bb->merge(-5);
    bb->axis(-4)->parallelize(ParallelType::TIDx);
    propagate_mma_input_schedule_to(nullptr, bcr);
  }

  // Parallelization strategy:
  // Here the top two rows indicate how we can index each axis. The third row
  // is what it represents: note that a suffix i means inner and o means outer
  // here. The fourth row is the parallelization strategy:
  //   - i means iterate (produce one value per element i.e. don't reduce)
  //   - r means reduce this dimension
  //   - B: block
  //   - T: thread
  //   - S: serial. This will become a for loop in the generated kernel
  //   - iMMA: uncontracted axis in an MMA tensor core operation.
  //   - rMMA: contract in an MMA tensor core operation.
  //
  // With split-K:
  //   mma_result
  //     nbatch +   1    2    3    4    5    6   7   8
  //              -15  -14  -13  -12  -11  -10  -9  -8
  //     [... iMo iNo (iKf) rKg rKwo iMwo iNwo iMw iNw     ...
  //          iBx iBy  iBz   rS   rS  iTz  iTy  iS  iS
  //                              9    10    11    12    13    14    15
  //                             -7    -6    -5    -4    -3    -2    -1
  //                    ...   iMino iNino iMin2 iNin2 rKino rKin4 rKin2]
  //                            iTx  iMMA  iMMA  iMMA  rMMA  rMMA  rMMA
  //   smem_epilogue   (unscheduled, same as original mma_result)
  //   splitk_sum      (nullptr)
  //
  // Without split-K:
  //   mma_result
  //     nbatch +   1   2    3    4    5   6   7    8
  //              -14 -13  -12  -11  -10  -9  -8   -7
  //     [... iMo iNo rKg rKwo iMwo iNwo iMw iNw iMino
  //    (iBz) iBx iBy  rS   rS  iTz  iTy  iS  iS  iTx
  //                                   9    10    11     12    13    14
  //                                  -6    -5    -4     -3    -2    -1
  //                               iNino iMin2 iNin2  rKino rKin4 rKin2]
  //                                iMMA  iMMA  iMMA   rMMA  rMMA  rMMA
  //   smem_epilogue   (unscheduled, same as original mma_result)
  //   splitk_sum
  //     [... iMo iNo rKf  iMi  iNi]

  // When we have both batch dims and splitk, parallelize splitk only.
  // If we only have batch dim, parallelize the batch dim.
  if (num_splitk_dims != 0) {
    mma_result->axis(num_device_and_batch_dims + 2)
        ->parallelize(ParallelType::BIDz);
  } else if (num_local_batch_dims > 0) {
    mma_result->axis(num_device_dims)->parallelize(ParallelType::BIDz);
  }
  switch (mparams->cta_order) {
    case MatmulParams::TileRasterizationOrder::RowMajor:
      mma_result->axis(num_device_and_batch_dims)
          ->parallelize(ParallelType::BIDx);
      mma_result->axis(num_device_and_batch_dims + 1)
          ->parallelize(ParallelType::BIDy);
      break;
    case MatmulParams::TileRasterizationOrder::ColumnMajor:
      mma_result->axis(num_device_and_batch_dims)
          ->parallelize(ParallelType::BIDy);
      mma_result->axis(num_device_and_batch_dims + 1)
          ->parallelize(ParallelType::BIDx);
      break;
    default:
      NVF_THROW("Invalid TileRasterizationOrder passed to Matmul scheduler");
  }

  // parallelize Mwo, Nwo by thread
  mma_result->axis(num_device_and_batch_dims + 4 + num_splitk_dims)
      ->parallelize(ParallelType::TIDz);
  mma_result->axis(num_device_and_batch_dims + 5 + num_splitk_dims)
      ->parallelize(ParallelType::TIDy);

  scheduler_utils::parallelizeAllLike(
      mma_result,
      -1,
      {acr, bcr, ab, bb},
      {ParallelType::TIDy, ParallelType::TIDz});

  // handle epilogue and always vectorize Ki
  if (mparams->use_smem_epilogue) {
    smem_epilogue->setMemoryType(MemoryType::Shared);
    auto swizzled_dom = swizzleSharedMemory<false>(smem_epilogue);
    smem_epilogue->setAllocationDomain(swizzled_dom.as<IterDomain*>(), true);
    scheduler_utils::BoundedDirectionalTransformPropagator::forward(
        mma_result,
        -1,
        {smem_epilogue},
        scheduler_utils::BoundedDirectionalTransformPropagator::Options()
            .propagateParallelType()
            .propagateToBoundary());
    smem_epilogue->axis(-1)->parallelize(ParallelType::Vectorize);

    for (auto [dc, d] : cached_outputs) {
      // Schedule output tensor differently for better global memory access
      // pattern.
      scheduleOutputTensor(
          mma_result, d, gemm_tile, mparams->supported_vec_size.epilogue);
      d->axis(-1)->parallelize(ParallelType::Vectorize);

      // Propagate output tensor transformations back to smem_epilogue
      scheduler_utils::BoundedDirectionalTransformPropagator::backward(
          d, -1, {smem_epilogue});
    }
  } else {
    for (auto [dc, d] : cached_outputs) {
      scheduler_utils::BoundedDirectionalTransformPropagator::forward(
          mma_result,
          -1,
          {d},
          scheduler_utils::BoundedDirectionalTransformPropagator::Options()
              .propagateParallelType()
              .propagateToBoundary());
      // We might propagate an inner dimension that is not compatible with the
      // output or bias-like inputs. In those cases, we will further split this
      // dimension with an outer unrolled loop to achieve the proper
      // vectorization as specified by mparams->supported_vec_size.epilogue.
      NVF_ERROR(d->axis(-1)->extent()->isConst());
      int64_t d_extent = d->axis(-1)->extent()->value().as<int64_t>();
      if (d_extent > mparams->supported_vec_size.epilogue) {
        // Should always be a divisible split
        NVF_ERROR(d_extent % mparams->supported_vec_size.epilogue == 0);
        d->split(
            -1, mparams->supported_vec_size.epilogue, /*inner_split=*/true);
        d->axis(-2)->parallelize(ParallelType::Unroll);
      }
      d->axis(-1)->parallelize(ParallelType::Vectorize);
    }
  }
  // propagate output transformations to all inputs that are part of epilogue
  //  operations, input tvs with non-core roles
  //  core roles: essential for matmul, for example mma inputs' producers
  if (has_non_mma_input_tvs) {
    scheduleFusionInputsForEpilogue(tensor_roles, mparams->use_smem_epilogue);
  }

  scheduleSplitKSum(
      splitk_sum, num_device_and_batch_dims, mparams->use_smem_epilogue);

  // auto inline for all tensors except register tensors
  inlineMost(ir_utils::allTvsExcept(fusion, {acr, bcr, ab, bb}));

  // if auto inline, will inline to position-7, leads to performance regression
  inlineSelectedAt(
      {acr, bcr, ab, bb},
      mma_result,
      num_device_and_batch_dims + 6 + num_splitk_dims);

  // Propagate mma output swizzle and parallelization down the DAG
  if (mparams->circular_buffer_options.circular_buffer_smem_write) {
    NVF_ERROR(
        mparams->circular_buffer_options.smem_circular_buffer_stage > 1,
        "Invalid buffer stage config")
    if (mparams->circular_buffer_options.smem_circular_buffer_stage > 2) {
      NVF_ERROR(
          mparams->async_gmem_load_operands,
          "Circular buffer only supports async load");
    }

    acw_smem->circularBuffer(
        mparams->circular_buffer_options.smem_circular_buffer_stage);
    bcw_smem->circularBuffer(
        mparams->circular_buffer_options.smem_circular_buffer_stage);
  }

  if (mparams->circular_buffer_options.circular_buffer_smem_read) {
    // Only apply circular buffering if we can fill the entire pipeline.
    auto safely_apply_circular_buffering = [](TensorView* tv) {
      constexpr int64_t number_of_stages = 2;
      IterDomain* cb_axis = getCircularBufferAxis(tv);
      NVF_ERROR(cb_axis != nullptr);
      NVF_ERROR(cb_axis->extent()->isConstScalar());
      if (cb_axis->extent()->evaluate() >= number_of_stages) {
        tv->circularBuffer(number_of_stages);
      }
    };
    safely_apply_circular_buffering(acr);
    safely_apply_circular_buffering(bcr);
  }

  if (mparams->circular_buffer_options.circular_buffer_smem_read &&
      mparams->circular_buffer_options.circular_buffer_smem_write) {
    // rotate Kg loop
    scheduler_utils::rotateLoop(
        mma_result,
        num_device_and_batch_dims + 2 + num_splitk_dims,
        {acr, bcr});
  }

  NVF_ERROR(!cached_outputs.empty());
  mma_utils::MmaDataTypes data_types = {
      a->dtype(), b->dtype(), mma_result->dtype()};
  // NOTE: Batch split-K matmuls cannot currently re-use smem due to outer
  // batch loop
  bool guaranteed_operand_reuse =
      num_local_batch_dims == 0 || num_splitk_dims == 0;
  int64_t estimated_smem = mma_utils::computeExpectedSharedMemoryUsage(
      mparams,
      data_types,
      /*smem_a_reuse_guaranteed=*/guaranteed_operand_reuse,
      /*smem_b_reuse_guaranteed=*/guaranteed_operand_reuse);
  fusion->setExpectedDynamicSmemBytes(estimated_smem);
}

} // namespace

bool MatmulScheduler::canScheduleCompileTime(Fusion* fusion) {
  const auto msg = matmul_utils::getMatmulCompileTimeRejectReason(fusion);
  if (!msg.empty()) {
    scheduler_debug_utils::canScheduleRejectReason(schedulerType(), msg);
    return false;
  }

  return true;
}

bool MatmulScheduler::canScheduleRunTime(
    Fusion* fusion,
    SchedulerRuntimeInfo& runtime_info,
    HeuristicDataCache* data_cache) {
  FUSER_PERF_SCOPE("MatmulScheduler::canSchedule");
  auto reason = matmul_utils::getMatmulRunTimeRejectReason(
      fusion, data_cache, runtime_info);
  if (!reason.empty()) {
    scheduler_debug_utils::canScheduleRejectReason(schedulerType(), reason);
    return false;
  }
  return true;
}

std::unique_ptr<HeuristicParams> MatmulScheduler::computeHeuristics(
    Fusion* fusion,
    SchedulerRuntimeInfo& runtime_info,
    HeuristicDataCache* data_cache) {
  auto mparams =
      matmul_utils::getMatmulHeuristics(fusion, runtime_info, data_cache);
  NVF_ERROR(mparams != nullptr);
  return mparams;
}

void MatmulScheduler::schedule(Fusion* fusion, const HeuristicParams* params) {
  FUSER_PERF_SCOPE("MatmulScheduler::schedule");
  auto mparams = dynamic_cast<const MatmulParams*>(params);
  NVF_ERROR(
      mparams != nullptr,
      "Incorrect parameters sent to MatmulScheduler::schedule",
      params);
  scheduleMatmul(fusion, mparams);
}

} // namespace nvfuser
