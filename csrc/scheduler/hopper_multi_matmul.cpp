// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#include <device_lower/analysis/circular_buffer.h>
#include <disjoint_set.h>
#include <id_model/schedule.h>
#include <instrumentation.h>
#include <ir/utils.h>
#include <scheduler/debug_utils.h>
#include <scheduler/hopper_multi_matmul.h>
#include <scheduler/matmul.h>
#include <scheduler/matmul_utils.h>
#include <scheduler/mma_utils.h>
#include <scheduler/tools/abstract_tensor.h>
#include <scheduler/tools/inlining.h>
#include <scheduler/utils.h>
#include <utils.h>
#include <val_graph.h>
#include <val_graph_visitor.h>

// NOTE: included to avoid compilation error caused by missing destructor in
// 'SchedulerRuntimeInfo'
#include <runtime/executor_utils.h>
#include "mma_type.h"

namespace nvfuser {

namespace {

// This function returns a pair of integers. The first integer is the gcd
// between megabanks and row stride. The second integer is the repeat pattern
// size. If the gcd is 1, then no swizzle is necessary to resolve bank
// conflicts. In that case, the second integer is irrelevant and -1 is returned.
std::pair<int64_t, int64_t> analyzeSwizzleSharedMemory(
    TensorView* shared_mem_tv) {
  NVF_ERROR(shared_mem_tv->getMemoryType() == MemoryType::Shared);
  AbstractTensor swizzle_domain(shared_mem_tv->getLoopDomain());

  // Check that the innermost 2 dimensions are concrete and static
  //  sized so that the swizzle function can be defined.
  NVF_ERROR(
      (int64_t)swizzle_domain.size() >= 2,
      "At least 2D input (excluding consecutive reduction domains starting from the innermost dim) needed for swizzling, but get ",
      shared_mem_tv->toString());
  mma_utils::checkConcreteStaticDim(swizzle_domain[-2]);
  mma_utils::checkConcreteStaticDim(swizzle_domain[-1]);

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
    return {g, -1}; // No need to swizzle in this case.
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
    return {g, -1}; // No need to swizzle in this case.
  }

  return {g, repeated_pattern_size};
}

//! Automatically generates the shared memory swizzled data layout for tma loads
//! in matmul mainloop. The shared memory data layout is always 2D currently.
//! This utility function assumes that the shared_mem_tv has the following
//! structure: [tile_row, tile_col]
//! Returns which swizzle format to use for mma inputs with tma loads.
MmaInputSmemSwizzle tmaSwizzleSharedMemory(TensorView* shared_mem_tv) {
  auto&& [g, repeated_pattern_size] = analyzeSwizzleSharedMemory(shared_mem_tv);

  if (g == 1) {
    return MmaInputSmemSwizzle::None; // No need to swizzle in this case.
  }

  // 128B swizzle results in 8 x 8 matrix given half precision inputs.
  constexpr int64_t n_rows = 8;

  NVF_ERROR(
      n_rows % repeated_pattern_size == 0,
      "Can not partition matrix into megarows");
  int64_t num_gigarows = n_rows / repeated_pattern_size;
  int64_t num_gigabanks = g; // also = num_megabanks / repeated_pattern_size

  /* To further simplify the problem, if we assume: */
  NVF_ERROR(
      num_gigarows % num_gigabanks == 0,
      "Requires non-square swizzle, which is not supported yet");

  AbstractTensor swizzle_domain(shared_mem_tv->getLoopDomain());
  // Extract the constant sizes of the swizzled tile
  const int64_t inner_dim_size =
      swizzle_domain[-1]->extent()->evaluate().as<int64_t>();

  auto dtype = shared_mem_tv->getDataType().value();
  const int64_t B128_elements = 128 / dataTypeSize(dtype);
  const int64_t B64_elements = 64 / dataTypeSize(dtype);
  const int64_t B32_elements = 32 / dataTypeSize(dtype);

  if (inner_dim_size % B128_elements == 0) {
    return MmaInputSmemSwizzle::B128;
  } else if (inner_dim_size % B64_elements == 0) {
    return MmaInputSmemSwizzle::B64;
  } else if (inner_dim_size % B32_elements == 0) {
    return MmaInputSmemSwizzle::B32;
  } else {
    NVF_THROW("Unsupported swizzle size for TMA shared memory mma inputs");
  }
}

//! Automatically generates the shared memory swizzled data layout for matmul
//! epilogue.
//! The shared mem data layout is always 2D currently, and this utility
//!  function assumes that the shared_mem_tv has the following structure:
//!  [tile_row, tile_col]
//! Returns the domain with swizzle. For the case of legacy swizzle, this
//! domain must be set as loop domain. For the case of new swizzle, this domain
//! must be set as allocation domain.
template <bool legacy = true>
AbstractTensor swizzleSharedMemory(TensorView* shared_mem_tv) {
  auto&& [g, repeated_pattern_size] = analyzeSwizzleSharedMemory(shared_mem_tv);

  // Create Abstract Tensor from shared memory tensor loop domain.
  AbstractTensor swizzle_domain(shared_mem_tv->getLoopDomain());

  if (g == 1) {
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

} // namespace

MatmulDimRole HopperMultipleMatmulScheduler::findMatmulDimRole(IterDomain* id) {
  ValGroup vg = graph_->toGroup(id);
  auto it = id_roles_.find(vg);
  NVF_ERROR(it != id_roles_.end());
  return it->second;
}

void HopperMultipleMatmulScheduler::run() {
  // Clears memory spaces on intermediate tensors, calls
  // cache{After,Before,Fork} on inputs and outputs
  cacheInputsAndOutputs();

  // Finds matmul patterns and translates them to MmaOps, then finds tensor
  // and dimension roles for all tensors in the fusion
  findPatterns();
  translatePatterns();
  findRoles();

  // Defines acw_smem/bcw_smem and acr/bcr by possibly calling cacheAfter.
  // This also collects mma_results_
  defineOperandCaches();

  inspectPrologues();

  scheduleOperands();

  // schedule mma instruction output (mma_result)
  scheduleMmaResults();

  // schedule epilogue
  scheduleEpilogue();

  // schedule splitk_sum
  scheduleSplitKSum();

  setUpInlining();

  // set up circular buffering. This must come after everything up to
  // mma_result is scheduled, since everything in the main loop will need to
  // be rotated
  setUpCircularBuffering();
}

void HopperMultipleMatmulScheduler::cacheInputsAndOutputs() {
  // Make sure we don't have global memory set on intermediate tensors from
  // fusion segmentation
  scheduler_utils::clearMemorySpace(fusion_);

  // Cache inputs
  scheduler_utils::cacheInputs(fusion_, /*unroll=*/true);

  // Cache and fork outputs
  scheduler_utils::cacheAndForkOutputs(fusion_, /*unroll=*/true);
}

void HopperMultipleMatmulScheduler::defineOperandCaches() {
  cacheOperandsToSmem(as_, acw_smems_);
  cacheOperandsToSmem(bs_, bcw_smems_);

  // Now that we are finished possibly redefining the inputs to the MmaOps,
  // we can set the macro for those ops
  for (TensorView* mma_result : mma_results_) {
    MmaOp* mma = dynamic_cast<MmaOp*>(mma_result->definition());
    NVF_ERROR(mma != nullptr);
    mma->setMacro(params_->mma_macro);
  }
}

void HopperMultipleMatmulScheduler::cacheOperandsToSmem(
    const std::vector<TensorView*>& operands,
    std::vector<TensorView*>& smem_operands) {
  // Use cp.async.bulk (tma) as requested in scheduler params.
  smem_operands.resize(operands.size(), nullptr);
  for (size_t i : c10::irange(operands.size())) {
    TensorView* operand = operands[i];

    NVF_ERROR(operand->uses().size() == 1);
    smem_operands[i] = ir_utils::consumerTvsOf(operand).at(0);

    LoadStoreOpType load_op = params_->async_gmem_load_operands
        ? LoadStoreOpType::CpAsyncBulkTensorTile
        : LoadStoreOpType::Set;

    smem_operands[i]->definition()->as<LoadStoreOp>()->setOpType(load_op);
    smem_operands[i]->setMemoryType(MemoryType::Shared);
  }
}

void HopperMultipleMatmulScheduler::swizzleBlockTiles(
    TensorView* tv,
    std::vector<MatmulDimRole>& outer_dim_roles) {
  if (params_->grid_swizzle_factor != 1) {
    // Find position of outer M and N dims in schedule_.tiled
    int64_t Mo_pos = -1, No_pos = -1;
    for (size_t i : c10::irange(outer_dim_roles.size())) {
      if (outer_dim_roles[i] == MatmulDimRole::M) {
        Mo_pos = (int64_t)i;
      } else if (outer_dim_roles[i] == MatmulDimRole::N) {
        No_pos = (int64_t)i;
      }
    }

    int factor = std::max(1, params_->grid_swizzle_factor); // must be >=1
    switch (params_->cta_order) {
      case MatmulParams::TileRasterizationOrder::RowMajor:
        // split   [I1, I2/factor, factor]
        // reorder [I1, factor, I2/factor]
        // merge   [I1*factor, I2/factor]
        // where I1 and I2 are the outer M and N dimensions, respectively
        if (No_pos >= 0) {
          tv->split(No_pos, factor);
          // If No_pos < Mo_pos, then the split above shifts Mo_pos by one
          if (No_pos < Mo_pos) {
            Mo_pos++;
          }
          tv->reorder({{No_pos, No_pos + 1}});
          if (Mo_pos >= 0) {
            tv->merge(Mo_pos, No_pos);
          } else {
            // M is missing, so we skip the merge above. In this case we
            // should update the dim roles to reflect the new split axis.
            outer_dim_roles.insert(
                outer_dim_roles.begin() + No_pos, MatmulDimRole::N);
          }
        }
        break;

      case MatmulParams::TileRasterizationOrder::ColumnMajor:
        // split   [I1/factor, factor, I2]
        // reorder [I1/factor, I2, factor]
        // merge   [I1/factor, I2*factor]
        // where I1 and I2 are the outer M and N dimensions, respectively
        if (Mo_pos >= 0) {
          tv->split(Mo_pos, factor);
          // If No_pos < Mo_pos, then the split above shifts Mo_pos by one
          if (No_pos > Mo_pos) {
            No_pos++;
          }
          if (No_pos >= 0) {
            tv->reorder({{Mo_pos + 1, No_pos}});
            tv->merge(Mo_pos + 1, No_pos);
          } else {
            // N is missing, so we skip the merge above. In this case we
            // should update the dim roles to reflect the new split axis.
            outer_dim_roles.insert(
                outer_dim_roles.begin() + Mo_pos, MatmulDimRole::M);
          }
        }
    }
  }
}

TensorView* HopperMultipleMatmulScheduler::cacheAfter(
    TensorView* orig,
    LoadStoreOpType op_type,
    CacheOp cache_op,
    bool propagate_allocation_domain) {
  const std::vector<IterDomain*> orig_alloc = orig->getMaybeAllocationDomain();

  TensorView* c =
      orig->cacheAfter(op_type, cache_op, propagate_allocation_domain);

  if (propagate_allocation_domain) {
    const std::vector<IterDomain*> cache_alloc = c->getMaybeAllocationDomain();
    NVF_ERROR(orig_alloc.size() == cache_alloc.size());
    for (size_t i : c10::irange(orig_alloc.size())) {
      ValGroup vg = graph_->toGroup(orig_alloc[i]);
      graph_->initializeVal(cache_alloc[i], vg);
    }
  }

  const std::vector<IterDomain*> orig_logical =
      TensorDomain::noReductions(orig->getLogicalDomain());
  const std::vector<IterDomain*> cache_logical = c->getLogicalDomain();
  // in split-K we do rFactor which gives us a full = sum(partial)
  // where partial has root domain that matches the logical domain of the
  // original tensor. The logical domain contains Iteration transforms of the
  // Reduction axis in the original mma output.
  NVF_ERROR(orig_logical.size() == cache_logical.size());
  for (size_t i : c10::irange(orig_logical.size())) {
    ValGroup vg = graph_->toGroup(orig_logical[i]);
    graph_->initializeVal(cache_logical[i], vg);
  }

  return c;
}

std::vector<std::vector<MatmulDimRole>> HopperMultipleMatmulScheduler::
    blockTileTensors(const std::vector<TensorView*>& tvs) {
  if (canonical_dim_ordering_.empty()) {
    canonical_dim_ordering_ =
        mma_utils::canonicalDimOrdering(tensor_roles_, id_roles_, *graph_);
  }

  std::vector<std::vector<MatmulDimRole>> all_merged_roles;
  for (TensorView* tv : tvs) {
    // Find dimensions in canonical_dim_ordering_ that exist in tv's loop
    // domain. Reorder those according to the canonical dim ordering then
    std::unordered_map<ValGroup, IterDomain*> tv_dims;
    std::unordered_set<MatmulDimRole> axis_roles;
    for (IterDomain* id : tv->getLoopDomain()) {
      ValGroup vg = graph_->toGroup(id);
      tv_dims.emplace(vg, id);
      // track axis roles in this tensor to use in makeTile
      auto it = id_roles_.find(vg);
      NVF_ERROR(it != id_roles_.end());
      axis_roles.insert(it->second);
    }
    std::vector<IterDomain*> new_loop;
    new_loop.reserve(tv->nDims());
    for (const ValGroup& vg : canonical_dim_ordering_) {
      auto it = tv_dims.find(vg);
      if (it != tv_dims.end()) {
        new_loop.push_back(it->second);
      }
    }
    NVF_ERROR((int64_t)new_loop.size() == tv->nDims());
    tv->setLoopDomain(new_loop);

    // There could be multiple dimensions with the same role at this point, so
    // now we collect them. After this, tv will be at most 4 dimensions e.g.
    // BMNK based on canonical_dim_ordering_, with any of these dimensions
    // possibly missing.
    mma_utils::mergeConsecutiveAxesWithSameRole(tv, id_roles_, graph_);

    // Find order the axes that are present in the merged tensor
    std::vector<MatmulDimRole> merged_roles;
    merged_roles.reserve(tv->nDims());
    for (const ValGroup& vg : canonical_dim_ordering_) {
      MatmulDimRole role = id_roles_[vg];
      if (axis_roles.count(role) != 0) {
        if (merged_roles.empty() || merged_roles.back() != role) {
          merged_roles.push_back(role);
        }
      }
    }
    NVF_ERROR(merged_roles.size() == axis_roles.size());

    // TODO: (to be pursued after the multi-matmul refactor is fully merged)
    // this currently creates a separate AbstractMatmulTensor for each
    // TensorView. Instead, we should create a single AbstractMatmulTensor
    // then apply it (with "forwarding") to each TV instead. We already cache
    // a vector<ValGroup> as canonical_dim_ordering_ so AbstractTensor
    // scheduling is the next step in this modernization.
    mma_utils::makeTile(tv, params_->tile_sizes.cta_tile, merged_roles);

    swizzleBlockTiles(tv, merged_roles);

    all_merged_roles.push_back(merged_roles);

    if (params_->splitk_factor > 1) {
      // Outer K dimension in tv is in same position found in merged_roles
      for (size_t i : c10::irange(merged_roles.size())) {
        if (merged_roles[i] == MatmulDimRole::K) {
          tv->split((int64_t)i, params_->splitk_factor, /*inner*/ false);
        }
      }
    }
  }
  return all_merged_roles;
}

void HopperMultipleMatmulScheduler::inspectPrologues() const {
  for (TensorView* mma_result : mma_results_) {
    for (Val* v : mma_result->definition()->inputs()) {
      TensorView* op_input = v->as<TensorView>();

      // We currently require all operands to lie in smem, meaning we cannot yet
      // handle any prologue computation. This includes `BroadcastOp` which
      // might be introduced when translating a MatmulOp or LinearOp to MmaOp.
      Expr* def = op_input->definition();
      NVF_ERROR(def != nullptr && def->isA<LoadStoreOp>());
      NVF_ERROR(def->input(0)->isFusionInput());
    }
  }
}

void HopperMultipleMatmulScheduler::scheduleOperands() {
  NVF_CHECK(
      params_->async_gmem_load_operands,
      "Hopper matmul scheduler currently requires TMA to be enabled");
  auto scheduleBranch = [&](const std::vector<TensorView*>& gmem_operands,
                            const std::vector<TensorView*>& smem_operands,
                            MmaOperand operand_type) {
    blockTileTensors(smem_operands);
    for (TensorView* tv : smem_operands) {
      if (params_->promote_prologue_smem_reuse) {
        tv->promoteReuse();
      }
      mma_utils::orderTiledConcreteIdAsMaybeAllocationDomain(tv);
      MmaInputSmemSwizzle swizzle_type = tmaSwizzleSharedMemory(tv);
      tv->applyMmaSwizzleForTMALoad(swizzle_type);
    }
  };
  scheduleBranch(as_, acw_smems_, MmaOperand::A);
  scheduleBranch(bs_, bcw_smems_, MmaOperand::B);
}

void HopperMultipleMatmulScheduler::parallelizeBlocks(
    const std::vector<TensorView*>& tvs) const {
  for (TensorView* tv : tvs) {
    switch (params_->cta_order) {
      // TODO: Should we instead check the roles of these dimensions to take the
      // outermost two M or N axes?
      case MatmulParams::TileRasterizationOrder::RowMajor:
        tv->axis(num_device_and_batch_dims_)->parallelize(ParallelType::BIDx);
        tv->axis(num_device_and_batch_dims_ + 1)
            ->parallelize(ParallelType::BIDy);
        break;
      case MatmulParams::TileRasterizationOrder::ColumnMajor:
        tv->axis(num_device_and_batch_dims_)->parallelize(ParallelType::BIDy);
        tv->axis(num_device_and_batch_dims_ + 1)
            ->parallelize(ParallelType::BIDx);
        break;
      default:
        NVF_THROW("Invalid TileRasterizationOrder passed to Matmul scheduler");
    }
  }
}

void HopperMultipleMatmulScheduler::scheduleMmaResults() {
  GemmTile instruction_tile = getMmaOpShape(params_->mma_macro);
  NVF_CHECK(
      params_->tile_sizes.cta_tile.k == params_->tile_sizes.warp_tile.k,
      "CTA tile must match warp tile K dimension for Hopper matmul but found ",
      toString(params_->tile_sizes));
  // If cta_tile is not divisible by instruction tile the mma instruction will
  // be predicated.
  NVF_CHECK(
      params_->tile_sizes.cta_tile.m % instruction_tile.m == 0 &&
          params_->tile_sizes.cta_tile.n % instruction_tile.n == 0 &&
          params_->tile_sizes.cta_tile.k % instruction_tile.k == 0,
      "CTA tile must be divisible by macro size but found cta_tile: ",
      toString(params_->tile_sizes.cta_tile),
      " and macro: ",
      toString(params_->mma_macro));

  // Schedule mma results and propagate forward
  auto all_merged_roles = blockTileTensors(mma_results_);
  parallelizeBlocks(mma_results_);
  for (size_t i : c10::irange(mma_results_.size())) {
    TensorView*& mma_result = mma_results_[i];
    const std::vector<MatmulDimRole>& merged_roles = all_merged_roles[i];

    // Test that mma_result logical is MNK
    // TODO: This currently checks leaf domain only which does not necessarily
    // match logical
    // TODO: Lift this constraint. Use commitLeafToLogical if necessary. We
    // might just want to match using id_roles_
    NVF_ERROR(merged_roles.size() >= 3);
    const auto checkSingleDimRole =
        [&merged_roles](int64_t pos, MatmulDimRole expected_role) {
          if (pos < 0) {
            pos += (int64_t)merged_roles.size();
          }
          NVF_ERROR(pos >= 0);
          NVF_ERROR(pos < (int64_t)merged_roles.size());
          const auto& actual_role = merged_roles[(size_t)pos];
          NVF_ERROR(actual_role == expected_role);
        };
    checkSingleDimRole(-3, MatmulDimRole::M);
    checkSingleDimRole(-2, MatmulDimRole::N);
    checkSingleDimRole(-1, MatmulDimRole::K);

    // do split-K rFactor to define splitk_sum and smem_epilogue
    if (params_->splitk_factor != 1) {
      // TODO: schedule split-K
      NVF_THROW("Hopper split-K is not yet tested");
      // Note that the split-K split is already done in blockTileTensors
      TensorView* splitk_sum = mma_result->rFactor({-4, -1});
      std::swap(splitk_sum, mma_result);
      splitk_sums_.push_back(splitk_sum);
    }

    mma_result->split(-3, getM(params_->mma_macro));
    mma_result->split(-2, getN(params_->mma_macro));
    // [Mo, No, Ko, Mio, Mii, Nio, Nii, Ki]
    // -> [Mo, No, Ko, Mio, Nio, Mii, Nii, Ki]
    mma_result->reorder({{-4, -3}});
    mma_result->merge(-5);
    mma_result->axis(-4)->parallelize(ParallelType::TIDy);

    auto s = mma_utils::MmaSwizzler::scheduleMmaOutputAllocation(
        mma_result->getLoopDomain());
    mma_result->setAllocationDomain(s.as<IterDomain*>(), true);
    mma_result->axis(-1)->parallelize(ParallelType::Mma);
    mma_result->axis(-2)->parallelize(ParallelType::Mma);
    mma_result->axis(-3)->parallelize(ParallelType::Mma);
  }
}

void HopperMultipleMatmulScheduler::scheduleOutputTensor(TensorView* c) {
  const MatMulTileOptions& gemm_tile = params_->tile_sizes;
  const int64_t vectorization_factor = params_->supported_vec_size.epilogue;
  // input tensor is in the form of [Mo,No,cta_tile_m,cta_tile_n]
  mma_utils::checkConcreteStaticDim(c->axis(-2));
  mma_utils::checkConcreteStaticDim(c->axis(-1));
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
  // We have fixed tidx, tidy, and tidz, so we need to make sure that the
  // output tensor is divisible by tidx * tidy * tidz * vectorization_factor
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

  for (TensorView* mma_result : mma_results_) {
    // step-5, Parallel first 2 dims same as mma_result
    scheduler_utils::parallelizeAllLike(
        mma_result,
        2,
        {c},
        {ParallelType::BIDx, ParallelType::BIDy, ParallelType::BIDz});
  }
}

void HopperMultipleMatmulScheduler::scheduleEpilogue() {
  // TODO: schedule epilogue by propagation backward from dc
  if (!params_->use_smem_epilogue) {
    for (Val* dv : fusion_->outputs()) {
      auto* d = dv->as<TensorView>();
      NVF_ERROR(d->definition() && d->definition()->isA<LoadStoreOp>());
      auto* dc = d->definition()->input(0)->as<TensorView>();

      // Block Schedule and Parallelize
      blockTileTensors({dc, d});
      parallelizeBlocks({dc, d});

      // Apply mma common transformation
      for (auto tv : {dc, d}) {
        // [..., Mo, No, Mi, Ni]
        tv->split(-2, getM(params_->mma_macro));
        tv->split(-1, getN(params_->mma_macro));
        // [..., Mo, No, Mio, Mii, Nio, Nii]
        // -> [..., Mo, No, Mio, Nio, Mii, Nii]
        tv->reorder({{-3, -2}});
        tv->merge(-4);
        auto s = mma_utils::MmaSwizzler::scheduleMmaOutputAllocation(
            tv->getLoopDomain());
        tv->setLoopDomain(s.as<IterDomain*>());
        tv->axis(-5)->parallelize(ParallelType::TIDy);
      }
      d->axis(-1)->parallelize(ParallelType::Vectorize);
    }
  } else {
    constexpr int64_t stmatrix_tile_m = 16;
    constexpr int64_t stmatrix_tile_n = 16;

    // TODO: Support tma tile sizes that are a multiple of mma_macro.
    // The wgmma operation creates an output matrix of mma_macro size. The TMA
    // tile is a multiple of the macro size because stmatrix stores results from
    // wgmma to shared memory. For maximum inlining and to reduce shared memory
    // usage, the tma tile is mma_macro size.
    const int64_t tma_m = getM(params_->mma_macro);
    const int64_t tma_n = getN(params_->mma_macro);

    fusion_->manage("st_matrix_m_tile", stmatrix_tile_m);
    fusion_->manage("st_matrix_n_tile", stmatrix_tile_n);
    fusion_->manage("st_matrix_m", tma_m);
    fusion_->manage("st_matrix_n", tma_n);

    // Manually schedule register cache and output TensorView
    for (Val* dv : fusion_->outputs()) {
      auto* d = dv->as<TensorView>();
      NVF_ERROR(d->definition() && d->definition()->isA<LoadStoreOp>());
      auto* dc = d->definition()->input(0)->as<TensorView>();

      // NOTE: cacheBefore does not work with blockTileTensors
      TensorView* d_smem = cacheAfter(dc, LoadStoreOpType::Set);

      // Set MemoryType
      dc->setMemoryType(MemoryType::Local);
      d_smem->setMemoryType(MemoryType::Shared);

      // Set LoadStoreOp
      d_smem->definition()->as<LoadStoreOp>()->setOpType(
          LoadStoreOpType::StMatrix);
      d->definition()->as<LoadStoreOp>()->setOpType(
          LoadStoreOpType::CpAsyncBulkTensorTile);

      // Block Schedule and Parallelize
      blockTileTensors({dc, d_smem, d});
      parallelizeBlocks({dc, d_smem, d});

      // Apply mma common transformation
      for (auto tv : {dc, d_smem, d}) {
        // Original: [..., Mo, No, Mi, Ni]
        tv->split(-2, getM(params_->mma_macro));
        tv->split(-1, getN(params_->mma_macro));
        // After Split: [..., Mo, No, Mio, Mii, Nio, Nii]
        tv->reorder({{-3, -2}});
        // After Reorder: [..., Mo, No, Mio, Nio, Mii, Nii]
        tv->merge(-4);
        // After Merge: [..., Mo, No, Mio * Nio, Mii, Nii]
        tv->axis(-3)->parallelize(ParallelType::TIDy);
        // After Parallelize: [..., Mo, No, Mio * Nio (TIDy), Mii, Nii]
      }

      // Schedule register cache; Output from epilogue
      {
        auto s = mma_utils::MmaSwizzler::scheduleMmaOutputAllocation(
            dc->getLoopDomain());
        dc->setLoopDomain(s.as<IterDomain*>());
        dc->setAllocationDomain(s.as<IterDomain*>(), true);
      }

      MmaInputSmemSwizzle swizzle = tmaSwizzleSharedMemory(d_smem);

      // Schedule shared memory cache; Output from StMatrix
      scheduleStMatrixForMmaOutput(
          d_smem, swizzle, stmatrix_tile_m, stmatrix_tile_n);

      // Schedule global memory output; Output from TMA Store
      scheduleTMAStoreForMmaOutput(d, swizzle);
    }
  }
}

//! Propagates transformations from fusion output to fusion tv inputs that are
//!  producers in the epilogue. Transformations' propagation aims at input tvs
//!  which are not assigned to core roles, that is, are not MMA inputs.
void HopperMultipleMatmulScheduler::scheduleFusionInputsForEpilogue() {
  std::vector<TensorView*> cached_tvs;

  // Handling transformations in fusion input tvs with assigned EPILOGUE_INPUT
  //  role by propagating fusion output transformations through cached views
  //  of EPILOGUE_INPUT fusion input tvs and by setting vectorization of the
  //  inner most iterdomain of these cached views
  if (tensor_roles_.count(MatmulTensorRole::EPILOGUE_INPUT)) {
    auto& c_tvs = tensor_roles_.at(MatmulTensorRole::EPILOGUE_INPUT);

    // The system supports only scenario where there is only one fusion output
    //  with assigned OUTPUT role, this condition is already verified so there
    //  is no need for an additional checks here
    auto output_d = tensor_roles_.at(MatmulTensorRole::OUTPUT).front();
    for (auto* c : c_tvs) {
      cached_tvs.push_back(c->cacheAfter());
    }

    scheduler_utils::BoundedDirectionalTransformPropagator::backward(
        output_d, -1, c_tvs);

    std::unordered_set<ParallelType> parallel_types = {};
    if (params_->use_smem_epilogue) {
      // In cases where smem epilogue feature is enabled, the vectorization
      //  of domains will be propagated to fusion inputs that are epilogue
      //  inputs, this may result in unaligned memory reads. Vectorization is
      //  explicitly excluded form parallelization types to avoid this issue.
      // This should be changed when vectorization analysis is available and
      //  enabled for matmul scheduler.
      parallel_types = allParallelTypesExcept({ParallelType::Vectorize});
    }
    scheduler_utils::parallelizeAllLike(
        output_d, -1, cached_tvs, parallel_types);

    // The cached EPILOGUE_INPUT tvs are not needed anymore
    cached_tvs.clear();
  }
}

void HopperMultipleMatmulScheduler::scheduleSplitKSum() {
  if (params_->splitk_factor == 1) {
    return;
  }
  NVF_THROW("Split-K scheduling is not yet implemented for Hopper matmul");
  for (TensorView* splitk_sum : splitk_sums_) {
    // Always use serial grid reduction for split-K sum
    splitk_sum->definition()->as<ReductionOp>()->requestSerialGridReduction();

    if (params_->use_smem_epilogue) {
      // Now that transforms are propagated backward to smem_epilogue, which
      // is before splitk_sum, we can vectorize the inner-most non-trivial
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
        // using Float for this reduction and Half for output. So here we
        // first check whether the vectorize size is at most 16 bytes. If not,
        // then we split into an unrolled loop that will do multiple
        // vectorized reads/writes instead. Note that we reorder such that the
        // axes are in order UR TIDx V.
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
}

void HopperMultipleMatmulScheduler::setUpInlining() {
  // auto inline for all tensors except register tensors
  std::unordered_set<TensorView*> smem_loads_and_mma_inputs;
  smem_loads_and_mma_inputs.insert(acrs_.begin(), acrs_.end());
  smem_loads_and_mma_inputs.insert(bcrs_.begin(), bcrs_.end());
  smem_loads_and_mma_inputs.insert(abs_.begin(), abs_.end());
  smem_loads_and_mma_inputs.insert(bbs_.begin(), bbs_.end());
  inlineMost(ir_utils::allTvsExcept(fusion_, smem_loads_and_mma_inputs));

  // if auto inline, will inline to position-7, leads to performance
  // regression
  for (TensorView* mma_result : mma_results_) {
    inlineSelectedAt(
        smem_loads_and_mma_inputs,
        mma_result,
        num_device_and_batch_dims_ + 6 + num_splitk_dims_);
  }
}

void HopperMultipleMatmulScheduler::setUpCircularBuffering() {
  // Propagate mma output swizzle and parallelization down the DAG
  if (params_->circular_buffer_options.circular_buffer_smem_write) {
    NVF_ERROR(
        params_->circular_buffer_options.smem_circular_buffer_stage > 1,
        "Invalid buffer stage config")
    if (params_->circular_buffer_options.smem_circular_buffer_stage > 2) {
      NVF_ERROR(
          params_->async_gmem_load_operands,
          "Circular buffer only supports async load");
    }

    for (TensorView* acw_smem : acw_smems_) {
      acw_smem->circularBuffer(
          params_->circular_buffer_options.smem_circular_buffer_stage,
          /*prefetch_distance=*/
          params_->circular_buffer_options.smem_circular_buffer_stage - 1);
    }
    for (TensorView* bcw_smem : bcw_smems_) {
      bcw_smem->circularBuffer(
          params_->circular_buffer_options.smem_circular_buffer_stage,
          /*prefetch_distance=*/
          params_->circular_buffer_options.smem_circular_buffer_stage - 1);
    }
  }

  // NOTE: circular_buffer_smem_read is ignored for Hopper matmul since we do
  // not do any cache reads

  /*
  // TODO Investigate. Disable loop rotation with tma circular buffering
  if (params_->circular_buffer_options.circular_buffer_smem_read &&
      params_->circular_buffer_options.circular_buffer_smem_write) {
    // rotate Kg loop
    // This assumes we have a single main loop. If there were multiple main
    // loops, then we would need to rotate each of them separately.
    std::unordered_set<Statement*> all_smem_loads;
    all_smem_loads.insert(acrs_.begin(), acrs_.end());
    all_smem_loads.insert(bcrs_.begin(), bcrs_.end());
    scheduler_utils::rotateLoop(
        mma_results_.front(),
        num_device_and_batch_dims_ + 2 + num_splitk_dims_,
        all_smem_loads);
  }
  */
}

void HopperMultipleMatmulScheduler::scheduleStMatrixForMmaOutput(
    TensorView* tv,
    MmaInputSmemSwizzle swizzle,
    int64_t tile_m,
    int64_t tile_n) {
  NVF_ERROR(
      ((tile_m == 16 && tile_n == 16) || (tile_m == 16 && tile_n == 8)),
      "We only support 16x16 and 16x16 stmatrix now");

  NVF_ERROR(
      tv->dtype() == DataType::Half, "we only support half type in stmatrix");

  // [M, N] -> [128(TIDx), N/8 ,  2 , 2]
  auto s =
      mma_utils::MmaSwizzler::scheduleMmaOutputAllocation(tv->getLoopDomain());

  if (swizzle != MmaInputSmemSwizzle::None) {
    // Create tma store allocation domain with swizzle
    scheduleTMAStoreForMmaOutput(tv, swizzle);
  }

  tv->setLoopDomain(s.as<IterDomain*>());

  if (tile_m == 16 && tile_n == 16) {
    // Let [M, N] be [64, 32]
    // After scheduleMmaOutputAllocation: [128(TIDx), 4, 2, 2]
    // [128(TIDx), 4(n), 2, 2] ->  [128(TIDx), 2(no), 2(ni), 2, 2]
    tv->split(-3, 2);
    // [128(TIDx), 2(no), 2(ni), 2, 2] -> [2(no), 128(TIDx), 2(ni), 2, 2]
    tv->reorder({{-4, 0}});
    // [128(TIDx), 2(no), 2(ni), 2, 2] -> [2(no), 128(TIDx), 8 (vectorize)]
    tv->merge(-3);
    tv->merge(-2);
  } else if (tile_m == 16 && tile_n == 8) {
    // Let [M, N] be [64, 16]
    // After scheduleMmaOutputAllocation: [128(TIDx), 2, 2, 2]
    // [128(TIDx), 2, 2, 2] -> [2, 128(TIDx), 2, 2]
    tv->reorder({{-3, 0}});
    // [2, 128(TIDx), 2, 2] -> [2, 128(TIDx), 4(vectorize)]
    tv->merge(-2);
  }
  tv->axis(-1)->parallelize(ParallelType::Vectorize);
}

void HopperMultipleMatmulScheduler::scheduleTMAStoreForMmaOutput(
    TensorView* tv,
    MmaInputSmemSwizzle swizzle) {
  // [BDX, BDY, TDY, MI, NI]
  // skip all but last 2 iterDomains
  int64_t num_ids_to_skip =
      static_cast<int64_t>(tv->getLoopDomain().size() - 2);

  NVF_ERROR(num_ids_to_skip >= 0);
  if (swizzle == MmaInputSmemSwizzle::None) {
    // For no-swizzle case, the entire tile are divided into 8x8 core matrices,
    // and each core matrix resides in a contiguous 8*8*2 bytes region in shared
    // memory. [K, N]
    tv->split(-2, 8);
    tv->split(-1, 8);
    // [Ko, K8, No, N8]
    tv->reorder({{-2, -3}});
  } else {
    auto dtype = tv->getDataType().value();

    // In the comments below I assume K=16, N=32, swizzle=32, dtype = half.

    // split the inner-dim
    // [K(16), N(32)] -> [K(16), NO(2), NI(16)]
    tv->split(-1, getBytesFromSwizzle(swizzle) / dataTypeSize(dtype));

    // [NO, K, NI] - the TMA Box is [K, NI]
    tv->reorder({{-2, -3}});

    // [NO, K, NI] ->
    // [NO, KO(2), KIO(2), KII(4), NIO(2), NII(8)]
    tv->swizzleTMABox(swizzle);
    num_ids_to_skip += 1;
  }

  // The shared memory producer must have the swizzled allocation domain.
  // The global memory consumer must have the ParallelType::Bulk iterDomains.
  if (tv->getMemoryType() == MemoryType::Shared) {
    // Set the allocation to the loop domain.
    tv->setAllocationDomain(tv->getLoopDomain(), true);
  } else {
    mma_utils::MmaSwizzler::parallelizeAsBulkSkippingFirstIDs(
        tv, num_ids_to_skip);
  }
}

} // namespace nvfuser
