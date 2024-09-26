// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#include <abstract_tensor.h>
#include <disjoint_set.h>
#include <id_model/schedule.h>
#include <inlining.h>
#include <instrumentation.h>
#include <ir/utils.h>
#include <multidevice/utils.h>
#include <scheduler/debug_utils.h>
#include <scheduler/matmul.h>
#include <scheduler/matmul_utils.h>
#include <scheduler/mma_utils.h>
#include <scheduler/utils.h>
#include <val_graph.h>
#include <val_graph_visitor.h>

// NOTE: included to avoid compilation error caused by missing destructor in
// 'SchedulerRuntimeInfo'
#include <fusion_executor/executor_utils.h>
#include "mma_type.h"

namespace nvfuser {

namespace {

// Returns true if given number is power of 2
constexpr bool isPowOf2(int64_t x) {
  return x > 1 && (x & (x - 1)) == 0;
}

inline IterDomain* representativeId(const AbstractId& abs_id) {
  if (abs_id.is<IterDomain*>()) {
    return abs_id.as<IterDomain*>();
  }
  NVF_ERROR(abs_id.is<ValGroupAndItsGraph>());
  return representativeId(abs_id.as<ValGroupAndItsGraph>().group);
}

// Utility to check concrete static size
inline void checkConcreteStaticDim(const AbstractId& abs_id) {
  IterDomain* id = representativeId(abs_id);
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
  checkConcreteStaticDim(swizzle_domain[-2]);
  checkConcreteStaticDim(swizzle_domain[-1]);

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
// MultipleMatmulScheduler::schedules_.
// TODO: Inheret from SchedulerEntry
class MultipleMatmulScheduler {
 public:
  MultipleMatmulScheduler(Fusion* fusion, const MatmulParams* params)
      : fusion_(fusion),
        params_(params),
        id_model_(fusion, /*build_graphs=*/false) {}

  void run() {
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

    // Schedules:
    //   - global->smem (cp.async)
    //   - smem->register (ldmatrix)
    //   - prologue computation in registers, including broadcast to e.g.
    //   ab=[iM, bN, iK]
    schedulePrologues();

    // TODO: Remove this as the methods below are implemented
    return;

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

 private:
  void cacheInputsAndOutputs() {
    // Make sure we don't have global memory set on intermediate tensors from
    // fusion segmentation
    scheduler_utils::clearMemorySpace(fusion_);

    // Cache inputs
    scheduler_utils::cacheInputs(fusion_, /*unroll=*/true);

    // Cache and fork outputs
    cached_outputs_ =
        scheduler_utils::cacheAndForkOutputs(fusion_, /*unroll=*/true);
  }

  void findPatterns() {
    patterns_ = mma_utils::findMatmulPatterns(fusion_);
    NVF_ERROR(!patterns_.empty(), "No matmul patterns were found");
  }

  void countDims() {
    NVF_ERROR(!patterns_.empty());
    TensorView* mma_result = patterns_.front().output;
    num_device_dims_ = numDeviceDims(mma_result);
    for (const auto& it : id_roles_) {
      if (it.second == MatmulDimRole::Batch) {
        // All batch dims will be merged into one, if any exist
        num_local_batch_dims_ = 1;
      }
    }
    num_splitk_dims_ = params_->splitk_factor > 1 ? 1 : 0;
    // Subtract 6 for the [Mo, No, Ko, Mi, Ni, Ki]
    num_device_and_batch_dims_ = num_device_dims_ + num_local_batch_dims_;
  }

  void translatePatterns() {
    mma_results_.reserve(patterns_.size());
    for (mma_utils::MatmulPattern& pattern : patterns_) {
      MmaOp* mma = pattern.translateToMmaOp();
      mma_results_.push_back(mma->out()->as<TensorView>());
    }

    // Build IdModel graphs now since translateToMmaOp creates new TVs. Before
    // this point the graphs are not yet built.
    updateIdModel();
  }

  // Get tensor roles and id roles
  // When there are multiple matmul patterns, we can have conflicting roles.
  // For now we throw an error if this is the case.
  // TODO: This should be checked in canScheduleCompileTime
  void findRoles() {
    const auto roles_opt = mma_utils::allPatternRoles(id_model_, patterns_);
    NVF_ERROR(
        roles_opt.has_value(),
        "Incompatible roles found between matmul patterns");
    std::tie(id_roles_, tensor_roles_) = roles_opt.value();

    mma_utils::MatmulOperandInnerDimsOpt inner_dims_opt =
        mma_utils::getOperandInnerDims(id_model_, id_roles_, tensor_roles_);
    NVF_ERROR(inner_dims_opt.isValid(), inner_dims_opt.getErrorMsg());
    inner_dims_ = inner_dims_opt.getData();

    as_ = tensor_roles_.at(MatmulTensorRole::OPERAND_A);
    bs_ = tensor_roles_.at(MatmulTensorRole::OPERAND_B);

    countDims();
  }

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
  void defineOperandCaches() {
    cacheOperandsToSmem(as_, acw_smems_, params_->supported_vec_size.a);
    addSetsForCacheReads(acw_smems_, acrs_);

    cacheOperandsToSmem(bs_, bcw_smems_, params_->supported_vec_size.b);
    addSetsForCacheReads(bcw_smems_, bcrs_);

    // Now that we are finished possibly redefining the inputs to the MmaOps,
    // we can set the macro for those ops
    for (TensorView* mma_result : mma_results_) {
      MmaOp* mma = dynamic_cast<MmaOp*>(mma_result->definition());
      NVF_ERROR(mma != nullptr);
      mma->setMacro(params_->mma_macro);
    }
  }

  void cacheOperandsToSmem(
      const std::vector<TensorView*>& operands,
      std::vector<TensorView*>& smem_operands,
      int64_t vec_size) {
    // Use cp.async as requested in scheduler params.
    smem_operands.resize(operands.size(), nullptr);
    for (size_t i : c10::irange(operands.size())) {
      TensorView* operand = operands[i];
      CacheOp cache_op = CacheOp::Unspecified;
      if (params_->async_gmem_load_operands) {
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
        cache_op = vec_bytes == 16LL ? CacheOp::Global : CacheOp::AllLevels;
      };

      NVF_ERROR(operand->uses().size() == 1);
      smem_operands[i] = ir_utils::consumerTvsOf(operand).at(0);

      LoadStoreOpType load_op = params_->async_gmem_load_operands
          ? LoadStoreOpType::CpAsync
          : LoadStoreOpType::Set;

      smem_operands[i]->definition()->as<LoadStoreOp>()->setOpType(load_op);
      smem_operands[i]->definition()->as<LoadStoreOp>()->setCacheOp(cache_op);
      smem_operands[i]->setMemoryType(MemoryType::Shared);
    }
  }

  // We add two LoadStore operators to the inputs of our fusions. The first
  // one is for a read from global memory and the second one (below) is for a
  // cache read. As an optimizaton, we avoid adding an operator if there's an
  // existing LoadStoreOp present. Please note that for the second LoadStore
  // we don't propagate the allocation domain, since the scheduler sets the
  // allocation domain in the registers.
  void addSetsForCacheReads(
      const std::vector<TensorView*>& tv_smems,
      std::vector<TensorView*>& tv_rs) {
    tv_rs.resize(tv_smems.size(), nullptr);
    for (size_t i : c10::irange(tv_smems.size())) {
      TensorView* tv_smem = tv_smems[i];
      TensorView*& tv_r = tv_rs[i];

      // There can be multiple uses for example if we have A @ B1 + A @ B2
      // then A will be cached to smem then it might be loaded into two
      // separate register buffers, one for each mma. Instead, we will load
      // it once into registers then re-use the register buffer for both
      // mmas.
      if (auto ldst = dynamic_cast<LoadStoreOp*>(tv_smem->uses().at(0));
          ldst && tv_smem->uses().size() == 1) {
        tv_r = ldst->out()->as<TensorView>();
        ldst->setOpType(LoadStoreOpType::LdMatrix);
      } else {
        tv_r = cacheAfter(
            tv_smem,
            LoadStoreOpType::LdMatrix,
            CacheOp::Unspecified,
            /*propagate_allocation_domain=*/false);
      }
    }
  }

  //! Rebuilds IdModel, then updates all ValGroups in abstract tensors to refer
  //! to the new IdModel. This is necessary whenever we perform an operation
  //! that creates a new TensorView, such as caching or rFactor
  void updateIdModel() {
    // Build new IdModel
    IdModel new_id_model(fusion_, /*build_graphs=*/false);
    new_id_model.buildPermissiveGraph();

    // Get new permissive graph
    ValGraph& new_graph = new_id_model.idGraph(IdMappingMode::PERMISSIVE);

    if (!id_roles_.empty()) {
      // Update id_roles_ to have keys corresponding to ValGroups in the new
      // IdModel
      std::unordered_map<ValGroup, MatmulDimRole> new_id_roles;
      for (auto& [k, v] : id_roles_) {
        const ValGroup& new_group = new_graph.toGroup(k->front());
        new_id_roles.emplace(new_group, v);
      }
      id_roles_ = new_id_roles;
    }

    graph_ = &new_id_model.idGraph(IdMappingMode::PERMISSIVE);

    // Set id_model_ after we are done using the old one
    id_model_ = std::move(new_id_model);
  }

  //! Swizzle the M and N outer dimensions after makeTile has been called.
  //! This updates outer_dim_roles if we introduce a new dimension, which can
  //! happen if tv is missing a merged axis, in which case we skip merging after
  //! the split. This is analogous to forwarding during transform propagation.
  void swizzleBlockTiles(
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

  //! This calls orig->cacheAfter() and also updates the permissive graph to
  //! reflect the new IterDomain mappings
  TensorView* cacheAfter(
      TensorView* orig,
      LoadStoreOpType op_type = LoadStoreOpType::Set,
      CacheOp cache_op = CacheOp::AllLevels,
      bool propagate_allocation_domain = false) {
    const std::vector<IterDomain*> orig_alloc =
        orig->getMaybeAllocationDomain();

    TensorView* c =
        orig->cacheAfter(op_type, cache_op, propagate_allocation_domain);

    if (propagate_allocation_domain) {
      const std::vector<IterDomain*> cache_alloc =
          c->getMaybeAllocationDomain();
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
      const std::vector<TensorView*>& tvs) {
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

  //! Schedule the loads of all operands from global memory to shared memory.
  //! Starting from the basic tiled schedule, we swizzle the operand memory.
  //! Note that the cache op and LoadStoreOpType are already set during
  //! defineOperandCaches().
  void scheduleOperandSmemStores() {
    auto scheduleBranch = [&](const std::vector<TensorView*>& gmem_operands,
                              const std::vector<TensorView*>& smem_operands,
                              const int64_t vec_size) {
      blockTileTensors(smem_operands);
      for (TensorView* tv : smem_operands) {
        if (params_->promote_prologue_smem_reuse) {
          tv->promoteReuse();
        }
        mma_utils::orderTiledConcreteIdAsMaybeAllocationDomain(tv);
        auto swizzled_dom = swizzleSharedMemory(tv);
        tv->setLoopDomain(swizzled_dom.as<IterDomain*>());
        tv->setHasSwizzleOp();
        tv->merge(-2);
        // NOTE: this splits and parallelizes the inner dimension as
        //   TIDz, TIDy, TIDx, V
        mma_utils::scheduleContiguousVectorLoad(
            tv, params_->tile_sizes, vec_size, /*vectorize=*/vec_size > 1);
      }
    };
    scheduleBranch(as_, acw_smems_, params_->supported_vec_size.a);
    scheduleBranch(bs_, bcw_smems_, params_->supported_vec_size.b);
  }

  void scheduleMmaOperands(
      std::vector<TensorView*>& tvs,
      const std::optional<MmaOperand> operand_type) {
    auto all_merged_roles = blockTileTensors(tvs);
    for (size_t i : c10::irange(tvs.size())) {
      TensorView*& operand = tvs[i];
      std::vector<MatmulDimRole>& merged_roles = all_merged_roles[i];

      // At this point we have the following schedule:
      //   No split-K
      //     mma_result      [..., iMo, iNo, rKo, iMi, iNi, rKi]
      //   Split-K
      //     mma_result      [..., iMo, iNo, iKf, rKg, iMi, iNi, rKi]
      //     splitk_sum      [..., iMo, iNo, rKf, iMi, iNi]

      // Schedule warp tile
      // Incoming mma_result = [... iMo iNo (iKf) rKg iMi iNi rKi]

      if (params_->use_smem_epilogue && params_->splitk_factor != 1) {
        // TODO:
        // This is a workaround for a problem that different dimensions in the
        // loop domain are mapped in the loop graph of IdModel due to the
        // mapping of compliment IDs. We should remove forwarding completely,
        // and remove this workaround.
        operand->split(-2, 1);
        operand->merge(-3);
      }

      // NOTE: this applies to either mma_result _or_ ab/bb since both have the
      // same number of dimensions.
      // TODO: use the version that uses merged_roles instead here
      mma_utils::scheduleWarpTileWithReduction(operand, params_->tile_sizes);

      // parallelize Mwo, Nwo by thread
      operand->axis((int64_t)merged_roles.size() + num_splitk_dims_ + 1)
          ->parallelize(ParallelType::TIDz);
      operand->axis((int64_t)merged_roles.size() + num_splitk_dims_ + 2)
          ->parallelize(ParallelType::TIDy);
    }
  }

  // MmaOperand contains only A and B. If tvs are outputs (i.e. not operands),
  // then operand_type should be std::nullopt.
  void scheduleMmaResults() {
    NVF_THROW("scheduleMmaResults is not yet implemented");
  }

  void schedulePrologues() {
    // schedule all transfers from gmem to smem (acw_smems_ and bcw_smems_)
    scheduleOperandSmemStores();

    // Hold this vector so we can use it as a boundary to propagate backward
    // from each mma input.
    std::vector<TensorView*> all_smem_stores = acw_smems_;
    all_smem_stores.insert(
        all_smem_stores.end(), bcw_smems_.begin(), bcw_smems_.end());

    // Now for each operand, we load from smem to registers and compute a
    // prologue (generally) in registers. We typically refer to the register
    // buffer that is loaded from operand A's smem buffer using ldmatrix as
    // "acr". This is the beginning of the register prologue region for that
    // operand. The end of that region is the first input to the MmaOp
    // expression, which we typically refer to as "ab". There is some special
    // handling of acr but otherwise we schedule ab and propagate backward
    // along this prologue region.
    auto schedulePrologueBranch = [&](const std::vector<TensorView*>&
                                          smem_stores,
                                      const std::vector<TensorView*>&
                                          smem_loads,
                                      std::vector<TensorView*>& mma_inputs,
                                      MmaOperand operand_type) {
      NVF_ERROR(smem_stores.size() == smem_loads.size());
      // TODO: we should not assume that each operand is used in only a single
      // mma op
      NVF_ERROR(mma_results_.size() >= smem_loads.size());
      // We will save abs_ and bbs_ here for later use
      // TODO: save all register prologue tensors instead to a new vector called
      // prologue_register_tensors_
      NVF_ERROR(mma_inputs.empty());
      for (TensorView* mma_result : mma_results_) {
        MmaOp* mma = dynamic_cast<MmaOp*>(mma_result->definition());
        NVF_ERROR(mma != nullptr);
        TensorView* mma_input = nullptr;
        if (operand_type == MmaOperand::A) {
          mma_input = mma->inA()->as<TensorView>();
        } else if (operand_type == MmaOperand::B) {
          mma_input = mma->inB()->as<TensorView>();
        }
        NVF_ERROR(mma_input != nullptr);
        mma_inputs.push_back(mma_input);
      }

      scheduleMmaOperands(mma_inputs, operand_type);

      // Propagate backward from all mma_results to smem_stores

      for (TensorView* mma_input : mma_inputs) {
        // Schedule mma_input, since we know it has the broadcast dimension M or
        // N, whereas the smem read might not
        moveInnerBroadcastLeft(mma_input);
        mma_input->applyMmaSwizzle(operand_type);
        scheduler_utils::BoundedDirectionalTransformPropagator::backward(
            mma_input,
            -1,
            smem_stores,
            scheduler_utils::BoundedDirectionalTransformPropagator::Options()
                .propagateParallelType());
      }
      // Find smem loads that are mma inputs and save them
      std::unordered_set<TensorView*> smem_load_mma_inputs;
      for (TensorView* smem_load : smem_loads) {
        // Insert only if smem_load is also in mma_inputs
        bool is_mma_input =
            std::find(mma_inputs.begin(), mma_inputs.end(), smem_load) !=
            mma_inputs.end();
        if (is_mma_input) {
          smem_load_mma_inputs.insert(smem_load);
        }
        if (!is_mma_input) {
          //  -5  -4   -3   -2   -1
          //[8mi, 4k, 2ko, 2mo, 2ki]
          smem_load->setAllocationDomain(smem_load->getLoopDomain(), true);
          mma_utils::MmaSwizzler::scheduleLdMatrix(smem_load, operand_type);
        }
      }
      for (TensorView* mma_input : mma_inputs) {
        if (smem_load_mma_inputs.count(mma_input) == 0) {
          mma_input->merge(-5);
          mma_input->axis(-4)->parallelize(ParallelType::TIDx);
          scheduler_utils::BoundedDirectionalTransformPropagator::backward(
              mma_input,
              -1,
              smem_loads,
              scheduler_utils::BoundedDirectionalTransformPropagator::Options()
                  .propagateParallelType());
        }
      }
    };
    schedulePrologueBranch(acw_smems_, acrs_, abs_, MmaOperand::A);
    schedulePrologueBranch(bcw_smems_, bcrs_, bbs_, MmaOperand::B);
  }

  void scheduleEpilogue() {
    NVF_THROW("scheduleEpilogue is not yet implemented");
  }

  void scheduleSplitKSum() {
    NVF_THROW("scheduleSplitKSum is not yet implemented");
  }

  void setUpInlining() {
    NVF_THROW("setUpInlining is not yet implemented");
  }

  // NOTE: this should be called after acw_smem, acr, ..., ab, and mma_result
  // transforms have been applied and inlining
  void setUpCircularBuffering() {
    NVF_THROW("setUpCircularBuffering is not yet implemented");
  }

 private:
  Fusion* fusion_;
  const MatmulParams* params_;
  IdModel id_model_;
  // Permissive graph of id_model_, which we modify at times using e.g.
  // AbstractTensor.split or by mapping vals in cacheAfter and rFactor
  ValGraph* graph_ = nullptr;
  std::vector<mma_utils::MatmulPattern> patterns_;
  mma_utils::DimRolesMap id_roles_;
  mma_utils::TensorRolesMap tensor_roles_;
  mma_utils::MatmulOperandInnerDims inner_dims_;

  int64_t num_splitk_dims_ = 0, num_device_dims_ = 0, num_local_batch_dims_ = 0,
          num_device_and_batch_dims_ = 0;

  std::vector<std::pair<TensorView*, TensorView*>> cached_outputs_;

  std::vector<ValGroup> canonical_dim_ordering_;

  std::vector<TensorView*> as_, bs_, acw_smems_, bcw_smems_, acrs_, bcrs_, abs_,
      bbs_, mma_results_, splitk_sums_, smem_epilogues_;
};

} // namespace

void scheduleMultipleMatmuls(Fusion* fusion, const MatmulParams* params) {
  FusionGuard fg(fusion);

  MultipleMatmulScheduler(fusion, params).run();
}

} // namespace nvfuser
