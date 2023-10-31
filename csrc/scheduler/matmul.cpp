// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#include <inlining.h>
#include <instrumentation.h>
#include <scheduler/debug_utils.h>
#include <scheduler/matmul.h>
#include <scheduler/matmul_utils.h>
#include <scheduler/mma_utils.h>
#include <scheduler/utils.h>

// NOTE: included to avoid compilation error caused by missing destructor in
// 'SchedulerRuntimeInfo'
#include <executor_utils.h>
#include "mma_type.h"

namespace nvfuser {

MatmulScheduler::MatmulScheduler(
    Fusion* fusion,
    SchedulerRuntimeInfo& runtime_info,
    HeuristicSummary* data_cache)
    : SchedulerEntry(heuristicType()) {
  computeHeuristics(fusion, runtime_info);
}

void MatmulScheduler::schedule(Fusion* fusion) {
  FUSER_PERF_SCOPE("Schedule Matmul Fusion");
  scheduleMatmul(fusion, matmulParams());
}

bool MatmulScheduler::canScheduleCompileTime(Fusion* fusion) {
  const auto msg = getMatmulCompileTimeRejectReason(fusion);
  if (!msg.empty()) {
    scheduler_debug_utils::canScheduleRejectReason(heuristicType(), msg);
    return false;
  }

  return true;
}

bool MatmulScheduler::canScheduleRunTime(
    Fusion* fusion,
    SchedulerRuntimeInfo& runtime_info,
    HeuristicSummary* data_cache) {
  FUSER_PERF_SCOPE("MatmulScheduler::canSchedule");
  auto reason = getMatmulRunTimeRejectReason(fusion, data_cache, runtime_info);
  if (!reason.empty()) {
    scheduler_debug_utils::canScheduleRejectReason(heuristicType(), reason);
    return false;
  }
  return true;
}

void MatmulScheduler::computeHeuristics(
    Fusion* fusion,
    SchedulerRuntimeInfo& runtime_info,
    HeuristicSummary* data_cache) {
  params_ = getMatmulHeuristics(fusion, runtime_info, data_cache);
  NVF_ERROR(params_ != nullptr);
}

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
  NVF_ERROR(int(tv->nDims()) >= number_of_inner_pos);
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
//!  [tile_row, tile_col, ***skip***] where the parameter `skip` is the number
//!  of reduction domains to be skipped. The IDs of tile_row and tile_col are
//!  the ones being swizzled.
//! If the input tensorview is not stored in shared memory, the function will
//! skip the actual swizzle. This is used to help the domain mapping between
//! mma_result and the epilogue tensor.
void swizzleSharedMemory(
    TensorView* shared_mem_tv,
    const MatmulParams& params) {
  // Set skip to skip all consecutive reduction domains starting from the
  //  innermost dimension.
  int skip = 0;
  for (int i = (int)shared_mem_tv->nDims() - 1; i >= 0; --i) {
    if (shared_mem_tv->axis(i)->isReduction()) {
      skip++;
    } else {
      break;
    }
  }

  // Check that the innermost 2 dimensions are concrete and static
  //  sized so that the swizzle function can be defined.
  NVF_ERROR(
      shared_mem_tv->nDims() >= (size_t)(2 + skip),
      "At least 2D input (excluding consecutive reduction domains starting from the innermost dim) needed for swizzling, but get ",
      shared_mem_tv->toString());
  checkConcreteStaticDim(shared_mem_tv->axis(-2 - skip));
  checkConcreteStaticDim(shared_mem_tv->axis(-1 - skip));

  // Extract the constant sizes of the swizzled tile
  const int64_t tile_size_x =
      shared_mem_tv->axis(-2 - skip)->extent()->evaluateInt();
  const int64_t tile_size_y =
      shared_mem_tv->axis(-1 - skip)->extent()->evaluateInt();

  if (isTuring(params.mma_macro) || isAmpere(params.mma_macro)) {
    // Only tested for (1) ldmatrix access with sizeof(T) == 16bit (i.e.
    // half/bfloat16) and (2) epilogue general access with sizeof(T) == 32bit
    // (i.e. float)
    const int64_t data_type_size =
        (int64_t)dataTypeSize(*shared_mem_tv->getDataType());
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

    if (repeated_pattern_size >= n_rows) {
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
      shared_mem_tv->split(-2 - skip, repeated_pattern_size);
    }
    shared_mem_tv->split(-1 - skip, n_cols);
    //      -4         -3       -2        -1
    // [gigarow id, gigarow, matrix id, matrix]
    shared_mem_tv->split(-2 - skip, num_gigabanks);
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
     * division-with-remainder property (see comment in expr_simplifier.h):
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
    shared_mem_tv->split(axis_of_gigarow_id - skip, num_gigabanks);
    //     -6     -5     -4       -3        -2         -1
    // [wave id, wave, gigarow, y outer, gigabank id, matrix]

    // swizzle wave with gigabank id to make threads in a wave access different
    // gigabank. Apply swizzle only when shared_mem_tv is stored in shared
    // memory.
    // TODO: This is a temporary workaround for the following issue:
    // For the mma output, we have the following schedule:
    // rFactor: [...., X, Y] -> mma-swizzle transformations -> leaf
    // For epilogue smem tensor, the schedule is
    // rFactor: [...., X, Y] -> split -> [...., X1, X2, X3, Y1, Y2, Y3]
    //   -> swizzle X2, Y2 -> [...., X1, X2', X3, Y1, Y2', Y3]
    //   -> merge back -> [...., X', Y']
    //   -> mma-swizzle transformations -> leaf
    // The mma-swizzle transformations for the mma output and epilogue smem
    // tensor are the same. In indexing, we do require {X, X'} and {Y, Y'} to be
    // mapped in CA map, however, we currently can not handle that. So we have
    // to do the same split and merge to the mma output without actually
    // applying the swizzle, and this check is to detect and handle this
    // specific case. We should remove this special handling when we fix our CA
    // mapping.
    if (shared_mem_tv->getMemoryType() == MemoryType::Shared) {
      int axis_of_gigarow_id = repeated_pattern_size > 1 ? -5 : -4;
      if (isPowOf2(num_gigabanks)) {
        shared_mem_tv->swizzle(
            Swizzle2DType::XOR, axis_of_gigarow_id - skip, -2 - skip);
      } else {
        shared_mem_tv->swizzle(
            Swizzle2DType::CyclicShift, axis_of_gigarow_id - skip, -2 - skip);
      }
    }

    if (repeated_pattern_size > 1) {
      shared_mem_tv->merge(-6 - skip);
    }
    shared_mem_tv->merge(-5 - skip);

    // merge back tile_size_y
    shared_mem_tv->merge(-3 - skip);
    shared_mem_tv->merge(-2 - skip);

  } else if (isVolta(params.mma_macro)) {
    // TODO: Volta is slightly more complex, and a fixed recipe would
    //  not scale. In a follow up this would be inferred from the mma
    //  macro layout themselves as we already have them registered in
    //  the utils.
    return;
  } else {
    NVF_ERROR(false, "Prolog swizzle: unsupported mma macro");
  }
}

//! Generates the prolog schedule on the shared memory buffer
//!  tensor. The scheduling performs two steps:
//!
//! 1. Swizzled the shared mem data layout.
//! 2. Coalesce and vectorize the read write schedule.
void scheduleProlog(TensorView* shared_mem_tv, const MatmulParams& params) {
  shared_mem_tv->setMemoryType(MemoryType::Shared);

  // The following line allows us to reclaim the memory allocated to
  // shared_mem_tv and reuse it for the epilogue, introducing one block sync if
  // needed. This is not done by default as we do not insert new syncs unless
  // requested to do so. If smem is not used for the epilogue, this call will
  // have no effect.
  if (params.promote_prologue_smem_reuse) {
    shared_mem_tv->promoteReuse();
  }

  mma_utils::orderTiledConcreteIdAsRoot(shared_mem_tv);

  // Swizzle the shared memory data layout
  swizzleSharedMemory(shared_mem_tv, params);
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

void scheduleOutputTensor(
    TensorView* mma_result,
    TensorView* c,
    const MatMulTileOptions& gemm_tile) {
  // input tensor is in the form of [Mo,No,cta_tile_m,cta_tile_n]
  checkConcreteStaticDim(c->axis(-2));
  checkConcreteStaticDim(c->axis(-1));
  const int64_t tile_size_m = c->axis(-2)->extent()->evaluateInt();
  const int64_t tile_size_n = c->axis(-1)->extent()->evaluateInt();
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
  const int64_t data_type_size = (int64_t)dataTypeSize(*c->getDataType());
  constexpr int64_t warp_size = 32l;
  const int64_t vectorization_factor = 16l / data_type_size;
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
    const mma_utils::RolesMap& roles_map,
    const bool with_smem_epilogue) {
  std::vector<TensorView*> cached_tvs;

  // Handling transformations in fusion input tvs with assigned INPUT_C role by
  //  propagating fusion output transformations through cached views of INPUT_C
  //  fusion input tvs and by setting vectorization of the inner most iterdomain
  //  of these cached views
  if (roles_map.count(MatmulRole::INPUT_C)) {
    auto& c_tvs = roles_map.at(MatmulRole::INPUT_C);

    // The system supports only scenario where there is only one fusion output
    //  with assigned OUTPUT_D role, this condition is already verified so there
    //  is no need for an additional checks here
    auto output_d = roles_map.at(MatmulRole::OUTPUT_D).front();
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

    // The cached INPUT_C tvs are not needed anymore
    cached_tvs.clear();
  }
}

} // namespace

void scheduleMatmul(Fusion* fusion, const MatmulParams& params) {
  const auto& roles_map_opt = mma_utils::getTensorsRoles(fusion);

  // NOTE: the contents of roles_map have been already validated during
  //  compute-time checks
  NVF_ERROR(roles_map_opt.isValid(), roles_map_opt.getErrorMsg());
  const auto roles_map = roles_map_opt.getData();

  auto mma_ops = ir_utils::getOpsOfType<MmaOp>(fusion);
  NVF_ERROR(
      mma_ops.size() == 1,
      "scheduleMatmul supports fusion with single mma op in definition, got ",
      mma_ops.size());

  // Core roles: there can be only one... TV with assigned core role
  TensorView* a = roles_map.at(MatmulRole::INPUT_A).front();
  TensorView* b = roles_map.at(MatmulRole::INPUT_B).front();
  TensorView* d = roles_map.at(MatmulRole::OUTPUT_D).front();

  // Collect mma swizzle info
  auto mma = mma_ops.front();
  const auto mma_layout_opt = mma->layout();
  NVF_ERROR(
      mma_layout_opt.has_value(), "fusion mma op has undefined input layout");
  const auto mma_layout = mma_layout_opt.value();
  const auto fusion_layout = mma_utils::getMatmulLayout(fusion);
  NVF_ERROR(fusion_layout.isValid(), fusion_layout.getErrorMsg());

  auto mma_builder =
      MmaBuilder(params.mma_macro, params.tile_sizes).layout(mma_layout);
  const auto& gemm_tile = params.tile_sizes;
  const bool has_epilogue = !mma->out()->isFusionOutput();

  const bool has_fusion_c_roles = (0 != roles_map.count(MatmulRole::INPUT_C));
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

  mma_builder.configureMma(mma);

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

  // Get the input to the mma op.
  auto ab = mma->inA()->as<TensorView>();
  auto bb = mma->inB()->as<TensorView>();

  // Setup accumulator register.
  auto dc = d->cacheBefore();
  // Mma object is valid only because cacheBefore has been done on
  //  TV which is not output of MmaOp, as there is an epilogue
  auto mma_result = has_epilogue ? mma->out()->as<TensorView>() : dc;

  // Unswizzle mma result in shared memory
  auto smem_epilogue =
      params.use_smem_epilogue ? mma_result->cacheAfter() : mma_result;

  // Clear MmaOp pointer, it's not needed from now on
  mma = nullptr;

  // Set accumulation tv for mma op.
  mma_builder.accumulatorTv(mma_result);

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
    CacheOp cache_op = CacheOp::Unspecified;
    if (params.async_gmem_load_operands) {
      load_op = LoadStoreOpType::CpAsync;
      cache_op = CacheOp::Global;
    }

    acw_smem = ar->cacheAfter(load_op, cache_op);
    bcw_smem = br->cacheAfter(load_op, cache_op);
    NVF_ERROR(acw_smem->uses().size() == 1);
    NVF_ERROR(bcw_smem->uses().size() == 1);
    if (auto ldst = dynamic_cast<LoadStoreOp*>(acw_smem->uses().at(0))) {
      acr = ldst->out()->as<TensorView>();
      if (ldst->hasInnerTranspose()) {
        ldst->setOpType(LoadStoreOpType::LdMatrixTranspose);
      } else {
        ldst->setOpType(LoadStoreOpType::LdMatrix);
      }
    } else {
      acr = acw_smem->cacheAfter(LoadStoreOpType::LdMatrix);
    }
    if (auto ldst = dynamic_cast<LoadStoreOp*>(bcw_smem->uses().at(0))) {
      bcr = ldst->out()->as<TensorView>();
      if (ldst->hasInnerTranspose()) {
        ldst->setOpType(LoadStoreOpType::LdMatrixTranspose);
      } else {
        ldst->setOpType(LoadStoreOpType::LdMatrix);
      }
    } else {
      bcr = bcw_smem->cacheAfter(LoadStoreOpType::LdMatrix);
    }

    // For Turing and Ampere, the layout of the MmaOp is always TN
    NVF_ERROR(
        mma_layout == MmaOptions::MmaLayout::TN,
        "MMAs in Turing and Ampere are TN only, transpose is handled either "
        "via ldmatrix.trans for fp16 or explicitly for other types.");
    mma_builder.layout(fusion_layout.getData());
  }

  // Make a CTA tile
  // ------------------------------------------------------------------
  mma_utils::canonicalizeMmaTvOrdering(mma_result);
  NVF_ERROR(
      mma_result->nDims() == 3 || mma_result->nDims() == 4,
      "Currently, we only support B, M, N and K being a single dimension.",
      " More general tensor contraction is not supported yet.");
  const int num_batch_dims = (int)mma_result->nDims() - 3;

  // [... M,N,K]
  mma_utils::makeTile(mma_result, gemm_tile.cta_tile.toVector());

  // Swizzle block tiles:
  if (params.grid_swizzle_factor != 1) {
    int factor = std::max(1, params.grid_swizzle_factor); // must be >=1
    if (params.cta_order == MatmulParams::TileRasterizationOrder::RowMajor) {
      mma_result->split(num_batch_dims + 1, factor);
      // [I1, I2/factor, factor]
      mma_result->reorder({{num_batch_dims + 1, num_batch_dims + 2}});
      // [I1, factor, I2/factor]
      mma_result->merge(num_batch_dims);
      // [I1*factor, I2/factor]
    } else if (
        params.cta_order == MatmulParams::TileRasterizationOrder::ColumnMajor) {
      mma_result->split(num_batch_dims, factor);
      // [I1/factor, factor, I2]
      mma_result->reorder({{num_batch_dims + 1, num_batch_dims + 2}});
      // [I1/factor, I2, factor]
      mma_result->merge(num_batch_dims + 1);
      // [I1/factor, I2*factor]
    }
  }

  // [..., Mo, No, Koo, Mi, Ni, Ki]
  int num_splitk_dims = 0;
  TensorView* splitk_sum = nullptr;
  if (params.splitk_factor != 1) {
    // Split Koo -> [Kf, Ko]
    mma_result->split(-4, params.splitk_factor, /*inner*/ false);
    // After split [..., Mo, No, Kf, Ko, Mi, Ni, Ki]
    // rFactor converts
    //   mma_result = mma(A, B, {/*Kf*/-5, /*Ko*/-4, /*Ki*/-1});
    // to
    //   intermediate = mma(A, B, {-4, -1});
    //   final_sum = sum(intermediate, {/*Kf*/-3});
    // and the method returns "intermediate". We need mma_result to refer to
    // the actual MmaOp output, so here we reassign that to the intermediate.
    splitk_sum = mma_result;
    mma_result = splitk_sum->rFactor({-4, -1});

    // the accumulator must be the output of the MMA op, which is now the
    // rfactor TV
    mma_builder.accumulatorTv(mma_result);

    num_splitk_dims = 1;
  }

  // Propagate tiling globally
  scheduler_utils::transformPropagateToAllFrom(mma_result, -1);

  if (params.use_smem_epilogue) {
    // Transform mma_result through the epilogue swizzle without actually
    // swizzling the axes. This is done to enable the domains
    // are mapped between mma_result and smem_epilogue.
    swizzleSharedMemory(mma_result, params);
  }

  // Schedule warp tile
  mma_utils::scheduleWarpTileWithReduction(mma_result, gemm_tile);
  // [..., Mo, No, (Kf,) Ko, Kw, Mwo, Nwo, Mwi, Nwi, Mi, Ni, Ki]

  // Propagate warp tile to main loop and epilog/output tvs
  scheduler_utils::BoundedDirectionalTransformPropagator::bothWays(
      mma_result, -1, {acw_smem, bcw_smem}, {smem_epilogue});

  // Schedule prolog:
  //   TODO: this section needs more configurability.
  // ------------------------------------------------------------------
  scheduleProlog(acw_smem, params);
  scheduleProlog(bcw_smem, params);
  // [..., Mo, No, (Kf,) Ko, Kw, Mwo, Nwo, Mwi, Nwi, Mi, Ni, Ki]

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

  mma_result->applyMmaSwizzle(
      mma_builder.operand(MmaOptions::Operand::Accumulator).build());

  // Set parallelization:
  //   TODO: this section goes to a separate matmul util,
  //   and needs more configurability.
  // ------------------------------------------------------------------

  // Vectorize smem stores/loads:
  acr->axis(-1)->parallelize(ParallelType::Vectorize);
  bcr->axis(-1)->parallelize(ParallelType::Vectorize);

  // Parallelization strategy:
  //  with splitk:
  // nbatch +   1   2   3   4    5    6    7    8     9    10    11   12
  //      -13 -12 -11 -10  -9   -8   -7   -6   -5    -4    -3    -2   -1
  // [..., Mo, No, Kf, Ko, Kw, Mwo, Nwo, Mwi, Nwi, MNi1, MNi2, MNi3,  Ki]
  //  (iS) iBx iBy rBz rS  rS  iTz  iTy   iS   iS  iMMA   iTx  iMMA rMMA
  //
  //  without splitk:
  // nbatch +   1       2   3    4    5    6    7     8     9    10   11
  //      -12 -11     -10  -9   -8   -7   -6   -5    -4    -3    -2   -1
  // [..., Mo, No,     Ko, Kw, Mwo, Nwo, Mwi, Nwi, MNi1, MNi2, MNi3,  Ki]
  // (iBz) iBx iBy     rS  rS  iTz  iTy   iS   iS  iMMA   iTx  iMMA rMMA

  // When we have both batch dims and splitk, parallelize splitk only.
  // If we only have batch dim, parallelize the batch dim.
  if (num_splitk_dims != 0) {
    mma_result->axis(2)->parallelize(ParallelType::BIDz);
  } else if (num_batch_dims != 0) {
    mma_result->axis(0)->parallelize(ParallelType::BIDz);
  }
  switch (params.cta_order) {
    case MatmulParams::TileRasterizationOrder::RowMajor:
      mma_result->axis(num_batch_dims)->parallelize(ParallelType::BIDx);
      mma_result->axis(num_batch_dims + 1)->parallelize(ParallelType::BIDy);
      break;
    case MatmulParams::TileRasterizationOrder::ColumnMajor:
      mma_result->axis(num_batch_dims)->parallelize(ParallelType::BIDy);
      mma_result->axis(num_batch_dims + 1)->parallelize(ParallelType::BIDx);
      break;
    default:
      NVF_ERROR(
          false, "Invalid TileRasterizationOrder passed to Matmul scheduler");
  }

  // parallelize Mwo, Nwo by thread
  mma_result->axis(num_batch_dims + 4 + num_splitk_dims)
      ->parallelize(ParallelType::TIDz);
  mma_result->axis(num_batch_dims + 5 + num_splitk_dims)
      ->parallelize(ParallelType::TIDy);

  scheduler_utils::parallelizeAllLike(
      mma_result,
      -1,
      {acr, bcr, ab, bb},
      {ParallelType::TIDy, ParallelType::TIDz});

  // handle epilogue and always vectorize Ki
  if (params.use_smem_epilogue) {
    smem_epilogue->setMemoryType(MemoryType::Shared);
    swizzleSharedMemory(smem_epilogue, params);
    scheduler_utils::BoundedDirectionalTransformPropagator::forward(
        mma_result,
        -1,
        {smem_epilogue},
        scheduler_utils::BoundedDirectionalTransformPropagator::Options()
            .propagateParallelType()
            .propagateToBoundary());
    smem_epilogue->axis(-1)->parallelize(ParallelType::Vectorize);

    // Schedule output tensor differently for better global memory access
    // pattern.
    scheduleOutputTensor(mma_result, d, gemm_tile);
    d->axis(-1)->parallelize(ParallelType::Vectorize);

    // Propagate output tensor transformations back to smem_epilogue
    scheduler_utils::BoundedDirectionalTransformPropagator::backward(
        d, -1, {smem_epilogue});
  } else {
    scheduler_utils::BoundedDirectionalTransformPropagator::forward(
        mma_result,
        -1,
        {d},
        scheduler_utils::BoundedDirectionalTransformPropagator::Options()
            .propagateParallelType()
            .propagateToBoundary());
    d->axis(-1)->parallelize(ParallelType::Vectorize);
  }
  // propagate output transformations to all inputs that are part of epilogue
  //  operations, input tvs with non-core roles
  //  core roles: essential for matmul, for example mma inputs' producers
  if (has_non_mma_input_tvs) {
    scheduleFusionInputsForEpilogue(roles_map, params.use_smem_epilogue);
  }

  if (num_splitk_dims) {
    // Inline the splitk sum with the output store
    // splitk_sum->computeAt(d, -2);
    // splitk_sum->inlineAt(-2, true);
    auto epilogue_vals = DependencyCheck::getAllValsBetween({splitk_sum}, {d});
    auto epilogue_tvs = ir_utils::filterByType<TensorView>(epilogue_vals);
    std::unordered_set<TensorView*> epilogue_tvs_set(
        epilogue_tvs.begin(), epilogue_tvs.end());
    inlineSelectedAt(epilogue_tvs_set, d, -2, true);
  }

  // auto inline for all tensors except register tensors
  inlineMost(ir_utils::allTvsExcept(fusion, {acr, bcr, ab, bb}));

  // if auto inline, will inline to position-7, leads to performance regression
  inlineSelectedAt(
      {acr, bcr, ab, bb}, mma_result, num_batch_dims + 6 + num_splitk_dims);

  // Propagate mma output swizzle and parallelization down the DAG
  if (params.double_buffer_options.double_buffer_smem_write) {
    NVF_ERROR(
        params.double_buffer_options.smem_double_buffer_stage > 1,
        "Invalid buffer stage config")
    if (params.double_buffer_options.smem_double_buffer_stage > 2) {
      NVF_ERROR(
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

  if (params.double_buffer_options.double_buffer_smem_read &&
      params.double_buffer_options.double_buffer_smem_write) {
    // rotate Ko loop
    scheduler_utils::rotateLoop(
        mma_result, num_batch_dims + 2 + num_splitk_dims, {acr, bcr});
  }
}

} // namespace nvfuser
