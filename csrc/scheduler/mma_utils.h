// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#pragma once

#include <fusion.h>
#include <mma_type.h>
#include <array>
#include <variant>
#include <vector>

namespace nvfuser {

namespace mma_utils {

//! Utilities in this namespace facilitates scheduling matmul kernels with
//!  hierarchichal tiling specified in MatMulTileOptions.

//! Schedule utility for matmul prolog:
//!   Use all the threads on a CTA tile to load matmul operands
//!  into shared memory with the given vectorization word.
//! TODO:
//!  will need to add bank conflict removal swizzle in a follow up.
TORCH_CUDA_CU_API void scheduleContiguousVectorLoad(
    TensorView* tv,
    MatMulTileOptions tile,
    int vector_word,
    bool vectorize = true);

//! Schedule utility for mma output in matmul main loop:
//!  Realize the hierarchical tiling based on the given tiling options.
//! TODO: rewrite this one with makeTile
TORCH_CUDA_CU_API void scheduleWarpTileWithReduction(
    TensorView* tv,
    MatMulTileOptions tile);

//! Schedule utility for mma output in matmul main loop:
//!  Realize the hierarchical tiling based on the given tiling options
//! on consumers of mma ops in epilog.
//! TODO: remove this one eventually.
TORCH_CUDA_CU_API void scheduleWarpTileWithNoReduction(
    TensorView* tv,
    MatMulTileOptions tile);

//! Lower level primitive spliting inner iterdomains into tiles:
//! Eg.
//!  A[B,I0,I1,I2] -> makeTile({1,2,3})
//! Gives A[B, I0o, I1o, I2o, I0i(1), I1i(2), I2i(3)]
TORCH_CUDA_CU_API void makeTile(TensorView* tv, std::vector<int> tile_sizes);

//! Order the inner tile dimensions as the original order in
//!  root domain. Also putting broadcast domains on the left.
//! Eg. A[I0o,I1o,B2o,I0i,I1i,B2i] (root domain: I1,B,I0)
//! -> A[I0o, I1o, B2o, B2i, I1i, I0i]
//! This is used to facilitate data layout swizzling and
//!  defining vectorized loads.
TORCH_CUDA_CU_API void orderTiledConcreteIdAsRoot(TensorView* tv);

//! Orders the root id ordering of the given tv as
//! [Batch, Previous Reduction, M, N, K]
//!  for easier processing of later scheduling steps.
//!
//! This matching works on root domain only, and
//!  will throw if the tv has a leaf iterdomain that is
//!  not a root id.
TORCH_CUDA_CU_API void canonicalizeMmaTvOrdering(TensorView* tv);

//! [WarpMmaSwizzler]:
//!   This class is used to implement the thread swizzle format
//!     required for the mma macros, cf. PTX ISA 9.7.13.4.
//!
//!   The mma instructions (Volta through Ampere) require specific
//!     thread mapping within a warp for both the mma inputs and
//!     mma outputs. All mma swizzle patterns seen so far turned out
//!     to be affine, so we could use the normal scheduler interface
//!     to fulfill the mma thread swizzle pattern. And fusion with
//!     other non-mma ops and validations can just natually rely on the current
//!     iterdomain infrastructure.
//!
//!   This is different from a normal scheduler utility though,
//!      as the thread mapping within a warp are *required* to be
//!      a specific pattern which currently translates to an enforced
//!      requirement that all the leaf domains produced by WarpMmaSwizzler
//!      cannot be further transformed (split/merge/reorder etc.).
//!
//!   Currently WarpMmaSwizzler can be accessed by schedulers through
//!     TensorView::applyMmaSwizzle, and the current scheduling procedure is
//!     as follows:
//!
//!   Step 1. Before scheduling, the mma op needs to be configured with a macro
//!   type, either manually or inferred (eg. Volta_16_16_4).
//!
//!   Step 2. Scheduler can tile the outer dimensions based on any heuristics,
//!   i.e. the CTA tiling, warp tiling, splitK etc.
//!
//!   Step 3. The scheduler will need to split the innermost part of the 3
//!   involved
//!    root dimensions, they need to be ordered as M,N,K on the rightmost of
//!    tensordomain (see [Operand Layout Convention] for exact definition).
//!
//!    For example before calling WarpMmaSwizzler, the domain could look like:
//!    [TileM, TileN, TileK, Im(16), In(16), Rk(4)], to use Volta_16_16_4.
//!    The rightmost 3 iterdomains need to be the innermost component of their
//!    corresponding root id, similar to vectorization except this requirement
//!    applies to all 3 rightmost dims.
//!
//!         Before applying swizzle, WarpMmaSwizzler will try to validate:
//!           1. The "innermost-ness" of the rightmost 3 iterdomains. E.g:
//!              Xo, Xi = split(X, 16),
//!               Xo doesn't check, Xi would check.
//!           2. The rightmost three are constant sized, and they are ordered as
//!           M,N,K.
//!             In the case of operand schedule before the broadcast, only 2 of
//!             the axis are see, and they still need to follow the same order,
//!             i.e. need to be M,K or N,K.
//!           3. The rightmost three axes have matching size with the selected
//!           mma macro.
//!
//!    Step 4. WarpMmaSwizzler will transform the rightmost 3 domains to the
//!    correct swizzle
//!     format and will parallelize the TIDx, which is reserved for lane id. The
//!     transformed inner iterdomains will be locked with WarpMapped tag so that
//!     they cannot be further transformed. Currently the only change that
//!     scheduler can still do after this step is to vectorize the innermost
//!     iterdomain.
//!
//! Notes:
//!   This version of implementation is trying to balance the composition
//!   flexibility and validation complexity. Currently the validation protocol
//!   is that if the rightmost 3 dimensions given to WarpMmaSwizzler are indeed
//!   innermost components of the 3 root id's and their dimensions match the mma
//!   macro, the swizzle format produced by WarpMmaSwizzler will be correct for
//!   the macro and we just lock the innermost iterdomains from further
//!   transformations.
//!
//!   Ninja users/schedulers might go for 2 cases that we currently don't
//!   support:
//!
//!   1. Equivalent affine transforms:
//!     Even though the mma swizzles are affine, there are still infinitely many
//!     equivalent ways to implement
//!      the same affine transform. E.g. io,ii = split(i,8); ioii =
//!      merge(io,ii); would make ioii equiv to i if it's a divisible split. One
//!      can use this to construct infinite many equivalent affine swizzles.
//!
//!     Users/schedulers might want to have a different but equivalent affine
//!     representation from the one provided
//!      by WarpMmaSwizzler, but validating them needs some extra work
//!      canonicalizing the affine transforms. So short term wouldn't support
//!      this flexibility.
//!
//!   2. Swizzled data input:
//!     It is also possible that the data input has other swizzles before
//!     entering the fusion already and some might be natively compatible
//!     with mma format. This is a very broad category of use cases
//!     and we'd have to consider enabling any use like this case-by-case.
class TORCH_CUDA_CU_API WarpMmaSwizzler {
 public:
  //! Applies the output mma swizzling to the given tv, should be used
  //!  on mma output or tv's involved in epilog fusion, i.e. bias.
  //! The rightmost iterdomains must follow the m,n,k convention before calling.
  static void scheduleMmaWarpOutput(TensorView* tv, MmaOptions options);

  //! Applies the input mma swizzling to the given tv, should be used
  //!  on mma input or tv's involved in any fusion before mma, but after smem
  //!  read.
  //! The rightmost iterdomains must follow the m,n,k convention before calling.
  static void scheduleOperandRead(
      TensorView* tv,
      MmaOptions options = MmaOptions());

 private:
  //! Operand swizzle implementations for Volta mma.
  static void scheduleVoltaOperandRead(TensorView* tv, MmaOptions options);

  //! Accumulator swizzle implementations for Volta mma.
  static void scheduleVoltaM16N16K4Fp32Output(
      TensorView* tv,
      const MmaOptions& options);

  //! Operand swizzle implementations for Turing and Ampere mma.
  static void scheduleTuringOperandRead(TensorView* tv, MmaOptions options);

  //! Accumulator swizzle implementation for Turing and Ampere mma.
  static void scheduleTuringM16N8K16MmaWarpOutput(
      TensorView* tv,
      const MmaOptions& options);

  //! Accumulator swizzle implementation for emulated 16x16x16 mma tile
  //!  that enables using ldmatrix.x4.
  //! Note:
  //!   Keeping both this option and the ldmatrix.x2 variant above for
  //! now for wider scheduler exploration space. Eventually both of
  //! these can be unified with a single affine utility.
  static void scheduleTuringM16N16K16MmaWarpOutput(
      TensorView* tv,
      const MmaOptions& options);

  //! Utility to lock the transformed dimensions from further transforms.
  static void setWarpMapped(TensorView* tv, int number_of_dims);
};

void checkDimSize(
    TensorView* tv,
    std::vector<int> axis,
    std::vector<int> expect);

// Returns if the loopnest is initializing for an mma op.
bool isMmaInitLoop(const kir::ForLoop* loop);

//! A constant with minimum number of fusion inputs that could be MMA inputs.
//!  TODO: update for square matmuls where both inputs are the same tensor
constexpr size_t MIN_MATMUL_INPUTS_NUMBER = 2;

//! An alias for data structure for passing IterDomains representing problem
//! shape dimensions
//!  TODO: extend definition for handling batch matmuls
using ProblemIterDomains = std::array<IterDomain*, 3>;

//! An alias for mapping between TensorView instance and its role in
//!  matmul fusion definition, some roles can be assigned to more than
//!  a single tv, for example input for beta scaling in epilogue
using RolesMap = std::map<MatmulRole, std::vector<TensorView*>>;

//! An alias for storing data types of the tensors in the mma op
//!  the order is INPUT_A, INPUT_B, OUTPUT_D
using MmaDataTypes = std::array<DataType, 3>;

//! A wrapper for data containers with optional error message stored if
//!  initialization of the data fails.
template <typename DataType>
class DataWrapperOpt {
 private:
  std::variant<std::string, DataType> data;

 public:
  DataWrapperOpt(std::string&& v) : data(std::move(v)) {}
  DataWrapperOpt(DataType&& v) : data(std::move(v)) {}

  bool isValid() const {
    return std::holds_alternative<DataType>(data);
  }
  DataType getData() const {
    return std::get<DataType>(data);
  }
  std::string getErrorMsg() const {
    if (data.valueless_by_exception() ||
        std::holds_alternative<std::string>(data)) {
      return "Uninitialized data in data holder object";
    } else {
      return std::get<std::string>(data);
    }
  }
};

using MatmulProblemLayoutOpt = DataWrapperOpt<MmaOptions::MmaLayout>;
using ProblemIterDomainsOpt = DataWrapperOpt<ProblemIterDomains>;
using RolesMapOpt = DataWrapperOpt<RolesMap>;

using DomainsDesc = std::vector<MatmulDomain>;
using DependenciesMap = std::map<TensorView*, DomainsDesc>;

//! Returns wrapped matmul input layout data, if supported, otherwise returned
//!  object contains a message with failure root cause.
//!
//! Matmul layout depends only on fusion definition while mma layout relies on
//!  HW implementation to handle input layout from fusion definition. Detailed
//!  explanation:
//! - matmul layout which contains information about transposition of matmul
//!  inputs, it is based on the order of key domains (M,N K) in fusion input
//!  tensors,
//! - mma layout, some architectures (e.g. Volta) support all combination of
//!  transposition of inputs in mma instructions, while other (e.g. Turing,
//!  Ampere) the only supported transposition is TN which means that mma
//!  instruction first input is transposed, the second input is non-transposed.
TORCH_CUDA_CU_API MatmulProblemLayoutOpt getMatmulLayout(Fusion* fusion);

//! Returns wrapped collection of IterDomains that can be used to get
//!  problem shape with runtime info.
//!  Data is stored in the order in which lables are defined in MatmulDomain
//!  enum class, that is in the following order: m, n, k.
//!  An error message is stored in retruned object if valid data cannot
//!  be gathered.
//!  TODO: 4th domain must be added for batch gemm support.
TORCH_CUDA_CU_API ProblemIterDomainsOpt getProblemIterDomains(Fusion* fusion);

//! Returns wrapped collection of TensorView roles in fusion.
//!  An error message is stored in retruned object if valid data cannot
//!  be gathered.
TORCH_CUDA_CU_API RolesMapOpt getTensorsRoles(Fusion* fusion);

//! Return whether use shared memory epilogue or not.
//!  Returns true if using shared memory epilogue won't cause
//!  the decrease of occupancy ratio. The occupancy ratio is
//!  estimated using register and shared memory usage.
//!  If ignore_occupancy_drop is set to true, returns true if
//!  there is enough shared memory to launch the kernel without
//!  considering the occupancy, useful for debug and validate
//!  shared memory epilogue implementation.
TORCH_CUDA_CU_API bool generateSharedMemoryEpilogueHeuristics(
    const MatMulTileOptions& gemm_tile,
    const int smem_double_buffer_stage,
    const MmaDataTypes& data_types,
    const bool ignore_occupancy_drop = false);

} // namespace mma_utils

} // namespace nvfuser
