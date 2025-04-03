// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#pragma once

#include <exceptions.h>
#include <fusion.h>
#include <id_model/id_model.h>
#include <mma_type.h>
#include <scheduler/matmul_heuristic.h>
#include <scheduler/tools/abstract_tensor.h>
#include <val_graph.h>
#include <visibility.h>

#include <array>
#include <variant>
#include <vector>

namespace nvfuser {

namespace mma_utils {

//! Utilities in this namespace facilitates scheduling matmul kernels with
//!  hierarchichal tiling specified in MatMulTileOptions.

//! A mapping from ValGroup pointers to MatmulDimRole. The ValGroups should
//! correspond to IterDomain groups from an IdModel's exact graph. This
using DimRolesMap = std::unordered_map<ValGroup, MatmulDimRole>;

//! Schedule utility for matmul prolog:
//!   Use all the threads on a CTA tile to load matmul operands
//!  into shared memory with the given vectorization word.
//! TODO:
//!  will need to add bank conflict removal swizzle in a follow up.
NVF_API void scheduleContiguousVectorLoad(
    TensorView* tv,
    MatMulTileOptions tile,
    int64_t vector_word,
    bool vectorize = true);

//! Schedule utility for mma output in matmul main loop:
//!  Realize the hierarchical tiling based on the given tiling options.
//! TODO: rewrite this one with makeTile
NVF_API void scheduleWarpTileWithReduction(
    TensorView* tv,
    MatMulTileOptions tile,
    MmaMacro macro);

//! Schedule utility for mma output in matmul main loop:
//!  Realize the hierarchical tiling based on the given tiling options
//! on consumers of mma ops in epilog.
//! TODO: remove this one eventually.
NVF_API void scheduleWarpTileWithNoReduction(
    TensorView* tv,
    MatMulTileOptions tile,
    MmaMacro macro);

//! Lower level primitive spliting inner iterdomains into tiles:
//! Eg.
//!  A[B,I0,I1,I2] -> makeTile({1,2,3})
//! Gives A[B, I0o, I1o, I2o, I0i(1), I1i(2), I2i(3)]
void makeTile(TensorView* tv, const std::vector<int64_t>& tile_sizes);

//! The above call assumes the axes are [(B), M, N, K]. In this version, we
//! provide the dimension roles that are present for this tensor.
void makeTile(
    TensorView* tv,
    const GemmTile& tile_sizes,
    const std::vector<MatmulDimRole>& axis_roles);

//! We model each dimension of every tensor in the Fusion with ID roles
//! described by MatmulDimRole.
using AbstractMatmulTensor = TaggedAbstractTensor<MatmulDimRole>;

//! Abstract version of the above utility. Schedules the provided
//! AbstractMatmulTensor instead of a concrete TensorView.
void makeTile(
    AbstractMatmulTensor& canonicalized_abstract_tensor,
    const std::vector<int64_t>& tile_sizes);

//! Order the inner tile dimensions as the original order in
//! (maybe allocation) domain. Also putting broadcast domains on the left.
//! Eg. A[I0o,I1o,B2o,I0i,I1i,B2i] (maybe allocation domain: I1,B,I0)
//! -> A[I0o, I1o, B2o, B2i, I1i, I0i]
//! This is used to facilitate data layout swizzling and
//!  defining vectorized loads.
void orderTiledConcreteIdAsMaybeAllocationDomain(TensorView* tv);

//! Orders the leaf ID canonically, and merges dims of the same role
//! The return value gives the role of each loop IterDomain in tv.
std::vector<MatmulDimRole> canonicalizeMmaTvOrdering(
    TensorView* tv,
    const ValGraph& broadcast_graph,
    const DimRolesMap& dim_roles,
    const std::vector<ValGroup>& ordering);

//! Given a TensorView matching the canonicalDimOrdering, schedule it by
//! merging dimensions with matching roles.
void mergeConsecutiveAxesWithSameRole(
    TensorView* tv,
    const DimRolesMap& dim_roles,
    const ValGraph* graph);

//! [MmaSwizzler]:
//!   This class is used to implement the thread swizzle format
//!     required for the mma macros, cf. PTX ISA 9.7.13.4.
//!
//!   The mma instructions (Volta and later arch) require specific
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
//!      requirement that all the loop domains produced by MmaSwizzler
//!      cannot be further transformed (split/merge/reorder etc.).
//!
//!   Currently MmaSwizzler can be accessed by schedulers through
//!     TensorView::applyMmaSwizzle, and the current scheduling procedure is
//!     as follows:
//!
//!   Step 1. Before scheduling, the mma op needs to be configured with a macro
//!   type, either manually or inferred (eg. Ampere_16_8_8).
//!
//!   Step 2. Scheduler can tile the outer dimensions based on any heuristics,
//!   i.e. the CTA tiling, warp tiling, splitK etc.
//!
//!   Step 3. The scheduler will need to split the innermost part of the 3
//!   involved
//!    root dimensions, they need to be ordered as M,N,K on the rightmost of
//!    tensordomain (see [Operand Layout Convention] for exact definition).
//!
//!    For example before calling MmaSwizzler, the domain could look like:
//!    [TileM, TileN, TileK, Im(16), In(8), Rk(8)], to use Ampere_16_8_8.
//!    The rightmost 3 iterdomains need to be the innermost component of their
//!    corresponding root id, similar to vectorization except this requirement
//!    applies to all 3 rightmost dims.
//!
//!         Before applying swizzle, MmaSwizzler will try to validate:
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
//!    Step 4. MmaSwizzler will transform the rightmost 3 domains to the
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
//!   is that if the rightmost 3 dimensions given to MmaSwizzler are indeed
//!   innermost components of the 3 root id's and their dimensions match the mma
//!   macro, the swizzle format produced by MmaSwizzler will be correct for
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
//!      by MmaSwizzler, but validating them needs some extra work
//!      canonicalizing the affine transforms. So short term wouldn't support
//!      this flexibility.
//!
//!   2. Swizzled data input:
//!     It is also possible that the data input has other swizzles before
//!     entering the fusion already and some might be natively compatible
//!     with mma format. This is a very broad category of use cases
//!     and we'd have to consider enabling any use like this case-by-case.
class MmaSwizzler {
 public:
  //! Applies the output mma swizzling to the given tv, should be used
  //!  on mma output or tv's involved in epilog fusion, i.e. bias.
  //! The rightmost iterdomains must follow the m,n,k convention before calling.
  static AbstractTensor scheduleMmaOutputAllocation(AbstractTensor t);

  //! Applies the input mma swizzling to the given tv as its allocation domain,
  //! should be used on mma input or tv's involved in any fusion before mma, but
  //! after smem read.
  //! The rightmost iterdomains must follow the m,n,k convention before calling.
  static void scheduleOperandRead(TensorView* tv, MmaOperand operand);
  static void scheduleOperandRead(TensorView* tv, MmaInputSmemSwizzle swizzle);

  //! Note [schedule of ldmatrix]
  //! If you look at the doc of ldmatrix and mma for Turing and Ampere:
  //! https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#warp-level-matrix-fragment-mma-16816-float
  //! https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#warp-level-matrix-instructions-ldmatrix
  //! you will find that, the memory layout of the output of ldmatrix, which
  //! matches with the input layout of MMA instruction, mismatch with the index
  //! that each thread uses to call ldmatrix. In nvFuser, we schedule the
  //! allocation domain of the ldmatrix output and mma inputs to be consistent
  //! with the memory layout of the output of ldmatrix, and we schedule the
  //! loop domain of the ldmatrix output to be consistent with the index that
  //! each thread uses to call ldmatrix. This function is used to schedule the
  //! loop domain of the ldmatrix output. The allocation domain of the ldmatrix
  //! output and mma inputs are scheduled in scheduleOperandRead, which must be
  //! called before this function.
  static void scheduleLdMatrix(TensorView* tv, MmaOperand operand);

  //! Function to schedule the load of the input operands of a
  //! Mma op. This internally calls swizzleTMABox. This function
  //! splits/tiles the inputs to correct 2D TMA boxes and calls the function
  //! above. Please note that we currently do not fully support not splitting
  //! the outer dimension. This only works when the inner-dimension is not
  //! split, that is the inner dim is less or equal to the swizzle size (in
  //! bytes). The outer dim here refers to the second ID from the end, so for
  //! the input [B, N, K], N would be outer. Broadcast is always moved
  //! outermost.
  static void scheduleTMALoadForMma(
      TensorView* tv,
      MmaInputSmemSwizzle swizzle);

  //! Parallelize all dims as bulk expect the first dims mentioned in the second
  //! param.
  static void parallelizeAsBulkSkippingFirstIDs(
      TensorView* tv,
      int64_t first_ids_to_skip);
};

//! Schedules the copy operation of output of a Mma op which resided in the
//! shared memory to global memory.
void scheduleTMAStoreForMmaOutput(TensorView* tv, MmaInputSmemSwizzle swizzle);

//! Schedules the loop domain of a TensorView to be compatible with LdMatrix or
//! StMatrix. The loop domain of input TensorView must already be scheduled to
//! match wgmma register accumulator.
void scheduleLdStMatrixForMmaOutput(
    TensorView* tv,
    int64_t tile_m,
    int64_t tile_n);

void checkDimSize(
    TensorView* tv,
    std::vector<int64_t> axis,
    std::vector<int64_t> expect);

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
using TensorRolesMap =
    std::unordered_map<MatmulTensorRole, std::vector<TensorView*>>;

//! An alias for storing data types of the tensors in the mma op
//!  the order is A, B, OUTPUT
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

//! This represents a single matmul operation, without a prologue or epilogue.
//! Each matmul has two inputs which might not be fusion inputs: A and B. It
//! also has one output, which can be Float or reduced precision. For MatmulOp
//! and LinearOp, the output is the same dtype as the inputs; so output does not
//! necessarily correspond to the output of a translated MmaOp and it might not
//! be a fusion output.
struct MatmulPattern {
  TensorView* A;
  TensorView* B;
  // This is not necessarily a Fusion output, but rather is the immediate output
  // representing a matmul in the current Fusion. The definition of this tensor
  // determines what kind of translation is needed, if any. Possible definition
  // Expr types are: MmaOp, ReductionOp (for mul-sum patterns), MatmulOp, and
  // LinearOp.
  TensorView* output;

  struct TranslationResult {
    MmaOp* mma = nullptr;
    // This is useful for replaying replacements of TVs in MatmulPatterns when
    // there are multiple patterns in a single fusion.
    std::unordered_map<TensorView*, TensorView*> replacements;
  };

  //! If the pattern is not already represented by an MmaOp, for example if
  //! there is a MatmulOp instead, this function modifies the fusion to insert
  //! an MmaOp. TensorViews A and B are unchanged, but this->output might be
  //! updated to reflect the replacement tensor.
  TranslationResult translateToMmaOp();

  //! Given an IdModel, map groups of IterDomains to dimension roles
  //! (MatmulDimRole). Note that ValGroup is a shared_ptr to a
  //! VectorOfUniqueEntries<Val*>. We copy these as keys so that the returned
  //! object can safely outlive id_model.
  DimRolesMap getDimRoles(IdModel& id_model) const;

  std::string toString() const;
};

//! Traverse the fusion to find supported matmul patterns
std::vector<MatmulPattern> findMatmulPatterns(Fusion* fusion);

//! This is a vector of roles describing the inner dimension of each operand
using MatmulOperandInnerDims = std::vector<MatmulDimRole>;

using MatmulOperandInnerDimsOpt = DataWrapperOpt<MatmulOperandInnerDims>;
using ProblemIterDomainsOpt = DataWrapperOpt<ProblemIterDomains>;
using DimRolesMapOpt = DataWrapperOpt<DimRolesMap>;
using TensorRolesMapOpt = DataWrapperOpt<TensorRolesMap>;

using DomainsDesc = std::vector<MatmulDimRole>;
using DependenciesMap = std::map<TensorView*, DomainsDesc>;

//! Returns wrapped matmul input memory layout data, if supported, otherwise
//! returned object contains a message with failure root cause.
//!
//! Matmul layout depends only on fusion definition while mma layout relies on
//!  HW implementation to handle input layout from fusion definition. Detailed
//!  explanation:
//! - matmul layout which contains information about transposition of matmul
//!  inputs, it is based on the order of key domains (M,N K) in fusion input
//!  tensors,
//! - mma layout, some architectures (e.g. Hopper) support all combination of
//!  transposition of inputs in mma instructions, while other (e.g. Turing,
//!  Ampere) the only supported transposition is TN which means that mma
//!  instruction first input is transposed, the second input is non-transposed.
NVF_API MatmulOperandInnerDimsOpt getOperandInnerDims(
    const IdModel& id_model,
    const DimRolesMap& dim_roles,
    const TensorRolesMap& tensor_roles);

//! This version assumes the Fusion contains a single MatmulPattern, then builds
//! an IdModel and infers dim roles then calls the above function.
NVF_API MatmulOperandInnerDimsOpt getOperandInnerDims(Fusion* fusion);

//! Returns wrapped collection of TensorView roles in fusion.
//!  An error message is stored in retruned object if valid data cannot
//!  be gathered.
TensorRolesMapOpt getTensorRoles(
    Fusion* fusion,
    const IdModel& id_model,
    const DimRolesMap& dim_roles);

//! Return pair of whether use shared memory epilogue or not and whether to
//!  reuse shared memory for the prologue at the expense of an additional block
//!  sync.
//!
//! Returns true in first position if using shared memory epilogue won't cause
//!  the decrease of occupancy ratio. The occupancy ratio is estimated using
//!  register and shared memory usage.  If ignore_occupancy_drop is set to true,
//!  returns true if there is enough shared memory to launch the kernel without
//!  considering the occupancy, useful for debug and validate shared memory
//!  epilogue implementation.
//!
//! Returns true in the second position if reusing shared memory for the
//!  epilogue does not increase occupancy.
std::pair<bool, bool> generateSharedMemoryEpilogueHeuristics(
    const MatMulTileOptions& gemm_tile,
    const int smem_circular_buffer_stage,
    const TensorRolesMap& tensor_roles,
    bool ignore_occupancy_drop = false);

//! This version assumes roles_map has been analyzed to determine smem datatypes
//! as well as guarantees about prologue smem reuse.
NVF_API std::pair<bool, bool> generateSharedMemoryEpilogueHeuristics(
    const MatMulTileOptions& gemm_tile,
    const int smem_circular_buffer_stage,
    const MmaDataTypes& data_types,
    bool smem_a_reuse_guaranteed = false,
    bool smem_b_reuse_guaranteed = false,
    bool ignore_occupancy_drop = false);

//! Compute the amount of shared memory we expect to need. The actual amount
//! allocated will be determined by aliasing (see alias_memory.cpp). This
//! function is useful for testing that we provide accurate information to our
//! heuristics.
int64_t computeExpectedSharedMemoryUsage(
    const MatmulParams* mparams,
    const MmaDataTypes& data_types,
    bool smem_a_reuse_guaranteed = false,
    bool smem_b_reuse_guaranteed = false);

//! Encode DataType as character using the following mapping (not all are
//! supported yet in nvFuser):
//!  B = Int8
//!  I = Int32
//!  Q = FP8 (E4M3)
//!  R = FP8 (E5M2)
//!  T = BFloat16
//!  H = Float16
//!  F = TensorFloat32
//!  S = Float32
//!  D = Float64
//!  C = complex<float>
//!  Z = complex<double>
char dtypeToChar(const DataType& dtype);

//! This function helps determine if ldmatrix requires a transpose.
bool isLdMatrixTranspose(const LoadStoreOp* ldst);

//! Get a total ordering of dimensions for known tensors. All dims of a
//! particular DimRole are adjacent in the output. We then set the order as
//! follows:
//! 1. Batch dimensions go first
//! 2. K dimensions are innermost
//! 3. M or N can be innermost, depending on the first output's allocation
//!    domain's innermost non-batch dimension.
//! 4. Within each DimRole, dims are ordered as follows:
//!    a. Batch, M, and N dimensions are ordered like the allocation domain of
//!       the first output
//!    b. K dimensions are ordered like the allocation domain of the first
//!       A operand
//!
//! NOTE: The broadcast graph is used for this so that we map broadcast
//! dimensions to non-broadcast.
// TODO: we might want more sophisticated ordering analysis for multi-dim role
// ordering to maximize vectorization across multiple tensors (rule 4)
std::vector<ValGroup> canonicalDimOrdering(
    const mma_utils::TensorRolesMap& tensor_roles,
    const mma_utils::DimRolesMap& dim_roles,
    const ValGraph& broadcast_graph);

//! Returns roles maps which have been merged across individual maps generated
//! by the provided matmul patterns.
//!
//! Returns std::nullopt if two patterns have incompatible roles
std::optional<std::pair<DimRolesMap, TensorRolesMap>> allPatternRoles(
    IdModel& id_model,
    const std::vector<MatmulPattern>& patterns);

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

//! Automatically generates the shared memory swizzled data layout for tma loads
//! in matmul mainloop. The shared memory data layout is always 2D currently.
//! This utility function assumes that the shared_mem_tv has the following
//! structure: [tile_row, tile_col]
//! Returns which swizzle format to use for mma inputs with tma loads.
MmaInputSmemSwizzle tmaSwizzleSharedMemory(TensorView* shared_mem_tv);

} // namespace mma_utils

std::string toString(const mma_utils::AbstractMatmulTensor& abten);

} // namespace nvfuser
