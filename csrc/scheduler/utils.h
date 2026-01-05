// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#pragma once

#include <disjoint_set.h>
#include <exceptions.h>
#include <fusion.h>
#include <ir/all_nodes.h>
#include <ir/cloner.h>
#include <scheduler/reduction_heuristic.h>
#include <scheduler/tools/maxinfo_propagator.h>
#include <visibility.h>
#include "utils.h"

namespace nvfuser {

class ComputeAtMap;
class SchedulerRuntimeInfo;
class HeuristicDataCache;

//! Utility enum to signify which direction
//! transform propagation passes will propagate the transforms.
//! For example, in sharding propagation or
//! BoundedDirectionalTransformPropagator.
enum class PropagateDirection { kBackward = 0, kForward };

namespace scheduler_utils {

// Assume any only half of the register file is available to spend on buffers,
// this is because when we allocate a buffer in register is has to be accesed
// with a compile time constant index. Unfortunately nvcc seems to be using
// many registers for indexing. This is a bad estimation of extra register use,
// but it's hard to get a better one.
constexpr int64_t register_file_size_bit_full = (int64_t)256 * 1024 * 8;
constexpr int64_t register_file_size_bit = register_file_size_bit_full / 2;
constexpr int64_t register_file_size_bit_56k = (int64_t)56 * 4 * 1024 * 8;

// Empirically observed number. Not guaranteed to be a good estimate
constexpr int64_t register_overhead = 40l;
constexpr int64_t max_registers_per_thread = 255l;
constexpr int64_t bits_per_register = 4l * 8;

constexpr int64_t x_grid_limit = ((int64_t)1 << (int64_t)31) - (int64_t)1;
constexpr int64_t y_grid_limit = 65535;
constexpr int64_t z_grid_limit = 65535;
constexpr int64_t z_block_limit = 64;

// Static shared memory usage (e.g., for magic zero).
// Currently, magic zero is the only user of static shared memory and takes 4
// bytes before alignment. All shared memory in nvFuser is aligned to
// kSharedMemoryAlignmentBytes.
constexpr int64_t static_smem_usage_in_bytes = kSharedMemoryAlignmentBytes;
constexpr int64_t static_smem_usage_in_bits = static_smem_usage_in_bytes * 8;

// Find largest power of 2 that is a factor of n. If n==0, return largest power
// of 2 representable by int64_t
constexpr int64_t maxVectorizationWidth(int64_t n) {
  if (n == 0) {
    // Max representable int has null sign bit then all ones. Shift right then
    // xor to preserve only the most significant bit.
    int64_t m = std::numeric_limits<int64_t>::max();
    return m ^ (m >> 1);
  }
  // For example
  //   n               = b101101000
  //           n - 1   = b101100111
  //        ~ (n - 1)  = b010011000
  //   n & (~ (n - 1)) = b000001000
  // The key is that subtracting one flips all trailing 0s as well as the least
  // significant 1, so all of the other bits will fail the &, leaving
  // only that 1.
  return n & (~(n - 1));
}

// Largest Power of 2 less-than n
constexpr int64_t lastPow2(int64_t n) {
  NVF_ERROR(n >= 0);
  n |= (n >> 1);
  n |= (n >> 2);
  n |= (n >> 4);
  n |= (n >> 8); // NOLINT(cppcoreguidelines-avoid-magic-numbers)
  n |= (n >> 16); // NOLINT(cppcoreguidelines-avoid-magic-numbers)
  n |= (n >> 32); // NOLINT(cppcoreguidelines-avoid-magic-numbers)
  return std::max((int64_t)1, n - (n >> 1));
}

// round up to multiple of 8 or pow2 whichever smaller
constexpr int64_t roundUpPow2Or8(const int64_t x) {
  auto round_up_pow2 = lastPow2(x);
  if (round_up_pow2 < x) {
    round_up_pow2 *= 2;
  }
  constexpr int64_t kEight = 8;
  auto round_up_8 = x % kEight == 0 ? x : x + (kEight - x % kEight);
  return std::min(round_up_8, round_up_pow2);
}

constexpr int64_t roundUpPow2(const int64_t x) {
  auto round_up_pow2 = scheduler_utils::lastPow2(x);
  if (round_up_pow2 < x) {
    round_up_pow2 *= 2;
  }
  return round_up_pow2;
}

constexpr int64_t roundUpToN(const int64_t x, const int64_t n) {
  return x % n == 0 ? x : x + (n - x % n);
}

constexpr int64_t roundDownToN(const int64_t x, const int64_t n) {
  return x % n == 0 ? x : x - x % n;
}

// Div x by y, but min at 1
inline int64_t safeDiv(const int64_t x, const int64_t y) {
  return std::max(x / y, (int64_t)1);
}

// Split the given dimensions in `to_split`. Also update the dimensions in
// `to_update` to the positions in the splitted tensor. Splitting one dimension
// multiple times is supported, and if this is the case, then the order of
// `to_split` matters. All given dimensions are numbers before any split.
void splitDims(
    TensorView* tv,
    std::vector<std::pair<int64_t, int64_t>> to_split, // (dim, size)
    std::vector<int64_t>& to_update);

inline void splitDims(
    TensorView* tv,
    std::vector<std::pair<int64_t, int64_t>> to_split) { // (dim, size)
  std::vector<int64_t> unused;
  splitDims(tv, std::move(to_split), unused);
}

// Merge all the given dimensions in `to_merge` into a single dimension. Also
// update the dimensions in `to_update` to the positions in the merged tensor.
// Returns the merged dimension. All given dimensions are numbers before any
// merge.
// NOTE: merged is done as the entries in the order of `to_merge`, assuming an
// order from inner to outer
std::optional<int64_t> mergeDims(
    TensorView* tv,
    std::vector<int64_t> to_merge,
    std::vector<int64_t>& to_update);

inline std::optional<int64_t> mergeDims(
    TensorView* tv,
    std::vector<int64_t> to_merge) {
  std::vector<int64_t> unused;
  return mergeDims(tv, std::move(to_merge), unused);
}

// Merge all reduction to the right side and returns total number of
// reduction axes.
int64_t mergeReduction(TensorView* tv);

// merge all non-reduction axes to the left side and returns total number of
// iteration axes.
int64_t mergeNonReduction(TensorView* tv);

// Propagate the parallelization from the selected dimensions of the reference
// tensor to their corresponding dimensions in all selected tensors in the DAG.
// Position `pos` means selecting all the dimensions [0, 1, ..., pos - 1]. pos =
// -1 means selecting all dimensions. `selected_tvs` are selected tensors in the
// DAG. Empty `selected_tvs` means selecting all tensors in the fusion of
// `reference_tv`. `selected_parallel_types` are the selected parallel types.
// Empty `selected_parallel_types` means selecting all parallel types.
// Fusion inputs are generally ignored since parallel types (BID/TID) do not
// mean anything on them. However, we have cases during scheduling where
// propagating transforms can inadvertently remove DID parallelization from the
// inputs of the fusion and needs to reapplied. `parallelize_inputs_on_did` is a
// boolean flag that determines whether to additionally parallelize the inputs
// of the fusion on DID parallel types. For eg: see propagateReshapeTransforms
// and scheduleTranspose.
NVF_API void parallelizeAllLike(
    TensorView* reference_tv,
    int64_t pos = -1,
    std::vector<TensorView*> selected_tvs = {},
    const std::unordered_set<ParallelType>& selected_parallel_types = {},
    bool propagate_padding = true,
    bool parallelize_inputs_on_did = false);

inline void parallelizeAllLike(
    TensorView* reference_tv,
    std::vector<TensorView*> selected_tvs,
    const std::unordered_set<ParallelType>& selected_parallel_types = {},
    bool propagate_padding = true,
    bool parallelize_inputs_on_did = false) {
  parallelizeAllLike(
      reference_tv,
      -1,
      std::move(selected_tvs),
      selected_parallel_types,
      propagate_padding,
      parallelize_inputs_on_did);
}

inline void parallelizeAllLike(
    TensorView* reference_tv,
    std::initializer_list<TensorView*> selected_tvs,
    const std::unordered_set<ParallelType>& selected_parallel_types = {},
    bool propagate_padding = true,
    bool parallelize_inputs_on_did = false) {
  parallelizeAllLike(
      reference_tv,
      std::vector<TensorView*>(selected_tvs),
      selected_parallel_types,
      propagate_padding,
      parallelize_inputs_on_did);
}

inline void parallelizeAllLike(
    TensorView* reference_tv,
    const std::unordered_set<ParallelType>& selected_parallel_types,
    bool propagate_padding = true,
    bool parallelize_inputs_on_did = false) {
  parallelizeAllLike(
      reference_tv,
      -1,
      std::vector<TensorView*>{},
      selected_parallel_types,
      propagate_padding,
      parallelize_inputs_on_did);
}

// Common hyperparameters used in heuristic scheduler. These hyperparameters
// are passed to SchedulerEntry::computeHeuristics through the
// HeuristicDataCache. These hyperparameters alter the generation of the
// HeuristicParams for the scheduler.
struct SchedulerHyperParameters {
  SchedulerHyperParameters(
      int64_t vectorize_factor_,
      int64_t unroll_factor_,
      int64_t threads_per_block_min_,
      int64_t threads_per_block_max_,
      bool is_warp_specialized_)
      : vectorize_factor(vectorize_factor_),
        unroll_factor(unroll_factor_),
        threads_per_block_min(threads_per_block_min_),
        threads_per_block_max(threads_per_block_max_),
        is_warp_specialized(is_warp_specialized_) {}

  //! Number of elements to load per vectorize load.
  int64_t vectorize_factor = 1;

  //! Number of iterations to unroll for-loop.
  int64_t unroll_factor = 1;

  //! Minimum number of threads per block.
  int64_t threads_per_block_min = 1;

  //! Maximum number of threads per block.
  int64_t threads_per_block_max = 1;

  //! Use warp specialized version
  bool is_warp_specialized = false;
};

struct PersistentBufferInfo {
  std::vector<TensorView*> persistent_buffers;
  std::unordered_set<IterDomain*> unmappable_dims;

  // Tensors with unmappable dims that cannot be persistent due to
  // broadcast inling
  std::vector<TensorView*> non_persistent_buffers;

  // Persistent buffers are needed until the path through the reduction -
  // broadcast chain is resolved by any other chain using the persistent buffer
  // that is not going through a reduction. This assumes all reduction paths
  // have the same reduction pattern. Order is the same as persistent_buffers
  std::vector<std::vector<TensorView*>> persistent_buffer_resolution_points;

  // Not all persistent buffers can be projected to inputs, if a buffer can be
  // projected to the inputs which may reduce the persistent buffer size (BN
  // Backwards specifically) then keep track of it here. Persistent buffers that
  // have a persistent buffer/reduction before them should not be projected
  // through that.
  std::vector<TensorView*> projectable_persistent_buffers;

  // Track inputs of input projectable buffers
  std::vector<TensorView*> projectable_buffer_inputs;

  // Map unmappable dims to projectable_buffer_inputs
  std::unordered_set<IterDomain*> unamppable_dims_projected_to_inputs;

  // Some parameters used in
  // normalization_scheduler_utils::isProjectBufferToInput
  bool has_view_ops = false;
  bool projection_with_exp_op = false;
  bool projection_with_rng_op = false;
};

// Buffers whos roots can't map to all producer roots based on compute at. These
// are the buffers we would make persistent in a persistent kerenl or would have
// to recompute if we can't make a persistent kernel. This function will also
// return inputs as being marked persistent if they follow this pattern. It is
// important to note however inputs don't strictly have to be persistent as they
// can simply be read multiple times from GMEM in the same kernel.
PersistentBufferInfo persistentBuffers(Fusion* fusion);

// A persistent tv can be projected to its producers when all the producers are
// persistent tvs and there is no reduction op.
bool canProjectToPersistentProducer(
    TensorView* buffer,
    const std::vector<TensorView*>& producers,
    const std::unordered_set<TensorView*>& persistent_buffer_set);

//! Evaluates if a persistent buffer can be projected to input tvs without
//! dependency on reduction tvs. Returns a std::pair with a boolean indicating
//! whether projection is feasible and a vector of projectable tvs.
//!
//! The function operates in two main steps:
//! (1) Checks if the persistent buffer has dependencies on any of the given
//!     reduction tvs. If no dependencies are found, it returns true with an
//!     empty vector of target broadcast tvs.
//! (2) If there are dependencies, it examines each reduction tv for an
//!     associated broadcast tv that can be projected to. If all reduction tvs
//!     have corresponding broadcast tvs, true is returned along with these tvs.
//!     If any reduction tv lacks a corresponding broadcast tv, false is
//!     returned with the current list of identified broadcast tvs.
std::pair<bool, std::vector<TensorView*>> canProjectToInputsWithoutReduction(
    const std::vector<TensorView*> reduction_tvs,
    TensorView* persistent_buffer);

struct ReductionTvProperties {
  // How many elements in tensor view are there to reduce.
  int64_t total_reduction_numel = 1;

  // How many reductions do we need to perform, i.e. how many iter dimension.
  // elements are there
  int64_t total_iteration_numel = 1;

  // Is the inner most dimension a reduction, if no reductions mark true.
  bool fastest_dim_reduction = true;

  // How many elements in the inner most dimension merging surrounding domains
  // that match in type. This is used for 3D schedulers in
  // reduction/normalization.
  int64_t inner_most_dimension_numel = 1;

  // Same thing as above, but the number of dimensions instead of the numel.
  int64_t inner_most_dimension_ndims = 1;

  // Merging neighboring iteration domains, and reduction domains, what's the
  // resulting dimensionality of the problem.
  int64_t dimensionality = 1;
};

// Fill ReductionTvProperties structure about tv
ReductionTvProperties getReductionProperties(
    Fusion* fusion,
    SchedulerRuntimeInfo& runtime_info,
    TensorView* tv);

// Struct to store persistent buffer sizes. also holds the persistent buffer
// size of the buffers are projected to the inputs.
struct PersistentBufferSizeReturn {
  int64_t persistent_buffer_size_bit = 0;
  int64_t projected_persistent_buffer_size_bit = 0;
};

// Compute the amount of register space would be needed to perform this kernel
// persistently, only based on buffers that must be persistent, and based on the
// maximum of all minimum size requirement. i.e. if must be persistent, only
// hold persistent dimension.
PersistentBufferSizeReturn persistentBufferSizeBit(
    Fusion* fusion,
    SchedulerRuntimeInfo& runtime_info,
    const PersistentBufferInfo& persistent_buffers,
    HeuristicDataCache* data_cache = nullptr);

// Merges tensor view to the form:
// [IterationDomain, ReductionDomain] Returns if <iteration dimensions,
// reduction dimensions>
std::pair<bool, bool> canonicalDimReduction(
    Fusion* fusion,
    TensorView* tv,
    bool schedule_3D = false);

// Return a list of tensor views that are outputs of reduction operations,
// excluding resharding reduce expressions. If multiple outputs of an expression
// are found, only include one in the list
std::vector<TensorView*> getReductionTvs(Fusion* fusion);

// Returns a list of TensorViews that are the consumer tv for a view operation.
std::vector<TensorView*> getViewTVs(Fusion* fusion);

// Returns a list of non-reduction TensorViews that have a root domain
std::vector<TensorView*> getTVsWithNonReductionRFactor(Fusion* fusion);

// Reset inputs and outputs to global memory, everything else to local.
void clearMemorySpace(Fusion* fusion);

// Returns the pairs of <cache, input_index> for each cached fusion input.
// input_index is the position in fusion->inputs(). Otherwise return empty
// vector.
std::vector<std::pair<TensorView*, int64_t>> cacheInputs(
    Fusion* fusion,
    bool unroll);

// Returns the pairs of <cache, output_index> for each cached fusion output.
// output_index is the position in fusion->outputs(). Otherwise return empty
// vector.
std::vector<std::pair<TensorView*, int64_t>> cacheAndForkOutputs(
    Fusion* fusion,
    bool unroll);

// Ignores broadcast and reduction, returns iter domain in allocation domain
// that's "inner most".
IterDomain* innerMostAllocDim(TensorView* tv);

// Looks through fusion and finds all dims that match to the one provided in
// the tensorview provided. Iter domain must be a root domain. If inner_only,
// will only map dimensions if they're the inner most position. This is
// important when projecting a dimension between an rfactor position and its
// root position when mapping from consumer to producer. If inner_only=true,
// takes the rfactor/root dimensions that maps, projects it to the root/rfactor
// domain, but only following the inner most pass when encounting split/merge.
// When propagating backward, for split it will only propagate backwards if the
// mapped dimension is the inner portion of the split. For merge, inner_only
// doesn't make a dimension and will propagate through the inner portion of the
// merge. When propagating forward, the logic is symmetric with the backward
// case.
class FindAllMappedDims : public MaxInfoSpanningTree::Propagator {
  std::unordered_map<TensorView*, IterDomain*> mapped_root_ids_;
  std::unordered_map<TensorView*, IterDomain*> mapped_allocation_ids_;
  TensorView* starting_tv_ = nullptr;
  IterDomain* starting_id_ = nullptr;
  bool inner_only_;
  bool vectorize_pass_;

 public:
  FindAllMappedDims(
      TensorView* from,
      IterDomain* starting_id,
      bool inner_only,
      bool vectorize_pass);
  void setUp() override;
  void propagateC2P(TensorView* from, TensorView* to) override;
  void propagateP2C(TensorView* from, TensorView* to) override;
  void propagateSibling(TensorView* from, TensorView* to) override;
  std::unordered_set<IterDomain*> get() const;
};

// Checks if tensor view has an iteration domain in vector dims in its inner
// most root position (excluding broadcast and reduction), and checks if it is a
// contiguous dimension
bool hasInnerDim(
    TensorView* tv,
    std::unordered_set<IterDomain*> vector_dims,
    bool should_vectorize);

// Returns all inputs and outputs that share the inner most dimension of the
// provided reference. If reference is an input it ignores reduction axes, will
// ignore all broadcast axes. If inner_only, will require inner->inner mapping
// in view, otherwise, it allows all inner->any mapping. If vectorize_pass, will
// check contiguity for vectorization, otherwise it just checks it has that
// inner dim.
std::vector<TensorView*> getInputsOutputsWithInnerDim(
    TensorView* reference_tv,
    bool inner_only,
    bool vectorize_pass);

// Holder return struct for the below function.
struct DisjointLogicalSetInfo {
  // const* to the disjoint set in disjoint_rfactor_set passed in to
  // getDisjointLogicalSetsOf each iterdomain in the rfactor of ref is mapped
  // to.
  //
  // WARNING: these pointers are relative to the disjoint_rfactor_set reference
  // passed into getDisjointLogicalSetsOf it's the user's responsibility to
  // maintain the lifetime of that reference to match this vector.
  std::vector<const VectorOfUniqueEntries<IterDomain*>*> disjoint_sets_of_ref;

  // Unique ID associated to the disjoint view group the logical id belongs to
  // in disjoint_sets_of_ref. It's straight forward to map from
  // disjoint_sets_of_ref to the vector, but not the other way around.
  std::vector<int64_t> disjoint_set_ids;

  // TensorView reference the above vectors are relative to.
  TensorView* ref;
};

// Returns disjoint rfactor sets mapped onto the given reference. Returns a pair
// of vectors of size rfactorDomain of reference. Vector of
// VectorOfUniqueEntries returns a const* to the disjoint set in
// disjoint_rfactor_set the iterdomain is mapped to. Integer vector represents
// which disjoint rfactor group the logical id belongs to. It's straightforward
// to map from the former to the latter, but not the latter to former.
//
// Since we return a const* to entries in disjoint_rfactor_set, it must be
// passed in as a reference. Algorithm is N^2 based on number of dims in
// reference, but generating the disjoint rfactor set is likely the limiter on
// perf of this function.
//
// logical_reorder_map is provided to assume TensorView `of` will be reordered
// per the map
DisjointLogicalSetInfo getDisjointLogicalSetsOf(
    Fusion* fusion,
    TensorView* of,
    DisjointSets<IterDomain*>& disjoint_rfactor_set,
    const std::unordered_map<int64_t, int64_t>& logical_reorder_map = {});

// Structure to hold byte multiples for break points. I.e. if we have the
// tensors:
// T0[I0, I1] float
// T1[I0, I1] bool
// T2[I0]     half
// T3    [I1] double
// and a break point of 1 the multiples would be:
// lhs_multiple = 4 + 1 + 2 = 7
// rhs_multiple = 4 + 1 + 8 = 13
struct BroadcastMultiple {
  int64_t rhs_multiple = 0;
  int64_t lhs_multiple = 0;
};

struct BroadcastMultipleInformation {
  std::vector<int64_t> view_disjoint_set_ids;
  std::vector<BroadcastMultiple> broadcast_multiples;
};

// Returns a vector of size reference_tv->getLogicalDomain().size() which
// is a view disjoint set id of each of those iter domains. If entries share the
// same value, they undergo view transformations in the fusion together.
// Broadcast multiples are also of size
// reference_tv->getLogicalDomain().size(), each entry [i] is the number of
// inputs/outputs that have a non-broadcast dimension mapped to the
// corresponding dimension in reference_tv. Broadcast multiples includes
// reference_tv if reference_tv is an input or output. Broadcast multiples is
// multiplied by data type size. In the case of view operations the broadcast
// multiple is the full multiple size if any domain in the group maps to a
// non-broadcast dimension in the given input/output. Otherwise if all
// dimensions are broadcast that input/output will not contribute to the
// multiple.
//
// logical_reorder_map is provided to assume reference_tv will be reordered per
// the map
BroadcastMultipleInformation getBroadcastMultiples(
    TensorView* reference_tv,
    DataType index_type,
    const std::unordered_map<int64_t, int64_t>& logical_reorder_map = {});

//! Propagate current transformations on from_tv up to the given
//!  position, to all tensorviews on the owning fusion that has
//!  a connection with `from_tv` on the fusion graph.
void transformPropagateToAllFrom(TensorView* from_tv, int64_t pos);

//! A type of custom transform propagator that propagates iterdomain
//!  transforms from a source tv to all tvs that are selected
//!  using a "direction" and a "boundary".
//!
//! The propagation model always assumes a `from_tv`, a `direction` and a
//! `boundary`.
//!
//! This propagator will only transform producers and consumers
//! of `from_tv`, and all propagation modes **require** a boundary to be
//! specified to signify where the propagation should stop.
//!
//! There are currently three modes of propagation: forward, backward and
//! both-way, see comment on the interface functions for details.
struct NVF_API BoundedDirectionalTransformPropagator {
  //! Custom option container for configuring
  //!  the transform propagation actions.
  //! All option values default to false unless
  //!  the corresponding setter is called.
  struct Options {
    //! If true, the transform propagator will
    //!   also propagate parallel types from
    //!   `from_tv` to all selected tvs.
    bool propagate_parallel_type = false;

    //! If true, the specified boundary tvs
    //!  will also be replayed as `from_tv`.
    //!  If false, they will not be affected
    //!  by the propagation pass.
    bool transform_boundary = false;

    //! Sets the position boundary in parallel
    //!  type propagation, see comment on
    //!  scheduler_utils::parallelizeAllLike.
    //! Only used if propagate_parallel_type==true.
    int64_t parallel_propagation_pos = -1;

    //! Setter for enabling parallel type
    //!  propagation. see comment on the variable.
    //!
    //! \param up_to_pos, sets the parallel type
    //!  propagation boundary. see comment on
    //!  scheduler_utils::parallelizeAllLike.
    Options propagateParallelType(int64_t up_to_pos = -1) {
      propagate_parallel_type = true;
      parallel_propagation_pos = up_to_pos;
      return *this;
    }

    //! Setter for enabling propagation to
    //!  boundary tvs. see comment on the variable
    Options propagateToBoundary() {
      transform_boundary = true;
      return *this;
    }
  };

  //! Replay transforms from tensorview `from`
  //!  to the tensorviews that are consumers
  //!  of boundary tensorviews in `to` and producers of `from`.
  static void backward(
      TensorView* from,
      int64_t pos,
      std::vector<TensorView*> to,
      std::optional<Options> options = std::nullopt);

  //! Replay transforms from tensorview `from`
  //! to the tensorviews that are producers
  //!  of boundary tensorviews in `to` and consumers of `from`.
  static void forward(
      TensorView* from,
      int64_t pos,
      std::vector<TensorView*> to,
      std::optional<Options> options = std::nullopt);

  //! Replay transforms from tensorview `from`
  //!  to all the tensorviews that are consumers
  //!  of tensorviews in `backward_to` and producers
  //!  of tensorviews in `forward_to` while being
  //!  either a producer or a consumer of tensorview `from`.
  static void bothWays(
      TensorView* from,
      int64_t pos,
      std::vector<TensorView*> backward_to,
      std::vector<TensorView*> forward_to,
      std::optional<Options> options = std::nullopt);

 private:
  //! Utility function:
  //!  Will realize the transform propagation to the
  //! tensorview's in `included_tvs`.
  //!  Assumes that all tvs in included_tvs are either
  //! a producer or a consumer of from_tv.
  static void propagate(
      TensorView* from_tv,
      int64_t pos,
      std::unordered_set<TensorView*> included_tvs,
      Options options);
};

// Schedulers typically start by merging some axes together then splitting,
// and propagating those transformations through the dag. What we want to
// understand is if these merges can be supported through view operations.
// For example it could be problematic to support a reduction fusion:
//
// tv0[2, 3, 4]
// tv1 = sum(tv0, {1, 2})
// tv2 = view(tv0, {6, 4})
//
// Since the first step of the reduction scheduler would be tv1->merge(1, 2).
// If we tried to propagate this transformation through the view it would make
// the view invalid. If we tried to propagate the view through the reduction,
// it would attempt to merge a reduction and non-reduction dimension. So for
// these types of fusions we would like to understand that the view considers
// axis 1 and 2 of tv1 as "non-separable" axes.
//
// If IterDomains are disjoint in the returned set, then they are considered
// "separable".
// Warning: This pass generates the IdGraphs, not intended for use at runtime.
DisjointSets<IterDomain*> disjointLogicalSets(Fusion* fusion);

// Makes sure that there are no group id's left of pos that match right of pos.
// e.g.
// [1, 0, 0] pos 2 would return false
// [1, 0, 0] pos 1 would return true
bool breakIsDisjoint(std::vector<int64_t> group_ids, int64_t pos);

// Update the vector of ids_to_transform as progressing through the
// `transform_exprs`. We'll always insert the result of split in the
// location of the input, and insert the merge result in the position of the
// inner dimension. Optionally accepts a callback after each transform is
// applied for analysis of the expr nodes.
void applyTransforms(
    std::vector<IterDomain*>& ids_to_transform,
    const std::vector<Expr*>& transform_exprs,
    std::optional<std::function<void(Expr*)>> post_transform = std::nullopt);

// Generates a permutation to reorder tv's domain as the logical order.
// Priority is given to inner most dimensions for example:
// logical [i0, i1, i2]
// domain [i0*i2, i1]
// will produce the permutation {1, 0}
// This is somewhat similar to orderTiledConcreteIdAsRoot
std::vector<int64_t> domainReorderAsLogicalMap(TensorView* tv);

// Generates an old to new map to reorder tv's logical domain as its allocation
// order. Allocation domain is canonicalized to find a permutation of the
// logical domain that satisfies the order in allocation domain.
std::unordered_map<int64_t, int64_t> reorderLogicalAsAllocationMap(
    TensorView* tv);

// Generates an old to new map to reorder tv's loop domain as its allocation
// order. Allocation domain is transformed to find a permutation of the loop
// domain that satisfies the order in allocation domain.
std::unordered_map<int64_t, int64_t> reorderLoopAsAllocationMap(TensorView* tv);

// Assumes view's are consistent as detected by
// registery.cpp::requiresForwardViewReplay returning false
void propagateReshapeTransforms(Fusion* fusion);

//! Check if tv is an output of a fastest-dim reduction
bool isFastestDimReduction(TensorView* tv);

//! Certain tensors may need to be placed on shared or global memory
//! due to data dependencies caused by resize operations. Create
//! caches of those tensors so that original operations producing
//! them should keep using the same memory. This avoids, for example,
//! reductions to global memory.
//!
//! Example:
//!
//! tv1 = sum(tv0)
//! tv2 = some_resize_op(tv1);
//! tv3 = some_other_op(tv1);
//!
//! When tv1 is promoted to Global, we want to avoid reducing to a
//! global memory tensor. After the transformation by this function,
//! the fusion should look like:
//!
//! tv1 = sum(tv0);
//! tv4 = tv1
//! tv4->setMemoryType(Global)
//! tv2 = some_resize_op(tv4)
//! tv3 = some_other_op(tv1);
//!
//! Note that the sum reduction is done using a Local buffer, i.e.,
//! tv1, but the data dependency for the resize op is still satisfied
//! by having a copy of tv1, i.e., tv4. Note that the other op using
//! tv1 still uses tv1.
void prepareForMemoryTypePromotion(Fusion* fusion);

//! If a consumer tensor induces a data dependency between threads,
//! move its producer to a shared memory that is sufficient to satisfy
//! the dependency. For example, if the domain is parallelized
//! with blockIdx, the producer memory type will be changed to
//! Global. A proper RAW sync will be automatically inserted when the
//! fusion is lowered.
void promoteProducerMemoryTypes(
    Fusion* fusion,
    const std::vector<std::pair<TensorView*, int64_t>>& input_caches);

//! Get all tensors that are connected to from_tvs without going through
//! any tvs in the cutoff_tv_set.
std::unordered_set<TensorView*> getAllTvsFrom(
    const std::vector<TensorView*>& from_tvs,
    const std::unordered_set<TensorView*>& cutoff_tv_set);

//! Get the persistent buffer size of a tensor
int64_t getPersistentBufferSizeBitOfTensor(
    const TensorView* buffer,
    SchedulerRuntimeInfo& runtime_info,
    const PersistentBufferInfo& persistent_buffer_info);

//! The required shared memory size for a block includes two parts: (1) smem
//! for persistent buffers and (2) reduction workspace which depends on the
//! number of threads per block specified by the parameter threads_per_block.
//! By default, the function uses the maximum allowed number of threads per
//! block (threads_per_block = -1) to calculate the overhead. The caller can
//! specify a different value if they are sure about the max value used at
//! runtime.
int64_t getReductionSmemWorkspaceBit(
    Fusion* fusion,
    const std::vector<TensorView*>& reduction_tvs,
    int64_t threads_per_block = -1);

// Returns true if any Expr in `fusion` is resharding.
bool isResharding(Fusion* fusion);

// Move non-concretized broadcast domains to innermost
// positions. Broadcast domains mapped with any domains of given tvs
// are ignored.
//
// The goal here is to find domains that are not scheduled by
// propagation from reference tensors (i.e., ignored_tvs). All
// schedulers make sure to include only schedulable domains but they
// may also allow to have non-concretized broadcast domains that have
// no mapping with any of reference tensors. Since they are
// non-concretized, they should be safe to ignore. Ideally, they
// should just be removed from the fusion. For now, they are moved to
// innermost positions to prevent them from interfering
// inlining. If they happened to be at the
// outermost position, the tensor wouldn't be inlined at all. See
// issue #2686 and PR #2799.
void moveNonConcretizedBroadcastInnermost(
    Fusion* fusion,
    const std::unordered_set<TensorView*>& ignored_tvs = {});

// Returns a factor represents the computation cost of the given fusion.
// Estimated using the number of MUFU operations, each weighted with a
// predefined factor.
int64_t getComputationCostFactor(Fusion* fusion);

// Returns the required bits in flight to saturate the memory bandwidth.
int64_t getRequiredBitsInFlight();

// Returns true if the device has a high bandwidth to compute raito.
bool isHighBandwidthFlopsRatio();

// Return true if the fusion has computation requires Floating-Point
// Multi-Function (MUFU) units, e.g. cos, sin, exponent, logarithm, sine,
// cosine, square root, hyperbolic tangent. Currently, we only tested tanh, exp,
// and Reciprocal. Note that, if compiled with fast math (not supported yet) or
// directly lowered with inlined ptx, needs to revise the inner reduction
// heuristics which uses this function to set the optimal unroll factor.
bool hasExpensiveMUFUops(Fusion* fusion);
// Reorder DID parallelized axes to outermost positions. Returns
// the position of the outermost non-DID axis.
int64_t reorderDevicesToOuter(TensorView* tv);

// Returns number of non-reduction/non-broadcas/non-device dims in logical
// domain
inline int64_t nLogicalDims(const TensorView* tv) {
  auto logical_dom = tv->getLogicalDomain();
  int64_t tv_n_dims = 0;
  for (auto dim : logical_dom) {
    if (!dim->isReduction() && !dim->isBroadcast() && !dim->isDeviceDim()) {
      tv_n_dims++;
    }
  }
  return tv_n_dims;
}

// Get a permutation vector to reorder a given domain to align with a
// given list of reference IDs. Non-matching loop IDs are placed outermost
// positions.
std::vector<int64_t> reorderDomainLike(
    const std::vector<IterDomain*>& domain,
    const std::vector<IterDomain*>& ref);

// If buffer_tv's definition is an upcast and the input to the cast is not a
// fusion input, return input to the cast. Otherwise, return nullptr. Used to
// recompute buffer_tv from its producer to save register/smem usage. Fusion
// input is skipped as it is handled by project to inputs.
TensorView* getUpCastInputOf(const TensorView* buffer_tv);

//! Given an input TV, try and schedule trivial ops as global to global ops that
//! will be skipped at lowering by modifying their allocation domains and memory
//! types. Returns the last resulting global consumer of the given TV: its
//! definition and those of all its producers will be skipped during lowering
//! with the tensor producer alias mechanism.
//! See device_lower/analysis/tensor_producer_aliases.h
TensorView* scheduleInputToSkipIntermediates(TensorView* tv);

// Returns true if any of the domains of the tensor is symbolic
bool isSymbolicTensor(const TensorView* tv);

// Builds the allocation domain of `tv` by reusing existing IDs from the loop
// domain. This avoids creating duplicate IDs when the loop domain already
// contains the transformed IDs we need. It ensures we can allocate the tensor
// based on its allocation domains and also verify that the allocated Ids are
// consistent with the compute at position.
void buildAllocationDomainFromLoopIds(TensorView* tv);

// For shared memory tensor, replay loop domain transformations to allocation
// domain
void buildAllocationDomainForSharedMemoryTvs(Fusion* fusion);

// Returns the maximum cluster size that can be used for the current device.
// Uses cuOccupancyMaxPotentialClusterSize to query the hardware directly,
// guaranteeing at most a single CTA per SM by requesting the maximum smem per
// CTA. Results are cached per device to avoid redundant queries. Returns 1 for
// pre-Hopper devices.
int64_t getMaxClusterSize();

//! Returns the number of clusters that can be active at once with the given
//! size, assuming a single resident CTA per SM.
//!
//! Note: This function uses maximum shared memory (not actual usage) to enable
//! caching results by cluster size, avoiding redundant queries for each call.
int64_t getMaxActiveClusters(const int64_t cluster_size);

// ============================================================================
// TMA (Tensor Memory Accelerator) Background
// For details see doc/dev/tma.md
// ============================================================================
// TMA is a hardware feature on NVIDIA GPUs that allows efficient loading of
// multi-dimensional tiles from global memory to shared memory. Instead of
// individual threads loading data, TMA enables hardware-accelerated bulk
// transfer of multi-dimensional tiles with a single instruction.
//
// Key TMA Concepts in nvFuser:
//
// 1. TMA Domain: A "virtual" view of how we think about the problem
//    dimensionality. For example, pointwise operations on a tensor of any shape
//    can be viewed as a 1D problem by flattening all dimensions. However, for
//    TMA scheduling, we typically use a 2D view to better utilize the hardware:
//    (1) Since each dimension only allows 256 elements, using 2D TMA allows us
//        to load a large number of elements, which is essential to achieve high
//        bandwidth.
//    (2) 2D TMA allows us to better re-use broadcasted data.
//
// 2. Box/Tile: The multi-dimensional region of data loaded by a single TMA
//    instruction. In TMA terminology, a "box" is a dense region,
//    while a "tile" can be a strided subset of a box. However, the TMA
//    pointwise scheduler always uses dense tiles, so box == tile.
//
//    For the pointwise scheduler, box and tile are identical and refer to
//    the contiguous multi-dimensional region loaded in one operation. For
//    example, a box/tile of size (8, 4) loads 8×4 = 32 contiguous elements
//    arranged in an 8-row by 4-column layout. Throughout this documentation,
//    "box" and "tile" are used interchangeably.
//
// ============================================================================
// Purpose of This Function
// ============================================================================
// The TMA pointwise scheduler views tensors as 2D domains: [tma_domain_outer,
// tma_domain_inner]. This function computes the optimal size for the
// "tma_domain_inner" dimension of this 2D view, given a flattened tensor of
// total_element items.
//
// The transformation flow is:
//   [total_element]                     # Flattened 1D tensor
//   -> [tma_domain_outer, tma_domain_inner] # Split into 2D TMA domain
//
// Where:
//   tma_domain_inner = return value of this function
//   tma_domain_outer = total_element / tma_domain_inner
//   total_element % tma_domain_inner == 0
//
// ============================================================================
// Parameters
// ============================================================================
//
// total_element:
//   Total number of elements in the flattened tensor. Must be divisible by
//   (2 * 16 / min_dtype_bytes) to satisfy 2D TMA alignment requirements.
//
//   Hardware constraint details:
//   - We use TMA without interleave; the byte size of the innermost TMA tile
//     must be divisible by 16 bytes.
//   - 2D TMA requires at least 2 tiles in the inner dimension.
//   - Therefore, the inner TMA domain size must be at least 2 * 16 bytes,
//     or (2 * 16 / min_dtype_bytes) elements.
//
// tma_domain_inner_target (default: 512):
//   Target size for the inner TMA domain. The function finds the divisor of
//   total_element closest to this target that satisfies all constraints.
//
//   Why 512 instead of 256?
//   Using 512 provides a safety margin to avoid "dimension collapse" - a
//   situation where a dimension has only 1 tile and gets virtually merged with
//   its neighbor, breaking the assumption of a 2D TMA structure.
//
//   Example of dimension collapse with 256:
//     Step 1: [total_element] -> [total_element/256, 256]  # 2D TMA domain
//     Step 2: Further split both dimensions to create 2D TMA tiles.
//             After tiling with tma_domain_inner=256:
//       [total_element/256/tma_domain_outer, tma_domain_outer, 256/256, 256]
//       = [total_element/256/tma_domain_outer, tma_domain_outer, 1, 256]
//
//     Problem: In [..., tma_domain_outer, 1, tma_domain_inner], since the
//     middle dimension is 1, tma_domain_inner is contiguous with
//     tma_domain_outer in the original tensor. This effectively creates a
//     single TMA virtual dimension of size (tma_domain_outer *
//     tma_domain_inner), which is subject to the 256 element limitation and may
//     fail. Note: It failes when tma_domain_outer * tma_domain_inner > 256,
//     becuase
//           MergeTileGroupsByRotation merges contiguous bulk dimensions,
//           but lacked the 256-element hardware constraint check.
//
//   With 512, even if tma_domain_inner=256:
//     [total_element/512, 512]
//     -> [total_element/512/tma_domain_outer, tma_domain_outer, 512/256, 256]
//     = [total_element/512/tma_domain_outer, tma_domain_outer, 2, 256]
//
//     We maintain a proper 2D structure with 2 tiles in the inner dimension,
//     preventing dimension collapse.
//
// min_dtype_bits:
//   Size in bits of the smallest data type in TMA-loaded tensors. Used to
//   ensure that the innermost TMA box dimension satisfies the 2 x 16-bytes
//   (256-bit) alignment requirement.
//
// ============================================================================
// Returns
// ============================================================================
// The size of the inner dimension of the 2D TMA domain. This value:
//   - Divides total_element evenly
//   - Is divisible by (256 / min_dtype_bits)
//   - Is as close as possible to tma_domain_inner_target
//   - Returns 1 if no suitable divisor exists (signaling TMA is not viable)
//
// ============================================================================
int64_t getTmaDomainInner(
    int64_t total_element,
    int64_t tma_domain_inner_target = 512,
    int64_t min_dtype_bits = 8);

// Calculate register sharing between TMA async threads and computation threads
// for warp specialization. Returns a pair of (tma_branch_registers,
// compute_branch_registers).
//
// Assumes padded threads keep [tma_branch_registers] registers and all others
// are moved to computation threads. The granularity is 8. When estimated
// compute_branch_regs is not divisible by granularity, it is rounded down and
// tma_branch_registers is recomputed.
//
// For example, assuming 256 computation threads, initial register = 168,
// tma_branch_regs = 32. then (168 - 32) * 128 / 256 = 68 which is not
// divisible by 8, compute_branch_registers = 168 + 68 = 236 --> rounded down to
// 232. re-calculate [tma_branch_registers] using: borrowed registers = (232 -
// 168) * 256 / 128 = 128. tma_branch_registers = 168 - 128 = 40
std::pair<int64_t, int64_t> getRegisterSharing(
    int64_t reg_per_thread,
    int64_t computation_threads,
    int64_t padded_threads);
} // namespace scheduler_utils
} // namespace nvfuser
