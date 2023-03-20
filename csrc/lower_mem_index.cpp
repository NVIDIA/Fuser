#include <c10/util/Exception.h>
#include <ir_builder.h>
#include <ir_utils.h>
#include <lower2device.h>
#include <lower_mem_index.h>
#include <lower_utils.h>
#include <root_domain_map.h>

namespace nvfuser {

// [Notes on memory index lifting]:
// The memory index lifting pass tries re-associate the index math
//  and pre-compute the heavy portion of the index math outside of
//  inner loops to reduce register pressure and integer math strength.
//
// The optimization is based on 2 key observations:
//  1. Memory instructions are able to take Immediate operands, i.e.
//  compile-time
// constants instead of registers, so the register usage can be saved: e.g.:
//   load ... [R0]
//   load ... [R1]
//   load ... [R2]
//   load ... [R3]
//
// would take 4 registers but when we have the opportunity to transform above
// code into:
//
//   load ... [R0]
//   load ... [R0+4]
//   load ... [R0+8]
//   load ... [R0+12]
// The same sequence of instructions now use 1 register instead of 4.
//
// The optimization in this pass tries to convert code to the above pattern in a
//  specific but very common family of cases (details see below).
//
// 2. Both register usage and math hotpaths tend to be confined within a serial
// loop
//  that's not unrolled, with the main loop in matmul kernels being a very
//  popular example. So this pass focuses on optimizing the register usage by
//  maximizing converting index math to the pattern mentioned above when there
//  is a serial loop that's not unrolled.
//
// The following example shows the optimization pattern that this pass is trying
// to do:
//
// Before:
//
// for tidx in ...      // parallel loop
//  for i in 0...T.SIZE  // serial loop
//   for j in 0..32      // unrolled loop 1
//    for k in 0..8      // unrolled loop 2
//      ... = T0[index(i, j, k, tidx)];
//
// After:
//
// alloc T1[32];
// for tidx in ...      // parallel loop
//   for j in 0..32      // unrolled loop
//     T1[j] = index(0, j, 0, tidx);
//
// for tidx in ...      // parallel loop
//  for i in 0...T.SIZE  // serial loop
//   for j in 0..32      // unrolled loop 1
//    for k in 0..8      // unrolled loop 2
//      ... = T0[T1[j]+i*123+k*456];
//
// [Separability Analysis]:
// A term "Separability" is introduced here to describe further detail into the
//  above optimization.
//
// An loop/iterdomain `i` is separable in the index function index(i, j, k,
// tidx,...) iff
//  index(i, j, k, tidx,...) = index(0, j, k, tidx,...) + i * C, where C is a
//  compile time constant integer.
//
// This property is needed for this optimization to be profitable as only
// constant addition can be inlined
//  into memory address expressions.
//
// An example of separable iterdomains:
//                   Id0
//                    |
//                  split
//                  /   |
//                Id0o  Id0i (32)
//                       |
//                     split
//                     /   |
//                  Id0io  Id0ii(4)
//
// In this case Id0o, Id0io, Id0ii are all separable, as the indexing on Id0 can
// be
//  written as
//     indexOf(Id0o)*32 + indexOf(Id0io)*4 + IndexOf(Id0ii)
//
// Another example involving non-separable iterdomains:
//          Id0    Id1
//           |     /
//            merge
//              |
//             Id3
//
// Id3 is not separable as index on Id0, for example is:
//  IndexOf(Id3) / Id1.extent, which cannot be rewritten as IndexOf(Id3) * C,
//  with
// C being a compile constant integer.

// Note [Predicate Lifting]:
// The separability analysis can be extended to save registers on predicates
// too,
//  which could be used as a short term alternative to binary predicate vector
//  packing.
// Could consider removing this variant of index lifting once the predicate
// vector
//  packing is fully sorted out.
//
// An example here would help illustrate how this variant helps saving registers
// in
//  predicate evaluation:
//
// Say we have the following loop:
//   for in i in 0..8:
//     if index(i, threadIdx, blockIdx) < T0.size: // predicate1
//        T0 = ...
//   In above code predicate1 would consume at least 8 registers
//     because the loop is unrolled, and in reality actually a lot
//     more because the index function tends to be quite convoluted
//     and takes addtional registers to evaluation.
//
//  With the separability assumption described above, where
//   index(i, threadIdx, blockIdx) can be re-written as:
//   index(0, threadIdx, blockIdx) + i * C, where C is a grid constant
//
//  The above code can be re-written as:
//   nvfuser_index_t base_index = index(0, threadIdx, blockIdx);
//   for in i in 0..8:
//     if base_index < T0.size-i*C: // predicate2
//        T0 = ...
//   Note that the expression on the RHS of predicate2 are all grid constants
//     which the underlying cuda compiler does a good job optimizing out.
//   So the entire predicate evaluation now only takes 1 register aka the
//   base_index.
//
//  Additional note:
//
//  The support for this optimization is very limited and currently only
//  supports
//    the usage within matmul scheduler. To extend for generic support there are
//    a few things that'd need to be sorted out: (if we do want to go further
//    with this route instead of moving to predicate vector packing.)
//
//  1. In the predicate indexing today, we do index compute once and share it
//  for all the contig_id's, which is ok if we don't consider index lifting
//  but would probably need one index compute pass per contig id to accomodate
//  lifting different contig_id's on the same tensor.
//
//  2. Matching the contig_id's is a very important aspect of the lifting
//  transform, and it cannot be done well under the current infrastructure for
//  predicate evaluation. The most ideal case would be all the predicate math
//  could be computed, through iterdomain graph, before materialization of any
//  loop nest, and that way all the separability analysis could be done reliably
//  on the fusion IR itself.
//
//  3. Related to point 1.:
//    - see [Restriction on Predicate Hoisting]:
//      only one contig_id is lifted from each tensor because we only have one
//    instance of index compute to update.
//    - see [Predicate Peeling - Predicate Lift WAR]
//      the predicate lifting has many unsupported cases because of having
//      only one index compute instance. Particularly, whenever there's any
//      predicate variant on the expression that requires the un-lifted indexing
//      the predicate lifting should be disabled.
//
//  4. There are currently quite a few predicate variants, see
//  lower_predicate.cpp,
//    and would need to revisit the interaction with all of them.

AddressRecord::AddressRecord(
    TensorView* data_tv,
    TensorView* address_tv,
    std::vector<IterDomain*> allocation_ids,
    TensorView* reference_tv,
    ReadWrite direction,
    IterDomain* loop_concrete_serial_id)
    : address_tv_(address_tv),
      data_tv_(data_tv),
      reference_tv_(reference_tv),
      access_direction_(direction),
      allocation_ids_(allocation_ids),
      loop_concrete_serial_id_(loop_concrete_serial_id) {}

AddressRecordKey AddressRecord::key() const {
  return AddressRecordKey(reference_tv_, data_tv_);
}

void AddressComputeInfo::build(Fusion* fusion) {
  for (auto expr : fusion->unordered_exprs()) {
    if (!ir_utils::isTvOp(expr)) {
      continue;
    }

    for (auto consumer_tv :
         ir_utils::filterByType<TensorView>(expr->outputs())) {
      if (consumer_tv->shouldLiftWriteAddress()) {
        // Build write address record if consumer index should
        //  be lifted.
        makeAddressRecord(consumer_tv, consumer_tv);
      }
      if (consumer_tv->shouldLiftPredicateIndex()) {
        // Build predicate record if the expression defininig
        //  consumer it is predicate lifted.
        makePredicateRecord(consumer_tv);
      }
      for (auto producer_tv :
           ir_utils::filterByType<TensorView>(expr->inputs())) {
        if (producer_tv->shouldLiftReadAddress()) {
          // Build read address record if producer index should
          //  be lifted.
          makeAddressRecord(producer_tv, consumer_tv);
        }
      }
    }
  }
}

namespace {

// Given a vector of root iterdomains and their contiguity
// Returns the resulting intermediate iterdomains after all the contiguous
//  merges. A contiguous merge is a merge expr that takes two consecutive
//  iterdomains that are both contiguous, and thus producing another contiguous
//  merged iterdomain as it's output.
//
// Example input: given the following tensor domain:
//
// contiguity = true, true, false, true
//   Id0, Id1, Id2, Id3
//    |   /     |
//     merge    |
//      |       |
//     Id4      |
//      |      /
//        merge
//         |
//        Id5
//
// This function with arguments ({Id0, Id1, Id2, Id3}, {true, true, false,
// true})
//  returns:
//    Id4, Id2, Id3, contig_set =  {Id4}
// TODO:
//  Similar analysis exist in contig finder and should consider unifying
// the logic paths.
std::pair<std::vector<IterDomain*>, std::unordered_set<IterDomain*>>
getContigMergeFrontier(
    std::vector<IterDomain*> domains,
    std::vector<c10::optional<bool>> contiguity,
    const std::unordered_map<IterDomain*, Expr*> id_to_use_map) {
  std::list<IterDomain*> domain_list{domains.begin(), domains.end()};
  std::unordered_set<IterDomain*> contiguity_id_set;

  TORCH_INTERNAL_ASSERT(domains.size() == contiguity.size());

  // Keep track of the set of Id's that are contiguous
  for (auto idx : c10::irange(domains.size())) {
    auto id = domains[idx];
    if (id->isBroadcast()) {
      continue;
    }
    if (*contiguity[idx]) {
      contiguity_id_set.insert(id);
    }
  }

  // Track if any update has been found,
  //  keep iterating until no more update is found.
  bool update = true;

  while (update) {
    update = false;

    // Iterate over the current domain list, and
    //  try to find 2 consecutive iterdomains that
    //  are both contiguous and merged together.
    auto domain_it = domain_list.begin();

    while (domain_it != domain_list.end()) {
      auto first_domain = *domain_it;
      auto second_domain_it = std::next(domain_it);
      if (second_domain_it == domain_list.end()) {
        // No next domain to merge with.
        //  no need to keep moving forward
        //  in this round.
        break;
      }
      auto second_domain = *second_domain_it;

      if ((!first_domain->isBroadcast() &&
           !contiguity_id_set.count(first_domain)) ||
          (!second_domain->isBroadcast() &&
           !contiguity_id_set.count(second_domain))) {
        // Only contiguous pairs of iterdomains are considered.
        domain_it++;
        continue;
      }

      auto first_domain_use_it = id_to_use_map.find(first_domain);
      TORCH_INTERNAL_ASSERT(first_domain_use_it != id_to_use_map.end());

      auto merge = dynamic_cast<Merge*>(first_domain_use_it->second);

      // See if this is contiguous merge
      if (merge && merge->inner() == second_domain) {
        update = true;
        auto merged_id = merge->out();
        // Insert the merged id and contiguity
        domain_list.insert(domain_it, merged_id);
        contiguity_id_set.erase(first_domain);
        contiguity_id_set.erase(second_domain);
        contiguity_id_set.insert(merged_id);

        // Erase the merged id.
        domain_list.erase(domain_it, std::next(second_domain_it));

        // Start over for simplicity, actually
        //  just decrement the domain it by one would be enough.
        break;
      }
      // Didn't find a merged domain candidate, just
      //  increment
      domain_it++;
    }
  }

  std::vector<IterDomain*> result_domain{
      domain_list.begin(), domain_list.end()};
  return std::make_pair(result_domain, contiguity_id_set);
}

//! Utility function that returns a map from each iterdomain from a tensordomain
//!  to its consumer, assuming that on this tensor domain each iterdomain has
//!  a unique iterdomain expression that consumes it.
std::unordered_map<IterDomain*, Expr*> getIdToUseMap(TensorDomain* td) {
  std::vector<Val*> all_id_vals{td->domain().begin(), td->domain().end()};
  auto all_inputs = InputsOf::outputs(td->fusion(), all_id_vals);
  std::unordered_map<IterDomain*, Expr*> id_to_use_map;

  auto all_exprs = DependencyCheck::getAllExprsBetween(
      {all_inputs.begin(), all_inputs.end()}, all_id_vals);

  for (auto expr : all_exprs) {
    for (auto id : ir_utils::filterByType<IterDomain>(expr->inputs())) {
      id_to_use_map[id] = expr;
    }
  }

  return id_to_use_map;
}

//! Given a pair of data_tv and reference_tv, see [Note on data tensor and
//! reference tensor]:
//!  Returns the set of iterdomains resulting from all the contiguous merges on
//!  the reference_tv's tensor domain.
//! The reason for needing both data_tv and reference_tv is that contiguity is
//! defined on data_tv
//!  while the actual index math is defined on reference tv.
//! TODO: unify with contiguity pass
std::unordered_set<IterDomain*> getInitialContiguousRootIdsOnReferenceTv(
    const TensorView* data_tv,
    const TensorView* reference_tv) {
  std::pair<std::vector<IterDomain*>, std::unordered_set<IterDomain*>>
      contig_merge_front;

  // Compute the merge frontier for contiguity info
  if (data_tv == reference_tv) {
    contig_merge_front = getContigMergeFrontier(
        reference_tv->getRootDomain(),
        reference_tv->domain()->contiguity(),
        getIdToUseMap(reference_tv->domain()));
  } else {
    std::unordered_set<IterDomain*> data_contig_domains;
    for (auto root_idx : c10::irange(data_tv->getRootDomain().size())) {
      auto root_id = data_tv->getRootDomain().at(root_idx);
      if (root_id->isBroadcast()) {
        continue;
      }
      if (data_tv->domain()->contiguity().at(root_idx)) {
        data_contig_domains.insert(root_id);
      }
    }

    auto c2p_root_map =
        PairwiseRootDomainMap(data_tv, reference_tv)
            .mapConsumerToProducer(reference_tv->domain(), data_tv->domain());

    auto root_size = reference_tv->getRootDomain().size();
    std::vector<c10::optional<bool>> contiguity_vec =
        TensorDomain::getContiguityFilledWith(
            reference_tv->getRootDomain(), false);

    for (auto root_idx = root_size - 1; root_idx >= 0; root_idx--) {
      auto root_id = reference_tv->getRootDomain()[root_idx];
      auto data_id_it = c2p_root_map.find(root_id);

      if (data_id_it == c2p_root_map.end()) {
        // Skip the rest if id did not map across to the producer side.
        break;
      }

      if (root_id->isBroadcast()) {
        continue;
      }

      if (data_contig_domains.count(data_id_it->second)) {
        contiguity_vec.at(root_idx) = true;
      } else {
        // Skip the rest if we find any domain
        //  that isn't contiguous from the inner most.
        break;
      }
    }

    contig_merge_front = getContigMergeFrontier(
        reference_tv->getRootDomain(),
        contiguity_vec,
        getIdToUseMap(reference_tv->domain()));
  }

  // Only record the contiguous ids from the innermost
  //  for the current optimization.
  std::unordered_set<IterDomain*> contig_ids;
  for (int root_idx = contig_merge_front.first.size() - 1; root_idx >= 0;
       root_idx--) {
    auto root_id = contig_merge_front.first[root_idx];
    if (root_id->isBroadcast()) {
      continue;
    }
    if (contig_merge_front.second.count(root_id)) {
      contig_ids.insert(root_id);
    } else {
      break;
    }
  }

  return contig_ids;
}

//! Returns all the nodes merged with a sequence of merge ops.
//!  ordered by their inner/outer position at the ops.
//! Example:
//!  Id0   Id1
//!   |    /   Id5  Id6
//!   merge    |     /
//!     |        merge
//!    Id2        |
//!     |        Id3
//!      |       /
//!        merge
//!          |
//!       Id4 (merged_id)
//! This function returns {Id0, Id1, Id5, Id6}
std::vector<IterDomain*> getFlattenedMergedIds(IterDomain* merged_id) {
  std::vector<IterDomain*> result;

  if (merged_id->definition() != nullptr) {
    if (auto merge = dynamic_cast<Merge*>(merged_id->definition())) {
      auto left_leaves = getFlattenedMergedIds(merge->outer());
      auto right_leaves = getFlattenedMergedIds(merge->inner());

      result.insert(result.end(), left_leaves.begin(), left_leaves.end());
      result.insert(result.end(), right_leaves.begin(), right_leaves.end());
      return result;
    }
  }

  // Not a merged id anymore so return as a leaf
  return {merged_id};
}

//! Get all the nodes merged with a sequence of merge ops.
//!  ordered by their inner/outer position at the ops.
//! Example: (split ops omitted)
//!               Id0 (split_id)
//!              /  |
//!             Id1  Id2
//!             / |
//!           Id3 Id4
//! This function returns {Id3, Id4, Id2}
std::vector<IterDomain*> getFlattenedSplitIds(
    IterDomain* split_id,
    std::unordered_set<IterDomain*> stop_set = {}) {
  std::vector<IterDomain*> result;

  auto maybe_single_use =
      ir_utils::getMaybeSingleUse(split_id, ir_utils::isIterDomainOp);

  // Assuming this means the iter domain is a leaf domain,
  //  the invariant that each iter domain should be consumed by only
  //  one iter domain op should be validated in another pass.
  if (maybe_single_use.has_value()) {
    if (auto split = dynamic_cast<Split*>(maybe_single_use.value())) {
      auto process_node = [&result, &stop_set](IterDomain* id) {
        if (stop_set.count(id)) {
          result.push_back(id);
        } else {
          auto leaves = getFlattenedSplitIds(id, stop_set);
          result.insert(result.end(), leaves.begin(), leaves.end());
        }
      };

      process_node(split->outer());
      process_node(split->inner());
      return result;
    }
  }

  // base case where the current node is even a leaf or
  //  not a split consumer:
  result.push_back(split_id);
  return result;
}

//! Returns the stride of the leading dimension given the vector of leaf
//!  iterdomains following their order defined by their outer/inner position
//!  as input/output of merge/split ops.
c10::optional<int64_t> getMaybeLeadingStride(
    std::vector<IterDomain*> leaf_ids,
    IterDomain* leading_id = nullptr) {
  if (leaf_ids.empty()) {
    return c10::nullopt;
  }
  if (leading_id == nullptr) {
    leading_id = leaf_ids.at(0);
  }

  bool leading_id_found = false;
  // Utility to compute the effective stride of id and prev_id:
  c10::optional<int64_t> result = c10::nullopt;
  for (auto leaf_it = leaf_ids.begin(); leaf_it != leaf_ids.end(); leaf_it++) {
    auto leaf_id = *leaf_it;
    if (leading_id_found) {
      if (!leaf_id->extent()->isConstInt()) {
        // Non-constant dimensions cannot be supported.
        return c10::nullopt;
      }
      result.value() *= leaf_id->extent()->evaluateInt();
    } else {
      if (leaf_id == leading_id) {
        leading_id_found = true;
        result = 1;
      }
    }
  }
  return result;
}

// Returns true if the given leaf (not checked) iterdomain,
//  from the given tensorview's tensor domain is separable, as defined
//  in [Note: Separability check]
//
// The analysis propagates backward, from the given leaf to the
//  contig merged frontier. There are two specific patterns that
//  this pass looks at, which are the ones that allows the index math
//  involving the given iterdomain to be separated out and more importantly
//  added to the rest, as memory index inlining is addition only so no need
//  to look at other ways of hoisting and combining here:
//
// pattern 1: (split-only propagation)
//       Id0 (root or contig merged id)
//        |
//      split
//      /   |
//    Id1   Id2(32)
//            |
//           split
//           /   |
//         Id3   Id4
// In the above example {Id1, Id3, Id4} are separable.
//
// pattern 2: (merge-then-split propagation):
//
//  Id0(root)  Id1(32)
//   |         /
//      merge
//       |
//      Id2    Id3(16)
//       |      /
//        merge
//          |
//         Id4
//          |
//         split
//         /  |
//        Id5  Id6(32)
//         |
//       split
//       /   |
//     Id7   Id8(16)
// In the case above, the leaf ids are {Id7, Id8, Id6},
//   and Id7 is the one that's separable while the others are not.
bool isSeparable(
    const TensorView* tv,
    IterDomain* id,
    const std::unordered_set<IterDomain*>& contig_merged_ids,
    bool ignore_swizzle = false,
    bool require_divisible = false) {
  auto id_def = id->definition();

  // Keep track of the id from the last iteration in the
  //  traversal below.
  IterDomain* prev_id = nullptr;

  // Traverse backward towards the root domains.
  while (id_def != nullptr) {
    if (auto split = dynamic_cast<Split*>(id_def)) {
      // When we hit a split, check for pattern 1

      // Keep track of the current id to support pattern 2
      //  checking.
      prev_id = id;

      // move the id pointer up the graph
      id = split->in();
      id_def = id->definition();

      // Already hitting contiguous root so stop
      //  traversing up.
      if (contig_merged_ids.count(id)) {
        return true;
      }

      // Check for non-divisible split:
      if (!id->extent()->isConstInt()) {
        return false;
      }
      auto id_extent = id->extent()->evaluateInt();
      auto factor_extent = split->factor()->getInt();

      if (!factor_extent.has_value()) {
        // Non constant sized tiling cannot be lifted.
        return false;
      }

      if (id_extent % factor_extent.value() != 0) {
        // Non-divisible split cannot be separated.
        return false;
      }
    } else if (dynamic_cast<Merge*>(id_def)) {
      // When we hit a merge, check for pattern 2

      if (prev_id == nullptr) {
        // A merge on the leaf domain not yet supported.
        return false;
      }

      if (dynamic_cast<Split*>(prev_id->definition())) {
        auto split_leaves = getFlattenedSplitIds(id, {prev_id});
        auto merged_leaves = getFlattenedMergedIds(id);

        if (prev_id != split_leaves.at(0)) {
          // Only support the case when both ids are leading dims.
          return false;
        }

        // Check the actual static dimensions, which would be the
        //  only way for now to prove that we'd not need to wrap
        //  around when incrementing this id:
        auto maybe_id_stride = getMaybeLeadingStride(merged_leaves);
        auto maybe_prev_id_stride = getMaybeLeadingStride(split_leaves);

        if (maybe_id_stride.has_value() && maybe_prev_id_stride.has_value()) {
          if (maybe_id_stride.value() <= maybe_prev_id_stride.value()) {
            if (!require_divisible ||
                (maybe_prev_id_stride.value() % maybe_id_stride.value() == 0)) {
              id = merged_leaves.at(0);
              id_def = id->definition();
              continue;
            }
          }
        }
      }

      return false;
    } else if (auto swizzle_2d = dynamic_cast<Swizzle2D*>(id_def)) {
      if (ignore_swizzle) {
        // Forward through swizzle if it is to be ignored.
        id = id == swizzle_2d->outX() ? swizzle_2d->inX() : swizzle_2d->inY();
        id_def = id->definition();
      } else {
        // When not ignored, swizzles are always assumed to be
        //  not separable, i.e. they will take extra space in the pre-computed
        //  tensor address space.
        return false;
      }
    }
  }

  // This covers the base case where we start
  //  with a contigous merged output of root.
  return contig_merged_ids.count(id);
}

//! Traverse forward on the iterdomain's use chain and returns
//!  true if a swizzle iterdomain op is found.
bool isSwizzleProducer(IterDomain* id) {
  auto single_use = ir_utils::getMaybeSingleUse(id, ir_utils::isIterDomainOp);
  if (!single_use.has_value()) {
    // Base case check.
    return false;
  }

  if (single_use.value()->isA<Swizzle2D>()) {
    return true;
  }

  for (auto id :
       ir_utils::filterByType<IterDomain>(single_use.value()->outputs())) {
    // The current id is swizzled if any subsequent consumer iterdomains
    //  is a producer of swizzle op.
    if (isSwizzleProducer(id)) {
      return true;
    }
  }

  return false;
}

//! Returns true if the given id is separable after taking the
//!  possible swizzle ops on producer's tensor domain. See also
//!  `IndexSwizzle` pass in index_compute.cpp.
//! When indexing a swizzled producer, there is an additional forward
//!  traversal that computes the swizzled address. This function is
//!  used to ensure that the given id would not be involved in any
//!  swizzle operation in this traversal and thus remains separable
//!  after the producer swizzle pass. It will return false conservatively
//!  whenever the forward path hits any swizzle op.
//!
//! `id` is assumed to be on consumer_tv's tensordomain.
//!
//!  The current iteration only looks for one specific pattern, and returns
//!  false
//!   for any domain that does not match:
//!
//!        consumer Id0(may not be root)  <-- exact mapped -->   producer
//!        Id0(may not be root)
//!               |                                                 |
//!        ...  divisible splits ...                    ... divisible  splits
//!           /           |                                      /          |
//!          id          {consumer inner ids}    producer_outer_id  {inner ids}
//! With the additional check that
//!  1. Product of consumer inner ids' extents are integer multiple of product
//!  of producer ids' extents.
//!  2. Producer_outer_id is not a producer of any swizzle op down its use
//!  chain.
//! This check intuitively ensures tha increment on `id` "strides over" a
//! integer multiple of swizzled
//!  tiles on the producer data layout.
bool isSeparableSmemSwizzledProducerIndex(
    const TensorView* producer_tv,
    const TensorView* consumer_tv,
    IterDomain* id,
    const std::unordered_set<IterDomain*>& contig_merged_ids) {
  // Collect exact iterdomains in producer tv on the right of the producer
  //  ca_axis, which would be the nodes that'd be materialzied in shared mem
  //  tile.
  std::unordered_map<IterDomain*, IterDomain*> exact_to_producer_id_map;
  auto all_producer_id_vals = DependencyCheck::getAllValsBetween(
      {producer_tv->getMaybeRFactorDomain().begin(),
       producer_tv->getMaybeRFactorDomain().end()},
      {producer_tv->domain()->domain().begin() +
           producer_tv->getComputeAtPosition(),
       producer_tv->domain()->domain().end()});

  for (auto producer_id :
       ir_utils::filterByType<IterDomain>(all_producer_id_vals)) {
    exact_to_producer_id_map.emplace(std::make_pair(
        GpuLower::current()->caMap()->getConcreteMappedID(
            producer_id, IdMappingMode::EXACT),
        producer_id));
  }

  if (id->definition() == nullptr) {
    // Cannot derive much information from a root id.
    return false;
  }

  if (auto split = dynamic_cast<Split*>(id->definition())) {
    auto exact_producer_it = exact_to_producer_id_map.find(
        GpuLower::current()->caMap()->getConcreteMappedID(
            split->in(), IdMappingMode::EXACT));

    // Find a tile split node that this needs to assume.
    while (exact_producer_it == exact_to_producer_id_map.end()) {
      auto in_def = split->in()->definition();
      if (in_def == nullptr) {
        return false;
      }
      split = dynamic_cast<Split*>(in_def);
      if (split == nullptr) {
        return false;
      }
      exact_producer_it = exact_to_producer_id_map.find(
          GpuLower::current()->caMap()->getConcreteMappedID(
              split->in(), IdMappingMode::EXACT));
    }

    if (!exact_producer_it->second->extent()->isConstInt()) {
      // TODO: more levels of splits and merges should not be too hard
      //  to support but would probably want to do after the next round
      //  of indexing cleanup.
      return false;
    }

    auto producer_in_id = exact_producer_it->second;
    auto split_leaves = getFlattenedSplitIds(split->in(), {id});
    auto producer_split_leaves = getFlattenedSplitIds(producer_in_id);

    // Check that the leading dimension of producer split leaves are not
    // directly swizzled:
    if (isSwizzleProducer(producer_split_leaves.at(0))) {
      // No support for lifting swizzled id.
      return false;
    }

    auto maybe_producer_stride = getMaybeLeadingStride(producer_split_leaves);
    auto maybe_consumer_stride = getMaybeLeadingStride(split_leaves, id);

    if (maybe_consumer_stride.has_value() &&
        maybe_producer_stride.has_value()) {
      if (maybe_consumer_stride.value() >= maybe_producer_stride.value() &&
          // Divisibility is required to ensure the increment is a multiple
          //  of the swizzle period.
          maybe_consumer_stride.value() % maybe_producer_stride.value() == 0) {
        // Only supports the case where the leading dimension maps to
        //  the leading dimension of the producer side.
        return true;
      }
    }
  }

  // TODO: more flexible checking would need to be built out.
  return false;
}

} // namespace

void AddressComputeInfo::makeAddressRecord(
    TensorView* data_tv,
    TensorView* reference_tv,
    bool is_predicate_record) {
  if (is_predicate_record) {
    TORCH_INTERNAL_ASSERT(
        data_tv == reference_tv, "predicate record only takes reference tv");
  }
  // Signify if this is caching shared mem access,
  //  assume it'd be global access if other wise.
  bool is_shared_mem_access =
      !is_predicate_record && data_tv->getMemoryType() == MemoryType::Shared;

  // Signify if this is a producer indexing.
  bool is_data_read = !is_predicate_record && data_tv != reference_tv;

  const auto& ca_map = GpuLower::current()->caMap();

  // Get allocation ids for the lifted index.
  //  These are the ids that we will need to allocate space for in the
  // address tensor. Only serial, unrolled loops that are in-separable
  // are allocated and materialized in the base index loop.
  std::deque<IterDomain*> alloc_ids;

  std::unordered_set<IterDomain*> contig_merged_ids;

  if (is_predicate_record) {
    // TODO:
    //   In order to generically turn on predicate lifting
    // the contig id logic in predicate indexing should be
    // used here. Yet currently that part of the logic requires
    // the actual loop nest so cleaning up that part would be
    // a huge todo step before this pass can be made generic.
    contig_merged_ids = std::unordered_set<IterDomain*>(
        reference_tv->getMaybeRFactorDomain().begin(),
        reference_tv->getMaybeRFactorDomain().end());
  } else {
    contig_merged_ids =
        getInitialContiguousRootIdsOnReferenceTv(data_tv, reference_tv);
  }

  // Mark the boundary iterdomain at the compute at
  //  position, where iterdomains on the left would
  //  be in shared loop.
  auto ref_ca_id = data_tv->axis(data_tv->getComputeAtPosition() - 1);

  if (data_tv->isFusionInput()) {
    // Reading input tensor looks at the consumer
    //  scheduling so use the CA of comsumer.
    ref_ca_id = reference_tv->axis(reference_tv->getComputeAtPosition() - 1);
  } else if (data_tv->isFusionOutput()) {
    ref_ca_id = data_tv->axis(data_tv->getMaxProducerPosition() - 1);
  } else if (is_data_read) {
    bool found_ref_ca_id = false;
    for (auto id : reference_tv->domain()->domain()) {
      if (GpuLower::current()->caMap()->areMapped(
              id, ref_ca_id, IdMappingMode::LOOP)) {
        ref_ca_id = id;
        found_ref_ca_id = true;
        break;
      }
    }

    TORCH_INTERNAL_ASSERT(found_ref_ca_id, "unsupported case in index lifting");
  }

  // Serial domain which the cached indexing will
  //  be lifted out of.
  // TODO: some extra work to support
  //  lifting all the way to global scope.
  // TODO: some extra work to lift index beyond
  //  the compute at position.
  IterDomain* serial_id = nullptr;

  // compute id's to cache index on:
  auto ref_id_it = reference_tv->domain()->domain().rbegin();
  while (ref_id_it != reference_tv->domain()->domain().rend()) {
    auto ref_id = *ref_id_it;

    if (ref_id == ref_ca_id) {
      // TODO:
      // Temporarily no support for lifting
      //  beyond the computeAt axis. The axes
      //  mapping tends to be more complex and
      //  would need extra information to resolve.
      break;
    }

    // Do not allocate parallel dims as they do not
    //  contribute to register lifetime.
    if (ref_id->isParallelized()) {
      ref_id_it++;
      continue;
    }

    // Current not attempting to lift outside of
    //  non-constant sized serial loops, due to
    //  limited profitability. Might want to
    //  revisit when we see useful cases.
    if (!ref_id->extent()->isConstScalar()) {
      serial_id = ref_id;
      break;
    }

    // Will require divisibility check to ensure separability across
    //  complete swizzled tiles.
    bool require_divisible = !is_data_read && data_tv->hasSwizzleOp();

    // TODO: Global memory lifting is not enabled here as their
    //  lifetime management is much more complex and addressed in follow ups.
    //  See internal doc, and will need to evaluate more internally in terms
    //  of how explicit should the heuristic involving global memory should be
    //  made.
    if (is_shared_mem_access &&
        isSeparable(
            reference_tv,
            ref_id,
            contig_merged_ids,
            is_predicate_record,
            require_divisible)) {
      if (
          // Checking reference tv is enough for non-swizzled producer.
          (
              // For data write, checking separability is enough.
              !is_data_read ||
              // For data read would need additionally check swizzles.
              (!data_tv->hasSwizzleOp() ||
               // Check supported lifting in the case of swizzled producer.
               isSeparableSmemSwizzledProducerIndex(
                   data_tv, reference_tv, ref_id, contig_merged_ids)))) {
        // Skipping here means not allocating for this id as it is separable
        //  and thus inlined.
        ref_id_it++;
        continue;
      }
    }

    // We are visiting the iterdomains from reference
    //  tv in reverse order but we want the outputs
    //  to be in original order.
    alloc_ids.push_front(ref_id);
    ref_id_it++;
  }

  // Continue searching back to locate the serial id to lift
  //  the indices over.
  // Current usage always assumes that there is always such serial
  //  iterdomain, i.e. no support yet for lifting kernels with unrolled
  //  loops only.
  while (ref_id_it != reference_tv->domain()->domain().rend() &&
         serial_id == nullptr) {
    auto ref_id = *ref_id_it;
    ref_id_it++;
    if (!ref_id->isParallelized()) {
      serial_id = ref_id;
    }
  }

  TORCH_INTERNAL_ASSERT(
      serial_id != nullptr,
      "Cannot target serial loop to lift indexing out of");

  std::vector<IterDomain*> alloc_ids_vec{alloc_ids.begin(), alloc_ids.end()};

  TORCH_INTERNAL_ASSERT(
      isSeparable(reference_tv, serial_id, contig_merged_ids),
      "The serial id is required to be separable for the index lifting to work.");

  // Assuming we are only having two scenarios,
  //  either accessing a consumer in the consumer's loop,
  //  or accessing the producer in producer's loop.
  auto access_direction = is_predicate_record
      ? AddressRecord::ReadWrite::PREDICATE
      : reference_tv == data_tv ? AddressRecord::ReadWrite::WRITE
                                : AddressRecord::ReadWrite::READ;

  bool is_cp_async_write =
      access_direction == AddressRecord::ReadWrite::WRITE &&
      ir_utils::isCpAsyncOp(data_tv->definition());

  // Create address record:
  auto address_tv = makeAddressTv(
      alloc_ids_vec,
      !is_shared_mem_access,
      is_predicate_record,
      is_cp_async_write);

  TORCH_INTERNAL_ASSERT(
      serial_id != nullptr, "no support yet for global scope hoisting");

  auto concrete_serial_id =
      ca_map->getConcreteMappedID(serial_id, IdMappingMode::LOOP);

  // Create and insert lifting record
  auto new_record_ptr = std::make_unique<AddressRecord>(
      data_tv,
      address_tv,
      alloc_ids_vec,
      reference_tv,
      access_direction,
      concrete_serial_id);

  address_tv_to_address_record_[address_tv] = new_record_ptr.get();

  // Add address lift record
  if (is_predicate_record) {
    predicate_lift_record_.insert(
        std::make_pair(new_record_ptr->key(), std::move(new_record_ptr)));
  } else {
    index_lift_record_.insert(
        std::make_pair(new_record_ptr->key(), std::move(new_record_ptr)));
  }
}

c10::optional<AddressRecord*> AddressComputeInfo::getMaybeLiftedAddress(
    const TensorView* data_tv,
    const TensorView* reference_tv) {
  // Use data tv as the reference if
  //  reference is missing.
  if (reference_tv == nullptr) {
    reference_tv = data_tv;
  }

  auto address_record_it =
      index_lift_record_.find(AddressRecordKey(reference_tv, data_tv));

  if (address_record_it == index_lift_record_.end()) {
    return c10::nullopt;
  }

  return address_record_it->second.get();
}

c10::optional<AddressRecord*> AddressComputeInfo::getMaybeLiftedPredicateIndex(
    const TensorView* reference_tv) {
  auto address_record_it =
      predicate_lift_record_.find(AddressRecordKey(reference_tv, reference_tv));

  if (address_record_it == predicate_lift_record_.end()) {
    return c10::nullopt;
  }

  return address_record_it->second.get();
}

TensorView* AddressComputeInfo::makeAddressTv(
    std::vector<IterDomain*> address_domains,
    bool is_global_address,
    bool is_predicate_index,
    bool is_cpasync_write) {
  DataType dtype = is_predicate_index ? DataType::Index : DataType::Pointer;

  // Note: [Lifting smem address decoding for cp.async]
  // A trick that saves register usage.
  // Before:
  //  char* smem_ptr;
  //  for i in ... // main loop
  //    cp.async [smem_ptr + 123], ...
  // After:
  //  unsigned smem_address = toSmem(smem_ptr);
  //  for i in ... // main loop
  //    cp.async [smem_addres+123], ...
  if (is_cpasync_write) {
    dtype = DataType::SMemAddress;
  }
  return IrBuilder::create<TensorView>(
      IrBuilder::create<TensorDomain>(
          address_domains,
          TensorDomain::getContiguityFilledWith(address_domains, true)),
      dtype);
}

c10::optional<AddressRecord*> AddressComputeInfo::getMaybeRecordForAddressTv(
    const TensorView* tv) {
  auto record_it = address_tv_to_address_record_.find(tv);
  if (record_it == address_tv_to_address_record_.end()) {
    return c10::nullopt;
  }
  return record_it->second;
}

c10::optional<kir::ForLoop*> AddressRecord::getMaybeSerialLoop(
    std::vector<kir::ForLoop*> loops) {
  auto concrete_loop_id = getConcreteSerialLoopId();

  auto serial_loop_it = std::find_if(
      loops.begin(), loops.end(), [concrete_loop_id](kir::ForLoop* for_loop) {
        return GpuLower::current()->caMap()->areMapped(
            for_loop->iter_domain(), concrete_loop_id, IdMappingMode::LOOP);
      });

  if (serial_loop_it != loops.end()) {
    return *serial_loop_it;
  }

  return c10::nullopt;
}

namespace {

struct AddressComputeInsertionInfo {
  std::vector<kir::ForLoop*> loop_nest;
  AddressRecord* address_compute_record;
};

class MemoryAddressComputeInserter : public kir::ExprMutator {
 public:
  static std::vector<Expr*> insert(const std::vector<Expr*>& exprs) {
    MemoryAddressComputeInserter inserter(exprs);
    return inserter.exprs_;
  }

 private:
  explicit MemoryAddressComputeInserter(const std::vector<Expr*>& exprs) {
    traverseAndInsert(exprs);
  }

  void handle(kir::ForLoop* loop) final {
    // Descend into the inner loop nest first
    //  to register lifting inner loops.
    kir::ExprMutator::handle(loop);
    auto concrete_loop_id = GpuLower::current()->caMap()->getConcreteMappedID(
        loop->iter_domain(), IdMappingMode::LOOP);

    // Currently assumes that this pass is run
    //  almost right after the loopnest generation,
    //  i.e. not any loop fission should have happened
    //  before.
    auto insertion_info_vec_it =
        pending_address_compute_insertions_.find(concrete_loop_id);
    if (insertion_info_vec_it != pending_address_compute_insertions_.end()) {
      insertAddressComputes(loop, *(insertion_info_vec_it->second));
      pending_address_compute_insertions_.erase(concrete_loop_id);
    }
  }

  void insertAddressComputes(
      kir::ForLoop* loop,
      const std::vector<AddressComputeInsertionInfo>& insertion_infos) {
    for (auto& insertion_info : insertion_infos) {
      std::vector<Val*> alloc_shape;
      std::transform(
          insertion_info.address_compute_record->allocationIterDomains()
              .begin(),
          insertion_info.address_compute_record->allocationIterDomains().end(),
          std::back_inserter(alloc_shape),
          [](IterDomain* id) { return id->extent(); });

      // allocate address tensor:
      auto alloc = IrBuilder::create<kir::Allocate>(
          insertion_info.address_compute_record->addressTensor(),
          MemoryType::Local,
          alloc_shape);
      registerInsertBefore(loop, alloc);

      // clone loop nest:
      auto outermost_innermost =
          scope_utils::makeLoopNest(insertion_info.loop_nest);

      // make the address compute op:
      auto compute_type = insertion_info.address_compute_record->isPredicate()
          ? kir::AddressCompute::AddressComputeOpType::PREDICATE_INDEX
          : kir::AddressCompute::AddressComputeOpType::BASE_ADDRESS;
      auto address_initialize_op = IrBuilder::create<kir::AddressCompute>(
          compute_type,
          insertion_info.address_compute_record->addressTensor(),
          insertion_info.address_compute_record->dataTensor());

      // put address compute in forloop
      outermost_innermost.second->body().push_back(address_initialize_op);

      // put the new loopnest before the hoisted loop
      registerInsertBefore(loop, outermost_innermost.first);

      auto data_tensor = insertion_info.address_compute_record->dataTensor();

      // Insert double buffer increment:
      // Note: [Inplace Double Buffer Update]:
      //
      // The trick used in [Uniform Double Buffer Offset] should be the default
      //  method of handling double buffer switching index when trying to save
      //  general purpose registers. But there are 2 exceptions:
      //   1. On sm70 or below, there are no unifrom regs to use.
      //   2. With cp.async, the consumer shared memory buffer currently
      // does not provide access to the uniform reg operand so we could not use
      // it. (will be actively discussed internally)
      //
      // To still avoid using too many registers on double buffered access,
      //  another code gen trick is used here, to enable near term progress:
      // The code before transformation:
      //   for i in ... // double buffer loop
      //     ... = ld.shared [... + (i%5) * double_buffer_size]
      // The code after transformation:
      //   R0 = ...
      //   for i in ... // double buffer loop
      //     ... = ld.shared [R0]
      //     doubleBufferUpdate(R0);
      // This way essentially the double buffer offset is calculated inplace
      //  into R0 in each double buffer loop iteration. Note that comparing with
      //  [Uniform Double Buffer Offset] this method uses more instructions as
      //  all of the pointers will need to be updated, while using uniform regs
      //  will only need to update the uniform switch index.

      // FIXME: should move this logic into lower_double_buffer.cpp.
      //  will need to formulate into a separate pass as it needs to
      //  clone the loop nest.
      if ((data_tensor->isDoubleBuffered() ||
           data_tensor->isCircularBuffered()) &&
          insertion_info.address_compute_record->isWrite() &&
          // Only have support doubleBufferUpdate for
          //  direct smem access for now.
          // FIXME:
          //   Would need to extend to use this on Volta.
          lower_utils::useDirectSmemAddress(data_tensor)) {
        // Insert double buffer index update if it is a double buffered write:
        // The insertion info loop nest starts with the serial loop,
        //  in the double buffer update we need to insert into the original
        //  serial loop itself, so remove the outermost level.
        auto db_loop_nest = std::vector<kir::ForLoop*>(
            std::next(insertion_info.loop_nest.begin()),
            insertion_info.loop_nest.end());

        // Clone the loop nest containing the double buffered write
        //  expression for the consumer index update.
        auto db_outer_inner = scope_utils::makeLoopNest(db_loop_nest);

        // Calculate the double buffer size.
        auto& db_info = GpuLower::current()->doubleBufferInfo();

        auto db_size_in_byte = SimplifyingIrBuilder::mulExpr(
            db_info.getOriginalAllocSize(data_tensor),
            SimplifyingIrBuilder::create<Int>(
                dataTypeSize(data_tensor->dtype())));

        // Create the double buffer update expression and insert
        //  them at the end of the double buffer loop.
        auto update_expr = SimplifyingIrBuilder::create<kir::AddressCompute>(
            insertion_info.address_compute_record->addressTensor(),
            db_size_in_byte,
            data_tensor->circularBufferDepth(),
            0,
            data_tensor);

        db_outer_inner.second->body().push_back(update_expr);
        loop->body().push_back(db_outer_inner.first);
      }

      // Insert gmem increment:
      // Note [Gmem address increment]:
      //  This is a trick that helps lifting some instructions out of main
      // loop.
      // The code before this transformation:
      //   R0 = ...
      //   for i in ... // The serial loop on index lifting record
      //     x = ld.global [i*123 + R0]
      // The code after transformation:
      //   R0 = ...
      //   for i in ... // The serial loop on index lifting record
      //     x = ld.global [R0]
      //     R0+=123;
      // Note that [Separability Analysis] will be checked on the serial
      //  loop when creating these address records so doing this transformation
      //  on the serial loop index variable is safe.
      if (data_tensor->getMemoryType() == MemoryType::Global &&
          insertion_info.address_compute_record->isRead()) {
        // Create the loopnest to contain the increment expression.
        auto increment_loop_vector =
            createIncrementLoop(insertion_info.loop_nest);
        auto increment_loop_outer_inner =
            scope_utils::makeLoopNest(increment_loop_vector);

        // Create the increment expression.
        auto inc_expr = SimplifyingIrBuilder::create<kir::AddressCompute>(
            insertion_info.address_compute_record->addressTensor(),
            insertion_info.address_compute_record->dataTensor());

        increment_loop_outer_inner.second->body().push_back(inc_expr);
        loop->body().push_back(increment_loop_outer_inner.first);
      }
    }
  }

  void handle(kir::IfThenElse* ite) final {
    TORCH_INTERNAL_ASSERT(
        false, "this pass cannot yet run after ite insertion");
  }

  //! Create an replica of the original loop except that
  //!  the extent and index are zeroed.
  kir::ForLoop* createZeroedLoop(kir::ForLoop* original_loop) {
    auto start = original_loop->iter_domain()->isThread()
        ? original_loop->start()
        : GpuLower::current()->kernel()->zeroVal();
    auto stop = original_loop->iter_domain()->isThread()
        ? original_loop->iter_domain()->extent()
        : GpuLower::current()->kernel()->oneVal();
    return IrBuilder::create<kir::ForLoop>(
        original_loop->iter_domain(),
        // index
        GpuLower::current()->kernel()->zeroVal(),
        // Start
        start,
        // Stop
        stop,
        // Step
        GpuLower::current()->kernel()->oneVal(),
        false,
        nullptr,
        original_loop->isUnrollRequired(),
        original_loop->loopTransformInfo().baseIndexLoop());
  }

  // Utility to create the loop nest for gmem increment,
  //  see [Gmem address increment].
  std::vector<kir::ForLoop*> createIncrementLoop(
      std::vector<kir::ForLoop*> address_compute_loop_vector) {
    std::vector<kir::ForLoop*> loop_nest_to_clone(
        std::next(address_compute_loop_vector.begin()),
        address_compute_loop_vector.end());

    std::vector<kir::ForLoop*> cloned_loop_nest;
    for (auto fl : loop_nest_to_clone) {
      cloned_loop_nest.push_back(IrBuilder::create<kir::ForLoop>(
          fl->iter_domain(),
          fl->index(),
          fl->start(),
          fl->stop(),
          fl->step(),
          fl->vectorize(),
          fl->vectorize_shift(),
          fl->isUnrollRequired(),
          fl->loopTransformInfo().incrementLoop()));
    }

    return cloned_loop_nest;
  }

  std::vector<kir::ForLoop*> createAddressComputeLoop(
      AddressRecord* address_record) {
    // Find the loop in the current loop nest that maps the concrete serial loop
    //  on record
    auto concrete_loop_id = address_record->getConcreteSerialLoopId();
    auto serial_loop_it = std::find_if(
        for_loops_.begin(),
        for_loops_.end(),
        [concrete_loop_id](kir::ForLoop* for_loop) {
          return GpuLower::current()->caMap()->areMapped(
              for_loop->iter_domain(), concrete_loop_id, IdMappingMode::LOOP);
        });
    TORCH_INTERNAL_ASSERT(
        serial_loop_it != for_loops_.end(),
        "invalid address record, serial loop not found");

    // make an empty outmost for loop:
    auto original_serial_loop = *serial_loop_it;
    serial_loop_it++;
    auto cloned_serial_loop = createZeroedLoop(original_serial_loop);

    std::vector<kir::ForLoop*> loop_vector;
    loop_vector.push_back(cloned_serial_loop);

    while (serial_loop_it != for_loops_.end()) {
      auto loop = *serial_loop_it;
      serial_loop_it++;
      if (std::any_of(
              address_record->allocationIterDomains().begin(),
              address_record->allocationIterDomains().end(),
              [loop](IterDomain* allocated_id) {
                return GpuLower::current()->caMap()->areMapped(
                    allocated_id, loop->iter_domain(), IdMappingMode::LOOP);
              })) {
        loop_vector.push_back(loop);
      } else {
        // Only materialize the loops that correspond to
        //  allocated dimensions of address tensor.
        loop_vector.push_back(createZeroedLoop(loop));
      }
    }

    return loop_vector;
  }

  AddressComputeInsertionInfo makeInsertionInfo(AddressRecord* address_record) {
    AddressComputeInsertionInfo address_info;
    address_info.loop_nest = createAddressComputeLoop(address_record);
    address_info.address_compute_record = address_record;
    return address_info;
  }

  void addAddressComputeInsertionInfo(
      IterDomain* serial_id,
      AddressComputeInsertionInfo insertion_info) {
    std::vector<AddressComputeInsertionInfo>* insertion_info_vec_ptr = nullptr;
    auto exisiting_info_it =
        pending_address_compute_insertions_.find(serial_id);
    if (exisiting_info_it == pending_address_compute_insertions_.end()) {
      // Make a new vector of address compute record if needed.
      auto new_address_compute_record =
          std::make_unique<std::vector<AddressComputeInsertionInfo>>();
      insertion_info_vec_ptr = new_address_compute_record.get();
      pending_address_compute_insertions_[serial_id] =
          std::move(new_address_compute_record);
    } else {
      insertion_info_vec_ptr = exisiting_info_it->second.get();
    }
    insertion_info_vec_ptr->push_back(insertion_info);
  }

  void handle(Expr* expr) final {
    if (ir_utils::isTvOp(expr) && !ir_utils::isTensorScalarFillOp(expr)) {
      auto& address_compute_info = GpuLower::current()->addressComputeInfo();

      // Register memory writes to insert
      for (auto consumer_tv :
           ir_utils::filterByType<TensorView>(expr->outputs())) {
        auto maybe_consumer_address_record =
            address_compute_info.getMaybeLiftedAddress(consumer_tv);

        if (maybe_consumer_address_record.has_value()) {
          auto insert_info =
              makeInsertionInfo(maybe_consumer_address_record.value());
          addAddressComputeInsertionInfo(
              maybe_consumer_address_record.value()->getConcreteSerialLoopId(),
              insert_info);
        }

        auto maybe_predicate_record =
            address_compute_info.getMaybeLiftedPredicateIndex(consumer_tv);

        if (maybe_predicate_record.has_value()) {
          auto insert_info = makeInsertionInfo(maybe_predicate_record.value());
          addAddressComputeInsertionInfo(
              maybe_predicate_record.value()->getConcreteSerialLoopId(),
              insert_info);
        }

        // Register memory reads to insert
        for (auto producer_tv :
             ir_utils::filterByType<TensorView>(expr->inputs())) {
          auto maybe_producer_address_record =
              address_compute_info.getMaybeLiftedAddress(
                  producer_tv, consumer_tv);
          if (maybe_producer_address_record.has_value()) {
            auto insert_info =
                makeInsertionInfo(maybe_producer_address_record.value());
            addAddressComputeInsertionInfo(
                maybe_producer_address_record.value()
                    ->getConcreteSerialLoopId(),
                insert_info);
          }
        }
      }
    } else {
      kir::ExprMutator::handle(expr);
    }
  }

 private:
  using AddressComputeInsertionInfoPtr =
      std::unique_ptr<std::vector<AddressComputeInsertionInfo>>;
  std::unordered_map<IterDomain*, AddressComputeInsertionInfoPtr>
      pending_address_compute_insertions_;
};

} // namespace

// TODO: add insert address compute pass
std::vector<Expr*> preComputeLiftedAddress(const std::vector<Expr*>& exprs) {
  return MemoryAddressComputeInserter::insert(exprs);
}

} // namespace nvfuser
