// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#include <device_lower/lower2device.h>
#include <id_model/contiguity.h>
#include <id_model/utils.h>

namespace nvfuser {

ContigIDGroups::ContigIDGroups(
    const std::vector<IterDomain*>& alloc_domains,
    std::vector<bool> contiguity,
    const ExprPath<ExprGroup>& path_from_alloc,
    const ValGraph& graph,
    bool is_predicate_pass)
    : graph_(graph),
      alloc_domains_(alloc_domains),
      alloc_contiguity_(std::move(contiguity)),
      is_predicate_pass_(is_predicate_pass),
      consistent_transform_info_(
          std::make_unique<const OrderedIdGroupInformation>(
              OrderedIdGroupInformation::get(
                  alloc_domains,
                  path_from_alloc,
                  graph))) {
  if (alloc_domains_.empty()) {
    return;
  }

  NVF_ERROR(
      alloc_domains_.size() == alloc_contiguity_.size(),
      "Arguments don't match ",
      alloc_domains_.size(),
      " != ",
      alloc_contiguity_.size());

  for (const auto index_domain_i : arange(alloc_domains_.size())) {
    IterDomain* index_domain = alloc_domains_.at(index_domain_i);
    NVF_ERROR(
        !index_domain->isBroadcast(),
        "Broadcast domain should not be an index domain: ",
        index_domain->toString());

    alloc_to_contig_ids_[index_domain] = graph_.toGroup(index_domain);

    auto alloc_contiguity = alloc_contiguity_.at(index_domain_i);

    if (alloc_contiguity &&
        index_domain->getIterType() != IterType::GatherScatter) {
      contig_ids_.emplace(graph_.toGroup(index_domain));
    }
  }

  for (const auto& [eg, direction] : path_from_alloc) {
    // Propagate resize and non-divisible dependencies
    const auto inputs = direction == Direction::Forward
        ? graph_.inputGroups(eg)
        : graph_.outputGroups(eg);
    const auto outputs = direction == Direction::Forward
        ? graph_.outputGroups(eg)
        : graph_.inputGroups(eg);
    if (std::any_of(inputs.begin(), inputs.end(), [&](const ValGroup& inp) {
          return resize_deps_.count(inp) > 0;
        })) {
      for (const auto& out : outputs) {
        resize_deps_.insert(out);
      }
    }

    if (std::any_of(inputs.begin(), inputs.end(), [&](const ValGroup& inp) {
          return non_divisible_deps_.count(inp) > 0;
        })) {
      for (const auto& out : outputs) {
        non_divisible_deps_.insert(out);
      }
    }

    dispatch(eg, direction);
  }
}

void ContigIDGroups::handle(Merge* merge, Direction direction) {
  // Only forward direction is supported for now
  if (direction != Direction::Forward) {
    // Backward propagation for Merge is not implemented.
    // Considering the backward case of a Merge (i.e., if merge->out() is given
    // as contiguous, what does it imply for merge->outer() and
    // merge->inner()?): While merge->outer() and merge->inner() would also be
    // contiguous in such a scenario, this information does not typically lead
    // to further simplification of indices beyond what using merge->out()
    // already provides if it was part of a contiguous chain. The primary goal
    // of contiguity analysis is to find larger, merged domains to simplify
    // indexing. Decomposing a known contiguous merged domain doesn't usually
    // serve this purpose.
    return;
  }

  const bool is_indexing_pass = !is_predicate_pass_;

  // If output is not consistently ordered or doesn't solely consume all
  // allocation domains in its dependencies, then it can't be a contiguously
  // indexable iterdomain. If it's a predicate pass, the ordering
  // doesn't matter since it does not index any actual memory.
  if (!(is_predicate_pass_ ||
        consistent_transform_info_->isConsistentlyOrdered(merge->out()))) {
    return;
  }

  if (!consistent_transform_info_->exclusivelyConsumesAllocs(merge->out())) {
    return;
  }

  // Check allocation domains for contiguity
  auto alloc_ids_it = consistent_transform_info_->findAllocIDs(merge->out());

  // Contiguity doesn't matter for predicates
  if (is_indexing_pass) {
    VectorOfUniqueEntries<IterDomain*> alloc_ids = alloc_ids_it->second;
    for (auto alloc_id_i : arange(alloc_domains_.size())) {
      auto alloc_id = alloc_domains_[alloc_id_i];
      if (alloc_ids.erase(alloc_id) == 0) {
        continue;
      }
      auto alloc_contiguity = alloc_contiguity_.at(alloc_id_i);
      // If we're indexing:
      // we could still potentially consider this ID linearly indexable, as we
      // could multiple the index by the last allocation's stride. See
      // ContigIndexingTest.NonContigInnermost for a concrete example.
      if (!alloc_contiguity && !alloc_ids.empty()) {
        return;
      }
    }
  }

  // If there's a non-divisible
  // split in the history of merge->out then the extents of the inputs
  // and also the outputs may be expanded due to ceilDiv. Predicate
  // indexing needs to avoid contiguous indexing. Non-predicate
  // indexing should have no such constraint.
  if (is_predicate_pass_ &&
      non_divisible_deps_.count(graph_.toGroup(merge->out()))) {
    return;
  }

  // Don't allow contig indexing after resize as we need traverse back
  // at least to direct outputs of resize ops
  if (resize_deps_.count(graph_.toGroup(merge->out()))) {
    return;
  }

  // Now we know merge->out is a contiguously indexable ID

  for (auto alloc_id : alloc_ids_it->second) {
    alloc_to_contig_ids_[alloc_id] = graph_.toGroup(merge->out());
  }

  contig_ids_.emplace(graph_.toGroup(merge->out()));
}

// Avoid contiguous indexing if going through non-divisible
// splits. Not all non-divisible splits need specific predicates, so
// this condition could be relaxed.
void ContigIDGroups::handle(Split* split, Direction direction) {
  if (direction == Direction::Forward) {
    const auto& divisible_splits = GpuLower::current()->divisibleSplitSet();
    const ExprGroup& split_group = graph_.toGroup(split);
    bool divisible = std::any_of(
        divisible_splits.begin(),
        divisible_splits.end(),
        [&](Split* divisible_split) -> bool {
          return split_group->has(divisible_split);
        });
    if (!divisible) {
      non_divisible_deps_.emplace(graph_.toGroup(split->outer()));
      non_divisible_deps_.emplace(graph_.toGroup(split->inner()));
    }
  } else { // Direction == Direction::Backward

    // --- Backward Propagation Logic for Split ---
    // A. Overall Purpose:
    // This section determines if the input to the split (`split->in()`) can be
    // considered a contiguous domain. This is not based on the contiguity of
    // `split->outer()` or `split->inner()` (which might not consume allocaiton
    // domains exclusively) Instead, it re-evaluates `split->in()` based on its
    // relationship to the original root allocation domains (`alloc_domains_`)
    // that `ContigIDGroups` was initialized with, and their properties. The
    // goal is to identify if `split->in()` itself represents a valid, larger
    // contiguous segment derived from those original allocation domains.

    const ValGroup& split_in_group = graph_.toGroup(split->in());

    // B. Pre-conditions: Consistent Ordering and Exclusive Allocation
    // Consumption `split->in()` must be consistently ordered with respect to
    // the original `alloc_domains_` and must exclusively consume all its
    // underlying allocation domains. If not, it cannot form a new contiguous
    // segment. These checks are skipped for predicate passes where physical
    // contiguity for indexing is not the concern.
    if (!(is_predicate_pass_ ||
          consistent_transform_info_->isConsistentlyOrdered(split->in()))) {
      return;
    }

    if (!consistent_transform_info_->exclusivelyConsumesAllocs(split->in())) {
      return;
    }

    // C. Core Contiguity Rule: Checking Original Allocation Domain Contiguity
    // This check applies if not a predicate pass. It verifies the contiguity
    // of the original allocation domains (from `alloc_domains_`) that
    // `split->in()` is composed of.
    // Rule: `split->in()` can be contiguous if, for all its constituent
    // original allocation domains, any non-contiguous original domain was the
    // "innermost" among them (i.e., all other original domains that are
    // "outer" to it must have been contiguous).
    // It also ensures all original domains forming `split->in()` are known
    // (i.e., part of the initial `alloc_domains_` list).
    if (!is_predicate_pass_) {
      auto original_allocs_it =
          consistent_transform_info_->findAllocIDs(split->in());
      if (original_allocs_it ==
          consistent_transform_info_->idToAllocIds().end()) {
        // Cannot find/verify the original allocation domains for split->in().
        return;
      } else {
        VectorOfUniqueEntries<IterDomain*> active_original_allocs =
            original_allocs_it->second;
        bool contiguity_ok = true;
        // Iterate through the master list of allocation domains provided at
        // init.
        for (const auto i : arange(alloc_domains_.size())) {
          IterDomain* current_master_alloc_domain = alloc_domains_.at(i);
          // Check if this master alloc domain is one of those that constitute
          // split->in().
          if (active_original_allocs.erase(current_master_alloc_domain) > 0) {
            bool original_domain_is_contig = alloc_contiguity_.at(i);
            if (!original_domain_is_contig && !active_original_allocs.empty()) {
              // This `current_master_alloc_domain` was not contiguous, AND
              // there are still other `active_original_allocs` of `split->in()`
              // that were "inner" to it (as erase proceeds from outer to inner
              // based on typical domain order in `findAllocIDs` results).
              // This violates the "non-contiguous only if innermost" rule.
              contiguity_ok = false;
              break;
            }
          }
        }
        if (!contiguity_ok || !active_original_allocs.empty()) {
          // Contiguity rule violated, or some original allocs that form
          // `split->in()` were not found in the master `alloc_domains_` list.
          return;
        }
      }
    }

    // D. Dependency Restrictions
    // `split->in()` cannot be marked as a new contiguous segment if it already
    // has resize dependencies. For predicate passes, it also cannot have
    // non-divisible split dependencies. These dependencies would have been
    // propagated to `split_in_group` by the main loop in the `ContigIDGroups`
    // constructor if they were applicable to its inputs (which are outputs
    // of the split expr in the backward view).
    if (is_predicate_pass_ && non_divisible_deps_.contains(split_in_group)) {
      return;
    }
    if (resize_deps_.contains(split_in_group)) {
      return;
    }

    // E. Outcome of Success: `split->in()` becomes a new contiguous domain.
    // All checks have passed. `split->in()` is now considered a contiguous
    // domain. It will represent the collective contiguity of the original
    // allocation domains it was derived from.
    contig_ids_.emplace(split_in_group);

    // Update `alloc_to_contig_ids_`: The original allocation domains that
    // `split->in()` is composed of are now mapped to `split_in_group`.
    // This means `split_in_group` is the new, larger contiguous ID that
    // represents them.
    auto final_original_allocs_it =
        consistent_transform_info_->findAllocIDs(split->in());
    NVF_ERROR(
        final_original_allocs_it !=
            consistent_transform_info_->idToAllocIds().end(),
        "Could not find original alloc IDs for split->in() after successful checks. This should not happen.");
    for (auto original_alloc_id : final_original_allocs_it->second) {
      alloc_to_contig_ids_[original_alloc_id] = split_in_group;
    }
  }
}

void ContigIDGroups::handle(Resize* resize, Direction direction) {
  if (direction == Direction::Forward) {
    resize_deps_.emplace(graph_.toGroup(resize->out()));
  } else {
    resize_deps_.emplace(graph_.toGroup(resize->in()));
  }
}

std::unordered_map<IterDomain*, ValGroup> getContigDomains(
    const std::vector<IterDomain*>& alloc_domains,
    const std::vector<bool>& alloc_contiguity,
    const ExprPath<ExprGroup>& path_from_alloc,
    const ValGraph& graph,
    bool is_predicate_pass) {
  ContigIDGroups contig_finder(
      alloc_domains,
      alloc_contiguity,
      path_from_alloc,
      graph,
      is_predicate_pass);

  return contig_finder.allocToContigIDs();
}

} // namespace nvfuser
