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

  for (const auto index_domain_i : c10::irange(alloc_domains_.size())) {
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
    for (auto alloc_id_i : c10::irange(alloc_domains_.size())) {
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
