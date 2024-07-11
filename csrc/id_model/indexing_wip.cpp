// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#include <id_model/contiguity.h>
#include <id_model/indexing.h>
#include <id_model/utils.h>

namespace nvfuser {

std::pair<std::deque<ValGroup>, std::deque<Val*>> TensorIndexer::
    getContigDomainsAndStrides(
        const std::vector<IterDomain*>& allocation_domains,
        const std::vector<Val*>& strides,
        const std::vector<bool>& contiguity,
        const ExprPath& traversal_path) const {
  const std::unordered_map<IterDomain*, ValGroup>& contig_domains =
      getContigDomains(
          allocation_domains,
          contiguity,
          reverse(traversal_path),
          traversalGraph(),
          concrete_info_,
          false);

  // Find contiguous domains to index
  std::unordered_set<ValGroup> already_indexed_domains;
  std::deque<ValGroup> contig_alloc_groups;
  std::deque<Val*> contig_strides;
  for (const auto i : c10::irange(allocation_domains.size())) {
    // Traverse back from the innermost domains so that the right
    // stride val is picked up for each contiguous domain
    auto i1 = allocation_domains.size() - 1 - i;
    IterDomain* allocation_domain = allocation_domains.at(i1);
    auto contig_domains_it = contig_domains.find(allocation_domain);
    NVF_ERROR(
        contig_domains_it != contig_domains.end(),
        "No contig domain mapping found for ",
        allocation_domain->toString());

    const ValGroup& contig_domain_group = contig_domains_it->second;
    if (already_indexed_domains.find(contig_domain_group) !=
        already_indexed_domains.end()) {
      VERBOSE() << "Already indexed: " << allocation_domain->toString()
                << std::endl;
      continue;
    }
    already_indexed_domains.emplace(contig_domain_group);

    if (!contig_domain_group->has(allocation_domain)) {
      VERBOSE() << "Contig indexing: "
                << contig_domain_group->front()->toString() << " instead of "
                << allocation_domain->toString() << std::endl;
    } else {
      VERBOSE() << "Non contig indexing: " << allocation_domain->toString()
                << std::endl;
    }

    VERBOSE() << "Stride: " << strides.at(i1)->toInlineString() << std::endl;

    contig_alloc_groups.push_front(contig_domain_group);
    contig_strides.push_front(strides.at(i1));
  }

  return {contig_alloc_groups, contig_strides};
}

} // namespace nvfuser
