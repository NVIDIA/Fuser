// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#include <unordered_map>
#include <vector>

#include <ir/interface_nodes.h>
#include <ir/internal_base_nodes.h>
#include <ir/utils.h>
#include <optimization/mark_identity.h>

namespace nvfuser::optimization {

namespace {

using AllocationDomain = std::vector<IterDomain*>;

bool isCompatible(
    const AllocationDomain& existing_allocation_domain,
    const AllocationDomain& preferred_allocation_domain) {
  if (existing_allocation_domain.empty()) {
    return true;
  }
  return existing_allocation_domain == preferred_allocation_domain;
}

AllocationDomain explicitAllocationDomain(const TensorView* in) {
  AllocationDomain allocation_domain = in->getAllocationDomain();
  if (allocation_domain.empty()) {
    allocation_domain.reserve(in->nDims());
    for (size_t i = 0; i < in->nDims(); i++) {
      allocation_domain.push_back(in->axis(i));
    }
  }
  return allocation_domain;
}

void findAliasingOutput(
    const TensorView* source,
    std::unordered_map<
        const TensorView*,
        std::pair<TensorView*, AllocationDomain>>&
        preferred_allocation_domain) {
  std::queue<const TensorView*> q;
  q.push(source);
  preferred_allocation_domain[source] = {
      const_cast<TensorView*>(source), explicitAllocationDomain(source)};
  while (q.empty()) {
    const TensorView* in_tv = q.front();
    auto in_allocation_domain = preferred_allocation_domain.at(in_tv).second;
    q.pop();

    for (Expr* use : in_tv->uses()) {
      if (!use->isOneOf<LoadStoreOp, ViewOp>()) {
        continue;
      }

      Val* out = use->output(0);
      TensorView* out_tv = dynamic_cast<TensorView*>(out);
      if (out_tv == nullptr) {
        continue;
      }

      // FIXME: set preferred allocation domain.
      if (use->isA<LoadStoreOp>() &&
          use->as<LoadStoreOp>()->opType() == LoadStoreOpType::Set) {
        q.push(out_tv);
      } else if (use->isA<ViewOp>()) {
        q.push(out_tv);
      }
    }
  }
}

void markIdentity(Fusion* fusion) {
  std::cerr << "[jingyue] MarkIdentity" << std::endl;
  fusion->print();

  std::
      unordered_map<const TensorView*, std::pair<TensorView*, AllocationDomain>>
          preferred_allocation_domain;

  for (const Val* in : fusion->inputs()) {
    const TensorView* in_tv = dynamic_cast<const TensorView*>(in);
    if (in_tv == nullptr) {
      continue;
    }

    findAliasingOutput(in_tv, preferred_allocation_domain);
  }

  for (Val* out : fusion->outputs()) {
    if (TensorView* out_tv = dynamic_cast<TensorView*>(out)) {
      if (auto i = preferred_allocation_domain.find(out_tv);
          i != preferred_allocation_domain.end()) {
        const auto& [in_tv, allocation_domain] = i->second;
        // FIXME: set the right contiguity.
        out_tv->setAllocationDomain(allocation_domain, /*new_contiguity=*/true);
        fusion->markIdentity(in_tv, out_tv);
      }
    }
  }
}

} // namespace

void MarkIdentityPass::runPass(Fusion* fusion) {
  markIdentity(fusion);
}

} // namespace nvfuser::optimization
