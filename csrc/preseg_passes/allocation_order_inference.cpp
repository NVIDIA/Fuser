// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#include <ir/all_nodes.h>
#include <ir/utils.h>
#include <iter_visitor.h>
#include <preseg_passes/allocation_order_inference.h>
#include <root_domain_map.h>

namespace nvfuser::preseg_passes {

namespace {

void allocationDomainUpdate(
    TensorView* tv,
    const AllocationOrder& alloc_order) {
  auto rfactor_dom = tv->getMaybeRFactorDomain();

  // Allocation order is only marked for non-reduction iterdomain
  auto no_bc_rfactor_dom = TensorDomain::noReductions(rfactor_dom);
  auto rank = no_bc_rfactor_dom.size();
  std::vector<IterDomain*> allocation_domain(rank, nullptr);
  allocation_domain.reserve(rfactor_dom.size());
  // specify allocation domain with non-reduction dimension per allocation
  // order.
  for (auto i : c10::irange(rank)) {
    allocation_domain[i] = no_bc_rfactor_dom.at(alloc_order.at(i));
  }

  // reduction iter domain's position in allocation domain doesn't matter,
  // insert them at the end
  std::copy_if(
      rfactor_dom.begin(),
      rfactor_dom.end(),
      std::back_inserter(allocation_domain),
      [](const IterDomain* id) { return id->isReduction(); });

  tv->setAllocationDomain(allocation_domain, true);
}

class AllocationOrderInferencer : public IterVisitor {
 public:
  AllocationOrderInferencer(
      std::unordered_map<const TensorView*, AllocationOrder>& alloc_order_map)
      : alloc_order_map_(alloc_order_map) {}

 protected:
  using IterVisitor::handle;

  void handle(UnaryOp*) override;
  void handle(BroadcastOp*) override;
  // TODO: Add more propagation rules
  // void handle(BinaryOp*) override;
  // void handle(Reduction*) override;
  // void handle(LoadStoreOp*) override;
  // void handle(SqueezeOp*) override;
  // void handle(ExpandOp*) override;

 private:
  // propagate allocation order from src to dst. Returns true when src has a
  // recorded allocation order, false otherwise.
  bool propagateAllocationOrder(TensorView* src, TensorView* dst) {
    if (auto iter = alloc_order_map_.find(src);
        iter != alloc_order_map_.end()) {
      alloc_order_map_[dst] = iter->second;
      return true;
    }
    return false;
  }

  // alloc_order_map_ records the allocation order of each TensorView.
  // Since it only handles permutation from a rfactor domain to allocation
  // domain, it can be interpreted as:
  //
  // e.g. TV0 rfactor domain [i0, i1, i2]
  //            alloc domain [i0, i2, i1]
  //        allocation order   0,  2,  1
  std::unordered_map<const TensorView*, AllocationOrder>& alloc_order_map_;
};

// UnaryOp propagation forward allocation order from input to output
void AllocationOrderInferencer::handle(UnaryOp* op) {
  auto* out = dynamic_cast<TensorView*>(op->out());
  if (out == nullptr) {
    return;
  }
  auto* in = op->in()->as<TensorView>();
  propagateAllocationOrder(in, out);
}

// BroadcastOp propagation:
//   1. preserves all allocation order of input iterdomain;
//   2. stacks all added broadcast iter domain on outputs as outer dimensions in
//   their natural position
//
// e.g.
//   TV0 rfactor dom [i0', i1', i2'] @ allocation order {0, 2, 1}
//    |    alloc dom [i0', i2', i1']
//    |
//    |
//    BroadcastOp
//    |
//    v
//   TV1 rfactor dom [i0, b3, i1, i2, b4]
//
//   step 0:
//       scan through all iterdomain in output TV1's rfactor domain
//       insert all broadcast domain to alloc_domain[b3, b4];
//
//   step 1:
//       computing iterdomain mapping from input to output;
//       [i0', i2', i1'] -> [i0, i2, i1]
//
//   step 2:
//       follow allocation order on input, insert the mapped iter domain on
//       output to alloc_domain[b3, b4, i0, i2, i1];
//
//   step 3:
//       compute permutation from alloc_domain to TV1's rfactor domain;
//       so output TV1 will have allocation order {1, 4, 0, 3, 2}
void AllocationOrderInferencer::handle(BroadcastOp* op) {
  auto* out = dynamic_cast<TensorView*>(op->out());
  if (out == nullptr) {
    return;
  }
  auto* in = op->in()->as<TensorView>();

  auto iter = alloc_order_map_.find(in);
  // early return when there's no recorded allocation order for `in`
  if (iter == alloc_order_map_.end()) {
    return;
  }

  size_t out_rank = out->nDims();
  std::vector<IterDomain*> alloc_domain;
  alloc_domain.reserve(out_rank);

  // step 0: insert all broadcast iterdomain in output
  for (auto i : c10::irange(out_rank)) {
    if (op->isBroadcastDim(i)) {
      alloc_domain.push_back(out->getMaybeRFactorDomain()[i]);
    }
  }

  // step 1: compute root domain map
  auto in_to_out_map = PairwiseRootDomainMap(in, out).mapProducerToConsumer();
  const auto& in_root_domain =
      TensorDomain::noReductions(in->getMaybeRFactorDomain());

  // step 2: push each mapped iterdomain
  for (auto index : iter->second) {
    alloc_domain.push_back(in_to_out_map.at(in_root_domain.at(index)));
  }

  // step 3: compute permutation
  std::optional<AllocationOrder> permutation =
      ir_utils::computePermutation(out->getMaybeRFactorDomain(), alloc_domain);

  NVF_ERROR(
      permutation.has_value(),
      "allocation order propagation on broadcast op failed to compute valid permutation");
  alloc_order_map_[out] = permutation.value();
}

} // namespace

// Note [ Allocation Order Propagation ]
//
// The propagation tries to propagate allocation order from inputs to the entire
// fusion:
//   1. Iterates through all inputs, looking for TensorView with allocation
//   domain that's a permutation of its corresponding rfactor domain and record
//   it as the allocation order of the tensor;
//   2. Traverse the fusion IR, propagate allocation order and record results in
//   alloc_order_map.
std::unordered_map<const TensorView*, AllocationOrder> inferenceAllocationOrder(
    Fusion* fusion) {
  std::unordered_map<const TensorView*, AllocationOrder> alloc_order_map;

  // Note: we only consider simple permutation of allocation domain to rfactor
  // domain.
  for (auto tv : ir_utils::filterByType<TensorView>(fusion->inputs())) {
    std::optional<AllocationOrder> permutation = ir_utils::computePermutation(
        TensorDomain::noReductions(tv->getMaybeRFactorDomain()),
        TensorDomain::noReductions(tv->getMaybeAllocationDomain()));
    if (permutation.has_value()) {
      alloc_order_map[tv] = permutation.value();
    }
  }

  // Initialize AllocationOrderInferencer with allocation order of input tensor
  // views
  AllocationOrderInferencer infer(alloc_order_map);
  infer.traverse(fusion);

  // return the propagated map
  return alloc_order_map;
}

void AllocationDomainPass::runPass(Fusion* fusion) {
  std::unordered_map<const TensorView*, AllocationOrder> stride_mapping =
      inferenceAllocationOrder(fusion);

  for (Val* out_val : fusion->outputs()) {
    auto* out_tv = dynamic_cast<TensorView*>(out_val);
    // skip:
    //   1. non-tensor output;
    //   2. tensor output with allocation specified, assuming everything is
    //   semantical
    //   3. tensor output that's aliasing (Does aliased src matter?)
    if (out_tv == nullptr || out_tv->hasAllocation() ||
        fusion->getOutputAlias(out_val).type != AllocationType::NoAlias) {
      continue;
    }

    auto mapped_entry = stride_mapping.find(out_tv);
    if (mapped_entry == stride_mapping.end()) {
      continue;
    }

    allocationDomainUpdate(out_tv, mapped_entry->second);
  }
}

} // namespace nvfuser::preseg_passes
