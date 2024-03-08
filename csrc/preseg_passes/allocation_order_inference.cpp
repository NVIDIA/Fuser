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

AllocationOrder adjustAllocationOrder(const TensorView* tv, const AllocationOrder& alloc_order) {
  AllocationOrder ret = alloc_order;

  int64_t tv_rank = static_cast<int64_t>(tv->nDims());
  auto& rf_dom = tv->getMaybeRFactorDomain();
  std::stack<int64_t> removed_stack;
  for (auto idx : c10::irange(tv_rank)) {
    if (rf_dom[idx]->isReduction()) {
      removed_stack.push(idx);
      auto erase_iter = ret.begin();
      for (auto i = ret.begin(); i != ret.end(); i++) {
	if (*i == idx) {
	  erase_iter = i;
	  break;
	}
      }
      ret.erase(erase_iter);
    }
  }
  while (!stack.empty()) {
    idx = stack.top();
    stack.pop();
    for (auto i = ret.begin(); i != ret.end(); i++) {
      if (*i > idx) {
        --(*i);
      }
    }
  }
#ifndef NDEBUG
  std::set<int> unique_check;
  for (auto idx : ret) {
    NVF_ERROR(unique_check.count(idx) == 0 && idx >= 0 && idx < (int64_t)ret.size());
    unique_check.insert(idx);
  }
#endif
  return ret;
}

void allocationDomainUpdate(
    TensorView* tv,
    const AllocationOrder& alloc_order) {
  auto rfactor_dom = tv->getMaybeRFactorDomain();
  auto rank = rfactor_dom.size();

  std::vector<IterDomain*> allocation_domain(rank, nullptr);
  // specify allocation domain with dimension per allocation order.
  for (auto i : c10::irange(rank)) {
    allocation_domain[i] = rfactor_dom.at(alloc_order.at(i));
  }

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
  void handle(BinaryOp*) override;
  void handle(TernaryOp*) override;
  void handle(PadOp*) override;
  void handle(ReductionOp*) override;
  // TODO: Add more propagation rules
  // void handle(LoadStoreOp*) override;
  // void handle(SqueezeOp*) override;
  // void handle(ExpandOp*) override;

 private:
  // propagate allocation order from src to dst. Returns true when src has a
  // recorded allocation order, false otherwise.
  bool propagateAllocationOrder(TensorView* src, TensorView* dst) {
    if (auto iter = alloc_order_map_.find(src);
        iter != alloc_order_map_.end()) {
      alloc_order_map_[dst] = adjustAllocationOrder(iter->first, iter->second);
      return true;
    }
    return false;
  }

  // returns the candidate that dominates the allocation order.
  //
  // It scans through each candidate to find the first one that:
  //   1. is a TensorView
  //   2. has an entry in alloc_order_map_
  //   3. has the highest number of non_broadcast IterDomain
  //
  // The function is used to resolve allocation order propagation for operator
  // with multiple operands. The one operand with the most number of
  // non-broadcast IterDomain will be dominating the output allocation order.
  // The motivation behind it to avoid breaking allocation order propagation
  // from operands produced by broadcast. e.g. When a binary operator could take
  // in a channels_last 4d tensor and an unsqueezed bias vector. We'll want to
  // propagate the channels_last allocation order to output.
  TensorView* resolveAllocationOrder(const std::vector<Val*>& candidates);

  // alloc_order_map_ records the allocation order of each TensorView.
  // Since it only handles permutation from a rfactor domain to allocation
  // domain, it can be interpreted as:
  //
  // e.g. TV0 rfactor domain [i0, i1, i2]
  //            alloc domain [i0, i2, i1]
  //        allocation order   0,  2,  1
  std::unordered_map<const TensorView*, AllocationOrder>& alloc_order_map_;
};

TensorView* AllocationOrderInferencer::resolveAllocationOrder(
    const std::vector<Val*>& candidates) {
  TensorView* src = nullptr;
  size_t non_bc_high_water_mark = 0;

  // helper utils to count the number of non broadcast / non reduction iterdomain
  auto countLoopID = [](const TensorView* tv) -> size_t {
    return std::count_if(
        tv->getMaybeRFactorDomain().begin(),
        tv->getMaybeRFactorDomain().end(),
        [&](auto ptr_id) { return !ptr_id->isBroadcast() && !ptr_id->isReduction(); });
  };

  for (auto* val_ptr : candidates) {
    auto* tv_ptr = dynamic_cast<TensorView*>(val_ptr);
    // skip non TensorView entry
    if (tv_ptr == nullptr) {
      continue;
    }

    // skip entry that doesn't have an allocation order
    if (alloc_order_map_.count(tv_ptr) == 0) {
      continue;
    }

    // check if current entry sets new record for num of non broadcast / non reduction iterdomain
    if (size_t non_bc_count = countLoopID(tv_ptr);
        non_bc_count > non_bc_high_water_mark) {
      non_bc_high_water_mark = non_bc_count;
      src = tv_ptr;
    }
  }

  return src;
}

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
  for (auto index : adjustAllocationOrder(iter->first, iter->second)) {
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

void AllocationOrderInferencer::handle(BinaryOp* op) {
  auto* out = dynamic_cast<TensorView*>(op->out());
  if (out == nullptr) {
    return;
  }
  propagateAllocationOrder(resolveAllocationOrder(op->inputs()), out);
}

void AllocationOrderInferencer::handle(TernaryOp* op) {
  auto* out = dynamic_cast<TensorView*>(op->out());
  if (out == nullptr) {
    return;
  }
  propagateAllocationOrder(resolveAllocationOrder(op->inputs()), out);
}

void AllocationOrderInferencer::handle(PadOp* op) {
  auto* out = dynamic_cast<TensorView*>(op->out());
  auto* in = dynamic_cast<TensorView*>(op->in());
  propagateAllocationOrder(in, out);
}

void AllocationOrderInferencer::handle(ReductionOp* op) {
  auto* out = dynamic_cast<TensorView*>(op->out());
  auto* in = dynamic_cast<TensorView*>(op->in());
  propagateAllocationOrder(in, out);
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
        fusion->getOutputAlias(out_val).type != AllocationType::New) {
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
