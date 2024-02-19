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

namespace nvfuser {

namespace {

int countNonBroadcastID(const TensorView* tv) {
  int count = 0;
  std::for_each(
      tv->getMaybeRFactorDomain().begin(),
      tv->getMaybeRFactorDomain().end(),
      [&](auto ptr_id) {
        if (!ptr_id->isBroadcast()) {
          ++count;
        }
      });
  return count;
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
  TensorView* out = dynamic_cast<TensorView*>(op->out());
  if (out == nullptr) {
    return;
  }
  TensorView* in = op->in()->as<TensorView>();
  propagateAllocationOrder(in, out);
}

// BroadcastOp propagation:
//   1. preserves all allocation order of input iterdomain;
//   2. stacks all added broadcast iter domain on outputs as outer dimensions in
//   their natural position
//
// e.g.
//   TV0 rfactor dom [i0, i1, i2] @ stride order {0, 2, 1}
//    |    alloc dom [i0, i2, i1]
//    |
//    |
//    BroadcastOp
//    |
//    v
//   TV1 rfactor dom [i0, b3, i1, i2, b4]
//         alloc dom [b3, b4, i0, i2, i1]  *see note 1
//                     1,  4,
//                             0,  3,  2   *see note 2
//   note 1:
//       keeping the alloc domain from input [i0, i2, i1]
//       stack all broadcast in rfactor [b3, b4].
//       concat([b3, b4], [i0, i2, i1])
//
//   note 2:
//       computing the new output allocation order
//       We'll scan through the rfactor domain of output where:
//       a. insert any broadcast iterdomain index as we encounter them
//       b. adjust the index of entry from input's rfactor domain
//
//   so output TV1 will have stride order {1, 4, 0, 3, 2}
void AllocationOrderInferencer::handle(BroadcastOp* op) {
  TensorView* out = dynamic_cast<TensorView*>(op->out());
  if (out == nullptr) {
    return;
  }
  TensorView* in = op->in()->as<TensorView>();
  if (const auto& iter = alloc_order_map_.find(in); iter != alloc_order_map_.end()) {
    AllocationOrder out_order;
    int64_t out_rank = static_cast<int64_t>(out->nDims());

    int broadcast_seen_so_far = 0;
    std::vector<int64_t> offset_table(in->nDims(), 0);
    int offset_entry = 0;

    for (auto i : c10::irange(out_rank)) {
      if (op->isBroadcastDim(i)) {
        broadcast_seen_so_far++;
        // broadcast dimensions are default to outer dimensions
        // see note 2.a
        out_order.push_back(i);
      } else {
        // adjusting entry point by recording index compensation
        // i.e. broadcast dimensions inserted on the left of the old iterdomain
        // see note 2.b
        offset_table[offset_entry++] = broadcast_seen_so_far;
      }
    }

    for (auto i : c10::irange(in->nDims())) {
      auto order_entry = iter->second[i];
      out_order.push_back(order_entry + offset_table[order_entry]);
    }
    alloc_order_map_[out] = out_order;
  }
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

} // namespace nvfuser
