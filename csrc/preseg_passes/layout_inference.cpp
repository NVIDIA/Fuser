// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#include <ir/all_nodes.h>
#include <ir/utils.h>
#include <preseg_passes/layout_inference.h>

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

class AllocationOrderInferencer : public OptOutConstDispatch {
 public:
  AllocationOrderInferencer(
      std::unordered_map<const TensorView*, AllocationOrder>& format_map)
      : format_map_(format_map) {}

 private:
  void handle(const UnaryOp*) override;
  void handle(const BinaryOp*) override;
  void handle(const BroadcastOp*) override;
  // TODO: Add more propagation rules
  //   void handle(const Reduction*) override;
  //   void handle(const LoadStoreOp*) override;
  //   void handle(const SqueezeOp*) override;
  //   void handle(const ExpandOp*) override;

 private:
  // format_map_ records the allocation order of each TensorView.
  // Since it only handles permutation from a rfactor domain to allocation
  // domain, it can be interpreted as:
  //
  // e.g. TV0 rfactor domain [i0, i1, i2]
  //            alloc domain [i0, i2, i1]
  //        allocation order   0,  2,  1
  std::unordered_map<const TensorView*, AllocationOrder>& format_map_;
};

// UnaryOp propagation forward allocation order from input to output
void AllocationOrderInferencer::handle(const UnaryOp* op) {
  TensorView* out = dynamic_cast<TensorView*>(op->out());
  if (out == nullptr) {
    return;
  }
  TensorView* in = op->in()->as<TensorView>();
  if (auto iter = format_map_.find(in); iter != format_map_.end()) {
    format_map_[out] = iter->second;
  }
}

// BinaryOp propagation tries to merge the allocation order of both inputs
//
//   1. when there's only one operand has a recorded allocation order, it
//   forwards that. This could happen when: we have inputs without recorded
//   allocation order, or one of the operands being a scalar;
//   2. When both tensor have recorded allocation order. The one tensor with
//   more non-broadcast iterdomain will be dominating the output allocation
//   order. The motivation behind it to avoid breaking allocation order
//   propagation from binary operation against unsqueezed vector tensors.
//
//   In the event of a tie, we'll just propagate the allocation order of lhs.
void AllocationOrderInferencer::handle(const BinaryOp* op) {
  TensorView* out = dynamic_cast<TensorView*>(op->out());
  if (out == nullptr) {
    return;
  }
  TensorView* lhs = dynamic_cast<TensorView*>(op->lhs());
  TensorView* rhs = dynamic_cast<TensorView*>(op->rhs());
  if (lhs == nullptr) {
    if (auto rhs_iter = format_map_.find(rhs); rhs_iter != format_map_.end()) {
      format_map_[out] = rhs_iter->second;
      return;
    }
  } else if (rhs == nullptr) {
    if (auto lhs_iter = format_map_.find(lhs); lhs_iter != format_map_.end()) {
      format_map_[out] = lhs_iter->second;
      return;
    }
  } else { // lhs != nullptr && rhs != nullptr
    auto lhs_iter = format_map_.find(lhs);
    auto rhs_iter = format_map_.find(rhs);
    if (lhs_iter != format_map_.end() && rhs_iter != format_map_.end()) {
      // if both allocation order agree, we just propagate it as-is.
      if (lhs_iter->second == rhs_iter->second) {
        format_map_[out] = lhs_iter->second;
        return;
      }

      if (countNonBroadcastID(lhs_iter->first) >=
          countNonBroadcastID(rhs_iter->first)) {
        format_map_[out] = lhs_iter->second;
        return;
      } else {
        format_map_[out] = rhs_iter->second;
        return;
      }
    } else if (lhs_iter != format_map_.end()) {
      format_map_[out] = lhs_iter->second;
      return;
    } else if (rhs_iter != format_map_.end()) {
      format_map_[out] = rhs_iter->second;
      return;
    }
    // we could reach here when neither operands has recorded allocation order.
    // We'll skip it for output
  }
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
void AllocationOrderInferencer::handle(const BroadcastOp* op) {
  TensorView* out = dynamic_cast<TensorView*>(op->out());
  if (out == nullptr) {
    return;
  }
  TensorView* in = op->in()->as<TensorView>();
  if (const auto& iter = format_map_.find(in); iter != format_map_.end()) {
    AllocationOrder out_format;
    int64_t out_rank = static_cast<int64_t>(out->nDims());

    int broadcast_seen_so_far = 0;
    std::vector<int64_t> offset_table(in->nDims(), 0);
    int offset_entry = 0;

    for (auto i : c10::irange(out_rank)) {
      if (op->isBroadcastDim(i)) {
        broadcast_seen_so_far++;
        // broadcast dimensions are default to outer dimensions
        // see note 2.a
        out_format.push_back(i);
      } else {
        // adjusting entry point by recording index compensation
        // i.e. broadcast dimensions inserted on the left of the old iterdomain
        // see note 2.b
        offset_table[offset_entry++] = broadcast_seen_so_far;
      }
    }

    for (auto i : c10::irange(in->nDims())) {
      auto format_entry = iter->second[i];
      out_format.push_back(format_entry + offset_table[format_entry]);
    }
    format_map_[out] = out_format;
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
//   memory_format_map.
std::unordered_map<const TensorView*, AllocationOrder> inferenceAllocationOrder(
    Fusion* fusion) {
  std::unordered_map<const TensorView*, AllocationOrder> memory_format_map;

  // Note: we only consider simple permutation of allocation domain to rfactor
  // domain.
  for (auto tv : ir_utils::filterByType<TensorView>(fusion->inputs())) {
    std::optional<AllocationOrder> permutation = ir_utils::computePermutation(
        TensorDomain::noReductions(tv->getMaybeRFactorDomain()),
        tv->getMaybeAllocationDomain());
    if (permutation.has_value()) {
      memory_format_map[tv] = permutation.value();
    }
  }

  // Initialize AllocationOrderInferencer with allocation order of input tensor
  // views
  AllocationOrderInferencer infer(memory_format_map);
  for (auto expr : fusion->exprs()) {
    infer.dispatch(expr);
  }

  // return the propagated map
  return memory_format_map;
}

} // namespace nvfuser
