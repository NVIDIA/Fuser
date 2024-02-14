// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#include <ir/all_nodes.h>
#include <ir/utils.h>
#include <optimization/layout_inference.h>

namespace nvfuser {

namespace {

// move this to util maybe?
std::vector<int64_t> ascendingAxes(const std::vector<int64_t>& permutation) {
  int64_t rank = static_cast<int64_t>(permutation.size());
  std::vector<int64_t> ret(rank, -1);
  for (int64_t i : c10::irange(rank)) {
    ret.at(rank - 1 - i) = permutation[i];
  }
  return ret;
}

class MemoryFormatInferencer : public OptOutConstDispatch {
 public:
  MemoryFormatInferencer(
      std::unordered_map<const TensorView*, MemoryFormat>& format_map)
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
  // format_map_ records the stride order (memory format) of each TensorView.
  // Since it only handles permutation from a rfactor domain to allocation
  // domain, it can be interpreted as:
  //
  // e.g. TV0 rfactor domain [i0, i1, i2]
  //            alloc domain [i0, i2, i1]
  //           memory format   0,  2,  1
  std::unordered_map<const TensorView*, MemoryFormat>& format_map_;
};

// UnaryOp propagation forward memory format from input to output
void MemoryFormatInferencer::handle(const UnaryOp* op) {
  TensorView* out = dynamic_cast<TensorView*>(op->out());
  if (out == nullptr) {
    return;
  }
  TensorView* in = op->in()->as<TensorView>();
  if (auto iter = format_map_.find(in); iter != format_map_.end()) {
    format_map_[out] = iter->second;
  }
}

// BinaryOp propagation tries to merge the memory format of both inputs
//
//   1. when there's only one operand has a recorded memory format, it forwards
//   that.
//   2. When both tensor have recorded memory format. It breaks tie based on the
//   innermost dimension. whichever operand has a "better match" dominates the
//   output format, where a "better match" meaning "less broadcast dimensions on
//   the inner dimension"
//
// e.g.
//   lhs TV0 rfactor_dom [i0, i1, b2]
//                         0   2   1
//   rhs TV0 rfactor_dom [i3, i4, b5]
//                         0   1   2
//   if we go from innermost to outermost order:
//       TV0 has i1 -> b2 -> i0
//       TV1 has b5 -> i4 -> i3
//   we see that TV0 encounters a non-broadcast iter domain first, so TV0 is the
//   dominating tensor. We'll produce an output with stride order identical to
//   that of TV0 in the record.
void MemoryFormatInferencer::handle(const BinaryOp* op) {
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
      // if both memory format agree, we just propagate it as-is.
      if (lhs_iter->second == rhs_iter->second) {
        format_map_[out] = lhs_iter->second;
        return;
      }
      // go from innermost to outermost until we find the first one that's
      // non-broadcast
      std::vector<int64_t> lhs_index = ascendingAxes(lhs_iter->second);
      std::vector<int64_t> rhs_index = ascendingAxes(rhs_iter->second);
      NVF_ERROR(lhs_index.size() == rhs_index.size());
      for (auto i : c10::irange(lhs_index.size())) {
        if (!rhs_iter->first->getMaybeRFactorDomain()[rhs_index[i]]
                 ->isBroadcast()) {
          format_map_[out] = rhs_iter->second;
          return;
        } else if (!lhs_iter->first->getMaybeRFactorDomain()[lhs_index[i]]
                        ->isBroadcast()) {
          format_map_[out] = lhs_iter->second;
          return;
        }
      }
    } else if (lhs_iter != format_map_.end()) {
      format_map_[out] = lhs_iter->second;
      return;
    } else if (rhs_iter != format_map_.end()) {
      format_map_[out] = rhs_iter->second;
      return;
    }
  }
}

// BroadcastOp propagation:
//   1. preserves all stride order of input iterdomain;
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
//         alloc dom [b3, b4, i0, i2, i1]
//                     1,  4,  
//                             0,  3,  2
//   so output TV1 will have stride order {1, 4, 0, 3, 2}
void MemoryFormatInferencer::handle(const BroadcastOp* op) {
  TensorView* out = dynamic_cast<TensorView*>(op->out());
  if (out == nullptr) {
    return;
  }
  TensorView* in = op->in()->as<TensorView>();
  if (const auto& iter = format_map_.find(in); iter != format_map_.end()) {
    MemoryFormat out_format;
    int64_t out_rank = static_cast<int64_t>(out->nDims());

    int broadcast_seen_so_far = 0;
    std::vector<int64_t> offset_per_entry(in->nDims(), 0);




    for (auto i : c10::irange(out_rank)) {
      // broadcast dimensions are default to outer dimensions
      out_format.push_back(
          op->isBroadcastDim(i) ? --cur_outer : iter->second[index_in++]);
    }
    format_map_[out] = out_format;
  }
}

} // namespace

// Note [ Memory Format Propagation ]
//
// The propagation tries to propagate memory format from inputs to the entire
// fusion:
//   1. Iterates through all inputs, looking for TensorView with allocatoin
//   domain that's a permutation of its corresponding rfactor domain and record
//   it as the memory format of the tensor;
//   2. Traverse the fusion IR, propagate memory format and record results in
//   memory_format_map.
std::unordered_map<const TensorView*, MemoryFormat> inferenceMemoryFormat(
    Fusion* fusion) {
  std::unordered_map<const TensorView*, MemoryFormat> memory_format_map;

  // Note: we only consider simple permutation of allocation domain to rfactor
  // domain.
  for (auto tv : ir_utils::filterByType<TensorView>(fusion->inputs())) {
    std::optional<MemoryFormat> permutation = ir_utils::computePermutation(
        TensorDomain::noReductions(tv->getMaybeRFactorDomain()),
        tv->getMaybeAllocationDomain());
    if (permutation.has_value()) {
      memory_format_map[tv] = permutation.value();
    }
  }

  // Initialize MemoryFormatInferencer with memory format of input tensor views
  MemoryFormatInferencer infer(memory_format_map);
  for (auto expr : fusion->exprs()) {
    infer.dispatch(expr);
  }

  // return the propagated map
  return memory_format_map;
}

} // namespace nvfuser
