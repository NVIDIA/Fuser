// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#include <optimization/layout_inference.h>
#include <ir/all_nodes.h>
#include <ir/utils.h>

namespace nvfuser {

namespace {

// move this to util maybe?
std::vector<int64_t> permutationIndex(const std::vector<int64_t>& permutation) {
  int rank = permutation.size();
  std::vector<int64_t> ret(rank, -1);
  for (int64_t i : c10::irange(permutation.size())) {
    ret.at(permutation[i]) = i;
  }
  return ret;
}

// Cases where we would want to skip modifying allocation domain
//   1. when allocation domain is already set on tv;
//   2. when tv is already an alias;
//
// NOTE: we should also check on contiguity, but I'm not sure if it make sense to look at contiguity flag without a meaningful allocation domain on tv.
bool skipPropagation(const TensorView* tv) {
  return tv->hasAllocation() || tv->fusion()->getOutputAlias(tv).type != AliasType::NoAlias;
}

class MemoryFormatInferencer : public OptOutConstDispatch {
 public:
  MemoryFormatInferencer(std::unordered_map<const TensorView*, MemoryFormat>& format_map) : format_map_(format_map) {}
 private:
  void handle(const UnaryOp*) override;
  void handle(const BinaryOp*) override;
  void handle(const BroadcastOp*) override;

  // void handle(const Reduction*) override;
  // void handle(const LoadStoreOp*) override;
  // void handle(const SqueezeOp*) override;
  // void handle(const ExpandOp*) override;
  std::unordered_map<const TensorView*, MemoryFormat> format_map_;
};

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

void MemoryFormatInferencer::handle(const BinaryOp* op) {
  // we only map the innermost dimension.
  // whichever one has a "better match" dominates the output format, where a better match meaning "less broadcast dimensions on the inside
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
      // go from innermost to outermost until we find the first one that's non-broadcast
      std::vector<int64_t> lhs_index = permutationIndex(lhs_iter->second);
      std::vector<int64_t> rhs_index = permutationIndex(rhs_iter->second);
      
      NVF_ERROR(lhs_index.size() == rhs_index.size());
      for (auto i : c10::irange(lhs_index.size())) {
        if (!rhs_iter->first->getMaybeRFactorDomain()[rhs_index[i]]->isBroadcast()) {
	  format_map_[out] = rhs_iter->second;
	  return;
	} else if (!lhs_iter->first->getMaybeRFactorDomain()[lhs_index[i]]->isBroadcast()) {
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

void MemoryFormatInferencer::handle(const BroadcastOp* op) {
  TensorView* out = dynamic_cast<TensorView*>(op->out());
  if (out == nullptr) {
    return;
  }
  TensorView* in = op->in()->as<TensorView>();
  // broadcast dimensions  are default to outer dimensions
  if (const auto& iter = format_map_.find(in), iter != format_map_.end()) {
    MemoryFormat out_format;
    int64_t cur_outer = out->nDims();
    int index_in = 0;
    for (auto i : c10::irange(out->nDims())) {
      out_format.push_back(op->isBroadcastDim(i) ? --cur_outer : iter->second[index_in++]);
    }
    format_map_[out] = out_format;
  }
}

} // namespace

std::unordered_map<const TensorView*, MemoryFormat> inferenceMemoryFormat(Fusion* fusion) {
  std::unordered_map<const TensorView*, MemoryFormat> memory_format_map;

  // keeping it simple, we are only handling permutation on alloc_domain
  for (auto tv : ir_utils::filterByType<TensorView>(fusion->inputs())) {
    std::optional<MemoryFormat> permutation = ir_utils::computePermutation(
      TensorDomain::noReductions(tv->getMaybeRFactorDomain()),
      tv->getMaybeAllocationDomain());
    if (permutation.has_value()) {
      memory_format_map[tv] = permutation.value();
    }
  }

  MemoryFormatInferencer infer(memory_format_map);
  for (auto expr : fusion->exprs()) {
    infer.dispatch(expr);
  }

  return memory_format_map;
}

} // namespace nvfuser
