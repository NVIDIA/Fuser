// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on

#include <expr_evaluator.h>
#include <ir/all_nodes.h>
#include <ir/builder.h>
#include <ir/cloner.h>
#include <ir/iostream.h>
#include <ir/utils.h>
#include <multidevice/utils.h>
#include <polymorphic_value.h>
#include <tensor_metadata.h>

namespace nvfuser {

namespace {

// Forward traverse from rFactor domain to allocation domain, compute frontier
// sizes and strides, validate that splits are divisible and merges are
// contiguous, and update active_ids_ correspondingly.
class ForwardTraverseFromRFactorToAlloc {
  ExpressionEvaluator& ee_;
  std::unordered_map<IterDomain*, std::pair<int64_t, int64_t>>& active_ids_;

  void handle(Split* split) {
    
    auto in = split->in();
    auto inner = split->inner();
    auto outer = split->outer();
    auto in_it = active_ids_.find(in);
    // NVF_ERROR(in_it != active_ids_.end())
    if (in_it == active_ids_.end()) {
      // TODO: see [Allocation domain on both side of rFactor]
      return;
    }
    auto [in_size, in_stride] = in_it->second;
    auto factor = ee_.evaluate(split->factor()).as<int64_t>();
    // std::cout << "Split " << inner->toString() << " " << outer->toString() << std::endl;
    // std::cout << "SHARDED In size/ factor " << in_size << " " << factor << std::endl;
    // A device parallelized outer axis indicates the logical axis was sharded. 
    // in_size and in_stride refer to sharded sizes, adjust both to refer to unsharded sizes. 
    // TODO: the logic seems to assume a split is always an inner split? But, sharding
    // uses an outersplit... 
    if (outer->isDeviceDim()) {
      in_size *= factor;
      factor *= factor;
    }
    // std::cout << "UNSHARDED In size/ factor " << in_size << " " << factor << std::endl;

    NVF_ERROR(
        in_size % factor == 0,
        "The rFactor domain and allocation domain of fusion input/output ",
        "tensors must be a one-to-one map, therefore, ",
        "non-divisible split is not allowed in allocation domain");
    NVF_ERROR(active_ids_.erase(in) == 1);
    NVF_ERROR(
        active_ids_
            .emplace(inner, std::pair<int64_t, int64_t>{factor, in_stride})
            .second);
    NVF_ERROR(active_ids_
                  .emplace(
                      outer,
                      std::pair<int64_t, int64_t>{
                          in_size / factor, in_stride * factor})
                  .second);
  }

  void handle(Merge* merge) {
    auto inner = merge->inner();
    auto outer = merge->outer();
    auto out = merge->out();
    auto inner_it = active_ids_.find(inner);
    auto outer_it = active_ids_.find(outer);
    // NVF_ERROR(inner_it != active_ids_.end())
    // NVF_ERROR(outer_it != active_ids_.end())
    if (inner_it == active_ids_.end() || outer_it == active_ids_.end()) {
      // TODO: see [Allocation domain on both side of rFactor]
      return;
    }
    auto [inner_size, inner_stride] = inner_it->second;
    auto [outer_size, outer_stride] = outer_it->second;
    NVF_ERROR(
        inner_stride * inner_size == outer_stride,
        "The rFactor domain and allocation domain of fusion input/output ",
        "tensors must be a one-to-one map, therefore, ",
        "merging of discontiguous dimensions is not allowed in allocation domain");
    NVF_ERROR(active_ids_.erase(inner) == 1);
    NVF_ERROR(active_ids_.erase(outer) == 1);
    NVF_ERROR(active_ids_
                  .emplace(
                      out,
                      std::pair<int64_t, int64_t>{
                          inner_size * outer_size, inner_stride})
                  .second);
  }

  void handle(Expr* expr) {
    if (auto split = dynamic_cast<Split*>(expr)) {
      handle(split);
    } else if (auto merge = dynamic_cast<Merge*>(expr)) {
      handle(merge);
    } else {
      NVF_ERROR(false, "Unsupported transormation in allocation domain");
    }
  }

 public:
  ForwardTraverseFromRFactorToAlloc(
      ExpressionEvaluator& ee,
      std::unordered_map<IterDomain*, std::pair<int64_t, int64_t>>& active_ids)
      : ee_(ee), active_ids_(active_ids) {}

  void run(
      TensorView* tv,
      const std::vector<IterDomain*>& rfactor,
      const std::vector<IterDomain*>& alloc) {
    auto forward_exprs = StmtSort::getExprsBetween(
        {rfactor.begin(), rfactor.end()}, {alloc.begin(), alloc.end()});
    for (auto expr : forward_exprs) {
      handle(expr);
    }
  }
};

// Similar to ForwardTraverseFromRFactorToAlloc, but in the opposite direction.
class BackwardTraverseFromRFactorToAlloc {
  at::Tensor tensor_;
  ExpressionEvaluator& ee_;
  std::unordered_map<IterDomain*, std::pair<int64_t, int64_t>>& active_ids_;

  void handle(Split* split) {
    auto in = split->in();
    auto inner = split->inner();
    auto outer = split->outer();
    auto inner_it = active_ids_.find(inner);
    auto outer_it = active_ids_.find(outer);
    // NVF_ERROR(inner_it != active_ids_.end())
    // NVF_ERROR(outer_it != active_ids_.end())
    if (inner_it == active_ids_.end() || outer_it == active_ids_.end()) {
      // TODO: see [Allocation domain on both side of rFactor]
      return;
    }
    auto [inner_size, inner_stride] = inner_it->second;
    auto [outer_size, outer_stride] = outer_it->second;
    NVF_ERROR(
        inner_stride * inner_size == outer_stride,
        "The rFactor domain and allocation domain of fusion input/output ",
        "tensors must be a one-to-one map, therefore, ",
        "splitting one dimension into discontiguous dimensions is not allowed in allocation domain");
    NVF_ERROR(active_ids_.erase(inner) == 1);
    NVF_ERROR(active_ids_.erase(outer) == 1);
    NVF_ERROR(active_ids_
                  .emplace(
                      in,
                      std::pair<int64_t, int64_t>{
                          inner_size * outer_size, inner_stride})
                  .second);
  }

  void handle(Merge* merge) {
    auto inner = merge->inner();
    auto outer = merge->outer();
    auto out = merge->out();
    auto factor = ee_.evaluate(inner->extent()).as<int64_t>();
    auto out_it = active_ids_.find(out);
    // NVF_ERROR(out_it != active_ids_.end())
    if (out_it == active_ids_.end()) {
      // TODO: see [Allocation domain on both side of rFactor]
      return;
    }
    auto [out_size, out_stride] = out_it->second;
    NVF_ERROR(
        out_size % factor == 0,
        "The rFactor domain and allocation domain of fusion input/output ",
        "tensors must be a one-to-one map, therefore, ",
        "the size of the output must divisible by the size of inner dimension");
    NVF_ERROR(active_ids_.erase(out) == 1);
    NVF_ERROR(
        active_ids_
            .emplace(inner, std::pair<int64_t, int64_t>{factor, out_stride})
            .second);
    NVF_ERROR(active_ids_
                  .emplace(
                      outer,
                      std::pair<int64_t, int64_t>{
                          out_size / factor, out_stride * factor})
                  .second);
  }

  void handle(Expr* expr) {
    if (auto split = dynamic_cast<Split*>(expr)) {
      handle(split);
    } else if (auto merge = dynamic_cast<Merge*>(expr)) {
      handle(merge);
    } else {
      NVF_ERROR(false, "Unsupported transormation in allocation domain");
    }
  }

 public:
  BackwardTraverseFromRFactorToAlloc(
      ExpressionEvaluator& ee,
      std::unordered_map<IterDomain*, std::pair<int64_t, int64_t>>& active_ids)
      : ee_(ee), active_ids_(active_ids) {}

  void run(
      TensorView* tv,
      const std::vector<IterDomain*>& rfactor,
      const std::vector<IterDomain*>& alloc) {
    auto backward_exprs = StmtSort::getExprsBetween(
        {alloc.begin(), alloc.end()}, {rfactor.begin(), rfactor.end()});
    std::reverse(backward_exprs.begin(), backward_exprs.end());
    for (auto expr : backward_exprs) {
      handle(expr);
    }
  }
};

void validateAllocationSizesAndStrides(
    const std::vector<IterDomain*>& alloc_dom_no_reductions,
    const std::vector<std::optional<bool>>& contiguity,
    c10::IntArrayRef sizes,
    c10::IntArrayRef strides) {
  NVF_ERROR(sizes.size() == strides.size());

  // Validate contiguity
  int64_t contiguous_stride = 1;
  auto contiguity_rev = contiguity.crbegin();
  for (int64_t i = (int64_t)sizes.size() - 1; i >= 0; i--) {
    if (alloc_dom_no_reductions.at(i)->isBroadcast()) {
      continue;
    }
    while (!contiguity_rev->has_value()) {
      contiguity_rev++;
    }
    auto size = sizes.at(i);
    auto stride = strides.at(i);
    NVF_ERROR(!contiguity.empty());
    auto last_contiguity = *contiguity_rev;
    NVF_ERROR(
        last_contiguity.has_value(),
        "I don't think this check makes sense, but unfortunately ",
        "clang-tidy is not smart enough to infer from the context that this is always true.");
    if (*last_contiguity) {
      NVF_CHECK(
          stride == contiguous_stride,
          "Stride mismatch with contiguity info. ",
          " allocation domain: ",
          ir_utils::toString(alloc_dom_no_reductions),
          " dim: ",
          i,
          " expected stride: ",
          contiguous_stride,
          " actual stride: ",
          stride);
    }
    // TODO: Not certain how to handle stride when size refers to unsharded tensor.
    if (!alloc_dom_no_reductions.at(i)->isDeviceDim())
      contiguous_stride = stride * size;
    contiguity_rev++;
  }
  NVF_ERROR(
      std::none_of(
          contiguity_rev,
          contiguity.crend(),
          [](auto c_flag) { return c_flag.has_value(); }),
      "The size of contiguity mismatch with the dimensionality of allocation domain");

  // Validate that for expanded broadcast, the stride must be zero.
  for (int64_t i : c10::irange((int64_t)strides.size())) {
    if (auto alloc_id = alloc_dom_no_reductions.at(i);
        alloc_id->hasExpandedExtent()) {
      auto stride = strides.at(i);
      NVF_CHECK(
          stride == 0,
          "Expecting an expanded dimension on dimension ",
          i,
          " but found stride ",
          stride);
    }
  }
}

} // namespace

std::pair<std::vector<int64_t>, std::vector<int64_t>>
inferAndValidateAllocationSizesAndStrides(
    const at::Tensor& tensor,
    TensorView* tv,
    ExpressionEvaluator ee) {
  if (tv == nullptr || !tv->hasAllocation()) {
    // When tv is nullptr, or tv does not have allocation, the given sizes and
    // strides should already be in the target format. So nothing to do here.
    std::vector<int64_t> sizes;
    std::vector<int64_t> strides;
    for (auto i : c10::irange(tensor.dim())) {
      sizes.emplace_back(tensor.size(i));
      strides.emplace_back(tensor.stride(i));
    }
    return {sizes, strides};
  }
  const auto& alloc =
      TensorDomain::noReductions(tv->getMaybeAllocationDomain());
  const auto& rfactor = TensorDomain::noReductions(tv->getMaybeRFactorDomain());

  // active IDs and their shape and stride
  // active IDs are initially bound to sharded shape and stride. 
  std::unordered_map<IterDomain*, std::pair<int64_t, int64_t>> active_ids;
  NVF_ERROR((int64_t)rfactor.size() == tensor.dim());
  for (int64_t i : c10::irange((int64_t)rfactor.size())) {
    auto rf_id = rfactor.at(i);
    active_ids[rf_id] = {tensor.size(i), tensor.stride(i)};
  }

  ForwardTraverseFromRFactorToAlloc(ee, active_ids).run(tv, rfactor, alloc);
  BackwardTraverseFromRFactorToAlloc(ee, active_ids).run(tv, rfactor, alloc);

  // Now active_ids should contain the final sizes and strides, unordered. We
  // need to put them to the correct order.
  std::vector<int64_t> sizes;
  std::vector<int64_t> strides;
  sizes.reserve(alloc.size());
  strides.reserve(alloc.size());
  for (auto i : c10::irange(alloc.size())) {
    auto id = alloc.at(i);
    sizes.emplace_back(active_ids.at(id).first);
    strides.emplace_back(active_ids.at(id).second);
  }
  // Only validate final sizes and strides when we have a non-empty tensor.
  if (tensor.numel() != 0) {
    validateAllocationSizesAndStrides(
        alloc, tv->getContiguity(), sizes, strides);
  }
  return {std::move(sizes), std::move(strides)};
}

std::vector<PolymorphicValue> GetMetaData::evaluate(
    const ExpressionEvaluator& ee,
    const std::vector<PolymorphicValue>& inputs) const {
  NVF_ERROR(inputs.size() == 1, "GetMetaData expects 1 input");
  NVF_ERROR(
      in()->isA<TensorView>(),
      "Currently, GetMetaData only supports TensorView");
  TensorView* tv = in()->as<TensorView>();

  const at::Tensor& input = inputs.at(0).as<at::Tensor>();

  NVF_ERROR(
      input.is_cuda() || input.is_meta(),
      "GetMetaData expects a CUDA tensor as input, but got undefined tensor");

  std::shared_ptr<Struct> struct_ = std::make_shared<TensorMetaData>();
  TensorMetaData* metadata = (TensorMetaData*)struct_.get();
  metadata->dtype =
      std::get<PrimDataType>(aten_to_data_type(input.scalar_type()).type);
  metadata->data = input.data_ptr();
  // If tensor is sharded then logical_size and logical_stride will
  // refer to size and stride of the sharded tensor.
  // TODO: Is this assumption is correct? Maybe it should be the unsharded data. 
  metadata->logical_size = input.sizes();
  metadata->logical_stride = input.strides();
  if (tv->hasAllocation()) {
    auto allocation_data =
        inferAndValidateAllocationSizesAndStrides(input, tv, ee);
    metadata->alloc_size_data = std::move(allocation_data.first);
    metadata->alloc_size = c10::makeArrayRef(metadata->alloc_size_data);
    metadata->alloc_stride_data = std::move(allocation_data.second);
    metadata->alloc_stride = c10::makeArrayRef(metadata->alloc_stride_data);
    if (isSharded(tv)) {
      // Explicitly update the logical for sharded tensors. 
      // TODO: All sharded tensors should have their allocation domain explicitly set so that
      // the logical sizes and strides are calculated correctly. 
      auto alloc = TensorDomain::noReductions(tv->getMaybeAllocationDomain());
      std::vector<int64_t> unsharded_size;
      std::vector<int64_t> unsharded_stride;
      int factor = 1;
      // TODO: This is a bit hacky...
      for (auto i : c10::irange(alloc.size())) {
        auto id = alloc[i];
        auto size = metadata->alloc_size_data[i];
        auto stride = metadata->alloc_stride_data[i];
        if (id->isDeviceDim()) {
          factor = tv->getDeviceMesh().vector().size();
        } else {
          unsharded_size.push_back(size * factor);
          unsharded_stride.push_back(stride);
          factor = 1;
        }
      }
      // TODO: the logical size is unsharded size, but the stride is sharded...
      metadata->logical_size_data = std::move(unsharded_size);
      metadata->logical_size = c10::makeArrayRef(metadata->logical_size_data);
      metadata->logical_stride_data = std::move(unsharded_stride);
      metadata->logical_stride = std::move(metadata->logical_stride_data);
    }
  } else {
    metadata->alloc_size = input.sizes();
    metadata->alloc_stride = input.strides();
    // TODO: validateAllocationSizesAndStrides
  }
  return {PolymorphicValue(std::move(struct_))};
}

} // namespace nvfuser
