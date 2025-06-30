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
#include <instrumentation.h>

namespace nvfuser {

namespace {

// Forward traverse from logical domain to allocation domain, compute frontier
// sizes and strides, validate that splits are divisible and merges are
// contiguous, and update active_ids_ correspondingly.
class ForwardTraverseFromLogicalToAlloc {
  ExpressionEvaluator& ee_;
  std::unordered_map<IterDomain*, std::pair<int64_t, int64_t>>& active_ids_;

  void handle(Split* split) {
    auto in = split->in();
    auto inner = split->inner();
    auto outer = split->outer();
    auto in_it = active_ids_.find(in);
    // NVF_ERROR(in_it != active_ids_.end())
    if (in_it == active_ids_.end()) {
      // TODO: see [Allocation domain on both side of logical]
      return;
    }

    auto [in_size, in_stride] = in_it->second;
    auto factor = ee_.evaluate(split->factor()).as<int64_t>();
    NVF_ERROR(
        in_size % factor == 0,
        "The logical domain and allocation domain of fusion input/output ",
        "tensors must be a one-to-one map, therefore, ",
        "non-divisible split is not allowed in allocation domain");

    int64_t inner_size = 0;
    int64_t outer_size = 0;
    if (split->innerSplit()) {
      outer_size = in_size / factor;
      inner_size = factor;
    } else {
      outer_size = factor;
      inner_size = in_size / factor;
    }

    NVF_ERROR(active_ids_.erase(in) == 1);
    NVF_ERROR(active_ids_.emplace(inner, std::make_pair(inner_size, in_stride))
                  .second);
    NVF_ERROR(
        active_ids_
            .emplace(outer, std::make_pair(outer_size, in_stride * inner_size))
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
      // TODO: see [Allocation domain on both side of logical]
      return;
    }
    auto [inner_size, inner_stride] = inner_it->second;
    auto [outer_size, outer_stride] = outer_it->second;
    NVF_ERROR(
        inner_stride * inner_size == outer_stride,
        "Merging of discontiguous dimensions is not allowed in allocation "
        "domain. An allocation IterDomain can't have two different strides.");
    NVF_ERROR(active_ids_.erase(inner) == 1);
    NVF_ERROR(active_ids_.erase(outer) == 1);
    NVF_ERROR(
        active_ids_
            .emplace(out, std::make_pair(inner_size * outer_size, inner_stride))
            .second);
  }

  void handle(Expr* expr) {
    if (auto split = dynamic_cast<Split*>(expr)) {
      handle(split);
    } else if (auto merge = dynamic_cast<Merge*>(expr)) {
      handle(merge);
    } else {
      NVF_THROW("Unsupported transormation in allocation domain");
    }
  }

 public:
  ForwardTraverseFromLogicalToAlloc(
      ExpressionEvaluator& ee,
      std::unordered_map<IterDomain*, std::pair<int64_t, int64_t>>& active_ids)
      : ee_(ee), active_ids_(active_ids) {}

  void run(
      TensorView* tv,
      const std::vector<IterDomain*>& logical,
      const std::vector<IterDomain*>& alloc) {
    FUSER_PERF_SCOPE("ForwardTraverseFromLogicalToAlloc::run");
    auto forward_exprs = StmtSort::getExprsBetween(
        {logical.begin(), logical.end()}, {alloc.begin(), alloc.end()});
    for (auto expr : forward_exprs) {
      handle(expr);
    }
  }
};

// Similar to ForwardTraverseFromLogicalToAlloc, but in the opposite direction.
class BackwardTraverseFromLogicalToAlloc {
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
      // TODO: see [Allocation domain on both side of logical]
      return;
    }
    auto [inner_size, inner_stride] = inner_it->second;
    auto [outer_size, outer_stride] = outer_it->second;
    NVF_ERROR(
        inner_stride * inner_size == outer_stride,
        "The logical domain and allocation domain of fusion input/output ",
        "tensors must be a one-to-one map, therefore, ",
        "splitting one dimension into discontiguous dimensions is not allowed "
        "in allocation domain");
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
      // TODO: see [Allocation domain on both side of logical]
      return;
    }
    auto [out_size, out_stride] = out_it->second;
    NVF_ERROR(
        out_size % factor == 0,
        "The logical domain and allocation domain of fusion input/output ",
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
      NVF_THROW("Unsupported transormation in allocation domain");
    }
  }

 public:
  BackwardTraverseFromLogicalToAlloc(
      ExpressionEvaluator& ee,
      std::unordered_map<IterDomain*, std::pair<int64_t, int64_t>>& active_ids)
      : ee_(ee), active_ids_(active_ids) {}

  void run(
      TensorView* tv,
      const std::vector<IterDomain*>& logical,
      const std::vector<IterDomain*>& alloc) {
    FUSER_PERF_SCOPE("BackwardTraverseFromLogicalToAlloc::run");
    auto backward_exprs = StmtSort::getExprsBetween(
        {alloc.begin(), alloc.end()}, {logical.begin(), logical.end()});
    std::reverse(backward_exprs.begin(), backward_exprs.end());
    for (auto expr : backward_exprs) {
      handle(expr);
    }
  }
};

void validateAllocationSizesAndStrides(
    const std::vector<IterDomain*>& alloc_dom,
    const std::vector<std::optional<bool>>& contiguity,
    c10::IntArrayRef sizes,
    c10::IntArrayRef strides) {
  NVF_ERROR(alloc_dom.size() == contiguity.size());
  checkAllEqual(
      {TensorDomain::noReductions(alloc_dom).size(),
       sizes.size(),
       strides.size()});

  int64_t expected_stride_if_contiguous = 1;
  auto dim_index = static_cast<int64_t>(sizes.size());
  // Go backwards because it's easier to compute the expected stride this way.
  for (auto domain_index = static_cast<int64_t>(alloc_dom.size()) - 1;
       domain_index >= 0;
       domain_index--) {
    IterDomain* alloc_id = alloc_dom[domain_index];
    if (alloc_id->isReduction()) {
      continue;
    }

    dim_index--;
    auto size = sizes.at(dim_index);
    auto stride = strides.at(dim_index);

    if (alloc_id->isBroadcast()) {
      NVF_CHECK(!contiguity[domain_index].has_value());
      if (alloc_id->hasExpandedExtent()) {
        NVF_CHECK(
            stride == 0,
            "Expecting an expanded dimension on dimension ",
            dim_index,
            " but found stride ",
            stride);
      }
      continue;
    }

    if (alloc_id->isDeviceDim()) {
      NVF_CHECK(size == 1);
      continue;
    }

    NVF_CHECK(contiguity[domain_index].has_value());
    if (*contiguity[domain_index]) {
      NVF_CHECK(
          stride == expected_stride_if_contiguous,
          "Stride mismatch with contiguity info. ",
          " allocation domain: ",
          ir_utils::toString(alloc_dom),
          ": sizes: ",
          sizes,
          ": strides: ",
          strides,
          "; contiguity: ",
          toDelimitedString(contiguity),
          "; dim: ",
          domain_index,
          "; expected stride: ",
          expected_stride_if_contiguous,
          "; actual stride: ",
          stride);
    }
    expected_stride_if_contiguous = stride * size;
  }
}

} // namespace

std::pair<std::vector<int64_t>, std::vector<int64_t>>
inferAndValidateAllocationSizesAndStrides(
    const at::Tensor& tensor,
    TensorView* tv,
    ExpressionEvaluator ee) {
  const auto& logical = tv->getLogicalDomain();
  const auto& alloc = tv->getMaybeAllocationDomain();

  // active IDs and their shape and stride
  std::vector<int64_t> logical_sizes = unshardedSizes(tv, tensor.sizes());
  std::unordered_map<IterDomain*, std::pair<int64_t, int64_t>> active_ids;
  int64_t dim_index = 0;
  for (IterDomain* id : TensorDomain::noReductions(logical)) {
    active_ids[id] = {logical_sizes.at(dim_index), tensor.stride(dim_index)};
    dim_index++;
  }
  NVF_ERROR(dim_index == tensor.dim());

  ForwardTraverseFromLogicalToAlloc(ee, active_ids).run(tv, logical, alloc);
  BackwardTraverseFromLogicalToAlloc(ee, active_ids).run(tv, logical, alloc);

  // Now active_ids should contain the final sizes and strides, unordered. We
  // need to put them to the correct order.
  std::vector<int64_t> allocation_sizes;
  std::vector<int64_t> allocation_strides;
  allocation_sizes.reserve(alloc.size());
  allocation_strides.reserve(alloc.size());
  for (IterDomain* id : TensorDomain::noReductions(alloc)) {
    if (id->isDeviceDim()) {
      allocation_sizes.push_back(1);
    } else {
      allocation_sizes.push_back(active_ids.at(id).first);
    }
    allocation_strides.push_back(active_ids.at(id).second);
  }

  // Only validate final sizes and strides when we have a non-empty tensor.
  if (tensor.numel() != 0) {
    validateAllocationSizesAndStrides(
        alloc, tv->getContiguity(), allocation_sizes, allocation_strides);
  }
  return {std::move(allocation_sizes), std::move(allocation_strides)};
}

std::vector<PolymorphicValue> GetMetaData::evaluate(
    const ExpressionEvaluator& ee,
    const std::vector<PolymorphicValue>& inputs) const {
  NVF_ERROR(inputs.size() == 1, "GetMetaData expects 1 input");
  NVF_ERROR(
      in()->isA<TensorView>(),
      "Currently, GetMetaData only supports TensorView");
  auto* tv = in()->as<TensorView>();

  const auto& input = inputs[0].as<at::Tensor>();

  NVF_ERROR(
      input.is_cuda() || input.is_meta(),
      "GetMetaData expects a CUDA/meta tensor as input, but got: ",
      input);

  std::shared_ptr<Struct> struct_ = std::make_shared<TensorMetaData>();
  TensorMetaData* metadata = (TensorMetaData*)struct_.get();
  metadata->dtype =
      std::get<PrimDataType>(aten_to_data_type(input.scalar_type()).type);
  metadata->data = input.data_ptr();

  if (isSharded(tv)) {
    std::vector<int64_t> unsharded_sizes = unshardedSizes(tv, input.sizes());
    metadata->logical_size_data = std::move(unsharded_sizes);
    metadata->logical_size = c10::makeArrayRef(metadata->logical_size_data);
  } else {
    metadata->logical_size = input.sizes();
  }
  metadata->logical_stride_data =
      std::vector<int64_t>(input.strides().begin(), input.strides().end());
  metadata->logical_stride = c10::makeArrayRef(metadata->logical_stride_data);

  auto [allocation_sizes, allocation_strides] =
      inferAndValidateAllocationSizesAndStrides(input, tv, ee);
  metadata->alloc_size_data = std::move(allocation_sizes);
  metadata->alloc_size = c10::makeArrayRef(metadata->alloc_size_data);
  metadata->alloc_stride_data = std::move(allocation_strides);
  metadata->alloc_stride = c10::makeArrayRef(metadata->alloc_stride_data);
  return {PolymorphicValue(std::move(struct_))};
}

} // namespace nvfuser
