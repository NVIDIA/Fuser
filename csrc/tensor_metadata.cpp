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

// Forward traverse from logical domain to allocation domain, compute frontier
// sizes and strides, validate that splits are divisible and merges are
// contiguous, and update active_ids_ correspondingly.
class ForwardSizesStridesTraverse {
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
  ForwardSizesStridesTraverse(
      ExpressionEvaluator& ee,
      std::unordered_map<IterDomain*, std::pair<int64_t, int64_t>>& active_ids)
      : ee_(ee), active_ids_(active_ids) {}

  void run(
      const std::vector<IterDomain*>& logical,
      const std::vector<IterDomain*>& alloc) {
    auto forward_exprs = StmtSort::getExprsBetween(
        {logical.begin(), logical.end()}, {alloc.begin(), alloc.end()});
    for (auto expr : forward_exprs) {
      handle(expr);
    }
  }
};

// Similar to ForwardSizesStridesTraverse, but in the opposite direction.
class BackwardSizesStridesTraverse {
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
  BackwardSizesStridesTraverse(
      ExpressionEvaluator& ee,
      std::unordered_map<IterDomain*, std::pair<int64_t, int64_t>>& active_ids)
      : ee_(ee), active_ids_(active_ids) {}

  void run(
      const std::vector<IterDomain*>& logical,
      const std::vector<IterDomain*>& alloc) {
    auto backward_exprs = StmtSort::getExprsBetween(
        {alloc.begin(), alloc.end()}, {logical.begin(), logical.end()});
    std::reverse(backward_exprs.begin(), backward_exprs.end());
    for (auto expr : backward_exprs) {
      handle(expr);
    }
  }
};

struct IDProperties {
  Val* size;
  Val* stride;
  bool contiguity;
};

struct TensorSizeStrideReturn {
  std::vector<IDProperties> id_properties;
  std::vector<Val*> validation_statements;
};

class ID_Dispatch {
 public:
  ID_Dispatch(
      std::vector<IterDomain*> from_domain,
      std::vector<Val*> from_strides = {},
      std::vector<bool> contiguity = {});

  class ForwardDispatch_ : public OptInDispatch {
   public:
    using OptInDispatch::dispatch;
    ID_Dispatch* id_dispatch_;
    ForwardDispatch_(ID_Dispatch* id_dispatch);
    void handle(Split* split);
    void handle(Merge* merge);
  };

  class BackwardDispatch_ : public OptInDispatch {
   public:
    using OptInDispatch::dispatch;
    ID_Dispatch* id_dispatch_;
    BackwardDispatch_(ID_Dispatch* id_dispatch);
    void handle(Split* split) override;
    void handle(Merge* merge) override;
  };

  std::vector<std::pair<IterDomain*, IDProperties>> transform(
      std::vector<IterDomain*> to_domain);

 public:
  // Grab entry in active_ids_ associated with target, error if not found.
  IDProperties findIdProperties(IterDomain* target) const;

  // At position in active_ids_ target is found, remove target and add
  // new_entries in order in that location
  void replaceAndInsert(
      IterDomain* target,
      std::vector<std::pair<IterDomain*, IDProperties>> new_entries);

  // Remove the entry in active_ids_ associated with target
  void remove(IterDomain* target);

  ForwardDispatch_ forward_dispatch_;
  BackwardDispatch_ backward_dispatch_;

  // This needs to be a std::vector and we need to keep track of placement,
  // because contiguity is relative to the original ordering, permutations of
  // that ordering (ignoring broadcast dims) change contiguity of the
  // projection.
  std::vector<std::pair<IterDomain*, IDProperties>> active_ids_;
  // The function may create Val* scalars for from_domain_'s strides, so store
  // it if it needs to be accessed.
  std::vector<std::pair<IterDomain*, IDProperties>> from_information_;
  std::vector<IterDomain*> from_domain_;
  std::vector<Val*> validation_stmts_;
};

ID_Dispatch::ID_Dispatch(
    std::vector<IterDomain*> from_domain,
    std::vector<Val*> from_strides,
    std::vector<bool> contiguity)
    : forward_dispatch_(this),
      backward_dispatch_(this),
      from_domain_(from_domain) {
  NVF_ERROR(!from_domain_.empty());
  FusionGuard fg(from_domain_[0]->fusion());
  from_domain_ = TensorDomain::noReductions(from_domain_);

  // Set contiguity to false if missing
  if (contiguity.empty()) {
    contiguity = std::vector<bool>(from_domain_.size(), false);
  }
  NVF_ERROR(from_domain_.size() == contiguity.size());
  // Setup symbolic strides based on contiguity if not provided
  if (from_strides.empty()) {
    from_strides = std::vector<Val*>(from_domain_.size(), nullptr);
    auto stride = FusionGuard::getCurFusion()->oneVal();
    for (int dim_i = (int)from_domain_.size() - 1; dim_i >= 0; dim_i--) {
      if (contiguity[dim_i]) {
        from_strides[dim_i] = stride;
        stride = IrBuilder::mulExpr(
            from_domain_[dim_i]->hasExpandedExtent()
                ? from_domain_[dim_i]->expandedExtent()
                : from_domain_[dim_i]->extent(),
            stride);
      } else if (from_domain_[dim_i]->isBroadcast()) {
        from_strides[dim_i] = FusionGuard::getCurFusion()->zeroVal();
      } else {
        stride = IrBuilder::create<Val>(DataType::Int);
        from_strides[dim_i] = stride;
      }
    }
  }
  NVF_ERROR(from_domain_.size() == from_strides.size());

  active_ids_.reserve(from_domain_.size());
  for (auto dim_i : c10::irange(from_domain_.size())) {
    active_ids_.push_back(
        {from_domain_[dim_i],
         IDProperties{
             from_domain_[dim_i]->getMaybeExpandedExtent(),
             from_strides.at(dim_i),
             contiguity.at(dim_i)}});
  }
  from_information_ = active_ids_;
}

ID_Dispatch::ForwardDispatch_::ForwardDispatch_(ID_Dispatch* id_dispatch)
    : id_dispatch_(id_dispatch) {}

void ID_Dispatch::ForwardDispatch_::handle(Split* split) {
  auto in_properties = id_dispatch_->findIdProperties(split->in());
  auto factor = split->factor();

  id_dispatch_->validation_stmts_.push_back(IrBuilder::eqExpr(
      FusionGuard::getCurFusion()->zeroVal(),
      IrBuilder::modExpr(in_properties.size, factor)));

  auto other = IrBuilder::divExpr(in_properties.size, factor);

  Val* inner_size = split->innerSplit() ? factor : other;
  Val* outer_size = split->innerSplit() ? other : factor;

  IDProperties inner_properties = {
      inner_size, in_properties.stride, in_properties.contiguity};
  IDProperties outer_properties = {
      outer_size,
      IrBuilder::mulExpr(in_properties.stride, inner_size),
      in_properties.contiguity};

  id_dispatch_->replaceAndInsert(
      split->in(),
      {{split->outer(), outer_properties}, {split->inner(), inner_properties}});
}

void ID_Dispatch::ForwardDispatch_::handle(Merge* merge) {
  auto inner_properties = id_dispatch_->findIdProperties(merge->inner());
  auto outer_properties = id_dispatch_->findIdProperties(merge->outer());
  id_dispatch_->validation_stmts_.push_back(IrBuilder::eqExpr(
      IrBuilder::mulExpr(inner_properties.size, inner_properties.stride),
      outer_properties.stride));
  id_dispatch_->replaceAndInsert(
      merge->inner(),
      {{merge->out(),
        {IrBuilder::mulExpr(inner_properties.size, outer_properties.size),
         inner_properties.stride,
         inner_properties.contiguity}}});
  id_dispatch_->remove(merge->outer());
}

ID_Dispatch::BackwardDispatch_::BackwardDispatch_(ID_Dispatch* id_dispatch)
    : id_dispatch_(id_dispatch) {}

void ID_Dispatch::BackwardDispatch_::handle(Split* split) {
  auto inner_properties = id_dispatch_->findIdProperties(split->inner());
  auto outer_properties = id_dispatch_->findIdProperties(split->outer());
  id_dispatch_->validation_stmts_.push_back(IrBuilder::eqExpr(
      IrBuilder::mulExpr(inner_properties.size, inner_properties.stride),
      outer_properties.stride));
  id_dispatch_->replaceAndInsert(
      split->inner(),
      {{split->in(),
        {IrBuilder::mulExpr(inner_properties.size, outer_properties.size),
         inner_properties.stride,
         inner_properties.contiguity}}});
  id_dispatch_->remove(split->outer());
}

void ID_Dispatch::BackwardDispatch_::handle(Merge* merge) {
  auto out_properties = id_dispatch_->findIdProperties(merge->out());
  // TODO: How should expanded extents be handled here?
  auto factor = merge->inner()->extent();

  id_dispatch_->validation_stmts_.push_back(IrBuilder::eqExpr(
      FusionGuard::getCurFusion()->zeroVal(),
      IrBuilder::modExpr(out_properties.size, factor)));

  // This impl should be safe, but could be specialized in case the outer
  // dimension has a const int

  IDProperties outer_properties = {
      IrBuilder::divExpr(out_properties.size, factor),
      IrBuilder::mulExpr(out_properties.size, factor),
      out_properties.contiguity};
  IDProperties inner_properties = {
      factor, out_properties.stride, out_properties.contiguity};

  id_dispatch_->replaceAndInsert(
      merge->out(),
      {{merge->outer(), outer_properties}, {merge->inner(), inner_properties}});
}

std::vector<std::pair<IterDomain*, IDProperties>> ID_Dispatch::transform(
    std::vector<IterDomain*> to_domain) {
  FusionGuard fg(from_domain_[0]->fusion());
  to_domain = TensorDomain::noReductions(to_domain);
  auto path_pair = getExprsBetween<IRBFS>(
      {from_domain_.begin(), from_domain_.end()},
      {to_domain.begin(), to_domain.end()});
  NVF_ERROR(path_pair.second, "Did not path between provided domains.");
  auto bfs_exprs = path_pair.first;
  for (auto bfs_expr : bfs_exprs) {
    auto expr = bfs_expr.first;
    auto direction = bfs_expr.second;
    if (direction == Direction::Forward) {
      forward_dispatch_.dispatch(expr);
    } else if (direction == Direction::Backward) {
      backward_dispatch_.dispatch(expr);
    }
    NVF_ERROR(
        direction != Direction::Undefined, "Error traversing provided domain");
  }

  std::vector<std::pair<IterDomain*, IDProperties>> to_information;
  for (auto to_id : to_domain) {
    to_information.push_back({to_id, findIdProperties(to_id)});
  }
  return to_information;
}

IDProperties ID_Dispatch::findIdProperties(IterDomain* target) const {
  auto it = std::find_if(
      active_ids_.begin(), active_ids_.end(), [target](const auto& pair) {
        return pair.first == target;
      });
  NVF_ERROR(it != active_ids_.end());
  return it->second;
}

void ID_Dispatch::replaceAndInsert(
    IterDomain* target,
    std::vector<std::pair<IterDomain*, IDProperties>> new_entries) {
  auto it = std::find_if(
      active_ids_.begin(), active_ids_.end(), [target](const auto& pair) {
        return pair.first == target;
      });

  NVF_ERROR(it != active_ids_.end());
  NVF_ERROR(!new_entries.empty());

  const size_t pos = std::distance(active_ids_.begin(), it);
  active_ids_.erase(it);

  // Move-insert new entries at original position
  active_ids_.insert(
      active_ids_.begin() + pos,
      std::make_move_iterator(new_entries.begin()),
      std::make_move_iterator(new_entries.end()));
}

void ID_Dispatch::remove(IterDomain* target) {
  auto it = std::find_if(
      active_ids_.begin(), active_ids_.end(), [target](const auto& pair) {
        return pair.first == target;
      });

  NVF_ERROR(it != active_ids_.end());
  active_ids_.erase(it);
}

} // namespace

void validateContiguity(
    const std::vector<IterDomain*>& domain,
    const std::vector<std::optional<bool>>& contiguity,
    c10::IntArrayRef sizes,
    c10::IntArrayRef strides) {
  NVF_ERROR(domain.size() == contiguity.size());
  checkAllEqual(
      {TensorDomain::noReductions(domain).size(),
       sizes.size(),
       strides.size()});

  int64_t expected_stride_if_contiguous = 1;
  auto dim_index = static_cast<int64_t>(sizes.size());
  // Go backwards because it's easier to compute the expected stride this way.
  for (auto domain_index = static_cast<int64_t>(domain.size()) - 1;
       domain_index >= 0;
       domain_index--) {
    IterDomain* alloc_id = domain[domain_index];
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
          ir_utils::toString(domain),
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

  ForwardSizesStridesTraverse(ee, active_ids).run(logical, alloc);
  BackwardSizesStridesTraverse(ee, active_ids).run(logical, alloc);

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
    validateContiguity(
        alloc, tv->getContiguity(), allocation_sizes, allocation_strides);
  }
  return {std::move(allocation_sizes), std::move(allocation_strides)};
}

std::pair<std::vector<int64_t>, std::vector<int64_t>> inferAndValidateProjection(
    std::vector<int64_t> from_sizes,
    std::vector<int64_t> from_strides,
    std::vector<IterDomain*> from_domain,
    std::vector<IterDomain*> to_domain,
    ExpressionEvaluator ee) {
  from_domain = TensorDomain::noReductions(from_domain);
  to_domain = TensorDomain::noReductions(to_domain);

  NVF_ERROR(
      from_sizes.size() == from_domain.size(),
      "Error projecting tensor sizes, from domain doesn't match tensor dims: ",
      from_sizes.size(),
      " vs ",
      from_domain.size(),
      ".");
  // active IDs and their shape and stride
  std::unordered_map<IterDomain*, std::pair<int64_t, int64_t>> active_ids;
  for (auto dim_i : c10::irange(from_domain.size())) {
    active_ids[from_domain[dim_i]] = {
        from_sizes.at(dim_i), from_strides.at(dim_i)};
  }

  ForwardSizesStridesTraverse(ee, active_ids).run(from_domain, to_domain);
  BackwardSizesStridesTraverse(ee, active_ids).run(from_domain, to_domain);

  // Now active_ids should contain the final sizes and strides, unordered. We
  // need to put them to the correct order.
  std::vector<int64_t> to_sizes;
  std::vector<int64_t> to_strides;
  to_sizes.reserve(to_domain.size());
  to_strides.reserve(to_domain.size());
  for (IterDomain* id : TensorDomain::noReductions(to_domain)) {
    if (id->isDeviceDim()) {
      to_sizes.push_back(1);
      to_strides.push_back(0);
    } else {
      to_sizes.push_back(active_ids.at(id).first);
      to_strides.push_back(active_ids.at(id).second);
    }
  }

  ID_Dispatch dispatch(from_domain);
  auto from_information = dispatch.from_information_;
  auto to_information = dispatch.transform(to_domain);
  NVF_ERROR(dispatch.from_information_.size() == from_domain.size());
  for (auto id_i : c10::irange(from_domain.size())) {
    auto stride = from_information[id_i].second.stride;
    if (!ee.isKnown(stride)) {
      ee.bind(stride, from_strides[id_i]);
    }
  }

  std::vector<int64_t> to_sizes_2;
  std::vector<int64_t> to_strides_2;
  for (auto id_i : c10::irange(to_domain.size())) {
    auto size = to_information[id_i].second.size;
    int64_t size_int = ee.evaluate(size).as<int64_t>();
    auto stride = to_information[id_i].second.stride;
    int64_t stride_int = ee.evaluate(stride).as<int64_t>();
    to_sizes_2.push_back(size_int);
    to_strides_2.push_back(stride_int);
  }

  NVF_ERROR(
      to_sizes == to_sizes_2 && to_strides == to_strides_2,
      "\nsizes{",
      to_sizes,
      "}\nsizes2 {",
      to_sizes_2,
      "}\nstrides{",
      to_strides_2,
      "}\nstrides2{",
      to_strides_2,
      "}");

  return {std::move(to_sizes), std::move(to_strides)};
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
  metadata->logical_stride = input.strides();

  auto [allocation_sizes, allocation_strides] =
      inferAndValidateAllocationSizesAndStrides(input, tv, ee);
  metadata->alloc_size_data = std::move(allocation_sizes);
  metadata->alloc_size = c10::makeArrayRef(metadata->alloc_size_data);
  metadata->alloc_stride_data = std::move(allocation_strides);
  metadata->alloc_stride = c10::makeArrayRef(metadata->alloc_stride_data);
  return {PolymorphicValue(std::move(struct_))};
}

} // namespace nvfuser
