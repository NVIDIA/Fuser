// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#include <ir/builder.h>
#include <ir/utils.h>
#include <ops/alias.h>
#include <ops/arith.h>
#include <ops/utils.h>
#include <transform_view.h>
#include <type_promotion.h>

namespace nvfuser {

Val* set(Val* v) {
  Val* out = ops::newValLike(v, v->getDataType().value());
  IrBuilder::create<LoadStoreOp>(LoadStoreOpType::Set, out, v);
  return out;
}

TensorView* set(TensorView* tv) {
  return set(tv->as<Val>())->as<TensorView>();
}

Val* segment_set(Val* v) {
  Val* out = ops::newValLike(v, v->getDataType().value());
  IrBuilder::create<LoadStoreOp>(LoadStoreOpType::SegmenterSet, out, v);
  return out;
}

TensorView* segment_set(TensorView* tv) {
  return segment_set(tv->as<Val>())->as<TensorView>();
}

TensorView* view(TensorView* x, DataType dtype) {
  TORCH_INTERNAL_ASSERT(x != nullptr, "Input is invalid.");
  if (x->getDataType() == dtype) {
    return x;
  }

  auto input_type = x->getDataType().value();
  auto input_size = dataTypeSize(input_type);
  auto newsize = dataTypeSize(dtype);

  if (input_size == newsize) {
    return bitCastOp(dtype, x);
  }
  // TODO: support view(dtype) for dtypes where input_size != newsize
  TORCH_INTERNAL_ASSERT(false, "Unsupported reinterpret casting view");
}

TensorView* reshape(
    TensorView* x,
    const std::vector<int64_t>& original_sizes,
    const std::vector<int64_t>& new_sizes) {
  TORCH_INTERNAL_ASSERT(x != nullptr, "Input is invalid.");
  TORCH_INTERNAL_ASSERT(
      TensorDomain::noReductions(x->getMaybeRFactorDomain()).size() ==
      original_sizes.size());

  auto view_analysis = analyzeView(x, original_sizes, new_sizes);

  return reshape(x, view_analysis);
}

namespace {

// Check if a dynamic reshape is actually static. Returns a reshaped
// tensor if static. Nullptr if not.
TensorView* tryStaticReshape(
    TensorView* inp_tv,
    const std::vector<IterDomain*>& inp_dom,
    const std::vector<Val*>& new_sizes) {
  std::vector<int64_t> inp_sizes(inp_dom.size());
  for (const auto i : c10::irange(inp_dom.size())) {
    auto id = inp_dom.at(i);
    auto id_size = id->extent()->getInt();
    if (!id_size.has_value()) {
      return nullptr;
    }
    inp_sizes.at(i) = id_size.value();
  }

  std::vector<int64_t> out_sizes(new_sizes.size());
  for (const auto i : c10::irange(new_sizes.size())) {
    auto id_size = new_sizes.at(i)->getInt();
    if (!id_size.has_value()) {
      return nullptr;
    }
    out_sizes.at(i) = id_size.value();
  }

  // Both inputs are outputs are static. Just use the static version
  // of reshape
  return reshape(inp_tv, inp_sizes, out_sizes);
}

} // namespace

TensorView* reshape(TensorView* inp_tv, const std::vector<Val*>& new_sizes) {
  auto inp_dom = TensorDomain::noReductions(inp_tv->getMaybeRFactorDomain());

  TORCH_CHECK(
      std::none_of(
          inp_dom.begin(),
          inp_dom.end(),
          [](auto inp_id) { return inp_id->maybePartial(); }),
      "Unsupported input tensor to reshape as its axes may be partial: ",
      inp_tv->toString());

  auto static_reshape_output = tryStaticReshape(inp_tv, inp_dom, new_sizes);
  if (static_reshape_output) {
    return static_reshape_output;
  }

  auto root_domain = ops::newOutputDomain({inp_tv}, inp_tv->dtype());

  // Create placeholder rfactor domain. Note it's not connected with the root
  // domain.
  std::vector<IterDomain*> rfactor_domain(new_sizes.size(), nullptr);
  for (const auto i : c10::irange(new_sizes.size())) {
    auto rf_id = IterDomainBuilder(
                     FusionGuard::getCurFusion()->zeroVal(), new_sizes.at(i))
                     .iter_type(IterType::Symbolic)
                     .is_rfactor_domain(true)
                     .build();
    rfactor_domain.at(i) = rf_id;
  }

  auto out_tv = IrBuilder::create<TensorView>(
      IrBuilder::create<TensorDomain>(
          root_domain,
          rfactor_domain,
          rfactor_domain,
          TensorDomain::getContiguityFilledWith(rfactor_domain, true)),
      inp_tv->dtype());

  IrBuilder::create<ViewOp>(inp_tv->container(), out_tv, inp_tv);

  return out_tv;
}

TensorView* flatten(TensorView* x, int64_t start_dim, int64_t end_dim) {
  TORCH_INTERNAL_ASSERT(x != nullptr, "Input is invalid.");
  auto inp_domain = TensorDomain::noReductions(x->getMaybeRFactorDomain());
  if (start_dim < 0) {
    start_dim += (int64_t)inp_domain.size();
  }
  if (end_dim < 0) {
    end_dim += (int64_t)inp_domain.size();
  }
  TORCH_CHECK(
      start_dim >= 0 && start_dim < (int64_t)inp_domain.size(),
      "Invalid start_dim ",
      start_dim);
  TORCH_CHECK(
      end_dim >= 0 && end_dim < (int64_t)inp_domain.size(),
      "Invalid end_dim ",
      end_dim);
  TORCH_CHECK(start_dim <= end_dim, "start_dim must be <= end_dim");

  if (start_dim == end_dim) {
    return x;
  }

  auto out = IrBuilder::create<TensorView>(
      x->container(),
      x->domain()->flatten(start_dim, end_dim),
      x->getDataType().value());

  IrBuilder::create<ViewOp>(out, x);
  return out;
}

TensorView* squeeze(TensorView* x, const std::vector<bool>& to_squeeze) {
  TORCH_INTERNAL_ASSERT(x != nullptr, "Input is invalid.");
  auto x_dom = x->domain()->noReductions();
  const auto ndims = static_cast<int>(x_dom.size());

  TORCH_INTERNAL_ASSERT(
      ndims == (int)to_squeeze.size(),
      "Invalid to_squeeze for squeeze: ",
      to_squeeze,
      ". Input tensor: ",
      x->toString());

  std::vector<IterDomain*> out_domain;
  for (const auto idx : c10::irange(ndims)) {
    auto id = x_dom[idx];
    if (to_squeeze[idx]) {
      TORCH_CHECK(
          id->isBroadcast(), "Can not squeeze non-broadcasting dimension(s).");
      TORCH_CHECK(
          !id->hasExpandedExtent(), "Can not squeeze expanded dimension(s).");
      TORCH_CHECK(
          id->extent()->isOneInt(),
          "Can not squeeze dimension(s) with size != 1.");
    } else {
      out_domain.push_back(id->cloneWithoutRFactor());
    }
  }

  auto out = IrBuilder::create<TensorView>(
      IrBuilder::create<TensorDomain>(
          out_domain, TensorDomain::getContiguityFilledWith(out_domain, true)),
      *x->getDataType());

  IrBuilder::create<SqueezeOp>(x->container(), out, x, to_squeeze);

  return out;
}

TensorView* squeeze(TensorView* x, const std::vector<int64_t>& sizes) {
  TORCH_INTERNAL_ASSERT(x != nullptr, "Input is invalid.");
  const auto ndims = static_cast<int>(x->domain()->noReductions().size());

  TORCH_INTERNAL_ASSERT(
      ndims == int(sizes.size()),
      "Invalid sizes for squeeze: ",
      sizes,
      ". Input tensor: ",
      x->toString());

  std::vector<bool> to_squeeze(ndims);
  for (const auto idx : c10::irange(sizes.size())) {
    to_squeeze[idx] = (sizes[idx] == 1);
  }
  return squeeze(x, to_squeeze);
}

TensorView* squeeze(TensorView* x, const std::vector<int64_t>& sizes, int dim) {
  TORCH_INTERNAL_ASSERT(x != nullptr, "Input is invalid.");
  const auto ndims = static_cast<int>(x->domain()->noReductions().size());

  TORCH_INTERNAL_ASSERT(
      ndims == int(sizes.size()),
      "Invalid sizes for squeeze: ",
      sizes,
      ". Input tensor: ",
      x->toString());

  if (dim < 0) {
    dim = ndims + dim;
  }

  TORCH_INTERNAL_ASSERT(
      dim >= 0 && dim < ndims,
      "Invalid position to squeeze: ",
      dim,
      ". Input tensor: ",
      x->toString());

  if (sizes[dim] == 1) {
    std::vector<bool> to_squeeze(ndims, false);
    to_squeeze[dim] = true;
    return squeeze(x, to_squeeze);
  } else {
    return set(x);
  }
}

TensorView* squeeze(
    TensorView* x,
    const std::vector<int64_t>& sizes,
    const std::vector<int64_t>& dims) {
  TORCH_INTERNAL_ASSERT(x != nullptr, "Input is invalid.");
  const auto ndims = static_cast<int>(x->domain()->noReductions().size());

  TORCH_INTERNAL_ASSERT(
      ndims == int(sizes.size()),
      "Invalid sizes for squeeze: ",
      sizes,
      ". Input tensor: ",
      x->toString());

  bool is_all_singleton_dimensions = true;

  std::vector<bool> to_squeeze(ndims);
  for (auto dim : dims) {
    if (dim < 0) {
      dim = ndims + dim;
    }

    TORCH_INTERNAL_ASSERT(
        dim >= 0 && dim < ndims,
        "Invalid position to squeeze: ",
        dim,
        ". Input tensor: ",
        x->toString());

    bool is_singleton_dim = (sizes[dim] == 1);
    to_squeeze.at(dim) = is_singleton_dim;
    is_all_singleton_dimensions &= is_singleton_dim;
  }

  if (is_all_singleton_dimensions) {
    return squeeze(x, to_squeeze);
  } else {
    return set(x);
  }
}

TensorView* unsqueeze(TensorView* x, int dim) {
  TORCH_INTERNAL_ASSERT(x != nullptr, "Input is invalid.");
  const auto ndims = static_cast<int>(x->domain()->noReductions().size());

  if (dim < 0) {
    dim = ndims + dim + 1;
  }

  TORCH_INTERNAL_ASSERT(
      dim >= 0 && dim <= ndims,
      "Invalid position to unsqueeze: ",
      dim,
      ". Input tensor: ",
      x->toString());

  std::vector<bool> broadcast_axes(ndims + 1, false);
  broadcast_axes[dim] = true;
  return broadcast(x, broadcast_axes);
}

TensorView* permute(TensorView* x, const std::vector<int64_t>& new2old) {
  TORCH_INTERNAL_ASSERT(x != nullptr, "Input is invalid.");
  if (new2old.empty()) {
    return set(x);
  }
  auto inp_domain = TensorDomain::noReductions(x->getMaybeRFactorDomain());

  TORCH_CHECK(
      inp_domain.size() == new2old.size(),
      "The number of dimensions in the tensor input does not match the length",
      " of the desired ordering of dimensions i.e. input.dim() = ",
      inp_domain.size(),
      " is not equal to len(dims) = ",
      new2old.size());

  // Return scalar tensors immediately
  if (inp_domain.empty()) {
    return set(x);
  }

  auto normalized_new2old =
      ir_utils::normalizeNew2Old(new2old, inp_domain.size());

  std::vector<IterDomain*> out_root;
  out_root.reserve(inp_domain.size());
  for (const auto id : inp_domain) {
    out_root.emplace_back(id->cloneWithoutRFactor());
  }

  std::vector<IterDomain*> out_rfactor;
  out_rfactor.reserve(inp_domain.size());
  for (const auto i : normalized_new2old) {
    out_rfactor.emplace_back(out_root.at(i));
  }

  TensorView* out_tensor = IrBuilder::create<TensorView>(
      IrBuilder::create<TensorDomain>(
          out_root,
          out_rfactor,
          out_rfactor,
          TensorDomain::getContiguityFilledWith(out_rfactor, true)),
      x->getDataType().value());
  IrBuilder::create<LoadStoreOp>(LoadStoreOpType::Set, out_tensor, x);
  return out_tensor;
}

TensorView* transpose(TensorView* x, int64_t dim0, int64_t dim1) {
  TORCH_INTERNAL_ASSERT(x != nullptr, "Input is invalid.");
  const auto ndims = static_cast<int>(x->domain()->noReductions().size());

  if (dim0 < 0) {
    dim0 = ndims + dim0;
  }

  if (dim1 < 0) {
    dim1 = ndims + dim1;
  }

  TORCH_CHECK(
      dim0 >= 0 && dim0 <= ndims, "Invalid transpose dimension 0: ", dim0);

  TORCH_CHECK(
      dim1 >= 0 && dim1 <= ndims, "Invalid transpose dimension 1: ", dim1);

  std::vector<int64_t> new2old(ndims);
  for (const auto i : c10::irange(ndims)) {
    if (i == dim0) {
      new2old[i] = dim1;
    } else if (i == dim1) {
      new2old[i] = dim0;
    } else {
      new2old[i] = i;
    }
  }
  return permute(x, new2old);
}

TensorView* transpose(TensorView* x) {
  TORCH_INTERNAL_ASSERT(x != nullptr, "Input is invalid.");
  const auto ndims = static_cast<int>(x->domain()->noReductions().size());

  TORCH_CHECK(
      ndims <= 2,
      "Expected a tensor with <= 2 dimensions, but it has ",
      ndims,
      "D.");

  // short-circuit: return original tensorview if less than 2 dimensions
  if (ndims < 2) {
    return x;
  }

  return transpose(x, 0, 1);
}

// Padding widths are assumed to be non-negative. Currently there's no
// validation.
TensorView* pad(
    TensorView* inp,
    const std::vector<Val*>& pad_widths,
    Val* value) {
  DataType dt = inp->getDataType().value();
  if (!value) {
    // Create a zero of the appropriate type
    if (isComplexType(dt)) {
      value = static_cast<Val*>(IrBuilder::create<ComplexDouble>(0, dt));
    } else if (isFloatingPointType(dt)) {
      value = static_cast<Val*>(IrBuilder::create<Double>(0, dt));
    } else if (isBooleanType(dt)) {
      value = static_cast<Val*>(IrBuilder::create<Bool>(false, dt));
    } else {
      value = static_cast<Val*>(IrBuilder::create<Int>(0, dt));
    }
  }
  if (value->getDataType().value() != dt) {
    // Insert an explicit castOp if dtype of value does not match TensorView's
    value = castOp(dt, value);
  }
  const auto inp_dom = TensorDomain::noReductions(inp->getMaybeRFactorDomain());
  const auto ndims = inp_dom.size();

  TORCH_CHECK(
      pad_widths.size() % 2 == 0 && pad_widths.size() / 2 <= ndims,
      "Invalid number of padding widths: ",
      pad_widths.size());

  const auto num_padded_dims = pad_widths.size() / 2;
  const auto num_non_padded_dims = ndims - num_padded_dims;

  std::vector<IterDomain*> root_ids(ndims);
  std::vector<IterDomain*> rfactor_ids(ndims);

  // PadOp requires pad widths for all dimensions, even for non-padded
  // ones.

  std::vector<Val*> normalized_pad_widths;

  // Fill zero for non padded dimensions
  for (const auto i : c10::irange(num_non_padded_dims)) {
    (void)i;
    normalized_pad_widths.push_back(FusionGuard::getCurFusion()->zeroVal());
    normalized_pad_widths.push_back(FusionGuard::getCurFusion()->zeroVal());
  }

  // torch.pad has padding widths of inner dimensions before outer
  // dimensions
  for (const auto i : c10::irange(num_padded_dims)) {
    auto left_pad = pad_widths.at(num_padded_dims * 2 - (i + 1) * 2);
    auto right_pad = pad_widths.at(num_padded_dims * 2 - (i + 1) * 2 + 1);
    normalized_pad_widths.push_back(left_pad);
    normalized_pad_widths.push_back(right_pad);
  }

  // Indicates if any dimension is actually padded. Can be false even
  // when non-empty padding width vector is passed
  bool is_padded_any = false;
  for (const auto idx : c10::irange(ndims)) {
    auto inp_root_id = inp_dom.at(idx);
    IterDomain* out_root_id = nullptr;
    IterDomain* out_rf_id = nullptr;
    auto left_pad = normalized_pad_widths.at(idx * 2);
    auto right_pad = normalized_pad_widths.at(idx * 2 + 1);
    if (idx < num_non_padded_dims ||
        (left_pad->isZeroInt() && right_pad->isZeroInt())) {
      out_root_id = inp_root_id->cloneWithoutRFactor();
      out_rf_id = out_root_id;
    } else {
      out_root_id =
          IterDomainBuilder(inp_root_id).is_rfactor_domain(true).build();
      // Expand the root domain and mark it as a rfactor domain
      out_rf_id = IterDomain::resize(out_root_id, left_pad, right_pad, true);
      is_padded_any = true;
    }
    root_ids.at(idx) = out_root_id;
    rfactor_ids.at(idx) = out_rf_id;
  }

  // If all of the padding widths are just zero, this is just a set op.
  if (!is_padded_any) {
    return set(inp);
  }

  auto out = IrBuilder::create<TensorView>(
      IrBuilder::create<TensorDomain>(
          root_ids,
          rfactor_ids,
          rfactor_ids,
          TensorDomain::getContiguityFilledWith(rfactor_ids, true)),
      *inp->getDataType());

  IrBuilder::create<PadOp>(out, inp, normalized_pad_widths, value);

  return out;
}

// cat is implemented as PadOp and CatOp. Padding is done first to
// account for the size difference between each of the inputs and the
// output. All of the inputs to CatOp have the same shape as the
// output shape.
TensorView* cat(const std::vector<TensorView*>& inputs, int64_t cat_dim) {
  TORCH_CHECK(!inputs.empty(), "No input tensor given");

  const auto dtype = inputs.at(0)->getDataType().value();

  std::vector<std::vector<IterDomain*>> inp_doms;
  int64_t ndims = -1;

  for (auto inp : inputs) {
    TORCH_CHECK(
        inp->getDataType().value() == dtype,
        "Can't concatenate tensors with different data types: ",
        dtype,
        ", ",
        inp->getDataType().value());
    inp_doms.emplace_back(
        TensorDomain::noReductions(inp->getMaybeRFactorDomain()));
    auto i_ndims = static_cast<int64_t>(inp_doms.back().size());
    if (ndims == -1) {
      ndims = i_ndims;
    } else {
      TORCH_CHECK(
          ndims == i_ndims,
          "Unexpected number of dimensions: ",
          inp->toString(),
          ", expected: ",
          ndims);
    }
  }

  if (cat_dim < 0) {
    cat_dim += ndims;
  }

  TORCH_CHECK(
      cat_dim >= 0 && cat_dim < ndims, "Invalid dimension to cat: ", cat_dim);

  // Special handling for the case where there's only one input
  if (inputs.size() == 1) {
    return set(inputs.at(0));
  }

  Val* concat_ext = nullptr;

  for (const auto i : c10::irange(inputs.size())) {
    auto input_dim_extent =
        inp_doms.at(i).at(cat_dim)->getMaybeExpandedExtent();
    concat_ext = SimplifyingIrBuilder::addExpr(concat_ext, input_dim_extent);
  }

  // For each of the input tensors, create a new rfactor tensor by
  // padding the concat dim. Padding is used here as it effectively
  // embeds the resizing information of the concat operation.

  Val* left_pad = FusionGuard::getCurFusion()->zeroVal();
  Val* right_pad = concat_ext;
  std::vector<Val*> resized_inputs(inputs.size());
  for (const auto input_idx : c10::irange(inputs.size())) {
    const auto& inp_dom = inp_doms.at(input_idx);
    std::vector<Val*> pad_widths(ndims * 2);
    for (const auto dim : c10::irange(ndims)) {
      auto inp_root_id = inp_dom.at(dim);
      Val* left_pad_i = nullptr;
      Val* right_pad_i = nullptr;
      if (dim != cat_dim) {
        left_pad_i = FusionGuard::getCurFusion()->zeroVal();
        right_pad_i = FusionGuard::getCurFusion()->zeroVal();
      } else {
        // Resize the root ID so that it has the same extent as the
        // concatenated ID. The expansion of both left and right sides
        // is done so that this input tensor is positioned in a way
        // that corresponds to the concatenated dimension. For
        // example, the first input should be at the
        // left-most position, so it is expanded only at the right side
        // with the expansion factor of
        // (total_concatenated_domain_extent -
        // extent_of_the_input_tensor). Similarly, the second tensor
        // is expanded by extent_of_the_input_tensor at its left side,
        // and by (total_concatenated_domain_extent -
        // extent_of_the_input_tensor - extent_of_the_second_tensor).
        //
        // TODO: what to do if inp_id is not a normal iterdomain, i.e.,
        // broadcast, partial, etc? For now, assume it's a normal
        // IterDomain.
        TORCH_INTERNAL_ASSERT(
            inp_root_id->getIterType() == IterType::Iteration &&
                !inp_root_id->maybePartial(),
            "Unsupported IterDomain to concatenate: ",
            inp_root_id->toString());
        // The right pad of the last tensor is just zero
        right_pad = input_idx < inputs.size() - 1
            ? sub(right_pad, inp_root_id->getMaybeExpandedExtent())
            : FusionGuard::getCurFusion()->zeroVal();
        left_pad_i = left_pad;
        right_pad_i = right_pad;
        left_pad = add(left_pad, inp_root_id->extent());
      }
      // The pad width argument to pad should be ordered such that the
      // widths of inner dimensions come first.
      pad_widths.at((ndims - dim - 1) * 2) = left_pad_i;
      pad_widths.at((ndims - dim - 1) * 2 + 1) = right_pad_i;
    }

    resized_inputs.at(input_idx) = pad(inputs.at(input_idx), pad_widths);
  }

  // Now all of resized_inputs have the same shape as the out tensor
  auto out = ops::newOutputTV(resized_inputs, dtype);

  IrBuilder::create<CatOp>(out, resized_inputs, cat_dim);

  return out;
}

// Currently there's no error check about  the actual values of the
// Slice parameters. For example, the start parameter of a range of a
// domain is assumed to be >= 0 and < the extent of the domain.
TensorView* slice(TensorView* inp, const std::vector<Slice>& ranges) {
  const auto inp_dom = TensorDomain::noReductions(inp->getMaybeRFactorDomain());
  const int ndims = static_cast<int>(inp_dom.size());

  TORCH_CHECK(
      ndims == static_cast<int>(ranges.size()),
      "The range vector must have the same number of Slice descriptors. Given: ",
      ranges.size(),
      ", Expected: ",
      ndims);

  auto normalize_slice_range = [](Slice range, Val* extent) -> Slice {
    if (range.start == nullptr) {
      range.start = FusionGuard::getCurFusion()->zeroVal();
    }
    if (range.stop == nullptr) {
      range.stop = extent;
    }
    if (range.step == nullptr) {
      range.step = FusionGuard::getCurFusion()->oneVal();
    }
    return range;
  };

  for (auto& range : ranges) {
    // Step not supported yet
    TORCH_CHECK(
        range.step == nullptr || range.step->isOneInt(),
        "Unsupported step: ",
        range.step->toString());
  }

  std::vector<IterDomain*> root_ids(ndims);
  std::vector<IterDomain*> rfactor_ids(ndims);
  std::vector<Slice> normalized_ranges(ndims);

  bool needs_real_slicing = false;
  for (const auto idx : c10::irange(ndims)) {
    auto inp_root_id = inp_dom[idx];
    auto range = normalize_slice_range(ranges.at(idx), inp_root_id->extent());
    normalized_ranges.at(idx) = range;
    IterDomain* out_root_id = nullptr;
    IterDomain* out_rf_id = nullptr;
    if (range.start->isZeroInt() && range.stop->sameAs(inp_root_id->extent()) &&
        range.step->isOneInt()) {
      // This dim doesn't need slicing
      out_root_id = inp_root_id->cloneWithoutRFactor();
      out_rf_id = out_root_id;
    } else {
      out_root_id =
          IterDomainBuilder(inp_root_id).is_rfactor_domain(true).build();
      out_rf_id = IterDomain::resize(
          out_root_id,
          SimplifyingIrBuilder::negExpr(range.start),
          sub(range.stop, inp_root_id->extent()),
          true);
      needs_real_slicing = true;
    }
    root_ids.at(idx) = out_root_id;
    rfactor_ids.at(idx) = out_rf_id;
  }

  // If slicing isn't actually needed, just return a copy
  if (!needs_real_slicing) {
    return set(inp);
  }

  auto out = IrBuilder::create<TensorView>(
      IrBuilder::create<TensorDomain>(
          root_ids,
          rfactor_ids,
          rfactor_ids,
          TensorDomain::getContiguityFilledWith(rfactor_ids, true)),
      *inp->getDataType());

  IrBuilder::create<SliceOp>(out, inp, normalized_ranges);
  return out;
}

} // namespace nvfuser
