// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#include <expr_evaluator.h>
#include <expr_simplifier.h>
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
  NVF_ERROR(x != nullptr, "Input is invalid.");
  if (x->getDataType() == dtype) {
    return x;
  }

  auto input_type = x->getDataType().value();
  auto input_size = dataTypeSizeByte(input_type);
  auto newsize = dataTypeSizeByte(dtype);

  if (input_size == newsize) {
    return bitCastOp(dtype, x);
  }
  // TODO: support view(dtype) for dtypes where input_size != newsize
  NVF_THROW("Unsupported reinterpret casting view");
}

TensorView* reshape(
    TensorView* x,
    const std::vector<int64_t>& original_sizes,
    const std::vector<int64_t>& new_sizes) {
  NVF_ERROR(x != nullptr, "Input is invalid.");
  NVF_ERROR(
      TensorDomain::noReductions(x->getLogicalDomain()).size() ==
      original_sizes.size());

  // handle empty reshapes by converting to full
  if (std::any_of(original_sizes.begin(), original_sizes.end(), [](int64_t s) {
        return s == 0l;
      })) {
    // The original
    bool has_dynamic_axis = false;
    bool has_zero_output_size = false;
    std::vector<Val*> new_shape;
    new_shape.reserve(new_sizes.size());
    for (int64_t s : new_sizes) {
      if (s == -1l) {
        NVF_CHECK(!has_dynamic_axis, "Only one dimension can be inferred");
        has_dynamic_axis = true;
        s = 0l;
      }
      if (s == 0l) {
        has_zero_output_size = true;
      }
      new_shape.push_back(IrBuilder::create<Val>(s, DataType::Index));
    }
    NVF_CHECK(
        has_zero_output_size,
        "Output shape must have at least one 0 when input shape does");
    return full(new_shape, x->fusion()->zeroVal(x->dtype()), x->dtype());
  }

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
  for (const auto i : arange(inp_dom.size())) {
    IterDomain* id = inp_dom[i];
    Val* id_size = id->getMaybeExpandedExtent();
    if (!id_size->isConstInt()) {
      return nullptr;
    }
    inp_sizes[i] = id_size->evaluate().as<int64_t>();
  }

  std::vector<int64_t> out_sizes(new_sizes.size());
  for (const auto i : arange(new_sizes.size())) {
    Val* id_size = new_sizes[i];
    if (!id_size->isConstInt()) {
      return nullptr;
    }
    out_sizes[i] = id_size->evaluate().as<int64_t>();
  }

  // Both inputs are outputs are static. Just use the static version
  // of reshape
  return reshape(inp_tv, inp_sizes, out_sizes);
}

} // namespace

TensorView* reshape(TensorView* inp_tv, const std::vector<Val*>& new_sizes) {
  auto inp_dom = TensorDomain::noReductions(inp_tv->getLogicalDomain());

  NVF_CHECK(
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

  auto root_domain = ops::newOutputDomain({inp_tv});

  // Create placeholder logical domain. Note it's not connected with the root
  // domain.
  std::vector<IterDomain*> logical_domain(new_sizes.size(), nullptr);
  bool found_neg_one = false;
  for (const auto i : arange(new_sizes.size())) {
    auto new_size = new_sizes.at(i);
    if (new_size->isConstScalar() && new_size->evaluate().as<int64_t>() == -1) {
      // It is usually safe to use the provided scalars as the output shapes.
      // However, if -1 is provided for some position, it will not correspond to
      // the actual extent in that position.

      NVF_CHECK(
          !found_neg_one,
          "A maximum of one value of -1 can be provided to reshape.");
      found_neg_one = true;

      Val* numel = FusionGuard::getCurFusion()->oneVal();
      Val* other_new_numel = FusionGuard::getCurFusion()->oneVal();
      for (const auto j : arange(inp_dom.size())) {
        numel = SimplifyingIrBuilder::mulExpr(numel, inp_dom.at(j)->extent());
      }
      for (const auto j : arange(new_sizes.size())) {
        if (i == j) {
          continue;
        }
        other_new_numel =
            SimplifyingIrBuilder::mulExpr(other_new_numel, new_sizes.at(j));
      }
      new_size = SimplifyingIrBuilder::divExpr(numel, other_new_numel);
      new_size = simplifyExpr(new_size);
    }
    new_size = SimplifyingIrBuilder::maybeCastExpr(DataType::Index, new_size);
    auto rf_id =
        IterDomainBuilder(FusionGuard::getCurFusion()->zeroVal(), new_size)
            .iter_type(IterType::Symbolic)
            .is_rfactor_domain(true)
            .build();
    logical_domain.at(i) = rf_id;
  }

  auto out_tv = IrBuilder::create<TensorView>(
      IrBuilder::create<TensorDomain>(
          root_domain,
          logical_domain,
          logical_domain,
          TensorDomain::getContiguityFilledWith(logical_domain, true)),
      inp_tv->dtype());

  IrBuilder::createInContainer<ViewOp>(inp_tv->container(), out_tv, inp_tv);

  return out_tv;
}

NVF_API TensorView* reshape(
    TensorView* x,
    std::function<void(AbstractTensor&)> transform) {
  auto root_domain = ops::newOutputDomain({x});
  AbstractTensor abst(root_domain);
  transform(abst);
  auto logical_domain = abst.as<IterDomain*>();
  auto out_tv = IrBuilder::create<TensorView>(
      IrBuilder::create<TensorDomain>(
          root_domain,
          logical_domain,
          logical_domain,
          TensorDomain::getContiguityFilledWith(logical_domain, true)),
      x->getDataType().value());
  IrBuilder::create<ViewOp>(out_tv, x);
  return out_tv;
}

TensorView* flatten(TensorView* x, int64_t start_dim, int64_t end_dim) {
  NVF_ERROR(x != nullptr, "Input is invalid.");
  auto inp_domain = TensorDomain::noReductions(x->getLogicalDomain());
  if (start_dim < 0) {
    start_dim += (int64_t)inp_domain.size();
  }
  if (end_dim < 0) {
    end_dim += (int64_t)inp_domain.size();
  }
  NVF_CHECK(
      start_dim >= 0 && start_dim < (int64_t)inp_domain.size(),
      "Invalid start_dim ",
      start_dim);
  NVF_CHECK(
      end_dim >= 0 && end_dim < (int64_t)inp_domain.size(),
      "Invalid end_dim ",
      end_dim);
  NVF_CHECK(start_dim <= end_dim, "start_dim must be <= end_dim");

  if (start_dim == end_dim) {
    return x;
  }

  auto out = IrBuilder::createInContainer<TensorView>(
      x->container(),
      x->domain()->flatten(start_dim, end_dim),
      x->getDataType().value());

  IrBuilder::create<ViewOp>(out, x);
  return out;
}

TensorView* squeeze(
    TensorView* x,
    const std::vector<int64_t>& dims,
    bool squeeze_expanded) {
  NVF_ERROR(x != nullptr, "Input is invalid.");
  auto x_dom = x->domain()->noReductions();
  const auto ndims = static_cast<int64_t>(x_dom.size());

  NVF_ERROR(
      (int64_t)dims.size() <= ndims,
      "The dims to squeeze must be <= the number of dims of the input tensor. ",
      "Squeeze dims: ",
      dims.size(),
      " Input Tensor dims: ",
      ndims);

  std::vector<bool> to_squeeze(ndims, false);
  for (auto dim : dims) {
    // Handle negative relative to the end dimensions specifications
    if (dim < 0) {
      dim = static_cast<int64_t>(to_squeeze.size()) + dim;
    }
    NVF_CHECK(
        (dim >= 0) && (static_cast<size_t>(dim) < to_squeeze.size()),
        "Squeeze dim is outside of Tensor size! Tensor Size: ",
        to_squeeze.size(),
        " Dim: ",
        dim);
    to_squeeze[dim] = true;
  }

  return squeeze(x, to_squeeze, squeeze_expanded);
}

TensorView* squeeze(TensorView* x, std::initializer_list<int64_t> dims) {
  return squeeze(x, std::vector<int64_t>(dims));
}

TensorView* squeeze(
    TensorView* x,
    const std::vector<bool>& to_squeeze,
    bool squeeze_expanded) {
  NVF_ERROR(x != nullptr, "Input is invalid.");
  auto x_dom = x->domain()->noReductions();
  const auto ndims = static_cast<int64_t>(x_dom.size());

  NVF_ERROR(
      ndims == (int64_t)to_squeeze.size(),
      "Invalid to_squeeze for squeeze: ",
      to_squeeze,
      ". Input tensor: ",
      x->toString());

  std::vector<IterDomain*> out_domain;
  for (const auto idx : arange(ndims)) {
    auto id = x_dom[idx];
    if (to_squeeze[idx]) {
      if (!id->isSymbolic()) {
        NVF_CHECK(
            id->isBroadcast(),
            "Can not squeeze non-broadcasting dimension(s).");
        NVF_CHECK(
            squeeze_expanded || !id->hasExpandedExtent(),
            "Refusing to squeeze expanded IterDomain ",
            id->toString(),
            ". To force removal of this axis, use squeeze_expanded=true.");
        NVF_CHECK(
            id->extent()->isConstScalar() &&
                id->extent()->evaluate().as<int64_t>() == 1,
            "Can not squeeze dimension(s) with size != 1.");
      }
    } else {
      out_domain.push_back(id->cloneWithoutRFactor());
    }
  }

  auto out = IrBuilder::create<TensorView>(
      IrBuilder::create<TensorDomain>(
          out_domain, TensorDomain::getContiguityFilledWith(out_domain, true)),
      *x->getDataType());
  if (x->hasDeviceMesh()) {
    out->setDeviceMesh(x->getDeviceMesh());
  }

  if (std::none_of(
          to_squeeze.begin(), to_squeeze.end(), [](bool b) { return b; })) {
    // If we did not squeeze any axes, this is just set()
    IrBuilder::create<LoadStoreOp>(LoadStoreOpType::Set, out, x);
  } else {
    IrBuilder::createInContainer<SqueezeOp>(x->container(), out, x, to_squeeze);
  }

  return out;
}

TensorView* unsqueeze(TensorView* x, int64_t dim) {
  NVF_ERROR(x != nullptr, "Input is invalid.");
  const auto ndims = static_cast<int64_t>(x->domain()->noReductions().size());

  if (dim < 0) {
    dim = ndims + dim + 1;
  }

  NVF_ERROR(
      dim >= 0 && dim <= ndims,
      "Invalid position to unsqueeze: ",
      dim,
      ". Input tensor: ",
      x->toString());

  std::vector<bool> broadcast_axes(ndims + 1, false);
  broadcast_axes[dim] = true;
  return broadcast(x, broadcast_axes);
}

TensorView* permute(
    TensorView* x,
    const std::initializer_list<int64_t>& new2old) {
  return permute(x, std::vector<int64_t>(new2old));
}

TensorView* permute(TensorView* x, const std::vector<int64_t>& new2old) {
  NVF_ERROR(x != nullptr, "Input is invalid.");
  if (new2old.empty()) {
    return set(x);
  }
  auto inp_domain = TensorDomain::noReductions(x->getLogicalDomain());

  NVF_CHECK(
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
      ir_utils::normalizeNew2Old(new2old, (int64_t)inp_domain.size());

  std::vector<IterDomain*> out_root;
  out_root.reserve(inp_domain.size());
  for (const auto id : inp_domain) {
    out_root.emplace_back(id->cloneWithoutRFactor());
  }

  std::vector<IterDomain*> out_logical;
  out_logical.reserve(inp_domain.size());
  for (const auto i : normalized_new2old) {
    out_logical.emplace_back(out_root.at(i));
  }

  TensorView* out_tensor = IrBuilder::create<TensorView>(
      IrBuilder::create<TensorDomain>(
          out_root,
          out_logical,
          out_logical,
          TensorDomain::getContiguityFilledWith(out_logical, true)),
      x->getDataType().value());
  if (x->hasDeviceMesh()) {
    out_tensor->setDeviceMesh(x->getDeviceMesh());
  }
  IrBuilder::create<LoadStoreOp>(LoadStoreOpType::Set, out_tensor, x);
  return out_tensor;
}

TensorView* permute(
    TensorView* x,
    const std::initializer_list<std::pair<const int64_t, int64_t>>& old2new) {
  return permute(x, std::unordered_map<int64_t, int64_t>(old2new));
}

TensorView* permute(
    TensorView* x,
    const std::unordered_map<int64_t, int64_t>& old2new) {
  auto y = set(x);
  y->reorder(old2new);
  y->commitLeafToLogical();
  return y;
}

TensorView* transpose(TensorView* x, int64_t dim0, int64_t dim1) {
  NVF_ERROR(x != nullptr, "Input is invalid.");
  const auto ndims = static_cast<int64_t>(x->domain()->noReductions().size());

  if (dim0 < 0) {
    dim0 = ndims + dim0;
  }

  if (dim1 < 0) {
    dim1 = ndims + dim1;
  }

  NVF_CHECK(
      dim0 >= 0 && dim0 <= ndims, "Invalid transpose dimension 0: ", dim0);

  NVF_CHECK(
      dim1 >= 0 && dim1 <= ndims, "Invalid transpose dimension 1: ", dim1);

  std::vector<int64_t> new2old(ndims);
  for (const auto i : arange(ndims)) {
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
  NVF_ERROR(x != nullptr, "Input is invalid.");
  const auto ndims = static_cast<int64_t>(x->domain()->noReductions().size());

  NVF_CHECK(
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

bool hasSimilarDtype(DataType base, DataType dt) {
  if (base == dt) {
    return true;
  } else if (isComplexType(base)) {
    return isComplexType(dt);
  } else if (isFloatingPointType(base)) {
    return isFloatingPointType(dt);
  } else if (isBooleanType(base)) {
    return isBooleanType(dt);
  } else if (isIntegralType(base)) {
    return isIntegralType(dt);
  }
  NVF_THROW("Unrecognized base dtype.");
}

// Padding widths are assumed to be non-negative. Currently there's no
// validation.
TensorView* pad(
    TensorView* inp,
    const std::vector<Val*>& pad_widths,
    Val* value,
    std::optional<IterType> iter_type_opt) {
  DataType dt = inp->getDataType().value();
  if (!value) {
    // Create a zero of the appropriate type
    if (isComplexType(dt)) {
      value = static_cast<Val*>(
          IrBuilder::create<Val>(std::complex<double>(0), dt));
    } else if (isFloatingPointType(dt)) {
      value = static_cast<Val*>(IrBuilder::create<Val>(0.0, dt));
    } else if (isBooleanType(dt)) {
      value = static_cast<Val*>(IrBuilder::create<Val>(false, dt));
    } else {
      value = static_cast<Val*>(IrBuilder::create<Val>(0L, dt));
    }
  }
  NVF_CHECK(
      hasSimilarDtype(dt, value->getDataType().value()),
      "Tensor arg and pad value must have the same dtype.");
  const auto inp_dom = TensorDomain::noReductions(inp->getLogicalDomain());
  const auto ndims = inp_dom.size();

  NVF_CHECK(
      pad_widths.size() % 2 == 0 && pad_widths.size() / 2 <= ndims,
      "Invalid number of padding widths: ",
      pad_widths.size());

  const auto num_padded_dims = pad_widths.size() / 2;
  const auto num_non_padded_dims = ndims - num_padded_dims;

  std::vector<IterDomain*> root_ids(ndims);
  std::vector<IterDomain*> logical_ids(ndims);

  // PadOp requires pad widths for all dimensions, even for non-padded
  // ones.

  std::vector<Val*> normalized_pad_widths;

  // Fill zero for non padded dimensions
  for (const auto i : arange(num_non_padded_dims)) {
    (void)i;
    normalized_pad_widths.push_back(FusionGuard::getCurFusion()->zeroVal());
    normalized_pad_widths.push_back(FusionGuard::getCurFusion()->zeroVal());
  }

  // torch.pad has padding widths of inner dimensions before outer
  // dimensions
  for (const auto i : arange(num_padded_dims)) {
    auto left_pad = pad_widths.at(num_padded_dims * 2 - (i + 1) * 2);
    auto right_pad = pad_widths.at(num_padded_dims * 2 - (i + 1) * 2 + 1);
    normalized_pad_widths.push_back(maybeCastOp(DataType::Index, left_pad));
    normalized_pad_widths.push_back(maybeCastOp(DataType::Index, right_pad));
  }

  // Indicates if any dimension is actually padded. Can be false even
  // when non-empty padding width vector is passed
  bool is_padded_any = false;
  for (const auto idx : arange(ndims)) {
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
      // Expand the root domain and mark it as a logical domain
      out_rf_id = IterDomain::resize(
          out_root_id, left_pad, right_pad, true, iter_type_opt);
      is_padded_any = true;
    }
    root_ids.at(idx) = out_root_id;
    logical_ids.at(idx) = out_rf_id;
  }

  // If all of the padding widths are just zero, this is just a set op.
  if (!is_padded_any) {
    return set(inp);
  }

  auto out = IrBuilder::create<TensorView>(
      IrBuilder::create<TensorDomain>(
          root_ids,
          logical_ids,
          logical_ids,
          TensorDomain::getContiguityFilledWith(logical_ids, true)),
      *inp->getDataType());

  IrBuilder::create<PadOp>(out, inp, normalized_pad_widths, value);

  return out;
}

// cat is implemented as PadOp and CatOp. Padding is done first to
// account for the size difference between each of the inputs and the
// output. All of the inputs to CatOp have the same shape as the
// output shape.
TensorView* cat(
    const std::vector<TensorView*>& inputs,
    int64_t cat_dim,
    std::optional<IterType> iter_type_opt,
    bool manual_padding) {
  NVF_CHECK(!inputs.empty(), "No input tensor given");

  const auto dtype = inputs.at(0)->getDataType().value();

  std::vector<std::vector<IterDomain*>> inp_doms;
  int64_t ndims = -1;

  for (auto inp : inputs) {
    NVF_CHECK(
        inp->getDataType().value() == dtype,
        "Can't concatenate tensors with different data types: ",
        dtype,
        ", ",
        inp->getDataType().value());
    inp_doms.emplace_back(TensorDomain::noReductions(inp->getLogicalDomain()));
    auto i_ndims = static_cast<int64_t>(inp_doms.back().size());
    if (ndims == -1) {
      ndims = i_ndims;
    } else {
      NVF_CHECK(
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

  NVF_CHECK(
      cat_dim >= 0 && cat_dim < ndims, "Invalid dimension to cat: ", cat_dim);

  // Special handling for the case where there's only one input
  if (inputs.size() == 1) {
    return set(inputs.at(0));
  }

  // short-circuit: If manual_padding is true, check that all inputs are already
  // padded. Assume the padding is correct and create the catOp immediately.
  // Primarily used for FusionTranslation, which adds the padOp for each tensor
  // separately.
  if (manual_padding) {
    bool all_padded =
        std::all_of(inputs.begin(), inputs.end(), [](TensorView* tv) {
          return tv->definition()->isA<PadOp>();
        });
    NVF_ERROR(
        all_padded,
        "Expected all inputs to be padded when manual_padding is True.");
    std::vector<Val*> input_vals(inputs.begin(), inputs.end());
    auto out = ops::newOutputTV(input_vals, dtype);
    IrBuilder::create<CatOp>(out, input_vals, cat_dim);
    return out;
  }

  Val* concat_ext = nullptr;

  for (const auto i : arange(inputs.size())) {
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
  for (const auto input_idx : arange(inputs.size())) {
    const auto& inp_dom = inp_doms.at(input_idx);
    std::vector<Val*> pad_widths(ndims * 2);
    for (const auto dim : arange(ndims)) {
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
        // IterDomain or Symbolic, i.e. Broadcast or Iteration.
        NVF_ERROR(
            inp_root_id->isSymbolic() ||
                ((inp_root_id->isIteration() || inp_root_id->isBroadcast()) &&
                 !inp_root_id->maybePartial()),
            "Unsupported IterDomain to concatenate: ",
            inp_root_id->toString());
        // The right pad of the last tensor is just zero
        right_pad = input_idx < inputs.size() - 1
            ? SimplifyingIrBuilder::subExpr(
                  right_pad, inp_root_id->getMaybeExpandedExtent())
            : FusionGuard::getCurFusion()->zeroVal();
        left_pad_i = left_pad;
        right_pad_i = right_pad;
        left_pad = add(left_pad, inp_root_id->getMaybeExpandedExtent());
      }
      // The pad width argument to pad should be ordered such that the
      // widths of inner dimensions come first.
      pad_widths.at((ndims - dim - 1) * 2) = left_pad_i;
      pad_widths.at((ndims - dim - 1) * 2 + 1) = right_pad_i;
    }

    TensorView* padded =
        pad(inputs.at(input_idx), pad_widths, nullptr, iter_type_opt);
    NVF_ERROR(padded->definition() != nullptr);
    if (padded->definition()->isA<LoadStoreOp>()) {
      // If the pad was actually a "set" of the input, that means we proved that
      // the pad widths are all zero. In that case, we don't need to check the
      // other input sizes since we know that they are all empty (i.e. they have
      // size zero in the cat dimension). We just return the "padded" tensor in
      // that case.
      return padded;
    }
    resized_inputs.at(input_idx) = padded;
  }

  // Now all of resized_inputs have the same shape as the out tensor
  auto out = ops::newOutputTV(resized_inputs, dtype);

  IrBuilder::create<CatOp>(out, resized_inputs, cat_dim);

  return out;
}

std::string Slice::toString() const {
  std::ostringstream oss;
  oss << "Slice(start=" << start->toInlineString()
      << ", stop=" << stop->toInlineString()
      << ", step=" << step->toInlineString() << ")";
  return oss.str();
}

TensorView* slice(
    TensorView* inp,
    const std::vector<Slice>& ranges,
    bool manual_normalization) {
  const auto inp_dom = TensorDomain::noReductions(inp->getLogicalDomain());
  const int64_t ndims = std::ssize(inp_dom);

  NVF_CHECK_EQ(
      ndims,
      std::ssize(ranges),
      "The range vector must have the same number of Slice descriptors.")

  ExpressionEvaluator expr_eval;

  const auto get_int = [&expr_eval](Val* x) -> std::optional<int64_t> {
    if (x == nullptr) {
      return std::nullopt;
    }
    auto inferred_val = expr_eval.evaluate(x);
    if (inferred_val.hasValue()) {
      return inferred_val.as<int64_t>();
    } else {
      return std::nullopt;
    }
  };

  // Specialized min for extents. Do some more simplification beyond
  // SimplifyingIrBuilder that are only valid for extents.
  const auto min_extents = [&](Val* x, Val* y) -> Val* {
    auto x_int = get_int(x);
    auto y_int = get_int(y);
    // Since extents are never negative, if one is 0, that must be the mininum.
    if (x_int == 0) {
      return x;
    } else if (y_int == 0) {
      return y;
    }
    // Simplify patterns like min(min(x, 32), 32) to min(x, 32) as it
    // isn't uncommon.
    auto bop = dynamic_cast<BinaryOp*>(x->definition());
    if (y_int != std::nullopt && bop != nullptr &&
        bop->getBinaryOpType() == BinaryOpType::Min) {
      if (auto lhs_int = get_int(bop->lhs()); lhs_int != std::nullopt) {
        return SimplifyingIrBuilder::minExpr(
            bop->rhs(), IrBuilder::create<Val>(std::min(*lhs_int, *y_int)));
      } else if (auto rhs_int = get_int(bop->rhs()); rhs_int != std::nullopt) {
        return SimplifyingIrBuilder::minExpr(
            bop->lhs(), IrBuilder::create<Val>(std::min(*rhs_int, *y_int)));
      }
    }

    return SimplifyingIrBuilder::minExpr(x, y);
  };

  const auto normalize_slice_range =
      [&manual_normalization, &min_extents, &get_int](
          Slice range, Val* extent) -> Slice {
    auto cast_extent =
        SimplifyingIrBuilder::maybeCastExpr(DataType::Index, extent);

    auto zero = FusionGuard::getCurFusion()->zeroVal(DataType::Index);

    auto start_int = get_int(range.start);
    auto stop_int = get_int(range.stop);

    // norm_start = max(0, start < 0 ? start + extent : start)
    if (range.start == nullptr) {
      range.start = zero;
      start_int = 0;
    } else if (start_int != 0) {
      range.start =
          SimplifyingIrBuilder::maybeCastExpr(DataType::Index, range.start);
      if (!manual_normalization) {
        range.start = SimplifyingIrBuilder::maxExpr(
            zero,
            SimplifyingIrBuilder::whereExpr(
                SimplifyingIrBuilder::ltExpr(range.start, zero),
                SimplifyingIrBuilder::addExpr(range.start, cast_extent),
                range.start));
      }
      start_int = get_int(range.start);
    }

    // norm_stop = max(norm_start, min(extent, stop < 0 ? stop + extent : stop)
    if (range.stop == nullptr) {
      range.stop = cast_extent;
    } else if (!range.stop->sameAs(extent)) {
      range.stop =
          SimplifyingIrBuilder::maybeCastExpr(DataType::Index, range.stop);
      // Commonly, range.start is zero and stop is non negative
      if (start_int == 0 && stop_int >= 0) {
        range.stop = min_extents(cast_extent, range.stop);
      } else {
        if (!manual_normalization) {
          range.stop = SimplifyingIrBuilder::maxExpr(
              range.start,
              min_extents(
                  cast_extent,
                  SimplifyingIrBuilder::whereExpr(
                      SimplifyingIrBuilder::ltExpr(range.stop, zero),
                      SimplifyingIrBuilder::addExpr(range.stop, cast_extent),
                      range.stop)));
        }
      }
    }

    // Ensure step is of type Index
    if (range.step == nullptr) {
      range.step = FusionGuard::getCurFusion()->oneVal(DataType::Index);
    } else {
      range.step =
          SimplifyingIrBuilder::maybeCastExpr(DataType::Index, range.step);
    }

    return range;
  };

  for (const Slice& range : ranges) {
    // Step not supported yet
    NVF_CHECK(
        range.step == nullptr || range.step->isOneInt(),
        "Unsupported step (must be 1 or null): ",
        range.step->toString());
  }

  std::vector<Slice> normalized_ranges;
  normalized_ranges.reserve(ndims);
  std::vector<IterDomain*> root_ids;
  root_ids.reserve(ndims);
  std::vector<IterDomain*> logical_ids;
  logical_ids.reserve(ndims);

  bool needs_real_slicing = false;
  for (const auto& [inp_root_id, range] : zip(inp_dom, ranges)) {
    Val* inp_root_size = inp_root_id->getMaybeExpandedExtent();
    Slice normalized_range = normalize_slice_range(range, inp_root_size);
    normalized_ranges.push_back(normalized_range);
    IterDomain* out_root_id = nullptr;
    IterDomain* out_rf_id = nullptr;
    if (normalized_range.start->isZeroInt() &&
        normalized_range.stop->sameAs(inp_root_size) &&
        normalized_range.step->isOneInt()) {
      // This dim doesn't need slicing
      out_root_id = inp_root_id->cloneWithoutRFactor();
      out_rf_id = out_root_id;
    } else {
      // Clip the start and stop values to the extent of the input
      out_root_id =
          IterDomainBuilder(inp_root_id).is_rfactor_domain(true).build();
      out_rf_id = IterDomain::resize(
          out_root_id,
          SimplifyingIrBuilder::negExpr(normalized_range.start),
          SimplifyingIrBuilder::subExpr(normalized_range.stop, inp_root_size),
          true);
      needs_real_slicing = true;
    }
    root_ids.push_back(out_root_id);
    logical_ids.push_back(out_rf_id);
  }

  // If slicing isn't actually needed, just return a copy
  if (!needs_real_slicing) {
    return set(inp);
  }

  auto out = IrBuilder::create<TensorView>(
      IrBuilder::create<TensorDomain>(
          root_ids,
          logical_ids,
          logical_ids,
          TensorDomain::getContiguityFilledWith(logical_ids, true)),
      *inp->getDataType());

  IrBuilder::create<SliceOp>(out, inp, normalized_ranges);
  return out;
}

TensorView* slice(
    TensorView* inp,
    const std::vector<int64_t>& starts,
    const std::vector<int64_t>& stops) {
  std::vector<int64_t> steps(starts.size(), 1);
  return slice(inp, starts, stops, steps);
}

TensorView* slice(
    TensorView* inp,
    const std::vector<int64_t>& starts,
    const std::vector<int64_t>& stops,
    const std::vector<int64_t>& steps) {
  std::vector<Slice> slices;
  slices.reserve(starts.size());
  for (size_t i = 0; i < starts.size(); i++) {
    slices.push_back(
        {IrBuilder::create<Val>(starts[i]),
         IrBuilder::create<Val>(stops[i]),
         IrBuilder::create<Val>(steps[i])});
  }
  return slice(inp, slices);
}

std::vector<TensorView*> chunk(
    TensorView* in,
    const int64_t chunks,
    int64_t dim) {
  NVF_CHECK(chunks > 0);

  const auto in_logical = TensorDomain::noReductions(in->getLogicalDomain());
  const auto num_dims = static_cast<int64_t>(in_logical.size());
  dim = wrapDim(dim, num_dims);
  Val* dim_size = in_logical[dim]->extent();
  Val* slice_size = SimplifyingIrBuilder::ceilDivExpr(
      dim_size, IrBuilder::create<Val>(chunks));

  std::vector<TensorView*> slices;
  slices.reserve(chunks);
  std::vector<Slice> ranges(num_dims);
  for (auto i : arange(chunks)) {
    ranges[dim].start = ranges[dim].stop;
    ranges[dim].stop =
        (i == chunks - 1 ? nullptr
                         : SimplifyingIrBuilder::mulExpr(
                               slice_size, IrBuilder::create<Val>(i + 1)));
    slices.push_back(slice(in, ranges));
  }
  return slices;
}

TensorView* broadcast(
    TensorView* inp,
    const std::vector<bool>& is_broadcast_dim) {
  auto nBCastDims = is_broadcast_dim.size();
  // Validate is_broadcast_dim
  unsigned int n_broadcasts = 0;
  for (auto ent : is_broadcast_dim) {
    if (ent) {
      n_broadcasts++;
    }
  }

  NVF_CHECK(
      nBCastDims - n_broadcasts ==
          TensorDomain::noReductions(inp->getLogicalDomain()).size(),
      "Invalid broadcast, number of false entries in is_broadcast_dim expected "
      "to be ",
      TensorDomain::noReductions(inp->getLogicalDomain()).size(),
      " but received ",
      nBCastDims - n_broadcasts);

  if (n_broadcasts == 0) {
    auto identity = set(inp);
    NVF_ERROR(
        identity->getValType().value() == ValType::TensorView,
        "Expected identity op, but didn't get a TensorView back.");
    return identity->as<TensorView>();
  }

  std::vector<IterDomain*> out_domain;
  // Don't propagate reduction IDs through arith ops.
  auto inp_domain = TensorDomain::noReductions(inp->getLogicalDomain());
  size_t iinp = 0, ibdim = 0;
  while (ibdim < is_broadcast_dim.size()) {
    if (is_broadcast_dim[ibdim]) {
      out_domain.push_back(IterDomainBuilder(
                               FusionGuard::getCurFusion()->zeroVal(),
                               FusionGuard::getCurFusion()->oneVal())
                               .iter_type(IterType::Broadcast)
                               .build());
    } else {
      out_domain.push_back(
          IterDomainBuilder(inp_domain[iinp]).resetSchedulingParams().build());
      iinp++;
    }
    ibdim++;
  }

  TensorView* out = IrBuilder::create<TensorView>(
      IrBuilder::create<TensorDomain>(
          out_domain, TensorDomain::getContiguityFilledWith(out_domain, true)),
      inp->getDataType().value());
  if (inp->hasDeviceMesh()) {
    out->setDeviceMesh(inp->getDeviceMesh());
  }
  IrBuilder::create<BroadcastOp>(out, inp, is_broadcast_dim);
  return out;
}

TensorView* expand(TensorView* inp, const std::vector<Val*>& expanded_sizes) {
  auto inp_domain = TensorDomain::noReductions(inp->getLogicalDomain());

  NVF_CHECK(
      expanded_sizes.size() >= inp_domain.size(),
      "Invalid expand, number of sizes provided is expected to be at least ",
      inp_domain.size(),
      " but received ",
      expanded_sizes.size());

  inp = ops::maybe_broadcast_inner_to_rank(inp, expanded_sizes.size());
  inp_domain = TensorDomain::noReductions(inp->getLogicalDomain());

  std::vector<Val*> maybe_expanded_sizes;
  maybe_expanded_sizes.resize(inp_domain.size(), nullptr);

  // Might a dimension actually get expanded? This will be true if any input
  // IterDomains are Symbolic, since these may or may not be Broadcast.
  bool expanded = false;

  std::vector<IterDomain*> out_domain;
  for (auto i : arange(inp_domain.size())) {
    auto inp_id = inp_domain[i];
    auto out_id_builder = IterDomainBuilder(inp_id);
    maybe_expanded_sizes[i] = inp_domain[i]->extent();

    auto expanded_size_int = expanded_sizes[i]->value();

    // If the expanded size is -1, let the input extent be propagated
    // as is
    if (expanded_size_int.hasValue() && expanded_size_int.as<int64_t>() == -1) {
      // This is just done for clarity. It isn't necessary as it's
      // already done when constructing out_id_builder.
      out_id_builder.extent(inp_id->extent());
    } else if (
        // special patch for Symbolic IterDomain with a static size-1 extent
        // since we know it will become broadcast at concretization
        // See Issue: https://github.com/NVIDIA/Fuser/pull/1393
        (inp_id->extent()->isConstInt() &&
         inp_id->extent()->evaluate().as<int64_t>() == 1) &&
        (!expanded_size_int.hasValue() ||
         expanded_size_int.as<int64_t>() != 1)) {
      // When input id is a broadcast, expand the extent to the given
      // size, which can be concrete or symbolic.
      expanded = true;
      auto expanded_extent = maybeCastOp(DataType::Index, expanded_sizes[i]);
      out_id_builder.expanded_extent(expanded_extent);
      // need to mark iter type as Broadcast for Symbolic input domains
      out_id_builder.iter_type(IterType::Broadcast);
      maybe_expanded_sizes[i] = expanded_extent;
    } else if (
        inp_id->isSymbolic() &&
        (!inp_id->extent()->isConstInt() &&
         !inp_id->extent()->sameAs(expanded_sizes[i]))) {
      // need to mark iter type as Symbolic since this might not be an expand
      // after concretization
      expanded = true;
      out_id_builder.iter_type(IterType::Symbolic);
      auto expanded_extent = maybeCastOp(DataType::Index, expanded_sizes[i]);
      // We set the extent instead of the expanded extent on a Symbolic
      // IterDomain. At concretization, if the IterType is determined to be
      // Broadcast, we will replace this with 1 and use the old extent as
      // expandedExtent.
      out_id_builder.extent(expanded_extent);
      maybe_expanded_sizes[i] = expanded_extent;
    } else if (!inp_id->extent()->isConstInt()) {
      // Input id is non-broadcast and its extent is symbolic. Promote
      // the extent to the given expanded size.
      // Note that expansion to 1 just means its extent becomes 1 and
      // does not mean the ID becomes a broadcast.
      out_id_builder.extent(maybeCastOp(DataType::Index, expanded_sizes[i]));
    } else {
      // Input id is non-expand and its extent is concrete. Nothing
      // to expand, but the input and expanded sizes should match if
      // the expanded size is also concrete.
      auto inp_id_size_int = inp_id->extent()->evaluate();
      if (expanded_size_int.is<int64_t>()) {
        NVF_CHECK(
            inp_id_size_int == expanded_size_int,
            "Invalid expand size, ",
            expanded_sizes[i]->toString(),
            ", for ",
            inp_id->toString());
      }
    }
    out_domain.push_back(out_id_builder.build());
  }

  auto* out = IrBuilder::create<TensorView>(
      IrBuilder::create<TensorDomain>(
          out_domain, TensorDomain::getContiguityFilledWith(out_domain, true)),
      inp->getDataType().value());
  if (!expanded) {
    IrBuilder::create<LoadStoreOp>(LoadStoreOpType::Set, out, inp);
  } else {
    IrBuilder::create<ExpandOp>(out, inp, maybe_expanded_sizes);
  }
  return out;
}

TensorView* expand_as(TensorView* inp, TensorView* other) {
  auto inp_domain = TensorDomain::noReductions(inp->getLogicalDomain());
  auto other_domain = TensorDomain::noReductions(other->getLogicalDomain());

  NVF_CHECK(
      inp_domain.size() <= other_domain.size(),
      "Invalid expand_as, dimensions of inp is higher than dimensions of "
      "other, expected other to be at least ",
      inp_domain.size(),
      " but received ",
      other_domain.size());

  inp = ops::maybe_broadcast_inner_to_rank(inp, other_domain.size());
  inp_domain = TensorDomain::noReductions(inp->getLogicalDomain());

  std::vector<IterDomain*> out_domain;
  std::vector<Val*> maybe_expanded_sizes;
  bool expanded = false;
  for (auto i : arange(inp_domain.size())) {
    auto inp_id = inp_domain[i];
    auto other_id = other_domain[i];

    auto out_id_builder = IterDomainBuilder(inp_id);
    Val* maybe_expanded_size = inp_id->extent();

    if (!inp_id->isBroadcast()) {
      NVF_ERROR(
          !other_id->isBroadcast(),
          "Cannot expand as a tensor if other has broadcast dimensions that "
          "don't map to broadcast dimensions in the input.");
      if (!inp_id->isConstInt() && other_id->isConstInt()) {
        out_id_builder.extent(
            ops::promoteSize(inp_id->extent(), other_id->extent()));
      }
    } else {
      if (!other_id->isBroadcast()) {
        expanded = true;
        out_id_builder.expanded_extent(other_id->extent());
        maybe_expanded_size = other_id->extent();
      } else if (other_id->isBroadcast() && other_id->hasExpandedExtent()) {
        expanded = true;
        out_id_builder.expanded_extent(other_id->expandedExtent());
        maybe_expanded_size = other_id->expandedExtent();
      }
    }
    out_domain.push_back(out_id_builder.build());
    maybe_expanded_sizes.push_back(maybe_expanded_size);
  }

  auto* out = IrBuilder::create<TensorView>(
      IrBuilder::create<TensorDomain>(
          out_domain, TensorDomain::getContiguityFilledWith(out_domain, true)),
      inp->getDataType().value());
  if (!expanded) {
    IrBuilder::create<LoadStoreOp>(LoadStoreOpType::Set, out, inp);
  } else {
    IrBuilder::create<ExpandOp>(out, inp, maybe_expanded_sizes);
  }
  return out;
}

TensorView* repeat(
    TensorView* inp_tv,
    const std::vector<int64_t>& repeat_times) {
  const auto ndims =
      TensorDomain::noReductions(inp_tv->getLogicalDomain()).size();

  // Handle repetitions of non-broadcast IDs first. Each ID is
  // individully repeated by:
  //
  // Step 1. Insert a broadcast ID immediately outside of the
  // repeated ID
  // Step 2. Expand the broadcast ID by the repetition factor
  // Step 3. Flatten the expanded ID and the repeated ID
  //
  // Note that it's also possible to repeat multiple non-broadcast IDs
  // once by inserting and expanding broadcast IDs by one BroadcastOp
  // and one ExpandOp.

  bool has_repetition_of_broadcast = false;
  auto intermediate_tv = inp_tv;
  for (const auto i : arange(ndims)) {
    if (repeat_times.at(i) == 1) {
      continue;
    }

    auto inp_id = intermediate_tv->getLogicalDomain().at(i);

    // Broadcast is handled after this
    if (inp_id->isBroadcast()) {
      has_repetition_of_broadcast = true;
      continue;
    }

    // Step 1: Insert a broadcast ID
    std::vector<bool> bcast_flags(ndims + 1, false);
    bcast_flags.at(i) = true;
    auto broadcast_tv = broadcast(intermediate_tv, bcast_flags);

    // Step 2: Expand the broadcast ID for the repetition factor
    std::vector<Val*> expanded_sizes(
        bcast_flags.size(), IrBuilder::create<Val>(-1L));
    expanded_sizes.at(i) = IrBuilder::create<Val>(repeat_times.at(i));
    auto expanded_tv = expand(broadcast_tv, expanded_sizes);

    // Step 3: Reshape to merge the broadcast ID and the repeated ID
    intermediate_tv = flatten(expanded_tv, (int64_t)i, (int64_t)i + 1);
  }

  if (!has_repetition_of_broadcast) {
    return intermediate_tv;
  }

  // Repeat broadcast IDs. The expand approach doesn't work as reshape
  // would just squeeze repeated IDs and thus there would be no
  // merge. Expanded IDs would remain to be expanded broadcast IDs. To
  // concretize them, use RepeatOp
  std::vector<IterDomain*> new_domain;
  new_domain.reserve(ndims);
  std::vector<std::optional<bool>> new_contiguity;
  new_contiguity.reserve(ndims);

  for (const auto i : arange(ndims)) {
    auto inp_id = intermediate_tv->getLogicalDomain().at(i);
    IterDomain* new_id = nullptr;

    if (repeat_times.at(i) > 1 && inp_id->isBroadcast()) {
      new_id = IterDomainBuilder(inp_id)
                   .extent(IrBuilder::create<Val>(
                       repeat_times.at(i), DataType::Index))
                   .iter_type(IterType::Iteration)
                   .build();
    } else {
      new_id = inp_id->cloneWithoutRFactor();
    }

    new_domain.push_back(new_id);
    new_contiguity.push_back(
        new_id->isBroadcast() ? std::optional<bool>(std::nullopt)
                              : std::optional<bool>(true));
  }

  auto out_tv = IrBuilder::create<TensorView>(
      IrBuilder::create<TensorDomain>(new_domain, new_contiguity),
      inp_tv->dtype());

  IrBuilder::create<RepeatOp>(out_tv, intermediate_tv);

  return out_tv;
}

} // namespace nvfuser
