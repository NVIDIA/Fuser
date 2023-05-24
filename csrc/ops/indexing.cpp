// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on

#include <ir/all_nodes.h>
#include <ir/builder.h>
#include <ir/iostream.h>
#include <ir/utils.h>
#include <ops/all_ops.h>
#include <ops/utils.h>
#include <type.h>

#include <c10/util/BFloat16.h>
#include <c10/util/Exception.h>
#include <c10/util/Half.h>
#include <c10/util/irange.h>

namespace nvfuser {

TensorView* select(TensorView* tv, int dim, Val* index) {
  auto dom = TensorDomain::noReductions(tv->getMaybeRFactorDomain());
  TORCH_CHECK(!dom.empty(), "select can not be applied to 0d tensor.");

  std::vector<IterDomain*> new_root;
  new_root.reserve(dom.size() - 1);

  if (dim < 0) {
    dim += (int)dom.size();
  }

  TORCH_CHECK(
      dim >= 0 && dim < (int)dom.size(),
      "Select on invalid axis, received: ",
      dim,
      " however tensor view only has ",
      dom.size(),
      " non-reduction dims.");

  for (auto i : c10::irange(dom.size())) {
    if ((int)i != dim) {
      new_root.emplace_back(dom[i]->cloneWithoutRFactor());
    }
  }

  auto td = IrBuilder::create<TensorDomain>(
      new_root, TensorDomain::getContiguityFilledWith(new_root, true));
  auto out = IrBuilder::create<TensorView>(td, *tv->getDataType());
  IrBuilder::create<SelectOp>(out, tv, dim, index);
  return out;
}

// index_select
TensorView* index_select(TensorView* lookup_tv, int dim, TensorView* index_tv) {
  DataType dtype = lookup_tv->getDataType().value();
  TORCH_CHECK(
      dtype != DataType::Null, "Invalid datatype provided for new value.");
  auto lookup_dom =
      TensorDomain::noReductions(lookup_tv->getMaybeRFactorDomain());
  auto index_dom =
      TensorDomain::noReductions(index_tv->getMaybeRFactorDomain());
  size_t n_dims = lookup_dom.size();
  TORCH_CHECK(n_dims > 0, "index_select can not be applied to 0d tensor.");
  TORCH_CHECK(
      index_dom.size() <= 1, "index array must be 1d or scalar tensor.");

  if (index_dom.empty()) {
    auto select_tv = select(lookup_tv, dim, index_tv);
    return unsqueeze(select_tv, dim);
  }

  if (dim < 0) {
    dim += (int)lookup_dom.size();
  }

  std::vector<IterDomain*> new_root;
  new_root.reserve(lookup_dom.size() - 1);
  TORCH_CHECK(
      dim >= 0 && dim < (int)lookup_dom.size(),
      "index_select on invalid axis, received: ",
      dim,
      " however tensor view only has ",
      lookup_dom.size(),
      " non-reduction dims.");

  for (auto i : c10::irange(lookup_dom.size())) {
    if ((int)i != dim) {
      new_root.emplace_back(lookup_dom[i]->cloneWithoutRFactor());
    } else {
      new_root.emplace_back(index_dom[0]->cloneWithoutRFactor());
    }
  }

  auto td = IrBuilder::create<TensorDomain>(
      new_root, TensorDomain::getContiguityFilledWith(new_root, true));
  auto out = IrBuilder::create<TensorView>(td, dtype);

  // broadcast index to lookup's rank.
  index_tv =
      ops::maybe_broadcast_index_tv(index_tv->as<TensorView>(), dim, n_dims);
  IrBuilder::create<IndexSelectOp>(out, lookup_tv, dim, index_tv);
  return out;
}

// torch.gather
TensorView* torch_gather(TensorView* inp, int dim, TensorView* index) {
  auto inp_domain = TensorDomain::noReductions(inp->getMaybeRFactorDomain());
  auto idx_domain = TensorDomain::noReductions(index->getMaybeRFactorDomain());
  TORCH_CHECK(
      !inp_domain.empty(), "torch.gather can not be applied to 0d tensor.");
  TORCH_CHECK(
      idx_domain.size() == inp_domain.size(),
      "the input and index tensor must have the same dimensions for torch.gather");

  if (dim < 0) {
    dim += (int)idx_domain.size();
  }
  TORCH_CHECK(
      dim >= 0 && dim < (int)inp_domain.size(),
      "torch.gather on invalid axis, received: ",
      dim,
      " however tensor view only has ",
      inp_domain.size(),
      " non-reduction dims.");
  std::vector<IterDomain*> out_domain;
  out_domain.reserve(idx_domain.size());
  for (auto idx_domain_ptr : idx_domain) {
    out_domain.push_back(
        IterDomainBuilder(idx_domain_ptr)
            .iter_type(
                idx_domain_ptr->getIterType() == IterType::Iteration
                    ? IterType::GatherScatter
                    : idx_domain_ptr->getIterType())
            .build());
  }

  TensorView* out_tensor = IrBuilder::create<TensorView>(
      IrBuilder::create<TensorDomain>(
          out_domain, TensorDomain::getContiguityFilledWith(out_domain, true)),
      inp->getDataType().value());

  IrBuilder::create<TorchGatherOp>(out_tensor, inp, dim, index, false);
  return out_tensor->as<TensorView>();
}

// torch.scatter torch.scatter_add
TensorView* scatterOp(
    ScatterOpType type,
    TensorView* self,
    int dim,
    TensorView* index,
    TensorView* src) {
  auto self_dom = TensorDomain::noReductions(self->getMaybeRFactorDomain());
  auto idx_dom = TensorDomain::noReductions(index->getMaybeRFactorDomain());
  auto src_dom = TensorDomain::noReductions(src->getMaybeRFactorDomain());

  TORCH_CHECK(!self_dom.empty(), "scatter can not be applied to 0d tensor.");
  TORCH_CHECK(
      self_dom.size() == idx_dom.size() && self_dom.size() == src_dom.size(),
      "self, index and src tensor should all have the same number of dimensions in scatter like ops.");
  if (dim < 0) {
    dim += (int)self_dom.size();
  }
  TORCH_CHECK(
      dim >= 0 && dim < (int)self_dom.size(),
      "Scatter on invalid axis, received: ",
      dim,
      " however tensor view only has ",
      self_dom.size(),
      " non-reduction dims.");

  // The shape of output tensor is same as self tensor.
  std::vector<IterDomain*> out_domain;
  for (const auto i : c10::irange(self_dom.size())) {
    out_domain.push_back(
        IterDomainBuilder(self_dom[i])
            .iter_type(
                self_dom[i]->getIterType() == IterType::Iteration
                    ? IterType::GatherScatter
                    : self_dom[i]->getIterType())
            .build());
  }

  TensorView* out_tensor = IrBuilder::create<TensorView>(
      IrBuilder::create<TensorDomain>(
          out_domain, TensorDomain::getContiguityFilledWith(out_domain, true)),
      self->getDataType().value());

  IrBuilder::create<ScatterOp>(type, out_tensor, self, dim, index, src);
  return out_tensor->as<TensorView>();
}

TensorView* scatter(
    TensorView* self,
    int dim,
    TensorView* index,
    TensorView* src) {
  return scatterOp(ScatterOpType::Set, self, dim, index, src);
}

TensorView* take_along_axis(TensorView* inp, TensorView* index, int64_t dim) {
  const auto inp_domain =
      TensorDomain::noReductions(inp->getMaybeRFactorDomain());
  const auto idx_domain =
      TensorDomain::noReductions(index->getMaybeRFactorDomain());

  TORCH_CHECK(
      !inp_domain.empty(), "take_along_axis can not be applied to 0d tensor.");
  TORCH_CHECK(
      idx_domain.size() == inp_domain.size(),
      "The input and index tensor must have the same dimensions for take_along_axis");

  if (dim < 0) {
    dim += (int)idx_domain.size();
  }

  TORCH_CHECK(
      dim >= 0 && dim < (int)inp_domain.size(),
      "take_along_axis on invalid axis, received: ",
      dim,
      " however tensor view only has ",
      inp_domain.size(),
      " non-reduction dims.");

  std::vector<IterDomain*> out_domain(idx_domain.size());

  for (const auto i : c10::irange(idx_domain.size())) {
    auto inp_id = inp_domain.at(i);
    auto idx_id = idx_domain.at(i);

    TORCH_CHECK(
        !inp_id->maybePartial(),
        "Partial domain not supported: ",
        inp_id->toString());
    TORCH_CHECK(
        !idx_id->maybePartial(),
        "Partial domain not supported: ",
        idx_id->toString());

    TORCH_CHECK(
        inp_id->getIterType() != IterType::Iteration ||
            inp_id->getIterType() != IterType::Broadcast,
        "Unsupported IterType of an input domian: ",
        inp_id->toString());
    TORCH_CHECK(
        idx_id->getIterType() != IterType::Iteration ||
            idx_id->getIterType() != IterType::Broadcast,
        "Unsupported IterType of an index domian: ",
        idx_id->toString());

    // Even for the non-indexed domains, the output ID should be
    // determined by the index ID when:
    // 1. The input is a broadcast but the index is not
    // 2. Both the input and index are broadcast, but the index is
    // expanded. The input may also be expanded, but that shouldn't
    // matter.
    if (static_cast<int>(i) == dim ||
        (inp_id->isBroadcast() && !idx_id->isBroadcast()) ||
        (inp_id->isBroadcast() && idx_id->isBroadcast() &&
         idx_id->hasExpandedExtent())) {
      out_domain.at(i) = IterDomainBuilder(idx_id).build();
    } else {
      out_domain.at(i) = IterDomainBuilder(inp_id).build();
    }
  }

  TensorView* out_tensor = IrBuilder::create<TensorView>(
      IrBuilder::create<TensorDomain>(
          out_domain, TensorDomain::getContiguityFilledWith(out_domain, true)),
      inp->getDataType().value());

  IrBuilder::create<TorchGatherOp>(out_tensor, inp, dim, index, true);

  return out_tensor->as<TensorView>();
}

} // namespace nvfuser
