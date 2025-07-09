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
#include <c10/util/Half.h>

namespace nvfuser {

TensorView* select(TensorView* tv, int64_t dim, Val* index) {
  auto dom = TensorDomain::noReductions(tv->getLogicalDomain());
  NVF_CHECK(!dom.empty(), "select can not be applied to 0d tensor.");

  std::vector<IterDomain*> new_root;
  new_root.reserve(dom.size() - 1);
  dim = wrapDim(dim, (int64_t)dom.size());

  for (auto i : arange((int64_t)dom.size())) {
    if (i != dim) {
      new_root.emplace_back(dom[i]->cloneWithoutRFactor());
    }
  }

  auto td = IrBuilder::create<TensorDomain>(
      new_root, TensorDomain::getContiguityFilledWith(new_root, true));
  auto out = IrBuilder::create<TensorView>(td, *tv->getDataType());
  IrBuilder::create<SelectOp>(out, tv, dim, index);
  return out;
}

// torch.index_select
TensorView* indexSelect(
    TensorView* lookup_tv,
    int64_t dim,
    TensorView* index_tv) {
  DataType dtype = lookup_tv->getDataType().value();
  NVF_CHECK(
      dtype != DataType::Null, "Invalid datatype provided for new value.");

  std::vector<IterDomain*> lookup_domain =
      TensorDomain::noReductions(lookup_tv->getLogicalDomain());

  int64_t n_dims = (int64_t)lookup_domain.size();
  dim = wrapDim(dim, n_dims);
  NVF_CHECK(
      n_dims > 0, "lookup_tv argument for indexSelect cannot be a 0-D tensor.");

  std::vector<IterDomain*> original_index_domain =
      TensorDomain::noReductions(index_tv->getLogicalDomain());

  // short-circuit: index_tv is a scalar tensor.
  if (original_index_domain.empty()) {
    TensorView* select_tv = select(lookup_tv, dim, index_tv);
    return unsqueeze(select_tv, dim);
  }

  if (!ops::isIndexAlreadyBroadcast(original_index_domain, dim, n_dims)) {
    // Broadcast index to lookup's rank.
    NVF_CHECK(
        original_index_domain.size() <= 1,
        "index_tv must be a 1d or scalar tensor.");
    index_tv =
        ops::maybeBroadcastIndexTv(index_tv->as<TensorView>(), dim, n_dims);
  }

  // create logical domain for output tensorview.
  std::vector<IterDomain*> new_logical;
  new_logical.reserve(n_dims);
  for (auto i : arange(n_dims)) {
    if (i != dim) {
      new_logical.emplace_back(lookup_domain.at(i)->cloneWithoutRFactor());
    } else {
      // Get new domain because maybeBroadcastIndexTv could have create a new
      // TensorView.
      std::vector<IterDomain*> index_domain =
          TensorDomain::noReductions(index_tv->getLogicalDomain());
      // Select the index for desired dimension.
      new_logical.emplace_back(index_domain.at(dim)->cloneWithoutRFactor());
    }
  }
  auto td = IrBuilder::create<TensorDomain>(
      new_logical, TensorDomain::getContiguityFilledWith(new_logical, true));
  auto out = IrBuilder::create<TensorView>(td, dtype);

  // create index_select expression
  IrBuilder::create<IndexSelectOp>(out, lookup_tv, dim, index_tv);
  return out;
}

// This is a restricted version of PyTorch's Tensor.index_put(indices,
// values, accumulate=true). We only support a 1-D index tensor at
// this moment. The 1-D index tensor is assumed to index dimension
// 0. The shape of the value tensor must be
// [index_tv->axis(0)->extent(), acc_tv->axis(1)->extent(),
// acc_tv->axis(2)->extent(), ...].
TensorView* indexPutAccumulate(
    TensorView* acc_tv,
    TensorView* index_tv,
    TensorView* value_tv) {
  DataType dtype = acc_tv->getDataType().value();
  NVF_CHECK(
      dtype != DataType::Null, "Invalid datatype provided for new value.");

  std::vector<IterDomain*> acc_domain =
      TensorDomain::noReductions(acc_tv->getLogicalDomain());
  std::vector<IterDomain*> index_domain =
      TensorDomain::noReductions(index_tv->getLogicalDomain());
  std::vector<IterDomain*> value_domain =
      TensorDomain::noReductions(value_tv->getLogicalDomain());

  // If acc_tv is a zero tensor and the ID of index_tv is a broadcast,
  // just scattering is sufficient
  auto is_zero_tensor = [](TensorView* tv) -> bool {
    auto full_op = dynamic_cast<FullOp*>(tv->definition());
    if (full_op == nullptr) {
      return false;
    }
    return full_op->getFillValue()->isZero();
  };
  if ((index_domain.at(0)->isBroadcast() ||
       index_domain.at(0)->extent()->isOne()) &&
      is_zero_tensor(acc_tv)) {
    return scatter(acc_tv, 0, index_tv, value_tv);
  }

  NVF_CHECK(acc_domain.size() == value_domain.size());
  NVF_CHECK(index_domain.size() == 1);

  auto* out = ops::newValLike(acc_tv, dtype)->as<TensorView>();
  IrBuilder::create<IndexPutAccumulateOp>(out, acc_tv, index_tv, value_tv);
  return out;
}

// torch.gather
TensorView* gather(TensorView* inp, int64_t dim, TensorView* index) {
  auto inp_domain = TensorDomain::noReductions(inp->getLogicalDomain());
  auto idx_domain = TensorDomain::noReductions(index->getLogicalDomain());
  NVF_CHECK(
      !inp_domain.empty(), "torch.gather can not be applied to 0d tensor.");
  NVF_CHECK(
      idx_domain.size() == inp_domain.size(),
      "the input and index tensor must have the same dimensions for "
      "torch.gather");
  dim = wrapDim(dim, (int64_t)idx_domain.size());

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

  IrBuilder::create<GatherOp>(out_tensor, inp, dim, index, false);
  return out_tensor->as<TensorView>();
}

TensorView* scatterOp(
    ScatterOpType type,
    TensorView* self,
    int64_t dim,
    TensorView* index,
    TensorView* src) {
  auto self_dom = TensorDomain::noReductions(self->getLogicalDomain());
  auto idx_dom = TensorDomain::noReductions(index->getLogicalDomain());
  auto src_dom = TensorDomain::noReductions(src->getLogicalDomain());

  NVF_CHECK(!self_dom.empty(), "scatter can not be applied to 0d tensor.");
  NVF_CHECK(
      self_dom.size() == idx_dom.size() && self_dom.size() == src_dom.size(),
      "self, index and src tensor should all have the same number of "
      "dimensions in scatter like ops.");
  dim = wrapDim(dim, (int64_t)self_dom.size());

  // The shape of output tensor is same as self tensor.
  std::vector<IterDomain*> out_logical;
  for (const auto i : arange(self_dom.size())) {
    out_logical.push_back(
        IterDomainBuilder(self_dom[i])
            .iter_type(
                self_dom[i]->getIterType() == IterType::Iteration
                    ? IterType::GatherScatter
                    : self_dom[i]->getIterType())
            .build());
  }

  TensorView* out_tensor = IrBuilder::create<TensorView>(
      IrBuilder::create<TensorDomain>(
          out_logical,
          TensorDomain::getContiguityFilledWith(out_logical, true)),
      self->getDataType().value());

  // Set the loop domain same as the logical domain of the index
  // tensor.
  // Note: the loop IDs are disconnected from the logical IDs. Revisit
  // if they should be connected with some IterDomain exprs.
  std::vector<IterDomain*> out_loop;
  out_loop.reserve(idx_dom.size());
  std::ranges::transform(
      idx_dom, std::back_inserter(out_loop), [](IterDomain* id) {
        return IterDomainBuilder(id).build();
      });
  out_tensor->domain()->setLoopDomain(
      out_loop,
      /*skip_validation=*/true);

  IrBuilder::create<ScatterOp>(type, out_tensor, self, dim, index, src);
  return out_tensor->as<TensorView>();
}

TensorView* scatter(
    TensorView* self,
    int64_t dim,
    TensorView* index,
    TensorView* src) {
  return scatterOp(ScatterOpType::Set, self, dim, index, src);
}

TensorView* takeAlongAxis(TensorView* inp, TensorView* index, int64_t dim) {
  const auto inp_domain = TensorDomain::noReductions(inp->getLogicalDomain());
  const auto idx_domain = TensorDomain::noReductions(index->getLogicalDomain());

  NVF_CHECK(
      !inp_domain.empty(), "take_along_axis can not be applied to 0d tensor.");
  NVF_CHECK(
      idx_domain.size() == inp_domain.size(),
      "The input and index tensor must have the same dimensions for "
      "take_along_axis");

  dim = wrapDim(dim, (int64_t)idx_domain.size());

  std::vector<IterDomain*> out_domain(idx_domain.size());

  for (const auto i : arange(idx_domain.size())) {
    auto inp_id = inp_domain.at(i);
    auto idx_id = idx_domain.at(i);

    NVF_CHECK(
        !inp_id->maybePartial(),
        "Partial domain not supported: ",
        inp_id->toString());
    NVF_CHECK(
        !idx_id->maybePartial(),
        "Partial domain not supported: ",
        idx_id->toString());

    NVF_CHECK(
        inp_id->getIterType() != IterType::Iteration ||
            inp_id->getIterType() != IterType::Broadcast,
        "Unsupported IterType of an input domian: ",
        inp_id->toString());
    NVF_CHECK(
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

  IrBuilder::create<GatherOp>(out_tensor, inp, dim, index, true);

  return out_tensor->as<TensorView>();
}

} // namespace nvfuser
