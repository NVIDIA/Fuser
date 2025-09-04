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

TensorView* scatter(
    TensorView* self,
    int64_t dim,
    TensorView* index,
    Val* src,
    std::optional<BinaryOpType> accumulate_op) {
  auto self_dom = TensorDomain::noReductions(self->getLogicalDomain());
  auto idx_dom = TensorDomain::noReductions(index->getLogicalDomain());

  NVF_CHECK(!self_dom.empty(), "scatter can not be applied to 0d tensor.");
  NVF_CHECK(
      self_dom.size() == idx_dom.size() &&
          (!src->isA<TensorView>() ||
           self_dom.size() ==
               TensorDomain::noReductions(
                   src->as<TensorView>()->getLogicalDomain())
                   .size()),
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

  // Create the loop domain based on the logical domain of the index
  // tensor.
  std::vector<IterDomain*> out_loop;
  out_loop.reserve(idx_dom.size());
  std::ranges::transform(
      idx_dom, std::back_inserter(out_loop), [](IterDomain* id) {
        return IterDomainBuilder(id).build();
      });

  // Create the output tensor. The validation of the loop domain needs
  // to be skipped as it is not guaranteed to be equivalent to the
  // logical domain.
  TensorView* out_tensor = IrBuilder::create<TensorView>(
      IrBuilder::create<TensorDomain>(
          /*logical_domain=*/out_logical,
          /*loop_domain=*/out_loop,
          /*contiguity=*/
          TensorDomain::getContiguityFilledWith(out_logical, true),
          /*skip_loop_validation=*/true),
      self->getDataType().value());

  if (accumulate_op.has_value()) {
    NVF_ERROR(
        accumulate_op.value() == BinaryOpType::Add ||
            accumulate_op.value() == BinaryOpType::Mul ||
            accumulate_op.value() == BinaryOpType::Max ||
            accumulate_op.value() == BinaryOpType::Min,
        "Unsupported accumulation op: ",
        accumulate_op.value());
  }

  IrBuilder::create<ScatterOp>(
      out_tensor, self, dim, index, src, accumulate_op);

  return out_tensor->as<TensorView>();
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

TensorView* groupedBlockSfLayout(
    TensorView* input,
    TensorView* expert_offsets,
    TensorView* sf_offsets,
    BlockScalingFactorLayout layout) {
  // only support input matrix;
  auto input_logical_dom =
      TensorDomain::noReductions(input->getLogicalDomain());
  NVF_ERROR_EQ(input_logical_dom.size(), 2);

  // This is used for both root and loop domain on output
  // maps directly to input's logical domain.
  std::vector<IterDomain*> out_root;
  out_root.reserve(input_logical_dom.size());
  std::ranges::transform(
      input_logical_dom, std::back_inserter(out_root), [](IterDomain* id) {
        return IterDomainBuilder(id).build();
      });

  // Create the logical domain of output.
  // Note: output logical domain handles potential padding required for the
  // layout. Since the actual padding size is data-dependent, we allocate for
  // the maximum padding (reflected on logical/allocation domain).
  std::vector<IterDomain*> out_logical;
  out_logical.reserve(input_logical_dom.size());

  // only Block128x4 is supported at this point.
  NVF_CHECK_EQ(layout, BlockScalingFactorLayout::Block128x4);
  constexpr int col_multiple = 4;
  constexpr int row_multiple = 128;

  auto* one_val = input->fusion()->oneVal(DataType::Index);
  Val* num_groups =
      SimplifyingIrBuilder::subExpr(offset_logical_dom[0]->extent(), one_val);
  // padded row size:
  // num_groups * (row_multiple - 1) + row_size
  auto pad_to_max_extent = [&](IterDomain* id, int multiple) -> IterDomain* {
    auto* maximum_pad_value_per_group =
        IrBuilder::create<Val>(multiple - 1, DataType::Index);
    std::vector<IterDomain*> offset_logical_dom =
        TensorDomain::noReductions(expert_offsets->getLogicalDomain());
    Val* padded_ext = SimplifyingIrBuilder::addExpr(
        id->extent(),
        SimplifyingIrBuilder::mulExpr(num_groups, maximum_pad_value_per_group));
    return IterDomainBuilder(id).extent(padded_ext).build();
  };
  out_logical.push_back(pad_to_max_extent(out_root[0], row_multiple));

  // padded col size:
  // (col_size + col_multiple - 1) / col_multiple * col_multiple
  auto pad_to_multiple = [&](IterDomain* id, int multiple) -> IterDomain* {
    Val* ext = id->extent();
    auto* multiple_val = IrBuilder::create<Val>(multiple, DataType::Index);
    Val* padded_ext = SimplifyingIrBuilder::mulExpr(
        SimplifyingIrBuilder::divExpr(
            SimplifyingIrBuilder::subExpr(
                SimplifyingIrBuilder::addExpr(ext, multiple_val), one_val),
            multiple_val),
        multiple_val);
    return IterDomainBuilder(id).extent(padded_ext).build();
  };
  out_logical.push_back(pad_to_multiple(out_root[1], col_multiple));

  // Create the output tensor. Validation needs to be skipped, because
  // (root/loop) doesn't converge with (logical)
  TensorView* out_tv = IrBuilder::create<TensorView>(
      IrBuilder::create<TensorDomain>(
          /*root_domain=*/out_root,
          /*logical_domain=*/out_logical,
          /*allocation=*/std::vector<IterDomain*>(),
          /*loop_domain=*/out_root,
          /*alternate_loop_domain=*/std::nullopt,
          /*contiguity=*/
          TensorDomain::getContiguityFilledWith(out_logical, true),
          /*additional_ids=*/std::vector<IterDomain*>(),
          /*skip_checks=*/true),
      input->getDataType().value());

  std::vector<IterDomain*> offsets_logical_domain =
      TensorDomain::noReductions(sf_offsets->getLogicalDomain());
  IrBuilder::create<GroupedBlockScalingFactorLayoutOp>(
      out_tv,
      input,
      expert_offsets,
      sf_offsets,
      layout,
      input_logical_dom[1]->getMaybeExpandedExtent(),
      num_groups);

  return out_tv;
}

} // namespace nvfuser
