// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#include <index_compute.h>

#include <ranges>

#include <ATen/cuda/CUDAContext.h>

#include <device_lower/lower2device.h>
#include <instrumentation.h>
#include <ir/all_nodes.h>
#include <ir/builder.h>
#include <ir/utils.h>
#include <ops/arith.h>

namespace nvfuser {

namespace {

Val* sumVals(std::vector<Val*> vals) {
  Val* result_index = GpuLower::current()->kernel()->zeroVal();
  for (auto v : vals) {
    result_index = SimplifyingIrBuilder::addExpr(result_index, v);
  }
  return result_index;
}

} // namespace

Val* Index::getLinearLogicalIndex(
    TensorView* consumer_tv,
    const std::vector<kir::ForLoop*>& loops) {
  NVF_ERROR(
      GpuLower::current()->idModelOptions().isTensorIndexerEnabled(),
      "Legacy indexer no longer available");

  const TensorIndexer& indexer = GpuLower::current()->tensorIndexer();
  auto per_dim_indices = indexer.getIndexFor(
      consumer_tv->definition(),
      /*as_consumer=*/true,
      consumer_tv->getLogicalDomain(),
      loops,
      /*use_magic_zero=*/true);
  Val* stride = consumer_tv->fusion()->oneVal();
  for (const auto [i, logical_id] :
       enumerate(consumer_tv->getLogicalDomain()) | std::views::reverse) {
    auto per_dim_index = per_dim_indices.at(i);
    auto per_dim_strided_index =
        SimplifyingIrBuilder::mulExpr(per_dim_index, stride);
    per_dim_indices.at(i) = per_dim_strided_index;
    stride = SimplifyingIrBuilder::mulExpr(stride, logical_id->extent());
  }
  return sumVals(per_dim_indices);
}

std::vector<Val*> Index::getConsumerPerDimLogicalIndex(
    TensorView* consumer_tv,
    const std::vector<kir::ForLoop*>& loops) {
  NVF_ERROR(
      GpuLower::current()->idModelOptions().isTensorIndexerEnabled(),
      "Legacy indexer no longer available");

  const TensorIndexer& indexer = GpuLower::current()->tensorIndexer();
  return indexer.getIndexFor(
      consumer_tv->definition(),
      /*as_consumer=*/true,
      consumer_tv->getLogicalDomain(),
      loops);
}

std::vector<Val*> Index::getProducerPerDimLogicalIndex(
    TensorView* producer_tv,
    const TensorView* consumer_tv,
    const std::vector<kir::ForLoop*>& loops,
    const std::unordered_map<IterDomain*, Val*>& override_index) {
  NVF_ERROR(
      GpuLower::current()->idModelOptions().isTensorIndexerEnabled(),
      "Legacy indexer no longer available");

  const TensorIndexer& indexer = GpuLower::current()->tensorIndexer();
  return indexer.getIndexFor(
      consumer_tv->definition(),
      /*as_consumer=*/false,
      producer_tv->getLogicalDomain(),
      loops);
}

// Producer is the inputs of an expression
kir::TensorIndex* Index::getProducerIndex(
    TensorView* producer,
    const TensorView* consumer,
    const std::vector<kir::ForLoop*>& loops,
    const std::unordered_map<IterDomain*, Val*>& override_index,
    bool generate_pointer,
    DataType as_type,
    bool ld_st_matrix) {
  NVF_ERROR(
      GpuLower::current()->idModelOptions().isTensorIndexerEnabled(),
      "Legacy indexer no longer available");

  Val* index = GpuLower::current()->tensorIndexer().getLinearIndex(
      producer, consumer->definition(), loops, override_index, ld_st_matrix);
  if (generate_pointer) {
    auto address_offset = index;
    if (producer->getMemoryType() == MemoryType::Shared) {
      auto producer_dt = producer->getDataType();
      auto index_dt = index->getDataType();
      address_offset = SimplifyingIrBuilder::mulExpr(
          address_offset,
          IrBuilder::create<Val>(dataTypeSizeByte(producer_dt), index_dt));
    }
    index = SimplifyingIrBuilder::addExpr(
        IrBuilder::baseAddressExpr(producer), address_offset);
  }

  index = GpuLower::current()->commonScalarMap().hoistScalar(index, loops);
  if (ir_utils::isLdMatrixOp(consumer->definition()) &&
      at::cuda::getCurrentDeviceProperties()->major < 8) {
    auto items_per_thread = ir_utils::getVectorizeSize(consumer);
    if (items_per_thread != 8) {
      // For Turing, unused indices for ldmatrix needs to be aligned, although
      // they are not used.
      auto orig_index = index;
      index = IrBuilder::create<Val>(index->dtype());
      UnaryOpType op = UnaryOpType::Print;
      if (items_per_thread == 2) {
        op = UnaryOpType::AdjustPartialLdMatrixAddrInTuring8;
      } else if (items_per_thread == 4) {
        op = UnaryOpType::AdjustPartialLdMatrixAddrInTuring16;
      } else {
        NVF_THROW(
            "Unexpected output vectorizaiton for ldmatrix, expect 2, 4, or 8, "
            "get ",
            items_per_thread);
      }
      IrBuilder::create<UnaryOp>(op, index, orig_index);
    }
  }
  return IrBuilder::create<kir::TensorIndex>(producer, index, as_type);
}

// Consumer is the output of an expression
kir::TensorIndex* Index::getConsumerIndex(
    TensorView* consumer,
    const std::vector<kir::ForLoop*>& loops,
    const std::unordered_map<IterDomain*, Val*>& override_index,
    bool generate_pointer,
    DataType as_type,
    bool ld_st_matrix) {
  NVF_ERROR(
      GpuLower::current()->idModelOptions().isTensorIndexerEnabled(),
      "Legacy indexer no longer available");

  Val* index = GpuLower::current()->tensorIndexer().getLinearIndex(
      consumer, consumer->definition(), loops, override_index, ld_st_matrix);
  if (generate_pointer) {
    auto address_offset = index;
    if (consumer->getMemoryType() == MemoryType::Shared) {
      auto consumer_dt = consumer->getDataType();
      auto index_dt = index->getDataType();
      address_offset = SimplifyingIrBuilder::mulExpr(
          index,
          IrBuilder::create<Val>(dataTypeSizeByte(consumer_dt), index_dt));
    }
    index = SimplifyingIrBuilder::addExpr(
        IrBuilder::baseAddressExpr(consumer), address_offset);
  }

  index = GpuLower::current()->commonScalarMap().hoistScalar(index, loops);
  return SimplifyingIrBuilder::create<kir::TensorIndex>(
      consumer, index, as_type);
}

PredicateInfo PredicateInfo::getFalseInfo() {
  PredicateInfo info;
  info.start_predicate_ = GpuLower::current()->kernel()->falseVal();
  info.stop_predicate_ = GpuLower::current()->kernel()->falseVal();

  return info;
}

Val* Index::iota(
    TensorView* consumer_tv,
    const std::vector<kir::ForLoop*>& loops,
    Val* start,
    Val* step,
    DataType dtype) {
  auto linear_index = Index::getLinearLogicalIndex(consumer_tv, loops);
  auto result = add(start, mul(step, linear_index));
  return GpuLower::current()->commonScalarMap().hoistScalar(result, loops);
}

Val* Index::eye(
    TensorView* consumer_tv,
    const std::vector<kir::ForLoop*>& loops,
    DataType dtype) {
  auto indices = Index::getConsumerPerDimLogicalIndex(consumer_tv, loops);
  NVF_ERROR(indices.size() == 2);
  auto result = maybeCastOp(dtype, eq(indices[0], indices[1]));
  return GpuLower::current()->commonScalarMap().hoistScalar(result, loops);
}

std::pair<Val*, Val*> Index::getCpAsyncBulkGmemIndex(
    const LoadStoreOp* ldst,
    Val* mbarrier,
    const std::vector<kir::ForLoop*>& loops) {
  FUSER_PERF_SCOPE("Index::getCpAsyncBulkGmemIndex");

  auto* producer_tv = ldst->in()->as<TensorView>();
  auto* consumer_tv = ldst->out()->as<TensorView>();

  bool is_load = false;
  TensorView* gmem_tv = nullptr;
  if (producer_tv->getMemoryType() == MemoryType::Shared) {
    NVF_ERROR(consumer_tv->getMemoryType() == MemoryType::Global);
    gmem_tv = consumer_tv;
    is_load = false;
  } else {
    NVF_ERROR(producer_tv->getMemoryType() == MemoryType::Global);
    NVF_ERROR(consumer_tv->getMemoryType() == MemoryType::Shared);
    gmem_tv = producer_tv;
    is_load = true;
  }

  NVF_ERROR(
      GpuLower::current()->consumerToTMAInfo().count(consumer_tv),
      "Unable to find TMA info for consumer_tv: ",
      consumer_tv->toString());
  const TMAInfo& tma_info =
      GpuLower::current()->consumerToTMAInfo().at(consumer_tv);
  int64_t dim = (int64_t)tma_info.dims().size();
  Val* expected_bytes = SimplifyingIrBuilder::maybeCastExpr(
      DataType::UInt32, tma_info.tileSizeBytes());
  expected_bytes =
      GpuLower::current()->commonScalarMap().hoistScalar(expected_bytes, loops);
  Val* index = nullptr;

  // 1D TMA without tensor map
  if (ldst->opType() == LoadStoreOpType::CpAsyncBulk) {
    if (is_load) {
      std::stringstream ss;
      ss << "Hopper::CpAsyncBulkG2SIndex";
      auto gmem_address =
          getProducerIndex(producer_tv, consumer_tv, loops, {}, true);
      index = IrBuilder::structExpr(
          {{"raw_gmem_addr", gmem_address},
           {"bytes", expected_bytes},
           {"mbarrier", mbarrier}},
          ss.str());
    } else {
      std::stringstream ss;
      ss << "Hopper::CpAsyncBulkS2GIndex";
      auto gmem_address =
          getConsumerIndex(consumer_tv, loops, {}, true);
      index = IrBuilder::structExpr(
          {{"raw_gmem_addr", gmem_address}, {"bytes", expected_bytes}},
          ss.str());
    }
  } else {
    // ND TMA with tensor map
    ValGroups groups_to_index = tma_info.getTMADomain();
    // TensorIndexer needs IterDomain instead of ValGroup to work around
    // the resize indexing issue
    std::vector<IterDomain*> ids_to_index;
    ids_to_index.reserve(groups_to_index.size());
    const auto tma_all_ids = is_load ? consumer_tv->domain()->allIDs()
                                     : producer_tv->domain()->allIDs();
    for (const auto& group : groups_to_index) {
      auto it = std::ranges::find_if(tma_all_ids, [&](IterDomain* gmem_id) {
        return group->has(gmem_id);
      });
      if (it != tma_all_ids.end()) {
        ids_to_index.push_back(*it);
      } else {
        ids_to_index.push_back(group->front()->as<IterDomain>());
      }
    }

    const TensorIndexer& indexer = GpuLower::current()->tensorIndexer();
    auto indices_inner_to_outer =
        indexer.getIndexFor(ldst, !is_load, ids_to_index, loops);

    // These are the box coordinates of the TMA box, which must be of type
    // int32_t. Possible overflow in each of these dims should be checked
    // elsewhere.
    for (size_t i : arange(indices_inner_to_outer.size())) {
      indices_inner_to_outer[i] =
          IrBuilder::maybeCastExpr(DataType::Int32, indices_inner_to_outer[i]);
    }

    auto coordinate = IrBuilder::arrayExpr(indices_inner_to_outer);
    auto descriptor = tma_info.tensorMap();
    if (is_load) {
      std::stringstream ss;
      ss << "Hopper::CpAsyncBulkTensorTileG2SIndex<" << dim << ">";
      index = IrBuilder::structExpr(
          {{"descriptor", IrBuilder::addressExpr(descriptor)},
           {"coordinate", coordinate},
           {"mbarrier", mbarrier}},
          ss.str());
    } else {
      std::stringstream ss;
      ss << "Hopper::CpAsyncBulkTensorTileS2GIndex<" << dim << ">";
      index = IrBuilder::structExpr(
          {{"descriptor", IrBuilder::addressExpr(descriptor)},
           {"coordinate", coordinate}},
          ss.str());
    }
  }

  index = GpuLower::current()->commonScalarMap().hoistScalar(index, loops);

  auto is_multiple_of_16B = SimplifyingIrBuilder::eqExpr(
      SimplifyingIrBuilder::modExpr(
          expected_bytes, IrBuilder::create<Val>(16, DataType::Index)),
      expected_bytes->fusion()->zeroVal());
  GpuLower::current()->validate(
      is_multiple_of_16B,
      "The expected bytes must be a multiple of 16 bytes, but ",
      expected_bytes->toInlineString(),
      " is not.");

  return {IrBuilder::create<kir::TensorIndex>(gmem_tv, index), expected_bytes};
}

} // namespace nvfuser
