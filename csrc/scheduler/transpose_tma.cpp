// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on

#include "scheduler/transpose_tma.h"

#include "ir/utils.h"
#include "scheduler/runtime_info.h"
#include "scheduler/tools/domain_map.h"
#include "scheduler/tools/inlining.h"
#include "scheduler/utils.h"
#include "transform_replay.h"
#include "type.h"

namespace nvfuser {
namespace transpose {
namespace tma {
constexpr int64_t kBytesPerChunk = 16;

std::unique_ptr<TransposeParams> getTransposeHeuristics(
    Fusion* fusion,
    SchedulerRuntimeInfo& runtime_info,
    HeuristicDataCache* data_cache) {
  auto tparams = std::make_unique<TransposeParams>();
  tparams->tag = "TMA Transpose heuristics";
  tparams->cparams.index_type = runtime_info.getIndexType();
  tparams->use_tma_load = true;
  tparams->use_tma_store = false;

  int64_t max_input_dtype_size = 1;
  int64_t n_input = 0;
  for (auto inp : ir_utils::filterByType<TensorView>(fusion->inputs())) {
    max_input_dtype_size = std::max(
        max_input_dtype_size, dataTypeSizeByte(inp->getDataType().value()));
    n_input++;
  }
  tparams->tma_swizzle_bytes = 128;
  // input layout: [I2, I2] -> [tile1, tile2]
  // output layout: [I2, I2] -> [tile2, tile1]
  // tile2 is the inner most dim of the input tvs, it must equals to tma swizzle
  // bytes.
  tparams->tile_size2 = tparams->tma_swizzle_bytes / max_input_dtype_size;
  // [Tunable] tile1 is the inner most dim of the output tvs
  tparams->tile_size1 =
      (n_input == 1) ? tparams->tile_size2 * 2 : tparams->tile_size2;
  // [Tunable] In 128-bytes swizzled tma load, inner most dim is split into 8
  // chunks each with 16 bytes. Each thread many handle multiple chunks along
  // the inner most dim, range is [1, 8]
  // bdimx = tile_size1 * 8 / chunks_per_thread
  const int64_t target_bdimx = (n_input == 1) ? 256 : 128;
  tparams->chunks_per_thread = tparams->tile_size1 * 8 / target_bdimx;
  tparams->elements_per_chunk = kBytesPerChunk / max_input_dtype_size;
  return tparams;
}

void scheduleTranspose(Fusion* fusion, const TransposeParams* tparams) {
  FusionGuard fg(fusion);

  // Make sure we don't have global memory set on intermediate tensors from
  // fusion segmentation
  scheduler_utils::clearMemorySpace(fusion);

  // maybe has_reduction for scheduling should be done on a per output tensor
  // basis.
  NVF_ERROR(
      !ir_utils::hasAnyReductionOps(fusion),
      "This scheduler only handles pointwise ops.");

  // Cache inputs
  auto cached_inputs = scheduler_utils::cacheInputs(fusion, true);

  // Cache and fork outputs
  auto cached_outputs = scheduler_utils::cacheAndForkOutputs(fusion, true);

  scheduler_utils::prepareForMemoryTypePromotion(fusion);

  // always use TMA load for inputs
  int64_t max_input_dims = 0;
  TensorView* input_ref = nullptr;
  std::vector<TensorView*> tma_load_tvs;
  for (auto [cached_input, input_idx] : cached_inputs) {
    if (auto load_op = dynamic_cast<LoadStoreOp*>(cached_input->definition())) {
      load_op->setOpType(LoadStoreOpType::CpAsyncBulkTensorTile);
      cached_input->setMemoryType(MemoryType::Shared);
      tma_load_tvs.push_back(cached_input);
    }
    // find the output with the most logical dimensions
    if (scheduler_utils::nLogicalDims(cached_input) > max_input_dims) {
      max_input_dims = scheduler_utils::nLogicalDims(cached_input);
      input_ref = cached_input;
    }
  }
  NVF_ERROR(!tma_load_tvs.empty());

  // find the output with the most logical dimensions
  TensorView* output_ref = nullptr;
  int64_t max_output_dims = 0;
  std::vector<TensorView*> tma_store_tvs;
  std::vector<TensorView*> output_tvs;
  for (auto [cached_output, output_idx] : cached_outputs) {
    TensorView* output_reg_cache = nullptr;
    auto output = fusion->outputs()[output_idx]->as<TensorView>();
    output_tvs.push_back(output);
    if (tparams->use_tma_store) {
      output->definition()->as<LoadStoreOp>()->setOpType(
          LoadStoreOpType::CpAsyncBulkTensorTile);
      cached_output->setMemoryType(MemoryType::Shared);
      output_reg_cache = cached_output->cacheBefore();
      tma_store_tvs.push_back(cached_output);
    } else {
      output_reg_cache = cached_output;
    }
    // find the output with the most logical dimensions
    if (scheduler_utils::nLogicalDims(output_reg_cache) > max_output_dims) {
      max_output_dims = scheduler_utils::nLogicalDims(output_reg_cache);
      output_ref = output_reg_cache;
    }
  }
  if (max_output_dims == 0 && max_input_dims == 0) {
    return;
  }

  scheduler_tools::TransposeDomainMap domain_map(fusion);

  // Step 1: TMA tiling
  // output_ref has output layout [I2, I1], tile with [tile_2, tile_1]
  // input_ref has input layout [I1, I2], tile with [tile_1, tile_2]
  // Target loop domain: [BIDx, tile_2, tile_1]
  auto output_inner_id = scheduler_utils::innerMostAllocDim(output_ref);
  auto input_inner_id = scheduler_utils::innerMostAllocDim(input_ref);
  int64_t inner_pos = domain_map.getInnerLeafDim(output_ref, output_inner_id);
  int64_t outer_pos = domain_map.getInnerLeafDim(output_ref, input_inner_id);
  NVF_ERROR(
      inner_pos >= 0 && outer_pos >= 0 && inner_pos != outer_pos,
      "Invalid inner/outer positions for TMA tiling");
  // [..., I2, ..., I1, ...]
  output_ref->split(inner_pos, tparams->tile_size1);
  output_ref->reorder({{inner_pos + 1, -1}});
  // [..., I2, ..., I1/tile_1, ..., tile_1]
  output_ref->split(outer_pos, tparams->tile_size2);
  output_ref->reorder({{outer_pos + 1, -2}});
  // [..., I2/tile_2, ..., I1/tile_1, ..., tile_2, tile_1]
  // merge all non-tiled dimensions except reduction and device dimensions
  int64_t rhs_i = output_ref->nDims() - 3;
  for (int64_t lhs_i = output_ref->nDims() - 4; lhs_i >= 0; lhs_i--) {
    if (output_ref->axis(lhs_i)->isReduction() ||
        output_ref->axis(lhs_i)->isDeviceDim()) {
      continue;
    }
    if (output_ref->axis(rhs_i)->isReduction() ||
        output_ref->axis(rhs_i)->isDeviceDim()) {
      rhs_i = lhs_i;
      continue;
    }
    output_ref->merge(lhs_i, rhs_i);
    rhs_i = lhs_i;
  }
  // [I1/tile_1 * I2/tile_2, tile_2, tile_1]
  output_ref->axis(rhs_i)->parallelize(ParallelType::BIDx);
  // [BIDx, tile_2, tile_1]

  TransformPropagator propagator(output_ref);
  MaxLogicalDomainInfoSpanningTree entire_dag(output_ref);
  entire_dag.traverse(&propagator);
  scheduler_utils::parallelizeAllLike(
      output_ref,
      /*selected_tvs=*/{},
      /*selected_parallel_types=*/{},
      /*propagate_padding=*/true,
      /*parallelize_inputs_on_did=*/true);
  // After propagation, all tvs have loop domain: [BIDx, tile_2, tile_1].

  // Step 2: Schedule output TMA store (Bulk parallel on tile dims).
  if (tparams->use_tma_store) {
    for (auto output_smem_cache : tma_store_tvs) {
      output_smem_cache->setAllocationDomain(
          output_smem_cache->getLoopDomain(), true);
    }
    for (auto output : output_tvs) {
      output->axis(-1)->parallelize(ParallelType::Bulk);
      output->axis(-2)->parallelize(ParallelType::Bulk);
    }
  }

  // Step 3: Schedule input shared memory with XOR swizzle.
  // input_smem_cache follows input layout [I1, I2]. Reorder so tile_2
  // (the I2/contiguous-in-input dim) is innermost before applying swizzle.
  // [BIDx, tile_2, tile_1]
  int64_t num_swizzle_chunks = tparams->tma_swizzle_bytes / kBytesPerChunk;
  for (auto input_smem_cache : tma_load_tvs) {
    input_smem_cache->reorder({{-1, -2}});
    // [BIDx, tile_1, tile_2]
    input_smem_cache->split(-1, tparams->elements_per_chunk);
    // [BIDx, tile_1, tile_2/chunk, chunk]
    input_smem_cache->split(-3, num_swizzle_chunks);
    // [BIDx, tile_1/S, S, tile_2/chunk, chunk] where S = num_swizzle_chunks
    input_smem_cache->swizzle(SwizzleType::XOR, -3, -2);
    input_smem_cache->axis(-1)->parallelize(ParallelType::Bulk);
    input_smem_cache->axis(-2)->parallelize(ParallelType::Bulk);
    input_smem_cache->axis(-3)->parallelize(ParallelType::Bulk);
    input_smem_cache->axis(-4)->parallelize(ParallelType::Bulk);
    input_smem_cache->setAllocationDomain(
        input_smem_cache->getLoopDomain(), true);
  }
  // Step 4: Schedule register tvs for per-thread access pattern.
  // Each thread handles multiple chunks of tile_2, which corresponds to
  // multiple rows of output and multiple columns of input.
  // [BIDx, tile_2, tile_1]
  output_ref->split(-2, tparams->elements_per_chunk);
  // [BIDx, tile_2/chunk, chunk, tile_1]
  output_ref->split(-3, tparams->chunks_per_thread);
  // [BIDx, tile_2/chunk/cpt, cpt, chunk, tile_1]
  output_ref->merge(-4, -1);
  // [BIDx, tile_2/chunk/cpt * tile_1, cpt, chunk]
  output_ref->axis(-3)->parallelize(ParallelType::TIDx);
  // [BIDx, TIDx, cpt, chunk]
  output_ref->axis(-1)->parallelize(ParallelType::Unroll);

  {
    // Propagate Step 4 transforms to all tvs except those already
    // independently scheduled (input_smem_cache and TMA store output).
    std::unordered_set<TensorView*> skip_tvs{
        tma_load_tvs.begin(), tma_load_tvs.end()};
    if (tparams->use_tma_store) {
      skip_tvs.insert(output_tvs.begin(), output_tvs.end());
    }
    auto propagate_tvs = ir_utils::allTvsExcept(fusion, skip_tvs);
    std::unordered_set<TensorView*> propagate_tvs_set(
        propagate_tvs.begin(), propagate_tvs.end());
    SetSelector selector(propagate_tvs_set);
    MaxLogicalDomainInfoSpanningTree propagate_dag(output_ref, &selector);
    TransformPropagator propagator(output_ref);
    propagate_dag.traverse(&propagator);
    scheduler_utils::parallelizeAllLike(
        output_ref,
        propagate_tvs,
        {},
        /*propagate_padding=*/true,
        /*parallelize_inputs_on_did=*/true);
  }
  for (auto tma_load_tv : tma_load_tvs) {
    for (auto consumer : ir_utils::consumerTvsOf(tma_load_tv)) {
      consumer->axis(-1)->parallelize(ParallelType::Vectorize);
    }
  }

  inlineMost();
}

} // namespace tma
} // namespace transpose
} // namespace nvfuser
