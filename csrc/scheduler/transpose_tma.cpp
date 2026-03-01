// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on

#include "scheduler/transpose_tma.h"

#include <ATen/cuda/CUDAContext.h>
#include "ir/utils.h"
#include "scheduler/mma_utils.h"
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
constexpr int64_t kTmaSwizzleBytes = 128;
std::unique_ptr<TransposeParams> getTransposeHeuristics(
    Fusion* fusion,
    SchedulerRuntimeInfo& runtime_info,
    HeuristicDataCache* data_cache) {
  auto tparams = std::make_unique<TransposeParams>();
  tparams->tag = "TMA Transpose heuristics";
  tparams->cparams.index_type = runtime_info.getIndexType();

  int64_t max_input_dtype_size = 1;
  int64_t n_input = 0;
  for (auto inp : ir_utils::filterByType<TensorView>(fusion->inputs())) {
    max_input_dtype_size = std::max(
        max_input_dtype_size,
        dataTypeSizeByte(valueOrError(inp->getDataType())));
    n_input++;
  }

  int64_t max_output_dtype_size = 1;
  int64_t n_output = 0;
  for (auto out : ir_utils::filterByType<TensorView>(fusion->outputs())) {
    max_output_dtype_size = std::max(
        max_output_dtype_size,
        dataTypeSizeByte(valueOrError(out->getDataType())));
    n_output++;
  }

  // Choose between input smem transpose and output smem transpose.
  // Input smem transpose: transpose happens when reading from swizzled input
  // shared memory to registers. columns to rows.
  // Output smem transpose: transpose happens when writing from registers to
  // swizzled output shared memory. rows to columns.
  //
  // Use input smem transpose when inputs <= outputs to reduce smem usage and
  // swizzle cost (fewer inputs to swizzle). Use output smem transpose when
  // inputs > outputs for the same reason (fewer outputs to swizzle).
  //
  // TMA load/store are independent of the transpose direction:
  // - TMA load stages inputs in shared memory (always beneficial for input
  //   smem transpose since inputs need swizzled smem anyway).
  // - TMA store writes outputs from shared memory (always beneficial for
  //   output smem transpose since outputs need swizzled smem anyway).
  tparams->is_output_smem_transpose = n_input > n_output;
  tparams->use_tma_load = true;
  tparams->use_tma_store = tparams->is_output_smem_transpose;

  // Inputs and outputs are grouped into two groups based on their inner most
  // dim. The group with smaller number of tvs is swizzled in shared memory. TMA
  // scheduler assumes all inputs are in the same group and all outputs are in
  // the same group.
  // tile_size2 is the tile size for the inner most dim of the group with
  // swizzled smem (group2), it should follow the restriction of TMA swizzle
  // size.
  int64_t swizzle_dtype_size = tparams->is_output_smem_transpose
      ? max_output_dtype_size
      : max_input_dtype_size;
  int64_t constrained_tile = kTmaSwizzleBytes / swizzle_dtype_size;
  tparams->tile_size2 = constrained_tile;

  // Fixed, 16 bytes per chunk for swizzle, 4 float32 or 8 float16 elements.
  tparams->elements_per_chunk = kBytesPerChunk / swizzle_dtype_size;

  // free dim that can be tuned, increase tile size when input count is small
  // assuming issue 64KB loading data per sm, 256 threads per cta.
  auto dev_props = at::cuda::getCurrentDeviceProperties();
  constexpr int64_t bytes_per_sm = 64 * 1024;
  constexpr int64_t threads_per_cta = 256;
  const int64_t cta_per_sm =
      dev_props->maxThreadsPerMultiProcessor / threads_per_cta;
  const int64_t bytes_per_cta = bytes_per_sm / cta_per_sm;
  const int64_t bytes_per_tile = bytes_per_cta / n_input;
  int64_t estimated_tile_size1 = bytes_per_tile / kTmaSwizzleBytes;

  // tile1 * tile2 = elements_per_chunk * chunks_per_thread * threads_per_cta
  // May further increase tile1 to ensure each thread has at least 4 chunks to
  // process for better efficiency.
  constexpr int64_t min_chunks_per_thread = 4;
  auto get_chunks_per_thread = [&]() {
    int64_t elements_per_thread =
        estimated_tile_size1 * tparams->tile_size2 / threads_per_cta;
    return elements_per_thread / tparams->elements_per_chunk;
  };
  while (get_chunks_per_thread() < min_chunks_per_thread) {
    estimated_tile_size1 *= 2;
  }
  tparams->tile_size1 = estimated_tile_size1;
  tparams->chunks_per_thread = get_chunks_per_thread();

  if (isDebugDumpEnabled(DebugDumpOption::SchedulerDebug)) {
    debug() << "\n===== TMA Transpose Stats ========\n"
            << "inputs: " << ir_utils::toString(fusion->inputs()) << "\n"
            << "outputs: " << ir_utils::toString(fusion->outputs()) << "\n"
            << "is_output_smem_transpose: " << tparams->is_output_smem_transpose
            << "\n"
            << "use_tma_load: " << tparams->use_tma_load << "\n"
            << "use_tma_store: " << tparams->use_tma_store << "\n"
            << "tile_size1: " << tparams->tile_size1 << "\n"
            << "tile_size2: " << tparams->tile_size2 << "\n"
            << "chunks_per_thread: " << tparams->chunks_per_thread << "\n"
            << "elements_per_chunk: " << tparams->elements_per_chunk << "\n"
            << "\n";
  }
  return tparams;
}

void scheduleTranspose(Fusion* fusion, const TransposeParams* tparams) {
  FusionGuard fg(fusion);

  scheduler_utils::clearMemorySpace(fusion);

  NVF_ERROR(
      !ir_utils::hasAnyReductionOps(fusion),
      "This scheduler only handles pointwise ops.");

  auto cached_inputs = scheduler_utils::cacheInputs(fusion, true);
  auto cached_outputs = scheduler_utils::cacheAndForkOutputs(fusion, true);

  scheduler_utils::prepareForMemoryTypePromotion(fusion);

  // Set up TMA load for inputs.
  std::vector<TensorView*> tma_load_tvs;
  if (tparams->use_tma_load) {
    for (auto [cached_input, input_idx] : cached_inputs) {
      auto load_op = dynamic_cast<LoadStoreOp*>(cached_input->definition());
      if (load_op == nullptr) {
        continue;
      }
      load_op->setOpType(LoadStoreOpType::CpAsyncBulkTensorTile);
      cached_input->setMemoryType(MemoryType::Shared);
      cached_input->cacheAfter();
      tma_load_tvs.push_back(cached_input);
    }
  }

  // Collect output TVs.
  std::vector<TensorView*> output_tvs;
  output_tvs.reserve(cached_outputs.size());
  for (auto [cached_output, output_idx] : cached_outputs) {
    output_tvs.push_back(fusion->outputs()[output_idx]->as<TensorView>());
  }

  // Set up output caching with TMA store when enabled.
  std::vector<TensorView*> tma_store_tvs;
  if (tparams->use_tma_store) {
    for (auto [cached_output, output_idx] : cached_outputs) {
      auto output = fusion->outputs()[output_idx]->as<TensorView>();
      output->definition()->as<LoadStoreOp>()->setOpType(
          LoadStoreOpType::CpAsyncBulkTensorTile);
      cached_output->setMemoryType(MemoryType::Shared);
      cached_output->cacheBefore();
      tma_store_tvs.push_back(cached_output);
    }
  }

  // Find transposed ids and positions, two groups, transpose happens in
  // group-2's cached smem.
  scheduler_tools::TransposeDomainMap domain_map(fusion);
  auto grouped_inputs_outputs = domain_map.groupInputsOutputsByInnerDim();
  NVF_ERROR(grouped_inputs_outputs.size() >= 2);

  // When there are more inputs than outputs, output smem transpose should be
  // used, however, if it is not, then input smem tranpose will be used, to
  // ensure group2 is always the one that is transposed, we should swap group1
  // and group2.
  if (!tparams->is_output_smem_transpose &&
      cached_inputs.size() > cached_outputs.size()) {
    std::swap(grouped_inputs_outputs[0], grouped_inputs_outputs[1]);
  }

  TensorView* reference1 =
      domain_map.findReferenceFor(grouped_inputs_outputs[0]);
  TensorView* reference2 =
      domain_map.findReferenceFor(grouped_inputs_outputs[1]);
  NVF_ERROR(
      reference1 != nullptr, "Unable to find reference tensor for group 1");
  NVF_ERROR(
      reference2 != nullptr, "Unable to find reference tensor for group 2");

  // Step 1: Tile two transpose dimensions on reference1, merge all other
  // dimensions into BIDx, and propagate tiling to the entire DAG.
  auto inner_most_id1 = scheduler_utils::innerMostAllocDim(reference1);
  auto inner_most_id2 = scheduler_utils::innerMostAllocDim(reference2);
  int64_t inner_most_pos1 =
      domain_map.getInnerLeafDim(reference1, inner_most_id1);
  int64_t inner_most_pos2 =
      domain_map.getInnerLeafDim(reference1, inner_most_id2);
  NVF_ERROR(
      inner_most_pos1 >= 0 && inner_most_pos2 >= 0 &&
          inner_most_pos1 != inner_most_pos2,
      "Invalid inner dim positions for TMA tiling");

  TensorView* ref_tv = reference1;
  if (reference1->isFusionInput() && tparams->use_tma_load) {
    // can't propagate due to tma load
    auto smem_consumer = ir_utils::consumerTvsOf(reference1).at(0);
    auto regs_consumer = ir_utils::consumerTvsOf(smem_consumer).at(0);
    ref_tv = regs_consumer;
  } else if (reference1->isFusionOutput() && tparams->use_tma_store) {
    // can't propagate due to tma store
    auto smem_producer = ir_utils::getSoleProducerTv(reference1);
    auto regs_producer = ir_utils::getSoleProducerTv(smem_producer);
    ref_tv = regs_producer;
  }

  // make tile, group2 is swizzled, its inner most dim is tile2
  // [..., I1, .., I2, ...]
  ref_tv->split(inner_most_pos1, tparams->tile_size1);
  ref_tv->reorder({{inner_most_pos1 + 1, -1}});
  ref_tv->split(inner_most_pos2, tparams->tile_size2);
  ref_tv->reorder({{inner_most_pos2 + 1, -1}});
  // [..., I1/tile1, .., I2/tile2, ..., tile1, tile2]

  // Merge all non-tiled dimensions into a single BIDx dim
  int64_t rhs_i = ref_tv->nDims() - 3;
  for (int64_t lhs_i = ref_tv->nDims() - 4; lhs_i >= 0; lhs_i--) {
    if (ref_tv->axis(lhs_i)->isReduction() ||
        ref_tv->axis(lhs_i)->isDeviceDim()) {
      continue;
    }
    if (ref_tv->axis(rhs_i)->isReduction() ||
        ref_tv->axis(rhs_i)->isDeviceDim()) {
      rhs_i = lhs_i;
      continue;
    }
    ref_tv->merge(lhs_i, rhs_i);
    rhs_i = lhs_i;
  }
  ref_tv->axis(rhs_i)->parallelize(ParallelType::BIDx);

  // Propagate tiling to all TVs
  TransformPropagator propagator(ref_tv);
  MaxLogicalDomainInfoSpanningTree entire_dag(ref_tv);
  entire_dag.traverse(&propagator);
  scheduler_utils::parallelizeAllLike(
      ref_tv,
      /*selected_tvs=*/{},
      /*selected_parallel_types=*/{},
      /*propagate_padding=*/true,
      /*parallelize_inputs_on_did=*/true);

  // Step 2: Schedule output TMA store.
  if (tparams->is_output_smem_transpose) {
    // Output smem path: swizzle output shared memory with TMA store.
    // Reorder so output inner dim is innermost, then apply TMA swizzle.
    MmaInputSmemSwizzle swizzle =
        mma_utils::tmaSwizzleSharedMemory(tma_store_tvs.at(0));
    for (auto output_smem_cache : tma_store_tvs) {
      mma_utils::scheduleTMAStoreForMmaOutput(output_smem_cache, swizzle);
    }
    for (auto output : output_tvs) {
      mma_utils::scheduleTMAStoreForMmaOutput(output, swizzle);
    }
  } else if (tparams->use_tma_store) {
    // Input smem path with TMA store: Bulk parallel on tile dims.
    for (auto output_smem_cache : tma_store_tvs) {
      // [.., tile1, tile2]
      output_smem_cache->reorder({{-1, -2}});
      // [.., tile2, tile1]
      output_smem_cache->setAllocationDomain(
          output_smem_cache->getLoopDomain(), true);
    }
    for (auto output : output_tvs) {
      output->axis(-1)->parallelize(ParallelType::Bulk);
      output->axis(-2)->parallelize(ParallelType::Bulk);
    }
  }

  // Step 3: Schedule input shared memory.
  if (!tparams->is_output_smem_transpose) {
    NVF_ERROR(
        tparams->use_tma_load,
        "TMA load must be used when input smem is transposed");
    // TMA load and swizzle
    for (auto input_smem_cache : tma_load_tvs) {
      MmaInputSmemSwizzle swizzle_type =
          mma_utils::tmaSwizzleSharedMemory(input_smem_cache);
      input_smem_cache->applyMmaSwizzleForTMALoad(swizzle_type);
    }
  } else if (tparams->use_tma_load) {
    // TMA load without swizzle, just contiguous load with Bulk parallel.
    // Needs to move tile_1 to inner most for contiguous access.
    for (auto input_smem_cache : tma_load_tvs) {
      // [.., tile1, tile2]
      input_smem_cache->reorder({{-1, -2}});
      // [.., tile2, tile1]
      input_smem_cache->axis(-1)->parallelize(ParallelType::Bulk);
      input_smem_cache->axis(-2)->parallelize(ParallelType::Bulk);
      input_smem_cache->setAllocationDomain(
          input_smem_cache->getLoopDomain(), true);
    }
  }

  // Step 4: Schedule register TVs for per-thread access.
  // Tile-2 was swizzled in smem, per-thread access should follow the swizzed
  // layout.
  // 1. split tile2 by elements_per_chunk defined in swizzle pattern
  // 2. further split by chunks_per_thread to get the right granularity for each
  // thread
  // 3. merge remainings with tile1 and parallelize with TIDx
  // [BIDx, tile1, tile2]
  ref_tv->split(-1, tparams->elements_per_chunk);
  // [BIDx, tile1, tile2/chunk, chunk]
  ref_tv->split(-2, tparams->chunks_per_thread);
  // [BIDx, tile1, tile2/chunk/cpt, cpt, chunk]
  ref_tv->merge(-4, -3);
  // [BIDx, tile1/chunk/cpt * tile2, cpt, chunk]
  ref_tv->axis(-3)->parallelize(ParallelType::TIDx);
  // ref_tv->axis(-1)->parallelize(ParallelType::Unroll);

  // Propagate to all TVs except smem/output TVs managed by TMA
  std::unordered_set<TensorView*> skip_tvs(
      tma_load_tvs.begin(), tma_load_tvs.end());
  if (tparams->use_tma_store) {
    skip_tvs.insert(output_tvs.begin(), output_tvs.end());
  }
  auto propagate_tvs = ir_utils::allTvsExcept(fusion, skip_tvs);
  std::unordered_set<TensorView*> propagate_tvs_set(
      propagate_tvs.begin(), propagate_tvs.end());
  SetSelector selector(propagate_tvs_set);
  MaxLogicalDomainInfoSpanningTree propagate_dag(ref_tv, &selector);
  TransformPropagator tp(ref_tv);
  propagate_dag.traverse(&tp);
  scheduler_utils::parallelizeAllLike(
      ref_tv,
      propagate_tvs,
      {},
      /*propagate_padding=*/true,
      /*parallelize_inputs_on_did=*/true);

  // Vectorize smem access at the transpose boundary.
  if (tparams->is_output_smem_transpose) {
    // Vectorize writes to output smem
    for (auto output_smem_cache : tma_store_tvs) {
      output_smem_cache->axis(-1)->parallelize(ParallelType::Vectorize);
    }
  } else {
    // Vectorize reads from input smem
    for (auto tma_load_tv : tma_load_tvs) {
      for (auto consumer : ir_utils::consumerTvsOf(tma_load_tv)) {
        consumer->axis(-1)->parallelize(ParallelType::Vectorize);
      }
    }
  }

  inlineMost();
}

} // namespace tma
} // namespace transpose
} // namespace nvfuser
