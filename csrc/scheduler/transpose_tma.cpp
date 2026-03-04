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
  // Pick the side with fewer tensors to minimize smem usage and swizzle cost.
  tparams->is_output_smem_transpose = n_input > n_output;
  tparams->use_tma_load = false;
  tparams->use_tma_store = tparams->is_output_smem_transpose;

  // Inputs and outputs are grouped by their innermost dim into two groups.
  // Group 2 is the swizzled side. tile_size2 is constrained by TMA swizzle.
  //
  // swizzleTMABox decomposes the tile into:
  //   [KO, KIO, KII, NIO, NII] where KI=8 rows, NII=8 elements
  //   KIO = 128 / swizzle_bytes, then applies XOR(KIO, NIO)
  //
  // For bf16 with B128: tile_size2=64, KIO = 128/128 = 1.
  //   XOR with a single KIO value is the identity — no actual swizzle.
  //   tile_size2=64 -> Array<bf16,64,2> per input -> 80 regs -> 34% occupancy.
  //
  // For bf16 with B64: tile_size2=32, KIO = 128/64 = 2.
  //   XOR(KIO=2, NIO=4) produces a real 2-way column permutation: rows
  //   in even vs odd KIO groups have their NIO chunks swapped.
  //   tile_size2=32 -> Array<bf16,32,2> per input -> 48 regs -> 57% occupancy.
  //
  // The main performance gain comes from halving tile_size2 which halves
  // register usage (48 vs 80) and smem footprint (16KB vs 32KB per tile),
  // allowing higher occupancy to hide DRAM latency.
  int64_t swizzled_dtype_size = tparams->is_output_smem_transpose
      ? max_output_dtype_size
      : max_input_dtype_size;
  constexpr int64_t kPreferredSwizzleBytes = 64;
  int64_t constrained_tile = kPreferredSwizzleBytes / swizzled_dtype_size;
  tparams->tile_size2 = constrained_tile;

  // 16 bytes per chunk: 4 float32 or 8 bfloat16 elements.
  tparams->elements_per_chunk = kBytesPerChunk / swizzled_dtype_size;

  // Vectorize along tile1 (the non-swizzled dim).
  // vec=2 for bf16 gives 4-byte stores (aligned to bank width), minimizing
  // bank conflicts on the smem write path.
  tparams->vectorize_factor1 = 2;

  // Heuristic for tile_size1 (the non-swizzled, tunable dim).
  auto dev_props = at::cuda::getCurrentDeviceProperties();
  constexpr int64_t bytes_per_sm = 64 * 1024;
  constexpr int64_t threads_per_cta = 256;
  const int64_t cta_per_sm =
      dev_props->maxThreadsPerMultiProcessor / threads_per_cta;
  const int64_t bytes_per_cta = bytes_per_sm / cta_per_sm;
  const int64_t bytes_per_tile = bytes_per_cta / n_input;
  int64_t estimated_tile_size1 = bytes_per_tile / kPreferredSwizzleBytes;

  // Ensure each thread processes at least min_chunks_per_thread chunks.
  constexpr int64_t min_chunks_per_thread = 2;
  auto get_chunks_per_thread = [&]() {
    int64_t elements_per_thread =
        estimated_tile_size1 * tparams->tile_size2 / threads_per_cta;
    return elements_per_thread / tparams->elements_per_chunk /
        tparams->vectorize_factor1;
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

  // Set up TMA load: input -> smem_cache (TMA) -> reg_cache
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

  // Collect global output TVs (needed for TMA store scheduling).
  std::vector<TensorView*> output_tvs;
  output_tvs.reserve(cached_outputs.size());
  for (auto [cached_output, output_idx] : cached_outputs) {
    output_tvs.push_back(fusion->outputs()[output_idx]->as<TensorView>());
  }

  // Set up TMA store: reg_cache -> smem_cache (TMA) -> output
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

  // Group tensors by innermost dim. Group 1 = non-swizzled, Group 2 = swizzled.
  scheduler_tools::TransposeDomainMap domain_map(fusion);
  auto grouped_inputs_outputs = domain_map.groupInputsOutputsByInnerDim();
  NVF_ERROR(grouped_inputs_outputs.size() >= 2);

  // When not using output smem transpose but inputs > outputs, swap groups
  // so group 2 remains the swizzled side.
  if (!tparams->is_output_smem_transpose &&
      cached_inputs.size() > cached_outputs.size()) {
    std::swap(grouped_inputs_outputs[0], grouped_inputs_outputs[1]);
  }

  TensorView* group1_ref =
      domain_map.findReferenceFor(grouped_inputs_outputs[0]);
  TensorView* group2_ref =
      domain_map.findReferenceFor(grouped_inputs_outputs[1]);
  NVF_ERROR(
      group1_ref != nullptr, "Unable to find reference tensor for group 1");
  NVF_ERROR(
      group2_ref != nullptr, "Unable to find reference tensor for group 2");

  // Step 1: Tile two transpose dimensions on group1_ref, merge all other
  // dimensions into BIDx, and propagate tiling to the entire DAG.
  auto group1_inner_id = scheduler_utils::innerMostAllocDim(group1_ref);
  auto group2_inner_id = scheduler_utils::innerMostAllocDim(group2_ref);
  int64_t group1_inner_pos =
      domain_map.getInnerLeafDim(group1_ref, group1_inner_id);
  int64_t group2_inner_pos =
      domain_map.getInnerLeafDim(group1_ref, group2_inner_id);
  NVF_ERROR(
      group1_inner_pos >= 0 && group2_inner_pos >= 0 &&
          group1_inner_pos != group2_inner_pos,
      "Invalid inner dim positions for TMA tiling");

  // Transform propagation can't pass through TMA tvs, select
  // a tv after tma load or before tma store.
  TensorView* ref_tv = group1_ref;
  if (group1_ref->isFusionInput() && tparams->use_tma_load) {
    auto smem_consumer = ir_utils::consumerTvsOf(group1_ref).at(0);
    auto regs_consumer = ir_utils::consumerTvsOf(smem_consumer).at(0);
    ref_tv = regs_consumer;
  } else if (group1_ref->isFusionOutput() && tparams->use_tma_store) {
    auto smem_producer = ir_utils::getSoleProducerTv(group1_ref);
    auto regs_producer = ir_utils::getSoleProducerTv(smem_producer);
    ref_tv = regs_producer;
  }

  // Split and reorder to create tiles:
  // [..., I1, .., I2, ...] → [..., I1/tile1, .., I2/tile2, ..., tile1, tile2]
  ref_tv->split(group1_inner_pos, tparams->tile_size1);
  ref_tv->reorder({{group1_inner_pos + 1, -1}});
  ref_tv->split(group2_inner_pos, tparams->tile_size2);
  ref_tv->reorder({{group2_inner_pos + 1, -1}});

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
  TransformPropagator tiling_propagator(ref_tv);
  MaxLogicalDomainInfoSpanningTree entire_dag(ref_tv);
  entire_dag.traverse(&tiling_propagator);
  scheduler_utils::parallelizeAllLike(
      ref_tv,
      /*selected_tvs=*/{},
      /*selected_parallel_types=*/{},
      /*propagate_padding=*/true,
      /*parallelize_inputs_on_did=*/true);

  // Step 2: Schedule output TMA store.
  if (tparams->is_output_smem_transpose) {
    // Output smem path: apply swizzle to output shared memory.
    // For bf16, tile_size2=64 gives 128-byte rows. B128 swizzle gives KIO=1
    // (trivial). Use B64 instead: splits row into 32-elem halves, KIO=2,
    // giving real XOR swizzle that eliminates bank conflicts.
    // Override: use B64 for bf16, B32 for fp32 (anything where B128 would be
    // trivial).
    MmaInputSmemSwizzle swizzle =
        mma_utils::tmaSwizzleSharedMemory(tma_store_tvs.at(0));

    for (auto output_smem_cache : tma_store_tvs) {
      mma_utils::scheduleTMAStoreForMmaOutput(output_smem_cache, swizzle);
    }
    for (auto output : output_tvs) {
      mma_utils::scheduleTMAStoreForMmaOutput(output, swizzle);
    }
  } else if (tparams->use_tma_store) {
    // Input smem path with TMA store: reorder for contiguous output and
    // set Bulk parallel on tile dims.
    for (auto output_smem_cache : tma_store_tvs) {
      // [.., tile1, tile2] → [.., tile2, tile1]
      output_smem_cache->reorder({{-1, -2}});
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
    // Input smem path: TMA load into swizzled shared memory.
    NVF_ERROR(
        tparams->use_tma_load,
        "TMA load must be used when input smem is transposed");
    for (auto input_smem_cache : tma_load_tvs) {
      MmaInputSmemSwizzle swizzle_type =
          mma_utils::tmaSwizzleSharedMemory(input_smem_cache);
      input_smem_cache->applyMmaSwizzleForTMALoad(swizzle_type);
    }
  } else if (tparams->use_tma_load) {
    // Output smem path: TMA load without swizzle. Reorder so tile1
    // (group 1's inner dim) is innermost for contiguous access.
    for (auto input_smem_cache : tma_load_tvs) {
      // [.., tile1, tile2] → [.., tile2, tile1]
      input_smem_cache->reorder({{-1, -2}});
      input_smem_cache->axis(-1)->parallelize(ParallelType::Bulk);
      input_smem_cache->axis(-2)->parallelize(ParallelType::Bulk);
      input_smem_cache->setAllocationDomain(
          input_smem_cache->getLoopDomain(), true);
    }
  }

  // Step 4: Schedule register TVs for per-thread access.
  // ref_tv's innermost tile is tile1 (group 1's inner dim).
  // The merge order is critical for bank conflicts on the non-swizzled smem
  // side. By merging tile2_outer as the outer dim and tile1 as the inner dim,
  // adjacent threads in a warp access adjacent tile1 positions. Since tile1 is
  // the contiguous (inner) dimension of the non-swizzled smem layout, this
  // means adjacent threads read from adjacent memory addresses.
  // [BIDx, tile1, tile2]
  ref_tv->split(-1, tparams->elements_per_chunk);
  // [BIDx, tile1, tile2/chunk, chunk]
  ref_tv->split(-2, tparams->chunks_per_thread);
  // [BIDx, tile1, tile2/chunk/cpt, cpt, chunk]
  ref_tv->merge(-3, -4);
  // [BIDx, tile2/chunk/cpt * tile1, cpt, chunk]
  if (tparams->vectorize_factor1 > 1) {
    ref_tv->split(-3, tparams->vectorize_factor1);
    // [BIDx, tile2/chunk/cpt * tile1/vec, vec, cpt, chunk]
    ref_tv->axis(-4)->parallelize(ParallelType::TIDx);
  } else {
    ref_tv->axis(-3)->parallelize(ParallelType::TIDx);
  }

  // Propagate register scheduling to all TVs except TMA smem/output TVs.
  std::unordered_set<TensorView*> skip_tvs(
      tma_load_tvs.begin(), tma_load_tvs.end());
  if (tparams->use_tma_store) {
    skip_tvs.insert(output_tvs.begin(), output_tvs.end());
  }
  auto reg_tvs = ir_utils::allTvsExcept(fusion, skip_tvs);
  std::unordered_set<TensorView*> reg_tvs_set(reg_tvs.begin(), reg_tvs.end());
  SetSelector selector(reg_tvs_set);
  MaxLogicalDomainInfoSpanningTree reg_dag(ref_tv, &selector);
  TransformPropagator reg_propagator(ref_tv);
  reg_dag.traverse(&reg_propagator);
  scheduler_utils::parallelizeAllLike(
      ref_tv,
      reg_tvs,
      {},
      /*propagate_padding=*/true,
      /*parallelize_inputs_on_did=*/true);

  // Vectorize smem reads at the transpose boundary: each consumer of
  // a TMA-loaded smem TV reads with vectorized access.
  auto vectorize_smem_reads = [&](int pos) {
    for (auto tma_load_tv : tma_load_tvs) {
      for (auto consumer : ir_utils::consumerTvsOf(tma_load_tv)) {
        consumer->axis(pos)->parallelize(ParallelType::Vectorize);
      }
    }
    if (!tparams->use_tma_load) {
      for (auto [cached_input, _] : cached_inputs) {
        cached_input->axis(-3)->parallelize(ParallelType::Vectorize);
      }
    }
  };
  if (tparams->is_output_smem_transpose) {
    // Vectorize writes to swizzled output smem (chunk dim).
    for (auto output_smem_cache : tma_store_tvs) {
      output_smem_cache->axis(-1)->parallelize(ParallelType::Vectorize);
    }
    // Vectorize reads from non-swizzled input smem (vec dim).
    // [BIDx, tile2/chunk/cpt * tile1/vec, vec, cpt, chunk]
    if (tparams->vectorize_factor1 > 1) {
      vectorize_smem_reads(-3);
    }
  } else {
    // Input smem path: vectorize reads from swizzled input smem (chunk dim).
    vectorize_smem_reads(-1);
  }

  inlineMost();
}

} // namespace tma
} // namespace transpose
} // namespace nvfuser
