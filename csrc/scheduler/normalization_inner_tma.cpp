// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#include <ops/arith.h>
#include <scheduler/cache_policy_refiner.h>
#include <scheduler/normalization_inner_tma.h>
#include <scheduler/normalization_utils.h>
#include <scheduler/reduction_utils.h>
#include <scheduler/tools/inlining.h>
#include <scheduler/utils.h>

#include <ATen/cuda/CUDAContext.h>

#include <algorithm>
#include <cstdint>
#include <memory>

#include <exceptions.h>
#include <iter_visitor.h>
#include <runtime/executor_params.h>

namespace nvfuser {
namespace normalization_inner {
namespace tma {
using PersistentKernelProperties =
    normalization_scheduler_utils::PersistentKernelProperties;

// Basic heuristics for TMA inner persistent scheduler, not tuned yet
std::unique_ptr<InnerNormTmaParams> getInnerPersistentHeuristics(
    Fusion* fusion,
    const PersistentKernelProperties& prop,
    HeuristicDataCache* data_cache) {
  FusionGuard fg(fusion);
  auto dev_prop = at::cuda::getCurrentDeviceProperties();
  const int64_t warp_size = dev_prop->warpSize;
  const int64_t max_threads_per_cta = dev_prop->maxThreadsPerBlock;
  auto params = std::make_unique<InnerNormTmaParams>(
      InnerPersistentKernelScheduler::schedulerType());
  params->tag = "Inner Persistent TMA heuristics";

  const int64_t total_redu_count = prop.inner_most_dimension_numel;

  // Always project persistent buffers to inputs since inputs are cached in
  // shared memory, reducing the size of persistent buffers
  params->project_persistent_buffers = true;

  // Heuristics not fully tuned yet:
  // For non-persistent inputs, use non-TMA load to save shared memory usage
  // when project_persistent_buffers is enabled
  params->tma_load_non_persistent_buffers = false;
  // Vectorize loads from shared memory to registers for better memory bandwidth
  params->vectorize_load_smem_to_regs = true;
  // Issue ldg (load global) instructions early to hide memory latency
  params->pre_load_ldg_tvs = true;

  // Configure reduction domain structure: [persistent_batch_size, bdimx, vect]
  // Prioritize vectorization for memory access efficiency
  params->vectorization_factor = prop.vectorize_factor;

  // Prioritize SM occupancy: ensure at least 4 warps per CTA (128 threads)
  const int64_t after_vect = total_redu_count / params->vectorization_factor;
  int64_t bdimx = std::min(4 * warp_size, after_vect);
  int64_t pbs = ceilDiv(after_vect, bdimx);

  // Derive persistent batch size; if too large, increase bdimx to reduce pbs
  int64_t max_pbs =
      normalization_scheduler_utils::getInnerPersistentMaxBatchSize(true);
  while (pbs > max_pbs && bdimx * 2 <= max_threads_per_cta) {
    bdimx *= 2;
    pbs = ceilDiv(after_vect, bdimx);
  }
  params->persistent_batch_size = pbs;

  // Set index type
  params->cparams.index_type = prop.index_type;

  // Set launch parameters
  params->lparams = LaunchParams(
      LaunchParams::UNINITIALIZED_VAL,
      LaunchParams::UNINITIALIZED_VAL,
      LaunchParams::UNINITIALIZED_VAL,
      LaunchParams::UNINITIALIZED_VAL,
      LaunchParams::UNINITIALIZED_VAL,
      LaunchParams::UNINITIALIZED_VAL);

  if (isDebugDumpEnabled(DebugDumpOption::SchedulerDebug)) {
    debug() << prop.toString() << std::endl;
    debug() << params->toString() << std::endl;
  }

  return params;
}

void scheduleInnerPersistent(Fusion* fusion, const InnerNormTmaParams* params) {
  FusionGuard fg(fusion);

  // Use the reduction tensor as the starting point for scheduling
  // Its transformations will be propagated to all non-TMA tensors
  const auto& reduction_tvs = scheduler_utils::getReductionTvs(fusion);
  TensorView* reduction_tv = reduction_tvs.at(0);

  const scheduler_utils::PersistentBufferInfo persistent_info =
      scheduler_utils::persistentBuffers(fusion);
  std::vector<TensorView*> dummy_outputs;
  if (params->project_persistent_buffers) {
    dummy_outputs = reduction_scheduler_utils::projectPersistentBuffers(
        fusion, persistent_info, params->project_persistent_buffers);
    for (auto output : dummy_outputs) {
      fusion->addOutput(output);
    }
  } else {
    NVF_ERROR(
        false, "Non-projectable buffers are not supported in TMA version yet");
  }
  // Identify persistent inputs that will be cached in shared memory using
  // TMA load (CpAsyncBulk) for efficient async memory transfers
  std::unordered_set<TensorView*> persistent_inputs{
      persistent_info.projectable_buffer_inputs.begin(),
      persistent_info.projectable_buffer_inputs.end()};
  for (auto buffer : persistent_info.persistent_buffers) {
    if (buffer->isFusionInput()) {
      persistent_inputs.insert(buffer);
    }
  }
  // Categorize cached inputs into TMA loads and regular LDG loads
  // - tma_tvs: Persistent inputs loaded via TMA (CpAsyncBulk) to shared memory
  // - ldg_tvs: Non-persistent inputs loaded via regular LDG instructions
  // - smem2reg_tvs: Intermediate register caches for vectorized smem->reg loads
  std::vector<TensorView*> ldg_tvs, tma_tvs, smem2reg_tvs;
  const auto& cached_inputs = scheduler_utils::cacheInputs(fusion, true);
  for (auto [tv, input_idx] : cached_inputs) {
    auto input = fusion->inputs().at(input_idx)->as<TensorView>();
    if (!params->tma_load_non_persistent_buffers &&
        !persistent_inputs.contains(input)) {
      // Non-persistent input: use regular global load
      ldg_tvs.push_back(tv);
      continue;
    }
    if (auto load_op = dynamic_cast<LoadStoreOp*>(tv->definition())) {
      // Persistent input: use TMA load to shared memory
      load_op->setOpType(LoadStoreOpType::CpAsyncBulk);
      tv->setMemoryType(MemoryType::Shared);
      tma_tvs.push_back(tv);
      if (params->vectorize_load_smem_to_regs) {
        // Create register cache for vectorized smem->reg loads
        auto regs_cache = tv->cacheAfter();
        smem2reg_tvs.push_back(regs_cache);
        // Replicate cache for multiple consumers to avoid conflicts
        const auto& consumers = ir_utils::consumerTvsOf(regs_cache);
        for (auto consumer : ir_utils::consumerTvsOf(regs_cache) | std::views::drop(1)) {
          auto consumer = consumers.at(i);
          auto cached_tv_replicate = RecomputeTv::recompute(regs_cache, {tv});
          ir_utils::replaceValInExprInputs(
              consumer->definition(), regs_cache, cached_tv_replicate);
          smem2reg_tvs.push_back(cached_tv_replicate);
        }
      }
    } else {
      ldg_tvs.push_back(tv);
    }
  }

  // Cache outputs in registers to enable vectorized writes to global memory
  const auto& cached_outputs =
      scheduler_utils::cacheAndForkOutputs(fusion, /*unroll=*/true);

  // Schedule TMA loads with shape [I, R] where:
  //   I = iteration dimension (batch dimension)
  //   R = reduction dimension
  // Parallelization strategy:
  //   - axis(0): BIDx - each block handles one or more batch elements
  //   - axis(1): Bulk - TMA asynchronously copies entire reduction dimension
  int64_t ipos = 0, rpos = 1;
  if (params->rows_per_block > 1) {
    // [I, R] -> [I/TIDy, TIDy, R]
    reduction_tv->split(ipos, params->rows_per_block);
    TransformPropagator propagator(reduction_tv);
    MaxLogicalDomainInfoSpanningTree(reduction_tv).traverse(&propagator);
    reduction_tv->axis(ipos)->parallelize(ParallelType::BIDx);
    reduction_tv->axis(ipos + 1)->parallelize(ParallelType::TIDy);
    rpos = ipos + 2;
  } else {
    reduction_tv->axis(ipos)->parallelize(ParallelType::BIDx);
  }
  reduction_tv->axis(rpos)->parallelize(ParallelType::Bulk);
  scheduler_utils::parallelizeAllLike(reduction_tv, tma_tvs);
  // Reset reduction_tv's reduction axis back to Serial (only TMA loads use
  // Bulk)
  reduction_tv->axis(rpos)->parallelize(ParallelType::Serial);

  // Transform reduction domain for efficient computation:
  //   [I, R] -> [I, b, us, x, v]
  // Where:
  //   - v: vectorization factor (elements per vector instruction)
  //   - x: thread dimension (TIDx, threads cooperating on reduction)
  //   - b: persistent batch size (register persistent buffer size)
  //   - us: unswitch dimension (loop optimization, reduces control flow
  //   overhead)
  reduction_tv->split(rpos, params->vectorization_factor);
  reduction_tv->split(rpos, params->persistent_batch_size, false);
  reduction_tv->split(rpos, 1);
  reduction_tv->axis(rpos + 1)->parallelize(ParallelType::Unswitch);
  reduction_tv->axis(rpos + 2)->parallelize(ParallelType::TIDx);
  reduction_tv->axis(rpos + 2)->padToMultipleOfWarp();

  // Create rfactor tensor to separate thread-local reduction from block
  // reduction This enables a two-stage reduction:
  //   1. Thread-local vectorized reduction (across b and v dimensions)
  //   2. Block-level reduction (across x dimension using warp/block primitives)
  // rfactor axes: {rpos, vectorize_pos} corresponding to b and v dimensions
  int64_t vectorize_pos = rpos + 3;
  auto reference_tv = reduction_tv->rFactor({rpos, vectorize_pos});

  // Propagate transformations from reference_tv to all non-TMA tensors
  // TMA tensors keep their simple [BIDx, Bulk] schedule
  std::vector<TensorView*> non_tma_tvs =
      ir_utils::allTvsExcept(fusion, {tma_tvs.begin(), tma_tvs.end()});
  TransformPropagator non_tma_propagator(reference_tv);
  SetSelector selector({non_tma_tvs.begin(), non_tma_tvs.end()});
  MaxLogicalDomainInfoSpanningTree(reference_tv, &selector)
      .traverse(&non_tma_propagator);

  // If reduction_tv is rfactored, rfactor all reductions.
  // Also needs to update non_tma_tvs to include newly rfactored tvs.
  if (reference_tv != reduction_tv) {
    reduction_scheduler_utils::propagateRFactor(
        reference_tv, reduction_tv, reduction_tvs);
    non_tma_tvs =
        ir_utils::allTvsExcept(fusion, {tma_tvs.begin(), tma_tvs.end()});
  }
  scheduler_utils::parallelizeAllLike(reference_tv, non_tma_tvs);

  // Helper lambda to find the vectorization position (one past TIDx axis)
  auto get_vect_pos = [](TensorView* tv) {
    auto it = std::find_if(
        tv->domain()->loop().begin(),
        tv->domain()->loop().end(),
        [](const IterDomain* id) {
          return id->getParallelType() == ParallelType::TIDx;
        });

    if (it == tv->domain()->loop().end()) {
      return -1;
    }
    return std::distance(tv->domain()->loop().begin(), it) + 1;
  };

  // Apply vectorization to non-TMA global loads
  for (auto tv : ldg_tvs) {
    auto vect_pos = get_vect_pos(tv);
    if (vect_pos > 0) {
      tv->axis(vect_pos)->parallelize(ParallelType::Vectorize);
    }
  }

  // Apply vectorization to global output writes for better memory bandwidth
  for (auto [_, output_idx] : cached_outputs) {
    auto output = fusion->outputs()[output_idx]->as<TensorView>();
    auto vect_pos = get_vect_pos(output);
    if (vect_pos > 0) {
      output->axis(vect_pos)->parallelize(ParallelType::Vectorize);
    }
  }

  // Apply vectorization to shared memory to register loads
  if (params->vectorize_load_smem_to_regs) {
    for (auto tv : smem2reg_tvs) {
      auto vect_pos = get_vect_pos(tv);
      if (vect_pos > 0) {
        tv->axis(vect_pos)->parallelize(ParallelType::Vectorize);
      }
    }
  }

  // Remove dummy outputs that were used for persistent buffer projection
  for (auto output : dummy_outputs) {
    fusion->removeOutput(output);
  }

  // Apply aggressive inlining to reduce register pressure and improve locality
  // Exclude ldg_tvs if pre-loading is enabled to control issue order
  std::unordered_set<TensorView*> exclude_tvs;
  if (params->pre_load_ldg_tvs) {
    exclude_tvs.insert(ldg_tvs.begin(), ldg_tvs.end());
  }
  std::vector<TensorView*> inline_most_tvs =
      ir_utils::allTvsExcept(fusion, exclude_tvs);
  inlineMost(inline_most_tvs);

  // Refine cache policies for optimal memory hierarchy usage
  refineCachePolicy(fusion);
}

} // namespace tma
} // namespace normalization_inner
} // namespace nvfuser
