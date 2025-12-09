// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on

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
#include "iter_visitor.h"
#include "runtime/executor_params.h"

namespace nvfuser {
namespace normalization_inner {
namespace tma {

std::unique_ptr<InnerNormTmaParams> getInnerPersistentHeuristics(
    Fusion* fusion,
    SchedulerRuntimeInfo& runtime_info,
    HeuristicDataCache* data_cache) {
  FusionGuard fg(fusion);
  auto dev_prop = at::cuda::getCurrentDeviceProperties();
  const int64_t warp_size = dev_prop->warpSize;
  const int64_t sm_count = dev_prop->multiProcessorCount;
  const int64_t max_threads_per_cta = dev_prop->maxThreadsPerBlock;
  // Create TMA-specific parameters
  auto params = std::make_unique<InnerNormTmaParams>(
      InnerPersistentKernelScheduler::schedulerType());
  params->tag = "Inner Persistent TMA heuristics";

  // Get properties of the fusion
  const auto& prop =
      normalization_scheduler_utils::getPersistentKernelProperties(
          fusion,
          runtime_info,
          data_cache,
          InnerPersistentKernelScheduler::schedulerType());
  // reduction domain heuristics: vectorization factor and bdimx
  const int64_t total_redu_count = prop.inner_most_dimension_numel;
  const int64_t total_iter_count = prop.total_iteration_numel;
  const int64_t max_vect = prop.vectorize_factor;
  const bool has_expensive_op = prop.has_exp_op || prop.has_rng_op;

  if (prop.disable_project_to_avoid_recompute &&
      total_redu_count <= 1024 * 12) {
    params->project_persistent_buffers = false;
  } else {
    params->project_persistent_buffers = true;
  }
  params->vectorize_load_smem_to_regs = true;
  params->may_pre_load_ldg_tvs = true;
  params->tma_load_non_persistent_buffers = false;

  // start from bdimx = warp size, vect = 1, then increase vect to maximum
  int64_t bdimx = std::min(warp_size, total_redu_count);
  int64_t vect = 1;
  while (vect * 2 <= max_vect && vect * 2 * bdimx <= total_redu_count) {
    vect *= 2;
  }
  params->vectorization_factor = vect;
  const int64_t after_vect = total_redu_count / vect;

  int64_t pbs = 1, bdimy = -1;

  auto useMultipleWarps = [&]() {
    int64_t max_pbs =
        normalization_scheduler_utils::getInnerPersistentMaxBatchSize(true);
    if (has_expensive_op && !params->project_persistent_buffers) {
      max_pbs /= 2;
    }
    bdimx = 4 * warp_size;
    pbs = ceilDiv(after_vect, bdimx);
    while (pbs > max_pbs && bdimx * 2 <= max_threads_per_cta) {
      bdimx *= 2;
      pbs = ceilDiv(after_vect, bdimx);
    }
  };

  auto trySingleWarp = [&]() {
    const int64_t after_vect_bdimx = ceilDiv(after_vect, bdimx);
    int64_t max_pbs = std::min(after_vect_bdimx, 16L);
    for (int ipbs = pbs + 1; ipbs <= max_pbs; ipbs++) {
      if (after_vect % ipbs == 0) {
        pbs = ipbs;
      }
    }
    if (bdimx * pbs >= after_vect) {
      bdimy = 1;
      const int64_t target_waves = 4;
      const int64_t target_bdimy = 128 / bdimx;
      while (bdimy * 2 <= target_bdimy &&
             ceilDiv(total_iter_count, bdimy * 2) > sm_count * target_waves) {
        bdimy *= 2;
      }
    }
  };

  if (!has_expensive_op) {
    trySingleWarp();
  }
  // single warp failed, try large bdimx
  if (bdimy > 0) {
    params->rows_per_block = bdimy;
  } else {
    useMultipleWarps();
  }
  params->persistent_batch_size = ceilDiv(total_redu_count / vect, bdimx);

  // set warp specialized circular buffer options
  int64_t gdimx = LaunchParams::UNINITIALIZED_VAL;
  int64_t bdimz = LaunchParams::UNINITIALIZED_VAL;
  if (bdimx >= 128 && std::getenv("WS") != nullptr) {
    gdimx = sm_count;
    bdimy = 2;
    if (std::getenv("TIDY")) {
      bdimy = std::stoi(std::getenv("TIDY"));
    }
    bdimz = 1;
    params->rows_per_block = bdimy;
    int64_t smem_size_bit =
        prop.max_persistent_buffer_size_bit * bdimy * params->rows_per_block;
    int64_t n_stages =
        (int64_t)dev_prop->sharedMemPerBlockOptin * 8 / smem_size_bit;
    ParallelType ws_pt = bdimy > 1 ? ParallelType::TIDy : ParallelType::TIDx;
    if (ws_pt == ParallelType::TIDy) {
      bdimy += 1;
      // TIDy warp specialization is used only when multiple compute warp groups
      // are used. Current runtime only supports grouped reduction.
      NVF_ERROR(bdimx == 128, "bdimx must be 128 for TIDy warp specialization");
      NVF_ERROR(
          params->rows_per_block > 1,
          "rows_per_block must be 1 for TIDy warp specialization");
    } else {
      bdimx += kWarpSpecializationPaddedThreads;
    }
    WarpSpecialized ws(ws_pt);
    if (bdimy > 1) {
      ws.stage_slice_position = 3;
    }
    int64_t total_threads = bdimx * bdimy;
    if (total_threads > 256) {
      int64_t reg_per_thread = getRegPerThreadGivenThreadsPerSM(total_threads);
      ws.num_registers = scheduler_utils::getRegisterSharing(
          reg_per_thread, bdimx * bdimy, kWarpSpecializationPaddedThreads);
    }
    CircularBufferOptions circular_buffer_options{
        .type = ws, .stage = n_stages, .prefetch = n_stages - 1};
    params->circular_buffer_options = circular_buffer_options;
  }

  // set persistent batch size
  if (std::getenv("PBS")) {
    params->persistent_batch_size = std::stoi(std::getenv("PBS"));
  }
  if (std::getenv("RPB")) {
    params->rows_per_block = std::stoi(std::getenv("RPB"));
  }
  // Set index type
  params->cparams.index_type = prop.index_type;

  // Set launch parameters
  params->lparams = LaunchParams(
      gdimx,
      LaunchParams::UNINITIALIZED_VAL,
      LaunchParams::UNINITIALIZED_VAL,
      bdimx,
      bdimy,
      bdimz);

  // debug print
  if (isDebugDumpEnabled(DebugDumpOption::SchedulerDebug)) {
    debug() << prop.toString() << std::endl;
    debug() << params->toString() << std::endl;
  }

  return params;
}

void scheduleInnerPersistent(Fusion* fusion, const InnerNormTmaParams* params) {
  FusionGuard fg(fusion);
  const scheduler_utils::PersistentBufferInfo persistent_info =
      scheduler_utils::persistentBuffers(fusion);
  std::vector<TensorView*> dummy_outputs;
  if (params->project_persistent_buffers) {
    dummy_outputs = reduction_scheduler_utils::projectPersistentBuffers(
        fusion, persistent_info, params->project_persistent_buffers);
    for (auto output : dummy_outputs) {
      fusion->addOutput(output);
    }
  }
  // Cache input tv0 in shared memory using TMA load (CpAsyncBulk)
  std::unordered_set<TensorView*> persistent_inputs{
      persistent_info.projectable_buffer_inputs.begin(),
      persistent_info.projectable_buffer_inputs.end()};
  for (auto tv : persistent_info.projectable_buffer_inputs) {
    std::cout << "projectable_buffer_inputs tv: " << tv->toString()
              << std::endl;
  }
  std::vector<TensorView*> ldg_tvs, tma_tvs, smem2reg_tvs;
  const auto& cached_inputs = scheduler_utils::cacheInputs(fusion, true);
  for (auto [tv, input_idx] : cached_inputs) {
    auto input = fusion->inputs()[input_idx]->as<TensorView>();
    std::cout << "cached input: " << tv->toString() << std::endl;
    std::cout << "xxxxxx input: " << input->toString() << std::endl;
    if (!params->tma_load_non_persistent_buffers &&
        !persistent_inputs.contains(input)) {
      ldg_tvs.push_back(tv);
      continue;
    }
    if (auto load_op = dynamic_cast<LoadStoreOp*>(tv->definition())) {
      load_op->setOpType(LoadStoreOpType::CpAsyncBulk);
      tv->setMemoryType(MemoryType::Shared);
      tma_tvs.push_back(tv);
      if (params->vectorize_load_smem_to_regs) {
        auto regs_cache = tv->cacheAfter();
        smem2reg_tvs.push_back(regs_cache);
        // TODO: just repalce persistent uses insead of all uses
        const auto& consumers = ir_utils::consumerTvsOf(regs_cache);
        for (auto i = 1; i < (int)consumers.size(); i++) {
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

  // Cache output tv5 in registers to enable vectorized write to global memory
  const auto& cached_outputs =
      scheduler_utils::cacheAndForkOutputs(fusion, /*unroll=*/true);

  // Use the reduction tensor tv2 as the starting point for scheduling
  // Its transformations will be propagated to all non-TMA tensors
  const auto& reduction_tvs = scheduler_utils::getReductionTvs(fusion);
  for (auto tv : reduction_tvs) {
    std::cout << "reduction_tv: " << tv->toString() << std::endl;
  }
  TensorView* reduction_tv = reduction_tvs.at(0);

  // Schedule TMA load: [I, R]
  // - No transformation is needed as we assume each block handles one row
  // - axis(0): parallelize with BIDx (each block handles one batch)
  // - axis(1): parallelize with Bulk (TMA async copy entire reduction
  // dimension)
  int64_t ipos = 0, rpos = 1, tidy_pos = -1, group_pos = -1;
  if (params->circular_buffer_options.isEnable()) {
    if (params->rows_per_block > 1) {
      // [I, R] -> [I/Group, Group, R]
      reduction_tv->split(ipos, params->rows_per_block);
      group_pos = ipos + 1;
      rpos++;
    }
    if (params->lparams.bdimy() > 2) {
      // [I/Group, Group, R] -> [I/Group/TIDy, TIDy, Group, R]
      reduction_tv->split(ipos, params->lparams.bdimy() - 1);
      tidy_pos = ipos + 1;
      rpos++;
      group_pos++;
    }
    if (params->lparams.gdimx() > 1) {
      // [I/Group/TIDy, TIDy, Group, R] -> [I/Group/TIDy/BIDx, BIDx, TIDy,
      // Group, R]
      reduction_tv->split(ipos, params->lparams.gdimx());
      reduction_tv->axis(ipos + 1)->parallelize(ParallelType::BIDx);
      rpos++;
      tidy_pos++;
      group_pos++;
    }
    TransformPropagator propagator(reduction_tv);
    MaxLogicalDomainInfoSpanningTree(reduction_tv).traverse(&propagator);
  } else if (params->rows_per_block > 1) {
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
  // Change reduction_tv's axis(1) back to Serial (only TMA tvs use Bulk)
  reduction_tv->axis(rpos)->parallelize(ParallelType::Serial);
  // For TMA tvs, we use serial to use 1 producer to serve all consumers
  // parallelized by TIDy
  if (tidy_pos > 0) {
    reduction_tv->axis(tidy_pos)->parallelize(ParallelType::TIDy);
  }

  // Schedule the reduction domain:
  // [I, R] -> [I, b, us, x, v]
  // Split R into multiple dimensions for efficient reduction:
  //   - v: vectorization factor (elements processed per vector instruction)
  //   - x: thread dimension (bdimx threads cooperate on reduction)
  //   - b: persistent batch size
  //   - us: unswitch dimension (for loop optimization)
  reduction_tv->split(rpos, params->vectorization_factor);
  reduction_tv->split(rpos, params->persistent_batch_size, false);
  reduction_tv->split(rpos, 1);
  reduction_tv->axis(rpos + 1)->parallelize(ParallelType::Unswitch);
  reduction_tv->axis(rpos + 2)->parallelize(ParallelType::TIDx);
  reduction_tv->axis(rpos + 2)->padToMultipleOfWarp();

  // Create rfactor tv to separate thread-local vectorized reduction from block
  // reduction rfactor axes: {rpos, vectorize_pos} = {1, 4} corresponding to
  // R/v/x and v dimensions
  int64_t vectorize_pos = rpos + 3;
  auto reference_tv = reduction_tv->rFactor({rpos, vectorize_pos});

  std::cout << "reference_tv: " << reference_tv->toString() << std::endl;
  std::cout << "reduction_tv: " << reduction_tv->toString() << std::endl;
  for (auto tma_tv : tma_tvs) {
    std::cout << "tma_tv: " << tma_tv->toString() << std::endl;
  }

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

  if (group_pos > 0) {
    for (auto reduction_tv : reduction_tvs) {
      reduction_tv->axis(group_pos)->parallelize(ParallelType::Group);
    }
  }

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
    return int(it - tv->domain()->loop().begin()) + 1;
  };
  // vectorize ldg tvs
  for (auto tv : ldg_tvs) {
    auto vect_pos = get_vect_pos(tv);
    if (vect_pos > 0) {
      tv->axis(vect_pos)->parallelize(ParallelType::Vectorize);
    }
  }
  // Vectorize output write to global memory
  for (auto [_, output_idx] : cached_outputs) {
    auto output = fusion->outputs()[output_idx]->as<TensorView>();
    auto vect_pos = get_vect_pos(output);
    if (vect_pos > 0) {
      output->axis(vect_pos)->parallelize(ParallelType::Vectorize);
    }
  }
  if (params->vectorize_load_smem_to_regs) {
    for (auto tv : smem2reg_tvs) {
      auto vect_pos = get_vect_pos(tv);
      if (vect_pos > 0) {
        tv->axis(vect_pos)->parallelize(ParallelType::Vectorize);
      }
    }
  }
  // Inline all tensors to minimize register usage
  for (auto output : dummy_outputs) {
    fusion->removeOutput(output);
  }

  // inline
  std::unordered_set<TensorView*> exclude_tvs;
  if (params->circular_buffer_options.isEnable()) {
    // when warp specialized, the iteration domain of tma tv is scheduled as:
    // 1. GridStrideLoop
    // 2. BIDx
    // 3. Serial (Compute Warp Groups, TIDy in compute warp groups)
    // 4. Serial (Multiple TMAs share one mbarrier, serial or grouped reduction
    //            in compuate warp groups)
    constexpr int64_t pos_after_bidx = 2;
    for (auto tv : tma_tvs) {
      inlineSelectedAt({tv}, tv, pos_after_bidx);
      exclude_tvs.insert(tv);
    }
  }
  if (params->may_pre_load_ldg_tvs && (int)ldg_tvs.size() == 1) {
    exclude_tvs.insert(ldg_tvs.begin(), ldg_tvs.end());
  }
  // result of the 1st reduction is used by the 2nd reduction
  if (group_pos > 0 && reduction_tvs.size() > 1) {
    auto all_vals = DependencyCheck::getAllValsBetween(
        {reduction_tvs.at(0)}, {reduction_tvs.at(1)});
    auto gp_tvs = ir_utils::filterByType<TensorView>(all_vals);
    for (auto gp_tv : gp_tvs) {
      if (gp_tv->hasBroadcast()) {
        std::cout << "gp_tvs: " << gp_tv->toString() << std::endl;
        inlineSelectedAt({gp_tv}, gp_tv, group_pos);
        exclude_tvs.insert(gp_tv);
      }
    }
  }

  std::vector<TensorView*> inline_most_tvs =
      ir_utils::allTvsExcept(fusion, exclude_tvs);
  inlineMost(inline_most_tvs);

  if (params->circular_buffer_options.isEnable()) {
    int64_t number_of_stages = params->circular_buffer_options.stage;
    int64_t prefetch_distance = params->circular_buffer_options.prefetch;
    CircularBufferType circular_buffer_type =
        params->circular_buffer_options.type;
    for (auto tv : tma_tvs) {
      if (tv->getComputeAtPosition() > 0) {
        tv->circularBuffer(
            number_of_stages, prefetch_distance, circular_buffer_type);
      }
      std::cout << "tma_tv: " << tv->toString()
                << " computeAtPosition: " << tv->getComputeAtPosition()
                << std::endl;
    }
  }
  // refine caching
  refineCachePolicy(fusion);

  // if (params->vectorize_load_smem_to_regs) {
  //   for (auto tv : smem2reg_tvs) {
  //     tv->computeWith(-1, /*best_effort=*/true);
  //   }
  // }
}

} // namespace tma
} // namespace normalization_inner
} // namespace nvfuser
