// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#include <ops/arith.h>
#include <options.h>
#include <scheduler/normalization_utils.h>
#include <scheduler/runtime_info.h>
#include <scheduler/tools/inlining.h>

#include <ATen/cuda/CUDAContext.h>
namespace nvfuser {
namespace inner_outer_tma_warp_specialized {
void getHeuristics(
    ReductionParams* rparams,
    const int64_t outer_dim_numel,
    const int64_t inner_dim_numel,
    const int64_t regs_buffer_size,
    const int64_t circular_buffered_smem_size,
    const int64_t non_circular_buffered_smem_size,
    const size_t tmp_gmem_dtype_size,
    const size_t max_allowed_vect_factor,
    const int64_t hp_threads_per_block_min,
    const int64_t hp_threads_per_block_max,
    const bool project_to_input,
    const PrimDataType index_type) {
  rparams->tma_warp_specialized = true;
  rparams->project_persistent_buffers = project_to_input;
  rparams->cparams.index_type = index_type;
  const auto dev_prop = at::cuda::getCurrentDeviceProperties();
  const int64_t sm_count = (int64_t)dev_prop->multiProcessorCount;

  // Params for 1st stage, inner reduction and partial outer reduction.
  // Inner dim: inner_vect, inner_batch, and bdimx
  // Outer dim: iter_unroll, independent computation groups, SM count
  // Circular buffer: n_stages
  // Use the maximum vectorization factor
  const int64_t vect_factor = (int64_t)max_allowed_vect_factor;
  const int64_t after_vect = inner_dim_numel / vect_factor;
  const int64_t gdimy = sm_count;

  // Shared memory controls max possible circular buffer stages and iter
  // unrolls. Check shared memory usage it is calculated as:
  // (1) Non-circular buffered smem size
  // (2) Circular buffered smem size, which is proportional to [iter_unroll] and
  //     [n_stages]
  // (3) Mbarrier size, which is proportional to [n_stages], each
  //     stage requires 16 bytes for WAR and RAW.
  // (4) Reduction workspace size, which is proportional to [iter_unroll] and
  //     [n_computation_warps]
  auto is_enough_smem =
      [&](int64_t iter_unroll, int64_t n_stages, int64_t bdimx, int64_t bdimy) {
        // non-circular buffered and circular buffered smem size
        int64_t buffer_size = non_circular_buffered_smem_size +
            circular_buffered_smem_size * iter_unroll * n_stages;
        // mbarrier size
        int64_t mbarrier_size = 16 * n_stages;
        // reduction workspace size, need to be aligned to 128 bytes since
        // other smems are stacked on top of it directly, see
        // assignNextAddress in StackBasedSharedMemAllocator
        int64_t reduction_workspace_size = roundUpToMultiple(
            iter_unroll * bdimx * bdimy * tmp_gmem_dtype_size, 128);
        return (int64_t)dev_prop->sharedMemPerBlockOptin >=
            buffer_size + mbarrier_size + reduction_workspace_size;
      };

  // Check register usage,  it is calculated as:
  // (1) Used to further cache circular buffered tv, optional [true]
  // (2) Used to further cache non-circular buffered tv, optional [true]
  // (3) Used to cache partial outer reduction results
  // (4) overhead for indexing, etc.
  const bool is_circular_buffer_regs_cached = true;
  const bool is_non_circular_buffer_regs_cached = true;
  // Given a smem buffer size, calculate the number of registers pre thread
  // required to cache it in registers. The total required register size may be
  // larger than smem size due to non-divisible split.
  auto smem_to_regs =
      [&](int64_t smem_buffer_size, int64_t bdimx, int64_t iter_unroll) {
        int persistent_batch = ceilDiv(after_vect, bdimx);
        int buffer_per_element = smem_buffer_size / inner_dim_numel;
        int elements_per_thread = persistent_batch * iter_unroll * vect_factor;
        int buffer_per_thread = buffer_per_element * elements_per_thread;
        return buffer_per_thread / scheduler_utils::bytes_per_register;
      };
  auto is_enough_regs = [&](int64_t iter_unroll, int64_t bdimx) {
    int64_t reg_count = 0;
    // cache circular buffered tv
    if (is_circular_buffer_regs_cached) {
      reg_count +=
          smem_to_regs(circular_buffered_smem_size, bdimx, iter_unroll);
    }

    // cache non-circular buffered tv
    if (is_non_circular_buffer_regs_cached) {
      reg_count +=
          smem_to_regs(non_circular_buffered_smem_size, bdimx, iter_unroll);
    }
    // regs for partial outer reduction results.
    reg_count += regs_buffer_size / scheduler_utils::bytes_per_register;
    // regs for indexing, etc.
    reg_count += scheduler_utils::register_overhead;
    // total usage should be less than 255
    return reg_count <= scheduler_utils::max_registers_per_thread;
  };
  // bdimx: number of threads for inner dim.
  int64_t bdimx = std::max(128, hp_threads_per_block_min);
  // bdimy: number of independent warp groups for outer dim.
  int64_t bdimy = 1;
  // iter_unroll: unroll for outer dim, these rows are grouped together in TMA
  // load and reduction.
  int64_t iter_unroll = 1;
  // n_stages: circular buffer stages.
  int64_t n_stages = 1;
  NVF_ERROR(
      is_enough_smem(iter_unroll, n_stages, bdimx, bdimy),
      "Not enough shared memory for TMA warp specialized.");
  // Try to update paras in each loop and break if no update is made.
  while (1) {
    bool is_updated = false;
    // increase circular buffer stages
    if (is_enough_smem(iter_unroll, n_stages * 2, bdimx, bdimy)) {
      is_updated = true;
      n_stages *= 2;
    }
    // increase iter_unroll
    // iter_unroll should be divisible by outer_dim_numel due to limitation of
    // 1D TMA predicate.
    if (is_enough_smem(iter_unroll * 2, n_stages, bdimx, bdimy) &&
        outer_dim_numel % (iter_unroll * 2) == 0) {
      is_updated = true;
      iter_unroll *= 2;
    }
    // increase bdimx but don't exceed hp_threads_per_block_max and only when
    // registers are not enough, e.g. each thread has too many elements.
    if (bdimx * 2 <= hp_threads_per_block_max &&
        !is_enough_regs(iter_unroll, bdimx) &&
        is_enough_smem(iter_unroll, n_stages, bdimx * 2, bdimy)) {
      is_updated = true;
      bdimx *= 2;
    }

    // increase bdimy when bdimx is not increased
    // multiple independent computation groups only supports bdimx == 128
    // disable this option for now as runtime is not ready yet.
    if (false && bdimx == 128 &&
        is_enough_smem(iter_unroll, n_stages, bdimx, bdimy * 2)) {
      is_updated = true;
      bdimy *= 2;
    }
    if (!is_updated) {
      break;
    }
  }
  int64_t inner_batch = ceilDiv(after_vect, bdimx);

  // The inner reduction part of the kernel also does a partial outer reduction
  // and stores the partial results in tmp gmem and then reloaded to finish the
  // outer reduction. This function set the vectorization factor for write and
  // and read of the partial outer reduction result.
  // For write to tmp gmem, follows vectorization factor of inner reduction
  //                        but don't exceed 16 bytes.
  // For read from tmp gmem, since the parallelization is changed, a different
  //                         vectorization factor is used to optimize the
  //                         number of reductions per thread.
  constexpr int64_t max_gmem_vect_access_bytes = 16;
  const int64_t max_tmp_gmem_vect_factor = std::min(
      max_gmem_vect_access_bytes / (int64_t)tmp_gmem_dtype_size, vect_factor);
  int64_t tmp_gmem_write_vect = max_tmp_gmem_vect_factor;
  const int64_t workload_per_thread = inner_dim_numel >= 4096 ? 4l : 2l;
  int64_t vectorization_factor_outer =
      std::min(workload_per_thread, max_tmp_gmem_vect_factor);

  rparams->combined_split_grid_inner_dim =
      vectorization_factor_outer * bdimx * gdimy < inner_dim_numel;
  rparams->persistent_kernel = true;
  rparams->fastest_dim = true;
  rparams->combined_inner_outer = true;

  // Non Warp Specialized dim can't have more than 128 threads
  // see https://github.com/NVIDIA/Fuser/pull/4398.
  // Still want to use TIDy if bdimx <=128, which is more convenient for
  // ping-pong computations.
  ParallelType ws_pt = bdimx > 128 ? ParallelType::TIDx : ParallelType::TIDy;
  WarpSpecialized ws(ws_pt);
  int64_t computation_threads = bdimx * bdimy;
  int64_t total_threads =
      kWarpSpecializationPaddedThreads + computation_threads;
  if (total_threads > 256) {
    int64_t reg_per_thread = getRegPerThreadGivenThreadsPerSM(total_threads);
    // Assume each padded threads keep [tma_branch_registers] registers and all
    // others are moved to computation threads. The granularity is 8.
    // [tma_branch_registers] is a tunable parameter,
    int64_t tma_branch_registers = 32;
    int64_t compute_branch_registers = reg_per_thread +
        (reg_per_thread - tma_branch_registers) *
            kWarpSpecializationPaddedThreads / computation_threads;
    compute_branch_registers =
        scheduler_utils::roundDownToN(compute_branch_registers, 8);
    ws.num_registers =
        std::make_pair(tma_branch_registers, compute_branch_registers);
  }
  CircularBufferOptions circular_buffer_options{
      .type = ws, .stage = n_stages, .prefetch = n_stages - 1};
  rparams->circular_buffer_options = circular_buffer_options;

  rparams->unroll_factor_iter_dom = iter_unroll;
  rparams->vectorization_factor_outer = vectorization_factor_outer;
  rparams->vectorization_factor_tmp_gmem_write = tmp_gmem_write_vect;
  rparams->unroll_factor_inner_reduction = vect_factor;
  rparams->batches_per_block_inner_reduction = inner_batch;
  rparams->block_dim_inner_reduction = ParallelType::TIDx;
  rparams->vectorize_inner_reduction = vect_factor > 1;
  rparams->split_grid_dim_iter_dom_outer = true;
  rparams->grid_dim_iter_dom = ParallelType::BIDy;
  rparams->pad_inner_reduction_to_warp = true;

  rparams->lparams = LaunchParams(
      LaunchParams::UNINITIALIZED_VAL,
      gdimy,
      LaunchParams::UNINITIALIZED_VAL,
      n_stages > 1 && ws_pt == ParallelType::TIDx
          ? bdimx + kWarpSpecializationPaddedThreads
          : bdimx,
      LaunchParams::UNINITIALIZED_VAL,
      LaunchParams::UNINITIALIZED_VAL);

  rparams->tag = "TMA Warp Specialized Persistent Heuristic.\n";

  if (isDebugDumpEnabled(DebugDumpOption::SchedulerDebug)) {
    debug() << "\n===== Combined InnerOuter Reduction Stats ========\n"
            << "outer_dim_numel: " << outer_dim_numel << "\n"
            << "inner_dim_numel: " << inner_dim_numel << "\n"
            << "regs_buffer_size: " << regs_buffer_size << "\n"
            << "circular_buffered_smem_size: " << circular_buffered_smem_size
            << "\n"
            << "non_circular_buffered_smem_size: "
            << non_circular_buffered_smem_size << "\n"
            << "max_allowed_vect_factor: " << max_allowed_vect_factor << "\n"
            << "vectorization_factor_tmp_gmem_write: " << tmp_gmem_write_vect
            << "\n"
            << "vectorization_factor_outer: " << vectorization_factor_outer
            << "\n"
            << "bdimx: " << bdimx << "\n"
            << "bdimy: " << bdimy << "\n"
            << "gdimy: " << gdimy << "\n";
    debug() << "smem_persistent_buffers: " << "\n";
    for (auto buffer : rparams->smem_persistent_buffers) {
      debug() << buffer->toString() << "\n";
    }
    debug() << rparams->toString() << std::endl;
  }
}

void scheduleOuterReduction(
    Fusion* fusion,
    const ReductionParams* rparams,
    const std::vector<TensorView*>& outer_reduction_tvs,
    std::vector<TensorView*>& cached_gmem,
    std::vector<TensorView*>& cached_gmem_reload,
    std::vector<TensorView*>& outer_reference_tvs,
    std::unordered_set<TensorView*>& boundaryNodesSet) {
  auto mergeReductionOrIterDomains = [](TensorView* tv, bool mergeReduction) {
    int prev_i = -1;
    for (int i = static_cast<int>(tv->nDims()) - 1; i >= 0; i--) {
      if (mergeReduction == tv->axis(i)->isReduction()) {
        if (prev_i == -1) {
          prev_i = i;
        } else {
          tv->merge(i, prev_i);
          prev_i = i;
        }
      }
    }
  };
  for (auto& outer_reduction_tv : outer_reduction_tvs) {
    // Similar to the inner reduction, we need to reorder the outer reduction tv
    // when there are view operations.
    if (!ir_utils::getViewOps(fusion).empty()) {
      // Reorder reference_tv after propagating the view operation. This will
      // reorder for better merging.
      outer_reduction_tv->reorder(
          scheduler_utils::domainReorderAsLogicalMap(outer_reduction_tv));
    }

    // merge tensorview to [reduction, iteraiton] domains
    mergeReductionOrIterDomains(outer_reduction_tv, true);
    mergeReductionOrIterDomains(outer_reduction_tv, false);

    // First-stage of outer reduction
    // [R, I]
    std::vector<int64_t> rfactor_axes{0};
    if (rparams->unroll_factor_iter_dom > 1) {
      // [R/Unroll, Unroll]
      // Should mark as serial to avoid unrolling the outer reduction
      // which requires extra registers
      outer_reduction_tv->split(0, rparams->unroll_factor_iter_dom);
      outer_reduction_tv->axis(1)->parallelize(ParallelType::Serial);
      rfactor_axes.push_back(2);
    }
    // [R/Unroll/BIDy, BIDy, Unroll]
    outer_reduction_tv->split(0, rparams->lparams.gdimy());

    TensorView* partialResult = outer_reduction_tv->rFactor(rfactor_axes);
    partialResult->cacheBefore();
    partialResult->setMemoryType(MemoryType::Global);
    TensorView* partialResultReload = partialResult->cacheAfter();

    boundaryNodesSet.insert(partialResultReload);
    cached_gmem.emplace_back(partialResult);
    cached_gmem_reload.emplace_back(partialResultReload);

    // Second-stage of outer reduction
    // Unroll 1 to WAR bug in validateAndPropagatePType which propagates BIDy
    // to final outer reduction domain {132}
    // reduction domain, [I1/Unroll, Unroll]
    outer_reduction_tv->split(0, 1);
    outer_reduction_tv->axis(1)->parallelize(ParallelType::Unroll);
    // iteration domain, [BIDy, TIDx, Vect]
    int axisID = -1;
    if (rparams->vectorization_factor_outer > 1) {
      outer_reduction_tv->split(axisID, rparams->vectorization_factor_outer);
      outer_reduction_tv->axis(axisID--)->parallelize(ParallelType::Vectorize);
    }

    if (rparams->lparams.bdimx() > 1) {
      int64_t compute_bdimx = rparams->lparams.bdimx();
      if (std::holds_alternative<WarpSpecialized>(
              rparams->circular_buffer_options.type) &&
          std::get<WarpSpecialized>(rparams->circular_buffer_options.type).on ==
              ParallelType::TIDx) {
        compute_bdimx = rparams->lparams.bdimx() - 128;
      }

      outer_reduction_tv->split(axisID, compute_bdimx);
      outer_reduction_tv->axis(axisID--)->parallelize(ParallelType::TIDx);
    }

    if (rparams->combined_split_grid_inner_dim) {
      outer_reduction_tv->split(
          axisID, NamedScalar::getParallelDim(ParallelType::BIDy));
    }

    outer_reduction_tv->axis(axisID--)->parallelize(ParallelType::BIDy);
    outer_reference_tvs.emplace_back(outer_reduction_tv);
  }
}

void scheduleFusion(Fusion* fusion, const ReductionParams* rparams) {
  FusionGuard fg(fusion);

  // Grab the reduction, input, and output tensor views. dummy_outputs are
  // helper tensors for persistent buffer projection.
  std::vector<TensorView*> dummy_outputs, cached_inputs, reduction_tvs,
      smem_consumers;
  std::vector<std::pair<TensorView*, TensorView*>> cached_outputs;
  normalization_scheduler_utils::beforeSchedule(
      fusion,
      rparams,
      dummy_outputs,
      cached_inputs,
      reduction_tvs,
      smem_consumers,
      cached_outputs);

  // split reduction_tvs into inner and outer reduction_tvs
  std::vector<TensorView*> inner_reduction_tvs, outer_reduction_tvs;
  for (auto tv : reduction_tvs) {
    if (scheduler_utils::isFastestDimReduction(tv)) {
      inner_reduction_tvs.emplace_back(tv);
    } else {
      outer_reduction_tvs.emplace_back(tv);
    }
  }
  NVF_ERROR(
      !inner_reduction_tvs.empty(),
      "schedulePersistentKernelInnerOuter is called but no inner reduction is found.");
  NVF_ERROR(
      !outer_reduction_tvs.empty(),
      "schedulePersistentKernelInnerOuter is called but no outer reduction is found.");

  // schedule inner reduction, only schedule the first inner reduction tv,
  // then will be propagated to other inner reduction tvs.
  TensorView* inner_reference_tv =
      normalization_scheduler_utils::scheduleReductionGeneral(
          fusion,
          rparams,
          inner_reduction_tvs,
          SchedulerType::InnerOuterPersistent);

  // schedule outer reduction, schedule all the outer reduction tvs since we
  // need to store the intermediate results.
  std::vector<TensorView*> cached_gmem;
  std::vector<TensorView*> cached_gmem_reload;
  std::vector<TensorView*> outer_reference_tvs;
  std::unordered_set<TensorView*> boundaryNodesSet;
  scheduleOuterReduction(
      fusion,
      rparams,
      outer_reduction_tvs,
      cached_gmem,
      cached_gmem_reload,
      outer_reference_tvs,
      boundaryNodesSet);

  // Propagate inner reduction and outer reductions
  for (auto output : dummy_outputs) {
    fusion->addOutput(output);
  }

  // Collect tvs loaded with TMA, they require special scheduling.
  std::vector<TensorView*> tma_load_tvs;
  if (rparams->tma_warp_specialized) {
    for (auto tv : smem_consumers) {
      auto smem_tv = ir_utils::getSoleProducerTv(tv);
      if (std::find(tma_load_tvs.begin(), tma_load_tvs.end(), smem_tv) ==
          tma_load_tvs.end()) {
        tma_load_tvs.emplace_back(smem_tv);
      }
    }
  }

  const bool is_unroll_or_vectorization = rparams->isUnrolled();
  const bool is_vectorize =
      rparams->vectorize_inner_reduction || rparams->vectorize_iter_dom;
  const bool group_inner_reduction = rparams->unroll_factor_iter_dom > 1;
  // The first part of the outer reduction is grid-stride thread local
  // reduction, can't be grouped. The second part of the outer reduction is
  // thread local reduction, can't be grouped.
  const bool group_outer_reduction = false;

  // Propagate transformations for inner reduction.
  // Two steps are used since tma tvs are scheduled differently.
  // Step-1, propagate iteration domain in inner reduction.
  // Step-2, propagate reduction domain in inner reduction.
  if (rparams->tma_warp_specialized) {
    // Find the axis that splits the reduction domain and iteration domain.
    int first_redu_axis = -1;
    int n_dims = (int)inner_reference_tv->nDims();
    for (auto i = 0; i < n_dims; i++) {
      if (inner_reference_tv->axis(i)->isReduction() ||
          inner_reference_tv->axis(i)->isRFactorProduct()) {
        first_redu_axis = i;
        break;
      }
    }

    // Step-1, propagate iteration domain in inner reduction.
    // outer_reference_tvs are excluded since they are already scheduled
    // with a different pattern for the final step of outer reduciton.
    if (first_redu_axis > 0) {
      TransformPropagator propagator(inner_reference_tv, first_redu_axis - 1);
      std::vector<TensorView*> all_tvs_except = ir_utils::allTvsExcept(
          fusion, {outer_reference_tvs.begin(), outer_reference_tvs.end()});
      SetSelector selector({all_tvs_except.begin(), all_tvs_except.end()});
      MaxLogicalDomainInfoSpanningTree(inner_reference_tv, &selector)
          .traverse(&propagator);
    }

    // Step-2, propagate reduction domain in inner reduction.
    // (a) Tvs in boundaryNodesSet are excluded since they should follow outer
    // reduction pattern.
    // (b) TMA tvs are excluded since they require special scheduling.
    // (3) Excluding tma tvs breaks the propagation path from inner reduction
    // tv to cached_gmem which stores the results of the first-stage of outer
    // reduction. The solution is adding a dummy output to link them. The same
    // trick is used when projecting persistent buffers to inputs.
    auto inner_reduction_input =
        ir_utils::getSoleProducerTv(inner_reference_tv);
    for (auto tv : cached_gmem) {
      // T1(smem) --> T2 (l) --> T3 = OuterRedu(T2) --> T4(cached_gmem)
      // outer_reduction_input: T2
      // partial_outer_redu_tv: T3
      auto partial_outer_redu_tv = ir_utils::getSoleProducerTv(tv);
      auto outer_reduction_input =
          ir_utils::getSoleProducerTv(partial_outer_redu_tv);
      auto dummy_output = add(inner_reduction_input, outer_reduction_input);
      fusion->addOutput(dummy_output);
      dummy_outputs.emplace_back(dummy_output);
    }

    // Tvs requiring special scheduling
    std::unordered_set<TensorView*> special_tvs{
        tma_load_tvs.begin(), tma_load_tvs.end()};
    for (auto tv : boundaryNodesSet) {
      if (special_tvs.count(tv) == 0) {
        special_tvs.emplace(tv);
      }
    }
    TransformPropagator propagator(inner_reference_tv);
    std::vector<TensorView*> all_tvs_except_cache = ir_utils::allTvsExcept(
        fusion, {special_tvs.begin(), special_tvs.end()});
    SetSelector selector(
        {all_tvs_except_cache.begin(), all_tvs_except_cache.end()});
    MaxLogicalDomainInfoSpanningTree(inner_reference_tv, &selector)
        .traverse(&propagator);
  } else {
    reduction_scheduler_utils::propagateTransformation(
        inner_reference_tv, boundaryNodesSet);
  }
  reduction_scheduler_utils::propagateRFactor(
      inner_reference_tv, inner_reduction_tvs[0], inner_reduction_tvs);

  // parallelization propagation
  const auto& selected_tvs_inner =
      scheduler_utils::getAllTvsFrom(inner_reduction_tvs, boundaryNodesSet);
  const auto& unroll_vectorizable_cached_tvs =
      reduction_scheduler_utils::getCachedTvsToUnrollOrVectorize(
          inner_reference_tv, is_vectorize, cached_inputs, cached_outputs);
  reduction_scheduler_utils::propagateParallelization(
      inner_reduction_tvs[0],
      inner_reference_tv,
      is_unroll_or_vectorization,
      group_inner_reduction,
      inner_reduction_tvs,
      unroll_vectorizable_cached_tvs,
      {selected_tvs_inner.begin(), selected_tvs_inner.end()});

  // Propagate outer reduction. Each outer reduction is connected with its
  // cached_gmem and output, since we added all the cached_gmem to the
  // boundaryNodesSet, the transformation from one outer reduction can't
  // propagate to other outer reductions due to the cutoff at
  // boundaryNodesSet. Thus, we need a loop to initiate the propagation from
  // each outer reduction. Don't allow parallelization propagation goes
  // through cached_gmem, see issue 246.
  for (long unsigned int i = 0; i < outer_reference_tvs.size(); i++) {
    const auto& selected_tvs_outer = scheduler_utils::getAllTvsFrom(
        {outer_reduction_tvs[i]}, {cached_gmem[i]});
    reduction_scheduler_utils::propagateTransformation(
        outer_reference_tvs[i], boundaryNodesSet);
    const auto& unroll_vectorizable_cached_tvs =
        reduction_scheduler_utils::getCachedTvsToUnrollOrVectorize(
            outer_reference_tvs[i],
            is_vectorize,
            cached_inputs,
            cached_outputs);
    reduction_scheduler_utils::propagateParallelization(
        outer_reduction_tvs[i],
        outer_reference_tvs[i],
        is_unroll_or_vectorization,
        group_outer_reduction,
        outer_reduction_tvs,
        unroll_vectorizable_cached_tvs,
        {selected_tvs_outer.begin(), selected_tvs_outer.end()});
  }

  // Up to this point, the outer dimension of the TMA tv is scheduled
  // the same way as the inner reduction tv. However, the inner dimension
  // has not been scheduled yet. Since 1D TMA allows unrestricted load size,
  // we can simply parallelize the entire inner dimension using bulk.
  // Example: 2D tensor, [BIDy, S, | Bulk]
  // Example: 1D tensor, [Bulk]
  if (rparams->tma_warp_specialized) {
    for (auto tv : tma_load_tvs) {
      tv->axis(-1)->parallelize(ParallelType::Bulk);
    }
  }

  // special vectorization of temp gmem, vectorization_factor_tmp_gmem_write
  // is guaranteed to be smaller or equal to input vectorization factor.
  if (rparams->vectorization_factor_tmp_gmem_write > 1) {
    for (auto tv : cached_gmem) {
      NVF_ERROR(
          rparams->vectorization_factor_tmp_gmem_write <=
              rparams->unroll_factor_inner_reduction,
          "vectorization factor of temp gmem write should be smaller than that of inner reduction.")
      if (rparams->vectorization_factor_tmp_gmem_write <
          rparams->unroll_factor_inner_reduction) {
        tv->split(-1, rparams->vectorization_factor_tmp_gmem_write);
      }
      tv->axis(-1)->parallelize(ParallelType::Vectorize);
    }
  }
  // vectorization propagate through propagateParallelization only works for
  // input and output tensors. propagate vectorization to cached_gmem_reload
  // directly from output tv using parallelizeAllLike. must propagate
  // seperaely for different tvs as outer reductions are transformed
  // seperately.
  if (rparams->vectorization_factor_outer > 1) {
    for (auto tv : cached_gmem_reload) {
      auto output_tvs = ir_utils::outputTvsOf(tv);
      NVF_ERROR(
          !output_tvs.empty(),
          "cached_gmem_reload should have at least one output tensor.")
      scheduler_utils::parallelizeAllLike(
          output_tvs[0],
          -1,
          {cached_gmem_reload.begin(), cached_gmem_reload.end()},
          {ParallelType::Vectorize});
    }
  }

  // Needs special handling of vectorized loading from shared memory due to
  // potential different data types of inputs and shared memory tensor.
  if (is_vectorize) {
    reduction_scheduler_utils::sharedMemoryConsumerVectorization(
        smem_consumers, rparams->unroll_factor_inner_reduction);
  }

  // Remove dummy outputs as they can inadvertently affect CA positions
  for (auto output : dummy_outputs) {
    fusion->removeOutput(output);
  }
  // inline
  if (rparams->circular_buffer_options.isEnable()) {
    std::unordered_map<TensorView*, int64_t> tv_inline_pos_map;
    // TMA loaded tv may have a domain of either
    // [I/Unroll/BIDy, BIDy, Unroll | Bulk]
    // or
    // [I/BIDy, BIDy, | Bulk]
    // or
    // [Bulk]
    // Set inline position after BIDy, so all the unrolled TMA loads
    // share the same barrier.
    constexpr int64_t tma_inline_pos = 2;
    for (auto tv : tma_load_tvs) {
      if (tv->nDims() >= tma_inline_pos + 1) {
        tv_inline_pos_map.emplace(tv, tma_inline_pos);
      }
    }
    // For smem consumers, set inline position to the same as tma load tvs.
    // This allows quick release the shared memory barrier to launch the
    // next TMA load. Otherwise, the inline position is to the right of the
    // Unroll axis. Which requires the tma tensor alive until the end of the
    // computation and delays the next TMA load until the end of the
    // computation.
    for (auto tv : smem_consumers) {
      if (ir_utils::getSoleProducerTv(tv)->nDims() >= tma_inline_pos + 1) {
        tv_inline_pos_map.emplace(tv, tma_inline_pos);
      }
    }

    // Cached input with inner bcast [Iter, 1], is not marked as vectorizable
    // in getCachedTvsToUnrollOrVectorize(). However, if iteration domain is
    // unrolled, the tv is scheduled as: [I/Unroll/BIDy, BIDy, Unroll, ...],
    // the unrolled domain, axis-2, can be vectorized.
    if (group_inner_reduction) {
      auto is_redu_mapped_to_bcast = [](TensorView* redu_tv,
                                        TensorView* bcast_tv) {
        if (bcast_tv->nDims() != redu_tv->nDims()) {
          return false;
        }
        for (int i = 0; i < bcast_tv->nDims(); i++) {
          if ((redu_tv->axis(i)->isReduction() ||
               redu_tv->axis(i)->isRFactorProduct()) &&
              !bcast_tv->axis(i)->isBroadcast()) {
            return false;
          }
        }
        return true;
      };

      // Heuristic ensures the iteration dim is divisible by the unroll
      // factor. Here, we only need to further confirm all the iteration
      // domains are contiguous.
      auto can_vectorize = [](TensorView* redu_tv, TensorView* bcast_tv) {
        const auto& alloc_dom_1 = redu_tv->getMaybeAllocationDomain();
        const auto& alloc_dom_2 = bcast_tv->getMaybeAllocationDomain();
        if (alloc_dom_1.size() != alloc_dom_2.size()) {
          return false;
        }
        const auto& contiguity = bcast_tv->domain()->contiguity();
        for (int i = 0; i < (int)alloc_dom_1.size(); i++) {
          if (alloc_dom_1[i]->isReduction()) {
            break;
          }
          if (!contiguity[i].has_value() || !contiguity[i].value()) {
            return false;
          }
        }
        return true;
      };
      for (auto cached_tv : cached_inputs) {
        if (cached_tv->hasBroadcast() &&
            is_redu_mapped_to_bcast(inner_reference_tv, cached_tv)) {
          if (can_vectorize(inner_reference_tv, cached_tv)) {
            cached_tv->axis(2)->parallelize(ParallelType::Vectorize);
          } else {
            cached_tv->axis(2)->parallelize(ParallelType::Unroll);
          }
        }
        // Unroll the consumers to prevent inlineMost from inlining them
        // to the right of the vectorized axis, which can cause expression
        // sort errors.
        // TODO: Revise inlineMost to handle this automatically.
        // TODO: Ideally, we only need to unroll the consumers that are
        // used in the for-loop before and after the iteration grouped
        // reduction, we will leave this for heuristic tuning since unroll all
        // consumers may lead to better performance if register usage is not a
        // concern.
        for (auto consumer : ir_utils::consumerTvsOf(cached_tv)) {
          consumer->axis(2)->parallelize(ParallelType::Unroll);
        }
      }
    }

    std::unordered_set<TensorView*> exclude_tvs;
    for (auto [k, v] : tv_inline_pos_map) {
      exclude_tvs.insert(k);
      inlineSelectedAt({k}, k, v);
    }
    std::vector<TensorView*> inline_most_tvs =
        ir_utils::allTvsExcept(fusion, exclude_tvs);
    inlineMost(inline_most_tvs);
    int64_t number_of_stages = rparams->circular_buffer_options.stage;
    int64_t prefetch_distance = rparams->circular_buffer_options.prefetch;
    CircularBufferType circular_buffer_type =
        rparams->circular_buffer_options.type;
    for (auto tv : tma_load_tvs) {
      // Circular buffer requires a valid axis to circulate on, and only
      // applies to TVs with a computeAt position. For example,  the weight
      // tensor in RMS Norm Bwd is scheduled as: T36_s___bfloat[iB91{i2}]
      //  logical domain : (iB91{i2})
      //  contiguity: t
      //  loop domain : (iB91{i2})
      // There is no way to apply circular buffer to this tensor.
      if (tv->getComputeAtPosition() > 0) {
        tv->circularBuffer(
            number_of_stages, prefetch_distance, circular_buffer_type);
      }
    }
  } else {
    inlineMost();
  }
}
} // namespace inner_outer_tma_warp_specialized
} // namespace nvfuser
