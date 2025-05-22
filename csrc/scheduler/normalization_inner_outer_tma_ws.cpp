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
    const int64_t smem_buffer_size,
    const int64_t smem_overhead,
    const size_t tmp_gmem_dtype_size,
    const size_t vectorize_factor,
    const int64_t hp_threads_per_block_min,
    const int64_t hp_threads_per_block_max,
    const bool project_to_input,
    const PrimDataType index_type) {
  rparams->tma_warp_specialized = true;
  rparams->project_persistent_buffers = project_to_input;
  rparams->cparams.index_type = index_type;
  const auto dev_prop = at::cuda::getCurrentDeviceProperties();
  const int64_t device_multiprocessor_count =
      (int64_t)dev_prop->multiProcessorCount;
  // Parameters for inner reduction:
  // Reduction dim: inner_vect, inner_batch, and bdimx
  // Iteration dim: gdimy

  // Parameters for outer reduction:
  // Reduction dim: serial
  // Iteration dim: vectorization_factor_outer, bdimx, gdimy
  struct InnerOuterParams {
    int64_t inner_vect = -1;
    int64_t inner_batch = -1;
    int64_t gdimy = -1;
    int64_t tmp_gmem_write_vect = -1;
    int64_t vectorization_factor_outer = -1;
    int64_t bdimx = -1;
    // derived metrics for sorting
    int64_t warps_per_sm = -1;
    int64_t required_register_per_thread = -1;
    int64_t available_register_per_thread = -1;

    void verify() {
      NVF_ERROR(inner_vect != -1, "inner_vect is not set.");
      NVF_ERROR(inner_batch != -1, "inner_batch is not set.");
      NVF_ERROR(bdimx != -1, "bdimx is not set.");
      NVF_ERROR(gdimy != -1, "gdimy is not set.");
      NVF_ERROR(tmp_gmem_write_vect != -1, "tmp_gmem_write_vect is not set.");
      NVF_ERROR(
          vectorization_factor_outer != -1,
          "vectorization_factor_outer is not set.");
    }
    std::string toString() const {
      std::stringstream ss;
      ss << "inner_vect: " << inner_vect << ", inner_batch: " << inner_batch
         << ", gdimy: " << gdimy
         << ", tmp_gmem_write_vect: " << tmp_gmem_write_vect
         << ", vectorization_factor_outer: " << vectorization_factor_outer
         << ", bdimx: " << bdimx << ", warps_per_sm: " << warps_per_sm
         << ", required_register_per_thread: " << required_register_per_thread
         << ", available_register_per_thread: "
         << available_register_per_thread;
      return ss.str();
    }
  };

  // Estimate register usage per thread based on buffer size.
  // Assuming a constant register overhead for non-buffer related usage,
  // and all the register buffers are stored in registers.
  auto get_estimated_register_usage = [&](int64_t batch_mul_vect) {
    int64_t persistent_buffer_size =
        regs_buffer_size / inner_dim_numel * batch_mul_vect;
    int64_t estimated_register_count =
        persistent_buffer_size / scheduler_utils::bytes_per_register +
        scheduler_utils::register_overhead;
    return std::min(
        estimated_register_count, scheduler_utils::max_registers_per_thread);
  };

  // The inner reduction part of the kernel also does a partial outer reduction
  // and stores the partial results in tmp gmem and then reloaded to finish the
  // outer reduciton. This function set the vectorization factor for write and
  // and read of the partial outer reduction result.
  // For write to tmp gmem, follows vectorization factor of inner reduction
  //                        but don't exceed 16 bytes.
  // For read from tmp gmem, since the paralelization is changed, a different
  //                         vectorization factor is used to optimize the
  //                         number of reaductions per thread.
  auto get_outer_reduction_buffer_vect_factor = [&](int64_t inner_vect) {
    constexpr int64_t max_gmem_vect_access_bytes = 16;
    const int64_t max_tmp_gmem_vect_factor = std::min(
        max_gmem_vect_access_bytes / (int64_t)tmp_gmem_dtype_size, inner_vect);
    int64_t tmp_gmem_write_vect = max_tmp_gmem_vect_factor;
    const int64_t workload_per_thread = inner_dim_numel >= 4096 ? 4l : 2l;
    int64_t vectorization_factor_outer =
        std::min(workload_per_thread, max_tmp_gmem_vect_factor);
    return std::make_pair(tmp_gmem_write_vect, vectorization_factor_outer);
  };

  // Get the heuristics given vectorization factor and bdimx.
  auto get_heuristics_given_vect_threads = [&](int64_t vect_factor,
                                               int64_t bdimx) {
    InnerOuterParams iop;
    // (1) inner reduction
    // Reduction dim: inner_batch, bdimx, vect_factor
    // Iteration dim: gdimy
    iop.inner_vect = vect_factor;
    iop.bdimx = bdimx;
    iop.inner_batch = ceilDiv(inner_dim_numel / iop.inner_vect, iop.bdimx);
    iop.gdimy = device_multiprocessor_count;

    // (2) outer reduction
    // Iteration dim: gdimy, bdimx, vectorization_factor_outer
    // Reduction dim: serial
    std::tie(iop.tmp_gmem_write_vect, iop.vectorization_factor_outer) =
        get_outer_reduction_buffer_vect_factor(iop.inner_vect);
    // (3) Derived metrics warps_per_sm and register usage for sorting
    iop.warps_per_sm = ceilDiv(iop.bdimx, dev_prop->warpSize) * iop.gdimy /
        device_multiprocessor_count;
    iop.available_register_per_thread =
        getRegPerThreadGivenThreadsPerSM(dev_prop->warpSize * iop.warps_per_sm);
    iop.required_register_per_thread =
        get_estimated_register_usage(iop.inner_vect * iop.inner_batch);
    return iop;
  };

  // Use the maximum vectorization factor
  const int64_t vect_factor = (int64_t)vectorize_factor;

  // Set bdimx, will be revised in heuristics tuning.
  // bdimy is reserved for warp specialization
  // bdimz is not used
  const int64_t max_persistent_batch = 7L;
  const int64_t after_vect = inner_dim_numel / vect_factor;
  int64_t bdimx = std::min(128L, after_vect);
  bdimx = std::max(bdimx, ceilDiv(after_vect, max_persistent_batch));
  bdimx = scheduler_utils::roundUpToN(bdimx, 128L);

  auto iop = get_heuristics_given_vect_threads(vect_factor, bdimx);
  rparams->combined_split_grid_inner_dim =
      iop.vectorization_factor_outer * iop.bdimx * iop.gdimy < inner_dim_numel;

  // check all the parameters in InnerOuterParams are set.
  iop.verify();

  rparams->persistent_kernel = true;
  rparams->fastest_dim = true;
  rparams->combined_inner_outer = true;

  // TODO: This is a heuristic, need to be tuned.
  // Set circular buffer, n_stages and n_prefetch are tunable parameters.
  // n_stages is also limited by smem.
  // Each circular buffer stage requires two smem barriers, one for RAW
  // (computation), the other for WAR (TMA loading). SMEM is aligned at
  // 128 Bytes, so we add 128 bytes for barriers, which allows a maximum
  // of 128 / (sizeof(uint64) * 2) = 16 stages.
  const int64_t max_stages = 16;
  const int64_t aligned_mbarrier_size = 128;
  const int64_t available_smem = (int64_t)dev_prop->sharedMemPerBlockOptin -
      smem_overhead - aligned_mbarrier_size;
  const int64_t max_n_copies =
      std::min(available_smem / smem_buffer_size, max_stages);
  NVF_ERROR(
      max_n_copies > 0,
      "Not enough shared memory for circular buffer, smem_buffer_size: ",
      smem_buffer_size,
      ", available_smem: ",
      available_smem);
  std::cout << "sharedMemPerBlockOptin: " << dev_prop->sharedMemPerBlockOptin
            << std::endl;
  std::cout << "smem_overhead: " << smem_overhead << std::endl;
  std::cout << "available_smem: " << available_smem << std::endl;
  std::cout << "smem_buffer_size: " << smem_buffer_size << std::endl;
  std::cout << "max_n_copies: " << max_n_copies << std::endl;

  int64_t iter_remaining = ceilDiv(outer_dim_numel, iop.gdimy);
  int64_t n_stages_prefered = std::min(2L, iter_remaining);
  int64_t n_stages = std::min(n_stages_prefered, max_n_copies);
  int64_t n_prefetch = n_stages - 1L;
  // Non Warp Specialized dim can't have more than 128 threads
  // see https://github.com/NVIDIA/Fuser/pull/4398.
  // Still want to use TIDy if bdimx <=128, which is more convenient for
  // ping-pong computations.
  ParallelType ws_pt =
      iop.bdimx > 128 ? ParallelType::TIDx : ParallelType::TIDy;
  WarpSpecialized ws(ws_pt);
  // This is a heuristic para, multiple independent computation warp groups
  // is not supported yet. Only need to enable register sharing when there
  // are more than 256 threads, otherwise each thread can use 255 registers
  // which is already the max allowed number.
  int64_t independent_computation_groups = 1;
  int64_t computation_threads = iop.bdimx * independent_computation_groups;
  int64_t total_threads = ws_padded_threads + computation_threads;
  if (total_threads > 256) {
    int64_t reg_per_thread = getRegPerThreadGivenThreadsPerSM(total_threads);
    // Assume each padded threads keep [tma_branch_registers] registers and all
    // others are moved to computation threads. The granularity is 8.
    // [tma_branch_registers] is a tunable parameter,
    int64_t tma_branch_registers = 32;
    int64_t compute_branch_registers = reg_per_thread +
        (reg_per_thread - tma_branch_registers) * ws_padded_threads /
            computation_threads;
    compute_branch_registers =
        scheduler_utils::roundDownToN(compute_branch_registers, 8);
    ws.num_registers =
        std::make_pair(tma_branch_registers, compute_branch_registers);
  }
  CircularBufferOptions circular_buffer_options{
      .type = ws, .stage = n_stages, .prefetch = n_prefetch};
  rparams->circular_buffer_options = circular_buffer_options;

  // TODO: This is a heuristic, need to be tuned.
  // Iteration unroll factor, limited by:
  // (1) heuristic selection
  // (2) max possible due to smem limitation
  // (3) Predicate of 1D TMA load requires iteration dim divisible by unroll
  //     factor.
  // (4) Round down to power of 2, since we need vectorized access in
  //     smem reduction and loading of inner broadcast tv.
  iter_remaining = scheduler_utils::safeDiv(iter_remaining, n_stages);
  int64_t heu_iter_unroll = std::min(2L, iter_remaining);
  int64_t max_iter_unroll = max_n_copies / n_stages;
  int64_t iter_unroll_factor = std::min(heu_iter_unroll, max_iter_unroll);
  iter_unroll_factor = scheduler_utils::lastPow2(iter_unroll_factor);
  while (outer_dim_numel % iter_unroll_factor) {
    iter_unroll_factor /= 2;
  }
  rparams->unroll_factor_iter_dom = iter_unroll_factor;
  rparams->vectorization_factor_outer = iop.vectorization_factor_outer;
  rparams->vectorization_factor_tmp_gmem_write = iop.tmp_gmem_write_vect;
  rparams->cparams.maxrregcount = iop.available_register_per_thread;
  rparams->unroll_factor_inner_reduction = iop.inner_vect;
  rparams->batches_per_block_inner_reduction = iop.inner_batch;
  rparams->block_dim_inner_reduction = ParallelType::TIDx;
  rparams->vectorize_inner_reduction = iop.inner_vect > 1;
  rparams->split_grid_dim_iter_dom_outer = true;
  rparams->grid_dim_iter_dom = ParallelType::BIDy;
  rparams->pad_inner_reduction_to_warp = true;

  rparams->lparams = LaunchParams(
      LaunchParams::UNINITIALIZED_VAL,
      iop.gdimy,
      LaunchParams::UNINITIALIZED_VAL,
      n_stages > 1 && ws_pt == ParallelType::TIDx
          ? iop.bdimx + ws_padded_threads
          : iop.bdimx,
      LaunchParams::UNINITIALIZED_VAL,
      LaunchParams::UNINITIALIZED_VAL);

  rparams->tag = "TMA Warp Specialized Persistent Heuristic.\n";

  if (isDebugDumpEnabled(DebugDumpOption::SchedulerDebug)) {
    debug() << "\n===== Combined InnerOuter Reduction Stats ========\n"
            << "outer_dim_numel: " << outer_dim_numel << "\n"
            << "inner_dim_numel: " << inner_dim_numel << "\n"
            << "regs_buffer_size: " << regs_buffer_size << "\n"
            << "smem_buffer_size: " << smem_buffer_size << "\n"
            << "smem_overhead: " << smem_overhead << "\n"
            << "vectorize_factor_input: " << iop.inner_vect << "\n"
            << "vectorization_factor_tmp_gmem_write: "
            << iop.tmp_gmem_write_vect << "\n"
            << "vectorization_factor_outer: " << iop.vectorization_factor_outer
            << "\n"
            << "bdimx: " << iop.bdimx << "\n"
            << "gdimy: " << iop.gdimy << "\n";
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
        const auto& alloc_dom_1 = redu_tv->getMaybeRootDomain();
        const auto& alloc_dom_2 = bcast_tv->getMaybeRootDomain();
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
        std::cout << "cached_tv: " << cached_tv->toString() << std::endl;
        if (!cached_tv->hasBroadcast()) {
          continue;
        }
        if (!is_redu_mapped_to_bcast(inner_reference_tv, cached_tv)) {
          continue;
        }

        if (can_vectorize(inner_reference_tv, cached_tv)) {
          cached_tv->axis(2)->parallelize(ParallelType::Vectorize);
        } else {
          cached_tv->axis(2)->parallelize(ParallelType::Unroll);
        }
      }
      for (auto tv : fusion->allTvs()) {
        if (tv->isFusionInput() || tv->isFusionOutput()) {
          continue;
        }
        if (!tv->hasBroadcast()) {
          continue;
        }
        if (!is_redu_mapped_to_bcast(inner_reference_tv, tv)) {
          continue;
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
        // if(tv->definition()->isA<UnaryOp>() &&
        //    tv->definition()->as<UnaryOp>()->getUnaryOpType() ==
        //        UnaryOpType::Reciprocal) {
        //   tv->axis(2)->parallelize(ParallelType::Unroll);
        // }
          tv->axis(2)->parallelize(ParallelType::Unroll);
          // if (tv->axis(2)->getParallelType() != ParallelType::Vectorize) {
        //   tv->axis(2)->parallelize(ParallelType::Unroll);
        // }
        // tv_inline_pos_map.emplace(tv, tma_inline_pos);
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
  fusion->printMath();
}
} // namespace inner_outer_tma_warp_specialized
} // namespace nvfuser
