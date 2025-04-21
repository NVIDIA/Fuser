// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#include <ops/arith.h>
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
  // Reduction dim: inner_vect, inner_batch, bdimx and bdimy
  // Iteration dim: gdimy

  // Parameters for outer reduction:
  // Reduction dim: bdimy
  // Iteration dim: vectorization_factor_outer, bdimx, gdimy
  struct InnerOuterParams {
    int64_t inner_vect = -1;
    int64_t inner_batch = -1;
    int64_t bdimx = -1;
    int64_t bdimy = -1;
    int64_t bdimz = -1;
    int64_t gdimy = -1;
    int64_t tmp_gmem_write_vect = -1;
    int64_t vectorization_factor_outer = -1;
    int64_t threads_per_block = -1;
    // derived metrics for sorting
    int64_t warps_per_sm = -1;
    int64_t required_register_per_thread = -1;
    int64_t available_register_per_thread = -1;

    void verify() {
      NVF_ERROR(inner_vect != -1, "inner_vect is not set.");
      NVF_ERROR(inner_batch != -1, "inner_batch is not set.");
      NVF_ERROR(bdimx != -1, "bdimx is not set.");
      NVF_ERROR(bdimy != -1, "bdimy is not set.");
      NVF_ERROR(gdimy != -1, "gdimy is not set.");
      NVF_ERROR(tmp_gmem_write_vect != -1, "tmp_gmem_write_vect is not set.");
      NVF_ERROR(
          vectorization_factor_outer != -1,
          "vectorization_factor_outer is not set.");
    }
    std::string toString() const {
      std::stringstream ss;
      ss << "inner_vect: " << inner_vect << ", inner_batch: " << inner_batch
         << ", bdimx: " << bdimx << ", bdimy: " << bdimy << ", bdimz: " << bdimz
         << ", gdimy: " << gdimy
         << ", tmp_gmem_write_vect: " << tmp_gmem_write_vect
         << ", vectorization_factor_outer: " << vectorization_factor_outer
         << ", threads_per_block: " << threads_per_block
         << ", warps_per_sm: " << warps_per_sm
         << ", required_register_per_thread: " << required_register_per_thread
         << ", available_register_per_thread: "
         << available_register_per_thread;
      return ss.str();
    }
  };

  // Set a minimum workload for each thread to take advantage of low
  // intra-threads communication cost.
  // Tuned for layer_norm backward on A100, still works fine on H100.
  auto get_minimum_batch = [&]() -> int64_t {
    if (inner_dim_numel >= 3072l) {
      if (outer_dim_numel <= 2048l && inner_dim_numel == 3072l) {
        return 3l;
      } else {
        return 4l;
      }
    } else if (inner_dim_numel >= 2048l) {
      return 2l;
    }
    return 1l;
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

  // In the outer reduction part of the kernel, inner and outer dims are
  // parallelized as:
  // --- inner dim: vect, bdimx, gdimy ----
  // --- outer dim: bdimy -----------------
  // This function splits the threads_per_block into bdimx and bdimy using:
  // bdimx = ceilDiv(inner_dim_numel / vect, gdimy)
  // bdimy = threads_per_block / bdimx
  auto get_bdimx_bdimy = [&](int64_t threads_per_block,
                             int64_t vectorization_factor_outer,
                             int64_t gdimy) {
    // For widely used hidden sizes, threads_per_block has factor of 8, roundup
    // to increase the probability of bdimx * bdimy == threads_per_block.
    int64_t bdimx = scheduler_utils::roundUpPow2Or8(
        ceilDiv(inner_dim_numel / vectorization_factor_outer, gdimy));
    // if still not divisible, e.g. threads_per_block = 256, bdimx = 40.
    // increase bdimx to make it divisible. Under worst case, bdimx equals to
    // threads_per_block.
    while (threads_per_block % bdimx) {
      bdimx = std::min(bdimx + 8, threads_per_block);
    }
    // Set OuterParams Reduction dim: bdimy.
    int64_t bdimy = threads_per_block / bdimx;
    NVF_ERROR(
        bdimy * bdimx == threads_per_block,
        " threads_per_block must be divisible by bdimx and bdimy.");
    return std::make_pair(bdimx, bdimy);
  };

  // Get the heuristics given vectorization factor and threads per block
  auto get_heuristics_given_vect_threads = [&](int64_t vect_factor,
                                               int64_t threads_per_block) {
    InnerOuterParams iop;
    // (1) inner reduction
    // Reduction dim: inner_batch, threads_per_block, vect_factor
    // Iteration dim: gdimy
    iop.inner_vect = vect_factor;
    iop.threads_per_block = threads_per_block;
    iop.inner_batch =
        ceilDiv(inner_dim_numel / iop.inner_vect, iop.threads_per_block);
    iop.gdimy = device_multiprocessor_count;

    // (2) outer reduction
    // Iteration dim: gdimy, bdimx, vectorization_factor_outer
    // Reduction dim: bdimy
    std::tie(iop.tmp_gmem_write_vect, iop.vectorization_factor_outer) =
        get_outer_reduction_buffer_vect_factor(iop.inner_vect);
    auto [bdimx, bdimy] = get_bdimx_bdimy(
        threads_per_block, iop.vectorization_factor_outer, iop.gdimy);
    iop.bdimx = bdimx;
    iop.bdimy = bdimy;
    // (3) Derived metrics warps_per_sm and register usage for sorting
    iop.warps_per_sm = ceilDiv(iop.threads_per_block, dev_prop->warpSize) *
        iop.gdimy / device_multiprocessor_count;
    iop.available_register_per_thread =
        getRegPerThreadGivenThreadsPerSM(dev_prop->warpSize * iop.warps_per_sm);
    iop.required_register_per_thread =
        get_estimated_register_usage(iop.inner_vect * iop.inner_batch);
    return iop;
  };

  // Use the maximum vectorization factor
  const int64_t vect_factor = (int64_t)vectorize_factor;

  // Set a reasonable range for threads per block based on the number of
  // elements in the inner dimension after vectorization.
  // Start from 128 or a smaller number if inner dim is small.
  const int64_t after_vect = inner_dim_numel / vect_factor;
  const int64_t batch_min = get_minimum_batch();
  int64_t threads_per_block_min = hp_threads_per_block_min;
  threads_per_block_min = std::min(threads_per_block_min, after_vect);
  threads_per_block_min = scheduler_utils::roundUpPow2(threads_per_block_min);

  // star max threads per block from min threads per block
  int64_t threads_per_block_max = threads_per_block_min;
  // increase to cover the whole inner dim
  threads_per_block_max =
      std::max(threads_per_block_max, ceilDiv(after_vect, batch_min));
  // round up to power of 2
  threads_per_block_max = scheduler_utils::roundUpPow2(threads_per_block_max);
  // don't go beyond the maximum threads per block
  threads_per_block_max =
      std::min(threads_per_block_max, hp_threads_per_block_max);

  // Store all the possible heuristics based on different threads per block.
  // Vectorizaton is fixed at the maximum value.
  std::vector<InnerOuterParams> iop_candidates;
  for (auto threads_per_block = threads_per_block_max;
       threads_per_block >= threads_per_block_min;
       threads_per_block /= 2) {
    iop_candidates.emplace_back(
        get_heuristics_given_vect_threads(vect_factor, threads_per_block));
  }

  // Sort the heuristics based on the register usage and occupancy.
  std::stable_sort(
      iop_candidates.begin(),
      iop_candidates.end(),
      [](const InnerOuterParams& a, const InnerOuterParams& b) {
        // If a thread can use more registers than required, there is a high
        // chance that it can avoid register spilling and compiler can optimize
        // for better instruction level parallelism.
        int64_t extra_regs_a =
            a.available_register_per_thread - a.required_register_per_thread;
        int64_t extra_regs_b =
            b.available_register_per_thread - b.required_register_per_thread;
        if (extra_regs_a > 0 && extra_regs_b < 0) {
          return true;
        } else if (extra_regs_a < 0 && extra_regs_b > 0) {
          return false;
        }
        // High occupancy provides better threads level parallelism.
        // 25% is sufficient since ILP is high due to persistent batch sizes
        // which is equivalent to unrolling inner dim.
        if (a.warps_per_sm != b.warps_per_sm &&
            (a.warps_per_sm < 16 || b.warps_per_sm < 16)) {
          return a.warps_per_sm > b.warps_per_sm;
        }
        // Tie breaker, smaller threads_per_block to reduce communication
        // overhead
        return a.threads_per_block < b.threads_per_block;
      });

  // Pick the best heuristic
  auto iop = iop_candidates.front();
  rparams->block_dim_inner_reduction_extra = ParallelType::TIDy;
  rparams->combined_split_grid_inner_dim =
      iop.vectorization_factor_outer * iop.bdimx * iop.gdimy < inner_dim_numel;
  rparams->static_bdimx = true;
  rparams->static_bdimy = true;
  iop.bdimz = ceilDiv(
      ceilDiv(ceilDiv(inner_dim_numel / iop.inner_vect, iop.bdimx), iop.bdimy),
      iop.inner_batch);
  NVF_ERROR(iop.bdimz == 1, "bdimz must be 1.");

  // check all the parameters in InnerOuterParams are set.
  iop.verify();

  rparams->persistent_kernel = true;
  rparams->fastest_dim = true;
  rparams->combined_inner_outer = true;
  // tmp_gmem is the intermediate result of outer reduction, its dtype is float,
  // so the maximum vectorization factor is 4.
  rparams->vectorization_factor_outer = iop.vectorization_factor_outer;
  rparams->vectorization_factor_tmp_gmem_write = iop.tmp_gmem_write_vect;
  rparams->cparams.maxrregcount = iop.available_register_per_thread;
  rparams->unroll_factor_inner_reduction = iop.inner_vect;
  rparams->batches_per_block_inner_reduction = iop.inner_batch;
  rparams->block_dim_inner_reduction = ParallelType::TIDx;
  rparams->vectorize_inner_reduction = iop.inner_vect > 1;
  rparams->split_grid_dim_iter_dom_outer = true;
  rparams->grid_dim_iter_dom = ParallelType::BIDy;

  rparams->lparams = LaunchParams(
      LaunchParams::UNINITIALIZED_VAL,
      iop.gdimy,
      LaunchParams::UNINITIALIZED_VAL,
      iop.bdimx,
      iop.bdimy,
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
            << "multiple_reds_per_blk: " << rparams->multiple_reds_per_blk
            << "\n"
            << "warps_per_sm: " << iop.warps_per_sm << "\n"
            << "gdimy: " << iop.gdimy << "\n"
            << "block(" << (iop.bdimx) << ", " << iop.bdimy << ", " << 1 << ")";
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
    outer_reduction_tv->split(0, rparams->lparams.gdimy());

    TensorView* partialResult = outer_reduction_tv->rFactor({0});
    partialResult->cacheBefore();
    partialResult->setMemoryType(MemoryType::Global);
    TensorView* partialResultReload = partialResult->cacheAfter();

    boundaryNodesSet.insert(partialResultReload);
    cached_gmem.emplace_back(partialResult);
    cached_gmem_reload.emplace_back(partialResultReload);

    // Second-stage of outer reduction
    // reduction domain, [I1/TIDy, TIDy]
    outer_reduction_tv->split(0, rparams->lparams.bdimy());
    outer_reduction_tv->axis(1)->parallelize(ParallelType::TIDy);
    // iteration domain, [BIDy, TIDx, Vect]
    int axisID = -1;
    if (rparams->vectorization_factor_outer > 1) {
      outer_reduction_tv->split(axisID, rparams->vectorization_factor_outer);
      outer_reduction_tv->axis(axisID--)->parallelize(ParallelType::Vectorize);
    }

    if (rparams->lparams.bdimx() > 1) {
      outer_reduction_tv->split(axisID, rparams->lparams.bdimx());
      outer_reduction_tv->axis(axisID--)->parallelize(ParallelType::TIDx);
    }

    if (rparams->combined_split_grid_inner_dim) {
      outer_reduction_tv->split(
          axisID, NamedScalar::getParallelDim(ParallelType::BIDy));
    }

    outer_reduction_tv->axis(axisID--)->parallelize(ParallelType::BIDy);

    auto outer_reference_tv =
        reduction_scheduler_utils::sortAndRFactor(outer_reduction_tv);
    outer_reference_tvs.emplace_back(outer_reference_tv);
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
  const bool is_outer_grid_persistence = rparams->persistent_kernel &&
      rparams->cross_grid_inner_reduction && !rparams->fastest_dim;

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
    // (3) Excluding tma tvs breaks the propagation path from inner reduction tv
    // to cached_gmem which stores the results of the first-stage of outer
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
      is_outer_grid_persistence,
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
        is_outer_grid_persistence,
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
  inlineMost();
}
} // namespace inner_outer_tma_warp_specialized
} // namespace nvfuser
