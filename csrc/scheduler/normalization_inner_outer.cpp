// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#include <ATen/cuda/CUDAContext.h>
#include <inlining.h>
#include <instrumentation.h>
#include <scheduler/debug_utils.h>
#include <scheduler/normalization_inner_outer.h>
#include <scheduler/normalization_utils.h>
#include <scheduler/reduction_utils.h>
#include <scheduler/utils.h>

namespace nvfuser {

constexpr auto schedule_heuristic = ScheduleHeuristic::InnerOuterPersistent;

std::shared_ptr<ReductionParams> getInnerOuterPersistentHeuristics(
    Fusion* fusion,
    const at::ArrayRef<c10::IValue>& runtime_inputs,
    HeuristicSummary* data_cache) {
  FUSER_PERF_SCOPE("getInnerOuterPersistentHeuristicsFromIValue");
  SchedulerRuntimeInfo runtime_info(fusion, runtime_inputs);
  auto reduction_type = reduction_scheduler_utils::getReductionType(fusion);
  NVF_CHECK(
      reduction_type == reduction_scheduler_utils::ReductionType::InnerOuter,
      "Reduction type should be ReductionType::InnerOuter.");
  return InnerOuterPersistentKernelScheduler::getPersistentHeuristic(
      fusion, runtime_info, data_cache);
}

InnerOuterPersistentKernelScheduler::InnerOuterPersistentKernelScheduler(
    Fusion* fusion,
    SchedulerRuntimeInfo& runtime_info,
    HeuristicSummary* data_cache)
    : SchedulerEntry(schedule_heuristic) {
  computeHeuristics(fusion, runtime_info, data_cache);
}

void InnerOuterPersistentKernelScheduler::computeHeuristics(
    Fusion* fusion,
    SchedulerRuntimeInfo& runtime_info,
    HeuristicSummary* data_cache) {
  params_ = getPersistentHeuristic(fusion, runtime_info, data_cache);
  NVF_ERROR(params_ != nullptr);
}

void InnerOuterPersistentKernelScheduler::schedule(Fusion* fusion) {
  FUSER_PERF_SCOPE("Schedule InnerOuterPersistent Fusion");
  schedulePersistentKernel(fusion, reductionParams());
}

bool InnerOuterPersistentKernelScheduler::canScheduleCompileTime(
    Fusion* fusion) {
  auto heuristic = schedule_heuristic;

  // (1) leading common checks for all persistent kernels.
  if (!normalization_scheduler_utils::checkOpsAndInputs(fusion, heuristic)) {
    return false;
  }

  // (2) check reduction type.
  const auto& reduction_tvs = scheduler_utils::getReductionTvs(fusion);
  if (!normalization_scheduler_utils::checkReductionType(
          reduction_tvs, heuristic)) {
    return false;
  }

  // (3) special check for InnerOuter persistent kernel.
  std::vector<TensorView*> inner_reduction_tvs;
  std::vector<TensorView*> outer_reduction_tvs;
  for (auto tv : reduction_tvs) {
    if (scheduler_utils::isFastestDimReduction(tv)) {
      inner_reduction_tvs.emplace_back(tv);
    } else {
      outer_reduction_tvs.emplace_back(tv);
    }
  }
  normalization_scheduler_utils::checkReductionAxis(
      fusion, inner_reduction_tvs, heuristic);
  normalization_scheduler_utils::checkReductionAxis(
      fusion, outer_reduction_tvs, heuristic);
  if (!normalization_scheduler_utils::checkIfReductionsAreInnerOuter(
          inner_reduction_tvs, outer_reduction_tvs)) {
    scheduler_debug_utils::canScheduleRejectReason(
        heuristic,
        "to use combined reduction, inner reduction tensor should be [I,I,...,R,R] and outer reduction tensor should be [R,R,...,I,I]");
    return false;
  }

  if (!normalization_scheduler_utils::hasSharedInput(
          inner_reduction_tvs, outer_reduction_tvs)) {
    scheduler_debug_utils::canScheduleRejectReason(
        heuristic,
        "to use combined reduction, inner reduction and outer reduction should have shared input.");
    return false;
  }

  if (!normalization_scheduler_utils::isConnectedOnlyThroughReductionProducer(
          inner_reduction_tvs, outer_reduction_tvs)) {
    scheduler_debug_utils::canScheduleRejectReason(
        heuristic,
        "to use combined reduction, inner reduction and outer reduction should not have shared consumer, their consumers should not have shared non-outer-reduction producer.");
    return false;
  }

  // (4) tailing common checks for all persistent kernels.
  if (!normalization_scheduler_utils::checkViewRootPersistentTopology(
          fusion, reduction_tvs, inner_reduction_tvs[0], heuristic)) {
    return false;
  }

  return true;
}

bool InnerOuterPersistentKernelScheduler::canScheduleRunTime(
    Fusion* fusion,
    SchedulerRuntimeInfo& runtime_info,
    HeuristicSummary* data_cache) {
  FUSER_PERF_SCOPE("InnerOuterPersistentKernelScheduler::canSchedule");

  // (1) check if there is enough shared memory or registers for persistent.
  auto reduction_tv_entry =
      HeuristicSummaryEntry<HeuristicCompileTime::ReductionTVs>(
          data_cache, [&fusion]() {
            return std::make_unique<std::vector<TensorView*>>(
                scheduler_utils::getReductionTvs(fusion));
          });
  auto& reduction_tvs = reduction_tv_entry.get();

  auto persistent_buffer_info_entry =
      HeuristicSummaryEntry<HeuristicCompileTime::PersistentBufferInfo>(
          data_cache, [&fusion]() {
            return std::make_unique<scheduler_utils::PersistentBufferInfo>(
                scheduler_utils::persistentBuffers(fusion));
          });
  auto& persistent_buffer_info = persistent_buffer_info_entry.get();

  const int64_t persistent_buffer_size =
      normalization_scheduler_utils::getPersistentBufferSize(
          fusion, runtime_info, data_cache, persistent_buffer_info) +
      normalization_scheduler_utils::partialReductionBufferSize(
          reduction_tvs, runtime_info);

  const int64_t available_shared_memory_size =
      normalization_scheduler_utils::getAvailableSmemSize(
          runtime_info, persistent_buffer_info.persistent_buffers);

  const int64_t available_persistent_buffer_size = std::max(
      scheduler_utils::register_file_size_full, available_shared_memory_size);

  if (persistent_buffer_size > available_persistent_buffer_size) {
    scheduler_debug_utils::canScheduleRejectReason(
        schedule_heuristic,
        "not enough registers or shared memory for persistence");
    return false;
  }

  // (2) check if we can schedule the combined reductions with a reasonable
  // batch size without register spills.
  const auto device_prop = at::cuda::getCurrentDeviceProperties();
  TensorView* reference_tv = nullptr;
  for (auto tv : reduction_tvs) {
    if (scheduler_utils::isFastestDimReduction(tv)) {
      reference_tv = tv;
      break;
    }
  }
  NVF_ERROR(
      reference_tv,
      "reference_tv is nullptr in PersistentKernelScheduler::canScheduleRunTimeInnerOuter");
  auto properties = scheduler_utils::getReductionProperties(
      fusion, runtime_info, reference_tv);
  auto reduced_tv = ir_utils::getSoleProducerTv(reference_tv);
  const auto vectorize_factor = vectorize_helper::getVectorizationFactor(
      runtime_info,
      reduced_tv,
      data_cache,
      (int)(reduced_tv->nDims() - properties.inner_most_dimension_ndims));
  if (!normalization_scheduler_utils::
           getOptionalInnerOuterPersistentBufferBatches(
               properties.total_reduction_numel,
               properties.total_iteration_numel,
               persistent_buffer_size,
               (int64_t)vectorize_factor,
               (int64_t)device_prop->warpSize,
               false)
               .first.has_value()) {
    scheduler_debug_utils::canScheduleRejectReason(
        schedule_heuristic,
        "Required batch number is larger than available batch number! Will cause register spills!");
    return false;
  }

  // (3) check iteration size
  // TODO: Needs check whether we need this check for innerOuter scheduler or
  // not.
  if (!normalization_scheduler_utils::runTimeCheckIterSize(
          properties, schedule_heuristic)) {
    return false;
  }

  return true;
}

namespace {

// The innerOuterPersistentHeuristic is tuned for layer_norm backward on A100
// ======= Method if hidden_size > 1024 =======
// (1) Inner reduction is one reduction per block. Reduction domain is
// parallelized by TIDx and TIDy, Iteration domain is parallelized by BIDy. (2)
// Outer reduction is done in two-steps. The first step is partial reduction,
// reduction domain is parallelized by BIDy, iteration domain is parallelized by
// TIDx and TIDy. The partial results are written to gmem followed by a grid
// sync. The second step is block reduction, the reduction domain is
// parallelized by TIDy, the iteration domain is parallelized by TIDx and BIDy.
// ======= Method if hidden_size <= 1024 =======
// (1) Inner reduction is multi-reductions per blocks. Reduction domain is
// parallelized by TIDx, Iteration domain is parallelized by BIDy and TIDy
// (2) Outer reduction is same to cases where hidden_size > 1024 except the
// second step where in this case, the reduction domain is parallelized by TIDx
// and the iteration domain is parallelized by TIDy and BIDy. This switch
// between TIDx and TIDy is because (a) We can do warp reduction with TIDx and
// (b) TIDx*BIDy is usually much larger than hidden_size, e.g. 128*216 = 1024*27
// this means without switch only 1/27 of the threads is used.
std::shared_ptr<ReductionParams> innerOuterPersistentHeuristic(
    const int64_t outer_dim_numel,
    const int64_t inner_dim_numel,
    const int64_t max_persistent_buffer_size,
    const size_t tmp_gmem_dtype_size,
    const size_t vectorize_factor) {
  auto rparams = std::make_shared<ReductionParams>();
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
    int64_t gdimy = -1;
    int64_t tmp_gmem_write_vect = -1;
    int64_t vectorization_factor_outer = -1;

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
  };

  InnerOuterParams iop;

  // Estimate register per thread based on buffer size, since inner reduction
  // dim is fully parallelized, the buffer size of each thread equals the total
  // buffer size divide by inner_dim_numel.
  auto getEstimatedRegisterUsage = [&](int64_t batch_mul_vect) {
    constexpr int64_t bytes_per_register = 4;
    const int64_t persistent_buffer_size =
        max_persistent_buffer_size / inner_dim_numel * batch_mul_vect;
    const int64_t estimated_register_count =
        persistent_buffer_size / bytes_per_register +
        scheduler_utils::register_overhead;
    return std::min(
        estimated_register_count, scheduler_utils::max_registers_per_thread);
  };

  auto getBlocksPerSM = [&](const int64_t threads_per_sm,
                            const int64_t threads_per_block,
                            const int64_t warp_size) {
    constexpr int64_t warp_allocation_granularity = 4;
    const int64_t allocated_warps_per_block =
        ceilDiv(
            ceilDiv(threads_per_block, warp_size),
            warp_allocation_granularity) *
        warp_allocation_granularity;
    return scheduler_utils::safeDiv(
        threads_per_sm / warp_size, allocated_warps_per_block);
  };

  const auto dev_prop = at::cuda::getCurrentDeviceProperties();
  const int64_t device_multiprocessor_count =
      (int64_t)dev_prop->multiProcessorCount;

  // Step-1, set InnerParams reduction dim: inner_vect, inner_batch,
  // threads_per_block (bdimx * bdimy). Start threads_per_block from a quarter
  // warp, gradually increase it. Runtime checkCombinedReductionShape ensures
  // inner_dim_numel is dividable by the multiplication of a quarter warp and
  // vectorize_factor.
  iop.inner_vect = (int64_t)vectorize_factor;

  // ignore_register_size_limit will return a valid batch size.
  // This is needed because we enforced projection for fp32 if the feature size
  // is less or equal 14K. It leads to register spills but still faster than the
  // unprojected version due to the reuse of a input para in this grid
  // persistent kernel. However, when we do register usage check in
  // canScheduleRuntime, the enforced projection is not considered. Thus,
  // max_persistent_buffer_size used here is larger than the value used in
  // canScheduleRuntime.
  // This is a tmp solution before we have a new persistent heuristics, where
  // the projection is not solely based on size of buffers. The enforced buffer
  // projection is not considered in canScheduleRuntime Thus,
  constexpr bool ignore_register_size_limit = true;
  const auto& batch_and_block_size = normalization_scheduler_utils::
      getOptionalInnerOuterPersistentBufferBatches(
          inner_dim_numel,
          outer_dim_numel,
          max_persistent_buffer_size,
          iop.inner_vect,
          dev_prop->warpSize,
          ignore_register_size_limit);
  auto opt_inner_batch = batch_and_block_size.first;
  NVF_ERROR(opt_inner_batch.has_value());
  iop.inner_batch = opt_inner_batch.value();
  int64_t threads_per_block = batch_and_block_size.second;

  NVF_ERROR(
      iop.inner_vect * iop.inner_batch * threads_per_block >= inner_dim_numel,
      " iop.inner_vect * iop.inner_batch * threads_per_block should >= inner_dim_numel.");

  // Step-2, set InnerParams Iteration dim: gdimy. reg_per_thread is estimated
  // from buffer size, then it is used to calculate threads_per_sm and gdimy.
  // gdimy_max ensures each block processes at least 8 rows to
  // reduce the workload of the final outer reduction.
  int64_t reg_per_thread =
      getEstimatedRegisterUsage(iop.inner_vect * iop.inner_batch);
  int64_t threads_per_sm = getThreadsPerSMGivenRegPerThread(reg_per_thread);
  int64_t blocks_per_sm =
      getBlocksPerSM(threads_per_sm, threads_per_block, dev_prop->warpSize);
  iop.gdimy = blocks_per_sm * device_multiprocessor_count;
  const int64_t outer_iter_min = 8;
  const int64_t gdimy_max = scheduler_utils::roundUpToN(
      ceilDiv(outer_dim_numel, outer_iter_min), device_multiprocessor_count);
  while (iop.gdimy > gdimy_max && blocks_per_sm > 1) {
    blocks_per_sm -= 1;
    iop.gdimy = blocks_per_sm * device_multiprocessor_count;
  }

  // set the vectorization factor for the write to tmp gmem, may be different
  // from inner_vect due to different data types, e.g. input is half and
  // tmp_gmem is float
  constexpr int64_t max_gmem_vect_access_bytes = 16;
  const int64_t max_tmp_gmem_vect_factor = std::min(
      max_gmem_vect_access_bytes / (int64_t)tmp_gmem_dtype_size,
      iop.inner_vect);
  iop.tmp_gmem_write_vect = max_tmp_gmem_vect_factor;

  // Step-3, set OuterParams Iteration dim: vectorization_factor_outer, bdimx,
  // gdimy (already done) The partial outer reduction result is stored in tmp
  // gmem, set the vectorization factor for write and read
  const int64_t workload_per_thread = inner_dim_numel >= 4096 ? 4l : 2l;
  iop.vectorization_factor_outer =
      std::min(workload_per_thread, max_tmp_gmem_vect_factor);
  // For widely used hidden sizes, threads_per_block has factor of 8, roundup to
  // increase the probability of bdimx * bdimy == threads_per_block.
  iop.bdimx = scheduler_utils::roundUpPow2Or8(
      ceilDiv(inner_dim_numel / iop.vectorization_factor_outer, iop.gdimy));
  // if still not divisible, e.g. threads_per_block = 256, bdimx = 40.
  // increase bdimx to make it divisible. Under worst case, bdimx equals to
  // threads_per_block.
  while (threads_per_block % iop.bdimx) {
    iop.bdimx = std::min(iop.bdimx + 8, threads_per_block);
  }
  // Step-4, set OuterParams Reduction dim: bdimy.
  iop.bdimy = threads_per_block / iop.bdimx;
  NVF_ERROR(
      iop.bdimy * iop.bdimx == threads_per_block,
      " threads_per_block must be divisible by bdimx and bdimy.");
  // Step-5, special case, when inner_dim_numel <= 1024, bdimx is usually small
  // after divide by inner_vect and inner_batch. In this case, bdimy is used to
  // parallelize outer_dim instead of inner_dim. This pattern is named multi
  // reductions per block (mrpb).
  if (inner_dim_numel <= 1024) {
    rparams->multiple_reds_per_blk = true;
    rparams->tidx_for_outer_reduction = true;
    constexpr int64_t threads_per_block_mrpb = 512;

    // Step-1, InnerParams, Reduction dim: inner_vect(reuse),
    // inner_batch(reuse), bdimx
    iop.bdimx = ceilDiv(inner_dim_numel, iop.inner_vect * iop.inner_batch);

    // Step-2, InnerParams, Iteration dim: gdimy, bdimy (in next step)
    reg_per_thread =
        getEstimatedRegisterUsage(iop.inner_vect * iop.inner_batch);
    threads_per_sm = getThreadsPerSMGivenRegPerThread(reg_per_thread);
    blocks_per_sm = getBlocksPerSM(
        threads_per_sm, threads_per_block_mrpb, dev_prop->warpSize);
    iop.gdimy = blocks_per_sm * device_multiprocessor_count;

    // Step-3, OuterParams, Iteration dim: vectorization_factor_outer(reuse),
    // bdimy, gdimy (in previous step). We prefer bdimy to be larger enough to
    // cover what is left in both the outer_dim and inner_dim. However, it
    // should not exceed the limitation set by threads_per_block_mrpb.
    int64_t bdimy_tmp = std::max(
        ceilDiv(outer_dim_numel, iop.gdimy),
        ceilDiv(inner_dim_numel, iop.vectorization_factor_outer * iop.gdimy));
    iop.bdimy = std::min(threads_per_block_mrpb / iop.bdimx, bdimy_tmp);

    // Step-4, OuterParams, Reduction dim: bdimx (already done)

    if (iop.bdimx % dev_prop->warpSize == 0) {
      rparams->pad_inner_reduction_to_warp = true;
      rparams->pad_outer_reduction_to_warp = true;
    }
    rparams->block_dim_iter_dom = ParallelType::TIDy;
  } else {
    rparams->block_dim_inner_reduction_extra = ParallelType::TIDy;
  }

  // check all the parameters in InnerOuterParams are set.
  iop.verify();

  rparams->persistent_kernel = true;
  rparams->fastest_dim = true;
  rparams->combined_inner_outer = true;
  // tmp_gmem is the intermediate result of outer reduction, its dtype is float,
  // so the maximum vectorization factor is 4.
  rparams->vectorization_factor_outer = iop.vectorization_factor_outer;
  rparams->vectorization_factor_tmp_gmem_write = iop.tmp_gmem_write_vect;
  rparams->cparams.maxrregcount = (int)getRegPerThreadGivenThreadsPerSM(
      iop.bdimx * iop.bdimy * blocks_per_sm);
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

  rparams->tag = "InnerOuter Persistent Heuristic.\n";

  if (isDebugDumpEnabled(DebugDumpOption::SchedulerDebug)) {
    debug() << "\n===== Combined InnerOuter Reduction Stats ========\n"
            << "outer_dim_numel: " << outer_dim_numel << "\n"
            << "inner_dim_numel: " << inner_dim_numel << "\n"
            << "max_persistent_buffer_size: " << max_persistent_buffer_size
            << "\n"
            << "vectorize_factor_input: " << iop.inner_vect << "\n"
            << "vectorization_factor_tmp_gmem_write: "
            << iop.tmp_gmem_write_vect << "\n"
            << "vectorization_factor_outer: " << iop.vectorization_factor_outer
            << "\n"
            << "multiple_reds_per_blk: " << rparams->multiple_reds_per_blk
            << "\n"
            << "threads_per_sm: " << threads_per_sm << "\n"
            << "gdimy: " << iop.gdimy << "\n"
            << "block(" << (iop.bdimx) << ", " << iop.bdimy << ", " << 1 << ")";
    debug() << rparams->toString() << std::endl;
  }
  return rparams;
}

int64_t getOutReductionDataTypeSize(
    const std::vector<TensorView*>& reduction_tvs) {
  for (auto tv : reduction_tvs) {
    if (!scheduler_utils::isFastestDimReduction(tv)) {
      return dataTypeSize(tv->getDataType().value());
    }
  }
  NVF_ERROR(
      false,
      "No outer reduction tv detected in InnerOuterPersistentScheduler.");
  return -1;
}

int64_t tryEnforceBufferProjection(
    SchedulerRuntimeInfo& runtime_info,
    HeuristicSummary* data_cache,
    const std::vector<TensorView*>& reduction_tvs,
    const bool can_project,
    bool& project_persistent_buffers,
    scheduler_utils::PersistentBufferSizeReturn persistent_buffer_size_info) {
  int64_t outer_reduction_buffer_size =
      normalization_scheduler_utils::partialReductionBufferSize(
          reduction_tvs, runtime_info);

  // for layer_norm backward, enable project to input can reuse weight shared
  // among different rows. Although it increased register usage and may lead
  // to register spills, the overall performance is increased. The following
  // code will check if we can do this projection by allowing more registers.
  // This is a temporary solution, the issue is tracked by
  // https://github.com/csarofeen/pytorch/issues/2525
  if (can_project && !project_persistent_buffers) {
    int64_t total_projected_buffer_size =
        persistent_buffer_size_info.projected_persistent_buffer_size +
        outer_reduction_buffer_size;
    // allow 10% more to allow project to input, 14K float should do project
    // and 16K float should't do. more_register_factor >= 14*1024*5(three
    // inputs, two outer reduction results)*sizeof(float) /
    // register_file_size_full
    constexpr float more_register_factor = 1.1;
    const int64_t avilable_register_file_size = static_cast<int64_t>(
        scheduler_utils::register_file_size_full * more_register_factor);
    if (avilable_register_file_size >= total_projected_buffer_size) {
      project_persistent_buffers = true;
    }
  }
  // now we have the final decision on whether we project to input or not.
  if (project_persistent_buffers) {
    return persistent_buffer_size_info.projected_persistent_buffer_size +
        outer_reduction_buffer_size;
  } else {
    return persistent_buffer_size_info.persistent_buffer_size +
        outer_reduction_buffer_size;
  }
}
} // namespace

std::shared_ptr<ReductionParams> InnerOuterPersistentKernelScheduler::
    getPersistentHeuristic(
        Fusion* fusion,
        SchedulerRuntimeInfo& runtime_info,
        HeuristicSummary* data_cache) {
  FUSER_PERF_SCOPE("getInnerOuterPersistentHeuristics");

  auto reduction_tv_entry =
      HeuristicSummaryEntry<HeuristicCompileTime::ReductionTVs>(
          data_cache, [&fusion]() {
            return std::make_unique<std::vector<TensorView*>>(
                scheduler_utils::getReductionTvs(fusion));
          });
  auto& reduction_tvs = reduction_tv_entry.get();

  NVF_ERROR(!reduction_tvs.empty(), "Need reduction tensor views to schedule.");
  TensorView* first_inner_reduction_tv = nullptr;
  for (auto tv : reduction_tvs) {
    if (scheduler_utils::isFastestDimReduction(tv)) {
      first_inner_reduction_tv = tv;
      break;
    }
  }
  auto ref_red_tv = first_inner_reduction_tv;

  // (1) reduction properties and vectorization factor
  auto [reduced_tv, properties, vectorize_factor] =
      normalization_scheduler_utils::getReductionPropertiesVectFactor(
          fusion, runtime_info, data_cache, reduction_tvs, ref_red_tv);

  // (2) info about persistent buffer.
  auto [can_project, persistent_buffer_size_info] =
      normalization_scheduler_utils::getBufferSizeInfo(
          fusion, runtime_info, data_cache);
  bool project_persistent_buffers = can_project &&
      persistent_buffer_size_info.projected_persistent_buffer_size <
          persistent_buffer_size_info.persistent_buffer_size;
  // add additional buffers for partial results of outer reductions.
  // reconsider whether project persistent buffers to inputs or not.
  auto max_persistent_buffer_size = tryEnforceBufferProjection(
      runtime_info,
      data_cache,
      reduction_tvs,
      can_project,
      project_persistent_buffers,
      persistent_buffer_size_info);

  // (3) dtype used to store partial outer reduction in combined reduction
  const int64_t tmp_gmem_dtype_size =
      getOutReductionDataTypeSize(reduction_tvs);
  std::shared_ptr<ReductionParams> rparams = innerOuterPersistentHeuristic(
      properties.total_iteration_numel,
      properties.total_reduction_numel,
      max_persistent_buffer_size,
      tmp_gmem_dtype_size,
      vectorize_factor);
  rparams->project_persistent_buffers = project_persistent_buffers;
  rparams->cparams.index_type = runtime_info.getIndexType();
  return rparams;
}

namespace {

void scheduleReductionCombinedOuter(
    Fusion* fusion,
    const ReductionParams& rparams,
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
    // merge tensorview to [reduction, iteraiton] domains
    mergeReductionOrIterDomains(outer_reduction_tv, true);
    mergeReductionOrIterDomains(outer_reduction_tv, false);
    if (rparams.multiple_reds_per_blk) {
      outer_reduction_tv->split(
          0, NamedScalar::getParallelDim(rparams.block_dim_iter_dom));
    }
    outer_reduction_tv->split(
        0, NamedScalar::getParallelDim(rparams.grid_dim_iter_dom), false);

    if (rparams.multiple_reds_per_blk) {
      outer_reduction_tv->rFactor({1});
    }
    TensorView* partialResult = outer_reduction_tv->rFactor({1});
    partialResult->cacheBefore();
    partialResult->setMemoryType(MemoryType::Global);
    TensorView* partialResultReload = partialResult->cacheAfter();

    boundaryNodesSet.insert(partialResultReload);
    cached_gmem.emplace_back(partialResult);
    cached_gmem_reload.emplace_back(partialResultReload);

    if (rparams.multiple_reds_per_blk) {
      if (rparams.tidx_for_outer_reduction) {
        outer_reduction_tv->split(
            0, NamedScalar::getParallelDim(ParallelType::TIDx));
        outer_reduction_tv->axis(1)->parallelize(ParallelType::TIDx);
        // to use warp reduction
        if (rparams.pad_outer_reduction_to_warp) {
          outer_reduction_tv->axis(1)->padToMultipleOfWarp();
        }
      } else {
        outer_reduction_tv->split(
            0, NamedScalar::getParallelDim(ParallelType::TIDy));
        outer_reduction_tv->axis(1)->parallelize(ParallelType::TIDy);
      }
      // iteration domain
      int axisID = -1;
      if (rparams.vectorization_factor_outer > 1) {
        outer_reduction_tv->split(axisID, rparams.vectorization_factor_outer);
        outer_reduction_tv->axis(axisID--)->parallelize(
            ParallelType::Vectorize);
      }

      if (rparams.tidx_for_outer_reduction) {
        outer_reduction_tv->split(
            axisID, NamedScalar::getParallelDim(ParallelType::TIDy));
        outer_reduction_tv->axis(axisID--)->parallelize(ParallelType::TIDy);
      } else {
        outer_reduction_tv->split(
            axisID, NamedScalar::getParallelDim(ParallelType::TIDx));
        outer_reduction_tv->axis(axisID--)->parallelize(ParallelType::TIDx);
      }

      outer_reduction_tv->split(
          axisID, NamedScalar::getParallelDim(ParallelType::BIDy));
      outer_reduction_tv->axis(axisID--)->parallelize(ParallelType::BIDy);

    } else {
      // reduction domain
      outer_reduction_tv->split(
          0, NamedScalar::getParallelDim(ParallelType::TIDy));
      outer_reduction_tv->axis(1)->parallelize(ParallelType::TIDy);

      // iteration domain
      int axisID = -1;
      if (rparams.vectorization_factor_outer > 1) {
        outer_reduction_tv->split(axisID, rparams.vectorization_factor_outer);
        outer_reduction_tv->axis(axisID--)->parallelize(
            ParallelType::Vectorize);
      }

      if (rparams.lparams.bdimx() > 1) {
        outer_reduction_tv->split(
            axisID, NamedScalar::getParallelDim(ParallelType::TIDx));
        outer_reduction_tv->axis(axisID--)->parallelize(ParallelType::TIDx);
      }

      outer_reduction_tv->split(
          axisID, NamedScalar::getParallelDim(ParallelType::BIDy));

      outer_reduction_tv->axis(axisID--)->parallelize(ParallelType::BIDy);
    }
    auto outer_reference_tv =
        reduction_scheduler_utils::sortAndRFactor(outer_reduction_tv);
    outer_reference_tvs.emplace_back(outer_reference_tv);
  }
}

} // namespace

void InnerOuterPersistentKernelScheduler::schedulePersistentKernel(
    Fusion* fusion,
    const ReductionParams& rparams) {
  FUSER_PERF_SCOPE("schedulePersistentKernelInnerOuter");

  FusionGuard fg(fusion);

  // Grab the reduction, input, and output tensor views. dummy_outputs are
  // helper tensors for persistent buffer projection.
  std::vector<TensorView*> dummy_outputs, cached_inputs, reduction_tvs;
  std::vector<std::pair<TensorView*, TensorView*>> cached_outputs;
  normalization_scheduler_utils::beforeSchedule(
      fusion,
      rparams,
      dummy_outputs,
      cached_inputs,
      reduction_tvs,
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

  // schedule inner reduction, only schedule the first inner reduction tv, then
  // will be propagated to other inner reduction tvs.
  TensorView* inner_reference_tv =
      normalization_scheduler_utils::scheduleReductionGeneral(
          fusion, rparams, inner_reduction_tvs);

  // schedule outer reduction, schedule all the outer reduction tvs since we
  // need to store the intermediate results.
  std::vector<TensorView*> cached_gmem;
  std::vector<TensorView*> cached_gmem_reload;
  std::vector<TensorView*> outer_reference_tvs;
  std::unordered_set<TensorView*> boundaryNodesSet;
  scheduleReductionCombinedOuter(
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

  const bool unroll = rparams.isUnrolled();
  const bool vectorize =
      rparams.vectorize_inner_reduction || rparams.vectorize_iter_dom;
  const bool is_outer_grid_persistence = rparams.persistent_kernel &&
      rparams.cross_grid_inner_reduction && !rparams.fastest_dim;

  // Propagate inner reduction. There is a cutoff at boundaryNodesSet, so this
  // propagation will not propagate to the final outer reduction.
  reduction_scheduler_utils::propagateTransformation(
      inner_reference_tv, boundaryNodesSet);
  reduction_scheduler_utils::propagateRFactor(
      inner_reference_tv, inner_reduction_tvs[0], inner_reduction_tvs);

  // Don't allow parallelization propagation goes through boundaryNodesSet
  const auto& selected_tvs_inner =
      scheduler_utils::getAllTvsFrom(inner_reduction_tvs, boundaryNodesSet);
  reduction_scheduler_utils::propagateParallelization(
      fusion,
      inner_reduction_tvs[0],
      inner_reference_tv,
      unroll,
      vectorize,
      is_outer_grid_persistence,
      inner_reduction_tvs,
      cached_inputs,
      cached_outputs,
      {selected_tvs_inner.begin(), selected_tvs_inner.end()});

  // Propagate outer reduction. Each outer reduction is connected with its
  // cached_gmem and output, since we added all the cached_gmem to the
  // boundaryNodesSet, the transformation from one outer reduction can't
  // propagate to other outer reductions due to the cutoff at boundaryNodesSet.
  // Thus, we need a loop to initiate the propagation from each outer reduction.
  // Don't allow parallelization propagation goes through cached_gmem, see issue
  // 246.
  for (long unsigned int i = 0; i < outer_reference_tvs.size(); i++) {
    const auto& selected_tvs_outer = scheduler_utils::getAllTvsFrom(
        {outer_reduction_tvs[i]}, {cached_gmem[i]});
    reduction_scheduler_utils::propagateTransformation(
        outer_reference_tvs[i], boundaryNodesSet);
    reduction_scheduler_utils::propagateParallelization(
        fusion,
        outer_reduction_tvs[i],
        outer_reference_tvs[i],
        unroll,
        vectorize,
        is_outer_grid_persistence,
        outer_reduction_tvs,
        cached_inputs,
        cached_outputs,
        {selected_tvs_outer.begin(), selected_tvs_outer.end()});
  }

  // special vectorization of temp gmem, vectorization_factor_tmp_gmem_write is
  // guaranteed to be smaller or equal to input vectorization factor.
  if (rparams.vectorization_factor_tmp_gmem_write > 1) {
    for (auto tv : cached_gmem) {
      NVF_ERROR(
          rparams.vectorization_factor_tmp_gmem_write <=
              rparams.unroll_factor_inner_reduction,
          "vectorization factor of temp gmem write should be smaller than that of inner reduction.")
      if (rparams.vectorization_factor_tmp_gmem_write <
          rparams.unroll_factor_inner_reduction) {
        tv->split(-1, rparams.vectorization_factor_tmp_gmem_write);
      }
      tv->axis(-1)->parallelize(ParallelType::Vectorize);
    }
  }
  // vectorization propagate through propagateParallelization only works for
  // input and output tensors. propagate vectorization to cached_gmem_reload
  // directly from output tv using parallelizeAllLike. must propagate seperaely
  // for different tvs as outer reductions are transformed seperately.
  if (rparams.vectorization_factor_outer > 1) {
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

  // Remove dummy outputs as they can inadvertently affect CA positions
  for (auto output : dummy_outputs) {
    fusion->removeOutput(output);
  }
  inlineMost();
}

} // namespace nvfuser
