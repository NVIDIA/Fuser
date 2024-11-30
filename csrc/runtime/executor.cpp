// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on

#include <runtime/executor.h>

#include <codegen.h>
#include <debug.h>
#include <device_lower/analysis/bank_conflict.h>
#include <driver_api.h>
#include <fusion_profiler.h>
#include <global_allocator.h>
#include <instrumentation.h>
#include <ir/all_nodes.h>
#include <ir/utils.h>
#include <iter_visitor.h>
#include <kernel_ir.h>
#include <options.h>
#include <polymorphic_value.h>
#include <runtime/allocations.h>
#include <runtime/executor_kernel_arg.h>
#include <runtime/executor_utils.h>
#include <serde/utils.h>
#include <tensor_metadata.h>
#include <utils.h>

#include <ATen/core/LegacyTypeDispatch.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/llvm_jit_strings.h>
#include <ATen/native/cuda/jit_utils.h>
#include <c10/core/DeviceGuard.h>
#include <c10/cuda/CUDAFunctions.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/util/irange.h>

#include <cmath>
#include <cstring>
#include <fstream>

namespace nvfuser {

std::unique_ptr<PrecomputedValues>& KernelExecutor::
    evaluatorPrecomputedValues() {
  if (!evaluator_precomputed_values_) {
    evaluator_precomputed_values_ =
        std::make_unique<PrecomputedValues>(compiledKernel()->kernel());
  }
  return evaluator_precomputed_values_;
}

bool ExprEvalExecutor::supported(Fusion* fusion) {
  FUSER_PERF_SCOPE("ExprEvalExecutor::supported");
  return std::all_of(
      fusion->outputs().begin(), fusion->outputs().end(), [&fusion](Val* out) {
        return fusion->getOutputAlias(out).type == AllocationType::Evaluate;
      });
}

void ExprEvalExecutor::compile(Fusion* fusion) {
  FUSER_PERF_SCOPE("ExprEvalExecutor::compile");
  if (isProfilerEnabled()) {
    FusionProfiler::segment(group_id_).startCompile();
  }
  NVF_ERROR(
      supported(fusion),
      "ExprEvalExecutor does not support the Fusion provided.");
  fusion_ = std::make_unique<Fusion>(*fusion);
  if (isProfilerEnabled()) {
    FusionProfiler::segment(group_id_).stopCompile();
  }
}

bool ExprEvalExecutor::isCompiled() const {
  return fusion_ != nullptr;
}

std::vector<at::Tensor> ExprEvalExecutor::run(
    KernelArgumentHolder& args,
    std::vector<at::Tensor> outputs) {
  FUSER_PERF_SCOPE("ExprEvalExecutor::run");

  if (isProfilerEnabled()) {
    NVF_CHECK(
        group_id_ >= 0,
        "An invalid segment id is passed to FusionProfiler!:",
        group_id_);
    SegmentProfiler& sprof = FusionProfiler::segment(group_id_);
    sprof.inputBytesAccessed(computeBytes(args));
    sprof.scheduler(toString(SchedulerType::ExprEval));
    sprof.startKernel();
  }

  NVF_ERROR(fusion_, "Need to compile before you can run.");
  // Bind fusion inputs
  auto expr_eval = executor_utils::bindInputs(args, fusion_.get());
  {
    NVF_ERROR(
        outputs.empty(),
        "Fusion executor is using expression evaluator,",
        " and expects that the outputs are not populated, which they were.");
    if (outputs.empty()) {
      for (const auto& out_val : fusion_->outputs()) {
        auto out_tensor =
            expr_eval.evaluate(out_val->as<TensorView>()).as<at::Tensor>();
        expr_eval.bind(out_val, out_tensor);
        outputs.emplace_back(out_tensor);
      }
    }
  }
  if (isProfilerEnabled()) {
    FusionProfiler::segment(group_id_).stopKernel();
    FusionProfiler::segment(group_id_).setDevice(args.getDeviceIndex());
  }
  return outputs;
}

namespace {
bool hasCpuScalarOutputs(Fusion* _fusion) {
  if (_fusion->exprs().empty()) {
    return false;
  }

  std::unordered_map<TensorView*, bool> tv_is_cpu_map;
  for (Expr* expr : StmtSort::getExprs(_fusion)) {
    bool has_cpu_scalar_input = false;
    bool has_cuda_input = false;
    for (Val* inp : expr->inputs()) {
      if (auto* inp_tv = dynamic_cast<TensorView*>(inp)) {
        if (inp_tv->isCpuScalar()) {
          has_cpu_scalar_input = true;
        } else {
          has_cuda_input = true;
          // Return early -- found atleast one CUDA input
          break;
        }
      }
    }
    if (!has_cuda_input && has_cpu_scalar_input) {
      // Expr is of the second category, and has all CPU scalar outputs
      for (Val* out : expr->outputs()) {
        if (auto* out_tv = dynamic_cast<TensorView*>(out)) {
          tv_is_cpu_map[out_tv] = true;
        }
      }
    }
  }

  bool has_any_cpu_output = std::any_of(
      _fusion->outputs().begin(),
      _fusion->outputs().end(),
      [&tv_is_cpu_map](Val* out) {
        return out->isA<TensorView>() && tv_is_cpu_map[out->as<TensorView>()];
      });
  return has_any_cpu_output;
}
} // namespace

bool KernelExecutor::supported(Fusion* _fusion) {
  FUSER_PERF_SCOPE("KernelExecutor::supported");
  return !hasCpuScalarOutputs(_fusion);
}

void KernelExecutor::compile(
    Fusion* _fusion,
    const KernelArgumentHolder& args,
    const LaunchParams& launch_constraints,
    CompileParams compile_params,
    SchedulerType scheduler_type) {
  FUSER_PERF_SCOPE("KernelExecutor::compile");
  fusion_ = std::make_unique<Fusion>(*_fusion);
  NVF_ERROR(
      supported(fusion_.get()),
      "KernelExecutor does not support the Fusion provided.");
  scheduler_type_ = scheduler_type;

  NVF_ERROR(
      !fusion_->outputs().empty(),
      "No output found for this kernel, aborting.");

  auto device = c10::Device(c10::DeviceType::CUDA, args.getDeviceIndex());

  if (isProfilerEnabled()) {
    NVF_CHECK(
        group_id_ >= 0,
        "An invalid segment id is passed to FusionProfiler!:",
        group_id_);
    FusionProfiler::segment(group_id_).setDevice(device.index());
    FusionProfiler::segment(group_id_).startCompile();
  }

  //! Force index_type to int and disable magic zero if we detect that the
  //! kernel contains any TMA memory operations.
  std::vector<Expr*> exprs = fusion_->exprs();
  bool has_cp_async_bulk = std::any_of(exprs.begin(), exprs.end(), [](Expr* e) {
    return ir_utils::isCpAsyncBulk(e);
  });

  // Disable magic zero if there are any TMA operations in Fusion
  if (has_cp_async_bulk) {
    compile_params.enable_magic_zero = false;
  }

  // Set the index type of compile params if not already set. If set,
  // make sure the compile param type is valid with the given kernel
  // arguments.
  auto arg_index_type = args.getSmallestIndexTypeOfArguments();
  if (compile_params.index_type.has_value()) {
    // If the int32 compilation is requested, but the arguments demand
    // int64, that's an error
    NVF_ERROR(
        !(compile_params.index_type.value() == PrimDataType::Int32 &&
          arg_index_type == PrimDataType::Int),
        "Compilation with int32 is requested but int64 is required for the arguments");
    NVF_ERROR(
        !has_cp_async_bulk ||
            (compile_params.index_type.value() == PrimDataType::Int32),
        "Compilation with int64 is requested but int32 is required because ",
        "of TMA operations.");

  } else if (arg_index_type == PrimDataType::Int) {
    // If the given compile option doesn't specify the index type, and
    // the arguments require 64-bit indexing, we need to use 64-bit
    // indexing. Note that if the arg type is 32-bit, it doesn't mean
    // it's safe to use 32-bit for the whole kernel, so unless it's
    // specified through CompileParams, we do not use 32-bit indexing.
    compile_params.index_type = arg_index_type;
    NVF_ERROR(
        !has_cp_async_bulk,
        "Compilation with int64 is required based on input arguments, but ",
        "int32 is required because of TMA operations.");
  } else if (has_cp_async_bulk) {
    // TMA operations require 32-bit indexing.
    compile_params.index_type = PrimDataType::Int32;
  }

  // TODO: Is this necessary?
  if (!compile_params.index_type.has_value()) {
    compile_params.index_type = arg_index_type;
  }

  c10::DeviceGuard dg(device);

  NVF_ERROR(device.is_cuda(), "Provided device to CUDA fuser is the CPU.");
  auto properties = at::cuda::getDeviceProperties(device.index());
  // TODO: These properties should be set as part of the constructor so that it
  // can be const
  device_smem_limit_ = static_cast<int64_t>(properties->sharedMemPerBlockOptin);
  warp_size_ = properties->warpSize;

  // Lowered is needed to compute launch parameters as it uses the CA map. We
  // could modify that, but simply generating that part first.
  compiled_kernel_ =
      std::make_unique<CompiledKernel>(fusion_.get(), compile_params);

  // TODO: pass block_size here;
  std::optional<int64_t> dynamic_smem = std::nullopt;
  std::optional<int64_t> block_size = std::nullopt;

  auto launch_params = launch_constraints;
  if (!args.empty()) {
    auto expr_eval = executor_utils::bindInputs(
        args, compiled_kernel_->lowered()->kernel()->as<Fusion>());
    NVF_ERROR(compile_params.index_type.has_value());
    launch_params = computeLaunchParams(
        launch_constraints,
        expr_eval,
        warp_size_,
        compile_params.index_type.value());
    block_size = launch_params.nThreads();
    dynamic_smem = launch_params.smem();
    NVF_ERROR(block_size > 0, "launch param inferred block size < 0");
  }

  // TODO: Fix, This hook doesn't do anything as we needed to lower compiled
  // kernel already
  for (const auto& hook : lowering_hooks_) {
    compiled_kernel_->registerLoweringHook(hook);
  }

  for (const auto& hook : post_lowering_hooks_) {
    compiled_kernel_->registerPostLoweringHook(hook);
  }

  // Now that we have launch parameters we can compile the kernel. It's a bit
  // odd we need launch parameters for compilation, need to go back and check
  // why this is the case.
  compiled_kernel_->compileFusion(
      device,
      launch_params.nThreads(),
      scheduler_type,
      fusion_id_,
      concrete_id_,
      runtime_id_,
      group_id_);

  // These should be nullopt at this point, but reset just in case
  resetCompiledKernelProperties();

  // If the dynamic shmem size is known, make sure the compiled kernel
  // has at least that size of dynamic shmem
  if (dynamic_smem.has_value()) {
    ensureAvailableDynamicSmemSize(dynamic_smem.value());
  }
}

LaunchParams KernelExecutor::computeLaunchParams(
    const LaunchParams& launch_constraints,
    ExpressionEvaluator& expr_eval,
    const int64_t warp_size,
    DataType index_type) {
  FUSER_PERF_SCOPE("KernelExecutor::computeLaunchParams");
  NVF_ERROR(warp_size > 0, "WARP_SIZE should be larger than 0");

  LaunchParams launch_params;

  auto data_cache = compileTimeDataCache();

  auto lower = compiled_kernel_->lowered().get();
  if (compiled_kernel_->getUsedTVs().empty()) {
    compiled_kernel_->setUsedTVs();
  }
  auto& used_tvs = compiled_kernel_->getUsedTVs();

  auto parallel_binding_ids_entry =
      executor_utils::caching::ExecutorCompileTimeEntry<
          executor_utils::caching::ParallelBindingIterDomains>(
          data_cache, [&used_tvs, &lower]() {
            return std::make_unique<std::vector<IterDomain*>>(
                executor_utils::getParallelBindingsIterDomains(
                    lower, used_tvs));
          });
  auto& parallel_binding_ids = parallel_binding_ids_entry.get();

  auto parallel_iter_extent_entry =
      executor_utils::caching::ExecutorCompileTimeEntry<
          executor_utils::caching::ParallelIterExtentMap>(
          data_cache, [&parallel_binding_ids]() {
            return executor_utils::getParallelIterExtents(parallel_binding_ids);
          });
  auto& parallel_iter_extents = parallel_iter_extent_entry.get();

  const auto& simplified_parallel_iter_extents =
      lower->parallelDimensionMap().getMap();

  // TODO: Need to redesign this part a bit to
  //   find the right place to trigger evaluate
  if (expr_eval.precomputedValues()) {
    expr_eval.precomputedValues()->bindParallelExtents(
        parallel_iter_extents, launch_constraints);
    expr_eval.precomputedValues()->evaluate();
  }

  // If any dimension was set in launch constraints we need to run through
  // IterDomains that have been parallelized, and bind those values. Or make
  // sure if they could be inferred the inference matches what was set.
  for (auto& entry : parallel_iter_extents) {
    auto p_type = entry.first;
    if (launch_constraints.hasDim(p_type)) {
      auto parallel_extents = entry.second;
      for (auto extent : parallel_extents) {
        auto inferred_val = expr_eval.evaluate(extent);
        if (inferred_val.hasValue()) {
          // This value could have been inferred, make sure it was set right.
          bool valid =
              inferred_val.as<int64_t>() == launch_constraints.getDim(p_type) ||
              launch_constraints.getRawVal(p_type) == -1;
          if (!useFallback() && !valid) {
            TORCH_WARN_ONCE(
                "Cannot validate parallelization scheme, "
                "this may be due to mixed broadcast axes that are parallelized.");
          }
        } else if (!expr_eval.precomputedValues()) {
          expr_eval.bind(extent, launch_constraints.getDim(p_type));
        }
        if (!launch_params.hasDim(p_type)) {
          // Bind the launch constraint into our evaluation context
          launch_params.bind(launch_constraints.getDim(p_type), p_type);
          // Makes sure the p-types bound to evaluators are the
          //  final values that will become the actual launch
          //  param size to ensure accurate smem buffer size
          //  computation.
          expr_eval.bind(p_type, launch_constraints.getDim(p_type));
        }
      }
    }
  }

  // Run through the rest of the parallel IterDomains and infer their size
  for (auto [p_type, extent] : simplified_parallel_iter_extents) {
    FUSER_PERF_SCOPE("KernelExecutor::ParallelBindingResolution");
    auto val = expr_eval.evaluate(extent);
    NVF_ERROR(
        val.hasValue(),
        "Tried to evaluate the extent, ",
        extent->toInlineString(),
        " for the ptype: ",
        p_type,
        " to set launch bounds but could not.");

    if (val > 0) {
      expr_eval.bind(p_type, val);
      launch_params.bind(val.as<int64_t>(), p_type);
    }
  }

  // Re-run the integer machine with all
  //  the thread sizes now determined.
  if (expr_eval.precomputedValues()) {
    expr_eval.precomputedValues()->evaluate();
  }

  const auto kernel = compiled_kernel_->lowered()->kernel();
  const auto& kernel_summary = kernel->summary();

  // Calculate Dynamic Shared Memory Size
  // Add workspace for reduction and broadcast
  int64_t reduction_broadcast_workspace = 0;
  const bool has_workspace = kernel_summary.has_block_reductions ||
      kernel_summary.has_grid_reductions ||
      kernel_summary.has_block_broadcasts || kernel_summary.has_grid_broadcasts;
  if (has_workspace &&
      kernel_summary.largest_smem_data_type != DataType::Null) {
    // Not using nThreads here since it does not handle uninitialized value

    // TODO: here is an optimization opportunity since welford uses int64_t for
    // N while the data type is not neccessarily double. But it may need more
    // work on the alignment
    const int welford_factor =
        kernel_summary.has_block_welford || kernel_summary.has_grid_welford ? 3
                                                                            : 1;
    // in outer reduction, may group iteration domain, e.g. when vectorized.
    const int64_t grouped_iter_factor = kernel_summary.num_grouped_iterations;

    NVF_CHECK(
        !(kernel_summary.has_iter_grouped_reductions && welford_factor == 3),
        "can't have welford and iter grouped reductions at the same time! Should be handled by grouped welford!");

    reduction_broadcast_workspace =
        (int64_t)dataTypeSize(
            kernel_summary.largest_smem_data_type, index_type) *
        grouped_iter_factor * welford_factor * launch_params.bdimx() *
        launch_params.bdimy() * launch_params.bdimz();

    if (kernel_summary.has_outer_grouped_grid_welford) {
      reduction_broadcast_workspace = std::max(
          reduction_broadcast_workspace,
          (int64_t)kernel_summary.outer_grouped_grid_welford_largest_smem_size);
    }
  }

  const auto dynamic_smem_size = computeSharedMemory(
      expr_eval,
      kernel_summary.dynamic_smem_allocations,
      index_type,
      reduction_broadcast_workspace);

  // Check that requested smem size can be dynamically allocated.
  //  This check is only done once a kernel has been compiled, since
  //  maybe_available_dynamic_smem_ needs to be evaluated on
  //  a compiled kernel.
  if (compiled_kernel_->hasCompiledKernel()) {
    validateDynamicSmemSize(dynamic_smem_size);
  }

  launch_params.setSmem(dynamic_smem_size);

  return launch_params;
}

std::vector<GlobalBufferInfo> KernelExecutor::getIntermediateBufferInfo(
    ExpressionEvaluator& expr_eval,
    DataType index_type) {
  FUSER_PERF_SCOPE("KernelExecutor::getIntermediateBufferInfo");
  std::vector<GlobalBufferInfo> global_buffers;

  const auto kernel = compiled_kernel_->lowered()->kernel();
  const auto& kernel_summary = kernel->summary();

  for (auto alloc : kernel_summary.global_allocations) {
    NVF_ERROR(
        alloc->buffer()->isA<TensorView>(),
        "Cannot allocate global buffers that are not tensors.");
    auto tv = alloc->buffer()->as<TensorView>();
    if (tv->isFusionOutput()) {
      continue;
    }
    GlobalBufferInfo info;
    info.tv = tv;
    info.zero_init = alloc->zeroInit();
    info.resets_to_zero = alloc->resetsToZero();
    // TODO: Allocation size needs to consider both expanded domains
    // as well as halo. Currently, allocation of tensors with halo is
    // only supported by inferShapeOfIntermediate, whereas expanded
    // domains are only supported by inferShapeOfOutput. Until the
    // halo support is revisited, use the former for all tensors
    // unless expanded and the latter otherwise. This assumes there's
    // no expanded domains with halo, which is fine for now.
    const auto has_expanded_domains = std::any_of(
        tv->getMaybeAllocationDomain().begin(),
        tv->getMaybeAllocationDomain().end(),
        [](IterDomain* id) { return id->hasExpandedExtent(); });
    std::tie(info.sizes, info.strides) = has_expanded_domains
        ? inferShapeOfOutput(tv, expr_eval)
        : inferShapeOfIntermediate(tv, alloc, expr_eval);
    auto dtype = (tv->dtype() == DataType::Index ? index_type : tv->dtype());
    info.type = data_type_to_aten(dtype);

    // Remember the tensor buffer used for storing kernel profile
    if (isOptionEnabled(EnableOption::KernelProfile) &&
        tv == kernel->profile().getBuffer()) {
      info.is_profile_buffer = true;
    }

    global_buffers.emplace_back(info);
  }

  return global_buffers;
}

namespace {

// Make sure the index type of Kernel is valid
void validateIndexType(
    kir::Kernel* kernel,
    const CompileParams& compile_params) {
  NVF_ERROR(
      !compile_params.index_type.has_value() ||
          kernel->indexType() == compile_params.index_type.value(),
      "Kernel index type and compilation index type don't match. Kernel type: ",
      kernel->indexType(),
      ". Compilation index type: ",
      compile_params.index_type.value());
}

void validateCooperativeLaunch(
    CUfunction kernel,
    const LaunchParams& launch_params,
    int64_t device_index) {
  int num_blocks_per_SM = -1;
  auto block_size =
      launch_params.bdimx() * launch_params.bdimy() * launch_params.bdimz();
  NVFUSER_CUDA_SAFE_CALL(cuOccupancyMaxActiveBlocksPerMultiprocessor(
      &num_blocks_per_SM,
      kernel,
      (int)block_size,
      (size_t)launch_params.smem()));

  auto grid_size =
      launch_params.gdimx() * launch_params.gdimy() * launch_params.gdimz();
  auto max_active_blocks = num_blocks_per_SM *
      at::cuda::getDeviceProperties((c10::DeviceIndex)device_index)
          ->multiProcessorCount;
  NVF_ERROR(
      (int64_t)(max_active_blocks) >= grid_size,
      "Wanted to launch a cooperative kernel, however the number of blocks is greater than ",
      "what can be resident on the GPU at once. Need: ",
      grid_size,
      " (",
      launch_params.gdimx(),
      " * ",
      launch_params.gdimy(),
      " * ",
      launch_params.gdimz(),
      ") but limited to ",
      num_blocks_per_SM,
      " * ",
      at::cuda::getDeviceProperties(device_index)->multiProcessorCount);
}

// Dump fusion inputs and outputs as well as some useful fusion
// information. Note that inputs and outputs are those that are passed
// to KernelExecutor::runFusion, so outputs may not be given.
void dumpFusionArgs(
    int64_t fusion_id,
    const KernelArgumentHolder& args,
    const LaunchParams& launch_constraints,
    const CompileParams& compile_params,
    const std::vector<at::Tensor>& outputs) {
  debug() << "Arguments for fusion" << fusion_id << ":" << std::endl
          << "Inputs:" << std::endl;
  for (auto i : c10::irange(args.size())) {
    debug() << "  " << args[i] << std::endl;
  }
  debug() << "Outputs:" << std::endl;
  for (const auto& output : outputs) {
    debug() << "  " << output.scalar_type() << " " << output.sizes()
            << " (strides = " << output.strides() << ")" << std::endl;
  }
  debug() << launch_constraints.toString();
  debug() << "maxrregcount= " << compile_params.maxrregcount << std::endl;
}

// Dump arguments that are passed to a CUDA kernel call, which include
// the inputs and outputs of the fusion as well as temporary
// global-memory buffers. Unlike dumpFusionArgs, which dumps inputs
// and outputs passed to KernelExecutor::runFusion, this function
// dumps those that are passed to a CUDA kernel.
void dumpKernelArgs(
    const int64_t fusion_id,
    const int64_t group_id,
    const KernelArgumentHolder& args,
    size_t num_inputs,
    const std::vector<at::Tensor>& allocated_outputs,
    const std::vector<at::Tensor>& intermediates,
    const std::vector<GlobalBufferInfo>& intermediates_info) {
  using namespace PolymorphicValue_functions;
  debug() << "Arguments for fusion " << fusion_id << " group " << group_id
          << ":" << std::endl
          << "Inputs:" << std::endl;
  for (auto i : c10::irange(num_inputs)) {
    debug() << "  " << toString(*args[i]) << std::endl;
  }
  debug() << "Outputs:" << std::endl;
  // note: add aliased outputs here.
  for (const auto& output : allocated_outputs) {
    debug() << "  " << output.scalar_type() << " " << output.sizes()
            << " (strides = " << output.strides()
            << ", address = " << output.data_ptr() << ")" << std::endl;
  }
  debug() << "Intermediate global buffers:" << std::endl;
  for (const auto i : c10::irange(intermediates.size())) {
    const auto& buffer = intermediates.at(i);
    const auto& zero_init = intermediates_info.at(i).zero_init;
    const auto& resets_to_zero = intermediates_info.at(i).resets_to_zero;
    debug() << "  " << buffer.scalar_type() << " " << buffer.sizes()
            << " is_zero_initialized: " << zero_init
            << " resets_to_zero: " << resets_to_zero << std::endl;
  }
}

} // namespace

void KernelExecutor::initializeExecutorEntry(
    ExecutorEntry& executor_entry,
    const KernelArgumentHolder& args,
    const LaunchParams& launch_constraints,
    const CompileParams& compile_params,
    const std::vector<at::Tensor>& outputs,
    DataType index_type) {
  FUSER_PERF_SCOPE("KernelExecutor::initializeExecutorEntry");

  ExpressionEvaluator expr_eval;
  evaluatorPrecomputedValues()->bindInputs(args);
  expr_eval.precomputedValues() = evaluatorPrecomputedValues().get();

  auto launch_params = computeLaunchParams(
      launch_constraints, expr_eval, warp_size_, index_type);

  for (const auto& entry : compiled_kernel_->kernel()->summary().validations) {
    NVF_CHECK(expr_eval.evaluate(entry.first).as<bool>(), entry.second);
  }

  executor_utils::validateVectorizedTensors(
      compiled_kernel_->kernel(),
      args,
      outputs,
      compileTimeDataCache(),
      expr_eval);

  executor_utils::validateCircularBuffering(
      compiled_kernel_->kernel(), expr_eval);

  // Check that a full warp exists in blockDim.x if the kernel contains
  // ElectSync predicate.
  constexpr int64_t warp_size = 32;
  NVF_ERROR(
      !compiled_kernel_->kernel()->summary().has_elect_sync_predicate ||
          launch_params.bdimx() >= warp_size,
      "This cuda kernel contains electSync predicate. "
      "Expected blockDim.x >= 32 but found ",
      launch_params.bdimx());

  std::vector<GlobalBufferInfo> output_info;

  if (outputs.empty()) {
    output_info = getBufferInfos(
        expr_eval, index_type, compiled_kernel_->kernel()->outputs());
  } else {
    // Need to save the information necessary for allocations as
    // future uses of this ExecutorEntry may not be provided with
    // allocated outputs
    for (const auto& output : outputs) {
      output_info.emplace_back(GlobalBufferInfo{
          .sizes = output.sizes().vec(),
          .strides = output.strides().vec(),
          .type = output.scalar_type()});
    }
  }

  auto intermediates = getIntermediateBufferInfo(expr_eval, index_type);

  // All information is gathered. Save it to ExecutorEntry
  executor_entry.launch_params = launch_params;
  executor_entry.outputs = output_info;
  executor_entry.intermediates = intermediates;
  executor_entry.init = true;
}

/// Copies the data, logical_size, and alloc_stride parameters to the
/// appropriate parts of entry.args[idx].
///
/// For GPU tensors, we pass a Tensor<type, rank, rank> struct (see
/// runtime/tensor.cu), where the rank describes the number of elements in the
/// shape and stride arrays. The actual shapes and strides are dynamic, but the
/// type and rank of the tensors are actually static (changing them would need
/// a new FusionDefinition). So we create the storage area for the
/// Tensor<t,r,r> during ::computeArgs, and then in this function we just
/// update that memory with the current values for the tensor's base address,
/// shape, and strides.
///
/// @param entry the entry we have previously setup for this fusion
/// @param idx the index into entry.args and related parallel arrays in the
///            entry.
/// @param idx_type_size generally sizeof(int32_t) or sizeof(int64_t); used for
///                      computing how large the arrays to copy are.
static void fillTensorArgMetadata(
    KernelExecutor::ExecutorEntry& entry,
    const PolymorphicValue& tensor_metadata,
    size_t idx,
    size_t idx_type_size) {
  void* data = tensor_metadata->*&TensorMetaData::data;
  // g++ has trouble inferring the types of more complicated fields through our
  // *& operators. Creating an `auto` alias as a temporary resolves this
  // problem.
#define TMD_ARRAY_REF(pv, field)                  \
  ({                                              \
    const auto& fld_tmp_ = pv->*&field;           \
    const c10::IntArrayRef& fld_aref_ = fld_tmp_; \
    fld_aref_;                                    \
  })
  const c10::IntArrayRef& shape =
      TMD_ARRAY_REF(tensor_metadata, TensorMetaData::logical_size);
  const c10::IntArrayRef& strides =
      TMD_ARRAY_REF(tensor_metadata, TensorMetaData::alloc_stride);
#undef TMD_ARRAY_REF

  // These are the three offsets we need to copy into.
  std::array<std::byte*, 3> offsets = {
      entry.args[idx].data(), // data ptr
      entry.args[idx].data() + sizeof(void*), // shape array
      // strides array:
      entry.args[idx].data() + sizeof(void*) + shape.size() * idx_type_size,
  };

  memcpy(offsets[0], &data, sizeof(void*));
  switch (idx_type_size) {
    case sizeof(int64_t): {
      // we use i64's for our sizes, so can use a simple copy here
      memcpy(offsets[1], shape.data(), shape.size() * sizeof(int64_t));
      memcpy(offsets[2], strides.data(), strides.size() * sizeof(int64_t));
    } break;
    case sizeof(int32_t): {
      // we need to cast per-element, so need a loop.
      // This case happens when the kernel uses 32bit indices. Since we
      // (specifically TensorMetaData) store indices in 64bit, we can't
      // directly copy our buffer into the args buffer. We thus have to
      // manually downcast each element to fit in the smaller buffer.
      for (size_t i = 0; i < shape.size(); ++i) {
        const int32_t shp = static_cast<int32_t>(shape[i]);
        memcpy(offsets[1] + i * sizeof(int32_t), &shp, sizeof(int32_t));
      }
      // In rare cases we have fewer strides than shapes
      for (size_t i = 0; i < strides.size(); ++i) {
        const int32_t strd = static_cast<int32_t>(strides[i]);
        memcpy(offsets[2] + i * sizeof(int32_t), &strd, sizeof(int32_t));
      }
    } break;
    default:
      NVF_CHECK(0, "Unhandled index type size");
      break;
  }
}

// set the arguments that we'll pass to cuLaunchKernel. This should happen
// when we change the rank of a tensor or the number of arguments to a kernel.
// It does not need to happen when only shapes change---use recomputeArgs for
// that.
void KernelExecutor::computeArgs(
    ExecutorEntry& entry,
    ExpressionEvaluator& expr_eval,
    const kir::Kernel* kernel) const {
  FUSER_PERF_SCOPE("KernelExecutor::computeArgs");

  const std::vector<Val*>& params = kernel->parameters();
  entry.args.resize(params.size());
  entry.arg_ptrs.resize(params.size());
  const PrimDataType idx_type = kernel->indexType();
  for (size_t p = 0; p < params.size(); ++p) {
    entry.args[p] = getKernelArgument(expr_eval, params[p], idx_type);
    entry.arg_ptrs[p] = entry.args[p].data();
  }
}

// Reset the arguments that we'll pass to cuLaunchKernel. This needs to be
// invoked on every shape change.
void KernelExecutor::recomputeArgs(
    ExecutorEntry& entry,
    ExpressionEvaluator& expr_eval,
    const kir::Kernel* kernel) const {
  FUSER_PERF_SCOPE("KernelExecutor::recomputeArgs");
  // assert(entry.init && "entry was never initialized");

  const std::vector<Val*>& params = kernel->parameters();
  const PrimDataType idx_type = kernel->indexType();
  // assert(entry.args.size() == params.size());
  // assert(entry.arg_ptrs.size() == params.size());
  // assert(params.size() >= args.size());
  for (size_t p = 0; p < params.size(); ++p) {
    PolymorphicValue pv = expr_eval.evaluate(params[p]);
    if (pv.is<at::Tensor>() && pv.as<at::Tensor>().is_cuda()) {
      // GPU tensors are not passed directly: instead we pass a Tensor<type,
      // rank, rank> struct. The pointer and dimensions are dynamic, but the
      // types and ranks are actually static (changing the rank or the types
      // would need to be done via a new FusionDefinition). As such, we created
      // the Tensor<t, r, r> struct during ::computeArgs, and here we just fill
      // in the base address, shape, and stride arrays to cover whatever new
      // tensors we got this round.
      TensorView* mtv = dynamic_cast<TensorView*>(params[p]);
      const Val* mdexpr = IrBuilder::metadataExpr(mtv);
      const PolymorphicValue& tmd = expr_eval.evaluate(mdexpr);
      const size_t idx_type_size =
          PrimDataType::Int == idx_type ? sizeof(int64_t) : sizeof(int32_t);
      fillTensorArgMetadata(entry, tmd, p, idx_type_size);
    } else {
      entry.args[p] = getKernelArgument(expr_eval, params[p], idx_type);
    }
    entry.arg_ptrs[p] = entry.args[p].data();
  }
}

int64_t KernelExecutor::getAvailableDynamicSmemSize() {
  if (!available_dynamic_smem_size_.has_value()) {
    int size = 0;
    NVFUSER_CUDA_SAFE_CALL(cuFuncGetAttribute(
        &size,
        CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
        compiled_kernel_->cudaExecutable()->function));
    available_dynamic_smem_size_ = size;
  }
  return available_dynamic_smem_size_.value();
}

int64_t KernelExecutor::getStaticSmemSize() {
  if (!static_smem_size_.has_value()) {
    int size = 0;
    // Is this really a costly operation worth caching?
    NVFUSER_CUDA_SAFE_CALL(cuFuncGetAttribute(
        &size,
        CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES,
        compiled_kernel_->cudaExecutable()->function));
    static_smem_size_ = size;
  }
  return static_smem_size_.value();
}

// TODO: Move to CompiledKernel
void KernelExecutor::validateDynamicSmemSize(int64_t dynamic_smem_size) {
  // If specified, check that dynamic smem size matches what the scheduler
  // expects
  int64_t expected_dynamic_smem_size =
      compiled_kernel_->fusion()->expectedDynamicSmemBytes();
  if (expected_dynamic_smem_size >= 0) {
    NVF_ERROR(
        dynamic_smem_size == expected_dynamic_smem_size,
        "Actual dynamic smem allocation ",
        dynamic_smem_size,
        " does not match expected size ",
        expected_dynamic_smem_size);
  }
  NVF_ERROR(
      getStaticSmemSize() + dynamic_smem_size < device_smem_limit_,
      "The total shared memory allocation is larger than available memory.",
      " Dynamic size: ",
      dynamic_smem_size,
      ". Static size: ",
      getStaticSmemSize(),
      ". Required total size: ",
      getStaticSmemSize() + dynamic_smem_size,
      ". Device limit size: ",
      device_smem_limit_);
}

// TODO: Move to CompiledKernel
int64_t KernelExecutor::ensureAvailableDynamicSmemSize(
    int64_t dynamic_smem_size) {
  NVF_ERROR(
      compiled_kernel_->hasCompiledKernel(),
      "Cannot set dynamic smem size unless kernel is compiled");
  if (dynamic_smem_size > getAvailableDynamicSmemSize()) {
    validateDynamicSmemSize(dynamic_smem_size);
    NVFUSER_CUDA_SAFE_CALL(cuFuncSetAttribute(
        compiled_kernel_->cudaExecutable()->function,
        CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
        dynamic_smem_size));
    available_dynamic_smem_size_ = dynamic_smem_size;
  }
  return getAvailableDynamicSmemSize();
}

// TODO: Move to CompiledKernel
void KernelExecutor::resetCompiledKernelProperties() {
  available_dynamic_smem_size_.reset();
  static_smem_size_.reset();
}

std::vector<at::Tensor> KernelExecutor::run(
    KernelArgumentHolder& args,
    const LaunchParams& launch_constraints,
    CompileParams compile_params,
    std::vector<at::Tensor> outputs) {
  FUSER_PERF_SCOPE("KernelExecutor::runFusion");

  if (isProfilerEnabled()) {
    NVF_CHECK(
        group_id_ >= 0,
        "An invalid segment id is passed to FusionProfiler!:",
        group_id_);
    SegmentProfiler& sprof = FusionProfiler::segment(group_id_);
    sprof.inputBytesAccessed(computeBytes(args));
    sprof.scheduler(toString(scheduler_type_));
    FusionProfiler::segment(group_id_).setDevice(args.getDeviceIndex());
    sprof.startKernel();
  }

  NVF_ERROR(isCompiled());
  NVF_ERROR(
      outputs.empty() || (outputs.size() == fusion()->outputs().size()),
      __func__,
      " provided number of outputs does not match fusion output");

  NVF_ERROR(
      !args.getCacheId().has_value() || outputs.empty(),
      "short cut input cache is not compatible with pre-allocated output");

  validateIndexType(compiled_kernel_->kernel(), compile_params);

  const auto num_inputs = args.size();

  if (isDebugDumpEnabled(DebugDumpOption::FusionArgs)) {
    dumpFusionArgs(
        fusion_id_, args, launch_constraints, compile_params, outputs);
  }

  c10::DeviceGuard dg(compiled_kernel_->options().device);
  auto stream = at::cuda::getCurrentCUDAStream();
  at::cuda::jit::initializeCudaContext();
  NVF_ERROR(compiled_kernel_->lowered());

  // Placeholder for the case where parameter cache is not used
  ExecutorEntry temporary_executor_entry;

  ExecutorEntry* executor_entry = args.getCacheId().has_value() &&
          !compiled_kernel_->disablePaarameterCache()
      ? &executor_entry_lookup_[*args.getCacheId()]
      : &temporary_executor_entry;

  // Initialize the executor entry if not initlized
  if (!executor_entry->init) {
    initializeExecutorEntry(
        *executor_entry,
        args,
        launch_constraints,
        compile_params,
        outputs,
        compiled_kernel_->kernel()->indexType());
  }

  if (!(executor_entry->launch_params.nThreads() <=
            compiled_kernel_->blockSizeHighWaterMark() &&
        compile_params.maxrregcount ==
            compiled_kernel_->maxrregcountHighWaterMark())) {
    compiled_kernel_->recompileKernel(
        executor_entry->launch_params, compile_params);
  }

  // TODO: Why does this need to be stored in the class?
  launch_params_ = executor_entry->launch_params;

  // context manager to disable auto grad for `empty_cuda` calls later
  at::AutoDispatchBelowADInplaceOrView non_variable_type_mode;

  // Bind fusion inputs
  auto expr_eval = executor_utils::bindInputs(args, fusion().get());

  // only allocate outputs when not given
  if (outputs.empty()) {
    outputs = allocateOutputs(
        compiled_kernel_->fusion(),
        executor_entry->outputs,
        compiled_kernel_->options().device,
        expr_eval);
  }
  args.push(outputs);

  for (const auto i : c10::irange(outputs.size())) {
    auto output = compiled_kernel_->kernel()->outputs()[i];
    if (std::any_of(
            compiled_kernel_->kernel()->inputs().begin(),
            compiled_kernel_->kernel()->inputs().end(),
            [&](const auto& in) { return in == output; })) {
      // Skip trivially forwarded outputs because they are just placeholders
      continue;
    }
    expr_eval.bind(
        output, *args[compiled_kernel_->kernel()->inputs().size() + i]);
  }

  std::vector<at::Tensor> intermediates;
  at::Tensor profile_buffer;
  {
    FUSER_PERF_SCOPE("KernelExecutor::runFusion::intermediates");
    for (const auto i : c10::irange(executor_entry->intermediates.size())) {
      const auto& buf_info = executor_entry->intermediates.at(i);
      bool has_expansion = false;
      std::vector<int64_t> unexpanded_sizes;
      unexpanded_sizes.reserve(buf_info.sizes.size());
      NVF_ERROR(buf_info.sizes.size() == buf_info.strides.size())
      for (const auto j : c10::irange(buf_info.sizes.size())) {
        if (buf_info.strides[j] == 0) {
          has_expansion = true;
          unexpanded_sizes.push_back(1L);
        } else {
          unexpanded_sizes.push_back(buf_info.sizes[j]);
        }
      }
      at::Tensor intermediate_buffer;
      if (buf_info.zero_init) {
        if (isOptionEnabled(EnableOption::ReuseZeroedMemory) ||
            buf_info.resets_to_zero) {
          // Allow access to reusable zeroed memory if buffer is guaranteed
          // to reset to zero upon completion of the kernel, or if we have
          // enabled the option (unsafe)
          intermediate_buffer = contigZeroedTensor(
              unexpanded_sizes,
              buf_info.type,
              compiled_kernel_->options().device);
        } else {
          intermediate_buffer = at::zeros(
              unexpanded_sizes,
              at::TensorOptions()
                  .dtype(buf_info.type)
                  .device(compiled_kernel_->options().device));
        }
      } else {
        intermediate_buffer = at::native::empty_cuda(
            unexpanded_sizes,
            buf_info.type,
            c10::nullopt,
            compiled_kernel_->options().device,
            c10::nullopt);
        if (shouldFillAllocationWithNan()) {
          fillTensorWithNan(intermediate_buffer);
        }
      }
      if (has_expansion) {
        intermediate_buffer =
            at::native::expand(intermediate_buffer, buf_info.sizes);
      }
      args.push(intermediate_buffer);
      intermediates.push_back(intermediate_buffer);
      expr_eval.bind(
          compiled_kernel_->kernel()
              ->summary()
              .global_allocations.at(i)
              ->buffer(),
          *args
              [compiled_kernel_->kernel()->inputs().size() + outputs.size() +
               i]);
      if (buf_info.is_profile_buffer) {
        profile_buffer = intermediate_buffer;
      }
    }
  }

  if (executor_entry->args.empty()) {
    computeArgs(*executor_entry, expr_eval, compiled_kernel_->kernel());
  }

  if (isDebugDumpEnabled(DebugDumpOption::LaunchParam)) {
    launch_params_.print();
  }

  if (isDebugDumpEnabled(DebugDumpOption::KernelArgs)) {
    dumpKernelArgs(
        fusion_id_,
        group_id_,
        args,
        num_inputs,
        outputs,
        intermediates,
        executor_entry->intermediates);
  }

  if (isDebugDumpEnabled(DebugDumpOption::IndexType)) {
    debug() << "Index type: " << compiled_kernel_->kernel()->indexType()
            << std::endl;
  }

  if (execute_kernel_ && !compiled_kernel_->kernel()->topLevelExprs().empty()) {
    FUSER_PERF_SCOPE("KernelExecutor::runFusion::execute_kernel");
    ensureAvailableDynamicSmemSize(executor_entry->launch_params.smem());

    recomputeArgs(*executor_entry, expr_eval, compiled_kernel_->kernel());

    if (isDebugDumpEnabled(DebugDumpOption::Occupancy) ||
        isDebugDumpEnabled(DebugDumpOption::PerfDebugVerbose)) {
      int blocks_per_sm = -1;
      NVFUSER_CUDA_SAFE_CALL(cuOccupancyMaxActiveBlocksPerMultiprocessor(
          &blocks_per_sm,
          compiled_kernel_->cudaExecutable()->function,
          launch_params_.nThreads(),
          launch_params_.smem()));

      const int64_t device_id = static_cast<unsigned char>(
          compiled_kernel_->options().device.index());
      const auto prop =
          at::cuda::getDeviceProperties((c10::DeviceIndex)device_id);
      const int64_t warps_per_sm =
          ceilDiv(blocks_per_sm * launch_params_.nThreads(), prop->warpSize);

      const int hw_max_warps =
          prop->maxThreadsPerMultiProcessor / prop->warpSize;
      const float occupancy = (float)warps_per_sm / (float)hw_max_warps * 100.f;
      setKernelOccupancy(occupancy);
      std::ostringstream oss;
      oss << std::fixed << std::setprecision(2) << occupancy << "%";

      debug() << "num_sms=" << prop->multiProcessorCount
              << ", blocks_per_sm=" << blocks_per_sm
              << ", warps_per_sm=" << warps_per_sm
              << ", occupancy=" << oss.str() << std::endl;
    }

    if (!compiled_kernel_->kernel()->summary().has_cooperative_grid_reduction) {
      FUSER_PERF_SCOPE("ExecutorRunFusion::cuLaunchKernel");
      NVFUSER_CUDA_SAFE_CALL(cuLaunchKernel(
          compiled_kernel_->cudaExecutable()->function,
          launch_params_.gdimx(),
          launch_params_.gdimy(),
          launch_params_.gdimz(),
          launch_params_.bdimx(),
          launch_params_.bdimy(),
          launch_params_.bdimz(),
          launch_params_.smem(),
          stream,
          executor_entry->arg_ptrs.data(),
          nullptr));
    } else {
      FUSER_PERF_SCOPE("ExecutorRunFusion::cuLaunchCooperativeKernel");
      NVFUSER_CUDA_SAFE_CALL(cuLaunchCooperativeKernel(
          compiled_kernel_->cudaExecutable()->function,
          launch_params_.gdimx(),
          launch_params_.gdimy(),
          launch_params_.gdimz(),
          launch_params_.bdimx(),
          launch_params_.bdimy(),
          launch_params_.bdimz(),
          launch_params_.smem(),
          stream,
          executor_entry->arg_ptrs.data()));
    }
  }

  releaseZeroedMemory();

  if (isOptionEnabled(EnableOption::KernelProfile)) {
    debug() << compiled_kernel_->kernel()->profile().toString(profile_buffer);
  }

  if (isProfilerEnabled()) {
    auto& sprof = FusionProfiler::segment(group_id_);
    sprof.stopKernel();
    sprof.outputBytesAccessed(computeBytes(outputs));
  }

  return outputs;
}

flatbuffers::Offset<serde::KernelExecutor> KernelExecutor::serialize(
    flatbuffers::FlatBufferBuilder& builder) const {
  // See table definition for KernelExecutor in serde/fusion_cache.fbs
  using fb_executor_entry = flatbuffers::Offset<serde::ExecutorEntry>;

  // Separate unordered_map for executor_entry_lookup into key and value
  // vectors. The key value is the cache_id value in the KernelArgumentHolder.
  std::vector<size_t> executor_entry_lookup_keys_fb;
  std::vector<fb_executor_entry> executor_entry_lookup_values_fb;
  for (const auto& [key, value] : executor_entry_lookup_) {
    executor_entry_lookup_keys_fb.push_back(key);
    executor_entry_lookup_values_fb.push_back(serialize(builder, value));
  }

  // When compilation is skipped, avoid serializing cubin because it doesn't
  // exist. The remaining fields are also not necessary in this case.
  if (!compiledKernel()->hasCompiledKernel()) {
    return serde::CreateKernelExecutorDirect(builder);
  }

  return serde::CreateKernelExecutorDirect(
      builder,
      device_smem_limit_,
      compiledKernel()->blockSizeHighWaterMark(),
      compiledKernel()->maxrregcountHighWaterMark(),
      warp_size_,
      toUnderlying(scheduler_type_),
      fusion_id_,
      concrete_id_,
      runtime_id_,
      group_id_,
      compiledKernel()->kernelCode().c_str(),
      &executor_entry_lookup_keys_fb,
      &executor_entry_lookup_values_fb,
      toUnderlying(compiledKernel()->kernel()->indexType()),
      serialize(builder, compiledKernel()->cudaExecutable().get()));
}

flatbuffers::Offset<serde::CudaKernel> KernelExecutor::serialize(
    flatbuffers::FlatBufferBuilder& builder,
    const executor_utils::CudaExecutable* compiled_kernel) const {
  NVF_ERROR(
      compiledKernel()->cudaExecutable() != nullptr &&
          (!compiled_kernel->cubin.empty() || !compiled_kernel->ptx.empty()),
      "Expected compiled cuda kernel before serializing KernelExecutor.");

  auto fb_kernel_name = builder.CreateString(compiled_kernel->kernel_name);
  auto fb_compile_args = builder.CreateString(compiled_kernel->compile_args);

  flatbuffers::Offset<flatbuffers::Vector<uint8_t>> fb_cubin = 0;
  flatbuffers::Offset<flatbuffers::String> fb_cubin_filename = 0;
  if (!compiled_kernel->cubin.empty()) {
    uint8_t* cubin_ptr = nullptr;
    fb_cubin = builder.CreateUninitializedVector(
        compiled_kernel->cubin.size(), &cubin_ptr);
    std::copy(
        compiled_kernel->cubin.begin(),
        compiled_kernel->cubin.end(),
        cubin_ptr);
    fb_cubin_filename = builder.CreateString(compiled_kernel->cubin_filename);
  }

  flatbuffers::Offset<flatbuffers::Vector<uint8_t>> fb_ptx = 0;
  flatbuffers::Offset<flatbuffers::String> fb_ptx_filename = 0;
  if (!compiled_kernel->ptx.empty()) {
    uint8_t* ptx_ptr = nullptr;
    fb_ptx = builder.CreateUninitializedVector(
        compiled_kernel->ptx.size(), &ptx_ptr);
    std::copy(
        compiled_kernel->ptx.begin(), compiled_kernel->ptx.end(), ptx_ptr);
    fb_ptx_filename = builder.CreateString(compiled_kernel->ptx_filename);
  }

  serde::CudaKernelBuilder ckb(builder);
  ckb.add_cubin(fb_cubin);
  ckb.add_cubin_filename(fb_cubin_filename);
  ckb.add_ptx(fb_ptx);
  ckb.add_ptx_filename(fb_ptx_filename);
  ckb.add_kernel_name(fb_kernel_name);
  ckb.add_compile_args(fb_compile_args);
  ckb.add_block_size(compiled_kernel->block_size);
  return ckb.Finish();
}

flatbuffers::Offset<serde::ExecutorEntry> KernelExecutor::serialize(
    flatbuffers::FlatBufferBuilder& builder,
    const ExecutorEntry& data) const {
  // See table definition for ExecutorEntry in serde/fusion_cache.fbs

  // Serialize GlobalBufferInfo for outputs.
  // We map the output TensorView pointer to its corresponding position in
  // fusion outputs assuming that the output ordering is consistent.
  using fb_global_buffer_info = flatbuffers::Offset<serde::GlobalBufferInfo>;
  std::vector<fb_global_buffer_info> outputs_fb;
  outputs_fb.reserve(data.outputs.size());
  for (const auto& buffer : data.outputs) {
    auto tv_iter = std::find(
        compiledKernel()->kernel()->outputs().cbegin(),
        compiledKernel()->kernel()->outputs().cend(),
        buffer.tv);
    auto tv_position = (tv_iter == compiledKernel()->kernel()->outputs().cend())
        ? -1
        : std::distance(
              compiledKernel()->kernel()->outputs().cbegin(), tv_iter);
    outputs_fb.push_back(
        serialize(builder, buffer, tv_position, true /* is_fusion_output */));
  }

  // Serialize GlobalBufferInfo for intermediates.
  // We map the intermediate TensorView pointer to its corresponding position in
  // KernelSummary global allocations. We assume that the ordering is consistent
  // between GpuLower objects with the same scheduled fusion.
  std::vector<fb_global_buffer_info> intermediates_fb;
  intermediates_fb.reserve(data.intermediates.size());
  for (const auto& buffer : data.intermediates) {
    auto match_tv_predicate = [buffer_tv = buffer.tv](const kir::Allocate* a) {
      return a->buffer() == buffer_tv;
    };
    auto tv_iter = std::find_if(
        compiledKernel()->kernel()->summary().global_allocations.cbegin(),
        compiledKernel()->kernel()->summary().global_allocations.cend(),
        match_tv_predicate);
    auto tv_position =
        (tv_iter ==
         compiledKernel()->kernel()->summary().global_allocations.cend())
        ? -1
        : std::distance(
              compiledKernel()->kernel()->summary().global_allocations.cbegin(),
              tv_iter);
    intermediates_fb.push_back(
        serialize(builder, buffer, tv_position, false /* is_fusion_output */));
  }

  return serde::CreateExecutorEntryDirect(
      builder,
      data.init,
      data.launch_params.serialize(builder),
      &outputs_fb,
      &intermediates_fb);
}

flatbuffers::Offset<serde::GlobalBufferInfo> KernelExecutor::serialize(
    flatbuffers::FlatBufferBuilder& builder,
    const GlobalBufferInfo& data,
    int64_t tv_position,
    bool is_fusion_output) const {
  // See table definition for GlobalBufferInfo in serde/fusion_cache.fbs
  return serde::CreateGlobalBufferInfoDirect(
      builder,
      tv_position,
      &data.sizes,
      &data.strides,
      nvfuser::toUnderlying(data.type),
      data.zero_init,
      data.resets_to_zero,
      data.is_profile_buffer,
      is_fusion_output);
}

void KernelExecutor::deserialize(
    const serde::KernelExecutor* buffer,
    Fusion* _fusion,
    int8_t device_index,
    CompileParams compile_params,
    SchedulerType heuristic,
    int64_t fusion_id,
    int64_t concrete_id,
    int64_t runtime_id,
    int64_t group_id) {
  // See table definition for KernelExecutor in serde/fusion_cache.fbs

  NVF_ERROR(buffer != nullptr, "serde::KernelExecutor is nullptr.");
  NVF_ERROR(_fusion != nullptr, "Fusion is nullptr.");

  NVF_ERROR(
      fusion_id == buffer->fusion_id(),
      "Expected given fusion_id to match serde fusion_id.");
  NVF_ERROR(
      concrete_id == buffer->concrete_id(),
      "Expected given concrete_id to match serde concrete_id.");
  NVF_ERROR(
      runtime_id == buffer->runtime_id(),
      "Expected given runtime_id to match serde runtime_id.");
  NVF_ERROR(
      group_id == buffer->group_id(),
      "Expected given group_id to match serde group_id.");
  NVF_ERROR(
      toUnderlying(heuristic) == buffer->heuristic(),
      ": ",
      toUnderlying(heuristic),
      " vs ",
      buffer->heuristic());

  // Initialize CompileOptions
  auto device = c10::Device(c10::DeviceType::CUDA, device_index);
  c10::DeviceGuard dg(device);

  // Initialize internal fields
  device_smem_limit_ = buffer->device_smem_limit();
  // blockSizeHighWaterMark() = buffer->block_size_high_water_mark();
  // maxrregcountHighWaterMark() = buffer->maxrregcount_high_water_mark();
  warp_size_ = buffer->warp_size();
  // kernel_code_ = buffer->kernel_code()->str();

  // // KernelDB query checks kernel_code string and compile_params before
  // // copying cubin.
  // compile_params.index_type = serde::mapToNvfuserDtype(buffer->index_type());
  // compile_params.maxrregcount = maxrregcountHighWaterMark();

  // // Get lowered fusion
  // lowered_ = std::make_unique<GpuLower>(fusion, compile_params);
  // lowered_->run();

  // // Replace integers that are tensor sizes by named scalars like
  // "T0.size[0]" createKernelId(
  //     heuristic,
  //     buffer->fusion_id(),
  //     buffer->concrete_id(),
  //     buffer->runtime_id(),
  //     buffer->group_id());
  // setUsedTVs();

  // GlobalBufferInfo requires lowered kernel before deserialization
  for (auto idx : c10::irange(buffer->executor_entry_lookup_keys()->size())) {
    executor_entry_lookup_.emplace(
        buffer->executor_entry_lookup_keys()->Get(idx),
        deserialize(buffer->executor_entry_lookup_values()->Get(idx)));
  }

  // compiled_kernel_ = executor_utils::getexecutable(
  //     buffer->compiled_kernel(), compile_params);

  // NVF_ERROR(hasCompiledKernel(), "Failed to deserialize KernelExecutor");
}

KernelExecutor::ExecutorEntry KernelExecutor::deserialize(
    const serde::ExecutorEntry* buffer) {
  // See table definition for ExecutorEntry in serde/fusion_cache.fbs

  NVF_ERROR(buffer != nullptr, "serde::ExecutorEntry is nullptr.");

  ExecutorEntry entry;

  entry.init = buffer->init();

  entry.launch_params.deserialize(buffer->launch_params());

  for (auto output_buffer : *buffer->outputs()) {
    entry.outputs.push_back(deserialize(output_buffer));
  }

  for (auto intermediate_buffer : *buffer->intermediates()) {
    entry.intermediates.push_back(deserialize(intermediate_buffer));
  }

  return entry;
}

GlobalBufferInfo KernelExecutor::deserialize(
    const serde::GlobalBufferInfo* buffer) {
  // See table definition for GlobalBufferInfo in serde/fusion_cache.fbs

  NVF_ERROR(buffer != nullptr, "serde::GlobalBufferInfo is nullptr.");

  NVF_ERROR(
      buffer->tv() != -1, "Serialization failed to encode buffer tv position.");

  NVF_ERROR(
      compiled_kernel_->lowered() != nullptr,
      "Lowered kernel is not initialized.");

  GlobalBufferInfo info;
  if (buffer->is_fusion_output()) {
    auto out_val = compiled_kernel_->kernel()->outputs().at(buffer->tv());
    NVF_ERROR(out_val != nullptr);
    info.tv = dynamic_cast<TensorView*>(out_val);
  } else {
    auto out_val = compiled_kernel_->kernel()->summary().global_allocations.at(
        buffer->tv());
    NVF_ERROR(out_val != nullptr);
    info.tv = dynamic_cast<TensorView*>(out_val->buffer());
  }

  for (auto dim_size : *buffer->sizes()) {
    info.sizes.emplace_back(dim_size);
  }

  for (auto dim_stride : *buffer->strides()) {
    info.strides.emplace_back(dim_stride);
  }

  info.type = serde::mapToAtenDtype(buffer->dtype());
  info.zero_init = buffer->zero_init();
  info.resets_to_zero = buffer->resets_to_zero();
  info.is_profile_buffer = buffer->is_profile_buffer();
  return info;
}

} // namespace nvfuser
