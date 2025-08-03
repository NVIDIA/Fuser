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
#include <device_lower/lower2device.h>
#include <device_lower/utils.h>
#include <driver_api.h>
#include <fusion_profiler.h>
#include <global_allocator.h>
#include <instrumentation.h>
#include <ir/all_nodes.h>
#include <ir/graphviz.h>
#include <ir/utils.h>
#include <iter_visitor.h>
#include <kernel_ir.h>
#include <multidevice/utils.h>
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

KernelArgumentHolder ExprEvalExecutor::run(
    const KernelArgumentHolder& args,
    KernelArgumentHolder outputs) {
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
        auto out_tensor = expr_eval.evaluate(out_val).as<at::Tensor>();
        expr_eval.bind(out_val, out_tensor);
        outputs.push(out_tensor);
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

bool KernelExecutor::supported(Fusion* fusion) {
  FUSER_PERF_SCOPE("KernelExecutor::supported");
  return !hasCpuScalarOutputs(fusion);
}

void KernelExecutor::compile(
    Fusion* fusion,
    const KernelArgumentHolder& args,
    const LaunchParams& launch_constraints,
    CompileParams compile_params,
    SchedulerType scheduler_type) {
  FUSER_PERF_SCOPE("KernelExecutor::compile");

  NVF_ERROR(
      supported(fusion),
      "KernelExecutor does not support the Fusion provided.");

  NVF_ERROR(
      !fusion->outputs().empty(), "No output found for this kernel, aborting.");

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
  std::vector<Expr*> exprs = fusion->exprs();
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
        "Compilation with int32 is requested but int64 is required for the "
        "arguments");
  } else {
    // If the given compile option doesn't specify the index type, and
    // the arguments require 64-bit indexing, we need to use 64-bit
    // indexing. Note that if the arg type is 32-bit, it doesn't mean
    // it's safe to use 32-bit for the whole kernel, so unless it's
    // specified through CompileParams, we do not use 32-bit indexing.
    compile_params.index_type = arg_index_type;
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
  compiled_kernel_ = std::make_unique<CompiledKernel>(
      fusion,
      compile_params,
      device,
      scheduler_type,
      fusion_id_,
      concrete_id_,
      runtime_id_,
      group_id_,
      lowering_hooks_,
      post_lowering_hooks_);

  // TODO: pass block_size here;
  std::optional<int64_t> dynamic_smem = std::nullopt;
  std::optional<int64_t> block_size = std::nullopt;

  auto launch_params = launch_constraints;
  if (!args.empty()) {
    auto expr_eval =
        executor_utils::bindInputs(args, compiled_kernel_->lowered()->kernel());
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

  // Launch parameters are required to compile the kernel to:
  // (1) validate register sharing
  // (2) runtime function may use static CTA shape, e.g.
  //     iterGroupedStaticWarpAllReduce
  compiled_kernel_->compile(launch_params);

  // These should be nullopt at this point, but reset just in case
  resetCompiledKernelProperties();

  // If the dynamic shmem size is known, make sure the compiled kernel
  // has at least that size of dynamic shmem
  if (dynamic_smem.has_value()) {
    ensureAvailableDynamicSmemSize(dynamic_smem.value());
  }
  if (isProfilerEnabled()) {
    FusionProfiler::segment(group_id_).stopCompile();
  }

  for (auto expr : exprs) {
    if (ir_utils::isCpAsyncBulk(expr)) {
      has_tma_ = true;
    }
    if (expr->isA<RNGOp>()) {
      has_rng_ = true;
    }
  }

  // If an output has an alias to an input and is marked Evaluate, then
  // expression evaluator evaluate is called on that output to produce the meta
  // data manipulation it requires. If that manipulation is something like a
  // slice, and that slice has a symbolic integer it depends on, then this
  // function returns true.
  //
  // This could happen for other examples and has_dynamic_alias_ will be true if
  // to evaluate the output that has an alias, other values besides the aliased
  // input need to be bound to the expression evaluator to evaluate the output.
  for (auto output : fusion->outputs()) {
    if (output->isA<TensorView>()) {
      auto out_tv = output->as<TensorView>();
      auto alias_info = fusion->getOutputAlias(out_tv);
      if (alias_info.type != AllocationType::Evaluate) {
        continue;
      }
      auto aliased_to = alias_info.aliased_io->as<TensorView>();
      auto inputs = InputsOf::output(out_tv);
      for (auto input : inputs) {
        if (input->isA<TensorView>() && input->sameAs(aliased_to)) {
          continue;
        }

        if (input->isConst()) {
          continue;
        }
        has_dynamic_alias_ = true;
      }
    }
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
                "this may be due to mixed broadcast axes that are "
                "parallelized.");
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
        "can't have welford and iter grouped reductions at the same time! "
        "Should be handled by grouped welford!");

    // For block reduction, each thread has a smem slot per reduction
    // When warp specialization is used, remove padded threads
    // For warp reduction, each warp has a smem slot per reduction
    int64_t n_compute_threads_or_warps = launch_params.nThreads();
    if (kernel_summary.circular_buffer_info.hasWarpSpecialized()) {
      n_compute_threads_or_warps -= kWarpSpecializationPaddedThreads;
    }
    if (kernel_summary.all_block_reductions_are_warp_reduction) {
      n_compute_threads_or_warps /= 32;
    }

    reduction_broadcast_workspace =
        dataTypeSizeByte(kernel_summary.largest_smem_data_type, index_type) *
        grouped_iter_factor * welford_factor * n_compute_threads_or_warps;

    if (kernel_summary.has_outer_grouped_grid_welford) {
      reduction_broadcast_workspace = std::max(
          reduction_broadcast_workspace,
          (int64_t)kernel_summary.outer_grouped_grid_welford_largest_smem_size);
    }

    // StackBasedSharedMemAllocator start from address 0 without considering the
    // shared memory reserved for reduction and broadcast workspace which is
    // only known at runtime. To avoid mis-alignment for TMA tensors, here we
    // enforce the workspace aligned at 128 Bytes. Same roundup is also added to
    // codegen.
    reduction_broadcast_workspace =
        roundUpToMultiple(reduction_broadcast_workspace, 128);

    if (isDebugDumpEnabled(DebugDumpOption::DynamicSharedMemory)) {
      debug() << "reduction_broadcast_workspace shared memory bytes: "
              << reduction_broadcast_workspace << std::endl;
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
  if (compiled_kernel_->isCompiled()) {
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
    // as well as halo. Currently, halo support has bene removed so we only need
    // to worry about the expand case which is handled in inferShapeofOutputs.
    // There used to also be a inferShapeOfIntermediate function before this
    // commit, but that was safely removed with halo support. This will need to
    // be revisited when halo support is added again.
    auto [sizes, strides] = inferShapeOfOutput(tv, expr_eval);
    info.shape_info.logical_sizes = sizes;
    info.shape_info.logical_strides = strides;
    auto dtype = tv->dtype() == DataType::Index ? index_type : tv->dtype();
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
      "Wanted to launch a cooperative kernel, however the number of blocks is "
      "greater than ",
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
    const KernelArgumentHolder& outputs) {
  debug() << "Arguments for fusion" << fusion_id << ":" << std::endl
          << "Inputs:" << std::endl;
  for (auto i : arange(args.size())) {
    debug() << "  " << args[i] << std::endl;
  }
  debug() << "Outputs:" << std::endl;
  for (const auto& output : outputs) {
    debug() << PolymorphicValue_functions::toString(output) << std::endl;
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
    const KernelArgumentHolder& allocated_outputs,
    const KernelArgumentHolder& intermediates,
    const std::vector<GlobalBufferInfo>& intermediates_info) {
  using namespace PolymorphicValue_functions;
  debug() << "Arguments for fusion " << fusion_id << " group " << group_id
          << ":" << std::endl
          << "Inputs:" << std::endl;
  for (auto i : arange(num_inputs)) {
    debug() << "  " << toString(args[i]) << std::endl;
  }
  debug() << "Outputs:" << std::endl;
  // note: add aliased outputs here.
  for (const auto& output : allocated_outputs) {
    debug() << "  " << PolymorphicValue_functions::toString(output)
            << std::endl;
  }
  debug() << "Intermediate global buffers:" << std::endl;
  for (const auto i : arange(intermediates.size())) {
    const auto& zero_init = intermediates_info.at(i).zero_init;
    const auto& resets_to_zero = intermediates_info.at(i).resets_to_zero;
    debug() << "  " << PolymorphicValue_functions::toString(intermediates[i])
            << " is_zero_initialized: " << zero_init
            << " resets_to_zero: " << resets_to_zero << std::endl;
  }
}

} // namespace

void KernelExecutor::initializeExecutorEntry(
    KernelExecutorEntry& executor_entry,
    const KernelArgumentHolder& args,
    const LaunchParams& launch_constraints,
    const CompileParams& compile_params,
    const KernelArgumentHolder& output_args,
    DataType index_type) {
  FUSER_PERF_SCOPE("KernelExecutor::initializeExecutorEntry");

  ExpressionEvaluator expr_eval =
      executor_utils::bindInputs(args, compiled_kernel_->kernel());

  auto launch_params = computeLaunchParams(
      launch_constraints, expr_eval, warp_size_, index_type);

  for (const auto& entry : compiled_kernel_->kernel()->summary().validations) {
    NVF_CHECK(expr_eval.evaluate(entry.first).as<bool>(), entry.second);
  }

  executor_utils::validateVectorizedTensors(
      compiled_kernel_->kernel(),
      args,
      output_args,
      compileTimeDataCache(),
      expr_eval);

  executor_utils::validateCircularBuffering(
      compiled_kernel_->kernel(), expr_eval);

  executor_utils::validateIndexCasts(
      compiled_kernel_->kernel(), expr_eval, launch_params);

  // Check that a full warp exists in blockDim.x if the kernel contains
  // ElectSync predicate.
  constexpr int64_t warp_size = 32;
  NVF_ERROR(
      !compiled_kernel_->kernel()->summary().has_elect_sync_predicate ||
          launch_params.bdimx() >= warp_size,
      "This cuda kernel contains electSync predicate. "
      "Expected blockDim.x >= 32 but found ",
      launch_params.bdimx());

  std::vector<GlobalBufferInfo> input_info;
  NVF_ERROR_EQ(std::ssize(compiled_kernel_->kernel()->inputs()), args.size());
  for (auto inp_idx : arange(compiled_kernel_->kernel()->inputs().size())) {
    auto input = compiled_kernel_->kernel()->inputs()[inp_idx];
    if (auto input_tv = dynamic_cast<TensorView*>(input)) {
      auto at_tensor = args[inp_idx].as<at::Tensor>();

      std::vector<int64_t> alloc_sizes;
      std::vector<int64_t> alloc_strides;
      if (input_tv->hasAllocation()) {
        std::tie(alloc_sizes, alloc_strides) =
            inferAndValidateAllocationSizesAndStrides(
                at_tensor, input_tv, expr_eval);
      }

      TensorShapeInfo shape_info;
      shape_info.logical_sizes = args[inp_idx].as<at::Tensor>().sizes().vec();
      shape_info.logical_strides =
          args[inp_idx].as<at::Tensor>().strides().vec();
      if (isSharded(input_tv)) {
        shape_info.unsharded_logical_sizes =
            unshardedSizes(input_tv, shape_info.logical_sizes);
      }
      shape_info.allocation_sizes = alloc_sizes;
      shape_info.allocation_strides = alloc_strides;
      GlobalBufferInfo info{
          input_tv,
          shape_info,
          data_type_to_aten(input_tv->dtype()),
          false,
          false,
          false};
      input_info.emplace_back(info);
    }
  }

  std::vector<GlobalBufferInfo> output_info;

  if (output_args.empty()) {
    output_info = getBufferInfos(
        expr_eval, index_type, compiled_kernel_->kernel()->outputs());
  } else {
    // Need to save the information necessary for allocations as
    // future uses of this KernelExecutorEntry may not be provided with
    // allocated outputs
    for (auto output_idx : arange(output_args.size())) {
      NVF_ERROR(
          output_args[output_idx].hasValue() &&
              output_args[output_idx].is<at::Tensor>(),
          "Output is not populated or not a Tensor");
      const auto& output_tensor = output_args[output_idx].as<at::Tensor>();
      GlobalBufferInfo info;
      info.type = output_tensor.scalar_type();
      auto out_val = compiled_kernel_->kernel()->outputs()[output_idx];
      NVF_ERROR(out_val->isA<TensorView>(), "Output is not a TensorView");
      info.tv = out_val->as<TensorView>();
      if (info.tv->hasAllocation()) {
        // Validate that the pre-allocated output tensor matches the allocation
        // domain requirements
        auto [alloc_sizes, alloc_strides] =
            inferAndValidateAllocationSizesAndStrides(
                output_tensor, info.tv, expr_eval);
        info.shape_info.allocation_sizes = alloc_sizes;
        info.shape_info.allocation_strides = alloc_strides;
      }
      info.shape_info.logical_sizes = output_tensor.sizes().vec();
      info.shape_info.logical_strides = output_tensor.strides().vec();
      output_info.emplace_back(info);
    }
  }

  auto intermediates = getIntermediateBufferInfo(expr_eval, index_type);

  // All information is gathered. Save it to KernelExecutorEntry
  executor_entry.launch_params = launch_params;
  executor_entry.outputs = output_info;
  executor_entry.output_aliased_to_input =
      executor_utils::getOutputAliasToInputMap(compiled_kernel_->kernel());
  executor_entry.intermediates = intermediates;
  executor_entry.inputs = input_info;
  executor_entry.init = true;
}

namespace {
GlobalBufferInfo& linear_buffer_info_getter(
    KernelExecutorEntry& entry,
    size_t idx) {
  if (idx < entry.inputs.size()) {
    return entry.inputs[idx];
  } else if (idx < entry.inputs.size() + entry.outputs.size()) {
    return entry.outputs[idx - entry.inputs.size()];
  } else if (
      idx <
      entry.inputs.size() + entry.outputs.size() + entry.intermediates.size()) {
    return entry
        .intermediates[idx - entry.inputs.size() - entry.outputs.size()];
  } else {
    NVF_CHECK(
        0,
        "Invalid buffer index: ",
        idx,
        " input size: ",
        entry.inputs.size(),
        " output size: ",
        entry.outputs.size(),
        " intermediate size: ",
        entry.intermediates.size());
  }
};
} // namespace

void KernelExecutor::computeArgs(
    KernelExecutorEntry& entry,
    const KernelArgumentHolder& args) const {
  FUSER_PERF_SCOPE("KernelExecutor::computeArgs");
  if (std::ssize(entry.args) != args.size()) {
    entry.args.resize(args.size());
    entry.arg_ptrs.resize(args.size());
  }

  NVF_ERROR_EQ(
      args.size(), std::ssize(compiled_kernel_->kernel()->parameters()));

  for (auto inp : compiled_kernel_->kernel()->inputs()) {
    if (!inp->isA<TensorView>()) {
      continue;
    }
  }

  const PrimDataType idx_type = compiled_kernel_->kernel()->indexType();
  int64_t buffer_info_idx = 0;
  for (auto&& [arg_idx, arg] : enumerate(args)) {
    if (arg.is<at::Tensor>() && arg.as<at::Tensor>().is_cuda()) {
      const auto& buffer_info =
          linear_buffer_info_getter(entry, buffer_info_idx++);
      entry.args[arg_idx] = tensorToBytes(
          arg,
          buffer_info.shape_info.logical_sizes,
          buffer_info.shape_info.allocation_strides.empty()
              ? buffer_info.shape_info.logical_strides
              : buffer_info.shape_info.allocation_strides,
          idx_type,
          getLastDimAdjustment(buffer_info.tv->dtype()),
          buffer_info.shape_info.unsharded_logical_sizes);
      entry.arg_ptrs[arg_idx] = entry.args[arg_idx].data();
    } else {
      if (arg.is<at::Tensor>()) {
        buffer_info_idx++;
      }
      auto bytes = polymorphicValueToBytes(
          arg,
          compiled_kernel_->kernel()->parameters()[arg_idx]->dtype(),
          idx_type);
      entry.args[arg_idx] = bytes;
      entry.arg_ptrs[arg_idx] = entry.args[arg_idx].data();
    }
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
      compiled_kernel_->kernel()->expectedDynamicSmemBytes();
  if (expected_dynamic_smem_size >= 0) {
    NVF_ERROR(
        dynamic_smem_size == expected_dynamic_smem_size,
        "Actual dynamic smem allocation ",
        dynamic_smem_size,
        " does not match expected size ",
        expected_dynamic_smem_size);
  }
  NVF_ERROR(
      getStaticSmemSize() + dynamic_smem_size <= device_smem_limit_,
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
      compiled_kernel_->isCompiled(),
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

namespace {
KernelArgumentHolder resolveRNGSeed(
    const kir::Kernel* kernel,
    KernelArgumentHolder& args) {
  ExpressionEvaluator expr_eval;
  KernelArgumentHolder resolved_args;
  resolved_args.reserve(args.size());
  int64_t arg_idx = 0;
  for (auto param : kernel->parameters()) {
    if (param->definition() &&
        param->definition()->isA<kir::GetRNGSeedAndOffsetFromHost>()) {
      resolved_args.push(expr_eval.evaluate(param));
    } else {
      resolved_args.push(args[arg_idx++]);
    }
  }
  return resolved_args;
}
} // namespace

// TODO: Reduce bindings to only those necessary to resolve missing params.
// TODO: Check if this could be reused to also resolve dynamic aliases.
KernelArgumentHolder KernelExecutor::resolveTMA(
    KernelExecutorEntry& entry,
    const KernelArgumentHolder& args) const {
  ExpressionEvaluator expr_eval;
  int64_t arg_idx = 0;
  NVF_ERROR(
      entry.inputs.size() == compiled_kernel_->kernel()->inputs().size(),
      "Input size mismatch");
  for (auto inp_idx : arange(entry.inputs.size())) {
    expr_eval.bind(
        compiled_kernel_->kernel()->inputs()[inp_idx], args[arg_idx++]);
  }

  NVF_ERROR(
      entry.outputs.size() == compiled_kernel_->kernel()->outputs().size(),
      "Output size mismatch");
  for (auto out_idx : arange(entry.outputs.size())) {
    expr_eval.bind(
        compiled_kernel_->kernel()->outputs()[out_idx], args[arg_idx++]);
  }

  for (const auto& intermediate_entry : entry.intermediates) {
    if (args[arg_idx].hasValue()) {
      expr_eval.bind(intermediate_entry.tv, args[arg_idx++]);
    }
  }

  KernelArgumentHolder resolved_args;
  for (auto param : compiled_kernel_->kernel()->parameters()) {
    resolved_args.push(expr_eval.evaluate(param));
  }
  return resolved_args;
}

KernelArgumentHolder KernelExecutor::run(
    KernelArgumentHolder args,
    KernelArgumentHolder output_args,
    const LaunchParams& launch_constraints,
    CompileParams compile_params) {
  FUSER_PERF_SCOPE("KernelExecutor::run");

  if (isProfilerEnabled()) {
    NVF_CHECK(
        group_id_ >= 0,
        "An invalid segment id is passed to FusionProfiler!:",
        group_id_);
    SegmentProfiler& sprof = FusionProfiler::segment(group_id_);
    sprof.inputBytesAccessed(computeBytes(args));
    sprof.scheduler(toString(compiledKernel()->schedulerType()));
    FusionProfiler::segment(group_id_).setDevice(args.getDeviceIndex());
    sprof.startKernel();
  }

  NVF_ERROR(isCompiled());
  NVF_ERROR(
      output_args.empty() ||
          (output_args.size() ==
           std::ssize(compiledKernel()->kernel()->outputs())),
      __func__,
      " provided number of outputs does not match fusion output");

  validateIndexType(compiled_kernel_->kernel(), compile_params);

  const auto num_inputs = args.size();

  if (isDebugDumpEnabled(DebugDumpOption::FusionArgs)) {
    dumpFusionArgs(
        fusion_id_, args, launch_constraints, compile_params, output_args);
  }

  c10::DeviceGuard dg(compiled_kernel_->device());
  auto stream = at::cuda::getCurrentCUDAStream();
  at::cuda::jit::initializeCudaContext();
  NVF_ERROR(compiled_kernel_->lowered());

  // Placeholder for the case where parameter cache is not used
  KernelExecutorEntry temporary_executor_entry;

  KernelExecutorEntry* executor_entry = args.getCacheId().has_value() &&
          !compiled_kernel_->launchParamCacheDisabled()
      ? &executor_entry_lookup_[*args.getCacheId()]
      : &temporary_executor_entry;

  // Initialize the executor entry if not initlized
  if (!executor_entry->init) {
    initializeExecutorEntry(
        *executor_entry,
        args,
        launch_constraints,
        compile_params,
        output_args,
        compiled_kernel_->kernel()->indexType());
  }

  if (!(executor_entry->launch_params.nThreads() <=
            compiled_kernel_->blockSizeHighWaterMark() &&
        compile_params.maxrregcount ==
            compiled_kernel_->maxrregcountHighWaterMark())) {
    NVF_ERROR(
        compiled_kernel_->blockSizeHighWaterMark() == 1,
        "Recompiling kernel because launch params or compile params changed. ",
        "water_mark = ",
        compiled_kernel_->blockSizeHighWaterMark(),
        ", ",
        compile_params.toString(),
        ", ",
        launch_constraints.toString());
    compiled_kernel_->recompileKernel(
        executor_entry->launch_params, compile_params);
  }

  // TODO: Why does this need to be stored in the class?
  launch_params_ = executor_entry->launch_params;

  // context manager to disable auto grad for `empty_cuda` calls later
  at::AutoDispatchBelowADInplaceOrView non_variable_type_mode;

  // only allocate outputs when not given
  if (output_args.empty()) {
    output_args = allocateOutputs(
        compiled_kernel_->kernel(),
        executor_entry->outputs,
        executor_entry->output_aliased_to_input,
        compiled_kernel_->device(),
        args,
        has_dynamic_alias_);
    if (has_dynamic_alias_) {
      ExpressionEvaluator expr_eval;
      if (has_dynamic_alias_ || has_tma_) {
        expr_eval =
            executor_utils::bindInputs(args, compiled_kernel_->kernel());
      }

      for (const auto i :
           arange(compiled_kernel_->kernel()->outputs().size())) {
        auto param = compiled_kernel_->kernel()->outputs()[i];
        if (!param->isA<TensorView>()) {
          continue;
        }
        if (compiled_kernel_->kernel()
                ->getOutputAlias(param->as<TensorView>())
                .type == AllocationType::Evaluate) {
          output_args[i] = expr_eval.evaluate(param);
        }
      }
    }
    NVF_ERROR(
        std::all_of(
            output_args.begin(),
            output_args.end(),
            [](const PolymorphicValue& arg) {
              return arg.hasValue() && arg.is<at::Tensor>();
            }),
        "Output is not populated or not a Tensor");
  }

  args.push(output_args);

  KernelArgumentHolder intermediate_args;
  at::Tensor profile_buffer;
  {
    FUSER_PERF_SCOPE("KernelExecutor::runFusion::intermediates");
    // Intermediates just use logical sizes and strides even though they're
    // really allocation sizes and strides.
    //
    // This is simply because the convention used is that allocation
    // sizes/strides are optional, logical are not.
    for (const auto intermediate_i :
         arange(executor_entry->intermediates.size())) {
      const auto& buf_info = executor_entry->intermediates.at(intermediate_i);
      bool has_expansion = false;
      std::vector<int64_t> unexpanded_sizes;
      unexpanded_sizes.reserve(buf_info.shape_info.logical_sizes.size());
      NVF_ERROR(
          buf_info.shape_info.logical_sizes.size() ==
          buf_info.shape_info.logical_strides.size())
      for (const auto j : arange(buf_info.shape_info.logical_sizes.size())) {
        if (buf_info.shape_info.logical_strides[j] == 0) {
          has_expansion = true;
          unexpanded_sizes.push_back(1L);
        } else {
          unexpanded_sizes.push_back(buf_info.shape_info.logical_sizes[j]);
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
              unexpanded_sizes, buf_info.type, compiled_kernel_->device());
        } else {
          intermediate_buffer = at::zeros(
              unexpanded_sizes,
              at::TensorOptions()
                  .dtype(buf_info.type)
                  .device(compiled_kernel_->device()));
        }
      } else {
        intermediate_buffer = at::native::empty_cuda(
            unexpanded_sizes,
            buf_info.type,
            c10::nullopt,
            compiled_kernel_->device(),
            c10::nullopt);
        if (shouldFillAllocationWithNan()) {
          fillTensorWithNan(intermediate_buffer);
        }
      }
      if (has_expansion) {
        intermediate_buffer = at::native::expand(
            intermediate_buffer, buf_info.shape_info.logical_sizes);
      }
      args.push(intermediate_buffer);
      intermediate_args.push(intermediate_buffer);
      if (buf_info.is_profile_buffer) {
        profile_buffer = intermediate_buffer;
      }
    }
  }

  if (args.size() != std::ssize(compiled_kernel_->kernel()->parameters())) {
    NVF_ERROR(
        has_tma_ || has_rng_,
        "No TMA or RNG found in the kernel, but detected an argument size "
        "mismatch.");
    // If args don't match one of two things is happening. We need to add TMA
    // related args or RNG related args. Resolve these scenarios.
    if (has_tma_) {
      // Resolving TMA requires binding all values and evaluating the TMA
      // arguments
      //
      // Resolving TMA also resolves RNG, so if TMA exists the resolveRNGSeed
      // function shouldn't also be called.
      args = resolveTMA(*executor_entry, args);
    } else if (has_rng_) {
      // Resolving RNG seed requires evaluating and adding those values, but
      // doesn't require binding all values as getting RNG seed and offset
      // doesn't depend on other values
      args = resolveRNGSeed(compiled_kernel_->kernel(), args);
    }
  }

  computeArgs(*executor_entry, args);

  if (isDebugDumpEnabled(DebugDumpOption::LaunchParam)) {
    launch_params_.print();
  }

  if (isDebugDumpEnabled(DebugDumpOption::KernelArgs)) {
    dumpKernelArgs(
        fusion_id_,
        group_id_,
        args,
        num_inputs,
        output_args,
        intermediate_args,
        executor_entry->intermediates);
  }

  if (isDebugDumpEnabled(DebugDumpOption::IndexType)) {
    debug() << "Index type: " << compiled_kernel_->kernel()->indexType()
            << std::endl;
  }

  if (execute_kernel_ && !compiled_kernel_->kernel()->topLevelExprs().empty()) {
    FUSER_PERF_SCOPE("KernelExecutor::runFusion::execute_kernel");
    ensureAvailableDynamicSmemSize(executor_entry->launch_params.smem());

    if (isDebugDumpEnabled(DebugDumpOption::Occupancy) ||
        isDebugDumpEnabled(DebugDumpOption::PerfDebugVerbose)) {
      int blocks_per_sm = -1;
      NVFUSER_CUDA_SAFE_CALL(cuOccupancyMaxActiveBlocksPerMultiprocessor(
          &blocks_per_sm,
          compiled_kernel_->cudaExecutable()->function,
          launch_params_.nThreads(),
          launch_params_.smem()));

      const int64_t device_id =
          static_cast<unsigned char>(compiled_kernel_->device().index());
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
    sprof.outputBytesAccessed(computeBytes(output_args));
  }

  return output_args;
}

flatbuffers::Offset<serde::KernelExecutor> KernelExecutor::serialize(
    flatbuffers::FlatBufferBuilder& builder) const {
  // See table definition for KernelExecutor in serde/fusion_cache.fbs
  using fb_executor_entry = flatbuffers::Offset<serde::KernelExecutorEntry>;

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
  if (!compiledKernel()->isCompiled()) {
    return serde::CreateKernelExecutorDirect(builder);
  }

  return serde::CreateKernelExecutorDirect(
      builder,
      device_smem_limit_,
      compiledKernel()->blockSizeHighWaterMark(),
      compiledKernel()->maxrregcountHighWaterMark(),
      warp_size_,
      toUnderlying(compiledKernel()->schedulerType()),
      fusion_id_,
      concrete_id_,
      runtime_id_,
      group_id_,
      compiledKernel()->kernelCode().c_str(),
      &executor_entry_lookup_keys_fb,
      &executor_entry_lookup_values_fb,
      toUnderlying(compiledKernel()->kernel()->indexType()),
      serialize(builder, compiledKernel()->cudaExecutable().get()),
      has_rng_,
      has_tma_,
      has_dynamic_alias_);
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

flatbuffers::Offset<serde::KernelExecutorEntry> KernelExecutor::serialize(
    flatbuffers::FlatBufferBuilder& builder,
    const KernelExecutorEntry& data) const {
  // See table definition for KernelExecutorEntry in serde/fusion_cache.fbs

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
    NVF_ERROR(
        tv_position != -1, "Output TensorView not found in kernel outputs");
    outputs_fb.push_back(serialize(
        builder,
        buffer,
        tv_position,
        true /* is_fusion_output */,
        false /* is_fusion_input */));
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
    NVF_ERROR(
        tv_position != -1,
        "Intermediate TensorView not found in kernel global allocations");
    intermediates_fb.push_back(serialize(
        builder,
        buffer,
        tv_position,
        false /* is_fusion_output */,
        false /* is_fusion_input */));
  }

  std::vector<fb_global_buffer_info> inputs_fb;
  inputs_fb.reserve(data.inputs.size());
  for (const auto& buffer : data.inputs) {
    auto tv_iter = std::find(
        compiledKernel()->kernel()->inputs().cbegin(),
        compiledKernel()->kernel()->inputs().cend(),
        buffer.tv);
    auto tv_position = (tv_iter == compiledKernel()->kernel()->inputs().cend())
        ? -1
        : std::distance(compiledKernel()->kernel()->inputs().cbegin(), tv_iter);
    NVF_ERROR(tv_position != -1, "Input TensorView not found in kernel inputs");
    inputs_fb.push_back(serialize(
        builder,
        buffer,
        tv_position,
        false /* is_fusion_output */,
        true /* is_fusion_input */));
  }
  return serde::CreateKernelExecutorEntryDirect(
      builder,
      data.init,
      data.launch_params.serialize(builder),
      &outputs_fb,
      &intermediates_fb,
      &inputs_fb,
      &data.output_aliased_to_input);
}

flatbuffers::Offset<serde::GlobalBufferInfo> KernelExecutor::serialize(
    flatbuffers::FlatBufferBuilder& builder,
    const GlobalBufferInfo& data,
    int64_t tv_position,
    bool is_fusion_output,
    bool is_fusion_input) const {
  // See table definition for GlobalBufferInfo in serde/fusion_cache.fbs
  return serde::CreateGlobalBufferInfoDirect(
      builder,
      tv_position,
      &data.shape_info.logical_sizes,
      &data.shape_info.logical_strides,
      &data.shape_info.unsharded_logical_sizes,
      &data.shape_info.allocation_sizes,
      &data.shape_info.allocation_strides,
      nvfuser::toUnderlying(data.type),
      data.zero_init,
      data.resets_to_zero,
      data.is_profile_buffer,
      is_fusion_output,
      is_fusion_input);
}

void KernelExecutor::deserialize(
    const serde::KernelExecutor* buffer,
    Fusion* _fusion,
    int8_t device_index,
    CompileParams compile_params,
    SchedulerType scheduler_type,
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
      "Expected given concrete_id to match serde concrete_id: ",
      concrete_id,
      " vs ",
      buffer->concrete_id());
  NVF_ERROR(
      runtime_id == buffer->runtime_id(),
      "Expected given runtime_id to match serde runtime_id.");
  NVF_ERROR(
      group_id == buffer->group_id(),
      "Expected given group_id to match serde group_id.");
  NVF_ERROR(
      toUnderlying(scheduler_type) == buffer->heuristic(),
      ": ",
      toUnderlying(scheduler_type),
      " vs ",
      buffer->heuristic());

  auto device = c10::Device(c10::DeviceType::CUDA, device_index);
  c10::DeviceGuard dg(device);

  // Initialize internal fields
  device_smem_limit_ = buffer->device_smem_limit();
  warp_size_ = buffer->warp_size();

  compiled_kernel_ = std::make_unique<CompiledKernel>(
      _fusion,
      compile_params,
      device,
      scheduler_type,
      fusion_id,
      concrete_id,
      runtime_id,
      group_id);

  compiled_kernel_->deserialize(buffer);

  // GlobalBufferInfo requires lowered kernel before deserialization
  for (auto idx : arange(buffer->executor_entry_lookup_keys()->size())) {
    executor_entry_lookup_.emplace(
        buffer->executor_entry_lookup_keys()->Get(idx),
        deserialize(buffer->executor_entry_lookup_values()->Get(idx)));
  }

  has_rng_ = buffer->has_rng();
  has_tma_ = buffer->has_tma();
  has_dynamic_alias_ = buffer->has_dynamic_alias();
}

KernelExecutorEntry KernelExecutor::deserialize(
    const serde::KernelExecutorEntry* buffer) {
  // See table definition for KernelExecutorEntry in serde/fusion_cache.fbs

  NVF_ERROR(buffer != nullptr, "serde::KernelExecutorEntry is nullptr.");

  KernelExecutorEntry entry;

  entry.init = buffer->init();

  entry.launch_params.deserialize(buffer->launch_params());

  for (auto output_buffer : *buffer->outputs()) {
    entry.outputs.push_back(deserialize(output_buffer));
  }

  for (auto intermediate_buffer : *buffer->intermediates()) {
    entry.intermediates.push_back(deserialize(intermediate_buffer));
  }

  for (auto input_buffer : *buffer->inputs()) {
    entry.inputs.push_back(deserialize(input_buffer));
  }

  for (auto output_aliased_to_input : *buffer->output_aliased_to_input()) {
    entry.output_aliased_to_input.push_back(output_aliased_to_input);
  }

  return entry;
}

GlobalBufferInfo KernelExecutor::deserialize(
    const serde::GlobalBufferInfo* buffer) {
  // See table definition for GlobalBufferInfo in serde/fusion_cache.fbs

  NVF_ERROR(buffer != nullptr, "serde::GlobalBufferInfo is nullptr.");

  NVF_ERROR(
      buffer->tv_pos() != -1,
      "Serialization failed to encode buffer tv position.");

  NVF_ERROR(
      compiled_kernel_->lowered() != nullptr,
      "Lowered kernel is not initialized.");

  GlobalBufferInfo info;
  if (buffer->is_fusion_output()) {
    auto out_val = compiled_kernel_->kernel()->outputs().at(buffer->tv_pos());
    NVF_ERROR(out_val != nullptr);
    info.tv = dynamic_cast<TensorView*>(out_val);
  } else if (buffer->is_fusion_input()) {
    auto in_val = compiled_kernel_->kernel()->inputs().at(buffer->tv_pos());
    NVF_ERROR(in_val != nullptr);
    info.tv = dynamic_cast<TensorView*>(in_val);
  } else {
    auto out_val = compiled_kernel_->kernel()->summary().global_allocations.at(
        buffer->tv_pos());
    NVF_ERROR(out_val != nullptr);
    info.tv = dynamic_cast<TensorView*>(out_val->buffer());
  }

  TensorShapeInfo shape_info;

  for (auto dim_size : *buffer->logical_sizes()) {
    shape_info.logical_sizes.emplace_back(dim_size);
  }

  for (auto dim_stride : *buffer->logical_strides()) {
    shape_info.logical_strides.emplace_back(dim_stride);
  }

  for (auto dim_size : *buffer->unsharded_logical_sizes()) {
    shape_info.unsharded_logical_sizes.emplace_back(dim_size);
  }

  for (auto dim_size : *buffer->alloc_sizes()) {
    shape_info.allocation_sizes.emplace_back(dim_size);
  }

  for (auto dim_stride : *buffer->alloc_strides()) {
    shape_info.allocation_strides.emplace_back(dim_stride);
  }

  info.shape_info = shape_info;

  info.type = serde::mapToAtenDtype(buffer->dtype());
  info.zero_init = buffer->zero_init();
  info.resets_to_zero = buffer->resets_to_zero();
  info.is_profile_buffer = buffer->is_profile_buffer();
  return info;
}

} // namespace nvfuser
