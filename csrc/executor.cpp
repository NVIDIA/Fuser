// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on

#include <executor.h>

#include <codegen.h>
#include <executor_kernel_arg.h>
#include <executor_utils.h>
#include <instrumentation.h>
#include <ir_all_nodes.h>
#include <ir_utils.h>
#include <iter_visitor.h>
#include <kernel_ir.h>
#include <lower_bank_conflict.h>
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
#include <fstream>

namespace nvfuser {

int FusionExecutor::fusion_id_counter_ = 0; // NOLINT

bool fill_allocation_with_nan_ = false;

bool shouldFillAllocationWithNan() {
  return fill_allocation_with_nan_;
}

void setFillAllocationWithNan(bool value) {
  fill_allocation_with_nan_ = value;
}

bool assert_out_of_bound_ = false;

bool shouldAssertOutOfBound() {
  return isDebugDumpEnabled(DebugDumpOption::AssertMemoryViolation) ||
      assert_out_of_bound_;
}

void setAssertOutOfBound(bool value) {
  assert_out_of_bound_ = value;
}

namespace {

static const char* defineIndexType(PrimDataType index_type) {
  if (index_type == DataType::Int32) {
    return "typedef int nvfuser_index_t;\n";
  } else if (index_type == DataType::Int) {
    return "typedef int64_t nvfuser_index_t;\n";
  } else {
    TORCH_INTERNAL_ASSERT(false, "invalid indexing type: ", index_type);
  }
}

static const char* defineIntegerTypes() {
  return R"(
typedef signed char int8_t;
typedef unsigned char uint8_t;
typedef short int int16_t;
typedef unsigned short int uint16_t;
typedef int int32_t;
typedef unsigned int uint32_t;
typedef long long int int64_t;
typedef unsigned long long int uint64_t;
)";
}

static const std::string& includeStdComplex() {
  static std::string result = std::string(R"ESCAPE(
#ifdef __NVCC__
#include <complex>
#endif // __NVCC__
)ESCAPE");
  return result;
}

} // namespace

std::string FusionExecutor::getStructuredCode(
    const std::string& kernel_str,
    PrimDataType index_type) {
  // generating cuda code;
  std::string code = "";
  if (shouldAssertOutOfBound()) {
    code += "#define ASSERT_OUT_OF_BOUND 1";
  }
  code += includeStdComplex();
  code += std::string("namespace ") + FusionExecutor::kernelNamespace() +
      " {\n" + defineIntegerTypes() + defineIndexType(index_type) +
      executor_utils::kernelPreamble() + kernel_str + "}\n";

  if (isDebugDumpEnabled(DebugDumpOption::CudaKernel)) {
    std::cout << "\n======= Codegen output for kernel: " << kernelName()
              << " =======\n\n"
              << kernel_str << "\n======================================\n\n";
  } else if (isDebugDumpEnabled(DebugDumpOption::CudaFull)) {
    std::cout << "\n======= Codegen output for kernel: " << kernelName()
              << " =======\n\n"
              << code << "\n======================================\n\n";
  }
  if (isDebugDumpEnabled(DebugDumpOption::CudaToFile) ||
      isDebugDumpEnabled(DebugDumpOption::DebugInfo)) {
    std::stringstream file_name;
    file_name << "__tmp_kernel" << fusion_id_ << ".cu";
    std::cout << "PRINTING: " << file_name.str() << std::endl;
    std::ofstream out(file_name.str());
    out << code << std::endl;
    out.close();
  }

  return code;
}

// TODO: come up with a more user friendly interface
void FusionExecutor::debugCompileFusionFromStr(
    Fusion* fusion,
    const std::string& code,
    const std::string& name,
    int id,
    CompileOptions options) {
  options_ = options;

  if (isDebugDumpEnabled(DebugDumpOption::FusionIr)) {
    fusion->print();
  } else if (isDebugDumpEnabled(DebugDumpOption::FusionIrMath)) {
    fusion->printMath();
  }

  if (isDebugDumpEnabled(DebugDumpOption::CudaFull)) {
    std::cout << "\n==== codegen output for kernel: " << kernelName()
              << " ====" << std::endl
              << code << std::endl
              << "======================================\n"
              << std::endl;
  }

  lowered_ = std::make_unique<GpuLower>(fusion);
  const auto kernel = lowered_->kernel();
  fusion_ = lowered_->kernel();

  fusion_id_ = id;
  setUsedTVs();

  if (isDebugDumpEnabled(DebugDumpOption::KernelIr)) {
    kernel->print();
  }

  const auto& kernel_summary = kernel->summary();

  if (!kernel_summary.static_smem_allocations.empty()) {
    ExpressionEvaluator static_evaluator;
    // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
    const auto static_smem_size = computeSharedMemory(
        static_evaluator, kernel_summary.static_smem_allocations);
    TORCH_INTERNAL_ASSERT(
        static_smem_size < max_static_smem_,
        "The static shared memory allocation is larger than available memory.");
  }

  std::tie(compiled_kernel_, last_compiler_log_, last_compiled_binary_) =
      executor_utils::getCompiledKernel(c10::nullopt, code, name, fusion_id_);
  TORCH_INTERNAL_ASSERT(
      fusion_id_ > 0, "assign a fusion_id_ <= 0 is not accepted.");
}

void FusionExecutor::compileFusion(
    Fusion* fusion,
    const KernelArgumentHolder& args,
    const LaunchParams& launch_constraints,
    CompileParams compile_params) {
  FUSER_PERF_SCOPE("compileFusion");

  TORCH_INTERNAL_ASSERT(
      !fusion->outputs().empty(), "No output found for this kernel, aborting.");

  for (auto out : fusion->outputs()) {
    TORCH_INTERNAL_ASSERT(
        out->getValType() == ValType::TensorView,
        "Output types from fusions that are not tensors are not supported at this point.");

    const auto maybe_rfactor_domain =
        out->as<TensorView>()->getMaybeRFactorDomain();
    // walking through outputs to see if output shapes are dependent on
    // non-tensor inputs. For which case, we should have disabled output
    // allocation, since the caching id only looks at tensor shapes.
    // See issue https://github.com/csarofeen/pytorch/issues/2002
    std::vector<Val*> output_extents;
    for (const auto id : maybe_rfactor_domain) {
      Val* extent = nullptr;
      if (id->isReduction() || id->isStride()) {
        continue;
      } else if (id->isBroadcast() && id->hasExpandedExtent()) {
        extent = id->expandedExtent();
      } else {
        extent = id->extent();
      }
      output_extents.emplace_back(extent);
    }
    auto dependencies = InputsOf::outputs(fusion, output_extents);
    if (std::any_of(dependencies.begin(), dependencies.end(), [](Val* val) {
          return val->isFusionInput();
        })) {
      // TODO: parameter cache is too big a hammer here. We should consider
      // separate the caching logic of output sizes & launch params. Since
      // output size dependency should only invalidate the output sizes
      disable_parameter_cache_ = true;
      break;
    }
  }

  if (isDebugDumpEnabled(DebugDumpOption::FusionIr)) {
    fusion->print();
  } else if (isDebugDumpEnabled(DebugDumpOption::FusionIrMath)) {
    fusion->printMath();
  }

  // TODO: refactor the options_ passed through
  options_.device =
      c10::Device(c10::DeviceType::CUDA, (int8_t)args.getDeviceIndex());

  // Set the index type of compile params if not already set. If set,
  // make sure the compile param type is valid with the given kernel
  // arguments.
  auto arg_index_type = args.getIndexType();
  if (compile_params.index_type.has_value()) {
    // If the int32 compilation is requested, but the arguments demand
    // int64, that's an error
    TORCH_INTERNAL_ASSERT(
        !(compile_params.index_type.value() == DataType::Int32 &&
          arg_index_type == DataType::Int),
        "Compilation with int32 is requested but int64 is required for the arguments");
  } else {
    // If the given compile option doesn't specify the index type, use
    // the type determined by the arguments
    compile_params.index_type = arg_index_type;
  }

  c10::DeviceGuard dg(options_.device);

  TORCH_INTERNAL_ASSERT(
      options_.device.is_cuda(), "Provided device to CUDA fuser is the CPU.");
  auto properties = at::cuda::getDeviceProperties(options_.device.index());
  configured_device_smem_ = properties->sharedMemPerBlock;
  device_smem_limit_ = properties->sharedMemPerBlockOptin;
  warp_size_ = properties->warpSize;

  lowered_ = std::make_unique<GpuLower>(fusion, compile_params);

  const auto kernel = lowered_->kernel();
  fusion_ = lowered_->kernel()->as<Fusion>();

  fusion_id_ = ++fusion_id_counter_;
  setUsedTVs();

  if (isDebugDumpEnabled(DebugDumpOption::KernelIr)) {
    kernel->print();
  }

  if (isDebugDumpEnabled(DebugDumpOption::BankConflictInfo)) {
    auto bank_conflict_info = getBankConflictInfo(kernel);
    if (bank_conflict_info.empty()) {
      std::cout << "===== No bank confliction =====" << std::endl;
    } else {
      std::cout << "======= Bank confliction =======" << std::endl;
      for (auto info : bank_conflict_info) {
        std::cout << "Expr: " << info.first->toString() << std::endl;
        auto conflict = info.second;
        if (conflict.first > 1) {
          std::cout << "input conflict: " << conflict.first << " way, ";
        }
        if (conflict.second > 1) {
          std::cout << "output conflict: " << conflict.second << " way";
        }
        std::cout << std::endl;
      }
      std::cout << "================================" << std::endl;
    }
  }

  kernel_code_ = codegen::generateCudaKernel(kernel, kernelName());

  auto load_external_code = [](const char* external_code_path) {
    std::cout << "--------> Compiling external cuda code: "
              << external_code_path << std::endl;
    std::ifstream cuda_src(external_code_path);
    std::stringstream buffer;
    buffer << cuda_src.rdbuf();
    return buffer.str();
  };
  auto external_code_path = std::getenv("PYTORCH_NVFUSER_EXTERNAL_SRC");
  const auto structured_code = external_code_path
      ? load_external_code(external_code_path)
      : getStructuredCode(kernel_code_, kernel->indexType());

  const auto& kernel_summary = kernel->summary();

  // We currently shouldn't allocate any more shared mem
  //  tensors statically but could keep this path if
  //  needed in later development.
  if (!kernel_summary.static_smem_allocations.empty()) {
    ExpressionEvaluator static_evaluator;
    // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
    const auto static_smem_size = computeSharedMemory(
        static_evaluator, kernel_summary.static_smem_allocations);
    TORCH_INTERNAL_ASSERT(
        static_smem_size < max_static_smem_,
        "The static shared memory allocation is larger than available memory.");
  }

  if (kernel_summary.has_dynamic_local_memory_allocations) {
    std::stringstream ss;
    ss << "Allocations must be based on constant integers for local memory. However, found: ";
    for (auto alloc : kernel_summary.dynamic_lmem_allocations) {
      ss << alloc->buffer()->toString() << ", ";
    }
    ss << " have dynamic allocations but are placed in local memory.";
    TORCH_INTERNAL_ASSERT(false, ss.str());
  }

  // TODO: pass block_size here;
  c10::optional<int> block_size = c10::nullopt;
  if (!args.empty()) {
    auto expr_eval = executor_utils::bindInputs(args, kernel);
    auto launch_params =
        computeLaunchParams(launch_constraints, expr_eval, warp_size_);
    block_size = launch_params.nThreads();
    TORCH_INTERNAL_ASSERT(
        block_size > 0, "launch param inferred block size < 0");
  }

  // TODO: high water mark should be computed via occupancy API after
  // compilation.

  // Basically setting high water martk as 1 when we don't provide args for
  // compilation, it will just generate a kernel that gets ditched at the first
  // run - not great. We should have better heuristics.
  block_size_high_water_mark = std::max<int64_t>(
      (block_size.has_value() ? block_size.value() : 1),
      block_size_high_water_mark);
  maxrregcount_high_water_mark = compile_params.maxrregcount;
  std::tie(compiled_kernel_, last_compiler_log_, last_compiled_binary_) =
      executor_utils::getCompiledKernel(
          kernel_code_,
          structured_code,
          (kernelNamespace() + "::" + kernelName()).c_str(),
          fusion_id_,
          block_size,
          maxrregcount_high_water_mark,
          save_compiled_binary_ || isDebugDumpEnabled(DebugDumpOption::Sass));
  TORCH_INTERNAL_ASSERT(
      fusion_id_ > 0, "failed to assign a fusion_id_ after compilation.");

  // The driver API call requires an int argument.
  int max_dynamic_smem = 0;
  CUDA_SAFE_CALL(cuFuncGetAttribute(
      &max_dynamic_smem,
      CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
      compiled_kernel_.function));
  maybe_available_dynamic_smem_ = max_dynamic_smem;

  if (isDebugDumpEnabled(DebugDumpOption::Sass)) {
    std::cout << disassembledKernelSASS() << std::endl;
  }
}

namespace {

void fillTensorWithNan(at::Tensor& t) {
  switch (t.scalar_type()) {
    case at::ScalarType::Byte:
      t.fill_(0xFF);
      break;
    case at::ScalarType::Char:
      t.fill_(0x7F);
      break;
    case at::ScalarType::Short:
      t.fill_(0x7FFF);
      break;
    case at::ScalarType::Int:
      t.fill_(0x7FFFFFFF);
      break;
    case at::ScalarType::Long:
      t.fill_(0x7FFFFFFFFFFFFFFFL);
      break;
    case at::ScalarType::Bool:
      t.fill_(true);
      break;
    case at::ScalarType::Half:
    case at::ScalarType::Float:
    case at::ScalarType::Double:
    case at::ScalarType::BFloat16:
      t.fill_(std::nan(""));
      break;
    case at::ScalarType::ComplexHalf:
    case at::ScalarType::ComplexFloat:
    case at::ScalarType::ComplexDouble:
      t.fill_(c10::complex<double>(std::nan(""), std::nan("")));
      break;
    default:
      TORCH_INTERNAL_ASSERT(false, "Unknown dtype");
  }
}

at::Tensor inferAndAlloc(
    const TensorView* tv,
    const std::vector<Val*>& sizes,
    ExpressionEvaluator& expr_eval,
    // Map from dim -> expanded size of TV if any expanded broadcast dimensions
    // exist
    std::unordered_map<int, Val*> expanded_map,
    const CompileOptions& options,
    bool zero_init = false) {
  FUSER_PERF_SCOPE("inferAndAlloc");

  // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
  // Going to infer all the sizes of the TensorView
  std::vector<int64_t> inferred_sizes;
  // Expanded sizes is at maximum the same size of inferred_sizes, as you could
  // have a fully broadcasted tensor that's being expanded
  std::vector<int64_t> expanded_sizes;
  bool expanded_dim = false;
  for (const auto size : sizes) {
    const auto inferred_val = expr_eval.evaluate(size);
    TORCH_INTERNAL_ASSERT(
        inferred_val.has_value(),
        "Could not launch kernel as program could not infer ",
        size->toString(),
        "(",
        size->name(),
        ") for the buffer ",
        tv->toString());
    inferred_sizes.push_back(inferred_val->as<int64_t>());
    if (expanded_map.count((int)expanded_sizes.size())) {
      auto expanded_size = expanded_map.at((int)expanded_sizes.size());
      const auto inferred_expanded_size = expr_eval.evaluate(expanded_size);
      TORCH_INTERNAL_ASSERT(
          inferred_expanded_size.has_value(),
          "Could not launch kernel as program could not infer the expanded extent ",
          expanded_size->toString(),
          "(",
          expanded_size->name(),
          ") for the buffer ",
          tv->toString());
      if (inferred_val.value() != 1) {
        TORCH_INTERNAL_ASSERT(
            inferred_val.value() == inferred_expanded_size.value(),
            "Attempted an expand on a non-broadcasted dimension,",
            " but the expand doesn't match the dimensions size.");
      } else {
        expanded_dim = true;
      }
      expanded_sizes.push_back(inferred_expanded_size->as<int64_t>());
    } else {
      expanded_sizes.push_back(inferred_val->as<int64_t>());
    }
  }

  const auto at_type = data_type_to_aten(tv->dtype());
  const auto tensor_options =
      at::TensorOptions().dtype(at_type).device(options.device);
  c10::IntArrayRef isizes(inferred_sizes);

  if (zero_init) {
    auto zeros = at::zeros(isizes, tensor_options);
    if (expanded_dim) {
      return zeros.expand(expanded_sizes);
    }
    return zeros;
  } else {
    // Non Variable type guard for empty_cuda call
    at::AutoDispatchBelowADInplaceOrView non_variable_type_mode;
    auto empty = at::empty(isizes, tensor_options);
    if (shouldFillAllocationWithNan()) {
      fillTensorWithNan(empty);
    }
    if (expanded_dim) {
      return empty.expand(expanded_sizes);
    }
    return empty;
  }
}

at::Tensor inferAndAllocOutput(
    const TensorView* tv,
    ExpressionEvaluator& expr_eval,
    const CompileOptions& options,
    bool zero_init = false) {
  const auto domain = tv->domain();
  const auto maybe_rfactor_domain = domain->hasRFactor()
      ? domain->getRFactorDomain()
      : domain->getRootDomain();

  std::vector<Val*> sizes;
  std::unordered_map<int, Val*> expand_map;

  for (const auto id : maybe_rfactor_domain) {
    if (id->isReduction() || id->isStride()) {
      continue;
    }
    sizes.push_back(id->extent());
    if (id->isBroadcast() && id->hasExpandedExtent()) {
      expand_map[(int)sizes.size() - 1] = id->expandedExtent();
    }
  }
  return inferAndAlloc(tv, sizes, expr_eval, expand_map, options, zero_init);
}

} // namespace

uint64_t FusionExecutor::computeSharedMemory(
    ExpressionEvaluator& expr_eval,
    const std::vector<const kir::Allocate*>& buffers,
    bool align_padding,
    uint64_t total) {
  FUSER_PERF_SCOPE("computeSharedMemory");
  for (auto smem_alloc : buffers) {
    // If this buffer aliases another buffer,
    // then do not allocate memory for this buffer.
    if (smem_alloc->alias() == nullptr) {
      const auto inferred_val = expr_eval.evaluate(smem_alloc->size());
      if (inferred_val.has_value()) {
        const uint64_t data_size = dataTypeSize(smem_alloc->buffer()->dtype());
        // Add padding to align dynamic shared memory
        if (align_padding) {
          const int align_size = 16; // always align to 16B/128b.
          total = ceilDiv((int64_t)total, align_size) * align_size;
        }
        total += inferred_val->as<int64_t>() * data_size;
      } else {
        TORCH_INTERNAL_ASSERT(
            false,
            "Failed to evaluate the size ",
            smem_alloc->size(),
            " of shared memory buffer - T",
            smem_alloc->buffer()->name());
      }
    }
  }
  return total;
}

LaunchParams FusionExecutor::computeLaunchParams(
    const LaunchParams& launch_constraints,
    ExpressionEvaluator& expr_eval,
    const int warp_size) {
  FUSER_PERF_SCOPE("FusionExecutor::ComputeLaunchParams");
  TORCH_INTERNAL_ASSERT(warp_size > 0, "WARP_SIZE should be larger than 0");

  LaunchParams launch_params;

  auto data_cache = compileTimeDataCache();

  auto lower = lowered_.get();
  auto& used_tvs = getUsedTVs();
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
        if (inferred_val.has_value()) {
          // This value could have been inferred, make sure it was set right.
          bool valid = inferred_val->as<int64_t>() ==
                  launch_constraints.getDim(p_type) ||
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
    FUSER_PERF_SCOPE("FusionExecutor::ParallelBindingResolution");
    auto val = expr_eval.evaluate(extent);
    TORCH_INTERNAL_ASSERT(
        val.has_value(),
        "Tried to evaluate the extent, ",
        extent->toInlineString(),
        " for the ptype: ",
        p_type,
        " to set launch bounds but could not.");

    if (val->as<int64_t>() > 0) {
      expr_eval.bind(p_type, val->as<int64_t>());
      launch_params.bind(val->as<int64_t>(), p_type);
    }
  }

  // Re-run the integer machine with all
  //  the thread sizes now determined.
  if (expr_eval.precomputedValues()) {
    expr_eval.precomputedValues()->evaluate();
  }

  const auto kernel = lowered_->kernel();
  const auto& kernel_summary = kernel->summary();

  // Calculate Dynamic Shared Memory Size
  // Add workspace for reduction and broadcast
  uint64_t reduction_broadcast_workspace = 0;
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
    reduction_broadcast_workspace =
        dataTypeSize(kernel_summary.largest_smem_data_type) * welford_factor *
        launch_params.bdimx() * launch_params.bdimy() * launch_params.bdimz();

    if (kernel_summary.has_outer_grouped_grid_welford) {
      reduction_broadcast_workspace = std::max(
          reduction_broadcast_workspace,
          (uint64_t)
              kernel_summary.outer_grouped_grid_welford_largest_smem_size);
    }
  }

  // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
  const uint64_t dynamic_smem_size = computeSharedMemory(
      expr_eval,
      kernel_summary.dynamic_smem_allocations,
      true,
      reduction_broadcast_workspace);

  // Check that requested smem size can be dynamically allocated.
  //  This check is only done once a kernel has been compiled, since
  //  maybe_available_dynamic_smem_ needs to be evaluated on
  //  a compiled kernel.
  if (maybe_available_dynamic_smem_.has_value()) {
    // Dynamic shared memory space that we can allocate without
    //  carving more space from L1.
    const uint64_t available_dynamic_smem_without_reconfiguration =
        maybe_available_dynamic_smem_.value();
    // Maximum additional shared memory size we could request
    //  if we do re-configuration.
    const uint64_t additional_dynamic_smem_available_through_reconfiguration =
        device_smem_limit_ - configured_device_smem_;

    TORCH_INTERNAL_ASSERT(
        (dynamic_smem_size) <
            (available_dynamic_smem_without_reconfiguration +
             additional_dynamic_smem_available_through_reconfiguration),
        "The total shared memory allocation is larger than available memory.",
        " Dynamic size: ",
        dynamic_smem_size,
        ". Available size: ",
        maybe_available_dynamic_smem_.value(),
        ". Configured smem size: ",
        configured_device_smem_,
        ". Device limit size: ",
        device_smem_limit_);
  }

  launch_params.setSmem((int64_t)dynamic_smem_size);

  return launch_params;
}

FusionExecutor::GlobalBuffers FusionExecutor::allocGlobalVals(
    ExpressionEvaluator& expr_eval) {
  FUSER_PERF_SCOPE("FusionExecutor::AllocGlobalVals");
  GlobalBuffers global_buffers;
  const auto kernel = lowered_->kernel();
  const auto& kernel_summary = kernel->summary();
  for (auto alloc : kernel_summary.global_allocations) {
    TORCH_INTERNAL_ASSERT(
        alloc->buffer()->isA<TensorView>(),
        "Cannot allocate global buffers that are not tensors.");
    auto tv = alloc->buffer()->as<TensorView>();
    if (tv->isFusionOutput()) {
      continue;
    }
    if (alloc->zeroInit()) {
      global_buffers.buffers.push_back(
          inferAndAlloc(tv, alloc->shape(), expr_eval, {}, options_, true));
      global_buffers.zero_init.push_back(true);
    } else {
      global_buffers.buffers.push_back(
          inferAndAlloc(tv, alloc->shape(), expr_eval, {}, options_, false));
      global_buffers.zero_init.push_back(false);
    }
    // Remember the tensor buffer used for storing kernel profile
    if (isOptionEnabled(EnableOption::KernelProfile) &&
        tv == kernel->profile().getBuffer()) {
      global_buffers.profile_buffer = global_buffers.buffers.back();
    }
  }

  return global_buffers;
}

std::vector<at::Tensor> FusionExecutor::allocOutputSpace(
    const at::ArrayRef<c10::IValue>& inputs) {
  auto kernel_inputs = KernelArgumentHolder::createKernelArgumentHolder(inputs);
  auto expr_eval =
      executor_utils::bindInputs(kernel_inputs, lowered_->kernel());
  auto outputs = allocOutputs(kernel_inputs, expr_eval);
  return outputs;
}

std::vector<at::Tensor> FusionExecutor::allocOutputs(
    const KernelArgumentHolder& args,
    ExpressionEvaluator& expr_eval,
    const std::unordered_set<int>& alias_indices) {
  FUSER_PERF_SCOPE("FusionExecutor::AllocOutputs");
  const auto kernel = lowered_->kernel();
  // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
  std::vector<at::Tensor> outputs;
  TORCH_INTERNAL_ASSERT(
      args.size() == kernel->inputs().size(),
      "kernel arguments length does not match runtime arguments.");
  for (const auto out_i : c10::irange(kernel->outputs().size())) {
    if (kernel->outputs()[out_i]->isFusionInput()) {
      // pushing empty tensor for trivial forwarding. Since we handle this in
      // integration, see step 1 - note [trivial forwarding]
      c10::Device device(c10::DeviceType::CUDA, (int8_t)args.getDeviceIndex());
      const auto tensor_options =
          at::TensorOptions().dtype(at::kFloat).device(device);
      outputs.emplace_back(at::empty({0}, tensor_options));
    } else {
      TORCH_INTERNAL_ASSERT(
          kernel->outputs()[out_i]->isA<TensorView>(),
          "Cannot allocate outputs that are not tensors.");
      auto output = kernel->outputs()[out_i]->as<TensorView>();
      if (alias_indices.count((int)out_i) != 0) {
        // aliasing to inputs, no need to allocate real output, just push empty
        // tensor here.
        outputs.emplace_back();
      } else {
        outputs.push_back(
            inferAndAllocOutput(output, expr_eval, options_, false));
      }
    }
  }
  return outputs;
}

void FusionExecutor::setUsedTVs() {
  auto used_vals = fusion_->usedMathVals();
  auto used_tvs = ir_utils::filterByType<TensorView>(used_vals);
  used_tvs_.clear();
  used_tvs_.insert(used_tvs_.begin(), used_tvs.begin(), used_tvs.end());
}

KernelArgumentHolder FusionExecutor::evaluateOutputSizes(
    const KernelArgumentHolder& args,
    ExpressionEvaluator& expr_eval,
    const std::unordered_set<int>& alias_indices) {
  FUSER_PERF_SCOPE("FusionExecutor::AllocOutputs");
  const auto kernel = lowered_->kernel();

  KernelArgumentHolder ret(args.getIndexMode());
  ret.setDeviceIndex(args.getDeviceIndex());

  CompileOptions meta_options = options_;
  meta_options.device = c10::Device(c10::DeviceType::Meta, 0);

  for (const auto out_i : c10::irange(kernel->outputs().size())) {
    // If the output is just trivially the input, just "copy" it over, see note
    // [trivial forwarding]
    if (kernel->outputs()[out_i]->isFusionInput()) {
      for (auto inp_i : c10::irange(kernel->inputs().size())) {
        if (kernel->inputs()[inp_i] == kernel->outputs()[out_i]) {
          TORCH_INTERNAL_ASSERT(
              inp_i < args.size(),
              "Issue with an input showing up as output, couldn't find input.");

          auto tensor_arg_abstract =
              dynamic_cast<const TensorArgAbstract*>(args[inp_i]);
          TORCH_INTERNAL_ASSERT(
              tensor_arg_abstract,
              "Cannot register a scalar as an output in a fusion.");
          ret.push(tensor_arg_abstract);
          break;
        }
      }
    } else {
      TORCH_INTERNAL_ASSERT(
          kernel->outputs()[out_i]->isA<TensorView>(),
          "Cannot allocate outputs that are not tensors.");
      auto output = kernel->outputs()[out_i]->as<TensorView>();
      if (alias_indices.count((int)out_i) != 0) {
        // aliasing to inputs, no need to allocate real output
        // but we still need to push an entry here.
        ret.push(int64_t(0));
      } else {
        // TODO: we are using meta here, which is bad since it doesn't account
        // for devices. Switch to fake tensor instead
        ret.push(inferAndAllocOutput(output, expr_eval, meta_options, false));
      }
    }
  }
  return ret;
}

KernelArgumentHolder FusionExecutor::inferOutputSizes(
    const KernelArgumentHolder& args,
    const LaunchParams& launch_constraints) {
  FUSER_PERF_SCOPE("FusionExecutor::RunFusion");

  ExecutorEntry* executor_entry = nullptr;
  c10::optional<size_t> opt_code = args.getCacheId();
  if (opt_code.has_value()) {
    executor_entry = &executor_entry_lookup_[*opt_code];
  }

  at::cuda::jit::initializeCudaContext();
  TORCH_INTERNAL_ASSERT(lowered_);

  TORCH_INTERNAL_ASSERT(
      !executor_entry || !executor_entry->init,
      "compile kernel shouldn't hit a pre-existing cache");
  FUSER_PERF_SCOPE("ExecutorRunFusion::ValidateAndInitialize");
  // TODO: validate kernel inputs currently won't be happy, since our fusion
  // args are mapped with `meta` tensor instead of `cuda` tensor, check if this
  // would be resolved with FakeTensor
  // executor_utils::validateKernelInputs(fusion_, args, options_.device);

  if (!evaluator_precomputed_values_) {
    evaluator_precomputed_values_ =
        std::make_unique<PrecomputedValues>(lowered_->kernel());
  }

  ExpressionEvaluator expr_eval;
  evaluator_precomputed_values_->bindInputs(args);
  expr_eval.precomputedValues() = evaluator_precomputed_values_.get();

  // I think this binds something to expr_eval, so even though we are not using
  // launch_params_, we still need this in order to infer output shapes.
  launch_params_ =
      computeLaunchParams(launch_constraints, expr_eval, warp_size_);

  executor_utils::validateVectorizedTensors(
      lowered_.get()->kernel(), args, {}, compileTimeDataCache(), expr_eval);

  auto alias_indices_entry = executor_utils::caching::ExecutorCompileTimeEntry<
      executor_utils::caching::InputAliasIndices>(
      compileTimeDataCache(), [&]() {
        return std::make_unique<std::vector<std::pair<int, int>>>(
            fusion_->getInputAliasIndices());
      });

  auto& alias_indices = alias_indices_entry.get();

  // NOLINTNEXTLINE(bugprone-branch-clone)
  auto output_alias_indices_entry =
      executor_utils::caching::ExecutorCompileTimeEntry<
          executor_utils::caching::OutputAliasIndices>(
          compileTimeDataCache(), [&]() {
            return std::make_unique<std::unordered_set<int>>(
                fusion_->getOutputAliasIndices());
          });

  auto& output_alias_indices = output_alias_indices_entry.get();

  auto ret = evaluateOutputSizes(args, expr_eval, output_alias_indices);

  for (const auto& entry : alias_indices) {
    auto aliased_output_index = entry.first;
    auto aliased_input_index = entry.second;
    TORCH_INTERNAL_ASSERT(
        args[aliased_input_index]->isType(ArgType::Tensor),
        "alias io only supports tensor");
    ret.swap(aliased_output_index, args[aliased_input_index]);
  }

  return ret;
}

namespace {

// Make sure the index type of Kernel is valid
// TODO: Check the size of all tensors, not just inputs.
void validateIndexType(
    kir::Kernel* kernel,
    KernelArgumentHolder& args,
    const CompileParams& compile_params) {
  // Currently, once a Fusion is lowered to a Kernel, the index type
  // has to be resolved completely. This means that
  // args.getIndexType() must be equal to the index type of the
  // compiled kernel.
  TORCH_INTERNAL_ASSERT(
      kernel->indexType() == args.getIndexType(),
      "Invalid pair of kernel index type and argument index type. Kernel type: ",
      kernel->indexType(),
      ". Argument index type: ",
      args.getIndexType());

  // Similarly, if the type of the index type in the given compile
  // parameters doesn't match, that's also an error.
  TORCH_INTERNAL_ASSERT(
      !compile_params.index_type.has_value() ||
          kernel->indexType() == compile_params.index_type.value(),
      "Kernel index type and compilation index type don't match. Kernel type: ",
      kernel->indexType(),
      ". Compilation index type: ",
      compile_params.index_type.value());
}

} // namespace

std::vector<at::Tensor> FusionExecutor::runFusion(
    KernelArgumentHolder& args,
    const LaunchParams& launch_constraints,
    CompileParams compile_params,
    const std::vector<at::Tensor>& outputs) {
  FUSER_PERF_SCOPE("FusionExecutor::RunFusion");
  TORCH_INTERNAL_ASSERT(compiled());
  TORCH_INTERNAL_ASSERT(
      fusion_id_ > 0, "Cannot run fusion, it was not compiled.");
  TORCH_INTERNAL_ASSERT(
      !args.getCacheId().has_value() || outputs.empty(),
      "short cut input cache is not compatible with pre-allocated output");

  validateIndexType(kernel(), args, compile_params);

  size_t num_inputs = args.size();

  if (isDebugDumpEnabled(DebugDumpOption::FusionArgs)) {
    std::cout << "Arguments for fusion" << fusion_id_ << ":" << std::endl
              << "Inputs:" << std::endl;
    for (auto i : c10::irange(args.size())) {
      std::cout << "  " << args[i]->toString() << std::endl;
    }
    std::cout << "Outputs:" << std::endl;
    for (const auto& output : outputs) {
      std::cout << "  " << output.scalar_type() << " " << output.sizes()
                << " (strides = " << output.strides() << ")" << std::endl;
    }
    std::cout << launch_constraints.toString();
    std::cout << "maxrregcount= " << compile_params.maxrregcount << std::endl;
  }

  ExecutorEntry* executor_entry = nullptr;
  if (args.getCacheId().has_value()) {
    executor_entry = &executor_entry_lookup_[*args.getCacheId()];
  }

  c10::DeviceGuard dg(options_.device);
  auto stream = at::cuda::getCurrentCUDAStream();
  at::cuda::jit::initializeCudaContext();
  TORCH_INTERNAL_ASSERT(lowered_);
  launch_params_ = LaunchParams();
  // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
  std::vector<at::Tensor> allocated_outputs;
  GlobalBuffers global_buffers;
  uint64_t rand_offset = 0;

  if (executor_entry && executor_entry->init && !disable_parameter_cache_) {
    {
      // context manager to disable auto grad for `empty_cuda` calls later
      at::AutoDispatchBelowADInplaceOrView non_variable_type_mode;
      // take the short-cut for launch if we see a recorded input set again
      launch_params_ = executor_entry->launch_params;
      // only allocate outputs when not given
      if (outputs.empty()) {
        FUSER_PERF_SCOPE("ExecutorRunFusion::OutputAlloc");
        for (const auto i : c10::irange(executor_entry->output_sizes.size())) {
          allocated_outputs.push_back(at::native::empty_strided_cuda(
              executor_entry->output_sizes[i],
              executor_entry->output_strides[i],
              executor_entry->output_types[i],
              c10::nullopt,
              options_.device,
              c10::nullopt));
          if (shouldFillAllocationWithNan()) {
            fillTensorWithNan(allocated_outputs.back());
          }
        }
        // Note: aliased output is not returned as output. But we still need it
        // for kernel execution, so would need to push them to args
        for (const auto& entry : executor_entry->io_alias_indices) {
          auto aliased_output_index = entry.first;
          auto aliased_input_index = entry.second;
          auto tensor_arg_abstract =
              dynamic_cast<const TensorArgAbstract*>(args[aliased_input_index]);
          TORCH_INTERNAL_ASSERT(
              tensor_arg_abstract, "alias io only supports tensor");
          allocated_outputs[aliased_output_index] =
              tensor_arg_abstract->getTensor();
        }
        args.push(allocated_outputs);
      } else {
        TORCH_INTERNAL_ASSERT(
            outputs.size() == fusion_->outputs().size(),
            __func__,
            " provided number of outputs does match fusion output");
        allocated_outputs = outputs;
        args.push(outputs);
      }

      {
        FUSER_PERF_SCOPE("ExecutorRunFusion::IntermediateBufferAlloc");
        for (const auto i : c10::irange(executor_entry->buffer_sizes.size())) {
          if (executor_entry->buffer_zero_init[i]) {
            global_buffers.buffers.push_back(at::zeros(
                executor_entry->buffer_sizes[i],
                at::TensorOptions()
                    .dtype(executor_entry->buffer_types[i])
                    .device(options_.device)));
            global_buffers.zero_init.push_back(true);
          } else {
            global_buffers.buffers.push_back(at::native::empty_cuda(
                executor_entry->buffer_sizes[i],
                executor_entry->buffer_types[i],
                c10::nullopt,
                options_.device,
                c10::nullopt));
            if (shouldFillAllocationWithNan()) {
              fillTensorWithNan(global_buffers.buffers.back());
            }
            global_buffers.zero_init.push_back(false);
          }
        }
      }
    }
    rand_offset = executor_entry->rand_offset;
  } else {
    FUSER_PERF_SCOPE("ExecutorRunFusion::ValidateAndInitialize");
    // code path to take when either:
    //   1. no opt_code is provided or
    //   2. `executor_entry` is not initialized
    executor_utils::validateKernelInputs(fusion_, args, options_.device);

    if (!evaluator_precomputed_values_) {
      evaluator_precomputed_values_ =
          std::make_unique<PrecomputedValues>(lowered_->kernel());
    }

    ExpressionEvaluator expr_eval;
    evaluator_precomputed_values_->bindInputs(args);
    expr_eval.precomputedValues() = evaluator_precomputed_values_.get();

    launch_params_ =
        computeLaunchParams(launch_constraints, expr_eval, warp_size_);

    // Recompile the kernel if the number of threads in the block has increased
    // or maxrregcount has changed
    if (launch_params_.nThreads() > block_size_high_water_mark ||
        compile_params.maxrregcount != maxrregcount_high_water_mark) {
      const auto kernel = lowered_->kernel();
      kernel_code_ = codegen::generateCudaKernel(kernel, kernelName());
      const auto structured_code =
          getStructuredCode(kernel_code_, kernel->indexType());
      block_size_high_water_mark = launch_params_.nThreads();
      maxrregcount_high_water_mark = compile_params.maxrregcount;

      std::tie(compiled_kernel_, last_compiler_log_, last_compiled_binary_) =
          executor_utils::getCompiledKernel(
              kernel_code_,
              structured_code,
              (kernelNamespace() + "::" + kernelName()).c_str(),
              fusion_id_,
              block_size_high_water_mark,
              maxrregcount_high_water_mark,
              save_compiled_binary_);
    }

    if (kernel()->summary().has_cooperative_grid_reduction) {
      int num_blocks_per_SM = -1;
      CUDA_SAFE_CALL(cuOccupancyMaxActiveBlocksPerMultiprocessor(
          &num_blocks_per_SM,
          compiled_kernel_.function,
          (int)(launch_params_.bdimx() * launch_params_.bdimy() * launch_params_.bdimz()),
          (size_t)launch_params_.smem()));

      TORCH_INTERNAL_ASSERT(
          (int64_t)(
              num_blocks_per_SM *
              at::cuda::getDeviceProperties(options_.device.index())
                  ->multiProcessorCount) >= launch_params_.gdimx() *
                  launch_params_.gdimy() * launch_params_.gdimz(),
          "Wanted to launch a cooperative kernel, however the number of blocks is greater than ",
          "what can be resident on the GPU at once. Need: ",
          launch_params_.gdimx() * launch_params_.gdimy() *
              launch_params_.gdimz(),
          " (",
          launch_params_.gdimx(),
          " * ",
          launch_params_.gdimy(),
          " * ",
          launch_params_.gdimz(),
          ") but limited to ",
          num_blocks_per_SM,
          " * ",
          at::cuda::getDeviceProperties(options_.device.index())
              ->multiProcessorCount);
    }

    executor_utils::validateVectorizedTensors(
        lowered_.get()->kernel(),
        args,
        outputs,
        compileTimeDataCache(),
        expr_eval);

    auto alias_indices_entry =
        executor_utils::caching::ExecutorCompileTimeEntry<
            executor_utils::caching::InputAliasIndices>(
            compileTimeDataCache(), [&]() {
              return std::make_unique<std::vector<std::pair<int, int>>>(
                  fusion_->getInputAliasIndices());
            });

    auto& alias_indices = alias_indices_entry.get();

    // NOLINTNEXTLINE(bugprone-branch-clone)
    if (outputs.empty()) {
      auto output_alias_indices_entry =
          executor_utils::caching::ExecutorCompileTimeEntry<
              executor_utils::caching::OutputAliasIndices>(
              compileTimeDataCache(), [&]() {
                return std::make_unique<std::unordered_set<int>>(
                    fusion_->getOutputAliasIndices());
              });

      auto& output_alias_indices = output_alias_indices_entry.get();

      allocated_outputs = allocOutputs(args, expr_eval, output_alias_indices);

      for (const auto& entry : alias_indices) {
        auto aliased_output_index = entry.first;
        auto aliased_input_index = entry.second;
        auto tensor_arg_abstract =
            dynamic_cast<const TensorArgAbstract*>(args[aliased_input_index]);
        TORCH_INTERNAL_ASSERT(
            tensor_arg_abstract, "alias io only supports tensor");
        allocated_outputs[aliased_output_index] =
            tensor_arg_abstract->getTensor();
      }
      args.push(allocated_outputs);
    } else {
      allocated_outputs = outputs;
      args.push(outputs);
      executor_utils::validateKernelOutputs(
          fusion_, allocated_outputs, options_.device);
    }

    global_buffers = allocGlobalVals(expr_eval);

    if (kernel()->summary().max_rng_offsets >= 0) {
      // NOTE: this is how we map offset to PW kernels in order to have
      // identical random number generator to match native PyTorch results.
      // But it doesn't really work as it takes assumption how threads are
      // binded but is not generally how we handle that in scheduler.
      // Refer to `Philox` in generated kernel to understand how the mapping
      // works.
      rand_offset = (uint64_t)(kernel()->summary().max_rng_offsets + 1) * 4;
    }

    // This is the entry when we have provided `opt_code` but the entry has not
    // been initialized yet.
    if (executor_entry) {
      FUSER_PERF_SCOPE("ExecutorRunFusion::FillCacheEntry");
      // record the the short-cut executor entry for the given input set;
      executor_entry->launch_params = launch_params_;
      executor_entry->io_alias_indices = alias_indices;
      for (const auto& output : allocated_outputs) {
        executor_entry->output_sizes.push_back(output.sizes().vec());
        executor_entry->output_strides.push_back(output.strides().vec());
        executor_entry->output_types.push_back(output.scalar_type());
      }

      for (const auto& i : c10::irange(global_buffers.buffers.size())) {
        executor_entry->buffer_sizes.push_back(
            global_buffers.buffers[i].sizes().vec());
        executor_entry->buffer_types.push_back(
            global_buffers.buffers[i].scalar_type());
        executor_entry->buffer_zero_init.push_back(global_buffers.zero_init[i]);
      }
      executor_entry->rand_offset = rand_offset;
      executor_entry->init = true;
    }
  }

  // push back global buffers
  args.push(global_buffers.buffers);

  // push back RNG state if needed
  if (lowered_->kernel()->summary().max_rng_offsets >= 0) {
    args.appendPhiloxRNGSeed(rand_offset);
  }

  if (isDebugDumpEnabled(DebugDumpOption::LaunchParam)) {
    launch_params_.print();
  }

  if (isDebugDumpEnabled(DebugDumpOption::KernelArgs)) {
    std::cout << "Arguments for kernel" << fusion_id_ << ":" << std::endl
              << "Inputs:" << std::endl;
    for (auto i : c10::irange(num_inputs)) {
      std::cout << "  " << args[i]->toString() << std::endl;
    }
    std::cout << "Outputs:" << std::endl;
    // note: add aliased outputs here.
    for (const auto& output : allocated_outputs) {
      std::cout << "  " << output.scalar_type() << " " << output.sizes()
                << " (strides = " << output.strides()
                << ", address = " << output.data_ptr() << ")" << std::endl;
    }
    std::cout << "Reduction and semaphore buffers:" << std::endl;
    TORCH_INTERNAL_ASSERT(
        global_buffers.buffers.size() == global_buffers.zero_init.size(),
        "global_buffer buffer & zero_init container should have identical sizes");
    for (const auto i : c10::irange(global_buffers.buffers.size())) {
      const auto& buffer = global_buffers.buffers[i];
      const auto& zero_init = global_buffers.zero_init[i];
      std::cout << "  " << buffer.scalar_type() << " " << buffer.sizes()
                << " is_zero_initialized: " << zero_init << std::endl;
    }
  }

  cudaEvent_t start_event = {};
  cudaEvent_t finish_event = {};

  if (measure_kernel_time_ ||
      isDebugDumpEnabled(DebugDumpOption::EffectiveBandwidth) ||
      isDebugDumpEnabled(DebugDumpOption::PerfDebugVerbose)) {
    CUDA_RT_SAFE_CALL(cudaEventCreate(&start_event));
    CUDA_RT_SAFE_CALL(cudaEventCreate(&finish_event));
    CUDA_RT_SAFE_CALL(cudaEventRecord(start_event, stream));
  }

  if (execute_kernel_) {
    if (maybe_available_dynamic_smem_.has_value() &&
        size_t(launch_params_.smem()) > maybe_available_dynamic_smem_.value()) {
      // Increase limit of dynamic shared memory if needed.
      CUDA_SAFE_CALL(cuFuncSetAttribute(
          compiled_kernel_.function,
          CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
          launch_params_.smem()));
    }
    if (!kernel()->summary().has_cooperative_grid_reduction) {
      FUSER_PERF_SCOPE("ExecutorRunFusion::cuLaunchKernel");
      CUDA_SAFE_CALL(cuLaunchKernel(
          compiled_kernel_.function,
          launch_params_.gdimx(),
          launch_params_.gdimy(),
          launch_params_.gdimz(),
          launch_params_.bdimx(),
          launch_params_.bdimy(),
          launch_params_.bdimz(),
          launch_params_.smem(),
          stream,
          args.getBuffer(),
          nullptr));
    } else {
      FUSER_PERF_SCOPE("ExecutorRunFusion::cuLaunchCooperativeKernel");
      CUDA_SAFE_CALL(cuLaunchCooperativeKernel(
          compiled_kernel_.function,
          launch_params_.gdimx(),
          launch_params_.gdimy(),
          launch_params_.gdimz(),
          launch_params_.bdimx(),
          launch_params_.bdimy(),
          launch_params_.bdimz(),
          launch_params_.smem(),
          stream,
          args.getBuffer()));
    }
  }

  if (measure_kernel_time_ ||
      isDebugDumpEnabled(DebugDumpOption::EffectiveBandwidth) ||
      isDebugDumpEnabled(DebugDumpOption::PerfDebugVerbose)) {
    CUDA_RT_SAFE_CALL(cudaEventRecord(finish_event, stream));
    CUDA_RT_SAFE_CALL(cudaEventSynchronize(start_event));
    CUDA_RT_SAFE_CALL(cudaEventSynchronize(finish_event));
    CUDA_RT_SAFE_CALL(
        cudaEventElapsedTime(&kernel_time_ms_, start_event, finish_event));
    CUDA_RT_SAFE_CALL(cudaEventDestroy(start_event));
    CUDA_RT_SAFE_CALL(cudaEventDestroy(finish_event));

    bytes_processed_ = 0;
    // Figure how many bytes are inputs, outputs, and temporary buffers
    for (auto i : c10::irange(num_inputs)) {
      if (auto tensor_arg_abstract =
              dynamic_cast<const TensorArgAbstract*>(args[i])) {
        bytes_processed_ += tensor_arg_abstract->numel() *
            (int64_t)dataTypeSize(tensor_arg_abstract->getDataType());
      }
    }
    for (const auto& output : allocated_outputs) {
      bytes_processed_ += output.numel() *
          (int64_t)dataTypeSize(aten_to_data_type(output.scalar_type()));
    }

    if (isDebugDumpEnabled(DebugDumpOption::EffectiveBandwidth)) {
      double gb_per_s =
          ((double)bytes_processed_ / ((double)kernel_time_ms_ / 1000)) /
          (double)1.0e9;
      std::cout << "kernel" << fusion_id_ << " run in " << kernel_time_ms_
                << " ms, achieved: " << gb_per_s << " GB/s" << std::endl;
    }
  }

  if (isOptionEnabled(EnableOption::KernelProfile)) {
    std::cout << kernel()->profile().toString(global_buffers.profile_buffer);
  }

  return allocated_outputs;
}

void FusionExecutor::compileRtc(
    const std::string& code,
    const std::string& name,
    bool structured,
    PrimDataType index_type) {
  FUSER_PERF_SCOPE("ExecutorRunFusion::compileRtc");
  TORCH_INTERNAL_ASSERT(
      index_type == PrimDataType::Int || index_type == PrimDataType::Int32 ||
          "Invalid index type: ",
      index_type);
  std::string scode;
  if (!structured) {
    scode = getStructuredCode(code, index_type);
  } else {
    scode = code;
  }
  fusion_id_ = 1;

  std::tie(compiled_kernel_, last_compiler_log_, last_compiled_binary_) =
      executor_utils::getCompiledKernel(c10::nullopt, scode, name, fusion_id_);
}

float FusionExecutor::runRtc(
    const LaunchParams& launch_params,
    const std::vector<at::Tensor>& args,
    PrimDataType index_type) {
  FUSER_PERF_SCOPE("runFusion");

  c10::DeviceGuard dg(options_.device);
  auto stream = at::cuda::getCurrentCUDAStream();

  cudaEvent_t start_event = {};
  cudaEvent_t finish_event = {};

  CUDA_RT_SAFE_CALL(cudaEventCreate(&start_event));
  CUDA_RT_SAFE_CALL(cudaEventCreate(&finish_event));

  KernelArgumentHolder kernel_arguments(index_type);
  kernel_arguments.push(args);

  CUDA_RT_SAFE_CALL(cudaEventRecord(start_event, stream));

  CUDA_SAFE_CALL(cuLaunchKernel(
      compiled_kernel_.function,
      launch_params.gdimx(),
      launch_params.gdimy(),
      launch_params.gdimz(),
      launch_params.bdimx(),
      launch_params.bdimy(),
      launch_params.bdimz(),
      launch_params.smem(),
      stream,
      kernel_arguments.getBuffer(),
      nullptr));

  CUDA_RT_SAFE_CALL(cudaEventRecord(finish_event, stream));
  CUDA_RT_SAFE_CALL(cudaEventSynchronize(start_event));
  CUDA_RT_SAFE_CALL(cudaEventSynchronize(finish_event));

  float kernel_time_ms = 0;
  CUDA_RT_SAFE_CALL(
      cudaEventElapsedTime(&kernel_time_ms, start_event, finish_event));
  CUDA_RT_SAFE_CALL(cudaEventDestroy(start_event));
  CUDA_RT_SAFE_CALL(cudaEventDestroy(finish_event));

  return kernel_time_ms;
}

} // namespace nvfuser
