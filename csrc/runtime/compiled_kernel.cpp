// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on

#include <runtime/compiled_kernel.h>

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
#include <multidevice/communication.h>
#include <multidevice/communicator.h>
#include <multidevice/lower_communication.h>
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
#include <c10/util/irange.h>

#include <cmath>
#include <cstring>
#include <fstream>

namespace nvfuser {

namespace {

static const char* defineIndexType(PrimDataType index_type) {
  if (index_type == DataType::Int32) {
    return "typedef int nvfuser_index_t;\n";
  } else if (index_type == DataType::Int) {
    return "typedef int64_t nvfuser_index_t;\n";
  } else {
    NVF_THROW("invalid indexing type: ", index_type);
  }
}

static const char* defineTypes() {
  return R"(
using int8_t = signed char;
using uint8_t = unsigned char;
using int16_t = short int;
using uint16_t = unsigned short int;
using int32_t = int;
using uint32_t = unsigned int;
using int64_t = long long int;
using uint64_t = unsigned long long int;

// Modified from cuda.h
struct TensorMap {
  alignas(64)
  uint64_t opaque[16];
};
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

// When executing nvFuser with: NVFUSER_EXTERNAL_SRC=file1.cu,file2.cu
// This function retrieves structured code from the specified files.
// The files should be comma-separated, and their order corresponds to the
// fusion_id order. If the provided number of files is fewer than the fusion
// segments, the function will resort to the available files in sequence
// and issue a warning.
std::string getStructuredCodeFromExternalFiles(const int64_t fusion_id) {
  auto external_code_path = getNvFuserEnv("EXTERNAL_SRC");
  if (!external_code_path) {
    return "";
  }
  std::string all_external_code_paths(external_code_path);
  if (all_external_code_paths.empty() || fusion_id < 1) {
    return "";
  }
  auto getExternalCodeFile =
      [fusion_id](const std::string& input) -> std::string {
    std::stringstream ss(input);
    std::string token;
    int64_t count = 0;
    while (std::getline(ss, token, ',')) {
      if (++count == fusion_id) {
        return token;
      }
    }
    debug()
        << "Didn't find requested external source code. Will use generated code!\n"
        << "Number of source code files should equal the number of fusion segments.\n"
        << "External source code filenames should be delineated with commas, e.g.: file1.cu,file2.cu.\n";
    return "";
  };

  std::string single_code_path = getExternalCodeFile(all_external_code_paths);
  if (single_code_path.empty()) {
    return "";
  }
  std::ifstream cuda_src(single_code_path);
  if (!cuda_src.is_open()) {
    debug() << "Failed to open external source file: " << single_code_path
            << std::endl;
    return "";
  }
  debug() << "--------> Compiling external CUDA code: " << single_code_path
          << std::endl;

  std::stringstream buffer;
  buffer << cuda_src.rdbuf();
  return buffer.str();
}
} // namespace

NVF_API CompiledKernel::CompiledKernel(
    Fusion* fusion,
    CompileParams compile_params)
    : compile_params_(compile_params),
      lowered_(std::make_unique<GpuLower>(fusion, compile_params)) {
  FUSER_PERF_SCOPE("CompiledKernel::CompiledKernel");
  for (const auto& hook : lowering_hooks_) {
    hook(lowered_.get());
  }
  lowered_->run();
}

void CompiledKernel::compileFusion(
    c10::Device device,
    const LaunchParams& launch_params,
    SchedulerType scheduler_type,
    int64_t fusion_id,
    int64_t concrete_id,
    int64_t runtime_id,
    int64_t group_id) {
  FUSER_PERF_SCOPE("CompiledKernel::compileFusion");

  NVF_ERROR(
      !fusion()->outputs().empty(),
      "No output found for this kernel, aborting.");

  options_.device = device;

  // NOTE: Profiling needs to be started below the isExpressionEvaluated query
  // given the conditional can exit early from compilation.
  if (isProfilerEnabled()) {
    NVF_CHECK(
        group_id >= 0,
        "An invalid segment id is passed to FusionProfiler!:",
        group_id);
    FusionProfiler::segment(group_id).startCompile(device.index());
  }

  for (auto out : fusion()->outputs()) {
    const auto logical_domain = out->as<TensorView>()->getLogicalDomain();
    // walking through outputs to see if output shapes are dependent on
    // non-tensor inputs. For which case, we should have disabled output
    // allocation, since the caching id only looks at tensor shapes.
    // See issue https://github.com/csarofeen/pytorch/issues/2002
    std::vector<Val*> output_extents;
    for (const auto id : logical_domain) {
      Val* extent = nullptr;
      if (id->isReduction() || id->isStride() || id->isDeviceDim()) {
        continue;
      } else if (id->isBroadcast() && id->hasExpandedExtent()) {
        extent = id->expandedExtent();
      } else {
        extent = id->extent();
      }
      output_extents.emplace_back(extent);
    }
    auto dependencies = InputsOf::outputs(output_extents);
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
    fusion()->print();
  } else if (isDebugDumpEnabled(DebugDumpOption::FusionIrMath)) {
    fusion()->printMath();
  }

  c10::DeviceGuard dg(options_.device);

  NVF_ERROR(
      options_.device.is_cuda(), "Provided device to CUDA fuser is the CPU.");
  auto properties = at::cuda::getDeviceProperties(options_.device.index());
  // TODO: These properties should be set as part of the constructor so that it
  // can be const
  warp_size_ = properties->warpSize;
  kir::Kernel* kernel = lowered_->kernel();

  for (const auto& hook : post_lowering_hooks_) {
    hook(kernel);
  }
  createKernelId(scheduler_type, fusion_id, concrete_id, runtime_id, group_id);
  setUsedTVs();

  if (isDebugDumpEnabled(DebugDumpOption::KernelIr)) {
    kernel->print();
  }

  if (isDebugDumpEnabled(DebugDumpOption::BankConflictInfo)) {
    auto bank_conflict_info = getBankConflictInfo(kernel);
    if (bank_conflict_info.empty()) {
      debug() << "===== No bank confliction =====" << std::endl;
    } else {
      debug() << "======= Bank confliction =======" << std::endl;
      for (auto info : bank_conflict_info) {
        debug() << "Expr: " << info.first->toString() << std::endl;
        auto conflict = info.second;
        if (conflict.first > 1) {
          debug() << "input conflict: " << conflict.first << " way, ";
        }
        if (conflict.second > 1) {
          debug() << "output conflict: " << conflict.second << " way";
        }
        debug() << std::endl;
      }
      debug() << "================================" << std::endl;
    }
  }

  // TODO: pass in kernel name?
  kernel_code_ = codegen::generateCudaKernel(kernel, kernelName());

  // If NVFUSER_EXTERNAL_SRC is set, utilize the external source code.
  // If the loaded external source code is empty, revert to the default codegen.
  // The external_structured_code is moved to structured_code and explicitly
  // cleared to avoid use-after-move scenarios.
  // Note: we index these with getGlobalFusionCount() instead of fusion_id_ in
  // order to match the numbering of files output with
  // NVFUSER_DUMP=cuda_to_file
  auto structured_code =
      getStructuredCodeFromExternalFiles(getGlobalFusionCount());
  if (structured_code.empty()) {
    structured_code = getStructuredCode();
  }

  const kir::KernelSummary& kernel_summary = kernel->summary();

  // TODO: this replicates the target GPU version computation from
  // executor_utils.
  std::pair<int64_t, int64_t> target_arch;
  bool compile_to_sass = false;
  executor_utils::queryTargetGPUVersion(
      properties,
      std::ref(target_arch.first),
      std::ref(target_arch.second),
      compile_to_sass);

  NVF_CHECK(
      target_arch >= kernel_summary.min_device_version,
      "Target compute capability is ",
      target_arch.first,
      ".",
      target_arch.second,
      " but this fusion requires at least ",
      kernel_summary.min_device_version.first,
      ".",
      kernel_summary.min_device_version.second,
      ". Reason: ",
      kernel_summary.min_device_version_reason);

  // We currently shouldn't allocate any more shared mem
  //  tensors statically but could keep this path if
  //  needed in later development.
  if (!kernel_summary.static_smem_allocations.empty()) {
    ExpressionEvaluator static_evaluator;
    const auto static_smem_size = computeSharedMemory(
        static_evaluator,
        kernel_summary.static_smem_allocations,
        kernel->indexType());
    NVF_ERROR(
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
    NVF_THROW(ss.str());
  }

  NVF_ERROR(
      launch_params.nThreads() > 0, "launch param inferred block size < 0");

  // TODO: high water mark should be computed via occupancy API after
  // compilation.

  // Basically setting high water mark as 1 when we don't provide args for
  // compilation, it will just generate a kernel that gets ditched at the first
  // run - not great. We should have better heuristics.
  block_size_high_water_mark_ =
      std::max<int64_t>(launch_params.nThreads(), block_size_high_water_mark_);
  maxrregcount_high_water_mark_ = compile_params_.maxrregcount;
  compiled_kernel_ = executor_utils::getCompiledKernel(
      kernel_code_,
      structured_code,
      kernelName(),
      kernel_id_,
      compile_params_,
      launch_params.nThreads());

  NVF_ERROR(validKernelId(), "Invalid kernel id for CompiledKernel.");

  if (isDebugDumpEnabled(DebugDumpOption::Sass)) {
    debug() << disassembledKernelSASS() << std::endl;
  }
  if (isProfilerEnabled()) {
    FusionProfiler::segment(group_id).stopCompile();
  }
}

std::string CompiledKernel::getStructuredCode(
    const std::string& kernel_str,
    PrimDataType index_type) const {
  // generating cuda code;
  std::string code = "";
  code += includeStdComplex();
  code += std::string("namespace {\n") + defineTypes() +
      defineIndexType(index_type) + executor_utils::kernelPreamble() +
      kernel_str + "}\n";

  if (isDebugDumpEnabled(DebugDumpOption::CudaKernel)) {
    debug() << "\n======= Codegen output for kernel: " << kernelName()
            << " =======\n\n"
            << kernel_str << "\n======================================\n\n";
  } else if (isDebugDumpEnabled(DebugDumpOption::CudaFull)) {
    debug() << "\n======= Codegen output for kernel: " << kernelName()
            << " =======\n\n"
            << code << "\n======================================\n\n";
  }
  if (isDebugDumpEnabled(DebugDumpOption::CudaToFile)) {
    std::stringstream file_name;
    file_name << "__tmp_kernel_" << kernel_id_ << ".cu";
    debug() << "PRINTING: " << file_name.str() << std::endl;
    std::ofstream out(file_name.str());
    out << code << std::endl;
    out.close();
  }

  return code;
}

std::string CompiledKernel::getStructuredCode() const {
  return getStructuredCode(kernelString(), kernel()->indexType());
}

void CompiledKernel::compileRtc(
    const std::string& code,
    const std::string& name,
    bool structured,
    PrimDataType index_type) {
  FUSER_PERF_SCOPE("CompiledKernel::compileRtc");
  NVF_ERROR(
      index_type == PrimDataType::Int || index_type == PrimDataType::Int32 ||
          "Invalid index type: ",
      index_type);

  createKernelId();

  std::string scode;
  if (!structured) {
    scode = getStructuredCode(code, index_type);
  } else {
    scode = code;
  }
  compiled_kernel_ =
      executor_utils::getCompiledKernel(std::nullopt, scode, name, kernel_id_);
}

float CompiledKernel::runRtc(
    const LaunchParams& launch_params,
    const std::vector<at::Tensor>& args,
    PrimDataType index_type) {
  FUSER_PERF_SCOPE("CompiledKernel::runRtc");

  c10::DeviceGuard dg(options_.device);
  auto stream = at::cuda::getCurrentCUDAStream();

  cudaEvent_t start_event = {};
  cudaEvent_t finish_event = {};

  NVFUSER_CUDA_RT_SAFE_CALL(cudaEventCreate(&start_event));
  NVFUSER_CUDA_RT_SAFE_CALL(cudaEventCreate(&finish_event));

  NVFUSER_CUDA_RT_SAFE_CALL(cudaEventRecord(start_event, stream));

  std::vector<std::vector<std::byte>> data;
  std::vector<void*> pointers;

  for (const auto& input : args) {
    auto dtype =
        std::get<PrimDataType>(aten_to_data_type(input.scalar_type()).type);
    DataType metadata_type = globalTensorMetaData(dtype, input.dim());

    std::shared_ptr<Struct> struct_ = std::make_shared<TensorMetaData>();
    TensorMetaData* metadata = (TensorMetaData*)struct_.get();
    metadata->dtype = dtype;
    metadata->data = input.data_ptr();
    metadata->logical_size = input.sizes();
    metadata->logical_stride = input.strides();
    metadata->alloc_size = input.sizes();
    metadata->alloc_stride = input.strides();

    data.emplace_back(polymorphicValueToBytes(
        PolymorphicValue(std::move(struct_)), metadata_type, index_type));
    pointers.emplace_back(data.back().data());
  }

  NVFUSER_CUDA_SAFE_CALL(cuLaunchKernel(
      compiled_kernel_->function,
      launch_params.gdimx(),
      launch_params.gdimy(),
      launch_params.gdimz(),
      launch_params.bdimx(),
      launch_params.bdimy(),
      launch_params.bdimz(),
      launch_params.smem(),
      stream,
      pointers.data(),
      nullptr));

  NVFUSER_CUDA_RT_SAFE_CALL(cudaEventRecord(finish_event, stream));
  NVFUSER_CUDA_RT_SAFE_CALL(cudaEventSynchronize(start_event));
  NVFUSER_CUDA_RT_SAFE_CALL(cudaEventSynchronize(finish_event));

  float kernel_time_ms = 0;
  NVFUSER_CUDA_RT_SAFE_CALL(
      cudaEventElapsedTime(&kernel_time_ms, start_event, finish_event));
  NVFUSER_CUDA_RT_SAFE_CALL(cudaEventDestroy(start_event));
  NVFUSER_CUDA_RT_SAFE_CALL(cudaEventDestroy(finish_event));

  return kernel_time_ms;
}

// flatbuffers::Offset<serde::CompiledKernel> CompiledKernel::serialize(
//     flatbuffers::FlatBufferBuilder& builder) const {
//   // See table definition for CompiledKernel in serde/fusion_cache.fbs
//   using fb_executor_entry = flatbuffers::Offset<serde::ExecutorEntry>;

//   // Separate unordered_map for executor_entry_lookup into key and value
//   // vectors. The key value is the cache_id value in the
//   KernelArgumentHolder. std::vector<size_t> executor_entry_lookup_keys_fb;
//   std::vector<fb_executor_entry> executor_entry_lookup_values_fb;
//   for (const auto& [key, value] : executor_entry_lookup_) {
//     executor_entry_lookup_keys_fb.push_back(key);
//     executor_entry_lookup_values_fb.push_back(serialize(builder, value));
//   }

//   // When compilation is skipped, avoid serializing cubin because it doesn't
//   // exist. The remaining fields are also not necessary in this case.
//   if (!hasCompiledKernel()) {
//     return serde::CreateCompiledKernelDirect(builder);
//   }

//   return serde::CreateCompiledKernelDirect(
//       builder,
//       device_smem_limit_,
//       block_size_high_water_mark_,
//       maxrregcount_high_water_mark_,
//       warp_size_,
//       toUnderlying(scheduler_type_),
//       fusion_id_,
//       concrete_id_,
//       runtime_id_,
//       group_id_,
//       kernel_code_.c_str(),
//       &executor_entry_lookup_keys_fb,
//       &executor_entry_lookup_values_fb,
//       toUnderlying(kernel()->indexType()),
//       serialize(builder, compiled_kernel_.get()));
// }

// flatbuffers::Offset<serde::CudaKernel> CompiledKernel::serialize(
//     flatbuffers::FlatBufferBuilder& builder,
//     const executor_utils::CompiledKernel* compiled_kernel) const {
//   NVF_ERROR(
//       compiled_kernel_ != nullptr &&
//           (!compiled_kernel->cubin.empty() || !compiled_kernel->ptx.empty()),
//       "Expected compiled cuda kernel before serializing CompiledKernel.");

//   auto fb_kernel_name = builder.CreateString(compiled_kernel->kernel_name);
//   auto fb_compile_args = builder.CreateString(compiled_kernel->compile_args);

//   flatbuffers::Offset<flatbuffers::Vector<uint8_t>> fb_cubin = 0;
//   flatbuffers::Offset<flatbuffers::String> fb_cubin_filename = 0;
//   if (!compiled_kernel->cubin.empty()) {
//     uint8_t* cubin_ptr = nullptr;
//     fb_cubin = builder.CreateUninitializedVector(
//         compiled_kernel->cubin.size(), &cubin_ptr);
//     std::copy(
//         compiled_kernel->cubin.begin(),
//         compiled_kernel->cubin.end(),
//         cubin_ptr);
//     fb_cubin_filename =
//     builder.CreateString(compiled_kernel->cubin_filename);
//   }

//   flatbuffers::Offset<flatbuffers::Vector<uint8_t>> fb_ptx = 0;
//   flatbuffers::Offset<flatbuffers::String> fb_ptx_filename = 0;
//   if (!compiled_kernel->ptx.empty()) {
//     uint8_t* ptx_ptr = nullptr;
//     fb_ptx = builder.CreateUninitializedVector(
//         compiled_kernel->ptx.size(), &ptx_ptr);
//     std::copy(
//         compiled_kernel->ptx.begin(), compiled_kernel->ptx.end(), ptx_ptr);
//     fb_ptx_filename = builder.CreateString(compiled_kernel->ptx_filename);
//   }

//   serde::CudaKernelBuilder ckb(builder);
//   ckb.add_cubin(fb_cubin);
//   ckb.add_cubin_filename(fb_cubin_filename);
//   ckb.add_ptx(fb_ptx);
//   ckb.add_ptx_filename(fb_ptx_filename);
//   ckb.add_kernel_name(fb_kernel_name);
//   ckb.add_compile_args(fb_compile_args);
//   ckb.add_block_size(compiled_kernel->block_size);
//   return ckb.Finish();
// }

// std::unique_ptr<PrecomputedValues>& CompiledKernel::
//     evaluatorPrecomputedValues() {
//   if (!evaluator_precomputed_values_) {
//     evaluator_precomputed_values_ =
//         std::make_unique<PrecomputedValues>(lowered_->kernel());
//   }
//   return evaluator_precomputed_values_;
// }

// void CompiledKernel::deserialize(
//     const serde::CompiledKernel* buffer,
//     Fusion* fusion,
//     int8_t device_index,
//     CompileParams compile_params,
//     SchedulerType heuristic,
//     int64_t fusion_id,
//     int64_t concrete_id,
//     int64_t runtime_id,
//     int64_t group_id) {
//   // See table definition for CompiledKernel in serde/fusion_cache.fbs

//   NVF_ERROR(buffer != nullptr, "serde::CompiledKernel is nullptr.");

//   // TODO Should we set fusion_id, concrete_id, runtime_id, and group_id when
//   we
//   // skip compilation?
//   if (isExpressionEvaluated(fusion)) {
//     fusion_ = std::make_unique<Fusion>(*fusion);
//     NVF_ERROR(!hasCompiledKernel(), "Failed to deserialize CompiledKernel");
//     return;
//   }

//   NVF_ERROR(
//       fusion_id == buffer->fusion_id(),
//       "Expected given fusion_id to match serde fusion_id.");
//   NVF_ERROR(
//       concrete_id == buffer->concrete_id(),
//       "Expected given concrete_id to match serde concrete_id.");
//   NVF_ERROR(
//       runtime_id == buffer->runtime_id(),
//       "Expected given runtime_id to match serde runtime_id.");
//   NVF_ERROR(
//       group_id == buffer->group_id(),
//       "Expected given group_id to match serde group_id.");
//   NVF_ERROR(
//       toUnderlying(heuristic) == buffer->heuristic(),
//       ": ",
//       toUnderlying(heuristic),
//       " vs ",
//       buffer->heuristic());

//   // Initialize CompileOptions
//   options_.device = c10::Device(c10::DeviceType::CUDA, device_index);
//   c10::DeviceGuard dg(options_.device);

//   // Initialize internal fields
//   device_smem_limit_ = buffer->device_smem_limit();
//   block_size_high_water_mark_ = buffer->block_size_high_water_mark();
//   maxrregcount_high_water_mark_ = buffer->maxrregcount_high_water_mark();
//   warp_size_ = buffer->warp_size();
//   kernel_code_ = buffer->kernel_code()->str();

//   // KernelDB query checks kernel_code string and compile_params before
//   // copying cubin.
//   compile_params.index_type = serde::mapToNvfuserDtype(buffer->index_type());
//   compile_params.maxrregcount = maxrregcount_high_water_mark_;

//   // Get lowered fusion
//   lowered_ = std::make_unique<GpuLower>(fusion, compile_params);
//   lowered_->run();

//   // Replace integers that are tensor sizes by named scalars like
//   "T0.size[0]" createKernelId(
//       heuristic,
//       buffer->fusion_id(),
//       buffer->concrete_id(),
//       buffer->runtime_id(),
//       buffer->group_id());
//   setUsedTVs();

//   // GlobalBufferInfo requires lowered kernel before deserialization
//   for (auto idx : c10::irange(buffer->executor_entry_lookup_keys()->size()))
//   {
//     executor_entry_lookup_.emplace(
//         buffer->executor_entry_lookup_keys()->Get(idx),
//         deserialize(buffer->executor_entry_lookup_values()->Get(idx)));
//   }

//   compiled_kernel_ = executor_utils::getCompiledKernel(
//       buffer->compiled_kernel(), compile_params);

//   NVF_ERROR(hasCompiledKernel(), "Failed to deserialize CompiledKernel");
// }

void CompiledKernel::setUsedTVs() {
  auto used_vals = fusion()->usedMathVals();
  auto used_tvs = ir_utils::filterByType<TensorView>(used_vals);
  used_tvs_.clear();
  used_tvs_.insert(used_tvs_.begin(), used_tvs.begin(), used_tvs.end());
}

namespace {
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
} // namespace

void CompiledKernel::recompileKernel(
    const LaunchParams& new_launch_params,
    const CompileParams& new_compile_params) {
  FUSER_PERF_SCOPE("CompiledKernel::runFusion::recompileKernel");

  const auto structured_code = getStructuredCode();
  block_size_high_water_mark_ = new_launch_params.nThreads();
  maxrregcount_high_water_mark_ = new_compile_params.maxrregcount;

  compiled_kernel_ = executor_utils::getCompiledKernel(
      kernel_code_,
      structured_code,
      kernelName(),
      kernel_id_,
      new_compile_params,
      block_size_high_water_mark_);

  if (kernel()->summary().has_cooperative_grid_reduction) {
    // We need to increase shared memory before kernel launch, but also before
    // calling into `validateCooperativeLaunch`!
    // So we need to do it there before calling into the validation, to avoid
    // false positives
    validateCooperativeLaunch(
        compiled_kernel_->function, new_launch_params, options_.device.index());
  }
}

} // namespace nvfuser
