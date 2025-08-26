// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on

#include <cuda.h>
#include <cuda_runtime.h>
#include <cutlass/codegen.h>
#include <debug.h>
#include <dlfcn.h>
#include <fusion.h>
#include <instrumentation.h>
#include <ir/all_nodes.h>
#include <ops/all_ops.h>
#include <options.h>
#include <runtime/compiled_kernel.h>
#include <runtime/cutlass_compiled_kernel.h>
#include <runtime/executor_kernel_arg.h>
#include <runtime/executor_params.h>
#include <unistd.h>
#include <utils.h>

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDACachingAllocator.h>
#include <c10/cuda/CUDAFunctions.h>
#include <c10/cuda/CUDAMathCompat.h>
#include <c10/util/Exception.h>

#include <chrono>
#include <filesystem>
#include <format>
#include <fstream>
#include <string>
#include <variant>
#include <vector>

namespace nvfuser {

namespace {

// Get CUTLASS install path from environment or use default
std::filesystem::path getCutlassPath() {
  if (const char* env_path = std::getenv("CUTLASS_PATH")) {
    return std::filesystem::path(env_path);
  }
  // Default to system CUDA path
  return {"/usr/local/cuda"};
}

// Get Torch install path from environment or use default
std::filesystem::path getTorchPath() {
  if (const char* env_path = std::getenv("TORCH_PATH")) {
    return std::filesystem::path(env_path);
  }
  return "";
}

// Get compute capability string
std::string getComputeCapabilityString(int compute_capability) {
  if (compute_capability == 0) {
    // Auto-detect
    int device_id = at::cuda::current_device();
    cudaDeviceProp prop;
    cudaError_t result = cudaGetDeviceProperties(&prop, device_id);
    if (result != cudaSuccess) {
      NVF_THROW(
          "Failed to get device properties: ", cudaGetErrorString(result));
    }
    compute_capability = prop.major * 10 + prop.minor;
  }
  return std::to_string(compute_capability);
}

} // namespace

CutlassCompiledKernel::CutlassCompiledKernel(
    Fusion* fusion,
    const CutlassParams& cutlass_params,
    const CompileParams& cparams,
    c10::Device device,
    int64_t fusion_id,
    int64_t concrete_id,
    int64_t runtime_id,
    int64_t group_id)
    : fusion_(fusion),
      cutlass_params_(cutlass_params),
      device_(device),
      fusion_id_(fusion_id),
      concrete_id_(concrete_id),
      runtime_id_(runtime_id),
      group_id_(group_id),
      cparams_(cparams) {
  NVF_CHECK(fusion != nullptr, "Fusion cannot be null");
}

CutlassCompiledKernel::~CutlassCompiledKernel() {
  if (cuda_module_) {
    cuModuleUnload(cuda_module_);
  }
  if (shared_library_handle_) {
    dlclose(shared_library_handle_);
  }
  // Clean up temporary directory if it exists
  if (!temp_dir_.empty() && std::filesystem::exists(temp_dir_)) {
    std::filesystem::remove_all(temp_dir_);
  }
}

void CutlassCompiledKernel::createKernelId() {
  NVF_ERROR(fusion_id_ > -1, "Invalid fusion_id.");
  NVF_ERROR(concrete_id_ > -1, "Invalid concrete_id.");
  NVF_ERROR(runtime_id_ > -1, "Invalid runtime_id.");
  NVF_ERROR(group_id_ > -1, "Invalid group_id");
  ++global_cutlass_fusion_count_;
  std::stringstream ss;
  if (isOptionEnabled(EnableOption::StaticFusionCount)) {
    ss << global_cutlass_fusion_count_.load();
  } else {
    ss << "cutlass";
    ss << "_f" << fusion_id_;
    ss << "_c" << concrete_id_;
    ss << "_r" << runtime_id_;
    ss << "_g" << group_id_;
  }
  kernel_id_ = ss.str();
}

void CutlassCompiledKernel::compile() {
  FUSER_PERF_SCOPE("CutlassCompiledKernel::compile");

  if (isCompiled()) {
    return;
  }

  createKernelId();

  NVF_ERROR(validKernelId(), "Invalid kernel id for CompiledKernel.");

  // Generate CUTLASS code
  generateCode();

  // Compile the code using nvcc
  compileWithNVCC();

  // Load the compiled kernel
  loadKernel();

  // Create launch parameters
  createLaunchParams();

  compiled_ = true;
}

float CutlassCompiledKernel::run(
    const KernelArgumentHolder& args,
    cudaStream_t stream) {
  FUSER_PERF_SCOPE("CutlassCompiledKernel::run");

  if (!isCompiled()) {
    NVF_ERROR(false, "Kernel not compiled");
  }

  // Use base class run method if available, otherwise implement
  // CUTLASS-specific run For now, we'll implement a basic CUTLASS kernel launch
  cudaEvent_t start_event = {};
  cudaEvent_t finish_event = {};

  NVFUSER_CUDA_RT_SAFE_CALL(cudaEventCreate(&start_event));
  NVFUSER_CUDA_RT_SAFE_CALL(cudaEventCreate(&finish_event));

  NVFUSER_CUDA_RT_SAFE_CALL(cudaEventRecord(start_event, stream));

  // Launch the CUTLASS kernel
  if (cuda_function_) {
    // TODO: Pattern match and find this ordering automatically (see below for
    // codegen of entry point which will unpack the discovered ordering) For
    // nvfp4_scaled_mm_kernel, we need to call it as a C function Extract
    // arguments for nvfp4_scaled_mm_kernel Expected order: output, a, b,
    // scales_a, scales_b, alpha, beta
    NVF_ERROR(
        args.size() == 6,
        "Expected 6 arguments for nvfp4_scaled_mm_kernel but found ",
        args.size());

    // Get tensors from arguments
    std::vector<void*> arg_pointers;
    arg_pointers.reserve(args.size());
    for (const PolymorphicValue& arg : args) {
      if (arg.is<at::Tensor>()) {
        arg_pointers.push_back((void*)&arg.as<at::Tensor>());
      } else {
        NVF_THROW(
            "Non-tensor arguments are not yet supported in "
            "CutlassCompiledKernel");
      }
    }

    // Define the function signature for the kernel
    using KernelFunc = void (*)(const std::vector<void*>&, cudaStream_t);

    auto kernel_func = reinterpret_cast<KernelFunc>(cuda_function_);

    // Call the kernel
    kernel_func(arg_pointers, stream);
  }

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

void CutlassCompiledKernel::generateCode() {
  FUSER_PERF_SCOPE("CutlassCompiledKernel::generateCode");

  // Generate CUTLASS kernel code using the code generator
  cutlass_code_ = cutlass_codegen::generateCode(fusion_, cutlass_params_);

  // Dump the kernel if requested. Note that we do not currently distinguish
  // between the kernel and the entire CUTLASS source file.
  if (isDebugDumpEnabled(DebugDumpOption::CudaFull) ||
      isDebugDumpEnabled(DebugDumpOption::CudaKernel)) {
    debug() << cutlass_code_ << std::endl;
  }
  if (isDebugDumpEnabled(DebugDumpOption::CudaToFile)) {
    std::stringstream file_name;
    // TODO: choose name based on kernel name
    file_name << "__tmp_" << kernelName() << ".cu";
    debug() << "PRINTING: " << file_name.str() << std::endl;
    std::ofstream out(file_name.str());
    out << cutlass_code_ << std::endl;
    out.close();
  }
}

std::string CutlassCompiledKernel::kernelName() const {
  NVF_ERROR(!kernel_id_.empty(), "Invalid kernel name for cutlass executor.");
  std::stringstream ss;
  ss << "kernel_" << kernel_id_;
  return ss.str();
}

std::string getCompileCommand(
    CompileParams& cparams,
    const std::filesystem::path& source_file,
    const std::filesystem::path& output_file) {
  std::string compile_cmd = "nvcc";

  const auto prop = at::cuda::getCurrentDeviceProperties();
  int64_t major = 0, minor = 0;
  bool compile_to_sass = false;
  queryTargetGPUVersion(prop, major, minor, compile_to_sass);
  const int64_t compute_capability = 10 * major + minor;

  // Add compute capability
  compile_cmd += " -arch=" + getComputeCapabilityString(compute_capability);

  // Add CUTLASS include paths
  const std::filesystem::path cutlass_path = getCutlassPath();
  cparams.include_paths.push_back(cutlass_path / "include");
  cparams.include_paths.push_back(cutlass_path / "tools" / "util" / "include");
  const std::filesystem::path torch_path = getTorchPath();
  cparams.include_paths.push_back(torch_path / "include");
  cparams.include_paths.push_back(
      torch_path / "include" / "torch" / "csrc" / "api" / "include");

  compile_cmd = "nvcc";
  compile_cmd += " -forward-unknown-to-host-compiler";

  // Disable some warnings in host code
  compile_cmd += " -Wno-conversion";

  for (const auto& path : cparams.include_paths) {
    compile_cmd += " -I" + path;
  }

  compile_cmd +=
      " -Xcudafe "
      "--diag_suppress=cc_clobber_ignored,"
      "--diag_suppress=field_without_dll_interface,"
      "--diag_suppress=base_class_has_different_dll_interface,"
      "--diag_suppress=dll_interface_conflict_none_assumed,"
      "--diag_suppress=dll_interface_conflict_dllexport_assumed,"
      "--diag_suppress=bad_friend_decl";

  compile_cmd += " --expt-relaxed-constexpr --expt-extended-lambda";

  compile_cmd += " -O3";

  compile_cmd += " -std=c++17";

  std::string arch = getComputeCapabilityString(compute_capability);
  // 90 -> 90a  or  100 -> 100a
  // Note that without this we get errors like
  //   Trying to use TMA Descriptor Prefetch without CUTE_ARCH_TMA_SM90_ENABLED.
  // TODO: This should be done properly
  // https://github.com/nvidia/cutlass#target-architecture
  arch += "a";
  std::string compute_arch = "compute_" + arch;
  std::string sm_arch = "sm_" + arch;
  compile_cmd +=
      " \"--generate-code="
      "arch=" +
      compute_arch +
      ","
      "code=[" +
      compute_arch + "," + sm_arch + "]\"";

  compile_cmd +=
      " -DCUTE_USE_PACKED_TUPLE=1"
      " -DCUTLASS_ENABLE_TENSOR_CORE_MMA=1"
      " -DCUTLASS_VERSIONS_GENERATED"
      " -DCUTLASS_DEBUG_TRACE_LEVEL=0";

  compile_cmd +=
      " --expt-relaxed-constexpr --expt-extended-lambda "
      "--threads=32";

  compile_cmd +=
      " -Xcompiler=-Wconversion -Xcompiler=-fno-strict-aliasing "
      "-Xcompiler=-Wno-deprecated-declarations "
      "-Xcompiler=-fPIC";

#ifndef NDEBUG
  // On debug builds, build the host code in debug mode with nvcc also
  compile_cmd += " -Xcompiler=-g";
#endif

  if (isOptionEnabled(EnableOption::KernelDebug)) {
    compile_cmd += " -G";
  } else {
    compile_cmd += " -DNDEBUG";
  }
  if (isOptionEnabled(EnableOption::KernelLineInfo)) {
    compile_cmd += " -lineinfo";
  }

  compile_cmd +=
      " -x cu -shared -o " + output_file.string() + " " + source_file.string();

  // TODO: enable dumping cubin, ptx, and sass as in CompiledKernel

  return compile_cmd;
}

void CutlassCompiledKernel::compileWithNVCC() {
  FUSER_PERF_SCOPE("CutlassCompiledKernel::compileWithNVCC");

  // Create temporary directory for compilation
  temp_dir_ = std::filesystem::temp_directory_path() /
      ("nvfuser_cutlass_compile_" + std::to_string(getpid()));
  if (std::filesystem::exists(temp_dir_)) {
    std::filesystem::remove_all(temp_dir_);
  }
  std::filesystem::create_directories(temp_dir_);

  // Write source file
  std::filesystem::path source_file = temp_dir_ / "cutlass_kernel.cu";
  std::ofstream source_out(source_file);
  source_out << cutlass_code_;
  source_out.close();

  // Build nvcc command
  std::filesystem::path output_file = temp_dir_ / "cutlass_kernel.so";
  std::filesystem::path log_file = temp_dir_ / "nvcc_output.log";

  std::string compile_cmd =
      getCompileCommand(cparams_, source_file, output_file);
  // Execute nvcc compilation and capture output
  std::string full_cmd = compile_cmd + " 2>&1 > " + log_file.string();

  int result = -1;
  if (isDebugDumpEnabled(DebugDumpOption::CutlassCompile)) {
    debug() << "Compiling CUTLASS kernel with command\n"
            << full_cmd << std::endl;

    using Clock = std::chrono::steady_clock;
    Clock::time_point start_timestamp = Clock::now();

    result = system(full_cmd.c_str());

    Clock::duration duration = Clock::now() - start_timestamp;
    debug()
        << "NVCC CUTLASS kernel compile time: "
        << std::chrono::duration_cast<std::chrono::seconds>(duration).count()
        << " seconds" << std::endl;
    std::ifstream log_stream(log_file);
    std::string log_content(
        (std::istreambuf_iterator<char>(log_stream)),
        std::istreambuf_iterator<char>());
    log_stream.close();
    debug() << "\nnvcc output: " << log_content << std::endl;
  } else {
    result = system(full_cmd.c_str());
  }

  if (result != 0) {
    // Read compilation output for error details
    std::ifstream log_stream(log_file);
    std::string log_content(
        (std::istreambuf_iterator<char>(log_stream)),
        std::istreambuf_iterator<char>());
    log_stream.close();

    NVF_THROW(
        "nvcc compilation failed with code: ",
        result,
        "\nCommand: ",
        compile_cmd,
        "\nnvcc output: ",
        log_content);
  }

  // Load shared library
  shared_library_handle_ = dlopen(output_file.c_str(), RTLD_LAZY);
  if (!shared_library_handle_) {
    NVF_THROW("Failed to load compiled CUTLASS library: ", dlerror());
  }
}

void CutlassCompiledKernel::loadKernel() {
  if (shared_library_handle_) {
    // Get function from dlopen-loaded library
    cuda_function_ = reinterpret_cast<CUfunction>(
        dlsym(shared_library_handle_, "run_kernel"));
    if (!cuda_function_) {
      NVF_THROW("Failed to get CUTLASS kernel function: ", dlerror());
    }
  }
}

void CutlassCompiledKernel::createLaunchParams() {
  // Block dimensions based on CUTLASS tile configuration
  // The kernel uses 256x256 tiles with 4x4 warps
  // TODO: adapt this to our actual generated kernel once we respect tile size
  // params
  int block_dim_x = 384;
  int block_dim_y = 1;
  int block_dim_z = 1;

  // Grid dimensions will be computed at runtime based on problem size
  // For now, use placeholder values that will be updated during kernel launch
  launch_params_ = LaunchParams(
      LaunchParams::UNINITIALIZED_VAL, // gdimx - computed at runtime
      LaunchParams::UNINITIALIZED_VAL, // gdimy - computed at runtime
      1, // gdimz
      block_dim_x,
      block_dim_y,
      block_dim_z);
}

} // namespace nvfuser
