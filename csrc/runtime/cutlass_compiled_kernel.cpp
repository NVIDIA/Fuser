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
#include <scheduler/cutlass.h>
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
#include <fstream>
#include <string>
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
    const CutlassParams& params,
    int64_t fusion_id,
    int64_t concrete_id,
    int64_t runtime_id,
    int64_t group_id)
    : CompiledKernelBase(
          c10::Device(c10::DeviceType::CUDA, at::cuda::current_device()),
          SchedulerType::Cutlass,
          fusion_id,
          concrete_id,
          runtime_id,
          group_id),
      fusion_(fusion),
      params_(params) {}

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

  compiled_ = true;
}

void CutlassCompiledKernel::run(
    const KernelArgumentHolder& args,
    cudaStream_t stream) const {
  FUSER_PERF_SCOPE("CutlassCompiledKernel::run");

  if (!isCompiled()) {
    NVF_ERROR(false, "Kernel not compiled");
  }

  NVF_ERROR(temp_tensor_sizes_function_);
  NVF_ERROR(run_kernel_function_);
  // Launch the CUTLASS kernel

  // Get tensors from arguments
  std::vector<TensorArg> tensor_args;
  tensor_args.reserve(args.size());
  for (const PolymorphicValue& arg : args) {
    if (arg.is<at::Tensor>()) {
      auto t = arg.as<at::Tensor>();
      tensor_args.emplace_back(
          t.data_ptr(),
          t.dim(),
          (int64_t*)t.sizes().data(),
          (int64_t*)t.strides().data());
    } else {
      NVF_THROW(
          "Non-tensor arguments are not yet supported in "
          "CutlassCompiledKernel");
    }
  }

  size_t num_inputs_and_outputs = tensor_args.size();
  tensor_args.resize(num_inputs_and_outputs + num_temp_tensors_);
  std::cout << "num_temp_tensors_=" << num_temp_tensors_ << std::endl;
  std::cout << "num_inputs_and_outputs=" << num_inputs_and_outputs << std::endl;
  std::cout << "tensor_args.size()=" << tensor_args.size() << std::endl;
  for (auto ta : tensor_args) {
    std::cout << " " << ta.data_ptr << std::endl;
    std::cout << " dim=" << ta.dim << std::endl;
    std::cout << " sizes=";
    for (size_t i : arange(ta.dim)) {
      std::cout << " " << ta.sizes[i];
    }
    std::cout << "\n strides=";
    for (size_t i : arange(ta.dim)) {
      std::cout << " " << ta.strides[i];
    }
  }

  // Temporary tensors are appended after outputs in the tensor_args vector
  std::vector<at::Tensor> temp_tensors;
  temp_tensors.reserve(num_temp_tensors_);
  {
    std::vector<int64_t> temp_tensor_sizes(num_temp_tensors_);

    temp_tensor_sizes_function_(temp_tensor_sizes.data(), tensor_args);

    auto const temp_tensor_options =
        at::TensorOptions().dtype(at::kByte).device(
            at::kCUDA, args.getDeviceIndex());
    for (auto [i, sz] : enumerate(temp_tensor_sizes)) {
      if (isDebugDumpEnabled(DebugDumpOption::CutlassCompile)) {
        debug() << "Allocating " << sz
                << " bytes to use for CUTLASS temporary space" << std::endl;
      }
      at::Tensor t = at::empty({sz}, temp_tensor_options);
      temp_tensors.push_back(t);
      TensorArg& targ = tensor_args.at(num_inputs_and_outputs + i);
      targ.data_ptr = t.data_ptr();
      targ.dim = t.dim();
      targ.sizes = (int64_t*)t.sizes().data();
      targ.strides = (int64_t*)t.strides().data();
    }
  }

  run_kernel_function_(tensor_args, stream);
}

void CutlassCompiledKernel::generateCode() {
  FUSER_PERF_SCOPE("CutlassCompiledKernel::generateCode");

  // Generate CUTLASS kernel code using the code generator
  const cutlass_codegen::CutlassGeneratedCode ck =
      cutlass_codegen::generateCode(fusion_, params_);
  cutlass_code_ = ck.code;
  num_temp_tensors_ = ck.num_temp_tensors;

  std::string external_code =
      getStructuredCodeFromExternalFiles(getGlobalFusionCount());
  if (!external_code.empty()) {
    cutlass_code_ = external_code;
    return;
  }

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

std::string getCompileCommand(
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
  std::vector<std::string> include_paths;
  include_paths.push_back(cutlass_path / "include");
  include_paths.push_back(cutlass_path / "tools" / "util" / "include");

  compile_cmd = "nvcc";
  compile_cmd += " -forward-unknown-to-host-compiler";

  // Disable some warnings in host code
  compile_cmd += " -Wno-conversion";

  for (const std::string& path : include_paths) {
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

  NVF_ERROR(!cutlass_code_.empty());

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

  std::string compile_cmd = getCompileCommand(source_file, output_file);
  // Execute nvcc compilation and capture output
  std::string full_cmd = compile_cmd + " 2>&1 > " + log_file.string();

  using Clock = std::chrono::high_resolution_clock;
  auto start = Clock::now();

  int result = -1;
  result = system(full_cmd.c_str());

  auto stop = Clock::now();
  if (isDebugDumpEnabled(DebugDumpOption::CutlassCompile)) {
    debug() << "Compiling CUTLASS kernel took "
            << (std::chrono::duration_cast<std::chrono::milliseconds>(
                    stop - start) *
                .001)
            << " seconds" << std::endl;
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
    // Get functions from dlopen-loaded library
    temp_tensor_sizes_function_ = reinterpret_cast<TempTensorSizesFunc>(
        dlsym(shared_library_handle_, "temp_tensor_sizes"));
    if (!temp_tensor_sizes_function_) {
      NVF_THROW("Failed to get CUTLASS temp tensor size function: ", dlerror());
    }
    run_kernel_function_ = reinterpret_cast<RunKernelFunc>(
        dlsym(shared_library_handle_, "run_kernel"));
    if (!run_kernel_function_) {
      NVF_THROW("Failed to get CUTLASS kernel function: ", dlerror());
    }
  }
}

CutlassCompiledKernel::~CutlassCompiledKernel() {
  if (shared_library_handle_) {
    dlclose(shared_library_handle_);
  }
}

} // namespace nvfuser
