// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on

#include <runtime/cutlass_compiled_kernel.h>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda.h>
#include <dlfcn.h>
#include <nvrtc.h>

#include <exceptions.h>
#include <fusion.h>
#include <instrumentation.h>
#include <ir/all_nodes.h>
#include <ir/printer.h>
#include <ops/all_ops.h>

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <sstream>

namespace nvfuser {

namespace {

// Helper to check NVRTC errors
#define NVFUSER_NVRTC_CHECK(call)                                              \
  do {                                                                         \
    nvrtcResult result = call;                                                 \
    if (result != NVRTC_SUCCESS) {                                             \
      NVF_THROW("NVRTC error: ", nvrtcGetErrorString(result));                 \
    }                                                                          \
  } while (0)

// Helper to check CUDA driver errors
#define NVFUSER_CUDA_DRIVER_CHECK(call)                                        \
  do {                                                                         \
    CUresult result = call;                                                    \
    if (result != CUDA_SUCCESS) {                                              \
      const char* error_string;                                                \
      cuGetErrorString(result, &error_string);                                 \
      NVF_THROW("CUDA driver error: ", error_string);                          \
    }                                                                          \
  } while (0)

// Get CUTLASS include path from environment or default location
std::string getCutlassIncludePath() {
  if (const char* env_path = std::getenv("CUTLASS_PATH")) {
    return std::string(env_path) + "/include";
  }
  // Default to system include path
  return "/usr/local/cuda/include";
}

// Get compute capability string
std::string getComputeCapabilityString(int compute_capability) {
  if (compute_capability == 0) {
    // Auto-detect
    int device_id = at::cuda::current_device();
    cudaDeviceProp prop;
    NVFUSER_CUDA_SAFE_CALL(cudaGetDeviceProperties(&prop, device_id));
    compute_capability = prop.major * 10 + prop.minor;
  }
  return "sm_" + std::to_string(compute_capability);
}

} // namespace

CutlassCompiledKernel::CutlassCompiledKernel(
    Fusion* fusion,
    const CutlassParams& cutlass_params,
    const CutlassCompileOptions& compile_options)
    : fusion_(fusion),
      cutlass_params_(cutlass_params),
      compile_options_(compile_options) {
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

void CutlassCompiledKernel::compile() {
  FUSER_PERF_SCOPE("CutlassCompiledKernel::compile");
  
  if (isCompiled()) {
    return;
  }
  
  // Generate CUTLASS code
  generateCode();
  
  // Compile the code
  if (compile_options_.use_nvrtc) {
    compileWithNVRTC();
  } else {
    compileWithNVCC();
  }
  
  // Load the compiled kernel
  loadKernel();
  
  // Generate kernel arguments structure
  generateKernelArguments();
  
  // Create launch parameters
  createLaunchParams();
  
  compiled_ = true;
}

void CutlassCompiledKernel::generateCode() {
  FUSER_PERF_SCOPE("CutlassCompiledKernel::generateCode");
  
  cutlass_code_ = CutlassCodeGenerator::generateCode(
      fusion_, cutlass_params_, descriptor_);
}

void CutlassCompiledKernel::compileWithNVRTC() {
  FUSER_PERF_SCOPE("CutlassCompiledKernel::compileWithNVRTC");
  
  nvrtcProgram program;
  
  // Create program
  NVFUSER_NVRTC_CHECK(nvrtcCreateProgram(
      &program,
      cutlass_code_.c_str(),
      descriptor_.kernel_name.c_str(),
      0,
      nullptr,
      nullptr));
  
  // Build compile options
  std::vector<std::string> options;
  
  // Add compute capability
  std::string arch_flag = "--gpu-architecture=" + 
      getComputeCapabilityString(compile_options_.compute_capability);
  options.push_back(arch_flag);
  
  // Add optimization level
  options.push_back("-O" + std::to_string(compile_options_.optimization_level));
  
  // Add C++ standard
  options.push_back("--std=c++17");
  
  // Add include paths
  std::string cutlass_include = "-I" + getCutlassIncludePath();
  options.push_back(cutlass_include);
  
  for (const auto& path : compile_options_.include_paths) {
    options.push_back("-I" + path);
  }
  
  // Add defines
  for (const auto& define : compile_options_.defines) {
    options.push_back("-D" + define);
  }
  
  // Add debug flags if needed
  if (compile_options_.debug) {
    options.push_back("-G");
    options.push_back("-lineinfo");
  }
  
  // Convert options to char*
  std::vector<const char*> option_ptrs;
  for (const auto& opt : options) {
    option_ptrs.push_back(opt.c_str());
  }
  
  // Compile
  nvrtcResult compile_result = nvrtcCompileProgram(
      program,
      static_cast<int>(option_ptrs.size()),
      option_ptrs.data());
  
  // Get compilation log
  size_t log_size;
  NVFUSER_NVRTC_CHECK(nvrtcGetProgramLogSize(program, &log_size));
  compilation_log_.resize(log_size);
  NVFUSER_NVRTC_CHECK(nvrtcGetProgramLog(program, compilation_log_.data()));
  
  if (compile_result != NVRTC_SUCCESS) {
    nvrtcDestroyProgram(&program);
    NVF_THROW("NVRTC compilation failed:\n", compilation_log_);
  }
  
  // Get PTX
  size_t ptx_size;
  NVFUSER_NVRTC_CHECK(nvrtcGetPTXSize(program, &ptx_size));
  binary_.resize(ptx_size);
  NVFUSER_NVRTC_CHECK(nvrtcGetPTX(program, binary_.data()));
  
  nvrtcDestroyProgram(&program);
}

void CutlassCompiledKernel::compileWithNVCC() {
  FUSER_PERF_SCOPE("CutlassCompiledKernel::compileWithNVCC");
  
  // Create temporary directory
  temp_dir_ = std::filesystem::temp_directory_path() / 
      ("nvfuser_cutlass_" + std::to_string(std::rand()));
  std::filesystem::create_directories(temp_dir_);
  
  // Write source file
  std::string source_file = temp_dir_ / (descriptor_.kernel_name + ".cu");
  std::ofstream ofs(source_file);
  ofs << cutlass_code_;
  ofs.close();
  
  // Build nvcc command
  std::stringstream cmd;
  cmd << "nvcc ";
  cmd << "-std=c++17 ";
  cmd << "-O" << compile_options_.optimization_level << " ";
  cmd << "--shared ";
  cmd << "-Xcompiler -fPIC ";
  
  // Add architecture
  cmd << "-arch=" << getComputeCapabilityString(compile_options_.compute_capability) << " ";
  
  // Add include paths
  cmd << "-I" << getCutlassIncludePath() << " ";
  for (const auto& path : compile_options_.include_paths) {
    cmd << "-I" << path << " ";
  }
  
  // Add defines
  for (const auto& define : compile_options_.defines) {
    cmd << "-D" << define << " ";
  }
  
  // Add debug flags
  if (compile_options_.debug) {
    cmd << "-G -lineinfo ";
  }
  
  // Output file
  std::string output_file = temp_dir_ / (descriptor_.kernel_name + ".so");
  cmd << "-o " << output_file << " ";
  cmd << source_file;
  
  // Execute nvcc
  int ret = std::system(cmd.str().c_str());
  if (ret != 0) {
    // Read nvcc output for error message
    NVF_THROW("nvcc compilation failed with code ", ret);
  }
  
  // Store the output path for loading
  temp_dir_ = output_file;
}

void CutlassCompiledKernel::loadKernel() {
  FUSER_PERF_SCOPE("CutlassCompiledKernel::loadKernel");
  
  if (compile_options_.use_nvrtc) {
    // Load PTX module
    NVFUSER_CUDA_DRIVER_CHECK(cuModuleLoadData(
        &cuda_module_, binary_.data()));
    
    // Get function
    NVFUSER_CUDA_DRIVER_CHECK(cuModuleGetFunction(
        &cuda_function_,
        cuda_module_,
        descriptor_.kernel_name.c_str()));
  } else {
    // Load shared library
    shared_library_handle_ = dlopen(temp_dir_.c_str(), RTLD_NOW);
    if (!shared_library_handle_) {
      NVF_THROW("Failed to load shared library: ", dlerror());
    }
    
    // For shared library approach, we would need a wrapper function
    // that can be called directly. This is a simplified version.
    // In practice, you'd need to generate a C-style wrapper.
    NVF_THROW("Shared library loading not fully implemented yet");
  }
}

void CutlassCompiledKernel::generateKernelArguments() {
  FUSER_PERF_SCOPE("CutlassCompiledKernel::generateKernelArguments");
  
  // This would generate the kernel arguments based on the fusion inputs/outputs
  // and the CUTLASS kernel requirements. For now, this is a placeholder.
  
  // Calculate total size needed for arguments
  size_t total_size = 0;
  for (const auto& size : descriptor_.argument_sizes) {
    total_size += size;
  }
  
  kernel_args_buffer_.resize(total_size);
  kernel_arg_pointers_.resize(descriptor_.argument_sizes.size());
  
  // Set up pointers
  size_t offset = 0;
  for (size_t i = 0; i < descriptor_.argument_sizes.size(); ++i) {
    kernel_arg_pointers_[i] = kernel_args_buffer_.data() + offset;
    offset += descriptor_.argument_sizes[i];
  }
}

void CutlassCompiledKernel::createLaunchParams() {
  FUSER_PERF_SCOPE("CutlassCompiledKernel::createLaunchParams");
  
  // Use the launch configuration from the descriptor
  launch_params_ = LaunchParams(
      descriptor_.grid_dim.x,
      descriptor_.grid_dim.y,
      descriptor_.grid_dim.z,
      descriptor_.block_dim.x,
      descriptor_.block_dim.y,
      descriptor_.block_dim.z);
  
  // Set shared memory size if needed
  if (descriptor_.shared_memory_size > 0) {
    launch_params_.smem = descriptor_.shared_memory_size;
  }
}

float CutlassCompiledKernel::run(
    const KernelArgumentHolder& args,
    cudaStream_t stream) {
  FUSER_PERF_SCOPE("CutlassCompiledKernel::run");
  
  NVF_CHECK(isCompiled(), "Kernel must be compiled before running");
  
  c10::cuda::CUDAGuard guard(fusion_->device());
  auto stream_to_use = stream ? stream : at::cuda::getCurrentCUDAStream();
  
  // Prepare kernel arguments
  // This is simplified - actual implementation would need to properly
  // marshal arguments according to the CUTLASS kernel interface
  
  // Launch kernel
  AT_CUDA_DRIVER_CHECK(at::globalContext().getNVRTC().cuLaunchKernel(
      cuda_function_,
      descriptor_.grid_dim.x,
      descriptor_.grid_dim.y, 
      descriptor_.grid_dim.z,
      descriptor_.block_dim.x,
      descriptor_.block_dim.y,
      descriptor_.block_dim.z,
      descriptor_.shared_memory_size,
      stream_to_use,
      kernel_arg_pointers_.data(),
      nullptr));
  
  // Synchronize and measure time
  c10::cuda::CUDAGuard device_guard(fusion_->device());
  cudaEvent_t start_event = {};
  cudaEvent_t finish_event = {};
  
  NVFUSER_CUDA_SAFE_CALL(cudaEventCreate(&start_event));
  NVFUSER_CUDA_SAFE_CALL(cudaEventCreate(&finish_event));
  
  NVFUSER_CUDA_SAFE_CALL(cudaEventRecord(start_event, stream_to_use));
  
  // Kernel is already launched above
  
  NVFUSER_CUDA_SAFE_CALL(cudaEventRecord(finish_event, stream_to_use));
  NVFUSER_CUDA_SAFE_CALL(cudaEventSynchronize(finish_event));
  
  float kernel_time_ms = 0;
  NVFUSER_CUDA_SAFE_CALL(
      cudaEventElapsedTime(&kernel_time_ms, start_event, finish_event));
  
  NVFUSER_CUDA_SAFE_CALL(cudaEventDestroy(start_event));
  NVFUSER_CUDA_SAFE_CALL(cudaEventDestroy(finish_event));
  
  return kernel_time_ms;
}

// CutlassCodeGenerator implementation
std::string CutlassCodeGenerator::generateCode(
    Fusion* fusion,
    const CutlassParams& params,
    CutlassKernelDescriptor& descriptor) {
  FUSER_PERF_SCOPE("CutlassCodeGenerator::generateCode");
  
  std::stringstream code;
  
  // Generate includes
  code << generateIncludes();
  code << "\n";
  
  // TODO: Analyze fusion to determine CUTLASS operation type
  // For now, assume it's a scaled GEMM
  descriptor.kernel_name = "cutlass_kernel_" + std::to_string(fusion->id());
  descriptor.operation_type = "cutlass::gemm::device::Gemm";
  
  // Generate kernel definition
  code << generateKernelDefinition(descriptor);
  code << "\n";
  
  // Generate launch wrapper
  code << generateLaunchWrapper(descriptor);
  
  return code.str();
}

std::string CutlassCodeGenerator::generateIncludes() {
  std::stringstream includes;
  
  includes << "#include <cutlass/cutlass.h>\n";
  includes << "#include <cutlass/gemm/device/gemm.h>\n";
  includes << "#include <cutlass/util/host_tensor.h>\n";
  includes << "#include <cutlass/util/reference/device/gemm.h>\n";
  includes << "#include <cutlass/util/reference/host/tensor_fill.h>\n";
  includes << "#include <cutlass/util/tensor_view_io.h>\n";
  includes << "#include <cuda_runtime.h>\n";
  
  // For EVT (Epilogue Visitor Tree)
  includes << "#include <cute/tensor.hpp>\n";
  includes << "#include <cutlass/epilogue/collective/collective_builder.hpp>\n";
  includes << "#include <cutlass/epilogue/collective/default_epilogue.hpp>\n";
  includes << "#include <cutlass/epilogue/thread/linear_combination.h>\n";
  
  return includes.str();
}

std::string CutlassCodeGenerator::generateKernelDefinition(
    const CutlassKernelDescriptor& descriptor) {
  std::stringstream kernel;
  
  // This is a simplified example - actual implementation would need to
  // generate proper CUTLASS kernel instantiation based on the fusion
  
  kernel << "// CUTLASS kernel definition\n";
  kernel << "using Gemm = cutlass::gemm::device::Gemm<\n";
  kernel << "    cutlass::half_t,\n";  // ElementA
  kernel << "    cutlass::layout::RowMajor,\n";  // LayoutA
  kernel << "    cutlass::half_t,\n";  // ElementB
  kernel << "    cutlass::layout::ColumnMajor,\n";  // LayoutB
  kernel << "    cutlass::half_t,\n";  // ElementC
  kernel << "    cutlass::layout::RowMajor,\n";  // LayoutC
  kernel << "    float,\n";  // ElementAccumulator
  kernel << "    cutlass::arch::OpClassTensorOp,\n";
  kernel << "    cutlass::arch::Sm80,\n";
  kernel << "    cutlass::gemm::GemmShape<128, 128, 32>,\n";  // ThreadblockShape
  kernel << "    cutlass::gemm::GemmShape<64, 64, 32>,\n";   // WarpShape
  kernel << "    cutlass::gemm::GemmShape<16, 8, 16>,\n";    // InstructionShape
  kernel << "    cutlass::epilogue::thread::LinearCombination<\n";
  kernel << "        cutlass::half_t,\n";
  kernel << "        128 / cutlass::sizeof_bits<cutlass::half_t>::value,\n";
  kernel << "        float,\n";
  kernel << "        float>,\n";
  kernel << "    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,\n";
  kernel << "    3>;\n";  // Stages
  
  return kernel.str();
}

std::string CutlassCodeGenerator::generateEpilogueVisitorTree(
    Fusion* fusion,
    const CutlassParams& params) {
  // TODO: Implement EVT generation based on fusion epilogue operations
  // This would analyze the fusion graph after the matmul to generate
  // the appropriate EVT structure
  return "";
}

std::string CutlassCodeGenerator::generateLaunchWrapper(
    const CutlassKernelDescriptor& descriptor) {
  std::stringstream wrapper;
  
  wrapper << "extern \"C\" __global__ void " << descriptor.kernel_name << "(\n";
  wrapper << "    const void* A,\n";
  wrapper << "    const void* B,\n";
  wrapper << "    const void* C,\n";
  wrapper << "    void* D,\n";
  wrapper << "    int M,\n";
  wrapper << "    int N,\n";
  wrapper << "    int K,\n";
  wrapper << "    float alpha,\n";
  wrapper << "    float beta) {\n";
  wrapper << "  // CUTLASS kernel launch wrapper\n";
  wrapper << "  // This is a placeholder - actual implementation would\n";
  wrapper << "  // instantiate and run the CUTLASS kernel\n";
  wrapper << "}\n";
  
  return wrapper.str();
}

std::string CutlassCodeGenerator::mapDataTypeToCutlass(DataType dtype) {
  switch (dtype) {
    case DataType::Half:
      return "cutlass::half_t";
    case DataType::BFloat16:
      return "cutlass::bfloat16_t";
    case DataType::Float:
      return "float";
    case DataType::Double:
      return "double";
    case DataType::Int32:
      return "int32_t";
    default:
      NVF_THROW("Unsupported data type for CUTLASS: ", dtype);
  }
}

std::string CutlassCodeGenerator::mapLayoutToCutlass(const TensorView* tv) {
  // Simplified - actual implementation would analyze the tensor's
  // allocation domain and stride order
  return "cutlass::layout::RowMajor";
}

} // namespace nvfuser