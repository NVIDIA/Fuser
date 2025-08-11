// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on

#include <cuda.h>
#include <cuda_runtime.h>
#include <dlfcn.h>
#include <filesystem>
#include <fstream>
#include <instrumentation.h>
#include <memory>
#include <sstream>
#include <string>
#include <vector>
#include <unistd.h>

#include <ATen/ATen.h>
#include <c10/cuda/CUDACachingAllocator.h>
#include <c10/cuda/CUDAFunctions.h>
#include <c10/cuda/CUDAMathCompat.h>
#include <c10/util/Exception.h>

#include <fusion.h>
#include <ir/all_nodes.h>
#include <ops/all_ops.h>
#include <runtime/cutlass_compiled_kernel.h>
#include <runtime/executor_kernel_arg.h>
#include <runtime/executor_params.h>
#include <scheduler/cutlass.h>
#include <scheduler/scheduler_types.h>
#include <utils.h>

namespace nvfuser {

namespace {

// Get CUTLASS include path from environment or use default
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
    cudaError_t result = cudaGetDeviceProperties(&prop, device_id);
    if (result != cudaSuccess) {
      NVF_THROW("Failed to get device properties: ", cudaGetErrorString(result));
    }
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

  // Compile the code using nvcc
  compileWithNVCC();

  // Load the compiled kernel
  loadKernel();

  // Generate kernel arguments structure
  generateKernelArguments();

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

  // Use base class run method if available, otherwise implement CUTLASS-specific run
  // For now, we'll implement a basic CUTLASS kernel launch
  cudaEvent_t start_event = {};
  cudaEvent_t finish_event = {};

  NVFUSER_CUDA_RT_SAFE_CALL(cudaEventCreate(&start_event));
  NVFUSER_CUDA_RT_SAFE_CALL(cudaEventCreate(&finish_event));

  NVFUSER_CUDA_RT_SAFE_CALL(cudaEventRecord(start_event, stream));

  // Launch the CUTLASS kernel
  if (cuda_function_) {
    // For nvfp4_scaled_mm_kernel, we need to call it as a C function
    // Extract arguments for nvfp4_scaled_mm_kernel
    // Expected order: output, a, b, scales_a, scales_b, alpha, m, n, k
    if (args.size() < 9) {
      NVF_THROW("Expected at least 9 arguments for nvfp4_scaled_mm_kernel");
    }

    // Get tensors from arguments
    auto output = args[0].as<at::Tensor>();
    auto a = args[1].as<at::Tensor>();
    auto b = args[2].as<at::Tensor>();
    auto scales_a = args[3].as<at::Tensor>();
    auto scales_b = args[4].as<at::Tensor>();
    auto alpha = args[5].as<at::Tensor>();
    
    // Get dimensions
    auto m = args[6].as<int64_t>();
    auto n = args[7].as<int64_t>();
    auto k = args[8].as<int64_t>();

    // Define the function signature for the kernel
    using KernelFunc = void(*)(at::Tensor&, const at::Tensor&, const at::Tensor&,
                              const at::Tensor&, const at::Tensor&, const at::Tensor&,
                              int64_t, int64_t, int64_t, cudaStream_t);
    
    auto kernel_func = reinterpret_cast<KernelFunc>(cuda_function_);
    
    // Call the kernel
    kernel_func(output, a, b, scales_a, scales_b, alpha, m, n, k, stream);
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
  cutlass_code_ = CutlassCodeGenerator::generateCode(
      fusion_, cutlass_params_, descriptor_);
}

void CutlassCompiledKernel::generateCutlassCode() {
  FUSER_PERF_SCOPE("CutlassCompiledKernel::generateCutlassCode");

  // Generate CUTLASS kernel code using the code generator
  cutlass_code_ = CutlassCodeGenerator::generateCode(
      fusion_, cutlass_params_, descriptor_);
}

void CutlassCompiledKernel::compileWithNVCC() {
  FUSER_PERF_SCOPE("CutlassCompiledKernel::compileWithNVCC");

  // Create temporary directory for compilation
  temp_dir_ = std::filesystem::temp_directory_path() / ("cutlass_compile_" + std::to_string(getpid()));
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
  std::string compile_cmd = "nvcc -shared -o " + output_file.string() + " " + source_file.string();
  
  // Add compute capability
  compile_cmd += " -arch=" + getComputeCapabilityString(compile_options_.compute_capability);
  
  // Add C++ standard
  compile_cmd += " -std=c++17";
  
  // Add optimization level
  compile_cmd += " -O" + std::to_string(compile_options_.optimization_level);
  
  // Add CUTLASS include paths
  compile_cmd += " -I" + getCutlassIncludePath();
  for (const auto& path : compile_options_.include_paths) {
    compile_cmd += " -I" + path;
  }
  
  // Add defines
  for (const auto& define : compile_options_.defines) {
    compile_cmd += " -D" + define;
  }

  // Add debug flags if needed
  if (compile_options_.debug) {
    compile_cmd += " -G -lineinfo";
  }

  // Execute nvcc compilation and capture output
  std::filesystem::path output_file_path = temp_dir_ / "nvcc_output.txt";
  std::string full_cmd = compile_cmd + " 2>&1 > " + output_file_path.string();
  int result = system(full_cmd.c_str());
  
  if (result != 0) {
    // Read compilation output for error details
    std::ifstream output_file(output_file_path);
    std::string output_content((std::istreambuf_iterator<char>(output_file)),
                              std::istreambuf_iterator<char>());
    output_file.close();
    
    NVF_THROW("nvcc compilation failed with code: ", result, 
               "\nCommand: ", compile_cmd,
               "\nOutput: ", output_content);
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
        dlsym(shared_library_handle_, descriptor_.kernel_name.c_str()));
    if (!cuda_function_) {
      NVF_THROW("Failed to get CUTLASS kernel function: ", dlerror());
    }
  }
}

void CutlassCompiledKernel::generateKernelArguments() {
  // Generate kernel arguments structure for CUTLASS
  // This would be specific to the CUTLASS kernel being generated
  kernel_args_buffer_.resize(1024); // Placeholder size
  kernel_arg_pointers_.clear();
}

void CutlassCompiledKernel::createLaunchParams() {
  // For CUTLASS kernels, the launch parameters are typically computed
  // based on the problem size and tile configuration
  // For nvfp4 scaled matmul, we use the tile configuration from the kernel
  
  // Block dimensions based on CUTLASS tile configuration
  // The kernel uses 256x256 tiles with 4x4 warps
  int block_dim_x = cutlass_params_.num_warps_m * 32;  // 4 * 32 = 128
  int block_dim_y = cutlass_params_.num_warps_n * 32;  // 4 * 32 = 128
  int block_dim_z = 1;
  
  // Grid dimensions will be computed at runtime based on problem size
  // For now, use placeholder values that will be updated during kernel launch
  launch_params_ = LaunchParams(
      LaunchParams::UNINITIALIZED_VAL,  // gdimx - computed at runtime
      LaunchParams::UNINITIALIZED_VAL,  // gdimy - computed at runtime
      1,  // gdimz
      block_dim_x,
      block_dim_y,
      block_dim_z
  );
}

// CutlassCodeGenerator implementation
std::string CutlassCodeGenerator::generateCode(
    Fusion* fusion,
    const CutlassParams& params,
    CutlassKernelDescriptor& descriptor) {
  
  // Set up descriptor for nvfp4 scaled matmul
  descriptor.kernel_name = "nvfp4_scaled_mm_kernel";
  descriptor.operation_type = "nvfp4_scaled_mm";
  
  // Generate the same kernel code as nvfp4_scaled_mm.cu
  return generateNvfp4ScaledMmKernel(fusion, params, descriptor);
}

std::string CutlassCodeGenerator::generateNvfp4ScaledMmKernel(
    Fusion* fusion,
    const CutlassParams& params,
    CutlassKernelDescriptor& descriptor) {
  
  std::string code = R"(
// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on

#include <cutlass_utils.h>
#include <exceptions.h>
#include <nvf_cutlass.h>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/torch.h>

#include "cutlass/cutlass.h"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/util/packed_stride.hpp"

namespace nvfuser::cutlass_kernels {

namespace {

using namespace cute;

#if defined(CUTLASS_ARCH_MMA_SM100_SUPPORTED)
// Kernel configuration traits for different output data types
// Defines tile shapes and cluster configurations.
template <typename T>
struct KernelTraits;

// Kernel traits for FP16 output
template <>
struct KernelTraits<cutlass::half_t> {
  using MmaTileShape = Shape<_256, _256, _256>;
  using ClusterShape = Shape<_4, _4, _1>;
  using PerSmTileShape_MNK = Shape<_128, _256, _256>;
};

// Kernel traits for BF16 output
template <>
struct KernelTraits<cutlass::bfloat16_t> {
  using MmaTileShape = Shape<_256, _256, _256>;
  using ClusterShape = Shape<_4, _4, _1>;
  using PerSmTileShape_MNK = Shape<_128, _256, _256>;
};

// Main GEMM configuration for NVFP4 scaled matrix multiplication on SM100+
// Defines all the types, layouts, and configurations needed for the CUTLASS
// kernel
template <typename T>
struct Fp4GemmSm100 {
  // A matrix configuration
  using ElementA = cutlass::nv_float4_t<cutlass::float_e2m1_t>;
  using LayoutATag = cutlass::layout::RowMajor;
  static constexpr int AlignmentA = 32;

  // B matrix configuration
  using ElementB = cutlass::nv_float4_t<cutlass::float_e2m1_t>;
  using LayoutBTag = cutlass::layout::ColumnMajor;
  static constexpr int AlignmentB = 32;

  // C/D matrix configuration
  using ElementD = T;
  using ElementC = T;
  using LayoutCTag = cutlass::layout::RowMajor;
  using LayoutDTag = cutlass::layout::RowMajor;
  static constexpr int AlignmentD = 128 / cutlass::sizeof_bits<ElementD>::value;
  static constexpr int AlignmentC = 128 / cutlass::sizeof_bits<ElementC>::value;
  // Kernel functional config
  using ElementAccumulator = float;
  using ArchTag = cutlass::arch::Sm100;
  using OperatorClass = cutlass::arch::OpClassBlockScaledTensorOp;

  // Kernel Perf config
  using MmaTileShape = typename KernelTraits<T>::MmaTileShape;
  using ClusterShape = typename KernelTraits<T>::ClusterShape;
  using PerSmTileShape_MNK = typename KernelTraits<T>::PerSmTileShape_MNK;

  using CollectiveEpilogue =
      typename cutlass::epilogue::collective::CollectiveBuilder<
          ArchTag,
          OperatorClass,
          PerSmTileShape_MNK,
          ClusterShape,
          cutlass::epilogue::collective::EpilogueTileAuto,
          ElementAccumulator,
          ElementAccumulator,
          ElementC,
          LayoutCTag,
          AlignmentC,
          ElementD,
          LayoutDTag,
          AlignmentD,
          cutlass::epilogue::collective::EpilogueScheduleAuto>::CollectiveOp;

  using CollectiveMainloop =
      typename cutlass::gemm::collective::CollectiveBuilder<
          ArchTag,
          OperatorClass,
          ElementA,
          LayoutATag,
          AlignmentA,
          ElementB,
          LayoutBTag,
          AlignmentB,
          ElementAccumulator,
          MmaTileShape,
          ClusterShape,
          cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(
              sizeof(typename CollectiveEpilogue::SharedStorage))>,
          cutlass::gemm::collective::KernelScheduleAuto>::CollectiveOp;

  using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
      Shape<int, int, int, int>,
      CollectiveMainloop,
      CollectiveEpilogue,
      void>;
  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
  using StrideA = typename Gemm::GemmKernel::StrideA;
  using LayoutA = decltype(cute::make_layout(make_shape(0, 0, 0), StrideA{}));
  using LayoutSFA = typename Gemm::GemmKernel::CollectiveMainloop::LayoutSFA;
  using StrideB = typename Gemm::GemmKernel::StrideB;
  using LayoutB = decltype(cute::make_layout(make_shape(0, 0, 0), StrideB{}));
  using LayoutSFB = typename Gemm::GemmKernel::CollectiveMainloop::LayoutSFB;
  using StrideC = typename Gemm::GemmKernel::StrideC;
  using LayoutC = decltype(cute::make_layout(make_shape(0, 0, 0), StrideC{}));
  using StrideD = typename Gemm::GemmKernel::StrideD;
  using LayoutD = decltype(cute::make_layout(make_shape(0, 0, 0), StrideD{}));
};

// Constructs CUTLASS GEMM arguments from PyTorch tensors and dimensions
//
// This function converts PyTorch tensor data and metadata into the format
// expected by CUTLASS GEMM kernels, including proper stride calculations
// and layout configurations for the scaled matrix multiplication.
//
// Parameters:
//   output: Output tensor for storing results
//   a: Input matrix A in NVFP4 format
//   b: Input matrix B in NVFP4 format
//   scales_a: Per-block scaling factors for matrix A
//   scales_b: Per-block scaling factors for matrix B
//   alpha: Global scaling factor
//   M, N, K: Matrix dimensions
//
// Returns: CUTLASS GEMM arguments structure ready for kernel execution
template <typename T>
typename T::Gemm::Arguments args_from_options(
    at::Tensor& output,
    const at::Tensor& a,
    const at::Tensor& b,
    const at::Tensor& scales_a,
    const at::Tensor& scales_b,
    const at::Tensor& alpha,
    int64_t M,
    int64_t N,
    int64_t K) {
  using ElementA = typename T::Gemm::ElementA;
  using ElementB = typename T::Gemm::ElementB;
  using ElementSFA = cutlass::float_ue4m3_t;
  using ElementSFB = cutlass::float_ue4m3_t;
  using ElementD = typename T::Gemm::ElementD;
  using ElementCompute = float;
  using StrideA = typename T::StrideA;
  using StrideB = typename T::StrideB;
  using StrideD = typename T::StrideD;
  using Sm1xxBlkScaledConfig =
      typename T::Gemm::GemmKernel::CollectiveMainloop::Sm1xxBlkScaledConfig;

  int m = static_cast<int>(M);
  int n = static_cast<int>(N);
  int k = static_cast<int>(K);
  auto stride_A = cutlass::make_cute_packed_stride(StrideA{}, {m, k, 1});
  auto stride_B = cutlass::make_cute_packed_stride(StrideB{}, {n, k, 1});
  auto stride_D = cutlass::make_cute_packed_stride(StrideD{}, {m, n, 1});

  auto layout_SFA = Sm1xxBlkScaledConfig::tile_atom_to_shape_SFA(
      cute::make_shape(m, n, k, 1));
  auto layout_SFB = Sm1xxBlkScaledConfig::tile_atom_to_shape_SFB(
      cute::make_shape(m, n, k, 1));

  typename T::Gemm::Arguments arguments{
      cutlass::gemm::GemmUniversalMode::kGemm,
      {m, n, k, 1},
      {// Mainloop arguments
       static_cast<ElementA const*>(a.data_ptr()),
       stride_A,
       static_cast<ElementB const*>(b.data_ptr()),
       stride_B,
       static_cast<ElementSFA const*>(scales_a.data_ptr()),
       layout_SFA,
       static_cast<ElementSFB const*>(scales_b.data_ptr()),
       layout_SFB},
      {// Epilogue arguments
       {}, // epilogue.thread
       static_cast<ElementD const*>(output.data_ptr()),
       stride_D,
       static_cast<ElementD*>(output.data_ptr()),
       stride_D}};
  auto& fusion_args = arguments.epilogue.thread;
  fusion_args.alpha_ptr = static_cast<ElementCompute const*>(alpha.data_ptr());
  return arguments;
}

// Executes the FP4 scaled matrix multiplication using CUTLASS kernels
//
// This function orchestrates the GEMM operation by setting up the kernel,
// allocating workspace memory, and running the computation on the GPU.
// It handles the complete lifecycle from kernel initialization to execution.
//
// Parameters:
//   output: Output tensor to store the result
//   a, b: Input matrices in FP4 format
//   scales_a, scales_b: Per-block scaling factors
//   alpha: Global scaling factor
//   m, n, k: Matrix dimensions
//   stream: CUDA stream for asynchronous execution
template <typename T>
void runGemm(
    at::Tensor& output,
    const at::Tensor& a,
    const at::Tensor& b,
    const at::Tensor& scales_a,
    const at::Tensor& scales_b,
    const at::Tensor& alpha,
    int64_t m,
    int64_t n,
    int64_t k,
    cudaStream_t stream) {
  typename Fp4GemmSm100<T>::Gemm gemm;

  auto arguments = args_from_options<Fp4GemmSm100<T>>(
      output, a, b, scales_a, scales_b, alpha, m, n, k);

  size_t workspace_size = Fp4GemmSm100<T>::Gemm::get_workspace_size(arguments);
  auto const workspace_options =
      torch::TensorOptions().dtype(torch::kUInt8).device(a.device());
  auto workspace = torch::empty(workspace_size, workspace_options);

  auto can_implement_status = gemm.can_implement(arguments);
  NVF_CHECK(
      can_implement_status == cutlass::Status::kSuccess,
      "Failed to implement GEMM");

  auto status = gemm.initialize(arguments, workspace.data_ptr(), stream);
  NVF_CHECK(status == cutlass::Status::kSuccess, "Failed to initialize GEMM");

  status = gemm.run(arguments, workspace.data_ptr(), stream);
  NVF_CHECK(status == cutlass::Status::kSuccess, "Failed to run GEMM");
}
#else
// Fallback implementation for unsupported CUTLASS versions
// Throws an error when SM100+ CUTLASS support is not available
template <typename T>
void runGemm(
    at::Tensor& output,
    at::Tensor const& a,
    at::Tensor const& b,
    at::Tensor const& scales_a,
    at::Tensor const& b,
    at::Tensor const& alpha,
    int64_t m,
    int64_t n,
    int64_t k,
    cudaStream_t stream) {
  NVF_THROW("Unsupported CUTLASS version.");
}
#endif // defined(CUTLASS_ARCH_MMA_SM100_SUPPORTED)

// Helper function to round up to the nearest multiple of y
inline int64_t roundUp(int64_t x, int64_t y) {
  return (x + y - 1) / y * y;
}

} // namespace

// Main kernel function that will be called from nvFuser
extern "C" void nvfp4_scaled_mm_kernel(
    at::Tensor& output,
    const at::Tensor& a,
    const at::Tensor& b,
    const at::Tensor& scales_a,
    const at::Tensor& scales_b,
    const at::Tensor& alpha,
    int64_t m,
    int64_t n,
    int64_t k,
    cudaStream_t stream) {
  
  // Determine output data type
  auto out_dtype = output.scalar_type();
  
  if (out_dtype == at::ScalarType::Half) {
    runGemm<cutlass::half_t>(
        output, a, b, scales_a, scales_b, alpha, m, n, k, stream);
  } else if (out_dtype == at::ScalarType::BFloat16) {
    runGemm<cutlass::bfloat16_t>(
        output, a, b, scales_a, scales_b, alpha, m, n, k, stream);
  } else {
    NVF_THROW("Unsupported output data type of nvfp4 scaled_mm.");
  }
}

} // namespace nvfuser::cutlass_kernels
)";

  return code;
}

std::string CutlassCodeGenerator::mapDataTypeToCutlass(DataType dtype) {
  if (dtype == DataType::Float) {
    return "cutlass::half_t";
  } else if (dtype == DataType::Half) {
    return "cutlass::half_t";
  } else if (dtype == DataType::BFloat16) {
    return "cutlass::bfloat16_t";
  } else {
    return "float";
  }
}

std::string CutlassCodeGenerator::mapLayoutToCutlass(const TensorView* tv) {
  // Map nvFuser layout to CUTLASS layout
  return "cutlass::layout::RowMajor";
}

} // namespace nvfuser
