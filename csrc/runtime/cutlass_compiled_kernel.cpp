// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on

#include <cuda.h>
#include <cuda_runtime.h>
#include <debug.h>
#include <dlfcn.h>
#include <fusion.h>
#include <instrumentation.h>
#include <ir/all_nodes.h>
#include <ops/all_ops.h>
#include <options.h>
#include <runtime/cutlass_compiled_kernel.h>
#include <runtime/executor_kernel_arg.h>
#include <runtime/executor_params.h>
#include <scheduler/cutlass.h>
#include <scheduler/scheduler_types.h>
#include <unistd.h>
#include <utils.h>

#include <ATen/ATen.h>
#include <c10/cuda/CUDACachingAllocator.h>
#include <c10/cuda/CUDAFunctions.h>
#include <c10/cuda/CUDAMathCompat.h>
#include <c10/util/Exception.h>

#include <chrono>
#include <filesystem>
#include <format>
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
  return "sm_" + std::to_string(compute_capability);
}

} // namespace

CutlassCompiledKernel::CutlassCompiledKernel(
    Fusion* fusion,
    const CutlassParams& cutlass_params,
    const CutlassCompileOptions& compile_options,
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
  cutlass_code_ =
      CutlassCodeGenerator::generateCode(fusion_, cutlass_params_, descriptor_);

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

void CutlassCompiledKernel::generateCutlassCode() {
  FUSER_PERF_SCOPE("CutlassCompiledKernel::generateCutlassCode");

  // Generate CUTLASS kernel code using the code generator
  cutlass_code_ =
      CutlassCodeGenerator::generateCode(fusion_, cutlass_params_, descriptor_);
}

void CutlassCompiledKernel::compileWithNVCC() {
  FUSER_PERF_SCOPE("CutlassCompiledKernel::compileWithNVCC");

  // Create temporary directory for compilation
  temp_dir_ = std::filesystem::temp_directory_path() /
      ("cutlass_compile_" + std::to_string(getpid()));
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
  std::string compile_cmd =
      "nvcc -shared -o " + output_file.string() + " " + source_file.string();

  // Add compute capability
  compile_cmd += " -arch=" +
      getComputeCapabilityString(compile_options_.compute_capability);

  // Add optimization level
  compile_cmd += " -O" + std::to_string(compile_options_.optimization_level);

  // Add CUTLASS include paths
  const std::filesystem::path cutlass_path = getCutlassPath();
  compile_options_.include_paths.push_back(cutlass_path / "include");
  compile_options_.include_paths.push_back(
      cutlass_path / "tools" / "util" / "include");
  const std::filesystem::path torch_path = getTorchPath();
  compile_options_.include_paths.push_back(torch_path / "include");
  compile_options_.include_paths.push_back(
      torch_path / "include" / "torch" / "csrc" / "api" / "include");

  for (const auto& path : compile_options_.include_paths) {
    compile_cmd += " -I" + path;
  }

  for (const std::string arg :
       {"-std=c++17",
        "--expt-relaxed-constexpr",
        "--expt-extended-lambda",
        "-Xcompiler=-fPIC,-Wno-deprecated-declarations,-Wno-conversion,-fno-"
        "strict-aliasing"}) {
    compile_cmd += " " + arg;
  }

  for (const std::string def :
       {"CUTE_USE_PACKED_TUPLE=1",
        "CUTLASS_ENABLE_TENSOR_CORE_MMA=1",
        "CUTLASS_VERSIONS_GENERATED",
        "CUTLASS_TEST_LEVEL=0",
        "CUTLASS_TEST_ENABLE_CACHED_RESULTS=1",
        "CUTLASS_DEBUG_TRACE_LEVEL=0"}) {
    compile_options_.defines.push_back(def);
  }

  // Add defines
  for (const auto& define : compile_options_.defines) {
    compile_cmd += " -D" + define;
  }

  if (isOptionEnabled(EnableOption::KernelDebug)) {
    compile_cmd += " -G";
  }
  if (isOptionEnabled(EnableOption::KernelLineInfo)) {
    compile_cmd += " -lineinfo";
  }

  // TODO: enable dumping cubin, ptx, and sass as in CompiledKernel

  // Execute nvcc compilation and capture output
  std::filesystem::path output_file_path = temp_dir_ / "nvcc_output.txt";
  std::string full_cmd = compile_cmd + " 2>&1 > " + output_file_path.string();

  using Clock = std::chrono::steady_clock;
  Clock::time_point start_timestamp = Clock::now();
  int result = system(full_cmd.c_str());
  Clock::duration duration = Clock::now() - start_timestamp;
  debug() << "NVCC CUTLASS kernel compile time: "
          << std::chrono::duration_cast<std::chrono::seconds>(duration).count()
          << " seconds" << std::endl;

  if (result != 0) {
    // Read compilation output for error details
    std::ifstream output_file(output_file_path);
    std::string output_content(
        (std::istreambuf_iterator<char>(output_file)),
        std::istreambuf_iterator<char>());
    output_file.close();

    NVF_THROW(
        "nvcc compilation failed with code: ",
        result,
        "\nCommand: ",
        compile_cmd,
        "\nOutput: ",
        output_content);
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
    cuda_function_ = reinterpret_cast<CUfunction>(dlsym(
        shared_library_handle_, descriptor_.launch_function_name.c_str()));
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

// CutlassCodeGenerator implementation
std::string CutlassCodeGenerator::generateCode(
    Fusion* fusion,
    const CutlassParams& params,
    CutlassKernelDescriptor& descriptor) {
  // Set up descriptor for nvfp4 scaled matmul
  descriptor.kernel_name = "nvfp4_scaled_mm_kernel";
  descriptor.launch_function_name = "run_kernel";
  descriptor.operation_type = "nvfp4_scaled_mm";

  // Generate the same kernel code as nvfp4_scaled_mm.cu
  return generateNvfp4ScaledMmKernel(fusion, params, descriptor);
}

std::string CutlassCodeGenerator::generateNvfp4ScaledMmKernel(
    Fusion* fusion,
    const CutlassParams& params,
    CutlassKernelDescriptor& descriptor) {
  std::string code =
      R"(
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/torch.h>

#include "cutlass/cutlass.h"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/util/packed_stride.hpp"

#define NVF_THROW(msg) throw std::runtime_error(msg);
#define NVF_ERROR(cond, msg)                                      \
  if (!(cond)) {                                                  \
    NVF_THROW("Condition " #cond " failed: " + std::string(msg)); \
  }

#define NVFUSER_CUDA_RT_SAFE_CALL(x)               \
  do {                                             \
    cudaError_t _result = x;                       \
    NVF_ERROR(                                     \
        _result == cudaSuccess,                    \
        std::string("CUDA error: ") +              \
        std::string(cudaGetErrorName(_result)) +   \
        std::string(" failed with error ") +       \
        std::string(cudaGetErrorString(_result))); \
  } while (0)

namespace nvfuser::cutlass_kernels {

namespace {
using namespace cute;

// Kernel configuration traits for different output data types
// Defines tile shapes and cluster configurations.
template <typename T>
struct KernelTraits;

// Kernel traits for FP16 output
template <>
struct KernelTraits<cutlass::half_t> {
)";
  code += std::vformat(
      R"(
  using MmaTileShape = Shape<_{}, _{}, _{}>;
  using ClusterShape = Shape<_4, _4, _1>;
  using PerSmTileShape_MNK = Shape<_128, _256, _256>;
)",
      std::make_format_args(
          params.cta_tile.m, params.cta_tile.n, params.cta_tile.k));

  code += R"(
};

// TODO: no template needed. KernelTraits is not needed when JITing either.
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
  NVF_ERROR(
      can_implement_status == cutlass::Status::kSuccess,
      "Failed to implement GEMM");

  auto status = gemm.initialize(arguments, workspace.data_ptr(), stream);
  NVF_ERROR(status == cutlass::Status::kSuccess, "Failed to initialize GEMM");

  status = gemm.run(arguments, workspace.data_ptr(), stream);
  NVF_ERROR(status == cutlass::Status::kSuccess, "Failed to run GEMM");
}

} // namespace

// Main kernel function that will be called from nvFuser
extern "C" void run_kernel(
    const std::vector<void*>& args,
    cudaStream_t stream) {

  // TODO: determine this ordering from fusion inputs and pattern matching
  NVF_ERROR(args.size() == 6, "Expected 6 arguments");
  const at::Tensor& a = *reinterpret_cast<at::Tensor*>(args[0]);
  const at::Tensor& b = *reinterpret_cast<at::Tensor*>(args[1]);
  const at::Tensor& scales_a = *reinterpret_cast<at::Tensor*>(args[2]);
  const at::Tensor& scales_b = *reinterpret_cast<at::Tensor*>(args[3]);
  const at::Tensor& alpha = *reinterpret_cast<at::Tensor*>(args[4]);
  at::Tensor& output = *reinterpret_cast<at::Tensor*>(args[5]);

  // Determine output data type
  auto out_dtype = output.scalar_type();

  int64_t m = a.size(0);
  int64_t n = b.size(1);
  int64_t k = a.size(1);

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
