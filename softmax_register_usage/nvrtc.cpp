#include <cstring>
#include <cuda_runtime.h>
#include <iostream>
#include <vector>
// CUDA kernel
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <nvrtc.h>
#include <cuda_fp16.h>

#define CUDA_SAFE_CALL(x)                                               \
  do {                                                                  \
    cudaError_t _result = x;                                            \
    if (_result != cudaSuccess) {                                       \
      std::cerr << "CUDA error: " << cudaGetErrorName(_result)          \
                << " failed with error " << cudaGetErrorString(_result) \
                << std::endl;                                           \
      std::exit(1);                                                     \
    }                                                                   \
  } while (0)

std::string readKernelSource(const char* filename) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Failed to open file: " << filename << std::endl;
        exit(EXIT_FAILURE);
    }
    std::stringstream buffer;
    buffer << file.rdbuf();
    return buffer.str();
}

void allocSet(void* ptr, size_t mem_size){
  CUDA_SAFE_CALL(cudaMalloc(&ptr, mem_size));
  CUDA_SAFE_CALL(cudaMemset(ptr, 0, mem_size));
}

// aligned register array for vectorized load/store
template <typename scalar_t, int size, int align_size = 1>
struct alignas(sizeof(scalar_t) * align_size) Array {
  scalar_t array[size];
  __device__ scalar_t& operator[](const unsigned int i) {
    return array[i];
  }
  __device__ const scalar_t& operator[](const unsigned int i) const {
    return array[i];
  }
};
template <typename T, int Dims, int AllocDims = Dims>
struct Tensor {
  __device__ T& operator[](int ind) {
    return data[ind];
  };

  T* data;
  Array<int, Dims, 1> logical_size;
  Array<int, AllocDims, 1> alloc_stride;
};
int run(nvrtcProgram& prog){

  // set up para
  const int batch_size = 2048;
  const int hidden_size = 128*1024;
  const dim3 bdim(1024, 1, 1);
  const dim3 gdim(batch_size, 1, 1);
  __half *d0, *d1, *d2, *d28;
  float  *d40, *d42;

  // x and ln(x)
  size_t mem_size_half_2d = batch_size * hidden_size * sizeof(float) / 2;
  allocSet(d0, mem_size_half_2d);
  allocSet(d28, mem_size_half_2d);
  Tensor<__half, 2, 2> T0; T0.data = d0;
  Tensor<__half, 2, 2> T28; T28.data = d28;

  // weight and bias
  size_t mem_size_half_1d = hidden_size * sizeof(float) / 2;
  allocSet(d1, mem_size_half_1d);
  allocSet(d2, mem_size_half_1d);
  Tensor<__half, 1, 1> T1; T1.data = d1;
  Tensor<__half, 1, 1> T2; T2.data = d2;

  // mean and var
  size_t mem_size_float_1d = batch_size * 1 * sizeof(float);
  size_t mem_size_float_2d = batch_size * 1 * sizeof(float);
  allocSet(d40, mem_size_float_1d);
  allocSet(d42, mem_size_float_1d);
  Tensor<float, 1, 1> T40; T40.data = d40;
  Tensor<float, 2, 2> T42; T42.data = d42;
  {
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        std::cerr << "cuda memory allocation failed: " << cudaGetErrorString(error) << std::endl;
        return 1;
    }
  }
  // smem for reduction & broadcast
  int smem_size = sizeof(float)  * bdim.x*2;

  // Get PTX code from the program
  size_t ptxSize;
  nvrtcGetPTXSize(prog, &ptxSize);
  std::string ptx(ptxSize, '\0');
  nvrtcGetPTX(prog, &ptx[0]);
  std::ofstream ptxFile("kernel.ptx");
  if (!ptxFile.is_open()) {
      std::cerr << "Failed to open file for writing PTX." << std::endl;
      // Handle error...
  } else {
      ptxFile << ptx;
      ptxFile.close();
      std::cout << "PTX code has been written to kernel.ptx" << std::endl;
  }
  // Load PTX and get kernel function
  CUmodule module;
  CUfunction kernel;
  cuModuleLoadData(&module, ptx.c_str());
  cuModuleGetFunction(&kernel, module, "nvfuser_inner_persistent_f0_c1_r0_g0");

  // Assuming kernel prototype looks something like:
  // __global__ void myKernel(Tensor<__half, 2, 2> T0, Tensor<__half, 1, 1> T1, Tensor<__half, 1, 1> T2, Tensor<__half, 2, 2> T28, Tensor<float, 1, 1> T40, Tensor<float, 2, 2> T42)
  // Setup kernel arguments array
  void* args[] = { &T0, &T1, &T2, &T28, &T40, &T42 };
  cuLaunchKernel(kernel,
                gdim.x, gdim.y, gdim.z, // Grid dimensions
                bdim.x, bdim.y, bdim.z, // Block dimensions
                smem_size, // Shared memory size
                NULL, // Stream
                args, // Kernel arguments
                NULL); // Extra options
  cudaError_t error = cudaGetLastError();
  if (error != cudaSuccess) {
      std::cerr << "Kernel launch failed: " << cudaGetErrorString(error) << std::endl;
      return 1;
  }
  std::cout << "cuLaunchKernel success." << std::endl;
  return 0;
}

int main(int argc, char** argv) {
  if (argc != 2) {
      std::cerr << "Usage: " << argv[0] << " <kernel file>" << std::endl;
      return -1;
  }
  // Get the NVRTC version
  int major, minor;
  nvrtcResult result = nvrtcVersion(&major, &minor);
  if (result == NVRTC_SUCCESS) {
      std::cout << "NVRTC Version: " << major << "." << minor << std::endl;
  } else {
      std::cerr << "Failed to get NVRTC version." << std::endl;
  }

  if (cuInit(0) != CUDA_SUCCESS) {
      std::cerr << "Failed to initialize CUDA Driver API." << std::endl;
      return -1;
  }
  CUdevice device;
  cuDeviceGet(&device, 0);
  CUcontext context;
  cuCtxCreate(&context, 0, device);

  // Read the kernel source code from the file
  std::string kernelSource = readKernelSource(argv[1]);
  const char* cudaSourceCode = kernelSource.c_str();
  nvrtcProgram prog;
  nvrtcCreateProgram(&prog, cudaSourceCode, "kernel.cu", 0, nullptr, nullptr);

  int max_reg = 64;
  if (getenv("MAX_REG")) {
    max_reg = std::atoi(getenv("MAX_REG"));
  }
  std::string max_reg_arg = "--maxrregcount=" + std::to_string(max_reg);

  std::vector<std::string> compile_options = {
      "--gpu-architecture=sm_90",
      "--ptxas-options=-v",
      "-default-device",
      "--std=c++17",
      "--diag-suppress=177",
      "--fmad=true",
      max_reg_arg,
      "-DNDEBUG"};

  if (getenv("AVOID_EXP")) {
    std::string avoid_exp = "-DAVOID_EXP";
    compile_options.push_back(avoid_exp);
  }

  std::cout << "Options:";
  for (const auto& opt : compile_options) {
    std::cout << " " << opt;
  }
  std::cout << std::endl;

  size_t numOptions = compile_options.size();
  {
    const char* compile_opts_for_nvrtc[numOptions];
    for (size_t i = 0; i < numOptions; ++i) {
      compile_opts_for_nvrtc[i] = compile_options.at(i).c_str();
    }
    nvrtcResult compileResult = nvrtcCompileProgram(prog, numOptions, compile_opts_for_nvrtc);
    if (compileResult != NVRTC_SUCCESS) {
      std::cerr << "Failed to compile CUDA program." << std::endl;
      nvrtcDestroyProgram(&prog);
      return 1;
    }
  }

  // Check results and print the compilation log in all cases
  size_t logSize;
  nvrtcGetProgramLogSize(prog, &logSize);
  std::vector<char> log(logSize);
  nvrtcGetProgramLog(prog, log.data());

  // Check compilation result after printing the log
  std::cout << "Compilation log:\n" << log.data() << std::endl;

  // optional to run the kernel
  //run(prog);

  cudaError_t err = cudaDeviceSynchronize();
  if (err != cudaSuccess) {
      std::cerr << "CUDA Error: " << cudaGetErrorString(err) << std::endl;
      // Handle error
  }

  // Destroy the program in case of success as well
  nvrtcDestroyProgram(&prog);
  return 0;

}

////////////////////////////
// (0) replace --gpu-architecture=sm_90 with appropriate target.
// (1) nvcc nvrtc.cpp -lnvrtc -lcuda
// (2) ./a.out ln_128k.cu


// ptxas info    : 3 bytes gmem
// ptxas info    : Compiling entry function '_ZN46_GLOBAL__N__00000000_9_kernel_cu_ad01352a_715536nvfuser_inner_persistent_f0_c1_r0_g0ENS_6TensorINS_6__halfELi2ELi2EEENS0_IS1_Li1ELi1EEES3_S2_NS0_IfLi1ELi1EEENS0_IfLi2ELi2EEE' for 'sm_90'
// ptxas info    : Function properties for _ZN46_GLOBAL__N__00000000_9_kernel_cu_ad01352a_715536nvfuser_inner_persistent_f0_c1_r0_g0ENS_6TensorINS_6__halfELi2ELi2EEENS0_IS1_Li1ELi1EEES3_S2_NS0_IfLi1ELi1EEENS0_IfLi2ELi2EEE
// ptxas         .     128 bytes stack frame, 492 bytes spill stores, 512 bytes spill loads
// ptxas info    : Used 64 registers, 16 bytes smem
