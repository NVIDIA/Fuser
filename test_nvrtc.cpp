#include <fstream>
#include <iostream>
#include <regex>
#include <vector>

#include <cuda.h>
#include <nvrtc.h>

void checkNvrtc(nvrtcResult result) {
  if (result != NVRTC_SUCCESS) {
    std::cerr << "nvrtc error: " << nvrtcGetErrorString(result) << std::endl;
    abort();
  }
}

void checkCuda(CUresult result) {
  if (result != CUDA_SUCCESS) {
    const char* error_string;
    const char* error_name;
    cuGetErrorName(result, &error_name);
    cuGetErrorString(result, &error_string);
    std::cerr << "cuda error: " << error_name << ", " << error_string
              << std::endl;
    abort();
  }
}

std::string readFileContent(const std::string& path) {
  std::ifstream file(path);
  if (!file.is_open()) {
    std::cerr << "Error opening file: " << path << std::endl;
    abort();
  }

  std::string content;
  std::string line;
  while (std::getline(file, line)) {
    content += line + "\n";
  }
  return content;
}

std::vector<const char*> getCompileOptions() {
  // Dumped from NvrtcCompileDriver::getOptions.
  std::vector<const char*> options;
  options.push_back("--std=c++17");
  options.push_back("--diag-suppress=177");
  options.push_back("--gpu-architecture=sm_80");
  options.push_back("-default-device");
  options.push_back("--fmad=true");
  options.push_back("-DNDEBUG");
  options.push_back("--maxrregcount=255");
  return options;
}

int main(int argc, char* argv[]) {
  checkCuda(cuInit(0));

  CUdevice device;
  checkCuda(cuDeviceGet(&device, 0));

  CUcontext context;
  checkCuda(cuCtxCreate(&context, 0, device));
  checkCuda(cuCtxSetCurrent(context));

  int nvrtc_major, nvrtc_minor;
  checkNvrtc(nvrtcVersion(&nvrtc_major, &nvrtc_minor));
  std::cout << "nvrtc version: " << nvrtc_major << "." << nvrtc_minor
            << std::endl;

  if (1 >= argc) {
    std::cerr << "Missing the path to the source code." << std::endl;
    abort();
  }
  std::string source_code = readFileContent(argv[1]);

  nvrtcProgram program;
  checkNvrtc(nvrtcCreateProgram(
      &program,
      source_code.c_str(),
      /*name=*/nullptr,
      /*numHeaders=*/0,
      /*headers=*/nullptr,
      /*includeNames=*/nullptr));

  std::regex regex("__global__\\s+void\\s+(.*)\\(");
  std::smatch match;
  if (!std::regex_search(source_code, match, regex)) {
    std::cerr << "Cannot find the kernel name." << std::endl;
    abort();
  }
  std::string kernel_name = match[1];
  checkNvrtc(nvrtcAddNameExpression(program, kernel_name.c_str()));

  std::vector<const char*> options = getCompileOptions();
  nvrtcResult result = nvrtcCompileProgram(
      program, static_cast<int>(options.size()), options.data());
  if (result != NVRTC_SUCCESS) {
    size_t log_size;
    checkNvrtc(nvrtcGetProgramLogSize(program, &log_size));

    std::vector<char> log_buffer(log_size);
    checkNvrtc(nvrtcGetProgramLog(program, log_buffer.data()));
    std::cerr << log_buffer.data() << std::endl;
  }
  checkNvrtc(result);

  const char* lowered_kernel_name = nullptr;
  checkNvrtc(
      nvrtcGetLoweredName(program, kernel_name.c_str(), &lowered_kernel_name));

  size_t cubin_size = 0;
  checkNvrtc(nvrtcGetCUBINSize(program, &cubin_size));
  std::vector<char> cubin_buffer(cubin_size);
  checkNvrtc(nvrtcGetCUBIN(program, cubin_buffer.data()));

  CUmodule module;
  checkCuda(
      cuModuleLoadDataEx(&module, cubin_buffer.data(), 0, nullptr, nullptr));

  CUfunction function;
  std::cout << "Loading kernel: " << lowered_kernel_name << std::endl;
  checkCuda(cuModuleGetFunction(&function, module, lowered_kernel_name));

  checkCuda(cuModuleUnload(module));

  checkCuda(cuCtxDestroy(context));

  std::cout << "Done." << std::endl;
  return 0;
}
