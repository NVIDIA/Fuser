// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on

#include <runtime/compiled_kernel.h>

#include <codegen.h>
#include <cuda_utils.h>
#include <debug.h>
#include <device_lower/analysis/bank_conflict.h>
#include <device_lower/lower2device.h>
#include <disjoint_set.h>
#include <driver_api.h>
#include <fusion_profiler.h>
#include <global_allocator.h>
#include <instrumentation.h>
#include <ir/all_nodes.h>
#include <ir/utils.h>
#include <iter_visitor.h>
#include <kernel_db/kernel_db.h>
#include <kernel_ir.h>
#include <multidevice/communication.h>
#include <multidevice/communicator.h>
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
#include <torch/csrc/jit/resource_guard.h>

#include <array>
#include <cmath>
#include <cstring>
#include <fstream>
#include <regex>
#include <vector>

#include <cuda_runtime.h>

#include <nvfuser_resources/argsort.h>
#include <nvfuser_resources/array.h>
#include <nvfuser_resources/basic_type_traits.h>
#include <nvfuser_resources/bf16_support.h>
#include <nvfuser_resources/bit.h>
#include <nvfuser_resources/block_reduction.h>
#include <nvfuser_resources/block_sync_atomic.h>
#include <nvfuser_resources/block_sync_default.h>
#include <nvfuser_resources/block_welford_outer.h>
#include <nvfuser_resources/broadcast.h>
#include <nvfuser_resources/casts.h>
#include <nvfuser_resources/cluster.h>
#include <nvfuser_resources/complex_number.h>
#include <nvfuser_resources/fp16_support.h>
#include <nvfuser_resources/fp4_support.h>
#include <nvfuser_resources/fp8_support.h>
#include <nvfuser_resources/fused_reduction.h>
#include <nvfuser_resources/fused_welford_helper.h>
#include <nvfuser_resources/fused_welford_impl.h>
#include <nvfuser_resources/fused_welford_impl_outer.h>
#include <nvfuser_resources/grid_broadcast.h>
#include <nvfuser_resources/grid_reduction.h>
#include <nvfuser_resources/grid_sync.h>
#include <nvfuser_resources/helpers.h>
#include <nvfuser_resources/index_utils.h>
#include <nvfuser_resources/mbarrier.h>
#include <nvfuser_resources/memory.h>
#include <nvfuser_resources/random_numbers.h>
#include <nvfuser_resources/tensor.h>
#include <nvfuser_resources/tensor_memory.h>
#include <nvfuser_resources/topk.h>
#include <nvfuser_resources/tuple.h>
#include <nvfuser_resources/type_traits.h>
#include <nvfuser_resources/warp.h>
#include <nvfuser_resources/welford.h>

namespace nvfuser {

namespace {

// Include all the functions we might need in generated code
std::string kernelPreamble() {
  std::stringstream ss;
  ss << nvfuser_resources::basic_type_traits_cu;
  ss << nvfuser_resources::bit_cu;
  ss << nvfuser_resources::complex_number_cu;

  ss << nvfuser_resources::fp16_support_cu;
  ss << nvfuser_resources::bf16_support_cu;
  ss << nvfuser_resources::fp8_support_cu;
  ss << nvfuser_resources::fp4_support_cu;

  // Base classes and helpers
  ss << nvfuser_resources::type_traits_cu;
  ss << nvfuser_resources::array_cu;
  ss << nvfuser_resources::casts_cu;
  ss << nvfuser_resources::tensor_memory_cu;
  ss << nvfuser_resources::tensor_cu;
  ss << nvfuser_resources::random_numbers_cu;
  ss << nvfuser_resources::helpers_cu;
  ss << nvfuser_resources::index_utils_cu;
  ss << nvfuser_resources::tuple_cu;

  // Synchronization classes
  if (getNvFuserEnv("USE_BLOCK_SYNC_ATOMIC")) {
    ss << nvfuser_resources::block_sync_atomic_cu;
  } else {
    ss << nvfuser_resources::block_sync_default_cu;
  }
  ss << nvfuser_resources::grid_sync_cu;
  ss << nvfuser_resources::mbarrier_cu;

  // Communication classes
  ss << nvfuser_resources::block_reduction_cu;
  ss << nvfuser_resources::grid_reduction_cu;
  ss << nvfuser_resources::grid_broadcast_cu;
  ss << nvfuser_resources::broadcast_cu;
  ss << nvfuser_resources::welford_cu;
  ss << nvfuser_resources::warp_cu;
  ss << nvfuser_resources::memory_cu;
  ss << nvfuser_resources::fused_welford_helper_cu;
  ss << nvfuser_resources::fused_reduction_cu;
  ss << nvfuser_resources::fused_welford_impl_cu;
  ss << nvfuser_resources::block_welford_outer_cu;
  ss << nvfuser_resources::fused_welford_impl_outer_cu;

  return ss.str();
}

//! Utility class to invoke nvrtcCompileProgram. Mainly for setting up
//! the c-str options.
//! TODO: Revisit if we should remove or restructure this utility function
class NvrtcCompileDriver {
 public:
  void setOption(const std::string& opt) {
    options_.push_back(opt);
  }

  const std::vector<std::string>& options() const {
    return options_;
  }

  std::string invoke(nvrtcProgram program, const std::string& src) const {
    FUSER_PERF_SCOPE("executor_utils::Nvrtc::CompileProgram");
    auto opts = getOptions();
    auto result = nvrtcCompileProgram(
        program, static_cast<int>(opts.size()), opts.data());
    size_t logsize = 0;
    NVFUSER_NVRTC_SAFE_CALL(nvrtcGetProgramLogSize(program, &logsize));
    // The log size, as returned by 'nvrtcGetProgramLogSize', appears larger
    // than its actual size by 2. This discrepancy was noticed in NVRTC
    // version 12.1. The log returned from 'nvrtcGetProgramLog' terminates with
    // a NULL character, ensuring it's safe to use 'std::vector<char>' for
    // storage before converting it to 'std::string'.
    std::vector<char> log_backing_buf(logsize);
    char* log_buf = log_backing_buf.data();
    NVFUSER_NVRTC_SAFE_CALL(nvrtcGetProgramLog(program, log_buf));
    if (result != NVRTC_SUCCESS) {
      // Print CUDA starting at generated utility
      size_t kernel_start = src.find("// Codegen generated code");
      NVF_THROW(
          "\n",
          src.substr(kernel_start),
          "\nCUDA NVRTC compile error: ",
          log_buf);
    }
    if (isDebugDumpEnabled(DebugDumpOption::PrintPtxasLog)) {
      debug() << log_buf << std::endl;
    }
    return std::string(log_buf);
  }

 private:
  // Get options that can be passed to nvrtcCompileProgram
  std::vector<const char*> getOptions() const {
    std::vector<const char*> opts(options_.size());
    for (const auto i : arange(options_.size())) {
      opts.at(i) = options_.at(i).c_str();
    }
    return opts;
  }

 private:
  std::vector<std::string> options_;
};

// Query the target GPU version number NVRTC compiles CUDA kernels for
void queryTargetGPUVersion(
    const cudaDeviceProp* const prop,
    int64_t& major,
    int64_t& minor,
    bool& compile_to_sass) {
  using CudaVersion = std::pair<int, int>;
  CudaVersion nvrtc_version;
  NVFUSER_NVRTC_SAFE_CALL(
      nvrtcVersion(&nvrtc_version.first, &nvrtc_version.second));

  NVF_CHECK(
      nvrtc_version.first >= 6,
      "NVRTC versions less than 6 are not supported. Is: ",
      nvrtc_version.first);

  // Version supported by device
  // Usually any lower version works too but is less efficient
  const CudaVersion dev_version = CudaVersion(prop->major, prop->minor);
  // Maximum version supported by the driver, cap dev_version to this
  CudaVersion max_dev_version;
  if (nvrtc_version.first <= 7) { // 7 supports 2-5.x
    max_dev_version = CudaVersion(5, 0);
  } else if (nvrtc_version.first <= 8) { // 8 supports 2-6.x
    max_dev_version = CudaVersion(6, 0);
  } else if (nvrtc_version.first <= 9) { // 9 supports 3-7.2
    max_dev_version = CudaVersion(7, 2);
  } else if (nvrtc_version.first <= 10) { // 10 supports 3-7.5
    max_dev_version = CudaVersion(7, 5);
  } else if (nvrtc_version == CudaVersion(11, 0)) { // 11.0 supports 3-8.0
    max_dev_version = CudaVersion(8, 0);
  } else if (nvrtc_version.first == 11 && nvrtc_version.second < 8) {
    max_dev_version = CudaVersion(8, 6);
  } else {
    // If the driver version is unknown (i.e. newer than this code)
    // assume the driver supports this device
    max_dev_version = dev_version;
  }
  if (dev_version > max_dev_version) {
    major = max_dev_version.first;
    minor = max_dev_version.second;
    // if we are clamping major/minor, sass is not compatible
    compile_to_sass = false;
  } else {
    major = dev_version.first;
    minor = dev_version.second;
    compile_to_sass = true;
  }
}

#if defined(__linux__)
std::string disassembleBinary(
    const std::vector<char>& cubin,
    const std::string& nvdisasm_args) {
  const char* err = "Failed to disassemble cubin";

  // Reference:
  // https://stackoverflow.com/a/3469651
  // https://linuxhint.com/dup2_system_call_c/

  constexpr int READ = 0, WRITE = 1;
  std::array<int, 2> cubin_pipe{-1, -1};
  std::array<int, 2> disasm_pipe = {-1, -1};
  std::array<int, 2> err_pipe = {-1, -1};

  NVF_ERROR(
      pipe(cubin_pipe.data()) == 0 && pipe(disasm_pipe.data()) == 0 &&
          pipe(err_pipe.data()) == 0,
      err);

  pid_t pid = fork();
  NVF_ERROR(pid != -1, err);

  if (pid) { // I am the parent
    // Parent only write cubin and read disasm, close unused pipe end
    NVF_ERROR(close(cubin_pipe[READ]) == 0, err);
    NVF_ERROR(close(disasm_pipe[WRITE]) == 0, err);
    NVF_ERROR(close(err_pipe[WRITE]) == 0, err);

    // Wrap pipe fileno as C file stream
    FILE* cubin_fp = fdopen(cubin_pipe[WRITE], "wb");
    FILE* disasm_fp = fdopen(disasm_pipe[READ], "r");
    FILE* err_fp = fdopen(err_pipe[READ], "r");
    NVF_ERROR(cubin_fp != nullptr, err);
    NVF_ERROR(disasm_fp != nullptr, err);
    NVF_ERROR(err_fp != nullptr, err);

    // Write cubin to nvdisasm
    size_t written = fwrite(cubin.data(), 1, cubin.size(), cubin_fp);
    NVF_ERROR(written == cubin.size(), err);
    fclose(cubin_fp);

    int ch = -1;

    // read disassembly result
    std::string result;
    result.reserve(cubin.size());
    while ((ch = fgetc(disasm_fp)) != EOF) {
      result.push_back((char)ch);
    }
    fclose(disasm_fp);

    // read error message
    std::string error;
    while ((ch = fgetc(err_fp)) != EOF) {
      error.push_back((char)ch);
    }
    fclose(err_fp);
    NVF_CHECK(error.empty(), error);

    return result;
  } else { // I am the child
    // For easier understanding, we can consider the fileno as a smart pointer
    // pointing to an underlying IO object in the kernel. Both the pointer and
    // the underlying objects are owned by the kernel, and multiple pointers
    // can point to the same object. `close` destroy the pointer, which does
    // not necessarily destroy the object.

    // Modify the stdin, stdout and stderr pointer to point to the pipe object
    NVF_ERROR(close(STDIN_FILENO) == 0, err);
    NVF_ERROR(close(STDOUT_FILENO) == 0, err);
    NVF_ERROR(close(STDERR_FILENO) == 0, err);
    NVF_ERROR(dup2(cubin_pipe[READ], STDIN_FILENO) != -1, err);
    NVF_ERROR(dup2(disasm_pipe[WRITE], STDOUT_FILENO) != -1, err);
    NVF_ERROR(dup2(err_pipe[WRITE], STDERR_FILENO) != -1, err);

    // Now we have stdin, stdout and stderr pointing to the pipe object, we no
    // longer need the original pointers.
    NVF_ERROR(close(cubin_pipe[READ]) == 0, err);
    NVF_ERROR(close(cubin_pipe[WRITE]) == 0, err);
    NVF_ERROR(close(disasm_pipe[READ]) == 0, err);
    NVF_ERROR(close(disasm_pipe[WRITE]) == 0, err);
    NVF_ERROR(close(err_pipe[READ]) == 0, err);
    NVF_ERROR(close(err_pipe[WRITE]) == 0, err);

    // If execl succeed, then the current process will be replaced by nvdisasm,
    // and all the remaining code in the current process will not be executed.
    // So, execl only returns when it fails.
    //
    // TODO: I was planning to use `nvdisasm /dev/stdin` which could avoid
    // creating temporary file, but unfortunately, that fails with:
    //   nvdisasm fatal   : Memory allocation failure
    // so I have to dump the stdin to a temp file and let nvdisasm read it. I am
    // hoping that nvdisasm will support reading from stdin one day.
    std::stringstream ss;
    ss << "export PATH=$PATH:/usr/local/cuda/bin;"
       << "TMPFILE=$(mktemp);"
       << "cat>$TMPFILE;"
       << "nvdisasm $TMPFILE " << nvdisasm_args << "; rm $TMPFILE";
    auto command = ss.str();
    execl("/bin/bash", "bash", "-c", command.c_str(), NULL);

    // only reachable when execl fails
    NVF_THROW(err);
  }
}
#else // #if defined(__linux__)
std::string disassembleBinary(const std::vector<char>& binary) {
  NVF_CHECK(false, "disassembling cubin is only supported on Linux");
}
#endif // #if defined(__linux__)

//! Utility class to invoke cuModuleLoadDataEx. Similar to
//! NvrtcCompileDriver, the main task is to set up the option lists
//! of type void**
class CuModuleLoadDataDriver {
 public:
  //! Valid option type is either int or char*
  using OptionType = std::variant<int, char*>;

  template <typename OptionValType>
  void setOption(CUjit_option key, OptionValType val) {
    options_.push_back(key);
    option_vals_.push_back(val);
  }

  //! Enable logging of cuModuleLoadData
  void enableLogging() {
    logging_enabled_ = true;
    log_.resize(kLogSize);
  }

  const std::string& log() const {
    NVF_ERROR(logging_enabled_, "Logging not enabled");
    return log_;
  }

  //! Invoke cuModuleLoadDataEx with ptx or cubin. Dump logging output
  //! if enabled
  std::string invoke(CUmodule& module, const void* image) {
    FUSER_PERF_SCOPE("executor_utils::Nvrtc::LoadPTX");

    auto [opts, opt_vals] = getOptions();

    NVFUSER_CUDA_SAFE_CALL(cuModuleLoadDataEx(
        &module, image, opts.size(), opts.data(), opt_vals.data()));

    if (logging_enabled_) {
      debug() << log_ << std::endl;
    }

    return log_;
  }

 private:
  // Get options that can be passed to cuModuleLoadDataEx
  std::pair<std::vector<CUjit_option>, std::vector<void*>> getOptions() {
    auto opts = options_;
    auto opt_vals = option_vals_;

    // Append options for saving log message to log_
    if (logging_enabled_) {
      opts.push_back(CU_JIT_LOG_VERBOSE);
      opt_vals.emplace_back(1);

      opts.push_back(CU_JIT_INFO_LOG_BUFFER);
      opt_vals.emplace_back(log_.data());

      opts.push_back(CU_JIT_INFO_LOG_BUFFER_SIZE_BYTES);
      opt_vals.emplace_back(kLogSize);
    }

    // Convert the options to void**. This is ugly, but that's how
    // cuModuleLoadDataEx works. See initCUDA in the
    // matrixMulDynlinkJIT sample
    // https://github.com/NVIDIA/cuda-samples/blob/master/Samples/0_Introduction/matrixMulDynlinkJIT/matrixMulDynlinkJIT.cpp#L169-L204.
    std::vector<void*> opt_val_voidp(opt_vals.size());
    for (const auto i : arange(opt_vals.size())) {
      auto opt_val = opt_vals.at(i);
      if (std::holds_alternative<int>(opt_val)) {
        // NOLINTNEXTLINE(performance-no-int-to-ptr)
        opt_val_voidp.at(i) = (void*)(int64_t)std::get<int>(opt_val);
      } else if (std::holds_alternative<char*>(opt_val)) {
        opt_val_voidp.at(i) = std::get<char*>(opt_val);
      } else {
        NVF_THROW("Invalid option");
      }
    }

    return std::make_pair(opts, opt_val_voidp);
  }

 private:
  static constexpr int kLogSize = 8196;
  //! cuModuleLoadDataEx options
  std::vector<CUjit_option> options_;
  //! Option parameters
  std::vector<OptionType> option_vals_;
  //! Save log to log_ if true
  bool logging_enabled_ = false;
  std::string log_;
};

// Get the max register count passed as -maxrregcount ptxas
// option. The count is determined based on block sizes, an optional
// heuristic and an environment variable.
std::optional<int64_t> getMaxRegCount(
    std::optional<int64_t> opt_block_size,
    const int64_t max_register_heuristic) {
  // The maximum possible count allowed by ptxas is 255
  constexpr int64_t max_register_limit = 255;

  // Temporary set the max register count to be larger than the
  // limit.
  int64_t max_register = max_register_limit + 1;

  // If the block size is known, set the maximum that at least allows
  // one block to be resident on an SM
  if (opt_block_size.has_value() && opt_block_size.value() > 0) {
    constexpr int64_t block_per_sm = 1;
    max_register = std::min(
        max_register_limit,
        getRegPerThreadGivenThreadsPerSM(
            opt_block_size.value() * block_per_sm));
  }

  // If a heuristic value is given, i.e., max_register_heuristic is
  // less than the limit, use that value if it's smaller than the
  // block-size based count
  if (max_register_heuristic < max_register_limit) {
    max_register = std::min(max_register, max_register_heuristic);
  }

  // Overwrite the count by the environment variable
  if (auto env_count = getNvFuserEnv("MAX_REG_COUNT")) {
    auto env_max_reg_count = std::atoi(env_count);
    NVF_CHECK(
        env_max_reg_count > 0 && env_max_reg_count <= max_register_limit,
        "Invalid max register count specified by NVFUSER_MAX_REG_COUNT: ",
        env_max_reg_count);
    max_register = env_max_reg_count;
  }

  // only need to set this option when we want to limit the register usage,
  // otherwise compiler with cuda-12.7 may use more registers than needed,
  // which may cause lower occupancy and performance regression.
  if (max_register < max_register_limit) {
    return max_register;
  } else {
    return std::optional<int64_t>();
  }
}

// Fill options for nvrtcCompileProgram and cuModuleLoadDataEx
void fillCompileOptions(
    NvrtcCompileDriver& nvrtc_compile_driver,
    CuModuleLoadDataDriver& module_load_driver,
    bool compile_to_sass,
    int64_t major,
    int64_t minor,
    const CompileParams& compile_params,
    std::optional<int64_t> opt_block_size) {
  nvrtc_compile_driver.setOption("--std=c++17");
  if (isOptionEnabled(EnableOption::KernelDebug)) {
    nvrtc_compile_driver.setOption("-G");
  }

  // Suppress warnings for functions that are defined but unused, since we have
  // many unused functions in the preamble.
  nvrtc_compile_driver.setOption("--diag-suppress=177");

  // CUDA 11.1 allows going directly to SASS (sm_) instead of PTX (compute_)
  // which gives better backwards compatibility to work on older driver,
  // (since older driver doesn't necessarily recognize PTX emitted by new
  // toolkit);
  // Meanwhile, for forward compatibility (future device with
  // `unsupported_arch==True`), since SASS are not necessarily compatible,
  // we fallback to PTX instead.
  std::string compute = std::string("--gpu-architecture=") +
      (compile_to_sass ? "sm_" : "compute_") + std::to_string(major) +
      std::to_string(minor);
  if (major >= 9) {
    // Use 90a and 100a so that arch-specific PTX is available
    compute += "a";
  }
  nvrtc_compile_driver.setOption(compute);

  nvrtc_compile_driver.setOption("-default-device");

  if (isOptionDisabled(DisableOption::Fma)) {
    nvrtc_compile_driver.setOption("--fmad=false");
  } else {
    nvrtc_compile_driver.setOption("--fmad=true");
  }

  // Add line info to generated kernels
  if (isOptionEnabled(EnableOption::KernelLineInfo)) {
    nvrtc_compile_driver.setOption("-lineinfo");
  }

#ifdef NDEBUG
  // Avoid excessive register usage from assertion
  nvrtc_compile_driver.setOption("-DNDEBUG");
#endif

  if (isOptionEnabled(EnableOption::KernelProfile)) {
    nvrtc_compile_driver.setOption("-DNVFUSER_PROFILE_KERNEL");
  }
  if (isDebugDumpEnabled(DebugDumpOption::PrintPtxasLog) ||
      isDebugDumpEnabled(DebugDumpOption::PerfDebugVerbose) ||
      isOptionEnabled(EnableOption::WarnRegisterSpill) ||
      compile_params.enable_ptxas_verbose) {
    // show register usage in compilation log
    if (compile_to_sass) {
      nvrtc_compile_driver.setOption("--ptxas-options");
      nvrtc_compile_driver.setOption("--verbose");
    } else {
      module_load_driver.enableLogging();
    }
  }

  const char* ptxas_opt_level = getNvFuserEnv("JIT_OPT_LEVEL");

  if (ptxas_opt_level) {
    int val = atoi(ptxas_opt_level);
    if (val <= 4 && val >= 0) {
      if (val < 4) {
        TORCH_WARN(
            "ptxas optimization level manually set as ",
            val,
            ", which could negatively affect performance. Try removing env "
            "variable NVFUSER_JIT_OPT_LEVEL for optimal performance.");
      }
      if (compile_to_sass) {
        nvrtc_compile_driver.setOption("--ptxas-options");
        nvrtc_compile_driver.setOption("-O" + std::to_string(val));
      } else {
        module_load_driver.setOption(CU_JIT_OPTIMIZATION_LEVEL, val);
      }
    } else {
      TORCH_WARN_ONCE(
          "acceptable range for NVFUSER_JIT_OPT_LEVEL is between 0 and 4, but "
          "received ",
          val,
          ", ignoring the option");
    }
  }

  const auto max_register =
      getMaxRegCount(opt_block_size, compile_params.maxrregcount);

  // If the max register count is set
  if (max_register.has_value()) {
    if (compile_to_sass) {
      nvrtc_compile_driver.setOption(
          "--maxrregcount=" + std::to_string(*max_register));
    } else {
      module_load_driver.setOption(CU_JIT_MAX_REGISTERS, (int)*max_register);
    }
  }

  if (isOptionDisabled(DisableOption::NvrtcCaching)) {
    // JIT caching is introduced in 12.9. It's always disabled in
    // prior versions.
    int major, minor;
    NVFUSER_NVRTC_SAFE_CALL(nvrtcVersion(&major, &minor));
    if ((major == 12 && minor >= 9) || major > 12) {
      nvrtc_compile_driver.setOption("--no-cache");
    }
  }

  for (const auto& include_path : compile_params.include_paths) {
    nvrtc_compile_driver.setOption("-I" + include_path);
  }
}

// Dump ptxas output if register spill is detected
int warnRegisterSpill(const std::string& compile_log) {
  // Get a matching int from a given nvrtc compile log that matches a
  // pattern of "\d+ sub_str", e.g., in the case of "4 bytes stack
  // frame", 4 would be returned.
  auto get_preceding_int = [](const std::string& log,
                              const std::string& sub_str) -> int64_t {
    std::regex pattern("(\\d+) " + sub_str);
    std::smatch matches;
    NVF_ERROR(
        std::regex_search(log, matches, pattern),
        "Unexpected compile log: ",
        log);
    return std::stoi(matches[1]);
  };

  const std::string str_stack = "bytes stack frame";
  const std::string str_store = "bytes spill stores";
  const std::string str_load = "bytes spill loads";
  int stack_count = get_preceding_int(compile_log, str_stack);
  int store_count = get_preceding_int(compile_log, str_store);
  int load_count = get_preceding_int(compile_log, str_load);
  int allowed_spill = 0;
  if (isOptionEnabled(EnableOption::WarnRegisterSpill)) {
    auto optionArgs = getEnableOptionArguments(EnableOption::WarnRegisterSpill);
    if (!optionArgs.empty()) {
      try {
        allowed_spill = std::stoi(optionArgs[0]);
      } catch (const std::exception& e) {
        debug() << "skip invalid argument for WarnRegisterSpill, arg = "
                << optionArgs[0] << std::endl;
      }
    }
  }
  if (stack_count > allowed_spill || store_count > allowed_spill ||
      load_count > allowed_spill) {
    debug() << "WARNING: Register spill detected\n" << compile_log << std::endl;
  }
  return store_count + load_count;
}

void createNvrtcProgram(
    nvrtcProgram& program,
    const std::string& kernel_name,
    const std::string& full_src_code) {
  std::stringstream ss;
  ss << "__tmp_" << kernel_name << ".cu";
  std::string name = ss.str();
  FUSER_PERF_SCOPE("executor_utils::NvrtcCreateProgram");
  NVFUSER_NVRTC_SAFE_CALL(nvrtcCreateProgram(
      &program, full_src_code.c_str(), name.c_str(), 0, nullptr, nullptr));
}

std::vector<char> compileNvrtcProgramToCubin(const nvrtcProgram& program) {
#if CUDA_VERSION < 11010
  NVF_THROW("SASS not supported in CUDA versions older than 11.1");
#endif

  size_t size = 0;
  NVFUSER_NVRTC_SAFE_CALL(nvrtcGetCUBINSize(program, &size));
  std::vector<char> code(size);
  NVFUSER_NVRTC_SAFE_CALL(nvrtcGetCUBIN(program, code.data()));
  return code;
}

// Returns the name of the dumped file.
std::string dumpCompiledCodeToFile(
    const std::vector<char>& code,
    const std::string& kernel_name,
    const std::string& suffix) {
  std::stringstream file_name;
  file_name << "__tmp_" << kernel_name << suffix;
  debug() << "PRINTING: " << file_name.str() << std::endl;
  std::ofstream out(file_name.str());
  NVF_ERROR(out.is_open());
  out.write(code.data(), (std::streamsize)code.size());
  out.close();
  return file_name.str();
}

std::vector<char> compileNvrtcProgramToPtx(const nvrtcProgram& program) {
  size_t size = 0;
  NVFUSER_NVRTC_SAFE_CALL(nvrtcGetPTXSize(program, &size));
  std::vector<char> code(size);
  NVFUSER_NVRTC_SAFE_CALL(nvrtcGetPTX(program, code.data()));
  return code;
}

// Compile the given source code with the NVRTC compiler driver.
std::unique_ptr<executor_utils::CudaExecutable> compileSource(
    const std::string& full_src_code,
    const std::string& func_name,
    const bool compile_to_sass,
    NvrtcCompileDriver& nvrtc_compile) {
  std::stringstream log;

  nvrtcProgram program; // NOLINT(cppcoreguidelines-init-variables)
  torch::jit::ResourceGuard holdProgram([&] {
    FUSER_PERF_SCOPE("executor_utils::NvrtcDestroyProgram");
    NVFUSER_NVRTC_SAFE_CALL(nvrtcDestroyProgram(&program));
  });

  createNvrtcProgram(program, func_name, full_src_code);

  std::string canonical_func_name =
      CompiledKernel::kernelNamespace() + "::" + func_name;

  NVFUSER_NVRTC_SAFE_CALL(
      nvrtcAddNameExpression(program, canonical_func_name.c_str()));
  log << nvrtc_compile.invoke(program, full_src_code) << std::endl;

  auto compiled_kernel = std::make_unique<executor_utils::CudaExecutable>();
  const char* lowered_kernel_name = nullptr;
  NVFUSER_NVRTC_SAFE_CALL(nvrtcGetLoweredName(
      program, canonical_func_name.c_str(), &lowered_kernel_name));
  compiled_kernel->kernel_name = lowered_kernel_name;
  compiled_kernel->compile_log = log.str();

  if (compile_to_sass) {
    compiled_kernel->cubin = compileNvrtcProgramToCubin(program);
    if (isDebugDumpEnabled(DebugDumpOption::Cubin)) {
      compiled_kernel->cubin_filename =
          dumpCompiledCodeToFile(compiled_kernel->cubin, func_name, ".cubin");
    }
    if (isDebugDumpEnabled(DebugDumpOption::SassToFile)) {
      std::string sass_str =
          disassembleBinary(compiled_kernel->cubin, "-fun 1 -c");
      compiled_kernel->sass = {sass_str.begin(), sass_str.end()};
      compiled_kernel->sass_filename =
          dumpCompiledCodeToFile(compiled_kernel->sass, func_name, ".sass");
    }
  }

  if (!compile_to_sass || isDebugDumpEnabled(DebugDumpOption::Ptx)) {
    compiled_kernel->ptx = compileNvrtcProgramToPtx(program);
    if (isDebugDumpEnabled(DebugDumpOption::Ptx)) {
      compiled_kernel->ptx_filename =
          dumpCompiledCodeToFile(compiled_kernel->ptx, func_name, ".ptx");
    }
  }

  return compiled_kernel;
}

// Compile the source if no existing compiled binary is found in KernelDB
std::unique_ptr<executor_utils::CudaExecutable> getCudaExecutable(
    std::optional<std::reference_wrapper<const std::string>> kernel_code,
    const std::string& full_src_code,
    const std::string& func_name,
    const std::string& id,
    const CompileParams& compile_params = CompileParams(),
    std::optional<int64_t> opt_block_size = std::nullopt) {
  FUSER_PERF_SCOPE("executor_utils::NVRTC");

  at::cuda::jit::initializeCudaContext();

  // The above initialization works in some cases. However, it seems to
  // occasionally fail to initialize a primary context. Here we check for that
  // and if we detect that no context exists, we create one manually.
  int device = 0;
  cudaGetDevice(&device);
  if (!at::detail::getCUDAHooks().hasPrimaryContext((c10::DeviceIndex)device)) {
    // CUDA>=12 creates a context when cudaSetDevice is called. However, before
    // cu12, that context is not necessarily created. In that case, we create
    // one here implicitly. See https://github.com/NVIDIA/Fuser/issues/429
    cudaFree(nullptr);
  }

  const auto prop = at::cuda::getCurrentDeviceProperties();

  int64_t major = 0, minor = 0;
  bool compile_to_sass = false;
  queryTargetGPUVersion(prop, major, minor, compile_to_sass);

#if CUDA_VERSION < 11010
  // compile to sass is not allowed prior to CUDA 11.1
  compile_to_sass = false;
#endif

  if (isOptionDisabled(DisableOption::CompileToSass)) {
    compile_to_sass = false;
  }

  NvrtcCompileDriver nvrtc_compile_driver;
  CuModuleLoadDataDriver module_load_driver;

  fillCompileOptions(
      nvrtc_compile_driver,
      module_load_driver,
      compile_to_sass,
      major,
      minor,
      compile_params,
      opt_block_size);

  std::stringstream log;

  if (compile_to_sass) {
    log << "\nCompile options: ";
    for (const auto& opt : nvrtc_compile_driver.options()) {
      log << opt << " ";
    }
    if (opt_block_size.has_value()) {
      log << " ; block size=" << opt_block_size.value() << "\n";
    }
  }

  auto compiled_kernel = std::make_unique<executor_utils::CudaExecutable>();
  const auto compile_args =
      toDelimitedString(nvrtc_compile_driver.options(), " ");

  auto& kernel_db = KernelDb::get();
  const auto use_kernel_db = kernel_db.enabled() && kernel_code.has_value();

  // If the Kernel Query fails, the Kernel is recompiled
  if (!(use_kernel_db &&
        kernel_db.query(
            kernel_code.value(),
            compile_args,
            compiled_kernel->kernel_name,
            (compile_to_sass ? compiled_kernel->cubin
                             : compiled_kernel->ptx)))) {
    compiled_kernel = compileSource(
        full_src_code, func_name, compile_to_sass, nvrtc_compile_driver);
    log << compiled_kernel->compile_log << std::endl;
    if (use_kernel_db) {
      auto result = kernel_db.write(
          kernel_code.value(),
          compile_args,
          compiled_kernel->kernel_name,
          (compile_to_sass ? compiled_kernel->cubin : compiled_kernel->ptx));
      if (!result) {
        TORCH_WARN(
            "kernel_db was unable to write kernel: ",
            compiled_kernel->kernel_name);
      }
    }
  }

  log << module_load_driver.invoke(
             compiled_kernel->module,
             (compile_to_sass ? compiled_kernel->cubin.data()
                              : compiled_kernel->ptx.data()))
      << std::endl;
  compiled_kernel->compile_log = log.str();
  compiled_kernel->compile_args = compile_args;

  if (isOptionEnabled(EnableOption::WarnRegisterSpill) ||
      compile_params.enable_ptxas_verbose) {
    compiled_kernel->register_spills =
        warnRegisterSpill(compiled_kernel->compile_log);
  }

  NVFUSER_CUDA_SAFE_CALL(cuModuleGetFunction(
      &(compiled_kernel->function),
      compiled_kernel->module,
      compiled_kernel->kernel_name.c_str()));

  // Store block size used to generate compile arguments
  if (opt_block_size.has_value()) {
    compiled_kernel->block_size = opt_block_size.value();
  }

  return compiled_kernel;
}

std::unique_ptr<executor_utils::CudaExecutable> getCudaExecutable(
    const serde::CudaKernel* buffer,
    const CompileParams& compile_params) {
  FUSER_PERF_SCOPE("executor_utils::serde_NVRTC");

  NVF_ERROR(buffer != nullptr, "serde::CudaKernel is nullptr.");

  // Deserialize flatbuffer into CudaExecutable
  auto compiled_kernel = std::make_unique<executor_utils::CudaExecutable>();
  compiled_kernel->kernel_name = buffer->kernel_name()->str();
  compiled_kernel->compile_args = buffer->compile_args()->str();
  compiled_kernel->block_size = buffer->block_size();

  if (buffer->cubin() != nullptr) {
    compiled_kernel->cubin.reserve(buffer->cubin()->size());
    std::copy(
        buffer->cubin()->begin(),
        buffer->cubin()->end(),
        std::back_inserter(compiled_kernel->cubin));
    compiled_kernel->cubin_filename = buffer->cubin_filename()->str();
  }

  if (buffer->ptx() != nullptr) {
    compiled_kernel->ptx.reserve(buffer->ptx()->size());
    std::copy(
        buffer->ptx()->begin(),
        buffer->ptx()->end(),
        std::back_inserter(compiled_kernel->ptx));
    compiled_kernel->ptx_filename = buffer->ptx_filename()->str();
  }

  at::cuda::jit::initializeCudaContext();

  // The above initialization works in some cases. However, it seems to
  // occasionally fail to initialize a primary context. Here we check for that
  // and if we detect that no context exists, we create one manually.
  int device = 0;
  cudaGetDevice(&device);
  if (!at::detail::getCUDAHooks().hasPrimaryContext((c10::DeviceIndex)device)) {
    // CUDA>=12 creates a context when cudaSetDevice is called. However, before
    // cu12, that context is not necessarily created. In that case, we create
    // one here implicitly. See https://github.com/NVIDIA/Fuser/issues/429
    cudaFree(nullptr);
  }

  const auto prop = at::cuda::getCurrentDeviceProperties();

  // Generate compile args and compare against saved args in compiled_kernel
  NvrtcCompileDriver nvrtc_compile_driver;
  CuModuleLoadDataDriver module_load_driver;

  int64_t major = 0, minor = 0;
  bool compile_to_sass = false;
  queryTargetGPUVersion(prop, major, minor, compile_to_sass);

  std::optional<int64_t> opt_block_size;
  if (compiled_kernel->block_size >= -1) {
    opt_block_size = compiled_kernel->block_size;
  }

  fillCompileOptions(
      nvrtc_compile_driver,
      module_load_driver,
      compile_to_sass,
      major,
      minor,
      compile_params,
      opt_block_size);

  const auto latest_compile_args =
      toDelimitedString(nvrtc_compile_driver.options(), " ");
  NVF_ERROR(
      latest_compile_args == compiled_kernel->compile_args,
      "The compile arguments for the serialized cuda kernel does not ",
      "match the latest generated compile args.\t",
      latest_compile_args,
      "\t",
      compiled_kernel->compile_args);

  NVF_ERROR(
      !compile_to_sass || !compiled_kernel->cubin.empty(),
      "Expected compiled cubin after deserializing CudaExecutable.");

  NVF_ERROR(
      compile_to_sass || !compiled_kernel->ptx.empty(),
      "Expected compiled ptx after deserializing CudaExecutable.");

  std::stringstream log;
  log << module_load_driver.invoke(
             compiled_kernel->module,
             (compile_to_sass ? compiled_kernel->cubin.data()
                              : compiled_kernel->ptx.data()))
      << std::endl;
  compiled_kernel->compile_log = log.str();

  NVFUSER_CUDA_SAFE_CALL(cuModuleGetFunction(
      &(compiled_kernel->function),
      compiled_kernel->module,
      compiled_kernel->kernel_name.c_str()));

  return compiled_kernel;
}

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

static const std::string& defineStdComplex() {
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
    debug() << "Didn't find requested external source code. Will use generated "
               "code!\n"
            << "Number of source code files should equal the number of fusion "
               "segments.\n"
            << "External source code filenames should be delineated with "
               "commas, e.g.: file1.cu,file2.cu.\n";
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

bool requiresDisabledParamCache(const kir::Kernel* kernel) {
  std::vector<Val*> output_extents;
  for (auto out : kernel->outputs()) {
    const auto logical_domain = out->as<TensorView>()->getLogicalDomain();
    // walking through outputs to see if output shapes are dependent on
    // non-tensor inputs. For which case, we should have disabled output
    // allocation, since the caching id only looks at tensor shapes.
    // See issue https://github.com/csarofeen/pytorch/issues/2002
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
  }

  VectorOfUniqueEntries<Val*> input_dependencies;
  for (auto inp : InputsOf::outputs(output_extents)) {
    if (inp->isFusionInput()) {
      input_dependencies.pushBack(inp);
    }
  }
  if (std::any_of(
          input_dependencies.begin(), input_dependencies.end(), [](Val* inp) {
            return inp->isScalar();
          })) {
    return true;
  } else if (!input_dependencies.empty()) {
    VectorOfUniqueEntries<Expr*> all_exprs(DependencyCheck::getAllExprsBetween(
        input_dependencies.set(), output_extents));

    VectorOfUniqueEntries<Val*> meta_data_outputs;
    for (auto meta_data_op :
         ir_utils::filterByType<GetMetaData>(all_exprs.vector())) {
      meta_data_outputs.pushBack(
          meta_data_op->outputs().begin(), meta_data_op->outputs().end());
    }

    VectorOfUniqueEntries<Expr*> before_meta_data_exprs(
        DependencyCheck::getAllExprsBetween(
            input_dependencies.set(), meta_data_outputs.vector()));

    VectorOfUniqueEntries<Expr*> after_meta_data_exprs(
        DependencyCheck::getAllExprsBetween(
            meta_data_outputs.set(), output_extents));

    auto subtraction = all_exprs;
    subtraction = subtraction.computeSubtract(before_meta_data_exprs);
    subtraction = subtraction.computeSubtract(after_meta_data_exprs);
    if (!subtraction.empty()) {
      return true;
    }
  }
  return false;
}

std::string _getStructuredCode(
    const std::string& kernel_str,
    PrimDataType index_type,
    std::string kernel_name,
    bool has_argsort = false,
    bool has_topk = false) {
  // generating cuda code;
  std::string code = "";
  code += defineStdComplex();
  code += std::string("namespace ") + CompiledKernel::kernelNamespace() +
      "{\n" + defineTypes() + defineIndexType(index_type) + kernelPreamble() +
      "} // namespace " + CompiledKernel::kernelNamespace() + "\n";

  if (has_argsort) {
    code += nvfuser_resources::argsort_cu;
  }

  if (has_topk) {
    code += nvfuser_resources::topk_cu;
  }

  code += "\nnamespace " + CompiledKernel::kernelNamespace() + " {\n\n";
  code += kernel_str;
  code += "\n} // namespace " + CompiledKernel::kernelNamespace() + "\n";

  if (isDebugDumpEnabled(DebugDumpOption::CudaKernel)) {
    debug() << "\n======= Codegen output for kernel: " << kernel_name
            << " =======\n\n"
            << kernel_str << "\n======================================\n\n";
  } else if (isDebugDumpEnabled(DebugDumpOption::CudaFull)) {
    debug() << "\n======= Codegen output for kernel: " << kernel_name
            << " =======\n\n"
            << code << "\n======================================\n\n";
  }
  if (isDebugDumpEnabled(DebugDumpOption::CudaToFile)) {
    std::stringstream file_name;
    file_name << "__tmp_" << kernel_name << ".cu";
    debug() << "PRINTING: " << file_name.str() << std::endl;
    std::ofstream out(file_name.str());
    out << code << std::endl;
    out.close();
  }

  return code;
}

} // namespace

NVF_API CompiledKernel::CompiledKernel(
    Fusion* fusion,
    CompileParams compile_params,
    c10::Device device,
    SchedulerType scheduler_type,
    int64_t fusion_id,
    int64_t concrete_id,
    int64_t runtime_id,
    int64_t group_id,
    const std::vector<std::function<void(GpuLower*)>>& pre_lowering_hooks,
    const std::vector<std::function<void(kir::Kernel*)>>& post_lowering_hooks)
    : compile_params_(compile_params),
      scheduler_type_(scheduler_type),
      fusion_id_(fusion_id),
      concrete_id_(concrete_id),
      runtime_id_(runtime_id),
      group_id_(group_id),
      lowered_(std::make_unique<GpuLower>(fusion, compile_params)),
      device_(device) {
  FUSER_PERF_SCOPE("CompiledKernel::CompiledKernel");

  // TODO: No hooks can be sent because this is in the constructor
  for (const auto& hook : pre_lowering_hooks) {
    hook(lowered_.get());
  }
  lowered_->run();
  for (const auto& hook : post_lowering_hooks) {
    hook(lowered_->kernel());
  }

  // Add CUDA include path if fusion contains ArgsortOp or TopKOp. Note that
  // this is a temporary measure. CUB header files need to be
  // installed as part of the nvFuser installation.
  if (lowered_->kernel()->summary().has_argsort ||
      lowered_->kernel()->summary().has_topk) {
    compile_params_.include_paths.push_back("/usr/local/cuda/include");
    // As of CUDA 13, the CUB header files are moved to the cccl
    // subdirectory. This include path is not necessary pre 13 but is
    // added anyway as it should be just a no-op.
    compile_params_.include_paths.push_back("/usr/local/cuda/include/cccl");
  }
}

NVF_API CompiledKernel::CompiledKernel(
    Fusion* fusion,
    CompileParams compile_params,
    c10::Device device,
    SchedulerType scheduler_type,
    int64_t fusion_id,
    int64_t concrete_id,
    int64_t runtime_id,
    int64_t group_id)
    : CompiledKernel(
          fusion,
          compile_params,
          device,
          scheduler_type,
          fusion_id,
          concrete_id,
          runtime_id,
          group_id,
          {},
          {}) {}

NVF_API CompiledKernel::~CompiledKernel() = default;

void CompiledKernel::compile(const LaunchParams& lparams) {
  FUSER_PERF_SCOPE("CompiledKernel::compile");

  NVF_ERROR(
      !kernel()->outputs().empty(),
      "No output found for this kernel, aborting.");

  // Parameter cache doesn't cache on input scalars, so if one is used as a
  // dynamic input size of a tensor the cache doesn't work correctly. This
  // should be enabled in the cache, but since it's not, for now we will disable
  // it under these circumstances.
  launch_param_cache_disabled_ = requiresDisabledParamCache(kernel());

  if (isDebugDumpEnabled(DebugDumpOption::FusionIrGraph)) {
    std::stringstream file_name;
    file_name << "__tmp_fusion_ir_graph_" << kernel_id_ << ".dot";
    IrGraphGenerator::print(
        kernel()->as<Fusion>(),
        file_name.str().c_str(),
        IrGraphGenerator::DetailLevel::ComputeOnly);
  }

  c10::DeviceGuard dg(device_);

  NVF_ERROR(device_.is_cuda(), "Provided device to CUDA fuser is the CPU.");
  auto properties = at::cuda::getDeviceProperties(device_.index());
  // TODO: These properties should be set as part of the constructor so that
  // it can be const
  warp_size_ = properties->warpSize;
  kir::Kernel* kernel = lowered_->kernel();

  createKernelId();
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

  kernel_code_ = codegen::generateCudaKernel(kernel, kernelName(), lparams);

  const kir::KernelSummary& kernel_summary = kernel->summary();

  std::pair<int64_t, int64_t> target_arch;
  bool compile_to_sass = false;
  queryTargetGPUVersion(
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

  if (!kernel_summary.dynamic_lmem_allocations.empty()) {
    std::stringstream ss;
    ss << "Allocations must be based on constant integers for local memory. "
          "However, found: ";
    for (auto alloc : kernel_summary.dynamic_lmem_allocations) {
      ss << alloc->buffer()->toString() << ", ";
    }
    ss << " have dynamic allocations but are placed in local memory.";
    NVF_THROW(ss.str());
  }
  int64_t block_size = lparams.nThreads();
  NVF_ERROR(block_size > 0, "launch param inferred block size < 0");

  // Basically setting high water mark as 1 when we don't provide args for
  // compilation, it will just generate a kernel that gets ditched at the
  // first run - not great. We should have better heuristics.
  block_size_high_water_mark_ =
      std::max<int64_t>(block_size, block_size_high_water_mark_);
  maxrregcount_high_water_mark_ = compile_params_.maxrregcount;
  compiled_kernel_ = getCudaExecutable(
      kernel_code_,
      getStructuredCode(),
      kernelName(),
      kernel_id_,
      compile_params_,
      block_size);

  NVF_ERROR(validKernelId(), "Invalid kernel id for CompiledKernel.");

  if (isDebugDumpEnabled(DebugDumpOption::Sass)) {
    debug() << disassembledKernelSASS() << std::endl;
  }
}

std::string CompiledKernel::getStructuredCode() const {
  // If NVFUSER_EXTERNAL_SRC is set, utilize the external source code.
  // If the loaded external source code is empty, revert to the default
  // codegen. The external_structured_code is moved to structured_code and
  // explicitly cleared to avoid use-after-move scenarios. Note: we index
  // these with getGlobalFusionCount() instead of fusion_id_ in order to match
  // the numbering of files output with NVFUSER_DUMP=cuda_to_file
  auto structured_code =
      getStructuredCodeFromExternalFiles(getGlobalFusionCount());
  if (!structured_code.empty()) {
    return structured_code;
  }
  return _getStructuredCode(
      kernelString(),
      kernel()->indexType(),
      kernelName(),
      kernel()->summary().has_argsort,
      kernel()->summary().has_topk);
}

std::string CompiledKernel::disassembledKernelSASS() const {
  return disassembleBinary(compiled_kernel_->cubin, "-fun 1 -c");
}

void CompiledKernel::createKernelId() {
  NVF_ERROR(fusion_id_ > -1, "Invalid fusion_id.");
  NVF_ERROR(concrete_id_ > -1, "Invalid concrete_id.");
  NVF_ERROR(runtime_id_ > -1, "Invalid runtime_id.");
  NVF_ERROR(group_id_ > -1, "Invalid group_id");
  ++global_fusion_count_;
  std::stringstream ss;
  if (isOptionEnabled(EnableOption::StaticFusionCount)) {
    ss << global_fusion_count_.load();
  } else {
    ss << toString(scheduler_type_);
    ss << "_f" << fusion_id_;
    ss << "_c" << concrete_id_;
    ss << "_r" << runtime_id_;
    ss << "_g" << group_id_;
  }
  kernel_id_ = ss.str();
}

kir::Kernel* CompiledKernel::kernel() const {
  NVF_ERROR(lowered_);
  return lowered_->kernel();
}

void RtcKernel::compile(
    const std::string& code,
    const std::string& name,
    bool structured,
    PrimDataType index_type,
    int64_t device_index) {
  FUSER_PERF_SCOPE("RtcKernel::compile");
  NVF_ERROR(
      index_type == PrimDataType::Int || index_type == PrimDataType::Int32 ||
          "Invalid index type: ",
      index_type);
  device_index_ = device_index;

  std::string scode;
  if (!structured) {
    scode = _getStructuredCode(code, index_type, name);
  } else {
    scode = code;
  }
  CompileParams cp;
  cp.device =
      c10::Device(c10::DeviceType::CUDA, (c10::DeviceIndex)device_index_);
  compiled_kernel_ = getCudaExecutable(std::nullopt, scode, name, "0", cp);
}

float RtcKernel::run(
    const LaunchParams& launch_params,
    const KernelArgumentHolder& args,
    PrimDataType index_type) {
  FUSER_PERF_SCOPE("RtcKernel::run");

  auto device =
      c10::Device(c10::DeviceType::CUDA, (c10::DeviceIndex)device_index_);
  c10::DeviceGuard dg(device);
  auto stream = at::cuda::getCurrentCUDAStream();

  cudaEvent_t start_event = {};
  cudaEvent_t finish_event = {};

  NVFUSER_CUDA_RT_SAFE_CALL(cudaEventCreate(&start_event));
  NVFUSER_CUDA_RT_SAFE_CALL(cudaEventCreate(&finish_event));

  NVFUSER_CUDA_RT_SAFE_CALL(cudaEventRecord(start_event, stream));

  std::vector<std::vector<std::byte>> data;
  std::vector<void*> pointers;

  for (const auto& input : args) {
    NVF_ERROR(
        input.is<at::Tensor>() && input.as<at::Tensor>().is_cuda(),
        "Only CUDA tensors are supported for direct nvRTC launches at this "
        "time.");
    auto input_tensor = input.as<at::Tensor>();
    data.emplace_back(tensorToBytes(
        input_tensor,
        input_tensor.sizes().vec(),
        input_tensor.strides().vec(),
        index_type));
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

void CompiledKernel::deserialize(const serde::KernelExecutor* buffer) {
  // Initialize CompileOptions
  c10::DeviceGuard dg(device_);

  // Initialize internal fields
  maxrregcount_high_water_mark_ = buffer->maxrregcount_high_water_mark();
  warp_size_ = buffer->warp_size();
  kernel_code_ = buffer->kernel_code()->str();

  // KernelDB query checks kernel_code string and compile_params before
  // copying cubin.
  compile_params_.index_type = serde::mapToNvfuserDtype(buffer->index_type());
  compile_params_.maxrregcount = maxrregcount_high_water_mark_;

  // Replace integers that are tensor sizes by named scalars like "T0.size[0]"
  createKernelId();
  setUsedTVs();

  compiled_kernel_ =
      getCudaExecutable(buffer->compiled_kernel(), compile_params_);
}

void CompiledKernel::setUsedTVs() {
  auto used_vals = kernel()->usedMathVals();
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
} // namespace

void CompiledKernel::recompileKernel(
    const LaunchParams& new_launch_params,
    const CompileParams& new_compile_params) {
  FUSER_PERF_SCOPE("CompiledKernel::runFusion::recompileKernel");
  const auto structured_code = getStructuredCode();
  block_size_high_water_mark_ = new_launch_params.nThreads();
  maxrregcount_high_water_mark_ = new_compile_params.maxrregcount;

  // TODO: This should send in the right device!
  compiled_kernel_ = getCudaExecutable(
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
        compiled_kernel_->function, new_launch_params, device_.index());
  }
}

} // namespace nvfuser
