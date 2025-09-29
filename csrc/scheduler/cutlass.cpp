// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on

#include <device_lower/utils.h>
#include <exceptions.h>
#include <instrumentation.h>
#include <ir/all_nodes.h>
#include <ops/all_ops.h>
#include <scheduler/cutlass.h>
#include <scheduler/runtime_info.h>
#include <scheduler/utils.h>

#include <dlfcn.h>

namespace nvfuser {

// CutlassParams implementation

std::string CutlassParams::toString() const {
  std::stringstream ss;
  ss << "CutlassParams (" << scheduler_type << ")\n";
  ss << "  MMA Tile: " << mma_tile.toVector() << "\n";
  ss << "  Per-SM MMA Tile: " << per_sm_tile.toVector() << "\n";
  ss << "  Cluster shape: " << cluster_shape.toVector() << "\n";
  return ss.str();
}

size_t CutlassParams::hash() const {
  size_t h = 0;
#define HASHTILE(t)            \
  hashCombine(h, (size_t)t.m); \
  hashCombine(h, (size_t)t.n); \
  hashCombine(h, (size_t)t.k);
  HASHTILE(mma_tile);
  HASHTILE(per_sm_tile);
  HASHTILE(cluster_shape);
#undef HASHTILE
  return h;
}

bool CutlassParams::sameAs(const HeuristicParams* other) const {
  if (!other->isStrictlyA<CutlassParams>()) {
    return false;
  }
  const auto* other_cutlass = other->as<CutlassParams>();
  return cparams == other->cparams && mma_tile == other_cutlass->mma_tile &&
      per_sm_tile == other_cutlass->per_sm_tile &&
      cluster_shape == other_cutlass->cluster_shape;
}

std::unique_ptr<HeuristicParams> CutlassParams::clone() const {
  return std::make_unique<CutlassParams>(*this);
}

// CutlassScheduler implementation

bool CutlassScheduler::canScheduleCompileTime(Fusion* fusion) {
  FUSER_PERF_SCOPE("CutlassScheduler::canScheduleCompileTime");

  // TODO: Enable this scheduler by default once we are confident in the pattern
  // matching and heuristic
  if (!isOptionEnabled(EnableOption::CutlassScheduler)) {
    return false;
  }

  // Check if fusion has a supported matmul pattern
  if (!hasSupportedMatmulPattern(fusion)) {
    return false;
  }

  // Check if epilogue is supported
  if (!hasSupportedEpilogue(fusion)) {
    return false;
  }

  return true;
}

bool CutlassScheduler::canScheduleRunTime(
    Fusion* fusion,
    SchedulerRuntimeInfo& runtime_info,
    HeuristicDataCache* data_cache) {
  FUSER_PERF_SCOPE("CutlassScheduler::canScheduleRunTime");

  // For now, all runtime checks are deferred to compile time checks
  // In the future, we may want to check tensor sizes, alignment, etc.

  return true;
}

namespace {

#ifdef HAS_NVMMH

#include <nvMatmulHeuristics.h>

static thread_local void* nvmmh_handle = nullptr;

namespace nvmmh_func {
#define ALL_NVMMH_API_WRAPPER(fn)                   \
  fn(nvMatmulHeuristicsCreate);                     \
  fn(nvMatmulHeuristicsDestroy);                    \
  fn(nvMatmulHeuristicsGetStatusString);            \
  fn(nvMatmulHeuristicsGetVersionMajor);            \
  fn(nvMatmulHeuristicsGetVersionMinor);            \
  fn(nvMatmulHeuristicsGetVersionPatch);            \
  fn(nvMatmulHeuristicsBackendCreate);              \
  fn(nvMatmulHeuristicsBackendSetCallbackProperty); \
  fn(nvMatmulHeuristicsLoadInternalDiscoverySet);   \
  fn(nvMatmulHeuristicsGetGemmConfigEx);

#define DECLARE_STATIC_FUNCTION_HANDLE(func) \
  [[maybe_unused]] static thread_local decltype(&func) func = nullptr;

ALL_NVMMH_API_WRAPPER(DECLARE_STATIC_FUNCTION_HANDLE);

#undef DECLARE_STATIC_FUNCTION_HANDLE

} // namespace nvmmh_func

#define NVMMH_SAFE_CALL(x)                                       \
  do {                                                           \
    nvmmhStatus_t _result = nvmmh_func::x;                       \
    NVF_ERROR(                                                   \
        _result == NVMMH_STATUS_SUCCESS,                         \
        "NVMMH error: " #x "failed with error ",                 \
        nvmmh_func::nvMatmulHeuristicsGetStatusString(_result)); \
  } while (0)

bool initNVMMH() {
  if (nvmmh_handle != nullptr) {
    return true;
  }

// Stringify already-expanded s
#define STRINGIFY(s) #s
// Force expansion of s
#define EXPAND_AND_STRINGIFY(s) STRINGIFY(s)
  constexpr std::string_view libname =
      "libnvMatmulHeuristics.so." EXPAND_AND_STRINGIFY(NVMMH_VERSION_MAJOR);
#undef STRINGIFY
#undef EXPAND_AND_STRINGIFY

  nvmmh_handle = dlopen(libname.data(), RTLD_LAZY);
  if (nvmmh_handle == nullptr) {
    TORCH_WARN_ONCE(
        "Could not link to ",
        libname,
        ". nvMatmulHeuristics support is disabled.");
    return false;
  }

#define DEFINE_NVMMH_SYMBOL(func)                                      \
  if (nvmmh_func::func == nullptr) {                                   \
    nvmmh_func::func =                                                 \
        reinterpret_cast<decltype(&func)>(dlsym(nvmmh_handle, #func)); \
  }

  // TODO: check that version of lib matches version we compiled against
  DEFINE_NVMMH_SYMBOL(nvMatmulHeuristicsGetVersionMajor)
  DEFINE_NVMMH_SYMBOL(nvMatmulHeuristicsGetVersionMinor)
  unsigned lib_major = nvmmh_func::nvMatmulHeuristicsGetVersionMajor();
  unsigned lib_minor = nvmmh_func::nvMatmulHeuristicsGetVersionMinor();
  if (lib_major != NVMMH_VERSION_MAJOR || lib_minor != NVMMH_VERSION_MINOR) {
    TORCH_WARN_ONCE(
        "nvFuser was compiled against nvMatmulHeuristics version ",
        NVMMH_VERSION_MAJOR,
        ".",
        NVMMH_VERSION_MINOR,
        " but found nvMatmulHeuristics shared library version ",
        lib_major,
        ".",
        lib_minor,
        ". Exactly matching versions are required");
  }

  ALL_NVMMH_API_WRAPPER(DEFINE_NVMMH_SYMBOL);

#undef DEFINE_NVMMH_SYMBOL

  return true;
}

#undef ALL_NVMMH_API_WRAPPER

#else // HAS_NVMMH

bool initNVMMH() {
  TORCH_WARN_ONCE("nvFuser was built without nvMatmulHeuristics support");
  return false;
}

#define NVMMH_SAFE_CALL(x)         \
  NVF_THROW("Attempted to call " x \
            " but nvFuser was not built with nvMatmulHeuristics support");

#endif // HAS_NVMMH

GemmTile getProblemSize(Fusion* fusion, SchedulerRuntimeInfo& runtime_info) {
  const std::vector<ScaledMmaOp*> scaled_mmas =
      ir_utils::getOpsOfType<ScaledMmaOp>(fusion);

  NVF_ERROR_EQ(
      scaled_mmas.size(),
      1,
      "Cutlass scheduler expects exactly one ScaledMmaOp");

  ScaledMmaOp* mma = scaled_mmas.front();

  NVF_ERROR(mma->matrix1()->isFusionInput());
  NVF_ERROR(mma->matrix2()->isFusionInput());
  const std::vector<int64_t>& a_shape =
      runtime_info.getInputAllocationSizes(mma->matrix1());
  const std::vector<int64_t>& b_shape =
      runtime_info.getInputAllocationSizes(mma->matrix2());

  NVF_ERROR_EQ(a_shape.size(), 2);
  NVF_ERROR_EQ(b_shape.size(), 2);
  int64_t m = a_shape.front();
  int64_t n = b_shape.back();
  int64_t k = a_shape.back();
  return {m, n, k};
}

int isValidScaledGemmConfig(const nvmmhKernelConfiguration_t* result) {
  std::array<uint16_t, 3> Cta;
  Cta[0] = result->cta[0];
  Cta[1] = result->cta[1];
  Cta[2] = result->cta[2];

  // https://github.com/NVIDIA/cutlass/blob/c6aeb9179c5f74a0fcdbd28527bf4b6ba8c60752/include/cutlass/gemm/collective/builders/sm100_common.inl#L693-L701
  if (Cta[0] != 128) {
    return false;
  }

  if (Cta[1] != 64 && Cta[1] != 128 && Cta[1] != 192 && Cta[1] != 256) {
    return false;
  }

  return true;
}

} // namespace

std::unique_ptr<HeuristicParams> CutlassScheduler::computeHeuristics(
    Fusion* fusion,
    SchedulerRuntimeInfo& runtime_info,
    HeuristicDataCache* data_cache) {
  FUSER_PERF_SCOPE("CutlassScheduler::computeHeuristics");

  auto params = std::make_unique<CutlassParams>();

  // For now, use default parameters
  // TODO: Implement actual heuristics based on problem size, GPU arch, etc.
  // Once libheuristics is available via pycutlass wheel, integrate it here
  if (initNVMMH()) {
#ifdef HAS_NVMMH
    nvmmhHandle_t handle = nullptr;
    NVMMH_SAFE_CALL(nvMatmulHeuristicsCreate(&handle));

    nvmmhBackend_t backend;
    NVMMH_SAFE_CALL(
        nvMatmulHeuristicsBackendCreate(&backend, NVMMH_TARGET_CUTLASS));

    NVMMH_SAFE_CALL(nvMatmulHeuristicsBackendSetCallbackProperty(
        backend,
        NVMMH_CALLBACK_KERNEL_ADDITIONAL_VALIDITY_CHECK,
        isValidScaledGemmConfig));

    // TODO: inspect both inputs and outputs to set problem precision using
    // nvMatmulHeuristics convention
    DataType out_dtype = fusion->outputs().front()->dtype();
    NVF_ERROR(out_dtype == DataType::Half || out_dtype == DataType::BFloat16);
    const std::string precision =
        out_dtype == DataType::BFloat16 ? "OOT" : "OOH";
    const nvmmhMatmulLayout_t layout = NVMMH_MATMUL_LAYOUT_TN_ROW_MAJOR;

    unsigned status = nvmmh_func::nvMatmulHeuristicsLoadInternalDiscoverySet(
        handle,
        precision.c_str(),
        NVMMH_TARGET_CUTLASS,
        layout,
        /*hardwareDescriptor=*/nullptr);
    if (status != NVMMH_STATUS_SUCCESS) {
      TORCH_WARN_ONCE(
          "WARNING: could not load nvMatmulHeuristics internal discovery set "
          "for precision ",
          precision);
    }

    nvmmhKernelConfiguration_t configs[1];

    const GemmTile problem_size = getProblemSize(fusion, runtime_info);
    // TODO: support Batch dimension
    constexpr uint32_t Batch = 1;
    nvmmhMatmulProblem_t problem{
        (uint32_t)problem_size.m,
        (uint32_t)problem_size.n,
        (uint32_t)problem_size.k,
        Batch,
        layout};

    unsigned num_configs = nvmmh_func::nvMatmulHeuristicsGetGemmConfigEx(
        handle,
        precision.c_str(),
        /*flags=*/NVMMH_FLAG_REFINE_CANDIDATES_USING_TIMING_MODEL |
            NVMMH_FLAG_PERF_MODEL_BASED_AUTO_TUNING |
            NVMMH_FLAG_AUTO_TUNE_THE_PERF_MODEL,
        backend,
        /*problemIn=*/&problem,
        /*kernelConfigOut=*/configs,
        /*requestedConfigurations=*/1,
        /*hardwareDescriptor=*/nullptr);

    NVF_ERROR(
        num_configs > 0, "nvMatmulHeuristics did not find any kernel configs");

    const nvmmhKernelConfiguration_t& config = configs[0];

    params->cluster_shape.m = config.cluster[0];
    params->cluster_shape.n = config.cluster[1];
    params->per_sm_tile.m = config.cta[0];
    params->per_sm_tile.n = config.cta[1];
    params->per_sm_tile.k = config.cta[2];
    params->mma_tile.m = config.cta[0] * config.cluster[0];
    params->mma_tile.n = config.cta[1] * config.cluster[1];
    params->mma_tile.k = config.cta[2];

    NVMMH_SAFE_CALL(nvMatmulHeuristicsDestroy(&handle));
#endif // HAS_NVMMH
  }

  if (isDebugDumpEnabled(DebugDumpOption::SchedulerDebug)) {
    debug() << params->toString() << std::endl;
  }

  return params;
}

void CutlassScheduler::schedule(Fusion* fusion, const HeuristicParams* params) {
  FUSER_PERF_SCOPE("CutlassScheduler::schedule");

  NVF_CHECK(
      params->isA<CutlassParams>(), "CutlassScheduler expects CutlassParams");

  // CUTLASS scheduling doesn't involve traditional scheduling operations
  // like split, reorder, etc. The scheduler type is already determined
  // by the time this method is called.

  // We may want to add metadata to the fusion or specific ops to guide CUTLASS
  // code generation
}

bool CutlassScheduler::hasSupportedMatmulPattern(Fusion* fusion) {
  // Only accept ScaledMmaOp for JIT CUTLASS kernels
  bool has_non_scaled_mma = false;
  int64_t num_scaled_mmas = 0;
  for (auto expr : fusion->exprs()) {
    if (!ir_utils::isTvOp(expr)) {
      continue;
    }
    if (expr->isA<ScaledMmaOp>()) {
      num_scaled_mmas++;
    } else {
      has_non_scaled_mma = true;
    }
  }
  // TODO: accept fusions with epilogue
  return num_scaled_mmas == 1 && !has_non_scaled_mma;
}

bool CutlassScheduler::hasSupportedEpilogue(Fusion* fusion) {
  // For now, we support all epilogues that don't involve complex reductions
  // or unsupported operations

  auto matmul_output = findMatmulOutput(fusion);
  if (!matmul_output) {
    return false;
  }

  // Check all uses of the matmul output
  for (auto use : matmul_output->uses()) {
    if (use->isA<ReductionOp>()) {
      // Complex reductions not supported yet
      return false;
    }
    // TODO: Add more checks for unsupported operations
  }

  return true;
}

TensorView* CutlassScheduler::findMatmulOutput(Fusion* fusion) {
  for (auto expr : fusion->exprs()) {
    if (expr->isA<ScaledMmaOp>()) {
      return expr->output(0)->as<TensorView>();
    }
  }
  return nullptr;
}

} // namespace nvfuser
