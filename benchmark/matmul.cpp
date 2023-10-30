// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#include <csrc/exceptions.h>
#include <device_lower/analysis/bank_conflict.h>
#include <executor.h>
#include <fusion.h>
#include <ir/all_nodes.h>
#include <ir/utils.h>
#include <ops/all_ops.h>
#include <optimization/pre_segmenter.h>
#include <scheduler/all_schedulers.h>
#include <scheduler/matmul.h>
#include <scheduler/matmul_heuristic.h>

#include <benchmark/benchmark.h>

#include <cuda_runtime.h>

#include <benchmark/utils.h>
#include <test/utils.h>

using namespace nvfuser;

bool hasRequiredSmemSize(size_t required_size) {
  // Only checking device 0
  return at::cuda::getDeviceProperties(0)->sharedMemPerBlockOptin >=
      required_size;
}

#define NVFUSER_BENCHMARK_ARCH_SMEM_GUARD(                       \
    REQUIRED_MAJOR, REQUIRED_MINOR, SMEM_SIZE, STATE)            \
  if (cudaArchGuardShouldSkip(REQUIRED_MAJOR, REQUIRED_MINOR) || \
      !hasRequiredSmemSize(SMEM_SIZE)) {                         \
    STATE.SkipWithError("Unsupported arch or not enough smem!"); \
    return;                                                      \
  }

// util to track support matmul operand layout.
using MatmulLayout = MmaOptions::MmaLayout;

// TODO: separate compute and schedule definition once the can schedule
//  logic and pattern matching is ready.
void setupMatmul(
    Fusion* fusion,
    MatmulLayout layout,
    MatmulParams params,
    bool turing_or_later // TODO: This is a temporary solution. Remove this!
) {
  // Only hgemm on the initial setup
  auto a = makeContigTensor(2, DataType::Half);
  auto b = makeContigTensor(2, DataType::Half);

  auto c = matmul(a, b, layout, turing_or_later);

  fusion->addInput(a);
  fusion->addInput(b);
  fusion->addOutput(c);

  scheduleMatmul(fusion, params);
}

void checkMatch(at::Tensor expect, at::Tensor result, int64_t k) {
  // tolerance
  double rtol = 1e-6 * k;
  double atol = 1e-6 * k;
  auto is_close = at::isclose(expect, result, rtol, atol);
  auto allclose = is_close.all().item<bool>();
  if (allclose) {
    return;
  }
  NVF_ERROR(is_close.dim() == 2);

  int64_t lower_row, higher_row, lower_col, higher_col;
  for (lower_row = 0; lower_row < is_close.size(0); lower_row++) {
    if (!is_close.select(0, lower_row).all().item<bool>()) {
      break;
    }
  }
  for (higher_row = is_close.size(0) - 1; higher_row >= 0; higher_row--) {
    if (!is_close.select(0, higher_row).all().item<bool>()) {
      break;
    }
  }
  for (lower_col = 0; lower_col < is_close.size(1); lower_col++) {
    if (!is_close.select(1, lower_col).all().item<bool>()) {
      break;
    }
  }
  for (higher_col = is_close.size(1) - 1; higher_col >= 0; higher_col--) {
    if (!is_close.select(1, higher_col).all().item<bool>()) {
      break;
    }
  }

  NVF_CHECK(
      false,
      "Fusion returns wrong results! ",
      "The result tensor has shape [",
      is_close.size(0),
      ",",
      is_close.size(1),
      "]. "
      "Mismatch happens at region result[",
      lower_row,
      ":",
      higher_row + 1,
      ",",
      lower_col,
      ":",
      higher_col + 1,
      "]");
}

static void SingleMatmulBase(
    benchmark::State& benchmark_state,
    MatmulLayout layout,
    MatmulParams params) {
  std::vector<int64_t> input_mnk{
      benchmark_state.range(0),
      benchmark_state.range(1),
      benchmark_state.range(2)};

  // Tensor inputs
  auto inputs =
      matmulAtInput(input_mnk.at(0), input_mnk.at(1), input_mnk.at(2), layout);
  auto expected_output = atMatmul(
      inputs.first.to(at::kDouble), inputs.second.to(at::kDouble), layout);

  // Architecture
  auto properties = at::cuda::getDeviceProperties(inputs.first.get_device());
  bool turing_or_later = properties->major >= 8 ||
      (properties->major == 7 && properties->minor >= 5);

  auto fusion_ptr = std::make_unique<Fusion>();
  auto fusion = fusion_ptr.get();
  FusionGuard fg(fusion);

  // Define fusion graph
  setupMatmul(fusion, layout, params, turing_or_later);

  optimization::OptimizationPass<optimization::PreSegmenter>::runPass(fusion);

  // inputs
  at::manual_seed(0);

  KernelArgumentHolder args = KernelArgumentHolder::createKernelArgumentHolder(
      {inputs.first, inputs.second});

  // Disable magic zero
  CompileParams cparams;
  cparams.enable_magic_zero = false;
  // Always use 32b indexing mode for now.
  cparams.index_type = PrimDataType::Int32;

  // Compile kernel
  auto launch_constraints = LaunchParams();
  FusionExecutor fe;
  fe.compileFusion(fusion, args, launch_constraints, cparams);
  if (turing_or_later) {
    NVF_CHECK(
        getBankConflictInfo(fe.kernel(), launch_constraints).empty(),
        "Shared memory bank conflict not removed.");
  }

  std::vector<c10::IValue> aten_inputs({inputs.first, inputs.second});

  // Warm up run
  auto outputs = fe.runFusion(aten_inputs);
  checkMatch(expected_output, outputs.at(0).to(at::kDouble), input_mnk.at(2));

  runBenchmarkIterations(benchmark_state, &fe, aten_inputs);

  // TODO: FLOPS calculation
}

static void Baseline_Matmul(
    benchmark::State& benchmark_state,
    MatmulLayout layout) {
  std::vector<int64_t> input_mnk{
      benchmark_state.range(0),
      benchmark_state.range(1),
      benchmark_state.range(2)};

  at::manual_seed(0);

  auto inputs =
      matmulAtInput(input_mnk.at(0), input_mnk.at(1), input_mnk.at(2), layout);

  // warm up run
  auto outputs = atMatmul(inputs.first, inputs.second, layout);

  for (auto _ : benchmark_state) {
    clearL2Cache();
    CudaKernelTimer timer;
    outputs = atMatmul(inputs.first, inputs.second, layout);
    benchmark_state.SetIterationTime(timer.elapsed() / 1000.0);
  }
  // Sync everything up before we're finished, don't want to run ahead on the
  // cpu while benchmarking.
  cudaDeviceSynchronize();
}

// Actual benchmarking
// -----------------------------------------------------------------

size_t getSmemSize(GemmTile cta_tile, int stage_number) {
  return ((cta_tile.m * cta_tile.k) + (cta_tile.n * cta_tile.k)) *
      dataTypeSize(DataType::Half) * stage_number;
}

// TODO: this part eventually will be automated by heuristics
MatmulParams getMatmulParams(
    GemmTile cta_tile,
    int stage_number,
    MatmulLayout layout) {
  MatMulTileOptions gemm_tile;
  gemm_tile.cta_tile = cta_tile;
  // TODO: pipe through split K
  gemm_tile.warp_tile = GemmTile(64, 64, cta_tile.k);
  gemm_tile.instruction_tile = GemmTile(16, 16, 16);

  MatmulParams params;
  params.mma_macro = MmaOptions::MacroType::Ampere_16_16_16;
  params.tile_sizes = gemm_tile;
  params.async_gmem_load_operands = true;
  params.double_buffer_options.double_buffer_smem_write = true;
  params.double_buffer_options.double_buffer_smem_read = true;
  params.double_buffer_options.smem_double_buffer_stage = stage_number;

  return params;
}

static void NvFuserScheduler_Matmul(
    benchmark::State& benchmark_state,
    MatmulLayout layout,
    bool partitionedk = false) {
  int num_warps = benchmark_state.range(3);
  int number_of_stage = benchmark_state.range(4);

  auto cta_tile = GemmTile(32 * num_warps, 128, 32);

  auto params = getMatmulParams(cta_tile, number_of_stage, layout);

  NVFUSER_BENCHMARK_ARCH_SMEM_GUARD(
      8, 0, getSmemSize(cta_tile, number_of_stage), benchmark_state);

  // Run benchmark:
  SingleMatmulBase(benchmark_state, layout, params);
}

// ----------------------------- Benchmark Instantiation-------

#define LegacyMs \
  { 2048 }
#define LegacyNs \
  { 3456 }
#define LegacyKs benchmark::CreateDenseRange(512, 4096, /*step=*/512)

// clang-format off
#define TIMMShapes       \
  {                      \
    {1024, 256, 1024},   \
    {8, 128, 8},         \
    {1152, 128, 784},    \
    /* {1152, 48, 1} */  \
    {128, 512, 4096},    \
    /* {192, 1, 672}, */ \
    /* {1, 64, 1}, */    \
    /* {2048, 1, 1}, */  \
    /* {1, 1152, 48}, */ \
    {64, 1152, 384},     \
    {72, 8, 784},        \
    {784, 128, 1152},    \
    {128, 512, 2048},    \
    {64, 384, 1152},     \
    {3136, 72, 8},       \
    {512, 2048, 128},    \
    /* {112, 1, 480}, */ \
    {1024, 512, 1024},   \
    /* {112, 1, 672}, */ \
    {784, 72, 8},        \
    {784, 8, 72},        \
    /* {1, 1, 2048}, */  \
    {1024, 1024, 1024}   \
  }
// clang-format on

#define Layouts \
  { MatmulLayout::TT, MatmulLayout::TN, MatmulLayout::NT, MatmulLayout::NN }
#define NumWarps \
  { 4, 8 }
#define NumStages \
  { 3, 4 }

//! Simple cartesian product of three integers. Used to emulate ArgsProduct
template <typename T>
static std::vector<std::tuple<T, T>> sizeProduct(
    std::vector<T> ms,
    std::vector<T> ns) {
  std::vector<std::tuple<T, T>> sizes;
  for (T m : ms) {
    for (T n : ns) {
      sizes.push_back({m, n});
    }
  }
  return sizes;
}

//! Simple cartesian product of three integers. Used to emulate ArgsProduct
template <typename T>
static std::vector<std::tuple<T, T, T>> sizeProduct(
    std::vector<T> ms,
    std::vector<T> ns,
    std::vector<T> ks) {
  std::vector<std::tuple<T, T, T>> sizes;
  for (T m : ms) {
    for (T n : ns) {
      for (T k : ks) {
        sizes.push_back({m, n, k});
      }
    }
  }
  return sizes;
}

// Use this to apply shape arguments to a benchmark without additional
// NVFuser-specific args. Used for eager benchmarks to avoid redundant
// benchmarks for combinations of num_warps and num_stages
static void MatmulShape(
    benchmark::internal::Benchmark* b,
    std::vector<std::tuple<long int, long int, long int>> sizes) {
  b->ArgNames({"M", "N", "K"});
  for (auto [m, n, k] : sizes) {
    b->Args({m, n, k});
  }
}

// Use this to apply shapes and num_warps. Used for splitk reduction benchmarks
// Note warps is number of warps per block in the associated partitionedk matmul
static void MatmulShapeWarp(
    benchmark::internal::Benchmark* b,
    std::vector<std::tuple<long int, long int>> sizes) {
  b->ArgNames({"M", "N", "warps"});
  for (int num_warps : NumWarps) {
    for (auto [m, n] : sizes) {
      b->Args({m, n, num_warps});
    }
  }
}

// Use this to apply shapes, num_warps, and stages. Used for NVFuser-specific
// benchmarks
static void MatmulShapeWarpStage(
    benchmark::internal::Benchmark* b,
    std::vector<std::tuple<long int, long int, long int>> sizes) {
  b->ArgNames({"M", "N", "K", "warps", "stages"});
  for (int num_warps : NumWarps) {
    for (int num_stages : NumStages) {
      for (auto [m, n, k] : sizes) {
        b->Args({m, n, k, num_warps, num_stages});
      }
    }
  }
}

#define EagerModeBenchmark(layout)                                            \
  BENCHMARK_CAPTURE(                                                          \
      Baseline_Matmul, eagermode_legacyshapes_##layout, MatmulLayout::layout) \
      ->Unit(benchmark::kMicrosecond)                                         \
      ->UseManualTime()                                                       \
      ->Apply([](benchmark::internal::Benchmark* b) {                         \
        MatmulShape(                                                   \
            b, sizeProduct<long int>(LegacyMs, LegacyNs, LegacyKs));          \
      });                                                                     \
  BENCHMARK_CAPTURE(                                                          \
      Baseline_Matmul, eagermode_timmshapes_##layout, MatmulLayout::layout)   \
      ->Unit(benchmark::kMicrosecond)                                         \
      ->UseManualTime()                                                       \
      ->Apply([](benchmark::internal::Benchmark* b) {                         \
        MatmulShape(b, TIMMShapes);                                    \
      });

#define NvfuserMatmulBenchmark(layout)                               \
  BENCHMARK_CAPTURE(                                                 \
      NvFuserScheduler_Matmul,                                       \
      nvfuser_legacyshapes_##layout,                                 \
      MatmulLayout::layout)                                          \
      ->Unit(benchmark::kMicrosecond)                                \
      ->UseManualTime()                                              \
      ->Apply([](benchmark::internal::Benchmark* b) {                \
        MatmulShapeWarpStage(                                 \
            b, sizeProduct<long int>(LegacyMs, LegacyNs, LegacyKs)); \
      });                                                            \
  BENCHMARK_CAPTURE(                                                 \
      NvFuserScheduler_Matmul,                                       \
      nvfuser_timmshapes_##layout,                                   \
      MatmulLayout::layout)                                          \
      ->Unit(benchmark::kMicrosecond)                                \
      ->UseManualTime()                                              \
      ->Apply([](benchmark::internal::Benchmark* b) {                \
        MatmulShapeWarpStage(b, TIMMShapes);                  \
      });

#define ForAllLayouts(run) \
  run(TT);                 \
  run(TN);                 \
  run(NT);                 \
  run(NN);

ForAllLayouts(EagerModeBenchmark);
ForAllLayouts(NvfuserMatmulBenchmark);
