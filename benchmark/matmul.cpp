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
#include <preseg_passes/pre_segmenter.h>
#include <scheduler/all_schedulers.h>
#include <scheduler/matmul.h>
#include <scheduler/matmul_heuristic.h>
#include <utils.h>

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

// TODO: separate compute and schedule definition once the can schedule
//  logic and pattern matching is ready.
void setupMatmul(Fusion* fusion, MmaLayout layout, MatmulParams params) {
  // Only hgemm on the initial setup
  auto a = makeContigTensor(2, DataType::Half);
  auto b = makeContigTensor(2, DataType::Half);

  auto c = matmul(a, b, layout);

  // Cast the output so that we perform an HSH matmul, which is what at::matmul
  // will perform
  auto d = castOp(DataType::Half, c);

  fusion->addInput(a);
  fusion->addInput(b);
  fusion->addOutput(d);

  scheduleMatmul(fusion, params);
}

void checkMatch(at::Tensor expect, at::Tensor result, int64_t k) {
  // tolerance
  double rtol = 1e-4 * k;
  double atol = 1e-4 * k;
  auto ndim = result.ndimension();
  auto is_close = at::isclose(expect, result, rtol, atol);

  auto allclose = is_close.all().item<bool>();
  if (allclose) {
    return;
  }
  NVF_ERROR(is_close.dim() >= 2);

  int64_t lower_row, higher_row, lower_col, higher_col;
  for (lower_row = 0; lower_row < is_close.size(ndim - 2); lower_row++) {
    if (!is_close.select(ndim - 2, lower_row).all().item<bool>()) {
      break;
    }
  }
  for (higher_row = is_close.size(ndim - 2) - 1; higher_row >= 0;
       higher_row--) {
    if (!is_close.select(ndim - 2, higher_row).all().item<bool>()) {
      break;
    }
  }
  for (lower_col = 0; lower_col < is_close.size(ndim - 1); lower_col++) {
    if (!is_close.select(ndim - 1, lower_col).all().item<bool>()) {
      break;
    }
  }
  for (higher_col = is_close.size(ndim - 1) - 1; higher_col >= 0;
       higher_col--) {
    if (!is_close.select(ndim - 1, higher_col).all().item<bool>()) {
      break;
    }
  }

  NVF_CHECK(
      false,
      "Fusion returns wrong results! ",
      "The result tensor has shape [..., ",
      is_close.size(ndim - 2),
      ",",
      is_close.size(ndim - 1),
      "]. "
      "Mismatch happens at region result[...,",
      lower_row,
      ":",
      higher_row + 1,
      ",",
      lower_col,
      ":",
      higher_col + 1,
      "]");
}

//! Compute the index type (either Int32 or, if needed, Int) for an MNK problem
//! size. This function ensures that an Int32 can represent the linear index of
//! any of the contiguous tensors involved in this problem. If not, then Int
//! (double) is returned.
PrimDataType computeIndexType(int m, int n, int k) {
  KernelIndexTypeCompute index_type_helper;
  index_type_helper.addDim(m, k); // A
  index_type_helper.addDim(n, k); // B
  index_type_helper.addDim(m, n); // D
  PrimDataType index_type = index_type_helper.getType();
  if (index_type == DataType::Int) {
    // Notify as this can have a slight perf impact, but is necessary for large
    // inputs
    debug() << "Using int64_t as index type" << std::endl;
  }
  return index_type;
}

static void SingleMatmulBase(
    benchmark::State& benchmark_state,
    MmaLayout layout,
    MatmulParams params) {
  int64_t m = benchmark_state.range(0);
  int64_t n = benchmark_state.range(1);
  int64_t k = benchmark_state.range(2);

  // Tensor inputs
  auto inputs = matmulAtInput2D(m, n, k, layout);
  auto expected_output = atMatmul(
      inputs.first.to(at::kDouble), inputs.second.to(at::kDouble), layout);

  auto fusion_ptr = std::make_unique<Fusion>();
  auto fusion = fusion_ptr.get();
  FusionGuard fg(fusion);

  // Define fusion graph
  setupMatmul(fusion, layout, params);

  preseg_passes::OptimizationPass<preseg_passes::PreSegmenter>::runPass(fusion);

  // inputs
  at::manual_seed(0);

  KernelArgumentHolder args = KernelArgumentHolder::createKernelArgumentHolder(
      {inputs.first, inputs.second});

  // Disable magic zero
  CompileParams cparams;
  cparams.enable_magic_zero = false;
  cparams.index_type = computeIndexType(m, n, k);

  // Compile kernel
  auto launch_constraints = LaunchParams();
  FusionExecutor fe;
  fe.compileFusion(fusion, args, launch_constraints, cparams);
  NVF_CHECK(
      getBankConflictInfo(fe.kernel(), launch_constraints).empty(),
      "Shared memory bank conflict not removed.");

  std::vector<c10::IValue> aten_inputs({inputs.first, inputs.second});

  // Warm up run
  auto outputs = fe.runFusion(aten_inputs);
  checkMatch(expected_output, outputs.at(0).to(at::kDouble), k);

  runBenchmarkIterations(benchmark_state, &fe, aten_inputs);

  // TODO: FLOPS calculation
}

static void Baseline_Matmul(
    benchmark::State& benchmark_state,
    MmaLayout layout) {
  std::vector<int64_t> input_mnk{
      benchmark_state.range(0),
      benchmark_state.range(1),
      benchmark_state.range(2)};

  bool allow_half_reduction = (bool)benchmark_state.range(3);

  at::manual_seed(0);

  auto inputs = matmulAtInput2D(
      input_mnk.at(0), input_mnk.at(1), input_mnk.at(2), layout);

  // Disable reduced-precision reduction for fair comparison since we do not use
  // it in nvFuser
  at::globalContext().setAllowFP16ReductionCuBLAS(allow_half_reduction);

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
    MmaLayout layout,
    int splitk_factor = 1) {
  MatMulTileOptions gemm_tile;
  gemm_tile.cta_tile = cta_tile;
  // TODO: pipe through split K
  gemm_tile.warp_tile = GemmTile(64, 64, cta_tile.k);
  gemm_tile.instruction_tile = GemmTile(16, 16, 16);

  MatmulParams params;
  params.mma_macro = MmaMacro::Ampere_16_16_16;
  params.tile_sizes = gemm_tile;
  params.async_gmem_load_operands = true;
  params.double_buffer_options.double_buffer_smem_write = true;
  params.double_buffer_options.double_buffer_smem_read = true;
  params.double_buffer_options.smem_double_buffer_stage = stage_number;
  params.splitk_factor = splitk_factor;

  return params;
}

// Compute splitk_factor that would fill a block
int computeAutoSplitKFactor(
    int M,
    int N,
    int tile_M,
    int tile_N,
    int num_SMs = -1) {
  if (num_SMs == -1) {
    num_SMs = getNumSMs();
  }
  int num_blocks = ceilDiv(M, tile_M) * ceilDiv(N, tile_N);
  NVF_CHECK(
      num_SMs % num_blocks == 0,
      "Matrix size ",
      M,
      " by ",
      N,
      " with tile size ",
      tile_M,
      " by ",
      tile_N,
      " uses ",
      num_blocks,
      " blocks which does not divide evenly in to ",
      num_SMs,
      " SMs");
  return num_SMs / num_blocks;
}

// This performs the splitk matmul WITHOUT any outer reduction, which is useful
// for comparing against the first kernel in Cutlass's two-kernel split-K.
static void SingleMatmulPartitionedK(
    benchmark::State& benchmark_state,
    MmaLayout layout,
    MatmulParams params,
    int64_t splitk_factor) {
  int64_t M = benchmark_state.range(0);
  int64_t N = benchmark_state.range(1);
  int64_t K = benchmark_state.range(2);

  // Pad K to next multiple of both splitk_factor * 8 (for alignment of fp16
  // values)
  if (K % (splitk_factor * 8) != 0) {
    int64_t pad_amount = splitk_factor * 8 - (K % (splitk_factor * 8));
    K += pad_amount;
    std::cerr << "Padding K to " << K
              << " to satisfy 16-byte alignment requirement" << std::endl;
  }
  int64_t Ki = K / splitk_factor;

  at::manual_seed(0);

  auto fusion_ptr = std::make_unique<Fusion>();
  auto fusion = fusion_ptr.get();
  FusionGuard fg(fusion);

  // Define fusion graph
  auto a = makeContigTensor(3, DataType::Half);
  auto b = makeContigTensor(3, DataType::Half);
  fusion->addInput(a);
  fusion->addInput(b);

  // batch matmul
  auto c = matmul(a, b, layout);

  fusion->addOutput(c);

  scheduleMatmul(fusion, params);

  at::Tensor aten_a = matmulAtInput2D(
      layout, TensorMatmulPos::A, at::kHalf, M, N, Ki, splitk_factor);
  at::Tensor aten_b = matmulAtInput2D(
      layout, TensorMatmulPos::B, at::kHalf, M, N, Ki, splitk_factor);
  std::vector<c10::IValue> aten_inputs = {aten_a, aten_b};
  at::Tensor expected_output = splitkLikeAtMatmul(
      aten_a.to(at::kDouble), aten_b.to(at::kDouble), layout);

  auto args = KernelArgumentHolder::createKernelArgumentHolder(aten_inputs);

  // Disable magic zero
  CompileParams cparams;
  cparams.enable_magic_zero = false;
  cparams.index_type = computeIndexType(M, N, K);

  // Compile kernel
  FusionExecutor fe;
  auto lparams = LaunchParams();
  fe.compileFusion(fusion, args, lparams, cparams);
  NVF_CHECK(
      getBankConflictInfo(fe.kernel(), lparams).empty(),
      "Shared memory bank conflict not removed.");

  // Warm up run
  auto outputs = fe.runFusion(aten_inputs);

  checkMatch(expected_output, outputs.at(0).to(at::kDouble), Ki);

  runBenchmarkIterations(benchmark_state, &fe, aten_inputs);

  // TODO: FLOPS calculation
}

static void NvFuserScheduler_Matmul(
    benchmark::State& benchmark_state,
    MmaLayout layout,
    int splitk_factor = 1,
    bool partitionedk = false) {
  int num_warps = benchmark_state.range(3);
  int number_of_stage = benchmark_state.range(4);

  auto cta_tile = GemmTile(32 * num_warps, 128, 32);

  if (splitk_factor == -1) {
    int M = benchmark_state.range(0);
    int N = benchmark_state.range(1);
    splitk_factor = computeAutoSplitKFactor(M, N, cta_tile.m, cta_tile.n);
  }

  auto params = getMatmulParams(
      cta_tile, number_of_stage, layout, partitionedk ? 1 : splitk_factor);

  NVFUSER_BENCHMARK_ARCH_SMEM_GUARD(
      8, 0, getSmemSize(cta_tile, number_of_stage), benchmark_state);

  // Run benchmark:
  if (partitionedk) {
    SingleMatmulPartitionedK(benchmark_state, layout, params, splitk_factor);
  } else {
    SingleMatmulBase(benchmark_state, layout, params);
  }
}

// This is the second kernel in a two-kernel split-K.
// The input is a contiguous [M, N, splitk_factor] tensor.
// The kernel sums the last dimension.
static void NvFuserScheduler_MatmulSplitKReduction(
    benchmark::State& benchmark_state,
    int64_t splitk_factor = -1) {
  int64_t M = benchmark_state.range(0);
  int64_t N = benchmark_state.range(1);

  if (splitk_factor == -1) {
    // Assumes tile size is (M, 128)
    splitk_factor = computeAutoSplitKFactor(M, N, M, 128);
  }

  at::manual_seed(0);

  auto fusion_ptr = std::make_unique<Fusion>();
  auto fusion = fusion_ptr.get();
  FusionGuard fg(fusion);

  // Note: although the matmul inputs may be Half, PartitionedK output is in
  // the accumulator type (Float), so we reduce Floats.
  auto c = makeContigTensor(3, DataType::Float);
  fusion->addInput(c);
  auto d = castOp(DataType::Float, c);
  auto e = sum(d, {-1});
  fusion->addOutput(e);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);

  auto aten_c = at::randn({M, N, splitk_factor}, options);
  std::vector<c10::IValue> aten_inputs = {aten_c};

  auto reduction_params = getReductionHeuristics(fusion, aten_inputs);
  NVF_CHECK(reduction_params, "Reduction schedule failed");
  scheduleReduction(fusion, *reduction_params);
  auto lparams = reduction_params->lparams; // copy LaunchParams

  auto expected_output = aten_c.to(at::kDouble).sum(-1);

  // Disable magic zero
  CompileParams cparams;
  cparams.enable_magic_zero = false;
  cparams.index_type = computeIndexType(M, N * splitk_factor, 1);

  KernelArgumentHolder args =
      KernelArgumentHolder::createKernelArgumentHolder(aten_inputs);

  // Compile kernel
  FusionExecutor fe;
  fe.compileFusion(fusion, args, lparams, cparams);

  NVF_CHECK(
      getBankConflictInfo(fe.kernel(), lparams).empty(),
      "Shared memory bank conflict not removed.");

  // Warm up run
  auto outputs = fe.runFusion(aten_inputs, lparams);

  checkMatch(expected_output, outputs.at(0).to(at::kDouble), splitk_factor);

  runBenchmarkIterations(benchmark_state, &fe, aten_inputs, lparams);

  // TODO: FLOPS calculation
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
    {1024, 1024, 1024},  \
    /* NanoGPT bwd sizes */  \
    {1024, 2048, 4096},      \
    {1024, 2048, 50304}     \
  }

#define SplitKSpecificShapes \
  {                          \
    /* NanoGPT bwd sizes */  \
    {1024, 2048, 4096},      \
    {1024, 2048, 50304},     \
    /* Symmetric M,N to make comparison in TN/NT fair with eager due to transpose/swap */ \
    {1024, 1024, 4096},     \
    {1024, 1024, 50304},     \
    /* Sizes mentioned by Michel */ \
    {136, 184, 175704},     \
    /* Other */ \
    {128, 128, 262144}     \
  }
// clang-format on

// For 4warp splitk experiments, we use a tile size of (128, 128). To avoid
// wave quantization, we look at sizes that are integer multiples of the block
// size. Below you will find all the factors of 108, which is the number of SMs
// on an A100. Note that 8warp uses tile size (256, 128) in which case SplitKMs
// should be changed to 256.
#define SplitKMs \
  { 128, 256 }

// Dynamically find all valid values of N that divide number of SMs
static std::vector<long int> splitKNs(long int tileN = 128) {
  const long int numSMs = getNumSMs();
  std::vector<long int> Ns;
  for (long int N : c10::irange(numSMs + 1)) {
    if (N > 0 && numSMs % N == 0) {
      Ns.push_back(N * tileN);
    }
  }
  return Ns;
}
#define SplitKKs \
  { 65536 }

#define Layouts \
  { MmaLayout::TT, MmaLayout::TN, MmaLayout::NT, MmaLayout::NN }
#define NumWarps \
  { 4, 8 }
#define NumStages \
  { 3, 4, 5 }

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
static void MatmulShapeEager(
    benchmark::internal::Benchmark* b,
    std::vector<std::tuple<long int, long int, long int>> sizes) {
  b->ArgNames({"M", "N", "K", "half_reduction"});
  for (auto [m, n, k] : sizes) {
    for (bool allow_half_reduction : {false, true}) {
      b->Args({m, n, k, allow_half_reduction});
    }
  }
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

// Use this to apply shapes, num_warps, and stages. Used for NVFuser-specific
// benchmarks
static void MatmulShapeWarpStage(
    benchmark::internal::Benchmark* b,
    std::vector<std::tuple<long int, long int, long int>> sizes) {
  b->ArgNames({"M", "N", "K", "warps", "stages"});
  for (long int num_warps : NumWarps) {
    for (long int num_stages : NumStages) {
      for (auto [m, n, k] : sizes) {
        b->Args({m, n, k, num_warps, num_stages});
      }
    }
  }
}

// Use this to for auto splitk. This is like MatmulShapeWarpStage, but sets M
// to match tile size which is determined by num warps
static void MatmulShapeWarpStageAutoSplitK(benchmark::internal::Benchmark* b) {
  b->ArgNames({"M", "N", "K", "warps", "stages"});
  for (long int num_warps : NumWarps) {
    long int m = num_warps * 32;
    for (long int num_stages : NumStages) {
      for (long int n : splitKNs()) {
        for (long int k : SplitKKs) {
          b->Args({m, n, k, num_warps, num_stages});
        }
      }
    }
  }
}

// Use this for manual splitk.
static void MatmulShapeWarpStageSpecificSplitK(
    benchmark::internal::Benchmark* b) {
  b->ArgNames({"M", "N", "K", "warps", "stages", "splitk_factor"});
  for (long int num_warps : NumWarps) {
    for (long int num_stages : NumStages) {
      for (auto [m, n, k] :
           std::vector<std::tuple<int, int, int>>(SplitKSpecificShapes)) {
        for (auto splitk_factor : {2, 3, 4, 5, 6}) {
          b->Args({m, n, k, num_warps, num_stages, splitk_factor});
        }
      }
    }
  }
}

#define EagerModeBenchmark(layout)                                         \
  BENCHMARK_CAPTURE(                                                       \
      Baseline_Matmul, eagermode_legacyshapes_##layout, MmaLayout::layout) \
      ->Unit(benchmark::kMicrosecond)                                      \
      ->UseManualTime()                                                    \
      ->Apply([](benchmark::internal::Benchmark* b) {                      \
        return MatmulShapeEager(                                           \
            b, sizeProduct<long int>(LegacyMs, LegacyNs, LegacyKs));       \
      });                                                                  \
  BENCHMARK_CAPTURE(                                                       \
      Baseline_Matmul, eagermode_timmshapes_##layout, MmaLayout::layout)   \
      ->Unit(benchmark::kMicrosecond)                                      \
      ->UseManualTime()                                                    \
      ->Apply([](benchmark::internal::Benchmark* b) {                      \
        return MatmulShapeEager(b, TIMMShapes);                            \
      });                                                                  \
  BENCHMARK_CAPTURE(                                                       \
      Baseline_Matmul, eagermode_splitkshapes_##layout, MmaLayout::layout) \
      ->Unit(benchmark::kMicrosecond)                                      \
      ->UseManualTime()                                                    \
      ->Apply([](benchmark::internal::Benchmark* b) {                      \
        return MatmulShapeEager(                                           \
            b, sizeProduct<long int>(SplitKMs, splitKNs(), SplitKKs));     \
      });

#define NvfuserMatmulBenchmark(layout)                                 \
  BENCHMARK_CAPTURE(                                                   \
      NvFuserScheduler_Matmul,                                         \
      nvfuser_nosplitk_legacyshapes_##layout,                          \
      MmaLayout::layout)                                               \
      ->Unit(benchmark::kMicrosecond)                                  \
      ->UseManualTime()                                                \
      ->Apply(MatmulShapeWarpStageAutoSplitK);                         \
  BENCHMARK_CAPTURE(                                                   \
      NvFuserScheduler_Matmul,                                         \
      nvfuser_nosplitk_timmshapes_##layout,                            \
      MmaLayout::layout)                                               \
      ->Unit(benchmark::kMicrosecond)                                  \
      ->UseManualTime()                                                \
      ->Apply([](benchmark::internal::Benchmark* b) {                  \
        return MatmulShapeWarpStage(b, TIMMShapes);                    \
      });                                                              \
  BENCHMARK_CAPTURE(                                                   \
      NvFuserScheduler_Matmul,                                         \
      nvfuser_nosplitk_splitkshapes_##layout,                          \
      MmaLayout::layout)                                               \
      ->Unit(benchmark::kMicrosecond)                                  \
      ->UseManualTime()                                                \
      ->Apply([](benchmark::internal::Benchmark* b) {                  \
        return MatmulShapeWarpStage(                                   \
            b, sizeProduct<long int>(SplitKMs, splitKNs(), SplitKKs)); \
      });

#define ForAllLayouts(run) \
  run(TT);                 \
  run(TN);                 \
  run(NT);                 \
  run(NN);

#define AutoSplitKBenchmark(layout)   \
  BENCHMARK_CAPTURE(                  \
      NvFuserScheduler_Matmul,        \
      nvfuser_auto_splitk_##layout,   \
      MmaLayout::layout,              \
      -1)                             \
      ->Unit(benchmark::kMicrosecond) \
      ->UseManualTime()               \
      ->Apply(MatmulShapeWarpStageAutoSplitK);

#define AutoPartitionedKBenchmark(layout) \
  BENCHMARK_CAPTURE(                      \
      NvFuserScheduler_Matmul,            \
      nvfuser_auto_partitionedk_##layout, \
      MmaLayout::layout,                  \
      -1,                                 \
      true)                               \
      ->Unit(benchmark::kMicrosecond)     \
      ->UseManualTime()                   \
      ->Apply(MatmulShapeWarpStageAutoSplitK);

static void NvFuserScheduler_Matmul_Manual(
    benchmark::State& benchmark_state,
    MmaLayout layout) {
  int splitk_factor = benchmark_state.range(5);
  NvFuserScheduler_Matmul(
      benchmark_state, layout, splitk_factor, /*partitionedk=*/false);
}

#define SpecificSplitKBenchmark(layout) \
  BENCHMARK_CAPTURE(                    \
      NvFuserScheduler_Matmul_Manual,   \
      nvfuser_splitk_##layout,          \
      MmaLayout::layout)                \
      ->Unit(benchmark::kMicrosecond)   \
      ->UseManualTime()                 \
      ->Apply(MatmulShapeWarpStageSpecificSplitK);

ForAllLayouts(EagerModeBenchmark);
ForAllLayouts(NvfuserMatmulBenchmark);
ForAllLayouts(AutoSplitKBenchmark);
ForAllLayouts(SpecificSplitKBenchmark);
ForAllLayouts(AutoPartitionedKBenchmark);

// Note: SplitK Reduction benchmarks are parametrized only by M, N. The splitk
// factor is deduced automatically from N
BENCHMARK_CAPTURE(
    NvFuserScheduler_MatmulSplitKReduction,
    nvfuser_auto_splitkreduction,
    -1)
    ->Unit(benchmark::kMicrosecond)
    ->UseManualTime()
    ->Apply([](benchmark::internal::Benchmark* b) {
      b->ArgNames({"M", "N"});
      for (long int num_warps : NumWarps) {
        long int m = num_warps * 32;
        for (long int n : splitKNs()) {
          b->Args({m, n});
        }
      }
    });
