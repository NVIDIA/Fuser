// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on

#include <inlining.h>
#include <ops/all_ops.h>
#include <tests/cpp/utils.h>
#include <tests/cpp/validator.h>

namespace nvfuser {

namespace {
class DoubleBufferingTest : public NVFuserTest {};
} // anonymous namespace

template <typename data_type>
void compare(int64_t tensor_dim, at::Tensor result, at::Tensor reference) {
  at::Tensor reference_cpu_data = reference.cpu();
  at::Tensor result_cpu_data = result.cpu();

  auto reference_cpu = reference_cpu_data.accessor<data_type, 1>();
  auto result_cpu = result_cpu_data.accessor<data_type, 1>();

  constexpr double tolerance = 1e-4;
  for (int64_t pos = 0; pos < tensor_dim; ++pos) {
    if (fabs((double)result_cpu[pos] - (double)reference_cpu[pos]) >
        tolerance) {
      std::cout << "[" << pos << "] - result: " << result_cpu[pos]
                << " | reference: " << reference_cpu[pos] << std::endl;
    }
  }
}

template <typename data_type>
void compare(
    int64_t tensor_outer_dim,
    int64_t tensor_inner_dim,
    at::Tensor result,
    at::Tensor reference) {
  at::Tensor reference_cpu_data = reference.cpu();
  at::Tensor result_cpu_data = result.cpu();

  auto reference_cpu = reference_cpu_data.accessor<data_type, 2>();
  auto result_cpu = result_cpu_data.accessor<data_type, 2>();

  constexpr double tolerance = 1e-4;
  for (int64_t out_pos = 0; out_pos < tensor_outer_dim; ++out_pos) {
    for (int64_t in_pos = 0; in_pos < tensor_inner_dim; ++in_pos) {
      if (fabs(
              (double)result_cpu[out_pos][in_pos] -
              (double)result_cpu[out_pos][in_pos]) > tolerance) {
        std::cout << "[" << out_pos << ", " << in_pos
                  << "] - result: " << result_cpu[out_pos][in_pos]
                  << " | ref: " << reference_cpu[out_pos][in_pos] << std::endl;
      }
    }
  }
}

TEST_F(DoubleBufferingTest, TmaDoubleBuffering1d) {
  NVFUSER_TEST_CUDA_ARCH_GUARD(9, 0);

  std::unique_ptr<Fusion> fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());

  TensorView* tv0 = makeContigTensor(1);
  fusion->addInput(tv0);

  TensorView* tv1 = exp(tv0);
  fusion->addOutput(tv1);

  TensorView* tv2 = tv0->cacheAfter(LoadStoreOpType::CpAsyncBulkTensorTile);
  tv2->setMemoryType(MemoryType::Shared);

  TensorView* reference = tv1;

  // [M] -> [M/bid, bid]
  constexpr size_t bulk_inner_dim = 32;
  reference->split(-1, bulk_inner_dim);

  // Propagate Transformations
  TransformPropagatorWithCheck propagator(reference);
  MaxRootDomainInfoSpanningTree(reference).traverse(&propagator);

  tv0->computeAt(tv1, 1);

  // Double Buffer with TMA loads
  tv2->axis(-1)->parallelize(ParallelType::Bulk);
  tv2->circularBuffer(/*stage=*/3);

  std::vector<int64_t> tensor_sizes = {10, 32, 50, 128};
  for (int64_t tensor_dim : tensor_sizes) {
    auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
    at::Tensor t0 = at::randn({tensor_dim}, options);
    at::Tensor t1 = at::exp(t0);

    FusionExecutor fe;
    CompileParams index32bit{DataType::Int32, 255, false};
    fe.compileFusion(fusion.get(), {t0}, {}, index32bit);

    std::vector<at::Tensor> cg_outputs = fe.runFusion({t0});
    compare<float>(tensor_dim, cg_outputs.front(), t1);
    testValidate(fusion.get(), cg_outputs, {t0}, {t1}, __LINE__, __FILE__);
  }
}

TEST_F(DoubleBufferingTest, TmaDoubleBufferingUnroll) {
  NVFUSER_TEST_CUDA_ARCH_GUARD(9, 0);

  std::unique_ptr<Fusion> fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());

  TensorView* tv0 = makeContigTensor(1);
  fusion->addInput(tv0);

  TensorView* tv1 = exp(tv0);
  fusion->addOutput(tv1);

  TensorView* tv2 = tv0->cacheAfter(LoadStoreOpType::CpAsyncBulkTensorTile);
  tv2->setMemoryType(MemoryType::Shared);

  TensorView* reference = tv1;

  constexpr size_t unroll_dim = 4;
  constexpr size_t bulk_inner_dim = 32;

  // [M] -> [M/bid, bid]
  reference->split(-1, bulk_inner_dim);
  // [M/bid, bid] -> [M/bid/unroll, unroll, bid]
  reference->split(0, unroll_dim);

  // Propagate Transformations
  TransformPropagatorWithCheck propagator(reference);
  MaxRootDomainInfoSpanningTree(reference).traverse(&propagator);

  tv0->computeAt(tv1, 1);

  tv1->axis(1)->parallelize(ParallelType::Unroll);

  // Double Buffer with TMA loads
  tv2->axis(-1)->parallelize(ParallelType::Bulk);
  tv2->circularBuffer(/*stage=*/3);

  std::vector<int64_t> tensor_sizes = {10, 32, 50, 128};
  for (int64_t tensor_dim : tensor_sizes) {
    auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
    at::Tensor t0 = at::randn({tensor_dim}, options);
    at::Tensor t1 = at::exp(t0);

    FusionExecutor fe;
    CompileParams index32bit{DataType::Int32, 255, false};
    fe.compileFusion(fusion.get(), {t0}, {}, index32bit);

    std::vector<at::Tensor> cg_outputs = fe.runFusion({t0});
    compare<float>(tensor_dim, cg_outputs.front(), t1);
    testValidate(fusion.get(), cg_outputs, {t0}, {t1}, __LINE__, __FILE__);
  }
}

TEST_F(DoubleBufferingTest, TmaDoubleBufferingUnswitch) {
  NVFUSER_TEST_CUDA_ARCH_GUARD(9, 0);

  std::unique_ptr<Fusion> fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());

  TensorView* tv0 = makeContigTensor(1);
  fusion->addInput(tv0);

  TensorView* tv1 = exp(tv0);
  fusion->addOutput(tv1);

  TensorView* tv2 = tv0->cacheAfter(LoadStoreOpType::CpAsyncBulkTensorTile);
  tv2->setMemoryType(MemoryType::Shared);

  TensorView* reference = tv1;

  constexpr size_t unroll_dim = 4;
  constexpr size_t bulk_inner_dim = 32;

  // [M] -> [M/bid, bid]
  reference->split(-1, bulk_inner_dim);
  // [M/bid, bid] -> [M/bid/unroll, unroll, bid]
  reference->split(0, unroll_dim);

  // Propagate Transformations
  TransformPropagatorWithCheck propagator(reference);
  MaxRootDomainInfoSpanningTree(reference).traverse(&propagator);

  tv0->computeAt(tv1, 1);

  tv1->axis(1)->parallelize(ParallelType::Unswitch);

  // Double Buffer with TMA loads
  tv2->axis(-1)->parallelize(ParallelType::Bulk);
  tv2->circularBuffer(/*stage=*/3);

  std::vector<int64_t> tensor_sizes = {10, 32, 50, 128};
  for (int64_t tensor_dim : tensor_sizes) {
    auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
    at::Tensor t0 = at::randn({tensor_dim}, options);
    at::Tensor t1 = at::exp(t0);

    FusionExecutor fe;
    CompileParams index32bit{DataType::Int32, 255, false};
    fe.compileFusion(fusion.get(), {t0}, {}, index32bit);

    std::vector<at::Tensor> cg_outputs = fe.runFusion({t0});
    compare<float>(tensor_dim, cg_outputs.front(), t1);
    testValidate(fusion.get(), cg_outputs, {t0}, {t1}, __LINE__, __FILE__);
  }
}

TEST_F(DoubleBufferingTest, TmaDoubleBuffering2d) {
  NVFUSER_TEST_CUDA_ARCH_GUARD(9, 0);

  std::unique_ptr<Fusion> fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());

  TensorView* tv0 = makeContigTensor(2);
  fusion->addInput(tv0);

  TensorView* tv1 = exp(tv0);
  fusion->addOutput(tv1);

  TensorView* tv2 = tv0->cacheAfter(LoadStoreOpType::CpAsyncBulkTensorTile);
  tv2->setMemoryType(MemoryType::Shared);

  TensorView* reference = tv1;
  constexpr int64_t tma_outer_dim = 4;
  constexpr int64_t tma_inner_dim = 32;
  // [M, N] -> [M, N/bid, bid]
  reference->split(-1, tma_inner_dim);
  // [M, N/bid, bid] -> [M/bod, bod, N/bid, bid]
  reference->split(0, tma_outer_dim);
  // [M/bod, bod, N/bid, bid] -> [M/bod, N/bid, bod, bid]
  reference->reorder({{-2, -3}});

  // Propagate TMA transform
  TransformPropagatorWithCheck propagator(reference);
  MaxRootDomainInfoSpanningTree(reference).traverse(&propagator);

  // Apply computeAt for TMA cache
  tv0->computeAt(tv1, 2);

  // Merge TMA tile and Parallelize
  reference->merge(-2, -1);
  reference->split(-1, 128);
  reference->axis(0)->parallelize(ParallelType::BIDx);
  reference->axis(-1)->parallelize(ParallelType::TIDx);

  // Double Buffer with TMA loads
  tv2->axis(0)->parallelize(ParallelType::BIDx);
  tv2->axis(-1)->parallelize(ParallelType::Bulk);
  tv2->axis(-2)->parallelize(ParallelType::Bulk);
  tv2->circularBuffer(/*stage=*/3);

  std::vector<int64_t> outer_tensor_sizes = {10, 32, 50, 128};
  // NOTE: Multiple of 16 required for inner dimension
  std::vector<int64_t> inner_tensor_sizes = {16, 32, 128};
  for (int64_t tensor_outer_dim : outer_tensor_sizes) {
    for (int64_t tensor_inner_dim : inner_tensor_sizes) {
      auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
      at::Tensor t0 = at::randn({tensor_outer_dim, tensor_inner_dim}, options);
      at::Tensor t1 = at::exp(t0);

      FusionExecutor fe;
      CompileParams index32bit{DataType::Int32, 255, false};
      fe.compileFusion(fusion.get(), {t0}, {}, index32bit);

      std::vector<at::Tensor> cg_outputs = fe.runFusion({t0});
      compare<float>(tensor_outer_dim, tensor_inner_dim, cg_outputs.front(), t1);
      testValidate(fusion.get(), cg_outputs, {t0}, {t1}, __LINE__, __FILE__);
    }
  }
}

TEST_F(DoubleBufferingTest, TmaDoubleBufferingPointwise) {
  NVFUSER_TEST_CUDA_ARCH_GUARD(9, 0);
  std::unique_ptr<Fusion> fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());

  TensorView* tv0 = makeContigTensor(2);
  TensorView* tv1 = makeContigTensor(2);
  fusion->addInput(tv0);
  fusion->addInput(tv1);

  TensorView* tv2 = add(tv0, tv1);
  fusion->addOutput(tv2);

  // Use TMA to load TV0 into shared memory
  TensorView* tv3 = tv0->cacheAfter(LoadStoreOpType::CpAsyncBulkTensorTile);
  tv3->setMemoryType(MemoryType::Shared);

  // Load TV0 into shared memory
  TensorView* tv4 = tv1->cacheAfter();
  tv4->setMemoryType(MemoryType::Shared);

  TensorView* reference = tv2;
  constexpr int64_t bulk_inner_dim = 32;
  // [M, N] -> [M, N/bid, bid]
  reference->split(-1, bulk_inner_dim);

  TransformPropagatorWithCheck propagator(reference);
  MaxRootDomainInfoSpanningTree(reference).traverse(&propagator);

  tv0->computeAt(tv2, 2);
  tv1->computeAt(tv2, 2);

  // Ciruclar Buffer with TMA loads
  tv3->axis(0)->parallelize(ParallelType::BIDx);
  tv3->axis(2)->parallelize(ParallelType::Bulk);
  tv3->circularBuffer(/*stage=*/3);

  // Ciruclar Buffer with set operation
  tv4->axis(0)->parallelize(ParallelType::BIDx);
  tv4->circularBuffer(/*stage=*/3);

  // split reference to parallelize TMA tile
  reference->split(-1, 32);
  reference->axis(0)->parallelize(ParallelType::BIDx);
  reference->axis(-1)->parallelize(ParallelType::TIDx);

  std::vector<int64_t> outer_tensor_sizes = {10, 32, 50, 128};
  // NOTE: Multiple of 16 required for inner dimension
  std::vector<int64_t> inner_tensor_sizes = {16, 32, 128};
  for (int64_t tensor_outer_dim : outer_tensor_sizes) {
    for (int64_t tensor_inner_dim : inner_tensor_sizes) {
      auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
      at::Tensor t0 = at::randn({tensor_outer_dim, tensor_inner_dim}, options);
      at::Tensor t1 = at::randn({tensor_outer_dim, tensor_inner_dim}, options);
      at::Tensor t2 = t0 + t1;

      FusionExecutor fe;
      CompileParams index32bit{DataType::Int32, 255, false};
      fe.compileFusion(fusion.get(), {t0, t1}, {}, index32bit);

      std::vector<at::Tensor> cg_outputs = fe.runFusion({t0, t1});
      testValidate(fusion.get(), cg_outputs, {t0, t1}, {t2}, __LINE__, __FILE__);
    }
  }
}

TEST_F(DoubleBufferingTest, TmaDoubleBufferingReduction) {
  NVFUSER_TEST_CUDA_ARCH_GUARD(9, 0);

  std::unique_ptr<Fusion> fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());

  TensorView* tv0 = makeContigTensor(2);
  fusion->addInput(tv0);

  TensorView* tv1 = sum(tv0, {-1});
  fusion->addOutput(tv1);

  TensorView* tv2 = tv0->cacheAfter(LoadStoreOpType::CpAsyncBulkTensorTile);
  tv2->setMemoryType(MemoryType::Shared);

  TensorView* reference = tv1;
  // TODO If examples_per_cta == num_stages, then cuda kernel is malformed.

  constexpr int64_t examples_per_cta = 4;
  constexpr int64_t bulk_inner_dim = 256;
  // [M, N] -> [M/epc, epc, N]
  reference->split(0, examples_per_cta);
  // [M/epc, epc, N] -> [M/epc, epc, N/bid, bid]
  reference->split(-1, bulk_inner_dim);

  TransformPropagatorWithCheck propagator(reference);
  MaxRootDomainInfoSpanningTree(reference).traverse(&propagator);

  // [M/epc, epc, N/bid, bid] -> [M/epc, epc, N]
  reference->merge(-2, -1);
  // [M/epc, epc, N] -> [M/epc, epc, N/tdx, tdx]
  constexpr int64_t tdx = 128;
  reference->split(-1, tdx);

  reference->axis(0)->parallelize(ParallelType::BIDx);
  reference->axis(-1)->parallelize(ParallelType::TIDx);

  inlineMost();

  // Double Buffer with TMA loads
  tv2->axis(0)->parallelize(ParallelType::BIDx);
  tv2->axis(-1)->parallelize(ParallelType::Bulk);
  tv2->circularBuffer(/*stage=*/2);

  std::vector<int64_t> outer_tensor_sizes = {10, 32, 50, 128};
  // NOTE: Multiple of 16 required for inner dimension
  std::vector<int64_t> inner_tensor_sizes = {16, 32, 128};
  for (int64_t tensor_outer_dim : outer_tensor_sizes) {
    for (int64_t tensor_inner_dim : inner_tensor_sizes) {
      auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
      at::Tensor t0 = at::randn({tensor_outer_dim, tensor_inner_dim}, options);
      at::Tensor t1 = sum(t0, {-1});

      FusionExecutor fe;
      CompileParams index32bit{DataType::Int32, 255, false};
      fe.compileFusion(fusion.get(), {t0}, {}, index32bit);

      std::vector<at::Tensor> cg_outputs = fe.runFusion({t0});
      testValidate(fusion.get(), cg_outputs, {t0}, {t1}, __LINE__, __FILE__);
    }
  }
}

TEST_F(DoubleBufferingTest, TmaDoubleBufferingPersistent) {
  NVFUSER_TEST_CUDA_ARCH_GUARD(9, 0);

  constexpr int64_t dim0 = 1024;
  constexpr int64_t dim1 = 4096; 
  constexpr at::ScalarType dtype = at::ScalarType::Float;
  constexpr int64_t correction = 0;
  constexpr int64_t reduction_axis = 1;
  constexpr bool keepdim = true;

  std::unique_ptr<Fusion> fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());

  TensorView* x = makeContigTensor(2, aten_to_data_type(dtype));
  fusion->addInput(x);

  Val* num_elem = x->getLeafDomain().at(reduction_axis)->extent();

  TensorView* sum_x = sum(x, {reduction_axis}, /*keepdim=*/false);
  TensorView* mean_x = div(sum_x, num_elem);
  TensorView* bcast_mean = broadcast(mean_x, {false, true});

  TensorView* x_mean_sub = sub(x, bcast_mean);
  TensorView* x_mean_sub_sq = mul(x_mean_sub, x_mean_sub);
  TensorView* sum_x_mean_sub_sq =
      sum(x_mean_sub_sq, {reduction_axis}, /*keepdim=*/false);
  TensorView* var_x = div(sum_x_mean_sub_sq, num_elem);
  TensorView* bcast_var = broadcast(var_x, {false, true});

  TensorView* x_norm = div(sub(x, bcast_mean), sqrt(bcast_var));
  fusion->addOutput(x_norm);

  // Create cache_tvs
  TensorView* x_cache_smem =
      x->cacheAfter(LoadStoreOpType::CpAsyncBulkTensorTile);
  x_cache_smem->setMemoryType(MemoryType::Shared);

  x_cache_smem->cacheAfter();
  x_norm->cacheBefore();

  std::vector<TensorView*> reduction_tvs =
      scheduler_utils::getReductionTvs(fusion.get());

  TensorView* reference_tv = x_norm;

  // boxDim array must be non-zero and less than or equal to 256
  constexpr int64_t width = 256;
  constexpr int64_t vectorize = 4;
  constexpr int64_t elem_per_compute_thread = dim1 / width / vectorize;
  constexpr int64_t examples_per_cta = 4;

  // Define TMA Box
  // split: [I0, I2, 256]
  // load entire example in shared memory
  x_cache_smem->split(0, examples_per_cta);
  x_cache_smem->split(-1, 256);

  // Schedule reference_tv
  //   root domain: [I1, I2]
  //         split: [I1, I2/V (width / tdx), V]
  reference_tv->split(-1, vectorize);
  //         split: [I1, EPCT, I2/V/EPCT (tdx), V]
  reference_tv->split(-2, elem_per_compute_thread, /*inner_split=*/false);
  //         split: [I1, EPCT, I2/V/EPCT (tdx), U, V]
  reference_tv->split(-2, 1);
  //         reorder: [I1, I2/V/EPCT (tdx), EPCT, U, V]
  reference_tv->reorder({{-4, -3}, {-3, -4}});
  //         reorder: [I1/EPC, EPC, I2/V/EPCT (tdx), EPCT, U, V]
  reference_tv->split(0, examples_per_cta);

  TransformPropagator propagator(reference_tv);
  std::vector<TensorView*> all_tvs_except_cache =
      ir_utils::allTvsExcept(fusion.get(), {x_cache_smem});
  SetSelector selector(
      {all_tvs_except_cache.begin(), all_tvs_except_cache.end()});
  MaxRootDomainInfoSpanningTree(reference_tv, &selector).traverse(&propagator);

  std::vector<TensorView*> rfactor_tvs;
  rfactor_tvs.reserve(reduction_tvs.size());
  std::transform(
      reduction_tvs.begin(),
      reduction_tvs.end(),
      std::back_inserter(rfactor_tvs),
      [](TensorView* tv) {
        return tv->rFactor({-3, -2, -1});
      });

  // Define Parallelization Schema
  reference_tv->axis(0)->parallelize(ParallelType::BIDx);
  reference_tv->axis(2)->parallelize(ParallelType::TIDx);
  reference_tv->axis(-2)->parallelize(ParallelType::Unroll);
  scheduler_utils::parallelizeAllLike(reference_tv);

  // Vectorize Cache
  reference_tv->axis(-1)->parallelize(ParallelType::Vectorize);

  // InlineMost automatically handles vectorize and tma dimensions
  inlineMost();

  // Handle TMA Tensor
  // Apply circular buffer after computeAt
  x_cache_smem->axis(-1)->parallelize(ParallelType::Bulk);
  if (examples_per_cta > 1) {
    x_cache_smem->circularBuffer(/*stages=*/2);
  }

  auto options = at::TensorOptions().dtype(dtype).device(at::kCUDA, 0);
  at::Tensor at_tv0 = at::randn({dim0, dim1}, options);
  at::Tensor at_tv1 = at::randn({dim0, dim1}, options);

  // Compile with FusionExecutor directly to avoid scheduling
  FusionExecutor fe;
  CompileParams index32bit{DataType::Int32, 255, false};
  fe.compileFusion(fusion.get(), {at_tv0}, {}, index32bit);
  std::vector<at::Tensor> cg_outputs = fe.runFusion({at_tv0});

  std::tuple<at::Tensor, at::Tensor> at_var_mean =
     at::var_mean(at_tv0, {-1}, correction, keepdim);
  at::Tensor at_var = std::get<0>(at_var_mean);
  at::Tensor at_mean = std::get<1>(at_var_mean);
  at::Tensor at_output = (at_tv0 - at_mean) / sqrt(at_var);

  testValidate(
      fusion.get(), cg_outputs, {at_tv0}, {at_output}, __LINE__, __FILE__);
}

TEST_F(DoubleBufferingTest, TmaDoubleBufferingMatmul) {
  NVFUSER_TEST_CUDA_ARCH_GUARD(9, 0);

  std::unique_ptr<Fusion> fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());

  // Algorithm
  TensorView* tv0 = makeContigTensor(2); // (M, K)
  TensorView* tv1 = makeContigTensor(2); // (K, N)
  fusion->addInput(tv0);
  fusion->addInput(tv1);

  TensorView* tv2 = broadcast(tv0, {false, false, true}); // (M, K, B)
  TensorView* tv3 = broadcast(tv1, {true, false, false}); // (B, K, N)
  TensorView* tv4 = mul(tv2, tv3); // M, K, N
  TensorView* tv5 = sum(tv4, {1}); // M, R, N
  fusion->addOutput(tv5);

  TensorView* tv6 = tv5->cacheBefore();

  // For register double buffering
  TensorView* tv0_cache_local = tv0->cacheAfter();
  TensorView* tv1_cache_local = tv1->cacheAfter();

  // For smem double buffering
  TensorView* tv0_cache_smem =
      tv0->cacheAfter(LoadStoreOpType::CpAsyncBulkTensorTile);
  TensorView* tv1_cache_smem =
      tv1->cacheAfter(LoadStoreOpType::CpAsyncBulkTensorTile);
  tv0_cache_smem->setMemoryType(MemoryType::Shared);
  tv1_cache_smem->setMemoryType(MemoryType::Shared);

  constexpr int64_t BSX = 32;
  constexpr int64_t TSX = 8;

  // [M, K, N]
  tv6->split(-1, BSX);
  tv6->split(-1, TSX);
  tv6->split(1, BSX);
  tv6->split(0, BSX);
  tv6->split(1, TSX);
  // [M/BSX, BSX/TSX, TSX, K/BSX, BSX, N/BSX, BSX/TSX, TSX]
  tv6->reorder(
      {{4, 7}, {7, 6}, {6, 5}, {2, 4}, {1, 3}, {3, 2}, {5, 1}, {0, 0}});
  // [M/BSX, N/BSX, K/BSX, BSX/TSX, BSX/TSX, TSX, TSX, BSX]

  TensorView* tv6_rf = tv6->rFactor({-1});

  TransformPropagatorWithCheck propagator(tv6_rf);
  MaxRootDomainInfoSpanningTree(tv6_rf).traverse(&propagator);

  tv0->computeAt(tv6, 3);
  tv1->computeAt(tv6, 3);

  tv6_rf->computeAt(tv6, -1);
  tv0_cache_local->computeAt(tv6_rf, -1);
  tv1_cache_local->computeAt(tv6_rf, -1);

  tv5->axis(0)->parallelize(ParallelType::BIDx);
  tv5->axis(1)->parallelize(ParallelType::BIDy);
  tv5->axis(-3)->parallelize(ParallelType::TIDy);
  tv5->axis(-1)->parallelize(ParallelType::TIDx);

  scheduler_utils::parallelizeAllLike(tv5);

  // Apply circular buffering to smem and local cache tensors
  tv0_cache_smem->axis(-3)->parallelize(ParallelType::Bulk);
  tv0_cache_smem->axis(-2)->parallelize(ParallelType::Bulk);
  tv0_cache_smem->axis(-1)->parallelize(ParallelType::Bulk);

  tv1_cache_smem->axis(-3)->parallelize(ParallelType::Bulk);
  tv1_cache_smem->axis(-2)->parallelize(ParallelType::Bulk);
  tv1_cache_smem->axis(-1)->parallelize(ParallelType::Bulk);

  tv0_cache_local->circularBuffer(2);
  tv1_cache_local->circularBuffer(2);

  tv0_cache_smem->circularBuffer(3);
  tv1_cache_smem->circularBuffer(3);

  constexpr int64_t K = 1024;
  std::vector<int64_t> M_tensor_sizes = {10, 32, 50, 128};
  // NOTE: Multiple of 16 required for inner dimension
  std::vector<int64_t> N_tensor_sizes = {16, 32, 128};
  for (int64_t M : M_tensor_sizes) {
    for (int64_t N : N_tensor_sizes) {
      auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
      at::Tensor t0 = at::randn({M, K}, options);
      at::Tensor t1 = at::randn({K, N}, options);
      std::vector<c10::IValue> aten_inputs = {t0, t1};
      at::Tensor aten_output = at::matmul(t0.to(at::kDouble), t1.to(at::kDouble));

      FusionExecutor fe;
      CompileParams index32bit{DataType::Int32, 255, false};
      fe.compileFusion(fusion.get(), aten_inputs, {}, index32bit);

      std::vector<at::Tensor> cg_outputs = fe.runFusion(aten_inputs);
      testValidate(
          fusion.get(), cg_outputs, aten_inputs, {aten_output}, __LINE__, __FILE__);
    }
  }
}

TEST_F(DoubleBufferingTest, FusionDoubleBuffering1) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto tv0 = makeContigTensor(1);
  fusion.addInput(tv0);

  auto tv1 = set(tv0);
  auto tv2 = add(tv1, IrBuilder::create<Val>(1.0));
  auto tv3 = set(tv2);
  fusion.addOutput(tv3);

  tv1->setMemoryType(MemoryType::Shared);

  tv3->split(-1, 128);
  tv3->split(-1, 32);
  TransformPropagatorWithCheck propagator(tv3);
  MaxRootDomainInfoSpanningTree(tv3).traverse(&propagator);

  tv0->computeAt(tv3, 1);

  tv3->axis(-2)->parallelize(ParallelType::BIDx);
  tv3->axis(-1)->parallelize(ParallelType::TIDx);
  scheduler_utils::parallelizeAllLike(tv3);

  tv1->doubleBuffer();

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  auto t0 = at::randn({1000}, options);

  FusionExecutor fe;
  fe.compileFusion(&fusion, {t0});
  auto cg_outputs = fe.runFusion({t0});

  auto ref = t0 + 1;

  testValidate(&fusion, cg_outputs, {t0}, {ref}, __LINE__, __FILE__);
}

TEST_F(DoubleBufferingTest, DoubleBuffering2) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto tv0 = makeContigTensor(1);
  fusion.addInput(tv0);

  auto tv1 = set(tv0);
  auto tv2 = add(tv1, IrBuilder::create<Val>(1.0));
  auto tv3 = set(tv2);
  fusion.addOutput(tv3);

  tv3->split(-1, 128);
  tv3->split(-1, 32);
  TransformPropagatorWithCheck propagator(tv3);
  MaxRootDomainInfoSpanningTree(tv3).traverse(&propagator);

  tv0->computeAt(tv3, -1);

  tv3->axis(-2)->parallelize(ParallelType::BIDx);
  tv3->axis(-1)->parallelize(ParallelType::TIDx);
  scheduler_utils::parallelizeAllLike(tv3);

  tv1->doubleBuffer();

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  auto t0 = at::randn({1000}, options);

  FusionExecutor fe;
  fe.compileFusion(&fusion, {t0});
  auto cg_outputs = fe.runFusion({t0});

  auto ref = t0 + 1;

  testValidate(&fusion, cg_outputs, {t0}, {ref}, __LINE__, __FILE__);
}

TEST_F(DoubleBufferingTest, DoubleBuffering3) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto tv0 = makeContigTensor(1);
  fusion.addInput(tv0);

  auto tv1 = add(tv0, IrBuilder::create<Val>(1.0));
  auto tv2 = set(tv1);
  auto tv3 = add(tv2, IrBuilder::create<Val>(1.0));
  fusion.addOutput(tv3);

  tv1->setMemoryType(MemoryType::Shared);

  tv3->split(-1, 128);
  tv3->split(-1, 32);
  TransformPropagatorWithCheck propagator(tv3);
  MaxRootDomainInfoSpanningTree(tv3).traverse(&propagator);

  tv0->computeAt(tv3, 1);

  // tv2 is invalid to double-buffer as its producer, tv1, is
  // computed inside the double-buffering loop.
  // NOLINTNEXTLINE(cppcoreguidelines-avoid-goto,hicpp-avoid-goto)
  ASSERT_ANY_THROW(tv2->doubleBuffer());

  // Moving tv2 inner makes tv1 large enough to double-buffer tv2
  tv2->computeAt(tv3, 2);

  tv2->doubleBuffer();

  tv3->axis(-1)->parallelize(ParallelType::TIDx);
  scheduler_utils::parallelizeAllLike(tv3);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  auto t0 = at::randn({1000}, options);

  FusionExecutor fe;
  fe.compileFusion(&fusion, {t0});
  auto cg_outputs = fe.runFusion({t0});

  auto ref = t0 + 2;

  testValidate(&fusion, cg_outputs, {t0}, {ref}, __LINE__, __FILE__);
}

// Double buffering smem to local and unswitch
TEST_F(DoubleBufferingTest, DoubleBuffering4) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto tv0 = makeContigTensor(1);
  fusion.addInput(tv0);

  auto tv1 = add(tv0, IrBuilder::create<Val>(1.0));
  auto tv2 = set(tv1);
  auto tv3 = add(tv2, IrBuilder::create<Val>(1.0));
  fusion.addOutput(tv3);

  tv1->setMemoryType(MemoryType::Shared);

  tv3->split(-1, 128);
  tv3->split(-1, 32);
  tv3->split(-1, 8);
  TransformPropagatorWithCheck propagator(tv3);
  MaxRootDomainInfoSpanningTree(tv3).traverse(&propagator);

  tv0->computeAt(tv3, 2);
  tv2->computeAt(tv3, -1);

  tv3->axis(-1)->parallelize(ParallelType::TIDx);
  tv3->axis(1)->parallelize(ParallelType::Unswitch);
  scheduler_utils::parallelizeAllLike(tv3);

  tv2->doubleBuffer();

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  auto t0 = at::randn({1000}, options);

  FusionExecutor fe;
  fe.compileFusion(&fusion, {t0});
  auto cg_outputs = fe.runFusion({t0});

  auto ref = t0 + 2;

  testValidate(&fusion, cg_outputs, {t0}, {ref}, __LINE__, __FILE__);
}

// Double buffering gmem to shared and unswitch
TEST_F(DoubleBufferingTest, DoubleBuffering5) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto tv0 = makeContigTensor(1);
  fusion.addInput(tv0);

  auto tv1 = set(tv0);
  auto tv2 = add(tv1, IrBuilder::create<Val>(1.0));
  fusion.addOutput(tv2);

  tv1->setMemoryType(MemoryType::Shared);

  tv2->split(-1, 128);
  tv2->split(-1, 32);
  tv2->split(-1, 8);
  TransformPropagatorWithCheck propagator(tv2);
  MaxRootDomainInfoSpanningTree(tv2).traverse(&propagator);

  tv0->computeAt(tv2, 2);
  tv1->computeAt(tv2, -1);

  tv2->axis(-1)->parallelize(ParallelType::TIDx);
  tv2->axis(1)->parallelize(ParallelType::Unswitch);
  scheduler_utils::parallelizeAllLike(tv2);

  tv1->doubleBuffer();

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  auto t0 = at::randn({1000}, options);

  FusionExecutor fe;
  fe.compileFusion(&fusion, {t0});
  auto cg_outputs = fe.runFusion({t0});

  auto ref = t0 + 1;

  testValidate(&fusion, cg_outputs, {t0}, {ref}, __LINE__, __FILE__);
}

// Double buffering smem to local and unroll
TEST_F(DoubleBufferingTest, DoubleBuffering6) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto tv0 = makeContigTensor(1);
  fusion.addInput(tv0);

  auto tv1 = add(tv0, IrBuilder::create<Val>(1.0));
  auto tv2 = set(tv1);
  auto tv3 = add(tv2, IrBuilder::create<Val>(1.0));
  fusion.addOutput(tv3);

  tv1->setMemoryType(MemoryType::Shared);

  tv3->split(-1, 128);
  tv3->split(-1, 16);
  tv3->split(-2, 4);
  tv3->split(-2, 2);
  TransformPropagatorWithCheck propagator(tv3);
  MaxRootDomainInfoSpanningTree(tv3).traverse(&propagator);

  tv0->computeAt(tv3, 1);
  tv2->computeAt(tv3, -1);

  tv3->axis(2)->parallelize(ParallelType::Unroll);
  tv3->axis(4)->parallelize(ParallelType::TIDx);

  tv2->doubleBuffer();

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  auto t0 = at::randn({199}, options);

  FusionExecutor fe;
  fe.compileFusion(&fusion, {t0});
  auto cg_outputs = fe.runFusion({t0});

  auto ref = t0 + 2;

  testValidate(&fusion, cg_outputs, {t0}, {ref}, __LINE__, __FILE__);
}

// Double buffering and vectorize
TEST_F(DoubleBufferingTest, DoubleBuffering7) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto tv0 = makeContigTensor(1);
  fusion.addInput(tv0);

  auto tv1 = set(tv0);
  auto tv2 = add(tv1, IrBuilder::create<Val>(1.0));
  fusion.addOutput(tv2);

  tv2->split(-1, 128);
  tv2->split(-1, 4);
  TransformPropagatorWithCheck propagator(tv2);
  MaxRootDomainInfoSpanningTree(tv2).traverse(&propagator);

  tv1->computeAt(tv2, 2);

  tv2->axis(-2)->parallelize(ParallelType::TIDx);

  tv1->axis(-1)->parallelize(ParallelType::Vectorize);

  tv1->doubleBuffer();

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  auto t0 = at::randn({200}, options);

  FusionExecutor fe;
  fe.compileFusion(&fusion, {t0});
  auto cg_outputs = fe.runFusion({t0});

  auto ref = t0 + 1;

  testValidate(&fusion, cg_outputs, {t0}, {ref}, __LINE__, __FILE__);
}

// Multiple tensors to double-buffer
TEST_F(DoubleBufferingTest, DoubleBuffering8) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto tv0 = makeContigTensor(1);
  fusion.addInput(tv0);
  auto tv1 = makeContigTensor(1);
  fusion.addInput(tv1);

  auto tv2 = set(tv0);
  auto tv3 = set(tv1);
  auto tv4 = add(tv2, tv3);
  fusion.addOutput(tv4);

  tv4->split(0, 32);
  tv4->split(0, 4);
  TransformPropagatorWithCheck propagator(tv4);
  MaxRootDomainInfoSpanningTree(tv4).traverse(&propagator);

  tv0->computeAt(tv4, 1);
  tv1->computeAt(tv4, 1);

  tv4->axis(-1)->parallelize(ParallelType::TIDx);
  scheduler_utils::parallelizeAllLike(tv4);

  tv2->doubleBuffer();
  tv3->doubleBuffer();

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  auto t0 = at::randn({100}, options);
  auto t1 = at::randn({100}, options);

  FusionExecutor fe;
  fe.compileFusion(&fusion, {t0, t1});
  auto cg_outputs = fe.runFusion({t0, t1});

  auto ref = t0 + t1;

  testValidate(&fusion, cg_outputs, {t0, t1}, {ref}, __LINE__, __FILE__);
}

// Nested double buffering from gmem to smem and smem to register
TEST_F(DoubleBufferingTest, DoubleBuffering9) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto tv0 = makeContigTensor(1);
  fusion.addInput(tv0);
  auto tv1 = add(tv0, IrBuilder::create<Val>(1.0));
  auto out = tv1;
  fusion.addOutput(out);

  auto tv2 = tv0->cacheAfter();
  auto tv3 = tv2->cacheAfter();

  out->split(0, 32);
  out->split(0, 4);
  TransformPropagatorWithCheck propagator(out);
  MaxRootDomainInfoSpanningTree(out).traverse(&propagator);

  tv2->setMemoryType(MemoryType::Shared);

  tv2->computeAt(out, 1);
  tv3->computeAt(out, -1);

  out->axis(-1)->parallelize(ParallelType::TIDx);
  scheduler_utils::parallelizeAllLike(out);

  tv2->doubleBuffer();
  tv3->doubleBuffer();

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  auto t0 = at::randn({1001}, options);

  FusionExecutor fe;
  fe.compileFusion(&fusion, {t0});
  auto cg_outputs = fe.runFusion({t0});

  auto ref = t0 + 1;

  testValidate(&fusion, cg_outputs, {t0}, {ref}, __LINE__, __FILE__);
}

// FusionSmemBlockGemmCache + double buffering at both smem and local
TEST_F(DoubleBufferingTest, SmemBlockGemmCacheDoubleBuffer) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  // Algorithm
  TensorView* tv0 = makeSymbolicTensor(2); // (M, K)
  TensorView* tv1 = makeSymbolicTensor(2); // (K, N)
  TensorView* tv2 = broadcast(tv0, {false, false, true}); // (M, K, B)
  TensorView* tv3 = broadcast(tv1, {true, false, false}); // (B, K, N)
  TensorView* tv4 = mul(tv2, tv3); // M, K, N
  TensorView* tv5 = sum(tv4, {1}); // M, R, N
  fusion.addInput(tv0);
  fusion.addInput(tv1);
  fusion.addOutput(tv5);

  TensorView* tv6 = tv5->cacheBefore();

  // For smem double buffering
  auto tv0_cache_local = tv0->cacheAfter();
  auto tv1_cache_local = tv1->cacheAfter();

  // For register double buffering
  auto tv0_cache_smem = tv0->cacheAfter();
  auto tv1_cache_smem = tv1->cacheAfter();

  const int BSX = 32;
  const int TSX = 8;

  // [M, K, N]
  tv6->split(-1, BSX);
  tv6->split(-1, TSX);
  tv6->split(1, BSX);
  tv6->split(0, BSX);
  tv6->split(1, TSX);
  // [M/BSX, BSX/TSX, TSX, K/BSX, BSX, N/BSX, BSX/TSX, TSX]
  tv6->reorder(
      {{4, 7}, {7, 6}, {6, 5}, {2, 4}, {1, 3}, {3, 2}, {5, 1}, {0, 0}});
  // [M/BSX, N/BSX, K/BSX, BSX/TSX, BSX/TSX, TSX, TSX, BSX]

  auto tv6_rf = tv6->rFactor({-1});

  TransformPropagatorWithCheck propagator(tv6_rf);
  MaxRootDomainInfoSpanningTree(tv6_rf).traverse(&propagator);

  tv0->computeAt(tv6, 3);
  tv1->computeAt(tv6, 3);

  tv6_rf->computeAt(tv6, -1);
  tv0_cache_local->computeAt(tv6_rf, -1);
  tv1_cache_local->computeAt(tv6_rf, -1);

  tv0_cache_smem->setMemoryType(MemoryType::Shared);
  tv1_cache_smem->setMemoryType(MemoryType::Shared);

  tv5->axis(0)->parallelize(ParallelType::BIDx);
  tv5->axis(1)->parallelize(ParallelType::BIDy);
  tv5->axis(-3)->parallelize(ParallelType::TIDy);
  tv5->axis(-1)->parallelize(ParallelType::TIDx);

  scheduler_utils::parallelizeAllLike(tv5);

  tv0_cache_local->doubleBuffer();
  tv1_cache_local->doubleBuffer();

  tv0_cache_smem->doubleBuffer();
  tv1_cache_smem->doubleBuffer();

  constexpr int M = 154, K = 45, N = 1524;

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor t0 = at::randn({M, K}, options);
  at::Tensor t1 = at::randn({K, N}, options);
  at::Tensor aten_output = at::matmul(t0.to(at::kDouble), t1.to(at::kDouble));

  std::vector<c10::IValue> aten_inputs = {t0, t1};

  FusionExecutor fe;
  fe.compileFusion(&fusion, aten_inputs);
  auto cg_outputs = fe.runFusion(aten_inputs);

  testValidate(
      &fusion, cg_outputs, aten_inputs, {aten_output}, __LINE__, __FILE__);
  // The smem cache write in this test case is redundant predicated,
  //   and also double buffered. Currently we are relying on WAR sync
  //   insertion to ensure ordering of double buffered tensor access.
  // The check below makes sure that the sync is inserted so that the
  //   test isn't running on a race condition.
  NVF_CHECK(fe.kernel()->summary().war_hazard_syncs_count > 0);
}

// Vectorized reset test for double buffered registers
TEST_F(DoubleBufferingTest, DoubleBufferVector) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto tv0 = makeContigTensor(1);
  fusion.addInput(tv0);

  auto tv1 = add(tv0, IrBuilder::create<Val>(1.0));
  auto tv2 = sum(tv1, {0});
  auto tv2c = tv2->cacheBefore();

  fusion.addOutput(tv2);

  auto tv1cw = tv1->cacheAfter();
  auto tv1cr = tv1cw->cacheAfter();

  tv1cw->split(-1, 32);
  tv1cr->split(-1, 32);
  tv1cr->split(-1, 4);
  tv1cr->axis(-1)->parallelize(ParallelType::Vectorize);

  tv1cw->computeAt(tv1cr, 1);
  tv0->computeAt(tv1cw, -1);
  tv2c->split(-1, 32);
  tv2c->split(-1, 4);
  tv1cr->computeAt(tv2c, 2);

  tv1cw->setMemoryType(MemoryType::Shared);
  tv1cr->doubleBuffer();

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);

  auto t0 = at::randn({200}, options);
  FusionExecutor fe;
  fe.compileFusion(&fusion, {t0});
  auto cg_outputs = fe.runFusion({t0});
  auto ref = (t0 + 1).sum({0});

  testValidate(&fusion, cg_outputs, {t0}, {ref}, __LINE__, __FILE__);
}

// Simple test of async copy primitive: double buffered
//   Double buffer case 1, both block sync and async wait
//  are needed.
TEST_F(DoubleBufferingTest, DoubleBufferCpAsync1) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  // Using vectorization so need to keep n multiple of 4.
  int m = 33, n = 48;

  TensorView* tv0 = makeContigConcreteTensor({m, n});
  TensorView* tv1 = makeContigConcreteTensor({m, n});

  fusion.addInput(tv0);
  fusion.addInput(tv1);

  TensorView* tv2 = add(tv0, tv1);

  fusion.addOutput(tv2);

  auto tv0_shared = tv0->cacheAfter(LoadStoreOpType::CpAsync);
  tv0_shared->setMemoryType(MemoryType::Shared);
  tv0->computeAt(tv2, 1);

  // Asynchronously load a tile in one schedule
  tv0_shared->split(1, 4);
  tv0_shared->axis(-1)->parallelize(ParallelType::Vectorize);
  tv0_shared->axis(-2)->parallelize(ParallelType::TIDx);

  // Consume the loaded tile in another schedule,
  //   triggering the need for a sync.
  tv2->split(1, 12);
  tv2->axis(-1)->parallelize(ParallelType::TIDx);

  // Double buffer the shared mem tensor.
  tv0_shared->doubleBuffer();

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor t0 = at::randn({m, n}, options);
  at::Tensor t1 = at::randn({m, n}, options);

  FusionExecutor fe;
  // requires ampere+ GPU
  if (!deviceMajorMinorCheck(8)) {
    ASSERT_ANY_THROW(fe.compileFusion(&fusion, {t0, t1}));
    GTEST_SKIP() << "skipping tests on pre-AMPERE GPUs";
  }
  fe.compileFusion(&fusion, {t0, t1});
  auto cg_outputs = fe.runFusion({t0, t1});

  auto ref = t0 + t1;

  testValidate(&fusion, cg_outputs, {t0, t1}, {ref}, __LINE__, __FILE__);
}

// Simple test of async copy primitive: double buffered
//   Double buffer case 2, only async wait is needed
TEST_F(DoubleBufferingTest, DoubleBufferCpAsync2) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  // Using vectorization so need to keep n multiple of 4.
  int m = 33, n = 48;

  TensorView* tv0 = makeConcreteTensor({m, n});
  TensorView* tv1 = makeConcreteTensor({m, n});

  fusion.addInput(tv0);
  fusion.addInput(tv1);

  TensorView* tv2 = add(tv0, tv1);

  fusion.addOutput(tv2);

  auto tv0_shared = tv0->cacheAfter(LoadStoreOpType::CpAsync);
  tv0_shared->setMemoryType(MemoryType::Shared);
  tv0->computeAt(tv2, 1);

  // Asynchronously load a tile in one schedule
  tv0_shared->split(1, 4);
  tv0_shared->axis(-2)->parallelize(ParallelType::TIDx);

  // Consume the loaded tile in another schedule,
  //   triggering the need for a sync.
  tv2->split(1, 4);
  tv2->axis(-2)->parallelize(ParallelType::TIDx);

  // Double buffer the shared mem tensor.
  tv0_shared->doubleBuffer();

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor t0 = at::randn({m, n}, options);
  at::Tensor t1 = at::randn({m, n}, options);

  FusionExecutor fe;
  // requires ampere+ GPU
  if (!deviceMajorMinorCheck(8)) {
    ASSERT_ANY_THROW(fe.compileFusion(&fusion, {t0, t1}));
    GTEST_SKIP() << "skipping tests on pre-AMPERE GPUs";
  }
  fe.compileFusion(&fusion, {t0, t1});
  auto cg_outputs = fe.runFusion({t0, t1});

  auto ref = t0 + t1;

  testValidate(&fusion, cg_outputs, {t0, t1}, {ref}, __LINE__, __FILE__);
}

// Simple test for double buffer in shared mem,
//  where we should not insert redundant syncs when
//  they are not needed.
TEST_F(DoubleBufferingTest, DoubleBufferNoSync) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  // Using vectorization so need to keep n multiple of 4.
  int m = 33, n = 48;

  TensorView* tv0 = makeConcreteTensor({m, n});
  TensorView* tv1 = makeConcreteTensor({m, n});

  fusion.addInput(tv0);
  fusion.addInput(tv1);

  TensorView* tv2 = add(tv0, tv1);

  fusion.addOutput(tv2);

  auto tv0_shared = tv0->cacheAfter();
  tv0_shared->setMemoryType(MemoryType::Shared);
  tv0->computeAt(tv2, 1);

  // Asynchronously load a tile in one schedule
  tv0_shared->split(1, 4);
  tv0_shared->axis(-2)->parallelize(ParallelType::TIDx);

  // Consume the loaded tile in another schedule,
  //   triggering the need for a sync.
  tv2->split(1, 4);
  tv2->axis(-2)->parallelize(ParallelType::TIDx);

  // Double buffer the shared mem tensor.
  tv0_shared->doubleBuffer();

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor t0 = at::randn({m, n}, options);
  at::Tensor t1 = at::randn({m, n}, options);

  GpuLower gpulw(&fusion);
  auto flattened_exprs =
      ir_utils::flattenScopedExprs(gpulw.run()->topLevelExprs());
  bool sync_inserted = std::any_of(
      flattened_exprs.begin(), flattened_exprs.end(), [](Expr* expr) {
        return expr->isA<kir::BlockSync>();
      });
  NVF_ERROR(!sync_inserted, "Un-expected block sync inserted");

  FusionExecutor fe;
  fe.compileFusion(&fusion, {t0, t1});
  auto cg_outputs = fe.runFusion({t0, t1});

  auto ref = t0 + t1;

  testValidate(&fusion, cg_outputs, {t0, t1}, {ref}, __LINE__, __FILE__);
}
} // namespace nvfuser
