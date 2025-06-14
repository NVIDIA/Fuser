// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#include <csrc/exceptions.h>
#include <gmock/gmock-matchers.h>
#include <gtest/gtest.h>

#include <grouped_reduction.h>
#include <ir/utils.h>
#include <ops/all_ops.h>
#include <runtime/fusion_executor_cache.h>
#include <scheduler/all_schedulers.h>
#include <scheduler/reduction_utils.h>
#include <scheduler/tools/inlining.h>
#include <scheduler/utils.h>
#include <tests/cpp/utils.h>
#include <tests/cpp/validator.h>
#include <utils.h>

#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/Exceptions.h>
#include <c10/cuda/CUDAStream.h>

#include <algorithm>
#include <iostream>
#include <sstream>
#include <thread>

namespace nvfuser {

using OuterReductionTest = NVFuserTest;

using namespace at::indexing;

// Shmoo testing of the optimized grouped grid welford
TEST_F(OuterReductionTest, GroupedGridWelfordOuterOpt) {
  struct OuterReductionParams {
    int vec = 1;
    int tidx = 1;
    int tidy = 1;
    int bidx = 1;
    int pb = 8;
    bool should_use_opt = false;
    DataType dtype = DataType::Half;
  };

  auto run_test = [&](const OuterReductionParams& params) {
    std::unique_ptr<Fusion> fusion_ptr = std::make_unique<Fusion>();
    Fusion& fusion = *fusion_ptr.get();
    FusionGuard fg(&fusion);

    auto tv0 = makeContigTensor(2, params.dtype);
    fusion.addInput(tv0);

    auto tv1 = set(tv0);
    auto tv2 =
        params.dtype == DataType::Half ? castOp(DataType::Float, tv1) : tv1;
    auto tvs = Welford(tv2, {0});
    auto tv3 = tvs.avg;
    auto tv4 = broadcast(tv3, {true, false});
    auto tv5 =
        params.dtype == DataType::Half ? castOp(DataType::Float, tv1) : tv1;
    auto tv6 = sub(tv5, tv4);
    fusion.addOutput(tv6);

    int64_t bidy = deviceSMCount() / params.bidx;

    // Skip if the available SM count is too small for this problem size
    if (deviceSMCount() <= params.bidx || bidy <= 1) {
      return;
    }

    int64_t reduction_size = params.tidy * bidy * params.pb;
    int64_t iteration_size = params.vec * params.tidx * params.bidx * 8;

    auto ref = tv3;

    ref->reorder({{0, 1}});

    ref->split(1, params.tidy);
    ref->split(1, params.pb);

    ref->split(0, params.vec);
    ref->split(0, params.tidx);
    ref->split(0, params.bidx);

    // Move the vectorized ID to the innermost position
    ref->reorder({{3, -1}});

    auto ref_rf = ref->rFactor({-3}, {tvs.avg, tvs.var_sum, tvs.n}).at(0);

    TransformPropagator propagator(ref_rf);
    MaxLogicalDomainInfoSpanningTree(ref_rf).traverse(&propagator);

    ref_rf->axis(1)->parallelize(ParallelType::BIDx);
    ref_rf->axis(2)->parallelize(ParallelType::TIDx);
    ref_rf->axis(3)->parallelize(ParallelType::BIDy);
    ref_rf->axis(5)->parallelize(ParallelType::TIDy);

    scheduler_utils::parallelizeAllLike(ref_rf, fusion.allTvs());

    tv1->axis(-1)->parallelize(ParallelType::Vectorize);
    tv3->axis(-1)->parallelize(ParallelType::Group);

    inlineMost();

    auto at_dtype = params.dtype == DataType::Half ? at::kHalf : at::kFloat;
    auto options = at::TensorOptions().dtype(at_dtype).device(at::kCUDA, 0);

    const std::vector<int64_t> input_shape{reduction_size, iteration_size};
    auto t0 = at::randn(input_shape, options);
    KernelArgumentHolder inputs({t0});

    KernelExecutor ke;
    ke.compile(&fusion, inputs);

    NVF_CHECK(
        ke.compiledKernel()
                ->kernel()
                ->summary()
                .has_outer_grouped_grid_welford == params.should_use_opt,
        (params.should_use_opt ? "Failed to use the optimized implementation"
                               : "Should not use the optimized implementation"),
        ": ",
        params.vec,
        ", ",
        params.tidx,
        ", ",
        params.tidy,
        ", ",
        params.bidx);

    auto cg_outputs = ke.run(inputs);

    auto t1 = t0;
    auto t2 = params.dtype == DataType::Half ? t1.to(at::kFloat) : t1;
    auto t3 = t2.mean({0});
    auto t4 = t3.unsqueeze(0);
    auto t5 = params.dtype == DataType::Half ? t1.to(at::kFloat) : t1;
    auto t6 = t5 - t4;

    testValidate(&fusion, cg_outputs, inputs, {t6}, __LINE__, __FILE__, "");
  };

  std::vector<OuterReductionParams> test_params;

  for (const auto& dtype : {DataType::Half, DataType::Float}) {
    for (int bidx = 1; bidx < 8; bidx *= 2) {
      if (dtype == DataType::Half) {
        test_params.push_back(
            {.vec = 8,
             .tidx = 16,
             .tidy = 16,
             .bidx = bidx,
             .should_use_opt = true,
             .dtype = dtype});
        test_params.push_back(
            {.vec = 8,
             .tidx = 8,
             .tidy = 32,
             .bidx = bidx,
             .should_use_opt = true,
             .dtype = dtype});
        test_params.push_back(
            {.vec = 8,
             .tidx = 4,
             .tidy = 64,
             .bidx = bidx,
             .should_use_opt = true,
             .dtype = dtype});
      }
      test_params.push_back(
          {.vec = 4,
           .tidx = 16,
           .tidy = 16,
           .bidx = bidx,
           .should_use_opt = true,
           .dtype = dtype});
      test_params.push_back(
          {.vec = 4,
           .tidx = 8,
           .tidy = 32,
           .bidx = bidx,
           .should_use_opt = true,
           .dtype = dtype});
      // warp_size/tidx too large
      test_params.push_back(
          {.vec = 4,
           .tidx = 4,
           .tidy = 64,
           .bidx = bidx,
           .should_use_opt = false,
           .dtype = dtype});
      test_params.push_back(
          {.vec = 2,
           .tidx = 16,
           .tidy = 16,
           .bidx = bidx,
           .should_use_opt = true,
           .dtype = dtype});
      // warp_size/tidx too large
      test_params.push_back(
          {.vec = 2,
           .tidx = 8,
           .tidy = 32,
           .bidx = bidx,
           .should_use_opt = false,
           .dtype = dtype});
      // warp_size/tidx too large
      test_params.push_back(
          {.vec = 2,
           .tidx = 4,
           .tidy = 64,
           .bidx = bidx,
           .should_use_opt = false,
           .dtype = dtype});
    }
  }

  for (const auto& params : test_params) {
    // Shmoo tests can occupy a lot of memory due to allocating many
    // different tensor sizes. So in order to avoid an OOM during this
    // test, we manually clear the allocator after it's reached a certain
    // threshold.
    maybeClearAllocator();
    run_test(params);
  }
}

namespace {

// A quick opt-in switch to enable performance measurements.
bool isBenchmarkMode() {
  return getNvFuserEnv("OUTER_REDUCTION_BENCHMARK") != nullptr;
}

struct OuterReductionParams {
  // Vectorization factor
  int vec = 4;
  // blockDim.x
  int tidx = 16;
  // blockDim.y
  int tidy = 16;
  // gridDim.x
  int bidx = 2;
  // Size of persistent buffer. Actual allocation size is multiplied
  // by the vectorization factor
  int pb = 88;
  // Unswitched inner split out of the persistent buffer domain. This
  // should evenly divide pb
  int unswitch = 4;
  // Outer split the reduction domain for the persistent buffer if true
  bool outer_pb = false;
  // Use computeWith if true
  bool use_compute_with = true;

  void configureForForward(int64_t N, int64_t HW, int64_t C, DataType dtype) {
    // Tuned for Titan RTX.
    int vec = dtype == DataType::Half ? 8 : 4;
    if (N == 256 && HW == 7) {
      *this = {
          .vec = vec,
          .tidx = 16,
          .tidy = 16,
          .bidx = 4,
          .pb = 44,
          .unswitch = 4};
    } else if (N == 256 && HW == 14) {
      *this = {
          .vec = vec,
          .tidx = 8,
          .tidy = 32,
          .bidx = 2,
          .pb = 44,
          .unswitch = 4};
    } else if (N == 256 && HW == 28) {
      *this = {
          .vec = vec,
          .tidx = 4,
          .tidy = 64,
          .bidx = 1,
          .pb = 44,
          .unswitch = 4};
    } else if (N == 32 && HW == 32) {
      *this = {
          .vec = vec,
          .tidx = 8,
          .tidy = 32,
          .bidx = 2,
          .pb = 30,
          .unswitch = 10};
    }
  }

  void configureForBackward(int64_t N, int64_t HW, int64_t C, DataType dtype) {
    // Tuned for Titan RTX.
    if (dtype == DataType::Half) {
      if (N == 256 && HW == 7) {
        *this = {
            .vec = 4,
            .tidx = 16,
            .tidy = 16,
            .bidx = 4,
            .pb = 44,
            .unswitch = 4};
      } else if (N == 256 && HW == 14) {
        *this = {
            .vec = 4,
            .tidx = 16,
            .tidy = 16,
            .bidx = 1,
            .pb = 44,
            .unswitch = 4};
      } else if (N == 256 && HW == 28) {
        *this = {
            .vec = 4,
            .tidx = 4,
            .tidy = 64,
            .bidx = 1,
            .pb = 44,
            .unswitch = 4};
      }
    } else {
      if (N == 256 && HW == 7) {
        *this = {
            .vec = 4,
            .tidx = 8,
            .tidy = 32,
            .bidx = 4,
            .pb = 22,
            .unswitch = 2};
      } else if (N == 256 && HW == 14) {
        *this = {
            .vec = 4,
            .tidx = 8,
            .tidy = 32,
            .bidx = 1,
            .pb = 22,
            .unswitch = 2};
      } else if (N == 256 && HW == 28) {
        *this = {
            .vec = 2,
            .tidx = 4,
            .tidy = 64,
            .bidx = 1,
            .pb = 44,
            .unswitch = 4};
      }
    }
  }

  // Quick configuration with environment variables
  void configureWithEnv() {
    if (getenv("TIDX")) {
      tidx = atoi(getenv("TIDX"));
    }

    if (getenv("TIDY")) {
      tidy = atoi(getenv("TIDY"));
    }

    if (getenv("BIDX")) {
      bidx = atoi(getenv("BIDX"));
    }

    if (getenv("PB")) {
      pb = atoi(getenv("PB"));
    }

    if (getenv("VEC")) {
      vec = atoi(getenv("VEC"));
    }

    if (getenv("UNSWITCH")) {
      unswitch = atoi(getenv("UNSWITCH"));
    }

    if (getenv("OUTER_PB")) {
      outer_pb = true;
    }

    if (getenv("COMPUTE_WITH")) {
      use_compute_with = atoi(getenv("COMPUTE_WITH")) != 0;
    }
  }
};

void scheduleNormalization(Fusion& fusion, const OuterReductionParams& params) {
  // Cache inputs
  std::vector<TensorView*> input_caches;
  for (auto input_tv : ir_utils::filterByType<TensorView>(fusion.inputs())) {
    // Skip unused inputs
    if (input_tv->uses().empty()) {
      continue;
    }

    // Assumes any 1D tensor is just a Float or Half tensor that can be reused
    // without recomputation
    if (input_tv->nDims() == 1) {
      input_caches.push_back(input_tv->cacheAfter());
      continue;
    }

    // Any other tensor is assumed to be a 4D tensor that requires recomputation
    NVF_CHECK(
        input_tv->nDims() == 4,
        "Unexpected input tensor: ",
        input_tv->toString());
    // If the input type is Half, it must be cast to Float
    if (input_tv->getDataType() == DataType::Half) {
      NVF_CHECK(
          input_tv->uses().size() == 1,
          "Unexpected input tensor: ",
          input_tv->toString());
      auto cast_expr = dynamic_cast<UnaryOp*>(input_tv->uses().at(0));
      NVF_CHECK(
          cast_expr != nullptr &&
              cast_expr->getUnaryOpType() == UnaryOpType::Cast,
          "Unexpected input tensor: ",
          input_tv->toString());
      auto cast_tv = dynamic_cast<TensorView*>(cast_expr->out());
      NVF_CHECK(cast_tv != nullptr);
      auto cast_tv_use_exprs = cast_tv->uses();
      for (auto use : cast_tv_use_exprs) {
        auto replica = RecomputeTv::recompute(cast_tv);
        ir_utils::replaceValInExprInputs(use, cast_tv, replica);
      }
    }
    input_caches.push_back(input_tv->cacheAfter());
  }

  // Cache outputs
  std::vector<TensorView*> output_caches;
  for (auto output_tv : ir_utils::filterByType<TensorView>(fusion.outputs())) {
    output_caches.push_back(output_tv->cacheBefore());
  }

  // Find reduction TVs
  const auto used_math_vals = fusion.usedMathVals();
  std::vector<Expr*> reduction_exprs;
  std::vector<TensorView*> reduction_tvs;

  for (auto expr : fusion.exprs()) {
    if (ir_utils::isReductionTvOp(expr)) {
      reduction_exprs.push_back(expr);
      reduction_tvs.push_back(ir_utils::getTvOutput(expr));
    }
  }

  NVF_CHECK(!reduction_exprs.empty(), "No reduction found");

  // Apply horizontal grouping before rfactor
  if (reduction_exprs.size() > 1) {
    groupReductions(reduction_tvs);
  }

  NVF_CHECK(
      dataTypeSizeByte(DataType::Half) * params.vec * reduction_tvs.size() <=
          16,
      "Invalid vectorization");

  auto reduction_tv = reduction_tvs.at(0);

  // Transform a reduction TV as the reference
  reduction_tv->merge(0)->merge(0);

  // [r(I0), i(I1)]
  reduction_tv->split(0, params.tidy);
  // [r/(I0/TIDy), r(TIDy), i(I1)]
  reduction_tv->axis(1)->parallelize(ParallelType::TIDy);

  // When outer_pb is true, use the outer split for the persistent
  // buffer domain
  int unswitch_pos = 0;
  if (params.outer_pb) {
    reduction_tv->split(0, params.pb, false);
    // [r(PB), r/(I0/TIDy/PB), r(TIDy), i(I1)]
    reduction_tv->axis(1)->parallelize(ParallelType::BIDy);
    unswitch_pos = 0;
  } else {
    reduction_tv->split(0, params.pb);
    // [r/(I0/TIDy/PB), r(PB), r(TIDy), i(I1)]
    reduction_tv->axis(0)->parallelize(ParallelType::BIDy);
    unswitch_pos = 1;
  }

  // Inner split the unswitch factor if it's greater than 1. If the
  // factor is 1, unswitch the entire persistent buffer
  if (params.unswitch > 1) {
    reduction_tv->split(unswitch_pos, params.unswitch);
    reduction_tv->axis(unswitch_pos + 1)->parallelize(ParallelType::Unswitch);
  } else if (params.unswitch == 1) {
    reduction_tv->split(unswitch_pos, 1, false);
    reduction_tv->axis(unswitch_pos)->parallelize(ParallelType::Unswitch);
  }

  // Parallelize the iteration domain with vectorization, TIDx, and
  // BIDx

  // [i(I1)] (the reduction domains are omitted)
  reduction_tv->split(-1, params.vec);
  reduction_tv->axis(-1)->parallelize(ParallelType::Vectorize);
  // [i(I1/Vec), v(Vec)]
  reduction_tv->split(-2, params.tidx);
  reduction_tv->axis(-2)->parallelize(ParallelType::TIDx);
  // [i(I1/Vec/TIDx), i(TIDx), v(Vec)]
  reduction_tv->split(-3, params.bidx);
  reduction_tv->axis(-3)->parallelize(ParallelType::BIDx);
  // [i(I1/Vec/TIDx/BIDx), i(BIDX), i(TIDx), v(Vec)]

  auto reduction_tv_rf =
      reduction_scheduler_utils::sortAndRFactor(reduction_tv);

  // Make sure the vectorized domain placed at the innermost position
  int64_t vec_id_cur_pos = -1;
  std::unordered_map<int64_t, int64_t> vec_reorder_map;
  for (const auto i : arange(reduction_tv_rf->nDims())) {
    auto id = reduction_tv_rf->axis(i);
    if (id->getParallelType() == ParallelType::Vectorize) {
      vec_id_cur_pos = i;
      vec_reorder_map[i] = -1;
    } else if (vec_id_cur_pos >= 0) {
      vec_reorder_map[i] = i - 1;
    }
  }
  NVF_CHECK(vec_id_cur_pos != -1, "Vectorized ID not found");
  reduction_tv_rf->reorder(vec_reorder_map);

  TransformPropagator propagator(reduction_tv_rf);
  MaxLogicalDomainInfoSpanningTree(reduction_tv_rf).traverse(&propagator);

  // Clear vectorization and unswitch as we want to selectively use
  // them

  // Clear the vectorization
  reduction_tv->axis(-1)->parallelize(ParallelType::Serial);
  reduction_tv_rf->axis(-1)->parallelize(ParallelType::Serial);

  // Clear unswitch
  IterDomain* unswitch_id = nullptr;
  auto unswitch_id_it = std::find_if(
      reduction_tv_rf->getLoopDomain().begin(),
      reduction_tv_rf->getLoopDomain().end(),
      [](auto id) { return id->getParallelType() == ParallelType::Unswitch; });
  if (unswitch_id_it != reduction_tv_rf->getLoopDomain().end()) {
    unswitch_id = *unswitch_id_it;
  }

  if (unswitch_id != nullptr) {
    unswitch_id->parallelize(ParallelType::Serial);
  }

  scheduler_utils::parallelizeAllLike(reduction_tv_rf, fusion.allTvs());

  // Vectorize inputs
  for (auto input_cache : input_caches) {
    // Skip if the maximum supported size is exceeded
    if (input_cache->getDataType() == DataType::Float && params.vec == 8) {
      continue;
    }
    input_cache->axis(-1)->parallelize(ParallelType::Vectorize);
  }

  // Vectorize outputs
  for (auto output : ir_utils::filterByType<TensorView>(fusion.outputs())) {
    // Skip if the maximum supported size is exceeded
    if (output->getDataType() == DataType::Float && params.vec == 8) {
      continue;
    }
    output->axis(-1)->parallelize(ParallelType::Vectorize);
  }

  // Cross-iteration grouping of grid reductions
  for (auto reduction_tv : reduction_tvs) {
    reduction_tv->axis(-1)->parallelize(ParallelType::Group);
  }

  // Unswitch local reductions. Experimental results suggest using
  // unswitch for other tensors doesn't seem to improve the
  // performance or even negatively affect the performance due to
  // increased register pressure
  if (unswitch_id != nullptr) {
    unswitch_id->parallelize(ParallelType::Unswitch);
  }

  inlineMost();

  if (params.use_compute_with) {
    // Only apply computeWith to the main 4D tensors
    for (auto input_cache : input_caches) {
      if (input_cache->getLogicalDomain().size() == 4) {
        input_cache->computeWith(-1, true);
      }
    }
  }
}

// Test a fusion with outer grid reduction. Similar to outer
// batchnorm but simplified for testing, e.g., just a single reduction
// rather than welford
void grid_persistent_reduction_outer_norm_like(
    int64_t N,
    int64_t HW,
    int64_t C,
    DataType dtype) {
  const bool benchmark_mode = isBenchmarkMode();
  std::unique_ptr<Fusion> fusion_ptr = std::make_unique<Fusion>();
  Fusion& fusion = *fusion_ptr.get();
  FusionGuard fg(&fusion);

  std::vector<bool> bcast_pattern{true, true, true, false};
  std::vector<int64_t> reduction_dims{2, 1, 0};

  auto tv0 = makeContigTensor(4, dtype);
  fusion.addInput(tv0);
  auto tv1 = dtype == DataType::Half ? castOp(DataType::Float, tv0) : tv0;
  auto tv2 = sum(tv1, reduction_dims);
  auto tv3 = broadcast(tv2, bcast_pattern);
  auto tv4 = sub(tv1, tv3);
  auto tv5 = dtype == DataType::Half ? castOp(DataType::Half, tv4) : tv4;
  fusion.addOutput(tv5);

  OuterReductionParams params;

  params.configureForForward(N, HW, C, dtype);

  if (benchmark_mode) {
    params.configureWithEnv();
  }

  scheduleNormalization(fusion, params);

  auto options =
      at::TensorOptions().dtype(data_type_to_aten(dtype)).device(at::kCUDA, 0);

  const std::vector<int64_t> input_shape{N, HW, HW, C};
  auto t0 = at::randn(input_shape, options);

  KernelExecutor ke;
  ke.compile(&fusion, {t0});

  auto bidy = ceilDiv(ceilDiv(N * HW * HW, params.tidy), params.pb);

  if (params.bidx * bidy > deviceSMCount()) {
    GTEST_SKIP() << "Not enough SMs to run this test. Required: "
                 << params.bidx * bidy << ", available: " << deviceSMCount();
  }

  auto cg_outputs = ke.run({t0});

  if (benchmark_mode) {
    for (int i = 0; i < 10; ++i) {
      clearL2Cache();
      cg_outputs = ke.run({t0});
    }
  }

  auto t1 = t0.to(at::kFloat);
  auto t2 = t1.sum({0, 1, 2});
  auto t3 = t2.unsqueeze(0).unsqueeze(0).unsqueeze(0);
  auto t4 = t1 - t3;

  testValidate(&fusion, cg_outputs, {t0}, {t4}, __LINE__, __FILE__, "");
}

} // namespace

TEST_F(OuterReductionTest, GridPersistentReductionOuterNormLikeHalf256x7x512) {
  grid_persistent_reduction_outer_norm_like(256, 7, 512, DataType::Half);
}

TEST_F(OuterReductionTest, GridPersistentReductionOuterNormLikeHalf256x14x512) {
  grid_persistent_reduction_outer_norm_like(256, 14, 512, DataType::Half);
}

TEST_F(OuterReductionTest, GridPersistentReductionOuterNormLikeHalf256x28x512) {
  grid_persistent_reduction_outer_norm_like(256, 28, 512, DataType::Half);
}

TEST_F(OuterReductionTest, GridPersistentReductionOuterNormLikeFloat256x7x512) {
  grid_persistent_reduction_outer_norm_like(256, 7, 512, DataType::Float);
}

TEST_F(
    OuterReductionTest,
    GridPersistentReductionOuterNormLikeFloat256x14x512) {
  grid_persistent_reduction_outer_norm_like(256, 14, 512, DataType::Float);
}

TEST_F(
    OuterReductionTest,
    GridPersistentReductionOuterNormLikeFloat256x28x512) {
  grid_persistent_reduction_outer_norm_like(256, 28, 512, DataType::Float);
}

namespace {

// Test a fusion with outer welford reduction. Similar to outer
// batchnorm but simplified for testing, e.g., no bias and weights
void grid_persistent_welford_outer_norm_like(
    int64_t N,
    int64_t HW,
    int64_t C,
    DataType dtype) {
  const bool benchmark_mode = isBenchmarkMode();
  std::unique_ptr<Fusion> fusion_ptr = std::make_unique<Fusion>();
  Fusion& fusion = *fusion_ptr.get();
  FusionGuard fg(&fusion);

  std::vector<bool> bcast_pattern{true, true, true, false};
  std::vector<int64_t> reduction_dims{2, 1, 0};

  auto tv0 = makeContigTensor(4, dtype);
  fusion.addInput(tv0);
  auto tv1 = dtype == DataType::Half ? castOp(DataType::Float, tv0) : tv0;
  auto tv2 = Welford(tv1, reduction_dims).avg;
  auto tv3 = broadcast(tv2, bcast_pattern);
  auto tv4 = sub(tv1, tv3);
  auto tv5 = dtype == DataType::Half ? castOp(DataType::Half, tv4) : tv4;
  fusion.addOutput(tv5);

  OuterReductionParams params;

  params.configureForForward(N, HW, C, dtype);

  if (benchmark_mode) {
    params.configureWithEnv();
  }

  scheduleNormalization(fusion, params);

  auto options_half =
      at::TensorOptions().dtype(data_type_to_aten(dtype)).device(at::kCUDA, 0);

  const std::vector<int64_t> input_shape{N, HW, HW, C};
  auto t0 = at::randn(input_shape, options_half);

  KernelExecutor ke;
  ke.compile(&fusion, {t0});

  auto bidy = ceilDiv(ceilDiv(N * HW * HW, params.tidy), params.pb);

  if (params.bidx * bidy > deviceSMCount()) {
    GTEST_SKIP() << "Not enough SMs to run this test. Required: "
                 << params.bidx * bidy << ", available: " << deviceSMCount();
  }

  auto cg_outputs = ke.run({t0});

  if (benchmark_mode) {
    for (int i = 0; i < 10; ++i) {
      clearL2Cache();
      cg_outputs = ke.run({t0});
    }
  }

  auto t1 = t0.to(at::kFloat);
  auto t2 = t1.mean({0, 1, 2});
  auto t3 = t2.unsqueeze(0).unsqueeze(0).unsqueeze(0);
  auto t4 = t1 - t3;

  testValidate(&fusion, cg_outputs, {t0}, {t4}, __LINE__, __FILE__, "");
}

} // namespace

TEST_F(OuterReductionTest, GridPersistentWelfordOuterNormLikeHalf256x7x512) {
  grid_persistent_welford_outer_norm_like(256, 7, 512, DataType::Half);
}

TEST_F(OuterReductionTest, GridPersistentWelfordOuterNormLikeHalf256x14x512) {
  grid_persistent_welford_outer_norm_like(256, 14, 512, DataType::Half);
}

TEST_F(OuterReductionTest, GridPersistentWelfordOuterNormLikeHalf256x28x512) {
  grid_persistent_welford_outer_norm_like(256, 28, 512, DataType::Half);
}

TEST_F(OuterReductionTest, GridPersistentWelfordOuterNormLikeHalf256x28x128) {
  grid_persistent_welford_outer_norm_like(256, 28, 128, DataType::Half);
}

TEST_F(OuterReductionTest, GridPersistentWelfordOuterNormLikeHalf32x32x128) {
  grid_persistent_welford_outer_norm_like(32, 32, 128, DataType::Half);
}

// Too large
#if 0
TEST_F(OuterReductionTest, GridPersistentWelfordOuterNormLike64x64x64) {
  grid_persistent_welford_outer_norm_like(64, 64, 64, DataType::Half);
}
#endif

TEST_F(OuterReductionTest, GridPersistentWelfordOuterNormLikeFloat256x7x512) {
  grid_persistent_welford_outer_norm_like(256, 7, 512, DataType::Float);
}

TEST_F(OuterReductionTest, GridPersistentWelfordOuterNormLikeFloat256x14x512) {
  grid_persistent_welford_outer_norm_like(256, 14, 512, DataType::Float);
}

TEST_F(OuterReductionTest, GridPersistentWelfordOuterNormLikeFloat256x28x512) {
  grid_persistent_welford_outer_norm_like(256, 28, 512, DataType::Float);
}

TEST_F(OuterReductionTest, GridPersistentWelfordOuterNormLikeFloat256x28x128) {
  grid_persistent_welford_outer_norm_like(256, 28, 128, DataType::Float);
}

TEST_F(OuterReductionTest, GridPersistentWelfordOuterNormLikeFloat32x32x128) {
  grid_persistent_welford_outer_norm_like(32, 32, 128, DataType::Float);
}

namespace {

// Test a forward outer batchorm
void grid_persistent_batchnorm_manual(
    int64_t N,
    int64_t HW,
    int64_t C,
    DataType dtype) {
  const bool benchmark_mode = isBenchmarkMode();
  std::unique_ptr<Fusion> fusion_ptr = std::make_unique<Fusion>();
  Fusion& fusion = *fusion_ptr.get();
  FusionGuard fg(fusion_ptr.get());

  const bool kTraining = true;
  const float kMomentum = 0.1;
  const float kEps = 1e-5;

  // setup fusion
  auto input = makeContigTensor(4, dtype);
  auto weight = makeContigTensor(1, dtype);
  auto bias = makeContigTensor(1, dtype);
  auto running_mean = makeContigTensor(1, DataType::Float);
  auto running_var = makeContigTensor(1, DataType::Float);

  fusion_ptr->addInput(input);
  fusion_ptr->addInput(weight);
  fusion_ptr->addInput(bias);
  fusion_ptr->addInput(running_mean);
  fusion_ptr->addInput(running_var);

  if (dtype == DataType::Half) {
    input = castOp(DataType::Float, input);
    weight = castOp(DataType::Float, weight);
    bias = castOp(DataType::Float, bias);
  }

  auto momentum_ptr = IrBuilder::create<Val>(kMomentum);
  auto eps_ptr = IrBuilder::create<Val>(kEps);

  auto result = batch_norm(
      input,
      weight,
      bias,
      running_mean,
      running_var,
      kTraining,
      momentum_ptr,
      eps_ptr,
      true);

  auto output = result.output;

  if (dtype == DataType::Half) {
    output = castOp(DataType::Half, output);
  }

  fusion_ptr->addOutput(output);

  OuterReductionParams params;

  params.configureForForward(N, HW, C, dtype);

  if (benchmark_mode) {
    params.configureWithEnv();
  }

  scheduleNormalization(fusion, params);

  auto options_float =
      at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  auto options =
      at::TensorOptions().dtype(data_type_to_aten(dtype)).device(at::kCUDA, 0);

  auto at_input = at::randn({N, C, HW, HW}, options)
                      .contiguous(c10::MemoryFormat::ChannelsLast);
  auto at_input_nvfuser = at_input.clone().detach().permute({0, 2, 3, 1});

  auto at_weight = at::randn({C}, options);
  auto at_bias = at::randn({C}, options);
  auto at_running_mean = at::randn({C}, options_float);
  auto at_running_var = at::randn({C}, options_float);

  KernelArgumentHolder inputs(
      {at_input_nvfuser, at_weight, at_bias, at_running_mean, at_running_var});

  KernelExecutor ke;
  ke.compile(fusion_ptr.get(), inputs);

  auto bidy = ceilDiv(ceilDiv(N * HW * HW, params.tidy), params.pb);

  if (params.bidx * bidy > deviceSMCount()) {
    GTEST_SKIP() << "Not enough SMs to run this test. Required: "
                 << params.bidx * bidy << ", available: " << deviceSMCount();
  }

  auto cg_outputs = ke.run(inputs);

  auto at_output = at::batch_norm(
      at_input,
      at_weight,
      at_bias,
      at_running_mean,
      at_running_var,
      kTraining,
      kMomentum,
      kEps,
      true);

  testValidate(
      ke.compiledKernel()->kernel(),
      cg_outputs[2].as<at::Tensor>().permute({0, 3, 1, 2}),
      inputs,
      {at_output},
      __LINE__,
      __FILE__,
      "");

  if (benchmark_mode) {
    for (int i = 0; i < 10; ++i) {
      clearL2Cache();
      cg_outputs = ke.run(inputs);
    }
  }
}

} // namespace

TEST_F(OuterReductionTest, GridPersistentBatchNormChannelsLastHalf256x7x512) {
  grid_persistent_batchnorm_manual(256, 7, 512, DataType::Half);
}

TEST_F(OuterReductionTest, GridPersistentBatchNormChannelsLastHalf256x14x512) {
  grid_persistent_batchnorm_manual(256, 14, 512, DataType::Half);
}

TEST_F(OuterReductionTest, GridPersistentBatchNormChannelsLastHalf256x28x512) {
  grid_persistent_batchnorm_manual(256, 28, 512, DataType::Half);
}

TEST_F(OuterReductionTest, GridPersistentBatchNormChannelsLastHalf256x28x128) {
  grid_persistent_batchnorm_manual(256, 28, 128, DataType::Half);
}

TEST_F(OuterReductionTest, GridPersistentBatchNormChannelsLastFloat256x7x512) {
  grid_persistent_batchnorm_manual(256, 7, 512, DataType::Float);
}

TEST_F(OuterReductionTest, GridPersistentBatchNormChannelsLastFloat256x14x512) {
  grid_persistent_batchnorm_manual(256, 14, 512, DataType::Float);
}

TEST_F(OuterReductionTest, GridPersistentBatchNormChannelsLastFloat256x28x512) {
  grid_persistent_batchnorm_manual(256, 28, 512, DataType::Float);
}

TEST_F(OuterReductionTest, GridPersistentBatchNormChannelsLastFloat256x28x128) {
  grid_persistent_batchnorm_manual(256, 28, 128, DataType::Float);
}

namespace {

// Test a fusion with two outer grid reductions. Similar to outer
// backward batchnorm but simplified for testing, e.g., just two single
// reductions rather than welford
void grid_persistent_reduction_outer_norm_bwd_like(
    int64_t N,
    int64_t HW,
    int64_t C,
    DataType dtype) {
  const bool benchmark_mode = isBenchmarkMode();
  std::unique_ptr<Fusion> fusion_ptr = std::make_unique<Fusion>();
  Fusion& fusion = *fusion_ptr.get();
  FusionGuard fg(&fusion);

  std::vector<bool> bcast_pattern{true, true, true, false};
  std::vector<int64_t> reduction_dims{2, 1, 0};

  // grad_output
  auto tv0 = makeContigTensor(4, dtype);
  fusion.addInput(tv0);
  // input
  auto tv1 = makeContigTensor(4, dtype);
  fusion.addInput(tv1);

  auto norm =
      IrBuilder::create<Val>(1.0 / ((double)N * (double)HW * (double)HW));

  auto tv2 = dtype == DataType::Half ? castOp(DataType::Float, tv0) : tv0;
  auto tv3 = dtype == DataType::Half ? castOp(DataType::Float, tv1) : tv1;
  // grad_output_sum-like pattern
  auto tv4 = sum(tv2, reduction_dims);
  auto tv5 = mul(tv4, norm);
  auto tv6 = broadcast(tv5, bcast_pattern);
  // dot_p-like pattern
  auto tv7 = sub(tv2, tv3);
  auto tv8 = sum(tv7, reduction_dims);
  auto tv9 = mul(tv8, norm);
  auto tv10 = broadcast(tv9, bcast_pattern);

  auto tv11 = mul(tv3, tv10);
  auto tv12 = sub(tv2, tv11);
  auto tv13 = sub(tv12, tv6);
  auto tv14 = dtype == DataType::Half ? castOp(DataType::Half, tv13) : tv13;
  fusion.addOutput(tv14);

  OuterReductionParams params;

  params.configureForBackward(N, HW, C, dtype);

  if (benchmark_mode) {
    params.configureWithEnv();
  }

  scheduleNormalization(fusion, params);

  auto options =
      at::TensorOptions().dtype(data_type_to_aten(dtype)).device(at::kCUDA, 0);

  const std::vector<int64_t> input_shape{N, HW, HW, C};
  auto t0 = at::randn(input_shape, options);
  auto t1 = at::randn(input_shape, options);
  KernelArgumentHolder inputs({t0, t1});

  KernelExecutor ke;
  ke.compile(&fusion, inputs);

  auto bidy = ceilDiv(ceilDiv(N * HW * HW, params.tidy), params.pb);

  if (params.bidx * bidy > deviceSMCount()) {
    GTEST_SKIP() << "Not enough SMs to run this test. Required: "
                 << params.bidx * bidy << ", available: " << deviceSMCount();
  }

  auto cg_outputs = ke.run(inputs);

  if (benchmark_mode) {
    for (int i = 0; i < 10; ++i) {
      clearL2Cache();
      cg_outputs = ke.run(inputs);
    }
  }

  auto norm_double = 1.0 / ((double)N * (double)HW * (double)HW);
  auto t4 = t0.to(at::kFloat);
  auto t5 = t1.to(at::kFloat);
  auto t6 = sum(t4, {0, 1, 2});
  auto t7 = t6 * norm_double;
  auto t8 = t7.unsqueeze(0).unsqueeze(0).unsqueeze(0);
  auto t9 = t4 - t5;
  auto t10 = sum(t9, {0, 1, 2});
  auto t11 = t10 * norm_double;
  auto t12 = t11.unsqueeze(0).unsqueeze(0).unsqueeze(0);

  // Second use of manually projected persistent buffer
  auto t13 = t0.to(at::kFloat);
  auto t14 = t1.to(at::kFloat);
  auto t15 = t14 * t12;
  auto t16 = t13 - t15;
  auto t17 = t16 - t8;

  testValidate(&fusion, cg_outputs, inputs, {t17}, __LINE__, __FILE__, "");
}

} // namespace

TEST_F(
    OuterReductionTest,
    GridPersistentReductionOuterNormBwdLikeHalf256x7x512) {
  grid_persistent_reduction_outer_norm_bwd_like(256, 7, 512, DataType::Half);
}

TEST_F(
    OuterReductionTest,
    GridPersistentReductionOuterNormBwdLikeHalf256x14x512) {
  grid_persistent_reduction_outer_norm_bwd_like(256, 14, 512, DataType::Half);
}

TEST_F(
    OuterReductionTest,
    GridPersistentReductionOuterNormBwdLikeHalf256x28x512) {
  grid_persistent_reduction_outer_norm_bwd_like(256, 28, 512, DataType::Half);
}

TEST_F(
    OuterReductionTest,
    GridPersistentReductionOuterNormBwdLikeFloat256x7x512) {
  grid_persistent_reduction_outer_norm_bwd_like(256, 7, 512, DataType::Float);
}

TEST_F(
    OuterReductionTest,
    GridPersistentReductionOuterNormBwdLikeFloat256x14x512) {
  grid_persistent_reduction_outer_norm_bwd_like(256, 14, 512, DataType::Float);
}

TEST_F(
    OuterReductionTest,
    GridPersistentReductionOuterNormBwdLikeFloat256x28x512) {
  grid_persistent_reduction_outer_norm_bwd_like(256, 28, 512, DataType::Float);
}

namespace {

// Test a backward outer batchorm
void grid_persistent_batchnorm_bwd_manual(
    int64_t N,
    int64_t HW,
    int64_t C,
    DataType dtype) {
  const bool benchmark_mode = isBenchmarkMode();
  std::unique_ptr<Fusion> fusion_ptr = std::make_unique<Fusion>();
  Fusion& fusion = *fusion_ptr.get();
  FusionGuard fg(fusion_ptr.get());

  const bool kTraining = true;
  const float kEps = 1e-5;

  // setup fusion
  auto input = makeContigTensor(4, dtype);
  auto grad_output = makeContigTensor(4, dtype);
  auto weight = makeContigTensor(1, DataType::Float);
  auto running_mean = makeContigTensor(1, DataType::Float);
  auto running_var = makeContigTensor(1, DataType::Float);
  auto save_mean = makeContigTensor(1, DataType::Float);
  auto save_var = makeContigTensor(1, DataType::Float);

  fusion.addInput(input);
  fusion.addInput(grad_output);
  fusion.addInput(weight);
  fusion.addInput(running_mean);
  fusion.addInput(running_var);
  fusion.addInput(save_mean);
  fusion.addInput(save_var);

  if (dtype == DataType::Half) {
    input = castOp(DataType::Float, input);
    grad_output = castOp(DataType::Float, grad_output);
  }

  auto eps_ptr = IrBuilder::create<Val>(kEps);

  auto result = batch_norm_backward(
      input,
      grad_output,
      weight,
      running_mean,
      running_var,
      save_mean,
      save_var,
      kTraining,
      eps_ptr,
      std::vector<bool>(3, true),
      true);

  auto grad_input = result.grad_input;
  auto grad_weight = result.grad_weight;
  auto grad_bias = result.grad_bias;

  if (dtype == DataType::Half) {
    grad_input = castOp(DataType::Half, grad_input);
    grad_weight = castOp(DataType::Half, grad_weight);
    grad_bias = castOp(DataType::Half, grad_bias);
  }

  fusion.addOutput(grad_input);
  fusion.addOutput(grad_weight);
  fusion.addOutput(grad_bias);

  OuterReductionParams params;

  params.configureForBackward(N, HW, C, dtype);

  if (benchmark_mode) {
    params.configureWithEnv();
  }

  scheduleNormalization(fusion, params);

  auto options_float =
      at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  auto options =
      at::TensorOptions().dtype(data_type_to_aten(dtype)).device(at::kCUDA, 0);

  const std::vector<int64_t> input_shape{N, HW, HW, C};

  auto at_input = at::randn({N, C, HW, HW}, options)
                      .contiguous(c10::MemoryFormat::ChannelsLast);
  auto at_input_nvfuser = at_input.clone().detach().permute({0, 2, 3, 1});

  auto at_grad_out = at::randn({N, C, HW, HW}, options)
                         .contiguous(c10::MemoryFormat::ChannelsLast);
  auto at_grad_out_nvfuser = at_grad_out.clone().detach().permute({0, 2, 3, 1});

  at::Tensor at_weight = at::ones({C}, options_float);
  at::Tensor at_run_mean = at::zeros({C}, options_float);
  at::Tensor at_run_var = at::ones({C}, options_float);
  at::Tensor at_save_mean = at::zeros({C}, options_float);
  at::Tensor at_save_var = at::ones({C}, options_float);

  KernelArgumentHolder inputs(
      {at_input_nvfuser,
       at_grad_out_nvfuser,
       at_weight,
       at_run_mean,
       at_run_var,
       at_save_mean,
       at_save_var});

  KernelExecutor ke;
  ke.compile(fusion_ptr.get(), inputs);

  auto bidy = ceilDiv(ceilDiv(N * HW * HW, params.tidy), params.pb);

  if (params.bidx * bidy > deviceSMCount()) {
    GTEST_SKIP() << "Not enough SMs to run this test. Required: "
                 << params.bidx * bidy << ", available: " << deviceSMCount();
  }

  auto cg_outputs = ke.run(inputs);
  // Permute grad_input output
  cg_outputs[0] = cg_outputs[0].as<at::Tensor>().permute({0, 3, 1, 2});

  auto at_output = at::native_batch_norm_backward(
      at_grad_out,
      at_input,
      at_weight,
      at_run_mean,
      at_run_var,
      at_save_mean,
      at_save_var,
      true,
      kEps,
      {true, true, true});

  testValidate(
      ke.compiledKernel()->kernel(),
      cg_outputs,
      inputs,
      {std::get<0>(at_output), std::get<1>(at_output), std::get<2>(at_output)},
      __LINE__,
      __FILE__,
      "");

  if (benchmark_mode) {
    for (int i = 0; i < 10; ++i) {
      clearL2Cache();
      cg_outputs = ke.run(inputs);
    }
  }
}

} // namespace

TEST_F(
    OuterReductionTest,
    GridPersistentBatchNormChannelsLastBwdHalf256x7x512) {
  grid_persistent_batchnorm_bwd_manual(256, 7, 512, DataType::Half);
}

TEST_F(
    OuterReductionTest,
    GridPersistentBatchNormChannelsLastBwdHalf256x14x512) {
  grid_persistent_batchnorm_bwd_manual(256, 14, 512, DataType::Half);
}

TEST_F(
    OuterReductionTest,
    GridPersistentBatchNormChannelsLastBwdHalf256x28x512) {
  grid_persistent_batchnorm_bwd_manual(256, 28, 512, DataType::Half);
}

TEST_F(
    OuterReductionTest,
    GridPersistentBatchNormChannelsLastBwdFloat256x7x512) {
  grid_persistent_batchnorm_bwd_manual(256, 7, 512, DataType::Float);
}

TEST_F(
    OuterReductionTest,
    GridPersistentBatchNormChannelsLastBwdFloat256x14x512) {
  grid_persistent_batchnorm_bwd_manual(256, 14, 512, DataType::Float);
}

TEST_F(
    OuterReductionTest,
    GridPersistentBatchNormChannelsLastBwdFloat256x28x512) {
  grid_persistent_batchnorm_bwd_manual(256, 28, 512, DataType::Float);
}

////////////////////////////////////////////////////////////////
/// Scheduler tests
////////////////////////////////////////////////////////////////

namespace {

TensorView* cast(TensorView* tv, DataType dtype) {
  if (tv->getDataType() != dtype) {
    return castOp(dtype, tv);
  } else {
    return tv;
  }
}

bool shouldBePersistent(
    int64_t N,
    int64_t HW,
    DataType dtype,
    bool is_bwd,
    bool use_weights = false,
    DataType weights_dtype = DataType::Float) {
  // Non-welford is disabled for now
  if (is_bwd) {
    return false;
  }

  const int64_t vec_factor = 16 /
      std::max(dataTypeSizeByte(dtype),
               (use_weights ? dataTypeSizeByte(weights_dtype) : 1));

  const int64_t num_threads = 256;
  const int64_t min_bdimx = 8;
  const int64_t max_bdimy = num_threads / min_bdimx;
  const int64_t max_gdimy =
      at::cuda::getCurrentDeviceProperties()->multiProcessorCount / 2;
  const int64_t pb_factor = ceilDiv(ceilDiv(N * HW * HW, max_bdimy), max_gdimy);
  const int64_t req_reg_count = pb_factor * vec_factor *
      dataTypeSizeByte(dtype) / sizeof(int) *
      (is_bwd ? 2 : 1); // Two tensors are cached in the backward batchnorm

  // The scheduler sets aside (pb_factor + 35) registers
  return req_reg_count <= 255 - (pb_factor + 35);
}

} // namespace

// TODO: Enable once non-welford grid reductions are supported
#if 0
namespace {

// Forward grid reduction
void grid_persistent_reduction_outer_norm_like_scheduler(
    int64_t N,
    int64_t HW,
    int64_t C,
    DataType dtype,
    bool use_weights = false,
    DataType weights_dtype = DataType::Float) {
  const bool benchmark_mode = isBenchmarkMode();
  std::unique_ptr<Fusion> fusion_ptr = std::make_unique<Fusion>();
  Fusion& fusion = *fusion_ptr.get();
  FusionGuard fg(&fusion);

  std::vector<bool> bcast_pattern{true, true, true, false};
  std::vector<int64_t> reduction_dims{2, 1, 0};

  auto inp = makeContigTensor(4, dtype);
  fusion.addInput(inp);

  TensorView* weights = nullptr;
  if (use_weights) {
    weights = makeContigTensor(1, weights_dtype);
    fusion.addInput(weights);
  }

  auto inp_cast = cast(inp, DataType::Float);
  auto inp_allreduce = broadcast(sum(inp_cast, reduction_dims), bcast_pattern);
  auto out = sub(inp_cast, inp_allreduce);

  if (use_weights) {
    out = add(out, broadcast(cast(weights, DataType::Float), bcast_pattern));
  }

  out = cast(out, dtype);
  fusion.addOutput(out);

  auto options =
      at::TensorOptions().dtype(data_type_to_aten(dtype)).device(at::kCUDA, 0);
  auto options_weight = at::TensorOptions()
                            .dtype(data_type_to_aten(weights_dtype))
                            .device(at::kCUDA, 0);

  const std::vector<int64_t> input_shape{N, HW, HW, C};
  auto t0 = at::randn(input_shape, options);
  auto t1 = at::randn({C}, options_weight);
  KernelArgumentHolder args = {t0};
  if (use_weights) {
    args.push(t1);
  }

  FusionExecutorCache executor_cache(std::move(fusion_ptr));
  auto cg_outputs = executor_cache.runFusionWithInputs(args);

  auto runtime = executor_cache.getMostRecentKernelRuntime();

  if (!shouldBePersistent(N, HW, dtype, false, use_weights, weights_dtype)) {
    NVF_CHECK(runtime->isSegmented(), "Expected to be segmented");
  } else {
    NVF_CHECK(
        !runtime->isSegmented(),
        "Unexpected number of segments: ",
        runtime->fusionSegments()->groups().size());

    const auto& heuristic_params =
        runtime->schedulerHeuristics()->heuristicsList().at(0);
    NVF_CHECK(
        heuristic_params->scheduler_type == SchedulerType::OuterPersistent,
        "Unexpected heuristic was chosen: ",
        heuristic_params->scheduler_type);

    if (benchmark_mode) {
      for (int i = 0; i < 10; ++i) {
        clearL2Cache();
        cg_outputs = executor_cache.runFusionWithInputs(args);
      }
    }
  }

  auto t0_cast = t0.to(at::kFloat);
  auto t0_allreduce =
      t0_cast.sum({0, 1, 2}).unsqueeze(0).unsqueeze(0).unsqueeze(0);
  auto ref = t0_cast - t0_allreduce;
  if (use_weights) {
    ref = ref + t1.to(at::kFloat);
  }

  testValidate(&fusion, cg_outputs, args, {ref}, __LINE__, __FILE__, "");
}

} // namespace

TEST_F(
    OuterReductionTest,
    GridPersistentReductionOuterNormLikeHalf256x7x512Scheduler) {
  grid_persistent_reduction_outer_norm_like_scheduler(
      256, 7, 512, DataType::Half);
}

TEST_F(
    OuterReductionTest,
    GridPersistentReductionOuterNormLikeHalf256x14x512Scheduler) {
  grid_persistent_reduction_outer_norm_like_scheduler(
      256, 14, 512, DataType::Half);
}

TEST_F(
    OuterReductionTest,
    GridPersistentReductionOuterNormLikeHalf256x28x512Scheduler) {
  grid_persistent_reduction_outer_norm_like_scheduler(
      256, 28, 512, DataType::Half);
}

TEST_F(
    OuterReductionTest,
    GridPersistentReductionOuterNormLikeFloat256x7x512Scheduler) {
  grid_persistent_reduction_outer_norm_like_scheduler(
      256, 7, 512, DataType::Float);
}

TEST_F(
    OuterReductionTest,
    GridPersistentReductionOuterNormLikeFloat256x14x512Scheduler) {
  grid_persistent_reduction_outer_norm_like_scheduler(
      256, 14, 512, DataType::Float);
}

TEST_F(
    OuterReductionTest,
    GridPersistentReductionOuterNormLikeFloat256x28x512Scheduler) {
  grid_persistent_reduction_outer_norm_like_scheduler(
      256, 28, 512, DataType::Float);
}

TEST_F(
    OuterReductionTest,
    GridPersistentReductionOuterNormWithWeightsLikeHalf256x7x512Scheduler) {
  grid_persistent_reduction_outer_norm_like_scheduler(
      256, 7, 512, DataType::Half, true, DataType::Float);
}

TEST_F(
    OuterReductionTest,
    GridPersistentReductionOuterNormWithWeightsLikeHalf256x14x512Scheduler) {
  grid_persistent_reduction_outer_norm_like_scheduler(
      256, 14, 512, DataType::Half, true, DataType::Float);
}

TEST_F(
    OuterReductionTest,
    GridPersistentReductionOuterNormWithWeightsLikeHalf256x28x512Scheduler) {
  grid_persistent_reduction_outer_norm_like_scheduler(
      256, 28, 512, DataType::Half, true, DataType::Float);
}
#endif

namespace {

// Forward welford
void grid_persistent_welford_outer_norm_like_scheduler(
    int64_t N,
    int64_t HW,
    int64_t C,
    DataType dtype,
    bool use_weights = false,
    DataType weights_dtype = DataType::Float) {
  const bool benchmark_mode = isBenchmarkMode();
  std::unique_ptr<Fusion> fusion_ptr = std::make_unique<Fusion>();
  Fusion& fusion = *fusion_ptr.get();
  FusionGuard fg(&fusion);

  std::vector<bool> bcast_pattern{true, true, true, false};
  std::vector<int64_t> reduction_dims{2, 1, 0};

  auto inp = makeContigTensor(4, dtype);
  fusion.addInput(inp);

  TensorView* weights = nullptr;
  if (use_weights) {
    weights = makeContigTensor(1, weights_dtype);
    fusion.addInput(weights);
  }

  auto inp_cast = cast(inp, DataType::Float);
  auto inp_allreduce =
      broadcast(Welford(inp_cast, reduction_dims).avg, bcast_pattern);
  auto out = sub(inp_cast, inp_allreduce);

  if (use_weights) {
    out = add(out, broadcast(cast(weights, DataType::Float), bcast_pattern));
  }

  out = cast(out, dtype);
  fusion.addOutput(out);

  auto options =
      at::TensorOptions().dtype(data_type_to_aten(dtype)).device(at::kCUDA, 0);
  auto options_weight = at::TensorOptions()
                            .dtype(data_type_to_aten(weights_dtype))
                            .device(at::kCUDA, 0);

  const std::vector<int64_t> input_shape{N, HW, HW, C};
  auto t0 = at::randn(input_shape, options);
  auto t1 = at::randn({C}, options_weight);
  KernelArgumentHolder args = {t0};
  if (use_weights) {
    args.push(t1);
  }

  FusionExecutorCache executor_cache(std::move(fusion_ptr));
  auto cg_outputs = executor_cache.runFusionWithInputs(args);

  auto runtime = executor_cache.getMostRecentKernelRuntime();

  if (!shouldBePersistent(N, HW, dtype, false, use_weights, weights_dtype)) {
    NVF_CHECK(runtime->isSegmented(), "Expected to be segmented");
  } else {
    NVF_CHECK(
        !runtime->isSegmented(),
        "Unexpected number of segments: ",
        runtime->fusionSegments()->groups().size());

    const auto& heuristic_params =
        runtime->schedulerHeuristics()->heuristicsList().at(0);
    NVF_CHECK(
        heuristic_params->scheduler_type == SchedulerType::OuterPersistent,
        "Unexpected heuristic was chosen: ",
        heuristic_params->scheduler_type);

    if (benchmark_mode) {
      for (int i = 0; i < 10; ++i) {
        clearL2Cache();
        cg_outputs = executor_cache.runFusionWithInputs(args);
      }
    }
  }

  auto t0_cast = t0.to(at::kFloat);
  auto t0_allreduce =
      t0_cast.mean({0, 1, 2}).unsqueeze(0).unsqueeze(0).unsqueeze(0);
  auto ref = t0_cast - t0_allreduce;
  if (use_weights) {
    ref = ref + t1.to(at::kFloat);
  }

  testValidate(&fusion, cg_outputs, args, {ref}, __LINE__, __FILE__, "");
}

} // namespace

TEST_F(
    OuterReductionTest,
    GridPersistentWelfordOuterNormLikeHalf256x7x512Scheduler) {
  grid_persistent_welford_outer_norm_like_scheduler(
      256, 7, 512, DataType::Half);
}

TEST_F(
    OuterReductionTest,
    GridPersistentWelfordOuterNormLikeHalf256x14x512Scheduler) {
  grid_persistent_welford_outer_norm_like_scheduler(
      256, 14, 512, DataType::Half);
}

TEST_F(
    OuterReductionTest,
    GridPersistentWelfordOuterNormLikeHalf256x28x512Scheduler) {
  grid_persistent_welford_outer_norm_like_scheduler(
      256, 28, 512, DataType::Half);
}

TEST_F(
    OuterReductionTest,
    GridPersistentWelfordOuterNormLikeFloat256x7x512Scheduler) {
  grid_persistent_welford_outer_norm_like_scheduler(
      256, 7, 512, DataType::Float);
}

TEST_F(
    OuterReductionTest,
    GridPersistentWelfordOuterNormLikeFloat256x14x512Scheduler) {
  grid_persistent_welford_outer_norm_like_scheduler(
      256, 14, 512, DataType::Float);
}

TEST_F(
    OuterReductionTest,
    GridPersistentWelfordOuterNormLikeFloat256x28x512Scheduler) {
  grid_persistent_welford_outer_norm_like_scheduler(
      256, 28, 512, DataType::Float);
}

TEST_F(
    OuterReductionTest,
    GridPersistentWelfordOuterNormWithWeithtsLikeHalf256x7x512Scheduler) {
  grid_persistent_welford_outer_norm_like_scheduler(
      256, 7, 512, DataType::Half, true, DataType::Float);
}

TEST_F(
    OuterReductionTest,
    GridPersistentWelfordOuterNormWithWeightsLikeWHalf256x14x512Scheduler) {
  grid_persistent_welford_outer_norm_like_scheduler(
      256, 14, 512, DataType::Half, true, DataType::Float);
}

TEST_F(
    OuterReductionTest,
    GridPersistentWelfordOuterNormWithWeightsLikeWHalf256x28x512Scheduler) {
  grid_persistent_welford_outer_norm_like_scheduler(
      256, 28, 512, DataType::Half, true, DataType::Float);
}

namespace {

// Forward batchnorm
void grid_persistent_batchnorm_scheduler(
    int64_t N,
    int64_t HW,
    int64_t C,
    DataType dtype) {
  const bool benchmark_mode = isBenchmarkMode();
  std::unique_ptr<Fusion> fusion_ptr = std::make_unique<Fusion>();
  Fusion& fusion = *fusion_ptr.get();
  FusionGuard fg(fusion_ptr.get());

  const bool kTraining = true;
  const float kMomentum = 0.1;
  const float kEps = 1e-5;

  // setup fusion
  auto input = makeContigTensor(4, dtype);
  auto weight = makeContigTensor(1, dtype);
  auto bias = makeContigTensor(1, dtype);
  auto running_mean = makeContigTensor(1, DataType::Float);
  auto running_var = makeContigTensor(1, DataType::Float);

  fusion_ptr->addInput(input);
  fusion_ptr->addInput(weight);
  fusion_ptr->addInput(bias);
  fusion_ptr->addInput(running_mean);
  fusion_ptr->addInput(running_var);

  if (dtype == DataType::Half) {
    input = castOp(DataType::Float, input);
    weight = castOp(DataType::Float, weight);
    bias = castOp(DataType::Float, bias);
  }

  auto momentum_ptr = IrBuilder::create<Val>(kMomentum);
  auto eps_ptr = IrBuilder::create<Val>(kEps);

  auto result = batch_norm(
      input,
      weight,
      bias,
      running_mean,
      running_var,
      kTraining,
      momentum_ptr,
      eps_ptr,
      true);

  auto output = result.output;

  if (dtype == DataType::Half) {
    output = castOp(DataType::Half, output);
  }

  fusion_ptr->addOutput(output);

  auto options_float =
      at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  auto options =
      at::TensorOptions().dtype(data_type_to_aten(dtype)).device(at::kCUDA, 0);

  auto at_input = at::randn({N, C, HW, HW}, options)
                      .contiguous(c10::MemoryFormat::ChannelsLast);
  auto at_input_nvfuser = at_input.clone().detach().permute({0, 2, 3, 1});

  auto at_weight = at::randn({C}, options);
  auto at_bias = at::randn({C}, options);
  auto at_running_mean = at::randn({C}, options_float);
  auto at_running_var = at::randn({C}, options_float);

  KernelArgumentHolder args = {
      at_input_nvfuser, at_weight, at_bias, at_running_mean, at_running_var};

  FusionExecutorCache executor_cache(std::move(fusion_ptr));
  auto cg_outputs = executor_cache.runFusionWithInputs(args);

  auto runtime = executor_cache.getMostRecentKernelRuntime();

  if (!shouldBePersistent(N, HW, dtype, false, true, DataType::Float)) {
    NVF_CHECK(runtime->isSegmented(), "Expected to be segmented");
  } else {
    NVF_CHECK(
        !runtime->isSegmented(),
        "Unexpected number of segments: ",
        runtime->fusionSegments()->groups().size());

    const auto& heuristic_params =
        runtime->schedulerHeuristics()->heuristicsList().at(0);
    NVF_CHECK(
        heuristic_params->scheduler_type == SchedulerType::OuterPersistent,
        "Unexpected heuristic was chosen: ",
        heuristic_params->scheduler_type);

    if (benchmark_mode) {
      for (int i = 0; i < 10; ++i) {
        clearL2Cache();
        cg_outputs = executor_cache.runFusionWithInputs(args);
      }
    }
  }

  auto at_output = at::batch_norm(
      at_input,
      at_weight,
      at_bias,
      at_running_mean,
      at_running_var,
      kTraining,
      kMomentum,
      kEps,
      true);

  cg_outputs[0] = cg_outputs[0].as<at::Tensor>().permute({0, 3, 1, 2});

  testValidate(&fusion, cg_outputs, args, {at_output}, __LINE__, __FILE__, "");
}

} // namespace

TEST_F(
    OuterReductionTest,
    GridPersistentBatchNormChannelsLastHalf256x7x512Scheduler) {
  grid_persistent_batchnorm_scheduler(256, 7, 512, DataType::Half);
}

TEST_F(
    OuterReductionTest,
    GridPersistentBatchNormChannelsLastHalf256x14x512Scheduler) {
  grid_persistent_batchnorm_scheduler(256, 14, 512, DataType::Half);
}

TEST_F(
    OuterReductionTest,
    GridPersistentBatchNormChannelsLastHalf256x28x512Scheduler) {
  grid_persistent_batchnorm_scheduler(256, 28, 512, DataType::Half);
}

TEST_F(
    OuterReductionTest,
    GridPersistentBatchNormChannelsLastFloat256x7x512Scheduler) {
  grid_persistent_batchnorm_scheduler(256, 7, 512, DataType::Float);
}

TEST_F(
    OuterReductionTest,
    GridPersistentBatchNormChannelsLastFloat256x14x512Scheduler) {
  grid_persistent_batchnorm_scheduler(256, 14, 512, DataType::Float);
}

TEST_F(
    OuterReductionTest,
    GridPersistentBatchNormChannelsLastFloat256x28x512Scheduler) {
  grid_persistent_batchnorm_scheduler(256, 28, 512, DataType::Float);
}

// TODO: Enable once non-welford grid reductions are supported
#if 0
namespace {

// Backward grid reduction
void grid_persistent_reduction_outer_norm_bwd_like_scheduler(
    int64_t N,
    int64_t HW,
    int64_t C,
    DataType dtype) {
  const bool benchmark_mode = isBenchmarkMode();
  std::unique_ptr<Fusion> fusion_ptr = std::make_unique<Fusion>();
  Fusion& fusion = *fusion_ptr.get();
  FusionGuard fg(&fusion);

  std::vector<bool> bcast_pattern{true, true, true, false};
  std::vector<int64_t> reduction_dims{2, 1, 0};

  // grad_output
  auto tv0 = makeContigTensor(4, dtype);
  fusion.addInput(tv0);
  // input
  auto tv1 = makeContigTensor(4, dtype);
  fusion.addInput(tv1);

  auto norm =
      IrBuilder::create<Val>(1.0 / ((double)N * (double)HW * (double)HW));

  auto tv2 = dtype == DataType::Half ? castOp(DataType::Float, tv0) : tv0;
  auto tv3 = dtype == DataType::Half ? castOp(DataType::Float, tv1) : tv1;
  // grad_output_sum-like pattern
  auto tv4 = sum(tv2, reduction_dims);
  auto tv5 = mul(tv4, norm);
  auto tv6 = broadcast(tv5, bcast_pattern);
  // dot_p-like pattern
  auto tv7 = sub(tv2, tv3);
  auto tv8 = sum(tv7, reduction_dims);
  auto tv9 = mul(tv8, norm);
  auto tv10 = broadcast(tv9, bcast_pattern);

  auto tv11 = mul(tv3, tv10);
  auto tv12 = sub(tv2, tv11);
  auto tv13 = sub(tv12, tv6);
  auto tv14 = dtype == DataType::Half ? castOp(DataType::Half, tv13) : tv13;
  fusion.addOutput(tv14);

  auto options =
      at::TensorOptions().dtype(data_type_to_aten(dtype)).device(at::kCUDA, 0);

  const std::vector<int64_t> input_shape{N, HW, HW, C};
  auto t0 = at::randn(input_shape, options);
  auto t1 = at::randn(input_shape, options);
  KernelArgumentHolder args = {t0, t1};

  FusionExecutorCache executor_cache(std::move(fusion_ptr));
  auto cg_outputs = executor_cache.runFusionWithInputs(args);

  auto runtime = executor_cache.getMostRecentKernelRuntime();

  if (!shouldBePersistent(N, HW, dtype, true)) {
    NVF_CHECK(runtime->isSegmented(), "Expected to be segmented");
  } else {
    NVF_CHECK(
        !runtime->isSegmented(),
        "Unexpected number of segments: ",
        runtime->fusionSegments()->groups().size());

    const auto& heuristic_params =
        runtime->schedulerHeuristics()->heuristicsList().at(0);
    NVF_CHECK(
        heuristic_params->scheduler_type == SchedulerType::OuterPersistent,
        "Unexpected heuristic was chosen: ",
        heuristic_params->scheduler_type);

    if (benchmark_mode) {
      for (int i = 0; i < 10; ++i) {
        clearL2Cache();
        cg_outputs = executor_cache.runFusionWithInputs(args);
      }
    }
  }

  testValidate(&fusion, cg_outputs, args, __LINE__, __FILE__, "");
}

} // namespace

TEST_F(
    OuterReductionTest,
    GridPersistentReductionOuterNormBwdLikeHalf256x7x512Scheduler) {
  grid_persistent_reduction_outer_norm_bwd_like_scheduler(
      256, 7, 512, DataType::Half);
}

TEST_F(
    OuterReductionTest,
    GridPersistentReductionOuterNormBwdLikeHalf256x14x512Scheduler) {
  grid_persistent_reduction_outer_norm_bwd_like_scheduler(
      256, 14, 512, DataType::Half);
}

TEST_F(
    OuterReductionTest,
    GridPersistentReductionOuterNormBwdLikeHalf256x28x512Scheduler) {
  grid_persistent_reduction_outer_norm_bwd_like_scheduler(
      256, 28, 512, DataType::Half);
}

TEST_F(
    OuterReductionTest,
    GridPersistentReductionOuterNormBwdLikeFloat256x7x512Scheduler) {
  grid_persistent_reduction_outer_norm_bwd_like_scheduler(
      256, 7, 512, DataType::Float);
}

TEST_F(
    OuterReductionTest,
    GridPersistentReductionOuterNormBwdLikeFloat256x14x512Scheduler) {
  grid_persistent_reduction_outer_norm_bwd_like_scheduler(
      256, 14, 512, DataType::Float);
}

TEST_F(
    OuterReductionTest,
    GridPersistentReductionOuterNormBwdLikeFloat256x28x512Scheduler) {
  grid_persistent_reduction_outer_norm_bwd_like_scheduler(
      256, 28, 512, DataType::Float);
}

namespace {

// Backward batchnorm
void grid_persistent_batchnorm_bwd_scheduler(
    int64_t N,
    int64_t HW,
    int64_t C,
    DataType dtype) {
  const bool benchmark_mode = isBenchmarkMode();
  std::unique_ptr<Fusion> fusion_ptr = std::make_unique<Fusion>();
  Fusion& fusion = *fusion_ptr.get();
  FusionGuard fg(fusion_ptr.get());

  const bool kTraining = true;
  const float kEps = 1e-5;

  // setup fusion
  auto input = makeContigTensor(4, dtype);
  auto grad_output = makeContigTensor(4, dtype);
  auto weight = makeContigTensor(1, DataType::Float);
  auto running_mean = makeContigTensor(1, DataType::Float);
  auto running_var = makeContigTensor(1, DataType::Float);
  auto save_mean = makeContigTensor(1, DataType::Float);
  auto save_var = makeContigTensor(1, DataType::Float);

  fusion.addInput(input);
  fusion.addInput(grad_output);
  fusion.addInput(weight);
  fusion.addInput(running_mean);
  fusion.addInput(running_var);
  fusion.addInput(save_mean);
  fusion.addInput(save_var);

  if (dtype == DataType::Half) {
    input = castOp(DataType::Float, input);
    grad_output = castOp(DataType::Float, grad_output);
  }

  auto eps_ptr = IrBuilder::create<Val>(kEps);

  auto result = batch_norm_backward(
      input,
      grad_output,
      weight,
      running_mean,
      running_var,
      save_mean,
      save_var,
      kTraining,
      eps_ptr,
      std::vector<bool>(3, true),
      true);

  auto grad_input = result.grad_input;
  auto grad_weight = result.grad_weight;
  auto grad_bias = result.grad_bias;

  if (dtype == DataType::Half) {
    grad_input = castOp(DataType::Half, grad_input);
    grad_weight = castOp(DataType::Half, grad_weight);
    grad_bias = castOp(DataType::Half, grad_bias);
  }

  fusion.addOutput(grad_input);
  fusion.addOutput(grad_weight);
  fusion.addOutput(grad_bias);

  auto options_float =
      at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  auto options =
      at::TensorOptions().dtype(data_type_to_aten(dtype)).device(at::kCUDA, 0);

  const std::vector<int64_t> input_shape{N, HW, HW, C};

  auto at_input = at::randn({N, C, HW, HW}, options)
                      .contiguous(c10::MemoryFormat::ChannelsLast);
  auto at_input_nvfuser = at_input.clone().detach().permute({0, 2, 3, 1});

  auto at_grad_out = at::randn({N, C, HW, HW}, options)
                         .contiguous(c10::MemoryFormat::ChannelsLast);
  auto at_grad_out_nvfuser = at_grad_out.clone().detach().permute({0, 2, 3, 1});

  at::Tensor at_weight = at::ones({C}, options_float);
  at::Tensor at_run_mean = at::zeros({C}, options_float);
  at::Tensor at_run_var = at::ones({C}, options_float);
  at::Tensor at_save_mean = at::zeros({C}, options_float);
  at::Tensor at_save_var = at::ones({C}, options_float);

  KernelArgumentHolder args = {
      at_input_nvfuser,
      at_grad_out_nvfuser,
      at_weight,
      at_run_mean,
      at_run_var,
      at_save_mean,
      at_save_var};

  FusionExecutorCache executor_cache(std::move(fusion_ptr));
  auto cg_outputs = executor_cache.runFusionWithInputs(args);

  auto runtime = executor_cache.getMostRecentKernelRuntime();

  if (!shouldBePersistent(N, HW, dtype, true, true, DataType::Float)) {
    NVF_CHECK(runtime->isSegmented(), "Expected to be segmented");
  } else {
    NVF_CHECK(
        !runtime->isSegmented(),
        "Unexpected number of segments: ",
        runtime->fusionSegments()->groups().size());

    const auto& heuristic_params =
        runtime->schedulerHeuristics()->heuristicsList().at(0);
    NVF_CHECK(
        heuristic_params->scheduler_type == SchedulerType::OuterPersistent,
        "Unexpected heuristic was chosen: ",
        heuristic_params->scheduler_type);

    if (benchmark_mode) {
      for (int i = 0; i < 10; ++i) {
        clearL2Cache();
        cg_outputs = executor_cache.runFusionWithInputs(args);
      }
    }
  }

  // Permute grad_input output
  cg_outputs[0] = cg_outputs[0].as<at::Tensor>().permute({0, 3, 1, 2});

  auto at_output = at::native_batch_norm_backward(
      at_grad_out,
      at_input,
      at_weight,
      at_run_mean,
      at_run_var,
      at_save_mean,
      at_save_var,
      true,
      kEps,
      {true, true, true});

  testValidate(
      &fusion,
      cg_outputs,
      args,
      {std::get<0>(at_output), std::get<1>(at_output), std::get<2>(at_output)},
      __LINE__,
      __FILE__,
      "");
}

} // namespace

TEST_F(
    OuterReductionTest,
    GridPersistentBatchNormChannelsLastBwdHalf256x7x512Scheduler) {
  grid_persistent_batchnorm_bwd_scheduler(256, 7, 512, DataType::Half);
}

TEST_F(
    OuterReductionTest,
    GridPersistentBatchNormChannelsLastBwdHalf256x14x512Scheduler) {
  grid_persistent_batchnorm_bwd_scheduler(256, 14, 512, DataType::Half);
}

TEST_F(
    OuterReductionTest,
    GridPersistentBatchNormChannelsLastBwdHalf256x28x512Scheduler) {
  grid_persistent_batchnorm_bwd_scheduler(256, 28, 512, DataType::Half);
}

TEST_F(
    OuterReductionTest,
    GridPersistentBatchNormChannelsLastBwdFloat256x7x512Scheduler) {
  grid_persistent_batchnorm_bwd_scheduler(256, 7, 512, DataType::Float);
}

TEST_F(
    OuterReductionTest,
    GridPersistentBatchNormChannelsLastBwdFloat256x14x512Scheduler) {
  grid_persistent_batchnorm_bwd_scheduler(256, 14, 512, DataType::Float);
}

TEST_F(
    OuterReductionTest,
    GridPersistentBatchNormChannelsLastBwdFloat256x28x512Scheduler) {
  grid_persistent_batchnorm_bwd_scheduler(256, 28, 512, DataType::Float);
}
#endif

TEST_F(OuterReductionTest, IterGroupedBlockReduction) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  DataType dtype = DataType::Half;

  auto tv0 = makeContigTensor(2, dtype);
  fusion.addInput(tv0);

  auto tv1 = castOp(DataType::Float, tv0);
  auto tv2 = sum(tv1, {0});
  auto tv3 = castOp(dtype, tv2);
  fusion.addOutput(tv3);

  std::vector<int64_t> shape({32, 2048});

  auto options =
      at::TensorOptions().dtype(data_type_to_aten(dtype)).device(at::kCUDA, 0);
  auto t0 = at::randn(shape, options);

  SchedulerRuntimeInfo runtime_info(&fusion, {t0});
  NVF_ERROR(
      Schedule::canSchedule(SchedulerType::Reduction, &fusion, runtime_info));
  auto scheduler =
      SchedulerEntry::makeSchedulerInstance(SchedulerType::Reduction);
  auto heuristic_params = scheduler->computeHeuristics(&fusion, runtime_info);
  auto rparams = heuristic_params->as<ReductionParams>();

  // only do block reduction, enforce vectorization so we can group them
  const int vect_factor = 8;
  rparams->cross_grid_inner_reduction = false;
  rparams->split_grid_dim_inner_reduction = false;
  rparams->vectorize_iter_dom = true;
  rparams->unroll_factor_iter_dom = vect_factor;

  scheduler->schedule(&fusion, rparams);
  KernelExecutor ke;
  ke.compile(&fusion, {t0}, heuristic_params->lparams);
  auto cg_outputs = ke.run({t0}, {}, heuristic_params->lparams);

  // lowering & check iteration grouped reductions
  NVF_CHECK(
      ke.compiledKernel()->kernel()->summary().has_iter_grouped_reductions,
      "There must be iter domain grouped reductions.");
  NVF_CHECK(
      ke.compiledKernel()->kernel()->summary().num_grouped_iterations ==
          vect_factor,
      "Expected ",
      vect_factor,
      " grouped iterations, found ",
      ke.compiledKernel()->kernel()->summary().num_grouped_iterations);

  testValidate(
      &fusion,
      cg_outputs,
      {t0},
      {t0.to(at::kFloat).sum(0)},
      __LINE__,
      __FILE__,
      "",
      rparams->lparams);
}

namespace {
void shmooTestsOfIterGroupedBlockOrGridReduction(
    DataType dtype,
    int vect,
    int bdimx,
    int gdimx,
    int unroll,
    int bdimy,
    int gdimy,
    int serial,
    bool is_grid_reduciton) {
  NVF_CHECK(
      is_grid_reduciton ^ (gdimy == 1),
      "Grid reduction should have gdimy > 1, block reduction should have gdimy "
      "== 1.");
  int iter_dim = vect * bdimx * gdimx;
  int redu_dim = unroll * bdimy * gdimy * serial;

  Fusion fusion;
  FusionGuard fg(&fusion);
  auto tv0 = makeContigTensor(2, dtype);
  fusion.addInput(tv0);
  auto tv1 = set(tv0);
  if (dtype == DataType::Half) {
    tv1 = castOp(DataType::Float, tv1);
  }
  auto tv2 = sum(tv1, {0});
  if (dtype == DataType::Half) {
    tv2 = castOp(DataType::Half, tv2);
  }
  fusion.addOutput(tv2);

  // manually set how to schedule the fusion
  auto rparams = std::make_unique<ReductionParams>();
  // vectorize
  rparams->vectorize_iter_dom = true;
  rparams->unroll_factor_iter_dom = vect;
  // use bdimx
  rparams->multiple_reds_per_blk = true;
  rparams->block_dim_iter_dom = ParallelType::TIDx;
  // use gdimx
  rparams->grid_dim_iter_dom = ParallelType::BIDx;
  // use unroll
  rparams->unroll_factor_inner_reduction = unroll;
  // use bdimy
  rparams->cross_block_inner_reduction = true;
  rparams->block_dim_inner_reduction = ParallelType::TIDy;
  // use gdimy
  rparams->cross_grid_inner_reduction = is_grid_reduciton;
  rparams->split_grid_dim_inner_reduction = is_grid_reduciton;
  if (is_grid_reduciton) {
    rparams->grid_dim_inner_reduction = ParallelType::BIDy;
  }
  // set launch para
  auto lparams = LaunchParams(
      gdimx,
      is_grid_reduciton ? gdimy : LaunchParams::UNINITIALIZED_VAL,
      LaunchParams::UNINITIALIZED_VAL,
      bdimx,
      bdimy,
      LaunchParams::UNINITIALIZED_VAL);
  rparams->lparams = lparams;
  SchedulerEntry::makeSchedulerInstance(SchedulerType::Reduction)
      ->schedule(&fusion, rparams.get());

  // lowering & check iteration grouped reductions
  GpuLower gpulw(&fusion);
  gpulw.run();
  NVF_CHECK(
      gpulw.kernel()->summary().has_iter_grouped_reductions,
      "There must be iter domain grouped reductions.");
  NVF_CHECK(
      gpulw.kernel()->summary().num_grouped_iterations == vect,
      "Expected ",
      vect,
      " grouped iterations, found ",
      gpulw.kernel()->summary().num_grouped_iterations);

  std::vector<int64_t> shape({redu_dim, iter_dim});
  auto options =
      at::TensorOptions().dtype(data_type_to_aten(dtype)).device(at::kCUDA, 0);
  auto t0 = at::randn(shape, options);
  KernelArgumentHolder inputs({t0});

  KernelExecutor ke;
  ke.compile(&fusion, inputs, lparams);
  auto cg_outputs = ke.run(inputs, {}, lparams);

  testValidate(
      &fusion,
      cg_outputs,
      inputs,
      {t0.to(at::kFloat).sum(0)},
      __LINE__,
      __FILE__,
      "",
      lparams);
};
} // namespace

TEST_F(OuterReductionTest, IterGroupedBlockReductionShmooTests) {
  // Shmoo over data types and 6 heuristic paras in iter grouped block
  // reduction. To avoid too many tests, gdimx, unroll, and serial are tested
  // with just 1 value since they are not directly related to block reduction.
  // vect, max vectorization factor is 8
  std::vector<int> vect_list({2, 4, 8});
  // bdimx, pow of 2 and a prime number
  std::vector<int> bdimx_list({8, 32, 64, 128, 256, 512, 127});
  // gdimx, blocks in x-dim are processing different reductions, no inference
  // with each other, just test a small set
  std::vector<int> gdimx_list({8});
  // unroll, current heuristic may use up to 16, but not directly related to
  // block reduction, just test a small set.
  std::vector<int> unroll_list({2});
  // bdimy, frequently used values and a large prime number
  std::vector<int> bdimy_list({1, 2, 3, 4, 8, 16, 32, 64, 128, 127});
  // serial iteration, may use any value, but not directly related to
  // grid reduction, just test a small set.
  std::vector<int> serial_list({4});
  auto properties = at::cuda::getDeviceProperties(
      c10::Device(c10::DeviceType::CUDA, 0).index());
  int max_threads_per_blk = (int)properties->maxThreadsPerBlock;
  const int gdimy = 1;
  const bool is_grid_reduciton = false;
  for (DataType dtype : {DataType::Half, DataType::Float}) {
    for (int vect : vect_list) {
      if (dtype == DataType::Float && vect == 8) {
        continue;
      }
      for (int bdimx : bdimx_list) {
        for (int gdimx : gdimx_list) {
          for (int unroll : unroll_list) {
            for (int bdimy : bdimy_list) {
              for (int serial : serial_list) {
                // skip invalid configures
                if (bdimx * bdimy > max_threads_per_blk) {
                  continue;
                }
                shmooTestsOfIterGroupedBlockOrGridReduction(
                    dtype,
                    vect,
                    bdimx,
                    gdimx,
                    unroll,
                    bdimy,
                    gdimy,
                    serial,
                    is_grid_reduciton);
              }
            }
          }
        }
      }
    }
  }
}

TEST_F(OuterReductionTest, IterGroupedGridReductionShmooTests) {
  // Shmoo over data types and 7 heuristic paras in iter grouped grid
  // reduction. To avoid too many tests, gdimx, unroll, and serial are tested
  // with just 1 value since they are not directly related to block or grid
  // reduction.

  // vect, max vectorization factor is 8
  std::vector<int> vect_list({2, 4, 8});
  // bdimx, pow of 2 and a prime number
  std::vector<int> bdimx_list({8, 64, 127});
  // gdimx, blocks in x-dim are processing different reductions, no inference
  // with each other, just test a small set
  std::vector<int> gdimx_list({8});
  // unroll, current heuristic may use up to 16, but not directly related to
  // grid reduction, just test a small set.
  std::vector<int> unroll_list({2});
  // bdimy
  std::vector<int> bdimy_list({8, 64, 128, 127});
  // gdimy, may use any value >= 2, power of 2 is more frequently used
  std::vector<int> gdimy_list({2, 3, 8, 64, 128});
  // serial iteration, may use any value, but not directly related to
  // grid reduction, just test a small set.
  std::vector<int> serial_list({4});
  auto properties = at::cuda::getDeviceProperties(
      c10::Device(c10::DeviceType::CUDA, 0).index());
  int max_threads_per_blk = (int)properties->maxThreadsPerBlock;
  const bool is_grid_reduciton = true;
  for (DataType dtype : {DataType::Half, DataType::Float}) {
    for (int vect : vect_list) {
      if (dtype == DataType::Float && vect == 8) {
        continue;
      }
      for (int bdimx : bdimx_list) {
        for (int gdimx : gdimx_list) {
          for (int unroll : unroll_list) {
            for (int bdimy : bdimy_list) {
              for (int gdimy : gdimy_list) {
                for (int serial : serial_list) {
                  // skip invalid configures
                  if (bdimx * bdimy > max_threads_per_blk) {
                    continue;
                  }
                  shmooTestsOfIterGroupedBlockOrGridReduction(
                      dtype,
                      vect,
                      bdimx,
                      gdimx,
                      unroll,
                      bdimy,
                      gdimy,
                      serial,
                      is_grid_reduciton);
                }
              }
            }
          }
        }
      }
    }
  }
}

// validation tests of outer reduction scheduler after
// adding IterGroupedGridReduction
TEST_F(OuterReductionTest, OuterReductionMagicScheduler) {
  auto test = [](int dim0, int dim1) {
    auto fusion = std::make_unique<Fusion>();
    FusionGuard fg(fusion.get());
    DataType dtype = DataType::Half;
    auto tv0 = makeContigTensor(2, dtype);
    fusion->addInput(tv0);
    auto tv1 = castOp(DataType::Float, tv0);
    auto tv2 = sum(tv1, {0});
    auto tv3 = castOp(dtype, tv2);
    fusion->addOutput(tv3);

    std::vector<int64_t> shape({dim0, dim1});
    auto options = at::TensorOptions()
                       .dtype(data_type_to_aten(dtype))
                       .device(at::kCUDA, 0);
    auto t0 = at::randn(shape, options);
    KernelArgumentHolder args = {t0};
    FusionExecutorCache executor_cache(std::move(fusion));
    auto cg_outputs = executor_cache.runFusionWithInputs(args);
    auto ref = t0.to(at::kFloat).sum({0});
    testValidate(
        executor_cache.fusion(), cg_outputs, args, {ref}, __LINE__, __FILE__);
  };

  maybeClearAllocator(0);
  for (int dim0 = 1024; dim0 <= 32768; dim0 *= 2) {
    for (int dim1 = 1024; dim1 <= 32768; dim1 *= 2) {
      test(dim0, dim1);
    }
  }
}

TEST_F(OuterReductionTest, IterGroupedMultipleReductions) {
  std::unique_ptr<Fusion> fusion_ptr = std::make_unique<Fusion>();
  Fusion& fusion = *fusion_ptr.get();
  FusionGuard fg(&fusion);

  auto tv0 = makeContigTensor(2);
  auto tv1 = makeContigTensor(2);
  fusion.addInput(tv0);
  fusion.addInput(tv1);
  auto tv2 = sum(tv0, {0});
  auto tv3 = add(tv0, tv1);
  auto tv4 = sum(tv3, {0});
  fusion.addOutput(tv2);
  fusion.addOutput(tv4);

  int vect = 4, unroll = 2;
  int bdimx = 32, bdimy = 16;
  int gdimx = 8, gdimy = 8;
  int serial = 2;
  int iter_dim = vect * bdimx * gdimx;
  int redu_dim = unroll * bdimy * gdimy * serial;

  // manually set how to schedule the fusion
  auto rparams = std::make_shared<ReductionParams>();
  // vectorize
  rparams->vectorize_iter_dom = true;
  rparams->unroll_factor_iter_dom = vect;
  // use bdimx
  rparams->multiple_reds_per_blk = true;
  rparams->block_dim_iter_dom = ParallelType::TIDx;
  // use gdimx
  rparams->grid_dim_iter_dom = ParallelType::BIDx;
  // use unroll
  rparams->unroll_factor_inner_reduction = unroll;
  // use bdimy
  rparams->cross_block_inner_reduction = true;
  rparams->block_dim_inner_reduction = ParallelType::TIDy;
  // use gdimy
  rparams->cross_grid_inner_reduction = true;
  rparams->split_grid_dim_inner_reduction = true;
  rparams->grid_dim_inner_reduction = ParallelType::BIDy;
  // set launch para
  auto lparams = LaunchParams(
      gdimx,
      gdimy,
      LaunchParams::UNINITIALIZED_VAL,
      bdimx,
      bdimy,
      LaunchParams::UNINITIALIZED_VAL);
  rparams->lparams = lparams;
  SchedulerEntry::makeSchedulerInstance(SchedulerType::Reduction)
      ->schedule(&fusion, rparams.get());

  // Ensure we have two iteration grouped reductions
  int num_iter_grouped_reductions = 0;
  const auto& reduction_tvs =
      scheduler_utils::getReductionTvs(fusion_ptr.get());
  for (auto tv : reduction_tvs) {
    bool has_grid_reduction = false;
    bool has_grouped_domain = false;
    for (auto id : tv->getLoopDomain()) {
      if (id->isReduction() && id->getParallelType() == ParallelType::BIDy) {
        has_grid_reduction = true;
      }
      if (id->getParallelType() == ParallelType::Group) {
        has_grouped_domain = true;
      }
    }
    if (has_grid_reduction) {
      EXPECT_TRUE(has_grouped_domain)
          << "Expect Iteration domain grouped grid reduction, tv: "
          << tv->toString();
    }
    if (has_grid_reduction && has_grouped_domain) {
      num_iter_grouped_reductions++;
    }
  }
  EXPECT_TRUE(num_iter_grouped_reductions == 2)
      << "Expect 2 Iteration domain grouped grid reductions, got: "
      << num_iter_grouped_reductions;

  KernelExecutor ke;
  std::vector<int64_t> shape({redu_dim, iter_dim});
  auto options = at::TensorOptions().device(at::kCUDA, 0);
  auto t0 = at::randn(shape, options);
  auto t1 = at::randn(shape, options);
  KernelArgumentHolder inputs({t0, t1});

  ke.compile(&fusion, inputs, lparams);
  auto cg_outputs = ke.run(inputs, {}, lparams);

  testValidate(
      &fusion,
      cg_outputs,
      inputs,
      {t0.sum(0), (t0 + t1).sum(0)},
      __LINE__,
      __FILE__,
      "",
      lparams);
}

// Repro of https://github.com/NVIDIA/Fuser/pull/2766
TEST_F(OuterReductionTest, SmallOuterBlockReductionIssue2766) {
  std::unique_ptr<Fusion> fusion_ptr = std::make_unique<Fusion>();
  auto& fusion = *fusion_ptr;
  FusionGuard fg(&fusion);

  std::vector<int64_t> shape{100, 2, 128};

  auto tv0 = makeContigTensor(2);
  fusion.addInput(tv0);

  auto tv1 = reshape(
      tv0,
      {IrBuilder::create<Val>(shape[0]),
       IrBuilder::create<Val>(shape[1]),
       IrBuilder::create<Val>(shape[2])});
  auto tv2 = sum(tv1, {1});
  fusion.addOutput(tv2);

  // Previously, after the extent replacement of the lowering, the reduction
  // reference tensor got a reduction domain of a static size, which is just 1,
  // but the pre-reshape tensors still kept using symbolic extents. Before
  // https://github.com/NVIDIA/Fuser/pull/2714, the scheduler decided to not use
  // TIDy because the reference tensor has a static size of 1, but since the
  // other tensors still had dynamic sizes, it resulted in the dynamic
  // allocation error.

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  auto t0 = at::randn({shape[0] * shape[1], shape[2]}, options);
  KernelArgumentHolder args = {t0};

  FusionExecutorCache executor_cache(std::move(fusion_ptr));
  auto outputs = executor_cache.runFusionWithInputs(args);

  testValidate(executor_cache.fusion(), outputs, args, __LINE__, __FILE__);
}

TEST_F(OuterReductionTest, SimpleThreadLocalSerialReduction) {
  auto fusion_ptr = std::make_unique<Fusion>();
  FusionGuard fg(fusion_ptr.get());
  Fusion& fusion = *fusion_ptr;

  std::vector<int64_t> shape{28, 8192, 128};

  auto T0 = makeContigConcreteTensor(shape, DataType::BFloat16);
  fusion.addInput(T0);
  auto T1 = castOp(DataType::Float, T0);
  auto T2 = sum(T1, {0});
  fusion.addOutput(T2);

  auto fusion_copy = fusion;

  auto options = at::TensorOptions().dtype(at::kBFloat16).device(at::kCUDA, 0);
  auto at_t0 = at::randn(shape, options);
  KernelArgumentHolder args = {at_t0};
  FusionExecutorCache executor_cache(std::move(fusion_ptr));
  auto outputs = executor_cache.runFusionWithInputs(args);

  // If thread local reduction is used on the tested GPU, the reduction tv
  // should be:  [..., rS{7}, iV{x}, rUS{1}, rUR{x}]
  auto runtime = executor_cache.getMostRecentKernelRuntime();
  for (auto& params : runtime->schedulerHeuristics()->heuristicsList()) {
    if (!params->isA<ReductionParams>()) {
      continue;
    }
    if (!params->as<ReductionParams>()->cross_block_inner_reduction) {
      Fusion* scheduled_fusion = runtime->executors()
                                     .back()
                                     ->as<KernelExecutor>()
                                     ->compiledKernel()
                                     ->kernel();
      auto redu_tv = scheduler_utils::getReductionTvs(scheduled_fusion).at(0);
      EXPECT_TRUE(redu_tv->axis(-4)->isReduction())
          << "Expected redu tv is [..., rS{7}, iV{x}, rUS{1}, rUR{x}], got: "
          << redu_tv->toString();
    }
  }

  testValidate(&fusion_copy, outputs, {at_t0}, __LINE__, __FILE__);
}

} // namespace nvfuser
