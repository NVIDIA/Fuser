// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#include <gmock/gmock-matchers.h>
#include <gtest/gtest.h>

#include <torch/torch.h>

#include <exceptions.h>
#include <ir/all_nodes.h>
#include <ir/builder.h>
#include <ops/all_ops.h>
#include <runtime/executor.h>
#include <runtime/fusion_executor_cache.h>
#include <scheduler/all_schedulers.h>
#include <scheduler/tools/inlining.h>
#include <tests/cpp/utils.h>
#include <tests/cpp/validator.h>

namespace nvfuser {

class ScatterTest : public NVFuserTest {
 protected:
  void SetUp() override {
    // To make the tests using std::rand deterministic
    std::srand(0);
  }
};

namespace {
auto randomVector(int64_t low, int64_t high, int rank) {
  std::vector<int64_t> out(rank, 0);
  for (int idim = 0; idim < rank; ++idim) {
    out[idim] = (std::rand() % (high - low)) + low;
  }
  return out;
}

// When takeAlongAxis is true, the extents of non-indexed dimensions
// are set to be the same as those of the input dimensions
auto randomIndexVector(
    const std::vector<int64_t>& input_dims,
    int64_t low,
    int rank,
    bool take_along_axis = false,
    int indexed_dim = -1) {
  std::vector<int64_t> index_dims(rank, 0);
  for (int idim = 0; idim < rank; ++idim) {
    if (!take_along_axis || idim == indexed_dim) {
      index_dims[idim] = (std::rand() % (input_dims[idim] - low)) + low;
    } else {
      index_dims[idim] = input_dims.at(idim);
    }
  }
  return index_dims;
}

at::Tensor generateScatter2DIndex(
    int64_t min,
    int64_t extent_1d,
    int64_t extent_2d,
    int select_id) {
  auto options_i =
      torch::TensorOptions().dtype(torch::kLong).device(at::kCUDA, 0);
  if (select_id == 0) {
    auto idx = at::randint(0, extent_2d, {extent_1d, extent_2d}, options_i);
    for (int64_t i = 0; i < extent_1d; ++i) {
      idx[i] = at::randperm(extent_2d, options_i) + min;
    }
    return idx.transpose(0, 1).contiguous();
  } else {
    auto idx = at::randint(0, extent_1d, {extent_2d, extent_1d}, options_i);
    for (int64_t i = 0; i < extent_2d; ++i) {
      idx[i] = at::randperm(extent_1d, options_i) + min;
    }
    return idx;
  }
}

} // namespace

TEST_F(ScatterTest, Scatter1DIndexZerosSelfTvSameShape) {
  const std::vector<std::vector<int64_t>> input_dims = {{2, 2}};

  const std::vector<std::vector<int64_t>> src_dims = {{2, 2}};

  const std::vector<std::vector<int64_t>> idx_dims = {{2, 2}};

  for (size_t test_id = 0; test_id < idx_dims.size(); ++test_id) {
    auto fusion_ptr = std::make_unique<Fusion>();
    Fusion& fusion = *fusion_ptr.get();
    FusionGuard fg(&fusion);

    TensorView* tv_input = makeContigTensor(2);
    TensorView* tv_idx_1 = makeContigTensor(2, DataType::Int);
    TensorView* tv_idx_2 = makeContigTensor(2, DataType::Int);
    TensorView* tv_src = makeContigTensor(2);

    fusion.addInput(tv_input);
    fusion.addInput(tv_idx_1);
    fusion.addInput(tv_idx_2);
    fusion.addInput(tv_src);

    auto tv_idx = add(tv_idx_1, tv_idx_2);
    auto tv_out = scatter(tv_input, 0, tv_idx, tv_src);
    fusion.addOutput(tv_out);

    auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
    auto options_i =
        torch::TensorOptions().dtype(torch::kLong).device(at::kCUDA, 0);

    at::Tensor idx = generateScatter2DIndex(
        0, idx_dims[test_id][1], idx_dims[test_id][0], 0);

    at::Tensor idx_1 = at::randint(0, 24, idx_dims[test_id], options_i);
    at::Tensor idx_2 = idx - idx_1;
    at::Tensor t0 = at::randn(input_dims[test_id], options);
    at::Tensor src = at::randn(src_dims[test_id], options);

    FusionExecutorCache executor_cache(std::move(fusion_ptr));
    auto cg_outputs =
        executor_cache.runFusionWithInputs({t0, idx_1, idx_2, src});
    testValidate(
        &fusion, cg_outputs, {t0, idx_1, idx_2, src}, __LINE__, __FILE__);
  }
}

} // namespace nvfuser
