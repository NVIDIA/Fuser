// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#include <gmock/gmock-matchers.h>
#include <gtest/gtest.h>

#include <executor.h>
#include <ir_all_nodes.h>
#include <ir_builder.h>
#include <kernel_cache.h>
#include <ops/all_ops.h>
#include <scheduler/all_schedulers.h>

#include <test/utils.h>
#include <test/validator.h>

#include <torch/torch.h>

namespace nvfuser {

class IndexingOpTest : public NVFuserTest {};

namespace {
auto randomVector(int64_t low, int64_t high, int rank) {
  std::vector<int64_t> out(rank, 0);
  for (int idim = 0; idim < rank; ++idim) {
    out[idim] = (std::rand() % (high - low)) + low;
  }
  return out;
}

// When take_along_axis is true, the extents of non-indexed dimensions
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

TEST_F(IndexingOpTest, Scatter1DIndexZerosSelfTvSameShape_CUDA) {
  const std::vector<std::vector<int64_t>> input_dims = {{2, 2}};

  const std::vector<std::vector<int64_t>> src_dims = {{2, 2}};

  const std::vector<std::vector<int64_t>> idx_dims = {{2, 2}};

  at::manual_seed(0);
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
    at::Tensor input = at::randn(input_dims[test_id], options);
    at::Tensor src = at::randn(src_dims[test_id], options);
    auto t_index = at::add(idx_1, idx_2);
    auto out_ref = at::scatter(input, 0, t_index, src);

    std::vector<c10::IValue> aten_inputs = {input, idx_1, idx_2, src};

    FusionExecutorCache executor_cache(std::move(fusion_ptr));
    auto cg_outputs = executor_cache.runFusionWithInputs(aten_inputs);
    testValidate(
        &fusion, cg_outputs, aten_inputs, {out_ref}, __LINE__, __FILE__);
  }
}

// all torch.gather test follow the FusionTorchGather* pattern

// Test the correctness of gather operator in different dimensions and selcted
// dim.
TEST_F(IndexingOpTest, TorchGatherAllRankAllSelectedDim_CUDA) {
  const int max_dim_size = 64;
  std::srand(0);
  at::manual_seed(0);
  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  auto options_i = at::TensorOptions().dtype(at::kLong).device(at::kCUDA, 0);
  for (const auto is_take_along : {false, true}) {
    for (int rank = 1; rank <= 5; ++rank) {
      for (int dim = 0; dim < rank; ++dim) {
        auto fusion_ptr = std::make_unique<Fusion>();
        Fusion& fusion = *fusion_ptr.get();
        FusionGuard fg(&fusion);

        TensorView* tv1 = makeContigTensor(rank);
        TensorView* tv_idx = makeContigTensor(rank, DataType::Int);
        fusion.addInput(tv1);
        fusion.addInput(tv_idx);
        TensorView* tv_out = is_take_along ? take_along_axis(tv1, tv_idx, dim)
                                           : torch_gather(tv1, dim, tv_idx);
        fusion.addOutput(tv_out);

        auto input_dims = randomVector(2, max_dim_size, rank);
        auto index_dims =
            randomIndexVector(input_dims, 1, rank, is_take_along, dim);
        at::Tensor input = at::randn(input_dims, options);
        at::Tensor input_idx =
            at::randint(0, input_dims[dim], index_dims, options_i);
        at::Tensor output = at::zeros(index_dims, options);

        auto tv_out_ref = at::gather(input, dim, input_idx);
        std::vector<c10::IValue> aten_inputs = {input, input_idx};

        FusionExecutorCache executor_cache(std::move(fusion_ptr));
        auto cg_outputs = executor_cache.runFusionWithInputs(aten_inputs);
        testValidate(
            &fusion, cg_outputs, aten_inputs, {tv_out_ref}, __LINE__, __FILE__);
      }
    }
  }
}
// Test the fusion support of gather operator(producer) and elemetwise(consumer)
TEST_F(IndexingOpTest, TorchGatherAddMul_CUDA) {
  const int max_dim_size = 64;
  std::srand(0);
  at::manual_seed(0);
  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  auto options_i = at::TensorOptions().dtype(at::kLong).device(at::kCUDA, 0);
  for (const auto is_take_along : {false, true}) {
    for (int rank = 1; rank <= 5; ++rank) {
      for (int dim = 0; dim < rank; ++dim) {
        auto fusion_ptr = std::make_unique<Fusion>();
        Fusion& fusion = *fusion_ptr.get();
        FusionGuard fg(&fusion);

        TensorView* tv1 = makeContigTensor(rank);
        TensorView* tv_idx = makeContigTensor(rank, DataType::Int);
        fusion.addInput(tv1);
        fusion.addInput(tv_idx);
        auto tv_gather = is_take_along ? take_along_axis(tv1, tv_idx, dim)
                                       : torch_gather(tv1, dim, tv_idx);
        auto tv_add = add(tv_gather, tv_gather);
        auto tv_out = mul(tv_gather, tv_add);
        fusion.addOutput(tv_out);

        auto input_dims = randomVector(2, max_dim_size, rank);
        auto index_dims =
            randomIndexVector(input_dims, 1, rank, is_take_along, dim);

        at::Tensor input = at::randn(input_dims, options); // lookup
        at::Tensor input_idx =
            at::randint(0, input_dims[dim], index_dims, options_i);
        at::Tensor output = at::zeros(index_dims, options);

        auto t_gather = at::gather(input, dim, input_idx);
        auto t_add = at::add(t_gather, t_gather);
        auto tv_out_ref = at::mul(t_gather, t_add);

        std::vector<c10::IValue> aten_inputs = {input, input_idx};

        FusionExecutorCache executor_cache(std::move(fusion_ptr));
        auto cg_outputs = executor_cache.runFusionWithInputs(aten_inputs);
        testValidate(
            &fusion, cg_outputs, aten_inputs, {tv_out_ref}, __LINE__, __FILE__);
      }
    }
  }
}
// Test the fusion support of index tensor as fusion input in gather operator
TEST_F(IndexingOpTest, AddGatherSumAdd_CUDA) {
  const int max_dim_size = 8;
  std::srand(0);
  at::manual_seed(0);
  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  auto options_i = at::TensorOptions().dtype(at::kLong).device(at::kCUDA, 0);
  for (const auto is_take_along : {false, true}) {
    for (int rank = 2; rank <= 5; ++rank) {
      for (int dim = 0; dim < rank; ++dim) {
        auto fusion_ptr = std::make_unique<Fusion>();
        Fusion& fusion = *fusion_ptr.get();
        FusionGuard fg(&fusion);

        TensorView* tv_lookup = makeContigTensor(rank);
        TensorView* tv_idx_1 = makeContigTensor(rank, DataType::Int);
        TensorView* tv_idx_2 = makeContigTensor(rank, DataType::Int);

        fusion.addInput(tv_lookup);
        fusion.addInput(tv_idx_1);
        fusion.addInput(tv_idx_2);

        auto tv_index = add(tv_idx_1, tv_idx_2);
        auto tv_out = is_take_along ? take_along_axis(tv_lookup, tv_index, dim)
                                    : torch_gather(tv_lookup, dim, tv_index);

        fusion.addOutput(tv_out);

        auto input_dims = randomVector(2, max_dim_size, rank);
        auto index_dims =
            randomIndexVector(input_dims, 1, rank, is_take_along, dim);

        at::Tensor t_lookup = at::randn(input_dims, options); // lookup
        at::Tensor t_idx_1 =
            at::randint(0, input_dims[dim] / 2, index_dims, options_i);
        at::Tensor t_idx_2 =
            at::randint(0, input_dims[dim] / 2, index_dims, options_i);

        auto t_index = at::add(t_idx_1, t_idx_2);
        auto t_out = at::gather(t_lookup, dim, t_index);

        std::vector<c10::IValue> aten_inputs = {t_lookup, t_idx_1, t_idx_2};
        FusionExecutorCache executor_cache(std::move(fusion_ptr));
        auto cg_outputs = executor_cache.runFusionWithInputs(aten_inputs);
        testValidate(
            &fusion, cg_outputs, aten_inputs, {t_out}, __LINE__, __FILE__);
      }
    }
  }
}
// Test the fusion support of gather operator and reduce
TEST_F(IndexingOpTest, TorchGatherSumAdd_CUDA) {
  const int max_dim_size = 64;
  std::srand(0);
  at::manual_seed(0);
  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  auto options_i = at::TensorOptions().dtype(at::kLong).device(at::kCUDA, 0);
  for (const auto is_take_along : {false, true}) {
    for (int rank = 2; rank <= 5; ++rank) {
      for (int dim = 0; dim < rank; ++dim) {
        auto fusion_ptr = std::make_unique<Fusion>();
        Fusion& fusion = *fusion_ptr.get();
        FusionGuard fg(&fusion);

        TensorView* tv1 = makeContigTensor(rank);
        TensorView* tv_idx = makeContigTensor(rank, DataType::Int);
        TensorView* tv2 = makeContigTensor(rank - 1);

        fusion.addInput(tv1);
        fusion.addInput(tv_idx);
        fusion.addInput(tv2);

        auto tv_gather = is_take_along ? take_along_axis(tv1, tv_idx, dim)
                                       : torch_gather(tv1, dim, tv_idx);
        auto tv_sum = sum(tv_gather, {0}, true);
        auto tv_out = add(tv_sum, tv2);

        fusion.addOutput(tv_out);

        auto input_dims = randomVector(2, max_dim_size, rank);
        auto index_dims =
            randomIndexVector(input_dims, 1, rank, is_take_along, dim);
        std::vector<int64_t> input2_dims(rank - 1, 0);
        for (int idim = 0; idim < rank - 1; ++idim) {
          input2_dims[idim] = index_dims[idim + 1];
        }

        at::Tensor input = at::randn(input_dims, options); // lookup
        at::Tensor input2 = at::randn(input2_dims, options); // lookup
        at::Tensor input_idx =
            at::randint(0, input_dims[dim], index_dims, options_i);
        at::Tensor output = at::zeros(index_dims, options);

        auto t_gather = at::gather(input, dim, input_idx);
        auto t_sum = at::sum(t_gather.to(at::kDouble), {0}, true);
        auto tv_out_ref = at::add(input2.to(at::kDouble), t_sum);

        std::vector<c10::IValue> aten_inputs = {input, input_idx, input2};

        FusionExecutorCache executor_cache(std::move(fusion_ptr));
        auto cg_outputs = executor_cache.runFusionWithInputs(aten_inputs);
        testValidate(
            &fusion, cg_outputs, aten_inputs, {tv_out_ref}, __LINE__, __FILE__);
      }
    }
  }
}
// Test the correctness when input/index tensor is very large
TEST_F(IndexingOpTest, TorchGatherAddMulHugeSize_CUDA) {
  const int max_dim_size = 16384;
  std::srand(0);
  at::manual_seed(0);
  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  auto options_i = at::TensorOptions().dtype(at::kLong).device(at::kCUDA, 0);
  for (const auto is_take_along : {false, true}) {
    for (int rank = 1; rank <= 2; ++rank) {
      for (int dim = 0; dim < rank; ++dim) {
        auto fusion_ptr = std::make_unique<Fusion>();
        Fusion& fusion = *fusion_ptr.get();
        FusionGuard fg(&fusion);

        TensorView* tv1 = makeContigTensor(rank);
        TensorView* tv_idx = makeContigTensor(rank, DataType::Int);

        fusion.addInput(tv1);
        fusion.addInput(tv_idx);
        auto tv_gather = is_take_along ? take_along_axis(tv1, tv_idx, dim)
                                       : torch_gather(tv1, dim, tv_idx);
        auto tv_add = add(tv_gather, tv_gather);
        auto tv_out = mul(tv_gather, tv_add);
        fusion.addOutput(tv_out);

        auto input_dims = randomVector(2, max_dim_size, rank);
        auto index_dims =
            randomIndexVector(input_dims, 1, rank, is_take_along, dim);

        at::Tensor input = at::randn(input_dims, options); // lookup
        at::Tensor input_idx =
            at::randint(0, input_dims[dim], index_dims, options_i);
        at::Tensor output = at::zeros(index_dims, options);

        auto t_gather = at::gather(input, dim, input_idx);
        auto t_add = at::add(t_gather, t_gather);
        auto tv_out_ref = at::mul(t_gather, t_add);

        std::vector<c10::IValue> aten_inputs = {input, input_idx};

        FusionExecutorCache executor_cache(std::move(fusion_ptr));
        auto cg_outputs = executor_cache.runFusionWithInputs(aten_inputs);
        testValidate(
            &fusion, cg_outputs, aten_inputs, {tv_out_ref}, __LINE__, __FILE__);
      }
    }
  }
}
// Test the fusion support of input tensor as fusion input
TEST_F(IndexingOpTest, TorchGatherInput_CUDA) {
  const int rank = 2;

  auto fusion_ptr = std::make_unique<Fusion>();
  Fusion& fusion = *fusion_ptr.get();
  FusionGuard fg(&fusion);

  TensorView* tv1 = makeContigTensor(rank);
  TensorView* tv_idx = makeContigTensor(rank, DataType::Int);
  fusion.addInput(tv1);
  fusion.addInput(tv_idx);

  auto tv_inp = add(tv1, tv1);
  auto tv_gather = torch_gather(tv_inp, 0, tv_idx);
  fusion.addOutput(tv_gather);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  auto options_i = at::TensorOptions().dtype(at::kLong).device(at::kCUDA, 0);
  at::Tensor t1 = at::randn({5, 5}, options);
  at::Tensor t_idx = at::randint(0, 5, {5, 5}, options_i);

  FusionExecutorCache executor_cache(std::move(fusion_ptr));
  auto cg_outputs = executor_cache.runFusionWithInputs({t1, t_idx});
}

// Test when then extent of iteration domain is euqal to one, and the iteration
// type is broadcast (IndexTv), used in RGCN model.
TEST_F(IndexingOpTest, TorchGatherIndexTvExtentIsOne_CUDA) {
  std::vector<int64_t> input_dims{16384, 60};
  std::vector<int64_t> index_dims{16384, 1};
  const int max_selected_index = 60;

  at::manual_seed(0);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  auto options_i = at::TensorOptions().dtype(at::kLong).device(at::kCUDA, 0);

  auto fusion_ptr = std::make_unique<Fusion>();
  Fusion& fusion = *fusion_ptr.get();
  FusionGuard fg(&fusion);

  TensorView* tv_in1 = makeConcreteTensor(input_dims);
  TensorView* tv_idx = makeConcreteTensor(index_dims, DataType::Int);
  TensorView* tv_in2 = makeConcreteTensor(index_dims);
  fusion.addInput(tv_in1);
  fusion.addInput(tv_idx);
  fusion.addInput(tv_in2);

  auto tv_gather = torch_gather(tv_in1, 1, tv_idx);
  auto tv_add =
      clamp(tv_gather, IrBuilder::create<Int>(-1), IrBuilder::create<Int>(1));
  auto tv_out = mul(tv_add, tv_in2);
  fusion.addOutput(tv_out);

  at::Tensor input_1 = at::randn(input_dims, options);
  at::Tensor input_2 = at::randn(index_dims, options);
  at::Tensor input_idx =
      at::randint(0, max_selected_index, index_dims, options_i);
  at::Tensor output = at::zeros(index_dims, options);

  auto t_gather = at::gather(input_1, 1, input_idx);
  auto t_add = at::clamp(t_gather, -1, 1);
  auto tv_out_ref = at::mul(input_2, t_add);

  std::vector<c10::IValue> aten_inputs = {input_1, input_idx, input_2};

  FusionExecutorCache executor_cache(std::move(fusion_ptr));
  auto cg_outputs = executor_cache.runFusionWithInputs(aten_inputs);
  testValidate(
      &fusion, cg_outputs, aten_inputs, {tv_out_ref}, __LINE__, __FILE__);
}

// Test take_along_axis with a broadcast index tensor
TEST_F(IndexingOpTest, TakeAlongBroadcastIndex_CUDA) {
  for (const auto index_dim : {1, 3}) {
    auto fusion_ptr = std::make_unique<Fusion>();
    Fusion& fusion = *fusion_ptr.get();
    FusionGuard fg(&fusion);

    auto tv0 = makeSymbolicTensor(3);
    auto tv1 =
        makeConcreteTensor({index_dim == 1 ? index_dim : -1}, DataType::Int);
    auto tv2 = makeConcreteTensor({-1, index_dim == 1 ? index_dim : -1, -1});
    fusion.addInput(tv0);
    fusion.addInput(tv1);
    fusion.addInput(tv2);

    auto tv3 = broadcast(tv1, {true, false, true});
    auto tv4 = take_along_axis(tv0, tv3, 1);
    auto tv5 = add(tv4, tv2);
    fusion.addOutput(tv5);

    std::vector<int64_t> input_dims{10, 11, 12};
    std::vector<int64_t> index_dims{index_dim};
    std::vector<int64_t> out_dims = input_dims;
    out_dims[1] = index_dims[0];

    at::manual_seed(0);
    auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
    auto options_i = at::TensorOptions().dtype(at::kLong).device(at::kCUDA, 0);
    at::Tensor t0 = at::randn(input_dims, options);
    at::Tensor t1 = at::randint(0, input_dims[1], index_dims, options_i);
    at::Tensor t2 = at::randn(out_dims, options);
    std::vector<c10::IValue> aten_inputs = {t0, t1, t2};

    FusionExecutorCache executor_cache(std::move(fusion_ptr));
    auto cg_outputs = executor_cache.runFusionWithInputs(aten_inputs);

    auto t4 = at::gather(t0, 1, t1.unsqueeze(0).unsqueeze(-1).expand(out_dims));
    auto ref = t4 + t2;

    testValidate(&fusion, cg_outputs, aten_inputs, {ref}, __LINE__, __FILE__);
  }
}

// Disabled for now as it fails likely due to
// https://github.com/NVIDIA/Fuser/issues/93
#if 0
TEST_F(IndexingOpTest, TakeAlongBroadcastInput_CUDA) {
  for (const auto inp_indexed_dim : {1, 11}) {
    for (const auto idx_index_dim : {1, 3}) {
      // [B, B, I] when inp_indexed_dim == 1, otherwise [B, I, I]
      std::vector<int64_t> input_dims{1, inp_indexed_dim, 12};
      // [I, B] when idx_index_dim == 1, otherwise [I, I]
      std::vector<int64_t> index_dims{5, idx_index_dim};
      // This needs to match with the take_along_axis output
      std::vector<int64_t> out_dims{
          index_dims.at(0), index_dims.at(1), input_dims.at(2)};

      auto fusion_ptr = std::make_unique<Fusion>();
      Fusion& fusion = *fusion_ptr.get();
      FusionGuard fg(&fusion);

      auto tv0 = makeConcreteTensor({1, inp_indexed_dim == 1 ? 1 : -1, -1});
      auto tv1 =
          makeConcreteTensor({-1, idx_index_dim == 1 ? 1 : -1}, DataType::Int);
      auto tv2 =
          makeConcreteTensor({-1, idx_index_dim == 1 ? idx_index_dim : -1, -1});
      fusion.addInput(tv0);
      fusion.addInput(tv1);
      fusion.addInput(tv2);

      auto tv3 = broadcast(tv1, {false, false, true});
      auto tv4 = take_along_axis(tv0, tv3, 1);
      auto tv5 = add(tv4, tv2);
      fusion.addOutput(tv5);

      at::manual_seed(0);
      auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
      auto options_i =
          at::TensorOptions().dtype(at::kLong).device(at::kCUDA, 0);
      at::Tensor t0 = at::randn(input_dims, options);
      at::Tensor t1 = at::randint(0, input_dims[1], index_dims, options_i);
      at::Tensor t2 = at::randn(out_dims, options);
      std::vector<c10::IValue> aten_inputs = {t0, t1, t2};

      FusionExecutorCache executor_cache(std::move(fusion_ptr));
      auto cg_outputs = executor_cache.runFusionWithInputs(aten_inputs);

      auto t4 =
          at::gather(t0, 1, t1.unsqueeze(0).unsqueeze(-1).expand(out_dims));
      auto ref = t4 + t2;

      testValidate(&fusion, cg_outputs, aten_inputs, {ref}, __LINE__, __FILE__);
    }
  }
}
#endif

} // namespace nvfuser
