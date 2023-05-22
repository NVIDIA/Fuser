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
#include <inlining.h>
#include <ir/all_nodes.h>
#include <ir/builder.h>
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
  const int max_dim_size = 32;
  std::srand(0);
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

    auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
    auto options_i = at::TensorOptions().dtype(at::kLong).device(at::kCUDA, 0);
    at::Tensor t0 = at::randn(input_dims, options);
    at::Tensor t1 = at::randint(0, input_dims[1], index_dims, options_i);
    at::Tensor t2 = at::randn(out_dims, options);
    std::vector<c10::IValue> aten_inputs = {t0, t1, t2};

    FusionExecutorCache executor_cache(std::move(fusion_ptr));
    auto cg_outputs = executor_cache.runFusionWithInputs(aten_inputs);

    auto t4 = at::take_along_dim(
        t0, t1.unsqueeze(0).unsqueeze(-1).expand(out_dims), 1);
    auto ref = t4 + t2;

    testValidate(&fusion, cg_outputs, aten_inputs, {ref}, __LINE__, __FILE__);
  }
}

TEST_F(IndexingOpTest, GatherBroadcastInput_CUDA) {
  for (const auto is_take_along : {false, true}) {
    // torch_gather not supported yet. The issue is one of the index
    // tensor has a broadcast domain, but its corresponding input
    // domain is a normal domain. The output domain is also a
    // broadcast in torch_gather, whereas it's a normal domain in
    // take_along_axis. In the case of torch_gather, indexing the
    // input domain needs to be able to index the normal producer
    // domain with a broadcast reference domain. getProduerIndex needs
    // some fix.
    if (!is_take_along) {
      continue;
    }
    for (const auto inp_indexed_dim : {1, 11}) {
      for (const auto idx_index_dim : {1, 3}) {
        // [B, B, I] when inp_indexed_dim == 1, otherwise [B, I, I]
        std::vector<int64_t> input_dims{1, inp_indexed_dim, 12};
        // [I, B] when idx_index_dim == 1, otherwise [I, I]
        // In torch_gather, an index dimension must be smaller or
        // equal to the corresponding input dimension
        std::vector<int64_t> index_dims{
            is_take_along ? 5 : input_dims.at(0), idx_index_dim};
        // This needs to match with the take_along_axis output
        std::vector<int64_t> out_dims{
            index_dims.at(0), index_dims.at(1), input_dims.at(2)};

        auto fusion_ptr = std::make_unique<Fusion>();
        Fusion& fusion = *fusion_ptr.get();
        FusionGuard fg(&fusion);

        auto tv0 = makeSymbolicTensor(input_dims);
        auto tv1 = makeSymbolicTensor(index_dims, DataType::Int);
        auto tv2 = makeSymbolicTensor(out_dims);
        fusion.addInput(tv0);
        fusion.addInput(tv1);
        fusion.addInput(tv2);

        auto tv3 = broadcast(tv1, {false, false, true});
        auto tv4 = take_along_axis(tv0, tv3, 1);
        auto tv5 = add(tv4, tv2);
        fusion.addOutput(tv5);

        auto options =
            at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
        auto options_i =
            at::TensorOptions().dtype(at::kLong).device(at::kCUDA, 0);
        at::Tensor t0 = at::randn(input_dims, options);
        at::Tensor t1 = at::randint(0, input_dims[1], index_dims, options_i);
        at::Tensor t2 = at::randn(out_dims, options);
        std::vector<c10::IValue> aten_inputs = {t0, t1, t2};

        FusionExecutorCache executor_cache(std::move(fusion_ptr));
        auto cg_outputs = executor_cache.runFusionWithInputs(aten_inputs);

        auto t4 = is_take_along ? at::take_along_dim(t0, t1.unsqueeze(-1), 1)
                                : at::gather(t0, 1, t1.unsqueeze(-1));
        auto ref = t4 + t2;

        testValidate(
            &fusion, cg_outputs, aten_inputs, {ref}, __LINE__, __FILE__);
      }
    }
  }
}

// Test take_along_axis with non fusion inputs
TEST_F(IndexingOpTest, TakeAlongAxisIntermediateTensorPointwise1_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  const std::vector<int64_t> shape({99, 101});

  auto tv0 = makeSymbolicTensor(2);
  fusion.addInput(tv0);
  auto tv1 = makeSymbolicTensor(1, DataType::Int);
  fusion.addInput(tv1);

  auto tv2 = add(tv0, IrBuilder::create<Double>(1));
  auto tv3 = broadcast(tv1, {false, true});
  auto tv4 = take_along_axis(tv2, tv3, 1);
  fusion.addOutput(tv4);

  scheduler_utils::prepareForMemoryTypePromotion(&fusion);

  // Test if this split is propagated through the indexed domain
  tv4->split(1, 10);

  TransformPropagator propagator(tv4);
  MaxRootDomainInfoSpanningTree(tv4).traverse(&propagator);

  // All of the tensors should have the split by 2, except for tv1.
  for (auto tv : ir_utils::allTvsExcept(&fusion, {tv1})) {
    TORCH_CHECK(tv->nDims() == 3, "Unexpected tensor: ", tv->toString());
    TORCH_CHECK(
        tv->axis(-1)->definition() &&
            tv->axis(-1)->definition()->isA<Split>() &&
            tv->axis(-1)->definition()->as<Split>()->in() ==
                tv->getMaybeRFactorDomain().at(1),
        "Unexpected tensor: ",
        tv->toString());
  }

  // This should not inline the indexed producer domain. Note that the
  // producer tensor of the take_along_axis expr is not tv2 as a copy
  // is inserted
  inlineMost();
  auto take_along_axis_input =
      tv4->definition()->as<TorchGatherOp>()->lookupTv();
  TORCH_CHECK(
      take_along_axis_input->getComputeAtPosition() == 1,
      "Unexpected computeAt position: ",
      take_along_axis_input->toString());

  // Test if parallelization is propagated through indexed domains
  tv4->axis(-2)->parallelize(ParallelType::TIDy);
  tv4->axis(-1)->parallelize(ParallelType::TIDx);
  scheduler_utils::parallelizeAllLike(tv4);

  for (auto tv : ir_utils::allTvs(&fusion)) {
    if (tv->isFusionInput()) {
      continue;
    }
    TORCH_CHECK(
        tv->axis(-2)->getParallelType() == tv4->axis(-2)->getParallelType() &&
            tv->axis(-1)->getParallelType() == tv4->axis(-1)->getParallelType(),
        "Unexpected parallelization of tensor: ",
        tv->toString());
  }

  // This should make the producer of take_along_axis saved in shared memory
  scheduler_utils::promoteProducerMemoryTypes(&fusion, {});

  TORCH_CHECK(
      take_along_axis_input->getMemoryType() == MemoryType::Shared,
      "Failed to promote memory type: ",
      take_along_axis_input->toString());

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  auto options_i = at::TensorOptions().dtype(at::kLong).device(at::kCUDA, 0);
  auto t0 = at::randn(shape, options);
  auto t1 = at::randint(0, shape[1], {shape[0]}, options_i);
  std::vector<c10::IValue> aten_inputs = {t0, t1};

  FusionExecutor fe;
  fe.compileFusion(&fusion, aten_inputs);

  auto outputs = fe.runFusion(aten_inputs);

  auto ref = at::take_along_dim(t0 + 1, t1.unsqueeze(-1), 1);

  testValidate(&fusion, outputs, aten_inputs, {ref}, __LINE__, __FILE__);
}

// Same as the above but with the pointwise scheduler
TEST_F(IndexingOpTest, TakeAlongAxisIntermediateTensorPointwise2_CUDA) {
  auto fusion_ptr = std::make_unique<Fusion>();
  Fusion& fusion = *fusion_ptr.get();
  FusionGuard fg(&fusion);

  std::vector<int64_t> shape({99, 101});

  auto tv0 = makeSymbolicTensor(2);
  fusion.addInput(tv0);
  auto tv1 = makeSymbolicTensor(1, DataType::Int);
  fusion.addInput(tv1);

  auto tv2 = add(tv0, IrBuilder::create<Double>(1));
  auto tv3 = broadcast(tv1, {false, true});
  auto tv4 = take_along_axis(tv2, tv3, 1);
  fusion.addOutput(tv4);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  auto options_i = at::TensorOptions().dtype(at::kLong).device(at::kCUDA, 0);
  auto t0 = at::randn(shape, options);
  auto t1 = at::randint(0, shape[1], {shape[0]}, options_i);
  std::vector<c10::IValue> aten_inputs = {t0, t1};

  FusionExecutorCache fec(std::move(fusion_ptr));
  auto outputs = fec.runFusionWithInputs(aten_inputs);

  validateSegmentation(
      fec.getMostRecentKernelRuntime(), {ScheduleHeuristic::PointWise});

  auto ref = at::take_along_dim(t0 + 1, t1.unsqueeze(-1), 1);

  testValidate(&fusion, outputs, aten_inputs, {ref}, __LINE__, __FILE__);
}

// Reduction then take_along_axis. This is currently segmented due to
// the post-reduction rule as documented in
// https://github.com/NVIDIA/Fuser/issues/260
TEST_F(IndexingOpTest, TakeAlongAxisIntermediateTensorReduction1_CUDA) {
  auto fusion_ptr = std::make_unique<Fusion>();
  Fusion& fusion = *fusion_ptr.get();
  FusionGuard fg(&fusion);

  std::vector<int64_t> shape({100, 1000});

  auto tv0 = makeSymbolicTensor(2);
  fusion.addInput(tv0);
  auto tv1 = makeSymbolicTensor(1, DataType::Int);
  fusion.addInput(tv1);

  auto tv2 = sum(tv0, {1});
  auto tv4 = take_along_axis(tv2, tv1, 0);
  fusion.addOutput(tv4);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  auto options_i = at::TensorOptions().dtype(at::kLong).device(at::kCUDA, 0);
  auto t0 = at::randn(shape, options);
  auto t1 = at::randint(0, shape[0], {2}, options_i);
  std::vector<c10::IValue> aten_inputs = {t0, t1};

  FusionExecutorCache fec(std::move(fusion_ptr));
  auto outputs = fec.runFusionWithInputs(aten_inputs);

  validateSegmentation(
      fec.getMostRecentKernelRuntime(),
      {ScheduleHeuristic::Reduction, ScheduleHeuristic::PointWise});

  auto ref = at::take_along_dim(t0.to(at::kDouble).sum({1}), t1, 0);

  testValidate(&fusion, outputs, aten_inputs, {ref}, __LINE__, __FILE__);
}

// take_along_axis to broadcast, squeeze, then reduction. Segmented
// before the reduction
TEST_F(IndexingOpTest, TakeAlongAxisIntermediateTensorReduction2_CUDA) {
  auto fusion_ptr = std::make_unique<Fusion>();
  Fusion& fusion = *fusion_ptr.get();
  FusionGuard fg(&fusion);

  std::vector<int64_t> shape({100, 100});

  auto tv0 = makeSymbolicTensor(2);
  fusion.addInput(tv0);
  auto tv1 = makeSymbolicTensor(1, DataType::Int);
  fusion.addInput(tv1);

  auto tv2 = add(tv0, IrBuilder::create<Double>(1));
  auto tv3 = broadcast(tv1, {false, true});
  auto tv4 = take_along_axis(tv2, tv3, 1);
  auto tv5 = squeeze(tv4, std::vector<bool>{false, true});
  auto tv6 = sum(tv5, {0});
  fusion.addOutput(tv6);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  auto options_i = at::TensorOptions().dtype(at::kLong).device(at::kCUDA, 0);
  auto t0 = at::randn(shape, options);
  auto t1 = at::randint(0, shape[1], {shape[0]}, options_i);
  std::vector<c10::IValue> aten_inputs = {t0, t1};

  FusionExecutorCache fec(std::move(fusion_ptr));
  auto outputs = fec.runFusionWithInputs(aten_inputs);

  validateSegmentation(
      fec.getMostRecentKernelRuntime(),
      {ScheduleHeuristic::PointWise, ScheduleHeuristic::Reduction});

  auto t4 = at::take_along_dim(t0.to(at::kDouble) + 1, t1.unsqueeze(-1), 1);
  auto ref = t4.squeeze(1).sum({0});

  testValidate(&fusion, outputs, aten_inputs, {ref}, __LINE__, __FILE__);
}

// take_along_axis then reduction. Should not be segmented.
TEST_F(IndexingOpTest, TakeAlongAxisIntermediateTensorReduction3_CUDA) {
  auto fusion_ptr = std::make_unique<Fusion>();
  Fusion& fusion = *fusion_ptr.get();
  FusionGuard fg(&fusion);

  std::vector<int64_t> shape_before_gather({100, 100});
  std::vector<int64_t> shape_after_gather({shape_before_gather[0], 120});

  auto tv0 = makeSymbolicTensor(2);
  fusion.addInput(tv0);
  auto tv1 = makeSymbolicTensor(2, DataType::Int);
  fusion.addInput(tv1);

  auto tv2 = add(tv0, IrBuilder::create<Double>(1));
  auto tv3 = take_along_axis(tv2, tv1, 1);
  auto tv4 = sum(tv3, {1});
  fusion.addOutput(tv4);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  auto options_i = at::TensorOptions().dtype(at::kLong).device(at::kCUDA, 0);
  auto t0 = at::randn(shape_before_gather, options);
  auto t1 =
      at::randint(0, shape_before_gather[1], shape_after_gather, options_i);
  std::vector<c10::IValue> aten_inputs = {t0, t1};

  FusionExecutorCache fec(std::move(fusion_ptr));
  auto outputs = fec.runFusionWithInputs(aten_inputs);

  validateSegmentation(
      fec.getMostRecentKernelRuntime(), {ScheduleHeuristic::Reduction});

  auto ref = at::take_along_dim(t0.to(at::kDouble) + 1, t1, 1).sum({1});

  testValidate(&fusion, outputs, aten_inputs, {ref}, __LINE__, __FILE__);
}

// Similar to TakeAlongAxisIntermediateTensorReduction2, but no
// squeeze of the consumer ID of the indexed domain. Should not be segmented.
TEST_F(IndexingOpTest, TakeAlongAxisIntermediateTensorReduction4_CUDA) {
  GTEST_SKIP() << "Disabled due to a bug. See #292";
  auto fusion_ptr = std::make_unique<Fusion>();
  Fusion& fusion = *fusion_ptr.get();
  FusionGuard fg(&fusion);

  std::vector<int64_t> shape_before_gather({10, 10});
  std::vector<int64_t> shape_after_gather({shape_before_gather[0]});

  auto tv0 = makeSymbolicTensor(2);
  fusion.addInput(tv0);
  auto tv1 = makeSymbolicTensor(1, DataType::Int);
  fusion.addInput(tv1);

  auto tv2 = add(tv0, IrBuilder::create<Double>(1));
  auto tv3 = broadcast(tv1, {false, true});
  auto tv4 = take_along_axis(tv2, tv3, 1);
  auto tv5 = sum(tv4, {0});
  // TODO: remove this. Currently, validation fails without this
  // likely because of a predication bug
  auto tv6 = set(tv5);
  fusion.addOutput(tv6);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  auto options_i = at::TensorOptions().dtype(at::kLong).device(at::kCUDA, 0);
  auto t0 = at::randn(shape_before_gather, options);
  auto t1 =
      at::randint(0, shape_before_gather[1], shape_after_gather, options_i);
  std::vector<c10::IValue> aten_inputs = {t0, t1};

  FusionExecutorCache fec(std::move(fusion_ptr));
  auto outputs = fec.runFusionWithInputs(aten_inputs);

  validateSegmentation(
      fec.getMostRecentKernelRuntime(), {ScheduleHeuristic::Reduction});

  auto ref =
      at::take_along_dim(t0.to(at::kDouble) + 1, t1.unsqueeze(-1), 1).sum({0});

  testValidate(&fusion, outputs, aten_inputs, {ref}, __LINE__, __FILE__);
}

// Normalization then take_along_axis
TEST_F(IndexingOpTest, TakeAlongAxisIntermediateTensorNormalization1_CUDA) {
  auto fusion_ptr = std::make_unique<Fusion>();
  Fusion& fusion = *fusion_ptr.get();
  FusionGuard fg(&fusion);

  std::vector<int64_t> shape({32, 1024});

  auto tv0 = makeSymbolicTensor(2);
  fusion.addInput(tv0);
  auto tv1 = makeSymbolicTensor(1, DataType::Int);
  fusion.addInput(tv1);

  auto tv2 = sum(tv0, {1});
  auto tv3 = broadcast(tv2, {false, true});
  auto tv4 = div(tv0, tv3);
  auto tv5 = broadcast(tv1, {false, true});
  auto tv6 = take_along_axis(tv4, tv5, 1);
  fusion.addOutput(tv6);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  auto options_i = at::TensorOptions().dtype(at::kLong).device(at::kCUDA, 0);
  auto t0 = at::randn(shape, options);
  auto t1 = at::randint(0, shape[1], {shape[0]}, options_i);
  std::vector<c10::IValue> aten_inputs = {t0, t1};

  FusionExecutorCache fec(std::move(fusion_ptr));
  auto outputs = fec.runFusionWithInputs(aten_inputs);

  validateSegmentation(
      fec.getMostRecentKernelRuntime(), {ScheduleHeuristic::Persistent});

  auto t0_d = t0.to(at::kDouble);
  auto ref = at::take_along_dim(
      t0_d / t0_d.sum({1}).unsqueeze(-1), t1.unsqueeze(-1), 1);

  testValidate(&fusion, outputs, aten_inputs, {ref}, __LINE__, __FILE__);
}

// take_along_dim to broadcast, squeeze, then normalization. Segmented
// as the input dim to take_along_dim cannot be scheduled by the
// reduction tv
TEST_F(IndexingOpTest, TakeAlongAxisIntermediateTensorNormalization2_CUDA) {
  auto fusion_ptr = std::make_unique<Fusion>();
  Fusion& fusion = *fusion_ptr.get();
  FusionGuard fg(&fusion);

  std::vector<int64_t> shape({32, 1024});

  auto tv0 = makeSymbolicTensor(2);
  fusion.addInput(tv0);
  auto tv1 = makeSymbolicTensor(1, DataType::Int);
  fusion.addInput(tv1);

  auto tv2 = add(tv0, IrBuilder::create<Double>(1));
  auto tv3 = broadcast(tv1, {false, true});
  auto tv4 = take_along_axis(tv2, tv3, 1);
  auto tv5 = squeeze(tv4, std::vector<bool>{false, true});
  auto tv6 = sum(tv5, {0});
  auto tv7 = broadcast(tv6, {true});
  auto tv8 = div(tv5, tv7);
  fusion.addOutput(tv8);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  auto options_i = at::TensorOptions().dtype(at::kLong).device(at::kCUDA, 0);
  auto t0 = at::randn(shape, options);
  auto t1 = at::randint(0, shape[1], {shape[0]}, options_i);
  std::vector<c10::IValue> aten_inputs = {t0, t1};

  FusionExecutorCache fec(std::move(fusion_ptr));
  auto outputs = fec.runFusionWithInputs(aten_inputs);

  validateSegmentation(
      fec.getMostRecentKernelRuntime(),
      {ScheduleHeuristic::PointWise, ScheduleHeuristic::Persistent});

  auto t5 = at::take_along_dim(t0.to(at::kDouble) + 1, t1.unsqueeze(-1), 1)
                .squeeze(1);
  auto ref = t5 / t5.sum({0}).unsqueeze(0);

  testValidate(&fusion, outputs, aten_inputs, {ref}, __LINE__, __FILE__);
}

// take_along_axis then normalization. Should not be segmented.
TEST_F(IndexingOpTest, TakeAlongAxisIntermediateTensorNormalization3_CUDA) {
  auto fusion_ptr = std::make_unique<Fusion>();
  Fusion& fusion = *fusion_ptr.get();
  FusionGuard fg(&fusion);

  std::vector<int64_t> shape_before_gather({100, 100});
  std::vector<int64_t> shape_after_gather({shape_before_gather[0], 120});

  auto tv0 = makeSymbolicTensor(2);
  fusion.addInput(tv0);
  auto tv1 = makeSymbolicTensor(2, DataType::Int);
  fusion.addInput(tv1);

  auto tv2 = add(tv0, IrBuilder::create<Double>(1));
  auto tv3 = take_along_axis(tv2, tv1, 1);
  auto tv4 = sum(tv3, {1});
  auto tv5 = broadcast(tv4, {false, true});
  auto tv6 = div(tv3, tv5);
  fusion.addOutput(tv6);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  auto options_i = at::TensorOptions().dtype(at::kLong).device(at::kCUDA, 0);
  auto t0 = at::randn(shape_before_gather, options);
  auto t1 =
      at::randint(0, shape_before_gather[1], shape_after_gather, options_i);
  std::vector<c10::IValue> aten_inputs = {t0, t1};

  FusionExecutorCache fec(std::move(fusion_ptr));
  auto outputs = fec.runFusionWithInputs(aten_inputs);

  validateSegmentation(
      fec.getMostRecentKernelRuntime(), {ScheduleHeuristic::Persistent});

  auto t3 = at::take_along_dim(t0.to(at::kDouble) + 1, t1, 1);
  auto ref = t3 / t3.sum({1}).unsqueeze(-1);

  testValidate(&fusion, outputs, aten_inputs, {ref}, __LINE__, __FILE__);
}

// Normalization, then take_along_axis, then reduction. Similar
// pattern as cross entropy.
TEST_F(
    IndexingOpTest,
    TakeAlongAxisIntermediateTensorNormalizationAndReduction1_CUDA) {
  auto fusion_ptr = std::make_unique<Fusion>();
  Fusion& fusion = *fusion_ptr.get();
  FusionGuard fg(&fusion);

  std::vector<int64_t> shape({32, 1024});

  auto tv0 = makeSymbolicTensor(2);
  fusion.addInput(tv0);
  auto tv1 = makeSymbolicTensor(2, DataType::Int);
  fusion.addInput(tv1);

  auto tv2 = sum(tv0, {1});
  auto tv3 = broadcast(tv2, {false, true});
  auto tv4 = div(tv0, tv3);
  auto tv5 = take_along_axis(tv4, tv1, 1);
  auto tv6 = sum(tv5, {0, 1});
  fusion.addOutput(tv6);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  auto options_i = at::TensorOptions().dtype(at::kLong).device(at::kCUDA, 0);
  auto t0 = at::randn(shape, options);
  auto t1 = at::randint(0, shape[1], {shape[0], 1}, options_i);
  std::vector<c10::IValue> aten_inputs = {t0, t1};

  FusionExecutorCache fec(std::move(fusion_ptr));
  auto outputs = fec.runFusionWithInputs(aten_inputs);

  // The reduction patterns of the normalization and the final
  // reduction are different, so they are segmented out
  validateSegmentation(
      fec.getMostRecentKernelRuntime(),
      {ScheduleHeuristic::Persistent, ScheduleHeuristic::Reduction});

  auto t0_d = t0.to(at::kDouble);
  auto t5 = at::take_along_dim(t0_d / t0_d.sum({1}).unsqueeze(-1), t1, 1);
  auto ref = t5.sum();

  testValidate(&fusion, outputs, aten_inputs, {ref}, __LINE__, __FILE__);
}

// Similar to
// TakeAlongAxisIntermediateTensorNormalizationAndReduction1, but the
// final reduction pattern is compatible with the first reduction, so
// no segmentation should be done
TEST_F(
    IndexingOpTest,
    TakeAlongAxisIntermediateTensorNormalizationAndReduction2_CUDA) {
  auto fusion_ptr = std::make_unique<Fusion>();
  Fusion& fusion = *fusion_ptr.get();
  FusionGuard fg(&fusion);

  std::vector<int64_t> shape({32, 1024});

  auto tv0 = makeSymbolicTensor(2);
  fusion.addInput(tv0);
  auto tv1 = makeSymbolicTensor(1, DataType::Int);
  fusion.addInput(tv1);

  auto tv2 = sum(tv0, {1});
  auto tv3 = broadcast(tv2, {false, true});
  auto tv4 = div(tv0, tv3);
  auto tv5 = broadcast(tv1, {false, true});
  auto tv6 = take_along_axis(tv4, tv5, 1);
  auto tv7 = add(tv0, tv6);
  auto tv8 = sum(tv7, {1});
  fusion.addOutput(tv8);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  auto options_i = at::TensorOptions().dtype(at::kLong).device(at::kCUDA, 0);
  auto t0 = at::randn(shape, options);
  auto t1 = at::randint(0, shape[1], {shape[0]}, options_i);
  std::vector<c10::IValue> aten_inputs = {t0, t1};

  FusionExecutorCache fec(std::move(fusion_ptr));
  auto outputs = fec.runFusionWithInputs(aten_inputs);

  validateSegmentation(
      fec.getMostRecentKernelRuntime(), {ScheduleHeuristic::Persistent});

  auto t0_d = t0.to(at::kDouble);
  auto t6 = at::take_along_dim(
      t0_d / t0_d.sum({1}).unsqueeze(-1), t1.unsqueeze(-1), 1);
  auto ref = (t0_d + t6).sum({1});

  testValidate(&fusion, outputs, aten_inputs, {ref}, __LINE__, __FILE__);
}

// take_along_axis then transpose
TEST_F(IndexingOpTest, TakeAlongAxisIntermediateTensorTranspose1_CUDA) {
  auto fusion_ptr = std::make_unique<Fusion>();
  Fusion& fusion = *fusion_ptr.get();
  FusionGuard fg(&fusion);

  // Make sure the shape is large enough to trigger the Transpose
  // scheduler. See also getTransposeRuntimeRejectReason for more details.
  std::vector<int64_t> shape(
      {deviceSMCount(),
       TransposeParams::getDefaultTileSize(),
       TransposeParams::getDefaultTileSize()});

  auto tv0 = makeSymbolicTensor(3);
  fusion.addInput(tv0);
  auto tv1 = makeSymbolicTensor(2, DataType::Int);
  fusion.addInput(tv1);

  auto tv2 = add(tv0, IrBuilder::create<Double>(1));
  auto tv3 = broadcast(tv1, {true, false, false});
  auto tv4 = take_along_axis(tv2, tv3, 0);
  auto tv5 = transpose(tv4, 1, 2);
  fusion.addOutput(tv5);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  auto options_i = at::TensorOptions().dtype(at::kLong).device(at::kCUDA, 0);
  auto t0 = at::randn(shape, options);
  auto t1 = at::randint(0, shape[0], {shape[1], shape[2]}, options_i);
  std::vector<c10::IValue> aten_inputs = {t0, t1};

  FusionExecutorCache fec(std::move(fusion_ptr));
  auto outputs = fec.runFusionWithInputs(aten_inputs);

  validateSegmentation(
      fec.getMostRecentKernelRuntime(), {ScheduleHeuristic::Transpose});

  auto ref = at::take_along_dim(t0 + 1, t1.unsqueeze(0), 0).transpose(1, 2);

  testValidate(&fusion, outputs, aten_inputs, {ref}, __LINE__, __FILE__);
}

// transpose then take_along_axis. Currently failed to pick the
// Transpose scheduler due to a limitation of the analysis for the
// scheduler. See DomainMap::findReferenceFor in transpose.cpp for
// more details.
TEST_F(IndexingOpTest, TakeAlongAxisIntermediateTensorTranspose2_CUDA) {
  auto fusion_ptr = std::make_unique<Fusion>();
  Fusion& fusion = *fusion_ptr.get();
  FusionGuard fg(&fusion);

  // Make sure the shape is large enough to trigger the Transpose
  // scheduler. See also getTransposeRuntimeRejectReason for more details.
  std::vector<int64_t> shape(
      {deviceSMCount(),
       TransposeParams::getDefaultTileSize(),
       TransposeParams::getDefaultTileSize()});

  auto tv0 = makeSymbolicTensor(3);
  fusion.addInput(tv0);
  auto tv1 = makeSymbolicTensor(3, DataType::Int);
  fusion.addInput(tv1);

  auto tv2 = transpose(tv0, 1, 2);
  auto tv4 = take_along_axis(tv2, tv1, 0);
  fusion.addOutput(tv4);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  auto options_i = at::TensorOptions().dtype(at::kLong).device(at::kCUDA, 0);
  auto t0 = at::randn(shape, options);
  auto t1 = at::randint(0, shape[0], {10, shape[2], shape[1]}, options_i);
  std::vector<c10::IValue> aten_inputs = {t0, t1};

  FusionExecutorCache fec(std::move(fusion_ptr));
  auto outputs = fec.runFusionWithInputs(aten_inputs);

  validateSegmentation(
      fec.getMostRecentKernelRuntime(), {ScheduleHeuristic::PointWise});

  auto ref = at::take_along_dim(t0.transpose(1, 2), t1, 0);

  testValidate(&fusion, outputs, aten_inputs, {ref}, __LINE__, __FILE__);
}

// transpose the dimension produced by take_along_axis. Currently not
// supported by the transpose scheduler
TEST_F(IndexingOpTest, TakeAlongAxisIntermediateTensorTranspose3_CUDA) {
  auto fusion_ptr = std::make_unique<Fusion>();
  Fusion& fusion = *fusion_ptr.get();
  FusionGuard fg(&fusion);

  std::vector<int64_t> shape_before(
      {deviceSMCount(),
       TransposeParams::getDefaultTileSize(),
       TransposeParams::getDefaultTileSize()});
  std::vector<int64_t> shape_after({shape_before[1], shape_before[2] - 1});

  auto tv0 = makeSymbolicTensor(3);
  fusion.addInput(tv0);
  auto tv1 = makeSymbolicTensor(2, DataType::Int);
  fusion.addInput(tv1);

  auto tv2 = add(tv0, IrBuilder::create<Double>(1));
  auto tv3 = broadcast(tv1, {true, false, false});
  auto tv4 = take_along_axis(tv2, tv3, 2);
  auto tv5 = transpose(tv4, 1, 2);
  fusion.addOutput(tv5);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  auto options_i = at::TensorOptions().dtype(at::kLong).device(at::kCUDA, 0);
  auto t0 = at::randn(shape_before, options);
  auto t1 = at::randint(0, shape_before[2], shape_after, options_i);
  std::vector<c10::IValue> aten_inputs = {t0, t1};

  FusionExecutorCache fec(std::move(fusion_ptr));
  auto outputs = fec.runFusionWithInputs(aten_inputs);

  // Transpose scheduler should work for this case but not currently
  // supported
  validateSegmentation(
      fec.getMostRecentKernelRuntime(), {ScheduleHeuristic::PointWise});

  auto ref = at::take_along_dim(t0 + 1, t1.unsqueeze(0), 2).transpose(1, 2);

  testValidate(&fusion, outputs, aten_inputs, {ref}, __LINE__, __FILE__);
}

TEST_F(IndexingOpTest, TakeAlongAxisCrossEntropyLoss_CUDA) {
  std::unique_ptr<Fusion> fusion_ptr = std::make_unique<Fusion>();
  auto fusion = fusion_ptr.get();
  FusionGuard fg(fusion);

  auto tv0 = makeContigTensor(2);
  fusion->addInput(tv0);
  auto tv1 = makeContigTensor(1, DataType::Int);
  fusion->addInput(tv1);
  auto tv2 = max(tv0, {1});
  auto tv3 = broadcast(tv2, {false, true});
  auto tv4 =
      expand(tv3, {IrBuilder::create<Int>(128), IrBuilder::create<Int>(371)});
  auto tv5 = sub(tv0, tv4);
  auto tv6 = exp(tv5);
  auto tv7 = sum(tv6, {1});
  auto tv8 = broadcast(tv7, {false, true});
  auto tv9 =
      expand(tv8, {IrBuilder::create<Int>(128), IrBuilder::create<Int>(371)});
  auto tv10 = div(tv6, tv9);
  auto tv11 = log(tv10);
  auto tv12 = neg(tv11);
  auto tv13 = reshape(tv1, {128}, {128, 1});
  auto tv14 = take_along_axis(tv12, tv13, 1);
  auto s15 = IrBuilder::create<Int>(5);
  auto tv16 = eq(tv13, s15);
  auto s17 = IrBuilder::create<Double>(0.0);
  auto tv18 = where(tv16, s17, tv14);
  auto tv19 = sum(tv18, {0, 1});
  auto tv20 = castOp(DataType::Float, tv16);
  auto tv21 = sum(tv20, {0, 1});
  auto s22 = IrBuilder::create<Double>(128.0);
  auto tv23 = sub(s22, tv21);
  auto tv24 = div(tv19, tv23);
  fusion->addOutput(tv24);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  auto t0 = at::randn({128, 371}, options);
  auto t1 = at::randint(371, {128}, options).to(at::ScalarType::Long);
  std::vector<c10::IValue> inputs({t0, t1});

  FusionExecutorCache fec(std::move(fusion_ptr));

  auto cg_outputs = fec.runFusionWithInputs(inputs);

  auto kernel_runtime = fec.getMostRecentKernelRuntime();

  validateSegmentation(
      kernel_runtime,
      {ScheduleHeuristic::Persistent, ScheduleHeuristic::Reduction});

  // Make sure take_along_axis is in the persistent group
  for (const auto group : kernel_runtime->fusionSegments()->groups()) {
    if (group->heuristic() == ScheduleHeuristic::Persistent) {
      TORCH_CHECK(std::any_of(
          group->exprs().begin(), group->exprs().end(), [](Expr* expr) {
            return expr->isA<TorchGatherOp>();
          }));
    } else {
      TORCH_CHECK(std::none_of(
          group->exprs().begin(), group->exprs().end(), [](Expr* expr) {
            return expr->isA<TorchGatherOp>();
          }));
    }
  }

  // note: reduction arg
  //   none -> 0
  //   mean -> 1
  //   sum  -> 2
  auto ref = at::cross_entropy_loss_symint(t0, t1, {}, 1, 5, 0.0);
  testValidate(fusion, cg_outputs, inputs, {ref}, __LINE__, __FILE__);
}

} // namespace nvfuser
