// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#include <c10/core/ScalarType.h>
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
using testing::UnorderedElementsAre;
class GatherTest : public NVFuserTest {
 protected:
  void SetUp() override {
    // To make the tests using std::rand deterministic
    std::srand(0);
    EnableOptionsGuard::getCurOptions().set(EnableOption::IdModel, {"all"});
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

} // namespace

// all torch.gather test follow the FusionGather* pattern

// Test the correctness of gather operator in different dimensions and selcted
// dim.
TEST_F(GatherTest, GatherAllRankAllSelectedDim) {
  const int max_dim_size = 64;
  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  auto options_i = at::TensorOptions().dtype(at::kLong).device(at::kCUDA, 0);
  for (const auto is_take_along : {false, true}) {
    for (int rank = 1; rank <= 3; ++rank) {
      for (int dim = 0; dim < rank; ++dim) {
        // this test uses a random input shape, clear the allocator to avoid
        // OOM.
        maybeClearAllocator();
        auto fusion_ptr = std::make_unique<Fusion>();
        Fusion& fusion = *fusion_ptr.get();
        FusionGuard fg(&fusion);

        TensorView* tv1 = makeContigTensor(rank);
        TensorView* tv_idx = makeContigTensor(rank, DataType::Int);
        fusion.addInput(tv1);
        fusion.addInput(tv_idx);
        TensorView* tv_out = is_take_along ? takeAlongAxis(tv1, tv_idx, dim)
                                           : gather(tv1, dim, tv_idx);
        fusion.addOutput(tv_out);

        auto input_dims = randomVector(2, max_dim_size, rank);
        auto index_dims =
            randomIndexVector(input_dims, 1, rank, is_take_along, dim);
        at::Tensor t0 = at::randn(input_dims, options);
        at::Tensor idx = at::randint(0, input_dims[dim], index_dims, options_i);
        at::Tensor output = at::zeros(index_dims, options);

        FusionExecutorCache executor_cache(std::move(fusion_ptr));
        auto cg_outputs = executor_cache.runFusionWithInputs({t0, idx});
        testValidate(&fusion, cg_outputs, {t0, idx}, __LINE__, __FILE__);
      }
    }
  }
}
// Test the fusion support of gather operator(producer) and elemetwise(consumer)
TEST_F(GatherTest, GatherAddMul) {
  const int max_dim_size = 64;
  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  auto options_i = at::TensorOptions().dtype(at::kLong).device(at::kCUDA, 0);
  for (const auto is_take_along : {false, true}) {
    for (int rank = 1; rank <= 4; ++rank) {
      for (int dim = 0; dim < rank; ++dim) {
        auto fusion_ptr = std::make_unique<Fusion>();
        Fusion& fusion = *fusion_ptr.get();
        FusionGuard fg(&fusion);

        TensorView* tv1 = makeContigTensor(rank);
        TensorView* tv_idx = makeContigTensor(rank, DataType::Int);
        fusion.addInput(tv1);
        fusion.addInput(tv_idx);
        auto tv_gather = is_take_along ? takeAlongAxis(tv1, tv_idx, dim)
                                       : gather(tv1, dim, tv_idx);
        auto tv_add = add(tv_gather, tv_gather);
        auto tv_out = mul(tv_gather, tv_add);
        fusion.addOutput(tv_out);

        auto input_dims = randomVector(2, max_dim_size, rank);
        auto index_dims =
            randomIndexVector(input_dims, 1, rank, is_take_along, dim);

        at::Tensor t0 = at::randn(input_dims, options); // lookup
        at::Tensor idx = at::randint(0, input_dims[dim], index_dims, options_i);

        FusionExecutorCache executor_cache(std::move(fusion_ptr));
        auto cg_outputs = executor_cache.runFusionWithInputs({t0, idx});
        testValidate(&fusion, cg_outputs, {t0, idx}, __LINE__, __FILE__);
      }
    }
  }
}
// Test the fusion support of index tensor as fusion input in gather operator
TEST_F(GatherTest, AddGatherSumAdd) {
  const int max_dim_size = 8;
  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  auto options_i = at::TensorOptions().dtype(at::kLong).device(at::kCUDA, 0);
  for (const auto is_take_along : {false, true}) {
    for (int rank = 2; rank <= 4; ++rank) {
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
        auto tv_out = is_take_along ? takeAlongAxis(tv_lookup, tv_index, dim)
                                    : gather(tv_lookup, dim, tv_index);

        fusion.addOutput(tv_out);

        auto input_dims = randomVector(2, max_dim_size, rank);
        auto index_dims =
            randomIndexVector(input_dims, 1, rank, is_take_along, dim);

        at::Tensor t_lookup = at::randn(input_dims, options); // lookup
        at::Tensor t_idx_1 =
            at::randint(0, input_dims[dim] / 2, index_dims, options_i);
        at::Tensor t_idx_2 =
            at::randint(0, input_dims[dim] / 2, index_dims, options_i);

        FusionExecutorCache executor_cache(std::move(fusion_ptr));
        auto cg_outputs =
            executor_cache.runFusionWithInputs({t_lookup, t_idx_1, t_idx_2});
        testValidate(
            &fusion,
            cg_outputs,
            {t_lookup, t_idx_1, t_idx_2},
            __LINE__,
            __FILE__);
      }
    }
  }
}
// Test the fusion support of gather operator and reduce
TEST_F(GatherTest, GatherSumAdd) {
  const int max_dim_size = 32;
  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  auto options_i = at::TensorOptions().dtype(at::kLong).device(at::kCUDA, 0);
  for (const auto is_take_along : {false, true}) {
    for (int rank = 2; rank <= 4; ++rank) {
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

        auto tv_gather = is_take_along ? takeAlongAxis(tv1, tv_idx, dim)
                                       : gather(tv1, dim, tv_idx);
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

        at::Tensor t0 = at::randn(input_dims, options); // lookup
        at::Tensor t1 = at::randn(input2_dims, options); // lookup
        at::Tensor idx = at::randint(0, input_dims[dim], index_dims, options_i);

        FusionExecutorCache executor_cache(std::move(fusion_ptr));
        auto cg_outputs = executor_cache.runFusionWithInputs({t0, idx, t1});
        testValidate(&fusion, cg_outputs, {t0, idx, t1}, __LINE__, __FILE__);
      }
    }
  }
}
// Test the correctness when input/index tensor is very large
TEST_F(GatherTest, GatherAddMulHugeSize) {
  const int max_dim_size = 16384;
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
        auto tv_gather = is_take_along ? takeAlongAxis(tv1, tv_idx, dim)
                                       : gather(tv1, dim, tv_idx);
        auto tv_add = add(tv_gather, tv_gather);
        auto tv_out = mul(tv_gather, tv_add);
        fusion.addOutput(tv_out);

        auto input_dims = randomVector(2, max_dim_size, rank);
        auto index_dims =
            randomIndexVector(input_dims, 1, rank, is_take_along, dim);

        at::Tensor t0 = at::randn(input_dims, options); // lookup
        at::Tensor idx = at::randint(0, input_dims[dim], index_dims, options_i);

        FusionExecutorCache executor_cache(std::move(fusion_ptr));
        auto cg_outputs = executor_cache.runFusionWithInputs({t0, idx});
        testValidate(&fusion, cg_outputs, {t0, idx}, __LINE__, __FILE__);
      }
    }
  }
}
// Test the fusion support of input tensor as fusion input
TEST_F(GatherTest, GatherInput) {
  const int rank = 2;

  auto fusion_ptr = std::make_unique<Fusion>();
  Fusion& fusion = *fusion_ptr.get();
  FusionGuard fg(&fusion);

  TensorView* tv1 = makeContigTensor(rank);
  TensorView* tv_idx = makeContigTensor(rank, DataType::Int);
  fusion.addInput(tv1);
  fusion.addInput(tv_idx);

  auto tv_inp = add(tv1, tv1);
  auto tv_gather = gather(tv_inp, 0, tv_idx);
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
TEST_F(GatherTest, GatherIndexTvExtentIsOne) {
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

  auto tv_gather = gather(tv_in1, 1, tv_idx);
  auto tv_add =
      clamp(tv_gather, IrBuilder::create<Val>(-1L), IrBuilder::create<Val>(1L));
  auto tv_out = mul(tv_add, tv_in2);
  fusion.addOutput(tv_out);

  at::Tensor t0 = at::randn(input_dims, options);
  at::Tensor t1 = at::randn(index_dims, options);
  at::Tensor idx = at::randint(0, max_selected_index, index_dims, options_i);
  at::Tensor output = at::zeros(index_dims, options);

  auto t_gather = at::gather(t0, 1, idx);
  auto t_add = at::clamp(t_gather, -1, 1);
  auto tv_out_ref = at::mul(t1, t_add);

  FusionExecutorCache executor_cache(std::move(fusion_ptr));
  auto cg_outputs = executor_cache.runFusionWithInputs({t0, idx, t1});
  testValidate(
      &fusion, cg_outputs, {t0, idx, t1}, {tv_out_ref}, __LINE__, __FILE__);
}

// Test takeAlongAxis with a broadcast index tensor
TEST_F(GatherTest, TakeAlongBroadcastIndex) {
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
    auto tv4 = takeAlongAxis(tv0, tv3, 1);
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

    FusionExecutorCache executor_cache(std::move(fusion_ptr));
    auto cg_outputs = executor_cache.runFusionWithInputs({t0, t1, t2});

    testValidate(&fusion, cg_outputs, {t0, t1, t2}, __LINE__, __FILE__);
  }
}

TEST_F(GatherTest, GatherBroadcastInput) {
  for (const auto is_take_along : {false, true}) {
    // gather not supported yet. The issue is one of the index
    // tensor has a broadcast domain, but its corresponding input
    // domain is a normal domain. The output domain is also a
    // broadcast in gather, whereas it's a normal domain in
    // takeAlongAxis. In the case of gather, indexing the
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
        // In gather, an index dimension must be smaller or
        // equal to the corresponding input dimension
        std::vector<int64_t> index_dims{
            is_take_along ? 5 : input_dims.at(0), idx_index_dim};
        // This needs to match with the takeAlongAxis output
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
        auto tv4 = takeAlongAxis(tv0, tv3, 1);
        auto tv5 = add(tv4, tv2);
        fusion.addOutput(tv5);

        auto options =
            at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
        auto options_i =
            at::TensorOptions().dtype(at::kLong).device(at::kCUDA, 0);
        at::Tensor t0 = at::randn(input_dims, options);
        at::Tensor t1 = at::randint(0, input_dims[1], index_dims, options_i);
        at::Tensor t2 = at::randn(out_dims, options);

        FusionExecutorCache executor_cache(std::move(fusion_ptr));
        auto cg_outputs = executor_cache.runFusionWithInputs({t0, t1, t2});
        testValidate(&fusion, cg_outputs, {t0, t1, t2}, __LINE__, __FILE__);
      }
    }
  }
}

// Test takeAlongAxis with non fusion inputs
TEST_F(GatherTest, TakeAlongAxisIntermediateTensorPointwise1) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  const std::vector<int64_t> shape({99, 101});

  auto tv0 = makeSymbolicTensor(2);
  fusion.addInput(tv0);
  auto tv1 = makeSymbolicTensor(1, DataType::Int);
  fusion.addInput(tv1);

  auto tv2 = add(tv0, IrBuilder::create<Val>(1.0));
  auto tv3 = broadcast(tv1, {false, true});
  auto tv4 = takeAlongAxis(tv2, tv3, 1);
  fusion.addOutput(tv4);

  scheduler_utils::prepareForMemoryTypePromotion(&fusion);

  // Test if this split is propagated through the indexed domain
  tv4->split(1, 10);

  TransformPropagator propagator(tv4);
  MaxLogicalDomainInfoSpanningTree(tv4).traverse(&propagator);

  // All of the tensors should have the split by 2, except for tv1.
  for (auto tv : ir_utils::allTvsExcept(&fusion, {tv1})) {
    NVF_CHECK(tv->nDims() == 3, "Unexpected tensor: ", tv->toString());
    NVF_CHECK(
        tv->axis(-1)->definition() &&
            tv->axis(-1)->definition()->isA<Split>() &&
            tv->axis(-1)->definition()->as<Split>()->in() ==
                tv->getLogicalDomain().at(1),
        "Unexpected tensor: ",
        tv->toString());
  }

  // This should not inline the indexed producer domain. Note that the
  // producer tensor of the takeAlongAxis expr is not tv2 as a copy
  // is inserted
  inlineMost();
  auto take_along_axis_input = tv4->definition()->as<GatherOp>()->lookupTv();
  NVF_CHECK(
      take_along_axis_input->getComputeAtPosition() == 1,
      "Unexpected computeAt position: ",
      take_along_axis_input->toString());

  // Test if parallelization is propagated through indexed domains
  tv4->axis(-2)->parallelize(ParallelType::TIDy);
  tv4->axis(-1)->parallelize(ParallelType::TIDx);
  scheduler_utils::parallelizeAllLike(tv4);

  for (auto tv : fusion.allTvs()) {
    if (tv->isFusionInput()) {
      continue;
    }
    NVF_CHECK(
        tv->axis(-2)->getParallelType() == tv4->axis(-2)->getParallelType() &&
            tv->axis(-1)->getParallelType() == tv4->axis(-1)->getParallelType(),
        "Unexpected parallelization of tensor: ",
        tv->toString());
  }

  // This should make the producer of take_along_axis saved in shared memory
  scheduler_utils::promoteProducerMemoryTypes(&fusion, {});

  NVF_CHECK(
      take_along_axis_input->getMemoryType() == MemoryType::Shared,
      "Failed to promote memory type: ",
      take_along_axis_input->toString());

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  auto options_i = at::TensorOptions().dtype(at::kLong).device(at::kCUDA, 0);
  auto t0 = at::randn(shape, options);
  auto t1 = at::randint(0, shape[1], {shape[0]}, options_i);

  KernelExecutor ke;
  ke.compile(&fusion, {t0, t1});

  auto outputs = ke.run({t0, t1});

  testValidate(&fusion, outputs, {t0, t1}, __LINE__, __FILE__);
}

// Same as the above but with the pointwise scheduler
TEST_F(GatherTest, TakeAlongAxisIntermediateTensorPointwise2) {
  auto fusion_ptr = std::make_unique<Fusion>();
  Fusion& fusion = *fusion_ptr.get();
  FusionGuard fg(&fusion);

  EnableOptionsGuard::getCurOptions().set(EnableOption::MemoryPromotion);

  std::vector<int64_t> shape({99, 101});

  auto tv0 = makeSymbolicTensor(2);
  fusion.addInput(tv0);
  auto tv1 = makeSymbolicTensor(1, DataType::Int);
  fusion.addInput(tv1);

  auto tv2 = add(tv0, IrBuilder::create<Val>(1.0));
  auto tv3 = broadcast(tv1, {false, true});
  auto tv4 = takeAlongAxis(tv2, tv3, 1);
  fusion.addOutput(tv4);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  auto options_i = at::TensorOptions().dtype(at::kLong).device(at::kCUDA, 0);
  auto t0 = at::randn(shape, options);
  auto t1 = at::randint(0, shape[1], {shape[0]}, options_i);

  FusionExecutorCache executor_cache(std::move(fusion_ptr));
  auto outputs = executor_cache.runFusionWithInputs({t0, t1});

  validateSegmentation(
      executor_cache.getMostRecentKernelRuntime(), {SchedulerType::PointWise});

  testValidate(&fusion, outputs, {t0, t1}, __LINE__, __FILE__);
}

// Reduction then takeAlongAxis. This is currently segmented due to
// the post-reduction rule as documented in
// https://github.com/NVIDIA/Fuser/issues/260
TEST_F(GatherTest, TakeAlongAxisIntermediateTensorReduction1) {
  auto fusion_ptr = std::make_unique<Fusion>();
  Fusion& fusion = *fusion_ptr.get();
  FusionGuard fg(&fusion);

  std::vector<int64_t> shape({100, 1000});

  auto tv0 = makeSymbolicTensor(2);
  fusion.addInput(tv0);
  auto tv1 = makeSymbolicTensor(1, DataType::Int);
  fusion.addInput(tv1);

  auto tv2 = sum(tv0, {1});
  auto tv4 = takeAlongAxis(tv2, tv1, 0);
  fusion.addOutput(tv4);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  auto options_i = at::TensorOptions().dtype(at::kLong).device(at::kCUDA, 0);
  auto t0 = at::randn(shape, options);
  auto t1 = at::randint(0, shape[0], {2}, options_i);

  FusionExecutorCache executor_cache(std::move(fusion_ptr));
  auto outputs = executor_cache.runFusionWithInputs({t0, t1});

  validateSegmentation(
      executor_cache.getMostRecentKernelRuntime(),
      {SchedulerType::Reduction, SchedulerType::PointWise});

  testValidate(&fusion, outputs, {t0, t1}, __LINE__, __FILE__);
}

// takeAlongAxis to broadcast, squeeze, then reduction. Segmented
// before the reduction
TEST_F(GatherTest, TakeAlongAxisIntermediateTensorReduction2) {
  auto fusion_ptr = std::make_unique<Fusion>();
  Fusion& fusion = *fusion_ptr.get();
  FusionGuard fg(&fusion);

  EnableOptionsGuard::getCurOptions().set(EnableOption::MemoryPromotion);

  std::vector<int64_t> shape({100, 100});

  auto tv0 = makeSymbolicTensor(2);
  fusion.addInput(tv0);
  auto tv1 = makeSymbolicTensor(1, DataType::Int);
  fusion.addInput(tv1);

  auto tv2 = add(tv0, IrBuilder::create<Val>(1.0));
  auto tv3 = broadcast(tv1, {false, true});
  auto tv4 = takeAlongAxis(tv2, tv3, 1);
  auto tv5 = squeeze(tv4, std::vector<bool>{false, true});
  auto tv6 = sum(tv5, {0});
  fusion.addOutput(tv6);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  auto options_i = at::TensorOptions().dtype(at::kLong).device(at::kCUDA, 0);
  auto t0 = at::randn(shape, options);
  auto t1 = at::randint(0, shape[1], {shape[0]}, options_i);

  FusionExecutorCache executor_cache(std::move(fusion_ptr));
  auto outputs = executor_cache.runFusionWithInputs({t0, t1});

  validateSegmentation(
      executor_cache.getMostRecentKernelRuntime(),
      {SchedulerType::PointWise, SchedulerType::Reduction});

  testValidate(&fusion, outputs, {t0, t1}, __LINE__, __FILE__);
}

// takeAlongAxis then reduction. Should not be segmented.
TEST_F(GatherTest, TakeAlongAxisIntermediateTensorReduction3) {
  auto fusion_ptr = std::make_unique<Fusion>();
  Fusion& fusion = *fusion_ptr.get();
  FusionGuard fg(&fusion);

  EnableOptionsGuard::getCurOptions().set(EnableOption::MemoryPromotion);

  std::vector<int64_t> shape_before_gather({100, 100});
  std::vector<int64_t> shape_after_gather({shape_before_gather[0], 120});

  auto tv0 = makeSymbolicTensor(2);
  fusion.addInput(tv0);
  auto tv1 = makeSymbolicTensor(2, DataType::Int);
  fusion.addInput(tv1);

  auto tv2 = add(tv0, IrBuilder::create<Val>(1.0));
  auto tv3 = takeAlongAxis(tv2, tv1, 1);
  auto tv4 = sum(tv3, {1});
  fusion.addOutput(tv4);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  auto options_i = at::TensorOptions().dtype(at::kLong).device(at::kCUDA, 0);
  auto t0 = at::randn(shape_before_gather, options);
  auto t1 =
      at::randint(0, shape_before_gather[1], shape_after_gather, options_i);

  FusionExecutorCache executor_cache(std::move(fusion_ptr));
  auto outputs = executor_cache.runFusionWithInputs({t0, t1});

  validateSegmentation(
      executor_cache.getMostRecentKernelRuntime(), {SchedulerType::Reduction});

  testValidate(&fusion, outputs, {t0, t1}, __LINE__, __FILE__);
}

// Similar to TakeAlongAxisIntermediateTensorReduction2, but no
// squeeze of the consumer ID of the indexed domain. Should not be segmented.
//
// Disabled due to #293.
TEST_F(GatherTest, DISABLED_TakeAlongAxisIntermediateTensorReduction4) {
  auto fusion_ptr = std::make_unique<Fusion>();
  Fusion& fusion = *fusion_ptr.get();
  FusionGuard fg(&fusion);

  std::vector<int64_t> shape_before_gather({10, 10});
  std::vector<int64_t> shape_after_gather({shape_before_gather[0]});

  auto tv0 = makeSymbolicTensor(2);
  fusion.addInput(tv0);
  auto tv1 = makeSymbolicTensor(1, DataType::Int);
  fusion.addInput(tv1);

  auto tv2 = add(tv0, IrBuilder::create<Val>(1.0));
  auto tv3 = broadcast(tv1, {false, true});
  auto tv4 = takeAlongAxis(tv2, tv3, 1);
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

  FusionExecutorCache executor_cache(std::move(fusion_ptr));
  auto outputs = executor_cache.runFusionWithInputs({t0, t1});

  validateSegmentation(
      executor_cache.getMostRecentKernelRuntime(), {SchedulerType::Reduction});

  testValidate(&fusion, outputs, {t0, t1}, __LINE__, __FILE__);
}

// Normalization then takeAlongAxis
TEST_F(GatherTest, TakeAlongAxisIntermediateTensorNormalization1) {
  auto fusion_ptr = std::make_unique<Fusion>();
  Fusion& fusion = *fusion_ptr.get();
  FusionGuard fg(&fusion);

  EnableOptionsGuard::getCurOptions().set(EnableOption::MemoryPromotion);

  std::vector<int64_t> shape({32, 1024});

  auto tv0 = makeSymbolicTensor(2);
  fusion.addInput(tv0);
  auto tv1 = makeSymbolicTensor(1, DataType::Int);
  fusion.addInput(tv1);

  auto tv2 = sum(tv0, {1});
  auto tv3 = broadcast(tv2, {false, true});
  auto tv4 = div(tv0, tv3);
  auto tv5 = broadcast(tv1, {false, true});
  auto tv6 = takeAlongAxis(tv4, tv5, 1);
  fusion.addOutput(tv6);

  at::manual_seed(0);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  auto options_i = at::TensorOptions().dtype(at::kLong).device(at::kCUDA, 0);
  auto t0 = at::randn(shape, options);
  auto t1 = at::randint(0, shape[1], {shape[0]}, options_i);

  FusionExecutorCache executor_cache(std::move(fusion_ptr));
  auto outputs = executor_cache.runFusionWithInputs({t0, t1});

  validateSegmentation(
      executor_cache.getMostRecentKernelRuntime(),
      {SchedulerType::InnerPersistent});

  auto t0_d = t0.to(at::kDouble);
  auto ref = at::take_along_dim(
      t0_d / t0_d.sum({1}).unsqueeze(-1), t1.unsqueeze(-1), 1);

  testValidate(&fusion, outputs, {t0, t1}, {ref}, __LINE__, __FILE__);
}

// take_along_dim to broadcast, squeeze, then normalization. Segmented
// as the input dim to take_along_dim cannot be scheduled by the
// reduction tv
//
// NOTE: Temporarily disabled as it results in non-deterministic
// validaiton errors (https://github.com/NVIDIA/Fuser/issues/4003).
TEST_F(GatherTest, DISABLED_TakeAlongAxisIntermediateTensorNormalization2) {
  auto fusion_ptr = std::make_unique<Fusion>();
  Fusion& fusion = *fusion_ptr.get();
  FusionGuard fg(&fusion);

  std::vector<int64_t> shape({32, 1024});

  auto tv0 = makeSymbolicTensor(2);
  fusion.addInput(tv0);
  auto tv1 = makeSymbolicTensor(1, DataType::Int);
  fusion.addInput(tv1);

  auto tv2 = add(tv0, IrBuilder::create<Val>(1.0));
  auto tv3 = broadcast(tv1, {false, true});
  auto tv4 = takeAlongAxis(tv2, tv3, 1);
  auto tv5 = squeeze(tv4, std::vector<bool>{false, true});
  auto tv6 = sum(tv5, {0});
  auto tv7 = broadcast(tv6, {true});
  auto tv8 = div(tv5, tv7);
  fusion.addOutput(tv8);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  auto options_i = at::TensorOptions().dtype(at::kLong).device(at::kCUDA, 0);
  auto t0 = at::randn(shape, options);
  auto t1 = at::randint(0, shape[1], {shape[0]}, options_i);

  FusionExecutorCache executor_cache(std::move(fusion_ptr));
  auto outputs = executor_cache.runFusionWithInputs({t0, t1});

  validateSegmentation(
      executor_cache.getMostRecentKernelRuntime(),
      {SchedulerType::PointWise, SchedulerType::InnerPersistent});

  auto t5 = at::take_along_dim(t0.to(at::kDouble) + 1, t1.unsqueeze(-1), 1)
                .squeeze(1);
  auto ref = t5 / t5.sum({0}).unsqueeze(0);

  testValidate(&fusion, outputs, {t0, t1}, {ref}, __LINE__, __FILE__);
}

// takeAlongAxis then normalization. Should not be segmented.
TEST_F(GatherTest, TakeAlongAxisIntermediateTensorNormalization3) {
  auto fusion_ptr = std::make_unique<Fusion>();
  Fusion& fusion = *fusion_ptr.get();
  FusionGuard fg(&fusion);

  EnableOptionsGuard::getCurOptions().set(EnableOption::MemoryPromotion);

  std::vector<int64_t> shape_before_gather({100, 100});
  std::vector<int64_t> shape_after_gather({shape_before_gather[0], 120});

  auto tv0 = makeSymbolicTensor(2);
  fusion.addInput(tv0);
  auto tv1 = makeSymbolicTensor(2, DataType::Int);
  fusion.addInput(tv1);

  auto tv2 = add(tv0, IrBuilder::create<Val>(1.0));
  auto tv3 = takeAlongAxis(tv2, tv1, 1);
  auto tv4 = sum(tv3, {1});
  auto tv5 = broadcast(tv4, {false, true});
  auto tv6 = div(tv3, tv5);
  fusion.addOutput(tv6);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  auto options_i = at::TensorOptions().dtype(at::kLong).device(at::kCUDA, 0);
  auto t0 = at::randn(shape_before_gather, options);
  auto t1 =
      at::randint(0, shape_before_gather[1], shape_after_gather, options_i);

  FusionExecutorCache executor_cache(std::move(fusion_ptr));
  auto outputs = executor_cache.runFusionWithInputs({t0, t1});

  validateSegmentation(
      executor_cache.getMostRecentKernelRuntime(),
      {SchedulerType::InnerPersistent});

  auto t3 = at::take_along_dim(t0.to(at::kDouble) + 1, t1, 1);
  auto ref = t3 / t3.sum({1}).unsqueeze(-1);

  testValidate(&fusion, outputs, {t0, t1}, {ref}, __LINE__, __FILE__);
}

// Normalization, then takeAlongAxis, then reduction. Similar
// pattern as cross entropy.
TEST_F(GatherTest, TakeAlongAxisIntermediateTensorNormalizationAndReduction1) {
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
  auto tv5 = takeAlongAxis(tv4, tv1, 1);
  auto tv6 = sum(tv5, {0, 1});
  fusion.addOutput(tv6);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  auto options_i = at::TensorOptions().dtype(at::kLong).device(at::kCUDA, 0);
  auto t0 = at::randn(shape, options);
  auto t1 = at::randint(0, shape[1], {shape[0], 1}, options_i);

  FusionExecutorCache executor_cache(std::move(fusion_ptr));
  auto outputs = executor_cache.runFusionWithInputs({t0, t1});

  // The reduction patterns of the normalization and the final
  // reduction are different, so they are segmented out
  validateSegmentation(
      executor_cache.getMostRecentKernelRuntime(),
      {SchedulerType::InnerPersistent, SchedulerType::Reduction});

  auto t0_d = t0.to(at::kDouble);
  auto t5 = at::take_along_dim(t0_d / t0_d.sum({1}).unsqueeze(-1), t1, 1);
  auto ref = t5.sum();

  testValidate(&fusion, outputs, {t0, t1}, {ref}, __LINE__, __FILE__);
}

// Similar to
// TakeAlongAxisIntermediateTensorNormalizationAndReduction1, but the
// final reduction pattern is compatible with the first reduction, so
// no segmentation should be done
TEST_F(GatherTest, TakeAlongAxisIntermediateTensorNormalizationAndReduction2) {
  auto fusion_ptr = std::make_unique<Fusion>();
  Fusion& fusion = *fusion_ptr.get();
  FusionGuard fg(&fusion);

  EnableOptionsGuard::getCurOptions().set(EnableOption::MemoryPromotion);

  std::vector<int64_t> shape({32, 1024});

  auto tv0 = makeSymbolicTensor(2);
  fusion.addInput(tv0);
  auto tv1 = makeSymbolicTensor(1, DataType::Int);
  fusion.addInput(tv1);

  auto tv2 = sum(tv0, {1});
  auto tv3 = broadcast(tv2, {false, true});
  auto tv4 = div(tv0, tv3);
  auto tv5 = broadcast(tv1, {false, true});
  auto tv6 = takeAlongAxis(tv4, tv5, 1);
  auto tv7 = add(tv0, tv6);
  auto tv8 = sum(tv7, {1});
  fusion.addOutput(tv8);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  auto options_i = at::TensorOptions().dtype(at::kLong).device(at::kCUDA, 0);
  auto t0 = at::randn(shape, options);
  auto t1 = at::randint(0, shape[1], {shape[0]}, options_i);

  FusionExecutorCache executor_cache(std::move(fusion_ptr));
  auto outputs = executor_cache.runFusionWithInputs({t0, t1});

  validateSegmentation(
      executor_cache.getMostRecentKernelRuntime(),
      {SchedulerType::InnerPersistent});

  auto t0_d = t0.to(at::kDouble);
  auto t6 = at::take_along_dim(
      t0_d / t0_d.sum({1}).unsqueeze(-1), t1.unsqueeze(-1), 1);
  auto ref = (t0_d + t6).sum({1});

  testValidate(&fusion, outputs, {t0, t1}, {ref}, __LINE__, __FILE__);
}

// takeAlongAxis then transpose
TEST_F(GatherTest, TakeAlongAxisIntermediateTensorTranspose1) {
  auto fusion_ptr = std::make_unique<Fusion>();
  Fusion& fusion = *fusion_ptr.get();
  FusionGuard fg(&fusion);

  EnableOptionsGuard::getCurOptions().set(EnableOption::MemoryPromotion);

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

  auto tv2 = add(tv0, IrBuilder::create<Val>(1.0));
  auto tv3 = broadcast(tv1, {true, false, false});
  auto tv4 = takeAlongAxis(tv2, tv3, 0);
  auto tv5 = transpose(tv4, 1, 2);
  fusion.addOutput(tv5);
  // specify output allocation domain to avoid allocation order pass changing
  // this to a pointwise kernel
  tv5->setAllocationDomain(tv5->getLogicalDomain(), true);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  auto options_i = at::TensorOptions().dtype(at::kLong).device(at::kCUDA, 0);
  auto t0 = at::randn(shape, options);
  auto t1 = at::randint(0, shape[0], {shape[1], shape[2]}, options_i);

  FusionExecutorCache executor_cache(std::move(fusion_ptr));
  auto outputs = executor_cache.runFusionWithInputs({t0, t1});

  validateSegmentation(
      executor_cache.getMostRecentKernelRuntime(), {SchedulerType::Transpose});

  testValidate(&fusion, outputs, {t0, t1}, __LINE__, __FILE__);
}

// transpose then takeAlongAxis. Currently failed to pick the
// Transpose scheduler due to a limitation of the analysis for the
// scheduler. See DomainMap::findReferenceFor in transpose.cpp for
// more details.
TEST_F(GatherTest, TakeAlongAxisIntermediateTensorTranspose2) {
  auto fusion_ptr = std::make_unique<Fusion>();
  Fusion& fusion = *fusion_ptr.get();
  FusionGuard fg(&fusion);

  EnableOptionsGuard::getCurOptions().set(EnableOption::MemoryPromotion);

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
  auto tv4 = takeAlongAxis(tv2, tv1, 0);
  fusion.addOutput(tv4);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  auto options_i = at::TensorOptions().dtype(at::kLong).device(at::kCUDA, 0);
  auto t0 = at::randn(shape, options);
  auto t1 = at::randint(0, shape[0], {10, shape[2], shape[1]}, options_i);

  FusionExecutorCache executor_cache(std::move(fusion_ptr));
  auto outputs = executor_cache.runFusionWithInputs({t0, t1});

  validateSegmentation(
      executor_cache.getMostRecentKernelRuntime(), {SchedulerType::PointWise});

  testValidate(&fusion, outputs, {t0, t1}, __LINE__, __FILE__);
}

// transpose the dimension produced by takeAlongAxis. Currently not
// supported by the transpose scheduler
TEST_F(GatherTest, TakeAlongAxisIntermediateTensorTranspose3) {
  auto fusion_ptr = std::make_unique<Fusion>();
  Fusion& fusion = *fusion_ptr.get();
  FusionGuard fg(&fusion);

  EnableOptionsGuard::getCurOptions().set(EnableOption::MemoryPromotion);

  std::vector<int64_t> shape_before(
      {deviceSMCount(),
       TransposeParams::getDefaultTileSize(),
       TransposeParams::getDefaultTileSize()});
  std::vector<int64_t> shape_after({shape_before[1], shape_before[2] - 1});

  auto tv0 = makeSymbolicTensor(3);
  fusion.addInput(tv0);
  auto tv1 = makeSymbolicTensor(2, DataType::Int);
  fusion.addInput(tv1);

  auto tv2 = add(tv0, IrBuilder::create<Val>(1.0));
  auto tv3 = broadcast(tv1, {true, false, false});
  auto tv4 = takeAlongAxis(tv2, tv3, 2);
  auto tv5 = transpose(tv4, 1, 2);
  // Without the `add`, the transpose will be taken by NoOp, defeating the
  // purpose of testing PointWise.
  auto tv6 = add(tv5, tv5);
  fusion.addOutput(tv6);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  auto options_i = at::TensorOptions().dtype(at::kLong).device(at::kCUDA, 0);
  auto t0 = at::randn(shape_before, options);
  auto t1 = at::randint(0, shape_before[2], shape_after, options_i);

  FusionExecutorCache executor_cache(std::move(fusion_ptr));
  auto outputs = executor_cache.runFusionWithInputs({t0, t1});

  // Transpose scheduler should work for this case but not currently
  // supported
  validateSegmentation(
      executor_cache.getMostRecentKernelRuntime(), {SchedulerType::PointWise});

  testValidate(&fusion, outputs, {t0, t1}, __LINE__, __FILE__);
}

TEST_F(GatherTest, TakeAlongAxisCrossEntropyLoss) {
  std::unique_ptr<Fusion> fusion_ptr = std::make_unique<Fusion>();
  auto fusion = fusion_ptr.get();
  FusionGuard fg(fusion);

  EnableOptionsGuard::getCurOptions().set(EnableOption::MemoryPromotion);

  auto tv0 = makeContigTensor(2);
  fusion->addInput(tv0);
  auto tv1 = makeContigTensor(1, DataType::Int);
  fusion->addInput(tv1);
  auto tv2 = max(tv0, {1});
  auto tv3 = broadcast(tv2, {false, true});
  auto tv4 =
      expand(tv3, {IrBuilder::create<Val>(128L), IrBuilder::create<Val>(371L)});
  auto tv5 = sub(tv0, tv4);
  auto tv6 = exp(tv5);
  auto tv7 = sum(tv6, {1});
  auto tv8 = broadcast(tv7, {false, true});
  auto tv9 =
      expand(tv8, {IrBuilder::create<Val>(128L), IrBuilder::create<Val>(371L)});
  auto tv10 = div(tv6, tv9);
  auto tv11 = log(tv10);
  auto tv12 = neg(tv11);
  auto tv13 = reshape(tv1, {128}, {128, 1});
  auto tv14 = takeAlongAxis(tv12, tv13, 1);
  auto s15 = IrBuilder::create<Val>(5L);
  auto tv16 = eq(tv13, s15);
  auto s17 = IrBuilder::create<Val>(0.0);
  auto tv18 = where(tv16, s17, tv14);
  auto tv19 = sum(tv18, {0, 1});
  auto tv20 = castOp(DataType::Float, tv16);
  auto tv21 = sum(tv20, {0, 1});
  auto s22 = IrBuilder::create<Val>(128.0);
  auto tv23 = sub(s22, tv21);
  auto tv24 = div(tv19, tv23);
  fusion->addOutput(tv24);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  auto t0 = at::randn({128, 371}, options);
  auto t1 = at::randint(371, {128}, options).to(at::ScalarType::Long);

  FusionExecutorCache executor_cache(std::move(fusion_ptr));

  auto cg_outputs = executor_cache.runFusionWithInputs({t0, t1});

  auto kernel_runtime = executor_cache.getMostRecentKernelRuntime();

  validateSegmentation(
      kernel_runtime,
      {SchedulerType::InnerPersistent, SchedulerType::Reduction});

  // Make sure takeAlongAxis is in the persistent group
  for (const auto group : kernel_runtime->fusionSegments()->groups()) {
    if (group->schedulerType() == SchedulerType::InnerPersistent) {
      NVF_CHECK(std::any_of(
          group->exprs().begin(), group->exprs().end(), [](Expr* expr) {
            return expr->isA<GatherOp>();
          }));
    } else {
      NVF_CHECK(std::none_of(
          group->exprs().begin(), group->exprs().end(), [](Expr* expr) {
            return expr->isA<GatherOp>();
          }));
    }
  }

  // note: reduction arg
  //   none -> 0
  //   mean -> 1
  //   sum  -> 2
  auto ref = at::cross_entropy_loss_symint(t0, t1, {}, 1, 5, 0.0);
  testValidate(fusion, cg_outputs, {t0, t1}, {ref}, __LINE__, __FILE__);
}

// Test grouped reduction on IterType::GatherScatter
TEST_F(GatherTest, GatherIterGoupedReduction) {
  const int max_dim_size = 128;
  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  auto options_i = at::TensorOptions().dtype(at::kLong).device(at::kCUDA, 0);

  int rank = 3;
  int dim = 2;

  auto fusion_ptr = std::make_unique<Fusion>();
  Fusion& fusion = *fusion_ptr.get();
  FusionGuard fg(&fusion);

  TensorView* tv1 = makeContigTensor(rank);
  TensorView* tv_idx = makeContigTensor(rank, DataType::Int);
  fusion.addInput(tv1);
  fusion.addInput(tv_idx);
  auto tv_gather = gather(tv1, dim, tv_idx);
  auto tv_sum = sum(tv_gather, {0}, false);
  fusion.addOutput(tv_sum);

  // simply gather all elements
  auto input_dims =
      std::vector<int64_t>({max_dim_size, max_dim_size, max_dim_size});
  auto index_dims = input_dims;
  std::vector<int64_t> input2_dims(rank - 1, 0);
  for (int idim = 0; idim < rank - 1; ++idim) {
    input2_dims[idim] = index_dims[idim + 1];
  }

  at::Tensor t0 = at::randn(input_dims, options);
  at::Tensor idx = at::randint(0, input_dims[dim], index_dims, options_i);

  auto reduction_scheduler =
      SchedulerEntry::makeSchedulerInstance(SchedulerType::Reduction);
  SchedulerRuntimeInfo runtime_info(&fusion, {t0, idx});
  auto heuristic_params =
      reduction_scheduler->computeHeuristics(&fusion, runtime_info);
  auto rparams = heuristic_params->as<ReductionParams>();

  // Enforce vectorization so we can group them
  const int vect_factor = 2;
  rparams->vectorize_iter_dom = true;
  rparams->unroll_factor_iter_dom = vect_factor;
  // Enforce grid reduction, which requires a determined BIDy
  // If the heuristic does not have a BIDy, bind it to 2
  rparams->cross_grid_inner_reduction = true;
  rparams->split_grid_dim_inner_reduction = true;
  rparams->grid_dim_inner_reduction = ParallelType::BIDy;
  if (!rparams->lparams.hasDim(ParallelType::BIDy)) {
    rparams->lparams.bind(2L, ParallelType::BIDy);
  }

  reduction_scheduler->schedule(&fusion, rparams);

  // lowering & check iteration grouped reductions
  GpuLower gpulw(&fusion);
  gpulw.run();
  NVF_CHECK(
      gpulw.kernel()->summary().has_iter_grouped_reductions,
      "There must be iter domain grouped reductions.");
  NVF_CHECK(
      gpulw.kernel()->summary().num_grouped_iterations == vect_factor,
      "Expected ",
      vect_factor,
      " grouped iterations, found ",
      gpulw.kernel()->summary().num_grouped_iterations);

  KernelExecutor ke;
  auto lparams = rparams->lparams;
  ke.compile(&fusion, {t0, idx}, lparams);
  auto cg_outputs = ke.run({t0, idx}, {}, lparams);

  auto t_gather = at::gather(t0, dim, idx);
  testValidate(
      &fusion,
      cg_outputs,
      {t0, idx},
      {t_gather.sum(0)},
      __LINE__,
      __FILE__,
      "",
      lparams);
}

// segmented, pointwise scheduler can't find the reference tv to schedule.
TEST_F(GatherTest, SameTvUsedAsLookupAndIndex1) {
  auto fusion_ptr = std::make_unique<Fusion>();
  Fusion& fusion = *fusion_ptr.get();
  FusionGuard fg(&fusion);

  // Create three input tensors
  auto tv0 = makeContigTensor(2);
  auto tv1 = makeContigTensor(2, DataType::Int);
  auto tv2 = makeContigTensor(2, DataType::Int);
  fusion.addInput(tv1);
  fusion.addInput(tv2);

  auto tv3 = gather(tv0, 1, tv1);
  auto tv4 = gather(tv1, 1, tv2);
  fusion.addOutput(tv3);
  fusion.addOutput(tv4);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  auto options_i = at::TensorOptions().dtype(at::kLong).device(at::kCUDA, 0);

  // Create test tensors
  std::vector<int64_t> dims{4, 6};
  at::Tensor t0 = at::randn(dims, options);
  at::Tensor t1 = at::randint(0, dims[1], dims, options_i);
  at::Tensor t2 = at::randint(0, dims[1], dims, options_i);

  FusionExecutorCache executor_cache(std::move(fusion_ptr));
  auto cg_outputs = executor_cache.runFusionWithInputs({t0, t1, t2});
  // Validate the result
  testValidate(&fusion, cg_outputs, {t0, t1, t2}, __LINE__, __FILE__);
}

TEST_F(GatherTest, SameTvUsedAsLookupAndIndex2) {
  auto fusion_ptr = std::make_unique<Fusion>();
  Fusion& fusion = *fusion_ptr.get();
  FusionGuard fg(&fusion);

  // Create three input tensors
  auto tv0 = makeContigTensor(2);
  auto tv1 = makeContigTensor(2, DataType::Int);
  auto tv2 = makeContigTensor(2, DataType::Int);
  fusion.addInput(tv0);
  fusion.addInput(tv1);
  fusion.addInput(tv2);

  auto tv3 = gather(tv0, 1, tv1);
  auto tv4 = gather(tv1, 1, tv2);
  auto tv5 = castOp(DataType::Float, tv4);
  auto tv6 = add(tv3, tv5);
  fusion.addOutput(tv6);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  auto options_i = at::TensorOptions().dtype(at::kLong).device(at::kCUDA, 0);

  // Create test tensors
  std::vector<int64_t> dims{4, 6};
  at::Tensor t0 = at::randn(dims, options);
  at::Tensor t1 = at::randint(0, dims[1], dims, options_i);
  at::Tensor t2 = at::randint(0, dims[1], dims, options_i);

  FusionExecutorCache executor_cache(std::move(fusion_ptr));
  auto cg_outputs = executor_cache.runFusionWithInputs({t0, t1, t2});

  auto runtime = executor_cache.getMostRecentKernelRuntime();
  auto scheduled_fusion = runtime->executors()
                              .back()
                              ->as<KernelExecutor>()
                              ->compiledKernel()
                              ->kernel();
  auto tv1_uses = scheduled_fusion->inputs().at(1)->uses();
  EXPECT_EQ(tv1_uses.size(), 2);
  EXPECT_THAT(
      tv1_uses,
      testing::UnorderedElementsAre(
          testing::Truly([](Expr* e) { return e->isA<GatherOp>(); }),
          testing::Truly([](Expr* e) { return e->isA<LoadStoreOp>(); })));

  // Validate the result
  testValidate(&fusion, cg_outputs, {t0, t1, t2}, __LINE__, __FILE__);
}
} // namespace nvfuser
