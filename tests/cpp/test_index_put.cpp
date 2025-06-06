// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#include <csrc/exceptions.h>
#include <gtest/gtest.h>

#include <ops/all_ops.h>
#include <runtime/fusion_executor_cache.h>
#include <tests/cpp/utils.h>
#include <tests/cpp/validator.h>

namespace nvfuser {

struct SizeParams {
  int64_t vocab_size;
  int64_t hidden_size;
  int64_t seq_size;
};

std::vector<SizeParams> generateSizeOneParams() {
  int64_t vocab_size = 1024;
  int64_t hidden_size = 3584;
  int64_t seq_size = 3000;
  std::vector<SizeParams> params;
  for (bool size_one_vocab : {true, false}) {
    for (bool size_one_hidden : {true, false}) {
      for (bool size_one_seq : {true, false}) {
        int64_t vocab = size_one_vocab ? 1 : vocab_size;
        int64_t hidden = size_one_hidden ? 1 : hidden_size;
        int64_t seq = size_one_seq ? 1 : seq_size;
        params.push_back({vocab, hidden, seq});
      }
    }
  }
  return params;
}

class IndexPut : public NVFuserFixtureParamTest<SizeParams> {
 protected:
  void SetUp() override {
    EnableOptionsGuard::getCurOptions().set(EnableOption::IdModel, {"all"});
    NVFuserTest::SetUp();
  }
};

INSTANTIATE_TEST_SUITE_P(
    ,
    IndexPut,
    ::testing::ValuesIn(generateSizeOneParams()));

// Note: The semantics doesn't support broadcast on operands, adding `size 1`
// check just to ensure the ID mapping is done correctly.
TEST_P(IndexPut, AccumulateOpWithBroadcastIDs) {
  auto fusion_ptr = std::make_unique<Fusion>();
  Fusion& fusion = *fusion_ptr.get();
  FusionGuard fg(&fusion);

  auto [vocab, hidden, seq] = GetParam();

  std::vector<int64_t> shape1({seq, hidden});
  std::vector<int64_t> shape2({seq, 1});

  auto tv_value = makeSymbolicTensor(shape1);
  fusion.addInput(tv_value);
  auto tv_index = makeSymbolicTensor(shape2, DataType::Int);
  fusion.addInput(tv_index);
  auto s_vocab = IrBuilder::create<Val>(vocab, DataType::Index);
  std::vector<nvfuser::Val*> buffer_size = {
      s_vocab, tv_value->axis(-1)->extent()};
  auto buf = zeros(buffer_size, DataType::Float, true);
  // TODO: this should be an inplace. handle it when we have codegen support
  auto out = indexPutAccumulate(buf, tv_index, tv_value);
  fusion.addOutput(out);

  // check PairwiseLogicalDomainMap check if tv0 and tv1 map pairwise on
  // position according to `expect_to_map`
  auto map_logical = [](const std::vector<bool>& expect_to_map,
                        TensorView* tv0,
                        TensorView* tv1) {
    std::unordered_map<IterDomain*, IterDomain*> pairwise_map =
        PairwiseLogicalDomainMap(tv0, tv1).mapProducerToConsumer();
    for (auto index : arange(expect_to_map.size())) {
      IterDomain* id0 = tv0->getLogicalDomain().at(index);
      IterDomain* id1 = tv1->getLogicalDomain().at(index);
      EXPECT_EQ(
          pairwise_map.find(id0) != pairwise_map.end() &&
              pairwise_map[id0] == id1,
          expect_to_map[index]);
    }
  };

  // see [ Note -- IndexPutAccumulateOp semantics ]
  // args:
  //     buf      [ ID_indexed_g0, ID_g0 ]
  //     tv_index [ ID_indexing_g1, ID_broadcast ]
  //     tv_value [ ID_indexing_g1, ID_g0 ]
  // output:
  //     out      [ ID_indexed_g0, ID_g0 ]
  map_logical({true, true}, buf, out);
  // depends on the size of ID_g0, it would map to ID_broadcast when hidden is
  // size-1 dimension
  map_logical({false, hidden == 1}, tv_index, out);
  map_logical({false, true}, tv_value, out);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  auto options_i = at::TensorOptions().dtype(at::kLong).device(at::kCUDA, 0);
  auto t_value = at::randn(shape1, options);
  auto t_index = at::randint(0, vocab, shape2, options_i);

  FusionExecutorCache executor_cache(std::move(fusion_ptr));
  auto outputs = executor_cache.runFusionWithInputs({t_value, t_index});

  testValidate(&fusion, outputs, {t_value, t_index}, __LINE__, __FILE__);
}

TEST_F(IndexPut, IndexShuffle) {
  auto fusion_ptr = std::make_unique<Fusion>();
  Fusion& fusion = *fusion_ptr.get();
  FusionGuard fg(&fusion);

  int64_t seq = 1024;
  int64_t hidden = 1024;

  std::vector<int64_t> src_shape({seq, hidden});
  std::vector<int64_t> index_shape({seq});

  auto tv_src = makeSymbolicTensor(src_shape);
  fusion.addInput(tv_src);
  auto tv_index = makeSymbolicTensor(index_shape, DataType::Int);
  fusion.addInput(tv_index);
  // TODO: this should be an inplace. handle it when we have codegen support
  auto out = indexShuffle(tv_index, 0, tv_src);
  fusion.addOutput(out);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  auto t_src = at::randn(src_shape, options);
  // argsort to get unique indices
  auto t_index = at::rand(index_shape).argsort();

  FusionExecutorCache executor_cache(std::move(fusion_ptr));
  auto outputs = executor_cache.runFusionWithInputs({t_src, t_index});

  auto ref = t_src.scatter(0, t_index.unsqueeze(-1).expand_as(t_src),t_src);
  testValidate(&fusion, outputs, {t_src, t_index}, __LINE__, __FILE__);

  // TODO: remove this after codegen for indexShuffle is added
  EXPECT_TRUE(ref.allclose(outputs[0].as<at::Tensor>()));
}

} // namespace nvfuser
