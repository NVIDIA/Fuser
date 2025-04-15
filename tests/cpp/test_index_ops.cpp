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

namespace nvfuser::IndexPutAccumulate {

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
}

class IndexPutAccumulate : public NVFuserFixtureParamTest<SizeParams> {
 protected:
  void SetUp() override {
    EnableOptionsGuard::getCurOptions().set(EnableOption::IdModel, {"all"});
    NVFuserTest::SetUp();
  }
};

INSTANTIATE_TEST_SUITE_P(
    ,
    IndexPutAccumulate,
    ::testing::ValuesIn(generateReshapeReductionParams()));

}

// Note: The semantics doesn't support broadcast on operands, adding `size 1`
// check just to ensure the ID mapping is done correctly.
TEST_P(IndexPutAccumulate, BroadcastIDs) {
  int64_t vocab = size_one_vocab ? 1 : vocab_size;
  int64_t hidden = size_one_hidden ? 1 : hidden_size;
  int64_t seq = size_one_seq ? 1 : seq_size;

  auto fusion_ptr = std::make_unique<Fusion>();
  Fusion& fusion = *fusion_ptr.get();
  FusionGuard fg(&fusion);

  std::vector<int64_t> shape1({seq, hidden});
  std::vector<int64_t> shape2({seq});

  auto tv_value = makeSymbolicTensor(shape1);
  fusion.addInput(tv_value);
  auto tv_index = makeSymbolicTensor(shape2, DataType::Int);
  fusion.addInput(tv_index);
  auto s_vocab = IrBuilder::create<Val>(vocab, DataType::Index);
  std::vector<nvfuser::Val*> buffer_size = {
      s_vocab, tv_value->axis(-1)->extent()};
  auto buf = zeros(buffer_size, DataType::Float, true);
  // this should be an inplace. handle it when we have codegen support
  auto out = indexPutAccumulate(buf, tv_index, tv_value);
  fusion.addOutput(out);

  auto options =
      at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  auto options_i =
      at::TensorOptions().dtype(at::kLong).device(at::kCUDA, 0);
  auto t_value = at::randn(shape1, options);
  auto t_index = at::randint(0, vocab, shape2, options_i);

  FusionExecutorCache executor_cache(std::move(fusion_ptr));
  auto outputs = executor_cache.runFusionWithInputs({t_value, t_index});

  testValidate(&fusion, outputs, {t_value, t_index}, __LINE__, __FILE__);
}

} // namespace nvfuser::IndexPutAccumulate
