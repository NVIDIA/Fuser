// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#include <fusion.h>
#include <ir/all_nodes.h>
#include <ops/all_ops.h>
#include <ops/arith.h>

#include <benchmark/benchmark.h>
#include <benchmark/utils.h>
#include <test/utils.h>

using namespace nvfuser;

//------------------------------------------------------------------------------

class ManyPointwiseOpsFixture : public benchmark::Fixture {
 public:
  void SetUp(const ::benchmark::State& state) override {
    fusion_ = std::make_unique<Fusion>();
    FusionGuard fg(fusion_.get());

    auto x = makeContigTensor(2);
    fusion_->addInput(x);

    for (int i = 0; i < state.range(0); ++i) {
      x = add(x, x);
    }

    fusion_->addOutput(x);
  }

  void TearDown(const ::benchmark::State& state) override {
    assert(fusion_.get() != nullptr);
    fusion_.reset();
  }

  ~ManyPointwiseOpsFixture() override {
    assert(fusion_ == nullptr);
  }

  std::unique_ptr<Fusion> fusion_ = nullptr;
};

BENCHMARK_DEFINE_F(ManyPointwiseOpsFixture, ManyPointwiseOpsCopyTest)
(benchmark::State& state) {
  for (auto _ : state) {
    Fusion fcopy = *fusion_;
  }
  state.SetComplexityN(state.range(0));
}

BENCHMARK_REGISTER_F(ManyPointwiseOpsFixture, ManyPointwiseOpsCopyTest)
    ->RangeMultiplier(2)
    ->Range(1 << 3, 1 << 12)
    ->Complexity();
