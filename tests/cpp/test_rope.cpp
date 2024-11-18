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

#include <fusion.h>
#include <fusion_profiler.h>
#include <ops/all_ops.h>
#include <preseg_passes/mark_aliases_prepare.h>
#include <preseg_passes/optimization_pass.h>
#include <runtime/executor.h>
#include <runtime/executor_utils.h>
#include <runtime/fusion_executor_cache.h>
#include <scheduler/tools/inlining.h>
#include <scheduler/tools/loop_domain_scheduler.h>
#include <scheduler/tools/resize_utils.h>
#include <tests/cpp/utils.h>
#include <tests/cpp/validator.h>

namespace nvfuser {

struct RopeConfig {
  int64_t n_head = -1;
  int64_t head_size = -1;
  int64_t n_query_groups = -1;
  int64_t rope_n_elem = -1;
  int64_t batches = -1;
  int64_t seq_length = -1;

  void verify() const {
    ASSERT_EQ(n_head % n_query_groups, 0);
  }

  std::string toString() const {
    std::stringstream ss;
    ss << "{n_head: " << n_head << ", head_size: " << head_size
       << ", n_query_groups: " << n_query_groups
       << ", rope_n_elem: " << rope_n_elem << ", batches: " << batches
       << ", seq_length: " << seq_length << "}";
    return ss.str();
  }

  std::string toCompactString() const {
    std::stringstream ss;
    ss << n_head << "_" << head_size << "_" << n_query_groups << "_"
       << rope_n_elem << "_" << batches << "_" << seq_length;
    return ss.str();
  }
};

std::ostream& operator<<(std::ostream& os, const RopeConfig& rope_config) {
  return os << rope_config.toString();
}

using RopeTest = NVFuserFixtureParamTest<RopeConfig>;

INSTANTIATE_TEST_SUITE_P(
    ,
    RopeTest,
    testing::Values(
        RopeConfig{32, 128, 32, 128, 2, 4096}, // Llama2-7b-hf
        RopeConfig{32, 128, 8, 128, 2, 8192}, // Llama3-8B
        RopeConfig{8, 128, 8, 128, 2, 8192},
        RopeConfig{4, 16, 4, 16, 2, 8}), // Small test
                                         // config
    [](const testing::TestParamInfo<RopeConfig>& info) {
      return info.param.toCompactString();
    });

TEST_P(RopeTest, LitGptFwd) {
  EnableOptionsGuard enable_options_guard;
  EnableOptionsGuard::getCurOptions().set(EnableOption::IdModel, {"all"});

  auto fusion_ptr = std::make_unique<Fusion>();
  FusionGuard fg(fusion_ptr.get());
  Fusion& fusion = *fusion_ptr;

  const RopeConfig config = GetParam();
  config.verify();

  int64_t q_per_kv = config.n_head / config.n_query_groups;
  int64_t total_qkv = q_per_kv + 2;

  int64_t rotation_num_splits = 2;

  std::vector<int64_t> shape_before_reshape{
      config.batches,
      config.seq_length,
      config.head_size * (config.n_head + 2 * config.n_query_groups)};
  std::vector<int64_t> shape_before_permutation{
      config.batches,
      config.seq_length,
      config.n_query_groups,
      total_qkv,
      config.head_size};
  std::vector<int64_t> shape_after_permutation{
      config.batches,
      config.n_query_groups,
      total_qkv,
      config.seq_length,
      config.head_size};
  std::vector<int64_t> shape_after_reshape{
      config.batches,
      config.n_query_groups * total_qkv,
      config.seq_length,
      config.head_size};

  std::vector<int64_t> input_shape = shape_before_reshape;

  std::cerr << "input shape: " << input_shape << "\n";

  // qkv after permutation
  auto tv0 = makeContigConcreteTensor(input_shape, DataType::BFloat16);
  fusion.addInput(tv0);

  std::cerr << "Input: " << tv0->toString() << "\n";

  // cos
  auto tv1 = makeContigConcreteTensor(
      {config.seq_length, config.rope_n_elem}, DataType::BFloat16);
  fusion.addInput(tv1);
  auto cos = tv1;

  // sin
  auto tv2 = makeContigConcreteTensor(
      {config.seq_length, config.rope_n_elem}, DataType::BFloat16);
  fusion.addInput(tv2);
  [[maybe_unused]] auto sin = tv2;

  auto zero = fusion.zeroVal();
  [[maybe_unused]] auto one = fusion.oneVal();

  auto qkv = reshape(tv0, shape_before_reshape, shape_before_permutation);
  qkv = permute(qkv, {0, 2, 3, 1, 4});

  std::cerr << "qkv: " << qkv->toString() << "\n";

  std::vector<Slice> slice_default_arg;
  slice_default_arg.reserve(shape_after_permutation.size());
  for (const auto s : shape_after_permutation) {
    slice_default_arg.push_back(Slice{zero, IrBuilder::create<Val>(s)});
  }

  int64_t qkv_slice_dim = 2;

  [[maybe_unused]] auto slice_arg_q = slice_default_arg;
  slice_arg_q[qkv_slice_dim].stop = IrBuilder::create<Val>(total_qkv - 2);

  [[maybe_unused]] auto slice_arg_k = slice_default_arg;
  slice_arg_k[qkv_slice_dim].start = IrBuilder::create<Val>(q_per_kv);
  slice_arg_k[qkv_slice_dim].stop = IrBuilder::create<Val>(total_qkv - 1);

  auto apply_rope = [&](TensorView* x,
                        bool is_q,
                        std::vector<Slice> slice_arg) -> TensorView* {
    [[maybe_unused]] auto x_slice = slice(x, slice_arg);

    std::cerr << "x_slice: " << x_slice->toString() << "\n";

    std::vector<int64_t> cur_shape = shape_after_permutation;
    cur_shape[qkv_slice_dim] = is_q ? q_per_kv : 1;
    std::cerr << "cur_shape: " << cur_shape << "\n";
    std::vector<int64_t> new_shape{
        cur_shape[0],
        config.n_query_groups * (is_q ? q_per_kv : 1),
        config.seq_length,
        config.rope_n_elem};
    std::cerr << "new shape: " << new_shape << "\n";
    x_slice = reshape(x_slice, cur_shape, new_shape);

    std::cerr << "Reshaped x_slice: " << x_slice->toString() << "\n";

    // x1
    std::vector<Slice> x1_slice_arg;
    x1_slice_arg.reserve(new_shape.size());
    for (const auto s : new_shape) {
      x1_slice_arg.push_back(Slice{zero, IrBuilder::create<Val>(s)});
    }

    x1_slice_arg.back().stop =
        IrBuilder::create<Val>(config.rope_n_elem / rotation_num_splits);
    auto x1 = slice(x_slice, x1_slice_arg);
    std::cerr << "x1: " << x1->definition()->toString() << "\n";

    // x2
    auto x2_slice_arg = x1_slice_arg;
    x2_slice_arg.back().start =
        IrBuilder::create<Val>(config.rope_n_elem / rotation_num_splits);
    x2_slice_arg.back().stop = IrBuilder::create<Val>(config.rope_n_elem);
    [[maybe_unused]] auto x2 = slice(x_slice, x2_slice_arg);
    std::cerr << "x2: " << x2->definition()->toString() << "\n";

    TensorView* rotated = nullptr;
    if (getenv("NO_ROTATION")) {
      rotated = cat({x1, x2}, -1);
    } else {
      rotated = cat({x2, x1}, -1);
    }

    [[maybe_unused]] std::vector<bool> bcast_flags(new_shape.size(), false);
    for (auto it = bcast_flags.begin();
         it != bcast_flags.begin() + bcast_flags.size() - 2;
         ++it) {
      *it = true;
    }
    [[maybe_unused]] auto cos_broadcast = broadcast(cos, bcast_flags);
    [[maybe_unused]] auto sin_broadcast = broadcast(sin, bcast_flags);

    TensorView* out = nullptr;
    if (getenv("NO_X1")) {
      out = x_slice;
    } else if (getenv("NO_COS")) {
      out = add(x_slice, rotated);
    } else {
      out = add(mul(x_slice, cos_broadcast), mul(rotated, sin_broadcast));
    }

    std::cerr << "apply_rope_result: " << out->toString() << "\n";

    out = castOp(DataType::BFloat16, out);
    return out;
  };

  [[maybe_unused]] auto q_out = apply_rope(qkv, true, slice_arg_q);
  if (!getenv("NO_Q")) {
    fusion.addOutput(q_out);
  }

  [[maybe_unused]] auto k_out = apply_rope(qkv, false, slice_arg_k);
  if (!getenv("NO_K")) {
    fusion.addOutput(k_out);
  }

  fusion.printMath();

  auto options = at::TensorOptions().dtype(at::kBFloat16).device(at::kCUDA, 0);
  auto t0 = at::randn(input_shape, options);
  auto t1 = at::randn({config.seq_length, config.rope_n_elem}, options);
  auto t2 = at::randn({config.seq_length, config.rope_n_elem}, options);
  std::vector<c10::IValue> inputs({t0, t1, t2});

  FusionExecutorCache executor_cache(std::move(fusion_ptr));
  auto outputs = executor_cache.runFusionWithInputs(inputs);

  if (getenv("BENCHMARK")) {
    [[maybe_unused]] int64_t mem_size = 1;
    for (const auto s : input_shape) {
      mem_size *= s;
    }
    // read and write
    mem_size *= 2;
    mem_size = mem_size / total_qkv;
    int64_t qkv_factor = 0;
    if (!getenv("NO_Q")) {
      qkv_factor += q_per_kv;
    }
    if (!getenv("NO_K")) {
      qkv_factor += 1;
    }
    mem_size *= qkv_factor;
    // sin and cos
    mem_size += config.seq_length * config.rope_n_elem * 2;
    // BFloat16
    mem_size *= 2;

    // ProfilerOptionsGuard::getCurOptions().set(ProfilerOption::Enable);

    for (int i = 0; i < 10; ++i) {
      clearL2Cache();
      // FusionProfiler::start();
      // FusionProfiler::createSegments(1);
      outputs = executor_cache.runFusionWithInputs(inputs);
      // FusionProfiler::stop();
      // auto t = FusionProfiler::lastKernelTime();
      // std::cout << "Elapsed time (us): " << (t * 1000) << "\n";
      // std::cout << "Bandwidth (GB/s): "
      //<< ((float)mem_size * 0.001 * 0.001 * 0.001 / (t * 0.001))
      //<< "\n";
    }
  }

  testValidate(&fusion, outputs, inputs, __LINE__, __FILE__);
}

} // namespace nvfuser
