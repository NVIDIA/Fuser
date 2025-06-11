// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#include <csrc/exceptions.h>
#include <fusion.h>
#include <gmock/gmock-matchers.h>
#include <gtest/gtest.h>
#include <ir/graphviz.h>
#include <ops/all_ops.h>
#include <runtime/fusion_executor_cache.h>
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

using RopeTest = NVFuserFixtureParamTest<RopeConfig>;

using MistralRopeTest = RopeTest;

INSTANTIATE_TEST_SUITE_P(
    ,
    MistralRopeTest,
    testing::Values(
        RopeConfig{/*n_head=*/32,
                   /*head_size=*/128,
                   /*n_query_groups=*/8,
                   /*rope_n_elem=*/128,
                   /*n_batches=*/1,
                   /*seq_length=*/4096}),
    [](const testing::TestParamInfo<RopeConfig>& info) {
      return info.param.toCompactString();
    });

// Mistral forward before matmul
// clang-format off
/*
def nvfuser_fusion_id0(fd : FusionDefinition) -> None :
    T0 = fd.define_tensor(shape=[1, 4096, 1024], contiguity=[None, True, True], dtype=DataType.BFloat16, is_cpu=False, stride_order=[2, 1, 0])
    T1 = fd.define_tensor(shape=[64], contiguity=[True], dtype=DataType.BFloat16, is_cpu=False, stride_order=[0])
    T2 = fd.define_tensor(shape=[1, 4096], contiguity=[None, True], dtype=DataType.Int, is_cpu=False, stride_order=[1, 0])
    T8 = fd.ops.reshape(T0, new_shape=[1, 4096, 8, 128])
    T9 = fd.ops.permute(T8, dims=[0, 2, 1, 3])
    T14 = fd.ops.broadcast_in_dim(T1, shape=[1, 64, 1], broadcast_dims=[1])
    T15 = fd.ops.cast(T14, dtype=DataType.Float)
    T20 = fd.ops.broadcast_in_dim(T15, shape=[1, 64, 1], broadcast_dims=[0, 1, 2])
    T25 = fd.ops.broadcast_in_dim(T2, shape=[1, 1, 4096], broadcast_dims=[0, 2])
    T26 = fd.ops.cast(T25, dtype=DataType.Float)
    T33 = fd.ops.broadcast_in_dim(T9, shape=[1, 8, 1, 4096, 128], broadcast_dims=[0, 1, 3, 4])
    T40 = fd.ops.broadcast_in_dim(T33, shape=[1, 8, 4, 4096, 128], broadcast_dims=[0, 1, 2, 3, 4])
    T46 = fd.ops.reshape(T40, new_shape=[1, 32, 4096, 128])
    fd.add_output(T20)
    fd.add_output(T26)
    fd.add_output(T46)
*/
// clang-format on
TEST_P(MistralRopeTest, Fwd1) {
  const RopeConfig config = GetParam();
  config.verify();

  const int64_t batch_size = config.batches;
  const int64_t seq_len = config.seq_length;
  const int64_t head_dim = config.head_size;
  const int64_t num_attention_heads = config.n_head;
  const int64_t num_key_value_heads = config.n_query_groups;

  auto fusion_ptr = std::make_unique<Fusion>();
  FusionGuard fg(fusion_ptr.get());
  Fusion& fusion = *fusion_ptr;

  std::vector<int64_t> shape1{
      batch_size, seq_len, head_dim * num_key_value_heads};
  std::vector<int64_t> shape2{head_dim / 2};
  std::vector<int64_t> shape3{batch_size, seq_len};

  auto tv0 = makeContigConcreteTensor(shape1, DataType::BFloat16);
  fusion.addInput(tv0);
  auto tv1 = makeContigConcreteTensor(shape2, DataType::BFloat16);
  fusion.addInput(tv1);
  auto tv2 = makeContigConcreteTensor(shape3, DataType::Int);
  fusion.addInput(tv2);

  // T3
  auto tv8 = reshape(
      tv0, shape1, {batch_size, seq_len, num_key_value_heads, head_dim});
  // T4
  auto tv9 = permute(tv8, {0, 2, 1, 3});
  // T5
  auto tv14 = broadcast(tv1, {true, false, true});
  // T6
  auto tv15 = castOp(DataType::Float, tv14);
  // T7. This is actually converted to just a set op
  auto tv20 = expand(
      tv15,
      std::vector<Val*>{
          IrBuilder::create<Val>(1L),
          IrBuilder::create<Val>(head_dim / 2),
          IrBuilder::create<Val>(1L)});
  // T8
  auto tv25 = broadcast(tv2, {false, true, false});
  // T9
  auto tv26 = castOp(DataType::Float, tv25);
  // T10
  auto tv33 = broadcast(tv9, {false, false, true, false, false});
  // T11
  auto tv40 = expand(
      tv33,
      std::vector<Val*>{
          IrBuilder::create<Val>(batch_size),
          IrBuilder::create<Val>(num_key_value_heads),
          IrBuilder::create<Val>(num_attention_heads / num_key_value_heads),
          IrBuilder::create<Val>(seq_len),
          IrBuilder::create<Val>(head_dim)});
  // T12
  auto tv46 = reshape(
      tv40,
      {batch_size,
       num_key_value_heads,
       num_attention_heads / num_key_value_heads,
       seq_len,
       head_dim},
      {batch_size, num_attention_heads, seq_len, head_dim});
  fusion.addOutput(tv20);
  fusion.addOutput(tv26);
  fusion.addOutput(tv46);

  auto options_float =
      at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  auto options_bf16 =
      at::TensorOptions().dtype(at::kBFloat16).device(at::kCUDA, 0);
  auto t0 = at::randn(shape1, options_bf16);
  auto t1 = at::randn(shape2, options_bf16);
  auto t2 = at::randn(shape3, options_float).to(at::kLong);

  FusionExecutorCache executor_cache(std::move(fusion_ptr));
  auto out_tensors = executor_cache.runFusionWithInputs({t0, t1, t2});
  testValidate(
      executor_cache.fusion(), out_tensors, {t0, t1, t2}, __LINE__, __FILE__);
}

// Mistral forward after matmul
// clang-format off
/*
def nvfuser_fusion_id1(fd : FusionDefinition) -> None :
    T0 = fd.define_tensor(shape=[1, 4096, 4096], contiguity=[None, True, True], dtype=DataType.BFloat16, is_cpu=False, stride_order=[2, 1, 0])
    T1 = fd.define_tensor(shape=[1, 4096, 1024], contiguity=[None, True, True], dtype=DataType.BFloat16, is_cpu=False, stride_order=[2, 1, 0])
    T2 = fd.define_tensor(shape=[1, 64, 4096], contiguity=[None, True, True], dtype=DataType.Float, is_cpu=False, stride_order=[2, 1, 0])
    T8 = fd.ops.reshape(T0, new_shape=[1, 4096, 32, 128])
    T9 = fd.ops.permute(T8, dims=[0, 2, 1, 3])
    T15 = fd.ops.reshape(T1, new_shape=[1, 4096, 8, 128])
    T16 = fd.ops.permute(T15, dims=[0, 2, 1, 3])
    T17 = fd.ops.permute(T2, dims=[0, 2, 1])
    T18 = fd.ops.cat([T17, T17], dim=-1, manual_padding=0)
    T19 = fd.ops.cos(T18)
    T20 = fd.ops.sin(T18)
    T21 = fd.ops.cast(T19, dtype=DataType.BFloat16)
    T22 = fd.ops.cast(T20, dtype=DataType.BFloat16)
    T28 = fd.ops.broadcast_in_dim(T21, shape=[1, 1, 4096, 128], broadcast_dims=[0, 2, 3])
    T34 = fd.ops.broadcast_in_dim(T22, shape=[1, 1, 4096, 128], broadcast_dims=[0, 2, 3])
    T40 = fd.ops.broadcast_in_dim(T28, shape=[1, 32, 4096, 128], broadcast_dims=[0, 1, 2, 3])
    T41 = fd.ops.cast(T9, dtype=DataType.Float)
    T42 = fd.ops.cast(T40, dtype=DataType.Float)
    T43 = fd.ops.mul(T41, T42)
    T59 = fd.ops.slice(T9, start_indices=[0, 0, 0, 0], end_indices=[1, 32, 4096, 64], strides=[1, 1, 1, 1], manual_normalization=0)
    T75 = fd.ops.slice(T9, start_indices=[0, 0, 0, 64], end_indices=[1, 32, 4096, 128], strides=[1, 1, 1, 1], manual_normalization=0)
    T76 = fd.ops.cast(T75, dtype=DataType.Float)
    T77 = fd.ops.neg(T76)
    T78 = fd.ops.cast(T77, dtype=DataType.BFloat16)
    T79 = fd.ops.cat([T78, T59], dim=-1, manual_padding=0)
    T85 = fd.ops.broadcast_in_dim(T34, shape=[1, 32, 4096, 128], broadcast_dims=[0, 1, 2, 3])
    T86 = fd.ops.cast(T79, dtype=DataType.Float)
    T87 = fd.ops.cast(T85, dtype=DataType.Float)
    T88 = fd.ops.mul(T86, T87)
    T89 = fd.ops.add(T43, T88)
    T90 = fd.ops.cast(T89, dtype=DataType.BFloat16)
    T96 = fd.ops.broadcast_in_dim(T28, shape=[1, 8, 4096, 128], broadcast_dims=[0, 1, 2, 3])
    T97 = fd.ops.cast(T16, dtype=DataType.Float)
    T98 = fd.ops.cast(T96, dtype=DataType.Float)
    T99 = fd.ops.mul(T97, T98)
    T115 = fd.ops.slice(T16, start_indices=[0, 0, 0, 0], end_indices=[1, 8, 4096, 64], strides=[1, 1, 1, 1], manual_normalization=0)
    T131 = fd.ops.slice(T16, start_indices=[0, 0, 0, 64], end_indices=[1, 8, 4096, 128], strides=[1, 1, 1, 1], manual_normalization=0)
    T132 = fd.ops.cast(T131, dtype=DataType.Float)
    T133 = fd.ops.neg(T132)
    T134 = fd.ops.cast(T133, dtype=DataType.BFloat16)
    T135 = fd.ops.cat([T134, T115], dim=-1, manual_padding=0)
    T141 = fd.ops.broadcast_in_dim(T34, shape=[1, 8, 4096, 128], broadcast_dims=[0, 1, 2, 3])
    T142 = fd.ops.cast(T135, dtype=DataType.Float)
    T143 = fd.ops.cast(T141, dtype=DataType.Float)
    T144 = fd.ops.mul(T142, T143)
    T145 = fd.ops.add(T99, T144)
    T146 = fd.ops.cast(T145, dtype=DataType.BFloat16)
    T153 = fd.ops.broadcast_in_dim(T146, shape=[1, 8, 1, 4096, 128], broadcast_dims=[0, 1, 3, 4])
    T160 = fd.ops.broadcast_in_dim(T153, shape=[1, 8, 4, 4096, 128], broadcast_dims=[0, 1, 2, 3, 4])
    T166 = fd.ops.reshape(T160, new_shape=[1, 32, 4096, 128])
    fd.add_output(T90)
    fd.add_output(T166)
*/
// clang-format on
TEST_P(MistralRopeTest, Fwd2) {
  const RopeConfig config = GetParam();
  config.verify();

  const int64_t batch_size = config.batches;
  const int64_t seq_len = config.seq_length;
  const int64_t head_dim = config.head_size;
  const int64_t num_attention_heads = config.n_head;
  const int64_t num_key_value_heads = config.n_query_groups;

  auto fusion_ptr = std::make_unique<Fusion>();
  FusionGuard fg(fusion_ptr.get());
  Fusion& fusion = *fusion_ptr;

  std::vector<int64_t> shape1{
      batch_size, seq_len, head_dim * num_attention_heads};
  std::vector<int64_t> shape2{
      batch_size, seq_len, head_dim * num_key_value_heads};
  std::vector<int64_t> shape3{batch_size, head_dim / 2, seq_len};

  auto tv0 = makeContigConcreteTensor(shape1, DataType::BFloat16);
  fusion.addInput(tv0);
  auto tv1 = makeContigConcreteTensor(shape2, DataType::BFloat16);
  fusion.addInput(tv1);
  auto tv2 = makeContigConcreteTensor(shape3, DataType::Float);
  fusion.addInput(tv2);

  // T3
  auto tv8 = reshape(
      tv0, shape1, {batch_size, seq_len, num_attention_heads, head_dim});
  // T4
  auto tv9 = permute(tv8, {0, 2, 1, 3});
  // T5
  auto tv15 = reshape(
      tv1, shape2, {batch_size, seq_len, num_key_value_heads, head_dim});
  // T6
  auto tv16 = permute(tv15, {0, 2, 1, 3});
  // T7
  auto tv17 = permute(tv2, {0, 2, 1});
  // T8 = pad(T7)
  // T9 = pad(T7)
  // T10
  auto tv18 = cat({tv17, tv17}, -1);
  // T11
  auto tv19 = cos(tv18);
  // T12
  auto tv20 = sin(tv18);
  // T13
  auto tv21 = castOp(DataType::BFloat16, tv19);
  // T14
  auto tv22 = castOp(DataType::BFloat16, tv20);
  // T15
  auto tv28 = broadcast(tv21, {false, true, false, false});
  // T16
  auto tv34 = broadcast(tv22, {false, true, false, false});
  // T17
  auto tv40 = expand(
      tv28,
      std::vector<Val*>{
          IrBuilder::create<Val>(batch_size),
          IrBuilder::create<Val>(num_attention_heads),
          IrBuilder::create<Val>(seq_len),
          IrBuilder::create<Val>(head_dim)});
  // T18
  auto tv41 = castOp(DataType::Float, tv9);
  // T19
  auto tv42 = castOp(DataType::Float, tv40);
  // T20
  auto tv43 = mul(tv41, tv42);
  // T21
  auto tv59 = slice(
      tv9,
      {{fusion.zeroVal(), tv9->getLogicalDomain().at(0)->extent()},
       {fusion.zeroVal(), tv9->getLogicalDomain().at(1)->extent()},
       {fusion.zeroVal(), tv9->getLogicalDomain().at(2)->extent()},
       {fusion.zeroVal(), IrBuilder::create<Val>(head_dim / 2)}});
  // T22
  auto tv75 = slice(
      tv9,
      {{fusion.zeroVal(), tv9->getLogicalDomain().at(0)->extent()},
       {fusion.zeroVal(), tv9->getLogicalDomain().at(1)->extent()},
       {fusion.zeroVal(), tv9->getLogicalDomain().at(2)->extent()},
       {IrBuilder::create<Val>(head_dim / 2),
        tv9->getLogicalDomain().at(3)->extent()}});
  // T23
  auto tv76 = castOp(DataType::Float, tv75);
  // T24
  auto tv77 = neg(tv76);
  // T25
  auto tv78 = castOp(DataType::BFloat16, tv77);
  // T26 = pad(T25)
  // T27 = pad(T21)
  // T28
  auto tv79 = cat({tv78, tv59}, -1);
  // T29
  auto tv85 = expand(
      tv34,
      std::vector<Val*>{
          IrBuilder::create<Val>(batch_size),
          IrBuilder::create<Val>(num_attention_heads),
          IrBuilder::create<Val>(seq_len),
          IrBuilder::create<Val>(head_dim)});
  // T30
  auto tv86 = castOp(DataType::Float, tv79);
  // T31
  auto tv87 = castOp(DataType::Float, tv85);
  // T32
  auto tv88 = mul(tv86, tv87);
  // T33
  auto tv89 = add(tv43, tv88);
  // T34
  auto tv90 = castOp(DataType::BFloat16, tv89);

  // T35
  auto tv96 = expand(
      tv28,
      std::vector<Val*>{
          IrBuilder::create<Val>(batch_size),
          IrBuilder::create<Val>(num_key_value_heads),
          IrBuilder::create<Val>(seq_len),
          IrBuilder::create<Val>(head_dim)});
  // T36
  auto tv97 = castOp(DataType::Float, tv16);
  // T37
  auto tv98 = castOp(DataType::Float, tv96);
  // T38
  auto tv99 = mul(tv97, tv98);
  // T39
  auto tv115 = slice(
      tv16,
      {{fusion.zeroVal(), tv16->getLogicalDomain().at(0)->extent()},
       {fusion.zeroVal(), tv16->getLogicalDomain().at(1)->extent()},
       {fusion.zeroVal(), tv16->getLogicalDomain().at(2)->extent()},
       {fusion.zeroVal(), IrBuilder::create<Val>(head_dim / 2)}});
  // T40
  auto tv131 = slice(
      tv16,
      {{fusion.zeroVal(), tv16->getLogicalDomain().at(0)->extent()},
       {fusion.zeroVal(), tv16->getLogicalDomain().at(1)->extent()},
       {fusion.zeroVal(), tv16->getLogicalDomain().at(2)->extent()},
       {IrBuilder::create<Val>(head_dim / 2),
        tv16->getLogicalDomain().at(3)->extent()}});
  // T41
  auto tv132 = castOp(DataType::Float, tv131);
  // T42
  auto tv133 = neg(tv132);
  // T43
  auto tv134 = castOp(DataType::BFloat16, tv133);
  // T44 = pad(T43)
  // T45 = pad(T39)
  // T46
  auto tv135 = cat({tv134, tv115}, -1);
  // T47
  auto tv141 = expand(
      tv34,
      std::vector<Val*>{
          IrBuilder::create<Val>(batch_size),
          IrBuilder::create<Val>(num_key_value_heads),
          IrBuilder::create<Val>(seq_len),
          IrBuilder::create<Val>(head_dim)});
  // T48
  auto tv142 = castOp(DataType::Float, tv135);
  // T49
  auto tv143 = castOp(DataType::Float, tv141);
  // T50
  auto tv144 = mul(tv142, tv143);
  // T51
  auto tv145 = add(tv99, tv144);
  // T52
  auto tv146 = castOp(DataType::BFloat16, tv145);
  // T53
  auto tv153 = broadcast(tv146, {false, false, true, false, false});
  // T54
  auto tv160 = expand(
      tv153,
      std::vector<Val*>{
          IrBuilder::create<Val>(batch_size),
          IrBuilder::create<Val>(num_key_value_heads),
          IrBuilder::create<Val>(num_attention_heads / num_key_value_heads),
          IrBuilder::create<Val>(seq_len),
          IrBuilder::create<Val>(head_dim)});
  // T55
  auto tv166 = reshape(
      tv160,
      std::vector<Val*>{
          IrBuilder::create<Val>(batch_size),
          IrBuilder::create<Val>(num_attention_heads),
          IrBuilder::create<Val>(seq_len),
          IrBuilder::create<Val>(head_dim)});

  fusion.addOutput(tv90);
  fusion.addOutput(tv166);

  auto options_fp32 =
      at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  auto options_bf16 =
      at::TensorOptions().dtype(at::kBFloat16).device(at::kCUDA, 0);
  auto t0 = at::randn(shape1, options_bf16);
  auto t1 = at::randn(shape2, options_bf16);
  auto t2 = at::randn(shape3, options_fp32);

  FusionExecutorCache executor_cache(std::move(fusion_ptr));
  auto out_tensors = executor_cache.runFusionWithInputs({t0, t1, t2});
  testValidate(
      executor_cache.fusion(), out_tensors, {t0, t1, t2}, __LINE__, __FILE__);
}

// clang-format off
/*
def nvfuser_fusion_id2(fd : FusionDefinition) -> None :
    T0 = fd.define_tensor(shape=[1, 32, 4096, 128], contiguity=[None, True, True, True], dtype=DataType.BFloat16, is_cpu=False, stride_order=[3, 2, 1, 0])
    T1 = fd.define_tensor(shape=[1, 64, 4096], contiguity=[None, True, True], dtype=DataType.Float, is_cpu=False, stride_order=[2, 1, 0])
    T2 = fd.define_tensor(shape=[1, 32, 4096, 128], contiguity=[None, True, True, True], dtype=DataType.BFloat16, is_cpu=False, stride_order=[3, 2, 1, 0])
    T3 = fd.define_tensor(shape=[1, 32, 4096, 128], contiguity=[None, True, True, True], dtype=DataType.BFloat16, is_cpu=False, stride_order=[3, 2, 1, 0])
    T10 = fd.ops.reshape(T0, new_shape=[1, 8, 4, 4096, 128])
    T11 = fd.ops.cast(T10, dtype=DataType.Float)
    T12 = fd.ops.sum(T11, dims=[0, 2], keepdim=False, dtype=DataType.Null)
    T13 = fd.ops.permute(T1, dims=[0, 2, 1])
    T14 = fd.ops.cast(T12, dtype=DataType.BFloat16)
    T15 = fd.ops.cat([T13, T13], dim=-1, manual_padding=0)
    T22 = fd.ops.broadcast_in_dim(T14, shape=[1, 8, 1, 4096, 128], broadcast_dims=[1, 3, 4])
    T23 = fd.ops.sin(T15)
    T24 = fd.ops.cast(T22, dtype=DataType.Float)
    T25 = fd.ops.cast(T23, dtype=DataType.BFloat16)
    T26 = fd.ops.sum(T24, dims=[0, 2], keepdim=False, dtype=DataType.Null)
    T32 = fd.ops.broadcast_in_dim(T25, shape=[1, 1, 4096, 128], broadcast_dims=[0, 2, 3])
    T33 = fd.ops.cast(T26, dtype=DataType.BFloat16)
    T39 = fd.ops.broadcast_in_dim(T32, shape=[1, 32, 4096, 128], broadcast_dims=[0, 1, 2, 3])
    T45 = fd.ops.broadcast_in_dim(T33, shape=[1, 8, 4096, 128], broadcast_dims=[1, 2, 3])
    T51 = fd.ops.broadcast_in_dim(T32, shape=[1, 8, 4096, 128], broadcast_dims=[0, 1, 2, 3])
    T52 = fd.ops.cast(T2, dtype=DataType.Float)
    T53 = fd.ops.cast(T39, dtype=DataType.Float)
    T54 = fd.ops.cast(T45, dtype=DataType.Float)
    T55 = fd.ops.cast(T51, dtype=DataType.Float)
    T56 = fd.ops.mul(T53, T52)
    T57 = fd.ops.mul(T55, T54)
    T58 = fd.ops.cast(T56, dtype=DataType.BFloat16)
    T59 = fd.ops.cast(T57, dtype=DataType.BFloat16)
    T75 = fd.ops.slice(T58, start_indices=[0, 0, 0, 0], end_indices=[1, 32, 4096, 64], strides=[1, 1, 1, 1], manual_normalization=0)
    T91 = fd.ops.slice(T59, start_indices=[0, 0, 0, 0], end_indices=[1, 8, 4096, 64], strides=[1, 1, 1, 1], manual_normalization=0)
    T98 = fd.ops.reshape(T3, new_shape=[1, 8, 4, 4096, 128])
    T99 = fd.ops.cos(T15)
    T100 = fd.ops.cast(T75, dtype=DataType.Float)
    T101 = fd.ops.cast(T91, dtype=DataType.Float)
    T102 = fd.ops.cast(T98, dtype=DataType.Float)
    T103 = fd.ops.cast(T99, dtype=DataType.BFloat16)
    T104 = fd.ops.neg(T100)
    T105 = fd.ops.neg(T101)
    T106 = fd.ops.sum(T102, dims=[0, 2], keepdim=False, dtype=DataType.Null)
    T112 = fd.ops.broadcast_in_dim(T103, shape=[1, 1, 4096, 128], broadcast_dims=[0, 2, 3])
    T128 = fd.ops.slice(T58, start_indices=[0, 0, 0, 64], end_indices=[1, 32, 4096, 128], strides=[1, 1, 1, 1], manual_normalization=0)
    T129 = fd.ops.cast(T104, dtype=DataType.BFloat16)
    T145 = fd.ops.slice(T59, start_indices=[0, 0, 0, 64], end_indices=[1, 8, 4096, 128], strides=[1, 1, 1, 1], manual_normalization=0)
    T146 = fd.ops.cast(T105, dtype=DataType.BFloat16)
    T147 = fd.ops.cast(T106, dtype=DataType.BFloat16)
    T153 = fd.ops.broadcast_in_dim(T112, shape=[1, 32, 4096, 128], broadcast_dims=[0, 1, 2, 3])
    S154 = fd.define_scalar(0.00000, dtype=DataType.Double)
    T164 = fd.ops.pad(T128, [0, 64, 0, 0, 0, 0, 0, 0], S154)
    S165 = fd.define_scalar(0.00000, dtype=DataType.Double)
    T175 = fd.ops.pad(T129, [64, 0, 0, 0, 0, 0, 0, 0], S165)
    T181 = fd.ops.broadcast_in_dim(T112, shape=[1, 8, 4096, 128], broadcast_dims=[0, 1, 2, 3])
    S182 = fd.define_scalar(0.00000, dtype=DataType.Double)
    T192 = fd.ops.pad(T145, [0, 64, 0, 0, 0, 0, 0, 0], S182)
    S193 = fd.define_scalar(0.00000, dtype=DataType.Double)
    T203 = fd.ops.pad(T146, [64, 0, 0, 0, 0, 0, 0, 0], S193)
    T210 = fd.ops.broadcast_in_dim(T147, shape=[1, 8, 1, 4096, 128], broadcast_dims=[1, 3, 4])
    T211 = fd.ops.cast(T153, dtype=DataType.Float)
    T212 = fd.ops.cast(T164, dtype=DataType.Float)
    T213 = fd.ops.cast(T175, dtype=DataType.Float)
    T214 = fd.ops.cast(T181, dtype=DataType.Float)
    T215 = fd.ops.cast(T192, dtype=DataType.Float)
    T216 = fd.ops.cast(T203, dtype=DataType.Float)
    T217 = fd.ops.cast(T210, dtype=DataType.Float)
    T218 = fd.ops.mul(T211, T52)
    T219 = fd.ops.add(T213, T212)
    T220 = fd.ops.mul(T214, T54)
    T221 = fd.ops.add(T216, T215)
    T222 = fd.ops.sum(T217, dims=[0, 2], keepdim=False, dtype=DataType.Null)
    T223 = fd.ops.add(T219, T218)
    T224 = fd.ops.add(T221, T220)
    T225 = fd.ops.cast(T222, dtype=DataType.BFloat16)
    T226 = fd.ops.cast(T223, dtype=DataType.BFloat16)
    T227 = fd.ops.cast(T224, dtype=DataType.BFloat16)
    T233 = fd.ops.broadcast_in_dim(T225, shape=[1, 8, 4096, 128], broadcast_dims=[1, 2, 3])
    T234 = fd.ops.permute(T226, dims=[0, 2, 1, 3])
    T235 = fd.ops.permute(T227, dims=[0, 2, 1, 3])
    T236 = fd.ops.permute(T233, dims=[0, 2, 1, 3])
    T241 = fd.ops.reshape(T234, new_shape=[1, 4096, 4096])
    T246 = fd.ops.reshape(T235, new_shape=[1, 4096, 1024])
    T251 = fd.ops.reshape(T236, new_shape=[1, 4096, 1024])
    fd.add_output(T251)
    fd.add_output(T246)
    fd.add_output(T241)
*/
// clang-format on
TEST_P(MistralRopeTest, Bwd) {
  const RopeConfig config = GetParam();
  config.verify();

  const int64_t batch_size = config.batches;
  const int64_t seq_len = config.seq_length;
  const int64_t head_dim = config.head_size;
  const int64_t num_attention_heads = config.n_head;
  const int64_t num_key_value_heads = config.n_query_groups;

  std::vector<int64_t> shape0{
      batch_size, num_attention_heads, seq_len, head_dim};
  std::vector<int64_t> shape1{batch_size, head_dim / 2, seq_len};
  std::vector<int64_t> shape2{
      batch_size, num_attention_heads, seq_len, head_dim};
  std::vector<int64_t> shape3{
      batch_size, num_attention_heads, seq_len, head_dim};

  auto fusion_ptr = std::make_unique<Fusion>();
  FusionGuard fg(fusion_ptr.get());
  Fusion& fusion = *fusion_ptr;

  auto T0 = makeContigConcreteTensor(shape0, DataType::BFloat16);
  fusion.addInput(T0);
  auto T1 = makeContigConcreteTensor(shape1, DataType::Float);
  fusion.addInput(T1);
  auto T2 = makeContigConcreteTensor(shape2, DataType::BFloat16);
  fusion.addInput(T2);
  auto T3 = makeContigConcreteTensor(shape3, DataType::BFloat16);
  fusion.addInput(T3);

  auto T10 = reshape(
      T0,
      std::vector<Val*>{
          IrBuilder::create<Val>(batch_size),
          IrBuilder::create<Val>(num_key_value_heads),
          IrBuilder::create<Val>(num_attention_heads / num_key_value_heads),
          IrBuilder::create<Val>(seq_len),
          IrBuilder::create<Val>(head_dim)});
  auto T11 = castOp(DataType::Float, T10);
  auto T12 = sum(T11, {0, 2});
  auto T13 = permute(T1, {0, 2, 1});
  auto T14 = castOp(DataType::BFloat16, T12);
  auto T15 = cat({T13, T13}, -1);
  auto T22 = broadcast(T14, {true, false, true, false, false});
  auto T23 = sin(T15);
  auto T24 = castOp(DataType::Float, T22);
  auto T25 = castOp(DataType::BFloat16, T23);
  auto T26 = sum(T24, {0, 2});
  auto T32 = broadcast(T25, {false, true, false, false});
  auto T33 = castOp(DataType::BFloat16, T26);
  auto T39 = expand(
      T32,
      std::vector<Val*>{
          IrBuilder::create<Val>(batch_size),
          IrBuilder::create<Val>(num_attention_heads),
          IrBuilder::create<Val>(seq_len),
          IrBuilder::create<Val>(head_dim)});
  auto T45 = broadcast(T33, {true, false, false, false});
  auto T51 = expand(
      T32,
      std::vector<Val*>{
          IrBuilder::create<Val>(batch_size),
          IrBuilder::create<Val>(num_key_value_heads),
          IrBuilder::create<Val>(seq_len),
          IrBuilder::create<Val>(head_dim)});
  auto T52 = castOp(DataType::Float, T2);
  auto T53 = castOp(DataType::Float, T39);
  auto T54 = castOp(DataType::Float, T45);
  auto T55 = castOp(DataType::Float, T51);
  auto T56 = mul(T53, T52);
  auto T57 = mul(T55, T54);
  auto T58 = castOp(DataType::BFloat16, T56);
  auto T59 = castOp(DataType::BFloat16, T57);
  auto T75 = slice(
      T58,
      {{IrBuilder::create<Val>(0L), IrBuilder::create<Val>(batch_size)},
       {IrBuilder::create<Val>(0L),
        IrBuilder::create<Val>(num_attention_heads)},
       {IrBuilder::create<Val>(0L), IrBuilder::create<Val>(seq_len)},
       {IrBuilder::create<Val>(0L), IrBuilder::create<Val>(head_dim / 2)}});
  auto T91 = slice(
      T59,
      {{IrBuilder::create<Val>(0L), IrBuilder::create<Val>(batch_size)},
       {IrBuilder::create<Val>(0L),
        IrBuilder::create<Val>(num_key_value_heads)},
       {IrBuilder::create<Val>(0L), IrBuilder::create<Val>(seq_len)},
       {IrBuilder::create<Val>(0L), IrBuilder::create<Val>(head_dim / 2)}});
  auto T98 = reshape(
      T3,
      std::vector<Val*>{
          IrBuilder::create<Val>(batch_size),
          IrBuilder::create<Val>(num_key_value_heads),
          IrBuilder::create<Val>(num_attention_heads / num_key_value_heads),
          IrBuilder::create<Val>(seq_len),
          IrBuilder::create<Val>(head_dim)});
  auto T99 = cos(T15);
  auto T100 = castOp(DataType::Float, T75);
  auto T101 = castOp(DataType::Float, T91);
  auto T102 = castOp(DataType::Float, T98);
  auto T103 = castOp(DataType::BFloat16, T99);
  auto T104 = neg(T100);
  auto T105 = neg(T101);
  auto T106 = sum(T102, {0, 2});
  auto T112 = broadcast(T103, {false, true, false, false});
  auto T128 = slice(
      T58,
      {{IrBuilder::create<Val>(0L), IrBuilder::create<Val>(batch_size)},
       {IrBuilder::create<Val>(0L),
        IrBuilder::create<Val>(num_attention_heads)},
       {IrBuilder::create<Val>(0L), IrBuilder::create<Val>(seq_len)},
       {IrBuilder::create<Val>(head_dim / 2),
        IrBuilder::create<Val>(head_dim)}});
  auto T129 = castOp(DataType::BFloat16, T104);
  auto T145 = slice(
      T59,
      {{IrBuilder::create<Val>(0L), IrBuilder::create<Val>(batch_size)},
       {IrBuilder::create<Val>(0L),
        IrBuilder::create<Val>(num_key_value_heads)},
       {IrBuilder::create<Val>(0L), IrBuilder::create<Val>(seq_len)},
       {IrBuilder::create<Val>(head_dim / 2),
        IrBuilder::create<Val>(head_dim)}});
  auto T146 = castOp(DataType::BFloat16, T105);
  auto T147 = castOp(DataType::BFloat16, T106);
  auto T153 = expand(
      T112,
      std::vector<Val*>{
          IrBuilder::create<Val>(batch_size),
          IrBuilder::create<Val>(num_attention_heads),
          IrBuilder::create<Val>(seq_len),
          IrBuilder::create<Val>(head_dim)});
  auto T164 = pad(
      T128, {IrBuilder::create<Val>(0L), IrBuilder::create<Val>(head_dim / 2)});
  auto T175 = pad(
      T129, {IrBuilder::create<Val>(head_dim / 2), IrBuilder::create<Val>(0L)});
  auto T181 = expand(
      T112,
      std::vector<Val*>{
          IrBuilder::create<Val>(batch_size),
          IrBuilder::create<Val>(num_key_value_heads),
          IrBuilder::create<Val>(seq_len),
          IrBuilder::create<Val>(head_dim)});
  auto T192 = pad(
      T145, {IrBuilder::create<Val>(0L), IrBuilder::create<Val>(head_dim / 2)});
  auto T203 = pad(
      T146, {IrBuilder::create<Val>(head_dim / 2), IrBuilder::create<Val>(0L)});
  auto T210 = broadcast(T147, {true, false, true, false, false});
  auto T211 = castOp(DataType::Float, T153);
  auto T212 = castOp(DataType::Float, T164);
  auto T213 = castOp(DataType::Float, T175);
  auto T214 = castOp(DataType::Float, T181);
  auto T215 = castOp(DataType::Float, T192);
  auto T216 = castOp(DataType::Float, T203);
  auto T217 = castOp(DataType::Float, T210);
  auto T218 = mul(T211, T52);
  auto T219 = add(T213, T212);
  auto T220 = mul(T214, T54);
  auto T221 = add(T216, T215);
  auto T222 = sum(T217, {0, 2});
  auto T223 = add(T219, T218);
  auto T224 = add(T221, T220);
  auto T225 = castOp(DataType::BFloat16, T222);
  auto T226 = castOp(DataType::BFloat16, T223);
  auto T227 = castOp(DataType::BFloat16, T224);
  auto T233 = broadcast(T225, {true, false, false, false});
  auto T234 = permute(T226, {0, 2, 1, 3});
  auto T235 = permute(T227, {0, 2, 1, 3});
  auto T236 = permute(T233, {0, 2, 1, 3});
  auto T241 = reshape(
      T234,
      std::vector<Val*>{
          IrBuilder::create<Val>(batch_size),
          IrBuilder::create<Val>(seq_len),
          IrBuilder::create<Val>(num_attention_heads * head_dim)});
  auto T246 = reshape(
      T235,
      std::vector<Val*>{
          IrBuilder::create<Val>(batch_size),
          IrBuilder::create<Val>(seq_len),
          IrBuilder::create<Val>(num_key_value_heads * head_dim)});
  auto T251 = reshape(
      T236,
      std::vector<Val*>{
          IrBuilder::create<Val>(batch_size),
          IrBuilder::create<Val>(seq_len),
          IrBuilder::create<Val>(num_key_value_heads * head_dim)});
  fusion.addOutput(T251);
  fusion.addOutput(T246);
  fusion.addOutput(T241);

  auto options_fp32 =
      at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  auto options_bf16 =
      at::TensorOptions().dtype(at::kBFloat16).device(at::kCUDA, 0);
  auto t0 = at::randn(shape0, options_bf16);
  auto t1 = at::randn(shape1, options_fp32);
  auto t2 = at::randn(shape2, options_bf16);
  auto t3 = at::randn(shape3, options_bf16);

  FusionExecutorCache executor_cache(std::move(fusion_ptr));
  auto out_tensors = executor_cache.runFusionWithInputs({t0, t1, t2, t3});
  testValidate(
      executor_cache.fusion(),
      out_tensors,
      {t0, t1, t2, t3},
      __LINE__,
      __FILE__);
}

using Phi3RopeTest = RopeTest;

INSTANTIATE_TEST_SUITE_P(
    ,
    Phi3RopeTest,
    testing::Values(
        RopeConfig{/*n_head=*/32,
                   /*head_size=*/96,
                   /*n_query_groups=*/32,
                   /*rope_n_elem=*/128,
                   /*n_batches=*/1,
                   /*seq_length=*/8192}),
    [](const testing::TestParamInfo<RopeConfig>& info) {
      return info.param.toCompactString();
    });

// clang-format off
/*
def nvfuser_fusion_id0(fd : FusionDefinition) -> None :
    T0 = fd.define_tensor(shape=[1, 8192, 9216], contiguity=[None, True, True], dtype=DataType.BFloat16, is_cpu=False, stride_order=[2, 1, 0])
    T1 = fd.define_tensor(shape=[48], contiguity=[True], dtype=DataType.BFloat16, is_cpu=False, stride_order=[0])
    T2 = fd.define_tensor(shape=[1, 8192], contiguity=[None, True], dtype=DataType.Int, is_cpu=False, stride_order=[1, 0])
    T15 = fd.ops.slice(T0, start_indices=[0, 0, 0], end_indices=[1, 8192, 3072], strides=[1, 1, 1], manual_normalization=0)
    T28 = fd.ops.slice(T0, start_indices=[0, 0, 3072], end_indices=[1, 8192, 6144], strides=[1, 1, 1], manual_normalization=0)
    T41 = fd.ops.slice(T0, start_indices=[0, 0, 6144], end_indices=[1, 8192, 9216], strides=[1, 1, 1], manual_normalization=0)
    T47 = fd.ops.reshape(T15, new_shape=[1, 8192, 32, 96])
    T48 = fd.ops.permute(T47, dims=[0, 2, 1, 3])
    T54 = fd.ops.reshape(T28, new_shape=[1, 8192, 32, 96])
    T55 = fd.ops.permute(T54, dims=[0, 2, 1, 3])
    T61 = fd.ops.reshape(T41, new_shape=[1, 8192, 32, 96])
    T62 = fd.ops.permute(T61, dims=[0, 2, 1, 3])
    T67 = fd.ops.broadcast_in_dim(T1, shape=[1, 48, 1], broadcast_dims=[1])
    T68 = fd.ops.cast(T67, dtype=DataType.Float)
    T73 = fd.ops.broadcast_in_dim(T68, shape=[1, 48, 1], broadcast_dims=[0, 1, 2])
    T78 = fd.ops.broadcast_in_dim(T2, shape=[1, 1, 8192], broadcast_dims=[0, 2])
    T79 = fd.ops.cast(T78, dtype=DataType.Float)
    T80 = fd.ops.matmul(T73, T79)
    T81 = fd.ops.permute(T80, dims=[0, 2, 1])
    T82 = fd.ops.cat([T81, T81], dim=-1, manual_padding=0)
    T83 = fd.ops.cos(T82)
    T84 = fd.ops.sin(T82)
    T85 = fd.ops.cast(T83, dtype=DataType.BFloat16)
    T86 = fd.ops.cast(T84, dtype=DataType.BFloat16)
    T92 = fd.ops.broadcast_in_dim(T85, shape=[1, 1, 8192, 96], broadcast_dims=[0, 2, 3])
    T98 = fd.ops.broadcast_in_dim(T86, shape=[1, 1, 8192, 96], broadcast_dims=[0, 2, 3])
    T104 = fd.ops.broadcast_in_dim(T92, shape=[1, 32, 8192, 96], broadcast_dims=[0, 1, 2, 3])
    T105 = fd.ops.cast(T48, dtype=DataType.Float)
    T106 = fd.ops.cast(T104, dtype=DataType.Float)
    T107 = fd.ops.mul(T105, T106)
    T123 = fd.ops.slice(T48, start_indices=[0, 0, 0, 0], end_indices=[1, 32, 8192, 48], strides=[1, 1, 1, 1], manual_normalization=0)
    T139 = fd.ops.slice(T48, start_indices=[0, 0, 0, 48], end_indices=[1, 32, 8192, 96], strides=[1, 1, 1, 1], manual_normalization=0)
    T140 = fd.ops.cast(T139, dtype=DataType.Float)
    T141 = fd.ops.neg(T140)
    T142 = fd.ops.cast(T141, dtype=DataType.BFloat16)
    T143 = fd.ops.cat([T142, T123], dim=-1, manual_padding=0)
    T149 = fd.ops.broadcast_in_dim(T98, shape=[1, 32, 8192, 96], broadcast_dims=[0, 1, 2, 3])
    T150 = fd.ops.cast(T143, dtype=DataType.Float)
    T151 = fd.ops.cast(T149, dtype=DataType.Float)
    T152 = fd.ops.mul(T150, T151)
    T153 = fd.ops.add(T107, T152)
    T154 = fd.ops.cast(T153, dtype=DataType.BFloat16)
    T155 = fd.ops.cast(T55, dtype=DataType.Float)
    T156 = fd.ops.mul(T155, T106)
    T172 = fd.ops.slice(T55, start_indices=[0, 0, 0, 0], end_indices=[1, 32, 8192, 48], strides=[1, 1, 1, 1], manual_normalization=0)
    T188 = fd.ops.slice(T55, start_indices=[0, 0, 0, 48], end_indices=[1, 32, 8192, 96], strides=[1, 1, 1, 1], manual_normalization=0)
    T189 = fd.ops.cast(T188, dtype=DataType.Float)
    T190 = fd.ops.neg(T189)
    T191 = fd.ops.cast(T190, dtype=DataType.BFloat16)
    T192 = fd.ops.cat([T191, T172], dim=-1, manual_padding=0)
    T193 = fd.ops.cast(T192, dtype=DataType.Float)
    T194 = fd.ops.mul(T193, T151)
    T195 = fd.ops.add(T156, T194)
    T196 = fd.ops.cast(T195, dtype=DataType.BFloat16)
    fd.add_output(T62)
    fd.add_output(T104)
    fd.add_output(T149)
    fd.add_output(T154)
    fd.add_output(T196)
*/
// clang-format on
TEST_P(Phi3RopeTest, Fwd) {
  const RopeConfig config = GetParam();
  config.verify();

  const int64_t batch_size = config.batches; // 1
  const int64_t seq_len = config.seq_length; // 8192
  const int64_t head_dim = config.head_size; // 96
  const int64_t num_attention_heads = config.n_head; // 32
  const int64_t num_key_value_heads = config.n_query_groups; // 32

  // [1, 8192, 9216]
  // 32 * 96 + 2 * 32 * 96
  std::vector<int64_t> qkv_shape{
      batch_size,
      seq_len,
      num_attention_heads * head_dim + 2 * num_key_value_heads * head_dim};
  std::vector<int64_t> position_ids_shape{batch_size, seq_len};

  auto fusion_ptr = std::make_unique<Fusion>();
  FusionGuard fg(fusion_ptr.get());
  Fusion& fusion = *fusion_ptr;

  auto T0 = makeContigConcreteTensor(qkv_shape, DataType::BFloat16);
  fusion.addInput(T0);
  // Where does this come from?
  auto T1 = makeContigConcreteTensor({head_dim / 2}, DataType::BFloat16);
  fusion.addInput(T1);
  auto T2 = makeContigConcreteTensor(position_ids_shape, DataType::Int);
  fusion.addInput(T2);

  auto T15 = slice(
      T0,
      {{IrBuilder::create<Val>(0L), IrBuilder::create<Val>(qkv_shape.at(0))},
       {IrBuilder::create<Val>(0L), IrBuilder::create<Val>(qkv_shape.at(1))},
       {IrBuilder::create<Val>(0L),
        IrBuilder::create<Val>(head_dim * num_attention_heads)}});
  auto T28 = slice(
      T0,
      {{IrBuilder::create<Val>(0L), IrBuilder::create<Val>(qkv_shape.at(0))},
       {IrBuilder::create<Val>(0L), IrBuilder::create<Val>(qkv_shape.at(1))},
       {IrBuilder::create<Val>(head_dim * num_attention_heads),
        IrBuilder::create<Val>(
            head_dim * (num_attention_heads + num_key_value_heads))}});
  auto T41 = slice(
      T0,
      {{IrBuilder::create<Val>(0L), IrBuilder::create<Val>(qkv_shape.at(0))},
       {IrBuilder::create<Val>(0L), IrBuilder::create<Val>(qkv_shape.at(1))},
       {IrBuilder::create<Val>(
            head_dim * (num_attention_heads + num_key_value_heads)),
        IrBuilder::create<Val>(
            head_dim * (num_attention_heads + 2 * num_key_value_heads))}});
  auto T47 = reshape(
      T15,
      std::vector<Val*>{
          IrBuilder::create<Val>(batch_size),
          IrBuilder::create<Val>(seq_len),
          IrBuilder::create<Val>(num_attention_heads),
          IrBuilder::create<Val>(head_dim)});
  auto T48 = permute(T47, {0, 2, 1, 3});
  auto T54 = reshape(
      T28,
      std::vector<Val*>{
          IrBuilder::create<Val>(batch_size),
          IrBuilder::create<Val>(seq_len),
          IrBuilder::create<Val>(num_key_value_heads),
          IrBuilder::create<Val>(head_dim)});
  auto T55 = permute(T54, {0, 2, 1, 3});
  auto T61 = reshape(
      T41,
      std::vector<Val*>{
          IrBuilder::create<Val>(batch_size),
          IrBuilder::create<Val>(seq_len),
          IrBuilder::create<Val>(num_key_value_heads),
          IrBuilder::create<Val>(head_dim)});
  auto T62 = permute(T61, {0, 2, 1, 3});

  auto T67 = broadcast(T1, {true, false, true});
  auto T68 = castOp(DataType::Float, T67);
  auto T73 = set(T68);
  auto T78 = broadcast(T2, {false, true, false});
  auto T79 = castOp(DataType::Float, T78);
  auto T80 = matmul(T73, T79);
  auto T81 = permute(T80, {0, 2, 1});
  auto T82 = cat({T81, T81}, -1);
  auto T83 = cos(T82);
  auto T84 = sin(T82);
  auto T85 = castOp(DataType::BFloat16, T83);
  auto T86 = castOp(DataType::BFloat16, T84);
  auto T92 = broadcast(T85, {false, true, false, false});
  auto T98 = broadcast(T86, {false, true, false, false});
  auto T104 = expand(
      T92,
      std::vector<Val*>{
          IrBuilder::create<Val>(batch_size),
          IrBuilder::create<Val>(num_attention_heads),
          IrBuilder::create<Val>(seq_len),
          IrBuilder::create<Val>(head_dim)});
  auto T105 = castOp(DataType::Float, T48);
  auto T106 = castOp(DataType::Float, T104);
  auto T107 = mul(T105, T106);
  auto T123 = slice(
      T48,
      {{IrBuilder::create<Val>(0L), IrBuilder::create<Val>(batch_size)},
       {IrBuilder::create<Val>(0L),
        IrBuilder::create<Val>(num_attention_heads)},
       {IrBuilder::create<Val>(0L), IrBuilder::create<Val>(seq_len)},
       {IrBuilder::create<Val>(0L), IrBuilder::create<Val>(head_dim / 2)}});
  auto T139 = slice(
      T48,
      {{IrBuilder::create<Val>(0L), IrBuilder::create<Val>(batch_size)},
       {IrBuilder::create<Val>(0L),
        IrBuilder::create<Val>(num_attention_heads)},
       {IrBuilder::create<Val>(0L), IrBuilder::create<Val>(seq_len)},
       {IrBuilder::create<Val>(head_dim / 2),
        IrBuilder::create<Val>(head_dim)}});
  auto T140 = castOp(DataType::Float, T139);
  auto T141 = neg(T140);
  auto T142 = castOp(DataType::BFloat16, T141);
  auto T143 = cat({T142, T123}, -1);
  auto T149 = expand(
      T98,
      std::vector<Val*>{
          IrBuilder::create<Val>(batch_size),
          IrBuilder::create<Val>(num_attention_heads),
          IrBuilder::create<Val>(seq_len),
          IrBuilder::create<Val>(head_dim)});
  auto T150 = castOp(DataType::Float, T143);
  auto T151 = castOp(DataType::Float, T149);
  auto T152 = mul(T150, T151);
  auto T153 = add(T107, T152);
  auto T154 = castOp(DataType::BFloat16, T153);
  auto T155 = castOp(DataType::Float, T55);
  auto T156 = mul(T155, T106);
  auto T172 = slice(
      T55,
      {{IrBuilder::create<Val>(0L), IrBuilder::create<Val>(batch_size)},
       {IrBuilder::create<Val>(0L),
        IrBuilder::create<Val>(num_attention_heads)},
       {IrBuilder::create<Val>(0L), IrBuilder::create<Val>(seq_len)},
       {IrBuilder::create<Val>(0L), IrBuilder::create<Val>(head_dim / 2)}});
  auto T188 = slice(
      T55,
      {{IrBuilder::create<Val>(0L), IrBuilder::create<Val>(batch_size)},
       {IrBuilder::create<Val>(0L),
        IrBuilder::create<Val>(num_attention_heads)},
       {IrBuilder::create<Val>(0L), IrBuilder::create<Val>(seq_len)},
       {IrBuilder::create<Val>(head_dim / 2),
        IrBuilder::create<Val>(head_dim)}});
  auto T189 = castOp(DataType::Float, T188);
  auto T190 = neg(T189);
  auto T191 = castOp(DataType::BFloat16, T190);
  auto T192 = cat({T191, T172}, -1);
  auto T193 = castOp(DataType::Float, T192);
  auto T194 = mul(T193, T151);
  auto T195 = add(T156, T194);
  auto T196 = castOp(DataType::BFloat16, T195);
  fusion.addOutput(T62);
  fusion.addOutput(T104);
  fusion.addOutput(T149);
  fusion.addOutput(T154);
  fusion.addOutput(T196);

  auto options_bf16 =
      at::TensorOptions().dtype(at::kBFloat16).device(at::kCUDA, 0);
  auto options_int = at::TensorOptions().dtype(at::kLong).device(at::kCUDA, 0);

  auto t0 = at::randn(qkv_shape, options_bf16);
  auto t1 = at::randn({head_dim / 2}, options_bf16);
  auto t2 = at::arange(seq_len, options_int).unsqueeze(0);

  FusionExecutorCache executor_cache(std::move(fusion_ptr));
  auto out_tensors = executor_cache.runFusionWithInputs({t0, t1, t2});
  testValidate(
      executor_cache.fusion(), out_tensors, {t0, t1, t2}, __LINE__, __FILE__);
}

// clang-format off
/*
def nvfuser_fusion_id1(fd : FusionDefinition) -> None :
    T0 = fd.define_tensor(shape=[1, 32, 8192, 96], contiguity=[None, True, True, True], dtype=DataType.BFloat16, is_cpu=False, stride_order=[3, 2, 1, 0])
    T1 = fd.define_tensor(shape=[1, 32, 8192, 96], contiguity=[None, None, True, True], dtype=DataType.BFloat16, is_cpu=False, stride_order=[3, 2, 1, 0])
    T2 = fd.define_tensor(shape=[1, 32, 8192, 96], contiguity=[None, True, True, True], dtype=DataType.BFloat16, is_cpu=False, stride_order=[3, 2, 1, 0])
    T3 = fd.define_tensor(shape=[1, 32, 8192, 96], contiguity=[None, None, True, True], dtype=DataType.BFloat16, is_cpu=False, stride_order=[3, 2, 1, 0])
    T4 = fd.define_tensor(shape=[1, 32, 8192, 96], contiguity=[None, True, True, True], dtype=DataType.BFloat16, is_cpu=False, stride_order=[3, 2, 1, 0])
    T5 = fd.ops.cast(T0, dtype=DataType.Float)
    T6 = fd.ops.cast(T1, dtype=DataType.Float)
    T7 = fd.ops.cast(T2, dtype=DataType.Float)
    T8 = fd.ops.mul(T6, T5)
    T9 = fd.ops.mul(T6, T7)
    T10 = fd.ops.cast(T8, dtype=DataType.BFloat16)
    T11 = fd.ops.cast(T9, dtype=DataType.BFloat16)
    T27 = fd.ops.slice(T10, start_indices=[0, 0, 0, 0], end_indices=[1, 32, 8192, 48], strides=[1, 1, 1, 1], manual_normalization=0)
    T43 = fd.ops.slice(T11, start_indices=[0, 0, 0, 0], end_indices=[1, 32, 8192, 48], strides=[1, 1, 1, 1], manual_normalization=0)
    T44 = fd.ops.cast(T27, dtype=DataType.Float)
    T45 = fd.ops.cast(T43, dtype=DataType.Float)
    T46 = fd.ops.neg(T44)
    T47 = fd.ops.neg(T45)
    T63 = fd.ops.slice(T10, start_indices=[0, 0, 0, 48], end_indices=[1, 32, 8192, 96], strides=[1, 1, 1, 1], manual_normalization=0)
    T64 = fd.ops.cast(T46, dtype=DataType.BFloat16)
    T80 = fd.ops.slice(T11, start_indices=[0, 0, 0, 48], end_indices=[1, 32, 8192, 96], strides=[1, 1, 1, 1], manual_normalization=0)
    T81 = fd.ops.cast(T47, dtype=DataType.BFloat16)
    S82 = fd.define_scalar(0.00000, dtype=DataType.Double)
    T92 = fd.ops.pad(T63, [0, 48, 0, 0, 0, 0, 0, 0], S82)
    S93 = fd.define_scalar(0.00000, dtype=DataType.Double)
    T103 = fd.ops.pad(T64, [48, 0, 0, 0, 0, 0, 0, 0], S93)
    S104 = fd.define_scalar(0.00000, dtype=DataType.Double)
    T114 = fd.ops.pad(T80, [0, 48, 0, 0, 0, 0, 0, 0], S104)
    S115 = fd.define_scalar(0.00000, dtype=DataType.Double)
    T125 = fd.ops.pad(T81, [48, 0, 0, 0, 0, 0, 0, 0], S115)
    T126 = fd.ops.cast(T3, dtype=DataType.Float)
    T127 = fd.ops.cast(T92, dtype=DataType.Float)
    T128 = fd.ops.cast(T103, dtype=DataType.Float)
    T129 = fd.ops.cast(T114, dtype=DataType.Float)
    T130 = fd.ops.cast(T125, dtype=DataType.Float)
    T131 = fd.ops.mul(T126, T5)
    T132 = fd.ops.add(T128, T127)
    T133 = fd.ops.mul(T126, T7)
    T134 = fd.ops.add(T130, T129)
    T135 = fd.ops.add(T132, T131)
    T136 = fd.ops.add(T134, T133)
    T137 = fd.ops.cast(T135, dtype=DataType.BFloat16)
    T138 = fd.ops.cast(T136, dtype=DataType.BFloat16)
    T139 = fd.ops.permute(T137, dims=[0, 2, 1, 3])
    T140 = fd.ops.permute(T4, dims=[0, 2, 1, 3])
    T141 = fd.ops.permute(T138, dims=[0, 2, 1, 3])
    T146 = fd.ops.reshape(T139, new_shape=[1, 8192, 3072])
    T151 = fd.ops.reshape(T140, new_shape=[1, 8192, 3072])
    T156 = fd.ops.reshape(T141, new_shape=[1, 8192, 3072])
    S157 = fd.define_scalar(0.00000, dtype=DataType.Double)
    T165 = fd.ops.pad(T146, [3072, 3072, 0, 0, 0, 0], S157)
    S166 = fd.define_scalar(0.00000, dtype=DataType.Double)
    T174 = fd.ops.pad(T151, [6144, 0, 0, 0, 0, 0], S166)
    S175 = fd.define_scalar(0.00000, dtype=DataType.Double)
    T183 = fd.ops.pad(T156, [0, 6144, 0, 0, 0, 0], S175)
    T184 = fd.ops.cast(T165, dtype=DataType.Float)
    T185 = fd.ops.cast(T174, dtype=DataType.Float)
    T186 = fd.ops.cast(T183, dtype=DataType.Float)
    T187 = fd.ops.add(T185, T184)
    T188 = fd.ops.add(T187, T186)
    T189 = fd.ops.cast(T188, dtype=DataType.BFloat16)
    fd.add_output(T189)
 */
// clang-format on
TEST_P(Phi3RopeTest, Bwd) {
  const RopeConfig config = GetParam();
  config.verify();

  const int64_t batch_size = config.batches; // 1
  const int64_t seq_len = config.seq_length; // 8192
  const int64_t head_dim = config.head_size; // 96
  const int64_t num_attention_heads = config.n_head; // 32

  std::vector<int64_t> shape{
      batch_size, num_attention_heads, seq_len, head_dim};

  auto fusion_ptr = std::make_unique<Fusion>();
  FusionGuard fg(fusion_ptr.get());
  Fusion& fusion = *fusion_ptr;

  auto T0 = makeContigConcreteTensor(shape, DataType::BFloat16);
  fusion.addInput(T0);
  auto T1 = TensorViewBuilder()
                .shape(shape)
                .dtype(DataType::BFloat16)
                .expanded({false, true, false, false})
                .contiguity({std::nullopt, std::nullopt, true, true})
                .build();
  fusion.addInput(T1);
  auto T2 = makeContigConcreteTensor(shape, DataType::BFloat16);
  fusion.addInput(T2);
  auto T3 = TensorViewBuilder()
                .shape(shape)
                .dtype(DataType::BFloat16)
                .expanded({false, true, false, false})
                .contiguity({std::nullopt, std::nullopt, true, true})
                .build();
  fusion.addInput(T3);
  auto T4 = makeContigConcreteTensor(shape, DataType::BFloat16);
  fusion.addInput(T4);

  auto T5 = castOp(DataType::Float, T0);
  auto T6 = castOp(DataType::Float, T1);
  auto T7 = castOp(DataType::Float, T2);
  auto T8 = mul(T6, T5);
  auto T9 = mul(T6, T7);
  auto T10 = castOp(DataType::BFloat16, T8);
  auto T11 = castOp(DataType::BFloat16, T9);
  auto T27 = slice(
      T10,
      {{IrBuilder::create<Val>(0L), IrBuilder::create<Val>(batch_size)},
       {IrBuilder::create<Val>(0L),
        IrBuilder::create<Val>(num_attention_heads)},
       {IrBuilder::create<Val>(0L), IrBuilder::create<Val>(seq_len)},
       {IrBuilder::create<Val>(0L), IrBuilder::create<Val>(head_dim / 2)}});
  auto T43 = slice(
      T11,
      {{IrBuilder::create<Val>(0L), IrBuilder::create<Val>(batch_size)},
       {IrBuilder::create<Val>(0L),
        IrBuilder::create<Val>(num_attention_heads)},
       {IrBuilder::create<Val>(0L), IrBuilder::create<Val>(seq_len)},
       {IrBuilder::create<Val>(0L), IrBuilder::create<Val>(head_dim / 2)}});
  auto T44 = castOp(DataType::Float, T27);
  auto T45 = castOp(DataType::Float, T43);
  auto T46 = neg(T44);
  auto T47 = neg(T45);
  auto T63 = slice(
      T10,
      {{IrBuilder::create<Val>(0L), IrBuilder::create<Val>(batch_size)},
       {IrBuilder::create<Val>(0L),
        IrBuilder::create<Val>(num_attention_heads)},
       {IrBuilder::create<Val>(0L), IrBuilder::create<Val>(seq_len)},
       {IrBuilder::create<Val>(head_dim / 2),
        IrBuilder::create<Val>(head_dim)}});
  auto T64 = castOp(DataType::BFloat16, T46);
  auto T80 = slice(
      T11,
      {{IrBuilder::create<Val>(0L), IrBuilder::create<Val>(batch_size)},
       {IrBuilder::create<Val>(0L),
        IrBuilder::create<Val>(num_attention_heads)},
       {IrBuilder::create<Val>(0L), IrBuilder::create<Val>(seq_len)},
       {IrBuilder::create<Val>(head_dim / 2),
        IrBuilder::create<Val>(head_dim)}});
  auto T81 = castOp(DataType::BFloat16, T47);
  auto T92 = pad(
      T63, {IrBuilder::create<Val>(0L), IrBuilder::create<Val>(head_dim / 2)});
  auto T103 = pad(
      T64, {IrBuilder::create<Val>(head_dim / 2), IrBuilder::create<Val>(0L)});
  auto T114 = pad(
      T80, {IrBuilder::create<Val>(0L), IrBuilder::create<Val>(head_dim / 2)});
  auto T125 = pad(
      T81, {IrBuilder::create<Val>(head_dim / 2), IrBuilder::create<Val>(0L)});
  auto T126 = castOp(DataType::Float, T3);
  auto T127 = castOp(DataType::Float, T92);
  auto T128 = castOp(DataType::Float, T103);
  auto T129 = castOp(DataType::Float, T114);
  auto T130 = castOp(DataType::Float, T125);
  auto T131 = mul(T126, T5);
  auto T132 = add(T128, T127);
  auto T133 = mul(T126, T7);
  auto T134 = add(T130, T129);
  auto T135 = add(T132, T131);
  auto T136 = add(T134, T133);
  auto T137 = castOp(DataType::BFloat16, T135);
  auto T138 = castOp(DataType::BFloat16, T136);
  auto T139 = permute(T137, {0, 2, 1, 3});
  auto T140 = permute(T4, {0, 2, 1, 3});
  auto T141 = permute(T138, {0, 2, 1, 3});
  auto T146 = reshape(
      T139,
      std::vector<Val*>{
          IrBuilder::create<Val>(batch_size),
          IrBuilder::create<Val>(seq_len),
          IrBuilder::create<Val>(num_attention_heads * head_dim)});
  auto T151 = reshape(
      T140,
      std::vector<Val*>{
          IrBuilder::create<Val>(batch_size),
          IrBuilder::create<Val>(seq_len),
          IrBuilder::create<Val>(num_attention_heads * head_dim)});

  auto T156 = reshape(
      T141,
      std::vector<Val*>{
          IrBuilder::create<Val>(batch_size),
          IrBuilder::create<Val>(seq_len),
          IrBuilder::create<Val>(num_attention_heads * head_dim)});
  auto T165 =
      pad(T146,
          {IrBuilder::create<Val>(head_dim * num_attention_heads),
           IrBuilder::create<Val>(head_dim * num_attention_heads)});
  auto T174 =
      pad(T151,
          {IrBuilder::create<Val>(head_dim * num_attention_heads * 2),
           IrBuilder::create<Val>(0L)});
  auto T183 =
      pad(T156,
          {IrBuilder::create<Val>(0L),
           IrBuilder::create<Val>(head_dim * num_attention_heads * 2)});
  auto T184 = castOp(DataType::Float, T165);
  auto T185 = castOp(DataType::Float, T174);
  auto T186 = castOp(DataType::Float, T183);
  auto T187 = add(T185, T184);
  auto T188 = add(T187, T186);
  auto T189 = castOp(DataType::BFloat16, T188);
  fusion.addOutput(T189);

  auto options_bf16 =
      at::TensorOptions().dtype(at::kBFloat16).device(at::kCUDA, 0);

  auto t0 = at::randn(shape, options_bf16);
  auto t1 = at::randn({seq_len, head_dim}, options_bf16)
                .as_strided({shape}, {0, 0, head_dim, 1});
  auto t2 = at::randn(shape, options_bf16);
  auto t3 = at::randn({seq_len, head_dim}, options_bf16)
                .as_strided({shape}, {0, 0, head_dim, 1});
  auto t4 = at::randn(shape, options_bf16);

  FusionExecutorCache executor_cache(std::move(fusion_ptr));
  auto out_tensors = executor_cache.runFusionWithInputs({t0, t1, t2, t3, t4});
  testValidate(
      executor_cache.fusion(),
      out_tensors,
      {t0, t1, t2, t3, t4},
      __LINE__,
      __FILE__);
}

using LitgptRopeTest = RopeTest;

INSTANTIATE_TEST_SUITE_P(
    ,
    LitgptRopeTest,
    testing::Values(
        RopeConfig{32, 128, 32, 128, 2, 4096}, // Llama2-7b-hf
        RopeConfig{32, 128, 8, 128, 2, 8192}, // Llama3-8B
        RopeConfig{4, 16, 4, 16, 2, 8}, // Small test config
        RopeConfig{8, 16, 4, 16, 2, 8} // Small test config
        ),
    [](const testing::TestParamInfo<RopeConfig>& info) {
      return info.param.toCompactString();
    });

TEST_P(LitgptRopeTest, Fwd) {
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

  const auto& input_shape = shape_before_reshape;

  // qkv after permutation
  auto tv0 = makeContigConcreteTensor(input_shape, DataType::BFloat16);
  fusion.addInput(tv0);

  // cos
  auto tv1 = makeContigConcreteTensor(
      {config.seq_length, config.rope_n_elem}, DataType::BFloat16);
  fusion.addInput(tv1);
  auto cos = tv1;

  // sin
  auto tv2 = makeContigConcreteTensor(
      {config.seq_length, config.rope_n_elem}, DataType::BFloat16);
  fusion.addInput(tv2);
  auto sin = tv2;

  auto zero = fusion.zeroVal();

  auto qkv = reshape(tv0, shape_before_reshape, shape_before_permutation);
  qkv = permute(qkv, {0, 2, 3, 1, 4});

  std::vector<Slice> slice_default_arg;
  slice_default_arg.reserve(shape_after_permutation.size());
  for (const auto s : shape_after_permutation) {
    slice_default_arg.push_back(Slice{zero, IrBuilder::create<Val>(s)});
  }

  int64_t qkv_slice_dim = 2;

  auto slice_arg_q = slice_default_arg;
  slice_arg_q[qkv_slice_dim].stop = IrBuilder::create<Val>(total_qkv - 2);

  auto slice_arg_k = slice_default_arg;
  slice_arg_k[qkv_slice_dim].start = IrBuilder::create<Val>(q_per_kv);
  slice_arg_k[qkv_slice_dim].stop = IrBuilder::create<Val>(total_qkv - 1);

  auto apply_rope = [&](TensorView* x,
                        bool is_q,
                        std::vector<Slice> slice_arg) -> TensorView* {
    auto x_slice = slice(x, slice_arg);

    std::vector<int64_t> cur_shape = shape_after_permutation;
    cur_shape[qkv_slice_dim] = is_q ? q_per_kv : 1;
    std::vector<int64_t> new_shape{
        cur_shape[0],
        config.n_query_groups * (is_q ? q_per_kv : 1),
        config.seq_length,
        config.rope_n_elem};
    x_slice = reshape(x_slice, cur_shape, new_shape);

    // x1
    std::vector<Slice> x1_slice_arg;
    x1_slice_arg.reserve(new_shape.size());
    for (const auto s : new_shape) {
      x1_slice_arg.push_back(Slice{zero, IrBuilder::create<Val>(s)});
    }

    x1_slice_arg.back().stop =
        IrBuilder::create<Val>(config.rope_n_elem / rotation_num_splits);
    auto x1 = slice(x_slice, x1_slice_arg);

    // x2
    auto x2_slice_arg = x1_slice_arg;
    x2_slice_arg.back().start =
        IrBuilder::create<Val>(config.rope_n_elem / rotation_num_splits);
    x2_slice_arg.back().stop = IrBuilder::create<Val>(config.rope_n_elem);
    auto x2 = slice(x_slice, x2_slice_arg);

    auto rotated = cat({x2, x1}, -1);

    std::vector<bool> bcast_flags(new_shape.size(), false);
    for (auto it = bcast_flags.begin();
         it != bcast_flags.begin() + (int64_t)bcast_flags.size() - 2;
         ++it) {
      *it = true;
    }
    auto cos_broadcast = broadcast(cos, bcast_flags);
    auto sin_broadcast = broadcast(sin, bcast_flags);

    TensorView* out =
        add(mul(x_slice, cos_broadcast), mul(rotated, sin_broadcast));
    out = castOp(DataType::BFloat16, out);
    return out;
  };

  auto q_out = apply_rope(qkv, true, slice_arg_q);
  fusion.addOutput(q_out);

  auto k_out = apply_rope(qkv, false, slice_arg_k);
  fusion.addOutput(k_out);

  auto options = at::TensorOptions().dtype(at::kBFloat16).device(at::kCUDA, 0);
  auto t0 = at::randn(input_shape, options);
  auto t1 = at::randn({config.seq_length, config.rope_n_elem}, options);
  auto t2 = at::randn({config.seq_length, config.rope_n_elem}, options);

  FusionExecutorCache executor_cache(std::move(fusion_ptr));
  auto outputs = executor_cache.runFusionWithInputs({t0, t1, t2});
  testValidate(&fusion, outputs, {t0, t1, t2}, __LINE__, __FILE__);
}

// clang-format off
/*
def nvfuser_fusion_id1(fd : FusionDefinition) -> None :
    T0 = fd.define_tensor(shape=[8192, 128], contiguity=[True, True], dtype=DataType.BFloat16, is_cpu=False, stride_order=[1, 0])
    T1 = fd.define_tensor(shape=[2, 32, 8192, 128], contiguity=[True, True, True, True], dtype=DataType.BFloat16, is_cpu=False, stride_order=[3, 2, 1, 0])
    T2 = fd.define_tensor(shape=[2, 32, 8192, 128], contiguity=[True, True, True, True], dtype=DataType.BFloat16, is_cpu=False, stride_order=[3, 2, 1, 0])
    T3 = fd.define_tensor(shape=[8192, 128], contiguity=[True, True], dtype=DataType.BFloat16, is_cpu=False, stride_order=[1, 0])
    T8 = fd.ops.broadcast_in_dim(T0, shape=[1, 8192, 128], broadcast_dims=[1, 2])
    T24 = fd.ops.slice(T1, start_indices=[0, 0, 0, 0], end_indices=[2, 32, 8192, 128], strides=[1, 1, 1, 1], manual_normalization=0)
    T30 = fd.ops.broadcast_in_dim(T8, shape=[2, 32, 8192, 128], broadcast_dims=[1, 2, 3])
    T31 = fd.ops.cast(T24, dtype=DataType.Float)
    T32 = fd.ops.cast(T30, dtype=DataType.Float)
    T33 = fd.ops.mul(T32, T31)
    T34 = fd.ops.cast(T33, dtype=DataType.BFloat16)
    T50 = fd.ops.slice(T34, start_indices=[0, 0, 0, 0], end_indices=[2, 32, 8192, 64], strides=[1, 1, 1, 1], manual_normalization=0)
    T66 = fd.ops.slice(T2, start_indices=[0, 0, 0, 0], end_indices=[2, 32, 8192, 128], strides=[1, 1, 1, 1], manual_normalization=0)
    T67 = fd.ops.cast(T50, dtype=DataType.Float)
    T72 = fd.ops.broadcast_in_dim(T3, shape=[1, 8192, 128], broadcast_dims=[1, 2])
    T73 = fd.ops.cast(T66, dtype=DataType.Float)
    T74 = fd.ops.neg(T67)
    T80 = fd.ops.broadcast_in_dim(T72, shape=[2, 32, 8192, 128], broadcast_dims=[1, 2, 3])
    S81 = fd.define_scalar(0, dtype=DataType.Int)
    T87 = fd.ops.full(shape=[2, 32, 8192, 0], fill_value=S81, dtype=DataType.BFloat16)
    T88 = fd.ops.mul(T32, T73)
    T89 = fd.ops.cast(T74, dtype=DataType.BFloat16)
    T90 = fd.ops.cast(T80, dtype=DataType.Float)
    S91 = fd.define_scalar(0.00000, dtype=DataType.Double)
    T101 = fd.ops.pad(T87, [0, 128, 0, 0, 0, 0, 0, 0], S91)
    T102 = fd.ops.cast(T88, dtype=DataType.BFloat16)
    T118 = fd.ops.slice(T34, start_indices=[0, 0, 0, 64], end_indices=[2, 32, 8192, 128], strides=[1, 1, 1, 1], manual_normalization=0)
    S119 = fd.define_scalar(0.00000, dtype=DataType.Double)
    T129 = fd.ops.pad(T89, [64, 0, 0, 0, 0, 0, 0, 0], S119)
    T130 = fd.ops.mul(T90, T31)
    T131 = fd.ops.cast(T101, dtype=DataType.Float)
    T147 = fd.ops.slice(T102, start_indices=[0, 0, 0, 0], end_indices=[2, 32, 8192, 64], strides=[1, 1, 1, 1], manual_normalization=0)
    S148 = fd.define_scalar(0.00000, dtype=DataType.Double)
    T158 = fd.ops.pad(T118, [0, 64, 0, 0, 0, 0, 0, 0], S148)
    T159 = fd.ops.cast(T129, dtype=DataType.Float)
    T160 = fd.ops.add(T131, T130)
    T161 = fd.ops.cast(T147, dtype=DataType.Float)
    T162 = fd.ops.cast(T158, dtype=DataType.Float)
    T163 = fd.ops.add(T160, T159)
    T164 = fd.ops.neg(T161)
    T165 = fd.ops.add(T163, T162)
    T166 = fd.ops.cast(T164, dtype=DataType.BFloat16)
    T167 = fd.ops.cast(T165, dtype=DataType.BFloat16)
    T183 = fd.ops.slice(T102, start_indices=[0, 0, 0, 64], end_indices=[2, 32, 8192, 128], strides=[1, 1, 1, 1], manual_normalization=0)
    S184 = fd.define_scalar(0.00000, dtype=DataType.Double)
    T194 = fd.ops.pad(T166, [64, 0, 0, 0, 0, 0, 0, 0], S184)
    T195 = fd.ops.mul(T90, T73)
    T202 = fd.ops.reshape(T167, new_shape=[2, 8, 4, 8192, 128])
    S203 = fd.define_scalar(0.00000, dtype=DataType.Double)
    T213 = fd.ops.pad(T183, [0, 64, 0, 0, 0, 0, 0, 0], S203)
    T214 = fd.ops.cast(T194, dtype=DataType.Float)
    T215 = fd.ops.add(T131, T195)
    T216 = fd.ops.cast(T202, dtype=DataType.Float)
    T217 = fd.ops.cast(T213, dtype=DataType.Float)
    T218 = fd.ops.add(T215, T214)
    T219 = fd.ops.sum(T216, dims=[2], keepdim=False, dtype=DataType.Null)
    T220 = fd.ops.add(T218, T217)
    T221 = fd.ops.cast(T219, dtype=DataType.BFloat16)
    T222 = fd.ops.cast(T220, dtype=DataType.BFloat16)
    S223 = fd.define_scalar(0, dtype=DataType.Int)
    T230 = fd.ops.full(shape=[2, 8, 1, 8192, 128], fill_value=S223, dtype=DataType.BFloat16)
    T237 = fd.ops.broadcast_in_dim(T221, shape=[2, 8, 1, 8192, 128], broadcast_dims=[0, 1, 3, 4])
    T244 = fd.ops.reshape(T222, new_shape=[2, 8, 4, 8192, 128])
    T245 = fd.ops.cat([T244, T237, T230], dim=2, manual_padding=0)
    T246 = fd.ops.permute(T245, dims=[0, 3, 1, 2, 4])
    T251 = fd.ops.reshape(T246, new_shape=[2, 8192, 6144])
    fd.add_output(T251)
*/
// clang-format on
TEST_P(LitgptRopeTest, Bwd) {
  auto fusion_ptr = std::make_unique<Fusion>();
  FusionGuard fg(fusion_ptr.get());
  Fusion& fusion = *fusion_ptr;

  const RopeConfig config = GetParam();
  config.verify();

  const auto seq_len = config.seq_length;
  const auto head_dim = config.head_size;
  const auto n_head = config.n_head;
  const auto n_query_groups = config.n_query_groups;
  const auto q_per_kv = n_head / n_query_groups;
  [[maybe_unused]] const auto total_qkv = q_per_kv + 2;

  auto T0 = makeContigConcreteTensor({seq_len, head_dim}, DataType::BFloat16);
  fusion.addInput(T0);
  auto T1 = makeContigConcreteTensor(
      {2, n_head, seq_len, head_dim}, DataType::BFloat16);
  fusion.addInput(T1);
  auto T2 = makeContigConcreteTensor(
      {2, n_head, seq_len, head_dim}, DataType::BFloat16);
  fusion.addInput(T2);
  auto T3 = makeContigConcreteTensor({seq_len, head_dim}, DataType::BFloat16);
  fusion.addInput(T3);

  auto T8 = broadcast(T0, {true, false, false});
  auto T24 = slice(
      T1,
      {{fusion.zeroVal(), T1->getLogicalDomain().at(0)->extent()},
       {fusion.zeroVal(), T1->getLogicalDomain().at(1)->extent()},
       {fusion.zeroVal(), T1->getLogicalDomain().at(2)->extent()},
       {fusion.zeroVal(), T1->getLogicalDomain().at(3)->extent()}});
  auto T30 = expand(
      broadcast(T8, {true, false, false, false}),
      std::vector<Val*>{
          IrBuilder::create<Val>(2L),
          IrBuilder::create<Val>(n_head),
          IrBuilder::create<Val>(-1),
          IrBuilder::create<Val>(-1)});
  auto T31 = castOp(DataType::Float, T24);
  auto T32 = castOp(DataType::Float, T30);
  auto T33 = mul(T32, T31);
  auto T34 = castOp(DataType::BFloat16, T33);
  auto T50 = slice(
      T34,
      {{fusion.zeroVal(), T1->getLogicalDomain().at(0)->extent()},
       {fusion.zeroVal(), T1->getLogicalDomain().at(1)->extent()},
       {fusion.zeroVal(), T1->getLogicalDomain().at(2)->extent()},
       {fusion.zeroVal(), IrBuilder::create<Val>(head_dim / 2)}});
  auto T66 = slice(
      T2,
      {{fusion.zeroVal(), T2->getLogicalDomain().at(0)->extent()},
       {fusion.zeroVal(), T2->getLogicalDomain().at(1)->extent()},
       {fusion.zeroVal(), T2->getLogicalDomain().at(2)->extent()},
       {fusion.zeroVal(), T2->getLogicalDomain().at(3)->extent()}});
  auto T67 = castOp(DataType::Float, T50);
  auto T72 = broadcast(T3, {true, false, false});
  auto T73 = castOp(DataType::Float, T66);
  auto T74 = neg(T67);
  auto T80 = expand(
      broadcast(T72, {true, false, false, false}),
      std::vector<Val*>{
          IrBuilder::create<Val>(2L),
          IrBuilder::create<Val>(n_head),
          IrBuilder::create<Val>(-1),
          IrBuilder::create<Val>(-1)});
  auto T87 = full(
      std::vector<Val*>{
          IrBuilder::create<Val>(2L),
          IrBuilder::create<Val>(n_head),
          IrBuilder::create<Val>(seq_len),
          IrBuilder::create<Val>(0)},
      fusion.zeroVal(DataType::BFloat16),
      DataType::BFloat16);
  auto T88 = mul(T32, T73);
  auto T89 = castOp(DataType::BFloat16, T74);
  auto T90 = castOp(DataType::Float, T80);
  auto T101 = pad(T87, {fusion.zeroVal(), IrBuilder::create<Val>(head_dim)});
  auto T102 = castOp(DataType::BFloat16, T88);
  auto T118 = slice(
      T34,
      {{fusion.zeroVal(), T34->getLogicalDomain().at(0)->extent()},
       {fusion.zeroVal(), T34->getLogicalDomain().at(1)->extent()},
       {fusion.zeroVal(), T34->getLogicalDomain().at(2)->extent()},
       {IrBuilder::create<Val>(head_dim / 2),
        IrBuilder::create<Val>(head_dim)}});
  auto T129 =
      pad(T89, {IrBuilder::create<Val>(head_dim / 2), fusion.zeroVal()});
  auto T130 = mul(T90, T31);
  auto T131 = castOp(DataType::Float, T101);
  auto T147 = slice(
      T102,
      {{fusion.zeroVal(), T102->getLogicalDomain().at(0)->extent()},
       {fusion.zeroVal(), T102->getLogicalDomain().at(1)->extent()},
       {fusion.zeroVal(), T102->getLogicalDomain().at(2)->extent()},
       {fusion.zeroVal(), IrBuilder::create<Val>(head_dim / 2)}});
  auto T158 =
      pad(T118, {fusion.zeroVal(), IrBuilder::create<Val>(head_dim / 2)});
  auto T159 = castOp(DataType::Float, T129);
  auto T160 = add(T131, T130);
  auto T161 = castOp(DataType::Float, T147);
  auto T162 = castOp(DataType::Float, T158);
  auto T163 = add(T160, T159);
  auto T164 = neg(T161);
  auto T165 = add(T163, T162);
  auto T166 = castOp(DataType::BFloat16, T164);
  auto T167 = castOp(DataType::BFloat16, T165);
  auto T183 = slice(
      T102,
      {{fusion.zeroVal(), T102->getLogicalDomain().at(0)->extent()},
       {fusion.zeroVal(), T102->getLogicalDomain().at(1)->extent()},
       {fusion.zeroVal(), T102->getLogicalDomain().at(2)->extent()},
       {IrBuilder::create<Val>(head_dim / 2),
        IrBuilder::create<Val>(head_dim)}});
  auto T194 =
      pad(T166, {IrBuilder::create<Val>(head_dim / 2), fusion.zeroVal()});
  auto T195 = mul(T90, T73);
  auto T202 = reshape(
      T167,
      {IrBuilder::create<Val>(2L),
       IrBuilder::create<Val>(n_query_groups),
       IrBuilder::create<Val>(q_per_kv),
       IrBuilder::create<Val>(seq_len),
       IrBuilder::create<Val>(head_dim)});
  auto T213 =
      pad(T183, {fusion.zeroVal(), IrBuilder::create<Val>(head_dim / 2)});
  auto T214 = castOp(DataType::Float, T194);
  auto T215 = add(T131, T195);
  auto T216 = castOp(DataType::Float, T202);
  auto T217 = castOp(DataType::Float, T213);
  auto T218 = add(T215, T214);
  auto T219 = sum(T216, {2});
  auto T220 = add(T218, T217);
  auto T221 = castOp(DataType::BFloat16, T219);
  auto T222 = castOp(DataType::BFloat16, T220);
  auto T230 = full(
      std::vector<Val*>{
          IrBuilder::create<Val>(2L),
          IrBuilder::create<Val>(n_query_groups),
          IrBuilder::create<Val>(1L),
          IrBuilder::create<Val>(seq_len),
          IrBuilder::create<Val>(head_dim)},
      fusion.zeroVal(DataType::BFloat16),
      DataType::BFloat16);
  auto T237 = broadcast(T221, {false, false, true, false, false});
  auto T244 = reshape(
      T222,
      {IrBuilder::create<Val>(2L),
       IrBuilder::create<Val>(n_query_groups),
       IrBuilder::create<Val>(q_per_kv),
       IrBuilder::create<Val>(seq_len),
       IrBuilder::create<Val>(head_dim)});
  auto T245 = cat({T244, T237, T230}, 2);
  auto T246 = permute(T245, {0, 3, 1, 2, 4});
  auto T251 = reshape(
      T246,
      {IrBuilder::create<Val>(2L),
       IrBuilder::create<Val>(seq_len),
       IrBuilder::create<Val>(head_dim * total_qkv * n_query_groups)});
  fusion.addOutput(T251);

  auto options = at::TensorOptions().dtype(at::kBFloat16).device(at::kCUDA, 0);
  auto t0 = at::randn({seq_len, head_dim}, options);
  auto t1 = at::randn({2, n_head, seq_len, head_dim}, options);
  auto t2 = at::randn({2, n_head, seq_len, head_dim}, options);
  auto t3 = at::randn({seq_len, head_dim}, options);

  FusionExecutorCache executor_cache(std::move(fusion_ptr));
  auto outputs = executor_cache.runFusionWithInputs({t0, t1, t2, t3});
  testValidate(&fusion, outputs, {t0, t1, t2, t3}, __LINE__, __FILE__);

  // Make sure the cat is grouped together with the pad ops of its inputs
  FusionKernelRuntime* runtime = executor_cache.getMostRecentKernelRuntime();
  CatOp* cat = nullptr;
  SegmentedGroup* cat_group = nullptr;
  for (const auto& group : runtime->fusionSegments()->groups()) {
    auto it = std::ranges::find_if(group->exprs(), [&](Expr* expr) {
      return expr->isA<CatOp>() && expr->output(0)->name() == T245->name();
    });
    if (it == group->exprs().end()) {
      continue;
    }
    cat = (*it)->as<CatOp>();
    cat_group = group;
    break;
  }
  EXPECT_NE(cat, nullptr)
      << "Could not find the cat expr in the scheduled segmented fusion";

  // Check if the inputs of `cat({T244, T237, T230}, 2)` are also
  // produced in the same segment
  for (const auto cat_input : cat->inputs()) {
    auto pad = dynamic_cast<PadOp*>(cat_input->definition());
    EXPECT_NE(pad, nullptr)
        << "Unexpected cat input: " << cat_input->toString();
    EXPECT_NE(
        std::ranges::find(cat_group->exprs(), pad), cat_group->exprs().end())
        << "Could not find the input pad in the same segment: "
        << pad->toString();
  }
}

// Testing the scheduling of an ending repeat pattern, which is
// commonly seen in RoPE.
TEST_F(RopeTest, EndingRepeat) {
  auto fusion_ptr = std::make_unique<Fusion>();
  FusionGuard fg(fusion_ptr.get());
  Fusion& fusion = *fusion_ptr;

  std::vector<int64_t> shape1{8, 126};

  auto tv0 = makeContigConcreteTensor(shape1);
  fusion.addInput(tv0);

  auto tv1 = pad(tv0, {fusion.oneVal(), fusion.oneVal()});
  auto tv2 = repeat(tv1, {2, 1});
  auto tv3 = segment_set(tv2);
  fusion.addOutput(tv3);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  auto t0 = at::randn(shape1, options);

  FusionExecutorCache executor_cache(std::move(fusion_ptr));
  auto outputs = executor_cache.runFusionWithInputs({t0});
  testValidate(&fusion, outputs, {t0}, __LINE__, __FILE__);

  FusionKernelRuntime* runtime = executor_cache.getMostRecentKernelRuntime();
  EXPECT_FALSE(runtime->isSegmented());
  const auto& heuristic_param =
      runtime->schedulerHeuristics()->heuristicsList().front();
  EXPECT_EQ(heuristic_param->scheduler_type, SchedulerType::Resize);
  Fusion* scheduled_fusion = runtime->executors()
                                 .at(0)
                                 ->as<KernelExecutor>()
                                 ->compiledKernel()
                                 ->kernel();

  // Check the loop domain of the reference. It should look like:
  //
  // T4_g_float[iS19{2 ex 2}, iblockIdx.x22{8}, ithreadIdx.x23{128}] ca_pos( 3 )
  // produce_pos( 3 )
  //  logical domain : (iS17{( 2 * 8 )}, iS18{128})
  //  contiguity: t t
  //   Merge: iS20{8} and iS18{128} -> iS21{1024}
  //   Split: iS21{1024} by factor 128 -> iblockIdx.x22{8}, ithreadIdx.x23{128}
  //  loop domain : (iS19{2 ex 2}, iblockIdx.x22{8}, ithreadIdx.x23{128})
  //
  // iS19 is the repeat ID, which should be just a Serial ID with an
  // extent of 2.
  auto ref_tv = scheduled_fusion->outputs().at(0)->as<TensorView>();
  // The outermost loop ID should be a Serial ID with an extent of 2.
  EXPECT_EQ(
      ref_tv->getLoopDomain().at(0)->getParallelType(), ParallelType::Serial);
  EXPECT_TRUE(ref_tv->getLoopDomain().at(0)->extent()->isConstInt());
  EXPECT_EQ(
      ref_tv->getLoopDomain().at(0)->extent()->evaluate().as<int64_t>(), 2L);

  IdModel id_model(scheduled_fusion, /*build_graphs=*/false);
  const auto& exact_graph = id_model.buildExactGraph();

  const auto ref_loop = exact_graph.toGroups(ref_tv->getLoopDomain());

  // The other tensors, except for the pad output, should be fully inlined into
  // the reference tensor.
  for (auto tv : scheduled_fusion->allTvs()) {
    if (tv->isFusionInput()) {
      continue;
    }
    auto tv_loop = exact_graph.toGroups(tv->getLoopDomain());
    if (tv->definition() != nullptr && tv->definition()->isA<PadOp>()) {
      ValGroups ref_groups{ref_loop.begin() + 1, ref_loop.end()};
      // In the case of pad, the loop domain of the output tensor
      // should be mapped with the loop domain of the reference
      // without the outermost ID.
      EXPECT_EQ(tv_loop, ref_groups);
    } else {
      EXPECT_EQ(tv_loop, ref_loop);
      EXPECT_EQ(tv->getLoopDomain().size(), tv->getComputeAtPosition());
    }
  }
}

// Similar to EndingRepeat but with a broadcast ID already found in an
// input tensor. A similar Pattern appears in the LitGPT Llama RoPE
// module.
TEST_F(RopeTest, EndingRepeatWithNoBroadcastOp) {
  auto fusion_ptr = std::make_unique<Fusion>();
  FusionGuard fg(fusion_ptr.get());
  Fusion& fusion = *fusion_ptr;

  std::vector<int64_t> shape1{3, 1, 200};

  auto tv0 = makeContigConcreteTensor(shape1);
  fusion.addInput(tv0);

  auto tv1 = pad(tv0, {fusion.oneVal(), fusion.oneVal()});
  auto tv2 = expand(
      tv1,
      {IrBuilder::create<Val>(-1),
       IrBuilder::create<Val>(2),
       IrBuilder::create<Val>(-1)});
  auto tv3 =
      reshape(tv2, {IrBuilder::create<Val>(6), IrBuilder::create<Val>(-1)});
  fusion.addOutput(tv3);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  auto t0 = at::randn(shape1, options);

  FusionExecutorCache executor_cache(std::move(fusion_ptr));
  auto outputs = executor_cache.runFusionWithInputs({t0});
  testValidate(&fusion, outputs, {t0}, __LINE__, __FILE__);

  FusionKernelRuntime* runtime = executor_cache.getMostRecentKernelRuntime();
  EXPECT_FALSE(runtime->isSegmented());
  const auto& heuristic_param =
      runtime->schedulerHeuristics()->heuristicsList().front();
  EXPECT_EQ(heuristic_param->scheduler_type, SchedulerType::Resize);
  Fusion* scheduled_fusion = runtime->executors()
                                 .at(0)
                                 ->as<KernelExecutor>()
                                 ->compiledKernel()
                                 ->kernel();

  // Similar to the EndingRepeat tensor, the repeat factor ID should
  // be placed at the outermost position.
  auto ref_tv = scheduled_fusion->outputs().at(0)->as<TensorView>();
  // The outermost loop ID should be a Serial ID with an extent of 2.
  EXPECT_EQ(
      ref_tv->getLoopDomain().at(0)->getParallelType(), ParallelType::Serial);
  EXPECT_TRUE(ref_tv->getLoopDomain().at(0)->extent()->isConstInt());
  EXPECT_EQ(
      ref_tv->getLoopDomain().at(0)->extent()->evaluate().as<int64_t>(), 2L);

  IdModel id_model(scheduled_fusion, /*build_graphs=*/false);
  const auto& exact_graph = id_model.buildExactGraph();

  const auto ref_loop = exact_graph.toGroups(ref_tv->getLoopDomain());

  // All of the tensors have a mapped ID as the factor ID, so they
  // should all have the same loop ID groups.
  for (auto tv : scheduled_fusion->allTvs()) {
    if (tv->isFusionInput()) {
      continue;
    }
    EXPECT_EQ(exact_graph.toGroups(tv->getLoopDomain()), ref_loop);
    EXPECT_EQ(tv->getLoopDomain().size(), tv->getComputeAtPosition());
  }
}

} // namespace nvfuser
