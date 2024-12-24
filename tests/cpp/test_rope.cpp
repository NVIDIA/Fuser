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

class RopeTest : public NVFuserTest {
 protected:
  void SetUp() override {
    EnableOptionsGuard::getCurOptions().set(EnableOption::ResizeScheduler);
    NVFuserTest::SetUp();
  }

 private:
  EnableOptionsGuard enable_options_guard_;
};

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
TEST_F(RopeTest, HFMistralNemoFwd1) {
  const int64_t batch_size = 1;
  const int64_t seq_len = 4096;
  const int64_t head_dim = 128;
  const int64_t num_attention_heads = 32;
  const int64_t num_key_value_heads = 8;

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

  fusion.printMath();

  std::stringstream file_name;
  file_name << "mistral1.dot";
  IrGraphGenerator::print(
      &fusion,
      file_name.str().c_str(),
      IrGraphGenerator::DetailLevel::ComputeOnly);

  auto options_float =
      at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  auto options_bf16 =
      at::TensorOptions().dtype(at::kBFloat16).device(at::kCUDA, 0);
  auto t0 = at::randn(shape1, options_bf16);
  auto t1 = at::randn(shape2, options_bf16);
  auto t2 = at::randn(shape3, options_float).to(at::kLong);
  std::vector<c10::IValue> inputs({t0, t1, t2});

  FusionExecutorCache executor_cache(std::move(fusion_ptr));
  auto out_tensors = executor_cache.runFusionWithInputs(inputs);
  testValidate(
      executor_cache.fusion(), out_tensors, inputs, __LINE__, __FILE__);
}

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
TEST_F(RopeTest, HFMistralNemoFwd2) {
  const int64_t batch_size = 1;
  const int64_t seq_len = 4096;
  const int64_t head_dim = 128;
  const int64_t num_attention_heads = 32;
  const int64_t num_key_value_heads = 8;

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

  std::stringstream file_name;
  file_name << "mistral.dot";
  IrGraphGenerator::print(
      &fusion,
      file_name.str().c_str(),
      IrGraphGenerator::DetailLevel::ComputeOnly);

  auto options_fp32 =
      at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  auto options_bf16 =
      at::TensorOptions().dtype(at::kBFloat16).device(at::kCUDA, 0);
  auto t0 = at::randn(shape1, options_bf16);
  auto t1 = at::randn(shape2, options_bf16);
  auto t2 = at::randn(shape3, options_fp32);
  std::vector<c10::IValue> inputs({t0, t1, t2});

  FusionExecutorCache executor_cache(std::move(fusion_ptr));
  auto out_tensors = executor_cache.runFusionWithInputs(inputs);
  testValidate(
      executor_cache.fusion(), out_tensors, inputs, __LINE__, __FILE__);
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
TEST_F(RopeTest, HFMistralNemoBwd) {
  const int64_t batch_size = 1;
  const int64_t seq_len = 4096;
  const int64_t head_dim = 128;
  const int64_t num_attention_heads = 32;
  const int64_t num_key_value_heads = 8;

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
  TensorView* T15 = nullptr;
  if (getenv("TRANSLATE_REPEAT")) {
    std::vector<bool> bc_flags(T13->nDims() + 1, false);
    bc_flags.at(bc_flags.size() - 2) = true;
    auto t13_bc = broadcast(T13, bc_flags);
    std::vector<Val*> expanded_sizes(
        t13_bc->nDims(), IrBuilder::create<Val>(-1L));
    expanded_sizes.at(expanded_sizes.size() - 2) = IrBuilder::create<Val>(2L);
    auto t13_expanded = expand(t13_bc, expanded_sizes);
    std::vector<Val*> reshape_sizes;
    for (const auto logical_id : T13->getLogicalDomain()) {
      reshape_sizes.push_back(logical_id->extent());
    }
    reshape_sizes.back() =
        SimplifyingIrBuilder::mulExpr(reshape_sizes.back(), 2);
    auto t13_reshaped = reshape(t13_expanded, reshape_sizes);
    T15 = t13_reshaped;
  } else {
    T15 = cat({T13, T13}, -1);
  }
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

  fusion.printMath();

  std::stringstream file_name;
  file_name << "mistral_bwd.dot";
  IrGraphGenerator::print(
      &fusion,
      file_name.str().c_str(),
      IrGraphGenerator::DetailLevel::ComputeOnly);

  auto options_fp32 =
      at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  auto options_bf16 =
      at::TensorOptions().dtype(at::kBFloat16).device(at::kCUDA, 0);
  auto t0 = at::randn(shape0, options_bf16);
  auto t1 = at::randn(shape1, options_fp32);
  auto t2 = at::randn(shape2, options_bf16);
  auto t3 = at::randn(shape3, options_bf16);
  std::vector<c10::IValue> inputs({t0, t1, t2, t3});

  FusionExecutorCache executor_cache(std::move(fusion_ptr));
  auto out_tensors = executor_cache.runFusionWithInputs(inputs);
  testValidate(
      executor_cache.fusion(), out_tensors, inputs, __LINE__, __FILE__);
}

} // namespace nvfuser
