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
TEST_F(RopeTest, HFMistralNemoFwd) {
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

} // namespace nvfuser
