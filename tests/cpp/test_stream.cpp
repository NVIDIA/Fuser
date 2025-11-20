// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#include <gtest/gtest.h>

#include <ATen/ops/randn.h>
#include <ATen/ops/zeros_like.h>

#include <fusion.h>
#include <fusion_guard.h>
#include <ir/interface_nodes.h>
#include <multidevice/utils.h>
#include <ops/all_ops.h>
#include <preseg_passes/pre_segmenter.h>
#include <preseg_passes/propagate_shardings.h>
#include <runtime/fusion_executor_cache.h>
#include <tests/cpp/utils.h>
#include <tests/cpp/validator.h>

namespace nvfuser {

class StreamTest : public NVFuserTest {
 public:
  StreamTest() {
    EnableOptionsGuard::getCurOptions().set(EnableOption::HostIrLowering);
  }
};

TEST_F(StreamTest, AddPerStream) {
  constexpr int64_t c = 3;
  Fusion fusion;
  FusionGuard fg(&fusion);

  TensorView* in = makeContigTensor(2);
  TensorView* out = add(in, in);
  fusion.addInput(in);
  fusion.addOutput(out);

  in->outer_split(1, c);
  in->axis(1)->parallelize(ParallelType::Stream);
  out->outer_split(1, c);
  out->axis(1)->parallelize(ParallelType::Stream);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA);
  at::Tensor in_tensor = at::randn({5, c * 2}, options);
  at::Tensor out_tensor = at::zeros_like(in_tensor);

  KernelExecutor ke;
  ke.compile(&fusion, {in_tensor});
  constexpr int64_t kStreamIndex = 1;
  ke.run({in_tensor, kStreamIndex}, {out_tensor});

  at::Tensor expected_out_tensor = in_tensor + in_tensor;
  std::vector<at::Tensor> chunks = expected_out_tensor.chunk(c, 1);
  for (auto [i, chunk] : enumerate(chunks)) {
    if (i != kStreamIndex) {
      chunk.zero_();
    }
  }
  EXPECT_TRUE(at::allclose(out_tensor, expected_out_tensor))
      << out_tensor << " vs " << expected_out_tensor;
}

TEST_F(StreamTest, Matmul) {
  constexpr int64_t c = 3;

  auto fusion = std::make_unique<Fusion>();
  {
    FusionGuard fg(fusion.get());
    TensorView* in = makeSymbolicTensor(2);
    TensorView* w = makeSymbolicTensor(2);
    TensorView* out = matmul(in, w);
    fusion->addInput(in);
    fusion->addInput(w);
    fusion->addOutput(out);

    out->outer_split(1, c);
    out->axis(1)->parallelize(ParallelType::Stream);
  }

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA);
  at::Tensor in_tensor = at::randn({5, 7}, options);
  at::Tensor w_tensor = at::randn({7, c * 2}, options);

  // With NVFUSER_DUMP=host_ir, you'll see the host IR container like the
  // following:
  // clang-format off
  // %HostIrContainer { (T0_g_float[iS0{i0}, iS1{i2}], T1_g_float[istreamIdx7{3}, iS11{i2}, iS8{( ceilDiv(i4, 3) )}]) -> (T2_g_float[istreamIdx9{3}, iS4{i0}, iS10{( ceilDiv(i4, 3) )}, rS6{i2}]) :
  //   FOR i18 from 0 to 3:
  //     T2_g_float[istreamIdx9{3}, iS4{i0}, iS10{( ceilDiv(i4, 3) )}, rS6{i2}]
  //        = matmul(T0_g_float[iS0{i0}, iS1{i2}],
  //                 T1_g_float[istreamIdx7{3}, iS11{i2}, iS8{( ceilDiv(i4, 3) )}])
  // } // %HostIrContainer
  // clang-format on
  FusionExecutorCache executor_cache(std::move(fusion));
  auto out_tensor = executor_cache.runFusionWithInputs({in_tensor, w_tensor})[0]
                        .as<at::Tensor>();

  testValidate(
      executor_cache.fusion(),
      {out_tensor},
      {in_tensor, w_tensor},
      __LINE__,
      __FILE__);
}

TEST_F(StreamTest, TwoMatmuls_Inlinable) {
  constexpr int64_t c = 3;

  auto fusion = std::make_unique<Fusion>();
  {
    FusionGuard fg(fusion.get());
    TensorView* in = makeSymbolicTensor(2);
    TensorView* w1 = makeSymbolicTensor(2);
    TensorView* w2 = makeSymbolicTensor(2);
    TensorView* out = matmul(in, w1);
    out = matmul(out, w2);
    fusion->addInput(in);
    fusion->addInput(w1);
    fusion->addInput(w2);
    fusion->addOutput(out);

    in->outer_split(0, c);
    in->axis(0)->parallelize(ParallelType::Stream);
  }

  {
    auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA);
    at::Tensor in = at::randn({c * 2, 3}, options);
    at::Tensor w1 = at::randn({3, 5}, options);
    at::Tensor w2 = at::randn({5, 3}, options);

    // With NVFUSER_DUMP=host_ir, you'll see the host IR container like the
    // following:
    // clang-format off
    // %HostIrContainer { (T0_g_float[istreamIdx12{3}, iS13{( ceilDiv(i0, 3) )}, iS1{i2}], T1_g_float[iS14{i2}, iS3{i4}], T2_g_float[iS15{i4}, iS5{i6}]) -> (T4_g_float[istreamIdx18{3}, iS19{( ceilDiv(i0, 3) )}, iS10{i6}, rS11{i4}]) :
    //   T4_g_float[istreamIdx18{3}, iS19{( ceilDiv(i0, 3) )}, iS10{i6}, rS11{i4}] = ALLOCATE(buffer=T4_g_float[istreamIdx18{3}, iS19{( ceilDiv(i0, 3) )}, iS10{i6}, rS11{i4}], mem_type=global, size=( i0 * i6 ), zero_init=false, resets_to_zero=false)
    //   FOR i99 from 0 to 3:
    //     T5_l_float[istreamIdx22{3}, iS23{( ceilDiv(i0, 3) )}, iS21{i2}] = ShardByStream(T0_g_float[istreamIdx12{3}, iS13{( ceilDiv(i0, 3) )}, iS1{i2}], stream_index = i99)
    //     T3_g_float[istreamIdx16{3}, iS17{( ceilDiv(i0, 3) )}, iS7{i4}, rS8{i2}]
    //        = matmul(T5_l_float[istreamIdx22{3}, iS23{( ceilDiv(i0, 3) )}, iS21{i2}],
    //                 T1_g_float[iS14{i2}, iS3{i4}])
    //     T6_l_float[istreamIdx26{3}, iS27{( ceilDiv(i0, 3) )}, iS25{i6}] = ShardByStream(T4_g_float[istreamIdx18{3}, iS19{( ceilDiv(i0, 3) )}, iS10{i6}, rS11{i4}], stream_index = i99)
    //     T6_l_float[istreamIdx26{3}, iS27{( ceilDiv(i0, 3) )}, iS25{i6}]
    //        = matmul(T3_g_float[istreamIdx16{3}, iS17{( ceilDiv(i0, 3) )}, iS7{i4}, rS8{i2}],
    //                 T2_g_float[iS15{i4}, iS5{i6}])
    // } // %HostIrContainer
    // clang-format on
    FusionExecutorCache executor_cache(std::move(fusion));
    auto out =
        executor_cache.runFusionWithInputs({in, w1, w2})[0].as<at::Tensor>();

    testValidate(
        executor_cache.fusion(), {out}, {in, w1, w2}, __LINE__, __FILE__);
  }
}

TEST_F(StreamTest, TwoMatmuls_NotInlinable) {
  constexpr int64_t c = 3;

  auto fusion = std::make_unique<Fusion>();
  {
    FusionGuard fg(fusion.get());
    TensorView* in = makeSymbolicTensor(2);
    TensorView* w1 = makeSymbolicTensor(2);
    TensorView* w2 = makeSymbolicTensor(2);
    TensorView* out = matmul(in, w1);
    out = matmul(out, w2);
    fusion->addInput(in);
    fusion->addInput(w1);
    fusion->addInput(w2);
    fusion->addOutput(out);

    // in: [m, k]
    // w1: [k, n/s]
    // out1: [m, n/s]
    // w2: [n, k]
    // out2: [m/s, k]
    w1->outer_split(1, c);
    w1->axis(1)->parallelize(ParallelType::Stream);
    out->outer_split(0, c);
    out->axis(0)->parallelize(ParallelType::Stream);
  }

  // The second matmul should have been decomposed because its input [m, n/s]
  // and output [m/s, k] are sharded differently on ParallelType::Stream.
  //
  // clang-format off
  // Fusion IR after pre-segmenter optimization passes:
  // Inputs:
  //   T0_g_float[iS0{i0}, iS1{i2}]
  //   T1_g_float[istreamIdx12{3}, iS16{i2}, iS13{( ceilDiv(i4, 3) )}]
  //   T2_g_float[iS17{i4}, iS5{i6}]
  // Outputs:
  //   T4_g_float[istreamIdx14{3}, iS15{( ceilDiv(i0, 3) )}, iS10{i6}, rS11{i4}]

  // %kernel {
  // T3_l_float[istreamIdx18{3}, iS6{i0}, iS19{( ceilDiv(i4, 3) )}, rS8{i2}]
  //    = matmul(T0_g_float[iS0{i0}, iS1{i2}],
  //             T1_g_float[istreamIdx12{3}, iS16{i2}, iS13{( ceilDiv(i4, 3) )}])
  // T4_g_float[istreamIdx14{3}, iS15{( ceilDiv(i0, 3) )}, iS10{i6}, rS11{i4}]
  //    = matmul(T3_l_float[istreamIdx18{3}, iS6{i0}, iS19{( ceilDiv(i4, 3) )}, rS8{i2}],
  //             T2_g_float[iS17{i4}, iS5{i6}])

  // TransformPrinter :
  // T0_g_float[iS0{i0}, iS1{i2}]
  //  logical domain : (iS0{i0}, iS1{i2})
  //  contiguity: f f
  //  loop domain : (iS0{i0}, iS1{i2})
  // T1_g_float[istreamIdx12{3}, iS16{i2}, iS13{( ceilDiv(i4, 3) )}]
  //  logical domain : (iS16{i2}, iS3{i4})
  //  allocation domain : (iS16{i2}, iS3{i4})
  //  contiguity: f f
  //   Outer split: iS3{i4} by factor 3 -> istreamIdx12{3}, iS13{( ceilDiv(i4, 3) )}
  //  loop domain : (istreamIdx12{3}, iS16{i2}, iS13{( ceilDiv(i4, 3) )})
  // T3_l_float[istreamIdx18{3}, iS6{i0}, iS19{( ceilDiv(i4, 3) )}, rS8{i2}]
  //  logical domain : (iS6{i0}, iS7{i4}, rS8{i2})
  //  allocation domain : (iS6{i0}, iS7{i4}, rS8{i2})
  //  contiguity: t t n
  //   Outer split: iS7{i4} by factor 3 -> istreamIdx18{3}, iS19{( ceilDiv(i4, 3) )}
  //  loop domain : (istreamIdx18{3}, iS6{i0}, iS19{( ceilDiv(i4, 3) )}, rS8{i2})
  // T2_g_float[iS17{i4}, iS5{i6}]
  //  logical domain : (iS17{i4}, iS5{i6})
  //  contiguity: f f
  //  loop domain : (iS17{i4}, iS5{i6})
  // T4_g_float[istreamIdx14{3}, iS15{( ceilDiv(i0, 3) )}, iS10{i6}, rS11{i4}]
  //  logical domain : (iS9{i0}, iS10{i6}, rS11{i4})
  //  allocation domain : (iS9{i0}, iS10{i6}, rS11{i4})
  //  contiguity: t t n
  //   Outer split: iS9{i0} by factor 3 -> istreamIdx14{3}, iS15{( ceilDiv(i0, 3) )}
  //  loop domain : (istreamIdx14{3}, iS15{( ceilDiv(i0, 3) )}, iS10{i6}, rS11{i4})
  // } // %kernel
  // clang-format on

  {
    auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA);
    at::Tensor in = at::randn({2, 3}, options);
    at::Tensor w1 = at::randn({3, 5 * c}, options);
    at::Tensor w2 = at::randn({5 * c, 3}, options);

    FusionExecutorCache executor_cache(std::move(fusion));
    auto out =
        executor_cache.runFusionWithInputs({in, w1, w2})[0].as<at::Tensor>();

    testValidate(
        executor_cache.fusion(), {out}, {in, w1, w2}, __LINE__, __FILE__);
  }
}

TEST_F(StreamTest, HaveDifferentShardings) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  constexpr int64_t s = 2;

  TensorView* tv0 = makeContigTensor(2);
  TensorView* tv1 = set(tv0);
  TensorView* tv2 = set(tv1);
  TensorView* tv3 = set(tv2);
  fusion.addInput(tv0);
  fusion.addOutput(tv2);

  // tv1: [s, i0/s, i1]
  tv1->outer_split(0, s);
  tv1->axis(0)->parallelize(ParallelType::Stream);

  // tv2: [s, i0/s, i1]
  tv2->outer_split(0, s);
  tv2->axis(0)->parallelize(ParallelType::Stream);

  // tv3: [s, i0, i1/s]
  tv3->outer_split(1, s);
  tv3->axis(1)->parallelize(ParallelType::Stream);

  EXPECT_FALSE(haveDifferentShardings(
      tv1, DomainType::kLoop, tv2, DomainType::kLoop, {ParallelType::Stream}));
  EXPECT_TRUE(haveDifferentShardings(
      tv2, DomainType::kLoop, tv3, DomainType::kLoop, {ParallelType::Stream}));
}

TEST_F(StreamTest, ForwardPropagation) {
  auto fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());

  const int64_t s = 2;

  TensorView* in = makeContigTensor(2);
  TensorView* w = makeContigTensor(2);
  TensorView* out = matmul(in, w);
  fusion->addInput(in);
  fusion->addInput(w);
  fusion->addOutput(out);

  w->outer_split(1, s);
  w->axis(1)->parallelize(ParallelType::Stream);

  preseg_passes::OptimizationPass<
      preseg_passes::PropagateShardingsPass>::runPass(fusion.get());
  EXPECT_TRUE(out->axis(1)->isStream()) << out;
}

TEST_F(StreamTest, BackwardPropagation) {
  auto fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());

  const int64_t s = 2;

  TensorView* tv0 = makeContigTensor(2);
  TensorView* tv1 = add(tv0, IrBuilder::create<Val>(1.0));
  TensorView* tv2 = add(tv1, tv1);
  fusion->addInput(tv0);
  fusion->addOutput(tv2);

  tv2->outer_split(0, s);
  tv2->axis(0)->parallelize(ParallelType::Stream);

  preseg_passes::OptimizationPass<
      preseg_passes::PropagateShardingsPass>::runPass(fusion.get());
  for (auto* tv : {tv0, tv1, tv2}) {
    EXPECT_TRUE(tv->axis(0)->isStream()) << tv;
  }
}

TEST_F(StreamTest, ShardedAllocation) {
  auto fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());

  const int64_t s = 2;

  TensorView* tv0 = makeContigTensor(3);
  TensorView* tv1 = add(tv0, IrBuilder::create<Val>(1.0));
  TensorView* tv2 = sum(tv1, {2});
  TensorView* tv3 = div(tv1, IrBuilder::create<Val>(2.0));
  fusion->addInput(tv0);
  fusion->addOutput(tv2);
  fusion->addOutput(tv3);

  tv0->outer_split(0, s);
  tv0->axis(0)->parallelize(ParallelType::Stream);

  preseg_passes::OptimizationPass<preseg_passes::PreSegmenter>::runPass(
      fusion.get());

  for (auto* tv : {tv0, tv1, tv2, tv3}) {
    EXPECT_TRUE(tv->axis(0)->isStream()) << tv;
    if (tv->isFusionOutput() || tv->isFusionInput()) {
      EXPECT_EQ(tv->getAllocationDomain(), tv->getLogicalDomain());
    } else {
      EXPECT_EQ(tv->getAllocationDomain(), tv->getLoopDomain());
    }
  }
}

TEST_F(StreamTest, ReplicatedAllocation) {
  auto fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());

  const int64_t s = 2;

  TensorView* tv0 = makeContigTensor(3);
  TensorView* tv1 = add(tv0, IrBuilder::create<Val>(1.0));
  TensorView* tv2 = sum(tv1, {2});
  TensorView* tv3 = div(tv1, IrBuilder::create<Val>(2.0));
  fusion->addInput(tv0);
  fusion->addOutput(tv2);
  fusion->addOutput(tv3);

  tv0->outer_split(0, s);
  tv0->axis(0)->parallelize(ParallelType::Stream);
  tv2->outer_split(1, s);
  tv2->axis(1)->parallelize(ParallelType::Stream);

  preseg_passes::OptimizationPass<preseg_passes::PreSegmenter>::runPass(
      fusion.get());
  for (auto* tv : {tv0, tv1, tv2, tv3}) {
    EXPECT_TRUE(tv->axis(0)->isStream()) << tv;
    EXPECT_EQ(tv->getAllocationDomain(), tv->getLogicalDomain());
  }
}

} // namespace nvfuser
