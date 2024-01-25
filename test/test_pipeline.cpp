// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#include <gtest/gtest.h>

#include <fusion.h>
#include <ir/all_nodes.h>
#include <ir/builder.h>
#include <multidevice/lower_communication.h>
#include <multidevice/pipeline.h>
#include <multidevice/utils.h>
#include <ops/all_ops.h>
#include <test/utils.h>

#include <algorithm>
#include <iostream>

namespace nvfuser {

using namespace at::indexing;

std::string sortByLine(const std::string& input) {
  auto ss = std::stringstream(input);
  std::vector<std::string> lines;
  std::string line;
  while (std::getline(ss, line, '\n')) {
    lines.push_back(line);
  }
  std::sort(lines.begin(), lines.end());
  std::stringstream output;
  bool first = true;
  for (auto line : lines) {
    if (!first) {
      output << std::endl;
    }
    first = false;
    output << line;
  }
  return output.str();
}

TEST_F(NVFuserTest, Pipeline_CUDA) {
  // Fusion definition
  Fusion fusion;
  FusionGuard fg(&fusion);

  TensorView* tv0 = makeContigTensor(2);
  fusion.addInput(tv0);
  TensorView* tv1 = sum(tv0, {0});

  TensorView* tv2 = set(tv1);
  TensorView* tv3 = sum(tv2, {0});
  fusion.addOutput(tv3);

  TensorView* tv4 = makeContigTensor(3);
  fusion.addInput(tv4);
  TensorView* tv5 = sum(tv4, {0});

  TensorView* tv6 = set(tv5);
  TensorView* tv7 = set(tv6);
  TensorView* tv8 = sum(tv7, {0});

  TensorView* tv9 = set(tv5);
  TensorView* tv10 = set(tv9);
  TensorView* tv11 = sum(tv10, {0});

  TensorView* tv12 = set(tv7);
  TensorView* tv13 = sum(tv12, {0});
  fusion.addOutput(tv13);

  TensorView* tv14 = set(tv8);
  TensorView* tv15 = set(tv11);
  TensorView* tv16 = set(tv13);
  TensorView* tv17 = add(tv14, tv15);
  TensorView* tv18 = add(tv17, tv16);
  TensorView* tv19 = sum(tv18, {0});
  fusion.addOutput(tv19);

  // Pipeline scheduling
  PipelineStageDescriptor stage0, stage1, stage2, stage3, stage4, stage5,
      stage6;
  stage0.addVal({tv0, tv1});
  stage1.addVal({tv2, tv3});
  stage2.addVal({tv4, tv5});
  stage3.addVal({tv6, tv7, tv8});
  stage4.addVal({tv9, tv10, tv11});
  stage5.addVal({tv12, tv13});
  stage6.addVal({tv14, tv15, tv16, tv17, tv18, tv19});

  PipelineDescriptor descriptor{.stage_descriptors{
      stage0, stage1, stage2, stage3, stage4, stage5, stage6}}; // the order
                                                                // doesnt matter
  Pipeline pipeline(&fusion, std::move(descriptor));

  // Validation
  std::string obtained_string = pipeline.toString();
  std::string ref_string{
      "Pipeline's inputs{:\n"
      " PipelineVal representing Val T0_g[ iS0{i0}, iS1{i2} ] on stage " +
      std::to_string(stage0.unique_id) +
      "\n"
      " PipelineVal representing Val T4_g[ iS6{i15}, iS7{i16}, iS8{i17} ] on stage " +
      std::to_string(stage2.unique_id) +
      "\n"
      "}\n"
      "Pipeline's Traversal inputs --> outputs {\n"
      "  PipelineStage representing Stage " +
      std::to_string(stage0.unique_id) +
      ".Inputs={T0_g[ iS0{i0}, iS1{i2} ], }. Outputs={T1_l[ rS2{i0}, iS3{i2} ], }.\n"
      "  PipelineVal representing Val T1_l[ rS2{i0}, iS3{i2} ] on stage " +
      std::to_string(stage0.unique_id) +
      "\n"
      "  PipelineCommunication that transfers PipelineVal representing Val T1_l[ rS2{i0}, iS3{i2} ] on stage " +
      std::to_string(stage0.unique_id) +
      " to PipelineVal representing Val T2_l[ iS4{i2} ] on stage " +
      std::to_string(stage1.unique_id) +
      "\n"
      "  PipelineVal representing Val T2_l[ iS4{i2} ] on stage " +
      std::to_string(stage1.unique_id) +
      "\n"
      "  PipelineStage representing Stage " +
      std::to_string(stage1.unique_id) +
      ".Inputs={T2_l[ iS4{i2} ], }. Outputs={T3_g[ rS5{i2} ], }.\n"
      "  PipelineStage representing Stage " +
      std::to_string(stage2.unique_id) +
      ".Inputs={T4_g[ iS6{i15}, iS7{i16}, iS8{i17} ], }. Outputs={T5_l[ rS9{i15}, iS10{i16}, iS11{i17} ], }.\n"
      "  PipelineVal representing Val T5_l[ rS9{i15}, iS10{i16}, iS11{i17} ] on stage " +
      std::to_string(stage2.unique_id) +
      "\n"
      "  PipelineCommunication that transfers PipelineVal representing Val T5_l[ rS9{i15}, iS10{i16}, iS11{i17} ] on stage " +
      std::to_string(stage2.unique_id) +
      " to PipelineVal representing Val T6_l[ iS12{i16}, iS13{i17} ] on stage " +
      std::to_string(stage3.unique_id) +
      "\n"
      "  PipelineVal representing Val T6_l[ iS12{i16}, iS13{i17} ] on stage " +
      std::to_string(stage3.unique_id) +
      "\n"
      "  PipelineStage representing Stage " +
      std::to_string(stage3.unique_id) +
      ".Inputs={T6_l[ iS12{i16}, iS13{i17} ], }. Outputs={T7_l[ iS14{i16}, iS15{i17} ], T8_l[ rS16{i16}, iS17{i17} ], }.\n"
      "  PipelineVal representing Val T7_l[ iS14{i16}, iS15{i17} ] on stage " +
      std::to_string(stage3.unique_id) +
      "\n"
      "  PipelineCommunication that transfers PipelineVal representing Val T7_l[ iS14{i16}, iS15{i17} ] on stage " +
      std::to_string(stage3.unique_id) +
      " to PipelineVal representing Val T12_l[ iS24{i16}, iS25{i17} ] on stage " +
      std::to_string(stage5.unique_id) +
      "\n"
      "  PipelineVal representing Val T12_l[ iS24{i16}, iS25{i17} ] on stage " +
      std::to_string(stage5.unique_id) +
      "\n"
      "  PipelineStage representing Stage " +
      std::to_string(stage5.unique_id) +
      ".Inputs={T12_l[ iS24{i16}, iS25{i17} ], }. Outputs={T13_g[ rS26{i16}, iS27{i17} ], }.\n"
      "  PipelineVal representing Val T8_l[ rS16{i16}, iS17{i17} ] on stage " +
      std::to_string(stage3.unique_id) +
      "\n"
      "  PipelineCommunication that transfers PipelineVal representing Val T8_l[ rS16{i16}, iS17{i17} ] on stage " +
      std::to_string(stage3.unique_id) +
      " to PipelineVal representing Val T14_l[ iS28{i17} ] on stage " +
      std::to_string(stage6.unique_id) +
      "\n"
      "  PipelineVal representing Val T14_l[ iS28{i17} ] on stage " +
      std::to_string(stage6.unique_id) +
      "\n"
      "  PipelineCommunication that transfers PipelineVal representing Val T5_l[ rS9{i15}, iS10{i16}, iS11{i17} ] on stage " +
      std::to_string(stage2.unique_id) +
      " to PipelineVal representing Val T9_l[ iS18{i16}, iS19{i17} ] on stage " +
      std::to_string(stage4.unique_id) +
      "\n"
      "  PipelineVal representing Val T9_l[ iS18{i16}, iS19{i17} ] on stage " +
      std::to_string(stage4.unique_id) +
      "\n"
      "  PipelineStage representing Stage " +
      std::to_string(stage4.unique_id) +
      ".Inputs={T9_l[ iS18{i16}, iS19{i17} ], }. Outputs={T11_l[ rS22{i16}, iS23{i17} ], }.\n"
      "  PipelineVal representing Val T11_l[ rS22{i16}, iS23{i17} ] on stage " +
      std::to_string(stage4.unique_id) +
      "\n"
      "  PipelineCommunication that transfers PipelineVal representing Val T11_l[ rS22{i16}, iS23{i17} ] on stage " +
      std::to_string(stage4.unique_id) +
      " to PipelineVal representing Val T15_l[ iS29{i17} ] on stage " +
      std::to_string(stage6.unique_id) +
      "\n"
      "  PipelineVal representing Val T15_l[ iS29{i17} ] on stage " +
      std::to_string(stage6.unique_id) +
      "\n"
      "  PipelineCommunication that transfers PipelineVal representing Val T13_g[ rS26{i16}, iS27{i17} ] on stage " +
      std::to_string(stage5.unique_id) +
      " to PipelineVal representing Val T16_l[ iS30{i17} ] on stage " +
      std::to_string(stage6.unique_id) +
      "\n"
      "  PipelineVal representing Val T16_l[ iS30{i17} ] on stage " +
      std::to_string(stage6.unique_id) +
      "\n"
      "  PipelineStage representing Stage " +
      std::to_string(stage6.unique_id) +
      ".Inputs={T14_l[ iS28{i17} ], T15_l[ iS29{i17} ], T16_l[ iS30{i17} ], }. Outputs={T19_g[ rS33{i17} ], }.\n"
      "}\n"
      "Pipeline's outputs:{\n"
      " PipelineVal representing Val T3_g[ rS5{i2} ] on stage " +
      std::to_string(stage1.unique_id) +
      "\n"
      " PipelineVal representing Val T13_g[ rS26{i16}, iS27{i17} ] on stage " +
      std::to_string(stage5.unique_id) +
      "\n"
      " PipelineVal representing Val T19_g[ rS33{i17} ] on stage " +
      std::to_string(stage6.unique_id) +
      "\n"
      "}"};

  // We sort the string by line so it doesn't depend on the order of the
  // Pipeline's DAG traversal
  obtained_string = sortByLine(obtained_string);
  ref_string = sortByLine(ref_string);

  EXPECT_EQ(obtained_string, ref_string);
}

TEST_F(NVFuserTest, ReshardingDetection) {
  auto fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());

  DeviceMesh mesh0, mesh1, mesh2;
  mesh0 = {0, 1};
  mesh1 = {0, 2};
  mesh2 = {0, 1, 2};

  TensorView* tv0 = makeContigTensor(3);
  fusion->addInput(tv0);
  tv0->setDeviceMesh(mesh0);

  TensorView* tv1 = set(tv0);
  tv1->setDeviceMesh(mesh0);

  TensorView* tv2 = set(tv1); // resharding
  tv2->setDeviceMesh(mesh1);

  TensorView* tv3 = set(tv2);
  tv3->setDeviceMesh(mesh1);

  TensorView* tv4 = set(tv3); // resharding
  tv4->setDeviceMesh(mesh2);

  TensorView* tv5 = set(tv4); // resharding
  tv5->setDeviceMesh(mesh2);
  tv5->axis(0)->parallelize(ParallelType::DIDx);

  TensorView* tv6 = set(tv5);
  tv6->setDeviceMesh(mesh2);
  tv6->axis(0)->parallelize(ParallelType::DIDx);

  TensorView* tv7 = set(tv6); // resharding
  tv7->setDeviceMesh(mesh2);

  TensorView* tv8 = sum(tv0, {0});
  tv8->setDeviceMesh(mesh0);

  TensorView* tv9 = sum(tv0, {0}); // resharding, but seems invalid
  tv9->setDeviceMesh(mesh0);
  tv9->axis(0)->parallelize(ParallelType::DIDx);

  TensorView* tv10 = sum(tv0, {0}); // resharding,
  tv10->setDeviceMesh(mesh0);
  tv10->axis(1)->parallelize(ParallelType::DIDx);

  TensorView* tv11 = sum(tv0, {0}); // resharding,
  tv11->setDeviceMesh(mesh1);

  TensorView* tv12 = sum(tv5, {0}); // not resharding
  tv12->setDeviceMesh(mesh2);
  tv12->axis(0)->parallelize(ParallelType::DIDx);

  TensorView* tv13 = sum(tv5, {0}); // resharding
  tv13->setDeviceMesh(mesh2);
  tv13->axis(1)->parallelize(ParallelType::DIDx);

  TensorView* tv14 = sum(tv5, {0}); // resharding
  tv14->setDeviceMesh(mesh2);
  tv14->axis(0)->parallelize(ParallelType::DIDx);
  tv14->axis(1)->parallelize(ParallelType::DIDx);

  TensorView* tv15 = add(tv0, tv1);
  tv15->setDeviceMesh(mesh0);

  TensorView* tv16 = add(tv0, tv1); // resharding
  tv16->setDeviceMesh(mesh1);

  TensorView* tv17 = add(tv0, tv1); // resharding
  tv17->setDeviceMesh(mesh0);
  tv17->axis(0)->parallelize(ParallelType::DIDx);

  TensorView* tv18 = add(tv5, tv6);
  tv18->setDeviceMesh(mesh2);
  tv18->axis(0)->parallelize(ParallelType::DIDx);

  TensorView* tv19 = add(tv5, tv7); // resharding
  tv19->setDeviceMesh(mesh2);
  tv19->axis(0)->parallelize(ParallelType::DIDx);

  TensorView* tv20 = add(tv5, tv7); // resharding
  tv20->setDeviceMesh(mesh2);

  TensorView* tv21 = add(tv0, tv7); // resharding
  tv21->setDeviceMesh(mesh2);

  TensorView* tv22 = sum(tv5, {1});
  tv22->setDeviceMesh(mesh2);
  tv22->axis(0)->parallelize(ParallelType::DIDx);

  TensorView* tv23 = sum(tv5, {1}); // resharding
  tv23->setDeviceMesh(mesh2);

  TensorView* tv24 = sum(tv7, {0});
  tv24->setDeviceMesh(mesh2);

  TensorView* tv25 = sum(tv7, {0}); // not resharding but invalid
  tv25->setDeviceMesh(mesh2);
  tv22->axis(0)->parallelize(ParallelType::DIDx);

  TensorView* tv26 = add(tv5, tv6); // resharding
  tv26->setDeviceMesh(mesh2);

  fusion->addOutput(tv1);
  fusion->addOutput(tv2);
  fusion->addOutput(tv3);
  fusion->addOutput(tv4);
  fusion->addOutput(tv5);
  fusion->addOutput(tv6);
  fusion->addOutput(tv7);
  fusion->addOutput(tv8);
  fusion->addOutput(tv9);
  fusion->addOutput(tv10);
  fusion->addOutput(tv11);
  fusion->addOutput(tv12);
  fusion->addOutput(tv13);
  fusion->addOutput(tv14);
  fusion->addOutput(tv15);
  fusion->addOutput(tv16);
  fusion->addOutput(tv17);
  fusion->addOutput(tv18);
  fusion->addOutput(tv19);
  fusion->addOutput(tv20);
  fusion->addOutput(tv21);
  fusion->addOutput(tv22);
  fusion->addOutput(tv23);
  fusion->addOutput(tv24);
  fusion->addOutput(tv25);
  fusion->addOutput(tv26);

  GTEST_EXPECT_TRUE(!isResharding(tv1->definition()));
  GTEST_EXPECT_TRUE(isResharding(tv2->definition()));
  GTEST_EXPECT_TRUE(!isResharding(tv3->definition()));
  GTEST_EXPECT_TRUE(isResharding(tv4->definition()));
  GTEST_EXPECT_TRUE(isResharding(tv5->definition()));
  GTEST_EXPECT_TRUE(!isResharding(tv6->definition()));
  GTEST_EXPECT_TRUE(isResharding(tv7->definition()));
  GTEST_EXPECT_TRUE(!isResharding(tv8->definition()));
  GTEST_EXPECT_TRUE(isResharding(tv9->definition()));
  GTEST_EXPECT_TRUE(isResharding(tv10->definition()));
  GTEST_EXPECT_TRUE(isResharding(tv11->definition()));
  GTEST_EXPECT_TRUE(!isResharding(tv12->definition()));
  GTEST_EXPECT_TRUE(isResharding(tv13->definition()));
  GTEST_EXPECT_TRUE(isResharding(tv14->definition()));
  GTEST_EXPECT_TRUE(!isResharding(tv15->definition()));
  GTEST_EXPECT_TRUE(isResharding(tv16->definition()));
  GTEST_EXPECT_TRUE(isResharding(tv17->definition()));
  GTEST_EXPECT_TRUE(!isResharding(tv18->definition()));
  GTEST_EXPECT_TRUE(isResharding(tv19->definition()));
  GTEST_EXPECT_TRUE(isResharding(tv20->definition()));
  GTEST_EXPECT_TRUE(isResharding(tv21->definition()));
  GTEST_EXPECT_TRUE(!isResharding(tv22->definition()));
  GTEST_EXPECT_TRUE(isResharding(tv23->definition()));
  GTEST_EXPECT_TRUE(!isResharding(tv24->definition()));
  GTEST_EXPECT_TRUE(!isResharding(tv25->definition()));
  GTEST_EXPECT_TRUE(isResharding(tv26->definition()));
}

using automaticSetInsertionTestParams =
    std::tuple<DeviceMesh, DeviceMesh, DeviceMesh, bool, bool, bool>;

class automaticReshardingTest
    : public NVFuserTest,
      public ::testing::WithParamInterface<automaticSetInsertionTestParams> {
 protected:
  void SetUp() override {
    fusion = std::make_unique<Fusion>();
    fg = std::make_unique<FusionGuard>(fusion.get());
  }
  void validate() {
    for (auto expr : fusion->exprs()) {
      GTEST_EXPECT_TRUE(!isResharding(expr) || isLowerableToCommunication(expr))
          << "on expr=" << expr;
    }
  }

  std::unique_ptr<Fusion> fusion;
  std::unique_ptr<FusionGuard> fg;
};

TEST_P(automaticReshardingTest, setInsertion) {
  auto
      [mesh0,
       mesh1,
       mesh2,
       is_tv0_tv3_tv5_sharded,
       is_tv1_tv4_sharded,
       is_tv2_sharded] = GetParam();

  TensorView* tv0 = makeContigTensor(3);
  TensorView* tv1 = binaryOp(BinaryOpType::Mul, tv0, tv0);
  TensorView* tv2 = binaryOp(BinaryOpType::Add, tv0, tv1);
  TensorView* tv3 = sum(tv2, {0});
  TensorView* tv4 = broadcast(tv3, {true, false, false});
  TensorView* tv5 = binaryOp(BinaryOpType::Mul, tv2, tv4);

  tv0->setDeviceMesh(mesh0);
  tv1->setDeviceMesh(mesh1);
  tv2->setDeviceMesh(mesh2);
  tv3->setDeviceMesh(mesh0);
  tv4->setDeviceMesh(mesh1);
  tv5->setDeviceMesh(mesh0);
  fusion->addInput(tv0);
  fusion->addOutput(tv1);
  fusion->addOutput(tv5);

  if (is_tv0_tv3_tv5_sharded) {
    tv0->axis(0)->parallelize(ParallelType::DIDx);
    tv3->axis(0)->parallelize(ParallelType::DIDx);
    tv5->axis(0)->parallelize(ParallelType::DIDx);
  }
  if (is_tv1_tv4_sharded) {
    tv1->axis(0)->parallelize(ParallelType::DIDx);
    tv4->axis(0)->parallelize(ParallelType::DIDx);
  }
  if (is_tv2_sharded) {
    tv2->axis(0)->parallelize(ParallelType::DIDx);
  }

  insertReshardings(fusion.get());
  validate();
}

namespace {

DeviceMesh Mesh0({0});
DeviceMesh Mesh1({1, 2});
DeviceMesh Mesh2({0, 1, 2, 3});

} // namespace

INSTANTIATE_TEST_SUITE_P(
    ,
    automaticReshardingTest,
    ::testing::Combine(
        ::testing::Values(Mesh0, Mesh2),
        ::testing::Values(Mesh1, Mesh2),
        ::testing::Values(Mesh2),
        ::testing::Bool(),
        ::testing::Bool(),
        ::testing::Bool()));

} // namespace nvfuser
