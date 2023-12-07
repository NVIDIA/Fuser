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
#include <multidevice/lower_resharding_expr.h>
#include <multidevice/pipeline.h>
#include <ops/all_ops.h>
#include <test/utils.h>
#include <fusion_segmenter.h>
#include <executor_kernel_arg.h>

#include <algorithm>
#include <iostream>

namespace nvfuser {

using namespace at::indexing;

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
      " PipelineVal representing Val T4_g[ iS6{i13}, iS7{i14}, iS8{i15} ] on stage " +
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
      ".Inputs={T4_g[ iS6{i13}, iS7{i14}, iS8{i15} ], }. Outputs={T5_l[ rS9{i13}, iS10{i14}, iS11{i15} ], }.\n"
      "  PipelineVal representing Val T5_l[ rS9{i13}, iS10{i14}, iS11{i15} ] on stage " +
      std::to_string(stage2.unique_id) +
      "\n"
      "  PipelineCommunication that transfers PipelineVal representing Val T5_l[ rS9{i13}, iS10{i14}, iS11{i15} ] on stage " +
      std::to_string(stage2.unique_id) +
      " to PipelineVal representing Val T6_l[ iS12{i14}, iS13{i15} ] on stage " +
      std::to_string(stage3.unique_id) +
      "\n"
      "  PipelineVal representing Val T6_l[ iS12{i14}, iS13{i15} ] on stage " +
      std::to_string(stage3.unique_id) +
      "\n"
      "  PipelineStage representing Stage " +
      std::to_string(stage3.unique_id) +
      ".Inputs={T6_l[ iS12{i14}, iS13{i15} ], }. Outputs={T7_l[ iS14{i14}, iS15{i15} ], T8_l[ rS16{i14}, iS17{i15} ], }.\n"
      "  PipelineVal representing Val T7_l[ iS14{i14}, iS15{i15} ] on stage " +
      std::to_string(stage3.unique_id) +
      "\n"
      "  PipelineCommunication that transfers PipelineVal representing Val T7_l[ iS14{i14}, iS15{i15} ] on stage " +
      std::to_string(stage3.unique_id) +
      " to PipelineVal representing Val T12_l[ iS24{i14}, iS25{i15} ] on stage " +
      std::to_string(stage5.unique_id) +
      "\n"
      "  PipelineVal representing Val T12_l[ iS24{i14}, iS25{i15} ] on stage " +
      std::to_string(stage5.unique_id) +
      "\n"
      "  PipelineStage representing Stage " +
      std::to_string(stage5.unique_id) +
      ".Inputs={T12_l[ iS24{i14}, iS25{i15} ], }. Outputs={T13_g[ rS26{i14}, iS27{i15} ], }.\n"
      "  PipelineVal representing Val T8_l[ rS16{i14}, iS17{i15} ] on stage " +
      std::to_string(stage3.unique_id) +
      "\n"
      "  PipelineCommunication that transfers PipelineVal representing Val T8_l[ rS16{i14}, iS17{i15} ] on stage " +
      std::to_string(stage3.unique_id) +
      " to PipelineVal representing Val T14_l[ iS28{i15} ] on stage " +
      std::to_string(stage6.unique_id) +
      "\n"
      "  PipelineVal representing Val T14_l[ iS28{i15} ] on stage " +
      std::to_string(stage6.unique_id) +
      "\n"
      "  PipelineCommunication that transfers PipelineVal representing Val T5_l[ rS9{i13}, iS10{i14}, iS11{i15} ] on stage " +
      std::to_string(stage2.unique_id) +
      " to PipelineVal representing Val T9_l[ iS18{i14}, iS19{i15} ] on stage " +
      std::to_string(stage4.unique_id) +
      "\n"
      "  PipelineVal representing Val T9_l[ iS18{i14}, iS19{i15} ] on stage " +
      std::to_string(stage4.unique_id) +
      "\n"
      "  PipelineStage representing Stage " +
      std::to_string(stage4.unique_id) +
      ".Inputs={T9_l[ iS18{i14}, iS19{i15} ], }. Outputs={T11_l[ rS22{i14}, iS23{i15} ], }.\n"
      "  PipelineVal representing Val T11_l[ rS22{i14}, iS23{i15} ] on stage " +
      std::to_string(stage4.unique_id) +
      "\n"
      "  PipelineCommunication that transfers PipelineVal representing Val T11_l[ rS22{i14}, iS23{i15} ] on stage " +
      std::to_string(stage4.unique_id) +
      " to PipelineVal representing Val T15_l[ iS29{i15} ] on stage " +
      std::to_string(stage6.unique_id) +
      "\n"
      "  PipelineVal representing Val T15_l[ iS29{i15} ] on stage " +
      std::to_string(stage6.unique_id) +
      "\n"
      "  PipelineCommunication that transfers PipelineVal representing Val T13_g[ rS26{i14}, iS27{i15} ] on stage " +
      std::to_string(stage5.unique_id) +
      " to PipelineVal representing Val T16_l[ iS30{i15} ] on stage " +
      std::to_string(stage6.unique_id) +
      "\n"
      "  PipelineVal representing Val T16_l[ iS30{i15} ] on stage " +
      std::to_string(stage6.unique_id) +
      "\n"
      "  PipelineStage representing Stage " +
      std::to_string(stage6.unique_id) +
      ".Inputs={T14_l[ iS28{i15} ], T15_l[ iS29{i15} ], T16_l[ iS30{i15} ], }. Outputs={T19_g[ rS33{i15} ], }.\n"
      "}\n"
      "Pipeline's outputs:{\n"
      " PipelineVal representing Val T3_g[ rS5{i2} ] on stage " +
      std::to_string(stage1.unique_id) +
      "\n"
      " PipelineVal representing Val T13_g[ rS26{i14}, iS27{i15} ] on stage " +
      std::to_string(stage5.unique_id) +
      "\n"
      " PipelineVal representing Val T19_g[ rS33{i15} ] on stage " +
      std::to_string(stage6.unique_id) +
      "\n"
      "}"};

  // We sort the string so it doesn't depend on the order of the Pipeline's DAG
  // traversal

  // TODO: we should sort on lines, not on characters
  std::sort(obtained_string.begin(), obtained_string.end());
  std::sort(ref_string.begin(), ref_string.end());

  EXPECT_EQ(obtained_string, ref_string);
}


TEST_F(NVFuserTest, ReshardingDetection) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  DeviceMesh mesh0,mesh1, mesh2;
  mesh0 = {0,1};
  mesh1 = {0,2};
  mesh2 = {0,1,2};

  TensorView* tv0 = makeContigTensor(3);
  fusion.addInput(tv0);
  tv0->setDeviceMesh(&mesh0);

  TensorView* tv1 = set(tv0);
  tv1->setDeviceMesh(&mesh0);

  TensorView* tv2 = set(tv1); // resharding
  tv2->setDeviceMesh(&mesh1);

  TensorView* tv3 = set(tv2);
  tv3->setDeviceMesh(&mesh1);

  TensorView* tv4 = set(tv3); // resharding
  tv4->setDeviceMesh(&mesh2);

  TensorView* tv5 = set(tv4); // resharding
  tv5->setDeviceMesh(&mesh2);
  tv5->axis(0)->parallelize(ParallelType::DIDx);

  TensorView* tv6 = set(tv5);
  tv6->setDeviceMesh(&mesh2);
  tv6->axis(0)->parallelize(ParallelType::DIDx);

  TensorView* tv7 = set(tv6); // resharding
  tv7->setDeviceMesh(&mesh2);

  TensorView* tv8 = sum(tv0, {0});
  tv8->setDeviceMesh(&mesh0);

  TensorView* tv9 = sum(tv0, {0}); //resharding, but seems invalid
  tv9->setDeviceMesh(&mesh0);
  tv9->axis(0)->parallelize(ParallelType::DIDx);

  TensorView* tv10 = sum(tv0, {0}); //resharding,
  tv10->setDeviceMesh(&mesh0);
  tv10->axis(1)->parallelize(ParallelType::DIDx);

  TensorView* tv11 = sum(tv0, {0}); //resharding,
  tv11->setDeviceMesh(&mesh1);

  TensorView* tv12 = sum(tv5, {0}); // not resharding
  tv12->setDeviceMesh(&mesh2);
  tv12->axis(0)->parallelize(ParallelType::DIDx);

  TensorView* tv13 = sum(tv5, {0}); // resharding
  tv13->setDeviceMesh(&mesh2);
  tv13->axis(1)->parallelize(ParallelType::DIDx);

  TensorView* tv14 = sum(tv5, {0}); // resharding
  tv14->setDeviceMesh(&mesh2);
  tv14->axis(0)->parallelize(ParallelType::DIDx);
  tv14->axis(1)->parallelize(ParallelType::DIDx);

  TensorView* tv15 = add(tv0, tv1);
  tv15->setDeviceMesh(&mesh0);

  TensorView* tv16 = add(tv0, tv1); //resharding
  tv16->setDeviceMesh(&mesh1);

  TensorView* tv17 = add(tv0, tv1); //resharding
  tv17->setDeviceMesh(&mesh0);
  tv17->axis(0)->parallelize(ParallelType::DIDx);

  TensorView* tv18 = add(tv5, tv6);
  tv18->setDeviceMesh(&mesh2);
  tv18->axis(0)->parallelize(ParallelType::DIDx);

  TensorView* tv19 = add(tv5, tv7); // resharding
  tv19->setDeviceMesh(&mesh2);
  tv19->axis(0)->parallelize(ParallelType::DIDx);

  TensorView* tv20 = add(tv5, tv7); // resharding
  tv20->setDeviceMesh(&mesh2);

  TensorView* tv21 = add(tv0, tv7); // resharding
  tv21->setDeviceMesh(&mesh2);

  TensorView* tv22 = sum(tv5, {1});
  tv22->setDeviceMesh(&mesh2);
  tv22->axis(0)->parallelize(ParallelType::DIDx);

  TensorView* tv23 = sum(tv5, {1}); //resharding
  tv23->setDeviceMesh(&mesh2);

  TensorView* tv24 = sum(tv7, {0});
  tv24->setDeviceMesh(&mesh2);

  TensorView* tv25 = sum(tv7, {0}); //not resharding but invalid
  tv25->setDeviceMesh(&mesh2);
  tv22->axis(0)->parallelize(ParallelType::DIDx);

  TensorView* tv26 = add(tv5, tv6); // resharding
  tv26->setDeviceMesh(&mesh2);

  fusion.addOutput(tv1);
  fusion.addOutput(tv2);
  fusion.addOutput(tv3);
  fusion.addOutput(tv4);
  fusion.addOutput(tv5);
  fusion.addOutput(tv6);
  fusion.addOutput(tv7);
  fusion.addOutput(tv8);
  fusion.addOutput(tv9);
  fusion.addOutput(tv10);
  fusion.addOutput(tv11);
  fusion.addOutput(tv12);
  fusion.addOutput(tv13);
  fusion.addOutput(tv14);
  fusion.addOutput(tv15);
  fusion.addOutput(tv16);
  fusion.addOutput(tv17);
  fusion.addOutput(tv18);
  fusion.addOutput(tv19);
  fusion.addOutput(tv20);
  fusion.addOutput(tv21);
  fusion.addOutput(tv22);
  fusion.addOutput(tv23);
  fusion.addOutput(tv24);
  fusion.addOutput(tv25);
  fusion.addOutput(tv26);

  GTEST_EXPECT_TRUE(!ir_utils::isResharding(tv1->definition()));
  GTEST_EXPECT_TRUE(ir_utils::isResharding(tv2->definition()));
  GTEST_EXPECT_TRUE(!ir_utils::isResharding(tv3->definition()));
  GTEST_EXPECT_TRUE(ir_utils::isResharding(tv4->definition()));
  GTEST_EXPECT_TRUE(ir_utils::isResharding(tv5->definition()));
  GTEST_EXPECT_TRUE(!ir_utils::isResharding(tv6->definition()));
  GTEST_EXPECT_TRUE(ir_utils::isResharding(tv7->definition()));
  GTEST_EXPECT_TRUE(!ir_utils::isResharding(tv8->definition()));
  GTEST_EXPECT_TRUE(ir_utils::isResharding(tv9->definition()));
  GTEST_EXPECT_TRUE(ir_utils::isResharding(tv10->definition()));
  GTEST_EXPECT_TRUE(ir_utils::isResharding(tv11->definition()));
  GTEST_EXPECT_TRUE(!ir_utils::isResharding(tv12->definition()));
  GTEST_EXPECT_TRUE(ir_utils::isResharding(tv13->definition()));
  GTEST_EXPECT_TRUE(ir_utils::isResharding(tv14->definition()));
  GTEST_EXPECT_TRUE(!ir_utils::isResharding(tv15->definition()));
  GTEST_EXPECT_TRUE(ir_utils::isResharding(tv16->definition()));
  GTEST_EXPECT_TRUE(ir_utils::isResharding(tv17->definition()));
  GTEST_EXPECT_TRUE(!ir_utils::isResharding(tv18->definition()));
  GTEST_EXPECT_TRUE(ir_utils::isResharding(tv19->definition()));
  GTEST_EXPECT_TRUE(ir_utils::isResharding(tv20->definition()));
  GTEST_EXPECT_TRUE(ir_utils::isResharding(tv21->definition()));
  GTEST_EXPECT_TRUE(!ir_utils::isResharding(tv22->definition()));
  GTEST_EXPECT_TRUE(ir_utils::isResharding(tv23->definition()));
  GTEST_EXPECT_TRUE(!ir_utils::isResharding(tv24->definition()));
  GTEST_EXPECT_TRUE(!ir_utils::isResharding(tv25->definition()));
  GTEST_EXPECT_TRUE(ir_utils::isResharding(tv26->definition()));
}

PipelineDescriptor segmentedFusionToPipelineDescriptor(SegmentedFusion* sf) {
  std::set<Val*> vals;
  for (auto val: ir_utils::filterByType<TensorView>(sf->completeFusion()->vals())) {
    if (val->isA<TensorView>()){
      vals.insert(val);
    }
  }
  PipelineDescriptor ret;
  for (auto group: sf->groups()) {
    if (group->exprs().empty() || ir_utils::isResharding(group->exprs().at(0))) {
      continue;
    }
    std::vector<Val*> vals_to_add;
    for (auto expr: group->exprs()) {
      for (auto io : {expr->inputs(), expr->outputs()}){
        for (auto val: io){
          if (vals.find(val) != vals.end()) {
            vals_to_add.push_back(val);
            vals.erase(val);
          }
        }
      }
    }
    PipelineStageDescriptor stage;
    stage.addVal(vals_to_add);
    ret.stage_descriptors.push_back(std::move(stage));
  }
  for (auto val: vals) {
    PipelineStageDescriptor stage;
    stage.addVal({val});
    ret.stage_descriptors.push_back(std::move(stage));
  }
  return ret;
}

using automaticSetInsertionTestParams =
    std::tuple<
    DeviceMesh,
    DeviceMesh,
    DeviceMesh,
    bool,
    bool,
    bool>;

class automaticReshardingTest :
  public NVFuserTest,
  public ::testing::WithParamInterface<automaticSetInsertionTestParams> {
protected:
  void SetUp() override {
    fusion = std::make_unique<Fusion>();
    fg = std::make_unique<FusionGuard>(fusion.get());
  }
  void validate() {
    for (auto expr: fusion->exprs()) {
      GTEST_EXPECT_TRUE(!ir_utils::isResharding(expr) || isLowerableToCommunication(expr)) << "on expr=" << expr;
    }

    SegmentCandidateFinderOptions options {
      .run_translate_welford = false,
      .run_combine_reductions = false,
      .run_herrmann_merge = true,
      .run_final_merge = true,
      .only_segment_resharding_exprs = true
    };

    auto segmented_fusion =
        SegmentCandidateFinder::segment(std::move(fusion), options);
    std::cout << toString(segmented_fusion.get()) << std::endl;

    for (auto group: segmented_fusion->groups()) {
      GTEST_EXPECT_TRUE(
        std::all_of(group->exprs().begin(),
                    group->exprs().end(),
                    [](auto expr) { return ir_utils::isResharding(expr);})
        || std::none_of(group->exprs().begin(),
                        group->exprs().end(),
                        [](auto expr) { return ir_utils::isResharding(expr);}));
    }
    // checks that the segments are disjoints and that the graph of segment is acyclic
    segmented_fusion->validate();

    auto pipeline_desc = segmentedFusionToPipelineDescriptor(segmented_fusion.get());
    Pipeline pipeline(segmented_fusion->completeFusion(), pipeline_desc);
    std::cout << "Pipeline:\n" << pipeline.toString() << std::endl;
  }

  std::unique_ptr<Fusion> fusion;
  std::unique_ptr<FusionGuard> fg;
};

TEST_P(automaticReshardingTest, setInsertion) {
  auto [
  mesh0,
  mesh1,
  mesh2,
  is_tv0_tv3_tv5_sharded,
  is_tv1_tv4_sharded,
  is_tv2_sharded
  ] = GetParam();

  TensorView* tv0 = makeContigTensor(3);
  TensorView* tv1 = unaryOp(UnaryOpType::Exp, tv0);
  TensorView* tv2 = binaryOp(BinaryOpType::Add, tv0, tv1);
  TensorView* tv3 = sum(tv2, {0});
  TensorView* tv4 = broadcast(tv3, {true, false, false});
  TensorView* tv5 = binaryOp(BinaryOpType::Mul, tv2, tv4);

  tv0->setDeviceMesh(&mesh0);
  tv1->setDeviceMesh(&mesh1);
  tv2->setDeviceMesh(&mesh2);
  tv3->setDeviceMesh(&mesh0);
  tv4->setDeviceMesh(&mesh1);
  tv5->setDeviceMesh(&mesh0);
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
DeviceMesh Mesh1({1,2});
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

TEST_F(NVFuserTest, 2DMesh) {
  auto fusion = std::make_unique<Fusion>();
  auto fg = std::make_unique<FusionGuard>(fusion.get());

  auto tv0 = makeContigTensor(3);
  tv0->axis(0)->parallelize(ParallelType::DIDx);
  tv0->axis(1)->parallelize(ParallelType::DIDy);
  DeviceMesh mesh ({0,1,2,3,4,5});
  mesh.reshape({2,3});
  tv0->setDeviceMesh(&mesh);
  std::cout << tv0 << std::endl;
}

TEST_F(NVFuserTest, pipelineSegmentation) {
  auto fusion = std::make_unique<Fusion>();
  auto fg = std::make_unique<FusionGuard>(fusion.get());

  auto tv0 = makeContigTensor(3);
  fusion->addInput(tv0);
  TensorView* tv1 = sum(tv0, {0});
  TensorView* tv2 = set(tv1);
  TensorView* tv3 = sum(tv2, {0});
  fusion->addOutput(tv3);

  DeviceMesh mesh0({0});
  DeviceMesh mesh1({0, 1, 2, 3});
  tv0->setDeviceMesh(&mesh0);
  tv1->setDeviceMesh(&mesh0);
  tv2->setDeviceMesh(&mesh1);
  tv3->setDeviceMesh(&mesh1);

  // KernelArgumentHolder inputs;
  SegmentCandidateFinderOptions options {
    .run_translate_welford = false,
    .run_combine_reductions = false,
    .run_herrmann_merge = true,
    .run_final_merge = true,
    .only_segment_resharding_exprs = true
  };

  // auto fusion_copy = std::make_unique<Fusion>(*fusion);
  auto segmented_fusion =
      SegmentCandidateFinder::segment(std::move(fusion), options);
  std::cout << toString(segmented_fusion.get()) << std::endl;

  for (auto group: segmented_fusion->groups()) {
    GTEST_EXPECT_TRUE(
      std::all_of(group->exprs().begin(),
                  group->exprs().end(),
                  [](auto expr) { return ir_utils::isResharding(expr);})
      || std::none_of(group->exprs().begin(),
                      group->exprs().end(),
                      [](auto expr) { return ir_utils::isResharding(expr);}));
  }
  auto pipeline_desc = segmentedFusionToPipelineDescriptor(segmented_fusion.get());
  Pipeline pipeline(segmented_fusion->completeFusion(), pipeline_desc);
  std::cout << "Pipeline:\n" << pipeline.toString() << std::endl;
}

} // namespace nvfuser
