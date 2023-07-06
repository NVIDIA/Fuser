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
#include <multidevice/pipeline.h>
#include <ops/all_ops.h>
#include <test/utils.h>

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
      std::move(stage0),
      std::move(stage1),
      std::move(stage2),
      std::move(stage3),
      std::move(stage4),
      std::move(stage5),
      std::move(stage6)}}; // the order doesnt matter
  Pipeline pipeline(&fusion, std::move(descriptor));

  // Validation
  std::string obtained_string = pipeline.toString();
  std::string ref_string{
      "Pipeline's inputs{:\n"
      " PipelineVal representing Val T0_g[ iS0{i0}, iS1{i2} ] on stage 0\n"
      " PipelineVal representing Val T4_g[ iS6{i7}, iS7{i8}, iS8{i9} ] on stage 2\n"
      "}\n"
      "Pipeline's Traversal inputs --> outputs {\n"
      "  PipelineStage representing Stage 0.Inputs={T0_g[ iS0{i0}, iS1{i2} ], }. Outputs={T1_l[ rS2{i0}, iS3{i2} ], }.\n"
      "  PipelineVal representing Val T1_l[ rS2{i0}, iS3{i2} ] on stage 0\n"
      "  PipelineCommunication that transfers PipelineVal representing Val T1_l[ rS2{i0}, iS3{i2} ] on stage 0 to PipelineVal representing Val T2_l[ iS4{i2} ] on stage 1\n"
      "  PipelineVal representing Val T2_l[ iS4{i2} ] on stage 1\n"
      "  PipelineStage representing Stage 1.Inputs={T2_l[ iS4{i2} ], }. Outputs={T3_g[ rS5{i2} ], }.\n"
      "  PipelineStage representing Stage 2.Inputs={T4_g[ iS6{i7}, iS7{i8}, iS8{i9} ], }. Outputs={T5_l[ rS9{i7}, iS10{i8}, iS11{i9} ], }.\n"
      "  PipelineVal representing Val T5_l[ rS9{i7}, iS10{i8}, iS11{i9} ] on stage 2\n"
      "  PipelineCommunication that transfers PipelineVal representing Val T5_l[ rS9{i7}, iS10{i8}, iS11{i9} ] on stage 2 to PipelineVal representing Val T6_l[ iS12{i8}, iS13{i9} ] on stage 3\n"
      "  PipelineVal representing Val T6_l[ iS12{i8}, iS13{i9} ] on stage 3\n"
      "  PipelineStage representing Stage 3.Inputs={T6_l[ iS12{i8}, iS13{i9} ], }. Outputs={T7_l[ iS14{i8}, iS15{i9} ], T8_l[ rS16{i8}, iS17{i9} ], }.\n"
      "  PipelineVal representing Val T7_l[ iS14{i8}, iS15{i9} ] on stage 3\n"
      "  PipelineCommunication that transfers PipelineVal representing Val T7_l[ iS14{i8}, iS15{i9} ] on stage 3 to PipelineVal representing Val T12_l[ iS24{i8}, iS25{i9} ] on stage 5\n"
      "  PipelineVal representing Val T12_l[ iS24{i8}, iS25{i9} ] on stage 5\n"
      "  PipelineStage representing Stage 5.Inputs={T12_l[ iS24{i8}, iS25{i9} ], }. Outputs={T13_g[ rS26{i8}, iS27{i9} ], }.\n"
      "  PipelineVal representing Val T8_l[ rS16{i8}, iS17{i9} ] on stage 3\n"
      "  PipelineCommunication that transfers PipelineVal representing Val T8_l[ rS16{i8}, iS17{i9} ] on stage 3 to PipelineVal representing Val T14_l[ iS28{i9} ] on stage 6\n"
      "  PipelineVal representing Val T14_l[ iS28{i9} ] on stage 6\n"
      "  PipelineCommunication that transfers PipelineVal representing Val T5_l[ rS9{i7}, iS10{i8}, iS11{i9} ] on stage 2 to PipelineVal representing Val T9_l[ iS18{i8}, iS19{i9} ] on stage 4\n"
      "  PipelineVal representing Val T9_l[ iS18{i8}, iS19{i9} ] on stage 4\n"
      "  PipelineStage representing Stage 4.Inputs={T9_l[ iS18{i8}, iS19{i9} ], }. Outputs={T11_l[ rS22{i8}, iS23{i9} ], }.\n"
      "  PipelineVal representing Val T11_l[ rS22{i8}, iS23{i9} ] on stage 4\n"
      "  PipelineCommunication that transfers PipelineVal representing Val T11_l[ rS22{i8}, iS23{i9} ] on stage 4 to PipelineVal representing Val T15_l[ iS29{i9} ] on stage 6\n"
      "  PipelineVal representing Val T15_l[ iS29{i9} ] on stage 6\n"
      "  PipelineCommunication that transfers PipelineVal representing Val T13_g[ rS26{i8}, iS27{i9} ] on stage 5 to PipelineVal representing Val T16_l[ iS30{i9} ] on stage 6\n"
      "  PipelineVal representing Val T16_l[ iS30{i9} ] on stage 6\n"
      "  PipelineStage representing Stage 6.Inputs={T14_l[ iS28{i9} ], T15_l[ iS29{i9} ], T16_l[ iS30{i9} ], }. Outputs={T19_g[ rS33{i9} ], }.\n"
      "}\n"
      "Pipeline's outputs:{\n"
      " PipelineVal representing Val T3_g[ rS5{i2} ] on stage 1\n"
      " PipelineVal representing Val T13_g[ rS26{i8}, iS27{i9} ] on stage 5\n"
      " PipelineVal representing Val T19_g[ rS33{i9} ] on stage 6\n"
      "}"};

  // We sort the string so it doesn't depend on the order of the Pipeline's DAG
  // traversal
  std::sort(obtained_string.begin(), obtained_string.end());
  std::sort(ref_string.begin(), ref_string.end());

  TORCH_INTERNAL_ASSERT(
      obtained_string == ref_string,
      "the obtained Pipeline is not the one expected");
}

} // namespace nvfuser
