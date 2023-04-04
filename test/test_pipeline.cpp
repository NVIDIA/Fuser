// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#include <gtest/gtest.h>

#include <fusion.h>
#include <ir_all_nodes.h>
#include <ir_builder.h>
#include <multidevice/pipeline.h>
#include <ops/all_ops.h>
#include <test/utils.h>

#include <algorithm>
#include <iostream>

namespace nvfuser {

using namespace at::indexing;

TEST_F(NVFuserTest, Pipeline_CUDA) {
  // Fusion definition
  Fusion fusion = fusion;
  FusionGuard fg(&fusion);

  TensorView* tv0_ = makeContigTensor(2);
  fusion.addInput(tv0_);
  TensorView* tv1_ = sum(tv0_, {0});

  TensorView* tv2_ = set(tv1_);
  TensorView* tv3_ = sum(tv2_, {0});
  fusion.addOutput(tv3_);

  TensorView* tv0 = makeContigTensor(3);
  fusion.addInput(tv0);
  TensorView* tv1 = sum(tv0, {0});

  TensorView* tv2 = set(tv1);
  TensorView* tv2a = set(tv2);
  TensorView* tv3 = sum(tv2a, {0});

  TensorView* tv4 = set(tv1);
  TensorView* tv4a = set(tv4);
  TensorView* tv5 = sum(tv4a, {0});

  TensorView* tv6 = set(tv2a);
  TensorView* tv7 = sum(tv6, {0});
  fusion.addOutput(tv7);

  TensorView* tv8 = set(tv3);
  TensorView* tv9 = set(tv5);
  TensorView* tv10 = set(tv7);
  TensorView* tv11 = add(tv8, tv9);
  TensorView* tv12 = add(tv11, tv10);
  TensorView* tv13 = sum(tv12, {0});
  fusion.addOutput(tv13);

  // Pipeline scheduling
  PipelineStageDescriptor stage0_, stage1_, stage0, stage1, stage2, stage3, stage4; 
  stage0_.addVal({tv0_, tv1_});
  stage1_.addVal({tv2_, tv3_});
  stage0.addVal({tv0, tv1});
  stage1.addVal({tv2, tv2a, tv3});
  stage2.addVal({tv4, tv4a, tv5});
  stage3.addVal({tv6, tv7});
  stage4.addVal({tv8, tv9, tv10, tv11, tv12, tv13});

  PipelineDescriptor descriptor {.stageDescriptors {&stage0_, &stage1_, &stage0, &stage1, &stage2, &stage3, &stage4}}; // the order doesnt matter
  Pipeline pipeline(&fusion, descriptor);

  // Validation
  std::string obtained_string = pipeline.toString();
  std::string ref_string {
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

  // We sort the string so it doesn't depend on the order of the Pipeline's DAG traversal
  std::sort(obtained_string.begin(), obtained_string.end());
  std::sort(ref_string.begin(), ref_string.end());

  TORCH_INTERNAL_ASSERT(
      obtained_string == ref_string,
      "the obtained Pipeline is not the one expected");
}

} // namespace nvfuser
