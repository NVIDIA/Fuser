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
#include <multidevice/aggregate_dag.h>
#include <multidevice/multicluster_fusion.h>
#include <ops/all_ops.h>
#include <test/utils.h>

#include <algorithm>
#include <iostream>

namespace nvfuser {

using namespace at::indexing;

TEST_F(NVFuserTest, MultiClusterFusion_CUDA) {
  MultiClusterFusion fusion;
  FusionGuard fg(&fusion);

  TensorView* tv = makeContigTensor(3);
  auto index_a = IrBuilder::create<Int>(0);
  auto index_b = IrBuilder::create<Int>(1);
  fusion.addInput(tv);

  fusion.newCluster({.process_rank = 0});
  auto tv0 = add(tv, tv);
  fusion.addClusterOutput(tv0);

  fusion.newCluster({.process_rank = 1});
  auto tva = select(tv0, 0, index_a);
  TensorView* tva1 = sum(tva, {0});
  fusion.addClusterOutput(tva1);

  fusion.newCluster({.process_rank = 2});
  auto tvb = select(tv0, 0, index_b);
  TensorView* tvb1 = sum(tvb, {0});
  fusion.addClusterOutput(tvb1);

  fusion.newCluster({.process_rank = 3});
  TensorView* tv2 = add(tva1, tvb1);
  fusion.addClusterOutput(tv2);

  fusion.addOutput(tv2);

  // Validation
  std::string obtained_string_MCF = fusion.toString();
  std::string ref_string_MCF{
      "MultiClusterFusion {\n"
      "  g0 {(auto_schedule=1, process_rank=0)\n"
      "    inputs:\n"
      "      T0_g[ iS0{i0}, iS1{i2}, iS2{i3} ]\n"
      "    exprs:\n"
      "          T1_l[ iS3{i0}, iS4{i2}, iS5{i3} ]\n"
      "       = T0_g[ iS0{i0}, iS1{i2}, iS2{i3} ]\n"
      "       + T0_g[ iS0{i0}, iS1{i2}, iS2{i3} ];\n"
      "\n"
      "    outputs:\n"
      "      T1_l[ iS3{i0}, iS4{i2}, iS5{i3} ]\n"
      "  }\n"
      "  g1 {(auto_schedule=1, process_rank=1)\n"
      "    inputs:\n"
      "      T1_l[ iS3{i0}, iS4{i2}, iS5{i3} ]\n"
      "    exprs:\n"
      "          T2_l[ iS6{i2}, iS7{i3} ]\n"
      "       = select( T1_l[ iS3{i0}, iS4{i2}, iS5{i3} ], axis = iS3{i0}, index = 0 )\n"
      "\n"
      "          T3_l[ rS8{i2}, iS9{i3} ]\n"
      "       = reduction( T2_l[ iS6{i2}, iS7{i3} ], op = add, initial value = double(0), allreduce = false )\n"
      "\n"
      "    outputs:\n"
      "      T3_l[ rS8{i2}, iS9{i3} ]\n"
      "  }\n"
      "  g2 {(auto_schedule=1, process_rank=2)\n"
      "    inputs:\n"
      "      T1_l[ iS3{i0}, iS4{i2}, iS5{i3} ]\n"
      "    exprs:\n"
      "          T4_l[ iS10{i2}, iS11{i3} ]\n"
      "       = select( T1_l[ iS3{i0}, iS4{i2}, iS5{i3} ], axis = iS3{i0}, index = 1 )\n"
      "\n"
      "          T5_l[ rS12{i2}, iS13{i3} ]\n"
      "       = reduction( T4_l[ iS10{i2}, iS11{i3} ], op = add, initial value = double(0), allreduce = false )\n"
      "\n"
      "    outputs:\n"
      "      T5_l[ rS12{i2}, iS13{i3} ]\n"
      "  }\n"
      "  g3 {(auto_schedule=1, process_rank=3)\n"
      "    inputs:\n"
      "      T3_l[ rS8{i2}, iS9{i3} ]\n"
      "      T5_l[ rS12{i2}, iS13{i3} ]\n"
      "    exprs:\n"
      "          T6_g[ iS14{i3} ]\n"
      "       = T3_l[ rS8{i2}, iS9{i3} ]\n"
      "       + T5_l[ rS12{i2}, iS13{i3} ];\n"
      "\n"
      "    outputs:\n"
      "      T6_g[ iS14{i3} ]\n"
      "  }\n"
      "} //MultiClusterFusion"};

  std::sort(obtained_string_MCF.begin(), obtained_string_MCF.end());
  std::sort(ref_string_MCF.begin(), ref_string_MCF.end());

  TORCH_INTERNAL_ASSERT(
      obtained_string_MCF == ref_string_MCF,
      "the obtained MultiClusterFusion is not the one expected");

  std::string obtained_string_aDag = fusion.aggregateDag()->toString();
  std::string ref_string_aDag{
      "AggregateDag's inputs{:\n"
      " AggregateVal representing Val T0_g[ iS0{i0}, iS1{i2}, iS2{i3} ] on cluster 0\n"
      "}\n"
      "AggregateDag's Traversal inputs --> outputs {\n"
      "  AggregateExpr representing Cluster 0.Inputs={T0_g[ iS0{i0}, iS1{i2}, iS2{i3} ], }. Outputs={T1_l[ iS3{i0}, iS4{i2}, iS5{i3} ], }.\n"
      "  AggregateVal representing Val T1_l[ iS3{i0}, iS4{i2}, iS5{i3} ] on cluster 0\n"
      "  Send/Receive Val {T1_l[ iS3{i0}, iS4{i2}, iS5{i3} ]} from cluster 0 to cluster 1\n"
      "  AggregateVal representing Val T1_l[ iS3{i0}, iS4{i2}, iS5{i3} ] on cluster 1\n"
      "  AggregateExpr representing Cluster 1.Inputs={T1_l[ iS3{i0}, iS4{i2}, iS5{i3} ], }. Outputs={T3_l[ rS8{i2}, iS9{i3} ], }.\n"
      "  AggregateVal representing Val T3_l[ rS8{i2}, iS9{i3} ] on cluster 1\n"
      "  Send/Receive Val {T3_l[ rS8{i2}, iS9{i3} ]} from cluster 1 to cluster 3\n"
      "  AggregateVal representing Val T3_l[ rS8{i2}, iS9{i3} ] on cluster 3\n"
      "  Send/Receive Val {T1_l[ iS3{i0}, iS4{i2}, iS5{i3} ]} from cluster 0 to cluster 2\n"
      "  AggregateVal representing Val T1_l[ iS3{i0}, iS4{i2}, iS5{i3} ] on cluster 2\n"
      "  AggregateExpr representing Cluster 2.Inputs={T1_l[ iS3{i0}, iS4{i2}, iS5{i3} ], }. Outputs={T5_l[ rS12{i2}, iS13{i3} ], }.\n"
      "  AggregateVal representing Val T5_l[ rS12{i2}, iS13{i3} ] on cluster 2\n"
      "  Send/Receive Val {T5_l[ rS12{i2}, iS13{i3} ]} from cluster 2 to cluster 3\n"
      "  AggregateVal representing Val T5_l[ rS12{i2}, iS13{i3} ] on cluster 3\n"
      "  AggregateExpr representing Cluster 3.Inputs={T3_l[ rS8{i2}, iS9{i3} ], T5_l[ rS12{i2}, iS13{i3} ], }. Outputs={T6_g[ iS14{i3} ], }.\n"
      "}\n"
      "AggregateDag's outputs:{\n"
      " AggregateVal representing Val T6_g[ iS14{i3} ] on cluster 3\n"
      "}"};

  std::sort(obtained_string_aDag.begin(), obtained_string_aDag.end());
  std::sort(ref_string_aDag.begin(), ref_string_aDag.end());

  TORCH_INTERNAL_ASSERT(
      obtained_string_aDag == ref_string_aDag,
      "the obtained AggregateDag is not the one expected");
}

} // namespace nvfuser
