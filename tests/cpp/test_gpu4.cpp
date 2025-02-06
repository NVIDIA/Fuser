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

#include <codegen.h>
#include <debug.h>
#include <device_lower/lower2device.h>
#include <device_lower/pass/magic_zero.h>
#include <device_lower/pass/replace_size.h>
#include <disjoint_set.h>
#include <expr_evaluator.h>
#include <fusion.h>
#include <fusion_segmenter.h>
#include <grouped_reduction.h>
#include <id_model/id_model.h>
#include <ir/all_nodes.h>
#include <ir/builder.h>
#include <ir/graphviz.h>
#include <ir/iostream.h>
#include <ir/utils.h>
#include <iter_visitor.h>
#include <kernel_ir.h>
#include <kernel_ir_dispatch.h>
#include <logical_domain_map.h>
#include <ops/all_ops.h>
#include <runtime/executor.h>
#include <runtime/executor_params.h>
#include <runtime/fusion_executor_cache.h>
#include <scheduler/all_schedulers.h>
#include <scheduler/reduction_utils.h>
#include <scheduler/tools/abstract_tensor.h>
#include <scheduler/tools/inlining.h>
#include <scheduler/utils.h>
#include <tests/cpp/utils.h>
#include <tests/cpp/validator.h>
#include <transform_replay.h>
#include <transform_rfactor.h>

#include <torch/csrc/jit/api/function_impl.h>
#include <torch/csrc/jit/codegen/cuda/interface.h>
#include <torch/torch.h>

#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/Exceptions.h>
#include <c10/cuda/CUDAStream.h>

#include <algorithm>
#include <cmath>
#include <sstream>
#include "parallel_dimension_map.h"

namespace nvfuser {

using namespace at::indexing;

TEST_F(NVFuserTest, IntRNG_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto input_tv = makeContigConcreteTensor({4 * 128 * 4});
  fusion.addInput(input_tv);

  constexpr float kDropoutProbability = 0.9;
  constexpr float kScale = 1.0f / kDropoutProbability;

  auto prob = IrBuilder::create<Val>(kDropoutProbability);
  auto scale = IrBuilder::create<Val>(kScale);

  // dropout start
  auto rand_vals = rand_like(input_tv);
  auto mask = lt(rand_vals, prob);
  auto apply_mask = mul(input_tv, mask);
  auto output_tv = mul(apply_mask, scale);
  // dropout end
  //   fusion.addOutput(mask);
  fusion.addOutput(output_tv);

  auto inp_cache = input_tv->cacheAfter();
  output_tv->cacheBefore();

  output_tv->split(0, 4);
  output_tv->split(0, 128);
  output_tv->axis(0)->parallelize(ParallelType::BIDx);

  TransformPropagator propagator(output_tv);
  MaxLogicalDomainInfoSpanningTree spanning_tree(output_tv);
  spanning_tree.traverse(&propagator);
  scheduler_utils::parallelizeAllLike(output_tv);

  inp_cache->axis(-1)->parallelize(ParallelType::Vectorize);
  rand_vals->axis(-1)->parallelize(ParallelType::Unroll);
  output_tv->axis(-1)->parallelize(ParallelType::Vectorize);

  inlineMost();

  fusion.printMath();
  fusion.printKernel();
}

} // namespace nvfuser
