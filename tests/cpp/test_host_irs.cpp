// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#include <gtest/gtest.h>

#include <executor_kernel_arg.h>
#include <fusion.h>
#include <fusion_segmenter.h>
#include <ir/all_nodes.h>
#include <ir/builder.h>
#include <multidevice/lower_communication.h>
#include <multidevice/utils.h>
#include <ops/all_ops.h>
#include <test/utils.h>

#include <algorithm>
#include <iostream>

namespace nvfuser {

class FusionExecutorWithExternalForLoop {
public:
  FusionExecutorWithExternalForLoop(
      std::unique_ptr<Fusion> fusion) : fec_(std::move(fusion)) {}

  at::Tensor runWithInputs_v0(const at::ArrayRef<c10::IValue>& inputs) {
    auto input = inputs.at(0);
    auto aten_input = input.toTensor();
    auto for_loop_extent = aten_input.sizes().at(0);
    std::vector<at::Tensor> outputs;
    for (int for_loop_index = 0; for_loop_index < for_loop_extent; for_loop_index++) {
      c10::IValue input_i = input.toTensor().index({at::indexing::Slice(for_loop_index, for_loop_index + 1), "..."});
      outputs.push_back(fec_.runFusionWithInputs({input_i}).at(0));
    }
    return at::concat(outputs);
  }

  at::Tensor runWithInputs(const at::ArrayRef<c10::IValue>& inputs) {
    auto input = inputs.at(0);
    auto aten_input = input.toTensor();
    auto for_loop_extent = aten_input.sizes().at(0);
    std::vector<at::Tensor> outputs;
    for (int for_loop_index = 0; for_loop_index < for_loop_extent; for_loop_index++) {
      std::cout << "for_loop_index=" << for_loop_index 
                << ", for_loop_extent=" << for_loop_extent
                << ", running kernel:\n";
      fec_.fusion()->printKernel();
      std::cout << std::endl;
      c10::IValue input_i = input.toTensor().index({at::indexing::Slice(for_loop_index, for_loop_index + 1), "..."});
      outputs.push_back(fec_.runFusionWithInputs({input_i}).at(0));
    }
    return at::concat(outputs);
  }

private:
  FusionExecutorCache fec_;
};

class CpuForLoopTest: public NVFuserTest {};

TEST_F(CpuForLoopTest, pointwiseKernelSingleIO) {
  auto fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());

  TensorView* tv0 = makeContigTensor({2});
  fusion->addInput(tv0);
  TensorView* tv1 = add(tv0, tv0);
  TensorView* tv2 = sum(tv1, {1});
  fusion->addOutput(tv2);

  auto options = at::TensorOptions().device(at::kCUDA, 0);
  c10::IValue input = at::randn({4,8}, options);
  auto ref_output = at::sum(input.toTensor() * 2, {1});

  FusionExecutorWithExternalForLoop executor(std::move(fusion));
  auto output = executor.runWithInputs_v0({input});

  GTEST_EXPECT_TRUE(torch::allclose(ref_output, output));
}

TEST_F(CpuForLoopTest, kernelSingleIO) {
  auto fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());

  TensorView* tv0 = makeContigTensor({2});
  fusion->addInput(tv0);
  TensorView* tv1 = add(tv0, tv0);
  TensorView* tv2 = sum(tv1, {1});
  fusion->addOutput(tv2);

  auto options = at::TensorOptions().device(at::kCUDA, 0);
  c10::IValue input = at::randn({4,8}, options);
  auto ref_output = at::sum(input.toTensor() * 2, {1});

  tv0->axis(0)->parallelize(ParallelType::CPU);
  tv1->axis(0)->parallelize(ParallelType::CPU);
  tv2->axis(0)->parallelize(ParallelType::CPU);
  fusion->print();

  FusionExecutorWithExternalForLoop executor(std::move(fusion));
  auto output = executor.runWithInputs({input});

  GTEST_EXPECT_TRUE(torch::allclose(ref_output, output));
}

} // namespace nvfuser
