// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#include <gtest/gtest.h>

#include <algorithm>
#include <iostream>

#include <torch/csrc/jit/codegen/cuda/interface.h>

#include <codegen.h>
#include <device_lower/lower2device.h>
#include <disjoint_set.h>
#include <expr_evaluator.h>
#include <fusion.h>
#include <fusion_segmenter.h>
#include <ir/all_nodes.h>
#include <ir/graphviz.h>
#include <ir/iostream.h>
#include <ir/printer.h>
#include <ir/utils.h>
#include <iter_visitor.h>
#include <kernel_ir.h>
#include <logical_domain_map.h>
#include <ops/all_ops.h>
#include <preseg_passes/optimization_pass.h>
#include <preseg_passes/reorder_sharded_axis.h>
#include <runtime/executor.h>
#include <runtime/executor_params.h>
#include <runtime/fusion_executor_cache.h>
#include <scheduler/all_schedulers.h>
#include <scheduler/reduction_utils.h>
#include <scheduler/tools/inlining.h>
#include <scheduler/utils.h>
#include <tests/cpp/multidevice.h>
#include <transform_replay.h>
#include <transform_rfactor.h>

namespace nvfuser {

class PipelineTest : public MultiDeviceTest {
 protected:
  PipelineTest();
  // Utility function used for validation in the tests. It compares the
  // (sharded) outputs with ref_unsharded_outputs. if
  // validate_with_prescribed_values is true, ref_unsharded_outputs is assumed
  // to be set manually in the test body. Otherwise, ref_unsharded_outputs is
  // computed by running a Fusion on a single device with the unsharded_args.
  void validate(bool validate_with_prescribed_values = false);
  void executeAndValidate(bool validate_with_prescribed_values = false);

  std::unique_ptr<MultiDeviceExecutor> runtime;
  std::unique_ptr<Fusion> fusion;
  KernelArgumentHolder args;
  KernelArgumentHolder unsharded_args;
  KernelArgumentHolder outputs;
  KernelArgumentHolder ref_unsharded_outputs;
  hir::HostIrEvaluatorParams host_ir_executor_params;

 private:
  preseg_passes::OptimizationPassGuard<preseg_passes::ReorderShardedAxisPass>
      optimization_guard_;
};

void PipelineTest::validate(bool validate_with_prescribed_values) {
  if (!validate_with_prescribed_values) {
    // execute the fusion on one device without pipeline scheduling
    auto fusion_copy = std::make_unique<Fusion>(*fusion);
    unshard(fusion_copy.get());
    FusionExecutorCache unsharded_fec(std::move(fusion_copy));
    ref_unsharded_outputs = unsharded_fec.runFusionWithInputs(unsharded_args);
  }

  if (debug_print) {
    std::stringstream ss;
    std::string indent = "  ";
    ss << "Device " << communicator_->deviceId()
       << "'s expected (unsharded) outputs:{\n";
    for (auto& t : ref_unsharded_outputs) {
      ss << indent << t;
    }
    ss << "\n}";
    std::cout << ss.str() << std::endl;
  }

  ASSERT_EQ(ref_unsharded_outputs.size(), outputs.size());
  for (int i : arange(fusion->outputs().size())) {
    ASSERT_TRUE(fusion->outputs().at(i)->isA<TensorView>());
    auto output_tv = fusion->outputs().at(i)->as<TensorView>();
    if (!output_tv->getDeviceMesh().has(communicator_->deviceId())) {
      continue;
    }
    auto ref_output =
        shardTensor(ref_unsharded_outputs[i].as<at::Tensor>(), output_tv);
    auto obtained_output = outputs[i].as<at::Tensor>();

    EXPECT_EQ(ref_output.strides(), obtained_output.strides())
        << "Strides are not equal: Ref: " << ref_output.strides()
        << " Output: " << obtained_output.strides() << std::endl;

    EXPECT_TRUE(torch::allclose(ref_output, obtained_output))
        << "Device " << communicator_->deviceId() << " has unexpected output "
        << i << " corresponding to tv " << output_tv
        << ". Expected values: " << ref_output
        << ", obtained values: " << obtained_output;
  }
}

// Run and validate a pipeline
// with given (possibly sharded) inputs
void PipelineTest::executeAndValidate(bool validate_with_prescribed_values) {
  ASSERT_EQ(unsharded_args.size(), fusion->inputs().size());
  for (int i : arange(fusion->inputs().size())) {
    ASSERT_TRUE(fusion->inputs().at(i)->isA<TensorView>());
    args.push(shardTensor(
        unsharded_args[i].as<at::Tensor>(),
        fusion->inputs().at(i)->as<TensorView>()));
  }

  if (debug_print) {
    if (!communicator_->deviceId()) {
      fusion->printKernel();
    }
    std::stringstream ss;
    std::string indent = "  ";
    ss << "Device " << communicator_->deviceId() << "'s args:{\n";
    for (auto& t : args) {
      ss << indent << t;
    }
    ss << "\n}";
    std::cout << ss.str() << std::endl;
  }

  MultiDeviceExecutorParams params;
  params.executor = host_ir_executor_params;
  runtime = std::make_unique<MultiDeviceExecutor>(
      std::make_unique<Fusion>(*fusion), *communicator_, params);
  auto error_msg = runtime->validate();
  if (error_msg != "") {
    GTEST_SKIP() << error_msg;
  }
  outputs = runtime->runWithInput(args);

  if (debug_print) {
    if (!communicator_->deviceId()) {
      runtime->print();
    }
    std::stringstream ss;
    std::string indent = "  ";
    ss << "Device " << communicator_->deviceId() << "'s outputs:{\n";
    for (auto& t : outputs) {
      ss << indent << t;
    }
    ss << "\n}";
    std::cout << ss.str() << std::endl;
  }
  validate(validate_with_prescribed_values);
}

PipelineTest::PipelineTest() : optimization_guard_(false) {
  fusion = std::make_unique<Fusion>();
}

// To run the following tests on several devices, pytorch must be installed with
// the flag USE_DISTRIBUTED=1 and nccl support. With that, nvFuser is built by
// default with NVFUSER_DISTRIBUTED defined. Then, on a node with at least 6
// GPUs, run the test using mpirun: `mpirun -np 6 build/test_multidevice
// --gtest_filter=PipelineTestTwoStages*`.

TEST_F(PipelineTest, Pipeline) {
  const std::vector<int64_t> input_shape1 = {6, 7};
  const std::vector<int64_t> input_shape2 = {3, 5, 2};
  // ===========================================================
  //        FUSION
  // ===========================================================
  FusionGuard fg(fusion.get());

  TensorView* tv0_ = makeConcreteTensor(input_shape1);
  fusion->addInput(tv0_);
  TensorView* tv1_ = sum(tv0_, {0});

  TensorView* tv2_ = set(tv1_);
  TensorView* tv3_ = sum(tv2_, {0});
  fusion->addOutput(tv3_);

  TensorView* tv0 = makeConcreteTensor(input_shape2);
  fusion->addInput(tv0);
  TensorView* tv1 = sum(tv0, {0});

  TensorView* tv2 = set(tv1);
  TensorView* tv2a = set(tv2);
  TensorView* tv3 = sum(tv2a, {0});

  TensorView* tv4 = set(tv1);
  TensorView* tv4a = set(tv4);
  TensorView* tv5 = sum(tv4a, {0});

  TensorView* tv6 = set(tv2a);
  TensorView* tv7 = sum(tv6, {0});
  fusion->addOutput(tv7);

  TensorView* tv8 = set(tv3);
  TensorView* tv9 = set(tv5);
  TensorView* tv10 = set(tv7);
  TensorView* tv11 = add(tv8, tv9);
  TensorView* tv12 = add(tv11, tv10);
  TensorView* tv13 = sum(tv12, {0});
  fusion->addOutput(tv13);

  DeviceMesh mesh0_({5});
  DeviceMesh mesh1_({2, 4});
  DeviceMesh mesh0({0});
  DeviceMesh mesh1({0, 1, 4});
  DeviceMesh mesh2({1, 3});
  DeviceMesh mesh3({2});
  DeviceMesh mesh4({4, 5});

  tv0_->setDeviceMesh(mesh0_);
  tv1_->setDeviceMesh(mesh0_);
  tv2_->setDeviceMesh(mesh1_);
  tv3_->setDeviceMesh(mesh1_);
  tv0->setDeviceMesh(mesh0);
  tv1->setDeviceMesh(mesh0);
  tv2->setDeviceMesh(mesh1);
  tv2a->setDeviceMesh(mesh1);
  tv3->setDeviceMesh(mesh1);
  tv4->setDeviceMesh(mesh2);
  tv4a->setDeviceMesh(mesh2);
  tv5->setDeviceMesh(mesh2);
  tv6->setDeviceMesh(mesh3);
  tv7->setDeviceMesh(mesh3);
  tv8->setDeviceMesh(mesh4);
  tv9->setDeviceMesh(mesh4);
  tv10->setDeviceMesh(mesh4);
  tv11->setDeviceMesh(mesh4);
  tv12->setDeviceMesh(mesh4);
  tv13->setDeviceMesh(mesh4);

  // Create input tensors.
  // Note: each process is binded to a different GPU
  // Note: the concrete values are only used at the relevant ranks
  unsharded_args = {
      at::randn(input_shape1, tensor_options),
      at::randn(input_shape2, tensor_options)};

  executeAndValidate();
}

//(backend type, first stage's mesh, second stage's mesh (if not null), is first
// stage sharded?, is second
// stage sharded?, do_reduction?, sharded dimension, use_fusion_executor_cache?)
using PipelineTestTwoStagesParams = std::tuple<
    CommunicatorBackend,
    DeviceMesh,
    DeviceMesh,
    bool,
    bool,
    bool,
    int,
    bool>;
class PipelineTestTwoStages
    : public PipelineTest,
      public testing::WithParamInterface<PipelineTestTwoStagesParams> {};

TEST_P(PipelineTestTwoStages, Communication) {
  auto
      [backend,
       mesh0,
       mesh1,
       is_stage0_sharded,
       is_stage1_sharded,
       do_reduction,
       sharded_dim,
       use_fusion_executor_cache] = GetParam();
  if (!disable_skip && !communicator_->isBackendAvailable(backend)) {
    GTEST_SKIP() << "Backend not available";
  }
  communicator_->setDefaultBackend(backend);

  if (mesh1.vector().empty()) {
    mesh1 = mesh0;
  }

  std::vector<int64_t> unsharded_input_sizes = {3, 2, 3, 5};
  if (is_stage0_sharded) {
    unsharded_input_sizes[sharded_dim] = mesh0.size();
  }
  if (is_stage1_sharded) {
    unsharded_input_sizes[sharded_dim] = mesh1.size();
    if (do_reduction) {
      ASSERT_EQ(mesh0.size(), mesh1.size());
      unsharded_input_sizes[sharded_dim + 1] = mesh1.size();
    }
  }

  FusionGuard fg(fusion.get());
  TensorView* tv0 = makeConcreteTensor(unsharded_input_sizes);
  TensorView* tv1 = sum(tv0, {3});
  TensorView* tv2 = do_reduction ? sum(tv1, {sharded_dim}) : set(tv1);
  TensorView* tv3 = add(tv2, tv2);
  fusion->addInput(tv0);
  fusion->addOutput(tv3);

  tv0->setDeviceMesh(mesh0);
  tv1->setDeviceMesh(mesh0);
  tv2->setDeviceMesh(mesh1);
  tv3->setDeviceMesh(mesh1);
  if (is_stage0_sharded) {
    tv0->axis(sharded_dim)->parallelize(ParallelType::DIDx);
    tv1->axis(sharded_dim)->parallelize(ParallelType::DIDx);
  }
  if (is_stage1_sharded) {
    // in case of reduction, tv2's sharded_dim is a reduction axis, except if it
    // was initially of size 1, in which case it is simply removed.
    int axis = (do_reduction) ? sharded_dim + 1 : sharded_dim;
    tv2->axis(axis)->parallelize(ParallelType::DIDx);
    tv3->axis(sharded_dim)->parallelize(ParallelType::DIDx);
  }

  unsharded_args = {at::randn(unsharded_input_sizes, tensor_options)};

  if (use_fusion_executor_cache) {
    host_ir_executor_params.use_fusion_executor_cache = true;
    host_ir_executor_params.skip_auto_scheduling = true;
  }

  executeAndValidate();
}

namespace {
auto all_backends =
    testing::Values(CommunicatorBackend::kNccl, CommunicatorBackend::kUcc);

DeviceMesh mesh_null;
DeviceMesh mesh0({0});
DeviceMesh mesh1({1});
DeviceMesh mesh2({0, 1, 2, 3});
DeviceMesh mesh3({0, 2, 3});
DeviceMesh mesh4({1, 0, 2});
DeviceMesh mesh5({1, 0});
auto all_meshes = testing::Values(mesh0, mesh1, mesh2, mesh3, mesh4, mesh5);
auto all_nontrivial_meshes = testing::Values(mesh2, mesh3, mesh4, mesh5);

} // namespace

INSTANTIATE_TEST_SUITE_P(
    Gather,
    PipelineTestTwoStages,
    testing::Combine(
        testing::Values(CommunicatorBackend::kNccl, CommunicatorBackend::kUcc),
        all_meshes,
        all_meshes,
        testing::Values(true),
        testing::Values(false),
        testing::Values(false),
        testing::Values(0),
        testing::Bool()));

INSTANTIATE_TEST_SUITE_P(
    Scatter,
    PipelineTestTwoStages,
    testing::Combine(
        testing::Values(CommunicatorBackend::kNccl, CommunicatorBackend::kUcc),
        testing::Values(mesh0, mesh1),
        testing::Values(mesh2, mesh4, mesh5),
        testing::Values(false),
        testing::Values(true),
        testing::Values(false),
        testing::Values(0),
        testing::Bool()));

INSTANTIATE_TEST_SUITE_P(
    Bcast,
    PipelineTestTwoStages,
    testing::Combine(
        testing::Values(CommunicatorBackend::kNccl, CommunicatorBackend::kUcc),
        all_meshes,
        all_meshes,
        testing::Values(false),
        testing::Values(false),
        testing::Values(false),
        testing::Values(0, 1),
        testing::Bool()));

INSTANTIATE_TEST_SUITE_P(
    Bcast_sharded,
    PipelineTestTwoStages,
    testing::Combine(
        // TODO(#2794): add back CommunicatorBackend::kUcc
        testing::Values(CommunicatorBackend::kNccl),
        testing::Values(mesh3, mesh4),
        testing::Values(mesh3, mesh4),
        testing::Values(true),
        testing::Values(true),
        testing::Values(false),
        testing::Values(0, 1),
        testing::Bool()));

INSTANTIATE_TEST_SUITE_P(
    Bcast_sharded_same_mesh,
    PipelineTestTwoStages,
    testing::Combine(
        testing::Values(CommunicatorBackend::kNccl, CommunicatorBackend::kUcc),
        testing::Values(mesh0, mesh1),
        testing::Values(mesh_null), // the same mesh is used for all tensors
        testing::Values(true),
        testing::Values(true),
        testing::Values(false),
        testing::Values(0, 1),
        testing::Bool()));

INSTANTIATE_TEST_SUITE_P(
    Reduce,
    PipelineTestTwoStages,
    testing::Combine(
        testing::Values(CommunicatorBackend::kNccl, CommunicatorBackend::kUcc),
        all_nontrivial_meshes,
        all_meshes,
        testing::Values(true),
        testing::Values(false),
        testing::Values(true),
        testing::Values(0, 1),
        testing::Values(false)));

INSTANTIATE_TEST_SUITE_P(
    ReduceScatter,
    PipelineTestTwoStages,
    testing::Combine(
        testing::Values(CommunicatorBackend::kNccl, CommunicatorBackend::kUcc),
        all_nontrivial_meshes,
        testing::Values(mesh_null), // the same mesh is used for all tensors
        testing::Values(true),
        testing::Values(true),
        testing::Values(true),
        testing::Values(0),
        testing::Values(false)));

// TODO: Distributed reduction tests using fusion executor cache are failing
// AllocationDomainPass might be re-ordering compute
INSTANTIATE_TEST_SUITE_P(
    DISABLED_FusionExecutorCache_Reduce,
    PipelineTestTwoStages,
    testing::Combine(
        testing::Values(CommunicatorBackend::kNccl, CommunicatorBackend::kUcc),
        all_nontrivial_meshes,
        all_meshes,
        testing::Values(true),
        testing::Values(false),
        testing::Values(true),
        testing::Values(0, 1),
        testing::Values(true)));

INSTANTIATE_TEST_SUITE_P(
    DISABLED_FusionExecutorCache_ReduceScatter,
    PipelineTestTwoStages,
    testing::Combine(
        testing::Values(CommunicatorBackend::kNccl, CommunicatorBackend::kUcc),
        all_nontrivial_meshes,
        testing::Values(mesh_null), // the same mesh is used for all tensors
        testing::Values(true),
        testing::Values(true),
        testing::Values(true),
        testing::Values(0, 1),
        testing::Values(true)));

} // namespace nvfuser
