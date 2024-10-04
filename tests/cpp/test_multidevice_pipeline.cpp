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
#include <inlining.h>
#include <ir/all_nodes.h>
#include <ir/graphviz.h>
#include <ir/iostream.h>
#include <ir/printer.h>
#include <ir/utils.h>
#include <iter_visitor.h>
#include <kernel_ir.h>
#include <logical_domain_map.h>
#include <ops/all_ops.h>
#include <runtime/executor.h>
#include <runtime/executor_params.h>
#include <runtime/fusion_executor_cache.h>
#include <scheduler/all_schedulers.h>
#include <scheduler/reduction_utils.h>
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
  // computed by running a Fusion on a single device with the unsharded_inputs
  void validate(bool validate_with_prescribed_values = false);
  void executeAndValidate(bool validate_with_prescribed_values = false);

  std::unique_ptr<MultiDeviceExecutor> runtime;
  std::unique_ptr<Fusion> fusion;
  std::vector<c10::IValue> inputs;
  std::vector<c10::IValue> unsharded_inputs;
  std::vector<at::Tensor> outputs;
  std::vector<at::Tensor> ref_unsharded_outputs;
  hir::HostIrExecutorParams host_ir_executor_params;
};

void PipelineTest::validate(bool validate_with_prescribed_values) {
  if (!validate_with_prescribed_values) {
    // execute the fusion on one device without pipeline scheduling
    auto fusion_copy = std::make_unique<Fusion>(*runtime->completeFusion());
    unshard(fusion_copy.get());
    FusionExecutorCache unsharded_fec(std::move(fusion_copy));
    ref_unsharded_outputs = unsharded_fec.runFusionWithInputs(unsharded_inputs);
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
  for (int i : c10::irange(runtime->completeFusion()->outputs().size())) {
    ASSERT_TRUE(runtime->completeFusion()->outputs().at(i)->isA<TensorView>());
    auto output_tv =
        runtime->completeFusion()->outputs().at(i)->as<TensorView>();
    if (!output_tv->getDeviceMesh().has(communicator_->deviceId())) {
      continue;
    }
    auto ref_output = shardTensor(ref_unsharded_outputs.at(i), output_tv);
    auto obtained_output = outputs.at(i);
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
  ASSERT_EQ(unsharded_inputs.size(), fusion->inputs().size());
  for (int i : c10::irange(fusion->inputs().size())) {
    ASSERT_TRUE(fusion->inputs().at(i)->isA<TensorView>());
    auto input_tv = fusion->inputs().at(i)->as<TensorView>();
    auto input = shardTensor(unsharded_inputs.at(i).toTensor(), input_tv);
    inputs.push_back(input);
  }

  if (debug_print) {
    if (!communicator_->deviceId()) {
      fusion->printKernel();
    }
    std::stringstream ss;
    std::string indent = "  ";
    ss << "Device " << communicator_->deviceId() << "'s inputs:{\n";
    for (auto& t : inputs) {
      ss << indent << t;
    }
    ss << "\n}";
    std::cout << ss.str() << std::endl;
  }

  runtime = std::make_unique<MultiDeviceExecutor>(
      std::move(fusion), *communicator_, host_ir_executor_params);
  auto error_msg = runtime->validate();
  if (error_msg != "") {
    GTEST_SKIP() << error_msg;
  }
  outputs = runtime->runWithInput(inputs);

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

PipelineTest::PipelineTest() {
  fusion = std::make_unique<Fusion>();
  communicator_->setDefaultBackend(CommunicatorBackend::nccl);
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
  unsharded_inputs = {
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

  unsharded_inputs = {at::randn(unsharded_input_sizes, tensor_options)};

  if (use_fusion_executor_cache) {
    host_ir_executor_params.use_fusion_executor_cache = true;
    host_ir_executor_params.skip_auto_scheduling = true;
  }

  executeAndValidate();
}

namespace {
auto all_backends =
    testing::Values(CommunicatorBackend::nccl, CommunicatorBackend::ucc);

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
        testing::Values(CommunicatorBackend::nccl, CommunicatorBackend::ucc),
        all_meshes,
        all_meshes,
        testing::Values(true),
        testing::Values(false),
        testing::Values(false),
        testing::Values(0, 1),
        testing::Bool()));

INSTANTIATE_TEST_SUITE_P(
    Scatter,
    PipelineTestTwoStages,
    testing::Combine(
        testing::Values(CommunicatorBackend::nccl, CommunicatorBackend::ucc),
        all_meshes,
        all_meshes,
        testing::Values(false),
        testing::Values(true),
        testing::Values(false),
        testing::Values(0, 1),
        testing::Bool()));

INSTANTIATE_TEST_SUITE_P(
    Bcast,
    PipelineTestTwoStages,
    testing::Combine(
        testing::Values(CommunicatorBackend::nccl, CommunicatorBackend::ucc),
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
        // TODO(#2794): add back CommunicatorBackend::ucc
        testing::Values(CommunicatorBackend::nccl),
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
        testing::Values(CommunicatorBackend::nccl, CommunicatorBackend::ucc),
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
        testing::Values(CommunicatorBackend::nccl, CommunicatorBackend::ucc),
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
        testing::Values(CommunicatorBackend::nccl, CommunicatorBackend::ucc),
        all_nontrivial_meshes,
        testing::Values(mesh_null), // the same mesh is used for all tensors
        testing::Values(true),
        testing::Values(true),
        testing::Values(true),
        testing::Values(0, 1),
        testing::Values(false)));

// TODO: Distributed reduction tests using fusion executor cache are failing
// AllocationDomainPass might be re-ordering compute
INSTANTIATE_TEST_SUITE_P(
    DISABLED_FusionExecutorCache_Reduce,
    PipelineTestTwoStages,
    testing::Combine(
        testing::Values(CommunicatorBackend::nccl, CommunicatorBackend::ucc),
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
        testing::Values(CommunicatorBackend::nccl, CommunicatorBackend::ucc),
        all_nontrivial_meshes,
        testing::Values(mesh_null), // the same mesh is used for all tensors
        testing::Values(true),
        testing::Values(true),
        testing::Values(true),
        testing::Values(0, 1),
        testing::Values(true)));

// Different scheduling modes used in
// PipelineTestStagedReduction.StagedReduction
enum class SchedulingMode {
  // Manual interdevice scheduling, no intra-device scheduling
  InterDeviceOnly,
  // Manual inter-/intra-device scheduling
  Manual,
  // Manual inter-device scheduling, composed with fully automated intra-device
  // scheduling (through FusionExecutorCache)
  Automatic,
};

std::ostream& operator<<(std::ostream& out, const SchedulingMode& mode) {
  switch (mode) {
    case SchedulingMode::InterDeviceOnly:
      out << "InterDeviceOnly";
      break;
    case SchedulingMode::Manual:
      out << "Manual";
      break;
    case SchedulingMode::Automatic:
      out << "Automatic";
      break;
  }
  return out;
}

class PipelineTestStagedReduction
    : public PipelineTest,
      public ::testing::WithParamInterface<SchedulingMode> {};

// 1D staged reduction
// Inputs: X[num_devices,B,C]
TEST_P(PipelineTestStagedReduction, StagedReduction) {
  auto scheduling_mode = GetParam();

  const int num_devices = communicator_->size();
  constexpr int B = 8;
  constexpr int C = 64;

  FusionGuard fg(fusion.get());
  // The first dimension is made symbolic so `tv_out->definition()` won't
  // become a squeeze when num_devices == 1. This wouldn't be a problem for
  // automatic mode. However, for the manual mode, the scheduling code below
  // assumes `tv_out->definition()` can be lowered to communication. A squeeze
  // can't.
  TensorView* tv0 = TensorViewBuilder()
                        .dtype(DataType::Float)
                        .contiguity(true)
                        .shape({-1, B, C})
                        .build();
  auto mesh = DeviceMesh::createForNumDevices(num_devices);
  tv0->setDeviceMesh(mesh);
  TensorView* tv1 = sum(tv0, {2});
  TensorView* tv_out = sum(tv1, {0});
  fusion->addInput(tv0);
  fusion->addOutput(tv_out);

  for (auto* tv : {tv0, tv1}) {
    tv->axis(0)->parallelize(ParallelType::DIDx);
  }

  // Intra-device reduction scheduling for the first reduction:
  switch (scheduling_mode) {
    case SchedulingMode::InterDeviceOnly:
      break;
    case SchedulingMode::Manual: {
      // inspired from NVFuserTest.FusionReduction1_CUDA
      // tv0[I0{A}, I1{B}, I2{C}]
      tv1->split(2, 32);
      // tv1[I0{A}, I1{B}, R2o{C/32}, R2i{32}] = tv0[I0{A}, I1{B}, I2{C}]
      tv1->split(2, 4);
      // clang-format off
      // tv1[I0{A}, I1{B}, R2oo{C/32/4)}, R2oi{4}, R2i{32}] = tv0[I0{A}, I1{B}, I2{C}]
      // clang-format on

      TensorView* tv2 = tv1->rFactor({2});
      // clang-format off
      // tv2[I0{A}, I1{B}, R2oo{C/32/4)}, I2oi{4}, I2i{32}] = tv0[I0{A}, I1{B}, I2{C}]
      // tv1[I0{A}, I1{B},                R2oi{4}, R2i{32}] = tv2[I0{A}, I1{B}, R2oo{C/32/4)}, I2oi{4}, I2i{32}]
      // clang-format on

      TensorView* tv3 = tv1->rFactor({2});
      // clang-format off
      // tv2[I0{A}, I1{B}, R2oo{C/32/4)}, I2oi{4}, I2i{32}] = tv0[I0{A}, I1{B}, I2{C}]
      // tv3[I0{A}, I1{B},                R2oi{4}, I2i{32}] = tv2[I0{A}, I1{B}, R2oo{C/32/4)}, I2oi{4}, I2i{32}]
      // tv1[I0{A}, I1{B},                         R2i{32}] = tv3[I0{A}, I1{B},                R2oi{4}, I2i{32}]
      // clang-format on

      // tv1 is a segment boundary so must be in global. This wouldn't be
      // needed if the fusion were scheduled automatically.
      tv1->setMemoryType(MemoryType::Global);

      // Use `tv2` as the reference tensor because it contains the most
      // parallel IterDomains.
      tv2->axis(1)->parallelize(ParallelType::BIDx);
      tv2->axis(3)->parallelize(ParallelType::Unroll);
      tv2->axis(-1)->parallelize(ParallelType::TIDx);
      scheduler_utils::parallelizeAllLike(
          tv2,
          /*pos=*/-1,
          // Don't propagate the parallelization to `tv_out` because that's in
          // a different, resharding segment.
          /*selected_tv=*/{tv0, tv1, tv2, tv3});
      inlineMost();
      break;
    }
    case SchedulingMode::Automatic:
      host_ir_executor_params.use_fusion_executor_cache = true;
      break;
  }

  at::Tensor unsharded_input_tensor =
      at::randn({num_devices, B, C}, tensor_options);
  at::Tensor ref_unsharded_output_tensor =
      unsharded_input_tensor.sum(at::IntArrayRef({0, 2}));
  unsharded_inputs = {unsharded_input_tensor};
  ref_unsharded_outputs = {ref_unsharded_output_tensor};

  executeAndValidate(/* validate_with_prescribed_values */ true);
}

INSTANTIATE_TEST_SUITE_P(
    ,
    PipelineTestStagedReduction,
    testing::Values(
        SchedulingMode::InterDeviceOnly,
        SchedulingMode::Manual,
        SchedulingMode::Automatic),
    testing::PrintToStringParamName());

} // namespace nvfuser
