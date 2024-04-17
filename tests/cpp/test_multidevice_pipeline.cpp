// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#include <gtest/gtest.h>

#include <codegen.h>
#include <device_lower/lower2device.h>
#include <disjoint_set.h>
#include <executor.h>
#include <executor_params.h>
#include <expr_evaluator.h>
#include <fusion.h>
#include <fusion_segmenter.h>
#include <ir/all_nodes.h>
#include <ir/graphviz.h>
#include <ir/iostream.h>
#include <ir/printer.h>
#include <ir/utils.h>
#include <iter_visitor.h>
#include <kernel_cache.h>
#include <kernel_ir.h>
#include <mma_type.h>
#include <ops/all_ops.h>
#include <root_domain_map.h>
#include <scheduler/all_schedulers.h>
#include <scheduler/reduction_utils.h>
#include <scheduler/utils.h>
#include <tests/cpp/multidevice.h>
#include <torch/csrc/jit/codegen/cuda/interface.h>
#include <transform_replay.h>
#include <transform_rfactor.h>

#include <algorithm>
#include <iostream>

namespace nvfuser {

using namespace torch::jit::fuser::cuda;
using namespace at::indexing;

// To run the following tests on several devices, pytorch must be installed with
// the flag USE_DISTRIBUTED=1 and nccl support. With that, nvFuser is built by
// default with NVFUSER_DISTRIBUTED defined.  Then simply run the tests on
// several processes. For example, run `mpirun -np 6 build/test_multidevice
// --gtest_filter=PipelineTwoStages*` on a node having at least GPUs.

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
  if (!disable_skip && !communicator->isBackendAvailable(backend)) {
    GTEST_SKIP() << "Backend not available";
  }
  communicator->setDefaultBackend(backend);

  if (mesh1.vector().empty()) {
    mesh1 = mesh0;
  }

  std::vector<int64_t> unsharded_input_sizes = {3, 2, 3, 5};
  if (is_stage0_sharded) {
    unsharded_input_sizes[sharded_dim] = mesh0.vector().size();
  }
  if (is_stage1_sharded) {
    unsharded_input_sizes[sharded_dim] = mesh1.vector().size();
    if (do_reduction) {
      ASSERT_EQ(mesh0.vector().size(), mesh1.vector().size());
      unsharded_input_sizes[sharded_dim + 1] = mesh1.vector().size();
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
    multi_device_executor_params.use_fusion_executor_cache = true;
    multi_device_executor_params.skip_auto_scheduling = true;
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
auto all_meshes = testing::Values(mesh0, mesh1, mesh2, mesh3, mesh4);
auto all_nontrivial_meshes = testing::Values(mesh2, mesh3, mesh4);

} // namespace

INSTANTIATE_TEST_SUITE_P(
    Gather,
    PipelineTestTwoStages,
    testing::Combine(
        testing::Values(CommunicatorBackend::nccl),
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
        testing::Values(CommunicatorBackend::nccl),
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
        testing::Values(CommunicatorBackend::nccl),
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
        testing::Values(CommunicatorBackend::nccl),
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
        testing::Values(CommunicatorBackend::nccl),
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
        testing::Values(CommunicatorBackend::nccl),
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
        testing::Values(CommunicatorBackend::nccl),
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
        testing::Values(CommunicatorBackend::nccl),
        all_nontrivial_meshes,
        testing::Values(mesh_null), // the same mesh is used for all tensors
        testing::Values(true),
        testing::Values(true),
        testing::Values(true),
        testing::Values(0, 1),
        testing::Values(true)));

// TODO: UCC PipelineTestTwoStages are hanging in UCC barrier
// when number of processes > number of gpus required by test.
INSTANTIATE_TEST_SUITE_P(
    DISABLED_UCC_Gather,
    PipelineTestTwoStages,
    testing::Combine(
        testing::Values(CommunicatorBackend::ucc),
        all_meshes,
        all_meshes,
        testing::Values(true),
        testing::Values(false),
        testing::Values(false),
        testing::Values(0, 1),
        testing::Bool()));

INSTANTIATE_TEST_SUITE_P(
    DISABLED_UCC_Scatter,
    PipelineTestTwoStages,
    testing::Combine(
        testing::Values(CommunicatorBackend::ucc),
        all_meshes,
        all_meshes,
        testing::Values(false),
        testing::Values(true),
        testing::Values(false),
        testing::Values(0, 1),
        testing::Bool()));

INSTANTIATE_TEST_SUITE_P(
    DISABLED_UCC_Bcast,
    PipelineTestTwoStages,
    testing::Combine(
        testing::Values(CommunicatorBackend::ucc),
        all_meshes,
        all_meshes,
        testing::Values(false),
        testing::Values(false),
        testing::Values(false),
        testing::Values(0, 1),
        testing::Bool()));

INSTANTIATE_TEST_SUITE_P(
    DISABLED_UCC_Bcast_sharded,
    PipelineTestTwoStages,
    testing::Combine(
        testing::Values(CommunicatorBackend::ucc),
        testing::Values(mesh3, mesh4),
        testing::Values(mesh3, mesh4),
        testing::Values(true),
        testing::Values(true),
        testing::Values(false),
        testing::Values(0, 1),
        testing::Bool()));

INSTANTIATE_TEST_SUITE_P(
    DISABLED_UCC_Bcast_sharded_same_mesh,
    PipelineTestTwoStages,
    testing::Combine(
        testing::Values(CommunicatorBackend::ucc),
        testing::Values(mesh0, mesh1),
        testing::Values(mesh_null), // the same mesh is used for all tensors
        testing::Values(true),
        testing::Values(true),
        testing::Values(false),
        testing::Values(0, 1),
        testing::Bool()));

INSTANTIATE_TEST_SUITE_P(
    DISABLED_UCC_Reduce,
    PipelineTestTwoStages,
    testing::Combine(
        testing::Values(CommunicatorBackend::ucc),
        all_nontrivial_meshes,
        all_meshes,
        testing::Values(true),
        testing::Values(false),
        testing::Values(true),
        testing::Values(0, 1),
        testing::Bool()));

INSTANTIATE_TEST_SUITE_P(
    DISABLED_UCC_ReduceScatter,
    PipelineTestTwoStages,
    testing::Combine(
        testing::Values(CommunicatorBackend::ucc),
        all_nontrivial_meshes,
        testing::Values(mesh_null), // the same mesh is used for all tensors
        testing::Values(true),
        testing::Values(true),
        testing::Values(true),
        testing::Values(0, 1),
        testing::Bool()));

// Different scheduling modes used in
// PipelineTestStagedReduction.StagedReduction
enum class SchedulingMode {
  // Manual interdevice scheduling, no intra-device scheduling
  InterDeviceOnly,
  // Manual inter-/intra-device scheduling
  Manual,
  // Manual inter-device scheduling, composed with ReductionOnly
  // intra-device schedule
  ReductionOnly,
  // Manual inter-device scheduling, composed with fully automated intra-device
  // scheduling (through FusionExecutorCache)
  Automatic,
};

std::ostream& operator<<(std::ostream& out, const SchedulingMode& mode) {
  switch (mode) {
    case SchedulingMode::InterDeviceOnly:
      return out << "SchedulingMode::InterDeviceOnly";
    case SchedulingMode::Manual:
      return out << "SchedulingMode::Manual";
    case SchedulingMode::ReductionOnly:
      return out << "SchedulingMode::ReductionOnly";
    case SchedulingMode::Automatic:
      return out << "SchedulingMode::Automatic";
    default:
      NVF_ERROR(false);
  }
  return out;
}
class PipelineTestStagedReduction
    : public PipelineTest,
      public ::testing::WithParamInterface<SchedulingMode> {};

// 1D staged reduction
// Inputs: X[A,B,C]
TEST_P(PipelineTestStagedReduction, StagedReduction) {
  auto scheduling_mode = GetParam();

  int num_devices = communicator->size();
  int A = num_devices;
  int B = 8;
  int C = 64;
  std::vector<int64_t> unsharded_input_sizes = {A, B, C};
  std::vector<int64_t> input_sizes(unsharded_input_sizes);
  input_sizes[0] = 1;

  FusionGuard fg(fusion.get());
  TensorView* tv0 = makeConcreteTensor(unsharded_input_sizes);
  TensorView* tv1 = sum(tv0, {2});
  TensorView* tv_out = sum(tv1, {0});
  fusion->addInput(tv0);
  fusion->addOutput(tv_out);

  // multi device scheduling:
  std::vector<int64_t> devices(num_devices);
  std::iota(devices.begin(), devices.end(), 0);
  DeviceMesh mesh(devices);
  for (auto tv : {tv0, tv1, tv_out}) {
    tv->setDeviceMesh(mesh);
  }
  for (auto tv : {tv0, tv1}) {
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

      // Incrementally, can print in between for debugging
      tv0->computeAt(tv2, 2);
      tv2->computeAt(tv3, 2);
      tv3->computeAt(tv1, 2);

      // Re do it all at once, because why not.
      tv0->computeAt(tv1, 2);

      tv2->axis(3)->parallelize(ParallelType::Unroll);
      tv1->axis(1)->parallelize(ParallelType::BIDx);
      tv1->setMemoryType(
          MemoryType::Global); // necessary to avoid runtime error

      tv1->axis(-1)->parallelize(ParallelType::TIDx);
      tv2->axis(-1)->parallelize(ParallelType::TIDx);
      tv3->axis(-1)->parallelize(ParallelType::TIDx);
      break;
    }
    case SchedulingMode::ReductionOnly: {
      auto reduction_params = getReductionHeuristics(
          fusion.get(), {at::empty(input_sizes, tensor_options)});
      NVF_CHECK(reduction_params, "Reduction schedule was not generated!");
      l_params = reduction_params->lparams;
      scheduleReduction(fusion.get(), *reduction_params);
      break;
    }
    case SchedulingMode::Automatic:
      multi_device_executor_params.use_fusion_executor_cache = true;
      break;
  }

  unsharded_inputs = {at::randn(unsharded_input_sizes, tensor_options)};
  ref_unsharded_outputs = {at::sum(
      unsharded_inputs.at(0).toTensor(), at::OptionalIntArrayRef({0, 2}))};

  executeAndValidate(/* validate_with_prescribed_values */ true);
}

INSTANTIATE_TEST_SUITE_P(
    SchedulingModes,
    PipelineTestStagedReduction,
    testing::Values(
        SchedulingMode::InterDeviceOnly,
        SchedulingMode::Manual,
        SchedulingMode::ReductionOnly,
        SchedulingMode::Automatic));
} // namespace nvfuser
