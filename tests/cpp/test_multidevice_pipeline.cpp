// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#ifdef NVFUSER_DISTRIBUTED
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
#include <ir/interface_nodes.h>
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
#include <scheduler/mma_utils.h>
#include <scheduler/reduction_utils.h>
#include <scheduler/utils.h>
#include <tests/cpp/validator.h>
#include <tests/cpp/multidevice.h>
#include <torch/csrc/jit/codegen/cuda/interface.h>
#include <transform_replay.h>
#include <transform_rfactor.h>

#include <algorithm>
#include <chrono>
#include <iostream>

namespace nvfuser {

using namespace torch::jit::fuser::cuda;
using namespace at::indexing;

/* To run the following tests on several devices, pytorch must be installed
   with the flag NVFUSER_DISTRIBUTED=1 and nccl support.
   Then simply run the tests on several processes, for example using mpirun
   on a node having at least 6 GPUs,
   e.g.: mpirun -np 6 build/nvfuser_tests
   --gtest_filter=PipelineTest.Pipeline
*/

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
// stage sharded?, do_reduction?, use_fusion_executor_cache?)
using PipelineTestTwoStagesParams = std::
    tuple<CommunicatorBackend, DeviceMesh, DeviceMesh, bool, bool, bool, bool>;
class PipelineTestTwoStages
    : public PipelineTest,
      public ::testing::WithParamInterface<PipelineTestTwoStagesParams> {};

TEST_P(PipelineTestTwoStages, Communication) {
  auto
      [backend,
       mesh0,
       mesh1,
       is_stage0_sharded,
       is_stage1_sharded,
       do_reduction,
       use_fusion_executor_cache] = GetParam();
  if (!disable_skip && !communicator->isBackendAvailable(backend)) {
    GTEST_SKIP() << "Backend not available";
  }
  communicator->setDefaultBackend(backend);

  if (mesh1.vector().empty()) {
    mesh1 = mesh0;
  }

  int first_axis_extent = 3;
  if (is_stage0_sharded) {
    first_axis_extent = mesh0.vector().size();
  } else if (is_stage1_sharded) {
    first_axis_extent = mesh1.vector().size();
  }
  int second_axis_extent = 2;
  if (is_stage1_sharded && do_reduction) {
    GTEST_ASSERT_EQ(mesh0.vector().size(), mesh1.vector().size());
    second_axis_extent = mesh1.vector().size();
  }
  std::vector<int64_t> unsharded_input_sizes = {
      first_axis_extent, second_axis_extent, 3, 5};

  FusionGuard fg(fusion.get());
  TensorView* tv0 = makeConcreteTensor(unsharded_input_sizes);
  TensorView* tv1 = sum(tv0, {3});
  TensorView* tv2 = do_reduction ? sum(tv1, {0}) : set(tv1);
  TensorView* tv3 = sum(tv2, {1});
  fusion->addInput(tv0);
  fusion->addOutput(tv3);

  tv0->setDeviceMesh(mesh0);
  tv1->setDeviceMesh(mesh0);
  tv2->setDeviceMesh(mesh1);
  tv3->setDeviceMesh(mesh1);
  if (is_stage0_sharded) {
    tv0->axis(0)->parallelize(ParallelType::DIDx);
    tv1->axis(0)->parallelize(ParallelType::DIDx);
  }
  if (is_stage1_sharded) {
    // in case of reduction, axis(0) of tv2 is a reduction axis, except if it
    // was initially of size 1, in which case it is simply removed.
    int tv2_outmost_axis = (do_reduction && second_axis_extent > 1) ? 1 : 0;
    tv2->axis(tv2_outmost_axis)->parallelize(ParallelType::DIDx);
    tv3->axis(0)->parallelize(ParallelType::DIDx);
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
    ::testing::Values(CommunicatorBackend::nccl, CommunicatorBackend::ucc);

DeviceMesh mesh_null;
DeviceMesh mesh0({0});
DeviceMesh mesh1({1});
DeviceMesh mesh2({0, 1, 2, 3});
DeviceMesh mesh3({0, 2, 3});
DeviceMesh mesh4({1, 0, 2});
auto all_meshes = ::testing::Values(mesh0, mesh1, mesh2, mesh3, mesh4);
auto all_nontrivial_meshes = ::testing::Values(mesh2, mesh3, mesh4);

} // namespace

INSTANTIATE_TEST_SUITE_P(
    Gather,
    PipelineTestTwoStages,
    ::testing::Combine(
        all_backends,
        all_meshes,
        all_meshes,
        ::testing::Values(true),
        ::testing::Values(false),
        ::testing::Values(false),
        ::testing::Bool()));

INSTANTIATE_TEST_SUITE_P(
    Scatter,
    PipelineTestTwoStages,
    ::testing::Combine(
        all_backends,
        all_meshes,
        all_meshes,
        ::testing::Values(false),
        ::testing::Values(true),
        ::testing::Values(false),
        ::testing::Bool()));

INSTANTIATE_TEST_SUITE_P(
    Bcast,
    PipelineTestTwoStages,
    ::testing::Combine(
        all_backends,
        all_meshes,
        all_meshes,
        ::testing::Values(false),
        ::testing::Values(false),
        ::testing::Values(false),
        ::testing::Bool()));

INSTANTIATE_TEST_SUITE_P(
    Bcast_sharded,
    PipelineTestTwoStages,
    ::testing::Combine(
        all_backends,
        ::testing::Values(mesh3, mesh4),
        ::testing::Values(mesh3, mesh4),
        ::testing::Values(true),
        ::testing::Values(true),
        ::testing::Values(false),
        ::testing::Bool()));

INSTANTIATE_TEST_SUITE_P(
    Bcast_sharded_same_mesh,
    PipelineTestTwoStages,
    ::testing::Combine(
        all_backends,
        ::testing::Values(mesh0, mesh1),
        ::testing::Values(mesh_null), // the same mesh is used for all tensors
        ::testing::Values(true),
        ::testing::Values(true),
        ::testing::Values(false),
        ::testing::Bool()));

INSTANTIATE_TEST_SUITE_P(
    Reduce,
    PipelineTestTwoStages,
    ::testing::Combine(
        all_backends,
        all_nontrivial_meshes,
        all_meshes,
        ::testing::Values(true),
        ::testing::Values(false),
        ::testing::Values(true),
        ::testing::Bool()));

INSTANTIATE_TEST_SUITE_P(
    ReduceScatter,
    PipelineTestTwoStages,
    ::testing::Combine(
        all_backends,
        all_nontrivial_meshes,
        ::testing::Values(mesh_null), // the same mesh is used for all tensors
        ::testing::Values(true),
        ::testing::Values(true),
        ::testing::Values(true),
        ::testing::Bool()));

enum class SchedulingMode {
  InterDeviceOnly,
  Manual,
};

std::ostream& operator<<(std::ostream& out, const SchedulingMode& mode) {
  switch (mode) {
    case SchedulingMode::InterDeviceOnly:
      return out << "InterDeviceOnly";
    case SchedulingMode::Manual:
      return out << "Manual";
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
  }

  unsharded_inputs = {at::randn(unsharded_input_sizes, tensor_options)};
  ref_unsharded_outputs = {at::sum(
      unsharded_inputs.at(0).toTensor(), at::OptionalIntArrayRef({0, 2}))};

  executeAndValidate(/* validate_with_prescribed_values */ true);
}

INSTANTIATE_TEST_SUITE_P(
    SchedulingModes,
    PipelineTestStagedReduction,
    ::testing::Values(SchedulingMode::InterDeviceOnly, SchedulingMode::Manual));

inline std::vector<at::Tensor> executeAndTime(
      MultiDeviceExecutor& runtime, std::vector<c10::IValue>& inputs) {
  auto start = std::chrono::high_resolution_clock::now();
  auto outputs = runtime.runWithInput(inputs);
  auto end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> duration = end - start;
  std::cout << "First run : " << duration.count() << "seconds" << std::endl;

  start = std::chrono::high_resolution_clock::now();
  for (auto i = 0; i < 5; i++) {
    runtime.runWithInput(inputs);
  }
  end = std::chrono::high_resolution_clock::now();
  duration = end - start;
  std::cout << "Average time taken: " << duration.count() / 5.0 << "seconds" << std::endl;
  return outputs;
}

// option = is output sharded
class DistributedMatmul : public MultiDeviceTest,
                          public ::testing::WithParamInterface<bool> {};

TEST_P(DistributedMatmul, LayoutTN) {
  // MmaLayout::TN matmul A(T), B(N), C(T)
  // Sharding outer-most axis
  //  A     B     C   |   Comms
  //  t     f     t   |   none
  //  t     f     f   |   allgather
  auto is_output_sharded = GetParam();
  std::unique_ptr<Fusion> fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());
  int num_devices = communicator->size();
  std::vector<int64_t> devices(num_devices);
  std::iota(devices.begin(), devices.end(), 0);
  DeviceMesh mesh(devices);

  int64_t M = 64, N = 32, K = 64;
  // TODO: until we support split, manually split axes
  int64_t Mo = num_devices;
  int64_t Mi = M / Mo;
  std::vector<int64_t> a_shape = {Mo, Mi, K};
  std::vector<int64_t> b_shape = {N, K};

  // MmaLayout::TN matmul
  TensorView* a =
      makeContigConcreteTensor(a_shape, DataType::Half); // (Mo, Mi, K)
  TensorView* b = makeContigConcreteTensor(b_shape, DataType::Half); // (N,K)
  TensorView* a_b = broadcast(a, {false, false, true, false}); // (Mo,Mi,b,K)
  TensorView* b_b = broadcast(b, {true, true, false, false}); // (b,b,N,K)
  TensorView* ab = mul(a_b, b_b); // (Mo,Mi,N,K)
  TensorView* c = sum(ab, {-1}); // (Mo,Mi,N,r)

  fusion->addInput(a);
  fusion->addInput(b);
  fusion->addOutput(c);

  // Sharding M dimension
  auto all_sharded_tvs = {a, a_b, b_b, ab};
  for (auto tv : all_sharded_tvs) {
    tv->axis(0)->parallelize(ParallelType::DIDx);
    tv->setDeviceMesh(mesh);
  }
  b->setDeviceMesh(mesh);
  c->setDeviceMesh(mesh);
  if (is_output_sharded) {
    c->axis(0)->parallelize(ParallelType::DIDx);
  }

  auto tensor_options =
      at::TensorOptions().dtype(at::kHalf).device(communicator->device());
  auto a_ = at::randn(a_shape, tensor_options);
  auto b_ = at::randn(b_shape, tensor_options);
  auto c_ =
      at::matmul(
          a_.view({M, K}).to(at::kDouble), b_.transpose(0, 1).to(at::kDouble))
          .view({Mo, Mi, N})
          .to(at::kFloat);

  std::vector<c10::IValue> inputs = {
      shardTensor(a_, mesh, communicator->deviceId()), b_};
  auto expected_output =
      is_output_sharded ? shardTensor(c_, mesh, communicator->deviceId()) : c_;

  MultiDeviceExecutorParams params{true, false, false};
  MultiDeviceExecutor runtime(std::move(fusion), *communicator, params);
  auto outputs = runtime.runWithInput(inputs);
  // auto outputs = executeAndTime(runtime, inputs);
  testValidate(
      runtime.completeFusion(),
      outputs,
      inputs,
      {expected_output},
      __LINE__,
      __FILE__);
}

INSTANTIATE_TEST_SUITE_P(NoComms, DistributedMatmul, ::testing::Values(true));
INSTANTIATE_TEST_SUITE_P(
    AllGather,
    DistributedMatmul,
    ::testing::Values(false));

TEST_F(MultiDeviceTest, MatmulNT_AllReduce) {
  // MmaLayout::NT matmul A(N), B(T), C(T)
  // Sharding: A, B are sharded along K. C is replicated.
  std::unique_ptr<Fusion> fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());
  int num_devices = communicator->size();
  std::vector<int64_t> devices(num_devices);
  std::iota(devices.begin(), devices.end(), 0);
  DeviceMesh mesh(devices);

  // Note: Manually split K into Ko(device dim), Ki until split supported.
  int64_t M = 64, N = 32, K = 64;
  int64_t Ko = num_devices, Ki = K / Ko;
  std::vector<int64_t> a_shape = {Ko, Ki, M};
  std::vector<int64_t> b_shape = {Ko, Ki, N};

  TensorView* a =
      makeContigConcreteTensor(a_shape, DataType::Half); // (Ko,Ki,M)
  TensorView* b =
      makeContigConcreteTensor(b_shape, DataType::Half); // (Ko,Ki,N)
  // Transpose into TN layout, keep Ko (device axis) as the outermost.
  TensorView* a_t = transpose(a, 1, 2); // (Ko, M, Ki)
  TensorView* b_t = transpose(b, 1, 2); // (Ko, N, Ki)
  TensorView* a_b = broadcast(a_t, {false, false, true, false}); // (Ko,M,b,Ki)
  TensorView* b_b = broadcast(b_t, {false, true, false, false}); // (Ko,b,N,Ki)
  TensorView* ab = mul(a_b, b_b); // (Ko,M,N,Ki)
  TensorView* c0 = sum(ab, {-1}); // (Ko,M,N,r)
  TensorView* c = sum(c0, {0}); // (r,M,N)

  // For architectures with NT support
  // TensorView* a_b = broadcast(a, {false, false, false, true}); // (Ko,Ki,M,b)
  // TensorView* b_b = broadcast(b, {false, false, true, false}); // (Ko,Ki,b,N)
  // TensorView* ab = mul(a_b, b_b); // (Ko,Ki,M,N)
  // TensorView* c0 = sum(ab, {1}); // (Ko,r,M,N)
  // TensorView* c = sum(c0, {0}); // (r,M,N)
  fusion->addInput(a);
  fusion->addInput(b);
  fusion->addOutput(c);

  // Parallelize K on all inputs and intermediates.
  auto all_sharded_tvs = {a, b, a_t, b_t, a_b, b_b, ab, c0};
  for (auto tv : all_sharded_tvs) {
    tv->axis(0)->parallelize(ParallelType::DIDx);
    tv->setDeviceMesh(mesh);
  }
  // output is replicated on all devices.
  c->setDeviceMesh(mesh);

  auto tensor_options =
      at::TensorOptions().dtype(at::kHalf).device(communicator->device());
  auto a_ = at::randn(a_shape, tensor_options) * 10;
  auto b_ = at::randn(b_shape, tensor_options) * 10;
  auto c_ = at::matmul(
                a_.view({K, M}).transpose(0, 1).to(at::kDouble),
                b_.view({K, N}).to(at::kDouble))
                .view({M, N})
                .to(at::kFloat);

  std::vector<c10::IValue> inputs = {
      shardTensor(a_, mesh, communicator->deviceId()),
      shardTensor(b_, mesh, communicator->deviceId())};

  MultiDeviceExecutorParams params{true, false, false};
  MultiDeviceExecutor runtime(std::move(fusion), *communicator, params);
  // auto outputs = runtime.runWithInput(inputs);
  auto outputs = executeAndTime(runtime, inputs);
  testValidate(
      runtime.completeFusion(), outputs, inputs, {c_}, __LINE__, __FILE__);
}

TEST_F(MultiDeviceTest, MatmulNT_ReduceScatter) {
  // MmaLayout::NT matmul A(N), B(T), C(T)
  // A, B are sharded on K. C is sharded on M
  std::unique_ptr<Fusion> fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());
  int num_devices = communicator->size();
  std::vector<int64_t> devices(num_devices);
  std::iota(devices.begin(), devices.end(), 0);
  DeviceMesh mesh(devices);

  // Note: Manually split K and M
  int64_t M = 64, N = 32, K = 64;
  int64_t Ko = num_devices, Ki = K / Ko;
  int64_t Mo = num_devices, Mi = M / Mo;
  std::vector<int64_t> a_shape = {Ko, Ki, M};
  std::vector<int64_t> b_shape = {Ko, Ki, N};

  TensorView* a =
      makeContigConcreteTensor(a_shape, DataType::Half); // (Ko,Ki,M)
  TensorView* b =
      makeContigConcreteTensor(b_shape, DataType::Half); // (Ko,Ki,N)
  TensorView* a_t = transpose(a, 1, 2); // (Ko, M, Ki)
  TensorView* b_t = transpose(b, 1, 2); // (Ko, N, Ki)
  TensorView* a_b = broadcast(a_t, {false, false, true, false}); // (Ko,M,b,Ki)
  TensorView* b_b = broadcast(b_t, {false, true, false, false}); // (Ko,b,N,Ki)
  TensorView* ab = mul(a_b, b_b); // (Ko,M,N,Ki)
  TensorView* c0 = sum(ab, {-1}); // (Ko,M,N,r)
  c0 = segment_set(c0);
  std::vector<int64_t> orig_size = {1, M, N}; // Note: {Ko,M,N}, but Ko is sharded. 48, 24
  std::vector<int64_t> new_size = {1, Mo, Mi, N}; 
  TensorView* c1 = reshape(c0, orig_size, new_size); 
  TensorView* c = sum(c1, {0}); 

  fusion->addInput(a);
  fusion->addInput(b);
  fusion->addOutput(c);

  // Sharding K dimension of all inputs and intermediates.
  auto all_sharded_tvs = {a, b, a_t, b_t, a_b, b_b, ab, c0, c1};
  for (auto tv : all_sharded_tvs) {
    tv->axis(0)->parallelize(ParallelType::DIDx);
    tv->setDeviceMesh(mesh);
  }
  // Sharding M on output
  c->setDeviceMesh(mesh);
  c->axis(1)->parallelize(ParallelType::DIDx);

  auto tensor_options =
      at::TensorOptions().dtype(at::kHalf).device(communicator->device());
  auto a_ = at::randn(a_shape, tensor_options) * 10;
  auto b_ = at::randn(b_shape, tensor_options) * 10;

  std::vector<c10::IValue> inputs = {
      shardTensor(a_, mesh, communicator->deviceId()),
      shardTensor(b_, mesh, communicator->deviceId())};

  auto c_ = at::matmul(
                a_.view({K, M}).transpose(0, 1).to(at::kDouble),
                b_.view({K, N}).to(at::kDouble))
                .view({Mo, Mi, N})
                .to(at::kFloat);
  auto expected_output =
      shardTensor(c_, mesh, communicator->deviceId())
          .view({1, Mi, N}); // TODO: hacked because split is manual
  
  MultiDeviceExecutorParams params{true, false, false};
  MultiDeviceExecutor runtime(std::move(fusion), *communicator, params);
  auto outputs = runtime.runWithInput(inputs);
  // auto outputs = executeAndTime(runtime, inputs);
  testValidate(
      runtime.completeFusion(),
      outputs,
      inputs,
      {expected_output},
      __LINE__,
      __FILE__);
}

} // namespace nvfuser

#endif
