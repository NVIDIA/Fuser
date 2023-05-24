// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#ifdef USE_DISTRIBUTED
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
#include <multidevice/aggregate_dag.h>
#include <multidevice/multicluster_fusion.h>
#include <multidevice/multidevice_runtime.h>
#include <mutator.h>
#include <ops/all_ops.h>
#include <root_domain_map.h>
#include <scheduler/all_schedulers.h>
#include <scheduler/reduction_utils.h>
#include <scheduler/utils.h>
#include <test/utils.h>
#include <test/validator.h>
#include <torch/csrc/jit/codegen/cuda/interface.h>
#include <transform_replay.h>
#include <transform_rfactor.h>

// fuser and IR parser
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/Exceptions.h>
#include <c10/cuda/CUDAStream.h>

#include <algorithm>
#include <iostream>

#include <multidevice/ProcessGroupBuilder.hpp>
#include <torch/csrc/distributed/c10d/TCPStore.hpp>

namespace nvfuser {

using namespace torch::jit::fuser::cuda;
using namespace at::indexing;

// To run the following tests on several devices, pytorch must be installed
// with the flag USE_DISTRIBUTED=1 and nccl support.
// Then simply run the tests on several processes, for example using mpirun,
// e.g.: mpirun -np 4 ./build/bin/nvfuser_tests
// --gtest_filter=NVFuserTest.FusionMultiGPU_Reduce

TEST_F(NVFuserTest, FusionMultiClusterProcessGroup_CUDA) {
  int grank, gsize;

  if (parseEnv(grank, gsize)) {
    GTEST_SKIP() << "distributed config is not provided";
  }

  c10d::TCPStoreOptions store_opts;
  store_opts.isServer = (grank == 0) ? true : false;
  auto store = c10::make_intrusive<c10d::TCPStore>("localhost", store_opts);

  ProcessGroupBuilder pgBuilder;
  auto pg = pgBuilder.getProcessGroup("nccl", store, grank, gsize);
  pg->barrier();
}

TEST_F(NVFuserTest, SendRecvTest_CUDA) {
  // Using the new interface to build multi-cluster fusion
  MultiClusterFusion fusion;
  int grank, gsize;

  if (parseEnv(grank, gsize)) {
    GTEST_SKIP() << "distributed config is not provided";
  }

  c10d::TCPStoreOptions store_opts;
  store_opts.isServer = (grank == 0) ? true : false;
  auto store = c10::make_intrusive<c10d::TCPStore>("localhost", store_opts);

  ProcessGroupBuilder pgBuilder;
  auto pg = pgBuilder.getProcessGroup("nccl", store, grank, gsize);

  if (gsize < 2) {
    GTEST_SKIP()
        << "this test must be run with at least 2 ranks, however gsize="
        << gsize;
  }

  auto number_of_gpus = at::cuda::getNumGPUs();
  if (number_of_gpus < 2) {
    GTEST_SKIP()
        << "this test must be run with at least 2 GPUs, however there are "
        << number_of_gpus << " GPUs available";
  }

  if (grank == 0) {
    auto options =
        at::TensorOptions().dtype(at::kFloat).device(at::Device("cuda:0"));
    at::Tensor input = at::randn({8}, options);
    std::vector<at::Tensor> tensor_to_send = {input};
    pg->send(tensor_to_send, 1, 0);
    std::cout << "sent tensor:\n" << tensor_to_send[0] << std::endl;
  } else if (grank == 1) {
    auto options =
        at::TensorOptions().dtype(at::kFloat).device(at::Device("cuda:1"));
    std::vector<at::Tensor> tensor_to_receive = {at::empty({8}, options)};
    auto work = pg->recv(tensor_to_receive, 0, 0);
    while (!work->isCompleted())
      ; // wait for completion
    std::cout << "received tensor:\n" << tensor_to_receive[0] << std::endl;
  }
  pg->barrier();
}

TEST_F(NVFuserTest, FusionMultiGPU_CUDA) {
  // ===========================================================
  //        FUSION
  // ===========================================================
  MultiClusterFusion fusion;
  FusionGuard fg(&fusion);

  TensorView* tv0 = makeContigTensor(3);
  fusion.addInput(tv0);

  // Each expression has to belong to some cluster,
  //  and each cluster will become one cuda kernel
  //  after lowering time.

  // Create the first cluster.
  //  The builder now points to the first created cluster,
  // all operations following this line will make changes
  // to the first cluster.
  fusion.newCluster({.process_rank = 0});
  TensorView* tv1 = sum(tv0, {0});
  fusion.addClusterOutput(tv1);

  // Create the second cluster.
  //  The builder now points to the second created cluster,
  // all operations following this line will make changes
  // to the second cluster.
  fusion.newCluster({.process_rank = 1});
  TensorView* tv2 = sum(tv1, {0});
  fusion.addClusterOutput(tv2);

  fusion.addOutput(tv2);

  // to print the MultiClusterFusion and the AggregateDag, use:
  // fusion.toString()
  // fusion.aggregateDag()->toString()

  // ===========================================================
  //        PROCESS GROUP & GPU BINDING
  // ===========================================================

  int grank, gsize;
  if (parseEnv(grank, gsize)) {
    GTEST_SKIP() << "distributed config is not provided";
  }

  c10d::TCPStoreOptions store_opts;
  store_opts.isServer = (grank == 0) ? true : false;
  auto store = c10::make_intrusive<c10d::TCPStore>("localhost", store_opts);

  ProcessGroupBuilder pgBuilder;
  auto pg = pgBuilder.getProcessGroup("nccl", store, grank, gsize);

  if (gsize < 2) {
    GTEST_SKIP()
        << "this test must be run with at least 2 ranks, however gsize="
        << gsize;
  }

  auto number_of_gpus = at::cuda::getNumGPUs();
  if (number_of_gpus < 2) {
    GTEST_SKIP()
        << "this test must be run with at least 2 GPUs, however there are "
        << number_of_gpus << " GPUs available";
  }
  auto device = at::Device("cuda:" + std::to_string(grank));

  // ===========================================================
  //        RUNTIME
  // ===========================================================

  // Build aggregateDag and pass it to a
  //  multi-device runtime.
  MultiDeviceRuntime runtime(fusion.aggregateDag(), pg);

  // Create input tensors. Each rank is binded to a different GPU
  c10::TensorOptions options;
  options = at::TensorOptions().dtype(at::kFloat).device(device);
  at::Tensor input_tv = at::randn(
      {2, 8, 8}, options); // caveat: concrete values only used on rank 0

  auto cg_outputs =
      runtime.runWithInput({input_tv}); // Run the multiple kernels created.

  if (grank == 0) {
    std::vector<at::Tensor> sent_tv = {input_tv};
    pg->send(sent_tv, 1, 0);
  } else if (grank == 1) {
    std::vector<at::Tensor> received_tv = {input_tv};
    auto work = pg->recv(received_tv, 0, 0);
    while (!work->isCompleted())
      ;
    auto ref = input_tv.sum({0}).sum({0});
    TORCH_INTERNAL_ASSERT(
        allclose(ref, cg_outputs[0]),
        "Obtained output is not the one expected");
  }

  pg->barrier();
}

TEST_F(NVFuserTest, FusionMultiGPU_Reduce_CUDA) {
  /*
  Test to be run on 4 ranks, each rank will be associated with a unique device
  and a unique cluster.

  Input: tensor tv of shape (2,8,8), initialized randomly on rank 0.

  =========

  rank 0:
    input: tv
    outputs: tv0 = tv + tv
      This operation is just to make the kernel non trivial

  =========

  rank 0 sends tva = tv0[0,:] of shape (8,8) to rank 1
  rank 0 sends tvb = tv0[1,:] of shape (8,8) to rank 2

  =========

  rank 1:
    input: tva
    output: tva1 = tva.sum(0)

  rank 2:
    input: tvb
    output: tvb1 = tvb.sum(0)

  =========

  rank 3 receives tva1 from rank 1
  rank 3 receives tvb1 from rank 2

  =========

  rank 3:
    input: tva1 and tvb1
    output: tv2 = tva1 + tvb1
      this output should match 2 * tv.sum({0,1})
  */

  // ===========================================================
  //        FUSION
  // ===========================================================

  MultiClusterFusion fusion;
  FusionGuard fg(&fusion);

  TensorView* tv = makeContigTensor(3);
  auto index_a = IrBuilder::create<Int>(0);
  auto index_b = IrBuilder::create<Int>(1);
  fusion.addInput(tv);

  // TODO: automate the device management. Bind device to rank and not to
  // cluster..?
  fusion.newCluster({.process_rank = 0});
  auto tv0 = add(tv, tv);
  fusion.addClusterOutput(tv0);

  fusion.newCluster({.process_rank = 1});
  auto tva = select(tv0, 0, index_a); // tva = tv0[0,:,:] of shape (8,8)
  TensorView* tva1 = sum(tva, {0}); // tva1 of shape (r8,8) or (8)
  fusion.addClusterOutput(tva1);

  fusion.newCluster({.process_rank = 2});
  auto tvb = select(tv0, 0, index_b); // tvb = tv0[1,:,:] of shape (8,8)
  TensorView* tvb1 = sum(tvb, {0});
  fusion.addClusterOutput(tvb1);

  fusion.newCluster({.process_rank = 3});
  TensorView* tv2 = add(tva1, tvb1);
  fusion.addClusterOutput(tv2);

  fusion.addOutput(tv2);

  // to print the MultiClusterFusion and the AggregateDag, use the methods:
  // fusion.toString()
  // fusion.aggregateDag()->toString()

  // ===========================================================
  //        PROCESS GROUP & GPU BINDING
  // ===========================================================

  int grank, gsize;
  if (parseEnv(grank, gsize)) {
    GTEST_SKIP() << "distributed config is not provided";
  }

  c10d::TCPStoreOptions store_opts;
  store_opts.isServer = (grank == 0) ? true : false;
  auto store = c10::make_intrusive<c10d::TCPStore>("localhost", store_opts);

  ProcessGroupBuilder pgBuilder;
  auto pg = pgBuilder.getProcessGroup("nccl", store, grank, gsize);

  if (gsize < 4) {
    GTEST_SKIP()
        << "this test must be run with at least 4 ranks, however gsize="
        << gsize;
  }

  auto number_of_gpus = at::cuda::getNumGPUs();
  if (number_of_gpus < 4) {
    GTEST_SKIP()
        << "this test must be run with at least 4 GPUs, however there are "
        << number_of_gpus << " GPUs available";
  }
  auto device = at::Device("cuda:" + std::to_string(grank));

  // ===========================================================
  //        RUNTIME
  // ===========================================================

  // create runtime
  MultiDeviceRuntime runtime(fusion.aggregateDag(), pg);

  // Create input tensors. Each rank is binded to a different GPU
  c10::TensorOptions options;
  options = at::TensorOptions().dtype(at::kFloat).device(device);
  at::Tensor input_tv = at::randn(
      {2, 8, 8}, options); // caveat: concrete values only used on rank 0

  // Run the multiple kernels created.
  auto cg_outputs = runtime.runWithInput({input_tv});

  // check the result
  if (grank == 0) {
    std::vector<at::Tensor> sent_tv = {input_tv};
    pg->send(sent_tv, 3, 0);
  } else if (grank == 3) {
    std::vector<at::Tensor> received_tv = {input_tv};
    auto work = pg->recv(received_tv, 0, 0);
    while (!work->isCompleted())
      ;
    auto ref = input_tv;
    ref = ref + ref;
    ref = ref.sum({0});
    ref = ref.sum({0});
    TORCH_INTERNAL_ASSERT(
        allclose(ref, cg_outputs[0]),
        "Obtained output is not the one expected");
  }
  pg->barrier();
}

} // namespace nvfuser

#endif
