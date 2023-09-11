// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#ifdef USE_DISTRIBUTED
#include <gtest/gtest.h>

#include <multidevice/collective.h>
#include <multidevice/communicator.h>
#include <test/utils.h>

#include <iostream>

namespace nvfuser {

static constexpr DeviceIdxType root = 0;
static constexpr int tensor_size = 1024;
static constexpr int number_of_repetitions = 8;

TEST_F(NVFuserTest, FusionMultiGPU_Collective_Gather_CUDA) {
  if (!comm.is_available() || comm.size() < 2) {
    GTEST_SKIP() << "This test needs at least 2 ranks";
  }
  c10::TensorOptions options =
      at::TensorOptions().dtype(at::kFloat).device(comm.device());

  CommParams params;
  params.root = root;
  params.team = std::vector<DeviceIdxType>(comm.size());
  std::iota(params.team.begin(), params.team.end(), 0);
  params.src_bufs = {at::ones(tensor_size, options) * comm.deviceId()};
  if (comm.deviceId() == root) {
    for (int i = 0; i < comm.size(); i++) {
      params.dst_bufs.push_back(-at::ones(tensor_size, options));
    }
  }
  for (int i=0; i< number_of_repetitions; i++) {
    for (auto& buf : params.dst_bufs) {
      buf *= 0;
    }

    auto coll = Gather(params);
    auto work = coll.post(comm);
    work->wait();

    if (comm.deviceId() == root) {
      for (int i : c10::irange(comm.size())) {
        auto obtained = params.dst_bufs.at(i);
        auto ref = at::ones(tensor_size, options) * i;
        TORCH_INTERNAL_ASSERT(
            obtained.equal(ref),
            "Device",
            comm.deviceId(),
            "expected tensor ",
            ref,
            "\nbut obtained tensor: ",
            obtained);
      }
    }
    comm.barrier();
  }
}

TEST_F(NVFuserTest, FusionMultiGPU_Collective_Allgather_CUDA) {
  if (!comm.is_available() || comm.size() < 2) {
    GTEST_SKIP() << "This test needs at least 2 ranks";
  }
  c10::TensorOptions options =
      at::TensorOptions().dtype(at::kFloat).device(comm.device());

  CommParams params;
  params.team = std::vector<DeviceIdxType>(comm.size());
  std::iota(params.team.begin(), params.team.end(), 0);
  params.src_bufs = {at::ones(tensor_size, options) * comm.deviceId()};
  for (int i = 0; i < comm.size(); i++) {
    params.dst_bufs.push_back(-at::ones(tensor_size, options));
  }

  for (int i=0; i< number_of_repetitions; i++) {
    for (auto& buf : params.dst_bufs) {
      buf *= 0;
    }

    auto coll = Allgather(params);
    auto work = coll.post(comm);
    work->wait();

    for (int i : c10::irange(comm.size())) {
      auto obtained = params.dst_bufs.at(i);
      auto ref = at::ones(tensor_size, options) * i;
      TORCH_INTERNAL_ASSERT(
          obtained.equal(ref),
          "Device",
          comm.deviceId(),
          " expected tensor: ",
          ref,
          "\n but obtained tensor: ",
          obtained);
    }
    comm.barrier();
  }
}

TEST_F(NVFuserTest, FusionMultiGPU_Collective_Scatter_CUDA) {
  if (!comm.is_available() || comm.size() < 2) {
    GTEST_SKIP() << "This test needs at least 2 ranks";
  }
  c10::TensorOptions options =
      at::TensorOptions().dtype(at::kFloat).device(comm.device());

  CommParams params;
  params.root = root;
  params.team = std::vector<DeviceIdxType>(comm.size());
  std::iota(params.team.begin(), params.team.end(), 0);
  if (comm.deviceId() == root) {
    for (int i = 0; i < comm.size(); i++) {
      params.src_bufs.push_back(at::ones(tensor_size, options) * i);
    }
  }
  params.dst_bufs = {at::zeros(tensor_size, options)};

  for (int i=0; i< number_of_repetitions; i++) {
    for (auto& buf : params.dst_bufs) {
      buf *= 0;
    }

    auto coll = Scatter(params);
    auto work = coll.post(comm);
    work->wait();

    auto obtained = params.dst_bufs.at(0);
    auto ref = at::ones(tensor_size, options) * comm.deviceId();
    TORCH_INTERNAL_ASSERT(
        obtained.equal(ref),
        "Device",
        comm.deviceId(),
        " expected tensor: ",
        ref,
        "\n but obtained tensor: ",
        obtained);

    comm.barrier();
  }
}

TEST_F(NVFuserTest, FusionMultiGPU_Collective_Broadcast_CUDA) {
  if (!comm.is_available() || comm.size() < 2) {
    GTEST_SKIP() << "This test needs at least 2 ranks";
  }
  c10::TensorOptions options =
      at::TensorOptions().dtype(at::kFloat).device(comm.device());

  CommParams params;
  params.root = root;
  params.team = std::vector<DeviceIdxType>(comm.size());
  std::iota(params.team.begin(), params.team.end(), 0);
  if (comm.deviceId() == root) {
    params.src_bufs = {at::ones(tensor_size, options)};
  } else {
    params.dst_bufs = {at::zeros(tensor_size, options)};
  }

  for (int i=0; i< number_of_repetitions; i++) {
    for (auto& buf : params.dst_bufs) {
      buf *= 0;
    }

    auto coll = Broadcast(params);
    auto work = coll.post(comm);
    work->wait();

    if (comm.deviceId() != root) {
      auto obtained = params.dst_bufs.at(0);
      auto ref = at::ones(tensor_size, options);
      TORCH_INTERNAL_ASSERT(
          obtained.equal(ref),
          "Device",
          comm.deviceId(),
          " expected tensor: ",
          ref,
          "\n but obtained tensor: ",
          obtained);
    }
    comm.barrier();
  }
}

TEST_F(NVFuserTest, FusionMultiGPU_Collective_SendRecv_CUDA) {
  DeviceIdxType sender = 0;
  DeviceIdxType receiver = 1;
  if (!comm.is_available() || comm.size() < 2) {
    GTEST_SKIP() << "This test needs at least 2 ranks";
  }
  if (comm.deviceId() > 1) { // only devices 0 and 1 participate
    comm.barrier();
    return;
  }
  c10::TensorOptions options =
      at::TensorOptions().dtype(at::kFloat).device(comm.device());

  CommParams params;
  params.root = sender;
  params.team = {0, 1};
  if (comm.deviceId() == sender) {
    params.src_bufs.push_back(at::ones(tensor_size, options));
  } else {
    params.dst_bufs.push_back(at::zeros(tensor_size, options));
  }

  for (int i=0; i< number_of_repetitions; i++) {
    for (auto& buf : params.dst_bufs) {
      buf *= 0;
    }

    auto coll = SendRecv(params);
    auto work = coll.post(comm);
    work->wait();

    if (comm.deviceId() == receiver) {
      auto obtained = params.dst_bufs.at(0);
      auto ref = at::ones(tensor_size, options);
      TORCH_INTERNAL_ASSERT(
          obtained.equal(ref),
          "Device",
          comm.deviceId(),
          " expected tensor: ",
          ref,
          "\n but obtained tensor: ",
          obtained);
    }
    comm.getBackendForTeam({0,1})->barrier();
    // comm.getBackendForTeam({0,1})->barrier();
  }
  comm.barrier();
}

} // namespace nvfuser

#endif
