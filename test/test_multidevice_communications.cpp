// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#ifdef USE_DISTRIBUTED
#include <gtest/gtest.h>

#include <multidevice/communication.h>
#include <multidevice/communicator.h>
#include <test/multidevice.h>

#include <C++20/ranges>
#include <iostream>

namespace nvfuser {

static constexpr DeviceIdxType root = 0;
static constexpr int tensor_size = 1024;
static constexpr int number_of_repetitions = 8;

TEST_F(MultiDeviceTest, Communication_Gather) {
  if (!comm.is_available() || comm.size() < 2) {
    GTEST_SKIP() << "This test needs at least 2 ranks";
  }
  c10::TensorOptions options =
      at::TensorOptions().dtype(at::kFloat).device(comm.device());

  CommParams params;
  params.root = root;
  params.team = std::vector<DeviceIdxType>(comm.size());
  std::iota(params.team.begin(), params.team.end(), 0);
  params.src_bufs = {at::empty(tensor_size, options)};
  if (comm.deviceId() == root) {
    for (int i = 0; i < comm.size(); i++) {
      params.dst_bufs.push_back(at::empty(tensor_size, options));
    }
  }
  auto communication = Gather(params);

  for (int j : std::views::iota(0, number_of_repetitions)) {
    params.src_bufs.at(0).copy_(
        at::arange(tensor_size, options) + (comm.deviceId() + 1) * j);
    for (auto& buf : params.dst_bufs) {
      buf.copy_(at::zeros(tensor_size, options));
    }

    auto work = communication.post(comm);
    work->wait();

    if (comm.deviceId() == root) {
      for (int i : std::views::iota((int64_t)0, comm.size())) {
        auto obtained = params.dst_bufs.at(i);
        auto ref = at::arange(tensor_size, options) + (i + 1) * j;
        NVF_ERROR(
            at::equal(obtained, ref),
            "Device ",
            comm.deviceId(),
            " expected tensor:\n",
            ref,
            "\nbut obtained tensor:\n",
            obtained);
      }
    }
  }
  comm.barrier();
}

TEST_F(MultiDeviceTest, Communication_Allgather) {
  if (!comm.is_available() || comm.size() < 2) {
    GTEST_SKIP() << "This test needs at least 2 ranks";
  }
  c10::TensorOptions options =
      at::TensorOptions().dtype(at::kFloat).device(comm.device());

  CommParams params;
  params.team = std::vector<DeviceIdxType>(comm.size());
  std::iota(params.team.begin(), params.team.end(), 0);
  params.src_bufs = {at::empty(tensor_size, options) * comm.deviceId()};
  for (int i = 0; i < comm.size(); i++) {
    params.dst_bufs.push_back(at::empty(tensor_size, options));
  }
  auto communication = Allgather(params);

  for (int j : std::views::iota(0, number_of_repetitions)) {
    params.src_bufs.at(0).copy_(
        at::arange(tensor_size, options) + (comm.deviceId() + 1) * j);
    for (auto& buf : params.dst_bufs) {
      buf.copy_(at::zeros(tensor_size, options));
    }

    auto work = communication.post(comm);
    work->wait();

    for (int i : std::views::iota((int64_t)0, comm.size())) {
      auto obtained = params.dst_bufs.at(i);
      auto ref = at::arange(tensor_size, options) + (i + 1) * j;
      NVF_ERROR(
          obtained.equal(ref),
          "Device",
          comm.deviceId(),
          " expected tensor:\n",
          ref,
          "\nbut obtained tensor:\n",
          obtained);
    }
  }
  comm.barrier();
}

TEST_F(MultiDeviceTest, Communication_Scatter) {
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
      params.src_bufs.push_back(at::empty(tensor_size, options) * i);
    }
  }
  params.dst_bufs = {at::empty(tensor_size, options)};
  auto communication = Scatter(params);

  for (int j : std::views::iota(0, number_of_repetitions)) {
    params.dst_bufs.at(0).copy_(at::zeros(tensor_size, options));
    for (int i : std::views::iota((size_t)0, params.src_bufs.size())) {
      params.src_bufs.at(i).copy_(
          at::arange(tensor_size, options) + (i + 1) * j);
    }

    auto work = communication.post(comm);
    work->wait();

    auto obtained = params.dst_bufs.at(0);
    auto ref = at::arange(tensor_size, options) + (comm.deviceId() + 1) * j;
    NVF_ERROR(
        obtained.equal(ref),
        "Device",
        comm.deviceId(),
        " expected tensor:\n",
        ref,
        "\nbut obtained tensor:\n",
        obtained);
  }
  comm.barrier();
}

TEST_F(MultiDeviceTest, Communication_Broadcast) {
  if (!comm.is_available()) {
    GTEST_SKIP() << "This test needs distributed setting";
  }
  c10::TensorOptions options =
      at::TensorOptions().dtype(at::kFloat).device(comm.device());

  CommParams params;
  params.root = root;
  params.team = std::vector<DeviceIdxType>(comm.size());
  std::iota(params.team.begin(), params.team.end(), 0);
  if (comm.deviceId() == root) {
    params.src_bufs = {at::empty(tensor_size, options)};
  }
  params.dst_bufs = {at::empty(tensor_size, options)};

  auto communication = Broadcast(params);

  for (int j : std::views::iota(0, number_of_repetitions)) {
    if (comm.deviceId() == root) {
      params.src_bufs.at(0).copy_(at::arange(tensor_size, options) + j);
    }
    params.dst_bufs.at(0).copy_(at::zeros(tensor_size, options));

    auto work = communication.post(comm);
    if (comm.size() > 1) {
      work->wait();
    }

    auto obtained = params.dst_bufs.at(0);
    auto ref = at::arange(tensor_size, options) + j;
    NVF_ERROR(
        obtained.equal(ref),
        "Device",
        comm.deviceId(),
        " expected tensor:\n",
        ref,
        "\nbut obtained tensor:\n",
        obtained);
  }
  comm.barrier();
}

TEST_F(MultiDeviceTest, Communication_SendRecv) {
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
    params.src_bufs.push_back(at::empty(tensor_size, options));
  } else {
    params.dst_bufs.push_back(at::empty(tensor_size, options));
  }
  auto communication = SendRecv(params);

  for (int j : std::views::iota(0, number_of_repetitions)) {
    if (comm.deviceId() == sender) {
      params.src_bufs.at(0).copy_(at::arange(tensor_size, options) + j);
    } else {
      params.dst_bufs.at(0).copy_(at::zeros(tensor_size, options));
    }

    auto work = communication.post(comm);
    work->wait();

    if (comm.deviceId() == receiver) {
      auto obtained = params.dst_bufs.at(0);
      auto ref = at::arange(tensor_size, options) + j;
      NVF_ERROR(
          obtained.equal(ref),
          "Device",
          comm.deviceId(),
          " expected tensor:\n",
          ref,
          "\nbut obtained tensor:\n",
          obtained);
    }
  }
  comm.barrier();
}

TEST_F(MultiDeviceTest, Communication_SendRecvToSelf) {
  DeviceIdxType sender = 0;
  if (!comm.is_available()) {
    GTEST_SKIP() << "This test needs distributed setting";
  }
  if (comm.deviceId() > 0) { // only device 0 participates
    comm.barrier();
    return;
  }
  c10::TensorOptions options =
      at::TensorOptions().dtype(at::kFloat).device(comm.device());

  CommParams params;
  params.root = sender;
  params.team = {0};
  params.src_bufs.push_back(at::empty(tensor_size, options));
  params.dst_bufs.push_back(at::empty(tensor_size, options));
  auto communication = SendRecv(params);

  for (int j : std::views::iota(0, number_of_repetitions)) {
    params.src_bufs.at(0).copy_(at::arange(tensor_size, options) + j);
    params.dst_bufs.at(0).copy_(at::zeros(tensor_size, options));

    communication.post(comm);

    auto obtained = params.dst_bufs.at(0);
    auto ref = at::arange(tensor_size, options) + j;
    NVF_ERROR(
        obtained.equal(ref),
        "Device",
        comm.deviceId(),
        " expected tensor:\n",
        ref,
        "\nbut obtained tensor:\n",
        obtained);
  }
  comm.barrier();
}

} // namespace nvfuser

#endif
