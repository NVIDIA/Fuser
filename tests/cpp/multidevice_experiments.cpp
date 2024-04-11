// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#ifdef NVFUSER_DISTRIBUTED
#include <tests/cpp/multidevice.h>
#include <c10/util/ArrayRef.h>
#include <ATen/Functions.h>

// #include <c10/core/DeviceType.h>
// #include <ATen/ATen.h>
// #include <ATen/Operators.h>

namespace nvfuser {

TEST_F(MultiDeviceTest, OverlapExperiment) {

  const int64_t num_devices = communicator->size();
  std::vector<int64_t> devices(num_devices);
  std::iota(devices.begin(), devices.end(), 0);
  auto world_comm = communicator->getBackendForTeam(
      devices, std::nullopt /* default backend */);

  // tv0[A,B,C], sharded accross first dimension (locally [1,B,C])
  // tv1[A,B,C], sharded accross first dimension (locally [1,B,C])
  // tv2[A,B,C] = tv0 + tv1
  // tv3[A,B,C] = allgather(tv2) unsharded (locally [1,B,C])
  const int64_t A = num_devices;
  const int64_t B = 2;
  const int64_t C = 4;
  std::vector<int64_t> unsharded_sizes{A, B, C};

  // init and compute of reference tensors for validation
  auto options =
      at::TensorOptions().dtype(at::kFloat).device(communicator->device());
  auto tv0_ref = at::randn(unsharded_sizes, options);
  auto tv1_ref = at::randn(unsharded_sizes, options);
  auto tv2_ref = tv0_ref + tv1_ref;
  auto tv3_ref = tv2_ref;

  const auto my_device_index = communicator->deviceId();
  auto get_slice = [] (at::Tensor t, int64_t i)
    {return t.index({at::indexing::Slice(i, i+1), "..."});};
  // inputs tv0 and tv1 are just slices of the unsharded inputs
  auto tv0 = get_slice(tv0_ref, my_device_index);
  auto tv1 = get_slice(tv1_ref, my_device_index);
  // we allocate tv3 buffer because the data will be allgathered
  auto tv3 = at::empty(unsharded_sizes, options);
  std::vector<at::Tensor> tv3_all_slices;
  for (int64_t i = 0; i < A; i++) {
    tv3_all_slices.push_back(get_slice(tv3_ref, i));
  }
  // c10d needs the destinations buffers to be in the following format
  std::vector<std::vector<at::Tensor>> dst_bufs = {std::move(tv3_all_slices)};

  // local compute
  auto tv2 = tv0 + tv1;

  // communication
  std::vector<at::Tensor> src_buf = {tv2};
  auto req_handle = world_comm->allgather(dst_bufs, src_buf);
  req_handle->wait();

  for (auto i: c10::irange(dst_bufs.at(0).size())) {
    EXPECT_TRUE(torch::allclose(get_slice(tv3_ref, i), dst_bufs.at(0).at(i))) << "On device " << my_device_index
                                                << "\niteration=" << i
                                                << "\nget_slice(tv3_ref, i)=" << get_slice(tv3_ref, i)
                                                << "\ntv0=" << tv0
                                                << "\ntv1=" << tv1
                                                << "\ntv2=" << tv2
                                                << "\ntv3(slice i)=" << dst_bufs.at(0).at(i);
  }
}

} // namespace nvfuser

#endif