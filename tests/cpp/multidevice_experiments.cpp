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

/*
The tensor program we target is the following:
    | tv0[A,B,C], sharded accross first dimension (locally [1,B,C])
    | tv1[A,B,C], sharded accross first dimension (locally [1,B,C])
    | tv2[A,B,C] = tv0 + tv1 (locally [1,B,C])
    | tv3[A,B,C] = allgather(tv2) (locally [A,B,C])

From the host perspective, this program could be schematized as:
    | tv2 = Compute(tv0,tv1)
    | tv3 = Comm(tv2)

We want to compare this baseline program with the one where we interleave and pipeline the Comm and Compute kernels:
    | for (j=0; j<number_of_tiles; j++) {
    |     tv2 = Compute(tv0[j],tv1[j])
    |     tv3 = Comm(tv2[j])
    | }
where "[j]" referes to taking a slice onto the "B" dimension
*/
TEST_F(MultiDeviceTest, OverlapExperiment) {
    // Setup the world communicator. We use one device per rank
    const int64_t num_devices = communicator->size();
    std::vector<int64_t> devices(num_devices);
    std::iota(devices.begin(), devices.end(), 0);
    auto world_comm = communicator->getBackendForTeam(
        devices, std::nullopt /* default backend */);

    // Define the constants
    const int64_t A = num_devices;
    const int64_t B = 4;
    const int64_t C = 8;
    std::vector<int64_t> unsharded_sizes{A, B, C};

    const int64_t number_of_tiles = B;
    NVF_ERROR(B % number_of_tiles == 0);
    const int64_t tile_size = B / number_of_tiles;

    // Input set-up
    // define the unsharded inputs for validation
    auto options =
        at::TensorOptions().dtype(at::kFloat).device(communicator->device());
    auto tv0_unsharded = at::randn(unsharded_sizes, options);
    auto tv1_unsharded = at::randn(unsharded_sizes, options);
    // Index into the unsharded inputs to get the local inputs tv0 and tv1
    const auto my_device_index = communicator->deviceId();
    auto tv0 = tv0_unsharded.index({at::indexing::Slice(my_device_index, my_device_index+1), "..."});
    auto tv1 = tv1_unsharded.index({at::indexing::Slice(my_device_index, my_device_index+1), "..."});

    // Allocate ouput global buffer for tv3 for the data to be allgathered
    auto tv3_buf = at::empty(unsharded_sizes, options);
    // Setup the buffers. c10d needs the destinations buffers to be in a certain format
    std::vector<std::vector<std::vector<at::Tensor>>> dst_bufs_storage;
    for (int64_t j=0; j<number_of_tiles; j++) {
        std::vector<at::Tensor> tv3_slices;
        for (int64_t i = 0; i < A; i++) {
            tv3_slices.push_back(tv3_buf.index({at::indexing::Slice(i, i+1), at::indexing::Slice(j*tile_size, (j+1)*tile_size), "..."}));
        }
        dst_bufs_storage.push_back({std::move(tv3_slices)});
    }

    // Iterate over the number of tiles and pipeline the comms and compute
    for (int64_t j=0; j<number_of_tiles; j++) {
        setCurrentCUDAStream(c10::cuda::getStreamFromPool(/* high priority */false, my_device_index));
        // local compute
        auto tv2 = tv0.index({at::indexing::Slice(), at::indexing::Slice(j, j+1), "..."}) + tv1.index({at::indexing::Slice(), at::indexing::Slice(j, j+1), "..."});

        // communication
        std::vector<at::Tensor> src_buf = {tv2};
        std::vector<std::vector<at::Tensor>>& dst_bufs = dst_bufs_storage.at(j);
        auto req_handle = world_comm->allgather(dst_bufs, src_buf);
        req_handle->wait();
        dst_bufs_storage.push_back(dst_bufs);
    }

    // validation
    // compute the expected output
    auto tv2_ref = tv0_unsharded + tv1_unsharded;
    auto tv3_ref = tv2_ref;
    // compare obtained and expected outputs
    for (int i=0; i<num_devices; i++) {
        for (int j=0; j<number_of_tiles; j++) {
            auto expected = tv3_ref.index({at::indexing::Slice(i, i+1), at::indexing::Slice(j, j+1), "..."});
            auto obtained = dst_bufs_storage.at(j).at(0).at(i);
            EXPECT_TRUE(torch::allclose(expected, obtained))
                << "On device " << my_device_index << "\n"
                << "i=" << i << "\n"
                << "j=" << j << "\n"
                << "obtained=" << obtained << "\n"
                << "expected=" << expected << "\n"
                << "tv0=" << tv0 << "\n"
                << "tv1=" << tv1;
        }
    }
}

} // namespace nvfuser

#endif