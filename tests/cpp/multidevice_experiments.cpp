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


class GlobalContext : public testing::Environment {
  public:
    void SetUp() override {}
    void TearDown() override {}
};

auto global_context = static_cast<GlobalContext*>(
    testing::AddGlobalTestEnvironment(new GlobalContext));

/*
The tensor program we target is the following:
    | input: tv0[A,B,C], sharded accross first dimension (locally [1,B,C])
    | tv1[A,B,C] = tv0 * 2 (locally [1,B,C])
    | output: tv2[A,B,C] = allgather(tv1) (locally [A,B,C])

From the host perspective, this program could be schematized as:
    | tv1 = Compute(tv0)
    | tv2 = Comm(tv1)

We want to compare this baseline program with the one where we interleave and pipeline the Comm and Compute kernels:
    | for (j=0; j<number_of_tiles; j++) {
    |     tv1 = Compute(tv0[j])
    |     tv2 = Comm(tv1[j])
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
    const int64_t B = getNvFuserEnv("OVERLAP_B")? pow(2,std::atoi(getNvFuserEnv("OVERLAP_C"))): pow(2,4);
    const int64_t C = getNvFuserEnv("OVERLAP_C")? pow(2,std::atoi(getNvFuserEnv("OVERLAP_C"))): pow(2,16);
    std::vector<int64_t> unsharded_sizes{A, B, C};

    const int64_t tile_size = getNvFuserEnv("OVERLAP_TILE_SIZE")? std::atoi(getNvFuserEnv("OVERLAP_TILE_SIZE")): 1;
    NVF_ERROR(B % tile_size == 0);
    const int64_t number_of_tiles = B / tile_size;

    auto compute = [](at::Tensor t) {
            const int n_iterations = getNvFuserEnv("OVERLAP_N_ITERATIONS")?
                                     std::atoi(getNvFuserEnv("OVERLAP_N_ITERATIONS"))
                                     : 1;
            for (auto _=0; _<n_iterations; _++) {
                t = t+t;
                t = t*t;
                t = t-t;
            }
            return t;
        };

    auto get_slice = [tile_size](at::Tensor t, int64_t i, int64_t j)
            {return t.index({at::indexing::Slice(i, i+1),
                             at::indexing::Slice(j*tile_size, (j+1)*tile_size),
                             "..."});};

    // Input set-up
    // define the unsharded inputs for validation
    auto options =
        at::TensorOptions().dtype(at::kFloat).device(communicator->device());
    auto tv0_unsharded = at::randn(unsharded_sizes, options);
    // Index into the unsharded inputs to get the local input tv0
    // We prepare the inputs slices, because profiling shows that slicing sometimes result in a copy instead of a view
    const auto my_device_index = communicator->deviceId();
    std::vector<at::Tensor> tv0_slices;
    for (auto j: c10::irange(number_of_tiles)) {
        tv0_slices.push_back(get_slice(tv0_unsharded, my_device_index, j));
    }

    // Allocate ouput global buffer for tv2 for the data to be allgathered
    auto tv2_buf = at::empty(unsharded_sizes, options);
    // Setup the recv buffer slices. c10d needs the destinations buffers to be in a certain format
    std::vector<std::vector<std::vector<at::Tensor>>> tv2_slices;
    for (auto j: c10::irange(number_of_tiles)) {
        std::vector<at::Tensor> tv2_j_slices;
        for (auto i: c10::irange(num_devices)) {
            tv2_j_slices.push_back(get_slice(tv2_buf, i, j));
        }
        tv2_slices.push_back({std::move(tv2_j_slices)});
    }

    // Iterate over the number of tiles and pipeline the comms and compute
    for (auto j: c10::irange(number_of_tiles)) {
        setCurrentCUDAStream(c10::cuda::getStreamFromPool(/* high priority */false, my_device_index));
        // local compute
        auto tv1_j = compute(tv0_slices.at(j));

        // communication
        std::vector<at::Tensor> src_buf = {tv1_j};
        std::vector<std::vector<at::Tensor>>& dst_bufs = tv2_slices.at(j);
        auto req_handle = world_comm->allgather(dst_bufs, src_buf);
        req_handle->wait();
    }

    // validation
    // compute the expected output
    auto tv2_ref = compute(tv0_unsharded);
    // compare obtained and expected outputs
    for (auto i: c10::irange(num_devices)) {
        for (auto j: c10::irange(number_of_tiles)) {
            auto expected = get_slice(tv2_ref, i, j);
            auto obtained = tv2_slices.at(j).at(0).at(i);
            EXPECT_TRUE(torch::allclose(expected, obtained))
                << "On device " << my_device_index << "\n"
                << "i=" << i << "\n"
                << "j=" << j << "\n"
                << "obtained=" << obtained << "\n"
                << "expected=" << expected << "\n";
        }
    }
}

} // namespace nvfuser

#endif