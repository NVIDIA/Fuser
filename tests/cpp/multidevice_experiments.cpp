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

class OverlapTest : public MultiDeviceTest {
  protected:
    void SetUp() override {
        MultiDeviceTest::SetUp();
        // Setup the world communicator. We use one device per rank
        num_devices = communicator->size();
        my_device_index = communicator->deviceId();
        std::vector<int64_t> devices(num_devices);
        std::iota(devices.begin(), devices.end(), 0);
        world_communicator = communicator->getBackendForTeam(
            devices, std::nullopt /* default backend */);

        // Define the constants
        A = num_devices;
        B = getNvFuserEnv("OVERLAP_B")? pow(2,std::atoi(getNvFuserEnv("OVERLAP_C"))): pow(2,4);
        C = getNvFuserEnv("OVERLAP_C")? pow(2,std::atoi(getNvFuserEnv("OVERLAP_C"))): pow(2,16);

        tile_size = getNvFuserEnv("OVERLAP_TILE_SIZE")? std::atoi(getNvFuserEnv("OVERLAP_TILE_SIZE")): 1;
        NVF_ERROR(B % tile_size == 0);
        number_of_tiles = B / tile_size;

        n_iterations = getNvFuserEnv("OVERLAP_N_ITERATIONS")?
                                std::atoi(getNvFuserEnv("OVERLAP_N_ITERATIONS"))
                                : 1;
    }

    // network backend
    int64_t num_devices;
    int64_t my_device_index;
    c10::intrusive_ptr<c10d::Backend> world_communicator;

    // tensors sizes
    int64_t A;
    int64_t B;
    int64_t C;

    // optimization parameters
    int64_t tile_size;
    int64_t number_of_tiles;

    // compute params
    int n_iterations;

    at::Tensor get_slice(at::Tensor t, int64_t i, int64_t j) {
        return t.index({at::indexing::Slice(i, i+1),
                        at::indexing::Slice(j*tile_size, (j+1)*tile_size),
                        "..."});
    }

    at::Tensor compute_ATen (at::Tensor t) {
        for (auto _=0; _<n_iterations; _++) {
            t = t+t;
            t = t*t;
            t = t-t;
        }
        return t;
    }
};

/*
The tensor program we target is the following:
    | input: tv0[A,B,C], sharded accross first dimension (locally [1,B,C])
    | tv1[A,B,C] = pointwise_compute(tv0) (locally [1,B,C])
    | output: tv2[A,B,C] = allgather(tv1) (locally [A,B,C])

We want to compare this baseline program with the following one, which is functionnally identical but where we interleave and pipeline the Comm and Compute kernels on different streams:
    | for (j=0; j<number_of_tiles; j++) {
    |     tv1 = pointwise_compute(tv0[j], stream[j])
    |     tv2 = allgather(tv1[j], stream[j])
    | }
where "[j]" referes to taking a slice onto the "B" dimension.
This program should in principle achieve overlap between comms and compute
*/
TEST_F(OverlapTest, OverlapExperiment) {

    // Input set-up
    // define the unsharded inputs for validation
    std::vector<int64_t> unsharded_sizes = {A, B, C};
    auto options =
        at::TensorOptions().dtype(at::kFloat).device(communicator->device());
    auto tv0_unsharded = at::randn(unsharded_sizes, options);
    // Index into the unsharded inputs to get the local input tv0
    // We prepare the inputs slices, because profiling shows that slicing sometimes result in a copy instead of a view
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
        auto tv1_j = compute_ATen(tv0_slices.at(j));

        // communication
        std::vector<at::Tensor> src_buf = {tv1_j};
        std::vector<std::vector<at::Tensor>>& dst_bufs = tv2_slices.at(j);
        auto req_handle = world_communicator->allgather(dst_bufs, src_buf);
        req_handle->wait();
    }

    // validation
    // compute the expected output
    auto tv2_ref = compute_ATen(tv0_unsharded);
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