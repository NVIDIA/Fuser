// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#ifdef NVFUSER_DISTRIBUTED
#include <tests/cpp/multidevice.h>
#include <ops/all_ops.h>
#include <ir/utils.h>
#include <fusion.h>
#include <c10/util/ArrayRef.h>
#include <ATen/Functions.h>

// #include <c10/core/DeviceType.h>
// #include <ATen/ATen.h>
// #include <ATen/Operators.h>

namespace nvfuser {


class GlobalContext : public testing::Environment {
  public:
    void SetUp() override {
        if (getNvFuserEnv("OVERLAP_USE_STREAMS")) {
            // call getStreamFromPool to trigger the lazy init
            c10::cuda::getStreamFromPool(/* high priority */true);
        }
    }
    void TearDown() override {}
};

auto global_context = static_cast<GlobalContext*>(
    testing::AddGlobalTestEnvironment(new GlobalContext));

enum class ComputeMode {
    Pytorch,
    nvFuserFusionExecutor,
    nvFuserFusionExecutorCache
};

std::ostream& operator<<(std::ostream& out, const ComputeMode& mode) {
  switch (mode) {
    case ComputeMode::Pytorch:
      return out << "ComputeMode::Pytorch";
    case ComputeMode::nvFuserFusionExecutor:
      return out << "ComputeMode::nvFuserFusionExecutor";
    case ComputeMode::nvFuserFusionExecutorCache:
      return out << "ComputeMode::nvFuserFusionExecutorCache";
    default:
      NVF_ERROR(false);
  }
  return out;
}

class OverlapTest : public MultiDeviceTest {
  protected:
    void SetUp() override {
        MultiDeviceTest::SetUp();
        // Setup the world communicator. We use one device per rank
        num_devices = communicator->size();
        my_device_index = communicator->deviceId();
        std::vector<int64_t> devices(num_devices);
        std::iota(devices.begin(), devices.end(), 0);
        CommunicatorBackend b = getNvFuserEnv("OVERLAP_USE_UCC")? CommunicatorBackend::ucc : CommunicatorBackend::nccl;
        std::cout << "Used Backend: " << b <<std::endl;
        world_communicator = communicator->getBackendForTeam(
            devices, /* backend */b);

        // Define the constants
        A = num_devices;
        B = getNvFuserEnv("OVERLAP_B")? std::pow(2,std::atoi(getNvFuserEnv("OVERLAP_B"))): std::pow(2,4);
        C = getNvFuserEnv("OVERLAP_C")? std::pow(2,std::atoi(getNvFuserEnv("OVERLAP_C"))): std::pow(2,16);
        unsharded_sizes = {A, B, C};

        tile_size = getNvFuserEnv("OVERLAP_NOT_INTERLEAVE")? B
                    : getNvFuserEnv("OVERLAP_TILE_SIZE")?
                        std::atoi(getNvFuserEnv("OVERLAP_TILE_SIZE"))
                        : 1;
        NVF_ERROR(B % tile_size == 0);
        number_of_tiles = B / tile_size;


        use_different_streams = getNvFuserEnv("OVERLAP_USE_STREAMS")?
                                true
                                : false;
        n_iterations = getNvFuserEnv("OVERLAP_N_ITERATIONS")?
                                std::atoi(getNvFuserEnv("OVERLAP_N_ITERATIONS"))
                                : 1;
        compute_mode = getNvFuserEnv("OVERLAP_COMPUTE_PYTORCH")? ComputeMode::Pytorch : ComputeMode::nvFuserFusionExecutor;
        if (compute_mode == ComputeMode::nvFuserFusionExecutor) {
            fusion = std::make_unique<Fusion>();
            FusionGuard fg(fusion.get());

            TensorView* tv = makeConcreteTensor({A,tile_size,C});
            fusion->addInput(tv);
            for (auto _=0; _<n_iterations; _++) {
                tv = add(tv, tv);
                tv = mul(tv,tv);
                tv = sub(tv,tv);
            }
            fusion->addOutput(tv);

            DeviceMesh mesh(devices);
            for (auto tv: ir_utils::filterByType<TensorView>(fusion->vals())) {
                tv->setDeviceMesh(mesh);
                tv->axis(0)->parallelize(ParallelType::DIDx);
            }

            fe = std::make_unique<FusionExecutor>();
        }
    }

    // network backend
    int64_t num_devices;
    int64_t my_device_index;
    c10::intrusive_ptr<c10d::Backend> world_communicator;

    // tensors sizes
    int64_t A;
    int64_t B;
    int64_t C;
    std::vector<int64_t> unsharded_sizes;

    // optimization parameters
    int64_t tile_size;
    int64_t number_of_tiles;
    bool use_different_streams;

    // compute params
    int n_iterations;
    ComputeMode compute_mode;

    std::unique_ptr<Fusion> fusion;
    std::unique_ptr<FusionExecutor> fe;

    c10::IValue get_slice(c10::IValue t, int64_t i, int64_t j) {
        return t.toTensor().index({at::indexing::Slice(i, i+1),
                        at::indexing::Slice(j*tile_size, (j+1)*tile_size),
                        "..."});
    }

    void compute_ATen (at::Tensor& t, at::Tensor& output) {
        for (auto _=0; _<n_iterations; _++) {
            at::add_out(output, t, t);
            at::mul_out(output, output, output);
            at::sub_out(output, output, output);
        }
    }

    void compute(std::vector<c10::IValue>& t, std::vector<at::Tensor>& output) {
        switch (compute_mode)
        {
        case ComputeMode::Pytorch:
            compute_ATen(t.at(0).toTensor(), output.at(0));
            break;
        case ComputeMode::nvFuserFusionExecutor:
            fe->runFusion(t, output);
            break;
        default:
            NVF_ERROR(false);
            break;
        }
        return;
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
TEST_F(OverlapTest, SimpleComputeComm) {
    // Input set-up
    // define the unsharded inputs for validation
    auto options =
        at::TensorOptions().dtype(at::kFloat).device(communicator->device());
    auto tv0_unsharded = at::randn(unsharded_sizes, options);
    // Index into the unsharded inputs to get the local input tv0
    // We prepare the inputs slices, because profiling shows that slicing sometimes result in a copy instead of a view
    std::vector<std::vector<c10::IValue>> tv0_slices;
    for (auto j: c10::irange(number_of_tiles)) {
        tv0_slices.push_back({get_slice(tv0_unsharded, my_device_index, j)});
    }

    std::vector<std::vector<at::Tensor>> tv1_slices;
    for (int _=0; _<number_of_tiles; _++) {
        tv1_slices.push_back({at::empty({1,tile_size,C}, options)});
    }

    // Allocate ouput global buffer for tv2 for the data to be allgathered
    auto tv2_buf = at::empty(unsharded_sizes, options);
    // Setup the recv buffer slices. c10d needs the destinations buffers to be in a certain format
    std::vector<std::vector<std::vector<at::Tensor>>> tv2_slices;
    for (auto j: c10::irange(number_of_tiles)) {
        std::vector<at::Tensor> tv2_j_slices;
        for (auto i: c10::irange(num_devices)) {
            tv2_j_slices.push_back(get_slice(tv2_buf, i, j).toTensor());
        }
        tv2_slices.push_back({std::move(tv2_j_slices)});
    }

    if (compute_mode == ComputeMode::nvFuserFusionExecutor) {
        fe->compileFusion(fusion.get(), tv0_slices.at(0));
    }
    // Iterate over the number of tiles and pipeline the comms and compute
    for (auto j: c10::irange(number_of_tiles)) {
        if (use_different_streams) {
            setCurrentCUDAStream(c10::cuda::getStreamFromPool(/* high priority */true, my_device_index));
        }
        // local compute
        compute(tv0_slices.at(j), tv1_slices.at(j));

        // communication
        auto req_handle = world_communicator->allgather(tv2_slices.at(j), tv1_slices.at(j));
        // req_handle->wait();
    }

    // validation
    // compute the expected output
    auto tv2_ref = at::empty(unsharded_sizes, options);
    compute_ATen(tv0_unsharded, tv2_ref);
    // compare obtained and expected outputs
    for (auto i: c10::irange(num_devices)) {
        for (auto j: c10::irange(number_of_tiles)) {
            auto expected = get_slice(tv2_ref, i, j).toTensor();
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

TEST_F(OverlapTest, DummyExample) {
    // Input set-up
    // define the unsharded inputs for validation
    auto options =
        at::TensorOptions().dtype(at::kFloat).device(communicator->device());
    std::vector<c10::IValue> input = {at::randn({1,tile_size,C}, options)};
    std::vector<at::Tensor> output = {at::empty({1,tile_size,C}, options)};
    std::vector<at::Tensor> tmp_dst_buffers;
    for (int i=0; i<num_devices; i++) {
        tmp_dst_buffers.push_back({at::empty({1,tile_size,C}, options)});
    }
    std::vector<std::vector<at::Tensor>> dst_buffers = {std::move(tmp_dst_buffers)};

    if (!getNvFuserEnv("OVERLAP_NO_COMPUTE")) {
        fe->compileFusion(fusion.get(), input);
    }

    for (int i=0; i<16; i++) {
        if (use_different_streams) {
            setCurrentCUDAStream(c10::cuda::getStreamFromPool(/* high priority */true, my_device_index));
        }

        if (!getNvFuserEnv("OVERLAP_NO_COMPUTE") && (i>0 || !getNvFuserEnv("OVERLAP_COMM_FIRST"))) {
            fe->runFusion(input, output);
        }

        if (!getNvFuserEnv("OVERLAP_NO_COMM")) {
            world_communicator->allgather(dst_buffers, output);
        }
    }
}

} // namespace nvfuser

#endif