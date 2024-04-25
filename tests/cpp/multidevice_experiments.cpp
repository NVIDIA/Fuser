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

int parseEnvVariable(const char* env_name) {
    const std::string prefix = "NVFUSER_OVERLAP_";
    auto prefixed_name = prefix + env_name;
    auto env = std::getenv(prefixed_name.c_str());
    if (!env) {
        return 0;
    }
    return std::atoi(env);
}

struct OverlapTestParams {
    // network backend
    CommunicatorBackend backend_type;

    // tensors sizes
    int64_t B;
    int64_t C;

    // optimization parameters
    int64_t tile_size;
    bool use_different_streams;
    bool wait_at_backend_creation;
    bool do_interleave;

    // compute params
    int n_iterations;
    ComputeMode compute_mode;

    void parseEnv() {
        backend_type = parseEnvVariable("USE_UCC")? CommunicatorBackend::ucc : CommunicatorBackend::nccl;
        B = parseEnvVariable("B")? std::pow(2, parseEnvVariable("B")): std::pow(2,4);
        C = parseEnvVariable("C")? std::pow(2, parseEnvVariable("C")): std::pow(2,5);
        do_interleave = !parseEnvVariable("NOT_INTERLEAVE");
        tile_size = (!do_interleave)? B
                    : (parseEnvVariable("TILE_SIZE")?
                        parseEnvVariable("TILE_SIZE")
                        : 1);

        use_different_streams = parseEnvVariable("USE_STREAMS");
        wait_at_backend_creation = parseEnvVariable("WAIT_BACKEND_CREATION");
        n_iterations = parseEnvVariable("N_ITERATIONS")?
                                parseEnvVariable("N_ITERATIONS")
                                : 1;
        compute_mode = parseEnvVariable("COMPUTE_PYTORCH")? ComputeMode::Pytorch : ComputeMode::nvFuserFusionExecutor;
    }
};

std::ostream& operator<<(std::ostream& out, const OverlapTestParams& params) {
    std::string indent = "  ";
    out << "params:{\n"
        << indent << "backend_type=" << params.backend_type << "\n"
        << indent << "B=" << params.B << "\n"
        << indent << "C=" << params.C << "\n"
        << indent << "tile_size=" << params.tile_size << "\n"
        << indent << "use_different_streams=" << params.use_different_streams << "\n"
        << indent << "wait_at_backend_creation=" << params.wait_at_backend_creation << "\n"
        << indent << "do_interleave=" << params.do_interleave << "\n"
        << indent << "n_iterations=" << params.n_iterations << "\n"
        << indent << "compute_mode=" << params.compute_mode << "\n"
        << "}";
    return out;
}

class OverlapTest : public MultiDeviceTest {
  protected:
    OverlapTestParams params;
    void SetUp() override {
        MultiDeviceTest::SetUp();

        params.parseEnv();
        if (!communicator->deviceId()) {
            std::cout << params << std::endl;
        }

        // Setup the world communicator. We use one device per rank
        num_devices = communicator->size();
        my_device_index = communicator->deviceId();
        std::vector<int64_t> devices(num_devices);
        std::iota(devices.begin(), devices.end(), 0);
        world_communicator = communicator->getBackendForTeam(
            devices, /* backend */params.backend_type, /*use cache*/true, /*wait*/ params.wait_at_backend_creation);

        // Define the constants
        A = num_devices;
        unsharded_sizes = {A, params.B, params.C};

        NVF_ERROR(params.B % params.tile_size == 0);
        number_of_tiles = params.B / params.tile_size;

        if (params.compute_mode == ComputeMode::nvFuserFusionExecutor) {
            fusion = std::make_unique<Fusion>();
            FusionGuard fg(fusion.get());

            TensorView* tv = makeConcreteTensor({A,params.tile_size,params.C});
            fusion->addInput(tv);
            for (auto _=0; _<params.n_iterations; _++) {
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
        if (params.use_different_streams) {
            // call getStreamFromPool to trigger the lazy init
            c10::cuda::getStreamFromPool(/* high priority */true, my_device_index);
        }
    }
    int64_t num_devices;
    int64_t my_device_index;
    c10::intrusive_ptr<c10d::Backend> world_communicator;

    int64_t A;
    int64_t number_of_tiles;
    std::vector<int64_t> unsharded_sizes;
    std::unique_ptr<Fusion> fusion;
    std::unique_ptr<FusionExecutor> fe;

    c10::IValue get_slice(c10::IValue t, int64_t i, int64_t j) {
        return t.toTensor().index({at::indexing::Slice(i, i+1),
                        at::indexing::Slice(j*params.tile_size, (j+1)*params.tile_size),
                        "..."});
    }

    void compute_ATen (at::Tensor& t, at::Tensor& output) {
        for (auto _=0; _<params.n_iterations; _++) {
            at::add_out(output, t, t);
            at::mul_out(output, output, output);
            at::sub_out(output, output, output);
        }
    }

    void compute(std::vector<c10::IValue>& t, std::vector<at::Tensor>& output) {
        switch (params.compute_mode)
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
        tv1_slices.push_back({at::empty({1,params.tile_size,params.C}, options)});
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

    if (params.compute_mode == ComputeMode::nvFuserFusionExecutor) {
        fe->compileFusion(fusion.get(), tv0_slices.at(0));
    }
    // Iterate over the number of tiles and pipeline the comms and compute
    for (auto j: c10::irange(number_of_tiles)) {
        if (params.use_different_streams) {
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
    std::vector<c10::IValue> input = {at::randn({1,params.tile_size,params.C}, options)};
    std::vector<at::Tensor> output = {at::empty({1,params.tile_size,params.C}, options)};
    std::vector<at::Tensor> tmp_dst_buffers;
    for (int i=0; i<num_devices; i++) {
        tmp_dst_buffers.push_back({at::empty({1,params.tile_size,params.C}, options)});
    }
    std::vector<std::vector<at::Tensor>> dst_buffers = {std::move(tmp_dst_buffers)};

    if (!parseEnvVariable("NO_COMPUTE")) {
        fe->compileFusion(fusion.get(), input);
    }

    for (int i=0; i<16; i++) {
        if (params.use_different_streams) {
            setCurrentCUDAStream(c10::cuda::getStreamFromPool(/* high priority */true, my_device_index));
        }

        if (!parseEnvVariable("NO_COMPUTE") && (i>0 || !parseEnvVariable("COMM_FIRST"))) {
            fe->runFusion(input, output);
        }

        if (!parseEnvVariable("NO_COMM")) {
            world_communicator->allgather(dst_buffers, output);
        }
    }
}

} // namespace nvfuser

#endif