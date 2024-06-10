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
    nvFuserFusionExecutor
};

std::ostream& operator<<(std::ostream& out, const ComputeMode& mode) {
  switch (mode) {
    case ComputeMode::Pytorch:
      return out << "ComputeMode::Pytorch";
    case ComputeMode::nvFuserFusionExecutor:
      return out << "ComputeMode::nvFuserFusionExecutor";
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
    CommunicatorBackend backend_type = CommunicatorBackend::nccl;
    int nbr_of_backends = 1;

    // tensors sizes
    int64_t B = std::pow(2,4);
    int64_t C = std::pow(2,5);

    // overlap optimization parameters
    int64_t tile_size = 1;
    bool use_different_streams = false;

    // compute params
    ComputeMode compute_mode = ComputeMode::nvFuserFusionExecutor;

    void parseEnv() {
        if (parseEnvVariable("USE_UCC")) {
            backend_type = CommunicatorBackend::ucc;
        }
        if (parseEnvVariable("B")) {
            B = std::pow(2, parseEnvVariable("B"));
        }
        if (parseEnvVariable("C")) {
            C = std::pow(2, parseEnvVariable("C"));
        }
        if (parseEnvVariable("NOT_INTERLEAVE")) {
            tile_size = B;
        } else if (parseEnvVariable("TILE_SIZE")) {
            tile_size = parseEnvVariable("TILE_SIZE");
        }
        if (parseEnvVariable("USE_STREAMS")) {
            use_different_streams = true;
        }
        if (parseEnvVariable("COMPUTE_PYTORCH")) {
            compute_mode = ComputeMode::Pytorch;
        }
        if (parseEnvVariable("NBR_BACKENDS")) {
            nbr_of_backends = parseEnvVariable("NBR_BACKENDS");
        }
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
        << indent << "compute_mode=" << params.compute_mode << "\n"
        << indent << "nbr_of_backends=" << params.nbr_of_backends << "\n"
        << "}";
    return out;
}

class OverlapTest : public MultiDeviceTest {
  protected:
    OverlapTestParams params;

    int64_t A;
    std::vector<int64_t> unsharded_sizes;
    std::vector<int64_t> sharded_sizes;
    int64_t number_of_tiles;

    int64_t my_device_index;
    at::TensorOptions options;
    at::Tensor t0_unsharded, t0, t1, t2, t2_ref;

    std::unique_ptr<Fusion> fusion;
    std::unique_ptr<FusionExecutor> fe;

    int communicator_running_counter = 0;
    std::vector<c10d::Backend*> world_communicators;

    OverlapTest() {
        MultiDeviceTest::SetUp();

        params.parseEnv();
        if (!communicator->deviceId()) {
            std::cout << params << std::endl;
        }

        options = at::TensorOptions().dtype(at::kFloat).device(communicator->device());

        // Setup the world communicator. We use one device per rank
        int64_t num_devices = communicator->size();
        my_device_index = communicator->deviceId();
        std::vector<int64_t> devices(num_devices);
        std::iota(devices.begin(), devices.end(), 0);
        for (int i=0; i<params.nbr_of_backends; i++) {
            world_communicators.push_back(communicator->getBackendForTeam(
                devices,
                /*backend=*/params.backend_type,
                /*prefix=*/std::to_string((communicator_running_counter))));
        }

        // Define the constants
        A = num_devices;
        NVF_ERROR(params.B % params.tile_size == 0);
        number_of_tiles = params.B / params.tile_size;

        if (params.compute_mode == ComputeMode::nvFuserFusionExecutor) {
            fusion = std::make_unique<Fusion>();
            FusionGuard fg(fusion.get());

            TensorView* tv = makeConcreteTensor({params.tile_size,A,params.C});
            fusion->addInput(tv);
            tv = add(tv,tv);
            tv = mul(tv,tv);
            fusion->addOutput(tv);

            DeviceMesh mesh(devices);
            for (auto tv: ir_utils::filterByType<TensorView>(fusion->vals())) {
                tv->setDeviceMesh(mesh);
                tv->axis(1)->parallelize(ParallelType::DIDx);
            }

            fe = std::make_unique<FusionExecutor>();
        }

        // Input set-up
        // define the unsharded inputs for validation
        unsharded_sizes = {params.B, A, params.C};
        sharded_sizes = {params.B, 1, params.C};
        t0_unsharded = at::randn(unsharded_sizes, options);
        t0 = at::empty(sharded_sizes, options);
        t0.copy_(t0_unsharded.index({at::indexing::Slice(),
                            at::indexing::Slice(my_device_index, my_device_index+1), "..."}));
        t1 = at::empty_like(t0);
        t2 = at::empty_like(t0_unsharded);

        // validation
        // compute the expected output
        t2_ref = at::empty(unsharded_sizes, options);
        compute_ATen(t0_unsharded, t2_ref);
    }

    void TearDown() override {
        // compare obtained and expected outputs
        EXPECT_TRUE(torch::allclose(t2_ref, t2))
            << "On device " << my_device_index << "\n"
            << "obtained=" << t2 << "\n"
            << "expected=" << t2_ref << "\n";
    }

    c10d::Backend* getWorldCommunicator() {
        auto& ret = world_communicators.at(communicator_running_counter);
        communicator_running_counter = (communicator_running_counter+1) % world_communicators.size();
        return ret;
    }

    void compute_ATen (at::Tensor& t, at::Tensor& output) {
        at::add_out(output, t, t);
        at::mul_out(output, output, output);
    }

    void compute(at::Tensor t, at::Tensor output) {
        std::vector<c10::IValue> inputs;
        switch (params.compute_mode)
        {
        case ComputeMode::Pytorch:
            compute_ATen(t, output);
            break;
        case ComputeMode::nvFuserFusionExecutor:
            inputs.push_back(t);
            fe->runFusion(inputs, {output});
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
    | input: t0[A,B,C], sharded accross first dimension (locally [1,B,C])
    | t1[A,B,C] = pointwise_compute(t0) (locally [1,B,C])
    | output: t2[A,B,C] = allgather(t1) (locally [A,B,C])

We want to compare this baseline program with the following one, which is functionnally identical but where we interleave and pipeline the Comm and Compute kernels on different streams:
    | for (j=0; j<number_of_tiles; j++) {
    |     t1 = pointwise_compute(t0[j], stream[j])
    |     t2 = allgather(t1[j], stream[j])
    | }
where "[j]" referes to taking a slice onto the "B" dimension.
This program should in principle achieve overlap between comms and compute
*/
TEST_F(OverlapTest, SimpleComputeComm) {
    if (params.compute_mode == ComputeMode::nvFuserFusionExecutor) {
        c10::IValue t0_ivalue = t0.index({at::indexing::Slice(0, params.tile_size), "..."});
        fe->compileFusion(fusion.get(), t0_ivalue);
    }
    // Iterate over the number of tiles and pipeline the comms and compute
    for (auto j: c10::irange(number_of_tiles)) {
        if (params.use_different_streams) {
            setCurrentCUDAStream(c10::cuda::getStreamFromPool(/* high priority */true, my_device_index));
        }
        // local compute
        compute(t0.index({at::indexing::Slice(j*params.tile_size, (j+1)*params.tile_size), "..."}),
                t1.index({at::indexing::Slice(j*params.tile_size, (j+1)*params.tile_size), "..."})
               );

        // communication
        for (int k=0; k<params.tile_size; k++) {
            int index = j*params.tile_size + k;
            auto dst = t2.index({at::indexing::Slice(index, index+1), "..."});
            auto src = t1.index({at::indexing::Slice(index, index+1), "..."});
            getWorldCommunicator()->_allgather_base(dst, src)->wait();
        }
    }
}

} // namespace nvfuser

#endif