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
    CommunicatorBackend backend_type;
    int nbr_of_backends;

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
        nbr_of_backends = parseEnvVariable("NBR_BACKENDS")? parseEnvVariable("NBR_BACKENDS") : 1;
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
        << indent << "nbr_of_backends=" << params.nbr_of_backends << "\n"
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

        options = at::TensorOptions().dtype(at::kFloat).device(communicator->device());

        // Setup the world communicator. We use one device per rank
        num_devices = communicator->size();
        my_device_index = communicator->deviceId();
        std::vector<int64_t> devices(num_devices);
        std::iota(devices.begin(), devices.end(), 0);
        for (int i=0; i<params.nbr_of_backends; i++) {
            world_communicators.push_back(communicator->getBackendForTeam(
                devices, /* backend */params.backend_type, /*use cache*/false));
        }

        if (params.wait_at_backend_creation) {
            auto x = at::empty({1}, options);
            std::vector<at::Tensor> X = {x};
            for (auto& backend: world_communicators) {
                auto work_request = backend->allreduce(X);
                work_request->wait();
                while (!work_request->isCompleted()) {}
            }
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
            for (auto _=0; _<params.n_iterations; _++) {
                tv = add(tv,tv);
                tv = mul(tv,tv);
                tv = sub(tv,tv);
            }
            fusion->addOutput(tv);

            DeviceMesh mesh(devices);
            for (auto tv: ir_utils::filterByType<TensorView>(fusion->vals())) {
                tv->setDeviceMesh(mesh);
                tv->axis(1)->parallelize(ParallelType::DIDx);
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
    std::vector<c10::intrusive_ptr<c10d::Backend>> world_communicators;
    int communicator_running_counter = 0;
    c10::intrusive_ptr<c10d::Backend> getWorldCommunicator() {
        auto& ret = world_communicators.at(communicator_running_counter);
        communicator_running_counter = (communicator_running_counter+1) % world_communicators.size();
        return ret;
    }

    int64_t A;
    int64_t number_of_tiles;
    std::unique_ptr<Fusion> fusion;
    std::unique_ptr<FusionExecutor> fe;
    at::TensorOptions options;

    c10::IValue get_slice(c10::IValue t, int64_t i, int64_t j) {
        return t.toTensor().index({
                        at::indexing::Slice(j*params.tile_size, (j+1)*params.tile_size),
                        at::indexing::Slice(i, i+1),
                        "..."});
    }

    void compute_ATen (at::Tensor& t, at::Tensor& output) {
        for (auto _=0; _<params.n_iterations; _++) {
            at::add_out(output, t, t);
            at::mul_out(output, output, output);
            at::sub_out(output, output, output);
        }
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
    std::vector<int64_t> unsharded_sizes = {params.B, A, params.C};
    std::vector<int64_t> sharded_sizes = {params.B, 1, params.C};
    auto tv0_unsharded = at::randn(unsharded_sizes, options);
    at::Tensor tv0 = at::empty(sharded_sizes, options);
    tv0.copy_(tv0_unsharded.index({at::indexing::Slice(),
                        at::indexing::Slice(my_device_index, my_device_index+1), "..."}));
    auto tv1 = at::empty_like(tv0);
    auto tv2 = at::empty_like(tv0_unsharded);

    if (params.compute_mode == ComputeMode::nvFuserFusionExecutor) {
        c10::IValue tv0_ivalue = tv0.index({at::indexing::Slice(0, params.tile_size), "..."});
        fe->compileFusion(fusion.get(), tv0_ivalue);
    }
    // Iterate over the number of tiles and pipeline the comms and compute
    for (auto j: c10::irange(number_of_tiles)) {
        if (params.use_different_streams) {
            setCurrentCUDAStream(c10::cuda::getStreamFromPool(/* high priority */true, my_device_index));
        }
        // local compute
        compute(tv0.index({at::indexing::Slice(j*params.tile_size, (j+1)*params.tile_size), "..."}),
                tv1.index({at::indexing::Slice(j*params.tile_size, (j+1)*params.tile_size), "..."})
               );

        // communication
        for (int k=0; k<params.tile_size; k++) {
            int index = j*params.tile_size + k;
            auto dst = tv2.index({at::indexing::Slice(index, index+1), "..."});
            auto src = tv1.index({at::indexing::Slice(index, index+1), "..."});
            getWorldCommunicator()->_allgather_base(dst, src);
        }
    }

    // validation
    // compute the expected output
    auto tv2_ref = at::empty(unsharded_sizes, options);
    compute_ATen(tv0_unsharded, tv2_ref);
    // compare obtained and expected outputs
    EXPECT_TRUE(torch::allclose(tv2_ref, tv2))
        << "On device " << my_device_index << "\n"
        << "obtained=" << tv2 << "\n"
        << "expected=" << tv2_ref << "\n";
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
            getWorldCommunicator()->allgather(dst_buffers, output);
        }
    }
}

} // namespace nvfuser

#endif