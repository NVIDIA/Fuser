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
        return -1;
    }
    return std::atoi(env);
}

bool isEnvVariableDefined(const char* env_name) {
    return parseEnvVariable(env_name) != -1;
}

struct OverlapTestParams {
    // network backend type
    CommunicatorBackend backend_type = CommunicatorBackend::nccl;
    // number of different process group instances to create, to potentially achieve comm/comm overlap
    int nbr_of_backends = 1;

    // tensors sizes. Tensor's unsharded size is [A,B,C], where B=number of devices
    int64_t M = std::pow(2,6);
    int64_t K = std::pow(2,5);
    int64_t N = std::pow(2,4);
    int64_t S = std::pow(2,3);

    // overlap optimization parameters
    bool use_different_streams = false; // whether to change CUDA stream at each iteration

    // compute params
    ComputeMode compute_mode = ComputeMode::nvFuserFusionExecutor;

    void parseEnv() {
        if (isEnvVariableDefined("USE_UCC")) {
            backend_type = CommunicatorBackend::ucc;
        }
        if (isEnvVariableDefined("LOG2_M")) {
            M = std::pow(2, parseEnvVariable("LOG2_M"));
        }
        if (isEnvVariableDefined("LOG2_K")) {
            K = std::pow(2, parseEnvVariable("LOG2_K"));
        }
        if (isEnvVariableDefined("LOG2_N")) {
            N = std::pow(2, parseEnvVariable("LOG2_N"));
        }
        if (isEnvVariableDefined("LOG2_S")) {
            S = std::pow(2, parseEnvVariable("LOG2_S"));
        }
        if (isEnvVariableDefined("USE_STREAMS")) {
            use_different_streams = true;
        }
        if (isEnvVariableDefined("COMPUTE_PYTORCH")) {
            compute_mode = ComputeMode::Pytorch;
        }
        if (isEnvVariableDefined("NBR_BACKENDS")) {
            nbr_of_backends = parseEnvVariable("NBR_BACKENDS");
        }
    }
};

std::ostream& operator<<(std::ostream& out, const OverlapTestParams& params) {
    std::string indent = "  ";
    out << "params:{\n"
        << indent << "backend_type=" << params.backend_type << "\n"
        << indent << "M=" << params.M << "\n"
        << indent << "K=" << params.K << "\n"
        << indent << "N=" << params.N << "\n"
        << indent << "S=" << params.S << "\n"
        << indent << "use_different_streams=" << params.use_different_streams << "\n"
        << indent << "compute_mode=" << params.compute_mode << "\n"
        << indent << "nbr_of_backends=" << params.nbr_of_backends << "\n"
        << "}";
    return out;
}

class OverlapTest : public MultiDeviceTest {
  protected:
    OverlapTestParams params;

    int64_t num_devices;
    std::vector<int64_t> unsharded_sizes;
    std::vector<int64_t> sharded_sizes;

    int64_t my_device_index;
    at::TensorOptions options;
    at::Tensor ta, tb, tc_unreduced, tc_locally_reduced, tc, tc_ref;

    std::unique_ptr<Fusion> fusion;
    std::unique_ptr<FusionExecutor> fe;

    int communicator_running_counter = 0;
    std::vector<c10d::Backend*> world_communicators;

    void SetUp() {
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
        for (int i=0; i<params.nbr_of_backends; i++) {
            world_communicators.push_back(communicator->getBackendForTeam(
                devices,
                /*backend=*/params.backend_type,
                /*prefix=*/std::to_string((communicator_running_counter))));
        }

        ASSERT_EQ(params.M % (params.S * num_devices), 0);
        ASSERT_EQ(params.K % num_devices, 0);

        // Input set-up and buffer allocation. Define the full unsharded inputs for validation
        std::vector<int64_t> ta_unsharded_sizes         = {params.S, num_devices, params.M/params.S, params.K/num_devices, 1       };
        std::vector<int64_t> ta_sizes                   = {params.S, 1          , params.M/params.S, params.K/num_devices, 1       };
        std::vector<int64_t> tb_unsharded_sizes         = {1       , num_devices, 1                , params.K/num_devices, params.N};
        std::vector<int64_t> tb_sizes                   = {1       , 1          , 1                , params.K/num_devices, params.N};
        std::vector<int64_t> tc_unreduced_sizes         = {params.S, 1          , params.M/params.S, params.K/num_devices, params.N};
        std::vector<int64_t> tc_partially_reduced_sizes = {params.S, 1          , params.M/params.S, 1                   , params.N};
        std::vector<int64_t> tc_sizes                   = {params.S, 1          , params.M/(params.S*num_devices)        , params.N};

        options = at::TensorOptions().dtype(at::kFloat).device(communicator->device());
        auto ta_unsharded = at::randn(ta_unsharded_sizes, options);
        auto tb_unsharded = at::randn(tb_unsharded_sizes, options);
        ta = at::empty(ta_sizes, options);
        ta.copy_(ta_unsharded.index({at::indexing::Slice(),
                            at::indexing::Slice(my_device_index, my_device_index+1), "..."}));
        tb = at::empty(tb_sizes, options);
        tb.copy_(tb_unsharded.index({at::indexing::Slice(),
                            at::indexing::Slice(my_device_index, my_device_index+1), "..."}));

        tc_unreduced = at::empty(tc_unreduced_sizes, options);
        tc_locally_reduced = at::empty(tc_partially_reduced_sizes, options);
        tc = at::empty(tc_sizes, options);

        // compute the expected output for validation
        // auto ta_unsharded_b = at::broadcast_to(ta_unsharded, {params.S, num_devices, params.M/params.S, params.K/num_devices, 1});
        // auto tb_unsharded_b = at::broadcast_to(tb_unsharded, {1       , num_devices, 1                , params.K/num_devices, params.N});
        auto tc_unsharded_unreduced = ta_unsharded * tb_unsharded;
        auto tc_unsharded_ref = at::sum(tc_unsharded_unreduced, {1,3});
        auto tc_unsharded_ref_reshaped = at::reshape(tc_unsharded_ref, {params.S, num_devices, params.M/(params.S*num_devices), params.N});
        tc_ref = tc_unsharded_ref_reshaped.index({at::indexing::Slice(),
                            at::indexing::Slice(my_device_index, my_device_index+1), "..."});


        if (!communicator->deviceId()) {
            std::cout << "ta.sizes()=" << ta.sizes() << std::endl;
            std::cout << "tb.sizes()=" << tb.sizes() << std::endl;
            std::cout << "tc_unreduced.sizes()=" << tc_unreduced.sizes() << std::endl;
            std::cout << "tc_locally_reduced.sizes()=" << tc_locally_reduced.sizes() << std::endl;
            std::cout << "tc.sizes()=" << tc.sizes() << std::endl;
            std::cout << "tc_ref.sizes()=" << tc_ref.sizes() << std::endl;
            std::cout << "tc_unsharded_unreduced.sizes()=" << tc_unsharded_unreduced.sizes() << std::endl;
            std::cout << "tc_unsharded_ref.sizes()=" << tc_unsharded_ref.sizes() << std::endl;
            std::cout << "tc_unsharded_ref_reshaped.sizes()=" << tc_unsharded_ref_reshaped.sizes() << std::endl;
        }


        // computeATen(t0_unsharded, t2_ref);


        // if (params.compute_mode == ComputeMode::nvFuserFusionExecutor) {
        //     fusion = std::make_unique<Fusion>();
        //     FusionGuard fg(fusion.get());

        //     TensorView* tv = makeConcreteTensor({params.tile_size,B,params.C});
        //     fusion->addInput(tv);
        //     tv = add(tv,tv);
        //     tv = mul(tv,tv);
        //     fusion->addOutput(tv);

        //     DeviceMesh mesh(devices);
        //     for (auto tv: ir_utils::filterByType<TensorView>(fusion->vals())) {
        //         tv->setDeviceMesh(mesh);
        //         tv->axis(1)->parallelize(ParallelType::DIDx);
        //     }

        //     fe = std::make_unique<FusionExecutor>();
        // }
    }

    // void TearDown() override {
    //     // compare obtained and expected outputs
    //     EXPECT_TRUE(torch::allclose(t2_ref, t2))
    //         << "On device " << my_device_index << "\n"
    //         << "obtained=" << t2 << "\n"
    //         << "expected=" << t2_ref << "\n";
    // }

    c10d::Backend* getWorldCommunicator() {
        auto& ret = world_communicators.at(communicator_running_counter);
        communicator_running_counter = (communicator_running_counter+1) % world_communicators.size();
        return ret;
    }

    at::Tensor getSlice(at::Tensor t, int64_t j) {
        return t.index({at::indexing::Slice(j, j+1), "..."});
    }

    void computeATen(at::Tensor ta_j, at::Tensor tb_j, at::Tensor tc_unreduced_j, at::Tensor tc_partially_reduced_j) {
        at::mul_out(tc_unreduced_j, ta_j, tb_j);
        at::sum_out(tc_partially_reduced_j, tc_unreduced_j, {3});
        // auto [S , D ,  M_div_S , K_div_D , Nb] = ta.sizes();
        // auto [Sb, D2,  M_div_Sb, K_div_D2, N ] = tb.sizes();
        // NVF_ERROR(D==D2);
        // NVF_ERROR(K_div_D==K_div_D2);
        // NVF_ERROR(M_div_Sb==1);
        // NVF_ERROR(Sb==1);
        // NVF_ERROR(Nb==1);
        // auto ta_b = at::broadcast_to(ta, {params.S, 1, params.M/params.S, params.K/num_devices, 1});
        // auto tb_b = at::broadcast_to(tb, {1       , 1, 1                , params.K/num_devices, params.N});
        // auto tc_unreduced = ta_b * ta_b;
        // auto tc_locally_reduced = at::sum(tc_unreduced, {3});
        // if (!communicator->deviceId()) {
        //     std::cout << tc_locally_reduced << std::endl;
        // }
        // return tc_locally_reduced;
    }

//     void compute(at::Tensor t, at::Tensor output) {
//         std::vector<c10::IValue> inputs;
//         switch (params.compute_mode)
//         {
//         case ComputeMode::Pytorch:
//             computeATen(t, output);
//             break;
//         // case ComputeMode::nvFuserFusionExecutor:
//         //     inputs.push_back(t);
//         //     fe->runFusion(inputs, {output});
//         //     break;
//         default:
//             NVF_ERROR(false);
//             break;
//         }
//         return;
//     }
};

/*
The tensor program that we target is the following, assuming a setup with `num_devices` devices:
    inputs: 
       - A[M,K] sharded column-wise:
         dimension K is split by the factor `num_devices`
         so A is viewed as [M, num_devices, K/num_devices]
         and the allocation size of A is [M, 1, K/num_devices]
       - B[K,N] sharded row-wise:
         locally of size [1, K/num_devices, N]
    output: 
       - C[M,N]=matmul(A,B), sharded on dimension M:
         dimension M is split by the factor `num_devices`
         so C is viewed as [num_devices, M/num_devices,N]
         and the allocation size of M is [1, M/num_devices,N]
Up to some broadcast and view ops, a straightforward program to generate the output could be summarized as
    | C_unreduced = pointwise_multiply(A,B) (with unsharded size [M,num_devices,K/num_devices,N], sharded on `num_devices`)
    | C_locally_reduce = local_reduction(C_unreduced, axis=`K/num_devices`, op=sum) (with unsharded size [M,num_devices,N], sharded on `num_devices`)
    | C = reduce_scatter(C_unreduced, op=sum) (with unsharded size [num_devices, M/num_devices, N] sharded on `num_devices`)
We want to compare this baseline program with one that is functionnally identical but achieves more overlap between computations and communications.
Our goal is to interlave the comms and compute using a technic called "reduce-scatter based pipelining"
To do so, we further split the row dimension M with a factor `S` representing the number of tiles, and we apply the operations successively on tensors slices accross S, changing stream at each iteration.
Assuming the following shapes:
    - A [S, num_devices, M/S, K/num_devices], sharded on num_devices
    - B [num_devices, K/num_devices, N], sharded on num_devices
    - C [S, num_devices, M/(S*num_devices), N], sharded on num_devices
the program implementing collective-based pipelining could be summarized as:
    | for (j=0; j<S; j++) {
    |   setCurrentStream(Stream[j])
    |   C_unreduced[j] = pointwise_multiply(A[j],B)
    |   C_locally_reduce[j] = local_reduction(C_unreduced[j], axis=`K/num_devices`, op=sum)
    |   C[j]=reduce_scatter(C_locally_reduce[j], op=sum)
    | }
where "[j]" referes to taking a slice onto the `S` dimension.
This program achieves overlap between comms and compute
Remarks:
    1) it is convenient to have "S" as being the outermost dimension so C_locally_reduce[j] is a contiguous buffer.
    2) The layout needs to match the reduce-scatter semantics, i.e., the first dimension is reduced and the second is scattered. This is why we choose the layouts to be [S, sharded_axis, M, ...]
*/
TEST_F(OverlapTest, GEMM_RS_without_overlap) {
    if (params.S != 1) {
        GTEST_SKIP() << "must set S to 1";
    }

    tc_unreduced = ta * tb; // {params.S, 1 , params.M/params.S, params.K/num_devices, params.N}
    auto tc_locally_reduced = at::sum(tc_unreduced, {3}); // {params.S, 1 , params.M/params.S, params.N}
    if (!communicator->deviceId()) {
        std::cout << "ta.sizes()=" << ta.sizes() << std::endl;
        std::cout << "tb.sizes()=" << tb.sizes() << std::endl;
        std::cout << "tc_unreduced.sizes()=" << tc_unreduced.sizes() << std::endl;
        std::cout << "tc_locally_reduced.sizes()=" << tc_locally_reduced.sizes() << std::endl;
        std::cout << "tc.sizes()=" << tc.sizes() << std::endl;
        std::cout << "tc_ref.sizes()=" << tc_ref.sizes() << std::endl;
    }
    getWorldCommunicator()->_reduce_scatter_base(tc, tc_locally_reduced)->wait();
    ASSERT_TRUE(tc.allclose(tc_ref, 1e-3, 1e-3)) <<"Unexpected results: obtained:"<< tc << "\n expected:" << tc_ref;
}

TEST_F(OverlapTest, SimpleComputeComm) {
    // if (params.compute_mode == ComputeMode::nvFuserFusionExecutor) {
    //     c10::IValue t0_ivalue = t0.index({at::indexing::Slice(0, params.tile_size), "..."});
    //     fe->compileFusion(fusion.get(), t0_ivalue);
    // }
    // Iterate over the number of tiles and pipeline the comms and compute
    for (auto j: c10::irange(params.S)) {
        auto ta_j                   = getSlice(ta, j);
        auto tc_unreduced_j         = getSlice(tc_unreduced, j);
        auto tc_partially_reduced_j = getSlice(tc_locally_reduced, j);
        auto tc_j                   = getSlice(tc, j);

        if (params.use_different_streams) {
            setCurrentCUDAStream(c10::cuda::getStreamFromPool(/* high priority */true, my_device_index));
        }
        // local compute
        computeATen(ta_j, tb, tc_unreduced_j, tc_partially_reduced_j);

        // communication
        getWorldCommunicator()->_reduce_scatter_base(tc_j, tc_partially_reduced_j)->wait();
    }
    ASSERT_TRUE(tc.allclose(tc_ref, 1e-3, 1e-3)) << "Unexpected results, obtained:"<< tc << "\n expected: " << tc_ref;
}

} // namespace nvfuser

#endif