// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#ifdef NVFUSER_DISTRIBUTED
#include <ATen/Functions.h>
#include <c10/util/ArrayRef.h>
#include <fusion.h>
#include <ir/utils.h>
#include <ops/all_ops.h>
#include <tests/cpp/multidevice.h>

namespace nvfuser {

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
  // Tensors sizes
  int64_t M = std::pow(2, 6);
  int64_t K = std::pow(2, 5);
  int64_t N = std::pow(2, 4);
  int64_t S = std::pow(2, 3);

  // network backend type
  CommunicatorBackend backend_type = CommunicatorBackend::nccl;
  // number of different process group instances to create, to potentially
  // achieve comm/comm overlap
  int nbr_of_backends = 1;

  // overlap optimization parameters
  bool use_different_streams =
      false; // whether to change CUDA stream at each iteration

  // debug
  bool debug_print = false;

  void parseEnv() {
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
    if (isEnvVariableDefined("USE_UCC")) {
      backend_type = CommunicatorBackend::ucc;
    }
    if (isEnvVariableDefined("NBR_BACKENDS")) {
      nbr_of_backends = parseEnvVariable("NBR_BACKENDS");
    }
    if (isEnvVariableDefined("USE_STREAMS")) {
      use_different_streams = true;
    }
    if (isEnvVariableDefined("DEBUG_PRINT")) {
      debug_print = true;
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
      << indent << "use_different_streams=" << params.use_different_streams
      << "\n"
      << indent << "nbr_of_backends=" << params.nbr_of_backends << "\n"
      << "}";
  return out;
}

class OverlapTest : public MultiDeviceTest {
 protected:
  OverlapTestParams params;

  int64_t num_devices;
  int64_t my_device_index;
  at::Tensor ta, tb, tc_unreduced, tc_locally_reduced, tc, tc_expected;

  // stores the backends
  std::vector<c10d::Backend*> world_communicators;
  // index to round-robin iterate in world_communicators
  int communicator_running_counter = 0;

  void SetUp() {
    MultiDeviceTest::SetUp();

    params.parseEnv();
    num_devices = communicator->size();
    my_device_index = communicator->deviceId();
    ASSERT_EQ(params.M % (params.S * num_devices), 0);
    ASSERT_EQ(params.K % num_devices, 0);

    // Setup the world communicators
    std::vector<int64_t> devices(num_devices);
    std::iota(devices.begin(), devices.end(), 0);
    for (int i = 0; i < params.nbr_of_backends; i++) {
      world_communicators.push_back(communicator->getBackendForTeam(
          devices,
          /*backend=*/params.backend_type,
          /*prefix=*/std::to_string((communicator_running_counter))));
    }

    // Define I/O and intermediate Tensor shapes
    // clang-format off
        std::vector<int64_t> ta_unsharded_sizes         = {params.S, num_devices, params.M/params.S, params.K/num_devices, 1       };
        std::vector<int64_t> ta_sizes                   = {params.S, 1          , params.M/params.S, params.K/num_devices, 1       };
        std::vector<int64_t> tb_unsharded_sizes         = {1       , num_devices, 1                , params.K/num_devices, params.N};
        std::vector<int64_t> tb_sizes                   = {1       , 1          , 1                , params.K/num_devices, params.N};
        std::vector<int64_t> tc_unreduced_sizes         = {params.S, 1          , params.M/params.S, params.K/num_devices, params.N};
        std::vector<int64_t> tc_partially_reduced_sizes = {params.S, 1          , params.M/params.S,                       params.N};
        std::vector<int64_t> tc_sizes                   = {params.S, 1          , params.M/(params.S*num_devices)        , params.N};
    // clang-format on

    // Set up input tensors. We create the full unsharded tensors and define the
    // actual input as the shard corresponding to the current device. Having the
    // full unsharded input on each rank makes it possible to compute the
    // expected result locally, hence, this way of doing is convenient for
    // validating data correctness.
    at::TensorOptions options =
        at::TensorOptions().dtype(at::kFloat).device(communicator->device());
    auto ta_unsharded = at::randn(ta_unsharded_sizes, options);
    auto tb_unsharded = at::randn(tb_unsharded_sizes, options);
    ta = at::empty(ta_sizes, options);
    ta.copy_(ta_unsharded.index(
        {at::indexing::Slice(),
         at::indexing::Slice(my_device_index, my_device_index + 1),
         "..."}));
    tb = at::empty(tb_sizes, options);
    tb.copy_(tb_unsharded.index(
        {at::indexing::Slice(),
         at::indexing::Slice(my_device_index, my_device_index + 1),
         "..."}));

    // We pre-allocate the output and some intermediate buffers so we do not
    // rely on torch allocator, which do not behave well with multi-stream
    // programming.
    tc_unreduced = at::empty(tc_unreduced_sizes, options);
    tc_locally_reduced = at::empty(tc_partially_reduced_sizes, options);
    tc = at::empty(tc_sizes, options);

    // compute the expected output for data correctness validation
    auto tc_unsharded_unreduced = ta_unsharded * tb_unsharded;
    auto tc_unsharded_expected = at::sum(tc_unsharded_unreduced, {1, 3});
    auto tc_unsharded_expected_reshaped = at::reshape(
        tc_unsharded_expected,
        {params.S, num_devices, params.M / (params.S * num_devices), params.N});
    tc_expected = tc_unsharded_expected_reshaped.index(
        {at::indexing::Slice(),
         at::indexing::Slice(my_device_index, my_device_index + 1),
         "..."});

    // Debug print
    if (!communicator->deviceId() && params.debug_print) {
      std::cout << params << std::endl;
      std::cout << "ta.sizes()=" << ta.sizes() << std::endl;
      std::cout << "tb.sizes()=" << tb.sizes() << std::endl;
      std::cout << "tc_unreduced.sizes()=" << tc_unreduced.sizes() << std::endl;
      std::cout << "tc_locally_reduced.sizes()=" << tc_locally_reduced.sizes()
                << std::endl;
      std::cout << "tc.sizes()=" << tc.sizes() << std::endl;
    }
  }

  c10d::Backend* getWorldCommunicator() {
    return world_communicators.at(
        (communicator_running_counter++) % world_communicators.size());
  }

  at::Tensor getSlice(at::Tensor t, int64_t j) {
    return t.index({at::indexing::Slice(j, j + 1), "..."});
  }

  void computeATen(
      at::Tensor ta_j,
      at::Tensor tb_j,
      at::Tensor tc_unreduced_j,
      at::Tensor tc_partially_reduced_j) {
    at::mul_out(tc_unreduced_j, ta_j, tb_j);
    at::sum_out(tc_partially_reduced_j, tc_unreduced_j, {3});
  }
};

// This test implements a reduce-scattered based pipelining overlapping technic,
// as used in NEMO-megatron transformer, precisely at the second layer of the
// MLP consisting of a GEMM+Reduce-scatter The tensor program that we target is
// the following, assuming a setup with `num_devices` devices:
//     inputs:
//        - A[M,K] sharded column-wise:
//          dimension K is split by the factor `num_devices`
//          so A is viewed as [M, num_devices, K/num_devices]
//          and the allocation size of A is [M, 1, K/num_devices]
//        - B[K,N] sharded row-wise:
//          locally of size [1, K/num_devices, N]
//     output:
//        - C[M,N]=matmul(A,B), sharded on dimension M:
//          dimension M is split by the factor `num_devices`
//          so C is viewed as [num_devices, M/num_devices,N]
//          and the allocation size of M is [1, M/num_devices,N]
// Up to some broadcast and view ops, a straightforward program to generate the
// output could be summarized as
//     | C_unreduced = pointwise_multiply(A,B) (with unsharded size
//     [M,num_devices,K/num_devices,N], sharded on `num_devices`) |
//     C_locally_reduce = local_reduction(C_unreduced, axis=`K/num_devices`,
//     op=sum) (with unsharded size [M,num_devices,N], sharded on `num_devices`)
//     | C = reduce_scatter(C_unreduced, op=sum) (with unsharded size
//     [num_devices, M/num_devices, N] sharded on `num_devices`)
// We want to compare this baseline program with one that is functionnally
// identical but achieves more overlap between computations and communications.
// Our goal is to interlave the comms and compute using a technic called
// "reduce-scatter based pipelining" To do so, we further split the row
// dimension M with a factor `S` representing the number of tiles, and we apply
// the operations successively on tensors slices accross S, changing stream at
// each iteration. Assuming the following shapes:
//     - A [S, num_devices, M/S, K/num_devices], sharded on num_devices
//     - B [num_devices, K/num_devices, N], sharded on num_devices
//     - C [S, num_devices, M/(S*num_devices), N], sharded on num_devices
// the program implementing collective-based pipelining could be summarized as:
//     | for (j=0; j<S; j++) {
//     |   setCurrentStream(Stream[j])
//     |   C_unreduced[j] = pointwise_multiply(A[j],B)
//     |   C_locally_reduce[j] = local_reduction(C_unreduced[j],
//     axis=`K/num_devices`, op=sum) | C[j]=reduce_scatter(C_locally_reduce[j],
//     op=sum) | }
// where "[j]" referes to taking a slice onto the `S` dimension.
// This program achieves overlap between comms and compute
// Remarks:
//     1) it is convenient to have "S" as being the outermost dimension so
//     C_locally_reduce[j] is a contiguous buffer. 2) The layout needs to match
//     the reduce-scatter semantics, i.e., the first dimension is reduced and
//     the second is scattered. This is why we choose the layouts to be [S,
//     sharded_axis, M, ...]

TEST_F(OverlapTest, SimpleComputeComm) {
  for (auto j : c10::irange(params.S)) {
    auto ta_j = getSlice(ta, j);
    auto tc_unreduced_j = getSlice(tc_unreduced, j);
    auto tc_partially_reduced_j = getSlice(tc_locally_reduced, j);
    auto tc_j = getSlice(tc, j);

    if (params.use_different_streams) {
      setCurrentCUDAStream(c10::cuda::getStreamFromPool(
          /*isHighPriority=*/ true, my_device_index));
    }
    // local compute
    computeATen(ta_j, tb, tc_unreduced_j, tc_partially_reduced_j);
    // communication
    getWorldCommunicator()
        ->_reduce_scatter_base(tc_j, tc_partially_reduced_j)
        ->wait();
  }

  // validation
  ASSERT_TRUE(tc.allclose(tc_expected, 1e-3, 1e-3))
      << "Unexpected results, obtained:" << tc
      << "\n expected: " << tc_expected;
}

} // namespace nvfuser

#endif