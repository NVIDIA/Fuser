// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#include <ATen/Functions.h>
#include <c10/cuda/CUDAStream.h>
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
    if (isEnvVariableDefined("USE_DIFFERENT_STREAMS")) {
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
      << "}";
  return out;
}

class OverlapTest : public MultiDeviceTest {
 protected:
  OverlapTestParams params;

  int64_t num_devices_;
  int64_t my_device_index_;
  at::Tensor ta_, tb_, tc_locally_reduced_, tc_, tc_expected_;
  // stores the backend
  c10d::Backend* world_communicator_;

  void SetUp() {
    MultiDeviceTest::SetUp();

    params.parseEnv();
    num_devices_ = communicator->size();
    my_device_index_ = communicator->deviceId();
    ASSERT_EQ(params.M % (params.S * num_devices_), 0);
    ASSERT_EQ(params.K % num_devices_, 0);

    // Setup the world communicators
    std::vector<int64_t> devices(num_devices_);
    std::iota(devices.begin(), devices.end(), 0);
    world_communicator_ =
        communicator->getBackendForTeam(devices, params.backend_type);

    // Define I/O and intermediate Tensor shapes
    // clang-format off
        std::vector<int64_t> ta_unsharded_sizes       = {params.S, num_devices_, params.M/params.S, params.K/num_devices_, 1       };
        std::vector<int64_t> ta_sizes                 = {params.S, 1           , params.M/params.S, params.K/num_devices_, 1       };
        std::vector<int64_t> tb_unsharded_sizes       = {1       , num_devices_, 1                , params.K/num_devices_, params.N};
        std::vector<int64_t> tb_sizes                 = {1       , 1           , 1                , params.K/num_devices_, params.N};
        std::vector<int64_t> tc_locally_reduced_sizes = {params.S, 1           , params.M/params.S,                        params.N};
        std::vector<int64_t> tc_sizes                 = {params.S, 1           , params.M/(params.S*num_devices_)        , params.N};
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
    ta_ = at::empty(ta_sizes, options);
    ta_.copy_(ta_unsharded.slice(1, my_device_index_, my_device_index_ + 1));
    tb_ = at::empty(tb_sizes, options);
    tb_.copy_(tb_unsharded.slice(1, my_device_index_, my_device_index_ + 1));

    // We pre-allocate the output and some intermediate buffers so we do not
    // rely on torch allocator, which do not behave well with multi-stream
    // programming.
    tc_locally_reduced_ = at::empty(tc_locally_reduced_sizes, options);
    tc_ = at::empty(tc_sizes, options);

    // compute the expected output for data correctness validation
    auto tc_unsharded_unreduced = ta_unsharded * tb_unsharded;
    auto tc_unsharded_expected = at::sum(tc_unsharded_unreduced, {1, 3});
    auto tc_unsharded_expected_reshaped = at::reshape(
        tc_unsharded_expected,
        {params.S,
         num_devices_,
         params.M / (params.S * num_devices_),
         params.N});
    tc_expected_ = tc_unsharded_expected_reshaped.slice(
        1, my_device_index_, my_device_index_ + 1);

    // Debug print
    if (communicator->deviceId() == 0 && params.debug_print) {
      std::cout << params << std::endl;
      std::cout << "ta_.sizes()=" << ta_.sizes() << std::endl;
      std::cout << "tb_.sizes()=" << tb_.sizes() << std::endl;
      std::cout << "tc_locally_reduced_.sizes()=" << tc_locally_reduced_.sizes()
                << std::endl;
      std::cout << "tc_.sizes()=" << tc_.sizes() << std::endl;
    }
  }

  at::Tensor getSlice(at::Tensor t, int64_t j) {
    return t.slice(0, j, j + 1);
  }

  void computeATen(
      at::Tensor ta_j,
      at::Tensor tb_j,
      at::Tensor tc_locally_reduced_j) {
    // we unsqueeze the output tensor to avoid a torch warning:
    // "W624 08:13:32.342934752 Resize.cpp:28] Warning: An output with one or
    // more elements was resized since it had shape [1, 1, 8, 16], which does
    // not match the required output shape [1, 1, 8, 1, 1, 1, 1, 16]. This
    // behavior is deprecated, and in a future PyTorch release outputs will not
    // be resized unless they have zero elements. You can explicitly reuse an
    // out tensor t by resizing it, inplace, to zero elements with t.resize_(0).
    // (function _resize_output_check)"
    auto unsqueezed_tc_locally_reduced_j =
        tc_locally_reduced_j.unsqueeze(3).unsqueeze(4).unsqueeze(5).unsqueeze(
            6);
    // we check that no unnecessary copy is performed
    EXPECT_EQ(
        tc_locally_reduced_j.data_ptr(),
        unsqueezed_tc_locally_reduced_j.data_ptr());
    at::tensordot_out(unsqueezed_tc_locally_reduced_j, ta_j, tb_j, {3}, {3});
  }
};
// clang-format off
// This test implements a reduce-scattered based pipelining overlapping technic,
// as used in NEMO-megatron transformer, precisely at the second layer of the
// MLP consisting of a GEMM+Reduce-scatter.
//
// The tensor program that we target is
// the following, assuming a setup with `num_devices_` devices:
//     inputs:
//        - A[M,K] sharded column-wise:
//          dimension K is split by the factor `num_devices_`
//          so A is viewed as [M, num_devices_, K/num_devices_]
//          and the allocation size of A is [M, 1, K/num_devices_]
//        - B[K,N] sharded row-wise:
//          locally of size [1, K/num_devices_, N]
//     output:
//        - C[M,N]=matmul(A,B), sharded on dimension M:
//          dimension M is split by the factor `num_devices_`
//          so C is viewed as [num_devices_, M/num_devices_,N]
//          and the allocation size of M is [1, M/num_devices_,N]
// Up to some broadcast and view ops, a straightforward program to generate the
// output could be summarized as
//     | C_unreduced = pointwise_multiply(A,B)
//     | C_locally_reduce = reduction(C_unreduced, axis=`K/num_devices_`, op=sum)
//     | C = reduce_scatter(C_unreduced, op=sum)
// where:
// - C has unsharded size [M,num_devices_,K/num_devices_,N],
//    and is sharded on `num_devices_`
// - C_locally_reduce has unsharded size [M,num_devices_,N],
//    and is sharded on `num_devices_`
// - C has unsharded size [num_devices_, M/num_devices_, N]
//    and is sharded on `num_devices_`
//
// We want to compare this baseline program with one that is functionnally
// identical but achieves more overlap between computations and communications.
// Our goal is to interlave the comms and compute using a technic called
// "reduce-scatter based pipelining". To do so, we further split the row
// dimension M with a factor `S` representing the number of tiles, and we apply
// the operations successively on tensors slices accross S, changing stream at
// each iteration. Assuming the following shapes:
//     - A [S, num_devices_, M/S, K/num_devices_], sharded on num_devices_
//     - B [num_devices_, K/num_devices_, N], sharded on num_devices_
//     - C [S, num_devices_, M/(S*num_devices_), N], sharded on num_devices_
// the program could be summarized as:
//     | for (j=0; j<S; j++) {
//     |   setCurrentStream(Stream[j])
//     |   C_unreduced[j] = pointwise_multiply(A[j],B)
//     |   C_locally_reduce[j] = local_reduction(C_unreduced[j], axis=`K/num_devices_`, op=sum)
//     |   C[j]=reduce_scatter(C_locally_reduce[j], op=sum)
//     | }
// where "[j]" referes to taking a slice onto the `S` dimension.
// Remarks:
//   1) it is convenient to have "S" as being the outermost dimension so
//      C_locally_reduce[j] is a contiguous buffer.
//   2) The layout needs to match
//      the reduce-scatter semantics, i.e., the first dimension is reduced and
//      the second is scattered. This is why we choose the layouts to be
//      [S, sharded_axis, M, ...]
// clang-format on
TEST_F(OverlapTest, SimpleComputeComm) {
  std::vector<c10::cuda::CUDAStream> streams;
  for (auto j : c10::irange(params.S)) {
    // define the sliced tensors
    auto ta_j = getSlice(ta_, j);
    auto tc_locally_reduced_j = getSlice(tc_locally_reduced_, j);
    auto tc_j = getSlice(tc_, j);

    if (params.use_different_streams) {
      auto new_stream = c10::cuda::getStreamFromPool(
          /*isHighPriority=*/true, my_device_index_);
      streams.push_back(new_stream);
      setCurrentCUDAStream(new_stream);
    }

    // local compute
    computeATen(ta_j, tb_, tc_locally_reduced_j);
    // communication
    world_communicator_->_reduce_scatter_base(tc_j, tc_locally_reduced_j)
        ->wait();
  }

  // synchronize default stream with all other streams
  setCurrentCUDAStream(c10::cuda::getDefaultCUDAStream());
  for (auto stream : streams) {
    stream.synchronize();
  }

  // validation
  EXPECT_TRUE(tc_.allclose(tc_expected_, 1e-3, 1e-3))
      << "Unexpected results, obtained:" << tc_
      << "\n expected: " << tc_expected_;
}

} // namespace nvfuser
