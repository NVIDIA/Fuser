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
#include <host_ir/container.h>
#include <host_ir/executor.h>
#include <host_ir/host_ir.h>
#include <ir/utils.h>
#include <ops/all_ops.h>
#include <tests/cpp/multidevice.h>

namespace nvfuser {

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
      true; // whether to change CUDA stream at each iteration
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
  std::vector<int64_t> all_devices_;
  at::Tensor ta_, tb_, tc_locally_reduced_, tc_, tc_expected_;
  // stores the backend
  c10d::Backend* world_communicator_;

  void SetUp() {
    MultiDeviceTest::SetUp();

    num_devices_ = communicator_->size();
    my_device_index_ = communicator_->deviceId();
    ASSERT_EQ(params.M % (params.S * num_devices_), 0);
    ASSERT_EQ(params.K % num_devices_, 0);

    // Setup the world communicators
    std::vector<int64_t> devices(num_devices_);
    std::iota(devices.begin(), devices.end(), 0);
    all_devices_ = std::move(devices);
    world_communicator_ =
        communicator_->getBackendForTeam(all_devices_, params.backend_type);

    // Define I/O and intermediate Tensor shapes
    std::vector<int64_t> ta_unsharded_sizes = {
        params.S, num_devices_, params.M / params.S, params.K / num_devices_};
    std::vector<int64_t> ta_sizes = {
        params.S, 1, params.M / params.S, params.K / num_devices_};
    std::vector<int64_t> tb_unsharded_sizes = {
        1, num_devices_, params.K / num_devices_, params.N};
    std::vector<int64_t> tb_sizes = {1, 1, params.K / num_devices_, params.N};
    std::vector<int64_t> tc_locally_reduced_sizes = {
        params.S, 1, params.M / params.S, params.N};
    std::vector<int64_t> tc_sizes = {
        params.S, 1, params.M / (params.S * num_devices_), params.N};

    // Set up input tensors. We create the full unsharded tensors and define the
    // actual input as the shard corresponding to the current device. Having the
    // full unsharded input on each rank makes it possible to compute the
    // expected result locally, hence, this way of doing is convenient for
    // validating data correctness.
    at::TensorOptions options =
        at::TensorOptions().dtype(at::kFloat).device(communicator_->device());
    auto ta_unsharded = at::randn(ta_unsharded_sizes, options);
    auto tb_unsharded = at::randn(tb_unsharded_sizes, options);
    ta_ = at::empty(ta_sizes, options);
    ta_.copy_(getSlice(ta_unsharded, 1, my_device_index_));
    tb_ = at::empty(tb_sizes, options);
    tb_.copy_(getSlice(tb_unsharded, 1, my_device_index_));

    // We pre-allocate the output and some intermediate buffers so we do not
    // rely on torch allocator, which do not behave well with multi-stream
    // programming.
    tc_locally_reduced_ = at::empty(tc_locally_reduced_sizes, options);
    tc_ = at::empty(tc_sizes, options);

    // compute the expected output for data correctness validation
    auto tc_unsharded_unreduced =
        ta_unsharded.unsqueeze(-1) * tb_unsharded.unsqueeze(-3);
    auto tc_unsharded_expected = at::sum(tc_unsharded_unreduced, {1, 3});
    auto tc_unsharded_expected_reshaped = at::reshape(
        tc_unsharded_expected,
        {params.S,
         num_devices_,
         params.M / (params.S * num_devices_),
         params.N});
    tc_expected_ =
        getSlice(tc_unsharded_expected_reshaped, 1, my_device_index_);

    // Debug print
    if (communicator_->deviceId() == 0 && debug_print) {
      debug() << params << std::endl
              << "ta_.sizes()=" << ta_.sizes() << std::endl
              << "tb_.sizes()=" << tb_.sizes() << std::endl
              << "tc_locally_reduced_.sizes()=" << tc_locally_reduced_.sizes()
              << std::endl
              << "tc_.sizes()=" << tc_.sizes() << std::endl;
    }
  }

  at::Tensor getSlice(at::Tensor t, int64_t axis, int64_t index) {
    return t.slice(axis, index, index + 1);
  }

  void computeATen(
      at::Tensor ta_j,
      at::Tensor tb_j,
      at::Tensor tc_locally_reduced_j) {
    torch::matmul_out(tc_locally_reduced_j, ta_j, tb_j);
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
TEST_F(OverlapTest, ReduceScatterBasedPipeliningATenImplementation) {
  std::vector<c10::cuda::CUDAStream> streams;
  for (auto j : c10::irange(params.S)) {
    // define the sliced tensors
    auto ta_j = getSlice(ta_, 0, j);
    auto tc_locally_reduced_j = getSlice(tc_locally_reduced_, 0, j);
    auto tc_j = getSlice(tc_, 0, j);

    if (params.use_different_streams) {
      auto new_stream = c10::cuda::getStreamFromPool(
          /*isHighPriority=*/false, my_device_index_);
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
  EXPECT_TRUE(tc_.allclose(tc_expected_, 1e-1, 1e-1))
      << "Unexpected results, obtained:" << tc_
      << "\n expected: " << tc_expected_;
}

TEST_F(OverlapTest, ReduceScatterBasedPipeliningHostIrImplementation) {
  // returns tv[j:j+1,...]
  auto getSymbolicSlice = [](TensorView* tv, Val* j) -> TensorView* {
    Val* one = tv->container()->oneVal();
    Slice range = {.start = j, .stop = add(j, one), .step = one};
    std::vector<Slice> ranges(tv->nDims());
    ranges.at(0) = range;
    return slice(tv, ranges);
  };

  auto hic = std::make_unique<hir::HostIrContainer>();
  FusionGuard::setCurFusion(hic.get());

  constexpr int64_t n_dims = 4;
  TensorView* tva = makeSymbolicTensor(n_dims);
  TensorView* tvb = makeSymbolicTensor(n_dims);
  TensorView* tvc = makeSymbolicTensor(n_dims);
  hic->addInput(tva);
  hic->addInput(tvb);
  hic->addInput(tvc);

  auto* j =
      IrBuilder::create<Val>(DataType::Index); // running index of the for-loop
  auto* start = hic->zeroVal();
  auto* stop = IrBuilder::create<Val>(params.S, DataType::Index);
  auto* step = hic->oneVal();
  auto* for_loop = IrBuilder::create<ForLoop>(
      /*IterDomain=*/tva->axis(0),
      /*index=*/j,
      start,
      stop,
      step,
      /*vectorize=*/false,
      /*vectorize_shift=*/nullptr,
      /*unroll_required=*/false,
      CircularBufferLoopStage::NotApplicable);

  TensorView* tva_j = getSymbolicSlice(tva, j);
  TensorView* tvc_j = getSymbolicSlice(tvc, j);
  TensorView* tvc_locally_reduced_j =
      matmul(tva_j, tvb); // ideally we should use the preallocated global
                          // buffer tc_locally_reduced, but ExpressionEvaluator
                          // do not support preallocated output buffer.

  // Setting the DeviceMesh of the communication's I/O is artificial but
  // required at this point
  DeviceMesh full_mesh(all_devices_);
  tvc_j->setDeviceMesh(full_mesh);
  tvc_locally_reduced_j->setDeviceMesh(full_mesh);

  auto* communication = IrBuilder::create<Communication>(
      CommunicationType::ReduceScatter,
      /*out=*/tvc_j,
      /*in=*/tvc_locally_reduced_j,
      /*team=*/all_devices_,
      /*(unused)root=*/-1,
      RedOpType::SUM,
      /*scattered_axis=*/0);
  auto* wait = IrBuilder::create<hir::Wait>(communication);

  // Slice and MatmulOp are present directly as Host IRs in the HostIrContainer.
  // It means that they are going to be executed at the host level (actually,
  // through ExpressionEvaluator). Alternatively, they could be embedded in a
  // separate Fusion and be added to the HostIrConainter through
  // PostOnStrean(HostUnit(.)), in which case the ops would be codegen-ed and
  // compiled.
  std::vector<Expr*> loop_body = {
      tva_j->definition(),
      tvc_j->definition(),
      tvc_locally_reduced_j->definition(),
      communication,
      wait};
  for (Expr* expr : loop_body) {
    for_loop->body().push_back(expr);
  }
  hic->pushBackTopLevelExprs(for_loop);

  // The following line is artificial but necessary to make
  // tva_j->isProducerOf(tvc_locally_reduced_j) == true
  hic->addOutput(tvc_locally_reduced_j);

  hir::HostIrExecutor hie(std::move(hic), communicator_);
  std::unordered_map<Val*, c10::IValue> inputs = {
      {tva, ta_}, {tvb, tb_}, {tvc, tc_}};
  hie.runWithInput(std::move(inputs));

  EXPECT_TRUE(tc_.allclose(tc_expected_, 1e-1, 1e-1))
      << "Unexpected results, obtained:" << tc_
      << "\n expected: " << tc_expected_;
}

} // namespace nvfuser
