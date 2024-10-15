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

namespace {

std::vector<c10::cuda::CUDAStream> createStreams(
    int64_t number_of_streams,
    int64_t my_device_index) {
  std::vector<c10::cuda::CUDAStream> streams;
  std::generate_n(
      std::back_inserter(streams), number_of_streams, [my_device_index]() {
        return c10::cuda::getStreamFromPool(
            /*isHighPriority=*/false, my_device_index);
      });
  return streams;
}

void synchronizeStreams(const std::vector<c10::cuda::CUDAStream>& streams) {
  std::for_each(streams.begin(), streams.end(), [](const auto& stream) {
    stream.synchronize();
  });
}

} // namespace

struct OverlapTestParams {
  // Tensors sizes
  int64_t M = std::pow(2, 6);
  int64_t K = std::pow(2, 5);
  int64_t N = std::pow(2, 4);
  int64_t S = std::pow(2, 3);

  // network backend type
  CommunicatorBackend backend_type = CommunicatorBackend::kNccl;

  // Overlap optimization parameters
  // fill input with new random values and repeat the operation
  int64_t number_of_iterations = 4;
  // Change CUDA stream at each iteration in a Round-Robin fashion
  int64_t number_of_streams = 3;
};

std::ostream& operator<<(std::ostream& out, const OverlapTestParams& params) {
  std::string indent = "  ";
  out << "params:{\n"
      << indent << "backend_type=" << params.backend_type << "\n"
      << indent << "M=" << params.M << "\n"
      << indent << "K=" << params.K << "\n"
      << indent << "N=" << params.N << "\n"
      << indent << "S=" << params.S << "\n"
      << indent << "number_of_streams=" << params.number_of_streams << "\n"
      << indent << "number_of_iterations=" << params.number_of_iterations
      << "\n}";
  return out;
}

class OverlapTest : public MultiDeviceTest {
 protected:
  OverlapTestParams params;

  int64_t num_devices_;
  int64_t my_device_index_;
  std::vector<int64_t> all_devices_;
  at::Tensor ta_unsharded_, tb_unsharded_;
  at::Tensor ta_, tb_, tc_;
  at::TensorOptions gpu_options_;
  // stores the backend
  c10d::Backend* world_communicator_;

  void SetUp() override {
    MultiDeviceTest::SetUp();
    if (!communicator_->is_available()) {
      return;
    }

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

    // Debug print
    if (communicator_->deviceId() == 0 && debug_print) {
      debug() << params << std::endl;
    }

    // Define I/O and intermediate Tensor shapes
    std::vector<int64_t> ta_unsharded_sizes = {
        params.S, num_devices_, params.M / params.S, params.K / num_devices_};
    std::vector<int64_t> ta_sizes = {
        params.S, params.M / params.S, params.K / num_devices_};
    std::vector<int64_t> tb_unsharded_sizes = {
        num_devices_, params.K / num_devices_, params.N};
    std::vector<int64_t> tb_sizes = {params.K / num_devices_, params.N};
    std::vector<int64_t> tc_sizes = {
        params.S, params.M / (params.S * num_devices_), params.N};

    // Set up input tensors. We create the full unsharded tensors and define the
    // actual input as the shard corresponding to the current device. Having the
    // full unsharded input on each rank makes it possible to compute the
    // expected result locally, hence, this way of doing is convenient for
    // validating data correctness.
    auto cpu_options = at::TensorOptions().dtype(at::kFloat);
    gpu_options_ = cpu_options.device(communicator_->device());

    // Unsharded tensors are large and only used for validating data corectness.
    // Therefore, to improve GPU memory footprint, we allocate those tensors on
    // the CPU
    ta_unsharded_ = at::empty(ta_unsharded_sizes, cpu_options);
    tb_unsharded_ = at::empty(tb_unsharded_sizes, cpu_options);
    ta_ = at::empty(ta_sizes, gpu_options_);
    tb_ = at::empty(tb_sizes, gpu_options_);
    tc_ = at::empty(tc_sizes, gpu_options_);

    // Debug print
    if (communicator_->deviceId() == 0 && debug_print) {
      debug() << "ta_sizes()=" << ta_.sizes() << std::endl
              << "tb_sizes()=" << tb_.sizes() << std::endl
              << "tc_sizes()=" << tc_.sizes() << std::endl;
    }
  }

  void initializeIO() {
    ta_unsharded_.uniform_();
    tb_unsharded_.uniform_();
    ta_.copy_(ta_unsharded_.select(1, my_device_index_));
    tb_.copy_(tb_unsharded_.select(0, my_device_index_));
  }

  // compute the expected output for data correctness validation
  at::Tensor getUnshardedExpectedResult() {
    auto tc_unsharded_unreduced =
        ta_unsharded_.unsqueeze(-1) * tb_unsharded_.unsqueeze(-3).unsqueeze(0);
    return at::sum(tc_unsharded_unreduced, {1, 3});
  }

  virtual at::Tensor getExpectedResult() {
    NVF_THROW("must be implemented in child class");
    return at::Tensor();
  }

  void validate() {
    auto tc_expected = getExpectedResult();
    auto tc_cpu = tc_.to(torch::kCPU);
    EXPECT_TRUE(tc_cpu.allclose(tc_expected, 1e-1, 1e-1))
        << "Unexpected results, obtained:" << tc_cpu
        << "\n expected: " << tc_expected;
  }

  void TearDown() override {
    validate();
    MultiDeviceTest::TearDown();
  }
};

class CollectiveBasedOverlapTest : public OverlapTest {
 protected:
  at::Tensor tc_locally_reduced_;
  void SetUp() override {
    OverlapTest::SetUp();

    std::vector<int64_t> tc_locally_reduced_sizes = {
        std::min(params.S, params.number_of_streams),
        params.M / params.S,
        params.N}; // we need at most `number_of_streams` slices
    tc_locally_reduced_ = at::empty(tc_locally_reduced_sizes, gpu_options_);
  }

  at::Tensor getExpectedResult() override {
    auto tc_unsharded_expected = getUnshardedExpectedResult();
    auto tc_unsharded_expected_reshaped = at::reshape(
        tc_unsharded_expected,
        {params.S,
         num_devices_,
         params.M / (params.S * num_devices_),
         params.N});
    return tc_unsharded_expected_reshaped.select(1, my_device_index_);
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
TEST_F(
    CollectiveBasedOverlapTest,
    ReduceScatterBasedPipeliningATenImplementation) {
  std::vector<c10::cuda::CUDAStream> streams =
      createStreams(params.number_of_streams, my_device_index_);

  for ([[maybe_unused]] const auto& _ :
       c10::irange(params.number_of_iterations)) {
    initializeIO();

    for (auto j : c10::irange(params.S)) {
      int64_t stream_index = j % streams.size();
      setCurrentCUDAStream(streams.at(stream_index));

      // define the sliced tensors
      auto ta_j = ta_.select(0, j);
      auto tc_locally_reduced_j = tc_locally_reduced_.select(0, stream_index);
      auto tc_j = tc_.select(0, j);

      // local compute
      torch::matmul_out(tc_locally_reduced_j, ta_j, tb_);
      // communication
      world_communicator_->_reduce_scatter_base(tc_j, tc_locally_reduced_j)
          ->wait();
    }
  }
  synchronizeStreams(streams);
}

TEST_F(
    CollectiveBasedOverlapTest,
    ReduceScatterBasedPipeliningHostIrImplementation) {
  auto hic = std::make_unique<hir::HostIrContainer>();
  FusionGuard::setCurFusion(hic.get());

  TensorView* tva = makeSymbolicTensor(ta_.dim());
  TensorView* tvb = makeSymbolicTensor(tb_.dim());
  TensorView* tvc = makeSymbolicTensor(tc_.dim());
  TensorView* tvc_locally_reduced =
      makeSymbolicTensor(tc_locally_reduced_.dim());
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

  auto* stream_index = mod(j, IrBuilder::create<Val>(params.number_of_streams));
  auto* set_stream = IrBuilder::create<hir::SetCurrentStream>(
      IrBuilder::create<hir::Stream>(stream_index));

  TensorView* tva_j = select(tva, 0, j);
  TensorView* tvc_j = select(tvc, 0, j);
  TensorView* tvc_locally_reduced_j =
      select(tvc_locally_reduced, 0, stream_index);
  auto* matmul = IrBuilder::create<MatmulOp>(tvc_locally_reduced_j, tva_j, tvb);

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
      set_stream,
      tva_j->definition(),
      tvc_j->definition(),
      tvc_locally_reduced_j->definition(),
      matmul,
      communication,
      wait};
  for (Expr* expr : loop_body) {
    for_loop->body().push_back(expr);
  }

  hic->pushBackTopLevelExprs(for_loop);

  // Synchronize all streams
  auto* i_stream =
      IrBuilder::create<Val>(DataType::Index); // running index of the for-loop
  auto* start_stream = hic->zeroVal();
  auto* stop_stream =
      IrBuilder::create<Val>(params.number_of_streams, DataType::Index);
  auto* step_stream = hic->oneVal();
  auto* for_loop_stream = IrBuilder::create<ForLoop>(
      /*IterDomain=*/makeContigConcreteTensor({params.number_of_streams})
          ->axis(0),
      /*index=*/i_stream,
      start_stream,
      stop_stream,
      step_stream,
      /*vectorize=*/false,
      /*vectorize_shift=*/nullptr,
      /*unroll_required=*/false,
      CircularBufferLoopStage::NotApplicable);
  auto* sync_stream = IrBuilder::create<hir::Synchronize>(
      IrBuilder::create<hir::Stream>(i_stream));
  for_loop_stream->body().push_back(sync_stream);
  hic->pushBackTopLevelExprs(for_loop_stream);

  // The following line is artificial but necessary to make
  // tva_j->isProducerOf(tvc_locally_reduced_j) == true
  hic->addOutput(tvc_locally_reduced_j);

  hir::HostIrExecutor hie(std::move(hic), communicator_);

  for ([[maybe_unused]] const auto& _ :
       c10::irange(params.number_of_iterations)) {
    initializeIO();
    std::unordered_map<Val*, c10::IValue> inputs = {
        {tva, ta_},
        {tvb, tb_},
        {tvc, tc_},
        {tvc_locally_reduced, tc_locally_reduced_}};

    hie.runWithInput(std::move(inputs));
  }
}

class RingBasedOverlapTest : public OverlapTest {
 protected:
  int64_t number_of_steps_per_ring_, number_of_rings_;
  at::Tensor src_buffer_, dst_buffer_;
  at::Tensor ta_reshaped_, tc_reshaped_;

  void SetUp() override {
    OverlapTest::SetUp();

    ASSERT_EQ(params.S % num_devices_, 0);
    number_of_steps_per_ring_ = num_devices_;
    number_of_rings_ = params.S / num_devices_;

    ta_reshaped_ = at::reshape(
        ta_,
        {number_of_steps_per_ring_,
         number_of_rings_,
         params.M / params.S,
         params.K / num_devices_});
    tc_reshaped_ =
        tc_.reshape({number_of_rings_, params.M / params.S, params.N});

    std::vector<int64_t> buffer_sizes = {
        number_of_steps_per_ring_,
        number_of_rings_,
        params.M / params.S,
        params.N};
    src_buffer_ = at::empty(buffer_sizes, gpu_options_);
    dst_buffer_ = at::empty(buffer_sizes, gpu_options_);
  }

  at::Tensor getExpectedResult() override {
    auto tc_unsharded_expected = getUnshardedExpectedResult();
    // the natural layout here differs from the collective based pipelining.
    // Here, the output is sharded on the outermost axis whereas, in the
    // collective based pipelining, it is sharded on axis(1). The two layouts
    // coincide in the classical case where params.S = num_devices_
    auto tc_unsharded_expected_reshaped = at::reshape(
        tc_unsharded_expected,
        {num_devices_, params.S / num_devices_, params.M / params.S, params.N});
    auto tc_expected =
        tc_unsharded_expected_reshaped.select(0, my_device_index_);
    return tc_expected.reshape(
        {params.S, params.M / (params.S * num_devices_), params.N});
  }
};

TEST_F(
    RingBasedOverlapTest,
    ReduceScatterRingBasedPipeliningATenImplementation) {
  std::vector<c10::cuda::CUDAStream> streams =
      createStreams(params.number_of_streams, my_device_index_);

  for ([[maybe_unused]] const auto& _ :
       c10::irange(params.number_of_iterations)) {
    initializeIO();

    for (auto i : c10::irange(number_of_rings_)) {
      for (auto j : c10::irange(number_of_steps_per_ring_)) {
        int64_t stream_index = (i + j) % streams.size();
        setCurrentCUDAStream(streams.at(stream_index));

        // define the sliced tensors
        auto slice_index =
            (my_device_index_ + j + 1) % number_of_steps_per_ring_;
        auto ta_j = ta_reshaped_.select(0, slice_index).select(0, i);
        auto src_buffer_j = src_buffer_.select(0, j).select(0, i);
        auto dst_buffer_j = dst_buffer_.select(0, j).select(0, i);

        // define the peer ranks
        auto send_rank = slice_index;
        auto recv_rank =
            (number_of_steps_per_ring_ + my_device_index_ - (j + 1)) %
            number_of_steps_per_ring_;

        // local compute
        torch::matmul_out(src_buffer_j, ta_j, tb_);
        // communication
        std::vector<at::Tensor> src = {src_buffer_j};
        std::vector<at::Tensor> dst = {dst_buffer_j};

        world_communicator_->startCoalescing();
        // "tags" are not supported by nccl, so set it to 0
        world_communicator_->send(src, send_rank, 0);
        world_communicator_->recv(dst, recv_rank, 0);
        world_communicator_->endCoalescing()->wait();
      }
    }
    synchronizeStreams(streams);
    torch::sum_out(tc_reshaped_, dst_buffer_, 0);
  }
}

} // namespace nvfuser
