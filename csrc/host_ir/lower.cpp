// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#include <device_lower/utils.h>
#include <fusion_segmenter.h>
#include <host_ir/lower.h>
#include <ir/builder.h>
#include <ir/interface_nodes.h>
#include <ir/iostream.h>
#include <multidevice/device_mesh.h>
#include <multidevice/utils.h>
#include <ops/all_ops.h>
#include <ops/utils.h>
#include <preseg_passes/insert_reshardings.h>
#include <preseg_passes/make_resharding_contiguous.h>
#include <preseg_passes/propagate_shardings.h>
#include <preseg_passes/reorder_sharded_axis.h>
#include <runtime/fusion_kernel_runtime.h>
#include <limits>

namespace nvfuser {

namespace {

// TODO: handle `c10d::RedOpType::reduceOp::AVG` and
// `c10d::RedOpType::reduceOp::PREMUL_SUM`
inline c10d::ReduceOp::RedOpType getC10dReduceOpType(BinaryOpType op) {
  switch (op) {
    case BinaryOpType::Add:
      return c10d::ReduceOp::RedOpType::SUM;
    case BinaryOpType::Mul:
      return c10d::ReduceOp::RedOpType::PRODUCT;
    case BinaryOpType::Min:
      return c10d::ReduceOp::RedOpType::MIN;
    case BinaryOpType::Max:
      return c10d::ReduceOp::RedOpType::MAX;
    case BinaryOpType::BitwiseAnd:
      return c10d::ReduceOp::RedOpType::BAND;
    case BinaryOpType::BitwiseOr:
      return c10d::ReduceOp::RedOpType::BOR;
    case BinaryOpType::BitwiseXor:
      return c10d::ReduceOp::RedOpType::BXOR;
    default:
      NVF_THROW("unsupported reduction operation");
      return c10d::ReduceOp::RedOpType::UNUSED;
  }
}

// Adds one or zero Scatter communication to the vector 'comms'
void lowerToScatter(
    TensorView* input_tv,
    TensorView* output_tv,
    const HostIrLowerParams& params,
    std::vector<Expr*>& comms) {
  // we arbitrarily choose the first device of the sender mesh to be the root
  const DeviceMesh& receiver_mesh = output_tv->getDeviceMesh();
  NVF_ERROR(
      receiver_mesh.rank() == 1,
      "Gather only supported on a 1D mesh. Given ",
      receiver_mesh);
  auto root = input_tv->getDeviceMesh().at(0);
  Team team = receiver_mesh.vector();
  if (!receiver_mesh.has(root)) {
    team.push_back(root);
  }
  comms.push_back(IrBuilder::create<Communication>(
      CommunicationType::Scatter,
      output_tv,
      input_tv,
      team,
      root,
      c10d::ReduceOp::RedOpType::UNUSED,
      /*scatter_axis=*/-1,
      params.communicator_backend));
}

/*
Adds zero or multiple Gather communications to the vector 'comms'

Note that since the root of a Gather collective is a destination, we possibly
need multiple Gathers if the tensor is replicated in the receiver mesh.
*/
void lowerToGather(
    TensorView* input_tv,
    TensorView* output_tv,
    const HostIrLowerParams& params,
    std::vector<Expr*>& comms) {
  // we create as many 'Gathers' as there are devices in the receiver mesh
  const DeviceMesh& sender_mesh = input_tv->getDeviceMesh();
  NVF_ERROR(
      sender_mesh.rank() == 1,
      "Currently only lower Gather on a 1D mesh. Given ",
      sender_mesh);
  for (auto root : output_tv->getDeviceMesh().vector()) {
    Team team = sender_mesh.vector();
    if (!sender_mesh.has(root)) {
      team.push_back(root);
    }
    comms.push_back(IrBuilder::create<Communication>(
        CommunicationType::Gather,
        output_tv,
        input_tv,
        team,
        root,
        c10d::ReduceOp::RedOpType::UNUSED,
        /*scatter_axis=*/-1,
        params.communicator_backend));
  }
}

// Add one or zero Allgather communication to the vector 'comms'
void lowerToAllgather(
    TensorView* input_tv,
    TensorView* output_tv,
    const HostIrLowerParams& params,
    std::vector<Expr*>& comms,
    DeviceIdxType my_device_idx) {
  const DeviceMesh& mesh = input_tv->getDeviceMesh();
  Team team = mesh.getSlice(my_device_idx, ParallelType::DIDx);
  comms.push_back(IrBuilder::create<Communication>(
      CommunicationType::Allgather,
      output_tv,
      input_tv,
      team,
      /*root=*/-1,
      c10d::ReduceOp::RedOpType::UNUSED,
      /*scatter_axis=*/-1,
      params.communicator_backend));
}

// Adds one or zero Broadcast communication to the vector 'comms'
void lowerToBroadcast(
    TensorView* input_tv,
    TensorView* output_tv,
    DeviceIdxType root,
    const HostIrLowerParams& params,
    std::vector<Expr*>& comms) {
  const DeviceMesh& mesh = output_tv->getDeviceMesh();
  NVF_ERROR(
      mesh.rank() == 1, "Broadcast only supported a 1D mesh. Given ", mesh);
  Team team = mesh.vector();
  if (!mesh.has(root)) {
    team.push_back(root);
  }
  comms.push_back(IrBuilder::create<Communication>(
      CommunicationType::Broadcast,
      output_tv,
      input_tv,
      team,
      root,
      c10d::ReduceOp::RedOpType::UNUSED,
      /*scatter_axis=*/-1,
      params.communicator_backend));
}

// Adds several Broadcast or SendRecv communications to the vector 'comms'
// For now, we assume that this function is called only if
// the input and output have the same sharding. Later we could support more
// general cases.
void lowerToBroadcastOrSendRecv(
    TensorView* input_tv,
    TensorView* output_tv,
    const HostIrLowerParams& params,
    std::vector<Expr*>& comms) {
  const DeviceMesh& sender_mesh = input_tv->getDeviceMesh();
  const DeviceMesh& receiver_mesh = output_tv->getDeviceMesh();
  NVF_ERROR(
      sender_mesh.rank() == 1,
      "Broadcast only supported a 1D mesh. Given ",
      sender_mesh);
  NVF_ERROR(
      receiver_mesh.rank() == 1,
      "Broadcast only supported a 1D mesh. Given ",
      receiver_mesh);
  if (isSharded(input_tv) && sender_mesh.size() > 1) {
    // if the inputs and ouputs are parallelized,
    // we create as many Broadcast as that will be handled in parallel
    NVF_ERROR(
        sender_mesh.size() == receiver_mesh.size(),
        "the receiver and sender meshes have different sizes: ",
        sender_mesh.size(),
        " vs ",
        receiver_mesh.size());
    for (auto i : arange(sender_mesh.size())) {
      const DeviceIdxType sender = sender_mesh.at(i);
      const DeviceIdxType receiver = receiver_mesh.at(i);
      comms.push_back(IrBuilder::create<Communication>(
          CommunicationType::SendRecv,
          output_tv,
          input_tv,
          Team({sender, receiver}),
          /*root=*/sender,
          c10d::ReduceOp::RedOpType::UNUSED,
          /*scatter_axis=*/-1,
          params.communicator_backend));
    }
  } else {
    // Either of the following two cases is happening.
    // 1. `sender_mesh` contains only one device. In this case, we broadcast
    // from that device.
    // 2. `sender_mesh` contains multiple devices but the input is not sharded.
    // In this case, we arbitrarily choose the first device of the sender mesh
    // to be the root.
    lowerToBroadcast(
        input_tv,
        output_tv,
        /*root=*/sender_mesh.at(0),
        params,
        comms);
  }
}

void lowerToReduce(
    TensorView* input_tv,
    TensorView* output_tv,
    BinaryOpType op_type,
    const HostIrLowerParams& params,
    std::vector<Expr*>& comms) {
  const DeviceMesh& receiver_mesh = output_tv->getDeviceMesh();
  const DeviceMesh& sender_mesh = input_tv->getDeviceMesh();
  NVF_ERROR(
      sender_mesh.rank() == 1,
      "Reduce only supported a 1D mesh. Given ",
      sender_mesh);
  NVF_ERROR(
      receiver_mesh.rank() == 1,
      "Reduce only supported a 1D mesh. Given ",
      receiver_mesh);
  const auto reduce_op_type = getC10dReduceOpType(op_type);
  // we create as many Reduces as there are devices in the receiver mesh
  for (auto root : receiver_mesh.vector()) {
    Team team = sender_mesh.vector();
    if (!sender_mesh.has(root)) {
      team.push_back(root);
    }
    comms.push_back(IrBuilder::create<Communication>(
        CommunicationType::Reduce,
        output_tv,
        input_tv,
        team,
        root,
        reduce_op_type,
        /*scatter_axis=*/-1,
        params.communicator_backend));
  }
}

void lowerToAllreduce(
    TensorView* input_tv,
    TensorView* output_tv,
    BinaryOpType op_type,
    const HostIrLowerParams& params,
    std::vector<Expr*>& comms,
    DeviceIdxType my_device_idx) {
  const DeviceMesh& mesh = input_tv->getDeviceMesh();
  Team team = mesh.getSlice(my_device_idx, ParallelType::DIDx);
  comms.push_back(IrBuilder::create<Communication>(
      CommunicationType::Allreduce,
      output_tv,
      input_tv,
      team,
      /*root=*/-1,
      getC10dReduceOpType(op_type),
      /*scatter_axis=*/-1,
      params.communicator_backend));
}

void lowerToReduceScatter(
    TensorView* input_tv,
    TensorView* output_tv,
    BinaryOpType op_type,
    const HostIrLowerParams& params,
    std::vector<Expr*>& comms,
    DeviceIdxType my_device_idx) {
  const DeviceMesh& mesh = input_tv->getDeviceMesh();
  Team team = mesh.getSlice(my_device_idx, ParallelType::DIDx);
  auto reduction_axis = output_tv->getReductionAxis().value();
  auto scattered_axis = getShardedLogicalAxis(output_tv, ParallelType::DIDx);
  // The output tensor is sharded on scattered_axis and needs to be mapped
  // back onto the input. The input has an reduced axis, so the scattered axis
  // is adjusted to account for this. Ex: [DIDx(i0), i1] -> [r0, DIDx(i1)] The
  // scattered_axis is axis=0 on the output and maps to axis=1 on the input.
  if (reduction_axis <= scattered_axis) {
    scattered_axis++;
  }

  comms.push_back(IrBuilder::create<Communication>(
      CommunicationType::ReduceScatter,
      output_tv,
      input_tv,
      /*team=*/team,
      /*root=*/-1,
      getC10dReduceOpType(op_type),
      scattered_axis,
      params.communicator_backend));
}

} // namespace

/*
TODO:
*) Propose several lowering paths for each given communication
   and provide a logic to decide which path to take
*) Leverage replication in the source to create several communications handled
   in parallel. The idea would be to evenly split the destinations accross the
   sources
*) Leverage the topology to ensure that the senders and recerivers are close
*/
std::vector<Expr*> HostIrLower::lower(Expr* c, DeviceIdxType my_device_idx) {
  FusionGuard fg(c->fusion());

  if (c->isOneOf<MatmulOp, LinearOp>()) {
    return lowerToCollectiveBasedPipelinedGemmComm(c);
  }

  std::vector<Expr*> comms;
  NVF_ERROR(
      c->inputs().size() == 1 && c->input(0)->isA<TensorView>() &&
          c->outputs().size() == 1 && c->output(0)->isA<TensorView>(),
      "Input/Output must be single TensorView: ",
      c);
  auto* input_tv = c->input(0)->as<TensorView>();
  auto* output_tv = c->output(0)->as<TensorView>();

  input_tv->setMemoryType(MemoryType::Global);
  output_tv->setMemoryType(MemoryType::Global);

  const DeviceMesh& sender_mesh = input_tv->getDeviceMesh();
  const DeviceMesh& receiver_mesh = output_tv->getDeviceMesh();
  const bool same_mesh = sender_mesh == receiver_mesh;

  // Stores whether the I/O has its first axis parallelized on DIDx
  const bool is_input_sharded = isSharded(input_tv) && sender_mesh.size() > 1;
  const bool is_output_sharded =
      isSharded(output_tv) && receiver_mesh.size() > 1;

  NVF_ERROR(
      HostIrLower::canLower(c),
      "Lowering expression ",
      c->toString(),
      " to communication is not supported");
  NVF_ERROR(
      !isInnerResharding(c),
      "Resharding on an inner axis is not lowerable ",
      c->toString());
  bool is_reduction = c->isA<ReductionOp>();

  if (is_reduction) {
    BinaryOpType op_type = c->as<ReductionOp>()->getReductionOpType();
    NVF_ERROR(
        is_input_sharded || sender_mesh.size() == 1,
        "the comm input must be sharded in case of reduce.",
        "Insert a `set` before the reduction to reshard")
    if (is_output_sharded) {
      NVF_ERROR(
          same_mesh,
          "ReduceScatter operation must have the same sender and receiver device mesh. "
          "Insert a Set operation before or after the reduction to reshard ot another device mesh");
      lowerToReduceScatter(
          input_tv, output_tv, op_type, params_, comms, my_device_idx);
    } else {
      if (same_mesh) {
        lowerToAllreduce(
            input_tv, output_tv, op_type, params_, comms, my_device_idx);
      } else {
        lowerToReduce(input_tv, output_tv, op_type, params_, comms);
      }
    }
  } else {
    if (!is_input_sharded && is_output_sharded) {
      lowerToScatter(input_tv, output_tv, params_, comms);
    } else if (is_input_sharded && !is_output_sharded) {
      if (same_mesh) {
        lowerToAllgather(input_tv, output_tv, params_, comms, my_device_idx);
      } else {
        lowerToGather(input_tv, output_tv, params_, comms);
      }
    } else {
      lowerToBroadcastOrSendRecv(input_tv, output_tv, params_, comms);
    }
  }

  return comms;
}

bool HostIrLower::canLower(Expr* expr, bool ignore_inner_resharding) {
  if (!isResharding(expr)) {
    return true;
  }
  if (!ir_utils::isTvOp(expr)) {
    return false;
  }
  if (auto* reduction = dynamic_cast<ReductionOp*>(expr)) {
    if (!ignore_inner_resharding && isInnerResharding(expr)) {
      return false;
    }
    auto in = reduction->in()->as<TensorView>();
    auto out = reduction->out()->as<TensorView>();
    // get the reduced axis
    std::vector<IterDomain*> reduction_axis;
    std::copy_if(
        out->getLogicalDomain().begin(),
        out->getLogicalDomain().end(),
        std::back_inserter(reduction_axis),
        [](IterDomain* id) { return id->isReduction(); });
    // check whether the reduction involves only one axis
    if (reduction_axis.size() != 1) {
      return false;
    }
    // We check whether the reduced axis is sharded on the input
    const auto c2p_map =
        PairwiseLogicalDomainMap(in, out).mapConsumerToProducer();
    auto c2p_map_it = c2p_map.find(reduction_axis.at(0));
    return c2p_map_it != c2p_map.end() && c2p_map_it->second->isDeviceDim();
  } else if (auto* ldst = dynamic_cast<LoadStoreOp*>(expr)) {
    if (!ignore_inner_resharding && isInnerResharding(expr)) {
      return false;
    }
    return ldst->as<LoadStoreOp>()->opType() == LoadStoreOpType::Set;
  } else if (auto* matmul = dynamic_cast<MatmulOp*>(expr)) {
    // For now we only support out = matmul(a,b) when b, out are fully
    // replicated, a is sharded on axis 1, and out i stream-parallelized on axis
    // 0.
    return !isSharded(matmul->inB()) && !isSharded(matmul->out()) &&
        matmul->inA()->axis(0)->getParallelType() == ParallelType::Serial &&
        getShardedLogicalAxis(matmul->inA(), ParallelType::DIDx) == 1 &&
        matmul->out()->axis(0)->getParallelType() == ParallelType::Stream;
  } else if (auto* linear = dynamic_cast<LinearOp*>(expr)) {
    // For now we only support out = linear(a, b, bias) when b, bias, and out
    // are fully replicated, a is sharded on axis 1, and out i
    // stream-parallelized on axis 0.
    auto* a = linear->inA()->as<TensorView>();
    auto* b = linear->inB()->as<TensorView>();
    auto* bias =
        (linear->has_bias() ? linear->bias()->as<TensorView>() : nullptr);
    auto* out = linear->out()->as<TensorView>();
    return !isSharded(b) && !(linear->has_bias() && isSharded(bias)) &&
        !isSharded(out) &&
        a->axis(0)->getParallelType() == ParallelType::Serial &&
        getShardedLogicalAxis(a, ParallelType::DIDx) == 1 &&
        out->axis(0)->getParallelType() == ParallelType::Stream;
  }
  return false;
}

std::vector<Expr*> HostIrLower::lowerToCollectiveBasedPipelinedGemmComm(
    Expr* expr) {
  NVF_ERROR(
      (expr->isOneOf<MatmulOp, LinearOp>()),
      "Expect a MatmulOp or a LinearOp, but got",
      expr);
  TensorView* tva = nullptr;
  TensorView* tvb = nullptr;
  TensorView* tv_bias = nullptr;
  TensorView* tv_out = nullptr;
  if (auto* matmul = dynamic_cast<MatmulOp*>(expr)) {
    tva = matmul->inA();
    tvb = matmul->inB();
    tv_out = matmul->out();
  } else {
    auto* linear = expr->as<LinearOp>();
    tva = linear->inA()->as<TensorView>();
    tvb = linear->inB()->as<TensorView>();
    tv_bias = (linear->has_bias() ? linear->bias()->as<TensorView>() : nullptr);
    tv_out = linear->out()->as<TensorView>();
    NVF_ERROR(
        !(linear->has_bias() && isSharded(tv_bias)),
        "The bias ",
        tv_bias,
        " is expected to not be sharded");
  }

  NVF_ERROR(
      !isSharded(tvb), "The B operand ", tvb, " is expected to not be sharded");
  NVF_ERROR(
      !isSharded(tv_out),
      "The output ",
      tv_out,
      " is expected to not be sharded");
  NVF_ERROR(
      tv_out->axis(0)->getParallelType() == ParallelType::Stream,
      "The output ",
      tv_out,
      " is expected to be stream-parallelized on axis 0");
  const int64_t sharded_axis_index =
      getShardedLogicalAxis(tva, ParallelType::DIDx);
  IterDomain* stream_axis = tva->axis(0);
  NVF_ERROR(
      stream_axis->getParallelType() == ParallelType::Serial &&
          sharded_axis_index == 1,
      "The operand A ",
      tva,
      " is expected to be sharded on the dimension 1");

  auto hic = FusionGuard::getCurFusion()->as<hir::HostIrContainer>();

  auto* get_current_stream = IrBuilder::create<hir::GetCurrentStream>();
  hir::Stream* original_stream = get_current_stream->stream();

  TensorView* tva_allgathered =
      ops::newValLike(tva, tva->dtype())->as<TensorView>();
  tva_allgathered->axis(sharded_axis_index)->parallelize(ParallelType::Serial);
  tva_allgathered->setMemoryType(MemoryType::Global);
  auto* allocate_tva_allgathered =
      IrBuilder::create<kir::Allocate>(tva_allgathered, MemoryType::Global);

  tv_out->setMemoryType(MemoryType::Global);
  auto* allocate_tv_out =
      IrBuilder::create<kir::Allocate>(tv_out, MemoryType::Global);

  auto* j =
      IrBuilder::create<Val>(DataType::Index); // running index of the for-loop
  auto* start = hic->zeroVal();
  auto* stop = stream_axis->extent();
  auto* step = hic->oneVal();
  auto* for_loop_initial_sync = IrBuilder::create<ForLoop>(
      stream_axis,
      /*index=*/j,
      start,
      stop,
      step,
      /*vectorize=*/false,
      /*vectorize_shift=*/nullptr,
      /*unroll_required=*/false,
      CircularBufferLoopStage::NotApplicable,
      /*circular_buffer_loop_stage_depth=*/0);

  auto* number_of_streams =
      IrBuilder::create<NamedScalar>("numberOfStreams", DataType::Int);
  auto* stream_index = mod(j, number_of_streams);
  auto* stream = IrBuilder::create<hir::Stream>(stream_index);
  auto* set_stream = IrBuilder::create<hir::SetCurrentStream>(stream);
  auto* initial_sync_stream =
      IrBuilder::create<hir::Synchronize>(original_stream);

  // the initial sync of the streams with the user's stream is done in a
  // separate for-loop for performance reasons with comms/compute overlap
  std::vector<Expr*> loop_body_initial_sync = {set_stream, initial_sync_stream};
  for (Expr* expr : loop_body_initial_sync) {
    for_loop_initial_sync->body().push_back(expr);
  }

  auto* for_loop = IrBuilder::create<ForLoop>(
      stream_axis,
      /*index=*/j,
      start,
      stop,
      step,
      /*vectorize=*/false,
      /*vectorize_shift=*/nullptr,
      /*unroll_required=*/false,
      CircularBufferLoopStage::NotApplicable,
      /*circular_buffer_loop_stage_depth=*/0);

  TensorView* tva_j = select(tva, 0, j);
  TensorView* tva_allgathered_j = select(tva_allgathered, 0, j);
  TensorView* tv_out_j = select(tv_out, 0, j);

  NVF_ERROR(
      tva->hasDeviceMesh(),
      "The matmul's input ",
      tva,
      "is expected to have a DeviceMesh");
  for (auto tv : {tva_j, tva_allgathered_j, tv_out_j}) {
    tv->setDeviceMesh(tva->getDeviceMesh());
  }

  auto* communication = IrBuilder::create<Communication>(
      CommunicationType::Allgather,
      /*out=*/tva_allgathered_j,
      /*in=*/tva_j,
      /*team=*/tva->getDeviceMesh().vector(),
      /*root=*/-1,
      /*red_op=*/RedOpType::UNUSED,
      /*scattered_axis=*/-1,
      params_.communicator_backend);
  auto* wait = IrBuilder::create<hir::Wait>(communication);

  Expr* compute = nullptr;
  if (expr->isA<MatmulOp>()) {
    compute = IrBuilder::create<MatmulOp>(tv_out_j, tva_allgathered_j, tvb);
  } else {
    compute =
        IrBuilder::create<LinearOp>(tv_out_j, tva_allgathered_j, tvb, tv_bias);
  }

  auto* set_back_original_stream =
      IrBuilder::create<hir::SetCurrentStream>(original_stream);
  auto* sync_stream = IrBuilder::create<hir::Synchronize>(stream);

  std::vector<Expr*> loop_body = {
      set_stream,
      tva_j->definition(),
      tva_allgathered_j->definition(),
      communication,
      wait,
      tv_out_j->definition(),
      compute,
      set_back_original_stream,
      sync_stream};
  for (Expr* expr : loop_body) {
    for_loop->body().push_back(expr);
  }

  return {
      get_current_stream,
      allocate_tva_allgathered,
      allocate_tv_out,
      for_loop_initial_sync,
      for_loop};
}

std::unique_ptr<hir::HostIrContainer> HostIrLower::lower(
    std::unique_ptr<Fusion> fusion,
    DeviceIdxType my_device_index) {
  // Sharding PreSegmenter passes.
  // Note: passes run before PreSegmenter optimization passes.
  preseg_passes::OptimizationPass<
      preseg_passes::PropagateShardingsPass>::runPass(fusion.get());
  preseg_passes::OptimizationPass<
      preseg_passes::InsertReshardingsPass>::runPass(fusion.get());
  preseg_passes::OptimizationPass<
      preseg_passes::ReorderShardedAxisPass>::runPass(fusion.get());
  preseg_passes::OptimizationPass<
      preseg_passes::MakeReshardingContiguousPass>::runPass(fusion.get());

  // Performs segmentation at the inter-device communications
  // Each SegmentedGroup represents a pipeline's stage, and can be either
  // 1) a Fusion which doesn't involve inter-device communication
  // 2) a Fusion comprised of one Expr, representing inter-device communication
  SegmentCandidateFinderOptions options{
      .run_translate_welford = false,
      .run_combine_reductions = false,
      .run_herrmann_merge = true,
      .run_final_merge = true,
      .only_segment_resharding_exprs = true};
  std::unique_ptr<SegmentedFusion> staged_fusion =
      SegmentCandidateFinder::segment(
          std::move(fusion), KernelArgumentHolder(), options, true);
  // Infer a topologically ordered traversal of the segmented fusion to
  // determine the order for launching the kernels/comms
  RuntimeWorkSpace workspace;
  prepareRuntimeOrder(staged_fusion.get(), workspace);
  // Create the HostIrContainer representing the host program. Each segment of
  // the segmented fusion will be translated to a HostIR
  auto hic = std::make_unique<hir::HostIrContainer>();
  FusionGuard fg(hic.get());
  IrCloner ir_cloner(hic.get());
  auto clone =
      [&ir_cloner](const std::vector<Val*>& vals) -> std::vector<Val*> {
    std::vector<Val*> cloned_vals(vals.size());
    std::transform(
        vals.begin(), vals.end(), cloned_vals.begin(), [&ir_cloner](Val* val) {
          return ir_cloner.clone(val);
        });
    return cloned_vals;
  };

  for (auto group : workspace.group_run_order) {
    NVF_ERROR(!group->exprs().empty(), "invalid segmentation");
    if (involvedDevices(group->exprs().at(0)).count(my_device_index) == 0) {
      continue;
    }
    const bool is_resharding = std::any_of(
        group->exprs().begin(), group->exprs().end(), [](auto expr) {
          return isResharding(expr);
        });
    if (is_resharding) {
      NVF_ERROR(
          group->exprs().size() == 1,
          "Communication segments must contain only one Expr");
      for (auto* expr : HostIrLower::lower(
               ir_cloner.clone(group->exprs().at(0)), my_device_index)) {
        // Allocate the recv buffers of communications
        if (expr->isA<Communication>()) {
          auto* communication = expr->as<Communication>();
          TensorView* tv = communication->out();
          if (tv->getDeviceMesh().has(my_device_index)) {
            auto* allocate =
                IrBuilder::create<kir::Allocate>(tv, MemoryType::Global);
            hic->pushBackTopLevelExprs(allocate);
          }
        }
        hic->pushBackTopLevelExprs(expr);
        if (expr->isA<Communication>()) {
          auto wait = IrBuilder::create<hir::Wait>(expr->as<Communication>());
          hic->pushBackTopLevelExprs(wait);
        }
      }
    } else {
      auto host_unit = IrBuilder::create<hir::HostUnit>(
          staged_fusion->makeFusion(group).second);
      auto post_on_stream = IrBuilder::create<hir::PostOnStream>(
          host_unit, clone(group->inputs()), clone(group->outputs()));
      hic->pushBackTopLevelExprs(post_on_stream);
    }
  }
  for (auto input : staged_fusion->inputs()) {
    hic->addInput(ir_cloner.clone(input));
  }
  for (auto output : staged_fusion->outputs()) {
    hic->addOutput(ir_cloner.clone(output));
  }

  return hic;
}

} // namespace nvfuser
