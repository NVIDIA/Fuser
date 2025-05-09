// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on

#include <host_ir/container.h>
#include <host_ir/lower.h>
#include <host_ir/pass/convert_op_to_communication.h>
#include <id_model/id_model.h>
#include <ir/all_nodes.h>
#include <ir/builder.h>
#include <ir/internal_base_nodes.h>
#include <ir/utils.h>
#include <kernel_ir.h>
#include <multidevice/communication.h>
#include <multidevice/utils.h>
#include <ops/all_ops.h>
#include <ops/utils.h>

namespace nvfuser::hir_pass {

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
    for (auto i : c10::irange(sender_mesh.size())) {
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
std::vector<Expr*> ConvertOpToCommunication::ConvertSingleOpToCommunication(
    Expr* c,
    DeviceIdxType my_device_idx,
    const HostIrLowerParams& params) {
  FusionGuard fg(c->fusion());

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
          input_tv, output_tv, op_type, params, comms, my_device_idx);
    } else {
      if (same_mesh) {
        lowerToAllreduce(
            input_tv, output_tv, op_type, params, comms, my_device_idx);
      } else {
        lowerToReduce(input_tv, output_tv, op_type, params, comms);
      }
    }
  } else {
    if (!is_input_sharded && is_output_sharded) {
      lowerToScatter(input_tv, output_tv, params, comms);
    } else if (is_input_sharded && !is_output_sharded) {
      if (same_mesh) {
        lowerToAllgather(input_tv, output_tv, params, comms, my_device_idx);
      } else {
        lowerToGather(input_tv, output_tv, params, comms);
      }
    } else {
      lowerToBroadcastOrSendRecv(input_tv, output_tv, params, comms);
    }
  }

  return comms;
}

void ConvertOpToCommunication::passImplementation(Fusion* fusion) {
  FusionGuard fg(fusion);
  hir::HostIrContainer* hic = dynamic_cast<hir::HostIrContainer*>(fusion);
  NVF_CHECK(hic, "Expected HostIrContainer");
  DeviceIdxType my_device_index = Communicator::getInstance().deviceId();

  auto handle_top_level_expr = [&](Expr* top_level_expr,
                                   std::vector<Expr*>& new_top_level_exprs) {
    if (!isResharding(top_level_expr)) {
      return new_top_level_exprs.push_back(top_level_expr);
    }
    for (auto* expr : ConvertSingleOpToCommunication(
             top_level_expr, my_device_index, params_)) {
      // Allocate the recv buffers of communications
      if (expr->isA<Communication>()) {
        auto* communication = expr->as<Communication>();
        TensorView* tv = communication->out();
        if (tv->getDeviceMesh().has(my_device_index) &&
            hic->alias().count(tv) == 0) {
          auto* allocate =
              IrBuilder::create<kir::Allocate>(tv, MemoryType::Global);
          new_top_level_exprs.push_back(allocate);
        }
      }
      new_top_level_exprs.push_back(expr);
      if (expr->isA<Communication>()) {
        auto wait = IrBuilder::create<hir::Wait>(expr->as<Communication>());
        new_top_level_exprs.push_back(wait);
      }
    }
  };

  std::vector<Expr*> new_top_level_exprs;
  for (auto top_level_expr : hic->topLevelExprs()) {
    if (top_level_expr->isA<ForLoop>()) {
      auto* for_loop = top_level_expr->as<ForLoop>();
      std::vector<Expr*> new_for_loop_body;
      for (auto* expr : for_loop->body().exprs()) {
        handle_top_level_expr(expr, new_for_loop_body);
      }
      for_loop->body().clear();
      for (auto* expr : new_for_loop_body) {
        for_loop->body().push_back(expr);
      }
      new_top_level_exprs.push_back(for_loop);
    } else {
      handle_top_level_expr(top_level_expr, new_top_level_exprs);
    }
  }
  hic->resetTopLevelExprs(new_top_level_exprs);
}

} // namespace nvfuser::hir_pass
