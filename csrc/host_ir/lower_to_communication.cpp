// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on

#include <host_ir/container.h>
#include <host_ir/lower_to_communication.h>
#include <ir/all_nodes.h>
#include <ir/allocation_utils.h>
#include <ir/builder.h>
#include <ir/internal_base_nodes.h>
#include <kernel_ir.h>
#include <multidevice/communication.h>
#include <multidevice/utils.h>
#include <ops/all_ops.h>

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
  const DeviceMesh& receiver_mesh = output_tv->getDeviceMesh();
  NVF_ERROR(
      receiver_mesh.rank() == 1,
      "Gather only supported on a 1D mesh. Given ",
      receiver_mesh);

  // Find a common device between input and receiver meshes to be the root
  auto it = std::ranges::find_if(
      input_tv->getDeviceMesh().vector(),
      [&receiver_mesh](DeviceIdxType device) {
        return receiver_mesh.has(device);
      });
  NVF_ERROR(
      it != input_tv->getDeviceMesh().vector().end(),
      "No common device found between input and receiver meshes");
  DeviceIdxType root = *it;

  Team team = receiver_mesh.vector();
  comms.push_back(IrBuilder::create<Communication>(
      CommunicationType::Scatter,
      output_tv,
      input_tv,
      team,
      root,
      c10d::ReduceOp::RedOpType::UNUSED,
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

  comms.push_back(IrBuilder::create<Communication>(
      CommunicationType::ReduceScatter,
      output_tv,
      input_tv,
      /*team=*/team,
      /*root=*/-1,
      getC10dReduceOpType(op_type),
      params.communicator_backend));
}

IterDomain* getLogicalFromLoopId(TensorView* tv, IterDomain* loop_id) {
  std::vector<IterDomain*> logical_ids =
      getInputsInTargetDomain(loop_id, tv->getLogicalDomain());
  NVF_ERROR(
      logical_ids.size() == 1,
      "Expected exactly one logical ID producing the device dimension ",
      loop_id);
  return logical_ids.at(0);
}

bool isLocalSizeOne(IterDomain* id) {
  return id->isParallelized() || id->isBroadcast() || id->isReduction();
}

} // namespace

bool isAllocationOrderCompliant(TensorView* tv, IterDomain* sharded_id) {
  NVF_ERROR(
      std::find(
          tv->getLogicalDomain().begin(),
          tv->getLogicalDomain().end(),
          sharded_id) != tv->getLogicalDomain().end(),
      "The sharded ID ",
      sharded_id->toString(),
      " is not in the logical domain ",
      tv->getLogicalDomain());

  if (isLocalSizeOne(sharded_id)) {
    // Parallelized dimension, broadcast, and reduction do not affect
    // allocation.
    return true;
  }

  // This sharded logical ID may not be directly present in allocation domain.
  // This indicates allocation domain has DID transformations.
  std::optional<Layout> layout = canonicalizeLayout(tv);
  if (!layout.has_value()) {
    return false;
  }

  const std::vector<IterDomain*>& allocation_domain = layout->allocation_domain;

  NVF_ERROR(
      std::is_permutation(
          allocation_domain.begin(),
          allocation_domain.end(),
          tv->getLogicalDomain().begin(),
          tv->getLogicalDomain().end()),
      "The allocation domain returned by canonicalizeLayout",
      allocation_domain,
      " should be a permutation of the logical domain ",
      tv->getLogicalDomain());

  // Check if sharded_id appears at the front.
  for (IterDomain* id : allocation_domain) {
    if (id == sharded_id) {
      return true;
    }
    if (!isLocalSizeOne(id)) {
      return false;
    }
  }
  NVF_THROW(
      "Should never reach here - sharded_id must be found in allocation "
      "domain");
}

std::optional<CommunicationInfo> getCommunicationInfo(Expr* expr) {
  auto* producer = expr->inputs().at(0)->as<TensorView>();
  auto* consumer = expr->outputs().at(0)->as<TensorView>();
  bool has_sharding_change = false;
  std::optional<CommunicationInfo> communication_info = std::nullopt;

  auto create_communication_info = [&](CommunicationType type,
                                       IterDomain* p_sharded_id,
                                       IterDomain* c_sharded_id) {
    NVF_ERROR(!has_sharding_change, "Expected at most one sharding change");
    has_sharding_change = true;
    communication_info = CommunicationInfo{type, p_sharded_id, c_sharded_id};
  };

  const auto pairwise_map = PairwiseLogicalDomainMap(producer, consumer);
  const auto p2c_map = pairwise_map.mapProducerToConsumer();
  const auto c2p_map = pairwise_map.mapConsumerToProducer();

  // This ignores device dimensions on reduction axis.
  auto producer_pt_to_did =
      mapDeviceAndStreamParallelTypeToId(producer->getLoopDomain());
  auto consumer_pt_to_did =
      mapDeviceAndStreamParallelTypeToId(consumer->getLoopDomain());

  for (ParallelType pt : kParallelTypeDIDs) {
    IterDomain* p_loop_did = getOrDefault(producer_pt_to_did, pt);
    IterDomain* c_loop_did = getOrDefault(consumer_pt_to_did, pt);

    if (p_loop_did == nullptr && c_loop_did == nullptr) {
      // Not sharded on this parallel type
      continue;
    }

    const DeviceMesh& producer_mesh = producer->getDeviceMesh();
    const DeviceMesh& consumer_mesh = consumer->getDeviceMesh();
    const bool p_sharded = p_loop_did != nullptr && producer_mesh.size() > 1;
    const bool c_sharded = c_loop_did != nullptr && consumer_mesh.size() > 1;
    const bool same_mesh = producer_mesh == consumer_mesh;

    if (expr->isA<LoadStoreOp>()) {
      if (p_sharded && !c_sharded) {
        IterDomain* p_logical_id = getLogicalFromLoopId(producer, p_loop_did);
        CommunicationType type = same_mesh ? CommunicationType::Allgather
                                           : CommunicationType::Gather;
        create_communication_info(type, p_logical_id, p2c_map.at(p_logical_id));
        continue;
      }
      if (!p_sharded && c_sharded) {
        // Scatter
        IterDomain* c_logical_id = getLogicalFromLoopId(consumer, c_loop_did);
        create_communication_info(
            CommunicationType::Scatter, c2p_map.at(c_logical_id), c_logical_id);
        continue;
      }
    } else if (expr->isA<ReductionOp>()) {
      if (!p_sharded) {
        // Not a reduction based communication.
        continue;
      }

      if (!c_sharded) {
        IterDomain* p_logical_id = getLogicalFromLoopId(producer, p_loop_did);
        CommunicationType type = same_mesh ? CommunicationType::Allreduce
                                           : CommunicationType::Reduce;
        create_communication_info(type, p_logical_id, p2c_map.at(p_logical_id));
        continue;
      }

      // Check if the p_logical_ids is reduced in the output.
      IterDomain* p_logical_id = getLogicalFromLoopId(producer, p_loop_did);
      IterDomain* c_logical_id = getLogicalFromLoopId(consumer, c_loop_did);

      auto c_it = p2c_map.find(p_logical_id);
      NVF_ERROR(
          c_it != p2c_map.end(),
          "Cannot find the mapped consumer logical ID for the producer logical "
          "ID ",
          p_logical_id->toString());
      if (!c_it->second->isReduction()) {
        continue;
      }
      create_communication_info(
          CommunicationType::ReduceScatter,
          c2p_map.at(c_logical_id),
          c_logical_id);
    } else {
      NVF_THROW("Unsupported expression: ", expr->toString());
    }
  }
  return communication_info;
}

bool isCommunicationLayoutCompliant(Expr* expr) {
  TensorView* producer = expr->inputs().at(0)->as<TensorView>();
  TensorView* consumer = expr->outputs().at(0)->as<TensorView>();

  auto communication_info = getCommunicationInfo(expr);
  if (!communication_info.has_value()) {
    return true;
  }

  if (!isTvContiguous(producer) || !isTvContiguous(consumer)) {
    return false;
  }

  if (communication_info->type == CommunicationType::Reduce ||
      communication_info->type == CommunicationType::Allreduce) {
    // Reduction axis in reduce/allreduce does not have to be outermost in
    // allocation domain.
    return true;
  }

  // Check if the gather/scatter axis is outermost in memory layout.
  if (!isAllocationOrderCompliant(producer, communication_info->p_sharded_id) ||
      !isAllocationOrderCompliant(consumer, communication_info->c_sharded_id)) {
    return false;
  }

  return true;
}

std::vector<Expr*> convertSingleOpToCommunication(
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
      isCommunicationLayoutCompliant(c),
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
          "ReduceScatter operation must have the same sender and receiver "
          "device mesh. "
          "Insert a Set operation before or after the reduction to reshard ot "
          "another device mesh");
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

} // namespace nvfuser
