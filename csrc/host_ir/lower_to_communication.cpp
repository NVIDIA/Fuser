// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on

#include "host_ir/lower_to_communication.h"

#include "ir/builder.h"
#include "ir/interface_nodes.h"
#include "ir/internal_base_nodes.h"
#include "ir/iostream.h"
#include "logical_domain_map.h"
#include "multidevice/communication.h"
#include "multidevice/resharding.h"
#include "multidevice/utils.h"

namespace nvfuser {

namespace {

// TODO: handle `c10d::RedOpType::reduceOp::AVG` and
// `c10d::RedOpType::reduceOp::PREMUL_SUM`
c10d::ReduceOp::RedOpType getC10dReduceOpType(BinaryOpType op) {
  switch (op) {
    case BinaryOpType::Add:
      return c10d::ReduceOp::RedOpType::SUM;
    case BinaryOpType::Mul:
      return c10d::ReduceOp::RedOpType::PRODUCT;
    case BinaryOpType::FMin:
    case BinaryOpType::Min:
      return c10d::ReduceOp::RedOpType::MIN;
    case BinaryOpType::FMax:
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
  }
}

// Adds one or zero Scatter communication to the vector 'comms'
void lowerToScatter(
    TensorView* input_tv,
    TensorView* output_tv,
    const CommunicatorBackend backend,
    std::vector<Expr*>& comms) {
  const DeviceMesh& receiver_mesh = output_tv->getDeviceMesh();
  NVF_ERROR_EQ(
      receiver_mesh.rank(),
      1,
      "Gather only supported on a 1D mesh. Given ",
      output_tv->toString());

  // Find a common device between input and receiver meshes to be the root
  std::vector<DeviceIdxType> input_devices = input_tv->getDeviceMesh().vector();
  auto it = std::ranges::find_if(
      input_devices, [&receiver_mesh](DeviceIdxType device) {
        return receiver_mesh.has(device);
      });
  NVF_ERROR(
      it != input_devices.end(),
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
      backend));
}

// Adds zero or multiple Gather communications to the vector 'comms'
//
// Note that since the root of a Gather collective is a destination, we possibly
// need multiple Gathers if the tensor is replicated in the receiver mesh.
void lowerToGather(
    TensorView* input_tv,
    TensorView* output_tv,
    const CommunicatorBackend backend,
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
        backend));
  }
}

// Add one or zero Allgather communication to the vector 'comms'
void lowerToAllgather(
    TensorView* input_tv,
    TensorView* output_tv,
    const CommunicatorBackend backend,
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
      backend));
}

// Adds one or zero Broadcast communication to the vector 'comms'
void lowerToBroadcast(
    TensorView* input_tv,
    TensorView* output_tv,
    const CommunicatorBackend backend,
    std::vector<Expr*>& comms) {
  // Either of the following two cases is happening.
  // 1. `sender_mesh` contains only one device. In this case, we broadcast
  // from that device.
  // 2. `sender_mesh` contains multiple devices but the input is not sharded.
  // In this case, we arbitrarily choose the first device of the sender mesh
  // to be the root.
  const DeviceMesh& sender_mesh = input_tv->getDeviceMesh();
  const DeviceMesh& receiver_mesh = output_tv->getDeviceMesh();

  NVF_ERROR_EQ(sender_mesh.rank(), 1, "sender: ", input_tv->toString());
  NVF_ERROR_EQ(receiver_mesh.rank(), 1, "receiver: ", output_tv->toString());

  DeviceIdxType root = sender_mesh.at(0);
  Team team = receiver_mesh.vector();
  if (!receiver_mesh.has(root)) {
    team.push_back(root);
  }
  comms.push_back(IrBuilder::create<Communication>(
      CommunicationType::Broadcast,
      output_tv,
      input_tv,
      team,
      root,
      c10d::ReduceOp::RedOpType::UNUSED,
      backend));
}

// Adds several SendRecv communications to the vector 'comms'
// For now, we assume that this function is called only if
// the input and output have the same sharding. Later we could support more
// general cases.
void lowerToSendRecv(
    TensorView* input_tv,
    TensorView* output_tv,
    const CommunicatorBackend backend,
    std::vector<Expr*>& comms) {
  const DeviceMesh& sender_mesh = input_tv->getDeviceMesh();
  const DeviceMesh& receiver_mesh = output_tv->getDeviceMesh();
  NVF_ERROR_EQ(
      sender_mesh.rank(),
      1,
      "SendRecv only supports a 1D sender mesh. Given ",
      sender_mesh);
  NVF_ERROR_EQ(
      receiver_mesh.rank(),
      1,
      "SendRecv only supports a 1D receiver mesh. Given ",
      receiver_mesh);
  NVF_ERROR_EQ(
      sender_mesh.size(),
      receiver_mesh.size(),
      "Receiver and sender meshes have different sizes.");
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
        backend));
  }
}

void lowerToReduce(
    TensorView* input_tv,
    TensorView* output_tv,
    BinaryOpType op_type,
    const CommunicatorBackend backend,
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
        backend));
  }
}

void lowerToAllreduce(
    TensorView* input_tv,
    TensorView* output_tv,
    BinaryOpType op_type,
    const CommunicatorBackend backend,
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
      backend));
}

void lowerToReduceScatter(
    TensorView* input_tv,
    TensorView* output_tv,
    BinaryOpType op_type,
    const CommunicatorBackend backend,
    std::vector<Expr*>& comms,
    DeviceIdxType my_device_idx) {
  NVF_ERROR_EQ(
      input_tv->getDeviceMesh(),
      output_tv->getDeviceMesh(),
      "ReduceScatter operation must have the same sender and receiver "
      "device mesh. "
      "Insert a Set operation before or after the reduction to reshard to "
      "another device mesh");
  const DeviceMesh& mesh = input_tv->getDeviceMesh();
  Team team = mesh.getSlice(my_device_idx, ParallelType::DIDx);

  comms.push_back(IrBuilder::create<Communication>(
      CommunicationType::ReduceScatter,
      output_tv,
      input_tv,
      /*team=*/team,
      /*root=*/-1,
      getC10dReduceOpType(op_type),
      backend));
}

void lowerToAllToAll(
    TensorView* input_tv,
    TensorView* output_tv,
    const CommunicatorBackend backend,
    std::vector<Expr*>& comms) {
  const DeviceMesh& sender_mesh = input_tv->getDeviceMesh();
  const DeviceMesh& receiver_mesh = output_tv->getDeviceMesh();
  NVF_ERROR_EQ(
      sender_mesh.rank(),
      1,
      "AllToAll sender mesh must be a 1D mesh. Given ",
      sender_mesh);
  NVF_ERROR_EQ(
      receiver_mesh.rank(),
      1,
      "AllToAll receiver mesh must be a 1D mesh. Given ",
      receiver_mesh);
  NVF_ERROR_EQ(
      sender_mesh,
      receiver_mesh,
      "AllToAll sender and receiver meshes must be the same. Given ",
      sender_mesh,
      " and ",
      receiver_mesh);
  comms.push_back(IrBuilder::create<Communication>(
      CommunicationType::AllToAll,
      output_tv,
      input_tv,
      sender_mesh.vector(),
      /*root=*/-1,
      c10d::ReduceOp::RedOpType::UNUSED,
      backend));
}

IterDomain* getLogicalFromLoopId(TensorView* tv, IterDomain* loop_id) {
  std::unordered_set<IterDomain*> logical_ids =
      getInputsInTargetDomain({loop_id}, tv->getLogicalDomain());
  NVF_ERROR(
      logical_ids.size() == 1,
      "Expected exactly one logical ID producing the device dimension ",
      loop_id);
  return *logical_ids.begin();
}

bool isLocalSizeOne(IterDomain* id) {
  return id->isParallelized() || id->isBroadcast() || id->isReduction();
}

} // namespace

CommunicationInfo getCommunicationInfo(Expr* e) {
  NVF_ERROR(
      isResharding(e),
      "getCommunicationInfo should only be called when `e` is known to be a "
      "communication. So `e` should be resharding. Given: ",
      e);

  // `sum` leads to a SqueezeOp when the reduction dimension is size-1.
  NVF_ERROR(
      (e->isOneOf<LoadStoreOp, ReductionOp, SqueezeOp>()),
      "getCommunicationInfo should only be called when `e` is known to be a "
      "communication. Given: ",
      e);
  NVF_ERROR_EQ(
      e->inputs().size(), 1, "Expected 1 input, but got ", e->toString());
  auto* producer = e->inputs().at(0)->as<TensorView>();
  NVF_ERROR_EQ(
      e->outputs().size(), 1, "Expected 1 output, but got ", e->toString());
  auto* consumer = e->outputs().at(0)->as<TensorView>();

  std::optional<CommunicationInfo> communication_info = std::nullopt;
  // Fill `communication_info` instead of returning the result, so we can
  // catch errors when more than one DIDs have sharding changes.
  auto fill_communication_info = [&](CommunicationType type,
                                     IterDomain* p_sharded_id,
                                     IterDomain* c_sharded_id) {
    NVF_ERROR(
        !communication_info.has_value(),
        "Expected at most one sharding change: ",
        e->toString());
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

  const DeviceMesh& producer_mesh = producer->getDeviceMesh();
  const DeviceMesh& consumer_mesh = consumer->getDeviceMesh();
  const bool same_mesh = producer_mesh == consumer_mesh;

  for (ParallelType pt : kParallelTypeDIDs) {
    if (!haveDifferentShardings(producer, consumer, {pt})) {
      continue;
    }

    IterDomain* p_loop_did = getOrDefault(producer_pt_to_did, pt);
    IterDomain* c_loop_did = getOrDefault(consumer_pt_to_did, pt);

    if (p_loop_did == nullptr && c_loop_did == nullptr) {
      // Not sharded on this parallel type
      NVF_THROW("Not sharded on this parallel type: ", pt);
    }

    if (e->isA<LoadStoreOp>()) {
      if (p_loop_did && !c_loop_did) {
        IterDomain* p_logical_id = getLogicalFromLoopId(producer, p_loop_did);
        CommunicationType type = same_mesh ? CommunicationType::Allgather
                                           : CommunicationType::Gather;
        fill_communication_info(type, p_logical_id, p2c_map.at(p_logical_id));
      }
      if (!p_loop_did && c_loop_did) {
        IterDomain* c_logical_id = getLogicalFromLoopId(consumer, c_loop_did);
        fill_communication_info(
            CommunicationType::Scatter, c2p_map.at(c_logical_id), c_logical_id);
      }
      if (p_loop_did && c_loop_did) {
        IterDomain* p_logical_id = getLogicalFromLoopId(producer, p_loop_did);
        IterDomain* c_logical_id = getLogicalFromLoopId(consumer, c_loop_did);
        // TODO(#4604): This is problematic for 2D sharding.

        if (c_logical_id == p2c_map.at(p_logical_id)) {
          fill_communication_info(
              CommunicationType::SendRecv, p_logical_id, c_logical_id);
        } else {
          fill_communication_info(
              CommunicationType::AllToAll, p_logical_id, c_logical_id);
        }
      }
    } else {
      NVF_ERROR(e->isA<ReductionOp>() || e->isA<SqueezeOp>());
      if (!p_loop_did) {
        // Not a reduction based communication.
        continue;
      }

      if (!c_loop_did) {
        IterDomain* p_logical_id = getLogicalFromLoopId(producer, p_loop_did);
        CommunicationType type = same_mesh ? CommunicationType::Allreduce
                                           : CommunicationType::Reduce;
        fill_communication_info(type, p_logical_id, p2c_map.at(p_logical_id));
        continue;
      }

      // Check if the p_logical_ids is reduced in the output.
      IterDomain* p_logical_id = getLogicalFromLoopId(producer, p_loop_did);
      IterDomain* c_logical_id = getLogicalFromLoopId(consumer, c_loop_did);

      auto c_it = p2c_map.find(p_logical_id);
      NVF_ERROR(
          c_it != p2c_map.end(),
          "Cannot find the mapped consumer logical ID for the producer "
          "logical "
          "ID ",
          p_logical_id->toString());
      if (!c_it->second->isReduction()) {
        continue;
      }
      fill_communication_info(
          CommunicationType::ReduceScatter,
          c2p_map.at(c_logical_id),
          c_logical_id);
    }
  }

  if (!communication_info.has_value()) {
    fill_communication_info(CommunicationType::Broadcast, nullptr, nullptr);
  }
  return *communication_info;
}

namespace {
int64_t posInDomain(const std::vector<IterDomain*>& domain, IterDomain* id) {
  auto pos = std::find(domain.begin(), domain.end(), id);
  if (pos == domain.end()) {
    return -1;
  }
  return std::distance(domain.begin(), pos);
}
} // namespace

Layout getCommunicationLayout(
    TensorView* tv,
    const CommunicationType type,
    IterDomain* sharded_id) {
  const Layout layout = canonicalizeLayout(tv)->contiguous();
  // For the following communication types, the sharded_id does not have to be
  // outermost in allocation domain. Nonetheless, `tv` still needs to be
  // contiguous and therefore .contiguous() at the beginning of this function.
  // Note: We do not yet reorder for AllToAll and only support cases where the
  // input and output do not require any reordering.
  if (type == CommunicationType::Reduce ||
      type == CommunicationType::Allreduce ||
      type == CommunicationType::Broadcast ||
      type == CommunicationType::SendRecv ||
      type == CommunicationType::AllToAll) {
    return layout;
  }

  const int64_t sharded_id_pos =
      posInDomain(layout.allocation_domain(), sharded_id);
  NVF_ERROR(
      sharded_id_pos >= 0,
      "Sharded ID (",
      sharded_id,
      ") not found in the allocation domain of the tensor view: ",
      tv);

  if (isLocalSizeOne(sharded_id)) {
    // Parallelized dimension, broadcast, and reduction do not affect
    // allocation.
    return layout;
  }

  for (int64_t i : arange(sharded_id_pos)) {
    IterDomain* id = layout.allocation_domain(i);
    if (!isLocalSizeOne(id)) {
      // We could put `sharded_id` to any position between 0 and i. I chose 0
      // for simplicity.
      std::vector<IterDomain*> new_allocation = TensorDomain::orderedAs(
          layout.allocation_domain(), {{sharded_id_pos, 0}});
      return Layout{
          new_allocation,
          TensorDomain::getContiguityFilledWith(new_allocation, true)};
    }
  }
  return layout;
}

bool isCommunicationLayoutCompliant(Expr* expr) {
  auto* producer = expr->inputs().at(0)->as<TensorView>();
  auto* consumer = expr->outputs().at(0)->as<TensorView>();

  CommunicationInfo communication_info = getCommunicationInfo(expr);

  Layout p_layout = getCommunicationLayout(
      producer, communication_info.type, communication_info.p_sharded_id);
  if (!isCompliantWith(*canonicalizeLayout(producer), p_layout)) {
    return false;
  }

  Layout c_layout = getCommunicationLayout(
      consumer, communication_info.type, communication_info.c_sharded_id);
  if (!isCompliantWith(*canonicalizeLayout(consumer), c_layout)) {
    return false;
  }

  return true;
}

std::vector<Expr*> convertSingleOpToCommunication(
    Expr* e,
    DeviceIdxType my_device_idx,
    const CommunicatorBackend backend) {
  FusionGuard fg(e->fusion());

  std::vector<Expr*> comms;
  NVF_ERROR(
      e->inputs().size() == 1 && e->input(0)->isA<TensorView>() &&
          e->outputs().size() == 1 && e->output(0)->isA<TensorView>(),
      "Input/Output must be single TensorView: ",
      e);
  auto* input_tv = e->input(0)->as<TensorView>();
  auto* output_tv = e->output(0)->as<TensorView>();

  if (input_tv->getMemoryType() != MemoryType::Symmetric) {
    input_tv->setMemoryType(MemoryType::Global);
  }
  if (output_tv->getMemoryType() != MemoryType::Symmetric) {
    output_tv->setMemoryType(MemoryType::Global);
  }

  NVF_ERROR(
      isCommunicationLayoutCompliant(e),
      "Resharding on an inner axis is not lowerable ",
      e->toString());

  CommunicationInfo communication_info = getCommunicationInfo(e);

  auto op_type = [](Expr* e) -> BinaryOpType {
    if (auto* reduce = dynamic_cast<ReductionOp*>(e)) {
      return reduce->getReductionOpType();
    }

    NVF_ERROR(e != nullptr);
    if (e->isA<SqueezeOp>()) {
      return BinaryOpType::Add;
    }

    NVF_THROW("Expected a ReductionOp or a SqueezeOp, but got: ", e);
  };

  switch (communication_info.type) {
    case CommunicationType::Scatter:
      lowerToScatter(input_tv, output_tv, backend, comms);
      break;
    case CommunicationType::Gather:
      lowerToGather(input_tv, output_tv, backend, comms);
      break;
    case CommunicationType::Allgather:
      lowerToAllgather(input_tv, output_tv, backend, comms, my_device_idx);
      break;
    case CommunicationType::Broadcast:
      lowerToBroadcast(input_tv, output_tv, backend, comms);
      break;
    case CommunicationType::SendRecv:
      lowerToSendRecv(input_tv, output_tv, backend, comms);
      break;
    case CommunicationType::ReduceScatter:
      lowerToReduceScatter(
          input_tv, output_tv, op_type(e), backend, comms, my_device_idx);
      break;
    case CommunicationType::Allreduce:
      lowerToAllreduce(
          input_tv, output_tv, op_type(e), backend, comms, my_device_idx);
      break;
    case CommunicationType::Reduce:
      lowerToReduce(input_tv, output_tv, op_type(e), backend, comms);
      break;
    case CommunicationType::AllToAll:
      lowerToAllToAll(input_tv, output_tv, backend, comms);
      break;
  }

  return comms;
}

} // namespace nvfuser
