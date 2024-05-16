// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#include <device_lower/utils.h>
#include <ir/builder.h>
#include <ir/interface_nodes.h>
#include <multidevice/device_mesh.h>
#include <multidevice/lower_communication.h>
#include <multidevice/utils.h>
#include <ops/all_ops.h>
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
      NVF_ERROR(false, "unsupported reduction operation");
      return c10d::ReduceOp::RedOpType::UNUSED;
  }
}

inline bool isDeviceInvolved(
    DeviceIdxType my_device_index,
    DeviceIdxType root,
    const DeviceMesh& mesh) {
  return my_device_index == root || mesh.has(my_device_index);
}

inline bool isDeviceInvolved(
    DeviceIdxType my_device_index,
    const DeviceMesh& sender_mesh,
    const DeviceMesh& receiver_mesh) {
  return sender_mesh.has(my_device_index) || receiver_mesh.has(my_device_index);
}

// Utility function used for setting up a scatter or gather communication
// params. Since most  of the steps are somewhat similar/opposite in those
// cases, we gathered the two implementations into one function. The argument
// "is_scatter" allows to discriminate between scatter and gather
CommParams createParamsForGatherScatter(
    DeviceIdxType my_device_index,
    DeviceIdxType root,
    TensorView* root_tv, // is_scatter ? input_tv : output_tv
    bool is_scatter) {
  const DeviceMesh& mesh = root_tv->getDeviceMesh();
  CommParams params;
  params.type =
      is_scatter ? CommunicationType::Scatter : CommunicationType::Gather;
  params.root = root;
  params.team = mesh.getTeam(my_device_index, 0);
  if (!mesh.has(root)) {
    params.is_root_in_mesh = false;
    params.team.push_back(root);
  }
  return params;
}

// Adds one or zero Scatter communication to the vector 'comms'
void lowerToScatter(
    DeviceIdxType my_device_index,
    TensorView* input_tv,
    TensorView* output_tv,
    std::vector<Communication*>& comms) {
  // we arbitrarily choose the first device of the sender mesh to be the root
  const DeviceMesh& receiver_mesh = output_tv->getDeviceMesh();
  auto root = input_tv->getDeviceMesh().at(0);
  if (!isDeviceInvolved(my_device_index, root, receiver_mesh)) {
    return;
  }
  auto params =
      createParamsForGatherScatter(my_device_index, root, output_tv, true);
  comms.push_back(IrBuilder::create<Communication>(std::move(params)));
}

/*
Adds zero or multiple Gather communications to the vector 'comms'

Note that since the root of a Gather collective is a destination, we possibly
need multiple Gather if the tensor is replicated in the receiver mesh.
*/
void lowerToGather(
    DeviceIdxType my_device_index,
    TensorView* input_tv,
    TensorView* output_tv,
    std::vector<Communication*>& comms) {
  // we create as many 'Gathers' as there are devices in the receiver mesh
  const DeviceMesh& sender_mesh = input_tv->getDeviceMesh();
  for (auto root : output_tv->getDeviceMesh().getTeam(my_device_index)) {
    if (!isDeviceInvolved(my_device_index, root, sender_mesh)) {
      continue;
    }
    auto params =
        createParamsForGatherScatter(my_device_index, root, input_tv, false);
    comms.push_back(IrBuilder::create<Communication>(std::move(params)));
  }
}

// Add one or zero Allgather communication to the vector 'comms'
void lowerToAllgather(
    DeviceIdxType my_device_index,
    TensorView* input_tv,
    TensorView* output_tv,
    std::vector<Communication*>& comms) {
  const DeviceMesh& mesh = input_tv->getDeviceMesh();
  if (!mesh.has(my_device_index)) {
    return;
  }

  CommParams params;
  params.type = CommunicationType::Allgather;
  params.team = mesh.getTeam(my_device_index, 0);
  comms.push_back(IrBuilder::create<Communication>(std::move(params)));
}

// Creates and set the CommParams for a Broadcast or Send/Recv communication
CommParams createParamsForBroadcastOrP2P(
    DeviceIdxType my_device_index,
    DeviceIdxType root,
    // receiver devices
    const DeviceMesh& mesh) {
  CommParams params;
  params.root = root;
  params.team = mesh.getTeam(my_device_index, 0);
  if (!mesh.has(root)) {
    params.is_root_in_mesh = false;
    params.team.push_back(root);
  }
  params.type = CommunicationType::Broadcast;

  return params;
}

// Adds one or zero Broadcast or Send/Recv communication to the vector 'comms'
void lowerToBroadcastOrP2P(
    DeviceIdxType my_device_index,
    DeviceIdxType root,
    const DeviceMesh& mesh, // receiver devices
    std::vector<Communication*>& comms) {
  if (!isDeviceInvolved(my_device_index, root, mesh)) {
    return;
  }
  auto params = createParamsForBroadcastOrP2P(my_device_index, root, mesh);
  comms.push_back(IrBuilder::create<Communication>(std::move(params)));
}

// Adds several Broadcast or Send/Recv communications to the vector 'comms'
// For now, we assume that this function is called only if
// the input and output have the same sharding. Later we could support more
// general cases.
void lowerToBroadcastOrP2P(
    DeviceIdxType my_device_index,
    TensorView* input_tv,
    TensorView* output_tv,
    bool is_sharded,
    std::vector<Communication*>& comms) {
  const DeviceMesh& sender_mesh = input_tv->getDeviceMesh();
  const DeviceMesh& receiver_mesh = output_tv->getDeviceMesh();
  if (is_sharded) {
    // if the inputs and ouputs are parallelized,
    // we create as many Broadcast as that will be handled in parallel
    for (auto i : c10::irange(sender_mesh.size())) {
      NVF_ERROR(
          sender_mesh.size() == receiver_mesh.size(),
          "the receiver and sender meshes have different sizes");
      lowerToBroadcastOrP2P(
          my_device_index,
          sender_mesh.at(i),
          DeviceMesh({receiver_mesh.at(i)}),
          comms);
    }
  } else {
    // we arbitrarily choose the first device of the sender mesh to be the root
    lowerToBroadcastOrP2P(
        my_device_index, sender_mesh.at(0), receiver_mesh, comms);
  }
}

CommParams createParamsForReduce(
    DeviceIdxType my_device_index,
    DeviceIdxType root,
    TensorView* input_tv,
    TensorView* output_tv,
    BinaryOpType op_type) {
  const DeviceMesh& mesh = input_tv->getDeviceMesh();
  CommParams params;
  params.type = CommunicationType::Reduce;
  params.root = root;
  params.redOp = getC10dReduceOpType(op_type);
  params.team = mesh.getTeam(my_device_index, 0);
  if (!mesh.has(root)) {
    params.is_root_in_mesh = false;
    params.team.push_back(root);
  }
  // FIXME: we may want to store sharded_dim to params for speed.
  return params;
}

void lowerToReduce(
    DeviceIdxType my_device_index,
    TensorView* input_tv,
    TensorView* output_tv,
    BinaryOpType op_type,
    std::vector<Communication*>& comms) {
  const DeviceMesh& receiver_mesh = output_tv->getDeviceMesh();
  const DeviceMesh& sender_mesh = input_tv->getDeviceMesh();
  // we create as many Reduces as there are devices in the receiver mesh
  for (auto root : receiver_mesh.getTeam(my_device_index)) {
    if (!isDeviceInvolved(my_device_index, root, sender_mesh)) {
      continue;
    }
    auto params = createParamsForReduce(
        my_device_index, root, input_tv, output_tv, op_type);
    comms.push_back(IrBuilder::create<Communication>(std::move(params)));
  }
}

void lowerToAllreduce(
    DeviceIdxType my_device_index,
    TensorView* input_tv,
    TensorView* output_tv,
    BinaryOpType op_type,
    std::vector<Communication*>& comms) {
  const DeviceMesh& mesh = input_tv->getDeviceMesh();
  if (!mesh.has(my_device_index)) {
    return;
  }

  CommParams params;
  params.type = CommunicationType::Allreduce;
  params.redOp = getC10dReduceOpType(op_type);
  params.team = mesh.getTeam(my_device_index, 0);
  comms.push_back(IrBuilder::create<Communication>(params));
}

void lowerToReduceScatter(
    DeviceIdxType my_device_index,
    TensorView* input_tv,
    TensorView* output_tv,
    BinaryOpType op_type,
    std::vector<Communication*>& comms) {
  const DeviceMesh& mesh = input_tv->getDeviceMesh();
  if (!mesh.has(my_device_index)) {
    return;
  }

  CommParams params;
  params.type = CommunicationType::ReduceScatter;
  params.redOp = getC10dReduceOpType(op_type);
  params.team = mesh.getTeam(my_device_index, 0);
  auto reduction_axis = output_tv->getReductionAxis().value();
  auto scattered_axis = getShardedAxis(output_tv);
  // The output tensor is sharded on scattered_axis and needs to be mapped
  // back onto the input. The input has an reduced axis, so the scattered axis
  // is adjusted to account for this. Ex: [DIDx(i0), i1] -> [r0, DIDx(i1)] The
  // scattered_axis is axis=0 on the output and maps to axis=1 on the input.
  if (reduction_axis <= scattered_axis) {
    scattered_axis++;
  }
  params.scattered_axis = scattered_axis;

  comms.push_back(IrBuilder::create<Communication>(params));
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
std::vector<Communication*> lowerCommunication(
    DeviceIdxType my_device_index,
    Expr* c) {
  std::vector<Communication*> comms;
  NVF_ERROR(
      c->inputs().size() == 1 && c->inputs().at(0)->isA<TensorView>() &&
          c->outputs().size() == 1 && c->outputs().at(0)->isA<TensorView>(),
      "I/O must be TensorViews");
  TensorView* input_tv = c->inputs().at(0)->as<TensorView>();
  TensorView* output_tv = c->outputs().at(0)->as<TensorView>();
  at::Tensor dummy;

  const DeviceMesh& sender_mesh = input_tv->getDeviceMesh();
  const DeviceMesh& receiver_mesh = output_tv->getDeviceMesh();
  const bool same_mesh = sender_mesh == receiver_mesh;

  // Stores whether the I/O has its first axis parallelized on Didx
  const bool is_input_sharded = isSharded(input_tv) && sender_mesh.size() > 1;
  const bool is_output_sharded =
      isSharded(output_tv) && receiver_mesh.size() > 1;

  auto original_expr = output_tv->definition();
  NVF_ERROR(
      isLowerableToCommunication(original_expr),
      "Lowering expression ",
      original_expr->toString(),
      " to communication is not supported");
  NVF_ERROR(
      !isInnerResharding(original_expr),
      "Resharding on an inner axis is not lowerable ",
      original_expr->toString());
  bool is_reduction = original_expr->isA<ReductionOp>();

  if (is_reduction) {
    BinaryOpType op_type =
        output_tv->definition()->as<ReductionOp>()->getReductionOpType();
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
          my_device_index, input_tv, output_tv, op_type, comms);
    } else {
      if (same_mesh) {
        lowerToAllreduce(my_device_index, input_tv, output_tv, op_type, comms);
      } else {
        lowerToReduce(my_device_index, input_tv, output_tv, op_type, comms);
      }
    }
  } else {
    if (!is_input_sharded && is_output_sharded) {
      lowerToScatter(my_device_index, input_tv, output_tv, comms);
    } else if (is_input_sharded && !is_output_sharded) {
      if (same_mesh) {
        lowerToAllgather(my_device_index, input_tv, output_tv, comms);
      } else {
        lowerToGather(my_device_index, input_tv, output_tv, comms);
      }
    } else {
      lowerToBroadcastOrP2P(
          my_device_index, input_tv, output_tv, is_input_sharded, comms);
    }
  }
  return comms;
}

bool isLowerableToCommunication(Expr* expr) {
  NVF_ERROR(
      ir_utils::isTvOp(expr),
      "Non-tv op is not supported yet: ",
      expr->toString());
  if (expr->isA<ReductionOp>()) {
    auto in = expr->as<ReductionOp>()->in()->as<TensorView>();
    auto out = expr->as<ReductionOp>()->out()->as<TensorView>();
    // get the reduced axis
    std::vector<IterDomain*> reduction_axis;
    std::copy_if(
        out->getRootDomain().begin(),
        out->getRootDomain().end(),
        std::back_inserter(reduction_axis),
        [](IterDomain* id) { return id->isReduction(); });
    // check whether the reduction involves only one axis
    if (reduction_axis.size() != 1) {
      return false;
    }
    // We check whether the reduced axis is sharded on the input
    const auto c2p_map = PairwiseRootDomainMap(in, out).mapConsumerToProducer();
    auto c2p_map_it = c2p_map.find(reduction_axis.at(0));
    return c2p_map_it != c2p_map.end() && c2p_map_it->second->isDeviceDim();
  } else {
    return expr->isA<LoadStoreOp>() &&
        (expr->as<LoadStoreOp>()->opType() == LoadStoreOpType::Set);
  }
}

} // namespace nvfuser
