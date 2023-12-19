// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#ifdef USE_DISTRIBUTED
#include <ir/interface_nodes.h>
#include <multidevice/device_mesh.h>
#include <multidevice/lower_communication.h>
#include <multidevice/utils.h>
#include <ops/all_ops.h>
#include <limits>

namespace nvfuser {

namespace {

template <typename T>
inline T getInitialValue(BinaryOpType op) {
  switch (op) {
    case BinaryOpType::Add:
      return 0;
    case BinaryOpType::Mul:
      return 1;
    case BinaryOpType::Min:
      return std::numeric_limits<T>::min();
    case BinaryOpType::Max:
      return std::numeric_limits<T>::max();
    case BinaryOpType::BitwiseAnd:
      return std::numeric_limits<T>::max();
    case BinaryOpType::BitwiseOr:
      return 0;
    case BinaryOpType::BitwiseXor:
      return 0;
    default:
      NVF_ERROR(false, "invalid binary op type");
      return 0;
  }
}

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

// Creates a dummy tensor for scatter/gather communications,
// see 'createParamsForGatherScatter'
inline at::Tensor createDummyTensor(at::Tensor reference) {
  return at::empty_like(reference, reference.options());
}

inline at::Tensor createDummyTensor(
    at::Tensor reference,
    BinaryOpType op_type) {
  return createDummyTensor(reference).fill_(getInitialValue<float>(op_type));
}

// Utility function used for setting up a scatter or gather communication
// params. Since most  of the steps are somewhat similar/opposite in those
// cases, we gathered the two implementations into one function. The argument
// "is_scatter" allows to discriminate between scatter and gather
CommParams createParamsForGatherScatter(
    DeviceIdxType my_device_index,
    DeviceIdxType root,
    const DeviceMesh& mesh, // is_scatter? receivers : senders
    at::Tensor root_buf, // is_scatter? input buf : output buf
    at::Tensor buf, // is_scatter? output buf : input buf
    bool is_scatter) {
  CommParams params;
  params.root = root;
  params.team = mesh.vector();
  bool is_root_in_mesh = mesh.has(root);
  if (!is_root_in_mesh) {
    params.team.push_back(root);
  }

  if (mesh.has(my_device_index)) {
    auto sliced_buf = buf.index({0, "..."});
    ((is_scatter) ? params.dst_bufs : params.src_bufs) = {sliced_buf};
  }

  if (my_device_index == root) {
    for (auto i : c10::irange(mesh.vector().size())) {
      ((is_scatter) ? params.src_bufs : params.dst_bufs)
          .push_back(root_buf.index({static_cast<int>(i), "..."}));
    }
    // The scatter/gather semantics imposes the root to be both
    // sender and receiver. If the root is not in the mesh, we thus
    // have to artificially make it send and receive a dummy buffer
    // Since it is an "inplace" operation, this should not cause any overhead
    if (!is_root_in_mesh) {
      at::Tensor dummy = createDummyTensor(root_buf.index({0, "..."}));
      params.src_bufs.push_back(dummy);
      params.dst_bufs.push_back(dummy);
    }
  }
  return params;
}

// Adds one or zero Scatter communication to the vector 'comms'
void lowerToScatter(
    DeviceIdxType my_device_index,
    const DeviceMesh& sender_mesh,
    const DeviceMesh& receiver_mesh,
    at::Tensor input_tensor,
    at::Tensor output_tensor,
    std::vector<std::shared_ptr<Communication>>& comms) {
  // we arbitrarily choose the first device of the sender mesh to be the root
  auto root = sender_mesh.vector().at(0);
  if (!isDeviceInvolved(my_device_index, root, receiver_mesh)) {
    return;
  }
  auto params = createParamsForGatherScatter(
      my_device_index, root, receiver_mesh, input_tensor, output_tensor, true);
  comms.push_back(std::make_shared<Scatter>(std::move(params)));
}

/*
Adds zero or multiple Gather communications to the vector 'comms'

Note that since the root of a Gather collective is a destination, we possibly
need multiple Gather if the tensor is replicated in the receiver mesh.
*/
void lowerToGather(
    DeviceIdxType my_device_index,
    const DeviceMesh& sender_mesh,
    const DeviceMesh& receiver_mesh,
    at::Tensor input_tensor,
    at::Tensor output_tensor,
    std::vector<std::shared_ptr<Communication>>& comms) {
  // we create as many 'Gathers' as there are devices in the receiver mesh
  for (auto root : receiver_mesh.vector()) {
    if (!isDeviceInvolved(my_device_index, root, sender_mesh)) {
      continue;
    }
    auto params = createParamsForGatherScatter(
        my_device_index, root, sender_mesh, output_tensor, input_tensor, false);
    comms.push_back(std::make_shared<Gather>(std::move(params)));
  }
}

// Add one or zero Allgather communication to the vector 'comms'
void lowerToAllgather(
    DeviceIdxType my_device_index,
    const DeviceMesh& mesh,
    at::Tensor input_tensor,
    at::Tensor output_tensor,
    std::vector<std::shared_ptr<Communication>>& comms) {
  if (!mesh.has(my_device_index)) {
    return;
  }

  CommParams params;
  params.team = mesh.vector();
  for (auto i : c10::irange(mesh.vector().size())) {
    params.dst_bufs.push_back(
        output_tensor.index({static_cast<int>(i), "..."}));
  }
  params.src_bufs = {input_tensor.index({0, "..."})};

  comms.push_back(std::make_shared<Allgather>(std::move(params)));
}

// Creates and set the CommParams for a Broadcast or Send/Recv communication
CommParams createParamsForBroadcastOrP2P(
    DeviceIdxType my_device_index,
    DeviceIdxType root,
    const DeviceMesh& mesh, // receiver devices
    at::Tensor input_tensor,
    at::Tensor output_tensor) {
  CommParams params;
  params.root = root;
  params.team = mesh.vector();
  if (!mesh.has(root)) {
    params.team.push_back(root);
  }

  if (my_device_index == root) {
    params.src_bufs = {input_tensor};
  }
  if (mesh.has(my_device_index)) {
    params.dst_bufs = {output_tensor};
  }

  return params;
}

// Adds one or zero Broadcast or Send/Recv communication to the vector 'comms'
void lowerToBroadcastOrP2P(
    DeviceIdxType my_device_index,
    DeviceIdxType root,
    const DeviceMesh& mesh, // receiver devices
    at::Tensor input_tensor,
    at::Tensor output_tensor,
    std::vector<std::shared_ptr<Communication>>& comms) {
  if (!isDeviceInvolved(my_device_index, root, mesh)) {
    return;
  }
  auto params = createParamsForBroadcastOrP2P(
      my_device_index, root, mesh, input_tensor, output_tensor);
  std::shared_ptr<Communication> comm;
  if (mesh.vector().size() == 1) {
    comm = std::make_shared<SendRecv>(std::move(params));
  } else {
    comm = std::make_shared<Broadcast>(std::move(params));
  }
  comms.push_back(comm);
}

// Adds several Broadcast or Send/Recv communications to the vector 'comms'
// For now, we assume that this function is called only if
// the input and output have the same sharding. Later we could support more
// general cases.
void lowerToBroadcastOrP2P(
    DeviceIdxType my_device_index,
    const DeviceMesh& sender_mesh,
    const DeviceMesh& receiver_mesh,
    at::Tensor input_tensor,
    at::Tensor output_tensor,
    bool is_sharded,
    std::vector<std::shared_ptr<Communication>>& comms) {
  if (is_sharded) {
    // if the inputs and ouputs are parallelized,
    // we create as many Broadcast as that will be handled in parallel
    for (auto i : c10::irange(sender_mesh.vector().size())) {
      NVF_ERROR(
          sender_mesh.vector().size() == receiver_mesh.vector().size(),
          "the receiver and sender meshes have different sizes");
      at::Tensor input, output;
      if (input_tensor.numel()) {
        input = input_tensor.index({static_cast<int>(0), "..."});
      }
      if (output_tensor.numel()) {
        output = output_tensor.index({static_cast<int>(0), "..."});
      }
      lowerToBroadcastOrP2P(
          my_device_index,
          sender_mesh.vector().at(i),
          DeviceMesh({receiver_mesh.vector().at(i)}),
          input,
          output,
          comms);
    }
  } else {
    // we arbitrarily choose the first device of the sender mesh to be the root
    lowerToBroadcastOrP2P(
        my_device_index,
        sender_mesh.vector().at(0),
        receiver_mesh,
        input_tensor,
        output_tensor,
        comms);
  }
}

CommParams createParamsForReduce(
    DeviceIdxType my_device_index,
    DeviceIdxType root,
    const DeviceMesh& mesh,
    at::Tensor input_tensor,
    at::Tensor output_tensor,
    BinaryOpType op_type) {
  CommParams params;
  params.root = root;
  params.redOp = getC10dReduceOpType(op_type);
  params.team = mesh.vector();
  bool is_root_in_mesh = mesh.has(root);
  if (!is_root_in_mesh) {
    params.team.push_back(root);
  }

  if (mesh.has(my_device_index)) {
    auto sliced_buf = input_tensor.index({0, "..."});
    params.src_bufs = {sliced_buf};
  }

  if (my_device_index == root) {
    params.dst_bufs = {output_tensor};
    // The reduce semantics imposes the root to be both
    // sender and receiver. If the root is not in the mesh, we thus
    // have to artificially make it send and receive a dummy buffer
    if (!is_root_in_mesh) {
      at::Tensor dummy = createDummyTensor(output_tensor, op_type);
      params.src_bufs.push_back(dummy);
    }
  }
  return params;
}

void lowerToReduce(
    DeviceIdxType my_device_index,
    const DeviceMesh& sender_mesh,
    const DeviceMesh& receiver_mesh,
    at::Tensor input_tensor,
    at::Tensor output_tensor,
    BinaryOpType op_type,
    std::vector<std::shared_ptr<Communication>>& comms) {
  // we create as many Reduces as there are devices in the receiver mesh
  for (auto root : receiver_mesh.vector()) {
    if (!isDeviceInvolved(my_device_index, root, sender_mesh)) {
      continue;
    }
    auto params = createParamsForReduce(
        my_device_index,
        root,
        sender_mesh,
        input_tensor,
        output_tensor,
        op_type);
    comms.push_back(std::make_shared<Reduce>(std::move(params)));
  }
}

void lowerToAllreduce(
    DeviceIdxType my_device_index,
    const DeviceMesh& mesh,
    at::Tensor input_tensor,
    at::Tensor output_tensor,
    BinaryOpType op_type,
    std::vector<std::shared_ptr<Communication>>& comms) {
  if (!mesh.has(my_device_index)) {
    return;
  }
  CommParams params;
  params.redOp = getC10dReduceOpType(op_type);
  params.team = mesh.vector();
  params.dst_bufs = {output_tensor};
  auto sliced_buf = input_tensor.index({0, "..."});
  params.src_bufs = {sliced_buf};

  comms.push_back(std::make_shared<Allreduce>(params));
}

void lowerToReduceScatter(
    DeviceIdxType my_device_index,
    const DeviceMesh& mesh,
    at::Tensor input_tensor,
    at::Tensor output_tensor,
    BinaryOpType op_type,
    std::vector<std::shared_ptr<Communication>>& comms) {
  if (!mesh.has(my_device_index)) {
    return;
  }
  CommParams params;
  params.redOp = getC10dReduceOpType(op_type);
  params.team = mesh.vector();
  params.dst_bufs = {output_tensor.index({0, "..."})};
  for (int i : params.team) {
    auto sliced_buf = input_tensor.index({0, i, "..."});
    params.src_bufs.push_back(sliced_buf);
  }

  comms.push_back(std::make_shared<ReduceScatter>(params));
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
std::vector<std::shared_ptr<Communication>> lowerCommunication(
    DeviceIdxType my_device_index,
    Expr* c,
    at::Tensor input_tensor,
    at::Tensor output_tensor) {
  std::vector<std::shared_ptr<Communication>> comms;
  NVF_ERROR(
      c->inputs().size() == 1 && c->inputs().at(0)->isA<TensorView>() &&
          c->outputs().size() == 1 && c->outputs().at(0)->isA<TensorView>(),
      "I/O must be TensorViews");
  TensorView* input_tv = c->inputs().at(0)->as<TensorView>();
  TensorView* output_tv = c->outputs().at(0)->as<TensorView>();
  at::Tensor dummy;

  const auto& sender_mesh = *input_tv->getDeviceMesh();
  const auto& receiver_mesh = *output_tv->getDeviceMesh();

  // Stores whether the I/O has its first axis parallelized on Didx
  const bool is_input_sharded =
      isSharded(input_tv) && sender_mesh.vector().size() > 1;
  const bool is_output_sharded =
      isSharded(output_tv) && receiver_mesh.vector().size() > 1;

  auto original_expr = output_tv->definition();
  NVF_ERROR(
      isLowerableToCommunication(original_expr),
      "Lowering expression ",
      original_expr,
      " to communication is not supported");
  bool is_reduction = original_expr->isA<ReductionOp>();

  NVF_ERROR(
      !is_input_sharded || !input_tensor.numel() ||
          sender_mesh.vector().size() ==
              static_cast<size_t>(input_tensor.size(0)),
      "the size of the mesh ",
      sender_mesh.vector().size(),
      " doesn't match the size of the tensor ",
      input_tensor.size(0));
  NVF_ERROR(
      !is_output_sharded || !output_tensor.numel() || is_reduction ||
          receiver_mesh.vector().size() ==
              static_cast<size_t>(output_tensor.size(0)),
      "the size of the mesh",
      receiver_mesh.vector().size(),
      " doesn't match the size of the tensor ",
      output_tensor.size(0));
  if (is_reduction) {
    BinaryOpType op_type =
        output_tv->definition()->as<ReductionOp>()->getReductionOpType();
    NVF_ERROR(
        is_input_sharded,
        "the comm input must be sharded in case of reduce.",
        "Insert a `set` before the reduction to reshard")
    if (is_output_sharded) {
      if (receiver_mesh == sender_mesh) {
        lowerToReduceScatter(
            my_device_index,
            sender_mesh,
            input_tensor,
            output_tensor,
            op_type,
            comms);
      }
    } else {
      if (receiver_mesh == sender_mesh) {
        lowerToAllreduce(
            my_device_index,
            sender_mesh,
            input_tensor,
            output_tensor,
            op_type,
            comms);
      } else {
        lowerToReduce(
            my_device_index,
            sender_mesh,
            receiver_mesh,
            input_tensor,
            output_tensor,
            op_type,
            comms);
      }
    }
  } else {
    if (!is_input_sharded && is_output_sharded) {
      lowerToScatter(
          my_device_index,
          sender_mesh,
          receiver_mesh,
          input_tensor,
          output_tensor,
          comms);
    } else if (is_input_sharded && !is_output_sharded) {
      if (receiver_mesh == sender_mesh) {
        lowerToAllgather(
            my_device_index, sender_mesh, input_tensor, output_tensor, comms);
      } else {
        lowerToGather(
            my_device_index,
            sender_mesh,
            receiver_mesh,
            input_tensor,
            output_tensor,
            comms);
      }
    } else {
      lowerToBroadcastOrP2P(
          my_device_index,
          sender_mesh,
          receiver_mesh,
          input_tensor,
          output_tensor,
          is_input_sharded,
          comms);
    }
  }
  return comms;
}

bool isLowerableToCommunication(Expr* expr) {
  if (expr->isA<ReductionOp>()) {
    auto out = expr->as<ReductionOp>()->out();
    NVF_ERROR(out->isA<TensorView>(), "output is not a TensorView");
    auto out_tv = out->as<TensorView>();
    NVF_ERROR(
        out_tv->domain()->nDims() ==
            TensorDomain::noReductions(out_tv->getMaybeRFactorDomain()).size() +
                1,
        "only reducing one-axis at a time is supported");
    return true;
  }
  return expr->isA<LoadStoreOp>() &&
      (expr->as<LoadStoreOp>()->opType() == LoadStoreOpType::Set);
}

} // namespace nvfuser

#endif
