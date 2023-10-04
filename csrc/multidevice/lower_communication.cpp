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
#include <multidevice/pipeline.h>

namespace nvfuser {

namespace {

// Returns whether a TensorView has its first axis parallelized on Didx
// Checks that the other axis are not parallelized on Didx
bool isParallelD(TensorView* tv) {
  std::vector<bool> is_parallel_d;
  for (IterDomain* id : tv->getLeafDomain()) {
    is_parallel_d.push_back(isParallelTypeDeviceDim(id->getParallelType()));
  }
  // Currently, only the most external dim is allowed to be parallelized
  NVF_ERROR(tv->getMaybeRFactorDomain() == tv->getLeafDomain());
  for (auto i : c10::irange(1, is_parallel_d.size())) {
    NVF_ERROR(
        !is_parallel_d.at(i),
        "only the outmost dimension can be device-parallelized");
  }
  return is_parallel_d.empty() ? false : is_parallel_d.at(0);
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
    auto sliced_buf =
        buf.index({static_cast<int>(mesh.findIndex(my_device_index)), "..."});
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

// Adds one or zero Gather communication to the vector 'comms'
void lowerToGather(
    DeviceIdxType my_device_index,
    const DeviceMesh& sender_mesh,
    const DeviceMesh& receiver_mesh,
    at::Tensor input_tensor,
    at::Tensor output_tensor,
    std::vector<std::shared_ptr<Communication>>& comms) {
  // we create as many 'Gathers' as there are devices in the receiver mesh
  for (auto root : receiver_mesh.vector()) {
    if (!isDeviceInvolved(my_device_index, root, sender_mesh))
      continue;
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
  if (!mesh.has(my_device_index))
    return;

  CommParams params;
  params.team = mesh.vector();
  for (auto i : c10::irange(mesh.vector().size())) {
    params.dst_bufs.push_back(
        output_tensor.index({static_cast<int>(i), "..."}));
  }
  params.src_bufs = {
      input_tensor.index({mesh.findIndex(my_device_index), "..."})};

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
  if (!isDeviceInvolved(my_device_index, root, mesh))
    return;
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
// the input and output have the same parallelization (given by
// the argument "is_parallelized"). Later we could support more general cases.
void lowerToBroadcastOrP2P(
    DeviceIdxType my_device_index,
    const DeviceMesh& sender_mesh,
    const DeviceMesh& receiver_mesh,
    at::Tensor input_tensor,
    at::Tensor output_tensor,
    bool is_parallelized,
    std::vector<std::shared_ptr<Communication>>& comms) {
  if (is_parallelized) {
    // if the inputs and ouputs are parallelized,
    // we create as many Broadcast as that will be handled in parallel
    for (auto i : c10::irange(sender_mesh.vector().size())) {
      NVF_ERROR(
          sender_mesh.vector().size() == receiver_mesh.vector().size(),
          "the receiver and sender meshes have different sizes");
      lowerToBroadcastOrP2P(
          my_device_index,
          sender_mesh.vector().at(i),
          DeviceMesh({receiver_mesh.vector().at(i)}),
          input_tensor.index({static_cast<int>(i), "..."}),
          output_tensor.index({static_cast<int>(i), "..."}),
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
    PipelineCommunication* c,
    at::Tensor input_tensor,
    at::Tensor output_tensor) {
  std::vector<std::shared_ptr<Communication>> comms;
  TensorView* input_tv =
      c->in()->as<PipelineVal>()->getOriginalVal()->as<TensorView>();
  TensorView* output_tv =
      c->out()->as<PipelineVal>()->getOriginalVal()->as<TensorView>();
  at::Tensor dummy;

  const auto& sender_mesh =
      c->in()->as<PipelineVal>()->getStage()->descriptor()->mesh;
  const auto& receiver_mesh =
      c->out()->as<PipelineVal>()->getStage()->descriptor()->mesh;

  // Stores whether the I/O has its first axis parallelized on Didx
  bool is_input_parallel_d = isParallelD(input_tv);
  bool is_output_parallel_d = isParallelD(output_tv);

  NVF_ERROR(
      !is_input_parallel_d ||
          sender_mesh.vector().size() ==
              static_cast<size_t>(input_tensor.size(0)),
      "the size of the mesh",
      sender_mesh.vector().size(),
      " doesn't match the size of the tensor ",
      input_tensor.size(0));
  NVF_ERROR(
      !is_output_parallel_d ||
          receiver_mesh.vector().size() ==
              static_cast<size_t>(output_tensor.size(0)),
      "the size of the mesh",
      receiver_mesh.vector().size(),
      " doesn't match the size of the tensor ",
      output_tensor.size(0));
  NVF_ERROR(!sender_mesh.vector().empty(), "sender mesh is empty");
  NVF_ERROR(!receiver_mesh.vector().empty(), "receiver mesh is empty");

  if (!isDeviceInvolved(my_device_index, sender_mesh, receiver_mesh))
    return {};

  if (!is_input_parallel_d && is_output_parallel_d) {
    lowerToScatter(
        my_device_index,
        sender_mesh,
        receiver_mesh,
        input_tensor,
        output_tensor,
        comms);
  } else if (is_input_parallel_d && !is_output_parallel_d) {
    if (receiver_mesh.vector() == sender_mesh.vector()) {
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
        is_input_parallel_d,
        comms);
  }
  return comms;
}

} // namespace nvfuser

#endif
