// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#include <python_frontend/python_bindings.h>

#include <multidevice/communicator.h>
#include <multidevice/device_mesh.h>
#include <multidevice/utils.h>

namespace nvfuser::python_frontend {

namespace {
void bindCommunicator(py::module& nvfuser) {
  // py::nodelete is necessary because Communicator doesn't have a destructor:
  // https://pybind11.readthedocs.io/en/stable/advanced/classes.html#non-public-destructors
  py::class_<Communicator, std::unique_ptr<Communicator, py::nodelete>>
      communicator(nvfuser, "Communicator");
  communicator.def(
      "instance",
      &Communicator::getInstance,
      "Returns the singleton communicator instance.",
      py::return_value_policy::reference);
  communicator.def(
      "size",
      &Communicator::size,
      "Returns the number of processes in the communicator.");
  communicator.def(
      "rank",
      &Communicator::deviceId,
      "Returns the device ID associated with the current process.");
  communicator.def(
      "local_size",
      &Communicator::local_size,
      "Returns the number of processes within the node.");
  communicator.def(
      "local_rank",
      &Communicator::local_rank,
      "Returns the in-node rank associated with the current process.");
  communicator.def(
      "barrier",
      [](Communicator& self) {
        // Communicator::barrier takes an optional backend argument, which we
        // don't use yet.
        self.barrier();
      },
      "Performs a blocking barrier across all ranks.");
}

void bindDeviceMesh(py::module& nvfuser) {
  py::class_<DeviceMesh> device_mesh(nvfuser, "DeviceMesh");
  device_mesh.def(py::init<std::vector<int64_t>>());
  device_mesh.def("__repr__", [](const DeviceMesh& self) {
    std::stringstream ss;
    ss << self;
    return ss.str();
  });
  device_mesh.def_property_readonly(
      "size",
      [](const DeviceMesh& self) -> int64_t { return self.size(); },
      "Returns the size of the mesh.");
  device_mesh.def(
      "shard_tensor",
      [](const DeviceMesh& self,
         at::Tensor tensor,
         const int64_t axis,
         int64_t device_id) -> at::Tensor {
        return shardTensor(tensor, axis, self, device_id);
      },
      py::arg("tensor"),
      py::arg("axis"),
      py::arg("device_id"));
}

void bindDistributedTensor(py::module& nvfuser) {
  py::class_<Sharding> distributed_tensor(nvfuser, "Sharding");
  distributed_tensor.def_property_readonly(
      "mesh",
      &Sharding::mesh,
      "Returns the device mesh.",
      py::return_value_policy::reference);
  distributed_tensor.def(
      "axis_sharded_on",
      &Sharding::axisShardedOn,
      R"(
      Returns the axis sharded on the given parallel type.

      If the distributed tensor is replicated on that parallel type, returns -1.
      )",
      py::arg("parallel_type"));
}

} // namespace

void bindMultidevice(py::module& nvfuser) {
  bindCommunicator(nvfuser);
  bindDeviceMesh(nvfuser);
  bindDistributedTensor(nvfuser);
}

} // namespace nvfuser::python_frontend
