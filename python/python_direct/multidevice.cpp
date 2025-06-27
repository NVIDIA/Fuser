// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#include <bindings.h>
#include <direct_utils.h>
#include <python_utils.h>

#include <multidevice/communicator.h>
#include <multidevice/device_mesh.h>
#include <multidevice/executor.h>
#include <multidevice/multidevice.h>
#include <runtime/fusion_kernel_runtime.h>

#include <python_common/distributed_tensor.h>

namespace nvfuser::python {

namespace {

void bindCommunicator(py::module& nvfuser) {
  // py::nodelete is necessary because Communicator does not have a destructor.
  // https://pybind11.readthedocs.io/en/stable/advanced/classes.html#non-public-destructors
  py::class_<Communicator, std::unique_ptr<Communicator, py::nodelete>>
      communicator(nvfuser, "Communicator", py::module_local());
  communicator.def(
      "instance",
      &Communicator::getInstance,
      R"(
Returns the singleton communicator instance.
)",
      py::return_value_policy::reference);
  communicator.def(
      "size",
      &Communicator::size,
      R"(
Returns the number of processes in the communicator.
)");
  communicator.def(
      "rank",
      &Communicator::deviceId,
      R"(
Returns the device ID associated with the current process.
)");
  communicator.def(
      "local_size",
      &Communicator::local_size,
      R"(
Returns the number of processes within the node.
)");
  communicator.def(
      "local_rank",
      &Communicator::local_rank,
      R"(
Returns the in-node rank associated with the current process.
)");
  communicator.def(
      "barrier",
      [](Communicator& self) {
        // Communicator::barrier takes an optional backend argument, which we
        // don't use yet.
        self.barrier();
      },
      R"(
Performs a blocking barrier across all ranks.
)");
}

void bindDeviceMesh(py::module& nvfuser) {
  py::class_<DeviceMesh>(nvfuser, "DeviceMesh", py::module_local())
      .def(
          py::init([](const std::vector<int64_t>& devices) {
            return new DeviceMesh(devices);
          }),
          py::arg("devices"),
          R"(
Create a new DeviceMesh.
)")
      .def(
          "__repr__",
          [](const DeviceMesh& self) {
            std::stringstream ss;
            ss << self;
            return ss.str();
          })
      .def_property_readonly(
          "size",
          [](const DeviceMesh& self) -> int64_t { return self.size(); },
          R"(
Returns the number of devices in the mesh.
)");
}

void bindSharding(py::module& nvfuser) {
  py::class_<Sharding>(nvfuser, "Sharding", py::module_local())
      .def_property_readonly(
          "mesh",
          &Sharding::mesh,
          R"(
Returns the device mesh of the sharding.
)",
          py::return_value_policy::reference)
      .def(
          "axis_sharded_on",
          &Sharding::axisShardedOn,
          py::arg("parallel_type"),
          R"(
Returns the axis sharded on the given parallel type.

If the distributed tensor is replicated on that parallel type, returns -1.
)");
}

} // namespace

void bindMultiDevice(py::module& nvfuser) {
  py::module_ nvf_multidevice = nvfuser.def_submodule(
      "multidevice",
      "This submodule contains all multi-device features for NvFuser.");
  bindCommunicator(nvf_multidevice);
  bindDeviceMesh(nvf_multidevice);
  bindSharding(nvf_multidevice);
}

} // namespace nvfuser::python
