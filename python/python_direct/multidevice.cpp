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
  bindDeviceMesh(nvf_multidevice);
  bindSharding(nvf_multidevice);
}

} // namespace nvfuser::python
