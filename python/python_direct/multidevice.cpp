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
  py::class_<DeviceMesh>(nvfuser, "DeviceMesh")
      .def(
          py::init([](const std::vector<int64_t>& devices) {
            return new DeviceMesh(devices);
          }),
          py::arg("devices"),
          R"(
Create a new DeviceMesh.
)")
      .def(
          "size",
          static_cast<int64_t (DeviceMesh::*)() const>(&DeviceMesh::size),
          R"(
Returns the number of devices in the mesh.
)");
}

void bindSharding(py::module& nvfuser) {
  py::class_<Sharding>(nvfuser, "Sharding")
      .def(
          "mesh",
          &Sharding::mesh,
          R"(
Returns the device mesh of the sharding.
)")
      .def(
          "axis_sharded_on",
          &Sharding::axisShardedOn,
          py::arg("parallel_type"),
          R"(
Returns the axis sharded on the given parallel type.
)");

  nvfuser.def(
      "get_output_shardings",
      [](Fusion* fusion) {
        std::vector<Sharding> output_shardings = getOutputShardings(fusion);
        NVF_ERROR(
            output_shardings.empty() ||
                std::ssize(output_shardings) ==
                    (int64_t)fusion->outputs().size(),
            "Found ",
            std::ssize(output_shardings),
            " output shardings but expected ",
            fusion->outputs().size(),
            " or 0.");
        return output_shardings;
      },
      py::arg("fusion"),
      R"(
Get the output shardings of the fusion.
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
