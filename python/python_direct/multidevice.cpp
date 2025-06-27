// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#include <bindings.h>
#include <direct_utils.h>
#include <multidevice/device_mesh.h>
#include <python_utils.h>

namespace nvfuser::python {

namespace {

void bindDeviceMesh(py::module& nvfuser) {
  py::class_<DeviceMesh>(nvfuser, "DeviceMesh")
      .def(py::init([](const std::vector<int64_t>& devices) {
        return new DeviceMesh(devices);
      }),
          py::arg("devices"),
          R"(
Create a new DeviceMesh.
)");
}

} // namespace

void bindMultiDevice(py::module& nvfuser) {
  py::module_ nvf_multidevice =
      nvfuser.def_submodule("multidevice", "This submodule contains all multi-device features for NvFuser.");
  bindDeviceMesh(nvf_multidevice);
}

} // namespace nvfuser::python