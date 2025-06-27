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

void bindMultiDeviceExecutor(py::module& nvfuser) {
  py::class_<MultiDeviceExecutor>(nvfuser, "MultiDeviceExecutor")
      .def(
          py::init([](const Fusion* fusion, CommunicatorBackend backend_type) {
            // Make a copy of the fusion for MultiDeviceExecutor to own.
            MultiDeviceExecutorParams params;
            params.lower.communicator_backend = backend_type;
            return new MultiDeviceExecutor(
                std::make_unique<Fusion>(*fusion),
                Communicator::getInstance(),
                std::move(params));
          }),
          py::arg("fusion"),
          py::arg("backend_type") = CommunicatorBackend::kNccl,
          R"(
Create a new MultiDeviceExecutor.

Parameters
----------
fusion : Fusion
    The fusion to be executed.
backend_type : CommunicatorBackend, optional
    The backend type to use for the communicator.
    Default is CommunicatorBackend.kNccl.
)");
}

} // namespace

void bindMultiDevice(py::module& nvfuser) {
  py::module_ nvf_multidevice = nvfuser.def_submodule(
      "multidevice",
      "This submodule contains all multi-device features for NvFuser.");
  bindDeviceMesh(nvf_multidevice);
  bindMultiDeviceExecutor(nvf_multidevice);
}

} // namespace nvfuser::python
