// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#include <python_frontend/python_bindings.h>

#include <multidevice/communicator.h>

namespace nvfuser::python_frontend {

void bindCommunicator(py::module& nvfuser) {
  // py::nodelete is necessary because Communicator doesn't have a destructor:
  // https://pybind11.readthedocs.io/en/stable/advanced/classes.html#non-public-destructors
  py::class_<Communicator, std::unique_ptr<Communicator, py::nodelete>>
      communicator(nvfuser, "Communicator");
  communicator.def(
      "instance",
      &Communicator::getInstance,
      py::return_value_policy::reference);
  communicator.def("size", &Communicator::size);
  communicator.def("rank", &Communicator::deviceId);
  communicator.def("local_size", &Communicator::local_size);
  communicator.def("local_rank", &Communicator::local_rank);
}

} // namespace nvfuser::python_frontend
