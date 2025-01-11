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

} // namespace nvfuser::python_frontend
