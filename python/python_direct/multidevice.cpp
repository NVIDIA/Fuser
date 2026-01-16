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
#include <multidevice/execution_utils.h>
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
  py::class_<DeviceMesh> device_mesh(nvfuser, "DeviceMesh", py::module_local());
  device_mesh.def(
      py::init([](at::Tensor devices) {
        return std::make_unique<DeviceMesh>(std::move(devices));
      }),
      py::arg("devices"),
      R"(
Create a new DeviceMesh from torch.Tensor.
)");
  device_mesh.def(
      py::init([](const std::vector<int64_t>& devices) {
        return std::make_unique<DeviceMesh>(at::tensor(devices));
      }),
      py::arg("devices"),
      R"(
Create a new DeviceMesh from an integer list, for backward compatibility.
)");
  device_mesh.def("__repr__", [](const DeviceMesh& self) {
    std::stringstream ss;
    ss << self;
    return ss.str();
  });
  device_mesh
      .def_property_readonly(
          "size",
          [](const DeviceMesh& self) -> int64_t { return self.size(); },
          R"(
Returns the number of devices in the mesh.
)")
      .def_property_readonly(
          "shape",
          [](const DeviceMesh& self) -> at::IntArrayRef {
            return self.shape();
          },
          R"(
Returns the shape of the mesh.
)");
  device_mesh.def(
      "shard_tensor",
      [](const DeviceMesh& self, at::Tensor tensor, const int64_t axis)
          -> at::Tensor { return shardTensor1D(tensor, axis, self); },
      py::arg("tensor"),
      py::arg("axis"),
      R"(
Shards the input tensor along `axis`. Returns the sharded tensor.)");
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

void bindMultiDeviceExecutor(py::module& nvfuser) {
  // Bind params type under the multidevice submodule. We'll alias it to the
  // top-level module in bindMultiDevice to allow direct imports.
  py::class_<MultiDeviceExecutorParams>(nvfuser, "MultiDeviceExecutorParams")
      .def(py::init<>())
      .def_property(
          "use_allocation_cache",
          [](const MultiDeviceExecutorParams& self) {
            return self.executor.use_allocation_cache;
          },
          [](MultiDeviceExecutorParams& self, bool value) {
            self.executor.use_allocation_cache = value;
          })
      .def_property(
          "backend_type",
          [](const MultiDeviceExecutorParams& self) {
            return self.lower.communicator_backend;
          },
          [](MultiDeviceExecutorParams& self, CommunicatorBackend value) {
            self.lower.communicator_backend = value;
          })
      .def_property(
          "offset_stream_indexing_by_rank",
          [](const MultiDeviceExecutorParams& self) {
            return self.lower.offset_stream_indexing_by_rank;
          },
          [](MultiDeviceExecutorParams& self, bool value) {
            self.lower.offset_stream_indexing_by_rank = value;
          });

  py::class_<MultiDeviceExecutor> multi_device_executor(
      nvfuser, "MultiDeviceExecutor");
  multi_device_executor.def(
      py::init(
          [](const Fusion& fusion, const MultiDeviceExecutorParams& params) {
            return std::make_unique<MultiDeviceExecutor>(
                std::make_unique<Fusion>(fusion),
                Communicator::getInstance(),
                params);
          }),
      R"(
Create a new MultiDeviceExecutor.

Parameters
----------
fusion : Fusion
    The fusion to be executed.
params : MultiDeviceExecutorParams
    Parameters configuring the executor and communicator backend.

Examples
--------
>>> params = MultiDeviceExecutorParams()
>>> params.backend_type = CommunicatorBackend.nccl
>>> multi_device_executor = MultiDeviceExecutor(fusion, params)
>>> outputs = multi_device_executor.run(inputs)
)",
      py::arg("fusion"),
      py::arg("params"));
  multi_device_executor.def(
      "__str__",
      [](MultiDeviceExecutor& self) {
        std::stringstream ss;
        self.print(ss);
        return ss.str();
      },
      R"(
Return a string representing the MultiDeviceExecutor.
)");
  multi_device_executor.def(
      "run",
      [](MultiDeviceExecutor& self, const py::iterable& args) {
        KernelArgumentHolder outputs = self.runWithInput(from_pyiterable(args));
        return to_tensor_vector(outputs);
      },
      R"(
              Run the fusion with the given arguments.

              Parameters
              ----------
              args : KernelArgumentHolder
                  The input arguments for the fusion.

              Returns
              -------
              list of Tensor
                  The output tensors containing the results.
            )",
      py::arg("args"));
}

} // namespace

void bindMultiDevice(py::module& nvfuser) {
  py::module_ nvf_multidevice = nvfuser.def_submodule(
      "multidevice",
      "This submodule contains all multi-device features for NvFuser.");
  bindCommunicator(nvf_multidevice);
  bindDeviceMesh(nvf_multidevice);
  bindSharding(nvf_multidevice);
  bindMultiDeviceExecutor(nvf_multidevice);

  // Standalone shard_tensor function that takes TensorView
  nvf_multidevice.def(
      "shard_tensor",
      [](at::Tensor tensor, TensorView* tv) -> at::Tensor {
        return shardTensor(tensor, tv);
      },
      py::arg("tensor"),
      py::arg("tv"),
      R"(
Shards the input tensor according to the TensorView's parallelization and device mesh.

This function automatically:
- Determines the sharding axis from the TensorView's DIDx parallelization
- Extracts the device mesh from the TensorView
- Uses the current rank from the Communicator singleton
- Returns the original tensor if the TensorView is not sharded

Parameters
----------
tensor : torch.Tensor
    The unsharded tensor to shard
tv : TensorView
    The TensorView that defines the sharding pattern (must have device mesh
    and DIDx parallelization if sharded)

Returns
-------
torch.Tensor
    The sharded tensor for the current device

Examples
--------
>>> mesh = nvfuser.multidevice.DeviceMesh(torch.arange(num_devices))
>>> with nvfuser.FusionDefinition() as fd:
...     inp_tv = fd.define_tensor([-1, -1])
...     inp_tv.set_device_mesh(mesh)
...     inp_tv.axis(0).parallelize(nvfuser.ParallelType.mesh_x)
...     # ... rest of fusion definition
>>>
>>> unsharded_tensor = torch.randn(num_devices, 4)
>>> sharded_tensor = nvfuser.multidevice.shard_tensor(unsharded_tensor, inp_tv)
)");
}

} // namespace nvfuser::python
