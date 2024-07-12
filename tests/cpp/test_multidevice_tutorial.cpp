// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on

#include <multidevice/communicator.h>
#include <ops/all_ops.h>
#include <tests/cpp/multidevice.h>

namespace nvfuser {

class MultiDeviceTutorial : public MultiDeviceTest {
 protected:
  static void SetUpTestSuite() {
    verbose_ = getNvFuserEnv("TUTORIAL_VERBOSE");
  }

 protected:
  static bool verbose_;
};

bool MultiDeviceTutorial::verbose_ = false;

// The first object we need to introduce is the class `Communicator` which is
// a convenience that is nvFuser's interface to runtime distributed setting
// and backend. Communicator's setup is expansive (because all IP addresses
// need to be exchanged to all ranks) and occupies a port. Therefore, we
// instantiate one Communicator globally and use it accross all tests. There
// is no benefit in having multiple communicator, and this class is thought to
// have a singleton instance. This communicator can be accessed through
// MultiDeviceTest::communicator_
TEST_F(MultiDeviceTutorial, CommunicatorAndC10d) {
  Communicator* communicator = communicator_;

  // To check if distributed setting is available:
  if (communicator->is_available() == false) {
    GTEST_SKIP() << "distributed setting not available";
  }
  // During instantiation, the Communicator will parse information from the
  // environment variable. Setting those env vars properly is what is requested
  // from the user for multidevice to be available in nvFuser. Those environment
  // variables are set automatically in the case of a single-node system where
  // "mpirun" (from open MPI) is used to launch the application. The actual list
  // of required environment variables are:
  //  - Set automatically by mpirun (so, no need to manually set them if using
  //  mpirun):
  //      - OMPI_COMM_WORLD_RANK
  //      - OMPI_COMM_WORLD_SIZE
  //      - OMPI_COMM_WORLD_LOCAL_RANK
  //      - OMPI_COMM_WORLD_LOCAL_SIZE
  //  - Required for multi-node systems only:
  //      - MASTER_ADDR (containing the address of one of the node which will be
  //      the server for the c10d::TCPStore)
  //  - Optional:
  //      - MASTER_PORT

  // All those information can then be accessed through the communicator:
  if (verbose_) {
    std::cout
        // We identify MPI rank and (global) device index
        << "rank=device_index=" << communicator->deviceId() << ", "
        << "local_rank=" << communicator->local_rank() << ", "
        << "(c10::Device)device=" << communicator->device()
        << ", " // = c10:Device("cuda:" +
                // std::to_string(communicator->local_rank()))
        << "size=" << communicator->size() << ", "
        << "local_size=" << communicator->local_size() << "." << std::endl;
  }
  // remark:  torch c++ library has several namespaces torch::, at:: (ATen),
  // c10:: ("see-Ten"), c10d:: ("d" for distributed)

  // The communicator is also responsible for managing the different backend
  // that we will use to execute network communication. Currently, the backend
  // we support go through pytorch Process Group (abbreviated as PG), and are
  // UCC PG and NCCL PG. We can request the communicator to return a
  // c10::Backend by specifying:
  // - the Team, aka, the group of ranks that this backend will serve
  std::vector<int64_t> all_devices(communicator->size());
  std::iota(
      all_devices.begin(),
      all_devices.end(),
      0); // all_devices = [0,1,..., communicator->size()-1]
  Team& team = all_devices;
  // - (optionally) the backend type, between UCC and NCCL
  CommunicatorBackend backend_type =
      CommunicatorBackend::nccl; // or CommunicatorBackend::ucc
  // the backend_type is an optional argument. By default it will choose nccl if
  // available, ucc otherwise. We can check that the requested backend is indeed
  // available
  if (communicator->isBackendAvailable(backend_type) == false) {
    GTEST_SKIP() << "Backend not available";
  }

  // The backend can be retrieved from the the communicator, which manages a
  // cache
  c10d::Backend* backend = communicator->getBackendForTeam(team, backend_type);

  // The c10d backend can then be used to execute collectives. This is used
  // internally in nvFuser to execute collectives, but this is typically too low
  // level to be exposed to the user. Let us show for convenience how this torch
  // library is used on a simple example, e.g., "allreduce" a single buffer
  constexpr int64_t tensor_length = 128;
  // each ranks will allocate a buffer on a different device
  c10::TensorOptions tensor_options =
      at::TensorOptions().device(communicator->device());
  std::vector<at::Tensor> buffers = {at::ones({tensor_length}, tensor_options)};
  //  Posting a collective is non-blocking and returns a work handle
  //  `c10d::Work`
  c10::intrusive_ptr<c10d::Work> work_handle = backend->allreduce(
      buffers); // c10::intrusive_ptr is nothing but a "smart pointer"

  // The collective launches a kernel on the GPU on an internal stream that is
  // progressed asynchronously from the main stream. We need to "wait" for
  // completion to synchronize the current stream with the backend's internal
  // stream
  work_handle->wait();
  // Note that the "wait" primitive is non-blocking from the CPU point of view,
  // it only adds a synchronization point between streams. The only exception to
  // that rule is the collective "barrier" which by convention blocks the CPU as
  // well.

  // Let us validate the result:
  at::Tensor expected_result =
      at::ones({tensor_length}, tensor_options) * communicator->size();
  EXPECT_TRUE(buffers.at(0).equal(expected_result));
}

// Now that we have introduce the class Communicator, which is the interface
// between nvFuser and lower-level backend, let us now turn to the higher level
// discussion of multidevice scheduling. The first piece we introduce in the
// following test is the one of Device Mesh, which allows us to select the
// devices on which the tensors will be sharded.
TEST_F(MultiDeviceTutorial, DeviceMeshesNoResharding) {
  Communicator* const communicator = communicator_;
  if (communicator->is_available() == false) {
    GTEST_SKIP() << "distributed setting not available";
  }

  // MODEL DEFINITION
  // Let us define a model expressing a simple memcpy
  Fusion fusion;
  FusionGuard fg(&fusion);

  TensorView* tv0 = makeContigTensor(1);
  fusion.addInput(tv0);
  TensorView* tv1 = set(tv0); // "set" means "copy"
  fusion.addOutput(tv1);

  const bool verbose_print = verbose_ && communicator->deviceId() == 0;

  if (verbose_print) {
    // Print a concise representation of the fusion expressions
    fusion.printMath();

    // Generate and print a CUDA kernel. Notice that at this point the
    // genereated code is just a sequential kernel as we have not
    // scheduled the fusion yet.
    fusion.printKernel();
  }

  // MULTIDEVICE SCHEDULING

  // The first scheduling piece we need is the DeviceMesh. A device mesh in
  // itself is nothing but a multi-dimensional array of integers representing
  // device indices. In this tutorial, we will focus on 1D Meshes, so a
  // DeviceMesh is nothing but a vector of indices. For example, we can consider
  // the DeviceMesh comprised of the device index "0" only
  DeviceMesh mesh_zero({0});
  // or the mesh with device "1" only
  DeviceMesh mesh_one({1});
  // We can also consider the mesh of device indices "0" "2"
  DeviceMesh mesh_zero_and_two({0, 2});
  // Or the mesh containing all available devices in the world communicator
  std::vector<int64_t> all_devices(communicator->size());
  std::iota(
      all_devices.begin(),
      all_devices.end(),
      0); // all_devices = [0,1,..., communicator->size()-1]
  DeviceMesh mesh_full(all_devices);

  // Device meshes are used to indicate the devices on which a Tensor is
  // sharded or replicated. Each tensor can be associated with a DeviceMesh. For
  // example,
  tv0->setDeviceMesh(mesh_zero);
  // means that tv0 is only present on device "0".
  if (verbose_print) {
    std::cout << tv0->toString() << std::endl;
  }
  // Alternatively:
  tv0->setDeviceMesh(mesh_full);
  // means that tv0 is replicated on all the devices.
  if (verbose_print) {
    std::cout << tv0->toString() << std::endl;
  }

  // Let us exercise this notion by examining two simple situations where we set
  // the device mesh. In those simple examples, no network communication will be
  // needed at all. We will simply see that only through setting the device mesh
  // we can, e.g., replicate computation on all devices or select a subset of
  // devices.

  // FULL REPLICATION ON ALL DEVICES
  // Let us set tv0 and tv1 to be replicated on all the devices:
  tv0->setDeviceMesh(mesh_full);
  tv1->setDeviceMesh(mesh_full);

  // RUNTIME
  // Set up the input
  constexpr int64_t tensor_size = 128;
  const c10::TensorOptions tensor_options =
      at::TensorOptions().device(communicator->device()).dtype(at::kFloat);
  // each rank allocate a tensor a on different device
  at::Tensor input = at::randn({tensor_size}, tensor_options);
  {
    // EXECUTION
    MultiDeviceExecutor multidevice_executor(
        std::make_unique<Fusion>(fusion), *communicator);
    if (verbose_print) {
      fusion.printMath();
      fusion.printKernel();
      multidevice_executor.print();
    }

    // The same computation is replicated on all the devices
    at::Tensor output = multidevice_executor.runWithInput({input}).at(0);

    // VALIDATION
    // Each device produces the full output
    EXPECT_TRUE(output.equal(input));
  }

  // SINGLE DEVICE EXECUTION
  // This indicates that tv0 and tv1 are present on device 0 only
  tv0->setDeviceMesh(mesh_zero);
  tv1->setDeviceMesh(mesh_zero);
  // RUNTIME
  {
    // EXECUTION
    MultiDeviceExecutor multidevice_executor(
        std::make_unique<Fusion>(fusion), *communicator);
    // here, the compute is done on device 0 only. Other devices don't even read
    // the input.
    at::Tensor output = multidevice_executor.runWithInput({input}).at(0);

    // VALIDATION
    // Only device 0 receives a non-void output
    if (communicator->deviceId() == 0) {
      EXPECT_TRUE(output.equal(input));
    } else {
      EXPECT_EQ(output.numel(), 0);
    }
  }
}

// Let us now complexify the above example by showing how setting the tensor's
// device mesh can encode pipelining using multiple devices. Let us define a
// Fusion which will be executed in a two-stage pipeline fashion. Executing this
// pipeling will require a network communication between the two pipeline
// Stages.
TEST_F(MultiDeviceTutorial, SimplePipelining) {
  constexpr int64_t tensor_size = 128;

  Communicator* const communicator = communicator_;
  if (communicator->is_available() == false || communicator->size() < 2) {
    GTEST_SKIP() << "distributed setting is not available";
  }
  if (communicator->deviceId() > 1) { // we only need 2 devices for this test
    return;
  }

  // MODEL DEFINITION
  Fusion fusion;
  FusionGuard fg(&fusion);

  TensorView* tv0 = makeContigConcreteTensor({tensor_size});
  fusion.addInput(tv0);
  TensorView* tv1 = add(tv0, tv0);
  // this will be the separation between our two pipeline stages.
  TensorView* tv2 = set(tv1); // "set" means "copy"
  TensorView* tv3 = mul(tv2, tv2);
  fusion.addOutput(tv3);

  // MULTIDEVICE SCHEDULING
  DeviceMesh mesh_zero({0});
  DeviceMesh mesh_one({1});

  // Let us assign different meshes to, {tv0, tv1} on the one hand, and {tv2,
  // tv3} on the other hand.
  tv0->setDeviceMesh(mesh_zero);
  tv1->setDeviceMesh(mesh_zero);
  tv2->setDeviceMesh(mesh_one);
  tv3->setDeviceMesh(mesh_one);
  // This means that {tv0, tv1} exist on device 0, while {tv2, tv3} exist on
  // device 1. This implies that a network communication needs to be executed.
  // More precisely, to produce tv2, we need device 0 to send tv1 to device 1.

  MultiDeviceExecutor multidevice_executor(
      std::make_unique<Fusion>(fusion), *communicator);
  if (verbose_) {
    std::cout << "Device ID = " << communicator->deviceId() << std::endl;
    multidevice_executor.print();
    // Printing shows that device 0 and device 1 execute different programs.
    // Both devices participate to a "Communication/Wait" which has been added
    // to broadcast tv1 on device 0 to tv2 on device 1. We also see that each
    // device executes a different pipeline stage, represented by "PostOnStream
    // (HostUnit)"

    // Print ouput is reproduced here for convenience:
    // clang-format off
    /*
    Device ID = 0

    %HostIrContainer { (T0_g[ iS0{128} ] (DeviceMesh{0})) -> (T3_g[ iS3{128} ] (DeviceMesh{1})) :
      PostOnStream (HostUnit0, Inputs:{T0_g[ iS0{128} ] (DeviceMesh{0}), }, Outputs:{T1_l[ iS1{128} ] (DeviceMesh{0}), })
      Communication 3 (type=Broadcast, team=(1 0), root=0, Input=T1_l[ iS1{128} ] (DeviceMesh{0}), Output=T2_l[ iS2{128} ] (DeviceMesh{1}))
      Wait Communication 3

    HostUnit0: Inputs={T0_g[ iS0{128} ] (DeviceMesh{0}), } -> Outputs={T1_g[ iS1{128} ] (DeviceMesh{0}), }
    %kernel {
    T1_g[ iS1{128} ] (DeviceMesh{0})
      = T0_g[ iS0{128} ] (DeviceMesh{0})
      + T0_g[ iS0{128} ] (DeviceMesh{0});
    } // %kernel

    } // %HostIrContainer


    Device ID = 1

    %HostIrContainer { (T0_g[ iS0{128} ] (DeviceMesh{0})) -> (T3_g[ iS3{128} ] (DeviceMesh{1})) :
      Communication 1 (type=Broadcast, team=(1 0), root=0, Input=T1_l[ iS1{128} ] (DeviceMesh{0}), Output=T2_l[ iS2{128} ] (DeviceMesh{1}))
      Wait Communication 1
      PostOnStream (HostUnit3, Inputs:{T2_l[ iS2{128} ] (DeviceMesh{1}), }, Outputs:{T3_g[ iS3{128} ] (DeviceMesh{1}), })

    HostUnit3: Inputs={T2_g[ iS2{128} ] (DeviceMesh{1}), } -> Outputs={T3_g[ iS3{128} ] (DeviceMesh{1}), }
    %kernel {
    T3_g[ iS3{128} ] (DeviceMesh{1})
      = T2_g[ iS2{128} ] (DeviceMesh{1})
      * T2_g[ iS2{128} ] (DeviceMesh{1});
    } // %kernel

    } // %HostIrContainer
    */
    // clang-format on
  }

  // RUNTIME
  // Set up the input
  const c10::TensorOptions tensor_options =
      at::TensorOptions().device(communicator->device()).dtype(at::kFloat);
  // each rank allocates a tensor on a different device
  at::Tensor input = at::ones({tensor_size}, tensor_options);
  at::Tensor output = multidevice_executor.runWithInput({input}).at(0);

  // VALIDATION
  if (communicator->deviceId() == 1) {
    at::Tensor ref_output = torch::square(input * 2);
    EXPECT_TRUE(output.equal(ref_output));
  } else {
    EXPECT_EQ(output.numel(), 0);
  }
}

} // namespace nvfuser
