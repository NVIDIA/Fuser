// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on

#include <host_ir/container.h>
#include <host_ir/evaluator.h>
#include <host_ir/host_ir.h>
#include <ir/iostream.h>
#include <multidevice/communicator.h>
#include <ops/all_ops.h>
#include <ops/utils.h>
#include <tests/cpp/multidevice.h>

namespace nvfuser {

class MultiDeviceTutorial : public MultiDeviceTest {
 protected:
  static void SetUpTestSuite() {
    verbose_ = getNvFuserEnv("TUTORIAL_VERBOSE");
  }

  void SetUp() {
    MultiDeviceTest::SetUp();
    if (!communicator_->is_available()) {
      GTEST_SKIP() << "Distributed setting not available. "
                   << "Make sure you are on a node with n>1 GPUs and run "
                   << "`mpirun -np n -x NVFUSER_TUTORIAL_VERBOSE=1 "
                      "tutorial_multidevice`";
    }
  }

 protected:
  static bool verbose_;
};

bool MultiDeviceTutorial::verbose_ = false;

// To run those tests, allocate a node with n>1 GPUs and run:
//
// mpirun -np n -x NVFUSER_TUTORIAL_VERBOSE=1 tutorial_multidevice
//
// We use a SPMD paradigm, where each host process manages one and only device,
// and each device executes the same program. Therefore, the number of process
// "n" aboves needs to be less or equal than the number of GPUs in the node.
//
// The first object we need to introduce is the class `Communicator` which is
// a convenience that is nvFuser's interface to runtime distributed setting
// and backend. Communicator's setup is expansive (because all IP addresses
// need to be exchanged to all ranks) and occupies a port. Therefore, we
// instantiate one Communicator globally and use it accross all tests. There
// is no benefit in having multiple communicator, and this class is thought to
// have a singleton instance. This communicator can be accessed through
// MultiDeviceTest::communicator_
TEST_F(MultiDeviceTutorial, CommunicatorAndC10d) {
  Communicator* communicator = communicator_; // setup by MultiDeviceTest

  // To check if distributed setting is available:
  ASSERT_TRUE(communicator->is_available());

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
  const Team& team = all_devices;
  // - (optionally) the backend type, between UCC and NCCL
  auto backend_type =
      CommunicatorBackend::kNccl; // or CommunicatorBackend::kUcc
  // the backend_type is an optional argument. By default it will choose nccl if
  // available, ucc otherwise. We can check that the requested backend is indeed
  // available
  if (!communicator->isBackendAvailable(backend_type)) {
    GTEST_SKIP() << "Backend not available";
  }

  // The backend can be retrieved from the the communicator, which manages a
  // cache
  c10d::Backend* backend = communicator->getBackendForTeam(team, backend_type);

  // The c10d backend can then be used to execute collectives. This is used
  // internally in nvFuser to execute collectives, but this is typically too low
  // level to be exposed to the user. Let us show for convenience how this torch
  // library is used on a simple example, e.g., "allreduce" a single buffer
  constexpr int64_t kTensorLength = 128;
  // each ranks will allocate a buffer on a different device
  c10::TensorOptions tensor_options =
      at::TensorOptions().device(communicator->device());
  std::vector<at::Tensor> buffers = {at::ones({kTensorLength}, tensor_options)};
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
      at::ones({kTensorLength}, tensor_options) * communicator->size();
  EXPECT_TRUE(buffers.at(0).equal(expected_result));
}

// Now that we have introduce the class Communicator, which is the interface
// between nvFuser and lower-level backend, let us now turn to the higher level
// discussion of multidevice scheduling. The first piece we introduce in the
// following test is the one of Device Mesh, which allows us to select the
// devices on which the tensors will be sharded.
TEST_F(MultiDeviceTutorial, DeviceMeshesNoResharding) {
  // MODEL DEFINITION
  // Let us define a model expressing a simple memcpy
  auto fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());

  TensorView* tv0 = makeContigTensor(1);
  fusion->addInput(tv0);
  TensorView* tv1 = set(tv0); // "set" means "copy"
  fusion->addOutput(tv1);

  const bool verbose_print = verbose_ && communicator_->deviceId() == 0;

  if (verbose_print) {
    // Print a concise representation of the fusion expressions
    fusion->printMath();

    // Generate and print a CUDA kernel. Notice that at this point the
    // genereated code is just a sequential kernel as we have not
    // scheduled the fusion yet.
    fusion->printKernel();
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
  // Or the mesh containing all available devices in the world communicator_,
  // i.e.,[0,1,..., communicator_->size()-1]
  auto mesh_full = DeviceMesh::createForNumDevices(communicator_->size());
  // However, it is forbidden to define a mesh with duplicates indices:
  EXPECT_ANY_THROW(DeviceMesh mesh_with_duplicates({1, 1}));

  // Device meshes are used to indicate the devices on which a Tensor is
  // sharded or replicated. Each tensor can be associated with a DeviceMesh. For
  // example,
  tv0->setDeviceMesh(mesh_zero);
  // means that tv0 is only present on device "0".
  if (verbose_print) {
    std::cout << tv0 << std::endl;
  }
  // Alternatively:
  tv0->setDeviceMesh(mesh_full);
  // means that tv0 is replicated on all the devices.
  if (verbose_print) {
    std::cout << tv0 << std::endl;
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
  constexpr int64_t kTensorSize = 128;
  const c10::TensorOptions tensor_options =
      at::TensorOptions().device(communicator_->device()).dtype(at::kFloat);
  // each rank allocate a tensor a on different device
  at::Tensor input = at::randn({kTensorSize}, tensor_options);
  {
    // EXECUTION
    // This class is responsible for managing a single device (given by
    // communicator_->deviceId()) in a SPMD multidevice program. Recall that
    // each rank manages one and only one GPU.
    MultiDeviceExecutor multidevice_executor(
        std::make_unique<Fusion>(*fusion), *communicator_);
    if (verbose_print) {
      fusion->printMath();
      fusion->printKernel();
      multidevice_executor.print();
    }

    // The same computation is replicated on all the devices
    at::Tensor output =
        multidevice_executor.runWithInput({input})[0].as<at::Tensor>();

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
    MultiDeviceExecutor multidevice_executor(std::move(fusion), *communicator_);
    // here, the compute is done on device 0 only. Other devices don't even read
    // the input's data. However, the shape of the input is used to infer the
    // concrete shape of tv0 and subsequent tensors' shape. Therefore, we still
    // need to give each device inputs with valid shapes.
    at::Tensor output =
        multidevice_executor.runWithInput({input})[0].as<at::Tensor>();

    // VALIDATION
    // Only device 0 receives a non-void output
    if (communicator_->deviceId() == 0) {
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
  // MODEL DEFINITION
  auto fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());

  constexpr int64_t kTensorSize = 128;
  TensorView* tv0 = makeContigConcreteTensor({kTensorSize});
  fusion->addInput(tv0);
  TensorView* tv1 = add(tv0, tv0);
  // this will be the separation between our two pipeline stages.
  TensorView* tv2 = set(tv1); // "set" means "copy"
  TensorView* tv3 = mul(tv2, tv2);
  fusion->addOutput(tv3);

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

  MultiDeviceExecutor multidevice_executor(std::move(fusion), *communicator_);
  if (verbose_ && communicator_->deviceId() < 2) {
    std::cout << "Device ID = " << communicator_->deviceId() << std::endl;
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
      at::TensorOptions().device(communicator_->device()).dtype(at::kFloat);
  // each rank allocates a tensor on a different device
  at::Tensor input = at::ones({kTensorSize}, tensor_options);
  at::Tensor output =
      multidevice_executor.runWithInput({input})[0].as<at::Tensor>();

  // VALIDATION
  if (communicator_->deviceId() == 1) {
    at::Tensor ref_output = torch::square(input * 2);
    EXPECT_TRUE(output.equal(ref_output));
  } else {
    EXPECT_EQ(output.numel(), 0);
  }
}

// While DeviceMesh allows us to select on which device a Tensor is
// materialized, we are so far only able to either fully replicate a tensor or
// not materialize it at all on given devices. Let us now introduce a new
// scheduling primitive which allows to shard tensors accross devices. This new
// primitive consists of a new parallel type `ParallelType::DIDx`, applied to a
// tensor's axis by doing tv->axis(0)->parallelize(ParallelType::DIDx). This is
// similar to how Fuser classically sets parallel strategy, using
// ParallelType::TIDx (for parallelizing an axis accross threads) and
// ParallelType::BIDx (for parallelizing an axis accross blocks). Here, "D" in
// DIDx stands for "device", and this parallel type indicates we want to
// parallelize an axis accross devices. Let us consider for example a tensor tv
// of shape {4,128}, assigned with the device mesh (0,1,2,3), which outermost
// axis is parallelized with DIDx. It means that each device will materialize a
// tensor of shape {1,128} representing a slice of the global tensor.

// Note 1: the extent of the DeviceMesh always needs to match the tensor's axis
// extent. This is similar to the fact that, in Fuser single device, if an axis
// of extent 32 is parallelized with TIDx, then the launch param blockDim
// (i.e. the number of threads per blocks) will be chosen equal to 32.

// Note 2: sharding a tensor (i.e. parallelizing an axis with DIDx) is
// meaningless without an underlying device mesh, because otherwise we cannot
// guess on what devices the tensor needs to be sharded. This is in apparence
// different to BIDx and TIDx parallel, where we don't access the physical
// threads and blocks to map to.
TEST_F(MultiDeviceTutorial, TensorShardingAndResharding) {
  // MODEL DEFINITION
  // Let us define a model expressing a simple memcpy
  auto fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());

  TensorView* tv0 = makeContigTensor(2);
  fusion->addInput(tv0);
  TensorView* tv1 = set(tv0); // "set" means "copy"
  fusion->addOutput(tv1);

  // MULTIDEVICE SCHEDULING
  // Let us define, as in previous tests, a 1D Device Mesh comprised of all
  // available device IDs, i.e., [0,1,..., communicator_->size()-1]
  auto mesh_full = DeviceMesh::createForNumDevices(communicator_->size());

  // Let us set tv0 and tv1's mesh:
  tv0->setDeviceMesh(mesh_full);
  tv1->setDeviceMesh(mesh_full);
  // Without further parallelization, this means that the tensors are fully
  // replicated on all the devices. Let us now introduce some parallelization to
  // describe how the tensors are sharded onto this device mesh.

  // ################################
  // Fully sharded with no resharding
  // ################################
  // Let us shard the tensors' outermost axis onto the device mesh:
  tv0->axis(0)->parallelize(ParallelType::DIDx);
  tv1->axis(0)->parallelize(ParallelType::DIDx);
  // It means that each device will own a slice of the the full tensors. In
  // particular, it implies that tv0's and tv1's outermost axis extent equals
  // the number of devices. The shapes of tv0 and tv1 are thus
  // [communicator_->size(), ?]
  const bool verbose_print = verbose_ && communicator_->deviceId() == 0;
  if (verbose_print) {
    std::cout << "tv0: " << tv0 << std::endl;
    std::cout << "tv1: " << tv1 << std::endl;
  }
  // However, the outermost axis with extent `communicator_->size()`is not
  // materialized on one device, but accross devices, therefore, each device
  // allocate a tensor of shape {1, ?}, representing a slice of the global
  // tensor.

  // RUNTIME
  // Set up the input
  constexpr int64_t kTensorSize = 128;
  const auto tensor_options =
      at::TensorOptions().device(communicator_->device()).dtype(at::kFloat);
  // each rank allocate a tensor a on different device.
  // Note here that the outermost axis needs to have extent 1 because
  // tv0->axis(0) is sharded
  at::Tensor input = at::randn({1, kTensorSize}, tensor_options);
  {
    // EXECUTION
    // Note that each device only copies a slice of the tv0 to a slice of tv1
    // (i.e. there is no loop over the outermost axis, which is anyway of extent
    // "1" as far as the device is concerned). Also, note that since tv0's and
    // tv1's sharding are consistent, no resharding is necessary, i.e., no
    // inter-device communication is needed; Executing the fusion is purely
    // local and consists of a simple kernel.
    MultiDeviceExecutor multidevice_executor(
        std::make_unique<Fusion>(*fusion), *communicator_);
    if (verbose_print) {
      fusion->printMath();
      fusion->printKernel();
      multidevice_executor.print();
    }

    // Each device is responsible for a fraction of the total compute
    at::Tensor output =
        multidevice_executor.runWithInput({input})[0].as<at::Tensor>();

    // VALIDATION
    // Each device produces a slice of the global output, which is also sharded
    // accross devices.
    EXPECT_TRUE(output.equal(input));
  }

  // ################################
  // Allgather
  // ################################
  // Let us now introduce a non-consistent sharding between tv0 and tv1 that
  // will result in an "allgather". Let us, as before set tv0's and tv1's mesh
  // to be the full device mesh
  tv0->setDeviceMesh(mesh_full);
  tv1->setDeviceMesh(mesh_full);

  // Let us also shard tv0 but, contrarily to what we considered before, let us
  // replicate (aka "not shard") tv1 (i.e. parallelize with
  // "ParallelType::Serial")
  tv0->axis(0)->parallelize(ParallelType::DIDx);
  tv1->axis(0)->parallelize(ParallelType::Serial);
  {
    // EXECUTION
    MultiDeviceExecutor multidevice_executor(
        std::make_unique<Fusion>(*fusion), *communicator_);
    // Since the input is sharded and the output is replicated, a network
    // communication is needed to share the data between devices. Here, a
    // "MPI-Allgather" communication is needed.
    if (verbose_print) {
      multidevice_executor.print();
      // Printout is reproduced here for convenience, run on 8 devices:
      // clang-format off
      /*
        %HostIrContainer { (T0_g[ ideviceIdx.x0{i0}, iS1{i2} ] (DeviceMesh{0 1 2 3 4 5 6 7})) -> (T1_g[ iS2{i0}, iS3{i2} ] (DeviceMesh{0 1 2 3 4 5 6 7})) :
          Communication 1 (type=Allgather, team=(0 1 2 3 4 5 6 7), Input=T0_g[ ideviceIdx.x0{i0}, iS1{i2} ] (DeviceMesh{0 1 2 3 4 5 6 7}), Output=T1_g[ iS2{i0}, iS3{i2} ] (DeviceMesh{0 1 2 3 4 5 6 7}))
          Wait Communication 1
        } // %HostIrContainer
      */
      // clang-format on
    }

    at::Tensor output =
        multidevice_executor.runWithInput({input})[0].as<at::Tensor>();

    // VALIDATION
    EXPECT_TRUE(
        output
            .slice(0, communicator_->deviceId(), communicator_->deviceId() + 1)
            .equal(input));
  }

  // ################################
  // Gather
  // ################################
  // To emulate the scenario of a "Gather", we need to change device mesh
  // between tv0 and tv1 in addition to changing the parallel type
  DeviceMesh mesh_zero({0});
  tv0->setDeviceMesh(mesh_full);
  tv1->setDeviceMesh(mesh_zero);

  tv0->axis(0)->parallelize(ParallelType::DIDx);
  tv1->axis(0)->parallelize(ParallelType::Serial);

  // This sharding indicates that tv0 is sharded accross all devices, while tv1
  // is fully materialized on device 0. To pass from tv0 to tv1, the devices
  // need to participate to a "MPI-Gather" operation rooted in "0"
  {
    // EXECUTION
    MultiDeviceExecutor multidevice_executor(
        std::make_unique<Fusion>(*fusion), *communicator_);
    if (verbose_print) {
      multidevice_executor.print();
      // Printout is reproduced here for convenience, run on 8 devices:
      // clang-format off
      /*
        %HostIrContainer { (T0_g[ ideviceIdx.x0{i0}, iS1{i2} ] (DeviceMesh{0 1 2 3 4 5 6 7})) -> (T1_g[ iS2{i0}, iS3{i2} ] (DeviceMesh{0})) :
          Communication 1 (type=Gather, team=(0 1 2 3 4 5 6 7), root=0, Input=T0_g[ ideviceIdx.x0{i0}, iS1{i2} ] (DeviceMesh{0 1 2 3 4 5 6 7}), Output=T1_g[ iS2{i0}, iS3{i2} ] (DeviceMesh{0}))
          Wait Communication 1
        } // %HostIrContainer
      */
      // clang-format on
    }

    at::Tensor output =
        multidevice_executor.runWithInput({input})[0].as<at::Tensor>();

    // VALIDATION
    if (communicator_->deviceId() == 0) {
      // device 0 produces the full output, which one slice corresponds to
      // device 0's input
      EXPECT_TRUE(
          output
              .slice(
                  0, communicator_->deviceId(), communicator_->deviceId() + 1)
              .equal(input));
    } else {
      // Other devices do not produce any output
      EXPECT_EQ(output.numel(), 0);
    }
  }

  // ################################
  // Scatter
  // ################################
  // Let us now consider the "opposite" situation where we start from a tensor
  // which is fully materialized on 0 but end up being sharded accross all
  // devices
  tv0->setDeviceMesh(mesh_zero);
  tv1->setDeviceMesh(mesh_full);

  tv0->axis(0)->parallelize(ParallelType::Serial);
  tv1->axis(0)->parallelize(ParallelType::DIDx);
  // To execute, devices need to perform a "MPI-scatter" collective rooted on
  // "0"
  {
    // EXECUTION
    MultiDeviceExecutor multidevice_executor(std::move(fusion), *communicator_);
    if (verbose_print) {
      multidevice_executor.print();
      // Printout is reproduced here for convenience, run on 8 devices:
      // clang-format off
      /*
        %HostIrContainer { (T0_g[ iS0{i0}, iS1{i2} ] (DeviceMesh{0})) -> (T1_g[ ideviceIdx.x2{i0}, iS3{i2} ] (DeviceMesh{0 1 2 3 4 5 6 7})) :
          Communication 1 (type=Scatter, team=(0 1 2 3 4 5 6 7), root=0, Input=T0_g[ iS0{i0}, iS1{i2} ] (DeviceMesh{0}), Output=T1_g[ ideviceIdx.x2{i0}, iS3{i2} ] (DeviceMesh{0 1 2 3 4 5 6 7}))
          Wait Communication 1
        } // %HostIrContainer
      */
      // clang-format on
    }

    // Note here that, contrarily to what we saw before, the first axis extent
    // must not be "1" but must equal the number of devices.
    input = at::randn({communicator_->size(), kTensorSize}, tensor_options);

    at::Tensor output =
        multidevice_executor.runWithInput({input})[0].as<at::Tensor>();

    // VALIDATION
    // Each device receives a slice of the global input.
    if (communicator_->deviceId() == 0) {
      //  Device 0, which owns the full input, can compare the obtained output
      //  with a certain slice of the input.
      EXPECT_TRUE(output.equal(input.slice(
          0, communicator_->deviceId(), communicator_->deviceId() + 1)));
    } else {
      EXPECT_EQ(output.sizes()[0], 1);
      EXPECT_EQ(output.sizes()[1], input.sizes()[1]);
    }
  }
}

// We have seen how to multidevice-schedule a Fusion by setting tensors'
// sharding, i.e., setting device mesh and setting axis parallel type to DIDx.

// At the time where this comment is written, we only support 1D tensor
// sharding, i.e., we have not introduced yet multidimensional meshes and other
// parallel types such as DIDy and DIDz. This will be done in the future. Let us
// poit out by the way that 1D multidevice parallelism already covers
// interesting real-world scenarios, such as Transformer with Tensor Parallel,
// see `tests/cpp/test_multidevice_transformer.cpp`.

// Let us now explore lower than the mere multidevice scheduling API and
// introduce the host IR. In classical single-device nvFuser, the "host program"
// could be summarized as launching one or a couple of CUDA Kernels. When
// dealing with multiple devices, the host plays a more proeminent role because
// it needs to orchestrate and synchronize compute Kernels and Communications
// (which are necessarily CPU-initiated when using NCCL or UCC). Other examples,
// not necessarily tied to multidevice setting, rely on complex host
// orchestration, such as multi streaming, kernel pipelining, overlap technics,
// using CUDA Graphs, etc.

// These examples suggest that Fuser should be able to reason about the
// interplay between host and device execution. This motivates the introduction
// of "host IRs" to represent the host program, in an abstract/symbolic way,
// allowing us reason about the host program, apply optimizations passes, and
// possibly compile it (in the future).

// The HostIr component is comprised of three parts:
// - Host IRs: each IR represents an elementary host operation
// - HostIrContainer: represents the host program
// - HostIrEvaluator: evaluates a HostIrContainer

// The host program is typically generated automatically during lowering. This
// is what is done at the instantiation of MultiDeviceExecutor, and what gets
// printed by `MultiDeviceExecutor::print()`. However, the Host Ir API has been
// designed to allow a fully manual host programmation. This is what we are
// going to introduce in the following tests.

//  Let us start with the simplest non-trivial host program possible: compiling
//  and running a single Fusion. It is a good starting point to understand the
//  Host Ir semantics. The host program could be illustrated as follows:
/*
  | tv0: input
  | tv1 = Fusion (tv0)
  | tv1: output
*/

namespace hir { // HostIr has its own namespace "hir"

TEST_F(MultiDeviceTutorial, HostIrLaunchingFusion) {
  // Let us consider an arbitrary fusion. We assume for simplicity (but without
  // loss of generality) that the fusion has one input and one output which are
  // both a 2D tensor.
  constexpr int64_t kNDims = 2;
  auto create_fusion = []() -> std::unique_ptr<Fusion> {
    auto fusion = std::make_unique<Fusion>();
    FusionGuard fg(fusion.get());
    TensorView* tv0 = makeSymbolicTensor(kNDims);
    fusion->addInput(tv0);
    auto tv1 = mul(tv0, IrBuilder::create<Val>(2.));
    tv1->setMemoryType(
        MemoryType::Global); // to avoid an error of the type "Allocations must
                             // be based on constant integers for local memory"
    auto tv2 = add(tv1, IrBuilder::create<Val>(1.));
    fusion->addOutput(tv2);
    return fusion;
  };

  // Instantiate an HostIrContainer. This container is used to 1) register the
  // Host IRs, and 2) represent the Host program through its `std::vector<Expr*>
  // top_level_exprs_`.
  auto hic = std::make_unique<HostIrContainer>();
  FusionGuard fg(hic.get());

  // The first Host IR we introduce is called `HostUnit`. It is used to
  // represent a fusion definition at the host IR level.
  auto fusion = IrBuilder::create<HostUnit>(create_fusion());

  // We then create an IR `PostOnStream` which represents compiling+executing
  // the fusion with some I/O.
  TensorView* input = makeSymbolicTensor(kNDims);
  TensorView* output = makeSymbolicTensor(kNDims);
  auto post_fusion = IrBuilder::create<PostOnStream>(
      fusion, std::vector<Val*>({input}), std::vector<Val*>({output}));

  // Let us add "post_fusion" to the host program, and define the program's
  // global I/O. These step will probably be automated out and simplified in the
  // future
  hic->pushBackTopLevelExprs(post_fusion);
  hic->addInput(input);
  hic->addOutput(output);

  if (verbose_ && communicator_->deviceId() == 0) {
    hic->print(debug());
    // We reproduce, for convenience, what gets printed:
    // clang-format off
    /*
    %HostIrContainer { (T0_g[ iS0{i0}, iS1{i2} ]) -> (T1_g[ iS2{i3}, iS3{i4} ]) :
      PostOnStream (HostUnit0, Inputs:{T0_g[ iS0{i0}, iS1{i2} ], }, Outputs:{T1_g[ iS2{i3}, iS3{i4} ], })

    HostUnit0: [...]
    } // %HostIrContainer
    */
    // clang-format on
    //  the "[...]" contains the result of Fusion::printMath(), which we omit
    //  here.
  }

  // define a concrete input
  at::Tensor aten_input =
      at::randn({16, 32}, at::TensorOptions().device(communicator_->device()));

  // Let us now execute the Host program.
  HostIrEvaluator hie(std::move(hic));
  auto outputs = hie.runWithInput({{input, aten_input}});

  // validate the result
  EXPECT_TRUE(torch::allclose(2 * aten_input + 1, outputs[0].as<at::Tensor>()));
}

// Let us now present a case where the host program consists of launching three
// fusions, with a non-linear dependency between the respective I/O. The host
// program could be illustrated as follows:
/*
  | tv0: input
  | (tv1, tv2) = Fusion0 (tv0)
  | tv3 = Fusion1 (tv1)
  | tv4 = Fusion2 (tv2, tv3)
  | tv5 = Fusion1 (tv4)
  | tv5: output
*/

TEST_F(MultiDeviceTutorial, HostIrLaunchingThreeFusions) {
  // Let us consider an arbitrary fusion. We assume for simplicity (but without
  // loss of generality) that the fusion has one input and one output which are
  // both a 2D tensor.
  constexpr int64_t kNDims = 2;
  auto create_fusion_0 = []() -> std::unique_ptr<Fusion> {
    auto fusion = std::make_unique<Fusion>();
    FusionGuard fg(fusion.get());
    TensorView* tv0 = makeSymbolicTensor(kNDims);
    fusion->addInput(tv0);
    auto tv1 = mul(tv0, IrBuilder::create<Val>(2.));
    fusion->addOutput(tv1);
    auto tv2 = add(tv1, IrBuilder::create<Val>(1.));
    fusion->addOutput(tv2);
    return fusion;
  };
  auto create_fusion_1 = []() -> std::unique_ptr<Fusion> {
    auto fusion = std::make_unique<Fusion>();
    FusionGuard fg(fusion.get());
    TensorView* tv1 = makeSymbolicTensor(kNDims);
    fusion->addInput(tv1);
    auto tv3 = add(tv1, IrBuilder::create<Val>(2.));
    fusion->addOutput(tv3);
    return fusion;
  };
  auto create_fusion_2 = []() -> std::unique_ptr<Fusion> {
    auto fusion = std::make_unique<Fusion>();
    FusionGuard fg(fusion.get());
    TensorView* tv2 = makeSymbolicTensor(kNDims);
    TensorView* tv3 = makeSymbolicTensor(kNDims);
    fusion->addInput(tv2);
    fusion->addInput(tv3);
    auto tv4 = add(tv2, tv3);
    fusion->addOutput(tv4);
    return fusion;
  };

  // Instantiate an HostIrContainer
  auto hic = std::make_unique<HostIrContainer>();
  FusionGuard fg(hic.get());

  // Create the HostUnit holding the fusions
  auto fusion0 = IrBuilder::create<HostUnit>(create_fusion_0());
  auto fusion1 = IrBuilder::create<HostUnit>(create_fusion_1());
  auto fusion2 = IrBuilder::create<HostUnit>(create_fusion_2());

  // Create TensorViews that are dealt with at the host level
  TensorView* tv0 = makeSymbolicTensor(kNDims);
  TensorView* tv1 = makeSymbolicTensor(kNDims);
  TensorView* tv2 = makeSymbolicTensor(kNDims);
  TensorView* tv3 = makeSymbolicTensor(kNDims);
  TensorView* tv4 = makeSymbolicTensor(kNDims);
  TensorView* tv5 = makeSymbolicTensor(kNDims);

  // Create the IR representing executing the fusion with some I/O
  auto post_fusion0 = IrBuilder::create<PostOnStream>(
      fusion0,
      /*inputs=*/std::vector<Val*>({tv0}),
      /*outputs=*/std::vector<Val*>({tv1, tv2}));
  auto post_fusion1 = IrBuilder::create<PostOnStream>(
      fusion1,
      /*inputs=*/std::vector<Val*>({tv1}),
      /*outputs=*/std::vector<Val*>({tv3}));
  auto post_fusion2 = IrBuilder::create<PostOnStream>(
      fusion2,
      /*inputs=*/std::vector<Val*>({tv2, tv3}),
      /*outputs=*/std::vector<Val*>({tv4}));
  // Note that the same host unit can be reused multiple times
  auto post_fusion1_bis = IrBuilder::create<PostOnStream>(
      fusion1,
      /*inputs=*/std::vector<Val*>({tv4}),
      /*outputs=*/std::vector<Val*>({tv5}));

  // Let us create the host program and its global I/O
  hic->pushBackTopLevelExprs(post_fusion0);
  hic->pushBackTopLevelExprs(post_fusion1);
  hic->pushBackTopLevelExprs(post_fusion2);
  hic->pushBackTopLevelExprs(post_fusion1_bis);

  hic->addInput(tv0);
  hic->addOutput(tv5);

  if (verbose_ && communicator_->deviceId() == 0) {
    hic->print(debug());
    // We reproduce, for convenience, what gets printed:
    // clang-format off
    /*
    %HostIrContainer { (T0_g[ iS0{i0}, iS1{i2} ]) -> (T5_g[ iS10{i11}, iS11{i12} ]) :
      PostOnStream (HostUnit0, Inputs:{T0_g[ iS0{i0}, iS1{i2} ], }, Outputs:{T1_l[ iS2{i3}, iS3{i4} ], T2_l[ iS4{i5}, iS5{i6} ], })
      PostOnStream (HostUnit1, Inputs:{T1_l[ iS2{i3}, iS3{i4} ], }, Outputs:{T3_l[ iS6{i7}, iS7{i8} ], })
      PostOnStream (HostUnit2, Inputs:{T2_l[ iS4{i5}, iS5{i6} ], T3_l[ iS6{i7}, iS7{i8} ], }, Outputs:{T4_l[ iS8{i9}, iS9{i10} ], })
      PostOnStream (HostUnit1, Inputs:{T4_l[ iS8{i9}, iS9{i10} ], }, Outputs:{T5_g[ iS10{i11}, iS11{i12} ], })

    HostUnit2: [...]
    HostUnit1: [...]
    HostUnit0: [...]
    } // %HostIrContainer
    */
    // clang-format on
    //  the "[...]" contains the result of Fusion::printMath(), which we omit
    //  here.
  }

  // define a concrete input
  at::Tensor aten_tv0 =
      at::randn({16, 32}, at::TensorOptions().device(communicator_->device()));

  // Let us now execute the Host program.
  HostIrEvaluator hie(std::move(hic));
  auto outputs = hie.runWithInput({{tv0, aten_tv0}});

  // validate the result
  EXPECT_TRUE(torch::allclose(4 * aten_tv0 + 5, outputs[0].as<at::Tensor>()));
}

// Let us now present a real world scenario, used in transformer, where we need
// to execute a matmul followed by a Reduce-Scatter MPI collective. This is the
// first example we provide where host IRs are used in a multidevice setting.
// The host program can be summarized as follows:
/*
  | tva, tvb: inputs
  | tvc = Matmul(tva, tvb)
  | tvd = Reduce-Scatter (tvc)
  | Wait for the completion of Reduce-Scatter
  | tvd: output
*/
// here, all the tensors are implicitely sharded tensors accross devices.
// `Matmul` and MPI-collectives are Host IRs that can be used directly in the
// host program.
TEST_F(MultiDeviceTutorial, HostIrGemmReduceScatter) {
  // Instantiate an HostIrContainer
  auto hic = std::make_unique<HostIrContainer>();
  FusionGuard fg(hic.get());

  constexpr int64_t kNDims = 2;
  TensorView* tva = makeContigTensor(kNDims);
  TensorView* tvb = makeContigTensor(kNDims);
  // some ops, like MatMulOp are natively supported as HostIrs, and do not need
  // to be implemented as a Fusion
  TensorView* tvc = matmul(tva, tvb);
  Expr* matmul_op = tvc->definition();

  TensorView* tvd = makeContigTensor(kNDims);
  // Before defining the communication (reduce-scatter) that produces tvd from
  // tvc, it is required to set tvd and tvc device mesh (this might be removed
  // in the future)
  auto mesh_full = DeviceMesh::createForNumDevices(communicator_->size());
  tvc->setDeviceMesh(mesh_full);
  tvd->setDeviceMesh(mesh_full);

  auto* reduce_scatter = IrBuilder::create<Communication>(
      CommunicationType::ReduceScatter,
      /*out=*/tvd,
      /*in=*/tvc,
      /*team=*/mesh_full.vector(),
      /*(unused)root=*/-1,
      RedOpType::SUM);

  // Since communications are non-blocking, it is always required to wait for a
  // posted communication. Node that "wait" blocks the stream but not the CPU
  // (except for barrier)
  auto* wait = IrBuilder::create<hir::Wait>(reduce_scatter);

  // Let us create the host program and the I/O
  hic->pushBackTopLevelExprs(matmul_op);
  hic->pushBackTopLevelExprs(reduce_scatter);
  hic->pushBackTopLevelExprs(wait);

  hic->addInput(tva);
  hic->addInput(tvb);
  hic->addInput(tvd); // a buffer must be provided for tvd, this is why it is
                      // tagged as an input
  hic->addOutput(tvd);

  if (verbose_ && communicator_->deviceId() == 0) {
    hic->print(debug());
    // We reproduce, for convenience, what gets printed:
    // clang-format off
    /*
    %HostIrContainer { (T0_g[ iS0{i0}, iS1{i2} ], T1_g[ iS2{i3}, iS3{i4} ], T3_g[ iS7{i11}, iS8{i12} ] (DeviceMesh{0 1 2 3 4 5 6 7})) -> (T3_g[ iS7{i11}, iS8{i12} ] (DeviceMesh{0 1 2 3 4 5 6 7})) :
      T2_l[ iS4{i0}, iS5{i4}, rS6{i2} ] (DeviceMesh{0 1 2 3 4 5 6 7})
        = matmul(T0_g[ iS0{i0}, iS1{i2} ],
                  T1_g[ iS2{i3}, iS3{i4} ])
      Communication 1 (type=ReduceScatter, team=(0 1 2 3 4 5 6 7), input=T2_l[ iS4{i0}, iS5{i4}, rS6{i2} ] (DeviceMesh{0 1 2 3 4 5 6 7}), output=T3_g[ iS7{i11}, iS8{i12} ] (DeviceMesh{0 1 2 3 4 5 6 7}))
      Wait Communication 1
    } // %HostIrContainer
    */
    // clang-format on
  }

  // define a concrete input
  constexpr int64_t M = 1680;
  constexpr int64_t K = 16;
  constexpr int64_t N = 64;
  ASSERT_EQ(M % communicator_->size(), 0)
      << "the test must be launched with a number of devices n that divides M="
      << M << ", but we have n=" << communicator_->size();

  auto options = at::TensorOptions().device(communicator_->device());
  at::Tensor aten_tva = at::randn({M, K}, options);
  at::Tensor aten_tvb = at::randn({K, N}, options);
  at::Tensor aten_tvd = at::empty({M / communicator_->size(), N}, options);

  // Let us now execute the Host program. When multidevice is requested, we
  // need to pass a pointer to a Communicator
  HostIrEvaluator hie(std::move(hic), communicator_);
  auto outputs =
      hie.runWithInput({{tva, aten_tva}, {tvb, aten_tvb}, {tvd, aten_tvd}});

  // "validate" the result
  EXPECT_EQ(
      outputs[0].as<at::Tensor>().numel(), (M * N) / communicator_->size());
}

// Let us now show how to implement Kernel Pipelining with Host IR. Let us
// consider a program summarized as
/*
  | tv0: input
  | tv1 = Fusion0 (tv0)
  | tv2 = Fusion1 (tv1)
  | tv2: output
*/
// and assume it is pointwise on the outermost dimension (i.e. tv0, tv1 and tv2
// share the "same" outermost axis) What we want to achieve is to tile the
// tensor and to iterate over the tiled dimension with a host for-loop, changing
// stream at each iteration:
/*
  | tv0: input
  | tv2: pre-allocated output
  | For (int i=0; i < tv0->axis(0)->extent; i++):
  |   Change Stream
    // here [i, ...] means we select index i on the outermost dim
  |   tv1_i = Fusion0 (tv0[i,...])
  |   tv2[i,...] = Fusion1 (tv1_i)
*/
// To do so, we will be using new Host IRs: Stream (a Val), SetStream, ForLoop.
TEST_F(MultiDeviceTutorial, HostIrKernekPipelining) {
  constexpr int64_t kNDims = 2;
  constexpr int64_t kPipelineAxis = 0;
  constexpr int64_t kNumberOfStreams = 4;

  auto create_fusion_0 = []() -> std::unique_ptr<Fusion> {
    auto fusion = std::make_unique<Fusion>();
    FusionGuard fg(fusion.get());
    TensorView* tv0 = makeSymbolicTensor(
        kNDims - 1); // kNDims-1 because the fusion receives a "selected" tensor
    fusion->addInput(tv0);
    auto tv1 = add(tv0, IrBuilder::create<Val>(1.));
    fusion->addOutput(tv1);
    return fusion;
  };
  auto create_fusion_1 = []() -> std::unique_ptr<Fusion> {
    auto fusion = std::make_unique<Fusion>();
    FusionGuard fg(fusion.get());
    TensorView* tv1 = makeSymbolicTensor(
        kNDims - 1); // kNDims-1 because the fusion receives a "selected" tensor
    TensorView* tv2 = makeSymbolicTensor(kNDims - 1);
    fusion->addInput(tv1);
    fusion->addInput(tv2);
    auto tv2_alias = add(tv1, IrBuilder::create<Val>(2.));
    fusion->addOutput(tv2_alias);
    fusion->aliasOutputToInput(tv2_alias, tv2, AllocationType::ReuseBuffer);
    return fusion;
  };

  // Instantiate an HostIrContainer
  auto hic = std::make_unique<HostIrContainer>();
  FusionGuard fg(hic.get());

  TensorView* tv0 = makeSymbolicTensor(kNDims);
  TensorView* tv2 = makeSymbolicTensor(kNDims);

  // Let us create the main for-loop of the program. Its index will be used in
  // the for-loop's body.
  auto* index = IrBuilder::create<Val>(DataType::Index);
  auto* for_loop = IrBuilder::create<ForLoop>(
      tv2->axis(0),
      index,
      /*start=*/hic->zeroVal(),
      /*stop=*/tv2->axis(0)->extent(),
      /*step=*/hic->oneVal(),
      /*vectorize=*/false,
      /*vectorize_shift=*/nullptr,
      /*unroll_required=*/false,
      CircularBufferLoopStage::NotApplicable,
      /*circular_buffer_loop_stage_depth=*/0);

  // We select a new stream in a round-robin fashion
  auto* stream_index = mod(index, IrBuilder::create<Val>(kNumberOfStreams));
  auto* set_stream = IrBuilder::create<hir::SetCurrentStream>(
      IrBuilder::create<hir::Stream>(stream_index));

  // Let us create the host-level TensorViews
  TensorView* tv0_i = select(tv0, /*dim=*/kPipelineAxis, index);
  auto* tv1_i =
      ops::newValLike(tv0_i, tv0_i->getDataType().value())->as<TensorView>();
  TensorView* tv2_i = select(tv2, /*dim=*/kPipelineAxis, index);

  // We post the two fusions with appropriate I/O
  auto post_fusion0 = IrBuilder::create<PostOnStream>(
      IrBuilder::create<HostUnit>(create_fusion_0()),
      /*inputs=*/std::vector<Val*>({tv0_i}),
      /*outputs=*/std::vector<Val*>({tv1_i}));
  // Note that tv2_i is both an Input and an Output here. The I/O are actually
  // aliased in the Fusion. We do so because the global buffer tv2 is
  // preallocated and tv2_i is a selection of this global buffer.
  auto post_fusion1 = IrBuilder::create<PostOnStream>(
      IrBuilder::create<HostUnit>(create_fusion_1()),
      /*inputs=*/std::vector<Val*>({tv1_i, tv2_i}),
      /*outputs=*/std::vector<Val*>({tv2_i}));

  // Let us add the different Exprs in the for-loop's body
  for_loop->body().push_back(set_stream);
  for_loop->body().push_back(tv0_i->definition());
  for_loop->body().push_back(tv2_i->definition());
  for_loop->body().push_back(post_fusion0);
  for_loop->body().push_back(post_fusion1);

  // The host program consists of the for-loop only
  hic->pushBackTopLevelExprs(for_loop);

  hic->addInput(tv0);
  hic->addInput(tv2);
  hic->addOutput(tv2);

  if (verbose_ && communicator_->deviceId() == 0) {
    hic->print(debug());
    // We reproduce, for convenience, what gets printed:
    // clang-format off
    /*
    %HostIrContainer { (T0_g[ iS0{i0}, iS1{i2} ], T1_g[ iS2{i3}, iS3{i4} ]) -> (T1_g[ iS2{i3}, iS3{i4} ]) :
      FOR i5 in iS2{i3}:
        SetCurrentStream to Stream ( i5 % 4 )
        T2_l[ iS4{i2} ]
          = select( T0_g[ iS0{i0}, iS1{i2} ], axis = iS0{i0}, index = i5 )
        T4_l[ iS6{i4} ]
          = select( T1_g[ iS2{i3}, iS3{i4} ], axis = iS2{i3}, index = i5 )
        PostOnStream (HostUnit5, Inputs:{T2_l[ iS4{i2} ], }, Outputs:{T3_l[ iS5{i2} ], })
        PostOnStream (HostUnit7, Inputs:{T3_l[ iS5{i2} ], T4_l[ iS6{i4} ], }, Outputs:{T4_l[ iS6{i4} ], })

    HostUnit5: [...]
    HostUnit7: [...]
    } // %HostIrContainer
    */
    // clang-format on
    //  the "[...]" contains the result of Fusion::printMath(), which we omit
    //  here.
  }

  // define a concrete input
  constexpr int64_t kSize = 128;
  constexpr int64_t kNumberOfTiles = 16;

  auto options = at::TensorOptions().device(communicator_->device());
  at::Tensor aten_tv0 = at::randn({kNumberOfTiles, kSize}, options);
  at::Tensor aten_tv2 = at::empty({kNumberOfTiles, kSize}, options);

  // Let us now execute the Host program. We indicate to use FusionExecutorCache
  // to execute the fusions -- this way, we don't need to recompile at each
  // iteration.
  HostIrEvaluator hie(
      std::move(hic),
      /*communicator=*/nullptr,
      {.use_fusion_executor_cache = true});
  auto outputs = hie.runWithInput({{tv0, aten_tv0}, {tv2, aten_tv2}});

  // validate the result
  EXPECT_TRUE(torch::allclose(outputs[0].as<at::Tensor>(), aten_tv0 + 3));
}

} // namespace hir

} // namespace nvfuser
