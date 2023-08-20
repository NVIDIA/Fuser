// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#pragma once
#ifdef USE_DISTRIBUTED

#include <multidevice/communicator.h>
#include <multidevice/multidevice.h>

namespace nvfuser {

/*
This file implements the class "Collective" which represents a MPI collective
communication operation to be executed on the network. The base class Collective
should not be used directly but through its derived classes:
Broadcast, Gather, Scatter, Allgather. Other collectives will be added later.

Later, Collective could be made a derived class of Expr and be thought
as a kernel IRs resulting of the lowering of a PipelineCommunication.

The flow for using this class is as follows:
1. After instantiation of a derived class, specify the argument of the
   collective through the class's interface using the methods "setCommunicator",
   "addDevice", "addSrcBuf", "addDstBuf", "setRoot" (if applicable). Each rank
   (associated with a device index) will fill the args differently, depending on
   the role they play in this collective. In particular, the ranks not
   participating in the collective should not instantiate it.
2. The method "post" triggers the execution of the collective. This call is
   non-blocking. After this call, the args of the collective cannot be changed
   (and the methods listed in 1. will throw). Once the collective has complete,
   it can be posted again (the initialization overhead occurs only on the first
   post)
3. Once posted, "test" allows to test for completion of the collective
   The method "wait" blocks until completion.
*/

class TORCH_CUDA_CU_API Collective {
 public:
  virtual ~Collective() = default;

  std::string toString(int indent = 0) const;

  // Set the backend communicator to be used for executing the collective
  void setCommunicator(Communicator* comm) {
    assertIsNotInit();
    comm_ = comm;
  }

  // Add a device index to the team that will perform the collective
  void addDevice(const DeviceIdxType device) {
    assertIsNotInit();
    team_.push_back(device);
  }

  // Add a source buffer
  void addSrcBuf(at::Tensor buf) {
    assertIsNotInit();
    src_bufs_.push_back(buf);
  }

  // Add a destination buffer
  void addDstBuf(at::Tensor buf) {
    assertIsNotInit();
    dst_bufs_.push_back(buf);
  }

  // Set the root of the collective (if applicable)
  void setRoot(const DeviceIdxType root) {
    TORCH_INTERNAL_ASSERT(has_root_, "this collective must not be rooted");
    assertIsNotInit();
    root_ = root;
  }

  // Triggers the execution of the collective. This is a non-blocking call.
  // Once this method is called, the argument of the collective cannot be
  // modified. Once the collective has complete, it can be posted again. The
  // initialization overhead occurs only on the first post
  virtual void post() final;

  // Returns true if the collective has complete, false if it is in progress.
  // This method can only be called after the collective has been posted.
  bool test();

  // Blocks until completion
  // This method can only be called after the collective has been posted.
  void wait();

 protected:
  // name is for printing purposes and has_root indicates is the collective is
  // rooted
  Collective(std::string name, bool has_root)
      : collective_type_(std::move(name)), has_root_(has_root){};

 private:
  void assertIsNotInit() {
    TORCH_INTERNAL_ASSERT(
        !is_init_, "this method cannot be called after the collective is init");
  }

  // Perform the prelimiary set up before the collective can be posted.
  // Once called, the arguments of the collective cannot be modified.
  // After the method has been called once, the collective can be posted
  // multiple times without the need of calling again this method
  virtual void init() final;
  // implemented in the derived classes, called inside "init"
  // performs the initialization specific to the derived class
  virtual void init_specialized() {
    TORCH_INTERNAL_ASSERT(false, "not implemented in the base class");
  }

  // implemented in the derived classes, called inside "post"
  virtual void post_specialized() {
    TORCH_INTERNAL_ASSERT(false, "not implemented in the base class");
  }

 protected:
  // Stores the world communicator
  Communicator* comm_ = nullptr;
  DeviceIdxType root_ = 0;
  std::vector<at::Tensor> src_bufs_;
  std::vector<at::Tensor> dst_bufs_;
  // Stores all the device indices that will participate in the collective
  std::vector<DeviceIdxType> team_;
  c10::intrusive_ptr<c10d::Work> work_ = nullptr;
  // Stores the team's backend that will perform the collective
  c10::intrusive_ptr<c10d::Backend> backend_ = nullptr;
  // utility buffer used in Gather and Scatter derived classes
  std::vector<std::vector<at::Tensor>> buf_list_;
  // stores the index of the root in the team
  DeviceIdxType root_rank_ = 0;

 private:
  // used for printing
  std::string collective_type_;
  // track if the collective has already been initialized
  bool is_init_ = false;
  // indicates if the collective is rooted
  bool has_root_ = false;
};

/*
Copies the unique src buffer of the root to the unique dst buffer of the other
device For convenience, we allow the root to have a unique dst buffer as well,
in which we perform a local copy when "post()" is called. Note that this extends
the classic MPI definition of Broadcast

Requirements:
  - the communicator is set
  - the root is set
  - the root has one src buffer, and zero or one dst buffer
  - non-roots have no src buffer and exactly one dst buffer
  - All buffers have the same size
*/
class TORCH_CUDA_CU_API Broadcast : public Collective {
 public:
  Broadcast() : Collective("broadcast", true){};

 private:
  void init_specialized() override;
  void post_specialized() override;
};

/*
Copies each of the unique src buffer of each device to the respective src
buffers of the root. The order of the sender devices matches the order of the
root's buffers

Requirements:
  - the communicator is set
  - the root is set
  - the root has one src buffer and <team_size> dst buffers
  - non-roots have one src buffer and no dst buffer
  - All buffers have the same size
*/
class TORCH_CUDA_CU_API Gather : public Collective {
 public:
  Gather() : Collective("gather", true){};

 private:
  void init_specialized() override;
  void post_specialized() override;
};

/*
Copies each of the unique src buffer of each device to the respective src
buffers of each device. The order of the devices matches the order of the
buffers

Requirements:
  - the communicator is set
  - all device have one src buffer and <team_size> dst buffers
  - All buffers have the same size
*/
class TORCH_CUDA_CU_API Allgather : public Collective {
 public:
  Allgather() : Collective("allgather", false){};

 private:
  void init_specialized() override;
  void post_specialized() override;
};

/*
Copies each of the <team_size> src buffers of the root to the unique dst buffer
of the respective receiver device. The order of the buffer matches the order of
the receiver devices

Requirements:
  - the communicator is set
  - the root is set
  - the root has <team_size> src buffers and one dst buffer
  - non-roots have no src buffer and one dst buffer
  - All buffers have the same size
*/
class TORCH_CUDA_CU_API Scatter : public Collective {
 public:
  Scatter() : Collective("scatter", true){};

 private:
  void init_specialized() override;
  void post_specialized() override;
};

} // namespace nvfuser

#endif
