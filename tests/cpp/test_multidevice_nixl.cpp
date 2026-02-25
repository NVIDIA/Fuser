// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#include <gtest/gtest.h>

#include "multidevice/nixl.h"
#include "tests/cpp/multidevice.h"

namespace nvfuser {

using NixlTest = MultiDeviceTest;

// -------------------------------------------------------------------
// NixlTransferHandle tests
// -------------------------------------------------------------------

TEST_F(NixlTest, TransferHandleDefaultConstruction) {
  NixlTransferHandle handle;
  EXPECT_FALSE(handle.isValid());
}

TEST_F(NixlTest, TransferHandleMoveConstruction) {
  NixlTransferHandle h1;
  EXPECT_FALSE(h1.isValid());

  NixlTransferHandle h2(std::move(h1));
  EXPECT_FALSE(h2.isValid());
}

TEST_F(NixlTest, TransferHandleMoveAssignment) {
  NixlTransferHandle h1;
  NixlTransferHandle h2;
  h2 = std::move(h1);
  EXPECT_FALSE(h2.isValid());
}

// -------------------------------------------------------------------
// NixlBackend singleton tests
// -------------------------------------------------------------------

TEST_F(NixlTest, SingletonIsAccessible) {
  NixlBackend& backend = NixlBackend::getInstance();
  // isAvailable() returns true only when USE_NIXL is defined and the
  // UCX backend loaded successfully. Either outcome is valid here.
  (void)backend.isAvailable();
}

// -------------------------------------------------------------------
// Input validation tests (these exercise the guards in the impl)
// -------------------------------------------------------------------

TEST_F(NixlTest, RegisterEmptyTensorListThrows) {
  NixlBackend& backend = NixlBackend::getInstance();
  if (!backend.isAvailable()) {
    GTEST_SKIP() << "NIXL backend not available";
  }

  std::vector<at::Tensor> empty;
  EXPECT_THROW(backend.registerTensors(empty), nvfError);
}

TEST_F(NixlTest, RegisterCpuTensorThrows) {
  NixlBackend& backend = NixlBackend::getInstance();
  if (!backend.isAvailable()) {
    GTEST_SKIP() << "NIXL backend not available";
  }

  auto cpu_tensor = at::randn({64});
  EXPECT_THROW(backend.registerTensors({cpu_tensor}), nvfError);
}

TEST_F(NixlTest, RegisterNonContiguousTensorThrows) {
  NixlBackend& backend = NixlBackend::getInstance();
  if (!backend.isAvailable()) {
    GTEST_SKIP() << "NIXL backend not available";
  }

  auto t = at::randn({8, 8}, tensor_options_);
  auto non_contig = t.transpose(0, 1);
  ASSERT_FALSE(non_contig.is_contiguous());
  EXPECT_THROW(backend.registerTensors({non_contig}), nvfError);
}

TEST_F(NixlTest, DeregisterEmptyTensorListThrows) {
  NixlBackend& backend = NixlBackend::getInstance();
  if (!backend.isAvailable()) {
    GTEST_SKIP() << "NIXL backend not available";
  }

  std::vector<at::Tensor> empty;
  EXPECT_THROW(backend.deregisterTensors(empty), nvfError);
}

// -------------------------------------------------------------------
// Transfer preparation validation
// -------------------------------------------------------------------

TEST_F(NixlTest, PrepareTransferWithoutMetadataExchangeThrows) {
  NixlBackend& backend = NixlBackend::getInstance();
  if (!backend.isAvailable()) {
    GTEST_SKIP() << "NIXL backend not available";
  }

  auto local = at::randn({64}, tensor_options_);
  auto remote = at::randn({64}, tensor_options_);
  backend.registerTensors({local});
  backend.registerTensors({remote});

  EXPECT_THROW(
      (void)backend.prepareTransfer({toTensorDesc(local)}, {toTensorDesc(remote)}, 0, NixlXferOp::kRead),
      nvfError);

  backend.deregisterTensors({local});
  backend.deregisterTensors({remote});
}

TEST_F(NixlTest, PrepareTransferMismatchedSizesThrows) {
  NixlBackend& backend = NixlBackend::getInstance();
  if (!backend.isAvailable()) {
    GTEST_SKIP() << "NIXL backend not available";
  }

  auto t1 = at::randn({64}, tensor_options_);
  auto t2 = at::randn({64}, tensor_options_);
  auto t3 = at::randn({64}, tensor_options_);
  backend.registerTensors({t1, t2, t3});
  backend.exchangeMetadata();

  EXPECT_THROW(
      (void)backend.prepareTransfer({toTensorDesc(t1), toTensorDesc(t2)}, {toTensorDesc(t3)}, 0, NixlXferOp::kRead), nvfError);

  backend.deregisterTensors({t1, t2, t3});
}

// -------------------------------------------------------------------
// Post / wait on invalid handles
// -------------------------------------------------------------------

TEST_F(NixlTest, PostInvalidHandleThrows) {
  NixlBackend& backend = NixlBackend::getInstance();
  if (!backend.isAvailable()) {
    GTEST_SKIP() << "NIXL backend not available";
  }

  NixlTransferHandle invalid_handle;
  EXPECT_THROW(backend.postTransfer(invalid_handle), nvfError);
}

TEST_F(NixlTest, WaitInvalidHandleThrows) {
  NixlBackend& backend = NixlBackend::getInstance();
  if (!backend.isAvailable()) {
    GTEST_SKIP() << "NIXL backend not available";
  }

  NixlTransferHandle invalid_handle;
  EXPECT_THROW(backend.waitTransfer(invalid_handle), nvfError);
}

TEST_F(NixlTest, GetStatusInvalidHandleThrows) {
  NixlBackend& backend = NixlBackend::getInstance();
  if (!backend.isAvailable()) {
    GTEST_SKIP() << "NIXL backend not available";
  }

  NixlTransferHandle invalid_handle;
  EXPECT_THROW((void)backend.getTransferStatus(invalid_handle), nvfError);
}

// -------------------------------------------------------------------
// End-to-end transfer test (requires >= 2 devices with NIXL)
// -------------------------------------------------------------------

TEST_F(NixlTest, ReadTransferEndToEnd) {
  NixlBackend& backend = NixlBackend::getInstance();
  if (!backend.isAvailable()) {
    GTEST_SKIP() << "NIXL backend not available";
  }
  if (communicator_->size() < 2) {
    GTEST_SKIP() << "Need at least 2 devices for transfer test";
  }

  const int64_t rank = communicator_->deviceId();
  const int64_t world_size = communicator_->size();
  const int64_t peer_rank = (rank + 1) % world_size;
  constexpr int64_t kSize = 1024;

  // Ring style transfer: each rank reads <src> from its peer's remote tensor to its local <dst>.
  auto src = at::full({kSize}, static_cast<float>(rank + 1), tensor_options_);
  auto dst = at::zeros({kSize}, tensor_options_);
  cudaDeviceSynchronize();

  backend.registerTensors({src, dst});
  backend.exchangeMetadata();
  
  // Fetch the remote tensor descriptor from the peer
  std::string src_key_prefix = "nixl_test_read_transfer_src_rank_";
  storeTensorDescs(*communicator_, src_key_prefix + std::to_string(rank), {src});
  auto remote_src_descs = fetchTensorDescs(*communicator_, src_key_prefix + std::to_string(peer_rank));
  communicator_->barrier();
  communicator_->getTcpStore()->deleteKey(src_key_prefix + std::to_string(rank));
  auto remote_src_desc = remote_src_descs[0]; // Only one remote tensor is expected

  // Each rank reads from its peer. After the read, local should contain
  // the values that the peer stored in *its* remote tensor.
  auto handle = backend.prepareTransfer(
      {toTensorDesc(dst)}, {remote_src_desc}, peer_rank, NixlXferOp::kRead);
  ASSERT_TRUE(handle.isValid());

  backend.postTransfer(handle);
  backend.waitTransfer(handle);

  auto local_cpu = dst.cpu();
  float expected_val = static_cast<float>(peer_rank + 1);
  EXPECT_TRUE(at::allclose(local_cpu, at::full({kSize}, expected_val)));

  backend.deregisterTensors({dst, src});
}

TEST_F(NixlTest, WriteTransferEndToEnd) {
  NixlBackend& backend = NixlBackend::getInstance();
  if (!backend.isAvailable()) {
    GTEST_SKIP() << "NIXL backend not available";
  }
  if (communicator_->size() < 2) {
    GTEST_SKIP() << "Need at least 2 devices for transfer test";
  }

  const int64_t rank = communicator_->deviceId();
  const int64_t world_size = communicator_->size();
  const int64_t peer_rank = (rank + 1) % world_size;
  constexpr int64_t kSize = 512;

  // Each rank writes its local <src> to the remote <dst> of its peer in a ring style
  auto src = at::full({kSize}, static_cast<float>(rank + 1), tensor_options_);
  auto dst = at::zeros({kSize}, tensor_options_);
  cudaDeviceSynchronize();

  backend.registerTensors({src, dst});
  backend.exchangeMetadata();

  // Fetch the remote tensor descriptor from the peer
  std::string dst_key_prefix = "nixl_test_write_transfer_dst_rank_";
  storeTensorDescs(*communicator_, dst_key_prefix + std::to_string(rank), {dst});
  auto remote_dst_descs = fetchTensorDescs(*communicator_, dst_key_prefix + std::to_string(peer_rank));
  communicator_->barrier();
  communicator_->getTcpStore()->deleteKey(dst_key_prefix + std::to_string(rank));
  auto remote_dst_desc = remote_dst_descs[0]; // Only one remote tensor is expected

  // Each rank writes its local tensor into its peer's remote tensor.
  auto handle = backend.prepareTransfer(
      {toTensorDesc(src)}, {remote_dst_desc}, peer_rank, NixlXferOp::kWrite);
  ASSERT_TRUE(handle.isValid());

  backend.postTransfer(handle);
  backend.waitTransfer(handle);

  // After a barrier, the peer should have written into our remote tensor <dst>.
  communicator_->barrier();

  auto remote_cpu = dst.cpu();
  int64_t writer_rank = (rank - 1 + world_size) % world_size;
  float expected_val = static_cast<float>(writer_rank + 1);
  EXPECT_TRUE(at::allclose(remote_cpu, at::full({kSize}, expected_val)));

  backend.deregisterTensors({src, dst});
}

TEST_F(NixlTest, RegisterDeregisterRoundTrip) {
  NixlBackend& backend = NixlBackend::getInstance();
  if (!backend.isAvailable()) {
    GTEST_SKIP() << "NIXL backend not available";
  }

  auto t1 = at::randn({256}, tensor_options_);
  auto t2 = at::randn({128}, tensor_options_);

  backend.registerTensors({t1, t2});
  backend.deregisterTensors({t1, t2});

  // Re-registering the same tensors should succeed.
  backend.registerTensors({t1, t2});
  backend.deregisterTensors({t1, t2});
}

} // namespace nvfuser
