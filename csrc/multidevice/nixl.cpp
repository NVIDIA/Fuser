// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#include "multidevice/nixl.h"

#include <unordered_set>

#ifdef USE_NIXL
#include <nixl.h>
#endif

namespace nvfuser {

// ===================================================================
// NixlTransferHandle
// ===================================================================

class NixlTransferHandleImpl {
 public:
#ifdef USE_NIXL
  nixl_xfer_req_t xfer_handle{};
  bool prepared = false;
  bool posted = false;
#endif
};

NixlTransferHandle::NixlTransferHandle() = default;
NixlTransferHandle::~NixlTransferHandle() = default;
NixlTransferHandle::NixlTransferHandle(NixlTransferHandle&&) noexcept =
    default;
NixlTransferHandle& NixlTransferHandle::operator=(
    NixlTransferHandle&&) noexcept = default;

bool NixlTransferHandle::isValid() const {
  if (!impl_) {
    return false;
  }
#ifdef USE_NIXL
  return impl_->prepared;
#else
  return false;
#endif
}

// ===================================================================
// Tensor validation and descriptor helpers
// ===================================================================

namespace {

void validateCudaTensors(const std::vector<at::Tensor>& tensors) {
  NVF_ERROR(!tensors.empty(), "Tensor list must not be empty");
  for (const auto& t : tensors) {
    NVF_ERROR(t.is_cuda(), "All tensors must be CUDA tensors");
    NVF_ERROR(t.is_contiguous(), "All tensors must be contiguous");
  }
}

#ifdef USE_NIXL
nixl_reg_dlist_t buildRegDlist(const std::vector<at::Tensor>& tensors) {
  nixl_reg_dlist_t dlist(VRAM, tensors.size());
  for (const auto& t : tensors) {
    dlist.addDesc(
        {reinterpret_cast<uintptr_t>(t.data_ptr()),
         static_cast<size_t>(t.numel()) * t.element_size(),
         static_cast<uint32_t>(t.device().index())});
  }
  return dlist;
}

nixl_xfer_dlist_t buildXferDlist(const std::vector<at::Tensor>& tensors) {
  nixl_xfer_dlist_t dlist(VRAM, tensors.size());
  for (const auto& t : tensors) {
    dlist.addDesc(
        {reinterpret_cast<uintptr_t>(t.data_ptr()),
         static_cast<size_t>(t.numel()) * t.element_size(),
         static_cast<uint32_t>(t.device().index())});
  }
  return dlist;
}

nixl_xfer_op_t toNixlXferOp(NixlXferOp op) {
  switch (op) {
    case NixlXferOp::kRead:
      return NIXL_XFER_READ;
    case NixlXferOp::kWrite:
      return NIXL_XFER_WRITE;
  }
  std::unreachable();
}
#endif

} // namespace

// ===================================================================
// NixlBackend::Impl
// ===================================================================

class NixlBackend::Impl {
 public:
  explicit Impl(Communicator& communicator);
  ~Impl();

  bool isAvailable() const {
    return available_;
  }

  void registerTensors(const std::vector<at::Tensor>& tensors);
  void deregisterTensors(const std::vector<at::Tensor>& tensors);
  void exchangeMetadata();

  NixlTransferHandle prepareTransfer(
      const std::vector<at::Tensor>& local_tensors,
      const std::vector<at::Tensor>& remote_tensors,
      int64_t remote_rank,
      NixlXferOp op);

  void postTransfer(NixlTransferHandle& handle);
  NixlXferStatus getTransferStatus(const NixlTransferHandle& handle) const;
  void waitTransfer(NixlTransferHandle& handle);

 private:
#ifdef USE_NIXL
  std::unique_ptr<nixlAgent> agent_;
#endif
  Communicator& communicator_;
  bool available_ = false;
  bool metadata_exchanged_ = false;
};

// -------------------------------------------------------------------
// Construction / destruction
// -------------------------------------------------------------------

NixlBackend::Impl::Impl(Communicator& communicator)
    : communicator_(communicator) {
#ifdef USE_NIXL
  std::string agent_name = constructAgentName(communicator_.deviceId());
  agent_ = std::make_unique<nixlAgent>(agent_name);
  if (!agent_) {
    NVF_THROW("Failed to create NIXL agent");
  }

  nixl_b_params_t params;
  nixl_status_t status = agent_->loadBackend("UCX", &params);
  if (status != NIXL_SUCCESS) {
    agent_.reset();
    NVF_THROW("Failed to load UCX backend for NIXL agent");
    return;
  }

  available_ = true;
#endif
}

NixlBackend::Impl::~Impl() {
#ifdef USE_NIXL
  agent_.reset();
#endif
}

std::string NixlBackend::Impl::constructAgentName(int deviceId){
  return "rank_" + std::to_string(deviceId);
}

// -------------------------------------------------------------------
// Memory registration
// -------------------------------------------------------------------

void NixlBackend::Impl::registerTensors(
    const std::vector<at::Tensor>& tensors) {
#ifdef USE_NIXL
  NVF_ERROR(available_, "NIXL backend is not available");
  validateCudaTensors(tensors);

  nixl_reg_dlist_t dlist = buildRegDlist(tensors);
  nixl_status_t status = agent_->registerMem(dlist);
  NVF_ERROR(
      status == NIXL_SUCCESS,
      "NIXL registerMem failed with status ",
      static_cast<int>(status));

  metadata_exchanged_ = false;
#else
  (void)tensors;
  NVF_THROW("NIXL support not compiled");
#endif
}

void NixlBackend::Impl::deregisterTensors(
    const std::vector<at::Tensor>& tensors) {
#ifdef USE_NIXL
  NVF_ERROR(available_, "NIXL backend is not available");
  validateCudaTensors(tensors);

  nixl_reg_dlist_t dlist = buildRegDlist(tensors);
  nixl_status_t status = agent_->deregisterMem(dlist);
  NVF_ERROR(
      status == NIXL_SUCCESS,
      "NIXL deregisterMem failed with status ",
      static_cast<int>(status));

  metadata_exchanged_ = false;
#else
  (void)tensors;
  NVF_THROW("NIXL support not compiled (USE_NIXL not defined)");
#endif
}

// -------------------------------------------------------------------
// Metadata exchange
// -------------------------------------------------------------------

void NixlBackend::Impl::exchangeMetadata() {
#ifdef USE_NIXL
  NVF_ERROR(available_, "NIXL backend is not available");

  std::string local_md = agent_->getLocalMD();
  auto* store = communicator_.getTcpStore();
  const int64_t my_rank = communicator_.deviceId();
  const int64_t world_size = communicator_.size();

  std::string key_prefix = "nixl_agent_md_rank_";
  store->set(
      key_prefix + std::to_string(my_rank),
      std::vector<uint8_t>(local_md.begin(), local_md.end()));

  for (int64_t rank = 0; rank < world_size; ++rank) {
    if (rank == my_rank) {
      continue;
    }
    auto bytes = store->get(key_prefix + std::to_string(rank));
    std::string remote_md(bytes.begin(), bytes.end());
    nixl_status_t status = agent_->loadRemoteMD(remote_md);
    NVF_ERROR(
        status == NIXL_SUCCESS,
        "NIXL loadRemoteMD failed for rank ",
        rank,
        " with status ",
        static_cast<int>(status));
  }

  // Barrier before deleting keys so no rank reads a deleted key.
  communicator_.barrier();

  store->deleteKey(key_prefix + std::to_string(my_rank));
  metadata_exchanged_ = true;
#else
  NVF_THROW("NIXL support not compiled (USE_NIXL not defined)");
#endif
}

// -------------------------------------------------------------------
// Transfer preparation
// -------------------------------------------------------------------

// Prepare a transfer between local and remote tensor pairs.
//
// The local and remote descriptor lists are built from the tensors'
// data pointers, byte sizes, and CUDA device indices. NIXL pairs
// local_tensors[i] with remote_tensors[i]. The direction depends on `op`:
//   kRead  -- data flows from remote_tensors[i] into local_tensors[i]
//   kWrite -- data flows from local_tensors[i] into remote_tensors[i]
//
// Preconditions:
//   - exchangeMetadata() has been called since the last registration change
//   - local_tensors and remote_tensors have the same length
//   - all tensors are contiguous CUDA tensors
//   - remote tensors must have been registered on remote_rank's agent
NixlTransferHandle NixlBackend::Impl::prepareTransfer(
    const std::vector<at::Tensor>& local_tensors,
    const std::vector<at::Tensor>& remote_tensors,
    int64_t remote_rank,
    NixlXferOp op) {
  NixlTransferHandle handle;
#ifdef USE_NIXL
  NVF_ERROR(available_, "NIXL backend is not available");
  NVF_ERROR(metadata_exchanged_, "exchangeMetadata() must be called first");
  NVF_ERROR(
      local_tensors.size() == remote_tensors.size(),
      "Local and remote tensor lists must have the same size. Got ",
      local_tensors.size(),
      " vs ",
      remote_tensors.size());
  validateCudaTensors(local_tensors);
  validateCudaTensors(remote_tensors);

  std::string remote_agent_name = constructAgentName(remote_rank);

  nixl_xfer_dlist_t local_dlist = buildXferDlist(local_tensors);
  nixl_xfer_dlist_t remote_dlist = buildXferDlist(remote_tensors);

  auto impl = std::make_unique<NixlTransferHandleImpl>();
  nixl_status_t status = agent_->prepXferDlist(
      toNixlXferOp(op),
      local_dlist,
      remote_dlist,
      remote_agent_name,
      impl->xfer_handle);
  NVF_ERROR(
      status == NIXL_SUCCESS,
      "NIXL prepXferDlist failed with status ",
      static_cast<int>(status));

  impl->prepared = true;
  handle.impl_ = std::move(impl);
#else
  (void)local_tensors;
  (void)remote_tensors;
  (void)remote_rank;
  (void)op;
  NVF_THROW("NIXL support not compiled (USE_NIXL not defined)");
#endif
  return handle;
}

// -------------------------------------------------------------------
// Transfer posting
// -------------------------------------------------------------------

void NixlBackend::Impl::postTransfer(NixlTransferHandle& handle) {
#ifdef USE_NIXL
  NVF_ERROR(available_, "NIXL backend is not available");
  NVF_ERROR(handle.isValid(), "Cannot post an invalid transfer handle");
  NVF_ERROR(
      !handle.impl_->posted,
      "Transfer already posted. Wait for completion before re-posting.");

  nixl_status_t status = agent_->postXferReq(handle.impl_->xfer_handle);
  NVF_ERROR(
      status == NIXL_SUCCESS || status == NIXL_IN_PROG,
      "NIXL postXferReq failed with status ",
      static_cast<int>(status));

  handle.impl_->posted = true;
#else
  (void)handle;
  NVF_THROW("NIXL support not compiled (USE_NIXL not defined)");
#endif
}

// -------------------------------------------------------------------
// Transfer status / wait
// -------------------------------------------------------------------

NixlXferStatus NixlBackend::Impl::getTransferStatus(
    const NixlTransferHandle& handle) const {
#ifdef USE_NIXL
  NVF_ERROR(available_, "NIXL backend is not available");
  NVF_ERROR(handle.isValid(), "Cannot query status of an invalid handle");
  NVF_ERROR(handle.impl_->posted, "Transfer has not been posted yet");

  nixl_status_t status = agent_->getXferStatus(handle.impl_->xfer_handle);
  switch (status) {
    case NIXL_SUCCESS:
      return NixlXferStatus::kDone;
    case NIXL_IN_PROG:
      return NixlXferStatus::kInProgress;
    default:
      return NixlXferStatus::kError;
  }
#else
  (void)handle;
  NVF_THROW("NIXL support not compiled (USE_NIXL not defined)");
#endif
}

void NixlBackend::Impl::waitTransfer(NixlTransferHandle& handle) {
#ifdef USE_NIXL
  NVF_ERROR(available_, "NIXL backend is not available");
  NVF_ERROR(handle.isValid(), "Cannot wait on an invalid handle");
  NVF_ERROR(handle.impl_->posted, "Transfer has not been posted yet");

  NixlXferStatus xfer_status;
  do {
    xfer_status = getTransferStatus(handle);
    NVF_ERROR(
        xfer_status != NixlXferStatus::kError,
        "NIXL transfer completed with an error");
  } while (xfer_status == NixlXferStatus::kInProgress);

  handle.impl_->posted = false;
#else
  (void)handle;
  NVF_THROW("NIXL support not compiled (USE_NIXL not defined)");
#endif
}

// ===================================================================
// NixlBackend singleton + public API
// ===================================================================

NixlBackend::NixlBackend()
    : impl_(std::make_unique<Impl>(Communicator::getInstance())) {}

NixlBackend& NixlBackend::getInstance() {
  static auto* instance = new NixlBackend();
  return *instance;
}

void NixlBackend::cleanup() {
  impl_.reset();
}

bool NixlBackend::isAvailable() const {
  return impl_ && impl_->isAvailable();
}

void NixlBackend::registerTensors(const std::vector<at::Tensor>& tensors) {
  impl_->registerTensors(tensors);
}

void NixlBackend::deregisterTensors(const std::vector<at::Tensor>& tensors) {
  impl_->deregisterTensors(tensors);
}

void NixlBackend::exchangeMetadata() {
  impl_->exchangeMetadata();
}

NixlTransferHandle NixlBackend::prepareTransfer(
    const std::vector<at::Tensor>& local_tensors,
    const std::vector<at::Tensor>& remote_tensors,
    int64_t remote_rank,
    NixlXferOp op) {
  return impl_->prepareTransfer(
      local_tensors, remote_tensors, remote_rank, op);
}

void NixlBackend::postTransfer(NixlTransferHandle& handle) {
  impl_->postTransfer(handle);
}

NixlXferStatus NixlBackend::getTransferStatus(
    const NixlTransferHandle& handle) const {
  return impl_->getTransferStatus(handle);
}

void NixlBackend::waitTransfer(NixlTransferHandle& handle) {
  impl_->waitTransfer(handle);
}

} // namespace nvfuser
