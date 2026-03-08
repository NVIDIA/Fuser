// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#include "multidevice/nixl.h"
#include "exceptions.h"

#include <algorithm>
#include <cstring>
#include <iostream>
#include <unordered_map>
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
  explicit NixlTransferHandleImpl(nixlAgent* agent) : agent(agent) {}
  nixlAgent* agent;
  nixlXferReqH* xfer_handle = nullptr;

  ~NixlTransferHandleImpl() {
    if (xfer_handle) {
      agent->releaseXferReq(xfer_handle);
    }
  }
#endif
  bool posted = false;
};

NixlTransferHandle::NixlTransferHandle() = default;
NixlTransferHandle::~NixlTransferHandle() = default;
NixlTransferHandle::NixlTransferHandle(NixlTransferHandle&&) noexcept =
    default;
NixlTransferHandle& NixlTransferHandle::operator=(
    NixlTransferHandle&&) noexcept = default;

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
  nixl_reg_dlist_t dlist(VRAM_SEG);
  for (const auto& t : tensors) {
    dlist.addDesc(
        {reinterpret_cast<uintptr_t>(t.data_ptr()),
         static_cast<size_t>(t.numel()) * t.element_size(),
         static_cast<uint32_t>(t.device().index())});
  }
  return dlist;
}

nixl_xfer_dlist_t buildXferDlist(const std::vector<TensorDesc>& descs) {
  nixl_xfer_dlist_t dlist(VRAM_SEG);
  for (const auto& desc : descs) {
    dlist.addDesc({desc.addr, desc.size, desc.dev});
  }
  return dlist;
}

nixl_xfer_op_t toNixlXferOp(NixlXferOp op) {
  switch (op) {
    case NixlXferOp::kRead:
      return NIXL_READ;
    case NixlXferOp::kWrite:
      return NIXL_WRITE;
  }
  NVF_THROW("Invalid NIXL transfer operation: ", static_cast<int>(op));
}

#endif

} // namespace

// ===================================================================
// NixlBackend::Impl
// ===================================================================

#ifdef USE_NIXL

class NixlBackend::Impl {
 public:
  static std::unique_ptr<Impl> create(Communicator& communicator);
  ~Impl();

  void registerTensors(const std::vector<at::Tensor>& tensors);
  void deregisterTensors(const std::vector<at::Tensor>& tensors);

  NixlTransferHandle prepareTransfer(
      const std::vector<TensorDesc>& local_descs,
      const std::vector<TensorDesc>& remote_descs,
      NixlXferOp op);

  void postTransfer(NixlTransferHandle& handle);
  NixlXferStatus getTransferStatus(const NixlTransferHandle& handle) const;
  void waitTransfer(NixlTransferHandle& handle);

 private:
  void exchangeMetadata();
  explicit Impl(Communicator& communicator);
  inline std::string getAgentName(int64_t device_id);

  std::unique_ptr<nixlAgent> agent_;
  nixlBackendH* backend_ = nullptr;
  Communicator& communicator_;
  bool metadata_exchanged_ = false;
};

// -------------------------------------------------------------------
// Construction / destruction
// -------------------------------------------------------------------

NixlBackend::Impl::Impl(Communicator& communicator)
    : communicator_(communicator) {}

std::unique_ptr<NixlBackend::Impl> NixlBackend::Impl::create(
    Communicator& communicator) {
  std::unique_ptr<Impl> impl(new Impl(communicator));

  std::string agent_name = impl->getAgentName(communicator.deviceId());
  nixlAgentConfig cfg(false);
  impl->agent_ = std::make_unique<nixlAgent>(agent_name, cfg);

  nixl_b_params_t params;
  nixl_status_t status =
      impl->agent_->createBackend("UCX", params, impl->backend_);
  if (status != NIXL_SUCCESS) {
    impl->agent_.reset();
    NVF_THROW("Failed to create UCX backend for NIXL agent");
  }

  // Probe: verify that VRAM (CUDA GPU memory) is actually usable with
  // the UCX backend. Some UCX installations lack CUDA support, causing
  // registerMem to silently misclassify VRAM as host memory. We detect
  // this by registering a small buffer and asking NIXL to prepare a
  // local descriptor list for VRAM -- if no backend claims VRAM, the
  // probe fails and we mark the backend as unavailable.
  {
    constexpr int64_t kProbeBytes = 1;
    auto probe = at::empty(
        {kProbeBytes},
        at::TensorOptions().dtype(at::kByte).device(
            at::kCUDA, communicator.deviceId()));
    size_t nbytes = static_cast<size_t>(probe.nbytes());
    uintptr_t addr = reinterpret_cast<uintptr_t>(probe.data_ptr());
    uint32_t dev_idx = static_cast<uint32_t>(probe.device().index());

    NVF_ERROR(nbytes > 0, "NIXL probe: unexpected zero-byte tensor");
    NVF_ERROR(addr != 0, "NIXL probe: null data pointer");

    nixl_reg_dlist_t reg_dlist(VRAM_SEG);
    reg_dlist.addDesc({addr, nbytes, static_cast<uint64_t>(dev_idx)});

    nixl_status_t reg_status = impl->agent_->registerMem(reg_dlist);
    if (reg_status != NIXL_SUCCESS) {
      return nullptr;
    }

    nixl_xfer_dlist_t xfer_dlist(VRAM_SEG);
    xfer_dlist.addDesc({addr, nbytes, static_cast<uint64_t>(dev_idx)});

    nixlDlistH* dlist_handle = nullptr;
    nixl_status_t prep_status =
        impl->agent_->prepXferDlist(NIXL_INIT_AGENT, xfer_dlist, dlist_handle);

    if (dlist_handle) {
      impl->agent_->releasedDlistH(dlist_handle);
    }
    impl->agent_->deregisterMem(reg_dlist);

    if (prep_status != NIXL_SUCCESS) {
      return nullptr;
    }
  }

  return impl;
}

NixlBackend::Impl::~Impl() = default;

std::string NixlBackend::Impl::getAgentName(int64_t rank) {
  return "rank_" + std::to_string(rank);
}

// -------------------------------------------------------------------
// Memory registration
// -------------------------------------------------------------------

// TODO - consider adding RAII wrapper
void NixlBackend::Impl::registerTensors(
    const std::vector<at::Tensor>& tensors) {
  validateCudaTensors(tensors);

  nixl_reg_dlist_t dlist = buildRegDlist(tensors);
  nixl_status_t status = agent_->registerMem(dlist);
  NVF_ERROR(
      status == NIXL_SUCCESS,
      "NIXL registerMem failed with status ",
      static_cast<int>(status));

  metadata_exchanged_ = false;
  exchangeMetadata();
}

void NixlBackend::Impl::deregisterTensors(
    const std::vector<at::Tensor>& tensors) {
  validateCudaTensors(tensors);

  nixl_reg_dlist_t dlist = buildRegDlist(tensors);
  nixl_status_t status = agent_->deregisterMem(dlist);
  NVF_ERROR(
      status == NIXL_SUCCESS,
      "NIXL deregisterMem failed with status ",
      static_cast<int>(status));

  metadata_exchanged_ = false;
  exchangeMetadata();
}

// -------------------------------------------------------------------
// Metadata exchange
// -------------------------------------------------------------------

void NixlBackend::Impl::exchangeMetadata() {
  nixl_blob_t local_md;
  nixl_status_t md_status = agent_->getLocalMD(local_md);
  NVF_ERROR(
      md_status == NIXL_SUCCESS,
      "NIXL getLocalMD failed with status ",
      static_cast<int>(md_status));

  auto* store = communicator_.getTcpStore();
  const auto my_rank = communicator_.deviceId();
  const auto world_size = communicator_.size();

  std::string md_key_prefix = "nixl_agent_md_rank_";
  store->set(
      md_key_prefix + std::to_string(my_rank),
      std::vector<uint8_t>(local_md.begin(), local_md.end()));

  for (int64_t rank = 0; rank < world_size; ++rank) {
    if (rank == my_rank) {
      continue;
    }
    auto bytes = store->get(md_key_prefix + std::to_string(rank));
    nixl_blob_t remote_md(bytes.begin(), bytes.end());
    std::string remote_agent_name;
    nixl_status_t status = agent_->loadRemoteMD(remote_md, remote_agent_name);
    NVF_ERROR(
        status == NIXL_SUCCESS,
        "NIXL loadRemoteMD failed for rank ",
        rank,
        " with status ",
        static_cast<int>(status));
  }

  // Barrier before deleting keys so no rank reads a deleted key.
  communicator_.barrier();

  store->deleteKey(md_key_prefix + std::to_string(my_rank));
  metadata_exchanged_ = true;
}

// -------------------------------------------------------------------
// Transfer preparation
// -------------------------------------------------------------------

// Prepare a transfer between local and remote tensor pairs.
//
// NIXL pairs local_tensors[i] with remote_tensors[i]. The direction
// depends on `op`:
//   kRead  -- data flows from remote into local
//   kWrite -- data flows from local into remote
//
NixlTransferHandle NixlBackend::Impl::prepareTransfer(
    const std::vector<TensorDesc>& local_descs,
    const std::vector<TensorDesc>& remote_descs,
    NixlXferOp op) {
  NVF_ERROR(metadata_exchanged_, "exchangeMetadata() must be called first");
  NVF_ERROR(
      !remote_descs.empty(),
      "remote_descs must not be empty");
  NVF_ERROR(
      local_descs.size() == remote_descs.size(),
      "Local and remote tensor lists must have the same size. Got ",
      local_descs.size(),
      " vs ",
      remote_descs.size());

  std::string remote_agent_name = getAgentName(remote_descs.at(0).dev);

  nixl_xfer_dlist_t local_dlist = buildXferDlist(local_descs);
  nixl_xfer_dlist_t remote_dlist = buildXferDlist(remote_descs);

  auto impl = std::make_unique<NixlTransferHandleImpl>(agent_.get());
  nixl_status_t status = agent_->createXferReq(
      toNixlXferOp(op),
      local_dlist,
      remote_dlist,
      remote_agent_name,
      impl->xfer_handle);
  NVF_ERROR(
      status == NIXL_SUCCESS,
      "NIXL createXferReq failed with status ",
      static_cast<int>(status));

  NixlTransferHandle handle;
  handle.impl_ = std::move(impl);
  return handle;
}

// -------------------------------------------------------------------
// Transfer posting
// -------------------------------------------------------------------

void NixlBackend::Impl::postTransfer(NixlTransferHandle& handle) {
  NVF_ERROR(handle.impl_, "Transfer handle is empty - was it moved from?");
  NVF_ERROR(
      !handle.impl_->posted,
      "Transfer already posted. Wait for completion before re-posting.");

  nixl_status_t status = agent_->postXferReq(handle.impl_->xfer_handle);
  NVF_ERROR(
      status == NIXL_SUCCESS || status == NIXL_IN_PROG,
      "NIXL postXferReq failed with status ",
      static_cast<int>(status));

  handle.impl_->posted = true;
}

// -------------------------------------------------------------------
// Transfer status / wait
// -------------------------------------------------------------------

NixlXferStatus NixlBackend::Impl::getTransferStatus(
    const NixlTransferHandle& handle) const {
  NVF_ERROR(handle.impl_, "Transfer handle is empty - was it moved from?");
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
}

void NixlBackend::Impl::waitTransfer(NixlTransferHandle& handle) {
  NVF_ERROR(handle.impl_, "Transfer handle is empty - was it moved from?");
  NVF_ERROR(handle.impl_->posted, "Transfer has not been posted yet");

  NixlXferStatus xfer_status;
  do {
    xfer_status = getTransferStatus(handle);
    NVF_ERROR(
        xfer_status != NixlXferStatus::kError,
        "NIXL transfer completed with an error");
    if (xfer_status == NixlXferStatus::kInProgress) {
      std::this_thread::yield();
    }
  } while (xfer_status == NixlXferStatus::kInProgress);

  handle.impl_->posted = false;
}

#else // !USE_NIXL

class NixlBackend::Impl {
 public:
  static std::unique_ptr<Impl> create(Communicator&) { return nullptr; }
  void registerTensors(const std::vector<at::Tensor>&) {
    NVF_THROW("NIXL not available");
  }
  void deregisterTensors(const std::vector<at::Tensor>&) {
    NVF_THROW("NIXL not available");
  }
  NixlTransferHandle prepareTransfer(
      const std::vector<TensorDesc>&,
      const std::vector<TensorDesc>&,
      NixlXferOp) {
    NVF_THROW("NIXL not available");
  }
  void postTransfer(NixlTransferHandle&) {
    NVF_THROW("NIXL not available");
  }
  NixlXferStatus getTransferStatus(const NixlTransferHandle&) const {
    NVF_THROW("NIXL not available");
  }
  void waitTransfer(NixlTransferHandle&) {
    NVF_THROW("NIXL not available");
  }
};

#endif // USE_NIXL

// ===================================================================
// NixlBackend singleton + public API
// ===================================================================

NixlBackend::NixlBackend() {
#ifdef USE_NIXL
  impl_ = Impl::create(Communicator::getInstance());
#endif
}

NixlBackend& NixlBackend::getInstance() {
  static auto* instance = new NixlBackend();
  NVF_CHECK(!instance->cleaned_up_, "NIXL backend has been cleaned up");
  return *instance;
}

void NixlBackend::cleanup() {
  cleaned_up_ = true;
  impl_.reset();
}

bool NixlBackend::isAvailable() const {
  return impl_ != nullptr;
}

void NixlBackend::registerTensors(const std::vector<at::Tensor>& tensors) {
  NVF_CHECK(isAvailable(), "NIXL backend is not available");
  impl_->registerTensors(tensors);
}

void NixlBackend::deregisterTensors(const std::vector<at::Tensor>& tensors) {
  NVF_CHECK(isAvailable(), "NIXL backend is not available");
  impl_->deregisterTensors(tensors);
}

NixlTransferHandle NixlBackend::prepareTransfer(
    const std::vector<TensorDesc>& local_descs,
    const std::vector<TensorDesc>& remote_descs,
    NixlXferOp op) {
  NVF_CHECK(isAvailable(), "NIXL backend is not available");
  return impl_->prepareTransfer(local_descs, remote_descs, op);
}

void NixlBackend::postTransfer(NixlTransferHandle& handle) {
  NVF_CHECK(isAvailable(), "NIXL backend is not available");
  impl_->postTransfer(handle);
}

NixlXferStatus NixlBackend::getTransferStatus(
    const NixlTransferHandle& handle) const {
  NVF_CHECK(isAvailable(), "NIXL backend is not available");
  return impl_->getTransferStatus(handle);
}

void NixlBackend::waitTransfer(NixlTransferHandle& handle) {
  NVF_CHECK(isAvailable(), "NIXL backend is not available");
  impl_->waitTransfer(handle);
}


} // namespace nvfuser