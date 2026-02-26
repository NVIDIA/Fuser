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
  // TODO - is it leaking when handleimpl is destroyed ? 
  nixlXferReqH* xfer_handle = nullptr;
#endif
  bool prepared = false;
  bool posted = false;
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
      const std::vector<TensorDesc>& local_descs, 
      const std::vector<TensorDesc>& remote_descs,
      int64_t remote_rank,
      NixlXferOp op);

  void postTransfer(NixlTransferHandle& handle);
  NixlXferStatus getTransferStatus(const NixlTransferHandle& handle) const;
  void waitTransfer(NixlTransferHandle& handle);

 private:
  std::string constructAgentName(int64_t rank);

#ifdef USE_NIXL
  std::unique_ptr<nixlAgent> agent_;
  nixlBackendH* backend_ = nullptr;
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
  nixlAgentConfig cfg(false);
  agent_ = std::make_unique<nixlAgent>(agent_name, cfg);

  nixl_b_params_t params;
  nixl_status_t status = agent_->createBackend("UCX", params, backend_);
  if (status != NIXL_SUCCESS) {
    agent_.reset();
    NVF_THROW("Failed to create UCX backend for NIXL agent");
  }

  // Probe: verify that VRAM (CUDA GPU memory) is actually usable with
  // the UCX backend. Some UCX installations lack CUDA support, causing
  // registerMem to silently misclassify VRAM as host memory. We detect
  // this by registering a small buffer and asking NIXL to prepare a
  // local descriptor list for VRAM -- if no backend claims VRAM, the
  // probe fails and we mark the backend as unavailable.
  {
    constexpr int64_t kProbeBytes = 64;
    auto probe = at::empty(
        {kProbeBytes},
        at::TensorOptions().dtype(at::kByte).device(
            at::kCUDA, communicator_.deviceId()));
    size_t nbytes = static_cast<size_t>(probe.nbytes());
    uintptr_t addr = reinterpret_cast<uintptr_t>(probe.data_ptr());
    uint32_t dev_idx = static_cast<uint32_t>(probe.device().index());

    std::cerr << "[NixlBackend probe] device=" << dev_idx
              << " addr=0x" << std::hex << addr << std::dec
              << " nbytes=" << nbytes
              << " numel=" << probe.numel()
              << " element_size=" << probe.element_size() << std::endl;

    NVF_ERROR(nbytes > 0, "NIXL probe: unexpected zero-byte tensor");
    NVF_ERROR(addr != 0, "NIXL probe: null data pointer");

    nixl_reg_dlist_t reg_dlist(VRAM_SEG);
    reg_dlist.addDesc({addr, nbytes, static_cast<uint64_t>(dev_idx)});

    std::cerr << "[NixlBackend probe] reg_dlist desc: addr=0x" << std::hex
              << reg_dlist[0].addr << std::dec
              << " len=" << reg_dlist[0].len
              << " devId=" << reg_dlist[0].devId << std::endl;

    nixl_status_t reg_status = agent_->registerMem(reg_dlist);
    std::cerr << "[NixlBackend probe] registerMem returned "
              << reg_status << std::endl;
    if (reg_status != NIXL_SUCCESS) {
      return;
    }

    nixl_xfer_dlist_t xfer_dlist(VRAM_SEG);
    xfer_dlist.addDesc({addr, nbytes, static_cast<uint64_t>(dev_idx)});

    nixlDlistH* dlist_handle = nullptr;
    nixl_status_t prep_status =
        agent_->prepXferDlist(NIXL_INIT_AGENT, xfer_dlist, dlist_handle);
    std::cerr << "[NixlBackend probe] prepXferDlist returned "
              << prep_status << std::endl;

    if (dlist_handle) {
      agent_->releasedDlistH(dlist_handle);
    }
    agent_->deregisterMem(reg_dlist);

    if (prep_status != NIXL_SUCCESS) {
      return;
    }
  }

  available_ = true;
#endif
}

NixlBackend::Impl::~Impl() = default;

std::string NixlBackend::Impl::constructAgentName(int64_t rank){
  return "rank_" + std::to_string(rank);
}

// -------------------------------------------------------------------
// Memory registration
// -------------------------------------------------------------------

// TODO - consider adding RAII wrapper
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
    // Fetch & load MD 
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
#else
  NVF_THROW("NIXL support not compiled (USE_NIXL not defined)");
#endif
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
// remote_tensors are LOCAL tensors whose data_ptr identifies the
// corresponding registration slot. The actual remote addresses are
// looked up from the descriptors exchanged during exchangeMetadata().
// This requires all ranks to register tensors in the same order.
NixlTransferHandle NixlBackend::Impl::prepareTransfer(
    const std::vector<TensorDesc>& local_descs, // Local addresses
    const std::vector<TensorDesc>& remote_descs, // Remote tensors (not valid on this rank)
    int64_t remote_rank,
    NixlXferOp op) {
  NixlTransferHandle handle;
#ifdef USE_NIXL
  NVF_ERROR(available_, "NIXL backend is not available");
  NVF_ERROR(metadata_exchanged_, "exchangeMetadata() must be called first");
  NVF_ERROR(
      local_descs.size() == remote_descs.size(),
      "Local and remote tensor lists must have the same size. Got ",
      local_descs.size(),
      " vs ",
      remote_descs.size());

  std::string remote_agent_name = constructAgentName(remote_rank);

  nixl_xfer_dlist_t local_dlist = buildXferDlist(local_descs);
  nixl_xfer_dlist_t remote_dlist = buildXferDlist(remote_descs);

  auto impl = std::make_unique<NixlTransferHandleImpl>();
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

  impl->prepared = true;
  handle.impl_ = std::move(impl);
#else
  (void)local_descs;
  (void)remote_descs;
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

  // TODO - check this spin loop
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
  NVF_CHECK(!instance->cleaned_up_, "NIXL backend has been cleaned up");
  return *instance;
}

void NixlBackend::cleanup() {
  cleaned_up_ = true;
  impl_.reset();
}

bool NixlBackend::isAvailable() const {
  return impl_ && impl_->isAvailable();
}

void NixlBackend::registerTensors(const std::vector<at::Tensor>& tensors) {
  NVF_CHECK(isAvailable(), "NIXL backend is not available");
  impl_->registerTensors(tensors);
}

void NixlBackend::deregisterTensors(const std::vector<at::Tensor>& tensors) {
  NVF_CHECK(isAvailable(), "NIXL backend is not available");
  impl_->deregisterTensors(tensors);
}

void NixlBackend::exchangeMetadata() {
  NVF_CHECK(isAvailable(), "NIXL backend is not available");
  impl_->exchangeMetadata();
}

NixlTransferHandle NixlBackend::prepareTransfer(
    const std::vector<TensorDesc>& local_descs,
    const std::vector<TensorDesc>& remote_descs,
    int64_t remote_rank,
    NixlXferOp op) {
  NVF_CHECK(isAvailable(), "NIXL backend is not available");
  return impl_->prepareTransfer(
      local_descs, remote_descs, remote_rank, op);
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