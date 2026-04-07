// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on

#include <debug.h>
#include <global_allocator.h>
#include <options.h>
#include <type.h>

#include <ATen/ATen.h>

namespace nvfuser {

namespace {

// For each device, we maintain an arena tensor which we will slice to provide
// individual tensors. These tensors will grow in size and remain at the
// high-water mark for their particular device until the thread terminates.
class Arena {
 public:
  // Mark allocated_bytes_ as 0, allowing all available zeroed memory to be
  // reused on subsequent calls to getTensor().
  void reset() {
    if (isDebugDumpEnabled(DebugDumpOption::GlobalZeroedMemory)) {
      debug() << "[global zeroed memory] Resetting allocated bytes to 0"
              << std::endl;
    }
    allocated_bytes_ = 0;
    tensor_.reset();
  }

  at::Tensor getTensor(
      const std::vector<int64_t>& sizes,
      const c10::ScalarType& aten_dtype,
      const c10::Device& device) {
    // determine number of bytes needed for this tensor
    int64_t new_bytes = dataTypeSizeByte(aten_to_data_type(aten_dtype));
    for (auto sz : sizes) {
      new_bytes *= sz;
    }

    // align at 16 bytes regardless of requested dtype
    int64_t aligned_allocated_bytes = (allocated_bytes_ + 15) & (~15);

    // after this function returns this will be the allocated size
    int64_t new_allocated_bytes = aligned_allocated_bytes + new_bytes;

    // resize tensor_ if needed. Minimum size is 128B.
    int64_t new_used_bytes = std::max((int64_t)128LL, tensor_.numel());
    while (new_used_bytes < new_allocated_bytes) {
      new_used_bytes *= 2;
    }
    if (new_used_bytes > tensor_.numel()) {
      if (isDebugDumpEnabled(DebugDumpOption::GlobalZeroedMemory)) {
        debug() << "[global zeroed memory] Resizing arena to " << new_used_bytes
                << " bytes" << std::endl;
      }
      tensor_ = at::zeros(
          {new_used_bytes},
          at::TensorOptions().dtype(at::kByte).device(device));
    }

    if (isDebugDumpEnabled(DebugDumpOption::GlobalZeroedMemory)) {
      debug() << "[global zeroed memory] Allocating byte range: "
              << aligned_allocated_bytes << " to " << new_used_bytes << " bytes"
              << std::endl;
    }

    // Check that memory is zeroed before allocating. Note that this launches
    // another kernel, so it is disabled for release builds.
#ifndef NDEBUG
    checkZeroed();
#endif

    allocated_bytes_ = new_allocated_bytes;

    // slice and view tensor
    return tensor_
        .index({at::indexing::Slice(
            aligned_allocated_bytes, new_allocated_bytes, 1)})
        .view(aten_dtype)
        .view(sizes);
  }

 private:
  void checkZeroed() const {
    c10::Scalar nnz = at::count_nonzero(tensor_).item();
    NVF_ERROR(
        nnz.equal(0),
        "Global memory arena was not properly zeroed. Found ",
        nnz,
        " bytes that are not zero");
  }

 private:
  at::Tensor tensor_;
  int64_t allocated_bytes_ = 0;
};

// We hold one Arena for each device
thread_local std::vector<Arena> arenas;

} // namespace

at::Tensor contigZeroedTensor(
    const std::vector<int64_t>& sizes,
    const c10::ScalarType& aten_dtype,
    const c10::Device& device) {
  NVF_ERROR(device.is_cuda(), "contigZeroTensor requires CUDA device");
  // Intermediate cast from int8_t to uint8_t for clarity:
  // https://clang.llvm.org/extra/clang-tidy/checks/bugprone/signed-char-misuse.html
  size_t device_num = (uint8_t)device.index();

  // get arena from device number, resizing arenas if needed
  if (device_num >= arenas.size()) {
    arenas.resize(device_num + 1);
  }

  // request tensor from arena
  return arenas[device_num].getTensor(sizes, aten_dtype, device);
}

// Note that this does not free allocated zeroed memory, but rather it marks all
// zeroed memory as available for re-use.
void releaseZeroedMemory() {
  for (Arena& a : arenas) {
    a.reset();
  }
}

} // namespace nvfuser
