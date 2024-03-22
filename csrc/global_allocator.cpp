// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on

#include <global_allocator.h>
#include <type.h>

#include <ATen/ATen.h>

namespace nvfuser {

namespace {

// For each device, we maintain an arena tensor which we will slice to provide
// individual tensors. These tensors will grow in size and remain at the
// high-water mark for their particular device until the thread terminates.
class Arena {
 public:
  void reset() {
    allocated_bytes_ = 0LL;
  }

  at::Tensor getTensor(
      const std::vector<int64_t>& sizes,
      const c10::ScalarType& aten_dtype,
      const c10::Device& device) {
    // determine number of bytes needed for this tensor
    int64_t new_bytes = dataTypeSize(aten_to_data_type(aten_dtype));
    for (auto sz : sizes) {
      new_bytes *= sz;
    }

    // align at 16 bytes regardless of requested dtype
    int64_t aligned_allocated_bytes = (allocated_bytes_ + 15) & (~ 15);
    
    // after this function returns this will be the allocated size
    int64_t new_allocated_bytes = aligned_allocated_bytes + new_bytes;

    // resize tensor_ if needed
    int64_t new_used_bytes = tensor_.numel();
    while (new_used_bytes < new_allocated_bytes) {
      new_used_bytes *= 2;
    }
    if (new_used_bytes > tensor_.numel()) {
      tensor_ = at::zeros(
          {new_used_bytes},
          at::TensorOptions().dtype(at::kByte).device(device));
    }

    allocated_bytes_ = new_allocated_bytes;

    // slice and view tensor
    return tensor_
        .index({at::indexing::Slice(
            aligned_allocated_bytes, new_allocated_bytes, 1)})
        .view(aten_dtype);
  }

 private:
  at::Tensor tensor_;
  int64_t allocated_bytes_ = 0LL;
};

// We hold one Arena for each device
thread_local std::vector<Arena> arenas;

} // namespace

at::Tensor contigZeroTensor(
    const std::vector<int64_t>& sizes,
    const c10::ScalarType& aten_dtype,
    const c10::Device& device) {
  NVF_ERROR(device.is_cuda(), "contigZeroTensor requires CUDA device");
  int64_t device_num = device.index();

  // get arena from device number, resizing arenas if needed
  if ((size_t)device_num >= arenas.size()) {
    arenas.resize(device_num + 1);
  }
 
  // request tensor from arena
  return arenas[device_num].getTensor(sizes, aten_dtype, device);
}

void releaseZeroedMemory() {
  for (Arena& a : arenas) {
    a.reset();
  }
}

} // namespace nvfuser

