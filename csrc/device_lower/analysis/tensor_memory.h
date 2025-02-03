// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#pragma once

namespace nvfuser {

class TensorView;
class Fusion;

// Information used to lower tensor memory. So far, it is just about allocation.
struct TensorMemoryInfo;
TensorMemoryInfo computeTMemInfo(Fusion* fusion);

// Note: [Tensor Memory Allocation]
//
// Tensor memory is a very special memory, so its allocation is also very
// different from other memory types.
//
// It is highly recommended to read the PTX documentation for tensor memory
// if you are not alreay familiar with it:
// https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#tensor-memory
//
// The first thing to note is, TMem does not have virtualization. This means:
// We can not just allocate starting from address 0 like how we allocate shared
// memory, and rely on page table to translate the same virtual address of
// different CTA to different physical address. There is no virtual TMem
// address. All addresses are physical addresses.
//
// Because multiple CTAs can execute on the same SM simultaneously, there must
// be some handshaking mechanism for each CTA to know the region of TMem that it
// can use. This is done by using the PTX instruction tcgen05.alloc. To ensure
// safety, there is a mutex "I have the right to allocate TMem" in the
// hardware. At the beginning of each CTA, the CTA will try to acquire the mutex
// automatically. If it fails, the CTA will be blocked until the mutex is free.
// This means, only one CTA can allocate TMem at a time. Once the CTA has
// finished allocating TMem, it should release the mutex to relinquish the right
// to allocate. After the right to allocate is relinquished, this CTA can not
// allocate new TMem any more, but it can still access the TMem that it has
// allocated, and it can also free the TMem that it has allocated. Once one CTA
// relinquishes the right to allocate, the next CTA that is blocked will be
// unblocked and can acquire the mutex to allocate TMem.
//
// Currently, our TMem allocation strategy is as naive as follows:
// We assume there is at most one TensorView on TMem in the fusion. With this
// assumption, we don't have to worry about where to place different tensors on
// TMem. We will traverse the fusion to look for a TMem TensorView. If we can
// find such a TensorView, we will generate a tcgen05.alloc and
// tcgen05.relinquish_alloc_permit at the beginning of the kernel. We do not
// dealloc TMem for now.

// The actual definition of TensorMemoryInfo.
struct TensorMemoryInfo {
  // The address returned by tcgen05.alloc.
  // tcgen05.alloc stores the allocated address in shared memory. So we use a
  // TensorView with MemoryType::Shared to store this address.
  TensorView* allocation_address = nullptr;
};

} // namespace nvfuser
