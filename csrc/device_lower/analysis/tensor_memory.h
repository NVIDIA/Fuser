// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#pragma once

namespace nvfuser {

class Fusion;

// Information used to lower tensor memory. So far, there is no information
// needed, the computeTMemInfo just check that there is only one tensor on TMem
// in the fusion. This limitation is described in the note below, and it is only
// for incremental development. This limitation will be removed soon in the
// future.
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
// Currently, the TMem allocation is not supported in nvFuser. We currently only
// allow one TensorView to be on TMem, and because we never relinquish the right
// to allocate TMem, CTA will be serialized on SM. A new CTA can be scheduled on
// an SM only after the previous CTA on that SM has completely finished
// executing. Thanks to this serialization, we can just skip allocating and
// think that our only TMem TensorView own the entire TMem, because we are sure
// that there will not be another CTA using that address. As a result, we could
// just provide address 0 to our instructions that access TMem. In principle, it
// is clearly wrong to write to an address that is not allocated, but because we
// are sure that it will in practice work for the specific unit test that we are
// targeting, we just do it so we have incremental development.

struct TensorMemoryInfo {};

} // namespace nvfuser
