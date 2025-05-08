// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#pragma once

#include <type.h>

#include <unordered_map>
#include <vector>

namespace nvfuser {

class Val;
class TensorView;
class Fusion;
class IterDomain;

// Information used to lower tensor memory.
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
// The tcgen05.alloc instruction is like the following:
//   tcgen05.alloc [dest], nCols
//
// There are three important things to note about this instruction:
//
// 1. The output of this instruction is in shared memory address.
// 2. The unit of allocation is 32 whole columns of tensor memory. And nCols
//    must be a power of two.
// 3. The right to allocate is like a mutex and will serialize CTA scheduling.
//    The tcgen05.alloc is blocking when there is no space to allocate.
//
// The point 1 above is not a big trouble for us, but we need to make sure we
// allocate the address tensor in shared memory before allocating the tensor
// memory. But the point 2 and 3 can be a big challenge. There are basically
// two things to worry about when allocating tensor memory:
//
// 1. Fragmentation. When the tensor does not occupy all lanes or the tensor's
// size is not a power of two columns or < 32 columns, naively allocating all
// lanes with 32 or higher power of 2 columns could waste some space. In a
// perfect world, it would be nice to have a 2D allocator that is capable
// merging the allocation of multiple tensors into a single tcgen05.alloc.
// For example, if tv0 and tv2 both has 64 rows and 32 columns, we can allocate
// tv0 on the first 64 lanes, and tv1 on the next 64 lanes. Another example is,
// if tv0 has 128 rows and 31 columns, and tv1 has 128 rows and 33 columns, we
// pack the two tensors into a single tcgen05.alloc of 64 columns.
//
// 2. Latency. We should relinquish the right to allocate as soon as we are done
// with allocating, so that other CTAs can grab the "right to allocate" mutex.
// We should also deallocate the tensor memory as soon as we are done with using
// it, so that other CTA's tcgen05.alloc can get unblocked. In a perfect world,
// it would be nice to able to break one TensorView into multiple deallocations.
// For example, if tv0 has 128 rows and 256 columns, and we are sequentially
// reading these 256 columns one by one. For this case, instead of waiting for
// the entire 256-size loop to finish, it would be nice to deallocate the first
// 128 columns if we are done with reading them, so that other CTAs have a
// chance to allocate their memory in the freed space.
//
// From the above analysis, it is important to realize that the allocation of
// TensorView and the allocation of the tensor memory are not a one-to-one
// correspondence. A TensorView can be allocated by multiple tcgen05.allocs, and
// a tcgen05.alloc can be used to allocate multiple TensorViews. For now, we
// limit ourselves that a TensorView can not span multiple tcgen05.allocs, and
// we call a piece of TMem area that is allocated by a single tcgen05.alloc and
// may span multiple TensorViews a "region". This design derives a
// TMem -> region -> TensorView hierarchy.
//
// In practice, it is very difficult to optimize both fragmentation and latency
// perfectly. Although tensor memory was originally designed for matmul, because
// it is a large and fast memory, it would be nice to use it for other purposes,
// such as persistent buffers. This could make it even more difficult to
// allocate tensor memory optimally. Considering the complexity of the problem,
// the development of a tensor memory allocator is likely an incremental
// process. With this in mind, we design the allocation of tensor memory in
// nvFuser to be hackable.
//
// There are three main components in the design:
// 1. A data structure, TMemAllocationInfo, that describes how we allocate
//    tensor memory.
// 2. A heuristic, executed as part of computeTMemInfo, that generates the
//    allocation information as an instance of TMemAlllocationInfo.
// 3. A pass, executed as part of insertAllocations, that generates the actual
//    IR nodes based on the TMemAlllocationInfo.
//
// The TMemAllocationInfo data structure and the insertAllocations support
// a wider range of allocation strategies than the heuristic in computeTMemInfo.
// This provides some flexibility for prototyping and experimentation by just
// manually specifying TMemAllocationInfo. To manually specify the allocation
// strategy, the user can specify a managed variable "tmem_regions" in the
// fusion. The type of this managed variable is vector<vector<TensorView*>>
// which specifies which TensorViews should be coalesced into the same region.

// The data structure that describes how we allocate tensor memory. It is
// assumed that:
// 1. TMem allocation are split into regions, with each region described by a
//    Region. Each region spans a full 128 lanes and N columns of tensor memory.
//    The number of columns must be a power of two and minimum 32. Each region
//    is allocated by a single tcgen05.alloc and deallocated by a matching
//    tcgen05.dealloc.
// 2. Each kernel can have multiple regions.
// 3. Each region can cover multiple TensorViews, but each TensorView can not
//    span multiple regions.
struct TMemAlllocationInfo {
  // Each entry describes a region of 128 rows x N columns of tensor memory
  // allocated by a single tcgen05.alloc.
  struct Region {
    // tcgen05.alloc stores the allocated address in shared memory. So we use a
    // TensorView with MemoryType::Shared to store this address.
    TensorView* address;
    // The number of columns to allocate. Must be >= 32 and a power of two.
    Val* num_columns;
    // The TMem TensorViews covered by this region. Each region can be used to
    // store multiple TensorViews. The (lane_offset, column_offset) specifies
    // the starting offset of each TensorView in this region.
    // The lane_allocation is the part of allocation domain on the left of the
    // TMem dimsep position that is actually being allocated.
    // The column_allocation is the part of allocation domain on the right of
    // the TMem dimsep position that is actually being allocated.
    struct TVInfo {
      TensorView* tensor;
      std::vector<IterDomain*> lane_allocation;
      std::vector<IterDomain*> column_allocation;
      Val* lane_offset;
      Val* column_offset;
    };
    std::vector<TVInfo> covered_tensors;
  };
  std::vector<Region> regions;

  const Region::TVInfo& getTVInfo(TensorView* tv) const;
};

// The actual definition of TensorMemoryInfo.
struct TensorMemoryInfo {
  TMemAlllocationInfo allocation;
  std::unordered_map<TensorView*, TMemRegisterDataPath> load_data_path;
  std::unordered_map<TensorView*, TMemRegisterDataPath> store_data_path;

  bool hasTMemTensor() const {
    return !allocation.regions.empty();
  }
};

} // namespace nvfuser
