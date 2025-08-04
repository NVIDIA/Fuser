// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#pragma once

#include <device_lower/dependencies.h>
#include <ir/all_nodes.h>
#include <type.h>

#include <vector>

namespace nvfuser {

//
// Types of required syncs:
//
// RAW sync: (see sync_information.cpp for more info)
//  These are needed when there is differing parallelization between producer
//  and consumer expressions such that communication between threads via shared
//  or global memory is required.
//
//  Ex 1.1)
//
//  Definition:
//    T0_g[iS0{i0}, iS1{32}]
//    T1_s[iS2{i0}, iS3{32}] = set(T0)   (expr1)
//    T2_l[iS4{i0}, iS5{32}] = set(T1)   (expr2)
//
//  Scheduled as:
//    Split iS3{32} by 4 -> ithreadIdx.x6{8}, iS7{4}
//    Split iS5{32} by 8 -> iS8{4}, ithreadIdx.x9{8}
//
//    T1_s[ iS2{i0}, ithreadIdx.x6{8}, iS7{4} ]
//    T2_l[ iS4{i0}, iS8{4}, ithreadIdx9{8} ]
//
//   i.e. swapping the inner and outer splits to be parallelized along TIDx. In
//   this case we are communicating data between threads via shared memory so a
//   __syncthreads or bar.sync is required between participating threads
//   somewhere in between expr1 and expr2.
//
//  Ex 1.2)
//
//  We also require RAW syncs for warp specialized circular buffering:
//
//    // In warp 0:
//    for i in circ buffer loop:
//      // Load data to slot i%stages
//
//    // In warp 1:
//    for i in circ buffer loop:
//      // Read data from slot i%stages
//
//   In order to prevent reading uninitialized data in a given slot before it is
//   loaded, we need RAW syncs using an mbarrier:
//
//    // In warp 0:
//    for i in circ buffer loop:
//      // Load data to slot i%stages
//      // Arrive at mbarrier slot_full[i%stages]
//
//    // In warp 1:
//    for i in circ buffer loop:
//      // Wait at mbarrier slot_full[i%stages]
//      // Read data from slot i%stages
//
// WAR sync: (see insert_syncs.cpp for more info)
//  These are needed when we overwrite a memory location that was previously
//  used, for example during circular buffering with warp specialization or when
//  we have aliased shared memory buffers.
//
//  Ex 2.1)
//
//    // In warp 0:
//    for i in circ buffer loop:
//      // Load data to slot i%stages
//
//    // In warp 1:
//    for i in circ buffer loop:
//      // Read data from slot i%stages
//
//   In order to prevent overwriting the data in a given slot before it is read,
//   we need WAR syncs, for example using an mbarrier:
//
//    // In warp 0:
//    for i in circ buffer loop:
//      // Wait at mbarrier slot_empty[i%stages]
//      // Load data to slot i%stages
//
//    // In warp 1:
//    for i in circ buffer loop:
//      // Read data from slot i%stages
//      // Arrive at mbarrier slot_empty[i%stages]
//
//  Ex 2.2)
//
//  When we alias memory for re-use, we also need to guarantee that all reads of
//  the prior tensors are completed before performing writes to the downstream
//  aliased tensor.
//
//    T1_s = foo()
//    T2_s = bar()
//
//    baz(T1_s)
//    qux(T2_s)
//
//    T3_s = quux()
//
//   Suppose that T1 and T2 occupy memory that overlaps that allocated to T3. If
//   some thread writes to T3 before baz or qux is executed for the final time,
//   then it might read incorrect values (a WAR hazard). Thus we require a block
//   sync like so:
//
//    T1_s = foo()
//    T2_s = bar()
//
//    baz(T1_s)
//    qux(T2_s)
//
//    __syncthreads()
//
//    T3_s = quux()
//
// RAWWithConsumerFence:
//  When the producer expr is in a different proxy than the consumer, we require
//  a proxy fence. Note that this should be placed  on the appropriate side of
//  any corresponding execution sync.
//
//  Ex 3.1)
//
//   For example, suppose we have an mbarrier RAW sync for a circular buffered
//   tensor loaded with cp.async (which accesses smem via the async proxy). Then
//   we wish to do an operation in the generic proxy. As described we have the
//   following RAW synced code:
//
//    // In warp 0:
//    for i in circ buffer loop:
//      // Load data to slot i%stages
//      // Arrive at mbarrier slot_full[i%stages]
//
//    // In warp 1:
//    for i in circ buffer loop:
//      // Wait at mbarrier slot_full[i%stages]
//      // Read data from slot i%stages in generic proxy
//
//   With this code, even though we waited at the mbarrier, the memory accesses
//   could be out of order. This can be prevented by inserting a proxy fence
//   between the RAW wait and the read:
//
//    // In warp 0:
//    for i in circ buffer loop:
//      // Load data to slot i%stages
//      // Arrive at mbarrier slot_full[i%stages]
//
//    // In warp 1:
//    for i in circ buffer loop:
//      // Wait at mbarrier slot_full[i%stages]
//      fence.proxy.async.shared::cta
//      // Read data from slot i%stages in generic proxy
//
//    NOTE: WAR syncs are needed here as well and are not shown
//
//
// Note each expression might take place in a different warp specialized role,
// and the synchronization mechanisms can be inserted into one or multiple of
// these roles as well. For example in the memory proxy example above we only
// insert the fence into the compute warps, while the RAW sync mbarrier
// instructions are inserted in to the code for both of the two different warp
// types. Note that the fence is placed in the thread for which the consumer
// expression exists.

class SyncRequirements {
 public:
  enum class Type {
    RAW,
    WAR,
    // These exist when we have RAW syncs where the producer and consumer
    // relationships are in different proxies.
    ConsumerProxyFence,
  };

  SyncRequirements(const std::vector<Expr*>& exprs);

 protected:
  /*
  // -1 here indicates that this is a compute warp i.e. a regular thread, not an
  // async warp such as a matmul operand load warp or the Blackwell mma warp.
  using WarpSpecializedRole = int8_t;

  class CircularBufferCloneMapper {
   public:
    void registerClone(
        Expr* orig,
        Expr* cloned,
        CircularBufferLoopStage stage,
        WarpSpecializedRole role) {
      orig_to_cloned_.emplace(
          std::tuple<Expr*, CircularBufferLoopStage, WarpSpecializedRole>{
              orig, stage, role},
          cloned);
      cloned_to_orig_[cloned] = orig;
    }

   private:
    std::unordered_map<
        std::tuple<Expr*, CircularBufferLoopStage, WarpSpecializedRole>,
        Expr*>
        orig_to_cloned_;
    std::unordered_map<Expr*, Expr*> cloned_to_orig_;
  } circbuf_mapper_;
  */

 private:
  DependencyMapper dependencies_;
  // Given a producer Expr and a consumer Expr, this tells us which types of
  // syncs are required. Note that these are always producer/consumer pairs in
  // that order, even for WAR syncs, so it is not always the case that the syncs
  // must lie _inside_ the interval between producer and consumer.
  struct PairHash {
    size_t operator()(const std::pair<Expr*, Expr*>& key) const {
      return std::hash<Expr*>()(key.first) ^ std::hash<Expr*>()(key.second);
    }
  };
  std::
      unordered_map<std::pair<Expr*, Expr*>, std::unordered_set<Type>, PairHash>
          required_syncs_;
};

//! This pass builds a list of required syncs
std::vector<Expr*> buildSyncRequirementsMap(const std::vector<Expr*>& exprs);

//! This is an experimental pass that subsumes the following previously-used
//! passes:
//!
//!  - reuseMemoryAllocations
//!  - CircularBufferPass
//!  - insertRawThreadSynchronization
//!  - insertWarThreadSynchronization
//!  - insertWarAsyncWait
//!
//! These passes all used similar analyses and had circular dependencies because
//! syncing, circular buffering, and memory reuse are intertwined topics.
std::vector<Expr*> circularBufferAndInsertSyncs(
    const std::vector<Expr*>& exprs);

//! This validates that sync requirements have been satisfied
std::vector<Expr*> validateSyncRequirements(const std::vector<Expr*>& exprs);

} // namespace nvfuser
