// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#pragma once

#include <exceptions.h>

#include <ir/all_nodes.h>
#include <kernel_ir.h>
#include <kernel_ir_dispatch.h>

// Double buffering a tensor doubles its allocation size and uses two
// buffers to facilitate computation and memory access
// overlapping. The basic form of code looks like as follows:
//
// Before:
// for i
//   x[S]; // allocation
//   for j:
//     x[j] = y[i, j]
//   for j:
//     ... = x[j]
//
// After:
// X[S * 2]; // allocation
// for i in 0 to 1: // Prologue
//   for j:
//     x[j] = y[i, j]
//
// for i in 0 to N-1: // Main
//   for j:
//     x[j + (1 - i % 2) * S] = y[i + 1, j]
//   for j:
//     ... = x[j + (i % 2) * S]
//
// for i in N-1 to N: // Epilogue
//   for j:
//     ... = x[j + (i % 2) * S]
//
// Here, S is the original size of tensor x.
//
// The i loop is the double buffer loop of tensor x, where double
// buffering is applied to the tensor. The first step of lowering is
// to find the double buffering axis for each double buffered
// tensor. It must not be parallelized as it isn't possible to double
// buffer parallelized loops. Also, an unrolled axis expands the
// allocation and is intended to make the loop completely unrolled,
// which also conflicts with double buffering. So, basically, the double
// buffering axis is the inner-most axis within the axes left
// of the CA position. However, when it is parallelized or unrolled, a
// further left axis is picked.
//
// Once the double buffer axis is determined, the main task is to
// replicate the corresponding double buffer loop as illustrated
// above. The Prologue loop is to just fetch the first element to
// populate the buffer. The main loop is mostly the same as the
// original loop, except for the indexing change to switch the two
// buffers. When used as a consumer, an offset of (1 - i % 2) * S is
// added, whereas (i % 2) * S is added when used as a producer. Here,
// i is the index of the double buffer loop. The Epilogue loop is just
// for the last iteration of the loop. Since the main loop reads one
// element ahead of the producer of the double buffered tensor, it
// would require an additional guard to prevent buffer overruns with
// the producer if the main loop were also used for the last
// iteration. However, the value loaded by the invalid load would not
// be used, so instead of adding the additional predicate, the Epilogue
// loop is replicated from the original loop, except for the load
// expression since it's not used. Note that this overrun does not
// happen when the producer is on gmem, so in that case, this
// additional replication is not done.
//
// When creating those three types of loops, additional care must be
// taken when multiple tensors are double buffered. When multiple
// tensors use the same loop as their double buffer loop, one pass of
// replication takes care of them at once, meaning the same Prologue,
// Main, Epilogue loops are used for the multiple tensors.
//
// Other tasks to do for a double buffer tensor include:
// - Move allocation to outside of the double buffer loop
// - Double the allocation size
// - Omit the RAW sync in the Main and Epilogue loops

// [Cicular buffer] An generalization of double buffering.
// On sm80+ hardware there is asynchronous copy infrastructure that
//  motivates a circular buffering generalization of double buffering.
// Almost all analyses previously done for double buffering are exactly
//  the same with circular buffering, except for the introduction of
//  new concept: `stage depth`.
//
// The `stage depth` is defined as the multiplier of extra buffering
//  space used. In the case of double buffering, the stage depth would
//  be 2.
//
// A circular buffered loop structure would look like follows, which
//  exactly parallels the case of double buffered loop structure, since
//  it is a exact generalization to the same purpose.
//
// Here S is the original allocation size as above,
//  D is the stage depth. With D=2, the below loop structure becomes
//  exactly the same as the case in double buffering.
//
// allocate X[S*D] // allocation
// for i in 0..D-1: // prolog
//   for j in ...
//     if pred:
//       x[i*S+j] = y[i, j];
//
// for i in 0..N: // main loop
//   for j in ...
//     if pred:
//       x[((i+D-1)%D)*S+j] = y[i+D-1, j];
//   for j in ...
//     .. = x[(i%D)*S+j]
//
// (Epilog omitted since this only makes sense in using
// cp.async, where producer will be in global mem and consumer will
// be in shared mem).
//
// The profitability of this optimization comes from extra tolerance
//  of global memory pipeline latency, as on the expression `.. = x[(i%D)*S+j]`
//  we only need to make sure the data for the current iteration is
//  completed while the remaining D-2 load iterations could still be in progress
//  and overlap with the computes of the current loop.
//
// To express this pattern on sm80+ hardware we can group the loads
//  in each iteration of the circular buffered loop as one "transaction",
//  and specify how many transactions we want to ensure completion when
//  we insert the async barriers.
//
// allocate X[S*D] // allocation
// for i in 0..D-1: // prolog
//   for j in ...
//     if pred:
//       x[i*S+j] = y[i, j];
//   cp.async.commit; // mark the transaction boundary
//
// # At this point we have D-1 transactions on the fly.
//   and for the first iteration of the main loop we need
//   one transaction completed, so we leave D-2 transactions
//   on the fly, which would be the input to the barrier instruction.
//
// cp.async.wait D-2 // ensure all but the last D-2 transactions complete.
//
// for i in 0..N: // main loop
//   # At this point we always have D-2 transactions on the fly.
//      and one completed.
//   for j in ...
//     if pred:
//       x[((i+D-1)%D)*S+j] = y[i+D-1, j];
//   for j in ...
//     .. = x[(i%D)*S+j]
//   cp.async.commit; // mark the transaction boundary for the
//                       load issued in this iteration.
//   # At this point we have D-1 transactions on the fly,
//       and none completed.
//   cp.async.wait D-2; // Ensure all but the last D-2 transactions complete.
//   __syncthreads(); // Need to syncthreads because each thread will only
//                      ensure completion of its own async copies so
//                      would need to sync to this point to ensure
//                      completion of the whole tile.

namespace nvfuser {

class TmaCircularBufferInfo {
 public:
  // Map cpAsyncBulk to its tensor index
  void recordTensorIndex(const Expr* expr, kir::TensorIndex* index);

  // Check if tensor index exists for expression
  bool existsTensorIndex(const Expr* expr) const;

  // Get tensor index for expression
  kir::TensorIndex* getTensorIndex(const Expr* expr);

 private:
  // Track mbarrier used for cpAsyncBulk load operation. Required by indexing
  // pass.
  std::unordered_map<const Expr*, kir::TensorIndex*> ldst_mbarrier_index_map_;
};

//! Assumptions:
//!  1. The padding is one for warp specialized axis.
//!  2. The number of warp groups is 2.
//!  3. Persistent outer for-loop exists to overlap TensorCores and Epilogue.
//!  4. Hopper WGMMA operation is detected.
//!  5. Independent warp groups are detected.
//!
//! Analysis Circular Buffering:
//!  * If the conditions for ping-pong, persistent warp-specialized matmul exist
//!    then create HopperPingPongMbarriers.
//!
//! Allocation pass:
//!  * Create mbarriers for ordered synchronization between warp groups when
//!   accessing TensorCores and CUDA Epilogue.
//!
//! Circular Buffering pass:
//!  * In persistent for_loop, after adding short-circuit but before
//!    TensorCores, mbarrier::wait for TensorCores to be available for this warp
//!    group.
//!
//!  ReadAfterWriteSyncs pass:
//!   * In persistent for_loop, after inserting RAW wgmma::wait_all, insert a
//!     mbarrier::arrive to next warp group to release TensorCores and
//!     mbarrier::wait for CUDA epilogue for this warp group.
//!
//!  WarAsyncWaitInserter pass:
//!   * In persistent for_loop, after inserting WAR tma_store::wait_all, insert
//!   a mbarrier::arrive to next warp group to release CUDA Epilogue.
class HopperPingPongMbarriers {
 public:
  HopperPingPongMbarriers(int64_t num_warp_groups, ParallelType ws_axis);

  //! Get the number of warp groups.
  int64_t getNumWarpGroups() const {
    return num_warp_groups_;
  }

  //! Track persistent for-loop. Its index variable determines the phase of
  //! ping-pong mbarriers.
  void trackPersistentForLoop(kir::ForLoop* loop) {
    persistent_for_loop_ = loop;
  }

  //! This helper function initializes ping-pong mbarriers.
  //!
  //! for (unsigned i = 0; i < num_ping_pong_mbarriers; ++i) {
  //!   if (warp_id == 0 && electSync()()) {
  //!     mbarrier::init(...);
  //!   }
  //! }
  kir::ForLoop* initializePingPongMbarrier();

  //! This helper function invalidates ping-pong mbarriers.
  //!
  //! for (unsigned i = 0; i < num_ping_pong_mbarriers; ++i) {
  //!   if (warp_id == 0 && electSync()()) {
  //!     mbarrier::inval(...);
  //!   }
  //! }
  kir::ForLoop* invalidatePingPongMbarrier();

  //! This helper function allocates, initializes and invalidates ping-pong
  //! mbarriers.
  std::tuple<kir::Allocate*, kir::ForLoop*, kir::ForLoop*>
  createPingPongMbarrier();

  //! Create a IfThenElse where the last warp group releases the TensorCore and
  //! Epilogue mbarriers for the first independent warp group.
  //!
  //! Pseudo-code:
  //! if (ws_axis == (all_compute_id - 1)) {
  //!   // offset 0 is the TensorCores mbarrier whereas offset 1 is the Cuda
  //!   // Epilogue mbarrier.
  //!   mbarrier::arrive(ping_pong_mbarriers[0]);
  //!   mbarrier::arrive(ping_pong_mbarriers[1]);
  //! }
  kir::IfThenElse* createPrefetchIfThenElse();

  //! Select mbarrier for the given computation phase and independent warp
  //! group.
  Val* getMbarrierIndex(bool next_warp_group, bool is_epilogue);

  //! Create mbarrier::wait to pause warp group until the computation phase is
  //! unused by the other warp groups.
  //!
  //! Pseudo-code:
  //!   mbarrier::wait(ping_pong_mbarriers[indexByComputeType(is_epilogue)],
  //!                  persistent_for_loop->index() % 2)
  Expr* createMbarrierWait(bool next_warp_group, bool is_epilogue);

  //! Create mbarrier::arrive to release given computation phase to the other
  //! warp groups.
  //!
  //! Pseudo-code:
  //!   mbarrier::arrive(ping_pong_mbarriers[indexByComputeType(is_epilogue)])
  Expr* createMbarrierArrive(bool next_warp_group, bool is_epilogue);

 private:
  int64_t num_warp_groups_ = 0;
  ParallelType ws_axis_ = ParallelType::Serial;
  TensorView* mbarriers_ = nullptr;
  kir::ForLoop* persistent_for_loop_ = nullptr;
};

class CircularBufferPass {
 public:
  //! Apply circular buffering transformations
  static std::vector<Expr*> run(const std::vector<Expr*>& exprs);
};

} // namespace nvfuser
