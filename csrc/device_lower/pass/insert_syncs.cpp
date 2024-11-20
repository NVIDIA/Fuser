// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#include <device_lower/lower2device.h>
#include <device_lower/pass/insert_syncs.h>
#include <device_lower/utils.h>
#include <dispatch.h>
#include <instrumentation.h>
#include <ir/utils.h>
#include <kernel_ir.h>
#include <kernel_ir_dispatch.h>

#include <unordered_set>

namespace nvfuser {

namespace {

//! Scan through Kernel IR for-loops to insert Sync nodes to avoid
//! Write-After-Read (WAR) race condition.
//!
//! Example:
//!   for () {
//!     smem_buf[threadIdx.x] = x;
//!     __syncthreads();
//!     buf[threadId.x] = smem_buf[threadIdx.x + 1];
//!  }
//!
//! In this case, additional syncthreads is needed at the end of the
//! loop body to avoid a hazard with smem_buf.

//! Keeping track the allocations of SMEM TVs
class SmemAllocMap {
 public:
  //! Insert a new node if it's a SMEM allocation
  void insert(kir::Allocate* alloc) {
    if (auto tv = dynamic_cast<TensorView*>(alloc->buffer())) {
      if (tv->getMemoryType() == MemoryType::Shared) {
        // Note that a TensorView can have two allocations due to
        // unswitch.
        auto p = map_.insert({tv, alloc});
        // If there's an existing entry, reset it with the new
        // alloc. Currently, the existing alloc is actually the same
        // as the new one as each expression is just inserted to both
        // then and else parts of the unswitched loop, but this should
        // be changed.
        if (!p.second) {
          p.first->second = alloc;
        }
      }
    }
  }

  //! Run through aliases to get the buffer that is actually allocated for a
  //! given TV
  TensorView* getRealBuffer(TensorView* tv) const {
    auto it = map_.find(tv);
    NVF_ERROR(it != map_.end(), "Allocation not found for ", tv->toString());
    const kir::Allocate* alloc = it->second;
    while (alloc->alias()) {
      alloc = alloc->alias();
    }
    auto buf = alloc->buffer();
    NVF_ERROR(buf->isA<TensorView>());
    return buf->as<TensorView>();
  }

 private:
  std::unordered_map<TensorView*, kir::Allocate*> map_;
};

struct WarMemoryInfo {
  // True if there's a sync after the last read within the alloc loop.
  bool sync_after_read = false;

  // True if there's a sync before the first write. There can be multiple writes
  // from memory aliasing.
  bool sync_before_write = false;

  // Has there been a read of this memory location
  bool read_hit = false;

  // Has there been *the* write to this memory location, assumes single write
  // instruction (needs to be before conditionals added to code)
  bool write_hit = false;

  // For loop this TV is compute_at'ed in.
  ForLoop* ca_loop = nullptr;
};

// To prevent shared memory from being over written before it is read, a
// synchronization point has to be inserted either between the allocation of an
// SMEM buffer and where we write into it, or after the buffer's last read
// before exiting the allocation's scope.
//
// e.g.
//  for i:
//    "alloc A" in shared memory - This is really marked by the compute_at point
//    sync_loc_0
//    for j:
//      sync_loc_1
//      for k:
//        sync_loc_2
//        A = ...
//      for k:
//        ... = ... A
//    for j:
//      for k:
//        ... = ... A
//        sync_loc_3
//      sync_loc_4
//    sync_loc_5
//
// All sync locations here provide valid protection that memory in A is finished
// being read before it is over written in the next iteration
//
// Insertion of sync threads will be done from the inner most position to the
// outer most. If a sync protecting the buffer is not already placed, the
// location prefered for the sync threads is the last possible position. One
// future optimization could be to not sync on the last iteration of the loop
// the sync is placed in.
class WarSyncInserter : private kir::ExprMutator {
 public:
  static std::vector<Expr*> insert(const std::vector<Expr*>& exprs) {
    WarSyncInserter inserter(exprs);
    return inserter.exprs_;
  }

 private:
  //! Insert Sync nodes at the end of a given for-loop when a WAR
  //! hazard may happen.
  WarSyncInserter(const std::vector<Expr*>& exprs) {
    kir::ExprMutator::traverseAndInsert(exprs);
  }

  void handle(kir::IfThenElse* ite) final {
    NVF_ERROR(
        ite->elseBody().empty(),
        "Pass does not support conditional flow,",
        " needs to be done before conditional execution is lowered.");
    kir::ExprMutator::handle(ite);
  }

  void handleSync() {
    // Register the sync for the active for loop
    sync_hit_.back() = true;
    // Run through the active allocations, if a read was hit, register there was
    // a sync after the read. If there's subsequent reads on this buffer the
    // sync_after_read will be cleared.
    for (auto& entry : smem_allocations_) {
      auto& alloc_stack = entry.second;
      if (alloc_stack.back().read_hit) {
        alloc_stack.back().sync_after_read = true;
      }
    }
  }

  void handle(kir::BlockSync*) final {
    handleSync();
  }

  void handle(kir::GridSync*) final {
    handleSync();
  }

  // Checks if fl or loops within it have hit a sync
  bool syncWithin(ForLoop* fl) {
    // If outer most scope check the first sync_hit_ position
    if (fl == nullptr) {
      return sync_hit_[0];
    }

    // Find the for loop we want to look within
    auto fl_it = std::find(for_loops_.begin(), for_loops_.end(), fl);

    // Convert it to an index, but add one for the outer most scope
    auto fl_i = std::distance(for_loops_.begin(), fl_it) + 1;

    // Start at that index and see if there's syncs within that for loop
    for (auto i : c10::irange(fl_i, sync_hit_.size())) {
      if (sync_hit_[i]) {
        return true;
      }
    }
    return false;
  }

  void handle(kir::Allocate* allocate) final {
    alloc_map_.insert(allocate);
  }

  void dispatch(Expr* expr) final {
    // If not a tensor view expression continue with dispatch
    if (!ir_utils::isTvOp(expr)) {
      kir::ExprMutator::dispatch(expr);
      return;
    }

    // Mark write has been hit for all output tvs
    auto out_tvs = ir_utils::filterByType<TensorView>(expr->outputs());
    for (auto out_tv : out_tvs) {
      if (out_tv->getMemoryType() != MemoryType::Shared ||
          GpuLower::current()->syncMap()->needsRawSync(out_tv).none()) {
        continue;
      }

      auto& entry = getMemInfo(out_tv);

      // If this is the first write and there's a sync in one of the loops after
      // the compute at loop, then this buffer is protected.
      if (syncWithin(entry.ca_loop) && !entry.write_hit) {
        entry.sync_before_write = true;
      }
      entry.write_hit = true;
    }

    // Mark read was hit, if sync_after_read was set, clear it.
    auto inp_tvs = ir_utils::filterByType<TensorView>(expr->inputs());
    for (auto inp_tv : inp_tvs) {
      if (inp_tv->getMemoryType() != MemoryType::Shared ||
          GpuLower::current()->syncMap()->needsRawSync(inp_tv).none()) {
        continue;
      }

      auto& entry = getMemInfo(inp_tv);
      entry.read_hit = true;
      // Clear the sync_after_read if it was set because there was another write
      entry.sync_after_read = false;
    }
  }

  void handle(ForLoop* for_loop) final {
    // Push loop scope information
    auto prev_within_iter_loop_ = within_iter_loop_;
    sync_hit_.push_back(false);

    // If there is no real iterating loop WAR syncs aren't necessary
    within_iter_loop_ = within_iter_loop_ || !for_loop->isTrivial();

    // Process the expressions in the for loop
    kir::ExprMutator::handle(for_loop);

    // Sync analysis and cleanup:
    //
    //   Pop for loop stack inside WarMemoryInfo structs if they match this one.
    //   Erase empty entries so we don't continue to search over them
    //
    //   Insert sync at end of this for loop if any of the entries require
    std::vector<TensorView*> to_erase;
    bool insert_sync = false;
    for (auto& entry : smem_allocations_) {
      auto& alloc_stack = entry.second;
      if (!alloc_stack.empty() && alloc_stack.back().ca_loop == for_loop) {
        if (!alloc_stack.back().sync_after_read &&
            !alloc_stack.back().sync_before_write) {
          insert_sync = within_iter_loop_;
        }

        alloc_stack.pop_back();
        if (alloc_stack.empty()) {
          to_erase.push_back(entry.first);
        }
      }
    }

    for (auto tv : to_erase) {
      smem_allocations_.erase(tv);
    }

    // WAR Sync is necessary in this loop, register its insertion.
    if (insert_sync) {
      auto sync_expr = IrBuilder::create<kir::BlockSync>(true);
      kir::ExprMutator::registerInsertAfter(
          for_loop->body().exprs().back(), sync_expr, &for_loop->body());
      handle(sync_expr);
    }

    // Pop for loop scope information
    sync_hit_.pop_back();
    within_iter_loop_ = prev_within_iter_loop_;
  }

  // Create a new WarMemoryInfo entry if required and return a reference to it,
  // else return the WarMemoryInfo associated with tv
  WarMemoryInfo& getMemInfo(TensorView* tv) {
    auto maybe_aliased_tv = alloc_map_.getRealBuffer(tv);
    auto alloc_it = smem_allocations_.find(maybe_aliased_tv);
    auto ca_loop =
        lower_utils::getAllocInformation(tv, for_loops_).init_for_loop;
    if (alloc_it == smem_allocations_.end()) {
      WarMemoryInfo mem_info;
      mem_info.ca_loop = ca_loop;
      auto entry_it =
          smem_allocations_
              .insert(std::make_pair(
                  maybe_aliased_tv, std::vector<WarMemoryInfo>({mem_info})))
              .first;
      return entry_it->second.back();
    } else if (
        maybe_aliased_tv != tv && alloc_it->second.back().ca_loop != ca_loop) {
      WarMemoryInfo mem_info;
      mem_info.ca_loop = ca_loop;
      auto& alloc_stack = alloc_it->second;
      alloc_stack.push_back(mem_info);
      return alloc_stack.back();
    }
    return alloc_it->second.back();
  }

  //! Allocation map of SMEM buffers. Needed because of SMEM buffer aliasing,
  //! need to track the root of the alias to properly insert WAR hazard syncs
  SmemAllocMap alloc_map_;

  //! Is there a loop nest that has a non-trivial iteration (extent != 1) and
  //! not bound to a block/thread. This indicates if a WAR sync is necessary,
  //! otherwise the Expr is not in an iterating for loop.
  bool within_iter_loop_ = false;

  // Track which loops have hit a sync. Used to see if there's a sync before
  // write.
  std::vector<bool> sync_hit_ = {false};

  // Keep track of the active allocations we need to protect. Key is the
  // "getRealBuffer", not the raw tv. There can be multiple WarMemoryInfo's
  // because of aliasing. If the "getRealBuffer" tv has a compute at outside the
  // alias tv, each aliased tv in a unique ca_loop has to be tracked separately
  // for WAR insertion.
  std::unordered_map<TensorView*, std::vector<WarMemoryInfo>> smem_allocations_;
};

class ValidatePlacementAfterWrites : private kir::IrVisitor {
 public:
  //! Validate no expr in writes found under loop
  static void validate(ForLoop* loop, const std::unordered_set<Expr*>& writes) {
    ValidatePlacementAfterWrites validator(writes);
    validator.handle(loop);
  }

 private:
  using kir::IrVisitor::handle;

  ValidatePlacementAfterWrites(const std::unordered_set<Expr*>& writes)
      : writes_(writes) {}

  void dispatch(Expr* expr) final {
    if (expr->isA<ForLoop>() || expr->isA<kir::IfThenElse>()) {
      kir::IrVisitor::dispatch(expr);
    } else {
      NVF_ERROR(
          writes_.find(expr) == writes_.end(),
          "Block sync must be placed after ",
          expr->toString());
    }
  }

 private:
  const std::unordered_set<Expr*>& writes_;
};

class ReadAfterWriteSyncs : public kir::ExprMutator {
 private:
  using kir::ExprMutator::handle;

  void dispatch(Expr* expr) final {
    if (!ir_utils::isTvOp(expr) || expr->isA<kir::Allocate>()) {
      kir::ExprMutator::dispatch(expr);
      return;
    }

    auto async_type = ir_utils::getAsyncOpType(expr);
    if (async_type != AsyncOpType::NotAsync &&
        std::any_of(
            expr->outputs().begin(), expr->outputs().end(), [](Val* val) {
              return val->isFusionOutput() && val->uses().empty();
            })) {
      // Typically, we insert waits before the first read of the output of an
      // async op. However, if the output is a terminating fusion output, there
      // will be no first read, but still, we need to wait for it to complete
      // before exiting the kernel.
      async_exprs_writing_fusion_output_.push_back(expr);
    }

    if (auto mma = dynamic_cast<MmaOp*>(expr)) {
      if (mma->isHopper()) {
        auto scope = scope_.empty() ? nullptr : scope_.back();
        if (!lower_utils::allMmaInputsGuardedByMBarrier(mma)) {
          // Makes sure that writes to operands in the generic proxy are visible
          // to the async proxy
          auto wgmma_fence = IrBuilder::create<kir::WgMmaFence>();
          registerInsertBefore(expr, wgmma_fence, scope);
          auto fence_async = IrBuilder::create<kir::FenceAsyncProxy>();
          registerInsertBefore(expr, fence_async, scope);
        }
      }
    } else if (ir_utils::isCpAsyncBulkStore(expr)) {
      // Add a fence before TMA store so that writes in the generic proxy is
      // visible to the async proxy.
      auto scope = scope_.empty() ? nullptr : scope_.back();
      auto fence_async = IrBuilder::create<kir::FenceAsyncProxy>();
      registerInsertBefore(expr, fence_async, scope);
    }

    // Insert sync exprs before async ops. For example, insert
    //   wgmma.commit_group.sync.aligned
    //   wgmma.wait_group.sync.aligned 0
    // before the use of mma results. Note that cp.async is not handled
    // here.
    // TODO: unify the handle of cp.async
    std::unordered_map<AsyncOpType, std::unordered_set<Expr*>> input_async_ops;
    for (auto inp : expr->inputs()) {
      auto def = inp->definition();
      auto async_type = ir_utils::getAsyncOpType(def);
      if (async_type != AsyncOpType::NotAsync &&
          async_type != AsyncOpType::CpAsync) {
        input_async_ops[async_type].insert(def);
      }
    }
    for (const auto& [async_type, ops] : input_async_ops) {
      auto sync_exprs = lower_utils::getSyncExprs(async_type, 0);
      for (auto sync_expr : sync_exprs) {
        insertSyncExpr(ops, expr, sync_expr, nullptr);
      }
      for (auto op : ops) {
        // Already waited for the write to complete, so no need to wait again
        // before exiting the kernel.
        auto it = std::find(
            async_exprs_writing_fusion_output_.begin(),
            async_exprs_writing_fusion_output_.end(),
            op);
        if (it != async_exprs_writing_fusion_output_.end()) {
          async_exprs_writing_fusion_output_.erase(it);
        }
      }
    }

    // An identical but separate flow of timing for cpasync_wait.
    //  The insertion and tracking mechanism is the same as RAW
    //  sync insertion since cp.async only writes smem.
    // Currently the only interaction which is realized by the
    //  ordering in this function is that in the case when we need both a
    //  cpasync wait and a block sync before the same expr, we want
    //  to place the wait before the block sync, since currently there
    //  shouldn't be any normal case where we explicitly want the wait after a
    //  block sync.
    if (!cpasync_wait_before_.empty() && cpasync_wait_before_.front() == expr) {
      cpasync_wait_before_.pop_front();
      auto last_writes = last_cpasync_writes_.front();
      last_cpasync_writes_.pop_front();

      auto sync_expr = IrBuilder::create<kir::AsyncWait>(AsyncOpType::CpAsync);
      insertSyncExpr(last_writes, expr, sync_expr, nullptr);
    }

    if (!sync_before_.empty() && sync_before_.front().first == expr) {
      auto sync_bitmap = sync_before_.front().second;
      sync_before_.pop_front();
      auto last_writes = last_writes_.front();
      last_writes_.pop_front();
      // Found that a sync is needed

      // TODO: Explicitly test the 3 cases below
      Expr* sync_expr = nullptr;
      kir::Allocate* maybe_alloc = nullptr;
      if (sync_bitmap.hasBID()) {
        maybe_alloc = lower_utils::allocGlobalBufferForGridComm(
            lower_utils::getGridSyncBufferSize(sync_bitmap),
            DataType::Int,
            true);
        sync_expr = IrBuilder::create<kir::GridSync>(
            sync_bitmap, maybe_alloc->buffer());
      } else {
        sync_expr = IrBuilder::create<kir::BlockSync>(false); // is not war sync
      }

      insertSyncExpr(last_writes, expr, sync_expr, maybe_alloc);
    }
  }

  // Find where a sync needs to be inserted and insert the given sync.
  // This is very similar to how allocations are placed, simply place sync
  // before the expression at the common alloc point of producers (really
  // last_writes because we may have other exprs we're syncing besides the
  // producers of this one)
  void insertSyncExpr(
      const std::unordered_set<Expr*>& last_writes,
      Expr* insert_before_expr,
      Expr* sync_expr,
      Expr* maybe_alloc) {
    // The expressions in last_writes are those we're protecting the read
    // from. To figure out which loop we need a syncthread in, take the inner
    // most compute at for loop of all the outputs of the last writes.
    std::unordered_set<ForLoop*> sync_within;

    for (auto last_write : last_writes) {
      auto write_out_tv = ir_utils::getTvOutput(last_write);
      NVF_ERROR(
          write_out_tv != nullptr,
          "Error in RAW sync insertion, expecting a TV expr, but didn't find one.");
      if (write_out_tv->getComputeAtPosition() == 0) {
        continue;
      }

      auto local_id =
          write_out_tv->axis(write_out_tv->getComputeAtPosition() - 1);

      auto loops_it = std::find_if(
          for_loops_.begin(), for_loops_.end(), [&local_id](const auto& loop) {
            return GpuLower::current()->caMap()->areMapped(
                loop->iter_domain(), local_id, IdMappingMode::PERMISSIVE);
          });

      NVF_ERROR(
          loops_it != for_loops_.end(),
          "Could not find loop associated with the alloc position of ",
          write_out_tv->toString());

      sync_within.emplace(*loops_it);
    }

    // The for loop the sync needs to be in
    ForLoop* sync_within_fl = nullptr;
    for (auto fl : for_loops_) {
      if (sync_within.count(fl)) {
        sync_within_fl = fl;
      }
    }

    if (sync_within_fl == nullptr) {
      // Sync should be placed at global scope, after its outer most loop if
      // it has one.
      Expr* place_before =
          !for_loops_.empty() ? for_loops_[0] : insert_before_expr;
      // Find location in exprs_
      auto place_before_it =
          std::find(exprs_.begin(), exprs_.end(), place_before);
      NVF_ERROR(
          place_before_it != exprs_.end(),
          "Could not figure out where to place synchronization. ",
          "Tried to place after, ",
          place_before->toString(),
          ", but could not find this expression at the global scope.");
      if (maybe_alloc != nullptr) {
        registerInsertBefore(place_before, maybe_alloc, nullptr);
      }
      registerInsertBefore(*(place_before_it), sync_expr, nullptr);
    } else {
      auto sync_within_loop_it =
          std::find(for_loops_.begin(), for_loops_.end(), sync_within_fl);

      auto place_in = *sync_within_loop_it;
      Expr* place_before = nullptr;

      if (sync_within_loop_it + 1 == for_loops_.end()) {
        // Inline, place before expr
        place_before = insert_before_expr;
      } else {
        place_before = *(sync_within_loop_it + 1);
      }

      registerInsertBefore(place_before, sync_expr, &place_in->body());
      if (maybe_alloc != nullptr) {
        registerInsertBefore(place_before, maybe_alloc, &place_in->body());
      }
    }
  }

  void handle(kir::IfThenElse*) final {
    NVF_THROW(
        "Pass does not support conditional statements, ",
        "this pass should be run before any conditionals are placed in code.");
  }

  // Return a set of expressions that modify shared-memory
  // tensors. Expressions are excluded when syncthreads are already
  // placed.
  std::unordered_set<Expr*> isModifiedSharedMemory(
      const std::unordered_map<Val*, Expr*>& smem,
      const std::vector<Val*>& tvs,
      bool check_sync_map = true) const {
    std::unordered_set<Expr*> last_writes;
    for (auto tv : ir_utils::filterByType<TensorView>(tvs)) {
      if (check_sync_map &&
          GpuLower::current()->syncMap()->needsRawSync(tv).none()) {
        continue;
      }
      if (tv->getMemoryType() != MemoryType::Shared) {
        continue;
      }
      auto it = smem.find(tv);
      if (it != smem.end()) {
        last_writes.insert(it->second);
      }
    }
    return last_writes;
  }

  std::unordered_set<Expr*> isModifiedGlobalMemory(
      const std::unordered_map<Val*, Expr*>& gmem,
      const std::vector<Val*>& tvs) const {
    std::unordered_set<Expr*> last_writes;
    for (auto tv : ir_utils::filterByType<TensorView>(tvs)) {
      if (GpuLower::current()->syncMap()->needsRawSync(tv).none()) {
        continue;
      }
      auto it = gmem.find(tv);
      if (it != gmem.end()) {
        last_writes.insert(it->second);
      }
    }
    return last_writes;
  }

  ReadAfterWriteSyncs(const std::vector<Expr*>& _exprs) {
    // Fusion shared_memory values
    // Tracks if shared memory is modified
    std::unordered_map<Val*, Expr*> smem;
    // Tracks if shared memory is asynchronously modified
    std::unordered_map<Val*, Expr*> smem_async;
    std::unordered_map<Val*, Expr*> gmem;

    // Flatten all the expressions
    auto flattened_exprs = ir_utils::flattenScopedExprs(_exprs);

    Expr* prev_tv_expr = nullptr;
    for (auto expr : flattened_exprs) {
      if (!ir_utils::isTvOp(expr) || expr->isA<kir::Allocate>()) {
        continue;
      }

      auto last_gmem_writes = isModifiedGlobalMemory(gmem, expr->inputs());
      if (!last_gmem_writes.empty()) {
        NVF_ERROR(
            prev_tv_expr != nullptr,
            "Can't require sync on inputs, however, detected it's needed.");
        ParallelTypeBitmap bitmap;
        for (auto entry : gmem) {
          NVF_ERROR(entry.first->isA<TensorView>());
          auto sync_bits = GpuLower::current()->syncMap()->needsRawSync(
              entry.first->as<TensorView>());
          bitmap |= sync_bits;
        }

        sync_before_.emplace_back(expr, bitmap);
        last_writes_.push_back(last_gmem_writes);
        gmem.clear();
      }

      auto last_smem_writes = isModifiedSharedMemory(smem, expr->inputs());
      auto last_async_smem_writes =
          isModifiedSharedMemory(smem_async, expr->inputs(), false);

      // Keep track of async smem writes before the current
      //  expr, following largely the same logic as block sync.
      if (!last_async_smem_writes.empty()) {
        cpasync_wait_before_.push_back(expr);
        std::unordered_set<Expr*> async_smem_writes;
        for (auto it : smem_async) {
          async_smem_writes.insert(it.second);
        }
        last_cpasync_writes_.push_back(async_smem_writes);
        smem_async.clear();
      }

      if (!last_smem_writes.empty()) {
        NVF_ERROR(
            prev_tv_expr != nullptr,
            "Can't require sync on inputs, however, detected it's needed.");
        ParallelTypeBitmap bitmap;
        bitmap.set(ParallelType::TIDx);
        bitmap.set(ParallelType::TIDy);
        bitmap.set(ParallelType::TIDz);
        sync_before_.emplace_back(expr, bitmap);

        // Before clearing `smem`, put all the currently pending smem writes
        //  in last_writes_. This will make sure all the smem writes will
        //  be taken into consideration when deciding which loopnest level
        //  to insert the block sync. see FusionRAWSyncInsertionPlace4.
        std::unordered_set<Expr*> smem_writes;
        for (auto it : smem) {
          // No need to keep track of shared mem writes that does not
          //  require a RAW block sync.
          if (GpuLower::current()
                  ->syncMap()
                  ->needsRawSync(it.first->as<TensorView>())
                  .hasTID()) {
            smem_writes.insert(it.second);
          }
        }
        last_writes_.push_back(smem_writes);
        smem.clear();
      }

      for (auto tv : ir_utils::filterByType<TensorView>(expr->outputs())) {
        // Circular buffered tensors do not need RAW sync to be inserted
        // here, except for the initial load part, which is taken care
        // separately by CircularBufferInserter.
        if (tv->getMemoryType() == MemoryType::Shared &&
            (!tv->isCircularBuffered() ||
             tv->circularBufferOptions().prefetch == 0)) {
          smem[tv] = expr;

          // only keep track of async writes in smem_async
          if (ir_utils::isCpAsyncOp(expr)) {
            smem_async[tv] = expr;
          }
        }
        if (tv->getMemoryType() == MemoryType::Global) {
          gmem[tv] = expr;
        }
      }

      prev_tv_expr = expr;
    }

    kir::ExprMutator::traverseAndInsert(_exprs);

    // If there are async exprs writing to fusion output that is not
    // being waited yet, we need to insert the wait before exiting the
    // kernel.
    for (auto expr : async_exprs_writing_fusion_output_) {
      auto async_type = ir_utils::getAsyncOpType(expr);
      auto sync_exprs = lower_utils::getSyncExprs(async_type, 0);
      exprs_.insert(exprs_.end(), sync_exprs.begin(), sync_exprs.end());
    }

    NVF_ERROR(sync_before_.empty(), "Didn't place all required syncs.");
  }

 private:
  //! Keep track of expressions that must be followed by syncthreads
  std::deque<std::pair<Expr*, ParallelTypeBitmap>> sync_before_;

  //! Keep track of write expressions that must be placed before
  //! syncthreads.
  //!
  //! syncthreads is placed before for each expression of
  //! sync_before_. last_writes_ keeps track of expressions
  //! modifying the smem buffer each syncthreads is used for so that
  //! it is not placed before those write expressions.
  std::deque<std::unordered_set<Expr*>> last_writes_;

  //! Keep track of expressions that must be wait for cp.async to finish.
  std::deque<Expr*> cpasync_wait_before_;

  //! Keep track of write expressions that must be placed before
  //! cp.async wait.
  std::deque<std::unordered_set<Expr*>> last_cpasync_writes_;

  //! Async expressions writing to non-terminating fusion outputs.
  //! These expressions need special logic to handle because typically, we
  //! insert waits before the first read. However, for the output of these
  //! expressions, there is no "first read", but still, to be waited before
  //! exiting the kernel.
  std::vector<Expr*> async_exprs_writing_fusion_output_;

 public:
  static std::vector<Expr*> insert(const std::vector<Expr*>& loop_nests) {
    ReadAfterWriteSyncs inserter(loop_nests);
    return inserter.exprs_;
  }
};

// Insert wait expressions for WAR harzard for async operations such as wgmma
// and tma store. To do so, we find the structure like the following example:
//   for 1
//     for 2
//       for 3
//         T1 = async_op(...)
//     for 4
//       for 5
//         T2 = expr(T1, ...)
// In the above example, we need to insert a wait expression for T1 at the end
// of loop 1, because otherwise, the T1 = async_op(...) in the next iteration
// will overwrite the operand of the T2 = expr(T1, ...).
class WarAsyncWaitInserter : private kir::ExprMutator {
 public:
  static std::vector<Expr*> insert(const std::vector<Expr*>& exprs) {
    WarAsyncWaitInserter inserter(exprs);
    return inserter.exprs_;
  }

 private:
  //! Is there a loop nest that has a non-trivial iteration (extent != 1) and
  //! not bound to a block/thread. This indicates if a WAR sync is necessary,
  //! otherwise the Expr is not in an iterating for loop.
  bool within_iter_loop_ = false;

  //! Inputs of async ops in the current scope. For example:
  //!  for 1:
  //!    for 2:
  //!      A = ...
  //!    for 3:
  //!      ... = async_op(A, ...)
  //! When in loop 1 and loop 2, async_inputs_in_current_scope_ will contain A.
  //! But when in loop 3, it will not contain A. We are only interested in
  //! protecting the inputs of async ops that is in the current scope. For
  //! example, in the above example, we do not want to add an async wait at the
  //! end of loop 3 because, although waiting there is functionally correct,
  //! waiting at the end of loop 1 is sufficient and cheaper.
  std::unordered_set<Val*> async_inputs_in_current_scope_;

  //! Async expressions that need to be protected by a wait expression, but we
  //! have not inserted the wait expression yet.
  //! Example 1:
  //!  for 1:
  //!    for 2:
  //!      for 3:
  //!        A = ...
  //!      for 4:
  //!        ... = async_op(A, ...)
  //! In the above example, during traversal of ... = async_op(A, ...), we will
  //! add async_op to async_exprs_to_protect_. But we will not insert the wait
  //! expression immediately after async_op, because there is no input buffer
  //! to protect in that scope. This async_op will remain in
  //! async_exprs_to_protect_ until we exit the handle of loop 4 and return to
  //! the handle of loop 2. At that point, we will insert the wait expression
  //! for async_op at the end of loop 2, because we do have an input buffer to
  //! protect (A) in that scope. Once we insert the wait expression, we will
  //! remove async_op from async_exprs_to_protect_, because it has already been
  //! protected, and there is no need to insert the wait expression again at the
  //! end of loop 1.
  //! Example 2:
  //!  for 1:
  //!    for 2:
  //!      for 3:
  //!        A = ...
  //!      for 4:
  //!        ... = async_op(A, ...)
  //!    ... = async_op2(A, ...)
  //! Similar to example 1, we will insert the wait expression for async_op at
  //! the end of loop 2. But after we return to the handle of loop 1, we will
  //! then insert async_op2 to async_exprs_to_protect_, which will make us to
  //! insert the wait expression for async_op2 at the end of loop 1.
  std::unordered_set<Expr*> async_exprs_to_protect_;

 private:
  WarAsyncWaitInserter(const std::vector<Expr*>& exprs) {
    kir::ExprMutator::traverseAndInsert(exprs);
  }

  // Get the async op types of the use expressions of a value.
  std::unordered_set<AsyncOpType> getUseAsyncOpTypes(Val* v) {
    std::unordered_set<AsyncOpType> async_ops;
    for (auto use : v->uses()) {
      auto type = ir_utils::getAsyncOpType(use);
      if (type != AsyncOpType::NotAsync) {
        async_ops.insert(type);
      }
    }
    return async_ops;
  }

  void dispatch(Expr* expr) final {
    // If not a tensor view expression continue with dispatch
    if (!ir_utils::isTvOp(expr)) {
      kir::ExprMutator::dispatch(expr);
      return;
    }

    // If the output of the current expression is used by an async op, then we
    // add the output to the async_inputs_in_current_scope_ so that we know we
    // need to protect it.
    // TODO: due to compute-with, we may have code like below:
    //   float T1[4][8];
    //   for i in range(4):
    //     for j in range(8):
    //       T1[i][j] = ...
    //     for j in range(8):
    //       ... = async_op(T1[i][j], ...)
    // For this case, there is no need to protect T1 because different
    // iterations of i is not accessing the same elements of T1, so there is no
    // WAR hazard. Today, we just ignore such case and conservatively protect
    // it. This is functionally correct but may not be performant. We need to
    // improve this if in the future, we want to use compute-with with async
    // ops.
    for (auto output : expr->outputs()) {
      auto use_async_ops = getUseAsyncOpTypes(output);
      if (!use_async_ops.empty()) {
        async_inputs_in_current_scope_.emplace(output);
      }
    }

    // If the current expression is an async op, then we add it to
    // async_exprs_to_protect_ so that we know we need to protect it.
    auto async_op_type = ir_utils::getAsyncOpType(expr);
    if (async_op_type != AsyncOpType::NotAsync) {
      async_exprs_to_protect_.insert(expr);
    }
  }

  // Open a scope, update the context of the "current" scope, and return the
  // context of the previous scope that will be saved on the stack of function
  // call frames for restoration later.
  std::unordered_set<Val*> openScope() {
    std::unordered_set<Val*> result;
    std::swap(result, async_inputs_in_current_scope_);
    return result;
  }

  // Restore the context of the previous scope that was saved on the stack of
  // function call frames.
  auto closeScope(std::unordered_set<Val*>& prev_async_inputs) {
    std::transform(
        async_inputs_in_current_scope_.begin(),
        async_inputs_in_current_scope_.end(),
        std::inserter(prev_async_inputs, prev_async_inputs.end()),
        [](const auto& entry) { return entry; });
    async_inputs_in_current_scope_ = std::move(prev_async_inputs);
  }

  void handle(kir::IfThenElse* ite) final {
    auto prev_async_inputs = openScope();
    kir::ExprMutator::handle(ite);
    closeScope(prev_async_inputs);
  }

  // The wait for async ops, for example, wgmma.wait_group.sync.aligned,
  // generally takes an argument "pending_ops" to specify how many pending
  // transactions are allowed to remain unfinished. For example, if we have:
  //   wgmma 1;
  //   wgmma 2;
  //   wgmma.commit;
  //   wgmma 3;
  //   wgmma.commit;
  //   wgmma 4;
  //   wgmma 5;
  //   wgmma.commit;
  // Then at this point, we have 3 pending transactions:
  //   transaction 1: wgmma 1, wgmma 2
  //   transaction 2: wgmma 3
  //   transaction 3: wgmma 4, wgmma 5
  // If we do wgmma.wait_group.sync.aligned 1, then we will wait until there is
  // at most 1 pending transaction. In this case, we will wait until transaction
  // 1 and transaction 2 is finished. This function calculates the
  // "pending_ops". Typically, the "pending_ops" is just 0, i.e., wait until all
  // pending ops are finished. But in some cases, especially for the expression
  // that consumes the circular buffered tensor, the "pending_ops" can be larger
  // than 0, depending on the prefetch distance and the stage depth of the
  // circular buffer loop. When the prefetch distance is smaller than
  // stage_depth - 1, we have have buffers for eliminating WAR harzards, so we
  // can allow more pending transactions.
  int64_t getPendingOpsFor(Expr* expr, ForLoop* current_loop) {
    auto for_loops_including_current = for_loops_;
    for_loops_including_current.push_back(current_loop);
    const auto gpu_lower = GpuLower::current();
    int64_t pending_ops = std::numeric_limits<int64_t>::max();
    for (auto inp : expr->inputs()) {
      if (async_inputs_in_current_scope_.count(inp) == 0) {
        continue;
      }
      auto tv = dynamic_cast<TensorView*>(inp);
      if (tv == nullptr) {
        continue;
      };
      if (!tv->isCircularBuffered()) {
        return 0;
      }
      auto circular_buffer_loop =
          gpu_lower->circularBufferInfo().getCircularBufferLoop(
              tv, for_loops_including_current);
      if (circular_buffer_loop != current_loop) {
        return 0;
      }
      auto stage = circular_buffer_loop->circularBufferLoopStage();
      NVF_ERROR(
          stage == CircularBufferLoopStage::Main,
          "Only main circular buffer loop needs WAR async wait, ",
          "so the code should not reach here. Stage:",
          stage);

      const auto& opt =
          GpuLower::current()->circularBufferInfo().getCircularBufferOptionsFor(
              circular_buffer_loop->iter_domain());
      pending_ops = std::min(pending_ops, opt.stage - opt.prefetch - 1);
    }
    return pending_ops;
  }

  void handle(ForLoop* for_loop) final {
    // Push loop scope information
    auto prev_within_iter_loop_ = within_iter_loop_;
    within_iter_loop_ = within_iter_loop_ || !for_loop->isTrivial();
    auto prev_async_inputs = openScope();

    // Process the expressions in the for loop
    kir::ExprMutator::handle(for_loop);

    // Insert async wait at the end of this for loop
    if (within_iter_loop_) {
      std::unordered_map<AsyncOpType, int64_t> types_and_pending_ops_to_protect;

      // Gather the information on what wait expressions we should insert.
      for (auto it = async_exprs_to_protect_.begin();
           it != async_exprs_to_protect_.end();) {
        auto expr = *it;
        // If the input of the async op is not in the current scope, then this
        // async op is not related, so nothing to protect.
        if (std::none_of(
                expr->inputs().begin(), expr->inputs().end(), [&](Val* val) {
                  return async_inputs_in_current_scope_.count(val);
                })) {
          it++;
          continue;
        }
        int64_t pending_ops = getPendingOpsFor(expr, for_loop);
        auto type = ir_utils::getAsyncOpType(expr);
        // If there are multiple async ops of the same type to protect, we will
        // only insert a single wait expressions with the smallest
        // "pending_ops".
        if (types_and_pending_ops_to_protect.count(type)) {
          auto& pending_ops_to_protect = types_and_pending_ops_to_protect[type];
          pending_ops_to_protect =
              std::min(pending_ops_to_protect, pending_ops);
        } else {
          types_and_pending_ops_to_protect.emplace(type, pending_ops);
        }
        it = async_exprs_to_protect_.erase(it);
      }

      // Actually insert these wait expressions.
      for (auto [type, pending_ops] : types_and_pending_ops_to_protect) {
        auto sync_exprs = lower_utils::getSyncExprs(type, pending_ops);
        while (!sync_exprs.empty()) {
          registerInsertAfter(
              for_loop->body().exprs().back(),
              sync_exprs.back(),
              &for_loop->body());
          sync_exprs.pop_back();
        }
      }
    }

    // Pop for loop scope information
    within_iter_loop_ = prev_within_iter_loop_;
    closeScope(prev_async_inputs);
  }
};

} // namespace

std::vector<Expr*> insertRawThreadSynchronization(
    const std::vector<Expr*>& exprs) {
  FUSER_PERF_SCOPE("GpuLower::Lower::insertRawThreadSynchronization");
  return ReadAfterWriteSyncs::insert(exprs);
}

std::vector<Expr*> insertWarThreadSynchronization(
    const std::vector<Expr*>& exprs) {
  FUSER_PERF_SCOPE("GpuLower::Lower::insertWarThreadSynchronization");
  return WarSyncInserter::insert(exprs);
}

std::vector<Expr*> insertWarAsyncWait(const std::vector<Expr*>& exprs) {
  FUSER_PERF_SCOPE("GpuLower::Lower::insertWarAsyncWait");
  return WarAsyncWaitInserter::insert(exprs);
}

} // namespace nvfuser
