// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#include <device_lower/lower2device.h>
#include <device_lower/pass/allocation.h>
#include <expr_evaluator.h>
#include <expr_simplifier.h>
#include <id_model/utils.h>
#include <instrumentation.h>
#include <ir/iostream.h>
#include <ir/utils.h>
#include <kernel_ir.h>
#include <kernel_ir_dispatch.h>

#include <unordered_set>

namespace nvfuser {

namespace {

enum class CircularBufferWaitType { ReadAfterWrite, WriteAfterRead };

// This function creates kir::Loop with range based on stage depth. It is
// used for mbarrier initialization and invalidation.
ForLoop* createStageDepthForLoop(ForLoop* circular_buffer_loop) {
  int64_t stage_depth =
      GpuLower::current()
          ->circularBufferInfo()
          .getCircularBufferOptionsFor(circular_buffer_loop->iter_domain())
          .stage;
  return ir_utils::createRangeLoop(stage_depth);
}

// This helper function initializes mbarrier for all circular buffer stage.
//
// Expected result:
// for (unsigned i = 0; i < stages; ++i) {
//   if (warp_id == 0 && electSync()()) {
//     mbarrier::init(...);
//   }
// }
Expr* initializeMbarrier(
    ForLoop* circular_buffer_loop,
    TensorView* all_mbarriers,
    CircularBufferWaitType wait_type) {
  NVF_ERROR(circular_buffer_loop != nullptr);
  ForLoop* loop = createStageDepthForLoop(circular_buffer_loop);

  int64_t stage_depth =
      GpuLower::current()
          ->circularBufferInfo()
          .getCircularBufferOptionsFor(circular_buffer_loop->iter_domain())
          .stage;

  // We use mbarrier[0:stage_depth] for RAW, and
  // mbarrier[stage_depth:2*stage_depth] for WAR.
  Val* mbarrier_index = wait_type == CircularBufferWaitType::ReadAfterWrite
      ? loop->index()
      : SimplifyingIrBuilder::addExpr(loop->index(), stage_depth);

  // Get mbarrier for this circular buffer stage.
  kir::TensorIndex* stage_mbarrier =
      IrBuilder::create<kir::TensorIndex>(all_mbarriers, mbarrier_index);

  auto circular_buffered_tvs =
      GpuLower::current()->circularBufferInfo().getCircularBufferTvs(
          circular_buffer_loop);

  Val* num_of_arrives = nullptr;
  if (wait_type == CircularBufferWaitType::ReadAfterWrite) {
    // The mbarrier of RAW is used to wait for the completion of the TMA
    // load of the circular buffer tensor. The number of arrives is the
    // number of TMA issued for the circular buffer tensor.
    int64_t num_of_tvs_loaded_by_tma = std::count_if(
        circular_buffered_tvs.begin(),
        circular_buffered_tvs.end(),
        [](const TensorView* tv) {
          return ir_utils::isCpAsyncBulkLoad(tv->definition());
        });
    num_of_arrives =
        IrBuilder::create<Val>(num_of_tvs_loaded_by_tma, DataType::UInt32);
  } else {
    // The mbarrier of WAR is used to wait for the completion of the reading
    // of the circular buffer tensor. The number of arrives is the number of
    // threads in the CTA.
    num_of_arrives = SimplifyingIrBuilder::maybeCastExpr(
        DataType::UInt32,
        GpuLower::current()
            ->parallelDimensionMap()
            .getNumComputeThreadsEachBlock());
  }

  // Initialize mbarrier for each circular buffer stage. Use the thread
  // count from the MBarrierInit created in the allocation pass. The wait
  // condition for mbarrier is a all threads in CTA and the expected number
  // of transaction bytes
  kir::MBarrierInit* mbarrier_init =
      IrBuilder::create<kir::MBarrierInit>(stage_mbarrier, num_of_arrives);

  Expr* pred_mbarrier_init = mbarrier_init->withPredicate(
      IrBuilder::create<kir::Predicate>(PredicateType::ElectSync));
  loop->body().push_back(pred_mbarrier_init);
  return loop;
}

// This helper function invalidates mbarrier for all circular buffer stage after
// TMA memory operations.
//
// Expected result:
// for (unsigned i = 0; i < stages; ++i) {
//   if (warp_id == 0 && electSync()()) {
//     mbarrier::inval(...);
//   }
// }
Expr* invalidateMbarrier(
    ForLoop* circular_buffer_loop,
    TensorView* all_mbarriers,
    CircularBufferWaitType wait_type) {
  NVF_ERROR(circular_buffer_loop != nullptr);
  ForLoop* loop = createStageDepthForLoop(circular_buffer_loop);

  int64_t stage_depth =
      GpuLower::current()
          ->circularBufferInfo()
          .getCircularBufferOptionsFor(circular_buffer_loop->iter_domain())
          .stage;

  // We use mbarrier[0:stage_depth] for RAW, and
  // mbarrier[stage_depth:2*stage_depth] for WAR.
  Val* mbarrier_index = wait_type == CircularBufferWaitType::ReadAfterWrite
      ? loop->index()
      : SimplifyingIrBuilder::addExpr(loop->index(), stage_depth);

  // Get mbarrier for this circular buffer stage.
  kir::TensorIndex* stage_mbarrier =
      IrBuilder::create<kir::TensorIndex>(all_mbarriers, mbarrier_index);

  // Invalidate the mbarrier for each circular buffer stage.
  kir::MBarrierInvalidate* mbarrier_inval =
      IrBuilder::create<kir::MBarrierInvalidate>(stage_mbarrier);

  Expr* pred_mbarrier_inval = mbarrier_inval->withPredicate(
      IrBuilder::create<kir::Predicate>(PredicateType::ElectSync));

  loop->body().push_back(pred_mbarrier_inval);
  return loop;
}

class AllocationInserter : public kir::ExprMutator {
 private:
  using kir::ExprMutator::handle;

  // Expanded version of BasicAllocInfo in lower_utils.h helps to track
  // additional information
  struct AllocationInformation {
    // The for loop that the initialization of this allocation must be
    // placed in, nullptr if not within a loop
    ForLoop* init_for_loop = nullptr;

    // The expression that the initialization of this allocation must
    // be placed before
    Expr* init_place_before = nullptr;

    // Keep track of the actual allocation loop. This can be different
    // from init_for_loop only with unswitched shared memory allocations,
    // which are moved outer loops to avoid duplicated allocations
    // (see issue #1133).
    ForLoop* alloc_for_loop = nullptr;

    // The expression that this allocation must be placed
    // before. Similar to alloc_for_loop, this is different from
    // init_place_before only with unswitched shared memory allocations.
    Expr* alloc_place_before = nullptr;

    // The allocation position relative to buffer
    int64_t alloc_pos = 0;

    // The buffer this allocation is for
    TensorView* buffer = nullptr;

    // Local Iterdomains that this allocation covers
    std::unique_ptr<std::vector<IterDomain*>> allocation_domains;
  };

  // Find allocation point
  // Fills info.buffer, info.alloc_pos, info.init_for_loop,
  // info.init_place_before, info.alloc_for_loop, info.alloc_place_before
  void fillAllocationInformation(AllocationInformation& info, Expr* expr) {
    auto loop_alloc_info =
        lower_utils::getAllocInformation(info.buffer, for_loops_);

    info.init_for_loop = loop_alloc_info.init_for_loop;
    info.alloc_for_loop = loop_alloc_info.alloc_for_loop;
    info.alloc_pos = loop_alloc_info.alloc_pos;

    auto next_fl = [](ForLoop* fl, const std::vector<ForLoop*> fls) {
      for (auto i : c10::irange(fls.size())) {
        if (fl == fls[i]) {
          if (i + 1 < fls.size()) {
            return fls[i + 1];
          }
        }
      }
      NVF_THROW("Could not find desired loop.");
    };

    if (info.init_for_loop == nullptr) {
      info.init_place_before = !for_loops_.empty() ? for_loops_[0] : expr;
    } else {
      if (info.init_for_loop == for_loops_.back()) {
        // Inline allocation, place before expr
        info.init_place_before = expr;
      } else {
        // Place allocation after the last computeAt axis
        // TODO: may be more efficient to place before the first non-computeAt
        // axis
        info.init_place_before = next_fl(info.init_for_loop, for_loops_);
      }
    }

    // Set the allocation loop and the place_before expression in the
    // same way as the initialization loop and place_before expression
    if (info.alloc_for_loop == info.init_for_loop) {
      info.alloc_for_loop = info.init_for_loop;
      info.alloc_place_before = info.init_place_before;
    } else {
      if (info.alloc_for_loop == nullptr) {
        info.alloc_place_before = !for_loops_.empty() ? for_loops_[0] : expr;
      } else {
        // Since there must be an inner unswitched domain,
        // alloc_for_loop should never be the inner-most loop.
        NVF_ERROR(info.alloc_for_loop != for_loops_.back());
        info.alloc_place_before = next_fl(info.alloc_for_loop, for_loops_);
      }
    }
  }

  // Create initialization expression if init_val is non-null.
  Expr* createInitExpr(AllocationInformation& info, Val* init_val) {
    if (init_val == nullptr) {
      return nullptr;
    }

    std::vector<IterDomain*> init_dims;
    for (const auto axis_i :
         c10::irange(info.alloc_pos, info.buffer->nDims())) {
      if (info.buffer->axis(axis_i)->isReduction() ||
          info.buffer->axis(axis_i)->isBroadcast()) {
        continue;
      }
      auto concrete_id =
          lower_utils::getConcreteLoopID(info.buffer->axis(axis_i));
      init_dims.push_back(concrete_id);
    }
    Expr* init_expr = IrBuilder::create<LoadStoreOp>(
        LoadStoreOpType::Set, info.buffer, init_val);
    for (auto init_loop_it = init_dims.rbegin();
         init_loop_it != init_dims.rend();
         ++init_loop_it) {
      auto id = *init_loop_it;
      ForLoop* new_loop = IrBuilder::create<ForLoop>(id);
      new_loop->body().push_back(init_expr);
      init_expr = new_loop;
    }
    return init_expr;
  }

  std::vector<Val*> getGlobalAllocationSizes(AllocationInformation& info) {
    const auto& allocation_domain = info.buffer->getMaybeAllocationDomain();

    std::vector<Val*> alloc_dims;

    for (const auto id : allocation_domain) {
      if (id->isReduction() || id->isStride()) {
        continue;
      } else if (id->isBroadcast() || id->isDeviceDim()) {
        // No matter whether this broadcast is expanded or not, we always
        // allocate size 1
        // Allocate devices axes as size 1
        alloc_dims.emplace_back(id->container()->oneVal());
        continue;
      }
      auto extent = id->extent();
      alloc_dims.emplace_back(extent);
    }

    return alloc_dims;
  }

  std::vector<Val*> getNonGlobalAllocExpr(AllocationInformation& info) {
    const auto memory_type = info.buffer->getMemoryType();
    NVF_ERROR(
        memory_type != MemoryType::Global,
        "Invalid memory type: ",
        memory_type);

    std::vector<Val*> alloc_dims;

    std::vector<IterDomain*> alloc_domains;

    info.allocation_domains = std::make_unique<std::vector<IterDomain*>>();

    // TODO: Today, we always allocate loop domain, even if the allocation
    // domain is explicitly set. This is clearly not the right thing to do,
    // and we should fix this in the future. However, today, we still don't
    // have a clear design of how to allocate tensors with explicit allocation
    // domain. This problem is very difficult to solve, and there are many
    // things to consider. For example, if the allocation domain contains a
    // subset of inlined loop IDs, we should not allocate the inlined IDs.
    // But what if the opposite is true? What if the allocation domain
    // does not contain all inlined IDs? Is this considered an error, or
    // a valid case that we need to infer which to allocate from the loop
    // domain? We need to think about this carefully and come up with a
    // clear design. For now, we just allocate the loop domain for historical
    // reasons for all cases except for the Hopper MMA output tensor.
    //
    // Hopper MMA output tensor is a special case because the loop domain
    // is scheduled in a way that the entire tile is parallelized by MMA, and
    // The TIDx parallelization is a new broadcast dimension that is not
    // connected to any other IterDomains. This way of scheduling effectively
    // makes the loop domain 128x larger than the allocation domain, because
    // the allocation domain is sharded on threads but the loop domain is not.
    if ((info.buffer->definition()->isA<MmaOp>() &&
         isHopper(info.buffer->definition()->as<MmaOp>()->macro()))) {
      const IdModel& id_model = GpuLower::current()->idModel();

      std::unordered_set<IterDomain*> exclude_ca_ids;
      for (auto i : c10::irange(info.alloc_pos)) {
        auto ca_id = info.buffer->axis(i);
        if (!ir_utils::isMemorySharedAcross(
                info.buffer->getMemoryType(), ca_id->getParallelType())) {
          exclude_ca_ids.insert(ca_id);
        }
      }

      const std::vector<IterDomain*>& domain_to_alloc =
          info.buffer->hasAllocation() ? info.buffer->getAllocationDomain()
                                       : info.buffer->getLoopDomain();

      for (auto id : domain_to_alloc) {
        if (exclude_ca_ids.find(id) == exclude_ca_ids.end()) {
          // Don't use reduction/stride/broadcast/device axis in the
          // allocation computation
          if (id->isReduction() || id->isStride() || id->isBroadcast() ||
              id->isDeviceDim()) {
            continue;
          }
          if (ir_utils::isMemoryPartitionedAcross(
                  info.buffer->getMemoryType(), id->getParallelType())) {
            continue;
          }
          info.allocation_domains->push_back(id);

          // Loop promotion may affect allocations. Promotions of intermediate
          // domains may not be defined correctly. Only consider loop domains
          // for now.
          bool is_loop = std::find(
                             info.buffer->getLoopDomain().begin(),
                             info.buffer->getLoopDomain().end(),
                             id) != info.buffer->getLoopDomain().end();
          if (is_loop) {
            id = getLoopPromotion(id, id_model);
          }

          alloc_dims.push_back(id->extent());
        } else {
          exclude_ca_ids.erase(id);
        }
      }
      NVF_ERROR(
          exclude_ca_ids.empty(),
          "The non-allocating compute-at IDs are not found in the allocation domain. ",
          "It is unclear how to allocate the tensor: ",
          info.buffer->toString(),
          " allocation domain: ",
          ir_utils::toString(info.buffer->getAllocationDomain()));

      return alloc_dims;
    }

    for (const auto axis_i : c10::irange(info.buffer->nDims())) {
      const auto local_id = info.buffer->axis(axis_i);

      // Don't use reduction/stride/broadcast/device axis in the
      // allocation computation
      if (local_id->isReduction() || local_id->isStride() ||
          local_id->isBroadcast() || local_id->isDeviceDim()) {
        continue;
      }

      auto concrete_id =
          lower_utils::getConcreteLoopID(info.buffer->axis(axis_i));
      const bool is_block_dim =
          isParallelTypeBlockDim(concrete_id->getParallelType());
      const bool is_thread_dim =
          isParallelTypeThreadDim(concrete_id->getParallelType());
      const bool is_thread =
          isParallelTypeThread(concrete_id->getParallelType());

      if (axis_i < info.alloc_pos) {
        // Even when the axis is outside the allocation position, if the
        // tensor is shared with respect to the axis, the buffer size
        // needs to be expanded for the axis. Sharing occurs in two
        // cases: 1) the tensor is on shared memory with the axis
        // parallelized by TIDs, and 2) the tensor is on global memory
        // with the axis parallelized by TIDs or BIDs.
        if (!((memory_type == MemoryType::Shared && is_thread_dim) ||
              (memory_type == MemoryType::Global && is_thread))) {
          continue;
        }
        alloc_domains.push_back(info.buffer->axis(axis_i));
      } else {
        if (
            // If shared memory, don't use any IDs bound to a grid dimension
            (memory_type == MemoryType::Shared && is_block_dim) ||
            // If local memory, don't use any IDs bound to a grid or block
            // dimension
            (memory_type == MemoryType::Local && is_thread)) {
          continue;
        }
        alloc_domains.push_back(info.buffer->axis(axis_i));
      }

      auto extent = concrete_id->extent();

      alloc_dims.push_back(extent);
      info.allocation_domains->push_back(local_id);
    }

    return alloc_dims;
  }

  kir::Allocate* createAllocExpr(AllocationInformation& info, bool is_output) {
    if (is_output) {
      return nullptr;
    }

    std::vector<Val*> alloc_dims;
    const MemoryType memory_type = info.buffer->getMemoryType();

    if (memory_type == MemoryType::Global) {
      alloc_dims = getGlobalAllocationSizes(info);
    } else {
      alloc_dims = getNonGlobalAllocExpr(info);
    }

    if (alloc_dims.empty() && !info.buffer->domain()->noReductions().empty()) {
      alloc_dims.push_back(info.buffer->container()->oneVal());
    }

    // Multiply the allocation size if circular-buffered. Record the
    // original size for indexing.
    if (info.buffer->isCircularBuffered()) {
      Val* original_alloc_size = nullptr;
      for (auto alloc_dim : alloc_dims) {
        if (original_alloc_size == nullptr) {
          original_alloc_size = alloc_dim;
        } else {
          original_alloc_size =
              IrBuilder::mulExpr(original_alloc_size, alloc_dim);
        }
      }
      GpuLower::current()->circularBufferInfo().setOriginalAllocSize(
          info.buffer, original_alloc_size);
      int64_t circular_buffer_stage =
          info.buffer->circularBufferOptions().stage;
      alloc_dims.push_back(
          IrBuilder::create<Val>(circular_buffer_stage, DataType::Index));
    }

    // Create the allocation node
    auto alloc_expr = IrBuilder::create<kir::Allocate>(
        info.buffer, info.buffer->getMemoryType(), alloc_dims);

    // Fill in the base address, lane offset, and column offset for tensor
    // memory allocations
    if (memory_type == MemoryType::Tensor) {
      const auto& regions = GpuLower::current()->tmemInfo().allocation.regions;
      for (const auto& region : regions) {
        auto tv_info_it = std::find_if(
            region.covered_tensors.begin(),
            region.covered_tensors.end(),
            [&](const auto& tv_info) { return tv_info.tensor == info.buffer; });
        if (tv_info_it != region.covered_tensors.end()) {
          auto address_ti = IrBuilder::create<kir::TensorIndex>(
              region.address, region.address->fusion()->zeroVal());
          alloc_expr->setAddress(address_ti);
          alloc_expr->setLaneOffset(tv_info_it->lane_offset);
          alloc_expr->setColOffset(tv_info_it->column_offset);
          break;
        }
      }
      NVF_ERROR(
          alloc_expr->address() != nullptr,
          "Could not find region for tensor memory allocation of ",
          info.buffer);
    }

    return alloc_expr;
  }

  void dispatch(Expr* expr) override {
    if (!ir_utils::isTvOp(expr) || expr->isA<kir::Allocate>() ||
        expr->isA<kir::AllocTMem>()) {
      ExprMutator::dispatch(expr);
      return;
    }

    int64_t circular_buffer_depth = 1;

    // Found where the allocation needs to be inserted

    for (const auto i : c10::irange(expr->outputs().size())) {
      auto out = expr->output(i);
      if (!out->isA<TensorView>()) {
        continue;
      }

      auto out_tv = out->as<TensorView>();
      auto default_val =
          gpu_lower_->predicateElimination().getInitValue(out_tv);

      Val* init = nullptr;
      if (expr->isA<ReductionOp>() && out_tv->hasReduction()) {
        NVF_ERROR(
            default_val == nullptr,
            "Reduction should not have a default initialization value for predicate elimination.");
        init = expr->as<ReductionOp>()->init();
      } else if (expr->isA<GroupedReductionOp>() && out_tv->hasReduction()) {
        NVF_ERROR(
            default_val == nullptr,
            "Reduction should not have a default initialization value for predicate elimination.");
        init = expr->as<GroupedReductionOp>()->initVal(i);
      } else if (expr->isA<MmaOp>()) {
        init = expr->as<MmaOp>()->init();
      } else if (expr->isA<WelfordOp>()) {
        NVF_ERROR(
            default_val == nullptr,
            "Welford should not have a default initialization value for predicate elimination.");
        const auto welford = expr->as<WelfordOp>();
        if (out->name() == welford->outVar()->name()) {
          init = welford->initVar() == nullptr ? IrBuilder::create<Val>(0.0)
                                               : welford->initVar();
        } else if (out->name() == welford->outAvg()->name()) {
          init = welford->initAvg() == nullptr ? IrBuilder::create<Val>(0.0)
                                               : welford->initAvg();
        } else {
          NVF_ERROR(out->name() == welford->outN()->name(), "Unreachable");
          init = welford->initN();
        }
      } else if (expr->isA<GroupedWelfordOp>()) {
        NVF_ERROR(
            default_val == nullptr,
            "Welford should not have a default initialization value for predicate elimination.");
        init = expr->as<GroupedWelfordOp>()->getInitValOfOutput(out);
      } else {
        init = default_val;
      }

      if (ir_utils::isCpAsyncOp(expr) || ir_utils::isCpAsyncBulk(expr)) {
        NVF_CHECK(
            init == nullptr || init->isZero(),
            "cp.async and cp.async.bulk initialized with non-zero is not supported");
        // cp.async will automatically fill zero when out of bound
        init = nullptr;
      }

      const bool is_output = out->isFusionOutput();

      // Don't need to alloc outputs, and if we don't need to initialize we're
      // done.
      if (is_output && init == nullptr) {
        continue;
      }

      AllocationInformation allocation;
      allocation.buffer = out_tv;
      fillAllocationInformation(allocation, expr);

      auto alloc_expr = createAllocExpr(allocation, is_output);
      auto init_expr = createInitExpr(allocation, init);

      // Check that all circular buffer depth match
      if (out_tv->isCircularBuffered() && circular_buffer_depth == 1) {
        circular_buffer_depth = out_tv->circularBufferOptions().stage;
      }
      NVF_ERROR(
          circular_buffer_depth == 1 ||
              circular_buffer_depth == out_tv->circularBufferOptions().stage,
          "Expected all output TensorViews for the same expression ",
          "to have the same circular_buffer_depth");

      // Write information to GPULower
      writeInfoToGPULower(allocation, alloc_expr);

      // Register allocations before initializations to keep them in the right
      // order
      if (alloc_expr != nullptr) {
        if (allocation.buffer->getMemoryType() == MemoryType::Shared) {
          // Shared allocations go at the begining of scope
          NVF_ERROR(!exprs_.empty());
          registerInsertBefore(exprs_[0], alloc_expr, nullptr);
        } else {
          NVF_ERROR(allocation.alloc_place_before != nullptr);
          Scope* scope = allocation.alloc_for_loop == nullptr
              ? nullptr
              : &allocation.alloc_for_loop->body();
          registerInsertBefore(
              allocation.alloc_place_before, alloc_expr, scope);
        }
      }

      if (init_expr != nullptr) {
        NVF_ERROR(allocation.init_place_before != nullptr);
        Scope* scope = allocation.init_for_loop == nullptr
            ? nullptr
            : &allocation.init_for_loop->body();
        registerInsertBefore(allocation.init_place_before, init_expr, scope);

        if (auto mma = dynamic_cast<MmaOp*>(expr)) {
          if (mma->isHopper()) {
            if (lower_utils::allMmaInputsGuardedByMBarrier(mma)) {
              // When all inputs are guarded by mbarrier, we will not insert
              // generic-async proxy fence and wgmma fence before each mma
              // instruction. For this case, we need to insert these fences
              // after the initialization of the accumulator, so that the
              // initialization is visible to the async proxy.
              // When all inputs are guarded by mbarrier, we will insert these
              // fences before each mma instruction, so there is no need to
              // insert them after the initialization of the accumulator here.
              auto wgmma_fence = IrBuilder::create<kir::WgMmaFence>();
              registerInsertBefore(
                  allocation.init_place_before, wgmma_fence, scope);
              auto fence_async = IrBuilder::create<kir::FenceAsyncProxy>();
              registerInsertBefore(
                  allocation.init_place_before, fence_async, scope);
            }
          }
        }
      }
    }

    // Allocate mbarrier for cp.async.bulk, for non-circular buffered cases by
    // lowering a single cp.async.bulk as:
    //    alloc mbarrier
    //    init mbarrier
    //    block_sync
    //    cp.async.bulk
    //    inval mbarrier
    //    block_sync
    //
    // The circular buffer case is handled in handle(ForLoop* fl) and the
    // circular buffering pass.
    if (ir_utils::isCpAsyncBulkLoad(expr) && circular_buffer_depth == 1) {
      // create and allocate a memory barrier
      TensorView* mbarrier = TensorViewBuilder()
                                 .shape(std::vector<int64_t>{})
                                 .dtype(DataType::UInt64)
                                 .contiguity(true)
                                 .build();
      mbarrier->setMemoryType(MemoryType::Shared);
      auto mbarrier_init = IrBuilder::create<kir::MBarrierInit>(
          mbarrier,
          simplifyExpr(SimplifyingIrBuilder::maybeCastExpr(
              DataType::UInt32,
              lower_utils::getNumThreadsInTensorView(
                  expr->output(0)->as<TensorView>()))));
      auto sync_init = IrBuilder::create<kir::BlockSync>();
      auto mbarrier_inval =
          IrBuilder::create<kir::MBarrierInvalidate>(mbarrier);
      auto sync_inval = IrBuilder::create<kir::BlockSync>();

      kir::Allocate* mbarrier_alloc =
          IrBuilder::create<kir::Allocate>(mbarrier, MemoryType::Shared);
      Scope* expr_scope = scope_.empty() ? nullptr : scope_.back();
      registerInsertBefore(expr, mbarrier_alloc, expr_scope);
      registerInsertBefore(expr, mbarrier_init, expr_scope);
      registerInsertBefore(expr, sync_init, expr_scope);
      registerInsertAfter(expr, mbarrier_inval, expr_scope);
      registerInsertAfter(expr, sync_inval, expr_scope);
      GpuLower::current()->ldstMBarrierMap()[expr] = mbarrier;
    }
  }

  void handle(ForLoop* fl) final {
    ExprMutator::handle(fl);

    // If fl is a circular buffered loop, the we should lowering the loop as:
    //    alloc mbarrier
    //    init mbarrier
    //    block_sync
    //    for-loop with cpAsyncBulk expression (the `fl` parameter)
    //    inval mbarrier

    auto circular_buffer_tvs =
        GpuLower::current()->circularBufferInfo().getCircularBufferTvs(fl);

    bool circular_buffer_load_is_tma = std::any_of(
        circular_buffer_tvs.begin(),
        circular_buffer_tvs.end(),
        [](const TensorView* tv) {
          return ir_utils::isCpAsyncBulkLoad(tv->definition());
        });

    if (circular_buffer_load_is_tma) {
      // Create and allocate a memory barrier. If this is a circular buffer,
      // then allocate an array of mbarier objects. mbarrier::init and
      // mbarrier::inval will be updated in circular buffering pass, but we
      // add them here to handle shared memory correctly in alias memory pass.
      const auto& opt =
          GpuLower::current()->circularBufferInfo().getCircularBufferOptionsFor(
              fl->iter_domain());

      // We use mbarrier[0:stage] for RAW, that is, to wait for the completion
      // of the TMA load of the circular buffer tensor, and
      // mbarrier[stage:2*stage] for WAR, that is, to wait for the completion of
      // the reading of the circular buffer tensor.
      int64_t num_mbarriers =
          opt.usesMBarrierForWAR() ? opt.stage * 2 : opt.stage;

      TensorView* mbarrier = TensorViewBuilder()
                                 .shape(std::vector<int64_t>{num_mbarriers})
                                 .dtype(DataType::UInt64)
                                 .contiguity(true)
                                 .build();
      mbarrier->setMemoryType(MemoryType::Shared);

      kir::Allocate* mbarrier_alloc =
          IrBuilder::create<kir::Allocate>(mbarrier, MemoryType::Shared);

      // Initialize and invalidate mbarriers that are used to notify that
      // the load of the circular buffer is complete.
      auto mbarrier_init_raw = initializeMbarrier(
          fl, mbarrier, CircularBufferWaitType::ReadAfterWrite);
      auto mbarrier_inval_raw = invalidateMbarrier(
          fl, mbarrier, CircularBufferWaitType::ReadAfterWrite);

      // Block sync is necessary to finish mbarrier initialization.
      kir::BlockSync* sync = IrBuilder::create<kir::BlockSync>(false);

      // Add mbarriers, init, and inval operations around tma expression like
      // this:
      //
      // __shared__ mbarrier[num_stages];
      // for (circular_buffer_stage) {
      //   // initialize mbarrier for RAW
      //   init(mbarrier[stage]);
      // }
      // for (circular_buffer_stage) {
      //   // initialize mbarrier for WAR
      //   init(mbarrier[stage]);
      // }
      // block_sync();
      //
      // for (circular_buffer_loop) {
      //   cp.async.bulk(data, mbarrier);
      // }
      //
      // for (circular_buffer_stage) {
      //   // invalidate mbarrier for WAR
      //   inval(mbarrier[stage]);
      // }
      // for (circular_buffer_stage) {
      //   // invalidate mbarrier for RAW
      //   inval(mbarrier[stage]);
      // }
      //
      Scope* current_scope = scope_.empty() ? nullptr : scope_.back();
      registerInsertBefore(fl, mbarrier_alloc, current_scope);
      registerInsertBefore(fl, mbarrier_init_raw, current_scope);
      registerInsertAfter(fl, mbarrier_inval_raw, current_scope);

      if (opt.usesMBarrierForWAR()) {
        auto mbarrier_init_war = initializeMbarrier(
            fl, mbarrier, CircularBufferWaitType::WriteAfterRead);
        auto mbarrier_inval_war = invalidateMbarrier(
            fl, mbarrier, CircularBufferWaitType::WriteAfterRead);
        registerInsertBefore(fl, mbarrier_init_war, current_scope);
        registerInsertAfter(fl, mbarrier_inval_war, current_scope);
      }
      registerInsertBefore(fl, sync, current_scope);

      for (auto tv : circular_buffer_tvs) {
        // short-circuit: circular buffered tv is not defined with TMA load.
        if (!ir_utils::isCpAsyncBulkLoad(tv->definition())) {
          continue;
        }
        // Map LoadStoreOp expression to ir nodes created in this pass
        GpuLower::current()->ldstMBarrierMap()[tv->definition()] = mbarrier;
      }
    }
  }

  // Sends alloc_expr, info.allocation_domains to GpuLower
  void writeInfoToGPULower(
      const AllocationInformation& allocation,
      kir::Allocate* alloc_expr) {
    auto& lower_alloc_info_map = GpuLower::current()->localAllocationInfoMap();
    if (alloc_expr == nullptr) {
      // Skip output allocation.
      return;
    }
    NVF_ERROR(
        !lower_alloc_info_map.count(alloc_expr),
        "duplicated allocation info entry");

    // Create info entry for GPULower
    auto lower_alloc_info_ptr = std::make_unique<LocalAllocationInfo>();
    lower_alloc_info_ptr->alloc_expr = alloc_expr;
    if (allocation.allocation_domains) {
      lower_alloc_info_ptr->alloc_domains = *(allocation.allocation_domains);
    }

    // Write entry to the stored map
    lower_alloc_info_map[alloc_expr] = std::move(lower_alloc_info_ptr);
  }

  void handle(kir::IfThenElse*) final {
    NVF_THROW(
        "Pass does not support conditional statements, ",
        "this pass should be run before any conditionals are placed in code.");
  }

  AllocationInserter(const std::vector<Expr*>& exprs)
      : gpu_lower_(GpuLower::current()) {
    kir::ExprMutator::traverseAndInsert(exprs);
  }

 private:
  GpuLower* gpu_lower_ = nullptr;

 public:
  static std::vector<Expr*> insert(const std::vector<Expr*>& exprs) {
    AllocationInserter inserter(exprs);
    return inserter.exprs_;
  }
};

// Insert IR nodes that allocate and deallocate TMem regions.
// See note [Tensor Memory Allocation] for the overall design.
// We insert the tcgen05.allocs of each region and the relinquish of the right
// to allocate at the beginning of the top-level scope of the kernel. We insert
// the tcgen05.deallocs after the outermost serial loop containing the last read
// of each TMem region into whatever scope containing this outermost serial
// loop. The allocation of each TMem TensorView within each region is inserted
// by AllocationInserter::insert, therefore not handled here.
std::vector<Expr*> insertTMemRegionAllocsAndDeallocs(
    const std::vector<Expr*>& exprs) {
  // Expressions to be inserted at the beginning of the top-level scope.
  std::list<Expr*> prologue;
  {
    const auto& regions = GpuLower::current()->tmemInfo().allocation.regions;
    // For each TMem region, allocate its address in shared memory, and insert
    // the tcgen05.alloc for tensor memory allocation.
    for (const auto& region : regions) {
      // kir::Allocate for the address tensor on shared memory
      auto address_alloc_expr =
          IrBuilder::create<kir::Allocate>(region.address, MemoryType::Shared);
      prologue.push_back(address_alloc_expr);
      // the tcgen05.alloc instruction
      auto alloc_expr =
          IrBuilder::create<kir::AllocTMem>(region.address, region.num_columns);
      prologue.push_back(alloc_expr);
    }

    if (!regions.empty()) {
      // Relinquish the right to allocate after all regions have been allocated
      auto tcgen05_relinquish_expr = IrBuilder::create<kir::Asm>(
          "tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned",
          std::vector<Val*>{},
          std::vector<Val*>{},
          kir::Asm::Options{/*volatile=*/true});
      prologue.push_back(tcgen05_relinquish_expr);

      // Block sync that makes allocation visible to all threads
      auto block_sync = IrBuilder::create<kir::BlockSync>();
      prologue.push_back(block_sync);
    }
  }

  // Add deallocations to existing expressions
  std::vector<Expr*> exprs_with_deallocs;
  {
    class DeallocInserter : public kir::ExprMutator {
      // A map:
      //   region -> a function that registers the deallocation expression for
      //             this region
      //
      // This map is updated during traversal. For example, if we have a kernel
      // like below:
      //   ...
      //   T1_t = T0_r; // expr1
      //   ...
      //   T2_r = T1_t; // expr2
      // Assume that T1_t is in region R1. Then after we handle(expr1), we will
      // have an entry:
      //    R1 -> a function registering insertion of "dealloc R1" after expr1
      // After handle(expr2), this entry becomes:
      //    R1 -> a function registering insertion of "dealloc R1" after expr2
      //
      // After traversing the entire kernel, this map will contain the final
      // register functions we want to execute.
      std::unordered_map<
          const TMemAlllocationInfo::Region*,
          std::function<void()>>
          region_to_register_dealloc_map_;

      // A map:
      //   expr -> the regions that this expr is accessing
      // Note that if expr is a container such as ForLoop or IfThenElse, then
      // the mapped regions will be all the regions the contained exprs are
      // accessing.
      //
      // This map only contain information of accesses that we have discovered,
      // and is updated during traversal. For example, if we have a kernel:
      //   ForLoop: // loop1
      //     T2_t = T0_r; // expr1
      //     ...
      //     T3_t = T1_r; // expr2
      // Assume T2_t is in region R2 and T3_t is in region R3. Then after
      // handle(expr1), we will have a map:
      //    expr1 -> {R2}
      //    loop1 -> {R2}
      // After handle(expr2), this map becomes:
      //    expr1 -> {R2}
      //    expr2 -> {R3}
      //    loop1 -> {R2, R3}
      std::unordered_map<
          Expr*,
          VectorOfUniqueEntries<const TMemAlllocationInfo::Region*>>
          access_map_;

      // Analyze expr to see if it has any accesses to tensor memory. If yes
      // update the access map for this expr and its container exprs.
      void updateAccessMap(Expr* expr) {
        std::unordered_set<Val*> io_vals;
        std::copy(
            expr->inputs().begin(),
            expr->inputs().end(),
            std::inserter(io_vals, io_vals.end()));
        std::copy(
            expr->outputs().begin(),
            expr->outputs().end(),
            std::inserter(io_vals, io_vals.end()));
        if (io_vals.empty()) {
          return;
        }
        for (const auto& region :
             GpuLower::current()->tmemInfo().allocation.regions) {
          for (auto tv_info : region.covered_tensors) {
            if (io_vals.count(tv_info.tensor)) {
              access_map_[expr].pushBack(&region);
              for (auto container : scope_exprs_) {
                access_map_[container].pushBack(&region);
              }
            }
            break;
          }
        }
      }

      // Update the region_to_register_dealloc_map_ to register insertion of
      // deallocation expression after expr for the regions accessed by expr.
      void maybeRegisterDeallocsAfterExpr(Expr* expr) {
        // If expr is a trivial for loop, then we don't need to move the
        // deallocation after it. This is because the trivial is not generated
        // in the final code.
        if (auto fl = dynamic_cast<ForLoop*>(expr)) {
          if (fl->isTrivial()) {
            return;
          }
        }
        // If expr is not accessing any tensor memory, then nothing to do.
        if (!access_map_.count(expr)) {
          return;
        }
        for (auto region : access_map_.at(expr)) {
          auto current_scope = scope_.empty() ? nullptr : scope_.back();
          region_to_register_dealloc_map_[region] =
              [this, expr, region, current_scope]() {
                auto tcgen05_dealloc_expr = IrBuilder::create<kir::Asm>(
                    "tcgen05.dealloc.cta_group::1.sync.aligned.b32",
                    std::vector<Val*>{},
                    std::vector<Val*>{
                        IrBuilder::create<kir::TensorIndex>(
                            region->address, expr->fusion()->zeroVal()),
                        region->num_columns},
                    kir::Asm::Options{/*volatile=*/true});
                registerInsertAfter(expr, tcgen05_dealloc_expr, current_scope);
              };
        }
      }

      void dispatch(Expr* expr) final {
        updateAccessMap(expr);
        ExprMutator::dispatch(expr);
        maybeRegisterDeallocsAfterExpr(expr);
      }

     public:
      DeallocInserter(
          const std::vector<Expr*>& exprs,
          std::vector<Expr*>& exprs_with_deallocs) {
        handle(exprs);
        for (const auto& region :
             GpuLower::current()->tmemInfo().allocation.regions) {
          region_to_register_dealloc_map_.at (&region)();
        }
        exprs_with_deallocs = mutate();
      }
    } inserter(exprs, exprs_with_deallocs);
  }

  // Combine prologue and exprs_with_deallocs
  std::vector<Expr*> result;
  result.reserve(prologue.size() + exprs_with_deallocs.size());
  result.insert(result.end(), prologue.begin(), prologue.end());
  result.insert(
      result.end(), exprs_with_deallocs.begin(), exprs_with_deallocs.end());
  return result;
}

} // namespace

std::vector<Expr*> insertAllocations(const std::vector<Expr*>& exprs) {
  FUSER_PERF_SCOPE("GpuLower::Lower::insertAllocations");
  // If the fusion uses tensor memory, insert the following things to the
  // fusion:
  // - A tcgen05.alloc for each tensor memory region
  // - A kir::Allocate for a shared memory TensorView for each tensor memory
  //   region for storing addresses of these regions. Because tcgen05.alloc
  //   writes the address of allocated memory to the shared memory, there must
  //   be shared memory TensorViews to store these addresses. These address
  //   TensorViews are not part of the fusion math, and not handled by
  //   AllocationInserter::insert. Note that these address TensorViews are not
  //   the tensor memory TensorViews in fusion math.
  // - A tcgen05.relinquish_alloc_permit after all tcgen05.allocs
  auto result = insertTMemRegionAllocsAndDeallocs(exprs);
  // Insert kir::Allocate for each Val, including the kir::Allocate for tensor
  // memory TensorViews, in fusion math.
  return AllocationInserter::insert(result);
}

} // namespace nvfuser
