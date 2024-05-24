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
#include <instrumentation.h>
#include <ir/iostream.h>
#include <kernel_ir.h>
#include <kernel_ir_dispatch.h>

#include <unordered_set>

namespace nvfuser {

namespace {

class AllocationInserter : public kir::ExprMutator {
 private:
  using kir::ExprMutator::handle;

  // Expanded version of BasicAllocInfo in lower_utils.h helps to track
  // additional information
  struct AllocationInformation {
    // The for loop that the initialization of this allocation must be
    // placed in, nullptr if not within a loop
    kir::ForLoop* init_for_loop = nullptr;

    // The expression that the initialization of this allocation must
    // be placed before
    Expr* init_place_before = nullptr;

    // Keep track of the actual allocation loop. This can be different
    // from init_for_loop only with unswitched shared memory allocations,
    // which are moved outer loops to avoid duplicated allocations
    // (see issue #1133).
    kir::ForLoop* alloc_for_loop = nullptr;

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

    auto next_fl = [](kir::ForLoop* fl, const std::vector<kir::ForLoop*> fls) {
      for (auto i : c10::irange(fls.size())) {
        if (fl == fls[i]) {
          if (i + 1 < fls.size()) {
            return fls[i + 1];
          }
        }
      }
      NVF_ERROR(false, "Could not find desired loop.");
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
      auto concrete_id = gpu_lower->caMap()->getConcreteMappedID(
          info.buffer->axis(axis_i), IdMappingMode::LOOP);
      init_dims.push_back(concrete_id);
    }
    Expr* init_expr = IrBuilder::create<LoadStoreOp>(
        LoadStoreOpType::Set, info.buffer, init_val);
    for (auto init_loop_it = init_dims.rbegin();
         init_loop_it != init_dims.rend();
         ++init_loop_it) {
      auto id = *init_loop_it;
      kir::ForLoop* new_loop = IrBuilder::create<kir::ForLoop>(id);
      new_loop->body().push_back(init_expr);
      init_expr = new_loop;
    }
    return init_expr;
  }

  std::vector<Val*> getGlobalAllocationSizes(AllocationInformation& info) {
    const auto& maybe_rfactor_domain = info.buffer->getMaybeAllocationDomain();

    std::vector<Val*> alloc_dims;

    for (const auto id : maybe_rfactor_domain) {
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

    for (const auto axis_i : c10::irange(info.buffer->nDims())) {
      const auto local_id = info.buffer->axis(axis_i);

      // Don't use reduction/stride/broadcast/device axis in the
      // allocation computation
      if (local_id->isReduction() || local_id->isStride() ||
          local_id->isBroadcast() || local_id->isDeviceDim()) {
        continue;
      }

      auto concrete_id = gpu_lower->caMap()->getConcreteMappedID(
          info.buffer->axis(axis_i), IdMappingMode::LOOP);
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

    // Double the allocation size if double-buffered. Record the
    // original size for indexing.
    if (info.buffer->isDoubleBuffered() || info.buffer->isCircularBuffered()) {
      Val* original_alloc_size = nullptr;
      for (auto alloc_dim : alloc_dims) {
        if (original_alloc_size == nullptr) {
          original_alloc_size = alloc_dim;
        } else {
          original_alloc_size =
              IrBuilder::mulExpr(original_alloc_size, alloc_dim);
        }
      }
      GpuLower::current()->doubleBufferInfo().setOriginalAllocSize(
          info.buffer, original_alloc_size);
      int64_t double_buffer_stage = 2L;
      if (info.buffer->isCircularBuffered()) {
        double_buffer_stage = (int64_t)info.buffer->circularBufferDepth();
      }
      alloc_dims.push_back(
          IrBuilder::create<Val>(double_buffer_stage, DataType::Index));
    }

    // Create the allocation node
    return IrBuilder::create<kir::Allocate>(
        info.buffer, info.buffer->getMemoryType(), alloc_dims);
  }

  void dispatch(Expr* expr) override {
    if (!ir_utils::isTvOp(expr) || expr->isA<kir::Allocate>()) {
      ExprMutator::dispatch(expr);
      return;
    }

    // Found where the allocation needs to be inserted

    for (const auto i : c10::irange(expr->outputs().size())) {
      auto out = expr->output(i);
      if (!out->isA<TensorView>()) {
        continue;
      }

      auto out_tv = out->as<TensorView>();
      auto default_val = gpu_lower->predicateElimination().getInitValue(out_tv);

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

      if (ir_utils::isCpAsyncOp(expr)) {
        NVF_CHECK(
            init == nullptr || init->isZero(),
            "cp.async initialized with non-zero is not supported");
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
          kir::Scope* scope = allocation.alloc_for_loop == nullptr
              ? nullptr
              : &allocation.alloc_for_loop->body();
          registerInsertBefore(
              allocation.alloc_place_before, alloc_expr, scope);
        }
      }

      if (init_expr != nullptr) {
        NVF_ERROR(allocation.init_place_before != nullptr);
        kir::Scope* scope = allocation.init_for_loop == nullptr
            ? nullptr
            : &allocation.init_for_loop->body();
        registerInsertBefore(allocation.init_place_before, init_expr, scope);
      }
    }

    // Allocate mbarrier for cp.async.bulk, note that this is only a temporary
    // solution, we should remove this after we have a better way to handle
    // synchronizations for cp.async.bulk.
    if (ir_utils::isCpAsyncBulkLoad(expr)) {
      // create and allocate a memory barrier
      TensorView* mbarrier = TensorViewBuilder()
                                 .shape(std::vector<int64_t>{})
                                 .dtype(DataType::UInt)
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
      kir::Scope* expr_scope = scope_.empty() ? nullptr : scope_.back();
      registerInsertBefore(expr, mbarrier_alloc, expr_scope);
      registerInsertBefore(expr, mbarrier_init, expr_scope);
      registerInsertBefore(expr, sync_init, expr_scope);
      registerInsertAfter(expr, mbarrier_inval, expr_scope);
      registerInsertAfter(expr, sync_inval, expr_scope);
      GpuLower::current()->ldstMBarrierMap()[expr] = mbarrier;
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
    NVF_ERROR(
        false,
        "Pass does not support conditional statements, ",
        "this pass should be run before any conditionals are placed in code.");
  }

  AllocationInserter(const std::vector<Expr*>& exprs)
      : gpu_lower(GpuLower::current()) {
    kir::ExprMutator::traverseAndInsert(exprs);
  }

 private:
  GpuLower* gpu_lower;

 public:
  static std::vector<Expr*> insert(const std::vector<Expr*>& exprs) {
    AllocationInserter inserter(exprs);
    return inserter.exprs_;
  }
};

} // namespace

std::vector<Expr*> insertAllocations(const std::vector<Expr*>& exprs) {
  FUSER_PERF_SCOPE("GpuLower::Lower::insertAllocations");
  return AllocationInserter::insert(exprs);
}

} // namespace nvfuser
