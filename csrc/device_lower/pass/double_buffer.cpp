// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#include <device_lower/lower2device.h>
#include <ir/utils.h>
#include <kernel_ir.h>
#include <transform_iter.h>
#include <transform_replay.h>

#include <device_lower/pass/double_buffer.h>

#include <algorithm>
#include <iterator>
#include <vector>

namespace nvfuser {

int64_t getDoubleBufferAxisPosition(const TensorView* tv) {
  // Double-buffering prefetches the next subregion of the tensor by
  // doubling the allocation. The subregion is defined by the axes
  // at the CA position till the inner-most position. There must be
  // at least one axis that is outside (left) of the CA position,
  // which defines the loop where prefetching is applied. Therefore,
  // the CA position must be larger than 0.

  NVF_ERROR(tv->getComputeAtPosition() > 0);

  // Unroll must not exist outside of double-buffer axis
  auto first_unroll_it = std::find_if(
      tv->getLeafDomain().begin(),
      tv->getLeafDomain().end(),
      [](const auto axis) {
        return axis->getParallelType() == ParallelType::Unroll;
      });

  const int64_t first_unroll_pos =
      (int64_t)std::distance(tv->getLeafDomain().begin(), first_unroll_it);

  const int64_t unroll_or_ca_pos =
      std::min(tv->getComputeAtPosition(), first_unroll_pos);

  NVF_ERROR(
      unroll_or_ca_pos > 0,
      "Invalid tensor to double-buffer. Valid double buffer axis not found due to Unroll. ",
      tv->toString());

  int64_t valid_pos = -1;
  // Skip parallelized or broadcast axes
  for (int64_t i = unroll_or_ca_pos - 1; i >= 0; --i) {
    auto pt = tv->axis(i)->getParallelType();
    if (!isParallelTypeThread(pt) && !tv->axis(i)->isBroadcast()) {
      valid_pos = i;
      break;
    }
  }

  NVF_ERROR(
      valid_pos >= 0,
      "Invalid tensor to double-buffer. Valid double buffer axis not found. ",
      tv->toString());

  return valid_pos;
}

IterDomain* getDoubleBufferAxis(const TensorView* tv) {
  return tv->axis(getDoubleBufferAxisPosition(tv));
}

void validateDoubleBufferedTensor(const TensorView* tv) {
  auto double_buffer_pos = getDoubleBufferAxisPosition(tv);

  // Like vectorization, only LoadStoreOp with another TensorView is
  // considered.
  auto def = tv->definition();
  NVF_ERROR(
      def->isA<LoadStoreOp>(),
      "Invalid tensor to double-buffer. Only tensor defined by LoadStoreOp is supported: ",
      def->toString());

  NVF_ERROR(
      def->input(0)->isA<TensorView>(),
      "Invalid tensor to double-buffer. Only tensor defined by LoadStoreOp with TensorView is supported: ",
      def->toString());

  NVF_ERROR(
      !tv->hasComputeWith(),
      "computeWith is not supported with double buffering: ",
      tv->toString());

  // Require the producer tensor to have been computed entirely for
  // the double-buffering loop. Otherwise, the producer itself would
  // also need to be double-bufferred.
  auto producer = def->input(0)->as<TensorView>();
  NVF_ERROR(
      producer->getComputePosition(tv) <= double_buffer_pos,
      "Invalid tensor to double-buffer. The computeAt position of the producer tensor must be moved left: ",
      producer->toString());

  // Not strictly necessary, but only gmem -> smem or local and smem -> local
  // are allowed.
  const auto p_mem_type = producer->getMemoryType();
  const auto c_mem_type = tv->getMemoryType();
  NVF_ERROR(
      (p_mem_type == MemoryType::Global &&
       (c_mem_type == MemoryType::Shared || c_mem_type == MemoryType::Local)) ||
          (p_mem_type == MemoryType::Shared && c_mem_type == MemoryType::Local),
      "Invalid tensor to double-buffer: ",
      tv->toString(),
      ". Producer memory type: ",
      p_mem_type,
      ". Consumer memory type: ",
      c_mem_type);

  return;
}

namespace {

// Initial inspection of a fusion to find and validate double buffered tensors
class DoubleBufferFusionInspector : private IterVisitor {
 public:
  DoubleBufferFusionInspector(Fusion* fusion, DoubleBufferInfo& db_info)
      : db_info_(db_info) {
    traverse(fusion);
  }

 private:
  using IterVisitor::handle;

  void handle(TensorView* tv) final {
    if (!(tv->isDoubleBuffered() || tv->isCircularBuffered())) {
      return;
    }

    NVF_ERROR(
        tv->definition(), "Fusion input shouldn't be double buffered.", tv);

    validateDoubleBufferedTensor(tv);

    auto db_axis = getDoubleBufferAxis(tv);

    db_info_.setDoubleBufferAxis(tv, db_axis);
  }

 private:
  DoubleBufferInfo& db_info_;
};

// Creates kir::IfThenElse with the following predicate:
// threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0
// TODO Replace with elect.sync ptx
kir::IfThenElse* createThreadPredicatedIfThenElse() {
  Val* zero_val = IrBuilder::create<Val>(0L, PrimDataType::UInt);
  Val* if_predicate_expr = IrBuilder::logicalAndExpr(
      IrBuilder::logicalAndExpr(
          IrBuilder::eqExpr(
              NamedScalar::getParallelIndex(ParallelType::TIDx), zero_val),
          IrBuilder::eqExpr(
              NamedScalar::getParallelIndex(ParallelType::TIDy), zero_val)),
      IrBuilder::eqExpr(
          NamedScalar::getParallelIndex(ParallelType::TIDz), zero_val));

  kir::IfThenElse* if_expr = IrBuilder::create<kir::IfThenElse>(
      IrBuilder::create<kir::Predicate>(if_predicate_expr));

  return if_expr;
}

// Creates kir::Loop with range based on stage depth
kir::ForLoop* createStagesForLoop(kir::ForLoop* double_buffer_loop) {
  int64_t stage_depth =
      GpuLower::current()->doubleBufferInfo().getStageDepthFor(
          double_buffer_loop->iter_domain());

  Val* loop_start = IrBuilder::create<Val>(0L, PrimDataType::Index);
  Val* loop_index = IrBuilder::create<Val>(PrimDataType::Index);
  Val* loop_extend = IrBuilder::create<Val>(stage_depth, PrimDataType::Index);
  IterDomainBuilder loop_domain_builder(loop_start, loop_extend);
  Val* loop_step = IrBuilder::create<Val>(1L, PrimDataType::Index);

  kir::ForLoop* loop = IrBuilder::create<kir::ForLoop>(
      loop_domain_builder.build(),
      loop_index,
      loop_start,
      loop_extend,
      loop_step,
      /*vectorize=*/false,
      /*vectorize_shift=*/nullptr,
      /*unroll_required=*/false,
      DoubleBufferLoopStage::NotApplicable);

  return loop;
}

// Get expected transaction count for given tma operation.
Val* getExpectedTransactionCount(LoadStoreOp* ldst) {
  TensorView* producer_tv = ldst->in()->as<TensorView>();
  TensorView* consumer_tv = ldst->out()->as<TensorView>();

  NVF_ERROR(
      producer_tv->getMemoryType() == MemoryType::Global &&
          consumer_tv->getMemoryType() == MemoryType::Shared,
      "Expected Transaction Count is necessary for load operation");

  auto allocation_domain = TensorDomain::noBroadcasts(
      TensorDomain::noReductions(producer_tv->getMaybeAllocationDomain()));
  std::unordered_set<Val*> allocation_domain_set(
      allocation_domain.begin(), allocation_domain.end());

  // Replay producer to look like consumer so we can index on producer since
  // our loop nests look like consumer
  auto pairwise_map =
      PairwiseRootDomainMap(producer_tv, consumer_tv).mapBroadcast(true);

  TensorDomain* producerAsC = TransformReplay::replayPasC(
                                  producer_tv,
                                  consumer_tv,
                                  -1,
                                  pairwise_map,
                                  TransformReplayOptions().replayResize())
                                  .first;

  // Make the producer_tv look like consumer while performing indexing math
  // For TMA load, we need to replay the gmem tensor as consumer.
  std::unique_ptr<ir_utils::TVDomainGuard> domain_guard =
      std::make_unique<ir_utils::TVDomainGuard>(producer_tv, producerAsC);

  // Map sent to best effort replay needs to match the exact incantation for
  // compute_at_mode.cpp with MappingMode::Index
  auto c2p_root_map = PairwiseRootDomainMap(producer_tv, consumer_tv)
                          .mapBroadcast(false)
                          .mapConsumerToProducer();

  // This replay has to be consistent with compute at index map.
  BestEffortReplay replay_producer_as_consumer(
      producer_tv->getLeafDomain(), consumer_tv->getLeafDomain(), c2p_root_map);

  const auto& c2p_map = replay_producer_as_consumer.getReplay();

  // Convert an id from the consumer tensor to its corresponding id in the
  // gmem tensor. If the consumer tensor is already a gmem tensor, then the
  // function is the identity function. Otherwise, the function is the
  // consumer-to-producer map.
  auto consumer_to_gmem = [=](IterDomain* id) -> IterDomain* {
    return c2p_map.at(id);
  };

  // Step 1: Get all bulk IterDomains and tile IterDomains.
  // An IterDomain is considered "bulk" if it has parallel type "Bulk" or all
  // its children are considered "bulk".
  // A "tile" IterDomain is a bulk IterDomain whose parents are not bulk.

  // Get all bulk IterDomains
  std::unordered_set<IterDomain*> bulk_ids;
  // Bulk IterDomains that we need to check its definition to see if it is a
  // tile IterDomain.
  std::deque<IterDomain*> pending;
  pending.push_back(nullptr); // use nullptr as a checkpoint
  // Start from leaf domain, where all the bulk IterDomains in the leaf domain
  // must be parallelized as ParallelType::Bulk.
  for (auto id : consumer_tv->getLeafDomain()) {
    if (id->getParallelType() == ParallelType::Bulk) {
      id = consumer_to_gmem(id);
      bulk_ids.insert(id);
      pending.push_back(id);
    }
  }
  // Use a BFS-like (not exactly BFS) algorithm to propagate back to get all
  // bulk IterDomains
  bool updated = true;
  while (true) {
    auto id = pending.front();
    pending.pop_front();
    if (id == nullptr) {
      if (updated) {
        // We discovered new bulk IterDomains in the last round, so we need to
        // continue start a new round to see if we can discover more bulk
        // IterDomains.
        pending.push_back(nullptr);
        updated = false;
        continue;
      } else {
        // We have visited all IterDomains in pending for one round, but nothing
        // has changed. This means that all IterDomains in pending are
        // tile IterDomains, so we can no longer propagate further.
        break;
      }
    }

    auto def = id->definition();
    bool should_propagate = false;
    if (allocation_domain_set.count(id) == 0) {
      // We only continue propagating if we have not reached the allocation
      // domain yet.
      NVF_ERROR(
          def != nullptr,
          "Allocation domain is unreachable from ",
          id->toString());

      if (bulk_ids.count(def->input(0)->as<IterDomain>()) > 0) {
        // already processed from another path
        continue;
      }

      should_propagate = std::all_of(
          def->outputs().begin(), def->outputs().end(), [&](Val* out) {
            return bulk_ids.count(out->as<IterDomain>()) > 0;
          });
    }

    if (should_propagate) {
      updated = true;
      for (auto id : def->inputs()) {
        if (bulk_ids.insert(id->as<IterDomain>()).second) {
          pending.push_back(id->as<IterDomain>());
        }
      }
    } else {
      // Not all outputs of def are bulk IterDomains, this could be because:
      // 1. id is a tile IterDomain
      // 2. id is not a tile IterDomain, we just haven't visited def's other
      //    outputs yet.
      pending.push_back(id);
    }
  }

  // Get tile IterDomains. Use VectorOfUniqueEntries instead of
  // std::unordered_set to make the algorithm deterministic. However, the order
  // here has no meaning, especially, is is not the order specifying which
  // IterDomain is inner and which is outer. The actual order must be determined
  // by propagating from the allocation domain.
  VectorOfUniqueEntries<IterDomain*> tile_ids;
  for (auto id : pending) {
    if (id == nullptr) {
      continue;
    }
    tile_ids.pushBack(id);
  }

  // Step 5: Compute the expected bytes for the complete_tx mechanism

  int64_t itemsize = dataTypeSize(producer_tv->dtype());
  Val* expected_bytes = IrBuilder::create<Val>(itemsize, DataType::Index);
  // Note that we need to use the extents of the tile IterDomains
  // to compute the expected bytes, not the extents of the box IterDomains.
  // They are different when element strides are not 1.
  for (auto id : tile_ids) {
    expected_bytes =
        SimplifyingIrBuilder::mulExpr(expected_bytes, id->extent());
  }
  return expected_bytes;
}

// Creates kir::MBarrierArriveExpectTx for given LoadStoreOp and index of
// loop in scope's which LoadStoreOp is present
//
// Example:
// __shared__ __mbarrier_t barriers[num_stages];
// __shared__ __mbarrier_token_t tokens[num_stages];
// for(nvfuser_index_t stage = 0; stage < num_stages; ++stage) {
//   if (elect_sync()) {
//     tokens[stage] =
//        mbarrier::arriveExpectTX(toSmem((&barriers[stage])), expected_bytes);
//   }
// }
kir::MBarrierArriveExpectTx* createMbarrierArriveExpectTx(
    LoadStoreOp* ldst,
    Val* loop_index) {
  // Get expected bytes for single TMA load operation
  Val* expected_bytes = getExpectedTransactionCount(ldst);

  // The expected_bytes for mbarrier::arriveExpectTX must account for all TMA
  // load operations launched for each circular buffer stage. We take the
  // product of all coordinate TMA iterDomains to the right of the circular
  // buffer axis.
  TensorView* ldst_out_tv = ldst->out()->as<TensorView>();
  const std::vector<IterDomain*>& leaf_domain = ldst_out_tv->getLeafDomain();
  for (size_t idx = ldst_out_tv->getComputeAtPosition();
       idx < leaf_domain.size();
       ++idx) {
    IterDomain* id = leaf_domain.at(idx);
    if (!isParallelTypeThread(id->getParallelType()) &&
        id->getParallelType() != ParallelType::Bulk) {
      expected_bytes =
          SimplifyingIrBuilder::mulExpr(expected_bytes, id->extent());
    }
  }

  expected_bytes =
      SimplifyingIrBuilder::maybeCastExpr(DataType::UInt32, expected_bytes);
  auto is_multiple_of_16B = SimplifyingIrBuilder::eqExpr(
      SimplifyingIrBuilder::modExpr(
          expected_bytes, IrBuilder::create<Val>(16, DataType::Index)),
      expected_bytes->fusion()->zeroVal());
  GpuLower::current()->validate(
      is_multiple_of_16B,
      "The expected bytes must be a multiple of 16 bytes, but ",
      expected_bytes,
      " is not.");

  TensorView* all_mbarrier_tokens =
      GpuLower::current()->ldstMBarrierTokenMap().at(ldst);
  kir::TensorIndex* stage_token =
      IrBuilder::create<kir::TensorIndex>(all_mbarrier_tokens, loop_index);

  TensorView* all_mbarriers = GpuLower::current()->ldstMBarrierMap().at(ldst);
  kir::TensorIndex* stage_mbarrier =
      IrBuilder::create<kir::TensorIndex>(all_mbarriers, loop_index);

  auto mbarrier_arrive_tx = IrBuilder::create<kir::MBarrierArriveExpectTx>(
      stage_token, stage_mbarrier, expected_bytes);

  return mbarrier_arrive_tx;
}

// Creates kir::MBarrierWait for given LoadStoreOp and loop index
kir::MBarrierWait* createMbarrierWait(LoadStoreOp* ldst, Val* loop_index) {
  TensorView* all_mbarriers = GpuLower::current()->ldstMBarrierMap().at(ldst);
  kir::TensorIndex* stage_mbarrier =
      IrBuilder::create<kir::TensorIndex>(all_mbarriers, loop_index);

  TensorView* all_mbarrier_tokens =
      GpuLower::current()->ldstMBarrierTokenMap().at(ldst);
  kir::TensorIndex* stage_token =
      IrBuilder::create<kir::TensorIndex>(all_mbarrier_tokens, loop_index);

  kir::MBarrierWait* mbarrier_wait =
      IrBuilder::create<kir::MBarrierWait>(stage_mbarrier, stage_token);
  return mbarrier_wait;
}

// The epilogue loop is only created when the producer of a double
// buffer tensor is on smem, in which case it would otherwise require
// an additional predicate to guard buffer overruns. When it's on
// gmem, that isn't the case, so it does not need to create an
// epilogue loop. For TMA cpAsyncBulk, there is always an epilogue loop.
bool requireEpilogue(const std::vector<Expr*>& exprs) {
  return std::any_of(exprs.begin(), exprs.end(), [](const Expr* expr) {
    return (expr->input(0)->as<TensorView>()->getMemoryType() ==
            MemoryType::Shared) ||
        (expr->as<LoadStoreOp>()->opType() ==
         LoadStoreOpType::CpAsyncBulkTensorTile);
  });
}

// Replicates double buffer loops for Prologue, Main, and
// Epilogue. Prologue only copies the load expressions of double
// buffered tensors, whereas Epilogue does any expression other than
// the loads. Main copies everything.
class DoubleBufferLoopCloner : public kir::IrVisitor {
 public:
  static kir::ForLoop* clone(
      kir::ForLoop* double_buffer_loop,
      const std::vector<Expr*>& double_buffer_load_exprs,
      DoubleBufferLoopStage loop_type,
      const std::unordered_set<Expr*>& exclude = {}) {
    DoubleBufferLoopCloner cloner(
        double_buffer_loop, double_buffer_load_exprs, loop_type, exclude);
    cloner.duplicate();
    return cloner.cloned_top_level_loop_;
  }
  virtual ~DoubleBufferLoopCloner() = default;

 protected:
  DoubleBufferLoopCloner(
      kir::ForLoop* double_buffer_loop,
      const std::vector<Expr*>& double_buffer_load_exprs,
      DoubleBufferLoopStage loop_type,
      const std::unordered_set<Expr*>& exclude)
      : double_buffer_loop_(double_buffer_loop),
        double_buffer_load_exprs_(double_buffer_load_exprs),
        loop_type_(loop_type),
        exclude_(exclude) {}

  using kir::IrVisitor::handle;

  void duplicate() {
    // Cloning the double buffer loop as follows:
    //
    // Prologue: 0 to 1
    // Main: 0 to (extent-1)
    // Epilogue: (extent-1) to extent

    auto index = GpuLower::current()->caMap()->getIndexVariable(
        double_buffer_loop_->iter_domain(), loop_type_);
    auto start = double_buffer_loop_->start();
    auto stop = double_buffer_loop_->stop();
    auto stage_depth = GpuLower::current()->doubleBufferInfo().getStageDepthFor(
        double_buffer_loop_->iter_domain());

    if (loop_type_ == DoubleBufferLoopStage::Prolog) {
      NVF_ERROR(start->isZeroInt());
      stop = SimplifyingIrBuilder::create<Val>(
          int64_t(stage_depth - 1), DataType::Index);
    } else if (
        loop_type_ == DoubleBufferLoopStage::Main &&
        requireEpilogue(double_buffer_load_exprs_)) {
      stop = IrBuilder::subExpr(
          double_buffer_loop_->stop(), GpuLower::current()->kernel()->oneVal());
    } else if (loop_type_ == DoubleBufferLoopStage::Epilog) {
      NVF_ERROR(requireEpilogue(double_buffer_load_exprs_));
      start = IrBuilder::subExpr(
          double_buffer_loop_->stop(), GpuLower::current()->kernel()->oneVal());
    }

    cloned_top_level_loop_ = IrBuilder::create<kir::ForLoop>(
        double_buffer_loop_->iter_domain(),
        index,
        start,
        stop,
        GpuLower::current()->kernel()->oneVal(),
        false,
        nullptr,
        double_buffer_loop_->isUnrollRequired(),
        loop_type_);

    handle(double_buffer_loop_);
  }

  void handle(kir::ForLoop* fl) override {
    kir::ForLoop* cloned_loop = fl == double_buffer_loop_
        ? cloned_top_level_loop_
        : IrBuilder::create<kir::ForLoop>(fl);

    cloned_scopes_.push_back(&cloned_loop->body());

    kir::IrVisitor::handle(fl);

    cloned_scopes_.pop_back();

    // Add the cloned loop into the parent loop body only when the
    // cloned loop contains expressions.
    if (!cloned_loop->body().empty() && !cloned_scopes_.empty()) {
      cloned_scopes_.back()->push_back(cloned_loop);
    }
  }

  void handle(kir::IfThenElse* ite) final {
    NVF_ERROR(false, "No IfThenElse should exist yet");
  }

  void dispatch(Expr* expr) override {
    if (exclude_.count(expr) > 0) {
      return;
    }

    if (expr->isA<kir::ForLoop>() || expr->isA<kir::IfThenElse>()) {
      kir::IrVisitor::dispatch(expr);
      return;
    }

    NVF_ERROR(!cloned_scopes_.empty());

    if (loop_type_ == DoubleBufferLoopStage::Main) {
      cloned_scopes_.back()->push_back(expr);
      return;
    }

    // In Prologue and Epilogue, either load expressions or anything
    // else are copied. Note that there can be multiple exprs defining
    // double buffered TVs (e.g., buffer initialization).

    auto out_tv = ir_utils::getTvOutput(expr);
    const auto is_double_buffer_load_expr = std::any_of(
        double_buffer_load_exprs_.begin(),
        double_buffer_load_exprs_.end(),
        [out_tv](const auto load_expr) {
          auto double_buffer_tv = ir_utils::getTvOutput(load_expr);
          NVF_ERROR(double_buffer_tv != nullptr);
          return out_tv == double_buffer_tv;
        });
    if ((loop_type_ == DoubleBufferLoopStage::Prolog &&
         is_double_buffer_load_expr) ||
        (loop_type_ == DoubleBufferLoopStage::Epilog &&
         !is_double_buffer_load_expr)) {
      cloned_scopes_.back()->push_back(expr);
    }
  }

 protected:
  kir::ForLoop* double_buffer_loop_ = nullptr;
  const std::vector<Expr*>& double_buffer_load_exprs_;
  const DoubleBufferLoopStage loop_type_;

  kir::ForLoop* cloned_top_level_loop_ = nullptr;
  std::deque<kir::Scope*> cloned_scopes_;
  const std::unordered_set<Expr*>& exclude_;
};

// Replicates double buffer loops for Prologue, Main, and
// Epilogue. Prologue only copies the load expressions of double
// buffered tensors, whereas Epilogue does any expression other than
// the loads. Main copies everything.
//
// Loop Structure:
// pre-prolog:
// - smem allocations (mbarriers, tokens)
// - mbarrier init for all stages
//
// prolog loop:
// - 0th thread:
//   - issue cp async bulks for all but last stages
//
// main loop:
// - select a single thread:
//   - issue next cp async bulk
// - copy body, without
//   - smem allocations
//   - mbarrier inits
//   - mbarrier inval
//
// epilogue loop:
// - copy body, without
//   - smem allocations
//   - issuing cp async
//   - mbarrier inits
//   - mbarrier inval
//
// post-epilogue:
//  - 0th thread: loop with mbarriers inval
class TmaDoubleBufferLoopCloner : public DoubleBufferLoopCloner {
 public:
  static kir::ForLoop* clone(
      kir::ForLoop* double_buffer_loop,
      const std::vector<Expr*>& double_buffer_load_exprs,
      DoubleBufferLoopStage loop_type,
      const std::unordered_set<Expr*>& exclude = {}) {
    TmaDoubleBufferLoopCloner cloner(
        double_buffer_loop, double_buffer_load_exprs, loop_type, exclude);
    cloner.duplicate();
    return cloner.cloned_top_level_loop_;
  }
  ~TmaDoubleBufferLoopCloner() override = default;

 private:
  TmaDoubleBufferLoopCloner(
      kir::ForLoop* double_buffer_loop,
      const std::vector<Expr*>& double_buffer_load_exprs,
      DoubleBufferLoopStage loop_type,
      const std::unordered_set<Expr*>& exclude)
      : DoubleBufferLoopCloner(
            double_buffer_loop,
            double_buffer_load_exprs,
            loop_type,
            exclude) {}

  void handle(kir::ForLoop* fl) final {
    kir::ForLoop* cloned_loop = fl == double_buffer_loop_
        ? cloned_top_level_loop_
        : IrBuilder::create<kir::ForLoop>(fl);

    cloned_scopes_.push_back(&cloned_loop->body());

    kir::IrVisitor::handle(fl);

    cloned_scopes_.pop_back();

    // Skip if cloned loop is empty or there is not an active for-loop
    // structure
    if (cloned_loop->body().empty() || cloned_scopes_.empty()) {
      return;
    }

    // Add cloned for_loop when mbarrier_arrive_tx_ is not active or
    // we are within a nested for-loop structure
    if (mbarrier_arrive_tx_ == nullptr || cloned_scopes_.size() > 1) {
      cloned_scopes_.back()->push_back(cloned_loop);
    } else {
      // mbarrier::arriveExpectTx and TMA operations occur in prologue and main
      // loops.
      //
      // Pseudo-code example:
      // if (elect_sync()) {
      //   mbarrier_tokens[stage] = mbarrier::arriveExpectTx(mbarriers[stage],
      //                                                     expected_tx);
      //   for (...) {
      //     Hopper::cpAsyncBulkTensorTileG2S(
      //       Hopper::CpAsyncBulkTensorTileG2SIndex<num_dims>{
      //         tma_descriptor, global_index, mbarrier[stage] },
      //       shared_index(stage, num_stages));
      //   }
      // }
      NVF_ERROR(cloned_scopes_.front() == &cloned_top_level_loop_->body());
      kir::IfThenElse* if_expr = createThreadPredicatedIfThenElse();
      kir::Scope& body = if_expr->thenBody();
      body.push_back(mbarrier_arrive_tx_);
      body.push_back(cloned_loop);
      cloned_scopes_.back()->push_back(if_expr);
      mbarrier_arrive_tx_ = nullptr;
    }

    // mbarrier::wait occur in main and epilogue loops.
    //
    // Pseudo-code example:
    //  mbarrier::wait(mbarriers[stage], mbarrier_tokens[stage]);
    if (mbarrier_wait_ != nullptr && cloned_scopes_.size() == 1) {
      NVF_ERROR(cloned_scopes_.front() == &cloned_top_level_loop_->body());
      cloned_top_level_loop_->body().push_back(mbarrier_wait_);
      mbarrier_wait_ = nullptr;
    }
  }

  void dispatch(Expr* expr) final {
    if (exclude_.count(expr) > 0) {
      return;
    }

    if (expr->isA<kir::ForLoop>() || expr->isA<kir::IfThenElse>()) {
      kir::IrVisitor::dispatch(expr);
      return;
    }

    NVF_ERROR(!cloned_scopes_.empty());

    TensorView* out_tv = ir_utils::getTvOutput(expr);

    bool is_double_buffer_load_expr = std::any_of(
        double_buffer_load_exprs_.begin(),
        double_buffer_load_exprs_.end(),
        [out_tv](Expr* load_expr) {
          TensorView* double_buffer_tv = ir_utils::getTvOutput(load_expr);
          NVF_ERROR(double_buffer_tv != nullptr);
          return out_tv == double_buffer_tv;
        });

    // This expr is a part of cpAsyncBulk synchronization process. It was
    // added earlier to satisfy checks in other passes. It was already handled
    // already, so it will not be pushed to the new scope. cpAsyncBulk exprs
    // that are not a part of double buffering, will be added to a new scope.
    bool mbarrier_token_exists =
        GpuLower::current()->ldstMBarrierTokenMap().count(expr) != 0;

    bool is_ignorable_tma_smem_alloc =
        (GpuLower::current()->mBarrierTokenSmemAllocSet().count(expr) != 0);

    bool is_ignorable_mbarrier_init =
        (expr->isA<kir::MBarrierInit>() && mbarrier_token_exists);

    bool is_ignorable_mbarrier_inval =
        (expr->isA<kir::MBarrierInvalidate>() && mbarrier_token_exists);

    int64_t stage_depth =
        GpuLower::current()->doubleBufferInfo().getStageDepthFor(
            double_buffer_loop_->iter_domain());

    switch (loop_type_) {
      case DoubleBufferLoopStage::Prolog: {
        // Skip if not LoadStoreOp expression
        if (!expr->isA<LoadStoreOp>()) {
          break;
        }

        // Skip expr if it is not circular buffer expression
        if (!is_double_buffer_load_expr) {
          break;
        }

        // NOTE: There can be circular buffered TVs without TMA load exprs.
        if (!mbarrier_token_exists) {
          cloned_scopes_.back()->push_back(expr);
          break;
        }

        // Handle cpAsyncBulk type LoadStoreOp that is registered with token
        // smem TVs as it requires synchronization
        //
        // See AllocationInserter for details when and how token map is filled
        // with data
        //
        // Replace cpAsyncBulk type LoadStoreOp with:
        //  if (elect_sync()) {
        //    for (int64_t loop_idx : irange(stages-1)) {
        //      token[loop_idx] =
        //        mbarrier::arriveExpectTx(mbarrier[loop_idx])
        //      cpAsyncBulk(mbarrier[loop_idx],...)
        //    }
        //  }

        LoadStoreOp* ldst = expr->as<LoadStoreOp>();

        NVF_ERROR(mbarrier_arrive_tx_ == nullptr);
        mbarrier_arrive_tx_ = createMbarrierArriveExpectTx(
            ldst, cloned_top_level_loop_->indexOrStartIfTrivial());

        // Clone LoadStoreOp and map it to mbarrier alloc
        Expr* new_ldst =
            IrBuilder::create<LoadStoreOp>(
                ldst->opType(), ldst->out(), ldst->in(), ldst->cacheOp())
                ->withPredicate(ldst->predicate());

        // Register mbarrier object to be used with new LoadStoreOp
        // from prolog loop
        GpuLower::current()->ldstMBarrierIndexMap().emplace(
            new_ldst, mbarrier_arrive_tx_->mbarrier());

        // If last cloned scope is the cloned_top_level_loop body, then
        // add mbarrier::arriveExpectTx and new loadStoreOp. Otherwise, we are
        // in a nested for-loop and should wait until we return to top-level
        // for loop.
        if (cloned_scopes_.size() == 1) {
          kir::IfThenElse* if_expr = createThreadPredicatedIfThenElse();
          kir::Scope& body = if_expr->thenBody();
          body.push_back(mbarrier_arrive_tx_);
          body.push_back(new_ldst);
          cloned_scopes_.back()->push_back(if_expr);
          mbarrier_arrive_tx_ = nullptr;
        } else {
          cloned_scopes_.back()->push_back(new_ldst);
        }
        break;
      }
      case DoubleBufferLoopStage::Main: {
        // Handle cpAsyncBulk type LoadStoreOp that is registered with
        //  token smem TVs as it requires synchronization
        if (expr->isA<LoadStoreOp>() && mbarrier_token_exists) {
          // cpAsyncBulk for double-buffered tensor has assigned a placeholder
          // for token objects

          //! Before waiting at the mbarrier for the current stage, we
          //! launch the load operation for the next available stage. The
          //! last buffer in the pipeline is the first available after the
          //! prologue loop launches the initial wave of tma loads.
          //!
          //! current_compute_stage = for_loop_index % stage_depth
          //! current_load_stage = (for_loop_index + (stage_depth - 1)) %
          //! stage_depth)
          if (current_compute_stage_ == nullptr) {
            current_compute_stage_ = IrBuilder::modExpr(
                double_buffer_loop_->index(),
                IrBuilder::create<Val>(stage_depth, PrimDataType::Index));
            kir::Allocate* current_compute_stage_alloc =
                IrBuilder::create<kir::Allocate>(
                    current_compute_stage_,
                    MemoryType::Local,
                    IrBuilder::create<Val>(1L, PrimDataType::Index),
                    /*zero_init=*/false);
            cloned_top_level_loop_->body().push_back(
                current_compute_stage_alloc);
            cloned_top_level_loop_->body().push_back(
                current_compute_stage_->definition());
          }

          if (current_load_stage_ == nullptr) {
            current_load_stage_ = IrBuilder::modExpr(
                IrBuilder::addExpr(
                    double_buffer_loop_->index(),
                    IrBuilder::subExpr(
                        IrBuilder::create<Val>(
                            stage_depth, PrimDataType::Index),
                        IrBuilder::create<Val>(1L, PrimDataType::Index))),
                IrBuilder::create<Val>(stage_depth, PrimDataType::Index));
            kir::Allocate* current_load_stage_alloc =
                IrBuilder::create<kir::Allocate>(
                    current_load_stage_,
                    MemoryType::Local,
                    IrBuilder::create<Val>(1L, PrimDataType::Index),
                    /*zero_init=*/false);
            cloned_top_level_loop_->body().push_back(current_load_stage_alloc);
            cloned_top_level_loop_->body().push_back(
                current_load_stage_->definition());
          }

          // Replace LoadStoreOp with:
          //  if (0th thread in block)
          //    token[next_stage] =
          //    mbarrier::arriveExpectTx(mbarrier[next_stage])
          //    cpAsyncBulk(mbarrier[next_stage],...)
          //  mbarrier::wait(token[curr_stage])
          //
          // Where mbarrier and token are smem arrays bound to the LoadStoreOp
          //
          LoadStoreOp* ldst = expr->as<LoadStoreOp>();

          NVF_ERROR(mbarrier_arrive_tx_ == nullptr);
          mbarrier_arrive_tx_ =
              createMbarrierArriveExpectTx(ldst, current_load_stage_);
          // Register mbarrier object to be used with LoadStoreOp
          //  from main loop
          GpuLower::current()->ldstMBarrierIndexMap().emplace(
              ldst, mbarrier_arrive_tx_->mbarrier());

          if (cloned_scopes_.size() == 1) {
            kir::IfThenElse* if_expr = createThreadPredicatedIfThenElse();
            kir::Scope& body = if_expr->thenBody();
            body.push_back(mbarrier_arrive_tx_);
            body.push_back(ldst);
            cloned_scopes_.back()->push_back(if_expr);
            mbarrier_arrive_tx_ = nullptr;
          } else {
            cloned_scopes_.back()->push_back(ldst);
          }

          // Construct mBarrier::wait for current stage
          NVF_ERROR(
              mbarrier_wait_ == nullptr,
              "Expected mbarrier_wait to inactive for current TMA operation");
          mbarrier_wait_ = createMbarrierWait(ldst, current_compute_stage_);
          break;
        }
        if (!(is_ignorable_tma_smem_alloc || is_ignorable_mbarrier_init ||
              is_ignorable_mbarrier_inval)) {
          cloned_scopes_.back()->push_back(expr);
        }
        break;
      }
      case DoubleBufferLoopStage::Epilog: {
        if (expr->isA<LoadStoreOp>() && mbarrier_token_exists) {
          // Construct mBarrier::wait for last stage
          LoadStoreOp* ldst = expr->as<LoadStoreOp>();
          Val* last_index = IrBuilder::subExpr(
              double_buffer_loop_->stop(),
              GpuLower::current()->kernel()->oneVal());
          Val* last_compute_stage = IrBuilder::modExpr(
              last_index,
              IrBuilder::create<Val>(stage_depth, PrimDataType::Index));
          kir::MBarrierWait* mbarrier_wait =
              createMbarrierWait(ldst, last_compute_stage);
          cloned_scopes_.back()->push_back(mbarrier_wait);
          break;
        }
        if (!(is_ignorable_tma_smem_alloc || is_ignorable_mbarrier_init ||
              is_ignorable_mbarrier_inval || is_double_buffer_load_expr)) {
          cloned_scopes_.back()->push_back(expr);
        }
        break;
      }
      case DoubleBufferLoopStage::NotApplicable: {
        NVF_ERROR(false, "Unsupported loop mode, got: ", loop_type_);
      }
    }
  }

 private:
  // Mbarrier_Wait to add to cloned_top_level_loop
  kir::MBarrierWait* mbarrier_wait_ = nullptr;

  // Mbarrier_ArriveExpectTx to add to cloned_top_level_loop
  kir::MBarrierArriveExpectTx* mbarrier_arrive_tx_ = nullptr;

  // Current stage, expectation:
  //  curr_stages_idx = (double_buffer_loop_idx % stages)
  Val* current_compute_stage_ = nullptr;

  // Next stage, expectation:
  //  next_stages_idx = (double_buffer_loop_idx + (stages -1)) % stages
  Val* current_load_stage_ = nullptr;
};

using InsertionInfo = std::unordered_map<kir::ForLoop*, std::vector<Expr*>>;

class IsDoubleBufferLoadLoop : public kir::IrVisitor {
 public:
  static bool check(
      Expr* expr,
      const std::vector<Expr*>& double_buffer_load_exprs) {
    IsDoubleBufferLoadLoop checker(double_buffer_load_exprs);
    return checker.check(expr);
  }

 private:
  IsDoubleBufferLoadLoop(const std::vector<Expr*>& double_buffer_load_exprs)
      : double_buffer_load_exprs_(double_buffer_load_exprs) {}

  using kir::IrVisitor::handle;

  bool check(Expr* expr) {
    dispatch(expr);
    return result_;
  }

  void dispatch(Expr* expr) final {
    if (result_) {
      return;
    }
    if (std::find(
            double_buffer_load_exprs_.begin(),
            double_buffer_load_exprs_.end(),
            expr) != double_buffer_load_exprs_.end()) {
      result_ = true;
      return;
    }
    IrVisitor::dispatch(expr);
  }

 private:
  const std::vector<Expr*>& double_buffer_load_exprs_;
  bool result_ = false;
};

// Traverse lowered loop-nests and find all double buffer loops and
// associated load expressions.
class DoubleBufferLoopNestInspector : private kir::IrVisitor {
 public:
  static InsertionInfo run(const std::vector<Expr*>& exprs) {
    DoubleBufferLoopNestInspector inspector(exprs);
    return inspector.insertion_info_;
  }

 private:
  DoubleBufferLoopNestInspector(const std::vector<Expr*>& exprs) {
    handle(exprs);
  }

  using kir::IrVisitor::handle;

  // Collect double buffer related information on a expr
  //  that is a memory load, i.e. a LoadStore or a Set.
  void handlePossibleLoadExpr(Expr* expr) {
    auto out_tv = ir_utils::getTvOutput(expr);

    if (out_tv == nullptr) {
      return;
    }

    // Ignore init loop
    if (!(out_tv->isDoubleBuffered() || out_tv->isCircularBuffered()) ||
        !expr->input(0)->isA<TensorView>()) {
      return;
    }

    auto double_buffer_loop =
        GpuLower::current()->doubleBufferInfo().getDoubleBufferLoop(
            out_tv, for_loops_);

    NVF_ERROR(
        double_buffer_loop != nullptr,
        "No double buffer loop found for a double buffered tensor: ",
        out_tv->toString());

    validateDoubleBufferLoop(double_buffer_loop);

    insertion_info_[double_buffer_loop].push_back(expr);
  }

  void handle(UnaryOp* uop) final {
    handlePossibleLoadExpr(uop);
  }

  void handle(LoadStoreOp* ldst) final {
    handlePossibleLoadExpr(ldst);
  }

  static void validateDoubleBufferLoop(kir::ForLoop* loop) {
    NVF_ERROR(
        loop->start()->isZeroInt(), "Unsupported loop: ", loop->toString());
    NVF_ERROR(loop->step()->isOneInt(), "Unsupported loop: ", loop->toString());
    NVF_ERROR(
        !loop->vectorize(),
        "Vectorized loop should not be the allocation loop for double-buffered tensor: ",
        loop->toString());
    NVF_ERROR(
        !loop->vectorize_shift(),
        "Vectorize shift loop should not be the allocation loop for double-buffered tensor: ",
        loop->toString());
  }

  InsertionInfo insertion_info_;
};

// TODO Move to DoubleBufferLoopCloner
// Creates pre-prologue section necessary for proper handling async TMA memory
// operations. It moves the allocation of mbarriers and its tokens outside of
// the main loop
//
// Expected result:
//   Allocate mbarriers and tokens in shared memory
//   if (threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0) {
//     for (unsigned i = 0; i < stages; ++i) {
//       mbarrier::init(...);
//     }
//   }
class CpAsyncBulkPrePrologue : public kir::IrVisitor {
 public:
  static std::vector<Expr*> create(
      kir::ForLoop* double_buffer_loop,
      const std::vector<Expr*>& double_buffer_load_exprs) {
    CpAsyncBulkPrePrologue creator(
        double_buffer_loop, double_buffer_load_exprs);
    creator.create();
    return creator.pre_prologue_exprs_;
  }

 private:
  CpAsyncBulkPrePrologue(
      kir::ForLoop* double_buffer_loop,
      const std::vector<Expr*>& double_buffer_load_exprs)
      : double_buffer_loop_(double_buffer_loop),
        double_buffer_load_exprs_(double_buffer_load_exprs) {}

  using kir::IrVisitor::handle;

  void create() {
    // Find and add smem allocations for tokens and mbarrier objects
    handle(double_buffer_loop_);

    // Define how many threads should arrive at the barrier
    //  we expect 0th thread to handle init/arrive & transaction config
    //  while other threads will wait for it
    Val* one_val = IrBuilder::create<Val>(1L, PrimDataType::UInt32);

    // Construct predicate
    // 'threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0'
    kir::IfThenElse* if_expr = createThreadPredicatedIfThenElse();

    // Construct for loop, a body for if expression
    kir::ForLoop* loop = createStagesForLoop(double_buffer_loop_);

    // Construct loop body with:
    // - mBarriers' initializations for each element in smem array for
    //   each double buffered tensor
    // - expected arrival: number of threads in the block
    for (const Expr* ldst : double_buffer_load_exprs_) {
      if (GpuLower::current()->ldstMBarrierMap().count(ldst) != 0) {
        TensorView* all_mbarriers =
            GpuLower::current()->ldstMBarrierMap()[ldst];
        kir::TensorIndex* stage_mbarrier =
            IrBuilder::create<kir::TensorIndex>(all_mbarriers, loop->index());
        kir::MBarrierInit* mbarrier_init =
            IrBuilder::create<kir::MBarrierInit>(stage_mbarrier, one_val);
        loop->body().push_back(mbarrier_init);
      }
    }

    if_expr->thenBody().push_back(loop);

    pre_prologue_exprs_.push_back(if_expr);
  }

  void handle(kir::ForLoop* fl) final {
    kir::IrVisitor::handle(fl);
  }

  void handle(kir::IfThenElse* ite) final {
    NVF_ERROR(false, "No IfThenElse should exist yet");
  }

  void dispatch(Expr* expr) final {
    if (expr->isA<kir::ForLoop>() || expr->isA<kir::IfThenElse>()) {
      kir::IrVisitor::dispatch(expr);
      return;
    }

    if (auto alloc = dynamic_cast<kir::Allocate*>(expr)) {
      if (alloc->memoryType() == MemoryType::Shared) {
        if (GpuLower::current()->mBarrierTokenSmemAllocSet().count(alloc) !=
            0) {
          pre_prologue_exprs_.push_back(expr);
        }
      }
    }
  }

 private:
  kir::ForLoop* double_buffer_loop_ = nullptr;
  const std::vector<Expr*>& double_buffer_load_exprs_;

  std::vector<Expr*> pre_prologue_exprs_;
};

// TODO Move to DoubleBufferLoopCloner
// Creates post-epilogue section needed for releasing mbarriers after TMA memory
// operations.
//
// Expected result:
//   if (threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0) {
//     for (unsigned i = 0; i < stages; ++i) {
//       mbarrier::inval(...);
//     }
//   }
class CpAsyncBulkPostEpilogue {
 public:
  static std::vector<Expr*> create(
      kir::ForLoop* double_buffer_loop,
      const std::vector<Expr*>& double_buffer_load_exprs) {
    CpAsyncBulkPostEpilogue creator(
        double_buffer_loop, double_buffer_load_exprs);
    creator.create();
    return creator.post_prologue_exprs_;
  }

 private:
  CpAsyncBulkPostEpilogue(
      kir::ForLoop* double_buffer_loop,
      const std::vector<Expr*>& double_buffer_load_exprs)
      : double_buffer_loop_(double_buffer_loop),
        double_buffer_load_exprs_(double_buffer_load_exprs) {}

  void create() {
    // Construct predicate
    // 'threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0'
    kir::IfThenElse* if_expr = createThreadPredicatedIfThenElse();

    // Construct for loop, a body for if expressiong
    kir::ForLoop* loop = createStagesForLoop(double_buffer_loop_);

    // Construct loop body with:
    // - mBarriers' invalidation for each element in smem array for
    //   each double buffered tensor
    for (const Expr* ldst : double_buffer_load_exprs_) {
      if (GpuLower::current()->ldstMBarrierMap().count(ldst) != 0) {
        TensorView* all_mbarriers =
            GpuLower::current()->ldstMBarrierMap()[ldst];
        kir::TensorIndex* stage_mbarrier =
            IrBuilder::create<kir::TensorIndex>(all_mbarriers, loop->index());
        kir::MBarrierInvalidate* mbarrier_inval =
            IrBuilder::create<kir::MBarrierInvalidate>(stage_mbarrier);
        loop->body().push_back(mbarrier_inval);
      }
    }

    if_expr->thenBody().push_back(loop);

    post_prologue_exprs_.push_back(if_expr);
  }

 private:
  kir::ForLoop* double_buffer_loop_ = nullptr;
  const std::vector<Expr*>& double_buffer_load_exprs_;

  std::vector<Expr*> post_prologue_exprs_;
};

namespace {

void getAllocInTrivialLoop(
    kir::ForLoop* fl,
    std::unordered_set<Expr*>& output) {
  if (!fl->isTrivial()) {
    return;
  }
  for (auto expr : fl->body().exprs()) {
    if (expr->isA<kir::Allocate>()) {
      output.emplace(expr);
    } else if (auto loop = dynamic_cast<kir::ForLoop*>(expr)) {
      getAllocInTrivialLoop(loop, output);
    }
  }
}

} // namespace

// Apply double buffering transformations
class DoubleBufferInserter : private kir::ExprMutator {
 public:
  // When there exist multiple double buffer loops, apply
  // transformations to inner-most loops first. A single ExprMutator
  // pass can only process one loop.
  static std::vector<Expr*> run(
      const std::vector<Expr*>& exprs,
      InsertionInfo insertion_info) {
    auto inserted_exprs = exprs;
    while (!insertion_info.empty()) {
      DoubleBufferInserter inserter(inserted_exprs, insertion_info);
      inserted_exprs = inserter.exprs_;
    }
    return inserted_exprs;
  }

 private:
  DoubleBufferInserter(
      const std::vector<Expr*>& exprs,
      InsertionInfo& insertion_info)
      : insertion_info_(insertion_info) {
    auto num_double_buffer_loops = insertion_info.size();
    traverseAndInsert(exprs);
    NVF_ERROR(processed_loop_ != nullptr);
    NVF_ERROR(insertion_info.size() == num_double_buffer_loops - 1);
  }

  using kir::ExprMutator::handle;

  void handle(kir::ForLoop* loop) final {
    kir::ExprMutator::handle(loop);

    // If another loop is already taken care of, no more loop should
    // be done in the same pass
    if (processed_loop_ != nullptr) {
      return;
    }

    auto it = insertion_info_.find(loop);
    if (it == insertion_info_.end()) {
      return;
    }

    auto hasCpAsyncBulk = std::any_of(
        it->second.begin(), it->second.end(), ir_utils::isCpAsyncBulk);

    if (hasCpAsyncBulk) {
      insertTma(loop, it->second);
    } else {
      insert(loop, it->second);
    }
    processed_loop_ = loop;
    insertion_info_.erase(loop);
  }

  void insertTma(
      kir::ForLoop* double_buffer_loop,
      const std::vector<Expr*>& loads) {
    // cpAsyncBulk (with TMA) requires some operations to be done prior to
    //  prologue loop, for example:
    // - smem allocation
    // - initialization of mbarrier objects
    std::vector<Expr*> pre_prologue_exprs =
        CpAsyncBulkPrePrologue::create(double_buffer_loop, loads);
    if (!pre_prologue_exprs.empty()) {
      for (Expr* expr : pre_prologue_exprs) {
        registerInsertBefore(double_buffer_loop, expr);
      }
    }

    // cpAsyncBulk (with TMA) block sync prior to entering main loop to
    //  make smem with mbarrier objects is initialized.
    kir::BlockSync* sync = IrBuilder::create<kir::BlockSync>(false);
    registerInsertBefore(double_buffer_loop, sync);

    kir::ForLoop* prologue_loop = TmaDoubleBufferLoopCloner::clone(
        double_buffer_loop, loads, DoubleBufferLoopStage::Prolog);
    registerInsertBefore(double_buffer_loop, prologue_loop);

    kir::ForLoop* main_loop = TmaDoubleBufferLoopCloner::clone(
        double_buffer_loop, loads, DoubleBufferLoopStage::Main);
    registerReplace(double_buffer_loop, main_loop);

    kir::ForLoop* last_for_loop = double_buffer_loop;

    if (requireEpilogue(loads)) {
      // In the case where the main loop is trivial (for example, ldmatrix in
      // matmul kernel), we need to be careful when copying epilog loop. For
      // example, if the main loop is:
      //   for (int i = 0; i < 1; ++i) {
      //     ...
      //     float T1[2];
      //     T1 = ...
      //     ...
      //   }
      // Because trivial loop is not generated, the allocation of T1 will be one
      // level above in the generated scope. So when we copy epilog, we need to
      // make sure we don't copy these allocation so that there is no duplicate
      // allocation.
      std::unordered_set<Expr*> alloc_in_main;
      getAllocInTrivialLoop(main_loop, alloc_in_main);
      kir::ForLoop* epilogue_loop = TmaDoubleBufferLoopCloner::clone(
          double_buffer_loop,
          loads,
          DoubleBufferLoopStage::Epilog,
          alloc_in_main);
      registerInsertAfter(double_buffer_loop, epilogue_loop);
      last_for_loop = epilogue_loop;
    }

    std::vector<Expr*> post_epilogue_exprs =
        CpAsyncBulkPostEpilogue::create(double_buffer_loop, loads);
    if (!post_epilogue_exprs.empty()) {
      for (Expr* expr : post_epilogue_exprs) {
        registerInsertAfter(last_for_loop, expr);
      }
    }
  }

  void insert(
      kir::ForLoop* double_buffer_loop,
      const std::vector<Expr*>& loads) {
    auto prologue_loop = DoubleBufferLoopCloner::clone(
        double_buffer_loop, loads, DoubleBufferLoopStage::Prolog);
    registerInsertBefore(double_buffer_loop, prologue_loop);

    auto write_to_smem =
        std::any_of(loads.begin(), loads.end(), [](const Expr* expr) {
          return expr->output(0)->as<TensorView>()->getMemoryType() ==
              MemoryType::Shared;
        });

    // RAW sync is not inserted for double buffered tensors. The only
    // exception is the prologue load.
    bool has_cpasync = false;
    if (write_to_smem) {
      // Here the initial sync before entering double buffer loop is
      //  inserted.

      // If any of the double buffered tensor in this double buffer
      //  loop is async copy. We want to wait for the gmem loads to
      //  finish before synchronizing the block.
      if (std::any_of(loads.begin(), loads.end(), ir_utils::isCpAsyncOp)) {
        auto stage_depth =
            GpuLower::current()->doubleBufferInfo().getStageDepthFor(
                double_buffer_loop->iter_domain());
        auto cp_async_wait = IrBuilder::create<kir::AsyncWait>(
            AsyncOpType::CpAsync, stage_depth - 2);
        prologue_loop->body().push_back(
            IrBuilder::create<kir::AsyncCommit>(AsyncOpType::CpAsync));
        registerInsertBefore(double_buffer_loop, cp_async_wait);
        has_cpasync = true;
      }

      // Insert the initial block sync before entering main loop.
      if (std::any_of(loads.begin(), loads.end(), [](Expr* expr) {
            return GpuLower::current()
                ->syncMap()
                ->needsRawSync(ir_utils::getTvOutput(expr))
                .hasTID();
          })) {
        // If any of the double buffered loads require sync, as indicated
        //  by sync info map, insert the sync before entering the double buffer
        //  loop.
        // TODO:
        //  Currently not supporting double buffer in gmem, but short to mid
        //  term not yet a priority to go for this case.
        auto sync = IrBuilder::create<kir::BlockSync>(false);
        registerInsertBefore(double_buffer_loop, sync);
      }
    }

    auto main_loop = DoubleBufferLoopCloner::clone(
        double_buffer_loop, loads, DoubleBufferLoopStage::Main);

    registerReplace(double_buffer_loop, main_loop);

    // Insert the wait instruction in this pass instead
    //  of relying on WAR sync pass to do it.
    // The WAR sync pass today would insert the wait function
    //  exactly where we need it but the purpose of this wait
    //  insertion isn't exactly WAR protection.
    //
    // TODO: [Double Buffer Sync]
    //  We might eventually want to move the block sync inserted
    //   by WAR pass here as well since this sync insertion is kind
    //   of both WAR and RAW (or neither RAW nor WAR, depends
    //   on how we look at it).
    // Eg. in the case when a intermediate
    //   tensor is double buffered.
    //
    //  __block_sync();    // This is the initial sync
    //  For i in ...       // Double buffer loop
    //     A[i%2] = ...;
    //     ...  = A[1-i%2];
    //     __block_sync();  // sync within loop
    //     ...
    //  The "sync within loop" can be placed anywhere in the
    //   double buffer loop while in the case of RAW and WAR
    //   there'd be extra insertion point restrictions.
    //  We are currently not actively exploring opportunities
    //   with this property of "double buffer sync" so this
    //   is more conceptual at the moment, aka low priority.
    if (has_cpasync) {
      insertCpAsyncCommitWaitInMainLoop(main_loop, loads);

      // The main loop will generate some async loads from invalid regions.
      // These populate the current cp.async group and they fill the smem with
      // zero. Subsequent code might assume an empty cp.async group (for example
      // an unparallelized batch matmul), or might re-use memory (WAW
      // hazard, see https://github.com/NVIDIA/Fuser/issues/2000). For safety,
      // we drain the group after the loops by waiting on these transfers.
      auto cp_async_wait_all =
          IrBuilder::create<kir::AsyncWait>(AsyncOpType::CpAsync, 0);
      registerInsertAfter(double_buffer_loop, cp_async_wait_all);
    }

    if (requireEpilogue(loads)) {
      // In the case where the main loop is trivial (for example, ldmatrix in
      // matmul kernel), we need to be careful when copying epilog loop. For
      // example, if the main loop is:
      //   for (int i = 0; i < 1; ++i) {
      //     ...
      //     float T1[2];
      //     T1 = ...
      //     ...
      //   }
      // Because trivial loop is not generated, the allocation of T1 will be one
      // level above in the generated scope. So when we copy epilog, we need to
      // make sure we don't copy these allocation so that there is no duplicate
      // allocation.
      std::unordered_set<Expr*> alloc_in_main;
      getAllocInTrivialLoop(main_loop, alloc_in_main);
      auto epilogue_loop = DoubleBufferLoopCloner::clone(
          double_buffer_loop,
          loads,
          DoubleBufferLoopStage::Epilog,
          alloc_in_main);
      registerInsertAfter(double_buffer_loop, epilogue_loop);
    }
  }

  // Simple conservative rule for inserting async copy wait
  //  primitive in the double buffer loop:
  void insertCpAsyncCommitWaitInMainLoop(
      kir::ForLoop* main_loop,
      const std::vector<Expr*>& loads) {
    NVF_ERROR(
        !main_loop->body().empty(),
        "Double buffer sync insertion: empty main loop.");
    auto& exprs = main_loop->body().exprs();
    // Note: This pass explicitly assumes that WAR sync has been
    //  inserted so would need to be updated if we re-order the
    //  passes. Cleanups suggested in [Double Buffer Sync]
    //  would resolve this dependency on pass ordering.
    auto stage_depth = GpuLower::current()->doubleBufferInfo().getStageDepthFor(
        main_loop->iter_domain());
    auto cp_async_commit =
        IrBuilder::create<kir::AsyncCommit>(AsyncOpType::CpAsync);
    auto cp_async_wait = IrBuilder::create<kir::AsyncWait>(
        AsyncOpType::CpAsync, stage_depth - 2);

    // Find the last double buffer load in the main loop, and insert
    // cp.async.commit after it.
    std::vector<Expr*>::const_iterator last_double_buffer_load = exprs.end();
    for (auto it = exprs.begin(); it != exprs.end(); ++it) {
      if (IsDoubleBufferLoadLoop::check(*it, loads)) {
        last_double_buffer_load = it;
      }
    }
    NVF_ERROR(last_double_buffer_load != exprs.end());
    std::vector<Expr*>::const_iterator commit_it =
        main_loop->body().insert(last_double_buffer_load + 1, cp_async_commit);

    // Check if a sync has been inserted by WAR sync pass.
    auto rend = std::make_reverse_iterator(commit_it);
    auto block_sync_it =
        std::find_if(exprs.rbegin(), rend, [](const Expr* expr) {
          return expr->isA<kir::BlockSync>();
        });
    if (block_sync_it == rend) {
      // If there's no sync, i.e. no tensor needs cross thread communication. We
      // still need a wait but it can just be anywhere after the cp.async.commit
      // in the loop. Chose to place at the end arbitrarily.
      main_loop->body().insert_after(exprs.back(), cp_async_wait);
    } else {
      // If a sync has been inserted, wait needs to be placed before the sync.
      main_loop->body().insert_before(*block_sync_it, cp_async_wait);
    }
  }

 private:
  InsertionInfo& insertion_info_;
  kir::ForLoop* processed_loop_ = nullptr;
};

} // namespace

void DoubleBufferInfo::build(Fusion* fusion) {
  DoubleBufferFusionInspector inspector(fusion, *this);

  // Build double buffered loop id's
  for (auto& info : map_) {
    auto double_buffer_axis = info.second.double_buffer_axis;
    // Keeps track of which loop disjoint set has been
    //  double buffered. In index allocation, one index
    //  variable would need to be allocated in each
    //  double buffer stage.
    concrete_double_buffered_loop_id_.insert(
        GpuLower::current()->caMap()->getConcreteMappedID(
            double_buffer_axis, IdMappingMode::LOOP));
  }
}

bool DoubleBufferInfo::isDoubleBufferedIterDomain(IterDomain* id) {
  auto concrete_loop_id = GpuLower::current()->caMap()->getConcreteMappedID(
      id, IdMappingMode::LOOP);
  return concrete_double_buffered_loop_id_.count(concrete_loop_id);
}

DoubleBufferInfo::TvInfo& DoubleBufferInfo::getTvInfo(const TensorView* tv) {
  NVF_ERROR(
      tv->isDoubleBuffered() || tv->isCircularBuffered(),
      "Not a double-buffered tensor: ",
      tv->toString());
  return map_[tv];
}

void DoubleBufferInfo::setDoubleBufferAxis(
    const TensorView* tv,
    IterDomain* axis) {
  getTvInfo(tv).double_buffer_axis = axis;

  // Also validate the stage consistency with CA map.
  unsigned int stage_depth = 0;
  if (tv->isCircularBuffered()) {
    stage_depth = tv->circularBufferDepth();
  } else {
    // Double buffer is essentially
    //  circular buffer with depth 2.
    stage_depth = 2;
  }

  // Set and validate the new stage depth.
  setStageDepth(axis, stage_depth);
}

void DoubleBufferInfo::setStageDepth(IterDomain* id, unsigned int stage_depth) {
  auto concrete_loop_id = GpuLower::current()->caMap()->getConcreteMappedID(
      id, IdMappingMode::LOOP);

  auto maybe_exisiting_depth_it = stage_depth_.find(concrete_loop_id);
  if (maybe_exisiting_depth_it == stage_depth_.end()) {
    stage_depth_[concrete_loop_id] = stage_depth;
  } else {
    NVF_ERROR(
        stage_depth == maybe_exisiting_depth_it->second,
        "Unsupported multiple depth pipelining, was set to ",
        maybe_exisiting_depth_it->second,
        " by ",
        maybe_exisiting_depth_it->first->toString(),
        " and then set to ",
        stage_depth,
        " by ",
        concrete_loop_id->toString());
  }
}

IterDomain* DoubleBufferInfo::getDoubleBufferAxis(const TensorView* tv) {
  if (!(tv->isDoubleBuffered() || tv->isCircularBuffered())) {
    return nullptr;
  }

  return getTvInfo(tv).double_buffer_axis;
}

int64_t DoubleBufferInfo::getStageDepthFor(IterDomain* double_buffer_axis) {
  IterDomain* concrete_id = GpuLower::current()->caMap()->getConcreteMappedID(
      double_buffer_axis, IdMappingMode::LOOP);

  auto maybe_depth_it = stage_depth_.find(concrete_id);

  NVF_ERROR(maybe_depth_it != stage_depth_.end(), "Stage depth not found");

  return maybe_depth_it->second;
}

kir::ForLoop* DoubleBufferInfo::getDoubleBufferLoop(
    IterDomain* axis,
    const std::vector<kir::ForLoop*>& loops,
    bool ignore_prologue) {
  auto loop_it = std::find_if(loops.begin(), loops.end(), [&](const auto loop) {
    return GpuLower::current()->caMap()->areMapped(
               loop->iter_domain(), axis, IdMappingMode::EXACT) &&
        (!ignore_prologue ||
         loop->doubleBufferLoopStage() != DoubleBufferLoopStage::Prolog);
  });

  if (loop_it != loops.end()) {
    return *loop_it;
  } else {
    return nullptr;
  }
}

kir::ForLoop* DoubleBufferInfo::getDoubleBufferLoop(
    const TensorView* tv,
    const std::vector<kir::ForLoop*>& loops,
    bool ignore_prologue) {
  auto axis = getDoubleBufferAxis(tv);

  if (axis == nullptr) {
    return nullptr;
  }

  return getDoubleBufferLoop(axis, loops, ignore_prologue);
}

void DoubleBufferInfo::setOriginalAllocSize(
    const TensorView* tv,
    Val* original_alloc_size) {
  getTvInfo(tv).original_alloc_size = original_alloc_size;
}

Val* DoubleBufferInfo::getOriginalAllocSize(const TensorView* tv) {
  if (!(tv->isDoubleBuffered() || tv->isCircularBuffered())) {
    return nullptr;
  }

  return getTvInfo(tv).original_alloc_size;
}

std::vector<Expr*> DoubleBufferPass::run(const std::vector<Expr*>& exprs) {
  auto insertion_info = DoubleBufferLoopNestInspector::run(exprs);
  return DoubleBufferInserter::run(exprs, insertion_info);
}

} // namespace nvfuser
