// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#include <device_lower/analysis/index_compute.h>
#include <device_lower/lower2device.h>
#include <device_lower/utils.h>
#include <id_model/schedule.h>
#include <index_compute.h>
#include <ir/iostream.h>
#include <ir/utils.h>
#include <ops/arith.h>
#include <options.h>
#include <predicate_compute.h>
#include <transform_iter.h>
#include <transform_replay.h>
#include <val_graph_visitor.h>

#include <device_lower/pass/index.h>

namespace nvfuser {

std::vector<Expr*> IndexLowering::getIndexedExprs(
    std::vector<Expr*> incoming_exprs) {
  FUSER_PERF_SCOPE("GpuLower::Lower::IndexLowering::getIndexedExprs");
  // Traverse the exprs and setup allocation domains before
  // generating indices.
  if (GpuLower::current()->isTensorIndexerEnabled()) {
    GpuLower::current()->tensorIndexer().setupAllocationDomains(incoming_exprs);
  }
  IndexLowering il;
  il.generate(incoming_exprs);
  return il.lowered_exprs_;
}

Val* IndexLowering::lowerSrcIndex(
    Val* src,
    Val* dst,
    const std::unordered_map<IterDomain*, Val*>& override_index,
    bool generate_pointer,
    DataType as_type) const {
  if (auto tv = dynamic_cast<TensorView*>(src)) {
    NVF_ERROR(dst->isA<TensorView>());
    return Index::getProducerIndex(
        tv,
        dst->as<TensorView>(),
        for_loops_,
        getRotatedLoop(),
        override_index,
        generate_pointer,
        as_type);
  } else {
    return src;
  }
}

Val* IndexLowering::lowerDstIndex(
    Val* dst,
    const std::unordered_map<int, Val*>& override_index,
    bool generate_pointer,
    DataType as_type) const {
  if (auto tv = dynamic_cast<TensorView*>(dst)) {
    return Index::getConsumerIndex(
        tv,
        for_loops_,
        getRotatedLoop(),
        override_index,
        generate_pointer,
        as_type);
  } else {
    return dst;
  }
}

void IndexLowering::pushBack(Expr* expr) {
  if (active_scope_ == nullptr) {
    lowered_exprs_.push_back(expr);
  } else {
    active_scope_->push_back(expr);
  }
}

Expr* IndexLowering::back() const {
  if (active_scope_ == nullptr) {
    NVF_ERROR(!lowered_exprs_.empty(), "IndexLowering::back: empty scope.");
    return lowered_exprs_.back();
  }
  NVF_ERROR(!active_scope_->empty(), "IndexLowering::back: empty scope.");
  return active_scope_->exprs().back();
}

void IndexLowering::insertAtTopLevel(Expr* expr) {
  NVF_ERROR(!lowered_exprs_.empty());
  lowered_exprs_.insert(lowered_exprs_.end() - 1, expr);
}

void IndexLowering::handle(const kir::IfThenElse* ite) {
  const auto prev_scope = active_scope_;

  // Loop rotation transform loops like
  //  for i ...
  //    statement1(i)
  //    statement2(i)
  //    statement3(i)
  //    statement4(i)
  // into
  //  statement1(0)
  //  statement2(0)
  //  for i ...
  //    statement3(i)
  //    statement4(i)
  //    if LoopRotation:
  //      statement1(i+1)
  //      statement2(i+1)
  // So when we see an `if LoopRotation` during visiting, the last loop is
  // rotated, and we need to use `i+1` instead of `i` as loop index.
  if (ite->predicate()->predicate_type() == PredicateType::LoopRotation) {
    rotated_loop_.insert(for_loops_.back());
  }

  auto new_ite = IrBuilder::create<kir::IfThenElse>(ite->predicate());
  pushBack(new_ite);

  active_scope_ = &new_ite->thenBody();

  for (auto expr : ite->thenBody().exprs()) {
    OptOutConstDispatch::dispatch(expr);
  }

  active_scope_ = &new_ite->elseBody();

  for (auto expr : ite->elseBody().exprs()) {
    OptOutConstDispatch::dispatch(expr);
  }

  active_scope_ = prev_scope;

  if (ite->predicate()->predicate_type() == PredicateType::LoopRotation) {
    rotated_loop_.erase(for_loops_.back());
  }
}

void IndexLowering::handle(const ForLoop* for_loop) {
  const auto prev_scope = active_scope_;

  auto new_for_loop = IrBuilder::create<ForLoop>(for_loop);
  pushBack(new_for_loop);

  active_scope_ = &new_for_loop->body();
  for_loops_.push_back(new_for_loop);

  for (auto expr : for_loop->body().exprs()) {
    OptOutConstDispatch::dispatch(expr);
  }

  for_loops_.pop_back();
  active_scope_ = prev_scope;
}

void IndexLowering::handle(const RNGOp* rop) {
  // Write random tensor indices into the consumer
  //  tensor index if the output is a tensor.
  auto out_tv = dynamic_cast<TensorView*>(rop->output(0));
  NVF_ERROR(out_tv != nullptr, "rand scalar not yet supported");

  // TensorIndex for philox subsequence and component.
  auto philox_index =
      Index::getLinearLogicalIndex(out_tv, for_loops_, getRotatedLoop());
  philox_index = GpuLower::current()->commonScalarMap().hoistScalar(
      philox_index, for_loops_);

  // TensorIndex for writing rand_like output.
  const auto out = lowerDstIndex(out_tv);

  auto lowered = IrBuilder::create<RNGOp>(
      rop->getRNGOpType(),
      out,
      rop->dtype(),
      rop->getParameters(),
      rop->getRNGSeedVal(),
      rop->getRNGOffsetVal(),
      philox_index);

  pushBack(lowered);
  GpuLower::current()->propagateExprInfo(rop, back());
}

void IndexLowering::handle(const FullOp* fop) {
  auto out_tv = dynamic_cast<TensorView*>(fop->output(0));
  NVF_ERROR(out_tv != nullptr);

  // TensorIndex for writing output.
  const auto out = lowerDstIndex(out_tv);
  auto result = fop->getFillValue();
  GpuLower::current()->commonScalarMap().hoistScalar(result, for_loops_);

  auto lowered =
      IrBuilder::create<LoadStoreOp>(LoadStoreOpType::Set, out, result);
  pushBack(lowered);
  GpuLower::current()->propagateExprInfo(fop, back());
}

void IndexLowering::handle(const IotaOp* aop) {
  // Write linear tensor indices into the consumer
  //  tensor index if the output is a tensor.
  auto out_tv = dynamic_cast<TensorView*>(aop->output(0));
  NVF_ERROR(out_tv != nullptr);

  // TensorIndex for writing iota output.
  const auto out = lowerDstIndex(out_tv);
  auto result = Index::iota(
      out_tv,
      for_loops_,
      getRotatedLoop(),
      aop->start(),
      aop->step(),
      aop->dtype());
  auto lowered =
      IrBuilder::create<LoadStoreOp>(LoadStoreOpType::Set, out, result);

  pushBack(lowered);
  GpuLower::current()->propagateExprInfo(aop, back());
}

void IndexLowering::handle(const EyeOp* eop) {
  auto out_tv = dynamic_cast<TensorView*>(eop->output(0));
  NVF_ERROR(out_tv != nullptr);

  // TensorIndex for writing eye output.
  const auto out = lowerDstIndex(out_tv);
  auto result = Index::eye(out_tv, for_loops_, getRotatedLoop(), eop->dtype());
  auto lowered =
      IrBuilder::create<LoadStoreOp>(LoadStoreOpType::Set, out, result);

  pushBack(lowered);
  GpuLower::current()->propagateExprInfo(eop, back());
}

void IndexLowering::handle(const UnaryOp* uop) {
  const auto in = lowerSrcIndex(uop->in(), uop->out());
  const auto out = lowerDstIndex(uop->out());
  pushBack(IrBuilder::create<UnaryOp>(uop->getUnaryOpType(), out, in));
  GpuLower::current()->propagateExprInfo(uop, back());
}

void IndexLowering::handle(const BinaryOp* bop) {
  const auto lhs = lowerSrcIndex(bop->lhs(), bop->out());
  const auto rhs = lowerSrcIndex(bop->rhs(), bop->out());
  const auto out = lowerDstIndex(bop->out());
  pushBack(IrBuilder::create<BinaryOp>(bop->getBinaryOpType(), out, lhs, rhs));
  GpuLower::current()->propagateExprInfo(bop, back());
}

void IndexLowering::handle(const TernaryOp* top) {
  const auto in1 = lowerSrcIndex(top->in1(), top->out());
  const auto in2 = lowerSrcIndex(top->in2(), top->out());
  const auto in3 = lowerSrcIndex(top->in3(), top->out());
  const auto out = lowerDstIndex(top->out());
  pushBack(IrBuilder::create<TernaryOp>(
      top->getTernaryOpType(), out, in1, in2, in3));
  GpuLower::current()->propagateExprInfo(top, back());
}

void IndexLowering::handle(const ArrayConstruct* aop) {
  std::vector<Val*> lowered_inputs;
  for (auto input : aop->inputs()) {
    lowered_inputs.push_back(lowerSrcIndex(input, aop->out()));
  }
  const auto out = lowerDstIndex(aop->out());
  pushBack(IrBuilder::create<ArrayConstruct>(out, lowered_inputs));
  GpuLower::current()->propagateExprInfo(aop, back());
}

void IndexLowering::handle(const StructConstruct* sop) {
  std::vector<std::pair<std::string, Val*>> lowered_named_inputs;
  for (auto i : c10::irange(sop->inputs().size())) {
    lowered_named_inputs.emplace_back(
        sop->fieldName(i), lowerSrcIndex(sop->inputs().at(i), sop->out()));
  }
  const auto out = lowerDstIndex(sop->out());
  pushBack(IrBuilder::create<StructConstruct>(out, lowered_named_inputs));
  GpuLower::current()->propagateExprInfo(sop, back());
}

void IndexLowering::handle(const GetAttr* gop) {
  const auto struct_ = lowerSrcIndex(gop->struct_(), gop->out());
  const auto attr = gop->attr();
  const auto out = lowerDstIndex(gop->out());
  pushBack(IrBuilder::create<GetAttr>(out, struct_, attr));
  GpuLower::current()->propagateExprInfo(gop, back());
}

void IndexLowering::handle(const GetItem* gop) {
  const auto array = lowerSrcIndex(gop->array(), gop->out());
  const auto index = lowerSrcIndex(gop->index(), gop->out());
  const auto out = lowerDstIndex(gop->out());
  pushBack(IrBuilder::create<GetItem>(out, array, index));
  GpuLower::current()->propagateExprInfo(gop, back());
}

void IndexLowering::handle(const GetMetaData* gop) {
  const auto in = gop->in();
  const auto out = lowerDstIndex(gop->out());
  pushBack(IrBuilder::create<GetMetaData>(out, in));
  GpuLower::current()->propagateExprInfo(gop, back());
}

void IndexLowering::handle(const TensorConstruct* cop) {
  const auto out = lowerDstIndex(cop->out());
  auto indices = Index::getConsumerPerDimLogicalIndex(
      cop->out(), for_loops_, getRotatedLoop());
  auto in = cop->in();
  for (auto index : indices) {
    in = IrBuilder::getItemExpr(in, index);
  }
  in = GpuLower::current()->commonScalarMap().hoistScalar(in, for_loops_);
  pushBack(IrBuilder::create<LoadStoreOp>(LoadStoreOpType::Set, out, in));
  GpuLower::current()->propagateExprInfo(cop, back());
}

void IndexLowering::handle(const IndexSelectOp* sop) {
  auto lowered_index = lowerSrcIndex(sop->input(1), sop->output(0));
  lowered_index = maybeCastOp(DataType::Index, lowered_index);

  const std::unordered_map<IterDomain*, Val*> override_index = {
      {sop->getIndexedID(), lowered_index}};
  const auto lookup =
      lowerSrcIndex(sop->input(0), sop->output(0), override_index);

  const auto out = lowerDstIndex(sop->output(0));
  pushBack(
      IrBuilder::create<IndexSelectOp>(out, lookup, sop->dim(), lowered_index));
  GpuLower::current()->propagateExprInfo(sop, back());
}

void IndexLowering::handle(const TorchGatherOp* top) {
  auto lowered_index = lowerSrcIndex(top->input(1), top->output(0));
  lowered_index = IrBuilder::maybeCastExpr(DataType::Index, lowered_index);

  const std::unordered_map<IterDomain*, Val*> override_index = {
      {top->getIndexedID(), lowered_index}};

  auto input = lowerSrcIndex(top->lookupTv(), top->output(0), override_index);

  const auto out = lowerDstIndex(top->output(0));
  pushBack(IrBuilder::create<LoadStoreOp>(LoadStoreOpType::Set, out, input));
  GpuLower::current()->propagateExprInfo(top, back());
}

void IndexLowering::handle(const ScatterOp* sop) {
  auto lowered_index = lowerSrcIndex(sop->indexTv(), sop->output(0));
  auto lowered_src = lowerSrcIndex(sop->srcTv(), sop->output(0));

  lowered_index = IrBuilder::maybeCastExpr(DataType::Index, lowered_index);

  const std::unordered_map<int, Val*> override_index_out = {
      {sop->dim(), lowered_index}};
  auto lowered_out = lowerDstIndex(sop->output(0), override_index_out);

  pushBack(IrBuilder::create<ScatterOp>(
      sop->getScatterOpType(),
      lowered_out,
      sop->selfTv(),
      sop->dim(),
      lowered_index,
      lowered_src));
  GpuLower::current()->propagateExprInfo(sop, back());
}

void IndexLowering::handle(const SelectOp* sop) {
  auto lowered_index = lowerSrcIndex(sop->input(1), sop->output(0));
  auto lowered_index_cast = lowered_index;

  // If the type of the index tensor is different from the kernel
  // index type, promote it to the kernel index type
  if (GpuLower::current()->kernel()->indexType() !=
      sop->input(1)->getDataType().value()) {
    lowered_index_cast =
        IrBuilder::create<Val>(GpuLower::current()->kernel()->indexType());
    IrBuilder::create<UnaryOp>(
        UnaryOpType::Cast, lowered_index_cast, lowered_index);
  }

  const std::unordered_map<IterDomain*, Val*> override_index = {
      {sop->getIndexedID(), lowered_index_cast}};
  const auto input =
      lowerSrcIndex(sop->input(0), sop->output(0), override_index);

  const auto out = lowerDstIndex(sop->output(0));

  pushBack(IrBuilder::create<LoadStoreOp>(LoadStoreOpType::Set, out, input));
  GpuLower::current()->propagateExprInfo(sop, back());
}

void IndexLowering::handle(const ViewAsScalar* uop) {
  const auto in = lowerSrcIndex(uop->in(), uop->out());
  const auto out = lowerDstIndex(uop->out());
  for (auto loop : for_loops_) {
    if (GpuLower::current()->caMap()->areMapped(
            loop->iter_domain(),
            uop->vector_id()->as<IterDomain>(),
            IdMappingMode::LOOP)) {
      // TODO: this doesn't work with loop rotation
      Val* index = loop->indexOrStartIfTrivial();
      pushBack(IrBuilder::create<LoadStoreOp>(
          LoadStoreOpType::Set, out, IrBuilder::getItemExpr(in, index)));
      GpuLower::current()->propagateExprInfo(uop, back());
      return;
    }
  }
  NVF_THROW("Can not find index for vector dim");
}

namespace {

struct GridCommWorkBufferSizeInfo {
  // Size of overall buffer. Can be expanded for privatization
  Val* size_of_privatized_buffer = nullptr;
  // Size of single buffer.
  Val* buffer_stride = nullptr;
};

// Get the size of the temporary work buffer for grid communication, this can be
// grid reduction, broadcast, or grid welford.
// The buffer is expanded for privatization when not persistent or grouped.
GridCommWorkBufferSizeInfo getGridCommWorkBufferSize(
    const TensorDomain* td,
    const std::vector<ForLoop*>& for_loops,
    bool is_persistent) {
  // The buffer size is the number of thread blocks multiplied by the
  // number of threads not used for reduction domains.
  // Note: Previously it was calculated based on the shape of the
  // tensor, but it makes more sense to compute the size based on the
  // shape of the thread block and grid since this buffer is used for
  // communications among them. Both methods should result in the same
  // size if the parallel dimensions are exact, but otherwise, just
  // computing the buffer size based on the tensor shape isn't
  // sufficient since there could be extra threads/blocks.
  Val* size_of_single_buffer = GpuLower::current()->kernel()->oneVal();
  for (auto pt : kParallelTypeThreads) {
    auto pt_dim = GpuLower::current()->parallelDimensionMap().get(pt);
    if (pt_dim == nullptr || pt_dim->isOneInt()) {
      continue;
    }
    if (isParallelTypeThreadDim(pt) &&
        std::any_of(td->loop().begin(), td->loop().end(), [&](auto out_id) {
          return out_id->getParallelType() == pt &&
              (out_id->isReduction() || out_id->isBroadcast());
        })) {
      continue;
    }
    size_of_single_buffer =
        SimplifyingIrBuilder::mulExpr(size_of_single_buffer, pt_dim);
  }

  // Expand the buffer for privatization. The buffer is expanded so
  // that each non-reduction IterDomain uses a different part of the
  // buffer. For persistent mode, this expansion is only done for
  // grouped IterDomains.

  Val* size_of_privatized_buffer = size_of_single_buffer;

  // In persistent mode, if non-grouped no-reduction domain is used,
  // double the buffer size to save a final grid sync
  bool is_doubled = false;

  for (auto fl : for_loops) {
    // Buffer size of parallelized domains are already taken care
    if (fl->isTrivial() || fl->iter_domain()->isReduction() ||
        fl->iter_domain()->isThread()) {
      continue;
    }
    // If persistent, i.e., allreduce, only IterDomains with
    // ParallelType::Group are privatized
    if (!is_persistent ||
        fl->iter_domain()->getParallelType() == ParallelType::Group) {
      size_of_privatized_buffer = SimplifyingIrBuilder::mulExpr(
          size_of_privatized_buffer, fl->iter_domain()->extent());
    } else if (is_persistent) {
      is_doubled = true;
    }
  }

  if (is_doubled) {
    size_of_privatized_buffer = SimplifyingIrBuilder::mulExpr(
        size_of_privatized_buffer, IrBuilder::create<Val>(2L, DataType::Index));
  }

  GridCommWorkBufferSizeInfo info;
  info.size_of_privatized_buffer = size_of_privatized_buffer;
  info.buffer_stride = size_of_single_buffer;
  if (is_doubled) {
    info.buffer_stride = SimplifyingIrBuilder::mulExpr(
        info.buffer_stride, IrBuilder::create<Val>(2L, DataType::Index));
  }

  return info;
}

Val* getGridSyncBufferSize(
    const TensorDomain* td,
    const std::vector<ForLoop*>& for_loops,
    bool is_persistent) {
  // See the comment above for getGridCommWorkBufferSize.
  Val* buffer_size = GpuLower::current()->kernel()->oneVal();
  for (auto pt : kParallelTypeBIDs) {
    auto pt_dim = GpuLower::current()->parallelDimensionMap().get(pt);
    if (pt_dim == nullptr || pt_dim->isOneInt()) {
      continue;
    }
    if (std::any_of(td->loop().begin(), td->loop().end(), [&](auto out_id) {
          return out_id->getParallelType() == pt &&
              (out_id->isReduction() || out_id->isBroadcast());
        })) {
      continue;
    }
    buffer_size = SimplifyingIrBuilder::mulExpr(buffer_size, pt_dim);
  }

  // If not persistent, all iteration domains require a separate
  // semaphore for re-entrant grid reductions
  if (!is_persistent) {
    for (auto fl : for_loops) {
      if (fl->isTrivial()) {
        continue;
      }
      if (fl->iter_domain()->isThread()) {
        // already accounted for.
        continue;
      }

      buffer_size = SimplifyingIrBuilder::mulExpr(
          buffer_size, fl->iter_domain()->extent());
    }
  }

  return buffer_size;
}

Val* getEntranceCountGridReduce(std::vector<ForLoop*>& for_loops) {
  Val* grid_reduction_entrances = GpuLower::current()->kernel()->oneVal();

  for (const auto loop : for_loops) {
    if (loop->isTrivial()) {
      continue;
    }
    if (loop->iter_domain()->isThread()) {
      // already accounted for.
      continue;
    }
    // TODO: Does this work for shift/gather?
    grid_reduction_entrances = SimplifyingIrBuilder::mulExpr(
        grid_reduction_entrances, loop->iter_domain()->extent());
  }
  return grid_reduction_entrances;
}

// Linear indexing of for loops for multiple entrances into grid reduce
// TODO: What happens if there's a broadcast that's resolved (not present in the
// grid reduce) but the global buffer isn't expanded?
Val* getEntranceLinIndGridReduce(std::vector<ForLoop*>& for_loops) {
  Val* linear_index = GpuLower::current()->kernel()->zeroVal();

  for (const auto loop : for_loops) {
    if (loop->isTrivial()) {
      continue;
    }
    if (loop->iter_domain()->isThread()) {
      // already accounted for.
      continue;
    }
    // TODO: Does this work for shift/gather/loop rotation?
    linear_index = SimplifyingIrBuilder::addExpr(
        SimplifyingIrBuilder::mulExpr(
            linear_index, loop->iter_domain()->extent()),
        loop->indexOrStartIfTrivial());
  }
  return linear_index;
}

} // namespace

void IndexLowering::handle(const ReductionOp* rop) {
  NVF_ERROR(ir_utils::isTvOp(rop));

  const auto out_tv = rop->out()->as<TensorView>();
  const auto out_domain = out_tv->domain();

  const bool has_block_reduce = out_domain->hasBlockReduction();
  const bool has_grid_reduce = out_domain->hasGridReduction();

  const auto out = lowerDstIndex(rop->out());
  const auto in = lowerSrcIndex(rop->in(), rop->out());

  if (has_grid_reduce) {
    handleGridReduction(rop, out, in);
  } else if (has_block_reduce) {
    handleBlockReduction(rop, out, in);
  } else {
    pushBack(
        IrBuilder::create<BinaryOp>(rop->getReductionOpType(), out, out, in));
    GpuLower::current()->propagateExprInfo(rop, back());
  }
}

void IndexLowering::handleBlockReduction(
    const ReductionOp* rop,
    Val* out,
    Val* in) {
  NVF_ERROR(ir_utils::isTvOp(rop));

  ReductionOp* indexed_rop = IrBuilder::create<ReductionOp>(
      rop->getReductionOpType(), rop->init(), out, in, rop->isAllreduce());
  if (rop->predicate()) {
    indexed_rop =
        indexed_rop->withPredicate(rop->predicate())->as<ReductionOp>();
  }
  if (rop->writePredicate()) {
    indexed_rop = indexed_rop->withWritePredicate(rop->writePredicate())
                      ->as<ReductionOp>();
  }

  pushBack(indexed_rop);
  GpuLower::current()->propagateExprInfo(rop, back());
}

void IndexLowering::handleSerialGridReduction(
    const ReductionOp* rop,
    Val* out,
    Val* in) {
  const auto out_tv = out->as<kir::TensorIndex>()->view();
  const auto out_domain = out_tv->domain();

  // If we do a grid reduction we can't have a reduction axis that is not bound
  // to a grid or block dim.
  NVF_ERROR(
      std::none_of(
          out_domain->loop().begin(),
          out_domain->loop().end(),
          [](IterDomain* id) {
            return !id->isThread() && id->isReduction() &&
                !id->extent()->isOneInt();
          }),
      "Found a reduction stage that has both a non-parallelized ",
      "reduction and a grid reduction. This is not supported, ",
      "please use rfactor to do the serialized reduction first, ",
      "then the grid reduction. ",
      rop->toString());

  NVF_ERROR(!rop->isAllreduce(), "Serial grid allReduce is not implemented");

  // Allocate global work buffer TensorIndex.
  //
  // For convenience, the global work buffer is allocated like the loop domain
  // of the ReductionOp output. In the future, we may want the allocation
  // domain to be different in order to enable re-use of global output buffers
  // for in-place reduction.
  std::vector<IterDomain*> work_buffer_root;
  work_buffer_root.reserve(out_tv->nDims());
  for (IterDomain* id : out_tv->getLoopDomain()) {
    work_buffer_root.push_back(IterDomainBuilder(id).build());
  }
  auto work_buffer_domain = IrBuilder::create<TensorDomain>(work_buffer_root);
  auto work_buffer_tv = IrBuilder::create<TensorView>(
      work_buffer_domain, out_tv->dtype(), MemoryType::Global);
  Val* work_buffer_idx_val = nullptr;
  for (auto v :
       Index::getGlobalConsumerStridedIndices(out_tv, for_loops_, {})) {
    work_buffer_idx_val = SimplifyingIrBuilder::addExpr(work_buffer_idx_val, v);
  }

  auto work_buffer_idx = IrBuilder::create<kir::TensorIndex>(
      work_buffer_tv,
      GpuLower::current()->commonScalarMap().hoistScalar(
          work_buffer_idx_val, for_loops_));

  auto work_alloc = IrBuilder::create<kir::Allocate>(
      work_buffer_tv, work_buffer_tv->getMemoryType());
  pushBack(work_alloc);

  // The thread predicate for GridReduction needs to be set
  // separately from the main predicate. Do not combine them like
  // other expressions.
  const auto& thread_pred =
      GpuLower::current()->threadPredMap().getPredicatedParallelTypes(out_tv);

  auto serial_grid_reduction = IrBuilder::create<kir::GridReduction>(
      rop->getReductionOpType(),
      rop->init(),
      out,
      in,
      // skip work_buffer, sync_buffer, entrance_ind, n_entrances for serial
      // reduction node
      nullptr,
      nullptr,
      nullptr,
      nullptr,
      false,
      work_buffer_idx);

  serial_grid_reduction =
      serial_grid_reduction->withThreadPredicate(thread_pred);

  if (rop->predicate()) {
    serial_grid_reduction =
        serial_grid_reduction->withPredicate(rop->predicate())
            ->as<kir::GridReduction>();
  }
  if (rop->writePredicate()) {
    serial_grid_reduction =
        serial_grid_reduction->withWritePredicate(rop->writePredicate())
            ->as<kir::GridReduction>();
  }

  pushBack(serial_grid_reduction);
  GpuLower::current()->propagateExprInfo(rop, back());
}

void IndexLowering::handleGridReduction(
    const ReductionOp* rop,
    Val* out,
    Val* in) {
  if (rop->serialGridReductionRequested()) {
    handleSerialGridReduction(rop, out, in);
    return;
  }

  const auto out_tv = out->as<kir::TensorIndex>()->view();
  const auto out_domain = out_tv->domain();

  NVF_ERROR(out_domain->hasGridReduction());

  // If we do a grid reduction we can't have a reduction axis that is not bound
  // to a grid or block dim.
  NVF_ERROR(
      std::none_of(
          out_domain->loop().begin(),
          out_domain->loop().end(),
          [](IterDomain* id) {
            return !id->isThread() && id->isReduction() &&
                !id->extent()->isOneInt();
          }),
      "Found a reduction stage that has both a non-parallelized ",
      "reduction and a grid reduction. This is not supported, ",
      "please use rfactor to do the serialized reduction first, ",
      "then the grid reduction. ",
      rop->toString());

  // Use a unique buffer for work and sync flag when called within a
  // loop unless it's persistent. Grid all reduce means persistence is
  // required. However, not being a grid all reduce does not mean
  // non-persistence. Currently, if a cooperative grid reduction is
  // required anywhere in the kernel, all grid reducitons are done in
  // a persistent manner, so all grid reductions should be consulted.
  // TODO: fix this
  const bool is_persistent = rop->isAllreduce();
  const auto buffer_size_info =
      getGridCommWorkBufferSize(out_domain, for_loops_, is_persistent);

  auto work_buffer = allocateUniqueBuffer(
      buffer_size_info.size_of_privatized_buffer,
      out_tv->dtype(),
      false,
      out_tv,
      work_buffer_map_);

  auto sync_buffer_size =
      getGridSyncBufferSize(out_domain, for_loops_, is_persistent);
  auto sync_buffer = allocateUniqueBuffer(
      sync_buffer_size, DataType::Int, true, out_tv, sync_buffer_map_);

  const auto entrance_ind = !is_persistent
      ? getEntranceLinIndGridReduce(for_loops_)
      : GpuLower::current()->kernel()->zeroVal();
  const auto n_entrances = !is_persistent
      ? getEntranceCountGridReduce(for_loops_)
      : GpuLower::current()->kernel()->oneVal();

  // The thread predicate for GridReduction needs to be set
  // separately from the main predicate. Do not combine them like
  // other expressions.
  const auto& thread_pred =
      GpuLower::current()->threadPredMap().getPredicatedParallelTypes(out_tv);

  auto grid_reduction = IrBuilder::create<kir::GridReduction>(
      rop->getReductionOpType(),
      rop->init(),
      out,
      in,
      work_buffer,
      sync_buffer,
      entrance_ind,
      n_entrances,
      rop->isAllreduce());

  grid_reduction = grid_reduction->withThreadPredicate(thread_pred);

  if (rop->predicate()) {
    grid_reduction = grid_reduction->withPredicate(rop->predicate())
                         ->as<kir::GridReduction>();
  }
  if (rop->writePredicate()) {
    grid_reduction = grid_reduction->withWritePredicate(rop->writePredicate())
                         ->as<kir::GridReduction>();
  }

  pushBack(grid_reduction);
  GpuLower::current()->propagateExprInfo(rop, back());

  if (rop->isAllreduce()) {
    allocateUniqueFusedReduction(grid_reduction, out_tv);
  }
}

void IndexLowering::handle(const GroupedReductionOp* grouped_rop) {
  NVF_ERROR(ir_utils::isTvOp(grouped_rop));

  const auto out_tv = ir_utils::getTvOutput(grouped_rop);
  const auto out_domain = out_tv->domain();

  const bool has_block_reduce = out_domain->hasBlockReduction();
  const bool has_grid_reduce = out_domain->hasGridReduction();

  std::vector<Val*> indexed_outputs(grouped_rop->numHorizontallyGroupedExprs());
  std::vector<Val*> indexed_inputs(grouped_rop->numHorizontallyGroupedExprs());

  for (const auto i : c10::irange(grouped_rop->numHorizontallyGroupedExprs())) {
    indexed_outputs.at(i) = lowerDstIndex(grouped_rop->output(i));
    indexed_inputs.at(i) =
        lowerSrcIndex(grouped_rop->input(i), grouped_rop->output(i));
  }

  if (has_grid_reduce) {
    handleGridReduction(grouped_rop, indexed_outputs, indexed_inputs);
  } else if (has_block_reduce) {
    handleBlockReduction(grouped_rop, indexed_outputs, indexed_inputs);
  } else {
    for (const auto i :
         c10::irange(grouped_rop->numHorizontallyGroupedExprs())) {
      pushBack(IrBuilder::create<BinaryOp>(
          grouped_rop->getReductionOpType(i),
          indexed_outputs.at(i),
          indexed_outputs.at(i),
          indexed_inputs.at(i)));
    }
  }
}

void IndexLowering::handleBlockReduction(
    const GroupedReductionOp* grouped_rop,
    const std::vector<Val*>& outputs,
    const std::vector<Val*>& inputs) {
  NVF_ERROR(ir_utils::isTvOp(grouped_rop));

  GroupedReductionOp* indexed_rop = IrBuilder::create<GroupedReductionOp>(
      grouped_rop->getReductionOpTypes(),
      grouped_rop->initVals(),
      outputs,
      inputs,
      grouped_rop->isAllreduce());
  if (grouped_rop->predicate()) {
    indexed_rop = indexed_rop->withPredicate(grouped_rop->predicate())
                      ->as<GroupedReductionOp>();
  }
  if (grouped_rop->writePredicate()) {
    indexed_rop = indexed_rop->withWritePredicate(grouped_rop->writePredicate())
                      ->as<GroupedReductionOp>();
  }

  pushBack(indexed_rop);
  GpuLower::current()->propagateExprInfo(grouped_rop, back());
}

void IndexLowering::handleGridReduction(
    const GroupedReductionOp* grouped_rop,
    const std::vector<Val*>& outputs,
    const std::vector<Val*>& inputs) {
  const auto out_tv = ir_utils::getTvOutput(grouped_rop);
  const auto out_domain = out_tv->domain();

  NVF_ERROR(out_domain->hasGridReduction());

  // If we do a grid reduction we can't have a reduction axis that is not bound
  // to a grid or block dim.
  NVF_ERROR(
      std::none_of(
          out_domain->loop().begin(),
          out_domain->loop().end(),
          [](IterDomain* id) {
            return !id->isThread() && id->isReduction() &&
                !id->extent()->isOneInt();
          }),
      "Found a reduction stage that has both a non-parallelized ",
      "reduction and a grid reduction. This is not supported, ",
      "please use rfactor to do the serialized reduction first, ",
      "then the grid reduction.");

  const bool is_persistent = grouped_rop->isAllreduce();
  auto work_buf_size_info =
      getGridCommWorkBufferSize(out_domain, for_loops_, is_persistent);

  std::vector<kir::Allocate*> work_buffers;
  std::transform(
      outputs.begin(),
      outputs.end(),
      std::back_inserter(work_buffers),
      [&](Val* output) {
        return allocateUniqueBuffer(
            work_buf_size_info.size_of_privatized_buffer,
            output->dtype(),
            false,
            output->as<kir::TensorIndex>()->view(),
            work_buffer_map_);
      });

  auto sync_buffer_size =
      getGridSyncBufferSize(out_domain, for_loops_, is_persistent);
  auto sync_buffer = allocateUniqueBuffer(
      sync_buffer_size, DataType::Int, true, out_tv, sync_buffer_map_);

  const auto entrance_ind = !is_persistent
      ? getEntranceLinIndGridReduce(for_loops_)
      : GpuLower::current()->kernel()->zeroVal();
  const auto n_entrances = !is_persistent
      ? getEntranceCountGridReduce(for_loops_)
      : GpuLower::current()->kernel()->oneVal();

  // The thread predicate for GridReduction needs to be set
  // separately from the main predicate. Do not combine them like
  // other expressions.
  const auto& thread_pred =
      GpuLower::current()->threadPredMap().getPredicatedParallelTypes(out_tv);

  auto grid_reduction = IrBuilder::create<kir::GroupedGridReduction>(
      grouped_rop->getReductionOpTypes(),
      grouped_rop->initVals(),
      outputs,
      inputs,
      work_buffers,
      sync_buffer,
      entrance_ind,
      n_entrances,
      work_buf_size_info.buffer_stride,
      grouped_rop->isAllreduce());

  grid_reduction = grid_reduction->withThreadPredicate(thread_pred);

  if (grouped_rop->predicate()) {
    grid_reduction = grid_reduction->withPredicate(grouped_rop->predicate())
                         ->as<kir::GroupedGridReduction>();
  }
  if (grouped_rop->writePredicate()) {
    grid_reduction =
        grid_reduction->withWritePredicate(grouped_rop->writePredicate())
            ->as<kir::GroupedGridReduction>();
  }

  pushBack(grid_reduction);
  GpuLower::current()->propagateExprInfo(grouped_rop, back());

  if (grouped_rop->isAllreduce()) {
    allocateUniqueFusedReduction(grid_reduction, out_tv);
  }
}

void IndexLowering::handle(const WelfordOp* wop) {
  NVF_ERROR(ir_utils::isTvOp(wop));

  const auto out_tv = wop->outAvg()->as<TensorView>();
  const auto out_domain = out_tv->domain();

  const bool has_block_reduce = out_domain->hasBlockReduction();
  const bool has_grid_reduce = out_domain->hasGridReduction();

  if (has_grid_reduce) {
    NVF_ERROR(
        std::none_of(
            out_domain->loop().begin(),
            out_domain->loop().end(),
            [](IterDomain* id) {
              return !id->isThread() && id->isReduction();
            }),
        "Found a reduction stage that has both a non-parallelized ",
        "reduction and a grid reduction.  This is not supported, ",
        "please use rfactor to do the serialized reduction first, ",
        "then the grid reduction.");
  }

  // lower IO tensors
  const auto in_var =
      wop->inVar() ? lowerSrcIndex(wop->inVar(), wop->outAvg()) : nullptr;
  const auto in_avg = lowerSrcIndex(wop->inAvg(), wop->outAvg());
  auto in_N = wop->inN();

  // in Rfactor-ed case, the input N is actually a TV
  if (!in_N->isScalar()) {
    in_N = lowerSrcIndex(in_N, wop->outN());
  }

  auto out_avg = lowerDstIndex(wop->outAvg());
  auto out_var = lowerDstIndex(wop->outVar());
  auto out_N = lowerDstIndex(wop->outN());

  WelfordOp* indexed_wop = IrBuilder::create<WelfordOp>(
      out_avg,
      out_var,
      out_N,
      in_avg,
      in_var,
      in_N,
      wop->initAvg(),
      wop->initVar(),
      wop->initN(),
      wop->isAllreduce());

  if (wop->predicate()) {
    indexed_wop = indexed_wop->withPredicate(wop->predicate())->as<WelfordOp>();
  }
  if (wop->writePredicate()) {
    indexed_wop =
        indexed_wop->withWritePredicate(wop->writePredicate())->as<WelfordOp>();
  }

  // Serial welford
  if (!has_block_reduce && !has_grid_reduce) {
    pushBack(indexed_wop);
    GpuLower::current()->propagateExprInfo(wop, back());
    return;
  }

  // Block-only welford
  if (!has_grid_reduce) {
    pushBack(indexed_wop);
    GpuLower::current()->propagateExprInfo(wop, back());
    return;
  }

  handleGridWelford(indexed_wop);
}

void IndexLowering::handleGridWelford(WelfordOp* indexed_wop) {
  const auto out_tv = indexed_wop->out()->as<kir::TensorIndex>()->view();
  const auto out_domain = out_tv->domain();

  // TODO: See the comment on the same variable in handleGridReduction
  const bool is_persistent = indexed_wop->isAllreduce();
  const auto buffer_size_info =
      getGridCommWorkBufferSize(out_domain, for_loops_, is_persistent);

  const auto work_buffer_size = buffer_size_info.size_of_privatized_buffer;
  auto out_avg_buffer = allocateUniqueBuffer(
      work_buffer_size,
      indexed_wop->outAvg()->dtype(),
      false,
      indexed_wop->outAvg()->as<kir::TensorIndex>()->view(),
      work_buffer_map_);
  auto out_var_buffer = allocateUniqueBuffer(
      work_buffer_size,
      indexed_wop->outVar()->dtype(),
      false,
      indexed_wop->outVar()->as<kir::TensorIndex>()->view(),
      work_buffer_map_);
  auto out_N_buffer = allocateUniqueBuffer(
      work_buffer_size,
      indexed_wop->outN()->dtype(),
      false,
      indexed_wop->outN()->as<kir::TensorIndex>()->view(),
      work_buffer_map_);

  auto sync_buffer_size =
      getGridSyncBufferSize(out_domain, for_loops_, is_persistent);
  auto sync_buffer = allocateUniqueBuffer(
      sync_buffer_size, DataType::Int, true, out_tv, sync_buffer_map_);

  const auto entrance_ind = !is_persistent
      ? getEntranceLinIndGridReduce(for_loops_)
      : GpuLower::current()->kernel()->zeroVal();
  const auto n_entrances = !is_persistent
      ? getEntranceCountGridReduce(for_loops_)
      : GpuLower::current()->kernel()->oneVal();

  // The thread predicate for GridReduction needs to be set
  // separately from the main predicate. Do not combine them like
  // other expressions.
  const auto& thread_pred =
      GpuLower::current()->threadPredMap().getPredicatedParallelTypes(out_tv);

  auto grid_welford = IrBuilder::create<kir::GridWelford>(
      indexed_wop,
      out_var_buffer,
      out_avg_buffer,
      out_N_buffer,
      sync_buffer,
      entrance_ind,
      n_entrances);

  grid_welford = grid_welford->withThreadPredicate(thread_pred);

  const bool block_reduce_separated =
      out_domain->hasBlockReduction() && !indexed_wop->isAllreduce();

  if (indexed_wop->predicate()) {
    if (block_reduce_separated) {
      grid_welford = grid_welford
                         ->withPredicate(IrBuilder::create<kir::Predicate>(
                             GpuLower::current()->kernel()->trueVal()))
                         ->as<kir::GridWelford>();
    } else {
      grid_welford = grid_welford->withPredicate(indexed_wop->predicate())
                         ->as<kir::GridWelford>();
    }
  }

  if (indexed_wop->writePredicate()) {
    grid_welford =
        grid_welford->withWritePredicate(indexed_wop->writePredicate())
            ->as<kir::GridWelford>();
  }

  if (block_reduce_separated) {
    pushBack(indexed_wop);
    GpuLower::current()->propagateExprInfo(indexed_wop, back());
  }

  pushBack(grid_welford);
  GpuLower::current()->propagateExprInfo(indexed_wop, back());

  if (indexed_wop->isAllreduce()) {
    // When using the fused reduction, allocate the reduction object at
    // the outer-most scope
    allocateUniqueFusedReduction(grid_welford, out_tv);
  }
}

void IndexLowering::handle(const GroupedWelfordOp* grouped_wop) {
  NVF_ERROR(ir_utils::isTvOp(grouped_wop));

  const auto out_tv = ir_utils::getTvOutput(grouped_wop);
  const auto out_domain = out_tv->domain();

  const bool has_grid_reduce = out_domain->hasGridReduction();

  std::vector<WelfordTriplet> indexed_outputs(
      grouped_wop->numHorizontallyGroupedExprs());
  std::vector<WelfordTriplet> indexed_inputs(
      grouped_wop->numHorizontallyGroupedExprs());

  auto output_vals = grouped_wop->outputVals();
  auto input_vals = grouped_wop->inputVals();

  for (const auto i : c10::irange(grouped_wop->numHorizontallyGroupedExprs())) {
    const auto& output = output_vals.at(i);
    const auto& input = input_vals.at(i);
    WelfordTriplet indexed_output;
    WelfordTriplet indexed_input;
    for (const auto j : c10::irange(3)) {
      indexed_output.get(j) = lowerDstIndex(output.get(j));
      indexed_input.get(j) = lowerSrcIndex(input.get(j), output.get(j));
    }
    indexed_outputs[i] = indexed_output;
    indexed_inputs[i] = indexed_input;
  }

  if (has_grid_reduce) {
    handleGroupedGridWelford(
        grouped_wop, indexed_outputs, indexed_inputs, grouped_wop->initVals());
  } else {
    NVF_THROW(
        "Only grid welford is supported. Validation should have caught non-grid welford grouping.");
  }
}

std::vector<kir::Allocate*> IndexLowering::allocateWelfordWorkBuffer(
    const std::vector<WelfordTriplet>& triplets,
    WelfordTriplet::ValName name,
    Val* buffer_size) {
  std::vector<kir::Allocate*> work_buffers;

  std::transform(
      triplets.begin(),
      triplets.end(),
      std::back_inserter(work_buffers),
      [&](const WelfordTriplet& output) {
        return allocateUniqueBuffer(
            buffer_size,
            output.get(name)->dtype(),
            false,
            output.get(name)->as<TensorView>(),
            work_buffer_map_);
      });

  return work_buffers;
}

namespace {

// Returns true if a GroupedWelfordOp op is eligible for using the
// outer-optimized grouped welford runtime function
bool canUseOuterOptRuntimeKernel(const GroupedWelfordOp* grouped_wop) {
  const auto out_tv = ir_utils::getTvOutput(grouped_wop);
  const auto out_domain = out_tv->domain();

  if (!out_domain->hasGridReduction()) {
    return false;
  }

  // TIDx and BIDx must be used for non-reduction domains. TIDy and
  // BIDy must be used for reduction domains.
  ParallelTypeBitmap used_pts;
  for (auto loop_id : out_domain->loop()) {
    auto pt = loop_id->getParallelType();
    if (isParallelTypeThread(pt)) {
      used_pts.set(pt);
      if ((loop_id->isReduction() &&
           (pt == ParallelType::BIDy || pt == ParallelType::TIDy)) ||
          (loop_id->getIterType() == IterType::Iteration &&
           (pt == ParallelType::BIDx || pt == ParallelType::TIDx))) {
        // valid pattern
        continue;
      } else {
        return false;
      }
    }
  }

  ParallelTypeBitmap valid_pt_map;
  valid_pt_map.set(ParallelType::BIDx);
  valid_pt_map.set(ParallelType::BIDy);
  valid_pt_map.set(ParallelType::TIDx);
  valid_pt_map.set(ParallelType::TIDy);
  if (used_pts != valid_pt_map) {
    return false;
  }

  // TIDx and TIDy must be static constant
  const auto& par_dim_map = GpuLower::current()->parallelDimensionMap();
  auto tidx_val = par_dim_map.get(ParallelType::TIDx);
  auto tidy_val = par_dim_map.get(ParallelType::TIDy);
  if (!tidx_val->isConstInt() || !tidy_val->isConstInt()) {
    return false;
  }
  auto tidx = tidx_val->evaluate().as<int64_t>();
  auto tidy = tidy_val->evaluate().as<int64_t>();

  // TIDz and BIDz must be unused or just 1. This contraint can be
  // lifted if necessary.
  auto tidz_val = par_dim_map.get(ParallelType::TIDz);
  if (tidz_val != nullptr && !tidz_val->isOneInt()) {
    return false;
  }
  auto bidz_val = par_dim_map.get(ParallelType::BIDz);
  if (bidz_val != nullptr && !bidz_val->isOneInt()) {
    return false;
  }

  // Warp reduction along threadIdx.y is a key factor for the
  // outer-optimized kernel. The larger (32 / blockDim.x) is, the more
  // effective. It shouldn't give any perf benefit when blockDim.x >=
  // 32 as there's no warp reduction. blockDim.x == 16 is not
  // preferable, but still would be better than the default
  // implementation. blockDim.x == 8 is preferred.
  if (tidx > 16) {
    return false;
  }

  int64_t num_grouped_iterations = 1;
  for (auto axis : out_domain->loop()) {
    if (axis->getParallelType() == ParallelType::Group) {
      NVF_ERROR(
          axis->extent()->isConstInt(),
          "Grouped IterDomain must have a static integer extent: ",
          axis->extent()->toInlineString());
      num_grouped_iterations *= axis->extent()->evaluate().as<int64_t>();
    }
  }

  // Assumptions about TIDx/TIDy and group size
  if (tidy % num_grouped_iterations != 0 || tidx > 32 || 32 % tidx != 0 ||
      num_grouped_iterations < 32 / tidx) {
    return false;
  }

  // Only considers the case where all outputs are local. This
  // eliminates thread predicates
  if (std::any_of(
          grouped_wop->outputs().begin(),
          grouped_wop->outputs().end(),
          [](const Val* output) {
            return !output->isA<TensorView>() ||
                output->as<TensorView>()->getMemoryType() != MemoryType::Local;
          })) {
    return false;
  }

  // Must not be predicated. If the per-thread serial reduction is
  // rfactored, the remaining block+grid reduction is not predicated.
  if (!((grouped_wop->predicate()->hasValue() &&
         grouped_wop->predicate()->value()) ||
        GpuLower::current()->predicateElimination().canOmitPredicate(
            grouped_wop))) {
    return false;
  }

  return true;
}

} // namespace

void IndexLowering::handleGroupedGridWelford(
    const GroupedWelfordOp* op,
    const std::vector<WelfordTriplet>& output_vals,
    const std::vector<WelfordTriplet>& input_vals,
    const std::vector<WelfordTriplet>& init_vals) {
  const auto out_tv = ir_utils::getTvOutput(op);
  const auto out_domain = out_tv->domain();

  NVF_ERROR(out_domain->hasGridReduction());

  // If we do a grid reduction we can't have a reduction axis that is not bound
  // to a grid or block dim.
  NVF_ERROR(
      std::none_of(
          out_domain->loop().begin(),
          out_domain->loop().end(),
          [](IterDomain* id) {
            return !id->isThread() && id->isReduction() &&
                !id->extent()->isOneInt();
          }),
      "Found a reduction stage that has both a non-parallelized ",
      "reduction and a grid reduction. This is not supported, ",
      "please use rfactor to do the serialized reduction first, ",
      "then the grid reduction.");

  const bool is_persistent = op->isAllreduce();
  auto work_buf_size_info =
      getGridCommWorkBufferSize(out_domain, for_loops_, is_persistent);

  const auto work_buffers_avg = allocateWelfordWorkBuffer(
      op->outputVals(),
      WelfordTriplet::ValName::Avg,
      work_buf_size_info.size_of_privatized_buffer);
  const auto work_buffers_var = allocateWelfordWorkBuffer(
      op->outputVals(),
      WelfordTriplet::ValName::Var,
      work_buf_size_info.size_of_privatized_buffer);
  const auto work_buffers_N = allocateWelfordWorkBuffer(
      op->outputVals(),
      WelfordTriplet::ValName::N,
      work_buf_size_info.size_of_privatized_buffer);

  auto sync_buffer_size =
      getGridSyncBufferSize(out_domain, for_loops_, is_persistent);
  auto sync_buffer = allocateUniqueBuffer(
      sync_buffer_size, DataType::Int, true, out_tv, sync_buffer_map_);

  const auto entrance_ind = !is_persistent
      ? getEntranceLinIndGridReduce(for_loops_)
      : GpuLower::current()->kernel()->zeroVal();
  const auto n_entrances = !is_persistent
      ? getEntranceCountGridReduce(for_loops_)
      : GpuLower::current()->kernel()->oneVal();

  // The thread predicate needs to be set separately from the main
  // predicate. Do not combine them like other expressions.
  const auto& thread_pred =
      GpuLower::current()->threadPredMap().getPredicatedParallelTypes(out_tv);

  bool use_outer_opt =
      !isOptionDisabled(DisableOption::GroupedGridWelfordOuterOpt) &&
      canUseOuterOptRuntimeKernel(op);

  auto indexed_op = IrBuilder::create<kir::GroupedGridWelford>(
      output_vals,
      input_vals,
      init_vals,
      std::array<std::vector<kir::Allocate*>, 3>{
          work_buffers_avg, work_buffers_var, work_buffers_N},
      sync_buffer,
      entrance_ind,
      n_entrances,
      work_buf_size_info.buffer_stride,
      op->isAllreduce(),
      use_outer_opt);

  indexed_op = indexed_op->withThreadPredicate(thread_pred);

  if (op->predicate()) {
    indexed_op = indexed_op->withPredicate(op->predicate())
                     ->as<kir::GroupedGridWelford>();
  }
  if (op->writePredicate()) {
    indexed_op = indexed_op->withWritePredicate(op->writePredicate())
                     ->as<kir::GroupedGridWelford>();
  }

  pushBack(indexed_op);
  GpuLower::current()->propagateExprInfo(op, back());

  if (op->isAllreduce()) {
    allocateUniqueFusedReduction(indexed_op, out_tv);
  }
}

void IndexLowering::handle(const kir::MBarrierInit* minit) {
  Val* smem_address_ptr = nullptr;

  if (minit->mbarrier()->isA<TensorView>()) {
    smem_address_ptr =
        lower_utils::u32IndexScalarSmemTv(minit->mbarrier()->as<TensorView>());
  } else if (minit->mbarrier()->isA<kir::TensorIndex>()) {
    smem_address_ptr = lower_utils::u32IndexScalarSmemTv(
        minit->mbarrier()->as<kir::TensorIndex>());
  } else {
    NVF_THROW("Unexpected MBarrierInit value.");
  }
  kir::MBarrierInit* minit_indexed = IrBuilder::create<kir::MBarrierInit>(
      smem_address_ptr, minit->threadCount());
  pushBack(minit_indexed);
  GpuLower::current()->propagateExprInfo(minit, minit_indexed);
}

void IndexLowering::handle(const kir::MBarrierInvalidate* minval) {
  Val* smem_address_ptr = nullptr;

  if (minval->mbarrier()->isA<TensorView>()) {
    smem_address_ptr =
        lower_utils::u32IndexScalarSmemTv(minval->mbarrier()->as<TensorView>());
  } else if (minval->mbarrier()->isA<kir::TensorIndex>()) {
    smem_address_ptr = lower_utils::u32IndexScalarSmemTv(
        minval->mbarrier()->as<kir::TensorIndex>());
  } else {
    NVF_THROW(
        "Unexpected MBarrierInvalidate barrier value: ",
        minval->mbarrier()->toString());
  }
  kir::MBarrierInvalidate* minval_indexed =
      IrBuilder::create<kir::MBarrierInvalidate>(smem_address_ptr);
  pushBack(minval_indexed);
  GpuLower::current()->propagateExprInfo(minval, minval_indexed);
}

void IndexLowering::handle(const kir::MBarrierArrive* arrive_transaction) {
  NVF_ERROR(
      arrive_transaction->mbarrier()->isA<kir::TensorIndex>(),
      "Expected kir::TensorIndex in MBarrierArriveExpectTx");

  Val* smem_address_ptr = lower_utils::u32IndexScalarSmemTv(
      arrive_transaction->mbarrier()->as<kir::TensorIndex>());
  pushBack(IrBuilder::create<kir::MBarrierArrive>(
      arrive_transaction->state(), smem_address_ptr));
}

void IndexLowering::handle(
    const kir::MBarrierArriveExpectTx* arrive_transaction) {
  NVF_ERROR(
      arrive_transaction->mbarrier()->isA<kir::TensorIndex>(),
      "Expected kir::TensorIndex in MBarrierArriveExpectTx");

  Val* smem_address_ptr = lower_utils::u32IndexScalarSmemTv(
      arrive_transaction->mbarrier()->as<kir::TensorIndex>());
  pushBack(IrBuilder::create<kir::MBarrierArriveExpectTx>(
      arrive_transaction->state(),
      smem_address_ptr,
      arrive_transaction->txCount()));
}

void IndexLowering::handle(const kir::MBarrierWait* mwait) {
  NVF_ERROR(
      mwait->mbarrier()->isA<kir::TensorIndex>(),
      "Expected kir::TensorIndex in MBarrierWait");
  Val* smem_address_ptr = lower_utils::u32IndexScalarSmemTv(
      mwait->mbarrier()->as<kir::TensorIndex>());
  pushBack(
      IrBuilder::create<kir::MBarrierWait>(smem_address_ptr, mwait->state()));
}

void IndexLowering::handleCpAsyncBulkLoad(const LoadStoreOp* ldst) {
  // If LoadStoreOp has a smem TV in ldstMBarrierTokenMap, then it is a part
  // of a circular buffer loop. The kir nodes for arrive_expect_tx and
  // mbarrier_wait are added by the circular buffer pass. Otherwise, those
  // nodes are added here.
  bool is_circular_buffered =
      (GpuLower::current()->tmaCircularBufferInfo().existsTensorIndex(ldst));

  if (is_circular_buffered) {
    kir::TensorIndex* mbarrier =
        GpuLower::current()->tmaCircularBufferInfo().getTensorIndex(ldst);
    Val* mbarrier_index = lower_utils::u32IndexScalarSmemTv(mbarrier);

    // gmem indexing and expect_bytes for mbarrier
    auto [in, _] = Index::getCpAsyncBulkGmemIndex(
        ldst, mbarrier_index, for_loops_, rotated_loop_);

    // indexing ldst op
    Val* out = lowerDstIndex(
        ldst->out(), /*override_index=*/{}, /*generate_pointer=*/true);
    Expr* new_ldst =
        IrBuilder::create<LoadStoreOp>(ldst->opType(), out, in, ldst->cacheOp())
            ->withPredicate(ldst->predicate());
    pushBack(new_ldst);

    // register new LoadStoreOp with mbarrier
    GpuLower::current()->tmaCircularBufferInfo().recordTensorIndex(
        new_ldst, mbarrier);

    GpuLower::current()->propagateExprInfo(ldst, back());
  } else {
    TensorView* mbarrier = GpuLower::current()->ldstMBarrierMap().at(ldst);
    Val* mbarrier_index = lower_utils::u32IndexScalarSmemTv(mbarrier);

    // gmem indexing and expect_bytes for mbarrier
    auto [in, expect_bytes] = Index::getCpAsyncBulkGmemIndex(
        ldst, mbarrier_index, for_loops_, rotated_loop_);

    // arrive and expect_tx mbarrier
    Val* state = IrBuilder::create<Val>(DataType::UInt);
    pushBack(IrBuilder::create<kir::Allocate>(
        state, MemoryType::Local, ldst->container()->oneVal()));
    pushBack(IrBuilder::create<kir::MBarrierArriveExpectTx>(
        state, mbarrier_index, expect_bytes));

    // indexing ldst op
    Val* out = lowerDstIndex(
        ldst->out(), /*override_index=*/{}, /*generate_pointer=*/true);
    Expr* new_ldst =
        IrBuilder::create<LoadStoreOp>(ldst->opType(), out, in, ldst->cacheOp())
            ->withPredicate(ldst->predicate());
    pushBack(new_ldst);

    GpuLower::current()->propagateExprInfo(ldst, back());
    // wait mbarrier
    pushBack(IrBuilder::create<kir::MBarrierWait>(mbarrier_index, state));
  }
}

void IndexLowering::handleCpAsyncBulkStore(const LoadStoreOp* ldst) {
  auto in = lowerSrcIndex(ldst->in(), ldst->out(), {}, true);
  auto [out, _] =
      Index::getCpAsyncBulkGmemIndex(ldst, nullptr, for_loops_, rotated_loop_);
  auto new_ldst =
      IrBuilder::create<LoadStoreOp>(ldst->opType(), out, in, ldst->cacheOp())
          ->withPredicate(ldst->predicate());
  pushBack(new_ldst);
  GpuLower::current()->propagateExprInfo(ldst, back());
}

static DataType getMmaInputAType(MmaMacro macro) {
  int warp_group_size = isHopper(macro) ? 128 : 32;
  int size = getM(macro) * getK(macro) / warp_group_size /
      2 /* halves per 32bit register */;
  return ArrayType{std::make_shared<DataType>(DataType::UInt32), (size_t)size};
}

static DataType getMmaInputBType(MmaMacro macro) {
  int size = getN(macro) * getK(macro) / 32 /* threads per warp */ /
      2 /* halves per 32bit register */;
  return ArrayType{std::make_shared<DataType>(DataType::UInt32), (size_t)size};
}

static inline DataType getMmaOutType(TensorView* mma_out) {
  int64_t size = 1;
  for (auto id : mma_out->getAllocationDomain()) {
    if (id->isMma() && !id->isReduction()) {
      size *= id->extent()->evaluate().as<int64_t>();
    }
  }
  return ArrayType{std::make_shared<DataType>(DataType::Float), (size_t)size};
}

namespace {
std::pair<Val*, Val*> hardCodedIndexGenerationForStMatrix(
    const LoadStoreOp* ldst,
    const int64_t output_m_extent,
    const int64_t output_n_extent) {
  NVF_ERROR(
      (output_m_extent == 8 && output_n_extent == 8) ||
          (output_m_extent == 16 && output_n_extent == 8) ||
          (output_m_extent == 16 && output_n_extent == 16),
      "size not currently supported for stmatrix");

  auto num_regs = (output_m_extent) / 8 * (output_n_extent) / 8;
  auto as_type = ArrayType{
      std::make_shared<DataType>(DataType::UInt32),
      static_cast<size_t>(num_regs)};

  Val* in = IrBuilder::create<kir::TensorIndex>(
      dynamic_cast<TensorView*>(ldst->in()),
      IrBuilder::create<Val>(0, DataType::Index),
      as_type);

  Val* out_index = nullptr;
  // This will hanlde 8x8 and 16x8.
  if (output_n_extent == 8) {
    // T_shared[toSmem(T_shared) + 16 * tidx.x]
    out_index = IrBuilder::addExpr(
        IrBuilder::baseAddressExpr(dynamic_cast<TensorView*>(ldst->out())),
        IrBuilder::mulExpr(
            IrBuilder::create<Val>(16, DataType::Index),
            IrBuilder::create<NamedScalar>("threadIdx.x", DataType::Index)));
  } else if (output_n_extent == 16) {
    // This will hanlde 16x16
    // T_shared[toSmem(T_shared) + 16 * (tidx.x / 16) +  32 * (tidx.x%16)  +

    // 16 * (tidx.x / 16)
    auto expr0 = IrBuilder::mulExpr(
        IrBuilder::create<Val>(16, DataType::Index),
        IrBuilder::divExpr(
            IrBuilder::create<NamedScalar>("threadIdx.x", DataType::Index),
            IrBuilder::create<Val>(16, DataType::Index)));

    // 32 * (tidx.x%16)
    auto expr1 = IrBuilder::mulExpr(
        IrBuilder::modExpr(
            IrBuilder::create<NamedScalar>("threadIdx.x", DataType::Index),
            IrBuilder::create<Val>(16, DataType::Index)),
        IrBuilder::create<Val>(32, DataType::Index));

    out_index = IrBuilder::addExpr(
        IrBuilder::baseAddressExpr(ir_utils::getTvOutput(ldst)),
        IrBuilder::addExpr(expr0, expr1));
  }
  Val* out = IrBuilder::create<kir::TensorIndex>(
      dynamic_cast<TensorView*>(ldst->out()), out_index);

  return {in, out};
}
} // namespace

void IndexLowering::handle(const LoadStoreOp* ldst) {
  Val* in = nullptr;
  Val* out = nullptr;
  if (ir_utils::isCpAsyncBulk(ldst)) {
    if (ir_utils::isCpAsyncBulkLoad(ldst)) {
      handleCpAsyncBulkLoad(ldst);
    } else if (ir_utils::isCpAsyncBulkStore(ldst)) {
      handleCpAsyncBulkStore(ldst);
    } else {
      NVF_THROW();
    }
  } else {
    DataType as_type = DataType::Null;
    if (ir_utils::isLdMatrixOp(ldst)) {
      as_type = ArrayType{
          std::make_shared<DataType>(DataType::UInt32),
          (size_t)ir_utils::getVectorizeSize(ldst->out()->as<TensorView>()) /
              2};
    } else if (ir_utils::isStMatrixOp(ldst)) {
      NVF_ERROR(
          ldst->out()->as<TensorView>()->getLogicalDomain().size() == 2,
          "We only support 2D inputs stmatrix");

      auto output_m_extent = ldst->out()
                                 ->as<TensorView>()
                                 ->getLogicalDomain()[0]
                                 ->extent()
                                 ->evaluate()
                                 .as<int64_t>();
      auto output_n_extent = ldst->out()
                                 ->as<TensorView>()
                                 ->getLogicalDomain()[1]
                                 ->extent()
                                 ->evaluate()
                                 .as<int64_t>();

      auto [in_idx, out_idx] = hardCodedIndexGenerationForStMatrix(
          ldst, output_m_extent, output_n_extent);
      in = in_idx;
      out = out_idx;
    } else if (ldst->out()->definition()->isA<MmaOp>()) {
      // For MMA accumulator initialization
      as_type = getMmaOutType(ldst->out()->as<TensorView>());
    }

    if (!ir_utils::isStMatrixOp(ldst)) {
      in = lowerSrcIndex(
          ldst->in(),
          ldst->out(),
          {},
          ir_utils::isLdMatrixOp(ldst) || ir_utils::isCpAsyncOp(ldst));
      out =
          lowerDstIndex(ldst->out(), {}, ir_utils::isCpAsyncOp(ldst), as_type);
    }
    auto new_ldst =
        IrBuilder::create<LoadStoreOp>(ldst->opType(), out, in, ldst->cacheOp())
            ->withPredicate(ldst->predicate());
    pushBack(new_ldst);
    GpuLower::current()->propagateExprInfo(ldst, back());
  }
}

// Reference:
// https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#asynchronous-warpgroup-level-matrix-shared-memory-layout-matrix-descriptor
static Val* matrixDescriptorEncode(Val* x) {
  auto x_cast = IrBuilder::maybeCastExpr(DataType::UInt, x);
  auto mask = IrBuilder::create<Val>(0x3FFFF, DataType::UInt);
  auto x_and = IrBuilder::bitwiseAndExpr(x_cast, mask);
  auto shift = IrBuilder::create<Val>(0x4, DataType::UInt);
  return IrBuilder::rShiftExpr(x_and, shift);
}

static Val* constructMatrixDescriptor(
    Val* start_address,
    Val* leading_dim_byte_offset,
    Val* stride_dim_byte_offset,
    Val* matrix_base_offset,
    MmaInputSmemSwizzle swizzle) {
  auto or0 = matrixDescriptorEncode(start_address);
  auto or1 = IrBuilder::lShiftExpr(
      matrixDescriptorEncode(leading_dim_byte_offset),
      IrBuilder::create<Val>(16, DataType::UInt));
  auto or2 = IrBuilder::lShiftExpr(
      matrixDescriptorEncode(stride_dim_byte_offset),
      IrBuilder::create<Val>(32, DataType::UInt));
  auto or3 = IrBuilder::lShiftExpr(
      matrix_base_offset, IrBuilder::create<Val>(49, DataType::UInt));
  auto or4 = IrBuilder::lShiftExpr(
      IrBuilder::create<Val>((int64_t)swizzle, DataType::UInt),
      IrBuilder::create<Val>(62, DataType::UInt));
  return IrBuilder::bitwiseOrExpr(
      IrBuilder::bitwiseOrExpr(
          IrBuilder::bitwiseOrExpr(IrBuilder::bitwiseOrExpr(or0, or1), or2),
          or3),
      or4);
}

static MmaInputSmemSwizzle getSwizzleMode(TensorView* tv) {
  const auto& alloc_domain = tv->getMaybeRootDomain();
  const auto& loop_domain = tv->getLoopDomain();
  auto exprs = StmtSort::getExprsBetween(
      {alloc_domain.begin(), alloc_domain.end()},
      {loop_domain.begin(), loop_domain.end()});
  auto swizzle_exprs = ir_utils::filterByType<Swizzle>(exprs);
  if (swizzle_exprs.empty()) {
    return MmaInputSmemSwizzle::None;
  }
  NVF_ERROR(
      swizzle_exprs.size() < 2,
      "expected 2 or less swizzle expressions in mma input, got ",
      swizzle_exprs.size());
  auto swizzle = *swizzle_exprs.begin();
  NVF_ERROR(swizzle->swizzleType() == SwizzleType::XOR, "expect xor swizzle");
  return getSwizzleFromBytes(
      swizzle->inX()->extent()->evaluate().as<int64_t>() * 16);
}

// Get the ValGroup of the ID in consumer's loop domain that corresponds to the
// innermost dimension in the allocation domain of tv. This ID must be
// parallelized on Mma.
ValGroup getInnerMmaLoopGroup(TensorView* tv, const MmaOp* mma) {
  ValGraph& id_graph = GpuLower::current()->tensorIndexer().traversalGraph();
  auto alloc_domain = id_graph.toGroups(
      TensorDomain::noBroadcasts(tv->getMaybeAllocationDomain()));
  auto loop_domain =
      id_graph.toGroups(mma->out()->as<TensorView>()->getLoopDomain());

  // Start from the innermost dim in the allocation domain, propagate all the
  // way to the consumer's loop domain, and keep track of the inner dimension.
  // After propagating, the inner dimension should be a dimension that is
  // parallelized on Mma.
  NVF_ERROR(
      !alloc_domain.empty(),
      "Matmul with all broadcasting dimension is not supported yet.");
  ValGroup inner = alloc_domain.back();

  auto exprs =
      ValGraphBFS::getExprsBetween(id_graph, loop_domain, alloc_domain);
  while (!exprs.empty()) {
    auto [expr, direction] = exprs.back();
    exprs.pop_back();
    auto from =
        (direction == Direction::Backward ? id_graph.inputGroups(expr)
                                          : id_graph.outputGroups(expr));
    auto to =
        (direction == Direction::Backward ? id_graph.outputGroups(expr)
                                          : id_graph.inputGroups(expr));
    bool in_from = std::find(from.begin(), from.end(), inner) != from.end();
    if (!in_from) {
      continue;
    }
    NVF_ERROR(
        from.back() == inner,
        "Expecting the innermost of the alloc domain to stay inner");
    inner = to.back();
  }
  IterDomain* inner_id = nullptr;
  for (auto id : mma->out()->as<TensorView>()->getLoopDomain()) {
    if (inner->has(id)) {
      inner_id = id;
      break;
    }
  }
  NVF_ERROR(
      inner_id != nullptr,
      "Could not find innermost ID in the loop domain of mma output");
  NVF_ERROR(
      inner_id->getParallelType() == ParallelType::Mma,
      "Expecting the innermost ID to be parallelized on Mma");
  return inner;
}

// Compute the "leading_bytes" in the matrix descriptor of Mma. The leading
// bytes is the stride of the innermost dimension in the allocation domain of tv
// considering core matrices. For example, if the tv is [M, K], where K is the
// inner, then the schedule of the loop domain of the mma output must be
// something like:
//      M            K
//      |            |
//     ...          ...
//      |            |
//     / \.         / \.
//   ...  m_inst  ...  k_inst
// where m_inst and k_inst are the instruction tiles of M and K, respectively,
// that is, the number of items each TensorCore instruction can execute. Both
// m_inst and k_inst must be parallelized on Mma. The leading_bytes is the
// stride of the outer (k_inst/swizzle_size) in the allocation domain of tv.
// That is, if we futher split k_inst as:
//      M            K
//      |            |
//     ...          ...
//      |            |
//     / \.         / \.
//   ...  m_inst  ...  k_inst
//                     /    \.
//               linear      swizzle_size
// Then we would need to prove that `linear` is linear in the allocation domain
// of tv, and the stride of `linear` is the leading_bytes. This function does
// the following things:
// 1. Find k_inst.
// 2. Split k_inst as above.
// 3. Prove that `linear` is linear in the allocation domain of tv, and get the
//    stride of `linear`.
Val* getInnerStrideBytes(TensorView* tv, const MmaOp* mma) {
  auto swizzle = getSwizzleMode(tv);
  auto swizzle_size = getBytesFromSwizzle(swizzle) / dataTypeSize(tv->dtype());
  ValGraph& id_graph = GpuLower::current()->tensorIndexer().traversalGraph();
  auto alloc_domain = id_graph.toGroups(tv->getMaybeAllocationDomain());
  auto inner = getInnerMmaLoopGroup(tv, mma);
  // At this point, we can just create the following schedule:
  //        inner
  //       /     \.
  //   linear   swizzle_size
  // and use proveLinearAndGetStride to find the stride of `linear` in the
  // allocation domain.
  auto outer_of_tiling = split(&id_graph, inner, swizzle_size).first;
  auto stride = lower_utils::proveLinearAndGetStride(
      id_graph, outer_of_tiling, alloc_domain);
  NVF_ERROR(stride != nullptr, "Could not get the stride of tiling");
  return SimplifyingIrBuilder::mulExpr(stride, dataTypeSize(tv->dtype()));
}

// Compute the "stride_bytes" in the matrix descriptor of Mma. The stride
// bytes is the stride of the outer dimension in the allocation domain of tv
// considering core matrices. For example, if the tv is [M, K], where K is the
// inner, then the schedule of the loop domain of the mma output must be
// something like:
//      M            K
//      |            |
//     ...          ...
//      |            |
//     / \.         / \.
//   ...  m_inst  ...  k_inst
// where m_inst and k_inst are the instruction tiles of M and K, respectively,
// that is, the number of items each TensorCore instruction can execute. Both
// m_inst and k_inst must be parallelized on Mma. The stride_bytes is the
// stride of the outer (m_inst/8) in the allocation domain of tv.
// That is, if we futher split m_inst as:
//       M            K
//       |            |
//      ...          ...
//       |            |
//      / \.         / \.
//    ...  m_inst  ...  k_inst
//         /    \.
//   linear      8
// Then we would need to prove that `linear` is linear in the allocation domain
// of tv, and the stride of `linear` is the stride_bytes. This function does
// the following things:
// 1. Find m_inst.
// 2. Split m_inst as above.
// 3. Prove that `linear` is linear in the allocation domain of tv, and get the
//    stride of `linear`.
Val* getOuterStrideBytes(TensorView* tv, const MmaOp* mma) {
  ValGraph& id_graph = GpuLower::current()->tensorIndexer().traversalGraph();
  auto logical_domain =
      id_graph.toGroups(TensorDomain::noBroadcasts(tv->getLogicalDomain()));
  auto loop_domain =
      id_graph.toGroups(mma->out()->as<TensorView>()->getLoopDomain());
  auto alloc_domain = id_graph.toGroups(tv->getMaybeAllocationDomain());

  // In the consumer's loop domain, there should be exactly 3 IDs parallelized
  // on Mma. Which of these three dims are M, N, and K? We don't know. But we
  // don't really care. What we do care is, which is the inner? which is the
  // broadcast? We would find the loop dim that is neither inner nor broadcast,
  // and the stride of that dim is the one we are looking for.
  auto inner = getInnerMmaLoopGroup(tv, mma);
  std::vector<ValGroup> mma_groups;
  mma_groups.reserve(2);
  for (auto id : mma->out()->as<TensorView>()->getLoopDomain()) {
    if (id->getParallelType() == ParallelType::Mma && !inner->has(id)) {
      mma_groups.emplace_back(id_graph.toGroup(id));
    }
  }
  NVF_ERROR(
      mma_groups.size() == 2,
      "Expecting 3 IDs in the loop domain of mma output to be parallelized on Mma,",
      " among which one must be the innermost of producer's allocation domain");

  // Get which group in mma_groups is projected to a concrete ID in the logical
  // domain of tv. There should be exactly one such group.
  auto is_projected_to_concrete = [&](const ValGroup& g) {
    auto projection_on_logical =
        ValGraphBFS::projectTo(id_graph, {g}, logical_domain);
    for (auto id : tv->getLogicalDomain()) {
      if (!id->isBroadcast() &&
          projection_on_logical.count(id_graph.toGroup(id))) {
        return true;
      }
    }
    return false;
  };
  ValGroup selected = nullptr;
  for (auto& g : mma_groups) {
    if (is_projected_to_concrete(g)) {
      NVF_ERROR(
          selected == nullptr,
          "Expecting exactly one group in mma output loop domain to be projected to a concrete ID in the logical domain of tv");
      selected = std::move(g);
    }
  }
  NVF_ERROR(
      selected != nullptr,
      "No group in mma output loop domain is projected to a concrete ID in the logical domain of tv");

  // At this point, we can just create the following schedule:
  //      selected
  //       /     \.
  //   linear     8
  // and use proveLinearAndGetStride to find the stride of `linear` in the
  // allocation domain.
  constexpr int64_t core_matrix_outer_size = 8;
  auto outer_of_tiling =
      split(&id_graph, selected, core_matrix_outer_size).first;
  auto stride = lower_utils::proveLinearAndGetStride(
      id_graph, outer_of_tiling, alloc_domain);
  NVF_ERROR(stride != nullptr, "Could not get the stride of tiling");
  return SimplifyingIrBuilder::mulExpr(stride, dataTypeSize(tv->dtype()));
}

// Reference for smem strides:
// https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#strides
void IndexLowering::handle(const MmaOp* mma) {
  Val* a = nullptr;
  Val* b = nullptr;
  const auto& [unitdim_a, unitdim_b] = lower_utils::getMmaLayout(mma);
  if (mma->inA()->as<TensorView>()->getMemoryType() == MemoryType::Shared) {
    // TODO: This is a temporary solution and only supports a single tile in
    // smem.
    auto tv = mma->inA()->as<TensorView>();
    auto swizzle = getSwizzleMode(tv);
    // Because the entire tile is parallelized on MMA, which are trivial
    // loops and always have zero loop variables, the result of lowerSrcIndex
    // will be the address of the first element of the tile, which happens to
    // be the information we need to provide to the hardware.
    auto base_addr = lowerSrcIndex(tv, mma->out(), {}, true)
                         ->as<kir::TensorIndex>()
                         ->index();
    Val* leading_bytes = getInnerStrideBytes(tv, mma);
    Val* stride_bytes = getOuterStrideBytes(tv, mma);
    if (swizzle == MmaInputSmemSwizzle::None && unitdim_a == UnitDim::M_or_N) {
      // tnspA and tnspB is ignored for NoSwizzle mode
      std::swap(leading_bytes, stride_bytes);
    }
    auto matrix_desc = constructMatrixDescriptor(
        base_addr,
        leading_bytes,
        stride_bytes,
        IrBuilder::create<Val>(0, DataType::UInt),
        getSwizzleMode(tv));
    a = IrBuilder::create<kir::TensorIndex>(
        tv,
        GpuLower::current()->commonScalarMap().hoistScalar(
            matrix_desc, for_loops_));
  } else {
    a = lowerSrcIndex(
        mma->inA(), mma->out(), {}, false, getMmaInputAType(mma->macro()));
  }
  if (mma->inB()->as<TensorView>()->getMemoryType() == MemoryType::Shared) {
    // TODO: This is a temporary solution and only supports a single tile in
    // smem.
    auto tv = mma->inB()->as<TensorView>();
    auto swizzle = getSwizzleMode(tv);
    // Because the entire tile is parallelized on MMA, which are trivial
    // loops and always have zero loop variables, the result of lowerSrcIndex
    // will be the address of the first element of the tile, which happens to
    // be the information we need to provide to the hardware.
    auto base_addr = lowerSrcIndex(tv, mma->out(), {}, true)
                         ->as<kir::TensorIndex>()
                         ->index();
    Val* leading_bytes = getInnerStrideBytes(tv, mma);
    Val* stride_bytes = getOuterStrideBytes(tv, mma);
    if (swizzle == MmaInputSmemSwizzle::None && unitdim_b == UnitDim::M_or_N) {
      // tnspA and tnspB is ignored for NoSwizzle mode
      std::swap(leading_bytes, stride_bytes);
    }
    auto matrix_desc = constructMatrixDescriptor(
        base_addr,
        leading_bytes,
        stride_bytes,
        IrBuilder::create<Val>(0, DataType::UInt),
        swizzle);
    b = IrBuilder::create<kir::TensorIndex>(
        tv,
        GpuLower::current()->commonScalarMap().hoistScalar(
            matrix_desc, for_loops_));
  } else {
    b = lowerSrcIndex(
        mma->inB(), mma->out(), {}, false, getMmaInputBType(mma->macro()));
  }
  const auto out = lowerDstIndex(
      mma->out(), {}, false, getMmaOutType(mma->out()->as<TensorView>()));
  auto mma_indexed =
      IrBuilder::create<MmaOp>(out, a, b, mma->init(), mma->macro());
  pushBack(mma_indexed);
  GpuLower::current()->propagateExprInfo(mma, back());
}

void IndexLowering::handle(const BroadcastOp* bop) {
  NVF_ERROR(ir_utils::isTvOp(bop));

  const auto out_tv = bop->out()->as<TensorView>();

  const auto out = lowerDstIndex(bop->out());
  const auto in = lowerSrcIndex(bop->in(), bop->out());
  auto indexed_expr =
      IrBuilder::create<BroadcastOp>(out, in, bop->getBroadcastDimFlags());

  const ParallelTypeBitmap parallel_bitmap =
      GpuLower::current()->threadPredMap().getParallelBroadcastDomains(out_tv);

  const bool block_x = parallel_bitmap.get(ParallelType::BIDx);
  const bool block_y = parallel_bitmap.get(ParallelType::BIDy);
  const bool block_z = parallel_bitmap.get(ParallelType::BIDz);

  if (bop->predicate()) {
    indexed_expr =
        indexed_expr->withPredicate(bop->predicate())->as<BroadcastOp>();
  }

  const bool grid_broadcast_needed = block_x || block_y || block_z;
  if (!grid_broadcast_needed) {
    pushBack(indexed_expr);
    GpuLower::current()->propagateExprInfo(bop, back());
    return;
  }

  // Grid broadcast
  const auto out_domain = out_tv->domain();
  const auto work_buffer_size =
      getGridCommWorkBufferSize(out_domain, for_loops_, true)
          .size_of_privatized_buffer;

  auto work_buffer = allocateUniqueBuffer(
      work_buffer_size, out->dtype(), false, out_tv, work_buffer_map_);

  auto sync_buffer_size = getGridSyncBufferSize(out_domain, for_loops_, true);
  auto sync_buffer = allocateUniqueBuffer(
      sync_buffer_size, DataType::Int, true, out_tv, sync_buffer_map_);

  auto grid_broadcast = IrBuilder::create<kir::GridBroadcast>(
      indexed_expr, work_buffer, sync_buffer);

  if (bop->predicate()) {
    grid_broadcast = grid_broadcast->withPredicate(bop->predicate())
                         ->as<kir::GridBroadcast>();
  }

  pushBack(grid_broadcast);
  GpuLower::current()->propagateExprInfo(bop, back());
}

void IndexLowering::handle(const kir::Asm* asm_) {
  // TODO(kir): remove the need for const_cast
  pushBack(const_cast<kir::Asm*>(asm_)); // NOLINT
}

void IndexLowering::handle(const kir::Allocate* allocate) {
  // TODO(kir): remove the need for const_cast
  pushBack(const_cast<kir::Allocate*>(allocate)); // NOLINT
}

void IndexLowering::handle(const kir::BlockSync* sync) {
  // TODO(kir): remove the need for const_cast
  pushBack(const_cast<kir::BlockSync*>(sync)); // NOLINT
}

void IndexLowering::handle(const kir::GridSync* sync) {
  // TODO(kir): remove the need for const_cast
  pushBack(const_cast<kir::GridSync*>(sync)); // NOLINT
}

void IndexLowering::handle(const kir::AsyncWait* wait) {
  // TODO(kir): remove the need for const_cast
  pushBack(const_cast<kir::AsyncWait*>(wait)); // NOLINT
}

void IndexLowering::handle(const kir::FenceAsyncProxy* fence) {
  // TODO(kir): remove the need for const_cast
  pushBack(const_cast<kir::FenceAsyncProxy*>(fence)); // NOLINT
}

void IndexLowering::handle(const kir::AsyncCommit* commit) {
  // TODO(kir): remove the need for const_cast
  pushBack(const_cast<kir::AsyncCommit*>(commit)); // NOLINT
}

void IndexLowering::handle(const kir::BlockSerializeWait* sync) {
  // TODO(kir): remove the need for const_cast
  pushBack(const_cast<kir::BlockSerializeWait*>(sync)); // NOLINT
}

void IndexLowering::handle(const kir::BlockSerializeRelease* sync) {
  // TODO(kir): remove the need for const_cast
  pushBack(const_cast<kir::BlockSerializeRelease*>(sync)); // NOLINT
}

void IndexLowering::generate(const std::vector<Expr*>& exprs) {
  for (auto expr : exprs) {
    OptOutConstDispatch::dispatch(expr);
  }
}

kir::Allocate* IndexLowering::allocateUniqueBuffer(
    Val* buffer_size,
    DataType dtype,
    bool zero_init,
    TensorView* out_tv,
    std::unordered_map<TensorView*, kir::Allocate*>& alloc_map) {
  // Return an existing allocation if exists
  auto it = alloc_map.find(out_tv);
  if (it != alloc_map.end()) {
    return it->second;
  }

  // No existing allocation found. Create a new one
  auto new_buffer =
      lower_utils::allocGlobalBufferForGridComm(buffer_size, dtype, zero_init);

  // Keep track of the allocation
  alloc_map.emplace(out_tv, new_buffer);

  // A buffer may be used in both the unswitched paths, so it must be
  // placed outside of the current scope. Simplying placing it at the
  // top-level scope should work.
  insertAtTopLevel(new_buffer);

  return new_buffer;
}

void IndexLowering::allocateUniqueFusedReduction(
    Expr* expr,
    TensorView* out_tv) {
  auto it = fused_reduction_map_.find(out_tv);
  if (it != fused_reduction_map_.end()) {
    return;
  }

  kir::AllocateFusedReduction* fused_reduction_alloc_reduction = nullptr;
  if (expr->isStrictlyA<kir::GridReduction>()) {
    fused_reduction_alloc_reduction =
        IrBuilder::create<kir::AllocateFusedReduction>(
            expr->as<kir::GridReduction>());
  } else if (expr->isStrictlyA<kir::GridWelford>()) {
    fused_reduction_alloc_reduction =
        IrBuilder::create<kir::AllocateFusedReduction>(
            expr->as<kir::GridWelford>());
  } else if (expr->isStrictlyA<kir::GroupedGridReduction>()) {
    fused_reduction_alloc_reduction =
        IrBuilder::create<kir::AllocateFusedReduction>(
            expr->as<kir::GroupedGridReduction>());
  } else if (expr->isStrictlyA<kir::GroupedGridWelford>()) {
    fused_reduction_alloc_reduction =
        IrBuilder::create<kir::AllocateFusedReduction>(
            expr->as<kir::GroupedGridWelford>());
  } else {
    NVF_THROW("Invalid expr: ", expr->toString());
  }

  fused_reduction_map_.emplace(out_tv, fused_reduction_alloc_reduction);

  // When using the fused reduction, allocate the reduction object at
  // the outer-most scope
  insertAtTopLevel(fused_reduction_alloc_reduction);
}

void IndexLowering::handle(const PadOp* pad) {
  // Convert to a where op as:
  // consumer[consumer_idx] = (consumer_idx >= left_pad && consumer_idx <
  //                           consumer_extent - right_pad) ?
  //     producer[producer_idx] :
  //     0;

  auto producer_tv = pad->in()->as<TensorView>();
  auto consumer_tv = pad->out()->as<TensorView>();
  auto producer_doms =
      TensorDomain::noReductions(producer_tv->getLogicalDomain());

  const auto in = lowerSrcIndex(pad->in(), pad->out());
  const auto out = lowerDstIndex(pad->out());

  const auto pad_val = pad->value();

  // Build a predicate for where
  auto consumer_root_indices = Index::getConsumerPerDimLogicalIndex(
      consumer_tv, for_loops_, getRotatedLoop());
  Val* pred = consumer_tv->fusion()->trueVal();
  for (auto padded_axis : pad->getPaddedAxes()) {
    auto consumer_idx = consumer_root_indices.at(padded_axis);
    auto consumer_root_id = consumer_tv->getLogicalDomain().at(padded_axis);
    NVF_ERROR(!consumer_root_id->maybePartial());
    const auto& pad_widths = pad->getPadWidths(padded_axis);
    pred = SimplifyingIrBuilder::logicalAndExpr(
        pred,
        // idx >= left_pad && idx < extent - right_pad
        SimplifyingIrBuilder::logicalAndExpr(
            SimplifyingIrBuilder::geExpr(consumer_idx, pad_widths.first),
            SimplifyingIrBuilder::ltExpr(
                consumer_idx,
                SimplifyingIrBuilder::subExpr(
                    consumer_root_id->getMaybeExpandedExtent(),
                    pad_widths.second))));
  }

  pred = GpuLower::current()->commonScalarMap().hoistScalar(pred, for_loops_);

  pushBack(IrBuilder::create<TernaryOp>(
      TernaryOpType::Where, out, pred, in, pad_val));
  GpuLower::current()->propagateExprInfo(pad, back());
}

void IndexLowering::handle(const SliceOp* slice) {
  // TODO: Consider converting SliceOp to Set at the beginning of
  // lowering
  const auto in = lowerSrcIndex(slice->in(), slice->out());
  const auto out = lowerDstIndex(slice->out());

  pushBack(IrBuilder::create<LoadStoreOp>(LoadStoreOpType::Set, out, in));
  GpuLower::current()->propagateExprInfo(slice, back());
}

void IndexLowering::handle(const CatOp* cat) {
  // It's possible to lower CatOp to a series of IfThenElse or Where,
  // but that would going to look really ugly. For now, rely on
  // CudaKernelGenerator to produce code based on the predicates
  // genereated here.

  const auto out = lowerDstIndex(cat->output(0));
  auto out_indices = Index::getConsumerPerDimLogicalIndex(
      cat->output(0)->as<TensorView>(), for_loops_, getRotatedLoop());
  auto concatenated_dim_idx = out_indices.at(cat->concatenatedDim());

  std::vector<Val*> inputs(cat->inputs().size());
  std::vector<Val*> preds(cat->inputs().size());
  Val* cur_extent = GpuLower::current()->kernel()->zeroVal();

  for (const auto i : c10::irange(cat->inputs().size())) {
    const auto inp = lowerSrcIndex(cat->input(i), cat->output(0));
    inputs.at(i) = inp;

    // Note the original extent is the extent of the root domain not
    // logical domain
    auto inp_concat_id = TensorDomain::noReductions(
                             cat->input(i)->as<TensorView>()->getRootDomain())
                             .at(cat->concatenatedDim());
    cur_extent = add(cur_extent, inp_concat_id->getMaybeExpandedExtent());
    preds.at(i) = IrBuilder::ltExpr(concatenated_dim_idx, cur_extent);
  }

  auto lowered = IrBuilder::create<CatOp>(
      out, inputs, cat->concatenatedDim(), concatenated_dim_idx, preds);

  pushBack(lowered);
  GpuLower::current()->propagateExprInfo(cat, lowered);
}

} // namespace nvfuser
