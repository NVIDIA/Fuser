// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#include <csrc/exceptions.h>
#include <gmock/gmock-matchers.h>
#include <gtest/gtest.h>

#include <tests/cpp/utils.h>
#include <tests/cpp/validator.h>

#include <fusion.h>
#include <id_model/id_model.h>
#include <id_model/indexing.h>
#include <id_model/indexing_utils.h>
#include <id_model/to_string.h>
#include <id_model/utils.h>
#include <ir/builder.h>
#include <kernel_ir_dispatch.h>
#include <ops/all_ops.h>
#include <scheduler/tools/abstract_tensor.h>
#include <scheduler/tools/inlining.h>
#include <scheduler/tools/loop_domain_scheduler.h>
#include <scheduler/tools/resize_utils.h>
#include <scheduler/utils.h>

#include <string.h>
#include <algorithm>
#include <utility>

namespace nvfuser {

using IndexingTest = NVFuserTest;
using PredicateIndexingTest = NVFuserFixtureParamTest<bool>;
using ContigIndexingTest = NVFuserTest;
using ContigPredicateIndexingTest = NVFuserTest;

namespace {

std::vector<Val*> getLoopIndices(
    TensorView* tv,
    const TensorIndexer& indexer,
    const std::vector<ForLoop*>& for_loops) {
  std::vector<Val*> loop_indices;
  for (const auto& loop_id : tv->getLoopDomain()) {
    loop_indices.push_back(indexer.getLoopIndex(loop_id, for_loops));
  }
  return loop_indices;
}

std::vector<IterDomain*> getLoopDomains(
    TensorView* tv,
    const IdModel& id_model) {
  std::vector<IterDomain*> loop_domains;
  for (auto loop_id : tv->getLoopDomain()) {
    loop_domains.push_back(getLoopPromotion(loop_id, id_model));
  }

  return loop_domains;
}

template <typename... Args>
Val* addExpr(Args&&... args) {
  return SimplifyingIrBuilder::addExpr(std::forward<Args>(args)...);
}

template <typename... Args>
Val* subExpr(Args&&... args) {
  return SimplifyingIrBuilder::subExpr(std::forward<Args>(args)...);
}

template <typename... Args>
Val* mulExpr(Args&&... args) {
  return SimplifyingIrBuilder::mulExpr(std::forward<Args>(args)...);
}

template <typename... Args>
Val* divExpr(Args&&... args) {
  return SimplifyingIrBuilder::divExpr(std::forward<Args>(args)...);
}

template <typename... Args>
Val* ceilDivExpr(Args&&... args) {
  return SimplifyingIrBuilder::ceilDivExpr(std::forward<Args>(args)...);
}

template <typename... Args>
Val* modExpr(Args&&... args) {
  return SimplifyingIrBuilder::modExpr(std::forward<Args>(args)...);
}

template <typename... Args>
Val* xorExpr(Args&&... args) {
  return IrBuilder::bitwiseXorExpr(std::forward<Args>(args)...);
}

template <typename... Args>
Val* andExpr(Args&&... args) {
  return SimplifyingIrBuilder::logicalAndExpr(std::forward<Args>(args)...);
}

template <typename... Args>
Val* geExpr(Args&&... args) {
  return SimplifyingIrBuilder::geExpr(std::forward<Args>(args)...);
}

template <typename... Args>
Val* ltExpr(Args&&... args) {
  return SimplifyingIrBuilder::ltExpr(std::forward<Args>(args)...);
}

Val* createInt(int64_t i) {
  return IrBuilder::create<Val>(i, DataType::Index);
}

// Predicates should be a composite val that should look like (((x &&
// y) && z). To make it easier to read error messages, decompose the
// composite predicate to each component and print each in a separate
// line.
std::string prettyPrintPredicate(Val* pred) {
  std::deque<Val*> pred_list;

  while (true) {
    NVF_ERROR(pred != nullptr);
    if (auto bop = dynamic_cast<BinaryOp*>(pred->definition());
        bop != nullptr && bop->getBinaryOpType() == BinaryOpType::LogicalAnd) {
      pred_list.push_front(bop->input(1));
      pred = bop->input(0);
    } else {
      pred_list.push_front(pred);
      break;
    }
  }

  std::stringstream ss;
  for (auto each_pred : pred_list) {
    ss << each_pred->toInlineString() << "\n";
  }

  return ss.str();
}

// AbstractGetReference and IndexValidator are used to validate
// lowered index vals. Each test subclasses either or both of
// getLinearIndex and getLinearIndexString of
// AbstractGetReference. IndexValidator traverses lowered exprs to
// validate each tensor indices.
class AbstractGetReference {
 public:
  AbstractGetReference(const TensorIndexer& indexer, const IdModel& id_model)
      : indexer_(indexer), id_model_(id_model) {}
  virtual ~AbstractGetReference() = default;

  // Returns the index of a given tensor. If maybe_consumer is not
  // nullptr, tv is indexed as a consumer. Otherwise, it's indexed as
  // a producer of maybe_consumer
  virtual Val* getLinearIndex(TensorView* tv, TensorView* maybe_consumer)
      const {
    return nullptr;
  }

  virtual std::string getLinearIndexString(
      TensorView* tv,
      TensorView* maybe_consumer) const {
    return std::string();
  }

  // Returns the inline predicate of a given tensor.
  virtual Val* getInlinePredicate(TensorView* tv) const {
    return nullptr;
  }

  // Returns the outer predicate of a given tensor. This only matters
  // when tv is unswitched or unrolled. Note that if it's vectorized,
  // the predicate is still inlined.
  virtual Val* getOuterPredicate(TensorView* tv) const {
    return nullptr;
  }

  void setForLoops(const std::vector<ForLoop*>& for_loops) {
    for_loops_ = for_loops;
  }

  void clearForLoops() {
    for_loops_.clear();
  }

  void clearCircularBufferInfo() {
    circular_buffer_loop_stage_ = CircularBufferLoopStage::NotApplicable;
  }

  void setCircularBufferInfo(CircularBufferLoopStage loop_stage) {
    circular_buffer_loop_stage_ = loop_stage;
  }

 protected:
  const TensorIndexer& indexer_;
  const IdModel& id_model_;
  // These could be getLinearIndex parameters, but it's just easier to
  // add them here since the function signature doesn't need to change.
  std::vector<ForLoop*> for_loops_;
  CircularBufferLoopStage circular_buffer_loop_stage_ =
      CircularBufferLoopStage::NotApplicable;
};

template <typename GetReference>
class IndexValidator : public kir::IrVisitor {
 public:
  IndexValidator(const GpuLower& lower, GetReference&& get_ref)
      : get_ref_(std::move(get_ref)) {}

  using kir::IrVisitor::dispatch;
  using kir::IrVisitor::handle;

  void dispatch(Expr* expr) override {
    if (expr->isA<kir::Asm>()) {
      kir::Asm* asm_expr = expr->as<kir::Asm>();
      const char* ldmatrix = R"(ldmatrix)";
      bool ldmatrix_match =
          strstr(asm_expr->utility().c_str(), ldmatrix) != nullptr;

      if (!ldmatrix_match) {
        kir::IrVisitor::dispatch(expr);
        return;
      }

      get_ref_.setForLoops(for_loops_);

      auto out_ti = expr->output(0)->as<kir::TensorIndex>();
      for (auto inp : expr->inputs()) {
        if (inp->isA<kir::TensorIndex>()) {
          validate(inp->as<kir::TensorIndex>(), out_ti);
        }
      }
      for (auto out : expr->outputs()) {
        if (out->isA<kir::TensorIndex>()) {
          validate(out->as<kir::TensorIndex>());
        }
      }

      get_ref_.clearForLoops();
      get_ref_.clearCircularBufferInfo();

      return;
    }

    if (!ir_utils::isTvOp(expr)) {
      kir::IrVisitor::dispatch(expr);
      return;
    }

    get_ref_.setForLoops(for_loops_);

    if (auto loop_it = std::find_if(
            for_loops_.begin(),
            for_loops_.end(),
            [](ForLoop* fl) {
              return fl->circularBufferLoopStage() !=
                  CircularBufferLoopStage::NotApplicable;
            });
        loop_it != for_loops_.end()) {
      auto loop = *loop_it;
      get_ref_.setCircularBufferInfo(loop->circularBufferLoopStage());
    }

    auto out_ti = expr->output(0)->as<kir::TensorIndex>();
    for (auto inp : expr->inputs()) {
      if (inp->isA<kir::TensorIndex>()) {
        validate(inp->as<kir::TensorIndex>(), out_ti);
      }
    }
    for (auto out : expr->outputs()) {
      if (out->isA<kir::TensorIndex>()) {
        validate(out->as<kir::TensorIndex>());
      }
    }

    get_ref_.clearForLoops();
    get_ref_.clearCircularBufferInfo();
  }

  void validate(kir::TensorIndex* ti, kir::TensorIndex* out_ti = nullptr) {
    TensorView* tv = ti->view();
    TensorView* maybe_consumer = out_ti != nullptr ? out_ti->view() : nullptr;
    Val* actual = ti->index();
    Val* ref = get_ref_.getLinearIndex(tv, maybe_consumer);
    if (ref != nullptr) {
      EXPECT_TRUE(actual->sameAs(ref))
          << "Validation failure of " << ti->view()->toString() << " as "
          << (out_ti != nullptr ? "producer" : "consumer")
          << "\nRef: " << ref->toInlineString()
          << "\nActual: " << actual->toInlineString();
      return;
    }

    // If nullptr is returned, check if a string ref is available
    std::string ref_str = get_ref_.getLinearIndexString(tv, maybe_consumer);
    if (!ref_str.empty()) {
      EXPECT_EQ(actual->toInlineString(), ref_str)
          << "Validation failure of " << ti->view()->toString() << " as "
          << (out_ti != nullptr ? "producer" : "consumer")
          << "\nRef: " << ref_str << "\nActual: " << actual->toInlineString();
      return;
    }

    // If no ref is obtained, skip validation
  }

  template <typename... Args>
  static void validate(
      Fusion* fusion,
      bool enable_contig_indexing,
      Args... args) {
    EnableOptionsGuard enable_options_guard;
    EnableOptionsGuard::getCurOptions().set(
        EnableOption::IdModel, {"consumer_index", "producer_index"});

    // Disable simplifications to make the pattern matching of sameAs work
    DisableOptionsGuard disable_options_guard;
    DisableOptionsGuard::getCurOptions().set(DisableOption::ExprSimplify);
    DisableOptionsGuard::getCurOptions().set(DisableOption::IndexHoist);
    // Magic zero is not yet supported
    DisableOptionsGuard::getCurOptions().set(DisableOption::MagicZero);
    if (!enable_contig_indexing) {
      DisableOptionsGuard::getCurOptions().set(DisableOption::ContigIndexing);
    }

    GpuLower lower(fusion);

    kir::Kernel* kernel = nullptr;
    // Suppress warnings due to using dynamic register tensors
    testing::internal::CaptureStderr();
    kernel = lower.run();
    testing::internal::GetCapturedStderr();

    IndexValidator<GetReference> validator(
        lower, GetReference(lower.tensorIndexer(), lower.idModel(), args...));

    FusionGuard fg(kernel);
    validator.handle(kernel->topLevelExprs());
  }

 private:
  GetReference get_ref_;
};

template <typename GetReference>
class PredicateIndexValidator : public kir::IrVisitor {
 public:
  PredicateIndexValidator(const GpuLower& lower, GetReference&& get_ref)
      : get_ref_(std::move(get_ref)) {}

  using kir::IrVisitor::dispatch;
  using kir::IrVisitor::handle;

  void dispatch(Expr* expr) override {
    if (!ir_utils::isTvOp(expr)) {
      kir::IrVisitor::dispatch(expr);
      return;
    }

    get_ref_.setForLoops(for_loops_);

    CircularBufferLoopStage loop_stage = CircularBufferLoopStage::NotApplicable;

    if (auto loop_it = std::find_if(
            for_loops_.begin(),
            for_loops_.end(),
            [](ForLoop* fl) {
              return fl->circularBufferLoopStage() !=
                  CircularBufferLoopStage::NotApplicable;
            });
        loop_it != for_loops_.end()) {
      auto loop = *loop_it;
      loop_stage = loop->circularBufferLoopStage();
      get_ref_.setCircularBufferInfo(loop_stage);
    }

    auto out_ti = expr->output(0)->as<kir::TensorIndex>();

    // This is just an initialization expr, likely by zero. Only the
    // actual expr will be validted.
    if (out_ti->view()->definition()->input(0)->isA<TensorView>() &&
        expr->input(0)->isScalar()) {
      return;
    }

    NVF_ERROR(!scope_exprs_.empty());
    auto inline_ite = dynamic_cast<kir::IfThenElse*>(scope_exprs_.back());
    if (inline_ite != nullptr) {
      validateInlinePredicate(out_ti, inline_ite->predicate()->value());
    }

    // If there's an other IfThenElse in the scope stack, validate the
    // predicate as well. The predicate should be for unswitch/unroll
    // loops. Only the innermost one is considered.
    for (auto it = scope_exprs_.rbegin(); it != scope_exprs_.rend(); ++it) {
      auto ite = dynamic_cast<kir::IfThenElse*>(*it);
      if (ite == nullptr) {
        continue;
      }
      if (ite == inline_ite) {
        continue;
      }

      validateOuterPredicate(out_ti, ite->predicate()->value(), loop_stage);
      break;
    }

    get_ref_.clearForLoops();
    get_ref_.clearCircularBufferInfo();
  }

  void validateInlinePredicate(kir::TensorIndex* ti, Val* actual) {
    TensorView* tv = ti->view();
    Val* ref = get_ref_.getInlinePredicate(tv);
    if (ref != nullptr) {
      EXPECT_TRUE(actual->sameAs(ref))
          << "Validation failure of inline predicate for "
          << ti->view()->toString() << "\nRef: " << ref->toInlineString()
          << "\nActual: " << actual->toInlineString();
      return;
    }

    // If no ref is obtained, skip validation
  }

  void validateOuterPredicate(
      kir::TensorIndex* ti,
      Val* actual,
      CircularBufferLoopStage loop_stage) {
    TensorView* tv = ti->view();
    Val* ref = get_ref_.getOuterPredicate(tv);
    if (ref != nullptr) {
      std::stringstream loop_stage_msg;
      if (loop_stage != CircularBufferLoopStage::NotApplicable) {
        loop_stage_msg << " in " << loop_stage;
      }
      std::stringstream actual_str;
      EXPECT_TRUE(actual->sameAs(ref))
          << "Validation failure of outer predicate for "
          << ti->view()->toString() << loop_stage_msg.str() << "\nRef:\n"
          << prettyPrintPredicate(ref) << "Actual:\n"
          << prettyPrintPredicate(actual);
      return;
    }

    // If no ref is obtained, skip validation
  }

  template <typename... Args>
  static void validate(
      Fusion* fusion,
      bool enable_contig_indexing,
      Args... args) {
    EnableOptionsGuard enable_options_guard;
    EnableOptionsGuard::getCurOptions().set(EnableOption::IdModel, {"all"});

    // Disable simplifications to make the pattern matching of sameAs work
    DisableOptionsGuard disable_options_guard;
    DisableOptionsGuard::getCurOptions().set(DisableOption::ExprSimplify);
    DisableOptionsGuard::getCurOptions().set(DisableOption::IndexHoist);
    // Magic zero is not yet supported
    DisableOptionsGuard::getCurOptions().set(DisableOption::MagicZero);
    DisableOptionsGuard::getCurOptions().set(
        DisableOption::PredicateElimination);
    if (!enable_contig_indexing) {
      DisableOptionsGuard::getCurOptions().set(DisableOption::ContigIndexing);
    }

    GpuLower lower(fusion);

    kir::Kernel* kernel = nullptr;
    // Suppress warnings due to using dynamic register tensors
    testing::internal::CaptureStderr();
    kernel = lower.run();
    testing::internal::GetCapturedStderr();

    PredicateIndexValidator<GetReference> validator(
        lower, GetReference(lower.tensorIndexer(), lower.idModel(), args...));

    FusionGuard fg(kernel);
    validator.handle(kernel->topLevelExprs());
  }

 private:
  GetReference get_ref_;
};

} // namespace

// Simple pointwise test with no parallelization
TEST_F(IndexingTest, SimplePointwise1) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto tv0 = makeSymbolicTensor(2);
  fusion.addInput(tv0);

  auto tv1 = add(tv0, IrBuilder::create<Val>(1.0));
  auto tv2 = add(tv1, IrBuilder::create<Val>(1.0));
  fusion.addOutput(tv2);

  tv2->flatten();
  tv2->split(0, 4);

  TransformPropagator propagator(tv2);
  MaxLogicalDomainInfoSpanningTree(tv2).traverse(&propagator);

  tv1->inlineAt(1);

  struct GetReference : AbstractGetReference {
    GetReference(const TensorIndexer& indexer, const IdModel& id_model)
        : AbstractGetReference(indexer, id_model) {}

    Val* getLinearIndex(TensorView* tv, TensorView* maybe_consumer)
        const override {
      bool as_consumer = maybe_consumer == nullptr;
      auto consumer_tv = as_consumer ? tv : maybe_consumer;
      std::vector<Val*> loop_indices =
          getLoopIndices(consumer_tv, indexer_, for_loops_);
      switch (tv->name()) {
        case 0: {
          NVF_ERROR(!as_consumer);
          return addExpr(
              mulExpr(
                  divExpr(
                      addExpr(
                          mulExpr(
                              consumer_tv->axis(1)->extent(),
                              loop_indices.at(0)),
                          loop_indices.at(1)),
                      tv->getLogicalDomain().at(1)->extent()),
                  IrBuilder::getItemExpr(
                      IrBuilder::getAttrExpr(
                          IrBuilder::metadataExpr(tv), "alloc_stride"),
                      IrBuilder::create<Val>(0))),
              mulExpr(
                  modExpr(
                      addExpr(
                          mulExpr(
                              consumer_tv->axis(1)->extent(),
                              loop_indices.at(0)),
                          loop_indices.at(1)),
                      tv->getLogicalDomain().at(1)->extent()),
                  IrBuilder::getItemExpr(
                      IrBuilder::getAttrExpr(
                          IrBuilder::metadataExpr(tv), "alloc_stride"),
                      IrBuilder::create<Val>(1))));
        }
        case 1: {
          return loop_indices.at(1);
        }
        case 2: {
          NVF_ERROR(as_consumer);
          return addExpr(
              mulExpr(
                  divExpr(
                      addExpr(
                          mulExpr(loop_indices.at(0), tv->axis(1)->extent()),
                          loop_indices.at(1)),
                      tv->getLogicalDomain().at(1)->extent()),
                  tv->getLogicalDomain().at(1)->extent()),
              modExpr(
                  addExpr(
                      mulExpr(loop_indices.at(0), tv->axis(1)->extent()),
                      loop_indices.at(1)),
                  tv->getLogicalDomain().at(1)->extent()));
        }
        default:
          NVF_THROW("Unexpected tensor: ", tv->toString());
          break;
      }
      return nullptr;
    }
  };

  IndexValidator<GetReference>::validate(&fusion, false);
}

// Almost same fusion as SimplePointwiseSerial but TID and BID
// parallelizaiton with no inlining
TEST_F(IndexingTest, SimplePointwise2) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto tv0 = makeContigTensor(2);
  fusion.addInput(tv0);

  auto tv1 = add(tv0, IrBuilder::create<Val>(1.0));
  auto tv2 = add(tv1, IrBuilder::create<Val>(1.0));
  auto tv3 = add(tv2, IrBuilder::create<Val>(1.0));
  fusion.addOutput(tv3);

  tv3->flatten();
  tv3->split(0, 4);

  TransformPropagator propagator(tv3);
  MaxLogicalDomainInfoSpanningTree(tv3).traverse(&propagator);

  tv3->axis(0)->parallelize(ParallelType::BIDx);
  tv3->axis(1)->parallelize(ParallelType::TIDx);

  scheduler_utils::parallelizeAllLike(tv3, fusion.allTvs());

  // Test shared memory indexing
  tv2->setMemoryType(MemoryType::Shared);

  // tv0 and tv3 are global tensors and should have the same index:
  // "(blockIdx.x * 4 + threadIdx.x) % tv0->axis(1)->extent() +
  // (blockIdx.x * 4 + threadIdx.x) / tv0->axis(1)->extent() *
  // tv0->axis(1)->extent()
  //
  // tv1 is a Local tensor. Since it's fully parallelized, its index
  // should be always zero
  //
  // tv2 is a Shared tensor. Only the TIDx parallelized domain should
  // contribute to the index

  struct GetReference : AbstractGetReference {
    GetReference(const TensorIndexer& indexer, const IdModel& id_model)
        : AbstractGetReference(indexer, id_model) {}

    Val* getLinearIndex(TensorView* tv, TensorView* maybe_consumer)
        const override {
      bool as_consumer = maybe_consumer == nullptr;
      auto contig_idx = SimplifyingIrBuilder::addExpr(
          SimplifyingIrBuilder::mulExpr(
              NamedScalar::getParallelIndex(ParallelType::BIDx),
              tv->axis(1)->extent()),
          NamedScalar::getParallelIndex(ParallelType::TIDx));

      auto global_ref = SimplifyingIrBuilder::addExpr(
          SimplifyingIrBuilder::mulExpr(
              SimplifyingIrBuilder::divExpr(
                  contig_idx, tv->getLogicalDomain().at(1)->extent()),
              tv->getLogicalDomain().at(1)->extent()),
          SimplifyingIrBuilder::modExpr(
              contig_idx, tv->getLogicalDomain().at(1)->extent()));

      auto shared_ref = NamedScalar::getParallelIndex(ParallelType::TIDx);

      switch (tv->name()) {
        case 0: {
          NVF_ERROR(!as_consumer);
          return global_ref;
        }
        case 1: {
          return tv->fusion()->zeroVal();
        }
        case 2: {
          return shared_ref;
        }
        case 3: {
          NVF_ERROR(as_consumer);
          return global_ref;
        }
        default:
          NVF_THROW("Unexpected tensor: ", tv->toString());
          break;
      }
    }
  };

  IndexValidator<GetReference>::validate(&fusion, false);
}

// Simple reduction with no parallelization
TEST_F(IndexingTest, SimpleReduction) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto tv0 = makeContigTensor(2);
  fusion.addInput(tv0);

  auto tv1 = sum(tv0, {1});
  auto tv2 = add(tv1, IrBuilder::create<Val>(1.0));
  fusion.addOutput(tv2);

  struct GetReference : AbstractGetReference {
    GetReference(const TensorIndexer& indexer, const IdModel& id_model)
        : AbstractGetReference(indexer, id_model) {}

    Val* getLinearIndex(TensorView* tv, TensorView* maybe_consumer)
        const override {
      bool as_consumer = maybe_consumer == nullptr;
      auto consumer_tv = as_consumer ? tv : maybe_consumer;
      std::vector<Val*> loop_indices =
          getLoopIndices(consumer_tv, indexer_, for_loops_);

      switch (tv->name()) {
        case 0: {
          NVF_ERROR(!as_consumer);
          return addExpr(
              mulExpr(
                  loop_indices.at(0), tv->getLogicalDomain().at(1)->extent()),
              loop_indices.at(1));
        }
        case 1: {
          return loop_indices.at(0);
        }
        case 2: {
          NVF_ERROR(as_consumer);
          return loop_indices.at(0);
        }
        default:
          NVF_THROW("Unexpected tensor: ", tv->toString());
          // gcc v11.4 requires this return statement
          return nullptr;
      }
    }
  };

  IndexValidator<GetReference>::validate(&fusion, false);
}

// Reduction with inlining. Loop promotion picks a reduction domain,
// which indexing should not ignore.
TEST_F(IndexingTest, PromotionToReductionDomain) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto tv0 = makeContigTensor(2);
  fusion.addInput(tv0);

  auto tv1 = add(tv0, IrBuilder::create<Val>(1.0));
  auto tv2 = sum(tv1, {1});
  fusion.addOutput(tv2);

  tv1->setMemoryType(MemoryType::Shared);

  inlineMost();

  tv1->axis(1)->parallelize(ParallelType::TIDx);
  tv2->axis(1)->parallelize(ParallelType::TIDx);

  // tv1's index should be "threadIdx.x". However, since its
  // allocation domain, tv1->axis(1), is promoted to tv2->axis(1),
  // which is a reduction domain, the initial version of indexing
  // mistakenly excluded the domain from indexing.

  struct GetReference : AbstractGetReference {
    GetReference(const TensorIndexer& indexer, const IdModel& id_model)
        : AbstractGetReference(indexer, id_model) {}

    Val* getLinearIndex(TensorView* tv, TensorView* maybe_consumer)
        const override {
      bool as_consumer = maybe_consumer == nullptr;
      auto consumer_tv = as_consumer ? tv : maybe_consumer;
      std::vector<Val*> loop_indices =
          getLoopIndices(consumer_tv, indexer_, for_loops_);

      switch (tv->name()) {
        case 1: {
          return loop_indices.at(1);
        }
        default:
          return nullptr;
      }
    }
  };

  IndexValidator<GetReference>::validate(&fusion, false);
}

// Fusion copied from AllocationDomainTest.TransposedIntermediate
TEST_F(IndexingTest, AllocationDomain) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto tv0 = makeContigConcreteTensor({32, 32});
  fusion.addInput(tv0);
  auto tv1 = set(tv0);
  auto tv2 = set(tv1);
  fusion.addOutput(tv2);
  tv1->setMemoryType(MemoryType::Shared);

  tv1->axis(0)->parallelize(ParallelType::TIDx);
  tv2->axis(0)->parallelize(ParallelType::TIDx);

  std::vector<IterDomain*> tv1_transposed = {tv1->axis(1), tv1->axis(0)};
  tv1->setAllocationDomain(tv1_transposed, true);

  struct GetReference : AbstractGetReference {
    GetReference(const TensorIndexer& indexer, const IdModel& id_model)
        : AbstractGetReference(indexer, id_model) {}

    Val* getLinearIndex(TensorView* tv, TensorView* maybe_consumer)
        const override {
      bool as_consumer = maybe_consumer == nullptr;
      auto consumer_tv = as_consumer ? tv : maybe_consumer;
      std::vector<Val*> loop_indices =
          getLoopIndices(consumer_tv, indexer_, for_loops_);

      switch (tv->name()) {
        case 1: {
          return addExpr(
              mulExpr(
                  tv->getLogicalDomain().at(0)->extent(), loop_indices.at(1)),
              loop_indices.at(0));
        }
        default:
          // Only validates tv1
          return nullptr;
      }
    }
  };

  IndexValidator<GetReference>::validate(&fusion, false);
}

TEST_F(IndexingTest, Reshape) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  const std::vector<int64_t> shape1({100});
  const std::vector<int64_t> shape2({4, 25});
  const std::vector<int64_t> shape3({5, 2, 10});

  // [i0]
  auto tv0 = makeContigConcreteTensor(shape1);
  fusion.addInput(tv0);

  auto tv1 = set(tv0);

  // [i2, i3]
  auto tv2 = reshape(tv1, shape1, shape2);

  // [i2, i3]
  auto tv3 = add(tv2, fusion.oneVal());

  // [i4, i5, i6]
  auto tv4 = reshape(tv3, shape2, shape3);

  // [i4, i5, i6]
  auto tv5 = add(tv4, fusion.oneVal());

  fusion.addOutput(tv5);

  TransformPropagator propagator(tv5);
  MaxLogicalDomainInfoSpanningTree(tv5).traverse(&propagator);

  inlineMost();

  struct GetReference : AbstractGetReference {
    GetReference(const TensorIndexer& indexer, const IdModel& id_model)
        : AbstractGetReference(indexer, id_model) {}

    Val* getLinearIndex(TensorView* tv, TensorView* maybe_consumer)
        const override {
      bool as_consumer = maybe_consumer == nullptr;
      auto consumer_tv = as_consumer ? tv : maybe_consumer;
      std::vector<Val*> loop_indices =
          getLoopIndices(consumer_tv, indexer_, for_loops_);

      switch (tv->name()) {
        case 1:
        case 2:
        case 3:
        case 4: {
          // All intermediate tensors should be fully inlined, so their
          // indices should be just zero.
          return tv->fusion()->zeroVal();
        }
        case 5: {
          // tv5 has no transformation and is fully contiguous
          NVF_ERROR(as_consumer);
          return addExpr(
              addExpr(
                  mulExpr(
                      mulExpr(
                          tv->getLogicalDomain().at(1)->extent(),
                          tv->getLogicalDomain().at(2)->extent()),
                      loop_indices.at(0)),
                  mulExpr(
                      loop_indices.at(1),
                      tv->getLogicalDomain().at(2)->extent())),
              loop_indices.at(2));
        }
        default:
          return nullptr;
      }
    }

    std::string getLinearIndexString(TensorView* tv, TensorView* maybe_consumer)
        const override {
      switch (tv->name()) {
        case 0: {
          // It isn't straightforward to do structural checking as the other
          // tests since there's no particular rule about which domain is used
          // to provide the extent of the group. However, since everything
          // should be deterministic, string match should also work.
          return std::string(
              "( ( ( ( ( i98 * 20 ) + ( ( i99 * 10 ) + i100 ) ) / 25 ) * 25 ) + ( ( ( i98 * 20 ) + ( ( i99 * 10 ) + i100 ) ) % 25 ) )");
        }
        default:
          return std::string();
      }
    }
  };

  IndexValidator<GetReference>::validate(&fusion, false);
}

// Simple non-concretized broadcast
TEST_F(IndexingTest, SimpleBroadcast1) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto tv0 = makeContigTensor(1);
  fusion.addInput(tv0);

  auto tv1 = broadcast(tv0, {false, true});
  auto tv2 = add(tv1, IrBuilder::create<Val>(1.0));
  fusion.addOutput(tv2);

  struct GetReference : AbstractGetReference {
    GetReference(const TensorIndexer& indexer, const IdModel& id_model)
        : AbstractGetReference(indexer, id_model) {}

    Val* getLinearIndex(TensorView* tv, TensorView* maybe_consumer)
        const override {
      bool as_consumer = maybe_consumer == nullptr;
      auto consumer_tv = as_consumer ? tv : maybe_consumer;
      std::vector<Val*> loop_indices =
          getLoopIndices(consumer_tv, indexer_, for_loops_);
      NVF_ERROR(loop_indices.at(1)->isZeroInt());
      switch (tv->name()) {
        case 0:
        case 1:
        case 2: {
          return loop_indices.at(0);
        }
        default:
          NVF_THROW();
      }
    }
  };

  IndexValidator<GetReference>::validate(&fusion, false);
}

// SimpleBroadcast1 + scheduling
TEST_F(IndexingTest, SimpleBroadcast2) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto tv0 = makeContigTensor(1);
  fusion.addInput(tv0);

  auto tv1 = broadcast(tv0, {false, true});
  auto tv2 = add(tv1, IrBuilder::create<Val>(1.0));
  fusion.addOutput(tv2);

  tv2->flatten();
  tv2->split(0, 4);

  TransformPropagator propagator(tv2);
  MaxLogicalDomainInfoSpanningTree(tv2).traverse(&propagator);

  // The first merge of the logical domains should be a trivial merge,
  // i.e., a merge with a extent-one domain. Thus, the indexing
  // traversal should return "x + y * 4", where x and y are the loop
  // indices, respectively.

  struct GetReference : AbstractGetReference {
    GetReference(const TensorIndexer& indexer, const IdModel& id_model)
        : AbstractGetReference(indexer, id_model) {}

    // tv0 is a global memory tensor, so the indexing is done with its
    // allocation domain, which is mapped with the merge of the two
    // logical domains of tv1 on the AlmostExact graph. Traverse back to
    // the merge output from the loop domains.
    //
    // tv1 is a Local tensor, so its allocation domains are just their
    // loop domains. This index is mathematically equivalent to the tv0
    // index, but the order of linearizing the two loop domains is
    // different from the order of computing the merge input index.
    //
    // In the end, the indices of all tensors are the same
    Val* getLinearIndex(TensorView* tv, TensorView* maybe_consumer)
        const override {
      bool as_consumer = maybe_consumer == nullptr;
      auto consumer_tv = as_consumer ? tv : maybe_consumer;
      std::vector<Val*> loop_indices =
          getLoopIndices(consumer_tv, indexer_, for_loops_);
      switch (tv->name()) {
        case 0:
        case 1:
        case 2: {
          return addExpr(
              mulExpr(loop_indices.at(0), tv->axis(1)->extent()),
              loop_indices.at(1));
        }
        default:
          NVF_THROW();
      }
    }
  };

  IndexValidator<GetReference>::validate(&fusion, false);
}

// Concretized broadcast
TEST_F(IndexingTest, SimpleBroadcast3) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto tv0 = makeContigTensor(1);
  fusion.addInput(tv0);
  auto tv1 = makeContigTensor(2);
  fusion.addInput(tv1);

  auto tv2 = broadcast(tv0, {false, true});
  auto tv3 = add(tv2, tv1);
  fusion.addOutput(tv3);

  tv3->flatten();

  TransformPropagator propagator(tv3);
  MaxLogicalDomainInfoSpanningTree(tv3).traverse(&propagator);

  inlineMost();

  struct GetReference : AbstractGetReference {
    GetReference(const TensorIndexer& indexer, const IdModel& id_model)
        : AbstractGetReference(indexer, id_model) {}

    Val* getLinearIndex(TensorView* tv, TensorView* maybe_consumer)
        const override {
      bool as_consumer = maybe_consumer == nullptr;
      auto consumer_tv = as_consumer ? tv : maybe_consumer;
      std::vector<Val*> loop_indices =
          getLoopIndices(consumer_tv, indexer_, for_loops_);
      switch (tv->name()) {
        case 0: {
          // tv0 is a 1D pre-broadcast input tensor, so it only needs the
          // index that corresponds to the outer dimension of the tv3 (or tv2)
          // logical domains
          auto tv3 = consumer_tv->uses().at(0)->output(0)->as<TensorView>();
          return divExpr(
              loop_indices.at(0), tv3->getLogicalDomain().at(1)->extent());
        }
        case 1:
        case 3: {
          return addExpr(
              mulExpr(
                  divExpr(
                      loop_indices.at(0),
                      tv->getLogicalDomain().at(1)->extent()),
                  tv->getLogicalDomain().at(1)->extent()),
              modExpr(
                  loop_indices.at(0), tv->getLogicalDomain().at(1)->extent()));
        }
        case 2: {
          // tv2 is fully inlined, so its index should be just zero
          return tv->fusion()->zeroVal();
        }
        default:
          NVF_THROW();
      }
    }
  };

  IndexValidator<GetReference>::validate(&fusion, false);
}

// Concretized broadcast with partial inlining. Loop promotion is
// required. Same fusion as IdModelTest.LoopPromotion4. See also
// Example 1 of the Loop Promotion doc.
TEST_F(IndexingTest, SimpleBroadcast4) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto tv0 = makeContigConcreteTensor({1, -1});
  fusion.addInput(tv0);
  auto tv1 = makeContigConcreteTensor({-1, -1});
  fusion.addInput(tv1);

  auto tv2 = set(tv0);
  auto tv3 = set(tv1);
  auto tv4 = add(tv2, tv3);
  fusion.addOutput(tv4);

  // [i0, i1]
  tv4->merge(0);
  // [i0*i1]
  tv4->split(0, 4, false); // outer split
  // [4, i0*i1/4]

  TransformPropagator propagator(tv4);
  MaxLogicalDomainInfoSpanningTree(tv4).traverse(&propagator);

  for (auto tv : fusion.allTvs()) {
    tv->inlineAt(-2);
  }

  // As discussed in the doc, the inner domain of tv2 is promoted to
  // a domain with the same extent as the inner domain of tv4. Since
  // tv2 is a Local tensor, its allocation domain is also promoted to
  // the same domain. Thus, its consumer index is just the loop index
  // of the inner loop of the tv2 loop domains, and its producer index
  // is also just the inner loop index of the loop domains of tv4.

  struct GetReference : AbstractGetReference {
    GetReference(const TensorIndexer& indexer, const IdModel& id_model)
        : AbstractGetReference(indexer, id_model) {}

    Val* getLinearIndex(TensorView* tv, TensorView* maybe_consumer)
        const override {
      bool as_consumer = maybe_consumer == nullptr;
      auto consumer_tv = as_consumer ? tv : maybe_consumer;
      std::vector<Val*> loop_indices =
          getLoopIndices(consumer_tv, indexer_, for_loops_);
      switch (tv->name()) {
        case 2: {
          return loop_indices.at(1);
        }
        default:
          return nullptr;
      }
    }
  };

  IndexValidator<GetReference>::validate(&fusion, false);
}

// Trivial example. 1D shared tensor. Each device only has one
// element, so the index should be always just zero.
TEST_F(IndexingTest, MultiDevice1DNoSplitMerge) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto tv0 = makeSymbolicTensor(1);
  fusion.addInput(tv0);

  auto tv1 = set(tv0);
  fusion.addOutput(tv1);

  tv0->axis(0)->parallelize(ParallelType::DIDx);
  tv1->axis(0)->parallelize(ParallelType::DIDx);

  struct GetReference : AbstractGetReference {
    GetReference(const TensorIndexer& indexer, const IdModel& id_model)
        : AbstractGetReference(indexer, id_model) {}

    Val* getLinearIndex(TensorView* tv, TensorView* maybe_consumer)
        const override {
      return tv->fusion()->zeroVal();
    }
  };

  IndexValidator<GetReference>::validate(&fusion, false);
}

// Same fusion as MultiDevice1DNoSplitMerge but with split.
TEST_F(IndexingTest, MultiDevice1DSplit) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto tv0 = makeContigTensor(1);
  fusion.addInput(tv0);

  auto tv1 = set(tv0);
  fusion.addOutput(tv1);

  Val* num_devices = IrBuilder::create<Val>(DataType::Index);

  tv0->split(0, num_devices, false);
  tv1->split(0, num_devices, false);

  tv0->axis(0)->parallelize(ParallelType::DIDx);
  tv1->axis(0)->parallelize(ParallelType::DIDx);

  struct GetReference : AbstractGetReference {
    GetReference(const TensorIndexer& indexer, const IdModel& id_model)
        : AbstractGetReference(indexer, id_model) {}

    Val* getLinearIndex(TensorView* tv, TensorView* maybe_consumer)
        const override {
      bool as_consumer = maybe_consumer == nullptr;
      auto consumer_tv = as_consumer ? tv : maybe_consumer;
      std::vector<Val*> loop_indices =
          getLoopIndices(consumer_tv, indexer_, for_loops_);
      return loop_indices.at(1);
    }
  };

  IndexValidator<GetReference>::validate(&fusion, false);
}

TEST_F(IndexingTest, MultiDevice2D) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto tv0 = makeContigTensor(2);
  fusion.addInput(tv0);

  auto tv1 = set(tv0);
  fusion.addOutput(tv1);

  Val* num_devices = IrBuilder::create<Val>(DataType::Index);

  tv1->flatten();
  tv1->split(0, num_devices, false);

  TransformPropagator propagator(tv1);
  MaxLogicalDomainInfoSpanningTree(tv1).traverse(&propagator);

  tv0->axis(0)->parallelize(ParallelType::DIDx);
  tv1->axis(0)->parallelize(ParallelType::DIDx);

  struct GetReference : AbstractGetReference {
    GetReference(const TensorIndexer& indexer, const IdModel& id_model)
        : AbstractGetReference(indexer, id_model) {}

    Val* getLinearIndex(TensorView* tv, TensorView* maybe_consumer)
        const override {
      bool as_consumer = maybe_consumer == nullptr;
      auto consumer_tv = as_consumer ? tv : maybe_consumer;
      std::vector<Val*> loop_indices =
          getLoopIndices(consumer_tv, indexer_, for_loops_);
      // Note that the allocation domain is the logical domain. See the
      // next test for a loop allocation example
      auto inner_dim = tv->getLogicalDomain().at(1)->extent();
      return addExpr(
          mulExpr(divExpr(loop_indices.at(1), inner_dim), inner_dim),
          modExpr(loop_indices.at(1), inner_dim));
    }
  };

  IndexValidator<GetReference>::validate(&fusion, false);
}

// Same fusion as MultiDevice2D but with loop allocation
TEST_F(IndexingTest, MultiDevice2DLeafAllocation) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto tv0 = makeContigTensor(2);
  fusion.addInput(tv0);

  auto tv1 = set(tv0);
  fusion.addOutput(tv1);

  Val* num_devices = IrBuilder::create<Val>(DataType::Index);

  tv1->flatten();
  tv1->split(0, num_devices, false);

  TransformPropagator propagator(tv1);
  MaxLogicalDomainInfoSpanningTree(tv1).traverse(&propagator);

  tv0->axis(0)->parallelize(ParallelType::DIDx);
  tv1->axis(0)->parallelize(ParallelType::DIDx);

  tv0->setAllocationDomain(tv0->getLoopDomain(), true);
  tv1->setAllocationDomain(tv1->getLoopDomain(), true);

  struct GetReference : AbstractGetReference {
    GetReference(const TensorIndexer& indexer, const IdModel& id_model)
        : AbstractGetReference(indexer, id_model) {}

    Val* getLinearIndex(TensorView* tv, TensorView* maybe_consumer)
        const override {
      bool as_consumer = maybe_consumer == nullptr;
      auto consumer_tv = as_consumer ? tv : maybe_consumer;
      std::vector<Val*> loop_indices =
          getLoopIndices(consumer_tv, indexer_, for_loops_);
      // Since the loop domain is the allocation domain, the index should
      // be just the non-parallelized loop index
      return loop_indices.at(1);
    }
  };

  IndexValidator<GetReference>::validate(&fusion, false);
}

TEST_F(IndexingTest, MultiDevice2DTranspose) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto tv0 = makeContigTensor(2);
  fusion.addInput(tv0);

  auto tv1 = transpose(tv0);
  fusion.addOutput(tv1);

  Val* num_devices = IrBuilder::create<Val>(DataType::Index);

  tv0->split(0, num_devices, false);
  tv1->split(0, num_devices, false);

  tv0->axis(0)->parallelize(ParallelType::DIDx);
  tv1->axis(0)->parallelize(ParallelType::DIDx);

  struct GetReference : AbstractGetReference {
    GetReference(const TensorIndexer& indexer, const IdModel& id_model)
        : AbstractGetReference(indexer, id_model) {}

    Val* getLinearIndex(TensorView* tv, TensorView* maybe_consumer)
        const override {
      bool as_consumer = maybe_consumer == nullptr;
      auto consumer_tv = as_consumer ? tv : maybe_consumer;
      std::vector<Val*> loop_indices =
          getLoopIndices(consumer_tv, indexer_, for_loops_);
      switch (tv->name()) {
        case 0: {
          return addExpr(
              mulExpr(
                  loop_indices.at(2), tv->getLogicalDomain().at(1)->extent()),
              loop_indices.at(1));
        }
        case 1: {
          return addExpr(
              mulExpr(
                  loop_indices.at(1), tv->getLogicalDomain().at(1)->extent()),
              loop_indices.at(2));
        }
        default:
          NVF_THROW();
      }
    }
  };

  IndexValidator<GetReference>::validate(&fusion, false);
}

// Allocation of broadcast domains should not need to be promoted.
TEST_F(IndexingTest, PromotedBroadcast) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto tv0 = makeContigTensor(1);
  fusion.addInput(tv0);
  auto tv1 = makeContigTensor(2);
  fusion.addInput(tv1);

  auto tv2 = broadcast(tv0, {true, false});
  auto tv3 = add(tv2, tv1);
  fusion.addOutput(tv3);

  // Note that tv2->inlineAt(1) just results in no inlining since
  // tv2->axis(0) is a broadcast domain.
  tv2->inlineAt(2);
  tv2->setMemoryType(MemoryType::Shared);
  tv3->axis(0)->parallelize(ParallelType::TIDx);

  // tv2->axis(0) is a broadcast domain promoted to
  // tv3->axis(0). While it's promoted, since it's a broadcast domain,
  // it doesn't need to be allocated. Since the inner domain is also
  // inlined, the resulting index of tv2 should just be zero.

  struct GetReference : AbstractGetReference {
    GetReference(const TensorIndexer& indexer, const IdModel& id_model)
        : AbstractGetReference(indexer, id_model) {}

    Val* getLinearIndex(TensorView* tv, TensorView* maybe_consumer)
        const override {
      bool as_consumer = maybe_consumer == nullptr;
      auto consumer_tv = as_consumer ? tv : maybe_consumer;
      std::vector<Val*> loop_indices =
          getLoopIndices(consumer_tv, indexer_, for_loops_);
      switch (tv->name()) {
        case 2:
          return tv->fusion()->zeroVal();
        default:
          return nullptr;
      }
    }
  };

  IndexValidator<GetReference>::validate(&fusion, false);
}

// Simple vectorized copy
TEST_F(IndexingTest, SimpleVectorize) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto tv0 = makeContigTensor(1);
  fusion.addInput(tv0);

  auto tv1 = set(tv0);
  auto tv2 = set(tv1);
  fusion.addOutput(tv2);

  tv2->split(0, 4);
  tv2->split(0, 128);

  tv2->axis(0)->parallelize(ParallelType::BIDx);
  tv2->axis(1)->parallelize(ParallelType::TIDx);
  tv2->axis(2)->parallelize(ParallelType::Vectorize);

  TransformPropagator propagator(tv2);
  MaxLogicalDomainInfoSpanningTree(tv2).traverse(&propagator);

  inlineMost();

  scheduler_utils::parallelizeAllLike(tv2, fusion.allTvs());

  struct GetReference : AbstractGetReference {
    GetReference(const TensorIndexer& indexer, const IdModel& id_model)
        : AbstractGetReference(indexer, id_model) {}

    Val* getLinearIndex(TensorView* tv, TensorView* maybe_consumer)
        const override {
      bool as_consumer = maybe_consumer == nullptr;
      auto consumer_tv = as_consumer ? tv : maybe_consumer;
      std::vector<Val*> loop_indices =
          getLoopIndices(consumer_tv, indexer_, for_loops_);
      switch (tv->name()) {
        case 0:
        case 2:
          // Vectorized index is generated by replacing the index of a
          // vectorized domain with zero, which doesn't go through the
          // simplification of SimplifyingIrBuilder. We could use simplifyExpr,
          // but for the sake of testing, just use IrBuilder::addExpr.
          return mulExpr(
              addExpr(
                  mulExpr(loop_indices.at(0), tv->axis(1)->extent()),
                  loop_indices.at(1)),
              tv->axis(2)->extent());
        case 1:
          return tv->fusion()->zeroVal();
        default:
          return nullptr;
      }
    }
  };

  IndexValidator<GetReference>::validate(&fusion, false);
}

// Test for reorderAllocationDomains. The vectorized
// domain must be at the innermost position in the allocation domain
// but that's not always the case within the loop domain. We might
// want to consider if we should just change the scheduler such that
// vectorized domains always be placed at the innermost position, but
// this test is to make sure the new indexer can reproduce what the
// old indexing does.
TEST_F(IndexingTest, NonInnermostVectorize) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto tv0 = makeContigTensor(1);
  fusion.addInput(tv0);
  auto tv1 = set(tv0);
  // For vectorized store
  auto tv2 = set(tv1);
  auto tv3 = set(tv2);
  fusion.addOutput(tv3);

  // Vectorize
  tv3->split(0, 4);
  // Serial
  tv3->split(0, 2);
  // TIDx
  tv3->split(0, 128);

  tv3->reorder({{-1, -2}});

  TransformPropagator propagator(tv3);
  MaxLogicalDomainInfoSpanningTree(tv3).traverse(&propagator);

  tv3->axis(0)->parallelize(ParallelType::BIDx);
  tv3->axis(1)->parallelize(ParallelType::TIDx);
  scheduler_utils::parallelizeAllLike(tv3, fusion.allTvs());

  tv1->axis(2)->parallelize(ParallelType::Vectorize);
  tv3->axis(2)->parallelize(ParallelType::Vectorize);

  struct GetReference : AbstractGetReference {
    GetReference(const TensorIndexer& indexer, const IdModel& id_model)
        : AbstractGetReference(indexer, id_model) {}

    Val* getLinearIndex(TensorView* tv, TensorView* maybe_consumer)
        const override {
      bool as_consumer = maybe_consumer == nullptr;
      auto consumer_tv = as_consumer ? tv : maybe_consumer;
      std::vector<Val*> loop_indices =
          getLoopIndices(consumer_tv, indexer_, for_loops_);
      bool vec_load_or_store =
          ((tv->name() == 1 && as_consumer) ||
           (tv->name() == 2 && !as_consumer));

      // Validate tv1 as vector load output and tv2 as vector store input
      switch (tv->name()) {
        case 1:
        case 2:
          if (vec_load_or_store) {
            return mulExpr(loop_indices.at(3), tv->axis(2)->extent());
          } else {
            return addExpr(
                mulExpr(loop_indices.at(3), tv->axis(2)->extent()),
                loop_indices.at(2));
          }
        default:
          return nullptr;
      }
    }
  };

  IndexValidator<GetReference>::validate(&fusion, false);
}

// Indexing traversal failure repro due to non-size-one broadcast
// domains. See issue #2393 as well.
TEST_F(IndexingTest, AlmostExactTraversalWithNonOneBroadcast) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  // [w]
  auto tv0 = makeSymbolicTensor(1);
  fusion.addInput(tv0);

  // [w, x]
  auto tv1 = makeSymbolicTensor(2);
  fusion.addInput(tv1);

  auto tv2 = broadcast(tv0, {false, true});
  auto tv3 = add(tv1, tv2);
  fusion.addOutput(tv3);

  tv3->split(0, 3);
  tv3->split(2, 4);
  tv3->merge(1);
  tv3->split(1, 5);

  MaxLogicalDomainInfoSpanningTree tree(tv3);
  TransformPropagator tp(tv3);
  tree.traverse(&tp);

  inlineAllAt(tv3, 1, true);

  struct GetReference : AbstractGetReference {
    GetReference(const TensorIndexer& indexer, const IdModel& id_model)
        : AbstractGetReference(indexer, id_model) {}

    Val* getLinearIndex(TensorView* tv, TensorView* maybe_consumer)
        const override {
      bool as_consumer = maybe_consumer == nullptr;
      auto consumer_tv = as_consumer ? tv : maybe_consumer;
      // Make sure tv2 as the producer is correctly indexed for
      // tv3. Skip validation for any other case
      if (tv->name() != 2 || as_consumer) {
        return nullptr;
      }

      // T2_l_float[ iS14{( ceilDiv(i0, 3) )}, iS19{1}, iS20{5}, bS17{4} ]
      // ca_pos( 1 ) logical domain : (iS3{i0}, bS4{1}) contiguity: t n
      //  Split: iS3{i0} by factor 3 -> iS14{( ceilDiv(i0, 3) )}, iS15{3}
      //  Split: bS4{1} by factor 4 -> bS16{1}, bS17{4}
      //  Merge: iS15{3} and bS16{1} -> iS18{3}
      //  Split: iS18{3} by factor 5 -> iS19{1}, iS20{5}
      // loop domain : (iS14{( ceilDiv(i0, 3) )}, iS19{1}, iS20{5}, bS17{4})

      // T3_g_float[ iS7{( ceilDiv(i2, 3) )}, iS12{( ceilDiv(( ( ceilDiv(i3, 4)
      // ) * 3 ), 5) )}, iS13{5}, iS10{4} ] ca_pos( 1 ) produce_pos( 1 )
      //  logical domain : (iS5{i2}, iS6{i3})
      //  contiguity: t t
      //   Split: iS5{i2} by factor 3 -> iS7{( ceilDiv(i2, 3) )}, iS8{3}
      //   Split: iS6{i3} by factor 4 -> iS9{( ceilDiv(i3, 4) )}, iS10{4}
      //   Merge: iS8{3} and iS9{( ceilDiv(i3, 4) )} -> iS11{( ( ceilDiv(i3, 4)
      //   ) * 3 )} Split: iS11{( ( ceilDiv(i3, 4) ) * 3 )} by factor 5 ->
      //   iS12{( ceilDiv(( ( ceilDiv(i3, 4) ) * 3 ), 5) )}, iS13{5}
      //  loop domain : (iS7{( ceilDiv(i2, 3) )}, iS12{( ceilDiv(( ( ceilDiv(i3,
      //  4) ) * 3 ), 5) )}, iS13{5}, iS10{4})

      // iS20 is the only ID to index.
      // T3's iS8 is mapped with id15. The AlmostExact graph maps iS15
      // with iS18 but not iS20 since the extent of iS18 is different
      // from that of iS20.
      std::vector<Val*> loop_indices =
          getLoopIndices(consumer_tv, indexer_, for_loops_);
      TensorView* tv2 = tv;
      TensorView* tv3 = consumer_tv;
      IterDomain* id11 = tv3->axis(1)->definition()->input(0)->as<IterDomain>();
      IterDomain* id9 = id11->definition()->input(1)->as<IterDomain>();
      Val* id11_idx = addExpr(
          mulExpr(loop_indices.at(1), tv3->axis(2)->extent()),
          loop_indices.at(2));
      Val* id8_idx = divExpr(id11_idx, id9->extent());
      IterDomain* id20 = tv2->axis(2);
      Val* id20_idx = modExpr(id8_idx, id20->extent());
      return id20_idx;
    }
  };

  IndexValidator<GetReference>::validate(&fusion, false);
}

TEST_F(IndexingTest, Swizzle) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto tv0 = makeContigConcreteTensor({32, 32});
  fusion.addInput(tv0);
  auto tv1 = set(tv0);
  auto tv2 = set(tv1);
  fusion.addOutput(tv2);
  tv1->setMemoryType(MemoryType::Shared);

  tv1->axis(0)->parallelize(ParallelType::TIDx);
  tv2->axis(0)->parallelize(ParallelType::TIDx);

  AbstractTensor alloc(tv1->getLogicalDomain());
  alloc.swizzle(SwizzleType::XOR, 0, 1);
  tv1->setAllocationDomain(alloc.as<IterDomain*>(), true);

  struct GetReference : AbstractGetReference {
    GetReference(const TensorIndexer& indexer, const IdModel& id_model)
        : AbstractGetReference(indexer, id_model) {}

    Val* getLinearIndex(TensorView* tv, TensorView* maybe_consumer)
        const override {
      bool as_consumer = maybe_consumer == nullptr;
      auto consumer_tv = as_consumer ? tv : maybe_consumer;
      std::vector<Val*> loop_indices =
          getLoopIndices(consumer_tv, indexer_, for_loops_);

      switch (tv->name()) {
        case 1: {
          return addExpr(
              mulExpr(
                  tv->getLogicalDomain().at(1)->extent(), loop_indices.at(0)),
              xorExpr(loop_indices.at(0), loop_indices.at(1)));
        }
        default:
          // Only validates tv1
          return nullptr;
      }
    }
  };

  IndexValidator<GetReference>::validate(&fusion, false);
}

// Simple Unroll test. Unlike Unswitch, Unroll moves up allocation
// points and thus index needs to be adjusted
TEST_F(IndexingTest, SimpleUnroll) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto tv0 = makeContigTensor(1);
  fusion.addInput(tv0);
  auto tv1 = makeContigTensor(1);
  fusion.addInput(tv1);

  auto tv2 = add(tv0, tv1);
  fusion.addOutput(tv2);

  auto tv0_cache = tv0->cacheAfter();
  auto tv1_cache = tv1->cacheAfter();

  tv2->flatten();
  // For TIDx
  tv2->split(0, 128);
  // For unroll
  tv2->split(0, 4);

  TransformPropagator propagator(tv2);
  MaxLogicalDomainInfoSpanningTree(tv2).traverse(&propagator);

  inlineMost();

  tv2->axis(0)->parallelize(ParallelType::BIDx);
  tv2->axis(1)->parallelize(ParallelType::Unroll);
  tv2->axis(2)->parallelize(ParallelType::TIDx);

  // Use these names in GetReference.
  NVF_ERROR(tv0_cache->name() == 3);
  NVF_ERROR(tv1_cache->name() == 4);

  // While the CA position of the cache tensors is -1, its allocation
  // isn't a scalar but has the unroll domain. Make sure the domain is
  // indeed indexed.
  struct GetReference : AbstractGetReference {
    GetReference(const TensorIndexer& indexer, const IdModel& id_model)
        : AbstractGetReference(indexer, id_model) {}

    Val* getLinearIndex(TensorView* tv, TensorView* maybe_consumer)
        const override {
      // Only check tv0_cache and tv1_cache
      if (tv->name() != 3 && tv->name() != 4) {
        return nullptr;
      }

      bool as_consumer = maybe_consumer == nullptr;
      auto consumer_tv = as_consumer ? tv : maybe_consumer;
      std::vector<Val*> loop_indices =
          getLoopIndices(consumer_tv, indexer_, for_loops_);
      NVF_ERROR(loop_indices.size() == 3);
      // Each of three domains corresponds to BIDx, Unroll and
      // TIDx. Only the Unroll domain is allocated.
      return loop_indices.at(1);
    }
  };

  IndexValidator<GetReference>::validate(&fusion, false);
}

// Unrolling with no unrolled loop domain
TEST_F(IndexingTest, InlinedUnroll) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto tv0 = makeContigConcreteTensor({4});
  fusion.addInput(tv0);
  auto tv1 = makeContigConcreteTensor({3, 4});
  fusion.addInput(tv1);

  auto tv2 = set(tv0);
  auto tv3 = broadcast(tv2, {true, false});
  auto tv4 = add(tv3, tv1);
  fusion.addOutput(tv4);

  tv4->split(0, 1);

  TransformPropagator propagator(tv4);
  MaxLogicalDomainInfoSpanningTree(tv4).traverse(&propagator);

  inlineMost();

  tv4->axis(1)->parallelize(ParallelType::Unroll);

  scheduler_utils::parallelizeAllLike(tv4, fusion.allTvs());

  // The CA position of tv2 is 1 as shown below:
  //
  // T2_l[ iS3{4} ] ca_pos( 1 )
  //
  // However, this doesn't mean the allocation domain of tv2 is a
  // scalar since it's inlined into tv4 that has an unrolled domain,
  // which effectively unrolls tv2 as well. Remember that unrolling of
  // a domain means the allocation of the domain is not inlined.
  //
  // The objective of the validation below is to make sure that tv2
  // indexing is done properly.

  // Make sure tv2 is indexed with the innermost loop domain
  struct GetReference : AbstractGetReference {
    GetReference(const TensorIndexer& indexer, const IdModel& id_model)
        : AbstractGetReference(indexer, id_model) {}

    Val* getLinearIndex(TensorView* tv, TensorView* maybe_consumer)
        const override {
      // Only check tv2
      if (tv->name() != 2) {
        return nullptr;
      }

      bool as_consumer = maybe_consumer == nullptr;
      auto consumer_tv = as_consumer ? tv : maybe_consumer;
      std::vector<Val*> loop_indices =
          getLoopIndices(consumer_tv, indexer_, for_loops_);
      return loop_indices.back();
    }
  };

  IndexValidator<GetReference>::validate(&fusion, false);
}

TEST_F(IndexingTest, SmemAllocationDomainForTranspose) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto tv0 = makeContigTensor(2);
  fusion.addInput(tv0);
  auto tv1 = set(tv0);
  auto tv2 = transpose(tv1, 0, 1);
  auto tv3 = set(tv2);
  auto tv4 = set(tv3);
  fusion.addOutput(tv4);

  // Reproduce the transpose scheduler manually
  tv3->setMemoryType(MemoryType::Shared);

  for (auto tv : {tv2, tv3}) {
    tv->reorder({{0, 1}});
  }

  // [I0, I1] -> [(I0/32 * I1/32), (32 * 32) / 4, 4]
  for (auto tv : fusion.allTvs()) {
    tv->split(0, 32);
    tv->split(2, 32);
    tv->reorder({{1, 2}});
    tv->merge(2, 3);
    if (tv == tv4) {
      tv->merge(1, 0);
    } else {
      tv->merge(0, 1);
    }
    tv->split(-1, 4);

    tv->axis(0)->parallelize(ParallelType::BIDx);
    tv->axis(1)->parallelize(ParallelType::TIDx);
  }

  // tv1->axis(-1)->parallelize(ParallelType::Vectorize);
  tv4->axis(-1)->parallelize(ParallelType::Vectorize);

  // Validate the smem tensor index. Its allocation domain should be
  // [32, 32], where each "32" comes from I1 and I0, respectively.
  struct GetReference : AbstractGetReference {
    GetReference(
        const TensorIndexer& indexer,
        const IdModel& id_model,
        StmtNameType smem_tv_name)
        : AbstractGetReference(indexer, id_model), smem_tv_name(smem_tv_name) {}

    Val* getLinearIndex(TensorView* tv, TensorView* maybe_consumer)
        const override {
      if (tv->name() != smem_tv_name) {
        return nullptr;
      }

      bool as_consumer = maybe_consumer == nullptr;
      auto consumer_tv = as_consumer ? tv : maybe_consumer;
      std::vector<Val*> loop_indices =
          getLoopIndices(consumer_tv, indexer_, for_loops_);

      auto merge_output_idx = IrBuilder::addExpr(
          mulExpr(loop_indices.at(1), tv->axis(2)->extent()),
          (as_consumer ? loop_indices.at(2) : tv->fusion()->zeroVal()));

      auto val32 = tv->axis(-1)
                       ->definition()
                       ->input(0)
                       ->definition()
                       ->input(0)
                       ->as<IterDomain>()
                       ->extent();
      auto merge_outer_idx = divExpr(merge_output_idx, val32);
      auto merge_inner_idx = modExpr(merge_output_idx, val32);

      if (as_consumer) {
        return addExpr(mulExpr(merge_inner_idx, val32), merge_outer_idx);
      } else {
        return addExpr(mulExpr(merge_outer_idx, val32), merge_inner_idx);
      }
    }

    StmtNameType smem_tv_name;
  };

  IndexValidator<GetReference>::validate(&fusion, false, tv3->name());

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor input0 = at::randn({256, 256}, options);

  KernelExecutor ke;
  ke.compile(&fusion, {input0});
  auto outputs = ke.run({input0});

  testValidate(&fusion, outputs, {input0}, __LINE__, __FILE__);
}

// Same fusion as ResizeTest.Slice2. There exist two paths from
// loop domains to allocation domains, and indexing needs to select
// the right one.
TEST_F(IndexingTest, ResizePath) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  std::vector<int64_t> shape({11, 30});

  NVF_CHECK(shape[1] % 2 == 0);

  auto tv0 = makeContigConcreteTensor(shape);
  fusion.addInput(tv0);

  auto tv1 = slice(tv0, {0, 0}, {shape[0], shape[1] / 2});
  auto tv2 = slice(tv0, {0, shape[1] / 2}, {shape[0], shape[1]});
  auto tv3 = add(tv1, tv2);
  fusion.addOutput(tv3);

  // TransformPrinter :
  // T0_g[ iS0{11}, iS1{30} ]
  //  logical domain : (iS0{11}, iS1{30})
  //  contiguity: f f
  //  loop domain : (iS0{11}, iS1{30})
  // T1_l[ iS2{11}rf, iS4{15}rf ]
  //  root domain : (iS2{11}rf, iS3{30}rf)
  //   Resize: iS3{30}rf by 0 and -15 -> iS4{15}rf
  //  logical domain : (iS2{11}rf, iS4{15}rf)
  //  contiguity: t t
  //  loop domain : (iS2{11}rf, iS4{15}rf)
  // T2_l[ iS5{11}rf, iS7{15}rf ]
  //  root domain : (iS5{11}rf, iS6{30}rf)
  //   Resize: iS6{30}rf by -15 and 0 -> iS7{15}rf
  //  logical domain : (iS5{11}rf, iS7{15}rf)
  //  contiguity: t t
  //  loop domain : (iS5{11}rf, iS7{15}rf)
  // T3_g[ iS8{11}, iS9{15} ]
  //  logical domain : (iS8{11}, iS9{15})
  //  contiguity: t t
  //  loop domain : (iS8{11}, iS9{15})
  // }

  // Notice that due to `add(tv1, tv2)`, the inner domains of tv1, tv2
  // and tv3 are all mapped, and that all of them are derived from
  // iS1. This means that when indexing tv0 as the consumer of tv1,
  // for example, there're two paths from the inner domain of tv1
  // (i.e., iS4) to the inner allocation domain of tv0 (i.e.,
  // iS1). One is through the resize of tv1 itself, but the resize for
  // tv2 can also allow traversal reaching iS1 from iS4 since iS7 is
  // mapped with iS7. In this case, indexing tv0 for tv1 needs to use
  // the resize for tv1 and ignore the other resize. Similarly when
  // indexing tv0 as the producer of tv2, the resize of tv2 needs to
  // be used.

  // Validate the smem tensor index. Its allocation domain should be
  // [32, 32], where each "32" comes from I1 and I0, respectively.

  struct GetReference : AbstractGetReference {
    GetReference(const TensorIndexer& indexer, const IdModel& id_model)
        : AbstractGetReference(indexer, id_model) {}

    Val* getLinearIndex(TensorView* tv, TensorView* maybe_consumer)
        const override {
      // We are only interested in producer indexing of tv0
      if (tv->name() != 0) {
        return nullptr;
      }

      NVF_ERROR(maybe_consumer != nullptr);
      auto consumer_tv = maybe_consumer;
      std::vector<Val*> loop_indices =
          getLoopIndices(consumer_tv, indexer_, for_loops_);

      switch (consumer_tv->name()) {
        case 1: {
          // Slice takes the first half, so the producer index should
          // be as the consumer index.
          return addExpr(
              mulExpr(
                  loop_indices.at(0), tv->getLogicalDomain().at(1)->extent()),
              loop_indices.at(1));
        }
        case 2: {
          // Slice takes the second half, so the producer index should
          // have an offset of the slice.
          return addExpr(
              mulExpr(
                  loop_indices.at(0), tv->getLogicalDomain().at(1)->extent()),
              addExpr(
                  loop_indices.at(1),
                  IrBuilder::create<Val>(15, DataType::Index)));
        }
        default:
          return nullptr;
      }
    }
  };

  IndexValidator<GetReference>::validate(&fusion, false);
}

// Same fusion as DoubleBufferingTest.DoubleBuffering1
TEST_F(IndexingTest, DoubleBuffering1) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto tv0 = makeContigTensor(1);
  fusion.addInput(tv0);

  auto tv1 = set(tv0);
  auto tv2 = add(tv1, IrBuilder::create<Val>(1.0));
  auto tv3 = set(tv2);
  fusion.addOutput(tv3);

  tv1->setMemoryType(MemoryType::Shared);

  tv3->split(-1, 128);
  tv3->split(-1, 32);
  TransformPropagatorWithCheck propagator(tv3);
  MaxLogicalDomainInfoSpanningTree(tv3).traverse(&propagator);

  tv1->inlineAt(-2);
  tv2->inlineAt(-2);

  tv3->axis(-2)->parallelize(ParallelType::BIDx);
  tv3->axis(-1)->parallelize(ParallelType::TIDx);
  scheduler_utils::parallelizeAllLike(tv3);

  tv1->circularBuffer(/*number_of_stages=*/2);

  struct GetReference : AbstractGetReference {
    GetReference(const TensorIndexer& indexer, const IdModel& id_model)
        : AbstractGetReference(indexer, id_model) {}

    Val* getLinearIndex(TensorView* tv, TensorView* maybe_consumer)
        const override {
      bool as_consumer = maybe_consumer == nullptr;
      auto consumer_tv = as_consumer ? tv : maybe_consumer;
      std::vector<Val*> loop_indices =
          getLoopIndices(consumer_tv, indexer_, for_loops_);

      // Don't care tensors outside circular buffered loops
      if (circular_buffer_loop_stage_ ==
          CircularBufferLoopStage::NotApplicable) {
        return nullptr;
      }

      // No epilog for this fusion
      NVF_ERROR(
          circular_buffer_loop_stage_ == CircularBufferLoopStage::Prolog ||
          circular_buffer_loop_stage_ == CircularBufferLoopStage::Main);

      switch (tv->name()) {
        case 0: {
          if (circular_buffer_loop_stage_ == CircularBufferLoopStage::Prolog) {
            // NOTE: Expression Simplification is disabled in IndexValidator,
            // so trivial index expression appears in the expression.
            Val* zero = tv->fusion()->zeroVal();
            return IrBuilder::addExpr(
                IrBuilder::mulExpr(zero, createInt(128)),
                IrBuilder::addExpr(
                    IrBuilder::mulExpr(
                        loop_indices.at(1), tv->axis(2)->extent()),
                    loop_indices.at(2)));
          } else if (
              circular_buffer_loop_stage_ == CircularBufferLoopStage::Main) {
            return addExpr(
                mulExpr(
                    addExpr(loop_indices.at(0), createInt(1)), createInt(128)),
                addExpr(
                    mulExpr(loop_indices.at(1), tv->axis(2)->extent()),
                    loop_indices.at(2)));
          } else {
            NVF_THROW(
                "Unexpected circular buffer stage: ",
                circular_buffer_loop_stage_);
          }
        }
        case 1: {
          if (as_consumer) {
            if (circular_buffer_loop_stage_ ==
                CircularBufferLoopStage::Prolog) {
              return loop_indices.at(2);
            } else if (
                circular_buffer_loop_stage_ == CircularBufferLoopStage::Main) {
              return addExpr(
                  loop_indices.at(2),
                  mulExpr(
                      modExpr(
                          addExpr(loop_indices.at(0), createInt(1)),
                          createInt(2)),
                      tv->axis(2)->extent()));
            }
          } else {
            NVF_ERROR(
                circular_buffer_loop_stage_ == CircularBufferLoopStage::Main);
            // There should be no read-ahead offset
            return addExpr(
                loop_indices.at(2),
                mulExpr(
                    modExpr(loop_indices.at(0), createInt(2)),
                    tv->axis(2)->extent()));
          }
        }
        default:
          return nullptr;
      }
    }
  };

  IndexValidator<GetReference>::validate(&fusion, false);
}

// Same fusion as DoubleBufferingTest.DoubleBuffering4
TEST_F(IndexingTest, DoubleBuffering4) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto tv0 = makeContigTensor(1);
  fusion.addInput(tv0);

  auto tv1 = add(tv0, IrBuilder::create<Val>(1.0));
  auto tv2 = set(tv1);
  auto tv3 = add(tv2, IrBuilder::create<Val>(1.0));
  fusion.addOutput(tv3);

  tv1->setMemoryType(MemoryType::Shared);

  tv3->split(-1, 128);
  tv3->split(-1, 32);
  tv3->split(-1, 8);
  TransformPropagatorWithCheck propagator(tv3);
  MaxLogicalDomainInfoSpanningTree(tv3).traverse(&propagator);

  tv0->computeAt(tv3, 2);
  tv2->computeAt(tv3, -1);

  tv3->axis(-1)->parallelize(ParallelType::TIDx);
  tv3->axis(1)->parallelize(ParallelType::Unswitch);
  scheduler_utils::parallelizeAllLike(tv3);

  tv2->circularBuffer(/*number_of_stages=*/2);

  // Check indices of:
  // - Producer of the producer of the circular buffered tensor
  // - Circular buffered tensor itself
  struct GetReference : AbstractGetReference {
    GetReference(const TensorIndexer& indexer, const IdModel& id_model)
        : AbstractGetReference(indexer, id_model) {}

    Val* getLinearIndex(TensorView* tv, TensorView* maybe_consumer)
        const override {
      bool as_consumer = maybe_consumer == nullptr;
      auto consumer_tv = as_consumer ? tv : maybe_consumer;
      std::vector<Val*> loop_indices =
          getLoopIndices(consumer_tv, indexer_, for_loops_);

      // Don't care tensors outside circular buffered loops
      if (circular_buffer_loop_stage_ ==
          CircularBufferLoopStage::NotApplicable) {
        return nullptr;
      }

      switch (tv->name()) {
        case 1: {
          if (!as_consumer) {
            if (circular_buffer_loop_stage_ ==
                CircularBufferLoopStage::Prolog) {
              return loop_indices.at(3);
            } else if (
                circular_buffer_loop_stage_ == CircularBufferLoopStage::Main) {
              return addExpr(
                  mulExpr(
                      addExpr(loop_indices.at(2), createInt(1)),
                      tv->axis(3)->extent()),
                  loop_indices.at(3));
            } else {
              NVF_THROW("Unexpected stage: ", circular_buffer_loop_stage_);
            }
          } else {
            return nullptr;
          }
        }
        case 2: {
          if (as_consumer) {
            if (circular_buffer_loop_stage_ ==
                CircularBufferLoopStage::Prolog) {
              return tv->fusion()->zeroVal();
            } else if (
                circular_buffer_loop_stage_ == CircularBufferLoopStage::Main) {
              return modExpr(
                  addExpr(loop_indices.at(2), createInt(1)), createInt(2));
            } else {
              NVF_THROW("Unexpected stage: ", circular_buffer_loop_stage_);
            }
          } else {
            if (circular_buffer_loop_stage_ == CircularBufferLoopStage::Main) {
              return modExpr(loop_indices.at(2), createInt(2));
            } else if (
                circular_buffer_loop_stage_ ==
                CircularBufferLoopStage::Epilog) {
              return modExpr(
                  subExpr(tv->axis(2)->extent(), createInt(1)), createInt(2));
            } else {
              NVF_THROW("Unexpected stage: ", circular_buffer_loop_stage_);
            }
          }
        }
        default:
          return nullptr;
      }
    }
  };

  IndexValidator<GetReference>::validate(&fusion, false);
}

// Same fusion as DoubleBufferingTest.DoubleBuffering6
TEST_F(IndexingTest, DoubleBuffering6) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto tv0 = makeContigTensor(1);
  fusion.addInput(tv0);

  auto tv1 = add(tv0, IrBuilder::create<Val>(1.0));
  auto tv2 = set(tv1);
  auto tv3 = add(tv2, IrBuilder::create<Val>(1.0));
  fusion.addOutput(tv3);

  tv1->setMemoryType(MemoryType::Shared);

  tv3->split(-1, 128);
  tv3->split(-1, 16);
  tv3->split(-2, 4);
  tv3->split(-2, 2);
  TransformPropagatorWithCheck propagator(tv3);
  MaxLogicalDomainInfoSpanningTree(tv3).traverse(&propagator);

  tv0->computeAt(tv3, 1);
  tv2->computeAt(tv3, -1);

  tv3->axis(2)->parallelize(ParallelType::Unroll);
  tv3->axis(4)->parallelize(ParallelType::TIDx);

  tv2->circularBuffer(/*number_of_stages=*/2);

  // The circular buffered tensor, tv2, is inlined into tv3, which is
  // unrolled. While tv2 is fully inlined, its allocation domain is
  // [iUR10{( ceilDiv(4, 2) )}, iS11{2}, ithreadIdx.x7{16}] due to the
  // unroll.

  // tv1 allocation domain: iS24{( ceilDiv(( ceilDiv(128, 16) ), 4) )}, iS26{(
  // ceilDiv(4, 2) )}, iS27{2}, iS23{16}

  // Check indices of:
  // - Producer of the producer of the circular buffered tensor
  // - Circular buffered tensor itself
  struct GetReference : AbstractGetReference {
    GetReference(const TensorIndexer& indexer, const IdModel& id_model)
        : AbstractGetReference(indexer, id_model) {}

    Val* getLinearIndex(TensorView* tv, TensorView* maybe_consumer)
        const override {
      bool as_consumer = maybe_consumer == nullptr;
      auto consumer_tv = as_consumer ? tv : maybe_consumer;
      std::vector<Val*> loop_indices =
          getLoopIndices(consumer_tv, indexer_, for_loops_);

      // Don't care tensors outside circular buffered loops
      if (circular_buffer_loop_stage_ ==
          CircularBufferLoopStage::NotApplicable) {
        return nullptr;
      }

      // This loop is double buffered. Since the loop originally has
      // just a trip count of 2, the double-buffered main loop has a
      // trip count of 1. Thus, this loop is always trivial
      loop_indices.at(1) = tv->fusion()->zeroVal();

      switch (tv->name()) {
        case 1: {
          if (!as_consumer) {
            if (circular_buffer_loop_stage_ ==
                CircularBufferLoopStage::Prolog) {
              return addExpr(
                  addExpr(
                      mulExpr(loop_indices.at(2), createInt(32)),
                      mulExpr(loop_indices.at(3), createInt(16))),
                  loop_indices.at(4));
            } else if (
                circular_buffer_loop_stage_ == CircularBufferLoopStage::Main) {
              return addExpr(
                  addExpr(
                      addExpr(
                          mulExpr(
                              addExpr(loop_indices.at(1), createInt(1)),
                              mulExpr(tv->axis(2)->extent(), createInt(32))),
                          mulExpr(loop_indices.at(2), createInt(32))),
                      mulExpr(loop_indices.at(3), createInt(16))),
                  loop_indices.at(4));
            } else {
              NVF_THROW("Unexpected stage: ", circular_buffer_loop_stage_);
            }
          } else {
            return nullptr;
          }
        }
        case 2: {
          if (as_consumer) {
            if (circular_buffer_loop_stage_ ==
                CircularBufferLoopStage::Prolog) {
              return addExpr(
                  mulExpr(loop_indices.at(2), createInt(2)),
                  loop_indices.at(3));
            } else if (
                circular_buffer_loop_stage_ == CircularBufferLoopStage::Main) {
              auto base_offset = addExpr(
                  mulExpr(loop_indices.at(2), tv->axis(3)->extent()),
                  loop_indices.at(3));
              // The size of the buffer is built without using
              // SimplifyingIrBuilder. Using it here would result in
              // using different dtypes (int64 vs nvfuser_index_t)
              auto buffer_offset = mulExpr(
                  modExpr(
                      addExpr(loop_indices.at(1), createInt(1)), createInt(2)),
                  IrBuilder::mulExpr(
                      tv->axis(2)->extent(), tv->axis(3)->extent()));
              return addExpr(base_offset, buffer_offset);
            } else {
              NVF_THROW("Unexpected stage: ", circular_buffer_loop_stage_);
            }
          } else {
            if (circular_buffer_loop_stage_ == CircularBufferLoopStage::Main) {
              auto base_offset = addExpr(
                  mulExpr(loop_indices.at(2), tv->axis(3)->extent()),
                  loop_indices.at(3));
              auto buffer_offset = mulExpr(
                  modExpr(loop_indices.at(1), createInt(2)),
                  IrBuilder::mulExpr(
                      tv->axis(2)->extent(), tv->axis(3)->extent()));
              return addExpr(base_offset, buffer_offset);
            } else if (
                circular_buffer_loop_stage_ ==
                CircularBufferLoopStage::Epilog) {
              auto base_offset = addExpr(
                  mulExpr(loop_indices.at(2), tv->axis(3)->extent()),
                  loop_indices.at(3));
              auto buffer_offset = mulExpr(
                  modExpr(
                      subExpr(tv->axis(1)->extent(), createInt(1)),
                      createInt(2)),
                  IrBuilder::mulExpr(
                      tv->axis(2)->extent(), tv->axis(3)->extent()));
              return addExpr(base_offset, buffer_offset);
            } else {
              NVF_THROW("Unexpected stage: ", circular_buffer_loop_stage_);
            }
          }
        }
        default:
          return nullptr;
      }
    }
  };

  IndexValidator<GetReference>::validate(&fusion, false);
}

// Same fusion as DoubleBuffering1 but with >2 stages
TEST_F(IndexingTest, CircularBuffering1) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto tv0 = makeContigTensor(1);
  fusion.addInput(tv0);

  auto tv1 = set(tv0);
  auto tv2 = add(tv1, IrBuilder::create<Val>(1.0));
  auto tv3 = set(tv2);
  fusion.addOutput(tv3);

  tv1->setMemoryType(MemoryType::Shared);

  tv3->split(-1, 128);
  tv3->split(-1, 32);
  TransformPropagatorWithCheck propagator(tv3);
  MaxLogicalDomainInfoSpanningTree(tv3).traverse(&propagator);

  tv1->inlineAt(-2);
  tv2->inlineAt(-2);

  tv3->axis(-2)->parallelize(ParallelType::BIDx);
  tv3->axis(-1)->parallelize(ParallelType::TIDx);
  scheduler_utils::parallelizeAllLike(tv3);

  tv1->circularBuffer(/*number_of_stages=*/4);

  struct GetReference : AbstractGetReference {
    GetReference(const TensorIndexer& indexer, const IdModel& id_model)
        : AbstractGetReference(indexer, id_model) {}

    Val* getLinearIndex(TensorView* tv, TensorView* maybe_consumer)
        const override {
      bool as_consumer = maybe_consumer == nullptr;
      auto consumer_tv = as_consumer ? tv : maybe_consumer;
      std::vector<Val*> loop_indices =
          getLoopIndices(consumer_tv, indexer_, for_loops_);

      // Don't care tensors outside circular buffered loops
      if (circular_buffer_loop_stage_ ==
          CircularBufferLoopStage::NotApplicable) {
        return nullptr;
      }

      // No epilog for this fusion
      NVF_ERROR(
          circular_buffer_loop_stage_ == CircularBufferLoopStage::Prolog ||
          circular_buffer_loop_stage_ == CircularBufferLoopStage::Main);

      switch (tv->name()) {
        case 0: {
          if (circular_buffer_loop_stage_ == CircularBufferLoopStage::Prolog) {
            // getLoopIndices returns the default index for each loop
            // group. Since circular buffering reuses the same loop
            // iter domain for the prologue, main and epilogue loops,
            // the loop index may not be the true index. The index
            // obtained from ForLoop should be always correct
            auto circular_buffer_index = for_loops_.at(0)->index();
            return addExpr(
                mulExpr(
                    circular_buffer_index,
                    tv->axis(0)->definition()->as<Split>()->factor()),
                addExpr(
                    mulExpr(loop_indices.at(1), tv->axis(2)->extent()),
                    loop_indices.at(2)));
          } else if (
              circular_buffer_loop_stage_ == CircularBufferLoopStage::Main) {
            return addExpr(
                mulExpr(
                    addExpr(loop_indices.at(0), createInt(3)), createInt(128)),
                addExpr(
                    mulExpr(loop_indices.at(1), tv->axis(2)->extent()),
                    loop_indices.at(2)));
          } else {
            NVF_THROW(
                "Unexpected circular buffer stage: ",
                circular_buffer_loop_stage_);
          }
        }
        case 1: {
          if (as_consumer) {
            if (circular_buffer_loop_stage_ ==
                CircularBufferLoopStage::Prolog) {
              return addExpr(
                  loop_indices.at(2),
                  mulExpr(for_loops_.at(0)->index(), tv->axis(2)->extent()));
            } else if (
                circular_buffer_loop_stage_ == CircularBufferLoopStage::Main) {
              return addExpr(
                  loop_indices.at(2),
                  mulExpr(
                      modExpr(
                          addExpr(loop_indices.at(0), createInt(3)),
                          createInt(4)),
                      tv->axis(2)->extent()));
            }
          } else {
            NVF_ERROR(
                circular_buffer_loop_stage_ == CircularBufferLoopStage::Main);
            // There should be no read-ahead offset
            return addExpr(
                loop_indices.at(2),
                mulExpr(
                    modExpr(loop_indices.at(0), createInt(4)),
                    tv->axis(2)->extent()));
          }
        }
        default:
          return nullptr;
      }
    }
  };

  IndexValidator<GetReference>::validate(&fusion, false);
}

// Same fusion as DoubleBuffering6 but with >2 stages
TEST_F(IndexingTest, CircularBuffering2) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto tv0 = makeContigTensor(1);
  fusion.addInput(tv0);

  auto tv1 = add(tv0, IrBuilder::create<Val>(1.0));
  auto tv2 = set(tv1);
  auto tv3 = add(tv2, IrBuilder::create<Val>(1.0));
  fusion.addOutput(tv3);

  tv1->setMemoryType(MemoryType::Shared);

  tv3->split(-1, 128);
  tv3->split(-1, 16);
  tv3->split(-2, 4);
  tv3->split(-2, 2);
  TransformPropagatorWithCheck propagator(tv3);
  MaxLogicalDomainInfoSpanningTree(tv3).traverse(&propagator);

  tv0->computeAt(tv3, 1);
  tv2->computeAt(tv3, -1);

  tv3->axis(2)->parallelize(ParallelType::Unroll);
  tv3->axis(4)->parallelize(ParallelType::TIDx);

  tv2->circularBuffer(/*number_of_stages=*/4);

  // The circular buffered tensor, tv2, is inlined into tv3, which is
  // unrolled. While tv2 is fully inlined, its allocation domain is
  // [iUR10{( ceilDiv(4, 2) )}, iS11{2}, ithreadIdx.x7{16}] due to the
  // unroll.

  // tv1 allocation domain: iS24{( ceilDiv(( ceilDiv(128, 16) ), 4) )}, iS26{(
  // ceilDiv(4, 2) )}, iS27{2}, iS23{16}

  // Check indices of:
  // - Producer of the producer of the circular buffered tensor
  // - Circular buffered tensor itself
  struct GetReference : AbstractGetReference {
    GetReference(const TensorIndexer& indexer, const IdModel& id_model)
        : AbstractGetReference(indexer, id_model) {}

    Val* getLinearIndex(TensorView* tv, TensorView* maybe_consumer)
        const override {
      bool as_consumer = maybe_consumer == nullptr;
      auto consumer_tv = as_consumer ? tv : maybe_consumer;
      std::vector<Val*> loop_indices =
          getLoopIndices(consumer_tv, indexer_, for_loops_);

      // Don't care tensors outside circular buffered loops
      if (circular_buffer_loop_stage_ ==
          CircularBufferLoopStage::NotApplicable) {
        return nullptr;
      }

      switch (tv->name()) {
        case 1: {
          if (!as_consumer) {
            if (circular_buffer_loop_stage_ ==
                CircularBufferLoopStage::Prolog) {
              auto buffer_offset = mulExpr(
                  for_loops_.at(1)->index(),
                  mulExpr(tv->axis(2)->extent(), createInt(32)));
              return addExpr(
                  addExpr(
                      addExpr(
                          buffer_offset,
                          mulExpr(loop_indices.at(2), createInt(32))),
                      mulExpr(loop_indices.at(3), createInt(16))),
                  loop_indices.at(4));
            } else if (
                circular_buffer_loop_stage_ == CircularBufferLoopStage::Main) {
              return addExpr(
                  addExpr(
                      addExpr(
                          mulExpr(
                              addExpr(loop_indices.at(1), createInt(3)),
                              mulExpr(tv->axis(2)->extent(), createInt(32))),
                          mulExpr(loop_indices.at(2), createInt(32))),
                      mulExpr(loop_indices.at(3), createInt(16))),
                  loop_indices.at(4));
            } else {
              NVF_THROW("Unexpected stage: ", circular_buffer_loop_stage_);
            }
          } else {
            return nullptr;
          }
        }
        case 2: {
          if (as_consumer) {
            if (circular_buffer_loop_stage_ ==
                CircularBufferLoopStage::Prolog) {
              auto buffer_offset = mulExpr(
                  for_loops_.at(1)->index(),
                  IrBuilder::mulExpr(
                      tv->axis(2)->extent(), tv->axis(3)->extent()));
              return addExpr(
                  addExpr(
                      mulExpr(loop_indices.at(2), createInt(2)),
                      loop_indices.at(3)),
                  buffer_offset);
            } else if (
                circular_buffer_loop_stage_ == CircularBufferLoopStage::Main) {
              auto base_offset = addExpr(
                  mulExpr(loop_indices.at(2), tv->axis(3)->extent()),
                  loop_indices.at(3));
              auto buffer_offset = mulExpr(
                  modExpr(
                      addExpr(loop_indices.at(1), createInt(3)), createInt(4)),
                  IrBuilder::mulExpr(
                      tv->axis(2)->extent(), tv->axis(3)->extent()));
              return addExpr(base_offset, buffer_offset);
            } else {
              NVF_THROW("Unexpected stage: ", circular_buffer_loop_stage_);
            }
          } else {
            if (circular_buffer_loop_stage_ == CircularBufferLoopStage::Main) {
              auto base_offset = addExpr(
                  mulExpr(loop_indices.at(2), tv->axis(3)->extent()),
                  loop_indices.at(3));
              auto buffer_offset = mulExpr(
                  modExpr(loop_indices.at(1), createInt(4)),
                  IrBuilder::mulExpr(
                      tv->axis(2)->extent(), tv->axis(3)->extent()));
              return addExpr(base_offset, buffer_offset);
            } else if (
                circular_buffer_loop_stage_ ==
                CircularBufferLoopStage::Epilog) {
              auto base_offset = addExpr(
                  mulExpr(loop_indices.at(2), tv->axis(3)->extent()),
                  loop_indices.at(3));
              auto buffer_offset = mulExpr(
                  modExpr(for_loops_.at(1)->index(), createInt(4)),
                  IrBuilder::mulExpr(
                      tv->axis(2)->extent(), tv->axis(3)->extent()));
              return addExpr(base_offset, buffer_offset);
            } else {
              NVF_THROW("Unexpected stage: ", circular_buffer_loop_stage_);
            }
          }
        }
        default:
          return nullptr;
      }
    }
  };

  IndexValidator<GetReference>::validate(&fusion, false);
}

// Same fusion as IndexingTest.SimplePointwise1
TEST_F(PredicateIndexingTest, SimplePointwise1) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto tv0 = makeSymbolicTensor(2);
  fusion.addInput(tv0);

  auto tv1 = add(tv0, IrBuilder::create<Val>(1.0));
  auto tv2 = add(tv1, IrBuilder::create<Val>(1.0));
  fusion.addOutput(tv2);

  tv2->flatten();
  tv2->split(0, 4);

  TransformPropagator propagator(tv2);
  MaxLogicalDomainInfoSpanningTree(tv2).traverse(&propagator);

  tv1->inlineAt(1);

  struct GetReference : AbstractGetReference {
    GetReference(const TensorIndexer& indexer, const IdModel& id_model)
        : AbstractGetReference(indexer, id_model) {}

    Val* getInlinePredicate(TensorView* tv) const override {
      std::vector<Val*> loop_indices = getLoopIndices(tv, indexer_, for_loops_);

      auto i0_idx = divExpr(
          addExpr(
              mulExpr(loop_indices.at(0), tv->axis(1)->extent()),
              loop_indices.at(1)),
          tv->getLogicalDomain().at(1)->extent());

      auto i1_idx = modExpr(
          addExpr(
              mulExpr(loop_indices.at(0), tv->axis(1)->extent()),
              loop_indices.at(1)),
          tv->getLogicalDomain().at(1)->extent());

      Val* zero = tv->fusion()->zeroVal();
      Val* cond = tv->fusion()->trueVal();
      cond = andExpr(
          andExpr(
              andExpr(
                  andExpr(cond, geExpr(i0_idx, zero)),
                  ltExpr(i0_idx, tv->getLogicalDomain().at(0)->extent())),
              geExpr(i1_idx, zero)),
          ltExpr(i1_idx, tv->getLogicalDomain().at(1)->extent()));

      return cond;
    }
  };

  PredicateIndexValidator<GetReference>::validate(&fusion, false);
}

// Testing predicate indexing with an rfactor reduction
TEST_F(PredicateIndexingTest, ReductionRfactor) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto tv0 = makeSymbolicTensor(2);
  fusion.addInput(tv0);

  auto tv1 = sum(tv0, {1});
  fusion.addOutput(tv1);

  tv1->split(1, 4, false);
  tv1->rFactor({1});

  inlineMost();

  struct GetReference : AbstractGetReference {
    GetReference(const TensorIndexer& indexer, const IdModel& id_model)
        : AbstractGetReference(indexer, id_model) {}

    Val* getInlinePredicate(TensorView* tv) const override {
      std::vector<Val*> loop_indices = getLoopIndices(tv, indexer_, for_loops_);

      bool is_init = tv->nDims() > (int64_t)for_loops_.size();

      switch (tv->name()) {
        case 1: {
          // T1_g[ iS10{i0}, rS11{( ceilDiv(i2, 4) )} ]
          //
          // If this is the initialization of the buffer, only iS10
          // should be predicated. If not, rS11 should also be predicated.
          auto is10_pred = andExpr(
              geExpr(loop_indices.at(0), tv->fusion()->zeroVal()),
              ltExpr(
                  loop_indices.at(0), tv->getLogicalDomain().at(0)->extent()));
          if (is_init) {
            return is10_pred;
          } else {
            return andExpr(
                andExpr(
                    is10_pred,
                    geExpr(loop_indices.at(1), tv->fusion()->zeroVal())),
                ltExpr(
                    loop_indices.at(1),
                    tv->getLogicalDomain().at(1)->extent()));
          }
        }
        case 2: {
          // T2_l[ iS6{i0}, rS8{4}rf, iS9{( ceilDiv(i2, 4) )}rf ]
          //
          // The initialization block should not be predicated at all.
          if (is_init) {
            return tv->fusion()->trueVal();
          } else {
            // Predicating the logical domains can result in wrong
            // outputs since the split may not be divisible, allowing
            // out-of-bounds accesses to the input
            // global-memory tensor. Instead, its root domain should be
            // used as predicate domains.
            auto is6_pred = andExpr(
                geExpr(loop_indices.at(0), tv->fusion()->zeroVal()),
                ltExpr(
                    loop_indices.at(0), tv->getRootDomain().at(0)->extent()));
            auto root_idx = addExpr(
                mulExpr(loop_indices.at(1), tv->axis(2)->extent()),
                loop_indices.at(2));
            return andExpr(
                andExpr(is6_pred, geExpr(root_idx, tv->fusion()->zeroVal())),
                ltExpr(root_idx, tv->getRootDomain().at(1)->extent()));
          }
        }
        default:
          return nullptr;
      }
    }
  };

  PredicateIndexValidator<GetReference>::validate(&fusion, false);
}

// Same fusion as IndexingTest.SimpleUnroll
TEST_F(PredicateIndexingTest, SimpleUnroll) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto tv0 = makeContigTensor(1);
  fusion.addInput(tv0);
  auto tv1 = makeContigTensor(1);
  fusion.addInput(tv1);

  auto tv2 = add(tv0, tv1);
  fusion.addOutput(tv2);

  [[maybe_unused]] auto tv0_cache = tv0->cacheAfter();
  [[maybe_unused]] auto tv1_cache = tv1->cacheAfter();

  tv2->flatten();
  // For TIDx
  tv2->split(0, 128);
  // For unroll
  tv2->split(0, 4);

  TransformPropagator propagator(tv2);
  MaxLogicalDomainInfoSpanningTree(tv2).traverse(&propagator);

  inlineMost();

  tv2->axis(0)->parallelize(ParallelType::BIDx);
  tv2->axis(1)->parallelize(ParallelType::Unroll);
  tv2->axis(2)->parallelize(ParallelType::TIDx);

  struct GetReference : AbstractGetReference {
    GetReference(const TensorIndexer& indexer, const IdModel& id_model)
        : AbstractGetReference(indexer, id_model) {}

    // T2 should look like:
    //
    // T2_g[ iblockIdx.x7{( ceilDiv(( ceilDiv(i0, 128) ), 4) )},
    // iUR8{4}, ithreadIdx.x6{128} ] ca_pos( 3 ) produce_pos( 3 )
    //
    // So, the unswitch predicate should look like:
    //
    // (blockIdx.x * 4 + 0) * 128 + threadId.x >= 0 &&
    // (blockIdx.x * 4 + 3) * 128 + threadId.x < tv0.logical_size[0]
    //
    // Note that "+ 0" remains since a symbolic Val is just replaced
    // with zero.
    Val* getOuterPredicate(TensorView* tv) const override {
      std::vector<Val*> loop_indices = getLoopIndices(tv, indexer_, for_loops_);
      auto start_idx = addExpr(
          mulExpr(
              IrBuilder::addExpr(
                  mulExpr(loop_indices.at(0), tv->axis(1)->extent()),
                  tv->fusion()->zeroVal()),
              tv->axis(2)->extent()),
          loop_indices.at(2));
      auto stop_idx = addExpr(
          mulExpr(
              addExpr(
                  mulExpr(loop_indices.at(0), tv->axis(1)->extent()),
                  subExpr(tv->axis(1)->extent(), tv->fusion()->oneVal())),
              tv->axis(2)->extent()),
          loop_indices.at(2));

      return andExpr(
          geExpr(start_idx, tv->fusion()->zeroVal()),
          ltExpr(stop_idx, tv->getLogicalDomain().at(0)->extent()));
    }
  };

  PredicateIndexValidator<GetReference>::validate(&fusion, false);
}

// Simple unswitch fusion. Unlike SimpleUnroll, it has multiple
// domains whose loop indices need to be adjusted.
TEST_F(PredicateIndexingTest, SimpleUnswitch) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto tv0 = makeContigTensor(1);
  fusion.addInput(tv0);
  auto tv1 = makeContigTensor(1);
  fusion.addInput(tv1);

  auto tv2 = add(tv0, tv1);
  fusion.addOutput(tv2);

  tv0->cacheAfter();
  tv1->cacheAfter();

  tv2->flatten();
  // For TIDx
  tv2->split(0, 128);
  // For serial
  tv2->split(0, 8);
  // For unswitch
  tv2->split(0, 4);

  TransformPropagator propagator(tv2);
  MaxLogicalDomainInfoSpanningTree(tv2).traverse(&propagator);

  inlineMost();

  tv2->axis(0)->parallelize(ParallelType::BIDx);
  tv2->axis(1)->parallelize(ParallelType::Unswitch);
  tv2->axis(3)->parallelize(ParallelType::TIDx);

  // T3_l[ iS27{( ceilDiv(( ceilDiv(( ceilDiv(i0, 128) ), 8) ), 4) )}, iS28{4},
  // iS26{8}, iS24{128} ] ca_pos( 4 ) T4_l[ iS15{( ceilDiv(( ceilDiv((
  // ceilDiv(i2, 128) ), 8) ), 4) )}, iS16{4}, iS14{8}, iS12{128} ] ca_pos( 4 )
  // T2_g[ iblockIdx.x9{( ceilDiv(( ceilDiv(( ceilDiv(i0, 128) ), 8) ), 4) )},
  // iUS10{4}, iS8{8}, ithreadIdx.x6{128} ] ca_pos( 4 ) produce_pos( 4 )

  struct GetReference : AbstractGetReference {
    GetReference(const TensorIndexer& indexer, const IdModel& id_model)
        : AbstractGetReference(indexer, id_model) {}

    // The unswitch predicate should look like:
    //
    // (((blockIdx.x * 4 + 0) * 8 + 0) * 128 + threadId.x >= 0 &&
    // (((blockIdx.x * 4 + 3) * 8 + 7) * 128 + threadId.x < tv0.logical_size[0]
    Val* getOuterPredicate(TensorView* tv) const override {
      std::vector<Val*> loop_indices = getLoopIndices(tv, indexer_, for_loops_);
      auto start_idx = addExpr(
          mulExpr(
              IrBuilder::addExpr(
                  mulExpr(
                      IrBuilder::addExpr(
                          mulExpr(loop_indices.at(0), tv->axis(1)->extent()),
                          tv->fusion()->zeroVal()),
                      tv->axis(2)->extent()),
                  tv->fusion()->zeroVal()),
              tv->axis(3)->extent()),
          loop_indices.at(3));
      auto stop_idx = addExpr(
          mulExpr(
              IrBuilder::addExpr(
                  mulExpr(
                      IrBuilder::addExpr(
                          mulExpr(loop_indices.at(0), tv->axis(1)->extent()),
                          subExpr(
                              tv->axis(1)->extent(), tv->fusion()->oneVal())),
                      tv->axis(2)->extent()),
                  subExpr(tv->axis(2)->extent(), tv->fusion()->oneVal())),
              tv->axis(3)->extent()),
          loop_indices.at(3));

      return andExpr(
          geExpr(start_idx, tv->fusion()->zeroVal()),
          ltExpr(stop_idx, tv->getLogicalDomain().at(0)->extent()));
    }
  };

  PredicateIndexValidator<GetReference>::validate(&fusion, false);
}

// Same fusion as IndexingTest.SimpleVectorize
TEST_F(PredicateIndexingTest, SimpleVectorize) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto tv0 = makeContigTensor(1);
  fusion.addInput(tv0);

  auto tv1 = set(tv0);
  auto tv2 = set(tv1);
  fusion.addOutput(tv2);

  tv2->split(0, 4);
  tv2->split(0, 128);

  tv2->axis(0)->parallelize(ParallelType::BIDx);
  tv2->axis(1)->parallelize(ParallelType::TIDx);
  tv2->axis(2)->parallelize(ParallelType::Vectorize);

  TransformPropagator propagator(tv2);
  MaxLogicalDomainInfoSpanningTree(tv2).traverse(&propagator);

  inlineMost();

  scheduler_utils::parallelizeAllLike(tv2, fusion.allTvs());

  // T1_l[ iblockIdx.x9{( ceilDiv(( ceilDiv(i0, 4) ), 128) )},
  // ithreadIdx.x10{128}, iV8{4} ] ca_pos( 2 ) T2_g[ iblockIdx.x5{( ceilDiv((
  // ceilDiv(i0, 4) ), 128) )}, ithreadIdx.x6{128}, iV4{4} ] ca_pos( 2 )
  // produce_pos( 2 )

  // Both tv1 and tv2 are vectorized. Their predicates are the same as
  // this is a simple memcpy.
  struct GetReference : AbstractGetReference {
    GetReference(const TensorIndexer& indexer, const IdModel& id_model)
        : AbstractGetReference(indexer, id_model) {}

    Val* getInlinePredicate(TensorView* tv) const override {
      std::vector<Val*> loop_indices = getLoopIndices(tv, indexer_, for_loops_);

      auto start_idx = mulExpr(
          addExpr(
              mulExpr(loop_indices.at(0), tv->axis(1)->extent()),
              loop_indices.at(1)),
          tv->axis(2)->extent());
      auto stop_idx = addExpr(
          mulExpr(
              addExpr(
                  mulExpr(loop_indices.at(0), tv->axis(1)->extent()),
                  loop_indices.at(1)),
              tv->axis(2)->extent()),
          subExpr(tv->axis(2)->extent(), createInt(1)));

      // ( ( ( ( blockIdx.x * 128 ) + threadIdx.x ) * 4 ) >= 0 ) &&
      // ( ( ( ( ( blockIdx.x * 128 ) + threadIdx.x ) * 4 ) + 3 ) < ( (( ((
      // getMetaData(T0) )).logical_size ))[0] ) ) )
      return andExpr(
          geExpr(start_idx, tv->fusion()->zeroVal()),
          ltExpr(stop_idx, tv->getLogicalDomain().at(0)->extent()));
    }
  };

  PredicateIndexValidator<GetReference>::validate(&fusion, false);
}

// Same as IndexingTest.NonInnermostVectorize
TEST_F(PredicateIndexingTest, NonInnermostVectorize) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto tv0 = makeContigTensor(1);
  fusion.addInput(tv0);
  auto tv1 = set(tv0);
  // For vectorized store
  auto tv2 = set(tv1);
  auto tv3 = set(tv2);
  fusion.addOutput(tv3);

  // Vectorize
  tv3->split(0, 4);
  // Serial
  tv3->split(0, 2);
  // TIDx
  tv3->split(0, 128);

  tv3->reorder({{-1, -2}});

  TransformPropagator propagator(tv3);
  MaxLogicalDomainInfoSpanningTree(tv3).traverse(&propagator);

  tv3->axis(0)->parallelize(ParallelType::BIDx);
  tv3->axis(1)->parallelize(ParallelType::TIDx);
  scheduler_utils::parallelizeAllLike(tv3, fusion.allTvs());

  tv1->axis(2)->parallelize(ParallelType::Vectorize);
  tv3->axis(2)->parallelize(ParallelType::Vectorize);

  // T1_l[ iblockIdx.x20{( ceilDiv(( ceilDiv(( ceilDiv(i0, 4) ), 2) ), 128) )},
  // ithreadIdx.x21{128}, iV17{4}, iS19{2} ] T2_l[ iblockIdx.x14{( ceilDiv((
  // ceilDiv(( ceilDiv(i0, 4) ), 2) ), 128) )}, ithreadIdx.x15{128}, iS11{4},
  // iS13{2} ] T3_g[ iblockIdx.x8{( ceilDiv(( ceilDiv(( ceilDiv(i0, 4) ), 2) ),
  // 128) )}, ithreadIdx.x9{128}, iV5{4}, iS7{2} ]

  // Check the vectorized tensors, i.e., tv1 and tv3. The vectorized domain is
  // not innermost. Make sure only the vectorized domain is predicated with
  // (extent - 1).
  struct GetReference : AbstractGetReference {
    GetReference(const TensorIndexer& indexer, const IdModel& id_model)
        : AbstractGetReference(indexer, id_model) {}

    Val* getInlinePredicate(TensorView* tv) const override {
      if (tv->name() != 1 && tv->name() != 3) {
        return nullptr;
      }

      std::vector<Val*> loop_indices = getLoopIndices(tv, indexer_, for_loops_);

      auto common_idx = addExpr(
          mulExpr(
              addExpr(
                  mulExpr(loop_indices.at(0), tv->axis(1)->extent()),
                  loop_indices.at(1)),
              tv->axis(3)->extent()),
          loop_indices.at(3));
      auto start_idx = mulExpr(common_idx, tv->axis(2)->extent());
      auto stop_idx = addExpr(
          mulExpr(common_idx, tv->axis(2)->extent()),
          subExpr(tv->axis(2)->extent(), createInt(1)));

      return andExpr(
          geExpr(start_idx, tv->fusion()->zeroVal()),
          ltExpr(stop_idx, tv->getLogicalDomain().at(0)->extent()));
    }
  };

  PredicateIndexValidator<GetReference>::validate(&fusion, false);
}

// Same fusion as IndexingTest.DoubleBuffering1
TEST_F(PredicateIndexingTest, DoubleBuffering1) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto tv0 = makeContigTensor(1);
  fusion.addInput(tv0);

  auto tv1 = set(tv0);
  auto tv2 = add(tv1, IrBuilder::create<Val>(1.0));
  auto tv3 = set(tv2);
  fusion.addOutput(tv3);

  tv1->setMemoryType(MemoryType::Shared);

  tv3->split(-1, 128);
  tv3->split(-1, 32);
  TransformPropagatorWithCheck propagator(tv3);
  MaxLogicalDomainInfoSpanningTree(tv3).traverse(&propagator);

  tv1->inlineAt(-2);
  tv2->inlineAt(-2);

  tv3->axis(-2)->parallelize(ParallelType::BIDx);
  tv3->axis(-1)->parallelize(ParallelType::TIDx);
  scheduler_utils::parallelizeAllLike(tv3);

  tv1->circularBuffer(/*number_of_stages=*/2);

  // T1_s[ iS12{( ceilDiv(i0, 128) )}, iblockIdx.x14{( ceilDiv(128, 32) )},
  // ithreadIdx.x15{32} ] ca_pos( 2 )

  struct GetReference : AbstractGetReference {
    GetReference(const TensorIndexer& indexer, const IdModel& id_model)
        : AbstractGetReference(indexer, id_model) {}

    Val* getInlinePredicate(TensorView* tv) const override {
      std::vector<Val*> loop_indices = getLoopIndices(tv, indexer_, for_loops_);

      // Don't care tensors outside circular buffered loops
      if (circular_buffer_loop_stage_ ==
          CircularBufferLoopStage::NotApplicable) {
        return nullptr;
      }

      // All other tensors are just predicated as usual
      if (tv->name() != 1) {
        return nullptr;
      }

      // No epilog for this fusion
      NVF_ERROR(
          circular_buffer_loop_stage_ == CircularBufferLoopStage::Prolog ||
          circular_buffer_loop_stage_ == CircularBufferLoopStage::Main);

      auto circular_buffer_index = for_loops_.at(0)->index();

      if (circular_buffer_loop_stage_ == CircularBufferLoopStage::Prolog) {
        // 0 * 128 + bidx.x * 32 + tidx.x >= 0 &&
        // 0 * 128 + bidx.x * 32 + tidx.x < N
        // NOTE: Expression Simplification is disabled in
        // PredicateIndexValidator, so trivial index expression appears in the
        // expression.
        Val* zero = tv->fusion()->zeroVal();
        auto idx = IrBuilder::addExpr(
            IrBuilder::mulExpr(zero, createInt(128)),
            IrBuilder::addExpr(
                IrBuilder::mulExpr(loop_indices.at(1), tv->axis(2)->extent()),
                loop_indices.at(2)));
        return andExpr(
            geExpr(idx, tv->fusion()->zeroVal()),
            ltExpr(idx, tv->getLogicalDomain().at(0)->extent()));
      } else {
        // (i + 1) * 128 + bidx.x * 32 + tidx.x >= 0 &&
        // (i + 1) * 128 + bidx.x * 32 + tidx.x < N
        auto idx = addExpr(
            mulExpr(
                addExpr(circular_buffer_index, tv->fusion()->oneVal()),
                createInt(128)),
            addExpr(
                mulExpr(loop_indices.at(1), tv->axis(2)->extent()),
                loop_indices.at(2)));
        return andExpr(
            geExpr(idx, tv->fusion()->zeroVal()),
            ltExpr(idx, tv->getLogicalDomain().at(0)->extent()));
      }
    }
  };

  PredicateIndexValidator<GetReference>::validate(&fusion, false);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor t0 = at::randn({1000}, options);

  EnableOptionsGuard enable_options_guard;
  EnableOptionsGuard::getCurOptions().set(EnableOption::IdModel, {"all"});

  KernelExecutor ke;
  ke.compile(&fusion, {t0});
  auto outputs = ke.run({t0});

  testValidate(&fusion, outputs, {t0}, __LINE__, __FILE__);
}

// Same fusion ad IndexingTest.CircularBuffering1
TEST_F(PredicateIndexingTest, CircularBuffering1) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto tv0 = makeContigTensor(1);
  fusion.addInput(tv0);

  auto tv1 = set(tv0);
  auto tv2 = add(tv1, IrBuilder::create<Val>(1.0));
  auto tv3 = set(tv2);
  fusion.addOutput(tv3);

  tv1->setMemoryType(MemoryType::Shared);

  tv3->split(-1, 128);
  tv3->split(-1, 32);
  TransformPropagatorWithCheck propagator(tv3);
  MaxLogicalDomainInfoSpanningTree(tv3).traverse(&propagator);

  tv1->inlineAt(-2);
  tv2->inlineAt(-2);

  tv3->axis(-2)->parallelize(ParallelType::BIDx);
  tv3->axis(-1)->parallelize(ParallelType::TIDx);
  scheduler_utils::parallelizeAllLike(tv3);

  tv1->circularBuffer(/*number_of_stages=*/4);

  // T1_s[ iS12{( ceilDiv(i0, 128) )}, iblockIdx.x14{( ceilDiv(128, 32) )},
  // ithreadIdx.x15{32} ] ca_pos( 2 )

  struct GetReference : AbstractGetReference {
    GetReference(const TensorIndexer& indexer, const IdModel& id_model)
        : AbstractGetReference(indexer, id_model) {}

    Val* getInlinePredicate(TensorView* tv) const override {
      std::vector<Val*> loop_indices = getLoopIndices(tv, indexer_, for_loops_);

      // Don't care tensors outside circular buffered loops
      if (circular_buffer_loop_stage_ ==
          CircularBufferLoopStage::NotApplicable) {
        return nullptr;
      }

      // All other tensors are just predicated as usual
      if (tv->name() != 1) {
        return nullptr;
      }

      // No epilog for this fusion
      NVF_ERROR(
          circular_buffer_loop_stage_ == CircularBufferLoopStage::Prolog ||
          circular_buffer_loop_stage_ == CircularBufferLoopStage::Main);

      auto circular_buffer_index = for_loops_.at(0)->index();

      Val* idx = nullptr;
      if (circular_buffer_loop_stage_ == CircularBufferLoopStage::Prolog) {
        // i * 128 + bidx.x * 32 + tidx.x >= 0 &&
        // i * 128 + bidx.x * 32 + tidx.x < N
        idx = addExpr(
            mulExpr(circular_buffer_index, createInt(128)),
            addExpr(
                mulExpr(loop_indices.at(1), tv->axis(2)->extent()),
                loop_indices.at(2)));
      } else {
        // (i + 3) * 128 + bidx.x * 32 + tidx.x >= 0 &&
        // (i + 3) * 128 + bidx.x * 32 + tidx.x < N
        idx = addExpr(
            mulExpr(
                addExpr(circular_buffer_index, createInt(3)), createInt(128)),
            addExpr(
                mulExpr(loop_indices.at(1), tv->axis(2)->extent()),
                loop_indices.at(2)));
      }

      return andExpr(
          geExpr(idx, tv->fusion()->zeroVal()),
          ltExpr(idx, tv->getLogicalDomain().at(0)->extent()));
    }
  };

  PredicateIndexValidator<GetReference>::validate(&fusion, false);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor t0 = at::randn({1000}, options);

  EnableOptionsGuard enable_options_guard;
  EnableOptionsGuard::getCurOptions().set(EnableOption::IdModel, {"all"});

  KernelExecutor ke;
  ke.compile(&fusion, {t0});
  auto outputs = ke.run({t0});

  testValidate(&fusion, outputs, {t0}, __LINE__, __FILE__);
}

// Same fusion as IndexingTest.CircularBuffering2. Combination of
// circular buffering and unrolling
TEST_F(PredicateIndexingTest, UnrolledCircularBuffering) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto tv0 = makeContigTensor(1);
  fusion.addInput(tv0);

  auto tv1 = add(tv0, IrBuilder::create<Val>(1.0));
  auto tv2 = set(tv1);
  auto tv3 = add(tv2, IrBuilder::create<Val>(1.0));
  fusion.addOutput(tv3);

  tv1->setMemoryType(MemoryType::Shared);

  // [I0]
  tv3->split(-1, 256);
  // [I0/256, 256]
  tv3->split(-1, 8);
  // [I0/256, 256/8, 8]
  tv3->split(-2, 4);
  // [I0/256, 256/8/4, 4, 8]
  tv3->split(-2, 2);
  // [I0/256, 256/8/4, 4/2, 2, 8]

  TransformPropagatorWithCheck propagator(tv3);
  MaxLogicalDomainInfoSpanningTree(tv3).traverse(&propagator);

  tv0->computeAt(tv3, 1);
  tv2->computeAt(tv3, -1);

  tv3->axis(0)->parallelize(ParallelType::BIDx);
  tv3->axis(2)->parallelize(ParallelType::Unroll);
  tv3->axis(4)->parallelize(ParallelType::TIDx);

  // axis(1) will be circular buffered
  tv2->circularBuffer(/*number_of_stages=*/4);

  // [I0/256, 256/8/4, 4/2, 2, 8]
  //    +        +       +      +
  //    |        |       |      +-- TIDx
  //    |        |       +-- unroll
  //    |        +-- circular buffering
  //    +-- BIDx

  struct GetReference : AbstractGetReference {
    GetReference(const TensorIndexer& indexer, const IdModel& id_model)
        : AbstractGetReference(indexer, id_model) {}

    Val* getOuterPredicate(TensorView* tv) const override {
      std::vector<Val*> loop_indices = getLoopIndices(tv, indexer_, for_loops_);

      // Don't care tensors outside circular buffered loops
      if (circular_buffer_loop_stage_ ==
          CircularBufferLoopStage::NotApplicable) {
        return nullptr;
      }

      // All other tensors are just predicated as usual
      if (tv->name() != 2) {
        return nullptr;
      }

      // Circular buffer tensor itself should not appear in Epilogue as
      // a consumer
      NVF_ERROR(
          circular_buffer_loop_stage_ == CircularBufferLoopStage::Prolog ||
          circular_buffer_loop_stage_ == CircularBufferLoopStage::Main);

      auto circular_buffer_index = for_loops_.at(1)->index();

      auto zero = tv->fusion()->zeroVal();
      auto one = tv->fusion()->oneVal();

      // The base index is:
      //
      // i0 * 256 + ((i1 * 4 + (i2 * 2 + i3)) * 16 + tidx)
      //
      // Here, i1 and i2 correspond to the circular buffer loop and
      // the unroll loop, respectively.

      Val* start_idx = nullptr;
      Val* stop_idx = nullptr;
      if (circular_buffer_loop_stage_ == CircularBufferLoopStage::Prolog) {
        // Start index: i0 * 256 + ((i1 * 4 + (0 * 2 + 0)) * 8 +
        // tidx)
        // Stop index: i0 * 256 + (((i1 + 3) * 4 + (1 * 2 + 1)) * 8 + tidx)
        start_idx = addExpr(
            mulExpr(loop_indices.at(0), createInt(256)),
            addExpr(
                mulExpr(
                    addExpr(
                        mulExpr(circular_buffer_index, createInt(4)),
                        IrBuilder::addExpr(
                            IrBuilder::mulExpr(zero, tv->axis(3)->extent()),
                            zero)),
                    tv->axis(4)->extent()),
                loop_indices.at(4)));
        stop_idx = addExpr(
            mulExpr(loop_indices.at(0), createInt(256)),
            addExpr(
                mulExpr(
                    addExpr(
                        mulExpr(circular_buffer_index, createInt(4)),
                        addExpr(
                            IrBuilder::mulExpr(
                                subExpr(tv->axis(2)->extent(), one),
                                tv->axis(3)->extent()),
                            one)),
                    tv->axis(4)->extent()),
                loop_indices.at(4)));
      } else {
        NVF_ERROR(circular_buffer_loop_stage_ == CircularBufferLoopStage::Main);
        // Start index: i0 * 256 + (((i1) * 4 + (0 * 2 + 0)) * 8 +
        // tidx)
        // Stop index: i0 * 256 + (((i1 + 3) * 4 + (1 * 2 + 1)) * 8 + tidx)
        start_idx = addExpr(
            mulExpr(loop_indices.at(0), createInt(256)),
            addExpr(
                mulExpr(
                    addExpr(
                        mulExpr(circular_buffer_index, createInt(4)),
                        IrBuilder::addExpr(
                            IrBuilder::mulExpr(zero, tv->axis(3)->extent()),
                            zero)),
                    tv->axis(4)->extent()),
                loop_indices.at(4)));
        stop_idx = addExpr(
            mulExpr(loop_indices.at(0), createInt(256)),
            addExpr(

                mulExpr(
                    addExpr(
                        mulExpr(
                            addExpr(circular_buffer_index, createInt(3)),
                            createInt(4)),
                        addExpr(
                            IrBuilder::mulExpr(
                                subExpr(tv->axis(2)->extent(), one),
                                tv->axis(3)->extent()),
                            one)),
                    tv->axis(4)->extent()),
                loop_indices.at(4)));
      }

      return andExpr(
          geExpr(start_idx, zero),
          ltExpr(stop_idx, tv->getLogicalDomain().at(0)->extent()));
    }
  };

  PredicateIndexValidator<GetReference>::validate(&fusion, false);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor t0 = at::randn({1000}, options);

  EnableOptionsGuard enable_options_guard;
  EnableOptionsGuard::getCurOptions().set(EnableOption::IdModel, {"all"});

  KernelExecutor ke;
  ke.compile(&fusion, {t0});
  auto outputs = ke.run({t0});

  testValidate(&fusion, outputs, {t0}, __LINE__, __FILE__);
}

// Completely unswitched circular buffering
TEST_F(PredicateIndexingTest, UnswitchedCircularBuffering1) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto tv0 = makeContigTensor(1);
  fusion.addInput(tv0);

  auto tv1 = set(tv0);
  auto tv2 = add(tv1, IrBuilder::create<Val>(1.0));
  fusion.addOutput(tv2);

  tv2->split(0, 4);
  tv2->split(0, 1);
  TransformPropagatorWithCheck propagator(tv2);
  MaxLogicalDomainInfoSpanningTree(tv2).traverse(&propagator);

  tv1->inlineAt(-1);

  // [I0/4/1, 1, 4]
  //          ^  ^
  //          |  +-- circular buffer
  //          |
  //          +-- unswitch
  tv1->circularBuffer(/*number_of_stages=*/2);
  tv1->axis(1)->parallelize(ParallelType::Unswitch);

  // T1_l[ iS9{( ceilDiv(( ceilDiv(i0, 4) ), 1) )}, iUS10{1}, iS8{4} ] ca_pos( 3
  // )

  struct GetReference : AbstractGetReference {
    GetReference(const TensorIndexer& indexer, const IdModel& id_model)
        : AbstractGetReference(indexer, id_model) {}

    Val* getOuterPredicate(TensorView* tv) const override {
      std::vector<Val*> loop_indices = getLoopIndices(tv, indexer_, for_loops_);

      // The base index is:
      //
      // i0 * 4 + i2
      //
      // where i2 is the circular buffer index. The index of iUS10 is
      // not included as its extent is 1.

      // Start index: i0 * 4
      Val* start_idx = IrBuilder::mulExpr(loop_indices.at(0), createInt(4));

      // Stop index: i0 * 4 + 4
      // Note that it isn't "i0 * 4 + 3" since i2 is circular buffered
      // and there's no epilog, so the main loop has a read of (i2 +
      // 1).
      Val* stop_idx = IrBuilder::addExpr(
          IrBuilder::mulExpr(loop_indices.at(0), createInt(4)), createInt(4));

      return andExpr(
          geExpr(start_idx, tv->fusion()->zeroVal()),
          ltExpr(stop_idx, tv->getLogicalDomain().at(0)->extent()));
    }
  };

  PredicateIndexValidator<GetReference>::validate(&fusion, false);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor t0 = at::randn({99}, options);

  EnableOptionsGuard enable_options_guard;
  EnableOptionsGuard::getCurOptions().set(EnableOption::IdModel, {"all"});

  KernelExecutor ke;
  ke.compile(&fusion, {t0});
  auto outputs = ke.run({t0});

  testValidate(&fusion, outputs, {t0}, __LINE__, __FILE__);
}

// Mostly the same as UnswitchedCircularBuffering1 but with Vectorize
TEST_F(PredicateIndexingTest, UnswitchedCircularBuffering2) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto tv0 = makeContigTensor(1);
  fusion.addInput(tv0);

  auto tv1 = set(tv0);
  auto tv2 = add(tv1, IrBuilder::create<Val>(1.0));
  fusion.addOutput(tv2);

  // Vectorize
  tv2->split(0, 4);
  // Circular buffering
  tv2->split(0, 128);
  // Unswitch
  tv2->split(0, 1);

  TransformPropagatorWithCheck propagator(tv2);
  MaxLogicalDomainInfoSpanningTree(tv2).traverse(&propagator);

  tv1->inlineAt(3);

  // [I0/4/128/1, 1, 128, 4]
  //      +       +   +   +
  //      |       |   |   +-- vectorize
  //      |       |   +-- circular buffering
  //      |       +-- unswitch
  //      +-- BIDx
  tv1->circularBuffer(/*number_of_stages=*/3);
  tv1->axis(3)->parallelize(ParallelType::Vectorize);
  tv1->axis(1)->parallelize(ParallelType::Unswitch);
  tv1->axis(0)->parallelize(ParallelType::BIDx);

  struct GetReference : AbstractGetReference {
    GetReference(const TensorIndexer& indexer, const IdModel& id_model)
        : AbstractGetReference(indexer, id_model) {}

    Val* getOuterPredicate(TensorView* tv) const override {
      std::vector<Val*> loop_indices = getLoopIndices(tv, indexer_, for_loops_);

      auto zero = tv->fusion()->zeroVal();

      // The base index is:
      //
      // (i0 * 128 + i2) * 4 + i3
      //
      // where i2 is the circular buffer index. Here, i3 corresponds
      // to the vectorization. Since it's vectorized, the predicate
      // uses 0 for start and (vec_factor - 1) for stop

      // Start index: (i0 * 128 + 0) * 4
      Val* start_idx = mulExpr(
          IrBuilder::addExpr(mulExpr(loop_indices.at(0), createInt(128)), zero),
          createInt(4));
      // Stop index: (i0 * 128 + 129) * 4 + 3
      Val* stop_idx = addExpr(
          mulExpr(
              addExpr(
                  mulExpr(loop_indices.at(0), createInt(128)), createInt(129)),
              createInt(4)),
          createInt(3));

      return andExpr(
          geExpr(start_idx, zero),
          ltExpr(stop_idx, tv->getLogicalDomain().at(0)->extent()));
    }
  };

  PredicateIndexValidator<GetReference>::validate(&fusion, false);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor t0 = at::randn({1000}, options);

  EnableOptionsGuard enable_options_guard;
  EnableOptionsGuard::getCurOptions().set(EnableOption::IdModel, {"all"});

  KernelExecutor ke;
  ke.compile(&fusion, {t0});
  auto outputs = ke.run({t0});

  testValidate(&fusion, outputs, {t0}, __LINE__, __FILE__);
}

// Test circular buffering with unswitch. This fusion has a non
// circular-buffered tensor that is unswitched together with a circular-buffered
// tensor. The order between the circular buffered and non circular buffered
// tensors should not affect the unswitch predicate, which should
// always be generated based on the circular buffered tensor as it has
// more restrictive conditions.
TEST_P(PredicateIndexingTest, UnswitchedCircularBuffering3) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto tv0 = makeContigTensor(1);
  fusion.addInput(tv0);
  auto tv1 = makeContigTensor(1);
  fusion.addInput(tv1);

  auto tv2 = set(tv0);
  auto tv3 = set(tv1);
  auto tv4 = add(tv2, tv3);
  fusion.addOutput(tv4);

  // Vectorize
  tv4->split(0, 4);
  // Circular buffering
  tv4->split(0, 128);
  // Unswitch
  tv4->split(0, 1);

  TransformPropagatorWithCheck propagator(tv4);
  MaxLogicalDomainInfoSpanningTree(tv4).traverse(&propagator);

  inlineAllAt(tv4, 3);

  // [I0/4/128/1, 1, 128, 4]
  //      +       +   +   +
  //      |       |   |   +-- vectorize
  //      |       |   +-- circular buffering
  //      |       +-- unswitch
  //      +-- BIDx
  tv3->axis(3)->parallelize(ParallelType::Vectorize);
  tv2->axis(3)->parallelize(ParallelType::Vectorize);

  tv4->axis(0)->parallelize(ParallelType::BIDx);
  tv4->axis(1)->parallelize(ParallelType::Unswitch);

  // Only one of the two inputs is circular buffered
  if (GetParam()) {
    tv2->circularBuffer(/*number_of_stages=*/3);
  } else {
    tv3->circularBuffer(/*number_of_stages=*/3);
  }

  struct GetReference : AbstractGetReference {
    GetReference(const TensorIndexer& indexer, const IdModel& id_model)
        : AbstractGetReference(indexer, id_model) {}

    Val* getOuterPredicate(TensorView* tv) const override {
      std::vector<Val*> loop_indices = getLoopIndices(tv, indexer_, for_loops_);

      auto zero = tv->fusion()->zeroVal();

      // The base index is:
      //
      // (i0 * 128 + i2) * 4 + i3
      //
      // where i2 is the circular buffer index. Here, i3 corresponds
      // to the vectorization. Since it's vectorized, the predicate
      // uses 0 for start and (vec_factor - 1) for stop

      // Start index: (i0 * 128 + 0) * 4
      Val* start_idx = mulExpr(
          IrBuilder::addExpr(mulExpr(loop_indices.at(0), createInt(128)), zero),
          createInt(4));
      // Stop index: (i0 * 128 + 129) * 4 + 3
      Val* stop_idx = addExpr(
          mulExpr(
              addExpr(
                  mulExpr(loop_indices.at(0), createInt(128)), createInt(129)),
              createInt(4)),
          createInt(3));

      return andExpr(
          geExpr(start_idx, zero),
          ltExpr(stop_idx, tv->getLogicalDomain().at(0)->extent()));
    }
  };

  PredicateIndexValidator<GetReference>::validate(&fusion, false);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor t0 = at::randn({1000}, options);
  at::Tensor t1 = at::randn({1000}, options);

  EnableOptionsGuard enable_options_guard;
  EnableOptionsGuard::getCurOptions().set(EnableOption::IdModel, {"all"});

  KernelExecutor ke;
  ke.compile(&fusion, {t0, t1});
  auto outputs = ke.run({t0, t1});

  testValidate(&fusion, outputs, {t0, t1}, __LINE__, __FILE__);
}

INSTANTIATE_TEST_SUITE_P(
    ,
    PredicateIndexingTest,
    testing::Bool(),
    testing::PrintToStringParamName());

// Repro for the issue with unswitched double buffer loops
// (https://github.com/NVIDIA/Fuser/issues/2159)
TEST_F(PredicateIndexingTest, UnswitchedCircularBuffering4) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto tv0 = makeContigTensor(1);
  fusion.addInput(tv0);

  auto tv1 = set(tv0);
  auto tv2 = add(tv1, IrBuilder::create<Val>(1.0));
  fusion.addOutput(tv2);

  tv2->split(-1, 8);
  tv2->split(0, 1, false);
  TransformPropagatorWithCheck propagator(tv2);
  MaxLogicalDomainInfoSpanningTree(tv2).traverse(&propagator);

  tv1->inlineAt(2);

  tv2->axis(0)->parallelize(ParallelType::Unswitch);
  scheduler_utils::parallelizeAllLike(tv2);

  tv1->circularBuffer(/*number_of_stages=*/2);

  // [ 1, i0/8/1, 8 ]
  //   +     +
  //   |     +-- circular buffering
  //   +-- unswitch

  struct GetReference : AbstractGetReference {
    GetReference(const TensorIndexer& indexer, const IdModel& id_model)
        : AbstractGetReference(indexer, id_model) {}

    Val* getOuterPredicate(TensorView* tv) const override {
      auto one = tv->fusion()->oneVal();

      // The base index is:
      //
      // (i1 * 8) + i2
      //
      // where i1 is the circular buffer index.
      //
      // Start index is 0 * 8 + 0, so it's completely eliminated by
      // SimplifyingIrBuilder

      // Stop index: ((tv->axis(1)->extent() - 1) + 1) * 8 + 7
      Val* stop_idx = addExpr(
          mulExpr(
              addExpr(subExpr(tv->axis(1)->extent(), one), one), createInt(8)),
          createInt(7));

      return ltExpr(stop_idx, tv->getLogicalDomain().at(0)->extent());
    }
  };

  PredicateIndexValidator<GetReference>::validate(&fusion, false);

  // Running this fusion with the legacy indexer would result in an
  // error if run with compute-sanitizer.
  EnableOptionsGuard enable_options_guard;
  EnableOptionsGuard::getCurOptions().set(EnableOption::IdModel, {"all"});

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  auto t0 = at::randn({16}, options);

  KernelExecutor ke;
  ke.compile(&fusion, {t0});
  auto cg_outputs = ke.run({t0});

  testValidate(&fusion, cg_outputs, {t0}, __LINE__, __FILE__);
}

// Same fusion as NVFuserTest.FusionNonDivisibleSplit1_CUDA. Just
// check if proper predicates are generated.
TEST_F(PredicateIndexingTest, NonDivisibleSplit1) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto tv0 = makeSymbolicTensor(1);
  fusion.addInput(tv0);

  auto tv1 = sum(tv0, {0});
  fusion.addOutput(tv1);

  // [I]
  tv1->split(0, 5);
  // [ceilDiv(I, 5), 5]

  // This second split is non-divisible. The split domain must be predicated.
  tv1->split(1, 3);
  // [ceilDiv(I, 5), 2, 3]

  auto tv2 = sum(tv0, {0});
  fusion.addOutput(tv2);

  // tv2 shouldn't need to have another predicate
  tv2->split(0, 4);
  tv2->split(1, 2);

  struct GetReference : AbstractGetReference {
    GetReference(const TensorIndexer& indexer, const IdModel& id_model)
        : AbstractGetReference(indexer, id_model) {}

    Val* getInlinePredicate(TensorView* tv) const override {
      std::vector<Val*> loop_indices = getLoopIndices(tv, indexer_, for_loops_);
      auto zero = tv->fusion()->zeroVal();

      // Initialization exprs should not be predicated
      if (for_loops_.empty()) {
        return tv->fusion()->trueVal();
      }

      // The predicate for the logical domain is:
      //
      // (i0 * first_split_factor + i1 * second_split_factor + i2) >=
      // 0 &&
      // (i0 * first_split_factor + i1 * second_split_factor + i2) <
      // logical_size

      IterDomain* second_split_input =
          tv->axis(1)->definition()->input(0)->as<IterDomain>();

      Val* second_split_idx = addExpr(
          mulExpr(loop_indices.at(1), tv->axis(2)->extent()),
          loop_indices.at(2));

      Val* logical_idx = addExpr(
          mulExpr(loop_indices.at(0), second_split_input->extent()),
          second_split_idx);

      Val* cond = andExpr(
          geExpr(logical_idx, zero),
          ltExpr(logical_idx, tv->getLogicalDomain().at(0)->extent()));

      // In the case of tv1, since the second split is non divisible,
      // it should have a predicate to protect the non-divisible split
      // input, which should be:
      //
      // i1 * second_split_factor + i2 < first_split_factor

      if (tv->name() == 1) {
        cond = andExpr(
            cond, ltExpr(second_split_idx, second_split_input->extent()));
      }

      return cond;
    }
  };

  PredicateIndexValidator<GetReference>::validate(&fusion, false);

  EnableOptionsGuard enable_options_guard;
  EnableOptionsGuard::getCurOptions().set(EnableOption::IdModel, {"all"});

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor t0 = at::randn({999}, options);

  KernelExecutor ke;
  ke.compile(&fusion, {t0});
  auto outputs = ke.run({t0});

  testValidate(&fusion, outputs, {t0}, __LINE__, __FILE__);
}

// Mostly same pattern as NonDivisibleSplit1 but with unswitch. The
// non divisible split predicate should also appear in the unswitch
// predicate.
TEST_F(PredicateIndexingTest, NonDivisibleSplitWithUnswitch) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto tv0 = makeSymbolicTensor(1);
  fusion.addInput(tv0);

  auto tv1 = set(tv0);
  auto tv2 = set(tv1);
  fusion.addOutput(tv2);

  // [I]
  tv1->split(0, 5);
  // [ceilDiv(I, 5), 5]

  // This second split is non-divisible. The split domain must be predicated.
  tv1->split(1, 3);
  // [ceilDiv(I, 5), 2, 3]

  // Schedule tv2 in the same way as tv1. tv2 should also have a non
  // divisible split predicate.
  TransformPropagatorWithCheck propagator(tv1);
  MaxLogicalDomainInfoSpanningTree(tv1).traverse(&propagator);

  tv1->inlineAt(-1);
  tv2->axis(0)->parallelize(ParallelType::BIDx);
  tv2->axis(1)->parallelize(ParallelType::Unswitch);
  tv2->axis(2)->parallelize(ParallelType::TIDx);

  // In this case, tv1 and tv2 are unswitched together. Both should
  // yield the same non divisible split predicate. The final unswitch
  // predicate should only have one.

  struct GetReference : AbstractGetReference {
    GetReference(const TensorIndexer& indexer, const IdModel& id_model)
        : AbstractGetReference(indexer, id_model) {}

    Val* getOuterPredicate(TensorView* tv) const override {
      std::vector<Val*> loop_indices = getLoopIndices(tv, indexer_, for_loops_);
      auto zero = tv->fusion()->zeroVal();
      auto one = tv->fusion()->oneVal();

      IterDomain* second_split_input =
          tv->axis(1)->definition()->input(0)->as<IterDomain>();

      Val* second_split_start_idx = addExpr(
          IrBuilder::mulExpr(zero, tv->axis(2)->extent()), loop_indices.at(2));

      Val* second_split_stop_idx = addExpr(
          IrBuilder::mulExpr(
              subExpr(tv->axis(1)->extent(), one), tv->axis(2)->extent()),
          loop_indices.at(2));

      Val* logical_start_idx = addExpr(
          mulExpr(loop_indices.at(0), second_split_input->extent()),
          second_split_start_idx);

      Val* logical_stop_idx = addExpr(
          mulExpr(loop_indices.at(0), second_split_input->extent()),
          second_split_stop_idx);

      Val* cond = andExpr(
          geExpr(logical_start_idx, zero),
          ltExpr(logical_stop_idx, tv->getLogicalDomain().at(0)->extent()));

      // Non divisible split. Only the stop predicate is included.
      cond = andExpr(
          cond, ltExpr(second_split_stop_idx, second_split_input->extent()));

      return cond;
    }
  };

  PredicateIndexValidator<GetReference>::validate(&fusion, false);

  EnableOptionsGuard enable_options_guard;
  EnableOptionsGuard::getCurOptions().set(EnableOption::IdModel, {"all"});

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor t0 = at::randn({999}, options);

  KernelExecutor ke;
  ke.compile(&fusion, {t0});
  auto outputs = ke.run({t0});

  testValidate(&fusion, outputs, {t0}, __LINE__, __FILE__);
}

// Testing non divisible split predicate with circular buffering
TEST_F(PredicateIndexingTest, NonDivisibleSplitWithCircularBuffering) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto tv0 = makeSymbolicTensor(1);
  fusion.addInput(tv0);

  auto tv1 = set(tv0);
  auto tv2 = set(tv1);
  fusion.addOutput(tv2);

  // [I]
  tv1->split(0, 10);
  // [ceilDiv(I, 10), 10]

  // This second split is non-divisible. The split domain must be predicated.
  tv1->split(1, 3);
  // [ceilDiv(I, 10), 4, 3]

  TransformPropagatorWithCheck propagator(tv1);
  MaxLogicalDomainInfoSpanningTree(tv1).traverse(&propagator);

  tv1->inlineAt(2);
  tv2->axis(0)->parallelize(ParallelType::TIDx);

  tv1->circularBuffer(3);

  // tv1 is circular buffered at its axis(1). The non divisible split
  // predicate should use the incremented index for the axis.

  struct GetReference : AbstractGetReference {
    GetReference(const TensorIndexer& indexer, const IdModel& id_model)
        : AbstractGetReference(indexer, id_model) {}

    Val* getInlinePredicate(TensorView* tv) const override {
      std::vector<Val*> loop_indices = getLoopIndices(tv, indexer_, for_loops_);
      auto zero = tv->fusion()->zeroVal();
      auto circular_buffer_index = for_loops_.at(1)->index();

      // Only interested in validating the circular buffer tensor
      if (tv->name() != 1) {
        return nullptr;
      }

      NVF_ERROR(
          circular_buffer_loop_stage_ == CircularBufferLoopStage::Prolog ||
          circular_buffer_loop_stage_ == CircularBufferLoopStage::Main);

      // Increment should be zero for prolog and two for main
      auto increment =
          circular_buffer_loop_stage_ == CircularBufferLoopStage::Prolog ? 0
                                                                         : 2;

      IterDomain* second_split_input =
          tv->axis(1)->definition()->input(0)->as<IterDomain>();

      Val* second_split_idx = addExpr(
          mulExpr(
              addExpr(circular_buffer_index, createInt(increment)),
              tv->axis(2)->extent()),
          loop_indices.at(2));

      Val* logical_idx = addExpr(
          mulExpr(loop_indices.at(0), second_split_input->extent()),
          second_split_idx);

      Val* cond = andExpr(
          geExpr(logical_idx, zero),
          ltExpr(logical_idx, tv->getLogicalDomain().at(0)->extent()));

      // Non divisible split. Only the stop predicate is included.
      cond =
          andExpr(cond, ltExpr(second_split_idx, second_split_input->extent()));

      return cond;
    }
  };

  PredicateIndexValidator<GetReference>::validate(&fusion, false);

  EnableOptionsGuard enable_options_guard;
  EnableOptionsGuard::getCurOptions().set(EnableOption::IdModel, {"all"});

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor t0 = at::randn({999}, options);

  KernelExecutor ke;
  ke.compile(&fusion, {t0});
  auto outputs = ke.run({t0});

  testValidate(&fusion, outputs, {t0}, __LINE__, __FILE__);
}

// Non divisible split with unswitched circular buffering. The non divisible
// predicate should only use the one generated from the main loop and
// the one from the prolog loop should not appear in the unswitch
// predicate.
TEST_F(
    PredicateIndexingTest,
    NonDivisibleSplitWithUnswitchedCircularBuffering) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto tv0 = makeSymbolicTensor(1);
  fusion.addInput(tv0);

  auto tv1 = set(tv0);
  auto tv2 = set(tv1);
  fusion.addOutput(tv2);

  // [I]
  tv1->split(0, 10);
  // [ceilDiv(I, 10), 10]

  // This second split is non-divisible. The split domain must be predicated.
  tv1->split(1, 3);
  // [ceilDiv(I, 10), 4, 3]

  tv1->split(0, 1);
  // [ceilDiv(I, 10), 1, 4, 3]

  TransformPropagatorWithCheck propagator(tv1);
  MaxLogicalDomainInfoSpanningTree(tv1).traverse(&propagator);

  tv1->inlineAt(3);
  tv2->axis(0)->parallelize(ParallelType::BIDx);
  tv2->axis(1)->parallelize(ParallelType::Unswitch);
  tv2->axis(-1)->parallelize(ParallelType::TIDx);
  tv1->axis(-1)->parallelize(ParallelType::TIDx);

  tv1->circularBuffer(3);

  // tv1 is circular buffered at its axis(1). The non divisible split
  // predicate should use the incremented index for the axis.

  // The unswitch predicate should look like:
  //
  // blockIdx.x * 10 + (0 * 3) + threadIdx.x >= 0 &&
  // blockIdx.x * 10 + ((ceilDiv(10, 3) - 1) + 2) * 3 + threadIdx.x <
  // T0.logical_size[0] &&
  // ((ceilDiv(10, 3) - 1) + 2) * 3 + threadIdx.x < 10

  struct GetReference : AbstractGetReference {
    GetReference(const TensorIndexer& indexer, const IdModel& id_model)
        : AbstractGetReference(indexer, id_model) {}

    Val* getOuterPredicate(TensorView* tv) const override {
      std::vector<Val*> loop_indices = getLoopIndices(tv, indexer_, for_loops_);
      auto zero = tv->fusion()->zeroVal();
      auto one = tv->fusion()->oneVal();

      IterDomain* second_split_input =
          tv->axis(-1)->definition()->input(0)->as<IterDomain>();

      int64_t increment = 2;

      // (0 * 3) + threadIdx.x
      Val* second_split_start_idx = addExpr(
          IrBuilder::mulExpr(zero, tv->axis(3)->extent()), loop_indices.at(3));
      // blockIdx.x * 10 + second_split_start_idx
      Val* logical_start_idx = addExpr(
          mulExpr(loop_indices.at(0), second_split_input->extent()),
          second_split_start_idx);

      // ((ceilDiv(10, 3) - 1) + 2) * 3 + threadIdx.x
      Val* second_split_stop_idx = addExpr(
          IrBuilder::mulExpr(
              addExpr(
                  subExpr(tv->axis(2)->extent(), one), createInt(increment)),
              tv->axis(3)->extent()),
          loop_indices.at(3));
      // blockIdx.x * 10 + second_split_stop_idx
      Val* logical_stop_idx = addExpr(
          mulExpr(loop_indices.at(0), second_split_input->extent()),
          second_split_stop_idx);

      Val* cond = andExpr(
          geExpr(logical_start_idx, zero),
          ltExpr(logical_stop_idx, tv->getLogicalDomain().at(0)->extent()));

      // Non divisible split
      cond = andExpr(
          cond, ltExpr(second_split_stop_idx, second_split_input->extent()));

      return cond;
    }
  };

  PredicateIndexValidator<GetReference>::validate(&fusion, false);

  EnableOptionsGuard enable_options_guard;
  EnableOptionsGuard::getCurOptions().set(EnableOption::IdModel, {"all"});

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor t0 = at::randn({999}, options);

  KernelExecutor ke;
  ke.compile(&fusion, {t0});
  auto outputs = ke.run({t0});

  testValidate(&fusion, outputs, {t0}, __LINE__, __FILE__);
}

// Repro of unswitch predicate issue #681
TEST_P(PredicateIndexingTest, UnswitchPredicateIssueRepro681) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto tv0 = makeSymbolicTensor(2);
  fusion.addInput(tv0);

  auto tv1 = sum(tv0, {0, 1});
  fusion.addOutput(tv1);

  // [i0, i1]
  tv1->split(1, 4);
  // [i0, i1/4, 4]
  tv1->merge(0);
  // [i0*i1/4, 4]
  tv1->split(0, 4);
  // [i0*i1/4/4, 4, 4]
  tv1->split(0, 1);
  // [i0*i1/4/4, 1, 4, 4]

  tv1->axis(1)->parallelize(ParallelType::Unswitch);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor t0 = at::randn({4, 10}, options);

  struct GetReference : AbstractGetReference {
    GetReference(const TensorIndexer& indexer, const IdModel& id_model)
        : AbstractGetReference(indexer, id_model) {}

    Val* getOuterPredicate(TensorView* tv) const override {
      std::vector<Val*> loop_indices = getLoopIndices(tv, indexer_, for_loops_);
      auto zero = tv->fusion()->zeroVal();
      auto one = tv->fusion()->oneVal();
      auto merge =
          dynamic_cast<Merge*>(tv->getLogicalDomain().at(0)->uses().at(0));
      NVF_ERROR(merge != nullptr);

      auto merge_out_start_idx =
          IrBuilder::addExpr(mulExpr(loop_indices.at(0), createInt(4)), zero);

      auto merge_out_stop_idx = IrBuilder::addExpr(
          mulExpr(loop_indices.at(0), createInt(4)), createInt(3));

      auto logical_id0_start_idx =
          divExpr(merge_out_start_idx, merge->inner()->extent());
      auto logical_id0_stop_idx =
          divExpr(merge_out_stop_idx, merge->inner()->extent());

      auto pred = andExpr(
          geExpr(logical_id0_start_idx, zero),
          ltExpr(logical_id0_stop_idx, tv->getLogicalDomain().at(0)->extent()));

      // Due to the modulo of the merge inner propagation, these are
      // generated by using zero or extent-1 for the merge inner path
      // propagation
      auto logical_id1_stop_idx = addExpr(
          mulExpr(subExpr(merge->inner()->extent(), one), createInt(4)),
          createInt(3));

      pred = andExpr(
          pred,
          ltExpr(logical_id1_stop_idx, tv->getLogicalDomain().at(1)->extent()));

      return pred;
    }
  };

  PredicateIndexValidator<GetReference>::validate(&fusion, false);

  EnableOptionsGuard enable_options_guard;
  if (GetParam()) {
    EnableOptionsGuard::getCurOptions().set(EnableOption::IdModel, {"all"});
  } else {
    EnableOptionsGuard::getCurOptions().unset(EnableOption::IdModel);
  }

  KernelExecutor ke;
  ke.compile(&fusion, {t0});
  auto outputs = ke.run({t0});

  auto ref = t0.to(at::kDouble).sum();

  testValidate(&fusion, outputs, {t0}, {ref}, __LINE__, __FILE__);
}

// Testing unswitched non-divisible predicates. For a given tensor,
// its required unswitch predicates should be generated from its
// indexing path, not its logical to loop dependencies. In this test,
// the tv2 transformation has an non-divisible split that needs to be
// predicated, but since it's inline into tv3, the actual non-divisible
// split should be based on tv3.
TEST_F(PredicateIndexingTest, NonDivisibleSplitWithUnswitchAndBroadcast) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto tv0 = makeSymbolicTensor(1);
  fusion.addInput(tv0);
  auto tv1 = makeSymbolicTensor(2);
  fusion.addInput(tv1);

  auto tv2 = broadcast(tv0, {false, true});
  auto tv3 = add(tv2, tv1);
  fusion.addOutput(tv3);

  // [i0, i1]
  tv3->merge(0);
  // [i0*i1]

  tv3->split(0, 100);
  // [i0*i1/100, 100]

  // This split needs to predicated as a non-divisible split
  tv3->split(1, 16);
  // [i0*i1/100, 100/16, 16]

  tv3->split(0, 1);
  // [i0*i1/100/1, 1, 100/16, 16]

  TransformPropagatorWithCheck propagator(tv3);
  MaxLogicalDomainInfoSpanningTree(tv3).traverse(&propagator);

  inlineMost();

  tv3->axis(1)->parallelize(ParallelType::Unswitch);
  tv3->axis(-2)->parallelize(ParallelType::TIDy);
  tv3->axis(-1)->parallelize(ParallelType::TIDx);

  // The reference is tv3. tv2 lacks the inner concrete domain. Since
  // tv2 is fully inlined, the effective loop domains should be the
  // same for both tensors.
  //
  // The unswitch predicates should be:
  //
  // i0_idx >= 0
  // i0_idx < i0
  // i1_idx >= 0
  // i1_idx < i1
  //
  // Additionally, since the split by 16 is non-divisible, it should
  // also have:
  //
  // threadIdx.y * 16 + threadIdx.x < 100

  struct GetReference : AbstractGetReference {
    GetReference(const TensorIndexer& indexer, const IdModel& id_model)
        : AbstractGetReference(indexer, id_model) {}

    Val* getOuterPredicate(TensorView* tv) const override {
      std::vector<Val*> loop_indices = getLoopIndices(tv, indexer_, for_loops_);
      std::vector<IterDomain*> loop_domains = getLoopDomains(tv, id_model_);
      auto zero = tv->fusion()->zeroVal();

      auto non_divisible_split_to_predicate =
          dynamic_cast<Split*>(loop_domains.at(2)->definition());
      NVF_ERROR(non_divisible_split_to_predicate != nullptr);

      auto first_merge_out_id = loop_domains.at(0)
                                    ->definition()
                                    ->input(0)
                                    ->definition()
                                    ->input(0)
                                    ->as<IterDomain>();
      auto ref_logical_id0 =
          first_merge_out_id->definition()->input(0)->as<IterDomain>();
      auto ref_logical_id1 =
          first_merge_out_id->definition()->input(1)->as<IterDomain>();

      auto non_divisible_domain_start_idx = addExpr(
          mulExpr(loop_indices.at(2), createInt(16)), loop_indices.at(3));
      auto non_divisible_domain_stop_idx = addExpr(
          mulExpr(loop_indices.at(2), createInt(16)), loop_indices.at(3));

      auto merge_out_start_idx = addExpr(
          mulExpr(loop_indices.at(0), createInt(100)),
          non_divisible_domain_start_idx);

      auto merge_out_stop_idx = addExpr(
          mulExpr(loop_indices.at(0), createInt(100)),
          non_divisible_domain_stop_idx);

      auto logical_id0_start_idx =
          divExpr(merge_out_start_idx, ref_logical_id1->extent());
      auto logical_id0_stop_idx =
          divExpr(merge_out_stop_idx, ref_logical_id1->extent());

      auto logical_id1_start_idx =
          modExpr(merge_out_start_idx, ref_logical_id1->extent());
      auto logical_id1_stop_idx =
          modExpr(merge_out_stop_idx, ref_logical_id1->extent());

      // Since tv2 is first visited, the unswitch predicate should
      // look like:
      //
      // i0_idx >= 0 // generated for tv2
      // i0_idx < i0 // generated for tv2
      // threadIdx.y * 16 + threadIdx.x < 100 // generated for tv2
      // i1_idx >= 0 // generated for tv3
      // i1_idx < i1 // generated for tv3
      //
      // Note that the unswitch predicate comes before the predicates
      // for i1.

      // i0_idx >= 0
      Val* pred = geExpr(logical_id0_start_idx, zero);

      // i0_idx < i0
      pred = andExpr(
          pred, ltExpr(logical_id0_stop_idx, ref_logical_id0->extent()));

      // threadIdx.y * 16 + threadIdx.x < 100
      pred = andExpr(
          pred,
          ltExpr(
              non_divisible_domain_stop_idx,
              non_divisible_split_to_predicate->in()->extent()));

      // i1_idx >= 0
      pred = andExpr(pred, geExpr(logical_id1_start_idx, zero));
      // i1_idx < i1
      pred = andExpr(
          pred, ltExpr(logical_id1_stop_idx, ref_logical_id1->extent()));

      return pred;
    }
  };

  PredicateIndexValidator<GetReference>::validate(&fusion, false);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor t0 = at::randn({5}, options);
  at::Tensor t1 = at::randn({5, 100}, options);

  EnableOptionsGuard enable_options_guard;
  EnableOptionsGuard::getCurOptions().set(EnableOption::IdModel, {"all"});

  KernelExecutor ke;
  ke.compile(&fusion, {t0, t1});
  auto outputs = ke.run({t0, t1});

  testValidate(&fusion, outputs, {t0, t1}, __LINE__, __FILE__);
}

// Repro of #4376. Predicating non-divisible splits that appear outside of
// logical-loop transformations.
TEST_F(PredicateIndexingTest, NonDivisibleSplitWithNonLogicalToLoopDomains) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  EnableOptionsGuard::getCurOptions().set(EnableOption::IdModel, {"all"});

  std::vector<int64_t> shape{5, 2};

  auto tv0 = makeConcreteTensor(shape);
  fusion.addInput(tv0);

  auto tv1 = set(tv0);
  auto tv2 = reshape(tv1, {IrBuilder::create<Val>(-1)});
  auto tv3 = set(tv2);
  fusion.addOutput(tv3);

  // tv1 logical: [i0, i1]
  // tv2 root: [i0, i1]
  // tv2 logical: [i2(i0*i1)]
  // tv3 logical: [i2(i0*i1)]

  // Revert the reshape
  tv2->setLoopDomain(tv2->getRootDomain());
  scheduler_tools::scheduleLoopDomainsLike({tv3}, tv2->getRootDomain());
  // [i0, i1]

  for (auto tv : {tv1, tv2, tv3}) {
    tv->split(-1, 8);
    // [i0, i3(i1/8), i4(8)]
  }

  inlineMost();

  tv3->axis(0)->parallelize(ParallelType::BIDx);
  tv3->axis(1)->parallelize(ParallelType::TIDy);
  tv3->axis(2)->parallelize(ParallelType::TIDx);

  /*
    %kernel {
    T1_l_float[iS2{5}, iS12{1}, iS13{8}] ca_pos( 3 )
     = Set( T0_g_float[iS0{5}, iS1{2}], cache_op=Streaming )
    T2_l_float[iS6{5}rf, iS14{1}, iS15{8}] ca_pos( 3 ) produce_pos( 3 ) = view(
    T1_l_float[iS2{5}, iS12{1}, iS13{8}] ca_pos( 3 ) )
    T3_g_float[iblockIdx.x10{5}, ithreadIdx.y16{1}, ithreadIdx.x17{8}] ca_pos( 3
    ) produce_pos( 3 ) = Set( T2_l_float[iS6{5}rf, iS14{1}, iS15{8}] ca_pos( 3 )
    produce_pos( 3 ), cache_op=Streaming )
  */

  // For tv3, the split of i1 to i3 and i4 is not divisible. It needs
  // to be predicated before the index is progated to i2 through the
  // merge op.
  //
  // i0  i1     i1
  // |   |    +-+-+
  // +-+-+    |   |
  //   |     i3  i4
  //   i2

  // Validate if tv3 has a non-divisible predicate for i1
  struct GetReference : AbstractGetReference {
    GetReference(const TensorIndexer& indexer, const IdModel& id_model)
        : AbstractGetReference(indexer, id_model) {}

    Val* getOuterPredicate(TensorView* tv) const override {
      // Only interested in validating tv3
      if (tv->name() != 3) {
        return nullptr;
      }

      std::vector<Val*> loop_indices = getLoopIndices(tv, indexer_, for_loops_);
      std::vector<IterDomain*> loop_domains = getLoopDomains(tv, id_model_);
      auto zero = tv->fusion()->zeroVal();

      // Predicates for the sole logical ID
      auto i2_idx = addExpr(
          mulExpr(loop_domains.at(0), createInt(2)), loop_domains.at(2));

      // i2_idx >= 0
      Val* pred = geExpr(i2_idx, zero);
      // i2_idx < i2->extent
      pred =
          andExpr(pred, ltExpr(i2_idx, tv->getLogicalDomain().at(0)->extent()));

      // Non-divisible predicate
      auto non_divisible_id_to_predicate =
          dynamic_cast<Split*>(loop_domains.at(1)->definition())->in();

      // i1 index is just threadIdx.x since the extent of i3 is 1
      auto non_divisible_pred =
          ltExpr(loop_domains.at(2), non_divisible_id_to_predicate->extent());

      pred = andExpr(pred, non_divisible_pred);

      return pred;
    }
  };

  PredicateIndexValidator<GetReference>::validate(&fusion, false);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor t0 = at::randn(shape, options);

  KernelExecutor ke;
  ke.compile(&fusion, {t0});
  auto outputs = ke.run({t0});

  testValidate(&fusion, outputs, {t0}, __LINE__, __FILE__);
}

TEST_F(PredicateIndexingTest, UnswitchConsolidationDifferentThreading) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto tv0 = makeSymbolicTensor(1);
  fusion.addInput(tv0);
  auto tv1 = makeSymbolicTensor(1);
  fusion.addInput(tv1);

  auto tv2 = set(tv0);
  auto tv3 = set(tv1);
  auto tv4 = add(tv2, tv3);
  fusion.addOutput(tv4);

  // [i0]
  tv4->split(0, 16);
  // [i0/16, 16]
  tv4->split(0, 8);
  // [i0/10/8, 8, 16]
  tv4->split(0, 1);
  // [i0/10/8/1, 1, 8, 16]

  TransformPropagatorWithCheck propagator(tv4);
  MaxLogicalDomainInfoSpanningTree(tv4).traverse(&propagator);

  inlineAllAt(tv4, 2);

  tv2->setMemoryType(MemoryType::Shared);
  tv3->setMemoryType(MemoryType::Shared);

  tv4->axis(0)->parallelize(ParallelType::BIDx);
  tv4->axis(1)->parallelize(ParallelType::Unswitch);
  tv4->axis(2)->parallelize(ParallelType::TIDy);
  tv4->axis(3)->parallelize(ParallelType::TIDx);

  tv2->axis(2)->parallelize(ParallelType::TIDy);
  tv2->axis(3)->parallelize(ParallelType::TIDx);

  tv3->axis(2)->parallelize(ParallelType::TIDx);
  tv3->axis(3)->parallelize(ParallelType::TIDy);

  // All tensors are unswitched but with different TID
  // parallelization. tv2 and tv4 are parallelized as [TIDy, TIDx],
  // whereas tv3 is as [TIDx,TIDy]. These two domains are both
  // unswitched, so the unswitch predicate should include both
  // patterns: one with TIDy and TIDx and another with TIDx and
  // TIDy. Specifically, the unswitch predicate should consist of:
  //
  // (bidx * 8 + tidy) * 16 + tidx < i0
  // (bidx * 8 + tidx) * 16 + tidy < i0
  //
  // Additionally, since tidx and tidy are both used for different
  // extents, there should be parallel domain predicates as well. The
  // overall predicate should look like as follows:
  //
  // tidx < 8 && tidy < 8 &&
  // (bidx * 8 + tidx) * 16 + tidy >= 0 &&
  // (bidx * 8 + tidx) * 16 + tidy < i0 &&
  // (bidx * 8 + tidy) * 16 + tidx >= 0 &&
  // (bidx * 8 + tidy) * 16 + tidx < i0
  struct GetReference : AbstractGetReference {
    GetReference(const TensorIndexer& indexer, const IdModel& id_model)
        : AbstractGetReference(indexer, id_model) {}

    Val* getOuterPredicate(TensorView* tv) const override {
      std::vector<Val*> loop_indices = getLoopIndices(tv, indexer_, for_loops_);
      std::vector<IterDomain*> loop_domains = getLoopDomains(tv, id_model_);
      auto zero = tv->fusion()->zeroVal();

      // Parallel domain indices
      int tidx_offset = tv->name() == 3 ? 2 : 3;
      int tidy_offset = tv->name() == 3 ? 3 : 2;

      Val* pred = ltExpr(loop_indices.at(tidx_offset), createInt(8));
      pred = andExpr(pred, ltExpr(loop_indices.at(tidy_offset), createInt(8)));

      // (bidx * 8 + tidx) * 16 + tidy >= 0 &&
      // (bidx * 8 + tidx) * 16 + tidy < i0
      auto tv3_idx = addExpr(
          mulExpr(
              addExpr(
                  mulExpr(loop_indices.at(0), createInt(8)),
                  loop_indices.at(tidx_offset)),
              createInt(16)),
          loop_indices.at(tidy_offset));
      pred = andExpr(pred, geExpr(tv3_idx, zero));
      pred = andExpr(
          pred, ltExpr(tv3_idx, tv->getLogicalDomain().at(0)->extent()));

      // (bidx * 8 + tidy) * 16 + tidx >= 0 &&
      // (bidx * 8 + tidy) * 16 + tidx < i0
      auto tv2_idx = addExpr(
          mulExpr(
              addExpr(
                  mulExpr(loop_indices.at(0), createInt(8)),
                  loop_indices.at(tidy_offset)),
              createInt(16)),
          loop_indices.at(tidx_offset));
      pred = andExpr(pred, geExpr(tv2_idx, zero));
      pred = andExpr(
          pred, ltExpr(tv2_idx, tv->getLogicalDomain().at(0)->extent()));

      return pred;
    }
  };

  PredicateIndexValidator<GetReference>::validate(&fusion, false);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor t0 = at::randn({1000}, options);
  at::Tensor t1 = at::randn({1000}, options);

  EnableOptionsGuard enable_options_guard;
  EnableOptionsGuard::getCurOptions().set(EnableOption::IdModel, {"all"});

  KernelExecutor ke;
  ke.compile(&fusion, {t0, t1});
  auto outputs = ke.run({t0, t1});

  testValidate(&fusion, outputs, {t0, t1}, __LINE__, __FILE__);
}

// Test for the conditions where omitting parallel dimension
// predicates is safe with unswitched loops.
TEST_F(PredicateIndexingTest, ParallelDimensionPredicateWithUnswitch1) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto tv0 = makeContigTensor(1);
  fusion.addInput(tv0);
  auto tv1 = makeContigTensor(1);
  fusion.addInput(tv1);
  auto tv2 = makeContigTensor(1);
  fusion.addInput(tv2);

  // Just to make TIDx non unique so the parallel dimension predicate
  // is required for TIDx
  auto tv3 = set(tv0);
  fusion.addOutput(tv3);
  tv3->axis(0)->parallelize(ParallelType::TIDx);

  auto tv4 = set(tv1);
  fusion.addOutput(tv4);

  auto tv5 = set(tv2);
  fusion.addOutput(tv5);

  // TIDx-parallelized ID is fully unswitched
  tv4->split(0, 128);
  tv4->split(0, 1, false);
  tv4->axis(0)->parallelize(ParallelType::Unswitch);
  tv4->axis(-1)->parallelize(ParallelType::TIDx);

  // TIDx-parallelized ID is not fully unswitched. The unswitch
  // predicate should have (threadIdx.x < 128)
  tv5->split(0, 128);
  tv5->split(0, 1);
  tv5->axis(-2)->parallelize(ParallelType::Unswitch);
  tv5->axis(-1)->parallelize(ParallelType::TIDx);

  struct GetReference : AbstractGetReference {
    GetReference(const TensorIndexer& indexer, const IdModel& id_model)
        : AbstractGetReference(indexer, id_model) {}

    // The unswitch predicate should look like:
    //
    // tv4:
    // ((nvfuser_index_t)threadIdx.x) >= 0LL &&
    // (ceilDiv(T4.logical_size[0LL], 128LL)) - 1LL) * 128LL) +
    // ((nvfuser_index_t)threadIdx.x)) < T4.logical_size[0LL]
    //
    // tv5:
    //  ((nvfuser_index_t)threadIdx.x) < 128LL &&
    // (i1 * 128LL) + ((nvfuser_index_t)threadIdx.x) >= 0LL &&
    // (i1 * 128LL) + ((nvfuser_index_t)threadIdx.x) < T5.logical_size[0LL]

    Val* getOuterPredicate(TensorView* tv) const override {
      std::vector<Val*> loop_indices = getLoopIndices(tv, indexer_, for_loops_);
      Val* zero = tv->fusion()->zeroVal();
      Val* one = tv->fusion()->oneVal();

      if (tv->name() == 4) {
        auto min_idx = addExpr(
            IrBuilder::mulExpr(zero, createInt(128)), loop_indices.back());
        auto min_pred = geExpr(min_idx, zero);
        auto max_idx = addExpr(
            IrBuilder::mulExpr(
                subExpr(
                    ceilDivExpr(
                        tv->getLogicalDomain().at(0)->extent(), createInt(128)),
                    one),
                createInt(128)),
            loop_indices.back());
        auto max_pred = ltExpr(max_idx, tv->getLogicalDomain().at(0)->extent());
        return andExpr(min_pred, max_pred);
      } else if (tv->name() == 5) {
        auto tidx_pred = ltExpr(loop_indices.back(), createInt(128));
        auto idx = addExpr(
            IrBuilder::mulExpr(loop_indices.at(0), createInt(128)),
            loop_indices.back());
        auto min_pred = geExpr(idx, zero);
        auto max_pred = ltExpr(idx, tv->getLogicalDomain().at(0)->extent());
        return andExpr(andExpr(tidx_pred, min_pred), max_pred);
      } else {
        return nullptr;
      }
    }
  };

  PredicateIndexValidator<GetReference>::validate(&fusion, false);
}

// Similar to ParallelDimensionPredicateWithUnswitch1 but uses other
// parallelized unswitched IDs, which means the parallel dimension
// predicate is not safe to omit.
TEST_F(PredicateIndexingTest, ParallelDimensionPredicateWithUnswitch2) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto tv0 = makeContigTensor(1);
  fusion.addInput(tv0);
  auto tv1 = makeContigTensor(1);
  fusion.addInput(tv1);
  auto tv2 = makeContigTensor(1);
  fusion.addInput(tv2);

  // Just to make TIDx non unique so the parallel dimension predicate
  // is required for TIDx
  auto tv3 = set(tv0);
  fusion.addOutput(tv3);
  tv3->axis(0)->parallelize(ParallelType::TIDx);

  // Just to make TIDx non unique so the parallel dimension predicate
  // is required for TIDx
  auto tv4 = set(tv1);
  fusion.addOutput(tv4);
  tv4->axis(0)->parallelize(ParallelType::TIDy);

  auto tv5 = set(tv2);
  fusion.addOutput(tv5);

  // Both TIDx and TIDy are not fully unswitched
  tv5->split(0, 32);
  tv5->split(0, 1, false);
  tv5->axis(0)->parallelize(ParallelType::Unswitch);
  tv5->axis(1)->parallelize(ParallelType::TIDy);
  tv5->axis(2)->parallelize(ParallelType::TIDx);

  struct GetReference : AbstractGetReference {
    GetReference(const TensorIndexer& indexer, const IdModel& id_model)
        : AbstractGetReference(indexer, id_model) {}

    // The unswitch predicate should look like:
    //
    // tv5:
    // (nvfuser_index_t)threadIdx.x < 32LL
    // (nvfuser_index_t)threadIdx.y < ceilDiv(T2.logical_size[0LL], 32LL)
    // (((nvfuser_index_t)threadIdx.y) * 32LL) + ((nvfuser_index_t)threadIdx.x)
    // >= 0LL
    // (((nvfuser_index_t)threadIdx.y) * 32LL) + ((nvfuser_index_t)threadIdx.x)
    // < T2.logical_size[0LL]

    Val* getOuterPredicate(TensorView* tv) const override {
      std::vector<Val*> loop_indices = getLoopIndices(tv, indexer_, for_loops_);
      Val* zero = tv->fusion()->zeroVal();

      if (tv->name() == 5) {
        auto tidx_pred = ltExpr(loop_indices.at(2), createInt(32));
        auto tidy_pred = ltExpr(
            loop_indices.at(1),
            ceilDiv(tv->getLogicalDomain().at(0)->extent(), createInt(32)));
        auto idx = addExpr(
            IrBuilder::mulExpr(loop_indices.at(1), createInt(32)),
            loop_indices.at(2));
        auto min_pred = geExpr(idx, zero);
        auto max_pred = ltExpr(idx, tv->getLogicalDomain().at(0)->extent());
        return andExpr(
            andExpr(andExpr(tidx_pred, tidy_pred), min_pred), max_pred);
      } else {
        return nullptr;
      }
    }
  };

  PredicateIndexValidator<GetReference>::validate(&fusion, false);
}

// Check if a parallel dimension predicate is propertly used with a
// loop domain set as the producer of a logical domain. Because of the
// reversed depency, BFS traversal is required. This test resulted in
// a validation failure before PR #3938.
TEST_F(
    PredicateIndexingTest,
    ParallelDimensionPredicateWithUnswitchAndSetLoopDomain) {
  // EnableOptionsGuard enable_options_guard;
  EnableOptionsGuard::getCurOptions().set(EnableOption::IdModel, {"all"});

  Fusion fusion;
  FusionGuard fg(&fusion);

  auto tv0 = makeContigTensor(1);
  fusion.addInput(tv0);

  auto tv1 = makeContigConcreteTensor({4, 8});
  fusion.addInput(tv1);

  // Just to make TIDx non unique so the parallel dimension predicate
  // is required for TIDx
  auto tv2 = set(tv0);
  fusion.addOutput(tv2);
  tv2->axis(0)->parallelize(ParallelType::TIDx);

  auto tv3 = reshape(tv1, {4, 8}, {32});
  auto tv4 = sum(tv3, {0});
  fusion.addOutput(tv4);

  // Cancel the reshape in the loop domain [4, 8]
  tv3->setLoopDomain(tv3->getRootDomain());

  // Make the loop domain of tv4 look like that of tv3.
  // TODO: use scheduler_tools::scheduleLoopDomainsLike, which doesn't
  // seem to propertly set the IterType of the new IDs.
  auto tv4_loop_id0 = IterDomainBuilder(tv3->getLoopDomain().at(0))
                          .iter_type(IterType::Reduction)
                          .build();
  auto tv4_loop_id1 = IterDomainBuilder(tv3->getLoopDomain().at(1))
                          .iter_type(IterType::Reduction)
                          .build();
  IrBuilder::create<Merge>(
      tv4->getLogicalDomain().at(0), tv4_loop_id0, tv4_loop_id1);
  tv4->setLoopDomain({tv4_loop_id0, tv4_loop_id1});

  // Schedule tv3 and tv4 as:
  // [Serial(4), Unswitch(1), TIDx(8)]
  for (auto tv : {tv3, tv4}) {
    tv->split(1, 1, false);
  }

  tv3->inlineAt(-1);

  tv4->axis(-2)->parallelize(ParallelType::Unswitch);
  tv4->axis(-1)->parallelize(ParallelType::TIDx);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor t0 = at::randn({128}, options);
  at::Tensor t1 = at::randn({4, 8}, options);

  KernelExecutor ke;
  ke.compile(&fusion, {t0, t1});
  auto outputs = ke.run({t0, t1});

  testValidate(&fusion, outputs, {t0, t1}, __LINE__, __FILE__);

  struct GetReference : AbstractGetReference {
    GetReference(const TensorIndexer& indexer, const IdModel& id_model)
        : AbstractGetReference(indexer, id_model) {}

    // The unswitch predicate should look like:
    //
    // tv3:
    // ((nvfuser_index_t)threadIdx.x) < 8LL &&
    // ((i0 * 8LL) + ((nvfuser_index_t)threadIdx.x)) >= 0LL
    // ((i0 * 8LL) + ((nvfuser_index_t)threadIdx.x)) < 32LL

    Val* getOuterPredicate(TensorView* tv) const override {
      std::vector<Val*> loop_indices = getLoopIndices(tv, indexer_, for_loops_);
      Val* zero = tv->fusion()->zeroVal();

      if (tv->name() == 3) {
        auto tidx = loop_indices.back();
        auto tid_pred = ltExpr(tidx, createInt(8));
        auto idx = addExpr(mulExpr(loop_indices.front(), createInt(8)), tidx);
        auto min_pred = geExpr(idx, zero);
        auto max_pred = ltExpr(idx, createInt(32));
        return andExpr(andExpr(tid_pred, min_pred), max_pred);
      } else {
        return nullptr;
      }
    }
  };

  PredicateIndexValidator<GetReference>::validate(&fusion, false);
}

// Same fusion as SimplePointwise1 but with contig indexing
TEST_F(ContigIndexingTest, SimplePointwise) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto tv0 = makeContigTensor(2);
  fusion.addInput(tv0);

  auto tv1 = add(tv0, IrBuilder::create<Val>(1.0));
  auto tv2 = add(tv1, IrBuilder::create<Val>(1.0));
  fusion.addOutput(tv2);

  tv2->flatten();
  tv2->split(0, 4);

  TransformPropagator propagator(tv2);
  MaxLogicalDomainInfoSpanningTree(tv2).traverse(&propagator);

  tv1->inlineAt(1);

  // Because of contig indexing, the index of tv0 and tv2 should be
  // just: i0 * 4 + i1.
  struct GetReference : AbstractGetReference {
    GetReference(const TensorIndexer& indexer, const IdModel& id_model)
        : AbstractGetReference(indexer, id_model) {}

    Val* getLinearIndex(TensorView* tv, TensorView* maybe_consumer)
        const override {
      bool as_consumer = maybe_consumer == nullptr;
      auto consumer_tv = as_consumer ? tv : maybe_consumer;
      std::vector<Val*> loop_indices =
          getLoopIndices(consumer_tv, indexer_, for_loops_);
      switch (tv->name()) {
        case 0: {
          NVF_ERROR(!as_consumer);
          return addExpr(
              mulExpr(loop_indices.at(0), consumer_tv->axis(1)->extent()),
              loop_indices.at(1));
        }
        case 1: {
          return loop_indices.at(1);
        }
        case 2: {
          NVF_ERROR(as_consumer);
          return addExpr(
              mulExpr(loop_indices.at(0), consumer_tv->axis(1)->extent()),
              loop_indices.at(1));
        }
        default:
          NVF_THROW("Unexpected tensor: ", tv->toString());
          break;
      }
      return nullptr;
    }
  };

  IndexValidator<GetReference>::validate(&fusion, true);
}

TEST_F(ContigIndexingTest, NonContigInnermost) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  // Innermost dimension is non contiguous but the other two
  // dimensions are contiguous.
  auto tv0 = TensorViewBuilder()
                 .ndims(3)
                 .dtype(DataType::Float)
                 .contiguity({true, true, false})
                 .build();
  fusion.addInput(tv0);

  auto tv1 = set(tv0);
  fusion.addOutput(tv1);

  // [I0, I1, I2]
  tv1->merge(1);
  // [I0, I1*I2]

  // Since the i1 contig flag is true, the merge is contiguous even
  // though i2 is not contiguous. The producer index of tv0 should be:
  // i0 * I0_stride + i1 * I2_stride. The stride of I0 should be
  // calculated as I2_stride * I2_extent * I1_extent.
  //
  // As for tv1, since it's fully contiguous, it should also be i0 *
  // I0_stride + i1. Here, I0_stride should be I2_extent * I1_extent.
  struct GetReference : AbstractGetReference {
    GetReference(const TensorIndexer& indexer, const IdModel& id_model)
        : AbstractGetReference(indexer, id_model) {}

    Val* getLinearIndex(TensorView* tv, TensorView* maybe_consumer)
        const override {
      bool as_consumer = maybe_consumer == nullptr;
      auto consumer_tv = as_consumer ? tv : maybe_consumer;
      std::vector<Val*> loop_indices =
          getLoopIndices(consumer_tv, indexer_, for_loops_);
      switch (tv->name()) {
        case 0: {
          NVF_ERROR(!as_consumer);
          auto i0_stride = mulExpr(
              mulExpr(
                  IrBuilder::getItemExpr(
                      IrBuilder::getAttrExpr(
                          IrBuilder::metadataExpr(tv), "alloc_stride"),
                      IrBuilder::create<Val>(2, DataType::Int)),
                  tv->getLogicalDomain().at(2)->extent()),
              tv->getLogicalDomain().at(1)->extent());
          auto i2_stride = IrBuilder::getItemExpr(
              IrBuilder::getAttrExpr(
                  IrBuilder::metadataExpr(tv), "alloc_stride"),
              IrBuilder::create<Val>(2, DataType::Int));
          return addExpr(
              mulExpr(loop_indices.at(0), i0_stride),
              mulExpr(loop_indices.at(1), i2_stride));
        }
        case 1: {
          NVF_ERROR(as_consumer);
          return addExpr(
              mulExpr(
                  loop_indices.at(0),
                  mulExpr(
                      consumer_tv->getLogicalDomain().at(2)->extent(),
                      consumer_tv->getLogicalDomain().at(1)->extent())),
              loop_indices.at(1));
        }
        default:
          NVF_THROW("Unexpected tensor: ", tv->toString());
          break;
      }
      return nullptr;
    }
  };

  IndexValidator<GetReference>::validate(&fusion, true);
}

// Contig indexing with broadcast inlining
TEST_F(ContigIndexingTest, BroadcastInlining) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto tv0 = makeContigTensor(1);
  fusion.addInput(tv0);
  auto tv1 = makeContigTensor(2);
  fusion.addInput(tv1);

  auto tv2 = set(tv0);
  auto tv3 = broadcast(tv2, {true, false});
  auto tv4 = add(tv3, tv1);
  fusion.addOutput(tv4);

  tv4->merge(0);

  TransformPropagator propagator(tv4);
  MaxLogicalDomainInfoSpanningTree(tv4).traverse(&propagator);

  inlineMost();

  // t4 is indexed at the merge output domain, so its index should be
  // just its sole loop index. t2 and t3 are fully inlined
  // intermediate tensors, so their indices are just zero. Since t1 is
  // contiguous, it's also just indexed with the loop index. t0, on
  // the other hand, needs to back traverse the merge since its sole
  // index domain corresponds to the inner merge input domain.
  struct GetReference : AbstractGetReference {
    GetReference(const TensorIndexer& indexer, const IdModel& id_model)
        : AbstractGetReference(indexer, id_model) {}

    Val* getLinearIndex(TensorView* tv, TensorView* maybe_consumer)
        const override {
      bool as_consumer = maybe_consumer == nullptr;
      auto consumer_tv = as_consumer ? tv : maybe_consumer;
      std::vector<Val*> loop_indices =
          getLoopIndices(consumer_tv, indexer_, for_loops_);
      switch (tv->name()) {
        case 0: {
          NVF_ERROR(!as_consumer);
          return modExpr(
              loop_indices.at(0), tv->getLogicalDomain().at(0)->extent());
        }
        case 1: {
          NVF_ERROR(!as_consumer);
          return loop_indices.at(0);
        }
        case 2:
        case 3:
          return tv->fusion()->zeroVal();
        case 4: {
          NVF_ERROR(as_consumer);
          return loop_indices.at(0);
        }
        default:
          NVF_THROW("Unexpected tensor: ", tv->toString());
          break;
      }
      return nullptr;
    }
  };

  IndexValidator<GetReference>::validate(&fusion, true);
}

// Merge after resize is not allowed to do contig indexing even when
// the original input domains are contiguous.
TEST_F(ContigIndexingTest, Resize) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  std::vector<int64_t> shape({11, 30});

  NVF_CHECK(shape[1] % 2 == 0);

  auto tv0 = makeContigConcreteTensor(shape);
  fusion.addInput(tv0);

  auto tv1 = slice(tv0, {0, shape[1] / 2}, {shape[0], shape[1]});
  auto tv2 = add(tv1, IrBuilder::create<Val>(1));
  fusion.addOutput(tv2);

  // Contig merge
  tv2->merge(0);

  TransformPropagator propagator(tv2);
  MaxLogicalDomainInfoSpanningTree(tv2).traverse(&propagator);

  // All tensors except for tv0 are indexed at the output of the merge
  // op, so their indices should be just loop_indices[0]. However, for
  // tv0, since the merge follows a resize, indexing is done at the
  // resize input domain.
  struct GetReference : AbstractGetReference {
    GetReference(const TensorIndexer& indexer, const IdModel& id_model)
        : AbstractGetReference(indexer, id_model) {}

    Val* getLinearIndex(TensorView* tv, TensorView* maybe_consumer)
        const override {
      bool as_consumer = maybe_consumer == nullptr;
      auto consumer_tv = as_consumer ? tv : maybe_consumer;
      std::vector<Val*> loop_indices =
          getLoopIndices(consumer_tv, indexer_, for_loops_);
      switch (tv->name()) {
        case 0: {
          NVF_ERROR(!as_consumer);
          auto id0 = mulExpr(
              divExpr(
                  loop_indices.at(0),
                  consumer_tv->getLogicalDomain().at(1)->extent()),
              tv->getLogicalDomain().at(1)->extent());
          auto resize = dynamic_cast<Resize*>(
              consumer_tv->getLogicalDomain().at(1)->definition());
          NVF_ERROR(resize != nullptr);
          auto id1 = subExpr(
              modExpr(
                  loop_indices.at(0),
                  consumer_tv->getLogicalDomain().at(1)->extent()),
              resize->leftExpand());
          return addExpr(id0, id1);
        }
        case 1:
        case 2:
          return loop_indices.at(0);
        default:
          NVF_THROW("Unexpected tensor: ", tv->toString());
          break;
      }
      return nullptr;
    }
  };

  IndexValidator<GetReference>::validate(&fusion, true);
}

// Contiguous tensor but merge order breaks contiguity
TEST_F(ContigIndexingTest, NonConsistentMerge) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto tv0 = makeContigTensor(3);
  fusion.addInput(tv0);

  auto tv1 = add(tv0, IrBuilder::create<Val>(1));
  fusion.addOutput(tv1);

  tv1->merge(0, 2);
  tv1->merge(0, 1);

  TransformPropagator propagator(tv1);
  MaxLogicalDomainInfoSpanningTree(tv1).traverse(&propagator);

  // Make sure both tv0 and tv1 are indexed without contig indexing
  struct GetReference : AbstractGetReference {
    GetReference(const TensorIndexer& indexer, const IdModel& id_model)
        : AbstractGetReference(indexer, id_model) {}

    Val* getLinearIndex(TensorView* tv, TensorView* maybe_consumer)
        const override {
      bool as_consumer = maybe_consumer == nullptr;
      auto consumer_tv = as_consumer ? tv : maybe_consumer;
      std::vector<Val*> loop_indices =
          getLoopIndices(consumer_tv, indexer_, for_loops_);

      auto id0 = divExpr(
          divExpr(loop_indices.at(0), tv->getLogicalDomain().at(1)->extent()),
          tv->getLogicalDomain().at(2)->extent());
      auto id0_extent = mulExpr(
          tv->getLogicalDomain().at(2)->extent(),
          tv->getLogicalDomain().at(1)->extent());
      auto id1 =
          modExpr(loop_indices.at(0), tv->getLogicalDomain().at(1)->extent());
      auto id1_extent = tv->getLogicalDomain().at(2)->extent();
      auto id2 = modExpr(
          divExpr(loop_indices.at(0), tv->getLogicalDomain().at(1)->extent()),
          tv->getLogicalDomain().at(2)->extent());
      return addExpr(
          addExpr(mulExpr(id0, id0_extent), mulExpr(id1, id1_extent)), id2);
    }
  };

  IndexValidator<GetReference>::validate(&fusion, true);
}

TEST_F(ContigIndexingTest, ConcretizedBroadcastMerge) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  // [I0, I1]
  auto tv0 = makeContigTensor(2);
  fusion.addInput(tv0);
  // [I0, I1, I2]
  auto tv1 = makeContigTensor(3);
  fusion.addInput(tv1);

  auto tv2 = broadcast(tv0, {false, false, true});
  auto tv3 = add(tv2, tv1);
  fusion.addOutput(tv3);

  tv3->merge(1, 2);
  tv3->merge(0, 1);

  TransformPropagator propagator(tv3);
  MaxLogicalDomainInfoSpanningTree(tv3).traverse(&propagator);

  tv3->axis(0)->parallelize(ParallelType::TIDx);
  scheduler_utils::parallelizeAllLike(tv3);

  tv2->setMemoryType(MemoryType::Shared);

  // tv2's broadcast domain is concretized. Previously, this would
  // have prevented contig indexing.

  struct GetReference : AbstractGetReference {
    GetReference(const TensorIndexer& indexer, const IdModel& id_model)
        : AbstractGetReference(indexer, id_model) {}

    Val* getLinearIndex(TensorView* tv, TensorView* maybe_consumer)
        const override {
      // Only interested in tv2 here since that's the one that has a
      // concretized broadcast domain
      if (tv->name() != 2) {
        return nullptr;
      }

      bool as_consumer = maybe_consumer == nullptr;
      auto consumer_tv = as_consumer ? tv : maybe_consumer;
      std::vector<Val*> loop_indices =
          getLoopIndices(consumer_tv, indexer_, for_loops_);

      // When indexed as a consumer, the second merge is a contig
      // merge, so the index should be just threadIdx.x
      if (as_consumer) {
        return loop_indices.at(0);
      }

      // When indexed as a producer of tv1, the loop domain has all
      // the concrete domains merged, so it needs to be
      // decomposed. Specifically, the loop domain, threadIdx.x, should be
      // decomposed as:
      //
      // Index of the outer logical domain: tidx / (I1 * I2)
      // Index of the inner logical domain: tidx % (I1 * I2) / I2
      //
      // Since the allocation domain of t2 is (I0 * I1), the final
      // index is (tidx / (I1 * I2) * I1 + tidx % (I1 * I2) / I2)

      auto logical0 = divExpr(
          loop_indices.at(0),
          mulExpr(
              consumer_tv->getLogicalDomain().at(1)->extent(),
              consumer_tv->getLogicalDomain().at(2)->extent()));

      auto logical1 = divExpr(
          modExpr(
              loop_indices.at(0),
              mulExpr(
                  consumer_tv->getLogicalDomain().at(1)->extent(),
                  consumer_tv->getLogicalDomain().at(2)->extent())),
          consumer_tv->getLogicalDomain().at(2)->extent());

      auto alloc0 = addExpr(
          mulExpr(logical0, tv->getLogicalDomain().at(1)->extent()), logical1);

      return alloc0;
    }
  };

  IndexValidator<GetReference>::validate(&fusion, true);

  EnableOptionsGuard enable_options_guard;
  EnableOptionsGuard::getCurOptions().set(EnableOption::IdModel, {"all"});

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  auto t0 = at::randn({5, 6}, options);
  auto t1 = at::randn({5, 6, 7}, options);

  KernelExecutor ke;
  ke.compile(&fusion, {t0, t1});
  auto cg_outputs = ke.run({t0, t1});

  testValidate(&fusion, cg_outputs, {t0, t1}, __LINE__, __FILE__);
}

TEST_F(ContigPredicateIndexingTest, SimplePointwise1) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto tv0 = makeSymbolicTensor(2);
  fusion.addInput(tv0);

  auto tv1 = add(tv0, IrBuilder::create<Val>(1.0));
  auto tv2 = add(tv1, IrBuilder::create<Val>(1.0));
  fusion.addOutput(tv2);

  // Merge inner and outer, not outer and inner. This would disable
  // contig indexing for normal tensor indexing but shouldn't matter
  // for predicate indexing
  tv2->merge(1, 0);
  tv2->split(0, 4);

  TransformPropagator propagator(tv2);
  MaxLogicalDomainInfoSpanningTree(tv2).traverse(&propagator);

  tv1->inlineAt(1);

  struct GetReference : AbstractGetReference {
    GetReference(const TensorIndexer& indexer, const IdModel& id_model)
        : AbstractGetReference(indexer, id_model) {}

    Val* getInlinePredicate(TensorView* tv) const override {
      std::vector<Val*> loop_indices = getLoopIndices(tv, indexer_, for_loops_);

      auto flatten_idx = addExpr(
          mulExpr(loop_indices.at(0), tv->axis(1)->extent()),
          loop_indices.at(1));

      // Since the flatten merge is ordered as inner then outer, the
      // extent is inner_extent * outer_extent
      auto flatten_extent = mulExpr(
          tv->getLogicalDomain().at(1)->extent(),
          tv->getLogicalDomain().at(0)->extent());

      Val* zero = tv->fusion()->zeroVal();
      return andExpr(
          geExpr(flatten_idx, zero), ltExpr(flatten_idx, flatten_extent));
    }
  };

  PredicateIndexValidator<GetReference>::validate(&fusion, true);
}

// Almost the same fusion as PredicateIndexingTest.SimpleUnswitch
TEST_F(ContigPredicateIndexingTest, SimpleUnswitch) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto tv0 = makeSymbolicTensor(2);
  fusion.addInput(tv0);
  auto tv1 = makeSymbolicTensor(2);
  fusion.addInput(tv1);

  auto tv2 = add(tv0, tv1);
  fusion.addOutput(tv2);

  tv0->cacheAfter();
  tv1->cacheAfter();

  tv2->flatten();
  // For TIDx
  tv2->split(0, 128);
  // For serial
  tv2->split(0, 8);
  // For unswitch
  tv2->split(0, 4);

  TransformPropagator propagator(tv2);
  MaxLogicalDomainInfoSpanningTree(tv2).traverse(&propagator);

  inlineMost();

  tv2->axis(0)->parallelize(ParallelType::BIDx);
  tv2->axis(1)->parallelize(ParallelType::Unswitch);
  tv2->axis(3)->parallelize(ParallelType::TIDx);

  struct GetReference : AbstractGetReference {
    GetReference(const TensorIndexer& indexer, const IdModel& id_model)
        : AbstractGetReference(indexer, id_model) {}

    // The unswitch predicate should look like:
    //
    // (((blockIdx.x * 4 + 0) * 8 + 0) * 128 + threadId.x >= 0 &&
    // (((blockIdx.x * 4 + 3) * 8 + 7) * 128 + threadId.x <
    // tv0.logical_size[0] * tv0.logical_size[1]
    Val* getOuterPredicate(TensorView* tv) const override {
      std::vector<Val*> loop_indices = getLoopIndices(tv, indexer_, for_loops_);
      auto start_idx = addExpr(
          mulExpr(
              IrBuilder::addExpr(
                  mulExpr(
                      IrBuilder::addExpr(
                          mulExpr(loop_indices.at(0), tv->axis(1)->extent()),
                          tv->fusion()->zeroVal()),
                      tv->axis(2)->extent()),
                  tv->fusion()->zeroVal()),
              tv->axis(3)->extent()),
          loop_indices.at(3));
      auto stop_idx = addExpr(
          mulExpr(
              IrBuilder::addExpr(
                  mulExpr(
                      IrBuilder::addExpr(
                          mulExpr(loop_indices.at(0), tv->axis(1)->extent()),
                          subExpr(
                              tv->axis(1)->extent(), tv->fusion()->oneVal())),
                      tv->axis(2)->extent()),
                  subExpr(tv->axis(2)->extent(), tv->fusion()->oneVal())),
              tv->axis(3)->extent()),
          loop_indices.at(3));

      return andExpr(
          geExpr(start_idx, tv->fusion()->zeroVal()),
          ltExpr(
              stop_idx,
              mulExpr(
                  tv->getLogicalDomain().at(0)->extent(),
                  tv->getLogicalDomain().at(1)->extent())));
    }
  };

  PredicateIndexValidator<GetReference>::validate(&fusion, true);
}

// Make sure non-divisible split to prevent contig predicate indexing
TEST_F(ContigPredicateIndexingTest, NonDivisibleSplit1) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto tv0 = makeSymbolicTensor(2);
  fusion.addInput(tv0);

  // [I0, I1] -> [I0/5, 5, I1]
  auto tv1 = reshape(tv0, {10, 20}, {2, 5, 20});
  fusion.addOutput(tv1);

  auto tv2 = set(tv0);
  fusion.addOutput(tv2);

  // Merge first one of the split outputs with the innermost. Since
  // the split for the reshape is guaranteed to be divisible, the
  // output domain of the last merge should be the one to predicate
  tv1->merge(1, 2);
  tv1->merge(0, 1);

  // While the transformations are the same as tv1, nvFuser doesn't
  // know the corresponding split is actually divisible (note that the
  // split for tv1 is an outer split by a factor of 2, whereas the
  // split for tv2 is an inner split by a factor of 5), so it's
  // considered non-divisible. The resulting predication should be done
  // with each of the two logical domains.
  tv2->split(0, 5);
  // [ceilDiv(I0, 5), 5, I1]
  // Merge first one of the split outputs with the innermost
  tv2->merge(1, 2);
  tv2->merge(0, 1);

  struct GetReference : AbstractGetReference {
    GetReference(const TensorIndexer& indexer, const IdModel& id_model)
        : AbstractGetReference(indexer, id_model) {}

    Val* getInlinePredicate(TensorView* tv) const override {
      std::vector<Val*> loop_indices = getLoopIndices(tv, indexer_, for_loops_);
      auto zero = tv->fusion()->zeroVal();

      // For tv1, since it's fully contiguous, the predicate should be
      // just:
      //
      // i >= 0 && i < I0 * I1
      //
      // where i is the loop index of the sole loop.
      //
      // For tv2, since it isn't contiguous:
      //
      // I0: i / (5 * I1) * 5 + i % (5 * I1) / I1
      // I1: i % (5 * I1) % I1

      auto i = loop_indices.at(0);

      if (tv->name() == 1) {
        return andExpr(
            geExpr(i, zero),
            ltExpr(
                i,
                mulExpr(
                    tv->getLogicalDomain().at(0)->extent(),
                    IrBuilder::mulExpr(
                        tv->getLogicalDomain().at(1)->extent(),
                        tv->getLogicalDomain().at(2)->extent()))));
      } else {
        NVF_ERROR(tv->name() == 2);

        auto i0_ext = tv->getLogicalDomain().at(0)->extent();
        auto i1_ext = tv->getLogicalDomain().at(1)->extent();
        auto five_i1 = mulExpr(createInt(5), i1_ext);
        auto i0 = addExpr(
            mulExpr(divExpr(i, five_i1), createInt(5)),
            divExpr(modExpr(i, five_i1), i1_ext));
        auto i1 = modExpr(modExpr(i, five_i1), i1_ext);
        return andExpr(
            andExpr(
                andExpr(geExpr(i0, zero), ltExpr(i0, i0_ext)),
                geExpr(i1, zero)),
            ltExpr(i1, i1_ext));
      }
    }
  };

  PredicateIndexValidator<GetReference>::validate(&fusion, true);

  EnableOptionsGuard enable_options_guard;
  EnableOptionsGuard::getCurOptions().set(EnableOption::IdModel, {"all"});

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor t0 = at::randn({10, 20}, options);

  KernelExecutor ke;
  ke.compile(&fusion, {t0});
  auto outputs = ke.run({t0});

  testValidate(&fusion, outputs, {t0}, __LINE__, __FILE__);
}

TEST_F(IndexingTest, PerDimLogicalIndices) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto tv0 = makeConcreteTensor({4, 8});
  fusion.addInput(tv0);
  auto tv1 = reshape(tv0, {4, 8}, {32});
  fusion.addOutput(tv1);

  tv1->split(0, 4);
  tv1->split(0, 128);

  tv1->axis(0)->parallelize(ParallelType::BIDx);
  tv1->axis(1)->parallelize(ParallelType::TIDx);
  tv1->axis(2)->parallelize(ParallelType::Unroll);

  auto validate_per_dim_indices =
      [](const std::vector<Expr*>& exprs) -> std::vector<Expr*> {
    class Validator : public kir::IrVisitor {
     public:
      using kir::IrVisitor::handle;
      using kir::IrVisitor::dispatch;

      void handle(LoadStoreOp* ls) override {
        // There should be only one expression of tv1 = Set(tv0).
        NVF_ERROR(ls->in()->isA<kir::TensorIndex>());
        auto tv0 = ls->in()->as<kir::TensorIndex>()->view();
        NVF_ERROR(tv0->name() == 0);

        NVF_ERROR(ls->out()->isA<kir::TensorIndex>());
        auto tv1 = ls->out()->as<kir::TensorIndex>()->view();
        NVF_ERROR(tv1->name() == 1);

        auto indexer = GpuLower::current()->tensorIndexer();
        auto loop_indices = getLoopIndices(tv1, indexer, for_loops_);

        // The logical domains of tv0 and tv1 are [i0, i1] and
        // [i0*i1], respectively. Since tv1 is split twice, the
        // logical domain of tv1 is obtained by traversing them from
        // the three loop iter domains.
        auto tv1_logical_index = addExpr(
            mulExpr(
                addExpr(
                    mulExpr(loop_indices.at(0), createInt(128)),
                    loop_indices.at(1)),
                createInt(4)),
            loop_indices.at(2));

        // The tv0 logical indices are obtained by traversing through
        // the merge for the reshape op.
        std::vector<Val*> tv0_logical_indices{
            divExpr(tv1_logical_index, tv0->getLogicalDomain().at(1)->extent()),
            modExpr(
                tv1_logical_index, tv0->getLogicalDomain().at(1)->extent())};

        // Check tv1 logical indices
        auto actual_tv1_logial_indices =
            Index::getConsumerPerDimLogicalIndex(tv1, for_loops_, {});
        ASSERT_EQ(actual_tv1_logial_indices.size(), 1);
        EXPECT_TRUE(actual_tv1_logial_indices[0]->sameAs(tv1_logical_index))
            << "Validation failure of " << tv1->toString() << " as consumer"
            << "\nRef: " << tv1_logical_index->toInlineString()
            << "\nActual: " << actual_tv1_logial_indices[0]->toInlineString();

        // Check tv0 logical indices
        auto actual_tv0_logial_indices =
            Index::getProducerPerDimLogicalIndex(tv0, tv1, for_loops_, {});
        ASSERT_EQ(actual_tv0_logial_indices.size(), tv0_logical_indices.size());
        for (const auto i : arange(tv0_logical_indices.size())) {
          EXPECT_TRUE(
              actual_tv0_logial_indices[i]->sameAs(tv0_logical_indices[i]))
              << "Validation failure of " << tv0->toString() << " as producer"
              << "\nRef: " << tv0_logical_indices[0]->toInlineString()
              << "\nActual: " << actual_tv0_logial_indices[i]->toInlineString();
        }
      }
    };

    Validator validator;
    validator.handle(exprs);

    return exprs;
  };

  EnableOptionsGuard enable_options_guard;
  EnableOptionsGuard::getCurOptions().set(EnableOption::IdModel, {"all"});
  DisableOptionsGuard disable_options_guard;
  DisableOptionsGuard::getCurOptions().set(DisableOption::ExprSimplify);
  DisableOptionsGuard::getCurOptions().set(DisableOption::IndexHoist);

  GpuLower lower(&fusion);
  lower.passes().insert(
      lower.passes().end(),
      {"validate_per_dim_indices", validate_per_dim_indices});
  lower.run();
}

// Repro of issue #3374
// (https://github.com/NVIDIA/Fuser/issues/3374). Previously failed
// with an error message of:
// Couldn't find allocation mapping for T14_l_float[ iblockIdx.x269{(
// ceilDiv(2, blockDim.x) )}, ithreadIdx.x270{blockDim.x}, iS278{(
// ceilDiv(( ceilDiv(( ceilDiv(( ceilDiv(32768, blockDim.y) ), 8) ),
// 1) ), gridDim.y) )}, iblockIdx.y277{gridDim.y},
// ithreadIdx.y272{blockDim.y}, iUS276{1}, iUR274{8} ] ca_pos( 6 )
// dim: 1 id: iS57{2}
TEST_F(IndexingTest, Issue3374) {
  auto fusion_ptr = std::make_unique<Fusion>();
  auto& fusion = *fusion_ptr;
  FusionGuard fg(fusion_ptr.get());

  std::vector<int64_t> shape1{28, 32768, 2};
  std::vector<int64_t> shape2{32768, 2};
  std::vector<int64_t> shape3{28, 32768, 1};
  std::vector<int64_t> shape4{32768, 56};

  auto tv0 =
      TensorViewBuilder().shape(shape1).contiguity({true, false, true}).build();
  fusion.addInput(tv0);
  auto tv1 = TensorViewBuilder().shape(shape2).contiguity({true, true}).build();
  fusion.addInput(tv1);
  auto tv2 = TensorViewBuilder()
                 .shape(shape3)
                 .contiguity({true, false, std::nullopt})
                 .build();
  fusion.addInput(tv2);
  auto tv3 = TensorViewBuilder()
                 .shape(shape3)
                 .contiguity({true, false, std::nullopt})
                 .build();
  fusion.addInput(tv3);

  auto tv4 = pad(tv2, {fusion.oneVal(), fusion.zeroVal()});
  auto tv5 = pad(tv3, {fusion.zeroVal(), fusion.oneVal()});
  auto tv6 = add(tv4, tv5);
  auto tv7 = broadcast(tv1, {true, false, false});
  auto tv8 = mul(tv7, tv0);
  auto tv9 = add(tv6, tv8);
  auto tv10 = permute(tv9, {1, 0, 2});
  std::vector<Val*> reshape_shape;
  std::transform(
      shape4.begin(),
      shape4.end(),
      std::back_inserter(reshape_shape),
      [](int64_t s) { return IrBuilder::create<Val>(s, DataType::Index); });
  auto tv11 = reshape(tv10, reshape_shape);
  auto tv12 = sum(tv11, {0});
  fusion.addOutput(tv12);
  fusion.addOutput(tv11);
  fusion.addOutput(tv7);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  auto t0 = at::randn(shape1, options);
  auto t1 = at::randn(shape2, options);
  auto t2 = at::randn(shape3, options);
  auto t3 = at::randn(shape3, options);

  FusionExecutorCache executor_cache(std::move(fusion_ptr));
  auto outputs = executor_cache.runFusionWithInputs({t0, t1, t2, t3});

  testValidate(
      executor_cache.fusion(), outputs, {t0, t1, t2, t3}, __LINE__, __FILE__);
}

// Repro of issue #3299
TEST_F(IndexingTest, Issue3299) {
  auto fusion_ptr = std::make_unique<Fusion>();
  auto& fusion = *fusion_ptr;
  FusionGuard fg(fusion_ptr.get());

  std::vector<int64_t> shape1{128000, 1024};
  std::vector<int64_t> shape2{128000, 8, 128};
  std::vector<int64_t> shape3{8, 4, 128000, 128};
  std::vector<int64_t> shape4{32, 128000, 128};

  auto tv0 = makeContigConcreteTensor(shape1);
  fusion.addInput(tv0);
  auto tv1 = reshape(tv0, shape1, shape2);
  auto tv2 = permute(tv1, {1, 0, 2});
  auto tv3 = broadcast(tv2, {false, true, false, false});
  auto tv4 = expand(
      tv3,
      {IrBuilder::create<Val>(shape3[0], DataType::Index),
       IrBuilder::create<Val>(shape3[1], DataType::Index),
       IrBuilder::create<Val>(shape3[2], DataType::Index),
       IrBuilder::create<Val>(shape3[3], DataType::Index)});
  auto tv5 = reshape(tv4, shape3, shape4);
  fusion.addOutput(tv5);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  auto t0 = at::randn(shape1, options);

  FusionExecutorCache executor_cache(std::move(fusion_ptr));
  auto outputs = executor_cache.runFusionWithInputs({t0});

  testValidate(executor_cache.fusion(), outputs, {t0}, __LINE__, __FILE__);
}

TEST_F(IndexingTest, ResizeRotation) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  const int64_t i0 = 32;

  EnableOptionsGuard enable_options_guard;
  EnableOptionsGuard::getCurOptions().set(EnableOption::IdModel, {"all"});

  auto zero = fusion.zeroVal();

  // concrete shapes to avoid dynamic Fusion
  auto tv0 = makeContigConcreteTensor({i0});
  fusion.addInput(tv0);

  // left half
  auto tv1 = slice(tv0, {{zero, IrBuilder::create<Val>(i0 / 2)}});
  // right half
  auto tv2 = slice(
      tv0, {{IrBuilder::create<Val>(i0 / 2), IrBuilder::create<Val>(i0)}});

  // Rotation
  auto tv3 = cat({tv2, tv1}, 0);

  auto tv4 = add(tv0, tv3);

  fusion.addOutput(tv4);

  // Some of the scheduling tools such as scheduleLoopDomain are
  // supposed to take care of the following transformations
  // automatically, however, it needs a workaround for the cyclic
  // graph pattern. For now, they are manually scheduled.

  // tv1
  {
    auto tv1_padded = tv3->definition()->input(1)->as<TensorView>();
    auto tv1_pad_resize = dynamic_cast<Resize*>(
        tv1_padded->getLogicalDomain().at(0)->definition());
    ASSERT_NE(tv1_pad_resize, nullptr);
    auto loop_domain = tv1->getLogicalDomain();
    auto padded_id = IterDomain::resize(
        loop_domain[0],
        tv1_pad_resize->leftExpand(),
        tv1_pad_resize->rightExpand());
    loop_domain[0] = padded_id;
    tv1->setLoopDomain(loop_domain);
  }

  // tv2
  {
    auto tv2_padded = tv3->definition()->input(0)->as<TensorView>();
    auto tv2_pad_resize = dynamic_cast<Resize*>(
        tv2_padded->getLogicalDomain().at(0)->definition());
    ASSERT_NE(tv2_pad_resize, nullptr);
    auto loop_domain = tv2->getLogicalDomain();
    auto padded_id = IterDomain::resize(
        loop_domain[0],
        tv2_pad_resize->leftExpand(),
        tv2_pad_resize->rightExpand());
    loop_domain[0] = padded_id;
    tv2->setLoopDomain(loop_domain);
  }

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  auto t0 = at::randn({i0}, options);

  KernelExecutor ke;
  ke.compile(&fusion, {t0});
  auto outputs = ke.run({t0});

  testValidate(&fusion, outputs, {t0}, __LINE__, __FILE__);
}

TEST_F(PredicateIndexingTest, VectorizedResizeRotation) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  const int64_t i0 = 32;

  EnableOptionsGuard enable_options_guard;
  EnableOptionsGuard::getCurOptions().set(EnableOption::IdModel, {"all"});

  auto zero = fusion.zeroVal();

  // concrete shapes to avoid dynamic Fusion
  auto tv0 = makeContigConcreteTensor({i0});
  fusion.addInput(tv0);

  auto tv1 = set(tv0);
  // left half
  auto tv2 = slice(tv1, {{zero, IrBuilder::create<Val>(i0 / 2)}});

  auto tv3 = set(tv0);
  // right half
  auto tv4 = slice(
      tv3, {{IrBuilder::create<Val>(i0 / 2), IrBuilder::create<Val>(i0)}});

  // Rotation
  auto tv5 = cat({tv4, tv2}, 0);

  auto tv6 = add(tv0, tv5);

  fusion.addOutput(tv6);

  for (Expr* expr : fusion.exprs()) {
    if (expr->isOneOf<SliceOp, PadOp>()) {
      scheduler_tools::propagateResizeToInputs(expr);
    }
  }

  for (auto tv : fusion.allTvs()) {
    if (tv->isFusionInput()) {
      continue;
    }

    tv->split(0, 4);
  }

  tv1->axis(-1)->parallelize(ParallelType::Vectorize);

  inlineMost();

  struct GetReference : AbstractGetReference {
    GetReference(const TensorIndexer& indexer, const IdModel& id_model)
        : AbstractGetReference(indexer, id_model) {}

    Val* getInlinePredicate(TensorView* tv) const override {
      if (tv->name() != 1) {
        return nullptr;
      }

      if (for_loops_.back()->iter_domain()->getParallelType() !=
          ParallelType::Vectorize) {
        return nullptr;
      }

      std::vector<Val*> loop_indices = getLoopIndices(tv, indexer_, for_loops_);

      Val* zero = tv->fusion()->zeroVal();

      auto second_resize = dynamic_cast<Resize*>(
          tv->axis(0)->definition()->input(0)->definition());
      EXPECT_NE(second_resize, nullptr);

      auto start_idx = addExpr(
          IrBuilder::addExpr(
              mulExpr(loop_indices.at(0), tv->axis(1)->extent()), zero),
          IrBuilder::negExpr(second_resize->leftExpand()));
      auto stop_idx = addExpr(
          IrBuilder::addExpr(
              mulExpr(loop_indices.at(0), tv->axis(1)->extent()), createInt(3)),
          IrBuilder::negExpr(second_resize->leftExpand()));

      return andExpr(
          geExpr(start_idx, tv->fusion()->zeroVal()),
          ltExpr(stop_idx, tv->getLogicalDomain().at(0)->extent()));
    }
  };

  PredicateIndexValidator<GetReference>::validate(&fusion, false);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  auto t0 = at::randn({i0}, options);

  KernelExecutor ke;
  ke.compile(&fusion, {t0});
  auto outputs = ke.run({t0});

  testValidate(&fusion, outputs, {t0}, __LINE__, __FILE__);
}

// Repro of issue #3505. The indexing WAR for resize triggered an
// assertion due to loop promotion.
TEST_F(IndexingTest, Issue3505Repro1) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  const int64_t i0 = 2;
  const int64_t i1 = 4;
  const int64_t i2 = 8;
  const auto zero = fusion.zeroVal();

  EnableOptionsGuard enable_options_guard;
  EnableOptionsGuard::getCurOptions().set(EnableOption::IdModel, {"all"});

  auto tv0 = makeContigConcreteTensor({i1, i2});
  fusion.addInput(tv0);
  auto tv1 = makeContigConcreteTensor({i0, i1 / 2, i2 / 2});
  fusion.addInput(tv1);

  // One slice can reproduce the error but just to trigger the
  // reachability check between multiple resize ops
  auto tv2 = slice(
      tv0,
      {{zero, IrBuilder::create<Val>(i1 / 2)},
       {zero, IrBuilder::create<Val>(i2 / 2)}});
  auto tv3 = broadcast(tv2, {true, false, false});
  auto tv4 = add(tv1, tv3);
  fusion.addOutput(tv4);

  for (auto tv : {tv2, tv3, tv4}) {
    tv->flatten();
  }
  inlineMost();

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  auto t0 = at::randn({i1, i2}, options);
  auto t1 = at::randn({i0, i1 / 2, i2 / 2}, options);

  KernelExecutor ke;
  ke.compile(&fusion, {t0, t1});
  auto outputs = ke.run({t0, t1});

  testValidate(&fusion, outputs, {t0, t1}, __LINE__, __FILE__);
}

// Another repro of issue #3505
TEST_F(IndexingTest, Issue3505Repro2) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  const int64_t i0 = 8;
  const int64_t i1 = 2;
  const auto zero = fusion.zeroVal();

  EnableOptionsGuard enable_options_guard;
  EnableOptionsGuard::getCurOptions().set(EnableOption::IdModel, {"all"});

  auto tv0 = makeContigConcreteTensor({i0});
  fusion.addInput(tv0);
  auto tv1 = makeContigConcreteTensor({i1, i0 / 2});
  fusion.addInput(tv1);

  // Left half
  auto tv2 = slice(tv0, {{zero, IrBuilder::create<Val>(i0 / 2)}});
  // Right half
  auto tv3 = slice(
      tv0, {{IrBuilder::create<Val>(i0 / 2), IrBuilder::create<Val>(i0)}});

  // The two inputs of this add expression have a resize of the same
  // ID, but this should not mean the resize war path is required.
  auto tv4 = add(tv2, tv3);
  auto tv5 = broadcast(tv4, {true, false});
  auto tv6 = add(tv1, tv5);
  fusion.addOutput(tv6);

  // Make loop promotion required
  for (auto tv : {tv2, tv3, tv4, tv5, tv6}) {
    tv->flatten();
  }
  inlineMost();

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  auto t0 = at::randn({i0}, options);
  auto t1 = at::randn({i1, i0 / 2}, options);

  KernelExecutor ke;
  ke.compile(&fusion, {t0, t1});
  auto outputs = ke.run({t0, t1});

  testValidate(&fusion, outputs, {t0, t1}, __LINE__, __FILE__);
}

TEST_F(IndexingTest, AlmostExactIndexingUpdate) {
  EnableOptionsGuard enable_options_guard;
  EnableOptionsGuard::getCurOptions().set(EnableOption::IdModel, {"all"});

  Fusion fusion;
  FusionGuard fg(&fusion);

  auto tv0 = makeContigConcreteTensor({4, 8});
  fusion.addInput(tv0);

  auto tv1 = slice(
      tv0,
      {{IrBuilder::create<Val>(1L), IrBuilder::create<Val>(2L)},
       {IrBuilder::create<Val>(0L), tv0->axis(1)->extent()}});

  fusion.addOutput(tv1);

  // [b0, i1]
  tv1->split(-1, 5);
  // [b0, i1/5, 5]
  tv1->split(-1, 3);
  // [b0, i1/5, 5/3, 3]
  tv1->merge(0, -1);
  // [b0*i1/5*3, 5/3]
  tv1->split(0, 2);
  // [b0*i1/5*3/2, 2, 5/3]

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  auto t0 = at::randn({4, 8}, options);

  KernelExecutor ke;
  ke.compile(&fusion, {t0});
  auto outputs = ke.run({t0});

  testValidate(&fusion, outputs, {t0}, __LINE__, __FILE__);
}

// Small repro of
// https://github.com/NVIDIA/Fuser/issues/3688. Broadcast logical
// IDs may not be reachable from loop IDs, thus the indexing for the
// logical IDs of the pad output failed.
TEST_F(IndexingTest, BroadcastLogicalDomainIndexing) {
  auto fusion_ptr = std::make_unique<Fusion>();
  auto& fusion = *fusion_ptr;
  FusionGuard fg(fusion_ptr.get());

  std::vector<int64_t> shape1{1, 32};
  std::vector<int64_t> shape2{8, 34};

  auto tv0 = makeConcreteTensor(shape1);
  fusion.addInput(tv0);
  auto tv1 = makeConcreteTensor(shape2);
  fusion.addInput(tv1);

  auto tv2 = pad(tv0, {fusion.oneVal(), fusion.oneVal()});
  auto tv3 = add(tv2, tv1);
  fusion.addOutput(tv3);

  tv2->inlineAt(-1);

  tv3->axis(0)->parallelize(ParallelType::BIDx);
  tv3->axis(1)->parallelize(ParallelType::TIDx);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  auto t0 = at::randn(shape1, options);
  auto t1 = at::randn(shape2, options);

  KernelExecutor ke;
  ke.compile(&fusion, {t0, t1});
  auto outputs = ke.run({t0, t1});
  testValidate(&fusion, outputs, {t0, t1}, __LINE__, __FILE__);
}

TEST_F(IndexingTest, Rng) {
  EnableOptionsGuard enable_options_guard;
  EnableOptionsGuard::getCurOptions().set(EnableOption::IdModel, {"all"});

  auto fusion_ptr = std::make_unique<Fusion>();
  auto& fusion = *fusion_ptr;
  FusionGuard fg(fusion_ptr.get());

  Val* i = IrBuilder::create<Val>(DataType::Int);
  fusion.addInput(i);

  auto tv0 = randn(
      {i},
      DataType::Float,
      /*Val* philox_seed=*/fusion.zeroVal(),
      /*Val* philox_offset=*/fusion.zeroVal());

  fusion.addOutput(tv0);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);

  FusionExecutorCache executor_cache(std::move(fusion_ptr));
  auto outputs = executor_cache.runFusionWithInputs({1});

  at::manual_seed(0);
  at::Tensor randn_sample = at::randn({1}, options);

  testValidate(&fusion, outputs, {1}, {randn_sample}, __LINE__, __FILE__);
}

// Loops should be annotated with "pragma unroll" when their indices
// are used for indexing of register tensors. This is one example a
// loop may not be unrolled.
TEST_F(IndexingTest, StaticIndexing) {
  EnableOptionsGuard enable_options_guard;
  EnableOptionsGuard::getCurOptions().set(EnableOption::IdModel, {"all"});

  auto fusion_ptr = std::make_unique<Fusion>();
  auto& fusion = *fusion_ptr;
  FusionGuard fg(fusion_ptr.get());

  auto tv0 = makeSymbolicTensor(1);
  fusion.addInput(tv0);
  auto tv1 = set(tv0);
  auto tv2 = set(tv1);
  fusion.addOutput(tv2);

  tv1->split(0, 4);
  tv2->split(0, 4);

  tv1->inlineAt(1);
  // Unswitched loops are not unrolled by default. This should be
  // overridden because tv1 is a register tensor.
  tv1->axis(1)->parallelize(ParallelType::Unswitch);

  // Check if tv1's innermost loop is required to be unrolled
  class Validator : public kir::IrVisitor {
   public:
    using kir::IrVisitor::handle;

    void handle(LoadStoreOp* ldst) override {
      if (ir_utils::getTvOutput(ldst)->name() == 1) {
        ASSERT_FALSE(for_loops_.empty());
        EXPECT_TRUE(for_loops_.back()->isUnrollRequired());
      }
    }
  };

  GpuLower lower(&fusion);
  kir::Kernel* kernel = lower.run();
  Validator validator;
  validator.handle(kernel->topLevelExprs());
}

// Repro of the issue with trival mapping of size-one IDs (PR #4214)
TEST_F(PredicateIndexingTest, NonTrivialSizeOneDomain) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto tv0 = makeContigConcreteTensor({8});
  fusion.addInput(tv0);

  auto tv1 = sum(tv0, {0});

  fusion.addOutput(tv1);

  // [r0(8)]
  tv1->split(0, 10);
  // [r1(1), r2(10)]
  tv1->split(0, 4);
  // [r3(1), r4(4), r2(10)]

  // The predicate of tv1 is given by the index of its sole logical
  // ID, r0. Suppose the three loop IDs get loop indies of i0, i1 and
  // i2, respectively, the correct predicate index is (i1 * 10 + i2).
  //
  // Here, if r1 and r3 were mapped, which is not unreasonable given
  // they have the same extent, the r1 index would be the same as that of
  // r3, which would be just 0. The index of r0 thus would be just the same
  // as r2, i.e., i2, which is not correct.
  //
  // This test ensures the index of r4 is indeed used in the predicate
  // correctly.

  struct GetReference : AbstractGetReference {
    GetReference(const TensorIndexer& indexer, const IdModel& id_model)
        : AbstractGetReference(indexer, id_model) {}

    Val* getInlinePredicate(TensorView* tv) const override {
      // [i0, i1, i2]
      std::vector<Val*> loop_indices = getLoopIndices(tv, indexer_, for_loops_);
      // i1 * 10 + i2
      Val* idx = addExpr(
          mulExpr(loop_indices.at(1), createInt(10)), loop_indices.at(2));
      Val* zero = tv->fusion()->zeroVal();
      return andExpr(
          geExpr(idx, zero),
          ltExpr(idx, tv->getLogicalDomain().at(0)->extent()));
    }
  };

  PredicateIndexValidator<GetReference>::validate(&fusion, false);

  EnableOptionsGuard::getCurOptions().set(EnableOption::IdModel, {"all"});

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor t0 = at::randn({8}, options);

  KernelExecutor ke;
  ke.compile(&fusion, {t0});
  auto outputs = ke.run({t0});

  testValidate(&fusion, outputs, {t0}, __LINE__, __FILE__);
}

// Simple repro of issue #4218
TEST_F(PredicateIndexingTest, AdditionalNonDivisibleSplit) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  EnableOptionsGuard::getCurOptions().set(EnableOption::IdModel, {"all"});

  auto tv0 = makeContigConcreteTensor({8});
  fusion.addInput(tv0);

  auto tv1 = sum(tv0, {0});

  fusion.addOutput(tv1);

  // [r0(8)]
  tv1->split(0, 1);
  // [r1(8), r2(1)]
  tv1->split(1, 4);
  // [r1(8), r3(1), r4(4)]

  struct GetReference : AbstractGetReference {
    GetReference(const TensorIndexer& indexer, const IdModel& id_model)
        : AbstractGetReference(indexer, id_model) {}

    Val* getInlinePredicate(TensorView* tv) const override {
      std::vector<Val*> loop_indices = getLoopIndices(tv, indexer_, for_loops_);
      auto zero = tv->fusion()->zeroVal();
      auto one = tv->fusion()->oneVal();

      auto i = loop_indices.at(0);
      auto k = loop_indices.at(2);

      if (tv->name() == 1) {
        // i >= 0 && i < 8 && k < 1
        return andExpr(
            andExpr(geExpr(i, zero), ltExpr(i, createInt(8))), ltExpr(k, one));
      } else {
        return nullptr;
      }
    }
  };

  PredicateIndexValidator<GetReference>::validate(&fusion, true);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor t0 = at::randn({8}, options);

  KernelExecutor ke;
  ke.compile(&fusion, {t0});
  auto outputs = ke.run({t0});

  testValidate(&fusion, outputs, {t0}, __LINE__, __FILE__);
}

TEST_F(PredicateIndexingTest, AdditionalNonDivisibleSplitAfterDivisibleSplit) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  EnableOptionsGuard::getCurOptions().set(EnableOption::IdModel, {"all"});

  auto tv0 = makeContigConcreteTensor({8});
  fusion.addInput(tv0);

  auto tv1 = sum(tv0, {0});

  fusion.addOutput(tv1);

  // [r0(8)]
  tv1->split(0, 1);
  // [r1(8), r2(1)]
  tv1->split(1, 1);
  // [r1(8), r3(1), r4(1)]
  tv1->split(2, 4);
  // [r1(8), r3(1), r5(1), r6(4)]

  struct GetReference : AbstractGetReference {
    GetReference(const TensorIndexer& indexer, const IdModel& id_model)
        : AbstractGetReference(indexer, id_model) {}

    Val* getInlinePredicate(TensorView* tv) const override {
      std::vector<Val*> loop_indices = getLoopIndices(tv, indexer_, for_loops_);
      auto zero = tv->fusion()->zeroVal();
      auto one = tv->fusion()->oneVal();

      auto i0 = loop_indices.at(0);
      auto i3 = loop_indices.at(3);

      if (tv->name() == 1) {
        // i0 >= 0 && i0 < 8 && i3 < 1
        return andExpr(
            andExpr(geExpr(i0, zero), ltExpr(i0, createInt(8))),
            ltExpr(i3, one));
      } else {
        return nullptr;
      }
    }
  };

  PredicateIndexValidator<GetReference>::validate(&fusion, true);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor t0 = at::randn({8}, options);

  KernelExecutor ke;
  ke.compile(&fusion, {t0});
  auto outputs = ke.run({t0});

  testValidate(&fusion, outputs, {t0}, __LINE__, __FILE__);
}

AbstractTensor scheduleLdStMatrixBase(TensorView* tv) {
  // Assume the input TensorView is block tiled. e.g., The last two iterDomains
  // are the warp tile except for k dimension.
  // The CTA tile is (128, 256).
  // The Warp tile is (64, 256).
  // The TMA box is (64, 64).
  // The LdStMatrix.x4 tile is (16, 16).
  // The core matrix for wgmma and LdStMatrix is (8, 8).

  AbstractTensor abstract_tensor(tv->getLoopDomain());
  // (GM, GN, cta_m(2), cta_n(1), m(64), n(256))

  // Split by TMA shared memory box
  abstract_tensor.split(-1, 64);
  abstract_tensor.reorder({{-2, -3}, {-3, -2}});
  // (GM, GN, cta_m(2), cta_n(1), no(4), m(64), ni(64))

  // Split by (16, 16) matrix for LdStMatrix.x4
  abstract_tensor.split(-2, 16);
  abstract_tensor.split(-1, 16);
  abstract_tensor.reorder({{-2, -3}, {-3, -2}});
  // (GM, GN, cta_m(2), cta_n(1), no(4), mo(4), nio(4), mi(16), nii(16))

  return abstract_tensor;
}

AbstractTensor scheduleLdStMatrixSharedMemory(
    const AbstractTensor& base_tensor) {
  // Assume the input TensorView is block tiled. e.g., The last two iterDomains
  // are the warp tile except for k dimension.
  // The CTA tile is (128, 256).
  // The Warp tile is (64, 256).
  // The TMA box is (64, 64).
  // The LdStMatrix.x4 tile is (16, 16).
  // The core matrix for wgmma and LdStMatrix is (8, 8).

  // Initial Abstract Tensor
  AbstractTensor abstract_tensor(base_tensor);
  // (GM, GN, cta_m(2), cta_n(1), no(4), mo(4), nio(4), mi(16), nii(16))
  // Omit (GM, GN, cta_m(2), cta_n(1)) after this for brevity.

  // For shared memory addressing, each thread specifies a row for each (8, 8)
  // matrix. e.g., For stmatrix.x4, 32 threads move a (16, 16) matrix.

  // Inside the tile box [16, 16], we can think of it as 4 8x8 tiles:
  // *****************
  // *       *       *
  // *       *       *
  // *  T0   *  T2   *
  // *       *       *
  // *       *       *
  // *****************
  // *       *       *
  // *       *       *
  // *  T1   *  T3   *
  // *       *       *
  // *       *       *
  // *****************

  // Split inner-dimension by 8 to traverse the rows of the (8, 8) matrices.
  abstract_tensor.split(-1, 8);
  // (no(4), mo(4), nio(4), mi(16), niio(2), niii(8))

  // The tile is stored in row-major order, so issue four stmatrix.x4
  // operations along the M dimension for a 128 thread warp group.
  // Also, traverse along 16 rows first before moving along column dimension.
  abstract_tensor.reorder({{-5, -4}, {-4, -5}, {-3, -2}, {-2, -3}});
  // (no(4), nio(4), mo(4), niio(2), mi(16), niii(8))

  abstract_tensor.merge(-4, -3);
  abstract_tensor.merge(-3, -2);
  // (no(4), nio(4), (niio * mo * mi)(128), niii(8))

  // Merge no and nio to create a single serial IterDomain
  // This ^^^ is an artifact of matmul scheduling functions.
  abstract_tensor.merge(-4, -3);
  // (no * nio)(16), (niio * mo * mi)(128), niii(8))

  return abstract_tensor;
}

AbstractTensor scheduleLdStMatrixRegisters(const AbstractTensor& base_tensor) {
  // Assume the input TensorView is block tiled. e.g., The last two iterDomains
  // are the warp tile except for k dimension.
  // The CTA tile is (128, 256).
  // The Warp tile is (64, 256).
  // The TMA box is (64, 64).
  // The LdStMatrix.x4 tile is (16, 16).
  // The core matrix for wgmma and LdStMatrix is (8, 8).

  // Initial Abstract Tensor
  AbstractTensor abstract_tensor(base_tensor);
  // (GM, GN, cta_m(2), cta_n(1), no(4), mo(4), nio(4), mi(16), nii(16))
  // Omit (GM, GN, cta_m(2), cta_n(1)) after this for brevity.

  // Split (16, 16) matrix into four (8, 8) sub-matrices
  abstract_tensor.split(-2, 8);
  abstract_tensor.split(-1, 8);

  // Each register handles two adjacent elements.
  abstract_tensor.split(-1, 2);

  // The four (8, 8) sub-matrices are traversed in this order to follow the
  // register layout for wgmma accumulator matrix.
  // *****************
  // *       *       *
  // *       *       *
  // *   0   *   2   *
  // *       *       *
  // *       *       *
  // *****************
  // *       *       *
  // *       *       *
  // *   1   *   3   *
  // *       *       *
  // *       *       *
  // *****************
  abstract_tensor.reorder({{-5, -2}, {-4, -5}, {-2, -4}});
  // (no(4), mo(4), nio(4), mii(8), niiio(4), niio(2), mio(2), niiii(2))

  // For an (16, 16) matrix, each register will hold 8 values. The LdStMatrix
  // instruction will load or store these values with a single instruction. We
  // remove this serial for-loop from the kernel by merging the last three
  // iterDomains together and then applying ParallelType::Vectorize.
  abstract_tensor.merge(-2, -1);
  abstract_tensor.merge(-2, -1);
  // (no(4), mo(4), nio(4), mii(8), niiio(4), (niio * mio * niiii)(8))

  // Reorder iterDomains so the serial IterDomain for (CTA_N / TMA_N) and
  // (TMA_N and LDST_N) are adjacent.
  abstract_tensor.reorder({{-5, -4}, {-4, -5}});

  // Four LdStMatrix.x4 instructions are issued simultaneously to process
  // (64, 16) tile. Merge mio, miii, and niiio iterDomains together.
  abstract_tensor.merge(-4, -3);
  abstract_tensor.merge(-3, -2);
  // (no(4), nio(4), (mo * mii * niiio)(128), (niio * mio * niiii)(8))

  // Merge no and nio to create a single serial IterDomain
  // This ^^^ is an artifact of matmul scheduling functions.
  abstract_tensor.merge(-4, -3);
  // (no * nio)(16), (mo * mii * niiio)(128), (niio * mio * niiii)(8))

  return abstract_tensor;
}

TEST_F(IndexingTest, LdStMatrix) {
  const auto dtype = DataType::BFloat16;

  // Fusion Definition
  Fusion fusion;
  FusionGuard fg(&fusion);

  TensorView* tv0 = makeContigConcreteTensor({-1, -1}, dtype); // M, K
  fusion.addInput(tv0);
  TensorView* tv1 = set(tv0);
  fusion.addOutput(tv1);

  // ===========================================================================

  // Constants
  constexpr int64_t cta_m = 128;
  constexpr int64_t cta_n = 256;
  constexpr int64_t warp_m = 64;
  constexpr int64_t warp_n = 256;
  constexpr int64_t ldst_matrix_tile_m = 16;
  constexpr int64_t ldst_matrix_tile_n = 16;
  fusion.manage("ldst_matrix_m_tile", ldst_matrix_tile_m);
  fusion.manage("ldst_matrix_n_tile", ldst_matrix_tile_n);
  fusion.manage("ldst_matrix_m_smem", warp_m);
  fusion.manage("ldst_matrix_n_smem", warp_n);

  // ===========================================================================
  // Create cache intermediate TensorViews
  // The definition for tv0_smem is tma load, which moves data from shared to
  // global memory.
  TensorView* tv0_smem = tv0->cacheAfter();
  tv0_smem->definition()->as<LoadStoreOp>()->setOpType(
      LoadStoreOpType::CpAsyncBulkTensorTile);
  tv0_smem->setMemoryType(MemoryType::Shared);

  // The definition for tv0_reg is ldmatrix, which moves data from shared memory
  // to registers.
  TensorView* tv0_reg = tv0_smem->cacheAfter();
  tv0_reg->definition()->as<LoadStoreOp>()->setOpType(
      LoadStoreOpType::LdMatrix);

  // The definition for tv1_smem is stmatrix, which moves data from registers to
  // shared memory.
  TensorView* tv1_smem = tv1->cacheBefore();
  tv1_smem->definition()->as<LoadStoreOp>()->setOpType(
      LoadStoreOpType::StMatrix);
  tv1_smem->setMemoryType(MemoryType::Shared);

  // The definition for tv1 is tma store, which moves data from shared to global
  // memory.
  tv1->definition()->as<LoadStoreOp>()->setOpType(
      LoadStoreOpType::CpAsyncBulkTensorTile);

  // ===========================================================================
  // General scheduling
  // Tile reference by cta_tile and warp_tile
  // (M, N)
  tv1->split(0, cta_m);
  tv1->split(-1, cta_n);
  tv1->reorder({{-2, -3}, {-3, -2}});
  // (GM, GN, cta_m(128), cta_n(256))

  tv1->split(-2, warp_m);
  tv1->split(-1, warp_n);
  tv1->reorder({{-2, -3}, {-3, -2}});
  // (GM, GN, cta_m(2), cta_n(1), warp_m(64), warp_n(256))

  TransformPropagator propagator(tv1);
  MaxLogicalDomainInfoSpanningTree(tv1).traverse(&propagator);

  tv1->axis(0)->parallelize(ParallelType::BIDx);
  tv1->axis(1)->parallelize(ParallelType::BIDy);
  tv1->axis(2)->parallelize(ParallelType::TIDy);
  scheduler_utils::parallelizeAllLike(tv1);

  // ===========================================================================
  // Schedule shared memory tensors using TMA Load and Store

  // Schedule output from TMA Load
  MmaInputSmemSwizzle input_swizzle =
      mma_utils::tmaSwizzleSharedMemory(tv0_smem);
  mma_utils::MmaSwizzler::scheduleTMALoadForMma(tv0_smem, input_swizzle);

  // Schedule global memory output from TMA Store
  MmaInputSmemSwizzle output_swizzle =
      mma_utils::tmaSwizzleSharedMemory(tv1_smem);
  mma_utils::scheduleTMAStoreForMmaOutput(tv1, output_swizzle);

  // ===========================================================================
  // Schedule register tensors using LdMatrix and StMatrix

  // NOTE: When using a custom allocation domain, all iterDomains to the left
  // of the computeAt position must exist in the loop domain. The utility
  // function for applying swizzle to TMA LoadStoreOp creates the appropriate
  // TMA Box. Creating the same TMA Box in the loop domain via AbstractTensor
  // allows for inlining iterDomains that are not identical, causing an
  // assertion in indexing pass.

  // Move data from tv0_reg to tv1_smem using StMatrix
  AbstractTensor tv1_smem_base_tensor = scheduleLdStMatrixBase(tv1_smem);
  AbstractTensor tv1_smem_abstract_tensor =
      scheduleLdStMatrixRegisters(tv1_smem_base_tensor);
  // Create tma store allocation domain with swizzle
  if (output_swizzle != MmaInputSmemSwizzle::None) {
    mma_utils::scheduleTMAStoreForMmaOutput(tv1_smem, output_swizzle);
  }
  tv1_smem->setLoopDomain(tv1_smem_abstract_tensor.as<IterDomain*>());
  // (GM(BDX), GN(BDY), cta_m(2), cta_n(1), (no * nio)(16), (mo * mii *
  // niiio)(128), (niio * mio * niiii)(8))

  // tv1_smem is the consumer for stmatrix. tv0_reg is the consumer.
  std::vector<IterDomain*> tv1_smem_stmatrix =
      scheduleLdStMatrixSharedMemory(tv1_smem_base_tensor).as<IterDomain*>();
  tv1_smem_stmatrix.at(tv1_smem_stmatrix.size() - 2)
      ->parallelize(ParallelType::TIDx);
  tv1_smem_stmatrix.at(tv1_smem_stmatrix.size() - 1)
      ->parallelize(ParallelType::Vectorize);
  tv1_smem->setAlternateLoopDomain(tv1_smem_stmatrix);

  // Use ParallelType::TIDx to launch four StMatrix.x4 in parallel.
  // Use ParallelType::Vectorize because StMatrix.x4 stores eight elements per
  // thread per operation.
  tv1_smem->axis(-2)->parallelize(ParallelType::TIDx);
  tv1_smem->axis(-1)->parallelize(ParallelType::Vectorize);
  // (GM(BDX), GN(BDY), cta_m(2)(TDY), cta_n(1), (no * nio)(16), (mo * mii *
  // niiio)(128)(TDX), (niio * mio * niiii)(8)(V))

  // ===========================================================================

  // Move data from tv0_smem to tv0_reg using LdMatrix
  AbstractTensor tv0_reg_base_tensor = scheduleLdStMatrixBase(tv0_reg);
  AbstractTensor tv0_reg_abstract_tensor =
      scheduleLdStMatrixRegisters(tv0_reg_base_tensor);
  tv0_reg->setLoopDomain(tv0_reg_abstract_tensor.as<IterDomain*>());
  // (GM(BDX), GN(BDY), cta_m(2), cta_n(1), (no * nio)(16), (mo * mii *
  // niiio)(128), (niio * mio * niiii)(8))

  std::vector<IterDomain*> tv0_reg_ldmatrix =
      scheduleLdStMatrixSharedMemory(tv0_reg_base_tensor).as<IterDomain*>();
  tv0_reg_ldmatrix.at(tv0_reg_ldmatrix.size() - 2)
      ->parallelize(ParallelType::TIDx);
  tv0_reg_ldmatrix.at(tv0_reg_ldmatrix.size() - 1)
      ->parallelize(ParallelType::Vectorize);
  tv0_reg->setAlternateLoopDomain(tv0_reg_ldmatrix);
  // tv0_reg is the consumer for ldmatrix and tv0_smem is the producer.

  // Set allocation domain according to loop domain
  tv0_reg->setAllocationDomain(
      tv0_reg->getLoopDomain(), /*new_contiguity=*/true);

  // Use ParallelType::TIDx to launch four LdMatrix.x4 in parallel.
  // Use ParallelType::Vectorize because LdMatrix.x4 stores eight elements per
  // thread per operation.
  tv0_reg->axis(-2)->parallelize(ParallelType::TIDx);
  tv0_reg->axis(-1)->parallelize(ParallelType::Vectorize);
  // (GM(BDX), GN(BDY), cta_m(2)(TDY), cta_n(1), (no * nio)(16), (mo * mii *
  // niiio)(128)(TDX), (niio * mio * niiii)(8)(V))

  // ===========================================================================

  inlineMost();

  // ===========================================================================

  constexpr int dim0 = 8192, dim1 = 8192;
  auto options = at::TensorOptions().dtype(at::kBFloat16).device(at::kCUDA, 0);
  at::Tensor at_tv0 = at::randn({dim0, dim1}, options);

  KernelExecutor ke;
  ke.compile(&fusion, {at_tv0});
  kir::Kernel* kernel = ke.compiledKernel()->kernel();
  ASSERT_TRUE(kernel != nullptr);
  auto cg_outputs = ke.run({at_tv0});
  NVF_CHECK(at::allclose(cg_outputs[0].as<at::Tensor>(), at_tv0));

  struct GetReference : AbstractGetReference {
    GetReference(const TensorIndexer& indexer, const IdModel& id_model)
        : AbstractGetReference(indexer, id_model) {}
    std::string getLinearIndexString(TensorView* tv, TensorView* maybe_consumer)
        const override {
      Expr* def = tv->definition();
      if (def == nullptr) {
        return std::string();
      }

      bool is_output_ldmatrix = ir_utils::isLdMatrixOp(def);
      bool is_input_ldmatrix =
          std::any_of(tv->uses().begin(), tv->uses().end(), [](Expr* e) {
            return ir_utils::isLdMatrixOp(e);
          });
      if (!(is_output_ldmatrix || is_input_ldmatrix)) {
        return std::string();
      }

      if (is_output_ldmatrix) {
        return std::string("0");
      }

      // Skip consumer case
      if (maybe_consumer == nullptr) {
        return std::string();
      }

      return std::string(
          R"(( ( toSmem(( getMetaData(T2) )) ) + ( ( ( ( ( ( ( threadIdx.y * 16384 ) + ( ( i234 / 4 ) * 4096 ) ) + ( ( ( ( ( ( threadIdx.x / 16 ) / 2 ) * 16 ) + ( threadIdx.x % 16 ) ) / 8 ) * 512 ) ) + ( ( ( ( ( ( threadIdx.x / 16 ) / 2 ) * 16 ) + ( threadIdx.x % 16 ) ) % 8 ) * 64 ) ) + ( ( ( ( ( ( ( threadIdx.x / 16 ) / 2 ) * 16 ) + ( threadIdx.x % 16 ) ) % 8 ) ^ ( ( ( ( i234 % 4 ) * 16 ) + ( ( ( ( threadIdx.x / 16 ) % 2 ) * 8 ) + 0 ) ) / 8 ) ) * 8 ) ) + ( ( ( ( i234 % 4 ) * 16 ) + ( ( ( ( threadIdx.x / 16 ) % 2 ) * 8 ) + 0 ) ) % 8 ) ) * 2 ) ))");
    }
  };
  IndexValidator<GetReference>::validate(&fusion, false);
}

} // namespace nvfuser
