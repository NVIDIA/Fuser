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

#include <abstract_tensor.h>
#include <fusion.h>
#include <id_model/id_model.h>
#include <id_model/indexing.h>
#include <id_model/indexing_utils.h>
#include <id_model/to_string.h>
#include <inlining.h>
#include <ir/builder.h>
#include <kernel_ir_dispatch.h>
#include <ops/all_ops.h>
#include <scheduler/utils.h>

#include <algorithm>
#include <utility>

namespace nvfuser {

using IndexingTest = NVFuserTest;
using PredicateIndexingTest = NVFuserFixtureParamTest<bool>;

namespace {

std::vector<Val*> getLoopIndices(TensorView* tv, const TensorIndexer& indexer) {
  std::vector<Val*> loop_indices;
  for (const auto& loop_id : tv->getLoopDomain()) {
    loop_indices.push_back(indexer.getLoopIndex(loop_id));
  }
  return loop_indices;
}

std::vector<IterDomain*> getLoopDomains(
    TensorView* tv,
    const IdModel& id_model) {
  std::vector<IterDomain*> loop_domains;
  for (auto loop_id : tv->getLoopDomain()) {
    loop_domains.push_back(indexing_utils::getLoopPromotion(loop_id, id_model));
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
  static void validate(Fusion* fusion, Args... args) {
    EnableOptionsGuard enable_options_guard;
    EnableOptionsGuard::getCurOptions().set(
        EnableOption::IdModel, {"consumer_index", "producer_index"});

    // Disable simplifications to make the pattern matching of sameAs work
    DisableOptionsGuard disable_options_guard;
    DisableOptionsGuard::getCurOptions().set(DisableOption::ExprSimplify);
    DisableOptionsGuard::getCurOptions().set(DisableOption::IndexHoist);
    // Magic zero is not yet supported
    DisableOptionsGuard::getCurOptions().set(DisableOption::MagicZero);

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

    NVF_ERROR(!scope_exprs_.empty());
    auto inline_ite = dynamic_cast<kir::IfThenElse*>(scope_exprs_.back());
    NVF_ERROR(
        inline_ite != nullptr,
        "No inline predicate detected: ",
        expr->toString());

    validateInlinePredicate(out_ti, inline_ite->predicate()->value());

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
  static void validate(Fusion* fusion, Args... args) {
    EnableOptionsGuard enable_options_guard;
    EnableOptionsGuard::getCurOptions().set(
        EnableOption::IdModel, {"predicate"});

    // Disable simplifications to make the pattern matching of sameAs work
    DisableOptionsGuard disable_options_guard;
    DisableOptionsGuard::getCurOptions().set(DisableOption::ExprSimplify);
    DisableOptionsGuard::getCurOptions().set(DisableOption::IndexHoist);
    // Magic zero is not yet supported
    DisableOptionsGuard::getCurOptions().set(DisableOption::MagicZero);
    DisableOptionsGuard::getCurOptions().set(
        DisableOption::PredicateElimination);

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
      std::vector<Val*> loop_indices = getLoopIndices(consumer_tv, indexer_);
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
          NVF_ERROR(false, "Unexpected tensor: ", tv->toString());
          break;
      }
      return nullptr;
    }
  };

  IndexValidator<GetReference>::validate(&fusion);
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

  scheduler_utils::parallelizeAllLike(tv3, ir_utils::allTvs(&fusion));

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
          NVF_ERROR(false, "Unexpected tensor: ", tv->toString());
          break;
      }
    }
  };

  IndexValidator<GetReference>::validate(&fusion);
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
      std::vector<Val*> loop_indices = getLoopIndices(consumer_tv, indexer_);

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
          NVF_ERROR(false, "Unexpected tensor: ", tv->toString());
          // gcc v11.4 requires this return statement
          return nullptr;
      }
    }
  };

  IndexValidator<GetReference>::validate(&fusion);
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
      std::vector<Val*> loop_indices = getLoopIndices(consumer_tv, indexer_);

      switch (tv->name()) {
        case 1: {
          return loop_indices.at(1);
        }
        default:
          return nullptr;
      }
    }
  };

  IndexValidator<GetReference>::validate(&fusion);
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
      std::vector<Val*> loop_indices = getLoopIndices(consumer_tv, indexer_);

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

  IndexValidator<GetReference>::validate(&fusion);
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
      std::vector<Val*> loop_indices = getLoopIndices(consumer_tv, indexer_);

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
          // It isn't straightforward to do structual checking as the other
          // tests since there's no particular rule about which domain is used
          // to provide the extent of the group. However, since everything
          // should be deterministic, string match should also work.
          return std::string(
              "( ( ( ( ( i118 * ( ceilDiv(( 4 * 25 ), 5) ) ) + ( ( i119 * ( ceilDiv(( ceilDiv(( 4 * 25 ), 5) ), 2) ) ) + i120 ) ) / 25 ) * ( ceilDiv(100, 4) ) ) + ( ( ( i118 * ( ceilDiv(( 4 * 25 ), 5) ) ) + ( ( i119 * ( ceilDiv(( ceilDiv(( 4 * 25 ), 5) ), 2) ) ) + i120 ) ) % 25 ) )");
        }
        default:
          return std::string();
      }
    }
  };

  IndexValidator<GetReference>::validate(&fusion);
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
      std::vector<Val*> loop_indices = getLoopIndices(consumer_tv, indexer_);
      NVF_ERROR(loop_indices.at(1)->isZeroInt());
      switch (tv->name()) {
        case 0:
        case 1:
        case 2: {
          return loop_indices.at(0);
        }
        default:
          NVF_ERROR(false);
      }
    }
  };

  IndexValidator<GetReference>::validate(&fusion);
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
  // travesal should return "x + y * 4", where x and y are the loop
  // indices, respecitvely.

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
      std::vector<Val*> loop_indices = getLoopIndices(consumer_tv, indexer_);
      switch (tv->name()) {
        case 0:
        case 1:
        case 2: {
          return addExpr(
              mulExpr(loop_indices.at(0), tv->axis(1)->extent()),
              loop_indices.at(1));
        }
        default:
          NVF_ERROR(false);
      }
    }
  };

  IndexValidator<GetReference>::validate(&fusion);
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
      std::vector<Val*> loop_indices = getLoopIndices(consumer_tv, indexer_);
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
          NVF_ERROR(false);
      }
    }
  };

  IndexValidator<GetReference>::validate(&fusion);
}

// Concretized broadcast with partial inlining. Loop promotion is
// required. Same fusion as IdModelTest.LoopPromotion4. See also
// Example 1 of the Loop Promotion doc.
TEST_F(IndexingTest, SimpleBroadcast4) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto tv0 = makeContigConcreteTensor({1, 4});
  fusion.addInput(tv0);
  auto tv1 = makeContigConcreteTensor({3, 4});
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

  for (auto tv : ir_utils::allTvs(&fusion)) {
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
      std::vector<Val*> loop_indices = getLoopIndices(consumer_tv, indexer_);
      switch (tv->name()) {
        case 2: {
          return loop_indices.at(1);
        }
        default:
          return nullptr;
      }
    }
  };

  IndexValidator<GetReference>::validate(&fusion);
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

  IndexValidator<GetReference>::validate(&fusion);
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
      std::vector<Val*> loop_indices = getLoopIndices(consumer_tv, indexer_);
      return loop_indices.at(1);
    }
  };

  IndexValidator<GetReference>::validate(&fusion);
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
      std::vector<Val*> loop_indices = getLoopIndices(consumer_tv, indexer_);
      // Note that the allocation domain is the logical domain. See the
      // next test for a loop allocation example
      auto inner_dim = tv->getLogicalDomain().at(1)->extent();
      return addExpr(
          mulExpr(divExpr(loop_indices.at(1), inner_dim), inner_dim),
          modExpr(loop_indices.at(1), inner_dim));
    }
  };

  IndexValidator<GetReference>::validate(&fusion);
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
      std::vector<Val*> loop_indices = getLoopIndices(consumer_tv, indexer_);
      // Since the loop domain is the allocation domain, the index should
      // be just the non-parallelized loop index
      return loop_indices.at(1);
    }
  };

  IndexValidator<GetReference>::validate(&fusion);
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
      std::vector<Val*> loop_indices = getLoopIndices(consumer_tv, indexer_);
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
          NVF_ERROR(false);
      }
    }
  };

  IndexValidator<GetReference>::validate(&fusion);
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
      std::vector<Val*> loop_indices = getLoopIndices(consumer_tv, indexer_);
      switch (tv->name()) {
        case 2:
          return tv->fusion()->zeroVal();
        default:
          return nullptr;
      }
    }
  };

  IndexValidator<GetReference>::validate(&fusion);
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

  scheduler_utils::parallelizeAllLike(tv2, ir_utils::allTvs(&fusion));

  struct GetReference : AbstractGetReference {
    GetReference(const TensorIndexer& indexer, const IdModel& id_model)
        : AbstractGetReference(indexer, id_model) {}

    Val* getLinearIndex(TensorView* tv, TensorView* maybe_consumer)
        const override {
      bool as_consumer = maybe_consumer == nullptr;
      auto consumer_tv = as_consumer ? tv : maybe_consumer;
      std::vector<Val*> loop_indices = getLoopIndices(consumer_tv, indexer_);
      switch (tv->name()) {
        case 0:
        case 2:
          // Vectorized index is generated by replacing the index of a
          // vectorized domain with zero, which doesn't go through the
          // simplification of SimplifyingIrBuilder. We could use simplifyExpr,
          // but for the sake of testing, just use IrBuilder::addExpr.
          return IrBuilder::addExpr(
              mulExpr(
                  addExpr(
                      mulExpr(loop_indices.at(0), tv->axis(1)->extent()),
                      loop_indices.at(1)),
                  tv->axis(2)->extent()),
              tv->fusion()->zeroVal());
        case 1:
          return tv->fusion()->zeroVal();
        default:
          return nullptr;
      }
    }
  };

  IndexValidator<GetReference>::validate(&fusion);
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
  scheduler_utils::parallelizeAllLike(tv3, ir_utils::allTvs(&fusion));

  tv1->axis(2)->parallelize(ParallelType::Vectorize);
  tv3->axis(2)->parallelize(ParallelType::Vectorize);

  struct GetReference : AbstractGetReference {
    GetReference(const TensorIndexer& indexer, const IdModel& id_model)
        : AbstractGetReference(indexer, id_model) {}

    Val* getLinearIndex(TensorView* tv, TensorView* maybe_consumer)
        const override {
      bool as_consumer = maybe_consumer == nullptr;
      auto consumer_tv = as_consumer ? tv : maybe_consumer;
      std::vector<Val*> loop_indices = getLoopIndices(consumer_tv, indexer_);
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

  IndexValidator<GetReference>::validate(&fusion);
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
      std::vector<Val*> loop_indices = getLoopIndices(consumer_tv, indexer_);
      TensorView* tv2 = tv;
      TensorView* tv3 = consumer_tv;
      IterDomain* id11 = tv3->axis(1)->definition()->input(0)->as<IterDomain>();
      IterDomain* id9 = id11->definition()->input(1)->as<IterDomain>();
      Val* id11_idx = addExpr(
          mulExpr(loop_indices.at(1), tv3->axis(2)->extent()),
          loop_indices.at(2));
      Val* id8_idx = divExpr(id11_idx, id9->extent());
      // id8 is mapped with id15, which should also be mapped with
      // id18
      IterDomain* id20 = tv2->axis(2);
      Val* id19_idx = divExpr(id8_idx, id20->extent());
      Val* id20_idx = modExpr(id8_idx, id20->extent());
      Val* tv2_producer_idx =
          addExpr(mulExpr(id19_idx, id20->extent()), id20_idx);
      return tv2_producer_idx;
    }
  };

  IndexValidator<GetReference>::validate(&fusion);
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
      std::vector<Val*> loop_indices = getLoopIndices(consumer_tv, indexer_);

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

  IndexValidator<GetReference>::validate(&fusion);
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
      std::vector<Val*> loop_indices = getLoopIndices(consumer_tv, indexer_);
      NVF_ERROR(loop_indices.size() == 3);
      // Each of three domains corresponds to BIDx, Unroll and
      // TIDx. Only the Unroll domain is allocated.
      return loop_indices.at(1);
    }
  };

  IndexValidator<GetReference>::validate(&fusion);
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

  scheduler_utils::parallelizeAllLike(tv4, ir_utils::allTvs(&fusion));

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
      std::vector<Val*> loop_indices = getLoopIndices(consumer_tv, indexer_);
      return loop_indices.back();
    }
  };

  IndexValidator<GetReference>::validate(&fusion);
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
  for (auto tv : ir_utils::allTvs(&fusion)) {
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
      std::vector<Val*> loop_indices = getLoopIndices(consumer_tv, indexer_);

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

  IndexValidator<GetReference>::validate(&fusion, tv3->name());

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor input0 = at::randn({256, 256}, options);

  FusionExecutor fe;
  fe.compileFusion(&fusion, {input0});
  auto outputs = fe.runFusion({input0});

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
  // mapped with iS7. In this case, indexng tv0 for tv1 needs to use
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
      std::vector<Val*> loop_indices = getLoopIndices(consumer_tv, indexer_);

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

  IndexValidator<GetReference>::validate(&fusion);
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
      std::vector<Val*> loop_indices = getLoopIndices(consumer_tv, indexer_);

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
            return addExpr(
                mulExpr(loop_indices.at(1), tv->axis(2)->extent()),
                loop_indices.at(2));
          } else if (
              circular_buffer_loop_stage_ == CircularBufferLoopStage::Main) {
            return addExpr(
                mulExpr(
                    addExpr(loop_indices.at(0), createInt(1)), createInt(128)),
                addExpr(
                    mulExpr(loop_indices.at(1), tv->axis(2)->extent()),
                    loop_indices.at(2)));
          } else {
            NVF_ERROR(
                false,
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

  IndexValidator<GetReference>::validate(&fusion);
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
      std::vector<Val*> loop_indices = getLoopIndices(consumer_tv, indexer_);

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
              NVF_ERROR(
                  false, "Unexpected stage: ", circular_buffer_loop_stage_);
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
              NVF_ERROR(
                  false, "Unexpected stage: ", circular_buffer_loop_stage_);
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
              NVF_ERROR(
                  false, "Unexpected stage: ", circular_buffer_loop_stage_);
            }
          }
        }
        default:
          return nullptr;
      }
    }
  };

  IndexValidator<GetReference>::validate(&fusion);
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
      std::vector<Val*> loop_indices = getLoopIndices(consumer_tv, indexer_);

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
              NVF_ERROR(
                  false, "Unexpected stage: ", circular_buffer_loop_stage_);
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
              NVF_ERROR(
                  false, "Unexpected stage: ", circular_buffer_loop_stage_);
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
              NVF_ERROR(
                  false, "Unexpected stage: ", circular_buffer_loop_stage_);
            }
          }
        }
        default:
          return nullptr;
      }
    }
  };

  IndexValidator<GetReference>::validate(&fusion);
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
      std::vector<Val*> loop_indices = getLoopIndices(consumer_tv, indexer_);

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
            NVF_ERROR(
                false,
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

  IndexValidator<GetReference>::validate(&fusion);
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
      std::vector<Val*> loop_indices = getLoopIndices(consumer_tv, indexer_);

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
              NVF_ERROR(
                  false, "Unexpected stage: ", circular_buffer_loop_stage_);
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
              NVF_ERROR(
                  false, "Unexpected stage: ", circular_buffer_loop_stage_);
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
              NVF_ERROR(
                  false, "Unexpected stage: ", circular_buffer_loop_stage_);
            }
          }
        }
        default:
          return nullptr;
      }
    }
  };

  IndexValidator<GetReference>::validate(&fusion);
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
      std::vector<Val*> loop_indices = getLoopIndices(tv, indexer_);

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

  PredicateIndexValidator<GetReference>::validate(&fusion);
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
      std::vector<Val*> loop_indices = getLoopIndices(tv, indexer_);

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

  PredicateIndexValidator<GetReference>::validate(&fusion);
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
    // Note that "+ 0" remains since a symboic Val is just replaced
    // with zero.
    Val* getOuterPredicate(TensorView* tv) const override {
      std::vector<Val*> loop_indices = getLoopIndices(tv, indexer_);
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

  PredicateIndexValidator<GetReference>::validate(&fusion);
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
      std::vector<Val*> loop_indices = getLoopIndices(tv, indexer_);
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

  PredicateIndexValidator<GetReference>::validate(&fusion);
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

  scheduler_utils::parallelizeAllLike(tv2, ir_utils::allTvs(&fusion));

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
      std::vector<Val*> loop_indices = getLoopIndices(tv, indexer_);

      auto start_idx = IrBuilder::addExpr(
          mulExpr(
              addExpr(
                  mulExpr(loop_indices.at(0), tv->axis(1)->extent()),
                  loop_indices.at(1)),
              tv->axis(2)->extent()),
          tv->fusion()->zeroVal());
      auto stop_idx = addExpr(
          mulExpr(
              addExpr(
                  mulExpr(loop_indices.at(0), tv->axis(1)->extent()),
                  loop_indices.at(1)),
              tv->axis(2)->extent()),
          subExpr(tv->axis(2)->extent(), createInt(1)));

      // ( ( ( ( ( blockIdx.x * 128 ) + threadIdx.x ) * 4 ) + 0 )>= 0 ) &&
      // ( ( ( ( ( blockIdx.x * 128 ) + threadIdx.x ) * 4 ) + 3 ) < ( (( ((
      // getMetaData(T0) )).logical_size ))[0] ) ) )
      return andExpr(
          geExpr(start_idx, tv->fusion()->zeroVal()),
          ltExpr(stop_idx, tv->getLogicalDomain().at(0)->extent()));
    }
  };

  PredicateIndexValidator<GetReference>::validate(&fusion);
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
  scheduler_utils::parallelizeAllLike(tv3, ir_utils::allTvs(&fusion));

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

      std::vector<Val*> loop_indices = getLoopIndices(tv, indexer_);

      auto common_idx = addExpr(
          mulExpr(
              addExpr(
                  mulExpr(loop_indices.at(0), tv->axis(1)->extent()),
                  loop_indices.at(1)),
              tv->axis(3)->extent()),
          loop_indices.at(3));
      auto start_idx = IrBuilder::addExpr(
          mulExpr(common_idx, tv->axis(2)->extent()), tv->fusion()->zeroVal());
      auto stop_idx = addExpr(
          mulExpr(common_idx, tv->axis(2)->extent()),
          subExpr(tv->axis(2)->extent(), createInt(1)));

      return andExpr(
          geExpr(start_idx, tv->fusion()->zeroVal()),
          ltExpr(stop_idx, tv->getLogicalDomain().at(0)->extent()));
    }
  };

  PredicateIndexValidator<GetReference>::validate(&fusion);
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
      std::vector<Val*> loop_indices = getLoopIndices(tv, indexer_);

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
        // bidx.x * 32 + tid.x >= 0 &&
        // bidx.x * 32 + tid.x < N
        auto idx = addExpr(
            mulExpr(loop_indices.at(1), tv->axis(2)->extent()),
            loop_indices.at(2));
        return andExpr(
            geExpr(idx, tv->fusion()->zeroVal()),
            ltExpr(idx, tv->getLogicalDomain().at(0)->extent()));
      } else {
        // (i + 1) * 128 + bidx.x * 32 + tid.x >= 0 &&
        // (i + 1) * 128 + bidx.x * 32 + tid.x < N
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

  PredicateIndexValidator<GetReference>::validate(&fusion);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor t0 = at::randn({1000}, options);
  std::vector<c10::IValue> inputs = {t0};

  EnableOptionsGuard enable_options_guard;
  EnableOptionsGuard::getCurOptions().set(EnableOption::IdModel, {"all"});

  FusionExecutor fe;
  fe.compileFusion(&fusion, inputs);
  auto outputs = fe.runFusion(inputs);

  testValidate(&fusion, outputs, inputs, __LINE__, __FILE__);
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
      std::vector<Val*> loop_indices = getLoopIndices(tv, indexer_);

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
        // i * 128 + bidx.x * 32 + tid.x >= 0 &&
        // i * 128 + bidx.x * 32 + tid.x < N
        idx = addExpr(
            mulExpr(circular_buffer_index, createInt(128)),
            addExpr(
                mulExpr(loop_indices.at(1), tv->axis(2)->extent()),
                loop_indices.at(2)));
      } else {
        // (i + 3) * 128 + bidx.x * 32 + tid.x >= 0 &&
        // (i + 3) * 128 + bidx.x * 32 + tid.x < N
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

  PredicateIndexValidator<GetReference>::validate(&fusion);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor t0 = at::randn({1000}, options);
  std::vector<c10::IValue> inputs = {t0};

  EnableOptionsGuard enable_options_guard;
  EnableOptionsGuard::getCurOptions().set(EnableOption::IdModel, {"all"});

  FusionExecutor fe;
  fe.compileFusion(&fusion, inputs);
  auto outputs = fe.runFusion(inputs);

  testValidate(&fusion, outputs, inputs, __LINE__, __FILE__);
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
      std::vector<Val*> loop_indices = getLoopIndices(tv, indexer_);

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
                            mulExpr(
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
                            mulExpr(
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

  PredicateIndexValidator<GetReference>::validate(&fusion);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor t0 = at::randn({1000}, options);
  std::vector<c10::IValue> inputs = {t0};

  EnableOptionsGuard enable_options_guard;
  EnableOptionsGuard::getCurOptions().set(EnableOption::IdModel, {"all"});

  FusionExecutor fe;
  fe.compileFusion(&fusion, inputs);
  auto outputs = fe.runFusion(inputs);

  testValidate(&fusion, outputs, inputs, __LINE__, __FILE__);
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
  //          |  +-- circular bufer
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
      std::vector<Val*> loop_indices = getLoopIndices(tv, indexer_);

      auto zero = tv->fusion()->zeroVal();

      // The base index is:
      //
      // i0 * 4 + i2
      //
      // where i2 is the circular buffer index. The index of iUS10 is
      // not included as its extent is 1.

      // Start index: i0 * 4
      Val* start_idx = mulExpr(loop_indices.at(0), createInt(4));

      // Stop index: i0 * 4 + 4
      // Note that it isn't "i0 * 4 + 3" since i2 is circular buffered
      // and there's no epilog, so the main loop has a read of (i2 +
      // 1).
      Val* stop_idx =
          addExpr(mulExpr(loop_indices.at(0), createInt(4)), createInt(4));

      return andExpr(
          geExpr(start_idx, zero),
          ltExpr(stop_idx, tv->getLogicalDomain().at(0)->extent()));
    }
  };

  PredicateIndexValidator<GetReference>::validate(&fusion);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor t0 = at::randn({99}, options);
  std::vector<c10::IValue> inputs = {t0};

  EnableOptionsGuard enable_options_guard;
  EnableOptionsGuard::getCurOptions().set(EnableOption::IdModel, {"all"});

  FusionExecutor fe;
  fe.compileFusion(&fusion, inputs);
  auto outputs = fe.runFusion(inputs);

  testValidate(&fusion, outputs, inputs, __LINE__, __FILE__);
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
      std::vector<Val*> loop_indices = getLoopIndices(tv, indexer_);

      auto zero = tv->fusion()->zeroVal();

      // The base index is:
      //
      // (i0 * 128 + i2) * 4 + i3
      //
      // where i2 is the circular buffer index. Here, i3 corresponds
      // to the vectorization. Since it's vectorized, the predicate
      // uses 0 for start and (vec_factor - 1) for stop

      // Start index: (i0 * 128 + 0) * 4 + 0
      Val* start_idx = IrBuilder::addExpr(
          mulExpr(
              IrBuilder::addExpr(
                  mulExpr(loop_indices.at(0), createInt(128)), zero),
              createInt(4)),
          zero);
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

  PredicateIndexValidator<GetReference>::validate(&fusion);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor t0 = at::randn({1000}, options);
  std::vector<c10::IValue> inputs = {t0};

  EnableOptionsGuard enable_options_guard;
  EnableOptionsGuard::getCurOptions().set(EnableOption::IdModel, {"all"});

  FusionExecutor fe;
  fe.compileFusion(&fusion, inputs);
  auto outputs = fe.runFusion(inputs);

  testValidate(&fusion, outputs, inputs, __LINE__, __FILE__);
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
      std::vector<Val*> loop_indices = getLoopIndices(tv, indexer_);

      auto zero = tv->fusion()->zeroVal();

      // The base index is:
      //
      // (i0 * 128 + i2) * 4 + i3
      //
      // where i2 is the circular buffer index. Here, i3 corresponds
      // to the vectorization. Since it's vectorized, the predicate
      // uses 0 for start and (vec_factor - 1) for stop

      // Start index: (i0 * 128 + 0) * 4 + 0
      Val* start_idx = IrBuilder::addExpr(
          mulExpr(
              IrBuilder::addExpr(
                  mulExpr(loop_indices.at(0), createInt(128)), zero),
              createInt(4)),
          zero);
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

  PredicateIndexValidator<GetReference>::validate(&fusion);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor t0 = at::randn({1000}, options);
  at::Tensor t1 = at::randn({1000}, options);
  std::vector<c10::IValue> inputs = {t0, t1};

  EnableOptionsGuard enable_options_guard;
  EnableOptionsGuard::getCurOptions().set(EnableOption::IdModel, {"all"});

  FusionExecutor fe;
  fe.compileFusion(&fusion, inputs);
  auto outputs = fe.runFusion(inputs);

  testValidate(&fusion, outputs, inputs, __LINE__, __FILE__);
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

  PredicateIndexValidator<GetReference>::validate(&fusion);

  // Running this fusion with the legacy indexer would result in an
  // error if run with compute-sanitizer.
  EnableOptionsGuard enable_options_guard;
  EnableOptionsGuard::getCurOptions().set(EnableOption::IdModel, {"all"});

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  auto t0 = at::randn({16}, options);

  FusionExecutor fe;
  fe.compileFusion(&fusion, {t0});
  auto cg_outputs = fe.runFusion({t0});

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
      std::vector<Val*> loop_indices = getLoopIndices(tv, indexer_);
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

  PredicateIndexValidator<GetReference>::validate(&fusion);

  EnableOptionsGuard enable_options_guard;
  EnableOptionsGuard::getCurOptions().set(EnableOption::IdModel, {"all"});

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor t0 = at::randn({999}, options);
  std::vector<c10::IValue> aten_inputs = {t0};

  FusionExecutor fe;
  fe.compileFusion(&fusion, aten_inputs);
  auto outputs = fe.runFusion(aten_inputs);

  testValidate(&fusion, outputs, aten_inputs, __LINE__, __FILE__);
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
      std::vector<Val*> loop_indices = getLoopIndices(tv, indexer_);
      auto zero = tv->fusion()->zeroVal();
      auto one = tv->fusion()->oneVal();

      IterDomain* second_split_input =
          tv->axis(1)->definition()->input(0)->as<IterDomain>();

      Val* second_split_start_idx = addExpr(
          IrBuilder::mulExpr(zero, tv->axis(2)->extent()), loop_indices.at(2));

      Val* second_split_stop_idx = addExpr(
          mulExpr(subExpr(tv->axis(1)->extent(), one), tv->axis(2)->extent()),
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

  PredicateIndexValidator<GetReference>::validate(&fusion);

  EnableOptionsGuard enable_options_guard;
  EnableOptionsGuard::getCurOptions().set(EnableOption::IdModel, {"all"});

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor t0 = at::randn({999}, options);
  std::vector<c10::IValue> aten_inputs = {t0};

  FusionExecutor fe;
  fe.compileFusion(&fusion, aten_inputs);
  auto outputs = fe.runFusion(aten_inputs);

  testValidate(&fusion, outputs, aten_inputs, __LINE__, __FILE__);
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
      std::vector<Val*> loop_indices = getLoopIndices(tv, indexer_);
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

  PredicateIndexValidator<GetReference>::validate(&fusion);

  EnableOptionsGuard enable_options_guard;
  EnableOptionsGuard::getCurOptions().set(EnableOption::IdModel, {"all"});

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor t0 = at::randn({999}, options);
  std::vector<c10::IValue> aten_inputs = {t0};

  FusionExecutor fe;
  fe.compileFusion(&fusion, aten_inputs);
  auto outputs = fe.runFusion(aten_inputs);

  testValidate(&fusion, outputs, aten_inputs, __LINE__, __FILE__);
}

// Non divisible split with unswitched circularing. The non divisible
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
      std::vector<Val*> loop_indices = getLoopIndices(tv, indexer_);
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
          mulExpr(
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

  PredicateIndexValidator<GetReference>::validate(&fusion);

  EnableOptionsGuard enable_options_guard;
  EnableOptionsGuard::getCurOptions().set(EnableOption::IdModel, {"all"});

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor t0 = at::randn({999}, options);
  std::vector<c10::IValue> aten_inputs = {t0};

  FusionExecutor fe;
  fe.compileFusion(&fusion, aten_inputs);
  auto outputs = fe.runFusion(aten_inputs);

  testValidate(&fusion, outputs, aten_inputs, __LINE__, __FILE__);
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
  std::vector<c10::IValue> aten_inputs = {t0};

  struct GetReference : AbstractGetReference {
    GetReference(const TensorIndexer& indexer, const IdModel& id_model)
        : AbstractGetReference(indexer, id_model) {}

    Val* getOuterPredicate(TensorView* tv) const override {
      std::vector<Val*> loop_indices = getLoopIndices(tv, indexer_);
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

  PredicateIndexValidator<GetReference>::validate(&fusion);

  EnableOptionsGuard enable_options_guard;
  if (GetParam()) {
    EnableOptionsGuard::getCurOptions().set(EnableOption::IdModel, {"all"});
  } else {
    EnableOptionsGuard::getCurOptions().unset(EnableOption::IdModel);
  }

  FusionExecutor fe;
  fe.compileFusion(&fusion, aten_inputs);
  auto outputs = fe.runFusion(aten_inputs);

  auto ref = t0.to(at::kDouble).sum();

  testValidate(&fusion, outputs, aten_inputs, {ref}, __LINE__, __FILE__);
}

// Testing unswitched non-divisible predicates. For a given tensor,
// its required unswitch predicates should be generated from its
// indexing path, not its logical to loop dependencies. In this test,
// the tv2 transformation has an non-divisible split that needs to be
// predicated, but since it's inline into tv3, the acual non-divisible
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
      std::vector<Val*> loop_indices = getLoopIndices(tv, indexer_);
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

  PredicateIndexValidator<GetReference>::validate(&fusion);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor t0 = at::randn({5}, options);
  at::Tensor t1 = at::randn({5, 100}, options);
  std::vector<c10::IValue> aten_inputs = {t0, t1};

  EnableOptionsGuard enable_options_guard;
  EnableOptionsGuard::getCurOptions().set(EnableOption::IdModel, {"all"});

  FusionExecutor fe;
  fe.compileFusion(&fusion, aten_inputs);
  auto outputs = fe.runFusion(aten_inputs);

  testValidate(&fusion, outputs, aten_inputs, __LINE__, __FILE__);
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
      std::vector<Val*> loop_indices = getLoopIndices(tv, indexer_);
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

  PredicateIndexValidator<GetReference>::validate(&fusion);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor t0 = at::randn({1000}, options);
  at::Tensor t1 = at::randn({1000}, options);
  std::vector<c10::IValue> aten_inputs = {t0, t1};

  EnableOptionsGuard enable_options_guard;
  EnableOptionsGuard::getCurOptions().set(EnableOption::IdModel, {"all"});

  FusionExecutor fe;
  fe.compileFusion(&fusion, aten_inputs);
  auto outputs = fe.runFusion(aten_inputs);

  testValidate(&fusion, outputs, aten_inputs, __LINE__, __FILE__);
}

} // namespace nvfuser
