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

namespace {

std::vector<Val*> getLoopIndices(TensorView* tv, const TensorIndexer& indexer) {
  std::vector<Val*> loop_indices;
  for (const auto& loop_id : tv->getLoopDomain()) {
    loop_indices.push_back(indexer.getLoopIndex(loop_id));
  }
  return loop_indices;
}

template <typename... Args>
Val* addExpr(Args&&... args) {
  return SimplifyingIrBuilder::addExpr(std::forward<Args>(args)...);
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
Val* modExpr(Args&&... args) {
  return SimplifyingIrBuilder::modExpr(std::forward<Args>(args)...);
}

void printAllIndices(
    std::ostream& os,
    Fusion* fusion,
    const TensorIndexer& indexer) {
  for (auto expr : fusion->exprs()) {
    if (!ir_utils::isTvOp(expr)) {
      continue;
    }

    os << "\n" << expr->toString();
    for (auto input : ir_utils::filterByType<TensorView>(expr->inputs())) {
      os << "Input: T" << input->name() << " -> "
         << indexer.getLinearIndex(input, expr)->toInlineString() << std::endl;
    }
    for (auto output : ir_utils::filterByType<TensorView>(expr->outputs())) {
      os << "Output: T" << output->name() << " -> "
         << indexer.getLinearIndex(output, expr)->toInlineString() << std::endl;
    }
  }
}

// AbstractGetReference and IndexValidator are used to validate
// lowered index vals. Each test subclasses either or both of
// getLinearIndex and getLinearIndexString of
// AbstractGetReference. IndexValidator traverses lowered exprs to
// validate each tensor indices.
class AbstractGetReference {
 public:
  AbstractGetReference(const TensorIndexer& indexer) : indexer_(indexer) {}
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

 protected:
  const TensorIndexer& indexer_;
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
  }

  void validate(kir::TensorIndex* ti, kir::TensorIndex* out_ti = nullptr) {
    TensorView* tv = ti->view();
    TensorView* maybe_consumer = out_ti != nullptr ? out_ti->view() : nullptr;
    Val* actual = ti->index();
    Val* ref = get_ref_.getLinearIndex(tv, maybe_consumer);
    if (ref != nullptr) {
      EXPECT_TRUE(actual->sameAs(ref))
          << "Validation failure of " << ti->view()->toString()
          << "\nRef: " << ref->toInlineString()
          << "\nActual: " << actual->toInlineString();
      return;
    }

    // If nullptr is returned, check if a string ref is available
    std::string ref_str = get_ref_.getLinearIndexString(tv, maybe_consumer);
    if (!ref_str.empty()) {
      EXPECT_EQ(actual->toInlineString(), ref_str)
          << "Validation failure of " << ti->view()->toString()
          << "\nRef: " << ref_str << "\nActual: " << actual->toInlineString();
      return;
    }

    // If no ref is obtained, skip validation
  }

  static void validate(Fusion* fusion) {
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
        lower, GetReference(lower.tensorIndexer()));

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
  MaxRootDomainInfoSpanningTree(tv2).traverse(&propagator);

  tv1->inlineAt(1);

  struct GetReference : AbstractGetReference {
    GetReference(const TensorIndexer& indexer)
        : AbstractGetReference(indexer) {}

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
                  IrBuilder::getItemExpr(
                      IrBuilder::getAttrExpr(
                          IrBuilder::metadataExpr(tv), "alloc_stride"),
                      IrBuilder::create<Val>(0)),
                  divExpr(
                      addExpr(
                          mulExpr(
                              consumer_tv->axis(1)->extent(),
                              loop_indices.at(0)),
                          loop_indices.at(1)),
                      tv->getLogicalDomain().at(1)->extent())),
              mulExpr(
                  IrBuilder::getItemExpr(
                      IrBuilder::getAttrExpr(
                          IrBuilder::metadataExpr(tv), "alloc_stride"),
                      IrBuilder::create<Val>(1)),
                  modExpr(
                      addExpr(
                          mulExpr(
                              consumer_tv->axis(1)->extent(),
                              loop_indices.at(0)),
                          loop_indices.at(1)),
                      tv->getLogicalDomain().at(1)->extent())));
        }
        case 1: {
          return loop_indices.at(1);
        }
        case 2: {
          NVF_ERROR(as_consumer);
          return addExpr(
              mulExpr(
                  tv->getLogicalDomain().at(1)->extent(),
                  divExpr(
                      addExpr(
                          mulExpr(loop_indices.at(0), tv->axis(1)->extent()),
                          loop_indices.at(1)),
                      tv->getLogicalDomain().at(1)->extent())),
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
  MaxRootDomainInfoSpanningTree(tv3).traverse(&propagator);

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
    GetReference(const TensorIndexer& indexer)
        : AbstractGetReference(indexer) {}

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
              tv->getLogicalDomain().at(1)->extent(),
              SimplifyingIrBuilder::divExpr(
                  contig_idx, tv->getLogicalDomain().at(1)->extent())),
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
    GetReference(const TensorIndexer& indexer)
        : AbstractGetReference(indexer) {}

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
                  tv->getLogicalDomain().at(1)->extent(), loop_indices.at(0)),
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
    GetReference(const TensorIndexer& indexer)
        : AbstractGetReference(indexer) {}

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
  MaxRootDomainInfoSpanningTree(tv5).traverse(&propagator);

  inlineMost();

  struct GetReference : AbstractGetReference {
    GetReference(const TensorIndexer& indexer)
        : AbstractGetReference(indexer) {}

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
    GetReference(const TensorIndexer& indexer)
        : AbstractGetReference(indexer) {}

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
  MaxRootDomainInfoSpanningTree(tv2).traverse(&propagator);

  // The first merge of the logical domains should be a trivial merge,
  // i.e., a merge with a extent-one domain. Thus, the indexing
  // travesal should return "x + y * 4", where x and y are the loop
  // indices, respecitvely.

  struct GetReference : AbstractGetReference {
    GetReference(const TensorIndexer& indexer)
        : AbstractGetReference(indexer) {}

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
  MaxRootDomainInfoSpanningTree(tv3).traverse(&propagator);

  inlineMost();

  struct GetReference : AbstractGetReference {
    GetReference(const TensorIndexer& indexer)
        : AbstractGetReference(indexer) {}

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
                  tv->getLogicalDomain().at(1)->extent(),
                  divExpr(
                      loop_indices.at(0),
                      tv->getLogicalDomain().at(1)->extent())),
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
  MaxRootDomainInfoSpanningTree(tv4).traverse(&propagator);

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
    GetReference(const TensorIndexer& indexer)
        : AbstractGetReference(indexer) {}

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
    GetReference(const TensorIndexer& indexer)
        : AbstractGetReference(indexer) {}

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
    GetReference(const TensorIndexer& indexer)
        : AbstractGetReference(indexer) {}

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
  MaxRootDomainInfoSpanningTree(tv1).traverse(&propagator);

  tv0->axis(0)->parallelize(ParallelType::DIDx);
  tv1->axis(0)->parallelize(ParallelType::DIDx);

  struct GetReference : AbstractGetReference {
    GetReference(const TensorIndexer& indexer)
        : AbstractGetReference(indexer) {}

    Val* getLinearIndex(TensorView* tv, TensorView* maybe_consumer)
        const override {
      bool as_consumer = maybe_consumer == nullptr;
      auto consumer_tv = as_consumer ? tv : maybe_consumer;
      std::vector<Val*> loop_indices = getLoopIndices(consumer_tv, indexer_);
      // Note that the allocation domain is the logical domain. See the
      // next test for a loop allocation example
      auto inner_dim = tv->getLogicalDomain().at(1)->extent();
      return addExpr(
          mulExpr(inner_dim, divExpr(loop_indices.at(1), inner_dim)),
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
  MaxRootDomainInfoSpanningTree(tv1).traverse(&propagator);

  tv0->axis(0)->parallelize(ParallelType::DIDx);
  tv1->axis(0)->parallelize(ParallelType::DIDx);

  tv0->setAllocationDomain(tv0->getLoopDomain(), true);
  tv1->setAllocationDomain(tv1->getLoopDomain(), true);

  struct GetReference : AbstractGetReference {
    GetReference(const TensorIndexer& indexer)
        : AbstractGetReference(indexer) {}

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
    GetReference(const TensorIndexer& indexer)
        : AbstractGetReference(indexer) {}

    Val* getLinearIndex(TensorView* tv, TensorView* maybe_consumer)
        const override {
      bool as_consumer = maybe_consumer == nullptr;
      auto consumer_tv = as_consumer ? tv : maybe_consumer;
      std::vector<Val*> loop_indices = getLoopIndices(consumer_tv, indexer_);
      switch (tv->name()) {
        case 0: {
          return addExpr(
              mulExpr(
                  tv->getLogicalDomain().at(1)->extent(), loop_indices.at(2)),
              loop_indices.at(1));
        }
        case 1: {
          return addExpr(
              mulExpr(
                  tv->getLogicalDomain().at(1)->extent(), loop_indices.at(1)),
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
    GetReference(const TensorIndexer& indexer)
        : AbstractGetReference(indexer) {}

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
  MaxRootDomainInfoSpanningTree(tv2).traverse(&propagator);

  inlineMost();

  scheduler_utils::parallelizeAllLike(tv2, ir_utils::allTvs(&fusion));

  struct GetReference : AbstractGetReference {
    GetReference(const TensorIndexer& indexer)
        : AbstractGetReference(indexer) {}

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

} // namespace nvfuser
