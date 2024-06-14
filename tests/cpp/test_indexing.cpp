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
#include <ops/all_ops.h>
#include <scheduler/utils.h>

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

  IdModel id_model(&fusion);
  id_model.validateAndPropagatePType();
  TensorIndexer indexer(id_model);

  std::vector<Val*> tv1_loop_indices = getLoopIndices(tv1, indexer);
  std::vector<Val*> tv2_loop_indices = getLoopIndices(tv2, indexer);

  auto tv0_producer_index = indexer.getLinearIndex(tv0, tv1->definition());

  auto tv1_consumer_index = indexer.getLinearIndex(tv1, tv1->definition());
  auto tv1_producer_index = indexer.getLinearIndex(tv1, tv2->definition());
  auto tv2_consumer_index = indexer.getLinearIndex(tv2, tv2->definition());

  auto tv0_producer_index_ref = SimplifyingIrBuilder::addExpr(
      SimplifyingIrBuilder::mulExpr(
          SimplifyingIrBuilder::modExpr(
              SimplifyingIrBuilder::addExpr(
                  SimplifyingIrBuilder::mulExpr(
                      tv1_loop_indices.at(0), tv1->axis(1)->extent()),
                  tv1_loop_indices.at(1)),
              tv1->getLogicalDomain().at(1)->extent()),
          IrBuilder::getItemExpr(
              IrBuilder::getAttrExpr(
                  IrBuilder::metadataExpr(tv0), "alloc_stride"),
              IrBuilder::create<Val>(1))),
      SimplifyingIrBuilder::mulExpr(
          SimplifyingIrBuilder::divExpr(
              SimplifyingIrBuilder::addExpr(
                  SimplifyingIrBuilder::mulExpr(
                      tv1_loop_indices.at(0), tv1->axis(1)->extent()),
                  tv1_loop_indices.at(1)),
              tv1->getLogicalDomain().at(1)->extent()),
          IrBuilder::getItemExpr(
              IrBuilder::getAttrExpr(
                  IrBuilder::metadataExpr(tv0), "alloc_stride"),
              IrBuilder::create<Val>(0))));

  auto tv1_consumer_index_ref = tv1_loop_indices.at(1);
  auto tv1_producer_index_ref = tv2_loop_indices.at(1);

  auto tv2_consumer_index_ref = SimplifyingIrBuilder::addExpr(
      SimplifyingIrBuilder::modExpr(
          SimplifyingIrBuilder::addExpr(
              SimplifyingIrBuilder::mulExpr(
                  tv2_loop_indices.at(0), tv2->axis(1)->extent()),
              tv2_loop_indices.at(1)),
          tv2->getLogicalDomain().at(1)->extent()),
      SimplifyingIrBuilder::mulExpr(
          SimplifyingIrBuilder::divExpr(
              SimplifyingIrBuilder::addExpr(
                  SimplifyingIrBuilder::mulExpr(
                      tv2_loop_indices.at(0), tv2->axis(1)->extent()),
                  tv2_loop_indices.at(1)),
              tv2->getLogicalDomain().at(1)->extent()),
          tv2->getLogicalDomain().at(1)->extent()));

  EXPECT_TRUE(tv0_producer_index->sameAs(tv0_producer_index_ref))
      << "Ref: " << tv0_producer_index_ref->toInlineString()
      << ". Actual: " << tv0_producer_index->toInlineString();

  EXPECT_TRUE(tv1_consumer_index->sameAs(tv1_consumer_index_ref))
      << "Ref: " << tv1_consumer_index_ref->toInlineString()
      << ". Actual: " << tv1_consumer_index->toInlineString();

  EXPECT_TRUE(tv1_producer_index->sameAs(tv1_producer_index_ref))
      << "Ref: " << tv1_producer_index_ref->toInlineString()
      << ". Actual: " << tv1_producer_index->toInlineString();

  EXPECT_TRUE(tv2_consumer_index->sameAs(tv2_consumer_index_ref))
      << "Ref: " << tv2_consumer_index_ref->toInlineString()
      << ". Actual: " << tv2_consumer_index->toInlineString();
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

  IdModel id_model(&fusion);
  id_model.validateAndPropagatePType();
  TensorIndexer indexer(id_model);

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

  auto tv0_producer_index = indexer.getLinearIndex(tv0, tv1->definition());
  auto tv1_consumer_index = indexer.getLinearIndex(tv1, tv1->definition());
  auto tv1_producer_index = indexer.getLinearIndex(tv1, tv2->definition());
  auto tv2_consumer_index = indexer.getLinearIndex(tv2, tv2->definition());
  auto tv2_producer_index = indexer.getLinearIndex(tv2, tv3->definition());
  auto tv3_consumer_index = indexer.getLinearIndex(tv3, tv3->definition());

  auto contig_idx = SimplifyingIrBuilder::addExpr(
      SimplifyingIrBuilder::mulExpr(
          NamedScalar::getParallelIndex(ParallelType::BIDx),
          tv2->axis(1)->extent()),
      NamedScalar::getParallelIndex(ParallelType::TIDx));

  auto global_ref = SimplifyingIrBuilder::addExpr(
      SimplifyingIrBuilder::modExpr(
          contig_idx, tv0->getLogicalDomain().at(1)->extent()),
      SimplifyingIrBuilder::mulExpr(
          SimplifyingIrBuilder::divExpr(
              contig_idx, tv0->getLogicalDomain().at(1)->extent()),
          tv0->getLogicalDomain().at(1)->extent()));

  auto shared_ref = NamedScalar::getParallelIndex(ParallelType::TIDx);

  EXPECT_TRUE(tv0_producer_index->sameAs(global_ref))
      << "Ref: " << global_ref->toInlineString()
      << ". Actual: " << tv0_producer_index->toInlineString();

  EXPECT_TRUE(tv1_consumer_index->isZeroInt())
      << "Actual: " << tv1_consumer_index->toInlineString();

  EXPECT_TRUE(tv1_producer_index->isZeroInt())
      << "Actual: " << tv1_producer_index->toInlineString();

  EXPECT_TRUE(tv2_producer_index->sameAs(shared_ref))
      << "Ref: " << shared_ref->toInlineString()
      << ". Actual: " << tv2_producer_index->toInlineString();

  EXPECT_TRUE(tv2_consumer_index->sameAs(shared_ref))
      << "Ref: " << shared_ref->toInlineString()
      << ". Actual: " << tv2_consumer_index->toInlineString();

  EXPECT_TRUE(tv3_consumer_index->sameAs(global_ref))
      << "Ref: " << global_ref->toInlineString()
      << ". Actual: " << tv3_consumer_index->toInlineString();
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

  IdModel id_model(&fusion);
  id_model.validateAndPropagatePType();
  TensorIndexer indexer(id_model);

  std::vector<Val*> tv1_loop_indices = getLoopIndices(tv1, indexer);
  std::vector<Val*> tv2_loop_indices = getLoopIndices(tv2, indexer);

  auto tv0_producer_index = indexer.getLinearIndex(tv0, tv1->definition());
  auto tv1_consumer_index = indexer.getLinearIndex(tv1, tv1->definition());
  auto tv1_producer_index = indexer.getLinearIndex(tv1, tv2->definition());
  auto tv2_consumer_index = indexer.getLinearIndex(tv2, tv2->definition());

  auto tv0_producer_index_ref = SimplifyingIrBuilder::addExpr(
      tv1_loop_indices.at(1),
      SimplifyingIrBuilder::mulExpr(
          tv1_loop_indices.at(0), tv0->getLogicalDomain().at(1)->extent()));

  auto tv1_consumer_index_ref = tv1_loop_indices.at(0);
  auto tv1_producer_index_ref = tv2_loop_indices.at(0);
  auto tv2_consumer_index_ref = tv2_loop_indices.at(0);

  EXPECT_TRUE(tv0_producer_index->sameAs(tv0_producer_index_ref));
  EXPECT_TRUE(tv1_consumer_index->sameAs(tv1_consumer_index_ref));
  EXPECT_TRUE(tv1_producer_index->sameAs(tv1_producer_index_ref));
  EXPECT_TRUE(tv2_consumer_index->sameAs(tv2_consumer_index_ref));
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

  IdModel id_model(&fusion);
  id_model.validateAndPropagatePType();
  TensorIndexer indexer(id_model);

  auto tv1_consumer_index = indexer.getLinearIndex(tv1, tv1->definition());
  auto tv1_producer_index = indexer.getLinearIndex(tv1, tv2->definition());

  std::vector<Val*> tv1_loop_indices = getLoopIndices(tv1, indexer);
  std::vector<Val*> tv2_loop_indices = getLoopIndices(tv2, indexer);

  // Note that the allocation domain is permuted
  auto tv1_consumer_index_ref = SimplifyingIrBuilder::addExpr(
      tv1_loop_indices.at(0),
      SimplifyingIrBuilder::mulExpr(
          tv1_loop_indices.at(1), tv1->getLogicalDomain().at(0)->extent()));

  auto tv1_producer_index_ref = SimplifyingIrBuilder::addExpr(
      tv2_loop_indices.at(0),
      SimplifyingIrBuilder::mulExpr(
          tv2_loop_indices.at(1), tv1->getLogicalDomain().at(0)->extent()));

  EXPECT_TRUE(tv1_consumer_index->sameAs(tv1_consumer_index_ref))
      << "Ref: " << tv1_consumer_index_ref->toInlineString()
      << ". Actual: " << tv1_consumer_index->toInlineString();

  EXPECT_TRUE(tv1_producer_index->sameAs(tv1_producer_index_ref))
      << "Ref: " << tv1_producer_index_ref->toInlineString()
      << ". Actual: " << tv1_producer_index->toInlineString();
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

  IdModel id_model(&fusion);
  id_model.validateAndPropagatePType();
  TensorIndexer indexer(id_model);

  // Validate tv0 indexing
  auto tv0_producer_index = indexer.getLinearIndex(tv0, tv1->definition());

  // It isn't straightforward to do structual checking as the other
  // tests since there's no particular rule about which domain is used
  // to provide the extent of the group. However, since everything
  // should be deterministic, string match should also work.
  std::string tv0_producer_index_ref =
      "( ( ( ( ( i114 * ( ceilDiv(( 4 * 25 ), 5) ) ) + ( ( i115 * ( ceilDiv(( ceilDiv(( 4 * 25 ), 5) ), 2) ) ) + i116 ) ) / 25 ) * ( ceilDiv(100, 4) ) ) + ( ( ( i114 * ( ceilDiv(( 4 * 25 ), 5) ) ) + ( ( i115 * ( ceilDiv(( ceilDiv(( 4 * 25 ), 5) ), 2) ) ) + i116 ) ) % 25 ) )";

  EXPECT_EQ(tv0_producer_index->toInlineString(), tv0_producer_index_ref);

  // All intermediate tensors should be fully inlined, so their
  // indices should be just zero.
  EXPECT_TRUE(indexer.getLinearIndex(tv1, tv1->definition())->isZeroInt());
  EXPECT_TRUE(indexer.getLinearIndex(tv1, tv2->definition())->isZeroInt());
  EXPECT_TRUE(indexer.getLinearIndex(tv2, tv2->definition())->isZeroInt());
  EXPECT_TRUE(indexer.getLinearIndex(tv2, tv3->definition())->isZeroInt());
  EXPECT_TRUE(indexer.getLinearIndex(tv3, tv3->definition())->isZeroInt());
  EXPECT_TRUE(indexer.getLinearIndex(tv3, tv4->definition())->isZeroInt());
  EXPECT_TRUE(indexer.getLinearIndex(tv4, tv4->definition())->isZeroInt());
  EXPECT_TRUE(indexer.getLinearIndex(tv4, tv5->definition())->isZeroInt());

  // tv5 has no transformation and is fully contiguous
  std::vector<Val*> tv5_loop_indices = getLoopIndices(tv5, indexer);
  auto tv5_consumer_index = indexer.getLinearIndex(tv5, tv5->definition());

  auto tv5_consumer_index_ref = SimplifyingIrBuilder::addExpr(
      SimplifyingIrBuilder::addExpr(
          tv5_loop_indices.at(2),
          SimplifyingIrBuilder::mulExpr(
              tv5_loop_indices.at(1), tv5->getLogicalDomain().at(2)->extent())),
      SimplifyingIrBuilder::mulExpr(
          tv5_loop_indices.at(0),
          SimplifyingIrBuilder::mulExpr(
              tv5->getLogicalDomain().at(1)->extent(),
              tv5->getLogicalDomain().at(2)->extent())));

  EXPECT_TRUE(tv5_consumer_index->sameAs(tv5_consumer_index_ref))
      << "Ref: " << tv5_consumer_index_ref->toInlineString()
      << ". Actual: " << tv5_consumer_index->toInlineString();
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

  IdModel id_model(&fusion);
  id_model.validateAndPropagatePType();
  TensorIndexer indexer(id_model);

  std::vector<Val*> tv1_loop_indices = getLoopIndices(tv1, indexer);

  EXPECT_TRUE(tv1_loop_indices.at(1)->isZeroInt());

  std::vector<Val*> tv2_loop_indices = getLoopIndices(tv2, indexer);

  EXPECT_TRUE(tv2_loop_indices.at(1)->isZeroInt());

  auto tv0_producer_index = indexer.getLinearIndex(tv0, tv1->definition());
  auto tv1_consumer_index = indexer.getLinearIndex(tv1, tv1->definition());
  auto tv1_producer_index = indexer.getLinearIndex(tv1, tv2->definition());
  auto tv2_consumer_index = indexer.getLinearIndex(tv2, tv2->definition());

  EXPECT_EQ(tv0_producer_index, tv1_loop_indices.at(0));
  EXPECT_EQ(tv1_consumer_index, tv1_loop_indices.at(0));
  EXPECT_EQ(tv1_producer_index, tv2_loop_indices.at(0));
  EXPECT_EQ(tv2_consumer_index, tv2_loop_indices.at(0));
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

  IdModel id_model(&fusion);
  id_model.validateAndPropagatePType();
  TensorIndexer indexer(id_model);

  // The first merge of the logical domains should be a trivial merge,
  // i.e., a merge with a extent-one domain. Thus, the indexing
  // travesal should return "x + y * 4", where x and y are the loop
  // indices, respecitvely.

  std::vector<Val*> tv1_loop_indices = getLoopIndices(tv1, indexer);
  std::vector<Val*> tv2_loop_indices = getLoopIndices(tv2, indexer);

  auto tv0_producer_index = indexer.getLinearIndex(tv0, tv1->definition());
  auto tv1_consumer_index = indexer.getLinearIndex(tv1, tv1->definition());
  auto tv1_producer_index = indexer.getLinearIndex(tv1, tv2->definition());
  auto tv2_consumer_index = indexer.getLinearIndex(tv2, tv2->definition());

  // tv0 is a global memory tensor, so the indexing is done with its
  // allocation domain, which is mapped with the merge of the two
  // logical domains of tv1 on the AlmostExact graph. Traverse back to
  // the merge output from the loop domains.
  auto tv0_producer_index_ref = addExpr(
      mulExpr(tv1_loop_indices.at(0), tv1->axis(1)->extent()),
      tv1_loop_indices.at(1));

  // tv1 is a Local tensor, so its allocation domains are just their
  // loop domains. This index is mathematically equivalent to the tv0
  // index, but the order of linearizing the two loop domains is
  // different from the order of computing the merge input index.
  auto tv1_consumer_index_ref = addExpr(
      tv1_loop_indices.at(1),
      mulExpr(tv1_loop_indices.at(0), tv1->axis(1)->extent()));

  auto tv1_producer_index_ref = addExpr(
      tv2_loop_indices.at(1),
      mulExpr(tv2_loop_indices.at(0), tv2->axis(1)->extent()));

  auto tv2_consumer_index_ref = addExpr(
      mulExpr(tv2_loop_indices.at(0), tv2->axis(1)->extent()),
      tv2_loop_indices.at(1));

  EXPECT_TRUE(tv0_producer_index->sameAs(tv0_producer_index_ref))
      << "Ref: " << tv0_producer_index_ref->toInlineString()
      << ". Actual: " << tv0_producer_index->toInlineString();

  EXPECT_TRUE(tv1_consumer_index->sameAs(tv1_consumer_index_ref))
      << "Ref: " << tv1_consumer_index_ref->toInlineString()
      << ". Actual: " << tv1_consumer_index->toInlineString();

  EXPECT_TRUE(tv1_producer_index->sameAs(tv1_producer_index_ref))
      << "Ref: " << tv1_producer_index_ref->toInlineString()
      << ". Actual: " << tv1_producer_index->toInlineString();

  EXPECT_TRUE(tv2_consumer_index->sameAs(tv2_consumer_index_ref))
      << "Ref: " << tv2_consumer_index_ref->toInlineString()
      << ". Actual: " << tv2_consumer_index->toInlineString();
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

  IdModel id_model(&fusion);
  id_model.validateAndPropagatePType();
  TensorIndexer indexer(id_model);

  std::vector<Val*> tv3_loop_indices = getLoopIndices(tv3, indexer);

  // Start with tv3 index as it's most straightforward
  auto tv3_consumer_index = indexer.getLinearIndex(tv3, tv3->definition());
  auto tv3_consumer_index_ref = addExpr(
      modExpr(tv3_loop_indices.at(0), tv3->getLogicalDomain().at(1)->extent()),
      mulExpr(
          divExpr(
              tv3_loop_indices.at(0), tv3->getLogicalDomain().at(1)->extent()),
          tv3->getLogicalDomain().at(1)->extent()));

  EXPECT_TRUE(tv3_consumer_index->sameAs(tv3_consumer_index_ref))
      << "Ref: " << tv3_consumer_index_ref->toInlineString()
      << ". Actual: " << tv3_consumer_index->toInlineString();

  // Since tv2 is fully inlined, its index should be just zero
  auto tv2_consumer_index = indexer.getLinearIndex(tv2, tv2->definition());
  auto tv2_producer_index = indexer.getLinearIndex(tv2, tv3->definition());

  EXPECT_TRUE(tv2_consumer_index->isZeroInt());
  EXPECT_TRUE(tv2_producer_index->isZeroInt());

  // tv0 is a 1D pre-broadcast input tensor, so it only needs the
  // index that corresponds to the outer dimension of the tv3 (or tv2)
  // logical domains
  auto tv0_producer_index = indexer.getLinearIndex(tv0, tv2->definition());
  auto tv0_producer_index_ref =
      divExpr(tv3_loop_indices.at(0), tv3->getLogicalDomain().at(1)->extent());

  EXPECT_TRUE(tv0_producer_index->sameAs(tv0_producer_index_ref))
      << "Ref: " << tv0_producer_index_ref->toInlineString()
      << ". Actual: " << tv0_producer_index->toInlineString();

  // tv1 should have the same index as tv3
  auto tv1_producer_index = indexer.getLinearIndex(tv1, tv3->definition());
  EXPECT_TRUE(tv1_producer_index->sameAs(tv3_consumer_index_ref))
      << "Ref: " << tv3_consumer_index_ref->toInlineString()
      << ". Actual: " << tv1_producer_index->toInlineString();
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

  IdModel id_model(&fusion);
  id_model.validateAndPropagatePType();
  TensorIndexer indexer(id_model);

  // As discussed in the doc, the inner domain of tv2 is promoted to
  // a domain with the same extent as the inner domain of tv4. Since
  // tv2 is a Local tensor, its allocation domain is also promoted to
  // the same domain. Thus, its consumer index is just the loop index
  // of the inner loop of the tv2 loop domains, and its producer index
  // is also just the inner loop index of the loop domains of tv4.

  std::vector<Val*> tv2_loop_indices = getLoopIndices(tv2, indexer);
  std::vector<Val*> tv4_loop_indices = getLoopIndices(tv4, indexer);

  auto tv2_consumer_index = indexer.getLinearIndex(tv2, tv2->definition());
  auto tv2_producer_index = indexer.getLinearIndex(tv2, tv4->definition());

  EXPECT_EQ(tv2_consumer_index, tv2_loop_indices.at(1));
  EXPECT_EQ(tv2_producer_index, tv4_loop_indices.at(1));
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

  IdModel id_model(&fusion);
  TensorIndexer indexer(id_model);

  EXPECT_TRUE(indexer.getLinearIndex(tv0, tv1->definition())->isZeroInt());
  EXPECT_TRUE(indexer.getLinearIndex(tv1, tv1->definition())->isZeroInt());
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

  IdModel id_model(&fusion);
  TensorIndexer indexer(id_model);

  std::vector<Val*> tv1_loop_indices = getLoopIndices(tv1, indexer);

  auto tv0_producer_index = indexer.getLinearIndex(tv0, tv1->definition());
  auto tv1_consumer_index = indexer.getLinearIndex(tv1, tv1->definition());

  auto tv0_producer_index_ref = tv1_loop_indices.at(1);
  auto tv1_consumer_index_ref = tv1_loop_indices.at(1);

  EXPECT_TRUE(tv0_producer_index->sameAs(tv0_producer_index_ref))
      << "Ref: " << tv0_producer_index_ref->toInlineString()
      << ". Actual: " << tv0_producer_index->toInlineString();

  EXPECT_TRUE(tv1_consumer_index->sameAs(tv1_consumer_index_ref))
      << "Ref: " << tv1_consumer_index_ref->toInlineString()
      << ". Actual: " << tv1_consumer_index->toInlineString();
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

  IdModel id_model(&fusion);
  TensorIndexer indexer(id_model);

  std::vector<Val*> tv1_loop_indices = getLoopIndices(tv1, indexer);

  auto tv0_producer_index = indexer.getLinearIndex(tv0, tv1->definition());
  auto tv1_consumer_index = indexer.getLinearIndex(tv1, tv1->definition());

  auto inner_dim = tv1->getLogicalDomain().at(1)->extent();

  // Note that the allocation domain is the logical domain. See the
  // next test for a loop allocation example
  auto tv0_producer_index_ref = addExpr(
      modExpr(tv1_loop_indices.at(1), inner_dim),
      mulExpr(divExpr(tv1_loop_indices.at(1), inner_dim), inner_dim));

  // Should use the same index
  auto tv1_consumer_index_ref = tv0_producer_index_ref;

  EXPECT_TRUE(tv0_producer_index->sameAs(tv0_producer_index_ref))
      << "Ref: " << tv0_producer_index_ref->toInlineString()
      << ". Actual: " << tv0_producer_index->toInlineString();

  EXPECT_TRUE(tv1_consumer_index->sameAs(tv1_consumer_index_ref))
      << "Ref: " << tv1_consumer_index_ref->toInlineString()
      << ". Actual: " << tv1_consumer_index->toInlineString();
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

  IdModel id_model(&fusion);
  TensorIndexer indexer(id_model);

  std::vector<Val*> tv1_loop_indices = getLoopIndices(tv1, indexer);

  auto tv0_producer_index = indexer.getLinearIndex(tv0, tv1->definition());
  auto tv1_consumer_index = indexer.getLinearIndex(tv1, tv1->definition());

  // Since the loop domain is the allocation domain, the index should
  // be just the non-parallelized loop index
  auto tv0_producer_index_ref = tv1_loop_indices.at(1);

  // Should use the same index
  auto tv1_consumer_index_ref = tv0_producer_index_ref;

  EXPECT_TRUE(tv0_producer_index->sameAs(tv0_producer_index_ref))
      << "Ref: " << tv0_producer_index_ref->toInlineString()
      << ". Actual: " << tv0_producer_index->toInlineString();

  EXPECT_TRUE(tv1_consumer_index->sameAs(tv1_consumer_index_ref))
      << "Ref: " << tv1_consumer_index_ref->toInlineString()
      << ". Actual: " << tv1_consumer_index->toInlineString();
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

  IdModel id_model(&fusion);
  TensorIndexer indexer(id_model);

  std::vector<Val*> tv1_loop_indices = getLoopIndices(tv1, indexer);

  auto tv0_producer_index = indexer.getLinearIndex(tv0, tv1->definition());
  auto tv1_consumer_index = indexer.getLinearIndex(tv1, tv1->definition());

  auto tv0_producer_index_ref = addExpr(
      tv1_loop_indices.at(1),
      mulExpr(tv1_loop_indices.at(2), tv0->getLogicalDomain().at(1)->extent()));

  // Should use the same index
  auto tv1_consumer_index_ref = addExpr(
      tv1_loop_indices.at(2),
      mulExpr(tv1_loop_indices.at(1), tv1->getLogicalDomain().at(1)->extent()));

  EXPECT_TRUE(tv0_producer_index->sameAs(tv0_producer_index_ref))
      << "Ref: " << tv0_producer_index_ref->toInlineString()
      << ". Actual: " << tv0_producer_index->toInlineString();

  EXPECT_TRUE(tv1_consumer_index->sameAs(tv1_consumer_index_ref))
      << "Ref: " << tv1_consumer_index_ref->toInlineString()
      << ". Actual: " << tv1_consumer_index->toInlineString();
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

  IdModel id_model(&fusion);
  id_model.validateAndPropagatePType();

  ASSERT_EQ(tv2->axis(0)->getParallelType(), ParallelType::TIDx);

  TensorIndexer indexer(id_model);

  std::vector<Val*> tv2_loop_indices = getLoopIndices(tv2, indexer);
  std::vector<Val*> tv3_loop_indices = getLoopIndices(tv3, indexer);

  auto tv2_consumer_index = indexer.getLinearIndex(tv2, tv2->definition());
  auto tv2_producer_index = indexer.getLinearIndex(tv2, tv3->definition());

  EXPECT_TRUE(tv2_consumer_index->isZero());
  EXPECT_TRUE(tv2_producer_index->isZero());
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

  fusion.print();
  fusion.printKernel();

  IdModel id_model(&fusion);
  id_model.validateAndPropagatePType();
  TensorIndexer indexer(id_model);

  printAllIndices(std::cerr, &fusion, indexer);

  std::vector<Val*> tv1_loop_indices = getLoopIndices(tv1, indexer);
  std::vector<Val*> tv2_loop_indices = getLoopIndices(tv2, indexer);

  auto tv0_producer_index = indexer.getLinearIndex(tv0, tv1->definition());
  auto tv1_consumer_index = indexer.getLinearIndex(tv1, tv1->definition());
  auto tv1_producer_index = indexer.getLinearIndex(tv1, tv2->definition());
  auto tv2_consumer_index = indexer.getLinearIndex(tv2, tv2->definition());

  // Vectorized index is generated by replacing the index of a vectorized domain
  // with zero, which doesn't go through the simplification of
  // SimplifyingIrBuilder. We could use simplifyExpr, but for the sake
  // of testing, just use IrBuilder::addExpr.
  auto tv0_producer_index_ref = IrBuilder::addExpr(
      mulExpr(
          addExpr(
              mulExpr(tv1_loop_indices.at(0), tv0->axis(1)->extent()),
              tv1_loop_indices.at(1)),
          tv0->axis(2)->extent()),
      fusion.zeroVal());

  EXPECT_TRUE(tv0_producer_index->sameAs(tv0_producer_index_ref))
      << "Ref: " << tv0_producer_index_ref->toInlineString()
      << ". Actual: " << tv0_producer_index->toInlineString();

  EXPECT_TRUE(tv1_consumer_index->isZero());
  EXPECT_TRUE(tv1_producer_index->isZero());

  // tv2 consumer index should be the same as tv0 producer index
  EXPECT_TRUE(tv2_consumer_index->sameAs(tv0_producer_index_ref))
      << "Ref: " << tv0_producer_index_ref->toInlineString()
      << ". Actual: " << tv2_consumer_index->toInlineString();
}

} // namespace nvfuser
