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

namespace nvfuser {

using IndexingTest = NVFuserTest;

namespace {

std::vector<Val*> getLoopIndices(TensorView* tv, const TensorIndexer& indexer) {
  std::vector<Val*> loop_indices;
  for (const auto& loop_id : tv->getLeafDomain()) {
    loop_indices.push_back(indexer.getLoopIndex(loop_id));
  }
  return loop_indices;
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
              (int64_t)1)),
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
              (int64_t)0)));

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

} // namespace nvfuser
