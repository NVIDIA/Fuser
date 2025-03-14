// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#include <gmock/gmock-matchers.h>
#include <gtest/gtest.h>

#include <tests/cpp/utils.h>

#include <scheduler/tools/abstract_tensor.h>

namespace nvfuser {

class AbstractTensorTest : public NVFuserTest {
  std::unique_ptr<Fusion> fusion_ptr_;
  std::unique_ptr<FusionGuard> fusion_guard_ptr_;
  std::unique_ptr<IterDomainBuilder> builder_;

  void SetUp() override {
    NVFuserTest::SetUp();
    fusion_ptr_ = std::make_unique<Fusion>();
    fusion_guard_ptr_ = std::make_unique<FusionGuard>(fusion_ptr_.get());
    auto size = IrBuilder::create<Val>(16, DataType::Index);
    builder_ =
        std::make_unique<IterDomainBuilder>(fusion_ptr_->zeroVal(), size);
  }

 protected:
  IterDomain* newID() const {
    return builder_->build();
  }
};

TEST_F(AbstractTensorTest, UseAbstractIdAsIdPtr) {
  auto id0 = newID();
  AbstractTensor v({id0});
  v[0]->parallelize(ParallelType::TIDx);
  EXPECT_EQ(id0->getParallelType(), ParallelType::TIDx);
}

TEST_F(AbstractTensorTest, MergeSingleIterDomains) {
  auto id0 = newID();
  auto id1 = newID();
  auto id2 = newID();
  auto id3 = newID();
  auto id4 = newID();
  AbstractTensor v({id0, id1, id2, id3, id4});
  v.merge(2);
  // [0, 1, 2*3, 4]
  v.merge(0, 3);
  // [0*4, 1, 2*3]
  v.merge(-1, -3);
  // [(2*3) * (0*4), 1]
  auto result = v.as<IterDomain*>();
  ASSERT_EQ(result.size(), 2);
  EXPECT_EQ(result[1], id1);
  auto m = dynamic_cast<Merge*>(result[0]->definition());
  ASSERT_NE(m, nullptr);
  auto id23 = m->outer();
  auto id04 = m->inner();
  ASSERT_NE(id23, nullptr);
  ASSERT_NE(id04, nullptr);
  auto m23 = dynamic_cast<Merge*>(id23->definition());
  auto m04 = dynamic_cast<Merge*>(id04->definition());
  ASSERT_NE(m23, nullptr);
  ASSERT_NE(m04, nullptr);
  EXPECT_EQ(m23->outer(), id2);
  EXPECT_EQ(m23->inner(), id3);
  EXPECT_EQ(m04->outer(), id0);
  EXPECT_EQ(m04->inner(), id4);
}

TEST_F(AbstractTensorTest, MergeIterDomainsLeftBroadcasting) {
  auto id0 = newID();
  auto id1 = newID();
  auto id2 = newID();
  AbstractTensor v({id0, {id1, id2}});
  v.merge(0);
  // [{0*1, 0*2}]
  auto result = v.as<std::vector<IterDomain*>>();
  ASSERT_EQ(result.size(), 1);
  ASSERT_EQ(result[0].size(), 2);
  auto id01 = result[0][0];
  auto id02 = result[0][1];
  ASSERT_NE(id01, nullptr);
  ASSERT_NE(id02, nullptr);
  auto m01 = dynamic_cast<Merge*>(id01->definition());
  auto m02 = dynamic_cast<Merge*>(id02->definition());
  ASSERT_NE(m01, nullptr);
  ASSERT_NE(m02, nullptr);
  EXPECT_EQ(m01->outer(), id0);
  EXPECT_EQ(m01->inner(), id1);
  EXPECT_EQ(m02->outer(), id0);
  EXPECT_EQ(m02->inner(), id2);
}

TEST_F(AbstractTensorTest, MergeIterDomainsRightBroadcasting) {
  auto id0 = newID();
  auto id1 = newID();
  auto id2 = newID();
  AbstractTensor v({{id0, id1}, id2});
  v.merge(0);
  // [{0*2, 1*2}]
  auto result = v.as<std::vector<IterDomain*>>();
  ASSERT_EQ(result.size(), 1);
  ASSERT_EQ(result[0].size(), 2);
  auto id02 = result[0][0];
  auto id12 = result[0][1];
  ASSERT_NE(id02, nullptr);
  ASSERT_NE(id12, nullptr);
  auto m02 = dynamic_cast<Merge*>(id02->definition());
  auto m12 = dynamic_cast<Merge*>(id12->definition());
  ASSERT_NE(m02, nullptr);
  ASSERT_NE(m12, nullptr);
  EXPECT_EQ(m02->outer(), id0);
  EXPECT_EQ(m02->inner(), id2);
  EXPECT_EQ(m12->outer(), id1);
  EXPECT_EQ(m12->inner(), id2);
}

TEST_F(AbstractTensorTest, MergeIterDomainsBatch) {
  auto id0 = newID();
  auto id1 = newID();
  auto id2 = newID();
  auto id3 = newID();
  AbstractTensor v({{id0, id1}, {id2, id3}});
  v.merge(0);
  // [{0*2, 1*3}]
  auto result = v.as<std::vector<IterDomain*>>();
  ASSERT_EQ(result.size(), 1);
  ASSERT_EQ(result[0].size(), 2);
  auto id02 = result[0][0];
  auto id13 = result[0][1];
  ASSERT_NE(id02, nullptr);
  ASSERT_NE(id13, nullptr);
  auto m02 = dynamic_cast<Merge*>(id02->definition());
  auto m13 = dynamic_cast<Merge*>(id13->definition());
  ASSERT_NE(m02, nullptr);
  ASSERT_NE(m13, nullptr);
  EXPECT_EQ(m02->outer(), id0);
  EXPECT_EQ(m02->inner(), id2);
  EXPECT_EQ(m13->outer(), id1);
  EXPECT_EQ(m13->inner(), id3);
}

TEST_F(AbstractTensorTest, MergeValGroups) {
  auto id0 = newID();
  auto id1 = newID();
  ValGraph g;
  g.initializeVal(id0);
  g.initializeVal(id1);
  ValGroupAndItsGraph g0{g.toGroup(id0), &g};
  ValGroupAndItsGraph g1{g.toGroup(id1), &g};
  AbstractTensor v({g0, g1});
  v.merge(0);
  // [0*1]
  EXPECT_EQ(g.disjointValSets().size(), 3);
  EXPECT_EQ(g.disjointExprSets().size(), 1);
  auto result = v.as<ValGroupAndItsGraph>();
  ASSERT_EQ(result.size(), 1);
  EXPECT_EQ(result[0].graph, &g);
  auto g01 = result[0];
  EXPECT_NE(g01, g0);
  EXPECT_NE(g01, g1);
  auto defs = g.getDefinitions(g01.group);
  ASSERT_EQ(defs.size(), 1);
  auto eg01 = defs.front();
  auto expect01 = std::vector<ValGroup>{g0.group, g1.group};
  EXPECT_EQ(g.inputGroups(eg01), expect01);
  EXPECT_EQ(g.outputGroups(eg01), std::vector<ValGroup>{g01.group});
  auto uses0 = g.getUses(g0.group);
  auto uses1 = g.getUses(g1.group);
  EXPECT_EQ(uses0.size(), 1);
  EXPECT_EQ(uses1.size(), 1);
  EXPECT_EQ(uses0.vector(), uses1.vector());
  EXPECT_EQ(uses0.front(), eg01);

  // Test reusing of existing merge
  AbstractTensor vv({g0, g1});
  vv.merge(0);
  EXPECT_EQ(g.disjointValSets().size(), 3);
  EXPECT_EQ(g.disjointExprSets().size(), 1);
  auto result2 = v.as<ValGroupAndItsGraph>();
  ASSERT_EQ(result2.size(), 1);
  EXPECT_EQ(result2[0], g01);
}

TEST_F(AbstractTensorTest, MergeIterDomainWithValGroup) {
  auto id0 = newID();
  auto id1 = newID();
  ValGraph g;
  g.initializeVal(id0);
  g.initializeVal(id1);
  ValGroupAndItsGraph g1{g.toGroup(id1), &g};
  AbstractTensor v({id0, g1});
  v.merge(0);
  // [0*1]
  EXPECT_EQ(g.disjointValSets().size(), 3);
  EXPECT_EQ(g.disjointExprSets().size(), 1);
  auto result = v.as<ValGroupAndItsGraph>();
  ASSERT_EQ(result.size(), 1);
  EXPECT_EQ(result[0].graph, &g);
  auto g01 = result[0];
  EXPECT_NE(g01.group, g.toGroup(id0));
  EXPECT_NE(g01, g1);
  auto defs = g.getDefinitions(g01.group);
  ASSERT_EQ(defs.size(), 1);
  auto eg01 = defs.front();
  auto expect01 = std::vector<ValGroup>{g.toGroup(id0), g1.group};
  EXPECT_EQ(g.inputGroups(eg01), expect01);
  EXPECT_EQ(g.outputGroups(eg01), std::vector<ValGroup>{g01.group});
  auto uses0 = g.getUses(g.toGroup(id0));
  auto uses1 = g.getUses(g1.group);
  EXPECT_EQ(uses0.size(), 1);
  EXPECT_EQ(uses1.size(), 1);
  EXPECT_EQ(uses0.vector(), uses1.vector());
  EXPECT_EQ(uses0.front(), eg01);
}

TEST_F(AbstractTensorTest, MergeValGroupWithIterDomain) {
  auto id0 = newID();
  auto id1 = newID();
  ValGraph g;
  g.initializeVal(id0);
  g.initializeVal(id1);
  ValGroupAndItsGraph g0{g.toGroup(id0), &g};
  AbstractTensor v({g0, id1});
  v.merge(0);
  // [0*1]
  EXPECT_EQ(g.disjointValSets().size(), 3);
  EXPECT_EQ(g.disjointExprSets().size(), 1);
  auto result = v.as<ValGroupAndItsGraph>();
  ASSERT_EQ(result.size(), 1);
  EXPECT_EQ(result[0].graph, &g);
  auto g01 = result[0];
  EXPECT_NE(g01, g0);
  EXPECT_NE(g01.group, g.toGroup(id1));
  auto defs = g.getDefinitions(g01.group);
  ASSERT_EQ(defs.size(), 1);
  auto eg01 = defs.front();
  auto expect01 = std::vector<ValGroup>{g0.group, g.toGroup(id1)};
  EXPECT_EQ(g.inputGroups(eg01), expect01);
  EXPECT_EQ(g.outputGroups(eg01), std::vector<ValGroup>{g01.group});
  auto uses0 = g.getUses(g0.group);
  auto uses1 = g.getUses(g.toGroup(id1));
  EXPECT_EQ(uses0.size(), 1);
  EXPECT_EQ(uses1.size(), 1);
  EXPECT_EQ(uses0.vector(), uses1.vector());
  EXPECT_EQ(uses0.front(), eg01);
}

TEST_F(AbstractTensorTest, SplitSingleIterDomain) {
  auto id0 = newID();
  auto id1 = newID();
  auto id2 = newID();
  AbstractTensor v({id0, id1, id2});
  v.split(1, 5);
  // [0, 1/5, 5, 2]
  v.split(1, 6);
  // [0, 1/5/6, 6, 5, 2]
  auto result = v.as<IterDomain*>();
  ASSERT_EQ(result.size(), 5);
  EXPECT_EQ(result[0], id0);
  EXPECT_EQ(result[4], id2);
  auto id156 = result[1];
  auto id6 = result[2];
  auto id5 = result[3];
  auto split6 = dynamic_cast<Split*>(id6->definition());
  ASSERT_NE(split6, nullptr);
  EXPECT_EQ(id156->definition(), split6);
  EXPECT_EQ(id156, split6->outer());
  EXPECT_EQ(id6, split6->inner());
  auto id15 = split6->in();
  auto split5 = dynamic_cast<Split*>(id5->definition());
  ASSERT_NE(split5, nullptr);
  EXPECT_EQ(id15->definition(), split5);
  EXPECT_EQ(id15, split5->outer());
  EXPECT_EQ(id5, split5->inner());
  EXPECT_EQ(id1, split5->in());
}

TEST_F(AbstractTensorTest, SplitIterDomainBatch) {
  auto id0 = newID();
  auto id1 = newID();
  auto id2 = newID();
  auto id3 = newID();
  AbstractTensor v({id0, {id1, id2}, id3});
  v.split(1, 5);
  // [0, {1/5,2/5}, {5, 5}, 3]
  v.split(1, 6);
  // [0, {1/5/6,2/5/6}, {6, 6}, {5, 5}, 3]
  ASSERT_EQ(v.size(), 5);
  EXPECT_EQ(v[0], id0);
  EXPECT_EQ(v[4], id3);
  auto ids1256 = v[1];
  auto ids6 = v[2];
  auto ids5 = v[3];
  ASSERT_EQ(ids1256.as<std::vector>().size(), 2);
  ASSERT_EQ(ids6.as<std::vector>().size(), 2);
  ASSERT_EQ(ids5.as<std::vector>().size(), 2);
  IterDomain* ids12[2]{id1, id2};
  for (auto i : {0, 1}) {
    auto id6 = ids6[i].as<IterDomain*>();
    auto id5 = ids5[i].as<IterDomain*>();
    auto id1256 = ids1256[i].as<IterDomain*>();
    auto split6 = dynamic_cast<Split*>(id6->definition());
    ASSERT_NE(split6, nullptr);
    EXPECT_EQ(id1256->definition(), split6);
    EXPECT_EQ(id1256, split6->outer());
    EXPECT_EQ(id6, split6->inner());
    auto id125 = split6->in();
    auto split5 = dynamic_cast<Split*>(id5->definition());
    ASSERT_NE(split5, nullptr);
    EXPECT_EQ(id125->definition(), split5);
    EXPECT_EQ(id125, split5->outer());
    EXPECT_EQ(id5, split5->inner());
    EXPECT_EQ(ids12[i], split5->in());
  }
}

TEST_F(AbstractTensorTest, SplitValGroup) {
  auto id0 = newID();
  auto id1 = newID();
  auto id2 = newID();
  ValGraph g;
  g.initializeVal(id0);
  g.initializeVal(id1);
  g.initializeVal(id2);
  ValGroupAndItsGraph g0{g.toGroup(id0), &g};
  ValGroupAndItsGraph g1{g.toGroup(id1), &g};
  ValGroupAndItsGraph g2{g.toGroup(id2), &g};
  AbstractTensor v1({g0, g1, g2});
  v1.split(1, 5);
  // [0, 1/5, 5, 2]
  AbstractTensor v2({g0, g1, g2});
  v2.split(1, 6);
  // [0, 1/6, 6, 2]
  EXPECT_EQ(g.disjointValSets().size(), 7);
  EXPECT_EQ(g.disjointExprSets().size(), 2);

  EXPECT_NE(v1[1], v2[1]);
  EXPECT_NE(v1[2], v2[2]);
  EXPECT_EQ(v1[0], v2[0]);
  EXPECT_EQ(v1[0], g0);
  EXPECT_EQ(v1[3], v2[3]);
  EXPECT_EQ(v1[3], g2);

  for (auto v : {v1, v2}) {
    auto result = v.as<ValGroupAndItsGraph>();
    ASSERT_EQ(result.size(), 4);
    EXPECT_EQ(result[1].graph, &g);
    EXPECT_EQ(result[2].graph, &g);
    EXPECT_EQ(result[0], g0);
    EXPECT_EQ(result[3], g2);
    auto g156 = result[1];
    auto g56 = result[2];
    EXPECT_NE(g156, g0);
    EXPECT_NE(g156, g1);
    EXPECT_NE(g156, g2);
    EXPECT_NE(g56, g0);
    EXPECT_NE(g56, g1);
    EXPECT_NE(g56, g2);
    EXPECT_NE(g56, g156);
    auto defs = g.getDefinitions(g156.group);
    ASSERT_EQ(defs.size(), 1);
    auto eg56 = defs.front();
    auto expect56 = std::vector<ValGroup>{g156.group, g56.group};
    EXPECT_EQ(g.outputGroups(eg56), expect56);
    EXPECT_EQ(g.inputGroups(eg56), std::vector<ValGroup>{g1.group});
    auto uses = g.getUses(g1.group);
    EXPECT_EQ(uses.size(), 2);
    EXPECT_TRUE(uses.has(eg56));
  }

  // Test reusing of existing split
  AbstractTensor vv1({g1});
  vv1.split(0, 5);
  EXPECT_EQ(g.disjointValSets().size(), 7);
  EXPECT_EQ(g.disjointExprSets().size(), 2);
  ASSERT_EQ(vv1.size(), 2);
  EXPECT_EQ(vv1[0], v1[1]);
  EXPECT_EQ(vv1[1], v1[2]);

  AbstractTensor vv2({g1});
  vv2.split(0, 6);
  EXPECT_EQ(g.disjointValSets().size(), 7);
  EXPECT_EQ(g.disjointExprSets().size(), 2);
  ASSERT_EQ(vv2.size(), 2);
  EXPECT_EQ(vv2[0], v2[1]);
  EXPECT_EQ(vv2[1], v2[2]);
}

TEST_F(AbstractTensorTest, Reorder) {
  auto id0 = newID();
  auto id1 = newID();
  auto id2 = newID();
  auto id3 = newID();
  using IDs = std::vector<AbstractId>;
  AbstractTensor v({id0, id1, id2, id3});
  v.reorder({{0, 1}});
  IDs expect0 = {id1, id0, id2, id3};
  EXPECT_EQ(v, expect0);
  v.reorder({{-1, 1}});
  IDs expect1 = {id1, id3, id0, id2};
  EXPECT_EQ(v, expect1);
  v.reorder({2, 3, 0, 1});
  IDs expect2 = {id0, id2, id1, id3};
  EXPECT_EQ(v, expect2);
  v.reorder({{1, 2}});
  IDs expect3 = {id0, id1, id2, id3};
  EXPECT_EQ(v, expect3);
  v.reorder({{0, 1}, {1, 2}});
  IDs expect4 = {id2, id0, id1, id3};
  EXPECT_EQ(v, expect4);
}

TEST_F(AbstractTensorTest, Flatten) {
  auto id0 = newID();
  auto id1 = newID();
  auto id2 = newID();
  auto id3 = newID();
  AbstractTensor v({id0, id1, id2, id3});
  v.flatten(1, 2);
  ASSERT_EQ(v.size(), 3);
  EXPECT_EQ(v[0], id0);
  EXPECT_EQ(v[2], id3);
  auto merge = dynamic_cast<Merge*>(v[1]->definition());
  ASSERT_NE(merge, nullptr);
  EXPECT_EQ(merge->inner(), id2);
  EXPECT_EQ(merge->outer(), id1);
}

TEST_F(AbstractTensorTest, SwizzleSingleIterDomains) {
  auto id0 = newID();
  auto id1 = newID();
  auto id2 = newID();
  auto id3 = newID();
  AbstractTensor v({id0, id1, id2, id3});
  v.swizzle(SwizzleType::XOR, 1, 2);
  // [0, 1', 2', 3]
  auto result = v.as<IterDomain*>();
  ASSERT_EQ(result.size(), 4);
  EXPECT_EQ(result[0], id0);
  EXPECT_EQ(result[3], id3);
  auto s1 = dynamic_cast<Swizzle*>(result[1]->definition());
  auto s2 = dynamic_cast<Swizzle*>(result[2]->definition());
  ASSERT_NE(s1, nullptr);
  EXPECT_EQ(s1, s2);
  EXPECT_EQ(s1->inX(), id1);
  EXPECT_EQ(s1->inY(), id2);
}

TEST_F(AbstractTensorTest, SwizzleIterDomainsLeftBroadcasting) {
  auto id0 = newID();
  auto id1 = newID();
  auto id2 = newID();
  AbstractTensor v({id0, {id1, id2}});
  v.swizzle(SwizzleType::XOR, 0, 1);
  // [{0', 0"}, {1', 2"}]

  auto result = v.as<std::vector<IterDomain*>>();
  ASSERT_EQ(result.size(), 2);
  ASSERT_EQ(result[0].size(), 2);
  ASSERT_EQ(result[1].size(), 2);

  auto s1 = dynamic_cast<Swizzle*>(result[0][0]->definition());
  auto s2 = dynamic_cast<Swizzle*>(result[1][0]->definition());
  ASSERT_NE(s1, nullptr);
  EXPECT_EQ(s1, s2);
  EXPECT_EQ(s1->inX(), id0);
  EXPECT_EQ(s1->inY(), id1);

  s1 = dynamic_cast<Swizzle*>(result[0][1]->definition());
  s2 = dynamic_cast<Swizzle*>(result[1][1]->definition());
  ASSERT_NE(s1, nullptr);
  EXPECT_EQ(s1, s2);
  EXPECT_EQ(s1->inX(), id0);
  EXPECT_EQ(s1->inY(), id2);
}

TEST_F(AbstractTensorTest, SwizzleIterDomainsRightBroadcasting) {
  auto id0 = newID();
  auto id1 = newID();
  auto id2 = newID();
  AbstractTensor v({{id0, id1}, id2});
  v.swizzle(SwizzleType::XOR, 0, 1);
  // [{0', 1"}, {2', 2"}]

  auto result = v.as<std::vector<IterDomain*>>();
  ASSERT_EQ(result.size(), 2);
  ASSERT_EQ(result[0].size(), 2);
  ASSERT_EQ(result[1].size(), 2);

  auto s1 = dynamic_cast<Swizzle*>(result[0][0]->definition());
  auto s2 = dynamic_cast<Swizzle*>(result[1][0]->definition());
  ASSERT_NE(s1, nullptr);
  EXPECT_EQ(s1, s2);
  EXPECT_EQ(s1->inX(), id0);
  EXPECT_EQ(s1->inY(), id2);

  s1 = dynamic_cast<Swizzle*>(result[0][1]->definition());
  s2 = dynamic_cast<Swizzle*>(result[1][1]->definition());
  ASSERT_NE(s1, nullptr);
  EXPECT_EQ(s1, s2);
  EXPECT_EQ(s1->inX(), id1);
  EXPECT_EQ(s1->inY(), id2);
}

TEST_F(AbstractTensorTest, SwizzleIterDomainsBatch) {
  auto id0 = newID();
  auto id1 = newID();
  auto id2 = newID();
  auto id3 = newID();
  AbstractTensor v({{id0, id1}, {id2, id3}});
  v.swizzle(SwizzleType::XOR, 0, 1);
  // [{0', 1"}, {2', 3"}]

  auto result = v.as<std::vector<IterDomain*>>();
  ASSERT_EQ(result.size(), 2);
  ASSERT_EQ(result[0].size(), 2);
  ASSERT_EQ(result[1].size(), 2);

  auto s1 = dynamic_cast<Swizzle*>(result[0][0]->definition());
  auto s2 = dynamic_cast<Swizzle*>(result[1][0]->definition());
  ASSERT_NE(s1, nullptr);
  EXPECT_EQ(s1, s2);
  EXPECT_EQ(s1->inX(), id0);
  EXPECT_EQ(s1->inY(), id2);

  s1 = dynamic_cast<Swizzle*>(result[0][1]->definition());
  s2 = dynamic_cast<Swizzle*>(result[1][1]->definition());
  ASSERT_NE(s1, nullptr);
  EXPECT_EQ(s1, s2);
  EXPECT_EQ(s1->inX(), id1);
  EXPECT_EQ(s1->inY(), id3);
}

TEST_F(AbstractTensorTest, SwizzleValGroups) {
  auto id0 = newID();
  auto id1 = newID();
  ValGraph g;
  g.initializeVal(id0);
  g.initializeVal(id1);
  ValGroupAndItsGraph g0{g.toGroup(id0), &g};
  ValGroupAndItsGraph g1{g.toGroup(id1), &g};
  AbstractTensor v({g0, g1});
  v.swizzle(SwizzleType::XOR, 0, 1);
  // [0', 1']

  EXPECT_EQ(g.disjointValSets().size(), 4);
  EXPECT_EQ(g.disjointExprSets().size(), 1);
  auto result = v.as<ValGroupAndItsGraph>();
  ASSERT_EQ(result.size(), 2);
  EXPECT_EQ(result[0].graph, &g);
  EXPECT_EQ(result[1].graph, &g);

  EXPECT_NE(result[0], g0);
  EXPECT_NE(result[1], g0);
  EXPECT_NE(result[0], g1);
  EXPECT_NE(result[1], g1);
  EXPECT_NE(result[0], result[1]);

  auto defs0 = g.getDefinitions(result[0].group);
  auto defs1 = g.getDefinitions(result[1].group);
  ASSERT_EQ(defs0, defs1);
  ASSERT_EQ(defs0.size(), 1);

  auto eg = defs1.front();
  auto expect_in = std::vector<ValGroup>{g0.group, g1.group};
  auto expect_out = std::vector<ValGroup>{result[0].group, result[1].group};
  EXPECT_EQ(g.inputGroups(eg), expect_in);
  EXPECT_EQ(g.outputGroups(eg), expect_out);

  auto uses0 = g.getUses(g0.group);
  auto uses1 = g.getUses(g1.group);
  EXPECT_EQ(uses0, uses1);
  EXPECT_EQ(uses0.size(), 1);
  EXPECT_EQ(uses0.front(), eg);

  // Test reusing of existing swizzle
  AbstractTensor vv({g0, g1});
  vv.swizzle(SwizzleType::XOR, 0, 1);
  EXPECT_EQ(g.disjointValSets().size(), 4);
  EXPECT_EQ(g.disjointExprSets().size(), 1);
  EXPECT_EQ(v, vv);
}

TEST_F(AbstractTensorTest, SwizzleIterDomainWithValGroup) {
  auto id0 = newID();
  auto id1 = newID();
  ValGraph g;
  g.initializeVal(id0);
  g.initializeVal(id1);
  ValGroupAndItsGraph g1{g.toGroup(id1), &g};
  AbstractTensor v({id0, g1});
  v.swizzle(SwizzleType::XOR, 0, 1);
  // [0', 1']

  EXPECT_EQ(g.disjointValSets().size(), 4);
  EXPECT_EQ(g.disjointExprSets().size(), 1);
  auto result = v.as<ValGroupAndItsGraph>();
  ASSERT_EQ(result.size(), 2);
  EXPECT_EQ(result[0].graph, &g);
  EXPECT_EQ(result[1].graph, &g);

  EXPECT_NE(result[0].group, g.toGroup(id0));
  EXPECT_NE(result[1].group, g.toGroup(id0));
  EXPECT_NE(result[0], g1);
  EXPECT_NE(result[1], g1);
  EXPECT_NE(result[0], result[1]);

  auto defs0 = g.getDefinitions(result[0].group);
  auto defs1 = g.getDefinitions(result[1].group);
  ASSERT_EQ(defs0, defs1);
  ASSERT_EQ(defs0.size(), 1);

  auto eg = defs1.front();
  auto expect_in = std::vector<ValGroup>{g.toGroup(id0), g1.group};
  auto expect_out = std::vector<ValGroup>{result[0].group, result[1].group};
  EXPECT_EQ(g.inputGroups(eg), expect_in);
  EXPECT_EQ(g.outputGroups(eg), expect_out);

  auto uses0 = g.getUses(g.toGroup(id0));
  auto uses1 = g.getUses(g1.group);
  EXPECT_EQ(uses0, uses1);
  EXPECT_EQ(uses0.size(), 1);
  EXPECT_EQ(uses0.front(), eg);

  // Test reusing of existing swizzle
  AbstractTensor vv({id0, g1});
  vv.swizzle(SwizzleType::XOR, 0, 1);
  EXPECT_EQ(g.disjointValSets().size(), 4);
  EXPECT_EQ(g.disjointExprSets().size(), 1);
  EXPECT_EQ(v, vv);
}

TEST_F(AbstractTensorTest, SwizzleValGroupWithIterDomain) {
  auto id0 = newID();
  auto id1 = newID();
  ValGraph g;
  g.initializeVal(id0);
  g.initializeVal(id1);
  ValGroupAndItsGraph g0{g.toGroup(id0), &g};
  AbstractTensor v({g0, id1});
  v.swizzle(SwizzleType::XOR, 0, 1);
  // [0', 1']
  EXPECT_EQ(g.disjointValSets().size(), 4);
  EXPECT_EQ(g.disjointExprSets().size(), 1);
  auto result = v.as<ValGroupAndItsGraph>();
  ASSERT_EQ(result.size(), 2);
  EXPECT_EQ(result[0].graph, &g);
  EXPECT_EQ(result[1].graph, &g);

  EXPECT_NE(result[0], g0);
  EXPECT_NE(result[1], g0);
  EXPECT_NE(result[0].group, g.toGroup(id1));
  EXPECT_NE(result[1].group, g.toGroup(id1));
  EXPECT_NE(result[0], result[1]);

  auto defs0 = g.getDefinitions(result[0].group);
  auto defs1 = g.getDefinitions(result[1].group);
  ASSERT_EQ(defs0, defs1);
  ASSERT_EQ(defs0.size(), 1);

  auto eg = defs1.front();
  auto expect_in = std::vector<ValGroup>{g0.group, g.toGroup(id1)};
  auto expect_out = std::vector<ValGroup>{result[0].group, result[1].group};
  EXPECT_EQ(g.inputGroups(eg), expect_in);
  EXPECT_EQ(g.outputGroups(eg), expect_out);

  auto uses0 = g.getUses(g0.group);
  auto uses1 = g.getUses(g.toGroup(id1));
  EXPECT_EQ(uses0, uses1);
  EXPECT_EQ(uses0.size(), 1);
  EXPECT_EQ(uses0.front(), eg);

  // Test reusing of existing swizzle
  AbstractTensor vv({g0, id1});
  vv.swizzle(SwizzleType::XOR, 0, 1);
  EXPECT_EQ(g.disjointValSets().size(), 4);
  EXPECT_EQ(g.disjointExprSets().size(), 1);
  EXPECT_EQ(v, vv);
}

TEST_F(AbstractTensorTest, Unzip) {
  auto id0 = newID();
  auto id1 = newID();
  auto id2 = newID();
  auto id3 = newID();
  const AbstractTensor v({{id0, id1}, {id2, id3}});
  auto uz = v.unzip();
  ASSERT_EQ(uz.size(), 2);
  AbstractTensor expect0{id0, id2};
  AbstractTensor expect1{id1, id3};
  EXPECT_EQ(uz[0], expect0);
  EXPECT_EQ(uz[1], expect1);
}

TEST_F(AbstractTensorTest, UnzipBroadcasting) {
  auto id0 = newID();
  auto id1 = newID();
  auto id2 = newID();
  const AbstractTensor v({id0, {id1, id2}});
  auto uz = v.unzip();
  ASSERT_EQ(uz.size(), 2);
  AbstractTensor expect0{id0, id1};
  AbstractTensor expect1{id0, id2};
  EXPECT_EQ(uz[0], expect0);
  EXPECT_EQ(uz[1], expect1);
}

TEST_F(AbstractTensorTest, PlaceHolder) {
  AbstractTensor v({{}, {}});
  EXPECT_EQ(v.size(), 2);
  for (auto i : v) {
    EXPECT_FALSE(i.hasValue());
  }

  v.split(0, 2);
  EXPECT_EQ(v.size(), 3);
  for (auto i : v) {
    EXPECT_FALSE(i.hasValue());
  }

  v.merge(0);
  EXPECT_EQ(v.size(), 2);
  for (auto i : v) {
    EXPECT_FALSE(i.hasValue());
  }

  v.swizzle(SwizzleType::XOR, 0, 1);
  EXPECT_EQ(v.size(), 2);
  for (auto i : v) {
    EXPECT_FALSE(i.hasValue());
  }

  v.strip();
  EXPECT_TRUE(v.empty());
}

TEST_F(AbstractTensorTest, Parallelize) {
  auto id0 = newID();
  auto id1 = newID();
  auto id2 = newID();
  AbstractTensor v({id0, {id1, id2}});
  v.parallelize(0, ParallelType::TIDx);
  EXPECT_EQ(id0->getParallelType(), ParallelType::TIDx);
  v.parallelize(1, ParallelType::TIDy);
  EXPECT_EQ(id1->getParallelType(), ParallelType::TIDy);
  EXPECT_EQ(id2->getParallelType(), ParallelType::TIDy);

  ValGraph g;
  g.initializeVal(id0);
  g.initializeVal(id1);
  g.mapVals(id0, id1);
  ValGroupAndItsGraph g0{g.toGroup(id0), &g};
  AbstractTensor vv({g0});
  vv.parallelize(0, ParallelType::BIDx);
  EXPECT_EQ(id0->getParallelType(), ParallelType::BIDx);
  EXPECT_EQ(id1->getParallelType(), ParallelType::BIDx);
  EXPECT_EQ(id2->getParallelType(), ParallelType::TIDy);
}

TEST_F(AbstractTensorTest, Zip) {
  auto id0 = newID();
  auto id1 = newID();
  auto id2 = newID();
  auto id3 = newID();
  const AbstractTensor v0{id0, id2};
  const AbstractTensor v1{id1, id3};
  const AbstractTensor v = AbstractTensor::zip({v0, v1});
  ASSERT_EQ(v.size(), 2);
  AbstractId expect0{id0, id1};
  AbstractId expect1{id2, id3};
  EXPECT_EQ(v[0], expect0);
  EXPECT_EQ(v[1], expect1);
}

TEST_F(AbstractTensorTest, AddRow) {
  auto id0 = newID();
  auto id1 = newID();
  auto id2 = newID();
  auto id3 = newID();
  auto id4 = newID();
  auto id5 = newID();
  AbstractTensor v({{id0, id1}, {id2, id3}});
  AbstractTensor v2({id4, id5});
  v.addRow(v2);
  ASSERT_EQ(v.size(), 2);
  AbstractId expect0{id0, id1, id4};
  AbstractId expect1{id2, id3, id5};
  EXPECT_EQ(v[0], expect0);
  EXPECT_EQ(v[1], expect1);
}

TEST_F(AbstractTensorTest, MergeTaggedTensor) {
  enum class TestTag { A, B };

  using EnumTaggedAbstractTensor = TaggedAbstractTensor<TestTag>;

  auto id0 = newID();
  auto id1 = newID();
  auto id2 = newID();

  {
    // Merge of matching tag sets preserves tag set
    EnumTaggedAbstractTensor v0({id0, id1}, {{TestTag::A}, {TestTag::A}});

    ASSERT_EQ(v0.size(), 2);
    EXPECT_EQ(v0.getTags(0), std::unordered_set<TestTag>{TestTag::A});
    EXPECT_EQ(v0.getTags(1), std::unordered_set<TestTag>{TestTag::A});

    EXPECT_TRUE(v0.hasTag(0, TestTag::A));
    EXPECT_EQ(v0.getTag(0), TestTag::A);
    EXPECT_TRUE(v0.hasTag(-1, TestTag::A));
    EXPECT_EQ(v0.getTag(1), TestTag::A);

    v0.merge(0);

    ASSERT_EQ(v0.size(), 1);
    EXPECT_EQ(v0.getTag(0), TestTag::A);
  }

  {
    // Tag sets should be unioned during merge
    EnumTaggedAbstractTensor v1({id0, id1}, {{TestTag::A}, {TestTag::B}});

    ASSERT_EQ(v1.size(), 2);
    EXPECT_EQ(v1.getTags(0), std::unordered_set<TestTag>{TestTag::A});
    EXPECT_EQ(v1.getTags(1), std::unordered_set<TestTag>{TestTag::B});

    EXPECT_TRUE(v1.hasTag(0, TestTag::A));
    EXPECT_EQ(v1.getTag(0), TestTag::A);
    EXPECT_TRUE(v1.hasTag(-1, TestTag::B));
    EXPECT_EQ(v1.getTag(1), TestTag::B);

    v1.merge(0);

    ASSERT_EQ(v1.size(), 1);
    EXPECT_TRUE(v1.hasTag(0, TestTag::A));
    EXPECT_TRUE(v1.hasTag(0, TestTag::B));
  }

  {
    // Merge outer dimensions and preserve inner dimension with different tag
    EnumTaggedAbstractTensor v2(
        {id0, id1, id2}, {{TestTag::A}, {TestTag::A}, {TestTag::B}});

    ASSERT_EQ(v2.size(), 3);
    EXPECT_EQ(v2.getTags(0), std::unordered_set<TestTag>{TestTag::A});
    EXPECT_EQ(v2.getTags(1), std::unordered_set<TestTag>{TestTag::A});
    EXPECT_EQ(v2.getTags(2), std::unordered_set<TestTag>{TestTag::B});

    EXPECT_EQ(v2.getTag(0), TestTag::A);
    EXPECT_EQ(v2.getTag(1), TestTag::A);
    EXPECT_EQ(v2.getTag(2), TestTag::B);

    v2.merge(0);

    ASSERT_EQ(v2.size(), 2);
    EXPECT_EQ(v2.getTag(0), TestTag::A);
    EXPECT_EQ(v2.getTag(1), TestTag::B);
  }
}

TEST_F(AbstractTensorTest, SwizzleTaggedTensor) {
  enum class TestTag { A, B, C };

  using EnumTaggedAbstractTensor = TaggedAbstractTensor<TestTag>;

  auto id0 = newID();
  auto id1 = newID();
  auto id2 = newID();

  {
    // Swizzle outer dimensions and preserve inner dimension with different tag
    EnumTaggedAbstractTensor v1(
        {id0, id1, id2}, {{TestTag::A}, {TestTag::B}, {TestTag::C}});

    ASSERT_EQ(v1.size(), 3);

    EXPECT_EQ(v1.getTag(0), TestTag::A);
    EXPECT_EQ(v1.getTag(1), TestTag::B);
    EXPECT_EQ(v1.getTag(2), TestTag::C);

    // NoSwizzle should not mix tags
    v1.swizzle(SwizzleType::NoSwizzle, 0, 1);

    ASSERT_EQ(v1.size(), 3);
    EXPECT_EQ(v1.getTag(0), TestTag::A);
    EXPECT_EQ(v1.getTag(1), TestTag::B);
    EXPECT_EQ(v1.getTag(2), TestTag::C);

    v1.swizzle(Swizzle2DType::NoSwizzle, 0, 1);

    EXPECT_EQ(v1.getTag(0), TestTag::A);
    EXPECT_EQ(v1.getTag(1), TestTag::B);
    EXPECT_EQ(v1.getTag(2), TestTag::C);

    // An XOR swizzle will mix the tags
    v1.swizzle(SwizzleType::XOR, 1, 0);

    ASSERT_EQ(v1.size(), 3);
    EXPECT_EQ(
        v1.getTags(0), std::unordered_set<TestTag>({TestTag::A, TestTag::B}));
    EXPECT_EQ(
        v1.getTags(1), std::unordered_set<TestTag>({TestTag::A, TestTag::B}));
    EXPECT_EQ(v1.getTags(2), std::unordered_set<TestTag>{TestTag::C});
  }
}

} // namespace nvfuser
