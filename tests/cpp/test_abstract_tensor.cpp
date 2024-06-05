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

#include <abstract_tensor.h>

namespace nvfuser {

class AbstractTensorTest : public NVFuserTest {
  std::unique_ptr<Fusion> fusion_ptr_;
  std::unique_ptr<FusionGuard> fusion_guard_ptr_;
  std::unique_ptr<IterDomainBuilder> builder_;

  void SetUp() override {
    NVFuserTest::SetUp();
    fusion_ptr_ = std::make_unique<Fusion>();
    fusion_guard_ptr_ = std::make_unique<FusionGuard>(fusion_ptr_.get());
    auto size = IrBuilder::create<Val>(DataType::Index);
    builder_ =
        std::make_unique<IterDomainBuilder>(fusion_ptr_->zeroVal(), size);
  }

 protected:
  IterDomain* newID() const {
    return builder_->build();
  }
};

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
  auto g01 = result[0].group;
  EXPECT_NE(g01, g.toGroup(id0));
  EXPECT_NE(g01, g.toGroup(id1));
  auto defs = g.getDefinitions(g01);
  ASSERT_EQ(defs.size(), 1);
  auto eg01 = defs.front();
  auto expect01 = std::vector<ValGroup>{g.toGroup(id0), g.toGroup(id1)};
  EXPECT_EQ(g.inputGroups(eg01), expect01);
  EXPECT_EQ(g.outputGroups(eg01), std::vector<ValGroup>{g01});
  auto uses0 = g.getUses(g.toGroup(id0));
  auto uses1 = g.getUses(g.toGroup(id1));
  EXPECT_EQ(uses0.size(), 1);
  EXPECT_EQ(uses1.size(), 1);
  EXPECT_EQ(uses0.vector(), uses1.vector());
  EXPECT_EQ(uses0.front(), eg01);

  // Test reusing of existing merge
  ValGroupAndItsGraph g0_{g.toGroup(id0), &g};
  ValGroupAndItsGraph g1_{g.toGroup(id1), &g};
  AbstractTensor vv({g0_, g1_});
  vv.merge(0);
  EXPECT_EQ(g.disjointValSets().size(), 3);
  EXPECT_EQ(g.disjointExprSets().size(), 1);
  auto result2 = v.as<ValGroupAndItsGraph>();
  ASSERT_EQ(result2.size(), 1);
  EXPECT_EQ(result2[0].graph, &g);
  EXPECT_EQ(result2[0].group, g01);
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
  auto g01 = result[0].group;
  EXPECT_NE(g01, g.toGroup(id0));
  EXPECT_NE(g01, g.toGroup(id1));
  auto defs = g.getDefinitions(g01);
  ASSERT_EQ(defs.size(), 1);
  auto eg01 = defs.front();
  auto expect01 = std::vector<ValGroup>{g.toGroup(id0), g.toGroup(id1)};
  EXPECT_EQ(g.inputGroups(eg01), expect01);
  EXPECT_EQ(g.outputGroups(eg01), std::vector<ValGroup>{g01});
  auto uses0 = g.getUses(g.toGroup(id0));
  auto uses1 = g.getUses(g.toGroup(id1));
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
  auto g01 = result[0].group;
  EXPECT_NE(g01, g.toGroup(id0));
  EXPECT_NE(g01, g.toGroup(id1));
  auto defs = g.getDefinitions(g01);
  ASSERT_EQ(defs.size(), 1);
  auto eg01 = defs.front();
  auto expect01 = std::vector<ValGroup>{g.toGroup(id0), g.toGroup(id1)};
  EXPECT_EQ(g.inputGroups(eg01), expect01);
  EXPECT_EQ(g.outputGroups(eg01), std::vector<ValGroup>{g01});
  auto uses0 = g.getUses(g.toGroup(id0));
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
  ValGroupAndItsGraph g2{g.toGroup(id2), &g};
  AbstractTensor v1({g0, ValGroupAndItsGraph{g.toGroup(id1), &g}, g2});
  v1.split(1, 5);
  // [0, 1/5, 5, 2]
  AbstractTensor v2({g0, ValGroupAndItsGraph{g.toGroup(id1), &g}, g2});
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
    EXPECT_EQ(result[0].graph, &g);
    EXPECT_EQ(result[1].graph, &g);
    EXPECT_EQ(result[2].graph, &g);
    EXPECT_EQ(result[3].graph, &g);
    EXPECT_EQ(result[0].group, g.toGroup(id0));
    EXPECT_EQ(result[3].group, g.toGroup(id2));
    auto g156 = result[1].group;
    auto g56 = result[2].group;
    EXPECT_NE(g156, g.toGroup(id0));
    EXPECT_NE(g156, g.toGroup(id1));
    EXPECT_NE(g156, g.toGroup(id2));
    EXPECT_NE(g56, g.toGroup(id0));
    EXPECT_NE(g56, g.toGroup(id1));
    EXPECT_NE(g56, g.toGroup(id2));
    EXPECT_NE(g56, g156);
    auto defs = g.getDefinitions(g156);
    ASSERT_EQ(defs.size(), 1);
    auto eg56 = defs.front();
    auto expect56 = std::vector<ValGroup>{g156, g56};
    EXPECT_EQ(g.outputGroups(eg56), expect56);
    EXPECT_EQ(g.inputGroups(eg56), std::vector<ValGroup>{g.toGroup(id1)});
    auto uses = g.getUses(g.toGroup(id1));
    EXPECT_EQ(uses.size(), 2);
    EXPECT_TRUE(uses.has(eg56));
  }

  // Test reusing of existing split
  ValGroupAndItsGraph g1{g.toGroup(id1), &g};

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

} // namespace nvfuser
