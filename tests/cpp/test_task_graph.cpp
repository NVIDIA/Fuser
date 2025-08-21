/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION &
 * AFFILIATES. All rights reserved. SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#include <macros.h>

#include <csrc/exceptions.h>
#include <gtest/gtest.h>

#include <graph/task_graph.h>
#include <tests/cpp/utils.h>
#include <tests/cpp/validator.h>

namespace nvfuser {

using Tasks = std::vector<TaskGraph::Task>;
using TaskGraphTest = NVFuserTest;

struct SimpleAlias {
  TaskGraph::DataId output;
  TaskGraph::DataId input;
};

std::vector<TaskGraph::Data> inferData(const Tasks& tasks) {
  // Find number of data items so we can resize
  TaskGraph::DataId max_data_id = 0;
  for (const TaskGraph::Task& task : tasks) {
    for (TaskGraph::DataId input_id : task.inputs) {
      max_data_id = std::max(max_data_id, input_id);
    }
    for (TaskGraph::DataId output_id : task.outputs) {
      max_data_id = std::max(max_data_id, output_id);
    }
  }
  std::vector<TaskGraph::Data> all_data((size_t)max_data_id + 1);

  for (const auto& [task_id, task] : enumerate(tasks)) {
    for (TaskGraph::DataId input_id : task.inputs) {
      all_data.at(input_id).uses.push_back(task_id);
    }
    for (TaskGraph::DataId output_id : task.outputs) {
      all_data.at(output_id).definition = task_id;
    }
  }

  // Detect inputs and outputs and ensure they are not freed
  for (TaskGraph::Data& data : all_data) {
    data.size = 1;
    data.can_free = data.definition.has_value() && !data.uses.empty();
  }

  return all_data;
}

std::vector<TaskGraph::TaskId> getTasks(const TaskGraph::SortResult& result) {
  const std::vector<TaskGraph::Step>& steps = result.steps;
  std::vector<TaskGraph::TaskId> tasks;
  tasks.reserve(steps.size());
  for (const TaskGraph::Step& step : steps) {
    tasks.push_back(step.task);
  }
  return tasks;
}

TEST_F(TaskGraphTest, Basic) {
  //   0   1
  //   |\ /
  //   | 2
  //   |/
  //   3
  Tasks tasks{{{0, 1}, {2}}, {{0, 2}, {3}}};
  std::vector<TaskGraph::Data> data = inferData(tasks);
  auto graph = TaskGraph(tasks, data);

  const TaskGraph::SortResult result = graph.findOptimalOrder();

  ASSERT_EQ(result.steps.size(), tasks.size());
  std::vector<TaskGraph::TaskId> expected{0, 1};
  EXPECT_EQ(getTasks(result), expected);
  EXPECT_EQ(result.steps.back().high_water_mark, 4);
}

// This example includes two segments, each of which aliases the other
TEST_F(TaskGraphTest, ImpossibleAlias) {
  //   0   1
  //   |\ /|
  //   | X |
  //   |/ \|
  //   2   3
  // Two tasks, each takes the same two inputs
  Tasks tasks{{{0, 1}, {2}}, {{0, 1}, {3}}};
  std::vector<TaskGraph::Data> data = inferData(tasks);
  // Each of the segment outputs aliases a different input
  data[2].aliases_input = 0;
  data[3].aliases_input = 1;
  // This graph can't be ordered without breaking the aliasing constraint
  auto graph = TaskGraph(tasks, data);

  EXPECT_THAT(
      [&graph]() { graph.findOptimalOrder(); },
      ::testing::ThrowsMessage<nvfuser::nvfError>(::testing::HasSubstr(
          "Ran out of ready tasks before completing ordering")));
}

TEST_F(TaskGraphTest, SelfEdge) {
  Tasks tasks{{{0}, {0}}};
  std::vector<TaskGraph::Data> data = inferData(tasks);
  // This graph can't be ordered because it contains an edge from a Data node
  // back to itself. A task can't be both producer and consumer to a Data.
  auto graph = TaskGraph(tasks, data);

  EXPECT_THAT(
      [&graph]() { graph.findOptimalOrder(); },
      ::testing::ThrowsMessage<nvfuser::nvfError>(::testing::HasSubstr(
          "Ran out of ready tasks before completing ordering")));
}

TEST_F(TaskGraphTest, TwoCycle) {
  Tasks tasks{{{0}, {1}}, {{1}, {0}}};
  std::vector<TaskGraph::Data> data = inferData(tasks);
  // This graph can't be ordered because it contains a cycle
  auto graph = TaskGraph(tasks, data);

  EXPECT_THAT(
      [&graph]() { graph.findOptimalOrder(); },
      ::testing::ThrowsMessage<nvfuser::nvfError>(::testing::HasSubstr(
          "Ran out of ready tasks before completing ordering")));
}

TEST_F(TaskGraphTest, ThreeCycle) {
  Tasks tasks{{{0}, {1}}, {{1}, {2}}, {{2}, {0}}};
  std::vector<TaskGraph::Data> data = inferData(tasks);
  // This graph can't be ordered because it contains a cycle
  auto graph = TaskGraph(tasks, data);

  EXPECT_THAT(
      [&graph]() { graph.findOptimalOrder(); },
      ::testing::ThrowsMessage<nvfuser::nvfError>(::testing::HasSubstr(
          "Ran out of ready tasks before completing ordering")));
}

TEST_F(TaskGraphTest, FreeableIntermediate) {
  //   0
  //  /|\
  // 1 2 3
  //     |
  //     4
  Tasks tasks{
      {{0}, {1}}, // Task 0
      {{0}, {2}}, // Task 1
      {{0}, {3}}, // Task 2
      {{3}, {4}}, // Task 3
  };
  std::vector<TaskGraph::Data> data = inferData(tasks);
  auto graph = TaskGraph(tasks, data);

  const TaskGraph::SortResult result = graph.findOptimalOrder();

  // Expect that we evaluate the branch with intermediate before the others,
  // since that intermediate 3 can take the space we'll need later for output 1
  // or 2
  ASSERT_EQ(result.steps.size(), tasks.size());
  EXPECT_NE(getTasks(result).back(), 3);
  EXPECT_EQ(result.steps.back().high_water_mark, 4);
}

TEST_F(TaskGraphTest, DifferentSizes) {
  //   0
  //  / \
  // 1   4
  // |   |
  // 2   5
  // |   |
  // 3   6
  //  \ /
  //   7
  Tasks tasks{
      {{0}, {1}}, // Task 0
      {{1}, {2}}, // Task 1
      {{2}, {3}}, // Task 2
      {{0}, {4}}, // Task 3
      {{4}, {5}}, // Task 4
      {{5}, {6}}, // Task 5
      {{3, 6}, {7}} // Task 6
  };
  std::vector<TaskGraph::Data> data = inferData(tasks);
  data[0].size = 1;

  data[1].size = 12;
  data[2].size = 10;
  data[3].size = 10;

  // Note that 4 and 5  are large but that 6 is smaller than the others, so we
  // should compute up to here then start on the 0-1-2-3 branch after freeing 4
  // and 5. Otherwise we would need to hold
  data[4].size = 11;
  data[5].size = 11;
  data[6].size = 7;

  data[7].size = 1;
  auto graph = TaskGraph(tasks, data);

  const TaskGraph::SortResult result = graph.findOptimalOrder();

  ASSERT_EQ(result.steps.size(), tasks.size());
  std::vector<TaskGraph::TaskId> expected{3, 4, 5, 0, 1, 2, 6};
  EXPECT_EQ(getTasks(result), expected);
  EXPECT_EQ(result.steps.back().high_water_mark, 30);
}

// This is the example from Figure 1 of Kayaaslan et al. 2018
// It includes temporary space needed for each task.
// This is a candidate for the Liu algorithm instead of brute force search.
// https://doi.org/10.1016/j.tcs.2017.09.037
TEST_F(TaskGraphTest, InTree) {
  // 0 3
  // | |
  // 1 4 7
  // | | |
  // 2 5 8
  //  \| |
  //   6 9
  //    \|
  //    10
  // The starred nodes are smaller than the others
  Tasks tasks{
      {{0}, {1}}, // Task 0
      {{1}, {2}}, // Task 1
      {{3}, {4}}, // Task 2
      {{4}, {5}}, // Task 3
      {{2, 5}, {6}}, // Task 4
      {{7}, {8}}, // Task 5
      {{8}, {9}}, // Task 6
      {{6, 9}, {10}}, // Task 7
  };
  std::vector<TaskGraph::Data> data = inferData(tasks);
  data[0].size = 1; // input
  data[1].size = 4;
  data[2].size = 1;
  data[3].size = 1; // input
  data[4].size = 2;
  data[5].size = 2;
  data[6].size = 2;
  data[7].size = 1; // input
  data[8].size = 1;
  data[9].size = 5;
  data[10].size = 1;
  tasks[0].temp_space = 4; // A
  tasks[1].temp_space = 3; // B
  tasks[2].temp_space = 1; // C
  tasks[3].temp_space = 2; // D
  tasks[4].temp_space = 2; // E
  tasks[5].temp_space = 8; // F
  tasks[6].temp_space = 2; // G
  tasks[7].temp_space = 1; // H

  auto graph = TaskGraph(tasks, data);

  const TaskGraph::SortResult result = graph.findOptimalOrder();

  ASSERT_EQ(result.steps.size(), tasks.size());
  // By Kayaaslan et al. 2018, Sn 3.1,
  // one optimal order is F A B C D E G H which has cost 34
  // There are others with the same cost such as F C D A B E G H
  EXPECT_EQ(result.steps.back().high_water_mark, 34);
}

} // namespace nvfuser
