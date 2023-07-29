// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#include <gmock/gmock-matchers.h>
#include <gtest/gtest.h>

#include <torch/torch.h>

#include <python_frontend/fusion_cache.h>
#include <test/utils.h>
#include <test/validator.h>

namespace nvfuser {
using namespace nvfuser::python_frontend;

// RUN CMD: bin/test_jit --gtest_filter="NVFuserTest*PyFusionCache*"
TEST_F(NVFuserTest, PyFusionCache_CUDA) {
  // Reset cache before testing.
  try {
    FusionCache::reset();
    SUCCEED();
  } catch (const std::exception& e) {
    FAIL() << "Did not properly reset cache!" << e.what();
  }

  // Create a fusion manager with a maximum of 1 Fusion
  FusionCache* fc = FusionCache::get(1);
  // You should never get a nullptr
  ASSERT_FALSE(fc == nullptr);
  ASSERT_TRUE(fc->numFusions() == 0);

  // Check that cache methods all assert when presented with a null record.
  {
    std::unique_ptr<RecordFunctor> null_record(nullptr);
    TrieNode* node = fc->rootTriePtr();

    try {
      fc->queryChildren(node, null_record.get());
      FAIL() << "Should trigger an assert when the record is looked up!";
    } catch (...) {
      SUCCEED();
    }

    try {
      fc->createChild(node, null_record.get());
      FAIL() << "Should trigger an assert when the record is looked up!";
    } catch (...) {
      SUCCEED();
    }
  }

  // Check that cache methods act appropriately when presenting a new
  // record to an empty cache.
  {
    std::unique_ptr<RecordFunctor> test_record(new TensorRecord(
        {State(0, serde::StateType_Tensor)}, {3}, {true}, DataType::Float));
    TrieNode* root = fc->rootTriePtr();
    TrieNode* node = nullptr;

    // Check Methods prior to adding an entry to the cache

    // Cache Lookup should not succeed becase no records are in the cache
    try {
      auto undefined_node = fc->queryChildren(root, test_record.get());
      ASSERT_TRUE(undefined_node == std::nullopt);
      SUCCEED();
    } catch (const std::exception& e) {
      FAIL() << "Unexpected assert during cache lookup!" << e.what();
    }

    // Add a cache entry and check methods

    try {
      fc->createChild(root, test_record.get());
      SUCCEED();
    } catch (const std::exception& e) {
      FAIL() << "An unexpected assert on Cache Entry creation!" << e.what();
    }

    try {
      auto child_node = fc->queryChildren(root, test_record.get());
      ASSERT_FALSE(child_node == std::nullopt);
      node = child_node.value();
      SUCCEED();
    } catch (const std::exception& e) {
      FAIL() << "An unexpected assert on cache lookup!" << e.what();
    }

    // Add a terminal cache entry and check methods

    std::unique_ptr<RecordFunctor> end_record(new EndRecord());
    try {
      node = fc->createChild(node, end_record.get());
      SUCCEED();
    } catch (const std::exception& e) {
      FAIL() << "An unexpected assert on Terminal Cache Entry creation!"
             << e.what();
    }

    try {
      fc->queryChildren(node, test_record.get());
      FAIL() << "Expected an assert from a terminal entry!";
    } catch (...) {
      SUCCEED();
    }
  }

  // Check that cache methods act appropriately when presenting a new
  // record to a cache with 1 fusion.
  {
    std::unique_ptr<RecordFunctor> cached_record(new TensorRecord(
        {State(0, serde::StateType_Tensor)}, {3}, {true}, DataType::Float));
    std::unique_ptr<RecordFunctor> new_record(new ScalarRecord(
        {State(1, serde::StateType_Scalar)},
        std::monostate{},
        DataType::Float));
    TrieNode* root = fc->rootTriePtr();
    TrieNode* node = nullptr;

    try {
      auto child_node = fc->queryChildren(root, cached_record.get());
      ASSERT_FALSE(child_node == std::nullopt);
      node = child_node.value();
      SUCCEED();
    } catch (const std::exception& e) {
      FAIL() << "Cache lookup unexpectedly asserted!" << e.what();
    }

    try {
      auto undefined_node = fc->queryChildren(node, new_record.get());
      ASSERT_TRUE(undefined_node == std::nullopt);
      SUCCEED();
    } catch (const std::exception& e) {
      FAIL() << "Cache lookup unexpectedly asserted!" << e.what();
    }

    try {
      node = fc->createChild(node, new_record.get());
      SUCCEED();
    } catch (const std::exception& e) {
      FAIL() << "An unexpected assert on Cache Entry creation!" << e.what();
    }

    std::unique_ptr<RecordFunctor> end_record(new EndRecord());
    try {
      fc->createChild(node, end_record.get());
      FAIL() << "Expected the cache to assert because it is full!";
    } catch (...) {
      SUCCEED();
    }
  }

  // Verify proper cache lookup up of complete fusion already cached.
  // This tends to flush out pointer problems in the cache.
  {
    std::unique_ptr<RecordFunctor> test_record(new TensorRecord(
        {State(0, serde::StateType_Tensor)}, {3}, {true}, DataType::Float));
    std::unique_ptr<RecordFunctor> dummy_record(new TensorRecord(
        {State(0, serde::StateType_Tensor)}, {3}, {true}, DataType::Float));
    TrieNode* root = fc->rootTriePtr();
    TrieNode* node = nullptr;

    try {
      auto child_node = fc->queryChildren(root, test_record.get());
      ASSERT_FALSE(child_node == std::nullopt);
      node = child_node.value();
      SUCCEED();
    } catch (const std::exception& e) {
      FAIL() << "An unexpected assert on cache lookup!" << e.what();
    }

    std::unique_ptr<RecordFunctor> end_record(new EndRecord());
    try {
      fc->queryChildren(node, end_record.get());
      SUCCEED();
    } catch (const std::exception& e) {
      FAIL() << "An unexpected assert on cache lookup!" << e.what();
    }
  }
}

} // namespace nvfuser
