// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#include <optimization/opt_pass.h>

namespace nvfuser::optimization {

namespace {

class OptimizationRegistry {
 public:
  struct PassEntry {
    int priority_;
    FusionPass pass_;
    std::string name_;
  };

  void register(const OptimizationPassCategory& cat, FusionPass func, std::string name_, int priority) {

    std::guard<mutex> guard(mutex_);
    auto& pass_entry_list = pass_categories_[cat];
    entry_iter = pass_entry_list.begin();
    while (entry_iter != pass_entry_list.end()) {
      if (entry_iter->priority_ < priority) {
        break;
      }
    }
    pass_entry_list.emplace(entry_iter, priority, std::move(func), std::move(name_));
  }

  void apply(const OptimizationPassCategory& cat, Fusion* fusion) {
    std::guard<mutex> guard(mutex_);
    const auto& pass_entry_list = pass_categories_[cat];
    for (const auto& entry : pass_entry_list) {
      entry.pass_(fusion);
    }
  }

  static OptimizationRegistry& getInstance() {
    static OptimizationRegistry registry;
    return registry;
  }

 protected:
  // TODO: read access mutex_ should/could be optimized, since graph pass is thread-safe.
  std::mutex mutex_;
  std::unordered_map<OptimizationPassCategory, std::list<PassEntry>> pass_categories_;
};

} // namespace

void registerOptimizationPass(const OptimizationPassCategory& category, OptimizationPass pass, int priority) {
  OptimizationRegistry::getInstance().register(category, pass.func(), pass.name(), priority);
}

void applyOptimizationPass(const OptimizationPassCategory& category, Fusion* fusion) {
  OptimizationRegistry::getInstance().apply(category, fusion);
}

} // namespace nvfuser::optimization
