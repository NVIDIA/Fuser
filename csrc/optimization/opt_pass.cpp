// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#include <optimization/opt_pass.h>

#include <mutex>

namespace nvfuser::optimization {

namespace {

thread_local std::unordered_set<OptimizationPassCategory> disabled_pass_flag;

class OptimizationRegistry {
 public:
  struct PassEntry {
    int priority_;
    FusionPass pass_;
    std::string name_;
    PassEntry(int priority, FusionPass pass, std::string name)
        : priority_(priority),
          pass_(std::move(pass)),
          name_(std::move(name)) {}
  };

  void registerPass(
      const OptimizationPassCategory& cat,
      FusionPass func,
      std::string name_,
      int priority) {
    std::lock_guard<std::mutex> guard(mutex_);
    auto& pass_entry_list = pass_categories_[cat];
    auto entry_iter = pass_entry_list.begin();
    while (entry_iter != pass_entry_list.end()) {
      if (entry_iter->priority_ < priority) {
        break;
      }
      entry_iter++;
    }
    pass_entry_list.emplace(
        entry_iter, priority, std::move(func), std::move(name_));
  }

  void apply(const OptimizationPassCategory& cat, Fusion* fusion) {
    std::lock_guard<std::mutex> guard(mutex_);
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
  // TODO: read access mutex_ should/could be optimized, since graph pass is
  // thread-safe.
  std::mutex mutex_;
  std::unordered_map<OptimizationPassCategory, std::list<PassEntry>>
      pass_categories_;
};

} // namespace

OptimizationPassGuard::OptimizationPassGuard(
    const OptimizationPassCategory& category,
    bool enable)
    : cat_(category) {
  prev_status_ = switchOptimizationPass(cat_, enable);
}

OptimizationPassGuard::~OptimizationPassGuard() {
  switchOptimizationPass(cat_, prev_status_);
}

void registerOptimizationPass(
    const OptimizationPassCategory& category,
    OptimizationPass* pass,
    int priority) {
  OptimizationRegistry::getInstance().registerPass(
      category, pass->func(), pass->name(), priority);
}

void applyOptimizationPass(
    const OptimizationPassCategory& category,
    Fusion* fusion) {
  if (disabled_pass_flag.count(category) == 0) {
    OptimizationRegistry::getInstance().apply(category, fusion);
  }
}

bool switchOptimizationPass(
    const OptimizationPassCategory& category,
    std::optional<bool> enable) {
  auto enabled = disabled_pass_flag.count(category) == 0;

  if (enable.has_value()) {
    if (enable.value()) {
      disabled_pass_flag.erase(category);
    } else {
      disabled_pass_flag.insert(category);
    }
  }
  return enabled;
}

} // namespace nvfuser::optimization
