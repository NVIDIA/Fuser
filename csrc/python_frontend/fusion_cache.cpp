// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#include <instrumentation.h>
#include <python_frontend/fusion_cache.h>

namespace nvfuser::python_frontend {

// FusionCache static data member definitions for singleton usage
std::mutex FusionCache::singleton_lock_;
FusionCache* FusionCache::singleton_ = nullptr;

UserSchedule::UserSchedule() : schedule(nullptr), executor(nullptr) {
  schedule = std::make_unique<Fusion>();
  executor = std::make_unique<FusionExecutor>();
}

FusionSchedules::FusionSchedules()
    : auto_gen_schedules(nullptr), user_def_schedules(), scheds_lock() {
  auto_gen_schedules =
      std::make_unique<FusionExecutorCache>(std::make_unique<Fusion>());
}

Fusion* FusionSchedules::preschedFusion() {
  auto fusion = auto_gen_schedules->fusion();
  TORCH_CHECK(fusion != nullptr, "Prescheduled Fusion is unexpectedly null!");
  return fusion;
}

TrieNode::TrieNode(RecordFunctor* rec, TrieNode* _parent, size_t _fusion_id)
    : record(rec),
      children(),
      fusion_id(_fusion_id),
      visits(0),
      parent(_parent),
      trie_node_lock() {}

bool TrieNode::isTerminal() const {
  return (record.get()->recordType() == serde::RecordType_End);
}

FusionCache* FusionCache::get(size_t max_fusions) {
  FUSER_PERF_SCOPE("FusionCache::get");
  std::lock_guard<std::mutex> guard(singleton_lock_);
  if (singleton_ == nullptr) {
    singleton_ = new FusionCache(max_fusions);
  }
  TORCH_CHECK(
      max_fusions >= singleton_->fusions_.size(),
      "The max fusions is set less than the number of fusions in the cache.");
  singleton_->max_fusions_ = max_fusions;
  return singleton_;
}

size_t FusionCache::numFusions() const {
  return fusions_.size();
}

void FusionCache::print(std::ostream& os) const {
  os << "Fusions by id:" << std::endl;
  std::vector<TrieNode*> stack;
  stack.push_back(root_.get());

  while (!stack.empty()) {
    TrieNode* node = stack.back();
    stack.pop_back();

    if (node->isTerminal()) {
      std::vector<TrieNode*> rev_fusion_records;
      TrieNode* end = node->parent;
      while (end) {
        if (end->record->recordType() != serde::RecordType_Start) {
          rev_fusion_records.emplace_back(end);
        }
        end = end->parent;
      }

      os << node->fusion_id << ":" << std::endl;
      std::for_each(
          rev_fusion_records.rbegin(),
          rev_fusion_records.rend(),
          [&os](const auto elem) {
            os << "    ";
            elem->record->print(os);
            os << std::endl;
          });
    } else {
      for (auto& iter : node->children) {
        stack.push_back(iter.second.get());
      }
    }
  }
}

void FusionCache::stats(std::ostream& os) const {
  os << "Total Fusions: " << fusions_.size() << "\n";

  // Does not make sense to print stats if the cache is disabled.
  if (fusions_.size() > 0) {
    os << "Cache Hits by Fusion Id:\n";
    auto total_cache_hits = 0;
    for (size_t i = 0; i < terminal_nodes_.size(); ++i) {
      // The first visit is a miss!
      auto visits = terminal_nodes_[i]->visits - 1;
      total_cache_hits += visits;
      os << "\t" << i << " -> " << visits << " hits\n";
    }

    auto hit_rate = static_cast<float>(total_cache_hits) /
        static_cast<float>(root_->visits) * 100.0;
    os << "Cache Lookups: " << root_->visits;
    os << " Cache Hits: " << total_cache_hits;
    os << " Hit Rate: " << hit_rate << "%\n";
  }
}

void FusionCache::reset() {
  std::lock_guard<std::mutex> guard(singleton_lock_);
  if (singleton_ != nullptr) {
    auto max_fusions = singleton_->max_fusions_;
    delete singleton_;
    singleton_ = new FusionCache(max_fusions);
  }
}

FusionCache::FusionCache(size_t max_fusions)
    : max_fusions_(max_fusions),
      root_(nullptr),
      fusions_(),
      terminal_nodes_(),
      user_def_input_encodings_() {
  RecordFunctor* start = new StartRecord();
  root_ = std::make_unique<TrieNode>(start);
}

// In order to keep queries fast, this method does not lock.
// In the worst case, the query should fail and if you try to create a child,
// it should give you back an already created child if two threads are walking
// the trie at the same time with the same definition.
c10::optional<TrieNode*> FusionCache::queryChildren(
    TrieNode* node,
    RecordFunctor* rec) const {
  TORCH_CHECK(
      !node->isTerminal(), "There should be no children from a Terminal Node!");
  TORCH_CHECK(rec, "Record is null!");
  auto trie_node = node->children.find(rec);
  if (trie_node == std::end(node->children)) {
    return c10::nullopt;
  } else {
    ++(trie_node->second.get()->visits);
    return c10::optional<TrieNode*>(trie_node->second.get());
  }
}
FusionSchedules* FusionCache::queryFusionSchedules(size_t fusion_id) const {
  TORCH_CHECK(
      fusion_id < fusions_.size(),
      "Invalid scheduler query for id:",
      fusion_id);
  FusionSchedules* ptr = fusions_.at(fusion_id).get();
  TORCH_CHECK(ptr != nullptr, "Unexpected null FusionSchedules object.");
  return ptr;
}
c10::optional<size_t> FusionCache::queryUserScheduleId(
    const FusionSchedules* scheds,
    const at::ArrayRef<c10::IValue>& inputs,
    int device) {
  c10::optional<size_t> result = c10::nullopt;

  auto& user_scheds = scheds->user_def_schedules;
  if (user_scheds.size() != 0) {
    auto input_id = user_def_input_encodings_.lookupId(inputs);
    auto user_sched = user_scheds.find(input_id.id);
    if (user_sched != user_scheds.end()) {
      return c10::optional<size_t>(user_sched->first);
    }
  }
  return result;
}
const UserSchedule& FusionCache::queryUserSchedule(
    const FusionSchedules* scheds,
    size_t id,
    int device) const {
  auto& user_scheds = scheds->user_def_schedules;
  TORCH_CHECK(
      user_scheds.size() > 0,
      "Expecting there to be at least one user schedule!");
  auto user_sched = user_scheds.find(id);
  TORCH_CHECK(
      user_sched != user_scheds.end(), "Lookup of non-existent user schedule!");
  return user_sched->second.at(device);
}

TrieNode* FusionCache::createChild(TrieNode* node, RecordFunctor* rec) {
  FUSER_PERF_SCOPE("FusionCache::createChild");
  TrieNode* child = nullptr;
  TORCH_CHECK(
      !node->isTerminal(), "Cannot create a trie node from a terminal node!");
  TORCH_CHECK(rec, "Record is null!");

  std::lock_guard<std::mutex> guard(node->trie_node_lock);

  // As a thread-safety compromise for fast queries, the node is re-queried
  // prior to child creation incase another thread slipped in the node.
  auto child_node = queryChildren(node, rec);
  if (child_node.has_value()) {
    child = child_node.value();
  } else {
    size_t fusion_id = 0;
    if (rec->recordType() == serde::RecordType_End) {
      TORCH_CHECK(
          (fusions_.size() + 1) <= max_fusions_,
          "The number of fusions in nvfuser has exceeded ",
          max_fusions_,
          "fusions.  The max_fusions for the FusionCache might need to be ",
          "increased if the max number is not being exceeded due to an error.");
      fusions_.emplace_back(std::make_unique<FusionSchedules>());
      fusion_id = fusions_.size() - 1;
    }

    // Copying the record owned by the FusionDefinition that calls this function
    // so the trie owns a copy when the FusionDefinition gets destroyed rather
    // than managing a shared pointer that would only share with
    // FusionDefinition that creates a trie node but not cache lookups
    RecordFunctor* new_rec = rec->clone();
    node->children[new_rec] =
        std::make_unique<TrieNode>(new_rec, node, fusion_id);
    child = node->children[new_rec].get();
    ++(child->visits);
    TORCH_CHECK(
        child != nullptr, "Created child of TrieNode should not be null!");
    if (rec->recordType() == serde::RecordType_End) {
      terminal_nodes_.push_back(node->children[new_rec].get());
    }
    if (isDebugDumpEnabled(DebugDumpOption::PythonFrontendDebug)) {
      std::stringstream ss;
      new_rec->print(ss);
      std::cout << "\nFusionDefinition: Create new trie node for: " << ss.str()
                << "\n";
    }
  }
  return child;
}

UserSchedule* FusionCache::createUserSchedule(
    FusionSchedules* scheds,
    const at::ArrayRef<c10::IValue>& inputs,
    int device) {
  FUSER_PERF_SCOPE("FusionCache::createUserSchedule");
  std::lock_guard<std::mutex> guard(scheds->scheds_lock);
  auto& user_scheds = scheds->user_def_schedules;
  auto input_id = user_def_input_encodings_.lookupId(inputs);
  auto user_sched = user_scheds.find(input_id.id);
  if (user_sched == user_scheds.end()) {
    user_scheds[input_id.id] = std::vector<UserSchedule>(device + 1);
  } else {
    if (static_cast<size_t>(device) >= user_scheds[input_id.id].size()) {
      user_scheds[input_id.id].resize(device + 1);
    }
  }
  return &user_scheds[input_id.id].at(device);
}

TrieNode* FusionCache::rootTriePtr() {
  ++(root_.get()->visits);
  return root_.get();
}

} // namespace nvfuser::python_frontend
