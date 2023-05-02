// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#include <instrumentation.h>
#include <python_frontend/fusion_cache.h>
#include <serde/fusion_record_serde.h>

#include <filesystem>
namespace fs = std::filesystem;

namespace nvfuser::python_frontend {

// FusionCache static data member definitions for singleton usage
std::mutex FusionCache::singleton_lock_;
FusionCache* FusionCache::singleton_ = nullptr;

UserSchedule::UserSchedule() : schedule(nullptr), executor(nullptr) {
  schedule = std::make_unique<Fusion>();
  executor = std::make_unique<FusionExecutor>();
}

FusionSchedules::FusionSchedules()
    : auto_gen_schedules(nullptr),
      user_def_schedules(),
      last_user_def_scheduled_ir(nullptr),
      last_user_def_executor(nullptr),
      scheds_lock() {
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

flatbuffers::Offset<serde::TrieNode> TrieNode::serialize(
    flatbuffers::FlatBufferBuilder& builder,
    const std::map<RecordFunctor*, size_t>&
        map_record_functor_to_trie_node_id) {
  // Map children TrieNode to its corresponding Integer index
  std::vector<size_t> children_trie_node_ids;
  children_trie_node_ids.reserve(children.size());
  for (auto&& c : children) {
    size_t id = map_record_functor_to_trie_node_id.at(c.first);
    children_trie_node_ids.push_back(id);
  }

  return serde::CreateTrieNodeDirect(
      builder,
      record->serialize(builder),
      &children_trie_node_ids,
      fusion_id,
      visits,
      isTerminal());
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
  if (!fusions_.empty()) {
    os << "Cache Hits by Fusion Id:\n";
    size_t total_cache_hits = 0;
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
    const at::ArrayRef<c10::IValue>& inputs) {
  c10::optional<size_t> result = c10::nullopt;

  auto& user_scheds = scheds->user_def_schedules;
  if (!user_scheds.empty()) {
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
      !user_scheds.empty(),
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
    TORCH_CHECK(child, "Created child of TrieNode should not be null!");
    ++(child->visits);
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
    } else {
      TORCH_WARN(
          "You are overwriting the current user schedule for a definition!");
      user_scheds[input_id.id].at(device) = UserSchedule();
    }
  }
  return &user_scheds[input_id.id].at(device);
}

TrieNode* FusionCache::rootTriePtr() {
  ++(root_.get()->visits);
  return root_.get();
}

void FusionCache::serialize(std::string filename) const {
  flatbuffers::FlatBufferBuilder builder(1024);
  // TODO: Serialize Fusion IR containers

  // 1. Flattened the TrieStructure using breadth-first search
  // 2. Map RecordFunctor pointer to its position in flattened order
  std::map<RecordFunctor*, size_t> map_record_functor_to_trie_node_id;
  std::vector<TrieNode*> bfs_order;
  std::deque<TrieNode*> queue = {root_.get()};
  while (!queue.empty()) {
    TrieNode* current_node = queue.front();
    queue.pop_front();

    map_record_functor_to_trie_node_id.emplace(
        current_node->record.get(), bfs_order.size());
    bfs_order.push_back(current_node);

    for (auto&& child : current_node->children) {
      queue.push_back(child.second.get());
    }
  }

  // 3. Serialize TrieNode in Breadth-First Search (BFS) order
  //
  // Note 1) All TrieNode pointers are mapped to their corresponding index in
  // BFS traversal order.
  //
  // Note 2) We cannot create nested Flatbuffer objects. e.g., All Flatbuffer
  // objects MUST be created before the start of the table they are referenced
  // in.
  //
  // Thus, it is simplier to get the entire BFS order first, and then serialize
  // the flattened Trie structure.
  std::vector<flatbuffers::Offset<serde::TrieNode>> fb_nodes;
  for (TrieNode* node : bfs_order) {
    auto serialized_trie_node =
        node->serialize(builder, map_record_functor_to_trie_node_id);
    fb_nodes.push_back(serialized_trie_node);
  }

  // 4. Map the terminal nodes to their BFS positions.
  std::vector<size_t> terminal_node_idx;
  terminal_node_idx.reserve(terminal_nodes_.size());
  for (auto node : terminal_nodes_) {
    terminal_node_idx.push_back(
        map_record_functor_to_trie_node_id.at(node->record.get()));
  }

  // 5. Build FusionCache flatbuffer object
  // table FusionCache {
  //  max_fusions: ulong;
  //  structure: [TrieNode];
  //  terminal_nodes: [ulong];
  // }
  auto fusion_cache = serde::CreateFusionCacheDirect(
      builder, max_fusions_, &fb_nodes, &terminal_node_idx);
  builder.Finish(fusion_cache, "NV00" /* file_identifier */);

  // 6. Write flatbuffer binary to file
  auto fb = builder.GetBufferSpan();
  auto file_handle = std::fopen(filename.c_str(), "wb");
  size_t write_status =
      std::fwrite(fb.data(), sizeof(uint8_t), fb.size(), file_handle);
  TORCH_INTERNAL_ASSERT(
      write_status == fb.size(),
      "Failed to write entire FusionCache Flatbuffer.\n");
  std::fclose(file_handle);
}

namespace {
typedef std::vector<uint8_t> BinaryBuffer;

BinaryBuffer openFusionCache(std::string filename) {
  auto file_handle = std::fopen(filename.c_str(), "rb");
  TORCH_CHECK(file_handle != nullptr, "Failed to open FusionCache buffer.");

  auto file_path = fs::path(filename.c_str());
  auto file_size = fs::file_size(file_path);
  TORCH_CHECK(file_size > 0, "FusionCache buffer is empty.");

  BinaryBuffer buffer(file_size);
  size_t read_status =
      std::fread(buffer.data(), sizeof(uint8_t), file_size, file_handle);
  TORCH_CHECK(
      read_status == file_size, "Failed to read entire FusionCache buffer.\n");
  return buffer;
}

const serde::FusionCache* verifyFusionCache(const BinaryBuffer& buffer) {
  auto fusion_cache_buffer = serde::GetFusionCache(buffer.data());
  flatbuffers::Verifier v(buffer.data(), buffer.size());
  TORCH_CHECK(
      fusion_cache_buffer->Verify(v),
      "Failed to verify the integrity of FusionCache buffer.");
  TORCH_CHECK(
      serde::FusionCacheBufferHasIdentifier(buffer.data()),
      "Failed to verify the schema version of the FusionCache buffer");
  return fusion_cache_buffer;
}

} // namespace

void FusionCache::deserialize(std::string filename) {
  // 0. Load flatbuffer binary from file
  // table FusionCache {
  //  max_fusions: ulong;
  //  structure: [TrieNode];
  //  terminal_nodes: [ulong];
  // }
  TORCH_CHECK(
      fusions_.empty(),
      "Deserialization is prohibited if FusionCache is already populated.");
  auto buffer = openFusionCache(filename);
  auto fusion_cache_buffer = verifyFusionCache(buffer);

  // 1. Deserialize max_fusions field
  max_fusions_ = fusion_cache_buffer->max_fusions();

  // 2. Deserialize fusions: (Fusion) and structure: (TrieNode) fields
  std::generate_n(
      std::back_inserter(fusions_),
      fusion_cache_buffer->terminal_nodes()->size(),
      [] { return std::make_unique<FusionSchedules>(); });

  serde::RecordFunctorFactory record_functor_factory;

  using BfsState = std::pair<TrieNode*, size_t>;
  std::deque<BfsState> queue = {
      {root_.get() /* TrieNode pointer */, 0 /* structure_idx */}};

  // state_queue holds the FusionState for each BfsState in the queue.
  std::deque<std::unique_ptr<FusionState>> state_queue;

  // Create empty fusion container for root node
  state_queue.emplace_back(std::make_unique<FusionState>());

  // bfs_order is used to map indices in the structure field to their
  // corresponding TrieNode pointers. It is used to reconstruct the
  // terminal_nodes vector.
  std::vector<TrieNode*> bfs_order;

  // Starting from the root node, we build the Trie structure in breadth-first
  // (BFS) order.
  while (!queue.empty()) {
    auto& [trie_ptr, structure_idx] = queue.front();

    // Update BFS order
    bfs_order.push_back(trie_ptr);

    // Get corresponding flatbuffer object for current TrieNode
    auto fb_trie_node = fusion_cache_buffer->structure()->Get(structure_idx);

    // While traversing the Trie Structure, build the Fusion Container by
    // adding the TrieNode's RecordFunctor
    auto state = state_queue.front().get();
    state->addRecord(trie_ptr->record.get()->clone());

    // Deserialize Table TrieNode => Field: visits (ulong)
    trie_ptr->visits = fb_trie_node->visits();

    // Build fusion container if current node is a terminal node
    if (fb_trie_node->is_terminal()) {
      TORCH_CHECK(
          fb_trie_node->children()->size() == 0,
          "This terminal node should not have any children.")
      TORCH_CHECK(
          fb_trie_node->record()->type() == serde::RecordType_End,
          "This terminal node should have an EndRecord RecordFunctor")
      TORCH_CHECK(
          trie_ptr->fusion_id == fb_trie_node->fusion_id(),
          "The fusion id for this TrieNode should already be set.")
      Fusion* fusion =
          queryFusionSchedules(fb_trie_node->fusion_id())->preschedFusion();
      state->buildFusionIr(fusion);
    }

    // Table TrieNode => Field: children: [ulong]
    // Create Children TrieNode
    for (auto child_bfs_idx : *fb_trie_node->children()) {
      auto fb_child_trie_node =
          fusion_cache_buffer->structure()->Get(child_bfs_idx);

      // Create child RecordFunctor
      auto serde_buffer = fb_child_trie_node->record();
      auto rec =
          record_functor_factory.parse(serde_buffer->type(), serde_buffer);

      // Deserialize the record and fusion id fields in the TrieNode table
      auto status = trie_ptr->children.emplace(
          rec,
          std::make_unique<TrieNode>(
              rec, trie_ptr, fb_child_trie_node->fusion_id()));
      TORCH_CHECK(
          status.second,
          "Fusion-Cache Deserialization: Failed to add child to the current TrieNode.");

      // Add child TrieNode to BFS queue
      queue.emplace_back(
          status.first->second.get() /* TrieNode pointer */, child_bfs_idx);
      state_queue.emplace_back(state->clone());
    }

    // Destroy current fusion state
    queue.pop_front();
    state_queue.pop_front();
  }

  // Deserialize terminal_nodes field in the FusionCache table
  for (auto idx : *fusion_cache_buffer->terminal_nodes()) {
    terminal_nodes_.push_back(bfs_order.at(idx));
  }
}

} // namespace nvfuser::python_frontend
