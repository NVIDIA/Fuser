// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#include <python_frontend/fusion_cache.h>
#include <python_frontend/fusion_record_serde.h>
#include <mutex>

namespace nvfuser::python_frontend {

static std::mutex fusion_cache_lock;
FusionCache* FusionCache::singleton_ = nullptr;

UserSchedule::UserSchedule() : schedule(nullptr), executor(nullptr) {
  schedule = std::make_unique<Fusion>();
  executor = std::make_unique<FusionExecutor>();
}

FusionSchedules::FusionSchedules()
    : auto_gen_schedules(nullptr), user_def_schedules() {
  auto_gen_schedules =
      std::make_unique<FusionExecutorCache>(std::make_unique<Fusion>());
}

Fusion* FusionSchedules::preschedFusion() {
  auto fusion = auto_gen_schedules->fusion();
  TORCH_CHECK(fusion != nullptr, "Prescheduled Fusion is unexpectedly null!");
  return fusion;
}

TrieNode::TrieNode(RecordFunctor* rec, size_t _fusion_id)
    : record(rec), children(), fusion_id(_fusion_id), visits(0) {}

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
  std::lock_guard<std::mutex> guard(fusion_cache_lock);
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

void FusionCache::print(std::ostream& os) {
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
  std::lock_guard<std::mutex> guard(fusion_cache_lock);
  if (singleton_ != nullptr) {
    auto max_fusions = singleton_->max_fusions_;
    delete singleton_;
    singleton_ = new FusionCache(max_fusions);
  }
}

FusionCache::FusionCache(size_t max_fusions)
    : max_fusions_(max_fusions),
      root_(nullptr),
      trie_ptr_(nullptr),
      fusions_(),
      user_def_input_encodings_() {
  RecordFunctor* start = new StartRecord();
  root_ = std::make_unique<TrieNode>(start);
  trie_ptr_ = root_.get();
}

c10::optional<TrieNode*> FusionCache::queryChildren(RecordFunctor* rec) const {
  std::lock_guard<std::mutex> guard(fusion_cache_lock);
  TORCH_CHECK(
      !triePtr()->isTerminal(),
      "There should be no children from a Terminal Node!");
  TORCH_CHECK(rec, "Record is null!");
  auto trie_node = triePtr()->children.find(rec);
  if (trie_node == std::end(triePtr()->children)) {
    return c10::nullopt;
  } else {
    return c10::optional<TrieNode*>(trie_node->second.get());
  }
}
FusionSchedules& FusionCache::queryFusionSchedules(size_t fusion_id) {
  std::lock_guard<std::mutex> guard(fusion_cache_lock);
  TORCH_CHECK(
      fusion_id < fusions_.size(),
      "Invalid scheduler query for id:",
      fusion_id);
  return fusions_.at(fusion_id);
}
c10::optional<size_t> FusionCache::queryUserScheduleId(
    const FusionSchedules& scheds,
    const at::ArrayRef<c10::IValue>& inputs,
    int device) {
  c10::optional<size_t> result = c10::nullopt;

  auto& user_scheds = scheds.user_def_schedules;
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
    const FusionSchedules& scheds,
    size_t id,
    int device) {
  auto& user_scheds = scheds.user_def_schedules;
  TORCH_CHECK(
      user_scheds.size() > 0,
      "Expecting there to be at least one user schedule!");
  auto user_sched = user_scheds.find(id);
  TORCH_CHECK(
      user_sched != user_scheds.end(), "Lookup of non-existent user schedule!");
  return user_sched->second.at(device);
}

c10::optional<size_t> FusionCache::createChild(RecordFunctor* rec) {
  std::lock_guard<std::mutex> guard(fusion_cache_lock);
  c10::optional<size_t> result = c10::nullopt;
  TORCH_CHECK(
      !triePtr()->isTerminal(),
      "Cannot create a trie node from a terminal node!");
  TORCH_CHECK(rec, "Record is null!");

  size_t fusion_id = 0;
  if (rec->recordType() == serde::RecordType_End) {
    TORCH_CHECK(
        (fusions_.size() + 1) <= max_fusions_,
        "The number of fusions in nvfuser has exceeded ",
        max_fusions_,
        "fusions.  The max_fusions for the FusionCache might need to be ",
        "increased if the max number is not being exceeded due to an error.");
    fusions_.emplace_back(FusionSchedules());
    fusion_id = fusions_.size() - 1;
    result = c10::optional<size_t>(fusion_id);
  }

  // Copying the record owned by the FusionDefinition that calls this function
  // so the trie owns a copy when the FusionDefinition gets destroyed rather
  // than managing a shared pointer that would only share with
  // FusionDefinition that creates a trie node but not cache lookups
  RecordFunctor* new_rec = rec->clone();
  triePtr()->children[new_rec] = std::make_unique<TrieNode>(new_rec, fusion_id);
  if (rec->recordType() == serde::RecordType_End) {
    terminal_nodes_.push_back(triePtr()->children[new_rec].get());
  }
  if (isDebugDumpEnabled(DebugDumpOption::PythonFrontendDebug)) {
    std::stringstream ss;
    new_rec->print(ss);
    std::cout << "\nFusionDefinition: Create new trie node for: " << ss.str()
              << "\n";
  }
  return result;
}

UserSchedule* FusionCache::createUserSchedule(
    FusionSchedules& scheds,
    const at::ArrayRef<c10::IValue>& inputs,
    int device) {
  auto& user_scheds = scheds.user_def_schedules;
  auto input_id = user_def_input_encodings_.lookupId(inputs);
  auto user_sched = user_scheds.find(input_id.id);
  // TODO: Make this better.  This just seems ugly and inefficient.
  if (user_sched == user_scheds.end()) {
    user_scheds[input_id.id] = std::vector<UserSchedule>(device + 1);
  } else {
    user_scheds[input_id.id].resize(device + 1);
  }
  return &user_scheds[input_id.id].at(device);
}

void FusionCache::resetTriePtr() {
  trie_ptr_ = root_.get();
  TORCH_CHECK(triePtr()->record->recordType() == serde::RecordType_Start);
  ++(triePtr()->visits);
}

void FusionCache::traverseTrie(RecordFunctor* rec) {
  TORCH_CHECK(
      !triePtr()->isTerminal(), "Cannot traverse trie from a terminal entry!");
  auto trie_node = triePtr()->children.find(rec);
  TORCH_CHECK(
      trie_node != std::end(triePtr()->children),
      "Trie Node for Trie Traverse is not found!");
  TORCH_CHECK(trie_node->second, "Record in Trie Node is null!");
  trie_ptr_ = trie_node->second.get();
  ++(triePtr()->visits);
}

TrieNode* FusionCache::triePtr() const {
  TORCH_INTERNAL_ASSERT(
      trie_ptr_ != nullptr, "The trie node is unexpectedly null.");
  return trie_ptr_;
}

void FusionCache::serialize(std::string filename) const {
  std::lock_guard<std::mutex> guard(fusion_cache_lock);
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
  using BfsState = std::pair<TrieNode*, size_t>;

  auto buffer = openFusionCache(filename);
  auto fusion_cache_buffer = verifyFusionCache(buffer);

  // 1. Deserialize max_fusions field
  max_fusions_ = fusion_cache_buffer->max_fusions();

  // 2. Deserialize fusions: (Fusion) and structure: (TrieNode) fields
  fusions_.resize(fusion_cache_buffer->terminal_nodes()->size());

  RecordFunctorFactory record_functor_factory;

  std::vector<TrieNode*> bfs_order;
  std::deque<BfsState> queue = {
      {root_.get() /* TrieNode pointer */, 0 /* structure_idx */}};

  // Create empty fusion container for root node
  std::deque<std::unique_ptr<FusionState>> state_queue;
  state_queue.emplace_back(std::make_unique<FusionState>());

  // Starting from the root node, we build the Trie structure in breadth-first
  // (BFS) order.
  while (!queue.empty()) {
    auto current = queue.front();
    // Replace with structure binding in c++17
    TrieNode* trie_ptr = current.first;
    size_t structure_idx = current.second;
    queue.pop_front();

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
          queryFusionSchedules(fb_trie_node->fusion_id()).preschedFusion();
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
          std::make_unique<TrieNode>(rec, fb_child_trie_node->fusion_id()));
      TORCH_CHECK(
          status.second,
          "Fusion-Cache Deserialization: Failed to add child to the current TrieNode.");

      // Add child TrieNode to BFS queue
      queue.emplace_back(
          status.first->second.get() /* TrieNode pointer */, child_bfs_idx);
      state_queue.emplace_back(state->clone());
    }

    // Destroy current fusion state
    state_queue.pop_front();
  }

  // Deserialize terminal_nodes field in the FusionCache table
  for (auto idx : *fusion_cache_buffer->terminal_nodes()) {
    terminal_nodes_.push_back(bfs_order.at(idx));
  }
}

} // namespace nvfuser::python_frontend
