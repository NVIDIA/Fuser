// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#include <ATen/cuda/CUDAContext.h>
#include <nvrtc.h>

#include <debug.h>
#include <instrumentation.h>
#include <options.h>
#include <python_frontend/fusion_cache.h>
#include <serde/fusion_record.h>
#include <utils.h>

#include <filesystem>
namespace fs = std::filesystem;

#ifdef _WIN32
#include <c10/util/win32-headers.h>
#else
#include <pthread.h>
#include <unistd.h>
#endif

namespace nvfuser::python_frontend {

namespace {
using BinaryBuffer = std::vector<uint8_t>;

// Generate temporary file for this FusionCacheBuffer
std::string getSerdeTmpFile() {
#ifdef _WIN32
  const unsigned int pid = GetCurrentProcessId();
#else
  const unsigned int pid = getpid();
#endif // _WIN32
  std::stringstream ss;
  ss << "nvf_serde_tmp_" << pid;
  return ss.str();
}

std::string getSerdeFile() {
  auto device_prop = at::cuda::getCurrentDeviceProperties();
  int cuda_major = 0;
  int cuda_minor = 0;
  NVFUSER_NVRTC_SAFE_CALL(nvrtcVersion(&cuda_major, &cuda_minor));

  std::stringstream ss;
  ss << "nvf_serde";
  ss << "_device" << device_prop->major << "_" << device_prop->minor;
  ss << "_cuda" << cuda_major << "_" << cuda_minor;
  return ss.str();
}

// Get std::filesystem::path to specified file in nvfuser kernel database
// directory.
fs::path getSerdeFilePath(const std::string& file_name) {
  fs::path kernel_db_path = fs::temp_directory_path() / "nvfuser_kernel_db";
  if (!fs::is_directory(kernel_db_path)) {
    try {
      fs::create_directory(kernel_db_path);
    } catch (const std::exception& e) {
      NVF_ERROR(
          "Unable to create nvFuser Kernel DB directory! ",
          kernel_db_path.string(),
          e.what());
    }
  }
  return kernel_db_path / file_name;
}

BinaryBuffer openFusionCache(std::string filename) {
  FUSER_PERF_SCOPE("Flatbuffers::openFusionCache");
  auto file_handle = std::fopen(filename.c_str(), "rb");
  NVF_CHECK(file_handle != nullptr, "Failed to open FusionCache buffer.");

  auto file_path = fs::path(filename.c_str());
  auto file_size = fs::file_size(file_path);
  NVF_CHECK(file_size > 0, "FusionCache buffer is empty.");

  BinaryBuffer buffer(file_size);
  size_t read_status =
      std::fread(buffer.data(), sizeof(uint8_t), file_size, file_handle);
  NVF_CHECK(
      read_status == file_size, "Failed to read entire FusionCache buffer.\n");
  return buffer;
}

// This check function only throws errors if strict flag is enabled.
const serde::FusionCache* verifyFusionCache(const BinaryBuffer& buffer) {
  FUSER_PERF_SCOPE("Flatbuffers::verifyFusionCache");
  auto fusion_cache_buffer = serde::GetFusionCache(buffer.data());

  // Check flatbuffer integrity
  flatbuffers::Verifier v(buffer.data(), buffer.size());
  NVF_CHECK(
      fusion_cache_buffer->Verify(v),
      "Failed to verify the integrity of FusionCache buffer.");

  // Check schema version
  NVF_CHECK(
      serde::FusionCacheBufferHasIdentifier(buffer.data()),
      "Failed to verify the schema version of the FusionCache buffer");

  // Check device major and minor versions
  auto device_prop = at::cuda::getCurrentDeviceProperties();
  NVF_CHECK(
      device_prop->major == fusion_cache_buffer->device_major() &&
          device_prop->minor == fusion_cache_buffer->device_minor(),
      false,
      "Expected cuda version ",
      device_prop->major,
      ".",
      device_prop->minor,
      " but flatbuffer has cuda version ",
      fusion_cache_buffer->device_major(),
      ".",
      fusion_cache_buffer->device_minor());

  // Check cuda installation
  int cuda_major = 0;
  int cuda_minor = 0;
  NVFUSER_NVRTC_SAFE_CALL(nvrtcVersion(&cuda_major, &cuda_minor));
  NVF_CHECK(
      cuda_major == fusion_cache_buffer->cuda_major() &&
          cuda_minor == fusion_cache_buffer->cuda_minor(),
      "Expected cuda version ",
      cuda_major,
      ".",
      cuda_minor,
      " but flatbuffer has cuda version ",
      fusion_cache_buffer->cuda_major(),
      ".",
      fusion_cache_buffer->cuda_minor());

  return fusion_cache_buffer;
}

} // namespace

void serialize() {
  auto tmp_file_path = getSerdeFilePath(getSerdeTmpFile());
  FusionCache::get()->serialize(tmp_file_path);

  // Save to a per-process temporary file to avoid multi-process contention.
  // Then, rename the temporary file to the actual file. If the actual file
  // already exists, then the rename may fail or replace the actual file.
  // Files replaced through this process should remain extant if they are being
  // read because of UNIX filesystem properties, but this behavior is
  // unverified.
  auto file_path = getSerdeFilePath(getSerdeFile());
  std::error_code rename_ec;
  fs::rename(tmp_file_path, file_path, rename_ec);

  // Failed to replace common workspace, so remove the temporary file.
  if (rename_ec) {
    try {
      fs::remove(tmp_file_path);
      std::cout
          << "Removed temporary file because we could not replace common workspace. Exception:\t"
          << rename_ec.message() << std::endl;
    } catch (const std::exception& e) {
      std::cout << "Failed to delete temporary file. Exception:\t" << e.what()
                << std::endl;
    }
  }
}

// FusionCache static data member definitions for singleton usage
std::mutex FusionCache::singleton_lock_;
FusionCache* FusionCache::singleton_ = nullptr;

UserSchedule::UserSchedule() : schedule(nullptr), executor(nullptr) {
  schedule = std::make_unique<Fusion>();
  executor = std::make_unique<FusionExecutor>();
}

FusionSchedules::FusionSchedules(int64_t fusion_id)
    : auto_gen_schedules(nullptr),
      user_def_schedules(),
      last_user_def_scheduled_ir(nullptr),
      last_user_def_executor(nullptr),
      scheds_lock(),
      fusion_id_{fusion_id} {
  auto_gen_schedules = std::make_unique<FusionExecutorCache>(
      std::make_unique<Fusion>(), fusion_id);
}

Fusion* FusionSchedules::preschedFusion() {
  auto fusion = auto_gen_schedules->fusion();
  NVF_CHECK(fusion != nullptr, "Prescheduled Fusion is unexpectedly null!");
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
  return (record.get()->recordType() == serde::RecordType::End);
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

FusionCache* FusionCache::get(
    size_t max_fusions,
    bool load_from_default_workspace) {
  FUSER_PERF_SCOPE("FusionCache::get");
  std::lock_guard<std::mutex> guard(singleton_lock_);
  if (singleton_ == nullptr) {
    singleton_ = new FusionCache(max_fusions);

    // Deserialize cache hierarchy from common workspace automatically
    auto file_path = getSerdeFilePath(getSerdeFile()).native();
    if (load_from_default_workspace && fs::exists(file_path)) {
      try {
        singleton_->deserialize(file_path);
      } catch (const std::exception& deserialize_exception) {
        // The saved workspace can become out-of-date between nvfuser updates.
        // Send warning and delete the incompatible workspace.
        // A new workspace will be saved upon program exit.
        std::cout
            << "Warning: Failed to deserialize common workspace.\n"
            << "A new workspace will be saved upon program exit after deleting incompatible workspace."
            << std::endl;

        // Hide exception message because it should be resolved by saving a new
        // workspace.
        if (!isOptionDisabled(DisableOption::ParallelSerde)) {
          std::cout
              << "Use NVFUSER_DISABLE=parallel_serde to print exception message."
              << std::endl;
        } else {
          std::cout << deserialize_exception.what() << std::endl;
        }

        // Delete incompatible workspace
        std::error_code remove_ec;
        fs::remove(file_path, remove_ec);
        if (remove_ec) {
          std::cout << "Failed to delete common workspace. Exception:\t"
                    << remove_ec.message() << std::endl;
        }

        // Reset FusionCache if there is an issue with the current workspace.
        delete singleton_;
        singleton_ = new FusionCache(max_fusions);
      }
    }
  }
  NVF_CHECK(
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
        if (end->record->recordType() != serde::RecordType::Start) {
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
std::optional<TrieNode*> FusionCache::queryChildren(
    TrieNode* node,
    RecordFunctor* rec) const {
  NVF_CHECK(
      !node->isTerminal(), "There should be no children from a Terminal Node!");
  NVF_CHECK(rec, "Record is null!");
  auto trie_node = node->children.find(rec);
  if (trie_node == std::end(node->children)) {
    return std::nullopt;
  } else {
    ++(trie_node->second.get()->visits);
    return std::optional<TrieNode*>(trie_node->second.get());
  }
}

FusionSchedules* FusionCache::queryFusionSchedules(size_t fusion_id) const {
  NVF_CHECK(
      fusion_id < fusions_.size(),
      "Invalid scheduler query for id:",
      fusion_id);
  FusionSchedules* ptr = fusions_.at(fusion_id).get();
  NVF_CHECK(ptr != nullptr, "Unexpected null FusionSchedules object.");
  return ptr;
}
std::optional<size_t> FusionCache::queryUserScheduleId(
    const FusionSchedules* scheds,
    const at::ArrayRef<c10::IValue>& inputs) {
  std::optional<size_t> result = std::nullopt;

  auto& user_scheds = scheds->user_def_schedules;
  if (!user_scheds.empty()) {
    auto input_id = user_def_input_encodings_.lookupId(inputs);
    auto user_sched = user_scheds.find(input_id.id);
    if (user_sched != user_scheds.end()) {
      return std::optional<size_t>(user_sched->first);
    }
  }
  return result;
}
const UserSchedule& FusionCache::queryUserSchedule(
    const FusionSchedules* scheds,
    size_t id,
    int device) const {
  auto& user_scheds = scheds->user_def_schedules;
  NVF_CHECK(
      !user_scheds.empty(),
      "Expecting there to be at least one user schedule!");
  auto user_sched = user_scheds.find(id);
  NVF_CHECK(
      user_sched != user_scheds.end(), "Lookup of non-existent user schedule!");
  return user_sched->second.at(device);
}

TrieNode* FusionCache::createChild(TrieNode* node, RecordFunctor* rec) {
  FUSER_PERF_SCOPE("FusionCache::createChild");
  TrieNode* child = nullptr;
  NVF_CHECK(
      !node->isTerminal(), "Cannot create a trie node from a terminal node!");
  NVF_CHECK(rec, "Record is null!");

  std::lock_guard<std::mutex> guard(node->trie_node_lock);

  // As a thread-safety compromise for fast queries, the node is re-queried
  // prior to child creation incase another thread slipped in the node.
  auto child_node = queryChildren(node, rec);
  if (child_node.has_value()) {
    child = child_node.value();
  } else {
    size_t fusion_id = 0;
    if (rec->recordType() == serde::RecordType::End) {
      NVF_CHECK(
          (fusions_.size() + 1) <= max_fusions_,
          "The number of fusions in nvfuser has exceeded ",
          max_fusions_,
          "fusions.  The max_fusions for the FusionCache might need to be ",
          "increased if the max number is not being exceeded due to an error.");
      fusion_id = fusions_.size();
      fusions_.emplace_back(std::make_unique<FusionSchedules>(fusion_id));
    }

    // Copying the record owned by the FusionDefinition that calls this function
    // so the trie owns a copy when the FusionDefinition gets destroyed rather
    // than managing a shared pointer that would only share with
    // FusionDefinition that creates a trie node but not cache lookups
    RecordFunctor* new_rec = rec->clone();
    node->children[new_rec] =
        std::make_unique<TrieNode>(new_rec, node, fusion_id);
    child = node->children[new_rec].get();
    NVF_CHECK(child, "Created child of TrieNode should not be null!");
    ++(child->visits);
    if (rec->recordType() == serde::RecordType::End) {
      terminal_nodes_.push_back(node->children[new_rec].get());
    }
    if (isDebugDumpEnabled(DebugDumpOption::PythonFrontendDebug)) {
      std::stringstream ss;
      new_rec->print(ss);
      debug() << "\nFusionDefinition: Create new trie node for: " << ss.str()
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
  user_scheds[input_id.id].at(device).fusion_id_ = scheds->fusion_id_;
  user_scheds[input_id.id].at(device).device_id_ = device;
  return &user_scheds[input_id.id].at(device);
}

TrieNode* FusionCache::rootTriePtr() {
  ++(root_.get()->visits);
  return root_.get();
}

void FusionCache::serialize(std::string filename) const {
  FUSER_PERF_SCOPE("FusionCache::serialize");
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
  // 5. Serialize each FusionExecutorCache for each fusion.
  std::vector<size_t> terminal_node_idx;
  terminal_node_idx.reserve(terminal_nodes_.size());

  using fb_fusion_executor_cache =
      flatbuffers::Offset<serde::FusionExecutorCache>;
  std::vector<fb_fusion_executor_cache> fb_auto_gen_schedules;
  fb_auto_gen_schedules.reserve(terminal_nodes_.size());

  for (auto node : terminal_nodes_) {
    terminal_node_idx.push_back(
        map_record_functor_to_trie_node_id.at(node->record.get()));

    auto schedule = queryFusionSchedules(node->fusion_id);
    fb_auto_gen_schedules.emplace_back(
        schedule->auto_gen_schedules->serialize(builder));
  }

  auto device_prop = at::cuda::getCurrentDeviceProperties();
  int cuda_major = 0;
  int cuda_minor = 0;
  NVFUSER_NVRTC_SAFE_CALL(nvrtcVersion(&cuda_major, &cuda_minor));

  // 6. Build FusionCache flatbuffer object
  // See table definition for FusionCache in serde/fusion_cache.fbs
  auto fusion_cache = serde::CreateFusionCacheDirect(
      builder,
      max_fusions_,
      &fb_nodes,
      &terminal_node_idx,
      &fb_auto_gen_schedules,
      FusionExecutor::getGlobalFusionCount(),
      device_prop->major,
      device_prop->minor,
      cuda_major,
      cuda_minor);
  builder.Finish(fusion_cache, /*file_identifier=*/"NV01");

  // 6. Write flatbuffer binary to file
  auto fb = builder.GetBufferSpan();
  auto file_handle = std::fopen(filename.c_str(), "wb");
  size_t write_status =
      std::fwrite(fb.data(), sizeof(uint8_t), fb.size(), file_handle);
  NVF_ERROR(
      write_status == fb.size(),
      "Failed to write entire FusionCache Flatbuffer.\n");
  std::fclose(file_handle);
}

void FusionCache::deserialize(std::string filename) {
  // See table definition for FusionCache in serde/fusion_cache.fbs
  // 0. Load flatbuffer binary from file
  FUSER_PERF_SCOPE("FusionCache::deserialize");
  NVF_CHECK(
      fusions_.empty(),
      "Deserialization is prohibited if FusionCache is already populated.");
  const BinaryBuffer& buffer = openFusionCache(filename);
  const serde::FusionCache* fusion_cache_buffer = verifyFusionCache(buffer);

  // See table definition for FusionCache in serde/fusion_cache.fbs
  FUSER_PERF_SCOPE("FusionCache::deserialize");
  NVF_CHECK(fusion_cache_buffer != nullptr, "Fusion Cache buffer is invalid.");

  // 0. Set static fusion count in Fusion Executor
  FusionExecutor::setGlobalFusionCount(
      fusion_cache_buffer->global_fusion_count());

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
      NVF_CHECK(
          fb_trie_node->children()->size() == 0,
          "This terminal node should not have any children.")
      NVF_CHECK(
          fb_trie_node->record()->type() == serde::RecordType::End,
          "This terminal node should have an EndRecord RecordFunctor")
      NVF_CHECK(
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
      NVF_CHECK(
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

  std::atomic<bool> detect_exception_in_thread_pool{false};
  // Deserialize terminal_nodes field in the FusionCache table
  for (auto idx : c10::irange(fusions_.size())) {
    auto node_idx = fusion_cache_buffer->terminal_nodes()->Get(idx);
    auto trie_node = bfs_order.at(node_idx);
    terminal_nodes_.push_back(trie_node);

    auto fb_fec_node = fusion_cache_buffer->auto_gen_schedules()->Get(idx);
    auto fusion_schedule = queryFusionSchedules(trie_node->fusion_id);

    if (!isOptionDisabled(DisableOption::ParallelSerde)) {
      // Parallelize the deserialization of each FusionExecutorCache.
      getThreadPool()->run([=, &detect_exception_in_thread_pool]() {
        FUSER_PERF_SCOPE("FusionCache::deserializeFusionParallel");
        try {
          fusion_schedule->auto_gen_schedules->deserialize(
              fb_fec_node, (int64_t)trie_node->fusion_id);
        } catch (const std::exception& e) {
          // Set flag inside lambda so we can throw an exception after thread
          // pool completes its work.
          detect_exception_in_thread_pool.store(true);
        }
      });
    } else {
      FUSER_PERF_SCOPE("FusionCache::deserializeFusionSerial");
      fusion_schedule->auto_gen_schedules->deserialize(
          fb_fec_node, (int64_t)trie_node->fusion_id);
    }
  }

  if (!isOptionDisabled(DisableOption::ParallelSerde)) {
    // Wait until all fusion executor caches are deserialized
    getThreadPool()->waitWorkComplete();
    NVF_ERROR(
        !detect_exception_in_thread_pool.load(),
        "Detected exception while deserializing fusions in parallel.\n",
        "Use NVFUSER_DISABLE=parallel_serde to print exception message.");
  }
}

} // namespace nvfuser::python_frontend
