// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#include <runtime/fusion_cache_utils.h>

#include <fusion_segmenter.h>
#include <instrumentation.h>
#include <ir/all_nodes.h>
#include <polymorphic_value.h>
#include <runtime/executor_kernel_arg.h>
#include <sys/select.h>

#include <chrono>
#include <unordered_set>
#include "expr_evaluator.h"

namespace nvfuser {

namespace {

// Copy bytes of value to back of buffer. This is templated in order to avoid
// implicit cast such as int64_t -> size_t that might lose information.
template <typename T>
void encodeBuffer(T value, std::string& buffer) {
  const char* v = reinterpret_cast<char*>(&value);
  for (const auto i : c10::irange(sizeof(T))) {
    (void)i; // Suppress unused variable warning
    buffer.push_back(*(v++));
  }
}
} // namespace

ArgumentManager::ArgumentManager(
    KernelArgumentHolder& args,
    const RuntimeWorkSpace& runtime_workspace,
    const std::vector<Val*>& fusion_inputs)
    : fusion_args_(args) {
  // map from val to args
  mapFusionInputsToArgs(
      fusion_inputs, runtime_workspace.group_extent_binding_order);
  setLastUsedSegmentID(runtime_workspace.group_run_order);
}

const std::unordered_map<Val*, const PolymorphicValue*>& ArgumentManager::
    getTensorMap() {
  return tensor_map_;
}
const PolymorphicValue* ArgumentManager::checkTensorMap(Val* v) {
  return tensor_map_.at(v);
}

template <typename T>
void ArgumentManager::addOutputsToArgsAndTensorMap(
    const std::vector<Val*>& group_outputs,
    const T& group_runtime_outputs) {
  // Insert graph segment output to tensor map
  NVF_ERROR(
      group_outputs.size() == group_runtime_outputs.size(),
      "Output size does not match.");

  for (const size_t group_out_i : c10::irange(group_outputs.size())) {
    Val* output = group_outputs[group_out_i];
    const PolymorphicValue*& runtime_output = tensor_map_[output];
    if (runtime_output != nullptr) {
      // A trivial forwarding output or a dupliated output shares the same
      // `Val*` as another fusion input/output. In those cases, we keep
      // mapping it to the same runtime output.
      continue;
    }

    if constexpr (std::is_pointer_v<
                      decltype(group_runtime_outputs[group_out_i])>) {
      fusion_args_.push(*group_runtime_outputs[group_out_i]);
    } else {
      fusion_args_.push(group_runtime_outputs[group_out_i]);
    }
    runtime_output = fusion_args_.back();
  }
}

template <typename T>
void ArgumentManager::updateWithSegmentOutputs(
    const std::vector<Val*>& group_outputs,
    const T& group_runtime_outputs,
    const int64_t group_id) {
  addOutputsToArgsAndTensorMap(group_outputs, group_runtime_outputs);
  deleteUnusedArgs(group_id);
}

template void ArgumentManager::addOutputsToArgsAndTensorMap<
    std::vector<at::Tensor>>(
    const std::vector<Val*>& group_outputs,
    const std::vector<at::Tensor>& group_runtime_outputs);

template void ArgumentManager::updateWithSegmentOutputs<
    std::vector<at::Tensor>>(
    const std::vector<Val*>&,
    const std::vector<at::Tensor>&,
    const int64_t);

template void ArgumentManager::addOutputsToArgsAndTensorMap<
    KernelArgumentHolder>(
    const std::vector<Val*>& group_outputs,
    const KernelArgumentHolder& group_runtime_outputs);

template void ArgumentManager::updateWithSegmentOutputs<KernelArgumentHolder>(
    const std::vector<Val*>&,
    const KernelArgumentHolder&,
    const int64_t);

void ArgumentManager::mapFusionInputsToArgs(
    const std::vector<Val*>& fusion_inputs,
    const std::vector<Val*>& group_extent_binding_order) {
  int extent_index = 0;
  auto original_args_size = fusion_args_.size();
  // Bind args in the tensor_map
  for (const auto i : c10::irange(original_args_size)) {
    tensor_map_.emplace(fusion_inputs[i], fusion_args_[i]);
    // Bind tensorview inputs values in case some segmented group
    //  needs it down the road.
    // TODO: we probably have done this already up to this point
    //      should consider caching the expression evaluators, both
    //      more convenient and safer than replication
    if (fusion_args_[i]->is<at::Tensor>()) {
      // Note this is very ugly way. We are pushing every single extent to
      // args, because we don't have a better place to hold them.
      auto rank = fusion_args_[i]->as<at::Tensor>().dim();
      for (const auto dim : c10::irange(rank)) {
        fusion_args_.push(
            PolymorphicValue(fusion_args_[i]->as<at::Tensor>().size(dim)));
        tensor_map_.emplace(
            group_extent_binding_order[extent_index++], fusion_args_.back());
      }
    }
  }
}

void ArgumentManager::setLastUsedSegmentID(
    const std::vector<SegmentedGroup*>& group_run_order) {
  // map val to segment_id where arg is lastly used
  std::unordered_map<Val*, int64_t> last_used_segment_map;
  const int64_t num_groups = (int64_t)group_run_order.size();
  // only need to set lifetime of vals if there are more than 3 groups
  if (num_groups >= 3) {
    // start from the 2nd group, since the input of the first group is always
    // the global input and its outputs are always used by at least one of the
    // following groups
    for (auto run_order_id : c10::irange(1l, num_groups)) {
      auto group_to_run = group_run_order.at(run_order_id);
      // set/update life of vals in inputs of this group
      for (auto val : group_to_run->inputs()) {
        // skip fusion inputs and outputs, they may be used by other fusions
        // or code
        if (!val->isFusionInput() && !val->isFusionOutput()) {
          last_used_segment_map[val] = run_order_id;
        }
      }
      // set/update life of vals in outputs of this group
      // skip the last group since its outputs are always the global outputs
      if (run_order_id < num_groups - 1) {
        for (auto val : group_to_run->outputs()) {
          // skip fusion inputs and outputs, they may be used by other fusions
          // or code
          if (!val->isFusionInput() && !val->isFusionOutput()) {
            last_used_segment_map[val] = run_order_id;
          }
        }
      }
    }
    // convert to vals_last_used_at_segment_, so we don't need to iterate over
    // all vals when erasing
    for (auto item : last_used_segment_map) {
      vals_last_used_at_segment_[item.second].push_back(item.first);
    }
  }
}

void ArgumentManager::deleteUnusedArgs(int64_t run_order_id) {
  // erase args corresponding to vals lastly used in this segment
  if (run_order_id >= 1 && vals_last_used_at_segment_.count(run_order_id)) {
    for (auto val : vals_last_used_at_segment_[run_order_id]) {
      fusion_args_.erase(tensor_map_.at(val));
      tensor_map_.erase(val);
    }
  }
}

namespace {

std::vector<SegmentedGroup*> naiveRuntimeOrder(
    const SegmentedFusion* segmented_fusion) {
  // Setup group run order:
  std::unordered_set<Val*> available_input;

  for (const size_t i : c10::irange(segmented_fusion->inputs().size())) {
    auto input_val = segmented_fusion->inputs()[i];
    available_input.insert(input_val);

    if (auto input_tv = dynamic_cast<TensorView*>(input_val)) {
      auto logical_dom =
          TensorDomain::noReductions(input_tv->getLogicalDomain());
      for (const size_t dim : c10::irange(logical_dom.size())) {
        const auto extent = logical_dom[dim]->getMaybeExpandedExtent();
        available_input.insert(extent);
      }
    }
  }

  // Keep track of groups that has run
  std::vector<bool> group_ran(segmented_fusion->groups().size(), false);
  std::vector<SegmentedGroup*> run_order;
  run_order.reserve(group_ran.size());

  while (!std::all_of(
      group_ran.begin(), group_ran.end(), [](bool b) { return b; })) {
    bool one_ran = false;

    // Find the first segment with all inputs available to run
    for (const size_t group_i :
         c10::irange(segmented_fusion->groups().size())) {
      auto& group = segmented_fusion->groups()[group_i];
      if (group_ran[group_i]) {
        continue;
      }
      const auto& group_inputs = group->inputs();
      bool ready_to_run = std::all_of(
          group_inputs.begin(),
          group_inputs.end(),
          [&available_input](Val* val) { return available_input.count(val); });

      if (ready_to_run) {
        run_order.push_back(group);
        const auto& group_outputs = group->outputs();

        // Insert graph segment output to tensor map
        for (const size_t group_out_i : c10::irange(group_outputs.size())) {
          available_input.insert(group_outputs[group_out_i]);
        }
        group_ran[group_i] = true;
        one_ran = true;
      }
    }
    NVF_ERROR(
        one_ran,
        "Couldn't run all groups, something must have gone wrong in segmentation.");
  }

  return run_order;
}

// Computes the memory required to compute a segmented fusion in the given
// order.
// NOTE: this assumes inputs to the unsegmented fusion already reside in memory
// and does not count them against the used memory at any time. Fusion outputs
// are allocated when their producer group is computed and are kept
// indefinitely.
int64_t computePeakMemory(
    const std::vector<SegmentedGroup*>& runtime_order,
    const std::unordered_map<TensorView*, int64_t>& tensor_size) {
  int64_t peak_mem = 0ll;

  // Record that last consumer group of each TV
  std::vector<int64_t> freed_bytes(runtime_order.size(), 0);
  std::unordered_set<TensorView*> seen_tvs;
  for (int64_t i = (int64_t)runtime_order.size() - 1; i >= 0; i--) {
    SegmentedGroup* group = runtime_order[(size_t)i];
    for (Val* val : group->inputs()) {
      if (auto* tv = dynamic_cast<TensorView*>(val);
          tv && !tv->isFusionInput() && !tv->isFusionOutput()) {
        if (seen_tvs.count(tv) == 0) {
          // This is the last consumer of this TV, so it will be freed after
          // execution
          freed_bytes[(size_t)i] += tensor_size.at(tv);
        }
      }
    }
  }

  int64_t cur_mem = 0ll;
  for (size_t j : c10::irange(runtime_order.size())) {
    SegmentedGroup* group = runtime_order[j];
    // TODO: find intermediate memory used local to each group

    // add all the memory required for outputs of this group
    int64_t allocated_bytes = 0ll;
    for (Val* val : group->outputs()) {
      if (auto* tv = dynamic_cast<TensorView*>(val);
          tv && !tv->isFusionInput()) {
        allocated_bytes += tensor_size.at(tv);
      }
    }
    cur_mem += allocated_bytes;

    // Bump peak memory
    if (cur_mem > peak_mem) {
      peak_mem = cur_mem;
    }

    // Now simulate freeing memory
    cur_mem -= freed_bytes[j];
  }

  std::cout << "Found peak memory " << peak_mem << " bytes" << std::endl;

  return peak_mem;
}

// This implements Kahn's algorithm for searching across topological orderings.
// Since for some graphs this can take combinatorially long, we limit the
// runtime in milliseconds. If the search exceeds this runtime, we return the
// best ordering we have inspected so far. We return the peak memory used in
// this ordering, and bool value indicating whether we were able to execute an
// exhaustive search in the time allotted.
//
// Kahn's algorithm topologically orders a DAG by repeatedly doing the
// following:
//   1. Select a node with in-degree zero (i.e. a source node) and append it to
//      the ordering.
//   2. Delete that node from the graph
// Actually deleting nodes from the graph is difficult, so instead one can
// simply track the in-degree of each node, initializing it to the true
// in-degree then decreasing the in-degree counts of all the out-neighbors
// whenever a node is selected. At that point, one can detect new nodes with
// in-degree zero and add them to a queue.
//
// Instead of selecting a single node, we will loop over all possible
// selections. To do this, we keep a stack of "next groups". We push groups onto
// this stack immediately whenever their in-degree becomes zero. At each
// iteration, we pop the next of these nodes and push it onto the runtime order,
// then update the in-degrees of all its out-neighbors. Whenever the runtime
// order reaches its final length, we backtrack until we're able to select the
// next group on the stack.
//
// We can view this as a depth-first traversal of a tree whose nodes correspond
// to choices of which group to run next. There is an out-edge from each node to
// each eligible group and each leaf has depth equal to the total number of
// groups in the segmented fusion. Moving from one node to the next means
// removing one eligible group (the one corresponding to that node) and
// posssibly adding more eligible groups (the ones whose in-degree would become
// zero after removal of that group). Likewise if we move backwards along an
// edge we can undo changes to the eligible set by adding to the in-degrees and
// removing any groups whose in-degrees were zero before removal of that edge's
// node. Then we add the group corresponding to the node we selected in the
// forward traversal of that edge. This lets us move forward and backward
// easily. Importantly, it lets us track our position and backtrack,
// facilitating an efficient traversal of this tree.
//
// By default, we preserve chains, meaning that neighboring nodes with in-degree
// at most one and out-degree at most one are kept adjacent in the returned
// order. In some fusions this can dramatically reduce the number of orderings
// we need to search, and it is typically a useful constraint since it
// encourages L2 locality by immediately consuming the outputs of the previous
// group for those chain segments.
class BruteForceRuntimeOrderOptimizer {
 public:
  static std::tuple<std::vector<SegmentedGroup*>, int64_t, bool> run(
      const SegmentedFusion* segmented_fusion,
      bool preserve_chains = true,
      const int64_t max_runtime_ms = 10000) {
    BruteForceRuntimeOrderOptimizer opt(
        segmented_fusion->groups(), preserve_chains, max_runtime_ms);
    opt.measureAllOrderings();

    std::cout << "Checked " << opt.num_checked_ << " topological orderings"
              << std::endl;

    std::cout << "Found range from " << opt.best_peak_memory_ << " to "
              << opt.worst_peak_memory_ << " bytes peak memory" << std::endl;

    NVF_ERROR(opt.best_run_order_.size() == opt.groups_.size());

    return {opt.best_run_order_, opt.best_peak_memory_, opt.complete_};
  }

 private:
  BruteForceRuntimeOrderOptimizer(
      const std::vector<SegmentedGroup*>& groups,
      bool preserve_chains,
      const int64_t max_runtime_ms)
      : groups_(groups),
        preserve_chains_(preserve_chains),
        max_runtime_ms_(max_runtime_ms) {
    best_run_order_.reserve(groups.size());
    run_order_.reserve(groups.size());

    computeGroupIdMapping();

    computeInDegrees();

    computeTensorSizes();
  }

  void measureOrdering() {
    NVF_ERROR(
        run_order_.size() == groups_.size(),
        "Cannot measure incomplete ordering. Expected ",
        groups_.size(),
        " groups in ordering but found ",
        run_order_.size(),
        ". run_order: ",
        run_order_);
    std::cout << " Checking memory of run order ";
    for (SegmentedGroup* group : run_order_) {
      std::cout << " " << group_id_map_.at(group);
    }
    std::cout << std::endl;
    int64_t peak_memory = computePeakMemory(run_order_, tensor_size_);
    if (best_peak_memory_ == -1 || peak_memory < best_peak_memory_) {
      std::cout << " Found new best run order with peak memory " << peak_memory
                << std::endl;
      best_peak_memory_ = peak_memory;
      best_run_order_ = run_order_;
    }
    if (peak_memory > worst_peak_memory_) {
      worst_peak_memory_ = peak_memory;
    }
    num_checked_++;
  }

  SegmentedGroup* selectNextGroup() {
    std::cout << "selectNextGroup" << std::endl;
    // Get next available group.
    NVF_ERROR(!current_coords_.empty());
    std::cout << "  current_coords_.size()=" << current_coords_.size()
              << std::endl;
    std::cout << "  current_coords_= " << current_coords_ << std::endl;
    std::cout << "  run_order_=";
    for (SegmentedGroup* group : run_order_) {
      std::cout << " " << group_id_map_.at(group);
    }
    std::cout << std::endl;
    int64_t current_index = current_coords_.back();

    NVF_ERROR(current_index <= available_groups_.size());

    NVF_ERROR(current_index < available_groups_.size());
    SegmentedGroup* next_group = available_groups_.at((size_t)current_index);
    // Remove this group from the list of available groups
    available_groups_.erase(available_groups_.begin() + (ssize_t)current_index);

    std::cout << "  current_index=" << current_index << std::endl;
    std::cout << "  next_group=" << (void*)next_group << " = "
              << group_id_map_.at(next_group) << std::endl;
    std::cout << "  available_groups_.size()=" << available_groups_.size()
              << std::endl;

    // Update in-degrees of all neighbors as if we deleted next_group
    for (const SegmentedEdge* edge : next_group->consumer_edges) {
      SegmentedGroup* neighbor_group = edge->to;
      int64_t& deg = in_degree_[group_id_map_[neighbor_group]];
      if (--deg == 0) {
        std::cout << "  Pushing group " << (void*)neighbor_group << std::endl;
        available_groups_.push_back(neighbor_group);
      }
    }

    run_order_.push_back(next_group);
    current_coords_.push_back(0);

    return next_group;
  }

  // This undoes the steps of selectNextGroup
  void undoMostRecentSelection() {
    std::cout << "undoMostRecentSelection" << std::endl;
    NVF_ERROR(!run_order_.empty());
    SegmentedGroup* group = run_order_.back();
    std::cout << "  run_order_=";
    for (SegmentedGroup* group : run_order_) {
      std::cout << " " << group_id_map_.at(group);
    }
    std::cout << std::endl;
    run_order_.pop_back();
    NVF_ERROR(!current_coords_.empty());
    int64_t current_index = current_coords_.back();

    std::cout << "  group=" << (void*)group << " = " << group_id_map_.at(group)
              << std::endl;
    std::cout << "  current_index=" << current_index << std::endl;
    std::cout << "  current_coords_= " << current_coords_ << std::endl;

    // Update in-degrees of all neighbors to undo deletion of group
    for (int64_t i = (int64_t)group->consumer_edges.size() - 1; i >= 0; --i) {
      SegmentedEdge* edge = group->consumer_edges[(size_t)i];
      SegmentedGroup* neighbor_group = edge->to;
      int64_t& deg = in_degree_[group_id_map_[neighbor_group]];
      if (deg++ == 0) {
        NVF_ERROR(!available_groups_.empty());
        NVF_ERROR(
            neighbor_group == available_groups_.back(),
            "Found inconsistency while backtracking");
        available_groups_.pop_back();
      }
    }
    available_groups_.insert(
        available_groups_.begin() + (ssize_t)current_index, group);
  }

  void backTrack() {
    while (!run_order_.empty()) {
      // Here we will undo the deletion of "most_recent_group". This means
      // we need to update in-degrees of consumer groups, remove those that
      // are no longer available, then insert the most recent group it back
      // into available_groups at the specified position
      undoMostRecentSelection();

      // Now increment the last coordinate. If this causes us to move beyond
      // the last element of available_groups then we are done processing
      // this level's subtree, so remove this coordinate and loop to move up
      // another level in the tree.
      int64_t& most_recent_index = current_coords_.back();
      NVF_ERROR(most_recent_index < available_groups_.size());
      if (++most_recent_index < available_groups_.size()) {
        break;
      }
      // If most_recent_index exhausts available_groups_, pop this coord and
      // continue backtracking
      current_coords_.pop_back();
    }
  }

  void measureAllOrderings() {
    NVF_ERROR(
        !preserve_chains_, "Chain-preserving traversal is not yet implemented");

    complete_ = false;
    start_time_ = std::chrono::high_resolution_clock::now();

    // Fill available groups
    for (size_t group_id : c10::irange(groups_.size())) {
      if (in_degree_[group_id] == 0) {
        available_groups_.push_back(groups_[group_id]);
      }
    }

    run_order_.clear();
    current_coords_.clear();
    current_coords_.push_back(0);

    while (!available_groups_.empty()) {
      selectNextGroup();

      if (available_groups_.empty()) {
        // We've reached a leaf node in the DFS tree, meaning we have completed
        // an ordering
        measureOrdering();

        // Now backtrack to the most recent point at which we still have choices
        // left to make
        backTrack();

        if (run_order_.empty()) {
          // We've reached the end of the DFS traversal if we backtrack all the
          // way out
          break;
        }
      }

      // Return early if we have reached the time limit
      int64_t elapsed =
          std::chrono::duration_cast<std::chrono::milliseconds>(
              std::chrono::high_resolution_clock::now() - start_time_)
              .count();
      if (elapsed >= max_runtime_ms_ && best_peak_memory_ != -1) {
        return;
      }
    }

    complete_ = true;
  }

  // Record the size of each TensorView. Symbolic sizes will be assumed to be 2
  void computeTensorSizes() {
    ExpressionEvaluator expr_eval;
    const auto recordTvSize = [&](TensorView* tv) {
      if (tensor_size_.find(tv) != tensor_size_.end()) {
        return;
      }
      int64_t size = 1;
      for (IterDomain* id : tv->getMaybeAllocationDomain()) {
        if (id->isBroadcast() || id->isReduction()) {
          continue;
        }
        PolymorphicValue extent = expr_eval.evaluate(id->extent());
        // If we can't evaluate the extent, just assume it's 2. This is the
        // smallest non-zero size for an Iteration axis.
        size *= extent.hasValue() ? extent.as<int64_t>() : 2;
      }
      size *= dataTypeSize(tv->dtype());
      tensor_size_[tv] = size;
    };
    for (SegmentedGroup* group : groups_) {
      for (Val* val : group->inputs()) {
        if (auto* tv = dynamic_cast<TensorView*>(val)) {
          recordTvSize(tv);
        }
      }
      for (Val* val : group->outputs()) {
        if (auto* tv = dynamic_cast<TensorView*>(val)) {
          recordTvSize(tv);
        }
      }
    }
  }

  void computeGroupIdMapping() {
    for (size_t group_id : c10::irange(groups_.size())) {
      group_id_map_[groups_[group_id]] = group_id;
    }
  }

  void computeInDegrees() {
    in_degree_.resize(groups_.size(), 0ll);
    for (const SegmentedGroup* group : groups_) {
      for (const SegmentedEdge* edge : group->consumer_edges) {
        in_degree_[group_id_map_[edge->to]]++;
      }
    }
  }

 private:
  const std::vector<SegmentedGroup*>& groups_;
  bool preserve_chains_;
  const int64_t max_runtime_ms_;

  std::unordered_map<TensorView*, int64_t> tensor_size_;
  std::unordered_set<Val*> available_input_;

  std::chrono::time_point<std::chrono::high_resolution_clock> start_time_;

  std::unordered_map<SegmentedGroup*, size_t> group_id_map_;

  // This tracks in-degree of each group after the nodes in the currently
  // selected runtime order have been removed.
  std::vector<int64_t> in_degree_;

  std::vector<SegmentedGroup*> available_groups_;

  std::vector<SegmentedGroup*> run_order_;
  std::vector<int64_t> current_coords_;

  std::vector<SegmentedGroup*> best_run_order_;
  int64_t best_peak_memory_ = -1;
  int64_t worst_peak_memory_ = -1;

  bool complete_ = false;

  int64_t num_checked_ = 0;
};

} // namespace

void prepareRuntimeOrder(
    SegmentedFusion* segmented_fusion,
    RuntimeWorkSpace& runtime_workspace) {
  FUSER_PERF_SCOPE("prepareRuntimeOrder");
  // setup the order tensor dimensions are bound
  for (const size_t i : c10::irange(segmented_fusion->inputs().size())) {
    auto input_val = segmented_fusion->inputs()[i];

    if (auto input_tv = dynamic_cast<TensorView*>(input_val)) {
      auto logical_dom =
          TensorDomain::noReductions(input_tv->getLogicalDomain());
      for (const size_t dim : c10::irange(logical_dom.size())) {
        const auto extent = logical_dom[dim]->getMaybeExpandedExtent();
        runtime_workspace.group_extent_binding_order.push_back(extent);
      }
    }
  }

  // runtime_workspace.group_run_order = naiveRuntimeOrder(segmented_fusion);
  int64_t search_peak_memory_bytes;
  bool search_completed;
  std::tie(
      runtime_workspace.group_run_order,
      search_peak_memory_bytes,
      search_completed) =
      BruteForceRuntimeOrderOptimizer::run(
          segmented_fusion, /*preserve_chains=*/false);
  if (!search_completed) {
    // TODO: use some heuristic methods to try and find a better ordering and
    // compare it to the one found in the incomplete brute force search
    TORCH_WARN(
        "Brute-force search was incomplete. Peak memory use might be sub-optimal")
  }
}

flatbuffers::Offset<serde::InputsIdLookup> InputsIdLookup::serialize(
    flatbuffers::FlatBufferBuilder& builder) const {
  // See definitions in serde/fusion_cache.fbs for table
  // InputsIdLookup and struct EncodingEntry

  using fb_string = flatbuffers::Offset<flatbuffers::String>;

  // For serialization, we require a consistent ordering for the
  // encoding_lookup_ map.
  std::unordered_map<std::string, size_t> str_key_ordering;

  // 1. Serialize used_entry_ list
  std::vector<fb_string> lru_cache_fb;
  for (const auto& str : used_entry_) {
    lru_cache_fb.push_back(builder.CreateString(str));
    str_key_ordering.emplace(str, str_key_ordering.size());
  }

  // 2. Serialize encoding_lookup_ map
  std::vector<fb_string> encoding_lookup_keys_fb;
  std::vector<serde::EncodingEntry> encoding_lookup_values_fb;
  for (auto&& [key, value] : encoding_lookup_) {
    encoding_lookup_keys_fb.push_back(builder.CreateString(key));
    encoding_lookup_values_fb.emplace_back(value.id, str_key_ordering.at(key));
  }

  return serde::CreateInputsIdLookupDirect(
      builder,
      max_cache_size_,
      current_id_,
      &lru_cache_fb,
      &encoding_lookup_keys_fb,
      &encoding_lookup_values_fb);
}

void InputsIdLookup::deserialize(const serde::InputsIdLookup* buffer) {
  // See definitions in serde/fusion_cache.fbs for tables
  // InputsIdLookup and EncodingEntry
  NVF_ERROR(buffer != nullptr, "serde::InputsIdLookup is nullptr.");
  using list_iter = std::list<std::string>::iterator;
  std::vector<list_iter> used_entry_iterators;

  max_cache_size_ = buffer->max_cache_size();
  current_id_ = buffer->current_id();
  for (auto fb_str : *buffer->lru_cache()) {
    used_entry_.emplace_back(fb_str->str());
    used_entry_iterators.emplace_back(std::prev(used_entry_.end()));
  }

  for (auto idx : c10::irange(buffer->encoding_lookup_keys()->size())) {
    auto fb_encoding_lookup_str = buffer->encoding_lookup_keys()->Get(idx);
    auto fb_encoding_entry = buffer->encoding_lookup_values()->Get(idx);

    EncodingEntry entry{
        fb_encoding_entry->id(),
        used_entry_iterators.at(fb_encoding_entry->lru_iter())};
    encoding_lookup_.emplace(fb_encoding_lookup_str->str(), entry);
  }
}

InputsIdLookup::IdLookupReturn InputsIdLookup::lookupId(
    const at::ArrayRef<c10::IValue>& inputs,
    const std::unordered_set<size_t>& scalar_inputs_to_record,
    int8_t device) {
  IdLookupReturn ret;

  // lock mutex_ because we are touching encoding_
  std::lock_guard<std::mutex> guard(mutex_);
  encoding_.clear();
  encodeBuffer(device, encoding_);
  for (const auto i : c10::irange(inputs.size())) {
    auto input = inputs[i];
    if (input.isTensor()) {
      auto& input_tensor = input.toTensor();

      for (auto size : input_tensor.sizes()) {
        encodeBuffer(size, encoding_);
        encoding_.push_back(' ');
      }
      encoding_.push_back('X');
      encoding_.push_back(' ');
      for (auto stride : input_tensor.strides()) {
        encodeBuffer(stride, encoding_);
        encoding_.push_back(' ');
      }
      encoding_.push_back('a');
      encodeBuffer(
          SchedulerRuntimeInfo::computeAlignmentSize(
              (size_t)input_tensor.data_ptr()),
          encoding_);
      // NOTE: device is set for the whole set of inputs first using device arg
    } else {
      // encode s for scalar;
      encoding_.push_back('s');
      if (scalar_inputs_to_record.find(i) != scalar_inputs_to_record.end()) {
        // Add value of scalars here only if it is one of the scalars
        // provided, as these are used in determining concretization.
        // Note that although most commonly these will be Int or Bool scalars,
        // any DataType might appear via `cast` and `where`, so we handle all
        // cases here.
        if (input.isInt()) {
          encodeBuffer(input.toInt(), encoding_);
        } else if (input.isBool()) {
          encodeBuffer(input.toBool(), encoding_);
        } else if (input.isDouble()) {
          encodeBuffer(input.toDouble(), encoding_);
        } else if (input.isComplexDouble()) {
          encodeBuffer(input.toComplexDouble(), encoding_);
        } else {
          NVF_THROW(
              "Unhandled input type when creating input ID. Cannot record ",
              input);
        }
      }
    }
    encoding_.push_back(';');
  }

  auto& entry = encoding_lookup_[encoding_];

  if (entry.id == 0) {
    // no entry existed for given input set, set id for given entry
    entry.id = current_id_++;
    if (used_entry_.size() == max_cache_size_) {
      // pop least recently used cache;
      const auto& remove_iter = encoding_lookup_.find(used_entry_.back());
      used_entry_.pop_back();
      ret.evict_id = remove_iter->second.id;
      ret.eviction = true;
      encoding_lookup_.erase(remove_iter);
    }
  } else {
    // short-cut to leave LRU entry as is
    if (entry.lru_iter == used_entry_.begin()) {
      ret.id = entry.id;
      return ret;
    }

    used_entry_.erase(entry.lru_iter);
  }

  ret.id = entry.id;
  entry.lru_iter = used_entry_.insert(used_entry_.begin(), encoding_);
  return ret;
}

} // namespace nvfuser
