// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#pragma once

#include <debug.h>
#include <exceptions.h>
#include <fusion.h>
#include <ir/base_nodes.h>
#include <options.h>
#include <scheduler/all_schedulers.h>
#include <scheduler/registry.h>
#include <scheduler/runtime_info.h>
#include <utils.h>
#include <visibility.h>

#include <deque>
#include <functional>
#include <list>
#include <unordered_set>
#include <vector>

namespace nvfuser {

class SegmentedGroup;
class SegmentCandidateFinder;

// A directed edge on DAG,
// Wrapper for values, edges between segmented groups which are made up
// of Exprs. Multiple edges can exist between segmented groups.
struct SegmentedEdge {
  SegmentedEdge(SegmentedGroup* from, SegmentedGroup* to, Val* val)
      : from(from), to(to), val(val) {}

  SegmentedGroup* from;
  SegmentedGroup* to;
  Val* val;

  void print() const;

  bool operator==(const SegmentedEdge& other) const {
    return from == other.from && to == other.to && val == other.val;
  }

  bool operator!=(const SegmentedEdge& other) const {
    return !(*this == other);
  }
};

std::ostream& operator<<(std::ostream& os, const SegmentedEdge* edge);

//! Groups together expressions which create a segmented group
//! Can be used to produce fusions
class SegmentedGroup {
 public:
  //! Utility struct to represent a group connection
  //!  both the group to connect with and the edge
  //!  to connect through
  struct NeighborGroup {
    NeighborGroup(SegmentedGroup* g, SegmentedEdge* e) : group(g), edge(e) {}
    SegmentedGroup* group;
    SegmentedEdge* edge;
  };

  SegmentedGroup(SegmentedFusion* segmented_fusion)
      : segmented_fusion_(segmented_fusion) {}

  SegmentedGroup(Expr* expr, SegmentedFusion* segmented_fusion)
      : segmented_fusion_(segmented_fusion) {
    exprs_.push_back(expr);
  }

  //! Serialize SegmentedGroup using flatbuffers
  flatbuffers::Offset<serde::SegmentedGroup> serialize(
      flatbuffers::FlatBufferBuilder& builder,
      const std::unordered_map<Val*, int64_t>& vals_map,
      const std::unordered_map<Expr*, int64_t>& exprs_map,
      const std::unordered_map<SegmentedGroup*, int64_t>& groups_map,
      const std::unordered_map<SegmentedEdge*, int64_t>& edges_map) const;

  //! Deserialize SegmentedGroup using flatbuffers
  void deserialize(
      const serde::SegmentedGroup* buffer,
      const std::deque<Val*>& vals,
      const std::deque<Expr*>& exprs,
      const std::vector<SegmentedGroup*>& groups,
      const std::vector<SegmentedEdge*>& edges);

  //! returns the id assigned by segment pass
  int groupId() const {
    return group_id_;
  }

  //! Returns inputs that this group shares with the original fusion
  const auto& inputs() const {
    return input_vals_.vector();
  }

  //! Returns outputs that this group shares with the original fusion
  const auto& outputs() const {
    return output_vals_.vector();
  }

  //! Returns the schedule heuristic associated with this group
  SchedulerType schedulerType() const {
    return scheduler_type_;
  }

  //! Returns the exprs that make up this group
  const std::vector<Expr*>& exprs() const {
    return exprs_;
  }

  // Returns a toposorted list of Exprs in this group, with equal cases
  // respecting the original order.
  std::vector<Expr*> stablyOrderedExprs() const;

  //! Returns the complete fusion inputs mapped to this segmented group's fusion
  const auto& getCompleteFusionInputs() const {
    return original_inputs_in_cloned_fusion_;
  }

  //! Returns cloned fusion for this segmented group.
  //! TODO Replace read-only uses of makeFusion with cached getFusion
  Fusion* getFusion() {
    // Build cloned fusion for this segmented group
    if (cloned_fusion_ == nullptr) {
      makeClonedFusion();
    }
    return cloned_fusion_.get();
  }

  //! Debug print function
  void print() const;

  //! Utility to re-collect the operators included in this
  //!  segmented group after updating the group boundary.
  void resetExprList();

  //! Try to get a scheduler entry for this group with
  //!  the given runtime info.
  //! Returns a new scheduler with the same heuristics
  //!  for this group if possible.
  //!  Note that the schedule params can be different.
  //! Returns a nullopt if this group cannot be scheduled
  //!  with the same heuristics.
  std::optional<std::unique_ptr<HeuristicParams>> getMaybeHeuristicParams(
      SchedulerRuntimeInfo& runtime_info);

  //! Get the SegmentedFusion this group belongs to
  const SegmentedFusion* segmentedFusion() const {
    return segmented_fusion_;
  }

 public:
  //! "Ancestor nodes", towards inputs of segmentedDAG
  std::vector<SegmentedEdge*> producer_edges;

  //! "Descendent nodes", towards outputs of segmentedDAG
  std::vector<SegmentedEdge*> consumer_edges;

  //! Inputs of this group, they could be composite fusion inputs, or inputs
  //! from other groups
  VectorOfUniqueEntries<Val*> input_vals_;

  //! Outputs of this group, they could be composite fusion outputs, or outputs
  //! to other groups
  VectorOfUniqueEntries<Val*> output_vals_;

  bool isMerged() const {
    return merged_;
  }

  //! Look at all neighbors of this and return who this could merge with based
  //! on level values of this, neighbors, and merged neighbors of neighbors
  std::vector<NeighborGroup> getMergeCandidates();

 private:
  friend class SegmentCandidateFinder;
  friend class SegmentedFusion;
  friend class FusionKernelRuntime;
  friend class TranslateApplicableWelford;

  //! unique identifier of group in the segmented fusion
  int group_id_ = -1;

  //! The scheduler to use for compiling this group
  SchedulerType scheduler_type_ = SchedulerType::None;

  //! Exprs that make up the group
  std::vector<Expr*> exprs_;

  //! Maximum path distance from an input segmented group required for
  //! Theorem 4.2
  int level_ = -1;

  //! Did we select another group to merge with
  SegmentedGroup* merge_with_ = nullptr;

  //! if we selected another group to merge, which edge is to be contracted
  SegmentedEdge* merge_through_ = nullptr;

  //! Has this node been merged?
  bool merged_ = false;

 private:
  //! To be called at the very end of segment fusion
  //!  no more segment merging should be done beyond
  void finalize();

  //! Make the cloned fusion for this segmented group
  void makeClonedFusion();

  //! Return all segmented groups connected with *this
  std::vector<SegmentedGroup*> getNeighbors();

  //! TODO: May want to sort this based on size of connections between this and
  //! neighbors as well as if the connection is an output of the fusion (has to
  //! be saved to gmem anyways)
  std::vector<NeighborGroup> getNeighborGroups();

  //! Assign scheduler type to this group
  void setSchedulerType(SchedulerType scheduler_type) {
    scheduler_type_ = scheduler_type;
  }

  //! Assign Id for this group
  void setID(int id) {
    NVF_ERROR(group_id_ == -1);
    group_id_ = id;
  }

  //! SegmentedFusion this group belongs to
  SegmentedFusion* segmented_fusion_;

  //! The cloned segmented fusion
  std::unique_ptr<Fusion> cloned_fusion_;

  //! These are the complete fusion's inputs mapped to the cloned fusion
  std::vector<Val*> original_inputs_in_cloned_fusion_;
};

std::ostream& operator<<(std::ostream& os, const SegmentedGroup* group);

//! Exported Interface for representing segmented fusion graph
//!   this class owns the segmented groups
class SegmentedFusion {
 public:
  explicit SegmentedFusion(std::unique_ptr<Fusion> fusion);

  //! Factory function for the un-segmented case, directly
  //!  constructs a "SegmentedFusion", with the given Fusion
  //!  as the only group.
  static std::unique_ptr<SegmentedFusion> fromCompleteFusion(
      std::unique_ptr<Fusion> fusion,
      SchedulerType scheduler_type,
      const KernelArgumentHolder& runtime_inputs);

  //! Is the fusion segmented?
  bool isSegmented() const {
    return !groups_.empty();
  }

  std::vector<SegmentedGroup*>& groups() {
    return groups_;
  }

  const std::vector<SegmentedGroup*>& groups() const {
    return groups_;
  }

  std::vector<SegmentedEdge*>& edges() {
    return edges_;
  }

  const std::vector<SegmentedGroup*>& cgroups() const {
    return groups_;
  }

  const std::vector<SegmentedEdge*>& cedges() const {
    return edges_;
  }

  //! Returns the original un-segmented fusion
  Fusion* completeFusion() const {
    return complete_fusion_.get();
  }

  const auto& inputs() const {
    return complete_fusion_->inputs();
  }

  const auto& outputs() const {
    return complete_fusion_->outputs();
  }

  //! Get the fusion for the segmented group and return the IrCloner used to
  //! clone the complete fusion
  std::pair<IrCloner, std::unique_ptr<Fusion>> makeFusion(
      SegmentedGroup* sg) const;

  //! Make a heuristics entry for a group and parameters
  std::unique_ptr<HeuristicParams> makeInitialHeuristicParams(
      SegmentedGroup* sg,
      SchedulerRuntimeInfo& runtime_info);

  //! Debug drawing for graphviz
  void draw();

  //! Debug print for segmented fusions
  void print() const;

  //! API for adding groups
  SegmentedGroup* newGroup();

  //! API shortcut for adding a singleton group
  SegmentedGroup* newGroup(Expr* expr);

  //! API shortcut for adding a new group for a fusion input
  SegmentedGroup* newFusionInputGroup();

  //! API for adding edges
  SegmentedEdge* newEdge(SegmentedGroup* from, SegmentedGroup* to, Val* val);

  //! Remove an edge from the segmented fusion graph and update all affected
  //! groups The edge object will be deleted and should not be used after this
  //! call
  void removeEdge(SegmentedEdge* edge);

  void connectGroups(SegmentedGroup* from, SegmentedGroup* to, Val* val);

  HeuristicDataCache* getCachedHeuristicDataFor(SegmentedGroup* group);

  //! Lower FP precision of inputs and outputs specified by the given
  //! edges.
  //!
  //! This function is used in two scenarios. One is when testing a
  //! merge of groups during the segmentation time. At that time,
  //! those groups are not yet merged, but we want to consider them as
  //! merged and see if there's a valid scheduler. So, we treat the
  //! groups given by groups_to_merge as a single group and insert
  //! cast ops into the group. No other group is modified unless it
  //! has an edge to any of the merged groups.
  //!
  //! The second scenario is when inserting cast ops to a whole
  //! segmented fusion. All groups are considered separate groups with
  //! no (temporary) merging. Each edge is considered a potential
  //! place to insert cast. In this case, groups_to_merge should be
  //! empty.
  std::vector<SegmentedEdge*> castInputOutputToLowerPrecision(
      const std::vector<SegmentedEdge*>& edges,
      const std::vector<SegmentedGroup*>& groups_to_merge = {});

  //! Revert the changes made by castInputOutputToLowerPrecision to the given
  //! edges
  void revertInputOutputPrecisionChanges(
      const std::vector<SegmentedEdge*>& edges);

  //! Grab edges with val
  std::vector<SegmentedEdge*> getEdgesByVal(Val* val) const;

  //! Get edges between two groups
  std::vector<SegmentedEdge*> getEdgesBetween(
      const SegmentedGroup* from,
      const SegmentedGroup* to) const;

  //! Serialize SegmentedFusion using flatbuffers
  flatbuffers::Offset<serde::SegmentedFusion> serialize(
      flatbuffers::FlatBufferBuilder& builder) const;

  //! Deserialize SegmentedFusion using flatbuffers
  void deserialize(const serde::SegmentedFusion* buffer);

  void validateDisjoint() const;

 private:
  //! Serialize SegmentedEdge using flatbuffers
  flatbuffers::Offset<serde::SegmentedEdge> serialize(
      flatbuffers::FlatBufferBuilder& builder,
      const nvfuser::SegmentedEdge* edge,
      const std::unordered_map<Val*, int64_t>& vals_map,
      const std::unordered_map<SegmentedGroup*, int64_t>& groups_map) const;

  //! Deserialize SegmentedEdge using flatbuffers
  nvfuser::SegmentedEdge deserialize(
      const serde::SegmentedEdge* buffer,
      const std::deque<Val*>& vals);

 private:
  //! Unique name for segmented fusion
  size_t segmented_fusion_name_;

  //! States representing segmentation
  std::vector<SegmentedEdge*> edges_;
  std::vector<SegmentedGroup*> groups_;

  //! Owning object to explicitly manage groups and edges
  class Impl {
   public:
    explicit Impl(SegmentedFusion* sf) : owning_fusion_(sf) {}

    SegmentedGroup* makeGroup();
    SegmentedGroup* makeGroup(Expr*);
    SegmentedEdge* makeEdge(SegmentedGroup* from, SegmentedGroup* to, Val* val);
    void cleanUnused();
    std::unordered_map<SegmentedGroup*, int64_t> groups_map() const;
    std::unordered_map<SegmentedEdge*, int64_t> edges_map() const;

   private:
    using GroupPtr = std::unique_ptr<SegmentedGroup>;
    using EdgePtr = std::unique_ptr<SegmentedEdge>;
    std::vector<GroupPtr> groups_;
    std::vector<EdgePtr> edges_;
    SegmentedFusion* owning_fusion_;
  };
  Impl impl_;

  //! A Copy of original full fusion
  std::unique_ptr<Fusion> complete_fusion_;

  //! A set of intermediate tensors that need to be cast to fp16
  std::unordered_set<TensorView*> force_fp16_tv_set_;

  DataType force_half_precision_type_;

  //! Static traversal information to be used for fast heuristics lookup
  std::unordered_map<SegmentedGroup*, std::unique_ptr<HeuristicDataCache>>
      heuristic_data_cache_;

  //! The number of values in fusion after constructing segmented fusion.
  //! Used for checking state during deserialization.
  size_t initial_vals_size_;

  //! The number of expressions in fusion after constructing segmented fusion.
  //! Used for checking state during deserialization.
  size_t initial_exprs_size_;

  // TODO: this class needs cleanup
 protected:
  friend class SegmentCandidateFinder;

  //! Cleanup function to be call at the end of fusion
  //!  segment pass
  void finalize();

  //! Collect all the intermediate tensors between segmented
  //!  groups that will cast to fp16
  void annotateFP16IntermediateTensors();

  //! Keep heuristic checking intermediate data
  void setCachedHeuristicDataFor(
      SegmentedGroup* group,
      std::unique_ptr<HeuristicDataCache> data);

  //! Utility to give unique name for each segmented fusion
  static size_t segmentedFusionName() {
    static size_t counter = 0;
    return counter++;
  }
};

std::ostream& operator<<(
    std::ostream& os,
    const SegmentedFusion* segmented_fusion);

//! This is a base class for segmenter analysis
//!  provides the minimal implementation on header so that
//!  a unique_ptr can use this base class
//!  actual implementations of analyses are in the .cpp files
//! TODO: In the next refactor PR, should put segment candidate
//!  finder in .cpp file completely since API doesn't require these
//!  details
class SegmenterAnalysis : public PolymorphicBase {};
class GroupDependencyAnalysis;

// Manual node merging passes
class CombineReductions;
class MergeUpAndDownCast;

//! Options to configure/debug candidate finder
struct SegmentCandidateFinderOptions {
  bool run_translate_welford = true;
  bool run_combine_reductions = true;
  bool run_herrmann_merge = true;
  bool run_final_merge = true;
  // if provided, this custom function will be used to determine if two groups
  // should be merged. If not provided, the tryMerge function will be used. This
  // option is used in the context of MultiGpus where we proceed to a first
  // segmentation to scoop out communications from compute.
  std::function<bool(SegmentedGroup*, SegmentedGroup*)>
      custom_should_merge_groups = nullptr;
};

//!  SegmentCandidateFinder
//!    Responsible for going through DAG and proposing things we could try to
//!    fuse together, calls "canGenerateCode" on these proposed segments to see
//!    if they are valid and we can generate code for them.
//!  FusionSegment
//!    A group of exprs that are segmented together
//!  FusionSegmentConnections
//!    Holds vals and what they connect. In other words it's a val that is an
//!    output of a FusionSegment "from" and an input of FusionSegment "to".
//!    There's nothing preventing from a val being between segments twice.
//!    TODO: make sure there's nothing wrong with segmentation on nodes that
//!    have the same value input twice. i.e. (B = A*A)
//! Selecting segments to propose is based on the theorem 4.2 in the paper which
//! makes sure when segment the segmented graph will be a DAG (assumes Fusion is
//! already a DAG). The segmentation code relies on assumptions of DAG-ness
//! during segmentation, meaning proposed merging of groups must maintain the
//! DAG property of the graph.
//!
//! Julien Herrmann, Yusuf Özkaya, Bora Uçar, Kamer Kaya, Umit Catalyurek.
//! Multilevel Algorithms for Acyclic Partitioning of Directed Acyclic Graphs.
//! SIAM Journal on Scientific Computing, Society for Industrial and Applied
//! Mathematics, 2019, 41 (4), pp.A2117-A2145. ff10.1137/18M1176865ff.
//! ffhal02306566f
class SegmentCandidateFinder {
 public:
  // Perform segmentation on a copy of the given fusion
  static std::unique_ptr<SegmentedFusion> segment(
      const Fusion* fusion,
      const KernelArgumentHolder& inputs,
      SegmentCandidateFinderOptions options = SegmentCandidateFinderOptions());

  // Perform segmentation on and take ownership of the given fusion
  static std::unique_ptr<SegmentedFusion> segment(
      std::unique_ptr<Fusion> fusion,
      const KernelArgumentHolder& inputs,
      SegmentCandidateFinderOptions options = SegmentCandidateFinderOptions(),
      bool multi_device = false);

  static std::unique_ptr<SegmentedFusion> segment(
      std::unique_ptr<Fusion> fusion,
      const KernelArgumentHolder& inputs,
      SchedulerRuntimeInfo& runtime_info);

  static bool hasSegmentHints(Fusion* fusion);

  NVF_API static bool translateWelfordInFusion(
      Fusion* fusion,
      const KernelArgumentHolder& runtime_inputs);

  //! Validate the graph is a DAG, and if require_disjoint that exprs are
  //! disjoint
  void validateIfDebug(bool require_disjoint = false);

 private:
  // Perform segmentation on and take ownership of the given fusion
  NVF_API SegmentCandidateFinder(
      std::unique_ptr<Fusion> fusion,
      const KernelArgumentHolder& inputs,
      SegmentCandidateFinderOptions options,
      bool multi_device = false);

  void resetLevels();

  SegmentedGroup* mergeNodes();

  bool codeGenSupportedMerge(SegmentedGroup* group1, SegmentedGroup* group2);

  void buildInitialSegments();

  // Replicate upcast ops when consumed by multiple expressions. This
  // promotes segmented fusions to share pre-upcast tensors rather
  // than post-upcast tensors. Replicated upcast ops will be reverted
  // when they are grouped into the same segment. See
  // https://github.com/NVIDIA/Fuser/pull/3776/ for more details.
  void privatizeUpcast();

  void findSegments();

  // Revert privatized upcast ops when not necessary
  void revertPrivatizedUpcast(SegmentedGroup* group);

  //! Find a group found in candidates that can be merged with the
  //! given group and set them to be merged if found. When no
  //! candidate is given, SegmentedGroup::getMergeCandidates is used
  //! to get candidates.
  void trySetUpMerge(
      SegmentedGroup* group,
      std::vector<SegmentedGroup::NeighborGroup> candidates = {});

  void disconnectGroup(SegmentedGroup* group);

  std::vector<SegmentedGroup*>& groups() {
    NVF_ERROR(
        segmented_fusion_ != nullptr, "Segment finder not owinging any fusion");
    return segmented_fusion_->groups();
  }

  std::vector<SegmentedEdge*>& edges() {
    NVF_ERROR(
        segmented_fusion_ != nullptr, "Segment finder not owinging any fusion");
    return segmented_fusion_->edges();
  }

  Fusion* completeFusion() {
    NVF_ERROR(
        segmented_fusion_ != nullptr, "Segment finder not owinging any fusion");
    return segmented_fusion_->completeFusion();
  }

  SchedulerRuntimeInfo& runtimeInfo();

  ExpressionEvaluator& expressionEvaluator() {
    return runtimeInfo().expressionEvaluator();
  }

  //! Additional merging iteration, clean up the rest of
  //!  the merging opportunities
  //!  Herrmann et al. is a fast and safe algorithm for finding merge candidates
  //!  but can become too conservative in our use cases because we place
  //!  additional qualifiers on valid merges other than having to generate DAGs,
  //!  i.e. canSchedule. So we need a bruteforce final merging iteration as a
  //!  clean up pass. Cost isn't expected to be high since the graph at this
  //!  stage is already quite merged. Example cf. test_gpu.cpp:
  //!  FusionDAGMerging_CUDA
  //!
  //!  This merging algorithm is based on Theorem 4.1 of Herrmann et al.,
  //!   to check if a producer-consumer pair can be merged into one group,
  //!   it's enough to check if any other consumer of the producer also
  //!   produces the consumer.
  void finalMerge();

  //! Duplicate and add all exprs producing the used
  //!  scalar values in group
  void resolveScalarsInGroup(SegmentedGroup* group);

  //! Duplicate and add all exprs from fusion inputs to `forwarded_input` into
  //! the group, to complete inputs. These expressions are simply unary ops of
  //! inputs that we want to recompute for each segment, instead of computing
  //! and producing a segmented val. For example if we have:
  //!
  //!   tv1 = tv0 * 2;
  //!   tv3 = tv1 + tv2;
  //!   tv4 = tv1 + tv4
  //!
  //! If we segmented on tv1, we would be producing an output for tv1 for 2
  //! groups that have tv3 or tv4, instead we could easily recompute tv1 from
  //! tv0.
  void resolveNonscalarForwardedInput(Val* forwarded_input);

  void resolveForwardedInputs();

  // Creates the input group that ends at `forwarded_input`, i.e., the region
  // between fusion inputs and `forwarded_input`.
  SegmentedGroup* createInputGroup(Val* forwarded_input);

  //! Remove all scalar edges in group
  //!  (TODO: need structure better so we don't have to do this)
  void removeScalarEdges();

  //! Utility function to merge a vector of groups in one step,
  //!  need to check for DAG condition before using this method
  SegmentedGroup* mergeAllGivenGroups(
      const std::vector<SegmentedGroup*>& groups);

  //! Utility to remove a group and corresponding edges
  //!  TODO: remove inline versions of this as much as possible
  void eraseGroups(std::unordered_set<SegmentedGroup*>& groups_to_erase);

  void finalize();

  //! Return the resulting SchedulerType corresponding to the merged
  //!  group built by merging the two groups connected by edge
  SchedulerType deriveSchedulerType(SegmentedGroup* edge);

  GroupDependencyAnalysis* getGroupDependency();

  //! Find all expresions that are simply unary ops from
  //! inputs. Don't segment
  //! these as they're easy targets for recomputation. Only go until the first
  //! expression that has multiple uses.
  //!
  //! The ending tensors, or the forwarded tensors, are considered
  //! fusion inputs for the sake of segmentation, and the expressions
  //! between the real inputs and the forwarded tensors are excluded
  //! from the segmentation steps until the finalization, at which
  //! point they are simply prepended to each final segment using the
  //! forwarded inputs.
  void forwardInputs();

  void cleanupForwardedInputs();

  //! Query if a val is a fusion input or a forwarded input
  bool isFusionInput(Val* val) const {
    return std::find(
               forwarded_fusion_inputs_.begin(),
               forwarded_fusion_inputs_.end(),
               val) != forwarded_fusion_inputs_.end();
  };

  // Get all auxiliary groups created for fusion inputs
  std::vector<SegmentedGroup*> getAuxiliaryInputGroups() const;

 protected:
  //! These are the merge node heuristic passes, should
  //!  eventually should have a dedicated interface
  //!  instead of keeping adding friends
  friend class CombineReductions;
  friend class MergeUpAndDownCast;

  //! options to configure and debug the segment process
  SegmentCandidateFinderOptions options_;

  std::unordered_set<SegmentedGroup*> clean_up_groups_;

  std::vector<SegmentedGroup*> to_merge_;

  std::unique_ptr<SegmentedFusion> segmented_fusion_;

  std::unique_ptr<SegmenterAnalysis> group_dependency_;

  //! List of vals to treat as complete fusion inputs for segmentation
  std::vector<Val*> forwarded_fusion_inputs_;

  //! Keep track of complete fusion input use
  std::unordered_map<Val*, SegmentedGroup*> input2group_;

  // Expressions to exclude from segmentation because they're just derived from
  // unary ops on inputs to the complete fusion
  VectorOfUniqueEntries<Expr*> excluded_inp_unary_exprs_;

  // This is allowed to be null in the multidevice case where the segmenter is
  // used for breaking the fusion into compute and communication segments
  std::optional<SchedulerRuntimeInfo> runtime_info_;

  std::unordered_map<UnaryOp*, std::unordered_set<UnaryOp*>>
      privatized_upcast_ops_;

  //! Note:
  //!  Segmenter should eventually rely only on runtime_info_ for
  //!  safe caching. runtime_inputs_ is only used in translateWelford
  //!  to initialize expression evaluators on copies of the original
  //!  fusion, which doesn't use any un-cached info and is safe.
  //!
  //!  Directly using runtime_inputs_ in other cases is in general
  //!   risky.
  //!
  //!  To get rid of runtime_inputs_ we need mechanisms
  //!  to copy expression evaluator values from fusion
  //!  to a copy, or even better to a copy of a
  //!  sub-graph of original fusion.
  //! TODO:
  //!  implement the expression evaluator transfer and
  //!  remove runtime_inputs_ in a follow up.
  const KernelArgumentHolder runtime_inputs_;
};

// TODO: Make as member functions on classes instead of global scope
std::string toString(const SegmentedGroup* group);
std::string toString(const SegmentedEdge* edge);
std::string toString(const SegmentedFusion* segmented_fusion);
std::string toString(const SegmentCandidateFinderOptions& segment_options);

} // namespace nvfuser
