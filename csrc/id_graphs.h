#pragma once

#include <disjoint_set.h>
#include <ir_all_nodes.h>
#include <kernel_ir.h>
#include <lower_trivial_broadcast.h>

#include <deque>
#include <unordered_map>

namespace nvfuser {

using IdGroup = std::shared_ptr<VectorOfUniqueEntries<IterDomain*>>;
using IdGroups = VectorOfUniqueEntries<IdGroup>;
using ExprGroup = std::shared_ptr<VectorOfUniqueEntries<Expr*>>;
using ExprGroups = VectorOfUniqueEntries<ExprGroup>;

class TORCH_CUDA_CU_API IdGraph {
 public:
  IdGraph() = default;

  IdGraph(const IdGraph& other);
  IdGraph(IdGraph&& other) = default;

  IdGraph& operator=(const IdGraph& other);
  IdGraph& operator=(IdGraph&& other) = default;

  // Returns the disjoint IterDomain set.
  const DisjointSets<IterDomain*>& disjointIdSets() const;

  DisjointSets<IterDomain*>& disjointIdSets();

  // Returns
  //   {
  //     (1) The disjoint set of the provided Iter Domain if it exists,
  //     otherwise a null shared ptr
  //     (2) If the disjoint set of the provided Iter Domain exists
  //   }
  std::pair<IdGroup, bool> disjointIdSet(IterDomain* id) const;

  // Returns the disjoint Expr set.
  const DisjointSets<Expr*>& disjointExprSets() const;

  DisjointSets<Expr*>& disjointExprSets();

  // Same as getDisjointIdSet but for the Expression sets.
  std::pair<ExprGroup, bool> disjointExprSet(Expr* expr) const;

  // Convert expr to its exprGroup, assert that it exists.
  ExprGroup toGroup(Expr* expr) const;

  // Convert iter domain to its IdGroup, assert that it exists.
  IdGroup toGroup(IterDomain* id) const;

  // Convert unique vector of expressions to unique vector of its groups
  ExprGroups toGroups(const VectorOfUniqueEntries<Expr*>& exprs) const;

  // Convert unique vector of IterDomain to unique vector of its groups
  IdGroups toGroups(const VectorOfUniqueEntries<IterDomain*>& ids) const;

  // Return output/input iter domain groups of provided expr
  std::vector<IdGroup> outputGroups(ExprGroup expr) const;
  std::vector<IdGroup> inputGroups(ExprGroup expr) const;

  // Returns if for each group in id_groups0 is the same as all groups in
  // id_groups1. Requires size and order to be exact.
  bool groupsMatch(
      std::vector<IdGroup> id_groups0,
      std::vector<IdGroup> id_groups1) const;

  // Returns if for each group in expr_groups0 is the same as all groups in
  // expr_groups1. Requires size and order to be exact.
  bool groupsMatch(
      std::vector<ExprGroup> expr_groups0,
      std::vector<ExprGroup> expr_groups1) const;

  // Traverses uses of the IdGroups in 'of' and returns all ExprGroups
  // that have a use in their definition of provided of IdGroups.
  ExprGroups allUsesOf(const IdGroups& of) const;

  // Traverses definitions of the IdGroups in 'of' and returns all ExprGroups
  // used in this history of defining the 'of' IdGroups.
  ExprGroups allDefinitionsOf(const IdGroups& of) const;

  // Return sorted expressions to go from the provided IterDomains in from to
  // the provided IterDomains in to with provided mode. Minimal expressions to
  // get from 'from' to 'to' returned.
  ExprGroups getExprsBetween(const IdGroups& from, const IdGroups& to) const;

  // Supports one to many mappings, uses the disjoint sets of the provided mode
  // to produce mappings between from and to. If multiple IterDomains in to map
  // to a single iter domain in from, the order of the IterDomains in value of
  // the map is preserved to be the order provided in to.
  std::unordered_map<IterDomain*, VectorOfUniqueEntries<IterDomain*>>
  buildMapBetween(
      const std::vector<IterDomain*>& from,
      const std::vector<IterDomain*>& to) const;

  // Alias of the above on unique vector entries
  std::unordered_map<IterDomain*, VectorOfUniqueEntries<IterDomain*>>
  buildMapBetween(
      const VectorOfUniqueEntries<IterDomain*>& from,
      const VectorOfUniqueEntries<IterDomain*>& to) const;

  //! Returns
  //!   (1) The expressions associated with the definitions of the provided
  //!     IterDomain group in the provided mapping mode (if it exists).
  //!   (2) If there is a definitions entry of the provided IterDomain group in
  //!     the provided mapping mode.
  //! First entry in the returned pair is a vector of vector of expressions. The
  //! inner vector is proven to be equivalent based on the provided mode. The
  //! outer vector are expression groups that are not equivalent based on the
  //! provided mode, but produce one of the IterDomains within the same disjoint
  //! Iter Domain set based on the provided mode.
  //! TODO: Change name to start with get
  std::pair<ExprGroups, bool> iterDomainGroupDefinitions(
      IdGroup id_group) const;

  //! Same as iterDomainGroupDefinitions but for uses instead of definitions
  //! TODO: Change name to start with get
  std::pair<ExprGroups, bool> iterDomainGroupUses(IdGroup id_group) const;

  std::string toString() const;

  // Checks if the expression is a trivial operation where an input is simply an
  // output of the transformation. Returns the mapped iter domains if found.
  static std::vector<std::vector<IterDomain*>> isTrivialExpr(Expr* expr);

  // Initializes entries for the provided IterDomain in the IterDomainGraphs
  void initializeId(
      IterDomain* id,
      const VectorOfUniqueEntries<Expr*>& definitions,
      const VectorOfUniqueEntries<Expr*>& uses);

  // Returns if first and second are expressions through which the provided
  // id_map have matching inputs (if forward), or outputs (if not forward).
  // Returning true means the expressions are "the same", in terms they modify
  // matching original extents, by the same amount.
  bool exprsMap(
      Expr* first,
      Expr* second,
      bool forward
      // , std::vector<IterDomain*> second_input_or_output_override
  ) const;

  // Returns entry in unique_definitions_ for provided group in provided mode,
  // otherwise errors if no entry is found.
  ExprGroups uniqueDefinitions(IdGroup group) const;

  // Returns entry in unique_uses_ for provided group in provided mode,
  // otherwise errors if no entry is found.
  ExprGroups uniqueUses(IdGroup group) const;

  std::unordered_map<IdGroup, ExprGroups>& uniqueUses() {
    return unique_uses_;
  }

  std::unordered_map<IdGroup, ExprGroups>& uniqueDefinitions() {
    return unique_definitions_;
  }

  // Set id0 and id1 to mapped in disjointIdsSet[mode], attempt to propagate
  // new mapping through id0/id1 definitions/uses.
  void mapIds(IterDomain* id0, IterDomain* id1);

  // Checks if expr0 and expr1 should map together, maps them together, and if
  // expression propagation is on, propagates mapping through them. This should
  // be the only call in IdGraph to mapThroughExpr
  void maybeMapThroughExprs(Expr* expr0, Expr* expr1, bool forward);

  // Map expr0 and expr1 with eachother, update unique_definitions_ unique_uses_
  // TODO: Make this variant hidden?
  void mapExprs(Expr* expr0, Expr* expr1);

  // Checks if expr's are considered "the same" where sameness inputs and
  // outputs in the same position across expressions map with  provided
  // MappingMode. If the expressions are determined the same then
  // if forward
  //   will map outputs
  // else
  //   will map inputs
  // in the provided mode.
  // Returns if expressions were mapped through.
  //
  // TODO: Make this private
  bool mapThroughExpr(Expr* first, Expr* second, bool forward);

  // Map through loop swizzles, as input/output IterDomains are exact, only the
  // order they're traversed differs.
  void mapThroughLoopSwizzles();

  // Maps iter domain pairs returned by calling that return mappings from
  // IdGraph::isTrivialExpr on every expression in the graph.
  void mapThroughTrivialExprs();

  // Removes expressions from unique_definitions_ and unique_uses_ that return
  // mappings from IdGraph::isTrivialExpr
  void removeTrivialExprs();

  // See comment on propagate_expr_ member bool for description
  // Once disabled this can't be reenabled on a graph. If it's reenabled it's
  // hard to predict how mappings will propagate, which will be triggered on the
  // next mapping. To support changing this flag, we should likely run through
  // all expressions currently registered and propagate through all of them on
  // switch. Then once enabled it couldn't be redisabled because we don't record
  // the history of mapId calls.
  void disableExprPropagation() {
    propagate_exprs_ = false;
  }

 private:
  // Removes the provided expression group from unique_definitions_ and
  // unique_uses_ breaking traversal through them.
  void eraseExprGroup(ExprGroup expr_group);

  // If propagate_exprs_ = false, then mapThroughExpr will not be called as a
  // consequence of calling mapIds. As well as mapThroughExpr will not be called
  // (again) as a result of calling mapThroughExpr.
  //
  // Note: For the second sentence of above... mapThroughExpr can call mapIds
  // which could in return call mapThoughExpr again, but propagate_exprs_ as
  // mentioned above prevents that from happening.
  //
  // TODO: Should propagate_exprs_ be a const member?
  bool propagate_exprs_ = true;

  // Keeps a disjoint set entry for all IterDomain for all mapping mode types.
  //
  // Using an array here might be nice, but it seems hard to use an enum as an
  // array key
  // https://stackoverflow.com/questions/2102582/how-can-i-count-the-items-in-an-enum
  DisjointSets<IterDomain*> disjoint_ids_;

  // Keeps a disjoint set entry for all Expressions for all mapping mode types.
  DisjointSets<Expr*> disjoint_exprs_;

  std::unordered_map<IdGroup, ExprGroups> unique_definitions_;

  std::unordered_map<IdGroup, ExprGroups> unique_uses_;

  // Hold a set of IterDomains that are considered view rfactor ids. This
  // identification is particularly important to understand if split operations
  // are divisible or not.
  //
  // TODO: This should just be in IterDomainGraphs, not here.
  std::unordered_set<IterDomain*> view_rfactor_ids_;
};

// Debuging print functions
namespace debug {
std::string toString(
    const std::vector<IterDomain*>& id_group,
    int indent_size = 0);
std::string toString(
    const IdGroup& id_group,
    int indent_size = 0,
    bool with_ptr = false);

std::string toString(
    const std::vector<IdGroup>& id_groups,
    int indent_size = 0,
    bool with_ptr = false);

std::string toString(
    const IdGroups& id_groups,
    int indent_size = 0,
    bool with_ptr = false);

std::string toInlineString(const std::vector<IdGroup>& id_groups);
std::string toInlineString(const IdGroups& id_groups);

std::string toString(const std::vector<Expr*>& expr_group, int indent_size = 0);
std::string toString(
    const ExprGroup& expr_group,
    int indent_size = 0,
    bool with_ptr = false);

std::string toString(
    const IdGraph& id_graph,
    const std::vector<Expr*>& expr_group,
    int indent_size = 0,
    bool with_ptr = false);
std::string toString(
    const IdGraph& id_graph,
    const ExprGroup& expr_groups,
    int indent_size = 0,
    bool with_ptr = false);

std::string toString(
    const IdGraph& id_graph,
    const std::vector<ExprGroup>& expr_groups,
    int indent_size = 0,
    bool with_ptr = false);
std::string toString(
    const IdGraph& id_graph,
    const ExprGroups& expr_groups,
    int indent_size = 0,
    bool with_ptr = false);

std::string idGroupsString(
    const IdGraph& id_graph,
    int indent_size = 0,
    bool with_ptr = false);
std::string exprGroupsString(
    const IdGraph& id_graph,
    int indent_size = 0,
    bool with_ptr = false);
std::string definitionsString(
    const IdGraph& id_graph,
    int indent_size = 0,
    bool with_ptr = false);
std::string usesString(
    const IdGraph& id_graph,
    int indent_size = 0,
    bool with_ptr = false);
} // namespace debug

// Iterates through an IterDomain Graph in topological order, calling handle on
// all Id and all Expr groups in a forward topological order.
//
// Warning: Expr groups that have an input and output in the same IdGroup are
// ignored.
//
// Warning: This is not a great iterator if there's a desire to minimize paths
// traveled to simply visit all IdGroups in order. See ExprsBetween to see how
// we might minimize paths.
class TORCH_CUDA_CU_API IdGraphVisitor {
 protected:
  // If sub_selection is assumed to be a set of iter domains by which form a
  // sub-regrion of the IdGraph provided. Only that sub-region will be visited.
  IdGraphVisitor(
      IdGraph& id_graph,
      const VectorOfUniqueEntries<IterDomain*> sub_selection = {})
      : id_graph_(id_graph), sub_selection_(sub_selection) {}

  virtual void handle(IdGroup id_group) = 0;
  virtual void handle(ExprGroup expr_group) = 0;

  void traverse();

  IdGraph& graph() {
    return id_graph_;
  };

  IdGraphVisitor() = delete;

  IdGraphVisitor(const IdGraphVisitor& other) = default;
  IdGraphVisitor& operator=(const IdGraphVisitor& other) = delete;

  IdGraphVisitor(IdGraphVisitor&& other) = default;
  IdGraphVisitor& operator=(IdGraphVisitor&& other) = delete;

  virtual ~IdGraphVisitor() = default;

 private:
  IdGraph& id_graph_;
  const VectorOfUniqueEntries<IterDomain*> sub_selection_;
};

// Statement sorting based on IdGraphVisitor, see warnings to IdGraph Visitor.
class IdGraphStmtSort : public IdGraphVisitor {
 public:
  IdGraphStmtSort(
      IdGraph& id_graph,
      const VectorOfUniqueEntries<IterDomain*> sub_selection = {})
      : IdGraphVisitor(id_graph, sub_selection) {
    IdGraphVisitor::traverse();
  }

  ExprGroups exprs() {
    return sorted_exprs;
  }

  IdGroups ids() {
    return sorted_ids;
  }

  ~IdGraphStmtSort() override = default;

 protected:
  using IdGraphVisitor::handle;
  void handle(IdGroup id_group) override {
    sorted_ids.pushBack(id_group);
  }

  void handle(ExprGroup expr_group) override {
    sorted_exprs.pushBack(expr_group);
  }

  ExprGroups sorted_exprs;
  IdGroups sorted_ids;
};

// TODO: Comment is stale, update.
//
// There's three modes of these iter domain mappings all uniquely important in
// the lowering process.
//
// For EXACT/PERMISSIVE mode consider:
//
// consumer[i0, b1] = producer[i0]
// consumer->merge(0) (consumer will now be [i0 * b1])
// When producer is replayed as consumer (the direction we use for mapping)
// with BestEffortReplay forward_bcast_mismatch = True the producer to
// consumer map will have both a mapping of consumer(i0) to producer(i0) as
// well as consumer(i0*b1) to producer(i0). This latter mapping is important
// for loop nest mappings as the consumer will generate a loop based on i0*b1
// and the producer may be computeAt inside this loop nest. However, for
// indexing we do not want these two maps as producer may be indexed as i0*i1
// depending on the loop nest structure and how it was built. Therefore we
// really need to carry (at least) two sets of maps around for lowering.
//
// LOOP mode is important if we have something like:
// consumer[i0o, threadIdx.x{i0i}] = producer[i0o, threadIdx.y{i0i}](computeAt
// = 1) which can easily happen when using shared memory. We want to make sure
// that the iteration domain used for loop construction (concreteId) has the
// proper parallelization strategy. In parallel mode we do typical iteration
// domain mapping, however we remove from it any iteration domains outside the
// computeAt of producer when mapping. This guarentees we won't map
// IterDomains that could have different parallelization strategies. We also
// propagate the parallel strategy in parallel mode so all mapped IDs that
// must have the same parallel type, do.
//
// IdMappingMode::LOOP
//   Only maps leaf axes to left of compute at
//   Forward broadcast axes in replay
// IdMappingMode::PERMISSIVE
//   Forward broadcast axes in replay
//   Map all iteration domains
//   Always contain root mappings (otherwise they could have been forwarded in
//   broadcast)
// IdMappingMode::EXACT
//   Don't map any broadcast axes to non-broadcast axes
//   Do not forward through any broadcast IDs
// IdMappingMode::AlmostExact
//   Forward through broadcast axes, but not through to a non-broadcast axis
//     i.e. id{b1*i0}, id{i0} are mapped
//          id{i1*i0}, id{i0} are not mapped (this part is the difference from
//          PERMISSIVE)
//   Forward through split one axes, i.e. id{ceilDiv(i0, 1)}, id{i0} are mapped
//
class TORCH_CUDA_CU_API IterDomainGraphs : public PolymorphicBase {
 public:
  IterDomainGraphs(
      const std::vector<Expr*>& exprs,
      const std::vector<TensorView*>& additional_tvs,
      bool allow_self_mapping = false);

  IterDomainGraphs(
      const std::vector<Expr*>& exprs,
      bool allow_self_mapping = false);

  // Same as the above constructor with fusion->exprs() excpet fusion may have
  // some dangling inputs/outputs that are expected to have IterDomain entries
  // even though there's no possible connections from them.
  IterDomainGraphs(Fusion* fusion, bool allow_self_mapping = false);

  // Returns iter domain graph of provided mode.
  const IdGraph& idGraph(IdMappingMode mode) const;
  IdGraph& idGraph(IdMappingMode mode);

  // IterDomains from the original fusion are only allowed to be used once in
  // the IterDomain graph, id->uses() are not directly used as there's no bounds
  // check that would prevent a use from being defined that's not part of the
  // actual fusion definition.
  //
  // Note, any iter domains used during something like loop or concrete id
  // resolution could actually have multiple Expr* uses, and uses on disjoint id
  // sets should be used, not this.
  //
  // TODO: Refactor or remove?
  Expr* idUse(IterDomain* id) const;
  Expr* idDef(IterDomain* id) const;

  // TODO: Seems a bit unfortunate that this isn't IterDomain local information.
  const std::unordered_set<IterDomain*>& viewRfactorIds() const {
    return view_rfactor_ids_;
  }

  // Returns if a self mapping was detected that would invalidate assumptions of
  // the overall lowering system.
  //
  // TODO: Can we make this more of an alias analysis?
  // Ref: https://github.com/csarofeen/pytorch/pull/1954#discussion_r961940498
  bool hasSelfMapping() const {
    return self_mapping_info_.has_value();
  }

  // Update the LOOP ID disjoint sets with resolved computeWith
  void updateComputeWith(TensorView* compute_with_tv);

  std::string toString() const;

  // Replay Expr but with the inputs provided. IterDomainGraphss will be updated
  // for all maps that have entries, adding the output iter domains of the
  // replayed expression and adding potential mappings through the expression.
  Expr* addReplayAs(const std::vector<IterDomain*>& new_inputs, Expr* expr);

  // Similar to addReplayAs, but clones the expr exactly instead of replaying it
  // forward. It's up to the calling code to make sure the replacements are
  // valid for the provided expr. It's generally recommended that the
  // IterDomains exactly match those in the expr.
  //
  // "forward" dictates the same argument for mapThroughExpr. If forward the
  // function will apply mapThroughExpr forward if inputs map in each
  // initialized map. Else does the same but backwards through the expression
  // from outputs.
  Expr* addExprWithReplacement(
      const std::unordered_map<IterDomain*, IterDomain*>& old_2_new_ids,
      Expr* old_expr);

  // Make a new expr matching that provided but using the outputs provided.
  // IterDomainGraphss will be updated for all maps that have entries. Adding
  // the input iter domains of the replayed expression and adding potential
  // mappings through the expressions. Input domains will match exactly in all
  // properties as those in expr. This is unlike addReplayAs which will produce
  // new outputs using transformations directly.
  Expr* addBackwardsReplayAs(
      const std::vector<IterDomain*>& new_outputs,
      Expr* expr);

  // Make an exact copy of provided IterDomain (without rfactor set), and map
  // the copy to the original in all registered IdGraphs. IterDomain copy will
  // not have any registered uses or definitions.
  IterDomain* cloneIterDomain(IterDomain* id);

  // TODO: Should this not be private?
 protected:
  // Sometimes fusion inputs or outputs are disconnected from expressions, in
  // those cases we still may want to send in some additional tensor views from
  // the Fusion that don't have expressions associated with them.
  void build(
      const std::vector<Expr*>& exprs,
      const std::vector<TensorView*>& additional_tvs);

  // ======= START Iteration domain build process in order called =======

  // Fills id_uses_ and id_definitions_ for all IterDomains active in the
  // fusion.
  void buildIterDomainDefinitionsAndUses(
      const std::vector<TensorView*>& all_tvs);

  // Iterates over all IterDomains in id_definitions_ and calls initializeID on
  // a new IdGraph and returns it.
  IdGraph initializeIdGraph();

  // Fills disjoint_ids_[IdMappingMode::EXACT] for relationships between inputs
  // and first output of expr
  void buildExactMap(const std::vector<Expr*>& exprs);

  // Fills disjoint_ids_[IdMappingMode::ALMOSTEXACT]. Initialize AlmostExact as
  // Exact entries, then map anything that's either merged with a size-1 or
  // split by a size-1 dimension.
  void buildAlmostExactMap();

  // Fills disjoint_ids_[IdMappingMode::PERMISSIVE]. Initialize PermissiveMap as
  // AlmostExact entries, then map through broadcasts
  void buildPermissiveMap(const std::vector<Expr*>& exprs);

  //! Run through disjoint sets in the LOOP map, make sure there's only one
  //! non-serial parallel type in each disjoint set, set the parallel type of
  //! all IterDomains in the disjoint set to that PType.
  void validateAndPropagatePType() const;

  void buildLoopPromotionMap(const std::vector<Expr*>& exprs);

  // Returns the terminal rfactor or input iter domains each group in the almost
  // exact map covers (in the almost exact map). This effectively returns all
  // the input almost exact iter domain groups for each almost exact iter domain
  // group. RFactor axes are considered an "input" as all broadcast dimensions
  // have to be resolved by or before the rfactor iter domain.
  std::unordered_map<IdGroup, IdGroups> buildCoveredAlmostExact();

  void buildIndexMap(const std::vector<TensorView*>& all_tvs);

  // ======= END Iteration domain build process in order called =======

  // Errors if self mapping occurs
  void assertNoSelfMapping();

  // Keeps a disjoint set entry for all IterDomain for all mapping mode types.
  //
  // Using an array here might be nice, but it seems hard to use an enum as an
  // array key
  // https://stackoverflow.com/questions/2102582/how-can-i-count-the-items-in-an-enum
  std::unordered_map<IdMappingMode, IdGraph> id_graphs_;

  // If multiple transformations occur IterDomains could have multiple uses,
  // however only one should be active in the given Fusion. When we resolve loop
  // promotions during lowering, we can generate new iter domains from existing
  // ones, so there can be multiple uses generated. Tracks all the active iter
  // domain uses.
  std::unordered_map<IterDomain*, VectorOfUniqueEntries<Expr*>> id_uses_;

  // Make sure we don't blindly use definitions as we don't want to grab
  // transformations before a tensor view's root domain.
  std::unordered_map<IterDomain*, VectorOfUniqueEntries<Expr*>> id_definitions_;

  // Debug information to hold if a self mapping in a TensorView is found.
  c10::optional<std::tuple<TensorView*, IterDomain*, IterDomain*, std::string>>
      self_mapping_info_ = c10::nullopt;

  std::unordered_map<IdGroup, IterDomain*> loop_promotion_map_;

  std::unordered_set<IterDomain*> view_rfactor_ids_;
};

using DoubleBufferIndices = std::unordered_map<DoubleBufferLoopStage, Int*>;

} // namespace nvfuser
