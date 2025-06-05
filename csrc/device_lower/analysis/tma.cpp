// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#include <device_lower/analysis/tma.h>
#include <device_lower/lower2device.h>
#include <id_model/id_model.h>
#include <id_model/schedule.h>
#include <ir/utils.h>
#include <scheduler/tools/abstract_tensor.h>
#include <val_graph.h>
#include <val_graph_visitor.h>

#include <list>
#include <unordered_map>
#include <vector>

// See doc/dev/tma.md for design

namespace nvfuser {

std::ostream& operator<<(std::ostream& os, const TMADim& d) {
  os << "TMADim{"
     << "partitioned="
     << (d.partitioned ? d.partitioned->toString() : "nullptr")
     << ", box=" << (d.box ? d.box->toString() : "nullptr")
     << ", tile=" << (d.tile ? d.tile->toString() : "nullptr")
     << ", stride=" << (d.stride ? d.stride->toString() : "nullptr")
     << ", gmem_stride_bytes="
     << (d.gmem_stride_bytes ? d.gmem_stride_bytes->toInlineString()
                             : "nullptr")
     << "}";
  return os;
}

namespace {

int64_t getCpAsyncBulkTensorSwizzleSize(TensorView* smem_tv) {
  auto exprs = DependencyCheck::getAllExprsBetween(
      {smem_tv->getLogicalDomain().begin(), smem_tv->getLogicalDomain().end()},
      {smem_tv->getMaybeAllocationDomain().begin(),
       smem_tv->getMaybeAllocationDomain().end()});
  for (auto expr : exprs) {
    if (auto s = dynamic_cast<Swizzle*>(expr)) {
      return s->inX()->extent()->evaluate().as<int64_t>();
    }
  }
  return 1;
}

// Infer roles (bulk, non-bulk, partitioned, box, tile, and stride) of ValGroups
// by traversing along the traversal graph of the tensor indexer from the
// consumer(the smem tensor for TMA load, the gmem tensor fo TMA store)'s loop
// domain to the gmem tensor's allocation domain. The end result of the
// traversal are two sets of ValGroups: bulk and non-bulk, and a list of TMADim
// objects describing the roles (partitioned, box, tile, and stride) of
// ValGroups in a TMA dimension. Note that the returned list of TMADim objects
// are not the final TMA dimensions, because: first, its order is not
// determined; second, some dimensions may be collapsed according to the "define
// box by compositing" mechanism; and third, implicit size-one box and implicit
// whole box are not included.
namespace infer_roles {
// The traversal is done by a series of passes, each pass walks along the full
// path of expressions from the consumer's loop domain to the gmem tensor's
// allocation domain, but only looks for a specific pattern of ValGroups. Note
// that different passes are designed in a way that each expression in the
// traversal path is only useful for one pass. When one pass successfully
// pattern match an expression and extracted its information, this expression
// will become useless, therefore removed from the pending traversal path after
// one pass consumes it.

// Base class for all passes
class Pass {
 protected:
  ValGroups from(const ExprGroup& expr, Direction direction) {
    ValGraph& id_graph = GpuLower::current()->tensorIndexer().traversalGraph();
    return direction == Direction::Backward ? id_graph.outputGroups(expr)
                                            : id_graph.inputGroups(expr);
  }

  ValGroups to(const ExprGroup& expr, Direction direction) {
    ValGraph& id_graph = GpuLower::current()->tensorIndexer().traversalGraph();
    return direction == Direction::Backward ? id_graph.inputGroups(expr)
                                            : id_graph.outputGroups(expr);
  }

  // Is this expr the pattern this pass is looking for?
  virtual bool condition(ExprGroup expr, Direction direction) = 0;
  // Extract information from the pattern
  virtual void action(ExprGroup expr, Direction direction) = 0;

 public:
  // Traverse exprs. For each expr, loop for pattern as defined by condition.
  // If condition is met, apply action and remove the expr from the list.
  // Returns if any expr is removed.
  bool run(std::list<std::pair<ExprGroup, Direction>>& exprs) {
    bool changed = false;
    for (auto it = exprs.begin(); it != exprs.end();) {
      auto [expr, direction] = *it;
      if (condition(expr, direction)) {
        action(expr, direction);
        it = exprs.erase(it);
        changed = true;
      } else {
        it++;
      }
    }
    return changed;
  }

  virtual ~Pass() = default;
};

// A bulk group is a ValGroup that satisfies any of the following conditions:
// - It contains an IterDomain in the loop domain of the consumer tensor that is
//   parallelized with ParallelType::Bulk.
// - On the path from the consumer tensor's loop group to the gmem tensor's
//   allocation group, there is an expression that has all its from groups as
//   bulk groups.
// This pass assumes that bulk_groups is already initialized with the bulk
// groups of the consumer tensor's loop domain.
class InferBulkGroups : public Pass {
  std::unordered_set<ValGroup>& bulk_groups_;

  bool condition(ExprGroup expr, Direction direction) override {
    auto from = this->from(expr, direction);
    return std::all_of(from.begin(), from.end(), [&](const ValGroup& g) {
      return bulk_groups_.count(g) > 0;
    });
  }

  void action(ExprGroup expr, Direction direction) override {
    auto from = this->from(expr, direction);
    auto to = this->to(expr, direction);
    for (auto& g : to) {
      bulk_groups_.insert(std::move(g));
    }
  }

 public:
  InferBulkGroups(std::unordered_set<ValGroup>& bulk_groups)
      : bulk_groups_(bulk_groups) {}
};

// A non-bulk group is a ValGroup that satisfies any of the following
// conditions:
// - It contains an IterDomain in the loop domain of the consumer tensor that is
//   NOT parallelized with ParallelType::Bulk.
// - On the path from the consumer tensor's loop group to the gmem tensor's
//   allocation group, there is an expression that has all its from groups as
//   non-bulk groups.
// This pass assumes that bulk_groups is already initialized with the non-bulk
// groups of the consumer tensor's loop domain.
class InferNonBulkGroups : public Pass {
  std::unordered_set<ValGroup>& non_bulk_groups_;

  bool condition(ExprGroup expr, Direction direction) override {
    auto from = this->from(expr, direction);
    return std::all_of(from.begin(), from.end(), [&](const ValGroup& g) {
      return non_bulk_groups_.count(g) > 0;
    });
  }

  void action(ExprGroup expr, Direction direction) override {
    auto from = this->from(expr, direction);
    auto to = this->to(expr, direction);
    for (auto& g : to) {
      non_bulk_groups_.insert(std::move(g));
    }
  }

 public:
  InferNonBulkGroups(std::unordered_set<ValGroup>& non_bulk_groups)
      : non_bulk_groups_(non_bulk_groups) {}
};

// Note that not all groups are either bulk or non-bulk. Some groups are
// neither. For example, if on the path from the consumer tensor's loop group to
// the gmem tensor's allocation group, there is an expression that has two from
// groups, one is bulk and the other is non-bulk, then the to group is neither
// bulk nor non-bulk.

// A striding split is a split or merge expression that splits a box group into
// a tile group (outer) and a stride group (inner). Depending on the direction
// of traversal, this expression can be either a split or merge. The stride
// group must be a non-bulk group, and the tile group must be a bulk group. The
// information extracted from a striding split is stored in inferred_dims_.
class AnalyzeStridingSplit : public Pass {
  const std::unordered_set<ValGroup>& bulk_groups_;
  const std::unordered_set<ValGroup>& non_bulk_groups_;
  std::list<TMADim>& inferred_dims_;

  bool condition(ExprGroup expr, Direction direction) override {
    if (!expr->front()->isOneOf<Split, Merge>()) {
      return false;
    }
    auto from = this->from(expr, direction);
    auto to = this->to(expr, direction);
    return from.size() == 2 && to.size() == 1 &&
        bulk_groups_.count(from.at(0)) > 0 &&
        non_bulk_groups_.count(from.at(1)) > 0;
  }

  void action(ExprGroup expr, Direction direction) override {
    auto from = this->from(expr, direction);
    auto to = this->to(expr, direction);
    inferred_dims_.emplace_back();
    inferred_dims_.back().box = to.at(0);
    inferred_dims_.back().tile = from.at(0);
    inferred_dims_.back().stride = from.at(1);
    // The partitioned group may or may not be the same as the box group,
    // depending on if there is a boxing split in the traversal path. Set
    // partitioned group to the box group for now, and it may be updated by
    // AnalyzeBoxingSplit later.
    inferred_dims_.back().partitioned = to.at(0);
  }

 public:
  AnalyzeStridingSplit(
      const std::unordered_set<ValGroup>& bulk_groups,
      const std::unordered_set<ValGroup>& non_bulk_groups,
      std::list<TMADim>& inferred_dims)
      : bulk_groups_(bulk_groups),
        non_bulk_groups_(non_bulk_groups),
        inferred_dims_(inferred_dims) {}
};

// A boxing split is a split or merge expression that splits a partitioned group
// into a coordinate group (outer) and a box group (inner). Depending on the
// direction of traversal, this expression can be either a split or merge. The
// coordinate group must be a non-bulk group and the box group must either be a
// bulk group or a group that has been inferred as a box group by previous
// passes.
class AnalyzeBoxingSplit : public Pass {
  const std::unordered_set<ValGroup>& bulk_groups_;
  const std::unordered_set<ValGroup>& non_bulk_groups_;
  std::list<TMADim>& inferred_dims_;

  bool condition(ExprGroup expr, Direction direction) override {
    if (!expr->front()->isOneOf<Split, Merge>()) {
      return false;
    }
    auto from = this->from(expr, direction);
    auto to = this->to(expr, direction);
    return from.size() == 2 && to.size() == 1 &&
        non_bulk_groups_.count(from.at(0)) > 0 &&
        (bulk_groups_.count(from.at(1)) > 0 ||
         std::any_of(
             inferred_dims_.begin(),
             inferred_dims_.end(),
             [&](const TMADim& dim) { return dim.box == from.at(1); }));
  }

  void action(ExprGroup expr, Direction direction) override {
    auto from = this->from(expr, direction);
    auto to = this->to(expr, direction);
    auto dim_it = std::find_if(
        inferred_dims_.begin(), inferred_dims_.end(), [&](const TMADim& dim) {
          return dim.box == from.at(1);
        });
    if (dim_it != inferred_dims_.end()) {
      // If the box group has been inferred as a box group by previous passes,
      // then there is no need to create a new entry in inferred_dims_. We just
      // update the existing entry.
      dim_it->partitioned = to.at(0);
    } else {
      // There is no box group discovered by previous passes. This means that
      // the box group is a bulk group. Create a new entry in inferred_dims_.
      inferred_dims_.emplace_back();
      inferred_dims_.back().partitioned = to.at(0);
      inferred_dims_.back().box = from.at(1);
      inferred_dims_.back().tile = from.at(1);
      inferred_dims_.back().stride = nullptr;
    }
  }

 public:
  AnalyzeBoxingSplit(
      const std::unordered_set<ValGroup>& bulk_groups,
      const std::unordered_set<ValGroup>& non_bulk_groups,
      std::list<TMADim>& inferred_dims)
      : bulk_groups_(bulk_groups),
        non_bulk_groups_(non_bulk_groups),
        inferred_dims_(inferred_dims) {}
};

// Find the following pattern:

//                to
//               /  \.
// from0[non-bulk]  from1[partitioned]
//                    \.
//                    ...
//                      \.
//                      box

// If the extent of box divide the extent of from1, then move the partitioned
// flag up to get:

//                to[partitioned]
//               /  \.
// from0[non-bulk]  from1
//                    \.
//                    ...
//                      \.
//                      box

// Why does this work? Because, before running this pass, box is always an inner
// child of partitioned. So, the first time this pass runs, it will find the
// following pattern:

//                to
//               /  \.
// from0[non-bulk]  from1[partitioned]
//                  / \.
//        X[non-bulk]  box

// According to Theorem 2.1 (Equivalence of Split-Split) in the document
// doc/reading/iterdomain.md, the above pattern is equivalent to the following
// pattern if from1 is divisible by box:

//                to[partitioned]
//               /  \.
//      Y[non-bulk]  box
//         /       \.
// from0[non-bulk]  X[non-bulk]

// Here, instead of actually materializing the entire pattern above, we choose
// to preserve the original pattern and just move the partitioned flag up.
// That is, we get:

//                to[partitioned]
//               /  \.
// from0[non-bulk]  from1
//                  / \.
//        X[non-bulk]  box

// Although after transformation, box is no longer guaranteed to be a child of
// partitioned, it is still guaranteed that, there exists a mathematically
// equivalent pattern that has box as a child of partitioned. This property of
// the existence of an equivalent pattern will be maintained as we continue
// applying this pass over and over again.
//
// Why are we interested in moving the partitioned flag up? There are two
// reasons:
// - We want to consume as much expressions as possible in the infer_roles
//   stage. Any unconsumed expression will be treated as a view of the gmem
//   tensor's allocation domain. If there is a split not consumed, it will be
//   treated as a view which adds an extra requirement that the extent of the
//   that dimension must be a multiple of the split factor. This requirement is
//   not always satisfied, and is making our system overly restrictive.
// - Moving the partitioned flag up makes other mathematical equivalences more
//   likely to be discovered by other passes. For example, although it is
//   possible to prove that, mathematically, the following two patterns are
//   equivalent:
//                    A                                  A
//                   / \                                / \.
//                  B   C         ===========         ...  IC
//                 / \                                    /  \.
//                D   E           ===========            I    C
//                   / \.
//                  F   G
//                     / \.
//                    H   I
//   It is very unlikely we will prove such a theorem in our document because it
//   is just too specific and tedious. It is also very difficult to find this
//   pattern in code. Thanks to the transitivity of the equivalence relation, in
//   order to prove x = y, we can find a z such that both x = z and z = y are
//   easy to prove. The pattern we get after moving the partitioned flag up is
//   the z we are looking for. In the above example, with the partitioned flag
//   moved up, we can easily see that:
//            A                            A                        A
//           / \                          / \                      / \.
//          B   C      ===========       B   C    ===========    ...  IC
//         / \                          / \.                         /  \.
//        D   E        ===========    ...  I      ===========       I    C
//           / \.
//          F   G
//             / \.
//            H   I
//
// Why do we just move the partitioned flag up, but not materialize the entire
// equivalent pattern? Because: First, implementing the entire pattern needs
// work. Second, we can consider a group of ValGroup transformations as a black
// box. During indexing, index will be fed into the box from one side, and the
// box will output indices on the other side. The internal structure of this
// black box is less important in the sense that, if we replace the internal
// structure of the box with a mathematically equivalent one, the generated
// indices will also be mathematically equivalent. So materializing the entire
// pattern provides no value for us in indexing. The value we are looking for is
// to consume more expressions in the infer_roles stage, and to make other
// mathematical equivalences more likely to be discovered by other passes. Just
// moving the partitioned flag up is sufficient to achieve these goals.
class MovePartitionedGroupUpByRotation : public Pass {
  const std::unordered_set<ValGroup>& non_bulk_groups_;
  std::list<TMADim>& inferred_dims_;

  bool condition(ExprGroup expr, Direction direction) override {
    if (!expr->front()->isOneOf<Split, Merge>()) {
      return false;
    }
    auto from_ = from(expr, direction);
    auto to_ = to(expr, direction);
    if (from_.size() != 2 || to_.size() != 1) {
      return false;
    }
    if (non_bulk_groups_.count(from_.at(0)) == 0) {
      return false;
    }
    auto dim_it = std::find_if(
        inferred_dims_.begin(), inferred_dims_.end(), [&](const TMADim& dim) {
          return dim.partitioned == from_.at(1);
        });
    if (dim_it == inferred_dims_.end()) {
      return false;
    }
    bool is_divisible =
        simplifyExpr(
            SimplifyingIrBuilder::eqExpr(
                SimplifyingIrBuilder::modExpr(
                    dim_it->partitioned->front()->as<IterDomain>()->extent(),
                    dim_it->box->front()->as<IterDomain>()->extent()),
                expr->front()->fusion()->zeroVal()))
            ->isTrue();
    return is_divisible;
  }

  void action(ExprGroup expr, Direction direction) override {
    auto from_ = from(expr, direction);
    auto to_ = to(expr, direction);
    std::find_if(inferred_dims_.begin(), inferred_dims_.end(), [&](auto& dim) {
      return dim.partitioned == from_.at(1);
    })->partitioned = to_.at(0);
  }

 public:
  MovePartitionedGroupUpByRotation(
      const std::unordered_set<ValGroup>& non_bulk_groups,
      std::list<TMADim>& inferred_dims)
      : non_bulk_groups_(non_bulk_groups), inferred_dims_(inferred_dims) {}
};

// Find the following pattern:

//                      to
//                     /  \.
//    from0[partitioned]  from1[bulk]
//    /               |
//  ...              ...
//                    |
//                   box[bulk]

// transform it to:

//                to[partitioned]
//                  /         \.
//               from0     from1[bulk]
//               /   \         |
//             ...   ...       |
//                     \       |
//                  box[bulk]  |
//                       \     |
//                    new_box[bulk]

// Why does this work? Because, as described in the comment of
// MovePartitionedGroupUpByRotation, a pattern like below:

//    from0[partitioned]
//    /                \.
//  ...                ...
//                      |
//                   box[bulk]

// is mathematically equivalent to:

//    from0[partitioned]
//    /                \.
//  ...              box[bulk]

// Therefore, the pattern we are looking for is mathematically equivalent to:

//                      to
//                     /  \.
//    from0[partitioned]  from1[bulk]
//    /               |
//  ...            box[bulk]

// According to Theorem 2.1 (Equivalence of Split-Split) in the document
// doc/reading/iterdomain.md, the above pattern is equivalent to the following
// pattern:

//                to[partitioned]
//                  /         \.
//                ...     new_box[bulk]
//                         /         \.
//                       box     from1[bulk]

// Similar to MovePartitionedGroupUpByRotation, instead of actually
// materializing the entire pattern above, we choose to preserve the original
// pattern and just move the partitioned flag up and create the new_box group on
// top of the original pattern.
class MergeTileGroupsByRotation : public Pass {
  std::unordered_set<ValGroup>& bulk_groups_;
  std::list<TMADim>& inferred_dims_;

  bool condition(ExprGroup expr, Direction direction) override {
    if (!expr->front()->isOneOf<Split, Merge>()) {
      return false;
    }
    auto from_ = from(expr, direction);
    auto to_ = to(expr, direction);
    if (from_.size() != 2 || to_.size() != 1) {
      return false;
    }
    auto partitioned_dim = std::find_if(
        inferred_dims_.begin(), inferred_dims_.end(), [&](const auto& dim) {
          return dim.partitioned == from_.at(0);
        });
    if (partitioned_dim == inferred_dims_.end()) {
      return false;
    }
    if (bulk_groups_.count(partitioned_dim->box) == 0 ||
        bulk_groups_.count(from_.at(1)) == 0) {
      return false;
    }
    NVF_ERROR(
        partitioned_dim->tile == partitioned_dim->box &&
        partitioned_dim->stride == nullptr);
    return true;
  }

  void action(ExprGroup expr, Direction direction) override {
    ValGraph& id_graph = GpuLower::current()->tensorIndexer().traversalGraph();
    auto from_ = from(expr, direction);
    auto to_ = to(expr, direction);
    auto partitioned_dim = std::find_if(
        inferred_dims_.begin(), inferred_dims_.end(), [&](const auto& dim) {
          return dim.partitioned == from_.at(0);
        });
    auto new_box = merge(&id_graph, partitioned_dim->tile, from_.at(1));
    bulk_groups_.insert(new_box);
    inferred_dims_.emplace_back();
    inferred_dims_.back().partitioned = to_.at(0);
    inferred_dims_.back().box = new_box;
    inferred_dims_.back().tile = new_box;
    inferred_dims_.back().stride = nullptr;
    inferred_dims_.erase(partitioned_dim);
  }

 public:
  MergeTileGroupsByRotation(
      std::unordered_set<ValGroup>& bulk_groups,
      std::list<TMADim>& inferred_dims)
      : bulk_groups_(bulk_groups), inferred_dims_(inferred_dims) {}
};

std::tuple<
    std::unordered_set<ValGroup>, // bulk groups, see the comment of
                                  // InferBulkGroups for its definition
    std::unordered_set<ValGroup>, // non-bulk groups, see the comment of
                                  // InferNonBulkGroups for its definition
    std::list<TMADim>> // inferred dimension information
run(
    // The whole path from consumer's loop domain to gmem tensor's allocation
    // domain. Passed in as a non-const reference because passes will remove
    // expressions from it when it has extracted information from them.
    std::list<std::pair<ExprGroup, Direction>>& exprs,
    TensorView* consumer_tv) {
  ValGraph& id_graph = GpuLower::current()->tensorIndexer().traversalGraph();

  std::unordered_set<ValGroup> bulk_groups;
  std::unordered_set<ValGroup> nonbulk_groups;
  for (auto id : consumer_tv->getLoopDomain()) {
    if (id->getParallelType() == ParallelType::Bulk) {
      bulk_groups.insert(id_graph.toGroup(id));
    } else {
      nonbulk_groups.insert(id_graph.toGroup(id));
    }
  }

  std::list<TMADim> inferred_dims;

  InferBulkGroups bulk_pass(bulk_groups);
  InferNonBulkGroups nonbulk_pass(nonbulk_groups);
  AnalyzeStridingSplit striding_split_pass(
      bulk_groups, nonbulk_groups, inferred_dims);
  AnalyzeBoxingSplit boxing_split_pass(
      bulk_groups, nonbulk_groups, inferred_dims);
  MovePartitionedGroupUpByRotation move_partitioned_pass(
      nonbulk_groups, inferred_dims);
  MergeTileGroupsByRotation merge_tile_pass(bulk_groups, inferred_dims);

  bool changed = true;
  while (changed) {
    changed = false;
    changed = changed || bulk_pass.run(exprs);
    changed = changed || nonbulk_pass.run(exprs);
    changed = changed || striding_split_pass.run(exprs);
    changed = changed || boxing_split_pass.run(exprs);
    changed = changed || move_partitioned_pass.run(exprs);
    changed = changed || merge_tile_pass.run(exprs);
  }
  return {bulk_groups, nonbulk_groups, inferred_dims};
}

} // namespace infer_roles

// Passes in namespace infer_roles traverse the list of expressions on the path
// from consumer's loop domain to gmem tensor's allocation domain and remove
// expressions if they are used to specify roles of a TMA schedule. The
// remaining expressions after these passes will be sent to this namespace.
// These remaining expression will be treated as view operations that view the
// allocation domain of the gmem tensor as a domain in the middle between the
// loop domain in the consumer and the the allocation domain of the gmem tensor.
// Code in this namespace is responsible for processing these expressions to
// compute the view and the contiguity and stride of each dimension in the view.
// We sometimes call this domain-in-the-middle the "raw TMA domain".
namespace view_gmem_alloc_domain_with_exprs {

// Get the allocation domain of the gmem tensor as ValGroups, and the contiguity
// and stride of each dimension.
std::list<std::tuple<ValGroup, /*contiguity*/ bool, /*stride*/ Val*>>
getGmemAllocDomain(TensorView* gmem_tv) {
  ValGraph& id_graph = GpuLower::current()->tensorIndexer().traversalGraph();
  std::list<std::tuple<ValGroup, /*contiguity*/ bool, /*stride*/ Val*>>
      allocation_domain;
  auto metadata = IrBuilder::metadataExpr(gmem_tv);
  auto alloc_strides = IrBuilder::getAttrExpr(metadata, "alloc_stride");
  auto gmem_alloc_dom =
      TensorDomain::noReductions(gmem_tv->getMaybeAllocationDomain());
  for (auto it = gmem_alloc_dom.begin(); it != gmem_alloc_dom.end(); it++) {
    auto id = *it;
    if (id->isBroadcast()) {
      continue;
    }
    int64_t pos = std::distance(gmem_alloc_dom.begin(), it);
    auto stride = IrBuilder::getItemExpr(alloc_strides, pos);
    allocation_domain.emplace_back(
        id_graph.toGroup(id), gmem_tv->getContiguity().at(pos).value(), stride);
  }
  return allocation_domain;
}

// Helper class that processes one expr in the traversal path
class HandleExpr {
  std::list<std::tuple<ValGroup, /*contiguity*/ bool, /*stride*/ Val*>>&
      frontier_;

  static auto from(ExprGroup expr, Direction direction) {
    ValGraph& id_graph = GpuLower::current()->tensorIndexer().traversalGraph();
    return direction == Direction::Forward ? id_graph.outputGroups(expr)
                                           : id_graph.inputGroups(expr);
  }

  static auto to(ExprGroup expr, Direction direction) {
    ValGraph& id_graph = GpuLower::current()->tensorIndexer().traversalGraph();
    return direction == Direction::Forward ? id_graph.inputGroups(expr)
                                           : id_graph.outputGroups(expr);
  }

  // Split one ValGroup in frontier_ into two ValGroups. Note that this is used
  // to handle both Split and Merge, depending on the direction.
  void handleOneToTwo(
      std::vector<ValGroup> from,
      std::vector<ValGroup> to,
      ExprGroup expr) {
    auto from_it =
        std::find_if(frontier_.begin(), frontier_.end(), [from](auto tuple) {
          return std::get<0>(tuple) == from[0];
        });
    NVF_ERROR(
        from_it != frontier_.end(),
        "The TMA domain must be equivalent to the allocation domain of the "
        "gmem tensor, but ",
        from[0]->toString(),
        " is not on the path.");
    if (auto split = dynamic_cast<Split*>(expr->front())) {
      // Only the forward of a split can be indivisible, otherwise, it is
      // guaranteed to be divisible.
      Val* is_divisible = SimplifyingIrBuilder::eqExpr(
          SimplifyingIrBuilder::modExpr(
              from[0]->front()->as<IterDomain>()->extent(), split->factor()),
          split->fusion()->zeroVal());
      GpuLower::current()->validate(
          is_divisible,
          "Invalid view in TMA: the extent of ",
          from[0]->toString(),
          " must be divisible by ",
          split->factor()->toInlineString());
    }
    frontier_.insert(
        from_it,
        std::make_tuple(
            to[0],
            true,
            SimplifyingIrBuilder::mulExpr(
                std::get<2>(*from_it),
                to[1]->front()->as<IterDomain>()->extent())));
    std::get<0>(*from_it) = to[1];
  }

  // "Merge" two ValGroups in frontier_ into one ValGroup, and update the stride
  // of the merged ValGroup. Note that this is used to handle both Split and
  // Merge, depending on the direction.
  void handleTwoToOne(std::vector<ValGroup> from, std::vector<ValGroup> to) {
    auto outer = from[0];
    auto outer_it =
        std::find_if(frontier_.begin(), frontier_.end(), [outer](auto tuple) {
          return std::get<0>(tuple) == outer;
        });
    NVF_ERROR(
        outer_it != frontier_.end(),
        "The TMA domain must be equivalent to the allocation domain of the "
        "gmem tensor, but ",
        outer->toString(),
        " is not on the path.");
    auto inner = from[1];
    auto inner_it = std::next(outer_it);
    NVF_ERROR(
        inner_it != frontier_.end(),
        "The TMA domain must be equivalent to the allocation domain, but ",
        inner->toString(),
        " is not on the path.");
    NVF_ERROR(
        std::get<0>(*inner_it) == inner && std::get<1>(*outer_it),
        "Can not merge discontiguous dimensions, but ",
        outer->toString(),
        " is merged with ",
        inner->toString());
    std::get<0>(*inner_it) = to[0];
    frontier_.erase(outer_it);
  }

 public:
  HandleExpr(
      std::list<std::tuple<ValGroup, /*contiguity*/ bool, /*stride*/ Val*>>&
          frontier)
      : frontier_(frontier) {}

  void handle(ExprGroup expr, Direction direction) {
    NVF_ERROR(!expr->empty());
    bool is_supported_expr = expr->front()->isOneOf<Split, Merge>();
    NVF_ERROR(
        is_supported_expr,
        "TMA domain must be a view of the allocation domain of the gmem "
        "tensor, but ",
        expr->toString(),
        " is not a valid expression for view.");
    auto from_ = from(expr, direction);
    auto to_ = to(expr, direction);
    if (from_.size() == 1) {
      handleOneToTwo(from_, to_, expr);
    } else {
      handleTwoToOne(from_, to_);
    }
  }
};

std::list<std::tuple<ValGroup, /*contiguity*/ bool, /*stride*/ Val*>> run(
    TensorView* gmem_tv,
    std::list<std::pair<ExprGroup, Direction>>& exprs) {
  // Initialize frontier as the allocation domain of gmem_tv
  auto frontier = getGmemAllocDomain(gmem_tv);
  // Propagate from the gmem allocation domain towards the consumer tensor's
  // loop domain with the given exprs. Because the given exprs often do not
  // contain the full path from the consumer tensor's loop domain to the gmem
  // tensor's allocation domain, the propagation will automatically stop in the
  // middle after all given expressions have been exhausted.. This propagation
  // must consume all expressions in the given list. If there is any
  // unrecognized expression, this means there is an error in the schedule. The
  // exprs are in the topology order from the consumer tensor's loop domain to
  // the gmem tensor's allocation domain, so we need to use the reverse iterator
  // to traverse.
  HandleExpr handle_expr(frontier);
  for (auto it = exprs.rbegin(); it != exprs.rend(); it++) {
    auto [expr, direction] = *it;
    handle_expr.handle(expr, direction);
  }
  return frontier;
}

} // namespace view_gmem_alloc_domain_with_exprs

// Collapse the inferred raw TMA domain to get the final TMA domain according to
// the "define box by compositing" mechanism.
namespace collapse_tma_domain {

// There can only be four types of ValGroups in the raw TMA domain:
// -  P: partitioned ValGroup
// -  C: coordinate ValGroup
// - SB: strided box ValGroup
// - CB: contiguous box ValGroup
enum IDType { P, C, SB, CB };

// Helper class for merging ValGroups and bookkeeping the information about
// these groups
class DomainMerger {
  AbstractTensor domain_;
  std::vector<std::pair<bool, Val*>> contiguity_and_stride_;
  std::unordered_set<ValGroup>& bulk_groups_;
  std::unordered_set<ValGroup>& nonbulk_groups_;
  std::list<TMADim>& dim_info_;
  MmaInputSmemSwizzle swizzle_;
  int64_t item_size_bytes_;

 public:
  DomainMerger(
      std::list<std::tuple<ValGroup, bool, Val*>> raw_tma_domain,
      std::unordered_set<ValGroup>& bulk_groups,
      std::unordered_set<ValGroup>& nonbulk_groups,
      std::list<TMADim>& dim_info,
      MmaInputSmemSwizzle swizzle,
      int64_t item_size_bytes)
      : bulk_groups_(bulk_groups),
        nonbulk_groups_(nonbulk_groups),
        dim_info_(dim_info),
        swizzle_(swizzle),
        item_size_bytes_(item_size_bytes) {
    ValGraph& id_graph = GpuLower::current()->tensorIndexer().traversalGraph();
    contiguity_and_stride_.reserve(raw_tma_domain.size());
    for (auto& item : raw_tma_domain) {
      domain_.emplaceBack(
          ValGroupAndItsGraph{std::move(std::get<0>(item)), &id_graph});
      contiguity_and_stride_.emplace_back(std::get<1>(item), std::get<2>(item));
    }
  }

  bool contiguity(int64_t i) const {
    return contiguity_and_stride_[i].first;
  }

  Val* stride(int64_t i) const {
    return contiguity_and_stride_[i].second;
  }

  const ValGroup& operator[](int64_t i) const {
    return domain_[i].as<ValGroupAndItsGraph>().group;
  }

  size_t size() const {
    return domain_.size();
  }

  IDType type(int64_t i) const {
    const auto& g = (*this)[i];
    // Partitioned group must have already been inferred as a dimension in
    // dim_info_, so we just try to find it. Note that dim.partitioned == g
    // alone does not guarantee that g is a partitioned group, because it may
    // also be a strided-box group.
    if (std::any_of(dim_info_.begin(), dim_info_.end(), [&](const TMADim& dim) {
          return dim.partitioned == g && dim.box != g;
        })) {
      return P;
    }
    auto dim_it = std::find_if(
        dim_info_.begin(), dim_info_.end(), [&](const TMADim& dim) {
          return dim.box == g;
        });
    // If there is an existing dimension info, use it.
    if (dim_it != dim_info_.end()) {
      if (dim_it->stride) {
        NVF_ERROR(dim_it->tile != g);
        return SB;
      } else {
        NVF_ERROR(dim_it->tile == g);
        return CB;
      }
    }
    // Otherwise, infer based on bulk_groups_ and nonbulk_groups_.
    if (bulk_groups_.count(g) > 0) {
      return CB;
    }
    NVF_ERROR(nonbulk_groups_.count(g) > 0);
    return C;
  }

  bool shouldMerge(int64_t i) {
    auto type0 = type(i);
    auto type1 = type(i + 1);

    bool may_increasing_box_size = (type0 == CB && type1 == CB);
    if (!may_increasing_box_size) {
      return true;
    }

    auto extent0 = (*this)[i]->front()->as<IterDomain>()->extent();
    auto extent1 = (*this)[i + 1]->front()->as<IterDomain>()->extent();
    Val* merged_extent = SimplifyingIrBuilder::mulExpr(extent0, extent1);

    bool merging_innermost = ((int64_t)size() == i + 2);

    // If merging makes the size of a dimension larger than 256, we should not
    // merge.
    constexpr int64_t largest_dim_size =
        256; // Dimension size must be <= 256 as limited by hardware.
    Val* too_large_after_merge = SimplifyingIrBuilder::gtExpr(
        merged_extent, IrBuilder::create<Val>(largest_dim_size));
    if (simplifyExpr(too_large_after_merge)->isTrue()) {
      return false;
    }

    // If merging makes the inner size larger than the swizzle size,
    // we should not merge
    if (merging_innermost && swizzle_ != MmaInputSmemSwizzle::None) {
      const int64_t swizzle_size =
          getBytesFromSwizzle(swizzle_) / item_size_bytes_;
      Val* merging_makes_gt_swizzle_size = SimplifyingIrBuilder::gtExpr(
          merged_extent, IrBuilder::create<Val>(swizzle_size));
      if (simplifyExpr(merging_makes_gt_swizzle_size)->isTrue()) {
        return false;
      }
    }

    // Because the shape is dynamic, we don't know if we should merge or
    // not. For this case, we always assume merging is better than not
    // merging.
    return true;
  }

  void merge(int64_t i) {
    auto type0 = type(i);
    auto type1 = type(i + 1);
    auto g0 = (*this)[i];
    auto g1 = (*this)[i + 1];
    domain_.merge(i);
    contiguity_and_stride_.erase(contiguity_and_stride_.begin() + i);
    const auto& g = (*this)[i];

    // Update bulk_groups_ and nonbulk_groups_ by propagating through the merge.
    if (bulk_groups_.count(g0) > 0 && bulk_groups_.count(g1) > 0) {
      bulk_groups_.insert(g);
    }
    if (nonbulk_groups_.count(g0) > 0 && nonbulk_groups_.count(g1) > 0) {
      nonbulk_groups_.insert(g);
    }
    // Set the dim_info_ for the merged group.
    if (type0 == CB && type1 == CB) {
      dim_info_.emplace_back();
      dim_info_.back().partitioned = g;
      dim_info_.back().box = g;
      dim_info_.back().tile = g;
      dim_info_.back().stride = nullptr;
    } else if (type0 == C && (type1 == CB || type1 == SB)) {
      dim_info_.emplace_back();
      dim_info_.back().partitioned = g;
      dim_info_.back().box = g1;
      auto dim_it = std::find_if(
          dim_info_.begin(), dim_info_.end(), [&](const TMADim& dim) {
            return dim.partitioned == g1;
          });
      ValGroup tile, stride;
      if (dim_it == dim_info_.end()) {
        dim_info_.back().tile = g1;
        dim_info_.back().stride = nullptr;
      } else {
        NVF_ERROR(dim_it->box == g1);
        dim_info_.back().tile = dim_it->tile;
        dim_info_.back().stride = dim_it->stride;
      }
    } else {
      NVF_ERROR(type0 == C && type1 == C, "Invalid merge");
      // Information about coordinate groups are not stored in dim_info_,
      // nothing to do here.
    }
    // Remove the original dimensions from dim_info_.
    auto it0 = std::find_if(
        dim_info_.begin(), dim_info_.end(), [&](const TMADim& dim) {
          return dim.partitioned == g0;
        });
    if (it0 != dim_info_.end()) {
      dim_info_.erase(it0);
    }
    auto it1 = std::find_if(
        dim_info_.begin(), dim_info_.end(), [&](const TMADim& dim) {
          return dim.partitioned == g1;
        });
    if (it1 != dim_info_.end()) {
      dim_info_.erase(it1);
    }
  }
};

// Do the collapse and returns the final TMA domnain. We first collapse
// contiguous C groups to form larger C groups. and contiguous CB groups to form
// larger CB groups. Then we collapse C group with its contiguous CB/SB groups
// to form partitioned groups.
std::vector<TMADim> run(
    std::list<std::tuple<ValGroup, /*contiguity*/ bool, /*stride*/ Val*>>
        raw_tma_domain,
    std::unordered_set<ValGroup>& bulk_groups,
    std::unordered_set<ValGroup>& nonbulk_groups,
    std::list<TMADim>& dim_info,
    int64_t item_size_bytes,
    MmaInputSmemSwizzle swizzle) {
  DomainMerger tma_domain(
      std::move(raw_tma_domain),
      bulk_groups,
      nonbulk_groups,
      dim_info,
      swizzle,
      item_size_bytes);
  // merge contiguous C groups and CB groups
  for (int64_t i = 0; i < (int64_t)tma_domain.size() - 1; i++) {
    if (!tma_domain.contiguity(i)) {
      continue;
    }
    if ((tma_domain.type(i) == C && tma_domain.type(i + 1) == C) ||
        (tma_domain.type(i) == CB && tma_domain.type(i + 1) == CB)) {
      if (tma_domain.shouldMerge(i)) {
        tma_domain.merge(i);
        i--;
      }
    }
  }
  // merge contiguous C with SB/CB
  for (int64_t i = 0; i < (int64_t)tma_domain.size() - 1; i++) {
    if (!tma_domain.contiguity(i)) {
      continue;
    }
    if (tma_domain.type(i) == C &&
        (tma_domain.type(i + 1) == SB || tma_domain.type(i + 1) == CB)) {
      if (tma_domain.shouldMerge(i)) {
        tma_domain.merge(i);
        i--;
      }
    }
  }

  // Compute the final TMA domain. As required by the hardware, tensors used by
  // TMA must be in column major, so our final TMA domain is also from innermost
  // to outermost.
  std::vector<TMADim> result;
  for (int64_t i = (int64_t)tma_domain.size() - 1; i >= 0; i--) {
    const auto& g = tma_domain[i];
    result.emplace_back();
    auto dim_it =
        std::find_if(dim_info.begin(), dim_info.end(), [&](const TMADim& dim) {
          return dim.partitioned == g;
        });
    if (dim_it != dim_info.end()) {
      // If there is already an entry in dim_info, just use it
      result.back() = std::move(*dim_it);
      dim_info.erase(dim_it);
    } else {
      // Otherwise, create a new entry
      if (bulk_groups.count(g) > 0) {
        // Implicit whole box dimension
        result.back().partitioned = g;
        result.back().box = g;
        result.back().tile = g;
        result.back().stride = nullptr;
      } else {
        // Implicit size-one box dimension
        NVF_ERROR(nonbulk_groups.count(g) > 0, g->toString());
        result.back().partitioned = g;
        result.back().box = nullptr;
        result.back().tile = nullptr;
        result.back().stride = nullptr;
      }
    }
    result.back().gmem_stride_bytes =
        SimplifyingIrBuilder::mulExpr(tma_domain.stride(i), item_size_bytes);
  }
  return result;
}

} // namespace collapse_tma_domain

TMAInfo getTMAInfo(LoadStoreOp* ldst) {
  TensorView* producer_tv = ldst->in()->as<TensorView>();
  // In case the producer is aliased, use the alias instead
  producer_tv = GpuLower::current()->getMaybeTensorProducerAlias(producer_tv);

  TensorView* consumer_tv = ldst->out()->as<TensorView>();
  TensorView *smem_tv = nullptr, *gmem_tv = nullptr;
  if (producer_tv->getMemoryType() == MemoryType::Shared) {
    NVF_ERROR(consumer_tv->getMemoryType() == MemoryType::Global);
    smem_tv = producer_tv;
    gmem_tv = consumer_tv;
  } else {
    NVF_ERROR(producer_tv->getMemoryType() == MemoryType::Global);
    NVF_ERROR(consumer_tv->getMemoryType() == MemoryType::Shared);
    smem_tv = consumer_tv;
    gmem_tv = producer_tv;
  }

  std::list<std::pair<ExprGroup, Direction>> exprs = [&]() {
    ValGraph& id_graph = GpuLower::current()->tensorIndexer().traversalGraph();
    auto exprs_vec = ValGraphBFS::getExprGroupsBetween(
                         id_graph,
                         id_graph.toGroups(consumer_tv->getLoopDomain()),
                         id_graph.toGroups(gmem_tv->getMaybeAllocationDomain()))
                         .first;
    return std::list(exprs_vec.begin(), exprs_vec.end());
  }();

  // Infer roles of ValGroups. In an expression is used to infer role, remove it
  // from exprs.
  auto [bulk_groups, nonbulk_groups, inferred_dims] =
      infer_roles::run(exprs, consumer_tv);

  // Treat the remaining expressions as view expressions to view the allocation
  // domain of the gmem tensor as something in the middle between the consumer
  // tensor's loop domain and the gmem tensor's allocation domain. The result is
  // also called "raw TMA domain".
  auto raw_tma_domain = view_gmem_alloc_domain_with_exprs::run(gmem_tv, exprs);

  NVF_ERROR(
      std::get<1>(raw_tma_domain.back()),
      "The innermost dimension of the TMA domain must be contiguous");
  auto inner_it = std::find_if(
      inferred_dims.begin(), inferred_dims.end(), [&](const TMADim& dim) {
        return dim.partitioned == std::get<0>(raw_tma_domain.back());
      });
  NVF_ERROR(
      inner_it == inferred_dims.end() || inner_it->stride == nullptr,
      "When interleave is CU_TENSOR_MAP_INTERLEAVE_NONE ",
      "(this is always the case for nvFuser now)",
      ", the first element of elementStrides must be one.");

  MmaInputSmemSwizzle swizzle = getSwizzle(smem_tv);

  // Handle "defining box by compositing" by collapsing some dimensions in the
  // raw TMA domain to get the final TMA domain.
  auto final_tma_domain = collapse_tma_domain::run(
      std::move(raw_tma_domain),
      bulk_groups,
      nonbulk_groups,
      inferred_dims,
      dataTypeSize(gmem_tv->dtype()),
      swizzle);
  return TMAInfo(std::move(final_tma_domain), swizzle, gmem_tv);
}

} // namespace

Val* TMAInfo::tensorMap() const {
  std::vector<Val*> tensor_sizes_inner_to_outer;
  std::transform(
      dims_.begin(),
      dims_.end(),
      std::back_inserter(tensor_sizes_inner_to_outer),
      [](const TMADim& d) { return d.tensorSize(); });

  std::vector<Val*> tensor_strides_inner_to_outer;
  std::transform(
      dims_.begin() + 1,
      dims_.end(),
      std::back_inserter(tensor_strides_inner_to_outer),
      [](const TMADim& d) { return d.gmem_stride_bytes; });

  std::vector<Val*> box_sizes_inner_to_outer;
  std::transform(
      dims_.begin(),
      dims_.end(),
      std::back_inserter(box_sizes_inner_to_outer),
      [](const TMADim& d) { return d.boxSize(); });

  std::vector<Val*> element_strides_inner_to_outer;
  std::transform(
      dims_.begin(),
      dims_.end(),
      std::back_inserter(element_strides_inner_to_outer),
      [](const TMADim& d) { return d.elementStride(); });

  int64_t dim = (int64_t)tensor_sizes_inner_to_outer.size();
  auto metadata = IrBuilder::metadataExpr(gmem_tv_);
  auto global_address = IrBuilder::getAttrExpr(metadata, "data");

  Val* global_stride =
      (dim > 1
           ? IrBuilder::arrayExpr(tensor_strides_inner_to_outer)
           : IrBuilder::create<Val>(
                 std::vector<int64_t>{},
                 ArrayType{std::make_shared<DataType>(DataType::Index), 0}));

  return tma::encodeTensorMapTiled(
      gmem_tv_->dtype(),
      global_address,
      IrBuilder::arrayExpr(tensor_sizes_inner_to_outer),
      global_stride,
      IrBuilder::arrayExpr(box_sizes_inner_to_outer),
      IrBuilder::arrayExpr(element_strides_inner_to_outer),
      tma::TensorMapInterleave::NoInterleave,
      swizzle_,
      tma::TensorMapL2Promotion::NoL2Promotion,
      tma::TensorMapFloatOOBFill::NoOOBFill);
}

std::unordered_map<TensorView*, const TMAInfo> getConsumerToTMAInfoMap(
    Fusion* fusion) {
  std::unordered_map<TensorView*, const TMAInfo> result;
  for (Expr* expr : fusion->exprs()) {
    if (auto ldst = dynamic_cast<LoadStoreOp*>(expr)) {
      if (ldst->opType() == LoadStoreOpType::CpAsyncBulkTensorTile ||
          ldst->opType() == LoadStoreOpType::CpAsyncBulk) {
        NVF_ERROR(
            result.emplace(ir_utils::getTvOutput(ldst), getTMAInfo(ldst))
                .second,
            "Ambiguous TMA information, likely something is wrong in the "
            "Fusion IR");
      }
    }
  }
  return result;
}

MmaInputSmemSwizzle getSwizzle(TensorView* tv) {
  NVF_ERROR(tv->getMemoryType() == MemoryType::Shared);
  return getSwizzleFromBytes(
      getCpAsyncBulkTensorSwizzleSize(tv) * core_matrix_width_bytes);
}

} // namespace nvfuser
