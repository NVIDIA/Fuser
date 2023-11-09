// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#pragma once

#include <disjoint_set.h>
#include <fusion.h>
#include <ir/all_nodes.h>
#include <val_graph.h>

#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace nvfuser {

class ValGraph;

// A collection of IterDomainGraphs that are built from a fusion or series of
// expressions. These graphs are related, but have some distinct features based
// on the IdMappingMode.
//
// EXACT/PERMISSIVE mode:
//
// consumer[i0, b1] = producer[i0]
// consumer->merge(0) (consumer will now be [i0 * b1])
//
// When producer is replayed as consumer (the direction we use for mapping)
// with forwarding from ForwardingInfo the producer to consumer map will have
// both a mapping of consumer(i0) to producer(i0) as well as consumer(i0*b1) to
// producer(i0). This latter mapping is important for loop nest mappings as the
// consumer will generate a loop based on i0*b1 and the producer may be
// computeAt inside this loop nest. However, for indexing we do not want these
// two iter domains mapped as producer may be indexed as i0*i1 depending on the
// loop nest structure and how it was built.
//
// Exact mode is if the iter domain relationships from producer to consumer are
// considered the exact same size operating on matching dimensions from the root
// domain mapping.
//
// IdMappingMode::EXACT
//   Don't map any broadcast axes to non-broadcast axes
//   Do not forward through any broadcast IDs
//
class IdModel : public PolymorphicBase {
 public:
  IdModel(
      const std::vector<Expr*>& exprs,
      const std::vector<TensorView*>& additional_tvs);

  IdModel(const std::vector<Expr*>& exprs);

  // Same as the above constructor with fusion->exprs() excpet fusion may have
  // some dangling inputs/outputs that are expected to have IterDomain entries
  // even though there's no possible connections from them.
  IdModel(Fusion* fusion);

  // Returns iter domain graph of provided mode.
  const ValGraph& idGraph(IdMappingMode mode) const;
  ValGraph& idGraph(IdMappingMode mode);

  // TODO: Seems a bit unfortunate that this isn't IterDomain local information.
  const std::unordered_set<IterDomain*>& viewRfactorIds() const {
    return view_rfactor_ids_;
  }

  // Update the LOOP ID disjoint sets with resolved computeWith
  void updateComputeWith(TensorView* compute_with_tv);

  std::string toString() const;

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
  ValGraph initializeIdGraph(bool propagate_through_exprs = true);

  // Fills disjoint_ids_[IdMappingMode::EXACT] for relationships between inputs
  // and first output of expr
  void buildExactGraph(const std::vector<Expr*>& exprs);

  // Keeps a disjoint set entry for all IterDomain for all mapping mode types.
  //
  // Using an array here might be nice, but it seems hard to use an enum as an
  // array key
  // https://stackoverflow.com/questions/2102582/how-can-i-count-the-items-in-an-enum
  std::unordered_map<IdMappingMode, ValGraph> id_graphs_;

  // If multiple transformations occur IterDomains could have multiple uses,
  // however only one should be active in the given Fusion. When we resolve loop
  // promotions during lowering, we can generate new iter domains from existing
  // ones, so there can be multiple uses generated. Tracks all the active iter
  // domain uses.
  std::unordered_map<IterDomain*, VectorOfUniqueEntries<Expr*>> id_uses_;

  // Make sure we don't blindly use definitions as we don't want to grab
  // transformations before a tensor view's root domain. There can be
  // multiple definitions due to replays.
  std::unordered_map<IterDomain*, VectorOfUniqueEntries<Expr*>> id_definitions_;

  std::unordered_set<IterDomain*> view_rfactor_ids_;
};

} // namespace nvfuser
