// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#include <device_lower/utils.h>
#include <id_model/id_model.h>
#include <instrumentation.h>
#include <ir/builder.h>
#include <ir/iostream.h>
#include <ir/utils.h>
#include <logical_domain_map.h>
#include <val_graph.h>

#include <device_lower/pass/replace_size.h>

namespace nvfuser {

namespace {
// Going to generate a map of tensor view root domain extents to reduce the
// number used during lowering. For example if we have:
//
// T2[i0, i1] = T1[i0, i1] + T2[i2, i3]
//
// We know it would be safe to use:
//
// T2[i0, i1] = T1[i0, i1] + T2[i0, i1]
//
// And that way we don't generate T2.size[0] and T2.size[1], instead we will
// reuse T1.size[0] and T1.size[1]
// This is important when doing CSE as T2 and T1 would otherwise look like
// they're using different values, even though we know they're the same
//
// There's some duplicate logic here that's in computeAt map, but it's not so
// concice there to pull out. May want to consider making this mapping its own
// class especially as it may be useful during scheduling.
std::unordered_map<Val*, Val*> getSimplificationMap(Fusion* fusion) {
  IdModel id_model(fusion, /*build_graphs=*/false);
  id_model.buildExactGraph();
  ValGraph& graph = id_model.idGraph(IdMappingMode::EXACT);

  std::unordered_set<IterDomain*> fusion_input_ids;
  for (Val* v : fusion->inputs()) {
    if (auto* tv = dynamic_cast<TensorView*>(v)) {
      for (IterDomain* id : tv->getLogicalDomain()) {
        fusion_input_ids.insert(id);
      }
    }
  }

  std::unordered_map<Val*, Val*> simplification_map;

  for (const ValGroup& group : graph.disjointValSets().disjointSets()) {
    // For each ValGroup, find a single extent to use for all extents of
    // IterDomains in the group. These are chosen in descending order of
    // preference:
    // 1. Constant ints. These might be non-immediate constants
    // 2. Extents of input TVs.
    // 3. Extents of non-input TVs.
    // Within these three classes, we find the IterDomain with the
    // smallest name(). For case 3, we also prefer the IterDomain with
    // the simplest extent, which has the smallest number of defining
    // expessions.
    bool group_is_const = false;
    IterDomain* rep = nullptr;
    bool rep_is_input_id = false;
    int64_t rep_num_defs = 0;
    std::unordered_set<Val*> dynamic_scalars;
    for (Val* v : *group) {
      auto* id = dynamic_cast<IterDomain*>(v);
      NVF_ERROR(
          id != nullptr, "Expected only IterDomains in exact graph ValGroups");
      bool is_input_id = fusion_input_ids.count(id) > 0;
      Val* ext = id->extent();
      bool ext_is_const = ext->isConstInt();
      if (!ext_is_const) {
        dynamic_scalars.insert(ext);
      }

      // Initializing rep with the first ID
      if (rep == nullptr) {
        rep = id;
        rep_is_input_id = is_input_id;
        group_is_const = ext_is_const;
        // If neigher const nor input, record the number of exprs
        if (!ext_is_const && !is_input_id) {
          rep_num_defs = ir_utils::getOperationCount(id->extent());
        }
        continue;
      }

      if (ext_is_const) {
        if (!group_is_const || id->name() < rep->name()) {
          rep = id;
          // This lets us avoid repeating the costly isConstInt check
          group_is_const = true;
          rep_is_input_id = is_input_id;
          continue;
        }
      } else if (is_input_id) {
        if (group_is_const) {
          continue;
        }
        if (!rep_is_input_id || id->name() < rep->name()) {
          rep = id;
          rep_is_input_id = is_input_id;
          continue;
        }
      } else {
        // id is a non-input TV
        if (group_is_const || rep_is_input_id) {
          continue;
        }
        auto num_defs = ir_utils::getOperationCount(id->extent());
        if (num_defs < rep_num_defs ||
            (num_defs == rep_num_defs && id->name() < rep->name())) {
          rep = id;
          rep_is_input_id = is_input_id;
          rep_num_defs = num_defs;
          continue;
        }
      }
    }
    NVF_ERROR(rep != nullptr);
    Val* rep_ext = rep->extent();
    for (Val* v : *group) {
      auto* id = v->as<IterDomain>();
      Val* ext = id->extent();
      // Don't remap constants or rep_ext itself
      if (!ext->sameAs(rep_ext) && dynamic_scalars.count(ext)) {
        simplification_map.emplace(ext, rep_ext);
      }
    }
  }
  return simplification_map;
}

} // namespace

void replaceSymbolicSizes(Fusion* fusion) {
  FUSER_PERF_SCOPE("GpuLower::Lower::replaceSymbolicSizes");
  std::unordered_map<Val*, Val*> tensor_dim_map;

  // Grab inputs and outputs
  std::vector<TensorView*> inputs_and_outputs;
  for (auto val : fusion->inputs()) {
    if (ir_utils::isTV(val)) {
      inputs_and_outputs.push_back(val->as<TensorView>());
    }
  }
  // Symbolic size is necessary for outputs if there are no inputs.
  // Otherwise infer output sizes from the inputs via expression evaluation.
  if (fusion->inputs().empty()) {
    for (auto val : fusion->outputs()) {
      if (ir_utils::isTV(val)) {
        inputs_and_outputs.push_back(val->as<TensorView>());
      }
    }
  }

  // After ExactMappedExtentSubstitutionPass, different inputs and outputs may
  // have same root domain extents e.g. T1[{i0}, {i2}], T2[{i2}]. When maping
  // {i2}, we want to use the lower labeled tensor size "T1.size[1]", instead of
  // "T2.size[0]".
  std::sort(
      inputs_and_outputs.begin(),
      inputs_and_outputs.end(),
      [](const TensorView* a, const TensorView* b) {
        return a->name() < b->name();
      });

  // Generate map for all tensorview logical domain values to map them to
  // symbolic values. i.e. T0->getLogicalDomain()[0] would map to a named scalar
  // "T0.size[0]". This map will be used when lowering fusion ir to kernel ir.
  for (TensorView* tv : inputs_and_outputs) {
    // Replace the domain with one based on Ti.size[j]
    const std::vector<IterDomain*>& logical_td = tv->getLogicalDomain();

    int64_t dim = 0;
    for (auto id : logical_td) {
      Val* orig_size = id->getMaybeExpandedExtent();
      // Output sizes could have reduction axes, which isn't what gets output.
      // NOLINTNEXTLINE(bugprone-branch-clone)
      if (id->isReduction()) {
        continue;
      } else if (orig_size->isConstScalar()) {
        dim++;
        continue;
      }

      // Currently turn off this part for inputs of segmented fusion,
      //  since FusionKernelRuntime will provide these as integer inputs
      if (tensor_dim_map.find(orig_size) == tensor_dim_map.end() &&
          !orig_size->isFusionInput()) {
        tensor_dim_map[orig_size] = IrBuilder::getItemExpr(
            IrBuilder::getAttrExpr(IrBuilder::metadataExpr(tv), "logical_size"),
            dim++);
      } else {
        dim++;
      }
    }
  }

  // Simplify extents for each exact ValGroup in the fusion
  auto extent_simplification_map = getSimplificationMap(fusion);

  // We now need to map replacement scalars to their targets in tensor_dim_map
  // if they exist. To do this we compose extent_simplification_map with
  // tensor_dim_map.
  //
  // Example:
  //
  //   T0[ i0, i1 ]
  //   T1[ i2, i3 ]
  //   T2[ i4 ]
  //   T3 = T0 + T1
  //   T4 = T2 * full({5}, 0)
  //   ...
  //
  // tensor_dim_map:
  //   i0 = getMetaData[T0].logical_size[0]
  //   i1 = getMetaData[T0].logical_size[1]
  //   i2 = getMetaData[T1].logical_size[0]
  //   i3 = getMetaData[T1].logical_size[1]
  //   i4 = getMetaData[T2].logical_size[0]
  //
  // extent_simplification_map:
  //   i2 = i0
  //   i3 = i1
  //   i4 = 5
  //
  // In this loop, we update the _target_ values like so:
  //
  // extent_simplification_map (updated):
  //   i2 = getMetaData[T0].logical_size[0]
  //   i3 = getMetaData[T0].logical_size[1]
  //   i4 = 5
  //
  // Note that i4's entry is not updated since i4 does not map to a key from
  // tensor_dim_map.
  for (auto& [orig_extent, simplified_extent] : extent_simplification_map) {
    auto it = tensor_dim_map.find(simplified_extent);
    if (it != tensor_dim_map.end()) {
      // Update the mapped extent value
      simplified_extent = it->second;
    }
  }
  // Now add entries from tensor_dim_map, being careful not to overwrite
  // existing replacements.
  //
  // Using the example from above, at this point extent_simplification_map is
  // missing entries for i0 and i1, so we add those directly from
  // tensor_dim_map:
  //
  // extent_simplification_map (updated):
  //   i0 = getMetaData[T0].logical_size[0]
  //   i1 = getMetaData[T0].logical_size[1]
  //   i2 = getMetaData[T0].logical_size[0]
  //   i3 = getMetaData[T0].logical_size[1]
  //   i4 = 5
  for (auto [tensor_dim, meta_expr] : tensor_dim_map) {
    if (extent_simplification_map.count(tensor_dim) == 0) {
      extent_simplification_map[tensor_dim] = meta_expr;
    }
  }

  // Iter domains in the fusion-managed exact mappings may be going to
  // be replaced by another exact-mapped ID, so it'll be reset and
  // recreated. Save a copy here before replacement to fix them up later.
  const auto registered_exact_mappings = fusion->registeredExactMappings();

  auto mutation_map = ir_utils::replaceValue(fusion, extent_simplification_map);

  fusion->resetExactMappings();

  auto get_maybe_mutated = [&mutation_map](IterDomain* id) -> IterDomain* {
    if (auto mutation_map_it = mutation_map.find(id);
        mutation_map_it != mutation_map.end()) {
      id = mutation_map_it->second->as<IterDomain>();
    }
    return id;
  };

  for (const auto& exact_id_group : registered_exact_mappings.disjointSets()) {
    auto first_id = get_maybe_mutated(exact_id_group->front());
    for (IterDomain* id : *exact_id_group) {
      fusion->registerExactMapping(first_id, get_maybe_mutated(id));
    }
  }
}

} // namespace nvfuser
