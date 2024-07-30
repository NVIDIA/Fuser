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

  std::unordered_map<Val*, Val*> simplification_map;

  int i = 0;
  for (const ValGroup& group : graph.disjointValSets().disjointSets()) {
    // For each ValGroup, find a single extent to use for all extents of
    // IterDomains in the group. These are chosen in descending order of
    // preference:
    // 1. Constant ints. These might be non-immediate constants
    // 2. Extents of input TVs.
    // 3. Extents of non-input TVs.
    // Within these three classes, we find the IterDomain with the smallest
    // name().
    bool group_is_const = false;
    IterDomain* rep = nullptr;
    for (Val* v : *group) {
      auto* id = dynamic_cast<IterDomain*>(v);
      NVF_ERROR(
          id != nullptr, "Expected only IterDomains in exact graph ValGroups");
      if (rep == nullptr) {
        rep = id;
        continue;
      }
      bool id_is_const = id->isConstInt();
      if (id_is_const) {
        if (!group_is_const || id->name() < rep->name()) {
          rep = id;
          // This lets us avoid repeating the costly isConstInt check
          group_is_const = true;
          continue;
        }
      } else if (id->isFusionInput()) {
        if (group_is_const) {
          continue;
        }
        if (!rep->isFusionInput() || id->name() < rep->name()) {
          rep = id;
          continue;
        }
      } else {
        // id is a non-input TV
        if (group_is_const || rep->isFusionInput()) {
          continue;
        }
        if (id->name() < rep->name()) {
          continue;
        }
      }
    }
    NVF_ERROR(rep != nullptr);
    Val* rep_ext = rep->extent();
    for (Val* v : *group) {
      auto* id = v->as<IterDomain>();
      if (id->isBroadcast() || id->isGatherScatter()) {
        continue;
      }
      Val* ext = id->extent();
      if (!ext->sameAs(rep_ext)) {
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

  // Use a minimal number of sizes from provided tensors.
  auto extent_simplification_map = getSimplificationMap(fusion);
  for (auto extent_entry : extent_simplification_map) {
    auto orig_extent = extent_entry.first;
    auto simplified_extent = extent_entry.second;
    if (tensor_dim_map.count(orig_extent)) {
      if (tensor_dim_map.count(simplified_extent)) {
        tensor_dim_map[orig_extent] = tensor_dim_map[simplified_extent];
      } else {
        tensor_dim_map[orig_extent] = simplified_extent;
      }
    } else {
      tensor_dim_map[orig_extent] = simplified_extent;
    }
  }

  // Run mutation on the fusion with the tensor_dim_map
  ir_utils::replaceValue(fusion, tensor_dim_map);
}

} // namespace nvfuser
