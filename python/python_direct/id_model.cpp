// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#include <nanobind/stl/string.h>

#include <bindings.h>
#include <id_model/id_model.h>
#include <val_graph.h>

namespace nvfuser::python {

namespace {

void bindIdModelClass(nb::module_& idm) {
  nb::class_<IdModel> id_model(idm, "IdModel");
  id_model.def(
      "__init__",
      [](IdModel* self,
         Fusion* fusion,
         bool build_graphs,
         bool allow_self_mapping,
         bool validate) {
        new (self) IdModel(fusion, build_graphs, allow_self_mapping, validate);
      },
      nb::arg("fusion"),
      nb::arg("build_graphs") = false,
      nb::arg("allow_self_mapping") = true,
      nb::arg("validate") = false,
      R"(
  Create a new IdModel for the given fusion.

  Parameters
  ----------
  fusion : Fusion
      The fusion to create the IdModel for
  build_graphs : bool
      Whether to build graphs
  allow_self_mapping : bool
      Whether to allow self mapping
  validate : bool
      Whether to validate graphs

  Returns
  -------
  IdModel
      The created IdModel
  )");
  id_model.def(
      "__str__",
      &IdModel::toString,
      R"(
      Returns the string representation of the IdModel.
      )");
  id_model.def(
      "maybe_build_graph",
      &IdModel::maybeBuildGraph,
      nb::arg("mode"),
      nb::rv_policy::reference_internal,
      R"(
      Build a graph if not already built.
      Dependent graphs are also built if not yet done.

      Parameters
      ----------
      mode : IdMappingMode
          The mode to build the graph for

      Returns
      -------
      ValGraph
        The graph built
      )");
}

void bindValGraph(nb::module_& idm) {
  nb::class_<ValGraph> val_graph(idm, "ValGraph");
  val_graph.def(
      "disjoint_val_sets",
      &ValGraph::disjointValSets,
      nb::rv_policy::reference_internal,
      R"(
    Returns the disjoint val set.

    Returns
    -------
    DisjointValSets
      The disjoint val set
    )");
  val_graph.def(
      "__str__",
      &ValGraph::toString,
      R"(
      Returns the string representation of the ValGraph.
      )");
  val_graph.def(
      "map_vals",
      &ValGraph::mapVals,
      nb::arg("val0"),
      nb::arg("val1"),
      R"(Maps the two values.

    Parameters
    ----------
    val0 : Val
      The first value to map
    val1 : Val
      The second value to map
    )");
}

void bindDisjointSets(nb::module_& id_model) {
  nb::class_<DisjointSets<Val*>> disjoint_sets(id_model, "DisjointValSets");
  disjoint_sets.def(
      "__str__",
      &DisjointSets<Val*>::toString,
      R"(
      Returns the string representation of the DisjointSets.
      )");
  disjoint_sets.def(
      "strict_are_mapped",
      &DisjointSets<Val*>::strictAreMapped,
      nb::arg("entry0"),
      nb::arg("entry1"),
      R"(
  Returns if the two entries are strictly mapped.

  Parameters
  ----------
  entry0 : Val
    The first entry to check
  entry1 : Val
    The second entry to check

  Returns
  -------
  bool
    True if the two entries are strictly mapped, False otherwise.
  )");
}

} // namespace

void bindIdModel(nb::module_& nvfuser) {
  nb::module_ idm = nvfuser.def_submodule(
      "idm", "This submodule contains all id model operators for NvFuser.");
  bindIdModelClass(idm);
  bindValGraph(idm);
  bindDisjointSets(idm);
}

} // namespace nvfuser::python
