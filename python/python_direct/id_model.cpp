// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#include <bindings.h>
#include <id_model/id_model.h>
#include <val_graph.h>

namespace nvfuser::python {

namespace {

void bindIdModelClass(py::module_& idm) {
  py::class_<IdModel, std::unique_ptr<IdModel>> id_model(idm, "IdModel");
  id_model.def(
      py::init([](Fusion* fusion,
                  bool build_graphs,
                  bool allow_self_mapping,
                  bool validate) {
        return std::make_unique<IdModel>(
            fusion, build_graphs, allow_self_mapping, validate);
      }),
      py::arg("fusion"),
      py::arg("build_graphs") = false,
      py::arg("allow_self_mapping") = true,
      py::arg("validate") = false,
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
      py::arg("mode"),
      py::return_value_policy::reference,
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

void bindValGraph(py::module_& idm) {
  py::class_<ValGraph, std::unique_ptr<ValGraph>> val_graph(idm, "ValGraph");
  val_graph.def(
      "disjoint_val_sets",
      &ValGraph::disjointValSets,
      py::return_value_policy::reference,
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
      py::arg("val0"),
      py::arg("val1"),
      R"(Maps the two values.

    Parameters
    ----------
    val0 : Val
      The first value to map
    val1 : Val
      The second value to map
    )");
}

void bindDisjointSets(py::module_& id_model) {
  py::class_<DisjointSets<Val*>, std::unique_ptr<DisjointSets<Val*>>>
      disjoint_sets(id_model, "DisjointValSets");
  disjoint_sets.def(
      "__str__",
      &DisjointSets<Val*>::toString,
      R"(
      Returns the string representation of the DisjointSets.
      )");
  disjoint_sets.def(
      "strict_are_mapped",
      &DisjointSets<Val*>::strictAreMapped,
      py::arg("entry0"),
      py::arg("entry1"),
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

void bindIdModel(py::module& nvfuser) {
  py::module_ idm = nvfuser.def_submodule(
      "idm", "This submodule contains all id model operators for NvFuser.");
  bindIdModelClass(idm);
  bindValGraph(idm);
  bindDisjointSets(idm);
}

} // namespace nvfuser::python
