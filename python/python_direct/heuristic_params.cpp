// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#include <bindings.h>

#include <scheduler/matmul_heuristic.h>
#include <scheduler/pointwise_heuristic.h>
#include <scheduler/reduction_heuristic.h>

namespace nvfuser::python {

void bindHeuristicParams(py::module& nvfuser) {
  py::class_<LaunchParams> launch_parameters(
      nvfuser, "LaunchParams", py::module_local());
  launch_parameters.def(
      py::init<int64_t, int64_t, int64_t, int64_t, int64_t, int64_t>());
  launch_parameters.def(
      "__repr__", [](const LaunchParams& self) { return self.toString(); });
  launch_parameters.def_property(
      "bdimx",
      [](LaunchParams& self) { return self.bdimx(); },
      [](LaunchParams& self, int64_t val) {
        self.bindUnsafe(val, ParallelType::TIDx);
      });
  launch_parameters.def_property(
      "bdimy",
      [](LaunchParams& self) { return self.bdimy(); },
      [](LaunchParams& self, int64_t val) {
        self.bindUnsafe(val, ParallelType::TIDy);
      });
  launch_parameters.def_property(
      "bdimz",
      [](LaunchParams& self) { return self.bdimz(); },
      [](LaunchParams& self, int64_t val) {
        self.bindUnsafe(val, ParallelType::TIDz);
      });
  launch_parameters.def_property(
      "gdimx",
      [](LaunchParams& self) { return self.gdimx(); },
      [](LaunchParams& self, int64_t val) {
        self.bindUnsafe(val, ParallelType::BIDx);
      });
  launch_parameters.def_property(
      "gdimy",
      [](LaunchParams& self) { return self.gdimy(); },
      [](LaunchParams& self, int64_t val) {
        self.bindUnsafe(val, ParallelType::BIDy);
      });
  launch_parameters.def_property(
      "gdimz",
      [](LaunchParams& self) { return self.gdimz(); },
      [](LaunchParams& self, int64_t val) {
        self.bindUnsafe(val, ParallelType::BIDz);
      });

#define DEFINECLASS(type) py::class_<type>(nvfuser, #type, py::module_local())

#define TOSTRINGTOPLEVEL(type) \
  def("__repr__", [](const type& self) { return toString(self); })
#define TOSTRINGMETHOD(type) \
  def("__repr__", [](const type& self) { return self.toString(); })

#define PARAM(internal_type, name) def_readwrite(#name, &internal_type::name)

  DEFINECLASS(CompileParams)
      .PARAM(CompileParams, index_type)
      .PARAM(CompileParams, maxrregcount)
      .PARAM(CompileParams, enable_magic_zero)
      .PARAM(CompileParams, enable_ptxas_verbose)
      .TOSTRINGMETHOD(CompileParams);

#undef PARAM
#undef INITPARAMS
}

} // namespace nvfuser::python
