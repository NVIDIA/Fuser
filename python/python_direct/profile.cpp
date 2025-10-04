// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#include <bindings.h>
#include <fusion_profiler.h>

namespace nvfuser::python {

namespace {

void bindFusionProfile(py::module& nvfuser) {
  py::class_<FusionProfile> fusion_prof(nvfuser, "FusionProfile");
  fusion_prof.def_property_readonly(
      "verbose", [](FusionProfile& self) { return self.verbose; }, R"(
Returns the verbosity of the fusion profile.
)");
  fusion_prof.def_property_readonly(
      "fusion_id", [](FusionProfile& self) { return self.fusion_id; }, R"(
Returns the fusion id of the fusion profile.
)");
  fusion_prof.def_property_readonly(
      "segments", [](FusionProfile& self) { return self.segments; }, R"(
Returns the segments in the fusion profile.
)");
  fusion_prof.def_property_readonly(
      "cuda_evt_time_ms",
      [](FusionProfile& self) { return self.cuda_evt_time_ms; },
      R"(
Returns the CUDA event time in milliseconds of the fusion profile.
)");
  fusion_prof.def_property_readonly(
      "host_time_ms", [](FusionProfile& self) { return self.host_time_ms; }, R"(
Returns the host time in milliseconds of the fusion profile.
)");
  fusion_prof.def_property_readonly(
      "compile_time_ms",
      [](FusionProfile& self) { return self.compile_time_ms; },
      R"(
Returns the compile time in milliseconds of the fusion profile.
)");
  fusion_prof.def_property_readonly(
      "kernel_time_ms",
      [](FusionProfile& self) { return self.kernel_time_ms; },
      R"(
Returns the kernel time in milliseconds of the fusion profile.
)");
  fusion_prof.def_property_readonly(
      "effective_bandwidth_gbs",
      [](FusionProfile& self) { return self.effective_bandwidth_gbs; },
      R"(
Returns the effective bandwidth in gigabytes per second of the fusion profile.
)");
  fusion_prof.def_property_readonly(
      "percentage_peak_bandwith",
      [](FusionProfile& self) { return self.percentage_peak_bandwidth; },
      R"(
Returns the percentage of peak bandwidth of the fusion profile.
)");
  fusion_prof.def_property_readonly(
      "input_bytes", [](FusionProfile& self) { return self.input_bytes; }, R"(
Returns the input bytes of the fusion profile.
)");
  fusion_prof.def_property_readonly(
      "output_bytes", [](FusionProfile& self) { return self.output_bytes; }, R"(
Returns the output bytes of the fusion profile.
)");
  fusion_prof.def_property_readonly(
      "kernel_profiles",
      [](FusionProfile& self) { return self.kernel_profiles; },
      R"(
Returns the kernel profiles of the fusion profile.
)");
}

} // namespace

void bindProfile(py::module& nvfuser) {}

} // namespace nvfuser::python
