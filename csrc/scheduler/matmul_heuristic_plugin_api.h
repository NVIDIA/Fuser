// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#pragma once

#include <array>
#include <cstdint>
#include <memory>

namespace nvfuser {

namespace matmul_heuristic_plugin {

//! This is intended as a minimal interface for enabling matmul heuristics.
//! In order to plug in your own custom heuristic, create a dynamic library
//! defining a subclass of KernelConfig, overriding the `configure` method. This
//! class does not need to be exported from the dll, but you should export a
//! std::unique_ptr<KernelConfig> makeConfig() function that returns a
//! unique_ptr to an object of that type. The `configure` method will be called
//! on that object by nvfuser in order to fill the correct values in the class.
//!
//! If that library is located at /path/to/libfoo.so you can set
//! NVFUSER_MATMUL_HEURISTIC_PLUGIN=/path/to/libfoo.so to use the plugin to
//! determine matmul parameters automatically.

struct KernelConfig {
  //! This is the information available to the plugin to determine the kernel
  //! configuration.
  struct ProblemDescription {
    uint32_t m = -1;
    uint32_t n = -1;
    uint32_t k = -1;
    uint32_t batch_size = -1;
    //! Explicit integer mapping for layout
    enum class Layout {
      NN = 0,
      NT = 1,
      TN = 2,
      TT = 3,
    };
    Layout layout = Layout::TN;
    //! Precision is a string like HSH or TSS indicating input, compute, and
    //! accumulate precision where the letters are mapped to types using the
    //! following mapping:
    //!  B = Int8
    //!  I = Int32
    //!  Q = FP8 (E4M3)
    //!  R = FP8 (E5M2)
    //!  T = BFloat16
    //!  H = Float16
    //!  F = TensorFloat32
    //!  S = Float32
    //!  D = Float64
    //!  C = complex<float>
    //!  Z = complex<double>
    //! Note that some of these are not currently supported by nvFuser.
    const char* precision = "SSS";

    //! Supported vectorization of operands and epilogue inputs (bias)
    struct SupportedVectorization {
      uint8_t a = 16;
      uint8_t b = 16;
      uint8_t epilogue = 16;
    } supported_vec_size;
  } problem;

  using Tile = std::array<uint16_t, 3>;
  Tile cta_tile = {128, 128, 32};
  Tile warp_tile = {64, 64, 32};
  Tile instruction_tile = {16, 16, 16};
  Tile cluster_dims = {1, 1, 1};
  uint16_t splitk_factor = 1;
  uint8_t load_stages = 2;
  // The circular buffering prefetch distance will be set to
  //   load_stages - prefetch_gap
  uint8_t prefetch_gap = 1;
  uint8_t grid_swizzle_factor = 0;
  uint8_t cta_order = 0;
  bool circular_buffer_smem_read = true;
  bool async_gmem_load_operands = true;

 public:
  // This should be overridden to implement the actual heuristic logic
  virtual void configure() = 0;

  // This allows us to use a std::unique_ptr<KernelConfig> and call derived
  // classes' destructors on deletion.
  // See
  // https://clang.llvm.org/extra/clang-tidy/checks/cppcoreguidelines/virtual-class-destructor.html
  virtual ~KernelConfig() = default;
};

} // namespace matmul_heuristic_plugin

} // namespace nvfuser
