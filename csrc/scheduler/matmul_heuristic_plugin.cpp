// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on

#include <ir/interface_nodes.h>
#include <mma_type.h>
#include <scheduler/matmul_heuristic_plugin.h>
#include <scheduler/mma_utils.h>
#include <utils.h>

#include <dlfcn.h>

namespace nvfuser {

namespace matmul_heuristic_plugin {

namespace {

class PluginInterface {
 public:
  PluginInterface() {
    filepath_ = getNvFuserEnv("MATMUL_HEURISTIC_PLUGIN");
    if (filepath_ != nullptr) {
      handle_ = dlopen(filepath_, RTLD_LAZY);
    }
  }

  bool available() const {
    return handle_ != nullptr;
  }

  ~PluginInterface() {
    if (handle_ != nullptr) {
      dlclose(handle_);
    }
  }

  KernelConfig getConfig(const ProblemDescription& problem) {
    NVF_ERROR(available());

    if (func_ == nullptr) {
      func_ = (HeuristicFunc*)dlsym(handle_, "getConfig");
      NVF_CHECK(
          func_ != nullptr,
          "Failed to load symbol \"getConfig\" from plugin file ",
          filepath_);
    }

    return (*func_)(problem);
  }

 private:
  char* filepath_ = nullptr;
  void* handle_ = nullptr;
  HeuristicFunc* func_ = nullptr;
} plugin;

// TODO: This should probably be in mma_type.cpp
char dtypeToChar(const DataType& dtype) {
  if (dtype == DataType::Half) {
    return 'H';
  } else if (dtype == DataType::BFloat16) {
    return 'T';
  } else if (dtype == DataType::Float) {
    return 'S';
  }
  NVF_ERROR(false, "Unsupported dtype for matmul: ", dtype);
  return 0;
}

} // namespace

uint8_t layoutToByte(MmaLayout layout) {
  switch (layout) {
    case MmaLayout::NN:
      return 0;
    case MmaLayout::NT:
      return 1;
    case MmaLayout::TN:
      return 2;
    case MmaLayout::TT:
      return 3;
    default:
      return 255;
  }
}

bool hasPlugin() {
  return plugin.available();
}

bool updateMatmulParams(
    MatmulParams& params,
    int64_t M,
    int64_t N,
    int64_t K,
    int64_t batch_size,
    MmaLayout layout,
    const mma_utils::RolesMap& roles_map) {
  if (!hasPlugin()) {
    return false;
  }

  char precision[] = "SSS";
  TensorView* a = roles_map.at(MatmulRole::INPUT_A).front();
  TensorView* b = roles_map.at(MatmulRole::INPUT_B).front();
  NVF_CHECK(
      a->dtype() == b->dtype(), "Differing A and B dtypes not yet supported");
  TensorView* d = roles_map.at(MatmulRole::OUTPUT_D).front();
  precision[0] = dtypeToChar(a->dtype());
  precision[2] = dtypeToChar(d->dtype());

  // Set up problem description
  const ProblemDescription problem{
      {
          .M = (uint32_t)M,
          .N = (uint32_t)N,
          .K = (uint32_t)K,
          .batch_size = (uint32_t)batch_size,
          .layout = layoutToByte(layout),
      },
      .precision = precision};

  const KernelConfig config = plugin.getConfig(problem);

  const auto setTile = [](GemmTile& gemm_tile, const uint16_t(&input)[3]) {
    gemm_tile.m = input[0];
    gemm_tile.n = input[1];
    gemm_tile.k = input[2];
  };

  params.double_buffer_options.smem_double_buffer_stage = config.load_stages;
  setTile(params.tile_sizes.cta_tile, config.cta_tile);
  setTile(params.tile_sizes.warp_tile, config.warp_tile);
  setTile(params.tile_sizes.instruction_tile, config.instruction_tile);
  params.splitk_factor = config.splitk_factor;
  params.grid_swizzle_factor = config.grid_swizzle_factor;
  switch (config.cta_order) {
    case 0:
      params.cta_order = MatmulParams::TileRasterizationOrder::RowMajor;
      break;
    case 1:
      params.cta_order = MatmulParams::TileRasterizationOrder::ColumnMajor;
      break;
    default:
      NVF_ERROR(
          false,
          "Unrecognized cta_order returned by plugin: ",
          config.cta_order,
          ". Expected 0 (row-major) or 1 (column-major)");
  }

  return true;
}

} // namespace matmul_heuristic_plugin

} // namespace nvfuser
