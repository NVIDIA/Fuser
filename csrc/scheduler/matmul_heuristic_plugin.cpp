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
#include <cstdint>

namespace nvfuser {

namespace matmul_heuristic_plugin {

namespace {

//! Defines HeuristicFuncPtr as type of the "getConfig" symbol
typedef void (*HeuristicFuncPtr)(KernelConfig*, const ProblemDescription*);

thread_local class PluginInterface {
 public:
  PluginInterface() : filepath_(getNvFuserEnv("MATMUL_HEURISTIC_PLUGIN")) {
    if (filepath_ != nullptr) {
      handle_ = dlopen(filepath_, RTLD_LAZY);
      NVF_CHECK(
          handle_ != nullptr,
          "Error occurred when loading matmul heuristic plugin ",
          filepath_,
          ". Error msg: ",
          dlerror());
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

  void getConfig(KernelConfig* config, const ProblemDescription* problem) {
    NVF_ERROR(available());

    if (func_ == nullptr) {
      func_ = (HeuristicFuncPtr)dlsym(handle_, "getConfig");
      NVF_CHECK(
          func_ != nullptr,
          "Failed to load symbol \"getConfig\" from plugin file ",
          filepath_,
          ". Error message: ",
          dlerror());
    }

    NVF_ERROR(config != nullptr);
    NVF_ERROR(problem != nullptr);

    (*func_)(config, problem);
  }

 private:
  char* filepath_ = nullptr;
  void* handle_ = nullptr;
  HeuristicFuncPtr func_ = nullptr;
} plugin;

//! Utility to standardize conversion of MmaLayout to uint8_t
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

//! Utility to standardize conversion of MmaLayout to uint8_t
uint8_t layoutToByte(MmaLayout layout);

} // namespace

struct ProblemDescription {
  uint32_t m;
  uint32_t n;
  uint32_t k;
  uint32_t batch_size;
  // layout is in row-major and takes values 0 thru 3 in order NN NT TN TT
  uint8_t layout;
  const char* precision; // e.g. HSH, TST, HSS, etc.
};
uint32_t getProblemM(const ProblemDescription* problem) {
  return problem->m;
}
uint32_t getProblemN(const ProblemDescription* problem) {
  return problem->n;
}
uint32_t getProblemK(const ProblemDescription* problem) {
  return problem->k;
}
uint32_t getProblemBatchSize(const ProblemDescription* problem) {
  return problem->batch_size;
}
uint8_t getProblemLayout(const ProblemDescription* problem) {
  return problem->layout;
}
const char* getProblemPrecision(const ProblemDescription* problem) {
  return problem->precision;
}

struct KernelConfig {
  using Tile = std::array<uint16_t, 3>;
  Tile cta_tile;
  Tile warp_tile;
  Tile instruction_tile;
  uint16_t splitk_factor;
  uint8_t load_stages;
  uint8_t grid_swizzle_factor;
  uint8_t cta_order; // 0 for row major, 1 for column major
};
void setCtaTile(KernelConfig* config, uint16_t m, uint16_t n, uint16_t k) {
  config->cta_tile[0] = m;
  config->cta_tile[1] = n;
  config->cta_tile[2] = k;
}
void setWarpTile(KernelConfig* config, uint16_t m, uint16_t n, uint16_t k) {
  config->warp_tile[0] = m;
  config->warp_tile[1] = n;
  config->warp_tile[2] = k;
}
void setInstructionTile(
    KernelConfig* config,
    uint16_t m,
    uint16_t n,
    uint16_t k) {
  config->instruction_tile[0] = m;
  config->instruction_tile[1] = n;
  config->instruction_tile[2] = k;
}
void setSplitKFactor(KernelConfig* config, uint16_t f) {
  config->splitk_factor = f;
}
void setLoadStages(KernelConfig* config, uint8_t s) {
  config->load_stages = s;
}
void setGridSwizzleFactor(KernelConfig* config, uint8_t g) {
  config->grid_swizzle_factor = g;
}
void setCtaOrder(KernelConfig* config, uint8_t o) {
  config->cta_order = o;
}

bool hasPlugin() {
  return plugin.available();
}

bool updateMatmulParams(
    MatmulParams& params,
    int64_t m,
    int64_t n,
    int64_t k,
    int64_t batch_size,
    MmaLayout layout,
    const mma_utils::RolesMap& roles_map) {
  if (!hasPlugin()) {
    return false;
  }

  std::string precision = "SSS";
  TensorView* a = roles_map.at(MatmulRole::INPUT_A).front();
  TensorView* b = roles_map.at(MatmulRole::INPUT_B).front();
  NVF_CHECK(
      a->dtype() == b->dtype(), "Differing A and B dtypes not yet supported");
  TensorView* d = roles_map.at(MatmulRole::OUTPUT_D).front();
  precision[0] = mma_utils::dtypeToChar(a->dtype());
  // NOTE: this assumes compute type is Float
  precision[2] = mma_utils::dtypeToChar(d->dtype());

  // Set up problem description
  const ProblemDescription problem{
      .m = (uint32_t)m,
      .n = (uint32_t)n,
      .k = (uint32_t)k,
      .batch_size = (uint32_t)batch_size,
      .layout = layoutToByte(layout),
      .precision = precision.c_str()};

  KernelConfig config{
      .cta_tile =
          {(uint16_t)params.tile_sizes.cta_tile.m,
           (uint16_t)params.tile_sizes.cta_tile.n,
           (uint16_t)params.tile_sizes.cta_tile.k},
      .warp_tile =
          {(uint16_t)params.tile_sizes.warp_tile.m,
           (uint16_t)params.tile_sizes.warp_tile.n,
           (uint16_t)params.tile_sizes.warp_tile.k},
      .instruction_tile =
          {(uint16_t)params.tile_sizes.instruction_tile.m,
           (uint16_t)params.tile_sizes.instruction_tile.n,
           (uint16_t)params.tile_sizes.instruction_tile.k},
      .splitk_factor = (uint16_t)params.splitk_factor,
      .load_stages =
          (uint8_t)params.double_buffer_options.smem_double_buffer_stage,
      .grid_swizzle_factor = (uint8_t)params.grid_swizzle_factor,
      .cta_order = (uint8_t)toUnderlying(params.cta_order),
  };
  plugin.getConfig(&config, &problem);

  const auto setTile = [](GemmTile& gemm_tile,
                          const KernelConfig::Tile& input) {
    gemm_tile.m = input[0];
    gemm_tile.n = input[1];
    gemm_tile.k = input[2];
  };

  params.double_buffer_options.smem_double_buffer_stage = config.load_stages;
  setTile(params.tile_sizes.cta_tile, config.cta_tile);
  setTile(params.tile_sizes.warp_tile, config.warp_tile);
  setTile(params.tile_sizes.instruction_tile, config.instruction_tile);

  // Update mma macro if necessary to match instruction tile
  MmaMacroEncode menc(params.mma_macro); // this will record the family
  menc.m = config.instruction_tile[0]; // update instruction tile size
  menc.n = config.instruction_tile[1];
  menc.k = config.instruction_tile[2];
  params.mma_macro = menc; // cast back to uint64_t

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
