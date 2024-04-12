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
#include <memory>
#include <mutex>

namespace nvfuser {

namespace matmul_heuristic_plugin {

namespace {

std::mutex plugin_mutex;
static class PluginInterface : NonCopyable {
 public:
  PluginInterface() : filepath_(getNvFuserEnv("MATMUL_HEURISTIC_PLUGIN")) {
    if (filepath_ != nullptr) {
      std::lock_guard<std::mutex> lock(plugin_mutex);
      library_handle_ = dlopen(filepath_, RTLD_LAZY);
      NVF_CHECK(
          library_handle_ != nullptr,
          "Error occurred when loading matmul heuristic plugin ",
          filepath_,
          ". Error msg: ",
          dlerror());
    }
  }

  bool available() const {
    return library_handle_ != nullptr;
  }

  ~PluginInterface() {
    if (library_handle_ != nullptr) {
      std::lock_guard<std::mutex> lock(plugin_mutex);
      dlclose(library_handle_);
      library_handle_ = nullptr;
      factory_func_ptr_ = nullptr;
    }
  }

  std::unique_ptr<KernelConfig> makeConfig() {
    NVF_ERROR(available());

    if (factory_func_ptr_ == nullptr) {
      std::lock_guard<std::mutex> lock(plugin_mutex);
      factory_func_ptr_ =
          (KernelConfigFactoryPointer)dlsym(library_handle_, "makeConfig");
      NVF_CHECK(
          factory_func_ptr_ != nullptr,
          "Failed to load symbol \"makeConfig\" from plugin file ",
          filepath_,
          ". Error message: ",
          dlerror());
    }

    return (*factory_func_ptr_)();
  }

 private:
  char* filepath_ = nullptr;
  void* library_handle_ = nullptr;
  KernelConfigFactoryPointer factory_func_ptr_ = nullptr;
} plugin;

std::unique_ptr<KernelConfig> defaultConfigFactory() {
  return plugin.makeConfig();
}

thread_local KernelConfigFactoryPointer config_factory_ptr =
    defaultConfigFactory;

void setKernelConfigFactoryPointer(KernelConfigFactoryPointer func) {
  config_factory_ptr = func == nullptr ? defaultConfigFactory : func;
}

std::unique_ptr<KernelConfig> makeConfig() {
  NVF_ERROR(config_factory_ptr != nullptr);
  return (*config_factory_ptr)();
}

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

} // namespace

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

  std::unique_ptr<KernelConfig> config = makeConfig();
  config->problem.m = (uint32_t)m;
  config->problem.n = (uint32_t)n;
  config->problem.k = (uint32_t)k;
  config->problem.batch_size = (uint32_t)batch_size;
  config->problem.layout =
      KernelConfig::ProblemDescription::Layout(layoutToByte(layout));
  config->problem.precision = precision.c_str();
  config->configure();

  const auto setTile = [](GemmTile& gemm_tile,
                          const KernelConfig::Tile& input) {
    gemm_tile.m = input[0];
    gemm_tile.n = input[1];
    gemm_tile.k = input[2];
  };

  params.double_buffer_options.smem_double_buffer_stage = config->load_stages;
  setTile(params.tile_sizes.cta_tile, config->cta_tile);
  setTile(params.tile_sizes.warp_tile, config->warp_tile);
  setTile(params.tile_sizes.instruction_tile, config->instruction_tile);

  // Update mma macro if necessary to match instruction tile
  MmaMacroEncode menc(params.mma_macro); // this will record the family
  menc.m = config->instruction_tile[0]; // update instruction tile size
  menc.n = config->instruction_tile[1];
  menc.k = config->instruction_tile[2];
  params.mma_macro = menc; // cast back to uint64_t

  params.splitk_factor = config->splitk_factor;
  params.grid_swizzle_factor = config->grid_swizzle_factor;
  switch (config->cta_order) {
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
          config->cta_order,
          ". Expected 0 (row-major) or 1 (column-major)");
  }
  params.double_buffer_options.double_buffer_smem_read =
      config->double_buffer_smem_read;
  params.rotate_ldmatrix_out_of_main_loop =
      config->rotate_ldmatrix_out_of_main_loop;

  // enable double buffering or circular buffering if configured
  params.double_buffer_options.double_buffer_smem_write =
      config->load_stages > 1;

  // async load only for circular buffering (stages > 2)
  params.async_gmem_load_operands = config->load_stages > 2;

  return true;
}

KernelConfigFactoryGuard::KernelConfigFactoryGuard(
    KernelConfigFactoryPointer func)
    : prev_factory_(config_factory_ptr) {
  config_factory_ptr = func;
}

KernelConfigFactoryGuard::~KernelConfigFactoryGuard() {
  config_factory_ptr = prev_factory_;
}

} // namespace matmul_heuristic_plugin

} // namespace nvfuser
