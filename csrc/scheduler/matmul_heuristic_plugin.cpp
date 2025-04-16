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
#include <sys_utils.h>
#include <utils.h>

#include <cstdint>
#include <memory>
#include <mutex>

namespace nvfuser {

namespace matmul_heuristic_plugin {

namespace {

std::mutex plugin_mutex;
typedef std::unique_ptr<KernelConfig> (*KernelConfigFactoryPointer)();
static class PluginInterface : LibraryLoader {
 public:
  PluginInterface() {
    const char* envvar = getNvFuserEnv("MATMUL_HEURISTIC_PLUGIN");
    if (envvar != nullptr) {
      setFilename(envvar);
    }
  }

  ~PluginInterface() = default;

  bool available() const {
    return !filename().empty();
  }

  std::unique_ptr<KernelConfig> getConfig() {
    NVF_ERROR(available());
    if (factory_func_ptr_ == nullptr) {
      std::lock_guard<std::mutex> lock(plugin_mutex);
      factory_func_ptr_ = (KernelConfigFactoryPointer)getSymbol("makeConfig");
    }
    return (*factory_func_ptr_)();
  }

 private:
  KernelConfigFactoryPointer factory_func_ptr_ = nullptr;
} plugin;

std::unique_ptr<KernelConfig> defaultConfigFactory() {
  return plugin.getConfig();
}

thread_local KernelConfigFactory config_factory = defaultConfigFactory;
// We just need to check for the special case of the default factory, so that we
// can further check for a user-provided plugin. However, comparing
// std::functions directly is difficult to do well. See
// https://stackoverflow.com/questions/20833453/comparing-stdfunctions-for-equality
// Instead, the following flag indicates whether the default factory has been
// overridden.
thread_local bool config_factory_modified = false;

//! Utility to standardize conversion of inner dimensions to uint8_t
//! This uses the 2-operand layout format:
//!   NN: 0
//!   NT: 1
//!   TN: 2
//!   TT: 3
uint8_t innerDimsToByte(const mma_utils::MatmulOperandInnerDims& inner_dims) {
  bool A_K_inner = inner_dims.empty() || inner_dims.front() == MatmulDimRole::K;
  bool B_K_inner = inner_dims.empty() || inner_dims.back() == MatmulDimRole::K;
  if (A_K_inner && B_K_inner) {
    return 0;
  } else if (A_K_inner && !B_K_inner) {
    return 1;
  } else if (!A_K_inner && B_K_inner) {
    return 2;
  } else {
    return 3;
  }
}

std::string rolesToPrecisionString(
    const mma_utils::TensorRolesMap& tensor_roles) {
  std::string precision = "   ";
  const std::vector<TensorView*>& a_operands =
      tensor_roles.at(MatmulTensorRole::OPERAND_A);
  NVF_ERROR(!a_operands.empty(), "We currently require at least one A operand");
  const std::vector<TensorView*>& b_operands =
      tensor_roles.at(MatmulTensorRole::OPERAND_B);
  NVF_ERROR(!b_operands.empty(), "We currently require at least one B operand");
  TensorView* a = a_operands.front();
  TensorView* b = b_operands.front();
  NVF_CHECK(
      a->dtype() == b->dtype(), "Differing A and B dtypes not yet supported");
  TensorView* d = tensor_roles.at(MatmulTensorRole::OUTPUT).front();
  precision[0] = mma_utils::dtypeToChar(a->dtype());
  // NOTE: this assumes compute type is Float
  precision[1] = 'S';
  precision[2] = mma_utils::dtypeToChar(d->dtype());
  return precision;
}

void fillProblemDescription(
    KernelConfig::ProblemDescription& problem,
    int64_t m,
    int64_t n,
    int64_t k,
    int64_t batch_size,
    const mma_utils::MatmulOperandInnerDims& inner_dims,
    const char* precision) {
  problem.m = (uint32_t)m;
  problem.n = (uint32_t)n;
  problem.k = (uint32_t)k;
  problem.batch_size = (uint32_t)batch_size;
  problem.layout =
      KernelConfig::ProblemDescription::Layout(innerDimsToByte(inner_dims));
  problem.precision = precision;
}

void copyParamsToConfig(KernelConfig* config, const MatmulParams* mparams) {
  const auto setConfigTile = [](KernelConfig::Tile& output,
                                const GemmTile& gemm_tile) {
    output[0] = gemm_tile.m;
    output[1] = gemm_tile.n;
    output[2] = gemm_tile.k;
  };
  config->load_stages =
      mparams->circular_buffer_options.smem_circular_buffer_stage;
  config->prefetch_gap =
      mparams->circular_buffer_options.smem_circular_buffer_prefetch_gap;
  config->async_gmem_load_operands = mparams->async_gmem_load_operands;
  setConfigTile(config->cta_tile, mparams->tile_sizes.cta_tile);
  setConfigTile(config->warp_tile, mparams->tile_sizes.warp_tile);
  setConfigTile(config->instruction_tile, getMmaOpShape(mparams->mma_macro));
  config->cluster_dims[0] = mparams->cluster_dims.x;
  config->cluster_dims[1] = mparams->cluster_dims.y;
  config->cluster_dims[2] = mparams->cluster_dims.z;
  config->splitk_factor = mparams->splitk_factor;
  config->grid_swizzle_factor = mparams->grid_traversal_factor.first;
  config->cta_order =
      mparams->cta_order == MatmulParams::TileRasterizationOrder::RowMajor ? 0
                                                                           : 1;
  config->circular_buffer_smem_read =
      mparams->circular_buffer_options.circular_buffer_smem_read;
  config->problem.supported_vec_size.a = (uint8_t)mparams->supported_vec_size.a;
  config->problem.supported_vec_size.b = (uint8_t)mparams->supported_vec_size.b;
  config->problem.supported_vec_size.epilogue =
      (uint8_t)mparams->supported_vec_size.epilogue;
}

void copyConfigToParams(MatmulParams* mparams, const KernelConfig* config) {
  const auto setGemmTile = [](GemmTile& gemm_tile,
                              const KernelConfig::Tile& input) {
    gemm_tile.m = input[0];
    gemm_tile.n = input[1];
    gemm_tile.k = input[2];
  };
  setGemmTile(mparams->tile_sizes.cta_tile, config->cta_tile);
  setGemmTile(mparams->tile_sizes.warp_tile, config->warp_tile);
  mparams->cluster_dims.x = config->cluster_dims[0];
  mparams->cluster_dims.y = config->cluster_dims[1];
  mparams->cluster_dims.z = config->cluster_dims[2];
  mparams->circular_buffer_options.smem_circular_buffer_stage =
      config->load_stages;
  mparams->circular_buffer_options.smem_circular_buffer_prefetch_gap =
      config->prefetch_gap;
  mparams->async_gmem_load_operands = config->async_gmem_load_operands;
  // Update mma macro if necessary to match provided instruction tile
  MmaMacroEncode menc(mparams->mma_macro); // this will record the family
  menc.m = config->instruction_tile[0]; // update instruction tile size
  menc.n = config->instruction_tile[1];
  menc.k = config->instruction_tile[2];
  mparams->mma_macro = menc; // cast back to uint64_t
  mparams->splitk_factor = config->splitk_factor;
  mparams->grid_traversal_factor.first = config->grid_swizzle_factor;
  switch (config->cta_order) {
    case 0:
      mparams->cta_order = MatmulParams::TileRasterizationOrder::RowMajor;
      break;
    case 1:
      mparams->cta_order = MatmulParams::TileRasterizationOrder::ColumnMajor;
      break;
    default:
      NVF_THROW(
          "Unrecognized cta_order returned by plugin: ",
          config->cta_order,
          ". Expected 0 (row-major) or 1 (column-major)");
  }
  mparams->circular_buffer_options.circular_buffer_smem_read =
      config->circular_buffer_smem_read;

  // enable circular buffering if configured
  mparams->circular_buffer_options.circular_buffer_smem_write =
      config->load_stages > 1;
}

} // namespace

bool updateMatmulParams(
    MatmulParams* mparams,
    int64_t m,
    int64_t n,
    int64_t k,
    int64_t batch_size,
    const mma_utils::MatmulOperandInnerDims& inner_dims,
    const mma_utils::TensorRolesMap& tensor_roles) {
  if (!hasPlugin()) {
    return false;
  }

  // Use factory function to create an empty config
  std::unique_ptr<KernelConfig> config = config_factory();

  // Set previous heuristic values so they are available to the plugin
  copyParamsToConfig(config.get(), mparams);

  // The heuristic must know the input shapes, precision, and layout.
  std::string precision = rolesToPrecisionString(tensor_roles);
  fillProblemDescription(
      config->problem, m, n, k, batch_size, inner_dims, precision.c_str());

  // Execute the user-provided heuristic
  config->configure();

  // Load values from config back into mparams
  copyConfigToParams(mparams, config.get());

  return true;
}

bool hasPlugin() {
  // If we have overridden the default config factory, we will count that as a
  // plugin. Otherwise, we need to check that the dynamically loaded plugin is
  // actually available.

  // To check whether we have set a non-default factory, find the address of
  // config_factory and compare it to defaultConfigFactory.
  return config_factory_modified || plugin.available();
}

KernelConfigFactoryGuard::KernelConfigFactoryGuard(KernelConfigFactory func)
    : prev_factory_(config_factory),
      prev_factory_modified_(config_factory_modified) {
  config_factory = func;
  config_factory_modified = true;
}

KernelConfigFactoryGuard::~KernelConfigFactoryGuard() {
  config_factory = prev_factory_;
  config_factory_modified = prev_factory_modified_;
}

} // namespace matmul_heuristic_plugin

} // namespace nvfuser
