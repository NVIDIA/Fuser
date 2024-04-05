// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on

#include <mma_type.h>
#include <scheduler/mma_utils.h>
#include <scheduler/matmul_heuristic_plugin.h>
#include <utils.h>

#include <dlfcn.h>

namespace nvfuser {

namespace matmul_heuristic_plugin {

namespace {

class PluginInterface {
 public:
  PluginInterface() {
    handle_ = dlopen(getNvFuserEnv("MATMUL_HEURISTIC_PLUGIN"), RTLD_LAZY);
  }

  bool available() const {
    return handle_ != nullptr;
  }

  ~PluginInterface() {
    if (handle_ != nullptr) {
      dlclose(handle_);
    }
  }

  KernelConfig getConfig(ProblemDescription& problem) {
    NVF_ERROR(available());

    if (func_ == nullptr) {
      func_ = (HeuristicFunc*)dlsym(handle_, "getConfig");
    }

    return func_(problem);
  }

 private:
  void* handle_ = nullptr;
  HeuristicFunc* func_ = nullptr;
} plugin;

// TODO: This should probably be in mma_type.cpp
char dtypeToChar(const DataType& dtype) {
  switch (dtype) {
    case DataType::Half:
      return 'H';
    case DataType::BFloat16:
      return 'T';
    case DataType::Float:
      return 'S';
    default:
      NVF_ERROR(false, "Unsupported dtype for matmul: ", dtype);
  }
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
  TensorView* c = roles_map.at(MatmulRole::INPUT_C).front();
  TensorView* d = roles_map.at(MatmulRole::INPUT_D).front();
  precision[0] = dtypeToChar(a->dtype());
  precision[1] = dtypeToChar(c->dtype());
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
      .precision = &precision};

  const KernelConfig config = plugin.getConfig(problem);
}

} // namespace matmul_heuristic_plugin

} // namespace nvfuser
