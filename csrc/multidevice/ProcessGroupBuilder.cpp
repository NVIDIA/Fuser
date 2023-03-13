// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#ifdef USE_DISTRIBUTED

#include <multidevice/ProcessGroupBuilder.hpp>
#include <torch/csrc/distributed/c10d/ProcessGroupGloo.hpp>
#include <torch/csrc/distributed/c10d/ProcessGroupNCCL.hpp>

namespace nvfuser {

c10::intrusive_ptr<c10d::Backend> ProcessGroupBuilder::getProcessGroup(
    std::string backend,
    const c10::intrusive_ptr<c10d::Store>& store,
    int rank,
    int size) {
#ifdef USE_C10D_NCCL
  if (backend == "nccl") {
    auto pg_opts = c10::make_intrusive<::c10d::ProcessGroupNCCL::Options>();
    return c10::make_intrusive<::c10d::ProcessGroupNCCL>(
        store, rank, size, pg_opts);
  }
#endif

#ifdef USE_C10D_GLOO
  if (backend == "gloo") {
    auto pg_opts = c10d::ProcessGroupGloo::Options::create();
    return c10::make_intrusive<::c10d::ProcessGroupGloo>(
        store, rank, size, pg_opts);
  }
#endif
  TORCH_CHECK(false, "no dist backend available");
}

int parseEnv(int& grank, int& gsize) {
  char* env;

  // retrieves the rank of the current process
  env = std::getenv("OMPI_COMM_WORLD_RANK");
  if (!env) {
    env = std::getenv("WORLD_RANK");
    if (!env) {
      return 1;
    }
  }
  grank = std::atoi(env);

  // retrieves the size of the communicator
  env = std::getenv("OMPI_COMM_WORLD_SIZE");
  if (!env) {
    env = std::getenv("WORLD_SIZE");
    if (!env) {
      return 1;
    }
  }
  gsize = std::atoi(env);
  return 0;
}

} // namespace nvfuser

#endif
