/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-present NVIDIA CORPORATION &
 * AFFILIATES. All rights reserved. SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#pragma once

#include <cstdint>

#include <runtime/fusion_executor_cache.h>

namespace nvfuser {

struct MlpResult {
  TensorView* linear0;
  TensorView* gelu;
  TensorView* matmul1;
  TensorView* linear1;
  TensorView* output;
};

struct MhaResult {
  TensorView* linear0;
  TensorView* sdpa;
  TensorView* matmul1;
  TensorView* linear1;
  TensorView* output;
};

class DistributedTransformer {
 public:
  DistributedTransformer(
      int64_t num_devices,
      int64_t batch_size,
      int64_t embedding_size,
      int64_t number_heads,
      int64_t sequence_length,
      double dropout_prob = 0.1,
      double sdpa_dropout_prob = 0.1)
      : D(num_devices),
        B(batch_size),
        E(embedding_size),
        H(number_heads),
        S(sequence_length),
        kDropoutProb(dropout_prob),
        kSdpaProb(sdpa_dropout_prob) {}

  std::unique_ptr<FusionExecutorCache> forward(
      DataType dtype,
      bool sequence_parallel = false);
  std::unique_ptr<FusionExecutorCache> backward(DataType dtype);

  MlpResult mlp(
      TensorView* x,
      TensorView* w0,
      TensorView* b0,
      TensorView* w1,
      TensorView* b1,
      const DeviceMesh& mesh,
      bool sequence_parallel = false);

  MhaResult mha(
      TensorView* x,
      TensorView* w0,
      TensorView* b0,
      TensorView* w1,
      TensorView* b1,
      const DeviceMesh& mesh,
      bool sequence_parallel = false);

  std::vector<TensorView*> mlp_backwards(
      TensorView* grad,
      TensorView* x,
      TensorView* mask,
      TensorView* w0,
      TensorView* w1,
      TensorView* linear0,
      const DeviceMesh& mesh);

  std::vector<TensorView*> mha_backwards(
      TensorView* x,
      TensorView* w0,
      TensorView* w1,
      TensorView* mask,
      TensorView* sdpa_output,
      TensorView* sdpa_log_sumexp,
      TensorView* sdpa_seed,
      TensorView* sdpa_offset,
      TensorView* grad,
      TensorView* linear0,
      const DeviceMesh& mesh);

  const int64_t D, B, E, H, S;
  const double kDropoutProb;
  const double kSdpaProb;
  static constexpr double kSdpaScale = 1e-3;
};
} // namespace nvfuser
