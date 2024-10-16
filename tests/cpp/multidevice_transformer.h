/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION &
 * AFFILIATES. All rights reserved. SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#pragma once

#include <stdint.h>

#include <runtime/fusion_executor_cache.h>

namespace nvfuser {
struct MHAQKVResult {
  TensorView* linear0;
  std::vector<TensorView*> qkv;
};

class DistributedTransformer {
 public:
  DistributedTransformer(
      int64_t num_devices,
      int64_t batch_size,
      int64_t embedding_size,
      int64_t number_heads,
      int64_t sequence_length)
      : D(num_devices),
        B(batch_size),
        E(embedding_size),
        H(number_heads),
        S(sequence_length) {}

  std::unique_ptr<FusionExecutorCache> forward(DataType dtype);
  std::unique_ptr<FusionExecutorCache> backward(DataType dtype);

  std::vector<TensorView*> mlp(
    TensorView* x,
    TensorView* w0,
    TensorView* b0,
    TensorView* w1,
    TensorView* b1,
    const DeviceMesh& mesh);

std::vector<TensorView*> mha(
    TensorView* x,
    TensorView* w0,
    TensorView* b0,
    TensorView* w1,
    TensorView* b1,
    const DeviceMesh& mesh);

// Backwards MLP block. Recomputes linear0 and gelu
// if either isn't provided as input.
std::vector<TensorView*> mlp_backwards(
    TensorView* grad,
    TensorView* x,
    TensorView* mask,
    TensorView* w0,
    TensorView* b0,
    TensorView* w1,
    const DeviceMesh& mesh,
    TensorView* linear0 = nullptr,
    TensorView* gelu = nullptr);

std::vector<TensorView*> mha_backwards(
    TensorView* x,
    TensorView* w0,
    TensorView* b0,
    TensorView* w1,
    TensorView* mask,
    TensorView* sdpa_output,
    TensorView* sdpa_log_sumexp,
    TensorView* sdpa_seed,
    TensorView* sdpa_offset,
    TensorView* grad,
    const std::vector<TensorView*>& qkv,
    const DeviceMesh& mesh);

MHAQKVResult mha_qkv(
    TensorView* x,
    TensorView* w0,
    TensorView* b0,
    const DeviceMesh& mesh);

  int64_t D, B, E, H, S;
  static constexpr double kDropoutProb = 0.1, kParamScale = 0.02, kSdpaProb = 0.0,
                   kSdpaScale = 1e-3;

};
} // namespace nvfuser
