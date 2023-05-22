// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#include <optimization/pre_segmenter.h>
#include <optimization/consecutive_cast_pass.h>

namespace nvfuser::optimization {

void PreSegmenter::runPass(Fusion* fusion) {
  ConsecutiveCastPass consecutive_cast_pass;
  consecutive_cast_pass.run(fusion);
}

} // namespace nvfuser::optimization
