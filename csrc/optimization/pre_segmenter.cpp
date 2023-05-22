// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#include <optimization/consecutive_cast_pass.h>
#include <optimization/pre_segmenter.h>

namespace nvfuser::optimization {

void PreSegmenter::runPass(Fusion* fusion) {
  // TODO: boilerplate code needed to enable on/off switch
  if (!flipEnabled(false)) {
    return;
  }
  // removes consecutive cast operations
  ConsecutiveCastPass consecutive_cast_pass;
  consecutive_cast_pass.run(fusion);
}

} // namespace nvfuser::optimization
