// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#pragma once

#include <c10/macros/Export.h>

#include <dispatch.h>
#include <ir/all_nodes.h>
#include <kernel_ir.h>

#include <vector>

namespace nvfuser {

//! Collects start and stop offsets of all split root domains. Offsets
//! are zero unless partially split.
class TORCH_CUDA_CU_API PartialSplitMap {
 public:
  void build(Fusion* fusion);

  Val* getStartOffset(IterDomain* root_domain) const;
  Val* getStopOffset(IterDomain* root_domain) const;

 private:
  std::unordered_map<IterDomain*, Val*> start_offset_map_;
  std::unordered_map<IterDomain*, Val*> stop_offset_map_;
};

} // namespace nvfuser
