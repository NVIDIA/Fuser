// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#include <optimization/segment_resize.h>

#include <iter_visitor.h>
#include <ir/internal_nodes.h>

#include <algorithm>
#include <limits>
#include <unordered_set>
#include <vector>

namespace nvfuser::optimization {

namespace {

//! ResizeSegmentSetInserter
class ResizeSegmentSetInserter : public IterVisitor {
 protected:
  using IterVisitor::handle;

  void handle(SliceOp* op) final {
    auto in = op->in()->as<TensorView>();
    if (!in->isFusionInput()) {
      in->cacheBefore(LoadStoreOpType::SegmenterSet);
    }
  }

  void handle(PadOp* op) final {
    auto in = op->in()->as<TensorView>();
    if (!in->isFusionInput()) {
      in->cacheBefore(LoadStoreOpType::SegmenterSet);
    }
  }
};

} // namespace

void ResizeSegmentPass::runPass(Fusion* fusion) {
  ResizeSegmentSetInserter().traverse(fusion);
}

} // namespace nvfuser::optimization
