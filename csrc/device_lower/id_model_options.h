// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#pragma once

#include <sstream>

#include "id_model/utils.h"
#include "options.h"

namespace nvfuser {

class IdModelOptions {
 public:
  IdModelOptions()
      : tensor_indexer_enabled_(isOptionEnabled(EnableOption::IdModel)) {}

  void setTensorIndexer(bool b) {
    tensor_indexer_enabled_ = b;
  }

  bool isTensorIndexerEnabled() const {
    return tensor_indexer_enabled_;
  }

  std::string toString() const {
    auto bool2str = [](bool b) { return b ? "true" : "false"; };

    std::stringstream ss;
    ss << "enable_tensor_indexer=" << bool2str(tensor_indexer_enabled_);
    return ss.str();
  }

 private:
  // Enable TensorIndexer
  bool tensor_indexer_enabled_ = false;
};

} // namespace nvfuser
