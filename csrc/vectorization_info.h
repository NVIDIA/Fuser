// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#pragma once

#include <c10/macros/Export.h>

#include <ir/all_nodes.h>

namespace nvfuser {

struct VectorizedSetInfo {
  //! Producer of a vectorized set
  TensorView* producer_tv = nullptr;
  //! Consumer of a vectorized set
  TensorView* consumer_tv = nullptr;
  //! Number of elements to vectorize
  int word_size = -1;
  //! Vectorized domain
  IterDomain* vectorized_leaf_id = nullptr;
  //! Right-most allocation dependent domain of the leaf domain for consumer
  IterDomain* vectorized_consumer_alloc_id = nullptr;
  //! Right-most allocation dependent domain of the leaf domain for producer
  IterDomain* vectorized_producer_alloc_id = nullptr;
  //! All of the dependent allocation domains that are contiguously merged
  std::unordered_set<IterDomain*> contig_alloc_ids;
};

} // namespace nvfuser
