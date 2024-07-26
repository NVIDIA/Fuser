// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#pragma once

#include <ir/all_nodes.h>

namespace nvfuser {

struct VectorizedSetInfo {
  //! Producer of a vectorized set
  TensorView* producer_tv = nullptr;
  //! Consumer of a vectorized set
  TensorView* consumer_tv = nullptr;
  //! Number of elements to vectorize
  int64_t word_size = -1;
  //! Vectorized domain
  IterDomain* vectorized_loop_id = nullptr;
  //! Right-most allocation dependent domain of the loop domain for consumer
  IterDomain* vectorized_consumer_alloc_id = nullptr;
  //! Right-most allocation dependent domain of the loop domain for producer
  IterDomain* vectorized_producer_alloc_id = nullptr;
  //! All of the dependent allocation domains that are contiguously merged
  std::unordered_set<IterDomain*> contig_alloc_ids;
};

} // namespace nvfuser
