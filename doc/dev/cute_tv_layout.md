/*
> [!NOTE]
> This file is both a [cpp](../../tests/cpp/tutorial_cute_tv_layout.cpp) and
> a Markdown. You may see some strange symbols in the rendered Markdown.

Tutorial Difficulty: **Low** because it requires knowledge of (shape, stride)
tenosr layouts.

<!--*/
#pragma GCC diagnostic ignored "-Wcomment"
// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on

#include <sstream>
#include <string>

#include <gmock/gmock-matchers.h>
#include <gtest/gtest.h>

#include <tests/cpp/utils.h>
#include <tests/cpp/validator.h>

#include <ops/all_ops.h>
#include <scheduler/mma_utils.h>
#include <scheduler/tools/abstract_tensor.h>
#include <scheduler/tools/inlining.h>
#include <scheduler/utils.h>

#define NOT_IMPLEMENTED GTEST_SKIP() << "Not implemented yet";

namespace nvfuser {

/* -->

# Creating a CuTe TV Layout in NvFuser

## What is CuTe?
CuTe is a layout algebra and a runtime abstraction for representing and handling
nested multi-dimensional layouts of threads, blocks, and data. It is used to
implement high-performance GEMM computation on Nvidia Hardware.

## What is a Layout?
A layout is a tuple of (Shape, Stride). It represents a mapping from shape
coordinate space to a 1D index using the stride. This is similar in concept to
the shape and stride of a PyTorch tensor. A key difference is the shape and
stride of a CuTe layout can be a nested tuple, allowing it to represent more
complex layouts.

# Layout Algebra

1. Coalesce
  * A simplify function to remove redundant shape dimensions without altering
    the Layout's function.
2. Composition
   * The composition of Layouts A and B is `(A o B)(c) := A(B(c))`.
   * Each coordinate is mapped to Layout B then onto Layout A.
   * The intuition behind composition is Layout B is selecting coordinated from
     Layout B.
3. Complement
   * Complement represents the rest of the elements not selected by the
     composition of A and B.
4. Division (Tiling)
   * Logical Divide of Layouts A and B returns the composition of A and B and
     its complement. i.e., the elements of Layout B selected by Layout A.
   * `logical_divide(A, B) = composition(A, (B, complement(B)))`
   * Use case is selecting a tile from a matrix for a given CTA.
5. Product (Tiling)
   * Logical Product of Layouts A and B returns the original Layout A and a
     transformed Layout B where each element is a replication of Layout A.
   * `logical_product(A, B) = (A, composition(complement(A), B))`

Reference: [CuTe Layout Algebra](https://docs.nvidia.com/cutlass/media/docs/cpp/cute/02_layout_algebra.html)

## What is Thread-Value (TV) Layout?
* The Thread-Value layout is a mapping of threads and values to an index.
`((thread_shape, value_shape), (thread_stride, value_stride))`

The main usage of TV layout is mapping the threads and its values of a CTA to
the index of a Tensor.

<!-- */ //-->\
```cpp
using CuTeTutorial = NVFuserTest;
TEST_F(CuTeTutorial, ThreadLayout) {
  NVFUSER_TEST_CUDA_ARCH_GUARD(9, 0);
  const auto dtype = DataType::BFloat16;

  // Fusion Definition
  std::unique_ptr<Fusion> fusion_ptr = std::make_unique<Fusion>();
  Fusion* fusion = fusion_ptr.get();
  FusionGuard fg(fusion);

  TensorView* tv0 = makeContigConcreteTensor({-1, -1}, dtype); // M, K
  fusion->addInput(tv0);
  TensorView* tv1 = set(tv0);
  fusion->addOutput(tv1);

  constexpr int dim0 = 8192, dim1 = 8192;
  auto options = at::TensorOptions().dtype(at::kBFloat16).device(at::kCUDA, 0);
  at::Tensor at_tv0 = at::randn({dim0, dim1}, options);

  FusionExecutorCache executor_cache(std::move(fusion_ptr));
  auto cg_outputs = executor_cache.runFusionWithInputs({at_tv0});
  NVF_CHECK(at::allclose(cg_outputs[0].as<at::Tensor>(), at_tv0));
}
/*
```
<!--*/
} // namespace nvfuser
// \-->
