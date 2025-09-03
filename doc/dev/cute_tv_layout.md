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

  DisableOptionsGuard disable_options_guard;
  DisableOptionsGuard::getCurOptions().set(DisableOption::MagicZero);

  // Fusion Definition
  std::unique_ptr<Fusion> fusion_ptr = std::make_unique<Fusion>();
  Fusion* fusion = fusion_ptr.get();
  FusionGuard fg(fusion);

  constexpr int dim0 = 4, dim1 = 4;
  TensorView* tv0 = makeContigConcreteTensor({dim0, dim1}, dtype);
  fusion->addInput(tv0);
  TensorView* tv1 = set(tv0);
  fusion->addOutput(tv1);

  // Set the allocation domain to column-major.
  tv0->setAllocationDomain({tv0->axis(1), tv0->axis(0)}, /*new_contiguity=*/true);
  tv1->reorder({1, 0}); // traverse rows then column.
  tv1->setAllocationDomain(tv1->getLoopDomain(), /*new_contiguity=*/true);

  fusion->printKernel();
  /*
  // The input and output tensors are column-major with shape (8, 4) and stride (1, 8).
  __global__ void CUDAGeneratedKernel(Tensor<__bfloat, 2, 2> T0, Tensor<__bfloat, 2, 2> T1) {
    #pragma unroll
    for(nvfuser_index_t i0 = 0LL; i0 < 4LL; ++i0) {
      nvfuser_index_t i1;
      i1 = 8LL * i0;
      #pragma unroll
      for(nvfuser_index_t i2 = 0LL; i2 < 8LL; ++i2) {
        nvfuser_index_t i3;
        i3 = i1 + i2;
        T1[i3]
          = T0[i3];
      }
    }
  }
  */

  tv1->split(-1, 2);
  tv1->split(0, 2);
  tv1->reorder({{0, 2}, {1, 0}, {2, 1}});

  TransformPropagator propagator(tv1);
  MaxLogicalDomainInfoSpanningTree(tv1).traverse(&propagator);

  inlineMost();

  fusion->printKernel();
  /*
  // After splitting and reordering the loop domain of TV1, the loop domain is [2, 2, 2, 2].
  // The strides of the loop domain are [4, 2, 8, 1].
  //
  // How to create CuTe TV layout from NvFuser TensorDomain?
  //   1) Reverse ordering from inner to outer loops.
  //   2) Gather modes 0 and 1 and 2 and 3 together.
  // This creates the shape ((2, 2), (2, 2)) and the stride ((1, 8), (2, 4)),
  // which corresponds with the CuTe Thread-Value Layout.
  __global__ void CUDAGeneratedKernel(Tensor<__bfloat, 2, 2> T0, Tensor<__bfloat, 2, 2> T1) {
    #pragma unroll
    for(nvfuser_index_t i0 = 0LL; i0 < 2LL; ++i0) {
      nvfuser_index_t i1;
      i1 = 4LL * i0;
      #pragma unroll
      for(nvfuser_index_t i2 = 0LL; i2 < 2LL; ++i2) {
        nvfuser_index_t i3;
        i3 = 2LL * i2;
        nvfuser_index_t i4;
        i4 = i1 + i3;
        #pragma unroll
        for(nvfuser_index_t i5 = 0LL; i5 < 2LL; ++i5) {
          nvfuser_index_t i6;
          i6 = i4 + (8LL * i5);
          bool b7;
          b7 = (i0 + (2LL * i5)) < 4LL;
          #pragma unroll
          for(nvfuser_index_t i8 = 0LL; i8 < 2LL; ++i8) {
            nvfuser_index_t i9;
            i9 = i6 + i8;
            if ((b7 && ((i3 + i8) < 4LL))) {
              T1[i9]
                 = T0[i9];
            }
          }
        }
      }
    }
  }
  */

  auto options = at::TensorOptions().dtype(at::kBFloat16).device(at::kCUDA, 0);
  at::Tensor at_tv0 = at::randn({dim0, dim1}, options).as_strided({dim0, dim1}, {1, dim0});

  KernelExecutor ke;
  ke.compile(fusion, {at_tv0});
  kir::Kernel* kernel = ke.compiledKernel()->kernel();
  ASSERT_TRUE(kernel != nullptr);
  auto cg_outputs = ke.run({at_tv0});
  NVF_CHECK(at::allclose(cg_outputs[0].as<at::Tensor>(), at_tv0));
}
/*
```
<!--*/
} // namespace nvfuser
// \-->
