// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
// Utility macro for this file

// MMA instruction wrappers:
//  The wrappers are subroutines that implement matrix of size
//    A(M,K) X B(K,N) = C(M,N)
//  The naming of the wrappers follow similar naming conventions
//    as the mma instructions.
//  All the mma macros follow the namespace and naming like
//    Arch::M (M-dim) N (N-dim) K(K-dim) (Layout), eg.
//    Volta::M16N16K4TT,
//  with the dimensions describing the size of the sub-matrices being
//   multiplied by this wrapper.
//  see [Operand Layout Convention] in mma_type.h for details on the layout
//   notation.
namespace Volta {

// MMA instruction wrappers (sm_70+):
// The instruction wrappers below are quarter-warp macros, which currently
//  nvfuser doesn't explicitly model.
// So they are currently only meant to be
//  used as building blocks in warp level mma macros

//  8x8x4 mma instruction, per quarter warp (8 threads), fp32 accumulate
//  per thread register:
//   A[4] x B[4] -> C[8]

__device__ inline void M16N16K4TT(
    Array<float, 8, 8>& C,
    Array<unsigned, 2, 2>& A,
    Array<unsigned, 2, 2>& B) {
  asm("mma.sync.aligned.m8n8k4.row.row.f32.f16.f16.f32 {%0,%1,%2,%3,%4,%5,%6,%7}, {%8,%9}, {%10,%11}, {%12,%13,%14,%15,%16,%17,%18,%19};\n"
      : "=f"(C[0]),
        "=f"(C[1]),
        "=f"(C[2]),
        "=f"(C[3]),
        "=f"(C[4]),
        "=f"(C[5]),
        "=f"(C[6]),
        "=f"(C[7])
      : "r"(A[0]),
        "r"(A[1]),
        "r"(B[0]),
        "r"(B[1]),
        "f"(C[0]),
        "f"(C[1]),
        "f"(C[2]),
        "f"(C[3]),
        "f"(C[4]),
        "f"(C[5]),
        "f"(C[6]),
        "f"(C[7]));
}

__device__ inline void M16N16K4TN(
    Array<float, 8, 8>& C,
    Array<unsigned, 2, 2>& A,
    Array<unsigned, 2, 2>& B) {
  asm("mma.sync.aligned.m8n8k4.row.col.f32.f16.f16.f32 {%0,%1,%2,%3,%4,%5,%6,%7}, {%8,%9}, {%10,%11}, {%12,%13,%14,%15,%16,%17,%18,%19};\n"
      : "=f"(C[0]),
        "=f"(C[1]),
        "=f"(C[2]),
        "=f"(C[3]),
        "=f"(C[4]),
        "=f"(C[5]),
        "=f"(C[6]),
        "=f"(C[7])
      : "r"(A[0]),
        "r"(A[1]),
        "r"(B[0]),
        "r"(B[1]),
        "f"(C[0]),
        "f"(C[1]),
        "f"(C[2]),
        "f"(C[3]),
        "f"(C[4]),
        "f"(C[5]),
        "f"(C[6]),
        "f"(C[7]));
}

__device__ inline void M16N16K4NT(
    Array<float, 8, 8>& C,
    Array<unsigned, 2, 2>& A,
    Array<unsigned, 2, 2>& B) {
  asm("mma.sync.aligned.m8n8k4.col.row.f32.f16.f16.f32 {%0,%1,%2,%3,%4,%5,%6,%7}, {%8,%9}, {%10,%11}, {%12,%13,%14,%15,%16,%17,%18,%19};\n"
      : "=f"(C[0]),
        "=f"(C[1]),
        "=f"(C[2]),
        "=f"(C[3]),
        "=f"(C[4]),
        "=f"(C[5]),
        "=f"(C[6]),
        "=f"(C[7])
      : "r"(A[0]),
        "r"(A[1]),
        "r"(B[0]),
        "r"(B[1]),
        "f"(C[0]),
        "f"(C[1]),
        "f"(C[2]),
        "f"(C[3]),
        "f"(C[4]),
        "f"(C[5]),
        "f"(C[6]),
        "f"(C[7]));
}

__device__ inline void M16N16K4NN(
    Array<float, 8, 8>& C,
    Array<unsigned, 2, 2>& A,
    Array<unsigned, 2, 2>& B) {
  asm("mma.sync.aligned.m8n8k4.col.col.f32.f16.f16.f32 {%0,%1,%2,%3,%4,%5,%6,%7}, {%8,%9}, {%10,%11}, {%12,%13,%14,%15,%16,%17,%18,%19};\n"
      : "=f"(C[0]),
        "=f"(C[1]),
        "=f"(C[2]),
        "=f"(C[3]),
        "=f"(C[4]),
        "=f"(C[5]),
        "=f"(C[6]),
        "=f"(C[7])
      : "r"(A[0]),
        "r"(A[1]),
        "r"(B[0]),
        "r"(B[1]),
        "f"(C[0]),
        "f"(C[1]),
        "f"(C[2]),
        "f"(C[3]),
        "f"(C[4]),
        "f"(C[5]),
        "f"(C[6]),
        "f"(C[7]));
}

// Same initialization for now, will be different in interleaved
//   macros
__device__ inline void initM16N16K4(Array<float, 8, 8>& accumulator) {
  accumulator.set(0);
}

} // namespace Volta

#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 750))

namespace Turing {

__device__ inline void initM16N8K16(Array<float, 4, 4>& accumulator) {
  accumulator.set(0);
}

__device__ inline void M16N8K16TN(
    Array<float, 4, 4>& C,
    Array<unsigned, 4, 4>& A,
    Array<unsigned, 2, 2>& B) {
  asm("mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5}, {%6}, {%7,%8,%9,%10};\n"
      : "=f"(C[0]), "=f"(C[1]), "=f"(C[2]), "=f"(C[3])
      : "r"(A[0]),
        "r"(A[1]),
        "r"(B[0]),
        "f"(C[0]),
        "f"(C[1]),
        "f"(C[2]),
        "f"(C[3]));
  asm("mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5}, {%6}, {%7,%8,%9,%10};\n"
      : "=f"(C[0]), "=f"(C[1]), "=f"(C[2]), "=f"(C[3])
      : "r"(A[2]),
        "r"(A[3]),
        "r"(B[1]),
        "f"(C[0]),
        "f"(C[1]),
        "f"(C[2]),
        "f"(C[3]));
}

__device__ inline void initM16N16K16(Array<float, 8, 8>& accumulator) {
  accumulator.set(0);
}

__device__ inline void M16N16K16TN(
    Array<float, 8, 8>& C,
    Array<unsigned, 4, 4>& A,
    Array<unsigned, 4, 4>& B) {
  auto* _C = reinterpret_cast<Array<float, 4, 4>*>(&C);
  auto* _B = reinterpret_cast<Array<unsigned, 2, 2>*>(&B);
  M16N8K16TN(_C[0], A, _B[0]);
  M16N8K16TN(_C[1], A, _B[1]);
}

} // namespace Turing

#endif // Arch 75

#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800))

namespace Ampere {

__device__ inline void initM16N8K16(Array<float, 4, 4>& accumulator) {
  accumulator.set(0);
}

__device__ inline void M16N8K16TNF16(
    Array<float, 4, 4>& C,
    Array<unsigned, 4, 4>& A,
    Array<unsigned, 2, 2>& B) {
  asm("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};\n"
      : "=f"(C[0]), "=f"(C[1]), "=f"(C[2]), "=f"(C[3])
      : "r"(A[0]),
        "r"(A[1]),
        "r"(A[2]),
        "r"(A[3]),
        "r"(B[0]),
        "r"(B[1]),
        "f"(C[0]),
        "f"(C[1]),
        "f"(C[2]),
        "f"(C[3]));
}

__device__ inline void M16N8K16TNBF16(
    Array<float, 4, 4>& C,
    Array<unsigned, 4, 4>& A,
    Array<unsigned, 2, 2>& B) {
  asm("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};\n"
      : "=f"(C[0]), "=f"(C[1]), "=f"(C[2]), "=f"(C[3])
      : "r"(A[0]),
        "r"(A[1]),
        "r"(A[2]),
        "r"(A[3]),
        "r"(B[0]),
        "r"(B[1]),
        "f"(C[0]),
        "f"(C[1]),
        "f"(C[2]),
        "f"(C[3]));
}

__device__ inline void initM16N16K16(Array<float, 8, 8>& accumulator) {
  accumulator.set(0);
}

__device__ inline void M16N16K16TNF16(
    Array<float, 8, 8>& C,
    Array<unsigned, 4, 4>& A,
    Array<unsigned, 4, 4>& B) {
  auto* _C = reinterpret_cast<Array<float, 4, 4>*>(&C);
  auto* _B = reinterpret_cast<Array<unsigned, 2, 2>*>(&B);
  M16N8K16TNF16(_C[0], A, _B[0]);
  M16N8K16TNF16(_C[1], A, _B[1]);
}

__device__ inline void M16N16K16TNBF16(
    Array<float, 8, 8>& C,
    Array<unsigned, 4, 4>& A,
    Array<unsigned, 4, 4>& B) {
  auto* _C = reinterpret_cast<Array<float, 4, 4>*>(&C);
  auto* _B = reinterpret_cast<Array<unsigned, 2, 2>*>(&B);
  M16N8K16TNBF16(_C[0], A, _B[0]);
  M16N8K16TNBF16(_C[1], A, _B[1]);
}

} // namespace Ampere

#endif // Arch 80
