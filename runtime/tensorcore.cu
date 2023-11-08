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
    Array<__half, 4, 4>& A,
    Array<__half, 4, 4>& B) {
  unsigned const* _A = reinterpret_cast<unsigned const*>(&A);
  unsigned const* _B = reinterpret_cast<unsigned const*>(&B);
  unsigned* _C = reinterpret_cast<unsigned*>(&C);

  asm("mma.sync.aligned.m8n8k4.row.row.f32.f16.f16.f32 {%0,%1,%2,%3,%4,%5,%6,%7}, {%8,%9}, {%10,%11}, {%12,%13,%14,%15,%16,%17,%18,%19};\n"
      : "=r"(_C[0]),
        "=r"(_C[1]),
        "=r"(_C[2]),
        "=r"(_C[3]),
        "=r"(_C[4]),
        "=r"(_C[5]),
        "=r"(_C[6]),
        "=r"(_C[7])
      : "r"(_A[0]),
        "r"(_A[1]),
        "r"(_B[0]),
        "r"(_B[1]),
        "r"(_C[0]),
        "r"(_C[1]),
        "r"(_C[2]),
        "r"(_C[3]),
        "r"(_C[4]),
        "r"(_C[5]),
        "r"(_C[6]),
        "r"(_C[7]));
}

__device__ inline void M16N16K4TN(
    Array<float, 8, 8>& C,
    Array<__half, 4, 4>& A,
    Array<__half, 4, 4>& B) {
  unsigned const* _A = reinterpret_cast<unsigned const*>(&A);
  unsigned const* _B = reinterpret_cast<unsigned const*>(&B);
  unsigned* _C = reinterpret_cast<unsigned*>(&C);

  asm("mma.sync.aligned.m8n8k4.row.col.f32.f16.f16.f32 {%0,%1,%2,%3,%4,%5,%6,%7}, {%8,%9}, {%10,%11}, {%12,%13,%14,%15,%16,%17,%18,%19};\n"
      : "=r"(_C[0]),
        "=r"(_C[1]),
        "=r"(_C[2]),
        "=r"(_C[3]),
        "=r"(_C[4]),
        "=r"(_C[5]),
        "=r"(_C[6]),
        "=r"(_C[7])
      : "r"(_A[0]),
        "r"(_A[1]),
        "r"(_B[0]),
        "r"(_B[1]),
        "r"(_C[0]),
        "r"(_C[1]),
        "r"(_C[2]),
        "r"(_C[3]),
        "r"(_C[4]),
        "r"(_C[5]),
        "r"(_C[6]),
        "r"(_C[7]));
}

__device__ inline void M16N16K4NT(
    Array<float, 8, 8>& C,
    Array<__half, 4, 4>& A,
    Array<__half, 4, 4>& B) {
  unsigned const* _A = reinterpret_cast<unsigned const*>(&A);
  unsigned const* _B = reinterpret_cast<unsigned const*>(&B);
  unsigned* _C = reinterpret_cast<unsigned*>(&C);

  asm("mma.sync.aligned.m8n8k4.col.row.f32.f16.f16.f32 {%0,%1,%2,%3,%4,%5,%6,%7}, {%8,%9}, {%10,%11}, {%12,%13,%14,%15,%16,%17,%18,%19};\n"
      : "=r"(_C[0]),
        "=r"(_C[1]),
        "=r"(_C[2]),
        "=r"(_C[3]),
        "=r"(_C[4]),
        "=r"(_C[5]),
        "=r"(_C[6]),
        "=r"(_C[7])
      : "r"(_A[0]),
        "r"(_A[1]),
        "r"(_B[0]),
        "r"(_B[1]),
        "r"(_C[0]),
        "r"(_C[1]),
        "r"(_C[2]),
        "r"(_C[3]),
        "r"(_C[4]),
        "r"(_C[5]),
        "r"(_C[6]),
        "r"(_C[7]));
}

__device__ inline void M16N16K4NN(
    Array<float, 8, 8>& C,
    Array<__half, 4, 4>& A,
    Array<__half, 4, 4>& B) {
  unsigned const* _A = reinterpret_cast<unsigned const*>(&A);
  unsigned const* _B = reinterpret_cast<unsigned const*>(&B);
  unsigned* _C = reinterpret_cast<unsigned*>(&C);

  asm("mma.sync.aligned.m8n8k4.col.col.f32.f16.f16.f32 {%0,%1,%2,%3,%4,%5,%6,%7}, {%8,%9}, {%10,%11}, {%12,%13,%14,%15,%16,%17,%18,%19};\n"
      : "=r"(_C[0]),
        "=r"(_C[1]),
        "=r"(_C[2]),
        "=r"(_C[3]),
        "=r"(_C[4]),
        "=r"(_C[5]),
        "=r"(_C[6]),
        "=r"(_C[7])
      : "r"(_A[0]),
        "r"(_A[1]),
        "r"(_B[0]),
        "r"(_B[1]),
        "r"(_C[0]),
        "r"(_C[1]),
        "r"(_C[2]),
        "r"(_C[3]),
        "r"(_C[4]),
        "r"(_C[5]),
        "r"(_C[6]),
        "r"(_C[7]));
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
    Array<__half, 8, 8>& A,
    Array<__half, 4, 4>& B) {
  unsigned const* _A = reinterpret_cast<unsigned const*>(&A);
  unsigned const* _B = reinterpret_cast<unsigned const*>(&B);
  unsigned* _C = reinterpret_cast<unsigned*>(&C);
  const unsigned* _D = reinterpret_cast<const unsigned*>(&C);

  asm("mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5}, {%6}, {%7,%8,%9,%10};\n"
      : "=r"(_C[0]), "=r"(_C[1]), "=r"(_C[2]), "=r"(_C[3])
      : "r"(_A[0]),
        "r"(_A[1]),
        "r"(_B[0]),
        "r"(_D[0]),
        "r"(_D[1]),
        "r"(_D[2]),
        "r"(_D[3]));
  asm("mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5}, {%6}, {%7,%8,%9,%10};\n"
      : "=r"(_C[0]), "=r"(_C[1]), "=r"(_C[2]), "=r"(_C[3])
      : "r"(_A[2]),
        "r"(_A[3]),
        "r"(_B[1]),
        "r"(_D[0]),
        "r"(_D[1]),
        "r"(_D[2]),
        "r"(_D[3]));
}

__device__ inline void initM16N16K16(Array<float, 8, 8>& accumulator) {
  accumulator.set(0);
}

__device__ inline void M16N16K16TN(
    Array<float, 8, 8>& C,
    Array<__half, 8, 8>& A,
    Array<__half, 8, 8>& B) {
  auto* _C = reinterpret_cast<Array<float, 4, 4>*>(&C);
  auto* _B = reinterpret_cast<Array<__half, 4, 4>*>(&B);
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

__device__ inline void M16N8K16TN(
    Array<float, 4, 4>& C,
    Array<__half, 8, 8>& A,
    Array<__half, 4, 4>& B) {
  unsigned const* _A = reinterpret_cast<unsigned const*>(&A);
  unsigned const* _B = reinterpret_cast<unsigned const*>(&B);
  unsigned* _C = reinterpret_cast<unsigned*>(&C);
  const unsigned* _D = reinterpret_cast<const unsigned*>(&C);

  asm("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};\n"
      : "=r"(_C[0]), "=r"(_C[1]), "=r"(_C[2]), "=r"(_C[3])
      : "r"(_A[0]),
        "r"(_A[1]),
        "r"(_A[2]),
        "r"(_A[3]),
        "r"(_B[0]),
        "r"(_B[1]),
        "r"(_D[0]),
        "r"(_D[1]),
        "r"(_D[2]),
        "r"(_D[3]));
}

__device__ inline void M16N8K16TN(
    Array<float, 4, 4>& C,
    Array<__bfloat, 8, 8>& A,
    Array<__bfloat, 4, 4>& B) {
  unsigned const* _A = reinterpret_cast<unsigned const*>(&A);
  unsigned const* _B = reinterpret_cast<unsigned const*>(&B);
  unsigned* _C = reinterpret_cast<unsigned*>(&C);
  const unsigned* _D = reinterpret_cast<const unsigned*>(&C);

  asm("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};\n"
      : "=r"(_C[0]), "=r"(_C[1]), "=r"(_C[2]), "=r"(_C[3])
      : "r"(_A[0]),
        "r"(_A[1]),
        "r"(_A[2]),
        "r"(_A[3]),
        "r"(_B[0]),
        "r"(_B[1]),
        "r"(_D[0]),
        "r"(_D[1]),
        "r"(_D[2]),
        "r"(_D[3]));
}

__device__ inline void initM16N16K16(Array<float, 8, 8>& accumulator) {
  accumulator.set(0);
}

template <typename T>
__device__ inline void M16N16K16TN(
    Array<float, 8, 8>& C,
    Array<T, 8, 8>& A,
    Array<T, 8, 8>& B) {
  auto* _C = reinterpret_cast<Array<float, 4, 4>&>(&C);
  auto* _B = reinterpret_cast<Array<T, 4, 4>&>(&B);
  M16N8K16TN(_C[0], A, _B[0]);
  M16N8K16TN(_C[1], A, _B[1]);
}

} // namespace Ampere

#endif // Arch 80
