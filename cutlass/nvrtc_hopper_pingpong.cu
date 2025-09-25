
#include "cutlass/cutlass.h"
#include "cutlass/gemm/dispatch_policy.hpp"
#include "cutlass/numeric_types.h"
#include "cutlass/pipeline/pipeline.hpp"
#include "cutlass/trace.h"

#include "cute/algorithm/functional.hpp"
#include "cute/algorithm/gemm.hpp"
#include "cute/arch/cluster_sm90.hpp"
#include "cute/arch/copy_sm90.hpp"
#include "cute/atom/mma_atom.hpp"
#include "cute/numeric/arithmetic_tuple.hpp"

using namespace cute;

// A matrix configuration
using ElementA = cutlass::bfloat16_t; // Element type for A matrix operand
using LayoutA =
    cutlass::layout::ColumnMajor; // Layout type for A matrix operand
constexpr int AlignmentA =
    128 /
    cutlass::sizeof_bits<
        ElementA>::value; // Memory access granularity/alignment of A matrix in
                          // units of elements (up to 16 bytes)

// B matrix configuration
using ElementB = cutlass::bfloat16_t; // Element type for B matrix operand
using LayoutB =
    cutlass::layout::ColumnMajor; // Layout type for B matrix operand
constexpr int AlignmentB =
    128 /
    cutlass::sizeof_bits<
        ElementB>::value; // Memory access granularity/alignment of B matrix in
                          // units of elements (up to 16 bytes)

// C/D matrix configuration
using ElementC =
    cutlass::bfloat16_t; // Element type for C and D matrix operands
using LayoutC =
    cutlass::layout::RowMajor; // Layout type for C and D matrix operands
constexpr int AlignmentC =
    128 /
    cutlass::sizeof_bits<
        ElementC>::value; // Memory access granularity/alignment of C matrix in
                          // units of elements (up to 16 bytes)

// Core kernel configurations
using ElementAccumulator = float; // Element type for internal accumulation
using ElementCompute = float; // Element type for epilogue computation
using ArchTag = cutlass::arch::Sm90; // Tag indicating the minimum SM that
                                     // supports the intended feature
using OperatorClass = cutlass::arch::OpClassTensorOp; // Operator class tag
using EpilogueTileType = cutlass::epilogue::collective::EpilogueTileAuto;
using StageCountType =
    cutlass::gemm::collective::StageCountAuto; // Stage count maximized based on
                                               // the tile size
using KernelSchedule =
    cutlass::gemm::KernelTmaWarpSpecializedPingpong; // Kernel to launch TMA
                                                     // warp specialization
using EpilogueSchedule =
    cutlass::epilogue::TmaWarpSpecialized; // Epilogue to use TMA warp
                                           // specialization

using ClusterShape =
    Shape<_1, _2, _1>; // Shape of the threadblocks in a cluster
using TileShape = Shape<_128, _128, _64>; // Threadblock-level tile size
using ElementA = cutlass::bfloat16_t;
using StrideA = cutlass::layout::ColumnMajor;
using ElementB = ElementB_;
using StrideB = StrideB_;
using TiledMma = TiledMma_;
using ElementAccumulator = typename TiledMma::ValTypeC;
using GmemTiledCopyA = GmemTiledCopyA_;
using GmemTiledCopyB = GmemTiledCopyB_;
using SmemLayoutAtomA = SmemLayoutAtomA_;
using SmemLayoutAtomB = SmemLayoutAtomB_;
using SmemCopyAtomA = SmemCopyAtomA_;
using SmemCopyAtomB = SmemCopyAtomB_;
using TransformA = TransformA_;
using TransformB = TransformB_;
using ArchTag = typename DispatchPolicy::ArchTag;

// Device side kernel params
struct Params {
  // Assumption: StrideA is congruent with Problem_MK
  using TMA_A = decltype(make_tma_copy_A_sm90(
      GmemTiledCopyA{},
      make_tensor(
          static_cast<InternalElementA const*>(nullptr),
          repeat_like(StrideA{}, int32_t(0)),
          StrideA{}),
      SmemLayoutA{}(_, _, cute::Int<0>{}),
      TileShape{},
      ClusterShape{}));
  // Assumption: StrideB is congruent with Problem_NK
  using TMA_B = decltype(make_tma_copy_B_sm90(
      GmemTiledCopyB{},
      make_tensor(
          static_cast<InternalElementB const*>(nullptr),
          repeat_like(StrideB{}, int32_t(0)),
          StrideB{}),
      SmemLayoutB{}(_, _, cute::Int<0>{}),
      TileShape{},
      ClusterShape{}));
  TMA_A tma_load_a;
  TMA_B tma_load_b;
  uint32_t tma_transaction_bytes = TmaTransactionBytes;
  uint32_t tma_transaction_bytes_mk = TmaTransactionBytesMK;
  uint32_t tma_transaction_bytes_nk = TmaTransactionBytesNK;
};

/// Issue Tma Descriptor Prefetch -- ideally from a single thread for best
/// performance
__forceinline__ __device__ static void prefetch_tma_descriptors(
    Params const& mainloop_params) {
  cute::prefetch_tma_descriptor(
      mainloop_params.tma_load_a.get_tma_descriptor());
  cute::prefetch_tma_descriptor(
      mainloop_params.tma_load_b.get_tma_descriptor());
}
