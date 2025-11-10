// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#include <bindings.h>
#include <multidevice/communicator.h>
#include <scheduler/matmul_heuristic.h>
#include <scheduler/scheduler_types.h>
#include <type.h>

namespace nvfuser::python {

void bindEnums(py::module& nvfuser) {
  //! DataTypes supported by nvFuser in the FusionDefinition. The python
  //! DataType maps to the CPP PrimDataType. On the CPP side, there is also a
  //! DateType enum that includes struct, array, pointer, or opaque datatypes.
  py::enum_<PrimDataType>(nvfuser, "DataType", py::module_local())
      .value("Double", DataType::Double)
      .value("Float", DataType::Float)
      .value("Half", DataType::Half)
      .value("Int", DataType::Int)
      .value("Int32", DataType::Int32)
      .value("UInt64", DataType::UInt64)
      .value("Bool", DataType::Bool)
      .value("BFloat16", DataType::BFloat16)
      .value("Float8_e4m3fn", DataType::Float8_e4m3fn)
      .value("Float8_e5m2", DataType::Float8_e5m2)
      .value("Float8_e8m0fnu", DataType::Float8_e8m0fnu)
      .value("Float4_e2m1fn", DataType::Float4_e2m1fn)
      .value("Float4_e2m1fn_x2", DataType::Float4_e2m1fn_x2)
      .value("ComplexFloat", DataType::ComplexFloat)
      .value("ComplexDouble", DataType::ComplexDouble)
      .value("Null", DataType::Null);

  py::enum_<ParallelType>(nvfuser, "ParallelType", py::module_local())
      .value("mesh_x", ParallelType::DIDx)
      .value("grid_x", ParallelType::BIDx)
      .value("grid_y", ParallelType::BIDy)
      .value("grid_z", ParallelType::BIDz)
      .value("block_x", ParallelType::TIDx)
      .value("block_y", ParallelType::TIDy)
      .value("block_z", ParallelType::TIDz)
      .value("mma", ParallelType::Mma)
      .value("serial", ParallelType::Serial)
      .value("tma", ParallelType::Bulk)
      .value("unroll", ParallelType::Unroll)
      .value("unswitch", ParallelType::Unswitch)
      .value("vectorize", ParallelType::Vectorize)
      .value("stream", ParallelType::Stream);

  py::enum_<CommunicatorBackend>(
      nvfuser, "CommunicatorBackend", py::module_local())
      .value("nccl", CommunicatorBackend::kNccl)
      .value("ucc", CommunicatorBackend::kUcc)
      .value("cuda", CommunicatorBackend::kCuda);

  py::enum_<SchedulerType>(nvfuser, "SchedulerType", py::module_local())
      .value("none", SchedulerType::None)
      .value("no_op", SchedulerType::NoOp)
      .value("pointwise", SchedulerType::PointWise)
      .value("matmul", SchedulerType::Matmul)
      .value("reduction", SchedulerType::Reduction)
      .value("inner_persistent", SchedulerType::InnerPersistent)
      .value("inner_outer_persistent", SchedulerType::InnerOuterPersistent)
      .value("outer_persistent", SchedulerType::OuterPersistent)
      .value("transpose", SchedulerType::Transpose)
      .value("expr_eval", SchedulerType::ExprEval)
      .value("resize", SchedulerType::Resize);

  py::enum_<LoadStoreOpType>(nvfuser, "LoadStoreOpType", py::module_local())
      .value("set", LoadStoreOpType::Set)
      .value("load_matrix", LoadStoreOpType::LdMatrix)
      .value("cp_async", LoadStoreOpType::CpAsync)
      .value("tma", LoadStoreOpType::CpAsyncBulkTensorTile);

  py::enum_<MemoryType>(nvfuser, "MemoryType", py::module_local())
      .value("tensor", MemoryType::Tensor)
      .value("local", MemoryType::Local)
      .value("shared", MemoryType::Shared)
      .value("global", MemoryType::Global);

  py::enum_<CacheOp>(nvfuser, "CacheOp", py::module_local())
      .value("unspecified", CacheOp::Unspecified)
      .value("all_levels", CacheOp::AllLevels)
      .value("streaming", CacheOp::Streaming)
      .value("global", CacheOp::Global);

  py::enum_<IdMappingMode>(nvfuser, "IdMappingMode")
      .value("exact", IdMappingMode::EXACT)
      .value("almost_exact", IdMappingMode::ALMOSTEXACT)
      .value("broadcast", IdMappingMode::BROADCAST)
      .value("permissive", IdMappingMode::PERMISSIVE)
      .value("loop", IdMappingMode::LOOP);

  py::enum_<MatmulParams::TilingStrategy> tiling_strategy(
      nvfuser, "MatmulTilingStrategy", py::module_local());
  tiling_strategy.value(
      "one_tile_per_cta", MatmulParams::TilingStrategy::OneTilePerCTA);
  tiling_strategy.value(
      "distribute_tiles_across_sms",
      MatmulParams::TilingStrategy::DistributeTilesAcrossSMs);
  tiling_strategy.value(
      "distribute_stages_across_sms",
      MatmulParams::TilingStrategy::DistributeStagesAcrossSMs);

  py::enum_<MatmulParams::BufferingLoopLevel> buffering_loop_level(
      nvfuser, "MatmulBufferingLoopLevel", py::module_local());
  buffering_loop_level.value(
      "cta_tiles", MatmulParams::BufferingLoopLevel::CTATiles);
  buffering_loop_level.value(
      "warp_tiles", MatmulParams::BufferingLoopLevel::WarpTiles);

  py::enum_<MatmulParams::CircularBufferingStrategy>
      circular_buffering_strategy(
          nvfuser, "MatmulCircularBufferingStrategy", py::module_local());
  circular_buffering_strategy.value(
      "pipelined", MatmulParams::CircularBufferingStrategy::Pipelined);
  circular_buffering_strategy.value(
      "warp_specialized",
      MatmulParams::CircularBufferingStrategy::WarpSpecialized);
}

} // namespace nvfuser::python
