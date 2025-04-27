// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#include <bindings.h>
#include <multidevice/communicator.h>
#include <scheduler/scheduler_types.h>
#include <type.h>

namespace nvfuser::python {

void bindEnums(py::module& nvfuser) {
  //! DataTypes supported by nvFuser in the FusionDefinition
  py::enum_<PrimDataType>(nvfuser, "DataType")
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
      .value("ComplexFloat", DataType::ComplexFloat)
      .value("ComplexDouble", DataType::ComplexDouble)
      .value("Null", DataType::Null);

  //! ParallelType used for scheduling
  py::enum_<ParallelType>(nvfuser, "ParallelType")
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

  //! LoadStoreOpType used for scheduling
  py::enum_<LoadStoreOpType>(nvfuser, "LoadStoreOpType")
      .value("set", LoadStoreOpType::Set)
      .value("load_matrix", LoadStoreOpType::LdMatrix)
      .value("cp_async", LoadStoreOpType::CpAsync)
      .value("tma", LoadStoreOpType::CpAsyncBulkTensorTile);

  //! CacheOp used for scheduling
  py::enum_<CacheOp>(nvfuser, "CacheOp")
      .value("unspecified", CacheOp::Unspecified)
      .value("all_levels", CacheOp::AllLevels)
      .value("streaming", CacheOp::Streaming)
      .value("global", CacheOp::Global);

  //! MemoryType used for scheduling
  py::enum_<MemoryType>(nvfuser, "MemoryType")
      .value("local", MemoryType::Local)
      .value("shared", MemoryType::Shared)
      .value("global", MemoryType::Global);

  //! Scheduler Type for scheduling
  py::enum_<SchedulerType>(nvfuser, "SchedulerType")
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

  py::enum_<CommunicatorBackend>(nvfuser, "CommunicatorBackend")
      .value("nccl", CommunicatorBackend::kNccl)
      .value("ucc", CommunicatorBackend::kUcc);
}

} // namespace nvfuser::python
