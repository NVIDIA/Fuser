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
      .value("ucc", CommunicatorBackend::kUcc);
}

} // namespace nvfuser::python
