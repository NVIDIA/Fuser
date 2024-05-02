// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on

#include <host_ir_container.h>
#include <ir/builder.h>
#include <ir/cloner.h>
#include <ir/host_ir.h>
#include <ir/printer.h>
#include <ir/utils.h>
#include <kernel_ir.h>
#include <ops/all_ops.h>

namespace nvfuser {

namespace hir {

std::ostream& HostIrContainer::print(
    std::ostream& os,
    bool include_tensor_transforms,
    int indent_size) const {
  os << "\n%HostIrContainer {\n";
  IrMathPrinter op_exprs(os, indent_size);
  op_exprs.handle(this);
  // TODO implement the case include_tensor_transforms=true
  NVF_ERROR(
      !include_tensor_transforms,
      "the case include_tensor_transforms=true is not implemented for now");
  os << "}\n";

  return os;
}

} // namespace hir

} // namespace nvfuser
