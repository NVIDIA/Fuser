// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on

#include <ops/all_ops.h>
#include <kernel_ir.h>
#include <ir/builder.h>
#include <ir/printer.h>
#include <ir/cloner.h>
#include <ir/utils.h>
#include <ir/host_ir.h>
#include <host_ir_container.h>

namespace nvfuser {

namespace hir {

std::ostream& HostIrContainer::print(std::ostream& os, bool include_tensor_transforms, int indent_size) const {
  os << "\n%HostIrContainer {\n";
  IrMathPrinter op_exprs(os, indent_size);
  op_exprs.handle(this);
  NVF_ERROR(!include_tensor_transforms, "not implemented for now");
//   if (include_tensor_transforms) {
//     os << "\nTransformPrinter : \n";
//     IrTransformPrinter t_exprs(os, indent_size);
//     t_exprs.handle(this);
//   }
  os << "}\n";

  return os;
}

} // namespace hir

} // namespace nvfuser
