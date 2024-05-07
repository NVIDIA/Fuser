// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on

#include <host_ir/container.h>
#include <ir/builder.h>
#include <ir/cloner.h>
#include <host_ir/host_ir.h>
#include <ir/printer.h>
#include <ir/utils.h>
#include <kernel_ir.h>
#include <ops/all_ops.h>

namespace nvfuser {

namespace hir {

std::ostream& HostIrContainer::print(std::ostream& os) const {
  os << "\n%HostIrContainer {\n";
  IrMathPrinter op_exprs(os);
  op_exprs.handle(this);
  return os;
}

} // namespace hir

} // namespace nvfuser
