// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#pragma once

#include <ir/iostream.h>
#include <iter_visitor.h>

#include <iostream>

namespace nvfuser {

//! Prints computation Fusion IR nodes
//!
//! IrMathPrinter and IrTransformPrinter allow the splitting up of fusion print
//! functions. IrMathPrinter as its name implies focuses solely on what tensor
//! computations are taking place. Resulting TensorView math will reflect the
//! series of split/merge/computeAts that have taken place, however these
//! nodes will not be displayed in what is printed. IrTransformPrinter does not
//! print any mathematical functions and only lists the series of
//! split/merge calls that were made. Both of these printing methods are
//! quite verbose on purpose as to show accurately what is represented in the IR
//! of a fusion.
//
//! \sa IrTransformPrinter
//!
class IrMathPrinter : public IrPrinter {
 public:
  IrMathPrinter(std::ostream& os) : IrPrinter(os) {}

  using IrPrinter::handle;

  void handle(Fusion* f) override {
    IrPrinter::handle(f);
  }
};

//! Prints transformation (schedule) Fusion IR nodes
//!
//! \sa IrMathPrinter
//!
class IrTransformPrinter : public IrPrinter {
 public:
  IrTransformPrinter(std::ostream& os) : IrPrinter(os) {}

  using IrPrinter::handle;

  void handle(Fusion* f) override;

  void printTransforms(const TensorView* tv);
};

} // namespace nvfuser
