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

#include <iosfwd>

namespace nvfuser {

//! Prints computation Fusion IR nodes
//!
//! IrPrinter and IrTransformPrinter allow the splitting up of fusion print
//! functions. IrPrinter as its name implies focuses solely on what tensor
//! computations are taking place. Resulting TensorView math will reflect the
//! series of split/merge/computeAts that have taken place, however these
//! nodes will not be displayed in what is printed. IrTransformPrinter does not
//! print any mathematical functions and only lists the series of
//! split/merge calls that were made. Both of these printing methods are
//! quite verbose on purpose as to show accurately what is represented in the IR
//! of a fusion.
//
//! \sa IrTransformPrinter
class IrPrinter {
 public:
  explicit IrPrinter(std::ostream& os, int indent_size = 0)
      : os_(os), indent_size_(indent_size) {}
  virtual ~IrPrinter() = default;

  void resetIndent() {
    indent_size_ = 0;
  }

  bool printInline() const {
    return print_inline_;
  }

  virtual void handle(const Fusion* f);
  virtual void handle(const kir::Kernel* kernel);
  virtual void handle(const hir::HostIrContainer* host_ir_container);

 protected:
  std::ostream& os() {
    return os_;
  }

 private:
  std::ostream& os_;
  bool print_inline_ = false;
  int indent_size_ = 0;
};

//! Prints transformation (schedule) Fusion IR nodes
//!
//! \sa IrPrinter
class IrTransformPrinter : public IrPrinter {
 public:
  IrTransformPrinter(std::ostream& os) : IrPrinter(os) {}

  using IrPrinter::handle;

  void handle(const Fusion* f) override;

  void printTransforms(const TensorView* tv);
};

} // namespace nvfuser
