// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#include <codegen.h>
#include <device_lower/utils.h>
#include <instrumentation.h>
#include <ir/utils.h>
#include <kernel_ir.h>
#include <kernel_ir_dispatch.h>
#include <scheduler/mma_utils.h>
#include <type.h>
#include <utils.h>

#include <array>
#include <cmath>
#include <sstream>
#include <typeindex>
#include <typeinfo>
#include <vector>

namespace nvfuser {
namespace codegen {

namespace {

std::string ptrType(DataType dt) {
  std::stringstream ss;
  ss << dt << "*";
  return ss.str();
}

//! Utility class to build an argument list
class ArgumentBuilder {
 public:
  //! Build an argument list where each argument is separated with a comma
  ArgumentBuilder() = default;

  //! Build an argument list where each argument has its own line
  ArgumentBuilder(int indent_level, const char* tab) {
    std::stringstream ss;
    for (const auto i : c10::irange(indent_level)) {
      (void)i; // Suppress unused variable warning
      ss << tab;
    }
    sep_ = ",\n" + ss.str();
  }

  //! Add a new argument
  template <typename T>
  ArgumentBuilder& arg(const T& x) {
    addSeparator();
    return append(x);
  }

  //! Append to the last argument
  template <typename T>
  ArgumentBuilder& append(const T& arg) {
    ss_ << arg;
    return *this;
  }

  //! Get a string of the argument list
  std::string str() const {
    return ss_.str();
  }

  friend std::ostream& operator<<(std::ostream& os, const ArgumentBuilder& ab) {
    return os << ab.str();
  }

 private:
  void addSeparator() {
    if (ss_.tellp() != 0) {
      ss_ << sep_;
    }
  }

 private:
  std::string sep_ = ", ";
  std::stringstream ss_;
};

//! Append to the last argument
template <>
ArgumentBuilder& ArgumentBuilder::append<bool>(const bool& arg) {
  ss_ << (arg ? "true" : "false");
  return *this;
}

//! Returns "template_name<template_arg>"
template <typename TemplateNameT, typename TemplateArgT>
std::string genTemplate(
    const TemplateNameT& template_name,
    const TemplateArgT& template_arg) {
  std::stringstream ss;
  ss << template_name << "<" << template_arg << ">";
  return ss.str();
}

//! Returns "func_name(func_arg)"
template <typename FuncNameT, typename FuncArgT>
std::string genCall(const FuncNameT& func_name, const FuncArgT& func_arg) {
  std::stringstream ss;
  ss << func_name << "(" << func_arg << ")";
  return ss.str();
}

//! Returns "func_name<template_arg>(func_arg)"
template <typename FuncNameT, typename TemplateArgT, typename FuncArgT>
std::string genCall(
    const FuncNameT& func_name,
    const TemplateArgT& template_arg,
    const FuncArgT& func_arg) {
  std::stringstream ss;
  ss << func_name << "<" << template_arg << ">(" << func_arg << ")";
  return ss.str();
}

template <typename T>
std::string genPtrType(const T& type) {
  std::stringstream ss;
  ss << type << "*";
  return ss.str();
}

template <typename CastType, typename ArgType>
std::string genStaticCast(const CastType& type, const ArgType& arg) {
  return genCall("static_cast", type, arg);
}

template <typename CastType, typename ArgType>
std::string genReinterpretCast(const CastType& type, const ArgType& arg) {
  return genCall("reinterpret_cast", type, arg);
}

class CudaKernelGenerator : private kir::ConstIrVisitor {
  static constexpr const char* kTab = "  ";

 public:
  static std::string generateKernelDefinition(
      const kir::Kernel* kernel,
      const std::string& kernel_name) {
    CudaKernelGenerator codegen(kernel);
    codegen.genDeclaration(kernel_name);
    codegen.startBlock();
    codegen.genPrologue();
    codegen.genBody();
    codegen.endBlock();
    TORCH_CHECK(codegen.block_nest_level_ == 0);
    return codegen.code_.str();
  }

 private:
  explicit CudaKernelGenerator(const kir::Kernel* kernel) : kernel_(kernel) {
    initStringStreamFormat(code_);
  }

  using kir::ConstIrVisitor::handle;

  void initStringStreamFormat(std::stringstream& ss) {
    ss.imbue(std::locale("C"));
    ss << std::scientific;
    // Set the default precision as Double
    setPrecision(ss, DataType::Double);
  }

  void setPrecision(std::stringstream& ss, DataType dtype) {
    TORCH_INTERNAL_ASSERT(isFloatingPointType(dtype));
    ss << std::setprecision(max_digits10(dtype));
  }

  std::string getLiteralSuffix(DataType dtype) {
    // The type of an integer literal is automatically picked from
    // int, long int, and long long int, so no suffix should be
    // required. https://en.cppreference.com/w/cpp/language/integer_literal
    switch (std::get<PrimDataType>(dtype.type)) {
      case DataType::Float:
        return "f";
      default:
        return "";
    }
  }

  // Generates the kernel function declaration
  void genDeclaration(const std::string& kernel_name) {
    const auto& kernel_summary = kernel_->summary();

    code_ << "__global__ void " << kernel_name << "(";

    std::unordered_set<Val*> unique_args;

    std::vector<Val*> params;

    // Inputs & Outputs
    for (auto val : kernel_->inputs()) {
      params.push_back(val);
    }
    for (auto val : kernel_->outputs()) {
      TORCH_INTERNAL_ASSERT(
          !val->isScalar(), "No scalar output is allowed: ", val->toString());
      params.push_back(val);
    }

    // Generate parameter declarations
    unsigned int duplicate_counter = 0;
    for (auto i : c10::irange(params.size())) {
      std::stringstream var_name_ss;
      if (params[i]->isA<TensorView>()) {
        var_name_ss << ir_utils::varName(params[i]->as<TensorView>());
      } else {
        var_name_ss << gen(params[i]);
      }

      // If value is duplicate in arguments change the name to avoid name
      // conflicts in args.
      if (!unique_args.emplace(params[i]).second) {
        var_name_ss << "_duplicate_" << duplicate_counter++;
      }

      if (const auto tv = dynamic_cast<TensorView*>(params[i])) {
        if (tv->isCpuScalar()) {
          code_ << " CpuScalarTensor<" << params[i]->dtype() << "> "
                << var_name_ss.str();
        } else {
          code_
              << "Tensor<" << params[i]->dtype() << ", "
              << TensorDomain::noReductions(tv->getMaybeRFactorDomain()).size()
              << ", "
              << TensorDomain::noReductions(tv->getMaybeAllocationDomain())
                     .size()
              << "> " << var_name_ss.str();
        }
      } else {
        TORCH_INTERNAL_ASSERT(params[i]->isScalar()); // NOLINT (LLVM bug 48525)
        TORCH_INTERNAL_ASSERT(params[i]->definition() == nullptr);
        code_ << params[i]->dtype() << " " << var_name_ss.str();
      }

      if (i + 1 != params.size()) {
        code_ << ", ";
      }
    }

    // Global buffers
    for (auto allocate : kernel_summary.global_allocations) {
      TORCH_INTERNAL_ASSERT(allocate->buffer()->isA<TensorView>());
      const auto tv = allocate->buffer()->as<TensorView>();
      const auto& alloc_domain =
          TensorDomain::noReductions(tv->getMaybeAllocationDomain());
      code_ << ", Tensor<" << tv->dtype() << ", " << alloc_domain.size() << ", "
            << alloc_domain.size() << "> " << ir_utils::varName(tv);
    }

    // Kernels generating random numbers take extra (seed, offset) arguments
    if (kernel_summary.max_rng_offsets >= 0) {
      code_ << ", at::PhiloxCudaState philox_args";
    }

    code_ << ") ";
  }

  // Generates setup code which is executed before the kernel body
  void genPrologue() {
    const auto& kernel_summary = kernel_->summary();

    // Random number generator (optional)
    if (kernel_summary.max_rng_offsets >= 0) {
      indent() << "auto philox_offset = philox_args.captured_ ?\n";
      indent()
          << "  static_cast<uint64_t>(*(philox_args.offset_.ptr) + philox_args.offset_intragraph_) :\n";
      indent() << "  philox_args.offset_.val;\n";
      indent() << "uint4 rng_result;\n";
      indent() << "nvfuser_index_t rng_subseq = -1;\n";
      indent() << "nvfuser_index_t rng_offset = -1;\n";
    }

    // Do we have any dynamic shared memory buffers?
    const bool has_dynamic_smem =
        !kernel_summary.dynamic_smem_allocations.empty();

    // Do we have any reductions?
    const bool has_reductions = kernel_summary.has_block_reductions ||
        kernel_summary.has_grid_reductions;
    const bool has_parallel_welford =
        kernel_summary.has_block_welford || kernel_summary.has_grid_welford;

    // Shared memory
    if (has_dynamic_smem || has_reductions || has_parallel_welford) {
      indent() << "alignas("
               << 16 // always align to 16B for any shared mem allocation
               << ") extern __shared__ char array[];\n";

      if (has_dynamic_smem) {
        indent() << "unsigned smem_offset = 0;\n";
      }

      if (has_reductions || has_parallel_welford) {
        indent() << "void* shared_mem = array;\n";
        if (has_dynamic_smem) {
          std::stringstream smem_buf_size_ss;
          smem_buf_size_ss << "blockDim.x * blockDim.y * blockDim.z * sizeof("
                           << kernel_summary.largest_smem_data_type << ")";
          if (has_parallel_welford) {
            smem_buf_size_ss << " * 3";
          }
          std::string smem_buf_size = smem_buf_size_ss.str();
          if (kernel_summary.has_outer_grouped_grid_welford) {
            std::stringstream smem_buf_size_with_outer_opt;
            smem_buf_size_with_outer_opt
                << "max(" << smem_buf_size << ", "
                << kernel_summary.outer_grouped_grid_welford_largest_smem_size
                << ")";
            smem_buf_size = smem_buf_size_with_outer_opt.str();
          }
          indent() << "smem_offset += " << smem_buf_size << ";\n";
        }

        if (has_parallel_welford) {
          // Unpack shared mem pointer
          auto space_type = kernel_summary.largest_smem_data_type;
          indent()
              << "nvfuser_index_t block_size = blockDim.x*blockDim.y*blockDim.z;\n";
          indent() << space_type << " *shared_mem_var = "
                   << "static_cast<" << space_type << "*>("
                   << "shared_mem);\n";
          indent() << space_type
                   << " *shared_mem_avg = shared_mem_var + block_size;\n";
          indent() << space_type
                   << " *shared_mem_n = shared_mem_avg + block_size;\n";
        }
      }
    }

    // Call the initialization function if using a custom block sync
    if (std::getenv("PYTORCH_NVFUSER_USE_BLOCK_SYNC_ATOMIC")) {
      indent() << "block_sync::init();\n";
    }
  }

  // Cannot just use ConstIrVisitor::handle as it expects a vector of
  // const Expr*, whereas most of the IR API returns a vector of
  // non-const Expr*.
  void handle(const std::vector<Expr*>& exprs) {
    for (Expr* expr : exprs) {
      kir::ConstIrVisitor::handle(expr);
    }
  }

  void genBody() {
    handle(kernel_->topLevelExprs());
  }

  void startBlock(bool continuation = false) {
    if (continuation) {
      code_ << "{\n";
    } else {
      indent() << "{\n";
    }
    ++block_nest_level_;
  }

  void endBlock(const char* sep = "\n") {
    --block_nest_level_;
    TORCH_CHECK(block_nest_level_ >= 0);
    indent() << "}" << sep;
  }

  //! Remember the alignment info of a new scope expr (IfThenElse or ForLoop)
  void pushAlignmentInfo(const Expr* scope_expr) {
    aligned_scope_exprs_.push_back(ir_utils::isAlignedScopeExpr(scope_expr));
  }

  void popAlignmentInfo() {
    aligned_scope_exprs_.pop_back();
  }

  //! Check if the current scope is aligned, i.e., guaranteed to cause
  //! no thread divergence
  bool isAligned() const {
    return std::all_of(
        aligned_scope_exprs_.begin(), aligned_scope_exprs_.end(), [](bool b) {
          return b;
        });
  }

  std::ostream& indent() {
    for (const auto i : c10::irange(block_nest_level_)) {
      (void)i; // Suppress unused variable warning
      code_ << kTab;
    }
    return code_;
  }

  std::string gen(const Statement* stmt) {
    if (stmt->isA<Expr>()) {
      // This expr should just be an individul expr with no nested
      // scope
      TORCH_INTERNAL_ASSERT(
          !stmt->isA<kir::IfThenElse>() && !stmt->isA<kir::ForLoop>(),
          "Invalid expr: ",
          stmt->toString());
    } else {
      TORCH_INTERNAL_ASSERT(
          stmt->isA<Val>(), "Unknown Statement IR type: ", stmt->toString());
    }

    std::stringstream tmp_code;
    initStringStreamFormat(tmp_code);
    std::swap(tmp_code, code_);
    handle(stmt);
    std::swap(tmp_code, code_);
    return tmp_code.str();
  }

  std::string genInline(const Statement* stmt) {
    const bool saved_inline = print_inline_;
    print_inline_ = true;
    auto result = gen(stmt);
    print_inline_ = saved_inline;
    // NOLINTNEXTLINE(performance-no-automatic-move)
    return result;
  }

  void handle(const kir::Predicate* pred) final {
    TORCH_INTERNAL_ASSERT(pred->hasValue());
    code_ << gen(pred->value());
  }

  void handle(const Bool* pred) final {
    const auto def = pred->definition();
    const bool has_alloc = alloc_map_.find(pred) != alloc_map_.end();
    if (def != nullptr && !has_alloc) {
      code_ << "(" << genInline(def) << ")";
    } else if (pred->isConst()) {
      code_ << (*pred->value() ? "true" : "false");
    } else {
      code_ << ir_utils::varName(pred);
    }
  }

  void handle(const Double* d) final {
    const auto def = d->definition();
    const bool has_alloc = alloc_map_.find(d) != alloc_map_.end();
    if (def != nullptr && !has_alloc) {
      code_ << "(" << genInline(def) << ")";
    } else if (d->isConst()) {
      auto val = *d->value();
      // note: default inf/nan doesn't work and should be replaced with macros
      // `NAN`, `POS_INFINITY` and `NEG_INFINITY` instead.
      if (std::isinf(val)) {
        if (val > 0) {
          code_ << "POS_INFINITY";
        } else {
          code_ << "NEG_INFINITY";
        }
      } else if (std::isnan(val)) {
        code_ << "NAN";
      } else {
        setPrecision(code_, d->getDataType().value());
        code_ << val << getLiteralSuffix(d->getDataType().value());
      }
    } else {
      code_ << ir_utils::varName(d);
    }
  }

  void handle(const Int* i) final {
    // Check the replacement map first. If there's an entry for i, use
    // the corresponding replacement.
    auto replace_it = index_replacement_map_.find(i);
    if (replace_it != index_replacement_map_.end()) {
      code_ << replace_it->second;
      return;
    }

    const auto def = i->definition();
    const bool has_alloc = alloc_map_.find(i) != alloc_map_.end();
    if (def != nullptr && !has_alloc) {
      code_ << "(" << genInline(def) << ")";
    } else if (i->isConst()) {
      code_ << *i->value() << getLiteralSuffix(i->getDataType().value());
    } else {
      code_ << ir_utils::varName(i);
    }
  }

  void handle(const ComplexDouble* c) final {
    const auto def = c->definition();
    const bool has_alloc = alloc_map_.find(c) != alloc_map_.end();
    if (def != nullptr && !has_alloc) {
      code_ << "(" << genInline(def) << ")";
    } else if (c->isConst()) {
      code_ << "std::complex<double>" << *c->value();
    } else {
      code_ << ir_utils::varName(c);
    }
  }

  void handle(const NamedScalar* ns) final {
    // dim3 components are unsigned int. Cast to signed integer to
    // support negative indexing
    if (ns->getParallelIndex().has_value() ||
        ns->getParallelDim().has_value()) {
      code_ << "((nvfuser_index_t)" << ns->name() << ")";
    } else if (ns->definition() == nullptr) {
      code_ << ns->name();
    } else {
      const bool has_alloc = alloc_map_.find(ns) != alloc_map_.end();
      if (!has_alloc) {
        code_ << genInline(ns->definition());
      } else {
        code_ << ns->name();
      }
    }
  }

  void handle(const kir::TensorIndex* ti) final {
    bool is_volatile = ti->view()->getMemoryType() == MemoryType::Global &&
        kernel_->summary().sync_map->needsRawSync(ti->view()).hasBID();
    if (is_volatile) {
      code_ << "*(volatile " << ti->getDataType().value() << "*)&";
    }
    code_ << ir_utils::varName(ti->view()) << "[" << genInline(ti->index())
          << "]";
  }

  void handle(const ViewAsScalar* sv) final {
    indent() << gen(sv->output(0)) << " = " << gen(sv->input(0)) << "["
             << gen(sv->index()) << "];\n";
  }

  void handle(const IterDomain*) final {
    TORCH_INTERNAL_ASSERT(false, "Unreachable");
  }

  void handle(const TensorDomain*) final {
    TORCH_INTERNAL_ASSERT(false, "Unreachable");
  }

  void handle(const TensorView*) final {
    TORCH_INTERNAL_ASSERT(false, "Unreachable");
  }

  //! Utility for generating vectorized pointer access in ldsm and
  //!  cpasync.
  //! TODO: this access pattern as is could be merged with exisiting
  //!  vectorization handling logic but this path will be updated in
  //!  follow ups to optimize the generated assembly so keeping them
  //!  separate path for now.
  std::string genVectorPointer(Val* val, DataType dtype, size_t vec_size) {
    std::stringstream ss;

    ss << "reinterpret_cast<Array<" << dtype << "," << vec_size << ","
       << vec_size << ">*>(&" << gen(val) << ")";

    return ss.str();
  }

  // Utility function to emit a cp.async intrinsic
  void genCpAsync(const LoadStoreOp* ldst, size_t vec_size) {
    auto dtype = ldst->in()->getDataType().value();
    bool is_cg = ldst->opType() == LoadStoreOpType::CpAsyncCg;

    if (is_cg) {
      indent() << "Ampere::cpAsyncCg";
    } else {
      indent() << "Ampere::cpAsyncCa";
    }

    if (ldst->predicate() == nullptr) {
      // Out of line predicate variant
      code_ << "<" << dtype << ", " << vec_size << ">("
            << genInline(ldst->out()->as<kir::TensorIndex>()->index()) << ","
            << genInline(ldst->in()->as<kir::TensorIndex>()->index()) << ");\n";
    } else {
      // Inline predicate variant
      code_ << "<" << dtype << ", " << vec_size << ">("
            << genInline(ldst->out()->as<kir::TensorIndex>()->index()) << ","
            << genInline(ldst->in()->as<kir::TensorIndex>()->index()) << ","
            << genInline(ldst->predicate()) << ");\n";
    }
  }

  void genLdMatrix(const LoadStoreOp* ldst, size_t vector_word_size) {
    auto dtype = ldst->in()->getDataType().value();
    indent() << "Turing::ldMatrix";
    if (ldst->opType() == LoadStoreOpType::LdMatrixTranspose) {
      code_ << "T";
    }
    code_ << " (";
    code_ << "*" << genVectorPointer(ldst->out(), dtype, vector_word_size)
          << "," << genInline(ldst->in()->as<kir::TensorIndex>()->index())
          << ");\n";
  }

  void handle(const kir::BaseAddress* sop) final {
    if (!print_inline_) {
      indent() << gen(sop->output(0)) << " = ";
    }
    switch (sop->tv()->getMemoryType()) {
      case MemoryType::Shared:
        code_ << "toSmem(" << ir_utils::varName(sop->tv()) << ")";
        break;
      case MemoryType::Global:
        code_ << ir_utils::varName(sop->tv()) << ".data";
        break;
      default:
        TORCH_INTERNAL_ASSERT(false, "Unsupported input for kir::BaseAddress");
    }
    if (!print_inline_) {
      code_ << ";\n";
    }
  }

  void handle(const UnaryOp* uop) final {
    const auto op_type = uop->getUnaryOpType();

    if (!print_inline_) {
      indent() << gen(uop->out());
      if (!uop->out()->isScalar() && !uop->in()->isScalar()) {
        code_ << "\n";
        indent() << kTab;
      }
      code_ << " = ";
    }

    if (auto op = inline_op_str(op_type)) {
      if (alsoBooleanOperator(op_type) &&
          uop->out()->dtype() == DataType::Bool) {
        code_ << stringifyBooleanOp(op_type) << gen(uop->in());
      } else {
        code_ << *op << gen(uop->in());
      }
    } else {
      if (op_type == UnaryOpType::Cast) {
        const auto cast_str =
            cast_func_str({uop->in()->dtype(), uop->out()->dtype()});
        TORCH_INTERNAL_ASSERT(
            cast_str.has_value(),
            "Invalid cast. Input type: ",
            uop->in()->dtype(),
            ", output type: ",
            uop->out()->dtype());
        code_ << cast_str.value();
      } else {
        code_ << op_type;
        if (needFloatSuffix(op_type) &&
            uop->out()->dtype() == DataType::Float) {
          code_ << "f";
        }
      }

      code_ << "(" << gen(uop->in()) << ")";
    }

    if (!print_inline_) {
      code_ << ";\n";
    }
  }

  void handle(const RNGOp* rop) final {
    // TODO: TORCH_INTERNAL_ASSERT that the scheduler correctly creates an
    // innermost ID of size 4 (float) or size 2 (double)?
    auto index = genInline(rop->getPhiloxIndex());
    int multiple = rop->getPhiloxMultiple();
    indent() << "nvfuser_index_t linear_index" << rop->name() << " = " << index
             << ";\n";
    indent() << "nvfuser_index_t rng_subseq" << rop->name() << " = linear_index"
             << rop->name() << " / " << multiple << ";\n";
    indent() << "nvfuser_index_t rng_component" << rop->name()
             << " = linear_index" << rop->name() << " % " << multiple << ";\n";
    indent() << "nvfuser_index_t rng_offset" << rop->name() << " = "
             << rop->getRNGOffset() << ";\n";
    indent() << "if (rng_subseq != rng_subseq" << rop->name()
             << " || rng_offset != rng_offset" << rop->name() << ") {\n";
    indent() << "  auto seed = philox_args.captured_ ?\n"
             << "      static_cast<uint64_t>(*(philox_args.seed_.ptr)) : \n"
             << "      philox_args.seed_.val;\n";
    indent() << "  rng_result = philox(seed, rng_subseq" << rop->name()
             << ", philox_offset / 4 + rng_offset" << rop->name() << ");\n";
    indent() << "  rng_subseq = rng_subseq" << rop->name() << ";\n";
    indent() << "  rng_offset = rng_offset" << rop->name() << ";\n";
    indent() << "}\n";
    auto op_type = rop->getRNGOpType();
    indent() << gen(rop->output(0)) << " = " << op_type;
    if (needFloatSuffix(op_type) && rop->dtype() == DataType::Float) {
      code_ << "f";
    }
    code_ << "(rng_result, rng_component" << rop->name();
    switch (op_type) {
      case RNGOpType::UniformRange: {
        auto parameters = rop->getParameters();
        TORCH_INTERNAL_ASSERT(parameters.size() == 2);
        code_ << ", " << gen(parameters[0]) << ", " << gen(parameters[1]);
        break;
      }
      case RNGOpType::NormalGeneral: {
        auto parameters = rop->getParameters();
        TORCH_INTERNAL_ASSERT(parameters.size() == 2);
        code_ << ", " << gen(parameters[0]) << ", " << gen(parameters[1]);
        break;
      }
      default:;
    }
    code_ << ");\n";
  }

  std::string genBinaryOp(
      BinaryOpType op_type,
      DataType data_type,
      const std::string& lhs,
      const std::string& rhs) {
    std::stringstream expr;
    if (auto op = inline_op_str(op_type)) {
      expr << lhs << " ";
      if (alsoBooleanOperator(op_type) && data_type == DataType::Bool) {
        expr << stringifyBooleanOp(op_type);
      } else {
        expr << *op;
      }
      expr << " " << rhs;
    } else {
      if (integer_op_str(op_type) && isIntegralType(data_type)) {
        auto int_op = integer_op_str(op_type);
        expr << *int_op;
      } else if (bool_op_str(op_type) && isBooleanType(data_type)) {
        auto bool_op = bool_op_str(op_type);
        expr << *bool_op;
      } else {
        expr << op_type;
        if (needFloatSuffix(op_type) && data_type == DataType::Float) {
          expr << "f";
        }
      }
      expr << "(" << lhs << ", " << rhs << ")";
    }
    return expr.str();
  }

  // If one argument is a tensorview and the other is a scalar, make sure we
  // cast the scalar to the tensorview type
  std::string scalarCast(Val* lhs, Val* rhs) {
    // If neither are scalars return
    if (!((lhs->isScalar() || rhs->isScalar()) &&
          (lhs->isA<kir::TensorIndex>() || rhs->isA<kir::TensorIndex>()))) {
      return "";
    }

    // Looking for mixed tensorview scalar options where types don't match
    // but are either both floating or both int types. We should cast
    // scalar to tensorview type in these instances.
    auto lhs_t = lhs->dtype();
    auto rhs_t = rhs->dtype();

    // If same type, don't cast anything
    if (lhs_t == rhs_t) {
      return "";
    }

    // Don't do anything when dealing with bools
    if (lhs_t == DataType::Bool || rhs_t == DataType::Bool) {
      return "";
    }

    // Mixing floating and int combination
    if ((isFloatingPointType(lhs_t) != isFloatingPointType(rhs_t)) ||
        (isIntegralType(lhs_t) != isIntegralType(rhs_t))) {
      return "";
    }

    std::stringstream cast;
    cast << "(" << (lhs->isA<kir::TensorIndex>() ? lhs_t : rhs_t) << ") ";
    return cast.str();
  }

  // If possible, replace pow with mul. Return true when successful.
  bool genPowerWithMul(const BinaryOp* bop) {
    if (bop->getBinaryOpType() != BinaryOpType::Pow) {
      return false;
    }

    auto rhs = bop->rhs();
    c10::optional<double> exponent;
    if (auto val_int = dynamic_cast<Int*>(rhs)) {
      if (val_int->isConst()) {
        exponent = val_int->value().value();
      }
    } else if (auto val_float = dynamic_cast<Double*>(rhs)) {
      if (val_float->isConst()) {
        auto fp_exp = val_float->value().value();
        double int_exp = 0;
        if (std::modf(fp_exp, &int_exp) == 0) {
          exponent = int_exp;
        }
      }
    }

    if (!exponent.has_value()) {
      return false;
    }

    // Only **2 and **3 are considered
    if (!(exponent.value() == 2 || exponent.value() == 3)) {
      return false;
    }

    auto lhs = gen(bop->lhs());

    if (print_inline_) {
      code_ << lhs << " * " << lhs;
      if (exponent.value() == 3) {
        code_ << " * " << lhs;
      }
    } else {
      indent() << gen(bop->out());
      if (bop->out()->isScalar()) {
        code_ << " = " << lhs << " * " << lhs;
        if (exponent.value() == 3) {
          code_ << " * " << lhs;
        }
      } else {
        code_ << "\n";
        indent() << kTab << "= " << lhs << "\n";
        indent() << kTab << "* " << lhs;
        if (exponent.value() == 3) {
          code_ << "\n";
          indent() << kTab << "* " << lhs;
        }
      }
    }

    code_ << ";\n";
    return true;
  }

  void handle(const BinaryOp* bop) final {
    // Try replacing pow with mul
    if (genPowerWithMul(bop)) {
      return;
    }

    const auto op_type = bop->getBinaryOpType();
    if (print_inline_) {
      // Inline expression: `lhs op rhs`
      code_ << genBinaryOp(
          op_type, bop->out()->dtype(), gen(bop->lhs()), gen(bop->rhs()));
    } else {
      indent() << gen(bop->out());
      if (bop->out()->isScalar()) {
        // Single line: `out = lhs op rhs;`
        code_ << " = "
              << genBinaryOp(
                     op_type,
                     bop->out()->dtype(),
                     gen(bop->lhs()),
                     gen(bop->rhs()));
      } else {
        // Split TensorView expressions across multiple lines:
        //
        // out
        //    =  lhs
        //    op rhs;
        //

        auto cast = scalarCast(bop->lhs(), bop->rhs());
        if (auto op = inline_op_str(op_type)) {
          code_ << "\n";
          indent() << kTab << "= " << (bop->lhs()->isScalar() ? cast : "")
                   << gen(bop->lhs()) << "\n";
          indent() << kTab;
          if (alsoBooleanOperator(op_type) &&
              bop->out()->dtype() == DataType::Bool) {
            code_ << stringifyBooleanOp(op_type);
          } else {
            code_ << *op;
          }
          code_ << " " << (bop->rhs()->isScalar() ? cast : "")
                << gen(bop->rhs());
        } else {
          if (integer_op_str(op_type) && isIntegralType(bop->out()->dtype())) {
            auto int_op = integer_op_str(op_type);
            code_ << " = " << *int_op << "(\n";
          } else if (
              bool_op_str(op_type) && isBooleanType(bop->out()->dtype())) {
            auto bool_op = bool_op_str(op_type);
            code_ << " = " << *bool_op << "(\n";
          } else {
            std::stringstream op_str;
            op_str << op_type;
            if (needFloatSuffix(op_type) &&
                bop->out()->dtype() == DataType::Float) {
              op_str << "f";
            }
            code_ << " = " << op_str.str() << "(\n";
          }
          indent() << kTab << (bop->lhs()->isScalar() ? cast : "")
                   << gen(bop->lhs()) << ",\n";
          indent() << kTab << (bop->rhs()->isScalar() ? cast : "")
                   << gen(bop->rhs()) << ")";
        }
      }
      code_ << ";\n";
    }
  }

  void handle(const TernaryOp* top) final {
    if (!print_inline_) {
      indent() << gen(top->out());
      if (!top->out()->isScalar()) {
        code_ << "\n";
        indent() << kTab;
      }
      code_ << " = ";
    }

    // Don't use a runtime device function for where as the second and
    // third aguments should not be evaluated unless picked by the
    // condition. If a device function is implemnted as pass-by-value,
    // both arguments would be evaluated. Could be worked around by
    // pass-by-reference, but it's just simpler to use the C++ ? operator.
    if (top->getTernaryOpType() == TernaryOpType::Where) {
      code_ << gen(top->in1()) << " ? ";
      // Make sure the two operands of where has the same
      // type. Note that compiling "where(0.0f, 0.0)" fails because of
      // the overloading ambiguity.
      auto cast = scalarCast(top->in2(), top->in3());
      code_ << (top->in2()->isScalar() ? cast : "") << gen(top->in2()) << " : "
            << (top->in3()->isScalar() ? cast : "") << gen(top->in3());
    } else {
      code_ << top->getTernaryOpType() << "(" << gen(top->in1()) << ", "
            << gen(top->in2()) << ", " << gen(top->in3()) << ")";
    }

    if (!print_inline_) {
      code_ << ";\n";
    }
  }

  void handle(const IndexSelectOp* sop) final {
    // generate code
    if (!print_inline_) {
      indent() << gen(sop->output(0));
      if (!sop->output(0)->isScalar()) {
        code_ << "\n";
        indent() << kTab;
      }
      code_ << " = ";
    }

    code_ << gen(sop->input(0)) << ";\n";
  }

  void handle(const ScatterOp* sop) final {
    // generate code like T_output[... T_index[...]] = op(T_src[...]);
    if (sop->getScatterOpType() == ScatterOpType::Set) {
      // When value of index_tv are not unique, the behavior of Set is
      // non-deterministic
      indent() << gen(sop->output(0)) << " = " << gen(sop->input(2)) << ";\n";
    } else {
      TORCH_INTERNAL_ASSERT(false, "unkown scatterOp");
    }
  }

  std::string genArchString(MmaOptions::MacroType macro) {
    std::stringstream ss;
    if (isVolta(macro)) {
      ss << "Volta";
    } else if (isTuring(macro)) {
      ss << "Turing";
    } else if (isAmpere(macro)) {
      ss << "Ampere";
    } else {
      TORCH_INTERNAL_ASSERT(false, "mma macro unknown arch");
    }
    return ss.str();
  }

  std::string genMmaOp(const MmaOp* mma, bool init = false) {
    std::stringstream ss;
    auto options = mma->options();
    ss << genArchString(options.macro) << "::";
    if (init) {
      ss << "init";
    }
    ss << toString(options.macro);

    // clang-tidy: bugprone-unchecked-optional-access
    // clang-tidy assumes that function result is unstable, so we need a copy.
    auto mma_layout_opt = mma->layout();
    TORCH_INTERNAL_ASSERT(
        mma_layout_opt.has_value(), "mma unknown input layout");
    if (isTuring(options.macro) || isAmpere(options.macro)) {
      TORCH_INTERNAL_ASSERT(
          mma_layout_opt == MmaOptions::MmaLayout::TN,
          "MMAs in Turing and Ampere are TN only, transpose is handled either "
          "via ldmatrix.trans for fp16 or explicitly for other types.");
    }
    ss << toString(mma_layout_opt.value());
    // TODO: additional parameter could be removed by swizzling iterdomain
    auto acc_stride = mma->accStride();
    TORCH_INTERNAL_ASSERT(acc_stride > 0);
    ss << "<" << acc_stride << ">";
    return ss.str();
  }

  void genMmaOperands(const MmaOp* mma) {
    std::stringstream ss;
    auto options = mma->options();
    auto in_a = mma->inA()->as<kir::TensorIndex>()->view();
    auto dtype = in_a->getDataType().value();
    indent() << kTab << "&(reinterpret_cast<Array<" << dtype << ","
             << getInputARegisterSize(options.macro) << ","
             << getInputARegisterSize(options.macro) << ">*>(&"
             << ir_utils::varName(mma->inA()->as<kir::TensorIndex>()->view())
             << ")[" << genInline(mma->inA()->as<kir::TensorIndex>()->index())
             << "])"
             << ",\n";
    indent() << kTab << "&(reinterpret_cast<Array<" << dtype << ","
             << getInputBRegisterSize(options.macro) << ","
             << getInputBRegisterSize(options.macro) << ">*>(&"
             << ir_utils::varName(mma->inB()->as<kir::TensorIndex>()->view())
             << ")[" << genInline(mma->inB()->as<kir::TensorIndex>()->index())
             << "])";
  }

  void genMmaInitialization(const MmaOp* mma, const LoadStoreOp* ldst) {
    auto options = mma->options();

    indent() << genMmaOp(mma, true) << "(reinterpret_cast<Array<"
             << mma->out()->getDataType().value() << ","
             << getOutputRegisterSize(options.macro) << ","
             << getOutputRegisterSize(options.macro) << ">*>"
             << "(&" << gen(ldst->out()) << "));\n";
  }

  void handle(const MmaOp* mma) final {
    auto options = mma->options();
    auto out = mma->out()->as<kir::TensorIndex>();
    indent() << genMmaOp(mma) << "(\n";
    indent() << kTab << "reinterpret_cast<Array<"
             << out->view()->getDataType().value() << ","
             << getOutputRegisterSize(options.macro) << ","
             << getOutputRegisterSize(options.macro) << ">*>(&"
             << gen(mma->out()) << "),\n";
    genMmaOperands(mma);
    code_ << ");\n";
  }

  std::string genReductionOp(BinaryOpType op_type, DataType data_type) {
    std::stringstream lambda;
    lambda << "[](" << data_type << " &a, " << data_type << " b) "
           << "{ a = " << genBinaryOp(op_type, data_type, "a", "b") << "; }";
    return lambda.str();
  }

  void handle(const BroadcastOp* stmt) final {
    TORCH_INTERNAL_ASSERT(stmt->out()->isA<kir::TensorIndex>());

    const ParallelTypeBitmap parallel_types =
        kernel_->summary().broadcast_parallel_types.at(stmt);

    if (parallel_types.none()) {
      // Not parallelized
      indent() << gen(stmt->out()) << "\n";
      indent() << kTab << " = " << gen(stmt->in()) << ";\n";
      return;
    }

    TORCH_INTERNAL_ASSERT(
        !parallel_types.hasBID(),
        "Parallel broadcast across blocks should have been translated to a GridBroadcast IR node");

    ArgumentBuilder template_args;
    for (const ParallelType pt : kParallelTypeTIDs) {
      template_args.arg(parallel_types.get(pt));
    }
    template_args.arg(isAligned());

    const auto data_type = stmt->out()->dtype();

    ArgumentBuilder func_args;
    func_args.arg(gen(stmt->out()));
    func_args.arg(gen(stmt->in()));
    func_args.arg(genStaticCast(genPtrType(data_type), "shared_mem"));
    TORCH_INTERNAL_ASSERT(
        stmt->predicate() != nullptr && stmt->predicate()->hasValue());
    func_args.arg(genInline(stmt->predicate()));

    indent() << genCall("broadcast::blockBroadcast", template_args, func_args)
             << ";\n";
  }

  void genSerialReduction(
      const kir::TensorIndex* output,
      const Val* input,
      BinaryOpType reduction_op_type) {
    const auto gen_out = gen(output);
    indent() << gen_out << " = "
             << genBinaryOp(
                    reduction_op_type, output->dtype(), gen_out, gen(input))
             << ";\n";
    return;
  }

  void genWarpReduction(
      const kir::TensorIndex* output,
      const kir::TensorIndex* input,
      const Val* init,
      BinaryOpType reduction_op_type,
      kir::Predicate* read_pred) {
    ArgumentBuilder template_args;
    template_args.arg(kernel_->getWarpPaddedParallelInfo().is_tidx_single_warp);
    template_args.arg(isAligned());

    ArgumentBuilder func_args;
    func_args.arg(gen(output));
    func_args.arg(gen(input));
    func_args.arg(genReductionOp(reduction_op_type, output->dtype()));
    func_args.arg(genStaticCast(genPtrType(output->dtype()), "shared_mem"));
    TORCH_INTERNAL_ASSERT(read_pred != nullptr && read_pred->hasValue());
    func_args.arg(genInline(read_pred));
    func_args.arg(genStaticCast(output->dtype(), genInline(init)));

    indent() << genCall("warp::warpReduceTIDX", template_args, func_args)
             << ";\n";
  }

  void genBlockReduction(
      const kir::TensorIndex* output,
      const kir::TensorIndex* input,
      const Val* init,
      BinaryOpType reduction_op_type,
      kir::Predicate* read_pred,
      kir::Predicate* write_pred) {
    const auto par_domains = ir_utils::getParallelDomains(output);
    // Get parallel reduction domains
    const bool tidx =
        par_domains.find(ParallelType::TIDx) != par_domains.end() &&
        par_domains.at(ParallelType::TIDx)->isReduction();
    const bool tidy =
        par_domains.find(ParallelType::TIDy) != par_domains.end() &&
        par_domains.at(ParallelType::TIDy)->isReduction();
    const bool tidz =
        par_domains.find(ParallelType::TIDz) != par_domains.end() &&
        par_domains.at(ParallelType::TIDz)->isReduction();

    const auto data_type = output->dtype();

    ArgumentBuilder template_args;
    template_args.arg(tidx).arg(tidy).arg(tidz);
    template_args.arg(isAligned());

    ArgumentBuilder func_args;
    func_args.arg(gen(output));
    func_args.arg(gen(input));
    func_args.arg(genReductionOp(reduction_op_type, output->dtype()));
    func_args.arg(genStaticCast(genPtrType(data_type), "shared_mem"));
    TORCH_INTERNAL_ASSERT(read_pred != nullptr && read_pred->hasValue());
    func_args.arg(genInline(read_pred));
    // Pass the write predicate if available and different from the
    // default predicate. The blockReduce runtime function uses the
    // default predicate for both read and write when only the
    // default one is given.
    if (write_pred != nullptr) {
      TORCH_INTERNAL_ASSERT(write_pred->hasValue());
      func_args.arg(genInline(write_pred));
    }
    func_args.arg(genCall(data_type, genInline(init)));

    indent() << genCall("blockReduce", template_args, func_args) << ";\n";
  }

  void handle(const ReductionOp* rop) final {
    TORCH_INTERNAL_ASSERT(rop->out()->isA<kir::TensorIndex>());

    const auto output = rop->out()->as<kir::TensorIndex>();
    const auto input = rop->in()->as<kir::TensorIndex>();
    const auto domain = output->view()->domain();
    const auto op_type = rop->getReductionOpType();

    const bool has_block_reduce = domain->hasBlockReduction();
    const bool has_grid_reduce = domain->hasGridReduction();

    TORCH_INTERNAL_ASSERT(
        !has_grid_reduce,
        "ReductionOp does not support block parallelization. GridReductionOp must be used. ",
        rop->toString());

    if (!has_block_reduce) {
      genSerialReduction(output, input, op_type);
    } else if (
        auto reduction_id = ir_utils::getMaybeWarpReductionDim(output, input)) {
      genWarpReduction(output, input, rop->init(), op_type, rop->predicate());
    } else {
      genBlockReduction(
          output,
          input,
          rop->init(),
          op_type,
          rop->predicate(),
          rop->writePredicate());
    }
  }

  void handle(const LoadStoreOp* ldst) final {
    auto optype = ldst->opType();
    if (ldst->out()->isA<kir::TensorIndex>()) {
      auto out_ti = ldst->out()->as<kir::TensorIndex>();
      auto out_tv = out_ti->view();

      // dispatch mma initialization
      if (std::any_of(
              out_tv->getLeafDomain().begin(),
              out_tv->getLeafDomain().end(),
              [&](IterDomain* id) { return id->isMma(); })) {
        auto mma = dynamic_cast<MmaOp*>(out_tv->definition());
        TORCH_INTERNAL_ASSERT(
            mma != nullptr, "CodeGen: mma op not in mma loop");
        TORCH_INTERNAL_ASSERT(optype == LoadStoreOpType::Set);
        genMmaInitialization(mma, ldst);
        return;
      }

      // Get vectorization information
      bool is_vector_op = false;
      size_t vector_word_size = 1;

      if (vectorize_scope_ && ldst->out()->isA<kir::TensorIndex>()) {
        for (auto id : out_tv->getLeafDomain()) {
          if (!isParallelTypeVectorize(id->getParallelType())) {
            continue;
          }

          TORCH_INTERNAL_ASSERT(
              id->extent()->isConstInt(),
              "Could not evaluate constant value bound to vectorized dim.");

          vector_word_size = id->extent()->evaluateInt();

          is_vector_op = true;
          break;
        }
      }

      if (is_vector_op && !ldst->in()->isScalar()) {
        TORCH_INTERNAL_ASSERT(
            ldst->out()->dtype() == ldst->in()->dtype(),
            "Vectorized store/load requires input and output datatypes match.");
      }

      // dispatch ldmatrix
      if (optype == LoadStoreOpType::LdMatrix ||
          optype == LoadStoreOpType::LdMatrixTranspose) {
        TORCH_INTERNAL_ASSERT(
            is_vector_op, "LdMatrix: Vectorization required: ", ldst);
        genLdMatrix(ldst, vector_word_size);
        return;
      }

      // dispatch cp.async
      if (optype == LoadStoreOpType::CpAsyncCa ||
          optype == LoadStoreOpType::CpAsyncCg) {
        if (optype == LoadStoreOpType::CpAsyncCg) {
          TORCH_INTERNAL_ASSERT(
              is_vector_op && vector_word_size == 8,
              "cp.async.cg only support vectorize 8");
        }
        genCpAsync(ldst, vector_word_size);
        return;
      }

      // dispatch vectorized load/store
      if (is_vector_op) {
        TORCH_INTERNAL_ASSERT(optype == LoadStoreOpType::Set);
        if (ldst->in()->isScalar()) {
          // Note:
          //  Double buffered local tensors need indexed initialization,
          //   so will need to use `arraySet` option.
          if (out_tv->getMemoryType() == MemoryType::Local &&
              !(out_tv->isDoubleBuffered() || out_tv->isCircularBuffered())) {
            // Vectorized initialization, explicit type conversion is needed for
            // complex numbers
            indent() << ir_utils::varName(out_tv) << ".set("
                     << genCall(out_tv->dtype(), gen(ldst->in())) << ");\n";
          } else {
            // Note: currently arraySet option is not vectorized, so it will
            //  rely on auto vectorization pass of cuda compiler.
            indent() << "arraySet<" << out_tv->getDataType().value() << ", "
                     << vector_word_size << ">(&" << gen(ldst->out()) << ", "
                     << "(" << out_tv->getDataType().value() << ")"
                     << gen(ldst->in()) << ");\n";
          }
        } else {
          // Vectorized load
          TORCH_INTERNAL_ASSERT(
              ldst->in()->isA<kir::TensorIndex>(),
              "Invalid input to unary op with tensor output, found: ",
              ldst->in()->toString());

          auto in_tv = ldst->in()->as<kir::TensorIndex>()->view();
          bool localToGlobal = out_tv->getMemoryType() == MemoryType::Global &&
              in_tv->getMemoryType() == MemoryType::Local;

          bool globalToLocal = out_tv->getMemoryType() == MemoryType::Local &&
              in_tv->getMemoryType() == MemoryType::Global;

          bool globalToGlobal = out_tv->getMemoryType() == MemoryType::Global &&
              in_tv->getMemoryType() == MemoryType::Global;

          bool is_volatile_to = out_tv->getMemoryType() == MemoryType::Global &&
              kernel_->summary().sync_map->needsRawSync(out_tv).hasBID();

          bool is_volatile_from =
              in_tv->getMemoryType() == MemoryType::Global &&
              kernel_->summary().sync_map->needsRawSync(in_tv).hasBID();

          if (localToGlobal) {
            indent() << "loadLocalToGlobal<" << ldst->out()->dtype() << ", "
                     << vector_word_size << ", "
                     << (is_volatile_to ? "true" : "false") << ">(";
            code_ << " &" << gen(ldst->out()) << ", &" << gen(ldst->in())
                  << ");\n";
          } else if (globalToLocal) {
            indent() << "loadGlobalToLocal<" << ldst->out()->dtype() << ", "
                     << vector_word_size << ", "
                     << (is_volatile_from ? "true" : "false") << ">(&"
                     << gen(ldst->out()) << ", ";
            code_ << " &" << gen(ldst->in()) << ");\n";
          } else if (globalToGlobal) {
            indent() << "loadGlobalToGlobal<" << ldst->out()->dtype() << ", "
                     << vector_word_size << ", "
                     << (is_volatile_to ? "true" : "false") << ", "
                     << (is_volatile_from ? "true" : "false") << ">(";
            code_ << " &" << gen(ldst->out()) << ", ";
            code_ << " &" << gen(ldst->in()) << ");\n";
          } else {
            indent() << "loadGeneric<" << ldst->out()->dtype() << ", "
                     << vector_word_size << ">(";
            code_ << " &" << gen(ldst->out()) << ", ";
            code_ << " &" << gen(ldst->in()) << ");\n";
          }
        }
        return;
      }
    }

    // Generic set op
    TORCH_INTERNAL_ASSERT(optype == LoadStoreOpType::Set);

    if (!print_inline_) {
      indent() << gen(ldst->out());
      if (!ldst->out()->isScalar() && !ldst->in()->isScalar()) {
        code_ << "\n";
        indent() << kTab;
      }
      code_ << " = ";
    }
    code_ << gen(ldst->in());
    if (!print_inline_) {
      code_ << ";\n";
    }
  }

  void genBlockWelford(const WelfordOp* wop) {
    TORCH_INTERNAL_ASSERT(
        ir_utils::getTvOutput(wop)->domain()->hasBlockReduction(),
        "Not block-parallel WelfordOp: ",
        wop->toString());

    const auto has_grid_reduce =
        ir_utils::getTvOutput(wop)->domain()->hasGridReduction();
    const auto data_type = wop->outAvg()->dtype();
    const auto index_type = wop->outN()->dtype();

    // TODO: Instead of decomposing block and grid-parallel welford
    // into blockWelford and gridWelford calls, thus requiring
    // temporary variables like below, extend gridWelford to
    // support block reductions. gridReduce already supports block
    // reductions as well.

    auto out_avg = has_grid_reduce
        ? "block_result_avg_" + std::to_string(block_reduce_name_)
        : gen(wop->outAvg());
    auto out_var = has_grid_reduce
        ? "block_result_var_" + std::to_string(block_reduce_name_)
        : gen(wop->outVar());
    auto out_n = has_grid_reduce
        ? "block_result_n_" + std::to_string(block_reduce_name_)
        : gen(wop->outN());

    if (has_grid_reduce) {
      // allocate block result
      indent() << data_type << " " << out_avg << " = " << gen(wop->initAvg())
               << ";\n";
      indent() << data_type << " " << out_var << " = " << gen(wop->initVar())
               << ";\n";
      indent() << index_type << " " << out_n << " = " << gen(wop->initN())
               << ";\n";
    }

    const auto par_domains = ir_utils::getParallelDomains(wop->out());
    // Get parallel reduction domains
    const bool tidx =
        par_domains.find(ParallelType::TIDx) != par_domains.end() &&
        par_domains.at(ParallelType::TIDx)->isReduction();
    const bool tidy =
        par_domains.find(ParallelType::TIDy) != par_domains.end() &&
        par_domains.at(ParallelType::TIDy)->isReduction();
    const bool tidz =
        par_domains.find(ParallelType::TIDz) != par_domains.end() &&
        par_domains.at(ParallelType::TIDz)->isReduction();

    ArgumentBuilder template_args;
    template_args.arg(tidx).arg(tidy).arg(tidz);
    template_args.arg(isAligned());

    ArgumentBuilder func_args;
    func_args.arg(out_avg);
    func_args.arg(out_var);
    func_args.arg(out_n);
    func_args.arg(gen(wop->inAvg()));
    // inVar can be ZeroVal, and in that case cast seems necessary
    if (wop->inVar()->isZeroInt()) {
      func_args.arg(genStaticCast(data_type, gen(wop->inVar())));
    } else {
      func_args.arg(gen(wop->inVar()));
    }
    // This seems always necessary
    func_args.arg(genStaticCast(index_type, gen(wop->inN())));
    func_args.arg(genReinterpretCast(genPtrType(data_type), "shared_mem_avg"));
    func_args.arg(genReinterpretCast(genPtrType(data_type), "shared_mem_var"));
    func_args.arg(genReinterpretCast(genPtrType(index_type), "shared_mem_n"));
    TORCH_INTERNAL_ASSERT(wop->predicate() != nullptr);
    TORCH_INTERNAL_ASSERT(
        wop->predicate() != nullptr && wop->predicate()->hasValue());
    func_args.arg(genInline(wop->predicate()));
    if (wop->writePredicate() != nullptr) {
      TORCH_INTERNAL_ASSERT(wop->writePredicate()->hasValue());
      func_args.arg(genInline(wop->writePredicate()));
    }
    func_args.arg(genStaticCast(data_type, 0));

    indent() << genCall("blockWelford", template_args, func_args) << ";\n";
  }

  void handle(const WelfordOp* wop) final {
    TORCH_INTERNAL_ASSERT(wop->out()->isA<kir::TensorIndex>());

    const auto out = wop->out()->as<kir::TensorIndex>();
    const auto& domain = out->view()->domain();

    const auto out_var = wop->outVar();
    const auto out_avg = wop->outAvg();
    const auto out_N = wop->outN();

    const auto in_var = wop->inVar();
    const auto in_avg = wop->inAvg();
    const auto in_N = wop->inN();

    // inVar was allowed to be nullptr. Make sure it isn't.
    TORCH_INTERNAL_ASSERT(
        in_var != nullptr, "Welford var input nullptr not allowed");

    const bool has_block_reduce = domain->hasBlockReduction();
    const bool has_grid_reduce = domain->hasGridReduction();

    if (!has_block_reduce && !has_grid_reduce) {
      indent() << "welfordCombine ("
               << "\n";
      indent() << kTab << gen(out_avg) << ",\n";
      indent() << kTab << gen(out_var) << ",\n";
      indent() << kTab << gen(out_N) << ",\n";
      indent() << kTab << gen(in_avg) << ",\n";
      indent() << kTab << "(" << out_avg->dtype() << ")" << gen(in_var)
               << ",\n";
      indent() << kTab << "(" << out_N->dtype() << ")" << gen(in_N) << ");\n";
    } else if (has_block_reduce) {
      genBlockWelford(wop);
    }
  }

  void handle(const kir::VectorizedWelfordOp* wop) final {
    const auto out_var = wop->outVar();
    const auto out_avg = wop->outAvg();
    const auto out_N = wop->outN();
    const auto in_avg = wop->inAvg();

    bool output_gmem =
        std::any_of(wop->outputs().begin(), wop->outputs().end(), [](Val* out) {
          return out->as<kir::TensorIndex>()->view()->getMemoryType() ==
              MemoryType::Global;
        });

    auto pred_bool = wop->hoistedPredicate()->getBool();
    bool is_predicated = !(pred_bool.has_value() && pred_bool.value());

    ArgumentBuilder func_args;
    func_args.arg(gen(out_avg));
    func_args.arg(gen(out_var));
    func_args.arg(gen(out_N));
    func_args.arg(gen(in_avg));
    func_args.arg(gen(wop->reciprocalOfCount()));
    func_args.arg(gen(wop->count()));
    if (is_predicated) {
      func_args.arg(gen(wop->hoistedPredicate()));
    }

    ArgumentBuilder template_args;
    template_args.arg(out_avg->getDataType().value());
    if (is_predicated) {
      template_args.arg(output_gmem);
    }

    indent() << genCall("welfordVectorized", template_args, func_args) << ";\n";
  }

  // Support ReductionOp and WelfordOp
  template <typename REDUCTION_OP>
  std::string generateGridReduceTemplateFlags(
      const REDUCTION_OP* rop,
      const ParallelTypeBitmap& thread_pred) {
    TORCH_INTERNAL_ASSERT(
        !rop->isAllreduce(),
        "This is not for the allreduce reduction kernel\n");

    const auto par_domains = ir_utils::getParallelDomains(rop->outputs()[0]);
    ArgumentBuilder flags;
    for (const ParallelType pt : kParallelTypeThreads) {
      const bool parallel_reduction =
          par_domains.find(pt) != par_domains.end() &&
          par_domains.at(pt)->isReduction();
      const bool pred = thread_pred.get(pt);
      TORCH_INTERNAL_ASSERT(
          !(parallel_reduction && pred), "Cannot reduce predicated axis: ", pt);
      bool flag = false;
      if (isParallelTypeBlockDim(pt)) {
        flag = parallel_reduction;
      } else {
        flag = !pred && !parallel_reduction;
      }
      flags.arg(flag);
    }
    return flags.str();
  }

  // TODO: This should replace generateGridReduceTemplateFlags once
  // GridWelford is refactored as GridReduction.
  template <typename REDUCTION_OP>
  std::string generateGridReduceTemplateFlags2(
      const REDUCTION_OP* rop,
      const ParallelTypeBitmap& thread_pred) {
    TORCH_INTERNAL_ASSERT(
        !rop->isAllreduce(),
        "This is not for the allreduce reduction kernel\n");

    const auto par_domains =
        ir_utils::getParallelDomains(ir_utils::getTvOutput(rop));
    ArgumentBuilder flags;
    for (const ParallelType pt : kParallelTypeThreads) {
      const bool parallel_reduction =
          par_domains.find(pt) != par_domains.end() &&
          par_domains.at(pt)->isReduction();
      const bool pred = thread_pred.get(pt);
      TORCH_INTERNAL_ASSERT(
          !(parallel_reduction && pred), "Cannot reduce predicated axis: ", pt);
      flags.arg(parallel_reduction);
    }
    return flags.str();
  }

  void addProfileArguments(ArgumentBuilder& func_args, const Expr* expr) {
    if (isOptionEnabled(EnableOption::KernelProfile) &&
        kernel_->profile().isProfiled(expr)) {
      const auto& buffer_indices =
          kernel_->profile().getIndicesInProfileBuffer(expr);
      auto buffer = kernel_->profile().getBuffer();
      TORCH_INTERNAL_ASSERT(buffer != nullptr);
      for (const auto& index : buffer_indices) {
        func_args.arg(ir_utils::varName(buffer))
            .append("[")
            .append(index)
            .append("]");
      }
    }
  }

  void handle(const kir::GridReduction* grop) final {
    TORCH_INTERNAL_ASSERT(grop->out()->isA<kir::TensorIndex>());

    const auto out = grop->out()->as<kir::TensorIndex>();
    const auto domain = out->view()->domain();
    TORCH_INTERNAL_ASSERT(domain->hasGridReduction());

    const auto data_type = grop->out()->dtype();
    const auto op_type = grop->getReductionOpType();

    TORCH_INTERNAL_ASSERT(
        grop->reduction_buffer()->buffer()->isA<TensorView>());
    TORCH_INTERNAL_ASSERT(grop->sync_buffer()->buffer()->isA<TensorView>());
    const auto work_buffer =
        grop->reduction_buffer()->buffer()->as<TensorView>();
    const auto sync_buffer = grop->sync_buffer()->buffer()->as<TensorView>();

    if (grop->isAllreduce()) {
      generateGridAllreduce(grop);
      return;
    }

    const std::string flags_str =
        generateGridReduceTemplateFlags2(grop, grop->threadPredicate());

    const bool persistent_sync =
        kernel_->summary().has_cooperative_grid_reduction;

    // Since block-level reduction is already done, those dimensions
    // with tidx/y/z being true do not participate in the grid
    // reduction.
    ArgumentBuilder template_args;
    template_args.arg(flags_str).arg(persistent_sync).arg(isAligned());

    ArgumentBuilder func_args(block_nest_level_ + 1, kTab);
    func_args.arg(gen(grop->out()));
    func_args.arg(gen(grop->in()));
    func_args.arg(genReductionOp(op_type, out->dtype()));
    func_args.arg("&").append(ir_utils::varName(work_buffer)).append("[0]");
    func_args.arg("&").append(ir_utils::varName(sync_buffer)).append("[0]");
    func_args.arg(genCall("static_cast", ptrType(data_type), "shared_mem"));
    // read and write predicates
    TORCH_INTERNAL_ASSERT(
        grop->predicate() != nullptr && grop->predicate()->hasValue());
    const auto read_pred = genInline(grop->predicate());
    func_args.arg(read_pred);
    if (grop->writePredicate() != nullptr) {
      TORCH_INTERNAL_ASSERT(grop->writePredicate()->hasValue());
      func_args.arg(genInline(grop->writePredicate()));
    } else {
      func_args.arg(read_pred);
    }
    // Init val
    func_args.arg(genCall(data_type, genInline(grop->init())));
    func_args.arg(genInline(grop->entrance_index()));
    func_args.arg(genInline(grop->entrances()));

    addProfileArguments(func_args, grop);

    indent() << "reduction::gridReduce<" << template_args << ">(\n";
    indent() << kTab << func_args << ");\n";
  }

  std::string genFusedReductionName(const TensorView* reduction_out) {
    return ir_utils::varName(reduction_out) + "_reduction";
  }

  void generateGridAllreduce(const kir::GridReduction* grop) {
    TORCH_INTERNAL_ASSERT(grop->isAllreduce());

    const auto out = grop->out()->as<kir::TensorIndex>();

    const auto data_type = grop->out()->dtype();
    const auto op_type = grop->getReductionOpType();

    const auto work_buffer =
        grop->reduction_buffer()->buffer()->as<TensorView>();
    const auto sync_buffer = grop->sync_buffer()->buffer()->as<TensorView>();

    const auto reduction_name = genFusedReductionName(out->view());

    // template <bool Aligned, typename Func, typename... Types>
    // __device__ __inline__ void reduce(
    //   RefTuple<Types...> out,
    //   const LocalTuple<Types...>& inp,
    //   VolatilePtrTuple<Types...> global_work_buffer,
    //   int64_t* global_sync_buffer, // Allocated as product of all
    //                                // non-participating Grid dimension
    //   PtrTuple<Types...> shared_buf,
    //   bool read_pred, // Prevent reading from out of bounds memory
    //   bool write_pred, // Prevent from writing out of bounds
    //   const LocalTuple<Types...>& init_val,
    //   Func reduction_op);

    ArgumentBuilder template_args;
    template_args.arg(isAligned());

    ArgumentBuilder func_args(block_nest_level_ + 1, kTab);
    // out
    func_args.arg(genCall("RefTuple", data_type, gen(grop->out())));
    // inp
    func_args.arg(genCall("ConstRefTuple", data_type, gen(grop->in())));
    // global_work_buffer
    func_args.arg(genCall(
        "VolatilePtrTuple",
        data_type,
        "&" + ir_utils::varName(work_buffer) + "[0]"));
    // global_sync_buffer
    func_args.arg("&").append(ir_utils::varName(sync_buffer)).append("[0]");
    // shared_buf
    func_args.arg(genCall(
        "PtrTuple",
        data_type,
        genCall("static_cast", ptrType(data_type), "shared_mem")));
    // read and write predicates
    TORCH_INTERNAL_ASSERT(
        grop->predicate() != nullptr && grop->predicate()->hasValue());
    const auto read_pred = genInline(grop->predicate());
    auto write_pred = read_pred;
    if (grop->writePredicate() != nullptr) {
      TORCH_INTERNAL_ASSERT(grop->writePredicate()->hasValue());
      write_pred = genInline(grop->writePredicate());
    }
    func_args.arg(read_pred).arg(write_pred);
    // init_val
    func_args.arg(genCall("LocalTuple", data_type, genInline(grop->init())));
    // reduction_op
    func_args.arg(genReductionOp(op_type, out->dtype()));

    addProfileArguments(func_args, grop);

    indent() << genCall(reduction_name + ".reduce", template_args, func_args)
             << ";\n";
  }

  void handle(const kir::GroupedGridReduction* grouped_grop) final {
    const auto out = ir_utils::getTvOutput(grouped_grop);
    const auto domain = out->domain();
    TORCH_INTERNAL_ASSERT(domain->hasGridReduction());

    TORCH_INTERNAL_ASSERT(
        grouped_grop->sync_buffer()->buffer()->isA<TensorView>());
    const auto sync_buffer =
        grouped_grop->sync_buffer()->buffer()->as<TensorView>();

    if (grouped_grop->isAllreduce()) {
      generateGroupedGridAllreduce(grouped_grop);
      return;
    }

    TORCH_INTERNAL_ASSERT(
        grouped_grop->numHorizontallyGroupedExprs() == 2,
        "Only grouping of 2 reductions is supported. ",
        grouped_grop->toString());

    const std::string flags_str = generateGridReduceTemplateFlags2(
        grouped_grop, grouped_grop->threadPredicate());

    const bool persistent_sync =
        kernel_->summary().has_cooperative_grid_reduction;

    // Since block-level reduction is already done, those dimensions
    // with tidx/y/z being true do not participate in the grid
    // reduction.
    ArgumentBuilder template_args;
    template_args.arg(flags_str).arg(persistent_sync).arg(isAligned());

    ArgumentBuilder func_args(block_nest_level_ + 1, kTab);

    // Append arguments for each reduction
    for (const auto i :
         c10::irange(grouped_grop->numHorizontallyGroupedExprs())) {
      TORCH_INTERNAL_ASSERT(
          grouped_grop->reduction_buffers().at(i)->buffer()->isA<TensorView>());
      const auto work_buffer =
          grouped_grop->reduction_buffers().at(i)->buffer()->as<TensorView>();

      func_args.arg(gen(grouped_grop->output(i)));
      func_args.arg(gen(grouped_grop->input(i)));
      func_args.arg(genCall(
          grouped_grop->output(i)->dtype(),
          genInline(grouped_grop->initVal(i))));
      func_args.arg(genReductionOp(
          grouped_grop->getReductionOpType(i),
          grouped_grop->output(i)->dtype()));
      func_args.arg("&").append(ir_utils::varName(work_buffer)).append("[0]");
    }

    // The rest of the arguments are common between the reductions
    func_args.arg("&").append(ir_utils::varName(sync_buffer)).append("[0]");
    func_args.arg("shared_mem");
    // read and write predicates
    TORCH_INTERNAL_ASSERT(
        grouped_grop->predicate() != nullptr &&
        grouped_grop->predicate()->hasValue());
    const auto read_pred = genInline(grouped_grop->predicate());
    func_args.arg(read_pred);
    if (grouped_grop->writePredicate() != nullptr) {
      TORCH_INTERNAL_ASSERT(grouped_grop->writePredicate()->hasValue());
      func_args.arg(genInline(grouped_grop->writePredicate()));
    } else {
      func_args.arg(read_pred);
    }

    func_args.arg(genInline(grouped_grop->entrance_index()));
    func_args.arg(genInline(grouped_grop->entrances()));

    addProfileArguments(func_args, grouped_grop);

    indent() << "reduction::gridReduceGroup<" << template_args << ">(\n";
    indent() << kTab << func_args << ");\n";
  }

  void handle(const kir::GroupedGridWelford* grouped_gwop) final {
    if (grouped_gwop->isAllreduce()) {
      if (grouped_gwop->useOuterOpt()) {
        generateGroupedGridAllreduceWelfordOuter(grouped_gwop);
      } else {
        generateGroupedGridAllreduceWelford(grouped_gwop);
      }
      return;
    } else {
      TORCH_INTERNAL_ASSERT(
          false, "Non-allreduce grouped grid welford is not yet supported");
    }
  }

  // Enumerates all combinations of index values of grouped
  // loops. Each combination is a vector of loop index values. The
  // length of the vector is the number of grouped loops.
  //
  // Example 1: only one domain of extent 2 is grouped: {{0}, {1}}.
  // Example 2: two domains of extents 2 and 3 are grouped: {{0, 0},
  // {0, 1}, {0, 2}, {1, 0}, {1, 1}, {1, 2}}
  std::vector<std::vector<int64_t>> getGroupedLoopIndexConcreteIntSets() {
    std::vector<std::vector<int64_t>> index_combinationsatoins;

    // Initialize with an empty vector
    index_combinationsatoins.emplace_back();

    // Incrementally build a combinatorial set
    for (const auto loop : grouped_loops_) {
      const auto iter_count = loop->stop()->evaluateInt();
      std::vector<std::vector<int64_t>> new_combinations;
      // Append integers from 0 to iter_count to all the vectors built
      // so far
      for (const auto& index_vec : index_combinationsatoins) {
        for (int64_t i = 0; i < iter_count; ++i) {
          auto index_vec_appended = index_vec;
          index_vec_appended.push_back(i);
          new_combinations.push_back(index_vec_appended);
        }
      }
      index_combinationsatoins = std::move(new_combinations);
    }

    return index_combinationsatoins;
  }

  //! Returns all combinations of maps from index Vals of grouped loops to their
  //! conrete integers.
  std::vector<std::unordered_map<const Int*, int64_t>>
  getLoopIndexReplacementMaps() {
    std::vector<std::unordered_map<const Int*, int64_t>> maps;

    if (grouped_loops_.empty()) {
      std::unordered_map<const Int*, int64_t> empty_map;
      return {empty_map};
    }

    // Vector of indices of grouped loops
    std::vector<Int*> loop_indices;
    std::transform(
        grouped_loops_.begin(),
        grouped_loops_.end(),
        std::back_inserter(loop_indices),
        [](const kir::ForLoop* loop) { return loop->index()->as<Int>(); });

    // All combinations of loop index integer values
    const auto index_val_sets = getGroupedLoopIndexConcreteIntSets();

    // Create maps from loop index Vals to integers
    for (const auto& index_values : index_val_sets) {
      TORCH_INTERNAL_ASSERT(loop_indices.size() == index_values.size());
      std::unordered_map<const Int*, int64_t> index_val_map;
      for (const auto i : c10::irange(loop_indices.size())) {
        auto loop_index = loop_indices.at(i);
        auto index_val = index_values.at(i);
        index_val_map.emplace(loop_index, index_val);
      }
      maps.emplace_back(std::move(index_val_map));
    }

    return maps;
  }

  void generateGroupedGridAllreduce(
      const kir::GroupedGridReduction* grouped_grop) {
    TORCH_INTERNAL_ASSERT(grouped_grop->isAllreduce());

    // There are two dimensions of grouping: horizontal grouping and
    // iteration grouping. The total number of individual reductions
    // is the number of horizontal reductions * the extent of grouped
    // iterations. All of them are packed into a single grid reduction
    // call. The number of reductions is limited, and currently it is
    // simply an error if exceeded. This could be avoided by
    // decomposing grouped_grop into smaller groups within the
    // limit. TODO: Support a larger number of reductions.

    // First, enumerate all combinations of loop index values of
    // grouped IterDomains. If only a single domain is grouped, this
    // is simply just a 1D vector of integer from 0 to extent-1. If
    // two domains are grouped, combinations of two integer vectors
    // are returned. These loop index value vectors are returned as a
    // map from loop index Vals to concrete int values.
    const auto index_replacement_maps = getLoopIndexReplacementMaps();
    const auto num_grouped_iterations = index_replacement_maps.size();

    // This is also checked at the lowering validaiton time, so it
    // isn't strictly necessary.
    TORCH_INTERNAL_ASSERT(
        num_grouped_iterations * grouped_grop->numHorizontallyGroupedExprs() <=
            kMaxNumGroupedReductions,
        "Too many grouped reductions: ",
        grouped_grop->toString(),
        ". Up to ",
        kMaxNumGroupedReductions,
        " reductions are allowed.");

    ArgumentBuilder template_flags;
    template_flags.arg(isAligned());

    ArgumentBuilder types;
    ArgumentBuilder outputs;
    ArgumentBuilder inputs;
    ArgumentBuilder work_bufs;
    ArgumentBuilder init_vals;
    ArgumentBuilder reduction_ops;

    ArgumentBuilder bool_types;
    ArgumentBuilder read_preds;
    ArgumentBuilder write_preds;

    for (const auto expr_index :
         c10::irange(grouped_grop->numHorizontallyGroupedExprs())) {
      const auto data_type = grouped_grop->outputs().at(expr_index)->dtype();
      TORCH_INTERNAL_ASSERT(grouped_grop->reduction_buffers()
                                .at(expr_index)
                                ->buffer()
                                ->isA<TensorView>());

      for (const auto& group_index :
           c10::irange(index_replacement_maps.size())) {
        // Set the index replacement map with the concrete values of
        // indices of grouped loops.
        index_replacement_map_ = index_replacement_maps.at(group_index);

        types.arg(data_type);

        // out
        outputs.arg(gen(grouped_grop->outputs().at(expr_index)));

        // inp
        inputs.arg(gen(grouped_grop->inputs().at(expr_index)));

        // global_work_buffer
        const auto work_buffer = grouped_grop->reduction_buffers()
                                     .at(expr_index)
                                     ->buffer()
                                     ->as<TensorView>();
        // Separate Work buffer is used for each reduction.
        auto work_buffer_offset = group_index == 0
            ? "0"
            : (genInline(grouped_grop->buffer_stride()) + " * " +
               std::to_string(group_index));
        work_bufs.arg("&")
            .append(ir_utils::varName(work_buffer))
            .append("[")
            .append(work_buffer_offset)
            .append("]");
        auto iv = (grouped_grop->initVal(expr_index));
        // Python scalar only has double, int64_t and complex double, there is
        // no float, int32 and complex float. PyTorch scalar has the same design
        // as python scalar, so the dtype might not match explicit type
        // conversion is needed for complex<float> to complex<double>
        if (iv->dtype() != data_type) {
          init_vals.arg(genCall(data_type, gen(iv)));
        } else {
          init_vals.arg(genInline(iv));
        }
        reduction_ops.arg(genReductionOp(
            grouped_grop->getReductionOpType(expr_index),
            grouped_grop->output(expr_index)->dtype()));

        // read and write predicates
        bool_types.arg("bool");
        // Same argument for all inputs. Different predicates would be
        // used when grouping is done across iterations
        TORCH_INTERNAL_ASSERT(
            grouped_grop->predicate() != nullptr &&
            grouped_grop->predicate()->hasValue());
        const auto read_pred = genInline(grouped_grop->predicate());
        read_preds.arg(read_pred);
        if (grouped_grop->writePredicate() != nullptr) {
          TORCH_INTERNAL_ASSERT(grouped_grop->writePredicate()->hasValue());
          write_preds.arg(genInline(grouped_grop->writePredicate()));
        } else {
          write_preds.arg(read_pred);
        }

        index_replacement_map_.clear();
      }
    }

    ArgumentBuilder func_args(block_nest_level_ + 1, kTab);
    func_args.arg(genCall("RefTuple", types, outputs));
    func_args.arg(genCall("ConstRefTuple", types, inputs));
    func_args.arg(genCall("VolatilePtrTuple", types, work_bufs));
    func_args.arg(genCall("LocalTuple", types, init_vals));

    // global_sync_buffer
    const auto sync_buffer =
        grouped_grop->sync_buffer()->buffer()->as<TensorView>();
    func_args.arg("&").append(ir_utils::varName(sync_buffer)).append("[0]");

    // shared_buf
    func_args.arg("shared_mem");

    func_args.arg(genCall("LocalTuple", bool_types, read_preds));
    func_args.arg(genCall("LocalTuple", bool_types, write_preds));

    addProfileArguments(func_args, grouped_grop);

    func_args.arg(reduction_ops);

    indent() << genCall(
                    genFusedReductionName(ir_utils::getTvOutput(grouped_grop)) +
                        ".reduceGroup",
                    template_flags,
                    func_args)
             << ";\n";
  }

  // Mostly the same as the grouped grid redution version
  void generateGroupedGridAllreduceWelford(
      const kir::GroupedGridWelford* grouped_gwop) {
    TORCH_INTERNAL_ASSERT(grouped_gwop->isAllreduce());

    const auto index_replacement_maps = getLoopIndexReplacementMaps();
    const auto num_grouped_iterations = index_replacement_maps.size();

    // This is also checked at the lowering validaiton time, so it
    // isn't strictly necessary.
    TORCH_INTERNAL_ASSERT(
        num_grouped_iterations * grouped_gwop->numHorizontallyGroupedExprs() <=
            kMaxNumGroupedReductions,
        "Too many grouped reductions: ",
        grouped_gwop->toString(),
        ". Up to ",
        kMaxNumGroupedReductions,
        " reductions are allowed.");

    ArgumentBuilder data_types;
    ArgumentBuilder index_types;

    // Note that the data type of var and avg and that of N are the
    // same with all the welford ops since we only support
    // grouping of iterations.
    const auto data_type = grouped_gwop->outputVals().at(0).avg()->dtype();
    const auto index_type = grouped_gwop->outputVals().at(0).N()->dtype();

    std::array<ArgumentBuilder, 3> out_args;
    std::array<ArgumentBuilder, 3> in_args;
    std::array<ArgumentBuilder, 3> init_args;
    std::array<ArgumentBuilder, 3> work_bufs;

    ArgumentBuilder bool_types;
    ArgumentBuilder read_preds;
    ArgumentBuilder write_preds;

    auto output_vals = grouped_gwop->outputVals();
    auto input_vals = grouped_gwop->inputVals();
    auto init_vals = grouped_gwop->initVals();

    for (const auto expr_index :
         c10::irange(grouped_gwop->numHorizontallyGroupedExprs())) {
      const auto& output = output_vals.at(expr_index);
      const auto& input = input_vals.at(expr_index);
      const auto& init = init_vals.at(expr_index);

      for (const auto& group_index :
           c10::irange(index_replacement_maps.size())) {
        // Set the index replacement map with the concrete values of
        // indices of grouped loops.
        index_replacement_map_ = index_replacement_maps.at(group_index);

        data_types.arg(data_type);
        index_types.arg(index_type);

        auto work_buffer_offset = group_index == 0
            ? "0"
            : (genInline(grouped_gwop->buffer_stride()) + " * " +
               std::to_string(group_index));

        // Setup arguments for avg, var, and N
        for (const auto i : c10::irange(3)) {
          out_args[i].arg(gen(output.get(i)));
          in_args[i].arg(gen(input.get(i)));
          init_args[i].arg(gen(init.get(i)));
          const auto work_buffer = grouped_gwop->reduction_buffers()[i]
                                       .at(expr_index)
                                       ->buffer()
                                       ->as<TensorView>();
          work_bufs[i]
              .arg("&")
              .append(ir_utils::varName(work_buffer))
              .append("[")
              .append(work_buffer_offset)
              .append("]");
        }

        // read and write predicates
        bool_types.arg("bool");
        // Same argument for all inputs. Different predicates would be
        // used when grouping is done across iterations
        TORCH_INTERNAL_ASSERT(grouped_gwop->predicate() != nullptr);
        TORCH_INTERNAL_ASSERT(
            grouped_gwop->predicate() != nullptr &&
            grouped_gwop->predicate()->hasValue());
        const auto read_pred = genInline(grouped_gwop->predicate());
        read_preds.arg(read_pred);
        if (grouped_gwop->writePredicate() != nullptr) {
          TORCH_INTERNAL_ASSERT(grouped_gwop->writePredicate()->hasValue());
          write_preds.arg(genInline(grouped_gwop->writePredicate()));
        } else {
          write_preds.arg(read_pred);
        }

        index_replacement_map_.clear();
      }
    }

    ArgumentBuilder func_args(block_nest_level_ + 1, kTab);
    // output
    func_args.arg(genCall("RefTuple", data_types, out_args[0]));
    func_args.arg(genCall("RefTuple", data_types, out_args[1]));
    func_args.arg(genCall("RefTuple", index_types, out_args[2]));
    // input
    func_args.arg(genCall("ConstRefTuple", data_types, in_args[0]));
    func_args.arg(genCall("ConstRefTuple", data_types, in_args[1]));
    func_args.arg(genCall("ConstRefTuple", index_types, in_args[2]));
    // init
    func_args.arg(genCall("LocalTuple", data_types, init_args[0]));
    func_args.arg(genCall("LocalTuple", data_types, init_args[1]));
    func_args.arg(genCall("LocalTuple", index_types, init_args[2]));
    // work buffer
    func_args.arg(genCall("VolatilePtrTuple", data_types, work_bufs[0]));
    func_args.arg(genCall("VolatilePtrTuple", data_types, work_bufs[1]));
    func_args.arg(genCall("VolatilePtrTuple", index_types, work_bufs[2]));
    // global_sync_buffer
    const auto sync_buffer =
        grouped_gwop->sync_buffer()->buffer()->as<TensorView>();
    func_args.arg("&").append(ir_utils::varName(sync_buffer)).append("[0]");

    // shared_buf
    ArgumentBuilder smem_buffer_args;
    smem_buffer_args.arg(
        genCall("reinterpret_cast", ptrType(data_type), "shared_mem_avg"));
    smem_buffer_args.arg(
        genCall("reinterpret_cast", ptrType(data_type), "shared_mem_var"));
    smem_buffer_args.arg(
        genCall("reinterpret_cast", ptrType(index_type), "shared_mem_n"));
    func_args.arg(genCall(
        "PtrTuple",
        ArgumentBuilder().arg(data_type).arg(data_type).arg(index_type),
        smem_buffer_args));

    func_args.arg(genCall("LocalTuple", bool_types, read_preds));
    func_args.arg(genCall("LocalTuple", bool_types, write_preds));

    addProfileArguments(func_args, grouped_gwop);

    ArgumentBuilder func_template_args;
    func_template_args.arg(isAligned());
    func_template_args.arg(
        grouped_gwop->numHorizontallyGroupedExprs() *
        index_replacement_maps.size());
    func_template_args.arg(data_type);
    func_template_args.arg(index_type);

    indent() << genCall(
                    genFusedReductionName(ir_utils::getTvOutput(grouped_gwop)) +
                        ".welfordGroup",
                    func_template_args,
                    func_args)
             << ";\n";
  }

  void generateGroupedGridAllreduceWelfordOuter(
      const kir::GroupedGridWelford* grouped_gwop) {
    TORCH_INTERNAL_ASSERT(grouped_gwop->isAllreduce());

    const auto num_grouped_iterations =
        getGroupedLoopIndexConcreteIntSets().size();

    // This is also checked at the lowering validaiton time, so it
    // isn't strictly necessary.
    TORCH_INTERNAL_ASSERT(
        num_grouped_iterations * grouped_gwop->numHorizontallyGroupedExprs() <=
            kMaxNumGroupedReductions,
        "Too many grouped reductions: ",
        grouped_gwop->toString(),
        ". Up to ",
        kMaxNumGroupedReductions,
        " reductions are allowed.");

    TORCH_INTERNAL_ASSERT(
        grouped_gwop->numHorizontallyGroupedExprs() == 1,
        "Horizontal grouped Welford reduciton is not yet supported: ",
        grouped_gwop->toString());

    const auto data_type = grouped_gwop->outputVals().at(0).avg()->dtype();

    std::array<ArgumentBuilder, 3> out_args;
    std::array<ArgumentBuilder, 3> in_args;
    std::array<ArgumentBuilder, 3> init_args;
    std::array<ArgumentBuilder, 3> work_bufs;

    ArgumentBuilder bool_types;
    ArgumentBuilder read_preds;
    ArgumentBuilder write_preds;

    const auto output = grouped_gwop->outputVals().at(0);
    const auto input = grouped_gwop->inputVals().at(0);

    ArgumentBuilder func_args;

    // outputs
    func_args.arg(ir_utils::varName(output.get(0)));
    func_args.arg(ir_utils::varName(output.get(1)));
    func_args.arg(ir_utils::varName(output.get(2)));
    // inputs
    func_args.arg(ir_utils::varName(input.get(0)));
    func_args.arg(ir_utils::varName(input.get(1)));
    func_args.arg(ir_utils::varName(input.get(2))).append("[0]");

    // global buf
    for (const auto i : c10::irange(3)) {
      const auto work_buffer = grouped_gwop->reduction_buffers()[i]
                                   .at(0)
                                   ->buffer()
                                   ->as<TensorView>();
      func_args.arg("&")
          .append(ir_utils::varName(work_buffer))
          .append("[")
          .append(0)
          .append("]");
    }

    // shared buf
    func_args.arg(
        genCall("reinterpret_cast", ptrType(data_type), "shared_mem"));

    // sync buf
    const auto sync_buffer =
        grouped_gwop->sync_buffer()->buffer()->as<TensorView>();
    func_args.arg("&").append(ir_utils::varName(sync_buffer)).append("[0]");

    addProfileArguments(func_args, grouped_gwop);

    ArgumentBuilder func_template_args;
    func_template_args.arg(isAligned());
    func_template_args.arg(num_grouped_iterations);
    func_template_args.arg(data_type);

    const auto& par_dim_map = kernel_->summary().parallel_dimension_map_;
    TORCH_INTERNAL_ASSERT(par_dim_map.get(ParallelType::TIDx)->isConstInt());
    TORCH_INTERNAL_ASSERT(par_dim_map.get(ParallelType::TIDy)->isConstInt());
    func_template_args.arg(genInline(par_dim_map.get(ParallelType::TIDx)));
    func_template_args.arg(genInline(par_dim_map.get(ParallelType::TIDy)));

    indent() << genCall(
                    genFusedReductionName(ir_utils::getTvOutput(grouped_gwop)) +
                        ".welfordGroupOuter",
                    func_template_args,
                    func_args)
             << ";\n";
  }

  void handle(const kir::GridBroadcast* grop) final {
    const auto bop = grop->broadcast_op();
    TORCH_INTERNAL_ASSERT(bop->out()->isA<kir::TensorIndex>());

    const ParallelTypeBitmap parallel_types =
        kernel_->summary().broadcast_parallel_types.at(bop);

    TORCH_INTERNAL_ASSERT(
        parallel_types.hasBID(),
        "GridBroadcast needs to be used with a broadcast op that is parallelized with the BID parallel types");

    TORCH_INTERNAL_ASSERT(
        grop->broadcast_buffer()->buffer()->isA<TensorView>());
    TORCH_INTERNAL_ASSERT(grop->sync_buffer()->buffer()->isA<TensorView>());
    const auto work_buffer =
        grop->broadcast_buffer()->buffer()->as<TensorView>();
    const auto sync_buffer = grop->sync_buffer()->buffer()->as<TensorView>();

    ArgumentBuilder template_args;
    for (const ParallelType pt : kParallelTypeThreads) {
      template_args.arg(parallel_types.get(pt));
    }
    template_args.arg(isAligned());

    // Since block-level broadcast has not necessarily been performed before
    // this function call, so grid broadcast may be broadcasting across both
    // the grid and the block level.
    ArgumentBuilder func_args;
    func_args.arg(gen(bop->out()));
    func_args.arg(gen(bop->in()));
    func_args.arg("&").append(ir_utils::varName(work_buffer)).append("[0]");
    func_args.arg(ir_utils::varName(sync_buffer));
    TORCH_INTERNAL_ASSERT(
        grop->predicate() != nullptr && grop->predicate()->hasValue());
    func_args.arg(genInline(grop->predicate()));

    indent() << genCall("grid_broadcast::broadcast", template_args, func_args)
             << ";\n";
  }

  void handle(const kir::GridWelford* gwop) final {
    const auto wop = gwop->welford_op();
    TORCH_INTERNAL_ASSERT(wop->outAvg()->isA<kir::TensorIndex>());

    const auto out = wop->out()->as<kir::TensorIndex>();
    const auto domain = out->view()->domain();
    TORCH_INTERNAL_ASSERT(domain->hasGridReduction());

    const auto data_type = out->dtype();
    const auto index_type = wop->outN()->dtype();

    TORCH_INTERNAL_ASSERT(gwop->var_buffer()->buffer()->isA<TensorView>());
    TORCH_INTERNAL_ASSERT(gwop->sync_buffer()->buffer()->isA<TensorView>());

    const auto avg_buffer = gwop->avg_buffer()->buffer()->as<TensorView>();
    const auto var_buffer = gwop->var_buffer()->buffer()->as<TensorView>();
    const auto n_buffer = gwop->N_buffer()->buffer()->as<TensorView>();
    const auto sync_buffer = gwop->sync_buffer()->buffer()->as<TensorView>();

    if (wop->isAllreduce()) {
      generateGridAllreduce(gwop);
      return;
    }

    const bool persistent_sync =
        kernel_->summary().has_cooperative_grid_reduction;

    const std::string flags_str =
        generateGridReduceTemplateFlags(wop, gwop->threadPredicate());

    ArgumentBuilder template_args;
    template_args.arg(flags_str);
    template_args.arg(persistent_sync);
    template_args.arg(isAligned());

    ArgumentBuilder func_args;
    func_args.arg(gen(wop->outAvg()));
    func_args.arg(gen(wop->outVar()));
    func_args.arg(gen(wop->outN()));
    if (domain->hasBlockReduction()) {
      func_args.arg("block_result_avg_").append(block_reduce_name_);
      func_args.arg("block_result_var_").append(block_reduce_name_);
      func_args.arg("block_result_n_").append(block_reduce_name_);
      block_reduce_name_++;
    } else {
      func_args.arg(gen(wop->inAvg()));
      TORCH_INTERNAL_ASSERT(
          wop->inVar() != nullptr, "Welford var input nullptr not allowed");
      func_args.arg(genStaticCast(data_type, gen(wop->inVar())));
      func_args.arg(genStaticCast(index_type, gen(wop->inN())));
    }
    func_args.arg("&").append(ir_utils::varName(avg_buffer)).append("[0]");
    func_args.arg("&").append(ir_utils::varName(var_buffer)).append("[0]");
    func_args.arg("&").append(ir_utils::varName(n_buffer)).append("[0]");
    func_args.arg(ir_utils::varName(sync_buffer));
    func_args.arg(genReinterpretCast(genPtrType(data_type), "shared_mem_avg"));
    func_args.arg(genReinterpretCast(genPtrType(data_type), "shared_mem_var"));
    func_args.arg(genReinterpretCast(genPtrType(index_type), "shared_mem_n"));
    TORCH_INTERNAL_ASSERT(
        gwop->predicate() != nullptr && gwop->predicate()->hasValue());
    auto read_pred = genInline(gwop->predicate());
    func_args.arg(read_pred);
    if (gwop->writePredicate() != nullptr) {
      TORCH_INTERNAL_ASSERT(gwop->writePredicate()->hasValue());
      auto write_pred = genInline(gwop->writePredicate());
      func_args.arg(write_pred);
    } else {
      func_args.arg(read_pred);
    }
    // TODO : init value support or remove.
    func_args.arg(genStaticCast(data_type, 0));
    func_args.arg(genInline(gwop->entrance_index()));
    func_args.arg(genInline(gwop->entrances()));

    indent() << genCall("welford::gridWelford", template_args, func_args)
             << ";\n";
  }

  void generateGridAllreduce(const kir::GridWelford* gwop) {
    const auto wop = gwop->welford_op();
    TORCH_INTERNAL_ASSERT(wop->isAllreduce());

    const auto out = wop->out()->as<kir::TensorIndex>();

    const auto data_type = wop->outAvg()->dtype();
    const auto index_type = wop->outN()->dtype();
    TORCH_INTERNAL_ASSERT(wop->outAvg()->dtype() == wop->outVar()->dtype());

    ArgumentBuilder data_type_args;
    data_type_args.arg(data_type).arg(data_type).arg(index_type);

    const auto sync_buffer = gwop->sync_buffer()->buffer()->as<TensorView>();

    const auto reduction_name = genFusedReductionName(out->view());

    // template <bool Aligned, typename Func, typename... Types>
    // __device__ __inline__ void reduce(
    //   RefTuple<Types...> out,
    //   const LocalTuple<Types...>& inp,
    //   VolatilePtrTuple<Types...> global_work_buffer,
    //   int64_t* global_sync_buffer, // Allocated as product of all
    //                                // non-participating Grid dimension
    //   PtrTuple<Types...> shared_buf,
    //   bool read_pred, // Prevent reading from out of bounds memory
    //   bool write_pred, // Prevent from writing out of bounds
    //   const LocalTuple<Types...>& init_val,
    //   Func reduction_op);

    ArgumentBuilder template_args;
    template_args.arg(isAligned());

    ArgumentBuilder out_args;
    out_args.arg(gen(wop->outAvg()));
    out_args.arg(gen(wop->outVar()));
    out_args.arg(gen(wop->outN()));

    ArgumentBuilder in_args;
    in_args.arg(gen(wop->inAvg()));
    if (wop->inVar() != nullptr) {
      in_args.arg(gen(wop->inVar()));
    } else {
      in_args.arg("(").append(data_type).append(")0");
    }
    in_args.arg(gen(wop->inN()));

    ArgumentBuilder init_args;
    init_args.arg(gen(wop->initAvg()));
    init_args.arg(gen(wop->initVar()));
    init_args.arg(gen(wop->initN()));

    ArgumentBuilder work_buffer_args;
    work_buffer_args.arg("&")
        .append(
            ir_utils::varName(gwop->avg_buffer()->buffer()->as<TensorView>()))
        .append("[0]");
    work_buffer_args.arg("&")
        .append(
            ir_utils::varName(gwop->var_buffer()->buffer()->as<TensorView>()))
        .append("[0]");
    work_buffer_args.arg("&")
        .append(ir_utils::varName(gwop->N_buffer()->buffer()->as<TensorView>()))
        .append("[0]");

    ArgumentBuilder smem_buffer_args;
    smem_buffer_args.arg(
        genCall("reinterpret_cast", ptrType(data_type), "shared_mem_avg"));
    smem_buffer_args.arg(
        genCall("reinterpret_cast", ptrType(data_type), "shared_mem_var"));
    smem_buffer_args.arg(
        genCall("reinterpret_cast", ptrType(index_type), "shared_mem_n"));

    ArgumentBuilder func_args(block_nest_level_ + 1, kTab);
    // out
    func_args.arg(genCall("RefTuple", data_type_args, out_args));
    // inp
    func_args.arg(genCall("ConstRefTuple", data_type_args, in_args));
    // global_work_buffer
    func_args.arg(
        genCall("VolatilePtrTuple", data_type_args, work_buffer_args));
    // global_sync_buffer
    func_args.arg("&").append(ir_utils::varName(sync_buffer)).append("[0]");
    // shared_buf
    func_args.arg(genCall("PtrTuple", data_type_args, smem_buffer_args));
    // read and write predicates
    TORCH_INTERNAL_ASSERT(
        gwop->predicate() != nullptr && gwop->predicate()->hasValue());
    const auto read_pred = genInline(gwop->predicate());
    auto write_pred = read_pred;
    if (gwop->writePredicate() != nullptr) {
      TORCH_INTERNAL_ASSERT(gwop->writePredicate()->hasValue());
      write_pred = genInline(gwop->writePredicate());
    }
    func_args.arg(read_pred).arg(write_pred);
    // init_val
    func_args.arg(genCall("LocalTuple", data_type_args, init_args));
    // reduction_op
    func_args.arg(genTemplate(
        "welfordCombine", ArgumentBuilder().arg(data_type).arg(index_type)));

    indent() << genCall(reduction_name + ".reduce", template_args, func_args)
             << ";\n";
  }

  void handle(const kir::AllocateFusedReduction* alloc_fused_reduction) final {
    // See the runtime file of the fused reduction
    enum class ReductionParallelTypeState { Reduce, Iter, Pred, Inactive };

    using ReductionParallelTypeStateArray =
        ParallelTypeMap<ReductionParallelTypeState>;

    ReductionParallelTypeStateArray states(
        ReductionParallelTypeState::Inactive);

    for (const ParallelType pt : kParallelTypeThreads) {
      // It may be better to predicate grid reductions on dimensions they don't
      // actively use, however since that should generally be discouraged (they
      // should be part of the iter portion of the operation, or they should be
      // predciated out) we're just going to assume they're part of the iter
      // dimension. This would cause more communication than strictly necessary
      // but should not be a common use case.
      auto pt_dim = kernel_->summary().parallel_dimension_map_.get(pt);
      if (pt_dim == nullptr || pt_dim->isOneInt()) {
        continue;
      }
      // Initialize pt_dim if used to an iter dimension. It may change to a
      // reduction or predicated dimension later.
      states[pt] = ReductionParallelTypeState::Iter;
    }

    for (auto id : alloc_fused_reduction->out()->view()->getLeafDomain()) {
      auto pt = id->getParallelType();
      if (isParallelTypeThread(pt)) {
        auto state = id->isReduction() ? ReductionParallelTypeState::Reduce
                                       : ReductionParallelTypeState::Iter;
        states[pt] = state;
      }
    }

    for (const auto predicated_pt : alloc_fused_reduction->threadPredicate()) {
      auto& state = states[predicated_pt];
      TORCH_INTERNAL_ASSERT(
          state != ReductionParallelTypeState::Reduce,
          "Invalid thread predication: ",
          predicated_pt);
      state = ReductionParallelTypeState::Pred;
    }

    ArgumentBuilder flags;
    for (auto pt : kParallelTypeThreads) {
      flags.arg(static_cast<int>(states[pt]));
    }

    // Persistent
    flags.arg(true);

    // Broadcast is fused
    flags.arg(true);

    const auto reduction_name =
        genFusedReductionName(alloc_fused_reduction->out()->view());

    indent() << genTemplate("fused_reduction::ParallelReduce", flags) << " "
             << reduction_name << ";\n";
  }

  void handleTrivialLoop(const kir::ForLoop* loop) {
    if (loop->vectorize()) {
      vectorize_scope_ = true;
    }
    kir::ConstIrVisitor::handle(loop);
    if (loop->vectorize()) {
      vectorize_scope_ = false;
    }
  }

  void handle(const GroupedReductionOp* grouped_rop) final {
    for (const auto i :
         c10::irange(grouped_rop->numHorizontallyGroupedExprs())) {
      TORCH_INTERNAL_ASSERT(grouped_rop->output(i)->isA<kir::TensorIndex>());

      const auto output = grouped_rop->output(i)->as<kir::TensorIndex>();
      const auto input = grouped_rop->input(i)->as<kir::TensorIndex>();
      const auto domain = output->view()->domain();
      const auto op_type = grouped_rop->getReductionOpType(i);

      const bool has_block_reduce = domain->hasBlockReduction();
      const bool has_grid_reduce = domain->hasGridReduction();

      TORCH_INTERNAL_ASSERT(
          !has_grid_reduce,
          "GroupedReductionOp does not support block parallelization. GroupedGridReduction must be used. ",
          grouped_rop->toString());

      if (!has_block_reduce) {
        genSerialReduction(output, input, op_type);
      } else if (
          auto reduction_id =
              ir_utils::getMaybeWarpReductionDim(output, input)) {
        genWarpReduction(
            output,
            input,
            grouped_rop->initVal(i),
            op_type,
            grouped_rop->predicate());
      } else {
        genBlockReduction(
            output,
            input,
            grouped_rop->initVal(i),
            op_type,
            grouped_rop->predicate(),
            grouped_rop->writePredicate());
      }
    }
  }

  void handle(const GroupedWelfordOp* grouped_wop) final {
    TORCH_INTERNAL_ASSERT(
        false,
        "Should not reach here as grouped welford is only enabled for grid welford,",
        " which is handled by its own handler");
  }

  void handle(const kir::ForLoop* loop) final {
    if (loop->isTrivial()) {
      handleTrivialLoop(loop);
      return;
    }

    // If a loop is grouped, no loop is created, but it isn't
    // considered trivial as the loop trip count is not one.
    if (loop->isGroup()) {
      grouped_loops_.push_back(loop);
      kir::ConstIrVisitor::handle(loop);
      grouped_loops_.pop_back();
      return;
    }

    const auto gen_index = gen(loop->index());
    const auto gen_start = genInline(loop->start());
    const auto gen_stop = genInline(loop->simplifiedStop());
    const auto gen_step = genInline(loop->step());

    std::stringstream step_code;
    if (loop->step()->isOneInt()) {
      step_code << "++" << gen_index;
    } else {
      step_code << gen_index << " += " << gen_step;
    }
    if (loop->isUnrolled()) {
      indent() << "#pragma unroll\n";
    } else {
      indent() << "#pragma unroll 1\n";
    }

    indent() << "for(nvfuser_index_t " << gen_index;
    if (loop->iter_domain()->isParallelized()) {
      code_ << " = " << gen_start << "; ";
    } else {
      // Do not start at  the start of the ID when not parallelized. Instead,
      // start at 0. Predicates will protect buffers between 0 and ID->start(),
      // however if we started at ID->start and extent == ID->start, we could
      // have a "degenerate" loop (loop with no iterations). It may not be an
      // issue to have a 0-sized loop, but all potential consequences haven't
      // been covered. One example is WAR analysis which could incorrectly think
      // a barrier inside a 0-sized loop actually provides protection.
      code_ << " = 0; ";
    }
    code_ << gen_index << " < " << gen_stop << "; " << step_code.str() << ") ";
    startBlock(true);
    kir::ConstIrVisitor::handle(loop);
    endBlock();
  }

  void handle(const kir::IfThenElse* ite) final {
    auto conditional = ite->predicate()->value();
    if (conditional->isConst()) {
      // If the conditional is a constant, then the IfThenElse is not required
      if (conditional->value().value()) {
        handle(ite->thenBody().exprs());
      } else {
        handle(ite->elseBody().exprs());
      }
      return;
    }

    pushAlignmentInfo(ite);

    indent() << "if (" << genInline(conditional) << ") ";

    // "then" block
    startBlock(true);
    handle(ite->thenBody().exprs());

    // "else" block (optional)
    if (ite->hasElse()) {
      endBlock(" else ");
      startBlock(true);
      handle(ite->elseBody().exprs());
    }

    endBlock();

    popAlignmentInfo();
  }

  void handle(const kir::Allocate* alloc) final {
    const auto buffer_dtype = alloc->buffer()->dtype();

    TORCH_INTERNAL_ASSERT(alloc->buffer() != nullptr);
    alloc_map_.emplace(alloc->buffer(), alloc);

    if (!alloc->buffer()->isA<TensorView>()) {
      indent() << buffer_dtype << " " << gen(alloc->buffer()) << ";\n";
      return;
    }

    const auto tv = alloc->buffer()->as<TensorView>();

    const auto size = alloc->size();
    TORCH_INTERNAL_ASSERT(size != nullptr);

    if (alloc->alias() != nullptr) {
      // Allocate alias another Allocate stmt
      const auto alias_tv = alloc->alias()->buffer()->as<TensorView>();
      indent() << "// Alias Allocation - " << alloc->memoryType() << "\n";
      indent() << "auto& " << ir_utils::varName(tv) << " = "
               << ir_utils::varName(alias_tv) << ";\n";

    } else {
      // Standard Memory Allocation
      switch (tv->getMemoryType()) {
        case MemoryType::Global:
          indent() << "// Allocate global tensor " << ir_utils::varName(tv)
                   << "\n";
          break;
        case MemoryType::Shared:
          // Align Offset Position
          indent() << "smem_offset = alignBufferSize(smem_offset, "
                   // Always align to 128b / 16B
                   << 16 << ");\n";
          // Shared Memory Pointer
          indent() << buffer_dtype << "* " << ir_utils::varName(tv)
                   << " = reinterpret_cast<" << buffer_dtype << "*>"
                   << "(array + smem_offset);\n";
          // Increment Offset Position
          indent() << "smem_offset += (" << genInline(size) << " * sizeof("
                   << buffer_dtype << "));\n";
          break;
        case MemoryType::Local: {
          auto va = kernel_->summary().vectorized_accesses;
          if (va.find(tv) != va.end()) {
            indent() << "Array<" << buffer_dtype << ", " << genInline(size)
                     << ", " << va.at(tv) << "> " << ir_utils::varName(tv)
                     << ";\n";
          } else {
            indent() << buffer_dtype << " " << ir_utils::varName(tv) << "["
                     << genInline(size) << "];\n";
          }
        } break;
        default:
          TORCH_INTERNAL_ASSERT(false, "Unexpected memory type");
      }
    }
  }

  void handle(const kir::BlockSync* sync) final {
    // Use a custom synchronization method if enabled
    if (std::getenv("PYTORCH_NVFUSER_USE_BLOCK_SYNC_ATOMIC")) {
      indent() << "block_sync::sync();\n";
    } else if (isAligned()) {
      indent() << "__syncthreads();\n";
    } else {
      indent() << "__barrier_sync(0);\n";
    }
  }

  void handle(const kir::CpAsyncWait* cpasync_wait) final {
    if (cpasync_wait->keepStages() > 0) {
      // Perform partial sync, see comment on kir::CpAsyncWait.
      indent() << "Ampere::cpAsyncPartialBarrier<" << cpasync_wait->keepStages()
               << ">();\n";
    } else {
      // Perform sync all, see comment on kir::CpAsyncWait.
      indent() << "Ampere::cpAsyncBarrier();\n";
    }
  }

  void handle(const kir::CpAsyncCommit* cpasync_wait) final {
    // Commit inflight cp.async transfers. See comment on kir::CpAsyncCommit.
    indent() << "Ampere::cpAsyncCommit();\n";
  }

  void handle(const kir::GridSync* sync) final {
    // Use a custom synchronization method if enabled
    bool bidx = sync->syncDims().get(ParallelType::BIDx);
    bool bidy = sync->syncDims().get(ParallelType::BIDy);
    bool bidz = sync->syncDims().get(ParallelType::BIDz);

    ArgumentBuilder sync_call_template_parms;
    sync_call_template_parms.arg(bidx).arg(bidy).arg(bidz).arg(true).arg(
        isAligned());

    auto sync_idx = genCall(
        "index_utils::maskedOffset",
        ArgumentBuilder().arg(!bidx).arg(!bidy).arg(!bidz),
        ArgumentBuilder().arg("blockIdx").arg("gridDim"));

    auto sync_segment_size = genCall(
        "index_utils::maskedSize",
        ArgumentBuilder().arg(bidx).arg(bidy).arg(bidz),
        ArgumentBuilder().arg("gridDim"));

    ArgumentBuilder sync_call_args;
    sync_call_args.arg(ir_utils::varName(sync->syncBuffer()))
        .append("[")
        .append(sync_idx)
        .append("]");
    sync_call_args.arg(sync_segment_size);

    auto sync_call =
        genCall("grid_sync::sync", sync_call_template_parms, sync_call_args);

    indent() << sync_call << ";\n";
  }

  void handle(const kir::InitMagicZero*) final {
    indent() << "NVFUSER_DEFINE_MAGIC_ZERO;\n";
  }

  void handle(const kir::UpdateMagicZero*) final {
    indent() << "NVFUSER_UPDATE_MAGIC_ZERO;\n";
  }

  void handle(const CatOp* cat) final {
    auto out = gen(cat->output(0));

    // Generate code like:
    // if (consumer_idx < producer_0_extent) {
    //   consumer[consumer_idx] = produce_0[producer_idx0];
    // } else if (consumer_idx < producer_1_extent) {
    //   consumer[consumer_idx] = produce_1[producer_idx1];
    // } else if (consumer_idx < producer_2_extent) {
    //   consumer[consumer_idx] = produce_2[producer_idx2];
    // } else {
    //   consumer[consumer_idx] = produce_3[producer_idx3];
    // }

    for (const auto i : c10::irange(cat->inputs().size())) {
      auto inp = cat->input(i)->as<kir::TensorIndex>();
      auto inp_str = gen(inp);
      if (i < cat->inputs().size() - 1) {
        if (i == 0) {
          indent() << "if (";
        } else {
          indent() << "} else if (";
        }
        code_ << gen(cat->getPred((int)i)) << ") {\n";
      } else {
        // last case doesn't need to be predicated
        indent() << "} else {\n";
      }

      indent() << kTab << out << " = " << gen(inp) << ";\n";
    }

    indent() << "}\n";
  }

 private:
  std::stringstream code_;
  const kir::Kernel* kernel_;
  int block_nest_level_ = 0;
  int block_reduce_name_ = 0;
  bool print_inline_ = false;

  // Mark when we are inside of a vectorized for-loop
  bool vectorize_scope_ = false;
  //! Keep track of Allocate node for Val. Used to determine if Val
  //! should be inlined.
  std::unordered_map<const Val*, const kir::Allocate*> alloc_map_;
  //! Keep track of grouped loops
  std::deque<const kir::ForLoop*> grouped_loops_;
  //! Used to replace symbolic indices with concrete values
  std::unordered_map<const Int*, int64_t> index_replacement_map_;
  //! Keep track of thread alignment property
  std::vector<bool> aligned_scope_exprs_;
};

} // namespace

std::string generateCudaKernel(
    const kir::Kernel* kernel,
    const std::string& kernel_name) {
  FUSER_PERF_SCOPE("generateCudaKernel");
  return CudaKernelGenerator::generateKernelDefinition(kernel, kernel_name);
}

} // namespace codegen
} // namespace nvfuser
