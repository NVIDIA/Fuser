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
#include <options.h>
#include <scheduler/mma_utils.h>
#include <scheduler/reduction_utils.h>
#include <type.h>
#include <utils.h>

#include <array>
#include <cmath>
#include <sstream>
#include <typeindex>
#include <vector>

namespace nvfuser {
namespace codegen {

namespace {

std::string ptrType(DataType dt) {
  std::stringstream ss;
  ss << dt << "*";
  return ss.str();
}

bool isTmaType(const DataType& dtype) {
  return std::visit(
      [](auto&& dtype) -> bool {
        using T = std::decay_t<decltype(dtype)>;
        if constexpr (std::is_same_v<T, PointerType>) {
          return isTmaType(*dtype.type);
        }
        if constexpr (std::is_same_v<T, OpaqueType>) {
          return 0 == dtype.name.compare("TensorMap");
        }
        return false;
      },
      dtype.type);
}

//! Utility class to build an argument list
class ArgumentBuilder {
 public:
  //! Build an argument list where each argument is separated with a comma
  ArgumentBuilder() = default;

  //! Build an argument list where each argument has its own line
  ArgumentBuilder(int indent_level, const char* tab) {
    std::stringstream ss;
    for (const auto i : arange(indent_level)) {
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
      const std::string& kernel_name,
      const LaunchParams& lparams) {
    CudaKernelGenerator codegen(kernel);
    codegen.lparams_ = lparams;
    const auto& cbi = kernel->summary().circular_buffer_info;
    codegen.has_warp_specialized_ = cbi.hasWarpSpecialized();
    codegen.warp_specialized_on_ = cbi.getWarpSpecializedOn();
    codegen.has_independent_compute_warp_groups_ =
        cbi.hasIndependentComputeWarpGroups();
    codegen.genDeclaration(kernel_name);
    codegen.startBlock();
    codegen.genPrologue();
    codegen.genBody();
    codegen.endBlock();
    NVF_CHECK(codegen.block_nest_level_ == 0);
    std::stringstream final_code;
    final_code << "// Codegen generated code\n";
    for (const auto& [ns, code] : codegen.utilities_) {
      if (!ns.empty()) {
        final_code << "namespace " << ns << " {\n"
                   << code.str() << "} // namespace " << ns << "\n";
      } else {
        final_code << code.str() << "\n";
      }
    }
    final_code << codegen.code_.str();
    return final_code.str();
  }

 private:
  explicit CudaKernelGenerator(const kir::Kernel* kernel) : kernel_(kernel) {
    initStringStreamFormat(code_);
  }

  // aligned array of registers used in the kernel
  std::unordered_set<Val*> aligned_array_of_regs_;

  using kir::ConstIrVisitor::handle;

  void initStringStreamFormat(std::stringstream& ss) {
    ss.imbue(std::locale("C"));
    ss << std::scientific;
    // Set the default precision as Double
    setPrecision(ss, DataType::Double);
  }

  void setPrecision(std::stringstream& ss, DataType dtype) {
    NVF_ERROR(isFloatingPointType(dtype));
    ss << std::setprecision(max_digits10(dtype));
  }

  std::string getLiteralSuffix(DataType dtype) {
    switch (std::get<PrimDataType>(dtype.type)) {
      case DataType::Float:
      case DataType::Half:
      case DataType::BFloat16:
      case DataType::Float8_e4m3fn:
      case DataType::Float8_e5m2:
      case DataType::Float8_e8m0fnu:
        return "f";
      case DataType::Int:
        // We use the LL suffix for int64_t literals
        // See https://en.cppreference.com/w/cpp/language/integer_literal
        // and https://en.cppreference.com/w/cpp/language/types
        // For 64-bit Unix systems, int is 32-bit, long and long long are 64-bit
        // For 64-bit Windows, int and long are 32-bit, long long are 64-bit
        return "LL";
      case DataType::UInt32:
        return "U";
      case DataType::UInt64:
        return "ULL";
      case DataType::Index:
        return getLiteralSuffix(kernel_->indexType());
      case DataType::Float4_e2m1fn_x2:
        NVF_THROW(
            "Float4_e2m1fn_x2 should be converted to Float4_e2m1fn in fusion "
            "definition");
      default:
        return "";
    }
  }

  std::string genVariableName(const Val* v) {
    if (auto ns = dynamic_cast<const NamedScalar*>(v)) {
      // dim3 components are unsigned int. Cast to signed integer to
      // support negative indexing
      if (ns->getParallelIndex().has_value() ||
          ns->getParallelDim().has_value()) {
        return "((nvfuser_index_t)" + ns->name() + ")";
      } else {
        return ns->name();
      }
    }
    // We keep the name of TensorIndex and TensorView as is, because in our unit
    // tests, we often write code like
    //  auto tv5 = someOp(tv3, tv4);
    // And we want to keep the name of tv5 as T5, so that it is more convenient
    // for debugging.
    if (v->isOneOf<kir::TensorIndex, TensorView>()) {
      return ir_utils::varName(v);
    }
    // Instead of using the original v->name(), we assign variables a new name,
    // so that the variable name in the generated code is not sensitive to the
    // number of intermediates generated by simplifyExpr. This is important
    // because we have multiple unit tests that assert the generated code string
    // match. We need to make sure that the generated variable name does not
    // easily change, so that we don't need to update these tests often. This
    // remapping also make the generated code more readable.
    if (isOptionDisabled(DisableOption::VarNameRemapping)) {
      return ir_utils::varName(v);
    }
    if (val_to_name_.find(v) == val_to_name_.end()) {
      val_to_name_[v] =
          typePrefix(v->dtype()) + std::to_string(val_to_name_.size());
    }
    return val_to_name_.at(v);
  }

  // If the variable is an aligned array, append ".array" to use the reguar
  // array. This avoid the type mismatch in template functions when one of the
  // arguments is an aligned array (Array<T,N>) while the other is a regular
  // array T[N].
  std::string genVariableNameConvertAlignedArray(Val* v) {
    TensorView* tv = nullptr;
    if (v->isA<kir::TensorIndex>()) {
      tv = v->as<kir::TensorIndex>()->view();
    } else if (v->isA<TensorView>()) {
      tv = v->as<TensorView>();
    }
    if (tv &&
        (aligned_array_of_regs_.count(tv) ||
         tv->getMemoryType() == MemoryType::Local)) {
      return genVariableName(tv).append(".array");
    } else {
      return genVariableName(v);
    }
  }

  // Generates the kernel function declaration
  void genDeclaration(const std::string& kernel_name) {
    code_ << "__global__ void ";
    if (kernel_->hasManaged("enable_register_sharing") &&
        kernel_->getManaged<bool>("enable_register_sharing")) {
      int64_t num_threads_per_cta = lparams_.nThreads();
      NVF_ERROR(
          num_threads_per_cta % 128 == 0,
          "The number of threads per CTA is not correctly set, check launch "
          "para",
          lparams_.toString());

      int64_t initial_reg_count =
          getRegPerThreadGivenThreadsPerSM(num_threads_per_cta);
      auto [decreased_reg_count, increased_register_count] =
          kernel_->summary().dec_inc_register_usage;
      NVF_ERROR(
          initial_reg_count >= decreased_reg_count,
          "Undefined behavior to decrease register count from ",
          initial_reg_count,
          " to ",
          decreased_reg_count);
      NVF_ERROR(
          initial_reg_count <= increased_register_count,
          "Undefined behavior to increase register count from ",
          initial_reg_count,
          " to ",
          increased_register_count);

      // leave a space between launch bound and kernel name
      code_ << "__launch_bounds__(/*maxThreadsPerBlock=*/"
            << num_threads_per_cta << ", /*minBlocksPerMultiprocessor=*/1) ";
    }
    if (kernel_->hasManaged("cluster_dims")) {
      auto cluster_dims =
          kernel_->getManaged<std::tuple<int64_t, int64_t, int64_t>>(
              "cluster_dims");
      code_ << "__cluster_dims__(" << std::get<0>(cluster_dims) << ", "
            << std::get<1>(cluster_dims) << ", " << std::get<2>(cluster_dims)
            << ") ";
    }
    code_ << kernel_name << "(";

    std::unordered_set<Val*> unique_args;

    std::vector<Val*> params;

    // Generate parameter declarations
    kernel_params_.reserve(kernel_->parameters().size());
    unsigned int duplicate_counter = 0;
    for (auto i : arange(kernel_->parameters().size())) {
      std::stringstream var_name_ss;
      auto param = kernel_->parameters().at(i);
      kernel_params_.insert(param);

      if (param->isA<TensorView>()) {
        var_name_ss << genVariableName(param->as<TensorView>());
      } else {
        var_name_ss << gen(param);
      }

      // If value is duplicate in arguments change the name to avoid name
      // conflicts in args.
      if (!unique_args.emplace(param).second) {
        var_name_ss << "_duplicate_" << duplicate_counter++;
      }

      if (const auto tv = dynamic_cast<TensorView*>(param)) {
        if (tv->isCpuScalar()) {
          code_ << " CpuScalarTensor<" << param->dtype() << "> "
                << var_name_ss.str();
        } else {
          code_ << "Tensor<" << param->dtype() << ", "
                << TensorDomain::noReductions(tv->getLogicalDomain()).size()
                << ", "
                << TensorDomain::noReductions(tv->getMaybeAllocationDomain())
                       .size()
                << "> " << var_name_ss.str();
        }
      } else {
        NVF_ERROR(param->isScalar()); // NOLINT (LLVM bug 48525)
        if (isTmaType(param->dtype())) {
          code_ << "const __grid_constant__ " << param->dtype() << " "
                << var_name_ss.str();
        } else {
          code_ << param->dtype() << " " << var_name_ss.str();
        }
      }

      if (i + 1 != kernel_->parameters().size()) {
        code_ << ", ";
      }
    }

    code_ << ") ";
  }

  std::string genInlineOrOne(Val* v) {
    return v == nullptr ? "1" : genInline(v);
  }

  // Generates setup code which is executed before the kernel body
  void genPrologue() {
    const auto& kernel_summary = kernel_->summary();

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

      if (has_reductions || has_parallel_welford) {
        indent() << "void* shared_mem = array;\n";
        if (has_dynamic_smem) {
          std::stringstream smem_buf_size_ss;
          const auto& pdim_map = kernel_->summary().parallel_dimension_map;
          auto bdimx =
              genInlineOrOne(pdim_map.getRawCompute(ParallelType::TIDx));
          auto bdimy =
              genInlineOrOne(pdim_map.getRawCompute(ParallelType::TIDy));
          auto bdimz =
              genInlineOrOne(pdim_map.getRawCompute(ParallelType::TIDz));
          smem_buf_size_ss << bdimx << " * " << bdimy << " * " << bdimz
                           << " * sizeof("
                           << kernel_summary.largest_smem_data_type << ")";
          if (has_parallel_welford) {
            smem_buf_size_ss << " * 3";
          }
          if (kernel_summary.num_grouped_iterations > 1) {
            smem_buf_size_ss << " * " << kernel_summary.num_grouped_iterations;
          }
          if (kernel_summary.all_block_reductions_are_warp_reduction) {
            smem_buf_size_ss << " / 32";
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
          // Ensure that smem_offset remains 128-byte aligned, like shared_mem
          indent() << "const unsigned smem_offset = alignBufferSize("
                   << smem_buf_size << ", 128);\n";
        }

        if (has_parallel_welford) {
          // Unpack shared mem pointer
          auto space_type = kernel_summary.largest_smem_data_type;
          indent() << "nvfuser_index_t block_size = "
                      "blockDim.x*blockDim.y*blockDim.z;\n";
          indent() << space_type << " *shared_mem_var = "
                   << "static_cast<" << space_type << "*>("
                   << "shared_mem);\n";
          indent() << space_type
                   << " *shared_mem_avg = shared_mem_var + block_size;\n";
          indent() << space_type
                   << " *shared_mem_n = shared_mem_avg + block_size;\n";
        }
      } else if (has_dynamic_smem) {
        indent() << "const unsigned smem_offset = 0;\n";
      }
    }

    // Call the initialization function if using a custom block sync
    if (getNvFuserEnv("USE_BLOCK_SYNC_ATOMIC")) {
      indent() << "block_sync::init();\n";
    }
  }

  void generateVectorizedLdSt(
      Val* in,
      Val* out,
      CacheOp cache_op,
      int64_t vector_word_size) {
    auto out_tv = out->as<kir::TensorIndex>()->view();
    auto in_tv = in->as<kir::TensorIndex>()->view();

    bool localToGlobal = out_tv->getMemoryType() == MemoryType::Global &&
        in_tv->getMemoryType() == MemoryType::Local;

    bool globalToLocal = out_tv->getMemoryType() == MemoryType::Local &&
        in_tv->getMemoryType() == MemoryType::Global;

    bool globalToGlobal = out_tv->getMemoryType() == MemoryType::Global &&
        in_tv->getMemoryType() == MemoryType::Global;

    bool is_volatile_to = out_tv->getMemoryType() == MemoryType::Global &&
        kernel_->summary().sync_map->needsRawSync(out_tv).hasBID();

    bool is_volatile_from = in_tv->getMemoryType() == MemoryType::Global &&
        kernel_->summary().sync_map->needsRawSync(in_tv).hasBID();

    if (localToGlobal) {
      code_ << "loadLocalToGlobal<" << out->dtype() << ", /*vec_size=*/"
            << vector_word_size << ", /*is_volatile=*/"
            << (is_volatile_to ? "true" : "false") << ">(";
      code_ << " &" << gen(out) << ", &" << gen(in) << ")";
    } else if (globalToLocal) {
      code_ << "loadGlobalToLocal<" << out->dtype() << ", /*vec_size=*/"
            << vector_word_size << ", /*is_volatile=*/"
            << (is_volatile_from ? "true" : "false") << ", "
            << "CacheOp::" << cache_op << ">(&" << gen(out) << ", ";
      code_ << " &" << gen(in) << ")";
    } else if (globalToGlobal) {
      code_ << "loadGlobalToGlobal<" << out->dtype() << ", /*vec_size=*/"
            << vector_word_size << ", /*is_volatile_to=*/"
            << (is_volatile_to ? "true" : "false") << ", /*is_volatile_from=*/"
            << (is_volatile_from ? "true" : "false") << ">(";
      code_ << " &" << gen(out) << ", ";
      code_ << " &" << gen(in) << ")";
    } else {
      code_ << "loadGeneric<" << out->dtype() << ", " << vector_word_size
            << ">(";
      code_ << " &" << gen(out) << ", ";
      code_ << " &" << gen(in) << ")";
    }
  }

  // Cannot just use ConstIrVisitor::handle as it expects a vector of
  // const Expr*, whereas most of the IR API returns a vector of
  // non-const Expr*.
  void handle(const std::vector<Expr*>& exprs) {
    for (Expr* expr : exprs) {
      kir::ConstIrVisitor::dispatch(expr);
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
    NVF_CHECK(block_nest_level_ >= 0);
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
    for (const auto i : arange(block_nest_level_)) {
      (void)i; // Suppress unused variable warning
      code_ << kTab;
    }
    return code_;
  }

  std::string gen(const Statement* stmt) {
    if (stmt->isA<Expr>()) {
      // This expr should just be an individul expr with no nested
      // scope
      NVF_ERROR(
          !stmt->isA<kir::IfThenElse>() && !stmt->isA<ForLoop>(),
          "Invalid expr: ",
          stmt->toString());
    } else {
      NVF_ERROR(
          stmt->isA<Val>(), "Unknown Statement IR type: ", stmt->toString());
    }

    std::stringstream tmp_code;
    initStringStreamFormat(tmp_code);
    std::swap(tmp_code, code_);
    dispatch(stmt);
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
    NVF_ERROR(pred->hasValue());
    code_ << gen(pred->value());
  }

  void stringify(const PolymorphicValue& value, const DataType dtype) {
    if (value.is<bool>()) {
      code_ << (value ? "true" : "false");
    } else if (value.is<int64_t>()) {
      if (isUnsignedIntegralType(dtype)) {
        code_ << (uint64_t)value << getLiteralSuffix(dtype);
      } else {
        code_ << value << getLiteralSuffix(dtype);
      }
    } else if (value.is<double>()) {
      auto val = value.as<double>();
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
        setPrecision(code_, dtype);
        code_ << val << getLiteralSuffix(dtype);
      }
    } else if (value.is<std::complex<double>>()) {
      if (dtype == DataType::ComplexFloat) {
        code_ << "std::complex<float>" << value;
      } else {
        NVF_ERROR(dtype == DataType::ComplexDouble);
        code_ << "std::complex<double>" << value;
      }
    } else if (std::holds_alternative<ArrayType>(dtype.type)) {
      // print out the vector.
      code_ << to_str(dtype);
      NVF_ERROR(
          value.is<std::vector>(),
          "Value expected to be a vector",
          dtype,
          " ",
          value);
      auto atype = std::get<ArrayType>(dtype.type);
      auto dims = static_cast<int64_t>(value.as<std::vector>().size());
      code_ << "{";
      for (auto i = 0; i < dims; i++) {
        if (i > 0) {
          code_ << ", ";
        }
        stringify(value[i], *atype.type);
      }
      code_ << "}";
    } else {
      NVF_THROW("Unhandled constant type: ", dtype, " ", value);
    }
  }

  void handle(const Val* s) final {
    // Check the replacement map first. If there's an entry for s, use
    // the corresponding replacement.
    auto replace_it = index_replacement_map_.find(s);
    if (replace_it != index_replacement_map_.end()) {
      code_ << replace_it->second;
      return;
    }
    const auto def = s->definition();
    const bool has_alloc = alloc_set_.find(s) != alloc_set_.end();
    const bool is_param = kernel_params_.find(s) != kernel_params_.end();
    if (def != nullptr && !has_alloc && !is_param) {
      if (def->isOneOf<GetAttr, GetItem, GetMetaData>() ||
          (def->isA<UnaryOp>() &&
           !inline_op_str(def->as<UnaryOp>()->getUnaryOpType()).has_value())) {
        code_ << genInline(def);
      } else {
        code_ << "(" << genInline(def) << ")";
      }
    } else if (s->isConst()) {
      stringify(s->value(), s->dtype());
    } else {
      code_ << genVariableName(s);
    }
  }

  void handle(const NamedScalar* ns) final {
    if (ns->definition() != nullptr &&
        alloc_set_.find(ns) == alloc_set_.end()) {
      code_ << genInline(ns->definition());
    } else {
      code_ << genVariableName(ns);
    }
  }

  void handle(const kir::TensorIndex* ti) final {
    if (isPointerType(ti->index()->dtype())) {
      const bool is_u32_ptr = ti->index()->dtype() == DataType::SMemAddress;
      if (is_u32_ptr) {
        // DataType::SMemAddress is implemented as uint32_t in C++. The problem
        // for this implementation is, the type promotion rule in C++ for
        // uint32_t mismatch with the type promotion rule for
        // DataType::SMemAddress in nvFuser. As a workaround, we always cast to
        // the correct type in the generated code.
        code_ << "(uint32_t)(";
      }
      code_ << genInline(ti->index());
      if (is_u32_ptr) {
        code_ << ")";
      }
      return;
    }

    if (ti->view()->getMemoryType() == MemoryType::Tensor) {
      // Generate code like:
      // (uint32_t)(T2 + Array<uint16_t, 2, 1>{0, 0})
      code_ << "(uint32_t)(" << genVariableName(ti->view()) << " + "
            << genInline(ti->index()) << ")";
      return;
    }

    if (ti->view()->getMemoryType() == MemoryType::Global &&
        kernel_->summary().sync_map->needsRawSync(ti->view()).hasBID()) {
      code_ << "*(volatile " << ti->getDataType().value() << "*)&";
    }

    const bool different_dtype = ti->view()->dtype() != ti->dtype();
    if (different_dtype) {
      code_ << "(*reinterpret_cast<" << ti->getDataType().value() << "*>(&";
    }
    code_ << genVariableName(ti->view()) << "[" << genInline(ti->index())
          << "]";
    if (different_dtype) {
      code_ << "))";
    }
  }

  void handle(const IterDomain*) final {
    NVF_THROW("Unreachable");
  }

  void handle(const TensorDomain*) final {
    NVF_THROW("Unreachable");
  }

  void handle(const TensorView* tv) final {
    code_ << genVariableName(tv);
  }

  void genCpAsyncBulkMaybeTensorTile(const LoadStoreOp* ldst) {
    auto in = ldst->in()->as<kir::TensorIndex>();
    auto out = ldst->out()->as<kir::TensorIndex>();

    auto in_tv = in->view();
    auto out_tv = out->view();

    kir::TensorIndex* gmem_ti = nullptr;
    kir::TensorIndex* smem_ti = nullptr;
    std::string func_name;

    bool is_tensor_tile =
        ldst->opType() == LoadStoreOpType::CpAsyncBulkTensorTile;

    if (out->view()->getMemoryType() == MemoryType::Shared) {
      func_name = is_tensor_tile ? "Hopper::cpAsyncBulkTensorTileG2S"
                                 : "Hopper::cpAsyncBulkG2S";
      NVF_ERROR(
          in_tv->getMemoryType() == MemoryType::Global,
          "Expected input in global for G2S operation");
      smem_ti = out;
      gmem_ti = in;
    } else {
      NVF_ERROR(
          in_tv->getMemoryType() == MemoryType::Shared,
          "Expected input in shared for S2G operation");
      NVF_ERROR(
          out_tv->getMemoryType() == MemoryType::Global,
          "Expected input in shared for S2G operation");
      func_name = is_tensor_tile ? "Hopper::cpAsyncBulkTensorTileS2G"
                                 : "Hopper::cpAsyncBulkS2G";
      smem_ti = in;
      gmem_ti = out;
    }

    ArgumentBuilder func_args;
    func_args.arg(genInline(gmem_ti->index()));
    func_args.arg(genInline(smem_ti->index()));

    indent() << genCall(func_name, func_args) << ";\n";
  }

  void handle(const GetMetaData* gop) final {
    if (print_inline_) {
      code_ << gen(gop->in());
    } else {
      auto out_type = gop->output(0)->dtype();
      std::visit(
          [&](auto&& dtype) {
            using T = std::decay_t<decltype(dtype)>;
            if constexpr (std::is_same_v<T, StructType>) {
              for (const auto& field : dtype.fields) {
                if (!field.used_in_kernel) {
                  continue;
                }
                indent() << gen(gop->output(0)) << "." << field.name << " = "
                         << gen(gop->in()) << "." << field.name << ";\n";
              }
            } else {
              indent() << gen(gop->output(0)) << " = " << gen(gop->in())
                       << ";\n";
            }
          },
          out_type.type);
    }
  }

  void handle(const StructConstruct* sop) final {
    if (!print_inline_) {
      indent() << gen(sop->output(0)) << " = ";
    }
    auto dtype = std::get<StructType>(sop->output(0)->dtype().type);
    code_ << dtype.name << "{ ";
    for (auto i : arange(sop->inputs().size())) {
      if (i > 0) {
        code_ << ", ";
      }
      // TODO: upgrade to C++20 and use dot initialization
      code_ << /*"." << sop->fieldName(i) << " = " <<*/ gen(sop->input(i));
    }
    code_ << " }";
    if (!print_inline_) {
      code_ << ";\n";
    }
  }

  void handle(const GetAttr* gop) final {
    if (!print_inline_) {
      indent() << gen(gop->output(0)) << " = ";
    }
    code_ << gen(gop->struct_()) << "." << gop->attr();
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
      code_ << *op << gen(uop->in());
    } else {
      if (op_type == UnaryOpType::Cast) {
        const auto cast_str =
            cast_func_str({uop->in()->dtype(), uop->out()->dtype()});
        NVF_ERROR(
            cast_str.has_value(),
            "Invalid cast. Input type: ",
            uop->in()->dtype(),
            ", output type: ",
            uop->out()->dtype());
        code_ << cast_str.value();
      } else if (op_type == UnaryOpType::BitCast) {
        code_ << "std::bit_cast<" << uop->out()->dtype() << ">";
      } else if (op_type == UnaryOpType::RefCast) {
        code_ << "(*reinterpret_cast<" << uop->out()->dtype() << "*>(&";
      } else {
        code_ << op_type;
        if (needFloatSuffix(op_type) &&
            uop->out()->dtype() == DataType::Float) {
          code_ << "f";
        }
      }

      code_ << "(" << gen(uop->in()) << ")";
      if (op_type == UnaryOpType::RefCast) {
        code_ << "))";
      }
    }

    if (!print_inline_) {
      code_ << ";\n";
    }
  }

  void handle(const RNGOp* rop) final {
    NVF_THROW("RNGOp should be lowered to kir::RNGOp");
  }

  void handle(const kir::RNGOp* rop) final {
    auto op_type = rop->getRNGOpType();
    indent() << gen(rop->output(0)) << " = " << op_type;
    if (needFloatSuffix(op_type)) {
      if (rop->dtype() == DataType::Float) {
        code_ << "f";
      } else if (rop->dtype() == DataType::BFloat16) {
        code_ << "_bfloat";
      } else if (rop->dtype() == DataType::Half) {
        code_ << "_half";
      }
      // Generate other datatypes in double
    }
    code_ << "(" << gen(rop->input(0));
    for (auto inp_i : arange(1, rop->inputs().size())) {
      code_ << ", " << gen(rop->input(inp_i));
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
      expr << lhs << " " << *op << " " << rhs;
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
    PolymorphicValue exponent;
    if (auto val_int = dynamic_cast<Val*>(rhs)) {
      if (val_int->isConst()) {
        exponent = val_int->value();
      }
    } else if (auto val_float = dynamic_cast<Val*>(rhs)) {
      if (val_float->isConst()) {
        auto fp_exp = val_float->value().as<double>();
        double int_exp = 0;
        if (std::modf(fp_exp, &int_exp) == 0) {
          exponent = int_exp;
        }
      }
    }

    if (!exponent.hasValue()) {
      return false;
    }

    // Only **1, **2 and **3 are considered
    if (!(exponent == 1 || exponent == 2 || exponent == 3)) {
      return false;
    }

    auto lhs = gen(bop->lhs());

    if (print_inline_) {
      for (int i = 0; i < exponent; ++i) {
        if (i != 0) {
          code_ << " * ";
        }
        code_ << lhs;
      }
    } else {
      indent() << gen(bop->out());
      if (bop->out()->isScalar()) {
        for (int i = 0; i < exponent; ++i) {
          if (i == 0) {
            code_ << " = " << lhs;
          } else {
            code_ << " * " << lhs;
          }
        }
      } else {
        for (int i = 0; i < exponent; ++i) {
          if (i == 0) {
            code_ << "\n";
            indent() << kTab << "= " << lhs;
          } else {
            code_ << "\n";
            indent() << kTab << "* " << lhs;
          }
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
          code_ << *op << " " << (bop->rhs()->isScalar() ? cast : "")
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
    // Note: vectorized TernaryOp looks something like:
    //   ```
    //     predicate
    //       ? LoadGlobalToLocal(&dst[0], &in2[index])
    //       : arraySet(&dst[0], in3);
    //   ```
    //
    // Current limitation:
    //   1. only TernaryOpType::Where is supported;
    //   2. predicate needs to be a scalar;
    //   3. output needs to be a TensorView;
    //   4. one and only one of the inputs needs to be a TensorView. (This is
    //   coming from validation analysis.)
    if (top->out()->isA<kir::TensorIndex>()) {
      // Get vectorization information
      auto out_tv = top->out()->as<kir::TensorIndex>()->view();
      int64_t vector_word_size = ir_utils::getVectorizeSize(out_tv);
      bool is_vector_op = vectorize_scope_ && vector_word_size != 1;

      if (is_vector_op) {
        NVF_CHECK(
            top->in1()->isScalar(),
            "predicate should be a scalar for vectorized TernaryOp::where");
        NVF_CHECK(
            !top->out()->isScalar(),
            "scalar output in vectorization isn't supported");
        NVF_CHECK(
            top->getTernaryOpType() == TernaryOpType::Where,
            "vectorization only works on TernaryOp::where");
        indent() << gen(top->in1()) << "\n";
        indent() << kTab << "? ";
        auto vec_load = [&out_tv, &top, &vector_word_size, this](Val* in) {
          if (in->isScalar()) {
            if (out_tv->getMemoryType() == MemoryType::Local &&
                !out_tv->isCircularBuffered()) {
              // Vectorized initialization, explicit type conversion is needed
              // for complex numbers
              code_ << genVariableName(out_tv) << ".set("
                    << genCall(out_tv->dtype(), gen(in)) << ")";
            } else {
              // Note: currently arraySet option is not vectorized, so it will
              //  rely on auto vectorization pass of cuda compiler.
              code_ << "arraySet<" << out_tv->getDataType().value() << ", "
                    << vector_word_size << ">(&" << gen(top->out()) << ", ("
                    << out_tv->getDataType().value() << ")" << gen(in) << ")";
            }
          } else {
            generateVectorizedLdSt(
                in, top->out(), CacheOp::AllLevels, vector_word_size);
          }
        };

        // TODO: should we have the option to specify cache level?
        vec_load(top->in2());
        code_ << "\n";
        indent() << kTab << ": ";
        vec_load(top->in3());
        code_ << ";\n";
        return;
      }
    }

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

  void handle(const ArrayConstruct* aop) final {
    if (!print_inline_) {
      indent() << gen(aop->out()) << " = ";
    }

    code_ << aop->out()->dtype() << "{";
    bool first = true;
    for (auto in : aop->inputs()) {
      if (!first) {
        code_ << ", ";
      }
      first = false;
      code_ << gen(in);
    }
    code_ << "}";

    if (!print_inline_) {
      code_ << ";\n";
    }
  }

  void handle(const GetItem* gop) final {
    if (!print_inline_) {
      indent() << gen(gop->out()) << " = ";
    }

    code_ << gen(gop->array()) << "[" << gen(gop->index()) << "]";

    if (!print_inline_) {
      code_ << ";\n";
    }
  }

  void handle(const IndexSelectOp* sop) final {
    NVF_ERROR(sop->output(0)->isA<kir::TensorIndex>());

    // Get vectorization information
    auto out_tv = sop->output(0)->as<kir::TensorIndex>()->view();
    int64_t vector_word_size = ir_utils::getVectorizeSize(out_tv);
    bool is_vector_op = vectorize_scope_ && vector_word_size != 1;
    // generate vectorized load and return.
    if (is_vector_op) {
      indent();
      generateVectorizedLdSt(
          sop->input(0), sop->output(0), CacheOp::AllLevels, vector_word_size);
      code_ << ";\n";
      return;
    }

    // generate non-vectorized load
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
      NVF_THROW("unkown scatterOp");
    }
  }

  void handle(const ArgsortOp* aop) final {
    NVF_ERROR(isAligned(), "Argsort with divergent threads not supported");

    const auto& parallel_dimension_map =
        kernel_->summary().parallel_dimension_map;

    NVF_ERROR(aop->out()->isA<kir::TensorIndex>());
    const auto output = aop->out()->as<kir::TensorIndex>();

    auto sorted_logical_id = output->view()->getLogicalDomain().at(aop->dim());
    auto sorted_ids = DependencyCheck::getAllValsBetween(
        {sorted_logical_id},
        {output->view()->getLoopDomain().begin(),
         output->view()->getLoopDomain().end()});
    std::vector<IterDomain*> sorted_loop_ids;
    std::ranges::copy_if(
        output->view()->getLoopDomain(),
        std::back_inserter(sorted_loop_ids),
        [&](IterDomain* id) {
          return std::ranges::find(sorted_ids, id) != sorted_ids.end();
        });

    // At this moment, we only support argsort on thread parallelized
    // dimensions. No serial dimension is allowed either.
    ParallelTypeBitmap sorted_parallel_types;
    for (auto id : sorted_loop_ids) {
      NVF_ERROR(
          isParallelTypeThreadDim(id->getParallelType()),
          "Argsort on non-thread dimension is not supported");
      sorted_parallel_types.set(id->getParallelType());
    }

    // TID parallel types must only be used for the sorted IDs with the static
    // dimension.
    ArgumentBuilder template_args;
    for (const auto pt : kParallelTypeTIDs) {
      // Unused parallel type should be just ignored
      if (parallel_dimension_map.get(pt) == nullptr) {
        template_args.arg(1); // BLOCK_DIM
        continue;
      }

      // If a parallel type used in the fusion, it must be also used in the
      // argsort.
      NVF_ERROR(
          sorted_parallel_types.get(pt),
          "Parallel type ",
          pt,
          " used in the fusion must be also used in the argsort");
      // Argsort only supports static dimension for now.
      NVF_ERROR(
          parallel_dimension_map.isExact(pt),
          "Argsort only supports exact dimension for now");
      auto pt_extent = parallel_dimension_map.get(pt);
      NVF_ERROR(
          pt_extent->isConstInt(),
          "Argsort only supports constant dimension for now: ",
          pt_extent->toInlineString());
      template_args.arg(pt_extent->evaluate().as<int64_t>()); // BLOCK_DIM
    }

    for (const auto pt : kParallelTypeTIDs) {
      if (parallel_dimension_map.get(pt) != nullptr) {
        template_args.arg(0); // State::Sort
      } else {
        template_args.arg(1); // State::Iter
      }
    }

    // TODO: support ITEMS_PER_THREAD > 1
    constexpr int items_per_thread = 1;

    const auto input = aop->in()->as<kir::TensorIndex>();

    template_args.arg(input->dtype()); // DataT
    template_args.arg(items_per_thread); // ITEMS_PER_THREAD

    // Call the runtime argsort function
    ArgumentBuilder func_args;
    func_args.arg("*(int64_t(*)[")
        .append(items_per_thread)
        .append("])")
        .append("(")
        .append(
            genVariableName(output) + ".array + " + genInline(output->index()))
        .append(")");
    func_args.arg("*(")
        .append(input->dtype())
        .append("(*)[")
        .append(std::to_string(items_per_thread))
        .append("])")
        .append("(")
        .append(
            genVariableName(input) + ".array + " + genInline(input->index()))
        .append(")");
    func_args.arg(aop->isDescending() ? "true" : "false"); // descending flag
    func_args.arg(genComputeBlockDim());

    indent() << genCall("argsort::blockArgsort", template_args, func_args)
             << ";\n";
  }

  std::string genLoadBlockDim() {
    std::stringstream ss;
    const auto& pdim_map = kernel_->summary().parallel_dimension_map;
    Val* tidx = pdim_map.getRawAsync(ParallelType::TIDx);
    Val* tidy = pdim_map.getRawAsync(ParallelType::TIDy);
    Val* tidz = pdim_map.getRawAsync(ParallelType::TIDz);
    int64_t num_threads = tidx->value().as<int64_t>() +
        tidy->value().as<int64_t>() + tidz->value().as<int64_t>();
    NVF_ERROR(
        num_threads == 128,
        "Expected 128 threads in AsyncWarp, but found ",
        num_threads);
    NVF_ERROR(pdim_map.hasWarpSpecialization());
    ss << "dim3(" << genInlineOrOne(tidx) << ", " << genInlineOrOne(tidy)
       << ", " << genInlineOrOne(tidz) << ")";
    return ss.str();
  }

  std::string genComputeBlockDim() {
    std::stringstream ss;
    const auto& pdim_map = kernel_->summary().parallel_dimension_map;
    if (!pdim_map.hasWarpSpecialization()) {
      ss << "DefaultBlockDim()";
    } else if (kernel_->summary()
                   .circular_buffer_info.hasIndependentComputeWarpGroups()) {
      // NOTE If there are independent compute warp groups, assume there is 128
      // active threads per warp group.
      // TODO Specify the actual shape of the independent warp group, rather
      // than 128 threads in TIDx axis.
      ss << "dim3(128, 1, 1)";
    } else {
      ss << "dim3("
         << genInlineOrOne(pdim_map.getRawCompute(ParallelType::TIDx)) << ", "
         << genInlineOrOne(pdim_map.getRawCompute(ParallelType::TIDy)) << ", "
         << genInlineOrOne(pdim_map.getRawCompute(ParallelType::TIDz)) << ")";
    }
    return ss.str();
  }

  void handle(const TopKOp* top) final {
    NVF_ERROR(isAligned(), "Topk with divergent threads not supported");

    const auto& parallel_dimension_map =
        kernel_->summary().parallel_dimension_map;

    // TopKOp has dual outputs: values and indices
    NVF_ERROR(top->outValues()->isA<kir::TensorIndex>());
    NVF_ERROR(top->outIndices()->isA<kir::TensorIndex>());
    const auto output_values = top->outValues()->as<kir::TensorIndex>();
    const auto output_indices = top->outIndices()->as<kir::TensorIndex>();

    auto sorted_logical_id =
        output_values->view()->getLogicalDomain().at(top->dim());
    auto sorted_ids = DependencyCheck::getAllValsBetween(
        {sorted_logical_id},
        {output_values->view()->getLoopDomain().begin(),
         output_values->view()->getLoopDomain().end()});
    std::vector<IterDomain*> sorted_loop_ids;
    std::ranges::copy_if(
        output_values->view()->getLoopDomain(),
        std::back_inserter(sorted_loop_ids),
        [&](IterDomain* id) {
          return std::ranges::find(sorted_ids, id) != sorted_ids.end();
        });

    // At this moment, we only support topk on thread parallelized
    // dimensions. No serial dimension is allowed either.
    ParallelTypeBitmap sorted_parallel_types;
    for (auto id : sorted_loop_ids) {
      NVF_ERROR(
          isParallelTypeThreadDim(id->getParallelType()),
          "TopK on non-thread dimension is not supported");
      sorted_parallel_types.set(id->getParallelType());
    }

    // TID parallel types must only be used for the sorted IDs with the static
    // dimension.
    ArgumentBuilder template_args;
    for (const auto pt : kParallelTypeTIDs) {
      // Unused parallel type should be just ignored
      if (parallel_dimension_map.get(pt) == nullptr) {
        template_args.arg(1); // BLOCK_DIM
        continue;
      }

      // If a parallel type used in the fusion, it must be also used in the
      // topk.
      NVF_ERROR(
          sorted_parallel_types.get(pt),
          "Parallel type ",
          pt,
          " used in the fusion must be also used in the topk");

      // TopK only supports static dimension for now.
      // TODO: Verify the extent of the parallelized ID is equal to
      // pt_extent. Can be done here, but should be done as part of
      // the lowering verification.
      auto pt_extent = parallel_dimension_map.get(pt);
      NVF_ERROR(
          pt_extent->isConstInt(),
          "TopK only supports constant dimension for now: ",
          pt_extent->toInlineString());
      template_args.arg(pt_extent->evaluate().as<int64_t>()); // BLOCK_DIM
    }

    for (const auto pt : kParallelTypeTIDs) {
      if (parallel_dimension_map.get(pt) != nullptr) {
        template_args.arg(0); // State::Sort
      } else {
        template_args.arg(1); // State::Iter
      }
    }

    // TODO: support ITEMS_PER_THREAD > 1
    constexpr int items_per_thread = 1;

    const auto input = top->in()->as<kir::TensorIndex>();

    template_args.arg(input->dtype()); // DataT
    template_args.arg(items_per_thread); // ITEMS_PER_THREAD

    // Call the runtime topk function with dual outputs
    ArgumentBuilder func_args;

    // First argument: top_values output array
    func_args.arg("*(")
        .append(input->dtype())
        .append("(*)[")
        .append(items_per_thread)
        .append("])")
        .append("(")
        .append(
            genVariableName(output_values) + ".array + " +
            genInline(output_values->index()))
        .append(")");

    // Second argument: top_indices output array
    func_args.arg("*(int64_t(*)[")
        .append(items_per_thread)
        .append("])")
        .append("(")
        .append(
            genVariableName(output_indices) + ".array + " +
            genInline(output_indices->index()))
        .append(")");

    // Third argument: input data array
    func_args.arg("*(")
        .append(input->dtype())
        .append("(*)[")
        .append(std::to_string(items_per_thread))
        .append("])")
        .append("(")
        .append(
            genVariableName(input) + ".array + " + genInline(input->index()))
        .append(")");

    // Fourth argument: k value
    func_args.arg(genInline(top->k()));

    // Fifth argument: largest flag
    func_args.arg(top->isLargest() ? "true" : "false");

    // Sixth argument: sorted flag
    func_args.arg(top->isSorted() ? "true" : "false");

    // Seventh argument: block dimensions
    func_args.arg(genComputeBlockDim());

    indent() << genCall("topk::blockTopK", template_args, func_args) << ";\n";
  }

  std::string genReductionOp(BinaryOpType op_type, DataType data_type) {
    std::stringstream lambda;
    lambda << "[](" << data_type << " &a, " << data_type << " b) "
           << "{ a = " << genBinaryOp(op_type, data_type, "a", "b") << "; }";
    return lambda.str();
  }

  void handle(const BroadcastOp* stmt) final {
    NVF_ERROR(stmt->out()->isA<kir::TensorIndex>());

    const ParallelTypeBitmap parallel_types =
        kernel_->summary().broadcast_parallel_types.at(stmt);

    if (parallel_types.none()) {
      // Not parallelized
      indent() << gen(stmt->out()) << "\n";
      indent() << kTab << " = " << gen(stmt->in()) << ";\n";
      return;
    }

    NVF_ERROR(
        !parallel_types.hasBID(),
        "Parallel broadcast across blocks should have been translated to a "
        "GridBroadcast IR node");

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
    NVF_ERROR(stmt->predicate() != nullptr && stmt->predicate()->hasValue());
    func_args.arg(genInline(stmt->predicate()));
    func_args.arg(genComputeBlockDim());

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
  int64_t getComputeThreadsBdimx() {
    return warp_specialized_on_ == ParallelType::TIDx
        ? lparams_.bdimx() - kWarpSpecializationPaddedThreads
        : lparams_.bdimx();
  }
  void genWarpReduction(
      const kir::TensorIndex* output,
      const kir::TensorIndex* input,
      const Val* init,
      BinaryOpType reduction_op_type,
      kir::Predicate* read_pred,
      std::pair<IterDomain*, IterDomain*> reduction_dims,
      bool is_all_reduce) {
    ArgumentBuilder func_args;
    func_args.arg(gen(output));
    func_args.arg(gen(input));
    func_args.arg(genReductionOp(reduction_op_type, output->dtype()));
    ArgumentBuilder template_args;

    if (has_warp_specialized_ && is_all_reduce) {
      if (has_independent_compute_warp_groups_) {
        func_args.arg(
            genStaticCast(genPtrType(output->dtype()), "shared_mem") + " + " +
            genSmemOffset());
        func_args.arg(genBarrierId(true));

      } else {
        func_args.arg(genStaticCast(genPtrType(output->dtype()), "shared_mem"));
      }
      template_args.arg(
          kernel_->getWarpPaddedParallelInfo().is_tidx_single_warp);
      template_args.arg(/*Aligned=*/false);
      template_args.arg(reduction_scheduler_utils::getComputeBdimx(
          warp_specialized_on_, lparams_.bdimx()));

      indent() << genCall(
                      "warp::staticWarpAllReduceTIDX", template_args, func_args)
               << ";\n";
      return;
    }

    func_args.arg(genStaticCast(genPtrType(output->dtype()), "shared_mem"));
    NVF_ERROR(read_pred != nullptr && read_pred->hasValue());
    func_args.arg(genInline(read_pred));
    func_args.arg(genStaticCast(output->dtype(), genInline(init)));
    func_args.arg(genComputeBlockDim());
    if (reduction_dims.first->getParallelType() == ParallelType::TIDx &&
        reduction_dims.second == nullptr) {
      template_args.arg(
          kernel_->getWarpPaddedParallelInfo().is_tidx_single_warp);
      template_args.arg(isAligned());
      indent() << genCall("warp::warpReduceTIDX", template_args, func_args)
               << ";\n";
    } else if (
        reduction_dims.first->getParallelType() == ParallelType::TIDx &&
        reduction_dims.second->getParallelType() == ParallelType::TIDy) {
      auto bdimx = reduction_dims.first->extent()->evaluate();
      auto bdimy = reduction_dims.second->extent()->evaluate();
      template_args.arg(bdimx);
      template_args.arg(bdimy);
      template_args.arg(isAligned());
      indent() << genCall("warp::warpReduceTIDXY", template_args, func_args)
               << ";\n";
    } else {
      NVF_ERROR(false, "Invalid warp reduction dims");
    }
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
    NVF_ERROR(read_pred != nullptr && read_pred->hasValue());
    func_args.arg(genInline(read_pred));
    // Pass the write predicate if available and different from the
    // default predicate. The blockReduce runtime function uses the
    // default predicate for both read and write when only the
    // default one is given.
    if (write_pred != nullptr) {
      NVF_ERROR(write_pred->hasValue());
      func_args.arg(genInline(write_pred));
    }
    func_args.arg(genCall(data_type, genInline(init)));
    func_args.arg(genComputeBlockDim());

    indent() << genCall("blockReduce", template_args, func_args) << ";\n";
  }

  void handle(const ReductionOp* rop) final {
    NVF_ERROR(rop->out()->isA<kir::TensorIndex>());

    const auto output = rop->out()->as<kir::TensorIndex>();
    const auto input = rop->in()->as<kir::TensorIndex>();
    const auto domain = output->view()->domain();
    const auto op_type = rop->getReductionOpType();

    const bool has_block_reduce = domain->hasBlockReduction();
    const bool has_grid_reduce = domain->hasGridReduction();

    NVF_ERROR(
        !has_grid_reduce,
        "ReductionOp does not support block parallelization. GridReductionOp "
        "must be used. ",
        rop->toString());

    if (!has_block_reduce) {
      genSerialReduction(output, input, op_type);
    } else if (
        auto reduction_ids =
            ir_utils::getMaybeWarpReductionDim(output, input)) {
      genWarpReduction(
          output,
          input,
          rop->init(),
          op_type,
          rop->predicate(),
          reduction_ids.value(),
          rop->isAllreduce());
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
    NVF_ERROR(
        optype != LoadStoreOpType::LdMatrix &&
            optype != LoadStoreOpType::StMatrix &&
            optype != LoadStoreOpType::CpAsync,
        "ldmatrix and cp.async should be lowered as kir::Asm");

    if (ldst->out()->isA<kir::TensorIndex>()) {
      auto out_ti = ldst->out()->as<kir::TensorIndex>();
      auto out_tv = out_ti->view();

      // dispatch mma initialization
      if (std::any_of(
              out_tv->getLoopDomain().begin(),
              out_tv->getLoopDomain().end(),
              [&](IterDomain* id) { return id->isMma(); })) {
        auto mma = dynamic_cast<MmaOp*>(out_tv->definition());
        NVF_ERROR(mma != nullptr, "CodeGen: mma op not in mma loop");
        NVF_ERROR(optype == LoadStoreOpType::Set);
        indent() << "(" << gen(ldst->out()) << ").set(0);\n";
        return;
      }

      // Get vectorization information
      int64_t vector_word_size = ir_utils::getVectorizeSize(out_tv);
      bool is_vector_op = vectorize_scope_ && vector_word_size != 1;

      if (is_vector_op && !ldst->in()->isScalar() &&
          !ir_utils::isLdMatrixOp(ldst)) {
        NVF_ERROR(
            ldst->out()->dtype() == ldst->in()->dtype(),
            "Vectorized store/load requires input and output datatypes match.");
      }

      // dispatch cp.async.bulk.{tensor}
      if (optype == LoadStoreOpType::CpAsyncBulk ||
          optype == LoadStoreOpType::CpAsyncBulkTensorTile) {
        genCpAsyncBulkMaybeTensorTile(ldst);
        return;
      }

      // dispatch vectorized load/store
      if (is_vector_op) {
        NVF_ERROR(optype == LoadStoreOpType::Set);
        if (ldst->in()->isScalar()) {
          // Note:
          //  Circular buffered local tensors need indexed initialization,
          //   so will need to use `arraySet` option.
          if (out_tv->getMemoryType() == MemoryType::Local &&
              !out_tv->isCircularBuffered()) {
            // Vectorized initialization, explicit type conversion is needed for
            // complex numbers
            indent() << genVariableName(out_tv) << ".set("
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
          NVF_ERROR(
              ldst->in()->isA<kir::TensorIndex>(),
              "Invalid input to unary op with tensor output, found: ",
              ldst->in()->toString());

          indent();
          generateVectorizedLdSt(
              ldst->in(), ldst->out(), ldst->cacheOp(), vector_word_size);
          code_ << ";\n";
        }
        return;
      }
    }

    // Generic set op
    NVF_ERROR(optype == LoadStoreOpType::Set);

    if (!print_inline_ &&
        std::holds_alternative<StructType>(ldst->out()->dtype().type)) {
      auto out_type = std::get<StructType>(ldst->out()->dtype().type);
      auto in_type = std::get<StructType>(ldst->in()->dtype().type);
      for (const auto& field : out_type.fields) {
        if (!field.used_in_kernel) {
          continue;
        }
        indent() << gen(ldst->out()) << "." << field.name << " = "
                 << gen(ldst->in()) << "." << field.name << ";\n";
      }
      return;
    }

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
    NVF_ERROR(
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
    NVF_ERROR(wop->predicate() != nullptr);
    NVF_ERROR(wop->predicate() != nullptr && wop->predicate()->hasValue());
    func_args.arg(genInline(wop->predicate()));
    if (wop->writePredicate() != nullptr) {
      NVF_ERROR(wop->writePredicate()->hasValue());
      func_args.arg(genInline(wop->writePredicate()));
    }
    func_args.arg(genStaticCast(data_type, 0));
    func_args.arg(genComputeBlockDim());

    indent() << genCall("blockWelford", template_args, func_args) << ";\n";
  }

  void handle(const WelfordOp* wop) final {
    NVF_ERROR(wop->out()->isA<kir::TensorIndex>());

    const auto out = wop->out()->as<kir::TensorIndex>();
    const auto& domain = out->view()->domain();

    const auto out_var = wop->outVar();
    const auto out_avg = wop->outAvg();
    const auto out_N = wop->outN();

    const auto in_var = wop->inVar();
    const auto in_avg = wop->inAvg();
    const auto in_N = wop->inN();

    // inVar was allowed to be nullptr. Make sure it isn't.
    NVF_ERROR(in_var != nullptr, "Welford var input nullptr not allowed");

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

    auto pred_bool = wop->hoistedPredicate()->value();
    bool is_predicated = !(pred_bool.hasValue() && pred_bool.as<bool>());

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
    NVF_ERROR(
        !rop->isAllreduce(),
        "This is not for the allreduce reduction kernel\n");

    const auto par_domains = ir_utils::getParallelDomains(rop->outputs()[0]);
    ArgumentBuilder flags;
    for (const ParallelType pt : kParallelTypeThreads) {
      const bool parallel_reduction =
          par_domains.find(pt) != par_domains.end() &&
          par_domains.at(pt)->isReduction();
      const bool pred = thread_pred.get(pt);
      NVF_ERROR(
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
    NVF_ERROR(
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
      NVF_ERROR(
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
      NVF_ERROR(buffer != nullptr);
      for (const auto& index : buffer_indices) {
        func_args.arg(genVariableName(buffer))
            .append("[")
            .append(index)
            .append("]");
      }
    }
  }

  void handle(const kir::GridReduction* grop) final {
    NVF_ERROR(grop->out()->isA<kir::TensorIndex>());

    const auto out = grop->out()->as<kir::TensorIndex>();
    const auto domain = out->view()->domain();
    NVF_ERROR(domain->hasGridReduction());

    const auto data_type = grop->out()->dtype();
    const auto op_type = grop->getReductionOpType();

    if (grop->isSerial()) {
      generateSerialGridReduction(grop);
      return;
    }

    NVF_ERROR(grop->reduction_buffer()->buffer()->isA<TensorView>());
    NVF_ERROR(grop->sync_buffer()->buffer()->isA<TensorView>());
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
    func_args.arg("&").append(genVariableName(work_buffer)).append("[0]");
    func_args.arg("&").append(genVariableName(sync_buffer)).append("[0]");
    func_args.arg(genCall("static_cast", ptrType(data_type), "shared_mem"));
    // read and write predicates
    NVF_ERROR(grop->predicate() != nullptr && grop->predicate()->hasValue());
    const auto read_pred = genInline(grop->predicate());
    func_args.arg(read_pred);
    if (grop->writePredicate() != nullptr) {
      NVF_ERROR(grop->writePredicate()->hasValue());
      func_args.arg(genInline(grop->writePredicate()));
    } else {
      func_args.arg(read_pred);
    }
    // Init val
    func_args.arg(genCall(data_type, genInline(grop->init())));
    func_args.arg(genInline(grop->entrance_index()));
    func_args.arg(genInline(grop->entrances()));
    func_args.arg(genComputeBlockDim());

    addProfileArguments(func_args, grop);

    indent() << "reduction::gridReduce<" << template_args << ">(\n";
    indent() << kTab << func_args << ");\n";
  }

  std::string genFusedReductionName(const TensorView* reduction_out) {
    return genVariableName(reduction_out) + "_reduction";
  }

  void generateSerialGridReduction(const kir::GridReduction* grop) {
    NVF_ERROR(grop->isSerial());

    const auto out = grop->out()->as<kir::TensorIndex>();

    const auto data_type = grop->out()->dtype();
    const auto op_type = grop->getReductionOpType();

    const auto par_domains =
        ir_utils::getParallelDomains(ir_utils::getTvOutput(grop));
    ArgumentBuilder block_flags;

    for (const ParallelType pt :
         {ParallelType::BIDx, ParallelType::BIDy, ParallelType::BIDz}) {
      const bool parallel_reduction =
          par_domains.find(pt) != par_domains.end() &&
          par_domains.at(pt)->isReduction();
      block_flags.arg(parallel_reduction);
    }

    std::string idx_in_segment = genCall(
        "index_utils::maskedOffset",
        block_flags,
        ArgumentBuilder().arg("blockIdx").arg("gridDim"));
    std::string segment_size = genCall(
        "index_utils::maskedSize",
        block_flags,
        ArgumentBuilder().arg("gridDim"));

    int64_t vectorize_size = ir_utils::getVectorizeSize(out->view());

    ArgumentBuilder template_args;
    template_args.arg("/*vec_size=*/").append(std::to_string(vectorize_size));

    ArgumentBuilder func_args(block_nest_level_ + 1, kTab);
    func_args.arg("&").append(gen(out));
    func_args.arg("&").append(gen(grop->in()));
    func_args.arg(gen(grop->init()));
    func_args.arg("&").append(gen(grop->serialReductionTensor()));
    func_args.arg(genReductionOp(op_type, out->dtype()));

    // Whether this is the first or last step
    func_args.arg(idx_in_segment).append(" == 0");
    func_args.arg(idx_in_segment)
        .append(" == ")
        .append(segment_size)
        .append(" - 1");
    // TODO: can we hoist the first and last step predicates? We might need to
    // attach them to grop in order to do that?

    // read and write predicates
    NVF_ERROR(grop->predicate() != nullptr && grop->predicate()->hasValue());
    const auto read_pred = genInline(grop->predicate());
    func_args.arg(read_pred);
    if (grop->writePredicate() != nullptr) {
      NVF_ERROR(grop->writePredicate()->hasValue());
      func_args.arg(genInline(grop->writePredicate()));
    } else {
      func_args.arg(read_pred);
    }

    indent() << "reduction::serialReductionStep<" << template_args << ">(\n";
    indent() << kTab << func_args << ");\n";
  }

  void generateGridAllreduce(const kir::GridReduction* grop) {
    NVF_ERROR(grop->isAllreduce());

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
        "&" + genVariableName(work_buffer) + "[0]"));
    // global_sync_buffer
    func_args.arg("&").append(genVariableName(sync_buffer)).append("[0]");
    // shared_buf
    func_args.arg(genCall(
        "PtrTuple",
        data_type,
        genCall("static_cast", ptrType(data_type), "shared_mem")));
    // read and write predicates
    NVF_ERROR(grop->predicate() != nullptr && grop->predicate()->hasValue());
    const auto read_pred = genInline(grop->predicate());
    auto write_pred = read_pred;
    if (grop->writePredicate() != nullptr) {
      NVF_ERROR(grop->writePredicate()->hasValue());
      write_pred = genInline(grop->writePredicate());
    }
    func_args.arg(read_pred).arg(write_pred);
    // init_val
    func_args.arg(genCall("LocalTuple", data_type, genInline(grop->init())));
    // block_dim
    func_args.arg(genComputeBlockDim());
    // reduction_op
    func_args.arg(genReductionOp(op_type, out->dtype()));

    addProfileArguments(func_args, grop);

    indent() << genCall(reduction_name + ".reduce", template_args, func_args)
             << ";\n";
  }

  void generateIterGroupedGridReduction(
      const int num_grouped_iterations,
      const kir::GroupedGridReduction* grop) {
    NVF_ERROR(grop->output(0)->isA<kir::TensorIndex>());
    const auto output = grop->output(0)->as<kir::TensorIndex>();
    const auto input = grop->input(0)->as<kir::TensorIndex>();
    const auto op_type = grop->getReductionOpType(0);
    const auto data_type = grop->output(0)->dtype();

    const std::string flags_str =
        generateGridReduceTemplateFlags2(grop, grop->threadPredicate());

    const bool persistent_sync =
        kernel_->summary().has_cooperative_grid_reduction;

    // Since block-level reduction is already done, those dimensions
    // with tidx/y/z being true do not participate in the grid
    // reduction.
    ArgumentBuilder template_args;
    template_args.arg(flags_str)
        .arg(persistent_sync)
        .arg(isAligned())
        .arg(num_grouped_iterations);

    const auto work_buffer =
        grop->reduction_buffers().at(0)->buffer()->as<TensorView>();
    const auto sync_buffer = grop->sync_buffer()->buffer()->as<TensorView>();

    ArgumentBuilder func_args(block_nest_level_ + 1, kTab);
    func_args.arg(genVariableNameConvertAlignedArray(output));
    func_args.arg(genVariableNameConvertAlignedArray(input));
    func_args.arg(genReductionOp(op_type, data_type));
    func_args.arg("&").append(genVariableName(work_buffer)).append("[0]");
    func_args.arg("&").append(genVariableName(sync_buffer)).append("[0]");
    func_args.arg(genCall("static_cast", ptrType(data_type), "shared_mem"));
    // read and write predicates
    NVF_ERROR(grop->predicate() != nullptr && grop->predicate()->hasValue());
    const auto read_pred = genInline(grop->predicate());
    func_args.arg(read_pred);
    if (grop->writePredicate() != nullptr) {
      NVF_ERROR(grop->writePredicate()->hasValue());
      func_args.arg(genInline(grop->writePredicate()));
    } else {
      func_args.arg(read_pred);
    }
    // Init val
    func_args.arg(genCall(data_type, genInline(grop->initVal(0))));
    // block_dim
    func_args.arg(genComputeBlockDim());

    addProfileArguments(func_args, grop);

    indent() << "reduction::iterGroupedGridReduce<" << template_args << ">(\n";
    indent() << kTab << func_args << ");\n";
  }

  void handle(const kir::GroupedGridReduction* grouped_grop) final {
    const auto out = ir_utils::getTvOutput(grouped_grop);
    const auto domain = out->domain();
    NVF_ERROR(domain->hasGridReduction());
    NVF_ERROR(grouped_grop->sync_buffer()->buffer()->isA<TensorView>());
    const auto sync_buffer =
        grouped_grop->sync_buffer()->buffer()->as<TensorView>();
    if (grouped_grop->isAllreduce()) {
      generateGroupedGridAllreduce(grouped_grop);
      return;
    }

    // iter domain grouped grid reduction, used for outer reduction
    // where iter domain is vectorized.
    if (grouped_grop->numHorizontallyGroupedExprs() == 1) {
      const auto num_grouped_iterations =
          getGroupedLoopIndexConcreteIntSets().size();
      NVF_ERROR(
          num_grouped_iterations > 1,
          "num_grouped_iterations should be greater than 1. Got: ",
          num_grouped_iterations);
      return generateIterGroupedGridReduction(
          (int)num_grouped_iterations, grouped_grop);
    }

    NVF_ERROR(
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
    for (const auto i : arange(grouped_grop->numHorizontallyGroupedExprs())) {
      NVF_ERROR(
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
      func_args.arg("&").append(genVariableName(work_buffer)).append("[0]");
    }

    // The rest of the arguments are common between the reductions
    func_args.arg("&").append(genVariableName(sync_buffer)).append("[0]");
    func_args.arg("shared_mem");
    // read and write predicates
    NVF_ERROR(
        grouped_grop->predicate() != nullptr &&
        grouped_grop->predicate()->hasValue());
    const auto read_pred = genInline(grouped_grop->predicate());
    func_args.arg(read_pred);
    if (grouped_grop->writePredicate() != nullptr) {
      NVF_ERROR(grouped_grop->writePredicate()->hasValue());
      func_args.arg(genInline(grouped_grop->writePredicate()));
    } else {
      func_args.arg(read_pred);
    }

    func_args.arg(genInline(grouped_grop->entrance_index()));
    func_args.arg(genInline(grouped_grop->entrances()));
    func_args.arg(genComputeBlockDim());

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
      NVF_THROW("Non-allreduce grouped grid welford is not yet supported");
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
      const auto iter_count = loop->stop()->evaluate();
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
  std::vector<std::unordered_map<const Val*, int64_t>>
  getLoopIndexReplacementMaps() {
    std::vector<std::unordered_map<const Val*, int64_t>> maps;

    if (grouped_loops_.empty()) {
      std::unordered_map<const Val*, int64_t> empty_map;
      return {empty_map};
    }

    // Vector of indices of grouped loops
    std::vector<Val*> loop_indices;
    std::transform(
        grouped_loops_.begin(),
        grouped_loops_.end(),
        std::back_inserter(loop_indices),
        [](const ForLoop* loop) { return loop->index(); });

    // All combinations of loop index integer values
    const auto index_val_sets = getGroupedLoopIndexConcreteIntSets();

    // Create maps from loop index Vals to integers
    for (const auto& index_values : index_val_sets) {
      NVF_ERROR(loop_indices.size() == index_values.size());
      std::unordered_map<const Val*, int64_t> index_val_map;
      for (const auto i : arange(loop_indices.size())) {
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
    NVF_ERROR(grouped_grop->isAllreduce());

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
    NVF_ERROR(
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
         arange(grouped_grop->numHorizontallyGroupedExprs())) {
      const auto data_type = grouped_grop->outputs().at(expr_index)->dtype();
      NVF_ERROR(grouped_grop->reduction_buffers()
                    .at(expr_index)
                    ->buffer()
                    ->isA<TensorView>());

      for (const auto& group_index : arange(index_replacement_maps.size())) {
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
            .append(genVariableName(work_buffer))
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
        NVF_ERROR(
            grouped_grop->predicate() != nullptr &&
            grouped_grop->predicate()->hasValue());
        const auto read_pred = genInline(grouped_grop->predicate());
        read_preds.arg(read_pred);
        if (grouped_grop->writePredicate() != nullptr) {
          NVF_ERROR(grouped_grop->writePredicate()->hasValue());
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
    func_args.arg(genComputeBlockDim());

    // global_sync_buffer
    const auto sync_buffer =
        grouped_grop->sync_buffer()->buffer()->as<TensorView>();
    func_args.arg("&").append(genVariableName(sync_buffer)).append("[0]");

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
    NVF_ERROR(grouped_gwop->isAllreduce());

    const auto index_replacement_maps = getLoopIndexReplacementMaps();
    const auto num_grouped_iterations = index_replacement_maps.size();

    // This is also checked at the lowering validaiton time, so it
    // isn't strictly necessary.
    NVF_ERROR(
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
         arange(grouped_gwop->numHorizontallyGroupedExprs())) {
      const auto& output = output_vals.at(expr_index);
      const auto& input = input_vals.at(expr_index);
      const auto& init = init_vals.at(expr_index);

      for (const auto& group_index : arange(index_replacement_maps.size())) {
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
        for (const auto i : arange(3)) {
          out_args[i].arg(gen(output.get(i)));
          in_args[i].arg(gen(input.get(i)));
          init_args[i].arg(gen(init.get(i)));
          const auto work_buffer = grouped_gwop->reduction_buffers()[i]
                                       .at(expr_index)
                                       ->buffer()
                                       ->as<TensorView>();
          work_bufs[i]
              .arg("&")
              .append(genVariableName(work_buffer))
              .append("[")
              .append(work_buffer_offset)
              .append("]");
        }

        // read and write predicates
        bool_types.arg("bool");
        // Same argument for all inputs. Different predicates would be
        // used when grouping is done across iterations
        NVF_ERROR(grouped_gwop->predicate() != nullptr);
        NVF_ERROR(
            grouped_gwop->predicate() != nullptr &&
            grouped_gwop->predicate()->hasValue());
        const auto read_pred = genInline(grouped_gwop->predicate());
        read_preds.arg(read_pred);
        if (grouped_gwop->writePredicate() != nullptr) {
          NVF_ERROR(grouped_gwop->writePredicate()->hasValue());
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
    // block_dim
    func_args.arg(genComputeBlockDim());
    // work buffer
    func_args.arg(genCall("VolatilePtrTuple", data_types, work_bufs[0]));
    func_args.arg(genCall("VolatilePtrTuple", data_types, work_bufs[1]));
    func_args.arg(genCall("VolatilePtrTuple", index_types, work_bufs[2]));
    // global_sync_buffer
    const auto sync_buffer =
        grouped_gwop->sync_buffer()->buffer()->as<TensorView>();
    func_args.arg("&").append(genVariableName(sync_buffer)).append("[0]");

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
    NVF_ERROR(grouped_gwop->isAllreduce());

    const auto num_grouped_iterations =
        getGroupedLoopIndexConcreteIntSets().size();

    // This is also checked at the lowering validaiton time, so it
    // isn't strictly necessary.
    NVF_ERROR(
        num_grouped_iterations * grouped_gwop->numHorizontallyGroupedExprs() <=
            kMaxNumGroupedReductions,
        "Too many grouped reductions: ",
        grouped_gwop->toString(),
        ". Up to ",
        kMaxNumGroupedReductions,
        " reductions are allowed.");

    NVF_ERROR(
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
    func_args.arg(genVariableNameConvertAlignedArray(output.get(0)));
    func_args.arg(genVariableNameConvertAlignedArray(output.get(1)));
    func_args.arg(genVariableNameConvertAlignedArray(output.get(2)));
    // inputs
    func_args.arg(genVariableNameConvertAlignedArray(input.get(0)));
    func_args.arg(genVariableNameConvertAlignedArray(input.get(1)));
    func_args.arg(genVariableNameConvertAlignedArray(input.get(2)))
        .append("[0]");
    // block_dim
    func_args.arg(genComputeBlockDim());

    // global buf
    for (const auto i : arange(3)) {
      const auto work_buffer = grouped_gwop->reduction_buffers()[i]
                                   .at(0)
                                   ->buffer()
                                   ->as<TensorView>();
      func_args.arg("&")
          .append(genVariableName(work_buffer))
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
    func_args.arg("&").append(genVariableName(sync_buffer)).append("[0]");

    addProfileArguments(func_args, grouped_gwop);

    ArgumentBuilder func_template_args;
    func_template_args.arg(isAligned());
    func_template_args.arg(num_grouped_iterations);
    func_template_args.arg(data_type);

    const auto& par_dim_map = kernel_->summary().parallel_dimension_map;
    NVF_ERROR(par_dim_map.get(ParallelType::TIDx)->isConstInt());
    NVF_ERROR(par_dim_map.get(ParallelType::TIDy)->isConstInt());
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
    NVF_ERROR(bop->out()->isA<kir::TensorIndex>());

    const ParallelTypeBitmap parallel_types =
        kernel_->summary().broadcast_parallel_types.at(bop);

    NVF_ERROR(
        parallel_types.hasBID(),
        "GridBroadcast needs to be used with a broadcast op that is "
        "parallelized with the BID parallel types");

    NVF_ERROR(grop->broadcast_buffer()->buffer()->isA<TensorView>());
    NVF_ERROR(grop->sync_buffer()->buffer()->isA<TensorView>());
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
    func_args.arg("&").append(genVariableName(work_buffer)).append("[0]");
    func_args.arg(genVariableName(sync_buffer));
    NVF_ERROR(grop->predicate() != nullptr && grop->predicate()->hasValue());
    func_args.arg(genInline(grop->predicate()));

    indent() << genCall("grid_broadcast::broadcast", template_args, func_args)
             << ";\n";
  }

  void handle(const kir::GridWelford* gwop) final {
    const auto wop = gwop->welford_op();
    NVF_ERROR(wop->outAvg()->isA<kir::TensorIndex>());

    const auto out = wop->out()->as<kir::TensorIndex>();
    const auto domain = out->view()->domain();
    NVF_ERROR(domain->hasGridReduction());

    const auto data_type = out->dtype();
    const auto index_type = wop->outN()->dtype();

    NVF_ERROR(gwop->var_buffer()->buffer()->isA<TensorView>());
    NVF_ERROR(gwop->sync_buffer()->buffer()->isA<TensorView>());

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
      NVF_ERROR(
          wop->inVar() != nullptr, "Welford var input nullptr not allowed");
      func_args.arg(genStaticCast(data_type, gen(wop->inVar())));
      func_args.arg(genStaticCast(index_type, gen(wop->inN())));
    }
    func_args.arg("&").append(genVariableName(avg_buffer)).append("[0]");
    func_args.arg("&").append(genVariableName(var_buffer)).append("[0]");
    func_args.arg("&").append(genVariableName(n_buffer)).append("[0]");
    func_args.arg(genVariableName(sync_buffer));
    func_args.arg(genReinterpretCast(genPtrType(data_type), "shared_mem_avg"));
    func_args.arg(genReinterpretCast(genPtrType(data_type), "shared_mem_var"));
    func_args.arg(genReinterpretCast(genPtrType(index_type), "shared_mem_n"));
    NVF_ERROR(gwop->predicate() != nullptr && gwop->predicate()->hasValue());
    auto read_pred = genInline(gwop->predicate());
    func_args.arg(read_pred);
    if (gwop->writePredicate() != nullptr) {
      NVF_ERROR(gwop->writePredicate()->hasValue());
      auto write_pred = genInline(gwop->writePredicate());
      func_args.arg(write_pred);
    } else {
      func_args.arg(read_pred);
    }
    // TODO : init value support or remove.
    func_args.arg(genStaticCast(data_type, 0));
    func_args.arg(genInline(gwop->entrance_index()));
    func_args.arg(genInline(gwop->entrances()));
    func_args.arg(genComputeBlockDim());

    indent() << genCall("welford::gridWelford", template_args, func_args)
             << ";\n";
  }

  void generateGridAllreduce(const kir::GridWelford* gwop) {
    const auto wop = gwop->welford_op();
    NVF_ERROR(wop->isAllreduce());

    const auto out = wop->out()->as<kir::TensorIndex>();

    const auto data_type = wop->outAvg()->dtype();
    const auto index_type = wop->outN()->dtype();
    NVF_ERROR(wop->outAvg()->dtype() == wop->outVar()->dtype());

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
        .append(genVariableName(gwop->avg_buffer()->buffer()->as<TensorView>()))
        .append("[0]");
    work_buffer_args.arg("&")
        .append(genVariableName(gwop->var_buffer()->buffer()->as<TensorView>()))
        .append("[0]");
    work_buffer_args.arg("&")
        .append(genVariableName(gwop->N_buffer()->buffer()->as<TensorView>()))
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
    func_args.arg("&").append(genVariableName(sync_buffer)).append("[0]");
    // shared_buf
    func_args.arg(genCall("PtrTuple", data_type_args, smem_buffer_args));
    // read and write predicates
    NVF_ERROR(gwop->predicate() != nullptr && gwop->predicate()->hasValue());
    const auto read_pred = genInline(gwop->predicate());
    auto write_pred = read_pred;
    if (gwop->writePredicate() != nullptr) {
      NVF_ERROR(gwop->writePredicate()->hasValue());
      write_pred = genInline(gwop->writePredicate());
    }
    func_args.arg(read_pred).arg(write_pred);
    // init_val
    func_args.arg(genCall("LocalTuple", data_type_args, init_args));
    // block_dim
    func_args.arg(genComputeBlockDim());
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
      auto pt_dim = kernel_->summary().parallel_dimension_map.get(pt);
      if (pt_dim == nullptr || pt_dim->isOneInt()) {
        continue;
      }
      // Initialize pt_dim if used to an iter dimension. It may change to a
      // reduction or predicated dimension later.
      states[pt] = ReductionParallelTypeState::Iter;
    }

    for (auto id : alloc_fused_reduction->out()->view()->getLoopDomain()) {
      auto pt = id->getParallelType();
      if (isParallelTypeThread(pt)) {
        auto state = id->isReduction() ? ReductionParallelTypeState::Reduce
                                       : ReductionParallelTypeState::Iter;
        states[pt] = state;
      }
    }

    for (const auto predicated_pt : alloc_fused_reduction->threadPredicate()) {
      auto& state = states[predicated_pt];
      NVF_ERROR(
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

  void handleTrivialLoop(const ForLoop* loop) {
    if (loop->vectorize()) {
      vectorize_scope_ = true;
    }
    kir::ConstIrVisitor::handle(loop);
    if (loop->vectorize()) {
      vectorize_scope_ = false;
    }
  }

  void genIterGroupedBlockReduction(
      const int num_grouped_iterations,
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

    NVF_ERROR(
        !tidx && tidy && !tidz,
        "blockIterGroupedYdimReduce only supports reduction along TIDy");

    const auto data_type = output->dtype();

    ArgumentBuilder template_args;
    template_args.arg(isAligned());
    template_args.arg(num_grouped_iterations);

    ArgumentBuilder func_args;
    func_args.arg(genVariableNameConvertAlignedArray(output->view()));
    func_args.arg(genVariableNameConvertAlignedArray(input->view()));
    func_args.arg(genReductionOp(reduction_op_type, output->dtype()));
    func_args.arg(genStaticCast(genPtrType(data_type), "shared_mem"));
    NVF_ERROR(read_pred != nullptr && read_pred->hasValue());
    func_args.arg(genInline(read_pred));
    // Pass the write predicate if available and different from the
    // default predicate. The blockReduce runtime function uses the
    // default predicate for both read and write when only the
    // default one is given.
    if (write_pred != nullptr) {
      NVF_ERROR(write_pred->hasValue());
      func_args.arg(genInline(write_pred));
    }
    func_args.arg(genCall(data_type, genInline(init)));
    func_args.arg(genComputeBlockDim());

    indent() << genCall("blockIterGroupedYdimReduce", template_args, func_args)
             << ";\n";
  }
  std::string genSmemOffset() {
    std::stringstream offset_ss;
    offset_ss << genVariableName(
        NamedScalar::getParallelIndex(warp_specialized_on_));
    offset_ss << " * "
              << reduction_scheduler_utils::getComputeBdimx(
                     warp_specialized_on_, lparams_.bdimx());
    if (kernel_->summary().num_grouped_iterations > 1) {
      offset_ss << " * " << kernel_->summary().num_grouped_iterations;
    }
    if (kernel_->summary().all_block_reductions_are_warp_reduction) {
      offset_ss << " / 32";
    }
    return offset_ss.str();
  }

  std::string genBarrierId(bool is_computation_warp_groups) {
    std::stringstream ss;
    if (is_computation_warp_groups && has_independent_compute_warp_groups_) {
      ss << next_barrier_id_ << " + "
         << genInline(NamedScalar::getParallelIndex(warp_specialized_on_));
      NVF_ERROR(
          warp_specialized_on_ == ParallelType::TIDy,
          "Independent compute warp groups only supported for TIDy.");
      int64_t n_compute_groups = lparams_.bdimy() - 1;
      next_barrier_id_ += n_compute_groups;
    } else {
      ss << "(uint32_t)" << next_barrier_id_;
      next_barrier_id_++;
    }
    return ss.str();
  }

  void genGroupedWarpReduction(
      const int num_grouped_iterations,
      kir::TensorIndex* output,
      kir::TensorIndex* input,
      const Val* init,
      BinaryOpType reduction_op_type,
      kir::Predicate* read_pred) {
    ArgumentBuilder func_args;
    func_args.arg(genVariableNameConvertAlignedArray(output));
    func_args.arg(genVariableNameConvertAlignedArray(input));
    func_args.arg(genReductionOp(reduction_op_type, output->dtype()));
    if (has_independent_compute_warp_groups_) {
      func_args.arg(
          genStaticCast(genPtrType(output->dtype()), "shared_mem") + " + " +
          genSmemOffset());
    } else {
      func_args.arg(genStaticCast(genPtrType(output->dtype()), "shared_mem"));
    }

    ArgumentBuilder template_args;
    template_args.arg(kernel_->getWarpPaddedParallelInfo().is_tidx_single_warp);
    template_args.arg(isAligned());
    template_args.arg(num_grouped_iterations);
    template_args.arg(reduction_scheduler_utils::getComputeBdimx(
        warp_specialized_on_, lparams_.bdimx()));
    if (has_independent_compute_warp_groups_) {
      func_args.arg(genBarrierId(true));
    }
    indent() << genCall(
                    "warp::iterGroupedStaticWarpAllReduce",
                    template_args,
                    func_args)
             << ";\n";
  }
  void handle(const GroupedReductionOp* grouped_rop) final {
    const auto num_grouped_iterations =
        getGroupedLoopIndexConcreteIntSets().size();

    const auto num_grouped_exprs = grouped_rop->numHorizontallyGroupedExprs();

    // special version where only iteration is grouped.
    // used for outer reduction with vectorized iteration domain.
    if (num_grouped_iterations > 1 && num_grouped_exprs == 1) {
      const auto output = grouped_rop->output(0)->as<kir::TensorIndex>();
      const auto input = grouped_rop->input(0)->as<kir::TensorIndex>();
      const auto op_type = grouped_rop->getReductionOpType(0);
      const auto domain = output->view()->domain();
      const bool has_block_reduce = domain->hasBlockReduction();
      const bool has_grid_reduce = domain->hasGridReduction();
      NVF_ERROR(
          !has_grid_reduce, "IterGroupedGridReduction not implemented yet");
      NVF_ERROR(
          has_block_reduce,
          "To use IterGroupedBlockReduction, must have block reduce!");
      if (auto reduction_ids =
              ir_utils::getMaybeWarpReductionDim(output, input)) {
        NVF_ERROR(
            lparams_.bdimx() % 128 == 0,
            "iterGroupedStaticWarpAllReduce() requires bdimx % 128 == 0.");
        NVF_ERROR(
            grouped_rop->isAllreduce(),
            "iterGroupedStaticWarpAllReduce should be used for allreduce.");
        NVF_ERROR(
            reduction_ids.value().first &&
                reduction_ids.value().first->getParallelType() ==
                    ParallelType::TIDx &&
                reduction_ids.value().second == nullptr,
            "Grouped warp reduction is only supported for TIDx reduction with "
            "no second dimension.");
        return genGroupedWarpReduction(
            (int)num_grouped_iterations,
            output,
            input,
            grouped_rop->initVal(0),
            op_type,
            grouped_rop->predicate());
      } else {
        return genIterGroupedBlockReduction(
            (int)num_grouped_iterations,
            output,
            input,
            grouped_rop->initVal(0),
            op_type,
            grouped_rop->predicate(),
            grouped_rop->writePredicate());
      }
    }

    for (const auto i : arange(num_grouped_exprs)) {
      NVF_ERROR(grouped_rop->output(i)->isA<kir::TensorIndex>());

      const auto output = grouped_rop->output(i)->as<kir::TensorIndex>();
      const auto input = grouped_rop->input(i)->as<kir::TensorIndex>();
      const auto domain = output->view()->domain();
      const auto op_type = grouped_rop->getReductionOpType(i);

      const bool has_block_reduce = domain->hasBlockReduction();
      const bool has_grid_reduce = domain->hasGridReduction();

      NVF_ERROR(
          !has_grid_reduce,
          "GroupedReductionOp does not support block parallelization. "
          "GroupedGridReduction must be used. ",
          grouped_rop->toString());

      if (!has_block_reduce) {
        genSerialReduction(output, input, op_type);
      } else if (
          auto reduction_ids =
              ir_utils::getMaybeWarpReductionDim(output, input)) {
        genWarpReduction(
            output,
            input,
            grouped_rop->initVal(i),
            op_type,
            grouped_rop->predicate(),
            reduction_ids.value(),
            grouped_rop->isAllreduce());
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
    NVF_THROW(
        "Should not reach here as grouped welford is only enabled for grid "
        "welford,",
        " which is handled by its own handler");
  }

  void handle(const ForLoop* loop) final {
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
    // Don't special unroll non-mma compute loop, it may contains
    // complex ops, e.g. reduction, unroll may lead to instruction cache miss
    // which hurts performance.
    auto cbls = loop->circularBufferLoopStage();
    bool special_unroll = kernel_->summary().has_mma_op ||
        (cbls != CircularBufferLoopStage::ComputeWarp);
    if (cbls != CircularBufferLoopStage::NotApplicable && special_unroll) {
      // NOTE: requireUnroll is sometimes called on a circular-buffered matmul
      // loops when static shapes are used. To avoid hinting that the compiler
      // should maximally unroll such loops leading to very long compiles, we
      // handle that case explicitly here and ignore loop->isUnrolled().
      //
      // Unroll "prefetch" many circular buffered loops regardless of buffer
      // stage (prologue, main, or epilogue)
      int64_t prefetch = kernel_->summary()
                             .circular_buffer_info
                             .getCircularBufferOptionsFor(loop->iter_domain())
                             .prefetch;
      indent() << "#pragma unroll " << prefetch << "\n";
    } else if (loop->isUnrolled()) {
      indent() << "#pragma unroll\n";
    } else {
      indent() << "#pragma unroll 1\n";
    }

    indent() << "for(nvfuser_index_t " << gen_index << " = " << gen_start
             << "; " << gen_index << " < " << gen_stop << "; "
             << step_code.str() << ") ";
    startBlock(true);
    kir::ConstIrVisitor::handle(loop);
    endBlock();
  }

  void handle(const kir::IfThenElse* ite) final {
    auto conditional = ite->predicate()->value();
    if (conditional->isConst()) {
      // If the conditional is a constant, then the IfThenElse is not required
      if (conditional->value()) {
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

    NVF_ERROR(alloc->buffer() != nullptr);
    alloc_set_.emplace(alloc->buffer());

    if (!alloc->buffer()->isA<TensorView>()) {
      // Pointer TensorMap allocation must be const as kernel parametr assigned
      // to it is const by definition, see genDeclaration(...) for details
      const bool add_const = isTmaType(buffer_dtype);
      indent() << (add_const ? "const " : "") << buffer_dtype << " "
               << gen(alloc->buffer()) << ";\n";
      return;
    }

    const auto tv = alloc->buffer()->as<TensorView>();

    const auto size = alloc->size();
    NVF_ERROR(size != nullptr);

    if (alloc->alias() != nullptr) {
      // Allocate alias another Allocate stmt
      const auto alias_tv = alloc->alias()->buffer()->as<TensorView>();
      if (alias_tv->getDataType() == tv->getDataType()) {
        indent() << "// Alias Allocation - " << alloc->memoryType() << "\n";
        indent() << "auto& " << genVariableName(tv) << " = "
                 << genVariableName(alias_tv) << ";\n";
      } else {
        indent() << "// Alias Allocation (changing dtype) - "
                 << alloc->memoryType() << "\n";
        auto va = kernel_->summary().vectorized_accesses;
        auto it = va.find(tv);
        int64_t alias_alignment = it == va.end() ? 1 : it->second;
        indent() << "auto " << genVariableName(tv)
                 << " = *reinterpret_cast<Array<" << buffer_dtype << ", "
                 << genInline(size) << ", " << alias_alignment << ">*>(&"
                 << genVariableName(alias_tv) << ");\n";
        if (alloc->memoryType() == MemoryType::Local) {
          aligned_array_of_regs_.insert(tv);
        }
      }
      // If the original allocation is aligned, its aliasing tv should also
      // be aligned due to auto type derivation. For example, in test
      // `CombinedSchedulerTest.LayerNormBackward/dtype_float_batch_216_hidden_65536`
      // we have: `Array<float, 4, 4> T32; auto& T29 = T32;`
      // Compiler treats `T29` as aligned array instead of regular array, when
      // passing `T29` to a runtime function, should use `T29.array` instead of
      // `T29`.
      if (aligned_array_of_regs_.count(alias_tv) > 0) {
        aligned_array_of_regs_.insert(tv);
      }
    } else {
      // Standard Memory Allocation
      switch (tv->getMemoryType()) {
        case MemoryType::Global:
          indent() << "// Allocate global tensor " << genVariableName(tv)
                   << "\n";
          break;
        case MemoryType::Shared:
          // Assume we have already aligned offsets to 16B
          NVF_CHECK(
              alloc->address() != nullptr,
              "Allocation did not receive an address: ",
              alloc->toString());
          // Shared Memory Pointer
          indent() << buffer_dtype << "* " << genVariableName(tv)
                   << " = reinterpret_cast<" << buffer_dtype << "*>"
                   << "(array + smem_offset + " << genInline(alloc->address())
                   << ");\n";
          break;
        case MemoryType::Local: {
          auto va = kernel_->summary().vectorized_accesses;
          indent() << "Array<" << buffer_dtype << ", " << genInline(size)
                   << ", " << (va.find(tv) != va.end() ? va.at(tv) : 1) << "> "
                   << genVariableName(tv) << ";\n";
          if (va.find(tv) != va.end()) {
            aligned_array_of_regs_.insert(tv);
          }
          break;
        }
        case MemoryType::Tensor: {
          // Generate code like:
          // TMemTensor T2(T5[0], 0, 0);
          indent() << "TMemTensor " << genVariableName(tv) << "("
                   << genInline(alloc->address()) << ", "
                   << genInline(alloc->laneOffset()) << ", "
                   << genInline(alloc->colOffset()) << ");\n";
          break;
        }
        default:
          NVF_THROW("Unexpected memory type");
      }
    }
  }

  // Reference:
  // https://docs.nvidia.com/cuda/inline-ptx-assembly
  void handle(const kir::Asm* asm_) final {
    auto get_type_or_index_type = [](Val* value) {
      if (auto ti = dynamic_cast<kir::TensorIndex*>(value)) {
        if (isPointerType(ti->index()->dtype())) {
          return ti->index()->dtype();
        }
      }
      return value->dtype();
    };
    // If asm_ has a utility name, we will wrap the PTX code in a utility
    // function with that name. Otherwise, we just generate the PTX code
    // directly in the kernel.
    const std::string utility_name = asm_->utility();
    std::string namespace_name = "";
    std::string utility_name_no_ns = utility_name;
    if (size_t pos = utility_name.rfind("::"); pos != std::string::npos) {
      namespace_name = utility_name.substr(0, pos);
      utility_name_no_ns = utility_name.substr(pos + 2);
    }
    bool as_utility = !utility_name.empty();
    bool utility_generated = false; // Is the same utility function already
                                    // generated when handling another asm_?
    if (as_utility) {
      if (!generated_utilities_.insert(asm_->signature()).second) {
        utility_generated = true;
      }
    }
    // The stream to write the PTX code to
    std::stringstream& utilities = utilities_[namespace_name];
    std::stringstream* asm_target = as_utility ? &utilities : &code_;
    // Indentation for the PTX code
    int utility_block_nest_level = 1;
    std::function<std::ostream&()> indent_utility = [&]() -> std::ostream& {
      for (auto _ : arange(utility_block_nest_level)) {
        (void)_;
        utilities << kTab;
      }
      return utilities;
    };
    std::function<std::ostream&()> indent_code = [this]() -> std::ostream& {
      return this->indent();
    };
    std::function<std::ostream&()> indent =
        (as_utility ? indent_utility : indent_code);
    // Increase or decrease the indentation level for the PTX code
    auto next_level = [&]() {
      as_utility ? utility_block_nest_level++ : block_nest_level_++;
    };
    auto prev_level = [&]() {
      as_utility ? utility_block_nest_level-- : block_nest_level_--;
    };
    // Generate the utility function signature like below:
    //   void myFunc(float& out1, float in1) {
    if (as_utility) {
      if (!utility_generated) {
        const auto& outputs = asm_->outputs();
        const auto& inputs = asm_->inputs();
        if (!asm_->options().immediate_inputs.empty()) {
          utilities << "template <";
          bool first = true;
          for (auto in_i : arange((int64_t)inputs.size())) {
            if (asm_->options().immediate_inputs.count(in_i)) {
              if (!first) {
                utilities << ", ";
              }
              utilities << inputs.at(in_i)->dtype() << " in" << in_i;
              first = false;
            }
          }
          utilities << ">\n";
        }
        utilities << "__device__ __inline__ void " << utility_name_no_ns << "(";
        for (auto out_i : arange(outputs.size())) {
          if (out_i > 0) {
            utilities << ", ";
          }
          utilities << outputs.at(out_i)->dtype() << "& out" << out_i;
        }
        if (!outputs.empty()) {
          utilities << ", ";
        }
        for (auto in_i : arange((int64_t)inputs.size())) {
          if (asm_->options().immediate_inputs.count(in_i)) {
            continue;
          }
          if (in_i > 0) {
            utilities << ", ";
          }
          utilities << get_type_or_index_type(inputs.at(in_i)) << " in" << in_i;
        }
        utilities << ") {\n";
      }
      this->indent() << utility_name;
    }
    // Generate the actual PTX code like below:
    //   asm("bla.bla.bla": "=f"(out1): "f"(in1));
    // We may either generate it in utilities or in code_, depending on
    // whether we are generating a utility function or not.
    if (!as_utility || !utility_generated) {
      indent() << "asm";
      if (asm_->volatile_()) {
        (*asm_target) << " volatile";
      }
      bool multiline = asm_->hasBooleanInput() ||
          (asm_->code().size() +
               (asm_->inputs().size() + asm_->outputs().size()) * 5 >
           80);
      if (!multiline) {
        // If any of the operand is an array type, force using multiline
        for (const auto& l :
             std::array<std::reference_wrapper<const std::vector<Val*>>, 2>{
                 asm_->inputs(), asm_->outputs()}) {
          for (const auto& v : l.get()) {
            if (std::holds_alternative<ArrayType>(v->dtype().type)) {
              multiline = true;
              break;
            }
          }
        }
      }
      (*asm_target) << "(";
      if (multiline) {
        (*asm_target) << "\n";
        next_level();
        indent();
      }

      if (asm_->hasBooleanInput()) {
        (*asm_target) << "\"{\\n\"\n";
        int64_t boolean_counter = 0;
        int64_t counter = 0;
        std::array<const std::vector<Val*>*, 2> outputs_and_inputs = {
            &asm_->outputs(), &asm_->inputs()};
        for (const auto* io : outputs_and_inputs) {
          for (auto val : *io) {
            // don't treat pointer to bool as bool
            auto val_dtype = get_type_or_index_type(val);
            if (val_dtype == DataType::Bool) {
              indent() << "\"  .reg .pred p" << boolean_counter << "; \\n\"\n";
              indent() << "\"  setp.ne.b32 p" << boolean_counter << ", %"
                       << counter << ", 0;\\n\"\n";
              boolean_counter++;
            }
            if (std::holds_alternative<ArrayType>(val_dtype.type)) {
              counter += (int64_t)std::get<ArrayType>(val_dtype.type).size;
            } else {
              counter++;
            }
          }
        }
        indent() << "\"  " << asm_->code();
      } else {
        (*asm_target) << "\"" << asm_->code();
      }

      auto parameters = asm_->parameters();
      if (!parameters.empty()) {
        (*asm_target) << " " << parameters;
      }
      (*asm_target) << R"(;\n")";

      if (asm_->hasBooleanInput()) {
        (*asm_target) << "\n";
        indent() << R"("}\n")";
      }

      auto next_section = [&]() {
        if (multiline) {
          (*asm_target) << "\n";
          indent();
        }
        (*asm_target) << ":";
      };

      auto print_constraints_and_registers =
          [&](const auto& constraints_and_registers, std::string prefix) {
            int64_t counter = 0;
            for (auto [constraint, register_] : constraints_and_registers) {
              auto next_line = [&]() {
                (*asm_target) << ",";
                if (multiline) {
                  (*asm_target) << "\n";
                  indent() << " ";
                } else {
                  (*asm_target) << " ";
                }
              };
              if (counter > 0) {
                next_line();
              }
              auto reg_dtype = get_type_or_index_type(register_);
              if (std::holds_alternative<ArrayType>(reg_dtype.type)) {
                for (auto i :
                     arange(std::get<ArrayType>(reg_dtype.type).size)) {
                  if (i > 0) {
                    next_line();
                  }
                  (*asm_target)
                      << "\"" << constraint
                      << "\"("
                      // If generating a utility function, we need to generate
                      // the parameter name like out1, in1, etc. If generating
                      // directly in the kernel, we need to generate the the
                      // actual argument value like T0[i * 4 + j].
                      << (as_utility ? prefix + std::to_string(counter)
                                     : gen(register_))
                      << "[" << i << "]"
                      << ")";
                }
              } else {
                (*asm_target) << "\"" << constraint << "\"(";
                if (reg_dtype == DataType::Bool) {
                  (*asm_target) << "(uint32_t)(";
                }
                (*asm_target)
                    // If generating a utility function, we need to generate the
                    // parameter name like out1, in1, etc. If generating the
                    // PTX code directly in the kernel, we need to generate the
                    // the actual argument value like T0[i * 4 + j].
                    << (as_utility ? prefix + std::to_string(counter)
                                   : gen(register_));
                if (reg_dtype == DataType::Bool) {
                  (*asm_target) << ")";
                }
                (*asm_target) << ")";
              }
              counter++;
            }
          };

      // outputs
      if (!asm_->outputs().empty() || !asm_->inputs().empty() ||
          asm_->memory()) {
        next_section();
      }
      print_constraints_and_registers(asm_->constraintsAndOutputs(), "out");

      if (!asm_->inputs().empty() || asm_->memory()) {
        next_section();
      }
      print_constraints_and_registers(asm_->constraintsAndInputs(), "in");

      if (asm_->memory()) {
        next_section();
        (*asm_target) << "\"memory\"";
      }
      if (multiline) {
        (*asm_target) << "\n";
        prev_level();
        indent();
      }
      (*asm_target) << ");\n";
    }
    // If the PTX code is wrapped as a utility function, we still need to
    // generate the function call to the utility function in the kernel code.
    // Something like:
    //   myFunc(T1[0], T0[0]);
    if (as_utility) {
      if (!asm_->options().immediate_inputs.empty()) {
        code_ << "<";
        bool first = true;
        for (auto&& [constraint, register_] : asm_->constraintsAndInputs()) {
          if (constraint == "n") {
            if (!first) {
              code_ << ", ";
            }
            code_ << gen(register_);
            first = false;
          }
        }
        code_ << ">";
      }
      code_ << "(";
      bool first = true;
      for (auto&& [_, register_] : asm_->constraintsAndOutputs()) {
        if (!first) {
          code_ << ", ";
        }
        code_ << gen(register_);
        first = false;
      }
      for (auto&& [constraint, register_] : asm_->constraintsAndInputs()) {
        if (constraint == "n") {
          continue;
        }
        if (!first) {
          code_ << ", ";
        }
        code_ << gen(register_);
        first = false;
      }
    }
    // The closing } for the utility function definition, and the ); for the
    // utility call.
    if (as_utility) {
      if (!utility_generated) {
        utilities << "}\n";
      }
      code_ << ");\n";
    }
  }

  void handle(const kir::BlockSync* sync) final {
    // Use a custom synchronization method if enabled
    if (getNvFuserEnv("USE_BLOCK_SYNC_ATOMIC")) {
      indent() << "block_sync::sync();\n";
    } else if (isAligned()) {
      indent() << "__syncthreads();\n";
    } else if (sync->isAsyncWarpSync()) {
      ArgumentBuilder template_args;
      template_args.arg(isAligned());
      ArgumentBuilder func_args;
      func_args.arg(genLoadBlockDim());
      indent() << genCall("block_sync::sync", template_args, func_args)
               << ";\n";
    } else if (sync->isComputeWarpSync()) {
      ArgumentBuilder template_args;
      template_args.arg(isAligned());
      ArgumentBuilder func_args;
      func_args.arg(genComputeBlockDim());
      indent() << genCall("block_sync::sync", template_args, func_args)
               << ";\n";
    } else {
      indent() << "__barrier_sync(0);\n";
    }
  }

  void handle(const kir::GridSync* sync) final {
    // Use a custom synchronization method if enabled
    bool bidx = sync->syncDims().get(ParallelType::BIDx);
    bool bidy = sync->syncDims().get(ParallelType::BIDy);
    bool bidz = sync->syncDims().get(ParallelType::BIDz);

    ArgumentBuilder sync_call_template_parms;
    sync_call_template_parms.arg(bidx)
        .arg(bidy)
        .arg(bidz)
        .arg(/*PERSISTENT=*/true)
        .arg(/*Aligned=*/
             has_warp_specialized_ ? false : isAligned());

    auto sync_idx = genCall(
        "index_utils::maskedOffset",
        ArgumentBuilder().arg(!bidx).arg(!bidy).arg(!bidz),
        ArgumentBuilder().arg("blockIdx").arg("gridDim"));

    auto sync_segment_size = genCall(
        "index_utils::maskedSize",
        ArgumentBuilder().arg(bidx).arg(bidy).arg(bidz),
        ArgumentBuilder().arg("gridDim"));

    ArgumentBuilder sync_call_args;
    sync_call_args.arg(genVariableName(sync->syncBuffer()))
        .append("[")
        .append(sync_idx)
        .append("]");
    sync_call_args.arg(sync_segment_size);
    sync_call_args.arg(genComputeBlockDim());
    if (has_independent_compute_warp_groups_) {
      sync_call_args.arg(genBarrierId(/*is_computation_warp_groups=*/false));
    }
    auto sync_call =
        genCall("grid_sync::sync", sync_call_template_parms, sync_call_args);

    indent() << sync_call << ";\n";
  }

  void handle(const kir::MBarrierInit* init) final {
    auto call = genCall(
        "mbarrier::init",
        ArgumentBuilder()
            .arg(genInline(init->mbarrier()))
            .arg(genInline(init->threadCount())));
    indent() << call << ";\n";
  }

  void handle(const kir::MBarrierInvalidate* inval) final {
    auto call = genCall(
        "mbarrier::inval", ArgumentBuilder().arg(genInline(inval->mbarrier())));
    indent() << call << ";\n";
  }

  void handle(const kir::MBarrierArrive* arrive) final {
    if (!print_inline_) {
      indent();
    }
    if (arrive->state() != nullptr) {
      code_ << gen(arrive->state()) << " = ";
    }
    auto call = genCall(
        "mbarrier::arrive",
        ArgumentBuilder().arg(genInline(arrive->mbarrier())));
    code_ << call;
    if (!print_inline_) {
      code_ << ";\n";
    }
  }

  void handle(const kir::MBarrierArriveExpectTx* arrive) final {
    if (!print_inline_) {
      indent();
    }
    if (arrive->state() != nullptr) {
      code_ << gen(arrive->state()) << " = ";
    }
    auto call = genCall(
        "mbarrier::arriveExpectTX",
        ArgumentBuilder()
            .arg(genInline(arrive->mbarrier()))
            .arg(genInline(arrive->txCount())));
    code_ << call;
    if (!print_inline_) {
      code_ << ";\n";
    }
  }

  void handle(const kir::MBarrierWait* wait) final {
    auto call = genCall(
        "mbarrier::wait",
        ArgumentBuilder()
            .arg(genInline(wait->mbarrier()))
            .arg(genInline(wait->state())));
    indent() << call << ";\n";
  }

  void handle(const kir::MBarrierWaitParity* wait) final {
    auto call = genCall(
        "mbarrier::waitParity",
        ArgumentBuilder()
            .arg(genInline(wait->mbarrier()))
            .arg(genInline(wait->parity())));
    indent() << call << ";\n";
  }

  void handle(const kir::BlockSerializeWait* sync) final {
    // Use a custom synchronization method if enabled
    bool bidx = sync->syncDims().get(ParallelType::BIDx);
    bool bidy = sync->syncDims().get(ParallelType::BIDy);
    bool bidz = sync->syncDims().get(ParallelType::BIDz);
    NVF_ERROR(
        isAligned(),
        "Serialization of blocks requires syncing in non-divergent threads");

    ArgumentBuilder sync_call_template_parms;
    sync_call_template_parms.arg(bidx).arg(bidy).arg(bidz);

    auto sync_idx = genCall(
        "index_utils::maskedOffset",
        ArgumentBuilder().arg(!bidx).arg(!bidy).arg(!bidz),
        ArgumentBuilder().arg("blockIdx").arg("gridDim"));

    ArgumentBuilder sync_call_args;
    sync_call_args.arg("&")
        .append(genVariableName(sync->syncBuffer()))
        .append("[")
        .append(sync_idx)
        .append("]");

    auto sync_call = genCall(
        "grid_sync::blockSerializeWait",
        sync_call_template_parms,
        sync_call_args);

    indent() << sync_call << ";\n";
  }

  void handle(const kir::BlockSerializeRelease* sync) final {
    // Use a custom synchronization method if enabled
    bool bidx = sync->syncDims().get(ParallelType::BIDx);
    bool bidy = sync->syncDims().get(ParallelType::BIDy);
    bool bidz = sync->syncDims().get(ParallelType::BIDz);
    NVF_ERROR(
        isAligned(),
        "Serialization of blocks requires syncing in non-divergent threads");

    ArgumentBuilder sync_call_template_parms;
    sync_call_template_parms.arg(bidx).arg(bidy).arg(bidz);

    auto sync_idx = genCall(
        "index_utils::maskedOffset",
        ArgumentBuilder().arg(!bidx).arg(!bidy).arg(!bidz),
        ArgumentBuilder().arg("blockIdx").arg("gridDim"));

    ArgumentBuilder sync_call_args;
    sync_call_args.arg("&")
        .append(genVariableName(sync->syncBuffer()))
        .append("[")
        .append(sync_idx)
        .append("]");

    auto sync_call = genCall(
        "grid_sync::blockSerializeRelease",
        sync_call_template_parms,
        sync_call_args);

    indent() << sync_call << ";\n";
  }

  void handle(const kir::InitMagicZero*) final {
    indent() << "NVFUSER_DEFINE_MAGIC_ZERO;\n";
  }

  void handle(const kir::UpdateMagicZero*) final {
    indent() << "NVFUSER_UPDATE_MAGIC_ZERO;\n";
  }

  void handle(const kir::Continue* cont) final {
    indent() << "continue;\n";
  }

  void handle(const kir::Return* ret) final {
    indent() << "return;\n";
  }

 private:
  // Our generated string has two parts: a utilities section that contains PTX
  // wrappers and other definitions derived from kernel IR, and a kernel section
  // that contains the kernel code itself.
  //
  //   // Utility section
  //   namespace feature1 {
  //   void myFunc(float& out1, float in1) {
  //     asm("bla.bla.bla": "=f"(out1): "f"(in1));
  //   }
  //   }
  //   // Kernel section
  //   __global__ void kernel_name(Tensor T0, Tensor T1, ...) {
  //     ...
  //     myFunc(T1[0], T0[0]);
  //     ...
  //   }

  // string for utility section. namespace -> utility code
  // using std::map instead of std::unordered_map for determinism
  std::map<std::string, std::stringstream> utilities_;
  // string kernel section
  std::stringstream code_;

  const kir::Kernel* kernel_;
  int block_nest_level_ = 0;
  int block_reduce_name_ = 0;
  bool print_inline_ = false;

  // Mark when we are inside of a vectorized for-loop
  bool vectorize_scope_ = false;
  //! Keep track of Allocate node for Val. Used to determine if Val
  //! should be inlined.
  std::unordered_set<const Val*> alloc_set_;
  //! Keep track of grouped loops
  std::deque<const ForLoop*> grouped_loops_;
  //! Used to replace symbolic indices with concrete values
  std::unordered_map<const Val*, int64_t> index_replacement_map_;
  //! Keep track of thread alignment property
  std::vector<bool> aligned_scope_exprs_;
  //! Keep track of the Val* and its generated variable name
  std::unordered_map<const Val*, std::string> val_to_name_;
  //! basically kernel_->parameters(), but as a set so it's faster to lookup
  std::unordered_set<const Val*> kernel_params_;
  //! Utility names already generated
  std::unordered_set<std::string> generated_utilities_;
  //! iterGroupedStaticWarpAllReduce requires static threads per CTA
  LaunchParams lparams_;
  //! Whether the kernel has warp specialization
  bool has_warp_specialized_ = false;
  //! Whether the kernel has independent compute warp groups
  bool has_independent_compute_warp_groups_ = false;
  //! Warp specialized on parallel type
  ParallelType warp_specialized_on_ = ParallelType::Serial;
  //! Track barrier ids used in the kernel
  //! 0 is system reserved, start from 1
  int64_t next_barrier_id_ = 1;
};

} // namespace

std::string generateCudaKernel(
    const kir::Kernel* kernel,
    const std::string& kernel_name,
    const LaunchParams& lparams) {
  FUSER_PERF_SCOPE("generateCudaKernel");
  return CudaKernelGenerator::generateKernelDefinition(
      kernel, kernel_name, lparams);
}

} // namespace codegen
} // namespace nvfuser
