// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#include <device_lower/lower2device.h>
#include <device_lower/utils.h>
#include <expr_evaluator.h>
#include <expr_simplifier.h>
#include <ir/builder.h>
#include <ir/cloner.h>
#include <ir/iostream.h>
#include <kernel.h>
#include <kernel_ir.h>
#include <type.h>

#include <ranges.h>
#include <iostream>

namespace nvfuser {
namespace kir {

namespace {

inline const char* boolLiteral(bool value) {
  return value ? "true" : "false";
}

} // namespace

Predicate::Predicate(
    IrBuilderPasskey passkey,
    PredicateType ptype,
    const Expr* expr,
    Val* thread_pred)
    : Val(passkey, ValType::Predicate, DataType::Bool),
      ptype_(ptype),
      expr_(expr),
      thread_pred_(thread_pred) {
  NVF_ERROR(passkey.ir_container_ != nullptr);
  NVF_ERROR(
      passkey.ir_container_->isA<kir::Kernel>(),
      "IR type only valid for Kernel container.");
  NVF_ERROR(ptype != PredicateType::Unswitch && ptype != PredicateType::Manual);
}

Predicate::Predicate(IrBuilderPasskey passkey, ForLoop* unrolled_loop)
    : Val(passkey, ValType::Predicate, DataType::Bool),
      ptype_(PredicateType::Unswitch),
      unrolled_loop_(unrolled_loop) {
  NVF_ERROR(passkey.ir_container_ != nullptr);
  NVF_ERROR(
      passkey.ir_container_->isA<kir::Kernel>(),
      "IR type only valid for Kernel container.");
  NVF_ERROR(unrolled_loop != nullptr);
}

Predicate::Predicate(IrBuilderPasskey passkey, Val* value)
    : Val(passkey, ValType::Predicate, DataType::Bool),
      ptype_(PredicateType::Manual),
      value_(value) {
  NVF_ERROR(passkey.ir_container_ != nullptr);
  NVF_ERROR(
      passkey.ir_container_->isA<kir::Kernel>(),
      "IR type only valid for Kernel container.");
  NVF_ERROR(value != nullptr);
}

std::string Predicate::toString(int indent_size) const {
  std::stringstream ss;
  ss << predicate_type2string(predicate_type());
  if (hasValue()) {
    ss << " " << value()->toInlineString();
  }
  return ss.str();
}

std::string Predicate::toInlineString(int indent_size) const {
  return toString(indent_size);
}

TensorIndex::TensorIndex(
    IrBuilderPasskey passkey,
    const TensorView* view,
    Val* index)
    : Val(passkey, ValType::TensorIndex, view->getDataType().value()),
      view_(view),
      index_(index) {
  NVF_ERROR(passkey.ir_container_ != nullptr);
  NVF_ERROR(
      passkey.ir_container_->isA<kir::Kernel>(),
      "IR type only valid for Kernel container.");
  NVF_ERROR(
      isPointerType(index->dtype()) || index->dtype() == DataType::Index ||
          isStructType(index->dtype()),
      "Cannot index with a value other than an int.");
}

std::string TensorIndex::toString(int indent_size) const {
  std::stringstream ss;
  ss << ir_utils::varName(this);
  switch (view()->getMemoryType()) {
    case MemoryType::Global:
      ss << "_g";
      break;
    case MemoryType::Shared:
      ss << "_s";
      break;
    case MemoryType::Local:
      ss << "_l";
      break;
    default:
      NVF_ERROR(false, "Unknown tensor memory type.");
  }
  ss << "[";
  ss << index()->toInlineString(indent_size);
  ss << "]";
  ss << " view( " << ir_utils::varName(view()) << " )";
  return ss.str();
}

std::string TensorIndex::toInlineString(int indent_size) const {
  return toString(indent_size);
}

Allocate::Allocate(
    IrBuilderPasskey passkey,
    Val* buffer,
    MemoryType memory_type,
    std::vector<Val*> shape,
    bool zero_init,
    Allocate* alias)
    : Expr(passkey) {
  NVF_ERROR(passkey.ir_container_ != nullptr);
  NVF_ERROR(
      passkey.ir_container_->isA<kir::Kernel>(),
      "IR type only valid for Kernel container.");
  if (!shape.empty()) {
    NVF_ERROR(
        (shape.size() == 1 && shape[0]->isOneInt()) ||
        buffer->isA<TensorView>());
  } else {
    NVF_ERROR(buffer->isA<TensorView>());
    NVF_ERROR(buffer->as<TensorView>()->getMemoryType() == memory_type);
    const auto domain = buffer->as<TensorView>()->domain();
    for (auto axis : domain->noReductions()) {
      shape.push_back(axis->extent());
    }
  }

  Val* size = nullptr;
  for (auto s : shape) {
    if (size == nullptr) {
      size = s;
    } else {
      size = IrBuilder::mulExpr(size, s);
    }
  }

  if (size == nullptr) {
    size = FusionGuard::getCurFusion()->oneVal();
  }

  if (alias != nullptr) {
    NVF_ERROR(alias != this, "Invalid alias");
    NVF_ERROR(alias->memoryType() == memory_type, "Invalid alias");
  }

  size = simplifyExpr(size);

  addInput(size);
  addAttribute(buffer);
  addDataAttribute(memory_type);
  addDataAttribute(zero_init);
  addAttribute(alias);
  // Always initialize shared memory address to nullptr
  addAttribute(nullptr);

  for (auto s : shape) {
    addAttribute(s);
  }
}

Allocate::Allocate(
    IrBuilderPasskey passkey,
    Val* buffer,
    MemoryType memory_type,
    Val* size,
    bool zero_init)
    : Allocate(
          passkey,
          buffer,
          memory_type,
          size == nullptr ? std::vector<Val*>{} : std::vector<Val*>{size},
          zero_init) {}

std::string Allocate::toString(int indent_size) const {
  std::stringstream ss;
  indent(ss, indent_size) << buffer()->toString();
  ss << " = ALLOCATE("
     << "buffer=" << buffer()->toString() << ", "
     << "mem_type=" << memoryType() << ", "
     << "size=" << size()->toInlineString();
  ss << ", "
     << "zero_init=" << boolLiteral(zeroInit()) << ")\n";
  if (alias() != nullptr) {
    indent(ss, indent_size) << kTab << ".alias=";
    ss << alias()->buffer()->toString() << "\n";
  }
  return ss.str();
}

std::string Allocate::toInlineString(int indent_size) const {
  NVF_CHECK(false, "Tensor op can not be printed inline");
}

NVFUSER_DEFINE_CLONE_AND_CREATE(Allocate)

BlockSync::BlockSync(IrBuilderPasskey passkey, bool war_sync) : Expr(passkey) {
  NVF_ERROR(passkey.ir_container_ != nullptr);
  NVF_ERROR(
      passkey.ir_container_->isA<kir::Kernel>(),
      "IR type only valid for Kernel container.");
  addDataAttribute(war_sync);
}

std::string BlockSync::toString(int indent_size) const {
  std::stringstream ss;
  indent(ss, indent_size) << "BLOCKSYNC(war_hazard="
                          << boolLiteral(isWarHazardSync()) << ")\n";
  return ss.str();
}

std::string BlockSync::toInlineString(int indent_size) const {
  NVF_CHECK(false, "Tensor op can not be printed inline");
}

NVFUSER_DEFINE_CLONE_AND_CREATE(BlockSync)

GridSync::GridSync(
    IrBuilderPasskey passkey,
    ParallelTypeBitmap sync_dims,
    Val* sync_buffer)
    : Expr(passkey) {
  NVF_ERROR(passkey.ir_container_ != nullptr);
  addDataAttribute(sync_dims);
  addAttribute(sync_buffer);
}

std::string GridSync::toString(int indent_size) const {
  std::stringstream ss;
  indent(ss, indent_size) << "GRIDSYNC(" << syncDims().toString() << ", "
                          << syncBuffer()->toString() << ")\n";
  return ss.str();
}

std::string GridSync::toInlineString(int indent_size) const {
  NVF_CHECK(false, "Tensor op can not be printed inline");
}

NVFUSER_DEFINE_CLONE_AND_CREATE(GridSync)

CpAsyncWait::CpAsyncWait(IrBuilderPasskey passkey, int64_t keep_stages)
    : Expr(passkey) {
  NVF_ERROR(passkey.ir_container_ != nullptr);
  NVF_ERROR(
      passkey.ir_container_->isA<kir::Kernel>(),
      "IR type only valid for Kernel container.");
  addDataAttribute(keep_stages);
}

std::string CpAsyncWait::toString(int indent_size) const {
  std::stringstream ss;
  indent(ss, indent_size) << "CPASYNC_WAIT(" << keepStages() << ")\n";
  return ss.str();
}

std::string CpAsyncWait::toInlineString(int indent_size) const {
  NVF_CHECK(false, "Tensor op can not be printed inline");
}

NVFUSER_DEFINE_CLONE_AND_CREATE(CpAsyncWait)

CpAsyncCommit::CpAsyncCommit(IrBuilderPasskey passkey) : Expr(passkey) {
  NVF_ERROR(passkey.ir_container_ != nullptr);
  NVF_ERROR(
      passkey.ir_container_->isA<kir::Kernel>(),
      "IR type only valid for Kernel container.");
}

std::string CpAsyncCommit::toString(int indent_size) const {
  std::stringstream ss;
  indent(ss, indent_size) << "CPASYNC_WAIT()\n";
  return ss.str();
}

std::string CpAsyncCommit::toInlineString(int indent_size) const {
  NVF_CHECK(false, "Tensor op can not be printed inline");
}

NVFUSER_DEFINE_CLONE_AND_CREATE(CpAsyncCommit)

InitMagicZero::InitMagicZero(IrBuilderPasskey passkey) : Expr(passkey) {
  NVF_ERROR(passkey.ir_container_ != nullptr);
  NVF_ERROR(
      passkey.ir_container_->isA<kir::Kernel>(),
      "IR type only valid for Kernel container.");
}

std::string InitMagicZero::toString(int indent_size) const {
  std::stringstream ss;
  indent(ss, indent_size) << "NVFUSER_DEFINE_MAGIC_ZERO;\n";
  return ss.str();
}

std::string InitMagicZero::toInlineString(int indent_size) const {
  NVF_CHECK(false, "Tensor op can not be printed inline");
}

NVFUSER_DEFINE_CLONE_AND_CREATE(InitMagicZero)

UpdateMagicZero::UpdateMagicZero(IrBuilderPasskey passkey) : Expr(passkey) {
  NVF_ERROR(passkey.ir_container_ != nullptr);
  NVF_ERROR(
      passkey.ir_container_->isA<kir::Kernel>(),
      "IR type only valid for Kernel container.");
}

std::string UpdateMagicZero::toString(int indent_size) const {
  std::stringstream ss;
  indent(ss, indent_size) << "NVFUSER_UPDATE_MAGIC_ZERO;\n";
  return ss.str();
}

std::string UpdateMagicZero::toInlineString(int indent_size) const {
  NVF_CHECK(false, "Tensor op can not be printed inline");
}

NVFUSER_DEFINE_CLONE_AND_CREATE(UpdateMagicZero)

std::string Scope::toString(int indent_size) const {
  std::stringstream ss;
  for (auto expr : exprs()) {
    ss << expr->toString(indent_size);
  }
  return ss.str();
}

std::vector<Expr*>::iterator Scope::insert(
    std::vector<Expr*>::const_iterator pos,
    Expr* expr) {
  return exprs_.insert(pos, expr);
}

std::vector<Expr*>::iterator Scope::insert_before(Expr* ref, Expr* expr) {
  const auto it = std::find(exprs_.begin(), exprs_.end(), ref);
  NVF_ERROR(
      it != exprs_.end(),
      "Tried to insert ",
      expr,
      " before the reference: ",
      ref,
      " @ ",
      (size_t)ref,
      " however the reference was not found in this scope.");
  return insert(it, expr);
}

std::vector<Expr*>::iterator Scope::insert_after(Expr* ref, Expr* expr) {
  const auto it = std::find(exprs_.begin(), exprs_.end(), ref);
  NVF_ERROR(
      it != exprs_.end(),
      "Tried to insert ",
      expr,
      " after the reference: ",
      ref,
      " however the reference was not found in this scope.");
  return insert(it + 1, expr);
}

std::vector<Expr*>::iterator Scope::insert(size_t pos, Expr* expr) {
  const auto it = exprs_.begin() + (std::ptrdiff_t)pos;
  return insert(it, expr);
}

void Scope::erase(std::vector<Expr*>::const_iterator pos) {
  // Remove the scope of the expr if this is the scope
  C10_UNUSED auto expr = *pos;
  exprs_.erase(pos);
}

void Scope::erase(Expr* ref) {
  const auto it = std::find(exprs_.begin(), exprs_.end(), ref);
  if (it != exprs_.end()) {
    erase(it);
  }
}

void Scope::erase(size_t pos) {
  erase(exprs_.begin() + (std::ptrdiff_t)pos);
}

bool Scope::contains(Expr* expr) const {
  const auto it = std::find(exprs_.begin(), exprs_.end(), expr);
  return it != exprs_.end();
}

void Scope::clear() {
  exprs_.clear();
}

ForLoop::ForLoop(
    IrBuilderPasskey passkey,
    IterDomain* iter_domain,
    Val* index,
    Val* start,
    Val* stop,
    Val* step,
    bool vectorize,
    Val* vectorize_shift,
    bool unroll_required,
    DoubleBufferLoopStage double_buffer_loop_stage)
    : Expr(passkey) {
  NVF_ERROR(passkey.ir_container_ != nullptr);
  NVF_ERROR(
      passkey.ir_container_->isA<kir::Kernel>(),
      "IR type only valid for Kernel container.");
  NVF_ERROR(isIntegralType(index->dtype()));
  addInput(index);
  addInput(iter_domain);
  if (start == nullptr && iter_domain->isThread()) {
    start = NamedScalar::getParallelIndex(iter_domain->getParallelType());
  }
  if (step == nullptr) {
    if (iter_domain->isThread()) {
      step = NamedScalar::getParallelDim(iter_domain->getParallelType());
    } else {
      step = FusionGuard::getCurFusion()->oneVal();
    }
  }
  NVF_ERROR(
      index->dtype() == DataType::Index, "Loop index must be an index type.");
  NVF_ERROR(
      start == nullptr || start->dtype() == DataType::Index,
      "Loop start must be an index type.");
  NVF_ERROR(
      step->dtype() == DataType::Index, "Loop step must be an index type.");
  NVF_ERROR(
      stop == nullptr || stop->dtype() == DataType::Index,
      "Loop stop must be an index type.");
  addAttribute(start);
  addAttribute(stop);
  addAttribute(step);
  addDataAttribute(vectorize);
  addAttribute(vectorize_shift);
  addDataAttribute(unroll_required);
  addDataAttribute(double_buffer_loop_stage);
  // Storing IR nodes as Attribute is not safe with IrCloner, but fortunately
  // kernel IR does not need this feature.
  addDataAttribute(Scope(this));
}

ForLoop::ForLoop(
    IrBuilderPasskey passkey,
    IterDomain* iter_domain,
    Val* index,
    DoubleBufferLoopStage double_buffer_loop_stage)
    : ForLoop(
          passkey,
          iter_domain,
          index,
          nullptr,
          nullptr,
          nullptr,
          !iter_domain->isBroadcast() &&
              isParallelTypeVectorize(iter_domain->getParallelType()),
          nullptr,
          false,
          double_buffer_loop_stage) {}

ForLoop::ForLoop(IrBuilderPasskey passkey, IterDomain* iter_domain)
    : ForLoop(
          passkey,
          iter_domain,
          GpuLower::current()->caMap()->getIndexVariable(iter_domain),
          DoubleBufferLoopStage::NotApplicable) {}

ForLoop::ForLoop(IrBuilderPasskey passkey, const ForLoop* other)
    : ForLoop(
          passkey,
          other->iter_domain(),
          other->index(),
          other->start(),
          other->stop(),
          other->step(),
          other->vectorize(),
          other->vectorize_shift(),
          other->isUnrollRequired(),
          other->doubleBufferLoopStage()) {}

std::string ForLoop::toString(int indent_size) const {
  std::stringstream ss;
  indent(ss, indent_size) << "FOR " << index()->toString() << " in "
                          << iter_domain()->toString() << ":\n"
                          << body().toString(indent_size + 1);
  return ss.str();
}

std::string ForLoop::toInlineString(int indent_size) const {
  NVF_CHECK(false, "Tensor op can not be printed inline");
}

bool ForLoop::isUnrollable() const {
  // Start and stop must be constant, must not be a broadcast
  // dimension, cannot be bound to a parallel dimension, must not be
  // vectorized.
  return start()->isConstScalar() && stop()->isConstScalar() &&
      !iter_domain()->isThread() && !iter_domain()->isBroadcast() &&
      !vectorize();
}

bool ForLoop::isUnrolled() const {
  if (isUnrollRequired() && !isUnrollable()) {
    TORCH_WARN(
        "Unroll required but not possible. Register allocation disabled. Loop index: ",
        index()->toString());
    return false;
  }

  // Size-one loop will not be materialized as a loop, so return false
  if (start()->isZeroInt() && stop()->isOneInt()) {
    return false;
  }

  // Unroll if required.
  if (isUnrollRequired()) {
    return true;
  }

  // Don't unroll if not possible
  if (!isUnrollable()) {
    return false;
  }

  // Unrolling is technically possible but avoided
  if (iter_domain()->getParallelType() == ParallelType::Unswitch) {
    // Use ParallelType::Unroll if unrolling is desired. Note that
    // unswitched size-one loops are not unrolled as they are not
    // materialized as actual for-loops.
    return false;
  }

  return true;
}

Val* ForLoop::start() const {
  if (attributeVal(0) != nullptr) {
    return attributeVal(0);
  } else {
    // clang-tidy complains without this
    NVF_ERROR(iter_domain() != nullptr);
    return iter_domain()->start();
  }
}

Val* ForLoop::stop() const {
  if (attributeVal(1) != nullptr) {
    return attributeVal(1);
  } else {
    // clang-tidy complains without this
    NVF_ERROR(iter_domain() != nullptr);
    return iter_domain()->extent();
  }
}

Val* ForLoop::step() const {
  NVF_ERROR(attributeVal(2) != nullptr);
  return attributeVal(2);
}

Val* ForLoop::simplifiedStop() const {
  if (simplified_stop_ == nullptr) {
    simplified_stop_ =
        GpuLower::current()->commonScalarMap().hoistScalar(stop(), {});
  }
  return simplified_stop_;
}

bool ForLoop::isTrivial() const {
  // These loops are not materialized
  if (vectorize() || iter_domain()->isBroadcast() ||
      iter_domain()->isStride() || iter_domain()->isMma() ||
      iter_domain()->isBulk()) {
    return true;
  }

  if (index()->isConstScalar() || index()->definition() != nullptr) {
    return true;
  }

  // By default, a parallelized loop would look like:
  //
  //   for (int x = threadIdx.x; x < stop; x += blockDim.x) {
  //     do_some_comp(x);
  //   }
  //
  // When stop is guaranteed to be smaller or equal to the number of
  // threads, the for-loop is not necessary. In the above case, we
  // would just generate the loop body without the for clause but
  // references to the loop index replaced by the loop start value.
  //
  // When the loop end is the same as the IterDomain extent, the
  // assumption can be safely made. This is more conservative than
  // necessary since the loop stop value just needs to be <= the
  // IterDomain extent. However, at this point, this conservative
  // analysis seems sufficient.
  if (stop() == iter_domain()->extent() && iter_domain()->isThread()) {
    return true;
  }

  // Extent-1 loop: for (int i = 0; i < 1; ++i) {
  if (start()->isZeroInt() && simplifiedStop()->isOneInt() &&
      step()->isOneInt()) {
    return true;
  }

  // Another extent-1 loop: for (int i = N - 1; i < N; ++i) {
  if (start()->definition() != nullptr &&
      start()->definition()->isA<BinaryOp>() &&
      start()->definition()->as<BinaryOp>()->getBinaryOpType() ==
          BinaryOpType::Sub &&
      start()->definition()->as<BinaryOp>()->lhs() == stop() &&
      start()->definition()->as<BinaryOp>()->rhs()->isOneInt()) {
    return true;
  }

  return false;
}

namespace {

//! A utility class to check if an expression of a particular type exists
class ExprFinder : kir::ConstIrVisitor {
 public:
  //! True if expr or any of its nested expressions is a type included in
  //! expr_types
  static bool exists(
      const Expr* expr,
      const std::unordered_set<std::type_index>& expr_types) {
    ExprFinder finder(expr_types);
    finder.handle(std::vector<const Expr*>{expr});
    return finder.is_found_;
  }

 private:
  ExprFinder(const std::unordered_set<std::type_index>& expr_types)
      : expr_types_(expr_types) {}

  using kir::ConstIrVisitor::handle;

  void dispatch(const Expr* expr) final {
    if (expr_types_.find(typeid(*expr)) != expr_types_.end()) {
      is_found_ = true;
      return;
    }
    kir::ConstIrVisitor::dispatch(expr);
  }

 private:
  const std::unordered_set<std::type_index>& expr_types_;
  bool is_found_ = false;
};

} // namespace

bool ForLoop::isGroup() const {
  //! True if loop is grouped. The IterDomain of the loop must have
  //! ParallelType::Group, but it isn't sufficient as the loop may be
  //! for an initialization expression, for which the loop shold not
  //! be grouped. Make sure a GroupedGridReduction is found.
  if (iter_domain()->getParallelType() != ParallelType::Group) {
    return false;
  }

  return ExprFinder::exists(
      this,
      {typeid(kir::GroupedGridReduction), typeid(kir::GroupedGridWelford)});
}

NVFUSER_DEFINE_CLONE_AND_CREATE(ForLoop)

IfThenElse::IfThenElse(IrBuilderPasskey passkey, Predicate* cond)
    : Expr(passkey) {
  setPredicate(cond);
  addInput(cond);
  // Storing IR nodes as Attribute is not safe with IrCloner, but fortunately
  // kernel IR does not need this feature.
  addDataAttribute(Scope(this));
  addDataAttribute(Scope(this));
}

std::string IfThenElse::toString(int indent_size) const {
  std::stringstream ss;
  indent(ss, indent_size) << "IF " << predicate()->toString() << ":\n"
                          << thenBody().toString(indent_size + 1);
  if (hasElse()) {
    indent(ss, indent_size) << "ELSE:\n"
                            << elseBody().toString(indent_size + 1);
  }
  return ss.str();
}

std::string IfThenElse::toInlineString(int indent_size) const {
  NVF_CHECK(false, "Tensor op can not be printed inline");
}

NVFUSER_DEFINE_CLONE_AND_CREATE(IfThenElse)

GridReduction::GridReduction(
    IrBuilderPasskey passkey,
    BinaryOpType reduction_op_type,
    Val* init,
    Val* out,
    Val* in,
    Allocate* reduction_buffer,
    Allocate* sync_buffer,
    Val* entrance_index,
    Val* entrances,
    bool is_allreduce)
    : ReductionOp(passkey, reduction_op_type, init, out, in, is_allreduce) {
  NVF_ERROR(passkey.ir_container_ != nullptr);
  NVF_ERROR(
      passkey.ir_container_->isA<kir::Kernel>(),
      "IR type only valid for Kernel container.");
  NVF_ERROR(
      attributes().size() == num_reduction_op_attr,
      "The num_reduction_op_attr does not match the number of attributes ReductionOp has."
      "If you changed ReductionOp, please change num_reduction_op_attr accordingly.");
  addAttribute(reduction_buffer);
  addAttribute(sync_buffer);
  addAttribute(entrance_index);
  addAttribute(entrances);
  addDataAttribute(ParallelTypeBitmap{});
}

std::string GridReduction::toString(int indent_size) const {
  std::stringstream ss;
  indent(ss, indent_size) << out()->toString() << " = reduction( "
                          << in()->toString()
                          << ", op = " << getReductionOpType()
                          << ", initial value = " << init()->toString()
                          << ",\n";
  ++indent_size;
  indent(ss, indent_size) << "reduction buffer = "
                          << reduction_buffer()->buffer()->toString() << ",\n";
  indent(ss, indent_size) << "sync buffer = "
                          << sync_buffer()->buffer()->toString() << ",\n";
  indent(ss, indent_size) << "read predicate = ";
  if (predicate() != nullptr) {
    ss << predicate()->toString();
  } else {
    ss << "nullptr";
  }
  ss << ",\n";
  indent(ss, indent_size) << "write predicate = ";
  if (writePredicate() != nullptr) {
    ss << writePredicate()->toString();
  } else {
    ss << "nullptr";
  }
  ss << ",\n";
  indent(ss, indent_size) << "thread predicate = "
                          << threadPredicate().toString() << ",\n";
  indent(ss, indent_size) << "allreduce = "
                          << (isAllreduce() ? "true" : "false") << " )\n";
  return ss.str();
}

std::string GridReduction::toInlineString(int indent_size) const {
  NVF_CHECK(false, "Tensor op can not be printed inline");
}

NVFUSER_DEFINE_CLONE_AND_CREATE(GridReduction)

GroupedGridReduction::GroupedGridReduction(
    IrBuilderPasskey passkey,
    std::vector<BinaryOpType> reduction_op_types,
    std::vector<Val*> init_vals,
    std::vector<Val*> outputs,
    std::vector<Val*> inputs,
    std::vector<Allocate*> reduction_buffers,
    Allocate* sync_buffer,
    Val* entrance_index,
    Val* entrances,
    Val* buffer_stride,
    bool is_allreduce)
    : GroupedReductionOp(
          passkey,
          std::move(reduction_op_types),
          std::move(init_vals),
          std::move(outputs),
          std::move(inputs),
          is_allreduce) {
  NVF_ERROR(passkey.ir_container_ != nullptr);
  NVF_ERROR(
      passkey.ir_container_->isA<kir::Kernel>(),
      "IR type only valid for Kernel container.");
  NVF_ERROR(
      attributes().size() == numGroupedReductionOpAttr(),
      "The numGroupedReductionOpAttr() does not match the number of attributes GroupedReductionOp has."
      "If you changed GroupedReductionOp, please change numGroupedReductionOpAttr() accordingly.");
  addAttribute(sync_buffer);
  addAttribute(entrance_index);
  addAttribute(entrances);
  addAttribute(buffer_stride);
  addDataAttribute(ParallelTypeBitmap{});
  for (auto buffer : reduction_buffers) {
    addAttribute(buffer);
  }
}

std::string GroupedGridReduction::toString(int indent_size) const {
  std::stringstream ss;
  indent(ss, indent_size) << "GroupedGridReduction(\n";
  ++indent_size;
  for (const auto i : irange(numHorizontallyGroupedExprs())) {
    indent(ss, indent_size)
        << output(i)->toString() << " = reduction( " << input(i)->toString()
        << ", op = " << getReductionOpType(i)
        << ", initial value = " << initVal(i)->toString()
        << ", reduction buffer = "
        << reduction_buffers().at(i)->buffer()->toString() << " )\n";
  }
  indent(ss, indent_size) << "sync buffer = "
                          << sync_buffer()->buffer()->toString() << ",\n";
  indent(ss, indent_size) << "read predicate = ";
  if (predicate() != nullptr) {
    ss << predicate()->toString();
  } else {
    ss << "nullptr";
  }
  ss << ",\n";
  indent(ss, indent_size) << "write predicate = ";
  if (writePredicate() != nullptr) {
    ss << writePredicate()->toString();
  } else {
    ss << "nullptr";
  }
  ss << ",\n";
  indent(ss, indent_size) << "thread predicate = "
                          << threadPredicate().toString() << ",\n";
  indent(ss, indent_size) << "allreduce = "
                          << (isAllreduce() ? "true" : "false") << " )\n";
  --indent_size;
  return ss.str();
}

std::string GroupedGridReduction::toInlineString(int indent_size) const {
  NVF_CHECK(false, "Tensor op can not be printed inline");
}

NVFUSER_DEFINE_CLONE_AND_CREATE(GroupedGridReduction)

GridBroadcast::GridBroadcast(
    IrBuilderPasskey passkey,
    BroadcastOp* broadcast_op,
    Allocate* broadcast_buffer,
    Allocate* sync_buffer)
    : Expr(passkey) {
  NVF_ERROR(passkey.ir_container_ != nullptr);
  NVF_ERROR(
      passkey.ir_container_->isA<kir::Kernel>(),
      "IR type only valid for Kernel container.");
  addAttribute(broadcast_op);
  addAttribute(broadcast_buffer);
  addAttribute(sync_buffer);
}

std::string GridBroadcast::toString(int indent_size) const {
  std::stringstream ss;
  const auto* broadcast_op = this->broadcast_op();
  indent(ss, indent_size) << broadcast_op->out()->toString() << " = "
                          << "GRID_BROADCAST(in="
                          << broadcast_op->in()->toString() << ")\n";
  indent(ss, indent_size) << kTab << ".broadcast_buffer="
                          << broadcast_buffer()->buffer()->toString() << "\n";
  indent(ss, indent_size) << kTab << ".sync_buffer="
                          << sync_buffer()->buffer()->toString() << "\n";
  return ss.str();
}

std::string GridBroadcast::toInlineString(int indent_size) const {
  NVF_CHECK(false, "Tensor op can not be printed inline");
}

NVFUSER_DEFINE_CLONE_AND_CREATE(GridBroadcast)

GridWelford::GridWelford(
    IrBuilderPasskey passkey,
    WelfordOp* welford_op,
    Allocate* var_buffer,
    Allocate* avg_buffer,
    Allocate* n_buffer,
    Allocate* sync_buffer,
    Val* entrance_index,
    Val* entrances)
    : Expr(passkey) {
  NVF_ERROR(passkey.ir_container_ != nullptr);
  NVF_ERROR(
      passkey.ir_container_->isA<kir::Kernel>(),
      "IR type only valid for Kernel container.");
  addAttribute(welford_op);
  addAttribute(var_buffer);
  addAttribute(avg_buffer);
  addAttribute(n_buffer);
  addAttribute(sync_buffer);
  addAttribute(entrance_index);
  addAttribute(entrances);
  addDataAttribute(ParallelTypeBitmap{});
}

std::string GridWelford::toString(int indent_size) const {
  std::stringstream ss;
  const auto* welford_op = this->welford_op();
  indent(ss, indent_size) << welford_op->outAvg()->toString() << " (Avg),\n";
  indent(ss, indent_size) << welford_op->outVar()->toString() << " (Var),\n";
  indent(ss, indent_size) << welford_op->outN()->toString() << " (Count)\n";
  indent(ss, indent_size) << " = Welford (\n";
  ++indent_size;
  indent(ss, indent_size) << welford_op->inAvg()->toString() << " (Avg),\n";
  indent(ss, indent_size) << welford_op->inVar()->toString() << " (Var),\n";
  indent(ss, indent_size) << welford_op->inN()->toString() << " (Count)\n";
  indent(ss, indent_size) << "initial value =\n";
  ++indent_size;
  indent(ss, indent_size) << welford_op->initAvg()->toString() << " (Avg),\n";
  indent(ss, indent_size) << welford_op->initVar()->toString() << " (Var),\n";
  indent(ss, indent_size) << welford_op->initN()->toString() << " (Count),\n";
  --indent_size;
  indent(ss, indent_size) << "reduction buffer =\n";
  ++indent_size;
  indent(ss, indent_size) << avg_buffer()->buffer()->toString() << " (Avg),\n";
  indent(ss, indent_size) << var_buffer()->buffer()->toString() << " (Var),\n";
  indent(ss, indent_size) << N_buffer()->buffer()->toString() << " (Count),\n";
  --indent_size;
  indent(ss, indent_size) << "sync buffer = "
                          << sync_buffer()->buffer()->toString() << ",\n";
  indent(ss, indent_size) << "read predicate = ";
  if (welford_op->predicate() != nullptr) {
    ss << welford_op->predicate();
  } else {
    ss << "nullptr";
  }
  ss << ",\n";
  indent(ss, indent_size) << "write predicate = ";
  if (welford_op->writePredicate() != nullptr) {
    ss << welford_op->writePredicate()->toString();
  } else {
    ss << "nullptr";
  }
  ss << ",\n";
  indent(ss, indent_size) << "grid read predicate = ";
  if (predicate() != nullptr) {
    ss << predicate()->toString();
  } else {
    ss << "nullptr";
  }
  ss << ",\n";
  indent(ss, indent_size) << "grid write predicate = ";
  if (writePredicate() != nullptr) {
    ss << writePredicate()->toString();
  } else {
    ss << "nullptr";
  }
  ss << ",\n";
  indent(ss, indent_size) << "thread predicate = "
                          << threadPredicate().toString() << ",\n";
  indent(ss, indent_size) << "allreduce = "
                          << (welford_op->isAllreduce() ? "true" : "false")
                          << " )\n";
  return ss.str();
}

std::string GridWelford::toInlineString(int indent_size) const {
  NVF_CHECK(false, "Tensor op can not be printed inline");
}

NVFUSER_DEFINE_CLONE_AND_CREATE(GridWelford)

GroupedGridWelford::GroupedGridWelford(
    IrBuilderPasskey passkey,
    std::vector<WelfordTriplet> output_vals,
    std::vector<WelfordTriplet> input_vals,
    std::vector<WelfordTriplet> init_vals,
    std::array<std::vector<Allocate*>, 3> reduction_buffers,
    Allocate* sync_buffer,
    Val* entrance_index,
    Val* entrances,
    Val* buffer_stride,
    bool is_allreduce,
    bool use_outer_opt)
    : GroupedWelfordOp(
          passkey,
          std::move(output_vals),
          std::move(input_vals),
          std::move(init_vals),
          is_allreduce) {
  NVF_ERROR(passkey.ir_container_ != nullptr);
  NVF_ERROR(
      passkey.ir_container_->isA<kir::Kernel>(),
      "IR type only valid for Kernel container.");
  NVF_ERROR(
      attributes().size() == numGroupedWelfordOpAttr(),
      "The numGroupedWelfordOpAttr() does not match the number of attributes GroupedWelfordOp has."
      "If you changed GroupedReductionOp, please change numGroupedWelfordOpAttr() accordingly.");
  addAttribute(sync_buffer);
  addAttribute(entrance_index);
  addAttribute(entrances);
  addAttribute(buffer_stride);
  addDataAttribute(ParallelTypeBitmap{});
  NVF_ERROR(reduction_buffers[0].size() == reduction_buffers[1].size());
  NVF_ERROR(reduction_buffers[0].size() == reduction_buffers[2].size());
  for (auto i : irange(reduction_buffers[0].size())) {
    addAttribute(reduction_buffers[0].at(i));
    addAttribute(reduction_buffers[1].at(i));
    addAttribute(reduction_buffers[2].at(i));
  }

  addDataAttribute(use_outer_opt);
}

int GroupedGridWelford::getSmemBufferSize(int bdimx, int bdimy, int bdimz)
    const {
  auto out_tv = ir_utils::getTvOutput(this);
  auto kernel = dynamic_cast<kir::Kernel*>(container());
  NVF_ERROR(kernel != nullptr);

  // By default, the required size is the same as the normal Welford reduction
  if (!useOuterOpt()) {
    return bdimx * bdimy * bdimz *
        (int)dataTypeSize(out_tv->getDataType().value()) * 2 +
        bdimx * bdimy * bdimz *
        (int)dataTypeSize(DataType::Index, kernel->indexType());
  }

  // In the outer-reduction version, the size is blockDim.x * NumberOfWarps *
  // GroupCount

  int group_count = 1;
  for (auto axis : out_tv->getLeafDomain()) {
    auto pt = axis->getParallelType();
    if (pt == ParallelType::Group) {
      auto extent_int = axis->extent()->getInt();
      NVF_ERROR(extent_int.has_value());
      group_count *= (int)extent_int.value();
    }
  }

  NVF_ERROR(group_count > 1);

  int num_warps = bdimx * bdimy / 32;
  NVF_ERROR((bdimx * bdimy) % 32 == 0);

  int buf_size_for_avg_var = bdimx * num_warps * group_count *
      (int)dataTypeSize(out_tv->getDataType().value());
  int buf_size_for_N =
      num_warps * (int)dataTypeSize(DataType::Index, kernel->indexType());

  return buf_size_for_avg_var * 2 + buf_size_for_N;
}

std::string GroupedGridWelford::toString(int indent_size) const {
  std::stringstream ss;
  indent(ss, indent_size) << "GroupedGridWelford(\n";
  ++indent_size;
  for (const auto i : irange(numHorizontallyGroupedExprs())) {
    indent(ss, indent_size) << outAvg(i)->toString() << " (Avg),\n";
    indent(ss, indent_size) << outVar(i)->toString() << " (Var),\n";
    indent(ss, indent_size) << outN(i)->toString() << " (Count)\n";
    indent(ss, indent_size) << " = Welford (\n";
    ++indent_size;
    indent(ss, indent_size) << inAvg(i)->toString() << " (Avg),\n";
    indent(ss, indent_size) << inVar(i)->toString() << " (Var),\n";
    indent(ss, indent_size) << inN(i)->toString() << " (Count)\n";
    indent(ss, indent_size) << "initial value =\n";
    ++indent_size;
    indent(ss, indent_size) << initAvg(i)->toString() << " (Avg),\n";
    indent(ss, indent_size) << initVar(i)->toString() << " (Var),\n";
    indent(ss, indent_size) << initN(i)->toString() << " (Count),\n";
    --indent_size;
    indent(ss, indent_size) << "reduction buffer =\n";
    ++indent_size;
    indent(ss, indent_size)
        << reduction_buffers()[0].at(i)->buffer()->toString() << " (Avg),\n";
    indent(ss, indent_size)
        << reduction_buffers()[1].at(i)->buffer()->toString() << " (Var),\n";
    indent(ss, indent_size)
        << reduction_buffers()[2].at(i)->buffer()->toString() << " (Count) )\n";
    indent_size -= 2;
  }
  indent(ss, indent_size) << "sync buffer = "
                          << sync_buffer()->buffer()->toString() << ",\n";
  indent(ss, indent_size) << "read predicate = ";
  if (predicate() != nullptr) {
    ss << predicate()->toString();
  } else {
    ss << "nullptr";
  }
  ss << ",\n";
  indent(ss, indent_size) << "write predicate = ";
  if (writePredicate() != nullptr) {
    ss << writePredicate()->toString();
  } else {
    ss << "nullptr";
  }
  ss << ",\n";
  indent(ss, indent_size) << "thread predicate = "
                          << threadPredicate().toString() << ",\n";
  indent(ss, indent_size) << "allreduce = "
                          << (isAllreduce() ? "true" : "false") << " )\n";
  return ss.str();
}

std::string GroupedGridWelford::toInlineString(int indent_size) const {
  NVF_CHECK(false, "Tensor op can not be printed inline");
}

NVFUSER_DEFINE_CLONE_AND_CREATE(GroupedGridWelford)

VectorizedWelfordOp::VectorizedWelfordOp(
    IrBuilderPasskey passkey,
    const WelfordTriplet& output,
    const WelfordTriplet& input,
    const WelfordTriplet& init,
    Val* count,
    Val* reciprocal_of_count,
    Val* hoisted_predicate)
    : WelfordOp(passkey, output, input, init, false) {
  NVF_ERROR(passkey.ir_container_ != nullptr);
  NVF_ERROR(
      passkey.ir_container_->isA<kir::Kernel>(),
      "IR type only valid for Kernel container.");
  addAttribute(count);
  addAttribute(reciprocal_of_count);
  addAttribute(hoisted_predicate);
}

NVFUSER_DEFINE_CLONE_AND_CREATE(VectorizedWelfordOp)

AllocateFusedReduction::AllocateFusedReduction(
    IrBuilderPasskey passkey,
    Expr* grid_expr)
    : Expr(passkey) {
  NVF_ERROR(passkey.ir_container_ != nullptr);
  NVF_ERROR(
      passkey.ir_container_->isA<kir::Kernel>(),
      "IR type only valid for Kernel container.");
  addAttribute(grid_expr);
}

std::string AllocateFusedReduction::toString(int indent_size) const {
  std::stringstream ss;
  indent(ss, indent_size) << "AllocateFusedReduction(reduction buffer="
                          << out()->toString() << ")\n";
  return ss.str();
}

std::string AllocateFusedReduction::toInlineString(int indent_size) const {
  NVF_CHECK(false, "Tensor op can not be printed inline");
}

TensorIndex* AllocateFusedReduction::out() const {
  NVF_ERROR(gridExpr() != nullptr);
  if (gridExpr()->isA<GridReduction>() ||
      gridExpr()->isA<GroupedGridReduction>()) {
    return gridExpr()->outputs().at(0)->as<kir::TensorIndex>();
  } else if (auto grid_welford = dynamic_cast<GridWelford*>(gridExpr())) {
    return grid_welford->welford_op()->out()->as<kir::TensorIndex>();
  } else if (
      auto grouped_grid_welford =
          dynamic_cast<GroupedGridWelford*>(gridExpr())) {
    return grouped_grid_welford->out(0)->as<kir::TensorIndex>();
  } else {
    NVF_ERROR(false, "Invalid grid expression: ", gridExpr()->toString());
  }
}

const ParallelTypeBitmap& AllocateFusedReduction::threadPredicate() const {
  NVF_ERROR(gridExpr() != nullptr);
  if (auto grid_reduction = dynamic_cast<GridReduction*>(gridExpr())) {
    return grid_reduction->threadPredicate();
  } else if (auto grid_welford = dynamic_cast<GridWelford*>(gridExpr())) {
    return grid_welford->threadPredicate();
  } else if (
      auto grouped_grid_reduction =
          dynamic_cast<GroupedGridReduction*>(gridExpr())) {
    return grouped_grid_reduction->threadPredicate();
  } else if (
      auto grouped_grid_welford =
          dynamic_cast<GroupedGridWelford*>(gridExpr())) {
    return grouped_grid_welford->threadPredicate();
  } else {
    NVF_ERROR(false, "Invalid grid expression: ", gridExpr()->toString());
  }
}

NVFUSER_DEFINE_CLONE_AND_CREATE(AllocateFusedReduction)

GetRNGSeedAndOffsetFromHost::GetRNGSeedAndOffsetFromHost(
    IrBuilderPasskey passkey,
    Val* seed_ptr,
    Val* seed_val,
    Val* first_offset_ptr,
    Val* first_offset_val,
    int64_t offsets)
    : Expr(passkey) {
  addOutput(seed_ptr);
  addOutput(seed_val);
  addOutput(first_offset_ptr);
  addOutput(first_offset_val);
  addDataAttribute(offsets);
}

std::string GetRNGSeedAndOffsetFromHost::toString(int indent_size) const {
  std::stringstream ss;
  indent(ss, indent_size) << "(" << output(0)->toString() << ", "
                          << output(1)->toString() << ", "
                          << output(2)->toString() << ", "
                          << output(3)->toString() << ") = " << getOpString()
                          << "()\n";
  return ss.str();
}

std::string GetRNGSeedAndOffsetFromHost::toInlineString(int indent_size) const {
  return std::string(getOpString()) + "()";
}

NVFUSER_DEFINE_CLONE_AND_CREATE(GetRNGSeedAndOffsetFromHost)

EncodeTensorMapTiled::EncodeTensorMapTiled(
    IrBuilderPasskey passkey,
    Val* output,
    DataType data_type,
    Val* global_address,
    Val* global_dim,
    Val* global_strides,
    Val* box_dim,
    Val* element_strides,
    tma::TensorMapInterleave interleave,
    tma::TensorMapSwizzle swizzle,
    tma::TensorMapL2Promotion l2_promotion,
    tma::TensorMapFloatOOBFill oob_fill)
    : Expr(passkey) {
  auto out_dtype = output->dtype();
  NVF_CHECK(std::holds_alternative<OpaqueType>(out_dtype.type));
  addOutput(output);

  NVF_CHECK(
      global_address->dtype() ==
      PointerType{std::make_shared<DataType>(data_type)});
  addInput(global_address);

  NVF_CHECK(std::holds_alternative<ArrayType>(global_dim->dtype().type));
  size_t tensor_rank = std::get<ArrayType>(global_dim->dtype().type).size;
  ArrayType expect_global_dim_type{
      std::make_shared<DataType>(DataType::Index), tensor_rank};
  NVF_CHECK(global_dim->dtype() == expect_global_dim_type);
  addInput(global_dim);

  ArrayType expect_global_strides_type{
      std::make_shared<DataType>(DataType::Index), tensor_rank - 1};
  NVF_CHECK(global_strides->dtype() == expect_global_strides_type);
  addInput(global_strides);

  ArrayType expect_box_dim_type{
      std::make_shared<DataType>(DataType::Index), tensor_rank};
  NVF_CHECK(box_dim->dtype() == expect_box_dim_type);
  addInput(box_dim);

  ArrayType expect_element_strides_type{
      std::make_shared<DataType>(DataType::Index), tensor_rank};
  NVF_CHECK(element_strides->dtype() == expect_element_strides_type);
  addInput(element_strides);

  addDataAttribute(data_type);
  addDataAttribute((int64_t)tensor_rank);
  addDataAttribute(interleave);
  addDataAttribute(swizzle);
  addDataAttribute(l2_promotion);
  addDataAttribute(oob_fill);
}

std::string EncodeTensorMapTiled::toString(int indent_size) const {
  std::stringstream ss;
  indent(ss, indent_size) << output(0)->toString() << " = " << getOpString()
                          << "(dtype=" << dataType()
                          << ", global_address=" << globalAddress()->toString()
                          << ", global_dim=" << globalDim()->toString()
                          << ", global_strides=" << globalStrides()
                          << ", box_dim=" << boxDim()->toString()
                          << ", element_strides="
                          << elementStrides()->toString()
                          << ", interleave=" << interleave()
                          << ", swizzle=" << swizzle()
                          << ", l2_promotion=" << l2Promotion()
                          << ", oob_fill=" << oobFill() << ")\n";
  return ss.str();
}

std::string EncodeTensorMapTiled::toInlineString(int indent_size) const {
  std::stringstream ss;
  ss << getOpString() << "(dtype=" << dataType()
     << ", global_address=" << globalAddress()->toInlineString()
     << ", global_dim=" << globalDim()->toInlineString()
     << ", global_strides=" << globalStrides()->toInlineString()
     << ", box_dim=" << boxDim()->toInlineString()
     << ", element_strides=" << elementStrides()->toInlineString()
     << ", interleave=" << interleave() << ", swizzle=" << swizzle()
     << ", l2_promotion=" << l2Promotion() << ", oob_fill=" << oobFill() << ")";
  return ss.str();
}

NVFUSER_DEFINE_CLONE_AND_CREATE(EncodeTensorMapTiled)

} // namespace kir
} // namespace nvfuser
