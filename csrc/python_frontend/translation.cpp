// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
//
#include <dispatch.h>
#include <python_frontend/translation.h>
#include <vector>

namespace nvfuser::python_frontend {

namespace {

template <typename ResultType, typename... ArgTypes>
std::function<ResultType(ArgTypes...)> getFunction(const BinaryOp* bop) {
  auto get_std_function = [](ResultType (*fn)(ArgTypes...)) {
    return static_cast<ResultType (*)(ArgTypes...)>(fn);
  };

  switch (bop->getBinaryOpType()) {
    case BinaryOpType::Add:
      return get_std_function(add);
      break;
    case BinaryOpType::Atan2:
      return get_std_function(atan2);
      break;
    case BinaryOpType::Div:
      return get_std_function(div);
      break;
    case BinaryOpType::Fmod:
      return get_std_function(fmod);
      break;
    case BinaryOpType::Mul:
      return get_std_function(mul);
      break;
    case BinaryOpType::Nextafter:
      return get_std_function(nextafter);
      break;
    case BinaryOpType::Pow:
      return get_std_function(pow);
      break;
    case BinaryOpType::Remainder:
      return get_std_function(remainder);
      break;
    case BinaryOpType::Sub:
      return get_std_function(sub);
      break;
    case BinaryOpType::Mod:
      return get_std_function(mod);
      break;
    case BinaryOpType::Eq:
      return get_std_function(eq);
      break;
    case BinaryOpType::NE:
      return get_std_function(ne);
      break;
    case BinaryOpType::GT:
      return get_std_function(gt);
      break;
    case BinaryOpType::GE:
      return get_std_function(ge);
      break;
    case BinaryOpType::LT:
      return get_std_function(lt);
      break;
    case BinaryOpType::LE:
      return get_std_function(le);
      break;
    case BinaryOpType::BitwiseAnd:
      return get_std_function(bitwise_and);
      break;
    case BinaryOpType::BitwiseOr:
      return get_std_function(bitwise_or);
      break;
    case BinaryOpType::BitwiseXor:
      return get_std_function(bitwise_xor);
      break;
    case BinaryOpType::Lshift:
      return get_std_function(bitwise_left_shift);
      break;
    case BinaryOpType::Rshift:
      return get_std_function(bitwise_right_shift);
      break;
    case BinaryOpType::Gcd:
      return get_std_function(gcd);
      break;
    default:
      NVF_CHECK(
          false,
          "Unexpected operator type: ",
          bop->getBinaryOpType(),
          " in ",
          bop->toString());
  }
}

static std::string getString(const BinaryOp* bop) {
  switch (bop->getBinaryOpType()) {
    case BinaryOpType::Add:
      return "add";
      break;
    case BinaryOpType::Atan2:
      return "atan2";
      break;
    case BinaryOpType::Div:
      return "div";
      break;
    case BinaryOpType::Fmod:
      return "fmod";
      break;
    case BinaryOpType::Mul:
      return "mul";
      break;
    case BinaryOpType::Nextafter:
      return "nextafter";
      break;
    case BinaryOpType::Pow:
      return "pow";
      break;
    case BinaryOpType::Remainder:
      return "remainder";
      break;
    case BinaryOpType::Sub:
      return "sub";
      break;
    case BinaryOpType::Mod:
      return "mod";
      break;
    case BinaryOpType::Eq:
      return "eq";
      break;
    case BinaryOpType::NE:
      return "ne";
      break;
    case BinaryOpType::GT:
      return "gt";
      break;
    case BinaryOpType::GE:
      return "ge";
      break;
    case BinaryOpType::LT:
      return "lt";
      break;
    case BinaryOpType::LE:
      return "le";
      break;
    case BinaryOpType::BitwiseAnd:
      return "bitwise_and";
      break;
    case BinaryOpType::BitwiseOr:
      return "bitwise_or";
      break;
    case BinaryOpType::BitwiseXor:
      return "bitwise_xor";
      break;
    case BinaryOpType::Lshift:
      return "bitwise_left_shift";
      break;
    case BinaryOpType::Rshift:
      return "bitwise_right_shift";
      break;
    case BinaryOpType::Gcd:
      return "gcd";
      break;
    default:
      NVF_CHECK(
          false,
          "Unexpected operator type: ",
          bop->getBinaryOpType(),
          " in ",
          bop->toString());
  }
}

class FusionTranslator : public OptInConstDispatch {
 public:
  static std::unique_ptr<FusionDefinition> clone(const Fusion* fusion) {
    FusionTranslator cloner(fusion);
    cloner.clone();
    return std::move(cloner.fd_);
  }

 private:
  FusionTranslator(const Fusion* fusion)
      : fusion_(fusion),
        fd_(std::make_unique<FusionDefinition>(/*id=*/std::nullopt)) {}

  using OptInConstDispatch::handle;

  void clone() {
    fd_->setupDefinition();

    std::deque<nvfuser::Expr*> to_visit;

    // Add Fusion inputs to FusionDefinition
    for (nvfuser::Val* v : fusion_->inputs()) {
      OptOutConstDispatch::dispatch(v);

      // Add uses for input value to to_visit
      for (Expr* e : v->uses()) {
        to_visit.push_back(e);
      }
    }

    // Topological search of Fusion expressions
    std::unordered_set<nvfuser::Expr*> visited;
    while (!to_visit.empty()) {
      Expr* e = to_visit.front();
      to_visit.pop_front();

      // short-circuit: skip if already visited
      if (visited.count(e) > 0) {
        continue;
      }

      visited.insert(e);

      // Create RecordFunctor given inputs, outputs, and attributes.
      OptOutConstDispatch::dispatch(e);

      // Add output uses to to_visit
      for (Val* v : e->outputs()) {
        for (Expr* e : v->uses()) {
          to_visit.push_back(e);
        }
      }
    }

    // Outputs and Aliasing
    for (nvfuser::Val* v : fusion_->outputs()) {
      // Handle only TensorViews
      NVF_ERROR(v->isA<TensorView>());
      handleOutput(v->as<TensorView>());
    }

    fd_->finalizeDefinition();
  }

  void handle(const TensorView* tv) final {
    Tensor output = fd_->defineTensor(tv->nDims());

    std::vector<int64_t> shape;
    std::transform(
        tv->domain()->logical().begin(),
        tv->domain()->logical().end(),
        std::back_inserter(shape),
        [](IterDomain* id) {
          return (id->extent()->isConstScalar())
              ? id->extent()->evaluate().as<int64_t>()
              : -1;
        });

    fd_->defineRecord(new TensorRecord(
        {fd_->recordingState(output())},
        shape,
        tv->domain()->contiguity(),
        std::get<PrimDataType>(tv->dtype().type),
        tv->isCpuScalar(),
        tv->domain()->strideOrder()));

    map_val_to_fd_index_.emplace(tv, output());
  }

  void handleOutput(const TensorView* tv) {
    // Add fusion outputs to FusionDefinition
    size_t output_index = map_val_to_fd_index_.at(tv);
    fd_->defineRecord(new OutputRecord<TensorView>(
        {fd_->recordingState(output_index)}, serde::RecordType::OutputTv));
  }

  template <typename ResultType, typename... ArgTypes>
  void handleOpRecord(
      const Expr* e,
      std::string op_name,
      std::function<ResultType(ArgTypes...)> fn,
      serde::RecordType record_type,
      ResultType result,
      ArgTypes... args) {
    std::vector<State> argument_states;
    std::transform(
        e->inputs().begin(),
        e->inputs().end(),
        std::back_inserter(argument_states),
        [&](auto arg) {
          return fd_->recordingState(map_val_to_fd_index_.at(arg));
        });

    fd_->defineRecord(new OpRecord<ResultType, ArgTypes...>(
        argument_states,
        {fd_->recordingState(map_val_to_fd_index_.at(result))},
        "ops." + op_name,
        record_type,
        fn));
  }

  void handle(const BinaryOp* bop) final {
    bool lhs_tv = bop->lhs()->isA<TensorView>();
    bool rhs_tv = bop->rhs()->isA<TensorView>();

    if (lhs_tv || rhs_tv) {
      Tensor output = fd_->defineTensor(bop->out()->as<TensorView>()->nDims());
      map_val_to_fd_index_.emplace(bop->out(), output());

      if (lhs_tv && rhs_tv) {
        handleOpRecord(
            bop,
            getString(bop),
            getFunction<TensorView*, TensorView*, TensorView*>(bop),
            serde::RecordType::Binary_TV,
            bop->out()->as<TensorView>(),
            bop->lhs()->as<TensorView>(),
            bop->rhs()->as<TensorView>());
      } else if (lhs_tv && !rhs_tv) {
        handleOpRecord(
            bop,
            getString(bop),
            getFunction<TensorView*, TensorView*, Val*>(bop),
            serde::RecordType::Binary_TV_VAL,
            bop->out()->as<TensorView>(),
            bop->lhs()->as<TensorView>(),
            bop->rhs());
      } else {
        handleOpRecord(
            bop,
            getString(bop),
            getFunction<TensorView*, Val*, TensorView*>(bop),
            serde::RecordType::Binary_VAL_TV,
            bop->out()->as<TensorView>(),
            bop->lhs(),
            bop->rhs()->as<TensorView>());
      }
    } else {
      NVF_ERROR(false, "Not Supported");
      handleOpRecord(
          bop,
          getString(bop),
          getFunction<Val*, Val*, Val*>(bop),
          serde::RecordType::Binary_VAL,
          bop->out(),
          bop->lhs(),
          bop->rhs());
    }
  }

 private:
  const Fusion* fusion_ = nullptr;
  std::unique_ptr<FusionDefinition> fd_;
  std::unordered_map<const nvfuser::Val*, size_t> map_val_to_fd_index_;
};

} // namespace

std::unique_ptr<FusionDefinition> clone(const Fusion* fusion) {
  return FusionTranslator::clone(fusion);
}

} // namespace nvfuser::python_frontend
