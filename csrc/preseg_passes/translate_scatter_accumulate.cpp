// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#include <preseg_passes/translate_scatter_accumulate.h>

#include <ir/utils.h>
#include <ops/all_ops.h>

#include <sstream>

namespace nvfuser::preseg_passes {

namespace {

struct ScatterAccumulateInfo {
  ScatterOp* scatter;
  ReductionOp* reduction;
  FullOp* full;
  IterDomain* scatter_out_id;
  IterDomain* scatter_index_id;
};

std::pair<std::optional<ScatterAccumulateInfo>, std::string>
getMaybeScatterAccumulate(ScatterOp* scatter) {
  auto fail = []<typename... Args>(Args&&... args) {
    std::stringstream reason;
    ((reason << args << " "), ...);
    return std::make_pair(std::nullopt, reason.str());
  };

  auto scatter_in = scatter->in()->as<TensorView>();
  auto scatter_out = scatter->out()->as<TensorView>();

  if (std::ssize(scatter_out->getLogicalDomain()) != 2) {
    return fail(
        "Unsupported scatter + reduction pattern. Invalid scatter: ",
        scatter->toString());
  }

  if (scatter->accumulate()) {
    return fail(
        "Unsupported scatter + reduction pattern due to: ",
        scatter->toString());
  }

  // Scatter output must not be used outside of this op sequence
  if (scatter_out->uses().size() != 1) {
    return fail(
        "Unsupported scatter + reduction pattern due to multiple uses of "
        "scatter output: ",
        scatter->toString());
  }

  // Scatter input must be created by a FullOp with the same value
  // as the reduction init value
  auto full = dynamic_cast<FullOp*>(scatter_in->definition());
  if (full == nullptr) {
    return fail(
        "Unsupported scatter + reduction pattern due to missing full op");
  }

  // The full shape will be replaced with a 1D tensor of the same
  // size as the scatter dimension, and thus the output must not be
  // used by anything else.
  if (full->output(0)->uses().size() > 1) {
    return fail(
        "Unsupported scatter + reduction pattern due to multiple uses of full "
        "output: ",
        full->toString());
  }

  auto scatter_out_use = scatter_out->uses().at(0);

  // int32_t is automatically upcast to int64_t before reduction.
  if (auto cast = dynamic_cast<UnaryOp*>(scatter_out_use); cast != nullptr &&
      cast->getUnaryOpType() == UnaryOpType::Cast &&
      cast->in()->dtype() == DataType::Int32 &&
      cast->out()->dtype() == DataType::Int) {
    if (cast->output(0)->uses().size() != 1) {
      return fail(
          "Unsupported scatter + reduction pattern due to multiple uses of "
          "cast output: ",
          cast->toString());
    }
    scatter_out_use = cast->output(0)->uses().at(0);
  }

  auto reduction = dynamic_cast<ReductionOp*>(scatter_out_use);

  if (reduction == nullptr) {
    return fail("Unsupported scatter + reduction pattern. No reduction found.");
  }

  auto reduction_type = reduction->getReductionOpType();

  if (reduction_type != BinaryOpType::Add) {
    // Not strictly required. To relax this condition, need to allow
    // the other init values that work for other reduction types
    return fail(
        "Unsupported scatter + reduction pattern. Not a sum reduction: ",
        reduction->toString());
  }

  auto reduction_out = reduction->out()->as<TensorView>();

  if (!full->getFillValue()->isZero()) {
    return fail(
        "Unsupported scatter + reduction pattern due to invalid full op: ",
        full->toString());
  }

  const bool is_deterministic =
      !isFloatingPointType(reduction->in()->dtype()) ||
      reduction_type == BinaryOpType::Max ||
      reduction_type == BinaryOpType::Min;

  if (!is_deterministic) {
    return fail(
        "Unsupported scatter + reduction pattern due to: not deterministic");
  }

  // Scatter dimension of the index tensor must be a size-one
  // dimension.
  auto index_logical = TensorDomain::noReductions(
      scatter->index()->as<TensorView>()->getLogicalDomain());
  IterDomain* index_scatter_dim = index_logical.at(scatter->dim());
  if (!index_scatter_dim->extent()->isOneInt()) {
    return fail(
        "Unsupported scatter + reduction pattern due to: invalid index "
        "tensor: ",
        scatter->index()->toString());
  }

  // The non-scatter dimension must be reduced
  NVF_ERROR_EQ(
      scatter_out->getLogicalDomain().size(),
      reduction_out->getLogicalDomain().size());
  for (const auto& [scatter_logical_id, reduction_logical_id] :
       zip(scatter_out->getLogicalDomain(),
           reduction_out->getLogicalDomain())) {
    if (scatter_out->getLogicalDomain().at(scatter->dim()) ==
        scatter_logical_id) {
      // scatter dimension -> should not be reduced
      if (reduction_logical_id->isReduction()) {
        return fail(
            "Unsupported scatter + reduction pattern due to: scatter dimension "
            "should not be reduced: ",
            scatter_out->toString(),
            ", ",
            reduction_out->toString());
      }
    } else {
      // Non scatter dimension -> should be reduced
      if (!reduction_logical_id->isReduction()) {
        return fail(
            "Unsupported scatter + reduction pattern due to: non-scatter "
            "dimension should be reduced: ",
            scatter_out->toString(),
            ", ",
            reduction_out->toString());
      }
    }
  }

  // At this point, it should be safe to translate scatter +
  // reduction to scatter-accumulate as long as the new
  // scatter-accumualte is schedulable. Note that both scatter in and
  // index must be 2D
  auto scatter_out_id = scatter_out->getLogicalDomain().at(scatter->dim());
  auto scatter_index_id = index_logical.at(scatter->dim() == 0 ? 1 : 0);

  return {
      ScatterAccumulateInfo{
          scatter, reduction, full, scatter_out_id, scatter_index_id},
      ""};
}

// Translate the scatter and reduciton pattern to
// scatter-accumulate. See also the comment at
// getMaybeScatterAccumulate.
class ScatterAccumulateTranslator : public OptOutMutator {
 public:
  static void run(Fusion* fusion) {
    ScatterAccumulateTranslator translator(fusion);
  }

 private:
  ScatterAccumulateTranslator(Fusion* fusion) {
    for (auto expr : fusion->usedExprs()) {
      dispatchMutate(expr);
    }

    for (auto out : fusion->outputs()) {
      auto new_out = maybeMutated(out);
      if (new_out != out) {
        fusion->replaceOutput(out, new_out);
      }
    }
  }

  void mutate(Expr* expr) final {
    auto sop = dynamic_cast<ScatterOp*>(expr);
    if (sop == nullptr) {
      OptOutMutator::mutate(expr);
      return;
    }

    const auto& [info, reason] = getMaybeScatterAccumulate(sop);

    if (isDebugDumpEnabled(DebugDumpOption::PreSegmenterLogging)) {
      if (info.has_value()) {
        debug() << "Scatter-accumulate pattern detected with "
                << sop->toString();
      } else {
        debug() << "Scatter-accumulate pattern not detected with "
                << sop->toString() << " due to " << reason << std::endl;
      }
    }

    if (!info.has_value()) {
      OptOutMutator::mutate(expr);
      return;
    }

    auto scatter_dim = sop->dim();
    // Note that the output dtype may be different from the scatter
    // input as the reduction may automatically upcast from int32_t to int64_t.
    auto dtype = info->reduction->out()->dtype();

    // Squeeze the broadcast of the index input
    auto index_tv = sop->index()->as<TensorView>();
    index_tv = squeeze(index_tv, {scatter_dim});

    // Src can be a scalar. If it's a tensor, squeeze it as well
    auto src = sop->src();
    if (auto src_tv = dynamic_cast<TensorView*>(src)) {
      src = squeeze(src_tv, {scatter_dim});
    }
    src = maybeCastOp(dtype, src);

    // Create a new scatter input of size [n]
    auto scatter_inp = full(
        {info->scatter_out_id->extent()},
        info->reduction->init(),
        info->reduction->output(0)->dtype());

    auto scatter_out = scatter(
        scatter_inp, 0, index_tv, src, info->reduction->getReductionOpType());

    registerMutation(info->reduction->out(), scatter_out);
  }
};

} // namespace

void TranslateScatterAccumulate::runPass(Fusion* fusion) {
  FusionGuard fg(fusion);
  ScatterAccumulateTranslator::run(fusion);
}

} // namespace nvfuser::preseg_passes
