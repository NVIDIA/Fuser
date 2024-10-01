// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#include <debug.h>
#include <ir/utils.h>
#include <logical_domain_map.h>
#include <ops/alias.h>
#include <ops/arith.h>
#include <options.h>
#include <preseg_passes/remove_bcast_squeeze.h>

namespace nvfuser::preseg_passes {

namespace {

enum class AxisOp { PRESERVE, SQUEEZE, BROADCAST };

std::ostream& operator<<(std::ostream& os, AxisOp op) {
  switch (op) {
    case AxisOp::PRESERVE:
      os << "PRESERVE";
      break;
    case AxisOp::SQUEEZE:
      os << "SQUEEZE";
      break;
    case AxisOp::BROADCAST:
      os << "BROADCAST";
      break;
  }
  return os;
}

//! This represents the combined state of a collection of broadcast and squeeze
//! operations
using AxisOps = std::vector<AxisOp>;

//! Convert a broadcast Expr to an AxisOps descriptor
AxisOps broadcastToAxisOps(BroadcastOp* bcast) {
  AxisOps ops;
  const std::vector<bool>& flags = bcast->getBroadcastDimFlags();
  ops.reserve(flags.size());
  for (bool flag : flags) {
    ops.push_back(flag ? AxisOp::BROADCAST : AxisOp::PRESERVE);
  }
  return ops;
}

//! Convert a squeeze Expr to an AxisOps descriptor
AxisOps squeezeToAxisOps(SqueezeOp* squeeze) {
  AxisOps ops;
  const std::vector<bool>& flags = squeeze->getSqueezeDimFlags();
  ops.reserve(flags.size());
  for (bool flag : flags) {
    ops.push_back(flag ? AxisOp::SQUEEZE : AxisOp::PRESERVE);
  }
  return ops;
}

//! Check whether the loop domains are the same size and have the same
//! parallelization, ignoring reduction dimensions.
bool hasCompatibleParallelization(TensorView* orig_tv, TensorView* new_tv) {
  std::vector<IterDomain*> orig_domain =
      TensorDomain::noReductions(orig_tv->getLoopDomain());
  std::vector<IterDomain*> new_domain =
      TensorDomain::noReductions(new_tv->getLoopDomain());
  NVF_ERROR(orig_domain.size() == new_domain.size());
  for (size_t i : c10::irange(orig_domain.size())) {
    if (orig_domain[i]->getParallelType() != new_domain[i]->getParallelType()) {
      return false;
    }
  }
  return true;
}

//! Checks whether this is a simple Set of a TensorView. If not, then this might
//! represent a scalar set, or a segment_set.
bool isSimpleTVSet(Expr* expr) {
  auto* ldst = dynamic_cast<LoadStoreOp*>(expr);
  if (ldst == nullptr) {
    return false;
  }
  auto in_tv = dynamic_cast<TensorView*>(ldst->in());
  auto out_tv = dynamic_cast<TensorView*>(ldst->out());
  return ldst->opType() == LoadStoreOpType::Set && in_tv != nullptr &&
      out_tv != nullptr
      // The hasRoot() check is to prevent picking up Set.Permute ops here
      && !ldst->out()->as<TensorView>()->hasRoot()
      // A set operation that changes parallelization is not considered trivial
      && hasCompatibleParallelization(in_tv, out_tv);
}

//! This defines the types of operations that are eligible for simplification in
//! this pass.
bool isReplaceableExpr(Expr* expr) {
  if (expr == nullptr) {
    return false;
  }
  return expr->isA<BroadcastOp>() || expr->isA<SqueezeOp>() ||
      isSimpleTVSet(expr);
}

//! Convert a LoadStoreOp to an AxisOps of all PRESERVE ops
AxisOps setToAxisOps(LoadStoreOp* ldst) {
  NVF_ERROR(isSimpleTVSet(ldst));
  return AxisOps(
      ldst->in()->as<TensorView>()->getLogicalDomain().size(),
      AxisOp::PRESERVE);
}

//! Convert a replaceable op to an AxisOps object
AxisOps exprToAxisOps(Expr* expr) {
  if (auto* squeeze = dynamic_cast<SqueezeOp*>(expr)) {
    return squeezeToAxisOps(squeeze);
  } else if (auto* bcast = dynamic_cast<BroadcastOp*>(expr)) {
    return broadcastToAxisOps(bcast);
  } else if (auto* ldst = dynamic_cast<LoadStoreOp*>(expr)) {
    return setToAxisOps(ldst);
  }
  NVF_THROW(
      "exprToAxisOps expects BroadcastOp or SqueezeOp. Found ",
      expr->toString());
}

//! Return non-null if we are unable to simplify this combination to a single
//! operation. Otherwise returns the type of operation (PRESERVE indicates this
//! is a set operation).
std::optional<AxisOp> getSimplifiedOpType(const AxisOps& ops) {
  bool has_broadcast = false, has_squeeze = false;
  for (const AxisOp op : ops) {
    switch (op) {
      case AxisOp::PRESERVE:
        break;
      case AxisOp::SQUEEZE:
        has_squeeze = true;
        break;
      case AxisOp::BROADCAST:
        has_broadcast = true;
        break;
    }
  }
  if (has_broadcast && has_squeeze) {
    // Composite op
    return std::nullopt;
  }
  if (has_broadcast) {
    return AxisOp::BROADCAST;
  }
  if (has_squeeze) {
    return AxisOp::SQUEEZE;
  }
  // PRESERVE indicates this is a set op
  return AxisOp::PRESERVE;
}

//! This is useful for getting the axis flags needed to create a new BroadcastOp
//! or SqueezeOp.
std::vector<bool> nonPreservedDims(const AxisOps& ops) {
  std::vector<bool> flags;
  flags.reserve(ops.size());
  for (size_t i : c10::irange(ops.size())) {
    flags.push_back(ops[i] != AxisOp::PRESERVE);
  }
  return flags;
}

//! Given a descriptors of two sequences of broadcast+squeeze ops, return a
//! descriptor of their composition
AxisOps composeOps(const AxisOps& prev, const AxisOps& next) {
  size_t prev_pos = 0, next_pos = 0;
  size_t prev_size = prev.size(), next_size = next.size();
  AxisOps ops;
  while (true) {
    // If there are previous squeezes, insert them since they don't affect next
    // position
    while (prev_pos < prev_size && prev[prev_pos] == AxisOp::SQUEEZE) {
      ops.push_back(AxisOp::SQUEEZE);
      prev_pos++;
    }

    // prev[prev_pos] now gives the provenance of the axis that next_pos points
    // to if it's not a broadcast. If it is a broadcast, then we'll unzip some
    // of these squeezes.
    while (next_pos < next_size && next[next_pos] == AxisOp::BROADCAST) {
      if (!ops.empty() && ops.back() == AxisOp::SQUEEZE) {
        // This is a "squeeze then broadcast" pattern
        ops.back() = AxisOp::PRESERVE;
      } else {
        ops.push_back(AxisOp::BROADCAST);
      }
      next_pos++;
    }

    if (prev_pos >= prev_size || next_pos >= next_size) {
      NVF_ERROR(
          prev_pos >= prev_size && next_pos >= next_size,
          "Failed to align some ops");
      break;
    }

    // Now we have prev[prev_pos] != SQUEEZE and next[next_pos] != BROADCAST,
    // which means prev[prev_pos] provides the next op with its input. So here
    // we are actually composing these aligned ops.
    if (prev[prev_pos] == AxisOp::PRESERVE) {
      ops.push_back(next[next_pos]);
    } else if (next[next_pos] == AxisOp::PRESERVE) {
      ops.push_back(prev[prev_pos]);
    } else {
      // Otherwise this is a "broadcast then squeeze" pattern, so skip it
      NVF_ERROR(
          next[next_pos] == AxisOp::SQUEEZE &&
          prev[prev_pos] == AxisOp::BROADCAST);
    }
    prev_pos++;
    next_pos++;
  }
  return ops;
}

TensorView* maybeDoReplacement(TensorView* orig) {
  Expr* second = orig->definition();
  if (!isReplaceableExpr(second)) {
    return orig;
  }
  Expr* first = second->input(0)->definition();
  if (!isReplaceableExpr(first)) {
    return orig;
  }

  AxisOps first_ops = exprToAxisOps(first);
  AxisOps second_ops = exprToAxisOps(second);
  AxisOps simplified_ops = composeOps(first_ops, second_ops);
  std::optional<AxisOp> simple_op_type_opt =
      getSimplifiedOpType(simplified_ops);
  if (!simple_op_type_opt.has_value()) {
    return orig;
  }
  TensorView* replacement = orig;
  if (simplified_ops == first_ops) {
    // The second op was simply a "Set" operation, so we just skip it
    replacement = first->output(0)->as<TensorView>();
  } else {
    TensorView* input_tv = first->input(0)->as<TensorView>();
    switch (simple_op_type_opt.value()) {
      case AxisOp::PRESERVE:
        // This is equivalent to a set Op
        replacement = input_tv;
        // Check that parallelization is consistent for replacement. We do not
        // want to alter the parallelization of input_tv here; we only
        // parallelize _new_ TVs. If the parallelization is inconsistent, we
        // simply refuse to make the replacement.
        if (!hasCompatibleParallelization(orig, replacement)) {
          return orig;
        }
        break;
      case AxisOp::SQUEEZE:
        replacement = squeeze(input_tv, nonPreservedDims(simplified_ops));
        break;
      case AxisOp::BROADCAST:
        replacement = broadcast(input_tv, nonPreservedDims(simplified_ops));
        break;
    }
  }
  NVF_ERROR(replacement != orig, "Expected non-trivial replacement");
  ir_utils::replaceValInAllExprInputsAndFusionOutputs(orig, replacement);
  std::vector<IterDomain*> old_loop =
      TensorDomain::noReductions(orig->getLoopDomain());
  std::vector<IterDomain*> new_loop =
      TensorDomain::noReductions(replacement->getLoopDomain());
  NVF_ERROR(new_loop.size() == old_loop.size());
  for (size_t i : c10::irange(old_loop.size())) {
    if (old_loop[i]->isParallelized()) {
      NVF_ERROR(
          old_loop[i]->isDeviceDim(),
          "Before scheduling, we expect the only parallelized ",
          "dimensions to be device dims");
      // In particular, we might have a Device dimension parallelized for the
      // output, which we need to preserve.
      new_loop[i]->parallelize(old_loop[i]->getParallelType());
    }
  }
  return replacement;
}

// Remove broadcast-squeeze and squeeze-broadcast patterns
void removeBcastSqueeze(Fusion* fusion) {
  // Iterate from outputs toward producers using a depth-first search for
  // replaceable patterns
  std::vector<TensorView*> stack;
  for (Val* outp : fusion->outputs()) {
    if (auto* tv = dynamic_cast<TensorView*>(outp)) {
      stack.push_back(tv);
    }
  }
  std::unordered_set<TensorView*> visited;
  while (!stack.empty()) {
    TensorView* tv = stack.back();
    stack.pop_back();
    if (visited.count(tv) > 0) {
      continue;
    }
    // avoid re-visiting producers when they have multiple uses
    visited.insert(tv);
    TensorView* maybe_replaced_tv = maybeDoReplacement(tv);
    if (maybe_replaced_tv != tv) {
      // If we made a replacement, process it before proceeding. For example, if
      // we have broadcast->broadcast->squeeze, we might combine the second
      // broadcast with the squeeze to form a new pattern of broadcast->squeeze,
      // which we then need to process next.
      stack.push_back(maybe_replaced_tv);
    } else if (tv->definition() != nullptr) {
      // Recurse to TensorView producers of tv
      const auto producers =
          ir_utils::filterByType<TensorView>(tv->definition()->inputs());
      stack.insert(stack.end(), producers.begin(), producers.end());
    }
  }
}

} // namespace

void RemoveBcastSqueeze::runPass(Fusion* fusion) {
  removeBcastSqueeze(fusion);
}

} // namespace nvfuser::preseg_passes
