// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#include <bfs.h>
#include <expr_evaluator.h>
#include <ir/cute_translation.h>
#include <ir/utils.h>
#include <iter_visitor.h>
#include <utils.h>

#include <ostream>
#include <sstream>

namespace nvfuser {

namespace cute_translation {

std::ostream& operator<<(std::ostream& os, const MultipliedString& s) {
  if (s.factor == 1) {
    os << s.str;
  } else {
    os << s.factor << "*" << s.str;
  }
  return os;
}

std::string MultipliedString::toString() const {
  std::stringstream ss;
  ss << *this;
  return ss.str();
}

std::ostream& operator<<(std::ostream& os, const Int& i) {
  if (std::holds_alternative<int64_t>(i)) {
    os << std::get<int64_t>(i);
  } else if (std::holds_alternative<MultipliedString>(i)) {
    os << std::get<MultipliedString>(i);
  } else if (std::holds_alternative<std::shared_ptr<IntTuple>>(i)) {
    os << *std::get<std::shared_ptr<IntTuple>>(i);
  }
  return os;
}

std::string Int::toString() const {
  std::stringstream ss;
  ss << *this;
  return ss.str();
}

// We multiply Ints to compute strides. When we multiply an IntTuple by a scalar
// we simply multiply each element of the tuple. Multiplying two IntTuples is
// more complicated. For example (a,b) * (c,d)
Int operator*(const Int& a, const Int& b) {
  if (std::holds_alternative<int64_t>(a) &&
      std::holds_alternative<int64_t>(b)) {
    return {std::get<int64_t>(a) * std::get<int64_t>(b)};
  } else if (
      std::holds_alternative<MultipliedString>(a) &&
      std::holds_alternative<int64_t>(b)) {
    return {std::get<MultipliedString>(a) * std::get<int64_t>(b)};
  } else if (
      std::holds_alternative<int64_t>(a) &&
      std::holds_alternative<MultipliedString>(b)) {
    return {std::get<int64_t>(a) * std::get<MultipliedString>(b)};
  }
  // TODO: Handle multiplication of IntTuples properly
  return {MultipliedString{a.toString() + "*" + b.toString()}};
}

// When we do ceilDiv((a, b), c) we are indicating that we're splitting the
// linearized index of (a, b) by factor c and taking the outer dim
Int ceilDivInt(const Int& a, const Int& b) {
  // TODO: Handle more cases
  if (std::holds_alternative<int64_t>(a) &&
      std::holds_alternative<int64_t>(b)) {
    return {ceilDiv(std::get<int64_t>(a), std::get<int64_t>(b))};
  } else {
    return {
        MultipliedString{"ceilDiv(" + a.toString() + "," + b.toString() + ")"}};
  }
}

std::ostream& operator<<(std::ostream& os, const IntTuple& t) {
  os << "(";
  bool first = true;
  for (const Int& i : t) {
    if (!first) {
      os << ",";
    }
    first = false;
    os << i;
  }
  os << ")";
  return os;
}

std::string IntTuple::toString() const {
  std::stringstream ss;
  ss << *this;
  return ss.str();
}

std::ostream& operator<<(std::ostream& os, const CuteLayout& layout) {
  os << layout.shape << ":" << layout.stride;
  return os;
}

std::string CuteLayout::toString() const {
  std::stringstream ss;
  ss << *this;
  return ss.str();
}

CuteLayout CuteConverter::getLayout(
    const std::vector<IterDomain*>& loop,
    const std::vector<IterDomain*>& alloc,
    MemoryType mtype,
    const std::vector<std::optional<bool>>& contiguity) const {
  // A CuTE layout is essentially a mapping from 1 to N many ints to a single
  // int. We can think of the inputs to this mapping as the loop indices of
  // the present expression.
  //
  // For example:
  //    (3, 7):(1, 3) maps input [1, 2] to the single int 1*1 + 7*3 = 22
  //
  // Note that not all allocation IDs are actually _allocated_. For example, a
  // tensor in shared memory might have an allocation ID that is parallelized
  // BIDy. Since shard memory is not shared across CTAs, that dimension will
  // not appear in the index so we should ignore it. The layouts we return
  // ignore the presence of such dimensions entirely.

  // Start by gathering the allocated allocation dimensions
  std::vector<IterDomain*> true_alloc;
  std::vector<std::optional<bool>> true_contig;
  for (const auto [id, contig] : zip(alloc, contiguity)) {
    if (!id->isIteration() &&
        !ir_utils::isMemorySharedAcross(mtype, id->getParallelType())) {
      continue;
    }
    true_alloc.push_back(id);
    true_contig.push_back(contig);
  }

  // Now set up a cute layout describing this
  CuteLayout base_alloc_layout = getInitialLayout(true_alloc, true_contig);

  // Now traverse from the base (allocation) layout to the loop domain. At
  // each stage, we fill out a size/stride combo for each ID
  struct IdInfo {
    Int shape;
    Int stride;
    // If this is the inner part of a split, or this started as a contiguous
    // dimension inner to another domain, this points to that outer contiguous
    // domain.
    IterDomain* contig_outer = nullptr;
  };
  std::unordered_map<IterDomain*, std::pair<Int, Int>> id_size_stride;
  NVF_ERROR_EQ(base_alloc_layout.size(), true_alloc.size());
  for (size_t i : arange(true_alloc.size())) {
    // Initialize id_size_stride with allocation domain mappings
    id_size_stride.emplace(
        true_alloc.at(i),
        std::pair<Int, Int>{
            base_alloc_layout.shape.at(i), base_alloc_layout.stride.at(i)});
  }
  for (const auto& [expr, direction] :
       getExprsBetween<IRBFS>(
           /*from=*/{true_alloc.begin(), true_alloc.end()},
           /*to=*/{loop.begin(), loop.end()},
           /*require_all_to_visited=*/false)
           .first) {
    NVF_ERROR_EQ(
        direction,
        Direction::Forward,
        "Only forward direction is supported at this time");
    if (auto* split = dynamic_cast<Split*>(expr)) {
      auto it = id_size_stride.find(split->in());
      if (it == id_size_stride.end()) {
        NVF_THROW("TODO: Use 1:1 here");
        continue;
      }
      auto& [in_shape, in_stride] = it->second;
      Int inner_shape = getIntFromVal(split->factor(), "split");
      Int outer_shape = ceilDivInt(in_shape, inner_shape);
      Int inner_stride = in_stride;
      Int outer_stride = inner_stride * inner_shape;
      id_size_stride.emplace(
          split->outer(), std::pair<Int, Int>{outer_shape, outer_stride});
      id_size_stride.emplace(
          split->inner(), std::pair<Int, Int>{inner_shape, inner_stride});
    } else if (auto* merge = dynamic_cast<Merge*>(expr)) {
      auto it = id_size_stride.find(merge->outer());
      // Use 1:1 when ID is not found since that's an unallocated dim
      Int outer_shape = it == id_size_stride.end() ? Int{1} : it->second.first;
      Int outer_stride = it == id_size_stride.end() ? Int{1} : it->second.second;
      it = id_size_stride.find(merge->inner());
      Int inner_shape = it == id_size_stride.end() ? Int{1} : it->second.first;
      Int inner_stride = it == id_size_stride.end() ? Int{1} : it->second.second;

      // TODO: Check if this is a contiguous merge. If so, then we can flatten
      // it IF NOT contiguous merge, then create a new IntTuple consisting of
      // the incoming shapes/strides
      // For now, we don't flatten any merges, we just represent them as
      // IntTuple always
      Int out_shape = std::make_shared<IntTuple>(
          std::vector<Int>{outer_shape, inner_shape});
      Int out_stride = std::make_shared<IntTuple>(
          std::vector<Int>{outer_stride, inner_stride});

      id_size_stride.emplace(
          merge->out(), std::pair<Int, Int>{out_shape, out_stride});
    } else if (auto* swizzle = dynamic_cast<Swizzle*>(expr)) {
      Int outX_shape = {MultipliedString{"swizsh"}};
      Int outX_stride = {MultipliedString{"swizstr"}};
      Int outY_shape = {MultipliedString{"swizsh"}};
      Int outY_stride = {MultipliedString{"swizstr"}};

      id_size_stride.emplace(
          swizzle->outX(), std::pair<Int, Int>{outX_shape, outX_stride});
      id_size_stride.emplace(
          swizzle->outY(), std::pair<Int, Int>{outY_shape, outY_stride});
    } else if (auto* swizzle = dynamic_cast<Swizzle2D*>(expr)) {
      // TODO
      NVF_THROW("Swizzle2D support not yet implemented ", swizzle->toString());
    } else {
      NVF_THROW("Unsupported expr ", expr->toString());
    }
  }
  // Now fill out the final layout by looking up the loop ID sizes and strides
  // in the id_size_stride map
  IntTuple final_shape;
  IntTuple final_stride;
  final_shape.reserve(loop.size());
  final_stride.reserve(loop.size());
  for (IterDomain* loop_id : loop) {
    auto it = id_size_stride.find(loop_id);
    if (it == id_size_stride.end()) {
      // TODO: I actually think this is fine but we need to check some
      // conditions and insert the right entry here
      continue;
    }
    final_shape.push_back(it->second.first);
    final_stride.push_back(it->second.second);
  }

  return {final_shape, final_stride};
}

Int CuteConverter::getIntFromVal(Val* v, const std::string& prefix) const {
  PolymorphicValue s = expr_eval_.evaluate(v);
  if (s.is<int64_t>()) {
    return {s.as<int64_t>()};
  } else {
    // Use a string to represent this extent if it's not constant
    return {MultipliedString{prefix + std::to_string(v->name())}};
  }
}

CuteLayout CuteConverter::getInitialLayout(
    const std::vector<IterDomain*>& alloc,
    const std::vector<std::optional<bool>>& contig) const {
  NVF_ERROR_EQ(alloc.size(), contig.size());
  IntTuple shape;
  IntTuple stride;
  shape.reserve(alloc.size());
  stride.reserve(alloc.size());
  // We build up the shape and strides in reverse from inner to outer
  Int inner_size{1};
  Int inner_stride{1};
  for (size_t i : std::ranges::views::reverse(arange(alloc.size()))) {
    IterDomain* id = alloc.at(i);
    std::optional<bool> c = contig.at(i);
    Int new_size = getIntFromVal(id->getMaybeExpandedExtent(), "sh");
    shape.push_back(new_size);

    Int new_stride = inner_stride * inner_size;
    if (id->isBroadcast() && id->hasExpandedExtent()) {
      // Set stride to zero for expanded broadcasts
      new_stride = {0};
    }

    if (c.has_value() && !c.value()) {
      stride.emplace_back(MultipliedString{"str" + std::to_string(id->name())});
    } else {
      // This ID is contiguous, so the stride is a multiple of the inner
      // stride and inner size
      stride.push_back(new_stride);
    }
    inner_stride = new_stride;
    inner_size = new_size;
  }
  std::reverse(shape.begin(), shape.end());
  std::reverse(stride.begin(), stride.end());
  return {shape, stride};
}

} // namespace cute_translation

} // namespace nvfuser
