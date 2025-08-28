// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#include <expr_evaluator.h>
#include <ir/cute_translation.h>
#include <ir/utils.h>

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

Int operator*(const Int& a, const Int& b) {
  NVF_ERROR(
      !std::holds_alternative<std::shared_ptr<IntTuple>>(a),
      "Cannot multiple IntTuple yet");
  NVF_ERROR(
      !std::holds_alternative<std::shared_ptr<IntTuple>>(b),
      "Cannot multiple IntTuple yet");
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
  return {MultipliedString{"UNIMPLEMENTED MUL"}};
}

std::ostream& operator<<(std::ostream& os, const IntTuple& t) {
  os << "(";
  bool first = true;
  for (const Int& i : t) {
    if (!first) {
      os << ", ";
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

    return base_alloc_layout;
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
    ExpressionEvaluator ee;
    for (size_t i : std::ranges::views::reverse(arange(alloc.size()))) {
      IterDomain* id = alloc.at(i);
      std::optional<bool> c = contig.at(i);
      Int new_size;
      PolymorphicValue s = ee.evaluate(id->getMaybeExpandedExtent());
      if (s.is<int64_t>()) {
        new_size = {s.as<int64_t>()};
      } else {
        // Use a string to represent this extent if it's not constant
        new_size = {MultipliedString{"sz" + std::to_string(id->name())}};
      }
      shape.push_back(new_size);

      Int new_stride = inner_stride * inner_size;
      if (id->isBroadcast() && id->hasExpandedExtent()) {
        // Set stride to zero for expanded broadcasts
        new_stride = {0};
      }

      if (c.has_value() && !c.value()) {
        stride.emplace_back(
            MultipliedString{"str" + std::to_string(id->name())});
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
