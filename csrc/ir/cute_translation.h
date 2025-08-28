// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#pragma once

#include <exceptions.h>
#include <id_model/id_model.h>
#include <ir/interface_nodes.h>
#include <ir/internal_base_nodes.h>
#include <type.h>

#include <cstdint>
#include <iosfwd>
#include <memory>
#include <optional>
#include <string>
#include <variant>
#include <vector>

namespace nvfuser {

namespace cute_translation {

// Forward declare IntTuple to help define nested Int type
struct IntTuple;

struct MultipliedString {
  std::string str;
  int64_t factor = 1;
  std::string toString() const;
};
std::ostream& operator<<(std::ostream& os, const MultipliedString& s);
inline MultipliedString operator*(
    const MultipliedString& a,
    const MultipliedString& b) {
  return {.str = a.str + "*" + b.str, .factor = a.factor * b.factor};
}
inline MultipliedString operator*(const int64_t& a, const MultipliedString& b) {
  return {.str = b.str, .factor = a * b.factor};
}
inline MultipliedString operator*(const MultipliedString& a, const int64_t& b) {
  return b * a;
}

// We can think of IntTuple as a tree where each node is an Int. For internal
// nodes, the Int is another IntTuple, but for leaf nodes it is either a
// concrete integer (int64_t) or a symbolic integer with a std::string name.
struct Int : public std::
                 variant<int64_t, MultipliedString, std::shared_ptr<IntTuple>> {
  std::string toString() const;
};
Int operator*(const Int& a, const Int& b);
std::ostream& operator<<(std::ostream& os, const Int& i);

//! This represents a cute::IntTuple
struct IntTuple : public std::vector<Int> {
  std::string toString() const;
};
std::ostream& operator<<(std::ostream& os, const IntTuple& t);

// A cute layout is a pair of IntTuples
// https://docs.nvidia.com/cutlass/media/docs/cpp/cute/01_layout.html
struct CuteLayout {
  IntTuple shape;
  IntTuple stride;

  inline size_t size() const {
    NVF_ERROR_EQ(shape.size(), stride.size());
    return shape.size();
  }

  std::string toString() const;
};
std::ostream& operator<<(std::ostream& os, const CuteLayout& layout);

class CuteConverter {
 public:
  CuteConverter(Fusion* fusion) : id_model_(fusion, /*build_graphs=*/true) {}

  inline CuteLayout logicalToAlloc(TensorView* tv) const {
    const std::vector<IterDomain*>& logical = tv->getLogicalDomain();
    const std::vector<IterDomain*>& alloc = tv->getMaybeAllocationDomain();
    return getLayout(logical, alloc, tv->getMemoryType(), tv->getContiguity());
  }

  inline CuteLayout loopToAlloc(TensorView* tv) const {
    const std::vector<IterDomain*>& loop = tv->getLoopDomain();
    const std::vector<IterDomain*>& alloc = tv->getMaybeAllocationDomain();
    return getLayout(loop, alloc, tv->getMemoryType(), tv->getContiguity());
  }

 private:
  CuteLayout getLayout(
      const std::vector<IterDomain*>& loop,
      const std::vector<IterDomain*>& alloc,
      MemoryType mtype,
      const std::vector<std::optional<bool>>& contiguity) const;

  CuteLayout getInitialLayout(
      const std::vector<IterDomain*>& alloc,
      const std::vector<std::optional<bool>>& contig) const;

 private:
  const IdModel id_model_;
};

} // namespace cute_translation

} // namespace nvfuser
