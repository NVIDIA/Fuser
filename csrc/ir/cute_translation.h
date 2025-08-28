// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#pragma once

#include <exceptions.h>
#include <expr_evaluator.h>
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

// [CuTE <-> IterDomain correspondence]
//
// A CuTE Layout is a hierarchical representation of data layout for
// multidimensional tensors. Its most basic functionality is to map a collection
// of integers to a single integer. In practice, Layouts are used to map a
// collection of _loop indices_ to a linear memory address. The hierarchical
// nature of Layouts (IntTuple) is the most powerful aspect, allowing
// cute::Layout to represent any split/merge relationships as well as striding.
//
// A cute::Layout consists of two attributes: a shape and a stride. Both of
// these are IntTuples, which basically are nested tuples of integers. Layouts
// are printed as shape:stride. For example a contiguous column-major matrix of
// size 8x12 is represented by the Layout (8, 12):(1, 8).
//
// Note that the _rank_ of a Layout is the length of the Layout's shape as a
// tuple. Just as with most tensor representations, the lenght of the shape and
// stride attributes must match. In fact, I am not sure if it's required but
// most examples have shapes and strides with the exact same shape _as a tree_.
// TODO: Verify conditions on shape and stride attributes
//
// Example: hierarchical shape/stride
//   A more complicated example showing how IntTuple and Int can be nested is
//   the layout ((2,2),2):((4,1),2). As trees, this looks like the following
//
//   shape:   ( (2,2) , 2 )  stride:   ( (4,1) , 2 )
//               / \                      / \              .
//              2   2                    4   1
//
//   This Layout can be used to convert any of the following to a linear index:
//
//    ((1,0),1)  -> (1*4 + 0*1) + 1*2 = 6
//    (3,1)     -> ((3%2)*4 + (3//2)*1) + 1*2 = 8
//    (7)        -> ((3%2)*4 + (3//2)*1) + 1*2 = 8
//
//  This example shows clearly how we convert a coordinate to a linear index. If
//  the coordinate is already in its "natural form" i.e. it is shaped the same
//  way as the shape/stride of the Layout, this looks like a simple inner
//  product with the stride: we just multiple the coordinates by the
//  corresponding strides and sum them all up.
//
//  The second and third examples above show how we can summarize a coordinate
//  subtree using a single coordinate. To do this, we just map the coordinate to
//  a multi-dimensional coordinate via an inner to outer (column-major) mapping.
//  After doing that recursively until termination, we arrive at a "natural"
//  coordinate that can be inner producted with the stride. Specifically, this
//  mapping looks like the following:
//
//    (3,1)   ->  ((3%2,3//2),1) = ((1,2),1)
//    (7)     ->  (7%(2*2),7//(2*2)) = (3,1)  ->  ((1,2,1)
//
//  This shows that we are really divmod'ing by the "size" of the shape in order
//  to unflatten it.
//
// https://docs.nvidia.com/cutlass/media/docs/cpp/cute/01_layout.html
//
// IterDomains are meant to solve the same indexing problem. We use vectors of
// IterDomains to represent a particular view of a Tensor, with one of these
// vectors corresponding to an outer-to-inner allocation view. We don't directly
// represent strides, but for that allocation domain we do associate a flag to
// indicate whether each dimension in contiguous, meaning whether the immediate
// inner dimension to that dimension has any padding.

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
  using std::variant<int64_t, MultipliedString, std::shared_ptr<IntTuple>>::
      variant;
  std::string toString() const;
};
Int operator*(const Int& a, const Int& b);
Int ceilDivInt(const Int& a, const Int& b);
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

  inline CuteLayout consumerLoopToProducerAlloc(
      TensorView* producer,
      TensorView* consumer) const {
    const std::vector<IterDomain*>& loop = consumer->getLoopDomain();
    const std::vector<IterDomain*>& alloc =
        producer->getMaybeAllocationDomain();
    return getLayout(
        loop, alloc, producer->getMemoryType(), producer->getContiguity());
  }

  inline CuteLayout loopToAlloc(TensorView* tv) const {
    return consumerLoopToProducerAlloc(tv, tv);
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

  //! Returns an Int that is an int64_t if v is constant, otherwise returns a
  //! MultipliedString like sz13 if v->name()==13 and prefix=="sz"
  Int getIntFromVal(Val* v, const std::string& prefix) const;

 private:
  const IdModel id_model_;
  ExpressionEvaluator expr_eval_;
};

} // namespace cute_translation

} // namespace nvfuser
