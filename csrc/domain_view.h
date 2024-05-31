// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#pragma once

#include <ir/builder.h>
#include <ir/internal_base_nodes.h>
#include <val_graph.h>

#ifndef DYNAMIC_TYPE_CHECK
#define DYNAMIC_TYPE_CHECK NVF_ERROR
#endif

#include <dynamic_type/dynamic_type.h>

// IterDomainLikeObjectView, or IDLOView in short, is a convenient helper class
// for scheduling IterDomain-like objects (IDLO), where IDLO can be either an
// IterDomain or a ValGroup of IterDomains. The interface of IDLOView is similar
// to that of TesorViews, that is, it has merge, split, etc. However, it only
// has a single "domain", instead of having multiple domains like "logical
// domain", "loop domain", etc. IDLOView is typically used as follows:
//
// Example 1:
//   IterDomain *id0, *id1;
//   IDLOView v({id0, id1});
//   v.merge(0);
// The above code will create a new Merge object whose inputs are (id0, id1),
// and output is a newly created IterDomain, say id01. After the merge, v will
// become [id01].
//
// IDLOView can do transformations in batch, like shown in the following
// example:
//
// Example 2:
//   IterDomain *id0, *id1, *id2, *id3;
//   IDLOView v({{id0, id1}, {id2, id3}});
//   v.merge(0);
// This is equivalent to say: please merge id0 with id2, and id1 with id3.
// The above code will create two new Merge objects whose inputs are (id0, id2),
// and (id1, id3), and outputs are newly created IterDomains, say id02 and id13.
// After the merge, v will become [{id02, id13}].
//
// IDLOView can also do transformations in a "broadcasting" manner, like shown
// in the following two examples:
//
// Example 3:
//   IterDomain *id0, *id1, *id2;
//   IDLOView v({id0, {id1, id2}});
//   v.merge(0);
// This is equivalent to say: please merge id0 with id1, and id0 with id2.
// The above code will create two new Merge objects whose inputs are (id0, id1),
// and (id0, id2), and outputs are newly created IterDomains, say id01 and id02.
// After the merge, v will become [{id01, id02}].
//
// Example 4:
//   IterDomain *id0, *id1, *id2;
//   IDLOView v({{id0, id1}, id2});
//   v.merge(0);
// This is equivalent to say: please merge id0 with id2, and id1 with id2.
// The above code will create two new Merge objects whose inputs are (id0, id2),
// and (id1, id2), and outputs are newly created IterDomains, say id02 and id12.
// After the merge, v will become [{id02, id12}].
//
// IDLOView also works on ValGraphs of IterDomains. For example:
//
// Example 5:
//   ValGraph graph;
//   IterDomain *id0, *id1;
//   ValGroup g0{id0}, g1{id1};
//   IDLOView v({ValGroupAndItsGraph{g0, &g}, ValGroupAndItsGraph{g1, &g}});
//   v.merge(0);
// If there is already a merge of g0 and g1 in graph that outputs g01, then v
// will reuse that output ValGroup and becomes [g01]. Otherwise, the above code
// will create a new ExprGroup containing a Merge of g0 and g1, and the
// output ValGroup of this ExprGroup is a newly created ValGroup, say g01. The
// newly created ExprGroups and ValGroups will be added to the ValGraph. The v
// after the merge will be [g01].
//
// Batching and broadcasting as demonstrated in Example 2, 3, 4 works for
// ValGroups as well. Besides, IDLOView also supports "type promotion" from
// IterDomain to ValGroup. For example:
//
// Example 6:
//   ValGraph graph;
//   IterDomain *id0, *id1;
//   ValGroup g0{id0}, g1{id1};
//   IDLOView v({id0, ValGroupAndItsGraph{g1, &g}});
//   v.merge(0);
// This is equivalent to Example 5. You will get [g01].
//
// Example 7:
//   ValGraph graph;
//   IterDomain *id0, *id1;
//   ValGroup g0{id0}, g1{id1};
//   IDLOView v({ValGroupAndItsGraph{g0, &g}, id1});
//   v.merge(0);
// This is also equivalent to Example 5. You will get [g01].

namespace nvfuser {

using IterDomainLikeObject = dynamic_type::DynamicType<
    dynamic_type::Containers<std::vector>,
    IterDomain*,
    ValGroupAndItsGraph>;

using IDLO = IterDomainLikeObject;

struct IterDomainLikeObjectView {
  std::vector<IterDomainLikeObject> domain;

  template <typename T>
  std::vector<T> as() const {
    std::vector<T> result;
    std::transform(
        domain.begin(), domain.end(), std::back_inserter(result), [](auto x) {
          return (T)x;
        });
    return result;
  }

  // TODO: split is not implemented yet
  void split(int64_t axis, Val* factor, bool inner_split = true);
  void split(int64_t axis, int64_t factor, bool inner_split = true);

  void merge(int64_t axis_o, int64_t axis_i);
  void merge(int64_t axis) {
    merge(axis, axis + 1);
  }
};

using IDLOView = IterDomainLikeObjectView;

} // namespace nvfuser
