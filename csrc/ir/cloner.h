// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#pragma once

#include <disjoint_set.h>
#include <dispatch.h>
#include <exceptions.h>
#include <ir/builder.h>
#include <visibility.h>

#include <tuple>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace nvfuser {

class IrContainer;

//! Clones nodes from an exiting Fusion
//!
//! \warning IrCloner machinery is a specialized helper for implementing
//!   Fusion copy operations and the and limited scope of RecomputeTv below.
//!   It is not intended for any other uses.
//!
class IrCloner {
  friend class Statement;
  friend class IrBuilder;

 public:
  // NOLINTNEXTLINE(cppcoreguidelines-pro-type-member-init)
  explicit IrCloner(IrContainer* container);
  virtual ~IrCloner() = default;

  NVF_API Statement* clone(const Statement* statement);

#define DEFINE_CLONE_FOR_PRIM_TYPES(_t_y_p_e_) \
  _t_y_p_e_ clone(_t_y_p_e_ x) {               \
    return x;                                  \
  }

  DEFINE_CLONE_FOR_PRIM_TYPES(bool)
  DEFINE_CLONE_FOR_PRIM_TYPES(int8_t)
  DEFINE_CLONE_FOR_PRIM_TYPES(uint8_t)
  DEFINE_CLONE_FOR_PRIM_TYPES(int16_t)
  DEFINE_CLONE_FOR_PRIM_TYPES(uint16_t)
  DEFINE_CLONE_FOR_PRIM_TYPES(int32_t)
  DEFINE_CLONE_FOR_PRIM_TYPES(uint32_t)
  DEFINE_CLONE_FOR_PRIM_TYPES(int64_t)
  DEFINE_CLONE_FOR_PRIM_TYPES(uint64_t)
  DEFINE_CLONE_FOR_PRIM_TYPES(float)
  DEFINE_CLONE_FOR_PRIM_TYPES(double)

#undef DEFINE_CLONE_FOR_PRIM_TYPES

  template <class T>
  T* clone(const T* node) {
    return node ? clone(node->template as<Statement>())->template as<T>()
                : nullptr;
  }

  template <class T>
  std::vector<T> clone(const std::vector<T>& container) {
    // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
    std::vector<T> copy;
    copy.reserve(container.size());
    for (auto p : container) {
      copy.push_back(clone(p));
    }
    return copy;
  }

  template <class T>
  std::unordered_set<T> clone(const std::unordered_set<T>& container) {
    std::unordered_set<T> copy;
    copy.reserve(container.size());
    for (auto p : container) {
      copy.insert(clone(p));
    }
    return copy;
  }

  template <typename... Ts>
  std::tuple<Ts...> clone(const std::tuple<Ts...>& tup) {
    return std::apply(
        [this](auto&... x) {
          return std::make_tuple<Ts...>(this->clone(x)...);
        },
        tup);
  }

  template <typename T, typename U>
  std::pair<T, U> clone(const std::pair<T, U>& p) {
    return std::make_pair<T, U>(this->clone(p.first), this->clone(p.second));
  }

  template <class T, class U>
  std::unordered_map<T, U> clone(const std::unordered_map<T, U>& container) {
    std::unordered_map<T, U> copy;
    copy.reserve(container.size());
    for (const auto& [k, v] : container) {
      copy.emplace(clone(k), clone(v));
    }
    return copy;
  }

  template <typename T, typename Hash = std::hash<T>>
  DisjointSets<T, Hash> clone(const DisjointSets<T, Hash>& disjoint_sets) {
    DisjointSets<T, Hash> cloned_disjoint_sets;
    for (const auto& original_set : disjoint_sets.disjointSets()) {
      NVF_ERROR(!original_set->empty());
      bool first = true;
      typename DisjointSets<T, Hash>::DisjointSet new_set;
      for (const auto& val : *original_set) {
        auto clone_of_val = clone(val);
        if (first) {
          auto it = cloned_disjoint_sets.initializeSet(clone_of_val).first;
          new_set = it->second;
          first = false;
        } else {
          cloned_disjoint_sets.appendToSet(clone_of_val, new_set);
        }
      }
    }

    return cloned_disjoint_sets;
  }

  IrContainer* container() const {
    return ir_container_;
  }

 protected:
  NVF_API void registerClone(const Statement* src, Statement* clone);
  virtual Statement* handle(const Statement* s);

 protected:
  // We keep track of the original -> clone map so we don't
  // duplicate clones of the same object if referenced multiple times
  std::unordered_map<const Statement*, Statement*> clones_map_;

 private:
  // The destination Fusion container
  IrContainer* ir_container_ = nullptr;

  // Builder to make all the new nodes
  IrBuilder builder_;
};

// Replicates all expressions used to generate the provided TensorView. Does not
// replicate inputs. Does not replicate scalar values. In other words the value
// provided will be recomputed from the inputs of the fusion.
class RecomputeTv : private IrCloner {
 public:
  // Replicates expressions and values in provided expressions.
  NVF_API static TensorView* recompute(
      TensorView* tv,
      const std::vector<Val*>& from = {});

 private:
  RecomputeTv(Fusion* fusion);
  Statement* handle(const Statement* s) override;
  Statement* handle(const TensorDomain*);
};

//! Clone an IR node, forwarding the arguments to the IrCloner constructor.
template <class T>
T* IrBuilder::clone(const T* src, IrCloner* ir_cloner) {
  NVF_ERROR(
      ir_cloner != nullptr,
      "Cannot use create when a cloner object is set. Use clone.");

  NVF_ERROR(
      ir_cloner->container() != nullptr,
      "Cloner doesn't have a valid container to store cloned object.");

  T* dest = new T(src, ir_cloner);
  const Statement* src_stmt = dynamic_cast<const Statement*>(src);
  Statement* dest_stmt = dynamic_cast<Statement*>(dest);

  auto dest_container = ir_cloner->container();
  auto src_container = src_stmt->container();

  dest_container->registerStmt(IrBuilderPasskey(dest_container), dest_stmt);

  if (src_container != dest_container) {
    dest_stmt->setName(IrBuilderPasskey(dest_container), src_stmt->name());
  }

  ir_cloner->registerClone(src_stmt, dest_stmt);

  return dest;
}

} // namespace nvfuser
