// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#pragma once

#include <ir/all_nodes.h>
#include <val_graph.h>

#include <string>
#include <vector>

namespace nvfuser {

std::string toString(
    const std::vector<IterDomain*>& id_group,
    int indent_size = 0);
std::string toString(
    const IdGroup& id_group,
    int indent_size = 0,
    bool with_ptr = false);

std::string toString(
    const std::vector<IdGroup>& id_groups,
    int indent_size = 0,
    bool with_ptr = false);

std::string toString(
    const IdGroups& id_groups,
    int indent_size = 0,
    bool with_ptr = false);

std::string toInlineString(const std::vector<IdGroup>& id_groups);
std::string toInlineString(const IdGroups& id_groups);

std::string toString(const std::vector<Expr*>& expr_group, int indent_size = 0);
std::string toString(
    const ExprGroup& expr_group,
    int indent_size = 0,
    bool with_ptr = false);

std::string toString(
    const IdGraph& id_graph,
    const std::vector<Expr*>& expr_group,
    int indent_size = 0,
    bool with_ptr = false);
std::string toString(
    const IdGraph& id_graph,
    const ExprGroup& expr_groups,
    int indent_size = 0,
    bool with_ptr = false);

std::string toString(
    const IdGraph& id_graph,
    const std::vector<ExprGroup>& expr_groups,
    int indent_size = 0,
    bool with_ptr = false);
std::string toString(
    const IdGraph& id_graph,
    const ExprGroups& expr_groups,
    int indent_size = 0,
    bool with_ptr = false);

std::string idGroupsString(
    const IdGraph& id_graph,
    int indent_size = 0,
    bool with_ptr = false);
std::string exprGroupsString(
    const IdGraph& id_graph,
    int indent_size = 0,
    bool with_ptr = false);
std::string definitionsString(
    const IdGraph& id_graph,
    int indent_size = 0,
    bool with_ptr = false);
std::string usesString(
    const IdGraph& id_graph,
    int indent_size = 0,
    bool with_ptr = false);

} // namespace nvfuser
