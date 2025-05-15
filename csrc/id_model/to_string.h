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

std::string NVF_API
toString(const std::vector<Val*>& val_group, int indent_size = 0);

std::string NVF_API
toString(const std::vector<IterDomain*>& id_group, int indent_size = 0);

std::string NVF_API
toString(const ValGroup& id_group, int indent_size = 0, bool with_ptr = false);

std::string NVF_API toString(
    const std::vector<ValGroup>& id_groups,
    int indent_size = 0,
    bool with_ptr = false);

std::string NVF_API toString(
    const ValGroups& id_groups,
    int indent_size = 0,
    bool with_ptr = false);

std::string toInlineString(const std::vector<ValGroup>& id_groups);
std::string toInlineString(const ValGroups& id_groups);

std::string NVF_API
toString(const std::vector<Expr*>& expr_group, int indent_size = 0);
std::string NVF_API toString(
    const ExprGroup& expr_group,
    int indent_size = 0,
    bool with_ptr = false);

std::string NVF_API toString(
    const ValGraph& id_graph,
    const std::vector<Expr*>& expr_group,
    int indent_size = 0,
    bool with_ptr = false);
std::string NVF_API toString(
    const ValGraph& id_graph,
    const ExprGroup& expr_groups,
    int indent_size = 0,
    bool with_ptr = false);

std::string NVF_API toString(
    const ValGraph& id_graph,
    const std::vector<ExprGroup>& expr_groups,
    int indent_size = 0,
    bool with_ptr = false);
std::string NVF_API toString(
    const ValGraph& id_graph,
    const ExprGroups& expr_groups,
    int indent_size = 0,
    bool with_ptr = false);

std::string idGroupsString(
    const ValGraph& id_graph,
    int indent_size = 0,
    bool with_ptr = false);
std::string exprGroupsString(
    const ValGraph& id_graph,
    int indent_size = 0,
    bool with_ptr = false);
std::string definitionsString(
    const ValGraph& id_graph,
    int indent_size = 0,
    bool with_ptr = false);
std::string usesString(
    const ValGraph& id_graph,
    int indent_size = 0,
    bool with_ptr = false);

} // namespace nvfuser
