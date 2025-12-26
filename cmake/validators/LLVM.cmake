# SPDX-FileCopyrightText: Copyright (c) 2025-present NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

# ------------------------------------------------------------------------------
# LLVM Validator
# ------------------------------------------------------------------------------

# Post-find hook for LLVM
# Maps LLVM components to library names for linking
function(llvm_post_find_hook)
  llvm_map_components_to_libnames(LLVM_LIBS
    support
    core
    orcjit
    executionengine
    irreader
    nativecodegen
    Target
    Analysis
    JITLink
    Demangle
  )
  set(LLVM_LIBS "${LLVM_LIBS}" PARENT_SCOPE)
endfunction()
