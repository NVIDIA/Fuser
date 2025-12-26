# SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

# ==============================================================================
# nvFuser Dependency Validators
# ==============================================================================
#
# This file includes individual validator modules for each dependency.
# Each validator is in its own file under cmake/validators/
#
# Validator types:
# - Pre-find hooks: Execute before find_package (e.g., Torch needs Python setup)
# - Post-find hooks: Execute after successful find_package (e.g., LLVM needs library mapping)
# - Standalone validators: For dependencies that don't use find_package
#
# ==============================================================================

# Include individual validator modules
include(cmake/validators/Torch.cmake)
include(cmake/validators/LLVM.cmake)
include(cmake/validators/Compiler.cmake)
include(cmake/validators/Ninja.cmake)
include(cmake/validators/GitSubmodules.cmake)

macro(validate_dependency)
  # Parse named arguments
  cmake_parse_arguments(
    ARG                    # prefix for parsed variables
    ""                     # options (boolean flags)
    "NAME"                 # one-value keywords
    ""                     # multi-value keywords
    ${ARGN}
  )

  if(NOT ARG_NAME)
    _message(FATAL_ERROR "validate_dependency: NAME argument required")
  endif()

  # Check if this is a constraint (pseudo-dependency for reporting only)
  set(is_constraint "${NVFUSER_REQUIREMENT_${ARG_NAME}_IS_CONSTRAINT}")
  set(dep_type "${NVFUSER_REQUIREMENT_${ARG_NAME}_TYPE}")

  if(NOT is_constraint)
    # Only validate if NOT a constraint

    message("")
    message("Finding ${ARG_NAME}...")

    # Check for special dependency types that don't use find_package
    if(dep_type STREQUAL "compiler")
      validate_compiler()
    elseif(dep_type STREQUAL "build_tool")
      validate_ninja()
    elseif(dep_type STREQUAL "git")
      validate_git_submodules()
    else()
      # Standard find_package dependency
      # Get metadata from requirements
      set(version_min "${NVFUSER_REQUIREMENT_${ARG_NAME}_VERSION_MIN}")
      set(components_list "${NVFUSER_REQUIREMENT_${ARG_NAME}_COMPONENTS}")
      set(pre_find_hook "${NVFUSER_REQUIREMENT_${ARG_NAME}_PRE_FIND_HOOK}")
      set(post_find_hook "${NVFUSER_REQUIREMENT_${ARG_NAME}_POST_FIND_HOOK}")

      # Execute pre-find hook if specified
      if(pre_find_hook)
        cmake_language(CALL ${pre_find_hook})
      endif()

      # Only proceed with find_package if hook didn't abort
      if(NOT DEFINED ${ARG_NAME}_FOUND OR ${ARG_NAME}_FOUND)
        # Standard find_package call (without REQUIRED to allow reporting all failures)
        if(components_list)
          # Use cmake_language(EVAL) to properly expand the components list
          cmake_language(EVAL CODE "find_package(${ARG_NAME} COMPONENTS ${components_list})")
        else()
          find_package(${ARG_NAME})
        endif()

        # Execute post-find hook if specified and package was found
        if(post_find_hook AND ${ARG_NAME}_FOUND)
          cmake_language(CALL ${post_find_hook})
        endif()
      endif()

      # Set status based on found state and version check
      # This will be used for JSON export
      # Note: Must call as function to get proper version checking
      set_dependency_status(${ARG_NAME})
    endif()
  endif()
endmacro()
