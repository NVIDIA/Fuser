# SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

# ==============================================================================
# nvFuser Dependency Utilities
# ==============================================================================
#
# This file provides utilities for dependency status tracking and reporting.
# Dependency metadata is defined in DependencyRequirements.cmake
# Individual dependency handlers are in cmake/deps/
#
# ==============================================================================

# Include requirement definitions
include(cmake/DependencyRequirements.cmake)

# --------------------------
# Conditionally add REQUIRED to find_package calls
# --------------------------

# When dependency report is enabled, we don't use REQUIRED so we can collect
# all failures and show a comprehensive Python report.
# When disabled, we use REQUIRED for immediate failure with standard CMake errors.
if(NVFUSER_ENABLE_DEPENDENCY_REPORT)
  set(MAYBE_REQUIRED "")
else()
  set(MAYBE_REQUIRED "REQUIRED")
endif()

# --------------------------
# Status Tracking for JSON Export
# --------------------------

# Set dependency status based on found state and version check
macro(set_dependency_report_status name)
  set(optional "${NVFUSER_REQUIREMENT_${name}_OPTIONAL}")

  if(${name}_FOUND)
    # Check version compatibility
    set(min_version "${NVFUSER_REQUIREMENT_${name}_VERSION_MIN}")
    set(version "${${name}_VERSION}")

    if(DEFINED min_version AND NOT "${min_version}" STREQUAL "")
      if("${version}" VERSION_GREATER_EQUAL "${min_version}")
        set(NVFUSER_REQUIREMENT_${name}_STATUS "SUCCESS")
      else()
        set(NVFUSER_REQUIREMENT_${name}_STATUS "INCOMPATIBLE")
        # Mark dependencies as failed if this is a required dependency
        if(NOT optional)
          set(NVFUSER_DEPENDENCIES_OK FALSE)
        endif()
      endif()
    else()
      set(NVFUSER_REQUIREMENT_${name}_STATUS "SUCCESS")
    endif()
  else()
    set(NVFUSER_REQUIREMENT_${name}_STATUS "NOT_FOUND")
    # Mark dependencies as failed if this is a required dependency
    if(NOT optional)
      set(NVFUSER_DEPENDENCIES_OK FALSE)
    endif()
  endif()
endmacro()

# --------------------------
# Python Export for Dependency Reporting
# --------------------------

function(export_dependency_json output_file)
  # Get all CMake variables
  get_cmake_property(all_vars VARIABLES)

  # Write JSON file with flat variable dict
  file(WRITE "${output_file}" "{\n")
  file(APPEND "${output_file}" "  \"cmake_vars\": {\n")

  # Export all variables (sorted for consistency)
  list(SORT all_vars)
  list(LENGTH all_vars var_count)
  set(var_index 0)
  foreach(var ${all_vars})
    set(value "${${var}}")
    # Escape for JSON strings
    string(REPLACE "\\" "\\\\" value "${value}")
    string(REPLACE "\"" "\\\"" value "${value}")
    string(REPLACE "\n" "\\n" value "${value}")
    string(REPLACE "\t" "\\t" value "${value}")
    string(REPLACE "\r" "\\r" value "${value}")

    # Add comma if not last item
    math(EXPR var_index "${var_index} + 1")
    if(var_index LESS var_count)
      file(APPEND "${output_file}" "    \"${var}\": \"${value}\",\n")
    else()
      file(APPEND "${output_file}" "    \"${var}\": \"${value}\"\n")
    endif()
  endforeach()

  file(APPEND "${output_file}" "  }\n")
  file(APPEND "${output_file}" "}\n")
endfunction()

# --------------------------
# Report Dependencies (Python-based with fallback)
# --------------------------

macro(report_dependencies)
  # Export dependency data to JSON
  export_dependency_json("${CMAKE_BINARY_DIR}/nvfuser_dependencies.json")

  # Try to use Python script for enhanced reporting
  set(python_script "${CMAKE_SOURCE_DIR}/python/tools/check_dependencies.py")

  if(NOT EXISTS "${python_script}")
    message(WARNING "Python reporting script not found: ${python_script}")
  elseif(NOT DEFINED Python_EXECUTABLE OR NOT Python_FOUND)
    message(WARNING "Python is not available - skipping enhanced dependency report")
  else()
    # Run Python reporting script
    execute_process(
      COMMAND "${Python_EXECUTABLE}" "${python_script}" "${CMAKE_BINARY_DIR}/nvfuser_dependencies.json"
      RESULT_VARIABLE python_result
      OUTPUT_VARIABLE python_output
      ERROR_VARIABLE python_error
    )

    if(NOT python_result EQUAL 0)
      message(WARNING "Python reporting failed (exit code ${python_result}): ${python_error}")
    else()
      # Display Python output
      message("${python_output}")
    endif()
  endif()

endmacro()


