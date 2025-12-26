# SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

# ==============================================================================
# nvFuser Dependency Utilities
# ==============================================================================
#
# This file provides utilities for finding and reporting on nvFuser dependencies.
# Dependency metadata is defined in DependencyRequirements.cmake
# Validation logic is defined in DependencyValidators.cmake
#
# ==============================================================================

# Include requirement definitions and validators
include(cmake/DependencyRequirements.cmake)
include(cmake/DependencyValidators.cmake)

# --------------------------
# Find all dependencies
# --------------------------
macro(find_nvfuser_dependencies)
  # Initialize success flag - will be set to FALSE if any required dependency fails
  set(NVFUSER_DEPENDENCIES_OK TRUE)

  # Iterate through all requirements and validate each
  foreach(dep_name ${NVFUSER_ALL_REQUIREMENTS})
    validate_dependency(NAME ${dep_name})
  endforeach()
endmacro()

# --------------------------
# Status Tracking for JSON Export
# --------------------------

# Set dependency status based on found state and version check
function(set_dependency_status name)
  set(is_constraint "${NVFUSER_REQUIREMENT_${name}_IS_CONSTRAINT}")

  if(is_constraint)
    # Status already set by post-find hook
    return()
  endif()

  set(optional "${NVFUSER_REQUIREMENT_${name}_OPTIONAL}")

  if(${name}_FOUND)
    # Check version compatibility
    set(min_version "${NVFUSER_REQUIREMENT_${name}_VERSION_MIN}")
    set(version "${${name}_VERSION}")

    if(DEFINED min_version AND NOT "${min_version}" STREQUAL "")
      if("${version}" VERSION_GREATER_EQUAL "${min_version}")
        set(${name}_STATUS "SUCCESS" PARENT_SCOPE)
      else()
        set(${name}_STATUS "INCOMPATIBLE" PARENT_SCOPE)
        # Mark dependencies as failed if this is a required dependency
        if(NOT optional)
          set(NVFUSER_DEPENDENCIES_OK FALSE PARENT_SCOPE)
        endif()
      endif()
    else()
      set(${name}_STATUS "SUCCESS" PARENT_SCOPE)
    endif()
  else()
    set(${name}_STATUS "NOT_FOUND" PARENT_SCOPE)
    # Mark dependencies as failed if this is a required dependency
    if(NOT optional)
      set(NVFUSER_DEPENDENCIES_OK FALSE PARENT_SCOPE)
    endif()
  endif()
endfunction()

# --------------------------
# JSON Export for Python Reporting
# --------------------------

function(export_dependency_json output_file)
  # Get all CMake variables to filter for dependency-related ones
  get_cmake_property(all_vars VARIABLES)

  # Build JSON structure
  set(json_content "{\n")
  string(APPEND json_content "  \"dependencies\": [\n")

  # Count totals for summary
  set(total 0)
  set(success_count 0)
  set(not_found_count 0)
  set(incompatible_count 0)

  # Iterate through all dependencies
  set(first TRUE)
  foreach(dep_name ${NVFUSER_ALL_REQUIREMENTS})
    # Skip constraints - they're reported but not exported separately
    set(is_constraint "${NVFUSER_REQUIREMENT_${dep_name}_IS_CONSTRAINT}")
    if(is_constraint)
      continue()
    endif()

    # Add comma separator after first entry
    if(NOT first)
      string(APPEND json_content ",\n")
    endif()
    set(first FALSE)

    # Set status if not already set (e.g., by post-find hook)
    if(NOT DEFINED ${dep_name}_STATUS)
      set_dependency_status(${dep_name})
    endif()

    # Get basic metadata
    set(dep_type "${NVFUSER_REQUIREMENT_${dep_name}_TYPE}")
    if(NOT dep_type)
      set(dep_type "find_package")
    endif()
    set(status "${${dep_name}_STATUS}")

    # Determine export name (Compiler -> GCC/Clang)
    set(export_name "${dep_name}")
    if(dep_name STREQUAL "Compiler" AND DEFINED Compiler_NAME)
      set(export_name "${Compiler_NAME}")
    endif()

    # Update counts
    math(EXPR total "${total} + 1")
    if(status STREQUAL "SUCCESS")
      math(EXPR success_count "${success_count} + 1")
    elseif(status STREQUAL "NOT_FOUND")
      math(EXPR not_found_count "${not_found_count} + 1")
    elseif(status STREQUAL "INCOMPATIBLE")
      math(EXPR incompatible_count "${incompatible_count} + 1")
    endif()

    # Build JSON entry header
    string(APPEND json_content "    {\n")
    string(APPEND json_content "      \"name\": \"${export_name}\",\n")
    string(APPEND json_content "      \"type\": \"${dep_type}\",\n")

    # --------------------------
    # Export CMake Variables
    # --------------------------
    string(APPEND json_content "      \"cmake_vars\": {\n")

    # Filter all CMake variables that start with this dependency name
    set(dep_vars)
    foreach(var ${all_vars})
      if(var MATCHES "^${dep_name}_")
        list(APPEND dep_vars ${var})
      endif()
    endforeach()

    # Sort for consistent output
    if(dep_vars)
      list(SORT dep_vars)
    endif()

    # Export each variable
    set(var_first TRUE)
    foreach(var ${dep_vars})
      if(NOT var_first)
        string(APPEND json_content ",\n")
      endif()
      set(var_first FALSE)

      # Get variable value
      set(value "${${var}}")

      # Escape special characters for JSON
      string(REPLACE "\\" "\\\\" value "${value}")
      string(REPLACE "\"" "\\\"" value "${value}")
      string(REPLACE "\n" "\\n" value "${value}")
      string(REPLACE "\r" "\\r" value "${value}")
      string(REPLACE "\t" "\\t" value "${value}")

      # Determine if value is boolean or string
      set(is_bool FALSE)
      if("${value}" STREQUAL "TRUE" OR "${value}" STREQUAL "ON" OR "${value}" STREQUAL "YES" OR "${value}" STREQUAL "1")
        set(value "true")
        set(is_bool TRUE)
      elseif("${value}" STREQUAL "FALSE" OR "${value}" STREQUAL "OFF" OR "${value}" STREQUAL "NO" OR "${value}" STREQUAL "0")
        set(value "false")
        set(is_bool TRUE)
      elseif("${value}" STREQUAL "")
        set(value "null")
        set(is_bool TRUE)
      endif()

      # Write JSON field
      if(is_bool)
        string(APPEND json_content "        \"${var}\": ${value}")
      else()
        string(APPEND json_content "        \"${var}\": \"${value}\"")
      endif()
    endforeach()

    string(APPEND json_content "\n      },\n")

    # --------------------------
    # Export Metadata (Requirements Config)
    # --------------------------
    string(APPEND json_content "      \"metadata\": {\n")

    # Collect all NVFUSER_REQUIREMENT_${dep_name}_* variables
    set(metadata_vars)
    set(metadata_prefix "NVFUSER_REQUIREMENT_${dep_name}_")
    foreach(var ${all_vars})
      if(var MATCHES "^${metadata_prefix}")
        list(APPEND metadata_vars ${var})
      endif()
    endforeach()

    # Sort for consistent output
    if(metadata_vars)
      list(SORT metadata_vars)
    endif()

    # Export each metadata variable
    set(meta_first TRUE)
    foreach(var ${metadata_vars})
      if(NOT meta_first)
        string(APPEND json_content ",\n")
      endif()
      set(meta_first FALSE)

      # Get variable value
      set(value "${${var}}")

      # Escape special characters for JSON
      string(REPLACE "\\" "\\\\" value "${value}")
      string(REPLACE "\"" "\\\"" value "${value}")
      string(REPLACE "\n" "\\n" value "${value}")
      string(REPLACE "\r" "\\r" value "${value}")
      string(REPLACE "\t" "\\t" value "${value}")

      # Determine if value is boolean or string
      set(is_bool FALSE)
      if("${value}" STREQUAL "TRUE" OR "${value}" STREQUAL "ON" OR "${value}" STREQUAL "YES" OR "${value}" STREQUAL "1")
        set(value "true")
        set(is_bool TRUE)
      elseif("${value}" STREQUAL "FALSE" OR "${value}" STREQUAL "OFF" OR "${value}" STREQUAL "NO" OR "${value}" STREQUAL "0")
        set(value "false")
        set(is_bool TRUE)
      elseif("${value}" STREQUAL "")
        set(value "null")
        set(is_bool TRUE)
      endif()

      # Write JSON field
      if(is_bool)
        string(APPEND json_content "        \"${var}\": ${value}")
      else()
        string(APPEND json_content "        \"${var}\": \"${value}\"")
      endif()
    endforeach()

    string(APPEND json_content "\n      }\n")
    string(APPEND json_content "    }")
  endforeach()

  # Close dependencies array and add summary
  string(APPEND json_content "\n  ],\n")
  string(APPEND json_content "  \"summary\": {\n")
  string(APPEND json_content "    \"total\": ${total},\n")
  string(APPEND json_content "    \"success\": ${success_count},\n")
  string(APPEND json_content "    \"not_found\": ${not_found_count},\n")
  string(APPEND json_content "    \"incompatible\": ${incompatible_count}\n")
  string(APPEND json_content "  }\n")
  string(APPEND json_content "}\n")

  # Write to file
  file(WRITE "${output_file}" "${json_content}")
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


