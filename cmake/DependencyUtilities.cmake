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

  if(${name}_FOUND)
    # Check version compatibility
    set(min_version "${NVFUSER_REQUIREMENT_${name}_VERSION_MIN}")
    set(version "${${name}_VERSION}")

    if(DEFINED min_version AND NOT "${min_version}" STREQUAL "")
      if("${version}" VERSION_GREATER_EQUAL "${min_version}")
        set(${name}_STATUS "SUCCESS" PARENT_SCOPE)
      else()
        set(${name}_STATUS "INCOMPATIBLE" PARENT_SCOPE)
      endif()
    else()
      set(${name}_STATUS "SUCCESS" PARENT_SCOPE)
    endif()
  else()
    set(${name}_STATUS "NOT_FOUND" PARENT_SCOPE)
  endif()
endfunction()

# --------------------------
# JSON Export for Python Reporting
# --------------------------

function(export_dependency_json output_file)
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

    # Get metadata
    set(dep_type "${NVFUSER_REQUIREMENT_${dep_name}_TYPE}")
    if(NOT dep_type)
      set(dep_type "find_package")
    endif()
    set(found "${${dep_name}_FOUND}")
    set(status "${${dep_name}_STATUS}")
    set(version "${${dep_name}_VERSION}")
    set(version_min "${NVFUSER_REQUIREMENT_${dep_name}_VERSION_MIN}")
    set(location_var "${NVFUSER_REQUIREMENT_${dep_name}_LOCATION_VAR}")
    if(location_var)
      set(location "${${location_var}}")
    else()
      set(location "")
    endif()
    set(optional "${NVFUSER_REQUIREMENT_${dep_name}_OPTIONAL}")

    # Update counts
    math(EXPR total "${total} + 1")
    if(status STREQUAL "SUCCESS")
      math(EXPR success_count "${success_count} + 1")
    elseif(status STREQUAL "NOT_FOUND")
      math(EXPR not_found_count "${not_found_count} + 1")
    elseif(status STREQUAL "INCOMPATIBLE")
      math(EXPR incompatible_count "${incompatible_count} + 1")
    endif()

    # Escape special characters in strings for JSON
    string(REPLACE "\\" "\\\\" location "${location}")
    string(REPLACE "\"" "\\\"" location "${location}")

    # Build JSON entry
    string(APPEND json_content "    {\n")
    string(APPEND json_content "      \"name\": \"${dep_name}\",\n")
    string(APPEND json_content "      \"type\": \"${dep_type}\",\n")
    if(found)
      string(APPEND json_content "      \"found\": true,\n")
    else()
      string(APPEND json_content "      \"found\": false,\n")
    endif()
    string(APPEND json_content "      \"status\": \"${status}\",\n")
    if(version)
      string(APPEND json_content "      \"version_found\": \"${version}\",\n")
    else()
      string(APPEND json_content "      \"version_found\": null,\n")
    endif()
    if(version_min)
      string(APPEND json_content "      \"version_required\": \"${version_min}\",\n")
    else()
      string(APPEND json_content "      \"version_required\": null,\n")
    endif()
    if(location)
      string(APPEND json_content "      \"location\": \"${location}\",\n")
    else()
      string(APPEND json_content "      \"location\": null,\n")
    endif()
    if(optional)
      string(APPEND json_content "      \"optional\": true")
    else()
      string(APPEND json_content "      \"optional\": false")
    endif()

    # Check if dependency has extra JSON data to include (e.g., constraints)
    set(extra_json_var "${dep_name}_EXTRA_JSON")
    if(DEFINED ${extra_json_var} AND NOT "${${extra_json_var}}" STREQUAL "")
      # Parse and merge the extra JSON
      string(APPEND json_content ",\n")
      string(APPEND json_content "      \"extra\": ${${extra_json_var}}\n")
    else()
      string(APPEND json_content "\n")
    endif()

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

  # Check for dependency failures based on JSON data
  # Iterate through dependencies and collect failures
  set(_DEPENDENCY_FAILURES "")
  foreach(dep_name ${NVFUSER_ALL_REQUIREMENTS})
    # Skip constraints - they're handled as part of their parent dependency
    set(is_constraint "${NVFUSER_REQUIREMENT_${dep_name}_IS_CONSTRAINT}")
    if(is_constraint)
      continue()
    endif()

    set(status "${${dep_name}_STATUS}")
    set(optional "${NVFUSER_REQUIREMENT_${dep_name}_OPTIONAL}")

    # Track failures for required dependencies
    if(NOT optional)
      if(status STREQUAL "NOT_FOUND")
        set(min_version "${NVFUSER_REQUIREMENT_${dep_name}_VERSION_MIN}")
        if(DEFINED min_version AND NOT "${min_version}" STREQUAL "")
          list(APPEND _DEPENDENCY_FAILURES "${dep_name}: not found (requires v${min_version} or higher)")
        else()
          list(APPEND _DEPENDENCY_FAILURES "${dep_name}: not found")
        endif()
      elseif(status STREQUAL "INCOMPATIBLE")
        set(version "${${dep_name}_VERSION}")
        set(min_version "${NVFUSER_REQUIREMENT_${dep_name}_VERSION_MIN}")
        list(APPEND _DEPENDENCY_FAILURES "${dep_name}: found v${version}, but requires v${min_version} or higher")
      endif()
    endif()
  endforeach()

  # Also check constraint failures (e.g., Torch_CUDA)
  foreach(dep_name ${NVFUSER_ALL_REQUIREMENTS})
    set(is_constraint "${NVFUSER_REQUIREMENT_${dep_name}_IS_CONSTRAINT}")
    if(is_constraint AND NOT ${dep_name}_FOUND)
      set(version "${${dep_name}_VERSION}")
      set(min_version "${NVFUSER_REQUIREMENT_${dep_name}_VERSION_MIN}")
      if(DEFINED version AND DEFINED min_version)
        list(APPEND _DEPENDENCY_FAILURES "${dep_name}: Torch built with CUDA v${version}, but CUDAToolkit is v${min_version}")
      else()
        list(APPEND _DEPENDENCY_FAILURES "${dep_name}: constraint validation failed")
      endif()
    endif()
  endforeach()

  # If there were any failures, error out
  list(LENGTH _DEPENDENCY_FAILURES failure_count)
  if(failure_count GREATER 0)
    message(FATAL_ERROR "Please install or upgrade the required dependencies listed above.")
  endif()
endmacro()


