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
# Reporting
# --------------------------

if(NOT WIN32 OR MSVC)
  string(ASCII 27 Esc)
  set(ColorReset  "${Esc}[m")
  set(ColorBold   "${Esc}[1m")
  set(ColorRed    "${Esc}[31m")
  set(ColorGreen  "${Esc}[32m")
  set(ColorYellow "${Esc}[33m")
  set(ColorBlue   "${Esc}[34m")
  set(ColorMagenta "${Esc}[35m")
  set(ColorCyan   "${Esc}[36m")
  set(ColorWhite  "${Esc}[37m")
  set(BoldRed     "${Esc}[1;31m")
  set(BoldGreen   "${Esc}[1;32m")
  set(BoldYellow  "${Esc}[1;33m")
  set(BoldBlue    "${Esc}[1;34m")
  set(BoldWhite   "${Esc}[1;37m")
endif()

function(message_colored color text)
  message(STATUS "${color}${text}${ColorReset}")
endfunction()

# --------------------------
# Helper Functions for Formatting
# --------------------------

# Calculate padding for name alignment (12 chars)
function(format_name_padding name out_var)
  string(LENGTH "${name}" name_len)
  math(EXPR pad_len "12 - ${name_len}")
  string(REPEAT " " ${pad_len} padding)
  set(${out_var} "${padding}" PARENT_SCOPE)
endfunction()

# Calculate padding for version alignment (10 chars)
function(format_version_padding version out_var)
  string(LENGTH "${version}" ver_len)
  math(EXPR ver_pad_len "10 - ${ver_len}")
  if(ver_pad_len LESS 0)
    set(ver_pad_len 0)
  endif()
  string(REPEAT " " ${ver_pad_len} padding)
  set(${out_var} "${padding}" PARENT_SCOPE)
endfunction()

# Format status badge with color
function(format_status_badge status out_var)
  if(status STREQUAL "OK")
    set(${out_var} "${ColorGreen}[ OK ]${ColorReset}" PARENT_SCOPE)
  elseif(status STREQUAL "FAIL")
    set(${out_var} "${BoldRed}[FAIL]${ColorReset}" PARENT_SCOPE)
  elseif(status STREQUAL "SKIP")
    set(${out_var} "${ColorYellow}[ -- ]${ColorReset}" PARENT_SCOPE)
  endif()
endfunction()

# Compare version and return display string + status
function(format_version_comparison version min_version is_constraint out_display out_ok)
  format_version_padding("${version}" ver_padding)

  if(is_constraint)
    # Constraint: exact match, use = symbol
    set(display "${ColorGreen}v${version}${ver_padding}${ColorReset}  (= ${min_version})")
    set(ok TRUE)
  elseif(DEFINED min_version AND NOT "${min_version}" STREQUAL "")
    # Regular version check: use ≥ or < symbol
    if("${version}" VERSION_GREATER_EQUAL "${min_version}")
      set(display "${ColorGreen}v${version}${ver_padding}${ColorReset}  (≥ ${min_version})")
      set(ok TRUE)
    else()
      set(display "${BoldRed}v${version}${ver_padding}${ColorReset}  (< ${min_version})")
      set(ok FALSE)
    endif()
  else()
    # No version requirement
    set(display "${ColorGreen}v${version}${ver_padding}${ColorReset}")
    set(ok TRUE)
  endif()

  set(${out_display} "${display}" PARENT_SCOPE)
  set(${out_ok} "${ok}" PARENT_SCOPE)
endfunction()

# --------------------------
# Report Functions for Different Scenarios
# --------------------------

# Handle case: dependency found
function(report_found name location is_constraint)
  # Get metadata
  set(min_version "${NVFUSER_REQUIREMENT_${name}_VERSION_MIN}")
  set(version "${${name}_VERSION}")

  # Format components
  format_name_padding("${name}" name_padding)
  format_version_comparison("${version}" "${min_version}" "${is_constraint}" version_display version_ok)

  # Determine status and location display
  if(is_constraint)
    set(location_display "${ColorCyan}Torch.CUDA == CUDAToolkit${ColorReset}")
  else()
    set(location_display "${ColorCyan}${location}${ColorReset}")
  endif()

  # Display message with appropriate status
  if(version_ok)
    format_status_badge("OK" status_badge)
  else()
    format_status_badge("FAIL" status_badge)
    # Track failure
    list(APPEND _DEPENDENCY_FAILURES "${name}: found v${version}, but requires v${min_version} or higher")
    set(_DEPENDENCY_FAILURES "${_DEPENDENCY_FAILURES}" PARENT_SCOPE)
  endif()

  message(STATUS "${status_badge} ${ColorWhite}${name}${name_padding}${ColorReset}  ${version_display}  ${location_display}")
endfunction()

# Handle case: dependency not found + optional
function(report_missing_optional name)
  set(min_version "${NVFUSER_REQUIREMENT_${name}_VERSION_MIN}")
  format_status_badge("SKIP" status_badge)

  if(DEFINED min_version AND NOT "${min_version}" STREQUAL "")
    message(STATUS "${status_badge} ${ColorWhite}${name}${ColorReset} NOT found (optional, v${min_version}+ recommended)")
  else()
    message(STATUS "${status_badge} ${ColorWhite}${name}${ColorReset} NOT found (optional)")
  endif()
endfunction()

# Handle case: dependency not found + required
function(report_missing_required name is_constraint)
  set(min_version "${NVFUSER_REQUIREMENT_${name}_VERSION_MIN}")
  set(version "${${name}_VERSION}")
  format_status_badge("FAIL" status_badge)

  if(is_constraint)
    # Constraint failure
    if(DEFINED version AND DEFINED min_version)
      message(STATUS "${status_badge} ${ColorWhite}${name}${ColorReset} v${version} != ${min_version} (Torch.CUDA != CUDAToolkit)")
      list(APPEND _DEPENDENCY_FAILURES "${name}: Torch built with CUDA v${version}, but CUDAToolkit is v${min_version}")
    else()
      message(STATUS "${status_badge} ${ColorWhite}${name}${ColorReset} constraint check failed")
      list(APPEND _DEPENDENCY_FAILURES "${name}: constraint validation failed")
    endif()
  else()
    # Regular dependency not found
    if(DEFINED min_version AND NOT "${min_version}" STREQUAL "")
      message(STATUS "${status_badge} ${ColorWhite}${name}${ColorReset} NOT found (requires v${min_version} or higher)")
      list(APPEND _DEPENDENCY_FAILURES "${name}: not found (requires v${min_version} or higher)")
    else()
      message(STATUS "${status_badge} ${ColorWhite}${name}${ColorReset} NOT found")
      list(APPEND _DEPENDENCY_FAILURES "${name}: not found")
    endif()
  endif()

  set(_DEPENDENCY_FAILURES "${_DEPENDENCY_FAILURES}" PARENT_SCOPE)
endfunction()

# --------------------------
# Main Report Dispatcher
# --------------------------

macro(report name location is_optional)
  # Check if this is a constraint (special reporting logic)
  set(is_constraint "${NVFUSER_REQUIREMENT_${name}_IS_CONSTRAINT}")

  if(${name}_FOUND)
    report_found("${name}" "${location}" "${is_constraint}")
    # Propagate failures list from function back to macro scope
    set(_DEPENDENCY_FAILURES "${_DEPENDENCY_FAILURES}")
  elseif(is_optional)
    report_missing_optional("${name}")
  else()
    report_missing_required("${name}" "${is_constraint}")
    # Propagate failures list from function back to macro scope
    set(_DEPENDENCY_FAILURES "${_DEPENDENCY_FAILURES}")
  endif()
endmacro()

macro(report_dependencies)
  # Initialize failure tracking
  set(_DEPENDENCY_FAILURES "")

  message("")
  message_colored("${BoldBlue}"   "///////////////////////////////////////////")
  message_colored("${BoldWhite}"  "===========================================")
  message_colored("${BoldGreen}"  "[nvFuser] Validating build prerequisites...")
  message_colored("${BoldWhite}"  "===========================================")

  # Iterate through requirements in order
  foreach(dep_name ${NVFUSER_ALL_REQUIREMENTS})
    set(optional "${NVFUSER_REQUIREMENT_${dep_name}_OPTIONAL}")
    set(location_var "${NVFUSER_REQUIREMENT_${dep_name}_LOCATION_VAR}")

    # Get location using the metadata-specified variable
    if(location_var)
      set(location "${${location_var}}")
    else()
      set(location "")
    endif()

    # Call report with optional flag
    report(${dep_name} "${location}" ${optional})
  endforeach()

  message_colored("${BoldWhite}"  "===========================================")
  message_colored("${BoldBlue}"   "///////////////////////////////////////////")

  # If there were any failures, show them and error out
  list(LENGTH _DEPENDENCY_FAILURES failure_count)
  if(failure_count GREATER 0)
    message("")
    message_colored("${BoldRed}" "Configuration failed due to missing or incompatible dependencies:")
    foreach(failure ${_DEPENDENCY_FAILURES})
      message_colored("${BoldRed}" "  - ${failure}")

      # Print install help if available
      string(REGEX MATCH "^([^:]+):" match "${failure}")
      if(CMAKE_MATCH_1)
        set(failed_dep "${CMAKE_MATCH_1}")
        set(help_text "${NVFUSER_REQUIREMENT_${failed_dep}_INSTALL_HELP}")
        if(help_text AND NOT "${help_text}" STREQUAL "")
          message_colored("${ColorYellow}" "    ${help_text}")
        endif()
      endif()
    endforeach()
    message("")
    message(FATAL_ERROR "Please install or upgrade the required dependencies listed above.")
  endif()
endmacro()


