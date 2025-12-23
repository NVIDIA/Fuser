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

macro(report name location is_optional)
  # Check if this is a constraint (special reporting logic)
  set(is_constraint "${NVFUSER_REQUIREMENT_${name}_IS_CONSTRAINT}")

  if (${name}_FOUND)
    # Pad the name to 12 characters for alignment
    string(LENGTH "${name}" name_len)
    math(EXPR pad_len "12 - ${name_len}")
    string(REPEAT " " ${pad_len} name_padding)

    # Get minimum required version
    set(min_version "${NVFUSER_REQUIREMENT_${name}_VERSION_MIN}")

    # Pad version number to 10 characters for alignment
    string(LENGTH "${${name}_VERSION}" ver_len)
    math(EXPR ver_pad_len "10 - ${ver_len}")
    if(ver_pad_len LESS 0)
      set(ver_pad_len 0)
    endif()
    string(REPEAT " " ${ver_pad_len} ver_padding)

    # Determine comparison symbol and color based on constraint type
    if(is_constraint)
      # Constraint check uses = symbol (exact match required)
      set(version_color "${ColorGreen}")
      set(comparison_symbol "=")
      set(version_ok TRUE)
      # For constraints, show special message
      set(version_display "${version_color}v${${name}_VERSION}${ver_padding}${ColorReset}  (${comparison_symbol} ${min_version})")
    elseif(DEFINED min_version AND NOT "${min_version}" STREQUAL "")
      # Regular version check uses ≥ symbol
      if("${${name}_VERSION}" VERSION_GREATER_EQUAL "${min_version}")
        set(version_color "${ColorGreen}")
        set(comparison_symbol "≥")
        set(version_ok TRUE)
      else()
        set(version_color "${BoldRed}")
        set(comparison_symbol "<")
        set(version_ok FALSE)
        # Track this as a failure with found vs required info
        list(APPEND _DEPENDENCY_FAILURES "${name}: found v${${name}_VERSION}, but requires v${min_version} or higher")
      endif()
      set(version_display "${version_color}v${${name}_VERSION}${ver_padding}${ColorReset}  (${comparison_symbol} ${min_version})")
    else()
      # No minimum version specified
      set(version_display "${ColorGreen}v${${name}_VERSION}${ver_padding}${ColorReset}")
      set(version_ok TRUE)
    endif()

    # Display message with location or constraint note
    if(is_constraint)
      # For constraints like Torch_CUDA, show validation note instead of path
      message(STATUS "${ColorGreen}[ OK ]${ColorReset} ${ColorWhite}${name}${name_padding}${ColorReset}  ${version_display}  ${ColorCyan}Torch.CUDA == CUDAToolkit${ColorReset}")
    elseif(version_ok)
      message(STATUS "${ColorGreen}[ OK ]${ColorReset} ${ColorWhite}${name}${name_padding}${ColorReset}  ${version_display}  ${ColorCyan}${location}${ColorReset}")
    else()
      message(STATUS "${BoldRed}[FAIL]${ColorReset} ${ColorWhite}${name}${name_padding}${ColorReset}  ${version_display}  ${ColorCyan}${location}${ColorReset}")
    endif()
  else()
    # Dependency not found at all
    set(min_version "${NVFUSER_REQUIREMENT_${name}_VERSION_MIN}")

    if(is_optional)
      # Yellow warning for optional deps
      if(DEFINED min_version AND NOT "${min_version}" STREQUAL "")
        message(STATUS "${ColorYellow}[ -- ]${ColorReset} ${ColorWhite}${name}${ColorReset} NOT found (optional, v${min_version}+ recommended)")
      else()
        message(STATUS "${ColorYellow}[ -- ]${ColorReset} ${ColorWhite}${name}${ColorReset} NOT found (optional)")
      endif()
      # Don't add to failure list for optional deps
    else()
      # Handle constraint failures specially
      if(is_constraint)
        # For constraints, show comparison failure
        if(DEFINED ${name}_VERSION AND DEFINED min_version)
          message(STATUS "${BoldRed}[FAIL]${ColorReset} ${ColorWhite}${name}${ColorReset} v${${name}_VERSION} != ${min_version} (Torch.CUDA != CUDAToolkit)")
          list(APPEND _DEPENDENCY_FAILURES "${name}: Torch built with CUDA v${${name}_VERSION}, but CUDAToolkit is v${min_version}")
        else()
          message(STATUS "${BoldRed}[FAIL]${ColorReset} ${ColorWhite}${name}${ColorReset} constraint check failed")
          list(APPEND _DEPENDENCY_FAILURES "${name}: constraint validation failed")
        endif()
      else()
        # Red fail for required deps
        if(DEFINED min_version AND NOT "${min_version}" STREQUAL "")
          message(STATUS "${BoldRed}[FAIL]${ColorReset} ${ColorWhite}${name}${ColorReset} NOT found (requires v${min_version} or higher)")
          list(APPEND _DEPENDENCY_FAILURES "${name}: not found (requires v${min_version} or higher)")
        else()
          message(STATUS "${BoldRed}[FAIL]${ColorReset} ${ColorWhite}${name}${ColorReset} NOT found")
          list(APPEND _DEPENDENCY_FAILURES "${name}: not found")
        endif()
      endif()
    endif()
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


