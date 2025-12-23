set(NVFUSER_CMake_VERSION_REQUIRED       "3.8")
set(NVFUSER_Ninja_VERSION_REQUIRED       "ANY")

set(NVFUSER_GCC_VERSION_REQUIRED       "0.0")
set(NVFUSER_Clang_VERSION_REQUIRED       "0.0")

set(NVFUSER_CUDAToolkit_VERSION_REQUIRED  "12.6")

set(NVFUSER_Python_VERSION_REQUIRED       "3.8")
set(NVFUSER_Torch_VERSION_REQUIRED        "2.0")
set(NVFUSER_pybind11_VERSION_REQUIRED     "2.0")

set(NVFUSER_LLVM_VERSION_REQUIRED         "18.1")

# --------------------------
# Find all dependencies
# --------------------------
macro(find_nvfuser_dependencies)

  # --------------------------
  # Python
  # --------------------------
  message("")
  message("Finding Python...")
  # Find without version requirement to see what's available
  find_package(Python COMPONENTS Interpreter Development)

  # --------------------------
  # Torch
  # --------------------------
  message("")
  message("Finding Torch...")
  if(Python_FOUND)
    execute_process(
      COMMAND "${Python_EXECUTABLE}" -c "import torch; print(torch.utils.cmake_prefix_path)"
      OUTPUT_VARIABLE TORCH_CMAKE_PATH
      OUTPUT_STRIP_TRAILING_WHITESPACE
      RESULT_VARIABLE TORCH_FIND_RESULT
    )

    if(TORCH_FIND_RESULT EQUAL 0)
      # need this since the pytorch execution uses a different name
      set(PYTHON_EXECUTABLE ${Python_EXECUTABLE})
      list(APPEND CMAKE_PREFIX_PATH "${TORCH_CMAKE_PATH}")

      # set CUDA_ARCH for cu tests.
      if(TORCH_CUDA_ARCH_LIST)
        set(ARCH_FLAGS)
        cuda_select_nvcc_arch_flags(ARCH_FLAGS ${TORCH_CUDA_ARCH_LIST})
        list(APPEND CUDA_NVCC_FLAGS ${ARCH_FLAGS})
      endif()

      # CXX flags is necessary since https://github.com/pytorch/pytorch/issues/98093
      string(APPEND CMAKE_CXX_FLAGS " ${TORCH_CXX_FLAGS}")

      # Find without version requirement to see what's available
      find_package(Torch)
    else()
      set(Torch_FOUND FALSE)
    endif()
  else()
    set(Torch_FOUND FALSE)
  endif()

  # --------------------------
  # pybind11
  # --------------------------
  message("")
  message("Finding pybind11...")
  find_package(pybind11)

  # --------------------------
  # CUDAToolkit
  # --------------------------
  message("")
  message("Finding CUDAToolkit...")
  find_package(CUDAToolkit COMPONENTS Cupti cuda_driver)

  # --------------------------
  # LLVM
  # --------------------------
  message("")
  message("Finding LLVM...")
  # Find without version requirement to see what's available
  find_package(LLVM)

  if (LLVM_FOUND)
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
  endif()

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

macro(report name location)
  if (${name}_FOUND)
    # Pad the name to 12 characters for alignment
    string(LENGTH "${name}" name_len)
    math(EXPR pad_len "12 - ${name_len}")
    string(REPEAT " " ${pad_len} name_padding)

    # Get minimum required version
    set(min_version "${NVFUSER_${name}_VERSION_REQUIRED}")

    # Pad version number to 10 characters for alignment
    string(LENGTH "${${name}_VERSION}" ver_len)
    math(EXPR ver_pad_len "10 - ${ver_len}")
    if(ver_pad_len LESS 0)
      set(ver_pad_len 0)
    endif()
    string(REPEAT " " ${ver_pad_len} ver_padding)

    # Compare versions to determine symbol and color
    if(DEFINED min_version AND NOT "${min_version}" STREQUAL "")
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

    if(version_ok)
      message(STATUS "${ColorGreen}[ OK ]${ColorReset} ${ColorWhite}${name}${name_padding}${ColorReset}  ${version_display}  ${ColorCyan}${location}${ColorReset}")
    else()
      message(STATUS "${BoldRed}[FAIL]${ColorReset} ${ColorWhite}${name}${name_padding}${ColorReset}  ${version_display}  ${ColorCyan}${location}${ColorReset}")
    endif()
  else()
    # Dependency not found at all
    set(min_version "${NVFUSER_${name}_VERSION_REQUIRED}")
    if(DEFINED min_version AND NOT "${min_version}" STREQUAL "")
      message(STATUS "${BoldRed}[FAIL]${ColorReset} ${ColorWhite}${name}${ColorReset} NOT found (requires v${min_version} or higher)")
      list(APPEND _DEPENDENCY_FAILURES "${name}: not found (requires v${min_version} or higher)")
    else()
      message(STATUS "${BoldRed}[FAIL]${ColorReset} ${ColorWhite}${name}${ColorReset} NOT found")
      list(APPEND _DEPENDENCY_FAILURES "${name}: not found")
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

  report(Python      ${Python_EXECUTABLE})
  report(Torch       ${Torch_DIR})
  report(pybind11    ${pybind11_DIR})
  report(CUDAToolkit ${CUDAToolkit_LIBRARY_ROOT})
  report(LLVM        ${LLVM_DIR})

  message_colored("${BoldWhite}"  "===========================================")
  message_colored("${BoldBlue}"   "///////////////////////////////////////////")

  # If there were any failures, show them and error out
  list(LENGTH _DEPENDENCY_FAILURES failure_count)
  if(failure_count GREATER 0)
    message("")
    message_colored("${BoldRed}" "Configuration failed due to missing or incompatible dependencies:")
    foreach(failure ${_DEPENDENCY_FAILURES})
      message_colored("${BoldRed}" "  - ${failure}")
    endforeach()
    message("")
    message(FATAL_ERROR "Please install or upgrade the required dependencies listed above.")
  endif()
endmacro()


