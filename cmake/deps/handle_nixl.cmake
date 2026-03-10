# SPDX-FileCopyrightText: Copyright (c) 2025-present NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

# ------------------------------------------------------------------------------
# NIXL Handler
# ------------------------------------------------------------------------------

macro(handle_nixl)
  message("")
  message("Finding NIXL...")

  if(NOT NVFUSER_BUILD_WITH_NIXL)
    set(NIXL_FOUND FALSE)
    message(STATUS "NIXL disabled (NVFUSER_BUILD_WITH_NIXL=OFF)")
  else()
    # User may need to set NIXL_PREFIX to the NIXL install directory.
    find_path(NIXL_INCLUDE_DIR nixl.h
      HINTS $ENV{NIXL_PREFIX}/include ENV CPATH
    )
    find_library(NIXL_LIBRARY nixl
      HINTS $ENV{NIXL_PREFIX}/lib $ENV{NIXL_PREFIX}/lib64 $ENV{NIXL_PREFIX}/lib/x86_64-linux-gnu
    )
    find_library(NIXL_BUILD_LIBRARY nixl_build
      HINTS $ENV{NIXL_PREFIX}/lib $ENV{NIXL_PREFIX}/lib64 $ENV{NIXL_PREFIX}/lib/x86_64-linux-gnu
    )

    if(NIXL_INCLUDE_DIR AND NIXL_LIBRARY)
      set(NIXL_FOUND TRUE)
      message(STATUS "Found NIXL: ${NIXL_LIBRARY} (include: ${NIXL_INCLUDE_DIR})")
      if(NIXL_BUILD_LIBRARY)
        message(STATUS "Found NIXL build lib: ${NIXL_BUILD_LIBRARY}")
      endif()
    else()
      set(NIXL_FOUND FALSE)
      message(WARNING "NIXL not found – building without NIXL support. Set NIXL_PREFIX to the NIXL install directory.")
    endif()

    # CUDA major version constraint check
    if(NIXL_FOUND AND Python_FOUND AND CUDAToolkit_FOUND)
      execute_process(
        COMMAND "${Python_EXECUTABLE}" -c "import nixl; print(nixl._pkg.__name__.split('_cu')[-1])"
        OUTPUT_VARIABLE nixl_cuda_major
        OUTPUT_STRIP_TRAILING_WHITESPACE
        ERROR_QUIET
        RESULT_VARIABLE nixl_cuda_result
      )

      if(nixl_cuda_result EQUAL 0 AND NOT nixl_cuda_major STREQUAL "")
        set(NIXL_CUDA_VERSION "${nixl_cuda_major}")
        set(cuda_toolkit_major "${CUDAToolkit_VERSION_MAJOR}")

        if(NOT nixl_cuda_major STREQUAL cuda_toolkit_major)
          set(NIXL_CUDA_constraint_status "mismatch")
          set(NIXL_CUDA_constraint_found "${nixl_cuda_major}")
          set(NIXL_CUDA_constraint_required "${cuda_toolkit_major}")
          message(WARNING "NIXL CUDA major version mismatch: NIXL built for CUDA ${nixl_cuda_major}, but CUDAToolkit major is ${cuda_toolkit_major}")
        else()
          set(NIXL_CUDA_constraint_status "match")
          set(NIXL_CUDA_constraint_version "${nixl_cuda_major}")
        endif()
      else()
        set(NIXL_CUDA_constraint_status "not_available")
      endif()
    else()
      set(NIXL_CUDA_constraint_status "not_available")
    endif()
  endif()

  message(STATUS "  NIXL_FOUND                         : ${NIXL_FOUND}")
  if(NIXL_FOUND)
    message(STATUS "    NIXL_INCLUDE_DIR: ${NIXL_INCLUDE_DIR}")
    message(STATUS "    NIXL_LIBRARY    : ${NIXL_LIBRARY}")
  endif()

  set_dependency_report_status(NIXL)
endmacro()

macro(link_nixl target)
  if(NIXL_FOUND)
    target_include_directories(${target} PRIVATE ${NIXL_INCLUDE_DIR})
    target_link_libraries(${target} PRIVATE ${NIXL_LIBRARY})
    if(NIXL_BUILD_LIBRARY)
      target_link_libraries(${target} PRIVATE ${NIXL_BUILD_LIBRARY})
    endif()
    target_compile_definitions(${target} PRIVATE USE_NIXL)
  endif()
endmacro()
