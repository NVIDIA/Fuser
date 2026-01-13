# SPDX-FileCopyrightText: Copyright (c) 2025-present NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

# ------------------------------------------------------------------------------
# PyTorch Handler with CUDA Constraint Validation
# ------------------------------------------------------------------------------

macro(handle_torch)
  message("")
  message("Finding Torch...")

  # Setup: Query Python for Torch path
  if(Python_FOUND)
    execute_process(
      COMMAND "${Python_EXECUTABLE}" -c "import torch; print(torch.utils.cmake_prefix_path)"
      OUTPUT_VARIABLE TORCH_CMAKE_PATH
      OUTPUT_STRIP_TRAILING_WHITESPACE
      RESULT_VARIABLE TORCH_FIND_RESULT
    )

    if(TORCH_FIND_RESULT EQUAL 0)
      # Setup environment for Torch find_package
      list(APPEND CMAKE_PREFIX_PATH "${TORCH_CMAKE_PATH}")

      # Direct find_package call
      find_package(Torch ${MAYBE_REQUIRED})

      # Set CUDA_ARCH for cu tests
      if(TORCH_CUDA_ARCH_LIST)
        set(ARCH_FLAGS)
        cuda_select_nvcc_arch_flags(ARCH_FLAGS ${TORCH_CUDA_ARCH_LIST})
        list(APPEND CUDA_NVCC_FLAGS ${ARCH_FLAGS})
      endif()

      # CXX flags necessary for https://github.com/pytorch/pytorch/issues/98093
      string(APPEND CMAKE_CXX_FLAGS " ${TORCH_CXX_FLAGS}")
    else()
      set(Torch_FOUND FALSE)
    endif()
  else()
    set(Torch_FOUND FALSE)
  endif()

  # Use common status function for basic version check
  set_dependency_report_status(Torch)

  # Additional validation: Check CUDA constraint
  # This must happen AFTER set_dependency_status since we need Torch to be found
  if(Torch_FOUND AND CUDAToolkit_FOUND)
    # Query torch Python package for CUDA version
    execute_process(
      COMMAND "${Python_EXECUTABLE}" -c "import torch; print(torch.version.cuda if torch.version.cuda else 'N/A')"
      OUTPUT_VARIABLE torch_cuda_version
      OUTPUT_STRIP_TRAILING_WHITESPACE
      RESULT_VARIABLE torch_cuda_result
    )

    if(torch_cuda_result EQUAL 0 AND NOT torch_cuda_version STREQUAL "N/A" AND NOT torch_cuda_version STREQUAL "None")
      # Get CUDAToolkit version (major.minor only for comparison)
      set(cuda_toolkit_version "${CUDAToolkit_VERSION_MAJOR}.${CUDAToolkit_VERSION_MINOR}")

      # Extract major.minor from Torch CUDA version
      string(REGEX MATCH "^([0-9]+\\.[0-9]+)" torch_cuda_major_minor "${torch_cuda_version}")

      # Compare major.minor versions
      if(NOT torch_cuda_major_minor STREQUAL cuda_toolkit_version)
        # Version mismatch
        set(Torch_CUDA_constraint_status "mismatch")
        set(Torch_CUDA_constraint_found "${torch_cuda_major_minor}")
        set(Torch_CUDA_constraint_required "${cuda_toolkit_version}")
        # Mark dependencies as failed (Torch_CUDA constraint is required)
        set(NVFUSER_DEPENDENCIES_OK FALSE)
      else()
        # Versions match!
        set(Torch_CUDA_constraint_status "match")
        set(Torch_CUDA_constraint_version "${torch_cuda_major_minor}")
      endif()
    else()
      # Torch might not have CUDA support or query failed
      set(Torch_CUDA_constraint_status "not_available")
    endif()
  elseif(NOT CUDAToolkit_FOUND)
    # Can't validate if CUDAToolkit wasn't found
    set(Torch_CUDA_constraint_status "not_available")
  endif()
endmacro()
