# --------------------------
# Find all dependencies
# --------------------------
macro(find_nvfuser_dependencies)

  find_package(Python ${NVFUSER_Python_VERSION_REQUIRED} COMPONENTS Interpreter Development REQUIRED)

  execute_process(
    COMMAND "${Python_EXECUTABLE}" -c "import torch; print(torch.utils.cmake_prefix_path)"
    OUTPUT_VARIABLE TORCH_CMAKE_PATH
    OUTPUT_STRIP_TRAILING_WHITESPACE
    RESULT_VARIABLE TORCH_FIND_RESULT
  )

  if(NOT TORCH_FIND_RESULT EQUAL 0)
    message(FATAL_ERROR "Failed to find PyTorch via Python. Make sure 'torch' is installed in the detected Python environment.")
  endif()

  list(APPEND CMAKE_PREFIX_PATH "${TORCH_CMAKE_PATH}")

  find_package(Torch ${NVFUSER_Torch_VERSION_REQUIRED} REQUIRED)

endmacro()

# --------------------------
# Reset Found Statuses
# --------------------------
macro(reset_dependencies)
  set(Python_FOUND False)
  set(Torch_FOUND False)
  set(Caffe2_FOUND False)
endmacro()

macro(report name location)

  if (${name}_FOUND)
    message(STATUS "[ OK ] ${name} v${${name}_VERSION} @ ${location}")
  else()
    message(STATUS "[FAIL] ${name} NOT found!!!")
  endif()

endmacro()

macro(report_dependencies)
  message(STATUS "")
  message(STATUS "=========================================")
  message(STATUS "      D E P E N D E N C Y   R E P O R T  ")
  message(STATUS "=========================================")

  report(Python ${Python_EXECUTABLE})
  report(Torch  ${Torch_DIR})

  message(STATUS "=========================================")
endmacro()


