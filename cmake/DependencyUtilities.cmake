# --------------------------
# Find all dependencies
# --------------------------
macro(find_nvfuser_dependencies)

  # --------------------------
  # Python
  # --------------------------
  message("")
  message("Finding Python...")
  find_package(Python ${NVFUSER_Python_VERSION_REQUIRED} COMPONENTS Interpreter Development REQUIRED)


  # --------------------------
  # Torch
  # --------------------------
  message("")
  message("Finding Torch...")
  execute_process(
    COMMAND "${Python_EXECUTABLE}" -c "import torch; print(torch.utils.cmake_prefix_path)"
    OUTPUT_VARIABLE TORCH_CMAKE_PATH
    OUTPUT_STRIP_TRAILING_WHITESPACE
    RESULT_VARIABLE TORCH_FIND_RESULT
  )

  if(NOT TORCH_FIND_RESULT EQUAL 0)
    message(FATAL_ERROR "Failed to find PyTorch via Python. Make sure 'torch' is installed in the detected Python environment.")
  endif()

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

  find_package(Torch ${NVFUSER_Torch_VERSION_REQUIRED} REQUIRED)

  
  # --------------------------
  # pybind11
  # --------------------------
  message("")
  message("Finding pybind11...")
  find_package(pybind11 REQUIRED)

  # --------------------------
  # CUDAToolkit
  # --------------------------
  message("")
  message("Finding CUDAToolkit...")
  find_package(CUDAToolkit REQUIRED)

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
endif()

function(message_colored color text)
    message(STATUS "${color}${text}${ColorReset}")
endfunction()

macro(report name location)
  if (${name}_FOUND)
    message_colored("${ColorGreen}" "[ OK ] ${name} v${${name}_VERSION} @ ${location}")
  else()
    message_colored("${BoldRed}" "[FAIL] ${name} NOT found!!!")
  endif()
endmacro()

macro(report_dependencies)
  message("")
  message_colored("${BoldYellow}" "=========================================")
  message_colored("${BoldGreen}"  "    D E P E N D E N C Y   R E P O R T  ")
  message_colored("${BoldYellow}" "=========================================")

  report(Python      ${Python_EXECUTABLE})
  report(Torch       ${Torch_DIR})
  report(pybind11    ${pybind11_DIR})
  report(CUDAToolkit ${CUDAToolkit_LIBRARY_ROOT})


  message(STATUS "=========================================")
endmacro()


