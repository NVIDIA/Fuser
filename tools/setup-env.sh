#!/bin/bash

export CUDACXX=/usr/local/cuda/bin/nvcc
export PATH=/usr/local/cuda/bin:${PATH}
export CUDA_INSTALL_PATH=/usr/local/cuda
export CUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda

# Use absolute paths for clang compilers to avoid PATH issues
# Try common locations where clang-19 might be installed
if [ -x "/usr/bin/clang-19" ]; then
    CLANG_PATH=/usr/bin/clang-19
    CLANGXX_PATH=/usr/bin/clang++-19
elif [ -x "/usr/local/bin/clang-19" ]; then
    CLANG_PATH=/usr/local/bin/clang-19
    CLANGXX_PATH=/usr/local/bin/clang++-19
elif command -v clang-19 >/dev/null 2>&1; then
    CLANG_PATH=$(command -v clang-19)
    CLANGXX_PATH=$(command -v clang++-19)
else
    echo "Error: clang-19 not found. Please ensure clang-19 is installed."
    exit 1
fi

# Set CC/CXX to the raw compiler paths. Use CMAKE_<LANG>_COMPILER_LAUNCHER
# for ccache instead of CC="ccache clang", because the "ccache clang" form
# causes CMake (and PyTorch's TorchConfig.cmake) to resolve the CUDA host
# compiler (-ccbin) to ccache instead of clang, breaking nvcc.
if [ -z "$CC" ]; then
    export CC=$CLANG_PATH
    export CXX=$CLANGXX_PATH
    if command -v ccache >/dev/null 2>&1; then
        export CMAKE_C_COMPILER_LAUNCHER=ccache
        export CMAKE_CXX_COMPILER_LAUNCHER=ccache
        export CMAKE_CUDA_COMPILER_LAUNCHER=ccache
    fi
fi

export TORCH_CUDA_ARCH_LIST="10.0"
