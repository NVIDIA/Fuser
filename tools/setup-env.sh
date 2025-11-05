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

# Use ccache if available and CC/CXX are not already set
if [ -z "$CC" ]; then
    if command -v ccache >/dev/null 2>&1; then
        export CC="ccache $CLANG_PATH"
        export CXX="ccache $CLANGXX_PATH"
    else
        export CC=$CLANG_PATH
        export CXX=$CLANGXX_PATH
    fi
fi

export TORCH_CUDA_ARCH_LIST="10.0"
