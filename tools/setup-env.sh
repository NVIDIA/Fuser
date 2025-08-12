#!/bin/bash

export CUDACXX=/usr/local/cuda/bin/nvcc
export PATH=/usr/local/cuda/bin:${PATH}
export CUDA_INSTALL_PATH=/usr/local/cuda
export CUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda

# Use absolute paths for clang compilers to avoid PATH issues
# Try common locations where clang-19 might be installed
if [ -x "/usr/bin/clang-19" ]; then
    export CC=/usr/bin/clang-19
    export CXX=/usr/bin/clang++-19
elif [ -x "/usr/local/bin/clang-19" ]; then
    export CC=/usr/local/bin/clang-19
    export CXX=/usr/local/bin/clang++-19
elif command -v clang-19 >/dev/null 2>&1; then
    export CC=$(command -v clang-19)
    export CXX=$(command -v clang++-19)
else
    echo "Error: clang-19 not found. Please ensure clang-19 is installed."
    exit 1
fi
