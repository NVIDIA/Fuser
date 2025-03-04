#!/bin/bash

export CUDACXX=/usr/local/cuda/bin/nvcc
export PATH=/usr/local/cuda/bin:${PATH}
export CUDA_INSTALL_PATH=/usr/local/cuda
export CUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda
export CC=clang-19
export CXX=clang++-19
export CXXFLAGS="--stdlib=libstdc++ -L/usr/lib/gcc/x86_64-linux-gnu/14 -I/usr/include/c++/14"
export CPATH=/usr/include/c++/13:$CPATH

