#!/bin/bash

export CUDACXX=/usr/local/cuda/bin/nvcc
export PATH=/usr/local/cuda/bin:${PATH}
export CUDA_INSTALL_PATH=/usr/local/cuda
export CC=clang-19
export CXX=clang++-19
export CXXFLAGS="--stdlib=libstdc++ -L/usr/lib/gcc/x86_64-linux-gnu/13 -I/usr/include/c++/13"
export CPATH=/usr/include/c++/13:$CPATH

