#!/bin/bash

set -e

# Install cuda
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt-get update
sudo apt-get -y install ninja-build cuda-compiler-12-8 cuda-command-line-tools-12-8 cuda-libraries-dev-12-8 libnccl-dev clang-19 nlohmann-json3-dev

# Remove headers of gcc-14 because it is too new and not very compatible with clang
sudo apt-get -y remove gcc-14 g++14 libstdc++-14-dev
dpkg -S /usr/include/c++/14/cstdint
