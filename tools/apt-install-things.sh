#!/bin/bash

set -e

# Install cuda keyring
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt-get update

# Remove some old toolchains. By default, the github action comes with multiple versions of gcc and clang installed.
# Having many versions of gcc and clang installed interfers with each other, causing weird build and clang-tidy errors.
# We only keep one version of gcc and clang in the system, and remove the rest.
sudo apt-get -y remove gcc-13 libstdc++-13-dev gcc-12 libstdc++-12-dev

# Install the latest version of clang and gcc.
sudo apt-get -y install --reinstall clang-19 gcc-14 nlohmann-json3-dev ninja-build

# Install minimal cuda toolkit.
sudo apt-get -y install cuda-compiler-12-8 cuda-command-line-tools-12-8 cuda-libraries-dev-12-8 libnccl-dev

# llvm-dev are for host IR compilation, which uses LLVM JIT.
sudo apt-get -y install llvm-dev
