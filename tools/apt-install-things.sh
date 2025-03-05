#!/bin/bash

set -e

# Remove headers of gcc-14 because it is too new and not very compatible with clang
sudo apt-get -y remove gcc-13 g++13 libstdc++-13-dev gcc-12 g++12 libstdc++-12-dev
#sudo rm -rf /usr/include/c++/14 #/usr/lib/gcc/x86_64-linux-gnu/13
sudo apt-get -y install --reinstall build-essential clang-19 clang gcc-14 g++-14 nlohmann-json3-dev #g++-13 libstdc++-13-dev 

# Install cuda
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt-get update
sudo apt-get -y install ninja-build cuda-compiler-12-8 cuda-command-line-tools-12-8 cuda-libraries-dev-12-8 libnccl-dev 

tree /usr/include/c++/
# rm /usr/include/c++/12 /usr/include/c++/13
