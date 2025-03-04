#!/bin/bash

set -eX

# Remove headers of gcc-14 because it is too new and not very compatible with clang
#sudo apt-get -y remove gcc-14 g++14 libstdc++-14-dev
#sudo rm -rf /usr/include/c++/14 #/usr/lib/gcc/x86_64-linux-gnu/13 
sudo apt-get -y install clang-19 libc++-dev libc++abi-dev nlohmann-json3-dev #g++-13 libstdc++-13-dev 

# Install cuda
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt-get update
sudo apt-get -y install ninja-build cuda-compiler-12-8 cuda-command-line-tools-12-8 cuda-libraries-dev-12-8 libnccl-dev 

echo | clang++ -E -dM -x c++ - | grep stdint
ls -l /usr/include/stdint.h
