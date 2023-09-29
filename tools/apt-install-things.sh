#!/bin/bash

# Install cuda
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.0-1_all.deb
sudo dpkg -i cuda-keyring_1.0-1_all.deb
sudo apt-get update
sudo apt-get -y install ninja-build cuda-compiler-12-1 cuda-command-line-tools-12-1 cuda-libraries-dev-12-1 libnccl-dev libgtest-dev libgmock-dev libbenchmark-dev libflatbuffers-dev
