#!/bin/bash

# Install pytorch in background
pip3 install --pre torch --index-url https://download.pytorch.org/whl/nightly/cu121 &
torch_pid=$!

# Install cuda
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.0-1_all.deb
sudo dpkg -i cuda-keyring_1.0-1_all.deb
sudo apt-get update
sudo apt-get -y install ninja-build cuda-compiler-12-1 cuda-command-line-tools-12-1 cuda-libraries-dev-12-1 libnccl-dev &

# wait for PyTorch installation to finish
wait $torch_pid

# cmake environment variables
export CUDAARCHS=86
export CUDACXX=/usr/local/cuda/bin/nvcc
export PATH=/usr/local/cuda/bin:${PATH}
export CUDA_INSTALL_PATH=/usr/local/cuda

# Install other requirements
pip install -r requirements.txt

# wait for CUDA installation to finish
wait
