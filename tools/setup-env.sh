#!/bin/bash

# Install cuda
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.0-1_all.deb
sudo dpkg -i cuda-keyring_1.0-1_all.deb
sudo apt-get update
sudo apt-get -y install cuda-toolkit-12-1 libnccl-dev

# cmake environment variables
export CUDACXX=/usr/local/cuda/bin/nvcc
export PATH=/usr/local/cuda/bin:${PATH}
export CUDA_INSTALL_PATH=/usr/local/cuda

# Install pytorch
pip3 install --pre torch --index-url https://download.pytorch.org/whl/nightly/cu121

# PyTorch's has a buildin Modules_CUDA which is super old. Remove it.
rm -rf /opt/hostedtoolcache/Python/3.10.11/x64/lib/python3.10/site-packages/torch/share/cmake/Caffe2/Modules_CUDA_fix

# Install other requirements
pip install -r requirements.txt
