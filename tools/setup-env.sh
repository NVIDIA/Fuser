#!/bin/bash

# Install latest CMake. Always use the latest version of CMake because older version
# of CMake might not have updated CUDA architecture list.
# https://apt.kitware.com/
wget -O - https://apt.kitware.com/keys/kitware-archive-latest.asc 2>/dev/null | gpg --dearmor - | sudo tee /usr/share/keyrings/kitware-archive-keyring.gpg >/dev/null
echo "deb [signed-by=/usr/share/keyrings/kitware-archive-keyring.gpg] https://apt.kitware.com/ubuntu/ $(lsb_release -cs) main" | sudo tee /etc/apt/sources.list.d/kitware.list >/dev/null
sudo apt-get update
sudo rm /usr/share/keyrings/kitware-archive-keyring.gpg
sudo apt-get -y install kitware-archive-keyring
sudo apt-get -y install cmake

cmake --version

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

# Install other requirements
pip install -r requirements.txt
