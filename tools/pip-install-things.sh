#!/bin/bash

pip3 install --pre torch --index-url https://download.pytorch.org/whl/nightly/cu121

# PyTorch's has a buildin Modules_CUDA which is super old. Remove it.
rm -rf /opt/hostedtoolcache/Python/3.10.11/x64/lib/python3.10/site-packages/torch/share/cmake/Caffe2/Modules_CUDA_fix

pip install -r requirements.txt
