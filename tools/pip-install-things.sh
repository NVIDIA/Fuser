#!/bin/bash

pip3 install --pre torch --index-url https://download.pytorch.org/whl/nightly/cu121

# PyTorch's has a buildin Modules_CUDA which is super old. Remove it.
rm -rf $(python -c 'import torch; print(torch.__path__[0])')/share/cmake/Caffe2/Modules_CUDA_fix

pip install -r requirements.txt
pip install meson
