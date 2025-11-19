# SPDX-FileCopyrightText: Copyright (c) 2025-present NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

CUDA_HOME?=/usr/local/cuda
ARCH=sm_100
CCBIN?=$(shell which g++)

NVCC_FLAGS=-lcuda -ccbin "$(CCBIN)" -g -lineinfo -std=c++20 --expt-relaxed-constexpr -arch=$(ARCH) -uumn --ptxas-options="--warn-on-local-memory-usage --warn-on-spills"

# Path to the extracted LibTorch
LIBTORCH_PATH := /usr/local/lib/python3.12/dist-packages/torch

# Include and library paths
# Install pytorch nightly
CXXFLAGS := -I$(LIBTORCH_PATH)/include -I$(LIBTORCH_PATH)/include/torch/csrc/api/include
CXXFLAGS += -I../../third_party/cutlass/include -I../../third_party/cutlass/include/tools/util/include
LDFLAGS := -L$(LIBTORCH_PATH)/lib

# Libraries to link against
LDLIBS := -ltorch -ltorch_cpu -lc10 -lnvrtc

RUNTIME_PATH := -Xlinker -rpath -Xlinker $(LIBTORCH_PATH)/lib

.PHONY: default
default: harness

.PHONY: clean
clean:
	rm -f harness *.cubin

harness: clean
	$(CUDA_HOME)/bin/nvcc $(NVCC_FLAGS) $(CXXFLAGS) $(LDFLAGS) $(RUNTIME_PATH) main.cpp -o $@ --ptxas-options=-v $(LDLIBS)
