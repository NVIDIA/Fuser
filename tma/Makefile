CUDA_HOME?=/usr/local/cuda
ARCH=sm_86
CCBIN?=$(shell which g++)

NVCC_FLAGS=-lcuda -ccbin "$(CCBIN)" -g -lineinfo -std=c++17 --expt-relaxed-constexpr -arch=$(ARCH) -uumn --ptxas-options="--warn-on-local-memory-usage --warn-on-spills" -DCUDA_PTX_KNOB_COLD_BLOCK_ENABLED=1

# Path to the extracted LibTorch
LIBTORCH_PATH := /home/rspring/miniconda3/lib/python3.10/site-packages/torch

# Include and library paths
CXXFLAGS := -I$(LIBTORCH_PATH)/include -I$(LIBTORCH_PATH)/include/torch/csrc/api/include -D_GLIBCXX_USE_CXX11_ABI=0
LDFLAGS := -L$(LIBTORCH_PATH)/lib

# Libraries to link against
LDLIBS := -ltorch -ltorch_cpu -lc10 -lc10_cuda

# If using CUDA
LDFLAGS += -L/usr/local/cuda/lib64
LDLIBS += -lcudnn -lcudart -lcublas -lcurand -lcusolver

.PHONY: default
default: harness

.PHONY: clean
clean:
	rm -f harness *.cubin

harness: clean pointwise_smem
	$(CUDA_HOME)/bin/nvcc $(NVCC_FLAGS) $(CXXFLAGS) $(LDFLAGS) harness.cpp -o $@ --ptxas-options=-v $(LDLIBS)

pointwise_smem:
	$(CUDA_HOME)/bin/nvcc $(NVCC_FLAGS) --cubin pointwise_smem.cu

tma_harness: clean tma_kernel
	$(CUDA_HOME)/bin/nvcc $(NVCC_FLAGS) $(CXXFLAGS) $(LDFLAGS) tma_harness.cpp -o $@ --ptxas-options=-v $(LDLIBS)

tma_kernel:
	$(CUDA_HOME)/bin/nvcc -ccbin "/usr/bin/g++" -g -lineinfo -std=c++17 -arch sm_90 --cubin tma_kernel.cu
