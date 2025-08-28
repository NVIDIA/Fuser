# Makefile for templated CUDA kernel example

# CUDA toolkit path (adjust if needed)
CUDA_PATH ?= /usr/local/cuda

# Compiler settings
NVCC = $(CUDA_PATH)/bin/nvcc
CXX = g++

# Compiler flags
NVCC_FLAGS = -O2 -std=c++17 --generate-code=arch=compute_100a,code=[compute_100a,sm_100a] -gencode arch=compute_120a,code=compute_120a -I. -I./cutlass/include
CXX_FLAGS = -std=c++14 -O3

# Include paths
INCLUDES = -I$(CUDA_PATH)/include -I./

# Library paths and libraries
LIBS = -L$(CUDA_PATH)/lib64 -lcudart

# Target executables
TARGETS = test_templated_kernel

# Default target
all: $(TARGETS)

# Build the templated kernel test program
test_templated_kernel: test_templated_kernel.cu templated_kernel.cuh
	$(NVCC) $(NVCC_FLAGS) $(INCLUDES) -o $@ $< $(LIBS)

# Clean build artifacts
clean:
	rm -f $(TARGETS) *.o

# Run the test program
test: test_templated_kernel
	./test_templated_kernel

# Install target (optional)
install: $(TARGETS)
	cp $(TARGETS) /usr/local/bin/

# Help target
help:
	@echo "Available targets:"
	@echo "  all                 - Build test_templated_kernel"
	@echo "  test_templated_kernel - Build templated kernel test program"
	@echo "  clean               - Remove build artifacts"
	@echo "  test                - Build and run test program"
	@echo "  help                - Show this help message"

# Phony targets
.PHONY: all clean test install help

# Additional debugging target
debug: NVCC_FLAGS += -g -G -lineinfo
debug: $(TARGETS)

# Profile target (requires Nsight Compute)
profile: test_templated_kernel
	ncu --set full ./test_templated_kernel
