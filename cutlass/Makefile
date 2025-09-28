# Get the include paths required for PyTorch C++ headers
# This command outputs a space-separated list of directories
PYTORCH_INCLUDES := $(shell python3 -c "import torch.utils.cpp_extension; print(' '.join(['-I{}'.format(p) for p in torch.utils.cpp_extension.include_paths()]))")

# Get the library paths required for linking
PYTORCH_LIB_DIRS := $(shell python3 -c "import torch.utils.cpp_extension; print(' '.join(['-L{}'.format(p) for p in torch.utils.cpp_extension.library_paths()]))")

# Define the libraries to link against. The most common ones are:
# -l: The library name stripped of 'lib' prefix and '.so' suffix (e.g., -lc10, -ltorch, -ltorch_cpu, -ltorch_python)
PYTORCH_LIBS := -ltorch -ltorch_cpu -lc10

LDFLAGS += $(PYTORCH_LIB_DIRS) $(PYTORCH_LIBS) -Xlinker -rpath -Xlinker /usr/local/lib/python3.12/dist-packages/torch/lib

CUTLASS_FLAGS = -DCUTLASS_NVCC_ARCHS="90a" -DCUTE_USE_PACKED_TUPLE=1 -DCUTLASS_ENABLE_TENSOR_CORE_MMA=1 -DCUTLASS_VERSIONS_GENERATED -DCUTLASS_TEST_LEVEL=0 -DCUTLASS_TEST_ENABLE_CACHED_RESULTS=1 -DCUTLASS_DEBUG_TRACE_LEVEL=0 -DCUTLASS_DISABLE_SYNCLOG
NVCC_FLAGS = -std=c++17 -O3 -DNDEBUG -w -gencode=arch=compute_90a,code=sm_90a
NVCC_FLAGS += --expt-relaxed-constexpr --expt-extended-lambda --threads=${NPROC} -Xcompiler=-Wconversion -Xcompiler=-fno-strict-aliasing -Xcompiler=-Wno-deprecated-declarations
INCLUDES = -I../third_party/cutlass/include -I../third_party/cutlass/include/tools/util/include $(PYTORCH_INCLUDES)

bulk_load: nvrtc_hopper_bulk_load.cu
	nvcc $(NVCC_FLAGS) $(CUTLASS_FLAGS) $(INCLUDES) $< -o $@ $(LDFLAGS)
