NVCC_FLAGS = -std=c++17 -O3 -DNDEBUG -w
NVCC_FLAGS += -DCUTE_USE_PACKED_TUPLE=1 -DCUTLASS_ENABLE_TENSOR_CORE_MMA=1 -DCUTLASS_VERSIONS_GENERATED -DCUTLASS_TEST_LEVEL=0 -DCUTLASS_TEST_ENABLE_CACHED_RESULTS=1 -DCUTLASS_DEBUG_TRACE_LEVEL=0 --expt-relaxed-constexpr --expt-extended-lambda --threads=${NPROC} -Xcompiler=-Wconversion -Xcompiler=-fno-strict-aliasing -Xcompiler=-Wno-deprecated-declarations
INCLUDES = -I../third_party/cutlass/include -I../third_party/cutlass/include/tools/util/include

gemm: nvrtc_hopper_pingpong.cu
	nvcc $(NVCC_FLAGS) $(INCLUDES) $< -o $@
