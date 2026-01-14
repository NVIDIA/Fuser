set(NVF_CUTLASS_CUDA_FLAGS
  "-DCUTE_USE_PACKED_TUPLE=1"
  "-DCUTLASS_ENABLE_TENSOR_CORE_MMA=1"
  "-DCUTLASS_VERSIONS_GENERATED"
  "-DCUTLASS_TEST_LEVEL=0"
  "-DCUTLASS_TEST_ENABLE_CACHED_RESULTS=1"
  "-DCUTLASS_DEBUG_TRACE_LEVEL=0"
  "--expt-relaxed-constexpr"
  "--expt-extended-lambda"
  # Equivalent to --threads=<number of CPUs>
  "--threads=0"
  # -----------------
  # Suppress warnings
  # -----------------
  "-Xcompiler=-Wconversion"
  "-Xcompiler=-fno-strict-aliasing"
  # CUDA 13 has deprecated old vector types such as ulong4: https://developer.nvidia.com/blog/whats-new-and-important-in-cuda-toolkit-13-0
  "-Xcompiler=-Wno-deprecated-declarations"
)

set(NVFUSER_CUTLASS_SRCS)
list(APPEND NVFUSER_CUTLASS_SRCS
    ${NVFUSER_CUTLASS}/group_mm.cu
    ${NVFUSER_CUTLASS}/mxfp8_scaled_mm.cu
    ${NVFUSER_CUTLASS}/nvfp4_scaled_mm.cu
    ${NVFUSER_CUTLASS}/nvfp4_scaled_mm_blockscale.cu
    ${NVFUSER_CUTLASS}/nvfp4_scaled_group_mm.cu
    ${NVFUSER_CUTLASS}/nvf_cutlass.cpp
    ${NVFUSER_CUTLASS}/cutlass_utils.cpp
)
add_library(nvf_cutlass SHARED ${NVFUSER_CUTLASS_SRCS})

target_include_directories(nvf_cutlass PRIVATE ${NVFUSER_THIRD_PARTY_DIR}/cutlass/include)
target_include_directories(nvf_cutlass PRIVATE ${NVFUSER_THIRD_PARTY_DIR}/cutlass/tools/util/include)
target_compile_options(nvf_cutlass PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:${NVF_CUTLASS_CUDA_FLAGS}>)
if(NOT MSVC)
  set(NVF_LIB_SUFFIX ".so")
else()
  set(NVF_LIB_SUFFIX ".pyd")
endif()

target_include_directories(nvf_cutlass PUBLIC
  "$<BUILD_INTERFACE:${NVFUSER_SRCS_DIR}>"
  "$<BUILD_INTERFACE:${NVFUSER_CUTLASS}>"
  "$<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/nvfuser>"
)
target_link_libraries(nvf_cutlass PRIVATE "${TORCH_LIBRARIES}" c10)
set_target_properties(nvf_cutlass PROPERTIES
  C_STANDARD ${NVFUSER_C_STANDARD}
  CUDA_STANDARD ${NVFUSER_CUDA_STANDARD}
  CXX_STANDARD ${NVFUSER_CPP_STANDARD}
  CXX_STANDARD_REQUIRED ON
  CXX_VISIBILITY_PRESET hidden
  INSTALL_RPATH
  "$ORIGIN/../../nvidia/cuda_runtime/lib:$ORIGIN/../../nvidia/cuda_nvrtc/lib:$ORIGIN/../../nvidia/cuda_cupti/lib:$ORIGIN/../../torch/lib"
  POSITION_INDEPENDENT_CODE Yes
  VISIBILITY_INLINES_HIDDEN Yes
  CUDA_ARCHITECTURES "100a"
)
# Our CUTLASS kernels require substantially more memory to compile—up to 6
# GB per file. To avoid exhausting system memory, it's helpful to limit
# concurrency specifically for these kernels, without throttling other
# compilation units. A pool of size 2 works well in practice: there are
# only a handful of CUTLASS kernel files, so this restriction has little
# impact on overall build time.
set(CUTLASS_MAX_JOBS 2 CACHE STRING "Max concurrent CUTLASS CUDA compiles (0 = no limit)" )
if(CUTLASS_MAX_JOBS GREATER 0)
  set_property(GLOBAL PROPERTY JOB_POOLS cutlass=${CUTLASS_MAX_JOBS})
  set_target_properties(nvf_cutlass PROPERTIES JOB_POOL_COMPILE cutlass)
endif()
install(TARGETS nvf_cutlass EXPORT NvfuserTargets DESTINATION lib)
