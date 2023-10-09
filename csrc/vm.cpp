#include <cstddef>
#include <cuda.h>

#include <vm.h>
#include <executor_kernel_arg.h>
#include <executor_params.h>
#include <cuda_utils.h>
#include <kernel.h>

namespace nvfuser {

/// The basic layout of a serialized VM object is:
///
///   NVF0
///   i64: header offset
///   ...
///
/// A header just contains offsets to other sections:
///
///   i64: modules offset
///   i64: tensors offset
///   i64: program offset
///
/// The modules section is just a number of modules, their sizes, and offsets:
///
///   i64: the number of modules, 'n_modules'
///   array of n_modules elements:
///     i64: offset of module of module 'i', where 'i' is the index
///   array of n_modules elements:
///     i64: size (number of bytes) of module 'i'
///
/// Yes, those are parallel arrays: all the offsets come before all the sizes.
///

vm_t::vm_t(const void* s) {
  const char* src = static_cast<const char*>(s); 
  NVF_CHECK(src[0] == 'N' && src[1] == 'V' && src[2] == 'F' && src[3] == 0);

  //const int64_t offs_header = 42;
}
vm_t::~vm_t() {}

#if 0
void
vm_t::initialize(KernelArgumentHolder& args,
                  const LaunchParams& lp,
                  const CompileParams& cp,
                  std::vector<at::Tensor> outputs,
                  const kir::Kernel* kernel) {
  strm_ = static_cast<cudaStream_t>(at::cuda::getCurrentCUDAStream());

  ExpressionEvaluator eval;
  const std::vector<Val*>& inputs = kernel->inputs();
  NVF_CHECK(inputs.size() == args.size());
  for(size_t i=0; i < inputs.size(); ++i) {
    eval.bind(inputs[i], *args[i]);
  }
}

void
vm_t::setKernelParams(CUfunction fqn, const std::array<unsigned,3>& grid,
                       const std::array<unsigned,3>& block, unsigned shmem,
                       CUstream strm, const std::vector<void*>& args) {
  function_ = fqn;
  gridDim_ = grid;
  blockDim_ = block;
  shmem_ = shmem;
  strm_ = strm;
  args_ = args;
}
#endif

void
vm_t::exec() const {
  NVFUSER_CUDA_SAFE_CALL(
    cuLaunchKernel(function_, gridDim_[0], gridDim_[1], gridDim_[2],
                   blockDim_[0], blockDim_[1], blockDim_[2], shmem_,
                   strm_, const_cast<void**>(args_.data()), nullptr)
  );
}

void
vm_t::coop_exec() const {
  NVFUSER_CUDA_SAFE_CALL(
    cuLaunchCooperativeKernel(
      function_, gridDim_[0], gridDim_[1], gridDim_[2],
      blockDim_[0], blockDim_[1], blockDim_[2], shmem_, strm_,
      const_cast<void**>(args_.data()))
  );
}

}
