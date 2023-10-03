#include <iostream>
#include <vector>

#include <cuda.h>
#include <cuda_runtime.h>

#define CHECK_CUDA_ERROR(error) CheckCudaError((error), __FILE__, __LINE__)

inline void CheckCudaError(
    cudaError_t error,
    const char* file,
    int line,
    bool abort = true) {
  if (error != cudaSuccess) {
    fprintf(
        stderr,
        "Unexpected error: %s at %s:%d\n",
        cudaGetErrorString(error),
        file,
        line);
    if (abort) {
      exit(error);
    }
  }
}

__global__ void Memzero(float* x) {
  const int row = blockIdx.x;
  const int column = threadIdx.x;

  float* out = &x[row * blockDim.x + column];
  __stwt(out, 0.0f);
}

class L2CacheFlusher {
 public:
  L2CacheFlusher() {
    int device_id;
    CHECK_CUDA_ERROR(cudaGetDevice(&device_id));

    CHECK_CUDA_ERROR(cudaDeviceGetAttribute(
        &l2_cache_size_, cudaDevAttrL2CacheSize, device_id));
    std::cerr << "L2 cache size = " << l2_cache_size_ << std::endl;
    if (l2_cache_size_ <= 0) {
      std::cerr << "The L2 cache size is expected to be positive. Got "
                << l2_cache_size_ << std::endl;
      abort();
    }

    CHECK_CUDA_ERROR(cudaMalloc(&buffer_, l2_cache_size_));
  }

  void Flush(cudaStream_t stream) {
    constexpr int kWarpSize = 32;
    const int num_elements = l2_cache_size_ / sizeof(float);
    Memzero<<<num_elements / kWarpSize, kWarpSize, 0, stream>>>(
        reinterpret_cast<float*>(buffer_));
  }

  ~L2CacheFlusher() {
    if (buffer_ != nullptr) {
      CHECK_CUDA_ERROR(cudaFree(buffer_));
    }
  }

 private:
  void* buffer_ = nullptr;
  int l2_cache_size_ = 0;
};

enum class CacheOp {
  Global,
  Streaming,
};

template <CacheOp cache_op>
__global__ void Axpy(const float alpha, const float* x, float* y) {
  const int row = blockIdx.x;
  const int column = threadIdx.x * 4;

  const float4* in =
      reinterpret_cast<const float4*>(&x[row * blockDim.x * 4 + column]);
  float4* out = reinterpret_cast<float4*>(&y[row * blockDim.x * 4 + column]);

  float4 vector;
  switch (cache_op) {
    case CacheOp::Global:
      vector = __ldcg(in);
      break;
    case CacheOp::Streaming:
      vector = __ldcs(in);
      break;
  }

  vector.x *= alpha;
  vector.y *= alpha;
  vector.z *= alpha;
  vector.w *= alpha;

  switch (cache_op) {
    case CacheOp::Global:
      __stcg(out, vector);
      break;
    case CacheOp::Streaming:
      __stcs(out, vector);
      break;
  }
}

int main(int argc, char* argv[]) {
  constexpr int kRows = 204800;
  constexpr int kColumns = 512;
  constexpr int kSize = kRows * kColumns;

  constexpr float alpha = 2.0f;
  std::vector<float> host_x(kSize);
  for (int i = 0; i < kSize; i++) {
    host_x[i] = i;
  }
  std::vector<float> host_y(kSize);

  L2CacheFlusher l2_cache_flusher;

  cudaStream_t stream;
  CHECK_CUDA_ERROR(cudaStreamCreate(&stream));

  float* device_x;
  float* device_y;
  CHECK_CUDA_ERROR(cudaMallocAsync(&device_x, kSize * sizeof(float), stream));
  CHECK_CUDA_ERROR(cudaMallocAsync(&device_y, kSize * sizeof(float), stream));

  CHECK_CUDA_ERROR(cudaMemcpyAsync(
      device_x,
      host_x.data(),
      kSize * sizeof(float),
      cudaMemcpyHostToDevice,
      stream));

  constexpr int kIterations = 3;
  for (int i = 0; i < kIterations; i++) {
    l2_cache_flusher.Flush(stream);
    Axpy<CacheOp::Global>
        <<<kRows, kColumns / 4, 0, stream>>>(alpha, device_x, device_y);
  }
  // Switching to __ldcs and __stcs slows the kernel down from 501us to 534us.
  for (int i = 0; i < kIterations; i++) {
    l2_cache_flusher.Flush(stream);
    Axpy<CacheOp::Streaming>
        <<<kRows, kColumns / 4, 0, stream>>>(alpha, device_x, device_y);
  }

  CHECK_CUDA_ERROR(cudaMemcpyAsync(
      host_y.data(),
      device_y,
      kSize * sizeof(float),
      cudaMemcpyDeviceToHost,
      stream));

  CHECK_CUDA_ERROR(cudaFreeAsync(device_x, stream));
  CHECK_CUDA_ERROR(cudaFreeAsync(device_y, stream));
  CHECK_CUDA_ERROR(cudaStreamSynchronize(stream));
  CHECK_CUDA_ERROR(cudaStreamDestroy(stream));

  // Compare actual and expected.
  for (int i = 0; i < kSize; ++i) {
    const float actual = host_y[i];
    const float expected = host_x[i] * alpha;
    if (fabs(actual - expected) > 1e-5) {
      std::cerr << "Mismatch at index " << i << ": expected = " << expected
                << ", actual = " << actual << std::endl;
      abort();
    }
  }
  std::cout << "All tests passed." << std::endl;

  return 0;
}
