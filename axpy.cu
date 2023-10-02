#include <iostream>
#include <vector>

#define CHECK_CUDA_ERROR(error) CheckCudaError((error), __FILE__, __LINE__)

inline void CheckCudaError(
    cudaError_t error,
    const char* file,
    int line,
    bool abort = true) {
  if (error != cudaSuccess) {
    fprintf(
        stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(error), file, line);
    if (abort) {
      exit(error);
    }
  }
}

constexpr int kRows = 204800;
constexpr int kColumns = 512;

__global__ void Axpy(const float alpha, const float* x, float* y) {
  const int row = blockIdx.x;
  const int column = threadIdx.x * 4;
  const float* in = &x[row * kColumns + column];
  float* out = &y[row * kColumns + column];
  float4 vector = *reinterpret_cast<const float4*>(in);
  vector.x *= alpha;
  vector.y *= alpha;
  vector.z *= alpha;
  vector.w *= alpha;
  *reinterpret_cast<float4*>(out) = vector;
}

int main(int argc, char* argv[]) {
  constexpr int kSize = kRows * kColumns;

  constexpr float alpha = 2.0f;
  std::vector<float> host_x(kSize);
  for (int i = 0; i < kSize; i++) {
    host_x[i] = i;
  }
  std::vector<float> host_y(kSize);

  // Copy input data to device.
  float* device_x;
  float* device_y;
  CHECK_CUDA_ERROR(cudaMalloc(&device_x, kSize * sizeof(float)));
  CHECK_CUDA_ERROR(cudaMalloc(&device_y, kSize * sizeof(float)));

  cudaStream_t stream;
  CHECK_CUDA_ERROR(cudaStreamCreate(&stream));
  CHECK_CUDA_ERROR(cudaMemcpyAsync(
      device_x,
      host_x.data(),
      kSize * sizeof(float),
      cudaMemcpyHostToDevice,
      stream));
  Axpy<<<kRows, kColumns / 4, 0, stream>>>(alpha, device_x, device_y);

  CHECK_CUDA_ERROR(cudaMemcpyAsync(
      host_y.data(),
      device_y,
      kSize * sizeof(float),
      cudaMemcpyDeviceToHost,
      stream));
  CHECK_CUDA_ERROR(cudaStreamSynchronize(stream));
  CHECK_CUDA_ERROR(cudaStreamDestroy(stream));

  // Print the results.
  for (int i = 0; i < kSize; ++i) {
    const float actual = host_y[i];
    const float expected = host_x[i] * alpha;
    if (fabs(actual - expected) > 1e-5) {
      std::cerr << "Mismatch at index " << i << ": expected = " << expected
                << ", actual = " << actual << std::endl;
      abort();
    }
  }

  return 0;
}
