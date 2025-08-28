#include <iomanip>
#include <iostream>
#include <vector>
#include "templated_kernel.cuh"

// Mock implementations for demonstration (normally these would be from nvfuser
// runtime)
__device__ __e2m1 __float2e2m1(float f) {
  // Simplified conversion - in reality this would use proper FP4 conversion
  __e2m1 result;
  result.data = static_cast<uint8_t>(f * 15.0f) & 0x0F; // 4-bit value
  return result;
}

__device__ __e4m3 __float2e4m3(float f) {
  // Simplified conversion - in reality this would use proper FP8 conversion
  return __e4m3(static_cast<uint8_t>(f * 255.0f));
}

void print_separator(const std::string& title) {
  std::cout << "\n" << std::string(50, '=') << std::endl;
  std::cout << title << std::endl;
  std::cout << std::string(50, '=') << std::endl;
}

int main() {
  print_separator("Templated CUDA Kernel Example");

  // Test parameters
  const int num_rows = 128;
  const int inner_dim = 16;
  const int total_elements = num_rows * inner_dim;

  std::cout << "Test configuration:" << std::endl;
  std::cout << "  Number of rows: " << num_rows << std::endl;
  std::cout << "  Inner dimension: " << inner_dim << " (fixed)" << std::endl;
  std::cout << "  Total elements: " << total_elements << std::endl;

  // Create test input data as 2D array
  // Using a dynamically allocated 2D array
  float(*input_data)[16] = new float[num_rows][16];
  for (int i = 0; i < num_rows; i++) {
    for (int j = 0; j < 16; j++) {
      input_data[i][j] = static_cast<float>((i * 16 + j) % 1000) /
          1000.0f; // Values 0.0 to 0.999
    }
  }

  std::cout << "\nFirst 10 input values:" << std::endl;
  for (int i = 0; i < 10; i++) {
    int row = i / 16;
    int col = i % 16;
    std::cout << "  input[" << row << "][" << col << "] = " << std::fixed
              << std::setprecision(3) << input_data[row][col] << std::endl;
  }

  // Test different kernel configurations (BLOCK_DIM_X must be 4)
  print_separator("Testing BLOCK_DIM_X=4, BLOCK_DIM_Y=1");

  __e2m1* output_e2m1 = nullptr;
  __e4m3* output_e4m3 = nullptr;

  cudaError_t err =
      LAUNCH_KERNEL_4x1(input_data, num_rows, &output_e2m1, &output_e4m3, 0);

  if (err == cudaSuccess) {
    std::cout << "Kernel execution successful!" << std::endl;

    std::cout << "\nFirst 10 conversion results:" << std::endl;
    for (int i = 0; i < 10; i++) {
      int row = i / 16;
      int col = i % 16;
      std::cout << "  " << std::fixed << std::setprecision(3)
                << input_data[row][col] << " -> E2M1: " << std::hex << "0x"
                << (int)output_e2m1[i].data << ", E4M3: 0x"
                << (int)output_e4m3[i].raw() << std::dec << std::endl;
    }

    delete[] output_e2m1;
    delete[] output_e4m3;
  } else {
    std::cerr << "Kernel execution failed: " << cudaGetErrorString(err)
              << std::endl;
  }

  print_separator("Testing BLOCK_DIM_X=4, BLOCK_DIM_Y=4");

  output_e2m1 = nullptr;
  output_e4m3 = nullptr;

  err = LAUNCH_KERNEL_4x4(input_data, num_rows, &output_e2m1, &output_e4m3, 0);

  if (err == cudaSuccess) {
    std::cout << "Kernel execution successful!" << std::endl;
    std::cout << "Block configuration: 4x4 = 16 threads per block" << std::endl;

    delete[] output_e2m1;
    delete[] output_e4m3;
  } else {
    std::cerr << "Kernel execution failed: " << cudaGetErrorString(err)
              << std::endl;
  }

  print_separator("Testing BLOCK_DIM_X=4, BLOCK_DIM_Y=8");

  output_e2m1 = nullptr;
  output_e4m3 = nullptr;

  err = LAUNCH_KERNEL_4x8(input_data, num_rows, &output_e2m1, &output_e4m3, 0);

  if (err == cudaSuccess) {
    std::cout << "Kernel execution successful!" << std::endl;
    std::cout << "Block configuration: 4x8 = 32 threads per block" << std::endl;

    delete[] output_e2m1;
    delete[] output_e4m3;
  } else {
    std::cerr << "Kernel execution failed: " << cudaGetErrorString(err)
              << std::endl;
  }

  // Test direct kernel launch function
  print_separator("Testing Direct Kernel Launch");

  float(*d_input)[16];
  __e2m1* d_output_e2m1;
  __e4m3* d_output_e4m3;

  // Allocate device memory
  cudaMalloc(&d_input, total_elements * sizeof(float));
  cudaMalloc(&d_output_e2m1, total_elements * sizeof(__e2m1));
  cudaMalloc(&d_output_e4m3, total_elements * sizeof(__e4m3));

  // Copy input to device
  cudaMemcpy(
      d_input,
      input_data,
      total_elements * sizeof(float),
      cudaMemcpyHostToDevice);

  // Launch kernel directly with BLOCK_DIM_X=4
  err = launch_float_to_fp_kernel<4, 16>(
      d_input, d_output_e2m1, d_output_e4m3, num_rows);

  if (err == cudaSuccess) {
    std::cout << "Direct kernel launch successful!" << std::endl;
    std::cout << "Block configuration: 4x16 = 64 threads per block"
              << std::endl;
    std::cout
        << "Vectorized loads: 4 threads × 4 elements each = 16 elements per row"
        << std::endl;
  } else {
    std::cerr << "Direct kernel launch failed: " << cudaGetErrorString(err)
              << std::endl;
  }

  // Cleanup
  cudaFree(d_input);
  cudaFree(d_output_e2m1);
  cudaFree(d_output_e4m3);

  print_separator("Summary");
  std::cout << "Vectorized Kernel Template parameters tested:" << std::endl;
  std::cout
      << "  BLOCK_DIM_X=4, BLOCK_DIM_Y=1   (4 threads/block, 1 row per block)"
      << std::endl;
  std::cout
      << "  BLOCK_DIM_X=4, BLOCK_DIM_Y=4   (16 threads/block, 4 rows per block)"
      << std::endl;
  std::cout
      << "  BLOCK_DIM_X=4, BLOCK_DIM_Y=8   (32 threads/block, 8 rows per block)"
      << std::endl;
  std::cout << "  BLOCK_DIM_X=4, BLOCK_DIM_Y=16  (64 threads/block, 16 rows "
               "per block)"
            << std::endl;
  std::cout << "\nVectorized loading pattern:" << std::endl;
  std::cout << "  Thread 0: loads elements 0-3   of each row" << std::endl;
  std::cout << "  Thread 1: loads elements 4-7   of each row" << std::endl;
  std::cout << "  Thread 2: loads elements 8-11  of each row" << std::endl;
  std::cout << "  Thread 3: loads elements 12-15 of each row" << std::endl;
  std::cout << "\nInput array layout: " << num_rows
            << " rows × 16 columns (fixed)" << std::endl;
  std::cout << "Output types: __e2m1 (FP4) and __e4m3 (FP8)" << std::endl;

  // Cleanup
  delete[] input_data;

  return 0;
}
