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

void print_separator(const std::string& title) {
  std::cout << "\n" << std::string(50, '=') << std::endl;
  std::cout << title << std::endl;
  std::cout << std::string(50, '=') << std::endl;
}

// Host function to convert __e4m3 to float
float e4m3_to_float(__e4m3 e4m3_val) {
  uint8_t data = e4m3_val.raw();

  // Handle zero case
  if (data == 0)
    return 0.0f;

  // Extract sign, exponent, and mantissa
  uint8_t sign = (data >> 7) & 0x1; // 1 bit
  uint8_t exp = (data >> 3) & 0xF; // 4 bits
  uint8_t mant = data & 0x7; // 3 bits

  float result;

  if (exp == 0) {
    // Subnormal numbers
    result = (1.0f + mant / 8.0f) * powf(2.0f, -6.0f);
  } else if (exp == 15) {
    // Infinity or NaN
    if (mant == 0) {
      result = INFINITY;
    } else {
      result = NAN;
    }
  } else {
    // Normal numbers
    result = (1.0f + mant / 8.0f) * powf(2.0f, exp - 7.0f);
  }

  return sign ? -result : result;
}

int main() {
  print_separator("Templated CUDA Kernel Example");

  // Test parameters
  const int inner_dim = 16;
  const int total_elements = 2 * 16 + 8;
  const int num_rows = (total_elements + inner_dim - 1) / inner_dim;

  std::cout << "Test configuration:" << std::endl;
  std::cout << "  Number of rows: " << num_rows << std::endl;
  std::cout << "  Inner dimension: " << inner_dim << " (fixed)" << std::endl;
  std::cout << "  Total elements: " << total_elements << std::endl;

  // Create test input data as 1D array
  // Using a dynamically allocated 1D array
  float* input_data = new float[total_elements];
  for (int i = 0; i < total_elements; i++) {
    input_data[i] = 1.0f; // Set all values to 1.0
  }

  std::cout << "\nFirst 10 input values:" << std::endl;
  for (int i = 0; i < 10; i++) {
    int row = i / 16;
    int col = i % 16;
    std::cout << "  input[" << row << "][" << col << "] = " << std::fixed
              << std::setprecision(3) << input_data[i] << std::endl;
  }

  // Test different kernel configurations (BLOCK_DIM_X must be 4)
  print_separator("Testing BLOCK_DIM_X=4, BLOCK_DIM_Y=1");

  __e2m1* output_e2m1 = nullptr;
  __e4m3* output_e4m3 = nullptr;

  cudaError_t err = LAUNCH_KERNEL_4x1(
      input_data, inner_dim, total_elements, &output_e2m1, &output_e4m3, 0);

  if (err == cudaSuccess) {
    std::cout << "Kernel execution successful!" << std::endl;

    std::cout << "\nFirst 10 conversion results:" << std::endl;
    for (int i = 0; i < total_elements; i += 2) {
      std::cout << "  input[" << i << "] = " << std::fixed
                << std::setprecision(10) << input_data[i / 2] << " -> E2M1: 0x"
                << std::hex << (output_e2m1[i / 2].data & 0x0F)
                << " (upper 4 bits: 0x"
                << ((output_e2m1[i / 2].data >> 4) & 0x0F) << ")" << std::dec;
      std::cout << std::endl;
    }
    // Convert and print E4M3 values
    std::cout << std::endl;
    // for (auto i=0; i<num_rows; i++){
    //   std::cout << "  output_block_scale[" << i << "] = " << std::fixed <<
    //   std::setprecision(5)
    //             << e4m3_to_float(output_e4m3[i]) << std::endl;
    // }

    std::cout << "\nFirst 32 E2M1 values (first 16 packed uint8 values):"
              << std::endl;
    for (int i = 0; i < 32; i++) {
      uint8_t packed_data = output_e2m1[i].data;
      uint8_t lower_4bits = packed_data & 0x0F;
      uint8_t upper_4bits = (packed_data >> 4) & 0x0F;
      std::cout << "  output_e2m1[" << i << "] = 0x" << std::hex
                << std::setfill('0') << std::setw(2)
                << static_cast<int>(packed_data) << " (lower: 0x"
                << static_cast<int>(lower_4bits) << ", upper: 0x"
                << static_cast<int>(upper_4bits) << ")" << std::dec
                << std::endl;
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

  err = LAUNCH_KERNEL_4x4(
      input_data, inner_dim, total_elements, &output_e2m1, &output_e4m3, 0);

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

  err = LAUNCH_KERNEL_4x8(
      input_data, inner_dim, total_elements, &output_e2m1, &output_e4m3, 0);

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

  float* d_input;
  __e2m1* d_output_e2m1;
  __e4m3* d_output_e4m3;

  // Allocate device memory
  cudaMalloc(&d_input, total_elements * sizeof(float));

  // Allocate with 4-byte alignment for __e2m1
  std::cout << "sizeof(__e2m1) = " << sizeof(__e2m1) << " bytes" << std::endl;
  size_t e2m1_size = ((total_elements + 1) / 2) * sizeof(__e2m1);
  size_t aligned_e2m1_size =
      (e2m1_size + 3) & ~3; // Round up to nearest 4-byte boundary
  std::cout << "Original e2m1_size = " << e2m1_size << " bytes" << std::endl;
  std::cout << "Aligned e2m1_size = " << aligned_e2m1_size << " bytes"
            << std::endl;
  cudaMalloc(&d_output_e2m1, aligned_e2m1_size);

  cudaMalloc(&d_output_e4m3, total_elements * sizeof(__e4m3));

  // Copy input to device
  cudaMemcpy(
      d_input,
      input_data,
      total_elements * sizeof(float),
      cudaMemcpyHostToDevice);

  // Launch kernel directly with BLOCK_DIM_X=4
  err = launch_float_to_fp_kernel<4, 16>(
      d_input, d_output_e2m1, d_output_e4m3, num_rows, total_elements);

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
      << "BLOCK_DIM_X=4, BLOCK_DIM_Y=1   (4 threads/block, 1 row per block)"
      << std::endl;
  std::cout
      << "BLOCK_DIM_X=4, BLOCK_DIM_Y=4   (16 threads/block, 4 rows per block)"
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
