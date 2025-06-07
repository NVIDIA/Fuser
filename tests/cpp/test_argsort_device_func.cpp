// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on

#include <gtest/gtest.h>

#include <tests/cpp/utils.h>
#include <tests/cpp/validator.h>
#include <tests/cpp/argsort_test_helper.h>

#include <vector>
#include <algorithm>
#include <random>
#include <ATen/cuda/CUDAContext.h>

namespace nvfuser {

using ArgSortDeviceFuncTest = NVFuserTest;

// Parameterized test fixture for comprehensive validation
using ArgSortComprehensiveTest = NVFuserFixtureParamTest<std::pair<int, int>>;

// Helper function to validate sorting correctness
template<typename DataT>
bool validate_argsort_order(const std::vector<DataT>& input_data, 
                           const std::vector<nvfuser_index_t>& indices,
                           bool descending = false) {
    
    int n = input_data.size();
    
    // Check valid range
    for (int i = 0; i < n; i++) {
        if (indices[i] < 0 || indices[i] >= n) {
            return false;
        }
    }
    
    // Check permutation
    std::vector<bool> used(n, false);
    for (int i = 0; i < n; i++) {
        if (used[indices[i]]) {
            return false;
        }
        used[indices[i]] = true;
    }
    
    // Check sorting order
    for (int i = 1; i < n; i++) {
        DataT prev_val = input_data[indices[i-1]];
        DataT curr_val = input_data[indices[i]];
        
        if (descending) {
            if (curr_val > prev_val) {
                return false;
            }
        } else {
            if (curr_val < prev_val) {
                return false;
            }
        }
    }
    
    return true;
}

// Basic functionality test
TEST_F(ArgSortDeviceFuncTest, BasicArgsortFloat) {
    const int BLOCK_SIZE = 4;
    const int ITEMS_PER_THREAD = 2;
    const int total_elements = BLOCK_SIZE * ITEMS_PER_THREAD;
    
    std::vector<float> test_data = {5.0f, 2.0f, 8.0f, 1.0f, 7.0f, 3.0f, 6.0f, 4.0f};
    
    auto input_tensor = at::tensor(test_data, at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0));
    auto output_tensor = at::empty({total_elements}, at::TensorOptions().dtype(at::kLong).device(at::kCUDA, 0));
    
    // Test ascending
    launch_basic_argsort_test_kernel<float>(
        at::cuda::getCurrentCUDAStream(),
        input_tensor.data_ptr<float>(), 
        output_tensor.data_ptr<nvfuser_index_t>(), 
        BLOCK_SIZE, ITEMS_PER_THREAD, false);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    
    auto output_cpu = output_tensor.cpu();
    std::vector<nvfuser_index_t> output_indices(output_cpu.data_ptr<nvfuser_index_t>(), 
                                               output_cpu.data_ptr<nvfuser_index_t>() + total_elements);
    
    EXPECT_TRUE(validate_argsort_order(test_data, output_indices, false));
    
    // Test descending
    launch_basic_argsort_test_kernel<float>(
        at::cuda::getCurrentCUDAStream(),
        input_tensor.data_ptr<float>(), 
        output_tensor.data_ptr<nvfuser_index_t>(), 
        BLOCK_SIZE, ITEMS_PER_THREAD, true);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    
    output_cpu = output_tensor.cpu();
    output_indices.assign(output_cpu.data_ptr<nvfuser_index_t>(), 
                         output_cpu.data_ptr<nvfuser_index_t>() + total_elements);
    EXPECT_TRUE(validate_argsort_order(test_data, output_indices, true));
}

// Data type support test
TEST_F(ArgSortDeviceFuncTest, DataTypeSupport) {
    const int BLOCK_SIZE = 4;
    const int ITEMS_PER_THREAD = 2;
    const int total_elements = BLOCK_SIZE * ITEMS_PER_THREAD;
    
    // Test double
    {
        std::vector<double> test_data = {5.5, 2.1, 8.3, 1.7, 7.2, 3.9, 6.4, 4.8};
        
        auto input_tensor = at::tensor(test_data, at::TensorOptions().dtype(at::kDouble).device(at::kCUDA, 0));
        auto output_tensor = at::empty({total_elements}, at::TensorOptions().dtype(at::kLong).device(at::kCUDA, 0));
        
        launch_basic_argsort_test_kernel<double>(
            at::cuda::getCurrentCUDAStream(),
            input_tensor.data_ptr<double>(), 
            output_tensor.data_ptr<nvfuser_index_t>(), 
            BLOCK_SIZE, ITEMS_PER_THREAD, false);
        ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
        
        auto output_cpu = output_tensor.cpu();
        std::vector<nvfuser_index_t> output_indices(output_cpu.data_ptr<nvfuser_index_t>(), 
                                                   output_cpu.data_ptr<nvfuser_index_t>() + total_elements);
        
        EXPECT_TRUE(validate_argsort_order(test_data, output_indices, false));
    }
    
    // Test int
    {
        std::vector<int> test_data = {5, 2, 8, 1, 7, 3, 6, 4};
        
        auto input_tensor = at::tensor(test_data, at::TensorOptions().dtype(at::kInt).device(at::kCUDA, 0));
        auto output_tensor = at::empty({total_elements}, at::TensorOptions().dtype(at::kLong).device(at::kCUDA, 0));
        
        launch_basic_argsort_test_kernel<int>(
            at::cuda::getCurrentCUDAStream(),
            input_tensor.data_ptr<int>(), 
            output_tensor.data_ptr<nvfuser_index_t>(), 
            BLOCK_SIZE, ITEMS_PER_THREAD, false);
        ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
        
        auto output_cpu = output_tensor.cpu();
        std::vector<nvfuser_index_t> output_indices(output_cpu.data_ptr<nvfuser_index_t>(), 
                                                   output_cpu.data_ptr<nvfuser_index_t>() + total_elements);
        
        EXPECT_TRUE(validate_argsort_order(test_data, output_indices, false));
    }
    
    // Test int64_t
    {
        std::vector<int64_t> test_data = {5L, 2L, 8L, 1L, 7L, 3L, 6L, 4L};
        
        auto input_tensor = at::tensor(test_data, at::TensorOptions().dtype(at::kLong).device(at::kCUDA, 0));
        auto output_tensor = at::empty({total_elements}, at::TensorOptions().dtype(at::kLong).device(at::kCUDA, 0));
        
        launch_basic_argsort_test_kernel<int64_t>(
            at::cuda::getCurrentCUDAStream(),
            input_tensor.data_ptr<int64_t>(), 
            output_tensor.data_ptr<nvfuser_index_t>(), 
            BLOCK_SIZE, ITEMS_PER_THREAD, false);
        ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
        
        auto output_cpu = output_tensor.cpu();
        std::vector<nvfuser_index_t> output_indices(output_cpu.data_ptr<nvfuser_index_t>(), 
                                                   output_cpu.data_ptr<nvfuser_index_t>() + total_elements);
        
        EXPECT_TRUE(validate_argsort_order(test_data, output_indices, false));
    }
}

// Multi-dimensional block tests
TEST_F(ArgSortDeviceFuncTest, MultiDimensionalBlocks) {
    const int ITEMS_PER_THREAD = 2;
    
    // Test 2D block: 4x2x1 (8 threads total)
    {
        const int total_elements = 8 * ITEMS_PER_THREAD;
        std::vector<float> test_data = {5.0f, 2.0f, 8.0f, 1.0f, 7.0f, 3.0f, 6.0f, 4.0f,
                                       9.0f, 0.5f, 3.5f, 7.5f, 2.5f, 8.5f, 1.5f, 6.5f};
        
        auto input_tensor = at::tensor(test_data, at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0));
        auto output_tensor = at::empty({total_elements}, at::TensorOptions().dtype(at::kLong).device(at::kCUDA, 0));
        
        launch_multi_dim_2d_argsort_test_kernel<float>(
            at::cuda::getCurrentCUDAStream(),
            input_tensor.data_ptr<float>(), 
            output_tensor.data_ptr<nvfuser_index_t>(), 
            ITEMS_PER_THREAD, false);
        ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
        
        auto output_cpu = output_tensor.cpu();
        std::vector<nvfuser_index_t> output_indices(output_cpu.data_ptr<nvfuser_index_t>(), 
                                                   output_cpu.data_ptr<nvfuser_index_t>() + total_elements);
        
        EXPECT_TRUE(validate_argsort_order(test_data, output_indices, false));
    }
    
    // Test 3D block: 2x2x2 (8 threads total)
    {
        const int total_elements = 8 * ITEMS_PER_THREAD;
        std::vector<float> test_data = {5.0f, 2.0f, 8.0f, 1.0f, 7.0f, 3.0f, 6.0f, 4.0f,
                                       9.0f, 0.5f, 3.5f, 7.5f, 2.5f, 8.5f, 1.5f, 6.5f};
        
        auto input_tensor = at::tensor(test_data, at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0));
        auto output_tensor = at::empty({total_elements}, at::TensorOptions().dtype(at::kLong).device(at::kCUDA, 0));
        
        launch_multi_dim_3d_argsort_test_kernel<float>(
            at::cuda::getCurrentCUDAStream(),
            input_tensor.data_ptr<float>(), 
            output_tensor.data_ptr<nvfuser_index_t>(), 
            ITEMS_PER_THREAD, false);
        ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
        
        auto output_cpu = output_tensor.cpu();
        std::vector<nvfuser_index_t> output_indices(output_cpu.data_ptr<nvfuser_index_t>(), 
                                                   output_cpu.data_ptr<nvfuser_index_t>() + total_elements);
        
        EXPECT_TRUE(validate_argsort_order(test_data, output_indices, false));
    }
}

// BFloat16 support test
TEST_F(ArgSortDeviceFuncTest, BFloat16Support) {
    const int ITEMS_PER_THREAD = 2;
    const int total_elements = 4 * ITEMS_PER_THREAD;
    
    // Test data as floats (will convert to bfloat16)
    std::vector<float> test_data_float = {5.0f, 2.0f, 8.0f, 1.0f, 7.0f, 3.0f, 6.0f, 4.0f};
    
    // Create tensors and convert float to bfloat16  
    auto temp_float_tensor = at::tensor(test_data_float, at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0));
    auto input_tensor = at::empty({total_elements}, at::TensorOptions().dtype(at::kBFloat16).device(at::kCUDA, 0));
    auto output_tensor = at::empty({total_elements}, at::TensorOptions().dtype(at::kLong).device(at::kCUDA, 0));
    
    // Launch conversion kernel
    launch_convert_float_to_bfloat16(
        at::cuda::getCurrentCUDAStream(),
        temp_float_tensor.data_ptr<float>(), 
        reinterpret_cast<__nv_bfloat16*>(input_tensor.data_ptr()), 
        total_elements);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    
    // Launch argsort kernel
    launch_bfloat16_argsort_test_kernel(
        at::cuda::getCurrentCUDAStream(),
        reinterpret_cast<__nv_bfloat16*>(input_tensor.data_ptr()), 
        output_tensor.data_ptr<nvfuser_index_t>(), 
        ITEMS_PER_THREAD, false);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    
    // Copy results back
    auto output_cpu = output_tensor.cpu();
    std::vector<nvfuser_index_t> output_indices(output_cpu.data_ptr<nvfuser_index_t>(), 
                                               output_cpu.data_ptr<nvfuser_index_t>() + total_elements);
    
    // Validate against original float data (since bfloat16 conversion should preserve order)
    EXPECT_TRUE(validate_argsort_order(test_data_float, output_indices, false));
}

// Parameterized comprehensive validation test
TEST_P(ArgSortComprehensiveTest, ComprehensiveValidation) {
    auto [BLOCK_SIZE, ITEMS_PER_THREAD] = GetParam();
    const int total_elements = BLOCK_SIZE * ITEMS_PER_THREAD;
    
    // Generate random test data using at::randn
    auto input_tensor = at::randn({total_elements}, at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0));
    auto output_tensor = at::empty({total_elements}, at::TensorOptions().dtype(at::kLong).device(at::kCUDA, 0));
    
    // Copy input data to CPU for validation
    auto input_cpu = input_tensor.cpu();
    std::vector<float> test_data(input_cpu.data_ptr<float>(), 
                                input_cpu.data_ptr<float>() + total_elements);
    
    // Test ascending
    launch_basic_argsort_test_kernel<float>(
        at::cuda::getCurrentCUDAStream(),
        input_tensor.data_ptr<float>(), 
        output_tensor.data_ptr<nvfuser_index_t>(), 
        BLOCK_SIZE, ITEMS_PER_THREAD, false);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    
    auto output_cpu = output_tensor.cpu();
    std::vector<nvfuser_index_t> ascending_indices(output_cpu.data_ptr<nvfuser_index_t>(), 
                                                  output_cpu.data_ptr<nvfuser_index_t>() + total_elements);
    
    EXPECT_TRUE(validate_argsort_order(test_data, ascending_indices, false));
    
    // Test descending
    launch_basic_argsort_test_kernel<float>(
        at::cuda::getCurrentCUDAStream(),
        input_tensor.data_ptr<float>(), 
        output_tensor.data_ptr<nvfuser_index_t>(), 
        BLOCK_SIZE, ITEMS_PER_THREAD, true);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    
    output_cpu = output_tensor.cpu();
    std::vector<nvfuser_index_t> descending_indices(output_cpu.data_ptr<nvfuser_index_t>(), 
                                                   output_cpu.data_ptr<nvfuser_index_t>() + total_elements);
    
    EXPECT_TRUE(validate_argsort_order(test_data, descending_indices, true));
}

// Instantiate parameterized tests
INSTANTIATE_TEST_SUITE_P(
    BlockSizeAndItemsPerThread,
    ArgSortComprehensiveTest,
    testing::Values(
        // Block size 32
        std::make_pair(32, 1), std::make_pair(32, 2), std::make_pair(32, 3), std::make_pair(32, 4), std::make_pair(32, 5),
        // Block size 64  
        std::make_pair(64, 1), std::make_pair(64, 2), std::make_pair(64, 3), std::make_pair(64, 4), std::make_pair(64, 5),
        // Block size 128
        std::make_pair(128, 1), std::make_pair(128, 2), std::make_pair(128, 3), std::make_pair(128, 4), std::make_pair(128, 5),
        // Block size 256
        std::make_pair(256, 1), std::make_pair(256, 2), std::make_pair(256, 3), std::make_pair(256, 4), std::make_pair(256, 5)
    ),
    [](const testing::TestParamInfo<std::pair<int, int>>& info) {
        return "BlockSize" + std::to_string(info.param.first) + "_ItemsPerThread" + std::to_string(info.param.second);
    });

} // namespace nvfuser 