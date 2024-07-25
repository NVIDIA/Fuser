// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on

#include <c10/cuda/CUDAStream.h>
#include <torch/torch.h>
#include <iostream>
#include <vector>

using namespace c10::cuda;

// Perform matrix multiplication I times
int main(int argc, char* argv[]) {
  // Parse command-line arguments
  if (argc != 6) {
    std::cerr << "Usage: " << argv[0]
              << " LOG2_M LOG2_N LOG2_K use_stream use_matmul_out" << std::endl;
    std::cerr << "use_stream: 0 for default stream, 1 to use multiple streams"
              << std::endl;
    std::cerr
        << "use_matmul_out: 0 for using at::matmul, 1 for using at::matmul_out with preallocated output"
        << std::endl;
    return 1;
  }
  int64_t M = std::pow(2, std::stoi(argv[1]));
  int64_t N = std::pow(2, std::stoi(argv[2]));
  int64_t K = std::pow(2, std::stoi(argv[3]));
  bool use_stream = std::stoi(argv[4]) != 0;
  bool use_matmul_out = std::stoi(argv[5]) != 0;

  constexpr int I = 8; // number of iterations

  std::cout << "M=" << M << std::endl;
  std::cout << "N=" << N << std::endl;
  std::cout << "K=" << K << std::endl;
  std::cout << "use_stream=" << use_stream << std::endl;
  std::cout << "use_matmul_out=" << use_matmul_out << std::endl;

  torch::Device device(torch::kCUDA);
  // input tensors
  std::vector<at::Tensor> mat1;
  std::vector<at::Tensor> mat2;
  // output tensors
  std::vector<at::Tensor> results;
  for (int i = 0; i < I; ++i) {
    mat1.push_back(torch::rand({M, K}, device));
    mat2.push_back(torch::rand({K, N}, device));
    if (use_matmul_out) {
      results.push_back(torch::empty({M, N}, device));
    }
  }

  // Streams init
  std::vector<CUDAStream> streams;
  if (use_stream) {
    // Create I CUDA streams
    for (int i = 0; i < I; ++i) {
      streams.push_back(getStreamFromPool(/*isHighPriority=*/false));
    }
  }

  // Matmul execution
  for (int i = 0; i < I; ++i) {
    if (use_stream) {
      setCurrentCUDAStream(streams[i]);
    }
    if (use_matmul_out) {
      torch::matmul_out(results[i], mat1[i], mat2[i]);
    } else {
      results.push_back(torch::matmul(mat1[i], mat2[i]));
    }
  }

  // Stream sync
  if (use_stream) {
    for (int i = 0; i < I; ++i) {
      streams[i].synchronize();
    }
  } else {
    torch::cuda::synchronize();
  }

  // "Validation"
  for (int i = 0; i < I; ++i) {
    assert(results.at(i).numel()>0);
  }

  return 0;
}
