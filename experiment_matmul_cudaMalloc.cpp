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

enum class StreamMode {
  NoStreams,
  PostOnDifferentStreams,
  AllocateAndPostOnDifferentStreams,
  Invalid
};

enum class ComputeMode { Matmul, MatmulOut, Unfused, Invalid };

std::ostream& operator<<(std::ostream& os, const StreamMode& mode) {
  switch (mode) {
    case StreamMode::NoStreams:
      os << "no_streams";
      break;
    case StreamMode::PostOnDifferentStreams:
      os << "post_on_different_streams";
      break;
    case StreamMode::AllocateAndPostOnDifferentStreams:
      os << "allocate_and_post_on_different_streams";
      break;
    default:
      os << "invalid";
      break;
  }
  return os;
}

std::ostream& operator<<(std::ostream& os, const ComputeMode& mode) {
  switch (mode) {
    case ComputeMode::Matmul:
      os << "matmul";
      break;
    case ComputeMode::MatmulOut:
      os << "matmul_out";
      break;
    case ComputeMode::Unfused:
      os << "unfused";
      break;
    default:
      os << "invalid";
      break;
  }
  return os;
}

// Function to convert a string to a StreamMode
StreamMode getStreamMode(const std::string& mode) {
  if (mode == "no_streams") {
    return StreamMode::NoStreams;
  } else if (mode == "post_on_different_streams") {
    return StreamMode::PostOnDifferentStreams;
  } else if (mode == "allocate_and_post_on_different_streams") {
    return StreamMode::AllocateAndPostOnDifferentStreams;
  } else {
    return StreamMode::Invalid;
  }
}

ComputeMode getComputeMode(const std::string& mode) {
  if (mode == "matmul") {
    return ComputeMode::Matmul;
  } else if (mode == "matmul_out") {
    return ComputeMode::MatmulOut;
  } else if (mode == "unfused") {
    return ComputeMode::Unfused;
  } else {
    return ComputeMode::Invalid;
  }
}

// Perform matrix multiplication I times
int main(int argc, char* argv[]) {
  if (argc != 6) {
    std::cerr
        << "Usage: " << argv[0]
        << " LOG2_M LOG2_N LOG2_K {no_streams|post_on_different_streams|allocate_and_post_on_different_streams} {matmul|matmul_out|unfused}"
        << std::endl;
    return 1;
  }
  int64_t M = std::pow(2, std::stoi(argv[1]));
  int64_t N = std::pow(2, std::stoi(argv[2]));
  int64_t K = std::pow(2, std::stoi(argv[3]));

  StreamMode stream_mode = getStreamMode(argv[4]);
  if (stream_mode == StreamMode::Invalid) {
    std::cerr << "Invalid stream mode: " << argv[4] << std::endl;
    std::cerr
        << "Valid stream modes: no_streams, post_on_different_streams, allocate_and_post_on_different_streams"
        << std::endl;
    return 1;
  }

  ComputeMode compute_mode = getComputeMode(argv[5]);
  if (compute_mode == ComputeMode::Invalid) {
    std::cerr << "Invalid compute mode: " << argv[5] << std::endl;
    std::cerr << "Valid compute modes: matmul, matmul_out, unfused"
              << std::endl;
    return 1;
  }

  constexpr int I = 4; // number of iterations changing stream
  constexpr int J = 4; // number of iterations reusing streams and buffer
  constexpr int L = 2; // number of iterations reusing streams

  std::cout << "M=" << M << std::endl;
  std::cout << "N=" << N << std::endl;
  std::cout << "K=" << K << std::endl;
  std::cout << "stream_mode=" << stream_mode << std::endl;
  std::cout << "compute_mode=" << compute_mode << std::endl;
  std::cout << "PYTORCH_NO_CUDA_MEMORY_CACHING="
            << std::getenv("PYTORCH_NO_CUDA_MEMORY_CACHING") << std::endl;

  // Streams init
  std::vector<CUDAStream> streams;
  const bool use_streams = stream_mode == StreamMode::PostOnDifferentStreams ||
      stream_mode == StreamMode::AllocateAndPostOnDifferentStreams;
  if (use_streams) {
    // Create I CUDA streams
    for (int i = 0; i < I; ++i) {
      streams.push_back(getStreamFromPool(/*isHighPriority=*/false));
    }
  }

  torch::Device device(torch::kCUDA);
  for (int l = 0; l < L; ++l) {
    // input tensors
    std::vector<at::Tensor> mat1;
    std::vector<at::Tensor> mat2;
    // output tensors
    std::vector<at::Tensor> unreduced_results;
    std::vector<at::Tensor> results;
    for (int i = 0; i < I; ++i) {
      if (stream_mode == StreamMode::AllocateAndPostOnDifferentStreams) {
        setCurrentCUDAStream(streams[i]);
      }
      mat1.push_back(torch::rand({M, K}, device));
      mat2.push_back(torch::rand({K, N}, device));
      if (compute_mode == ComputeMode::Unfused) {
        unreduced_results.push_back(torch::empty({M, K, N}, device));
      }
      if (compute_mode == ComputeMode::MatmulOut ||
          compute_mode == ComputeMode::Unfused) {
        results.push_back(torch::empty({M, N}, device));
      }
    }

    cudaDeviceSynchronize();

    // Matmul execution
    for (int j = 0; j < J; ++j) {
      for (int i = 0; i < I; ++i) {
        if (use_streams) {
          setCurrentCUDAStream(streams[i]);
        }
        switch (compute_mode) {
          case ComputeMode::Matmul:
            results.push_back(torch::matmul(mat1[i], mat2[i]));
            break;
          case ComputeMode::MatmulOut:
            torch::matmul_out(results[i], mat1[i], mat2[i]);
            break;
          case ComputeMode::Unfused:
            at::mul_out(
                unreduced_results[i], mat1[i].unsqueeze(-1), mat2[i].unsqueeze(-3));
            at::sum_out(results[i], unreduced_results[i], {-2});
            break;
          default:
            assert(false);
            break;
        }
      }

      // Stream sync
      if (use_streams) {
        for (int i = 0; i < I; ++i) {
          streams[i].synchronize();
        }
      } else {
        torch::cuda::synchronize();
      }

      // "Validation"
      for (int i = 0; i < I; ++i) {
        assert(results.at(i).numel() > 0);
      }
    }
  }

  return 0;
}
