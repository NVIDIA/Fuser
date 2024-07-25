#include <torch/torch.h>
#include <iostream>

int main(int argc, char* argv[]) {
    if (argc != 4) {
        std::cerr << "Usage: " << argv[0] << " M N K" << std::endl;
        return 1;
    }

    // Parse command-line arguments
    int M = std::stoi(argv[1]);
    int N = std::stoi(argv[2]);
    int K = std::stoi(argv[3]);

    // Initialize tensors with random values
    torch::Device device(torch::kCUDA);
    torch::Tensor mat1 = torch::rand({M, K}, device);
    torch::Tensor mat2 = torch::rand({K, N}, device);

    // Perform matrix multiplication
    torch::Tensor result = torch::matmul(mat1, mat2);

    // Output the result
    std::cout << "Matrix 1 (" << M << "x" << K << "):\n" << mat1 << "\n";
    std::cout << "Matrix 2 (" << K << "x" << N << "):\n" << mat2 << "\n";
    std::cout << "Resultant Matrix (" << M << "x" << N << "):\n" << result << "\n";

    return 0;
}
