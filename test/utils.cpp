// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#include <test/utils.h>

#include <c10/util/Exception.h>

#include <ops/all_ops.h>

#include <sstream>
#include <string_view>

namespace nvfuser {

int64_t prime_number(int64_t i) {
  static std::vector<int64_t> p{
      2,    3,    5,    7,    11,   13,   17,   19,   23,   29,   31,   37,
      41,   43,   47,   53,   59,   61,   67,   71,   73,   79,   83,   89,
      97,   101,  103,  107,  109,  113,  127,  131,  137,  139,  149,  151,
      157,  163,  167,  173,  179,  181,  191,  193,  197,  199,  211,  223,
      227,  229,  233,  239,  241,  251,  257,  263,  269,  271,  277,  281,
      283,  293,  307,  311,  313,  317,  331,  337,  347,  349,  353,  359,
      367,  373,  379,  383,  389,  397,  401,  409,  419,  421,  431,  433,
      439,  443,  449,  457,  461,  463,  467,  479,  487,  491,  499,  503,
      509,  521,  523,  541,  547,  557,  563,  569,  571,  577,  587,  593,
      599,  601,  607,  613,  617,  619,  631,  641,  643,  647,  653,  659,
      661,  673,  677,  683,  691,  701,  709,  719,  727,  733,  739,  743,
      751,  757,  761,  769,  773,  787,  797,  809,  811,  821,  823,  827,
      829,  839,  853,  857,  859,  863,  877,  881,  883,  887,  907,  911,
      919,  929,  937,  941,  947,  953,  967,  971,  977,  983,  991,  997,
      1009, 1013, 1019, 1021, 1031, 1033, 1039, 1049, 1051, 1061, 1063, 1069,
      1087, 1091, 1093, 1097, 1103, 1109, 1117, 1123, 1129, 1151, 1153, 1163,
      1171, 1181, 1187, 1193, 1201, 1213, 1217, 1223};
  return p.at(i);
}

void assertCUDAKernel(Fusion* fusion, const std::string& expected_kernel) {
  const std::string actual_kernel =
      "\n" + codegen::generateCudaKernel(GpuLower(fusion).kernel());
  if (expected_kernel.size() != actual_kernel.size() ||
      expected_kernel.compare(actual_kernel) != 0) {
    std::cerr
        << " Codegen mismatch, codegen possibly changed, or is incorrect. "
        << " \n ========= EXPECTED ========= \n"
        << expected_kernel << "\n========= ACTUAL ========== \n"
        << actual_kernel << "\n=================" << std::endl;
    auto it = std::mismatch(
        expected_kernel.begin(),
        expected_kernel.end(),
        actual_kernel.begin(),
        actual_kernel.end());
    std::string actual_mismatched_snippet(it.second, actual_kernel.end());
    actual_mismatched_snippet = actual_mismatched_snippet.substr(0, 10);
    std::string expected_mismatched_snippet(it.first, expected_kernel.end());
    expected_mismatched_snippet = expected_mismatched_snippet.substr(0, 10);
    std::cerr << "First mismatch found at: " << actual_mismatched_snippet
              << ", expected: " << expected_mismatched_snippet << std::endl;
    TORCH_CHECK(false);
  }
}

namespace sass {

// For SASS instruction definitions, see:
// https://docs.nvidia.com/cuda/cuda-binary-utilities/index.html#instruction-set-reference
//
// Some useful informations for Ampere/Ada:
//
// Instruction format:
//   (instruction) (destination) (source1), (source2) ...
//
// Registers:
// - RX for registers
// - URX for uniform registers
// - SRX for special system-controlled registers
// - PX for predicate registers
// - UPX for uniform predicate registers
// - c[X][Y] for constant memory

namespace {

// trim: remove spaces before and after the string view
// implementation borrowed from https://stackoverflow.com/a/17976541
inline std::string_view trim(const std::string_view& s) {
  auto wsfront = std::find_if_not(
      s.begin(), s.end(), [](int c) { return std::isspace(c); });
  auto wsback = std::find_if_not(s.rbegin(), s.rend(), [](int c) {
                  return std::isspace(c);
                }).base();
  return (
      wsback <= wsfront ? "" : s.substr(wsfront - s.begin(), wsback - wsfront));
}

// Copied from LLVM libcxx C++20 implementation of
// basic_string_view::starts_with
// https://github.com/llvm/llvm-project/blob/11d8f726d24d90c67e0e99aa8e9de48d17adb750/libcxx/include/string_view#L696-L697
bool starts_with(std::string_view self, std::string_view __s) noexcept {
  return self.size() >= __s.size() && self.compare(0, __s.size(), __s) == 0;
}

} // namespace

std::string Instruction::predicate() {
  if (str[0] == '@') {
    std::stringstream ss(str);
    char ignore_at = '\0';
    std::string result;
    ss >> ignore_at >> result;
    return result;
  }
  return {};
}

std::string Instruction::action() {
  std::string result;
  std::stringstream ss(str);
  if (str[0] == '@') {
    ss >> result;
  }
  std::getline(ss, result);
  result = trim(result);
  return result;
}

std::string Instruction::op() {
  std::stringstream ss(action());
  std::string result;
  ss >> result;
  return result;
}

std::string Instruction::opCode() {
  std::string result;
  for (auto i : op()) {
    if (i == '.') {
      return result;
    }
    result.push_back(i);
  }
  return result;
}

std::vector<std::string> Instruction::args() {
  std::stringstream ss(action());
  std::string all_args;
  ss >> all_args; // discard
  std::getline(ss, all_args);
  all_args = trim(all_args);
  std::vector<std::string> result;

  std::string_view args_view(all_args);
  while (!args_view.empty()) {
    auto comma_pos = args_view.find_first_of(',');
    auto token = args_view.substr(0, comma_pos);
    token = trim(token);
    result.emplace_back(token);

    args_view = (comma_pos != std::string_view::npos)
        ? args_view.substr(comma_pos + 1)
        : "";
  }
  return result;
}

std::vector<std::string> Instruction::modifiers() {
  std::vector<std::string> result;
  std::string current;
  bool found_opcode = false;
  for (auto i : op()) {
    if (i == '.') {
      if (found_opcode) {
        result.push_back(current);
      }
      found_opcode = true;
      current.clear();
      continue;
    }
    current.push_back(i);
  }
  if (found_opcode) {
    result.push_back(current);
  }
  return result;
}

std::string Container::toString() {
  std::stringstream ss;
  for (auto& [key, value] : attributes) {
    ss << "." << key << ":\t" << value << std::endl;
  }
  for (auto& i : code) {
    std::visit(
        [&ss](auto&& i) {
          using T = std::decay_t<decltype(i)>;
          if constexpr (std::is_same_v<Instruction, T>) {
            ss << i.str << std::endl;
          } else if constexpr (std::is_same_v<T, Label>) {
            ss << "." << i.name << ":" << std::endl;
          }
        },
        i);
  }
  return ss.str();
}

Container parse(const std::string& nvdisasm_output) {
  Container result;
  bool started = false;
  std::stringstream ss(nvdisasm_output);
  std::string header;
  for (std::string line; std::getline(ss, line);) {
    line = trim(line);
    if (line.empty() || starts_with(line, "//")) {
      continue;
    }
    if (started) {
      if (line[0] == '.') {
        std::stringstream ss(line);
        Label l;
        char ignore_dot = '\0';
        ss >> ignore_dot >> l.name;
        l.name.resize(l.name.size() - 1); // remove trailing :
        result.code.emplace_back(l);
      } else {
        Instruction i;
        std::stringstream ss(line);
        char ignore = '\0';
        // parse /*address*/
        ss >> ignore >> ignore >> std::hex >> i.address >> ignore >> ignore;
        std::getline(ss, i.str);
        i.str = trim(i.str);
        i.str.resize(i.str.size() - 1); // remove trailing ;
        i.str = trim(i.str);
        result.code.emplace_back(i);
      }
    } else {
      if (line == header) {
        started = true;
      } else if (line[0] == '.') {
        std::stringstream ss(line);
        std::string key, value;
        char ignore = '\0';
        ss >> ignore >> key >> value;
        result.attributes[key] = value;
        if (key == "global") {
          header = ".text." + value + ":";
        }
      }
    }
  }
  return result;
}

} // namespace sass

TensorView* matmulVolta(TensorView* a, TensorView* b, MatmulLayout layout) {
  TORCH_CHECK(
      a->nDims() == 2 && b->nDims() == 2, "only pure matmuls for these tests");
  // Here, we canonicalize the mma output as M, N, K, but the position of K does
  // not really matter. So the implicit transpose is only required for NN.
  TensorView *tv2 = nullptr, *tv0b = nullptr, *tv1b = nullptr;
  switch (layout) {
    case MatmulLayout::TT:
      tv0b = broadcast(a, {false, false, true});
      tv1b = broadcast(b, {true, false, false});
      tv2 = fusedMultiplySum(tv0b, tv1b, {1});
      // M, K, N -> M, N, K
      tv2->reorder({{1, -1}});
      tv2->commitLeafToRFactor();
      break;
    case MatmulLayout::TN:
      tv0b = broadcast(a, {false, true, false});
      tv1b = broadcast(b, {true, false, false});
      tv2 = fusedMultiplySum(tv0b, tv1b, {2});
      // M, N, K
      break;
    case MatmulLayout::NT:
      tv0b = broadcast(a, {false, false, true});
      tv1b = broadcast(b, {false, true, false});
      tv2 = fusedMultiplySum(tv0b, tv1b, {0});
      // K, M, N -> M, N, K
      tv2->reorder({{0, -1}});
      tv2->commitLeafToRFactor();
      break;
    case MatmulLayout::NN:
      tv0b = broadcast(a, {true, false, false});
      tv1b = broadcast(b, {false, false, true});
      tv2 = fusedMultiplySum(tv0b, tv1b, {1});
      // N, K, M -> M, N, K
      tv2->reorder({{-1, 0}});
      tv2->commitLeafToRFactor();
      break;
    default:
      TORCH_CHECK(false, "unsupported data layout.");
  }
  return tv2;
}

TensorView* matmulTuringOrLater(
    TensorView* a,
    TensorView* b,
    MatmulLayout layout) {
  TORCH_CHECK(
      a->nDims() == 2 && b->nDims() == 2, "only pure matmuls for these tests");
  TensorView *tv2 = nullptr, *tv0t = nullptr, *tv1t = nullptr, *tv0b = nullptr,
             *tv1b = nullptr;
  switch (layout) {
      // Canonicalize all inputs to [M, K] and [N, K]
    case MatmulLayout::TT:
      tv0t = a;
      tv1t = transpose(b, 0, 1);
      break;
    case MatmulLayout::TN:
      tv0t = a;
      tv1t = b;
      break;
    case MatmulLayout::NT:
      tv0t = transpose(a, 0, 1);
      tv1t = transpose(b, 0, 1);
      break;
    case MatmulLayout::NN:
      tv0t = transpose(a, 0, 1);
      tv1t = b;
      break;
    default:
      TORCH_CHECK(false, "unsupported data layout.");
  }
  tv0b = broadcast(tv0t, {false, true, false});
  tv1b = broadcast(tv1t, {true, false, false});
  tv2 = fusedMultiplySum(tv0b, tv1b, {2});
  return tv2;
}

TensorView* matmul(
    TensorView* a,
    TensorView* b,
    MatmulLayout layout,
    bool turing_or_later // TODO: This is a temporary solution. Remove this!
) {
  if (turing_or_later) {
    return matmulTuringOrLater(a, b, layout);
  } else {
    return matmulVolta(a, b, layout);
  }
}

TensorView* splitkLikeBatchedMatmul(
    TensorView* a,
    TensorView* b,
    MatmulLayout layout) {
  TORCH_CHECK(
      a->nDims() == 3 && b->nDims() == 3,
      "only splitk-like batched matmuls for these tests");
  TensorView *tv2 = nullptr, *tv0t = nullptr, *tv1t = nullptr, *tv0b = nullptr,
             *tv1b = nullptr;
  switch (layout) {
      // Canonicalize all inputs to [B, M, K] and [B, N, K]
    case MatmulLayout::TT:
      // [M, B, K] -> [B, M, K]
      tv0t = transpose(a, 0, 1);
      // [B, K, N] -> [B, N, K]
      tv1t = transpose(b, 1, 2);
      break;
    case MatmulLayout::TN:
      // [M, B, K] -> [B, M, K]
      tv0t = transpose(a, 0, 1);
      // [N, B, K] -> [B, N, K]
      tv1t = transpose(b, 0, 1);
      break;
    case MatmulLayout::NT:
      // [B, K, M] -> [B, M, K]
      tv0t = transpose(a, 1, 2);
      // [B, K, N] -> [B, N, K]
      tv1t = transpose(b, 1, 2);
      break;
    case MatmulLayout::NN:
      // [B, K, M] -> [B, M, K]
      tv0t = transpose(a, 1, 2);
      // [N, B, K] -> [B, N, K]
      tv1t = transpose(b, 0, 1);
      break;
    default:
      TORCH_CHECK(false, "unsupported data layout.");
  }
  tv0b = broadcast(tv0t, {false, false, true, false});
  tv1b = broadcast(tv1t, {false, true, false, false});
  tv2 = fusedMultiplySum(tv0b, tv1b, {3});
  return tv2;
}

at::Tensor atMatmul(at::Tensor a, at::Tensor b, MatmulLayout layout) {
  switch (layout) {
    case MatmulLayout::TT:
      return a.matmul(b);
    case MatmulLayout::TN:
      return a.matmul(b.t());
    case MatmulLayout::NT:
      return a.t().matmul(b);
    case MatmulLayout::NN:
      return a.t().matmul(b.t());
    default:
      TORCH_CHECK(false, "unsupported data layout.");
  }
  return at::Tensor();
}

at::Tensor splitkLikeAtMatmul(at::Tensor a, at::Tensor b, MatmulLayout layout) {
  switch (layout) {
    case MatmulLayout::TT:
      // [M, B, K] @ [B, K, N] -> [B, M, N]
      return a.transpose(0, 1).matmul(b);
    case MatmulLayout::TN:
      // [M, B, K] @ [N, B, K] -> [B, M, N]
      return a.transpose(0, 1).matmul(b.permute({1, 2, 0}));
    case MatmulLayout::NT:
      // [B, K, M] @ [B, K, N] -> [B, M, N]
      return a.transpose(1, 2).matmul(b);
    case MatmulLayout::NN:
      // [B, K, M] @ [N, B, K] -> [B, M, N]
      return a.transpose(1, 2).matmul(b.permute({1, 2, 0}));
    default:
      TORCH_CHECK(false, "unsupported data layout.");
  }
  return at::Tensor();
}

std::pair<at::Tensor, at::Tensor> matmulAtInput(
    int M,
    int N,
    int K,
    MatmulLayout layout,
    c10::ScalarType dtype) {
  auto options = at::TensorOptions().dtype(dtype).device(at::kCUDA, 0);

  switch (layout) {
    case MatmulLayout::TT:
      return std::make_pair(
          at::randn({M, K}, options), at::randn({K, N}, options));
    case MatmulLayout::TN:
      return std::make_pair(
          at::randn({M, K}, options), at::randn({N, K}, options));
    case MatmulLayout::NT:
      return std::make_pair(
          at::randn({K, M}, options), at::randn({K, N}, options));
    case MatmulLayout::NN:
      return std::make_pair(
          at::randn({K, M}, options), at::randn({N, K}, options));
    default:
      TORCH_CHECK(false, "unsupported data layout.");
  }
  return std::make_pair(at::Tensor(), at::Tensor());
}

at::Tensor matmulAtInput(
    const MatmulLayout layout,
    const TensorMatmulPos tensor,
    const c10::ScalarType dtype,
    const int M,
    const int N,
    const int K,
    const int B,
    const int device) {
  const auto options =
      at::TensorOptions().dtype(dtype).device(at::kCUDA, device);
  const auto is_batch = B != 0;

  // handle C and D tensors, layout does not impact shape
  switch (tensor) {
    case TensorMatmulPos::C:
    case TensorMatmulPos::D:
      return is_batch ? at::randn({B, M, N}, options)
                      : at::randn({M, N}, options);
    case TensorMatmulPos::Bias:
      return is_batch ? at::randn({B, M}, options) : at::randn({M}, options);
    default:
      break;
  }

  switch (layout) {
    case MatmulLayout::TT:
      switch (tensor) {
        case TensorMatmulPos::A:
          return is_batch ? at::randn({M, B, K}, options)
                          : at::randn({M, K}, options);
        case TensorMatmulPos::B:
          return is_batch ? at::randn({B, K, N}, options)
                          : at::randn({K, N}, options);
        default:
          break;
      }
      break;
    case MatmulLayout::TN:
      switch (tensor) {
        case TensorMatmulPos::A:
          return is_batch ? at::randn({M, B, K}, options)
                          : at::randn({M, K}, options);
        case TensorMatmulPos::B:
          return is_batch ? at::randn({N, B, K}, options)
                          : at::randn({N, K}, options);
        default:
          break;
      }
      break;
    case MatmulLayout::NT:
      switch (tensor) {
        case TensorMatmulPos::A:
          return is_batch ? at::randn({B, K, M}, options)
                          : at::randn({K, M}, options);
        case TensorMatmulPos::B:
          return is_batch ? at::randn({B, K, N}, options)
                          : at::randn({K, N}, options);
        default:
          break;
      }
      break;
    case MatmulLayout::NN:
      switch (tensor) {
        case TensorMatmulPos::A:
          return is_batch ? at::randn({B, K, M}, options)
                          : at::randn({K, M}, options);
        case TensorMatmulPos::B:
          return is_batch ? at::randn({N, B, K}, options)
                          : at::randn({N, K}, options);
        default:
          break;
      }
      break;
    default:
      TORCH_CHECK(false, "unsupported data layout, got ", (size_t)layout);
  }
  TORCH_CHECK(false, "unsupported tensor position, got ", (size_t)tensor);
}

bool isSchedulerInUse(
    nvfuser::FusionKernelRuntime* kernel_rt,
    const ScheduleHeuristic& scheduler) {
  if (nullptr == kernel_rt) {
    return false;
  }
  const auto scheduler_heurs = kernel_rt->schedulerHeuristics();
  if (nullptr == scheduler_heurs) {
    return false;
  }
  const auto& heurs = scheduler_heurs->heuristicsList();

  for (const auto& heur_entry : heurs) {
    if (heur_entry && (scheduler == heur_entry->heuristic())) {
      return true;
    }
  }

  return false;
}

void validateSegmentation(
    FusionKernelRuntime* runtime,
    const std::vector<ScheduleHeuristic>& expected_heuristics) {
  const auto& segment_groups = runtime->fusionSegments()->groups();

  TORCH_CHECK(
      segment_groups.size() == expected_heuristics.size(),
      "Unexpected segments. Expected: ",
      expected_heuristics.size(),
      ". Actual: ",
      segment_groups.size());

  // Assumes up to two segments exist for simplicity
  TORCH_INTERNAL_ASSERT(
      segment_groups.size() <= 2, "True segment order analysis is required");

  for (auto& group : segment_groups) {
    int segment_order = group->producer_edges.empty() ? 0 : 1;
    TORCH_CHECK(
        group->heuristic() == expected_heuristics.at(segment_order),
        "Expected to use the ",
        expected_heuristics.at(segment_order),
        " scheduler but ",
        group->heuristic(),
        " was used");
  }
}

TensorView* biasEpilogue(TensorView* tensor, TensorView* bias) {
  TORCH_CHECK(
      tensor->dtype() == bias->dtype(),
      "bias vector must have the same type as tensor with two domains, bias: ",
      bias->dtype(),
      ", tensor: ",
      tensor->dtype());
  TORCH_CHECK(
      tensor->nDims() >= 2,
      "Tensors to have bias applied needs to have 2 or more domains, got ",
      tensor->nDims());

  const auto concrete = TensorDomain::noReductions(
      TensorDomain::noBroadcasts(tensor->getLeafDomain()));

  TensorView *biasb = nullptr, *biased = nullptr;

  switch (concrete.size()) {
    case 2:
      // regular matmul (non-strided batch gemm)
      TORCH_CHECK(
          bias->nDims() == 1,
          "bias vector must have one domain, got",
          bias->nDims());
      biasb = broadcast(bias, {false, true});
      break;
    case 3:
      // strided batch gemm case
      TORCH_CHECK(
          (bias->nDims() == 2 || bias->nDims() == 1),
          "bias vector must have one (single bias for batch) "
          "or two (bias for each batch entries)), got",
          bias->nDims());
      biasb = broadcast(bias, {false, false, true});
      break;
    default:
      TORCH_CHECK(
          false,
          "Only tensors with two (matmul) or three (strided batch matmul) "
          "concrete domains have support for bias epilogue enabled, got ",
          concrete.size());
  }

  biased = add(tensor, biasb);
  return biased;
}

at::Tensor atBiasEpilogue(const at::Tensor& tensor, const at::Tensor& bias) {
  switch (tensor.dim()) {
    case 2:
      TORCH_CHECK(
          bias.dim() == 1,
          "For single matmul problem bias must be a vector, got ",
          bias.dim());
      TORCH_CHECK(
          tensor.dim() == 2,
          "For single matmul problem only 2d tensors have"
          " bias epilogue support, got ",
          tensor.dim());
      break;
    case 3:
      TORCH_CHECK(
          bias.dim() == 2,
          "For strided batch matmul problem bias must be a 2d tensor, got ",
          bias.dim());
      TORCH_CHECK(
          tensor.dim() == 3,
          "For strided batch matmul matmul problem only 3d tensors have"
          " bias epilogue support, got ",
          tensor.dim());
      break;
    default:
      TORCH_CHECK(
          false,
          "Only tensors with two (matmul) or three (strided batch matmul) "
          "concrete domains have support for bias epilogue enabled, got ",
          tensor.dim());
  }

  // The inner most dimension of bias tensor contains the rows number
  const int64_t rows = bias.size(-1);

  // We skip number of columns and access directly dim for rows, hence '-2'
  TORCH_CHECK(
      tensor.size(tensor.dim() - 2) == rows,
      "Tensor must have the same number of rows as bias vector");

  return tensor.add(bias.unsqueeze(-1));
}

} // namespace nvfuser
