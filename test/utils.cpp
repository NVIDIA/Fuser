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

#include <regex>
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
  GpuLower gpulw(fusion);
  const std::string actual_kernel =
      "\n" + codegen::generateCudaKernel(gpulw.run());
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
    NVF_CHECK(false);
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

std::string Instruction::predicate() const {
  if (str[0] == '@') {
    std::stringstream ss(str);
    char ignore_at = '\0';
    std::string result;
    ss >> ignore_at >> result;
    return result;
  }
  return {};
}

std::string Instruction::action() const {
  std::string result;
  std::stringstream ss(str);
  if (str[0] == '@') {
    ss >> result;
  }
  std::getline(ss, result);
  result = trim(result);
  return result;
}

std::string Instruction::op() const {
  std::stringstream ss(action());
  std::string result;
  ss >> result;
  return result;
}

std::string Instruction::opCode() const {
  std::string result;
  for (auto i : op()) {
    if (i == '.') {
      return result;
    }
    result.push_back(i);
  }
  return result;
}

std::vector<std::string> Instruction::args() const {
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

std::vector<std::string> Instruction::modifiers() const {
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
  std::regex find_header_regex(".text.(.+):");
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
        std::smatch line_match;
        std::regex_match(line, line_match, find_header_regex);
        if (line_match.size() == 2) {
          header = line_match.str(1) + ":";
        }
      }
    }
  }
  return result;
}

} // namespace sass

// matmulAtInput provides batched inputs in a splitk-like ordering. It provides
// contiguous tensors with these shapes
//   TT: [M, B, K] [B, K, N]
//   TN: [M, B, K] [N, B, K]
//   NT: [B, K, M] [B, K, N]
//   NN: [B, K, M] [N, B, K]
// fusedMultiplySum assumes [B, M, K] [B, N, K] so here we transpose into that
// order
TensorView* matmulTuringOrLater(
    TensorView* a,
    TensorView* b,
    MatmulLayout layout) {
  NVF_CHECK(a->nDims() == b->nDims());
  NVF_CHECK(a->nDims() == 2 || a->nDims() == 3);
  TensorView *tv2 = nullptr, *tv0t = nullptr, *tv1t = nullptr, *tv0b = nullptr,
             *tv1b = nullptr;
  if (a->nDims() == 3) { // bmm
    switch (layout) {
        // Canonicalize all inputs to [B, M, K] and [B, N, K]
      case MatmulLayout::TT:
        tv0t = transpose(a, 0, 1);
        tv1t = transpose(b, 1, 2);
        break;
      case MatmulLayout::TN:
        tv0t = transpose(a, 0, 1);
        tv1t = transpose(b, 0, 1);
        break;
      case MatmulLayout::NT:
        tv0t = transpose(a, 1, 2);
        tv1t = transpose(b, 1, 2);
        break;
      case MatmulLayout::NN:
        tv0t = transpose(a, 1, 2);
        tv1t = transpose(b, 0, 1);
        break;
      default:
        NVF_CHECK(false, "unsupported data layout.");
    }
  } else {
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
        NVF_CHECK(false, "unsupported data layout.");
    }
  }
  std::vector<bool> bcast_dims(a->nDims() + 1, false);
  bcast_dims.at(bcast_dims.size() - 2) = true;
  tv0b = broadcast(tv0t, bcast_dims);
  bcast_dims.at(bcast_dims.size() - 2) = false;
  bcast_dims.at(bcast_dims.size() - 3) = true;
  tv1b = broadcast(tv1t, bcast_dims);
  tv2 = fusedMultiplySum(tv0b, tv1b, {-1});
  return tv2;
}

TensorView* matmul(
    TensorView* a,
    TensorView* b,
    MatmulLayout layout,
    bool turing_or_later // TODO: This is a temporary solution. Remove this!
) {
  NVF_ERROR(turing_or_later, "Only Turing or later is supported for now.");
  return matmulTuringOrLater(a, b, layout);
}

TensorView* splitkLikeBatchedMatmul(
    TensorView* a,
    TensorView* b,
    MatmulLayout layout) {
  NVF_CHECK(
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
      NVF_CHECK(false, "unsupported data layout.");
  }
  tv0b = broadcast(tv0t, {false, false, true, false});
  tv1b = broadcast(tv1t, {false, true, false, false});
  tv2 = fusedMultiplySum(tv0b, tv1b, {3});
  return tv2;
}

// matmulAtInput provides batched inputs in a splitk-like ordering. It provides
// contiguous tensors with these shapes
//   TT: [M, B, K] [B, K, N]
//   TN: [M, B, K] [N, B, K]
//   NT: [B, K, M] [B, K, N]
//   NN: [B, K, M] [N, B, K]
// ATen matmul assumes [B, M, K] [B, K, N] so here we transpose into that order
at::Tensor atMatmul(at::Tensor a, at::Tensor b, MatmulLayout layout) {
  NVF_CHECK(
      a.dim() == b.dim(), "Either both or none of A and B should be batch");
  NVF_CHECK(
      a.dim() == 2 || a.dim() == 3,
      "Must have either zero or one batch dimensions");
  if (a.dim() == 3) { // bmm
    switch (layout) {
      case MatmulLayout::TT:
        return a.transpose(0, 1).matmul(b);
      case MatmulLayout::TN:
        return a.transpose(0, 1).matmul(b.transpose(0, 1).transpose(1, 2));
      case MatmulLayout::NT:
        return a.transpose(1, 2).matmul(b);
      case MatmulLayout::NN:
        return a.transpose(1, 2).matmul(b.transpose(0, 1).transpose(1, 2));
      default:
        NVF_CHECK(false, "unsupported data layout.");
    }
  } else {
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
        NVF_CHECK(false, "unsupported data layout.");
    }
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
      NVF_CHECK(false, "unsupported data layout.");
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
      NVF_CHECK(false, "unsupported data layout.");
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
      NVF_CHECK(false, "unsupported data layout, got ", (size_t)layout);
  }
  NVF_CHECK(false, "unsupported tensor position, got ", (size_t)tensor);
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

  NVF_CHECK(
      segment_groups.size() == expected_heuristics.size(),
      "Unexpected segments. Expected: ",
      expected_heuristics.size(),
      ". Actual: ",
      segment_groups.size());

  // Assumes up to two segments exist for simplicity
  NVF_ERROR(
      segment_groups.size() <= 2, "True segment order analysis is required");

  for (auto& group : segment_groups) {
    int segment_order = group->producer_edges.empty() ? 0 : 1;
    NVF_CHECK(
        group->heuristic() == expected_heuristics.at(segment_order),
        "Expected to use the ",
        expected_heuristics.at(segment_order),
        " scheduler but ",
        group->heuristic(),
        " was used");
  }
}

TensorView* biasEpilogue(TensorView* tensor, TensorView* bias) {
  NVF_CHECK(
      tensor->dtype() == bias->dtype(),
      "bias vector must have the same type as tensor with two domains, bias: ",
      bias->dtype(),
      ", tensor: ",
      tensor->dtype());
  NVF_CHECK(
      tensor->nDims() >= 2,
      "Tensors to have bias applied needs to have 2 or more domains, got ",
      tensor->nDims());

  const auto concrete = TensorDomain::noReductions(
      TensorDomain::noBroadcasts(tensor->getLeafDomain()));

  TensorView *biasb = nullptr, *biased = nullptr;

  switch (concrete.size()) {
    case 2:
      // regular matmul (non-strided batch gemm)
      NVF_CHECK(
          bias->nDims() == 1,
          "bias vector must have one domain, got",
          bias->nDims());
      biasb = broadcast(bias, {false, true});
      break;
    case 3:
      // strided batch gemm case
      if (bias->nDims() == 1) {
        // case with a single bias used through whole batch
        biasb = broadcast(bias, {true, false, true});
      } else if (bias->nDims() == 2) {
        // case with dedicated bias for each problem in the batch
        biasb = broadcast(bias, {false, false, true});
      } else {
        NVF_CHECK(
            false,
            "bias vector must have one (single bias for batch) "
            "or two (bias for each batch entries)), got",
            bias->nDims());
      }
      break;
    default:
      NVF_CHECK(
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
      NVF_CHECK(
          bias.dim() == 1,
          "For single matmul problem bias must be a vector, got ",
          bias.dim());
      break;
    case 3:
      NVF_CHECK(
          (bias.dim() == 1 || bias.dim() == 2),
          "For strided batch matmul problem bias must be 1d or 2d tensor, got ",
          bias.dim());
      break;
    default:
      NVF_CHECK(
          false,
          "Only tensors with two (matmul) or three (strided batch matmul) "
          "concrete domains have support for bias epilogue enabled, got ",
          tensor.dim());
  }

  // The inner most dimension of bias tensor contains the rows number
  const int64_t rows = bias.size(-1);

  // We skip number of columns and access directly dim for rows, hence '-2'
  NVF_CHECK(
      tensor.size(tensor.dim() - 2) == rows,
      "Tensor must have the same number of rows as bias vector");

  return tensor.add(bias.unsqueeze(-1));
}

size_t getCRandomSeed() {
  static thread_local bool found_seed = false;
  static thread_local size_t seed = 0L;

  if (!found_seed) {
    const char* env_var = "TEST_RANDOM_SEED";
    auto seed_str = getNvFuserEnv(env_var);
    if (seed_str) {
      try {
        seed = std::stol(seed_str);
      } catch (const std::exception& e) {
        std::cerr << "Could not parse environment variable NVFUSER_" << env_var
                  << std::endl;
        throw e;
      }
    } else {
      // We default to setting the C random seed to the system time in seconds
      // since the epoch in order to promote structural randomness in tests. For
      // example, if a test uses `std::rand()` to choose random sizes or
      // dimensions, then the number of combinations grows combinatorially.
      // Using a changing, but controlled, seed like this is helpful in these
      // cases since it increases the coverage of the test when numerous
      // different runs of the test suite are considered. When an error is
      // encountered, we can repeat with that seed  since it will be printed
      // upon error.
      seed = std::chrono::duration_cast<std::chrono::seconds>(
                 std::chrono::system_clock::now().time_since_epoch())
                 .count();
    }
  }

  return seed;
}

size_t getATenRandomSeed() {
  static thread_local bool found_seed = false;
  static thread_local size_t seed = 0L;

  if (!found_seed) {
    const char* env_var = "TEST_ATEN_RANDOM_SEED";
    auto seed_str = getNvFuserEnv(env_var);
    if (seed_str) {
      try {
        seed = std::stol(seed_str);
      } catch (const std::exception& e) {
        std::cerr << "Could not parse environment variable NVFUSER_" << env_var
                  << std::endl;
        throw e;
      }
    } else {
      // We default to setting the ATen seed to zero, instead of system time.
      // The C PRNG, std::rand() is typically used for structural parameters
      // such as choosing random sizes or indices. These combinatorial tests
      // benefit from increased coverage since exhaustive testing would be ideal
      // but impractical in those cases. ATen randomness, on the other hand, is
      // typically used for filling in data in test tensors, and results are
      // then used in inexact matching tests. Those tests pass with high
      // probability but might fail often were the test exhaustive. By fixing
      // the default ATen seed to zero, we can avoid some false positive test
      // failures while still ensuring there is at least one seed for which the
      // tests pass.
      seed = 0L;
    }
  }

  return seed;
}

int getNumSMs() {
  // Since cudaGetDeviceProperties can be slow, we memoize the value in num_SMs
  static std::vector<int> num_SMs;

  int dev_idx = 0;
  NVFUSER_CUDA_RT_SAFE_CALL(cudaGetDevice(&dev_idx));

  if (num_SMs.size() <= (size_t)dev_idx) {
    num_SMs.resize(dev_idx + 1, -1);
  }

  if (num_SMs[dev_idx] == -1) {
    cudaDeviceProp prop{};
    NVFUSER_CUDA_RT_SAFE_CALL(cudaGetDeviceProperties(&prop, dev_idx));

    num_SMs[dev_idx] = prop.multiProcessorCount;
  }
  return num_SMs[dev_idx];
}

} // namespace nvfuser
