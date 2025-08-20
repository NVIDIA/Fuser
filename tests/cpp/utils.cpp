// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#include <regex>
#include <sstream>
#include <string>
#include <string_view>

#include <c10/core/Allocator.h>
#include <c10/core/CachingDeviceAllocator.h>
#include <c10/cuda/CUDACachingAllocator.h>

#include <exceptions.h>
#include <ops/all_ops.h>
#include <scheduler/mma_utils.h>
#include <tests/cpp/utils.h>

namespace nvfuser {

NVFuserTest::NVFuserTest() {
  // Enable logging so debug messages in PyTorch can be printed out
  // via `TORCH_CPP_LOG_LEVEL`.
  c10::initLogging();

  setFillAllocationWithNan(true);

  maybeClearAllocator();

  // If NVFUSER_TEST_RANDOM_SEED is provided, use that for the C random seed.
  // Otherwise, use system time. If a test fails, this seed will be printed.
  at::manual_seed(getATenRandomSeed());

  // If NVFUSER_TEST_ATEN_RANDOM_SEED is provided, use that for the ATen
  // random seed. Otherwise, use zero. If a test fails, this seed will be
  // printed.
  std::srand(getCRandomSeed());

  EnableOptionsGuard::getCurOptions().set(EnableOption::IdModelExtraValidation);

  constexpr const char* kTf32Override = "NVIDIA_TF32_OVERRIDE";
  if (setenv(kTf32Override, "0", /*overwrite=*/1) != 0) {
    TORCH_WARN("Failed to set ", kTf32Override, " to 0");
  }
}

void NVFuserTest::SetUp() {
  // requires PASCAL or newer
  if (!deviceMajorMinorCheck(6)) {
    GTEST_SKIP() << "skipping tests on pre-PASCAL GPUs";
  }

  EnableOptionsGuard::getCurOptions().set(EnableOption::GreedyScheduler);
}

NVFuserTest::~NVFuserTest() {
  if (::testing::Test::HasFailure()) {
    auto test_info = ::testing::UnitTest::GetInstance()->current_test_info();
    std::cerr << "To reproduce: NVFUSER_TEST_RANDOM_SEED=" << getCRandomSeed()
              << " NVFUSER_TEST_ATEN_RANDOM_SEED=" << getATenRandomSeed()
              << " test_nvfuser --gtest_filter='"
              << test_info->test_suite_name() << "." << test_info->name() << "'"
              << std::endl;
  }

  // Make sure capturing of stdout is stopped
  ensureStopCaptureStdout();

  // Make sure profiler is unset in case it was set during test
  ProfilerOptionsGuard::getCurOptions().unset(ProfilerOption::Enable);
  ProfilerOptionsGuard::getCurOptions().unset(ProfilerOption::EnableNocupti);
}

void NVFuserTest::captureStdout() {
  if (!capturing_) {
    testing::internal::CaptureStdout();
    capturing_ = true;
  }
}

void NVFuserTest::ensureStopCaptureStdout() {
  if (capturing_) {
    testing::internal::GetCapturedStdout();
    capturing_ = false;
  }
}

std::string NVFuserTest::getCapturedStdout() {
  NVF_ERROR(capturing_, "Not captured");
  auto str = testing::internal::GetCapturedStdout();
  capturing_ = false;
  return str;
}

const KernelExecutor* onlyKernelExecutorInMostRecentRuntime(
    const FusionExecutorCache& executor_cache) {
  const auto& executors =
      executor_cache.getMostRecentKernelRuntime()->executors();
  NVF_CHECK(executors.size() == 1);
  NVF_CHECK(executors.front()->isA<KernelExecutor>());
  return executors.front()->as<KernelExecutor>();
}

CGResultsPackage scheduleAndRun(
    Fusion* fusion,
    SchedulerType scheduler_type,
    const KernelArgumentHolder& runtime_inputs,
    bool validate_scheduler) {
  auto heuristic_params = SchedulerEntry::scheduleWith(
      fusion, scheduler_type, runtime_inputs, validate_scheduler);
  auto ke = std::make_unique<KernelExecutor>();
  ke->compile(fusion, runtime_inputs, heuristic_params->lparams);
  auto cg_outputs = ke->run(runtime_inputs, {}, heuristic_params->lparams);
  CGResultsPackage results = {
      .outputs = std::move(cg_outputs),
      .heuristic_params = std::move(heuristic_params),
      .kernel_executor = std::move(ke)};
  return results;
}

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
      s.begin(), s.end(), [](int64_t c) { return std::isspace(c); });
  auto wsback = std::find_if_not(s.rbegin(), s.rend(), [](int64_t c) {
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
  std::regex zero_pattern_regex("/\\*0+\\*/");
  for (std::string line; std::getline(ss, line);) {
    line = trim(line);
    if (line.empty() || starts_with(line, "//")) {
      continue;
    }
    if (!started) {
      if (std::regex_search(line, zero_pattern_regex)) {
        started = true;
      }
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
    }
  }
  return result;
}

} // namespace sass

// matmulAtInput2D provides batched inputs in a splitk-like ordering. It
// provides contiguous tensors with these shapes
//   TT: [M, B, K] [B, K, N]
//   TN: [M, B, K] [N, B, K]
//   NT: [B, K, M] [B, K, N]
//   NN: [B, K, M] [N, B, K]
// ATen matmul assumes [B, M, K] [B, K, N] so here we transpose into that order
at::Tensor atMatmul(at::Tensor a, at::Tensor b, MmaLayout layout) {
  a = a.squeeze();
  b = b.squeeze();
  NVF_CHECK(
      a.dim() == b.dim(), "Either both or none of A and B should be batch");
  NVF_CHECK(
      a.dim() == 2 || a.dim() == 3,
      "Must have either zero or one batch dimensions");
  if (a.dim() == 3) { // bmm
    switch (layout) {
      case MmaLayout::TT:
        return a.transpose(0, 1).matmul(b);
      case MmaLayout::TN:
        return a.transpose(0, 1).matmul(b.transpose(0, 1).transpose(1, 2));
      case MmaLayout::NT:
        return a.transpose(1, 2).matmul(b);
      case MmaLayout::NN:
        return a.transpose(1, 2).matmul(b.transpose(0, 1).transpose(1, 2));
      default:
        NVF_CHECK(false, "unsupported data layout.");
    }
  } else {
    switch (layout) {
      case MmaLayout::TT:
        return a.matmul(b);
      case MmaLayout::TN:
        return a.matmul(b.t());
      case MmaLayout::NT:
        return a.t().matmul(b);
      case MmaLayout::NN:
        return a.t().matmul(b.t());
      default:
        NVF_CHECK(false, "unsupported data layout.");
    }
  }
  return at::Tensor();
}

at::Tensor splitkLikeAtMatmul(at::Tensor a, at::Tensor b, MmaLayout layout) {
  switch (layout) {
    case MmaLayout::TT:
      // [M, B, K] @ [B, K, N] -> [B, M, N]
      return a.transpose(0, 1).matmul(b);
    case MmaLayout::TN:
      // [M, B, K] @ [N, B, K] -> [B, M, N]
      return a.transpose(0, 1).matmul(b.permute({1, 2, 0}));
    case MmaLayout::NT:
      // [B, K, M] @ [B, K, N] -> [B, M, N]
      return a.transpose(1, 2).matmul(b);
    case MmaLayout::NN:
      // [B, K, M] @ [N, B, K] -> [B, M, N]
      return a.transpose(1, 2).matmul(b.permute({1, 2, 0}));
    default:
      NVF_CHECK(false, "unsupported data layout.");
  }
  return at::Tensor();
}

std::pair<at::Tensor, at::Tensor> matmulAtInput2D(
    int64_t M,
    int64_t N,
    int64_t K,
    MmaLayout layout,
    c10::ScalarType dtype) {
  auto options = at::TensorOptions().dtype(dtype).device(at::kCUDA, 0);

  switch (layout) {
    case MmaLayout::TT:
      return std::make_pair(
          at::randn({M, K}, options), at::randn({K, N}, options));
    case MmaLayout::TN:
      return std::make_pair(
          at::randn({M, K}, options), at::randn({N, K}, options));
    case MmaLayout::NT:
      return std::make_pair(
          at::randn({K, M}, options), at::randn({K, N}, options));
    case MmaLayout::NN:
      return std::make_pair(
          at::randn({K, M}, options), at::randn({N, K}, options));
    default:
      NVF_CHECK(false, "unsupported data layout.");
  }
  return std::make_pair(at::Tensor(), at::Tensor());
}

std::pair<std::vector<int64_t>, std::vector<int64_t>> matmulAtInputShape3DTuring(
    int64_t M,
    int64_t N,
    int64_t K,
    MmaLayout layout) {
  switch (layout) {
    case MmaLayout::TT:
      return {{M, 1, K}, {1, K, N}};
    case MmaLayout::TN:
      return {{M, 1, K}, {1, N, K}};
    case MmaLayout::NT:
      return {{K, 1, M}, {1, K, N}};
    case MmaLayout::NN:
      return {{K, 1, M}, {1, N, K}};
    default:
      NVF_CHECK(false, "unsupported data layout.");
  }
}

std::pair<at::Tensor, at::Tensor> matmulAtInput3DTuring(
    int64_t M,
    int64_t N,
    int64_t K,
    MmaLayout layout,
    c10::ScalarType dtype) {
  auto options = at::TensorOptions().dtype(dtype).device(at::kCUDA, 0);
  auto shapes = matmulAtInputShape3DTuring(M, N, K, layout);
  return std::make_pair(
      at::randn(shapes.first, options), at::randn(shapes.second, options));
}

at::Tensor matmulAtInput2D(
    const MmaLayout layout,
    const TensorMatmulPos tensor,
    const c10::ScalarType dtype,
    const int64_t M,
    const int64_t N,
    const int64_t K,
    const int64_t B,
    const int64_t device) {
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
    case MmaLayout::TT:
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
    case MmaLayout::TN:
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
    case MmaLayout::NT:
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
    case MmaLayout::NN:
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

// matmulAtInput2D/matmulAtInput3DTuring provides batched inputs in a
// splitk-like ordering. It provides contiguous tensors with these shapes
//   TT: [M, (N,) B, K] [(M,) B, K, N]
//   TN: [M, (N,) B, K] [(M,) N, B, K]
//   NT: [B, K, (N,) M] [(M,) B, K, N]
//   NN: [B, K, (N,) M] [(M,) N, B, K]
// where the dimension in parentheses is a broadcast dimension and may be
// omitted.
TensorView* canonicalizeInputToBMNK(
    TensorView* tv,
    MmaLayout layout,
    MmaOperand operand) {
  auto lgnob = TensorDomain::noBroadcasts(tv->getLogicalDomain());
  NVF_ERROR(
      lgnob.size() == 2 || lgnob.size() == 3,
      "Expected 2 or 3 domains, got ",
      lgnob.size());
  bool has_batch = lgnob.size() == 3;
  bool already_broadcasted = tv->hasBroadcast();

  // Step 1: insert permute as needed.
  if (operand == MmaOperand::A) {
    if (layout == MmaLayout::TT || layout == MmaLayout::TN) {
      // [M, (N,) B, K] -> [B, M, (N,) K]
      if (has_batch) {
        // Using reorder + commitLeafToLogical instead of permute here because
        // the former's API is more convenient here
        tv = permute(tv, {{-2, 0}});
      }
    } else { // NT, NN
      // [B, K, (N,) M] -> [B, M, (N,) K]
      tv = transpose(tv, has_batch, -1);
    }
  } else { // B
    if (layout == MmaLayout::TT || layout == MmaLayout::NT) {
      // [(M,) B, K, N] -> [B, (M,) N, K]
      std::unordered_map<int64_t, int64_t> old2new = {{-1, -2}};
      if (has_batch && already_broadcasted) {
        old2new[0] = 1;
      }
      tv = permute(tv, old2new);
    } else { // TN, NN
      // [(M,) N, B, K] -> [B, (M,) N, K]
      if (has_batch) {
        tv = permute(tv, {{-2, 0}});
      }
    }
  }

  // Step 2: insert broadcast as needed.
  if (already_broadcasted) {
    return tv;
  }
  if (operand == MmaOperand::A) {
    // [B, M, K] -> [B, M, (N,) K]
    std::vector<bool> bcast_dims(tv->nDims() + 1, false);
    bcast_dims.at(bcast_dims.size() - 2) = true;
    tv = broadcast(tv, bcast_dims);
  } else { // B
    // [B, N, K] -> [B, (M,) N, K]
    std::vector<bool> bcast_dims(tv->nDims() + 1, false);
    bcast_dims.at(bcast_dims.size() - 3) = true;
    tv = broadcast(tv, bcast_dims);
  }
  return tv;
}

bool isSchedulerInUse(
    const nvfuser::FusionKernelRuntime* kernel_rt,
    const SchedulerType& scheduler_type) {
  if (nullptr == kernel_rt) {
    return false;
  }
  const auto heuristics = kernel_rt->schedulerHeuristics();
  if (nullptr == heuristics) {
    return false;
  }
  const auto& heuristics_list = heuristics->heuristicsList();

  for (const auto& heuristic_params : heuristics_list) {
    if (heuristic_params &&
        (scheduler_type == heuristic_params->scheduler_type)) {
      return true;
    }
  }

  return false;
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
      TensorDomain::noBroadcasts(tensor->getLoopDomain()));

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

int64_t getNumSMs() {
  // Since cudaGetDeviceProperties can be slow, we memoize the value in num_SMs
  static std::vector<int64_t> num_SMs;

  int dev_idx = 0;
  NVFUSER_CUDA_RT_SAFE_CALL(cudaGetDevice(&dev_idx));

  if ((int64_t)num_SMs.size() <= dev_idx) {
    num_SMs.resize(dev_idx + 1, -1);
  }

  if (num_SMs[dev_idx] == -1) {
    cudaDeviceProp prop{};
    NVFUSER_CUDA_RT_SAFE_CALL(cudaGetDeviceProperties(&prop, dev_idx));

    num_SMs[dev_idx] = prop.multiProcessorCount;
  }
  return num_SMs[dev_idx];
}

bool checkMapped(const ValGraph& vg, IterDomain* x, IterDomain* y) {
  if (!vg.hasGroup(x) || !vg.hasGroup(y)) {
    return false;
  }
  const ValGroup& gx = vg.toGroup(x);
  const ValGroup& gy = vg.toGroup(y);
  return gx.get() == gy.get();
};

MmaLayout getMatmulProblemLayout(Fusion* fusion) {
  const mma_utils::MatmulOperandInnerDimsOpt inner_dims_opt =
      mma_utils::getOperandInnerDims(fusion);

  NVF_ERROR(
      inner_dims_opt.isValid(),
      "Could not get operand inner dims: ",
      inner_dims_opt.getErrorMsg());

  const mma_utils::MatmulOperandInnerDims inner_dims = inner_dims_opt.getData();

  NVF_ERROR(inner_dims.size() == 2, "Found other than two operands");

  const bool A_K_inner = inner_dims.front() == MatmulDimRole::K;
  const bool B_K_inner = inner_dims.back() == MatmulDimRole::K;

  if (A_K_inner && B_K_inner) {
    return MmaLayout::TN;
  } else if (A_K_inner && !B_K_inner) {
    return MmaLayout::TT;
  } else if (!A_K_inner && B_K_inner) {
    return MmaLayout::NN;
  } else {
    return MmaLayout::NT;
  }
}

// get supported floating data types
std::vector<DataType> getFloatingDataTypes(bool include_complex) {
  std::vector<DataType> dtypes = {
      DataType::Double, DataType::Float, DataType::Half};
  if (include_complex) {
    dtypes.push_back(DataType::ComplexFloat);
    dtypes.push_back(DataType::ComplexDouble);
  }
#if defined(CUDA_VERSION) && CUDA_VERSION >= 11000
  if (at::cuda::getDeviceProperties(0)->major >= 8) {
    dtypes.push_back(DataType::BFloat16);
  }
#endif
  return dtypes;
}

std::string sanitizeTestName(const std::string& name) {
  // Replace all non-alphanumeric characters with underscores
  return std::regex_replace(name, std::regex("[^a-zA-Z0-9]"), "_");
}

bool isVectorized(TensorView* tv) {
  for (auto id : tv->getLoopDomain()) {
    if (id->getParallelType() == ParallelType::Vectorize) {
      return true;
    }
  }
  return false;
}

std::string macroToString(const MmaMacro macro) {
  std::stringstream ss;
  ss << "m" << getM(macro);
  ss << "_n" << getN(macro);
  ss << "_k" << getK(macro);
  return ss.str();
}

std::pair<TensorView*, TensorView*> createSdpaRngTvs() {
  DataType dtype = DataType::Int;
  std::vector<int64_t> philox_shape = {};

#if NVF_TORCH_VERSION_NO_LESS(2, 7, 0)
  dtype = DataType::UInt64;
  philox_shape = {2};
#endif
  TensorView* philox_seed =
      TensorViewBuilder().shape(philox_shape).dtype(dtype).build();
  TensorView* philox_offset = TensorViewBuilder().dtype(dtype).build();
#if !(NVF_TORCH_VERSION_NO_LESS(2, 7, 0))
  philox_seed->setCpuScalar(true);
  philox_offset->setCpuScalar(true);
#endif
  return std::make_pair(philox_seed, philox_offset);
}

std::pair<at::Tensor, at::Tensor> createSdpaRngTensors() {
  at::Tensor philox_seed, philox_offset;
  int64_t max_int64 = std::numeric_limits<int64_t>::max();
#if NVF_TORCH_VERSION_NO_LESS(2, 7, 0)
  philox_seed = at::randint(
      max_int64, // Using int64_t range to avoid
                 // overflow in randint
      {2}, // shape
      at::dtype(c10::kUInt64).device(at::kCUDA));
  philox_offset = at::empty({}, at::dtype(c10::kUInt64).device(at::kCUDA));
#else
  philox_seed = at::randint(max_int64, {}, at::kLong);
  philox_offset = at::randint(max_int64, {}, at::kLong);
#endif
  return std::make_pair(philox_seed, philox_offset);
}

void resetPeakMemoryStats(const c10::DeviceIndex device) {
  c10::cuda::CUDACachingAllocator::CUDAAllocator* allocator =
      c10::cuda::CUDACachingAllocator::get();
  NVF_CHECK(allocator != nullptr);

  allocator->resetPeakStats(device);
}

namespace {
// Stats like allocated_bytes comes as a size-3 array (cf.
// https://github.com/pytorch/pytorch/blob/feb503c1df78afd46962ed04e446d6e88ac0522d/c10/core/Allocator.h#L365-L370).
// The 0-th element is an aggregation of both the small pool and the large.
//
// To avoid hardcoded values, I initially wrote
// c10::CachingAllocator::StatType::AGGREGATE here but ran into compilation
// errors with slightly older PyTorch because that enum was introduced in a
// recent commit:
// https://github.com/pytorch/pytorch/commit/c65ee728f069ea9544bdcac815eb0825f45d1633.
// NVF_TORCH_VERSION_* don't seem to be good enough to distinguish before and
// after this commit.
constexpr int64_t kAggregateStatsIndex = 0;
} // namespace

int64_t maxMemoryAllocated(const c10::DeviceIndex device) {
  c10::cuda::CUDACachingAllocator::CUDAAllocator* allocator =
      c10::cuda::CUDACachingAllocator::get();
  NVF_CHECK(allocator != nullptr);

  c10::CachingDeviceAllocator::DeviceStats device_stats =
      allocator->getDeviceStats(device);

  return device_stats.allocated_bytes.at(kAggregateStatsIndex).peak;
}

int64_t memoryAllocated(const c10::DeviceIndex device) {
  c10::cuda::CUDACachingAllocator::CUDAAllocator* allocator =
      c10::cuda::CUDACachingAllocator::get();
  NVF_CHECK(allocator != nullptr);

  c10::CachingDeviceAllocator::DeviceStats device_stats =
      allocator->getDeviceStats(device);

  return device_stats.allocated_bytes.at(kAggregateStatsIndex).current;
}

} // namespace nvfuser
