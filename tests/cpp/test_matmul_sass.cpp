// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#include <csrc/exceptions.h>
#include <gtest/gtest.h>

#include <ops/arith.h>
#include <scheduler/matmul.h>
#include <scheduler/matmul_heuristic.h>
#include <scheduler/mma_utils.h>
#include <tests/cpp/utils.h>

#include <sstream>
#include <string>
#include <string_view>
#include <unordered_map>
#include <unordered_set>

namespace nvfuser {

using MatmulSASSTest = NVFuserTest;

class MatmulSASSTestWithLayout
    : public NVFuserTest,
      public ::testing::WithParamInterface<MmaLayout> {
 protected:
  MmaLayout layout;
  void SetUp() override {
    layout = GetParam();
    NVFuserTest::SetUp();
  }
};

// For SASS instruction definitions, see:
// https://docs.nvidia.com/cuda/cuda-binary-utilities/index.html#instruction-set-reference
//
// Some useful instructions for Ampere matmul:
// - LDGDEPBAR: Global Load Dependency Barrier
// - LDGSTS: Asynchronous Global to Shared Memcopy
// - LDSM: Load Matrix from Shared Memory with Element Size Expansion
// - HMMA: Matrix Multiply and Accumulate
// - BAR: Barrier Synchronization
// - DEPBAR: Dependency Barrier

namespace {

sass::Container getSASSFor(
    MmaLayout layout,
    GemmTile cta_tile,
    GemmTile warp_tile,
    MmaMacro macro,
    int M,
    int N,
    int K,
    const int smem_circular_buffer_stage = 4,
    const bool use_smem_epilogue = false,
    const bool promote_prologue_smem_reuse = false) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto shapes = matmulAtInputShape3DTuring(-1, -1, -1, layout);

  auto tv0 = makeContigConcreteTensor(shapes.first, DataType::Half);
  auto tv1 = makeContigConcreteTensor(shapes.second, DataType::Half);

  fusion.addInput(tv0);
  fusion.addInput(tv1);

  tv0 = canonicalizeInputToBMNK(tv0, layout, MmaOperand::A);
  tv1 = canonicalizeInputToBMNK(tv1, layout, MmaOperand::B);
  auto tv2 = fusedMultiplySum(tv0, tv1, {-1});

  fusion.addOutput(tv2);

  MatMulTileOptions gemm_tile;
  gemm_tile.cta_tile = cta_tile;
  gemm_tile.warp_tile = warp_tile;

  MatmulParams mparams;
  mparams.supported_vec_size = {8, 8, 4};
  mparams.mma_macro = macro;
  mparams.tile_sizes = gemm_tile;
  mparams.async_gmem_load_operands = true;
  mparams.circular_buffer_options.circular_buffer_smem_write = true;
  mparams.circular_buffer_options.circular_buffer_smem_read = true;
  mparams.circular_buffer_options.smem_circular_buffer_stage =
      smem_circular_buffer_stage;
  mparams.use_smem_epilogue = use_smem_epilogue;
  mparams.promote_prologue_smem_reuse = promote_prologue_smem_reuse;

  auto inputs = matmulAtInput3DTuring(M, N, K, layout);

  SchedulerEntry::makeSchedulerInstance(SchedulerType::Matmul)
      ->schedule(&fusion, &mparams);
  KernelExecutor ke;
  ke.compile(
      &fusion, {inputs.first, inputs.second}, LaunchParams(), matmul_cparams);
  auto cg_outputs = ke.run({inputs.first, inputs.second});
  auto tref = atMatmul(
      inputs.first.to(at::kFloat), inputs.second.to(at::kFloat), layout);

  NVF_CHECK(at::allclose(cg_outputs[0].as<at::Tensor>(), tref, 0.0001, 0.0001));

  return sass::parse(ke.compiledKernel()->disassembledKernelSASS());
}

// A fusion with epilogue made of binary op (scalar multiplication)
sass::Container getBinaryOpMulEpilogueSASSFor(
    MmaLayout layout,
    GemmTile cta_tile,
    GemmTile warp_tile,
    MmaMacro macro,
    int M,
    int N,
    int K) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto s0 = IrBuilder::create<Val>(DataType::Double);

  auto shapes = matmulAtInputShape3DTuring(-1, -1, -1, layout);

  auto tv0 = makeContigConcreteTensor(shapes.first, DataType::Half);
  auto tv1 = makeContigConcreteTensor(shapes.second, DataType::Half);

  fusion.addInput(tv0);
  fusion.addInput(tv1);
  fusion.addInput(s0);

  tv0 = canonicalizeInputToBMNK(tv0, layout, MmaOperand::A);
  tv1 = canonicalizeInputToBMNK(tv1, layout, MmaOperand::B);
  auto tv2 = fusedMultiplySum(tv0, tv1, {-1});
  auto tv3 = mul(s0, tv2);

  fusion.addOutput(tv3);

  MatMulTileOptions gemm_tile;
  gemm_tile.cta_tile = cta_tile;
  gemm_tile.warp_tile = warp_tile;

  MatmulParams mparams;
  mparams.supported_vec_size = {8, 8, 4};
  mparams.mma_macro = macro;
  mparams.tile_sizes = gemm_tile;
  mparams.async_gmem_load_operands = true;
  mparams.circular_buffer_options.circular_buffer_smem_write = true;
  mparams.circular_buffer_options.circular_buffer_smem_read = true;
  mparams.circular_buffer_options.smem_circular_buffer_stage = 4;
  SchedulerEntry::makeSchedulerInstance(SchedulerType::Matmul)
      ->schedule(&fusion, &mparams);

  at::manual_seed(0);
  auto inputs = matmulAtInput3DTuring(M, N, K, layout);
  const double alpha = 2.5;

  KernelExecutor ke;
  ke.compile(
      &fusion,
      {inputs.first, inputs.second, alpha},
      LaunchParams(),
      matmul_cparams);
  auto cg_outputs = ke.run({inputs.first, inputs.second, alpha});
  auto tref = at::mul(
                  atMatmul(
                      inputs.first.to(at::kFloat),
                      inputs.second.to(at::kFloat),
                      layout),
                  alpha)
                  .to(at::kFloat);

  NVF_CHECK(at::allclose(cg_outputs[0].as<at::Tensor>(), tref, 0.0001, 0.0001));

  return sass::parse(ke.compiledKernel()->disassembledKernelSASS());
}

} // namespace

TEST_P(MatmulSASSTestWithLayout, AmpereSanity) {
  NVFUSER_TEST_CUDA_ARCH_RANGE_GUARD(8, 0, 9, 0);

  // Keep multiples of 8 to keep vectorizable.
  int M = 504, N = 136, K = 248;

  bool found_LDGSTS = false;
  bool found_LDSM = false;
  bool found_HMMA = false;

  sass::Container sass;
  NVFUSER_TEST_CUDA_ARCH_COMPILE_CHECK(
      8,
      0,
      sass = getSASSFor(
          layout,
          GemmTile(128, 128, 32),
          GemmTile(64, 64, 32),
          MmaMacro::Ampere_16_8_16,
          M,
          N,
          K));
  for (const auto& inst : sass.code) {
    std::visit(
        [&](auto&& i) {
          using T = std::decay_t<decltype(i)>;
          if constexpr (std::is_same_v<sass::Instruction, T>) {
            if (i.opCode() == "LDGSTS") {
              found_LDGSTS = true;
            } else if (i.opCode() == "LDSM") {
              found_LDSM = true;
            } else if (i.opCode() == "HMMA") {
              found_HMMA = true;
            }
          }
        },
        inst);
  }
  NVF_CHECK(found_LDGSTS);
  NVF_CHECK(found_LDSM);
  NVF_CHECK(found_HMMA);
}

INSTANTIATE_TEST_SUITE_P(
    ,
    MatmulSASSTestWithLayout,
    testing::ValuesIn(kAllSupportedMmaLayout),
    mmaLayoutName);

// Check the modifiers of instructions. We are particularily interested in
// load/store, mma, and sync instructions. Currently, the ground truth in this
// test's asserts are based on experimental result of this test itself. In the
// future, we should use cutlass's kernel as ground truth.
TEST_F(MatmulSASSTest, AmpereModifiers) {
  NVFUSER_TEST_CUDA_ARCH_RANGE_GUARD(8, 0, 9, 0);
  // Keep multiples of 8 to keep vectorizable.
  int M = 504, N = 136, K = 248;
  bool found_LDGSTS = false;
  bool found_LDSM = false;
  bool found_HMMA = false;
  bool found_LDGDEPBAR = false;
  bool found_BAR = false;
  bool found_DEPBAR = false; // kAllSupportedMmaLayout;
  for (auto layout : {MmaLayout::TT}) {
    sass::Container sass;
    NVFUSER_TEST_CUDA_ARCH_COMPILE_CHECK(
        8,
        0,
        sass = getSASSFor(
            layout,
            GemmTile(128, 128, 32),
            GemmTile(64, 64, 32),
            MmaMacro::Ampere_16_8_16,
            M,
            N,
            K));
    for (const auto& inst : sass.code) {
      std::visit(
          [&](auto&& i) {
            using T = std::decay_t<decltype(i)>;
            if constexpr (std::is_same_v<sass::Instruction, T>) {
              if (i.opCode() == "LDGSTS") {
                const std::vector<std::string> expect = {
                    "E", "BYPASS", "LTC128B", "128"};
                NVF_CHECK(
                    i.modifiers() == expect,
                    "Modifiers for LDGSTS has changed. "
                    "Please manually check if the new modifiers makes sense "
                    "and update this test. "
                    "Expect: ",
                    expect,
                    " Get: ",
                    i.modifiers());
                found_LDGSTS = true;
              } else if (i.opCode() == "LDGDEPBAR") {
                const std::vector<std::string> expect;
                NVF_CHECK(
                    i.modifiers() == expect,
                    "Modifiers for LDGDEPBAR has changed. "
                    "Please manually check if the new modifiers makes sense "
                    "and update this test. "
                    "Expect: ",
                    expect,
                    " Get: ",
                    i.modifiers());
                found_LDGDEPBAR = true;
              } else if (i.opCode() == "LDSM") {
                const std::vector<std::string> expect1 = {"16", "M88", "2"};
                const std::vector<std::string> expect2 = {"16", "M88", "4"};
                const std::vector<std::string> expect3 = {"16", "MT88", "2"};
                const std::vector<std::string> expect4 = {"16", "MT88", "4"};
                NVF_CHECK(
                    i.modifiers() == expect1 || i.modifiers() == expect2 ||
                        i.modifiers() == expect3 || i.modifiers() == expect4,
                    "Modifiers for LDGDEPBAR has changed. "
                    "Please manually check if the new modifiers makes sense "
                    "and update this test.");
                found_LDSM = true;
              } else if (i.opCode() == "HMMA") {
                const std::vector<std::string> expect = {"16816", "F32"};
                NVF_CHECK(
                    i.modifiers() == expect,
                    "Modifiers for HMMA has changed. "
                    "Please manually check if the new modifiers makes sense "
                    "and update this test. "
                    "Expect: ",
                    expect,
                    " Get: ",
                    i.modifiers());
                found_HMMA = true;
              } else if (i.opCode() == "BAR") {
                const std::vector<std::string> expect = {
                    "SYNC", "DEFER_BLOCKING"};
                NVF_CHECK(
                    i.modifiers() == expect,
                    "Modifiers for BAR has changed. "
                    "Please manually check if the new modifiers makes sense "
                    "and update this test. "
                    "Expect: ",
                    expect,
                    " Get: ",
                    i.modifiers());
                found_BAR = true;
              } else if (i.opCode() == "DEPBAR") {
                const std::vector<std::string> expect = {"LE"};
                NVF_CHECK(
                    i.modifiers() == expect,
                    "Modifiers for DEPBAR has changed. "
                    "Please manually check if the new modifiers makes sense "
                    "and update this test. "
                    "Expect: ",
                    expect,
                    " Get: ",
                    i.modifiers());
                found_DEPBAR = true;
              }
            }
          },
          inst);
    }
    NVF_CHECK(found_LDGSTS);
    NVF_CHECK(found_LDSM);
    NVF_CHECK(found_HMMA);
    NVF_CHECK(found_LDGDEPBAR);
    NVF_CHECK(found_BAR);
    NVF_CHECK(found_DEPBAR);
  }
}

// Check the modifiers of instructions. We are particularily interested in
// load/store, mma, and sync instructions. Currently, the ground truth in this
// test's asserts are based on experimental result of this test itself. In the
// future, we should use cutlass's kernel as ground truth.
TEST_F(MatmulSASSTest, AmpereModifiersSharedMemoryEpilogue) {
  NVFUSER_TEST_CUDA_ARCH_RANGE_GUARD(8, 0, 9, 0);
  MatMulTileOptions gemm_tile;
  gemm_tile.cta_tile = GemmTile(128, 128, 32);
  gemm_tile.warp_tile = GemmTile(64, 64, 32);
  const int smem_circular_buffer_stage = 4;
  const bool ignore_occupancy_drop = true;
  const auto [use_smem_epilogue, promote_prologue_smem_reuse] =
      mma_utils::generateSharedMemoryEpilogueHeuristics(
          gemm_tile,
          smem_circular_buffer_stage,
          {DataType::Half, DataType::Half, DataType::Float},
          ignore_occupancy_drop);
  if (!use_smem_epilogue) {
    GTEST_SKIP() << "Test skipped due to the device's constrained shared "
                    "memory capacity.";
  }
  // Keep multiples of 8 to keep vectorizable.
  int M = 504, N = 136, K = 248;
  for (auto layout : {MmaLayout::TT}) {
    bool found_LDGSTS = false;
    bool found_LDSM = false;
    bool found_HMMA = false;
    bool found_LDGDEPBAR = false;
    bool found_DEPBAR = false; // kAllSupportedMmaLayout;
    int BAR_COUNT = 0;
    // we have at least 5 shared memory barriers in the kernel if
    // use_shared_epilogue. If promote_prologue_smem_reuse, then 7
    const int EXPECTED_BAR_COUNT = promote_prologue_smem_reuse ? 7 : 5;
    sass::Container sass;
    NVFUSER_TEST_CUDA_ARCH_COMPILE_CHECK(
        8,
        0,
        sass = getSASSFor(
            layout,
            gemm_tile.cta_tile,
            gemm_tile.warp_tile,
            MmaMacro::Ampere_16_8_16,
            M,
            N,
            K,
            smem_circular_buffer_stage,
            use_smem_epilogue,
            promote_prologue_smem_reuse));
    for (const auto& inst : sass.code) {
      std::visit(
          [&](auto&& i) {
            using T = std::decay_t<decltype(i)>;
            if constexpr (std::is_same_v<sass::Instruction, T>) {
              if (i.opCode() == "LDGSTS") {
                const std::vector<std::string> expect = {
                    "E", "BYPASS", "LTC128B", "128"};
                NVF_CHECK(
                    i.modifiers() == expect,
                    "Modifiers for LDGSTS has changed. "
                    "Please manually check if the new modifiers makes sense "
                    "and update this test. "
                    "Expect: ",
                    expect,
                    " Get: ",
                    i.modifiers());
                found_LDGSTS = true;
              } else if (i.opCode() == "LDGDEPBAR") {
                const std::vector<std::string> expect;
                NVF_CHECK(
                    i.modifiers() == expect,
                    "Modifiers for LDGDEPBAR has changed. "
                    "Please manually check if the new modifiers makes sense "
                    "and update this test. "
                    "Expect: ",
                    expect,
                    " Get: ",
                    i.modifiers());
                found_LDGDEPBAR = true;
              } else if (i.opCode() == "LDSM") {
                const std::vector<std::string> expect1 = {"16", "M88", "2"};
                const std::vector<std::string> expect2 = {"16", "M88", "4"};
                const std::vector<std::string> expect3 = {"16", "MT88", "2"};
                const std::vector<std::string> expect4 = {"16", "MT88", "4"};
                NVF_CHECK(
                    i.modifiers() == expect1 || i.modifiers() == expect2 ||
                        i.modifiers() == expect3 || i.modifiers() == expect4,
                    "Modifiers for LDGDEPBAR has changed. "
                    "Please manually check if the new modifiers makes sense "
                    "and update this test.");
                found_LDSM = true;
              } else if (i.opCode() == "HMMA") {
                const std::vector<std::string> expect = {"16816", "F32"};
                NVF_CHECK(
                    i.modifiers() == expect,
                    "Modifiers for HMMA has changed. "
                    "Please manually check if the new modifiers makes sense "
                    "and update this test. "
                    "Expect: ",
                    expect,
                    " Get: ",
                    i.modifiers());
                found_HMMA = true;
              } else if (i.opCode() == "BAR") {
                const std::vector<std::string> expect = {
                    "SYNC", "DEFER_BLOCKING"};
                NVF_CHECK(
                    i.modifiers() == expect,
                    "Modifiers for BAR has changed. "
                    "Please manually check if the new modifiers makes sense "
                    "and update this test. "
                    "Expect: ",
                    expect,
                    " Get: ",
                    i.modifiers());
                BAR_COUNT++;
              } else if (i.opCode() == "DEPBAR") {
                const std::vector<std::string> expect = {"LE"};
                NVF_CHECK(
                    i.modifiers() == expect,
                    "Modifiers for DEPBAR has changed. "
                    "Please manually check if the new modifiers makes sense "
                    "and update this test. "
                    "Expect: ",
                    expect,
                    " Get: ",
                    i.modifiers());
                found_DEPBAR = true;
              }
            }
          },
          inst);
    }
    NVF_CHECK(found_LDGSTS);
    NVF_CHECK(found_LDSM);
    NVF_CHECK(found_HMMA);
    NVF_CHECK(found_LDGDEPBAR);
    NVF_CHECK(
        BAR_COUNT == EXPECTED_BAR_COUNT,
        "Expect ",
        EXPECTED_BAR_COUNT,
        " BARs, got ",
        BAR_COUNT);
    NVF_CHECK(found_DEPBAR);
  }
}

TEST_F(MatmulSASSTest, AmpereEpilogueBinaryOpMul) {
  NVFUSER_TEST_CUDA_ARCH_RANGE_GUARD(8, 0, 9, 0);
  // Keep multiples of 8 to keep vectorizable.
  int M = 504, N = 136, K = 248;
  bool found_LDGSTS = false;
  bool found_LDSM = false;
  bool found_HMMA = false;
  bool found_LDGDEPBAR = false;
  bool found_BAR = false;
  bool found_DEPBAR = false;
  for (auto layout : {MmaLayout::TT}) {
    sass::Container sass;
    NVFUSER_TEST_CUDA_ARCH_COMPILE_CHECK(
        8,
        0,
        sass = getBinaryOpMulEpilogueSASSFor(
            layout,
            GemmTile(128, 128, 32),
            GemmTile(64, 64, 32),
            MmaMacro::Ampere_16_8_16,
            M,
            N,
            K));
    for (const auto& inst : sass.code) {
      std::visit(
          [&](auto&& i) {
            using T = std::decay_t<decltype(i)>;
            if constexpr (std::is_same_v<sass::Instruction, T>) {
              if (i.opCode() == "LDGSTS") {
                const std::vector<std::string> expect = {
                    "E", "BYPASS", "LTC128B", "128"};
                NVF_CHECK(
                    i.modifiers() == expect,
                    "Modifiers for LDGSTS has changed. "
                    "Please manually check if the new modifiers makes sense "
                    "and update this test. "
                    "Expect: ",
                    expect,
                    " Get: ",
                    i.modifiers());
                found_LDGSTS = true;
              } else if (i.opCode() == "LDGDEPBAR") {
                const std::vector<std::string> expect;
                NVF_CHECK(
                    i.modifiers() == expect,
                    "Modifiers for LDGDEPBAR has changed. "
                    "Please manually check if the new modifiers makes sense "
                    "and update this test. "
                    "Expect: ",
                    expect,
                    " Get: ",
                    i.modifiers());
                found_LDGDEPBAR = true;
              } else if (i.opCode() == "LDSM") {
                const std::vector<std::string> expect1 = {"16", "M88", "2"};
                const std::vector<std::string> expect2 = {"16", "M88", "4"};
                const std::vector<std::string> expect3 = {"16", "MT88", "2"};
                const std::vector<std::string> expect4 = {"16", "MT88", "4"};
                NVF_CHECK(
                    i.modifiers() == expect1 || i.modifiers() == expect2 ||
                        i.modifiers() == expect3 || i.modifiers() == expect4,
                    "Modifiers for LDGDEPBAR has changed. "
                    "Please manually check if the new modifiers makes sense "
                    "and update this test.");
                found_LDSM = true;
              } else if (i.opCode() == "HMMA") {
                const std::vector<std::string> expect = {"16816", "F32"};
                NVF_CHECK(
                    i.modifiers() == expect,
                    "Modifiers for HMMA has changed. "
                    "Please manually check if the new modifiers makes sense "
                    "and update this test. "
                    "Expect: ",
                    expect,
                    " Get: ",
                    i.modifiers());
                found_HMMA = true;
              } else if (i.opCode() == "BAR") {
                const std::vector<std::string> expect = {
                    "SYNC", "DEFER_BLOCKING"};
                NVF_CHECK(
                    i.modifiers() == expect,
                    "Modifiers for BAR has changed. "
                    "Please manually check if the new modifiers makes sense "
                    "and update this test. "
                    "Expect: ",
                    expect,
                    " Get: ",
                    i.modifiers());
                found_BAR = true;
              } else if (i.opCode() == "DEPBAR") {
                const std::vector<std::string> expect = {"LE"};
                NVF_CHECK(
                    i.modifiers() == expect,
                    "Modifiers for DEPBAR has changed. "
                    "Please manually check if the new modifiers makes sense "
                    "and update this test. "
                    "Expect: ",
                    expect,
                    " Get: ",
                    i.modifiers());
                found_DEPBAR = true;
              }
            }
          },
          inst);
    }
    NVF_CHECK(found_LDGSTS);
    NVF_CHECK(found_LDSM);
    NVF_CHECK(found_HMMA);
    NVF_CHECK(found_LDGDEPBAR);
    NVF_CHECK(found_BAR);
    NVF_CHECK(found_DEPBAR);
  }
}

#if 0

TODO: With swizzle, the cuda code looks like:

#pragma unroll
for(nvfuser_index_t i507 = 0; i507 < 8; ++i507) {
  int i18439;
  i18439 = i18438 + i507;
  Turing::ldMatrixT (*reinterpret_cast<Array<__half,4,4>*>(&T9[(4 * i507)]),((i18437 + (128 * (i18439 / 8))) + (16 * (i6455 ^ (i18439 % 8)))));
}

where i6455 = (((nvfuser_index_t)threadIdx.x) % 16) % 8 so it no longer make sense to require the memory access pattern below.
We need to reinvestigate the test below to determine whether to change it or delete it.

// Check that all LDSM instructions has the following pattern:
//   LDSM.16.M88.2 R2,   [R213] ;
//   LDSM.16.M88.2 R136, [R213+0x200] ;
//   LDSM.16.M88.2 R138, [R213+0x400] ;
//   LDSM.16.M88.2 R140, [R213+0x600] ;
//   LDSM.16.M88.2 R142, [R213+0x800] ;
//   LDSM.16.M88.2 R144, [R213+0xa00] ;
//   LDSM.16.M88.2 R146, [R213+0xc00] ;
//   LDSM.16.M88.2 R148, [R213+0xe00] ;
TEST_F(MatmulSASSTest, AmpereRegisterUsageLDSM) {
  // Keep multiples of 8 to keep vectorizable.
  int M = 504, N = 136, K = 248;

  for (auto layout : kAllSupportedMmaLayout) {
    std::unordered_map<std::string, std::unordered_set<int>> base_offsets;

    sass::Container sass;
    NVFUSER_TEST_CUDA_ARCH_COMPILE_CHECK(
        8,
        0,
        sass = getSASSFor(
            layout,
            GemmTile(128, 128, 32),
            GemmTile(64, 64, 32),
            MmaMacro::Ampere_16_8_16,
            M,
            N,
            K));
    for (const auto& inst : sass.code) {
      std::visit(
          [&](auto&& i) {
            using T = std::decay_t<decltype(i)>;
            if constexpr (std::is_same_v<sass::Instruction, T>) {
              if (i.opCode() != "LDSM") {
                return;
              }
              auto args = i.args();
              NVF_ERROR(args.size() == 2);
              std::string smem_address = args[1];
              // get base shared memory address
              std::string_view view(smem_address); // example: [R0+UR0+0x200]
              view = view.substr(1, view.size() - 2); // example: R0+UR0+0x200
              std::string_view base;
              int offset = 0;
              using namespace std::literals;
              auto last = view.find_last_of("+"sv);
              if (last == std::string::npos ||
                  view.substr(last + 1, 2) != "0x"sv) {
                // [R0] or [R0+UR0]
                base = view;
                offset = 0;
              } else {
                // [R0+0x200] or [R0+UR0+0x200]
                base = view.substr(0, last);
                std::stringstream ss(std::string(view.substr(last + 1)));
                ss >> std::hex >> offset;
              }
              base_offsets[std::string(base)].insert(offset);
            }
          },
          inst);
    }
    for (auto& [base, offsets] : base_offsets) {
      NVF_CHECK(
          offsets.size() > 1,
          "Expect base addresses to be used multiple times, but ",
          base,
          " is only used once");
    }
  }
}
#endif

} // namespace nvfuser
