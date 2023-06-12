// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#include <c10/util/Exception.h>
#include <exceptions.h>

#if defined(__linux__)

#include <array>
#include <filesystem>
#include <sstream>
#include <string>
#include <vector>

#include <sys/types.h>
#include <unistd.h>
#include <cstdio>
#include <cstdlib>

namespace nvfuser::executor_utils {

std::string disassembleBinary(
    const std::vector<char>& cubin,
    const std::string& nvdisasm_args) {
  const char* err = "Failed to disassemble cubin";

  // Reference:
  // https://stackoverflow.com/a/3469651
  // https://linuxhint.com/dup2_system_call_c/

  constexpr int READ = 0, WRITE = 1;
  std::array<int, 2> cubin_pipe{-1, -1};
  std::array<int, 2> disasm_pipe = {-1, -1};
  std::array<int, 2> err_pipe = {-1, -1};

  NVF_ERROR(
      pipe(cubin_pipe.data()) == 0 && pipe(disasm_pipe.data()) == 0 &&
          pipe(err_pipe.data()) == 0,
      err);

  pid_t pid = fork();
  NVF_ERROR(pid != -1, err);

  if (pid) { // I am the parent
    // Parent only write cubin and read disasm, close unused pipe end
    NVF_ERROR(close(cubin_pipe[READ]) == 0, err);
    NVF_ERROR(close(disasm_pipe[WRITE]) == 0, err);
    NVF_ERROR(close(err_pipe[WRITE]) == 0, err);

    // Wrap pipe fileno as C file stream
    FILE* cubin_fp = fdopen(cubin_pipe[WRITE], "wb");
    FILE* disasm_fp = fdopen(disasm_pipe[READ], "r");
    FILE* err_fp = fdopen(err_pipe[READ], "r");
    NVF_ERROR(cubin_fp != nullptr, err);
    NVF_ERROR(disasm_fp != nullptr, err);
    NVF_ERROR(err_fp != nullptr, err);

    // Write cubin to nvdisasm
    size_t written = fwrite(cubin.data(), 1, cubin.size(), cubin_fp);
    NVF_ERROR(written == cubin.size(), err);
    fclose(cubin_fp);

    int ch = -1;

    // read disassembly result
    std::string result;
    result.reserve(cubin.size());
    while ((ch = fgetc(disasm_fp)) != EOF) {
      result.push_back((char)ch);
    }
    fclose(disasm_fp);

    // read error message
    std::string error;
    while ((ch = fgetc(err_fp)) != EOF) {
      error.push_back((char)ch);
    }
    fclose(err_fp);
    NVF_CHECK(error.empty(), error);

    return result;
  } else { // I am the child
    // For easier understanding, we can consider the fileno as a smart pointer
    // pointing to an underlying IO object in the kernel. Both the pointer and
    // the underlying objects are owned by the kernel, and multiple pointers
    // can point to the same object. `close` destroy the pointer, which does
    // not necessarily destroy the object.

    // Modify the stdin, stdout and stderr pointer to point to the pipe object
    NVF_ERROR(close(STDIN_FILENO) == 0, err);
    NVF_ERROR(close(STDOUT_FILENO) == 0, err);
    NVF_ERROR(close(STDERR_FILENO) == 0, err);
    NVF_ERROR(dup2(cubin_pipe[READ], STDIN_FILENO) != -1, err);
    NVF_ERROR(dup2(disasm_pipe[WRITE], STDOUT_FILENO) != -1, err);
    NVF_ERROR(dup2(err_pipe[WRITE], STDERR_FILENO) != -1, err);

    // Now we have stdin, stdout and stderr pointing to the pipe object, we no
    // longer need the original pointers.
    NVF_ERROR(close(cubin_pipe[READ]) == 0, err);
    NVF_ERROR(close(cubin_pipe[WRITE]) == 0, err);
    NVF_ERROR(close(disasm_pipe[READ]) == 0, err);
    NVF_ERROR(close(disasm_pipe[WRITE]) == 0, err);
    NVF_ERROR(close(err_pipe[READ]) == 0, err);
    NVF_ERROR(close(err_pipe[WRITE]) == 0, err);

    // If execl succeed, then the current process will be replaced by nvdisasm,
    // and all the remaining code in the current process will not be executed.
    // So, execl only returns when it fails.
    //
    // TODO: I was planning to use `nvdisasm /dev/stdin` which could avoid
    // creating temporary file, but unfortunately, that fails with:
    //   nvdisasm fatal   : Memory allocation failure
    // so I have to dump the stdin to a temp file and let nvdisasm read it. I am
    // hoping that nvdisasm will support reading from stdin one day.
    std::stringstream ss;
    ss << "export PATH=$PATH:/usr/local/cuda/bin;"
       << "TMPFILE=$(mktemp);"
       << "cat>$TMPFILE;"
       << "nvdisasm $TMPFILE " << nvdisasm_args << "; rm $TMPFILE";
    auto command = ss.str();
    execl("/bin/bash", "bash", "-c", command.c_str(), NULL);

    // only reachable when execl fails
    NVF_ERROR(false, err);
  }
}

} // namespace nvfuser::executor_utils

#else

namespace nvfuser::executor_utils {

std::string disassembleBinary(const std::vector<char>& binary) {
  NVF_CHECK(false, "disassembling cubin is only supported on Linux");
}

} // namespace nvfuser::executor_utils

#endif
