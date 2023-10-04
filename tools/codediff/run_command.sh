#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# run_command.sh - run a single command and record environment info
#

set -e
set -o pipefail

usage() {
  echo "Usage: $0 <output_directory> -- command to run and arguments"
}

if [[ $# -lt 1 ]]
then
    usage
    exit 1
fi

testdir=$1
shift

commandmissing=true
while [[ $# -gt 0 ]]
do
  if [[ "$1" == "--" ]]
  then
    if [[ $# -gt 1 ]]
    then
        commandmissing=false
    fi
    shift
    break
  fi
  shift
done
if $commandmissing
then
    usage
    exit 1
fi

if [[ -d "$testdir/command" ]]
then
    echo -n "Skipping since $testdir/command exists. "
    echo "To re-run, remove the $testdir and try again."
    exit 1
fi

testcmd=$*

mkdir -p "$testdir"

movecudafiles() {
    mkdir -p "$1/cuda" "$1/ptx"
    find . -maxdepth 1 -name '__tmp_kernel*.cu' -exec mv '{}' "$1/cuda" \;
    find . -maxdepth 1 -name '__tmp_kernel*.ptx' -exec mv '{}' "$1/ptx" \;
}

removecudafiles() {
    tmpdir="./.nvfuser_run_command_tmp"
    mkdir -p "$tmpdir"
    movecudafiles "$tmpdir"
    rm -rf "$tmpdir"
}

stdoutfile="$testdir/stdout-$(date +%Y%m%d_%H%M%S).log"

cleanup() {
    numcu=$(find . -maxdepth 1 -name '__tmp_kernel*.cu' | wc -l)
    numptx=$(find . -maxdepth 1 -name '__tmp_kernel*.ptx' | wc -l)

    if (( numcu + numptx > 0 ))
    then
        echo "Interrupted. Removing $numcu temporary .cu files and $numptx temporary .ptx files"
        removecudafiles
    fi

    mv "$stdoutfile" "$testdir/interrupted-$(basename "$stdoutfile")"
}

trap "cleanup" EXIT

# Allow command to fail, but record exit code
set +e
$testcmd | tee "$stdoutfile"
echo $? > "$testdir/exitcode"
set -e

# Save command metadata in output directory.
# We do this after running the command, so that if $testdir/command exists, we
# know there was already a completed run.
echo "$testcmd" > "$testdir/command"
git rev-parse > /dev/null
in_git_repo=$?
if [ "$in_git_repo" ]
then
    git rev-parse HEAD > "$testdir/git_hash"
else
    echo -n "WARNING: $0 expects to be run from the NVFuser git repository."
    echo -n "You do not appear to be in a git repository, so we will not be "
    echo "able to track git information for this command output to $testdir."
fi
printenv > "$testdir/env"
nvcc --version > "$testdir/nvcc_version"
nvidia-smi --query-gpu=gpu_name --format=csv,noheader > "$testdir/gpu_names"

# save generated cuda and ptx files
movecudafiles "$testdir"

cleanup
