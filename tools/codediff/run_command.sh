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
echo "Usage: $0 [-h] [-n <run_name>] -o <output_directory> -- command to run and arguments"
}
while getopts "n:o:h" arg
do
case $arg in
n)
runname=$OPTARG
;;
o)
testdir=$OPTARG
;;
h | *)
usage
exit 1
;;
esac
done
if [[ -z $testdir ]]
then
echo "The output directory must be specified with -o"
usage
exit 1
fi
# getopts stops parsing if it sees "--" but does not shift. We can detect that case and record command
while [[ $# -gt 0 ]]
do
if [[ "$1" == "--" ]]
then
shift
break
fi
shift
done
testcmd=$*
if [[ -z "$testcmd" ]]
then
usage
exit 1
fi
if [[ -f "$testdir/command" ]]
then
echo -n "Skipping since $testdir/command exists. "
echo "To re-run, remove $testdir or specify another and try again."
exit 1
fi
mkdir -p "$testdir"
movecudafiles() {
mkdir -p "$1/cuda" "$1/ptx"
find . -maxdepth 1 -name '__tmp_kernel*.cu' -print0 | xargs -0 mv -t "$1/cuda"
find . -maxdepth 1 -name '__tmp_kernel*.ptx' -print0 | xargs -0 mv -t "$1/ptx"
}
removecudafiles() {
tmpdir="./.nvfuser_run_command_tmp"
mkdir -p "$tmpdir"
movecudafiles "$tmpdir"
rm -rf "$tmpdir"
}
stdoutfile="$testdir/incomplete-stdout-$(date +%Y%m%d_%H%M%S).log"
cleanup() {
numcu=$(find . -maxdepth 1 -name '__tmp_kernel*.cu' | wc -l)
numptx=$(find . -maxdepth 1 -name '__tmp_kernel*.ptx' | wc -l)
if (( numcu + numptx > 0 ))
then
echo "Interrupted. Removing $numcu temporary .cu files and $numptx temporary .ptx files"
removecudafiles
fi
logbase=$(basename "$stdoutfile")
# strip incomplete- from base name
completelogbase=${logbase#incomplete-}
mv "$stdoutfile" "$testdir/$completelogbase" 2> /dev/null || true
}
trap "cleanup" EXIT

# Ensure given strings are elements of comma-separated string
# Usage: ensure_in_list input_str a b c ...
# Returns: comma-separated string input_str with a, b, c, ... appended if
# needed
ensure_in_list() {
    IFS=","
    read -ra l <<< "$1"
    shift
    for req in "$@"
    do
        joined="${l[*]}"
        if [[ ",$joined," != *",$req,"* ]]
        then
            l+=("$req")
        fi
    done
    echo "${l[*]}"
}
# ensure some NVFUSER_DUMP options are enabled
NVFUSER_DUMP=$(ensure_in_list "$NVFUSER_DUMP" cuda_kernel ptxas_verbose ptx)

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
# the following shows both staged and unstaged changes
git diff --no-ext-diff HEAD > "$testdir/git_diff"
else
echo -n "WARNING: $0 expects to be run from the NVFuser git repository."
echo -n "You do not appear to be in a git repository, so we will not be "
echo "able to track git information for this command output to $testdir."
fi
if [ -n "$runname" ]
then
echo "$runname" > "$testdir/run_name"
fi
printenv | sort > "$testdir/env"
nvcc --version > "$testdir/nvcc_version"
nvidia-smi --query-gpu=gpu_name --format=csv,noheader > "$testdir/gpu_names"
# save generated cuda and ptx files
movecudafiles "$testdir"
