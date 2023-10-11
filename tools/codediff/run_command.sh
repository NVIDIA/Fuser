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
    echo -n "Usage: $0 [-h] [-n <run_name>] [-t <command_type>] "
    echo "-o <output_directory> -- command to run and arguments"

    cat << EOF

This script will run the command line (given as all arguments following --) and
gather the results into output_directory. It is meant to be used with
diff_report.py, which can consume pairs of output directories in this format.

It is assumed that multiple commands might be run using this script for any
given \"run\". A run is then a collection of commands all using the same state
such as the same git commit or environment variables. To aid in readability,
you may provide a short name for this run, which if given should match across
commands within the same run.

The diff_report.py tool will parse the STDOUT of these commands to collect
information about CUDA kernels and group them appropriately. It must know what
type of command this is in order to do so properly. By default, we look for
nvfuser_tests, nvfuser_bench, pytest, and python_tests as substrings in the
given command. If this fails, you may provide the -t option to record a
different command type. See diff_report.py (CommandType) for possible types.
EOF

}
while getopts "n:o:t:h" arg
    do
        case $arg in
            n)
                runname=$OPTARG
                ;;
            o)
                testdir=$OPTARG
                ;;
            t)
                commandtype=$OPTARG
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
# Save command type
if [[ -z $commandtype ]]
then
    case "$testcmd" in
        *nvfuser_tests*)
            commandtype="GOOGLETEST"
            ;;
        *nvfuser_bench*)
            commandtype="GOOGLEBENCH"
            ;;
        *pytest*)
            commandtype="PYTEST"
            ;;
        *python_tests*)
            commandtype="PYTEST"
            ;;
        *)
            >&2 echo "WARNING: Could not determine command type. Assuming UNKNOWN."
            commandtype="UNKNOWN"
            ;;
    esac
fi
echo "$commandtype" > "$testdir/command_type"
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
