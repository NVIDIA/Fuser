# Codegen comparison tools

This directory contains tools that are useful for studying generated code under
changing circumstances like different commits or different `NVFUSER_ENABLE`
options.

## Typical workflows

There are two intended use cases for these tools: comparing changes in a PR
branch with another branch (`origin/main` by default), or comparing other types
of changes. The latter case is more general, so the former case is handled by a
wrapper called `compare_codegen.sh`. Both types of use cases are described in
the sections that follow.

### Comparing general changes

You may want to look at the effect of some changes to code, or under different
environment variables or command line options between two command invocations.
To do that, use the following steps.

1. Call `run_command.sh -o outdir1 -- first command` where the required
   argument `outdir1` is replaced by any output directory name you like, and
   `first command` can be any command that launches **a single NVFuser
   process*. If multiple NVFuser processes are launched by this command, we
   will be unable to differentiate their generated kernels, and some of them
   will be lost in the analysis. This command will fill the directory `outdir1`
   with files. Note that setting `NVFUSER_DUMP=launch_param` can clutter the
   output log so it is not recommended when a huge number of kernel launches is
   expected; however, including it will enable us to extract block and grid
   dimensions and dynamic shared memory size.
2. Do the same thing for the second command. I'll call its output directory
   `outdir2`. These commands do not need to be identical, so you could change
   some options for example.
3. Run `diff_report.py --html -o diff.html outdir1 outdir2` to generate an HTML
   report of differences, including differences in NVFuser git repository
   status, environment variables, and code differences in both NVFuser and the
   generated CUDA code. See `diff_report.py --help` for a complete list of options.

### Comparing git revisions of NVFuser

We include the `compare_codegen.sh` script to automate calling `run_command.sh`
and `diff_report.py` for comparing different git repository commits. You can
replace all three steps above with `compare_codegen.sh`, which will run all
binary and python tests (but not benchmarks), and diff them each to generate
HTML reports in the specified output directory (default:
`codegen_comparison/`). By default `origin/main` is compared to (we do not do
`git fetch` first), but you might prefer to compare against `git merge-base
origin/main HEAD` or to a specific revision. You can pass the preferred
commit/branch with the `-r` option.
