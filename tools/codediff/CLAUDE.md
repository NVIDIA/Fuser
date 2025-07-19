# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Common Development Commands

### Running Tools
- **Install dependencies**: `pip install -r requirements.txt`
- **Compare codegen between commits**: `./compare_codegen.sh` (compares against origin/main by default)
- **Compare specific git revisions**: `./compare_codegen.sh -r <commit_hash>`
- **Generate diff report**: `python diff_report.py --html -o diff.html <dir1> <dir2>`
- **Run single command comparison**: `./run_command.sh -o <output_dir> -- <command>`

### Example Workflows
```bash
# Compare current branch against main
./compare_codegen.sh

# Compare against specific commit
./compare_codegen.sh -r abc123

# Run specific test and compare
./compare_codegen.sh -- build/test_nvfuser --gtest_filter='*TestFoo*'

# Generate manual comparison
./run_command.sh -o run1 -- <first_command>
./run_command.sh -o run2 -- <second_command>
python diff_report.py --html -o report.html run1 run2
```

## Code Architecture

### Core Components
- **codediff.py**: Main library containing data structures and parsing logic
  - `TestRun`: Represents a single process execution with generated kernels
  - `TestDifferences`: Compares two TestRun instances and identifies changes
  - `GitRev`, `CompiledKernel`, `CompiledTest`: Supporting data structures
  - Multiple log parsers for different test frameworks (GTest, GBench, PyTest)

- **diff_report.py**: CLI tool that creates HTML reports from TestRun directories
- **compare_codegen.sh**: High-level script for comparing git commits
- **run_command.sh**: Low-level script for capturing single command execution

### Data Flow
1. `run_command.sh` executes commands and captures output + environment info
2. `codediff.py` parses logs and extracts generated CUDA kernels (.cu files)
3. `diff_report.py` compares two runs and generates HTML reports with diffs
4. Templates in `templates/` directory generate the final HTML output

### Key Design Patterns
- Uses dataclasses with JSON serialization (`dataclasses-json`)
- Jinja2 templating for HTML report generation
- Enum-based command type detection (binary tests, python tests, benchmarks)
- Modular log parsing with inheritance hierarchy

## Important Notes
- Tools expect to run from within NVFuser git repository
- Generated CUDA kernels are captured when `NVFUSER_DUMP=cuda_kernel` is set
- Multiple NVFuser processes in a single command will cause kernel collision
- Setting `NVFUSER_DUMP=launch_param` provides additional kernel launch info but can clutter output