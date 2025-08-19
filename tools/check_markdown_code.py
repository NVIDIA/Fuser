#!/usr/bin/env python3
"""
CI check script that reads all *.md files recursively, extracts Python code blocks,
and executes them to ensure no errors are thrown.

This script:
1. Finds all *.md files while respecting .gitignore and excluding third_party/
2. Extracts code blocks marked with ```python
3. Skips blocks preceded by <!-- CI IGNORE --> comments
4. Uses AST analysis to detect 'fd' variable usage and auto-wraps with FusionDefinition context manager
5. Sets __name__ = '__main__' to allow execution of main blocks
6. Automatically imports common nvfuser modules (torch, nvfuser, FusionDefinition, DataType)
7. Executes Python code blocks and reports any errors
"""

import ast
import sys
import subprocess
from pathlib import Path
from typing import List, Tuple, Dict


def is_git_ignored(file_path: Path, repo_root: Path) -> bool:
    """Check if a file is git-ignored using git check-ignore."""
    try:
        result = subprocess.run(
            ["git", "check-ignore", str(file_path)],
            cwd=repo_root,
            capture_output=True,
            text=True,
        )
        return result.returncode == 0
    except (subprocess.SubprocessError, FileNotFoundError):
        return False


def find_markdown_files(repo_root: Path) -> List[Path]:
    """Find all *.md files recursively, excluding git-ignored and third_party/ files."""
    md_files = []

    for md_file in repo_root.rglob("*.md"):
        # Skip third_party directory
        if "third_party" in md_file.parts:
            continue

        # Skip git-ignored files
        if is_git_ignored(md_file, repo_root):
            continue

        md_files.append(md_file)

    return sorted(md_files)


def extract_code_blocks(content: str) -> List[Tuple[str, int, str, bool]]:
    """
    Extract code blocks from markdown content.

    Returns:
        List of tuples: (code, line_number, language, should_ignore)
    """
    code_blocks = []
    lines = content.split("\n")
    i = 0

    while i < len(lines):
        line = lines[i].strip()

        # Look for code block start
        if line.startswith("```"):
            start_line = i + 1
            language = line[3:].strip() if len(line) > 3 else ""

            # Check if the previous line contains CI IGNORE comment
            should_ignore = False
            if i > 0:
                prev_line = lines[i - 1].strip()
                if prev_line == "<!-- CI IGNORE -->":
                    should_ignore = True

            # Find end of code block
            i += 1
            code_lines = []
            while i < len(lines) and not lines[i].strip().startswith("```"):
                code_lines.append(lines[i])
                i += 1

            if code_lines:
                code = "\n".join(code_lines)
                code_blocks.append((code, start_line, language, should_ignore))

        i += 1

    return code_blocks


def analyze_code_for_fd_usage(code: str) -> bool:
    """
    Use AST to check if code uses 'fd' variable but doesn't define it.

    Returns:
        True if code uses 'fd' but doesn't define it (needs context manager)
    """
    try:
        tree = ast.parse(code)

        # Track if 'fd' is defined (assigned to)
        fd_defined = False
        # Track if 'fd' is used (accessed)
        fd_used = False

        class FdAnalyzer(ast.NodeVisitor):
            def visit_Assign(self, node):
                # Check if 'fd' is being assigned to
                for target in node.targets:
                    if isinstance(target, ast.Name) and target.id == "fd":
                        nonlocal fd_defined
                        fd_defined = True
                self.generic_visit(node)

            def visit_Name(self, node):
                # Check if 'fd' is being accessed
                if node.id == "fd" and isinstance(node.ctx, ast.Load):
                    nonlocal fd_used
                    fd_used = True
                self.generic_visit(node)

            def visit_Attribute(self, node):
                # Check for fd.something usage
                if isinstance(node.value, ast.Name) and node.value.id == "fd":
                    nonlocal fd_used
                    fd_used = True
                self.generic_visit(node)

        analyzer = FdAnalyzer()
        analyzer.visit(tree)

        # Return True if fd is used but not defined
        return fd_used and not fd_defined

    except SyntaxError:
        # If code has syntax errors, don't try to wrap it
        return False


def execute_python_code(
    code: str, file_path: Path, line_number: int
) -> Tuple[bool, str]:
    """
    Execute Python code and return success status and error message.

    Returns:
        Tuple of (success, error_message)
    """
    try:
        # Prepend common imports for nvfuser code
        imports = """
from typing import Callable
import torch
import nvfuser
from nvfuser import FusionDefinition, DataType
__name__ = '__main__'
"""

        # Check if we need to wrap with FusionDefinition context manager
        if analyze_code_for_fd_usage(code):
            # Indent the code and wrap with context manager
            indented_code = "\n".join("    " + line for line in code.split("\n"))
            full_code = imports + "\nwith FusionDefinition() as fd:\n" + indented_code
        else:
            # Just prepend imports
            full_code = imports + "\n" + code

        # Execute the code using python -c
        result = subprocess.run(
            [sys.executable, "-c", full_code],
            capture_output=True,
            text=True,
            timeout=30,  # 30 second timeout
        )

        if result.returncode == 0:
            return True, ""
        else:
            return False, result.stderr.strip()

    except subprocess.TimeoutExpired:
        return False, "Code execution timed out (30s)"
    except Exception as e:
        return False, f"Execution error: {str(e)}"


def main():
    """Main function to run the markdown code check."""
    repo_root = Path(__file__).parent.parent  # Assume script is in tools/

    print("🔍 Finding markdown files...")
    md_files = find_markdown_files(repo_root)
    print(f"Found {len(md_files)} markdown files to check")

    total_code_blocks = 0
    total_python_blocks = 0
    failed_blocks = 0
    errors: List[Dict] = []

    for md_file in md_files:
        print(f"📄 Checking {md_file.relative_to(repo_root)}")

        try:
            content = md_file.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            print(f"⚠️  Skipping {md_file} - encoding error")
            continue

        code_blocks = extract_code_blocks(content)
        total_code_blocks += len(code_blocks)

        for code, line_number, language, should_ignore in code_blocks:
            if language == "python":
                if should_ignore:
                    print(
                        f"  ⏭️ Skipping Python code at line {line_number} (CI IGNORE)"
                    )
                    continue

                total_python_blocks += 1
                print(f"  🐍 Executing Python code at line {line_number}")

                success, error_msg = execute_python_code(code, md_file, line_number)

                if not success:
                    failed_blocks += 1
                    error_info = {
                        "file": md_file,
                        "line": line_number,
                        "error": error_msg,
                        "code": code[:200] + "..." if len(code) > 200 else code,
                    }
                    errors.append(error_info)
                    print(f"    ❌ Error: {error_msg}")
                else:
                    print("    ✅ OK")

    # Print summary
    print("\n" + "=" * 60)
    print("📊 SUMMARY")
    print("=" * 60)
    print(f"Markdown files checked: {len(md_files)}")
    print(f"Total code blocks found: {total_code_blocks}")
    print(f"Python code blocks executed: {total_python_blocks}")
    print(f"Failed executions: {failed_blocks}")

    if errors:
        return 1
    else:
        print("\n✅ All Python code blocks executed successfully!")
        return 0


if __name__ == "__main__":
    sys.exit(main())
