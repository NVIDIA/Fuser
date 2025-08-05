import subprocess
import pathlib
import argparse
import re 

# server.py
from mcp.server.fastmcp import FastMCP

# Create an MCP server
mcp = FastMCP("Nvfuser Dev MCP Server")
PROJECT_ROOT = pathlib.Path(__file__).parent.parent.parent
BUILD_DIR = PROJECT_ROOT / "build"


def run_command(command: list[str], cwd: pathlib.Path) -> str: 
    """ 
    Helper function for running commands in the MCP server
    
    Args: 
        command (list[str]): The command to run as a list of strings.
        cwd (pathlib.Path): The directory in which to run the command.
    Returns:
        str: The output of the command.
    """
    try: 
        p = subprocess.run(
            command, 
            cwd=cwd, 
            check=False, 
            capture_output=True,
            text=True, 
            encoding='utf-8'
        )
        success = p.returncode == 0
        return success, p.stdout, p.stderr 
    except Exception as e:
        return False, "", f"Error: {str(e)}"


@mcp.tool()
def build_nvfuser() -> str:
    """
    Build the nvfuser project

    Returns:
        str:
            A message indicating whether the build was successful or failed.
            If the build failed, the full build log is included in the message.
    """
    args = parse_args()
    # if script `/usr/local/bin/_bn` exists, use it
    if pathlib.Path("/usr/local/bin/_bn").exists():
        command = ["/usr/local/bin/_bn"]
    else:
        command = [
            pathlib.Path(args.python_path),
            "-m",
            "pip",
            "install",
            "-v",
            "-e",
            "./python",
            "--no-build-isolation",
        ]

    p = subprocess.run(
        command,
        cwd=PROJECT_ROOT,  # go to nvfuser root
        check=False,
        capture_output=True,
        text=True,
    )
    if p.returncode == 0:
        return "Nvfuser build was successful"
    else:
        return "Nvfuser build was failed, here are the full build log\n\n\n" + p.stderr


def _generate_test_selection_prompt(diff_content: str) -> str: 
    """
    This function generated a prompt for LLMs to select relevant 
    tests based on diff content. 

    Args: 
        diff_content (str): The content of the diff to analyze.
    
    Returns: 
        str: The prompt for the LLM.
    """
    prompt = (
        "You are a test selection assistant. Given the following diff content, "
        "select the relevant tests that should be run based on the changes made. "
        "Return the names of the tests as a list of strings.\n\n"
        f"Diff Content:\n{diff_content}\n\n"
        "Your Task:\n"
        "Carefully review the code changes. Based on the functions, data structure, and logic"
        "affected identify the tests from that we should run."
        "- Prioritize tests that directly cover the changed code.\n"
        " - Consider tests of downstream components that might be affected by these changes.\n"
        " - If you are unsure about the relevance of a test, include it.\n"
        " - Tests are located in the `Fuser/tests/cpp` directory.\n"
        "Output format:\n"
        "Return a single line, containing a comma-separate list wiht the name of the test names to run."
        "Do not add any other text, explanation or formatting.\n"
        "Example of output:  nvfuser_tests.test_fusion,nvfuser_tests.test_scheduler,nvfuser_tests.test_codegen\n"
    )
    return prompt.strip()


@mcp.tool()
def run_targeted_tests(target_branch: str = "devel") -> str: 
    """
    This function checks for changes in the Cpp files, against a target_branch.
    It then finds the corresponding existent tests, and runs only that subset.
    
    Args: 
        target_branch (str): The branch to compare against, defaults to "devel".
    Returns: 
        str: The result of the tests that have been run.
    """
    # retrieve all the files that have been changed wrt to HEAD 
    diff_command = ["git", "diff", "--name-only", f"{target_branch}...HEAD", "--", "*.cpp", "*.h"]
    success, diff_content, stderr = run_command(diff_command, PROJECT_ROOT)
    if not success: 
        return f"Failed to get the diff against {target_branch}. Error: {stderr}"
    if not diff_content.strip(): 
        return f"No Cpp files have been changed against {target_branch}. No tests to run."
    
    # TODO: should we have a simple heuristic search based on regex from changed files? 
    #changed_files = diff_content.strip().split("\n")

    # detect all the possibel tests from Fuser/tests/cpp
    return _generate_test_selection_prompt(diff_content.strip())


def parse_args():
    """ Input parser"""
    parser = argparse.ArgumentParser(description="MCP server")
    parser.add_argument(
       "--python-path",
        type=str,
        default=None,
    )

    args = parser.parse_args()

    return args


if __name__ == "__main__":
    # Start the server
    mcp.run()
