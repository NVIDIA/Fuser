import subprocess
import pathlib
import argparse

# server.py
from mcp.server.fastmcp import FastMCP

# Create an MCP server
mcp = FastMCP("Nvfuser Dev MCP Server")


@mcp.tool()
def build_nvfuser() -> str:
    """
    Build the nvfuser project

    Returns:
        str:
            A message indicating whether the build was successful or failed.
            If the build failed, the full build log is included in the message.
    """

    # if script `/usr/local/bin/_bn` exists, use it
    command = []
    if pathlib.Path("/usr/local/bin/_bn").exists():
        command = ["/usr/local/bin/_bn"]
    else:
        args = parse_args()
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
        cwd=pathlib.Path(__file__).parent.parent.parent,  # go to nvfuser root
        check=False,
        capture_output=True,
        text=True,
    )
    if p.returncode == 0:
        return "Nvfuser build was successful"

    return "Nvfuser build was failed, here are the full build log\n\n\n" + p.stderr


def parse_args():
    parser = argparse.ArgumentParser(description="MCP server")
    parser.add_argument(
        "--python-path",
        type=str,
        default=None,
    )

    args = parser.parse_args()
    print(args)

    return args


if __name__ == "__main__":
    # Start the server
    mcp.run()
