#!/usr/bin/env python3
"""
Core Wrapper script for running Gemini CLI in CI.
Accepts a prompt string, expected verdict marker, and output directory path.

Exit codes:
 0 - Success (CLI ran, verdict was PASSED)
 1 - Network/timeout errors, 'gemini' command not found, or unexpected error
 2 - API errors (rate limits, etc.)  [best-effort; CLI may not distinguish]
 3 - Review Failed (CLI ran, verdict was FAILED)
 4 - Parsing Error (CLI ran, but the required verdict marker was missing or ambiguous)
"""

import os
import subprocess
from pathlib import Path
from tools.utils import ensure_dir, write_to_path


def launch(prompt: str, verdict_marker: str, output_dir: Path | str, timeout_seconds: int = 180) -> int:
   """
   Run Gemini CLI with the given prompt, check for verdict, and write outputs.
   Returns an exit code: 0 (success), 3 (review failed), 4 (parsing error), 1/2 (errors).
   """
   OUTPUT_DIR = Path(output_dir)
   VERDICT_MARKER = verdict_marker.strip().upper()
   ensure_dir(OUTPUT_DIR)

   # Prefer explicit model override via env, fallback to a fast public model
#   model = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")

   try:
      # Invoke gemini CLI; pass prompt as a single argument
      result = subprocess.run(
         ["node", "gemini", prompt],
         capture_output=True,
         text=True,
         timeout=timeout_seconds,
         check=False,
      )

      # Combine stdout and stderr for complete output
      full_output = result.stdout
      if result.stderr:
         full_output += f"\n\n--- STDERR ---\n{result.stderr}"

      # Phase 1: Check for CLI Failure (Exit Codes 1 or 2)
      if result.returncode != 0:
         error_text = (result.stderr or result.stdout or "").lower()
         # Try to detect API-ish errors
         if any(phrase in error_text for phrase in ["rate limit", "too many requests", "quota", "api error", "unauthorized", "permission"]):
            print("!!!!!!!!!!!!!!!!!!!!!!")
            print(error_text)
            print("!!!!!!!!!!!!!!!!!!!!!!")
            write_to_path(OUTPUT_DIR, "error.txt", f"API Error (Exit 2):\n{full_output}")
            return 2
         write_to_path(OUTPUT_DIR, "error.txt", f"Error (Exit 1 - Subprocess failed with code {result.returncode}):\n{full_output}")
         return 1

      # Phase 2: CLI Success - Analyze Output (Exit Codes 0, 3, or 4)
      write_to_path(OUTPUT_DIR, "success_raw_output.txt", full_output)

      # Look for the verdict marker
      verdict_line = next(
         (line.strip().upper() for line in full_output.splitlines() if line.strip().upper().startswith(VERDICT_MARKER)),
         None,
      )

      if verdict_line is None:
         write_to_path(OUTPUT_DIR, "error.txt", f"Parsing Error (Exit 4): Verdict marker '{VERDICT_MARKER}' not found in output.")
         return 4

      # Check the verdict
      if "PASSED" in verdict_line:
         write_to_path(OUTPUT_DIR, "review_verdict.txt", "VERDICT: PASSED")
         return 0
      elif "FAILED" in verdict_line:
         write_to_path(OUTPUT_DIR, "review_verdict.txt", "VERDICT: FAILED")
         return 3
      else:
         write_to_path(OUTPUT_DIR, "error.txt", f"Parsing Error (Exit 4): Found verdict line but value is ambiguous: {verdict_line}")
         return 4

   except subprocess.TimeoutExpired:
      write_to_path(OUTPUT_DIR, "error.txt", "Gemini CLI timed out after 180 seconds")
      return 1
   except FileNotFoundError:
      write_to_path(OUTPUT_DIR, "error.txt", "Error: 'gemini' command not found. Is it installed and in PATH?")
      return 1
   except Exception as e:
      write_to_path(OUTPUT_DIR, "error.txt", f"Unexpected error: {type(e).__name__}: {str(e)}")
      return 1


