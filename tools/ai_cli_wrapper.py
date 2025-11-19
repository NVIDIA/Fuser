#!/usr/bin/env python3
"""
Core Wrapper script for running Gemini or Claude CLI in CI.
Accepts a prompt string, expected verdict marker, and output directory path.

Exit codes:
 0 - Success (CLI ran, verdict was PASSED)
 1 - Network/timeout errors, 'gemini' command not found, or unexpected error
 2 - API errors (rate limits, etc.)  [best-effort; CLI may not distinguish]
 3 - Review Failed (CLI ran, verdict was FAILED)
 4 - Parsing Error (CLI ran, but the required verdict marker was missing or ambiguous)
"""

import subprocess
from pathlib import Path
from tools.utils import ensure_dir, write_to_path


def launch_ai_cli(
   prompt: str, 
   tool: None,
   tool_args : [],
   verdict_marker: str, 
   output_dir: Path | str, 
   timeout_seconds: int = 180
   ) -> int:


   """
   Run Gemini / Claude CLI with the given prompt, check for verdict, and write outputs.
   Returns an exit code: 0 (success), 3 (review failed), 4 (parsing error), 1/2 (errors).
   """
   OUTPUT_DIR = Path(output_dir)
   VERDICT_MARKER = verdict_marker.strip().upper()
   ensure_dir(OUTPUT_DIR)

   if tool == None:
      write_to_path(OUTPUT_DIR, "error.txt", f"Error (Exit 1 - no tool specified")
      return 1

   try:
      # Invoke CLI; pass prompt as a single argument
      safety_instructions = (
          "CRITICAL RULE: If a tool execution fails, "
          "OR if I have exceeded my API quote, DO NOT retry. "
          "Stop immediately and report the error."
      )

      prompt = f"{safety_instructions}\n\n{prompt}"
      launch_array = [tool]
      launch_array += tool_args
      launch_array.append(prompt)

      result = subprocess.run(
         launch_array, 
         capture_output=True,
         text=True,
         timeout=timeout_seconds,
         check=False,
      )

      # Combine stdout and stderr for complete output
      full_output = result.stdout
      if result.stderr:
         full_output += f"\n\n--- STDERR ---\n{result.stderr}"

      # --- Phase 1: Check for Tool/API Failure (Exit Codes 1 or 2) ---
      if result.returncode != 0:
         error_text = (result.stderr or result.stdout or "").lower()
         # Try to detect API-ish errors
         if any(phrase in error_text for phrase in ["rate limit", "too many requests", "quota", "api error", "unauthorized", "permission"]):
            write_to_path(OUTPUT_DIR, "error.txt", f"API Error (Exit 2):\n{full_output}")
            return 2

         # All other subprocess errors (Exit 1)
         write_to_path(OUTPUT_DIR, "error.txt", f"Error (Exit 1 - Subprocess failed with code {result.returncode}):\n{full_output}")
         return 1

      # --- Phase 2: Tool Success - Analyze Output (Exit Codes 0, 3, or 4) ---
      write_to_path(OUTPUT_DIR, "success_raw_output.txt", full_output)

      # Look for the verdict marker
      verdict_line = next(
         (line.strip().upper() for line in full_output.splitlines() if line.strip().upper().startswith(VERDICT_MARKER)),
         None
      )

      if verdict_line is None:
         # Verdict marker not found
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
         # Marker found, but value is ambiguous
         write_to_path(OUTPUT_DIR, "error.txt", f"Parsing Error (Exit 4): Found verdict line but value is ambiguous: {verdict_line}")
         return 4

   except subprocess.TimeoutExpired:
      error_msg = "{tool} command timed out after {timeout_seconds} seconds"
      write_to_path(OUTPUT_DIR, "error.txt", error_msg)
      return 1

   except FileNotFoundError:
      error_msg = "Error: '{tool}' command not found. Is it installed and in PATH?"
      write_to_path(OUTPUT_DIR, "error.txt", error_msg)
      return 1

   except Exception as e:
      error_msg = f"Unexpected error: {type(e).__name__}: {str(e)}"
      write_to_path(OUTPUT_DIR, "error.txt", error_msg)
      return 1
