#!/usr/bin/env python3

import argparse
import sys
from tools.utils import get_env
from tools.git_helpers import (
	run_git_command,
	resolve_base_sha_from_ref,
	compute_merge_base_sha,
)
from tools.gemini_wrapper import launch

VERDICT_MARKER = "VERDICT:"


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(
		description=(
			"Preflight launcher for PR quality checks (Gemini). "
			"Prefers SHAs from environment variables, with optional CLI overrides."
		)
	)
	parser.add_argument("--pr-number", help="Pull request number")
	parser.add_argument("--head-sha", help="Head (PR) commit SHA to evaluate")
	parser.add_argument("--base-sha", help="Base commit SHA to compare against")
	parser.add_argument("--head-ref", help="Head branch name")
	parser.add_argument("--base-ref", help="Base branch name")
	return parser.parse_args()


def build_context_from_env_and_args(args: argparse.Namespace) -> argparse.Namespace:
	# Prefer explicit CLI args, then CI-provided envs.
	pr_number = args.pr_number or get_env("PR_NUMBER")

	# Head SHA: prefer PR_HEAD_SHA, fallback to PR_SHA
	head_sha = args.head_sha or get_env("PR_HEAD_SHA") or get_env("PR_SHA")
	if not head_sha:
		raise SystemExit(
			"Missing head SHA. Provide --head-sha or set PR_HEAD_SHA (or PR_SHA)."
		)

	head_ref = args.head_ref or get_env("PR_HEAD_REF") or get_env("GITHUB_HEAD_REF")

	# Base SHA: prefer PR_BASE_SHA, otherwise attempt to resolve from base ref
	base_sha = args.base_sha or get_env("PR_BASE_SHA")
	base_ref = args.base_ref or get_env("PR_BASE_REF") or get_env("GITHUB_BASE_REF")

	if not base_sha and base_ref:
		base_sha = resolve_base_sha_from_ref(base_ref)

	if not base_sha:
		raise SystemExit(
			"Missing base SHA. Provide --base-sha or set PR_BASE_SHA. "
			"Alternatively set PR_BASE_REF (or GITHUB_BASE_REF) so it can be resolved."
		)

	return argparse.Namespace(
		pr_number=pr_number,
		head_sha=head_sha,
		base_sha=base_sha,
		head_ref=head_ref,
		base_ref=base_ref,
	)


def main() -> int:
	args = parse_args()
	context = build_context_from_env_and_args(args)

	# Use the merge-base as the diff baseline
	merge_base_sha = compute_merge_base_sha(context.base_sha, context.head_sha)
	if not merge_base_sha:
		print(
			"Warning: Unable to compute merge-base; proceeding with provided SHAs.",
			file=sys.stderr,
		)

	# Build prompt for Gemini
	merge_base_line = f"\n - merge-base: {merge_base_sha}" if merge_base_sha else ""
	prompt = (
		f"Review this pull request for readiness. Analyze:\n"
		f"- Code quality and style\n"
		f"- Test coverage\n"
		f"- Documentation\n"
		f"- Potential bugs or issues\n"
		f"\n"
		f"---\n"
		f"Commits to compare (sha / branch):\n"
		f" - {context.head_sha} / {context.head_ref} (candidate)\n"
		f" - {context.base_sha} / {context.base_ref} (base).{merge_base_line}\n"
		f"---\n"
		f"\n"
		f"At the start of your analysis, echo the branches and SHAs you compared.\n"
		f"\n"
		f"At the end of your analysis, you MUST include exactly one line in this format:\n"
		f"{VERDICT_MARKER} PASSED\n"
		f"or\n"
		f"{VERDICT_MARKER} FAILED\n"
		f"\n"
		f"The verdict line must be on its own line with no other text."
	)

	# Determine output directory (Gemini-specific)
	default_dir = f"artifacts/gemini_review/{(context.pr_number or 'local')}-{context.head_sha}"
	output_dir = get_env("GEMINI_OUTPUT_DIR") or default_dir

	# Invoke Gemini and propagate exit code
	exit_code = launch(prompt, VERDICT_MARKER, output_dir)

	# Short summary for logs
	print(f"Gemini review artifacts: {output_dir}")
	print(f"Gemini review exit code: {exit_code}")

	return exit_code


if __name__ == "__main__":
	raise SystemExit(main())


