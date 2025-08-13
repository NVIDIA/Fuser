import json
from pathlib import Path
from typing import Any, Dict
from openai import OpenAI

REPO_ROOT = Path("/opt/pytorch/nvfuser")  # adjust if needed
RULES_PATH = REPO_ROOT / "rules.json"

def load_text(path: Path) -> str:
	return path.read_text(encoding="utf-8")

def load_constitution(entry: Dict[str, Any]) -> str:
	c = entry.get("constitution", {})
	source = c.get("source")
	if source == "inline":
		return c.get("content", "")
	if source == "file":
		return load_text((REPO_ROOT / c["path"]).resolve())
	if source == "url":
		# Lazy import to avoid dependency if unused
		import requests
		resp = requests.get(c["url"], timeout=30)
		resp.raise_for_status()
		return resp.text
	return ""

def build_system_prompt(instructions_md: str, constitution_md: str) -> str:
	parts = []
	if constitution_md.strip():
		parts.append("Constitution:\n" + constitution_md.strip())
	if instructions_md.strip():
		parts.append("Bot Instructions:\n" + instructions_md.strip())
	return "\n\n---\n\n".join(parts)

def main():
	config = json.loads(RULES_PATH.read_text(encoding="utf-8"))
	bot = next(b for b in config["bots"] if b["id"] == "documenter")

	instructions_md = load_text((REPO_ROOT / bot["instructions_file"]).resolve())
	constitution_md = load_constitution(bot)

	system_prompt = build_system_prompt(instructions_md, constitution_md)
	model = bot.get("model", "gpt-4o-mini")
	defaults = bot.get("defaults", {})
	temperature = defaults.get("temperature", 0.2)
	max_tokens = defaults.get("max_tokens", 2000)

	client = OpenAI()
	user_prompt = "Create a tight outline for documenting the nvFuser documentation workflow. Keep it to 6 bullets."
	resp = client.chat.completions.create(
		model=model,
		messages=[
			{"role": "system", "content": system_prompt},
			{"role": "user", "content": user_prompt},
		],
		temperature=temperature,
		max_tokens=max_tokens,
	)

	print(resp.choices[0].message.content)

if __name__ == "__main__":
	main()
