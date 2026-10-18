"""Maintainer-only inventory of the Procedura engine used by MCP runs."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]


def engine_files(root):
    files = []
    for directory in ("src", "prompts", "lib", "vendor/harness"):
        files.extend(p for p in (root / directory).rglob("*") if p.is_file()
                     and not {"__pycache__", "node_modules", ".git"}.intersection(p.parts))
    files.extend((root / "scripts").glob("_*.py"))
    files.extend(root / name for name in ("scripts/procedura.ts", "package.json", "bun.lock", "bunfig.toml", "tsconfig.json"))
    return sorted(set(files))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write-lock", action="store_true", required=True)
    parser.parse_args()
    record = {"schema": 1, "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
              "description": "Procedura engine with two session-transport seams; review changes before updating this lock.",
              "files": {p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in engine_files(ROOT)}}
    (ROOT / "mcp/engine.lock.json").write_text(json.dumps(record, indent=2) + "\n")
    print("Recorded", len(record["files"]), "engine files")
