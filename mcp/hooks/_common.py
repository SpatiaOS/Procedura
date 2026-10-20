"""Read-only hook helpers; hooks never write modelling state or run geometry."""
import json
from pathlib import Path
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]


def event():
    return json.loads(sys.stdin.read() or "{}")


def active():
    pointer = ROOT / "outputs/mcp/ACTIVE_V2"
    if not pointer.exists():
        return None, {}
    name = pointer.read_text().strip()
    if not re.fullmatch(r"[a-z][a-z0-9_]*", name):
        raise ValueError("Invalid active run")
    run = ROOT / "outputs/mcp" / name
    return run, json.loads((run / ".cx/state.json").read_text())


def context(name, text):
    print(json.dumps({"hookSpecificOutput": {"hookEventName": name, "additionalContext": text}}))


def deny(reason):
    print(json.dumps({"hookSpecificOutput": {"hookEventName": "PreToolUse", "permissionDecision": "deny", "permissionDecisionReason": reason}}))


def status():
    result = subprocess.run([sys.executable, str(ROOT / "mcp/workflow.py"), "status"],
                            capture_output=True, text=True, timeout=20, cwd=ROOT)
    return result.stdout + result.stderr
