#!/usr/bin/env python3
"""During an active model run, restrict the outer operator to the fixed launcher."""
from pathlib import Path
import json
import re
import shlex
from _common import ROOT, active, deny, event


def permitted(ev, run):
    full_name = ev.get("tool_name", "")
    name = full_name.split(".")[-1]
    args = ev.get("tool_input") or {}
    if name == "Skill":
        return isinstance(args, dict) and args.get("skill") == "procedura"
    if name in ("ToolSearch", "tool_search"):
        return isinstance(args, dict) and isinstance(args.get("query"), str) and "procedura" in args["query"].lower()
    if re.fullmatch(r"(?:mcp__procedura__|mcp__procedura\.|procedura\.)(next_request|submit_artifact)", full_name):
        return isinstance(args, dict)
    if name in ("image_gen", "image_generation", "generate_image"):
        worker = run / ".cx/worker.json"
        if not worker.exists():
            return False
        attempt = json.loads(worker.read_text()).get("id")
        for path in (run / ".cx/calls").glob("*/ready.json"):
            if (path.parent / "result.json").exists() or (path.parent / "submission.json").exists():
                continue
            req = json.loads((path.parent / "request.json").read_text())
            if req.get("attempt_id") == attempt and req.get("kind") == "reference_image":
                return True
        return False
    if name in ("sleep", "clocksleep"):
        return isinstance(args, dict) and isinstance(args.get("duration_ms"), (int, float)) and 0 < args["duration_ms"] <= 60000
    if name == "exec":
        # Current Codex models may wrap native tools in a JavaScript execution
        # host. Accept ONE literal tool call, not arbitrary JavaScript programs.
        code = args if isinstance(args, str) else args.get("code", args.get("input", ""))
        code = re.sub(r"^\s*// @exec:[^\n]*\n", "", code).strip()
        submit = re.fullmatch(r"text\(await tools\.(mcp__procedura__submit_artifact)\((\{.*\})\)\);?", code, re.S)
        request = re.fullmatch(r'const r = await tools\.(mcp__procedura__next_request)\((\{.*\})\);\s*for \(const c of r.content\) \{ if \(c.type === "image"\) image\(c\); else if \(c.type === "text"\) text\(c.text\); \}', code, re.S)
        if submit or request:
            match = submit or request
            return permitted({**ev, "tool_name": match[1], "tool_input": json.loads(match[2])}, run)
        match = re.fullmatch(r"text\(await tools\.(exec_command|write_stdin|clock__sleep)\((\{.*\})\)\);?", code, re.S)
        if not match:
            match = re.fullmatch(r"const r = await tools\.(exec_command|write_stdin)\((\{.*\})\);\s*text\(r\);?", code, re.S)
        if not match:
            match = re.fullmatch(r"const r = await tools\.(view_image)\((\{.*\})\);\s*image\(r.image_url\);?", code, re.S)
        if not match:
            return False
        try:
            nested = json.loads(match[2])
        except ValueError:
            return False
        return permitted({**ev, "tool_name": "sleep" if match[1] == "clock__sleep" else match[1], "tool_input": nested}, run)
    if name == "write_stdin":
        return isinstance(args, dict) and args.get("chars", "") in ("", "\x03")
    if name == "TaskOutput":
        # Claude Code polls its own background Bash task through this read-only tool.
        return isinstance(args, dict) and isinstance(args.get("task_id"), str)
    if name == "Read":
        path = Path(args.get("file_path", "")).resolve()
        return path in (ROOT / "AGENTS.md", ROOT / "CLAUDE.md", ROOT / ".agents/skills/procedura/SKILL.md") or (
            path.is_relative_to(run.resolve()) and path.suffix.lower() in (".png", ".jpg", ".jpeg"))
    if name == "view_image":
        path = Path(args.get("path", "")).resolve()
        return path.is_relative_to(run.resolve())
    if name not in ("Bash", "bash", "exec_command", "shell_command"):
        return False
    cmd = args if isinstance(args, str) else args.get("cmd", args.get("command", ""))
    if any(x in cmd for x in ("`", "$(", "${", "\n", "\r")):
        return False
    tokens = shlex.split(cmd)
    if any(t in (";", "&&", "||", "|", ">", ">>", "<", "&") for t in tokens):
        return False
    cwd = Path((args.get("workdir") if isinstance(args, dict) else None) or ev.get("cwd") or ROOT)
    if len(tokens) >= 3 and Path(tokens[0]).name in ("python", "python3") and (cwd / tokens[1]).resolve() == ROOT / "mcp/workflow.py":
        # No alternate backends, scripts, stage commands, environment prefixes,
        # shell chains or interpreter flags. Argparse rejects invalid cx options.
        return tokens[2] in ("status", "wait", "ref", "run", "finish", "abort", "doctor", "verify") and not any(
            any(c in t for c in (";", "&", "|", ">", "<")) for t in tokens[3:])
    if len(tokens) >= 2 and tokens[0] == "cat":
        allowed = [run / ".cx/completion.json", run / ".cx/result.json", ROOT / "AGENTS.md", ROOT / ".agents/skills/procedura/SKILL.md"]
        return all((cwd / t).resolve() in allowed for t in tokens[1:])
    return False


def main():
    ev = event()
    try:
        run, st = active()
        if run is None or st.get("phase") in ("done", "done_with_limitations", "aborted"):
            return
        if permitted(ev, run):
            return
        deny("Active Procedura MCP run: answer only the program's procedura.next_request through procedura.submit_artifact, or use the fixed launcher/status/wait/preview tools. Do not edit run files, spawn model calls or choose pipeline stages. Maintenance requires an explicit user maintenance request.")
    except (ValueError, OSError, TypeError, AttributeError) as error:
        deny("Cannot verify the active workflow: " + str(error))


if __name__ == "__main__":
    main()
