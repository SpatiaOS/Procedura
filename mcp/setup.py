#!/usr/bin/env python3
"""Configure project-local Codex/Claude MCP, hooks and the verified native toolchain."""
import argparse
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import sys
import tomllib

import workflow

ROOT = Path(__file__).resolve().parents[1]
BEGIN = "# BEGIN Procedura MCP"
END = "# END Procedura MCP"


def read_json(path):
    value = json.loads(path.read_text()) if path.exists() else {}
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return value


def hooks():
    events = {"SessionStart": "session_start", "PreToolUse": "pre_tool_use", "Stop": "stop"}
    return {event: [{**({"matcher": "*"} if event == "PreToolUse" else {}), "hooks": [{
        "type": "command", "command": shlex.join([sys.executable, str(ROOT / "mcp/hooks" / (script + ".py"))]),
        "timeout": 30}]}] for event, script in events.items()}


def merge_hooks(existing):
    merged = dict(existing)
    for event, entries in hooks().items():
        retained = []
        for entry in merged.get(event, []):
            others = [h for h in entry.get("hooks", []) if "/mcp/hooks/" not in h.get("command", "")]
            if others:
                retained.append({**entry, "hooks": others})
        merged[event] = retained + entries
    return merged


def codex_config(content):
    # Preserve existing TOML, including model, provider, credentials and comments.
    clean = re.sub(re.escape(BEGIN) + r"\n.*?" + re.escape(END) + r"\n?", "", content, flags=re.S).rstrip()
    config = tomllib.loads(clean)
    if "procedura" in config.get("mcp_servers", {}):
        raise ValueError("An unmanaged mcp_servers.procedura already exists; rename/remove that entry before setup")
    feature = config.get("features", {})
    if feature.get("hooks") is False:
        raise ValueError("Codex features.hooks is explicitly false; enable hooks in .codex/config.toml before setup")
    if "hooks" not in feature:
        header = re.search(r"(?m)^\[features\][ \t]*(?:#[^\n]*)?$", clean)
        if header:
            clean = clean[:header.end()] + "\nhooks = true" + clean[header.end():]
        else:
            clean += "\n\n[features]\nhooks = true"
    if "model_auto_compact_token_limit" not in config:
        clean = "# Large motion requests need room to fit before compaction.\nmodel_auto_compact_token_limit = 180000\n\n" + clean.lstrip()
    block = (f"{BEGIN}\n[mcp_servers.procedura]\ncommand = {json.dumps(sys.executable)}\n"
             f"args = {json.dumps([str(ROOT / 'mcp/server.py'), '--workspace', str(ROOT)])}\n"
             'startup_timeout_sec = 15\ntool_timeout_sec = 45\nenabled_tools = ["next_request", "submit_artifact"]\n'
             f"{END}\n")
    result = clean.rstrip() + "\n\n" + block
    tomllib.loads(result)
    return result


def configurations(agent):
    files = {}
    if agent in ("codex", "both"):
        path = ROOT / ".codex/config.toml"
        files[path] = codex_config(path.read_text() if path.exists() else "")
        path = ROOT / ".codex/hooks.json"
        value = read_json(path)
        value["hooks"] = merge_hooks(value.get("hooks", {}))
        files[path] = json.dumps(value, indent=2) + "\n"
    if agent in ("claude", "both"):
        path = ROOT / ".mcp.json"
        value = read_json(path)
        server = {"type": "stdio", "command": sys.executable,
                  "args": [str(ROOT / "mcp/server.py"), "--workspace", str(ROOT)]}
        servers = value.setdefault("mcpServers", {})
        old = servers.get("procedura")
        if old and old != server and not any(str(a).endswith("/mcp/server.py") for a in old.get("args", [])):
            raise ValueError("An unrelated .mcp.json server named procedura already exists; rename it before setup")
        servers["procedura"] = server
        files[path] = json.dumps(value, indent=2) + "\n"
        path = ROOT / ".claude/settings.local.json"
        value = read_json(path)
        value["hooks"] = merge_hooks(value.get("hooks", {}))
        files[path] = json.dumps(value, indent=2) + "\n"
        source = ROOT / ".agents/skills/procedura/SKILL.md"
        files[ROOT / ".claude/skills/procedura/SKILL.md"] = source.read_text()
    return files


def executable(*candidates):
    for candidate in candidates:
        if candidate:
            path = Path(candidate).expanduser()
            if path.is_file() and os.access(path, os.X_OK):
                return str(path.resolve())
    raise ValueError("Executable not found; checked " + repr(candidates))


def configured_path(key):
    # Read only binary paths from the API CLI's .env; never execute/source it.
    path = ROOT / ".env"
    value = None
    if path.exists():
        for line in path.read_text().splitlines():
            match = re.fullmatch(r"\s*" + re.escape(key) + r"\s*=\s*(.*?)\s*", line)
            if match:
                value = match[1].strip("\"'")
    return value


def output(argv):
    p = subprocess.run(argv, capture_output=True, text=True, timeout=30)
    if p.returncode:
        raise ValueError(f"{Path(argv[0]).name} failed: {p.stderr[-1000:]}")
    return (p.stdout + p.stderr).strip()


def discover(args):
    from PIL import Image
    home = Path.home()
    prefix = Path(os.environ.get("PREFIX", str(home / "opt")))
    previous = workflow.toolchain() if workflow.TOOLCHAIN.exists() else {}
    chain = {"bun": executable(shutil.which("bun"), home / ".bun/bin/bun"),
             # Preserve a venv path: resolving its symlink would lose Pillow.
             "python": str(Path(sys.executable).absolute()),
             "openscad": executable(args.openscad, os.environ.get("OPENSCAD_PATH"), configured_path("OPENSCAD_PATH"),
                                    previous.get("openscad"), prefix / "openscad", shutil.which("openscad")),
             "blender": executable(args.blender, os.environ.get("PROCEDURA_BLENDER_PATH"), configured_path("PROCEDURA_BLENDER_PATH"),
                                   previous.get("blender"), prefix / "blender/blender", shutil.which("blender"))}
    if "--backend" not in output([chain["openscad"], "--help"]):
        raise ValueError("OpenSCAD needs the Manifold backend; run scripts/install-deps.sh or pass --openscad")
    isaac = args.isaac or os.environ.get("PROCEDURA_ISAACSIM_PATH") or configured_path("PROCEDURA_ISAACSIM_PATH") or previous.get("isaac") or str(home / "isaacsim")
    chain["isaac"] = str(Path(isaac).expanduser().resolve())
    chain["versions"] = {name: output([chain[name], "--version"]).splitlines()[0] for name in ("bun", "python", "openscad", "blender")}
    chain["versions"]["pillow"] = Image.__version__
    paths = {name: chain[name] for name in ("bun", "python", "openscad", "blender")}
    # Common AppImage extraction wrapper: record its actual engine as well.
    engine = Path(chain["openscad"]).parent / "squashfs-root/usr/bin/openscad"
    if engine.is_file():
        paths["openscad_engine"] = str(engine)
    launcher = Path(chain["isaac"]) / "python.sh"
    if launcher.is_file():
        paths["isaac_launcher"] = str(launcher)
    chain["artifacts"] = {name: {"path": str(path), "sha256": workflow.sha(Path(path))} for name, path in paths.items()}
    return chain


def configure(agent, chain, check=False):
    # Calculate and validate all merges before the first write.
    files = configurations(agent)
    workflow.check_tools(chain)
    if check:
        stale = [str(p.relative_to(ROOT)) for p, text in files.items() if not p.exists() or p.read_text() != text]
        if stale:
            raise ValueError("Missing/stale MCP setup: " + ", ".join(stale))
        return
    if workflow.ACTIVE.exists():
        run = workflow.resolve_run()
        if workflow.state(run)["phase"] not in ("done", "done_with_limitations", "aborted"):
            raise ValueError("Finish or explicitly abandon the active MCP run before changing setup")
    for path, text in files.items():
        if not path.exists() or path.read_text() != text:
            path.parent.mkdir(parents=True, exist_ok=True)
            temporary = path.with_name(path.name + ".procedura-tmp")
            temporary.write_text(text)
            temporary.replace(path)
    workflow.write(workflow.TOOLCHAIN, {**chain, "agent": agent})


def check_packages():
    installed = ROOT / "node_modules/@harness/template/src/llm/route.ts"
    if not installed.is_file():
        raise ValueError("Bun dependencies are missing; run bun install --frozen-lockfile before setup")
    if workflow.sha(installed) != workflow.sha(ROOT / "vendor/harness/src/llm/route.ts"):
        raise ValueError("Installed harness is stale; run bun install --force --frozen-lockfile before setup")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--agent", choices=("codex", "claude", "both"))
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--openscad")
    parser.add_argument("--blender")
    parser.add_argument("--isaac")
    args = parser.parse_args()
    if sys.version_info < (3, 11):
        raise ValueError("Python 3.11+ is required")
    source = workflow.verify_source()
    check_packages()
    chain = workflow.toolchain() if args.check else discover(args)
    agent = args.agent or chain.get("agent", "both")
    configure(agent, chain, args.check)
    print(json.dumps({"agent": agent, "checked": args.check, "engine": source,
                      "toolchain": chain["versions"], "isaac_available": (Path(chain["isaac"]) / "python.sh").is_file()}, indent=2))
    print("Restart your coding agent in " + str(ROOT))
    print("Review project trust, the procedura MCP server and hooks if prompted. Your model and credentials are unchanged.")
    print('Then ask: Use Procedura to build <description> from <reference.png>, and run the full pipeline.')


if __name__ == "__main__":
    try:
        main()
    except (ValueError, OSError, ImportError, subprocess.SubprocessError) as error:
        sys.exit("procedura setup: " + str(error))
