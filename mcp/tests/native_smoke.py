#!/usr/bin/env python3
"""Optional native integration fixture: real MCP, pipeline, OpenSCAD and Blender; no model/API."""
import argparse
import json
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[2]
PLAN = [{"name": "block", "level": "L0", "description": "One solid rectangular block, 80 mm X by 60 mm Y by 20 mm Z, centered in X/Y with its bottom on z=0."}]
SCAD = "```openscad\n// PARAMS\n// HELPERS\n// PART\nmodule block() { translate([-40,-30,0]) cube([80,60,20]); }\n// PLACE\nblock();\n```"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--openscad", required=True)
    parser.add_argument("--blender", required=True)
    args = parser.parse_args()
    root = Path(tempfile.mkdtemp(prefix="procedura-native-smoke-")) / "project with spaces"
    root.mkdir()
    for name in ("mcp", "src", "prompts", "lib", "vendor", "scripts", "node_modules", ".agents"):
        shutil.copytree(ROOT / name, root / name, symlinks=True,
                        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    for name in ("AGENTS.md", "CLAUDE.md", "README.md", "package.json", "bun.lock", "bunfig.toml", "tsconfig.json"):
        shutil.copy2(ROOT / name, root / name)
    print("Smoke checkout:", root, flush=True)
    subprocess.run([sys.executable, "mcp/setup.py", "--agent", "both", "--openscad", args.openscad, "--blender", args.blender], cwd=root, check=True)
    # Simulate a downloaded installer read from stdin, without downloads.
    installer = (ROOT / "scripts/install-mcp.sh").read_text()
    for _ in range(2):
        subprocess.run(["bash", "-s", "--", "--dir", str(root), "--agent", "both", "--no-deps"],
                       cwd=root.parent, input=installer, text=True, check=True)
    subprocess.run(["bash", "scripts/install-mcp.sh", "--agent", "both", "--check"], cwd=root, check=True)
    subprocess.run([sys.executable, "mcp/workflow.py", "new", "smoke_block", "--brief", PLAN[0]["description"],
                    "--text-only", "--no-motion", "--no-paint", "--no-assembly"], cwd=root, check=True)
    run = root / "outputs/mcp/smoke_block"
    # A normal client may save an approval after new() freezes the run. That
    # must not invalidate the pipeline; edits to MCP/hooks remain frozen.
    settings = root / ".claude/settings.local.json"
    approved = json.loads(settings.read_text())
    approved["permissions"] = {"allow": ["Bash(python3 mcp/workflow.py run)"]}
    settings.write_text(json.dumps(approved, indent=2) + "\n")
    server = subprocess.Popen([sys.executable, "mcp/server.py"], cwd=root, stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
    sequence = 0

    def rpc(method, params):
        nonlocal sequence
        sequence += 1
        server.stdin.write(json.dumps({"jsonrpc": "2.0", "id": sequence, "method": method, "params": params}) + "\n")
        server.stdin.flush()
        response = json.loads(server.stdout.readline())
        if "error" in response:
            raise RuntimeError(response["error"])
        result = response["result"]
        if result.get("isError"):
            raise RuntimeError(result)
        return result

    rpc("initialize", {"protocolVersion": "2024-11-05", "capabilities": {}, "clientInfo": {"name": "native-fixture", "version": "1"}})
    log = (root / "worker-operator.log").open("w")
    worker = subprocess.Popen([sys.executable, "mcp/workflow.py", "run"], cwd=root, stdout=log, stderr=subprocess.STDOUT)
    deadline = time.time() + 600
    answered = set()
    image_count = 0
    try:
        while worker.poll() is None and time.time() < deadline:
            if not (run / ".cx/worker.json").exists():
                time.sleep(0.1)
                continue
            result = rpc("tools/call", {"name": "next_request", "arguments": {"wait_seconds": 1}})
            metadata = json.loads(result["content"][0]["text"])
            if metadata.get("status") != "generation_requested":
                time.sleep(0.1)
                continue
            request_id = metadata["request_id"]
            if request_id in answered:
                raise RuntimeError("Fixture saw a duplicate pending response")
            image_count += sum(p["type"] == "image" for p in result["content"])
            while metadata["next_cursor"] is not None:
                time.sleep(3.1)
                result = rpc("tools/call", {"name": "next_request", "arguments": {"request_id": request_id, "cursor": metadata["next_cursor"], "wait_seconds": 0}})
                metadata = json.loads(result["content"][0]["text"])
                image_count += sum(p["type"] == "image" for p in result["content"])
            packet = json.loads((run / ".cx/calls" / request_id / "request.json").read_text())
            system = "\n".join(p["text"] for p in packet["system"])
            if "build-plan reviewer" in system:
                answer = json.dumps({"ok": True, "notes": "Fixture is one rectangular block; plan is complete.", "plan": PLAN})
            elif "// PARAMS" in system and "// PART" in system:
                answer = SCAD
            elif "structured diagnosis" in system:
                answer = "SUMMARY: The fixture matches the specified rectangular block.\n\nISSUES: (none — current build matches the text within tolerance)"
            elif "build plan" in system or "build-plan" in system:
                answer = json.dumps(PLAN)
            else:
                raise RuntimeError("Unexpected fixture request: " + system[:300])
            rpc("tools/call", {"name": "submit_artifact", "arguments": {"request_id": request_id, "artifact_text": answer}})
            answered.add(request_id)
            print("Fixture submitted", request_id, flush=True)
        if worker.poll() is None:
            raise RuntimeError("Native fixture exceeded 600 seconds")
        if worker.returncode:
            raise RuntimeError((run / ".cx/pipeline.log").read_text()[-6000:] + (root / "worker-operator.log").read_text()[-2000:])
        finished = subprocess.run([sys.executable, "mcp/workflow.py", "finish"], cwd=root, capture_output=True, text=True, check=True)
        summary = json.loads(finished.stdout)
        assert summary["parts_generated"] == summary["parts_planned"] == 1, summary
        assert summary["refine_verdict"] == "ok", summary
        assert summary["verified"] is True, summary
        assert len(answered) >= 4 and image_count >= 7, (len(answered), image_count)
        evidence = {"checkout": str(root), "requests": len(answered), "images": image_count, "completion": summary}
        (root / "smoke-report.json").write_text(json.dumps(evidence, indent=2) + "\n")
        print(json.dumps(evidence, indent=2), flush=True)
    finally:
        if worker.poll() is None:
            # Let the Python launcher stop its separate Bun process group too.
            worker.send_signal(signal.SIGINT)
            worker.wait(timeout=15)
        server.terminate()
        server.wait(timeout=10)
        log.close()


if __name__ == "__main__":
    main()
