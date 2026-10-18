#!/usr/bin/env python3
"""Run Procedura with generation supplied by the current coding-agent session."""
from __future__ import annotations

import argparse
import contextlib
from collections import deque
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import sys
import time
import tomllib
import uuid

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "outputs/mcp"
ACTIVE = RUNS / "ACTIVE_V2"
PROFILE = ROOT / "mcp/profile.json"
LOCK = ROOT / "mcp/engine.lock.json"
TOOLCHAIN = ROOT / ".procedura/toolchain.local.json"
def valid_model(model):
    # Metadata for the outer session, never an API route or a model allowlist.
    return isinstance(model, str) and bool(model.strip()) and len(model) <= 200 and not any(ord(c) < 32 for c in model)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path: Path):
    return json.loads(path.read_text())


def write(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(value, indent=2) + "\n")
    tmp.replace(path)


def fail(message: str):
    raise ValueError(message)


def engine_files():
    from inventory import engine_files as inventory
    return inventory(ROOT)


def verify_source() -> dict:
    lock = read(LOCK)
    bad = []
    for rel, record in lock["files"].items():
        p = ROOT / rel
        if not p.is_file() or sha(p) != record:
            bad.append(rel)
    actual = {p.relative_to(ROOT).as_posix() for p in engine_files()}
    bad.extend(sorted(actual - lock["files"].keys()))
    if bad:
        fail("Pinned engine drift: " + ", ".join(bad) + ". Restore the reviewed snapshot; do not improvise a replacement.")
    return {"commit": lock["source_commit"], "files": len(lock["files"])}


def pack_hash() -> str:
    files = [LOCK, PROFILE, ROOT / "bun.lock", ROOT / "package.json", ROOT / "AGENTS.md", ROOT / "CLAUDE.md"]
    files += sorted((ROOT / "mcp").glob("*.py"))
    files += sorted((ROOT / "mcp/hooks").glob("*.py"))
    files += sorted((ROOT / ".agents/skills").glob("*/SKILL.md"))
    files += sorted((ROOT / "mcp/runtime").glob("*.ts"))
    files += [p for rel in (".codex/config.toml", ".codex/hooks.json", ".mcp.json", ".claude/settings.local.json") if (p := ROOT / rel).exists()]
    digest = hashlib.sha256()
    for p in files:
        digest.update(p.relative_to(ROOT).as_posix().encode())
        # Client approvals and unrelated model/server settings can change while
        # answering a request. Freeze the integration, not saved permissions.
        rel = p.relative_to(ROOT).as_posix()
        if rel == ".codex/config.toml":
            config = tomllib.loads(p.read_text())
            value = {"server": config.get("mcp_servers", {}).get("procedura"),
                     "hooks_enabled": config.get("features", {}).get("hooks")}
            digest.update(json.dumps(value, sort_keys=True).encode())
        elif rel in (".codex/hooks.json", ".claude/settings.local.json"):
            digest.update(json.dumps(read(p).get("hooks", {}), sort_keys=True).encode())
        elif rel == ".mcp.json":
            digest.update(json.dumps(read(p).get("mcpServers", {}).get("procedura"), sort_keys=True).encode())
        else:
            digest.update(p.read_bytes())
    return digest.hexdigest()


def save_inputs(run, spec):
    write(run / ".cx/inputs.json", spec)
    (run / ".cx/inputs.sha256").write_text(sha(run / ".cx/inputs.json") + "\n")


def output_hashes(run):
    return {p.relative_to(run).as_posix(): sha(p) for p in sorted(run.rglob("*"))
            if p.is_file() and ".cx" not in p.relative_to(run).parts}


def toolchain() -> dict:
    if not TOOLCHAIN.exists():
        fail("Run python3 mcp/setup.py to pin this machine's toolchain.")
    return read(TOOLCHAIN)


def check_tools(chain: dict):
    for name in ("bun", "python", "openscad", "blender"):
        if not chain.get(name) or not os.access(chain[name], os.X_OK):
            fail(f"Pinned {name} is unavailable. Run python3 mcp/setup.py --check.")
    if not chain.get("artifacts"):
        fail("Toolchain has no executable hashes. Run python3 mcp/setup.py.")
    for name, artifact in chain["artifacts"].items():
        if not Path(artifact["path"]).is_file() or sha(Path(artifact["path"])) != artifact["sha256"]:
            fail(f"Pinned {name} changed. Reinstall explicitly before starting a new run.")


def engine_env(spec: dict) -> dict:
    # Keep operating-system and proxy settings. Clear every upstream workflow knob
    # before applying the reviewed profile; a caller cannot switch in an ablation.
    env = {k: v for k, v in os.environ.items() if not k.startswith(("PROCEDURA_", "CX_", "BUN_"))
           and k not in ("NODE_OPTIONS", "NODE_PATH", "PYTHONPATH", "PYTHONHOME", "OPENSCADPATH")
           and not re.search(r"(?:API_KEY|AUTH_TOKEN)$", k)}
    env.update(spec["profile"]["environment"])
    env.update({"OPENSCAD_PATH": spec["toolchain"]["openscad"],
                "PROCEDURA_BLENDER_PATH": spec["toolchain"]["blender"],
                "PROCEDURA_ISAACSIM_PATH": spec["toolchain"].get("isaac", str(Path.home() / "isaacsim")),
                "PROCEDURA_MCP_SESSION": "1"})
    env["PATH"] = str(Path(spec["toolchain"]["python"]).parent) + os.pathsep + env.get("PATH", os.defpath)
    return env


def resolve_run(name=None) -> Path:
    if name is None:
        if not ACTIVE.exists():
            fail("No MCP run. Start with python3 mcp/workflow.py new <name> --brief '…'.")
        name = ACTIVE.read_text().strip()
    if not re.fullmatch(r"[a-z][a-z0-9_]*", name):
        fail("Run name must be lowercase snake_case; paths are not accepted.")
    run = RUNS / name
    if run.is_symlink() or not (run / ".cx/inputs.json").is_file():
        fail(f"{name} is not an MCP run. Legacy outputs remain readable and are never auto-migrated.")
    return run


def state(run):
    return read(run / ".cx/state.json")


def set_state(run, phase, **fields):
    previous = state(run) if (run / ".cx/state.json").exists() else {}
    if phase in ("running", "ready", "done", "done_with_limitations"):
        previous.pop("error", None)
    write(run / ".cx/state.json", {**previous, "phase": phase, "updated": time.time(), **fields})


def validate_inputs(run):
    verify_source()
    if (run / ".cx/inputs.sha256").read_text().strip() != sha(run / ".cx/inputs.json"):
        fail("Frozen input manifest changed. Do not edit run settings or receipts.")
    spec = read(run / ".cx/inputs.json")
    if spec["pack_sha256"] != pack_hash():
        fail("Pack/profile changed since this run was created. Restore its version; never resume under different settings.")
    if spec["brief_sha256"] != sha(run / "brief.md"):
        fail("Run brief changed; create a new run for a new request.")
    if spec.get("image_sha256") and sha(run / "image.png") != spec["image_sha256"]:
        fail("Reference image changed since it was locked.")
    if spec["profile"] != read(PROFILE) or not valid_model(spec["model"]) or spec["reasoning_effort"] != spec["profile"]["reasoning_effort"]:
        fail("Run profile/model does not match the fixed profile.")
    if spec["text_only"] and (run / "image.png").exists():
        fail("Text-only run acquired a reference image; upstream stages would change mode.")
    expected_prompt = (ROOT / "prompts/image_prompt.md").read_text().replace("{text}", (run / "brief.md").read_text().strip())
    if (run / ".cx/reference_prompt.txt").read_text() != expected_prompt:
        fail("Reference prompt changed.")
    check_tools(spec["toolchain"])
    return spec


@contextlib.contextmanager
def run_lock(run):
    with (run / ".cx/process.lock").open("a+") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            fail("This run already has a worker. Use python3 mcp/workflow.py status; do not start a second worker.")
        try:
            yield lock
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)


def image_check(path: Path):
    # Reference bytes remain untouched. A PNG signature guards accidental path/text input.
    if not path.is_file() or path.read_bytes()[:8] != b"\x89PNG\r\n\x1a\n":
        fail("Reference must be a PNG image. Convert explicitly before setting it.")


def new(a):
    verify_source()
    chain = toolchain()
    check_tools(chain)
    if not re.fullmatch(r"[a-z][a-z0-9_]*", a.name):
        fail("Use a lowercase snake_case name.")
    if ACTIVE.exists():
        current = resolve_run()
        if state(current)["phase"] not in ("done", "done_with_limitations", "aborted"):
            fail(f"Run {current.name} is unfinished. Resume it, or abort if the user abandoned it.")
    run = RUNS / a.name
    if run.exists():
        fail("Run directory already exists; use a new name. Existing outputs are never replaced.")
    brief = Path(a.brief_file).read_text() if a.brief_file else a.brief
    if not brief or not brief.strip():
        fail("A non-empty description is required.")
    if a.ref and a.text_only:
        fail("--ref and --text-only are mutually exclusive.")
    if a.ref:
        image_check(Path(a.ref))
    profile = read(PROFILE)
    model = a.model or profile["model"]
    if not valid_model(model):
        fail("Model metadata must be non-empty text of at most 200 characters, without control characters")
    options = {}
    if a.no_motion:
        options.update(motion=False, motionAware=False, incrementalMotion=False)
    if a.no_paint:
        options["paint"] = False
    if a.no_assembly:
        options["assembly"] = False
    (run / ".cx").mkdir(parents=True)
    (run / "brief.md").write_text(brief)
    spec = {"schema": 2, "name": a.name, "profile": profile, "options": options,
            "model": model, "reasoning_effort": profile["reasoning_effort"],
            "text_only": a.text_only, "brief_sha256": sha(run / "brief.md"),
            "pack_sha256": pack_hash(), "toolchain": chain, "image_sha256": None}
    if a.ref:
        shutil.copyfile(a.ref, run / "image.png")
        spec["image_sha256"] = sha(run / "image.png")
    prompt = (ROOT / "prompts/image_prompt.md").read_text().replace("{text}", brief.strip())
    (run / ".cx/reference_prompt.txt").write_text(prompt)
    save_inputs(run, spec)
    set_state(run, "ready" if a.text_only or a.ref else "reference")
    ACTIVE.parent.mkdir(exist_ok=True)
    ACTIVE.write_text(a.name + "\n")
    status(a)


def execute(run: Path, spec: dict, script: str, lock_fd: int) -> int:
    cmd = [spec["toolchain"]["bun"], "--no-env-file", "run", str(ROOT / script), str(run)]
    write(run / ".cx/worker.json", {"id": str(uuid.uuid4()), "script": script, "pack_sha256": spec["pack_sha256"], "started": time.time()})
    with (run / ".cx/pipeline.log").open("ab") as log:
        child = subprocess.Popen(cmd, cwd=ROOT, env=engine_env(spec), stdout=log, stderr=subprocess.STDOUT,
                                 start_new_session=True, pass_fds=(lock_fd,))
        set_state(run, "running", pid=child.pid, operation=script)
        print(f"Running {run.name}; log: {run / '.cx/pipeline.log'}", flush=True)
        try:
            return child.wait()
        except BaseException:
            os.killpg(child.pid, signal.SIGTERM)
            try:
                child.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGKILL)
                child.wait()
            set_state(run, "interrupted", pid=None)
            raise


def reference(a):
    run = resolve_run()
    with run_lock(run) as lock:
        spec = validate_inputs(run)
        if state(run)["phase"] not in ("reference", "error", "interrupted") or spec.get("image_sha256") or spec["text_only"]:
            fail("Reference is already fixed or this run is text-only.")
        if a.ref_action == "set":
            image_check(Path(a.path))
            shutil.copyfile(a.path, run / "image.png")
        elif a.ref_action == "generate":
            rc = execute(run, spec, "mcp/runtime/reference.ts", lock.fileno())
            if rc:
                set_state(run, "error", pid=None, error="Reference generation failed. Inspect the log, or provide a PNG with python3 mcp/workflow.py ref set.")
                fail("Reference generation failed; no text-only/API fallback. See python3 mcp/workflow.py status --log.")
        image_check(run / "image.png")
        spec["image_sha256"] = sha(run / "image.png")
        save_inputs(run, spec)
        set_state(run, "ready", pid=None, error=None)
    status(a)


def completion(run, spec, result) -> dict:
    required = ["final.scad", "final.obj", "final_summary.txt", "plan.json", "parts_summary.json"]
    missing = [p for p in required if not (run / p).is_file() or not (run / p).stat().st_size]
    if missing:
        fail("Missing pipeline outputs: " + ", ".join(missing))
    parts = read(run / "parts_summary.json")
    limitations = []
    skipped = [p["name"] for p in parts.get("parts", []) if not p.get("generated")]
    if skipped:
        limitations.append("Parts not generated: " + ", ".join(skipped))
    if parts.get("floaterParts", 0):
        limitations.append(f"{parts['floaterParts']} draft parts committed with connectivity warnings")
    refine = result.get("refine", {})
    if not refine.get("ok"):
        limitations.append("Refine did not report successful output")
    if refine.get("verdict") != "ok":
        limitations.append("Refine verdict: " + str(refine.get("verdict", "missing")))
    opts = {**spec["profile"]["options"], **spec["options"]}
    if opts["paint"]:
        paint = result.get("paint", {})
        limitations.extend("Paint: " + str(x) for x in paint.get("degraded", []))
        if not paint.get("ok"):
            limitations.append("Paint failed: " + str(paint.get("error", "no successful receipt")))
        else:
            for name in ("final_materials.json", "final_painted.scad", "final_painted.obj", "final_painted.mtl"):
                if not (run / name).is_file():
                    limitations.append("Paint output missing: " + name)
        if not list((run / "preview_painted").glob("*.png")):
            limitations.append("Painted preview missing")
    if opts["motion"]:
        motion = result.get("motion", {})
        limitations.extend("Motion: " + str(x) for x in motion.get("warnings", []))
        if not motion.get("ok"):
            limitations.append("Motion export failed: " + str(motion.get("errors", [])))
        validation = motion.get("validation", {}) or {}
        if validation.get("ran") is not True or validation.get("ok") is not True:
            limitations.append("Isaac validation did not pass: " + str(validation.get("errors") or validation.get("skippedReason") or "not run"))
        for label in ("usdaPath", "urdfPath", "manifestPath", "planPath"):
            if (label != "urdfPath" or opts.get("motionUrdf")) and (not motion.get(label) or not Path(motion[label]).is_file()):
                limitations.append("Motion output missing: " + label)
    if not list((run / "preview_final").glob("*.png")):
        limitations.append("Final preview missing")
    artifacts = output_hashes(run)
    for name in ("inputs.json", "result.json"):
        if (run / ".cx" / name).exists():
            artifacts[".cx/" + name] = sha(run / ".cx" / name)
    limitations = list(dict.fromkeys(limitations))
    return {"status": "done_with_limitations" if limitations else "done", "limitations": limitations,
            "parts_generated": parts.get("partsGenerated"), "parts_planned": len(parts.get("plan", [])),
            "refine_verdict": refine.get("verdict"), "source_commit": read(LOCK)["source_commit"],
            "pack_sha256": spec["pack_sha256"], "artifacts": artifacts}


def publish_completion(run, receipt):
    write(run / ".cx/completion.json", receipt)
    (run / ".cx/completion.sha256").write_text(sha(run / ".cx/completion.json") + "\n")
    set_state(run, receipt["status"], pid=None, error=None)


def recover_worker_result(run, spec):
    """Adopt a sealed successful worker after its parent died; never rerun stages."""
    success_path = run / ".cx/worker-success.json"
    worker_path = run / ".cx/worker.json"
    if not success_path.exists() or not worker_path.exists():
        return False
    success, worker = read(success_path), read(worker_path)
    if (not worker.get("id") or success.get("worker_id") != worker["id"]
            or success.get("pack_sha256") != spec["pack_sha256"]):
        return False  # stale proof from a different attempt
    if (success.get("result_sha256") != sha(run / ".cx/result.json")
            or success.get("artifacts") != output_hashes(run)):
        fail("Completed worker evidence changed before recovery.")
    publish_completion(run, completion(run, spec, read(run / ".cx/result.json")))
    return True


def completion_summary(run, receipt, *, detailed=False):
    """Keep the immutable inventory on disk; operator context stays bounded."""
    limitations = list(dict.fromkeys(receipt.get("limitations", [])))
    result = read(run / ".cx/result.json") if (run / ".cx/result.json").exists() else {}
    refine = result.get("refine") or {}
    paint = result.get("paint") or {}
    motion = result.get("motion") or {}
    validation = motion.get("validation") or {}
    recovered = []
    if validation.get("ran") is True and validation.get("ok") is True:
        # The reference retains warnings from its first Isaac attempt even if
        # sim-feedback repair succeeds. Preserve them, explicitly as history.
        recovered = [x for x in limitations if x.startswith("Motion: isaac validation:")]
        limitations = [x for x in limitations if x not in recovered]
    previews = [p.relative_to(run).as_posix() for directory in ("preview_painted", "preview_final")
                for p in sorted((run / directory).glob("*.png"))]
    # Upstream writes parts_summary only at draft end. A crash can omit earlier
    # warnings from that final summary, so retain the original log evidence too.
    historical_parts = []
    current_part = None
    log_path = run / ".cx/pipeline.log"
    if log_path.exists():
        with log_path.open(errors="replace") as log:
            for line in log:
                match = re.search(r"\[inc-draft\] part \d+/\d+: ([a-z][a-z0-9_]*)", line)
                if match:
                    current_part = match[1]
                if current_part and "committed WITH WARNING" in line and current_part not in historical_parts:
                    historical_parts.append(current_part)
    summary = {k: receipt.get(k) for k in ("status", "parts_generated", "parts_planned", "refine_verdict", "pack_sha256")}
    summary.update({"artifact_count": len(receipt.get("artifacts", {})), "receipt": str(run / ".cx/completion.json"),
                    "previews": previews, "limitation_count": len(limitations),
                    "limitations": limitations if detailed else limitations[:5],
                    "more_limitations": 0 if detailed else max(0, len(limitations) - 5),
                    "recovered_attempt_warnings": recovered,
                    "historical_draft_warning_parts": historical_parts,
                    "refine_summary": refine.get("summary"), "paint_ok": paint.get("ok"),
                    "motion_export_ok": motion.get("ok"),
                    "isaac": {k: validation.get(k) for k in ("ran", "ok", "strictOk", "assetRuleErrors", "actuationOk", "contactsOk", "mobilityOk", "reportPath")}})
    return summary


def run_pipeline(a):
    run = resolve_run()
    with run_lock(run) as lock:
        phase = state(run)["phase"]
        if phase in ("done", "done_with_limitations"):
            print("Run already complete. Next: python3 mcp/workflow.py finish")
            return
        if phase == "aborted":
            fail("Run was aborted; create a new run.")
        spec = validate_inputs(run)
        if recover_worker_result(run, spec):
            print("Recovered the completed worker without rerunning stages. Next: python3 mcp/workflow.py finish")
            return
        if not spec["text_only"] and not spec.get("image_sha256"):
            fail("Set the reference first: python3 mcp/workflow.py ref generate or python3 mcp/workflow.py ref set <png>.")
        checkpoint = run / ".cx/checkpoint.json"
        if checkpoint.exists():
            previous = read(checkpoint)
            current = output_hashes(run)
            if previous != current:
                changed = sorted(k for k in previous.keys() | current.keys() if previous.get(k) != current.get(k))
                fail("Pipeline checkpoint changed outside the worker: " + ", ".join(changed))
        elif not (run / ".cx/worker.json").exists() and any((run / p).exists() for p in ("plan.json", "draft.scad", "final.scad")):
            fail("Unowned pipeline outputs exist before the first run.")
        try:
            rc = execute(run, spec, "mcp/runtime/run.ts", lock.fileno())
        finally:
            write(checkpoint, output_hashes(run))
        if rc:
            set_state(run, "error", pid=None, error=f"Pinned pipeline exited {rc}. Review python3 mcp/workflow.py status --log, then resume with python3 mcp/workflow.py run.")
            fail(f"Pipeline exited {rc}; no workflow substitutions are permitted.")
        try:
            result = read(run / ".cx/result.json")
            receipt = completion(run, spec, result)
        except (ValueError, OSError) as error:
            set_state(run, "error", pid=None, error=str(error))
            raise
        publish_completion(run, receipt)
    status(a)


def is_running(run):
    with (run / ".cx/process.lock").open("a+") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return True
        fcntl.flock(lock, fcntl.LOCK_UN)
        return False


def status(a):
    if not ACTIVE.exists():
        print("No MCP run. Next: python3 mcp/workflow.py new <name> --brief '…' [--ref image.png]")
        old = RUNS / "ACTIVE"
        if old.exists():
            print(f"Legacy run preserved: {old.read_text().strip()} (not resumed by MCP)")
        return
    run = resolve_run()
    st = state(run)
    phase = st["phase"]
    alive = is_running(run)
    if alive:
        spec = read(run / ".cx/inputs.json")
        next_action = ("Call procedura.next_request; generate its requested artifact in this session, then procedura.submit_artifact"
                       if spec["profile"].get("transport") == "session"
                       else "Wait for the worker; inspect python3 mcp/workflow.py status --log")
    elif phase in ("done", "done_with_limitations"):
        next_action = "python3 mcp/workflow.py finish; deliver files and every completion limitation"
    elif phase == "aborted":
        next_action = "Start a new run only when requested"
    else:
        spec = read(run / ".cx/inputs.json")
        next_action = ("python3 mcp/workflow.py run" if spec["text_only"] or spec.get("image_sha256")
                       else "python3 mcp/workflow.py ref generate (or ref set <png>)")
    info = {"run": str(run), "phase": phase, "worker_running": alive, "next": next_action}
    if alive and spec["profile"].get("transport") == "session":
        from bridge import pending
        packets = pending(run)
        info["session_request"] = ({"request_id": packets[0][1]["id"], "kind": packets[0][1]["kind"]} if packets else None)
        if packets:
            info["phase"] = "awaiting_artifact"
    if phase == "running" and not alive:
        info["phase"] = "interrupted"
        info["error"] = "Worker exited without a final receipt; resume the same run"
    if st.get("error") and phase == "error" and not alive:
        info["error"] = st["error"]
    if (run / ".cx/completion.json").exists():
        info["completion"] = completion_summary(run, read(run / ".cx/completion.json"))
    print(json.dumps(info, indent=2))
    if getattr(a, "log", False) and (run / ".cx/pipeline.log").exists():
        with (run / ".cx/pipeline.log").open(errors="replace") as log:
            print("".join(deque(log, maxlen=40)).rstrip())


def wait_worker(a):
    """Wait for this run's lock to release, without starting another worker."""
    run = resolve_run()
    deadline = time.monotonic() + a.seconds
    while is_running(run) and time.monotonic() < deadline:
        time.sleep(min(1, max(0, deadline - time.monotonic())))
    status(a)


def abort(a):
    run = resolve_run()
    with run_lock(run):
        if state(run)["phase"] in ("done", "done_with_limitations"):
            fail("Completed run is already closed.")
        set_state(run, "aborted", pid=None, reason=a.reason)
    status(a)


def finish(a):
    run = resolve_run()
    if state(run)["phase"] not in ("done", "done_with_limitations"):
        fail("Finish is a verification, not a stage skip. Follow python3 mcp/workflow.py status until python3 mcp/workflow.py run completes.")
    with run_lock(run):
        receipt_path = run / ".cx/completion.json"
        receipt = read(receipt_path)
        seal = run / ".cx/completion.sha256"
        if not seal.exists():
            # Older receipts may acquire a seal only after their original pack
            # still validates and recomputes the identical completion receipt.
            spec = validate_inputs(run)
            current = completion(run, spec, read(run / ".cx/result.json"))
            if current != receipt:
                fail("Unsealed completion differs; verify it with its original pack.")
            seal.write_text(sha(receipt_path) + "\n")
        if seal.read_text().strip() != sha(receipt_path):
            fail("Completion receipt changed after it was sealed.")
        artifacts = receipt.get("artifacts", {})
        if not {".cx/inputs.json", ".cx/result.json"}.issubset(artifacts):
            fail("Receipt lacks the frozen input/result hashes.")
        expected_outputs = {k: v for k, v in artifacts.items() if not k.startswith(".cx/")}
        if output_hashes(run) != expected_outputs:
            fail("Published artifacts changed after completion.")
        for name in ("inputs.json", "result.json"):
            if sha(run / ".cx" / name) != artifacts[".cx/" + name]:
                fail("Published evidence changed after completion: " + name)
        spec = read(run / ".cx/inputs.json")
        if (spec["pack_sha256"] != receipt.get("pack_sha256")
                or spec["brief_sha256"] != sha(run / "brief.md")
                or (spec.get("image_sha256") and spec["image_sha256"] != sha(run / "image.png"))):
            fail("Receipt does not match the frozen run inputs.")
        if state(run)["phase"] != receipt["status"]:
            fail("Run phase does not match the published completion.")
    summary = completion_summary(run, receipt, detailed=True)
    summary["verified"] = True
    summary["current_pack_differs"] = pack_hash() != receipt["pack_sha256"]
    print(json.dumps(summary, indent=2))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="command", required=True)
    n = sub.add_parser("new", help="Freeze a new request and the fixed profile")
    n.add_argument("name")
    brief = n.add_mutually_exclusive_group(required=True)
    brief.add_argument("--brief")
    brief.add_argument("--brief-file")
    n.add_argument("--ref")
    n.add_argument("--model", help="Record the outer session's model; does not change or launch a model")
    for flag in ("text-only", "no-motion", "no-paint", "no-assembly"):
        n.add_argument("--" + flag, action="store_true")
    n.set_defaults(func=new)
    s = sub.add_parser("status"); s.add_argument("--log", action="store_true"); s.set_defaults(func=status)
    w = sub.add_parser("wait", help="Wait on the existing worker, then print concise status")
    w.add_argument("--seconds", type=int, choices=range(1, 61), metavar="1..60", default=30)
    w.add_argument("--log", action="store_true"); w.set_defaults(func=wait_worker)
    r = sub.add_parser("ref").add_subparsers(dest="ref_action", required=True)
    r.add_parser("generate").set_defaults(func=reference)
    rset = r.add_parser("set"); rset.add_argument("path"); rset.set_defaults(func=reference)
    sub.add_parser("run", help="Run or resume the pinned pipeline").set_defaults(func=run_pipeline)
    sub.add_parser("finish", help="Verify the published completion receipt").set_defaults(func=finish)
    ab = sub.add_parser("abort"); ab.add_argument("--reason", required=True); ab.set_defaults(func=abort)
    v = sub.add_parser("verify", help="Verify the reviewed Procedura engine bytes")
    v.set_defaults(func=lambda a: print(json.dumps(verify_source(), indent=2)))
    sub.add_parser("doctor").set_defaults(func=lambda _: subprocess.run([toolchain()["python"], str(ROOT / "mcp/setup.py"), "--check"], check=True))
    args = ap.parse_args(argv)
    try:
        if args.command == "new":
            RUNS.mkdir(parents=True, exist_ok=True)
            with (RUNS / ".create.lock").open("a+") as lock:
                try:
                    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError:
                    fail("Another launcher is creating a run; wait and inspect python3 mcp/workflow.py status.")
                args.func(args)
        else:
            args.func(args)
    except (ValueError, FileNotFoundError, subprocess.CalledProcessError) as error:
        print(f"procedura: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
