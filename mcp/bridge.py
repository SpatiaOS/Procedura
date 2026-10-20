"""Durable in-session request/submit bridge. No model subprocess or API client."""
from __future__ import annotations
import base64
import contextlib
import fcntl
import hashlib
import io
import json
from pathlib import Path
import re
import time
import workflow as cx

# Codex stores each tool output under a fixed token budget and silently drops the
# remainder ("[omitted N text items ...]"). The 2026-09-21 assault_buggy_sol run
# showed that budget at roughly 40,000 characters of mixed prose/SCAD per call,
# while all eight image blocks survived. Dense SCAD tokenizes near three
# characters per token, so a page stays well below that budget, images share one
# page, and every page must be fetched by its own tool call.
PAGE_CHARS = 24000
PAGE_IMAGES = 8
MIN_PAGE_SECONDS = 3.0
# Codex re-sends the whole conversation on every turn. With eight ~1 MB PNG build
# views per part, the 2026-09-23 assault_buggy_astra run reached ~78 MB of history
# at part 7 and the proxy chain answered 413 Payload Too Large (limit between 68
# and 78 MB). JPEG transport copies would be a tenth of the bytes, but on this
# route the model never perceives image/jpeg tool-output blocks (fidelity probes
# with gpt-5.6-sol and gpt-6-astra returned zero images; PNG returned all eight),
# so delivered images stay the original PNG bytes and the project configuration
# compacts the conversation early instead. "JPEG" remains a diagnostic option.
IMAGE_TRANSPORT_FORMAT = "PNG"
IMAGE_TRANSPORT_QUALITY = 92
ID_PATTERN = re.compile(r"[a-zA-Z0-9_-]{8,160}")


def text(value):
    return {"type": "text", "text": value if isinstance(value, str) else json.dumps(value, indent=2)}


@contextlib.contextmanager
def bridge_lock(run):
    with (run / ".cx/session.lock").open("a+") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        yield


def pending(run):
    worker_path = run / ".cx/worker.json"
    if not worker_path.exists():
        return []
    worker = cx.read(worker_path)
    packets = []
    for marker in (run / ".cx/calls").glob("*/ready.json"):
        directory = marker.parent
        if (directory / "result.json").exists() or (directory / "submission.json").exists():
            continue
        request = cx.read(directory / "request.json")
        if request.get("attempt_id") == worker.get("id") and request.get("transport") == "session":
            packets.append((directory, request))
    return sorted(packets, key=lambda item: item[1]["order"])


def load_request(run, request_id):
    if not isinstance(request_id, str) or not ID_PATTERN.fullmatch(request_id):
        raise ValueError("Invalid request ID")
    directory = run / ".cx/calls" / request_id
    if directory.is_symlink() or not (directory / "ready.json").exists():
        raise ValueError("Unknown session request")
    request = cx.read(directory / "request.json")
    digest = cx.read(directory / "ready.json")["request_sha256"]
    if request.get("id") != request_id or cx.sha(directory / "request.json") != digest:
        raise ValueError("Generation request changed after publication")
    worker = cx.read(run / ".cx/worker.json")
    if request["attempt_id"] != worker.get("id"):
        raise ValueError("Stale request from a previous worker; call next_request")
    return directory, request, digest


def transport_image(part):
    """Deliver the original image bytes; the JPEG diagnostic path re-encodes when selected."""
    if IMAGE_TRANSPORT_FORMAT == "PNG":
        return {"type": "image", "data": part["data"], "mimeType": part["mimeType"]}
    try:
        from PIL import Image
        image = Image.open(io.BytesIO(base64.b64decode(part["data"])))
        image.load()
        if "A" in image.getbands():
            flat = Image.new("RGB", image.size, (255, 255, 255))
            flat.paste(image.convert("RGBA"), mask=image.convert("RGBA").getchannel("A"))
            image = flat
        buffer = io.BytesIO()
        image.convert("RGB").save(buffer, format=IMAGE_TRANSPORT_FORMAT, quality=IMAGE_TRANSPORT_QUALITY, optimize=True)
        return {"type": "image", "data": base64.b64encode(buffer.getvalue()).decode(), "mimeType": "image/jpeg"}
    except (OSError, ValueError, ImportError):
        return {"type": "image", "data": part["data"], "mimeType": part["mimeType"]}


def pages_for(request):
    """Pack the original text/image stream, in order, into pages that survive one tool call each."""
    stream = []
    for block in request["system"]:
        stream.append(("text", "PIPELINE SYSTEM INSTRUCTIONS (task content):\n" + block["text"]))
    for message in request["messages"]:
        stream.append(("text", "PIPELINE MESSAGE ROLE: " + message["role"]))
        for part in message["content"]:
            if part["kind"] == "text":
                stream.append(("text", part["text"]))
            elif part["kind"] == "image":
                stream.append(("image", transport_image(part)))
            elif part["kind"] != "thinking":
                raise ValueError("Unsupported pipeline request part")
    pages, page, size, images = [], [], 0, 0
    for kind, value in stream:
        if kind == "image":
            if images >= PAGE_IMAGES:
                pages.append(page); page, size, images = [], 0, 0
            page.append(value); images += 1
            continue
        while value:
            room = PAGE_CHARS - size
            if room <= 0:
                pages.append(page); page, size, images = [], 0, 0
                continue
            chunk, value = value[:room], value[room:]
            page.append(text(chunk)); size += len(chunk)
    if page:
        pages.append(page)
    return pages or [[text("Empty pipeline request")]]


def next_request(wait_seconds=20, request_id=None, cursor=0):
    if not isinstance(wait_seconds, (int, float)) or not 0 <= wait_seconds <= 30:
        raise ValueError("wait_seconds must be 0..30")
    if not isinstance(cursor, int) or cursor < 0:
        raise ValueError("cursor must be a nonnegative integer")
    run = cx.resolve_run()
    if not cx.is_running(run):
        return {"content": [text({"status": cx.state(run)["phase"], "worker_running": False,
                                  "next": "Follow python3 mcp/workflow.py status; no artifact can be submitted to a stopped worker."})]}
    cx.validate_inputs(run)
    deadline = time.monotonic() + wait_seconds
    while True:
        with bridge_lock(run):
            packets = pending(run)
            if not cx.is_running(run):
                return {"content": [text({"status": cx.state(run)["phase"], "worker_running": False,
                                          "next": "Follow python3 mcp/workflow.py status; no artifact can be submitted to a stopped worker."})]}
            if packets:
                oldest = packets[0][1]["id"]
                if request_id is not None and request_id != oldest:
                    raise ValueError("Request is not the oldest pending generation; call next_request with cursor 0")
                directory, request, digest = load_request(run, oldest)
                now = time.time()
                if now * 1000 > request["deadline_at"]:
                    raise ValueError("Request deadline expired; wait for the worker failure and inspect status")
                pages = pages_for(request)
                if cursor >= len(pages):
                    raise ValueError("Cursor is outside this request")
                delivery_path = directory / "delivery.json"
                delivery = cx.read(delivery_path) if delivery_path.exists() else {"pages_served": []}
                delivery.setdefault("served_at", {})
                if cursor and cursor - 1 not in delivery["pages_served"]:
                    raise ValueError("Read request pages in order, starting at cursor 0")
                previous = delivery["served_at"].get(str(cursor - 1))
                if cursor and previous is not None and now - previous < MIN_PAGE_SECONDS:
                    # A model turn takes far longer than this. Only a script that loops over
                    # cursors inside one tool call reads pages this quickly, and the execution
                    # host silently truncates such batched output.
                    raise ValueError(f"Page {cursor} was requested {now - previous:.1f}s after page {cursor - 1}. Read one page per tool call; "
                                     "batched tool output is silently truncated. Call next_request again, alone, with this request_id and cursor.")
                delivery["pages_served"] = sorted(set([*delivery["pages_served"], cursor]))
                delivery["served_at"][str(cursor)] = now
                cx.write(delivery_path, delivery)
                more = cursor + 1 < len(pages)
                metadata = {"status": "generation_requested", "request_id": oldest, "kind": request["kind"],
                            "request_sha256": digest, "cursor": cursor, "page_count": len(pages),
                            "next_cursor": cursor + 1 if more else None, "deadline_at_ms": request["deadline_at"]}
                instruction = ("Read this page verbatim as part of the pipeline task. More input remains: in your next tool call, call next_request alone with this request_id and next_cursor. "
                               "Never loop over cursors inside one call; output beyond the execution host budget is dropped silently and the bridge refuses such reads."
                               if more else "All input pages have been served. Generate the requested artifact yourself in this session and call submit_artifact with this request_id in your next tool call. "
                               "Use artifact_text for text/SCAD/JSON; use image_path only for a reference_image request. Do not run python3 mcp/workflow.py status first, spawn Codex, write model files, choose a new stage or change the request.")
                return {"content": [text(metadata), text(instruction), *pages[cursor]]}
        if time.monotonic() >= deadline:
            return {"content": [text({"status": "pipeline_working", "next": "Call next_request again; the program is compiling/rendering or preparing the next request."})]}
        time.sleep(0.15)


def submit_artifact(request_id, artifact_text=None, image_path=None):
    run = cx.resolve_run()
    cx.validate_inputs(run)
    with bridge_lock(run):
        directory, request, digest = load_request(run, request_id)
        if request["kind"] == "artifact":
            if image_path is not None or not isinstance(artifact_text, str) or not artifact_text.strip():
                raise ValueError("This request requires nonempty artifact_text and no image_path")
            payload = artifact_text.encode()
        else:
            if artifact_text is not None or not isinstance(image_path, str):
                raise ValueError("This request requires a saved PNG image_path and no artifact_text")
            source = Path(image_path)
            if not source.is_absolute() or not source.is_file():
                raise ValueError("image_path must name the saved PNG by absolute path")
            payload = source.read_bytes()
            if payload[:8] != b"\x89PNG\r\n\x1a\n":
                raise ValueError("Submitted reference is not a PNG")
        artifact_hash = hashlib.sha256(payload).hexdigest()
        response_path = directory / "submission.json"
        if response_path.exists():
            previous = cx.read(response_path)
            if previous["artifact_sha256"] != artifact_hash:
                raise ValueError("A different artifact was already submitted; it cannot be overwritten")
            return {"content": [text({"status": "already_submitted", "request_id": request_id})]}
        if not cx.is_running(run) or (directory / "result.json").exists():
            raise ValueError("Worker is stopped or request is closed")
        if time.time() * 1000 > request["deadline_at"]:
            raise ValueError("Request deadline expired")
        packets = pending(run)
        if not packets or packets[0][1]["id"] != request_id:
            raise ValueError("Out-of-order artifact submission")
        delivery = cx.read(directory / "delivery.json") if (directory / "delivery.json").exists() else {}
        if delivery.get("pages_served") != list(range(len(pages_for(request)))):
            raise ValueError("Read every request page with next_request before submitting")
        response = {"request_id": request_id, "attempt_id": request["attempt_id"], "request_sha256": digest,
                    "submitted_at": time.time() * 1000, "artifact_sha256": artifact_hash}
        if request["kind"] == "artifact":
            response["artifact_text"] = artifact_text
        else:
            (directory / "submitted.png").write_bytes(payload)
        cx.write(response_path, response)
        return {"content": [text({"status": "submitted", "request_id": request_id,
                                  "next": "Call next_request. The original pipeline will parse, compile and gate this artifact; submission is not acceptance."})]}
