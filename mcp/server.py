#!/usr/bin/env python3
"""Small dependency-free MCP stdio server for the current coding-agent session."""
import argparse
import json
from pathlib import Path
import sys
import workflow as cx
import bridge

TOOLS = [
    {"name": "next_request", "description": "Get the oldest generation task from the running fixed pipeline: original prompt text and images, one page per call. Read every page in order, each in its own tool call (never loop over cursors inside one call: output may be truncated and the bridge refuses such reads). When next_cursor is null, generate the artifact in this coding-agent session and call submit_artifact. No nested agent/model calls.",
     "inputSchema": {"type": "object", "properties": {
         "wait_seconds": {"type": "number", "minimum": 0, "maximum": 30, "default": 20},
         "request_id": {"type": "string", "description": "Omit for the first page; use the returned ID for further pages"},
         "cursor": {"type": "integer", "minimum": 0, "default": 0}}, "additionalProperties": False}},
    {"name": "submit_artifact", "description": "Submit your complete answer to the current generation task. Use artifact_text for SCAD/JSON/critique text or image_path for a reference PNG. The program owns writing outputs, validation, acceptance, retries and stage order. Do not edit run files.",
     "inputSchema": {"type": "object", "properties": {
         "request_id": {"type": "string"}, "artifact_text": {"type": "string"},
         "image_path": {"type": "string", "description": "Absolute saved PNG path; reference-image requests only"}},
         "required": ["request_id"], "additionalProperties": False}},
]
INSTRUCTIONS = "The fixed Procedura MCP pipeline owns stage order and every gate. Use next_request to read its full prompt/images one page per tool call, generate the answer in this session, then submit_artifact. Read all pages before submitting; never batch cursors in one call. While the worker runs, do not run python3 mcp/workflow.py status between requests: next_request reports a stopped worker. After context compaction, reread the current request from cursor 0. Never spawn a model, manually write geometry/state, or invent a stage. Submission is not acceptance: the original code parses/compiles/gates it. If no worker is active, follow python3 mcp/workflow.py status. Honor the user's requested outer coding-agent launcher and model."


def dispatch(message):
    method, params = message.get("method"), message.get("params") or {}
    if method == "initialize":
        return {"protocolVersion": "2024-11-05", "capabilities": {"tools": {"listChanged": False}},
                "serverInfo": {"name": "procedura", "version": "3.0.0"}, "instructions": INSTRUCTIONS}
    if method == "ping":
        return {}
    if method == "tools/list":
        return {"tools": TOOLS}
    if method == "tools/call":
        name, args = params.get("name"), params.get("arguments") or {}
        try:
            if name == "next_request":
                return bridge.next_request(**args)
            if name == "submit_artifact":
                return bridge.submit_artifact(**args)
            raise ValueError("Unknown tool")
        except (ValueError, TypeError, OSError, KeyError) as error:
            return {"isError": True, "content": [bridge.text(str(error))]}
    raise ValueError("Method not found")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, default=cx.ROOT)
    args = parser.parse_args()
    root = args.workspace.resolve()
    # Tests and isolated packs explicitly choose their workspace at server start;
    # tools cannot change it or address arbitrary run paths.
    cx.ROOT = root; cx.RUNS = root / "outputs/mcp"; cx.ACTIVE = cx.RUNS / "ACTIVE_V2"
    cx.PROFILE = root / "mcp/profile.json"; cx.LOCK = root / "mcp/engine.lock.json"
    cx.TOOLCHAIN = root / ".procedura/toolchain.local.json"
    for line in sys.stdin:
        try:
            message = json.loads(line)
            if "id" not in message:
                continue
            try:
                result = dispatch(message)
                response = {"jsonrpc": "2.0", "id": message["id"], "result": result}
            except ValueError as error:
                response = {"jsonrpc": "2.0", "id": message["id"], "error": {"code": -32601, "message": str(error)}}
        except (ValueError, TypeError) as error:
            response = {"jsonrpc": "2.0", "id": None, "error": {"code": -32700, "message": "Invalid JSON-RPC message"}}
        print(json.dumps(response), flush=True)


if __name__ == "__main__":
    main()
