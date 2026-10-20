---
name: procedura
description: Build an editable procedural 3D model from a complete brief and optional reference image using the local Procedura MCP pipeline. Use for 3D modelling, not repository maintenance or setup.
---

# Procedura in the current coding-agent session

Use `python3 mcp/workflow.py status` and follow its next action. The native MCP
tools `procedura.next_request` and `procedura.submit_artifact` must be available.
If the client defers tools, use its tool search to discover the `procedura` tools.
If missing, stop and report that the user must run the setup and restart the
coding agent in this project. Never replace MCP with another model process/API.

## Create and start

For a new request, preserve the entire brief and reference:

```bash
python3 mcp/workflow.py new <snake_case_name> --brief-file <brief.txt> --ref <reference.png>
```

The user may supply the brief with `--brief` instead. `--model` records the outer
session's model name; it does not select a model. The default metadata is
`current-session`. Models and effort come from the user's agent configuration.
Do not set a model or launch wrapper on their behalf unless requested.

Default stages: incremental draft with visual feedback and assembly/motion
awareness, direct refinement, paint, motion, USD/URDF and Isaac validation.
Use `--no-motion`, `--no-paint`, `--no-assembly` or `--text-only` only when the user
requests that reduction. Without a reference, `ref generate` requires the host's
built-in image generation tool. If unavailable, ask for a PNG; never silently
switch to text-only or an image API.

Start `python3 mcp/workflow.py run` with a short execution yield so the worker can
wait while you answer MCP requests. In Claude Code use Bash
`run_in_background=true`. In Codex use `yield_time_ms=1000`; retain that execution
session for polling. One worker per run, one active run per checkout. Concurrent
cases use separate checkouts and coding-agent sessions.

## Read, generate, submit

1. Call `procedura.next_request(wait_seconds=20)`.
2. Read all text and image blocks. When `next_cursor` is non-null, request that
   cursor with the same ID in a separate tool call until every page is read.
   Do not batch pages, summarize the input away or drop images. A page requested
   too quickly must be retried in its own tool call. After compaction or reconnect,
   read the pending request again from cursor 0.
3. Generate exactly the requested artifact yourself in this conversation,
   preserving the requested JSON/SCAD/critique format.
4. Submit its complete text using `procedura.submit_artifact`. For a
   `reference_image` task, use the built-in image generation tool with the exact
   prompt and submit its saved PNG via `image_path`.
5. Repeat until the worker completes. Submission is not acceptance: Procedura
   parses, compiles, renders, measures, retries and commits through its own code.

Never manually write geometry, state, material/motion files or receipts. Never
skip stages, alter frozen settings/gates, choose the next part or delegate model
generation. If no request is ready, wait through `next_request(wait_seconds=20)`.
Do not run `status` between pending requests. `next_request` reports a stopped
worker. Only then inspect `python3 mcp/workflow.py status --log`. For an unchanged
infrastructure failure resume the same program once. Preserve and report a repeated
blocker. A slow compile/render is not a failure.

Use native MCP tools when available. If Codex exposes them through `functions.exec`,
forward every text and image block with exactly one call:

```javascript
const r = await tools.mcp__procedura__next_request({"wait_seconds":20}); for (const c of r.content) { if (c.type === "image") image(c); else if (c.type === "text") text(c.text); }
```

```javascript
text(await tools.mcp__procedura__submit_artifact({"request_id":"<id>","artifact_text":"<complete artifact>"}));
```

Lifecycle commands use one literal checked wrapper, without command chaining:
`text(await tools.exec_command({"cmd":"python3 mcp/workflow.py status"}));`.
Use the same wrapper for `write_stdin` polling. To view a preview:
`const r = await tools.view_image({"path":"<absolute preview path>"}); image(r.image_url);`.

## Finish

After completion, run `python3 mcp/workflow.py finish`, inspect the listed preview,
and give the output directory and deliverables. Report generated/planned parts,
refinement verdict, paint and simulator results with all limitations. Distinguish
Isaac hard `ok` from `strictOk`; retain recovered failures as history. Keep the
full hash inventory on disk. Never describe missing or skipped validation as a pass.

Session-mode requests share conversation history and compaction. Per-call model
selection and token accounting are unavailable. Do not claim fresh-context/API
parity or infer subscription billing from tool execution.
