# Use Procedura inside Codex or Claude Code

Procedura can build a model using the model already running in your coding-agent
conversation. A local MCP server returns each original pipeline task and its
reference/render images. Your coding agent generates the response, and Procedura
compiles, measures, retries and advances the pipeline. The session transport
starts no additional coding agent and makes no separate model API calls.

Your coding agent's account, provider, model availability, usage limits and
permissions still apply. Procedura does not log into accounts or guarantee that
a particular plan covers the run. A custom gateway uses that gateway's model
access. The earlier Claude Code/Gemini tests used a custom gateway, not native
Claude subscription authentication.

## Quick setup

Supported host: Linux with Python 3.11+. The native dependency downloader targets
Linux x86_64. macOS users can supply compatible OpenSCAD/Blender executables; that
platform has not been validated end to end. Windows needs a Linux environment
such as WSL. The coding agent needs shell execution and native MCP image support;
a chat client with only text tools is insufficient.

Install Git, Python 3.11+ with `venv` support, and your chosen coding agent first.
Authenticate that agent normally. The MCP release is currently on
[`feat/mcp-session-setup`](https://github.com/SpatiaOS/Procedura/tree/feat/mcp-session-setup);
the commands below install that branch:

```bash
curl -fsSL https://raw.githubusercontent.com/SpatiaOS/Procedura/feat/mcp-session-setup/scripts/install-mcp.sh \
  | bash -s -- --ref feat/mcp-session-setup --agent codex
```

Use `--agent claude` for Claude Code or `--agent both` for both agents. The online
installer clones into `~/.local/share/Procedura`; pass `--dir /your/path/Procedura`
for another location. It leaves an existing checkout at its current revision;
it never pulls, resets or overwrites your work. To install a published branch or
tag, use its bootstrap URL and pass the matching `--ref <branch-or-tag>`.

From an existing clone:

```bash
git clone --branch feat/mcp-session-setup https://github.com/SpatiaOS/Procedura.git
cd Procedura
bash scripts/install-mcp.sh --agent both
```

The installer installs Bun dependencies, checks/downloads a Manifold-capable
OpenSCAD and Blender, installs Pillow in `.procedura/venv`, records executable
paths/hashes and creates project-local MCP configuration and hooks. It does not
request model API keys, create an API `.env`, change global agent settings,
select a model or disable permission checks. Existing project settings and other
MCP servers/hooks are preserved; a conflicting server name is reported.

Isaac Sim is optional and is never automatically downloaded. Without it, motion
export may complete but physics validation is recorded as unavailable. For the
complete validation stage, install Isaac Sim separately and pass its directory:

```bash
bash scripts/install-mcp.sh --agent both --isaac /path/to/isaacsim
```

If dependencies already exist:

```bash
bash scripts/install-mcp.sh --agent both --no-deps \
  --openscad /path/to/openscad --blender /path/to/blender
```

`--no-deps` needs existing Bun packages and Pillow in the Python interpreter used
for setup. `PROCEDURA_SETUP_PYTHON=/path/to/python3` selects that interpreter.
Alternatively call `python3 mcp/setup.py --agent both` with the same binary flags.
Neither route needs Codex installed to configure Claude Code.

## Start a coding-agent session

Open a terminal in the installed checkout. Start `codex` or `claude` using your
normal launcher and chosen model. Review project trust, the `procedura` MCP
server and its hooks when the client asks. The installer does not approve those
reviews for you. Restart an already-open coding-agent session after setup.

For Codex, project-local `.codex/config.toml` registers the stdio MCP server and
`.codex/hooks.json` loads the run hooks. The skill lives at
`.agents/skills/procedura/SKILL.md`. Setup preserves your configured model and
effort; it adds a 180k compaction threshold only when the project has no explicit
threshold, because large motion requests need room to fit before compaction.

For Claude Code, `.mcp.json` registers the same server,
`.claude/settings.local.json` holds the hooks, and a copy of the skill lives at
`.claude/skills/procedura/SKILL.md`. `CLAUDE.md` loads the project instructions.
Use `/mcp` to inspect the connection. A custom authenticated launcher also works;
use the launcher you normally use for that agent. Local aliases are not supplied
by Procedura.

Configuration formats and trust behavior are described in the official
[Codex MCP documentation](https://developers.openai.com/codex/mcp),
[Codex hooks documentation](https://developers.openai.com/codex/hooks),
[Claude Code MCP documentation](https://code.claude.com/docs/en/mcp) and
[Claude Code hooks documentation](https://code.claude.com/docs/en/hooks).

## Ask for a model

Put a PNG reference somewhere the agent can read. Then give the agent a complete
brief, for example:

> Use Procedura to reconstruct the rover in `/absolute/path/rover.png`. Build its
> six wheels, rocker-bogie suspension, body, camera mast and solar panels. Run the
> full pipeline, then show the final preview, output paths and all validation
> limitations.

In Codex you can invoke the skill directly with `$procedura`; in Claude Code use
`/procedura`. Plain-language requests also work when the agent selects the skill.
You should not need to manually call `next_request` or `submit_artifact`.

For a text-only model, explicitly request it. Without a reference, the reference
generation stage requires the agent's built-in image-generation capability. If
that capability is unavailable, supply a PNG or explicitly request text-only;
the MCP pipeline will not call an image API or silently change modes.

The default MCP profile enables incremental visual drafting, assembly, motion
awareness, direct refinement, paint, motion, USD/URDF and Isaac validation. The
program owns all content retries and gates. The agent must preserve the brief,
read every request page/image and generate in the current conversation.

Keep the coding-agent session working until completion. Large objects can take
hours; subscription/provider limits and shared-conversation history affect time
and output. This mode does not reproduce the API workflow's independent fresh
conversation or model choice for each stage.

## Commands the agent runs

These are lifecycle commands, not a headless model-generation service:

```bash
python3 mcp/workflow.py new my_rover --brief-file brief.txt --ref rover.png
python3 mcp/workflow.py run
python3 mcp/workflow.py finish
```

`run` waits for the current session to answer MCP tasks. In Claude Code the agent
starts it with Bash `run_in_background=true`; in Codex it uses a short execution
yield. A terminal running `workflow.py run` by itself will wait for answers.

`new --model <name>` records optional metadata about the host session; it cannot
change or verify that model. The default metadata is `current-session`.
`--no-motion`, `--no-paint`, `--no-assembly` and `--text-only` are explicit scope
choices made when the user requests them. The model cannot silently add them.

One checkout has one active MCP run. For parallel cases, use separate clones,
run setup in each, and open a coding-agent session in each directory. A Herdr
workspace is useful for organizing tabs, but is not required by Procedura.

## Outputs, recovery and troubleshooting

Outputs live in `outputs/mcp/<run_name>/`:

| File | Purpose |
| --- | --- |
| `final.scad`, `final.obj` | Editable parametric source and compiled mesh |
| `preview_final/`, `preview_painted/` | Shape and material previews |
| `final_materials.json`, `final_painted.*` | Material assignments and painted deliverables |
| `motion/` | USD, URDF, meshes and simulator evidence |
| `.cx/completion.json` | Sealed receipt, limitations and full artifact inventory |

The final answer must include limitations. Missing Isaac, failed parts, assembly
warnings or strict-validation advisories are not silently converted into passes.
`finish` verifies the sealed receipt and hashes without dumping the whole
inventory into the conversation.

Check setup without downloading or changing configuration:

```bash
bash scripts/install-mcp.sh --agent both --check
```

Use the agent option you originally installed. When a worker stops, inspect:

```bash
python3 mcp/workflow.py status --log
```

For a transient infrastructure failure, resume the same unchanged run once with
`python3 mcp/workflow.py run`. After context compaction or an operator reconnect,
the agent reads the pending request from cursor 0. Existing successful worker
evidence can recover completion without regenerating the model.

Do not update code, inputs, binaries, configuration or the profile during an
unfinished run: their recorded hashes prevent resuming under changed settings.
Finish first, or explicitly abandon the run before reconfiguring. A repeated
infrastructure blocker should be reported with the original evidence preserved.

If the MCP tools are absent, check the installed agent configuration, project
trust and server startup, then restart that agent. If OpenSCAD lacks `--backend`,
install a Manifold build. If a large request repeatedly compacts, give the host
conversation enough context to read the complete request; do not skip pages.

## Maintenance and validation

Two full reference-based cases, `assault_buggy` and `00001198`, completed on this
Procedura integration in Claude Code with Opus 5.5 through a custom gateway.
Both completed refinement, paint, USD/URDF and strict Isaac validation. The buggy
retained two motion warnings. See the [release validation results](../README.md#release-validation)
for part counts, elapsed times and known limitations. Earlier native Codex and
Claude Code/Gemini runs used the Tether prototype and are separate evidence.

The integration uses Procedura's current checked-in engine with two transport
seams in `vendor/harness/src/llm/route.ts` and `src/imagegen/images.ts`, plus a
file-URL decoding fix so native pipeline assets load from paths with spaces.

`mcp/engine.lock.json` records reviewed engine bytes; each new run additionally
freezes its prompt/reference, profile, MCP wrapper/configuration and toolchain.
The ordinary API CLI and Studio keep their existing transport unless MCP mode
is explicitly selected by the launcher.

Run checks:

```bash
python3 -m unittest discover -s mcp/tests -p 'test_*.py'
bun test mcp/tests
bun run typecheck
```

For an optional native integration fixture using real OpenSCAD and Blender:

```bash
python3 mcp/tests/native_smoke.py --openscad /path/to/openscad --blender /path/to/blender
```

It creates an isolated checkout in a temporary path containing spaces, exercises
setup from a downloaded-style stdin script twice, and feeds fixed test artifacts
through the real MCP/pipeline. It checks compilation, image delivery, rendering,
refinement and the sealed receipt without invoking a model. Its explicit
text-only/no-paint/no-motion scope belongs to this small fixture; the normal MCP
profile retains all stages. The temporary evidence directory is printed and kept.

After intentional engine changes, review the diff and regenerate its lock with
`python3 mcp/inventory.py --write-lock` before creating a new run. Never refresh
the lock as a workaround for resuming an unfinished run with changed code.
