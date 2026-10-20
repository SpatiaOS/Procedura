<div align="center">

<img src="assets/wordmark.png" alt="Procedura" width="360">

<h1>Agentic 3D Modeling with Procedural Control</h1>

<p>
<a href="https://spatiaos.github.io/projects/procedura/"><img src="https://img.shields.io/badge/Project-Page-0071e3?style=flat-square" alt="Project page"></a>
&nbsp;
<a href="https://spatiaos.github.io/projects/procedura/assets/procedura.pdf"><img src="https://img.shields.io/badge/Paper-PDF-0071e3?style=flat-square" alt="Paper (PDF)"></a>
</p>

<img src="assets/teaser.jpg" alt="Procedura turns a text prompt into an editable procedural assembly">

</div>

**Procedura** builds editable procedural 3D assemblies from text and reference
images. Each model is an OpenSCAD program with named parts and mating features,
with compiled meshes, material previews and articulation exports alongside it.

**New: run the pipeline inside Codex or Claude Code through MCP.** Your current
coding-agent conversation generates the responses; Procedura compiles, renders,
checks, retries and advances each stage. The local bridge starts no extra model
process and makes no separate model API calls. Your agent's configured account,
provider, usage limits and billing apply.

[Quick start](#quick-start) · [MCP tutorial](docs/mcp.md) ·
[Outputs](#outputs) · [Release validation](#release-validation) ·
[API CLI and Studio](docs/api.md)

## What's new

- **Generation in your coding-agent session:** use the model and reasoning effort
  selected in Codex or Claude Code, with image requests delivered through MCP.
- **The complete pipeline by default:** incremental visual drafting, assembly,
  motion awareness, refinement, PBR paint, USD/URDF export and Isaac validation.
- **Local setup for either agent:** an installer configures dependencies, project
  MCP settings, skills and hooks while preserving existing agent settings.
- **Resumable runs and verifiable outputs:** frozen inputs and toolchain hashes,
  recorded stage results and a sealed completion receipt with limitations.

The API CLI and Studio remain available. MCP adds another way to supply the
pipeline's model responses.

## Quick start

You need **Linux, Git, curl, Python 3.11+ with `venv`, and an authenticated Codex or
Claude Code installation**. Automatic native dependency downloads support Linux
x86_64. See [platforms and dependencies](#platforms-and-dependencies) for details.

> The MCP release is currently published on
> [`feat/mcp-session-setup`](https://github.com/SpatiaOS/Procedura/tree/feat/mcp-session-setup).
> These commands install that branch.

### 1. Install

```bash
curl -fsSL https://raw.githubusercontent.com/SpatiaOS/Procedura/feat/mcp-session-setup/scripts/install-mcp.sh \
  | bash -s -- --ref feat/mcp-session-setup --agent both
```

Choose `--agent codex` or `--agent claude` to configure just one agent.
The online installer uses `~/.local/share/Procedura`; add `--dir /your/path/Procedura`
to choose another location. It leaves existing checkouts at their current revision.

Alternatively, install from a clone:

```bash
git clone --branch feat/mcp-session-setup https://github.com/SpatiaOS/Procedura.git
cd Procedura
bash scripts/install-mcp.sh --agent both
```

Setup installs Bun packages, Manifold-capable OpenSCAD, Blender and a local Python
environment with Pillow. It registers project-local MCP settings, skills and
hooks. It does not request a model API key or create an API `.env`.

### 2. Start your coding agent

For the online installation:

```bash
cd ~/.local/share/Procedura
codex  # or: claude
```

For a clone, launch from that clone's directory. Use your usual authenticated
launcher and select your model and reasoning effort in the host agent. Review
project trust, MCP and hook prompts, then restart any session opened before setup.
Setup preserves your model, credentials and other agent settings.

### 3. Give it a brief and a reference

Use a PNG reference the agent can read, then ask:

> Use Procedura to reconstruct the rover in `/absolute/path/rover.png`. Build its
> six wheels, rocker-bogie suspension, body, camera mast and solar panels. Run the
> complete pipeline. Show the final preview, output directory, generated/planned
> part count, refinement result, paint and simulator results, and all limitations.

You can explicitly invoke **`$procedura` in Codex** or **`/procedura` in Claude
Code**. The agent starts the worker and answers its MCP requests until completion.
Keep that session running; detailed objects can take hours.

For a model without a supplied reference, explicitly request text-only generation
or use a host with built-in image generation. The bridge does not fall back to an
image API. See the [full tutorial](docs/mcp.md) for lifecycle commands and recovery.

## How MCP runs the pipeline

1. The agent creates a run from your complete brief and optional reference.
2. Procedura sends the next generation task, including reference and rendered
   images, through the local MCP server.
3. The current conversation generates and submits the requested artifact.
4. Procedura parses, compiles, renders and validates it, then retries or advances
   according to the existing pipeline. Completion records outputs and limitations.

The default MCP profile includes incremental drafting with visual feedback,
assembly, motion-aware generation, direct refinement, paint, articulation,
USD/URDF and simulator validation. Missing Isaac Sim is reported as unavailable
validation. Request a smaller scope explicitly if you want to omit stages.

Requests share the host conversation and its compaction behavior. Choose the
model in your coding agent; a workflow `--model` value only records metadata.
Per-stage model selection, independent fresh contexts and per-request token
accounting are unavailable in this transport.

## Outputs

Each run writes to **`outputs/mcp/<run_name>/`** inside its checkout:

```text
final.scad                      editable procedural source
final.obj                       compiled mesh
final_summary.txt               refinement verdict and geometry summary
preview_final/                  shape previews
final_materials.json            per-part material assignments
final_painted.scad/.obj/.mtl     painted source and mesh deliverables
preview_painted/                 PBR previews, including pbr-isometric.png
motion/final_motion.usda         articulated OpenUSD scene
motion/urdf/robot.urdf           URDF, with associated link meshes
motion/isaac_validation.json     simulator report when validation runs
.cx/completion.json             sealed receipt, limitations and artifact hashes
```

Paint and motion files depend on the requested stages and their results. A
completion receipt distinguishes `done` from `done_with_limitations`; inspect the
reported limitations and refine summary alongside the previews. Simulator hard
`ok` and `strictOk` are separate results.

## Platforms and dependencies

| Requirement | Setup behavior |
| --- | --- |
| Git, curl, Python 3.11+ with `venv` | Install before running the bootstrap command |
| Codex or Claude Code | Install and authenticate normally; shell execution and native MCP image support are required |
| Bun and pipeline packages | Installed by the helper |
| OpenSCAD with the Manifold backend | Checked or downloaded on Linux x86_64; older CGAL-only builds are rejected by default |
| Blender | Checked or downloaded on Linux x86_64 for render feedback and previews |
| Pillow | Installed in `.procedura/venv` inside the checkout |
| NVIDIA Isaac Sim | Optional separate installation for physics validation; never downloaded automatically |

For existing native tools or an Isaac installation:

```bash
bash scripts/install-mcp.sh --agent both \
  --openscad /path/to/openscad --blender /path/to/blender \
  --isaac /path/to/isaacsim
```

macOS users can supply compatible native binaries, but end-to-end macOS support
has not been validated. On Windows, use a Linux environment such as WSL.
Additional setup options are in the [MCP tutorial](docs/mcp.md#quick-setup).

## Parallel cases and recovery

Use **one active MCP run per checkout**. To run two cases at once, create two
clones, run setup in each, and open one coding-agent session in each directory.
Separate terminal tabs are enough; no particular terminal manager is required.

Run these commands from the checkout when diagnosing a setup or stopped worker:

```bash
bash scripts/install-mcp.sh --agent both --check
python3 mcp/workflow.py status --log
```

Use the agent option you originally installed. The agent normally handles the
worker lifecycle; `python3 mcp/workflow.py run` in an ordinary terminal waits for
the coding-agent session to answer MCP tasks. Follow the
[recovery instructions](docs/mcp.md#outputs-recovery-and-troubleshooting) before
resuming. Keep code, inputs, settings and binaries unchanged during an active run.

## Release validation

Two complete reference-based cases ran in parallel with **Claude Code and Opus
5.5 through a custom gateway**, using the default MCP profile at commit
[`23d1094`](https://github.com/SpatiaOS/Procedura/commit/23d1094bd13a14abfeff4e670f5ff79b8aeb15e1).
Both completed refinement with verdict `ok`, painted previews, USD/URDF export,
and Isaac validation with hard `ok: true` and `strictOk: true`.

| Case | Generated/planned parts | Elapsed time | Completion |
| --- | --- | --- | --- |
| `assault_buggy` | 26/26 | 4h 57m | `done_with_limitations` |
| `00001198` rover | 22/22 | 4h 28m | `done` |

The buggy retained two motion warnings for `machine_gun`: its measured geometry
had moved, so that measurement was demoted to a categorical prior, and missing
anchor evidence left the joint frame at the origin. Refine summaries also noted
remaining visual differences: the buggy's gun proportions and the rover's
camera-head proportions, chimney height and antenna offset.

Elapsed times include local compilation/rendering/simulation and gateway latency;
they vary with hardware, host model and provider limits. These runs validate this
integration path; their gateway authentication does not establish subscription
coverage for another account.

The bridge also has Python and TypeScript tests, type checking, and a native
OpenSCAD/Blender fixture covering MCP image delivery and sealed artifacts from a
checkout whose path contains spaces. See
[maintenance and validation](docs/mcp.md#maintenance-and-validation) for commands.

## API CLI and Studio

For direct model API access, run the dependency installer and configure your
endpoint and key in `.env`, then use the CLI:

```bash
bash scripts/install-deps.sh
# Configure .env before starting the API run.
bun run scripts/procedura.ts -o outputs/daybed \
  --prompt "a brass daybed with tapered legs"
```

The API CLI has different defaults: the command above is text-only incremental
drafting and refinement. See the [API CLI guide](docs/api.md) for reference images,
full-stage flags, model configuration, batch runs and stage reruns.

Studio uses the same API configuration and launches the CLI from the checkout:

```bash
cd web
bun install
bun run start  # http://localhost:8080
```

See [web/README.md](web/README.md) for Studio details.

## Acknowledgements

The vendored LLM harness in `vendor/harness/` is an independent implementation
of architectural patterns from two MIT-licensed projects —
[opencode](https://github.com/anomalyco/opencode) (the four-axis
`Protocol / Endpoint / Auth / Framing` route model, the permission evaluator,
the cache-breakpoint policy) and
[openclaw](https://github.com/openclaw/openclaw) (the sandbox bridge, the
context engine, trajectory events). Credit for those designs belongs to them.

## License

MIT. See [LICENSE](LICENSE).

Procedura invokes OpenSCAD and Blender as external programs; both are GPL and
neither is redistributed here. The four Blender scripts under `scripts/` use
Blender's `bpy` API and are MIT-licensed, which is GPL-compatible. No
third-party assets ship in this repo — the PBR studio environment is read from
your own Blender installation at render time.
