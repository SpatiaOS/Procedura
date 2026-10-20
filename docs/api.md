# API CLI and Studio guide

[Back to the README](../README.md) · [Coding-agent MCP tutorial](mcp.md)

This guide runs Procedura through a configured model API endpoint. It requires
provider credentials and uses the API CLI defaults described below. For the MCP
release, use the coding-agent tutorial: generation happens in your existing
Codex or Claude Code session, and its default profile enables the full pipeline.

## Install

One command installs everything: Bun, the packages, a Manifold-capable OpenSCAD,
Blender, and a starter `.env`.

```bash
git clone git@github.com:SpatiaOS/Procedura.git && cd Procedura
bash scripts/install-deps.sh
```

It is safe to re-run — anything already working is left alone — so it doubles as
a "what am I missing?" check:

```bash
bash scripts/install-deps.sh --check        # report only, install nothing
bash scripts/install-deps.sh --no-blender   # skip the ~350 MB download
PREFIX=/opt bash scripts/install-deps.sh    # install binaries somewhere else
```

| what | where it goes | why |
|---|---|---|
| **[Bun](https://bun.sh) ≥ 1.3** | `~/.bun` | the whole pipeline is one TypeScript program |
| **OpenSCAD** (newest snapshot) | `~/opt/openscad` | compiles the generated program to a mesh |
| **Blender** (newest release) | `~/opt/blender/` | the AO / parts-colour / PBR renders the critic looks at |
| **Packages** | `node_modules/`, `web/node_modules/` | pipeline + Studio dependencies |
| **`.env`** | repo root | copied from `.env.example`; add your endpoint and key |

Two things it deliberately does not do. It never installs **NVIDIA Isaac Sim**
(multi-GB and account-gated) — that is only needed for `--motion` physics
validation, and everything else runs without it. And it never touches an
existing `.env`.

> **OpenSCAD must have the Manifold backend.** A pre-Manifold build does not
> fail — it falls back to CGAL and gets orders of magnitude slower. One `hull()`
> measured **1774s under CGAL against 1.77s under Manifold**, which turns a
> working run into one that looks hung for hours. The installer checks for
> `--backend` rather than merely "is openscad installed", because the version
> most distributions package is 2021.01 and predates it. Procedura makes the
> same check at startup and refuses to run without it unless you set
> `PROCEDURA_ALLOW_CGAL_OPENSCAD=1`.

### Configuration

Procedura ships **no keys and no hosted backend**. It talks to whatever endpoint
you configure, through two variables:

```bash
OPENAI_API_KEY=sk-...
OPENAI_BASE_URL=https://api.openai.com/v1
```

Anything that speaks `POST {base}/chat/completions` with SSE works here — the
OpenAI API, OpenRouter, a corporate gateway, or a local vLLM / Ollama / LM
Studio server. For Gemini on its *native* GenAI path (where `thinkingConfig` and
first-class thought parts are available), set `GEMINI_API_KEY` /
`GEMINI_BASE_URL` instead and use a `gemini:` model key.

Model selection is just a string. Any id your endpoint serves works, whether or
not this repo has heard of it:

```bash
PROCEDURA_MODEL=gpt-5.2                      # via the openai provider
PROCEDURA_MODEL=gemini:gemini-3-pro-preview  # force the native Gemini transport
```

`src/config/models.ts` holds a small catalog of known-good presets and explains
the resolution order. `.env.example` documents every other knob — reasoning
effort, binary paths, timeouts, retry budgets.

**Reference images are an input, not an assumption.** Procedura will not call an
image API on your behalf: with no `--image` and no image model configured, a run
goes text-only rather than quietly spending on an endpoint you never named.
Supplying a reference is the single largest fidelity gain available, so pass
`--image <path>` when you have one — or name an image model and let the run
render its own:

```bash
PROCEDURA_IMAGE_MODEL=gpt-image-1    # unset = generation off; --image-model for one run
```

It reuses `OPENAI_BASE_URL` / `OPENAI_API_KEY`, so nothing else needs setting.

---

## Usage

### The default run

Nothing to configure past the endpoint. Plans the parts, builds them one at a
time, refines the whole shape — from the prompt alone:

```bash
bun run scripts/procedura.ts -o outputs/daybed \
  --prompt "a brutalist brass daybed with tapered legs"
```

That is text-only, part-by-part, with no image API and no Blender in the draft
loop, and it is what you get with no flags at all. It is the cheapest useful
configuration, not the strongest one.

### The best-quality run

**Everything on.** This is what the paper's results use, and every switch below
buys accuracy at the cost of time or money:

```bash
# big plans mean long single calls; the default 600s cap is not enough
export PROCEDURA_LLM_TIMEOUT_MS=1800000
export PROCEDURA_LLM_DEADLINE_MS=1800000

bun run scripts/procedura.ts -o outputs/gripper \
  --image gripper_ref.png \
  --3d-feedback \
  --assembly \
  --paint \
  --motion --motion-urdf \
  --max-steps 12 \
  --prompt "a two-finger parallel robot gripper"
```

| switch | what it buys |
|---|---|
| `--image ref.png` | a reference to reconstruct — the single largest fidelity gain. Set `PROCEDURA_IMAGE_MODEL` instead to have the run render its own |
| `--3d-feedback` | the generator sees the build-so-far before each part instead of reading about it. One Blender pass per part |
| `--assembly` | parts join through real mating features — shared-nominal pegs/sockets, bolt patterns, snaps — rather than bare overlap |
| `--paint` | per-part PBR materials assigned by a vision call |
| `--motion --motion-urdf` | articulation planned, exported to OpenUSD + URDF, and validated headlessly in Isaac |
| `--max-steps 12` | a longer refine budget; each cycle is render → critic → patch → gate, at 2 LLM calls |

Raise the timeouts rather than capping the part count — a long plan is the
pipeline working, and `PROCEDURA_MAX_PARTS` degrades the model to buy back time.

### In between

```bash
# reconstruct a reference image you already have — the biggest single win
bun run scripts/procedura.ts -o outputs/daybed --image ref.png --prompt "..."

# generate the reference too (needs PROCEDURA_IMAGE_MODEL, or pass --image-model)
bun run scripts/procedura.ts -o outputs/daybed --image-model gpt-image-1 --prompt "..."

# long prompts, without fighting your shell
bun run scripts/procedura.ts -o outputs/daybed --prompt-file my_prompt.txt

# resume the refine loop on an existing run, or force a re-draft
bun run scripts/procedura.ts -o outputs/daybed --max-steps 12
bun run scripts/procedura.ts -o outputs/daybed --prompt "..." --redo

# the old monolithic draft: one call, no plan stage. Needs a reference
bun run scripts/procedura.ts -o outputs/daybed --one-shot --image ref.png --prompt "..."
```

### Over a dataset

The paper's batches run the shape and articulation together, then paint each
finished case as a separate pass — so a paint failure never costs the geometry:

```bash
bun run scripts/batch.ts \
  --prompts cases.jsonl --images \
  --incremental \
  --motion --validate --urdf \
  --max-steps 3 \
  --parallel 2 \
  --out-root outputs/batch_v1

# Replace my_case with a generated case ID.
bun run scripts/paint.ts outputs/batch_v1/my_case
```

Keep `--parallel` at ~2 with `--motion --validate`: concurrent headless Isaac
instances exhaust GPU VRAM. Refine-only batches tolerate 8-12; each refine cycle
spawns Blender, so running every case at once will exhaust memory long before it
exhausts your rate limit.

### Flags

| flag | meaning |
|---|---|
| `-o, --output DIR` | output directory (required) |
| `--prompt TEXT` / `--prompt-file PATH` | the text prompt |
| `--image PATH` | reconstruct this reference image |
| `--no-image` | text-only — the default when no reference is available |
| `--image-model M` | generate the reference with this model, for this run only |
| `--3d-feedback` | show the generator a render of the build-so-far before each part (off by default) |
| `--one-shot` | opt out of the part-by-part draft: one call, no plan stage |
| `--assembly` | inline the mating-feature library so parts join through real interfaces |
| `--paint` | material pass: a vision call assigns each part a PBR material |
| `--motion` | OpenUSD/Isaac articulation export |
| `--motion-urdf` | also emit URDF + per-link meshes |
| `--max-steps N` | refine-loop budget in edit cycles (default 6; 2 LLM calls each) |
| `--no-refine` | draft only — promote the draft to `final.*` |
| `--redo` | force a re-draft even if artifacts exist |
| `--agent-model` / `--scad-model` / `--paint-model` / `--motion-model` | per-stage model override |

`bun run scripts/procedura.ts --help` lists all of them, including the ablation
switches (`--no-plan`, `--motion-aware`, `--motion-only`) used in the paper.

### What a run writes

```
image.png                       the reference (generated, or the one you passed)
plan.json  _parts/NN_<name>/    --incremental: the part list + per-part artifacts
draft.scad  draft.obj           the drafted program and its mesh
final.scad  final.obj           after refine — the deliverable
final_summary.txt               finish verdict + shipped connectivity
preview_final/                  Blender AO render of the final mesh
final_materials.json            --paint: part → colour / metalness / roughness
final_painted.scad/.obj/.mtl    --paint, plus preview_painted/ (PBR Cycles render)
motion/final_motion.usda        --motion, plus motion/urdf/ and motion/links/
_trajectory/procedura-<id>.jsonl   every prompt, response, tool call and gate decision
_refine_steps/step_NNN/         one dir per refine cycle (render, diagnosis, edit, compile)
```

Every OBJ Procedura writes is normalized to a unit bounding box; STL is opt-in
(`--export-stl`).

### Re-running a stage

`paint.ts` and `motion.ts` both take a finished run directory, so either stage
can be re-run — with a different model, or after a failure — without touching
the geometry:

```bash
bun run scripts/paint.ts outputs/daybed --refine-steps 2
bun run scripts/motion.ts outputs/daybed --urdf
```

Both commands accept `--model KEY`. Use `--refine-steps N` for the paint budget
or `--no-validate` to skip motion simulation when that is your intended scope.

To compare runs across configurations — renders, source, metrics and trajectory,
side by side per case:

```bash
bun run scripts/results-server.ts --root a=outputs/run_a --root b=outputs/run_b
```

---

## Studio

A web front-end for the same pipeline: compose a run (prompt, optional
reference image, preset or per-stage toggles), watch it build part by part,
and inspect every intermediate artifact of any run — plan, parts, refine loop,
final mesh in 3D, materials, articulation.

```bash
cd web && bun install && bun run start      # http://localhost:8080
```

It spawns the CLI from the repo root and reads the same `.env`, so there is
nothing extra to configure. See [web/README.md](../web/README.md).
