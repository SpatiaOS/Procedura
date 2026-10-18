/**
 * Incremental draft stage — the part-by-part variant of the draft phase.
 *
 *   text → image-gen → PLAN (ordered part list) → for each part:
 *           render the build so far → generate ONE part → compile → gate → next
 *
 * Where the monolithic `runDraft` emits the entire model in a single SCAD-gen
 * call, this builder grows the model one top-level module at a time:
 *
 *   1. image-gen (identical to runDraft).
 *   2. a PLAN call decomposes the object into an ordered list of top-level
 *      parts (structural first; each attaches to something earlier).
 *   3. for each planned part, in order:
 *        a. a generation call emits ONLY that part (new params + helpers +
 *           the part module + its assembly placement), which we splice
 *           deterministically into the accumulated SCAD via marker comments;
 *        b. a compile-fix retry loop re-prompts on a broken splice;
 *        c. a delta-aware connectivity gate rejects a part that lands as a new
 *           floater, re-prompting it with the measured reason.
 *
 * A per-part focused refine used to sit after (c). It changed 64% of parts and
 * improved none of them measurably, so it was removed; the whole-model Phase 2
 * refine is the only refine.
 *
 * The accumulated SCAD carries three marker comments (PARAMS / MODULES /
 * PLACEMENTS) that the builder owns and splices against. They are re-derived
 * defensively by `ensureMarkers` and stripped from the final draft.scad.
 *
 * Outputs match runDraft so the rest of the pipeline (Phase 2 refine, resume
 * detection) is unchanged: image.png, draft.scad, draft.stl, draft.obj,
 * prompt.txt, effective_text.txt — plus plan.json and per-part artifacts under
 * `_parts/NN_<name>/`.
 */

import { fileURLToPath } from "node:url";
import {
  mkdirSync, writeFileSync, readFileSync, existsSync, statSync, copyFileSync, rmSync,
} from "node:fs";
import { join, resolve, dirname, basename } from "node:path";
import { tmpdir } from "node:os";

import { createStageEmitter } from "../trajectory/emitter.ts";
import { generateOnce } from "../llm/generate.ts";
import type { ModelRef } from "@harness/template/types";
import type {
  CanonicalPart,
} from "@harness/template/llm/protocol";

import { routeForModel } from "../llm/routes.ts";
import { longTimeoutFetch } from "../llm/long-timeout-fetch.ts";
import { resolveModel, DEFAULT_MODEL } from "../config/models.ts";
import {
  generateImage, resolveImageModel, imageGenAvailable, imageGenDisabledReason,
} from "../imagegen/images.ts";
import { loadImageBase64 } from "../imagegen/resize.ts";
import { extractOpenscadCode } from "../scad/extract.ts";
import { compileScad } from "../scad/compile.ts";
import { timeStage, addStage } from "./stage-timer.ts";
import { analyzeConnectivity, evaluateConnectivityGate } from "../mesh/connectivity.ts";
import { loadSTL, computeBBox } from "../mesh/stl.ts";
import type { STLMesh } from "../mesh/stl.ts";
import { publishMesh } from "../mesh/normalize.ts";
import {
  findModuleSpans, findFunctionSpans, stripCommentsAndStrings,
  sanitizeIdentifier, compilePartsInAssembly, listModuleInstances,
} from "../scad/parts.ts";
import {
  analyzeRotationalSymmetry, analyzeContactRegion, DEFAULT_CONTACT_DISTANCE_FRAC,
  analyzeMateRegistration, computeMeshVolume,
} from "../motion/geometry.ts";
import {
  createIncrementalMotionSidecar, saveIncrementalMotionSidecar, sanitizeMotionDecl,
  extractBalancedJson, INCREMENTAL_MOTION_FILE,
} from "../motion/incremental.ts";
import type {
  PartMotionPlan, GenMotionHint, PartMotionRecord, IncrementalMotionSidecar, MeasuredBBox,
  MeasuredSymmetryAxis, MeasuredParentContact, MotionJointKind, MotionLimitHint, WorldAxis,
} from "../motion/incremental.ts";
import {
  createIncrementalAssemblySidecar, saveIncrementalAssemblySidecar,
  sanitizeAssemblyDecl, parseAssemblyHint, ASSEMBLY_INCREMENTAL_FILE,
} from "../motion/assembly.ts";
import type {
  AssemblyInterfacePlan, AssemblyGenHint, AssemblyInterfaceRecord,
  IncrementalAssemblySidecar, MeasuredMate, InterfaceKind, FitClass, FastenKind,
} from "../motion/assembly.ts";
import { renderPartsColorViews } from "../render/parts_color.ts";
import { isKnownView, type ViewName } from "../render/views.ts";
import { createNoopSandbox } from "../sandbox/noop.ts";
import { createFileTrajectoryWriter } from "../trajectory/writer.ts";
import { DEFAULT_SCAD_MODEL, DEFAULT_IMAGE_MODEL } from "./draft.ts";

const PROCEDURA_ROOT = resolve(dirname(fileURLToPath(new URL(import.meta.url))), "..", "..");
const IMAGE_PROMPT_PATH     = join(PROCEDURA_ROOT, "prompts", "image_prompt.md");
const PLAN_SYSTEM_PATH      = join(PROCEDURA_ROOT, "prompts", "plan_system.md");
const PLAN_REVIEW_SYSTEM_PATH = join(PROCEDURA_ROOT, "prompts", "plan_review_system.md");
// NO-PLAN ablation only: the per-step "name the next part, or say done" prompt
// that stands in for the whole plan stage. Same rules as plan_system.md (what a
// part is, ordering/connectivity, granularity, mirrors, naming, sides) applied
// one step at a time — what the ablation removes is the LOOKAHEAD, not the
// modelling conventions.
const PLAN_NEXT_SYSTEM_PATH = join(PROCEDURA_ROOT, "prompts", "plan_next_system.md");
const SCAD_PART_SYSTEM_PATH = join(PROCEDURA_ROOT, "prompts", "scad_part_system.md");
// Appended to plan/plan-review system prompts ONLY in motion-aware mode; with
// the flag off the prompts stay byte-identical to the non-motion pipeline.
const PLAN_MOTION_ADDENDUM_PATH = join(PROCEDURA_ROOT, "prompts", "plan_motion_addendum.md");

// Assembly-aware mode (Slice 1): a mating-feature addendum appended to each
// part's gen USER text, and a helper library inlined into the build seed. Both
// apply ONLY when assemblyAware is on — with the flag off, prompts and the seed
// are byte-identical to the baseline pipeline.
const SCAD_PART_ASSEMBLY_ADDENDUM_PATH = join(PROCEDURA_ROOT, "prompts", "scad_part_assembly_addendum.md");
const ASSEMBLY_LIB_PATH = join(PROCEDURA_ROOT, "lib", "assembly.scad");
// Slice 2: appended to the plan/plan-review system prompts in assembly-aware
// mode so the planner declares a per-part `assembly` interface object (which
// earlier part each part mates to + mate kind + fit). Byte-identical when off.
const PLAN_ASSEMBLY_ADDENDUM_PATH = join(PROCEDURA_ROOT, "prompts", "plan_assembly_addendum.md");

// Plan-review note appended (assembly-aware only) so the reviewer preserves and
// sharpens `assembly` fields rather than dropping them.
const PLAN_REVIEW_ASSEMBLY_NOTE =
  "Assembly fields in review: preserve and sharpen each part's `assembly` interface; " +
  "never merge parts to satisfy a mate, and keep `partner` an earlier part.\n" +
  "If unsure of an interface detail, omit that field rather than guess.";

// Two extra reviewer lines appended (after the schema addendum) to the
// plan-review system prompt in motion-aware mode only.
const PLAN_REVIEW_MOTION_NOTE =
  "Motion fields in review: preserve and sharpen `motion` fields; never merge separate moving parts.\n" +
  "If unsure of a joint detail, omit that field rather than guess.";

export const GEN_MAX_ATTEMPTS = 3;          // SCAD-gen attempts (incl. compile-fix) per part
export const PLAN_MAX_ATTEMPTS = 2;         // plan-call attempts (retry on unparseable JSON)
// Plan-review loop: after the initial plan, a critic reviews it against the
// reference + text. ADD-AND-SHARPEN ONLY (enforced by mergeReviewedPlan): it
// may add missing parts and sharpen descriptions, never merge/remove/rename/
// reorder — a benchmark showed consolidation directly deletes detail (each
// part is one generation call) and full rewrites churn the planner's
// attachment story. Nearly all value lands in iteration 1 (missing parts,
// wrong counts) while extra rounds just inflate the plan with micro-detail,
// so the default is a single review pass. 0 disables. Env-overridable.
export const DEFAULT_PLAN_REVIEW_ITERS = Number(process.env["PROCEDURA_PLAN_REVIEW_ITERS"] ?? "1");
export const DEFAULT_PART_REFINE_STEPS = 4; // edit cycles per part's focused refine
// Plan-length cap. 0 = UNLIMITED (the default): the plan is however many parts
// the planner returns. Set PROCEDURA_MAX_PARTS to a positive number to re-impose a
// cap (a safety valve against a pathological planner response).
export const DEFAULT_MAX_PARTS = Number(process.env["PROCEDURA_MAX_PARTS"] ?? "0");

// ── NO-PLAN ablation ────────────────────────────────────────────────────────
// With `noPlan` there is no upfront decomposition and no plan review: before
// each part, the model sees the reference, the build so far (same context
// renders the generator gets) and the parts already built, and returns EITHER
// the single next part OR done. Everything downstream of that decision — the
// per-part gen call, compile-fix retries, the connectivity/assembly/motion
// gates, the sidecars, Phase-2 refine — is unchanged, so a difference between
// this and the normal pipeline is attributable to the plan stage alone.
export const NEXT_PART_MAX_ATTEMPTS = 2;    // next-part call attempts (retry on unparseable JSON)
// Wedge breaker, NOT a part budget: a planner returns a finite list, but a
// greedy loop that never says done would run until the process is killed. Set
// far above any observed plan (the campaign's plans run 20–35 parts), and every
// stop by this cap is logged as one — a run that ends here is not a run that
// decided it was finished.
export const NOPLAN_HARD_CAP = Number(process.env["PROCEDURA_NOPLAN_HARD_CAP"] ?? "80");

// Context-render views attached to each part's generation call: the 6 ortho
// faces plus the hero isometric. This render runs once per part over a module
// set that grows with every part, so its cost is the steepest per-view line in
// a draft run — and the 6 iso corners this used to also carry were largely
// redundant with the faces for the one thing the generator needs them for,
// which is judging where the next part goes. Listed explicitly rather than
// sliced off the catalog so reordering the catalog cannot silently change the
// set. Well under the provider's 16-image cap, which the reference image and
// these views share.
/**
 * `$fn` for the per-part context render's throwaway meshes.
 *
 * These images are a 512px placement aid — the generator reads them to decide
 * where the next part goes — so the scaffold's `$fn = 128` buys nothing here and
 * costs a lot. Measured on one module of a real 26-part model: 2.34s / 303k tris
 * / 14.4 MB at 128, against 0.54s / 71k tris / 3.4 MB at 48. That is 4.3x off
 * the split AND a much smaller mesh for Blender to import, which is the other
 * half of the context render's cost.
 *
 * The SHIPPED model is untouched — it still builds at the scaffold's `$fn`.
 * Set PROCEDURA_CONTEXT_RENDER_FN=0 to disable the override entirely.
 */
const CONTEXT_RENDER_FN = (() => {
  const raw = process.env["PROCEDURA_CONTEXT_RENDER_FN"];
  if (raw === undefined) return 48;
  const n = Number(raw);
  return Number.isFinite(n) && n > 0 ? n : 0;
})();

/**
 * Local scratch for the context render's bulk STL intermediates.
 *
 * The output dir is often a shared network mount (here: 4.6 MB/s write,
 * 2.4 MB/s read, against 1158/7936 MB/s local). A 26-part run pushes 3.3 GB of
 * per-part STL through the split, which cost 12.4 min of write inside the split
 * stage and 23.3 min of read inside the Blender stage — ~36 of 52.7 compute
 * minutes spent moving bytes, not computing. Locally the same bytes are 3.6s.
 *
 * Only the throwaway meshes move; the PNGs and legend still land in the output
 * dir. Set PROCEDURA_PARTS_SCRATCH to choose the root, or to "" to keep the old
 * behaviour of writing them beside the renders.
 */
function contextPartsScratch(sessionId: string): string | undefined {
  const root = process.env["PROCEDURA_PARTS_SCRATCH"];
  if (root === "") return undefined;            // explicit opt-out
  return join(root || tmpdir(), "procedura-parts", sessionId);
}

/**
 * Views of the build-so-far shown to the generator before it authors the next
 * part: the six ortho faces plus the hero isometric.
 *
 * OFF by default. Each part costs a Blender pass to render, which dominates the
 * wall-clock of a long build, and the plan text plus the SCAD-so-far buffer
 * already tell the generator what exists. Turn it on with `--3d-feedback` when
 * you want the generator looking at the geometry rather than reading about it.
 *
 * PROCEDURA_CONTEXT_VIEWS overrides the set either way — a comma-separated view
 * list, or the empty string for none at all — so an ablation can pin the exact
 * views without touching the flag.
 */
const DEFAULT_CONTEXT_VIEWS: readonly ViewName[] =
  ["front", "back", "left", "right", "top", "bottom", "isometric"] as const;

function resolveContextViews(enabled: boolean): readonly ViewName[] {
  const raw = process.env["PROCEDURA_CONTEXT_VIEWS"];
  if (raw !== undefined) {
    return raw.split(",").map((v) => v.trim()).filter(Boolean)
      .filter((v): v is ViewName => isKnownView(v));
  }
  return enabled ? DEFAULT_CONTEXT_VIEWS : [];
}

// Scaffold markers the builder splices against. They are comments, so they are
// inert in OpenSCAD and survive the focused refine's edit_module calls.
const MARK_PREFIX = "// <<PROCEDURA:";
const MARK_PARAMS = "// <<PROCEDURA:PARAMS>>";
const MARK_MODULES = "// <<PROCEDURA:MODULES>>";
const MARK_PLACE = "// <<PROCEDURA:PLACEMENTS>>";

/**
 * Facet policy for every generated model.
 *
 * `$fn` OVERRIDES `$fa`/`$fs`, so a global `$fn = 128` — which this seed used to
 * set — silently forces a 128-gon onto every cylinder and sphere that no module
 * happens to override. That only stayed affordable because models overrode it
 * out of habit: gemini sets a local $fn on 90-96% of its round primitives on a
 * graduated 6/8/12/16/24/32/48/64 ladder, so the global rarely applied.
 *
 * GPT-5.6 sets it on 0% of 450 primitives, and produced 4,863,388 triangles
 * that took 21.9s to compile. The pipeline's whole tessellation economy rested
 * on an unrequested behaviour that one model family has and another does not.
 *
 * `$fn = 0` restores OpenSCAD's own size-adaptive rule —
 * `fragments = min(360/$fa, r*2*PI/$fs)`, floor 5 — so a bolt head gets few
 * facets and a wheel gets many, with no cooperation from the model. Measured on
 * the same two drafts:
 *
 *   gpt    (0% local $fn):  4,863,388 -> 481,722 tris,  21.9s -> 2.0s
 *   gemini (96% local $fn):   366,048 -> 271,100 tris,   1.5s -> 1.3s
 *
 * It corrects the model that does not self-manage and barely touches the one
 * that does, because a local `$fn` still wins where a module sets one
 * deliberately. AO renders of the gpt model at both settings are visually
 * indistinguishable.
 *
 * $fa=6 caps a large curve at 60 facets, which reads as smooth; $fs=1 stops
 * small features from being subdivided for nothing.
 */
export const SEED_FACET_POLICY = "$fn = 0;\n$fa = 6;\n$fs = 1;";

export const SEED_SCAD =
  SEED_FACET_POLICY + "\n\n" +
  "// ===== parameters =====\n" + MARK_PARAMS + "\n\n" +
  "// ===== modules =====\n" + MARK_MODULES + "\n\n" +
  "// ===== assembly =====\nunion() {\n  " + MARK_PLACE + "\n}\n";

/** Inline the assembly helper library into the modules region of the seed
 *  (assembly-aware mode only). The library is pure function/module defs (no
 *  top-level globals), so it never pollutes PARAMS dedup, and it sits ABOVE the
 *  MODULES marker so per-part modules splice after it. Inlining (vs. `use <>`)
 *  keeps draft.scad and every derived/isolated compile self-contained — no
 *  external-path resolution across temp dirs. Splice dedups helper modules by
 *  name against the whole buffer, so a part re-defining `asm_*` is dropped in
 *  favour of the shared library version. Idempotent-safe: only called once at
 *  seed init. Returns the seed unchanged if the anchor is missing. */
export function injectAssemblyLib(seed: string, libText: string): string {
  const anchor = "// ===== modules =====\n";
  const idx = seed.indexOf(anchor);
  if (idx < 0) return seed;
  const at = idx + anchor.length;
  return seed.slice(0, at) + libText.replace(/\s+$/, "") + "\n\n" + seed.slice(at);
}

// ──────────────────────────────────────────────────────────────────────────
// Public types
// ──────────────────────────────────────────────────────────────────────────

export interface PartPlanItem {
  name: string;
  level?: string;
  description: string;
  /** Declared articulation intent (motion-aware mode; categorical only). */
  motion?: PartMotionPlan;
  /** Declared static mating interface (assembly-aware mode; categorical only):
   *  which earlier part this one mates to and how. */
  assembly?: AssemblyInterfacePlan;
}

export interface IncrementalPartResult {
  name: string;
  /** Module name actually placed (may be renamed on collision). */
  placedName?: string;
  generated: boolean;
  refined: boolean;
  refineVerdict?: string;
  genAttempts: number;
  error?: string;
  /** Per-part connectivity gate (delta-aware), set when the part is committed.
   *  `false` ⇒ the part shipped as a new visible floater after exhausting
   *  retries (accepted-with-warning rather than dropped). */
  connected?: boolean;
  /** visibleFloaterCount of the whole build-so-far after committing this part. */
  floatersAfter?: number;
  /** Human-readable offender list when the part shipped detached. */
  connectivityNote?: string;
  /** Motion-aware mode: declared/measured articulation summary for the part. */
  motion?: {
    moving: boolean;
    measuredAxis?: string | null;
    axisAgrees?: boolean | null;
    /** Set when the opt-in motion gate still failed on the last attempt. */
    note?: string;
  };
  /** Assembly-aware mode: declared/measured mate summary for the part. */
  assembly?: {
    mate: InterfaceKind | null;
    partnerResolved: string | null;
    contactAreaFrac?: number;
    interpenetrationFrac?: number;
    /** True ⇒ measured touch without gross interpenetration; null ⇒ unmeasured. */
    mates: boolean | null;
  };
}

export interface PartGateDecision {
  /** Discard this candidate and re-prompt (it floated and attempts remain). */
  retry: boolean;
  /** Commit the part (true whenever `retry` is false). */
  commit: boolean;
  /** Connectivity verdict to stamp on a committed part. */
  connected: boolean;
  /** Visible-floater count to carry forward as the next part's baseline. */
  floatersAfter: number;
  /** New visible floaters this part introduced (0 when none). */
  newFloaters: number;
}

/**
 * Delta-aware per-part connectivity gate decision (pure; unit-tested).
 *
 * A part is rejected only if it *increases* the build's visible-floater count,
 * so a legitimately late-connecting part is never blocked. Part 0 is exempt
 * (nothing to attach to yet). A missing analysis (`visibleFloatersNow == null`)
 * is treated as "no new floater" — we never block on analysis failure. When a
 * floater is introduced but no attempts remain, the part is committed
 * with-warning rather than dropped (completion matters more than a transient
 * floater the whole-model refine can still fix).
 */
export function decidePartGate(args: {
  partIndex: number;
  visibleFloatersNow: number | null;
  floatersBefore: number;
  attempt: number;
  maxAttempts: number;
}): PartGateDecision {
  const { partIndex, visibleFloatersNow, floatersBefore, attempt, maxAttempts } = args;
  const floatersNow = visibleFloatersNow ?? floatersBefore;
  const introducedFloater =
    partIndex > 0 && visibleFloatersNow != null && floatersNow > floatersBefore;
  const newFloaters = introducedFloater ? floatersNow - floatersBefore : 0;
  if (introducedFloater && attempt < maxAttempts) {
    return { retry: true, commit: false, connected: false, floatersAfter: floatersBefore, newFloaters };
  }
  return { retry: false, commit: true, connected: !introducedFloater, floatersAfter: floatersNow, newFloaters };
}

export interface IncrementalDraftOpts {
  text: string;
  outputDir: string;
  /** Model for the plan + per-part generation calls. Default = scad model. */
  scadModel?: string;
  imageModel?: string;
  /** When set, skip image-gen and use this image file as the reference (copied
   * to <outputDir>/image.png). Mirrors runDraft's `inputImage`. */
  inputImage?: string;
  /**
   * Text-only mode: generate from the prompt with NO reference image at all.
   *
   * The pipeline's normal path renders a reference from the text and then
   * reconstructs it, so the image — not the sentence — is what every stage is
   * actually matching. This removes that step: no image is generated, none is
   * attached to the plan or the per-part calls, and the text spec becomes the
   * target those prompts name.
   *
   * Downstream stages follow automatically: refine and paint detect the absent
   * image.png and switch to the text. `extraRefs` is meaningless here and is
   * forced to 0.
   */
  textOnly?: boolean;
  /** Render the build-so-far and show it to the generator before each part.
   *  Default false — see resolveContextViews. */
  contextRenders?: boolean;
  /** Multi-ref: generate this many EXTRA text-to-image reference views (from the
   * case text, each a complementary viewpoint) on top of the primary image, and
   * attach ALL of them to the plan / plan-review / per-part generation calls.
   * 0 (default) = single reference. */
  extraRefs?: number;
  /** Persist the binary STL alongside the OBJ. Default false — the OBJ
   * deliverable is normalized; the STL stays in the internal build dir. */
  exportStl?: boolean;
  /** Resume a killed incremental draft: reuse the existing plan.json + the parts
   * already committed to draft.scad, skip re-planning/review, and continue
   * generating from the first uncommitted part. Off (default) = fresh draft. */
  resume?: boolean;
  /** Cap on plan length. 0 = unlimited (the default); a positive value caps it. */
  maxParts?: number;
  /** Plan-review/refine iterations after the initial plan (default 10; 0 = off). */
  planReviewIters?: number;
  /** NO-PLAN ablation: skip the plan call AND the plan review entirely and pick
   * the next part one step at a time (the model decides, and declares when the
   * object is done). Everything else in the build is unchanged. Default false.
   * Env: PROCEDURA_NO_PLAN=1. */
  noPlan?: boolean;
  /** Motion-aware incremental mode: the plan declares per-part articulation
   * intent (categorical only), moving parts get shaping guidance in their gen
   * USER text, and each committed moving part is mesh-measured into the
   * motion_incremental.json sidecar for Phase 4. Default false — with the flag
   * off, prompts and behavior are byte-identical to the non-motion pipeline.
   * Env kill switch: PROCEDURA_INCREMENTAL_MOTION=0. */
  motionAware?: boolean;
  /** Assembly-aware incremental mode (Slice 1): inline an FDM-tuned mating-
   * feature helper library (lib/assembly.scad) into the build seed and append a
   * mating addendum to each part's gen USER text, so parts join through real
   * interfaces (shared-nominal pegs/sockets, bolt patterns, snaps, tabs) rather
   * than bare overlap. No new LLM calls. Default false — with the flag off,
   * prompts and the seed are byte-identical to the baseline pipeline.
   * Env kill switch: PROCEDURA_INCREMENTAL_ASSEMBLY=0. */
  assemblyAware?: boolean;
  log?: (line: string) => void;
  trajectorySink?: (event: import("@harness/template/trajectory").TrajectoryEvent) => void | Promise<void>;
  trajectoryPathOverride?: string;
  signal?: AbortSignal;
}

export interface IncrementalDraftResult {
  ok: boolean;
  outputDir: string;
  imagePath: string;
  scadPath: string;
  stlPath?: string;
  objPath?: string;
  textPath: string;
  plan: PartPlanItem[];
  parts: IncrementalPartResult[];
  partsGenerated: number;
  durationMs: number;
  trajectoryPath: string;
  sessionId: string;
  /** Present when the build never produced a compilable part. */
  compileError?: string;
}

// ──────────────────────────────────────────────────────────────────────────
// Small utilities
// ──────────────────────────────────────────────────────────────────────────

function buildImagePrompt(text: string): string {
  return readFileSync(IMAGE_PROMPT_PATH, "utf8").replace(/\{text\}/g, text);
}

// Multi-ref: viewpoint phrases that REPLACE the canonical "isometric three-
// quarter" opening of the image prompt so each extra reference renders the
// object from a complementary angle. Used for the first `extraRefs` images.
const EXTRA_REF_VIEWS = [
  "A direct head-on FRONT elevation product render",
  "A direct SIDE-PROFILE product render",
  "A REAR three-quarter product render",
  "A TOP-DOWN plan-view product render",
];

/** Build an extra-reference image prompt: same styling/structure as the
 *  canonical reference prompt, but rendered from `viewPhrase` instead of the
 *  default three-quarter view. Falls back to appending a directive if the
 *  canonical opening clause isn't present (prompt file edited). */
function buildExtraRefPrompt(text: string, viewPhrase: string): string {
  const base = readFileSync(IMAGE_PROMPT_PATH, "utf8");
  const opening = /^An isometric three-quarter product render of/;
  const swapped = opening.test(base)
    ? base.replace(opening, `${viewPhrase} of`)
    : base + `\n\nRender from this viewpoint instead: ${viewPhrase}.`;
  return swapped.replace(/\{text\}/g, text);
}
function fileSize(p: string): number {
  return existsSync(p) ? statSync(p).size : 0;
}
function nextId(prefix: string): string {
  return `${prefix}_${Date.now().toString(36)}${Math.random().toString(36).slice(2, 8)}`;
}
function pad2(n: number): string {
  return n.toString().padStart(2, "0");
}

// ──────────────────────────────────────────────────────────────────────────
// Marker scaffold: insert / ensure / strip
// ──────────────────────────────────────────────────────────────────────────

/** Insert `text` immediately before the line carrying `marker`, indented to
 *  match the marker's own indentation. */
function insertBeforeMarker(scad: string, marker: string, text: string): string {
  const idx = scad.indexOf(marker);
  if (idx < 0) throw new Error(`scaffold marker missing: ${marker}`);
  const lineStart = scad.lastIndexOf("\n", idx) + 1;
  const indent = scad.slice(lineStart, idx);
  const block = text.replace(/\s+$/, "")
    .split("\n")
    .map((l) => (l.length ? indent + l : l))
    .join("\n");
  return scad.slice(0, lineStart) + block + "\n" + scad.slice(lineStart);
}

/** Restore all three markers from structure if any are missing (e.g. after an
 *  edit_full). Idempotent: returns the input unchanged when all are present. */
export function ensureMarkers(scad: string): string {
  if (scad.includes(MARK_PARAMS) && scad.includes(MARK_MODULES) && scad.includes(MARK_PLACE)) {
    return scad;
  }
  // Drop any stale/partial markers, then rebuild from the parsed structure.
  let s = scad.split("\n").filter((l) => !l.includes(MARK_PREFIX)).join("\n");
  const spans = findModuleSpans(s);
  let assemblyStart: number;
  if (spans.length) {
    assemblyStart = spans[spans.length - 1]!.end;
  } else {
    const u = s.search(/\bunion\s*\(/);
    assemblyStart = u >= 0 ? u : s.length;
  }
  const firstModuleStart = spans.length ? spans[0]!.start : assemblyStart;
  const params = s.slice(0, firstModuleStart).replace(/\s+$/, "");
  const modules = s.slice(firstModuleStart, assemblyStart).replace(/\s+$/, "");
  let assembly = s.slice(assemblyStart);
  const lb = assembly.lastIndexOf("}");
  assembly = lb >= 0
    ? assembly.slice(0, lb) + "  " + MARK_PLACE + "\n" + assembly.slice(lb)
    : assembly.replace(/\s+$/, "") + "\n" + MARK_PLACE + "\n";
  return (
    params + "\n\n" + MARK_PARAMS + "\n\n" +
    (modules ? modules + "\n\n" : "") + MARK_MODULES + "\n\n" +
    assembly.replace(/^\n+/, "")
  );
}

export function stripMarkers(scad: string): string {
  return scad
    .split("\n")
    .filter((l) => !l.includes(MARK_PREFIX))
    .join("\n")
    .replace(/\n{3,}/g, "\n\n");
}

// ──────────────────────────────────────────────────────────────────────────
// Plan parsing
// ──────────────────────────────────────────────────────────────────────────

export function parsePlanJson(
  rawText: string, maxParts: number, opts?: { motion?: boolean; assembly?: boolean },
): PartPlanItem[] {
  let t = rawText.trim();
  const fence = /```(?:json)?\s*([\s\S]*?)```/.exec(t);
  if (fence) t = fence[1]!.trim();
  const start = t.indexOf("[");
  const end = t.lastIndexOf("]");
  if (start < 0 || end < 0 || end <= start) {
    throw new Error("no JSON array found in plan response");
  }
  const arr = JSON.parse(t.slice(start, end + 1)) as unknown;
  if (!Array.isArray(arr)) throw new Error("plan response is not a JSON array");

  const seen = new Set<string>();
  const out: PartPlanItem[] = [];
  for (const raw of arr) {
    if (!raw || typeof raw !== "object") continue;
    const o = raw as Record<string, unknown>;
    const rawName = typeof o["name"] === "string" ? (o["name"] as string) : "";
    const description = typeof o["description"] === "string" ? (o["description"] as string) : "";
    if (!rawName || !description) continue;
    // Normalize to the lowercase snake_case the prompts mandate
    // (^[a-z_][a-z0-9_]*$) — models occasionally return Title Case.
    let name = sanitizeIdentifier(rawName).toLowerCase();
    if (seen.has(name)) {
      let k = 2;
      while (seen.has(`${name}_${k}`)) k++;
      name = `${name}_${k}`;
    }
    seen.add(name);
    const item: PartPlanItem = {
      name,
      description,
      ...(typeof o["level"] === "string" ? { level: o["level"] as string } : {}),
    };
    // Accepted only in motion-aware mode so a flag-off run's plan.json stays
    // schema-identical to today; untrusted enums are dropped by the sanitizer.
    if (opts?.motion === true) {
      const motion = sanitizeMotionDecl(o["motion"]);
      if (motion) item.motion = motion;
    }
    // Accepted only in assembly-aware mode (schema-identical when off).
    if (opts?.assembly === true) {
      const assembly = sanitizeAssemblyDecl(o["assembly"]);
      if (assembly) item.assembly = assembly;
    }
    out.push(item);
    if (maxParts > 0 && out.length >= maxParts) break;   // 0 = unlimited
  }
  if (out.length === 0) throw new Error("plan parsed to zero valid parts");
  return out;
}

/** One step of the NO-PLAN ablation: either the next part, or "the object is
 *  finished". Lenient about the wrapper (fence, prose around the JSON, a bare
 *  part object with no `done` field, or a one-element array) and strict about
 *  the part itself — it goes through the same normalization/sanitization as a
 *  planned part, so a no-plan part is schema-identical to a planned one.
 *  Throws when nothing usable is in the reply; the caller retries. */
export function parseNextPartJson(
  rawText: string, opts?: { motion?: boolean; assembly?: boolean },
): { done: true; reason: string } | { done: false; part: PartPlanItem } {
  let t = rawText.trim();
  const fence = /```(?:json)?\s*([\s\S]*?)```/.exec(t);
  if (fence) t = fence[1]!.trim();
  const start = t.indexOf("{");
  const end = t.lastIndexOf("}");
  if (start < 0 || end < 0 || end <= start) {
    throw new Error("no JSON object found in next-part response");
  }
  const obj = JSON.parse(t.slice(start, end + 1)) as unknown;
  if (!obj || typeof obj !== "object") throw new Error("next-part response is not a JSON object");
  const o = obj as Record<string, unknown>;

  // `done` is only believed when it is literally true: a model that answers
  // with the part alone (no `done` key) is adding a part, not finishing.
  if (o["done"] === true || o["done"] === "true") {
    const reason = typeof o["reason"] === "string" ? (o["reason"] as string) : "";
    return { done: true, reason };
  }
  const rawPart = o["part"] && typeof o["part"] === "object" ? o["part"] : o;
  // Reuse the plan parser for one item so naming, level, motion and assembly
  // are normalized by exactly the code that normalizes a planned part.
  const parsed = parsePlanJson(JSON.stringify([rawPart]), 1, opts);
  return { done: false, part: parsed[0]! };
}

/**
 * Earlier-parent rule for declared motion: `motion.parent` must name a part
 * that appears EARLIER in the plan (it has to exist before this part can
 * attach to it). Violating parent refs are deleted in place — the rest of the
 * motion object is kept. Returns the names of the parts whose parent was
 * dropped (for logging). Run after the initial plan parse and after every
 * review merge.
 */
export function enforceMotionParentOrder(plan: PartPlanItem[]): string[] {
  const dropped: string[] = [];
  const earlier = new Set<string>();
  for (const item of plan) {
    if (item.motion?.parent !== undefined && !earlier.has(item.motion.parent)) {
      delete item.motion.parent;
      dropped.push(item.name);
    }
    earlier.add(item.name);
  }
  return dropped;
}

/**
 * Earlier-partner rule for declared assembly interfaces: `assembly.partner`
 * must name a part appearing EARLIER in the plan (it must exist before this
 * part can mate to it). Violating partner refs are deleted in place — the rest
 * of the assembly object is kept. Returns the names whose partner was dropped.
 * Run after the initial plan parse and after every review merge (mirrors
 * enforceMotionParentOrder).
 */
export function enforceAssemblyPartnerOrder(plan: PartPlanItem[]): string[] {
  const dropped: string[] = [];
  const earlier = new Set<string>();
  for (const item of plan) {
    if (item.assembly?.partner !== undefined && !earlier.has(item.assembly.partner)) {
      delete item.assembly.partner;
      dropped.push(item.name);
    }
    earlier.add(item.name);
  }
  return dropped;
}

/**
 * Constrained merge of a reviewed plan into the original: the review loop is
 * ADD-AND-SHARPEN only, and this function is what enforces it (the prompt
 * alone is not trusted).
 *
 *   - every ORIGINAL part survives, with its original name, in the original
 *     order (a reviewer "merge"/"remove"/"rename"/"reorder" is discarded);
 *   - a kept part adopts the reviewer's sharpened description/level, and its
 *     sharpened `motion` when the reviewer returned one (an omitted motion
 *     keeps the original — the reviewer never silently deletes intent);
 *   - genuinely NEW parts are inserted where the reviewer placed them
 *     relative to the surviving parts (so "right after its attachment
 *     target" is preserved).
 *
 * Returns the merged plan plus what was accepted/rejected for logging.
 */
export function mergeReviewedPlan(
  original: PartPlanItem[], reviewed: PartPlanItem[],
): { plan: PartPlanItem[]; added: string[]; sharpened: string[]; rejectedDrops: string[] } {
  const origNames = new Set(original.map((p) => p.name));
  const revByName = new Map(reviewed.map((p) => [p.name, p] as const));

  const sharpened: string[] = [];
  const out: PartPlanItem[] = original.map((p) => {
    const r = revByName.get(p.name);
    if (!r) return p;
    if (r.description && r.description !== p.description) sharpened.push(p.name);
    return {
      ...p,
      description: r.description || p.description,
      ...(r.level ? { level: r.level } : {}),
      ...(r.motion ? { motion: r.motion } : {}),
      ...(r.assembly ? { assembly: r.assembly } : {}),
    };
  });

  const rejectedDrops = original.filter((p) => !revByName.has(p.name)).map((p) => p.name);

  // Insert NEW parts following the reviewer's relative placement: walk the
  // reviewer's order, tracking a cursor in `out`; each new part goes right
  // after the last part we anchored on (consecutive additions stay in order).
  // EXCEPT when the reviewer also tried to DROP originals: then its additions
  // are usually the merged replacements (e.g. "arms" for left_arm+right_arm),
  // and accepting them alongside the restored originals would duplicate
  // geometry. Reject the whole round's additions; the next iteration reviews
  // the enforced plan and can re-add anything genuinely missing.
  const added: string[] = [];
  if (rejectedDrops.length === 0) {
    let cursor = -1;
    for (const r of reviewed) {
      if (origNames.has(r.name)) {
        cursor = out.findIndex((p) => p.name === r.name);
        continue;
      }
      if (out.some((p) => p.name === r.name)) continue; // duplicate add
      out.splice(cursor + 1, 0, r);
      cursor += 1;
      added.push(r.name);
    }
  }

  return { plan: out, added, sharpened, rejectedDrops };
}

/** Compact one-line rendering of a declared motion for the plan-review list,
 *  e.g. `revolute about Y rel chassis, role=wheel` or `static`. */
function describeMotionDecl(m: PartMotionPlan): string {
  if (!m.moving) return "static";
  const bits: string[] = [m.jointType ?? "moving"];
  if (m.axis) bits.push(`${m.jointType === "prismatic" ? "along" : "about"} ${m.axis}`);
  if (m.parent) bits.push(`rel ${m.parent}`);
  const head = bits.join(" ");
  return m.role ? `${head}, role=${m.role}` : head;
}

/** Compact one-line rendering of a declared assembly interface for the
 *  plan-review list, e.g. `bolt_pattern×4 to base (location)`. */
function describeAssemblyDecl(a: AssemblyInterfacePlan): string {
  const bits: string[] = [];
  bits.push(a.mate ?? "mates");
  if (a.count && a.count > 1) bits[bits.length - 1] += `×${a.count}`;
  if (a.partner) bits.push(`to ${a.partner}`);
  if (a.fit) bits.push(`(${a.fit})`);
  if (a.fasten && a.fasten !== "none") bits.push(`+${a.fasten}`);
  return bits.join(" ");
}

export interface PlanReview {
  /** True when the reviewer says the plan is ready to build as-is. */
  ok: boolean;
  /** Short critique / summary of what was changed. */
  notes: string;
  /** The reviewer's (possibly corrected) full plan, or null if it didn't
   *  return a usable one (then the caller keeps the current plan). */
  plan: PartPlanItem[] | null;
}

/** Parse the plan-reviewer response: a JSON object {ok, notes, plan}. Lenient
 *  about how "ok" is expressed and tolerant of a code fence / surrounding prose.
 *  Returns null if no JSON object is found at all. */
export function parsePlanReview(
  rawText: string, maxParts: number, opts?: { motion?: boolean; assembly?: boolean },
): PlanReview | null {
  let t = rawText.trim();
  const fence = /```(?:json)?\s*([\s\S]*?)```/.exec(t);
  if (fence) t = fence[1]!.trim();
  const objM = /\{[\s\S]*\}/.exec(t);
  if (!objM) return null;
  let obj: Record<string, unknown>;
  try { obj = JSON.parse(objM[0]) as Record<string, unknown>; }
  catch { return null; }

  const okRaw = obj["ok"] ?? obj["approved"] ?? obj["verdict"] ?? obj["ready"] ?? obj["start"];
  const ok = okRaw === true ||
    /^(ok|yes|true|good|approved|ready|start|ship)$/i.test(String(okRaw ?? "").trim());
  const notes = typeof obj["notes"] === "string" ? (obj["notes"] as string)
    : typeof obj["critique"] === "string" ? (obj["critique"] as string)
    : "";
  let plan: PartPlanItem[] | null = null;
  if (Array.isArray(obj["plan"]) && (obj["plan"] as unknown[]).length > 0) {
    try { plan = parsePlanJson(JSON.stringify(obj["plan"]), maxParts, opts); }
    catch { plan = null; }
  }
  return { ok, notes, plan };
}

// ──────────────────────────────────────────────────────────────────────────
// Per-part response parsing
// ──────────────────────────────────────────────────────────────────────────

export interface ParsedPart {
  params: string;        // newline-joined NEW param assignment lines ("" if none)
  helpers: string[];     // NEW helper module definitions
  partModule: string;    // the part module's full definition
  partName: string;      // module name parsed from partModule
  placement: string;     // assembly placement statement(s)
  /** Optional `// MOTION` block hint (categorical only; never spliced). */
  motion?: GenMotionHint;
  /** Optional `// INTERFACE` block hint (categorical only; never spliced). */
  interface?: AssemblyGenHint;
}

/** Split a code block into the labelled sections, if the headers exist.
 *  `allowMotion` (motion-aware mode only) additionally buckets an optional
 *  `// MOTION` trailer, which keeps its content out of PLACE (and thus out of
 *  the SCAD buffer). The MOTION header is matched STRICTLY — case-sensitive,
 *  nothing but whitespace after the word — so a prose comment like
 *  `// Motion: rotates with the turret` is never mistaken for a header, and
 *  it NEVER counts toward `found`: the header-path/fallback decision is made
 *  on the original four headers exactly as before. */
function splitSections(
  block: string,
  allowMotion: boolean,
  allowAssembly = false,
): { params?: string; helpers?: string; part?: string; place?: string; motion?: string; interface?: string; found: number } {
  const lines = block.split("\n");
  const headerRe = /^\s*\/\/\s*(PARAMS|HELPERS|PART|PLACE)\b/i;
  const motionHeaderRe = /^\s*\/\/\s*MOTION\s*$/;
  const interfaceHeaderRe = /^\s*\/\/\s*INTERFACE\s*$/;
  const buckets: Record<string, string[]> = {};
  let current: string | null = null;
  let found = 0;
  for (const line of lines) {
    const m = headerRe.exec(line);
    if (m) {
      current = m[1]!.toUpperCase();
      buckets[current] = buckets[current] ?? [];
      found += 1;
      continue;
    }
    if (allowMotion && motionHeaderRe.test(line)) {
      current = "MOTION";
      buckets[current] = buckets[current] ?? [];
      continue; // metadata trailer — excluded from the header count
    }
    if (allowAssembly && interfaceHeaderRe.test(line)) {
      current = "INTERFACE";
      buckets[current] = buckets[current] ?? [];
      continue; // metadata trailer — excluded from the header count
    }
    if (current) buckets[current]!.push(line);
  }
  return {
    ...(buckets["PARAMS"] ? { params: buckets["PARAMS"].join("\n") } : {}),
    ...(buckets["HELPERS"] ? { helpers: buckets["HELPERS"].join("\n") } : {}),
    ...(buckets["PART"] ? { part: buckets["PART"].join("\n") } : {}),
    ...(buckets["PLACE"] ? { place: buckets["PLACE"].join("\n") } : {}),
    ...(buckets["MOTION"] ? { motion: buckets["MOTION"].join("\n") } : {}),
    ...(buckets["INTERFACE"] ? { interface: buckets["INTERFACE"].join("\n") } : {}),
    found,
  };
}

/** Parse the optional `// MOTION` block: comment-prefixed one-object JSON with
 *  categorical fields only (sanitized like the plan's motion). Any failure
 *  yields undefined — a malformed block never fails the part. */
function parseMotionBlock(text: string | undefined): GenMotionHint | undefined {
  if (!text) return undefined;
  try {
    const joined = text.split("\n").map((l) => l.replace(/^\s*\/\/ ?/, "")).join("\n");
    const jsonText = extractBalancedJson(joined);
    if (jsonText === null) return undefined;
    const raw = JSON.parse(jsonText) as Record<string, unknown>;
    const decl = sanitizeMotionDecl(raw);
    const hint: GenMotionHint = {};
    // `moving` is kept only when explicitly stated (an absent field must not
    // read as a flip of the plan's declaration).
    const moving = raw["moving"];
    if (typeof moving === "boolean") hint.moving = moving;
    if (decl?.jointType) hint.jointType = decl.jointType;
    if (decl?.parent) hint.parent = decl.parent;
    if (decl?.axis) hint.axis = decl.axis;
    if (decl?.role) hint.role = decl.role;
    if (decl?.limitHint) hint.limitHint = decl.limitHint;
    const anchorHint = raw["anchorHint"];
    if (typeof anchorHint === "string" && anchorHint.trim()) {
      hint.anchorHint = anchorHint.trim().slice(0, 120);
    }
    return Object.keys(hint).length > 0 ? hint : undefined;
  } catch {
    return undefined;
  }
}

/** Parse the optional `// INTERFACE` block: comment-prefixed one-object JSON
 *  with categorical fields only. Any failure yields undefined — a malformed
 *  block never fails the part (mirrors parseMotionBlock). */
function parseAssemblyBlock(text: string | undefined): AssemblyGenHint | undefined {
  if (!text) return undefined;
  try {
    const joined = text.split("\n").map((l) => l.replace(/^\s*\/\/ ?/, "")).join("\n");
    const jsonText = extractBalancedJson(joined);
    if (jsonText === null) return undefined;
    const raw = JSON.parse(jsonText) as Record<string, unknown>;
    return parseAssemblyHint(raw);
  } catch {
    return undefined;
  }
}

/** Every top-level `function ... ;` definition in `code`, as source text. */
function functionDefs(code: string): string[] {
  return findFunctionSpans(code).map((s) => code.slice(s.start, s.end));
}

/** Blank out spans, preserving length and line breaks so indices measured on
 *  the original string stay valid. */
function blankSpans(code: string, spans: { start: number; end: number }[]): string {
  if (spans.length === 0) return code;
  const chars = code.split(""); // UTF-16 units, so span indices line up
  for (const sp of spans) {
    for (let i = sp.start; i < sp.end && i < chars.length; i++) {
      if (chars[i] !== "\n") chars[i] = " ";
    }
  }
  return chars.join("");
}

function removeModuleSpans(code: string, spans: { start: number; end: number }[]): string {
  if (spans.length === 0) return code;
  let out = "";
  let cursor = 0;
  for (const sp of spans) {
    out += code.slice(cursor, sp.start);
    cursor = sp.end;
  }
  out += code.slice(cursor);
  return out;
}

/**
 * Keep only top-level `name = ...;` assignments from a PARAMS block.
 *
 * STATEMENT-aware, not line-aware. The previous version filtered line by line,
 * so a model that wrapped a long assignment across lines —
 *
 *     rung_positions = [
 *       10, 20, 30
 *     ];
 *
 * — kept only `rung_positions = [` and dropped the continuation, splicing a
 * dangling assignment into the buffer and guaranteeing a syntax error on the
 * next compile (observed killing both attempts on a part). We now accumulate
 * from the opening `name =` to its terminating `;` at bracket depth 0, tracking
 * strings and comments so a `;` inside either does not end the statement early.
 *
 * A trailing statement with no terminating `;` is DROPPED: that only happens
 * when the response was truncated mid-assignment, and splicing the fragment in
 * would break the buffer exactly as before.
 */
function onlyParamLines(text: string): string {
  const out: string[] = [];
  const lines = text.split("\n");
  let buf: string[] = [];
  let depth = 0;
  let inBlockComment = false;

  const scan = (line: string): { closed: boolean } => {
    let i = 0;
    while (i < line.length) {
      const c = line[i]!;
      const d = line[i + 1];
      if (inBlockComment) {
        if (c === "*" && d === "/") { inBlockComment = false; i += 2; continue; }
        i++; continue;
      }
      if (c === "/" && d === "/") return { closed: false };       // rest is a comment
      if (c === "/" && d === "*") { inBlockComment = true; i += 2; continue; }
      if (c === '"') {                                            // skip string literal
        i++;
        while (i < line.length && line[i] !== '"') { if (line[i] === "\\") i++; i++; }
        i++; continue;
      }
      if (c === "[" || c === "(" || c === "{") depth++;
      else if (c === "]" || c === ")" || c === "}") depth = Math.max(0, depth - 1);
      else if (c === ";" && depth === 0) return { closed: true };
      i++;
    }
    return { closed: false };
  };

  for (const raw of lines) {
    if (buf.length === 0) {
      if (!/^\s*[A-Za-z_]\w*\s*=/.test(raw)) continue;          // not an assignment start
      buf.push(raw.trim());
      if (scan(raw).closed) { out.push(buf.join(" ").trim()); buf = []; depth = 0; }
      continue;
    }
    buf.push(raw.trim());
    if (scan(raw).closed) { out.push(buf.join(" ").trim()); buf = []; depth = 0; }
  }
  // buf non-empty here == unterminated tail from a truncated response -> drop it.
  return out.join("\n");
}

/** Why a block of SCAD cannot be a complete generation, or null if it can.
 *
 *  Valid OpenSCAD always balances its brackets, so an imbalance is conclusive
 *  evidence that the response was cut off mid-stream or is malformed — and a
 *  cut-off response is DANGEROUS to salvage, not merely useless: the structural
 *  parser skips the unterminated part module, the fallback path then picks a
 *  surviving HELPER as the part, and scrapes fragments out of the dead module
 *  body ("difference() {", "translate([", ...) into the placement. That reaches
 *  OpenSCAD as a syntax error at a line number that means nothing to the model
 *  on retry. Rejecting here costs one regeneration instead of a compile plus a
 *  regeneration, and the retry reason is the truth.
 *
 *  Measured on 6,602 real generations across six P3D-Bench runs: 14 flagged,
 *  and all 14 were already rejected downstream (8 after burning a compile) —
 *  zero false positives. */
export function truncationReason(code: string): string | null {
  const s = stripCommentsAndStrings(code);
  const opener: Record<string, string> = { ")": "(", "]": "[", "}": "{" };
  const stack: string[] = [];
  for (let i = 0; i < s.length; i++) {
    const c = s[i]!;
    if (c === "(" || c === "[" || c === "{") stack.push(c);
    else if (c === ")" || c === "]" || c === "}") {
      const top = stack.pop();
      if (top === undefined) return `stray '${c}'`;
      if (top !== opener[c]) return `'${top}' closed by '${c}'`;
    }
  }
  if (stack.length) {
    return `${stack.length} unclosed '${stack[stack.length - 1]}' at end of response`;
  }
  return null;
}

export function parsePartResponse(
  rawText: string, requestedName: string, opts?: { motion?: boolean; assembly?: boolean },
): ParsedPart | null {
  const block = extractOpenscadCode(rawText);
  if (!block || block.trim().length < 10) return null;
  if (truncationReason(block)) return null;

  // MOTION / INTERFACE are only recognized headers in their respective modes;
  // parsed here and attached on BOTH paths, but never part of the header-path
  // decision (in the fallback their lines are plain comments and drop out).
  const allowMotion = opts?.motion === true;
  const allowAssembly = opts?.assembly === true;
  const sections = splitSections(block, allowMotion, allowAssembly);
  const motion = allowMotion ? parseMotionBlock(sections.motion) : undefined;
  const iface = allowAssembly ? parseAssemblyBlock(sections.interface) : undefined;
  const hasHeaders = sections.found >= 2 && sections.part !== undefined;

  if (hasHeaders) {
    const partText = sections.part ?? "";
    const partSpans = findModuleSpans(partText);
    if (partSpans.length > 0) {
      const chosen = partSpans.find((s) => s.name === requestedName) ?? partSpans[0]!;
      const partModule = partText.slice(chosen.start, chosen.end);
      const helpers: string[] = [];
      const helperText = sections.helpers ?? "";
      for (const s of findModuleSpans(helperText)) helpers.push(helperText.slice(s.start, s.end));
      // Any extra modules accidentally placed in PART (besides the chosen one)
      // are treated as helpers too.
      for (const s of partSpans) {
        if (s !== chosen) helpers.push(partText.slice(s.start, s.end));
      }
      // ... and so are `function` definitions, wherever in the block they sit.
      // They used to be dropped on the floor (only MODULES were collected), so
      // a part built around a helper function — an extruded profile, a swept
      // radius, a gear tooth path — was spliced in with every call to it
      // dangling. OpenSCAD answers an unknown function with a WARNING and
      // `undef`, so `polygon(profile())` silently yields nothing: the part
      // still compiles, still passes the gates, and the shape it was supposed
      // to have is simply absent. Measured over six P3D-Bench runs, 41 of
      // 3,533 committed parts had come out this way, 92 definitions lost, none
      // kept. Scanning the WHOLE block (not just HELPERS) also rescues the ones
      // the model files under PARAMS or PART.
      helpers.push(...functionDefs(block));
      let placement = (sections.place ?? "").trim();
      if (!placement) placement = `${chosen.name}();`;
      // A function filed under PARAMS is collected above as a helper; blank it
      // here so its body can't also be scraped as a stray assignment.
      const paramsText = sections.params ?? "";
      return {
        params: onlyParamLines(blankSpans(paramsText, findFunctionSpans(paramsText))),
        helpers,
        partModule,
        partName: chosen.name,
        placement,
        ...(motion ? { motion } : {}),
        ...(iface ? { interface: iface } : {}),
      };
    }
  }

  // Fallback: parse the whole block structurally.
  const spans = findModuleSpans(block);
  if (spans.length === 0) return null;
  const chosen = spans.find((s) => s.name === requestedName) ?? spans[spans.length - 1]!;
  const partModule = block.slice(chosen.start, chosen.end);
  const helpers = spans.filter((s) => s !== chosen).map((s) => block.slice(s.start, s.end));
  helpers.push(...functionDefs(block));
  // Blank the function definitions out of the remainder as well as the modules:
  // their bodies span many lines, and a `translate(` or a bare `x = ...` inside
  // one would otherwise be scraped into the placement or the params.
  // Blanking is length-preserving, so it must come FIRST: `removeModuleSpans`
  // shifts every index after the span it cuts, and `spans` was measured on the
  // original block.
  const remainder = removeModuleSpans(blankSpans(block, findFunctionSpans(block)), spans);
  const paramLines: string[] = [];
  const placeLines: string[] = [];
  for (const rawLine of remainder.split("\n")) {
    const t = rawLine.trim();
    if (!t || t.startsWith("//")) continue;
    if (/^[A-Za-z_]\w*\s*=/.test(t)) paramLines.push(t);
    else if (t.includes(`${chosen.name}(`) || /\b(translate|rotate|scale|mirror|union|difference)\b/.test(t)) {
      placeLines.push(t);
    }
  }
  return {
    params: paramLines.join("\n"),
    helpers,
    partModule,
    partName: chosen.name,
    placement: placeLines.join("\n").trim() || `${chosen.name}();`,
    ...(motion ? { motion } : {}),
    ...(iface ? { interface: iface } : {}),
  };
}

// ──────────────────────────────────────────────────────────────────────────
// Splicing a parsed part into the accumulated SCAD
// ──────────────────────────────────────────────────────────────────────────

function collectParamNames(scad: string): Set<string> {
  // The params region ends at the first DEFINITION of either kind. Stopping
  // only at the first module would sweep any function definitions that precede
  // it, and a line of a function body can look like a top-level assignment.
  const spans = [...findModuleSpans(scad), ...findFunctionSpans(scad)]
    .sort((a, b) => a.start - b.start);
  const paramsRegion = spans.length ? scad.slice(0, spans[0]!.start) : scad;
  const names = new Set<string>();
  for (const m of paramsRegion.matchAll(/^\s*([A-Za-z_]\w*)\s*=/gm)) names.add(m[1]!);
  return names;
}

/** Name a helper definition declares — `module` or `function`. */
function defNameOf(def: string): string | null {
  const sp = findModuleSpans(def)[0] ?? findFunctionSpans(def)[0];
  return sp ? sp.name : null;
}

/** Splice the parsed part into `scad`. Returns the new SCAD + the module name
 *  actually placed (renamed on collision). Throws if it can't splice. */
export function spliceParsedPart(
  scad: string, parsed: ParsedPart,
): { scad: string; placedName: string } {
  let s = ensureMarkers(scad);

  const existingParams = collectParamNames(s);
  const existingModules = new Set(findModuleSpans(s).map((x) => x.name));
  // Helpers dedup against BOTH namespaces: a later part re-emitting the same
  // profile function must not redefine it, and a function name must not be
  // shadowed by a module of the same name from an earlier part.
  const existingDefs = new Set([
    ...existingModules,
    ...findFunctionSpans(s).map((x) => x.name),
  ]);

  // Rename the part if it collides with an already-defined module.
  let placedName = parsed.partName;
  let partModule = parsed.partModule;
  let placement = parsed.placement;
  if (existingModules.has(placedName)) {
    let k = 2;
    while (existingModules.has(`${placedName}_${k}`)) k++;
    const renamed = `${placedName}_${k}`;
    partModule = partModule.replace(new RegExp(`\\bmodule\\s+${placedName}\\b`), `module ${renamed}`);
    placement = placement.replace(new RegExp(`\\b${placedName}\\s*\\(`, "g"), `${renamed}(`);
    placedName = renamed;
  }

  // New params only (dedup against what's already declared).
  const newParamLines = parsed.params
    .split("\n")
    .filter((l) => {
      const m = /^\s*([A-Za-z_]\w*)\s*=/.exec(l);
      return m ? !existingParams.has(m[1]!) : false;
    });
  if (newParamLines.length) {
    s = insertBeforeMarker(s, MARK_PARAMS, newParamLines.join("\n"));
  }

  // New helper definitions only (dedup by name), then the part module. A helper
  // repeated within ONE response is also deduped: OpenSCAD would take the last
  // definition, but two identical bodies in the file is noise the refine agent
  // then has to read.
  const seen = new Set<string>();
  const newHelpers = parsed.helpers.filter((h) => {
    const n = defNameOf(h);
    if (!n || existingDefs.has(n) || seen.has(n)) return false;
    seen.add(n);
    return true;
  });
  const moduleBlock = [...newHelpers, partModule].join("\n\n");
  s = insertBeforeMarker(s, MARK_MODULES, moduleBlock);

  // The placement.
  s = insertBeforeMarker(s, MARK_PLACE, placement);

  return { scad: s, placedName };
}

// ──────────────────────────────────────────────────────────────────────────
// Motion-aware mode: declare / shape / measure / gate
// ──────────────────────────────────────────────────────────────────────────

/** Plan motion reconciled with the gen response's `// MOTION` hint. */
export interface MergedMotionDecl {
  moving: boolean;
  jointType: MotionJointKind;
  parent?: string;
  axis?: WorldAxis;
  role?: string;
  limitHint?: MotionLimitHint;
  anchorHint?: string;
  source: "plan" | "gen" | "plan+gen";
  warnings: string[];
}

/**
 * Merge the plan's declared motion with the gen response's optional `// MOTION`
 * hint. Gen fields override plan fields when present (the generator saw the
 * real geometry); a gen `moving` flip is honored but flagged. A missing
 * jointType defaults categorically (revolute when moving — the dominant
 * articulation — else fixed) with a warning. Returns null when neither side
 * declared anything.
 */
export function mergeMotionDecl(
  planMotion: PartMotionPlan | undefined,
  genHint: GenMotionHint | undefined,
): MergedMotionDecl | null {
  if (!planMotion && !genHint) return null;
  const warnings: string[] = [];
  const moving = genHint?.moving ?? planMotion?.moving ?? false;
  if (planMotion && genHint?.moving !== undefined && genHint.moving !== planMotion.moving) {
    warnings.push(`gen flipped moving: ${planMotion.moving} -> ${genHint.moving}`);
  }
  let jointType = genHint?.jointType ?? planMotion?.jointType;
  if (!jointType) {
    jointType = moving ? "revolute" : "fixed";
    if (moving) warnings.push("jointType defaulted to revolute");
  }
  const parent = genHint?.parent ?? planMotion?.parent;
  const axis = genHint?.axis ?? planMotion?.axis;
  const role = genHint?.role ?? planMotion?.role;
  const limitHint = genHint?.limitHint ?? planMotion?.limitHint;
  return {
    moving,
    jointType,
    ...(parent ? { parent } : {}),
    ...(axis ? { axis } : {}),
    ...(role ? { role } : {}),
    ...(limitHint ? { limitHint } : {}),
    ...(genHint?.anchorHint ? { anchorHint: genHint.anchorHint } : {}),
    source: planMotion && genHint ? "plan+gen" : genHint ? "gen" : "plan",
    warnings,
  };
}

/** Plan interface reconciled with the gen response's `// INTERFACE` hint. */
export interface MergedAssemblyDecl {
  partner?: string;
  mate?: InterfaceKind;
  fit?: FitClass;
  role?: string;
  count?: number;
  fasten?: FastenKind;
  locateHint?: string;
  source: "plan" | "gen" | "plan+gen";
  warnings: string[];
}

/**
 * Merge the plan's declared assembly interface with the gen response's optional
 * `// INTERFACE` hint. Gen fields override plan fields when present (the
 * generator saw the real neighbour geometry). Returns null when neither side
 * declared anything (mirrors mergeMotionDecl).
 */
export function mergeAssemblyDecl(
  planAsm: AssemblyInterfacePlan | undefined,
  genHint: AssemblyGenHint | undefined,
): MergedAssemblyDecl | null {
  if (!planAsm && !genHint) return null;
  const warnings: string[] = [];
  const partner = genHint?.partner ?? planAsm?.partner;
  if (planAsm?.partner && genHint?.partner && genHint.partner !== planAsm.partner) {
    warnings.push(`gen changed partner: ${planAsm.partner} -> ${genHint.partner}`);
  }
  const mate = genHint?.mate ?? planAsm?.mate;
  const fit = genHint?.fit ?? planAsm?.fit;
  const role = genHint?.role ?? planAsm?.role;
  const count = genHint?.count ?? planAsm?.count;
  const fasten = genHint?.fasten ?? planAsm?.fasten;
  return {
    ...(partner ? { partner } : {}),
    ...(mate ? { mate } : {}),
    ...(fit ? { fit } : {}),
    ...(role ? { role } : {}),
    ...(count ? { count } : {}),
    ...(fasten ? { fasten } : {}),
    ...(genHint?.locateHint ? { locateHint: genHint.locateHint } : {}),
    source: planAsm && genHint ? "plan+gen" : genHint ? "gen" : "plan",
    warnings,
  };
}

/** The INTERFACE addendum appended to a part's gen USER text when the plan
 *  declared a specific mate. Names the partner + mate kind + fit so the
 *  generator reproduces the matching counterpart feature (on top of the generic
 *  mating guidance already in the assembly addendum). Categorical only. */
function buildAssemblyBlock(a: AssemblyInterfacePlan): string {
  const partnerRef = a.partner ? `\`${a.partner}\`` : "the neighbour it attaches to";
  const mate = a.mate ?? "a keyed mating feature";
  const fit = a.fit ?? "location";
  const countClause = a.count && a.count > 1 ? ` (×${a.count})` : "";
  const fastenClause = a.fasten && a.fasten !== "none" ? ` Fasten with a ${a.fasten}.` : "";
  // Mate-specific seating hint. A benchmark run showed the interlocks
  // (snap_tab/tab_slot/key) and the vague seat_face mate register worst — the
  // model floats them or aligns them to a solid wall — so name the exact
  // failure mode for those kinds.
  const mateHint = ((): string => {
    switch (a.mate) {
      case "seat_face":
        return `- Seat FLAT and FLUSH on the exact face the reference shows on ${partnerRef}; overlap it by\n` +
               `  ~0.5–1 mm so the two faces actually TOUCH — not hovering above it, not sunk into the body.\n`;
      case "snap_tab":
        return `- Read the catch window/ledge ${partnerRef} already carries from the buffer and align YOUR\n` +
               `  cantilever tab to it exactly (same width and position); the hook must land in the OPENING,\n` +
               `  not press into a solid wall. If ${partnerRef} has no window, seat on its face instead.\n`;
      case "tab_slot":
        return `- Read the slot ${partnerRef} already carries from the buffer and size/place YOUR tab to\n` +
               `  ENTER it exactly — not butt against a solid face. If ${partnerRef} has no slot, seat\n` +
               `  flush on its face instead.\n`;
      case "key":
        return `- Read the keyway ${partnerRef} already carries from the buffer and size/place YOUR key to\n` +
               `  SIT IN that groove — not overlap solid stock. If ${partnerRef} has no keyway, use a\n` +
               `  locating boss on your own part instead.\n`;
      case "press_fit":
        return `- Interference fit: the male is slightly OVER the bore, so a little overlap is CORRECT — keep it\n` +
               `  a pin-in-bore, not a pin buried deep in solid material.\n`;
      default:
        return "";
    }
  })();
  return (
    `INTERFACE — this part MATES to ${partnerRef} via a **${mate}**${countClause} (${fit} fit):\n` +
    `- ${partnerRef} is FROZEN — you cannot modify it. Read its existing mating geometry and the\n` +
    `  relevant nominal from the SCAD-so-far buffer, and build the matching half on THIS part\n` +
    `  reusing that SAME value (female = nominal + clearance, male = nominal − clearance). Use the\n` +
    `  lib/assembly.scad helper for a ${mate} where one fits, with the "${fit}" fit class, and a\n` +
    `  lead-in chamfer on the entry.${fastenClause}\n` +
    mateHint +
    `- Seat the two on their real contact face — a proper interface, not a floating overlap and not\n` +
    `  a deep interpenetration. Keep the part's faithful size/pose (the mate is added at the contact).\n` +
    `- Optionally add a 5th block after PLACE describing the interface (categorical only, no numbers):\n` +
    `// INTERFACE\n` +
    `// {"partner": "${a.partner ?? "<earlier part>"}", "mate": "${a.mate ?? "<peg_socket|bolt_pattern|...>"}", ` +
    `"fit": "${a.fit ?? "<clearance|location|press|snap>"}", "locateHint": "<few words locating it>"}`
  );
}

/** The receiving/female counterpart a downstream mate needs cut into THIS part. */
function receivingFeatureFor(mate: InterfaceKind | undefined): string {
  switch (mate) {
    case "peg_socket":   return "cut the receiving SOCKET / bore (asm_socket) to accept its peg";
    case "press_fit":    return "cut the receiving bore (asm_socket) for its press pin";
    case "bolt_pattern": return "add the matching BOLT-HOLE pattern / screw bosses (asm_bolt_circle / asm_boss)";
    case "snap_tab":     return "cut the CATCH WINDOW / ledge (asm_snap_window) for its snap tab";
    case "tab_slot":     return "cut the receiving SLOT (asm_slot) for its tab";
    case "key":          return "cut the KEYWAY (asm_slot) for its key";
    case "flange":       return "provide the matching FLANGE face + bolt holes";
    case "lip_rabbet":   return "provide the reciprocal lip / rabbet edge";
    default:             return "provide the receiving / seating face";
  }
}

/**
 * (#3) Forward-looking obligations: the append-only loop freezes each part after
 * it is built, so a later part can NEVER cut a socket / bolt-hole / catch window
 * into its (already frozen) partner. This addendum tells the CURRENT part which
 * later parts will mate to it, so it pre-builds the receiving counterpart now
 * and two-sided mates seat by construction. Only emitted when there are incoming
 * obligations; models only the RECEPTACLES, never the incoming parts.
 */
function buildIncomingInterfaceBlock(
  incoming: Array<{ name: string; mate?: InterfaceKind; count?: number }>,
): string {
  const lines = incoming.map((c) => {
    const cnt = c.count && c.count > 1 ? ` (×${c.count})` : "";
    return `- \`${c.name}\` will mate via ${c.mate ?? "a mating feature"}${cnt} → ${receivingFeatureFor(c.mate)}.`;
  });
  return (
    "INCOMING INTERFACES — later parts will MATE TO this part, which is FROZEN after this\n" +
    "step, so pre-build the RECEIVING counterpart NOW so those mates seat by construction:\n" +
    lines.join("\n") + "\n" +
    "Place each receptacle where the reference shows that part attaches, at a shared nominal\n" +
    "(female = nominal + clearance) with a lead-in chamfer. Model only the receptacles on THIS\n" +
    "part — NOT the incoming parts themselves."
  );
}

/** The ARTICULATION addendum appended to a MOVING part's gen USER text.
 *  Placeholders come from the declared motion; clauses whose field is absent
 *  degrade to a generic phrasing rather than dangling. Categorical only. */
function buildArticulationBlock(m: PartMotionPlan): string {
  const aboutAlong = m.jointType === "prismatic" ? "along" : "about";
  const jointAxis = [m.jointType ?? "", m.axis ? `${aboutAlong} world ${m.axis}` : ""]
    .filter(Boolean).join(" ");
  const head = [jointAxis, m.parent ? `relative to \`${m.parent}\`` : ""]
    .filter(Boolean).join(", ");
  const parentRef = m.parent ? `\`${m.parent}\`` : "its joint parent";
  const edge = m.parent ? `the edge shared with \`${m.parent}\`` : "its mounting edge";
  const aligned = m.axis ? `world-aligned to ${m.axis}` : "world-axis-aligned";
  return (
    `ARTICULATION — this part MOVES${head ? ` (${head})` : ""}:\n` +
    `- Shape it articulable: model the real pivot (axle/hub bore at a wheel's center; hinge\n` +
    `  knuckle/pin boss at ${edge}). Where practical, make the part\n` +
    `  rotationally symmetric about its motion axis, centered on that axis in its local frame,\n` +
    `  so that after PLACE the axis is ${aligned}.\n` +
    `- Overlap rule override: the required >=0.5mm overlap must be with ${parentRef} ONLY. Leave\n` +
    `  visible CLEARANCE (no solid intersection) between this part and every OTHER existing part —\n` +
    `  a moving part welded to a non-parent jams in simulation.\n` +
    `- Emit a SINGLE placement statement in PLACE (no for() loops) so this instance stays\n` +
    `  individually addressable.\n` +
    `- Optionally add a 5th block after PLACE describing the joint (categorical only, no numbers):\n` +
    `// MOTION\n` +
    `// {"moving": true, "jointType": "${m.jointType ?? "<revolute|prismatic|spherical>"}", ` +
    `"parent": "${m.parent ?? "<earlier part name>"}", "axis": "${m.axis ?? "<X|Y|Z>"}", ` +
    `"role": "${m.role ?? "<short role>"}", "anchorHint": "<few words locating the pivot>"}`
  );
}

/**
 * Base USER text for one part's generation call (retry feedback is appended by
 * the caller). Exported for tests. With `motionAware` off the construction is
 * byte-identical to the pre-motion pipeline; when on, MOVING parts get an
 * ARTICULATION addendum after the faithfulness mandates. The part-gen SYSTEM
 * prompt is never touched in either mode.
 */
export function buildPartGenUserText(args: {
  text: string;
  part: PartPlanItem;
  remaining: string;
  cleanBuffer: string;
  motionAware: boolean;
  /** Assembly-aware mode: append the mating-feature addendum to every part's
   *  text. Empty/undefined when the flag is off (byte-identical construction). */
  assemblyAddendum?: string;
  /** (#3) Later parts that will mate TO this one — so it pre-builds their
   *  receiving counterparts (this part is frozen after this step). */
  incomingInterfaces?: Array<{ name: string; mate?: InterfaceKind; count?: number }>;
  /** Text-only mode: no reference image exists, so the instruction cannot tell
   *  the model to match one. */
  textOnly?: boolean;
}): string {
  const { text, part, remaining, cleanBuffer } = args;
  let userText =
    "You are adding ONE part to a model being built incrementally.\n\n" +
    `=== TEXT DESCRIPTION (whole object) ===\n${text}\n\n` +
    `=== PART TO ADD ===\nname: ${part.name}\nlevel: ${part.level ?? "?"}\n` +
    `description: ${part.description}\n\n` +
    `=== PARTS NOT YET BUILT (do NOT build these now) ===\n${remaining}\n\n` +
    `=== SCAD SO FAR (already compiles; do NOT redefine any of this) ===\n` +
    "```openscad\n" + cleanBuffer + "\n```\n\n" +
    `Emit ONLY the new part '${part.name}', placed so it overlaps an already-built ` +
    (args.textOnly
      ? `neighbour. There is NO reference image: build it from the description above. ` +
        `Choose its size RELATIVE to the whole object and to its already-built ` +
        `neighbours, its aspect ratio, and its pose/orientation from what the text ` +
        `says and from what a real example of this object looks like — and rotate it ` +
        `in PLACE where that is what the real part does, rather than defaulting to ` +
        `axis-aligned. `
      : `neighbour. Reconstruct it FAITHFULLY from the reference image: match its ` +
        `size RELATIVE to the whole object and to its already-built neighbours, its ` +
        `aspect ratio, and its pose/orientation (use rotate(...) in PLACE if the ` +
        `reference shows it tilted/splayed — do not default to axis-aligned). `) +
    `Read the existing parameters in the SCAD-so-far buffer so this part is to-scale ` +
    `with what's already built. Use the strict // PARAMS // HELPERS // PART // PLACE format.`;
  // Assembly-aware: generic mating-feature guidance for every part (Slice 1),
  // then the specific declared-interface directive when the plan named a mate
  // for THIS part (Slice 2). The faithfulness mandates above still win.
  if (args.assemblyAddendum) {
    userText += "\n\n" + args.assemblyAddendum.trimEnd();
  }
  if (part.assembly) {
    userText += "\n\n" + buildAssemblyBlock(part.assembly);
  }
  if (args.incomingInterfaces && args.incomingInterfaces.length > 0) {
    userText += "\n\n" + buildIncomingInterfaceBlock(args.incomingInterfaces);
  }
  // Motion-aware: articulation shaping for MOVING parts (its overlap-override
  // is intentionally last so it wins over the generic assembly guidance).
  if (args.motionAware && part.motion?.moving) {
    userText += "\n\n" + buildArticulationBlock(part.motion);
  }
  return userText;
}

/** Cached isolated-compile result for one placed part (world frame). Entries
 *  are small single-part meshes, so the parsed mesh is cached alongside the
 *  STL path + bbox — a parent shared by N children is compiled AND parsed once. */
export interface MotionMeshCacheEntry {
  stlPath: string;
  bbox: MeasuredBBox;
  mesh: STLMesh;
}

export interface PartMotionMeasurement {
  bbox?: MeasuredBBox;
  symmetryAxis?: MeasuredSymmetryAxis;
  parentContact?: MeasuredParentContact;
  /** True when a parent was requested AND its mesh compiled/loaded. */
  parentMeshOk?: boolean;
  warnings: string[];
}

/**
 * Deterministic post-commit mesh analysis of one placed part (world frame):
 * compile the part in isolation-in-assembly, then take its bbox, its
 * rotational-symmetry axis (non-degenerate only) and — when the joint parent
 * is committed — the part↔parent contact region. Part + parent compile
 * concurrently (only the contact computation needs both). Never throws: every
 * failure degrades to a warning on the result. `cache` memoizes compiled
 * meshes across parts (the part itself is always recompiled; only neighbours
 * are trusted from cache). Contact distance uses the same bbox-diagonal
 * fraction as `buildGeometricEvidence` at the motion exporter's call site.
 */
export async function measurePartMotion(args: {
  /** Accumulated SCAD including the part (scaffold markers are inert comments). */
  scad: string;
  placedName: string;
  parentPlacedName?: string;
  /** Scratch dir for the isolated compiles (one subdir per mesh name). */
  workDir: string;
  cache?: Map<string, MotionMeshCacheEntry>;
}): Promise<PartMotionMeasurement> {
  const out: PartMotionMeasurement = { warnings: [] };
  const cache = args.cache ?? new Map<string, MotionMeshCacheEntry>();

  const meshFor = async (
    name: string, reuseCache: boolean,
  ): Promise<{ mesh: STLMesh; bbox: MeasuredBBox } | null> => {
    const hit = reuseCache ? cache.get(name) : undefined;
    if (hit && existsSync(hit.stlPath)) return { mesh: hit.mesh, bbox: hit.bbox };
    const stlPath = await timeStage("openscad.part_measure",
      () => compilePartsInAssembly(args.scad, [name], join(args.workDir, name)));
    if (stlPath === null) return null;
    const mesh = loadSTL(stlPath);
    if (mesh.triCount === 0) return null;
    const bbox = computeBBox(mesh);
    cache.set(name, { stlPath, bbox, mesh });
    return { mesh, bbox };
  };

  const [partRes, parentRes] = await Promise.allSettled([
    meshFor(args.placedName, false),
    args.parentPlacedName !== undefined
      ? meshFor(args.parentPlacedName, true)
      : Promise.resolve(null),
  ]);
  const part = partRes.status === "fulfilled" ? partRes.value : null;
  if (partRes.status === "rejected") {
    out.warnings.push(`part mesh failed: ${(partRes.reason as Error).message.slice(0, 120)}`);
  }
  if (!part) {
    if (out.warnings.length === 0) {
      out.warnings.push("part mesh unavailable (isolated compile failed or empty)");
    }
    return out;
  }
  out.bbox = part.bbox;

  try {
    const sym = analyzeRotationalSymmetry(args.placedName, part.mesh);
    if (sym && !sym.degenerate) {
      out.symmetryAxis = {
        axisPoint: sym.axisPoint,
        axisDir: sym.axisDir,
        snappedAxis: sym.snappedAxis,
        score: sym.symmetryScore,
        confidence: sym.confidence,
      };
    }
  } catch (e) {
    out.warnings.push(`symmetry analysis failed: ${(e as Error).message.slice(0, 120)}`);
  }

  if (args.parentPlacedName) {
    const parent = parentRes.status === "fulfilled" ? parentRes.value : null;
    if (parentRes.status === "rejected") {
      out.warnings.push(`parent mesh failed: ${(parentRes.reason as Error).message.slice(0, 120)}`);
    } else if (!parent) {
      out.warnings.push(`parent mesh unavailable: ${args.parentPlacedName}`);
    } else {
      out.parentMeshOk = true;
      try {
        const diag = (b: MeasuredBBox): number => Math.hypot(b.size[0], b.size[1], b.size[2]);
        const maxDistance =
          DEFAULT_CONTACT_DISTANCE_FRAC * Math.max(diag(part.bbox), diag(parent.bbox));
        const contact = maxDistance > 0
          ? analyzeContactRegion(
              args.placedName, part.mesh, args.parentPlacedName, parent.mesh, { maxDistance },
            )
          : null;
        if (contact) {
          out.parentContact = {
            parent: args.parentPlacedName,
            anchor: contact.anchor,
            extent: contact.extent,
            principalDir: contact.principalDir,
            snappedAxis: contact.snappedAxis,
            elongation: contact.elongation,
            sampleCount: contact.sampleCount,
          };
        }
      } catch (e) {
        out.warnings.push(`contact analysis failed: ${(e as Error).message.slice(0, 120)}`);
      }
    }
  }
  return out;
}

function bboxOverlapVolume(a: MeasuredBBox, b: MeasuredBBox): number {
  const dx = Math.min(a.max[0], b.max[0]) - Math.max(a.min[0], b.min[0]);
  const dy = Math.min(a.max[1], b.max[1]) - Math.max(a.min[1], b.min[1]);
  const dz = Math.min(a.max[2], b.max[2]) - Math.max(a.min[2], b.min[2]);
  return dx > 0 && dy > 0 && dz > 0 ? dx * dy * dz : 0;
}

/**
 * Pure decision for the opt-in per-part motion gate (default OFF; enable with
 * PROCEDURA_INCREMENTAL_MOTION_GATE=1): a MOVING part must show a measurable
 * contact with its committed joint parent (check 1) and must not bury itself
 * in non-parent parts (check 2 — crude bbox-overlap against up to 4 cached
 * committed parts). Returns the retry-feedback string, or null when the part
 * passes. Measurement gaps (uncompiled parent, missing bboxes) pass — the
 * gate only acts on positive evidence, and the caller never drops a part for
 * motion reasons.
 */
export function evaluateMotionGate(args: {
  parentPlacedName: string | undefined;
  parentMeshOk: boolean;
  parentContact: MeasuredParentContact | undefined;
  partBBox: MeasuredBBox | undefined;
  others: Array<{ name: string; bbox: MeasuredBBox }>;
}): string | null {
  if (args.parentPlacedName && args.parentMeshOk && !args.parentContact) {
    return (
      `MOTION GATE: this moving part must attach to its joint parent ` +
      `\`${args.parentPlacedName}\` — place it so it overlaps ` +
      `\`${args.parentPlacedName}\` (>=0.5mm) and does not weld to other parts.`
    );
  }
  if (args.partBBox) {
    const vol = args.partBBox.size[0] * args.partBBox.size[1] * args.partBBox.size[2];
    if (vol > 0) {
      for (const other of args.others.slice(0, 4)) {
        if (bboxOverlapVolume(args.partBBox, other.bbox) > 0.2 * vol) {
          return (
            `MOTION GATE: moving part interpenetrates non-parent ` +
            `\`${other.name}\`; leave clearance.`
          );
        }
      }
    }
  }
  return null;
}

/**
 * Post-commit declare + measure + record for one committed part: build the
 * declared block from the merged motion, mesh-measure MOVING parts (reusing
 * the gate's measurement of this exact candidate when it ran), reconcile
 * declared-vs-measured axis agreement, push the record and save the sidecar.
 * Returns the per-part result summary (lands in parts_summary.json). The
 * caller wraps the call in try/catch — a failure may never break the commit.
 */
async function recordPartMotion(args: {
  sidecar: IncrementalMotionSidecar;
  decl: MergedMotionDecl;
  planName: string;
  placedName: string;
  parentPlaced: string | undefined;
  /** Committed accumulated SCAD (scaffold markers are inert comments). */
  scad: string;
  workDir: string;
  cache: Map<string, MotionMeshCacheEntry>;
  outDir: string;
  /** The motion gate's measurement of this exact candidate, when it ran. */
  premeasured: PartMotionMeasurement | null;
  /** Gate offence carried into a commit-with-warning on the last attempt. */
  gateNote: string | null;
  log: (line: string) => void;
}): Promise<NonNullable<IncrementalPartResult["motion"]>> {
  const { decl, placedName, parentPlaced } = args;
  const declared: PartMotionRecord["declared"] = {
    moving: decl.moving,
    jointType: decl.jointType,
    ...(decl.parent ? { parentPlanName: decl.parent } : {}),
    ...(parentPlaced ? { parentPlacedName: parentPlaced } : {}),
    ...(decl.axis ? { axis: decl.axis } : {}),
    ...(decl.role ? { role: decl.role } : {}),
    ...(decl.limitHint ? { limitHint: decl.limitHint } : {}),
    ...(decl.anchorHint ? { anchorHint: decl.anchorHint } : {}),
    source: decl.source,
  };
  const warnings = [...decl.warnings];
  if (decl.parent && !parentPlaced) warnings.push("unknown_parent");
  if (args.gateNote) warnings.push(`motion_gate: ${args.gateNote}`);
  const record: PartMotionRecord = {
    planName: args.planName, placedName, instanceIds: [], declared,
  };

  let measured: PartMotionMeasurement | null = null;
  if (declared.moving) {
    measured = args.premeasured ?? await measurePartMotion({
      scad: args.scad, placedName,
      ...(parentPlaced !== undefined ? { parentPlacedName: parentPlaced } : {}),
      workDir: args.workDir, cache: args.cache,
    });
    warnings.push(...measured.warnings);
    const m: NonNullable<PartMotionRecord["measured"]> = {
      ...(measured.bbox ? { bbox: measured.bbox } : {}),
      ...(measured.symmetryAxis ? { symmetryAxis: measured.symmetryAxis } : {}),
      ...(measured.parentContact ? { parentContact: measured.parentContact } : {}),
    };
    if (Object.keys(m).length > 0) record.measured = m;
    // Declared-vs-measured axis agreement (measured wins downstream).
    const snapped = measured.symmetryAxis?.confidence === "high"
      ? measured.symmetryAxis.snappedAxis : null;
    const axisAgrees = snapped !== null && declared.axis !== undefined
      ? snapped === declared.axis : null;
    record.agreement = {
      axisAgrees,
      ...(axisAgrees === false
        ? { note: `declared ${declared.axis} vs measured ${snapped}; seed will prefer measured` }
        : {}),
    };
  }

  if (warnings.length) record.warnings = warnings;
  args.sidecar.records.push(record);
  try { await saveIncrementalMotionSidecar(args.outDir, args.sidecar); }
  catch (e) { args.log(`      motion sidecar save failed (non-fatal): ${(e as Error).message}`); }

  const measuredAxis = measured?.symmetryAxis?.snappedAxis ?? null;
  if (declared.moving) {
    args.log(`      [motion] declared ${declared.jointType}` +
        `${declared.axis ? ` (${declared.axis})` : ""}` +
        `${parentPlaced ? ` rel '${parentPlaced}'` : ""}` +
        ` — measured axis ${measuredAxis ?? "none"}` +
        `${record.agreement?.axisAgrees === false ? " [DISAGREES — measured wins]" : ""}`);
  }
  return {
    moving: declared.moving,
    measuredAxis,
    axisAgrees: record.agreement?.axisAgrees ?? null,
    ...(args.gateNote ? { note: `motion gate: ${args.gateNote}` } : {}),
  };
}

// ──────────────────────────────────────────────────────────────────────────
// Assembly-aware mode: measure / record static mating interfaces
// ──────────────────────────────────────────────────────────────────────────

/** A part registers a mate when it TOUCHES its partner (>=1% of its surface
 *  near the partner) and does NOT bury itself in it beyond the fit-aware ceiling. */
export const ASSEMBLY_CONTACT_MIN_FRAC = 0.01;
export const ASSEMBLY_INTERPEN_MAX_FRAC = 0.20;

/** Modest interference allowance for a real press / snap engagement. Well below
 *  the earlier 0.55/0.45 (which were compensating for the missing pre-built
 *  socket, #3): with the receiving feature pre-built, an interference part sits
 *  IN the hole and overlaps only its engagement band. */
export const ASSEMBLY_INTERFERENCE_INTERPEN_FRAC = 0.35;

/**
 * Interpenetration a mate may show before it reads as "buried".
 * FIT-FIRST: an EXPLICIT fit class governs (a `clearance` fit never gets an
 * interference allowance, even on a `press_fit` mate); only when the fit is
 * unstated do we infer interference from a press/snap mate kind. A clearance/
 * location/seat mate should barely overlap (the flat 20% bar).
 */
export function interpenCeiling(
  mate: InterfaceKind | undefined, fit: FitClass | undefined,
): number {
  // Explicit fit wins.
  if (fit === "clearance" || fit === "location") return ASSEMBLY_INTERPEN_MAX_FRAC;
  if (fit === "press" || fit === "snap") return ASSEMBLY_INTERFERENCE_INTERPEN_FRAC;
  // No explicit fit — infer from the mate kind.
  if (mate === "press_fit" || mate === "snap_tab") return ASSEMBLY_INTERFERENCE_INTERPEN_FRAC;
  return ASSEMBLY_INTERPEN_MAX_FRAC; // clearance / location / seat / flange / ...
}

/**
 * Pure decision for the opt-in per-part assembly gate (default OFF; enable with
 * PROCEDURA_INCREMENTAL_ASSEMBLY_GATE=1): a part that declares a mate to a
 * committed partner must SHOW measurable contact with it and must not be
 * grossly buried inside it (past 1.5× the fit-aware verdict ceiling, min 40%, so
 * the gate only ever rejects egregious burial). Returns the retry-feedback
 * string, or null when the part passes. Measurement gaps (no partner,
 * unmeasured) pass — the gate acts only on positive evidence and the caller
 * never drops a part for this reason.
 */
export function evaluateAssemblyGate(args: {
  partnerPlacedName: string | undefined;
  measured: MeasuredMate | null;
  mate?: InterfaceKind;
  fit?: FitClass;
}): string | null {
  if (!args.partnerPlacedName || !args.measured) return null;
  const m = args.measured;
  if (m.contactAreaFrac < ASSEMBLY_CONTACT_MIN_FRAC) {
    return (
      `ASSEMBLY GATE: this part declares a mate to \`${args.partnerPlacedName}\` but does not ` +
      `touch it — seat its mating feature ON \`${args.partnerPlacedName}\` (reproduce the ` +
      `counterpart and overlap the contact face by >=0.5mm).`
    );
  }
  // Only reject on burial when the overlap was actually measured — a failed
  // intersection compile leaves it unknown and must not gate (nor pass) on it.
  const gateCeiling = Math.max(0.45, interpenCeiling(args.mate, args.fit) + 0.15);
  if (m.interpenComputed && m.interpenetrationFrac > gateCeiling) {
    return (
      `ASSEMBLY GATE: this part is buried inside \`${args.partnerPlacedName}\` ` +
      `(${(m.interpenetrationFrac * 100).toFixed(0)}% volume overlap) — seat it on the contact ` +
      `face with clearance, not interpenetrating.`
    );
  }
  return null;
}

/**
 * Classify an EMPTY (no-STL) softFail result from the interpenetration
 * intersection() compile into the tri-state. OpenSCAD exits non-zero for BOTH a
 * genuinely empty top-level object AND a compile failure, so the exit code alone
 * cannot separate "0 overlap, verified" from "overlap unknown". Three outcomes:
 *   - exitCode === null  → the process was KILLED (our timeout `proc.kill()`),
 *                          so it produced no STL for an UNKNOWN reason ⇒ unverifiable.
 *   - stderr has a real error / CGAL / parser / non-manifold marker ⇒ unverifiable.
 *   - otherwise           → a clean empty intersection ⇒ genuinely zero overlap.
 * Pure + exported so the tri-state invariant is unit-testable (a timed-out
 * compile must never read as a verified zero — the failure this guards).
 */
export function classifyIntersectionEmpty(
  r: { stderr: string; exitCode: number | null },
): { computed: boolean; reason?: string } {
  if (r.exitCode === null) {
    return { computed: false, reason: "intersection compile timed out / killed" };
  }
  const err = (r.stderr || "").toLowerCase();
  const realError = /error:|cgal error|parser error|warning: object may not be a valid 2-manifold/.test(err);
  if (realError) return { computed: false, reason: "intersection compile error" };
  return { computed: true }; // empty intersection ⇒ genuinely no overlap (frac 0)
}

/**
 * Deterministic post-commit mesh analysis of one placed part against its
 * declared assembly partner (world frame): contact fraction + contact anchor +
 * an interface-normal proxy (mesh proximity), plus an interpenetration fraction
 * from an OpenSCAD intersection() compile of the two isolated solids. Never
 * throws: every failure degrades to a warning / a 0 interpenetration. Reuses
 * the shared isolated-mesh cache (a partner shared by N children compiles once).
 */
export async function measurePartAssembly(args: {
  /** Committed accumulated SCAD (scaffold markers are inert comments). */
  scad: string;
  placedName: string;
  partnerPlacedName: string;
  workDir: string;
  cache?: Map<string, MotionMeshCacheEntry>;
}): Promise<{ measured: MeasuredMate | null; warnings: string[] }> {
  const warnings: string[] = [];
  const cache = args.cache ?? new Map<string, MotionMeshCacheEntry>();

  const meshFor = async (name: string): Promise<MotionMeshCacheEntry | null> => {
    const hit = cache.get(name);
    if (hit && existsSync(hit.stlPath)) return hit;
    const stlPath = await timeStage("openscad.part_measure",
      () => compilePartsInAssembly(args.scad, [name], join(args.workDir, name)));
    if (stlPath === null) return null;
    const mesh = loadSTL(stlPath);
    if (mesh.triCount === 0) return null;
    const entry: MotionMeshCacheEntry = { stlPath, bbox: computeBBox(mesh), mesh };
    cache.set(name, entry);
    return entry;
  };

  const [partR, partnerR] = await Promise.allSettled([
    meshFor(args.placedName), meshFor(args.partnerPlacedName),
  ]);
  const part = partR.status === "fulfilled" ? partR.value : null;
  const partner = partnerR.status === "fulfilled" ? partnerR.value : null;
  if (partR.status === "rejected") warnings.push(`part mesh failed: ${(partR.reason as Error).message.slice(0, 120)}`);
  if (partnerR.status === "rejected") warnings.push(`partner mesh failed: ${(partnerR.reason as Error).message.slice(0, 120)}`);
  if (!part) { warnings.push("part mesh unavailable (isolated compile failed or empty)"); return { measured: null, warnings }; }
  if (!partner) { warnings.push(`partner mesh unavailable: ${args.partnerPlacedName}`); return { measured: null, warnings }; }

  const diag = (b: MeasuredBBox): number => Math.hypot(b.size[0], b.size[1], b.size[2]);
  // Interface tolerance scales with the SMALLER part's diagonal (the local
  // feature scale), so a tiny part cannot "contact" a large body across a big
  // absolute gap (the max-diagonal version did). Symmetric analysis inside.
  const maxDistance = DEFAULT_CONTACT_DISTANCE_FRAC * Math.min(diag(part.bbox), diag(partner.bbox));
  const reg = maxDistance > 0 ? analyzeMateRegistration(part.mesh, partner.mesh, { maxDistance }) : null;
  if (!reg) { warnings.push("mate registration unavailable (empty/degenerate meshes)"); return { measured: null, warnings }; }

  // Interpenetration (C_overlap): intersect the two isolated world-frame solids
  // in OpenSCAD and measure the overlap volume. TRI-STATE: a compile ERROR /
  // timeout / degenerate volume leaves interpenetration UNKNOWN (interpenComputed
  // = false) — it must NOT read as "0 overlap"; only an exit-0 empty result is a
  // true zero. (softFail lets us tell an empty intersection from a failed one.)
  let interpenetrationFrac = 0;
  let interpenComputed = true;
  try {
    const volA = computeMeshVolume(part.mesh);
    const volB = computeMeshVolume(partner.mesh);
    const minVol = Math.min(volA, volB);
    if (!(minVol > 0)) {
      interpenComputed = false;
      warnings.push("interpenetration unverifiable: degenerate part volume");
    } else {
      const scad =
        `intersection() {\n  import("${resolve(part.stlPath)}");\n` +
        `  import("${resolve(partner.stlPath)}");\n}\n`;
      const r = await compileScad(scad, {
        outputDir: join(args.workDir, "_intersect"), timeoutMs: 120_000, softFail: true,
      });
      if (!r.empty) {
        // Overlap geometry produced → measure its volume.
        const ix = loadSTL(r.stlPath);
        const volIx = ix.triCount > 0 ? computeMeshVolume(ix) : 0;
        interpenetrationFrac = Math.min(1, volIx / minVol);
      } else {
        // No STL. OpenSCAD ALSO exits non-zero for an EMPTY top-level object, so
        // the exit code can't tell an empty intersection (0 overlap, verified)
        // from a real compile/CGAL failure or a timeout kill (overlap unknown).
        const cls = classifyIntersectionEmpty(r);
        if (!cls.computed) {
          interpenComputed = false;
          warnings.push(`interpenetration unverifiable: ${cls.reason}`);
        } else {
          interpenetrationFrac = 0; // empty intersection ⇒ genuinely no overlap
        }
      }
    }
  } catch (e) {
    interpenComputed = false;
    warnings.push(`interpenetration unverifiable: ${(e as Error).message.slice(0, 100)}`);
  }

  return {
    measured: {
      partner: args.partnerPlacedName,
      contactAnchor: reg.contactAnchor,
      contactNormal: reg.contactNormal,
      contactAreaFrac: reg.contactAreaFrac,
      interpenetrationFrac,
      interpenComputed,
      partBBox: part.bbox,
      sampleCount: reg.sampleCount,
    },
    warnings,
  };
}

/**
 * Post-commit declare + measure + record for one committed STATIC part: build
 * the declared block from the merged assembly interface, measure the mate vs
 * its committed partner, judge whether it registers (touches without burying),
 * push the record and save the sidecar. Skipped for MOVING parts (motion owns
 * that contact — the "one edge per pair" rule). Mirrors recordPartMotion; the
 * caller wraps it in try/catch so a failure never breaks the commit.
 */
async function recordPartAssembly(args: {
  sidecar: IncrementalAssemblySidecar;
  decl: MergedAssemblyDecl;
  planName: string;
  placedName: string;
  partnerPlaced: string | undefined;
  scad: string;
  workDir: string;
  cache: Map<string, MotionMeshCacheEntry>;
  outDir: string;
  /** The assembly gate's measurement of this exact candidate, when it ran. */
  premeasured?: { measured: MeasuredMate | null; warnings: string[] } | null;
  log: (line: string) => void;
}): Promise<NonNullable<IncrementalPartResult["assembly"]>> {
  const { decl, placedName, partnerPlaced } = args;
  const declared: AssemblyInterfaceRecord["declared"] = {
    ...(decl.partner ? { partnerPlanName: decl.partner } : {}),
    ...(partnerPlaced ? { partnerPlacedName: partnerPlaced } : {}),
    ...(decl.mate ? { mate: decl.mate } : {}),
    ...(decl.fit ? { fit: decl.fit } : {}),
    ...(decl.role ? { role: decl.role } : {}),
    ...(decl.count ? { count: decl.count } : {}),
    ...(decl.fasten ? { fasten: decl.fasten } : {}),
    ...(decl.locateHint ? { locateHint: decl.locateHint } : {}),
    source: decl.source,
  };
  const warnings = [...decl.warnings];
  if (decl.partner && !partnerPlaced) warnings.push("unknown_partner");
  const record: AssemblyInterfaceRecord = {
    planName: args.planName, placedName, instanceIds: [], declared,
  };

  let measured: MeasuredMate | null = null;
  if (partnerPlaced) {
    const m = args.premeasured ?? await measurePartAssembly({
      scad: args.scad, placedName, partnerPlacedName: partnerPlaced,
      workDir: args.workDir, cache: args.cache,
    });
    warnings.push(...m.warnings);
    measured = m.measured;
    if (measured) {
      record.measured = measured;
      const touches = measured.contactAreaFrac >= ASSEMBLY_CONTACT_MIN_FRAC;
      const ceiling = interpenCeiling(decl.mate, decl.fit);
      // Tri-state verdict: no contact ⇒ false (definitive); contact but
      // interpenetration unverifiable (overlap compile failed) ⇒ null (never
      // affirm "not buried" from a failed measurement); else touches && !buried.
      let mates: boolean | null;
      let note: string | undefined;
      if (!touches) {
        mates = false;
        note = `no measurable contact with ${partnerPlaced} (${(measured.contactAreaFrac * 100).toFixed(1)}%)`;
      } else if (!measured.interpenComputed) {
        mates = null;
        note = `contact ${(measured.contactAreaFrac * 100).toFixed(1)}% with ${partnerPlaced}, but interpenetration is unverifiable`;
      } else {
        const buried = measured.interpenetrationFrac > ceiling;
        mates = !buried;
        if (buried) note = `interpenetrates ${partnerPlaced} (${(measured.interpenetrationFrac * 100).toFixed(0)}% of min volume)`;
      }
      record.agreement = { mates, ...(note ? { note } : {}) };
    }
  }

  if (warnings.length) record.warnings = warnings;
  args.sidecar.records.push(record);
  try { await saveIncrementalAssemblySidecar(args.outDir, args.sidecar); }
  catch (e) { args.log(`      assembly sidecar save failed (non-fatal): ${(e as Error).message}`); }

  const verdict = record.agreement?.mates;
  args.log(
    `      [assembly] ${decl.mate ?? "mate"}${partnerPlaced ? ` → '${partnerPlaced}'` : ""}` +
      (measured
        ? ` — contact ${(measured.contactAreaFrac * 100).toFixed(0)}%, ` +
          `interpen ${measured.interpenComputed ? `${(measured.interpenetrationFrac * 100).toFixed(0)}%` : "?"}` +
          `${verdict === false ? " [DOES NOT REGISTER]" : verdict === true ? " [ok]" : " [unverifiable]"}`
        : partnerPlaced ? " — not measured" : " — no committed partner"),
  );

  return {
    mate: decl.mate ?? null,
    partnerResolved: partnerPlaced ?? null,
    ...(measured
      ? { contactAreaFrac: measured.contactAreaFrac, interpenetrationFrac: measured.interpenetrationFrac }
      : {}),
    mates: measured ? record.agreement?.mates ?? null : null,
  };
}

// ──────────────────────────────────────────────────────────────────────────
// Main entry
// ──────────────────────────────────────────────────────────────────────────

export async function runIncrementalDraft(
  opts: IncrementalDraftOpts,
): Promise<IncrementalDraftResult> {
  const text = opts.text.trim();
  if (!text) throw new Error("runIncrementalDraft: `text` is required (and non-empty)");
  const log = opts.log ?? ((s) => console.log(s));
  const t0 = Date.now();

  const outDir = resolve(opts.outputDir);
  mkdirSync(outDir, { recursive: true });
  writeFileSync(join(outDir, "prompt.txt"), text, "utf8");
  writeFileSync(join(outDir, "effective_text.txt"), text, "utf8");
  const imagePath = join(outDir, "image.png");
  const scadPath = join(outDir, "draft.scad");
  const stlPathOut = join(outDir, "draft.stl");
  const objPathOut = join(outDir, "draft.obj");
  const partsDir = join(outDir, "_parts");
  mkdirSync(partsDir, { recursive: true });
  // A fresh draft invalidates any previous run's motion/assembly measurements
  // (e.g. a --redo over an old run): drop the stale sidecars so the Phase-4
  // consumer can never read poisoned data. A RESUME keeps the committed parts'
  // sidecars so their priors survive into Phase 4.
  if (!opts.resume) {
    rmSync(join(outDir, INCREMENTAL_MOTION_FILE), { force: true });
    rmSync(join(outDir, ASSEMBLY_INCREMENTAL_FILE), { force: true });
  }

  const scadModelKey = opts.scadModel ?? DEFAULT_SCAD_MODEL;
  const scadModelRef: ModelRef = resolveModel(scadModelKey);
  const route = routeForModel(scadModelKey);

  const genImage = async (prompt: string, outputPath: string): Promise<void> => {
    await generateImage({
      prompt, outputPath,
      ...(opts.imageModel !== undefined ? { model: opts.imageModel } : {}),
      log,
    });
  };
  const exportStl = opts.exportStl ?? false;
  const maxParts = opts.maxParts ?? DEFAULT_MAX_PARTS;
  // Motion-aware incremental mode (declare → shape → measure → sidecar). Set
  // by the pipeline when --incremental and --motion are both on;
  // PROCEDURA_INCREMENTAL_MOTION=0 is the env kill switch (same pattern as
  // PROCEDURA_INCREMENTAL_CONN_GATE). With this off, prompts and behavior are
  // byte-identical to the non-motion pipeline.
  const motionAware = (opts.motionAware ?? false) &&
    (process.env["PROCEDURA_INCREMENTAL_MOTION"] ?? "1") !== "0";
  // Image-resolution normalization. DEFAULT 1024: every REFERENCE image is
  // downscaled to 1024px longest-side and every parts-colour FEEDBACK render is
  // produced at 1024px, so the reference and the renders the model compares sit
  // at the same resolution. Override via env (set 0 to disable / restore the old
  // 1254-ref, 512-feedback behavior; or any other px).
  const refImageSize = Number(process.env["PROCEDURA_REF_IMAGE_SIZE"] ?? "1024");
  // Text-only and a supplied input image are contradictory; the explicit flag
  // wins and says so, rather than silently using the image it was told to skip.
  const textOnly = opts.textOnly ?? false;
  const feedbackRenderSize = Number(process.env["PROCEDURA_FEEDBACK_RENDER_SIZE"] ?? "1024");
  const CONTEXT_VIEWS = resolveContextViews(opts.contextRenders ?? false);
  // Assembly-aware incremental mode (Slice 1: mating-feature library + prompt).
  // Same opt-in-plus-kill-switch shape as motionAware; independent of it.
  const assemblyAware = (opts.assemblyAware ?? false) &&
    (process.env["PROCEDURA_INCREMENTAL_ASSEMBLY"] ?? "1") !== "0";
  // Read the mating addendum once (empty when off → byte-identical gen text).
  const assemblyAddendum = assemblyAware
    ? readFileSync(SCAD_PART_ASSEMBLY_ADDENDUM_PATH, "utf8")
    : "";
  // NO-PLAN ablation: no upfront decomposition, no plan review; the next part is
  // decided one step at a time. Opt-in from the caller or from the environment
  // (the env form is what the benchmark rig sets, so an ablation arm is a config
  // push and not a second code path). Off → every prompt and every call in this
  // file is byte-identical to the planned pipeline.
  const noPlan = (opts.noPlan ?? false) || process.env["PROCEDURA_NO_PLAN"] === "1";

  // Trajectory: shared sink (unified pipeline) or a local file.
  const trajectoryDir = join(outDir, "_trajectory");
  mkdirSync(trajectoryDir, { recursive: true });
  const localWriter = opts.trajectorySink
    ? null
    : createFileTrajectoryWriter(trajectoryDir, `inc-${nextId("d").slice(2)}`);
  const sink = opts.trajectorySink ?? localWriter!.sink;
  const trajectoryPath = opts.trajectoryPathOverride ?? localWriter!.path;

  // No harness. This stage never calls a tool: it makes one-shot generations
  // (plan, plan review, one per part) and writes trajectory events. The harness
  // was carrying a session store, an event bus, a sandbox and a ruleset for a
  // loop that does not exist here — see src/trajectory/emitter.ts.
  const sessionId = `sess_${nextId("d").slice(2)}`;
  const runId = nextId("run");
  const emitter = createStageEmitter({
    sink, sessionId, workspaceDir: outDir, runId,
    source: "draft-incremental",
    provider: scadModelRef.providerId,
    modelId: scadModelRef.modelId,
  });
  const emit = (type: string, payload: Record<string, unknown> = {}): void =>
    emitter.emit(type, { sessionId, ...payload });
  emit("run.started", { runId });

  async function llmGenerate(
    system: string, userContent: CanonicalPart[],
  ): Promise<{ rawText: string; reasoning: string }> {
    const tLLM = Date.now();
    const r = await generateOnce({
      route, model: scadModelRef, system, parts: userContent,
      ...(opts.signal !== undefined ? { signal: opts.signal } : {}),
    });
    addStage("llm.generate", Date.now() - tLLM);
    return { rawText: r.text, reasoning: r.reasoning };
  }

  /** Record one generation in the trajectory. The full text is already on disk
   *  per stage (plan_response.txt, gen_response_N.txt); this is the index. */
  function recordTurn(
    label: string, _hasImage: boolean,
    assistantText: string, reasoning: string,
    meta: Record<string, unknown>,
  ): void {
    emit("llm.turn", {
      ...meta, label,
      textChars: assistantText.length,
      reasoningChars: reasoning.length,
    });
  }

  const plan: PartPlanItem[] = [];
  const partResults: IncrementalPartResult[] = [];
  const planJsonPath = join(outDir, "plan.json");
  // Resume a killed draft: reuse the plan + the parts already committed to
  // draft.scad, and continue from the first uncommitted part.
  const resuming = Boolean(opts.resume) && existsSync(planJsonPath) && existsSync(scadPath);
  // Assembly-aware: inline the mating-feature helper library into the seed so
  // both mating parts call identical helpers (fit by construction) and every
  // derived compile stays self-contained. On RESUME the committed draft.scad
  // already carries the lib, so read it as-is. Off → the historical bare seed.
  let accumulated = resuming
    ? ensureMarkers(readFileSync(scadPath, "utf8"))
    : assemblyAware
      ? injectAssemblyLib(SEED_SCAD, readFileSync(ASSEMBLY_LIB_PATH, "utf8"))
      : SEED_SCAD;
  let partsGenerated = 0;
  if (resuming) {
    plan.push(...(JSON.parse(readFileSync(planJsonPath, "utf8")) as PartPlanItem[]));
    log(`[inc-draft] RESUME: reusing ${plan.length}-part plan.json + committed parts in draft.scad`);
  }
  let lastCompileErr: string | null = null;

  // Motion-aware state: the sidecar the Phase-4 motion planner consumes, a
  // per-part compiled-mesh/bbox cache (parents recur across parts), and the
  // planName→placedName map of COMMITTED parts for joint-parent resolution.
  const motionSidecar = motionAware ? createIncrementalMotionSidecar() : null;
  const assemblySidecar = assemblyAware ? createIncrementalAssemblySidecar() : null;
  // Shared isolated-mesh cache: motion AND assembly measurement both compile a
  // part (and its parent/partner) in isolation-in-assembly; a part shared across
  // both concerns, or a partner shared by N children, compiles once.
  const motionMeshCache = new Map<string, MotionMeshCacheEntry>();
  const placedByPlanName = new Map<string, string>();

  try {
    // ── Stage A: image (provided input, generated, or skipped entirely) ──
    if (textOnly) {
      // Nothing to do. Saying so explicitly matters because every later stage
      // reads the ABSENCE of image.png as the mode switch, and a silent skip
      // here would look identical to a failed image-gen in the log.
      log(`[inc-draft] TEXT-ONLY — no reference image will be generated or used`);
      emit("draft.image.skipped", { source: "text-only", bytes: 0 });
    } else if (opts.inputImage) {
      const src = resolve(opts.inputImage);
      if (!existsSync(src)) throw new Error(`inputImage not found: ${src}`);
      log(`[inc-draft] image-gen SKIPPED — using provided image ${src}`);
      if (resolve(imagePath) !== src) copyFileSync(src, imagePath);
      emit("draft.image.skipped", { source: src, bytes: fileSize(imagePath) });
      log(`      image ok (provided, ${fileSize(imagePath)} bytes)`);
    } else {
      // Image generation is opt-in. Refuse here rather than reaching for an
      // image API the user never configured.
      if (!imageGenAvailable(opts.imageModel)) {
        throw new Error(imageGenDisabledReason(opts.imageModel));
      }
      const imageModel = resolveImageModel(opts.imageModel)!;
      log(`[inc-draft] image-gen via ${imageModel}`);
      const imagePrompt = buildImagePrompt(text);
      writeFileSync(join(outDir, "image_prompt.txt"), imagePrompt, "utf8");
      emit("draft.image.started", { model: imageModel, promptChars: imagePrompt.length });
      await genImage(imagePrompt, imagePath);
      emit("draft.image.finished", { path: imagePath, bytes: fileSize(imagePath) });
      log(`      image ok (${fileSize(imagePath)} bytes)`);
    }
    // ── Build the reference set (primary image + optional generated extras) ─
    // Extra refs are GENERATED views, so they need image-gen even when the
    // primary reference was supplied with --image.
    const wantExtraRefs = textOnly ? 0 : Math.max(0, opts.extraRefs ?? 0);
    const canGenExtras = wantExtraRefs === 0 || imageGenAvailable(opts.imageModel);
    if (wantExtraRefs > 0 && !canGenExtras) {
      log(`[inc-draft] multi-ref: SKIPPED — image generation is off ` +
          `(set PROCEDURA_IMAGE_MODEL to enable extra reference views)`);
    }
    const extraRefs = canGenExtras ? wantExtraRefs : 0;
    if (!textOnly && refImageSize > 0) log(`[inc-draft] reference images normalized to ${refImageSize}px longest-side`);
    const refImages: { label: string; b64: string }[] = textOnly ? [] : [
      { label: "primary", b64: loadImageBase64(imagePath, refImageSize) },
    ];
    for (let k = 0; k < extraRefs; k++) {
      if (opts.signal?.aborted) break;
      const refPath = join(outDir, `image_ref${k + 2}.png`);
      const viewPhrase = EXTRA_REF_VIEWS[k % EXTRA_REF_VIEWS.length]!;
      try {
        log(`[inc-draft] multi-ref: generating extra ref ${k + 2} — ${viewPhrase}`);
        await genImage(buildExtraRefPrompt(text, viewPhrase), refPath);
        refImages.push({ label: viewPhrase, b64: loadImageBase64(refPath, refImageSize) });
      } catch (e) {
        log(`[inc-draft] extra ref ${k + 2} gen failed: ${(e as Error).message} — skipping`);
      }
    }
    if (extraRefs > 0) {
      log(`[inc-draft] multi-ref: attaching ${refImages.length} reference image(s) to plan + gen calls`);
      emit("draft.multiref.ready", { refCount: refImages.length });
    }

    // Attach the full reference set under `header`. With a single ref the format
    // is unchanged; with several, the primary is flagged authoritative and the
    // generated alternates as supplementary viewpoints.
    const refParts = (header: string): CanonicalPart[] => {
      // Text-only: say plainly that there is no image rather than attaching
      // none silently. A prompt whose body says "match the reference" with no
      // reference attached invites the model to invent one and describe it.
      if (refImages.length === 0) {
        return [{
          kind: "text",
          text: "There is NO reference image for this object — the text description " +
                "above is the complete and only specification. Do not describe or " +
                "assume an image. Where the text is silent, choose what a competent " +
                "engineer would build and keep it consistent with the parts already " +
                "committed.",
        }];
      }
      if (refImages.length === 1) {
        return [
          { kind: "text", text: `${header}:` },
          { kind: "image", data: refImages[0]!.b64, mimeType: "image/png" },
        ];
      }
      const parts: CanonicalPart[] = [{
        kind: "text",
        text: `${header} — ${refImages.length} views. View 1 is the PRIMARY, ` +
          `authoritative reference (match it); the others are generated alternate ` +
          `viewpoints for extra shape context:`,
      }];
      refImages.forEach((r, idx) => {
        parts.push({
          kind: "text",
          text: idx === 0 ? `view ${idx + 1} (primary):` : `view ${idx + 1} (${r.label}):`,
        });
        parts.push({ kind: "image", data: r.b64, mimeType: "image/png" });
      });
      return parts;
    };

    // ── Stage B: plan ───────────────────────────────────────────────────
    if (!noPlan) {
      log(`[inc-draft] planning parts`);
      emit("draft.plan.requested");
    }
    // Articulation addendum: appended to the plan + plan-review SYSTEM prompts
    // only in motion-aware mode — when off, both prompts are byte-identical to
    // the non-motion pipeline (the addendum string is empty).
    const planMotionAddendum = motionAware
      ? "\n\n" + readFileSync(PLAN_MOTION_ADDENDUM_PATH, "utf8")
      : "";
    const planAssemblyAddendum = assemblyAware
      ? "\n\n" + readFileSync(PLAN_ASSEMBLY_ADDENDUM_PATH, "utf8")
      : "";
    const planSystem = readFileSync(PLAN_SYSTEM_PATH, "utf8") + planMotionAddendum + planAssemblyAddendum;
    // NO-PLAN ablation: Stage B and Stage B2 do not happen at all. The
    // per-step decision that replaces them lives in Stage C, where it can see
    // the build so far.
    const nextPartSystem = noPlan
      ? readFileSync(PLAN_NEXT_SYSTEM_PATH, "utf8") + planMotionAddendum + planAssemblyAddendum
      : "";
    if (noPlan) {
      log(`[inc-draft] NO-PLAN ABLATION: no build plan and no plan review — ` +
          `the next part is decided one step at a time (hard cap ${NOPLAN_HARD_CAP} parts)`);
      emit("draft.plan.skipped", { reason: "no-plan-ablation", hardCap: NOPLAN_HARD_CAP });
      // Written now (empty) and rewritten after every accepted part, so a
      // killed run still leaves the parts it decided on, in order.
      if (!resuming) writeFileSync(planJsonPath, "[]", "utf8");
    }
    if (!noPlan && !resuming) {
      let planErr: string | null = null;
      for (let attempt = 1; attempt <= PLAN_MAX_ATTEMPTS; attempt++) {
        const planUserText =
          "Object to decompose:\n\n" +
          `=== TEXT DESCRIPTION ===\n${text}\n\n` +
          (textOnly ? "" : "=== REFERENCE IMAGE ===\nAttached below.\n\n") +
          (planErr ? `Your previous reply could not be parsed (${planErr}). ` : "") +
          "Produce the ordered JSON build plan now. Return ONLY the JSON array.";
        const { rawText, reasoning } = await llmGenerate(planSystem, [
          { kind: "text", text: planUserText },
          ...refParts("Reference image"),
        ]);
        writeFileSync(join(outDir, "plan_response.txt"), rawText, "utf8");
        if (reasoning.trim()) writeFileSync(join(outDir, "plan_thinking.txt"), reasoning, "utf8");
        recordTurn("(plan request)", true, rawText, reasoning, { stage: "plan", attempt });
        try {
          plan.push(...parsePlanJson(rawText, maxParts, { motion: motionAware, assembly: assemblyAware }));
          break;
        } catch (e) {
          planErr = (e as Error).message;
          log(`      plan parse failed (attempt ${attempt}): ${planErr}`);
        }
      }
      if (plan.length === 0) {
        throw new Error(`planning failed: ${planErr ?? "unparseable"}`);
      }
    }
    if (!noPlan) {
      {
        const dropped = enforceMotionParentOrder(plan);
        if (dropped.length) {
          log(`      plan motion: dropped forward/unknown parent ref on ${dropped.join(", ")}`);
        }
        const droppedAsm = enforceAssemblyPartnerOrder(plan);
        if (droppedAsm.length) {
          log(`      plan assembly: dropped forward/unknown partner ref on ${droppedAsm.join(", ")}`);
        }
      }
      writeFileSync(join(outDir, "plan.json"), JSON.stringify(plan, null, 2), "utf8");
      emit("draft.plan.ready", { partCount: plan.length });
      log(`      plan: ${plan.length} parts — ${plan.map((p) => p.name).join(", ")}`);
    }

    // ── Stage B2: plan review/refine loop ───────────────────────────────
    // A critic reviews the plan against the reference + text (completeness,
    // ordering, decomposition, description detail) and returns a corrected
    // plan + an `ok` flag. Loop until ok or DEFAULT_PLAN_REVIEW_ITERS. The
    // whole loop is best-effort: a transport failure or an unparseable review
    // stops reviewing and proceeds with the best plan so far (never kills the
    // run). Set PROCEDURA_PLAN_REVIEW_ITERS=0 to disable.
    const planReviewIters = opts.planReviewIters ?? DEFAULT_PLAN_REVIEW_ITERS;
    if (!noPlan && !resuming && planReviewIters > 0) {
      const planReviewSystem = readFileSync(PLAN_REVIEW_SYSTEM_PATH, "utf8") +
        planMotionAddendum + planAssemblyAddendum +
        (motionAware ? "\n\n" + PLAN_REVIEW_MOTION_NOTE : "") +
        (assemblyAware ? "\n\n" + PLAN_REVIEW_ASSEMBLY_NOTE : "");
      emit("draft.plan.review_started", { maxIters: planReviewIters });
      for (let iter = 1; iter <= planReviewIters; iter++) {
        const planText = plan
          .map((p, i) => `${i + 1}. ${p.name} (${p.level ?? "L?"}): ${p.description}` +
            (motionAware && p.motion ? ` [motion: ${describeMotionDecl(p.motion)}]` : "") +
            (assemblyAware && p.assembly ? ` [assembly: ${describeAssemblyDecl(p.assembly)}]` : ""))
          .join("\n");
        const reviewParts: CanonicalPart[] = [
          {
            kind: "text",
            text:
              (textOnly
                ? "Review this build plan against the text description. "
                : "Review this build plan against the reference image and text. ") + +
              "ADD-AND-SHARPEN ONLY: add genuinely missing parts and sharpen vague " +
              "descriptions — never merge, remove, rename, or reorder the planned " +
              "parts, and keep the planner's left/right assignments.\n\n" +
              `=== TEXT DESCRIPTION ===\n${text}\n\n` +
              `=== CURRENT PLAN (${plan.length} parts) ===\n${planText}\n\n` +
              "Return ONLY the JSON object {ok, notes, plan}.",
          },
          ...refParts("Reference image (whole object)"),
        ];
        let rawText = ""; let reasoning = "";
        try {
          ({ rawText, reasoning } = await llmGenerate(planReviewSystem, reviewParts));
        } catch (e) {
          log(`      plan-review iter ${iter} call failed: ${(e as Error).message} — keeping plan`);
          break;
        }
        writeFileSync(join(outDir, `plan_review_${pad2(iter)}.txt`), rawText, "utf8");
        if (reasoning.trim()) writeFileSync(join(outDir, `plan_review_${pad2(iter)}_thinking.txt`), reasoning, "utf8");
        recordTurn(`(plan review ${iter})`, true, rawText, reasoning, { stage: "plan-review", iter });
        const review = parsePlanReview(rawText, maxParts, { motion: motionAware, assembly: assemblyAware });
        if (!review) {
          log(`      plan-review iter ${iter}: unparseable response — keeping plan, ending review`);
          break;
        }
        if (review.plan && review.plan.length) {
          const m = mergeReviewedPlan(plan, review.plan);
          plan.splice(0, plan.length, ...m.plan);
          const droppedParents = enforceMotionParentOrder(plan);
          if (droppedParents.length) {
            log(`      plan review ${iter}: dropped forward/unknown motion parent on ${droppedParents.join(", ")}`);
          }
          const droppedPartners = enforceAssemblyPartnerOrder(plan);
          if (droppedPartners.length) {
            log(`      plan review ${iter}: dropped forward/unknown assembly partner on ${droppedPartners.join(", ")}`);
          }
          writeFileSync(join(outDir, "plan.json"), JSON.stringify(plan, null, 2), "utf8");
          if (m.rejectedDrops.length) {
            log(`      plan review ${iter}: REJECTED removal/rename of ${m.rejectedDrops.join(", ")} (add-only)`);
          }
          if (m.added.length || m.sharpened.length) {
            log(`      plan review ${iter}: +${m.added.length} part(s)` +
                (m.added.length ? ` [${m.added.join(", ")}]` : "") +
                `, sharpened ${m.sharpened.length}`);
          }
        }
        log(`      plan review ${iter}/${planReviewIters}: ${review.ok ? "OK" : "revise"} → ` +
            `${plan.length} parts${review.notes ? " — " + review.notes.slice(0, 160) : ""}`);
        emit("draft.plan.reviewed", { iter, ok: review.ok, partCount: plan.length });
        if (review.ok) break;
      }
    }

    const partSystem = readFileSync(SCAD_PART_SYSTEM_PATH, "utf8");

    // Running baseline for the delta-aware per-part connectivity gate: the
    // visible-floater count of the build-so-far after the last committed part.
    // A part is rejected only if it *increases* this count, so a legitimately
    // late-connecting part (bridged by a later part) is never blocked.
    let floatersBefore = 0;
    // Gate is on by default; set PROCEDURA_INCREMENTAL_CONN_GATE=0 to reproduce the
    // pre-gate baseline (commit every part unconditionally, no connectivity check).
    const connGateEnabled = (process.env["PROCEDURA_INCREMENTAL_CONN_GATE"] ?? "1") !== "0";
    log(`[inc-draft] per-part connectivity gate: ${connGateEnabled ? "ON" : "OFF"}`);
    log(`[inc-draft] 3D feedback per part: ${CONTEXT_VIEWS.length ? CONTEXT_VIEWS.join(", ") : "OFF"}`);
    // Opt-in motion gate (pre-commit parent-attachment / non-parent-fusion
    // check on MOVING parts); default OFF — set PROCEDURA_INCREMENTAL_MOTION_GATE=1.
    const motionGateEnabled =
      motionAware && process.env["PROCEDURA_INCREMENTAL_MOTION_GATE"] === "1";
    if (motionAware) {
      log(`[inc-draft] motion-aware incremental: ON (sidecar → ${INCREMENTAL_MOTION_FILE}); ` +
          `motion gate: ${motionGateEnabled ? "ON" : "OFF"}`);
    }
    // Opt-in pre-commit assembly gate (contact + interpenetration on parts with a
    // declared partner); default OFF — set PROCEDURA_INCREMENTAL_ASSEMBLY_GATE=1.
    const assemblyGateEnabled =
      assemblyAware && process.env["PROCEDURA_INCREMENTAL_ASSEMBLY_GATE"] === "1";
    // ── Profiling escape hatches. Both DEGRADE the build; they exist to
    // measure the floor cost of gen + split + render, not to ship with. ──
    // Skips every mate measurement, including the floater-override path that
    // rescues a correctly-seated clearance fit — so more parts read as floaters.
    const skipMateCheck = process.env["PROCEDURA_SKIP_MATE_CHECK"] === "1";
    // Skips the per-part whole-assembly compile. That compile is ALSO how a
    // broken part is detected: without it there is no compile error to retry
    // on, no connectivity gate, and no build-so-far mesh, so parts commit
    // unvalidated and the draft can end up structurally wrong.
    const skipPartCompile = process.env["PROCEDURA_SKIP_PART_COMPILE"] === "1";
    if (skipMateCheck || skipPartCompile) {
      log(`  WARNING: profiling flags active — ` +
          `${skipPartCompile ? "per-part compile+gate OFF " : ""}` +
          `${skipMateCheck ? "mate check OFF " : ""}` +
          `— this build is NOT validated per part.`);
    }
    if (assemblyAware) {
      log(`[inc-draft] assembly-aware incremental: ON (lib/assembly.scad inlined + mating prompt` +
          `${assemblySidecar ? `; sidecar → ${ASSEMBLY_INCREMENTAL_FILE}` : ""}); ` +
          `assembly gate: ${assemblyGateEnabled ? "ON" : "OFF"}`);
    }

    /** Parts-colour context render of the build so far (the 6 ortho faces +
     *  the hero isometric). Best-effort: an empty list means the render was
     *  unavailable, never that nothing is built. The generation call has always
     *  used it; the no-plan ablation's decision step uses the SAME render, so a
     *  step costs one render and both calls look at the same picture. */
    const renderContextViews = async (
      ctxDir: string, index: number,
    ): Promise<{ view: string; b64: string }[]> => {
      // Gate on parts committed so far (not on draft.stl, which is no longer
      // written by default — the context render recompiles from draft.scad).
      if (CONTEXT_VIEWS.length === 0 || partsGenerated === 0 || !existsSync(scadPath)) return [];
      try {
        mkdirSync(ctxDir, { recursive: true });
        const scratch = contextPartsScratch(sessionId);
        const r = await renderPartsColorViews({
          scadPath, outDir: ctxDir, views: CONTEXT_VIEWS,
          size: feedbackRenderSize > 0 ? feedbackRenderSize : 512, samples: 16,
          fnOverride: CONTEXT_RENDER_FN,
          ...(scratch ? { partsScratchDir: join(scratch, `p${index}`) } : {}),
        });
        if (r.ok) {
          return r.views.map((v) => ({
            view: v.view, b64: readFileSync(v.path).toString("base64"),
          }));
        }
      } catch { /* best-effort */ }
      return [];
    };

    /** NO-PLAN ablation, one step: name the single next part, or declare the
     *  object complete. Returns null for done, for a reply that stays
     *  unparseable, and for a failed call — in all three the build stops with
     *  what it has, which is the honest outcome for an arm whose whole claim is
     *  that the model can drive its own decomposition. */
    const decideNextPart = async (
      index: number, views: { view: string; b64: string }[],
    ): Promise<PartPlanItem | null> => {
      const builtList = plan.length === 0
        ? "(nothing built yet — this is the FIRST part)"
        : plan.map((p, k) => {
            const r = partResults.find((x) => x.name === p.name);
            const failed = r && !r.generated ? "  [FAILED to build — NOT in the model]" : "";
            return `${k + 1}. ${p.name} (${p.level ?? "L?"}): ${p.description}${failed}`;
          }).join("\n");
      let parseErr: string | null = null;
      for (let attempt = 1; attempt <= NEXT_PART_MAX_ATTEMPTS; attempt++) {
        const userParts: CanonicalPart[] = [
          {
            kind: "text",
            text:
              "Decide the SINGLE next part to add to this object, or declare the object " +
              "complete.\n\n" +
              `=== TEXT DESCRIPTION (whole object) ===\n${text}\n\n` +
              `=== PARTS ALREADY BUILT, in build order (${plan.length}) ===\n${builtList}\n\n` +
              (parseErr ? `Your previous reply could not be parsed (${parseErr}). ` : "") +
              "Return ONLY the JSON object.",
          },
          ...refParts("Reference image (whole object)"),
        ];
        if (views.length > 0) {
          userParts.push({
            kind: "text",
            text: "The MODEL BUILT SO FAR (parts-coloured renders of what exists right now):",
          });
          for (const v of views) {
            userParts.push({ kind: "text", text: `${v.view}:` });
            userParts.push({ kind: "image", data: v.b64, mimeType: "image/png" });
          }
        }
        let rawText = ""; let reasoning = "";
        try {
          ({ rawText, reasoning } = await llmGenerate(nextPartSystem, userParts));
        } catch (e) {
          log(`      next-part call failed: ${(e as Error).message} — ending the build here`);
          return null;
        }
        writeFileSync(join(outDir, `next_part_${pad2(index + 1)}.txt`), rawText, "utf8");
        if (reasoning.trim()) {
          writeFileSync(join(outDir, `next_part_${pad2(index + 1)}_thinking.txt`), reasoning, "utf8");
        }
        recordTurn(`(next part ${index + 1})`, true, rawText, reasoning,
          { stage: "next-part", index: index + 1, attempt });
        let decision: ReturnType<typeof parseNextPartJson>;
        try {
          decision = parseNextPartJson(rawText, { motion: motionAware, assembly: assemblyAware });
        } catch (e) {
          parseErr = (e as Error).message;
          log(`      next-part parse failed (attempt ${attempt}): ${parseErr}`);
          continue;
        }
        if (decision.done) {
          log(`[inc-draft] no-plan: model declared the object COMPLETE after ${plan.length} part(s)` +
              (decision.reason ? ` — ${decision.reason.slice(0, 160)}` : ""));
          emit("draft.noplan.finished", { parts: plan.length, reason: decision.reason.slice(0, 200) });
          return null;
        }
        // Uniqueness across the whole build: parsePlanJson dedupes within one
        // call, and here every part is its own call.
        const taken = new Set(plan.map((p) => p.name));
        let name = decision.part.name;
        if (taken.has(name)) {
          let k = 2;
          while (taken.has(`${name}_${k}`)) k++;
          log(`      next-part: '${name}' is already built — renamed to '${name}_${k}'`);
          name = `${name}_${k}`;
        }
        return { ...decision.part, name };
      }
      log(`[inc-draft] no-plan: next-part reply unparseable after ${NEXT_PART_MAX_ATTEMPTS} ` +
          `attempts — ending the build here`);
      return null;
    };

    // ── Stage C: per-part generate → refine ─────────────────────────────
    // Planned mode iterates the plan. NO-PLAN mode grows `plan` one decided
    // part at a time and stops when the model says the object is done — so the
    // loop bound is not known up front and the exit is inside.
    let carriedViews: { view: string; b64: string }[] | null = null;
    for (let i = 0; ; i++) {
      if (opts.signal?.aborted) { log(`[inc-draft] aborted before part ${i + 1}`); break; }
      if (noPlan && i >= plan.length) {
        if (plan.length >= NOPLAN_HARD_CAP) {
          log(`[inc-draft] no-plan: HARD CAP of ${NOPLAN_HARD_CAP} parts reached — stopping. ` +
              `The model never declared the object complete; this build is CUT, not finished.`);
          emit("draft.noplan.capped", { parts: plan.length, cap: NOPLAN_HARD_CAP });
          break;
        }
        log(`\n[inc-draft] no-plan: deciding part ${i + 1} (${plan.length} built so far)`);
        // Renders live OUTSIDE _parts/: everything that walks that directory
        // treats each entry as a built part, and a decision render is not one.
        carriedViews = await renderContextViews(
          join(outDir, "_noplan", `next_${pad2(i + 1)}`), i);
        const chosen = await decideNextPart(i, carriedViews);
        if (!chosen) break;
        plan.push(chosen);
        // Backward-only references, enforced by exactly the code that enforces
        // them for a planned part (an appended item's "earlier" set is every
        // part already built).
        const droppedParent = enforceMotionParentOrder(plan);
        if (droppedParent.includes(chosen.name)) {
          log(`      next-part: dropped forward/unknown motion parent on ${chosen.name}`);
        }
        const droppedPartner = enforceAssemblyPartnerOrder(plan);
        if (droppedPartner.includes(chosen.name)) {
          log(`      next-part: dropped forward/unknown assembly partner on ${chosen.name}`);
        }
        writeFileSync(planJsonPath, JSON.stringify(plan, null, 2), "utf8");
        emit("draft.noplan.part_chosen", {
          index: i + 1, name: chosen.name, level: chosen.level ?? "",
        });
      }
      if (i >= plan.length) break;
      const part = plan[i]!;
      // How many parts there will be is a fact only the planned mode has; in
      // the ablation it is genuinely unknown until the model says done, and
      // printing the count decided so far would read as a total.
      const total = noPlan ? "?" : String(plan.length);
      const tag = `${pad2(i + 1)}_${part.name}`;
      const partDir = join(partsDir, tag);
      mkdirSync(partDir, { recursive: true });
      // Resume: a part with a committed after_gen.scad was already built in the
      // prior run — restore its bookkeeping and skip regeneration.
      if (resuming && existsSync(join(partDir, "after_gen.scad"))) {
        partsGenerated += 1;
        placedByPlanName.set(part.name, part.name);
        partResults.push({
          name: part.name, generated: true, refined: false, genAttempts: 0,
          placedName: part.name,
        });
        log(`[inc-draft] part ${i + 1}/${total}: ${part.name} — already built (resume), skipping`);
        continue;
      }
      const pr: IncrementalPartResult = {
        name: part.name, generated: false, refined: false, genAttempts: 0,
      };
      log(`\n[inc-draft] part ${i + 1}/${total}: ${part.name} (${part.level ?? "L?"})`);
      emit("draft.part.started", { index: i + 1, total: plan.length, name: part.name });

      // Context render: parts-colour of the build-so-far from CONTEXT_VIEWS
      // (the 6 ortho faces + the hero isometric), so the generator can pick
      // whichever views help it place the next part. Best-effort. In no-plan
      // mode the decision step above already rendered exactly this; reuse it.
      const buildViews: { view: string; b64: string }[] =
        carriedViews ?? await renderContextViews(join(partDir, "context_render"), i);
      carriedViews = null;

      // Generation + compile-fix + connectivity loop.
      let lastBrokenPart = "";
      let genErr: string | null = null;
      // Single retry-feedback slot — compile/floater/motion rejections are
      // mutually exclusive, so one value with a kind tag replaces three
      // variables. (The function-scoped lastCompileErr remains the RUN-level
      // compile error surfaced via the result's `compileError`; it is never
      // read for retry feedback.)
      let lastFailure: { kind: "compile" | "floater" | "motion" | "assembly"; detail: string } | null = null;
      let placedThisPart: string | null = null;
      for (let attempt = 1; attempt <= GEN_MAX_ATTEMPTS; attempt++) {
        pr.genAttempts = attempt;
        const cleanBuffer = stripMarkers(accumulated);
        // The lookahead. In the ablation there is none — and "(none — this is
        // the last part)", which is what an empty slice would print, is a
        // false statement there: nothing has decided that this is the last
        // part. Say what is true instead, so the generator does not close the
        // object out on a part that is merely the most recent one.
        const remaining = noPlan
          ? "(unknown — this object is being decomposed one part at a time, and the " +
            "parts after this one have not been decided yet. Build ONLY this part; " +
            "do not close the object out, and do not build ahead.)"
          : plan.slice(i + 1).map((p) => `- ${p.name}: ${p.description}`).join("\n")
            || "(none — this is the last part)";
        const userParts: CanonicalPart[] = [];
        // Base user text (byte-identical to the historical construction when
        // motionAware is off; MOVING parts get an ARTICULATION addendum).
        // (#3) Later parts that declare a mate TO this part — so it pre-builds
        // their receiving counterparts before it is frozen.
        const incomingInterfaces = assemblyAware
          ? plan.slice(i + 1)
              .filter((p) => p.assembly?.partner === part.name)
              .map((p) => ({
                name: p.name,
                ...(p.assembly?.mate ? { mate: p.assembly.mate } : {}),
                ...(p.assembly?.count ? { count: p.assembly.count } : {}),
              }))
          : [];
        let userText = buildPartGenUserText({
          text, part, remaining, cleanBuffer, motionAware, textOnly,
          ...(assemblyAddendum ? { assemblyAddendum } : {}),
          ...(incomingInterfaces.length ? { incomingInterfaces } : {}),
        });
        if (attempt > 1 && lastFailure?.kind === "compile") {
          userText +=
            "\n\n=== PRIOR ATTEMPT FAILED TO COMPILE — FIX IT ===\n" +
            "After splicing your part, OpenSCAD reported:\n\n" +
            lastFailure.detail.slice(0, 2500) +
            "\n\nYour prior part code was:\n```openscad\n" + lastBrokenPart + "\n```\n\n" +
            "Re-emit the corrected part (same four-block format). Return ONLY the block.";
        } else if (attempt > 1 && lastFailure?.kind === "floater") {
          userText +=
            "\n\n=== PRIOR ATTEMPT LEFT THE PART DETACHED — FIX IT ===\n" +
            "Your part compiled, but it did NOT connect to the rest of the model — " +
            "it landed as a free-floating solid. The connectivity check reported:\n\n" +
            lastFailure.detail.slice(0, 1500) +
            "\n\nYour prior part code was:\n```openscad\n" + lastBrokenPart + "\n```\n\n" +
            "Re-emit the part so its volume OVERLAPS an already-built neighbour — " +
            "deepen the overlap, extend it toward a neighbour, or add a connecting " +
            "strut. A merely touching face is not enough; the solids must intersect. " +
            "Return ONLY the block (same four-block format).";
        } else if (attempt > 1 && lastFailure?.kind === "motion") {
          userText +=
            "\n\n=== PRIOR ATTEMPT FAILED THE MOTION GATE — FIX IT ===\n" +
            "Your part compiled and attached, but as a MOVING part its placement " +
            "breaks articulation:\n\n" +
            lastFailure.detail.slice(0, 1200) +
            "\n\nYour prior part code was:\n```openscad\n" + lastBrokenPart + "\n```\n\n" +
            "Re-emit the part so it overlaps ONLY its joint parent and keeps visible " +
            "clearance to every other part. Return ONLY the block (same format).";
        } else if (attempt > 1 && lastFailure?.kind === "assembly") {
          userText +=
            "\n\n=== PRIOR ATTEMPT'S INTERFACE DID NOT REGISTER — FIX IT ===\n" +
            "Your part compiled and attached, but its declared mating interface did " +
            "not seat correctly on its partner:\n\n" +
            lastFailure.detail.slice(0, 1200) +
            "\n\nYour prior part code was:\n```openscad\n" + lastBrokenPart + "\n```\n\n" +
            "Re-emit the part so its mating feature seats on the partner's contact face " +
            "(reproduce the counterpart; touch it without burying into it). " +
            "Return ONLY the block (same format).";
        }
        userParts.push({ kind: "text", text: userText });
        userParts.push(...refParts("Reference image (whole object)"));
        // Image budget: the provider caps a message at 16 images. Reserve slots
        // for the reference set, then attach as many build views as still fit.
        const viewBudget = Math.max(0, 16 - refImages.length);
        const attachViews = buildViews.slice(0, viewBudget);
        if (attachViews.length) {
          userParts.push({
            kind: "text",
            text:
              `Current build so far — parts-colour from ${attachViews.length} angles ` +
              `(each already-placed part a distinct colour). Pick whichever views help ` +
              `you see where '${part.name}' must attach and overlap a neighbour:`,
          });
          for (const v of attachViews) {
            userParts.push({ kind: "text", text: `build view — ${v.view}:` });
            userParts.push({ kind: "image", data: v.b64, mimeType: "image/png" });
          }
        }

        emit("draft.part.gen_requested", { name: part.name, attempt });
        const { rawText, reasoning } = await llmGenerate(partSystem, userParts);
        writeFileSync(join(partDir, `gen_response_${attempt}.txt`), rawText, "utf8");
        if (reasoning.trim()) writeFileSync(join(partDir, `gen_thinking_${attempt}.txt`), reasoning, "utf8");
        recordTurn(`(generate part ${part.name})`, true, rawText, reasoning, {
          stage: "part-gen", name: part.name, attempt,
        });

        const parsed = parsePartResponse(rawText, part.name, { motion: motionAware, assembly: assemblyAware });
        if (!parsed) {
          // Name the truncation when that is what it was: "your code stopped
          // mid-way, emit the whole part" is something the model can act on,
          // where "no parsable part module" invites it to reformat instead.
          const cut = truncationReason(extractOpenscadCode(rawText));
          genErr = cut
            ? `response was cut off (${cut}) — emit the complete part`
            : "no parsable part module in response";
          log(`      [gen ${attempt}] ${genErr}`);
          continue;
        }
        let candidate: string; let placedName: string;
        try {
          ({ scad: candidate, placedName } = spliceParsedPart(accumulated, parsed));
        } catch (e) {
          genErr = `splice failed: ${(e as Error).message}`;
          log(`      [gen ${attempt}] ${genErr}`);
          continue;
        }
        lastBrokenPart =
          (parsed.params ? `// PARAMS\n${parsed.params}\n` : "") +
          (parsed.helpers.length ? `// HELPERS\n${parsed.helpers.join("\n\n")}\n` : "") +
          `// PART\n${parsed.partModule}\n// PLACE\n${parsed.placement}`;

        // Declared motion for this part: plan intent refined by the response's
        // optional // MOTION block (gen wins field-by-field — it saw the real
        // geometry). Parent resolves against COMMITTED parts only.
        const mergedMotion = motionAware ? mergeMotionDecl(part.motion, parsed.motion) : null;
        const parentPlaced = mergedMotion?.parent
          ? placedByPlanName.get(mergedMotion.parent)
          : undefined;
        // Declared static interface for this part: plan intent refined by the
        // response's optional // INTERFACE block. Partner resolves against
        // COMMITTED parts only (same as the motion parent).
        const mergedAssembly = assemblyAware ? mergeAssemblyDecl(part.assembly, parsed.interface) : null;
        let assemblyPartnerPlaced = mergedAssembly?.partner
          ? placedByPlanName.get(mergedAssembly.partner)
          : undefined;
        // If a gen `// INTERFACE` OVERRODE the plan partner with an unusable name
        // (typo / self / future part), keep the valid plan partner rather than
        // silently losing measurement (the plan partner is order-enforced).
        if (mergedAssembly && !assemblyPartnerPlaced && part.assembly?.partner
          && part.assembly.partner !== mergedAssembly.partner) {
          const planPlaced = placedByPlanName.get(part.assembly.partner);
          if (planPlaced) {
            mergedAssembly.warnings.push(
              `gen partner '${mergedAssembly.partner}' unresolved; kept plan partner '${part.assembly.partner}'`,
            );
            mergedAssembly.partner = part.assembly.partner;
            assemblyPartnerPlaced = planPlaced;
          }
        }

        // Compile the spliced candidate, gate on connectivity, then promote.
        try {
          const buildDir = join(outDir, "_draft_build");
          mkdirSync(buildDir, { recursive: true });
          const r = skipPartCompile
            ? null
            : await timeStage("openscad.assembly",
                () => compileScad(candidate, { outputDir: buildDir }));

          // ── Per-part connectivity gate (delta-aware) ──────────────────────
          // The freshly compiled mesh is the whole build-so-far INCLUDING this
          // part. IHMAW's contact insight, in its connectivity form: a part that
          // does not fuse into the assembly shows up as a new disconnected solid
          // (surface contact is not enough — an OpenSCAD union of merely-abutting
          // parts compiles to separate solids). We reject a part that *increases*
          // the visible-floater count, re-prompt while attempts remain, and
          // otherwise accept-with-warning rather than drop it (completion matters
          // more than a transient floater the whole-model refine can still fix).
          // Hoisted so the connectivity gate (#6) and the assembly gate can share
          // one measurement of this candidate, reused again post-commit.
          let gateMeasuredAsm: { measured: MeasuredMate | null; warnings: string[] } | null = null;
          let mateConnectivityOverride = false;
          let conn: ReturnType<typeof analyzeConnectivity> | null = null;
          if (connGateEnabled) {
            try { if (r) conn = analyzeConnectivity(loadSTL(r.stlPath)); } catch { /* non-fatal */ }
          }
          const decision = decidePartGate({
            partIndex: i,
            visibleFloatersNow: conn?.visibleFloaterCount ?? null,
            floatersBefore, attempt, maxAttempts: GEN_MAX_ATTEMPTS,
          });
          if (decision.retry) {
            // (#6) A face-seated / clearance-fit mate can read as a floater (an
            // OpenSCAD union of clearance-separated solids is disconnected) yet be
            // correctly seated. If it declares a mate to a committed partner and
            // MEASURES contact within its fit's interpenetration ceiling, accept
            // it as connected-by-mate instead of forcing overlap-deepening (which
            // would defeat the clean seating).
            if (assemblyAware && mergedAssembly && assemblyPartnerPlaced) {
              try {
                const chk = await measurePartAssembly({
                  scad: candidate, placedName, partnerPlacedName: assemblyPartnerPlaced,
                  workDir: join(partDir, "assembly"), cache: motionMeshCache,
                });
                const mm = chk.measured;
                if (mm && mm.contactAreaFrac >= ASSEMBLY_CONTACT_MIN_FRAC
                  && (!mm.interpenComputed
                    || mm.interpenetrationFrac <= interpenCeiling(mergedAssembly.mate, mergedAssembly.fit))) {
                  mateConnectivityOverride = true;
                  gateMeasuredAsm = chk; // reuse post-commit; avoids re-measuring
                  log(`      [gen ${attempt}] floater OVERRIDDEN — verified mate with ` +
                      `'${assemblyPartnerPlaced}' (contact ${(mm.contactAreaFrac * 100).toFixed(0)}%)`);
                } else {
                  motionMeshCache.delete(placedName); // stale candidate mesh
                }
              } catch { /* best-effort — fall through to the normal floater retry */ }
            }
            if (!mateConnectivityOverride) {
              lastFailure = { kind: "floater", detail: evaluateConnectivityGate(conn!).detail };
              lastCompileErr = null;
              genErr = `part '${placedName}' detached: +${decision.newFloaters} ` +
                `visible floater(s) (${floatersBefore} → ${decision.floatersAfter + decision.newFloaters})`;
              log(`      [gen ${attempt}] ${genErr} — not committing, will retry`);
              emit("draft.part.floater", { name: part.name, attempt,
                floatersBefore, newFloaters: decision.newFloaters });
              continue; // discard this candidate; re-prompt with floater feedback
            }
          }

          // ── Opt-in per-part motion gate (default OFF) ─────────────────────
          // A MOVING part must show measurable contact with its committed
          // joint parent and must not bury itself in non-parent parts. Fully
          // best-effort: a measurement failure passes, and a part is NEVER
          // dropped for motion reasons — on the last attempt it commits with a
          // warning. Gate measurement is reused post-commit (same candidate).
          let gateMeasured: PartMotionMeasurement | null = null;
          let motionGateNote: string | null = null;
          if (motionGateEnabled && mergedMotion?.moving) {
            let gateOffence: string | null = null;
            try {
              gateMeasured = await measurePartMotion({
                scad: candidate, placedName,
                ...(parentPlaced !== undefined ? { parentPlacedName: parentPlaced } : {}),
                workDir: join(partDir, "motion"), cache: motionMeshCache,
              });
              const others = [...motionMeshCache.entries()]
                .filter(([n]) => n !== placedName && n !== parentPlaced)
                .map(([n, e]) => ({ name: n, bbox: e.bbox }));
              gateOffence = evaluateMotionGate({
                parentPlacedName: parentPlaced,
                parentMeshOk: gateMeasured.parentMeshOk ?? false,
                parentContact: gateMeasured.parentContact,
                partBBox: gateMeasured.bbox,
                others,
              });
            } catch { /* best-effort — never block a commit on gate failure */ }
            if (gateOffence && attempt < GEN_MAX_ATTEMPTS) {
              motionMeshCache.delete(placedName); // candidate discarded — its mesh is stale
              lastFailure = { kind: "motion", detail: gateOffence };
              lastCompileErr = null;
              genErr = `motion gate rejected '${placedName}'`;
              log(`      [gen ${attempt}] ${genErr}: ${gateOffence.slice(0, 120)} — not committing, will retry`);
              emit("draft.part.motion_gate", { name: part.name, attempt, offence: gateOffence.slice(0, 200) });
              continue; // discard this candidate; re-prompt with motion feedback
            }
            if (gateOffence) {
              motionGateNote = gateOffence.slice(0, 400);
              log(`      [gen ${attempt}] motion gate still failing on last attempt — committing WITH WARNING`);
            }
          }

          // ── Opt-in per-part assembly gate (default OFF) ───────────────────
          // A STATIC part that declares a mate to a committed partner must show
          // measurable contact and must not bury itself in the partner. Fully
          // best-effort: a measurement failure passes, and a part is NEVER
          // dropped for assembly reasons — on the last attempt it commits with a
          // warning. The gate's measurement is reused post-commit (same candidate).
          if (!skipMateCheck && assemblyGateEnabled && mergedAssembly && !mergedMotion?.moving && assemblyPartnerPlaced) {
            let gateOffence: string | null = null;
            try {
              // Reuse the #6 connectivity-override measurement of this exact
              // candidate when it ran; otherwise measure now.
              gateMeasuredAsm = gateMeasuredAsm ?? await measurePartAssembly({
                scad: candidate, placedName, partnerPlacedName: assemblyPartnerPlaced,
                workDir: join(partDir, "assembly"), cache: motionMeshCache,
              });
              gateOffence = evaluateAssemblyGate({
                partnerPlacedName: assemblyPartnerPlaced,
                measured: gateMeasuredAsm.measured,
                ...(mergedAssembly.mate ? { mate: mergedAssembly.mate } : {}),
                ...(mergedAssembly.fit ? { fit: mergedAssembly.fit } : {}),
              });
            } catch { /* best-effort — never block a commit on gate failure */ }
            if (gateOffence && attempt < GEN_MAX_ATTEMPTS) {
              motionMeshCache.delete(placedName); // candidate discarded — its mesh is stale
              gateMeasuredAsm = null;
              lastFailure = { kind: "assembly", detail: gateOffence };
              lastCompileErr = null;
              genErr = `assembly gate rejected '${placedName}'`;
              log(`      [gen ${attempt}] ${genErr}: ${gateOffence.slice(0, 120)} — not committing, will retry`);
              emit("draft.part.assembly_gate", { name: part.name, attempt, offence: gateOffence.slice(0, 200) });
              continue; // discard this candidate; re-prompt with assembly feedback
            }
            if (gateOffence) {
              log(`      [gen ${attempt}] assembly gate still failing on last attempt — committing WITH WARNING`);
            }
          }

          // Commit: gate passed, first part, or accepted-with-warning. Publish
          // the build-so-far as the top-level deliverable: a NORMALIZED draft.obj
          // (always) and draft.stl only when exportStl (the raw build STL/OBJ
          // stay internal to _draft_build). Keeps the live intermediate honoring
          // the same OBJ-normalized / STL-opt-in invariant as the final output.
          const buildStlBytes = r ? fileSize(r.stlPath) : 0;
          if (r) {
            publishMesh({
              buildStlPath: r.stlPath, buildObjPath: r.objPath,
              objOut: objPathOut, stlOut: stlPathOut, exportStl,
              log: (m) => log(`      [gen ${attempt}] ${m}`),
            });
          }
          accumulated = candidate;
          placedThisPart = placedName;
          placedByPlanName.set(part.name, placedName);
          writeFileSync(scadPath, accumulated, "utf8");
          writeFileSync(join(partDir, "after_gen.scad"), accumulated, "utf8");
          lastCompileErr = null;
          lastFailure = null;
          pr.connected = decision.connected || mateConnectivityOverride;
          // On a #6 override the part IS a visible floater in the union (mate-
          // seated with clearance), so advance the baseline to the real count so
          // later parts near it aren't all rejected as new floaters.
          pr.floatersAfter = mateConnectivityOverride
            ? (conn?.visibleFloaterCount ?? decision.floatersAfter)
            : decision.floatersAfter;
          if (mateConnectivityOverride) {
            pr.connectivityNote = `connected via verified mate to '${assemblyPartnerPlaced}' (clearance-seated, not fused)`;
          } else if (!decision.connected) {
            pr.connectivityNote = evaluateConnectivityGate(conn!).detail.slice(0, 400);
            log(`      [gen ${attempt}] committed WITH WARNING — still +` +
                `${decision.newFloaters} floater(s) after ${GEN_MAX_ATTEMPTS} attempts`);
          }
          floatersBefore = pr.floatersAfter; // advance baseline so later parts aren't all rejected
          log(`      [gen ${attempt}] compiled (${buildStlBytes >> 10} KB build STL), ` +
              `placed as '${placedName}'${pr.connected ? "" : " [floater]"}`);

          // ── Motion declare + measure (motion-aware mode; best-effort) ─────
          // recordPartMotion mesh-measures the as-built part and appends the
          // record to the motion_incremental.json sidecar the Phase-4 planner
          // consumes. A failure here may never break the commit or the run.
          if (motionSidecar && mergedMotion) {
            try {
              const summary = await recordPartMotion({
                sidecar: motionSidecar, decl: mergedMotion,
                planName: part.name, placedName, parentPlaced,
                scad: accumulated, workDir: join(partDir, "motion"),
                cache: motionMeshCache, outDir,
                premeasured: gateMeasured, gateNote: motionGateNote, log,
              });
              pr.motion = summary;
              emit("draft.part.motion", { name: part.name, placedName,
                moving: summary.moving,
                measuredAxis: summary.measuredAxis ?? null,
                axisAgrees: summary.axisAgrees ?? null });
            } catch (e) {
              log(`      motion measure failed (non-fatal): ${(e as Error).message.slice(0, 140)}`);
            }
          }

          // ── Assembly declare + measure (assembly-aware mode; best-effort) ──
          // recordPartAssembly mesh-measures the mate vs the committed partner
          // and appends to assembly_incremental.json. A MOVING part's contact is
          // owned by the motion record (one edge per pair), so only STATIC parts
          // are recorded here. A failure may never break the commit or the run.
          if (assemblySidecar && mergedAssembly && !mergedMotion?.moving) {
            try {
              pr.assembly = await recordPartAssembly({
                sidecar: assemblySidecar, decl: mergedAssembly,
                planName: part.name, placedName, partnerPlaced: assemblyPartnerPlaced,
                scad: accumulated, workDir: join(partDir, "assembly"),
                cache: motionMeshCache, outDir, premeasured: gateMeasuredAsm, log,
              });
              emit("draft.part.assembly", { name: part.name, placedName,
                mate: pr.assembly.mate,
                partnerResolved: pr.assembly.partnerResolved,
                mates: pr.assembly.mates });
            } catch (e) {
              log(`      assembly measure failed (non-fatal): ${(e as Error).message.slice(0, 140)}`);
            }
          }
          break;
        } catch (e) {
          lastCompileErr = (e as Error).message;
          lastFailure = { kind: "compile", detail: lastCompileErr };
          genErr = lastCompileErr;
          log(`      [gen ${attempt}] compile failed: ${lastCompileErr.slice(0, 140)}`);
        }
      }

      if (!placedThisPart) {
        pr.error = genErr ?? "generation exhausted";
        partResults.push(pr);
        emit("draft.part.failed", { name: part.name, error: pr.error.slice(0, 200) });
        log(`      part '${part.name}' FAILED to generate — skipping to next part`);
        continue;
      }
      pr.generated = true;
      pr.placedName = placedThisPart;
      partsGenerated += 1;

      partResults.push(pr);
    }
  } finally {
    emit("run.finished", { runId, reason: partsGenerated > 0 ? "stop" : "error" });
    if (localWriter) await localWriter.close();
  }

  // ── Finalize: strip scaffold markers, write clean draft.scad + mesh ───
  let finalCompileOk = false;
  if (partsGenerated > 0) {
    const finalScad = stripMarkers(accumulated);
    writeFileSync(scadPath, finalScad, "utf8");
    writeFileSync(join(outDir, "response.txt"), finalScad, "utf8");
    // Motion sidecar finalize: resolve each record's instance ids against the
    // clean draft (single-placement modules keep the plain name) and re-save.
    // Best-effort — a failure here never affects the shape outputs.
    if (motionSidecar) {
      try {
        const instances = listModuleInstances(finalScad);
        for (const record of motionSidecar.records) {
          record.instanceIds = instances
            .filter((inst) => inst.module === record.placedName)
            .map((inst) => inst.instanceId);
          if (record.instanceIds.length > 1) {
            (record.warnings ??= []).push("multi_instance_statement");
          }
        }
      } catch (e) {
        motionSidecar.warnings.push(
          `finalize instance resolution failed: ${(e as Error).message.slice(0, 160)}`,
        );
      }
      try { await saveIncrementalMotionSidecar(outDir, motionSidecar); }
      catch { /* best-effort */ }
    }
    // Assembly sidecar finalize: same instance-id resolution as motion.
    if (assemblySidecar) {
      try {
        const instances = listModuleInstances(finalScad);
        for (const record of assemblySidecar.records) {
          record.instanceIds = instances
            .filter((inst) => inst.module === record.placedName)
            .map((inst) => inst.instanceId);
          if (record.instanceIds.length > 1) {
            // A reused/mirrored module is measured as ONE aggregate union solid,
            // so its verdict is not per-instance evidence — one seated and one
            // floating copy would both inherit it. Demote to unverified so the
            // seed doesn't fan an affirmative verdict across every instance.
            (record.warnings ??= []).push("multi_instance_aggregate_measurement");
            if (record.agreement && record.agreement.mates === true) {
              record.agreement = {
                mates: null,
                note: "aggregate measurement over multiple placements is not per-instance evidence",
              };
            }
          }
        }
      } catch (e) {
        assemblySidecar.warnings.push(
          `finalize instance resolution failed: ${(e as Error).message.slice(0, 160)}`,
        );
      }
      try { await saveIncrementalAssemblySidecar(outDir, assemblySidecar); }
      catch { /* best-effort */ }
    }
    try {
      const buildDir = join(outDir, "_draft_build");
      mkdirSync(buildDir, { recursive: true });
      const r = await compileScad(finalScad, { outputDir: buildDir });
      // Publish the normalized OBJ deliverable; STL only when exportStl.
      const buildStlBytes = fileSize(r.stlPath);
      publishMesh({
        buildStlPath: r.stlPath, buildObjPath: r.objPath,
        objOut: objPathOut, stlOut: stlPathOut, exportStl,
        log: (m) => log(`[inc-draft] ${m}`),
      });
      finalCompileOk = buildStlBytes > 0;
    } catch (e) {
      lastCompileErr = (e as Error).message;
      log(`[inc-draft] final compile failed: ${lastCompileErr.slice(0, 160)}`);
    }
  }

  const floaterParts = partResults.filter(
    (p) => p.generated && p.connected === false,
  ).length;
  writeFileSync(join(outDir, "parts_summary.json"), JSON.stringify({
    plan, parts: partResults, partsGenerated, floaterParts,
  }, null, 2), "utf8");

  const dur = Date.now() - t0;
  const ok = partsGenerated > 0 && finalCompileOk;
  log(`\n[inc-draft] done — ${partsGenerated}/${plan.length} parts built, ` +
      `${ok ? "ok" : "FAILED"} (${Math.round(dur / 1000)}s)`);

  return {
    ok,
    outputDir: outDir,
    imagePath,
    scadPath,
    ...(finalCompileOk && exportStl ? { stlPath: stlPathOut } : {}),
    ...(finalCompileOk && existsSync(objPathOut) ? { objPath: objPathOut } : {}),
    textPath: join(outDir, "effective_text.txt"),
    plan,
    parts: partResults,
    partsGenerated,
    durationMs: dur,
    trajectoryPath,
    sessionId,
    ...(ok ? {} : { compileError: lastCompileErr ?? "no parts generated" }),
  };
}
