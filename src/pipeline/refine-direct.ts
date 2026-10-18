/**
 * Direct refine — the same context → critic → fix cycle, run as a pipeline
 * instead of as an agent.
 *
 * ## Why this replaces the agent loop
 *
 * The agent version gave the model seventeen tools and asked it to sequence
 * them itself: render, then diagnose, then measure, then pick one of six edit
 * tools, then compile, then render again, then accept or revert. Measured on a
 * 19-part assault buggy, one run spent **53 tool calls to land 3 accepted
 * edits**. Twenty-five of those calls were `read_scad` — the model paging the
 * source back in a module at a time, because we never gave it the file, while
 * the draft stage next door puts the whole buffer in every prompt. Two of six
 * cycles ended in a revert having produced nothing at all.
 *
 * None of those seven steps is a decision. The order never varies, and every
 * deviation from it was a bug we then wrote scaffolding to contain: a nudge for
 * when the model stopped early, a call cap for when nudging blew the context
 * window, a transaction log for when it forgot to accept, a facet-ratio rescue
 * for when `edit_full` silently re-authored the model at 1/30 the detail.
 *
 * So the sequence is code now, and the model does the two things that actually
 * need judgement: **what is wrong** (critic) and **what the corrected code is**
 * (patch). Two LLM calls per cycle, no tools, no transcript growth across
 * cycles — each patch call is fresh, so a long refine can't blow the window.
 *
 * ## The cycle
 *
 *   compile → render 7 views → measure connectivity
 *          → CRITIC  (vision call → prioritised issue list)
 *          → measure the flagged modules (world bboxes + pairwise gaps)
 *          → PATCH   (one call → new module bodies / placements)
 *          → apply → compile → regression gates → keep or revert
 *
 * Reverting is a plain assignment here: the buffer is a string, the previous
 * value is the previous string. That is the whole of what `edit-transaction.ts`,
 * `accept_edit` and `revert_edit` were for.
 */

import { fileURLToPath } from "node:url";
import { join } from "node:path";
import { readFileSync, writeFileSync, mkdirSync } from "node:fs";
import { basename } from "node:path";

import type { CanonicalPart } from "@harness/template/llm/protocol";

import { resolveWorkspace } from "../config/workspace.ts";
import { resolveModel, DEFAULT_MODEL } from "../config/models.ts";
import type { ModelRef } from "@harness/template/types";
import { routeForModel } from "../llm/routes.ts";
import { generateWithRetry } from "../llm/generate.ts";
import { createSessionState, pad, type SessionProceduraState } from "../tools/state.ts";
import { ensureConnectivity } from "../tools/connectivity-cache.ts";
import { buildDiagnoseLeadText } from "../tools/diagnose.ts";
import { compileScad } from "../scad/compile.ts";
import { renderPartsColorViews } from "../render/parts_color.ts";
import { listTopLevelModules, compileModuleInAssembly } from "../scad/parts.ts";
import { loadSTL, computeBBox } from "../mesh/stl.ts";
import { parsePatchResponse, applyPatch } from "./refine-patch.ts";
import { writeFinalOutputs, DEFAULT_REFINE_STEPS } from "./refine.ts";
import type { RefineOpts, RefineResult } from "./refine.ts";
import { timeStage } from "./stage-timer.ts";

const DIAGNOSE_PROMPT_PATH = fileURLToPath(new URL("./diagnose-prompt.md", import.meta.url));
const PATCH_PROMPT_PATH = fileURLToPath(new URL("./refine-patch-prompt.md", import.meta.url));

/** The fixed review set: the six ortho faces plus the hero isometric. */
const REFINE_VIEWS = [
  "front", "back", "left", "right", "top", "bottom", "isometric",
] as const;

const RENDER_SIZE = Number(process.env["PROCEDURA_FEEDBACK_RENDER_SIZE"] ?? "1024") || 1024;
/** Patch attempts per cycle: one, plus repairs when it fails to parse/compile. */
const PATCH_MAX_ATTEMPTS = 3;
/** A rewrite that drops below this fraction of the prior facet count is a
 *  regression, not a fix — the same guard that caught `edit_full` gutting. */
const MIN_FACET_RATIO = Number(process.env["PROCEDURA_REFINE_MIN_FACET_RATIO"] ?? 0.8);

/**
 * The floor used when the reviewer has asked for geometry to be REMOVED.
 *
 * The gate exists to catch a model re-authoring a part more simply than it was.
 * It cannot, on facet count alone, tell that apart from a model deleting
 * geometry that should never have been there — and deleting is sometimes
 * exactly the fix. On a GPT-5.6 run the critic reported "clipping solids
 * envelop the entire model"; the repair took it from 4,835,310 to 3,160,414
 * facets, the gate called that 65% and rejected it, and the model re-proposed
 * the same fix in the next two cycles to be rejected identically. Three of six
 * cycles spent deadlocked against a correct patch.
 *
 * So when the diagnosis itself asks for removal, the floor drops. Not removed —
 * a repair that deletes four fifths of the model is still far more likely to be
 * gutting than surgery.
 */
const REMOVAL_FACET_RATIO = Number(process.env["PROCEDURA_REFINE_REMOVAL_FACET_RATIO"] ?? 0.4);

/**
 * How many consecutive cycles may end with no accepted edit before the run
 * stops.
 *
 * A cycle that lands nothing leaves the buffer untouched, so the next cycle
 * renders the same geometry, gets the same diagnosis, and proposes the same
 * patch. On a GPT-5.6 run that loop ran five times: cycles 2 through 6 each
 * burned a critic call and three patch attempts against an unchanged model, 15
 * rejected patch calls in all, every rejection inside a cycle identical to the
 * others. 32 of 47 minutes spent re-deriving one answer that was already
 * refused.
 *
 * Two rather than one, because the critic and the patcher are both stochastic
 * and one cycle can fail by luck. Two identical failures is not luck.
 */
const MAX_BARREN_CYCLES = Math.max(1, Number(process.env["PROCEDURA_REFINE_MAX_BARREN"] ?? 2));

/** Rejection text with all numbers flattened, so "3,160,414 facets" and
 *  "3,160,428 facets" compare equal — the same complaint, not a new one. */
export function rejectionShape(text: string): string {
  return text.replace(/[\d,.]+/g, "N").slice(0, 400);
}

/**
 * Does the reviewer want geometry gone?
 *
 * Deliberately narrow: it matches words that describe UNWANTED geometry, not
 * every mention of a change. "Make the hood smaller" is not removal; "a stray
 * block envelops the cockpit" is.
 */
export function diagnosisAsksForRemoval(diagnosis: string): boolean {
  return /\b(remove|removing|removal|delete|deleting|stray|leftover|left-over|spurious|extraneous|extra solid|clipping solid|envelop|enclos\w*|encas\w*|obscur\w*|blocks? the view|should not be there|shouldn't be there)\b/i
    .test(diagnosis);
}
/** Markdown fence, built rather than written, so this source stays greppable. */
const FENCE = "`".repeat(3);

// ──────────────────────────────────────────────────────────────────────────
// Diagnosis parsing
// ──────────────────────────────────────────────────────────────────────────

/**
 * Does the diagnosis list at least one HIGH issue?
 *
 * The reviewer prompt specifies NUMBERED issue lines —
 * `1. [HIGH] [modules: a, b] <problem>. FIX: <direction>` — so the tag sits
 * after an enumerator, not at the start of the line. An earlier version of this
 * matched only a leading/bulleted tag and therefore read a diagnosis with four
 * HIGH issues as clean, ending the refine on its first cycle having changed
 * nothing. Accept a numbered, bulleted, or bare tag; require the tag to open
 * the issue so prose like "no [HIGH] issues remain" cannot trip it.
 */
export function hasHighIssue(diagnosis: string): boolean {
  return /^[ \t]*(?:\d+[.)]|[-*])?[ \t]*\[HIGH\]/im.test(diagnosis);
}

/**
 * Module names the reviewer named, in the order it named them. Matched against
 * the buffer's real module list so a hallucinated or misspelled name is dropped
 * rather than sent to the measurer.
 */
export function flaggedModules(diagnosis: string, known: readonly string[]): string[] {
  const out: string[] = [];
  const seen = new Set<string>();
  // Word-boundary scan over the whole diagnosis: the reviewer cites modules both
  // inside `[modules: …]` tags and inline in prose, and we want both.
  for (const name of known) {
    const at = new RegExp(`\\b${name}\\b`).exec(diagnosis);
    if (at) out.push(name);
  }
  // Order by first mention so the highest-severity issue's modules come first.
  out.sort((a, b) => diagnosis.indexOf(a) - diagnosis.indexOf(b));
  return out.filter((n) => (seen.has(n) ? false : (seen.add(n), true)));
}

// ──────────────────────────────────────────────────────────────────────────
// Measurement
// ──────────────────────────────────────────────────────────────────────────

interface WorldBox { min: [number, number, number]; max: [number, number, number] }

const fmt = (n: number): string => Number(n.toFixed(1)).toString();

function boxText(b: WorldBox): string {
  const size = [0, 1, 2].map((a) => b.max[a]! - b.min[a]!);
  return `x[${fmt(b.min[0])}..${fmt(b.max[0])}] y[${fmt(b.min[1])}..${fmt(b.max[1])}] ` +
    `z[${fmt(b.min[2])}..${fmt(b.max[2])}]  size ${size.map(fmt).join(" × ")} mm`;
}

function pairText(aName: string, a: WorldBox, bName: string, b: WorldBox): string {
  const axes = ["x", "y", "z"] as const;
  const parts: string[] = [];
  for (let i = 0; i < 3; i++) {
    const lo = Math.max(a.min[i]!, b.min[i]!);
    const hi = Math.min(a.max[i]!, b.max[i]!);
    const aLen = Math.max(1e-6, a.max[i]! - a.min[i]!);
    parts.push(hi >= lo
      ? `${axes[i]}: overlap ${fmt(hi - lo)}mm (${((hi - lo) / aLen * 100).toFixed(0)}% of ${aName})`
      : `${axes[i]}: GAP ${fmt(lo - hi)}mm`);
  }
  return `${aName} ↔ ${bName}:  ${parts.join("  |  ")}`;
}

/**
 * Measure the flagged modules at their true assembly placement.
 *
 * This is the step the agent had to remember to take (`module_context` with
 * `with_measurements: true`), and frequently didn't — which is how a cycle
 * ended up guessing a magnitude, missing, and spending the next cycle undoing
 * itself. Here it always runs, on exactly the modules the critic named.
 */
async function measureModules(
  scad: string, names: readonly string[], workDir: string,
): Promise<string> {
  if (names.length === 0) return "";
  const boxes = new Map<string, WorldBox>();
  for (const name of names) {
    try {
      const stl = await compileModuleInAssembly(scad, name, join(workDir, `m_${name}`));
      if (!stl) continue;
      const bb = computeBBox(loadSTL(stl));
      boxes.set(name, { min: bb.min as [number, number, number], max: bb.max as [number, number, number] });
    } catch { /* unmeasurable module — omit it rather than fail the cycle */ }
  }
  if (boxes.size === 0) return "";

  const lines = ["=== MEASURED WORLD GEOMETRY (exact — derive your numbers from these) ==="];
  for (const [name, b] of boxes) lines.push(`  ${name}:  ${boxText(b)}`);
  const entries = [...boxes.entries()];
  if (entries.length > 1) {
    lines.push("", "Pairwise (positive = interpenetration depth, GAP = air between them):");
    for (let i = 0; i < entries.length; i++) {
      for (let j = i + 1; j < entries.length; j++) {
        lines.push(`  ${pairText(entries[i]![0], entries[i]![1], entries[j]![0], entries[j]![1])}`);
      }
    }
  }
  return lines.join("\n") + "\n";
}

// ──────────────────────────────────────────────────────────────────────────
// The driver
// ──────────────────────────────────────────────────────────────────────────

export async function runDirectRefine(opts: RefineOpts): Promise<RefineResult> {
  const workspace = resolveWorkspace(opts.outputDir);
  const state = createSessionState(
    workspace,
    opts.artifactsSubdir ? { artifactsSubdir: opts.artifactsSubdir } : undefined,
  );
  const exportStl = opts.exportStl ?? false;
  const modelKey = opts.model ?? DEFAULT_MODEL;
  const model: ModelRef = resolveModel(modelKey);
  const route = routeForModel(modelKey);
  const maxCycles = opts.maxSteps ?? DEFAULT_REFINE_STEPS;
  state.maxRefineSteps = maxCycles;

  const criticSystem = readFileSync(DIAGNOSE_PROMPT_PATH, "utf8");
  const patchSystem = readFileSync(PATCH_PROMPT_PATH, "utf8");
  // In a text-only run there is no image.png, so the TARGET the critic compares
  // renders against is the spec sentence. Both call sites take this block whole,
  // so neither has to know which mode it is in.
  const referenceParts: CanonicalPart[] = workspace.hasImage
    ? [
        { kind: "text", text: "REFERENCE image (target):" },
        { kind: "image", data: readFileSync(workspace.imagePath).toString("base64"), mimeType: "image/png" },
      ]
    : [{
        kind: "text",
        text: "TARGET — there is NO reference image for this object. The text " +
              "specification is the complete and only target:\n\n" + workspace.text,
      }];

  const log = (s: string): void => console.log(s);
  log(`\n=== ${opts.bannerLabel ?? "Procedura refine (direct)"} ===`);
  log(`  workspace:     ${workspace.rootDir}`);
  log(`  model:         ${model.modelId}`);
  log(`  max-cycles:    ${maxCycles}`);
  log(`  refine-steps:  ${state.refineStepsDir}`);
  log(``);

  // The STL for the CURRENT buffer, or null when it needs (re)building.
  //
  // Deliberately not state.latestStlPath: that field is pre-seeded with
  // draft.stl, which is a LAZY-UNION export. Overlapping solids are left
  // unmerged there, so its facet count is not comparable to a normal compile's
  // — using it as the regression baseline would make the very first patch look
  // like it destroyed detail and get it thrown away. Starting at null forces
  // one honest compile before the first comparison, and every later cycle
  // reuses the STL the patch verification already produced.
  let builtStl: string | null = null;
  /** Consecutive cycles that accepted nothing. */
  let barren = 0;
  let verdict: RefineResult["verdict"] = "max-steps";
  let summary = "(no summary)";
  let accepted = 0;
  let llmCalls = 0;

  for (let cycle = 1; cycle <= maxCycles; cycle++) {
    if (opts.signal?.aborted) { verdict = "aborted"; break; }
    state.refineStep = cycle;
    const stepDir = join(state.refineStepsDir, `step_${pad(cycle)}`);
    mkdirSync(stepDir, { recursive: true });
    log(`--- cycle ${cycle}/${maxCycles} ---`);

    // ── 1. Compile + render. Not a decision: it happens every cycle. ──────
    // The compile is only needed for the facet baseline (connectivity does its
    // own union-wrapped compile, and the renderer works from the source), so
    // skip it when the buffer is already built: after an accepted patch we
    // compiled that exact buffer to verify it, and after a rejected cycle the
    // buffer never changed. That is one full compile per cycle saved on a model
    // whose STL runs to tens of megabytes.
    const compileDir = join(state.agentCompilesDir, `cycle_${pad(cycle)}`);
    if (builtStl === null) {
      mkdirSync(compileDir, { recursive: true });
      try {
        const r = await timeStage("openscad.refine", () => compileScad(state.scad, { outputDir: compileDir }));
        builtStl = r.stlPath;
        state.latestStlPath = r.stlPath;
        state.stlIsStale = false;
      } catch (e) {
        // The buffer we were handed does not build. Nothing downstream helps.
        log(`  compile failed on entry: ${(e as Error).message.slice(0, 200)}`);
        verdict = "error";
        break;
      }
    }

    const viewsDir = join(stepDir, "views");
    mkdirSync(viewsDir, { recursive: true });
    const scadOnDisk = join(viewsDir, "input.scad");
    writeFileSync(scadOnDisk, state.scad, "utf8");
    const rv = await timeStage("blender.refine", () => renderPartsColorViews({
      scadPath: scadOnDisk, outDir: viewsDir,
      views: [...REFINE_VIEWS], size: RENDER_SIZE, samples: 16,
    }));
    if (!rv.ok) {
      log(`  render failed: ${rv.error} — ending refine with what we have`);
      verdict = "error";
      break;
    }
    state.partsColorLegend = rv.legend;
    state.latestViews = rv.views.map((v) => ({ label: `CURRENT — parts-colour ${v.view}`, path: v.path }));
    state.latestViewsScad = state.scad;
    await ensureConnectivity(state);

    // ── 2. Critic. ────────────────────────────────────────────────────────
    const criticParts: CanonicalPart[] = [
      { kind: "text", text: buildDiagnoseLeadText(state, { fixerHasTools: false }) },
      ...referenceParts,
    ];
    for (const v of state.latestViews) {
      criticParts.push({ kind: "text", text: `${v.label}:` });
      criticParts.push({ kind: "image", data: readFileSync(v.path).toString("base64"), mimeType: "image/png" });
    }
    criticParts.push({
      kind: "text",
      text: "=== CURRENT SCAD CODE ===\n" + FENCE + "openscad\n" + state.scad + "\n" + FENCE + "\n\n" +
        "Return ONLY the diagnosis block in the format your system prompt specifies.",
    });

    let diagnosis: string;
    try {
      const r = await timeStage("llm.critic", () => generateWithRetry({
        route, model, system: criticSystem, parts: criticParts,
        label: `critic c${cycle}`, ...(opts.signal ? { signal: opts.signal } : {}),
      }));
      llmCalls += 1;
      diagnosis = r.text.trim();
      writeFileSync(join(stepDir, "diagnosis.txt"), diagnosis, "utf8");
      if (r.reasoning) writeFileSync(join(stepDir, "diagnose_thinking.txt"), r.reasoning, "utf8");
    } catch (e) {
      log(`  critic failed: ${(e as Error).message.slice(0, 200)} — ending refine`);
      break;
    }

    const summaryLine = diagnosis.split("\n").find((l) => /^\s*summary:/i.test(l)) ?? diagnosis.slice(0, 120);
    summary = summaryLine.replace(/^\s*summary:\s*/i, "").trim();
    state.diagnosisHistory.push({ cycle, summary, raw: diagnosis });
    log(`  critic: ${summary.slice(0, 160)}`);

    if (!hasHighIssue(diagnosis)) {
      log(`  no HIGH issues remain — finishing`);
      verdict = "ok";
      break;
    }

    // ── 3. Measure what the critic flagged. ───────────────────────────────
    const known = listTopLevelModules(state.scad);
    const flagged = flaggedModules(diagnosis, known).slice(0, 8);
    const measurements = await timeStage("openscad.measure",
      () => measureModules(state.scad, flagged, join(stepDir, "measure")));
    if (flagged.length) log(`  measured: ${flagged.join(", ")}`);

    // ── 4. Patch, with repair attempts for parse/compile failures. ────────
    const scadBefore = state.scad;
    const facetsBefore = countFacets(builtStl);
    const removalWanted = diagnosisAsksForRemoval(diagnosis);
    if (removalWanted) {
      log(`  diagnosis asks for geometry REMOVAL — facet floor relaxed to ` +
          `${(REMOVAL_FACET_RATIO * 100).toFixed(0)}% for this cycle`);
    }
    let landed = false;
    let repairNote = "";
    let repeatedRejection = false;

    for (let attempt = 1; attempt <= PATCH_MAX_ATTEMPTS; attempt++) {
      // Ordering is a COST decision, not a stylistic one. Everything stable
      // within the cycle goes first — spec, images, the ~70k-token SCAD, the
      // diagnosis, the measurements — and the one thing that changes between
      // repair attempts (the failure note) goes last. Put the note first and
      // every attempt re-bills the whole prefix; put it last and attempts 2
      // and 3 hit the prompt cache.
      const patchParts: CanonicalPart[] = [
        {
          kind: "text",
          text:
            `=== TEXT SPEC ===\n${workspace.text}\n\n` +
            (state.partsColorLegend
              ? `=== PARTS-COLOUR LEGEND (module → RGB) ===\n${state.partsColorLegend.trim()}\n\n`
              : "") +
            (workspace.hasImage
              ? "The reference image and the current build views follow, then the full "
              : "The target specification and the current build views follow, then the full ") + +
            "SCAD source, then the reviewer's diagnosis.",
        },
        ...referenceParts,
      ];
      for (const v of state.latestViews) {
        patchParts.push({ kind: "text", text: `${v.label}:` });
        patchParts.push({ kind: "image", data: readFileSync(v.path).toString("base64"), mimeType: "image/png" });
      }
      patchParts.push({
        kind: "text",
        text: "=== CURRENT SCAD SOURCE (complete) ===\n" + FENCE + "openscad\n" +
          scadBefore + "\n" + FENCE + "\n",
      });
      patchParts.push({
        kind: "text",
        text:
          `=== REVIEWER DIAGNOSIS (cycle ${cycle}) ===\n${diagnosis}\n\n` +
          (measurements ? measurements + "\n" : ""),
      });
      patchParts.push({
        kind: "text",
        text:
          (repairNote ? `=== YOUR PREVIOUS ATTEMPT FAILED — FIX IT ===\n${repairNote}\n\n` : "") +
          "Fix the highest-severity issue. Emit only the patch blocks.",
      });

      let raw: string;
      try {
        const r = await timeStage("llm.patch", () => generateWithRetry({
          route, model, system: patchSystem, parts: patchParts,
          label: `patch c${cycle}a${attempt}`, ...(opts.signal ? { signal: opts.signal } : {}),
        }));
        llmCalls += 1;
        raw = r.text;
        writeFileSync(join(stepDir, `patch_response_${attempt}.txt`), raw, "utf8");
        if (r.reasoning) writeFileSync(join(stepDir, `patch_thinking_${attempt}.txt`), r.reasoning, "utf8");
      } catch (e) {
        log(`  patch call failed: ${(e as Error).message.slice(0, 160)}`);
        break;
      }

      const reject = (why: string): void => {
        // A rejection identical to the last one means the model re-sent the
        // same patch: re-prompting again asks the same question of the same
        // inputs. Every cycle of the gpt run burned all three attempts this
        // way, the three rejections landing within fourteen triangles.
        if (repairNote && rejectionShape(repairNote) === rejectionShape(why)) {
          repeatedRejection = true;
        }
        repairNote = why;
        try {
          writeFileSync(join(stepDir, `patch_rejected_${attempt}.txt`), why, "utf8");
        } catch { /* non-fatal */ }
      };

      if (/^\s*NOCHANGE\s*$/im.test(raw)) {
        log(`  patch declined to edit (NOCHANGE) — finishing`);
        verdict = "ok";
        landed = false;
        cycle = maxCycles; // no more useful work
        break;
      }

      const parsed = parsePatchResponse(raw);
      if (!parsed) {
        reject("Your reply contained no `=== MODULE name ===` or `=== PLACE name ===` block. " +
          "Emit at least one block, in exactly that format.");
        log(`  [patch ${attempt}] unparseable — re-prompting`);
        continue;
      }

      const applied = applyPatch(scadBefore, parsed);
      if (!applied.ok) {
        reject(applied.error);
        log(`  [patch ${attempt}] rejected:\n${applied.error.split("\n").map((l) => "    " + l).join("\n")}`);
        if (repeatedRejection) { log(`  [patch ${attempt}] same rejection as the last attempt — not re-prompting`); break; }
        continue;
      }

      // Compile the patched buffer. A failure here is the model's to fix, and
      // it gets the compiler's own words.
      const patchDir = join(compileDir, `patch_${attempt}`);
      mkdirSync(patchDir, { recursive: true });   // compileDir may have been skipped
      let stlPath: string;
      try {
        const r = await timeStage("openscad.refine",
          () => compileScad(applied.scad, { outputDir: patchDir }));
        stlPath = r.stlPath;
      } catch (e) {
        reject("After applying your patch, OpenSCAD reported:\n\n" +
          (e as Error).message.slice(0, 2000));
        log(`  [patch ${attempt}] compile failed — re-prompting: ` +
            `${(e as Error).message.split("\n").find((l) => /error/i.test(l))?.slice(0, 160) ?? (e as Error).message.slice(0, 160)}`);
        if (repeatedRejection) { log(`  [patch ${attempt}] same failure as the last attempt — not re-prompting`); break; }
        continue;
      }

      // ── 5. Regression gate. ─────────────────────────────────────────────
      // A patch that compiles can still be a loss: the classic failure is a
      // "simplified" rewrite that fixes the proportion and throws away the
      // detail. Facet count is the cheap, reliable tell.
      const facetsAfter = countFacets(stlPath);
      // A repair the reviewer asked for can legitimately REMOVE geometry, and
      // facet count alone cannot tell that from gutting — so the floor follows
      // the diagnosis's intent.
      const floor = removalWanted ? REMOVAL_FACET_RATIO : MIN_FACET_RATIO;
      if (facetsBefore !== null && facetsAfter !== null && facetsAfter < facetsBefore * floor) {
        reject(
          `Your patch compiled but destroyed detail: the model went from ` +
          `${facetsBefore.toLocaleString()} to ${facetsAfter.toLocaleString()} facets ` +
          `(${(facetsAfter / facetsBefore * 100).toFixed(0)}% — the floor is ` +
          `${(floor * 100).toFixed(0)}%). You rewrote a module more simply ` +
          `than it was. Re-emit the fix carrying over every feature the reviewer did ` +
          `not complain about.`);
        log(`  [patch ${attempt}] facet regression ${facetsBefore} → ${facetsAfter} — reverting`);
        if (repeatedRejection) { log(`  [patch ${attempt}] same rejection as the last attempt — not re-prompting`); break; }
        continue;
      }

      // Accepted.
      state.scad = applied.scad;
      state.lastGoodScad = applied.scad;
      builtStl = stlPath;              // verified compile of the new buffer
      state.latestStlPath = stlPath;
      state.stlIsStale = false;
      state.completedEdits += 1;
      accepted += 1;
      landed = true;
      writeFileSync(join(stepDir, "scad.scad"), applied.scad, "utf8");
      writeFileSync(join(stepDir, "summary.json"), JSON.stringify({
        cycle, accepted: true, touched: applied.touched,
        reason: parsed.reason, attempt,
        facets: { before: facetsBefore, after: facetsAfter },
      }, null, 2), "utf8");
      log(`  [patch ${attempt}] ACCEPTED — ${applied.touched.join(", ")}`);
      if (parsed.reason) log(`      reason: ${parsed.reason.replace(/^reason:\s*/i, "").slice(0, 200)}`);
      break;
    }

    if (landed) barren = 0;
    if (!landed && verdict !== "ok") {
      barren += 1;
      log(`  cycle ${cycle} produced no accepted edit (${barren} in a row)`);
      writeFileSync(join(stepDir, "summary.json"), JSON.stringify({
        cycle, accepted: false, barren, note: repairNote.slice(0, 500),
      }, null, 2), "utf8");
      if (barren >= MAX_BARREN_CYCLES) {
        // The buffer is unchanged, so the next cycle renders the same geometry,
        // gets the same diagnosis and proposes the same patch. Stopping leaves
        // budget unspent, which is strictly better than spending it here.
        log(`  ${barren} consecutive cycles accepted nothing — stopping rather than ` +
            `re-deriving a patch that has already been refused`);
        break;
      }
    }
  }

  // ── Finalize. The buffer is whatever survived; write the deliverables. ──
  // Record the REAL verdict. Mapping everything non-ok to "give_up" would tell
  // the results server and the benchmark that the model declared itself unable
  // to improve, when in fact it simply spent its budget — and that field is
  // what those tools rank on.
  const writeResult = await writeFinalOutputs(workspace, state, {
    verdict,
    summary: `${accepted} accepted edit(s) over ${llmCalls} LLM call(s). ${summary}`,
  }, exportStl);

  log(`\n=== Done ===`);
  log(`  verdict:        ${verdict}`);
  log(`  refine cycles:  ${accepted} accepted / ${maxCycles} budget`);
  log(`  LLM calls:      ${llmCalls}`);
  log(`  outputs:        ${writeResult.scadPath}`);

  return {
    ok: verdict === "ok",
    verdict,
    summary,
    outputs: {
      scadPath: writeResult.scadPath,
      ...(writeResult.stlPath !== undefined ? { stlPath: writeResult.stlPath } : {}),
      ...(writeResult.objPath !== undefined ? { objPath: writeResult.objPath } : {}),
      ...(writeResult.previewDir !== undefined ? { previewDir: writeResult.previewDir } : {}),
      trajectoryPath: opts.trajectoryPathOverride ?? "",
      diagnosisPath: writeResult.diagnosisPath,
    },
    steps: accepted,
    toolCalls: llmCalls,
  };
}

/** Binary-STL facet count from its 80-byte header, without parsing the mesh. */
function countFacets(stlPath: string | null): number | null {
  if (!stlPath) return null;
  try {
    const fd = readFileSync(stlPath);
    if (fd.length < 84) return null;
    return fd.readUInt32LE(80);
  } catch { return null; }
}
