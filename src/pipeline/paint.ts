/**
 * Paint stage — the optional Phase 3 material pass of the text → param3d
 * pipeline.
 *
 *   final.scad (+ image.png) → list parts → vision LLM assigns per-part PBR
 *   material → emit material-annotated deliverables.
 *
 * The shape is already finished and frozen; this stage NEVER changes geometry.
 * It keys off the top-level modules of `final.scad` (the parts), so it works
 * identically whether the shape was built incrementally or in one shot.
 *
 * One vision LLM call (no agent loop, mirroring the draft stage's harness
 * usage): the reference image + the part list go in, a JSON material map comes
 * back. From that map it writes, into the run's output dir:
 *
 *   final_materials.json   — part → { color, material, roughness, metalness, … }
 *   final_painted.scad     — final.scad with each part wrapped in color([...])
 *                            (OpenSCAD preview; colour is lost on STL export)
 *   final_painted.obj      — multi-group OBJ (one usemtl group per part),
 *   final_painted.mtl        normalized to the same unit bbox as final.obj
 *   preview_painted/       — hardcore PBR Cycles render of the painted model
 *                            (studio HDRI, edge-wear, grime, clearcoat)
 */

import { fileURLToPath } from "node:url";
import { mkdirSync, writeFileSync, readFileSync, existsSync } from "node:fs";
import { join, resolve, dirname, basename } from "node:path";

import { createHarness, applyAutoCache, createLLMClient } from "@harness/template";
import { addStage, timeStage } from "./stage-timer.ts";
import { mapPool, COMPILE_CONCURRENCY } from "../util/pool.ts";
/**
 * Attribute a paint LLM call to a per-stage bucket from its label.
 *
 * Labels carry the pass number and, when a pass is split across calls, the
 * batch index: `refine1`, `subparts2`, `subparts1_b7`. Strip BOTH so every call
 * of a stage lands in one bucket — a bare trailing-digit trim turned
 * `subparts1_b7` into `subparts1_b` and scattered a 40-call pass across as many
 * rows as it had batches.
 */
function addStageFor(label: string, ms: number): void {
  const stage = label.replace(/\d+(?:_b\d+)?$/, "");
  addStage(`llm.paint-${stage}`, ms);
}

import type { ModelRef } from "@harness/template/types";
import type {
  CanonicalRequest, CanonicalMessage, CanonicalPart,
} from "@harness/template/llm/protocol";

import { routeForModel } from "../llm/routes.ts";
import { longTimeoutFetch } from "../llm/long-timeout-fetch.ts";
import { splitThinkTags } from "../llm/think-tags.ts";
import { resolveModel, DEFAULT_MODEL } from "../config/models.ts";
import {
  listTopLevelModules, extractModuleDefinition, replaceModuleDefinition,
  compileModuleStandalone, stripCommentsAndStrings,
} from "../scad/parts.ts";
import { loadSTL } from "../mesh/stl.ts";
import { splitScadByColor } from "../render/color_split.ts";
import { splitScadToColoredParts, type SplitResult } from "../render/parts_split.ts";
import { renderPbrViews, type PbrPartSpec } from "../render/pbr.ts";
import type { RenderedView } from "../render/ao.ts";
import { DEFAULT_VIEWS, type ViewName } from "../render/views.ts";
import { writePaintedOBJ, type PaintedPart } from "../mesh/obj-mtl.ts";
import { createNoopSandbox } from "../sandbox/noop.ts";
import { createFileTrajectoryWriter } from "../trajectory/writer.ts";

const PROCEDURA_ROOT = resolve(dirname(fileURLToPath(new URL(import.meta.url))), "..", "..");
/** Each paint prompt can be pointed elsewhere by env var, so prompt variants can
 *  be A/B'd against a real run without editing the shipped file. */
function promptPath(file: string, envVar: string): string {
  const override = process.env[envVar];
  if (override && existsSync(override)) {
    console.log(`[paint] using ${envVar}=${override}`);
    return override;
  }
  if (override) console.warn(`[paint] WARNING: ${envVar}=${override} does not exist; using the shipped prompt`);
  return join(PROCEDURA_ROOT, "prompts", file);
}
const PAINT_EXTRACT_SYSTEM_PATH = promptPath("paint_extract_system.md", "PROCEDURA_PAINT_EXTRACT_PROMPT");
const PAINT_ASSIGN_SYSTEM_PATH = promptPath("paint_assign_system.md", "PROCEDURA_PAINT_ASSIGN_PROMPT");
const PAINT_REFINE_SYSTEM_PATH = promptPath("paint_refine_system.md", "PROCEDURA_PAINT_REFINE_PROMPT");
const PAINT_SUBPARTS_SYSTEM_PATH = promptPath("paint_subparts_system.md", "PROCEDURA_PAINT_SUBPARTS_PROMPT");

/** Default to the multimodal reasoning model — paint is a vision task. */
export const DEFAULT_PAINT_MODEL = DEFAULT_MODEL;
export /**
 * How many sub-part colour calls run at once.
 *
 * The sub-part stage was ONE call carrying every module's source and returning
 * every rewritten body: 140,749 bytes of response against 11,804 for the other
 * three paint calls combined, and 6.3 of a 9.9-minute paint phase. The model is
 * echoing back the whole program's geometry so that parts of it can be wrapped
 * in color() — and the code then VALIDATES that the geometry did not change,
 * so almost all of those bytes are known in advance to be unchanged source.
 *
 * Splitting it per module does not reduce the tokens, but it removes the
 * serialisation: each module is independent, so they overlap. It also isolates
 * failure (one unparseable module no longer loses the pass) and lets a module
 * that needs no sub-colouring answer cheaply instead of competing for attention
 * inside a prompt carrying seventeen others.
 */
const SUBPART_CONCURRENCY = Math.max(1, Number(process.env["PROCEDURA_PAINT_SUBPART_CONCURRENCY"] ?? 6));

/**
 * Resolution and samples for a render that only the paint-refine CRITIC will
 * look at. Such a render is overwritten by the sub-part render minutes later,
 * so it never reaches the user and never needs shipping quality: 4 views cost
 * 15s here against 50s at the 1280/350 default.
 *
 * These ARE the settings refine was measured under when it was shown to be
 * worth its cost, and what the critic judges — a wrong material over a large
 * area — is not what resolution reveals.
 */
const CRITIC_RENDER_SIZE = Math.max(256, Number(process.env["PROCEDURA_PAINT_CRITIC_SIZE"] ?? 768));
const CRITIC_RENDER_SAMPLES = Math.max(16, Number(process.env["PROCEDURA_PAINT_CRITIC_SAMPLES"] ?? 200));

const PAINT_MAX_ATTEMPTS = 3;

/** Material class → default PBR params when the model omits them. */
interface MatDefault { roughness: number; metalness: number; clearcoat: number; wear: number; dirt: number }
// Faithful defaults — moderate weathering matching a real reference (not the
// grimy extreme, not a fake-clean showroom). The model drives the real values
// from the image; these only fill gaps.
const MATERIAL_DEFAULTS: Record<string, MatDefault> = {
  metal:    { roughness: 0.32, metalness: 1.0, clearcoat: 0.0,  wear: 0.15, dirt: 0.12 },
  plastic:  { roughness: 0.4,  metalness: 0.0, clearcoat: 0.15, wear: 0.1,  dirt: 0.1  },
  rubber:   { roughness: 0.88, metalness: 0.0, clearcoat: 0.0,  wear: 0.08, dirt: 0.18 },
  glass:    { roughness: 0.08, metalness: 0.0, clearcoat: 0.0,  wear: 0.0,  dirt: 0.06 },
  ceramic:  { roughness: 0.3,  metalness: 0.0, clearcoat: 0.25, wear: 0.1,  dirt: 0.08 },
  wood:     { roughness: 0.6,  metalness: 0.0, clearcoat: 0.15, wear: 0.12, dirt: 0.12 },
  fabric:   { roughness: 0.9,  metalness: 0.0, clearcoat: 0.0,  wear: 0.08, dirt: 0.15 },
  leather:  { roughness: 0.6,  metalness: 0.0, clearcoat: 0.1,  wear: 0.12, dirt: 0.15 },
  stone:    { roughness: 0.82, metalness: 0.0, clearcoat: 0.0,  wear: 0.1,  dirt: 0.18 },
  painted:  { roughness: 0.45, metalness: 0.0, clearcoat: 0.2,  wear: 0.25, dirt: 0.15 },
  emissive: { roughness: 0.45, metalness: 0.0, clearcoat: 0.0,  wear: 0.0,  dirt: 0.0  },
};
const FALLBACK_MATERIAL: MatDefault = { roughness: 0.5, metalness: 0.0, clearcoat: 0.12, wear: 0.12, dirt: 0.1 };

export interface PartMaterial {
  name: string;
  /** 0..1 RGB. */
  color: [number, number, number];
  /** "#rrggbb" mirror of `color`. */
  hex: string;
  material: string;
  roughness: number;
  metalness: number;
  /** PBR clearcoat (glossy lacquer over the base). */
  clearcoat: number;
  /** Edge-wear / chipped-paint amount (exposes bare metal at edges). */
  wear: number;
  /** Grime / dust build-up in crevices. */
  dirt: number;
  emission: number;
  note?: string;
  /** The library material this part was assigned (Stage A → Stage B). */
  materialId?: string;
  materialName?: string;
  /** True when this part had no entry in the LLM response (defaulted). */
  defaulted?: boolean;
}

/** One material in the image-derived library (Stage A output). */
export interface MaterialLibEntry {
  id: string;
  name: string;
  color: [number, number, number];
  hex: string;
  material: string;
  roughness: number;
  metalness: number;
  clearcoat: number;
  wear: number;
  dirt: number;
  emission: number;
  /** Where in the image this material appears (free text). */
  where?: string;
}

export interface PaintOpts {
  outputDir: string;
  /** Vision model for material assignment. Default DEFAULT_PAINT_MODEL. */
  model?: string;
  /** Which SCAD/image stem to paint: "final" (default) or "draft". Falls back
   *  to the other stem if the requested one is absent. */
  stem?: "final" | "draft";
  /** Render the painted preview with Blender. Default true. */
  render?: boolean;
  /** Views for the painted preview render. Default DEFAULT_VIEWS. */
  views?: readonly ViewName[];
  /** Paint-refine steps: after the first render, show the rendered views back to
   *  the model and let it fix/improve the paint, then re-render. Each step is
   *  one critic call + one re-render. Default 1; 0 disables. Requires render. */
  refineSteps?: number;
  /** Sub-part colour pass: after the paint settles, let the model give distinct
   *  sub-features their own material (e.g. a brass wheel hub inside a painted
   *  wheel) by partitioning module geometry into colour blocks, then render via
   *  a colour-region split. Default true. */
  subparts?: boolean;
  /** Sub-part decomposition passes: each pass feeds the prior pass's
   *  already-sub-coloured modules back and pushes for finer materials.
   *
   *  Default 1. A second pass existed to recover what the first lost when every
   *  module went out in ONE request and the reply truncated; one call per module
   *  removes that. Measured on an 18-module case it costs ~3 min and lands the
   *  same mesh count, though the perceptual measure could not separate the two
   *  at n=2. Raise it for a subject with unusual material variety. */
  subpartSteps?: number;
  /** Pixel size of the SHIPPING preview render. Default 1280 (pbr.ts). The
   *  critic's own render is separate — see PROCEDURA_PAINT_CRITIC_SIZE. 2048 is
   *  the measured max-quality setting; beyond ~500 samples the denoiser eats
   *  the difference. */
  size?: number;
  /** Cycles samples for the SHIPPING preview render. Default 350 (pbr.ts). */
  samples?: number;
  log?: (line: string) => void;
  trajectorySink?: (event: import("@harness/template/trajectory").TrajectoryEvent) => void | Promise<void>;
  trajectoryPathOverride?: string;
  signal?: AbortSignal;
}

export interface PaintResult {
  ok: boolean;
  outputDir: string;
  /** The material library extracted from the reference image (Stage A). */
  palette: MaterialLibEntry[];
  parts: PartMaterial[];
  materialsPath?: string;
  palettePath?: string;
  paintedScadPath?: string;
  paintedObjPath?: string;
  paintedMtlPath?: string;
  previewDir?: string;
  /** Non-fatal degradations that cost a deliverable (empty when all is well).
   *  Surfaced so a batch can flag "painted but lossy" instead of reading ok:true. */
  degraded?: string[];
  error?: string;
  durationMs: number;
  trajectoryPath: string;
  sessionId: string;
}

function nextId(prefix: string): string {
  return `${prefix}_${Date.now().toString(36)}${Math.random().toString(36).slice(2, 8)}`;
}

function fmt4(x: number): string {
  return (Math.round(x * 10000) / 10000).toString();
}

function clamp01(x: number): number {
  return Number.isFinite(x) ? Math.min(1, Math.max(0, x)) : 0;
}

/** The per-colour material a colour mesh is rendered with. */
interface MatVals {
  color: [number, number, number];
  roughness: number; metalness: number; clearcoat: number;
  wear: number; dirt: number; emission: number;
  name: string;
  /** Material class, so the renderer can pick its transmissive path. */
  material?: string;
}

/** Canonical `[r,g,b]` key (3 decimals) — matches color_split's normalisation. */
function normColorKey(rgb: readonly number[]): string {
  return rgb.slice(0, 3).map((n) => (Math.round(n * 1000) / 1000).toString()).join(",");
}

/**
 * Resolve a colour mesh back to the material that produced it.
 *
 * The colour split keys meshes by a 3-decimal colour string, so an exact map
 * lookup misses whenever the literal in the SCAD differs from the material's
 * colour by even 0.002 — which the LLM's sub-part rewrites routinely do. Those
 * misses used to fall straight through to FALLBACK_MATERIAL, silently turning a
 * brass hub or a glass lens into grey plastic (21 of 191 shipped colour regions
 * across the hard_surface_v2 corpus). Fall back to the NEAREST known colour
 * instead, and only give up past a visible distance.
 */
function nearestMaterial(
  byColor: ReadonlyMap<string, MatVals>, key: string, rgb: readonly number[],
  tol = 0.06,
): { mv: MatVals; exact: boolean; dist: number } | null {
  const exact = byColor.get(key);
  if (exact) return { mv: exact, exact: true, dist: 0 };
  let best: MatVals | undefined;
  let bestD = Infinity;
  for (const mv of byColor.values()) {
    const d = Math.hypot(mv.color[0] - (rgb[0] ?? 0), mv.color[1] - (rgb[1] ?? 0),
                         mv.color[2] - (rgb[2] ?? 0));
    if (d < bestD) { bestD = d; best = mv; }
  }
  return best && bestD <= tol ? { mv: best, exact: false, dist: bestD } : null;
}

/** True if `name`'s body invokes another top-level module (a reuse/mirror
 *  wrapper) — such a module must NOT get an outer color() wrap, or it would
 *  override the colours the reused module already carries. */
function bodyInvokesTopLevel(scadCode: string, name: string, topLevel: readonly string[]): boolean {
  const def = extractModuleDefinition(scadCode, name);
  if (!def) return false;
  const open = def.indexOf("{");
  if (open === -1) return false;
  // Comments must not count as invocations: a `// Cutout for the wing_intake (…)`
  // note used to make a part look like a reuse wrapper, leaving it uncoloured and
  // therefore duplicated into every colour mesh. stripCommentsAndStrings is
  // index-preserving, so the slice arithmetic still lines up.
  const body = stripCommentsAndStrings(def).slice(open + 1, def.length - 1);
  for (const other of topLevel) {
    if (other === name) continue;
    if (new RegExp(`\\b${other}\\s*\\(`).test(body)) return true;
  }
  return false;
}

// ── colour parsing ──────────────────────────────────────────────────────────

function hexOf(rgb: [number, number, number]): string {
  const h = (c: number) => Math.round(clamp01(c) * 255).toString(16).padStart(2, "0");
  return `#${h(rgb[0])}${h(rgb[1])}${h(rgb[2])}`;
}

/** Parse a colour from the model: "#rgb"/"#rrggbb", "r,g,b" or [r,g,b] (0..1 or
 *  0..255). Returns null on anything unrecognisable. */
function parseColor(raw: unknown): [number, number, number] | null {
  if (Array.isArray(raw) && raw.length >= 3) {
    const a = raw.slice(0, 3).map(Number);
    if (a.some((n) => !Number.isFinite(n))) return null;
    const max = Math.max(...a);
    const s = max > 1.001 ? 1 / 255 : 1;
    return [clamp01(a[0]! * s), clamp01(a[1]! * s), clamp01(a[2]! * s)];
  }
  if (typeof raw !== "string") return null;
  let s = raw.trim().toLowerCase();
  if (s.startsWith("#")) s = s.slice(1);
  if (/^[0-9a-f]{3}$/.test(s)) {
    return [parseInt(s[0]! + s[0]!, 16) / 255, parseInt(s[1]! + s[1]!, 16) / 255, parseInt(s[2]! + s[2]!, 16) / 255];
  }
  if (/^[0-9a-f]{6}$/.test(s)) {
    return [parseInt(s.slice(0, 2), 16) / 255, parseInt(s.slice(2, 4), 16) / 255, parseInt(s.slice(4, 6), 16) / 255];
  }
  const m = s.split(",").map((t) => Number(t.trim()));
  if (m.length >= 3 && m.slice(0, 3).every((n) => Number.isFinite(n))) {
    const max = Math.max(m[0]!, m[1]!, m[2]!);
    const sc = max > 1.001 ? 1 / 255 : 1;
    return [clamp01(m[0]! * sc), clamp01(m[1]! * sc), clamp01(m[2]! * sc)];
  }
  return null;
}

/** Resolve a raw {color, material, roughness, …} object (from the LLM) into
 *  clamped PBR fields, filling class-based defaults and enforcing the
 *  metalness↔class agreement. Shared by the library extract + part assign. */
function resolveMaterialFields(e: Record<string, unknown>): {
  color: [number, number, number]; hex: string; material: string;
  roughness: number; metalness: number; clearcoat: number;
  wear: number; dirt: number; emission: number;
} {
  const matClass = (typeof e["material"] === "string" ? (e["material"] as string) : "plastic").toLowerCase();
  const def = MATERIAL_DEFAULTS[matClass] ?? FALLBACK_MATERIAL;
  const color = parseColor(e["color"]) ?? [0.7, 0.7, 0.7];
  const num = (k: string, d: number) => clamp01(typeof e[k] === "number" ? (e[k] as number) : d);
  return {
    color, hex: hexOf(color), material: matClass,
    roughness: num("roughness", def.roughness),
    metalness: matClass === "metal" ? clamp01((e["metalness"] as number) || 1.0) : 0.0,
    clearcoat: num("clearcoat", def.clearcoat),
    wear: num("wear", def.wear),
    dirt: num("dirt", def.dirt),
    emission: num("emission", matClass === "emissive" ? 0.7 : 0),
  };
}

// ── JSON extraction ───────────────────────────────────────────────────────

/** Pull the first JSON object out of an LLM response — a ```json fenced block
 *  if present, else the first brace-balanced `{...}` (string-aware). */
function extractJsonObject(text: string): string | null {
  const fenced = /```(?:json)?\s*\n?([\s\S]*?)\n?```/g;
  for (let m: RegExpExecArray | null; (m = fenced.exec(text)); ) {
    const body = m[1]!.trim();
    if (body.startsWith("{")) return body;
  }
  const start = text.indexOf("{");
  if (start === -1) return null;
  let depth = 0, inStr = false, esc = false;
  for (let i = start; i < text.length; i++) {
    const c = text[i]!;
    if (inStr) {
      if (esc) esc = false;
      else if (c === "\\") esc = true;
      else if (c === '"') inStr = false;
      continue;
    }
    if (c === '"') inStr = true;
    else if (c === "{") depth++;
    else if (c === "}") { depth--; if (depth === 0) return text.slice(start, i + 1); }
  }
  return null;
}

/**
 * Characters of module source per sub-part call. The reply carries a full
 * rewritten body for each module sent, so this bounds the RESPONSE, which is
 * what actually truncates. Measured: a ~95KB reply completed while a ~75KB one
 * was cut mid-string, so the ceiling is token- not byte-based.
 *
 * **0 (the default) means one module per call** — the finest granularity, and
 * the fastest, because every reply is then a single body and the calls overlap
 * `SUBPART_CONCURRENCY`-wide. 30000 reproduces the batched behaviour that the
 * 36-model study measured (7 same-case A/Bs, all improved) if you want to
 * compare the two; anything in between trades sibling context for reply size.
 */
const SUBPART_BATCH_CHARS = Math.max(
  0, Number(process.env["PROCEDURA_PAINT_SUBPART_BATCH_CHARS"] ?? 0),
);

/** Group modules so each batch's combined source stays under the budget.
 *  A single module larger than the budget still gets its own batch — splitting
 *  a module is not possible, and one oversized call is better than none. */
export function batchBySourceSize(
  names: string[], srcOf: (name: string) => string, budget = SUBPART_BATCH_CHARS,
): string[][] {
  const batches: string[][] = [];
  let cur: string[] = [], curLen = 0;
  for (const n of names) {
    const len = srcOf(n).length;
    if (cur.length > 0 && curLen + len > budget) { batches.push(cur); cur = []; curLen = 0; }
    cur.push(n); curLen += len;
  }
  if (cur.length > 0) batches.push(cur);
  return batches;
}

// ── colored-SCAD emission ───────────────────────────────────────────────────

/** Parameter list of a module head, with nested `()`/`[]` respected. */
function moduleParams(head: string): { text: string; names: string[] } {
  const lp = head.indexOf("(");
  if (lp === -1) return { text: "", names: [] };
  let depth = 0, rp = -1;
  for (let i = lp; i < head.length; i++) {
    const ch = head[i]!;
    if (ch === "(" || ch === "[") depth++;
    else if (ch === ")" || ch === "]") { depth--; if (depth === 0) { rp = i; break; } }
  }
  if (rp === -1) return { text: "", names: [] };
  const text = head.slice(lp + 1, rp);
  const parts: string[] = [];
  let cur = "", d = 0;
  for (const ch of text) {
    if (ch === "(" || ch === "[") d++;
    else if (ch === ")" || ch === "]") d--;
    if (ch === "," && d === 0) { parts.push(cur); cur = ""; continue; }
    cur += ch;
  }
  parts.push(cur);
  return {
    text,
    names: parts.map((s) => s.split("=")[0]!.trim())
                .filter((s) => /^[A-Za-z_$][\w$]*$/.test(s)),
  };
}

/**
 * `module N(p){body}` → `module N(p){ color(c) N__paint_geom(p) children(); }`
 * plus a verbatim `module N__paint_geom(p){body}`.
 *
 * The body must be MOVED, not wrapped in place: OpenSCAD rejects a nested
 * `module`/`function` definition inside a transform's child block, and many
 * generated parts define helpers inside their body — which used to make
 * `final_painted.scad` unparseable (7 of 22 runs in the hard_surface_v2 corpus),
 * silently costing the colour split, the painted OBJ and the final render.
 */
function wrapModuleInColor(scad: string, name: string, rgb: readonly number[]): string {
  const def = extractModuleDefinition(scad, name);
  if (!def) return scad;
  // Find the body brace on a COMMENT-BLANKED copy: a `module ring(d = 8) // bezel`
  // line comment, or a `/* body {main} */` block comment, would otherwise put the
  // brace (and with it the whole colour wrap) inside a comment. stripComments-
  // AndStrings is index-preserving, so the offsets still apply to `def`.
  const masked = stripCommentsAndStrings(def);
  const open = masked.indexOf("{");
  if (open === -1) return scad;
  const head = def.slice(0, open).trimEnd();   // "module N(a, b = 3)"
  const body = def.slice(open);                // "{ … }" verbatim
  // Defaults must be forwarded verbatim, but the NAMES have to be split off the
  // masked copy — a default like `txt = "a,b,c"` or a trailing `// width` comment
  // otherwise mis-splits the list and silently forwards the wrong arguments.
  const { text } = moduleParams(head);
  const { names } = moduleParams(masked.slice(0, open));
  const geom = `${name}__paint_geom`;
  return replaceModuleDefinition(scad, name,
    // `head` can end in a line comment, so the brace goes on its own line.
    `${head}\n{\n    color([${fmt4(rgb[0]!)}, ${fmt4(rgb[1]!)}, ${fmt4(rgb[2]!)}])\n` +
    `        ${geom}(${names.join(", ")}) children();\n}\n\n` +
    `module ${geom}(${text}) ${body}`);
}

/** Wrap every painted top-level module's BODY in `color([r,g,b])` so an
 *  OpenSCAD preview of the returned source shows the assigned colours. Geometry
 *  is untouched (color() is a no-op for CSG / STL). */
function paintScad(
  scad: string, mats: ReadonlyMap<string, [number, number, number]>,
): string {
  let out = scad;
  for (const [name, rgb] of mats) {
    out = wrapModuleInColor(out, name, rgb);
  }
  return out;
}

/** Test-only alias: exercises the real colour-wrap path from scripts/tests. */
export const __testPaintScad = paintScad;

// ── prompt building ─────────────────────────────────────────────────────────

function loadPlanDescriptions(outDir: string): Map<string, string> {
  const map = new Map<string, string>();
  const p = join(outDir, "plan.json");
  if (!existsSync(p)) return map;
  try {
    const plan = JSON.parse(readFileSync(p, "utf8")) as Array<{ name?: string; description?: string }>;
    for (const part of plan) {
      if (part?.name && part?.description) map.set(part.name, part.description);
    }
  } catch { /* best-effort */ }
  return map;
}

function buildPartListBlock(modules: string[], descs: Map<string, string>): string {
  return modules
    .map((name) => {
      const d = descs.get(name);
      return d ? `- ${name}: ${d}` : `- ${name}`;
    })
    .join("\n");
}

// ── main ─────────────────────────────────────────────────────────────────────

export async function runPaint(opts: PaintOpts): Promise<PaintResult> {
  const log = opts.log ?? ((s) => console.log(s));
  const t0 = Date.now();
  const outDir = resolve(opts.outputDir);
  mkdirSync(outDir, { recursive: true });

  // Resolve which stem to paint (prefer requested, fall back to the other).
  const order: Array<"final" | "draft"> = opts.stem === "draft" ? ["draft", "final"] : ["final", "draft"];
  let stem: "final" | "draft" | null = null;
  for (const s of order) {
    if (existsSync(join(outDir, `${s}.scad`))) { stem = s; break; }
  }
  const trajectoryPath = opts.trajectoryPathOverride ?? "";
  if (!stem) {
    return {
      ok: false, outputDir: outDir, palette: [], parts: [],
      error: `no final.scad / draft.scad found in ${outDir} to paint`,
      durationMs: Date.now() - t0, trajectoryPath, sessionId: "",
    };
  }
  const scadPath = join(outDir, `${stem}.scad`);
  const imagePath = join(outDir, "image.png");
  // A text-only run never wrote an image, so its absence is a MODE, not an
  // error — paint then derives the library from the spec. It is still an error
  // when there is no text either, because then there is nothing to paint from.
  const hasImage = existsSync(imagePath);
  const hasText = ["effective_text.txt", "prompt.txt"].some((f) => existsSync(join(outDir, f)));
  if (!hasImage && !hasText) {
    return {
      ok: false, outputDir: outDir, palette: [], parts: [],
      error: `paint needs a reference image or a text spec; ${outDir} has neither ` +
             `(no image.png, no effective_text.txt, no prompt.txt)`,
      durationMs: Date.now() - t0, trajectoryPath, sessionId: "",
    };
  }

  const scadCode = readFileSync(scadPath, "utf8");
  const modules = listTopLevelModules(scadCode);
  if (modules.length === 0) {
    return {
      ok: false, outputDir: outDir, palette: [], parts: [],
      error: `no top-level modules (parts) in ${scadPath} to paint`,
      durationMs: Date.now() - t0, trajectoryPath, sessionId: "",
    };
  }
  log(`[paint] ${stem}.scad — ${modules.length} part(s): ${modules.join(", ")}`);

  // ── route + model selection (mirrors the draft stage) ───────────────────
  const modelKey = opts.model ?? DEFAULT_PAINT_MODEL;
  const modelRef: ModelRef = resolveModel(modelKey);
  const route = routeForModel(modelKey);

  const trajectoryDir = join(outDir, "_trajectory");
  mkdirSync(trajectoryDir, { recursive: true });
  const runStamp = nextId("p").slice(2);
  const localWriter = opts.trajectorySink
    ? null
    : createFileTrajectoryWriter(trajectoryDir, `paint-${runStamp}`);
  const sink = opts.trajectorySink ?? localWriter!.sink;
  const effTrajectoryPath = opts.trajectoryPathOverride ?? localWriter!.path;

  const harness = await createHarness({
    workspace: { rootDir: outDir },
    llm: { route, client: createLLMClient({ fetch: longTimeoutFetch }) },
    sandbox: createNoopSandbox({ rootDir: outDir }),
    includeBuiltins: false,
    customTools: [],
    trajectorySink: sink,
    defaultRuleset: [{ permission: "*", pattern: "*", action: "allow" }],
  });

  const sessionId = await harness.sessions.create({
    title: `paint: ${basename(outDir)}`,
    agentKind: "paint",
    model: modelRef,
  });
  harness.trajectory.attachToBus(harness.bus, { sessionId, workspaceDir: outDir });
  const runId = nextId("run");
  harness.bus.emit("run.started", { sessionId, runId });

  let parts: PartMaterial[] = [];
  let error: string | undefined;
  /** Non-fatal but deliverable-costing problems, surfaced in PaintResult. */
  const degraded: string[] = [];

  let library: MaterialLibEntry[] = [];
  let palettePath: string | undefined;

  try {
    // ── Inputs ────────────────────────────────────────────────────────────
    const text =
      (existsSync(join(outDir, "effective_text.txt"))
        ? readFileSync(join(outDir, "effective_text.txt"), "utf8")
        : existsSync(join(outDir, "prompt.txt"))
        ? readFileSync(join(outDir, "prompt.txt"), "utf8")
        : "(no text description available)").trim();
    const descs = loadPlanDescriptions(outDir);
    const partListBlock = buildPartListBlock(modules, descs);
    // The subject block every vision call carries: the reference image, or —
    // when there is none — the text spec standing in for it. Built once so no
    // call site has to branch.
    const subjectParts: CanonicalPart[] = hasImage
      ? [
          { kind: "text", text: "Reference image of the object:" },
          { kind: "image", data: readFileSync(imagePath).toString("base64"), mimeType: "image/png" },
        ]
      : [{
          kind: "text",
          text: "There is NO reference image of this object. Work from the " +
                "description and the part names alone, choosing materials a real " +
                "example of this object would be built from:\n\n" + text,
        }];
    if (!hasImage) log(`[paint] TEXT-ONLY — no image.png; materials come from the spec`);

    // Shared JSON call: attempt loop → strip <think> → extract JSON, persisting
    // both turns for trajectory replay.
    const genJson = async (
      label: string, systemPrompt: string, userText: string,
      extraImages?: Array<{ label: string; b64: string }>,
    ): Promise<{ raw: string; reasoning: string; parsed: Record<string, unknown> | null }> => {
      const userContent: CanonicalPart[] = [
        { kind: "text", text: userText },
        ...subjectParts,
      ];
      for (const img of extraImages ?? []) {
        userContent.push({ kind: "text", text: img.label });
        userContent.push({ kind: "image", data: img.b64, mimeType: "image/png" });
      }
      const um = await harness.store.appendMessage({
        id: nextId("msg") as never, sessionId, role: "user",
        data: { text: userText, hasImage, stage: label },
      });
      harness.bus.emit("message.append", { sessionId, messageId: um.id, role: "user" });

      let raw = "", reasoning = "", parsed: Record<string, unknown> | null = null;
      for (let attempt = 1; attempt <= PAINT_MAX_ATTEMPTS; attempt++) {
        const req: CanonicalRequest = {
          model: modelRef, system: [{ text: systemPrompt }],
          messages: [{ role: "user", content: userContent } satisfies CanonicalMessage],
        };
        applyAutoCache(req, { protocolId: route.protocol.id });
        log(`[paint] ${label} attempt ${attempt}/${PAINT_MAX_ATTEMPTS} via ${modelRef.modelId}`);
        raw = ""; reasoning = "";
        try {
          const tLLM = Date.now();
          const events = await harness.llm.generate(route, req);
          addStageFor(label, Date.now() - tLLM);
          for (const ev of events) {
            if (ev.kind === "text-delta") raw += ev.text;
            else if (ev.kind === "thinking-delta") reasoning += ev.text;
            else if (ev.kind === "error") throw ev.error;
          }
        } catch (e) { log(`      [paint] ${label} attempt ${attempt} threw: ${(e as Error).message}`); continue; }
        const split = splitThinkTags(raw);
        if (split.think) reasoning += (reasoning ? "\n\n" : "") + split.think;
        const jsonStr = extractJsonObject(split.text);
        if (!jsonStr) { log(`      [paint] ${label}: no JSON found`); continue; }
        try { parsed = JSON.parse(jsonStr) as Record<string, unknown>; break; }
        catch (e) { log(`      [paint] ${label}: JSON parse failed: ${(e as Error).message}`); }
      }
      const am = await harness.store.appendMessage({
        id: nextId("msg") as never, sessionId, role: "assistant",
        data: { modelId: modelRef.modelId, providerId: modelRef.providerId, stage: label },
      });
      harness.bus.emit("message.append", { sessionId, messageId: am.id, role: "assistant" });
      if (raw) {
        const tp = await harness.store.appendPart({
          id: nextId("part") as never, messageId: am.id, sessionId, kind: "text", data: { text: raw },
        });
        harness.bus.emit("part.append", { sessionId, messageId: am.id, partId: tp.id, kind: "text" });
      }
      return { raw, reasoning, parsed };
    };

    // The per-part geometry split depends on the SCAD and NOTHING else — the
    // colours it used to be handed were never read back (they are re-applied
    // per render from `parts`). So start it here and let it compile underneath
    // Stage A and Stage B rather than waiting for them: on a large model this
    // is minutes of OpenSCAD that used to sit behind two vision calls.
    //
    // `.catch` is not optional. If a stage below throws while this is still in
    // flight, an unattached rejection takes the process down with it.
    const paintBuildDir = join(outDir, "_paint_build");
    const splitPromise: Promise<SplitResult> =
      timeStage("paint.split", () => splitScadToColoredParts({ scadCode, outDir: paintBuildDir, log }))
        .catch((e: unknown) => ({ ok: false as const, error: (e as Error).message }));

    // ── Stage A: extract a DETAILED material library from the image ───────
    harness.bus.emit("paint.extract.started", { sessionId, model: modelRef.modelId } as never);
    const extractUser =
      "=== OBJECT (text description, context only) ===\n" + text + "\n\n" +
      "=== REFERENCE IMAGE ===\nAttached below. Study it closely.\n\n" +
      "TASK: Extract the COMPLETE, DETAILED material library of this object — " +
      "every distinct colour and surface material you can see — as the strict " +
      "JSON specified in your system prompt. Be exhaustive and granular: separate " +
      "base paints, trims, metals, glass, rubber, decals, and weathering.";
    const exRes = await genJson("extract", readFileSync(PAINT_EXTRACT_SYSTEM_PATH, "utf8"), extractUser);
    writeFileSync(join(outDir, "paint_extract_response.txt"), exRes.raw, "utf8");
    const rawMats = Array.isArray((exRes.parsed as { materials?: unknown[] } | null)?.materials)
      ? ((exRes.parsed as { materials: Array<Record<string, unknown>> }).materials)
      : [];
    library = rawMats.map((e, i): MaterialLibEntry => {
      const f = resolveMaterialFields(e);
      const id = typeof e["id"] === "string" && e["id"] ? (e["id"] as string) : `m${i + 1}`;
      const name = typeof e["name"] === "string" && e["name"] ? (e["name"] as string) : f.material;
      return { id, name, ...f, ...(typeof e["where"] === "string" ? { where: e["where"] as string } : {}) };
    });
    if (library.length === 0) {
      throw new Error(`material extraction produced no materials after ${PAINT_MAX_ATTEMPTS} attempts`);
    }
    palettePath = join(outDir, `${stem}_palette.json`);
    writeFileSync(palettePath, JSON.stringify({ materials: library }, null, 2), "utf8");
    log(`[paint] extracted ${library.length} materials: ${library.map((m) => m.name).join("; ")}`);
    harness.bus.emit("paint.extract.finished", { sessionId, count: library.length } as never);

    // ── Stage B: assign each part a library material ──────────────────────
    const libById = new Map(library.map((m) => [m.id, m]));
    const libByName = new Map(library.map((m) => [m.name.toLowerCase(), m]));
    const libBlock = library.map((m) =>
      `- ${m.id} | ${m.name} | ${m.hex} | ${m.material} | rough ${m.roughness} metal ${m.metalness} coat ${m.clearcoat} wear ${m.wear} dirt ${m.dirt}` +
      (m.where ? ` | seen on: ${m.where}` : "")).join("\n");
    const assignUser =
      "=== OBJECT (text description) ===\n" + text + "\n\n" +
      "=== MATERIAL LIBRARY (extracted from the image — assign each part one by id) ===\n" +
      libBlock + "\n\n" +
      "=== PARTS IN THE MODEL (assign ONE library material to EACH, by exact name) ===\n" +
      partListBlock + "\n\n" +
      "=== REFERENCE IMAGE ===\nAttached below.\n\n" +
      "TASK: For each part, pick the library material `id` that best matches what " +
      "that part is in the image, and return the strict JSON specified in your " +
      "system prompt. Use part names verbatim; cover every part exactly once.";
    const asRes = await genJson("assign", readFileSync(PAINT_ASSIGN_SYSTEM_PATH, "utf8"), assignUser);
    writeFileSync(join(outDir, "paint_assign_response.txt"), asRes.raw, "utf8");
    if (exRes.reasoning || asRes.reasoning) {
      writeFileSync(join(outDir, "paint_thinking.txt"),
        `=== extract ===\n${exRes.reasoning}\n\n=== assign ===\n${asRes.reasoning}\n`, "utf8");
    }

    const rawAssign = Array.isArray((asRes.parsed as { parts?: unknown[] } | null)?.parts)
      ? ((asRes.parsed as { parts: Array<Record<string, unknown>> }).parts)
      : [];
    const aByName = new Map<string, Record<string, unknown>>();
    for (const e of rawAssign) {
      const n = typeof e["name"] === "string" ? (e["name"] as string) : "";
      if (n) aByName.set(n.toLowerCase(), e);
    }

    parts = modules.map((name): PartMaterial => {
      const a = aByName.get(name.toLowerCase());
      let lib: MaterialLibEntry | undefined;
      if (a) {
        const id = typeof a["materialId"] === "string" ? (a["materialId"] as string) : undefined;
        const mn = typeof a["material"] === "string" ? (a["material"] as string) : undefined;
        lib = (id ? libById.get(id) : undefined) ?? (mn ? libByName.get(mn.toLowerCase()) : undefined);
      }
      if (!lib) {
        // Assignment gave raw fields instead of an id → resolve directly; else grey.
        if (a && (a["color"] !== undefined || a["material"] !== undefined)) {
          const f = resolveMaterialFields(a);
          return { name, ...f, ...(typeof a["note"] === "string" ? { note: a["note"] as string } : {}) };
        }
        log(`      [paint] no library material for '${name}' — neutral grey`);
        const rgb: [number, number, number] = [0.7, 0.7, 0.7];
        return { name, color: rgb, hex: hexOf(rgb), material: "plastic", ...FALLBACK_MATERIAL, emission: 0, defaulted: true };
      }
      // Inherit the library material; allow small per-part wear/dirt overrides.
      const wear = a && typeof a["wear"] === "number" ? clamp01(a["wear"] as number) : lib.wear;
      const dirt = a && typeof a["dirt"] === "number" ? clamp01(a["dirt"] as number) : lib.dirt;
      const note = a && typeof a["note"] === "string" ? (a["note"] as string) : undefined;
      return {
        name, color: lib.color, hex: lib.hex, material: lib.material,
        roughness: lib.roughness, metalness: lib.metalness, clearcoat: lib.clearcoat,
        wear, dirt, emission: lib.emission, materialId: lib.id, materialName: lib.name,
        ...(note ? { note } : {}),
      };
    });

    harness.bus.emit("paint.materials", {
      sessionId,
      materials: parts.map((p) => ({ name: p.name, hex: p.hex, material: p.material })),
    } as never);

    // ── Stage C: split geometry, render, refine, then write deliverables ──
    const buildColorByName = (ps: PartMaterial[]) =>
      new Map<string, [number, number, number]>(ps.map((p) => [p.name, p.color]));

    // Split ONCE — geometry only (per-part world-position STLs). Colours/PBR are
    // driven by `parts` and re-applied each render, so geometry never
    // re-compiles. Started before Stage A; by here it is usually already done.
    const split = await splitPromise;
    const stlByName = split.ok
      ? new Map(split.parts.map((cp) => [cp.name, cp.stl]))
      : new Map<string, string>();

    const pbrSpecFrom = (ps: PartMaterial[]): PbrPartSpec[] =>
      ps.flatMap((p) => {
        const stl = stlByName.get(p.name);
        return stl ? [{
          name: p.name, stl, color: p.color, material: p.material,
          metalness: p.metalness, roughness: p.roughness, clearcoat: p.clearcoat,
          wear: p.wear, dirt: p.dirt, emission: p.emission,
        }] : [];
      });

    const previewDirPath = join(outDir, "preview_painted");
    type RenderOutcome = { ok: true; views: RenderedView[] } | { ok: false; error: string };
    /**
     * `critic: true` renders for the paint-refine critic rather than for the
     * user. That render is overwritten by the sub-part render a few minutes
     * later, so paying the shipping resolution for it buys nothing: measured
     * 4 views at 15s (768/200) against 50s (1280/350).
     *
     * 768/200 is not a guess — it is the configuration the refine pass was
     * MEASURED under when it was shown to be worth keeping (inverse-Simpson
     * 4.31 with refine against 2.24-3.25 without). What the critic looks for is
     * big-area material assignment, which does not need the resolution that
     * makes edge fillets and machining read.
     */
    const doRender = async (ps: PartMaterial[], critic = false): Promise<RenderOutcome> => {
      const specs = pbrSpecFrom(ps);
      if (specs.length === 0) return { ok: false, error: "no part geometry" };
      mkdirSync(previewDirPath, { recursive: true });
      return timeStage("paint.render", () => renderPbrViews({
        parts: specs, outDir: previewDirPath,
        views: opts.views ?? DEFAULT_VIEWS, log: (l) => log(`  ${l}`),
        ...(critic
          ? { size: CRITIC_RENDER_SIZE, samples: CRITIC_RENDER_SAMPLES }
          : {
              ...(opts.size !== undefined ? { size: opts.size } : {}),
              ...(opts.samples !== undefined ? { samples: opts.samples } : {}),
            }),
      }));
    };

    let previewDir: string | undefined;
    let lastRender: RenderOutcome | undefined;

    if (!split.ok) {
      degraded.push(`per-part split failed (${split.error}) — no render, no painted OBJ`);
      log(`[paint] WARN: per-part split failed (${split.error}); skipping render & OBJ`);
    } else if (opts.render ?? true) {
      // The v0 render has exactly one consumer: the paint-refine critic. If
      // refine is not going to run, and the sub-part pass is going to overwrite
      // the file anyway, then rendering it is producing an image that nothing
      // reads and nothing keeps. Same waste the refine loop's own intermediate
      // render had, one stage earlier.
      const refineWillRun = (opts.refineSteps ?? 1) > 0 && existsSync(PAINT_REFINE_SYSTEM_PATH);
      const subpartsWillRender = (opts.subparts ?? true)
        && existsSync(PAINT_SUBPARTS_SYSTEM_PATH);
      const needV0 = refineWillRun || !subpartsWillRender;

      // Initial render (v0).
      harness.bus.emit("paint.render.started", { sessionId } as never);
      if (!needV0) {
        log(`[paint] skipping the v0 render — no refine pass to read it, ` +
            `and the sub-part pass renders next`);
      }
      lastRender = needV0 ? await doRender(parts, true) : undefined;
      if (lastRender && !lastRender.ok) {
        // The paint-refine critic below only runs when v0 succeeded, so a
        // transient render failure silently costs the refine pass entirely —
        // and refine is the stage that fixes the BIG-AREA material assignments,
        // which dominate how varied the frame reads. The common cause is an OOM
        // (Blender exit 137) from two large assemblies rendering concurrently,
        // which a retry clears. Retry once before giving up on refine.
        log(`[paint] PBR render v0 failed (${lastRender.error}); retrying once`);
        lastRender = await doRender(parts, true);
      }
      if (lastRender?.ok) { previewDir = previewDirPath; log(`[paint] PBR render v0 (${lastRender.views.length} views)`); }
      else if (lastRender) {
        degraded.push(`PBR render v0 failed twice: ${lastRender.error} — paint-refine skipped`);
        log(`[paint] WARN: PBR render failed twice: ${lastRender.error}; refine will be skipped`);
      }

      // ── Stage D: paint-refine — the critic SEES the rendered views + the
      // reference, then fixes the paint; re-render after each step. ─────────
      const refineSteps = opts.refineSteps ?? 1;
      const refinePrompt = existsSync(PAINT_REFINE_SYSTEM_PATH) ? readFileSync(PAINT_REFINE_SYSTEM_PATH, "utf8") : "";
      for (let step = 1; step <= refineSteps && lastRender?.ok && refinePrompt; step++) {
        const viewImgs = lastRender.views.map((v) => ({
          label: `Current painted render — ${v.view} view:`,
          b64: readFileSync(v.path).toString("base64"),
        }));
        const curBlock = parts.map((p) =>
          `- ${p.name} | ${p.hex} | ${p.material} | rough ${p.roughness} metal ${p.metalness} coat ${p.clearcoat} wear ${p.wear} dirt ${p.dirt} emis ${p.emission}`).join("\n");
        const refineUser =
          "=== OBJECT ===\n" + text + "\n\n" +
          "=== CURRENT PER-PART PAINT ===\n" + curBlock + "\n\n" +
          "=== RENDERS OF THE CURRENT PAINT (attached after the reference image) ===\n" +
          "Study all four views and compare them to the REFERENCE image. The goal is to " +
          "make the painted model look like the reference.\n\n" +
          "TASK: Critique the current paint against the reference, then return the CORRECTED " +
          "full per-part material list (strict JSON per your system prompt). Fix any colour " +
          "that is the wrong hue or too grey/washed-out vs the reference, fix wrong materials, " +
          "and match the reference's finish/weathering (lower dirt on parts that look grimier " +
          "than the reference). Cover every part exactly once, names verbatim.";
        harness.bus.emit("paint.refine.started", { sessionId, step } as never);
        const ref = await genJson(`refine${step}`, refinePrompt, refineUser, viewImgs);
        writeFileSync(join(outDir, `paint_refine${step}_response.txt`), ref.raw, "utf8");
        const rawFixed = Array.isArray((ref.parsed as { parts?: unknown[] } | null)?.parts)
          ? (ref.parsed as { parts: Array<Record<string, unknown>> }).parts : [];
        if (rawFixed.length === 0) { log(`[paint] refine ${step}: no usable JSON; keeping prior paint`); break; }
        const fByName = new Map<string, Record<string, unknown>>();
        for (const e of rawFixed) { const n = typeof e["name"] === "string" ? (e["name"] as string) : ""; if (n) fByName.set(n.toLowerCase(), e); }
        let changed = 0;
        parts = parts.map((p): PartMaterial => {
          const e = fByName.get(p.name.toLowerCase());
          if (!e || (e["color"] === undefined && e["material"] === undefined && e["wear"] === undefined && e["dirt"] === undefined && e["clearcoat"] === undefined && e["emission"] === undefined)) return p;
          const f = resolveMaterialFields({ material: p.material, roughness: p.roughness, metalness: p.metalness, clearcoat: p.clearcoat, wear: p.wear, dirt: p.dirt, emission: p.emission, color: p.hex, ...e });
          changed++;
          const note = typeof e["note"] === "string" ? (e["note"] as string) : p.note;
          return {
            name: p.name, ...f,
            ...(p.materialId ? { materialId: p.materialId } : {}),
            ...(p.materialName ? { materialName: p.materialName } : {}),
            ...(note ? { note } : {}),
          };
        });
        // Only re-render if something will READ the result: the next refine
        // step, or the final preview when the sub-part stage is not going to
        // overwrite it anyway. In the default config (refineSteps 1, subparts
        // on) this render was produced and then immediately overwritten.
        const willReRender = (opts.subparts ?? true) && existsSync(PAINT_SUBPARTS_SYSTEM_PATH) && split.ok;
        const needRender = step < refineSteps || !willReRender;
        log(`[paint] refine ${step}: updated ${changed}/${parts.length} parts` +
            (needRender ? "; re-rendering" : "; skipping render (sub-part pass renders next)"));
        if (needRender) {
          // Feeding another critic step -> critic grade. Being the last render
          // anyone will see (no sub-part pass to redo it) -> shipping grade.
          lastRender = await doRender(parts, step < refineSteps);
          if (lastRender.ok) previewDir = previewDirPath;
        }
        harness.bus.emit("paint.refine.finished", { sessionId, step, changed } as never);
      }
      if (lastRender?.ok) harness.bus.emit("paint.render.finished", { sessionId, views: lastRender.views.length } as never);
    }

    // ── Stage E: sub-part colour pass — give distinct sub-features (a brass
    // wheel hub, a metal lens bezel…) their own material by partitioning module
    // geometry into sibling colour blocks, then rendering via a colour split. ──
    let coloredScad = paintScad(scadCode, buildColorByName(parts));   // whole-module colours
    let colorMeshParts: PaintedPart[] | undefined;                   // set when sub-parts land
    let subpartCount = 0;

    if ((opts.subparts ?? true) && existsSync(PAINT_SUBPARTS_SYSTEM_PATH) && split.ok) {
      try {
        const partByName = new Map(parts.map((p) => [p.name, p]));
        // Built from the library AND the refined per-part colours, at 4 decimals.
        // At 2 dp the model echoes a rounded literal that then fails to key back
        // to its material; and quoting only the library hides the refine pass's
        // corrections, so sub-parts were being coloured from pre-refine values.
        const refinedByHex = new Map(parts.map((pp) => [pp.hex, pp]));
        const libBlock2 = [
          ...library.map((m) => `- ${m.id} | ${m.name} | ${m.hex} | rgb [${m.color.map((c) => c.toFixed(4)).join(", ")}]`),
          ...[...refinedByHex.values()]
            .filter((pp) => !library.some((m) => m.hex === pp.hex))
            .map((pp) => `- ${pp.materialId ?? "part"} | ${pp.materialName ?? pp.material} (as painted on ${pp.name}) | ${pp.hex} | rgb [${pp.color.map((c) => c.toFixed(4)).join(", ")}]`),
        ].join("\n");
        const subPrompt = readFileSync(PAINT_SUBPARTS_SYSTEM_PATH, "utf8");
        const rewrites = new Map<string, string>();      // moduleName → latest rewritten body
        // The pre-rewrite mesh of a module depends only on the ORIGINAL source
        // and the name, neither of which changes, so compile it at most once
        // per module rather than once per candidate per pass.
        // Memoised on the PROMISE, not the value: validators run concurrently, so
        // two of them asking for the same module before either finishes would
        // otherwise start the same compile twice.
        const origMesh = new Map<string, Promise<string | null>>();
        const origMeshFor = (name: string): Promise<string | null> => {
          const hit = origMesh.get(name);
          if (hit) return hit;
          const pending = timeStage("paint.subpart_validate",
            () => compileModuleStandalone(scadCode, name, join(outDir, "_subpart_val", name + "_o")));
          origMesh.set(name, pending);
          return pending;
        };
        const subColorToMat = new Map<string, MatVals>();
        let working = scadCode;                          // accumulates accepted rewrites across passes
        // ONE pass by default. A second pass existed to recover what the first
        // one lost: when every module went out in a single request, the reply
        // truncated and the model rationed its palette, so re-asking picked up
        // the leftovers. One call per module removes both failure modes, and
        // with them the reason for the second pass.
        //
        // Measured on `assault_buggy`, same model, same everything else:
        // 1 pass -> 37 distinct colours in 5.8 min; 2 passes -> 37 in 8.4 min,
        // and a third run of the 2-pass config gave 34, which puts both inside
        // the +-3 run-to-run spread the 36-model study measured. Pass 2 rewrote
        // all 18 modules again and produced the same palette for 31% of the
        // wall clock.
        //
        // This is one model. `--subpart-steps 2` is one flag away if a subject
        // with far more material variety turns out to need it.
        const subSteps = Math.max(1, opts.subpartSteps ?? 1);

        for (let sstep = 1; sstep <= subSteps; sstep++) {
          const furtherNote = sstep === 1 ? "" :
            "NOTE: some modules already contain color([...]) blocks from an earlier pass. Go " +
            "FINER — split out sub-materials still missing (small hardware, edge trim, seals, " +
            "rivets, seams, glass, wiring), and further partition any block that still mixes " +
            "materials. Do not undo existing detail; return the fully re-partitioned bodies.\n\n";

          // The reply must carry a FULL rewritten body for every module sent, so
          // the response — not the prompt — is what overruns the model's output
          // ceiling. One call for the whole model truncated 3/3 times on a 363KB
          // SCAD (259 open braces vs 254 closed, cut mid-string) and returned
          // ZERO sub-part colour on exactly the models that need it most; even
          // when it completed, the model visibly rationed its palette across the
          // modules it was juggling. `batchBySourceSize` bounds each reply, and
          // the batches are INDEPENDENT, so they run concurrently rather than
          // one after another. Granularity is one knob: 0 (the default) puts
          // each module in its own call — the finest split and the fastest,
          // since every reply is then a single body.
          //
          // All calls in a pass see the same `working` snapshot. A later batch
          // therefore does not see an earlier one's accepts, which is the price
          // of running them concurrently; passes are still sequential, so pass 2
          // sees everything pass 1 landed.
          const batches = batchBySourceSize(modules, (n) => extractModuleDefinition(working, n) ?? "");
          harness.bus.emit("paint.subparts.started", { sessionId, step: sstep } as never);
          log(`[paint] sub-part pass ${sstep}: ${modules.length} module(s) in ${batches.length} call(s), ` +
              `${Math.min(SUBPART_CONCURRENCY, batches.length)} at a time`);

          const results = await mapPool(batches, SUBPART_CONCURRENCY, async (batch, bi) => {
            const defsBlock = batch.map((name) => {
              const pm = partByName.get(name);
              return `### module ${name} — base ${pm?.hex ?? "#cccccc"} ` +
                     `(${pm?.materialName ?? pm?.material ?? "?"})\n${extractModuleDefinition(working, name) ?? ""}`;
            }).join("\n\n");
            const subUser =
              "=== MATERIAL LIBRARY (pick sub-materials from these) ===\n" + libBlock2 + "\n\n" +
              "=== MODULES (name, base material, current SCAD source) ===\n" + defsBlock + "\n\n" +
              "=== REFERENCE IMAGE ===\nAttached below.\n\n" + furtherNote +
              "TASK: For any module ABOVE whose geometry contains a sub-feature of a DIFFERENT " +
              "material than the part base, rewrite that module into sibling color([...]) blocks " +
              "per the strict JSON in your system prompt. Return only modules listed above — no " +
              "others. If every solid in them is already the base material, return " +
              "{\"modules\": []}; that is a valid and expected answer.";
            const tag = batches.length > 1 ? `subparts${sstep}_b${bi + 1}` : `subparts${sstep}`;
            try {
              const r = await genJson(tag, subPrompt, subUser);
              return { tag, batch, raw: r.raw, parsed: r.parsed };
            } catch (e) {
              log(`[paint] ${tag}: call failed (${(e as Error).message.slice(0, 80)}); skip`);
              return null;
            }
          });
          writeFileSync(join(outDir, `paint_subparts${sstep}_response.txt`),
            results.filter(Boolean).map((r) => `### ${r!.tag}: ${r!.batch.join(", ")}\n${r!.raw}`).join("\n\n"),
            "utf8");

          const rawMods = results.filter(Boolean).flatMap((r) =>
            Array.isArray((r!.parsed as { modules?: unknown[] } | null)?.modules)
              ? (r!.parsed as { modules: Array<Record<string, unknown>> }).modules : []);
          // One candidate per module — a later batch returning the same module
          // wins, and a module the model invented is REPORTED rather than
          // dropped in silence (that bare `continue` was the last unlogged path
          // in this loop).
          const candidates = new Map<string, Record<string, unknown>>();
          for (const rm of rawMods) {
            const name = typeof rm["name"] === "string" ? (rm["name"] as string) : "";
            const body = typeof rm["body"] === "string" ? (rm["body"] as string) : "";
            if (!name || !body) continue;
            if (!modules.includes(name)) {
              log(`[paint] subpart: ignoring "${name}" — not a top-level module`);
              continue;
            }
            candidates.set(name, rm);
          }

          // Every candidate is validated by compiling THAT module alone against
          // the same `working` snapshot, so the checks are independent and run
          // concurrently. They used to run one at a time, which on a 30-module
          // model meant 30 serial OpenSCAD processes for a step whose only job
          // is to confirm that colouring changed no geometry.
          const verdicts = await mapPool([...candidates.entries()], COMPILE_CONCURRENCY,
            async ([name, rm]) => {
              const body = rm["body"] as string;
              const spliced = (() => { try { return replaceModuleDefinition(working, name, body); } catch { return null; } })();
              if (!spliced) return { name, rm, ok: false, why: "splice failed" };
              // Validate against the ORIGINAL geometry (colour never changes
              // tris). The original compile depends only on scadCode + name,
              // both invariant, so it is memoised across passes and candidates.
              const [oStl, nStl] = await Promise.all([
                origMeshFor(name),
                timeStage("paint.subpart_validate",
                  () => compileModuleStandalone(spliced, name, join(outDir, "_subpart_val", name + `_n${sstep}`))),
              ]);
              if (!oStl || !nStl) return { name, rm, ok: false, why: "compile failed" };
              const oT = loadSTL(oStl).triCount, nT = loadSTL(nStl).triCount;
              if (oT === 0 || Math.abs(nT - oT) / oT > 0.2) {
                return { name, rm, ok: false, why: `geometry changed (${oT}→${nT})` };
              }
              return { name, rm, ok: true, why: "" };
            });

          // Fold the survivors in a fixed order. Splicing mutates `working`, so
          // this half stays sequential — but it is pure string work.
          let accepted = 0;
          for (const v of verdicts) {
            if (!v.ok) { log(`[paint] subpart ${v.name}: ${v.why}; skip`); continue; }
            const body = v.rm["body"] as string;
            working = replaceModuleDefinition(working, v.name, body);
            rewrites.set(v.name, body);
            for (const sp of (Array.isArray(v.rm["subparts"]) ? v.rm["subparts"] as Array<Record<string, unknown>> : [])) {
              const rgb = parseColor(sp["rgb"]); if (!rgb) continue;
              const pe = library.find((m) => m.id === (typeof sp["materialId"] === "string" ? sp["materialId"] : ""));
              subColorToMat.set(normColorKey(rgb), pe
                ? { color: rgb, roughness: pe.roughness, metalness: pe.metalness, clearcoat: pe.clearcoat, wear: pe.wear, dirt: pe.dirt, emission: pe.emission, name: pe.name, material: pe.material }
                : { color: rgb, ...FALLBACK_MATERIAL, emission: 0, name: "sub-material" });
            }
            accepted++;
          }
          log(`[paint] sub-part pass ${sstep}/${subSteps}: ${accepted} module(s) (re)written`);
          if (accepted === 0 && sstep > 1) break;   // converged — no further detail found
        }
        subpartCount = rewrites.size;

        if (subpartCount > 0) {
          // `working` already carries the rewritten (sub-coloured) modules; wrap
          // the remaining plain modules in their base colour, leaving reuse
          // wrappers uncoloured so they inherit the reused module's colours.
          let cs = working;
          for (const name of modules) {
            if (rewrites.has(name)) continue;
            if (bodyInvokesTopLevel(scadCode, name, modules)) continue;   // reuse wrapper
            const rgb = partByName.get(name)?.color ?? [0.7, 0.7, 0.7] as [number, number, number];
            cs = wrapModuleInColor(cs, name, rgb);
          }
          coloredScad = cs;

          const cmeshes = await timeStage("paint.color_split",
            () => splitScadByColor(coloredScad, join(outDir, "_paint_color")));
          if (cmeshes.length === 0) {
            // The colour split is what carries sub-part paint into the render and
            // the OBJ. Zero meshes after a successful rewrite means the painted
            // SCAD did not compile — a real data loss, not a cosmetic warning.
            degraded.push(
              `colour split produced 0 meshes from ${subpartCount} rewritten module(s) — ` +
              `${stem}_painted.scad likely does not compile; sub-part paint lost`,
            );
            log(`[paint] ERROR: colour split produced 0 meshes (sub-part paint lost)`);
          }
          const matByColor = new Map<string, MatVals>();
          for (const p of parts) matByColor.set(normColorKey(p.color), { color: p.color, roughness: p.roughness, metalness: p.metalness, clearcoat: p.clearcoat, wear: p.wear, dirt: p.dirt, emission: p.emission, name: p.materialName ?? p.material, material: p.material });
          for (const [k, v] of subColorToMat) matByColor.set(k, v);

          const used = new Set<string>();
          const uniq = (b: string) => { let n = b || "color"; let i = 2; while (used.has(n)) n = `${b}_${i++}`; used.add(n); return n; };
          let inexact = 0, unmatched = 0;
          const cparts = cmeshes.map((cm) => {
            const hit = nearestMaterial(matByColor, cm.key, cm.color);
            if (hit && !hit.exact) {
              inexact++;
              log(`      [paint] colour ${cm.key} matched '${hit.mv.name}' at distance ${hit.dist.toFixed(3)}`);
            } else if (!hit) {
              unmatched++;
              log(`      [paint] WARN: colour ${cm.key} matches no material — neutral fallback`);
            }
            const mv = hit?.mv ?? { color: cm.color, ...FALLBACK_MATERIAL, emission: 0, name: "color" };
            return { name: uniq((mv.name || "color").replace(/[^A-Za-z0-9_]+/g, "_").replace(/^_+|_+$/g, "")), stl: cm.stl, mv };
          });
          colorMeshParts = cparts.map((c) => ({ name: c.name, stlPath: c.stl, color: c.mv.color, roughness: c.mv.roughness, metalness: c.mv.metalness, emission: c.mv.emission }));
          log(`[paint] sub-parts: ${subpartCount} module(s) rewritten → ${cmeshes.length} colour meshes` +
              (inexact ? `, ${inexact} nearest-matched` : "") +
              (unmatched ? `, ${unmatched} UNMATCHED` : ""));
          if (unmatched > 0) {
            degraded.push(`${unmatched}/${cmeshes.length} colour region(s) matched no material and fell back to neutral grey`);
          }
          harness.bus.emit("paint.subparts.finished", { sessionId, modules: subpartCount, meshes: cmeshes.length } as never);

          if (opts.render ?? true) {
            const specs: PbrPartSpec[] = cparts.map((c) => ({ name: c.name, stl: c.stl, color: c.mv.color, material: c.mv.material, metalness: c.mv.metalness, roughness: c.mv.roughness, clearcoat: c.mv.clearcoat, wear: c.mv.wear, dirt: c.mv.dirt, emission: c.mv.emission }));
            mkdirSync(previewDirPath, { recursive: true });
            const rr = await timeStage("paint.render", () => renderPbrViews({ parts: specs, outDir: previewDirPath, views: opts.views ?? DEFAULT_VIEWS, log: (l) => log(`  ${l}`) }));
            if (rr.ok) { previewDir = previewDirPath; log(`[paint] sub-part render → ${previewDirPath}`); }
            else {
              degraded.push(`sub-part render failed: ${rr.error}`);
              log(`[paint] WARN: sub-part render failed: ${rr.error}`);
            }
          }
        } else {
          // Zero rewrites means the sub-part pass — the whole point of the deep
          // bundle — silently no-op'd: the model returned nothing parseable, or
          // every rewrite was rejected. The run still succeeds and still ships a
          // painted SCAD, just at per-part colour separation instead of sub-part.
          // Without this the loss is invisible to any caller reading `degraded`.
          degraded.push(
            `sub-part pass produced 0 rewritten modules over ${subSteps} step(s) — ` +
            `shipping per-part colours only (no sub-part colour separation)`,
          );
          log(`[paint] WARN: sub-parts: no modules sub-coloured (per-part colours only)`);
        }
      } catch (e) {
        degraded.push(`sub-part pass failed: ${(e as Error).message}`);
        log(`[paint] WARN: sub-part pass failed: ${(e as Error).message}`);
      }
    }

    // Skipping the v0 render assumed the sub-part pass would produce the
    // shipped one. If it did not — nothing sub-coloured, or the pass threw —
    // there is no preview at all, so render the per-part paint here. Before the
    // skip, v0 was this fallback by accident; now it is one on purpose.
    if ((opts.render ?? true) && split.ok && !previewDir) {
      log(`[paint] no preview yet — rendering the per-part paint`);
      const fb = await doRender(parts);
      if (fb.ok) previewDir = previewDirPath;
      else {
        degraded.push(`no painted preview rendered: ${fb.error}`);
        log(`[paint] WARN: fallback render failed: ${fb.error}`);
      }
    }

    // ── Write deliverables from the FINAL paint ───────────────────────────
    const materialsPath = join(outDir, `${stem}_materials.json`);
    writeFileSync(materialsPath, JSON.stringify({ stem, palette: library, parts, subpartModules: subpartCount }, null, 2), "utf8");
    log(`[paint] wrote ${basename(materialsPath)}`);

    const paintedScadPath = join(outDir, `${stem}_painted.scad`);
    writeFileSync(paintedScadPath, coloredScad, "utf8");
    log(`[paint] wrote ${basename(paintedScadPath)}`);

    // OBJ + MTL — from the colour meshes when sub-parts landed, else per-part.
    let paintedObjPath: string | undefined;
    let paintedMtlPath: string | undefined;
    const objParts: PaintedPart[] = (colorMeshParts?.length ? colorMeshParts : undefined)
      ?? parts.flatMap((p) => {
      const stl = stlByName.get(p.name);
      return stl ? [{ name: p.name, stlPath: stl, color: p.color, roughness: p.roughness, metalness: p.metalness, emission: p.emission }] : [];
    });
    if (objParts.length > 0) {
      const objOut = join(outDir, `${stem}_painted.obj`);
      const mtlOut = join(outDir, `${stem}_painted.mtl`);
      try {
        const r = writePaintedOBJ({ parts: objParts, objOut, mtlOut });
        paintedObjPath = r.objPath; paintedMtlPath = r.mtlPath;
        log(`[paint] wrote ${basename(objOut)} + ${basename(mtlOut)} (${r.partsWritten} groups)`);
      } catch (e) {
        degraded.push(`painted OBJ write failed: ${(e as Error).message}`);
        log(`[paint] WARN: painted OBJ write failed: ${(e as Error).message}`);
      }
    }

    const durationMs = Date.now() - t0;
    log(`[paint] done in ${Math.round(durationMs / 1000)}s — ${parts.length} parts painted`);
    if (degraded.length > 0) {
      for (const d of degraded) log(`[paint] DEGRADED: ${d}`);
    }
    return {
      ok: true, outputDir: outDir, palette: library, parts,
      materialsPath, ...(palettePath ? { palettePath } : {}), paintedScadPath,
      ...(paintedObjPath ? { paintedObjPath } : {}),
      ...(paintedMtlPath ? { paintedMtlPath } : {}),
      ...(previewDir ? { previewDir } : {}),
      ...(degraded.length > 0 ? { degraded } : {}),
      durationMs, trajectoryPath: effTrajectoryPath, sessionId,
    };
  } catch (e) {
    error = (e as Error).message;
    log(`[paint] FAILED: ${error}`);
    return {
      ok: false, outputDir: outDir, palette: library, parts,
      ...(palettePath ? { palettePath } : {}),
      ...(degraded.length > 0 ? { degraded } : {}),
      error, durationMs: Date.now() - t0,
      trajectoryPath: effTrajectoryPath, sessionId,
    };
  } finally {
    harness.bus.emit("run.finished", { sessionId, runId, reason: error ? "error" : "stop" });
    if (localWriter) await localWriter.close();
    await harness.dispose();
  }
}
