/**
 * Per-module-coloured parts render — TypeScript orchestrator.
 *
 * Pipeline:
 *   1. List top-level modules in the SCAD source.
 *   2. Compile each via compileModuleInAssembly (stubs other modules but
 *      keeps the assembly intact, so each part lands at its true world pos).
 *   3. Assign a colour from a 12-step palette (cycled if more modules).
 *   4. Write `<outDir>/parts_color_meta.txt` (module → RGB → STL legend).
 *   5. Spawn Blender with `scripts/_render_parts_color_blender.py` and a
 *      `--meshes <stl>:r,g,b ...` argument per part.
 *   6. Verify the requested color-<view>.png files landed.
 */

import { fileURLToPath } from "node:url";
import { existsSync, mkdirSync, readFileSync, statSync, writeFileSync } from "node:fs";
import { join, resolve, dirname } from "node:path";

import { BLENDER_BIN } from "./ao.ts";
import type { RenderedView } from "./ao.ts";
import { DEFAULT_PALETTE, splitScadToColoredParts } from "./parts_split.ts";
import { DEFAULT_VIEWS, type ViewName } from "./views.ts";
import { addStage } from "../pipeline/stage-timer.ts";
import { deviceLine } from "./device_line.ts";

// Re-exported for back-compat with existing importers.
export { DEFAULT_PALETTE };

const SRC_RENDER_DIR = resolve(dirname(fileURLToPath(new URL(import.meta.url))));
const PROCEDURA_ROOT = resolve(SRC_RENDER_DIR, "..", "..");
export const PARTS_COLOR_RENDER_SCRIPT = join(
  PROCEDURA_ROOT, "scripts", "_render_parts_color_blender.py",
);

export interface RenderPartsColorOpts {
  scadPath: string;
  outDir: string;
  size?: number;
  samples?: number;
  palette?: ReadonlyArray<[number, number, number]>;
  /** Override per-part colour by module name (paint stage's semantic colours). */
  colorByName?: ReadonlyMap<string, [number, number, number]>;
  /** Per-part PBR material by module name — emits roughness/metalness so the
   *  render shades each part with its true finish. */
  materialByName?: ReadonlyMap<string, { roughness: number; metalness: number }>;
  /** Pre-computed split (meshSpecs + legend). When given, the SCAD is NOT
   *  re-split/re-compiled — used by the paint stage to avoid a second per-part
   *  compile pass after it already split for the OBJ export. */
  preSplit?: { meshSpecs: string[]; legend: string };
  gpu?: boolean;
  /**
   * Blender render engine. "cycles" (default) is unchanged. "eevee" is EEVEE
   * Next with raytraced shadows + Fast GI: measured ~2.8x cheaper PER VIEW
   * (0.164s vs 0.453s at 512px), with identical fixed startup/import cost — so
   * it only pays off where per-view render, not mesh import, dominates. It
   * resolves fine crevice detail less crisply, which is why it is opt-in.
   * Defaults from PROCEDURA_RENDER_ENGINE so a run can be switched without a code
   * change.
   */
  engine?: "cycles" | "eevee";
  /** Recompile the per-part meshes at this `$fn` — see parts_split.ts. Only for
   *  throwaway renders; never for meshes that ship or are judged. */
  fnOverride?: number;
  /** Write the bulk `_parts/` STL scratch here instead of under outDir — see
   *  parts_split.ts. Keeps 3.3 GB per run off a slow shared mount. */
  partsScratchDir?: string;
  edges?: boolean;
  views?: readonly ViewName[];
  timeoutMs?: number;
  log?: (line: string) => void;
}

export type RenderPartsColorResult =
  | { ok: true; views: RenderedView[]; legend: string; metaPath: string }
  | { ok: false; error: string; logPath?: string };

export async function renderPartsColorViews(
  opts: RenderPartsColorOpts,
): Promise<RenderPartsColorResult> {
  const outDir = opts.outDir;
  const log = opts.log ?? (() => undefined);
  const size = opts.size ?? 640;
  const samples = opts.samples ?? 64;
  const palette = opts.palette ?? DEFAULT_PALETTE;
  const gpu = opts.gpu ?? true;
  const edges = opts.edges ?? false;
  const engine = opts.engine
    ?? (process.env["PROCEDURA_RENDER_ENGINE"] === "eevee" ? "eevee" : "cycles");
  const views = opts.views ?? DEFAULT_VIEWS;
  const timeoutMs = opts.timeoutMs ?? 900_000;

  mkdirSync(outDir, { recursive: true });

  if (!existsSync(BLENDER_BIN)) {
    return { ok: false, error: `Blender not found at ${BLENDER_BIN}` };
  }
  if (!existsSync(PARTS_COLOR_RENDER_SCRIPT)) {
    return { ok: false, error: `parts-colour script missing at ${PARTS_COLOR_RENDER_SCRIPT}` };
  }

  let meshSpecs: string[];
  let legend: string;
  if (opts.preSplit) {
    ({ meshSpecs, legend } = opts.preSplit);
  } else {
    const scadCode = readFileSync(opts.scadPath, "utf8");
    const split = await splitScadToColoredParts({
      scadCode, outDir, palette, log,
      ...(opts.fnOverride ? { fnOverride: opts.fnOverride } : {}),
      ...(opts.partsScratchDir ? { partsScratchDir: opts.partsScratchDir } : {}),
      ...(opts.colorByName ? { colorByName: opts.colorByName } : {}),
      ...(opts.materialByName ? { materialByName: opts.materialByName } : {}),
    });
    if (!split.ok) {
      return { ok: false, error: split.error };
    }
    ({ meshSpecs, legend } = split);
  }

  const metaPath = join(outDir, "parts_color_meta.txt");
  writeFileSync(metaPath, legend, "utf8");

  const args = [
    "--background",
    "--python", PARTS_COLOR_RENDER_SCRIPT, "--",
    "--out", outDir,
    "--views", views.join(","),
    "--samples", String(samples),
    "--size", String(size),
    "--engine", engine,
    "--z-up",
    "--meshes", ...meshSpecs,
  ];
  if (gpu) args.push("--gpu");
  if (edges) args.push("--edges");

  const tBlender = Date.now();
  const proc = Bun.spawn([BLENDER_BIN, ...args], { stdout: "pipe", stderr: "pipe" });
  const killer = setTimeout(() => proc.kill(), timeoutMs);
  const [stdout, stderr, exitCode] = await Promise.all([
    new Response(proc.stdout).text(),
    new Response(proc.stderr).text(),
    proc.exited,
  ]);
  clearTimeout(killer);
  addStage("blender.parts_color", Date.now() - tBlender);

  if (exitCode !== 0) {
    const logPath = join(outDir, "parts_color_blender_stderr.log");
    writeFileSync(logPath, `${stderr}\n---STDOUT---\n${stdout}\n`, "utf8");
    return {
      ok: false,
      error: `parts-colour Blender render failed (exit ${exitCode})`,
      logPath,
    };
  }

  const dev = deviceLine(stdout);
  if (dev) log(`  ${dev}`);

  const rendered: RenderedView[] = [];
  const missing: string[] = [];
  for (const v of views) {
    const p = join(outDir, `color-${v}.png`);
    if (existsSync(p) && statSync(p).size > 1024) {
      rendered.push({ view: v, path: p, sizeKb: Math.floor(statSync(p).size / 1024) });
    } else {
      missing.push(`color-${v}.png`);
    }
  }
  if (missing.length > 0) {
    return { ok: false, error: `parts-colour render incomplete — missing: ${missing.join(", ")}` };
  }
  return { ok: true, views: rendered, legend, metaPath };
}
