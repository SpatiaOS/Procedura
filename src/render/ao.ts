/**
 * Blender Cycles AO + Freestyle-edge view renderer.
 *
 * Spawns Blender in `--background` mode with our Python script asset
 * (`scripts/_render_ao_blender.py`). The script is a Blender-Python asset:
 * Blender runs it inside its embedded Python, so this lives in TS only as
 * a subprocess invocation + flag builder + output verifier.
 *
 * Binary discovery: $PROCEDURA_BLENDER_PATH, else the first BLENDER_CANDIDATES
 * entry that exists, else `blender` on $PATH.
 *
 * Failure on any of: missing binary, non-zero exit, timeout, or a missing
 * (or trivially small) output PNG. Stderr is preserved at
 * `<outDir>/blender_stderr.log` for diagnosis.
 */

import { fileURLToPath } from "node:url";
import { existsSync, mkdirSync, statSync, writeFileSync } from "node:fs";
import { join, resolve, dirname } from "node:path";

import { DEFAULT_VIEWS, type ViewName } from "./views.ts";
import { addStage } from "../pipeline/stage-timer.ts";
import { deviceLine } from "./device_line.ts";

// Re-exported for back-compat with existing importers.
export { DEFAULT_VIEWS };
/** @deprecated use ViewName from ./views.ts */
export type AOView = ViewName;

// scripts/_render_ao_blender.py is a sibling asset of this TS module's dir.
// dirname(import.meta.url) → .../Procedura/src/render
const SRC_RENDER_DIR = resolve(dirname(fileURLToPath(new URL(import.meta.url))));
const PROCEDURA_ROOT = resolve(SRC_RENDER_DIR, "..", "..");
export const AO_RENDER_SCRIPT = join(PROCEDURA_ROOT, "scripts", "_render_ao_blender.py");

/** First existing wins; a bare `blender` on $PATH is the last resort. Set
 *  PROCEDURA_BLENDER_PATH to skip the search entirely. */
const BLENDER_CANDIDATES = [
  join(process.env["HOME"] ?? "", "opt", "blender", "blender"),
  "/usr/local/bin/blender",
  "/opt/blender/blender",
  "blender",
];

function resolveBlenderBin(): string {
  const fromEnv = process.env["PROCEDURA_BLENDER_PATH"];
  if (fromEnv) return fromEnv;
  for (const c of BLENDER_CANDIDATES) {
    try { if (existsSync(c)) return c; } catch { /* unreadable mount — keep looking */ }
  }
  return BLENDER_CANDIDATES[0]!; // report the canonical path in the "not found" error
}

export const BLENDER_BIN = resolveBlenderBin();

export interface RenderAOOpts {
  stlPath: string;
  outDir: string;
  size?: number;
  samples?: number;
  aoSamples?: number;
  bg?: number;
  edgeThickness?: number;
  views?: readonly ViewName[];
  timeoutMs?: number;
  gpu?: boolean;
  /** Import as Z-up (OpenSCAD/CAD convention). Default true. Pass false for
   *  glTF-derived native meshes (TRELLIS/Hunyuan/UltraShape) which are Y-up. */
  zUp?: boolean;
  /** If the mesh exceeds this vertex count, decimate at render-time only
   *  (never written to disk) to keep Freestyle from OOMing. 0/undefined = off. */
  decimateAbove?: number;
  log?: (line: string) => void;
}

export interface RenderedView {
  view: string;
  path: string;
  sizeKb: number;
}

export type RenderAOResult =
  | { ok: true; views: RenderedView[] }
  | { ok: false; error: string; logPath?: string };

export async function renderAOViews(opts: RenderAOOpts): Promise<RenderAOResult> {
  const stl = opts.stlPath;
  const outDir = opts.outDir;
  const log = opts.log ?? (() => undefined);
  const size = opts.size ?? 768;
  const samples = opts.samples ?? 64;
  const aoSamples = opts.aoSamples ?? 16;
  const bg = opts.bg ?? 0.0;
  const edgeThickness = opts.edgeThickness ?? 1.2;
  const views = opts.views ?? DEFAULT_VIEWS;
  const timeoutMs = opts.timeoutMs ?? (Number(process.env.PROCEDURA_RENDER_TIMEOUT_MS) || 600_000);
  const gpu = opts.gpu ?? true;
  const zUp = opts.zUp ?? true;
  const decimateAbove = opts.decimateAbove ?? 0;

  mkdirSync(outDir, { recursive: true });

  if (!existsSync(BLENDER_BIN)) {
    const msg = `Blender not found at ${BLENDER_BIN}`;
    log(`      ${msg}`);
    return { ok: false, error: msg };
  }
  if (!existsSync(AO_RENDER_SCRIPT)) {
    const msg = `AO render script missing at ${AO_RENDER_SCRIPT}`;
    log(`      ${msg}`);
    return { ok: false, error: msg };
  }

  const args = [
    "--background",
    "--python", AO_RENDER_SCRIPT, "--",
    "--mesh", stl,
    "--out", outDir,
    "--views", views.join(","),
    "--samples", String(samples),
    "--size", String(size),
    "--ao-samples", String(aoSamples),
    "--bg", String(bg),
    "--edges",
    "--edge-thickness", String(edgeThickness),
  ];
  if (zUp) args.push("--z-up");
  if (decimateAbove > 0) args.push("--decimate-above", String(decimateAbove));
  if (gpu) args.push("--gpu");

  const tBlender = Date.now();
  const proc = Bun.spawn([BLENDER_BIN, ...args], {
    stdout: "pipe", stderr: "pipe",
  });
  const killer = setTimeout(() => proc.kill(), timeoutMs);
  const [stdout, stderr, exitCode] = await Promise.all([
    new Response(proc.stdout).text(),
    new Response(proc.stderr).text(),
    proc.exited,
  ]);
  clearTimeout(killer);
  addStage("blender.ao", Date.now() - tBlender);

  if (exitCode !== 0) {
    const logPath = join(outDir, "blender_stderr.log");
    writeFileSync(logPath, `${stderr}\n---STDOUT---\n${stdout}\n`, "utf8");
    const msg = `Blender AO render failed (exit ${exitCode})`;
    log(`      ${msg} — see ${logPath}`);
    return { ok: false, error: msg, logPath };
  }

  const dev = deviceLine(stdout);
  if (dev) log(`  ${dev}`);

  const rendered: RenderedView[] = [];
  const missing: string[] = [];
  for (const v of views) {
    const p = join(outDir, `ao-${v}.png`);
    if (existsSync(p) && statSync(p).size > 1024) {
      rendered.push({ view: v, path: p, sizeKb: Math.floor(statSync(p).size / 1024) });
    } else {
      missing.push(`ao-${v}.png`);
    }
  }
  if (missing.length > 0) {
    const msg = `AO render incomplete — missing: ${missing.join(", ")}`;
    log(`      ${msg}`);
    return { ok: false, error: msg };
  }
  return { ok: true, views: rendered };
}
