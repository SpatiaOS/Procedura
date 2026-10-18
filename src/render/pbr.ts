/**
 * Hardcore PBR renderer — TypeScript orchestrator for `_render_pbr_blender.py`.
 *
 * Unlike the flat parts-colour renderer (soft 3-point rig, plastic Principled),
 * this drives a studio-HDRI-lit Cycles render with a full procedural material
 * per part: clearcoat, edge-wear (chipped paint → bare metal), AO grime, noise
 * roughness + micro-bump, on a reflective studio floor, denoised, AgX/Khronos
 * tone-mapped. Used by the paint stage for the `preview_painted/` deliverable.
 *
 * Materials are passed as a JSON spec file (richer than the parts-colour
 * `--meshes stl:r,g,b` string can carry), one entry per part STL.
 */

import { fileURLToPath } from "node:url";
import { existsSync, mkdirSync, writeFileSync, statSync, readdirSync } from "node:fs";
import { join, resolve, dirname } from "node:path";

import { BLENDER_BIN } from "./ao.ts";
import type { RenderedView } from "./ao.ts";
import { DEFAULT_VIEWS, type ViewName } from "./views.ts";
import { deviceLine } from "./device_line.ts";

const SRC_RENDER_DIR = resolve(dirname(fileURLToPath(new URL(import.meta.url))));
const PROCEDURA_ROOT = resolve(SRC_RENDER_DIR, "..", "..");
export const PBR_RENDER_SCRIPT = join(PROCEDURA_ROOT, "scripts", "_render_pbr_blender.py");

export interface PbrPartSpec {
  name: string;
  /** Per-part STL at its true assembled world position. */
  stl: string;
  color: [number, number, number];
  /** Material class from the paint stage ("glass", "metal", "painted", …).
   *  Drives the transmissive branch — without it every lens is an opaque slab. */
  material?: string;
  metalness: number;
  roughness: number;
  clearcoat?: number;
  /** Edge-wear / chipped-paint amount 0..1 (exposes bare metal at edges). */
  wear?: number;
  /** Grime / dust build-up in crevices 0..1. */
  dirt?: number;
  emission?: number;
}

export interface RenderPbrOpts {
  parts: PbrPartSpec[];
  outDir: string;
  size?: number;
  samples?: number;
  gpu?: boolean;
  views?: readonly ViewName[];
  /** Studio environment .exr. Auto-discovered from the Blender install if omitted. */
  hdri?: string;
  outPrefix?: string;
  /** Crease angle in degrees for CAD shading — normals are averaged only across
   *  edges softer than this, so panels stay flat and cylinders stay round.
   *  Script default 30. */
  smoothAngle?: number;
  /** View transform: "AgX" | "Standard" | "Khronos PBR Neutral" | "Filmic". */
  tone?: string;
  exposure?: number;
  /** Studio backdrop value 0..1. */
  bg?: number;
  /** Bloom/contrast/vignette compositor. Default true. */
  comp?: boolean;
  /** Shader-bevel radius in normalized units (model spans 2.0) — gives sharp CSG
   *  edges a machined fillet that catches a highlight. Script default 0.006;
   *  above ~0.012 the form starts to soften. 0 disables. */
  edgeBevel?: number;
  /** Soft-box size multiplier; <1 is harder, more directional light that
   *  separates planes and makes edge highlights read. Script default 0.4. */
  keyHard?: number;
  timeoutMs?: number;
  log?: (line: string) => void;
}

export type RenderPbrResult =
  | { ok: true; views: RenderedView[]; specPath: string }
  | { ok: false; error: string; logPath?: string };

/** Locate a bundled studio HDRI next to the Blender binary
 *  (<dir>/<ver>/datafiles/studiolights/world/studio.exr). */
export function findStudioHdri(): string | undefined {
  try {
    const dir = dirname(BLENDER_BIN);
    for (const ver of readdirSync(dir)) {
      const p = join(dir, ver, "datafiles", "studiolights", "world", "studio.exr");
      if (existsSync(p)) return p;
    }
  } catch { /* ignore */ }
  return undefined;
}

export async function renderPbrViews(opts: RenderPbrOpts): Promise<RenderPbrResult> {
  const outDir = opts.outDir;
  const log = opts.log ?? (() => undefined);
  const views = opts.views ?? DEFAULT_VIEWS;
  const prefix = opts.outPrefix ?? "pbr";
  // 20 min suits a render that owns the GPU. Several cases painting at once
  // share one, so a 4-view render can take well past that and get SIGTERMed
  // (exit 143) — which costs the run its refine pass. Batch drivers raise the
  // ceiling via PROCEDURA_PBR_TIMEOUT_MS rather than each call site guessing.
  const envTimeout = Number(process.env["PROCEDURA_PBR_TIMEOUT_MS"]);
  const timeoutMs = opts.timeoutMs
    ?? (Number.isFinite(envTimeout) && envTimeout > 0 ? envTimeout : 1_200_000);

  mkdirSync(outDir, { recursive: true });
  if (!existsSync(BLENDER_BIN)) return { ok: false, error: `Blender not found at ${BLENDER_BIN}` };
  if (!existsSync(PBR_RENDER_SCRIPT)) return { ok: false, error: `PBR script missing at ${PBR_RENDER_SCRIPT}` };
  if (opts.parts.length === 0) return { ok: false, error: "no parts to PBR-render" };

  const specPath = join(outDir, "pbr_render_spec.json");
  writeFileSync(specPath, JSON.stringify({
    parts: opts.parts.map((p) => ({
      stl: p.stl, name: p.name, color: p.color,
      ...(p.material ? { material: p.material } : {}),
      metalness: p.metalness, roughness: p.roughness,
      clearcoat: p.clearcoat ?? 0, wear: p.wear ?? 0, dirt: p.dirt ?? 0,
      emission: p.emission ?? 0,
    })),
  }, null, 1), "utf8");

  const hdri = opts.hdri ?? findStudioHdri();
  const args = [
    "--background", "--python", PBR_RENDER_SCRIPT, "--",
    "--spec", specPath, "--out", outDir, "--out-prefix", prefix,
    "--views", views.join(","),
    "--samples", String(opts.samples ?? 350),
    "--size", String(opts.size ?? 1280),
    "--z-up",
  ];
  if (opts.smoothAngle !== undefined) args.push("--smooth-angle", String(opts.smoothAngle));
  if (opts.tone) args.push("--tone", opts.tone);
  if (opts.exposure !== undefined) args.push("--exposure", String(opts.exposure));
  if (opts.bg !== undefined) args.push("--bg", String(opts.bg));
  if (opts.comp === false) args.push("--no-comp");
  if (opts.edgeBevel !== undefined) args.push("--edge-bevel", String(opts.edgeBevel));
  if (opts.keyHard !== undefined) args.push("--key-hard", String(opts.keyHard));
  if (hdri) args.push("--hdri", hdri);
  if (opts.gpu ?? true) args.push("--gpu");
  log(`[pbr] ${opts.parts.length} parts, ${views.length} views${hdri ? " (studio HDRI)" : " (procedural env)"}`);

  const proc = Bun.spawn([BLENDER_BIN, ...args], { stdout: "pipe", stderr: "pipe" });
  const killer = setTimeout(() => proc.kill(), timeoutMs);
  const [stdout, stderr, exitCode] = await Promise.all([
    new Response(proc.stdout).text(),
    new Response(proc.stderr).text(),
    proc.exited,
  ]);
  clearTimeout(killer);

  if (exitCode !== 0) {
    const logPath = join(outDir, "pbr_blender_stderr.log");
    writeFileSync(logPath, `${stderr}\n---STDOUT---\n${stdout}\n`, "utf8");
    return { ok: false, error: `PBR Blender render failed (exit ${exitCode})`, logPath };
  }

  const dev = deviceLine(stdout);
  if (dev) log(`  ${dev}`);

  const rendered: RenderedView[] = [];
  const missing: string[] = [];
  for (const v of views) {
    const p = join(outDir, `${prefix}-${v}.png`);
    if (existsSync(p) && statSync(p).size > 1024) {
      rendered.push({ view: v, path: p, sizeKb: Math.floor(statSync(p).size / 1024) });
    } else {
      missing.push(`${prefix}-${v}.png`);
    }
  }
  if (missing.length > 0) return { ok: false, error: `PBR render incomplete — missing: ${missing.join(", ")}` };
  return { ok: true, views: rendered, specPath };
}
