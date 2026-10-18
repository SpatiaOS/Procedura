/**
 * Draft stage of the text → param3d pipeline.
 *
 *   text → image-gen → SCAD-gen → compile
 *
 * One LLM call per stage (no agent loop). On compile failure, re-prompts the
 * model with the broken SCAD + compiler error up to ONE_SHOT_COMPILE_FIX_ATTEMPTS
 * times.
 *
 * Integrates with the harness on the same axes as the refine stage:
 *   - `createHarness` provides the LLMClient, bus, store, trajectory.
 *   - Each call creates a fresh session (agentKind = "draft") and attaches
 *     the trajectory recorder to the bus. The stage publishes standard events
 *     (`run.started`, `message.append`, `part.append`, `run.finished`) plus
 *     pipeline-specific events (`draft.image.*`, `draft.scad.*`,
 *     `draft.compile.*`).
 *   - The SCAD-gen LLM call uses `applyAutoCache` on its canonical request.
 *   - The user prompt + assistant SCAD response land in the session store as
 *     a single-turn conversation, so trajectory replay can reconstruct what
 *     the model saw and what it produced.
 *
 * Writes to the run's output dir:
 *   image.png             — the generated reference render
 *   draft.scad            — the LLM's SCAD source
 *   draft.stl, draft.obj  — the compiled mesh (copied out of _draft_build/)
 *   _draft_build/         — raw OpenSCAD intermediates (input.scad + summary)
 *   prompt.txt, effective_text.txt, image_prompt.txt
 *   response.txt          — full raw LLM response
 *   thinking.txt          — model reasoning trace (if any)
 */

import { fileURLToPath } from "node:url";
import { mkdirSync, writeFileSync, readFileSync, existsSync, statSync, copyFileSync } from "node:fs";
import { join, resolve, dirname, basename } from "node:path";

import { createHarness, applyAutoCache, createLLMClient } from "@harness/template";
import type { ModelRef } from "@harness/template/types";
import type {
  CanonicalRequest, CanonicalMessage, CanonicalPart,
} from "@harness/template/llm/protocol";

import { routeForModel } from "../llm/routes.ts";
import { longTimeoutFetch } from "../llm/long-timeout-fetch.ts";
import { splitThinkTags } from "../llm/think-tags.ts";
import { resolveModel, DEFAULT_MODEL } from "../config/models.ts";
import {
  generateImage, resolveImageModel, imageGenAvailable, imageGenDisabledReason,
} from "../imagegen/images.ts";
import { extractOpenscadCode } from "../scad/extract.ts";
import { compileScad } from "../scad/compile.ts";
import { loadSTL } from "../mesh/stl.ts";
import { publishMesh } from "../mesh/normalize.ts";
import {
  analyzeConnectivity, formatConnectivityDetail, summarizeConnectivity,
  evaluateConnectivityGate,
} from "../mesh/connectivity.ts";
import { createNoopSandbox } from "../sandbox/noop.ts";
import { createFileTrajectoryWriter } from "../trajectory/writer.ts";

const PROCEDURA_ROOT = resolve(dirname(fileURLToPath(new URL(import.meta.url))), "..", "..");
const IMAGE_PROMPT_PATH  = join(PROCEDURA_ROOT, "prompts", "image_prompt.md");
const SCAD_SYSTEM_PATH   = join(PROCEDURA_ROOT, "prompts", "scad_system.md");

// Both stages share one default so a run is comparable across them; override
// per stage with --scad-model / --agent-model, or globally with $PROCEDURA_MODEL.
export const DEFAULT_SCAD_MODEL = DEFAULT_MODEL;
/** Undefined when image generation is not configured — see imagegen/images.ts. */
export const DEFAULT_IMAGE_MODEL = resolveImageModel();
export const SCAD_MAX_ATTEMPTS = 3;
export const ONE_SHOT_COMPILE_FIX_ATTEMPTS = 3;

// Post-compile floater check: if the draft SCAD compiles but produces a mesh
// with any VISIBLE floater (a disconnected sub-body whose bbox spans ≥ 1% of
// the model — see VISIBLE_SPAN_FRACTION), re-prompt the LLM up to N times to fix
// the connectivity. This is the SAME span-based gate the refine ship-gate uses
// (evaluateConnectivityGate), not the old volume metric — thin/flat panels span
// a lot yet enclose ~0 volume, so a volume threshold let them through. Sub-1%-
// span specks (slivers / numerical noise) are still accepted.
export const ONE_SHOT_CONNECTIVITY_FIX_ATTEMPTS = 2;

export interface DraftOpts {
  text: string;
  outputDir: string;
  scadModel?: string;
  imageModel?: string;
  /** When set, skip image-gen and use this image file as the reference. The
   *  file is copied to <outputDir>/image.png. SCAD-gen still runs against it. */
  inputImage?: string;
  /** Persist the binary STL alongside the OBJ. Default false — the OBJ
   * deliverable is normalized; the STL stays in the internal build dir. */
  exportStl?: boolean;
  log?: (line: string) => void;
  /** Optional shared trajectory sink. When set, this call's events flow
   * into the caller's writer instead of a per-run JSONL. */
  trajectorySink?: (event: import("@harness/template/trajectory").TrajectoryEvent) => void | Promise<void>;
  /** Used as the trajectory path reported in the result when trajectorySink
   * is set. */
  trajectoryPathOverride?: string;
}

export interface DraftResult {
  ok: boolean;
  outputDir: string;
  imagePath: string;
  scadPath: string;
  stlPath?: string;
  objPath?: string;
  textPath: string;
  scadAttempts: number;
  compileAttempts: number;
  compileError?: string;
  /** Floater count of the final accepted STL (after connectivity-fix loop). */
  finalFloaterCount?: number;
  /** Floater volume fraction of the final accepted STL. */
  finalFloaterVolumeFraction?: number;
  /** Number of connectivity-fix re-prompts attempted. */
  connectivityFixAttempts?: number;
  durationMs: number;
  trajectoryPath: string;
  sessionId: string;
}

function buildImagePrompt(text: string): string {
  return readFileSync(IMAGE_PROMPT_PATH, "utf8").replace(/\{text\}/g, text);
}

function fileSize(p: string): number {
  return existsSync(p) ? statSync(p).size : 0;
}

function nextId(prefix: string): string {
  return `${prefix}_${Date.now().toString(36)}${Math.random().toString(36).slice(2, 8)}`;
}

export async function runDraft(opts: DraftOpts): Promise<DraftResult> {
  const text = opts.text.trim();
  if (!text) throw new Error("runDraft: `text` is required (and non-empty)");
  const log = opts.log ?? ((s) => console.log(s));
  const t0 = Date.now();

  const outDir = resolve(opts.outputDir);
  mkdirSync(outDir, { recursive: true });
  writeFileSync(join(outDir, "prompt.txt"), text, "utf8");
  writeFileSync(join(outDir, "effective_text.txt"), text, "utf8");
  const imagePath = join(outDir, "image.png");
  const scadPath  = join(outDir, "draft.scad");
  const stlPathOut = join(outDir, "draft.stl");
  const objPathOut = join(outDir, "draft.obj");
  const exportStl = opts.exportStl ?? false;

  // ── route + model selection ────────────────────────────────────────────
  const scadModelKey = opts.scadModel ?? DEFAULT_SCAD_MODEL;
  const scadModelRef: ModelRef = resolveModel(scadModelKey);
  const route = routeForModel(scadModelKey);

  // ── harness composition (matches the refine stage's pattern) ──────────
  const trajectoryDir = join(outDir, "_trajectory");
  mkdirSync(trajectoryDir, { recursive: true });
  const runStamp = nextId("d").slice(2);
  const localWriter = opts.trajectorySink
    ? null
    : createFileTrajectoryWriter(trajectoryDir, `draft-${runStamp}`);
  const sink = opts.trajectorySink ?? localWriter!.sink;
  const trajectoryPath = opts.trajectoryPathOverride ?? localWriter!.path;

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
    title: `draft: ${basename(outDir)}`,
    agentKind: "draft",
    model: scadModelRef,
  });
  harness.trajectory.attachToBus(harness.bus, { sessionId, workspaceDir: outDir });

  const runId = nextId("run");
  harness.bus.emit("run.started", { sessionId, runId });

  let scadAttempts = 0;
  let compileAttempts = 0;
  let lastCompileErr: string | null = null;
  let finalCompile: { stlPath: string; objPath: string } | null = null;
  let finalFloaterCount: number | undefined;
  let finalFloaterVolumeFraction: number | undefined;
  let connectivityFixAttempts = 0;
  // The current best SCAD that survives compile (used as starting point for
  // the connectivity-fix loop's re-prompts).
  let acceptedScad = "";

  try {
    // ── Stage A: image (provided input, or generated) ───────────────────
    if (opts.inputImage) {
      // Skip image-gen: use the supplied reference image as-is. SCAD-gen below
      // still runs against it. This is how the benchmark feeds frozen images.
      const src = resolve(opts.inputImage);
      if (!existsSync(src)) throw new Error(`inputImage not found: ${src}`);
      log(`[draft 1/3] image-gen SKIPPED — using provided image ${src}`);
      if (resolve(imagePath) !== src) copyFileSync(src, imagePath);
      harness.bus.emit("draft.image.skipped", {
        sessionId, source: src, bytes: fileSize(imagePath),
      } as never);
      log(`      image ok (provided, ${fileSize(imagePath)} bytes)`);
    } else {
      // Image generation is opt-in. Refuse here rather than reaching for an
      // image API the user never configured.
      if (!imageGenAvailable(opts.imageModel)) {
        throw new Error(imageGenDisabledReason(opts.imageModel));
      }
      const imageModel = resolveImageModel(opts.imageModel)!;
      log(`[draft 1/3] image-gen via ${imageModel}`);
      const imagePrompt = buildImagePrompt(text);
      writeFileSync(join(outDir, "image_prompt.txt"), imagePrompt, "utf8");
      harness.bus.emit("draft.image.started", {
        sessionId, model: imageModel, promptChars: imagePrompt.length,
      } as never);
      await generateImage({
        prompt: imagePrompt,
        outputPath: imagePath,
        ...(opts.imageModel !== undefined ? { model: opts.imageModel } : {}),
        log,
      });
      harness.bus.emit("draft.image.finished", {
        sessionId, path: imagePath, bytes: fileSize(imagePath),
      } as never);
      log(`      image ok (${fileSize(imagePath)} bytes)`);
    }

    // ── Stage B: SCAD-gen (compile-fix retry loop) ──────────────────────
    const systemPrompt = readFileSync(SCAD_SYSTEM_PATH, "utf8");
    const baseUserText =
      "You have TWO inputs — both are authoritative.\n\n" +
      `=== INPUT 1: TEXT DESCRIPTION ===\n${text}\n\n` +
      "=== INPUT 2: REFERENCE IMAGE ===\n" +
      "A single clean isometric render of the object described above, " +
      "attached below. Use the text for WHAT features must exist, and " +
      "the image for HOW they look (proportions, placement, surface " +
      "appearance).\n\n" +
      "TASK: Produce extremely-detailed parametric OpenSCAD code " +
      "following the L0..L3 detail mandate in your system prompt. " +
      "Every feature mentioned in the text OR visible in the image " +
      "must be modeled — not approximated away. Return ONLY raw " +
      "OpenSCAD code.";

    let lastBroken: string | null = null;

    for (let compileAttempt = 1; compileAttempt <= ONE_SHOT_COMPILE_FIX_ATTEMPTS; compileAttempt++) {
      compileAttempts = compileAttempt;
      const userText = compileAttempt === 1 ? baseUserText : baseUserText +
        "\n\n=== PRIOR ATTEMPT FAILED TO COMPILE — FIX THIS ===\n" +
        "OpenSCAD rejected the previous attempt with this error:\n\n" +
        (lastCompileErr ?? "").slice(0, 3000) +
        "\n\nThe prior (broken) SCAD source was:\n\n" +
        "```openscad\n" + (lastBroken ?? "") + "\n```\n\n" +
        "Identify the syntax / semantic problem and re-emit a CORRECTED, " +
        "COMPLETE SCAD file that compiles cleanly. Return ONLY raw OpenSCAD.";

      // Record the user message in the session store so trajectory replay
      // can reconstruct exactly what the model was asked.
      const imageB64 = readFileSync(imagePath).toString("base64");
      const userContent: CanonicalPart[] = [
        { kind: "text", text: userText },
        { kind: "text", text: "Isometric render of the object:" },
        { kind: "image", data: imageB64, mimeType: "image/png" },
      ];
      const userMsg = await harness.store.appendMessage({
        id: nextId("msg") as never,
        sessionId,
        role: "user",
        data: { text: userText, hasImage: true, compileAttempt },
      });
      harness.bus.emit("message.append", {
        sessionId, messageId: userMsg.id, role: "user",
      });

      let rawText = ""; let reasoning = "";
      let succeeded = false;
      let scadOut = "";
      for (let scadAttempt = 1; scadAttempt <= SCAD_MAX_ATTEMPTS; scadAttempt++) {
        scadAttempts += 1;
        const req: CanonicalRequest = {
          model: scadModelRef,
          system: [{ text: systemPrompt }],
          messages: [{ role: "user", content: userContent } satisfies CanonicalMessage],
        };
        applyAutoCache(req, { protocolId: route.protocol.id });

        log(`[draft 2/3] SCAD-gen attempt ${scadAttempt}/${SCAD_MAX_ATTEMPTS} (compile pass ${compileAttempt})`);
        harness.bus.emit("draft.scad.requested", {
          sessionId, model: scadModelRef.modelId, scadAttempt, compileAttempt,
        } as never);

        rawText = ""; reasoning = "";
        let errored = false;
        try {
          const events = await harness.llm.generate(route, req);
          for (const ev of events) {
            if (ev.kind === "text-delta") rawText += ev.text;
            else if (ev.kind === "thinking-delta") reasoning += ev.text;
            else if (ev.kind === "error") throw ev.error;
          }
        } catch (e) {
          errored = true;
          log(`      [scad-gen] attempt ${scadAttempt} threw: ${(e as Error).message}`);
        }
        if (errored) continue;

        // Some models (e.g. GPT-5.5) emit reasoning inline as <think>…</think>
        // in content. Strip it before extraction so it never lands in the SCAD
        // buffer; keep it as reasoning so thinking.txt still captures it.
        {
          const split = splitThinkTags(rawText);
          if (split.think) reasoning += (reasoning ? "\n\n" : "") + split.think;
          rawText = split.text;
        }

        const extracted = extractOpenscadCode(rawText);
        if (extracted && extracted.length >= 20) {
          scadOut = extracted;
          succeeded = true;
          break;
        }
        log(`      [scad-gen] empty/too-short extraction (${extracted.length} chars from ${rawText.length})`);
      }

      if (!succeeded) {
        harness.bus.emit("draft.scad.failed", { sessionId, compileAttempt } as never);
        const dur = Date.now() - t0;
        harness.bus.emit("run.finished", { sessionId, runId, reason: "error" });
        if (localWriter) await localWriter.close();
        await harness.dispose();
        return {
          ok: false, outputDir: outDir, imagePath, scadPath,
          textPath: join(outDir, "effective_text.txt"),
          scadAttempts, compileAttempts,
          compileError: `scad extraction exhausted after ${SCAD_MAX_ATTEMPTS} attempts`,
          durationMs: dur,
          trajectoryPath,
          sessionId,
        };
      }

      // Persist the assistant turn so it shows in trajectory + store.
      const assistantMsg = await harness.store.appendMessage({
        id: nextId("msg") as never,
        sessionId, role: "assistant",
        data: { modelId: scadModelRef.modelId, providerId: scadModelRef.providerId },
      });
      harness.bus.emit("message.append", {
        sessionId, messageId: assistantMsg.id, role: "assistant",
      });
      const textPart = await harness.store.appendPart({
        id: nextId("part") as never,
        messageId: assistantMsg.id,
        sessionId, kind: "text",
        data: { text: rawText },
      });
      harness.bus.emit("part.append", {
        sessionId, messageId: assistantMsg.id, partId: textPart.id, kind: "text",
      });
      if (reasoning) {
        const reasonPart = await harness.store.appendPart({
          id: nextId("part") as never,
          messageId: assistantMsg.id,
          sessionId, kind: "reasoning",
          data: { text: reasoning },
        });
        harness.bus.emit("part.append", {
          sessionId, messageId: assistantMsg.id, partId: reasonPart.id, kind: "reasoning",
        });
      }
      writeFileSync(join(outDir, "response.txt"), rawText, "utf8");
      if (reasoning) writeFileSync(join(outDir, "thinking.txt"), reasoning, "utf8");
      writeFileSync(scadPath, scadOut, "utf8");
      harness.bus.emit("draft.scad.extracted", {
        sessionId, chars: scadOut.length, compileAttempt,
      } as never);

      // ── Stage C: compile (into sub-dir, then promote .stl / .obj) ────
      log(`[draft 3/3] compile`);
      harness.bus.emit("draft.compile.started", {
        sessionId, compileAttempt,
      } as never);
      try {
        const buildDir = join(outDir, "_draft_build");
        mkdirSync(buildDir, { recursive: true });
        const r = await compileScad(scadOut, { outputDir: buildDir });
        // Keep the working mesh INTERNAL to the build dir; the top-level
        // draft.obj/draft.stl deliverables (normalized OBJ, opt-in STL) are
        // published once at finalize.
        finalCompile = { stlPath: r.stlPath, objPath: r.objPath };
        acceptedScad = scadOut;
        harness.bus.emit("draft.compile.finished", {
          sessionId, stlBytes: fileSize(r.stlPath), durationMs: r.durationMs,
        } as never);
        log(`      compile ok (${scadOut.split("\n").length} lines, ${fileSize(r.stlPath) >> 10} KB build STL)`);
        lastCompileErr = null;
        break;
      } catch (e) {
        lastBroken = scadOut;
        lastCompileErr = (e as Error).message;
        harness.bus.emit("draft.compile.failed", {
          sessionId, compileAttempt, error: lastCompileErr.slice(0, 200),
        } as never);
        if (compileAttempt < ONE_SHOT_COMPILE_FIX_ATTEMPTS) {
          log(`      compile FAILED (${lastCompileErr.slice(0, 160)}); re-prompting`);
        } else {
          log(`      compile FAILED after ${compileAttempt} attempts; giving up`);
        }
      }
    }

    // ── Stage D: connectivity-fix loop ─────────────────────────────────
    // If the compiled mesh has significant floaters, re-prompt the LLM to
    // fix them (typically: deepen overlaps or add struts). Up to N attempts;
    // a compile failure in a fix attempt reverts to the prior good SCAD.
    if (finalCompile && acceptedScad) {
      const imageB64 = readFileSync(imagePath).toString("base64");
      const systemPrompt = readFileSync(SCAD_SYSTEM_PATH, "utf8");

      for (let cAttempt = 0; cAttempt <= ONE_SHOT_CONNECTIVITY_FIX_ATTEMPTS; cAttempt++) {
        // Analyse the currently accepted STL.
        let conn: ReturnType<typeof analyzeConnectivity>;
        try {
          conn = analyzeConnectivity(loadSTL(finalCompile.stlPath));
        } catch (e) {
          log(`[draft connectivity] analyse failed: ${(e as Error).message}; skipping`);
          break;
        }
        finalFloaterCount = conn.floaterCount;
        finalFloaterVolumeFraction = conn.floaterVolumeFraction;
        const gate = evaluateConnectivityGate(conn);
        harness.bus.emit("draft.connectivity.checked", {
          sessionId, floaterCount: conn.floaterCount,
          visibleFloaterCount: conn.visibleFloaterCount,
          maxFloaterSpanFraction: conn.maxFloaterSpanFraction,
          floaterVolumeFraction: conn.floaterVolumeFraction,
          totalVolume: conn.totalVolume, attempt: cAttempt,
        } as never);

        if (gate.ok) {
          log(`[draft connectivity] ${summarizeConnectivity(conn)} — accepting`);
          break;
        }
        if (cAttempt === ONE_SHOT_CONNECTIVITY_FIX_ATTEMPTS) {
          log(`[draft connectivity] ${summarizeConnectivity(conn)} — giving up after ${cAttempt} fix attempts (refine will enforce)`);
          break;
        }

        const nextAttempt = cAttempt + 1;
        connectivityFixAttempts = nextAttempt;
        log(`[draft connectivity] ${summarizeConnectivity(conn)} — attempting fix ${nextAttempt}/${ONE_SHOT_CONNECTIVITY_FIX_ATTEMPTS}`);
        harness.bus.emit("draft.connectivity.fix_requested", {
          sessionId, attempt: nextAttempt,
          floaterCount: conn.floaterCount,
          visibleFloaterCount: conn.visibleFloaterCount,
          floaterVolumeFraction: conn.floaterVolumeFraction,
        } as never);

        const connDetail = formatConnectivityDetail(conn).slice(0, 2500);
        const connUserText =
          "You have TWO inputs — both are authoritative.\n\n" +
          `=== INPUT 1: TEXT DESCRIPTION ===\n${text}\n\n` +
          "=== INPUT 2: REFERENCE IMAGE ===\n" +
          "A single clean isometric render of the object, attached below.\n\n" +
          "=== PRIOR ATTEMPT COMPILED BUT HAS DISCONNECTED FLOATERS ===\n" +
          `The SCAD compiles cleanly, but the resulting STL has ${conn.visibleFloaterCount} ` +
          `VISIBLE disconnected sub-bodies (each spans ≥ 1% of the model; worst spans ` +
          `${(conn.maxFloaterSpanFraction * 100).toFixed(0)}%), ${conn.floaterCount} total. ` +
          `These violate the connectivity mandate: every translated part must overlap its ` +
          `host by ≥ 0.5 mm, or be connected via a visible strut. Focus on the parts flagged ` +
          `◀ VISIBLE below — judge by span%, not volume.\n\n` +
          `Connectivity breakdown (largest first; bboxes are world-space):\n` +
          connDetail + "\n\n" +
          "FIX TASK: Compare each floater's bbox to the `translate([x,y,z])` calls in " +
          "the prior SCAD below — the floater whose bbox center is at (cx,cy,cz) is " +
          "produced by the translate whose offset matches. Re-emit a CORRECTED, " +
          "COMPLETE SCAD that resolves the floaters. Common fixes:\n" +
          "  - increase overlap depth in the offending translate (move the part 1-3 mm " +
          "    into its host body)\n" +
          "  - add an explicit cylinder strut between detached parts (a thin rod that " +
          "    spans both bboxes)\n" +
          "  - extend a flange / boss on the host body to meet the orphan part\n\n" +
          "Keep all other modules intact. Return ONLY the complete corrected SCAD.\n\n" +
          "Prior (floater-bearing) SCAD:\n```openscad\n" + acceptedScad + "\n```";

        // Persist user message in the session store (parallels stage B).
        const userMsg = await harness.store.appendMessage({
          id: nextId("msg") as never,
          sessionId,
          role: "user",
          data: { text: "(connectivity-fix prompt)", hasImage: true, connectivityAttempt: nextAttempt },
        });
        harness.bus.emit("message.append", {
          sessionId, messageId: userMsg.id, role: "user",
        });

        // Inner SCAD-gen retry (same shape as stage B's inner loop).
        const userContent: CanonicalPart[] = [
          { kind: "text", text: connUserText },
          { kind: "text", text: "Isometric render of the object:" },
          { kind: "image", data: imageB64, mimeType: "image/png" },
        ];
        let rawText = ""; let reasoning = "";
        let newScad = "";
        let succeeded = false;
        for (let scadAttempt = 1; scadAttempt <= SCAD_MAX_ATTEMPTS; scadAttempt++) {
          scadAttempts += 1;
          const req: CanonicalRequest = {
            model: scadModelRef,
            system: [{ text: systemPrompt }],
            messages: [{ role: "user", content: userContent } satisfies CanonicalMessage],
          };
          applyAutoCache(req, { protocolId: route.protocol.id });
          log(`[draft connectivity-fix ${nextAttempt}] SCAD-gen attempt ${scadAttempt}/${SCAD_MAX_ATTEMPTS}`);
          harness.bus.emit("draft.scad.requested", {
            sessionId, model: scadModelRef.modelId,
            scadAttempt, connectivityAttempt: nextAttempt,
          } as never);

          rawText = ""; reasoning = "";
          try {
            const events = await harness.llm.generate(route, req);
            for (const ev of events) {
              if (ev.kind === "text-delta") rawText += ev.text;
              else if (ev.kind === "thinking-delta") reasoning += ev.text;
              else if (ev.kind === "error") throw ev.error;
            }
          } catch (e) {
            log(`      [scad-gen] threw: ${(e as Error).message}`);
            continue;
          }
          // Strip inline <think>…</think> (GPT-5.5 et al.) before extraction, as
          // in stage B — otherwise it poisons the connectivity-fixed SCAD buffer.
          {
            const split = splitThinkTags(rawText);
            if (split.think) reasoning += (reasoning ? "\n\n" : "") + split.think;
            rawText = split.text;
          }
          const extracted = extractOpenscadCode(rawText);
          if (extracted.length >= 20) { newScad = extracted; succeeded = true; break; }
          log(`      [scad-gen] empty extraction (${extracted.length} chars from ${rawText.length})`);
        }

        if (!succeeded) {
          log(`[draft connectivity-fix ${nextAttempt}] SCAD-gen exhausted; keeping previous SCAD`);
          harness.bus.emit("draft.connectivity.fix_failed", {
            sessionId, attempt: nextAttempt, reason: "scad_extraction_exhausted",
          } as never);
          break;
        }

        // Persist assistant turn.
        const assistantMsg = await harness.store.appendMessage({
          id: nextId("msg") as never,
          sessionId, role: "assistant",
          data: { modelId: scadModelRef.modelId, providerId: scadModelRef.providerId },
        });
        harness.bus.emit("message.append", {
          sessionId, messageId: assistantMsg.id, role: "assistant",
        });
        const textPart = await harness.store.appendPart({
          id: nextId("part") as never,
          messageId: assistantMsg.id,
          sessionId, kind: "text",
          data: { text: rawText },
        });
        harness.bus.emit("part.append", {
          sessionId, messageId: assistantMsg.id, partId: textPart.id, kind: "text",
        });
        if (reasoning) {
          const reasonPart = await harness.store.appendPart({
            id: nextId("part") as never,
            messageId: assistantMsg.id,
            sessionId, kind: "reasoning",
            data: { text: reasoning },
          });
          harness.bus.emit("part.append", {
            sessionId, messageId: assistantMsg.id, partId: reasonPart.id, kind: "reasoning",
          });
        }
        writeFileSync(join(outDir, `response_conn${nextAttempt}.txt`), rawText, "utf8");
        if (reasoning) writeFileSync(join(outDir, `thinking_conn${nextAttempt}.txt`), reasoning, "utf8");

        // Compile the connectivity-fix candidate. If it breaks compile, REVERT.
        log(`[draft connectivity-fix ${nextAttempt}] compile`);
        try {
          const buildDir = join(outDir, "_draft_build");
          const r = await compileScad(newScad, { outputDir: buildDir });
          acceptedScad = newScad;
          writeFileSync(scadPath, newScad, "utf8");
          finalCompile = { stlPath: r.stlPath, objPath: r.objPath };
          harness.bus.emit("draft.connectivity.fix_applied", {
            sessionId, attempt: nextAttempt,
            stlBytes: fileSize(r.stlPath), durationMs: r.durationMs,
          } as never);
          log(`      compile ok — re-checking connectivity`);
          // Loop continues — next iteration re-analyses the new STL.
        } catch (e) {
          const msg = (e as Error).message;
          log(`[draft connectivity-fix ${nextAttempt}] NEW SCAD failed to compile (${msg.slice(0, 120)}); reverting to prior good SCAD`);
          harness.bus.emit("draft.connectivity.fix_failed", {
            sessionId, attempt: nextAttempt, reason: "compile_failed",
            error: msg.slice(0, 200),
          } as never);
          // Re-write the accepted SCAD back so the on-disk file matches state.
          writeFileSync(scadPath, acceptedScad, "utf8");
          break;
        }
      }
    }

    // Finalize the draft mesh export: publish a NORMALIZED draft.obj (always)
    // and draft.stl only when exportStl is set (STL is not a default output).
    if (finalCompile && existsSync(finalCompile.stlPath)) {
      publishMesh({
        buildStlPath: finalCompile.stlPath, buildObjPath: finalCompile.objPath,
        objOut: objPathOut, stlOut: stlPathOut, exportStl,
        log: (m) => log(`  draft ${m}`),
      });
    }
  } finally {
    harness.bus.emit("run.finished", {
      sessionId, runId,
      reason: finalCompile ? "stop" : "error",
    });
    if (localWriter) await localWriter.close();
    await harness.dispose();
  }

  const dur = Date.now() - t0;
  return {
    ok: finalCompile !== null,
    outputDir: outDir,
    imagePath,
    scadPath,
    ...(exportStl && existsSync(stlPathOut) ? { stlPath: stlPathOut } : {}),
    ...(existsSync(objPathOut) ? { objPath: objPathOut } : {}),
    textPath: join(outDir, "effective_text.txt"),
    scadAttempts, compileAttempts,
    ...(lastCompileErr ? { compileError: lastCompileErr } : {}),
    ...(finalFloaterCount !== undefined ? { finalFloaterCount } : {}),
    ...(finalFloaterVolumeFraction !== undefined ? { finalFloaterVolumeFraction } : {}),
    ...(connectivityFixAttempts > 0 ? { connectivityFixAttempts } : {}),
    durationMs: dur,
    trajectoryPath,
    sessionId,
  };
}
