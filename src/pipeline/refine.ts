/**
 * Refine stage of the text → param3d pipeline.
 *
 * Composes the harness with the configured LLM route, all seven refine tools,
 * the ImageAwareEngine, and an in-memory session store. Runs the agent
 * loop on the given output_dir (which must already have draft.scad +
 * draft.stl from the draft stage) and writes the final outputs:
 *
 *   final.scad, final.stl, final.obj      — post-refine artifacts
 *   preview_final/                         — AO render of the final mesh
 *   final_summary.txt                      — finish-call verdict + summary
 *   _final_build/                          — compile intermediates
 */

import { fileURLToPath } from "node:url";
import { readFileSync, mkdirSync, writeFileSync, appendFileSync, existsSync, readdirSync, rmSync, renameSync, copyFileSync } from "node:fs";
import { join, basename } from "node:path";

import { createHarness, applyAutoCache, createLLMClient } from "@harness/template";
import type { ModelRef } from "@harness/template/types";
import type { CanonicalRequest, CanonicalPart } from "@harness/template/llm/protocol";

import { routeForModel } from "../llm/routes.ts";
import { longTimeoutFetch } from "../llm/long-timeout-fetch.ts";
import { resolveModel, DEFAULT_MODEL } from "../config/models.ts";
import { resolveWorkspace } from "../config/workspace.ts";
import { createSessionState } from "../tools/state.ts";
import { makeRenderViewsTool } from "../tools/render_views.ts";
import { makeCompileTool } from "../tools/compile.ts";
import { makeInspectModuleTool } from "../tools/inspect_module.ts";
import { makeReadScadTool } from "../tools/read_scad.ts";
import { makeEditModuleTool } from "../tools/edit_module.ts";
import { makeEditModulesTool } from "../tools/edit_modules.ts";
import { makeModuleContextTool } from "../tools/module_context.ts";
import { makeCheckConnectivityTool } from "../tools/check_connectivity.ts";
import { makeCheckCollisionsTool } from "../tools/check_collisions.ts";
import { makeMovePartsTool } from "../tools/move_parts.ts";
import { makeScalePartsTool } from "../tools/scale_parts.ts";
import { makeSnapFloatersTool } from "../tools/snap_floaters.ts";

/** Set PROCEDURA_NO_SNAP_FLOATERS=1 to drop the snap_floaters tool from the refine
 *  loop entirely. Default on — this only exists to measure and to bypass it. */
const SNAP_FLOATERS_ENABLED = process.env["PROCEDURA_NO_SNAP_FLOATERS"] !== "1";
import { makeAcceptEditTool, makeRevertEditTool } from "../tools/edit-transaction.ts";
import { makeEditFullTool } from "../tools/edit_full.ts";
import { makeFinishTool, type FinishSignal } from "../tools/finish.ts";
import { makeDiagnoseTool, type DiagnoseLLMCall } from "../tools/diagnose.ts";
import { createImageAwareEngine } from "../context/image-aware.ts";
import { createNoopSandbox } from "../sandbox/noop.ts";
import { createFileTrajectoryWriter } from "../trajectory/writer.ts";
import { REFINE_TOOLS } from "./refine-tools.ts";
import { subscribeRefineStepSaver } from "./refine-step-saver.ts";
import { compileScad } from "../scad/compile.ts";
import { snapFloaters } from "../scad/snap-floaters.ts";
import { publishMesh } from "../mesh/normalize.ts";
import { loadSTL } from "../mesh/stl.ts";
import { analyzeConnectivity, summarizeConnectivity } from "../mesh/connectivity.ts";
import { renderAOViews } from "../render/ao.ts";

const REFINE_PROMPT_PATH = fileURLToPath(new URL("./refine-prompt.md", import.meta.url));

export interface RefineOpts {
  outputDir: string;
  model?: string;
  maxSteps?: number;
  signal?: AbortSignal;
  /** Optional shared trajectory sink. When supplied, this run's events flow
   * into the caller's writer instead of a per-run JSONL. Useful when the
   * unified `runProcedura` pipeline wants one trajectory file across phases. */
  trajectorySink?: (event: import("@harness/template/trajectory").TrajectoryEvent) => void | Promise<void>;
  /** When set, used as the trajectory path reported in the result instead of
   * the locally created file (only meaningful when trajectorySink is set). */
  trajectoryPathOverride?: string;

  // ── Focused per-part refine (incremental draft stage) ─────────────────────
  /** When set, this refine pass is scoped to ONE top-level module (the part
   * just added by the incremental draft stage). The whole accumulated model is
   * still rendered so the agent sees the new part in situ, but edits are locked
   * to this module (edit_module rejects other names, edit_full is disabled). */
  focusModule?: string;
  /** When false, skip the heavy final-output write (final.scad + normalize +
   * AO preview). Instead persist the refined SCAD back to the workspace's
   * draft.scad and recompile draft.stl/.obj, so the incremental builder can
   * continue from the refined buffer. Default true. */
  writeFinal?: boolean;
  /** Persist the binary STL alongside the OBJ export. Default false — by
   * default the only mesh deliverable is the normalized OBJ; the STL stays in
   * the internal build dir (used for connectivity + AO preview). */
  exportStl?: boolean;
  /** Override the kickoff user message. The incremental stage supplies a
   * part-focused instruction here. */
  kickoffText?: string;
  /** Label shown in the console banner (e.g. "refine part 3/18: left_arm"). */
  bannerLabel?: string;
  /** Nest per-run debug artifacts (renders, compiles, refine-steps) under
   * `<outputDir>/<artifactsSubdir>/` instead of at the root. */
  artifactsSubdir?: string;
}

export interface RefineResult {
  ok: boolean;
  verdict: "ok" | "give_up" | "max-steps" | "error" | "aborted" | "skipped";
  summary: string;
  outputs: {
    scadPath: string;
    stlPath?: string;
    objPath?: string;
    previewDir?: string;
    trajectoryPath: string;
    diagnosisPath: string;
  };
  /** Refine cycles completed (= count of successful edits). */
  steps: number;
  /** Total tool calls executed during refine. */
  toolCalls: number;
}

/**
 * Default refine cycles.
 *
 * Was 3, chosen when a cycle cost the agent loop ~16 tool calls and a growing
 * transcript — six cycles there risked `context_too_large`, which had already
 * killed a run at cycle 2 of 3 and cost every edit banked before it.
 *
 * The direct cycle is 2 LLM calls and ~1.4 min, and each patch call is FRESH
 * rather than appended to a transcript, so depth costs linear time and no
 * context growth. Six is affordable now; that headroom is what the
 * simplification actually bought.
 *
 * Shared by both implementations so the A/B stays fair — an arm that silently
 * got twice the budget would not be a comparison.
 */
export const DEFAULT_REFINE_STEPS = Number(process.env["PROCEDURA_REFINE_STEPS"] ?? 6);

export async function runRefine(opts: RefineOpts): Promise<RefineResult> {
  const workspace = resolveWorkspace(opts.outputDir);
  const state = createSessionState(
    workspace,
    opts.artifactsSubdir ? { artifactsSubdir: opts.artifactsSubdir } : undefined,
  );
  const focusModule = opts.focusModule;
  const writeFinal = opts.writeFinal ?? true;
  const exportStl = opts.exportStl ?? false;
  const modelKey = opts.model ?? DEFAULT_MODEL;
  const modelRef: ModelRef = resolveModel(modelKey);
  const route = routeForModel(modelKey);
  const systemPrompt = readFileSync(REFINE_PROMPT_PATH, "utf8");

  const finishHolder: { signal: FinishSignal | null } = { signal: null };
  const onFinish = (s: FinishSignal) => { finishHolder.signal = s; };

  // The diagnose (critic) tool runs its own vision LLM call. The harness
  // doesn't exist yet (the tool is constructed below, inside createHarness), so
  // the tool calls through this indirection; we bind the real implementation
  // right after the harness is built.
  const diagnoseCallHolder: { fn: DiagnoseLLMCall | null } = { fn: null };
  const diagnoseCall: DiagnoseLLMCall = (args) => {
    if (!diagnoseCallHolder.fn) throw new Error("diagnose LLM not wired");
    return diagnoseCallHolder.fn(args);
  };

  // Trajectory sink — either externally supplied (unified pipeline mode) or
  // a per-run JSONL file created here.
  const runStamp = Date.now().toString(36);
  const localWriter = opts.trajectorySink
    ? null
    : createFileTrajectoryWriter(state.trajectoryDir, `ortho-${runStamp}`);
  const sink = opts.trajectorySink ?? localWriter!.sink;
  const trajectoryPath = opts.trajectoryPathOverride ?? localWriter!.path;

  // Harness composition. We supply a noop sandbox (our tools shell out via
  // Bun.spawn directly) and an allow-all ruleset (we trust the model + the
  // doom-loop guard inside runLoop; nothing asks the user mid-run).
  const harness = await createHarness({
    workspace: { rootDir: workspace.rootDir },
    llm: { route, client: createLLMClient({ fetch: longTimeoutFetch }) },
    sandbox: createNoopSandbox({ rootDir: workspace.rootDir }),
    includeBuiltins: false,
    customTools: [
      makeRenderViewsTool(state),
      makeCompileTool(state),
      makeInspectModuleTool(state),
      makeModuleContextTool(state),
      makeReadScadTool(state),
      makeEditModuleTool(state, focusModule ? { restrictTo: focusModule } : undefined),
      makeEditModulesTool(state),
      makeEditFullTool(state),
      makeMovePartsTool(state),
      makeScalePartsTool(state),
      // snap_floaters is OMITTED, not stubbed, when disabled: a tool the model
      // can see and call but that refuses is worse than one that was never
      // offered — it burns a turn and invites a retry. It measured 11.3 min of
      // a 29.1-min refine phase, and its job (nudging a detached module into
      // contact) overlaps with move_parts, so it is worth being able to skip.
      ...(SNAP_FLOATERS_ENABLED ? [makeSnapFloatersTool(state)] : []),
      makeAcceptEditTool(state),
      makeRevertEditTool(state),
      makeCheckConnectivityTool(state),
      makeCheckCollisionsTool(state),
      makeDiagnoseTool(state, diagnoseCall),
      makeFinishTool(state, onFinish),
    ],
    context: createImageAwareEngine({
      workspace, store: undefined as never, // wired below
      getPartsColorLegend: () => state.partsColorLegend,
    }) as never,
    trajectorySink: sink,
    defaultRuleset: [{ permission: "*", pattern: "*", action: "allow" }],
  });

  // Patch context engine's store binding now that harness is built.
  // (The factory above intentionally placed `undefined as never` because the
  // store comes from createHarness; we replace it now.)
  const realEngine = createImageAwareEngine({
    workspace, store: harness.store,
    getPartsColorLegend: () => state.partsColorLegend,
  });
  // The harness composes context at build time so we can't easily swap.
  // Simplest fix: wrap the engine's assemble to use the real store.
  // We do this by re-wrapping the harness's context engine in place:
  Object.assign(harness.context, realEngine);

  // Now that the harness exists, bind the diagnose tool's vision LLM call. It
  // runs a single-turn generate over the same route/model as the agent loop and
  // returns the reviewer's text + reasoning (reasoning needs the protocol's
  // reasoning_content branch — see llm-harness openai-chat.ts).
  diagnoseCallHolder.fn = async ({ system, userParts }) => {
    const req: CanonicalRequest = {
      model: modelRef,
      system: [{ text: system }],
      messages: [{ role: "user", content: userParts as CanonicalPart[] }],
    };
    applyAutoCache(req, { protocolId: route.protocol.id });
    const events = await harness.llm.generate(route, req);
    let text = "";
    let thinking = "";
    for (const ev of events) {
      if (ev.kind === "text-delta") text += ev.text;
      else if (ev.kind === "thinking-delta") thinking += ev.text;
      else if (ev.kind === "error") throw ev.error;
    }
    return { text, thinking };
  };

  // Live console mirror of key events.
  const sub = harness.bus.onAny((event, payload) => {
    if (event === "part.append") return;       // too noisy
    if (event === "tool.requested") {
      const p = payload as { toolName: string };
      console.log(`  [tool.requested] ${p.toolName}`);
      return;
    }
    if (event === "tool.executed") {
      const p = payload as { toolName: string };
      console.log(`  [tool.executed]  ${p.toolName}`);
      return;
    }
    console.log(`  [${event}]`);
  });

  const sessionId = await harness.sessions.create({
    title: `refine: ${basename(workspace.rootDir)}`,
    agentKind: "refine",
    model: modelRef,
  });

  // Trajectory attach for full event log.
  harness.trajectory.attachToBus(harness.bus, {
    sessionId,
    workspaceDir: workspace.rootDir,
  });

  // Refine-step budget. 1 step = 1 edit cycle bounded by edit_module / edit_full;
  // the verification compile after the Nth edit still runs because the abort
  // fires at the next iteration's signal check.
  const maxRefineSteps = opts.maxSteps ?? DEFAULT_REFINE_STEPS;
  state.maxRefineSteps = maxRefineSteps;

  // We bridge two abort sources: the caller's signal (if any) and the
  // edit-cap signal we own. Both feed into one controller passed to harness.run.
  const editCapController = new AbortController();
  // Named handler so we can detach it on completion. {once:true} only auto-
  // removes the listener when abort FIRES, not when the run ends — without an
  // explicit removeEventListener, a sequential caller (e.g. the incremental
  // draft's per-part refine loop) would pile up one stale listener per part on
  // the shared parent signal and trip Node's MaxListeners warning.
  const onParentAbort = (): void => { editCapController.abort(); };
  if (opts.signal) {
    if (opts.signal.aborted) editCapController.abort();
    else opts.signal.addEventListener("abort", onParentAbort, { once: true });
  }

  // Per-iteration snapshot writer — drops one dir per refine step into
  // <output_dir>/_refine_steps/. Also enforces the edit-cap: when
  // state.completedEdits reaches maxRefineSteps, fires editCapController.abort().
  const stepSaver = subscribeRefineStepSaver({
    bus: harness.bus, store: harness.store, state,
    workspaceDir: workspace.rootDir, sessionId,
    abortController: editCapController,
    stepsDirOverride: state.refineStepsDir,
    log: (s) => console.log(s),
  });

  console.log(`\n=== ${opts.bannerLabel ?? "Procedura refine"} ===`);
  console.log(`  workspace:     ${workspace.rootDir}`);
  console.log(`  model:         ${modelRef.modelId}`);
  console.log(`  max-steps:     ${maxRefineSteps} edit cycle(s)`);
  if (focusModule) console.log(`  focus module:  ${focusModule} (others frozen)`);
  console.log(`  trajectory:    ${trajectoryPath}`);
  console.log(`  refine-steps:  ${stepSaver.stepsDir}`);
  console.log(``);

  // Send the system prompt + initial user message as the kickoff turn.
  // We bypass harness.ask because we want to control runtimeToolAllowlist
  // and maxSteps, which ask() doesn't expose.
  await harness.store.appendMessage({
    id: `msg_sys_${Date.now().toString(36)}` as never,
    sessionId,
    role: "user",
    data: { text: systemPrompt },
  });
  await harness.store.appendMessage({
    id: `msg_kick_${Date.now().toString(36)}` as never,
    sessionId,
    role: "user",
    data: {
      text:
        opts.kickoffText ??
        "Begin the first cycle. Call render_views to see the current SCAD, then " +
        "diagnose for the reviewer's issue list. If it reports visible floaters, " +
        "run snap_floaters first (budget-free). Then: measure the flagged modules " +
        "(module_context with_measurements), compute the fix arithmetically, apply " +
        "ONE edit with the right tool (move_parts groups for repositioning, " +
        "scale_parts for group size, edit_module/edit_modules for geometry), " +
        "compile + render to judge it, and accept_edit or revert_edit. Repeat; " +
        "call finish when diagnose reports no HIGH issues.",
    },
  });
  harness.bus.emit("message.append", { sessionId, messageId: "kick", role: "user" });

  // Harness runLoop's maxSteps is now a generous CEILING on tool calls
  // (≈ 8 calls per edit cycle is plenty). The real cap on edit cycles is
  // enforced by the saver via editCapController above.
  // In focus mode, drop every edit that reaches beyond the focus part —
  // edit_full (rewrites the whole file), edit_modules (touches a group) and
  // move_parts (shifts assembly-level placements of frozen parts). The focused
  // per-part refine may only touch the one module via edit_module.
  // check_collisions is also dropped: it's a whole-model scan, irrelevant while
  // building a single part in place. module_context stays — it is read-only and
  // knowing the neighbours is exactly what a part-in-situ fix needs.
  const focusBlocked = new Set([
    "edit_full", "edit_modules", "move_parts", "scale_parts", "snap_floaters",
    "check_collisions",
  ]);
  const toolAllowlist = focusModule
    ? new Set([...REFINE_TOOLS].filter((t) => !focusBlocked.has(t)))
    : REFINE_TOOLS;

  let outcome = await harness.run({
    sessionId,
    model: modelRef,
    runtimeToolAllowlist: toolAllowlist,
    maxSteps: maxRefineSteps * 8,
    signal: editCapController.signal,
  });

  // NUDGE: the loop can end with the budget untouched.
  //
  // Observed on refine_v5/assault_buggy: render_views -> diagnose (which needed
  // two retries after the reviewer returned reasoning-only) -> module_context ->
  // a text-only assistant turn -> run.finished{reason:"stop"} with 0 of 3 edit
  // cycles spent, after 178 minutes. Whether the model gave up or the gateway
  // emitted a trailing stop alongside the tool call (the split-chunk pathology
  // seen on the Gemini routes), the outcome is the same: three hours for nothing.
  // So if the run stopped cleanly with edits still available and no finish()
  // verdict, tell it to carry on. Bounded, and only for a clean stop — never
  // after an abort (edit cap / caller signal) or a hard error.
  const MAX_NUDGES = Number(process.env["PROCEDURA_REFINE_MAX_NUDGES"] ?? 2);
  // Don't nudge a conversation that is already near the context window. Every
  // cycle appends full-size render images plus the SCAD source to ONE growing
  // transcript, so "keep going" is not free: motorcycle_rolling_chassis was
  // nudged onward to 58 tool calls and the run then died on
  // `400 context_too_large` at cycle 2 of 3, losing the remaining budget.
  // Survivors topped out around 49 calls, so cap the nudge below that and let
  // the run end cleanly with whatever it has banked instead.
  const NUDGE_MAX_TOOLCALLS = Number(process.env["PROCEDURA_REFINE_NUDGE_MAX_CALLS"] ?? 40);
  for (let nudge = 1; nudge <= MAX_NUDGES; nudge++) {
    const cleanStop = outcome.kind !== "error" && !editCapController.signal.aborted;
    const budgetLeft = state.completedEdits < maxRefineSteps;
    if (!cleanStop || !budgetLeft || finishHolder.signal !== null) break;
    if (state.step >= NUDGE_MAX_TOOLCALLS) {
      console.log(
        `  loop stopped with ${state.completedEdits}/${maxRefineSteps} edit cycle(s) used, but ` +
        `${state.step} tool calls have accumulated (cap ${NUDGE_MAX_TOOLCALLS}) — NOT nudging: ` +
        `another cycle of renders would risk a context-window failure that would ` +
        `discard the edits already accepted.`,
      );
      break;
    }

    console.log(
      `  loop stopped with ${state.completedEdits}/${maxRefineSteps} edit cycle(s) used ` +
      `and no finish verdict — nudging it to continue (${nudge}/${MAX_NUDGES})`,
    );
    await harness.store.appendMessage({
      id: `msg_nudge${nudge}_${Date.now().toString(36)}` as never,
      sessionId,
      role: "user",
      data: {
        text:
          `You stopped without completing the cycle: ${state.completedEdits} of ` +
          `${maxRefineSteps} edit cycles are used and you have not called finish. ` +
          (state.pendingEdit
            ? `A ${state.pendingEdit.tool} edit is still PENDING — compile it, render to ` +
              `check it, then accept_edit or revert_edit.`
            : `Continue now: render_views, diagnose, measure the flagged modules ` +
              `(module_context with_measurements), apply ONE measured edit, compile + ` +
              `render, then accept_edit.`) +
          ` If the model genuinely has no HIGH issues left, call finish(verdict="ok") ` +
          `with a summary; if it cannot be improved, call finish(verdict="give_up"). ` +
          `Do not reply with prose only — every turn must call a tool.`,
      },
    });
    harness.bus.emit("message.append", { sessionId, messageId: `nudge${nudge}`, role: "user" });
    outcome = await harness.run({
      sessionId,
      model: modelRef,
      runtimeToolAllowlist: toolAllowlist,
      maxSteps: maxRefineSteps * 8,
      signal: editCapController.signal,
    });
  }

  if (outcome.kind === "error") {
    console.log(`\n=== Run error ===\n${outcome.error.stack ?? outcome.error.message}\n`);
  }

  await sub.dispose();
  await stepSaver.dispose();
  if (localWriter) await localWriter.close();

  // Resolve a pending edit the loop never adjudicated (abort / max-steps hit
  // mid-transaction). A compiled pending edit is auto-accepted — the compile
  // proved it builds, and the regression gate below still protects the output.
  // An UNCOMPILED pending edit is rolled back: nothing ever verified it.
  if (state.pendingEdit) {
    if (state.stlIsStale) {
      console.log(`  pending ${state.pendingEdit.tool} edit was never compiled — rolling it back`);
      state.scad = state.pendingEdit.scadBefore;
    } else {
      console.log(`  pending ${state.pendingEdit.tool} edit auto-accepted at run end (compiled ok)`);
      state.completedEdits += 1;
    }
    state.pendingEdit = null;
  }

  // Post-process. In focused/incremental mode (writeFinal=false) we persist
  // the refined buffer back to draft.scad + recompile draft.stl/.obj so the
  // builder can continue, skipping the heavy normalize + AO-preview pass that
  // only the final whole-model refine needs.
  const writeResult = writeFinal
    ? await writeFinalOutputs(workspace, state, finishHolder.signal, exportStl)
    : await writePartRefineOutputs(workspace, state, exportStl);

  await harness.dispose();
  if (opts.signal) opts.signal.removeEventListener("abort", onParentAbort);

  // If our edit-cap fired the abort, treat it as "max-steps" rather than
  // "aborted" — semantically the run hit its budget, just measured in edits.
  const hitEditCap = state.completedEdits >= state.maxRefineSteps;
  const verdict =
    finishHolder.signal?.verdict ??
    (outcome.kind === "max-steps" ? "max-steps" :
      outcome.kind === "error" ? "error" :
        outcome.kind === "break" && outcome.reason === "aborted"
          ? (hitEditCap ? "max-steps" : "aborted") :
          "max-steps");
  const ok = verdict === "ok";

  console.log(`\n=== Done ===`);
  console.log(`  verdict:        ${verdict}`);
  console.log(`  refine cycles:  ${state.completedEdits} / ${state.maxRefineSteps}`);
  console.log(`  tool calls:     ${state.step}`);
  console.log(`  outputs:        ${writeResult.scadPath}`);
  if (writeResult.stlPath) console.log(`                  ${writeResult.stlPath}`);

  return {
    ok,
    verdict,
    summary: finishHolder.signal?.summary ?? "(no finish summary)",
    outputs: {
      scadPath: writeResult.scadPath,
      ...(writeResult.stlPath !== undefined ? { stlPath: writeResult.stlPath } : {}),
      ...(writeResult.objPath !== undefined ? { objPath: writeResult.objPath } : {}),
      ...(writeResult.previewDir !== undefined ? { previewDir: writeResult.previewDir } : {}),
      trajectoryPath,
      diagnosisPath: writeResult.diagnosisPath,
    },
    // `steps` now reports refine CYCLES (= completed edits) — matches the
    // user-facing semantic of --max-steps. Tool-call count is `toolCalls`.
    steps: state.completedEdits,
    toolCalls: state.step,
  };
}

// ──────────────────────────────────────────────────────────────────────────
// Post-processing: write the canonical final.{scad,stl,obj} +
// preview directory: final.{scad,stl,obj} + preview_final/ + final_summary.txt.
// ──────────────────────────────────────────────────────────────────────────

interface WriteResult {
  scadPath: string;
  stlPath?: string;
  objPath?: string;
  previewDir?: string;
  diagnosisPath: string;
}

export async function writeFinalOutputs(
  workspace: { rootDir: string; initialScadPath: string },
  state: { scad: string; lastGoodScad: string; latestStlPath: string | null },
  // Not FinishSignal: that type is the TOOL's vocabulary ("ok" | "give_up"),
  // and a run can end for reasons the tool has no word for — max-steps,
  // aborted, error. final_summary.txt is read by the results server and the
  // benchmark, and "give_up" (the model declared it could not improve) versus
  // "max-steps" (it ran out of budget) are very different claims about a run.
  finish: { verdict: string; summary: string } | null,
  exportStl: boolean,
): Promise<WriteResult> {
  const root = workspace.rootDir;
  const scadPath = join(root, "final.scad");
  const diagnosisPath = join(root, "final_summary.txt");

  // Compile a clean mesh into a build sub-dir. The STL stays internal (drives
  // normalize + connectivity + AO preview); the normalized OBJ is the default
  // deliverable. final.stl is written only when exportStl is set.
  const buildDir = join(root, "_final_build");
  mkdirSync(buildDir, { recursive: true });

  // Parse-gate: NEVER ship an unparseable final.scad. Compile the current
  // buffer; if the agent's last edit left it broken (e.g. a dropped brace or a
  // truncated statement), fall back to the last buffer that compiled
  // (lastGoodScad, seeded from the valid draft) — losing only the final broken
  // edit, not the whole model.
  let finalScad = state.scad;
  let buildStl: string | undefined, buildObj: string | undefined;
  let finalFacets: number | null = null;
  try {
    const r = await compileScad(state.scad, { outputDir: buildDir });
    buildStl = r.stlPath; buildObj = r.objPath; finalFacets = meshFacets(r.summary, r.stlPath);
  } catch (e) {
    console.log(`  WARN: final buffer does not compile (${(e as Error).message}) — reverting to last-good SCAD`);
    finalScad = state.lastGoodScad;
    if (finalScad !== state.scad) {
      try {
        const r = await compileScad(finalScad, { outputDir: buildDir });
        buildStl = r.stlPath; buildObj = r.objPath; finalFacets = meshFacets(r.summary, r.stlPath);
      } catch (e2) {
        console.log(`  WARN: last-good SCAD also failed (${(e2 as Error).message}) — keeping SCAD only`);
      }
    }
  }

  // Deterministic floater cleanup on the SHIPPED buffer. The agent may never
  // have called snap_floaters (or new floaters appeared after its last check);
  // this guarantees whatever refine ships got one mechanical re-seat attempt.
  // snapFloaters does its own TRUE-connectivity analysis internally (union-
  // wrapped compile — the lazy-union buildStl would call every overlapping part
  // a floater) and is a verified no-op when nothing actually floats.
  let finalFloaters: { real: number; micro: number } | null = null;
  if (buildStl !== undefined && process.env["PROCEDURA_FINAL_SNAP"] !== "0") {
    try {
      const snap = await snapFloaters({
        scad: finalScad,
        workDir: buildDir,
        log: (l) => console.log(l),
      });
      if (snap.ok) finalFloaters = { real: snap.realAfter, micro: snap.microAfter };
      if (snap.changed) {
        // Re-export the snapped buffer through the NORMAL compile path so the
        // shipped artifacts keep the standard (lazy-union) export semantics.
        const r = await compileScad(snap.scad, { outputDir: buildDir });
        finalScad = snap.scad;
        buildStl = r.stlPath;
        buildObj = r.objPath;
        finalFacets = meshFacets(r.summary, r.stlPath);
      }
    } catch (e) {
      console.log(`  WARN: final snap_floaters pass failed: ${(e as Error).message}`);
    }
  }

  // Geometry-regression gate. "It compiles" is not enough: a refine edit that
  // silently deletes most of the model compiles perfectly. Compare the shipped
  // mesh against the draft refine started from, and on a large loss fall back
  // to the newest refine snapshot that kept its geometry (so earlier good edits
  // survive), else to the draft itself.
  let regressionNote: string | null = null;
  if (buildStl !== undefined && finalFacets !== null) {
    const rescue = await rescueGeometryRegression({
      rootDir: root, buildDir, draftScadPath: workspace.initialScadPath,
      currentScad: finalScad, currentFacets: finalFacets,
    });
    if (rescue) {
      finalScad = rescue.scad;
      buildStl = rescue.stlPath;
      buildObj = rescue.objPath;
      finalFacets = rescue.facets;
      regressionNote = rescue.note;
    }
  }

  // Write the parse-verified buffer (never the broken one) + summary sidecar.
  writeFileSync(scadPath, finalScad, "utf8");
  writeFileSync(
    diagnosisPath,
    finish
      ? `verdict: ${finish.verdict}\n\nsummary:\n${finish.summary}\n`
      : "(no finish summary — loop terminated without finish call)\n",
    "utf8",
  );
  if (regressionNote) appendFileSync(diagnosisPath, `\n${regressionNote}\n`, "utf8");

  if (buildStl === undefined || buildObj === undefined) {
    return { scadPath, diagnosisPath };
  }

  // Publish the normalized final.obj (always) and final.stl (only on request).
  // publishMesh normalizes the build STL in place, so the connectivity + AO
  // passes below read the same unit-scaled geometry the OBJ ships.
  const objPath = join(root, "final.obj");
  const pub = publishMesh({
    buildStlPath: buildStl, buildObjPath: buildObj,
    objOut: objPath, stlOut: join(root, "final.stl"), exportStl,
    log: (m) => console.log(`  WARN: ${m}`),
  });
  const haveObj = pub.objPath !== undefined;
  const stlPath = pub.stlPath; // undefined unless exportStl

  // TRUE connectivity of the SHIPPED buffer (union analysis via the snap pass
  // above). The old line analyzed the lazy-union export and reported every
  // overlapping part as a floater — "77 visible floaters" on healthy models.
  if (finalFloaters !== null) {
    const tag = finalFloaters.real > 0 ? "  ⚠ FINAL" : "  final";
    const line = finalFloaters.real > 0
      ? `${finalFloaters.real} floater(s) with REAL air gaps remain` +
        `${finalFloaters.micro > 0 ? ` (+${finalFloaters.micro} micro-gap shell(s), tolerated)` : ""}` +
        ` (union analysis)`
      : finalFloaters.micro > 0
      ? `no real air gaps; ${finalFloaters.micro} micro-gap shell(s) sit flush against a ` +
        `neighbour and are tolerated (union analysis)`
      : `1 connected body (union analysis) — no real floaters`;
    console.log(`${tag} connectivity: ${line}`);
    appendFileSync(diagnosisPath, `\nfinal mesh connectivity: ${line}\n`, "utf8");
  }

  // AO preview from the normalized build STL.
  //
  // Rendered into a TEMP dir and swapped in only on success. renderAOViews
  // writes each view as it finishes, so a timeout partway through used to leave
  // the live dir half-updated: full_suspension_mountain_bike ended up with two
  // 640px views next to two stale 448px ones — an internally inconsistent set,
  // worse than either the old or the new one alone. Either all four views are
  // replaced or none are.
  const previewDir = join(root, "preview_final");
  const previewTmp = join(root, "preview_final.tmp");
  rmSync(previewTmp, { recursive: true, force: true });
  mkdirSync(previewTmp, { recursive: true });
  // decimateAbove: Blender's Freestyle+AO on a 9-14M-triangle mesh exceeds the
  // 600s render timeout (exit 143) every time — assault_buggy and 00001198
  // shipped with an empty preview_final/ in three consecutive runs. The render-
  // time Decimate modifier caps the cost; the mesh on disk is untouched and a
  // 640px preview cannot show the difference.
  // Size the DECIMATE and the timeout to the mesh, but never the resolution:
  // previews are compared side by side across cases, so 640px/32 samples has to
  // hold for every case. Getting this backwards produced exactly that mess —
  // 00001198 (13.7M facets) and full_suspension_mountain_bike (2.9M) blew a
  // one-size-fits-all 15-min budget on attempt 1 and fell back to 448px/8
  // samples, so 2 of 11 previews were visibly softer than the other 9. Freestyle
  // cost is driven by triangle count, not pixels: decimating harder is nearly
  // free visually at 640px, whereas dropping resolution is not.
  const previewFacets = finalFacets ?? 0;
  const heavyPreview = previewFacets > 6_000_000;
  let pr = await renderAOViews({
    stlPath: buildStl, outDir: previewTmp,
    size: 640, samples: 32, aoSamples: 8,
    decimateAbove: heavyPreview ? 800_000 : 2_000_000,
    // Generous even for mid-size meshes: this render happens once per case, and
    // 2.9M-facet full_suspension_mountain_bike overran a 15-min cap while three
    // refine cases shared the box. A long cap costs nothing when it succeeds.
    timeoutMs: heavyPreview ? 2_400_000 : 1_500_000,
  });
  if (!pr.ok) {
    // Last resort: same resolution, cheaper sampling and a harder decimate. A
    // slightly noisier 640px preview still composites with the others; a 448px
    // one does not.
    console.log(`  WARN: AO preview failed (${pr.error}) — retrying with a harder decimate`);
    pr = await renderAOViews({
      stlPath: buildStl, outDir: previewTmp,
      size: 640, samples: 8, aoSamples: 4,
      decimateAbove: 300_000,
      timeoutMs: 2_400_000,
    });
  }
  // Swap the complete set in, or discard it and keep whatever was there before.
  if (pr.ok) {
    try {
      rmSync(join(previewTmp, "blender_stderr.log"), { force: true });
      rmSync(previewDir, { recursive: true, force: true });
      renameSync(previewTmp, previewDir);
    } catch (e) {
      console.log(`  WARN: could not publish preview dir: ${(e as Error).message}`);
    }
  } else {
    // Keep the failed attempt's stderr for diagnosis, drop the partial renders.
    try {
      const log = join(previewTmp, "blender_stderr.log");
      if (existsSync(log)) { mkdirSync(previewDir, { recursive: true }); copyFileSync(log, join(previewDir, "blender_stderr.log")); }
      rmSync(previewTmp, { recursive: true, force: true });
    } catch { /* diagnosis only */ }
  }
  if (!pr.ok) {
    console.log(`  WARN: AO preview failed: ${pr.error}`);
    return {
      scadPath,
      ...(stlPath ? { stlPath } : {}),
      ...(haveObj ? { objPath } : {}),
      diagnosisPath,
    };
  }
  return {
    scadPath,
    ...(stlPath ? { stlPath } : {}),
    ...(haveObj ? { objPath } : {}),
    previewDir, diagnosisPath,
  };
}

// ──────────────────────────────────────────────────────────────────────────
// Geometry-regression gate
//
// Refine is supposed to fix the model, and mostly it does — but a whole-file
// rewrite can quietly re-author it at a fraction of the detail, and that
// rewrite compiles, so the parse-gate above waves it through. Measured on the
// hard_surface_v2 batch: 5 of 72 runs shipped under 80% of the draft's facets,
// the worst two at 6.2% and 1.9%. This compares the shipped mesh with the draft
// and, on a large loss, ships the newest refine snapshot that still has its
// geometry (keeping the earlier good edits) or failing that the draft itself.
//
// Set PROCEDURA_REFINE_MIN_FACET_RATIO=0 to disable.
// ──────────────────────────────────────────────────────────────────────────

const MIN_FACET_RATIO = Number(process.env["PROCEDURA_REFINE_MIN_FACET_RATIO"] ?? 0.8);
/** Refine snapshots to try before falling back to the draft (bounds compiles). */
const MAX_RESCUE_CANDIDATES = 2;

function facetsOf(summary: Record<string, unknown>): number | null {
  const geometry = summary["geometry"] as { facets?: unknown } | undefined;
  return typeof geometry?.facets === "number" ? geometry.facets : null;
}

/**
 * Facet count of a compile, preferring OpenSCAD's summary and falling back to
 * the STL. The summary omits `geometry` entirely when the top level isn't a
 * single object, so the STL is what makes this reliable.
 */
function meshFacets(summary: Record<string, unknown>, stlPath: string): number | null {
  const fromSummary = facetsOf(summary);
  if (fromSummary !== null) return fromSummary;
  if (!existsSync(stlPath)) return null;
  try { return loadSTL(stlPath).triCount; } catch { return null; }
}

/** The draft's facet count, from the artifacts the draft stage left behind. */
function draftFacets(rootDir: string): number | null {
  const buildDir = join(rootDir, "_draft_build");
  const summaryPath = join(buildDir, "output.summary.json");
  if (existsSync(summaryPath)) {
    try {
      const f = facetsOf(JSON.parse(readFileSync(summaryPath, "utf8")) as Record<string, unknown>);
      if (f !== null) return f;
    } catch { /* fall through to the STL */ }
  }
  const stl = join(buildDir, "output.stl");
  if (!existsSync(stl)) return null;
  try { return loadSTL(stl).triCount; } catch { return null; }
}

interface RescueResult {
  scad: string;
  stlPath: string;
  objPath: string;
  facets: number;
  note: string;
}

/** Exported for test/smoke-refine-guard.ts; not part of the stage's API. */
export async function rescueGeometryRegression(args: {
  rootDir: string;
  buildDir: string;
  draftScadPath: string;
  currentScad: string;
  currentFacets: number;
}): Promise<RescueResult | null> {
  if (!(MIN_FACET_RATIO > 0)) return null;                  // gate disabled
  const base = draftFacets(args.rootDir);
  if (base === null || base <= 0) return null;              // nothing to compare with
  const ratio = args.currentFacets / base;
  if (ratio >= MIN_FACET_RATIO) return null;                // healthy

  console.log(
    `  ⚠ REGRESSION: final mesh has ${args.currentFacets} facets vs draft ${base} ` +
    `(${(ratio * 100).toFixed(1)}%, floor ${(MIN_FACET_RATIO * 100).toFixed(0)}%) — ` +
    `refine lost geometry; looking for a healthy fallback`,
  );

  const draftScad = existsSync(args.draftScadPath)
    ? readFileSync(args.draftScadPath, "utf8")
    : null;
  // Cheap pre-filter: a snapshot far smaller than the draft is the damage, not
  // the recovery. Skip it without paying for a compile.
  const minChars = draftScad ? draftScad.length * 0.6 : 0;

  const candidates: { label: string; scad: string }[] = [];
  const stepsDir = join(args.rootDir, "_refine_steps");
  if (existsSync(stepsDir)) {
    const steps = readdirSync(stepsDir)
      .filter((d) => d.startsWith("step_"))
      .sort()
      .reverse();                                            // newest first
    for (const s of steps) {
      if (candidates.length >= MAX_RESCUE_CANDIDATES) break;
      const p = join(stepsDir, s, "scad.scad");
      if (!existsSync(p)) continue;
      let text: string;
      try { text = readFileSync(p, "utf8"); } catch { continue; }
      if (text === args.currentScad) continue;               // this is the damaged one
      if (text.length < minChars) continue;
      candidates.push({ label: `_refine_steps/${s}`, scad: text });
    }
  }
  if (draftScad) candidates.push({ label: "draft.scad", scad: draftScad });
  if (candidates.length === 0) {
    console.log(`  WARN: no fallback candidates — shipping the regressed model`);
    return null;
  }

  // Trial-compile into a scratch dir so a failed trial can't clobber the
  // _final_build outputs the caller is still holding paths to.
  const scratch = join(args.buildDir, "_rescue");
  mkdirSync(scratch, { recursive: true });
  for (const c of candidates) {
    let facets: number | null;
    try {
      const trial = await compileScad(c.scad, { outputDir: scratch });
      facets = meshFacets(trial.summary, trial.stlPath);
    } catch (e) {
      console.log(`    ${c.label}: does not compile (${(e as Error).message.slice(0, 100)})`);
      continue;
    }
    if (facets === null || facets / base < MIN_FACET_RATIO) {
      console.log(`    ${c.label}: ${facets ?? "?"} facets — still short, trying next`);
      continue;
    }
    // Re-compile the winner into the real build dir so _final_build matches
    // what actually ships.
    try {
      const final = await compileScad(c.scad, { outputDir: args.buildDir });
      console.log(
        `    recovered from ${c.label} — ${facets} facets ` +
        `(${((facets / base) * 100).toFixed(0)}% of draft)`,
      );
      return {
        scad: c.scad,
        stlPath: final.stlPath,
        objPath: final.objPath,
        facets,
        note:
          `geometry-regression gate: the refined buffer kept only ` +
          `${(ratio * 100).toFixed(1)}% of the draft's facets ` +
          `(${args.currentFacets} vs ${base}) — shipped ${c.label} instead ` +
          `(${facets} facets).`,
      };
    } catch (e) {
      // Compiled in scratch but not here: treat as a failed candidate and make
      // sure the build dir is restored below.
      console.log(`    ${c.label}: rebuild into _final_build failed (${(e as Error).message.slice(0, 100)})`);
    }
  }

  // Nothing healthy. Restore _final_build to the regressed-but-real outputs so
  // the caller's paths stay valid.
  console.log(`  WARN: no healthy fallback found — shipping the regressed model`);
  try { await compileScad(args.currentScad, { outputDir: args.buildDir }); }
  catch { /* the caller's paths were already written by the original compile */ }
  return null;
}

// ──────────────────────────────────────────────────────────────────────────
// Lightweight post-processing for a focused per-part refine: write the refined
// buffer back to draft.scad and recompile draft.stl/.obj. No normalize, no AO
// preview, no final.* — those belong to the final whole-model refine only.
// ──────────────────────────────────────────────────────────────────────────

async function writePartRefineOutputs(
  workspace: { rootDir: string; initialScadPath: string; initialStlPath: string },
  state: { scad: string },
  exportStl: boolean,
): Promise<WriteResult> {
  const scadPath = workspace.initialScadPath; // draft.scad
  const stlPath = workspace.initialStlPath;   // draft.stl (only when exportStl)
  const objPath = join(workspace.rootDir, "draft.obj");
  const diagnosisPath = join(workspace.rootDir, "final_summary.txt"); // path only; satisfies WriteResult

  // Compile the refined buffer FIRST, and persist it back to draft.scad/mesh
  // ONLY if it compiles. A focused refine can legitimately end on a NON-compiling
  // buffer (edit_module commits without compiling; finish(give_up)/max-steps/
  // aborted don't gate on compile). Writing that buffer to draft.scad would
  // corrupt the accumulated model the incremental builder reads back to splice
  // the next part — and nothing else wrote draft.scad during the loop, so simply
  // NOT writing leaves the valid pre-refine version on disk.
  const buildDir = join(workspace.rootDir, "_draft_build");
  mkdirSync(buildDir, { recursive: true });
  try {
    const r = await compileScad(state.scad, { outputDir: buildDir });
    writeFileSync(scadPath, state.scad, "utf8");
    // Same invariant as everywhere else: normalized draft.obj, STL opt-in only.
    const pub = publishMesh({
      buildStlPath: r.stlPath, buildObjPath: r.objPath,
      objOut: objPath, stlOut: stlPath, exportStl,
      log: (m) => console.log(`  per-part refine ${m}`),
    });
    return {
      scadPath,
      ...(pub.stlPath ? { stlPath: pub.stlPath } : {}),
      ...(pub.objPath ? { objPath: pub.objPath } : {}),
      diagnosisPath,
    };
  } catch (e) {
    console.log(
      `  WARN: per-part refine produced non-compiling SCAD ` +
      `(${(e as Error).message.slice(0, 140)}) — discarding it and keeping the ` +
      `pre-refine draft.scad so the next part can still build on a valid model.`,
    );
    return {
      scadPath,
      ...(existsSync(stlPath) ? { stlPath } : {}),
      ...(existsSync(objPath) ? { objPath } : {}),
      diagnosisPath,
    };
  }
}
