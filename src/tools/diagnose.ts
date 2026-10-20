/**
 * diagnose tool — the CRITIC step of the context → critic → fix cycle.
 *
 * This is a SEPARATE vision LLM call (not the agent's own turn): it loads the
 * dedicated reviewer prompt (src/pipeline/diagnose-prompt.md), shows the
 * reference image + the current-build views cached by the last render_views
 * call + the current SCAD + prior-iteration history, and returns a structured
 * `SUMMARY: … / ISSUES: [SEV] [modules: …] → FIX: …` diagnosis. It writes no
 * code — the agent (the fixer) applies the diagnosis with edit_module/edit_full.
 *
 * A separate vision call, not a tool the agent answers itself. Per cycle it saves, into
 * `_refine_steps/step_NNN/`:
 *   diagnosis.txt           — the structured diagnosis (what the agent acts on)
 *   diagnose_response.txt    — the critic's full raw response
 *   diagnose_thinking.txt    — the critic's reasoning (when the model emits any)
 *
 * The diagnosis is also appended to state.diagnosisHistory so the next cycle's
 * critic sees what was previously flagged and focuses on what's still wrong.
 */

import { fileURLToPath } from "node:url";
import { join } from "node:path";
import { readFileSync, writeFileSync, mkdirSync } from "node:fs";
import type { JsonObject } from "@harness/template/types";
import type { ToolExecutor, ToolDescriptor } from "@harness/template/tool";
import type { CanonicalPart } from "@harness/template/llm/protocol";
import type { SessionProceduraState } from "./state.ts";
import { pad } from "./state.ts";
import { splitThinkTags } from "../llm/think-tags.ts";
import { ensureConnectivity } from "./connectivity-cache.ts";

const DIAGNOSE_PROMPT_PATH = fileURLToPath(new URL("../pipeline/diagnose-prompt.md", import.meta.url));

/** How many times to re-issue the reviewer call when it returns an empty
 *  (reasoning-only) diagnosis or errors, before punting back to the agent. */
const MAX_DIAGNOSE_ATTEMPTS = 3;

/** One vision LLM call: takes a system prompt + user parts, returns the
 *  assistant text and its reasoning. Bound in refine.ts over the harness's
 *  llm.generate + the active route/model. */
export type DiagnoseLLMCall = (
  args: { system: string; userParts: CanonicalPart[] },
) => Promise<{ text: string; thinking: string }>;

const DESCRIPTOR: ToolDescriptor = {
  name: "diagnose",
  description:
    "Run the parametric-3D-model REVIEWER over the current build. This is the " +
    "critic step: it independently compares the latest rendered views against the " +
    "reference image + text spec and returns a structured diagnosis — a SUMMARY " +
    "line plus a prioritised ISSUES list, each tagged [HIGH|MED|LOW], naming the " +
    "module(s) at fault with a suggested FIX direction. It writes no code. " +
    "Call render_views FIRST (the reviewer reviews those exact views), then call " +
    "diagnose, then apply the top issue with the right edit tool. When diagnose " +
    "reports no HIGH issues, the model is good enough — call finish.",
  owner: { kind: "core" },
  inputSchema: { type: "object", properties: {} } satisfies JsonObject,
};

/**
 * Build the critic's lead text: prior history, MEASURED CONNECTIVITY, edit
 * budget, spec, and view-reading guidance. Exported so tests can assert the
 * wiring end-to-end — the v2 connectivity block was syntactically present but
 * provably never reached an LLM, a class of bug only an assertion on the BUILT
 * prompt catches.
 */
export function buildDiagnoseLeadText(
  state: SessionProceduraState,
  opts: { fixerHasTools?: boolean } = {},
): string {
  // The floater guidance below names snap_floaters, which only exists in the
  // AGENT refine. The direct refine's fixer is a patch call with no tools, so
  // telling it to "run snap_floaters first" prescribes an action it cannot
  // take. Default true so agent-mode text is byte-identical — this is the
  // control arm of an A/B and must not drift.
  const fixerHasTools = opts.fixerHasTools ?? true;
  // Prior-iteration history so the reviewer focuses on what's still wrong.
  let priorBlock = "";
  const history = state.diagnosisHistory;
  if (history.length > 0) {
    const trail = history
      .map((h) => `  - Cycle ${h.cycle}: ${h.summary}`)
      .join("\n");
    const last = history[history.length - 1]!;
    priorBlock =
      "=== PRIOR-ITERATION HISTORY ===\n" +
      `${state.completedEdits} edit(s) have been applied since the first review. ` +
      "Verify the earlier issues are actually resolved in the current views; " +
      "do not re-list fixed items, and flag any regression a fix introduced.\n\n" +
      `What each prior cycle flagged:\n${trail}\n\n` +
      `Most recent full diagnosis (cycle ${last.cycle}):\n${last.raw}\n\n`;
  }

  const legendBlock = state.partsColorLegend
    ? "=== PARTS-COLOUR LEGEND (module → RGB) ===\n" +
      "Use this to cite module names from the colours you see.\n" +
      `${state.partsColorLegend.trim()}\n\n`
    : "";

  // Measured TRUE connectivity (union-wrapped compile — see
  // connectivity-cache.ts). Vision cannot see a 0.3mm gap, so without this the
  // reviewer never raises floaters. Sync read of the mirror render_views /
  // diagnose.execute populated; stale entries (other buffer) are ignored.
  const conn = state.latestConnectivity && state.latestConnectivity.scad === state.scad
    ? state.latestConnectivity
    : null;
  const connBlock = conn
    ? "=== MEASURED CONNECTIVITY (from the compiled mesh — trust this over the views) ===\n" +
      `${conn.summary}\n` +
      (conn.realFloaters > 0
        ? `Treat the ${conn.realFloaters} floater(s) with REAL air gaps as [HIGH] issues: ` +
          `a visibly detached part outranks every proportion or detail complaint. ` +
          (fixerHasTools
            ? `Tell the fixer to run snap_floaters first (deterministic re-seat), and only ` +
              `prescribe manual module edits for what it reports unfixable.\n`
            : `Name the detached module and the neighbour it should attach to, and state ` +
              `the fix as a placement change that makes their volumes OVERLAP — abutting ` +
              `faces are not enough, an OpenSCAD union of two touching solids is still two ` +
              `solids.\n`)
        : (conn.microFloaters > 0
          ? `Only micro-gap shells remain (parts within a hair of their neighbour) — ` +
            `these are tolerated and NOT issues; do not list them. Focus on form.\n`
          : "No visible floaters — do not invent disconnection issues.\n")) +
      "\n"
    : "";

  // How many edits the fixer can still make. A 15-issue list against a 2-edit
  // budget guarantees 13 issues are theatre; ask for triage instead.
  const remaining = Number.isFinite(state.maxRefineSteps)
    ? Math.max(0, state.maxRefineSteps - state.completedEdits)
    : null;
  const budgetBlock = remaining !== null
    ? `=== REMAINING EDIT BUDGET: ${remaining} ===\n` +
      `The fixer can make ${remaining} more ACCEPTED edit(s) before this run ends — ` +
      `one per cycle, and each edit may touch a group of modules together. List AT ` +
      `MOST ${Math.max(1, remaining) + 2} issues, hardest-hitting first, and put the ` +
      `single change that would most improve faithfulness at #1. Prefer one issue ` +
      `that names a whole group of modules (a shared datum, a mirrored pair) over ` +
      `several separate single-module issues — the fixer can apply that in one ` +
      `edit. State each proportion issue as a MEASURABLE target (e.g. "nest the ` +
      `lower 40% of the shell into the chest", "reduce outer track to 1.8× cage ` +
      `width") so the fixer can hit it with measured arithmetic instead of a ` +
      `guess. Do not pad the list with cosmetic items.\n\n`
    : "";

  return (
    "You are reviewing the structural faithfulness of a compiled OpenSCAD " +
    "model against a reference image + text spec.\n\n" +
    priorBlock +
    connBlock +
    budgetBlock +
    `=== TEXT SPEC ===\n${state.workspace.text}\n\n` +
    "=== REFERENCE IMAGE ===\nShown first below, then the current-build views.\n\n" +
    legendBlock +
    "Any orthographic views are tight bounding-box fits — length ratios " +
    "WITHIN one view are exact, so measure proportions as ratios inside a " +
    "single view; each view is independently scaled, so never compare raw " +
    "pixel sizes across two views."
  );
}

export function makeDiagnoseTool(
  state: SessionProceduraState,
  callLLM: DiagnoseLLMCall,
): ToolExecutor {
  const systemPrompt = readFileSync(DIAGNOSE_PROMPT_PATH, "utf8");

  return {
    descriptor: DESCRIPTOR,
    async execute() {
      state.step += 1;

      // The critic reviews COMMITTED states. While an edit is pending the agent
      // must judge its own edit (compile + render, then accept_edit or
      // revert_edit) — spending the critic on an unaccepted edit is how a bad
      // guess used to cost two full cycles.
      if (state.pendingEdit) {
        return {
          ok: false,
          error:
            `You have a pending ${state.pendingEdit.tool} edit. Verify it yourself ` +
            `(compile, then render_views and compare), then call accept_edit if it ` +
            `landed as intended or revert_edit to roll it back and try a corrected ` +
            `version. diagnose reviews accepted states only.`,
        };
      }

      // Make sure the TRUE-connectivity entry exists for this buffer before
      // building the lead text (memoized — free when render already did it).
      await ensureConnectivity(state);

      // The reviewer must see the CURRENT build. Enforce the context→critic
      // ordering: render_views has to have run, and the views must reflect the
      // current SCAD (an edit since the last render invalidates them).
      if (state.latestViews.length === 0) {
        return {
          ok: false,
          error:
            "No current views to review. Call render_views first, then diagnose.",
        };
      }
      if (state.latestViewsScad !== state.scad) {
        return {
          ok: false,
          error:
            "The SCAD changed since the last render, so the views are stale. " +
            "Call render_views again so the reviewer sees the current geometry, " +
            "then call diagnose.",
        };
      }

      const leadText = buildDiagnoseLeadText(state);

      const userParts: CanonicalPart[] = [
        { kind: "text", text: leadText },
        // No image.png in a text-only run: the spec is the target.
        ...(state.workspace.hasImage
          ? [
              { kind: "text", text: "REFERENCE image (target, PBR render):" } as CanonicalPart,
              {
                kind: "image",
                data: readFileSync(state.workspace.imagePath).toString("base64"),
                mimeType: "image/png",
              } as CanonicalPart,
            ]
          : [{
              kind: "text",
              text: "TARGET — there is NO reference image for this object. The text " +
                    `specification is the complete and only target:\n\n${state.workspace.text}`,
            } as CanonicalPart]),
      ];
      for (const v of state.latestViews) {
        userParts.push({ kind: "text", text: `${v.label}:` });
        userParts.push({
          kind: "image",
          data: readFileSync(v.path).toString("base64"),
          mimeType: "image/png",
        });
      }
      userParts.push({
        kind: "text",
        text:
          "=== CURRENT SCAD CODE ===\n```openscad\n" +
          `${state.scad}\n` +
          "```\n\n" +
          "Return ONLY the diagnosis block in the format your system prompt " +
          "specifies. No code, no preamble.",
      });

      // Content-level retry. The transport layer (longTimeoutFetch + the harness)
      // retries network/timeout/5xx, but a reviewer call can return HTTP 200 with
      // an EMPTY diagnosis — the model spent the turn on <think> reasoning and
      // emitted no SUMMARY/ISSUES text (splitThinkTags strips the reasoning). That
      // is a transport SUCCESS, so it isn't retried below us; re-issue it here
      // (the call is stochastic — a fresh attempt usually yields the diagnosis).
      let text = "";
      let thinking = "";
      let lastFailure = "";
      for (let attempt = 1; attempt <= MAX_DIAGNOSE_ATTEMPTS; attempt++) {
        try {
          const r = await callLLM({ system: systemPrompt, userParts });
          // Strip inline <think>…</think> (GPT-5.5 et al.) so the diagnosis is
          // clean SUMMARY/ISSUES; fold that reasoning into the thinking channel.
          const split = splitThinkTags(r.text ?? "");
          text = split.text;
          thinking = (r.thinking ?? "") + (split.think ? ((r.thinking ?? "") ? "\n\n" : "") + split.think : "");
          if (text.trim()) break; // got a usable diagnosis
          lastFailure = "the reviewer returned an empty diagnosis (reasoning only, no SUMMARY/ISSUES)";
        } catch (e) {
          text = "";
          lastFailure = `the reviewer call failed (${(e as Error).message})`;
        }
        if (attempt < MAX_DIAGNOSE_ATTEMPTS) {
          console.error(`  [diagnose] attempt ${attempt}/${MAX_DIAGNOSE_ATTEMPTS}: ${lastFailure}; re-issuing`);
        }
      }

      const diagnosis = (text ?? "").trim();
      if (!diagnosis) {
        return {
          ok: false,
          error: `Reviewer produced no diagnosis after ${MAX_DIAGNOSE_ATTEMPTS} attempts — ${lastFailure}. ` +
            "Call diagnose again, or proceed with edit_module if you already know the fix.",
        };
      }

      // Persist the critic artifacts into this cycle's step dir.
      const stepDir = join(state.refineStepsDir, `step_${pad(state.refineStep)}`);
      try {
        mkdirSync(stepDir, { recursive: true });
        writeFileSync(join(stepDir, "diagnosis.txt"), diagnosis, "utf8");
        writeFileSync(join(stepDir, "diagnose_response.txt"), text, "utf8");
        if (thinking) writeFileSync(join(stepDir, "diagnose_thinking.txt"), thinking, "utf8");
      } catch {
        /* non-fatal: saving artifacts must not break the loop */
      }

      // Record in history for the next cycle's reviewer.
      const summaryLine =
        diagnosis.split("\n").find((l) => l.trim().toUpperCase().startsWith("SUMMARY:")) ??
        diagnosis.slice(0, 120);
      const summary = summaryLine.replace(/^\s*summary:\s*/i, "").trim();
      state.diagnosisHistory.push({ cycle: state.refineStep, summary, raw: diagnosis });
      state.hasFreshDiagnosis = true;   // unlocks one edit; consumed by edit_*

      return {
        ok: true,
        output: {
          text:
            diagnosis +
            "\n\nApply the highest-severity issue: move_parts if the parts are the " +
            "right shape but in the wrong place, edit_module for one module, " +
            "edit_modules when the SAME fix spans a group (one edit, up to 8 " +
            "modules). Call module_context first if the fix has to keep parts " +
            "aligned with their neighbours. Then compile and check_connectivity. " +
            "If there are no HIGH issues, call finish(verdict=\"ok\").",
          summary,
          cycle: state.refineStep,
        },
      };
    },
  };
}
