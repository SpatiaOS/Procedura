/** Run the existing Procedura pipeline; the coding-agent conversation supplies generation. */
import { readFileSync, writeFileSync, existsSync } from "node:fs";
import { join } from "node:path";
import { configureSession } from "./session.ts";
import { sealWorkerResult } from "./receipt.ts";

const runDir = process.argv[2];
if (!runDir) throw new Error("Invoke python3 mcp/workflow.py run");
const spec = JSON.parse(readFileSync(join(runDir, ".cx/inputs.json"), "utf8"));
configureSession({ run_dir: runDir,
  attempt_id: JSON.parse(readFileSync(join(runDir, ".cx/worker.json"), "utf8")).id,
  call_timeout_seconds: spec.profile.call_timeout_seconds });
// Python verifies the frozen inputs and executable hashes, clears workflow
// overrides and credentials, and starts Bun with --no-env-file. No API fallback.
globalThis.fetch = Object.assign(async () => {
  throw new Error("Model/API calls from the MCP pipeline are disabled; answer procedura.next_request in the current session");
}, { preconnect: () => { throw new Error("Network preconnect is disabled in MCP mode"); } }) as typeof fetch;

const { runProcedura } = await import("../../src/pipeline/procedura.ts");
const { beginRun, report, stageSnapshot } = await import("../../src/pipeline/stage-timer.ts");
beginRun();
const result = await runProcedura({
  ...spec.profile.options, ...spec.options,
  outputDir: runDir, text: readFileSync(join(runDir, "brief.md"), "utf8"),
  ...(spec.text_only ? { textOnly: true } : { inputImage: join(runDir, "image.png") }),
  scadModel: spec.model, agentModel: spec.model, paintModel: spec.model, motionModel: spec.model,
});
writeFileSync(join(runDir, ".cx/result.json"), JSON.stringify(result, null, 2) + "\n");
const timings = stageSnapshot();
writeFileSync(join(runDir, ".cx/timings.json"), JSON.stringify({ note: "Nested/concurrent timings overlap; do not sum them", stages: timings }, null, 2) + "\n");
console.log(report("Procedura MCP", Object.keys(timings).sort().map(key => [key, key, 0] as const)));
if (!existsSync(join(runDir, "final.scad")) || !existsSync(join(runDir, "final.obj")))
  throw new Error("Procedura did not publish final.scad and final.obj");
sealWorkerResult(runDir);
