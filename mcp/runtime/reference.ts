import { readFileSync } from "node:fs";
import { join } from "node:path";
import { configureSession, generateSessionReference } from "./session.ts";

const dir = process.argv[2];
if (!dir) throw new Error("Invoke python3 mcp/workflow.py ref generate");
const spec = JSON.parse(readFileSync(join(dir, ".cx/inputs.json"), "utf8"));
configureSession({ run_dir: dir,
  attempt_id: JSON.parse(readFileSync(join(dir, ".cx/worker.json"), "utf8")).id,
  call_timeout_seconds: spec.profile.call_timeout_seconds });
await generateSessionReference({ prompt: readFileSync(join(dir, ".cx/reference_prompt.txt"), "utf8"), outputPath: join(dir, "image.png") });
