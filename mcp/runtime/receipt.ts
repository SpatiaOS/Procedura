/** Durable worker result for recovery if the outer Codex/Python operator exits. */
import { createHash } from "node:crypto";
import { readFileSync, readdirSync, renameSync, statSync, writeFileSync } from "node:fs";
import { join } from "node:path";

function sha(path: string): string {
  return createHash("sha256").update(readFileSync(path)).digest("hex");
}

export function sealWorkerResult(runDir: string): void {
  const worker = JSON.parse(readFileSync(join(runDir, ".cx/worker.json"), "utf8"));
  const artifacts: Record<string, string> = {};
  function walk(relative: string) {
    for (const entry of readdirSync(join(runDir, relative), { withFileTypes: true })) {
      if (entry.name === ".cx") continue;
      const name = relative ? `${relative}/${entry.name}` : entry.name;
      const path = join(runDir, name);
      if (entry.isDirectory()) walk(name);
      else if (statSync(path).isFile()) artifacts[name] = sha(path);
    }
  }
  walk("");
  const record = { worker_id: worker.id, pack_sha256: worker.pack_sha256,
    result_sha256: sha(join(runDir, ".cx/result.json")), artifacts };
  const dest = join(runDir, ".cx/worker-success.json");
  writeFileSync(dest + ".tmp", JSON.stringify(record, null, 2) + "\n");
  renameSync(dest + ".tmp", dest);
}
