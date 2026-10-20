import { afterAll, expect, test } from "bun:test";
import { mkdtempSync, mkdirSync, readFileSync, readdirSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { createHash } from "node:crypto";
import { configureSession, generateInSession, generateSessionReference } from "../runtime/session.ts";
const root = mkdtempSync(join(tmpdir(), "cx-session-transport-"));
afterAll(() => rmSync(root, { recursive: true, force: true }));
const sha = (data: string | Buffer) => createHash("sha256").update(data).digest("hex");
function setup(name: string, timeout = 3) {
  const dir = join(root, name); mkdirSync(dir);
  configureSession({ run_dir: dir, attempt_id: "test-attempt", call_timeout_seconds: timeout });
  return dir;
}
const request = () => ({ model: { providerId: "current-session", modelId: "current-session" },
  system: [{ text: "Original upstream instructions" }], messages: [{ role: "user" as const, content: [{ kind: "text" as const, text: "Original task" }] }] });
async function pending(dir: string, count = 1) {
  const base = join(dir, ".cx/calls");
  for (let i = 0; i < 100; i++) {
    const dirs = (() => { try { return readdirSync(base).sort(); } catch { return []; } })();
    if (dirs.length >= count) return join(base, dirs[count - 1]!);
    await new Promise(r => setTimeout(r, 10));
  }
  throw new Error("No session request was published");
}
function submit(dir: string, text: string, image = false) {
  const req = JSON.parse(readFileSync(join(dir, "request.json"), "utf8"));
  const bytes = image ? Buffer.from([137,80,78,71,13,10,26,10,1,2,3]) : text;
  if (image) writeFileSync(join(dir, "submitted.png"), bytes);
  writeFileSync(join(dir, "submission.json"), JSON.stringify({ request_id: req.id, attempt_id: req.attempt_id,
    request_sha256: sha(readFileSync(join(dir, "request.json"))), submitted_at: Date.now(),
    artifact_sha256: sha(bytes), ...(image ? {} : { artifact_text: text }) }));
}

test("session transport queues concurrent upstream calls and records unknown usage", async () => {
  const dir = setup("queue");
  const first = generateInSession(request()), second = generateInSession(request());
  const call1 = await pending(dir);
  expect(readdirSync(join(dir, ".cx/calls")).length).toBe(1);
  expect(JSON.parse(readFileSync(join(call1, "request.json"), "utf8")).system).toEqual(request().system);
  submit(call1, "first artifact");
  expect((await first).some(e => e.kind === "text-delta" && e.text === "first artifact")).toBe(true);
  const call2 = await pending(dir, 2); submit(call2, "second artifact"); await second;
  expect(JSON.parse(readFileSync(join(call2, "result.json"), "utf8")).usage).toBeNull();
});

test("a session reference response copies the submitted PNG unchanged", async () => {
  const dir = setup("reference"); const output = join(dir, "image.png");
  const work = generateSessionReference({ prompt: "Exact reference description", outputPath: output });
  const call = await pending(dir);
  expect(JSON.parse(readFileSync(join(call, "request.json"), "utf8")).kind).toBe("reference_image");
  submit(call, "", true);
  expect(await work).toBe(output);
  expect(readFileSync(output)).toEqual(readFileSync(join(call, "submitted.png")));
});

test("deadline closes a request; no silent subprocess fallback exists", async () => {
  const dir = setup("timeout", .02);
  await expect(generateInSession(request())).rejects.toThrow("timed out");
  const call = await pending(dir);
  expect(JSON.parse(readFileSync(join(call, "result.json"), "utf8")).status).toBe("error");
});

test("invalid identity cannot feed an artifact into the waiting pipeline", async () => {
  const dir = setup("identity"); const work = generateInSession(request());
  const outcome = work.then(() => null, error => error as Error);
  const call = await pending(dir); submit(call, "x");
  const path = join(call, "submission.json"); const value = JSON.parse(readFileSync(path, "utf8"));
  value.attempt_id = "old-attempt"; writeFileSync(path, JSON.stringify(value));
  expect((await outcome)?.message).toContain("identity");
});
