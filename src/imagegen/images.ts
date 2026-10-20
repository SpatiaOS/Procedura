/**
 * Reference-image generation — POST {base}/images/generations.
 *
 * OFF BY DEFAULT. Generating the reference costs money on an endpoint many
 * users will not have, and silently spending it is worse than asking: with no
 * image model configured, Procedura requires you to supply a reference with
 * `--image` (or to opt out of references entirely with `--no-image`) instead of
 * reaching for an image API you never asked it to use.
 *
 * Turning it on is naming the model — everything else reuses the chat
 * transport's endpoint configuration, so one endpoint covers the whole pipeline:
 *
 *   PROCEDURA_IMAGE_MODEL   e.g. gpt-image-1 — UNSET disables image generation
 *   OPENAI_API_KEY          required (the same key the chat transport uses)
 *   OPENAI_BASE_URL         default https://api.openai.com/v1
 *   PROCEDURA_IMAGE_API_URL full-URL override, for a gateway that puts image
 *                           generation somewhere other than /images/generations
 *
 * `--image-model M` on the CLI enables it for one run without touching `.env`.
 *
 * Reference images as *input* are not supported here (that would need
 * /images/edits + multipart); pass one with `--image` instead, which skips this
 * stage entirely.
 */

import { writeFileSync, mkdirSync } from "node:fs";
import { dirname } from "node:path";
import { generateSessionReference } from "../../mcp/runtime/session.ts";

export interface GenerateImageOpts {
  prompt: string;
  outputPath: string;
  model?: string;             // else $PROCEDURA_IMAGE_MODEL
  size?: string;              // "1024x1024", "1024x1792", etc.
  timeoutMs?: number;         // default 300_000
  downloadRetries?: number;   // default 5
  log?: (line: string) => void;
}

const DEFAULT_BASE_URL = process.env["OPENAI_BASE_URL"] ?? "https://api.openai.com/v1";

/** The image model for this run: an explicit override, else the configured
 *  default, else nothing — and nothing means image generation is off. */
export function resolveImageModel(override?: string): string | undefined {
  return override ?? process.env["PROCEDURA_IMAGE_MODEL"] ?? undefined;
}

/** Image generation needs BOTH a model to call and a key to call it with.
 *  Checked before a run starts so a missing reference fails in the first
 *  second rather than after the planner has already been paid for. */
export function imageGenAvailable(override?: string): boolean {
  if (process.env["PROCEDURA_MCP_SESSION"] === "1") return true;
  return Boolean(resolveImageModel(override) && process.env["OPENAI_API_KEY"]);
}

/** Why image generation is unavailable, and the three ways out of it. */
export function imageGenDisabledReason(override?: string): string {
  const missing = resolveImageModel(override)
    ? "OPENAI_API_KEY is not set"
    : "no image model is configured (PROCEDURA_IMAGE_MODEL is unset)";
  return (
    `Reference-image generation is off: ${missing}.\n` +
    `  Supply a reference instead:  --image <path>\n` +
    `  Or run without one at all:   --no-image  (text-only, needs --incremental)\n` +
    `  Or enable generation:        set PROCEDURA_IMAGE_MODEL (e.g. gpt-image-1)\n` +
    `                               and OPENAI_API_KEY in .env, or pass --image-model`
  );
}

function endpointUrl(): string {
  const explicit = process.env["PROCEDURA_IMAGE_API_URL"];
  if (explicit) return explicit;
  const base = DEFAULT_BASE_URL.replace(/\/+$/, "");
  return `${base}/images/generations`;
}

function normSize(s: string | undefined): string {
  if (!s) return "1024x1024";
  return /\dx\d/.test(s.toLowerCase()) ? s : "1024x1024";
}

interface ImagesGenerationsResponse {
  data?: Array<{ b64_json?: string; url?: string }>;
  error?: unknown;
}

export async function generateImage(opts: GenerateImageOpts): Promise<string> {
  if (process.env["PROCEDURA_MCP_SESSION"] === "1") return generateSessionReference(opts);
  const apiKey = process.env["OPENAI_API_KEY"];
  if (!apiKey) {
    throw new Error("OPENAI_API_KEY not set in env / .env");
  }
  const url = endpointUrl();
  const model = resolveImageModel(opts.model);
  if (!model) throw new Error(imageGenDisabledReason(opts.model));
  const size = normSize(opts.size);
  const log = opts.log ?? (() => undefined);
  const timeoutMs = opts.timeoutMs ?? 300_000;

  log(`      [image-gen] ${model} via ${new URL(url).host} (size ${size})`);

  // RETRY the generation call. The download loop below always retried, but this
  // request did not -- and one transient upstream hiccup threw straight out of
  // the pipeline, killing a whole run with no mesh to show for it. In practice
  // these failures are near-always retryable transport errors: a 500 "unexpected
  // EOF", a closed socket, a timeout. The LLM path gets 5 attempts via
  // longTimeoutFetch; this one gets 4.
  const tries = Number(process.env["PROCEDURA_IMAGEGEN_ATTEMPTS"] ?? 4);
  let res!: Response;
  let lastErr: Error | undefined;
  for (let attempt = 1; attempt <= tries; attempt++) {
    try {
      res = await fetch(url, {
        method: "POST",
        headers: {
          "content-type": "application/json",
          "authorization": `Bearer ${apiKey}`,
        },
        body: JSON.stringify({ model, prompt: opts.prompt, n: 1, size }),
        signal: AbortSignal.timeout(timeoutMs),
      });
      if (res.ok) { lastErr = undefined; break; }
      const body = await res.text().catch(() => "");
      lastErr = new Error(`images/generations failed: ${res.status} ${res.statusText} — ${body.slice(0, 300)}`);
      // 4xx other than 429 is our request being wrong; retrying cannot fix it.
      if (res.status < 500 && res.status !== 429) throw lastErr;
    } catch (e) {
      lastErr = e as Error;
      if ((e as Error).message?.includes("images/generations failed") &&
          !/\b(429|5\d\d)\b/.test((e as Error).message)) throw e;
    }
    if (attempt < tries) {
      const wait = 4000 * attempt;
      log(`      [image-gen] attempt ${attempt}/${tries} failed: ${String(lastErr?.message).slice(0, 120)}; retrying in ${wait}ms`);
      await new Promise((r) => setTimeout(r, wait));
    }
  }
  if (lastErr) throw lastErr;
  const json = (await res.json()) as ImagesGenerationsResponse;
  if (json.error) {
    throw new Error(`image endpoint returned error: ${JSON.stringify(json.error)}`);
  }
  const item = (json.data ?? [])[0];
  if (!item) {
    throw new Error(`image endpoint returned no data: ${JSON.stringify(json).slice(0, 300)}`);
  }

  mkdirSync(dirname(opts.outputPath), { recursive: true });
  if (item.b64_json) {
    const bytes = Buffer.from(item.b64_json, "base64");
    writeFileSync(opts.outputPath, bytes);
    log(`      [image-gen] saved ${bytes.length} bytes inline → ${opts.outputPath}`);
    return opts.outputPath;
  }
  if (item.url) {
    const tries = opts.downloadRetries ?? 5;
    let lastErr: Error | undefined;
    for (let attempt = 1; attempt <= tries; attempt++) {
      try {
        const dl = await fetch(item.url, { signal: AbortSignal.timeout(120_000) });
        if (!dl.ok) throw new Error(`download ${dl.status} ${dl.statusText}`);
        const buf = Buffer.from(await dl.arrayBuffer());
        writeFileSync(opts.outputPath, buf);
        log(`      [image-gen] downloaded ${buf.length} bytes → ${opts.outputPath}`);
        return opts.outputPath;
      } catch (e) {
        lastErr = e as Error;
        if (attempt < tries) {
          await new Promise((r) => setTimeout(r, 10_000));
        }
      }
    }
    throw lastErr ?? new Error("download failed after retries");
  }
  throw new Error("image response item had neither b64_json nor url");
}
