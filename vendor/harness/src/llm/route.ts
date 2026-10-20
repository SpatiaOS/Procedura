/**
 * Route.make — four-axis decomposition of an LLM call.
 *
 * Pattern source: opencode packages/llm/src/route/client.ts:358-398.
 *
 *   Route = { Protocol, Endpoint, Auth, Framing }
 *
 * Adding a new provider:
 *   const myProvider = Route.make({
 *     id: "deepseek-chat",
 *     protocol: OpenAIChat.protocol,    // reuse verbatim
 *     endpoint: Endpoint.path("/v1/chat/completions"),
 *     auth: Auth.bearer("DEEPSEEK_API_KEY"),
 *     framing: Framing.sse,
 *     baseUrl: "https://api.deepseek.com",
 *   });
 *
 * Five-line provider files become possible because everything orthogonal
 * is factored out into the route axes.
 */

import type {
  CanonicalRequest, LLMEvent, Protocol,
} from "./protocol.ts";

/** A mid-stream failure worth resuming from a partial response (network/stream
 *  timeout, socket reset, gateway termination) — as opposed to a hard error or
 *  a caller-initiated abort, neither of which should be resumed. */
function isResumableInterruption(e: Error): boolean {
  const s = `${e.name}: ${e.message}`.toLowerCase();
  return (
    s.includes("timeout") || s.includes("timed out") ||
    s.includes("terminated") || s.includes("econnreset") ||
    s.includes("socket") || s.includes("fetch failed") || s.includes("network")
  );
}

// ──────────────────────────────────────────────────────────────────────────
// Endpoint, Auth, Framing
// ──────────────────────────────────────────────────────────────────────────

export type EndpointResolver = string | ((req: CanonicalRequest) => string);

export const Endpoint = {
  path(p: string): EndpointResolver { return p; },
  dynamic(fn: (req: CanonicalRequest) => string): EndpointResolver { return fn; },
};

export interface AuthApplier {
  apply(headers: Headers, req: CanonicalRequest): Promise<void>;
}

export const Auth = {
  /** Bearer token from env var (default name matches uppercase provider id). */
  bearer(envVar: string): AuthApplier {
    return {
      async apply(headers) {
        const token = (globalThis as { process?: { env?: Record<string, string | undefined> } })
          .process?.env?.[envVar];
        if (!token) throw new Error(`Missing env var ${envVar}`);
        headers.set("Authorization", `Bearer ${token}`);
      },
    };
  },
  apiKeyHeader(headerName: string, envVar: string): AuthApplier {
    return {
      async apply(headers) {
        const token = (globalThis as { process?: { env?: Record<string, string | undefined> } })
          .process?.env?.[envVar];
        if (!token) throw new Error(`Missing env var ${envVar}`);
        headers.set(headerName, token);
      },
    };
  },
  /** No auth header — useful for local/lmstudio/ollama. */
  passthrough: { async apply() { /* noop */ } } satisfies AuthApplier,
  /** Custom (e.g. SigV4 for Bedrock). */
  custom(applier: AuthApplier): AuthApplier { return applier; },
};

export interface Framing<Raw = unknown> {
  /** Convert raw fetch body stream into a Raw event iterator. */
  parse(body: ReadableStream<Uint8Array> | null): AsyncIterable<Raw>;
}

/** Standard SSE framing (`text/event-stream`). */
export const Framing = {
  sse: makeSSE(),
} as const;

function makeSSE(): Framing<{ event: string; data: string }> {
  return {
    async *parse(body) {
      if (!body) return;
      const reader = body.getReader();
      const decoder = new TextDecoder();
      let buf = "";
      for (;;) {
        const { value, done } = await reader.read();
        if (done) break;
        buf += decoder.decode(value, { stream: true });
        const lines = buf.split(/\r?\n/);
        buf = lines.pop() ?? "";
        let event = "message", data = "";
        for (const line of lines) {
          if (line === "") {
            if (data) yield { event, data };
            event = "message"; data = "";
          } else if (line.startsWith("event:")) {
            event = line.slice(6).trim();
          } else if (line.startsWith("data:")) {
            data += (data ? "\n" : "") + line.slice(5).trim();
          }
        }
      }
    },
  };
}

// ──────────────────────────────────────────────────────────────────────────
// Route — composes the four axes into a runnable client
// ──────────────────────────────────────────────────────────────────────────

export interface RouteDef<Raw = unknown> {
  id: string;
  protocol: Protocol<Raw>;
  endpoint: EndpointResolver;
  auth: AuthApplier;
  framing: Framing<Raw>;
  baseUrl: string;
  /** Additional headers; merged before auth applies. */
  defaultHeaders?: Record<string, string>;
}

export interface PreparedRequest {
  url: string;
  method: "POST";
  headers: Headers;
  body: string;
}

export interface LLMClient {
  prepare(route: RouteDef<unknown>, req: CanonicalRequest): Promise<PreparedRequest>;
  stream(route: RouteDef<unknown>, req: CanonicalRequest): AsyncIterable<LLMEvent>;
  generate(route: RouteDef<unknown>, req: CanonicalRequest): Promise<LLMEvent[]>;
}

// Global registry — loading a provider module side-effects registration.
const routeRegistry = new Map<string, RouteDef<unknown>>();

export const Route = {
  make<Raw>(def: RouteDef<Raw>): RouteDef<Raw> {
    routeRegistry.set(def.id, def as RouteDef<unknown>);
    return def;
  },
  get(id: string): RouteDef<unknown> | undefined { return routeRegistry.get(id); },
  list(): string[] { return [...routeRegistry.keys()]; },
};

// ──────────────────────────────────────────────────────────────────────────
// Reference LLMClient implementation — fetch + framing + protocol
// ──────────────────────────────────────────────────────────────────────────

export function createLLMClient(opts: {
  /** Optional fetch override for proxy-capture / replay. */
  fetch?: typeof fetch;
  /** Transport retry attempts for this client (default 4). Pass 1 when the
   *  injected fetch does its own retrying — stacking two retry layers
   *  multiplies into attempts² POSTs against a failing gateway. */
  maxAttempts?: number;
} = {}): LLMClient {
  if (process.env["PROCEDURA_MCP_SESSION"] === "1") {
    const factory: unknown = Reflect.get(globalThis, Symbol.for("procedura.mcp.client"));
    if (typeof factory !== "function") throw new Error("Procedura MCP session was not configured; refusing API fallback");
    return factory() as LLMClient;
  }
  const f = opts.fetch ?? fetch;
  const MAX_ATTEMPTS = Math.max(1, opts.maxAttempts ?? 4);

  async function prep(route: RouteDef<unknown>, req: CanonicalRequest): Promise<PreparedRequest> {
    const headers = new Headers(route.defaultHeaders ?? {});
    headers.set("content-type", "application/json");
    await route.auth.apply(headers, req);
    const pathOrFn = route.endpoint;
    const path = typeof pathOrFn === "string" ? pathOrFn : pathOrFn(req);
    return {
      url: route.baseUrl.replace(/\/$/, "") + path,
      method: "POST",
      headers,
      body: JSON.stringify(route.protocol.buildBody(req)),
    };
  }

  // Transient HTTP statuses worth retrying with backoff (Cloudflare 5xx
   // upstream errors are common in front of LLM gateways).
  const RETRYABLE = new Set([408, 429, 500, 502, 503, 504, 520, 521, 522, 524, 525]);

  /** Errors that must NEVER be re-submitted: the caller aborted, or the
   *  request already consumed its full generation budget (re-posting would
   *  re-bill an expensive generation that will be discarded again). */
  function isTerminal(e: Error, signal?: AbortSignal): boolean {
    return signal?.aborted === true || e.name === "AbortError" || e.name === "LlmBudgetExceeded";
  }

  async function fetchWithRetry(
    prepared: PreparedRequest, signal?: AbortSignal,
  ): Promise<Response> {
    let lastErr: Error | undefined;
    for (let attempt = 1; attempt <= MAX_ATTEMPTS; attempt++) {
      try {
        const res = await f(prepared.url, {
          method: prepared.method,
          headers: prepared.headers,
          body: prepared.body,
          ...(signal !== undefined ? { signal } : {}),
        });
        if (res.ok) return res;
        if (!RETRYABLE.has(res.status) || attempt === MAX_ATTEMPTS) return res;
        const body = await res.text().catch(() => "");
        lastErr = new Error(`${res.status} ${res.statusText}: ${body.slice(0, 200)}`);
      } catch (e) {
        if (attempt === MAX_ATTEMPTS || isTerminal(e as Error, signal)) throw e;
        lastErr = e as Error;
      }
      if (signal?.aborted) throw lastErr ?? new Error("aborted");
      // Exponential backoff with full jitter — N parallel callers hitting a
      // 503ing gateway must not retry in lockstep.
      const delayMs = Math.round(500 * Math.pow(2, attempt - 1) * (0.5 + Math.random()));
      console.error(`  [llm] attempt ${attempt}/${MAX_ATTEMPTS} failed: ${lastErr?.message ?? "unknown"}; retrying in ${delayMs}ms`);
      await new Promise((r) => setTimeout(r, delayMs));
    }
    throw lastErr ?? new Error("retries exhausted");
  }

  return {
    prepare: prep,
    async *stream(route, req) {
      // Resume-on-interruption: if the stream dies mid-flight (network/stream
      // timeout) or the model is cut off by finish_reason=length, re-issue with
      // the partial output as an assistant turn + a "continue" nudge, and stitch
      // the segments together. We do
      // NOT resume across an in-flight tool call (can't reconstruct partial args
      // safely) or a caller-initiated abort.
      const MAX_CONTINUATIONS = 2;
      const CONTINUE_PROMPT =
        "Your previous response was cut off before it finished. Continue EXACTLY " +
        "where you left off — do not repeat anything you already wrote.";

      let currentReq = req;
      let carriedText = "";   // visible content accumulated across all segments

      for (let cont = 0; ; cont++) {
        const prepared = await prep(route, currentReq);
        let res: Response;
        try {
          res = await fetchWithRetry(prepared, req.signal);
        } catch (e) {
          yield { kind: "error", error: e as Error, recoverable: false };
          return;
        }
        if (!res.ok) {
          const text = await res.text().catch(() => "");
          yield { kind: "error", error: new Error(`${res.status} ${res.statusText}: ${text}`), recoverable: false };
          return;
        }

        let segmentText = "";
        let sawToolCall = false;
        let finishReason: string | undefined;
        let pendingFinish: LLMEvent | undefined;   // buffered so we don't signal completion prematurely
        let interrupted = false;

        try {
          for await (const raw of route.framing.parse(res.body)) {
            for (const ev of route.protocol.parseEvent(raw)) {
              if (ev.kind === "tool-call-start" || ev.kind === "tool-call-delta") sawToolCall = true;
              if (ev.kind === "text-delta") segmentText += ev.text;
              if (ev.kind === "message-finish") {
                // Some gateways (e.g. Gemini behind an OpenAI-compatible proxy) append a
                // spurious finish_reason="stop" chunk AFTER the real tool_calls
                // finish. tool-calls must win: a later non-tool-calls finish
                // must not overwrite it, or the loop stops after executing one
                // tool instead of continuing the agent turn.
                if (finishReason === "tool-calls" && ev.reason !== "tool-calls") continue;
                finishReason = ev.reason; pendingFinish = ev; continue;
              }
              yield ev;
            }
            if (route.protocol.isTerminal?.(raw)) break;
          }
        } catch (e) {
          // Caller abort is intentional; non-timeout errors are hard. Neither resumes.
          if (req.signal?.aborted || !isResumableInterruption(e as Error)) {
            yield { kind: "error", error: e as Error, recoverable: false };
            return;
          }
          interrupted = true;
        }

        carriedText += segmentText;
        const truncated = finishReason === "length";

        // Clean finish → emit the buffered message-finish and stop.
        if (!interrupted && !truncated) {
          if (pendingFinish) yield pendingFinish;
          return;
        }
        // Can't safely resume across a partial tool call.
        if (sawToolCall) {
          if (pendingFinish) { yield pendingFinish; return; }
          yield { kind: "error", error: new Error("stream interrupted mid-tool-call; cannot resume"), recoverable: false };
          return;
        }
        // No budget left, or nothing to anchor a continuation → stop with what we have.
        if (cont >= MAX_CONTINUATIONS || !carriedText) {
          if (pendingFinish) { yield pendingFinish; return; }
          yield { kind: "error", error: new Error("stream interrupted and could not be resumed"), recoverable: false };
          return;
        }

        // Resume: replay the partial as an assistant turn + a continue nudge.
        currentReq = {
          ...req,
          messages: [
            ...req.messages,
            { role: "assistant", content: [{ kind: "text", text: carriedText }] },
            { role: "user", content: [{ kind: "text", text: CONTINUE_PROMPT }] },
          ],
        };
      }
    },
    async generate(route, req) {
      const events: LLMEvent[] = [];
      for await (const ev of this.stream(route, req)) events.push(ev);
      return events;
    },
  };
}
