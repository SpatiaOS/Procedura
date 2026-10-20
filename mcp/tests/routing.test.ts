import { afterEach, expect, test } from "bun:test";
import { createLLMClient, Auth, Endpoint, Route, Framing, OpenAIChat } from "@harness/template";
import { createSessionClient } from "../runtime/session.ts";

const key = Symbol.for("procedura.mcp.client");
const old = process.env.PROCEDURA_MCP_SESSION;
const originalFactory = Reflect.get(globalThis, key);
afterEach(() => {
  if (old === undefined) delete process.env.PROCEDURA_MCP_SESSION;
  else process.env.PROCEDURA_MCP_SESSION = old;
  if (originalFactory === undefined) Reflect.deleteProperty(globalThis, key);
  else Reflect.set(globalThis, key, originalFactory);
});

test("MCP mode refuses an unconfigured session instead of falling back to an API", () => {
  process.env.PROCEDURA_MCP_SESSION = "1";
  Reflect.deleteProperty(globalThis, key);
  expect(() => createLLMClient()).toThrow("refusing API fallback");
});

test("every harness client uses the configured session transport in MCP mode", () => {
  process.env.PROCEDURA_MCP_SESSION = "1";
  const session = createSessionClient();
  Reflect.set(globalThis, key, () => session);
  expect(createLLMClient()).toBe(session);
  expect(createLLMClient({ maxAttempts: 1 })).toBe(session);
});

test("ordinary API mode still prepares its original route", async () => {
  delete process.env.PROCEDURA_MCP_SESSION;
  Reflect.set(globalThis, key, () => { throw new Error("must not use MCP"); });
  const client = createLLMClient();
  const route = Route.make({ id: "fixture", protocol: OpenAIChat, endpoint: Endpoint.path("/v1/chat/completions"),
    auth: Auth.passthrough, framing: Framing.sse, baseUrl: "https://example.invalid" });
  const prepared = await client.prepare(route, { model: { providerId: "fixture", modelId: "test" }, messages: [] });
  expect(prepared.url).toBe("https://example.invalid/v1/chat/completions");
});
