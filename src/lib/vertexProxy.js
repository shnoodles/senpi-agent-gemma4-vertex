/**
 * Local Vertex AI proxy server.
 *
 * Runs an HTTP server on a local port that accepts OpenAI-compatible
 * chat/completions requests and forwards them to Vertex AI rawPredict,
 * wrapping/unwrapping the Vertex AI "instances" format.
 *
 * When VERTEX_DEDICATED_DNS is set, uses the dedicated domain instead of
 * the shared aiplatform.googleapis.com domain. The API path and format
 * remain identical — only the hostname changes.
 *
 * Architecture:
 *   OpenClaw → http://127.0.0.1:{PORT}/v1/chat/completions (OpenAI format)
 *     → rawPredict (instances wrapped) on shared or dedicated domain
 *     → vLLM container (GPU)
 *     → unwrap predictions → OpenAI response back to OpenClaw
 */

import http from "node:http";

const VERTEX_PROXY_PORT = parseInt(process.env.VERTEX_PROXY_PORT || "7199", 10);

// Circular buffer for request/response logging (last 20 requests)
const REQUEST_LOG = [];
const MAX_LOG_ENTRIES = 20;
function logRequest(entry) {
  REQUEST_LOG.push({ ...entry, timestamp: new Date().toISOString() });
  if (REQUEST_LOG.length > MAX_LOG_ENTRIES) REQUEST_LOG.shift();
}
export function getRequestLog() {
  return REQUEST_LOG;
}

// Vertex AI endpoint config
const VERTEX_PROJECT = process.env.VERTEX_PROJECT || "vertex-test-492617";
const VERTEX_LOCATION = process.env.VERTEX_LOCATION || "us-central1";
const VERTEX_ENDPOINT_ID = process.env.VERTEX_ENDPOINT_ID || "mg-endpoint-f7db8545-2b55-4aff-b9b4-fcb26209917d";

// Dedicated DNS — uses dedicated domain instead of shared aiplatform.googleapis.com.
// Same API path and format, just different hostname. Bypasses the "use dedicated DNS" error.
const VERTEX_DEDICATED_DNS = process.env.VERTEX_DEDICATED_DNS || "";

function getRawPredictUrl() {
  const host = VERTEX_DEDICATED_DNS
    ? VERTEX_DEDICATED_DNS
    : `${VERTEX_LOCATION}-aiplatform.googleapis.com`;
  return `https://${host}/v1/projects/${VERTEX_PROJECT}/locations/${VERTEX_LOCATION}/endpoints/${VERTEX_ENDPOINT_ID}:rawPredict`;
}

let server = null;

/**
 * Forward an OpenAI-format request to Vertex AI rawPredict.
 */
async function handleChatCompletions(openaiBody) {
  const token = process.env.VERTEX_API_TOKEN;
  if (!token) {
    throw new Error("VERTEX_API_TOKEN not set — vertexAuth.js may not have refreshed yet");
  }

  // Always strip stream and stream_options — rawPredict doesn't support
  // streaming, and Vertex AI may strip stream but leave stream_options,
  // causing vLLM to reject with a validation error.
  const wantsStream = !!openaiBody.stream;
  delete openaiBody.stream;
  delete openaiBody.stream_options;
  // Strip non-standard fields that vLLM doesn't understand
  delete openaiBody.store;

  // Build Vertex AI instances wrapper
  const instance = { "@requestFormat": "chatCompletions" };
  for (const [key, value] of Object.entries(openaiBody)) {
    if (key !== "model" && key !== "stream" && key !== "stream_options") {
      instance[key] = value;
    }
  }

  const url = getRawPredictUrl();
  const shortUrl = VERTEX_DEDICATED_DNS ? `dedicated-dns/${VERTEX_ENDPOINT_ID}` : VERTEX_ENDPOINT_ID;
  const startTime = Date.now();
  console.log(`[vertex-proxy] → rawPredict: ${shortUrl}`);
  console.log(`[vertex-proxy] Request keys: ${Object.keys(instance).join(", ")}`);
  // Debug: log tool presence
  if (instance.tools) {
    console.log(`[vertex-proxy] Tools: ${instance.tools.length} function(s): ${instance.tools.map(t => t?.function?.name || "?").join(", ")}`);
  } else {
    console.log(`[vertex-proxy] Tools: NONE (no tools in request)`);
  }
  if (instance.tool_choice) {
    console.log(`[vertex-proxy] tool_choice: ${JSON.stringify(instance.tool_choice)}`);
  }

  // 120s timeout to avoid infinite hangs
  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), 120000);

  let res;
  try {
    res = await fetch(url, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        Authorization: `Bearer ${token}`,
      },
      body: JSON.stringify({ instances: [instance] }),
      signal: controller.signal,
    });
  } catch (fetchErr) {
    clearTimeout(timeout);
    if (fetchErr.name === "AbortError") {
      throw new Error(`rawPredict timed out after 120s`);
    }
    throw new Error(`rawPredict network error: ${fetchErr.cause?.code || fetchErr.cause?.message || fetchErr.message}`);
  }
  clearTimeout(timeout);

  const elapsed = ((Date.now() - startTime) / 1000).toFixed(1);
  console.log(`[vertex-proxy] ← response: ${res.status} (${elapsed}s)`);

  const text = await res.text();
  if (!res.ok) {
    throw new Error(`rawPredict failed (${res.status}): ${text}`);
  }

  // Unwrap: Vertex AI wraps the response in {"predictions": {...}}
  const parsed = JSON.parse(text);
  const result = parsed.predictions || parsed;

  // Clean up response for OpenClaw compatibility:
  // - Remove empty tool_calls arrays (confuses some clients)
  // - Log first choice content for debugging
  if (result?.choices) {
    for (const choice of result.choices) {
      if (choice?.message) {
        // Remove empty tool_calls array
        if (Array.isArray(choice.message.tool_calls) && choice.message.tool_calls.length === 0) {
          delete choice.message.tool_calls;
        }
        // Remove null fields that might confuse parsers
        for (const key of ["refusal", "annotations", "audio", "function_call", "reasoning"]) {
          if (choice.message[key] === null) {
            delete choice.message[key];
          }
        }
      }
    }
    const firstContent = result.choices[0]?.message?.content || "";
    const hasTool = !!result.choices[0]?.message?.tool_calls;
    const toolNames = result.choices[0]?.message?.tool_calls?.map(t => t.function?.name).join(", ") || "";
    console.log(`[vertex-proxy] ✓ ${elapsed}s | finish=${result.choices[0]?.finish_reason} | tools=${hasTool}${toolNames ? ` [${toolNames}]` : ""} | content=${firstContent.slice(0, 80).replace(/\n/g, "\\n")}...`);
  } else {
    console.log(`[vertex-proxy] ✓ ${elapsed}s | unexpected shape: ${Object.keys(result).join(",")}`);
  }

  return { result, wantsStream };
}

/**
 * Convert a non-streaming chat completion to SSE format.
 * This is needed when OpenClaw requests streaming but rawPredict only
 * returns non-streaming responses.
 */
function completionToSSE(completion) {
  // Convert the completion to a streaming chunk format
  const chunk = {
    id: completion.id,
    object: "chat.completion.chunk",
    created: completion.created,
    model: completion.model,
    choices: (completion.choices || []).map((c) => ({
      index: c.index,
      delta: {
        role: c.message?.role,
        content: c.message?.content || "",
        ...(c.message?.tool_calls ? { tool_calls: c.message.tool_calls } : {}),
      },
      finish_reason: c.finish_reason,
    })),
    ...(completion.usage ? { usage: completion.usage } : {}),
  };

  return `data: ${JSON.stringify(chunk)}\n\ndata: [DONE]\n\n`;
}

/**
 * Handle /v1/models endpoint.
 */
function handleModels() {
  return {
    object: "list",
    data: [
      {
        id: "gemma-4-31b-it",
        object: "model",
        owned_by: "vertex-ai",
        permission: [],
      },
    ],
  };
}

/**
 * Start the local proxy server.
 */
export function startVertexProxy() {
  return new Promise((resolve, reject) => {
    server = http.createServer(async (req, res) => {
      res.setHeader("Access-Control-Allow-Origin", "*");
      res.setHeader("Content-Type", "application/json");

      if (req.method === "OPTIONS") {
        res.writeHead(204);
        res.end();
        return;
      }

      const url = req.url || "";

      // Log every request for debugging
      console.log(`[vertex-proxy] ${req.method} ${url}`);

      // GET /v1/models
      if (req.method === "GET" && url.includes("/models")) {
        res.writeHead(200);
        res.end(JSON.stringify(handleModels()));
        return;
      }

      // POST /v1/chat/completions
      if (req.method === "POST" && (url.includes("/chat/completions") || url === "/")) {
        let body = "";
        for await (const chunk of req) body += chunk;

        try {
          const openaiRequest = JSON.parse(body);
          // Log the incoming request for debugging
          const toolNames = (openaiRequest.tools || []).map(t => t?.function?.name || "?");
          const msgCount = (openaiRequest.messages || []).length;
          const lastMsg = openaiRequest.messages?.[msgCount - 1];
          // Extract what tool was called (from assistant messages with tool_calls)
          const toolCallsMade = [];
          for (const m of (openaiRequest.messages || [])) {
            if (m.role === "assistant" && m.tool_calls) {
              for (const tc of m.tool_calls) {
                toolCallsMade.push({ name: tc.function?.name, args: tc.function?.arguments?.slice(0, 100) });
              }
            }
          }
          logRequest({
            type: "chat_completions",
            model: openaiRequest.model,
            stream: !!openaiRequest.stream,
            toolCount: toolNames.length,
            toolNames,
            tool_choice: openaiRequest.tool_choice || null,
            messageCount: msgCount,
            lastMessageRole: lastMsg?.role,
            lastMessagePreview: typeof lastMsg?.content === "string" ? lastMsg.content.slice(0, 200) : JSON.stringify(lastMsg?.content)?.slice(0, 200),
            toolCallsMade,
            hasStore: !!openaiRequest.store,
            requestKeys: Object.keys(openaiRequest),
          });
          const { result, wantsStream } = await handleChatCompletions(openaiRequest);

          if (wantsStream) {
            // Client wanted streaming — convert to SSE format
            console.log(`[vertex-proxy] Converting to SSE (client requested stream)`);
            res.writeHead(200, {
              "Content-Type": "text/event-stream",
              "Cache-Control": "no-cache",
              Connection: "keep-alive",
            });
            res.end(completionToSSE(result));
          } else {
            res.writeHead(200);
            res.end(JSON.stringify(result));
          }
        } catch (err) {
          console.error("[vertex-proxy] Error:", err.message);
          res.writeHead(502);
          res.end(JSON.stringify({
            error: {
              message: err.message,
              type: "proxy_error",
              code: 502,
            },
          }));
        }
        return;
      }

      // Catch-all: log unhandled routes with body for debugging
      let fallbackBody = "";
      for await (const chunk of req) fallbackBody += chunk;
      console.log(`[vertex-proxy] UNHANDLED ${req.method} ${url} body=${fallbackBody.slice(0, 500)}`);
      res.writeHead(404);
      res.end(JSON.stringify({ error: "Not found", path: url }));
    });

    server.listen(VERTEX_PROXY_PORT, "127.0.0.1", () => {
      const baseUrl = `http://127.0.0.1:${VERTEX_PROXY_PORT}/v1`;
      console.log(`[vertex-proxy] Listening on ${baseUrl}`);
      if (VERTEX_DEDICATED_DNS) {
        console.log(`[vertex-proxy] Mode: rawPredict via dedicated DNS (${VERTEX_DEDICATED_DNS})`);
      } else {
        console.log(`[vertex-proxy] Mode: rawPredict via shared domain`);
      }
      console.log(`[vertex-proxy] Endpoint: ${VERTEX_ENDPOINT_ID}`);
      resolve(baseUrl);
    });

    server.on("error", (err) => {
      console.error("[vertex-proxy] Server error:", err.message);
      reject(err);
    });
  });
}

export function stopVertexProxy() {
  if (server) {
    server.close();
    server = null;
    console.log("[vertex-proxy] Stopped");
  }
}

export function getVertexProxyBaseUrl() {
  return `http://127.0.0.1:${VERTEX_PROXY_PORT}/v1`;
}
