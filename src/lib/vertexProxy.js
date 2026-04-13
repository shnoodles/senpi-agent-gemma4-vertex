/**
 * Local Vertex AI proxy server.
 *
 * Runs an HTTP server on a local port that accepts OpenAI-compatible
 * chat/completions requests and forwards them to the Vertex AI vLLM endpoint.
 *
 * Supports TWO modes (auto-detected):
 *
 * 1. DEDICATED DNS (preferred): When VERTEX_DEDICATED_DNS is set, calls the
 *    dedicated endpoint directly with standard OpenAI format. No wrapping needed.
 *    Example: https://<dedicated-dns>/v1/chat/completions
 *
 * 2. rawPredict (fallback): When no dedicated DNS, uses the shared Vertex AI
 *    domain with rawPredict API, wrapping in instances format.
 *
 * Architecture:
 *   OpenClaw → http://127.0.0.1:{PORT}/v1/chat/completions (OpenAI format)
 *     → Dedicated DNS (direct OpenAI) OR rawPredict (instances wrapped)
 *     → vLLM container (GPU)
 *     → response back to OpenClaw
 */

import http from "node:http";

const VERTEX_PROXY_PORT = parseInt(process.env.VERTEX_PROXY_PORT || "7199", 10);

// Vertex AI endpoint config
const VERTEX_PROJECT = process.env.VERTEX_PROJECT || "vertex-test-492617";
const VERTEX_LOCATION = process.env.VERTEX_LOCATION || "us-central1";
const VERTEX_ENDPOINT_ID = process.env.VERTEX_ENDPOINT_ID || "mg-endpoint-5efab3bd-c06e-4cd3-bb6d-3a26b0ea5f93";

// Dedicated DNS — set this to skip rawPredict and call the endpoint directly.
// Example: "4482718252291588096.us-central1-518090926810.prediction.vertexai.goog"
const VERTEX_DEDICATED_DNS = process.env.VERTEX_DEDICATED_DNS || "";

function useDedicatedDns() {
  return VERTEX_DEDICATED_DNS.length > 0;
}

function getDedicatedDnsUrl(path) {
  return `https://${VERTEX_DEDICATED_DNS}${path}`;
}

function getRawPredictUrl() {
  return `https://${VERTEX_LOCATION}-aiplatform.googleapis.com/v1/projects/${VERTEX_PROJECT}/locations/${VERTEX_LOCATION}/endpoints/${VERTEX_ENDPOINT_ID}:rawPredict`;
}

let server = null;

/**
 * Forward an OpenAI-format request to Vertex AI.
 * Uses dedicated DNS if available, otherwise falls back to rawPredict.
 */
async function handleChatCompletions(openaiBody) {
  const token = process.env.VERTEX_API_TOKEN;
  if (!token) {
    throw new Error("VERTEX_API_TOKEN not set — vertexAuth.js may not have refreshed yet");
  }

  if (useDedicatedDns()) {
    // ── Dedicated DNS mode: call OpenAI-compatible endpoint directly ──
    const url = getDedicatedDnsUrl("/v1/chat/completions");
    console.log(`[vertex-proxy] → Dedicated DNS: ${VERTEX_DEDICATED_DNS}`);

    const res = await fetch(url, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        Authorization: `Bearer ${token}`,
      },
      body: JSON.stringify(openaiBody),
    });

    const text = await res.text();
    if (!res.ok) {
      throw new Error(`Vertex AI dedicated endpoint failed (${res.status}): ${text}`);
    }

    return JSON.parse(text);
  } else {
    // ── rawPredict mode: wrap in instances format ──
    const instance = { "@requestFormat": "chatCompletions" };
    for (const [key, value] of Object.entries(openaiBody)) {
      if (key !== "model") {
        instance[key] = value;
      }
    }

    const vertexBody = JSON.stringify({ instances: [instance] });

    const res = await fetch(getRawPredictUrl(), {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        Authorization: `Bearer ${token}`,
      },
      body: vertexBody,
    });

    const text = await res.text();
    if (!res.ok) {
      throw new Error(`Vertex AI rawPredict failed (${res.status}): ${text}`);
    }

    // Unwrap: Vertex AI wraps the response in {"predictions": {...}}
    const parsed = JSON.parse(text);
    return parsed.predictions || parsed;
  }
}

/**
 * Handle /v1/models endpoint.
 */
function handleModels() {
  return {
    object: "list",
    data: [
      {
        id: "qwen3.5-35b-a3b",
        object: "model",
        owned_by: "vertex-ai",
        permission: [],
      },
    ],
  };
}

/**
 * Start the local proxy server.
 * Returns a promise that resolves with the base URL once listening.
 */
export function startVertexProxy() {
  return new Promise((resolve, reject) => {
    server = http.createServer(async (req, res) => {
      // CORS
      res.setHeader("Access-Control-Allow-Origin", "*");
      res.setHeader("Content-Type", "application/json");

      if (req.method === "OPTIONS") {
        res.writeHead(204);
        res.end();
        return;
      }

      const url = req.url || "";

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
          const openaiResponse = await handleChatCompletions(openaiRequest);
          res.writeHead(200);
          res.end(JSON.stringify(openaiResponse));
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

      // Fallback
      res.writeHead(404);
      res.end(JSON.stringify({ error: "Not found" }));
    });

    server.listen(VERTEX_PROXY_PORT, "127.0.0.1", () => {
      const baseUrl = `http://127.0.0.1:${VERTEX_PROXY_PORT}/v1`;
      console.log(`[vertex-proxy] Listening on ${baseUrl}`);
      if (useDedicatedDns()) {
        console.log(`[vertex-proxy] Mode: DEDICATED DNS → ${VERTEX_DEDICATED_DNS}`);
      } else {
        console.log(`[vertex-proxy] Mode: rawPredict → endpoint ${VERTEX_ENDPOINT_ID}`);
      }
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
