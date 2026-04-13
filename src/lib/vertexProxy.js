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

// Vertex AI endpoint config
const VERTEX_PROJECT = process.env.VERTEX_PROJECT || "vertex-test-492617";
const VERTEX_LOCATION = process.env.VERTEX_LOCATION || "us-central1";
const VERTEX_ENDPOINT_ID = process.env.VERTEX_ENDPOINT_ID || "mg-endpoint-9b2a3306-a284-4942-b51d-87dee5d4b7c5";

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

  // Strip fields that vLLM rejects when stream is not true
  if (!openaiBody.stream) {
    delete openaiBody.stream_options;
  }

  // Build Vertex AI instances wrapper
  const instance = { "@requestFormat": "chatCompletions" };
  for (const [key, value] of Object.entries(openaiBody)) {
    if (key !== "model") instance[key] = value;
  }

  const url = getRawPredictUrl();
  const shortUrl = VERTEX_DEDICATED_DNS ? `dedicated-dns/${VERTEX_ENDPOINT_ID}` : VERTEX_ENDPOINT_ID;
  console.log(`[vertex-proxy] → rawPredict: ${shortUrl}`);

  let res;
  try {
    res = await fetch(url, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        Authorization: `Bearer ${token}`,
      },
      body: JSON.stringify({ instances: [instance] }),
    });
  } catch (fetchErr) {
    throw new Error(`rawPredict network error: ${fetchErr.cause?.code || fetchErr.cause?.message || fetchErr.message}`);
  }

  const text = await res.text();
  if (!res.ok) {
    throw new Error(`rawPredict failed (${res.status}): ${text}`);
  }

  // Unwrap: Vertex AI wraps the response in {"predictions": {...}}
  const parsed = JSON.parse(text);
  return parsed.predictions || parsed;
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
