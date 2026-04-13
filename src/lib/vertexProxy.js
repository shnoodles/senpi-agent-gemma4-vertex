/**
 * Local Vertex AI proxy server.
 *
 * Runs an HTTP server on a local port that accepts OpenAI-compatible
 * chat/completions requests and forwards them to Vertex AI rawPredict,
 * wrapping/unwrapping the Vertex AI "instances" format.
 *
 * This is needed because Vertex AI Model Garden dedicated endpoints have
 * a broken DNS proxy layer. The Google Cloud SDK (gRPC) bypasses this,
 * but OpenClaw needs a plain HTTP endpoint. This local proxy bridges
 * the gap using the rawPredict REST API with OAuth tokens from vertexAuth.js.
 *
 * Architecture:
 *   OpenClaw → http://127.0.0.1:{PORT}/v1/chat/completions (OpenAI format)
 *     → Vertex AI rawPredict (instances format)
 *     → vLLM container (GPU)
 *     → unwrapped OpenAI response back to OpenClaw
 */

import http from "node:http";

const VERTEX_PROXY_PORT = parseInt(process.env.VERTEX_PROXY_PORT || "7199", 10);

// Vertex AI rawPredict endpoint. Uses REST API (not gRPC) with OAuth token.
const VERTEX_PROJECT = process.env.VERTEX_PROJECT || "vertex-test-492617";
const VERTEX_LOCATION = process.env.VERTEX_LOCATION || "us-central1";
const VERTEX_ENDPOINT_ID = process.env.VERTEX_ENDPOINT_ID || "mg-endpoint-5cb880c9-c56f-4d80-a8ed-028e809363c1";

function getRawPredictUrl() {
  return `https://${VERTEX_LOCATION}-aiplatform.googleapis.com/v1/projects/${VERTEX_PROJECT}/locations/${VERTEX_LOCATION}/endpoints/${VERTEX_ENDPOINT_ID}:rawPredict`;
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

  // Build Vertex AI instances wrapper
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
      console.log(`[vertex-proxy] Forwarding to Vertex AI endpoint: ${VERTEX_ENDPOINT_ID}`);
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
