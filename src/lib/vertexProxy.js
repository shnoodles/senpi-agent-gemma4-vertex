/**
 * Local Vertex AI proxy server.
 *
 * Runs an HTTP server on a local port that accepts OpenAI-compatible
 * chat/completions requests and forwards them to the Vertex AI vLLM endpoint.
 *
 * Supports THREE modes (in priority order):
 *
 * 1. DEDICATED DNS: When VERTEX_DEDICATED_DNS is set, calls the dedicated
 *    endpoint directly with standard OpenAI format. Only works from within GCP.
 *
 * 2. rawPredict REST: Default mode. Uses shared Vertex AI domain with
 *    rawPredict API, wrapping in instances format. Works when dedicated DNS
 *    is DISABLED on the endpoint.
 *
 * 3. Python SDK (gRPC): Fallback when VERTEX_USE_PYTHON_SDK=true. Shells out
 *    to a Python script using google-cloud-aiplatform SDK, which handles
 *    dedicated DNS routing via gRPC transparently. Works from anywhere.
 *
 * Architecture:
 *   OpenClaw → http://127.0.0.1:{PORT}/v1/chat/completions (OpenAI format)
 *     → Vertex AI (via one of 3 modes above)
 *     → vLLM container (GPU)
 *     → response back to OpenClaw
 */

import http from "node:http";
import { execFile } from "node:child_process";
import { writeFileSync, mkdtempSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

const VERTEX_PROXY_PORT = parseInt(process.env.VERTEX_PROXY_PORT || "7199", 10);

// Vertex AI endpoint config
const VERTEX_PROJECT = process.env.VERTEX_PROJECT || "vertex-test-492617";
const VERTEX_LOCATION = process.env.VERTEX_LOCATION || "us-central1";
const VERTEX_ENDPOINT_ID = process.env.VERTEX_ENDPOINT_ID || "mg-endpoint-5efab3bd-c06e-4cd3-bb6d-3a26b0ea5f93";

// Dedicated DNS — set this to call the endpoint's dedicated domain directly.
const VERTEX_DEDICATED_DNS = process.env.VERTEX_DEDICATED_DNS || "";

// Python SDK mode — set VERTEX_USE_PYTHON_SDK=true to use gRPC via google-cloud-aiplatform
const VERTEX_USE_PYTHON_SDK = (process.env.VERTEX_USE_PYTHON_SDK || "").toLowerCase() === "true";

function getMode() {
  if (VERTEX_DEDICATED_DNS) return "dedicated-dns";
  if (VERTEX_USE_PYTHON_SDK) return "python-sdk";
  return "rawpredict";
}

function getDedicatedDnsUrl(path) {
  return `https://${VERTEX_DEDICATED_DNS}${path}`;
}

function getRawPredictUrl() {
  return `https://${VERTEX_LOCATION}-aiplatform.googleapis.com/v1/projects/${VERTEX_PROJECT}/locations/${VERTEX_LOCATION}/endpoints/${VERTEX_ENDPOINT_ID}:rawPredict`;
}

let server = null;

/**
 * Forward via rawPredict REST API (shared domain).
 * Wraps request in instances format, unwraps predictions.
 */
async function handleViaRawPredict(openaiBody) {
  const token = process.env.VERTEX_API_TOKEN;
  if (!token) throw new Error("VERTEX_API_TOKEN not set");

  const instance = { "@requestFormat": "chatCompletions" };
  for (const [key, value] of Object.entries(openaiBody)) {
    if (key !== "model") instance[key] = value;
  }

  const url = getRawPredictUrl();
  console.log(`[vertex-proxy] → rawPredict: ${url.split("/endpoints/")[1]}`);

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
    throw new Error(`rawPredict network error: ${fetchErr.cause?.code || fetchErr.message}`);
  }

  const text = await res.text();
  if (!res.ok) throw new Error(`rawPredict failed (${res.status}): ${text}`);

  const parsed = JSON.parse(text);
  return parsed.predictions || parsed;
}

/**
 * Forward via dedicated DNS (direct OpenAI-compatible endpoint).
 * Only reachable from within GCP network.
 */
async function handleViaDedicatedDns(openaiBody) {
  const token = process.env.VERTEX_API_TOKEN;
  if (!token) throw new Error("VERTEX_API_TOKEN not set");

  const url = getDedicatedDnsUrl("/v1/chat/completions");
  console.log(`[vertex-proxy] → Dedicated DNS: ${VERTEX_DEDICATED_DNS}`);

  let res;
  try {
    res = await fetch(url, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        Authorization: `Bearer ${token}`,
      },
      body: JSON.stringify(openaiBody),
    });
  } catch (fetchErr) {
    throw new Error(`Dedicated DNS unreachable: ${fetchErr.cause?.code || fetchErr.message}`);
  }

  const text = await res.text();
  if (!res.ok) throw new Error(`Dedicated endpoint failed (${res.status}): ${text}`);

  return JSON.parse(text);
}

/**
 * Forward via Python google-cloud-aiplatform SDK (gRPC).
 * Handles dedicated DNS routing transparently. Works from anywhere.
 */
function handleViaPythonSdk(openaiBody) {
  return new Promise((resolve, reject) => {
    // Build the inline Python script
    const instance = { "@requestFormat": "chatCompletions" };
    for (const [key, value] of Object.entries(openaiBody)) {
      if (key !== "model") instance[key] = value;
    }

    const pyScript = `
import json, sys, os
os.environ["GOOGLE_CLOUD_PROJECT"] = "${VERTEX_PROJECT}"

from google.oauth2.credentials import Credentials
from google.cloud import aiplatform

# Build credentials from access token
creds = Credentials(token="${(process.env.VERTEX_API_TOKEN || "").replace(/"/g, '\\"')}")
aiplatform.init(project="${VERTEX_PROJECT}", location="${VERTEX_LOCATION}", credentials=creds)

endpoint = aiplatform.Endpoint("projects/${VERTEX_PROJECT}/locations/${VERTEX_LOCATION}/endpoints/${VERTEX_ENDPOINT_ID}")
payload = json.loads(sys.stdin.read())
resp = endpoint.raw_predict(
    body=json.dumps({"instances": [payload]}).encode("utf-8"),
    headers={"Content-Type": "application/json"}
)
result = json.loads(resp.text)
# Unwrap predictions wrapper
output = result.get("predictions", result)
print(json.dumps(output))
`;

    const child = execFile("python3", ["-c", pyScript], {
      timeout: 120000,
      maxBuffer: 10 * 1024 * 1024,
    }, (err, stdout, stderr) => {
      if (err) {
        return reject(new Error(`Python SDK error: ${stderr || err.message}`));
      }
      try {
        resolve(JSON.parse(stdout));
      } catch (parseErr) {
        reject(new Error(`Python SDK invalid JSON: ${stdout.slice(0, 500)}`));
      }
    });

    child.stdin.write(JSON.stringify(instance));
    child.stdin.end();
  });
}

/**
 * Main handler — dispatches to the appropriate mode.
 */
async function handleChatCompletions(openaiBody) {
  const mode = getMode();

  switch (mode) {
    case "dedicated-dns":
      return handleViaDedicatedDns(openaiBody);
    case "python-sdk":
      return handleViaPythonSdk(openaiBody);
    case "rawpredict":
    default:
      return handleViaRawPredict(openaiBody);
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
      const mode = getMode();
      console.log(`[vertex-proxy] Listening on ${baseUrl}`);
      console.log(`[vertex-proxy] Mode: ${mode}`);
      if (mode === "dedicated-dns") {
        console.log(`[vertex-proxy] DNS: ${VERTEX_DEDICATED_DNS}`);
      } else if (mode === "python-sdk") {
        console.log(`[vertex-proxy] Using Python google-cloud-aiplatform SDK (gRPC)`);
      } else {
        console.log(`[vertex-proxy] Endpoint: ${VERTEX_ENDPOINT_ID}`);
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
