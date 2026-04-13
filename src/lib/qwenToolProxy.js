/**
 * Qwen Tool Call Proxy
 *
 * Qwen 3.5 models output tool calls as XML:
 *   <tool_call>
 *   <function=name>
 *   <parameter=key>value</parameter>
 *   </function>
 *   </tool_call>
 *
 * OpenClaw expects OpenAI-compatible JSON tool_calls in the response.
 * This proxy sits between OpenClaw and Together's API, intercepting
 * responses to convert XML tool calls → JSON tool_calls format.
 *
 * Usage: starts a local HTTP server that proxies to Together's API.
 * Configure OpenClaw's together provider baseUrl to point here.
 */

import http from "node:http";
import https from "node:https";
import { URL } from "node:url";

const TOGETHER_BASE = "https://api.together.xyz";
const DEFAULT_PORT = 18790;

/**
 * Parse Qwen XML tool calls from text content.
 * Returns { cleanContent, toolCalls } where toolCalls is in OpenAI format.
 */
export function parseQwenXmlToolCalls(text) {
  if (!text || typeof text !== "string") return { cleanContent: text || "", toolCalls: [] };

  const toolCalls = [];
  // Match <tool_call>...<function=name>...<parameter=key>value</parameter>...</function></tool_call>
  // Also handle the variant without <tool_call> wrapper
  const toolCallRegex = /<tool_call>\s*([\s\S]*?)\s*<\/tool_call>/g;
  const functionRegex = /<function=([^>]+)>\s*([\s\S]*?)\s*<\/function>/g;
  const paramRegex = /<parameter=([^>]+)>\s*([\s\S]*?)\s*<\/parameter>/g;

  let match;
  let callId = 0;

  // Try <tool_call> wrapped version first
  const toolCallMatches = [...text.matchAll(toolCallRegex)];

  if (toolCallMatches.length > 0) {
    for (const tcMatch of toolCallMatches) {
      const inner = tcMatch[1];
      const fnMatches = [...inner.matchAll(functionRegex)];
      for (const fnMatch of fnMatches) {
        const fnName = fnMatch[1].trim();
        const fnBody = fnMatch[2];
        const params = {};
        const paramMatches = [...fnBody.matchAll(paramRegex)];
        for (const pm of paramMatches) {
          const key = pm[1].trim();
          let value = pm[2].trim();
          // Try to parse as JSON if it looks like JSON
          try { value = JSON.parse(value); } catch { /* keep as string */ }
          params[key] = value;
        }
        toolCalls.push({
          id: `call_qwen_${callId++}`,
          type: "function",
          function: {
            name: fnName,
            arguments: JSON.stringify(params),
          },
        });
      }
    }
  } else {
    // Try bare <function=...> without <tool_call> wrapper
    const bareFnMatches = [...text.matchAll(functionRegex)];
    for (const fnMatch of bareFnMatches) {
      const fnName = fnMatch[1].trim();
      const fnBody = fnMatch[2];
      const params = {};
      const paramMatches = [...fnBody.matchAll(paramRegex)];
      for (const pm of paramMatches) {
        const key = pm[1].trim();
        let value = pm[2].trim();
        try { value = JSON.parse(value); } catch { /* keep as string */ }
        params[key] = value;
      }
      toolCalls.push({
        id: `call_qwen_${callId++}`,
        type: "function",
        function: {
          name: fnName,
          arguments: JSON.stringify(params),
        },
      });
    }
  }

  // Remove tool call XML from content
  let cleanContent = text
    .replace(/<tool_call>\s*[\s\S]*?\s*<\/tool_call>/g, "")
    .replace(/<function=[^>]+>\s*[\s\S]*?\s*<\/function>/g, "")
    .trim();

  // Also strip <think>...</think> blocks — return as clean content
  // (OpenClaw doesn't need the reasoning trace)
  const thinkMatch = cleanContent.match(/<think>([\s\S]*?)<\/think>/);
  if (thinkMatch) {
    cleanContent = cleanContent.replace(/<think>[\s\S]*?<\/think>/g, "").trim();
  }

  return { cleanContent, toolCalls };
}

/**
 * Transform a Together API chat completion response to inject tool_calls.
 */
function transformResponse(body) {
  try {
    const data = JSON.parse(body);

    if (!data.choices || !Array.isArray(data.choices)) return body;

    let modified = false;
    for (const choice of data.choices) {
      const msg = choice.message || choice.delta;
      if (!msg || !msg.content) continue;

      const { cleanContent, toolCalls } = parseQwenXmlToolCalls(msg.content);

      if (toolCalls.length > 0) {
        msg.content = cleanContent || null;
        msg.tool_calls = toolCalls;
        choice.finish_reason = "tool_calls";
        modified = true;
        console.log(
          `[qwen-proxy] Converted ${toolCalls.length} XML tool call(s) → JSON: ${toolCalls.map((t) => t.function.name).join(", ")}`
        );
      }
    }

    return modified ? JSON.stringify(data) : body;
  } catch (err) {
    console.error("[qwen-proxy] Failed to transform response:", err.message);
    return body;
  }
}

/**
 * Proxy a request to Together's API, transforming the response.
 */
function proxyRequest(req, res) {
  const targetUrl = new URL(req.url, TOGETHER_BASE);

  const chunks = [];
  req.on("data", (chunk) => chunks.push(chunk));
  req.on("end", () => {
    const requestBody = Buffer.concat(chunks).toString("utf8");

    // Check if this is a streaming request — don't transform those (yet)
    let isStream = false;
    try {
      const parsed = JSON.parse(requestBody);
      isStream = parsed.stream === true;
    } catch { /* not JSON, pass through */ }

    const headers = { ...req.headers };
    delete headers.host;
    headers.host = targetUrl.host;

    const options = {
      hostname: targetUrl.hostname,
      port: 443,
      path: targetUrl.pathname + targetUrl.search,
      method: req.method,
      headers,
    };

    const proxyReq = https.request(options, (proxyRes) => {
      if (isStream) {
        // For streaming, pass through without transformation (for now)
        res.writeHead(proxyRes.statusCode, proxyRes.headers);
        proxyRes.pipe(res);
        return;
      }

      const respChunks = [];
      proxyRes.on("data", (chunk) => respChunks.push(chunk));
      proxyRes.on("end", () => {
        let body = Buffer.concat(respChunks).toString("utf8");

        // Only transform successful chat completion responses
        if (
          proxyRes.statusCode === 200 &&
          req.url.includes("/chat/completions")
        ) {
          body = transformResponse(body);
        }

        // Forward response with updated content-length
        const respHeaders = { ...proxyRes.headers };
        respHeaders["content-length"] = Buffer.byteLength(body);
        delete respHeaders["transfer-encoding"];

        res.writeHead(proxyRes.statusCode, respHeaders);
        res.end(body);
      });
    });

    proxyReq.on("error", (err) => {
      console.error("[qwen-proxy] Upstream error:", err.message);
      res.writeHead(502, { "content-type": "application/json" });
      res.end(JSON.stringify({ error: `Proxy error: ${err.message}` }));
    });

    if (requestBody) proxyReq.write(requestBody);
    proxyReq.end();
  });
}

/**
 * Start the proxy server.
 * @param {number} [port] - port to listen on (default 18790)
 * @returns {Promise<http.Server>}
 */
export function startQwenToolProxy(port = DEFAULT_PORT) {
  return new Promise((resolve, reject) => {
    const server = http.createServer(proxyRequest);

    server.on("error", (err) => {
      if (err.code === "EADDRINUSE") {
        console.log(`[qwen-proxy] Port ${port} already in use, proxy may already be running`);
        resolve(null);
      } else {
        reject(err);
      }
    });

    server.listen(port, "127.0.0.1", () => {
      console.log(`[qwen-proxy] XML→JSON tool call proxy running on http://127.0.0.1:${port}`);
      console.log(`[qwen-proxy] Proxying to ${TOGETHER_BASE}`);
      resolve(server);
    });
  });
}

export const PROXY_BASE_URL = `http://127.0.0.1:${DEFAULT_PORT}/v1`;
