/**
 * Vertex AI authentication helper.
 * Generates and auto-refreshes GCP access tokens.
 *
 * Supports two auth methods (checked in order):
 *
 * 1. OAuth Refresh Token (recommended when SA keys are blocked by org policy)
 *    Set VERTEX_REFRESH_TOKEN, VERTEX_CLIENT_ID, VERTEX_CLIENT_SECRET env vars.
 *    Obtained via: gcloud auth application-default login
 *
 * 2. Service Account Key (traditional)
 *    Set VERTEX_SA_KEY (base64-encoded JSON) or VERTEX_SA_KEY_FILE (path).
 *
 * The refreshed access token is written to VERTEX_API_TOKEN env var so the
 * local Vertex AI proxy (vertexProxy.js) can use it for rawPredict calls.
 */

import { createSign } from "node:crypto";
import fs from "node:fs";

const TOKEN_URL = "https://oauth2.googleapis.com/token";
const SCOPE = "https://www.googleapis.com/auth/cloud-platform";
const TOKEN_LIFETIME_SECS = 3600;
const REFRESH_MARGIN_MS = 5 * 60 * 1000; // refresh 5 min before expiry

let refreshTimer = null;

// ── Auth Method 1: OAuth Refresh Token ──────────────────────────────────────

function loadRefreshTokenConfig() {
  const refreshToken = process.env.VERTEX_REFRESH_TOKEN?.trim();
  const clientId = process.env.VERTEX_CLIENT_ID?.trim();
  const clientSecret = process.env.VERTEX_CLIENT_SECRET?.trim();

  if (refreshToken && clientId && clientSecret) {
    return { refreshToken, clientId, clientSecret };
  }
  return null;
}

async function fetchAccessTokenFromRefreshToken({ refreshToken, clientId, clientSecret }) {
  const res = await fetch(TOKEN_URL, {
    method: "POST",
    headers: { "Content-Type": "application/x-www-form-urlencoded" },
    body: new URLSearchParams({
      grant_type: "refresh_token",
      refresh_token: refreshToken,
      client_id: clientId,
      client_secret: clientSecret,
    }).toString(),
  });

  if (!res.ok) {
    const text = await res.text();
    throw new Error(`Refresh token exchange failed (${res.status}): ${text}`);
  }

  const data = await res.json();
  return data.access_token;
}

// ── Auth Method 2: Service Account Key ──────────────────────────────────────

function loadServiceAccountKey() {
  const b64 = process.env.VERTEX_SA_KEY?.trim();
  if (b64) {
    try {
      return JSON.parse(Buffer.from(b64, "base64").toString("utf8"));
    } catch {
      try { return JSON.parse(b64); } catch { /* fall through */ }
    }
  }
  const filePath = process.env.VERTEX_SA_KEY_FILE?.trim();
  if (filePath) {
    return JSON.parse(fs.readFileSync(filePath, "utf8"));
  }
  return null;
}

function createJwt(sa) {
  const now = Math.floor(Date.now() / 1000);
  const header = Buffer.from(JSON.stringify({ alg: "RS256", typ: "JWT" })).toString("base64url");
  const payload = Buffer.from(JSON.stringify({
    iss: sa.client_email,
    scope: SCOPE,
    aud: TOKEN_URL,
    iat: now,
    exp: now + TOKEN_LIFETIME_SECS,
  })).toString("base64url");

  const sign = createSign("RSA-SHA256");
  sign.update(`${header}.${payload}`);
  const signature = sign.sign(sa.private_key, "base64url");

  return `${header}.${payload}.${signature}`;
}

async function fetchAccessTokenFromSA(sa) {
  const jwt = createJwt(sa);
  const res = await fetch(TOKEN_URL, {
    method: "POST",
    headers: { "Content-Type": "application/x-www-form-urlencoded" },
    body: `grant_type=urn:ietf:params:oauth:grant-type:jwt-bearer&assertion=${jwt}`,
  });

  if (!res.ok) {
    const text = await res.text();
    throw new Error(`SA token fetch failed (${res.status}): ${text}`);
  }

  const data = await res.json();
  return data.access_token;
}

// ── Main: start / stop ──────────────────────────────────────────────────────

export async function startVertexAuthRefresh() {
  // Try refresh token first (works when SA keys are blocked by org policy)
  const rtConfig = loadRefreshTokenConfig();
  if (rtConfig) {
    console.log("[vertex-auth] Using OAuth refresh token for authentication");

    async function refresh() {
      try {
        const token = await fetchAccessTokenFromRefreshToken(rtConfig);
        process.env.VERTEX_API_TOKEN = token;
        console.log(`[vertex-auth] Access token refreshed via refresh_token (expires in ~${TOKEN_LIFETIME_SECS}s)`);
      } catch (err) {
        console.error(`[vertex-auth] Token refresh failed: ${err.message}`);
      }
    }

    await refresh();
    const intervalMs = (TOKEN_LIFETIME_SECS * 1000) - REFRESH_MARGIN_MS;
    refreshTimer = setInterval(refresh, intervalMs);
    console.log(`[vertex-auth] Auto-refresh scheduled every ${Math.round(intervalMs / 60000)} minutes`);
    return;
  }

  // Fall back to service account key
  const sa = loadServiceAccountKey();
  if (sa) {
    console.log("[vertex-auth] Using service account key for authentication");

    async function refresh() {
      try {
        const token = await fetchAccessTokenFromSA(sa);
        process.env.VERTEX_API_TOKEN = token;
        console.log(`[vertex-auth] Access token refreshed via SA key (expires in ~${TOKEN_LIFETIME_SECS}s)`);
      } catch (err) {
        console.error(`[vertex-auth] Token refresh failed: ${err.message}`);
      }
    }

    await refresh();
    const intervalMs = (TOKEN_LIFETIME_SECS * 1000) - REFRESH_MARGIN_MS;
    refreshTimer = setInterval(refresh, intervalMs);
    console.log(`[vertex-auth] Auto-refresh scheduled every ${Math.round(intervalMs / 60000)} minutes`);
    return;
  }

  console.warn("[vertex-auth] No auth configured. Set VERTEX_REFRESH_TOKEN+VERTEX_CLIENT_ID+VERTEX_CLIENT_SECRET, or VERTEX_SA_KEY. Falling back to VERTEX_API_TOKEN env var directly.");
}

export function stopVertexAuthRefresh() {
  if (refreshTimer) {
    clearInterval(refreshTimer);
    refreshTimer = null;
  }
}
