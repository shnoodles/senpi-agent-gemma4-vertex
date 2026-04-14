/**
 * Entry point: composes config, auth, gateway, onboarding, setup routes, and proxy.
 */

import fs from "node:fs";
import path from "node:path";

import express from "express";
import {
  PORT,
  configPath,
  isConfigured,
  SETUP_PASSWORD,
  AI_PROVIDER,
  AI_API_KEY,
  PROVIDER_TO_AUTH_CHOICE,
  resolveEffectiveApiKey,
  TELEGRAM_BOT_TOKEN,
  TELEGRAM_USERNAME,
} from "./lib/config.js";
import { resolveGatewayToken } from "./lib/auth.js";
import { getGatewayProcess, restartGateway } from "./gateway.js";
import { stopAutoApprovalLoop } from "./lib/deviceAuth.js";
import {
  autoOnboard,
  canAutoOnboard,
  shouldReOnboardDueToEnvChange,
  isOnboardingInProgress,
  AUTO_ONBOARD_FINGERPRINT_FILE,
} from "./onboard.js";
import { bootstrapOpenClaw } from "./bootstrap.mjs";
import { resolveTelegramUserId, readCachedTelegramId, writeCachedTelegramId, readChatIdFromUserMd, discoverTelegramUserFromUpdates } from "./lib/telegramId.js";
import { createSetupRouter } from "./routes/setup.js";
import {
  controlUiMiddleware,
  controlUiHandler,
  catchAllMiddleware,
  attachUpgrade,
} from "./routes/proxy.js";

if (!SETUP_PASSWORD) {
  console.error("================================================================");
  console.error("WARNING: SETUP_PASSWORD is not configured.");
  console.error("  /setup and gateway routes (/, /openclaw) will be disabled.");
  console.error("  Set SETUP_PASSWORD in Railway Variables to enable the setup");
  console.error("  wizard and Control UI access.");
  console.error("================================================================");
}

const OPENCLAW_GATEWAY_TOKEN = resolveGatewayToken();
process.env.OPENCLAW_GATEWAY_TOKEN = OPENCLAW_GATEWAY_TOKEN;

// Propagate effective API key to provider-specific env vars
{
  const PROVIDER_ENV_MAP = {
    together: "TOGETHER_API_KEY",
    deepseek: "DEEPSEEK_API_KEY",
    moonshot: "MOONSHOT_API_KEY",
    gemini: "GEMINI_API_KEY",
    venice: "VENICE_API_KEY",
    zai: "ZAI_API_KEY",
  };
  const envVar = PROVIDER_ENV_MAP[AI_PROVIDER];
  if (envVar && !process.env[envVar]) {
    const effectiveKey = resolveEffectiveApiKey();
    if (effectiveKey) {
      process.env[envVar] = effectiveKey;
      console.log(`[wrapper] Propagated effective key to ${envVar}`);
    }
  }
}

const app = express();
app.disable("x-powered-by");
app.use(express.json({ limit: "1mb" }));

// Setup wizard and API
app.use("/setup", createSetupRouter());

// Control UI (/, /openclaw) — intercept HTML and inject token script
app.get(
  ["/", "/openclaw", "/openclaw/"],
  controlUiMiddleware,
  controlUiHandler
);

// Debug endpoints: proxy request logs and MCP config
import { getRequestLog } from "./lib/vertexProxy.js";
import { STATE_DIR } from "./lib/config.js";
app.get("/debug/proxy-logs", (_req, res) => {
  res.json({ logs: getRequestLog(), count: getRequestLog().length });
});
app.get("/debug/mcp-config", (_req, res) => {
  try {
    const mcporterPath = path.join(STATE_DIR, "config", "mcporter.json");
    const ocConfigPath = path.join(STATE_DIR, "openclaw.json");
    const configDir = path.join(STATE_DIR, "config");
    const mcporter = fs.existsSync(mcporterPath) ? JSON.parse(fs.readFileSync(mcporterPath, "utf8")) : "FILE NOT FOUND";
    const ocConfig = fs.existsSync(ocConfigPath) ? JSON.parse(fs.readFileSync(ocConfigPath, "utf8")) : "FILE NOT FOUND";
    // Check directory existence
    const stateExists = fs.existsSync(STATE_DIR);
    const configDirExists = fs.existsSync(configDir);
    const configDirContents = configDirExists ? fs.readdirSync(configDir) : "DIR NOT FOUND";
    const stateDirContents = stateExists ? fs.readdirSync(STATE_DIR) : "DIR NOT FOUND";
    // Mask tokens
    const maskToken = (t) => t ? `${t.slice(0, 6)}...${t.slice(-4)} (${t.length} chars)` : "(empty)";
    if (mcporter?.mcpServers?.senpi?.env?.SENPI_AUTH_TOKEN) {
      mcporter.mcpServers.senpi.env.SENPI_AUTH_TOKEN = maskToken(mcporter.mcpServers.senpi.env.SENPI_AUTH_TOKEN);
    }
    res.json({
      mcporterPath,
      mcporter,
      configDirContents,
      stateDirContents,
      ocConfigModels: ocConfig?.models,
      ocConfigMcp: ocConfig?.mcp || "not set in main config",
      envSenpiToken: process.env.SENPI_AUTH_TOKEN ? maskToken(process.env.SENPI_AUTH_TOKEN) : "(not set)",
    });
  } catch(e) {
    res.json({ error: e.message, stack: e.stack });
  }
});
// POST to inject Senpi MCP into openclaw.json and restart gateway
app.post("/debug/fix-mcp", async (_req, res) => {
  try {
    const ocPath = path.join(STATE_DIR, "openclaw.json");
    const cfg = JSON.parse(fs.readFileSync(ocPath, "utf8"));
    const senpiToken = process.env.SENPI_AUTH_TOKEN || "";
    const mcpUrl = process.env.SENPI_MCP_URL || "https://mcp.dev.senpi.ai/mcp";
    cfg.mcp = cfg.mcp || {};
    cfg.mcp.servers = cfg.mcp.servers || {};
    cfg.mcp.servers.senpi = {
      command: "npx",
      args: ["mcp-remote", mcpUrl, "--header", "Authorization: Bearer ${SENPI_AUTH_TOKEN}"],
      env: { SENPI_AUTH_TOKEN: senpiToken },
    };
    fs.writeFileSync(ocPath, JSON.stringify(cfg, null, 2));
    // Restart gateway to pick up new MCP config
    const { restartGateway: rg } = await import("./gateway.js");
    const gatewayToken = process.env.OPENCLAW_GATEWAY_TOKEN || "";
    await rg(gatewayToken);
    res.json({ ok: true, wrote: ocPath, mcpServers: Object.keys(cfg.mcp.servers), gatewayRestarted: true });
  } catch(e) {
    res.json({ error: e.message });
  }
});

// Everything else → proxy to gateway (with auth and onboarding redirect)
app.use(catchAllMiddleware);

const server = app.listen(PORT, () => {
  console.log(`[wrapper] listening on port ${PORT}`);
  console.log(`[wrapper] setup wizard: http://localhost:${PORT}/setup`);
  console.log(`[wrapper] configured: ${isConfigured()}`);
  const effectiveKey = resolveEffectiveApiKey();
  console.log(`[wrapper] AI_PROVIDER: ${AI_PROVIDER ? `"${AI_PROVIDER}"` : "(not set)"}`);
  console.log(`[wrapper] AI_API_KEY: ${AI_API_KEY ? `set (${AI_API_KEY.length} chars)` : "(not set)"}`);
  if (!AI_API_KEY && effectiveKey) {
    console.log(`[wrapper] Effective key: resolved from provider env var (${effectiveKey.length} chars)`);
  }
  console.log(`[wrapper] canAutoOnboard: ${canAutoOnboard()}`);

  if (isConfigured() && shouldReOnboardDueToEnvChange()) {
    console.log(
      "[wrapper] Env vars changed since last auto-onboard — re-onboarding with current Variables..."
    );
    try {
      fs.unlinkSync(configPath());
      try {
        fs.unlinkSync(AUTO_ONBOARD_FINGERPRINT_FILE);
      } catch {
        /* ignore */
      }
    } catch (e) {
      console.error(`[wrapper] Failed to remove old config: ${e.message}`);
    }
    autoOnboard(OPENCLAW_GATEWAY_TOKEN).catch((err) => {
      console.error(`[wrapper] Re-onboard failed: ${err}`);
    });
  } else if (canAutoOnboard()) {
    console.log("[wrapper] Auto-onboarding from environment variables...");
    autoOnboard(OPENCLAW_GATEWAY_TOKEN).catch((err) => {
      console.error(`[wrapper] Auto-onboard failed: ${err}`);
    });
  } else if (isConfigured()) {
    console.log(
      "[wrapper] Already configured, syncing configs and starting gateway..."
    );
    (async () => {
      // Resolve Telegram user ID via API BEFORE bootstrap, so allowlist config is correct
      if (TELEGRAM_BOT_TOKEN) {
        if (TELEGRAM_USERNAME) {
          console.log(`[wrapper] Resolving Telegram user ID: ${TELEGRAM_USERNAME}`);
          await resolveTelegramUserId(TELEGRAM_BOT_TOKEN, TELEGRAM_USERNAME).catch((err) => {
            console.warn(`[telegram] Pre-bootstrap ID resolution failed (non-fatal): ${err.message}`);
          });
        }
        // Fallback chain when cache file is still empty:
        // 1. Read chat ID from USER.md (written during initial onboarding)
        // 2. Discover from recent bot updates via Telegram API
        if (!readCachedTelegramId()) {
          const userMdId = readChatIdFromUserMd();
          if (userMdId) {
            console.log(`[wrapper] Recovered Telegram chat ID from USER.md: ${userMdId}`);
            writeCachedTelegramId(userMdId);
          } else {
            console.log("[wrapper] No cached Telegram ID — trying to discover from recent bot updates...");
            await discoverTelegramUserFromUpdates(TELEGRAM_BOT_TOKEN).catch((err) => {
              console.warn(`[telegram] Bot update discovery failed (non-fatal): ${err.message}`);
            });
          }
        }
      }
      try {
        bootstrapOpenClaw();
      } catch (err) {
        console.error(`[wrapper] Bootstrap sync error (non-fatal): ${err}`);
      }
      await restartGateway(OPENCLAW_GATEWAY_TOKEN);
    })().catch((err) => {
      console.error(`[wrapper] Gateway startup failed: ${err}`);
    });
  } else {
    // Not configured and can't auto-onboard — explain why.
    console.log("[wrapper] ================================================================");
    console.log("[wrapper] Not configured and auto-onboard is not possible.");
    if (!AI_PROVIDER) {
      console.log("[wrapper]   ✗ AI_PROVIDER is not set");
    } else if (!PROVIDER_TO_AUTH_CHOICE[AI_PROVIDER]) {
      console.log(`[wrapper]   ✗ AI_PROVIDER="${AI_PROVIDER}" is not a recognized provider`);
      console.log(`[wrapper]     Supported: ${Object.keys(PROVIDER_TO_AUTH_CHOICE).join(", ")}`);
    } else {
      console.log(`[wrapper]   ✓ AI_PROVIDER="${AI_PROVIDER}"`);
    }
    if (!AI_API_KEY && !effectiveKey) {
      console.log("[wrapper]   ✗ AI_API_KEY is not set (no provider-specific key found either)");
    } else if (!AI_API_KEY && effectiveKey) {
      console.log(`[wrapper]   ~ AI_API_KEY is not set, but provider key found (${effectiveKey.length} chars)`);
    } else {
      console.log("[wrapper]   ✓ AI_API_KEY is set");
    }
    if (SETUP_PASSWORD) {
      console.log("[wrapper] → Visit /setup to configure manually");
    } else {
      console.log("[wrapper]   ✗ SETUP_PASSWORD is not set (/setup is disabled)");
      console.log("[wrapper] → Set AI_PROVIDER + AI_API_KEY, or set SETUP_PASSWORD to use the setup wizard");
    }
    console.log("[wrapper] ================================================================");
  }

});

attachUpgrade(server);

process.on("SIGTERM", () => {
  stopAutoApprovalLoop();
  try {
    const proc = getGatewayProcess();
    if (proc) proc.kill("SIGTERM");
  } catch {
    // ignore
  }
  process.exit(0);
});
