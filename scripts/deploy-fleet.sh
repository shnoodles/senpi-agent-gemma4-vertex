#!/usr/bin/env bash
# =============================================================================
# Senpi Agent Fleet Deployer (Railway GraphQL API)
# =============================================================================
# Fully automated — no interactive prompts.
#
# Prerequisites:
#   1. Railway API token: https://railway.com/account/tokens
#   2. jq installed: brew install jq
#   3. fleet-config.json filled in
#
# Usage:
#   export RAILWAY_API_TOKEN="your-token-here"
#   bash scripts/deploy-fleet.sh
# =============================================================================

set -uo pipefail

CONFIG_FILE="${1:-scripts/fleet-config.json}"
REPO="shnoodles/senpi-hyperclaw-railway-template"
BRANCH="main"
API="https://backboard.railway.app/graphql/v2"

# ── Preflight ──
command -v jq &>/dev/null || { echo "❌ jq not found. brew install jq"; exit 1; }
command -v curl &>/dev/null || { echo "❌ curl not found"; exit 1; }
[ -f "$CONFIG_FILE" ] || { echo "❌ $CONFIG_FILE not found"; exit 1; }
[ -n "${RAILWAY_API_TOKEN:-}" ] || { echo "❌ Set RAILWAY_API_TOKEN first. Get one at https://railway.com/account/tokens"; exit 1; }

# ── Helper: call Railway GraphQL API ──
gql() {
  local query="$1"
  local variables="$2"
  curl -s -X POST "$API" \
    -H "Content-Type: application/json" \
    -H "Authorization: Bearer $RAILWAY_API_TOKEN" \
    -d "{\"query\": $(echo "$query" | jq -Rs .), \"variables\": $variables}"
}

AGENT_COUNT=$(jq '.agents | length' "$CONFIG_FILE")
SHARED_SETUP_PASSWORD=$(jq -r '.shared.setup_password // ""' "$CONFIG_FILE")
SHARED_TOGETHER_KEY=$(jq -r '.shared.together_api_key // ""' "$CONFIG_FILE")

echo "🚀 Deploying $AGENT_COUNT agents via Railway API"
echo ""

for i in $(seq 0 $(($AGENT_COUNT - 1))); do
  AGENT_NAME=$(jq -r ".agents[$i].name" "$CONFIG_FILE")
  AI_PROVIDER=$(jq -r ".agents[$i].ai_provider" "$CONFIG_FILE")
  AI_API_KEY=$(jq -r ".agents[$i].ai_api_key // \"\"" "$CONFIG_FILE")
  MODEL=$(jq -r ".agents[$i].model // \"\"" "$CONFIG_FILE")
  SENPI_AUTH_TOKEN=$(jq -r ".agents[$i].senpi_auth_token" "$CONFIG_FILE")
  TELEGRAM_BOT_TOKEN=$(jq -r ".agents[$i].telegram_bot_token" "$CONFIG_FILE")
  TELEGRAM_USERID=$(jq -r ".agents[$i].telegram_userid // \"\"" "$CONFIG_FILE")

  [ -z "$AI_API_KEY" ] && [ "$AI_PROVIDER" = "together" ] && AI_API_KEY="$SHARED_TOGETHER_KEY"
  SETUP_PASSWORD="${SHARED_SETUP_PASSWORD:-$(openssl rand -hex 16)}"

  echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
  echo "🤖 Agent $((i+1))/$AGENT_COUNT: $AGENT_NAME"
  echo "   Model: $MODEL"
  echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

  # ── Step 1: Create project ──
  echo "   📁 Creating project..."
  PROJECT_RESULT=$(gql '
    mutation($input: ProjectCreateInput!) {
      projectCreate(input: $input) {
        id
        environments { edges { node { id name } } }
      }
    }' "{\"input\": {\"name\": \"$AGENT_NAME\"}}")

  PROJECT_ID=$(echo "$PROJECT_RESULT" | jq -r '.data.projectCreate.id // empty')
  ENV_ID=$(echo "$PROJECT_RESULT" | jq -r '.data.projectCreate.environments.edges[0].node.id // empty')

  if [ -z "$PROJECT_ID" ] || [ -z "$ENV_ID" ]; then
    echo "   ❌ Failed to create project:"
    echo "   $PROJECT_RESULT" | jq .
    continue
  fi
  echo "   ✅ Project: $PROJECT_ID"
  echo "   ✅ Environment: $ENV_ID"

  # ── Step 2: Create service with GitHub repo + variables ──
  echo "   📦 Creating service with repo..."

  VARS_JSON=$(jq -n \
    --arg ai_provider "$AI_PROVIDER" \
    --arg ai_api_key "$AI_API_KEY" \
    --arg ai_model "$MODEL" \
    --arg senpi_token "$SENPI_AUTH_TOKEN" \
    --arg tg_bot "$TELEGRAM_BOT_TOKEN" \
    --arg tg_user "$TELEGRAM_USERID" \
    --arg setup_pw "$SETUP_PASSWORD" \
    '{
      AI_PROVIDER: $ai_provider,
      AI_API_KEY: $ai_api_key,
      AI_MODEL: $ai_model,
      SENPI_AUTH_TOKEN: $senpi_token,
      TELEGRAM_BOT_TOKEN: $tg_bot,
      TELEGRAM_USERID: $tg_user,
      SETUP_PASSWORD: $setup_pw,
      OPENCLAW_STATE_DIR: "/data/.openclaw",
      OPENCLAW_WORKSPACE_DIR: "/data/workspace",
      SENPI_STATE_DIR: "/data/.openclaw/senpi-state"
    }')

  SERVICE_RESULT=$(gql '
    mutation($name: String, $projectId: String!, $environmentId: String!, $source: ServiceSourceInput, $branch: String, $variables: EnvironmentVariables) {
      serviceCreate(input: {
        name: $name,
        projectId: $projectId,
        environmentId: $environmentId,
        source: $source,
        variables: $variables,
        branch: $branch
      }) { id name }
    }' "{
      \"name\": \"$AGENT_NAME\",
      \"projectId\": \"$PROJECT_ID\",
      \"environmentId\": \"$ENV_ID\",
      \"source\": {\"repo\": \"$REPO\"},
      \"branch\": \"$BRANCH\",
      \"variables\": $VARS_JSON
    }")

  SERVICE_ID=$(echo "$SERVICE_RESULT" | jq -r '.data.serviceCreate.id // empty')

  if [ -z "$SERVICE_ID" ]; then
    echo "   ❌ Failed to create service:"
    echo "   $SERVICE_RESULT" | jq .
    continue
  fi
  echo "   ✅ Service: $SERVICE_ID"

  # ── Step 3: Add volume ──
  echo "   💾 Adding volume..."
  VOL_RESULT=$(gql '
    mutation($input: VolumeCreateInput!) {
      volumeCreate(input: $input) { id }
    }' "{\"input\": {
      \"projectId\": \"$PROJECT_ID\",
      \"environmentId\": \"$ENV_ID\",
      \"serviceId\": \"$SERVICE_ID\",
      \"mountPath\": \"/data\"
    }}")

  VOL_ID=$(echo "$VOL_RESULT" | jq -r '.data.volumeCreate.id // empty')
  if [ -n "$VOL_ID" ]; then
    echo "   ✅ Volume: $VOL_ID"
  else
    echo "   ⚠️  Volume might need manual setup"
  fi

  # ── Step 4: Generate domain ──
  echo "   🌐 Generating domain..."
  DOMAIN_RESULT=$(gql '
    mutation($input: ServiceDomainCreateInput!) {
      serviceDomainCreate(input: $input) { domain }
    }' "{\"input\": {
      \"environmentId\": \"$ENV_ID\",
      \"serviceId\": \"$SERVICE_ID\"
    }}")

  DOMAIN=$(echo "$DOMAIN_RESULT" | jq -r '.data.serviceDomainCreate.domain // empty')
  if [ -n "$DOMAIN" ]; then
    echo "   ✅ Domain: https://$DOMAIN"
  else
    echo "   ⚠️  Domain: enable public networking in dashboard"
  fi

  echo ""
  echo "   🎉 $AGENT_NAME deployed!"
  echo "   🔗 https://railway.com/project/$PROJECT_ID"
  echo "   📝 Password: $SETUP_PASSWORD"
  echo "   🤖 Model: $MODEL"
  echo ""

done

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "🎉 All $AGENT_COUNT agents deployed!"
echo ""
echo "Next steps:"
echo "  1. Send /start to each Telegram bot"
echo "  2. Wait ~3-5 min for agents to boot"
echo "  3. Each starts with its assigned model automatically"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
