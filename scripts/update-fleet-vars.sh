#!/usr/bin/env bash
# =============================================================================
# Update env vars on existing Railway fleet projects
# =============================================================================
# Usage:
#   export RAILWAY_API_TOKEN="your-token"
#   bash scripts/update-fleet-vars.sh
# =============================================================================

set -uo pipefail

CONFIG_FILE="${1:-scripts/fleet-config.json}"
API="https://backboard.railway.com/graphql/v2"

command -v jq &>/dev/null || { echo "❌ jq not found"; exit 1; }
[ -f "$CONFIG_FILE" ] || { echo "❌ $CONFIG_FILE not found"; exit 1; }
[ -n "${RAILWAY_API_TOKEN:-}" ] || { echo "❌ Set RAILWAY_API_TOKEN"; exit 1; }

gql() {
  curl -s -X POST "$API" \
    -H "Content-Type: application/json" \
    -H "Authorization: Bearer $RAILWAY_API_TOKEN" \
    -d "{\"query\": $(echo "$1" | jq -Rs .), \"variables\": $2}"
}

SHARED_TOGETHER_KEY=$(jq -r '.shared.together_api_key // ""' "$CONFIG_FILE")
SHARED_SETUP_PASSWORD=$(jq -r '.shared.setup_password // ""' "$CONFIG_FILE")
AGENT_COUNT=$(jq '.agents | length' "$CONFIG_FILE")

# Get all projects to match by name
echo "🔍 Fetching your Railway projects..."
PROJECTS=$(gql '
  query { me { projects { edges { node { id name environments { edges { node { id name } } } services { edges { node { id name } } } } } } } }' '{}')

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

  echo ""
  echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
  echo "🤖 $AGENT_NAME — updating vars..."

  # Find project by name
  PROJECT_ID=$(echo "$PROJECTS" | jq -r ".data.me.projects.edges[] | select(.node.name == \"$AGENT_NAME\") | .node.id" 2>/dev/null)
  ENV_ID=$(echo "$PROJECTS" | jq -r ".data.me.projects.edges[] | select(.node.name == \"$AGENT_NAME\") | .node.environments.edges[0].node.id" 2>/dev/null)
  SERVICE_ID=$(echo "$PROJECTS" | jq -r ".data.me.projects.edges[] | select(.node.name == \"$AGENT_NAME\") | .node.services.edges[0].node.id" 2>/dev/null)

  if [ -z "$PROJECT_ID" ] || [ -z "$SERVICE_ID" ]; then
    echo "   ❌ Project '$AGENT_NAME' not found or has no service. Skipping."
    continue
  fi

  echo "   Project:  $PROJECT_ID"
  echo "   Service:  $SERVICE_ID"
  echo "   Env:      $ENV_ID"

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

  RESULT=$(gql '
    mutation($input: VariableCollectionUpsertInput!) {
      variableCollectionUpsert(input: $input)
    }' "{\"input\": {
      \"projectId\": \"$PROJECT_ID\",
      \"environmentId\": \"$ENV_ID\",
      \"serviceId\": \"$SERVICE_ID\",
      \"variables\": $VARS_JSON,
      \"replace\": true
    }}")

  if echo "$RESULT" | jq -e '.data.variableCollectionUpsert' &>/dev/null; then
    echo "   ✅ Variables updated! Redeploying..."

    # Trigger redeploy
    gql '
      mutation($serviceId: String!, $environmentId: String!) {
        serviceInstanceRedeploy(serviceId: $serviceId, environmentId: $environmentId)
      }' "{\"serviceId\": \"$SERVICE_ID\", \"environmentId\": \"$ENV_ID\"}" >/dev/null

    echo "   🚀 Redeploy triggered"
  else
    echo "   ❌ Failed:"
    echo "   $RESULT" | jq .
  fi
done

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "🎉 All agents updated and redeploying!"
