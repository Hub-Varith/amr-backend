#!/usr/bin/env bash
# Point the deployed frontend at a new API URL, e.g. after restarting the laptop tunnel:
#
#   ssh -R 80:127.0.0.1:8000 nokey@localhost.run        # prints https://<random>.lhr.life
#   scripts/point_frontend.sh https://<random>.lhr.life
#
# Checks the API answers through that URL and allows the site's origin, sets VITE_API_BASE_URL on Vercel,
# and redeploys production (Vite inlines the URL at build time). About a minute.
set -euo pipefail

API="${1:-}"
SITE="https://dnagen-app.vercel.app"
API="${API%/}"
case "$API" in
    https://*) ;;
    *) echo "usage: $0 https://<api-host>   (must be https)" >&2; exit 2 ;;
esac

echo "Checking $API ..."
if ! curl -sf -m 20 "$API/ready" | grep -q '"ready":true'; then
    echo "API not ready at $API/ready: is Docker running and the tunnel open?" >&2
    exit 1
fi
if ! curl -s -m 20 -D - -o /dev/null -H "Origin: $SITE" "$API/health" | grep -qi "access-control-allow-origin: $SITE"; then
    echo "API does not allow $SITE (CORS). Check G2M_CORS_ORIGINS, and that nothing else is on port 8000." >&2
    exit 1
fi

cd "$(dirname "$0")/../frontend"
printf '%s' "$API" | npx vercel env update VITE_API_BASE_URL production --yes >/dev/null 2>&1 \
    || printf '%s' "$API" | npx vercel env add VITE_API_BASE_URL production >/dev/null
npx vercel deploy --prod --yes >/dev/null
echo "Done: $SITE now uses $API"
