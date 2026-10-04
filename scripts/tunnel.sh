#!/usr/bin/env bash
# Keeps the laptop API reachable from the deployed frontend: opens a localhost.run tunnel to port 8000,
# reconnects whenever it drops or hangs, and repoints the Vercel site each time the tunnel URL changes.
#
#   docker compose up -d            # the API, first
#   bash scripts/tunnel.sh          # Git Bash; leave this terminal open; Ctrl+C to stop
#
# Free localhost.run URLs change on every reconnect, so each one costs a ~1 min Vercel redeploy
# (scripts/point_frontend.sh). The site is unreachable from the moment a tunnel fails until that finishes.
set -uo pipefail
cd "$(dirname "$0")/.."

CHECK_EVERY=30     # seconds between health checks of the public URL
MAX_FAILURES=3     # consecutive failed checks before the tunnel counts as hung

# A tunnel can hang with ssh still connected (requests just time out), which ssh's keepalive does not
# notice. This polls the public URL and kills ssh when it stops answering, so the outer loop reconnects.
# localhost.run can hand out a new URL mid-session, so the URL is re-read from $URL_FILE on every check.
watchdog() {
    local ssh_pid="$1" failures=0 url
    sleep 20   # let the first repoint and routing settle
    while kill -0 "$ssh_pid" 2>/dev/null; do
        url="$(cat "$URL_FILE")"
        if curl -sf -m 15 -o /dev/null "$url/health"; then
            failures=0
        else
            failures=$((failures + 1))
            echo "$(date +%T) Health check $failures/$MAX_FAILURES failed for $url"
            if [ "$failures" -ge "$MAX_FAILURES" ]; then
                echo "$(date +%T) Tunnel is hung; restarting it"
                kill "$ssh_pid" 2>/dev/null
                return
            fi
        fi
        sleep "$CHECK_EVERY"
    done
}

URL_FILE="$(mktemp)"
LOG="$(mktemp)"
trap 'rm -f "$URL_FILE" "$LOG"' EXIT
current=""
while true; do
    echo "$(date +%T) Opening tunnel..."
    : > "$LOG"
    # A plain background job, so $! is ssh itself and the watchdog can kill it.
    ssh -T -o ServerAliveInterval=30 -o ServerAliveCountMax=3 -o ExitOnForwardFailure=yes \
        -o StrictHostKeyChecking=accept-new -R 80:127.0.0.1:8000 nokey@localhost.run > "$LOG" 2>&1 &
    ssh_pid=$!
    watching=""
    # tail exits when ssh does (--pid), which ends this read loop.
    while IFS= read -r line; do
        echo "$line"
        url="$(grep -oE 'https://[a-z0-9]+\.lhr\.life' <<<"$line" | head -1)"
        [ -z "$url" ] && continue
        printf '%s' "$url" > "$URL_FILE"
        if [ -z "$watching" ]; then
            watchdog "$ssh_pid" &
            watching=1
        fi
        if [ "$url" != "$current" ]; then
            current="$url"
            echo "$(date +%T) New tunnel URL: $url. Repointing the site..."
            # A fresh tunnel can take a few seconds to route; point_frontend.sh checks /ready first.
            ( for attempt in 1 2 3 4 5; do
                  scripts/point_frontend.sh "$url" && exit 0
                  sleep 5
              done
              echo "$(date +%T) Could not repoint the site to $url; run scripts/point_frontend.sh $url" >&2 ) &
        fi
    done < <(tail -n +1 -f --pid="$ssh_pid" "$LOG")
    kill "$ssh_pid" 2>/dev/null
    wait "$ssh_pid" 2>/dev/null
    echo "$(date +%T) Tunnel dropped; reconnecting in 5 s (Ctrl+C to stop)"
    sleep 5
done
