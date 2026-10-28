#!/usr/bin/env bash
# Run SearXNG in Podman for council's local web search ([research.local] in config.toml).
# Usage: scripts/searxng.sh start|stop|status|logs
# Binds to 127.0.0.1 only; settings live in data/searxng/ (gitignored).
set -euo pipefail

NAME=council-searxng
IMAGE=${SEARXNG_IMAGE:-docker.io/searxng/searxng:latest}
PORT=${SEARXNG_PORT:-8888}
ROOT=$(cd "$(dirname "$0")/.." && pwd)
CONF="$ROOT/data/searxng"

command -v podman >/dev/null || { echo "podman not found: install it first" >&2; exit 1; }

setup() {
  mkdir -p "$CONF"
  if [[ ! -f "$CONF/settings.yml" ]]; then
    key=$(head -c 32 /dev/urandom | od -An -tx1 | tr -d ' \n')
    sed "s/@SECRET_KEY@/$key/" "$ROOT/scripts/searxng/settings.yml.template" > "$CONF/settings.yml"
    echo "wrote $CONF/settings.yml"
  fi
}

case "${1:-}" in
  start)
    setup
    if podman container exists "$NAME"; then
      podman start "$NAME" >/dev/null
    else
      podman run -d --name "$NAME" -p "127.0.0.1:$PORT:8080" \
        -v "$CONF:/etc/searxng:Z" "$IMAGE" >/dev/null
    fi
    echo "SearXNG on http://127.0.0.1:$PORT (council doctor checks it)"
    ;;
  stop) podman stop "$NAME" >/dev/null && echo "stopped" ;;
  status) podman ps -a --filter "name=^$NAME$" --format '{{.Names}} {{.Status}} {{.Ports}}' ;;
  logs) podman logs --tail 50 "$NAME" ;;
  *) echo "usage: $0 start|stop|status|logs" >&2; exit 2 ;;
esac
