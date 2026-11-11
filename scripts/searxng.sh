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

# A reused container keeps its old image, and `podman run` never refreshes a local
# :latest, so pull on each start and recreate the container when the image changed.
refresh() {
  if ! podman pull -q "$IMAGE" >/dev/null 2>&1; then
    echo "warning: could not pull $IMAGE (offline?); using the local image" >&2
    return
  fi
  if podman container exists "$NAME"; then
    local have want
    have=$(podman container inspect --format '{{.Image}}' "$NAME")
    want=$(podman image inspect --format '{{.Id}}' "$IMAGE")
    if [[ "$have" != "$want" ]]; then
      echo "new $IMAGE: recreating $NAME (settings in $CONF are kept)"
      podman rm -f "$NAME" >/dev/null
    fi
  fi
}

image_date() {
  podman image inspect --format '{{.Created}}' "$IMAGE" 2>/dev/null | cut -c1-16
}

case "${1:-}" in
  start)
    setup
    refresh
    if podman container exists "$NAME"; then
      podman start "$NAME" >/dev/null
    else
      podman run -d --name "$NAME" -p "127.0.0.1:$PORT:8080" \
        -v "$CONF:/etc/searxng:Z" "$IMAGE" >/dev/null
    fi
    echo "SearXNG on http://127.0.0.1:$PORT, image built $(image_date) (council doctor checks it)"
    ;;
  stop) podman stop "$NAME" >/dev/null && echo "stopped" ;;
  status)
    podman ps -a --filter "name=^$NAME$" --format '{{.Names}} {{.Status}} {{.Ports}}'
    echo "image $IMAGE built $(image_date)"
    ;;
  logs) podman logs --tail 50 "$NAME" ;;
  *) echo "usage: $0 start|stop|status|logs" >&2; exit 2 ;;
esac
