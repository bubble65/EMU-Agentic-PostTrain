#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TARGET="${1:-all}"

SFT_TAG="${SFT_TAG:-local/univr-sft:cu128-py312}"
RL_TAG="${RL_TAG:-local/univr-rl:cu128-py312}"
DOCKER_BUILD_NETWORK="${DOCKER_BUILD_NETWORK:-host}"
DOCKER_BUILDKIT="${DOCKER_BUILDKIT:-1}"

docker_cmd=(docker)
if [[ "${USE_SUDO:-0}" == "1" ]]; then
  docker_cmd=(sudo -E docker)
fi

extra_args=()
if [[ "${NO_CACHE:-0}" == "1" ]]; then
  extra_args+=(--no-cache)
fi
if [[ -n "${PLATFORM:-}" ]]; then
  extra_args+=(--platform "${PLATFORM}")
fi

usage() {
  cat <<EOF
Usage: $(basename "$0") [all|sft|rl]

Environment:
  SFT_TAG               Image tag for SFT. Default: ${SFT_TAG}
  RL_TAG                Image tag for RL. Default: ${RL_TAG}
  USE_SUDO=1            Run docker via sudo -E.
  NO_CACHE=1            Build without cache.
  PLATFORM=linux/amd64  Optional docker build platform.
  DOCKER_BUILD_NETWORK  Docker build network. Default: host.
  DOCKER_BUILDKIT       Docker BuildKit switch. Default: 1.
EOF
}

build_image() {
  local name="$1"
  local dockerfile="$2"
  local tag="$3"

  echo "[build] ${name}: ${tag}"
  DOCKER_BUILDKIT="${DOCKER_BUILDKIT}" "${docker_cmd[@]}" build \
    --network "${DOCKER_BUILD_NETWORK}" \
    -f "${SCRIPT_DIR}/${dockerfile}" \
    -t "${tag}" \
    "${extra_args[@]}" \
    "${SCRIPT_DIR}"
}

case "${TARGET}" in
  all)
    build_image "sft" "Dockerfile.sft.public" "${SFT_TAG}"
    build_image "rl" "Dockerfile.rl.public" "${RL_TAG}"
    ;;
  sft)
    build_image "sft" "Dockerfile.sft.public" "${SFT_TAG}"
    ;;
  rl)
    build_image "rl" "Dockerfile.rl.public" "${RL_TAG}"
    ;;
  -h|--help|help)
    usage
    ;;
  *)
    usage >&2
    exit 2
    ;;
esac
