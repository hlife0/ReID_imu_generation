#!/usr/bin/env bash
# Apply / revert / check the destroy_pairing_in_batch patch against the main
# project (Rd-id-Project). Never commits to the main repo; leaves its working
# tree pristine after `revert`.
set -euo pipefail

MAIN="${MAIN:-/home/hrli/ReID-workspace/Rd-id-Project}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PATCH="$HERE/destroy_pairing_in_batch.patch"

usage() { echo "usage: $0 {apply|revert|check}"; exit 2; }
[ $# -eq 1 ] || usage

case "$1" in
  apply)  git -C "$MAIN" apply "$PATCH"        && echo "applied to $MAIN" ;;
  revert) git -C "$MAIN" apply -R "$PATCH"     && echo "reverted; $MAIN pristine" ;;
  check)  git -C "$MAIN" apply --check "$PATCH" && echo "applies cleanly on current $MAIN" ;;
  *) usage ;;
esac
