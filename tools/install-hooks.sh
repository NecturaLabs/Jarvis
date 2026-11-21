#!/usr/bin/env bash
# Usage: tools/install-hooks.sh [--uninstall]
#
# Opt-in git hooks: points this repository's core.hooksPath at tools/hooks, so the pre-commit (checks
# and tests) and commit-msg (Conventional Commits) hooks there run before every commit in every
# worktree of this repository. --uninstall removes the setting. It changes this repository's own
# config only, and refuses to replace a core.hooksPath that is already set to something else.
set -euo pipefail

cd "$(dirname "$0")/.."

wanted=tools/hooks
current=$(git config --local --get core.hooksPath || true)
effective=$(git config --show-scope --get core.hooksPath || true)

case "${1:-}" in
  --uninstall)
    if [[ -z ${current} ]]; then
      echo "hooks not installed: core.hooksPath is not set in this repository"
    elif [[ ${current} == "${wanted}" ]]; then
      git config --local --unset core.hooksPath
      echo "hooks uninstalled: core.hooksPath unset"
    else
      echo "not uninstalling: core.hooksPath is ${current}, not ${wanted}" >&2
      exit 1
    fi
    ;;
  "")
    if [[ ${current} == "${wanted}" ]]; then
      echo "hooks already installed: core.hooksPath = ${wanted}"
    elif [[ -n ${effective} ]]; then
      echo "not installing: core.hooksPath is already set (${effective}); unset it first" >&2
      exit 1
    else
      chmod +x tools/hooks/pre-commit tools/hooks/commit-msg
      git config --local core.hooksPath "${wanted}"
      echo "hooks installed: core.hooksPath = ${wanted} (undo with tools/install-hooks.sh --uninstall)"
    fi
    ;;
  *)
    echo "usage: tools/install-hooks.sh [--uninstall]" >&2
    exit 2
    ;;
esac
