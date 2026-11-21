#!/usr/bin/env bash
# Usage: tools/install-hooks.sh [--uninstall]
#
# Opt-in git hooks: copies tools/hooks/pre-commit (checks and tests) and tools/hooks/commit-msg
# (Conventional Commits) into this repository's own hooks folder (`git rev-parse --git-path hooks`),
# shared by its worktrees. It copies rather than pointing core.hooksPath at tools/hooks: with a path
# into the tracked tree, any hook a branch adds there (post-checkout, post-merge) would run the moment
# that branch is checked out, before anyone has read it. A copy changes only when this script is run
# again. --uninstall removes the copies it made and leaves every other hook alone.
set -euo pipefail

cd "$(dirname "$0")/.."

marker="# jarvis-hook: installed by tools/install-hooks.sh"
hooks=(pre-commit commit-msg)
legacy=tools/hooks # what an earlier version of this script set core.hooksPath to
if [[ ${1:-} == --uninstall && $(git config --local --get core.hooksPath || true) == "${legacy}" ]]; then
  git config --local --unset core.hooksPath
  echo "removed the old setting core.hooksPath = ${legacy}"
fi
folder=$(git rev-parse --git-path hooks)
effective=$(git config --show-scope --get core.hooksPath || true)

ours() { [[ -f $1 && ! -L $1 ]] && grep -qxF -- "${marker}" "$1"; }

case "${1:-}" in
  --uninstall)
    if [[ -n ${effective} ]]; then
      echo "not uninstalling: core.hooksPath is set (${effective}); its hooks are not this script's to remove" >&2
      exit 1
    fi
    removed=0
    for hook in "${hooks[@]}"; do
      if ours "${folder}/${hook}"; then
        rm -f -- "${folder}/${hook}"
        removed=$((removed + 1))
      fi
    done
    if ((removed)); then
      echo "hooks uninstalled from ${folder}"
    else
      echo "hooks not installed: no hook of ours in ${folder}"
    fi
    ;;
  "")
    if [[ -n ${effective} ]]; then
      echo "not installing: core.hooksPath is set (${effective}), so git would not run hooks from ${folder}; unset it first (tools/install-hooks.sh --uninstall undoes the old tools/hooks setting)" >&2
      exit 1
    fi
    for hook in "${hooks[@]}"; do
      if [[ -e ${folder}/${hook} || -L ${folder}/${hook} ]] && ! ours "${folder}/${hook}"; then
        echo "not installing: ${folder}/${hook} exists and was not installed by this script; move it first" >&2
        exit 1
      fi
    done
    mkdir -p -- "${folder}"
    changed=0
    for hook in "${hooks[@]}"; do
      if ! cmp -s -- "tools/hooks/${hook}" "${folder}/${hook}"; then
        install -m 0755 -- "tools/hooks/${hook}" "${folder}/${hook}"
        changed=$((changed + 1))
      fi
    done
    if ((changed)); then
      echo "hooks installed: ${hooks[*]} copied into ${folder} (run again after they change; undo with tools/install-hooks.sh --uninstall)"
    else
      echo "hooks already installed and up to date in ${folder}"
    fi
    ;;
  *)
    echo "usage: tools/install-hooks.sh [--uninstall]" >&2
    exit 2
    ;;
esac
