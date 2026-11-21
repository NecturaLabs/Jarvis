#!/usr/bin/env bash
# Everything CI runs (.github/workflows/ci.yml), in one command, before calling work done. Audio is cut for every
# step. The advisories check needs the network; offline it is reported as not run and the script exits 3, so a
# pass is never claimed without it.
set -euo pipefail

cd "$(git rev-parse --show-toplevel)"
export PIPEWIRE_REMOTE=/nonexistent PULSE_SERVER=unix:/nonexistent

failed=()
step() {
  local name=$1
  shift
  printf '\n=== %s\n' "${name}"
  if "$@"; then
    printf 'PASS     %s\n' "${name}"
  else
    printf 'FAIL     %s\n' "${name}"
    failed+=("${name}")
  fi
}

audit() {
  local requirements
  requirements=$(mktemp)
  local status=0
  uv export --frozen --all-groups --no-emit-project --no-hashes -q >"${requirements}" \
    && uv run pip-audit --strict --disable-pip --no-deps -r "${requirements}" || status=$?
  rm -f "${requirements}"
  return "${status}"
}

step "locked environment" uv sync --locked
step "lint and types" uv run .lint/check.sh
step "tests" uv run pytest -q -m "not local"
step "licence information (REUSE)" uv run reuse lint
step "dependency licences and sources" uv run python tools/check_licences.py
step "docs" uv run python tools/check_docs.py

offline=0
if getent hosts pypi.org >/dev/null 2>&1; then
  step "known vulnerabilities" audit
else
  printf '\nNOT RUN  known vulnerabilities (pypi.org is unreachable)\n'
  offline=1
fi

if ((${#failed[@]})); then
  printf '\n%d failed: %s\n' "${#failed[@]}" "${failed[*]}"
  exit 1
fi
if ((offline)); then
  exit 3
fi
printf '\nall checks passed\n'
