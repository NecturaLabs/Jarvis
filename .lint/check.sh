#!/usr/bin/env bash
# Managed by agentic-setup lint/common. Do not edit; update with: lint.py sync.
# The shared lint runner. Run from anywhere in a project: .lint/check.sh
# Runs .lint/check.d/<bundle>.sh for each bundle in the record .lint/bundles, in record order, with the
# bundle's package directories in LINT_PACKAGES (default: .), and prints one line per step.
# Exit 0: every step ran and passed. 1: a step failed (under CI=true a missing tool is a failure).
# 3: every step that ran passed, but a tool was missing so a step did not run.
# A tool is found in <pkg>/node_modules/.bin, then PATH, then $LINT_<TOOL> (LINT_RUFF=/path/to/ruff).
# Nothing is downloaded. Locally, when agentic-setup is present ($AGENTIC_SETUP, default
# /data/dev/Projects/agentic-setup) the managed files are also compared with it, as a warning.
set -u

lint_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
cd "${lint_dir}/.." || exit 1
record=.lint/bundles
if [[ ! -f ${record} ]]; then
  printf 'check.sh: %s not found; install the bundles with lint.py install\n' "${record}" >&2
  exit 1
fi

# shellcheck source=lib.sh
. .lint/lib.sh

while IFS=$'\t' read -r -a fields; do
  [[ ${fields[0]:-} == bundle ]] || continue
  bundle=${fields[1]}
  script=.lint/check.d/${bundle}.sh
  if [[ ! -f ${script} ]]; then
    printf 'skip %s: no %s\n' "${bundle}" "${script}"
    continue
  fi
  # shellcheck disable=SC2034 # read by the sourced check.d script
  LINT_PACKAGES=("${fields[@]:2}")
  [[ ${#LINT_PACKAGES[@]} -gt 0 ]] || LINT_PACKAGES=(.)
  # shellcheck source=/dev/null
  . "${script}"
done <"${record}"

agentic=${AGENTIC_SETUP:-/data/dev/Projects/agentic-setup}
if [[ ${CI:-} != true ]] && [[ -f "${agentic}/lint/lint.py" ]]; then
  if ! drift=$(python3 "${agentic}/lint/lint.py" check --target . 2>&1); then
    printf '\nWARN lint configs behind agentic-setup: run lint.py sync\n%s\n' "${drift}"
  fi
fi

printf '\n=== lint summary\n'
printf '%s\n' "${LINT_RESULTS[@]}"
printf '%d passed, %d failed, %d not run\n' "${LINT_PASS}" "${LINT_FAIL}" "${LINT_SKIP}"
if [[ ${LINT_FAIL} -gt 0 ]]; then exit 1; fi
if [[ ${LINT_SKIP} -gt 0 ]]; then exit 3; fi
exit 0
