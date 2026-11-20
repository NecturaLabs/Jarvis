#!/usr/bin/env bash
# Managed by agentic-setup lint/common. Do not edit; update with: lint.py sync.
# Helpers for the .lint/check.d/<bundle>.sh scripts. Sourced by .lint/check.sh from the repository
# root, never run on its own. A check.d script only calls these functions and never exits.

LINT_PASS=0
LINT_FAIL=0
LINT_SKIP=0
LINT_RESULTS=()

# lint_tool <name> [<pkg dir>]: prints the tool's absolute path and returns 0, or returns 1 and prints
# nothing. Looks in <pkg>/node_modules/.bin, then PATH, then $LINT_<NAME> (upper-case, - as _).
# Never downloads anything.
lint_tool() {
  local name=$1 pkg=${2:-.} path var
  if [[ -x "${pkg}/node_modules/.bin/${name}" ]]; then
    (cd "${pkg}" && printf '%s/node_modules/.bin/%s\n' "${PWD}" "${name}")
    return 0
  fi
  if path=$(type -P -- "${name}"); then
    printf '%s\n' "${path}"
    return 0
  fi
  var=LINT_$(printf '%s' "${name}" | tr 'a-z-' 'A-Z_')
  if [[ -n ${!var:-} ]] && [[ -x ${!var} ]]; then
    printf '%s\n' "${!var}"
    return 0
  fi
  return 1
}

# lint_step <label> <command...>: runs the command with its output shown and records PASS or FAIL.
lint_step() {
  local label=$1
  shift
  printf '\n--- %s\n' "${label}"
  if "$@"; then
    LINT_PASS=$((LINT_PASS + 1))
    LINT_RESULTS+=("PASS     ${label}")
  else
    LINT_FAIL=$((LINT_FAIL + 1))
    LINT_RESULTS+=("FAIL     ${label}")
  fi
}

# lint_not_run <label> <tool>: records that a step could not run because its tool is missing. Under
# CI=true that is a failure.
lint_not_run() {
  local label=$1 tool=$2
  printf '\n--- %s\n%s missing\n' "${label}" "${tool}"
  if [[ ${CI:-} == true ]]; then
    LINT_FAIL=$((LINT_FAIL + 1))
    LINT_RESULTS+=("FAIL     ${label} (${tool} missing; CI requires it)")
  else
    LINT_SKIP=$((LINT_SKIP + 1))
    LINT_RESULTS+=("NOT RUN  ${label} (${tool} missing)")
  fi
}

# lint_files <pathspec...>: the tracked files matching the git pathspecs, NUL-separated. Files that
# are deleted in the working tree, and the backups of lint.py (.lint-backup/), are left out.
lint_files() {
  local f
  git ls-files -z -- "$@" ':(exclude).lint-backup' | while IFS= read -r -d '' f; do
    [[ -f ${f} ]] && printf '%s\0' "${f}"
  done
  return 0
}

# lint_shell_files: tracked *.sh and *.bash files, plus tracked files without an extension that start
# with a sh or bash shebang, NUL-separated.
# shortcut: shebangs are read only for names without a dot (hooks and tools), to keep large repositories fast.
lint_shell_files() {
  local f line
  lint_files '*.sh' '*.bash'
  lint_files ':(exclude)*.sh' ':(exclude)*.bash' | while IFS= read -r -d '' f; do
    [[ ${f##*/} == *.* ]] && continue
    IFS= read -r line <"${f}" || [[ -n ${line} ]] || continue
    if [[ ${line} =~ ^#![[:space:]]*(/usr/bin/env[[:space:]]+(-S[[:space:]]+)?)?([^[:space:]]*/)?(ba)?sh([[:space:]]|$) ]]; then
      printf '%s\0' "${f}"
    fi
  done
  return 0
}
