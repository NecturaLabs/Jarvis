# Managed by agentic-setup lint/common. Do not edit; update with: lint.py sync.
# common: shellcheck and shfmt over the tracked shell files, editorconfig-checker over every tracked
# file, and two opt-in CI checks: a test registry (.lint/test-registry) and a self-hosted runner guard
# (.lint/ci-self-hosted). Sourced by .lint/check.sh; uses the helpers of .lint/lib.sh.
# shellcheck shell=bash

mapfile -d '' -t shell_files < <(lint_shell_files)

if [[ ${#shell_files[@]} -gt 0 ]]; then
  if tool=$(lint_tool shellcheck); then
    lint_step "common: shellcheck" "${tool}" "${shell_files[@]}"
  else
    lint_not_run "common: shellcheck" shellcheck
  fi
  # shfmt prints a diff and fails when a file differs. The flags repeat the [*.{sh,bash}] section of
  # .editorconfig, because a script without an extension (a git hook) falls under [*] there; given
  # flags, shfmt reads no .editorconfig.
  if tool=$(lint_tool shfmt); then
    lint_step "common: shfmt" "${tool}" --indent 2 --binary-next-line --case-indent --simplify -d "${shell_files[@]}"
  else
    lint_not_run "common: shfmt" shfmt
  fi
fi

# With no file arguments editorconfig-checker checks every file git tracks, and honours
# .editorconfig-checker.json for a project's excludes. Indent size is left to the formatters (shfmt,
# ruff, Biome, clang-format), which own it: the checker reads a docstring's or a comment's
# continuation as a wrong indent. Tabs against spaces is still checked.
if tool=$(lint_tool editorconfig-checker); then
  lint_step "common: editorconfig" "${tool}" -disable-indent-size
else
  lint_not_run "common: editorconfig" editorconfig-checker
fi

# Opt-in: with a registry file, every tracked script test must be listed in it, so a new test is
# never skipped silently. The file holds one repository-relative path per line; blank lines and
# '#' comments (a whole line, or after a space) are ignored. Only tracked files are seen, so
# `git add` a new test before running this.
lint_test_registry() {
  local registry=.lint/test-registry listed file status=0
  listed=$(sed -E -e 's/(^|[[:space:]])#.*//' -e 's/^[[:space:]]+//' -e 's/[[:space:]]+$//' "${registry}")
  # A here-string, not a pipe: under pipefail, grep -q leaving at its first match would fail a listed file.
  while IFS= read -r -d '' file; do
    grep -qxF -- "${file}" <<<"${listed}" || {
      printf '%s is not listed in %s\n' "${file}" "${registry}"
      status=1
    }
  done < <(lint_files '*_test.py' '*_test.sh' '*_unit.py')
  return "${status}"
}
if [[ -f .lint/test-registry ]]; then
  lint_step "common: test registry" lint_test_registry
fi

# Opt-in: with the flag file, every job's runs-on in .github/workflows must require self-hosted.
if [[ -f .lint/ci-self-hosted ]]; then
  if tool=$(lint_tool python3); then
    lint_step "common: ci runner guard" "${tool}" -I .lint/ci_runner_guard.py
  else
    lint_not_run "common: ci runner guard" python3
  fi
fi
