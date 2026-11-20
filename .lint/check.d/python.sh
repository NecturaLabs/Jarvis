# Managed by agentic-setup lint/python. Do not edit; there is no project override. Update with: lint.py sync.
# shellcheck shell=bash
# Sourced by .lint/check.sh from the repository root, after .lint/lib.sh. Never exits. Checks the
# tracked *.py files only, with the project's ruff.toml (which extends .lint/ruff.toml). Set
# LINT_RUFF and LINT_PYRIGHT when the tools are not on PATH.

if ruff=$(lint_tool ruff); then
  # --force-exclude keeps the project's exclude list in force for the files named here.
  lint_ruff() { lint_files '*.py' | xargs -0 -r "${ruff}" "$@" --no-cache --force-exclude; }
  lint_step "python: ruff check" lint_ruff check
  lint_step "python: ruff format" lint_ruff format --check
else
  lint_not_run "python: ruff check" ruff
  lint_not_run "python: ruff format" ruff
fi

if pyright=$(lint_tool pyright); then
  lint_step "python: pyright" "${pyright}" --project pyrightconfig.json
else
  lint_not_run "python: pyright" pyright
fi
