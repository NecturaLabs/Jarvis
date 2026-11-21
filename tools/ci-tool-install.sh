#!/usr/bin/env bash
# shellcheck shell=bash
# Pinned CI tool installs: a pin per tool (version, URL and sha256 written in the workflow, bumped
# together), a download kept in a cache and verified by its sha256 on every use, and Python packages
# installed only from a requirements file whose every line carries hashes. Nothing is installed from a
# floating version, and nothing unverified is ever used.
#
# Sourced by a workflow step, never run on its own (see .github/workflows/ci.yml). Linux runners: it needs
# sha256sum, curl and python3. Adapted from agentic-setup's project-setup reference.

# The cache survives between runs on a self-hosted runner (RUNNER_TOOL_CACHE), so a run needs the network
# only for a new pin; elsewhere it is the run's temporary folder.
CI_TOOL_CACHE=${CI_TOOL_CACHE:-${RUNNER_TOOL_CACHE:-${RUNNER_TEMP:-${TMPDIR:-/tmp}}}/ci-tools-cache}

# ci_fetch <url> <sha256> <file name>: leaves the verified download at $CI_TOOL_CACHE/<file name>. A cached
# file is used only if its hash still matches; a download is kept only if its hash matches, and is
# written beside the final name first so an interrupted one is never taken for a good one.
ci_fetch() {
  local url=$1 sha=$2 file=${CI_TOOL_CACHE}/$3
  mkdir -p "${CI_TOOL_CACHE}" || return 1
  if [[ -f ${file} ]] && echo "${sha}  ${file}" | sha256sum -c --quiet >/dev/null 2>&1; then
    return 0
  fi
  curl --proto '=https' --proto-redir '=https' --tlsv1.2 -fsSL --retry 5 --retry-all-errors "${url}" -o "${file}.part" || {
    rm -f "${file}.part"
    return 1
  }
  echo "${sha}  ${file}.part" | sha256sum -c --quiet || {
    rm -f "${file}.part"
    return 1
  }
  mv "${file}.part" "${file}"
}

# ci_pip_install <venv dir> <requirements file>: a fresh virtual environment with exactly the pinned,
# hashed wheels. --require-hashes makes pip refuse any line without a hash, and any dependency not
# listed; --only-binary :all: refuses a source build, which would run unverified code.
ci_pip_install() {
  python3 -m venv "$1" || return 1
  "$1/bin/pip" install -q --disable-pip-version-check --cache-dir "${CI_TOOL_CACHE}/pip" --retries 5 \
    --only-binary :all: --require-hashes -r "$2"
}
