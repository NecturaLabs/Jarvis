#!/usr/bin/env bash

usage() {
  cat <<'HELP'
Usage: install.sh [options]
With curl: curl -fsSL https://raw.githubusercontent.com/NecturaLabs/Jarvis/main/install.sh | bash -s -- [options]

  --install-dir PATH               JARVIS_INSTALL_DIR; checkout location
  --branch NAME                    JARVIS_BRANCH; release branch (main by default)
  --run-config PATH                JARVIS_RUN_CONFIG; JSON written by strata launch
  --coding-run-config PATH         JARVIS_CODING_RUN_CONFIG; optional second model
  --strata-home PATH               JARVIS_STRATA_HOME; defaults to STRATA_DATA_DIR if set
  --strata-components-parent PATH  JARVIS_STRATA_COMPONENTS_PARENT; Strata's XDG data home
  --strata-vram-reserve-mib N        JARVIS_STRATA_VRAM_RESERVE_MIB; VRAM to leave free
  --data-dir PATH                  JARVIS_DATA_DIR; Jarvis wake, VAD and voice models
  --mic NAME                       JARVIS_MIC; optional PipeWire input
  --spotify-client-id ID           JARVIS_SPOTIFY_CLIENT_ID; optional Spotify app
  --enable                         JARVIS_ENABLE=1; enable and start both user services
  --non-interactive                JARVIS_NON_INTERACTIVE=1; never prompt
  --dry-run                        JARVIS_DRY_RUN=1; print commands without running them
  --help                           print this help
HELP
}

die() {
  printf 'jarvis install: %s\n' "$*" >&2
  exit 1
}

need() {
  if ! command -v "$1" >/dev/null; then
    printf 'Missing %s. %s\n' "$1" "$2" >&2
    missing=1
  fi
}

run() {
  if [[ ${dry} == 1 ]]; then
    local word
    printf '+'
    for word in "$@"; do
      printf ' %q' "${word}"
    done
    printf '\n'
  else
    "$@"
  fi
}

absolute() {
  case $1 in
    /*) printf '%s' "$1" ;;
    *) printf '%s/%s' "${original_dir}" "$1" ;;
  esac
}

choose_run_config() {
  local -a found=("${strata_data}"/install/strata-*.json)
  if (("${#found[@]}" == 1)) && [[ -r ${found[0]} ]]; then
    run_config=${found[0]}
    return
  fi
  if [[ ${dry} == 1 ]]; then
    run_config='/path/to/strata-model.json'
    return
  fi
  if [[ ${interactive} == 1 ]] && { exec 3<>/dev/tty; } 2>/dev/null; then
    printf 'Path to the run config written by strata launch: ' >&3
    IFS= read -r run_config <&3 || run_config=
    exec 3>&-
  fi
  [[ -n ${run_config} ]] || die 'run strata launch, then strata launch --stop, and rerun with --run-config PATH (or JARVIS_RUN_CONFIG).'
}

main() {
  set -euo pipefail
  shopt -s nullglob dotglob
  local state=${XDG_STATE_HOME:-"${HOME}/.local/state"} data=${XDG_DATA_HOME:-"${HOME}/.local/share"}
  [[ ${state} == /* ]] || state="${HOME}/.local/state"
  [[ ${data} == /* ]] || data="${HOME}/.local/share"
  local original_dir=${PWD} missing=0
  local repo=https://github.com/NecturaLabs/Jarvis.git
  local checkout=${JARVIS_INSTALL_DIR:-"${state}/jarvis/checkout"} branch=${JARVIS_BRANCH:-main}
  local run_config=${JARVIS_RUN_CONFIG:-} coding=${JARVIS_CODING_RUN_CONFIG:-}
  local strata_home=${JARVIS_STRATA_HOME:-${STRATA_DATA_DIR:-}}
  local components=${JARVIS_STRATA_COMPONENTS_PARENT:-} reserve=${JARVIS_STRATA_VRAM_RESERVE_MIB:-}
  local models=${JARVIS_DATA_DIR:-} mic=${JARVIS_MIC:-} spotify=${JARVIS_SPOTIFY_CLIENT_ID:-}
  local enable=${JARVIS_ENABLE:-0} interactive=1 dry=${JARVIS_DRY_RUN:-0}
  local chrome_found='from pathlib import Path; from jarvis.web import browser; raise SystemExit(not browser.chrome_found(Path.home()))'
  [[ ${JARVIS_NON_INTERACTIVE:-0} != 1 ]] || interactive=0
  while (($#)); do
    case $1 in
      --help | -h)
        usage
        return
        ;;
      --enable) enable=1 ;;
      --non-interactive) interactive=0 ;;
      --dry-run) dry=1 ;;
      --install-dir | --branch | --run-config | --coding-run-config | --strata-home | --strata-components-parent | --strata-vram-reserve-mib | --data-dir | --mic | --spotify-client-id)
        (($# >= 2)) && [[ -n $2 && $2 != --* ]] || die "$1 needs a value."
        case $1 in
          --install-dir) checkout=$2 ;;
          --branch) branch=$2 ;;
          --run-config) run_config=$2 ;;
          --coding-run-config) coding=$2 ;;
          --strata-home) strata_home=$2 ;;
          --strata-components-parent) components=$2 ;;
          --strata-vram-reserve-mib) reserve=$2 ;;
          --data-dir) models=$2 ;;
          --mic) mic=$2 ;;
          --spotify-client-id) spotify=$2 ;;
          *) die "unknown option $1." ;;
        esac
        shift
        ;;
      *) die "unknown option $1 (see --help)." ;;
    esac
    shift
  done

  local strata_data=${strata_home:-"${data}/strata"}
  strata_data=$(absolute "${strata_data}")
  checkout=$(absolute "${checkout}")
  [[ ${branch} != -* && -n ${branch} ]] || die 'invalid branch name.'
  if ! command -v uv >/dev/null && [[ -x ${HOME}/.local/bin/uv ]]; then
    export PATH="${HOME}/.local/bin:${PATH}"
  fi
  need omarchy-version 'Install Omarchy from https://omarchy.org/, then run this in its desktop session.'
  need git 'Run: sudo pacman -S git'
  need uv 'Run: sudo pacman -S uv (or curl -LsSf https://astral.sh/uv/install.sh | sh)'
  need python3.14 'Run: sudo pacman -Syu python'
  need systemctl 'Run: sudo pacman -S systemd'
  need voxtype 'Run: sudo pacman -S voxtype'
  if [[ ! -x ${strata_data}/terminal/current/bin/strata ]]; then
    need strata 'Install the Strata terminal from https://github.com/niko1221/strata, then run: strata launch'
  fi
  need pw-record 'Run: sudo pacman -S pipewire'
  need pw-play 'Run: sudo pacman -S pipewire'
  need pw-dump 'Run: sudo pacman -S pipewire'
  need wpctl 'Run: sudo pacman -S wireplumber'
  need agent-browser 'Run: yay -S agent-browser (AUR), or install it as https://github.com/vercel-labs/agent-browser says'
  ((missing == 0)) || [[ ${dry} == 1 ]] || exit 1
  if [[ ${dry} != 1 ]]; then
    git check-ref-format "refs/heads/${branch}" || die 'invalid branch name.'
    systemctl --user show-environment >/dev/null || die 'run this from your Omarchy desktop session so systemd can reach your user manager.'
  else
    run git check-ref-format "refs/heads/${branch}"
    run systemctl --user show-environment
  fi

  local existing=0 jarvis="${checkout}/.venv/bin/jarvis" has_config=0
  if [[ -e ${checkout}/.git ]]; then
    existing=1
    if [[ ${dry} != 1 ]]; then
      [[ $(git -C "${checkout}" rev-parse --show-toplevel) == "$(cd -P -- "${checkout}" && pwd)" ]] || die 'the install folder is not the root of a Git checkout.'
      [[ $(git -C "${checkout}" remote get-url origin) == "${repo}" ]] || die 'the install folder does not track the public Jarvis repository on origin.'
      [[ $(git -C "${checkout}" symbolic-ref --quiet --short HEAD) == "${branch}" ]] || die "the install folder is not on ${branch}. Use --branch or choose a different folder."
      local git_status
      git_status=$(git -C "${checkout}" status --porcelain --untracked-files=all) || die 'could not read the checkout status.'
      [[ -z ${git_status} ]] || die 'the Jarvis checkout has local changes. Commit or move them before rerunning.'
      if [[ -x ${jarvis} ]]; then
        if "${jarvis}" setup --has-run-config; then
          has_config=1
        elif (($? != 1)); then
          die 'could not read the existing Jarvis config.'
        fi
      fi
    else
      run git -C "${checkout}" rev-parse --show-toplevel
      run git -C "${checkout}" remote get-url origin
      run git -C "${checkout}" symbolic-ref --quiet --short HEAD
      run git -C "${checkout}" status --porcelain --untracked-files=all
      if [[ -x ${jarvis} ]]; then
        run "${jarvis}" setup --has-run-config
        has_config=1
      fi
    fi
  elif [[ -e ${checkout} || -L ${checkout} ]]; then
    local -a contents=("${checkout}"/*)
    [[ -d ${checkout} && ! -L ${checkout} && ${#contents[@]} == 0 ]] || die "the install folder is not empty and has no Jarvis checkout: ${checkout}"
  fi

  if [[ -z ${run_config} && ${has_config} == 0 ]]; then
    choose_run_config
  fi
  local -a setup=(setup --update-branch "${branch}")
  if [[ -n ${run_config} ]]; then
    run_config=$(absolute "${run_config}")
    [[ ${dry} == 1 || -r ${run_config} ]] || die "cannot read the Strata run config: ${run_config}"
    setup+=(--run-config "${run_config}")
  fi
  [[ -z ${coding} ]] || setup+=(--coding-run-config "$(absolute "${coding}")")
  [[ -z ${strata_home} ]] || setup+=(--strata-home "${strata_data}")
  [[ -z ${components} ]] || setup+=(--strata-components-parent "$(absolute "${components}")")
  [[ -z ${reserve} ]] || setup+=(--strata-vram-reserve-mib "${reserve}")
  [[ -z ${mic} ]] || setup+=(--mic "${mic}")
  [[ -z ${spotify} ]] || setup+=(--spotify-client-id "${spotify}")
  local config_home=${XDG_CONFIG_HOME:-"${HOME}/.config"}
  [[ ${config_home} == /* ]] || config_home="${HOME}/.config"
  if [[ -z ${models} && ! -e ${config_home}/jarvis/config.toml ]]; then
    models="${state}/jarvis/models"
  fi
  [[ -z ${models} ]] || setup+=(--data-dir "$(absolute "${models}")")

  if ((existing == 0)); then
    run mkdir -p -- "${checkout%/*}"
    run git clone --branch="${branch}" --single-branch -- "${repo}" "${checkout}"
  fi
  if [[ ${dry} == 1 ]]; then
    run cd -- "${checkout}"
  else
    cd -- "${checkout}"
  fi
  run export UV_PROJECT_ENVIRONMENT="${checkout}/.venv"
  if ((has_config)); then
    run "${jarvis}" update --apply
  else
    local python
    python=$(command -v python3.14) || python=python3.14
    run uv sync --locked --python "${python}"
  fi
  run "${jarvis}" "${setup[@]}"
  run uv run --no-sync tools/get_models.py
  # The headless browser drives a Chrome of its own, downloaded once into the home folder (no sudo).
  if [[ ${dry} == 1 ]] || ! uv run --no-sync python -c "${chrome_found}"; then
    run agent-browser install
  fi
  run "${jarvis}" doctor --no-service --essential
  if [[ ${enable} == 1 ]]; then
    run "${jarvis}" setup --enable
    run "${jarvis}" doctor --wait --essential
  fi
  printf '\nJarvis checkout: %s\n' "${checkout}"
  if [[ ${enable} == 1 ]]; then
    printf 'Say “Hey Jarvis” once it is listening. Check with %q status.\n' "${jarvis}"
  else
    printf 'When ready, run %q setup --enable, then %q doctor.\n' "${jarvis}" "${jarvis}"
  fi
  printf 'Optional: run %q spotify-login after setting a client ID, or %q update --auto on.\n' "${jarvis}" "${jarvis}"
}

(main "$@")
