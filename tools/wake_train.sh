#!/bin/bash
# Train and score one LiveKit wake phrase on the training machine (a CUDA GPU or Apple silicon).
#
#   tools/wake_train.sh KEY_FILE "PHRASE" [--ungated]
#
# KEY_FILE holds only LIVEKIT_URL, LIVEKIT_API_KEY and LIVEKIT_API_SECRET (NAME=value lines; blank
# lines and # comments are fine) for a LiveKit Cloud project; its free credit pays for the generated
# speech. It must be mode 0600 or stricter, is read as data, never run, and its values reach only
# the trainer's setup and train. TRAINER is the folder of the trainer's virtual environment (default
# ./.venv, with livekit-wakeword[train]); WORK is where configs, speech, models and the log go
# (default ./work, created private). Relative paths are taken from where you run the script.
# --ungated runs through tools/wake_ungated.py until the gated test sets are approved; it marks WORK,
# and a full run refuses a WORK that holds the reduced test set: use a fresh WORK for it. The model
# is WORK/<phrase as the trainer names it>.onnx and the scores are WORK/eval-<name>.json, written
# only when the eval succeeds. Exit 3 when the credit ran out during setup: rerun with another
# project's key, the clips made so far are cached.
set -euo pipefail
umask 077
shopt -s extglob

usage() {
  sed -n '2,/^set -/{/^set -/!p}' "$0" >&2
  exit 2
}

fail() {
  local status=$1
  shift
  echo "wake_train.sh: $*" >&2
  exit "${status}"
}

if [[ $# -lt 2 || $# -gt 3 ]] || [[ -n ${3:-} && ${3} != --ungated ]]; then
  usage
fi
phrase=$2
ungated=0
if [[ ${3:-} == --ungated ]]; then
  ungated=1
fi

# Every path is made absolute here, before the script moves into WORK.
here=$(cd "$(dirname "$0")" && pwd)
key=$(realpath -m -- "$1")
trainer=$(realpath -m -- "${TRAINER:-.venv}")
work=$(realpath -m -- "${WORK:-work}")

# The trainer names the model after its slug of the phrase: lower case, every run of characters
# that are not letters, digits or "_" becomes "_", the ends are trimmed, "wakeword" if nothing is left.
name=${phrase,,}
name=${name//+([^[:alnum:]_])/_}
name=${name##+(_)}
name=${name%%+(_)}
name=${name:-wakeword}

# Read the key file as data: three assignments, nothing else.
[[ -f ${key} ]] || fail 2 "the key file ${key} does not exist"
mode=$(stat -L -c '%a' -- "${key}")
if ((8#${mode} & 8#177)); then
  fail 2 "the key file ${key} is mode ${mode}: run chmod 600 on it"
fi
livekit_url=
livekit_key=
livekit_secret=
while IFS= read -r line || [[ -n ${line} ]]; do
  line=${line%$'\r'}
  if [[ ${line} =~ ^[[:space:]]*(#.*)?$ ]]; then
    continue
  fi
  if [[ ! ${line} =~ ^(LIVEKIT_URL|LIVEKIT_API_KEY|LIVEKIT_API_SECRET)=(.*)$ ]]; then
    fail 2 "the key file ${key} may hold only LIVEKIT_URL, LIVEKIT_API_KEY and LIVEKIT_API_SECRET lines, comments and blank lines"
  fi
  variable=${BASH_REMATCH[1]}
  value=${BASH_REMATCH[2]}
  if [[ ${value} =~ ^\"(.*)\"$ || ${value} =~ ^\'(.*)\'$ ]]; then
    value=${BASH_REMATCH[1]}
  fi
  case ${variable} in
    LIVEKIT_URL) livekit_url=${value} ;;
    LIVEKIT_API_KEY) livekit_key=${value} ;;
    LIVEKIT_API_SECRET) livekit_secret=${value} ;;
    *) ;;
  esac
done <"${key}"
[[ -n ${livekit_url} ]] || fail 2 "the key file ${key} has no LIVEKIT_URL value"
[[ -n ${livekit_key} ]] || fail 2 "the key file ${key} has no LIVEKIT_API_KEY value"
[[ -n ${livekit_secret} ]] || fail 2 "the key file ${key} has no LIVEKIT_API_SECRET value"

if ((ungated)); then
  run=("${trainer}/bin/python" "${here}/wake_ungated.py")
else
  run=("${trainer}/bin/livekit-wakeword")
fi
[[ -x ${run[0]} ]] || fail 2 "the trainer is not installed: ${run[0]} is missing (set TRAINER to its virtual environment)"
if ((! ungated)) && [[ -e ${work}/.ungated ]]; then
  fail 2 "${work} holds the reduced test set of an --ungated run: use a fresh WORK for a full run"
fi

mkdir -p "${work}"
cd "${work}"
if ((ungated)); then
  : >.ungated
fi
log=${work}/train-${name}.log
tmpdir=$(mktemp -d "${work}/.run.XXXXXX")
trap 'rm -rf -- "${tmpdir}"' EXIT

step() { echo "=== $(date '+%F %T') $*" | tee -a "${log}"; }

# Run the trainer with (keys) or without (plain) the LiveKit credentials in its environment.
trainer_run() {
  local credentials=$1
  shift
  (
    if [[ ${credentials} == keys ]]; then
      export LIVEKIT_URL=${livekit_url} LIVEKIT_API_KEY=${livekit_key} LIVEKIT_API_SECRET=${livekit_secret}
    else
      unset LIVEKIT_URL LIVEKIT_API_KEY LIVEKIT_API_SECRET
    fi
    exec "${run[@]}" "$@"
  )
}

# trainer_logged OUT_FILE keys|plain ARGS...: show the output, append it to the log and to OUT_FILE
# (this run's output alone), and return the trainer's own status.
trainer_logged() {
  local out=$1 status=0
  shift
  set +e
  trainer_run "$@" 2>&1 | tee -a "${log}" "${out}"
  status=${PIPESTATUS[0]}
  set -e
  return "${status}"
}

step "setup ${phrase}"
setup_status=0
trainer_logged "${tmpdir}/setup.out" keys setup --wakeword "${phrase}" --plan build --eval || setup_status=$?
if grep -aqiF -e 'credit quota exhausted' -e MaxGatewayCredits "${tmpdir}/setup.out"; then
  step "the LiveKit credit ran out during setup: use another project's key; clips made so far are cached"
  exit 3
fi
if ((setup_status)); then
  step "setup failed (status ${setup_status})"
  fail 1 "setup failed (status ${setup_status}); see ${log}"
fi

step "train ${phrase}"
train_status=0
trainer_logged "${tmpdir}/train.out" keys train --wakeword "${phrase}" --type standard || train_status=$?
if ((train_status)); then
  step "train failed (status ${train_status})"
  fail 1 "train failed (status ${train_status}); see ${log}"
fi
[[ -f ${work}/${name}.onnx ]] || fail 1 "train finished but ${work}/${name}.onnx is not there; see ${log}"

step "eval ${phrase}"
if trainer_run plain eval --wakeword "${phrase}" --json >"${tmpdir}/eval.json" 2>>"${log}" && [[ -s ${tmpdir}/eval.json ]]; then
  mv -f -- "${tmpdir}/eval.json" "${work}/eval-${name}.json"
else
  tail -n 20 "${log}" >&2
  step "eval failed"
  fail 1 "eval failed; no scores were written; see ${log}"
fi
step "done: ${work}/${name}.onnx, ${work}/eval-${name}.json"
