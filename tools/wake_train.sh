#!/bin/bash
# Train and score one LiveKit wake phrase on the training machine (a CUDA GPU or Apple silicon).
#
#   tools/wake_train.sh KEY_FILE "PHRASE" [--ungated]
#
# KEY_FILE holds LIVEKIT_URL, LIVEKIT_API_KEY and LIVEKIT_API_SECRET for a LiveKit Cloud project
# (mode 0600); its free credit pays for the generated speech. TRAINER is the folder of the trainer's
# virtual environment (default ./.venv, with livekit-wakeword[train]); WORK is where configs,
# speech and models go (default ./work). --ungated runs through tools/wake_ungated.py until the
# gated test sets are approved. The model is WORK/<phrase_with_underscores>.onnx and the scores
# are WORK/eval-<name>.json. It stops before training when the credit ran out during setup.
set -euo pipefail

if [[ $# -lt 2 ]]; then
  sed -n '2,11p' "$0" >&2
  exit 2
fi
key=$1
phrase=$2
here=$(cd "$(dirname "$0")" && pwd)
trainer=${TRAINER:-${PWD}/.venv}
work=${WORK:-${PWD}/work}
name=${phrase// /_}
log=${work}/train-${name}.log
if [[ ${3:-} == --ungated ]]; then
  run=("${trainer}/bin/python" "${here}/wake_ungated.py")
else
  run=("${trainer}/bin/livekit-wakeword")
fi

mkdir -p "${work}"
cd "${work}"
set -a
# shellcheck source=/dev/null
. "${key}"
set +a
step() { echo "=== $(date '+%F %T') $*"; }

step "setup ${phrase}" | tee -a "${log}"
"${run[@]}" setup --wakeword "${phrase}" --plan build --eval 2>&1 | tee -a "${log}"
if tr '\r' '\n' <"${log}" | grep -q "credit quota"; then
  step "the LiveKit credit ran out during setup: use another project's key; clips made so far are cached" | tee -a "${log}"
  exit 3
fi
step "train ${phrase}" | tee -a "${log}"
"${run[@]}" train --wakeword "${phrase}" --type standard 2>&1 | tee -a "${log}"
step "eval ${phrase}" | tee -a "${log}"
"${run[@]}" eval --wakeword "${phrase}" --json >"${work}/eval-${name}.json"
step "done: ${work}/${name}.onnx, ${work}/eval-${name}.json" | tee -a "${log}"
