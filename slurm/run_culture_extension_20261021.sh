#!/usr/bin/env bash
#SBATCH --job-name=culture-extension
#SBATCH --output=slurm/culture-extension.%x.%A_%a.out
#SBATCH --error=slurm/culture-extension.%x.%A_%a.err
set -euo pipefail

REPO_ROOT="${SLURM_SUBMIT_DIR:?Submit from the immutable snapshot}"
cd "${REPO_ROOT}"
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HOME="${HF_HOME:?Set HF_HOME to the model cache}"
export HSA_XNACK=1
export VIPER_PYTHON_OVERLAY="${VIPER_PYTHON_OVERLAY:?Set the verified Python overlay}"
IMAGE="${VIPER_TRAIN_CONTAINER:?Set the verified ROCm container image}"
STAGE="${1:?Expected smoke, baseline or train}"
INDEX="${SLURM_ARRAY_TASK_ID:-0}"

# Before any GPU work, ensure the transferred code and inputs match the snapshot.
if [[ -n "${CULTURE_SNAPSHOT_MANIFEST:-}" ]]; then
  python3 scripts/verify_culture_extension_snapshot.py --snapshot-manifest "${CULTURE_SNAPSHOT_MANIFEST}"
fi
mapfile -t RUN_INFO < <(python3 - "${STAGE}" "${INDEX}" <<'PY'
import json,sys
from pathlib import Path
stage,index=sys.argv[1],int(sys.argv[2])
runs=json.loads(Path('data_cache/culture_extensions_20261021/runs/run_index.json').read_text())['runs']
if stage in ('smoke','baseline'):
    dataset=['cao','anllo'][index]
    run=next(r for r in runs if r['dataset']==dataset and r['seed']==0 and r['fold']==0 and r['condition']=='us')
    output=Path('outputs/culture_extensions_20261021')/('baseline_precision_aligned' if stage=='baseline' else stage)/dataset
elif stage=='train':
    run=runs[index]
    manifest=Path(run['manifest'])
    output=(manifest.parent/json.loads(manifest.read_text())['output_dir']).resolve()
else:
    raise ValueError(stage)
print(run['manifest'])
print(output)
print(24 if run['dataset']=='cao' else 12)
PY
)
[[ ${#RUN_INFO[@]} -eq 3 ]] || { echo 'Failed to select the run' >&2; exit 2; }
MANIFEST="${RUN_INFO[0]}"
OUTPUT="${RUN_INFO[1]}"
MICROBATCH="${CULTURE_MICROBATCH_OVERRIDE:-${RUN_INFO[2]}}"
source scripts/container_helpers.sh
ARGS=(scripts/culture_extension_centaur70b.py --stage "${STAGE}" --run-manifest "${MANIFEST}"
      --output-dir "${OUTPUT}" --micro-batch-size "${MICROBATCH}")
if [[ "${STAGE}" == smoke ]]; then ARGS+=(--max-optimizer-steps 1); fi
container_python "${IMAGE}" 1 "${ARGS[@]}"
python3 scripts/validate_culture_extension_run.py --stage "${STAGE}" --run-manifest "${MANIFEST}" --output-dir "${OUTPUT}"
