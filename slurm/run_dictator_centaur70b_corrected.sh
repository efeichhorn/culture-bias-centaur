#!/usr/bin/env bash
#SBATCH --job-name=c70-corrected
#SBATCH --output=slurm/c70-corrected.%x.%A_%a.out
#SBATCH --error=slurm/c70-corrected.%x.%A_%a.err
set -euo pipefail

REPO_ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "${REPO_ROOT}"
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HOME="${HF_HOME:?Set HF_HOME to the model cache}"
export HSA_XNACK="${HSA_XNACK:-1}"
export VIPER_PYTHON_OVERLAY="${VIPER_PYTHON_OVERLAY:?Set the verified Python overlay}"
if [[ -n "${HF_TOKEN:-}" ]]; then export HUGGING_FACE_HUB_TOKEN="${HF_TOKEN}"; fi

KIND="${C70_RUN_KIND:?C70_RUN_KIND must be smoke, full, or base}"
INDEX="${SLURM_ARRAY_TASK_ID:-0}"
ROOT="${C70_OUTPUT_ROOT:-outputs/corrected70b}"
BEHAVIOR="${C70_BEHAVIOR_PATH:-data/corrected/Behavior.csv}"
GROUP_PATH="${C70_GROUP_PATH:-data/corrected/GroupNo.csv}"
CONDITIONS=(us china pooled)
case "${KIND}" in
  smoke) SEED=0; FOLD=0; CONDITION=us; STEPS=1; OUTPUT="${ROOT}/smoke/seed0/fold0/us" ;;
  full)
    (( INDEX >= 0 && INDEX < 45 )) || { echo "Invalid full array index" >&2; exit 2; }
    SEED=$((INDEX / 15)); FOLD=$(((INDEX % 15) / 3)); CONDITION="${CONDITIONS[$((INDEX % 3))]}"; STEPS=25
    OUTPUT="${ROOT}/seed${SEED}/fold${FOLD}/${CONDITION}" ;;
  base)
    (( INDEX >= 0 && INDEX < 3 )) || { echo "Invalid base array index" >&2; exit 2; }
    SEED="${INDEX}"; FOLD=-1; CONDITION=base; STEPS=0; OUTPUT="${ROOT}/seed${SEED}/base" ;;
  *) echo "Unknown C70_RUN_KIND=${KIND}" >&2; exit 2 ;;
esac
CORPUS="${C70_CORPUS_ROOT:-corpora}/seed${SEED}"
VALIDATE=(python3 scripts/validate_dictator_centaur70b_corrected.py --behavior-path "${BEHAVIOR}" --corpus-dir "${CORPUS}" --seed "${SEED}")
"${VALIDATE[@]}"
if [[ -e "${OUTPUT}" ]]; then
  if [[ "${C70_SKIP_VERIFIED_DONE:-0}" == "1" && -f "${OUTPUT}/verified_complete.json" ]]; then
    "${VALIDATE[@]}" --output-dir "${OUTPUT}" --fold "${FOLD}" --condition "${CONDITION}" --steps "${STEPS}"
    echo "Verified completed job skipped: ${OUTPUT}"
    exit 0
  fi
  echo "Refusing existing output ${OUTPUT}; only explicit C70_SKIP_VERIFIED_DONE=1 with verified completion may skip it." >&2
  exit 2
fi
mkdir -p "$(dirname "${OUTPUT}")"
mkdir "${OUTPUT}"

COMMON=(scripts/dictator_centaur70b_item_holdout.py
  --condition "${CONDITION}" --fold "${FOLD}" --seed "${SEED}"
  --behavior-path "${BEHAVIOR}" --group-path "${GROUP_PATH}" --corpus-dir "${CORPUS}" --output-dir "${OUTPUT}"
  --base-model unsloth/Meta-Llama-3.1-70B-bnb-4bit
  --centaur-adapter marcelbinz/Llama-3.1-Centaur-70B-adapter
  --learning-rate 5e-5 --weight-decay 0.01 --warmup-steps 3
  --micro-batch-size 20 --gradient-accumulation-steps 1 --max-seq-length 768
  --lora-r 8 --lora-alpha 8 --lora-dropout 0 --use-rslora --gradient-checkpointing
  --optimizer adamw --training-objective item_distribution --distribution-repeats 25
  --attn-implementation sdpa --step0-items 5 --step0-tolerance 1e-6)
python3 - "${OUTPUT}" "${KIND}" "${BEHAVIOR}" "${STEPS}" "${COMMON[@]}" <<'PY'
import datetime, hashlib, json, os, pathlib, sys
output, kind, behavior, steps, *common = sys.argv[1:]
commands = [] if kind == 'base' else [common + ['--stage', 'train'] + (['--max-optimizer-steps', '1'] if kind == 'smoke' else [])]
commands.append(common + ['--stage', 'evaluate'])
record = dict(status='started', started_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
              kind=kind, expected_optimizer_steps=int(steps), commands=commands,
              source_behavior_sha256=hashlib.sha256(pathlib.Path(behavior).read_bytes()).hexdigest(),
              slurm_job_id=os.environ.get('SLURM_JOB_ID'), slurm_array_job_id=os.environ.get('SLURM_ARRAY_JOB_ID'),
              slurm_array_task_id=os.environ.get('SLURM_ARRAY_TASK_ID'),
              container=os.environ.get('VIPER_TRAIN_CONTAINER'), overlay=os.environ.get('VIPER_PYTHON_OVERLAY'))
pathlib.Path(output, 'job_manifest.json').write_text(json.dumps(record, indent=2)+'\n')
PY
source scripts/container_helpers.sh
: "${VIPER_TRAIN_CONTAINER:?VIPER_TRAIN_CONTAINER is required}"
if [[ "${KIND}" != "base" ]]; then
  TRAIN=("${COMMON[@]}" --stage train)
  [[ "${KIND}" != "smoke" ]] || TRAIN+=(--max-optimizer-steps 1)
  container_python "${VIPER_TRAIN_CONTAINER}" 1 "${TRAIN[@]}"
fi
container_python "${VIPER_TRAIN_CONTAINER}" 1 "${COMMON[@]}" --stage evaluate
"${VALIDATE[@]}" --output-dir "${OUTPUT}" --fold "${FOLD}" --condition "${CONDITION}" --steps "${STEPS}" --write-completion
