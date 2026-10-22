#!/usr/bin/env bash
# Helpers for running the repo inside Viper Apptainer/Singularity images.

set -euo pipefail

load_apptainer() {
    if command -v module >/dev/null 2>&1; then
        module load apptainer/1.4.3 2>/dev/null \
            || module load apptainer/1.2.2 2>/dev/null \
            || true
    fi
    if ! command -v singularity >/dev/null 2>&1 && command -v apptainer >/dev/null 2>&1; then
        alias singularity=apptainer
    fi
    if ! command -v singularity >/dev/null 2>&1; then
        echo "singularity/apptainer not found on PATH" >&2
        exit 1
    fi
}

container_python() {
    local image="$1"
    local use_rocm="$2"
    shift 2

    load_apptainer

    local bind_args=()
    bind_args+=(--bind "${REPO_ROOT}:${REPO_ROOT}")
    if [[ -n "${HF_HOME:-}" ]]; then
        mkdir -p "${HF_HOME}"
        bind_args+=(--bind "${HF_HOME}:/root/.cache/huggingface")
    fi
    local python_overlay_args=()
    if [[ -n "${VIPER_PYTHON_OVERLAY:-}" ]]; then
        if [[ ! -d "${VIPER_PYTHON_OVERLAY}" ]]; then
            echo "VIPER_PYTHON_OVERLAY does not exist: ${VIPER_PYTHON_OVERLAY}" >&2
            exit 1
        fi
        bind_args+=(--bind "${VIPER_PYTHON_OVERLAY}:/opt/viper_python_overlay")
        python_overlay_args+=(--env PYTHONPATH=/opt/viper_python_overlay)
    fi

    local rocm_arg=()
    if [[ "${use_rocm}" == "1" ]]; then
        rocm_arg+=(--rocm)
    fi

    singularity exec "${rocm_arg[@]}" \
        "${bind_args[@]}" \
        --env PYTHONNOUSERSITE=1 \
        --env TOKENIZERS_PARALLELISM=false \
        --env HF_HOME=/root/.cache/huggingface \
        --env TRANSFORMERS_CACHE=/root/.cache/huggingface/transformers \
        "${python_overlay_args[@]}" \
        --pwd "${REPO_ROOT}" \
        "${image}" \
        python "$@"
}
