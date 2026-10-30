#!/usr/bin/env bash
# Helpers for running the repo inside Viper Apptainer/Singularity images.

set -euo pipefail

load_apptainer() {
    if command -v module >/dev/null 2>&1; then
        module load apptainer/1.4.3 2>/dev/null \
            || module load apptainer/1.2.2 2>/dev/null \
            || true
    fi
    if command -v apptainer >/dev/null 2>&1; then
        CONTAINER_RUNTIME="$(command -v apptainer)"
    elif command -v singularity >/dev/null 2>&1; then
        CONTAINER_RUNTIME="$(command -v singularity)"
    else
        echo "singularity/apptainer not found on PATH" >&2
        exit 1
    fi
}

container_python() {
    local image="$1"
    local use_rocm="$2"
    shift 2

    load_apptainer

    local container_args=(exec --bind "${REPO_ROOT}:${REPO_ROOT}")
    if [[ -n "${HF_HOME:-}" ]]; then
        mkdir -p "${HF_HOME}"
        container_args+=(--bind "${HF_HOME}:/root/.cache/huggingface")
    fi
    if [[ -n "${VIPER_PYTHON_OVERLAY:-}" ]]; then
        if [[ ! -d "${VIPER_PYTHON_OVERLAY}" ]]; then
            echo "VIPER_PYTHON_OVERLAY does not exist: ${VIPER_PYTHON_OVERLAY}" >&2
            exit 1
        fi
        container_args+=(--bind "${VIPER_PYTHON_OVERLAY}:/opt/viper_python_overlay")
        container_args+=(--env PYTHONPATH=/opt/viper_python_overlay)
    fi

    if [[ "${use_rocm}" == "1" ]]; then
        container_args+=(--rocm)
    fi

    "${CONTAINER_RUNTIME}" "${container_args[@]}" \
        --env PYTHONNOUSERSITE=1 \
        --env TOKENIZERS_PARALLELISM=false \
        --env HF_HOME=/root/.cache/huggingface \
        --env TRANSFORMERS_CACHE=/root/.cache/huggingface/transformers \
        --pwd "${REPO_ROOT}" \
        "${image}" \
        python "$@"
}
