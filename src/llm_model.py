from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version

from .config import ExperimentConfig
from .llm_data import prepare_tokenizer


def _version_tuple(raw: str) -> tuple[int, ...]:
    parts = []
    for item in raw.split("."):
        digits = ""
        for char in item:
            if char.isdigit():
                digits += char
            else:
                break
        if digits:
            parts.append(int(digits))
        else:
            break
    return tuple(parts)


def require_llm_dependencies() -> None:
    missing = []
    too_old = []
    for module in ["torch", "transformers", "peft", "accelerate"]:
        try:
            __import__(module)
        except Exception:
            missing.append(module)
    min_versions = {
        "transformers": "4.43",
        "peft": "0.12",
        "accelerate": "0.33",
    }
    for package, minimum in min_versions.items():
        if package in missing:
            continue
        try:
            found = version(package)
        except PackageNotFoundError:
            missing.append(package)
            continue
        if _version_tuple(found) < _version_tuple(minimum):
            too_old.append(f"{package}>={minimum} required, found {found}")
    if missing or too_old:
        parts = []
        if missing:
            parts.append("missing: " + ", ".join(sorted(set(missing))))
        if too_old:
            parts.append("outdated: " + "; ".join(too_old))
        raise RuntimeError(
            "LLM dependency problems ("
            + " | ".join(parts)
            + "). Install a GPU-compatible PyTorch build, then run "
            "`pip install -r requirements-training.txt`."
        )


def resolve_torch_dtype(bf16: bool):
    import torch

    if not torch.cuda.is_available():
        return torch.float32
    if bf16:
        if resolve_training_bf16(bf16):
            return torch.bfloat16
        print(
            "[resolve_torch_dtype] BF16 was requested, but this GPU does not "
            "report native BF16 support; using FP16 instead.",
            flush=True,
        )
        return torch.float16
    return torch.float32


def resolve_training_bf16(bf16: bool) -> bool:
    import torch

    if not bf16 or not torch.cuda.is_available():
        return False
    supported = getattr(torch.cuda, "is_bf16_supported", None)
    if supported is None:
        return True
    try:
        return bool(supported())
    except Exception:
        return True


def load_tokenizer(model_name: str):
    require_llm_dependencies()
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    return prepare_tokenizer(tokenizer)


def load_base_model(model_name: str, bf16: bool, attn_implementation: str = "auto"):
    require_llm_dependencies()
    from transformers import AutoModelForCausalLM

    dtype = resolve_torch_dtype(bf16)
    kwargs = {"torch_dtype": dtype}
    if attn_implementation != "auto":
        kwargs["attn_implementation"] = attn_implementation
    try:
        model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
    except Exception:
        if attn_implementation == "auto":
            raise
        print(
            f"[load_base_model] Falling back to default attention after "
            f"{attn_implementation!r} failed.",
            flush=True,
        )
        model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=dtype)
    return model


def build_lora_model(base_model, cfg: ExperimentConfig):
    require_llm_dependencies()
    from peft import LoraConfig, TaskType, get_peft_model

    lora_config = LoraConfig(
        r=cfg.lora_r,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        target_modules=cfg.target_modules,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    return get_peft_model(base_model, lora_config)


def device_for_eval():
    import torch

    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
