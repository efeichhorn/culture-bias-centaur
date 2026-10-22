from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from typing import Sequence


DEFAULT_TARGET_MODULES = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
]


@dataclass
class ExperimentConfig:
    output_dir: str = "single_task_transfer_outputs"
    bootstrap_resamples: int = 2000
    model_name: str = "marcelbinz/Llama-3.1-Minitaur-8B"
    num_train_epochs: float = 1.0
    max_steps: int = -1
    learning_rate: float = 5e-5
    warmup_ratio: float = 0.03
    weight_decay: float = 0.01
    per_device_train_batch_size: int = 8
    gradient_accumulation_steps: int = 4
    max_seq_length: int = 768
    dynamic_padding: bool = True
    length_bucketed_batches: bool = True
    data_parallel: bool = False
    bf16: bool = False
    gradient_checkpointing: bool = True
    train_on_eos: bool = True
    train_answer_value_only: bool = False
    attn_implementation: str = "sdpa"
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    target_modules: list[str] = field(default_factory=lambda: list(DEFAULT_TARGET_MODULES))
    seed: int = 0


def _target_modules(value: str | Sequence[str]) -> list[str]:
    if isinstance(value, str):
        return [item.strip() for item in value.split(",") if item.strip()]
    return list(value)


def add_config_args(parser: argparse.ArgumentParser, include_seed: bool = False) -> None:
    parser.add_argument("--output-dir", default=ExperimentConfig.output_dir)
    parser.add_argument("--bootstrap-resamples", type=int, default=ExperimentConfig.bootstrap_resamples)
    parser.add_argument("--model-name", default=ExperimentConfig.model_name)
    parser.add_argument("--num-train-epochs", type=float, default=ExperimentConfig.num_train_epochs)
    parser.add_argument("--max-steps", type=int, default=ExperimentConfig.max_steps)
    parser.add_argument("--learning-rate", type=float, default=ExperimentConfig.learning_rate)
    parser.add_argument("--warmup-ratio", type=float, default=ExperimentConfig.warmup_ratio)
    parser.add_argument("--weight-decay", type=float, default=ExperimentConfig.weight_decay)
    parser.add_argument("--per-device-train-batch-size", type=int, default=ExperimentConfig.per_device_train_batch_size)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=ExperimentConfig.gradient_accumulation_steps)
    parser.add_argument("--max-seq-length", type=int, default=ExperimentConfig.max_seq_length)
    parser.add_argument("--dynamic-padding", dest="dynamic_padding", action="store_true", default=ExperimentConfig.dynamic_padding)
    parser.add_argument("--fixed-padding", dest="dynamic_padding", action="store_false")
    parser.add_argument(
        "--length-bucketed-batches",
        dest="length_bucketed_batches",
        action="store_true",
        default=ExperimentConfig.length_bucketed_batches,
    )
    parser.add_argument("--random-batches", dest="length_bucketed_batches", action="store_false")
    parser.add_argument("--data-parallel", dest="data_parallel", action="store_true", default=ExperimentConfig.data_parallel)
    parser.add_argument("--no-data-parallel", dest="data_parallel", action="store_false")
    parser.add_argument("--bf16", dest="bf16", action="store_true", default=ExperimentConfig.bf16)
    parser.add_argument("--no-bf16", dest="bf16", action="store_false")
    parser.add_argument(
        "--gradient-checkpointing",
        dest="gradient_checkpointing",
        action="store_true",
        default=ExperimentConfig.gradient_checkpointing,
    )
    parser.add_argument("--no-gradient-checkpointing", dest="gradient_checkpointing", action="store_false")
    parser.add_argument("--train-on-eos", dest="train_on_eos", action="store_true", default=ExperimentConfig.train_on_eos)
    parser.add_argument("--no-train-on-eos", dest="train_on_eos", action="store_false")
    parser.add_argument(
        "--train-answer-value-only",
        dest="train_answer_value_only",
        action="store_true",
        default=ExperimentConfig.train_answer_value_only,
        help="Mask response wrappers, punctuation, and EOS from the training loss.",
    )
    parser.add_argument(
        "--train-full-completion",
        dest="train_answer_value_only",
        action="store_false",
        help="Train on the complete rendered response, including wrapper and EOS.",
    )
    parser.add_argument(
        "--attn-implementation",
        choices=["auto", "sdpa", "eager", "flash_attention_2"],
        default=ExperimentConfig.attn_implementation,
    )
    parser.add_argument("--lora-r", type=int, default=ExperimentConfig.lora_r)
    parser.add_argument("--lora-alpha", type=int, default=ExperimentConfig.lora_alpha)
    parser.add_argument("--lora-dropout", type=float, default=ExperimentConfig.lora_dropout)
    parser.add_argument("--target-modules", type=_target_modules, default=",".join(DEFAULT_TARGET_MODULES))
    if include_seed:
        parser.add_argument("--seed", type=int, default=ExperimentConfig.seed)


def config_from_args(args: argparse.Namespace) -> ExperimentConfig:
    values = vars(args).copy()
    if isinstance(values.get("target_modules"), str):
        values["target_modules"] = _target_modules(values["target_modules"])
    return ExperimentConfig(**values)
