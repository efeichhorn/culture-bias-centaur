from __future__ import annotations

from pathlib import Path
import math
import os
import random
import time

import pandas as pd

from .config import ExperimentConfig
from .llm_data import CompletionOnlyCollator, PromptCompletionDataset
from .llm_model import build_lora_model, load_base_model, load_tokenizer
from .utils import ensure_dir, save_json, set_global_seed


class LengthGroupedBatchSampler:
    def __init__(
        self,
        lengths: list[int],
        batch_size: int,
        seed: int,
        mega_batch_mult: int = 50,
    ):
        self.lengths = lengths
        self.batch_size = batch_size
        self.seed = seed
        self.mega_batch_size = max(batch_size, batch_size * mega_batch_mult)
        self.epoch = 0

    def __iter__(self):
        rng = random.Random(self.seed + self.epoch)
        self.epoch += 1
        indices = list(range(len(self.lengths)))
        rng.shuffle(indices)

        batches = []
        for offset in range(0, len(indices), self.mega_batch_size):
            chunk = indices[offset : offset + self.mega_batch_size]
            chunk.sort(key=lambda idx: self.lengths[idx])
            for batch_offset in range(0, len(chunk), self.batch_size):
                batch = chunk[batch_offset : batch_offset + self.batch_size]
                if batch:
                    batches.append(batch)

        rng.shuffle(batches)
        return iter(batches)

    def __len__(self) -> int:
        return (len(self.lengths) + self.batch_size - 1) // self.batch_size


class DistributedBatchSampler:
    def __init__(self, batch_sampler, rank: int, world_size: int):
        self.batch_sampler = batch_sampler
        self.rank = rank
        self.world_size = world_size

    def __iter__(self):
        for index, batch in enumerate(self.batch_sampler):
            if index % self.world_size == self.rank:
                yield batch

    def __len__(self) -> int:
        total = len(self.batch_sampler)
        return (total + self.world_size - 1 - self.rank) // self.world_size


def _base_causal_lm(model):
    candidates = []
    if hasattr(model, "get_base_model"):
        candidates.append(model.get_base_model())
    candidates.append(model)
    base_model = getattr(model, "base_model", None)
    if base_model is not None:
        candidates.append(base_model)
        nested = getattr(base_model, "model", None)
        if nested is not None:
            candidates.append(nested)
    for candidate in candidates:
        if hasattr(candidate, "model") and hasattr(candidate, "lm_head"):
            return candidate
    return model


def supports_completion_only_loss(model) -> bool:
    causal_lm = _base_causal_lm(model)
    return hasattr(causal_lm, "model") and hasattr(causal_lm, "lm_head")


def training_schedule(
    dataloader_batches: int,
    gradient_accumulation_steps: int,
    num_train_epochs: float,
    max_steps: int,
) -> tuple[int, int, int]:
    """Resolve updates per epoch, requested updates, and required data passes."""
    updates_per_epoch = max(
        1,
        dataloader_batches // max(1, gradient_accumulation_steps),
    )
    if max_steps > 0:
        planned_steps = max_steps
    else:
        planned_steps = max(1, int(updates_per_epoch * num_train_epochs))
    max_epochs = max(1, math.ceil(planned_steps / updates_per_epoch))
    return updates_per_epoch, planned_steps, max_epochs


def completion_only_causal_lm_loss(model, batch):
    import torch.nn.functional as F

    causal_lm = _base_causal_lm(model)
    backbone = getattr(causal_lm, "model", None)
    lm_head = getattr(causal_lm, "lm_head", None)
    if backbone is None or lm_head is None:
        return model(**batch).loss

    labels = batch["labels"]
    outputs = backbone(
        input_ids=batch["input_ids"],
        attention_mask=None,
        use_cache=False,
        return_dict=True,
    )
    hidden_states = outputs.last_hidden_state
    shift_hidden = hidden_states[:, :-1, :]
    shift_labels = labels[:, 1:]
    label_mask = shift_labels.ne(-100)
    if not bool(label_mask.any()):
        return hidden_states.sum() * 0.0

    selected_hidden = shift_hidden[label_mask]
    selected_labels = shift_labels[label_mask]
    logits = lm_head(selected_hidden)
    return F.cross_entropy(logits.float(), selected_labels, reduction="mean")


class CompletionOnlyLossModule:
    def __init__(self, model):
        import torch

        class _Module(torch.nn.Module):
            def __init__(self, wrapped_model):
                super().__init__()
                self.wrapped_model = wrapped_model

            def forward(self, input_ids, labels, attention_mask=None):
                batch = {
                    "input_ids": input_ids,
                    "labels": labels,
                }
                if attention_mask is not None:
                    batch["attention_mask"] = attention_mask
                return completion_only_causal_lm_loss(self.wrapped_model, batch)

        self.module = _Module(model)


def train_lora_adapter(
    train_rows: pd.DataFrame,
    cfg: ExperimentConfig,
    output_dir: str | Path,
    metrics_path: str | Path,
    seed: int,
) -> dict[str, float]:
    import torch
    from torch.utils.data import DataLoader
    from transformers import get_linear_schedule_with_warmup

    set_global_seed(seed)
    distributed = bool(cfg.data_parallel and int(os.environ.get("WORLD_SIZE", "1")) > 1)
    local_rank = int(os.environ.get("LOCAL_RANK", "0")) if distributed else 0
    rank = int(os.environ.get("RANK", "0")) if distributed else 0
    world_size = int(os.environ.get("WORLD_SIZE", "1")) if distributed else 1
    if distributed:
        torch.cuda.set_device(local_rank)
        torch.distributed.init_process_group(backend="nccl")
    is_main_process = rank == 0

    output_dir = ensure_dir(output_dir)
    tokenizer = load_tokenizer(cfg.model_name)
    model = load_base_model(cfg.model_name, cfg.bf16, cfg.attn_implementation)
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    if hasattr(model, "config"):
        model.config.use_cache = False
    if cfg.gradient_checkpointing:
        model.gradient_checkpointing_enable()
    model = build_lora_model(model, cfg)
    if cfg.gradient_checkpointing and hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    model.to(device)
    loss_model = CompletionOnlyLossModule(model).module
    using_data_parallel = bool(distributed and torch.cuda.is_available() and torch.cuda.device_count() >= world_size)
    if using_data_parallel:
        loss_model = torch.nn.parallel.DistributedDataParallel(
            loss_model,
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=False,
        )
    if is_main_process:
        model.print_trainable_parameters()

    dataset = PromptCompletionDataset(
        train_rows,
        tokenizer=tokenizer,
        max_seq_length=cfg.max_seq_length,
        train_on_eos=cfg.train_on_eos,
        answer_value_only=cfg.train_answer_value_only,
    )
    pad_to_length = None if cfg.dynamic_padding else cfg.max_seq_length
    collator = CompletionOnlyCollator(tokenizer, pad_to_length=pad_to_length)

    generator = torch.Generator()
    generator.manual_seed(seed)
    if cfg.length_bucketed_batches:
        lengths = [len(example["input_ids"]) for example in dataset.examples]
        batch_sampler = LengthGroupedBatchSampler(lengths, cfg.per_device_train_batch_size, seed=seed)
        if distributed:
            batch_sampler = DistributedBatchSampler(batch_sampler, rank=rank, world_size=world_size)
        dataloader = DataLoader(
            dataset,
            batch_sampler=batch_sampler,
            collate_fn=collator,
            pin_memory=False,
        )
    else:
        sampler = None
        shuffle = True
        if distributed:
            sampler = torch.utils.data.distributed.DistributedSampler(
                dataset,
                num_replicas=world_size,
                rank=rank,
                shuffle=True,
                seed=seed,
            )
            shuffle = False
        dataloader = DataLoader(
            dataset,
            batch_size=cfg.per_device_train_batch_size,
            shuffle=shuffle,
            sampler=sampler,
            collate_fn=collator,
            generator=generator,
            pin_memory=False,
        )

    optimizer = torch.optim.AdamW(
        [param for param in model.parameters() if param.requires_grad],
        lr=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
    )
    updates_per_epoch, planned_steps, max_epochs = training_schedule(
        dataloader_batches=len(dataloader),
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        num_train_epochs=cfg.num_train_epochs,
        max_steps=cfg.max_steps,
    )
    warmup_steps = int(planned_steps * cfg.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=planned_steps,
    )

    loss_model.train()
    optimizer.zero_grad(set_to_none=True)
    total_loss = 0.0
    optimizer_steps = 0
    micro_steps = 0
    last_loss = 0.0
    first_loss = None
    accumulation_start = None
    accumulation_transfer_seconds = 0.0
    accumulation_forward_seconds = 0.0
    accumulation_backward_seconds = 0.0
    if is_main_process:
        print(
            f"[train_lora_adapter] device={device} examples={len(dataset)} "
            f"batch={cfg.per_device_train_batch_size} grad_accum={cfg.gradient_accumulation_steps} "
            f"updates_per_epoch={updates_per_epoch} planned_steps={planned_steps} "
            f"required_data_passes={max_epochs} attn={cfg.attn_implementation} "
            f"train_on_eos={cfg.train_on_eos} "
            f"train_answer_value_only={cfg.train_answer_value_only} "
            f"dynamic_padding={cfg.dynamic_padding} length_bucketed={cfg.length_bucketed_batches} "
            f"data_parallel={using_data_parallel} world_size={world_size}",
            flush=True,
        )
    first_param = next(model.parameters(), None)
    if is_main_process:
        print(
            "[train_lora_adapter] visible_devices "
            f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', '')} "
            f"HIP_VISIBLE_DEVICES={os.environ.get('HIP_VISIBLE_DEVICES', '')} "
            f"ROCR_VISIBLE_DEVICES={os.environ.get('ROCR_VISIBLE_DEVICES', '')} "
            f"device_count={torch.cuda.device_count() if torch.cuda.is_available() else 0} "
            f"dtype={getattr(first_param, 'dtype', 'unknown')} "
            f"bf16_supported={torch.cuda.is_bf16_supported() if torch.cuda.is_available() else False} "
            f"selective_lm_head_loss={supports_completion_only_loss(model)}",
            flush=True,
        )

    profile_steps = os.environ.get("PB_TRAIN_PROFILE_STEPS", "").lower() in {"1", "true", "yes"}
    stop = False
    for epoch in range(max_epochs):
        for batch in dataloader:
            if micro_steps % max(1, cfg.gradient_accumulation_steps) == 0:
                accumulation_start = time.perf_counter()
                accumulation_transfer_seconds = 0.0
                accumulation_forward_seconds = 0.0
                accumulation_backward_seconds = 0.0
            if profile_steps and torch.cuda.is_available():
                torch.cuda.synchronize()
            step_start = time.perf_counter()
            batch = {key: value.to(device) for key, value in batch.items()}
            transfer_seconds = time.perf_counter() - step_start
            forward_start = time.perf_counter()
            loss = loss_model(**batch)
            if hasattr(loss, "mean"):
                loss = loss.mean()
            loss = loss / max(1, cfg.gradient_accumulation_steps)
            if profile_steps and torch.cuda.is_available():
                torch.cuda.synchronize()
            forward_seconds = time.perf_counter() - forward_start
            backward_start = time.perf_counter()
            loss.backward()
            if profile_steps and torch.cuda.is_available():
                torch.cuda.synchronize()
            backward_seconds = time.perf_counter() - backward_start
            accumulation_transfer_seconds += transfer_seconds
            accumulation_forward_seconds += forward_seconds
            accumulation_backward_seconds += backward_seconds
            micro_steps += 1
            last_loss = float(loss.detach().cpu()) * max(1, cfg.gradient_accumulation_steps)
            if first_loss is None:
                first_loss = last_loss
            total_loss += last_loss

            if micro_steps % max(1, cfg.gradient_accumulation_steps) == 0:
                optimizer_start = time.perf_counter()
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                if profile_steps and torch.cuda.is_available():
                    torch.cuda.synchronize()
                optimizer_seconds = time.perf_counter() - optimizer_start
                step_seconds = time.perf_counter() - (accumulation_start or step_start)
                optimizer_steps += 1
                message = (
                    f"[train_lora_adapter] step={optimizer_steps}/{planned_steps} "
                    f"loss={last_loss:.6f} step_seconds={step_seconds:.2f}"
                )
                if profile_steps:
                    message += (
                        f" transfer={accumulation_transfer_seconds:.2f} "
                        f"forward={accumulation_forward_seconds:.2f} "
                        f"backward={accumulation_backward_seconds:.2f} "
                        f"optimizer={optimizer_seconds:.2f}"
                    )
                if is_main_process:
                    print(message, flush=True)
                if optimizer_steps >= planned_steps:
                    stop = True
                    break
        if stop:
            break

    metrics = {
        "train_loss": float(total_loss / max(1, micro_steps)),
        "last_train_loss": float(last_loss),
        "first_train_loss": float(first_loss if first_loss is not None else last_loss),
        "train_steps": int(optimizer_steps),
        "train_micro_steps": int(micro_steps),
        "planned_steps": int(planned_steps),
        "updates_per_epoch": int(updates_per_epoch),
        "required_data_passes": int(max_epochs),
        "train_answer_value_only": bool(cfg.train_answer_value_only),
        "mean_labeled_tokens_per_row": float(
            sum(sum(label != -100 for label in example["labels"]) for example in dataset.examples)
            / max(1, len(dataset.examples))
        ),
    }
    metrics["train_rows"] = int(len(train_rows))
    metrics["adapter_path"] = str(output_dir)
    if is_main_process:
        model.save_pretrained(str(output_dir))
        tokenizer.save_pretrained(str(output_dir))
        save_json(metrics_path, metrics)
    if distributed:
        torch.distributed.barrier()
        torch.distributed.destroy_process_group()

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return metrics
