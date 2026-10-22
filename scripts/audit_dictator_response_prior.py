#!/usr/bin/env python3
"""Reproduce the INVALID historic response-prior diagnostic for forensic review.

SOURCE AUDIT: the archived source Token column is competence, not allocation.
Every output from this script is invalid as evidence about Giving/Taking
behavior. Retained to document how the source-column error affected analysis.

The diagnostic multiplies each base response probability by a learned weight
shared across target groups. Twenty free logit offsets are fitted to the exact
Giving training participants and 20 training targets of each archived fold.
Offsets transfer unchanged to the five excluded Giving and Taking targets.
This changes response priors while retaining base differences in target log
odds. It is a lower-capacity comparator, not a causal decomposition of LoRA.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

PARTICIPANT_KEYS = ["Dataset", "Nation", "SubNo"]
COUNTRIES = {0: "United States", 1: "China"}
CONDITIONS = ("us", "china", "pooled")
VALUES = np.arange(21)


def log_softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=-1, keepdims=True)
    return shifted - np.log(np.exp(shifted).sum(axis=-1, keepdims=True))


def objective_derivatives(
    theta: np.ndarray, base_logp: np.ndarray, counts: np.ndarray, ridge: float
) -> tuple[float, np.ndarray, np.ndarray]:
    """Mean choice CE plus .5 * ridge * squared centered-offset norm."""
    offsets = np.r_[theta, 0.0]
    centered = offsets - offsets.mean()
    logp = log_softmax(base_logp + offsets)
    p = np.exp(logp)
    normalized_counts = counts / counts.sum()
    weights = normalized_counts.sum(axis=1)
    value = -(normalized_counts * logp).sum() + 0.5 * ridge * (centered @ centered)
    full_gradient = (weights[:, None] * p - normalized_counts).sum(axis=0)
    full_gradient += ridge * centered
    k = len(offsets)
    full_hessian = np.diag((weights[:, None] * p).sum(axis=0))
    full_hessian -= np.einsum("i,ij,ik->jk", weights, p, p)
    full_hessian += ridge * (np.eye(k) - np.ones((k, k)) / k)
    return float(value), full_gradient[:-1], full_hessian[:-1, :-1]


def fit_response_offsets(
    base_logp: np.ndarray, counts: np.ndarray, ridge: float = 0.001
) -> tuple[np.ndarray, dict]:
    """Damped Newton optimization of the convex 20-parameter objective."""
    base_logp = np.asarray(base_logp, dtype=float)
    counts = np.asarray(counts, dtype=float)
    if base_logp.shape != counts.shape or base_logp.ndim != 2:
        raise ValueError("Base probabilities and counts must be matching matrices.")
    if not np.isfinite(base_logp).all() or not np.isfinite(counts).all():
        raise ValueError("Inputs must be finite.")
    if np.any(counts < 0) or counts.sum() <= 0 or ridge < 0:
        raise ValueError("Counts must be nonnegative and nonempty; ridge nonnegative.")
    if not np.allclose(np.exp(base_logp).sum(axis=1), 1, atol=1e-8):
        raise ValueError("Base conditional probabilities must sum to one.")
    theta = np.zeros(counts.shape[1] - 1)
    initial = objective_derivatives(theta, base_logp, counts, ridge)[0]
    for iteration in range(100):
        value, gradient, hessian = objective_derivatives(theta, base_logp, counts, ridge)
        if np.max(np.abs(gradient)) < 1e-10:
            break
        direction = np.linalg.solve(hessian, -gradient)
        step = 1.0
        directional_derivative = gradient @ direction
        for _ in range(50):
            proposed = theta + step * direction
            proposed_value = objective_derivatives(proposed, base_logp, counts, ridge)[0]
            if proposed_value <= value + 1e-4 * step * directional_derivative:
                theta = proposed
                break
            step *= 0.5
        else:
            raise RuntimeError("Newton line search did not converge.")
    value, gradient, _ = objective_derivatives(theta, base_logp, counts, ridge)
    if np.max(np.abs(gradient)) > 1e-8:
        raise RuntimeError(f"Response-offset fit failed: gradient={gradient}")
    offsets = np.r_[theta, 0.0]
    offsets -= offsets.mean()
    diagnostics = {
        "converged": True,
        "newton_iterations": iteration,
        "initial_objective": initial,
        "final_objective": value,
        "max_abs_gradient": float(np.max(np.abs(gradient))),
        "training_responses": int(counts.sum()),
        "offset_l2_norm": float(np.linalg.norm(offsets)),
    }
    return offsets, diagnostics


def apply_offsets(candidates: pd.DataFrame, offsets: np.ndarray) -> pd.DataFrame:
    """Preserve base valid-response mass when constructing raw log likelihood."""
    frame = candidates.sort_values(["GroupNo", "response_value"]).copy()
    matrix = frame.pivot(index="GroupNo", columns="response_value", values="conditional_probability")
    if list(matrix.columns) != list(VALUES) or matrix.isna().any().any():
        raise ValueError("Expected 21 complete candidate probabilities per target.")
    calibrated = log_softmax(np.log(matrix.to_numpy()) + offsets)
    frame["conditional_log_probability"] = calibrated.ravel()
    frame["conditional_probability"] = np.exp(calibrated).ravel()
    frame["full_vocab_log_probability"] = (
        frame["conditional_log_probability"] + np.log(frame["valid_response_mass"])
    )
    frame["full_vocab_probability"] = np.exp(frame["full_vocab_log_probability"])
    return frame


def training_counts(
    behavior: pd.DataFrame, selected_participants: pd.DataFrame, groups: list[int]
) -> tuple[np.ndarray, int]:
    """Select by explicit training participant keys and explicit training targets."""
    if selected_participants[PARTICIPANT_KEYS].duplicated().any():
        raise ValueError("Repeated participant in training selection.")
    selected = behavior.loc[behavior.Dataset.isin([1, 2])].merge(
        selected_participants[PARTICIPANT_KEYS], on=PARTICIPANT_KEYS, validate="many_to_one"
    )
    selected = selected.loc[selected.GroupNo.isin(groups)]
    counts_per_person = selected.groupby(PARTICIPANT_KEYS).size()
    if not counts_per_person.eq(len(groups)).all() or not len(counts_per_person):
        raise ValueError("Each training participant must have every selected target.")
    counts = pd.crosstab(selected.GroupNo, selected.response_value).reindex(
        index=groups, columns=VALUES, fill_value=0
    )
    return counts.to_numpy(dtype=float), len(counts_per_person)


def score(humans: pd.DataFrame, candidates: pd.DataFrame) -> pd.DataFrame:
    probabilities = candidates[["GroupNo", "response_value", "conditional_probability", "full_vocab_log_probability"]]
    if probabilities.duplicated(["GroupNo", "response_value"]).any():
        raise ValueError("OOF predictions contain repeated targets or responses.")
    merged = humans.merge(probabilities, on=["GroupNo", "response_value"], validate="many_to_one")
    if len(merged) != len(humans):
        raise ValueError("Some human responses have no matching prediction.")
    merged["conditional_ll"] = np.log(merged.conditional_probability)
    participant = merged.groupby(PARTICIPANT_KEYS, as_index=False).agg(
        conditional_ll=("conditional_ll", "mean"),
        raw_response_ll=("full_vocab_log_probability", "mean"),
        responses=("GroupNo", "size"),
    )
    if not participant.responses.eq(25).all():
        raise ValueError("OOF evaluation must score all 25 targets per participant.")
    participant["country"] = participant.Nation.map(COUNTRIES)
    return participant


def make_plot(metrics: pd.DataFrame, output: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(10, 7), sharey=True)
    order = ["us", "pooled", "china"]
    styles = [("response_prior", "Response prior only", "#168078", -0.12, "o"),
              ("lora", "LoRA adapter", "#6556A5", 0.12, "s")]
    for row, task in enumerate(["Giving", "Taking"]):
        for col, country in enumerate(COUNTRIES.values()):
            ax = axes[row, col]
            for family, label, color, shift, marker in styles:
                subset = metrics.loc[(metrics.task == task) & (metrics.country == country) & (metrics.family == family)]
                for i, condition in enumerate(order):
                    values = subset.loc[subset.condition == condition, "conditional_gain"].to_numpy()
                    ax.scatter(i + shift + np.linspace(-0.022, 0.022, len(values)), values,
                               s=20, alpha=.48, color=color, marker=marker)
                    ax.scatter(i + shift, values.mean(), s=85, color=color, marker=marker,
                               edgecolor="white", linewidth=.8, label=label if i == 0 else None)
            ax.axhline(0, color="#555555", linewidth=.8)
            ax.set_xticks(range(3), ["US corpus", "Pooled corpus", "China corpus"])
            ax.set_title(f"{task} · {country}", loc="left", fontsize=11)
            ax.spines[["top", "right"]].set_visible(False)
            ax.grid(axis="y", alpha=.15)
            ax.set_axisbelow(True)
            if col == 0:
                ax.set_ylabel("Conditional log-likelihood gain\n(nats per response)")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(.5, .96))
    fig.suptitle("INVALID as behavioral evidence: archived competence targets", fontsize=13, y=.995, color="#A32626")
    fig.text(.5, .018, "Large symbols: mean of 3 seeds. Small symbols: individual seeds. Zero: frozen Centaur.\n"
             "Fit on Giving training targets only; evaluate the 5 excluded targets in each fold.",
             ha="center", fontsize=9)
    fig.tight_layout(rect=(0, .07, 1, .91))
    fig.savefig(output / "response_prior_comparison.png", dpi=200)
    fig.savefig(output / "response_prior_comparison.pdf")
    plt.close(fig)


def run(workspace: Path, output: Path, ridge: float = 0.001) -> None:
    output.mkdir(parents=True, exist_ok=True)
    root = workspace / "model_results/dictator_centaur70b_item_holdout"
    corpus_root = workspace / "data_cache/dictator_centaur70b"
    protocol = {
        "status": "INVALID_AS_BEHAVIORAL_EVIDENCE",
        "actual_target": "source Token column is competence rating, discretized to0..20",
        "source_column_audit": "All5372 retained participants matched all75 values against raw questionnaires; corrected Token=source Warmth.",
        "diagnostic": "shared_response_logit_offsets",
        "ridge_fixed_before_evaluation": ridge,
        "objective": "mean training choice cross entropy + 0.5 * ridge * sum(centered_offsets**2)",
        "free_parameters": 20,
        "solver": "damped Newton with Armijo backtracking; no test-based tuning",
        "fit_inputs": "exact archived 70B Giving training participants and 20 training targets per fold",
        "evaluation": "five excluded Giving/Taking targets per fold, stitched into 25 targets per seed",
        "giving_people": "70B manifest test participants only",
        "taking_people": "all separate Taking sample rows (Dataset 3,4)",
        "raw_likelihood": "preserves frozen Centaur valid-response mass per target",
        "primary_metric": "conditional log likelihood over response numbers 0 through 20",
        "interpretation": "Adjusts only response priors; retains base target log-odds differences. Lower-capacity diagnostic, not causal attribution.",
        "uncertainty": "three seed values and mean/range; these are not confidence intervals",
    }
    # This fixed design is persisted before any fit or test scoring occurs.
    (output / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")
    behavior_path = workspace / "data_cache/dictator_Behavior.csv"
    behavior = pd.read_csv(behavior_path)
    behavior["response_value"] = np.floor(behavior.Token / 5 + .5).clip(0, 20).astype(int)
    input_paths = {behavior_path}
    metric_parts, participant_parts, fit_rows, offset_rows, candidate_parts = [], [], [], [], []
    for seed in [0, 1, 2]:
        split_path = corpus_root / f"seed{seed}/participant_split_manifest.csv"
        train_path = corpus_root / f"seed{seed}/training_participant_manifest.csv"
        input_paths.update([split_path, train_path])
        split = pd.read_csv(split_path)
        training_manifest = pd.read_csv(train_path)
        test_keys = split.loc[split.split == "test", PARTICIPANT_KEYS]
        train_keys = split.loc[split.split == "train", PARTICIPANT_KEYS]
        if len(train_keys.merge(test_keys, on=PARTICIPANT_KEYS)):
            raise ValueError("Giving test participants leaked into training.")
        humans = {
            "Giving": behavior.loc[behavior.Dataset.isin([1, 2])].merge(test_keys, on=PARTICIPANT_KEYS, validate="many_to_one"),
            "Taking": behavior.loc[behavior.Dataset.isin([3, 4])],
        }
        bases = {}
        for task in humans:
            path = root / f"seed{seed}/base/evaluations/{task.lower()}/candidate_probabilities.csv"
            input_paths.add(path)
            bases[task] = pd.read_csv(path)
        predictions = {(task, "base", "base"): frame for task, frame in bases.items()}
        for condition in CONDITIONS:
            selected = training_manifest.loc[training_manifest.condition == condition]
            if len(selected.merge(train_keys, on=PARTICIPANT_KEYS)) != len(selected):
                raise ValueError("Condition includes participants outside the training split.")
            assembled = {(task, family): [] for task in humans for family in ["lora", "response_prior"]}
            for fold in range(5):
                folder = root / f"seed{seed}/fold{fold}/{condition}"
                metadata_path = folder / "train_metadata.json"
                input_paths.add(metadata_path)
                metadata = json.loads(metadata_path.read_text())
                training_groups = sorted(metadata["training_group_nos"])
                test_groups = sorted(metadata["heldout_group_nos"])
                if set(training_groups) & set(test_groups) or sorted(training_groups + test_groups) != list(range(1, 26)):
                    raise ValueError("Invalid archived target split.")
                counts, people = training_counts(behavior, selected, training_groups)
                if people != metadata["training_participants"] or counts.sum() != metadata["training_responses"]:
                    raise ValueError("Calibration corpus does not match archived LoRA corpus size.")
                base_matrix = bases["Giving"].pivot(index="GroupNo", columns="response_value", values="conditional_probability").loc[training_groups, VALUES]
                offsets, diagnostics = fit_response_offsets(np.log(base_matrix.to_numpy()), counts, ridge)
                fit_rows.append(dict(seed=seed, fold=fold, condition=condition, participants=people, **diagnostics))
                offset_rows.extend(dict(seed=seed, fold=fold, condition=condition, response_value=int(v), centered_logit_offset=float(offsets[v])) for v in VALUES)
                for task in humans:
                    path = folder / f"evaluations/{task.lower()}/candidate_probabilities.csv"
                    input_paths.add(path)
                    lora = pd.read_csv(path)
                    if sorted(lora.GroupNo.unique()) != test_groups:
                        raise ValueError("Archived LoRA evaluation has wrong target groups.")
                    assembled[(task, "lora")].append(lora)
                    base_test = bases[task].loc[bases[task].GroupNo.isin(test_groups)]
                    prior = apply_offsets(base_test, offsets)
                    prior["item_fold"] = fold
                    assembled[(task, "response_prior")].append(prior)
            for (task, family), parts in assembled.items():
                predictions[(task, family, condition)] = pd.concat(parts, ignore_index=True)
        for (task, family, condition), candidates in predictions.items():
            participant = score(humans[task], candidates)
            participant = participant.assign(seed=seed, task=task, family=family, condition=condition)
            participant_parts.append(participant)
            aggregated = participant.groupby("country", as_index=False).agg(
                conditional_ll=("conditional_ll", "mean"), raw_response_ll=("raw_response_ll", "mean"),
                participants=("SubNo", "size"), responses=("responses", "sum"),
            )
            metric_parts.append(aggregated.assign(seed=seed, task=task, family=family, condition=condition))
            if family == "response_prior":
                candidate_parts.append(candidates.assign(seed=seed, task=task, family=family, condition=condition))
    metrics = pd.concat(metric_parts, ignore_index=True)
    base_metrics = metrics.loc[metrics.family == "base", ["seed", "task", "country", "conditional_ll", "raw_response_ll"]].rename(
        columns={"conditional_ll": "base_conditional_ll", "raw_response_ll": "base_raw_response_ll"})
    metrics = metrics.merge(base_metrics, on=["seed", "task", "country"], validate="many_to_one")
    metrics["conditional_gain"] = metrics.conditional_ll - metrics.base_conditional_ll
    metrics["raw_response_gain"] = metrics.raw_response_ll - metrics.base_raw_response_ll
    metrics.to_csv(output / "per_seed_metrics.csv", index=False)
    pd.concat(participant_parts).to_csv(output / "participant_scores.csv", index=False)
    pd.concat(candidate_parts).to_csv(output / "oof_response_prior_probabilities.csv", index=False)
    pd.DataFrame(fit_rows).to_csv(output / "fit_diagnostics.csv", index=False)
    pd.DataFrame(offset_rows).to_csv(output / "fitted_response_offsets.csv", index=False)
    summary = metrics.groupby(["task", "country", "family", "condition"], as_index=False).agg(
        conditional_gain_mean=("conditional_gain", "mean"), conditional_gain_min=("conditional_gain", "min"),
        conditional_gain_max=("conditional_gain", "max"), raw_gain_mean=("raw_response_gain", "mean"),
        conditional_ll_mean=("conditional_ll", "mean"), raw_ll_mean=("raw_response_ll", "mean"),
    )
    summary.to_csv(output / "across_seed_metrics.csv", index=False)
    contrasts = []
    for (seed, task, country, family), frame in metrics.loc[metrics.family != "base"].groupby(["seed", "task", "country", "family"]):
        frame = frame.set_index("condition")
        own, other = ("us", "china") if country == "United States" else ("china", "us")
        for metric in ["conditional_gain", "raw_response_gain"]:
            for name, left in [("own_minus_other", own), ("pooled_minus_other", "pooled")]:
                contrasts.append(dict(seed=seed, task=task, country=country, family=family,
                                      metric=metric, contrast=name, value=float(frame.loc[left, metric] - frame.loc[other, metric])))
    contrast_frame = pd.DataFrame(contrasts)
    contrast_frame.to_csv(output / "per_seed_contrasts.csv", index=False)
    contrast_frame.groupby(["task", "country", "family", "metric", "contrast"], as_index=False).agg(
        mean=("value", "mean"), minimum=("value", "min"), maximum=("value", "max")
    ).to_csv(output / "across_seed_contrasts.csv", index=False)
    comparisons = metrics.loc[metrics.family == "lora"].merge(
        metrics.loc[metrics.family == "response_prior"], on=["seed", "task", "country", "condition"], suffixes=("_lora", "_prior"), validate="one_to_one")
    comparisons["conditional_lora_minus_prior"] = comparisons.conditional_ll_lora - comparisons.conditional_ll_prior
    comparisons["raw_lora_minus_prior"] = comparisons.raw_response_ll_lora - comparisons.raw_response_ll_prior
    comparisons[["seed", "task", "country", "condition", "conditional_lora_minus_prior", "raw_lora_minus_prior"]].to_csv(output / "lora_minus_response_prior.csv", index=False)
    files = [{"path": str(p.resolve()), "sha256": hashlib.sha256(p.read_bytes()).hexdigest()} for p in sorted(input_paths)]
    (output / "input_hashes.json").write_text(json.dumps(files, indent=2) + "\n")
    make_plot(metrics, output)
    print(summary.loc[summary.family != "base"].to_string(index=False))
    print(f"Wrote {output}; {len(fit_rows)} converged fits.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--allow-legacy-mislabeled", action="store_true",
                        help="Explicitly reproduce the invalid historical competence-target diagnostic.")
    args = parser.parse_args()
    if not args.allow_legacy_mislabeled:
        parser.error("Archived targets are competence ratings. Use --allow-legacy-mislabeled only for forensic reproduction.")
    run(args.workspace, args.output_dir or args.workspace / "model_results/a1_response_prior_20260930")
