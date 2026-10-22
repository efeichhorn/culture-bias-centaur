#!/usr/bin/env python3
"""Reproduce ARCHIVED A1 scores; the behavioral response column is under audit.

No model inference is performed. Five fixed item folds are stitched once per
training seed; they are never counted as five independent training repetitions.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
from scripts.evaluate_dictator_giving_base import load_giving_data
from scripts.evaluate_dictator_taking_transfer import load_taking_data
from scripts.dictator_culture_mixed_split import heldout_rows

KEYS = ["Dataset", "Nation", "SubNo"]
COUNTRIES = {"United States": "us", "China": "china"}
METRICS = ["raw_response_ll", "conditional_ll"]
MIN_REFERENCE_GAIN = 0.05  # A ratio below this denominator is suppressed.


def check_candidates(frame: pd.DataFrame) -> None:
    if frame.duplicated(["GroupNo", "response_value"]).any():
        raise ValueError("Duplicate out-of-fold item/response pairs")
    expected = pd.MultiIndex.from_product([range(1, 26), range(21)])
    actual = pd.MultiIndex.from_frame(frame[["GroupNo", "response_value"]])
    if len(actual) != len(expected) or set(actual) != set(expected):
        raise ValueError("Expected exactly 25 items x 21 responses")
    conditional = frame.groupby("GroupNo")["conditional_probability"].sum()
    if not np.allclose(conditional, 1, atol=1e-8):
        raise ValueError("Conditional probabilities do not sum to one")
    if not np.isfinite(frame[["full_vocab_log_probability", "conditional_probability"]]).all().all():
        raise ValueError("Nonfinite probabilities")
    if (frame["conditional_probability"] <= 0).any():
        raise ValueError("Zero candidate probability")


def score_participants(humans: pd.DataFrame, candidates: pd.DataFrame) -> pd.DataFrame:
    check_candidates(candidates)
    lookup = candidates[["GroupNo", "response_value", "full_vocab_log_probability", "conditional_probability"]].rename(columns={"response_value": "TokenChoice"})
    scored = humans.merge(lookup, on=["GroupNo", "TokenChoice"], how="left", validate="many_to_one")
    if scored["full_vocab_log_probability"].isna().any():
        raise ValueError("Missing human response probability")
    scored["raw_response_ll"] = scored["full_vocab_log_probability"]
    scored["conditional_ll"] = np.log(scored["conditional_probability"])
    groups = scored.groupby(KEYS + ["Country"])
    if not groups.size().eq(25).all():
        raise ValueError("Every evaluation participant must contribute 25 items")
    return groups[METRICS].mean().reset_index()


def histogram_candidates(training: pd.DataFrame) -> pd.DataFrame:
    """Laplace/add-one histogram per target, using training participants only."""
    counts = training.groupby(["GroupNo", "TokenChoice"]).size().reindex(
        pd.MultiIndex.from_product([range(1, 26), range(21)], names=["GroupNo", "response_value"]), fill_value=0
    ).rename("count").reset_index()
    counts["conditional_probability"] = (counts["count"] + 1) / (counts.groupby("GroupNo")["count"].transform("sum") + 21)
    # A discrete reference is normalized over the 21 valid responses. Its LL is
    # directly comparable to conditional LL; its raw-space advantage additionally
    # includes the LLM's invalid-response penalty and is labeled as such.
    counts["full_vocab_log_probability"] = np.log(counts["conditional_probability"])
    return counts


def shared_bootstrap(values: pd.DataFrame, resamples: int, rng: np.random.Generator) -> tuple[np.ndarray, list[int], list[str]]:
    """Return [replicate, seed, outcome] with shared participant multiplicities.

    Within each (Dataset, Nation) stratum, sample the union of observed IDs once
    and use the same multiplicity wherever an ID occurs across seeds. Each seed
    retains its original wave weight; weights are normalized among its observed
    test participants. This preserves dependence when test sets overlap.
    """
    seeds = sorted(values["seed"].unique().tolist())
    outcomes = sorted(values["outcome"].unique().tolist())
    out = np.zeros((resamples, len(seeds), len(outcomes)))
    totals = values[KEYS + ["seed"]].drop_duplicates().groupby("seed").size()
    for _, stratum in values.groupby(["Dataset", "Nation"], sort=True):
        ids = sorted(stratum["SubNo"].unique())
        arrays = []
        for seed in seeds:
            table = stratum.loc[stratum["seed"].eq(seed)].pivot(index="SubNo", columns="outcome", values="value").reindex(index=ids, columns=outcomes)
            present = table.notna().all(axis=1).to_numpy(dtype=float)
            if ((table.notna().sum(axis=1) != 0) & (table.notna().sum(axis=1) != len(outcomes))).any():
                raise ValueError("Inconsistent outcome coverage in bootstrap")
            arrays.append((np.nan_to_num(table.to_numpy()), present, present.sum() / totals.loc[seed]))
        for start in range(0, resamples, 100):
            stop = min(start + 100, resamples)
            weights = rng.multinomial(len(ids), np.full(len(ids), 1 / len(ids)), size=stop-start)
            for j, (matrix, present, wave_weight) in enumerate(arrays):
                if wave_weight == 0:
                    continue
                denominator = weights @ present
                if (denominator == 0).any():
                    raise ValueError("Empty bootstrap stratum; use larger strata")
                out[start:stop, j] += (weights @ matrix) / denominator[:, None] * wave_weight
    return out, seeds, outcomes


class Audit:
    def __init__(self, root: Path, output: Path, resamples: int, seed: int):
        self.root, self.output = root, output
        self.resamples, self.seed = resamples, seed
        self.inputs: set[Path] = set()
        self.parts: list[pd.DataFrame] = []
        self.coverage: list[dict] = []
        self.histograms: list[pd.DataFrame] = []
        self.validation: list[dict] = []
        behavior, groups = root / "data_cache/dictator_Behavior.csv", root / "data_cache/dictator_GroupNo.csv"
        self.inputs.update([behavior, groups])
        # This path exists solely to reproduce invalid archived numbers under an
        # explicit warning, not to certify the old column as allocation behavior.
        self.giving, _ = load_giving_data(behavior, groups, allow_legacy_mislabeled=True)
        self.taking, _ = load_taking_data(behavior, groups, allow_legacy_mislabeled=True)

    def read(self, path: Path) -> pd.DataFrame:
        self.inputs.add(path)
        return pd.read_csv(path)

    def test_rows(self, corpus: Path) -> pd.DataFrame:
        for name in ["protocol.json", "corpus_manifest.csv"]:
            if (corpus / name).exists():
                self.inputs.add(corpus / name)
        manifest = self.read(corpus / "participant_split_manifest.csv")
        return heldout_rows(self.giving, manifest)

    def add(self, protocol: str, task: str, seed: int, condition: str, humans: pd.DataFrame, paths: list[Path], candidates: pd.DataFrame | None = None) -> None:
        if candidates is None:
            candidates = pd.concat([self.read(path) for path in paths], ignore_index=True)
        scores = score_participants(humans, candidates)
        scores = scores.assign(protocol=protocol, task=task, seed=seed, condition=condition)
        # The frozen 8B Giving files use all participants, so their stored metrics
        # are deliberately not compared to our manifest-held-out recomputation.
        if paths and all(path.name == "candidate_probabilities.csv" for path in paths) and not (protocol == "8B_full_item" and task == "Giving" and condition in ["base", "llama"]):
            saved = pd.concat([self.read(path.parent / "model_metrics.csv") for path in paths])
            expected = saved.loc[saved["sample"].eq("Combined")].groupby("country")[["centaur_nll_nats", "conditional_nll_nats"]].mean()
            actual = scores.groupby("Country")[METRICS].mean()
            for country in COUNTRIES:
                for metric, source in [("raw_response_ll", "centaur_nll_nats"), ("conditional_ll", "conditional_nll_nats")]:
                    error = abs(actual.loc[country, metric] + expected.loc[country, source])
                    self.validation.append(dict(protocol=protocol, task=task, seed=seed, condition=condition, Country=country, metric=metric, absolute_error=error))
                    if error > 1e-10:
                        raise ValueError(f"Recomputed score disagrees with stored metric: {protocol}, {task}, {seed}, {condition}, {country}, {metric}, {error}")
        self.parts.append(scores)
        self.coverage.append(dict(protocol=protocol, task=task, seed=seed, condition=condition,
            items=candidates.GroupNo.nunique(), participants=len(scores), responses=len(humans),
            source_files=";".join(str(p.relative_to(self.root)) for p in paths),
            evaluation_population="manifest test participants" if task == "Giving" else "separately collected Taking samples; no Taking responses used to train"))

    def add_histograms(self, protocol: str, seed: int, corpus: Path, humans: pd.DataFrame) -> None:
        manifest = self.read(corpus / "training_participant_manifest.csv")
        test_ids = set(map(tuple, humans[KEYS].drop_duplicates().to_numpy()))
        for condition in ["us", "china", "pooled"]:
            ids = manifest.loc[manifest.condition.eq(condition), KEYS].drop_duplicates()
            if test_ids & set(map(tuple, ids.to_numpy())):
                raise ValueError("Histogram train/test participant overlap")
            training = self.giving.merge(ids, on=KEYS, validate="many_to_one")
            candidates = histogram_candidates(training)
            self.histograms.append(candidates.assign(protocol=protocol, seed=seed, condition=condition))
            self.add(protocol, "Giving", seed, f"histogram_{condition}", humans, [corpus / "training_participant_manifest.csv"], candidates)

    def collect(self) -> None:
        results = self.root / "model_results"
        self.inputs.update([self.root / f"data_cache/dictator_centaur70b_item_holdout/{name}" for name in ["protocol.json", "item_fold_manifest.csv"]])
        for seed in range(3):
            corpus = self.root / f"data_cache/dictator_centaur70b/seed{seed}"
            humans = self.test_rows(corpus)
            run = results / f"dictator_centaur70b/seed{seed}"
            for condition in ["base", "us", "china", "pooled"]:
                self.add("70B_full_item", "Giving", seed, condition, humans, [run / condition / "evaluation/candidate_probabilities.csv"])
            self.add_histograms("70B_full_item", seed, corpus, humans)
            run = results / f"dictator_centaur70b_item_holdout/seed{seed}"
            for task, people in [("Giving", humans), ("Taking", self.taking)]:
                self.add("70B_item_holdout", task, seed, "base", people, [run / f"base/evaluations/{task.lower()}/candidate_probabilities.csv"])
                for condition in ["us", "china", "pooled"]:
                    paths = [run / f"fold{fold}/{condition}/evaluations/{task.lower()}/candidate_probabilities.csv" for fold in range(5)]
                    self.add("70B_item_holdout", task, seed, condition, people, paths)
        for seed in range(5):
            corpus = self.root / f"data_cache/dictator_culture_mixed_split_seed{seed}"
            humans = self.test_rows(corpus)
            for task, people in [("Giving", humans), ("Taking", self.taking)]:
                run = results / (f"dictator_culture_mixed_split/minitaur_seed{seed}" if task == "Giving" else f"dictator_taking_transfer/minitaur_seed{seed}")
                base = results / ("dictator_giving_base/marcelbinz_Llama-3.1-Minitaur-8B" if task == "Giving" else "dictator_taking_base/minitaur_seed0/evaluations/base")
                self.add("8B_full_item", task, seed, "base", people, [base / "candidate_probabilities.csv"])
                for condition in ["us", "china", "pooled", "pooled_full"]:
                    self.add("8B_full_item", task, seed, condition, people, [run / f"evaluations/{condition}/candidate_probabilities.csv"])
                if task == "Giving":
                    llama = results / "dictator_giving_base/meta-llama_Llama-3.1-8B/candidate_probabilities.csv"
                    self.add("8B_full_item", task, seed, "llama", people, [llama])
            self.add_histograms("8B_full_item", seed, corpus, humans)

    def summarize(self) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        participants = pd.concat(self.parts, ignore_index=True)
        participants.to_csv(self.output / "participant_log_likelihoods.csv", index=False)
        pd.concat(self.histograms, ignore_index=True).to_csv(self.output / "training_histograms.csv", index=False)
        pd.DataFrame(self.coverage).to_csv(self.output / "coverage.csv", index=False)
        pd.DataFrame(self.validation).to_csv(self.output / "saved_metric_validation.csv", index=False)
        seed_metrics = participants.groupby(["protocol", "task", "seed", "condition", "Country"])[METRICS].mean().reset_index()
        seed_metrics["invalid_mass_penalty_nats"] = seed_metrics.conditional_ll - seed_metrics.raw_response_ll
        seed_metrics.to_csv(self.output / "seed_metrics.csv", index=False)
        metric_summary = seed_metrics.melt(id_vars=["protocol", "task", "seed", "condition", "Country"], value_vars=METRICS+["invalid_mass_penalty_nats"], var_name="metric")
        metric_summary.groupby(["protocol", "task", "condition", "Country", "metric"])["value"].agg(mean="mean", seed_sd="std", seed_min="min", seed_max="max", seeds="count").reset_index().to_csv(self.output / "metrics_summary.csv", index=False)
        contrast_parts = []
        for (protocol, task, country), part in participants.groupby(["protocol", "task", "Country"]):
            own = COUNTRIES[country]
            other = "china" if own == "us" else "us"
            for metric in METRICS:
                wide = part.pivot(index=["seed"]+KEYS, columns="condition", values=metric)
                pairs = {f"{c}_gain_vs_base": (c, "base") for c in ["us", "china", "pooled", "pooled_full"] if c in wide}
                pairs.update({"own_minus_other": (own, other), "own_minus_pooled": (own, "pooled"), "pooled_minus_other": ("pooled", other)})
                if "llama" in wide:
                    pairs.update({"base_minus_llama": ("base", "llama"), "own_minus_llama": (own, "llama")})
                if f"histogram_{own}" in wide:
                    pairs.update({"histogram_own_minus_base": (f"histogram_{own}", "base"), "histogram_own_minus_model_own": (f"histogram_{own}", own), "histogram_own_minus_histogram_other": (f"histogram_{own}", f"histogram_{other}")})
                for name, (left, right) in pairs.items():
                    frame = (wide[left] - wide[right]).rename("value").reset_index()
                    contrast_parts.append(frame.assign(protocol=protocol, task=task, Country=country, metric=metric, contrast=name))
        contrasts = pd.concat(contrast_parts, ignore_index=True)
        seed_contrasts = contrasts.groupby(["protocol", "task", "Country", "metric", "contrast", "seed"])["value"].mean().reset_index()
        seed_contrasts.to_csv(self.output / "seed_contrasts.csv", index=False)
        summaries = seed_contrasts.groupby(["protocol", "task", "Country", "metric", "contrast"])["value"].agg(mean="mean", seed_sd="std", seed_min="min", seed_max="max", seeds="count").reset_index()
        summaries.to_csv(self.output / "contrast_summary.csv", index=False)
        return contrasts, seed_contrasts, summaries

    def bootstrap(self, contrasts: pd.DataFrame) -> pd.DataFrame:
        rng = np.random.default_rng(self.seed)
        primary = contrasts.loc[contrasts.protocol.eq("70B_item_holdout")]
        rows = []
        ratio_rows = []
        for task, task_frame in primary.groupby("task", sort=True):
            selected_seeds = rng.integers(0, task_frame.seed.nunique(), size=(self.resamples, task_frame.seed.nunique()))
            for country, frame in task_frame.groupby("Country", sort=True):
                frame = frame.copy()
                frame["outcome"] = frame.metric + ":" + frame.contrast
                bootstrap, seeds, outcomes = shared_bootstrap(frame, self.resamples, rng)
                fixed = bootstrap.mean(axis=1)
                combined = np.take_along_axis(bootstrap, selected_seeds[:, :, None], axis=1).mean(axis=1)
                observed = frame.groupby(["seed", "outcome"])["value"].mean().groupby("outcome").mean()
                for k, outcome in enumerate(outcomes):
                    metric, contrast = outcome.split(":")
                    for method, array in [("shared_participant_fixed_seeds", fixed), ("shared_participant_and_seed", combined)]:
                        lower, upper = np.quantile(array[:, k], [.025, .975])
                        rows.append(dict(protocol="70B_item_holdout", task=task, Country=country, metric=metric, contrast=contrast, mean=observed[outcome], ci_low=lower, ci_high=upper, method=method, resamples=self.resamples, seeds=len(seeds)))
                own = COUNTRIES[country]
                for metric in METRICS:
                    denominator_name = f"{metric}:{own}_gain_vs_base"
                    denominator = combined[:, outcomes.index(denominator_name)]
                    denominator_point = observed[denominator_name]
                    for numerator_name in ["own_minus_other", "own_minus_pooled"]:
                        numerator_key = f"{metric}:{numerator_name}"
                        numerator = combined[:, outcomes.index(numerator_key)]
                        valid = denominator_point > MIN_REFERENCE_GAIN and np.quantile(denominator, .025) > MIN_REFERENCE_GAIN
                        ratio = observed[numerator_key] / denominator_point if valid else np.nan
                        low, high = np.quantile(numerator / denominator, [.025, .975]) if valid else [np.nan, np.nan]
                        ratio_rows.append(dict(task=task, Country=country, metric=metric, contrast=numerator_name, own_gain_vs_base=denominator_point, fraction_of_own_gain_lost=ratio, ci_low=low, ci_high=high, minimum_reference_gain=MIN_REFERENCE_GAIN, ratio_status="reported" if valid else "suppressed_small_or_nonpositive_reference_gain"))
        result = pd.DataFrame(rows)
        result.to_csv(self.output / "primary_bootstrap_intervals.csv", index=False)
        pd.DataFrame(ratio_rows).to_csv(self.output / "primary_share_lost.csv", index=False)
        return result

    def provenance(self) -> None:
        records = []
        for path in sorted(self.inputs):
            records.append(dict(path=str(path), bytes=path.stat().st_size, sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
        pd.DataFrame(records).to_csv(self.output / "input_provenance.csv", index=False)
        metadata = dict(status="archived_result_reproduction_not_validated_behavior", warning="INVALID as a behavioral analysis: source response column is under audit. Existing adapters may have been trained on a misidentified response column. Numerical reproduction does not establish construct validity.", script=str(Path(__file__).resolve()), script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), bootstrap_seed=self.seed, bootstrap_resamples=self.resamples,
            primary_protocol="70B_item_holdout: 3 training/split seeds; five fixed folds stitched to 25 out-of-fold predictions per seed",
            raw_response_ll="Mean natural log of the raw full-vocabulary response-token probability; nats per response",
            conditional_ll="Mean natural log probability after normalization over valid responses 0..20; nats per response",
            uncertainty="95% percentile bootstrap; shared participant multiplicities across seeds, stratified by Dataset and Nation; separate fixed-seed and resampled-seed intervals. Only three 70B seeds; folds and targets are fixed, not resampled.",
            histogram="Per-target add-one probabilities over 21 outcomes fitted on matching saved training manifests. Only full-item Giving. No held-out participant responses used.",
            missing_comparisons=["Plain Llama-70B response probabilities were absent from original stored runs", "70B full-item Taking absent; 70B item-holdout Taking present"],
            contrast_definitions={"own_minus_other": "Specialist advantage: own-country adapter minus other-country adapter LL on this population", "pooled_minus_other": "Selection penalty: same-size pooled minus other-country adapter LL on this outside population", "own_minus_pooled": "Own-country specialist minus same-size pooled adapter LL", "share_lost": "(own LL - alternative LL)/(own LL - frozen Centaur LL); only reported when denominator and its bootstrap lower limit exceed 0.05 nats"},
            limitations=["Taking rows are all Dataset 3/4 participants from separately collected samples; no Taking responses were used in fitting. Do not describe them as manifest-held-out participants.", "Giving test participants are excluded from culture adapter fitting; upstream Centaur pretraining overlap is not established here.", "Histogram has target-specific training responses and is only compared within full-item Giving. No matched unseen-task Taking histogram is available.", "8B pooled_full uses twice the training participant count and is a secondary size control.", "Seed ranges are observed variation, not confidence intervals.", "Country-specific absolute LLs alone do not establish Western tilt; human response entropy differs by country."])
        (self.output / "analysis_metadata.json").write_text(json.dumps(metadata, indent=2)+"\n")

    def control_analysis(self) -> bool:
        """Use optional fresh inference; the three seed indices are split IDs only."""
        path = self.root / "model_results/dictator_a1_controls_20260930/candidate_probabilities.csv"
        if not path.exists():
            return False
        candidates = self.read(path)
        if len(candidates) != 6300:
            raise ValueError("Controls table must contain all 12 model/task/prompt combinations")
        parts = []
        checks = []
        valid_mass = candidates.groupby(["model_condition", "task", "prompt_variant", "GroupNo"])["valid_response_mass"].first().reset_index()
        valid_mass.groupby(["model_condition", "task", "prompt_variant"])["valid_response_mass"].agg(mean="mean", minimum="min", maximum="max").reset_index().to_csv(self.output / "control_valid_mass.csv", index=False)
        for (model, task, variant), frame in candidates.groupby(["model_condition", "task", "prompt_variant"]):
            task = task.capitalize()
            for seed in range(3):
                humans = self.test_rows(self.root / f"data_cache/dictator_centaur70b/seed{seed}") if task == "Giving" else self.taking
                scores = score_participants(humans, frame).assign(seed=seed, task=task, model=model, prompt_variant=variant)
                parts.append(scores)
            if model == "centaur" and variant == "original":
                previous = self.read(self.root / f"model_results/dictator_centaur70b_item_holdout/seed0/base/evaluations/{task.lower()}/candidate_probabilities.csv")
                joined = frame.merge(previous, on=["GroupNo", "response_value"], suffixes=("_new", "_old"), validate="one_to_one")
                for metric in ["full_vocab_log_probability", "conditional_probability", "valid_response_mass"]:
                    delta = joined[f"{metric}_new"] - joined[f"{metric}_old"]
                    checks.append(dict(task=task, metric=metric, mean_difference=delta.mean(), mean_absolute_difference=delta.abs().mean(), maximum_absolute_difference=delta.abs().max()))
        participants = pd.concat(parts, ignore_index=True)
        participants.to_csv(self.output / "control_participant_log_likelihoods.csv", index=False)
        participants.groupby(["task", "seed", "model", "prompt_variant", "Country"])[METRICS].mean().reset_index().to_csv(self.output / "control_seed_metrics.csv", index=False)
        pd.DataFrame(checks).to_csv(self.output / "control_centaur_repeat_check.csv", index=False)
        contrast_parts = []
        for (task, country), frame in participants.groupby(["task", "Country"]):
            matched = COUNTRIES[country]
            opposite = "china" if matched == "us" else "us"
            for metric in METRICS:
                wide = frame.pivot(index=["seed"]+KEYS, columns=["model", "prompt_variant"], values=metric)
                pairs = {"centaur_minus_llama_original": (("centaur", "original"), ("llama", "original"))}
                for model in ["llama", "centaur"]:
                    pairs[f"{model}_matched_minus_original"] = ((model, matched), (model, "original"))
                    pairs[f"{model}_matched_minus_opposite"] = ((model, matched), (model, opposite))
                for contrast, (left, right) in pairs.items():
                    contrast_parts.append((wide[left]-wide[right]).rename("value").reset_index().assign(task=task, Country=country, metric=metric, contrast=contrast))
        contrasts = pd.concat(contrast_parts, ignore_index=True)
        contrasts.groupby(["task", "Country", "metric", "contrast", "seed"])["value"].mean().reset_index().to_csv(self.output / "control_seed_contrasts.csv", index=False)
        rng = np.random.default_rng(self.seed + 1)
        rows = []
        for (task, country), frame in contrasts.groupby(["task", "Country"]):
            frame = frame.copy()
            frame["outcome"] = frame.metric + ":" + frame.contrast
            bootstrap, _, outcomes = shared_bootstrap(frame, self.resamples, rng)
            fixed = bootstrap.mean(axis=1)
            observed = frame.groupby(["seed", "outcome"])["value"].mean().groupby("outcome").mean()
            for k, outcome in enumerate(outcomes):
                metric, contrast = outcome.split(":")
                lower, upper = np.quantile(fixed[:,k], [.025,.975])
                rows.append(dict(task=task, Country=country, metric=metric, contrast=contrast, mean=observed[outcome], ci_low=lower, ci_high=upper, method="shared_participant_fixed_evaluation_splits", resamples=self.resamples))
        pd.DataFrame(rows).to_csv(self.output / "control_contrast_summary.csv", index=False)
        return True


def plot_primary(summary: pd.DataFrame, seeds: pd.DataFrame, output: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.spines.top": False, "axes.spines.right": False, "pdf.fonttype": 42})
    rows = summary.loc[summary.metric.eq("raw_response_ll") & summary.method.eq("shared_participant_and_seed")]
    seeds = seeds.loc[seeds.protocol.eq("70B_item_holdout") & seeds.metric.eq("raw_response_ll")]
    fig, axes = plt.subplots(2, 2, figsize=(10.4, 7.0), sharey="row")
    fig.suptitle("INVALID BEHAVIOR ANALYSIS — response column under audit", color="#AA2222", fontsize=13, fontweight="bold", y=.995)
    colors = {"United States": "#2476A8", "China": "#C76B26"}
    for column, task in enumerate(["Giving", "Taking"]):
        for row, (contrasts, labels) in enumerate([
            (["us_gain_vs_base", "pooled_gain_vs_base", "china_gain_vs_base"], ["US adapter", "Pooled adapter", "China adapter"]),
            (["own_minus_other", "pooled_minus_other"], ["Own − other country", "Pooled − other country"]),
        ]):
            ax = axes[row, column]
            for index, (country, color) in enumerate(colors.items()):
                offset = -.12 if index == 0 else .12
                table = rows.loc[rows.task.eq(task) & rows.Country.eq(country)].set_index("contrast").loc[contrasts]
                x = np.arange(len(contrasts)) + offset
                y = table["mean"].to_numpy()
                ax.errorbar(x, y, yerr=np.vstack([y-table.ci_low, table.ci_high-y]), fmt="o" if index == 0 else "s", capsize=4, color=color, markersize=6, label=country, zorder=4)
                for j, contrast in enumerate(contrasts):
                    dots = seeds.loc[seeds.task.eq(task) & seeds.Country.eq(country) & seeds.contrast.eq(contrast), "value"].to_numpy()
                    ax.scatter(j+offset+np.linspace(-.04,.04,len(dots)), dots, s=12, color=color, alpha=.6, zorder=5)
            ax.axhline(0, color="0.5", lw=.8)
            ax.grid(axis="y", alpha=.18)
            ax.set_xticks(np.arange(len(labels)), labels)
            ax.set_xlim(-.5, len(labels)-.5)
            if row == 0:
                ax.set_title(f"{'a' if column == 0 else 'b'}  {task}: held-out target groups", loc="left", fontweight="bold")
            else:
                ax.set_title(f"{'c' if column == 0 else 'd'}  Paired country contrasts", loc="left", fontweight="bold")
    axes[0,0].set_ylabel("LL gain over frozen Centaur-70B\n(nats per response)")
    axes[1,0].set_ylabel("Specialist advantage / selection penalty\n(nats per response)")
    handles, labels = axes[0,0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, bbox_to_anchor=(.5,.08), frameon=False)
    fig.text(.5,.037,"Points: mean of 3 seeds; small dots: individual seeds. Bars: shared-participant + seed bootstrap 95% intervals.",ha="center",fontsize=9)
    fig.text(.5,.012,"Five fixed folds supply one prediction per target and seed. Giving uses held-out participants; Taking uses all task participants.",ha="center",fontsize=8.6)
    fig.tight_layout(rect=(0,.14,1,.96),h_pad=2)
    for ext in ["png", "pdf"]:
        fig.savefig(output / f"a1_70b_holdout.{ext}", dpi=220)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=REPO_ROOT.parent)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--bootstrap-resamples", type=int, default=4000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260930)
    args = parser.parse_args()
    output = args.output_dir or args.root / "model_results/a1_audit_20260930"
    output.mkdir(parents=True, exist_ok=True)
    audit = Audit(args.root, output, args.bootstrap_resamples, args.bootstrap_seed)
    audit.collect()
    contrasts, seeds, summary = audit.summarize()
    bootstrap = audit.bootstrap(contrasts)
    plot_primary(bootstrap, seeds, output)
    controls_available = audit.control_analysis()
    audit.provenance()
    metadata_path = output / "analysis_metadata.json"
    metadata = json.loads(metadata_path.read_text())
    metadata["fresh_inference_controls_available"] = controls_available
    if controls_available:
        metadata["missing_comparisons"] = ["70B full-item Taking absent; 70B item-holdout Taking present"]
        metadata["control_uncertainty"] = "Participant bootstrap with fixed evaluation splits; the three split IDs are not three independent inference/training runs. One saved probability per model/task/prompt/item."
    metadata_path.write_text(json.dumps(metadata, indent=2)+"\n")
    print(summary.loc[summary.protocol.eq("70B_item_holdout") & summary.metric.eq("raw_response_ll")].to_string(index=False))
    print(f"Wrote audit to {output}")


if __name__ == "__main__":
    main()
