#!/usr/bin/env python3
"""Rescore Taking on a fixed, wave-matched sample without model inference.

Each country contributes 529 Wave 1 and 800 Wave 2 participants. Whole US
participants are sampled once; all Chinese participants are retained. Selection
is independent of responses and predictions. Only the private selection manifest
contains participant identifiers; all versionable outputs are aggregates.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.analyze_dictator_corrected_baselines import load_explicit_behavior
from scripts.analyze_dictator_corrected_adapters import validate_evaluation, validate_fit_metadata
from scripts.audit_dictator_a1 import score_participants, shared_bootstrap
from scripts.audit_dictator_response_prior import apply_offsets
from scripts.dictator_item_holdout_transfer import ITEM_FOLDS

KEYS = ["Dataset", "Nation", "SubNo"]
METRICS = ["raw_response_ll", "conditional_ll"]
COUNTRIES = {0: "United States", 1: "China"}
CONDITIONS = ["us", "pooled", "china"]
WAVE_COUNTS = {3: 529, 4: 800}
SOURCE_SHA = "65f8913600fddeae3ca99196c2b0e2db76f2bf2021e1b32f4419f954645a5ecd"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def select_balanced_participants(behavior: pd.DataFrame, seed: int) -> pd.DataFrame:
    """Sort before sampling so selection does not depend on source row order."""
    ids = behavior.loc[behavior.Dataset.isin(WAVE_COUNTS), KEYS].drop_duplicates().sort_values(KEYS)
    if set(map(tuple, ids[["Dataset", "Nation"]].drop_duplicates().to_numpy())) != {
        (dataset, nation) for dataset in WAVE_COUNTS for nation in COUNTRIES
    }:
        raise ValueError("Taking must contain both countries in both study waves")
    rng = np.random.default_rng(seed)
    parts = []
    for (dataset, nation), frame in ids.groupby(["Dataset", "Nation"], sort=True):
        count = WAVE_COUNTS[dataset]
        if len(frame) < count or (nation == 1 and len(frame) != count):
            raise ValueError("Unexpected Taking population count; refuse to change the sampling design")
        if len(frame) > count:
            frame = frame.iloc[np.sort(rng.choice(len(frame), count, replace=False))]
        parts.append(frame)
    return pd.concat(parts).sort_values(KEYS).reset_index(drop=True)


def subset_people(frame: pd.DataFrame, ids: pd.DataFrame) -> pd.DataFrame:
    if ids.duplicated(KEYS).any():
        raise ValueError("Selection manifest contains duplicate participants")
    selected = frame.merge(ids, on=KEYS, validate="many_to_one")
    if set(map(tuple, selected[KEYS].drop_duplicates().to_numpy())) != set(map(tuple, ids[KEYS].to_numpy())):
        raise ValueError("Selected participant is missing from an evaluation")
    return selected


def bootstrap_outcomes(values: pd.DataFrame, resamples: int, seed: int, task: str = "Taking") -> tuple[pd.DataFrame, pd.DataFrame]:
    """Paired participant weights across outcomes/seeds, stratified by wave.

Countries have independent participant draws. Observed fitted-seed draws are
shared across countries, preserving the paired model comparison. Targets,
selection, fitted adapters, folds, and response adjustments remain fixed.
"""
    rng = np.random.default_rng(seed)
    seeds = sorted(values.seed.unique())
    chosen = rng.integers(0, len(seeds), size=(resamples, len(seeds)))
    rows, country_rows, samples_by_country, points_by_country = [], [], {}, {}
    outcomes_expected = None
    for country, frame in values.groupby("Country", sort=True):
        frame = frame.copy()
        frame["outcome"] = frame.metric + ":" + frame.contrast
        samples, actual_seeds, outcomes = shared_bootstrap(frame, resamples, rng)
        if actual_seeds != seeds or (outcomes_expected is not None and outcomes != outcomes_expected):
            raise ValueError("Countries do not contain exactly the same model/seed outcomes")
        outcomes_expected = outcomes
        samples_by_country[country] = samples
        points = frame.groupby(["seed", "outcome"]).value.mean().groupby("outcome").mean()
        points_by_country[country] = points
        for method, array in (
            ("shared_participant_fixed_seeds", samples.mean(axis=1)),
            ("shared_participant_and_seed", np.take_along_axis(samples, chosen[:, :, None], axis=1).mean(axis=1)),
        ):
            for j, outcome in enumerate(outcomes):
                metric, contrast = outcome.split(":")
                low, high = np.quantile(array[:, j], [.025, .975])
                rows.append(dict(task=task, Country=country, metric=metric, contrast=contrast,
                                 mean=points[outcome], ci_low=low, ci_high=high, method=method,
                                 seeds=len(seeds), resamples=resamples))
    difference = samples_by_country["United States"] - samples_by_country["China"]
    observed = points_by_country["United States"] - points_by_country["China"]
    for method, array in (
        ("shared_participant_fixed_seeds", difference.mean(axis=1)),
        ("shared_participant_and_seed", np.take_along_axis(difference, chosen[:, :, None], axis=1).mean(axis=1)),
    ):
        for j, outcome in enumerate(outcomes_expected):
            metric, contrast = outcome.split(":")
            low, high = np.quantile(array[:, j], [.025, .975])
            country_rows.append(dict(task=task, population_contrast="United States_minus_China",
                                     metric=metric, contrast=contrast, mean=observed[outcome],
                                     ci_low=low, ci_high=high, method=method,
                                     seeds=len(seeds), resamples=resamples))
    return pd.DataFrame(rows), pd.DataFrame(country_rows)


def score_contrasts(participants: pd.DataFrame) -> pd.DataFrame:
    parts = []
    for country, frame in participants.groupby("Country"):
        own, other = ("us", "china") if country == "United States" else ("china", "us")
        for metric in METRICS:
            wide = frame.pivot(index=["seed"] + KEYS, columns="condition", values=metric)
            if wide.isna().any().any():
                raise ValueError("All models must score identical participants and seeds")
            pairs = {f"{condition}_gain_vs_base": (condition, "base") for condition in CONDITIONS}
            pairs.update({"own_minus_other": (own, other), "pooled_minus_other": ("pooled", other),
                          "own_minus_pooled": (own, "pooled")})
            pairs.update({f"{condition}_prior_minus_lora": (condition + "_prior", condition)
                          for condition in CONDITIONS})
            pairs.update({f"{condition}_prior_gain": (condition + "_prior", "base")
                          for condition in CONDITIONS})
            # Signed US-versus-China adapters, common definition for both test populations.
            pairs["us_minus_china_adapter"] = ("us", "china")
            for name, (left, right) in pairs.items():
                parts.append((wide[left] - wide[right]).rename("value").reset_index().assign(
                    task="Taking", Country=country, metric=metric, contrast=name))
            for condition in wide:
                parts.append(wide[condition].rename("value").reset_index().assign(
                    task="Taking", Country=country, metric=metric, contrast=condition + "_score"))
    return pd.concat(parts, ignore_index=True)


def run(args: argparse.Namespace) -> None:
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    inputs: set[Path] = set()
    validation = []

    def read(path: Path) -> pd.DataFrame:
        inputs.add(path)
        return pd.read_csv(path)

    def save(frame: pd.DataFrame, name: str) -> None:
        path = output / name
        path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(path, index=False)

    source = args.workspace / "data_cache/dictator_Behavior_corrected_20260930.csv"
    if sha(source) != SOURCE_SHA:
        raise ValueError("Corrected allocation source hash differs from the independently audited source")
    inputs.update([source, source.with_suffix(".provenance.json")])
    sidecar = json.loads(source.with_suffix(".provenance.json").read_text())
    if sidecar.get("output_sha256") != SOURCE_SHA:
        raise ValueError("Corrected source provenance does not match")
    behavior = load_explicit_behavior(source, "Token", "percent")
    ids = select_balanced_participants(behavior, args.selection_seed)
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    if args.manifest.exists() and not pd.read_csv(args.manifest).equals(ids):
        raise ValueError("Existing private manifest differs; refuse to silently change selected people")
    ids.to_csv(args.manifest, index=False)
    taking = behavior.loc[behavior.Dataset.isin(WAVE_COUNTS)]
    selected_humans = subset_people(taking, ids)
    if len(selected_humans) != 2658 * 25:
        raise ValueError("Balanced evaluation must have 2,658 whole participants and all 25 targets")
    counts = ids.groupby(["Dataset", "Nation"]).size().rename("selected_participants").reset_index()
    counts["available_participants"] = [len(taking[(taking.Dataset == r.Dataset) & (taking.Nation == r.Nation)][KEYS].drop_duplicates()) for r in counts.itertuples()]
    counts["Country"] = counts.Nation.map(COUNTRIES)
    counts["sample"] = counts.Dataset.map({3: "Wave 1", 4: "Wave 2"})
    save(counts, "sample_counts.csv")

    # Frozen controls reproduce the existing H1 scores before applying selection.
    controls = read(args.workspace / "model_results/dictator_a1_controls_20260930/candidate_probabilities.csv")
    original = read(ROOT / "results/giving_taking/frozen/metrics_summary.csv")
    frozen_parts = []
    for model in ("centaur", "llama"):
        candidates = controls.loc[controls.task.eq("taking") & controls.model_condition.eq(model) & controls.prompt_variant.eq("original")]
        full_scores = score_participants(taking, candidates)
        for country, frame in full_scores.groupby("Country"):
            expected = original.loc[original.task.eq("Taking") & original.Country.eq(country) & original.condition.eq(model + "_original")]
            for metric in METRICS:
                error = abs(frame[metric].mean() - expected.iloc[0][metric])
                if len(expected) != 1 or error > 1e-10:
                    raise ValueError("Full-sample frozen score failed aggregate reproduction")
                validation.append(dict(family="frozen", seed=-1, condition=model, Country=country, metric=metric, absolute_error=error))
        frozen_parts.append(subset_people(full_scores, ids).assign(seed=0, condition=model))
    frozen = pd.concat(frozen_parts, ignore_index=True)
    frozen_values = []
    for country, frame in frozen.groupby("Country"):
        for metric in METRICS:
            wide = frame.pivot(index=["seed"] + KEYS, columns="condition", values=metric)
            for condition in wide:
                frozen_values.append(wide[condition].rename("value").reset_index().assign(Country=country, metric=metric, contrast=condition + "_score"))
            frozen_values.append((wide.centaur - wide.llama).rename("value").reset_index().assign(Country=country, metric=metric, contrast="centaur_minus_llama"))
    frozen_intervals, frozen_country = bootstrap_outcomes(pd.concat(frozen_values), args.resamples, args.bootstrap_seed)
    frozen_intervals = frozen_intervals.loc[frozen_intervals.method.eq("shared_participant_fixed_seeds")].copy()
    frozen_country = frozen_country.loc[frozen_country.method.eq("shared_participant_fixed_seeds")].copy()
    save(frozen_intervals, "frozen/bootstrap_intervals.csv")
    save(frozen_country, "frozen/country_contrasts.csv")
    frozen_summary = frozen.groupby(["condition", "Country"])[METRICS].mean().reset_index()
    frozen_summary["task"] = "Taking"
    frozen_summary.condition += "_original"
    save(frozen_summary, "frozen/metrics_summary.csv")
    frozen_figure = []
    for row in frozen_intervals[frozen_intervals.metric.eq("conditional_ll") & frozen_intervals.contrast.str.endswith("_score")].itertuples():
        frozen_figure.append(dict(task="Taking", Country=row.Country, model=row.contrast.removesuffix("_score"), metric=row.metric,
                                  prediction_error=-row.mean, error_ci_low=-row.ci_high, error_ci_high=-row.ci_low))
    save(pd.DataFrame(frozen_figure), "figure_inputs/frozen_sample_error.csv")
    save(pd.DataFrame(frozen_figure), "frozen_sample_error.csv")

    # H1 compares countries directly. Giving uses the original three evaluation
    # splits and shared weights for overlapping people; models are never refit.
    giving_parts = []
    for seed in range(3):
        split = read(args.workspace / f"data_cache/dictator_centaur70b/seed{seed}/participant_split_manifest.csv")
        giving_ids = split.loc[split.split.eq("test"), KEYS]
        giving_humans = subset_people(behavior.loc[behavior.Dataset.isin([1, 2])], giving_ids)
        for model in ("centaur", "llama"):
            candidates = controls.loc[controls.task.eq("giving") & controls.model_condition.eq(model) & controls.prompt_variant.eq("original")]
            giving_parts.append(score_participants(giving_humans, candidates).assign(seed=seed, condition=model))
    giving = pd.concat(giving_parts, ignore_index=True)
    giving_values = []
    for country, frame in giving.groupby("Country"):
        for metric in METRICS:
            wide = frame.pivot(index=["seed"] + KEYS, columns="condition", values=metric)
            for condition in wide:
                giving_values.append(wide[condition].rename("value").reset_index().assign(Country=country, metric=metric, contrast=condition + "_score"))
                estimate = wide[condition].groupby("seed").mean().mean()
                reference = original.loc[original.task.eq("Giving") & original.Country.eq(country) & original.condition.eq(condition + "_original")]
                error = abs(estimate - reference.iloc[0][metric])
                if len(reference) != 1 or error > 1e-10:
                    raise ValueError("Giving frozen scores failed aggregate reproduction")
                validation.append(dict(family="frozen_giving", seed=-1, condition=condition, Country=country, metric=metric, absolute_error=error))
            giving_values.append((wide.centaur - wide.llama).rename("value").reset_index().assign(Country=country, metric=metric, contrast="centaur_minus_llama"))
    giving_intervals, giving_country = bootstrap_outcomes(pd.concat(giving_values), args.resamples, args.bootstrap_seed, task="Giving")
    giving_intervals = giving_intervals.loc[giving_intervals.method.eq("shared_participant_fixed_seeds")]
    giving_country = giving_country.loc[giving_country.method.eq("shared_participant_fixed_seeds")]
    save(giving_intervals, "frozen/giving_bootstrap_intervals.csv")
    save(giving_country, "frozen/giving_country_contrasts.csv")
    h1_countries = pd.concat([giving_country, frozen_country])
    h1_countries = h1_countries.loc[h1_countries.metric.eq("conditional_ll") & h1_countries.contrast.str.endswith("_score")].copy()
    h1_countries["model"] = h1_countries.contrast.str.removesuffix("_score")
    # China NLL minus US NLL equals US LL minus China LL: no sign inversion.
    h1_countries["contrast"] = "China_minus_United_States_NLL"
    h1_countries["metric"] = "conditional_nll_difference"
    h1_countries["method"] = "shared_participant_fixed_evaluation_splits"
    h1_countries = h1_countries.rename(columns={"mean": "estimate"})
    save(h1_countries[["task", "model", "contrast", "metric", "estimate", "ci_low", "ci_high", "method", "seeds", "resamples"]], "h1_population_contrasts.csv")
    h1_models = pd.concat([giving_intervals, frozen_intervals])
    h1_models = h1_models.loc[h1_models.metric.eq("conditional_ll") & h1_models.contrast.eq("centaur_minus_llama")].copy()
    h1_models = h1_models.rename(columns={"mean": "estimate"})
    h1_models["method"] = "shared_participant_fixed_evaluation_splits"
    save(h1_models, "h1_model_comparisons.csv")

    results_root = args.workspace / "model_results/dictator_centaur70b_item_holdout_corrected_20260930"
    previous = read(ROOT / "results/giving_taking/seed_metrics.csv")
    prior_reference = read(ROOT / "results/giving_taking/response_adjustment/prior_metrics_by_seed.csv")
    offsets = read(ROOT / "results/giving_taking/response_adjustment/prior_offsets.csv")
    parts, shift_rows = [], []
    previous_shifts = read(ROOT / "results/giving_taking/figure_inputs/prediction_shifts_by_seed.csv")
    for seed in range(3):
        corpus_path = args.workspace / f"data_cache/dictator_centaur70b_corrected_20260930/seed{seed}/protocol.json"
        inputs.add(corpus_path)
        corpus = json.loads(corpus_path.read_text())
        if corpus["source_behavior_sha256"] != SOURCE_SHA or corpus["seed"] != seed:
            raise ValueError("Training corpus provenance mismatch")
        base_dir = results_root / f"seed{seed}/base/evaluations/taking"
        base = read(base_dir / "candidate_probabilities.csv")
        inputs.add(base_dir / "run_metadata.json")
        validate_evaluation(json.loads((base_dir / "run_metadata.json").read_text()), base, SOURCE_SHA, sha(corpus_path), seed, -1, "base", "Taking")
        predictions = {"base": base}
        for condition in CONDITIONS:
            lora_parts, prior_parts = [], []
            for fold, held in ITEM_FOLDS.items():
                folder = results_root / f"seed{seed}/fold{fold}/{condition}"
                train_path, metrics_path = folder / "train_metadata.json", folder / "train_metrics.json"
                inputs.update([train_path, metrics_path])
                validate_fit_metadata(json.loads(train_path.read_text()), json.loads(metrics_path.read_text()), SOURCE_SHA, seed, fold, condition, 25)
                evaluation = folder / "evaluations/taking"
                candidates = read(evaluation / "candidate_probabilities.csv")
                inputs.add(evaluation / "run_metadata.json")
                validate_evaluation(json.loads((evaluation / "run_metadata.json").read_text()), candidates, SOURCE_SHA, sha(corpus_path), seed, fold, condition, "Taking")
                lora_parts.append(candidates)
                fitted = offsets.loc[offsets.seed.eq(seed) & offsets.condition.eq(condition) & offsets.fold.eq(fold)].sort_values("response_value")
                if fitted.response_value.tolist() != list(range(21)):
                    raise ValueError("Missing fitted response offsets")
                prior_parts.append(apply_offsets(base.loc[base.GroupNo.isin(held)], fitted.offset.to_numpy()))
            predictions[condition] = pd.concat(lora_parts)
            predictions[condition + "_prior"] = pd.concat(prior_parts)
            expectation = (predictions[condition].conditional_probability * predictions[condition].response_value).sum() / 25
            base_expectation = (base.conditional_probability * base.response_value).sum() / 25
            reference = previous_shifts.loc[previous_shifts.task.eq("Taking") & previous_shifts.seed.eq(seed) & previous_shifts.condition.eq(condition)]
            if len(reference) != 1 or abs(expectation - reference.iloc[0].expected_tokens) > 1e-10:
                raise ValueError("Saved model mean does not match candidate recomputation")
            shift_rows.append(dict(task="Taking", condition=condition, seed=seed, expected_tokens=expectation,
                                   frozen_expected_tokens=base_expectation, shift_tokens=expectation-base_expectation))
        for condition, candidates in predictions.items():
            full_scores = score_participants(taking, candidates)
            for country, frame in full_scores.groupby("Country"):
                for metric in METRICS:
                    if condition.endswith("_prior"):
                        reference = prior_reference.loc[prior_reference.seed.eq(seed) & prior_reference.task.eq("Taking") & prior_reference.country.eq(country) & prior_reference.condition.eq(condition.removesuffix("_prior")) & prior_reference.metric.eq(metric)]
                        expected = reference.iloc[0].prior_ll
                    else:
                        reference = previous.loc[previous.seed.eq(seed) & previous.task.eq("Taking") & previous.Country.eq(country) & previous.condition.eq(condition)]
                        expected = reference.iloc[0][metric]
                    error = abs(frame[metric].mean() - expected)
                    if len(reference) != 1 or error > 1e-10:
                        raise ValueError("Full-sample adapter/prior score failed aggregate reproduction")
                    validation.append(dict(family="adapter_or_prior", seed=seed, condition=condition, Country=country, metric=metric, absolute_error=error))
            parts.append(subset_people(full_scores, ids).assign(seed=seed, task="Taking", condition=condition))
    participants = pd.concat(parts, ignore_index=True)
    if not participants.groupby(["seed", "condition", "Country"]).size().eq(1329).all():
        raise ValueError("Model evaluation samples differ")
    metric_summary = participants.groupby(["seed", "task", "condition", "Country"])[METRICS].mean().reset_index()
    save(metric_summary.loc[~metric_summary.condition.str.endswith("_prior")], "seed_metrics.csv")
    save(pd.DataFrame(shift_rows), "figure_inputs/prediction_shifts_by_seed.csv")
    values = score_contrasts(participants)
    intervals, country_intervals = bootstrap_outcomes(values, args.resamples, args.bootstrap_seed)
    save(intervals, "bootstrap_intervals.csv")
    save(country_intervals, "country_contrasts.csv")
    figure_intervals = intervals.loc[intervals.metric.eq("conditional_ll") & intervals.method.eq("shared_participant_and_seed")]
    gains = figure_intervals.loc[figure_intervals.contrast.str.endswith("_gain_vs_base")].copy()
    gains["condition"] = gains.contrast.str.removesuffix("_gain_vs_base")
    gains = gains.rename(columns={"Country": "country", "mean": "value"})
    save(gains[["task", "country", "condition", "value", "ci_low", "ci_high"]], "prediction_gains.csv")
    controls = figure_intervals.loc[figure_intervals.contrast.str.endswith("_prior_minus_lora")].copy()
    controls["condition"] = controls.contrast.str.removesuffix("_prior_minus_lora")
    controls = controls.rename(columns={"Country": "country", "mean": "value"})
    controls["value"] = -controls.value
    controls["ci_low"], controls["ci_high"] = -controls.ci_high.copy(), -controls.ci_low.copy()
    save(controls[["task", "country", "condition", "value", "ci_low", "ci_high"]], "response_adjustment_gains.csv")
    seed_contrasts = values.groupby(["task", "Country", "metric", "contrast", "seed"]).value.mean().reset_index()
    save(seed_contrasts, "seed_contrasts.csv")
    save(intervals.loc[intervals.contrast.str.contains("_prior_")], "response_adjustment/prior_paired_intervals.csv")
    prior_metrics = []
    for (seed, task, country), frame in metric_summary.groupby(["seed", "task", "Country"]):
        indexed = frame.set_index("condition")
        for condition in CONDITIONS:
            for metric in METRICS:
                prior, lora, base = (indexed.loc[name, metric] for name in (condition + "_prior", condition, "base"))
                prior_metrics.append(dict(seed=seed, task=task, country=country, condition=condition, metric=metric,
                                          prior_ll=prior, base_ll=base, lora_ll=lora, prior_gain=prior-base,
                                          lora_gain=lora-base, prior_minus_lora=prior-lora))
    save(pd.DataFrame(prior_metrics), "response_adjustment/prior_metrics_by_seed.csv")
    save(pd.DataFrame(validation), "unbalanced_reproduction_validation.csv")
    for name in ("analyze_balanced_taking.py", "analyze_dictator_corrected_baselines.py", "analyze_dictator_corrected_adapters.py", "audit_dictator_a1.py", "audit_dictator_response_prior.py", "dictator_item_holdout_transfer.py"):
        inputs.add(ROOT / "scripts" / name)
    save(pd.DataFrame([dict(path=str(p.resolve()), sha256=sha(p), bytes=p.stat().st_size) for p in sorted(inputs)]), "input_provenance.csv")
    metadata = dict(
        status="complete_validated_balanced_taking_reanalysis", task="Taking; ancillary frozen H1 Giving contrasts",
        source_behavior_sha256=SOURCE_SHA, selection_seed=args.selection_seed,
        selection_algorithm="Sort unique Dataset/Nation/SubNo keys; NumPy default_rng; iterate Dataset/Nation strata in sorted order; sample without replacement only when available count exceeds target; keep whole participants.",
        numpy_version=np.__version__, pandas_version=pd.__version__,
        selection_manifest=str(args.manifest.resolve()), selection_manifest_sha256=sha(args.manifest),
        participants_per_country=1329, wave_counts_per_country={"Wave 1": 529, "Wave 2": 800},
        sampling="US: select 529/546 in Wave 1 and 800/825 in Wave 2; China: retain all 529/800. Same people for every saved model, seed and human comparison.",
        seeds=[0,1,2], bootstrap_seed=args.bootstrap_seed, resamples=args.resamples,
        uncertainty="95% percentile bootstrap. Resample participants independently by country and wave; share weights across models and seeds; resample the three observed fitted seeds together across countries. Frozen controls use participant-only intervals. Selection, model fits, response adjustment offsets, target groups and folds are held fixed. Only three seeds limit training-run inference.",
        primary_metric="conditional_ll (log probability normalized across 21 legal token responses); NLL is its negative",
        response_adjustment="Reuse the previously fitted fixed Giving-only response offsets (ridge .001), applied unchanged to Taking; no model training or response-offset refitting.",
        population_predictions="Model predictions do not depend on evaluation participant; recomputed means equal saved preselection means. Human means and scores must use this manifest.",
        h1_giving="Frozen H1 Giving comparisons preserve the original three heldout participant splits (data_cache/dictator_centaur70b/seed*/participant_split_manifest.csv), with shared weights for overlapping test participants, fixed splits and fixed wave proportions. Original published means are reproduced before exporting intervals; Giving is not subsampled.",
        schemas="Taking-only replacements for results/giving_taking/{seed_metrics.csv,bootstrap_intervals.csv,figure_inputs/*,frozen/metrics_summary.csv,response_adjustment/prior_metrics_by_seed.csv,response_adjustment/prior_paired_intervals.csv}; original unbalanced outputs retained. *_score contrasts are log likelihood, not NLL. country_contrasts subtract China from US for the named outcome; own_minus_other names vary the matching adapter with test country.",
        exploratory=True, models_retrained=False, private_data="Selected participant keys are saved only in ignored build/. No individual observations or scores are included in aggregate outputs.",
        command=f"python scripts/analyze_balanced_taking.py --workspace {args.workspace} --output {output} --manifest {args.manifest} --selection-seed {args.selection_seed} --bootstrap-seed {args.bootstrap_seed} --resamples {args.resamples}",
    )
    (output / "analysis_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    shown = intervals.loc[intervals.metric.eq("conditional_ll") & intervals.method.eq("shared_participant_and_seed") & ~intervals.contrast.str.endswith("_score")]
    print(shown[["Country", "contrast", "mean", "ci_low", "ci_high"]].to_string(index=False))
    print(f"Validated {len(validation)} unchanged full-sample scores; wrote balanced aggregate results to {output}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, default=ROOT.parent)
    parser.add_argument("--output", type=Path, default=ROOT / "results/giving_taking/balanced_taking_20261029")
    parser.add_argument("--manifest", type=Path, default=ROOT / "build/balanced_taking_20261029/selected_participants.csv")
    parser.add_argument("--selection-seed", type=int, default=20261029)
    parser.add_argument("--bootstrap-seed", type=int, default=20261029)
    parser.add_argument("--resamples", type=int, default=4000)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
