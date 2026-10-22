#!/usr/bin/env python3
"""Collect verified culture-extension results without treating partial folds as complete.

Only verified 25-update production fits enter analysis. Probabilities are mapped
to the semantic response in each A/B order before the two orders are averaged.
No confidence intervals or significance tests are computed by this collector.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RIDGE = 0.001


def require(value, message):
    if not value:
        raise ValueError(message)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def csv_rows(path):
    with Path(path).open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def log_sigmoid(value):
    return -max(-value, 0.0) - math.log1p(math.exp(-abs(value)))


def sigmoid(value):
    return math.exp(log_sigmoid(value))


def log_mean_exp(a, b):
    high = max(a, b)
    return high + math.log(math.exp(a - high) + math.exp(b - high)) - math.log(2)


def semantic_logit(row):
    require(row["order"] in {"ab", "ba"}, "Unknown answer order")
    a, b = float(row["logp_a"]), float(row["logp_b"])
    require(math.isfinite(a) and math.isfinite(b), "Nonfinite candidate log probability")
    return a - b if row["order"] == "ab" else b - a


def average_orders(rows, offset=0.0):
    """Return one semantic distribution; retain the original valid-choice mass.

    For the response adjustment, valid_mass remains a frozen-model diagnostic:
    the adjustment defines conditional A/B probabilities, not a new vocabulary
    distribution.
    """
    require(len(rows) == 2 and {row["order"] for row in rows} == {"ab", "ba"},
            "Exactly one ab and one ba row are required")
    ordered = sorted(rows, key=lambda row: row["order"])
    logits = [semantic_logit(row) + offset for row in ordered]
    lp = log_mean_exp(*(log_sigmoid(value) for value in logits))
    ln = log_mean_exp(*(log_sigmoid(-value) for value in logits))
    return {
        "p_positive": math.exp(lp), "log_p_positive": lp, "log_p_negative": ln,
        "order_difference_ab_minus_ba": sigmoid(logits[0]) - sigmoid(logits[1]),
        "original_valid_mass": sum(float(row["valid_mass"]) for row in rows) / 2,
        "original_log_valid_mass": sum(float(row["log_valid_mass"]) for row in rows) / 2,
    }


def fit_semantic_offset(targets, frozen_rows, ridge=RIDGE):
    """Fit one shared semantic log-odds shift to equal-item/equal-order targets.

    Inherit the existing centered-logit penalty: offsets (+delta/2,-delta/2)
    incur .5 * ridge * sum(offsets**2) = ridge * delta**2 / 4.
    Strictly convex scalar optimization uses bisection, avoiding SciPy.
    """
    require(ridge > 0, "A positive ridge is required for a finite unique fit")
    by_item = defaultdict(list)
    for row in frozen_rows:
        by_item[str(row["item_id"])].append(row)
    terms = []
    require(bool(targets), "No training targets")
    ids = [str(row["item_id"]) for row in targets]
    require(len(ids) == len(set(ids)), "Duplicate training target item")
    counts = {int(row["n_responses"]) for row in targets}
    require(len(counts) == 1 and min(counts) > 0, "Unequal training item budgets")
    for target in targets:
        rows = by_item[str(target["item_id"])]
        average_orders(rows)  # validates a complete, unique order pair
        probability = float(target["p_positive"])
        require(0 <= probability <= 1, "Invalid empirical training target")
        terms.extend((semantic_logit(row), probability) for row in rows)

    def gradient(delta):
        return sum(sigmoid(value + delta) - y for value, y in terms) / len(terms) + ridge * delta / 2

    def objective(delta):
        ce = -sum(y * log_sigmoid(value + delta) + (1-y) * log_sigmoid(-value-delta)
                  for value, y in terms) / len(terms)
        return ce + ridge * delta * delta / 4

    lo, hi = -1.0, 1.0
    while gradient(lo) > 0:
        lo *= 2
    while gradient(hi) < 0:
        hi *= 2
    for _ in range(100):
        middle = (lo + hi) / 2
        if gradient(middle) < 0:
            lo = middle
        else:
            hi = middle
    delta = (lo + hi) / 2
    require(abs(gradient(delta)) < 1e-10, "Offset optimizer did not converge")
    return delta, {
        "delta_semantic_log_odds": delta, "ridge": ridge,
        "n_training_items": len(targets), "n_training_responses": sum(int(row["n_responses"]) for row in targets),
        "order_weight": 0.5, "objective_at_zero": objective(0),
        "objective_at_solution": objective(delta), "gradient_at_solution": gradient(delta),
        "penalty": "ridge * delta^2 / 4 (centered two-choice offsets)",
    }


def verified_output(output, stage, manifest_hash=None):
    marker_path = output / "verified_complete.json"
    require(marker_path.exists(), "Missing verified_complete.json")
    marker = json.loads(marker_path.read_text())
    require(marker.get("verified") is True and marker.get("stage") == stage,
            f"Output is not verified stage={stage}; smoke is excluded")
    if stage == "train":
        require(marker.get("complete_production_training") is True and marker.get("optimizer_steps") == 25,
                "Output is not verified 25-update production training")
    else:
        require(marker.get("complete_baseline_evaluation") is True, "Baseline evaluation not complete")
    if manifest_hash:
        require(marker.get("run_manifest_sha256") == manifest_hash, "Verified output uses a different run manifest")
    require(isinstance(marker.get("verified_files_sha256"), dict), "Missing verified-file hashes")
    return marker


def verified_csv(output, relative_path, marker):
    path = output / relative_path
    expected = marker["verified_files_sha256"].get(relative_path)
    require(expected and path.exists() and sha256(path) == expected,
            f"Missing or changed verified probabilities: {relative_path}")
    return csv_rows(path)


def group_candidates(rows, sources, items):
    expected = {(source, str(item), order) for source in sources for item in items for order in ("ab", "ba")}
    keys = [(row["source"], str(row["item_id"]), row["order"]) for row in rows]
    require(len(keys) == len(set(keys)) and set(keys) == expected,
            "Candidate table has duplicated, missing, or unexpected source/item/order rows")
    result = defaultdict(list)
    for row in rows:
        calculated = sigmoid(semantic_logit(row))
        require(abs(float(row["p_positive"]) - calculated) < 1e-6, "Recorded semantic probability disagrees with A/B logits")
        require(row["stratum"] == items[str(row["item_id"])]["stratum"], "Item stratum differs from frozen input")
        result[(row["source"], str(row["item_id"]))].append(row)
    return result


def check_file(base, manifest, name):
    path = (base / manifest[f"{name}_path"]).resolve()
    require(path.exists() and sha256(path) == manifest[f"{name}_sha256"], f"Frozen input mismatch: {name}")
    return path


def coverage(available_folds, expected_folds):
    require(set(available_folds) <= set(expected_folds), "Unexpected folds")
    return "complete" if set(available_folds) == set(expected_folds) else "partial" if available_folds else "missing"


def frozen_reference_difference(left, right, items):
    """Check each response order separately: averaging must not hide drift."""
    differences = []
    for item in items:
        key = ("frozen_centaur", item)
        a = {row["order"]: row for row in left[key]}
        b = {row["order"]: row for row in right[key]}
        require(set(a) == set(b) == {"ab", "ba"}, "Frozen reference orders differ")
        for order in ("ab", "ba"):
            for field in ("p_positive", "valid_mass"):
                differences.append(abs(float(a[order][field]) - float(b[order][field])))
    return max(differences)


def collect(run_index, repo_root, outputs_root, baseline_root=None):
    index = json.loads(run_index.read_text())
    require(len(index["runs"]) == index["count"], "Run index count mismatch")
    statuses, predictions, all_predictions, prior_fits = [], [], [], []
    datasets, splits, expected = {}, {}, defaultdict(set)
    baseline_cache = {}
    for entry in index["runs"]:
        expected[(entry["dataset"], entry["seed"], entry["condition"])].add(entry["fold"])
    for entry in index["runs"]:
        meta = {key: entry[key] for key in ("dataset", "seed", "fold", "condition")}
        path = repo_root / entry["manifest"]
        require(sha256(path) == entry["manifest_sha256"], "Frozen run-index manifest hash mismatch")
        manifest = json.loads(path.read_text())
        require(all(manifest[key] == value for key, value in meta.items()), "Run-index metadata mismatch")
        require(not set(manifest["heldout_item_ids"]) & set(manifest["train_item_ids"]), "Training and held-out item overlap")
        item_path = check_file(path.parent, manifest, "items")
        response_path = check_file(path.parent, manifest, "responses")
        split_path = check_file(path.parent, manifest, "participant_split")
        target_path = check_file(path.parent, manifest, "training_targets")
        dataset = entry["dataset"]
        if dataset not in datasets:
            items = {str(item["item_id"]): item for item in json.loads(item_path.read_text())}
            responses = pd.read_csv(response_path, dtype={"participant_id": str, "item_id": str})
            datasets[dataset] = (items, responses)
        items, _ = datasets[dataset]
        split_key = (dataset, entry["seed"])
        split_rows = csv_rows(split_path)
        if split_key in splits:
            require(splits[split_key] == split_rows, "Participant splits differ across folds or conditions")
        splits[split_key] = split_rows
        output = outputs_root / dataset / f"seed{entry['seed']}" / f"fold{entry['fold']}" / entry["condition"]
        status = {**meta, "status": "missing", "reason": "No verified production output", "output_dir": str(output),
                  "run_manifest_sha256": entry["manifest_sha256"],
                  "items_sha256": manifest["items_sha256"], "responses_sha256": manifest["responses_sha256"],
                  "participant_split_sha256": manifest["participant_split_sha256"],
                  "training_targets_sha256": manifest["training_targets_sha256"]}
        try:
            marker = verified_output(output, "train", entry["manifest_sha256"])
            for key, value in meta.items():
                require(marker.get(key) == value, f"Verified output metadata mismatch: {key}")
            rows = verified_csv(output, "evaluation/candidate_probabilities.csv", marker)
            adapted = group_candidates(rows, {"finetuned_centaur"}, items)
            local_frozen_rows = verified_csv(output, "evaluation/frozen_centaur_probabilities.csv", marker)
            local_frozen = group_candidates(local_frozen_rows, {"frozen_centaur"}, items)
            status.update(
                verified_marker_sha256=sha256(output / "verified_complete.json"),
                candidate_probabilities_sha256=marker["verified_files_sha256"]["evaluation/candidate_probabilities.csv"],
                frozen_candidate_probabilities_sha256=marker["verified_files_sha256"]["evaluation/frozen_centaur_probabilities.csv"],
            )
            # The per-run frozen reference is verified as unchanged by training.
            sources = {"finetuned_centaur": adapted, "frozen_centaur": local_frozen}
            baseline_warning = ""
            if dataset not in baseline_cache:
                baseline_output = (baseline_root or outputs_root / "baseline") / dataset
                try:
                    bm = verified_output(baseline_output, "baseline")
                    require(bm.get("dataset") == dataset, "Baseline dataset mismatch")
                    require(bm.get("run_manifest_sha256") in {row["manifest_sha256"] for row in index["runs"] if row["dataset"] == dataset},
                            "Baseline is not linked to this frozen run index")
                    br = verified_csv(baseline_output, "candidate_probabilities.csv", bm)
                    baseline_cache[dataset] = (
                        group_candidates(br, {"frozen_centaur", "base_llama"}, items), "",
                        {"baseline_marker_sha256": sha256(baseline_output / "verified_complete.json"),
                         "baseline_candidate_probabilities_sha256": bm["verified_files_sha256"]["candidate_probabilities.csv"]},
                    )
                except (ValueError, FileNotFoundError, KeyError) as error:
                    baseline_cache[dataset] = (None, str(error), {})
            baseline, baseline_warning, baseline_provenance = baseline_cache[dataset]
            status.update(baseline_provenance)
            if baseline is not None:
                sources["base_llama"] = baseline
                # Numerical equality is checked at the probability level. Keep
                # the exact local frozen reference for adaptation contrasts.
                largest = frozen_reference_difference(local_frozen, baseline, items)
                status["frozen_baseline_max_probability_difference"] = largest
                require(largest <= 1e-5, "Standalone and per-run frozen Centaur predictions disagree")
            targets = csv_rows(target_path)
            require({row["item_id"] for row in targets} == set(manifest["train_item_ids"]), "Training target item mismatch")
            delta, diagnostics = fit_semantic_offset(targets, local_frozen_rows)
            run_predictions = []
            for item in items:
                item = str(item)
                item_scope = "unseen_items" if item in manifest["heldout_item_ids"] else "seen_items"
                for source, groups in sources.items():
                    run_predictions.append({**meta, "source": source, "item_id": item,
                                        "evaluation_item_scope": item_scope, "is_heldout_item": item_scope == "unseen_items",
                                        "stratum": items[item]["stratum"], **average_orders(groups[(source, item)])})
                run_predictions.append({**meta, "source": "response_adjustment", "item_id": item,
                                    "evaluation_item_scope": item_scope, "is_heldout_item": item_scope == "unseen_items",
                                    "stratum": items[item]["stratum"],
                                    **average_orders(local_frozen[("frozen_centaur", item)], delta)})
            all_predictions.extend(run_predictions)
            predictions.extend(row for row in run_predictions if row["is_heldout_item"])
            prior_fits.append({**meta, **diagnostics})
            status.update(status="verified_production", reason="", baseline_warning=baseline_warning)
        except (ValueError, FileNotFoundError, KeyError) as error:
            status.update(status="unverified_or_invalid" if output.exists() else "missing", reason=str(error))
        statuses.append(status)
    return statuses, predictions, all_predictions, prior_fits, datasets, splits, expected


def participant_and_seed_metrics(predictions, datasets, splits, expected, evaluation_item_scope="unseen_items"):
    if not predictions:
        return [], []
    frame = pd.DataFrame(predictions)
    people = []
    group_keys = ["dataset", "seed", "condition", "source"]
    for (dataset, seed, condition, source), group in frame.groupby(group_keys, sort=True):
        require(not group["item_id"].duplicated().any(), "Held-out items appear in multiple folds")
        split = splits[(dataset, int(seed))]
        test = {row["participant_id"]: row["country"] for row in split if row["split"] == "test"}
        require(set(test.values()) == {"us", "china"}, "Test split must contain both countries")
        responses = datasets[dataset][1]
        responses = responses[responses["participant_id"].isin(test)]
        require(all(test[row.participant_id] == row.country for row in responses.itertuples()), "Test country mismatch")
        joined = responses.merge(group.drop(columns=["dataset", "seed", "condition", "source"]),
                                 on="item_id", how="inner", validate="many_to_one")
        joined["conditional_nll"] = -(joined["response_positive"] * joined["log_p_positive"] +
                                      (1-joined["response_positive"]) * joined["log_p_negative"])
        strata = {name: part for name, part in joined.groupby("stratum")}
        strata["all"] = joined
        for stratum, part in strata.items():
            fold_list = sorted(group["fold"].unique().tolist())
            state = coverage(fold_list, expected[(dataset, int(seed), condition)])
            for (participant, country), individual in part.groupby(["participant_id", "country"]):
                # Average repeats within item, then items within person.
                item_means = individual.groupby("item_id")[
                    ["conditional_nll", "response_positive", "p_positive", "original_log_valid_mass",
                     "original_valid_mass", "order_difference_ab_minus_ba"]].mean()
                people.append({"dataset": dataset, "seed": int(seed), "condition": condition, "source": source,
                               "evaluation_item_scope": evaluation_item_scope,
                               "stratum": stratum, "participant_id": participant, "country": country,
                               "coverage": state, "completed_folds": ",".join(map(str, fold_list)),
                               "expected_fold_count": len(expected[(dataset, int(seed), condition)]),
                               "item_ids": ",".join(sorted(item_means.index)), "n_items": len(item_means),
                               "n_responses": len(individual),
                               "conditional_nll": item_means["conditional_nll"].mean(),
                               "human_p_positive": item_means["response_positive"].mean(),
                               "predicted_p_positive": item_means["p_positive"].mean(),
                               "original_valid_mass": item_means["original_valid_mass"].mean(),
                               "original_log_valid_mass": item_means["original_log_valid_mass"].mean(),
                               "order_difference_ab_minus_ba": item_means["order_difference_ab_minus_ba"].mean()})
    table = pd.DataFrame(people)
    summaries = []
    keys = ["dataset", "seed", "condition", "source", "evaluation_item_scope", "stratum", "coverage", "completed_folds",
            "expected_fold_count", "item_ids", "n_items"]
    metrics = ["conditional_nll", "human_p_positive", "predicted_p_positive", "original_valid_mass", "original_log_valid_mass", "order_difference_ab_minus_ba"]
    for key, part in table.groupby(keys, dropna=False, sort=True):
        meta = dict(zip(keys, key))
        country_means = part.groupby("country")[metrics].mean()
        for country, values in country_means.iterrows():
            summaries.append({**meta, "country": country, "n_participants": int((part["country"] == country).sum()), **values.to_dict()})
        require(set(country_means.index) == {"us", "china"}, "Country-balanced summary lacks a country")
        summaries.append({**meta, "country": "balanced_us_china", "n_participants": len(part),
                          **country_means.mean().to_dict()})
    return people, summaries


def paired_contrasts(predictions, datasets, splits, expected):
    """Compute contrasts only on identical completed folds/items/test people."""
    if not predictions:
        return [], []
    frame = pd.DataFrame(predictions)
    specifications = []
    for dataset, seed in sorted(set(zip(frame["dataset"], frame["seed"]))):
        base = frame[(frame.dataset == dataset) & (frame.seed == seed)]
        if {"us", "china"} <= set(base.condition):
            specifications.append((dataset, seed, "us_training_minus_china_training", "us", "finetuned_centaur", "china", "finetuned_centaur"))
        for condition in sorted(set(base.condition)):
            for comparator in ("frozen_centaur", "response_adjustment", "base_llama"):
                specifications.append((dataset, seed, f"{condition}_finetuned_minus_{comparator}", condition, "finetuned_centaur", condition, comparator))
    summaries, people = [], []
    for dataset, seed, contrast, condition_a, source_a, condition_b, source_b in specifications:
        base = frame[(frame.dataset == dataset) & (frame.seed == seed)]
        a = base[(base.condition == condition_a) & (base.source == source_a)]
        b = base[(base.condition == condition_b) & (base.source == source_b)]
        paired = a.merge(b, on=["item_id", "fold", "stratum"], suffixes=("_a", "_b"), validate="one_to_one")
        if paired.empty:
            continue
        folds = set(paired.fold)
        state = coverage(folds, expected[(dataset, int(seed), condition_a)])
        response = datasets[dataset][1]
        test = {row["participant_id"] for row in splits[(dataset, int(seed))] if row["split"] == "test"}
        joined = response[response.participant_id.isin(test)].merge(paired, on="item_id", validate="many_to_one")
        joined["nll_difference"] = -(joined.response_positive * (joined.log_p_positive_a - joined.log_p_positive_b) +
                                      (1-joined.response_positive) * (joined.log_p_negative_a - joined.log_p_negative_b))
        strata = {name: part for name, part in joined.groupby("stratum")}
        strata["all"] = joined
        for stratum, part in strata.items():
            meta = {"dataset": dataset, "seed": int(seed), "contrast": contrast, "stratum": stratum,
                    "evaluation_item_scope": "unseen_items",
                    "coverage": state, "paired_folds": ",".join(map(str, sorted(folds))),
                    "item_ids": ",".join(sorted(part.item_id.unique())), "n_items": part.item_id.nunique()}
            pairs = part.drop_duplicates("item_id")
            shift = (pairs.p_positive_a - pairs.p_positive_b).mean()
            per_person = part.groupby(["participant_id", "country", "item_id"]).nll_difference.mean().groupby(level=[0, 1]).mean()
            for (person, country), difference in per_person.items():
                people.append({**meta, "participant_id": person, "country": country,
                               "conditional_nll_difference": difference, "prediction_shift_pp": 100 * shift})
            country_means = per_person.groupby(level=1).mean()
            summaries.append({**meta, "country": "balanced_us_china", "prediction_shift_pp": 100 * shift,
                              "conditional_nll_difference": country_means.mean(), "n_participants": len(per_person),
                              "interpretation": "Positive shift favors the semantic response; negative NLL difference favors the first model."})
    return summaries, people


def seen_item_secondary_metrics(all_predictions, datasets, splits, expected):
    """Preserve seen-item results per fold; never treat repeated folds as new data."""
    groups = defaultdict(list)
    for row in all_predictions:
        if row["evaluation_item_scope"] == "seen_items":
            groups[(row["dataset"], row["seed"], row["condition"], row["fold"])].append(row)
    people, summaries = [], []
    for (_, _, _, fold), rows in sorted(groups.items()):
        p, s = participant_and_seed_metrics(rows, datasets, splits, expected, "seen_items")
        for table, output in ((p, people), (s, summaries)):
            output.extend({**row, "fold": fold, "coverage": "per_fold_secondary",
                           "analysis_scope": "heldout participants on training items; retain separate fold estimates"}
                          for row in table)
    return people, summaries


def frozen_country_audit(datasets, outputs_root, index, baseline_root=None):
    """Descriptive frozen audit of every retained country, independent of fitting."""
    people, summaries, statuses = [], [], []
    for dataset, (items, responses) in sorted(datasets.items()):
        output = (baseline_root or outputs_root / "baseline") / dataset
        status = {"dataset": dataset, "status": "missing", "output_dir": str(output)}
        try:
            marker = verified_output(output, "baseline")
            require(marker.get("dataset") == dataset, "Baseline dataset mismatch")
            require(marker.get("run_manifest_sha256") in {row["manifest_sha256"] for row in index["runs"] if row["dataset"] == dataset},
                    "Baseline is not linked to this frozen run index")
            candidates = group_candidates(verified_csv(output, "candidate_probabilities.csv", marker),
                                          {"frozen_centaur", "base_llama"}, items)
            for source in ("frozen_centaur", "base_llama"):
                item_predictions = pd.DataFrame([
                    {"item_id": item, "stratum": items[item]["stratum"], **average_orders(candidates[(source, item)])}
                    for item in items
                ])
                joined = responses.merge(item_predictions, on="item_id", validate="many_to_one")
                joined["conditional_nll"] = -(joined.response_positive * joined.log_p_positive +
                                              (1-joined.response_positive) * joined.log_p_negative)
                strata = {key: part for key, part in joined.groupby("stratum")}
                strata["all"] = joined
                for stratum, part in strata.items():
                    rows = []
                    for (participant, country), group in part.groupby(["participant_id", "country"]):
                        means = group.groupby("item_id")[["response_positive", "p_positive", "conditional_nll",
                                                           "original_valid_mass", "order_difference_ab_minus_ba"]].mean()
                        row = {"dataset": dataset, "source": source, "stratum": stratum,
                               "sample_scope": "all_retained_participants_frozen_audit", "evaluation_item_scope": "all_items_frozen",
                               "participant_id": participant, "country": country, "n_items": len(means),
                               "human_p_positive": means.response_positive.mean(), "predicted_p_positive": means.p_positive.mean(),
                               "conditional_nll": means.conditional_nll.mean(), "original_valid_mass": means.original_valid_mass.mean(),
                               "order_difference_ab_minus_ba": means.order_difference_ab_minus_ba.mean()}
                        people.append(row)
                        rows.append(row)
                    table = pd.DataFrame(rows)
                    for country, group in table.groupby("country"):
                        summaries.append({"dataset": dataset, "source": source, "stratum": stratum, "country": country,
                                          "sample_scope": "all_retained_participants_frozen_audit", "evaluation_item_scope": "all_items_frozen",
                                          "n_participants": len(group), "n_items": int(group.n_items.iloc[0]),
                                          **group[["human_p_positive", "predicted_p_positive", "conditional_nll", "original_valid_mass",
                                                   "order_difference_ab_minus_ba"]].mean().to_dict()})
            status.update(status="verified_baseline", verified_marker_sha256=sha256(output / "verified_complete.json"),
                          candidate_probabilities_sha256=marker["verified_files_sha256"]["candidate_probabilities.csv"],
                          run_manifest_sha256=marker["run_manifest_sha256"], countries=sorted(responses.country.unique().tolist()))
        except (ValueError, FileNotFoundError, KeyError) as error:
            status.update(status="unverified_or_invalid" if output.exists() else "missing", reason=str(error))
        statuses.append(status)
    return people, summaries, statuses


def write_rows(path, rows):
    if rows:
        pd.DataFrame(rows).to_csv(path, index=False)
    else:
        path.write_text("")


def run(run_index, repo_root, outputs_root, analysis_dir, baseline_root=None):
    statuses, predictions, all_predictions, priors, datasets, splits, expected = collect(run_index, repo_root, outputs_root, baseline_root)
    people, seeds = participant_and_seed_metrics(predictions, datasets, splits, expected)
    contrasts, paired_people = paired_contrasts(predictions, datasets, splits, expected)
    seen_people, seen_folds = seen_item_secondary_metrics(all_predictions, datasets, splits, expected)
    frozen_people, frozen_countries, frozen_statuses = frozen_country_audit(datasets, outputs_root, json.loads(run_index.read_text()), baseline_root)
    analysis_dir.mkdir(parents=True, exist_ok=True)
    for filename, rows in {"run_status.csv": statuses, "item_predictions.csv": predictions,
                           "all_item_predictions_per_fold.csv": all_predictions,
                           "response_adjustment_fits.csv": priors, "participant_metrics.csv": people,
                           "seed_metrics.csv": seeds, "contrasts.csv": contrasts,
                           "paired_participant_contrasts.csv": paired_people,
                           "seen_item_participant_metrics_per_fold.csv": seen_people,
                           "seen_item_metrics_per_fold.csv": seen_folds,
                           "frozen_baseline_participant_metrics.csv": frozen_people,
                           "frozen_baseline_country_metrics.csv": frozen_countries,
                           "frozen_baseline_status.csv": frozen_statuses}.items():
        write_rows(analysis_dir / filename, rows)
    finished = sum(row["status"] == "verified_production" for row in statuses)
    grouped = []
    for (dataset, seed, condition), folds in sorted(expected.items()):
        present = {row["fold"] for row in statuses if row["dataset"] == dataset and row["seed"] == seed
                   and row["condition"] == condition and row["status"] == "verified_production"}
        grouped.append({"dataset": dataset, "seed": seed, "condition": condition,
                        "coverage": coverage(present, folds), "completed_folds": sorted(present),
                        "expected_folds": sorted(folds)})
    baselines_available = sorted(row["dataset"] for row in frozen_statuses if row["status"] == "verified_baseline")
    all_baselines = set(baselines_available) == set(datasets)
    report = {
        "analysis_status": "complete" if finished == len(statuses) and all_baselines else "partial" if finished else "no_verified_production_results",
        "expected_runs": len(statuses), "verified_production_runs": finished,
        "datasets_with_verified_standalone_baselines": baselines_available,
        "run_index_sha256": sha256(run_index), "analysis_script_sha256": sha256(Path(__file__)),
        "provenance": {"run_index": str(run_index.resolve()), "outputs_root": str(outputs_root.resolve()),
                       "baseline_root": str((baseline_root or outputs_root / "baseline").resolve()),
                       "per_run_hashes": "run_status.csv", "verified_output_policy": "Stage train, 25 updates, matching frozen manifest; consumed probability CSV hashes rechecked against validation marker."},
        "groups": grouped, "contrasts": contrasts,
        "frozen_baseline_status": frozen_statuses,
        "methods": {
            "evaluation": "Test participants and held-out items only. Fold predictions stitched once per canonical item within seed/condition.",
            "seen_items": "Secondary seen-item performance on held-out participants is retained per fold in separate files; never pooled with unseen items or counted as independent folds.",
            "frozen_audit": "Separate descriptive audit uses all retained participants and countries, including all 11 Anllo countries. This sample scope differs from held-out adaptation scoring.",
            "order": "Map A/B logits to category or 10-point response, then average conditional probabilities across orders equally.",
            "aggregation": "Average repetitions within item, items within participant, participants within country; US and China each weight 1/2.",
            "adjustment": "One shared semantic log-odds offset fitted only to exact training targets; equal item/order weights; penalty 0.001*delta^2/4.",
            "partial": "Partial groups use their available held-out items; contrasts use paired completed folds only. Never compare unmatched partial means.",
            "uncertainty": "Descriptive seed and participant values only. No inferential claims, independent-fold assumptions, or confidence intervals.",
            "source_diagnostics": "original_valid_mass and original_log_valid_mass refer to the vocabulary distribution before conditional normalization or scalar adjustment.",
        },
    }
    (analysis_dir / "status_and_contrasts.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=ROOT)
    parser.add_argument("--run-index", type=Path, default=ROOT / "data_cache/culture_extensions_20261021/runs/run_index.json")
    parser.add_argument("--outputs-root", type=Path, default=ROOT / "outputs/culture_extensions_20261021")
    parser.add_argument("--analysis-dir", type=Path, default=ROOT / "outputs/culture_extensions_20261021/analysis_precision_aligned")
    parser.add_argument("--baseline-root", type=Path, default=ROOT / "outputs/culture_extensions_20261021/baseline_precision_aligned", help="Explicit separately verified baseline directory; original outputs are preserved")
    args = parser.parse_args()
    report = run(args.run_index, args.repo_root, args.outputs_root, args.analysis_dir, args.baseline_root)
    print(json.dumps({key: report[key] for key in ("analysis_status", "expected_runs", "verified_production_runs")}, indent=2))


if __name__ == "__main__":
    main()
