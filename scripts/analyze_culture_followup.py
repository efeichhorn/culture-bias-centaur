#!/usr/bin/env python3
"""CPU follow-up: clustered uncertainty and training-only answer-position control.

Reuses the verified collector, never trains model weights, and writes only
aggregate data to the public results directory. Participant rows stay in memory.
Intervals are conditional on the fitted models, observed seeds, and fixed items.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

import analyze_culture_extensions as original

ROOT = Path(__file__).resolve().parents[1]
RIDGE = 0.001


def fit_position_control(targets, frozen_rows, ridge=RIDGE):
    """Fit semantic and first-position shifts to equal-item/order training CE.

    z_semantic = z_frozen + semantic + position * (+1 for ab, -1 for ba).
    Penalty ridge*(semantic**2+position**2)/4. In order-specific offsets
    this separates into two strictly convex scalar problems, solved exactly
    by bisection. This matches the scalar control's ridge convention.
    """
    original.require(ridge > 0, "Positive ridge required")
    by_item = defaultdict(list)
    for row in frozen_rows:
        by_item[str(row["item_id"])].append(row)
    ids = [str(t["item_id"]) for t in targets]
    original.require(len(ids) == len(set(ids)) and bool(ids), "Invalid training items")
    original.require(len({int(t["n_responses"]) for t in targets}) == 1,
                     "Unequal training item budgets")
    terms = {"ab": [], "ba": []}
    for target in targets:
        rows = by_item[str(target["item_id"])]
        original.average_orders(rows)
        y = float(target["p_positive"])
        original.require(0 <= y <= 1, "Invalid training probability")
        for row in rows:
            terms[row["order"]].append((original.semantic_logit(row), y))
    offsets = {}
    for order, values in terms.items():
        def grad(delta):
            return np.mean([original.sigmoid(z + delta) - y for z, y in values]) + ridge * delta / 2
        low, high = -1., 1.
        while grad(low) > 0:
            low *= 2
        while grad(high) < 0:
            high *= 2
        for _ in range(100):
            middle = (low + high) / 2
            if grad(middle) < 0:
                low = middle
            else:
                high = middle
        offsets[order] = (low + high) / 2
        original.require(abs(grad(offsets[order])) < 1e-10, "Position fit did not converge")
    semantic = (offsets["ab"] + offsets["ba"]) / 2
    position = (offsets["ab"] - offsets["ba"]) / 2

    def objective(ds, dp):
        loss = []
        for order, values in terms.items():
            delta = ds + (dp if order == "ab" else -dp)
            loss.extend(-y * original.log_sigmoid(z + delta) - (1-y) * original.log_sigmoid(-z-delta)
                        for z, y in values)
        return float(np.mean(loss) + ridge * (ds * ds + dp * dp) / 4)

    return semantic, position, {"delta_semantic": semantic, "delta_position": position,
                               "ridge": ridge, "objective_at_zero": objective(0, 0),
                               "objective_at_solution": objective(semantic, position),
                               "n_training_items": len(targets)}


def adjusted_orders(rows, semantic, position):
    original.average_orders(rows)  # validate exactly one row per order
    modified = []
    for row in rows:
        # Add centered offsets in A/B coordinates, preserving original mass
        # only as a diagnostic; this defines a conditional distribution.
        sem_sign = 1 if row["order"] == "ab" else -1
        delta_ab = sem_sign * semantic + position
        modified.append({**row, "logp_a": float(row["logp_a"]) + delta_ab / 2,
                         "logp_b": float(row["logp_b"]) - delta_ab / 2})
    return original.average_orders(modified)


def validate_training_targets(manifest, manifest_path, responses, split_rows):
    """Independently reconstruct the fitting targets from exact training IDs."""
    targets = original.csv_rows(original.check_file(manifest_path.parent, manifest, "training_targets"))
    train = original.csv_rows(original.check_file(manifest_path.parent, manifest, "training_participants"))
    train_ids = {r["participant_id"] for r in train}
    test_ids = {r["participant_id"] for r in split_rows if r["split"] == "test"}
    original.require(not train_ids & test_ids, "Training/test participant overlap")
    original.require(len(train_ids) == manifest["n_training_participants"], "Training count mismatch")
    original.require(set(manifest["train_item_ids"]).isdisjoint(manifest["heldout_item_ids"]), "Item leakage")
    original.require({str(t["item_id"]) for t in targets} == set(manifest["train_item_ids"]), "Target item mismatch")
    selected = responses[responses.participant_id.isin(train_ids) & responses.item_id.isin(manifest["train_item_ids"])]
    original.require(len(selected) == manifest["n_training_responses"], "Training responses mismatch")
    for target in targets:
        r = selected[selected.item_id == str(target["item_id"])]
        original.require(len(r) == int(target["n_responses"]), "Target response count mismatch")
        original.require(abs(r.response_positive.mean() - float(target["p_positive"])) < 1e-12,
                         "Training target differs from original responses")
    return targets


def position_predictions(index, repo, outputs, datasets, splits):
    predictions, fits, hashes = [], [], []
    for entry in index["runs"]:
        path = repo / entry["manifest"]
        original.require(original.sha256(path) == entry["manifest_sha256"], "Manifest mismatch")
        manifest = json.loads(path.read_text())
        ds, seed, condition, fold = (entry[k] for k in ("dataset", "seed", "condition", "fold"))
        targets = validate_training_targets(manifest, path, datasets[ds][1], splits[(ds, seed)])
        out = outputs / ds / f"seed{seed}" / f"fold{fold}" / condition
        marker = original.verified_output(out, "train", entry["manifest_sha256"])
        frozen = original.verified_csv(out, "evaluation/frozen_centaur_probabilities.csv", marker)
        sem, pos, diagnostics = fit_position_control(targets, frozen)
        groups = original.group_candidates(frozen, {"frozen_centaur"}, datasets[ds][0])
        meta = dict(dataset=ds, seed=seed, condition=condition, fold=fold)
        fits.append({**meta, **diagnostics, "n_training_participants": manifest["n_training_participants"],
                     "n_training_responses": manifest["n_training_responses"]})
        for item in manifest["heldout_item_ids"]:
            predictions.append({**meta, "source": "response_adjustment_position", "item_id": item,
                                "stratum": datasets[ds][0][item]["stratum"], "evaluation_item_scope": "unseen_items",
                                "is_heldout_item": True, **adjusted_orders(groups[("frozen_centaur", item)], sem, pos)})
        hashes.append({**meta, "manifest_sha256": original.sha256(path),
                       "training_targets_sha256": manifest["training_targets_sha256"],
                       "training_participants_sha256": manifest["training_participants_sha256"],
                       "responses_sha256": manifest["responses_sha256"],
                       "participant_split_sha256": manifest["participant_split_sha256"],
                       "verified_marker_sha256": original.sha256(out / "verified_complete.json"),
                       "frozen_probabilities_sha256": marker["verified_files_sha256"]["evaluation/frozen_centaur_probabilities.csv"]})
    return predictions, fits, hashes


class SharedBootstrap:
    """One multinomial multiplicity per person, shared over all seed appearances.

    Stratification is handled by a separate object per dataset/country/scope.
    Each seed mean normalizes within its test set; the estimand averages the
    observed seed means equally. Draws with empty seed subsets are discarded.
    """
    def __init__(self, frame, resamples, rng):
        memberships = frame[["participant_id", "seed"]].drop_duplicates()
        self.ids = sorted(memberships.participant_id.unique())
        self.seeds = sorted(memberships.seed.unique())
        self.positions = {pid: i for i, pid in enumerate(self.ids)}
        self.subsets = {s: [self.positions[p] for p in memberships[memberships.seed == s].participant_id]
                        for s in self.seeds}
        all_weights, rejected = [], 0
        remaining = resamples
        while remaining:
            weights = rng.multinomial(len(self.ids), np.full(len(self.ids), 1 / len(self.ids)), size=remaining)
            valid = np.logical_and.reduce([weights[:, ix].sum(axis=1) > 0 for ix in self.subsets.values()])
            rejected += int((~valid).sum())
            all_weights.append(weights[valid])
            remaining -= int(valid.sum())
        self.weights = np.concatenate(all_weights)
        self.rejected = rejected

    def summarize(self, frame, metric):
        original.require(not frame.duplicated(["seed", "participant_id"]).any(), "Duplicate participant/seed score")
        means, draws, counts = [], [], []
        for seed in self.seeds:
            part = frame[frame.seed == seed].set_index("participant_id")
            expected = {self.ids[i] for i in self.subsets[seed]}
            original.require(set(part.index) == expected, "Unpaired participant coverage")
            ix = [self.positions[p] for p in part.index]
            weights = self.weights[:, ix]
            values = part[metric].to_numpy(float)
            original.require(np.isfinite(values).all(), "Nonfinite participant score")
            means.append(float(values.mean()))
            counts.append(len(values))
            draws.append((weights * values[None, :]).sum(axis=1) / weights.sum(axis=1))
        distribution = np.mean(draws, axis=0)
        original.require(np.isfinite(distribution).all(), "Nonfinite bootstrap draws")
        summary = dict(estimate=float(np.mean(means)), ci_low=float(np.quantile(distribution, .025)),
                       ci_high=float(np.quantile(distribution, .975)),
                       seed_sd=float(np.std(means, ddof=1)) if len(means) > 1 else 0.,
                       seed_min=float(min(means)), seed_max=float(max(means)),
                       n_unique_participants=len(self.ids), n_per_seed=",".join(map(str, counts)),
                       n_seeds=len(means), resamples=len(distribution),
                       uncertainty="shared_participant_fixed_fits_seeds_items")
        return summary, distribution, np.asarray(means)


def difference(a, b):
    """A minus B; inputs are (summary, paired draws, seed means)."""
    draws, seeds = a[1] - b[1], a[2] - b[2]
    out = dict(a[0])
    out.update(estimate=float(a[0]["estimate"]-b[0]["estimate"]),
               ci_low=float(np.quantile(draws, .025)), ci_high=float(np.quantile(draws, .975)),
               seed_sd=float(np.std(seeds, ddof=1)) if len(seeds) > 1 else 0.,
               seed_min=float(seeds.min()), seed_max=float(seeds.max()))
    return out, draws, seeds


def inferential_tables(people, frozen, resamples, random_seed):
    rng = np.random.default_rng(random_seed)
    frame, frozen = pd.DataFrame(people), pd.DataFrame(frozen)
    frame = frame[frame.country.isin(["us", "china"])]
    frozen = frozen[frozen.country.isin(["us", "china"])].copy()
    frozen["seed"] = -1
    tables = {k: [] for k in ("score_intervals", "contrasts", "frozen_intervals", "frozen_contrasts", "human_rates")}
    bootstraps, scores, frozen_scores, human = {}, {}, {}, {}
    for scope, data in [("heldout", frame), ("all_retained", frozen)]:
        for (ds, country), part in data.groupby(["dataset", "country"], sort=True):
            bootstraps[(scope, ds, country)] = SharedBootstrap(part, resamples, rng)
        keys = ["dataset", "stratum", "country"] + (["condition", "source"] if scope == "heldout" else ["source"])
        for key, part in data.groupby(keys, sort=True):
            meta = dict(zip(keys, key))
            boot = bootstraps[(scope, meta["dataset"], meta["country"])]
            score = boot.summarize(part, "conditional_nll")
            dest = "score_intervals" if scope == "heldout" else "frozen_intervals"
            tables[dest].append({**meta, "metric": "conditional_nll", **score[0]})
            (scores if scope == "heldout" else frozen_scores)[key] = score
            human_key = (scope, meta["dataset"], meta["stratum"], meta["country"])
            if human_key not in human:
                summary = boot.summarize(part, "human_p_positive")
                human[human_key] = summary
                tables["human_rates"].append(dict(scope=scope, dataset=meta["dataset"], stratum=meta["stratum"],
                                                  country=meta["country"], **summary[0]))
    for ds, stratum, country in sorted({k[:3] for k in scores}):
        def get(condition, source="finetuned_centaur"):
            return scores[(ds, stratum, country, condition, source)]
        own, other = country, "china" if country == "us" else "us"
        for name, comparator in [("own_vs_other", get(other)), ("own_vs_frozen", get(own, "frozen_centaur")),
                                 ("own_vs_pooled", get("pooled"))]:
            tables["contrasts"].append(dict(dataset=ds, stratum=stratum, country=country, condition=own,
                                            contrast=name, metric="log_score_gain", **difference(comparator, get(own))[0]))
        for condition in ["us", "pooled", "china"]:
            for name, source in [("finetuned_vs_semantic", "response_adjustment"),
                                 ("finetuned_vs_semantic_position", "response_adjustment_position")]:
                tables["contrasts"].append(dict(dataset=ds, stratum=stratum, country=country, condition=condition,
                                                contrast=name, metric="log_score_gain",
                                                **difference(get(condition, source), get(condition))[0]))
    for ds, stratum in sorted({k[:2] for k in frozen_scores}):
        gains = {}
        for country in ["us", "china"]:
            gains[country] = difference(frozen_scores[(ds, stratum, country, "base_llama")],
                                        frozen_scores[(ds, stratum, country, "frozen_centaur")])
            tables["frozen_contrasts"].append(dict(dataset=ds, stratum=stratum, country=country,
                                                  contrast="centaur_gain_vs_llama", metric="log_score_gain", **gains[country][0]))
        for source in ["frozen_centaur", "base_llama"]:
            d = difference(frozen_scores[(ds, stratum, "china", source)], frozen_scores[(ds, stratum, "us", source)])
            d[0]["n_unique_participants"] += frozen_scores[(ds, stratum, "us", source)][0]["n_unique_participants"]
            d[0]["n_per_seed"] = f"us:{frozen_scores[(ds,stratum,'us',source)][0]['n_per_seed']};china:{frozen_scores[(ds,stratum,'china',source)][0]['n_per_seed']}"
            tables["frozen_contrasts"].append(dict(dataset=ds, stratum=stratum, country="china_minus_us",
                                                  contrast=f"{source}_country_nll_gap", metric="conditional_nll_difference", **d[0]))
        d = difference(gains["us"], gains["china"])
        d[0]["n_unique_participants"] += gains["china"][0]["n_unique_participants"]
        d[0]["n_per_seed"] = f"us:{gains['us'][0]['n_per_seed']};china:{gains['china'][0]['n_per_seed']}"
        tables["frozen_contrasts"].append(dict(dataset=ds, stratum=stratum, country="us_minus_china",
                                              contrast="centaur_gain_differential", metric="log_score_gain_difference", **d[0]))
    diagnostics = [{"scope": k[0], "dataset": k[1], "country": k[2], "n_unique_participants": len(b.ids),
                    "rejected_empty_subset_draws": b.rejected} for k, b in bootstraps.items()]
    return tables, diagnostics


def prediction_summaries(seed_metrics, predictions):
    # Identical predictions across evaluated countries: use one country only.
    data = pd.DataFrame(seed_metrics)
    data = data[data.country == "us"]
    rows = []
    for key, group in data.groupby(["dataset", "stratum", "condition", "source"], sort=True):
        meta = dict(zip(["dataset", "stratum", "condition", "source"], key))
        for metric in ["predicted_p_positive", "order_difference_ab_minus_ba"]:
            values = group[metric].to_numpy(float)
            rows.append({**meta, "metric": metric, "estimate": float(values.mean()), "seed_sd": float(values.std(ddof=1)),
                         "seed_min": float(values.min()), "seed_max": float(values.max()), "n_seeds": len(values),
                         "n_items": int(group.n_items.iloc[0])})
    # Absolute, not only signed, order sensitivity avoids cancellation by item.
    pred = pd.DataFrame(predictions)
    strata = [(name, pred[pred.stratum == name]) for name in sorted(pred.stratum.unique())] + [("all", pred)]
    for stratum, part in strata:
        group = part.assign(abs_order_difference=part.order_difference_ab_minus_ba.abs()).groupby(
            ["dataset", "condition", "source", "seed"]).abs_order_difference.mean().reset_index()
        for key, g in group.groupby(["dataset", "condition", "source"]):
            values = g.abs_order_difference.to_numpy(float)
            rows.append(dict(zip(["dataset", "condition", "source"], key)) | dict(
                stratum=stratum, metric="mean_absolute_order_difference", estimate=float(values.mean()),
                seed_sd=float(values.std(ddof=1)), seed_min=float(values.min()), seed_max=float(values.max()), n_seeds=len(values)))
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-repo", type=Path, default=ROOT.parent / "publication-bias-transfer-demo")
    parser.add_argument("--output", type=Path, default=ROOT / "results/cao_anllo/followup_20261029")
    parser.add_argument("--resamples", type=int, default=4000)
    parser.add_argument("--seed", type=int, default=20261029)
    args = parser.parse_args()
    repo = args.source_repo.resolve()
    run_index = repo / "data_cache/culture_extensions_20261021/runs/run_index.json"
    outputs = repo / "outputs/culture_extensions_20261021"
    baseline = outputs / "baseline_precision_aligned"
    statuses, pred, _, _, datasets, splits, expected = original.collect(run_index, repo, outputs, baseline)
    original.require(len(statuses) == 81 and all(r["status"] == "verified_production" for r in statuses),
                     "Follow-up requires all 81 verified original runs")
    index = json.loads(run_index.read_text())
    position, fits, hashes = position_predictions(index, repo, outputs, datasets, splits)
    pred += position
    people, seeds = original.participant_and_seed_metrics(pred, datasets, splits, expected)
    frozen_people, _, frozen_status = original.frozen_country_audit(datasets, outputs, index, baseline)
    original.require(all(s["status"] == "verified_baseline" for s in frozen_status), "Missing frozen baselines")
    tables, diagnostics = inferential_tables(people, frozen_people, args.resamples, args.seed)
    tables["prediction_summary"] = prediction_summaries(seeds, pred)
    tables["seed_metrics"] = seeds
    tables["control_fits"] = fits
    tables["input_hashes"] = hashes
    # Reproduce every existing held-out aggregate before publishing new outputs.
    old = pd.read_csv(ROOT / "results/cao_anllo/seed_metrics.csv")
    new = pd.DataFrame(seeds)
    keys = ["dataset", "seed", "condition", "source", "stratum", "country"]
    joined = old.merge(new, on=keys, suffixes=("_old", "_new"), validate="one_to_one")
    original.require(len(joined) == len(old), "Existing score coverage changed")
    delta = float((joined.conditional_nll_old - joined.conditional_nll_new).abs().max())
    original.require(delta < 1e-10, "Existing NLL failed reproduction")
    args.output.mkdir(parents=True, exist_ok=True)
    for name, rows in tables.items():
        pd.DataFrame(rows).to_csv(args.output / f"{name}.csv", index=False)
    metadata = {
        "status": "complete_exploratory_cpu_followup", "date": "2026-10-29", "verified_runs": len(statuses),
        "script_sha256": original.sha256(Path(__file__)), "collector_sha256": original.sha256(Path(original.__file__)),
        "run_index_sha256": original.sha256(run_index), "source_repo": "${WORKSPACE}/publication-bias-transfer-demo",
        "input_hashes": "input_hashes.csv", "baseline_hashes": [{k: v for k, v in s.items() if "sha256" in k or k == "dataset"} for s in frozen_status],
        "resamples": args.resamples, "bootstrap_seed": args.seed, "bootstrap_diagnostics": diagnostics,
        "uncertainty": "Percentile 95% whole-participant bootstrap, stratified by country. A person's multiplicity is shared across overlapping seed test sets; each seed normalizes within its original test participants, then seeds average equally. Fixed fitted models, observed seeds, items, folds, and splits. No refitting; intervals exclude training-sample uncertainty and generalization to new items. Observed seed SD/min/max reported separately; three seeds do not establish training-run reliability. Empty seed-subset draws rejected.",
        "aggregation": "Average repetitions within item, items within participant, participants within country, observed seeds equally. Frozen H1 uses all retained US/China participants; adaptation uses only held-out participants and stitched held-out items.",
        "control": "Two offsets fitted only to the original verified training participants/items: semantic logit += delta_semantic + delta_position for ab; += delta_semantic - delta_position for ba. Minimize equal-item/equal-order training cross entropy + .001*(delta_semantic^2+delta_position^2)/4. Frozen probabilities are unchanged inputs. Both controls average probabilities across answer orders after adjustment. Position control added post hoc as a diagnostic; no hyperparameter tuning or evaluation-data fitting.",
        "prediction_intervals": "No participant-bootstrap confidence intervals for model probability shifts: shared prompts yield fixed item probabilities independent of evaluated participant identity. Observed seed spread only.",
        "multiplicity": "Intervals are exploratory pointwise intervals, not simultaneous or multiplicity-adjusted.",
        "privacy": "Only aggregate scores, fit parameters and input hashes exported; no participant identifiers or responses written.",
        "original_score_max_absolute_difference": delta,
        "outputs": {name + ".csv": original.sha256(args.output / f"{name}.csv") for name in tables},
    }
    (args.output / "metadata.json").write_text(json.dumps(metadata, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"status": metadata["status"], "verified_runs": len(statuses), "output": str(args.output),
                      "original_score_max_absolute_difference": delta}, indent=2))


if __name__ == "__main__":
    main()
