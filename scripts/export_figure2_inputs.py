#!/usr/bin/env python3
"""Build Figure 2 inputs from prepared responses and saved model predictions."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from analyze_culture_extensions import average_orders

ROOT = Path(__file__).resolve().parents[1]
PARTICIPANT_KEYS = ["Dataset", "Nation", "SubNo"]
TASKS = ("Giving", "Taking", "Categorization", "Risky choice")


class Inputs:
    """Read inputs and keep their hashes for the export record."""

    def __init__(self, workspace: Path, source_repo: Path):
        self.workspace = workspace.resolve()
        self.source_repo = source_repo.resolve()
        self.hashes = {}

    def record(self, path: Path) -> None:
        path = path.resolve()
        for base, prefix in ((ROOT, ""), (self.workspace, "workspace/"),
                             (self.source_repo, "source_repo/")):
            try:
                name = prefix + str(path.relative_to(base))
                break
            except ValueError:
                continue
        else:
            name = path.name
        self.hashes[name] = hashlib.sha256(path.read_bytes()).hexdigest()

    def csv(self, path: Path, **kwargs) -> pd.DataFrame:
        self.record(path)
        return pd.read_csv(path, **kwargs)

    def json(self, path: Path):
        self.record(path)
        return json.loads(path.read_text())


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def allocation_distributions(inputs: Inputs, taking_manifest: Path):
    """Average target-level probabilities; keep each participant's items together."""
    behavior = inputs.csv(inputs.workspace / "data_cache/dictator_Behavior_corrected_20260930.csv")
    require((behavior.Token % 5 == 0).all(), "Allocation tokens must be multiples of five.")
    behavior["response"] = (behavior.Token / 5).astype(int)
    require(behavior.response.between(0, 20).all(), "Invalid allocation response.")
    require(not behavior.duplicated(PARTICIPANT_KEYS + ["GroupNo"]).any(), "Duplicate allocation response.")
    models = inputs.csv(inputs.workspace / "model_results/dictator_a1_controls_20260930/candidate_probabilities.csv")
    rows, samples, scores = [], [], {}
    for task in ("Giving", "Taking"):
        splits = []
        for seed in (range(3) if task == "Giving" else range(1)):
            if task == "Giving":
                path = inputs.workspace / f"data_cache/dictator_centaur70b/seed{seed}/participant_split_manifest.csv"
                split = inputs.csv(path)
                ids = split.loc[split.split.eq("test"), PARTICIPANT_KEYS]
                require(ids.groupby("Nation").size().to_dict() == {0: 196, 1: 196},
                        "Giving requires 196 held-out participants per country and seed.")
            else:
                ids = inputs.csv(taking_manifest)[PARTICIPANT_KEYS]
                require(ids.groupby(["Dataset", "Nation"]).size().to_dict()
                        == {(3, 0): 529, (3, 1): 529, (4, 0): 800, (4, 1): 800},
                        "Taking requires 529 and 800 participants per country in the two waves.")
            require(not ids.duplicated().any(), "Duplicate participant in a split.")
            part = behavior.merge(ids, on=PARTICIPANT_KEYS, validate="many_to_one")
            require(len(part) == len(ids) * 25 and part.groupby(PARTICIPANT_KEYS).size().eq(25).all(),
                    "Each allocation participant must have 25 responses.")
            require(part.Dataset.isin([1, 2] if task == "Giving" else [3, 4]).all(), "Wrong study wave.")
            splits.append(part.assign(seed=seed))
        humans = pd.concat(splits, ignore_index=True)
        outcome = "tokens_given" if task == "Giving" else "tokens_taken"
        for nation, source in ((0, "us"), (1, "china")):
            part = humans.loc[humans.Nation.eq(nation)]
            by_seed = [group.response.value_counts(normalize=True).reindex(range(21), fill_value=0)
                       for _, group in part.groupby("seed")]
            probabilities = pd.concat(by_seed, axis=1).mean(axis=1)
            samples.append(dict(task=task, source=source,
                                unique_participants=len(part[PARTICIPANT_KEYS].drop_duplicates()),
                                participants_per_split=part[PARTICIPANT_KEYS + ["seed"]].drop_duplicates().groupby("seed").size().tolist()))
            rows.extend(dict(task=task, source=source, option=str(response), probability=probability,
                             outcome=outcome) for response, probability in probabilities.items())
        for source in ("centaur", "llama"):
            part = models.loc[models.task.eq(task.lower()) & models.model_condition.eq(source)
                              & models.prompt_variant.eq("original")]
            require(len(part) == 25 * 21 and not part.duplicated(["GroupNo", "response_value"]).any(),
                    f"Incomplete allocation predictions: {task}/{source}.")
            require(np.allclose(part.groupby("GroupNo").conditional_probability.sum(), 1, atol=1e-12),
                    "Allocation probabilities must sum to one within each target.")
            probabilities = part.groupby("response_value").conditional_probability.mean().reindex(range(21))
            rows.extend(dict(task=task, source=source, option=str(response), probability=probability,
                             outcome=outcome) for response, probability in probabilities.items())
            scored = humans.merge(part[["GroupNo", "response_value", "conditional_probability"]],
                                  left_on=["GroupNo", "response"], right_on=["GroupNo", "response_value"],
                                  validate="many_to_one")
            scored["nll"] = -np.log(scored.conditional_probability)
            country_scores = scored.groupby(["Nation", "seed"]).nll.mean().groupby("Nation").mean()
            scores[(task, source)] = country_scores.loc[1] - country_scores.loc[0]
    return pd.DataFrame(rows), samples, scores


def binary_distributions(inputs: Inputs):
    """Average repeated human choices within participant and both model answer orders."""
    rows, samples, item_labels, scores = [], [], {}, {}
    for dataset, task in (("cao", "Categorization"), ("anllo", "Risky choice")):
        base = inputs.source_repo / "data_cache/culture_extensions_20261021" / dataset
        items = inputs.json(base / "items.json")
        item_ids = [str(item["item_id"]) for item in items]
        item_labels[task] = items
        human = inputs.csv(base / "responses.csv", dtype={"item_id": str})
        human = human.loc[human.country.isin(["us", "china"]) & human.item_id.isin(item_ids)]
        people = human.groupby(["country", "participant_id", "item_id"]).response_positive.mean()
        require(people.index.to_frame(index=False).groupby(["country", "participant_id"]).size().eq(len(items)).all(),
                f"Incomplete participant responses: {dataset}.")
        outcome = "taxonomic_choice" if dataset == "cao" else "10_point_choice"
        country_probabilities = {}
        for source in ("us", "china"):
            part = people.loc[source].unstack("item_id").reindex(columns=item_ids)
            count = ({"us": 295, "china": 178} if dataset == "cao" else {"us": 50, "china": 49})[source]
            require(len(part) == count and not part.isna().any().any(), f"Wrong participant count: {dataset}/{source}.")
            samples.append(dict(task=task, source=source, unique_participants=len(part)))
            country_probabilities[source] = part.mean()
            rows.extend(dict(task=task, source=source, option=item_id, probability=probability,
                             outcome=outcome) for item_id, probability in part.mean().items())
        path = inputs.source_repo / "outputs/culture_extensions_20261021/baseline_precision_aligned" / dataset / "candidate_probabilities.csv"
        predictions = inputs.csv(path, dtype={"item_id": str})
        expected = {(source, item, order) for source in ("frozen_centaur", "base_llama")
                    for item in item_ids for order in ("ab", "ba")}
        keys = list(predictions[["source", "item_id", "order"]].itertuples(index=False, name=None))
        require(len(keys) == len(expected) and set(keys) == expected, f"Incomplete binary predictions: {dataset}.")
        for source, saved_source in (("centaur", "frozen_centaur"), ("llama", "base_llama")):
            probabilities = {}
            for item_id in item_ids:
                pair = predictions.loc[predictions.source.eq(saved_source) & predictions.item_id.eq(item_id)]
                probabilities[item_id] = average_orders(pair.to_dict("records"))["p_positive"]
                rows.append(dict(task=task, source=source, option=item_id,
                                 probability=probabilities[item_id], outcome=outcome))
            model = pd.Series(probabilities)
            nll = {country: float((-human * np.log(model) - (1-human) * np.log1p(-model)).mean())
                   for country, human in country_probabilities.items()}
            scores[(task, source)] = nll["china"] - nll["us"]
    return pd.DataFrame(rows), samples, item_labels, scores


def population_gaps(inputs: Inputs, response_scores: dict, results_dir: Path) -> pd.DataFrame:
    """Use the saved paired-bootstrap intervals and check their point estimates."""
    allocation = inputs.csv(results_dir / "giving_taking/balanced_taking_20261029/h1_population_contrasts.csv")
    binary = inputs.csv(results_dir / "cao_anllo/followup_20261029/frozen_contrasts.csv")
    rows = []
    for task in TASKS:
        for source in ("llama", "centaur"):
            if task in ("Giving", "Taking"):
                selected = allocation.loc[allocation.task.eq(task) & allocation.model.eq(source)]
            else:
                dataset = "cao" if task == "Categorization" else "anllo"
                contrast = ("base_llama" if source == "llama" else "frozen_centaur") + "_country_nll_gap"
                selected = binary.loc[binary.dataset.eq(dataset) & binary.stratum.eq("all") & binary.contrast.eq(contrast)]
            require(len(selected) == 1, f"Expected one NLL contrast: {task}/{source}.")
            row = selected.iloc[0]
            require(np.isclose(response_scores[(task, source)], row.estimate, atol=1e-12, rtol=0),
                    f"NLL contrast differs from the response-level calculation: {task}/{source}.")
            rows.append(dict(panel="b", task=task, source=source, estimate=row.estimate,
                             ci_low=row.ci_low, ci_high=row.ci_high, units="nats",
                             outcome="China minus US conditional NLL", interval="95% participant bootstrap"))
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, default=ROOT / "data/workspace")
    parser.add_argument("--source-repo", type=Path)
    parser.add_argument("--taking-manifest", type=Path)
    parser.add_argument("--results-dir", type=Path, default=ROOT / "results")
    parser.add_argument("--output", type=Path, default=ROOT / "results/figure2")
    args = parser.parse_args()
    source_repo = args.source_repo or args.workspace / "publication-bias-transfer-demo"
    taking_manifest = args.taking_manifest or args.workspace / "data_cache/balanced_taking_20261029/selected_participants.csv"
    inputs = Inputs(args.workspace, source_repo)
    allocation, allocation_samples, allocation_scores = allocation_distributions(inputs, taking_manifest)
    binary, binary_samples, items, binary_scores = binary_distributions(inputs)
    data = pd.concat([allocation, binary], ignore_index=True)
    gaps = population_gaps(inputs, {**allocation_scores, **binary_scores}, args.results_dir)
    require(len(data) == 260 and not data.duplicated(["task", "source", "option"]).any(), "Wrong response table coverage.")
    require(data.probability.between(0, 1).all(), "Invalid response probabilities.")
    risky_items = sorted(items["Risky choice"], key=lambda item: (
        item["stratum"] != "tradeoff", item["source_details"]["p_10"], item["source_details"]["p_1"]))
    args.output.mkdir(parents=True, exist_ok=True)
    data.to_csv(args.output / "response_probabilities.csv", index=False)
    gaps.to_csv(args.output / "population_nll_gaps.csv", index=False)
    metadata = {
        "figure": "Observed responses and population NLL gaps (H1)",
        "samples": allocation_samples + binary_samples,
        "categorization_item_order": [str(item["item_id"]) for item in items["Categorization"]],
        "risky_choice_order": [{"item_id": str(item["item_id"]), "group": item["stratum"],
                                "p_10": item["source_details"]["p_10"], "p_1": item["source_details"]["p_1"]}
                               for item in risky_items],
        "aggregation": {
            "allocation_human": "Equal target means; Giving also averages three held-out participant splits. Taking uses 1329 participants per country matched within wave.",
            "allocation_model": "Conditional response probabilities averaged over 25 targets.",
            "binary_human": "Repeated choices averaged within participant and item, then across participants.",
            "binary_model": "Conditional probabilities averaged across both answer orders.",
            "nll_gap": "Chinese minus US mean conditional NLL. Intervals use 4000 participant-bootstrap draws. Risky choice includes all eight items.",
        },
        "file_sha256": {name: hashlib.sha256((args.output / name).read_bytes()).hexdigest()
                        for name in ("response_probabilities.csv", "population_nll_gaps.csv")},
        "source_sha256": inputs.hashes,
    }
    (args.output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"Saved 260 response probabilities and 8 NLL contrasts in {args.output}")


if __name__ == "__main__":
    main()
