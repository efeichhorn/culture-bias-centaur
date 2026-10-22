#!/usr/bin/env python3
"""Freeze participant splits, empirical targets and run manifests for two studies."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from collections import Counter, defaultdict
from pathlib import Path


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_csv(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows({key: row[key] for key in fields} for row in rows)


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")


def ranked(values: list[str], salt: str) -> list[str]:
    return sorted(values, key=lambda value: hashlib.sha256(f"{salt}|{value}".encode()).hexdigest())


def make_splits(participants: list[dict], dataset: str, seed: int) -> tuple[list[dict], dict[str, list[dict]]]:
    countries = {country: sorted(row["participant_id"] for row in participants if row["country"] == country)
                 for country in ("us", "china")}
    if any(len(ids) != len(set(ids)) for ids in countries.values()):
        raise ValueError("Duplicate participant in population table")
    limiting = min(map(len, countries.values()))
    train_n, validation_n = int(limiting * .70), int(limiting * .15)
    if train_n % 2:
        raise ValueError("The frozen protocol requires an even specialist training budget")
    records, conditions = [], {"us": [], "china": [], "pooled": []}
    for country, ids in countries.items():
        selected = ranked(ids, f"{dataset}|{seed}|{country}|balance")[:limiting]
        selected = ranked(selected, f"{dataset}|{seed}|{country}|split")
        assignment = {pid: ("train" if i < train_n else "validation" if i < train_n + validation_n else "test")
                      for i, pid in enumerate(selected)}
        records.extend({"participant_id": pid, "country": country, "split": assignment.get(pid, "unused")}
                       for pid in ids)
        training = [pid for pid in selected if assignment[pid] == "train"]
        conditions[country] = [{"participant_id": pid, "country": country} for pid in training]
        pooled = ranked(training, f"{dataset}|{seed}|{country}|pooled")[:train_n // 2]
        conditions["pooled"].extend({"participant_id": pid, "country": country} for pid in pooled)
    if any(len(rows) != train_n for rows in conditions.values()):
        raise AssertionError("Conditions have unequal participant budgets")
    return records, conditions


def aggregate_targets(responses: list[dict], training: list[dict], item_ids: list[str], repeats: int) -> list[dict]:
    selected = {(row["participant_id"], row["country"]) for row in training}
    if len(selected) != len(training):
        raise ValueError("Duplicate training participant")
    totals, positives, people, multiplicity = Counter(), Counter(), defaultdict(set), Counter()
    items = set(item_ids)
    for row in responses:
        person = (row["participant_id"], row["country"])
        item = str(row["item_id"])
        if person not in selected or item not in items:
            continue
        value = int(row["response_positive"])
        if value not in (0, 1):
            raise ValueError("Nonbinary response")
        totals[item] += 1
        positives[item] += value
        people[item].add(person)
        multiplicity[(person, item)] += 1
    if set(multiplicity) != {(person, item) for person in selected for item in items}:
        raise ValueError("Missing participant-item responses")
    if set(multiplicity.values()) != {repeats}:
        raise ValueError("Unexpected repetition count")
    return [{"item_id": item, "p_positive": positives[item] / totals[item],
             "n_responses": totals[item], "n_participants": len(people[item])} for item in sorted(items)]


def prepare(repo: Path, data_root: Path, run_root: Path, protocol_path: Path, datasets: list[str]) -> dict:
    protocol = json.loads(protocol_path.read_text())
    all_runs, all_sources = [], {}
    for dataset in datasets:
        spec = protocol["datasets"][dataset]
        source = data_root / dataset
        items_path, responses_path = source / "items.json", source / "responses.csv"
        participants_path, source_manifest = source / "participants.csv", source / "manifest.json"
        items = json.loads(items_path.read_text())
        item_ids = [str(item["item_id"]) for item in items]
        if len(item_ids) != len(set(item_ids)):
            raise ValueError("Repeated canonical item")
        responses, participants = read_csv(responses_path), read_csv(participants_path)
        folds = spec["heldout_item_folds"] if dataset == "cao" else spec["heldout_condition_folds"]
        folds = [[str(item) for item in fold] for fold in folds]
        if sorted(sum(folds, [])) != sorted(item_ids):
            raise ValueError("Held-out folds do not partition canonical items")
        expected_repeats = 1 if dataset == "cao" else 4
        source_hashes = {"protocol": digest(protocol_path), "prepared_manifest": digest(source_manifest),
                         "responses": digest(responses_path), "items": digest(items_path), "participants": digest(participants_path)}
        all_sources[dataset] = source_hashes
        for seed in protocol["inherited_method"]["participant_seeds"]:
            split, conditions = make_splits(participants, dataset, seed)
            split_path = run_root / dataset / f"seed{seed}" / "participant_split.csv"
            write_csv(split_path, split, ["participant_id", "country", "split"])
            observed = Counter((row["country"], row["split"]) for row in split)
            for country in ("us", "china"):
                for partition in ("train", "validation", "test"):
                    if observed[country, partition] != spec["split"][f"{partition}_per_country"]:
                        raise ValueError(f"Counts differ from protocol: {dataset} {observed}")
            for fold, heldout in enumerate(folds):
                training_items = sorted(set(item_ids) - set(heldout))
                for condition, training in conditions.items():
                    directory = run_root / dataset / f"seed{seed}" / f"fold{fold}" / condition
                    targets = aggregate_targets(responses, training, training_items, expected_repeats)
                    if sum(target["n_responses"] for target in targets) != spec["observed_responses_per_adapter"]:
                        raise ValueError("Response budget differs from protocol")
                    participants_file = directory / "training_participants.csv"
                    targets_file = directory / "training_targets.csv"
                    write_csv(participants_file, training, ["participant_id", "country"])
                    write_csv(targets_file, targets, ["item_id", "p_positive", "n_responses", "n_participants"])
                    rel = lambda path: os.path.relpath(path, directory)
                    output = repo / "outputs" / "culture_extensions_20261021" / dataset / f"seed{seed}" / f"fold{fold}" / condition
                    manifest = {
                        "schema_version": 1, "dataset": dataset, "seed": seed, "fold": fold, "condition": condition,
                        "items_path": rel(items_path), "items_sha256": digest(items_path),
                        "responses_path": rel(responses_path), "responses_sha256": digest(responses_path),
                        "participant_split_path": rel(split_path), "participant_split_sha256": digest(split_path),
                        "training_participants_path": rel(participants_file), "training_participants_sha256": digest(participants_file),
                        "training_targets_path": rel(targets_file), "training_targets_sha256": digest(targets_file),
                        "protocol_path": rel(protocol_path), "protocol_sha256": digest(protocol_path),
                        "source_hashes": source_hashes, "train_item_ids": training_items, "heldout_item_ids": heldout,
                        "output_dir": rel(output), "expected_optimizer_steps": 25,
                        "n_training_participants": len(training), "n_training_responses": sum(t["n_responses"] for t in targets),
                        "training_order_weight": .5, "positive_response": spec["semantic_response"],
                    }
                    manifest_path = directory / "run_manifest.json"
                    write_json(manifest_path, manifest)
                    all_runs.append({"dataset": dataset, "seed": seed, "fold": fold, "condition": condition,
                                     "manifest": os.path.relpath(manifest_path, repo), "manifest_sha256": digest(manifest_path)})
    plan = {"protocol_sha256": digest(protocol_path), "sources": all_sources, "runs": all_runs,
            "count": len(all_runs), "note": "Frozen inputs and split manifests only; no model training performed by preparation."}
    write_json(run_root / "run_index.json", plan)
    return plan


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--datasets", nargs="+", choices=["cao", "anllo"], default=["cao", "anllo"])
    args = parser.parse_args()
    repo = args.repo.resolve()
    plan = prepare(repo, repo / "data_cache/culture_extensions_20261021", repo / "data_cache/culture_extensions_20261021/runs",
                   repo / "designs/culture_extension_protocol_20261021.json", args.datasets)
    print(json.dumps({"runs": plan["count"], "sources": plan["sources"]}, indent=2))


if __name__ == "__main__":
    main()
