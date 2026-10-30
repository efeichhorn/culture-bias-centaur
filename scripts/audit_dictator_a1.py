#!/usr/bin/env python3
"""Score allocation responses and bootstrap paired model comparisons.

The same participant receives the same bootstrap weight across model seeds.
Fixed item folds supply one prediction for each held-out social target.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

KEYS = ["Dataset", "Nation", "SubNo"]
METRICS = ["raw_response_ll", "conditional_ll"]


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
                denominator = np.einsum("ij,j->i", weights, present, optimize=False)
                if (denominator == 0).any():
                    raise ValueError("Empty bootstrap stratum; use larger strata")
                weighted = np.einsum("ij,jk->ik", weights, matrix, optimize=False)
                out[start:stop, j] += weighted / denominator[:, None] * wave_weight
    return out, seeds, outcomes
