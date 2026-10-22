#!/usr/bin/env python3
"""Recompute summary tables from the committed, participant-free results."""
import argparse
from pathlib import Path

import pandas as pd


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    root = Path(__file__).resolve().parents[1]
    parser.add_argument("--results-dir", type=Path, default=root / "results")
    parser.add_argument("--output-dir", type=Path, default=root / "build/results")
    args = parser.parse_args()
    source, output = args.results_dir, args.output_dir
    output.mkdir(parents=True, exist_ok=True)

    status = pd.read_csv(source / "cao_anllo/run_status.csv")
    if len(status) != 81 or not status.status.eq("verified_production").all():
        raise ValueError("Expected 81 verified extension fits")
    if not status.frozen_baseline_max_probability_difference.eq(0).all():
        raise ValueError("Precision-aligned frozen references no longer match")

    seeds = pd.read_csv(source / "cao_anllo/seed_metrics.csv")
    primary = seeds[((seeds.dataset == "cao") & (seeds.stratum == "critical")) |
                    ((seeds.dataset == "anllo") & (seeds.stratum == "tradeoff"))]
    if primary.groupby("dataset").size().to_dict() != {"anllo": 108, "cao": 108}:
        raise ValueError("Expected both datasets, each with 3 seeds × 3 conditions × 4 models × 3 country summaries")
    if not primary.coverage.eq("complete").all():
        raise ValueError("Incomplete held-out item coverage")
    keys = ["dataset", "source", "condition", "country", "stratum"]
    if primary[keys].isna().any().any():
        raise ValueError("Missing group labels")
    summary = primary.groupby(keys).agg(
        nll_mean=("conditional_nll", "mean"),
        nll_seed_min=("conditional_nll", "min"),
        nll_seed_max=("conditional_nll", "max"),
        predicted_probability=("predicted_p_positive", "mean"),
        seeds=("seed", "nunique"),
    ).reset_index()
    if not summary.seeds.eq(3).all():
        raise ValueError("Expected three joint sampling/fitting seeds")
    summary.to_csv(output / "cao_anllo_summary.csv", index=False)

    contrasts = pd.read_csv(source / "cao_anllo/contrasts.csv")
    contrasts = contrasts[((contrasts.dataset == "cao") & (contrasts.stratum == "critical")) |
                          ((contrasts.dataset == "anllo") & (contrasts.stratum == "tradeoff"))]
    shifts = contrasts[contrasts.contrast == "us_training_minus_china_training"]
    if shifts.groupby("dataset").size().to_dict() != {"anllo": 3, "cao": 3}:
        raise ValueError("Expected one population-shift estimate per dataset and seed")
    shifts.groupby("dataset").prediction_shift_pp.agg(
        mean="mean", seed_min="min", seed_max="max"
    ).to_csv(output / "population_shifts.csv")

    giving = pd.read_csv(source / "giving_taking/seed_contrasts.csv")
    giving.groupby(["task", "Country", "metric", "contrast"]).value.agg(
        mean="mean", seed_sd="std", seed_min="min", seed_max="max"
    ).to_csv(output / "giving_taking_contrasts.csv")
    print(f"Wrote three summary tables to {output}")
    print("Seed ranges are descriptive, not confidence intervals. Raw data were not reanalysed.")


if __name__ == "__main__":
    main()
