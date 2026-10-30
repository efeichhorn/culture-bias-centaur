#!/usr/bin/env python3
"""Reanalyse saved responses and predictions, then check the paper's tables."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def run(script: str, **options) -> None:
    command = [sys.executable, str(ROOT / "scripts" / script)]
    for key, value in options.items():
        if value is False or value is None:
            continue
        command.append("--" + key.replace("_", "-"))
        if value is not True:
            command.append(str(value))
    print(f"Running {script}", flush=True)
    subprocess.run(command, cwd=ROOT, check=True)


def compare_tables(output: Path) -> list[dict]:
    tables = [
        "giving_taking/frozen/metrics_summary.csv",
        "giving_taking/frozen/contrast_summary.csv",
        "giving_taking/frozen/split_metrics.csv",
        "giving_taking/seed_metrics.csv",
        "giving_taking/seed_contrasts.csv",
        "giving_taking/bootstrap_intervals.csv",
        "giving_taking/response_adjustment/prior_metrics_by_seed.csv",
        "giving_taking/response_adjustment/prior_paired_intervals.csv",
        "cao_anllo/seed_metrics.csv",
        "cao_anllo/contrasts.csv",
        "cao_anllo/item_predictions.csv",
        "cao_anllo/frozen_baseline_country_metrics.csv",
        "figure2/response_probabilities.csv",
        "figure2/population_nll_gaps.csv",
    ]
    for folder in ("giving_taking/balanced_taking_20261029", "giving_taking/followup_20261029",
                   "cao_anllo/followup_20261029"):
        for path in sorted((ROOT / "results" / folder).glob("*.csv")):
            if path.name != "input_provenance.csv":
                tables.append(str(path.relative_to(ROOT / "results")))
    checked = []
    for name in tables:
        expected = pd.read_csv(ROOT / "results" / name)
        actual = pd.read_csv(output / name)
        pd.testing.assert_frame_equal(actual, expected, check_dtype=False,
                                      check_exact=False, rtol=1e-10, atol=1e-10)
        numeric = expected.select_dtypes(include="number").columns
        differences = (actual[numeric] - expected[numeric]).abs().to_numpy().ravel()
        finite = differences[np.isfinite(differences)]
        checked.append({"table": name, "rows": len(actual),
                        "max_absolute_difference": float(finite.max()) if finite.size else 0.0})
    return checked


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, default=ROOT / "data/workspace")
    parser.add_argument("--output", type=Path, default=ROOT / "build/reanalysis")
    args = parser.parse_args()
    workspace, output = args.workspace.resolve(), args.output.resolve()
    behavior = workspace / "data_cache/dictator_Behavior_corrected_20260930.csv"
    if not behavior.is_file():
        parser.error("Prepared inputs are missing; run make data first")
    if output == ROOT / "results" or (ROOT / "results") in output.parents:
        parser.error("Choose an output outside results/ to preserve the reference tables")
    allocation = output / "giving_taking"
    binary = workspace / "publication-bias-transfer-demo"
    binary_outputs = binary / "outputs/culture_extensions_20261021"
    balanced = allocation / "balanced_taking_20261029"
    selection = output / "selected_taking_participants.csv"

    run("analyze_dictator_corrected_baselines.py", root=workspace,
        behavior_path=behavior, response_column="Token", response_scale="percent",
        candidates_path=workspace / "model_results/dictator_a1_controls_20260930/candidate_probabilities.csv",
        validation_status="validated", output_dir=allocation / "frozen")
    run("analyze_dictator_corrected_adapters.py", behavior_path=behavior,
        corpus_root=workspace / "data_cache/dictator_centaur70b_corrected_20260930",
        results_root=workspace / "model_results/dictator_centaur70b_item_holdout_corrected_20260930",
        output_dir=allocation, no_plots=True)
    run("audit_corrected_response_adjustment.py", workspace=workspace,
        output=allocation / "response_adjustment")
    run("analyze_culture_extensions.py", repo_root=binary,
        run_index=binary / "data_cache/culture_extensions_20261021/runs/run_index.json",
        outputs_root=binary_outputs, baseline_root=binary_outputs / "baseline_precision_aligned",
        analysis_dir=output / "cao_anllo")
    run("analyze_balanced_taking.py", workspace=workspace, output=balanced, manifest=selection)
    run("analyze_culture_followup.py", source_repo=binary, output=output / "cao_anllo/followup_20261029")
    run("analyze_h2_h3_gains.py", source_repo=binary, output=output / "cao_anllo/followup_20261029")
    run("export_h2_h3_allocation_items.py", workspace=workspace,
        output=allocation / "followup_20261029/h2_h3_item_predictions.csv")
    run("analyze_population_gap_alignment.py", workspace=workspace, taking_manifest=selection,
        output=allocation / "followup_20261029", no_figure=True)
    run("export_figure2_inputs.py", workspace=workspace, source_repo=binary,
        results_dir=output, taking_manifest=selection, output=output / "figure2")

    checked = compare_tables(output)
    (output / "verification.json").write_text(json.dumps({"status": "passed", "tables": checked}, indent=2) + "\n")
    print(f"Verified {len(checked)} tables against results/. New outputs: {output}")


if __name__ == "__main__":
    main()
