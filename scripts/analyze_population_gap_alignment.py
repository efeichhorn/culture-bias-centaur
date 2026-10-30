#!/usr/bin/env python3
"""Compare human and fitted-model population gaps on all 25 allocation targets.

CPU-only exploratory follow-up. Reconstructs actual held-out Giving samples and
uses a supplied, ignored participant manifest for balanced Taking. The same
Giving-trained adapters are compared with each human wave; no wave-specific
models are fitted. Only aggregate outputs are written outside ignored build/.
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
from scripts.audit_dictator_a1 import shared_bootstrap

BEHAVIOR_SHA = "65f8913600fddeae3ca99196c2b0e2db76f2bf2021e1b32f4419f954645a5ecd"
KEYS = ["Dataset", "Nation", "SubNo"]
SAMPLES = ["Wave 1", "Wave 2", "Combined"]
GROUPS = list(range(1, 26))


def combine_waves(waves: list[np.ndarray], counts: np.ndarray) -> np.ndarray:
    """Combine [draw, seed, target] waves using original within-seed counts."""
    counts = np.asarray(counts, dtype=float)
    if len(waves) != 2 or counts.shape != (2, waves[0].shape[1]):
        raise ValueError("Expected two waves and one participant count per wave/seed")
    if waves[0].shape != waves[1].shape or not (counts > 0).all():
        raise ValueError("Wave shapes must match and every stratum must be nonempty")
    weights = counts / counts.sum(axis=0, keepdims=True)
    return sum(wave * weight[None, :, None] for wave, weight in zip(waves, weights))


def alignment_metrics(human: np.ndarray, model: np.ndarray) -> dict:
    """Descriptive alignment over the fixed 25 targets; exact ties are excluded."""
    human, model = np.asarray(human), np.asarray(model)
    nonzero = (~np.isclose(human, 0, atol=1e-12, rtol=0)
               & ~np.isclose(model, 0, atol=1e-12, rtol=0))
    leave_one_out = [np.corrcoef(np.delete(human, j), np.delete(model, j))[0, 1]
                     for j in range(len(human))]
    return dict(
        human_mean_gap=float(human.mean()), model_mean_gap=float(model.mean()),
        model_minus_human_mean_gap=float((model-human).mean()),
        target_gap_correlation=float(np.corrcoef(human, model)[0, 1]),
        target_gap_rmse=float(np.sqrt(np.mean((human-model)**2))),
        constant_model_gap_rmse=float(np.sqrt(np.mean((human-model.mean())**2))),
        leave_one_target_out_correlation_min=float(np.min(leave_one_out)),
        leave_one_target_out_correlation_max=float(np.max(leave_one_out)),
        target_gap_sign_matches=int(((np.sign(human) == np.sign(model)) & nonzero).sum()),
        targets_with_nonzero_gaps=int(nonzero.sum()),
        positive_human_gap_targets=int((human > 0).sum()),
        positive_model_gap_targets=int((model > 0).sum()),
        human_gap_sd_across_targets=float(human.std(ddof=1)),
        model_gap_sd_across_targets=float(model.std(ddof=1)),
    )


class Alignment:
    def __init__(self, workspace: Path, output: Path, taking_manifest: Path | None,
                 resamples: int, bootstrap_seed: int):
        self.workspace, self.output = workspace, output
        self.taking_manifest = taking_manifest
        self.resamples, self.bootstrap_seed = resamples, bootstrap_seed
        self.inputs: set[Path] = {Path(__file__), ROOT / "scripts/audit_dictator_a1.py",
                                  ROOT / "scripts/paper_plot_style.py"}
        self.validation: list[dict] = []
        self.counts: list[dict] = []
        self.run = workspace / "model_results/dictator_centaur70b_item_holdout_corrected_20260930"

    def read(self, path: Path):
        self.inputs.add(path)
        return json.loads(path.read_text()) if path.suffix == ".json" else pd.read_csv(path)

    def check(self, name: str, actual, expected):
        actual, expected = np.asarray(actual), np.asarray(expected)
        if actual.shape != expected.shape:
            raise ValueError(f"Shape mismatch validating {name}: {actual.shape}, {expected.shape}")
        error = float(np.max(np.abs(actual - expected)))
        self.validation.append(dict(check=name, values=actual.size, maximum_absolute_error=error))
        if not np.isfinite(error) or error > 1e-10:
            raise ValueError(f"Recomputed values differ from saved results: {name}: {error}")

    def load(self):
        source = self.workspace / "data_cache/dictator_Behavior_corrected_20260930.csv"
        if hashlib.sha256(source.read_bytes()).hexdigest() != BEHAVIOR_SHA:
            raise ValueError("Expected verified corrected allocation responses")
        behavior = self.read(source)
        behavior["value"] = behavior.Token / 5
        if not behavior.value.between(0, 20).all() or not behavior.Token.mod(5).eq(0).all():
            raise ValueError("Invalid 0--20 token units")
        if behavior.duplicated(KEYS + ["GroupNo"]).any():
            raise ValueError("Duplicate human participant/target rows")
        if not behavior.groupby(KEYS).size().eq(25).all():
            raise ValueError("Every human participant must contribute all 25 targets")
        folds = self.read(self.workspace / "data_cache/dictator_centaur70b_item_holdout_corrected_20260930/item_fold_manifest.csv")
        if sorted(folds.GroupNo.tolist()) != GROUPS:
            raise ValueError("Expected exactly 25 fixed target folds")
        self.names = folds.set_index("GroupNo").item.to_dict()
        giving = []
        for seed in range(3):
            corpus = self.workspace / f"data_cache/dictator_centaur70b_corrected_20260930/seed{seed}"
            split = self.read(corpus / "participant_split_manifest.csv")
            ids = split.loc[split.split.eq("test"), KEYS]
            training = self.read(corpus / "training_participant_manifest.csv")
            if ids.duplicated().any() or len(ids.merge(training[KEYS].drop_duplicates(), on=KEYS)):
                raise ValueError("Invalid Giving test IDs or overlap with training participants")
            part = behavior.merge(ids, on=KEYS, validate="many_to_one")
            if len(part) != len(ids)*25 or not part.Dataset.isin([1, 2]).all():
                raise ValueError("Giving held-out participant join failed")
            giving.append(part.assign(seed=seed))
        full_taking = behavior.loc[behavior.Dataset.isin([3, 4])].copy()
        original = {"Giving": pd.concat(giving, ignore_index=True),
                    "Taking": pd.concat([full_taking.assign(seed=s) for s in range(3)], ignore_index=True)}
        fitted, saved_humans = [], []
        for seed in range(3):
            for condition in ["us", "china", "pooled"]:
                for fold, group_frame in folds.groupby("fold"):
                    directory = self.run / f"seed{seed}/fold{fold}/{condition}"
                    marker = self.read(directory / "verified_complete.json")
                    if (marker["status"] != "verified_complete" or marker["source_behavior_sha256"] != BEHAVIOR_SHA
                            or marker["optimizer_steps"] != 25 or sorted(marker["groups"]) != sorted(group_frame.GroupNo)):
                        raise ValueError("Expected completed corrected fit with the target held out")
                    for task in ["Giving", "Taking"]:
                        predictions = self.read(directory / f"evaluations/{task.lower()}/item_predictions.csv")
                        if sorted(predictions.GroupNo.tolist()) != sorted(group_frame.GroupNo):
                            raise ValueError("Prediction targets differ from held-out targets")
                        fitted.append(predictions[["GroupNo", "expected_tokens"]].assign(seed=seed, condition=condition, task=task))
                        saved = self.read(directory / f"evaluations/{task.lower()}/human_item_comparison.csv")
                        saved_humans.append(saved.assign(seed=seed, condition=condition, task=task))
        self.models = pd.concat(fitted, ignore_index=True)
        self.saved = pd.concat(saved_humans, ignore_index=True)
        if self.models.duplicated(["task", "seed", "condition", "GroupNo"]).any():
            raise ValueError("Duplicated stitched model targets")
        if not self.models.groupby(["task", "seed", "condition"]).size().eq(25).all():
            raise ValueError("Incomplete stitched model target coverage")
        # First validate the unmodified source sample against historical saved
        # predictions and every task/wave/country/seed/target, then apply balance.
        historical = self.point_means(original)
        for (task, seed, condition, sample, country), saved in self.saved.groupby(
                ["task", "seed", "condition", "sample", "country"]):
            nation = 0 if country == "United States" else 1
            actual = historical[(task, sample, nation)][seed]
            saved = saved.sort_values("GroupNo")
            self.check(f"historical_humans/{task}/{sample}/{country}/{seed}/{condition}", actual, saved.human_expected_tokens)
            model = self.models.loc[(self.models.task == task) & (self.models.seed == seed)
                                    & (self.models.condition == condition)].sort_values("GroupNo")
            self.check(f"saved_models/{task}/{sample}/{country}/{seed}/{condition}", model.expected_tokens, saved.model_expected_tokens)
        old_means = self.read(ROOT / "results/giving_taking/response_adjustment/human_model_means_by_seed.csv")
        for row in old_means.itertuples():
            task = row.task.capitalize()
            nation = 0 if row.country == "United States" else 1
            self.check(f"historical_aggregate_humans/{task}/{row.sample}/{row.country}/{row.seed}/{row.condition}",
                       historical[(task, row.sample, nation)][row.seed].mean(), row.human_expected_tokens)
            model = self.models.loc[(self.models.task == task) & (self.models.seed == row.seed)
                                    & (self.models.condition == row.condition)]
            self.check(f"historical_aggregate_models/{task}/{row.sample}/{row.country}/{row.seed}/{row.condition}",
                       model.expected_tokens.mean(), row.model_expected_tokens)
        self.people = original
        if self.taking_manifest is not None:
            ids = self.read(self.taking_manifest)
            if ids.duplicated(KEYS).any() or not ids.Dataset.isin([3, 4]).all():
                raise ValueError("Invalid balanced Taking participant manifest")
            sizes = ids.groupby(["Dataset", "Nation"]).size().to_dict()
            if sizes != {(3, 0): 529, (3, 1): 529, (4, 0): 800, (4, 1): 800}:
                raise ValueError(f"Expected 529 and 800 Taking people per country: {sizes}")
            taking = full_taking.merge(ids[KEYS], on=KEYS, validate="many_to_one")
            if len(taking) != len(ids)*25:
                raise ValueError("Balanced Taking participant join failed")
            self.people["Taking"] = pd.concat([taking.assign(seed=s) for s in range(3)], ignore_index=True)
        self.points = self.point_means(self.people)
        self.historical = historical

    @staticmethod
    def point_means(people: dict[str, pd.DataFrame]) -> dict:
        result = {}
        for task, all_people in people.items():
            for sample in SAMPLES:
                datasets = ([1, 3] if sample == "Wave 1" else [2, 4]) if sample != "Combined" else [1, 2, 3, 4]
                for nation in [0, 1]:
                    part = all_people.loc[all_people.Dataset.isin(datasets) & all_people.Nation.eq(nation)]
                    result[(task, sample, nation)] = part.groupby(["seed", "GroupNo"]).value.mean().unstack().reindex(index=range(3), columns=GROUPS).to_numpy()
        return result

    def bootstrap(self) -> dict:
        rng = np.random.default_rng(self.bootstrap_seed)
        result = {}
        for task, all_people in self.people.items():
            for nation in [0, 1]:
                waves, wave_counts = [], []
                for sample in SAMPLES[:2]:
                    datasets = [1, 3] if sample == "Wave 1" else [2, 4]
                    part = all_people.loc[all_people.Dataset.isin(datasets) & all_people.Nation.eq(nation)].copy()
                    part["outcome"] = part.GroupNo.map(lambda group: f"{group:02}")
                    draws, seeds, outcomes = shared_bootstrap(part, self.resamples, rng)
                    if seeds != [0, 1, 2] or outcomes != [f"{group:02}" for group in GROUPS]:
                        raise ValueError("Unexpected seed/target ordering")
                    if not np.isfinite(draws).all() or not ((draws >= 0) & (draws <= 20)).all():
                        raise ValueError("Invalid participant-bootstrap mean tokens")
                    counts = part[KEYS + ["seed"]].drop_duplicates().groupby("seed").size().reindex(range(3)).to_numpy()
                    for seed, count in enumerate(counts):
                        self.counts.append(dict(task=task, sample=sample, nation=nation, seed=seed, participants=int(count),
                                                unique_participants_across_seeds=len(part[KEYS].drop_duplicates())))
                    result[(task, sample, nation)] = draws
                    waves.append(draws)
                    wave_counts.append(counts)
                result[(task, "Combined", nation)] = combine_waves(waves, np.array(wave_counts))
                # Point estimates must use exactly the same wave weighting.
                combined = combine_waves([self.points[(task, s, nation)][None] for s in SAMPLES[:2]], np.array(wave_counts))[0]
                self.check(f"combined_wave_weighting/{task}/{nation}", combined, self.points[(task, "Combined", nation)])
        return result

    def export(self):
        bootstrap = self.bootstrap()
        items, summaries, seed_rows, seed_summaries, historical_sensitivity = [], [], [], [], []
        for task in ["Giving", "Taking"]:
            model = self.models.loc[self.models.task.eq(task)].pivot(index=["seed", "GroupNo"], columns="condition", values="expected_tokens")
            model_gap_seed = (model.us-model.china).unstack().reindex(index=range(3), columns=GROUPS).to_numpy()
            model_gap = model_gap_seed.mean(axis=0)
            for sample in SAMPLES:
                us, china = (self.points[(task, sample, nation)] for nation in [0, 1])
                human_gap_seed, human_gap = us-china, (us-china).mean(axis=0)
                human_draws = (bootstrap[(task, sample, 0)]-bootstrap[(task, sample, 1)]).mean(axis=1)
                low, high = np.quantile(human_draws, [.025, .975], axis=0)
                for j, group in enumerate(GROUPS):
                    items.append(dict(task=task, sample=sample, GroupNo=group, item=self.names[group],
                                      human_us_mean=us[:, j].mean(), human_china_mean=china[:, j].mean(),
                                      human_gap=human_gap[j], human_gap_ci_low=low[j], human_gap_ci_high=high[j],
                                      model_gap=model_gap[j], model_gap_seed_sd=model_gap_seed[:, j].std(ddof=1),
                                      model_gap_seed_min=model_gap_seed[:, j].min(), model_gap_seed_max=model_gap_seed[:, j].max()))
                    for seed in range(3):
                        seed_rows.append(dict(task=task, sample=sample, GroupNo=group, seed=seed,
                                              human_us_mean=us[seed, j], human_china_mean=china[seed, j],
                                              human_gap=human_gap_seed[seed, j], model_gap=model_gap_seed[seed, j]))
                summary = dict(task=task, sample=sample, targets=25, **alignment_metrics(human_gap, model_gap))
                human_centered = human_draws - human_draws.mean(axis=1, keepdims=True)
                model_centered = model_gap-model_gap.mean()
                correlations = (human_centered * model_centered).sum(axis=1) / np.sqrt((human_centered**2).sum(axis=1)*(model_centered**2).sum())
                for name, values in dict(human_mean_gap=human_draws.mean(axis=1),
                                         model_minus_human_mean_gap=model_gap.mean()-human_draws.mean(axis=1),
                                         target_gap_rmse=np.sqrt(((human_draws-model_gap)**2).mean(axis=1)),
                                         target_gap_correlation=correlations).items():
                    if not np.isfinite(values).all():
                        raise ValueError(f"Nonfinite bootstrap statistic: {task}/{sample}/{name}")
                    lower, upper = np.quantile(values, [.025, .975])
                    summary[name+"_human_ci_low"] = lower
                    summary[name+"_human_ci_high"] = upper
                summary.update(model_mean_gap_seed_sd=model_gap_seed.mean(axis=1).std(ddof=1),
                               model_mean_gap_seed_min=model_gap_seed.mean(axis=1).min(),
                               model_mean_gap_seed_max=model_gap_seed.mean(axis=1).max())
                summaries.append(summary)
                for seed in range(3):
                    seed_summaries.append(dict(task=task, sample=sample, seed=seed,
                                               **alignment_metrics(human_gap_seed[seed], model_gap_seed[seed])))
                historical_gap = (self.historical[(task, sample, 0)]-self.historical[(task, sample, 1)]).mean(axis=0)
                historical_sensitivity.append(dict(task=task, sample=sample, **alignment_metrics(historical_gap, model_gap)))
        self.output.mkdir(parents=True, exist_ok=True)
        for filename, rows in [("population_gap_items.csv", items), ("population_gap_summary.csv", summaries),
                               ("population_gap_by_seed.csv", seed_rows), ("population_gap_seed_summary.csv", seed_summaries),
                               ("population_gap_historical_full_sample.csv", historical_sensitivity),
                               ("population_gap_sample_counts.csv", self.counts), ("population_gap_validation.csv", self.validation)]:
            pd.DataFrame(rows).to_csv(self.output / filename, index=False)
        metadata = dict(status="exploratory_corrected_cpu_followup", source_behavior_sha256=BEHAVIOR_SHA,
                        difference="US minus China, in 0--20 tokens; Given for Giving, taken for Taking",
                        giving="Manifest-held-out Giving test participants for each fitted seed; no training overlap",
                        taking="529 Wave-1 + 800 Wave-2 participants per country from supplied deterministic manifest" if self.taking_manifest else "Historical full Taking sample, unbalanced",
                        model="Same Giving-trained US and China adapters across all wave panels; every target predicted by its held-out-target fit; mean over three fitted seeds",
                        uncertainty="95% percentile participant bootstrap, stratified by wave and country, shared participant multiplicities across targets and overlapping seed test splits. Fixed observed seed fits and target set. Combined retains original country/seed wave weights. Model dispersion is one SD across three fitted seeds, not a confidence interval.",
                        interval_scope="Summary correlation/RMSE intervals propagate only human sampling uncertainty, conditional on these fixed model predictions and 25 targets. No target-population or refitted-model inference.",
                        seeds=[0, 1, 2], bootstrap_seed=self.bootstrap_seed, bootstrap_resamples=self.resamples,
                        manifest_path=str(self.taking_manifest) if self.taking_manifest else None,
                        validation_checks=len(self.validation), maximum_validation_error=max(r["maximum_absolute_error"] for r in self.validation),
                        input_sha256={str(p.relative_to(self.workspace)) if p.is_relative_to(self.workspace) else str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(self.inputs)})
        (self.output / "population_gap_provenance.json").write_text(json.dumps(metadata, indent=2)+"\n")
        return pd.DataFrame(items), pd.DataFrame(summaries)


def draw(items: pd.DataFrame, summaries: pd.DataFrame, output: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from scripts.paper_plot_style import paper_style

    with plt.rc_context(paper_style(3.9, nrows=2, ncols=3)):
        fig, axes = plt.subplots(2, 3, sharex=True, sharey=True)
        xlo, xhi = np.floor(items.human_gap_ci_low.min())-.25, np.ceil(items.human_gap_ci_high.max())+.25
        ylo = np.floor((items.model_gap-items.model_gap_seed_sd).min()*2)/2-.15
        yhi = np.ceil((items.model_gap+items.model_gap_seed_sd).max()*2)/2+.55
        for i, task in enumerate(["Giving", "Taking"]):
            for j, sample in enumerate(SAMPLES):
                ax = axes[i, j]
                part = items.loc[items.task.eq(task) & items["sample"].eq(sample)]
                summary = summaries.loc[summaries.task.eq(task) & summaries["sample"].eq(sample)].iloc[0]
                color = "#526676"
                ax.axhline(0, color="0.6", lw=.45, zorder=0)
                ax.axvline(0, color="0.6", lw=.45, zorder=0)
                ax.plot([xlo, xhi], [xlo, xhi], color="0.4", ls="--", lw=.6, zorder=0)
                ax.errorbar(part.human_gap, part.model_gap,
                            xerr=np.array([part.human_gap-part.human_gap_ci_low, part.human_gap_ci_high-part.human_gap]),
                            yerr=part.model_gap_seed_sd, fmt="o", markersize=2.6, color=color,
                            ecolor=color, elinewidth=.45, capsize=0, alpha=.8, markeredgecolor="white", markeredgewidth=.25)
                ax.set_xlim(xlo, xhi)
                ax.set_ylim(ylo, yhi)
                ax.set_title(f"{task}: {sample}", loc="left", fontsize=8)
                ax.text(.04, .96, f"$r$ = {summary.target_gap_correlation:.2f}; same sign: {int(summary.target_gap_sign_matches)}/25",
                        transform=ax.transAxes, va="top", fontsize=6.5)
                if j == 0:
                    ax.set_ylabel("Adapter US − China (tokens)", fontsize=7.5)
                if i == 1:
                    ax.set_xlabel("Human US − China (tokens)", fontsize=7.5)
                ax.tick_params(labelsize=6.5)
        fig.legend(handles=[Line2D([0], [0], color="0.4", ls="--", lw=.6, label="Equal human and adapter differences")],
                   loc="outside lower center", ncol=1, fontsize=7)
        output.mkdir(parents=True, exist_ok=True)
        fig.savefig(output / "population_gap_alignment.pdf", metadata={"CreationDate": None, "ModDate": None})
        fig.savefig(output / "population_gap_alignment.png", dpi=300)
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path)
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--taking-manifest", type=Path)
    selection.add_argument("--historical-full-taking", action="store_true")
    parser.add_argument("--output", type=Path, default=ROOT / "results/giving_taking/followup_20261029")
    parser.add_argument("--figure-dir", type=Path, default=ROOT / "paper/figures")
    parser.add_argument("--resamples", type=int, default=4000)
    parser.add_argument("--bootstrap-seed", type=int, default=20261029)
    parser.add_argument("--no-figure", action="store_true")
    parser.add_argument("--plot-only", action="store_true", help="Redraw from exported aggregate CSVs; no private inputs required")
    args = parser.parse_args()
    if args.plot_only:
        draw(pd.read_csv(args.output / "population_gap_items.csv"),
             pd.read_csv(args.output / "population_gap_summary.csv"), args.figure_dir)
        return
    if args.workspace is None or not (args.taking_manifest or args.historical_full_taking):
        parser.error("Analysis requires --workspace and either --taking-manifest or --historical-full-taking")
    audit = Alignment(args.workspace.resolve(), args.output.resolve(), args.taking_manifest,
                      args.resamples, args.bootstrap_seed)
    audit.load()
    items, summary = audit.export()
    if not args.no_figure:
        draw(items, summary, args.figure_dir)
    print(summary[["task", "sample", "human_mean_gap", "model_mean_gap", "target_gap_correlation", "target_gap_sign_matches", "target_gap_rmse"]].round(4).to_string(index=False))
    print(f"Validated {len(audit.validation)} source comparisons; aggregate outputs: {args.output}")


if __name__ == "__main__":
    main()
