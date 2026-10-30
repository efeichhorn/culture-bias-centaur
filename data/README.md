# Study data

`paper_inputs.zip` contains the prepared responses, participant splits, prompts,
model predictions, training records, and fitted response controls used in the
paper. It includes 45 allocation fits and 81 binary-task fits across three seeds.
The archive is 47 MB and expands to 2.81 GB. Repeated training prompts account
for most of the expanded size.

```bash
python scripts/prepare_paper_data.py
```

This checks every SHA-256 hash and extracts to `data/workspace/`. Repeating the
command keeps matching files. A changed file causes an error; use a new
`--output` directory to keep both versions. `--verify-only` checks the archive
without extracting it. `paper_inputs.manifest.json` lists every file and hash.

The bundle contains derived data. Original source downloads and model weights
are excluded. Download links are below. Participant IDs are the source studies'
codes; they are needed to reproduce splits and participant bootstraps.

## Files after extraction

| Path under `data/workspace/` | Contents |
| --- | --- |
| `data_cache/dictator_Behavior_corrected_20260930.csv` | Allocation responses with corrected column labels |
| `data_cache/dictator_centaur70b_corrected_20260930/` | Giving training prompts and participant splits |
| `data_cache/dictator_centaur70b/seed*/` | Original splits used for the frozen-model comparison |
| `data_cache/dictator_centaur70b_item_holdout_corrected_20260930/` | Five fixed item folds |
| `data_cache/balanced_taking_20261029/` | Balanced Taking participant selection |
| `model_results/dictator_centaur70b_item_holdout_corrected_20260930/` | Corrected allocation run records, probabilities, and scores |
| `model_results/dictator_a1_controls_20260930/` | Frozen Llama and Centaur probabilities |
| `model_results/a1_corrected_baselines_20260930/` | Frozen-model participant scores and summaries |
| `model_results/wang_source_audit_20260930/` | Checks of the allocation source correction |
| `publication-bias-transfer-demo/data_cache/culture_extensions_20261021/` | Prepared categorization and lottery data, splits, and targets |
| `publication-bias-transfer-demo/outputs/culture_extensions_20261021/` | Binary-task predictions, scores, and run records |
| `run_records/` | Production job logs |

The long directory names preserve links and hashes in the original run
manifests. `publication-bias-transfer-demo` is only a saved data folder here.
It contains no separate code checkout. Historical machine paths in run records
describe the original jobs; the analyses use the paths supplied on the command
line.

## Columns and units

- Allocation: one row per participant and target. `(Dataset, Nation, SubNo)`
  identifies a participant; `GroupNo` identifies one of 25 targets. `Dataset`
  1–2 means Giving and 3–4 means Taking, with the two values giving study waves.
  `Nation` is 0 for the United States and 1 for China. Corrected `Token` is the
  percentage given or taken (0–100 in steps of 5); divide by 5 for 0–20 tokens.
  The original demographic and rating fields remain in this corrected table.
  Their codes are in the source codebook.
- Binary-task `responses.csv`: one row per response. `participant_id`,
  `country`, `item_id`, and `repeat_id` identify the observation.
  `response_positive=1` means a taxonomic category choice for categorization,
  or the 10-point option for lotteries. Categorization has one response per
  item; lotteries have four repetitions. `items.json` stores prompts, answer
  options, and item strata. `participants.csv` stores the available people.
- Split tables use `train`, `validation`, `test`, and, where needed, `unused`.
  The reserved validation participants were not used for checkpoint selection.
  `training_participants.csv` and `training_targets.csv` define each fit.
- Allocation `candidate_probabilities.csv`: one row per target and response
  value, with full-vocabulary log probability, valid-response mass, and the
  probability conditional on the 21 allowed responses. The frozen control
  table also labels model, task, and prompt variant; paper comparisons use
  `prompt_variant=original`.
- Binary `candidate_probabilities.csv`: one row per item, model, and answer
  order (`ab` or `ba`). `logp_a` and `logp_b` are answer-token log probabilities;
  `p_positive` is the conditional semantic-choice probability. Analyses average
  the two orders. Frozen references use `baseline_precision_aligned/`.
- Participant scores use natural logarithms. Lower negative log likelihood
  (NLL) is better. Positive log-score gain means improvement. Seed ranges show
  variation across the three fitted seeds; bootstrap intervals resample people.

The allocation source had rotated outcome labels. Only the corrected file is
included. Its SHA-256 is
`65f8913600fddeae3ca99196c2b0e2db76f2bf2021e1b32f4419f954645a5ecd`.
The `.provenance.json` sidecar records the correction and source checks.

## Original sources

- Allocation: [Wang et al.](https://doi.org/10.1093/pnasnexus/pgag172),
  [OSF project](https://osf.io/syta3/),
  [Behavior.xlsx](https://osf.io/download/wtmpu/),
  [GroupNo.xlsx](https://osf.io/download/684051ee2434947ecd85b513/),
  [codebook](https://osf.io/download/684051f8ee16d7e551ab83f8/).
  The download and correction commands are in `REPRODUCING.txt`.
- Categorization: [Cao et al.](https://doi.org/10.1037/xge0001559),
  [response data](https://github.com/anjiecao/CCRR_writeups), and
  [stimuli](https://github.com/anjiecao/uschinareplication.github.io).
  `prepare_cao_culture_extension.py` downloads six files at fixed commits and
  checks their hashes. It selects the 15 critical items from Experiment 2.
- Lotteries: [Anlló et al.](https://doi.org/10.1038/s41562-024-01894-9) and
  [source CSV](https://osf.io/download/56gyd/).
  `prepare_anllo_culture_extension.py` checks the source hash and selects
  explicit lotteries, phase 3, conditions 5–12. The prepared table retains all
  561 people; fitting uses the US and Chinese participant splits.

Source licenses and access terms still apply. The source URLs and hashes are
also stored in the prepared dataset manifests.

## Models

Obtain any required Hugging Face access, then download the two model repositories:

```bash
hf download unsloth/Meta-Llama-3.1-70B-bnb-4bit
hf download marcelbinz/Llama-3.1-Centaur-70B-adapter
```

Set `HF_HOME` to the cache used by the training jobs. Frozen Llama evaluation
disables the Centaur adapter. The additional trained adapters are not bundled;
their configurations, training targets, and predictions are included. Runtime
versions are recorded in `provenance/` and the individual training records.
The CPU reanalysis needs neither Hugging Face access nor model weights.
