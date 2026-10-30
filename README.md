# Population Composition Changes Centaur's Predictions

Code and data for comparing Centaur predictions after fine-tuning on US,
Chinese, or equally pooled responses. The tasks are social allocation
(Giving and Taking), categorization, and risky choice.

## Run

Use Python 3.10–3.12. From this folder:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
make results figures test
```

This checks the saved results, rebuilds all six paper figures, and runs the tests.
It needs no GPU or model download. Tables go to `build/results/`; figures go to
`paper/figures/`.

To repeat the analyses from participant responses and saved model predictions:

```bash
make data
make analysis
```

The data archive includes prepared responses, participant splits, prompts,
predictions, and records for 126 fits. Model weights and unchanged source
datasets are excluded. See [data/README.md](data/README.md) for the file formats,
source links, and extraction details.

## Files

| Path | Contents |
|---|---|
| `scripts/` | Data preparation, training, evaluation, analysis, and plots |
| `src/` | Shared model and training code |
| `data/` | Prepared inputs and saved model outputs |
| `results/` | Analysis tables used in the paper |
| `designs/` | Training settings and fixed item folds |
| `provenance/` | Source checks and recorded software versions |
| `tests/` | Checks for splits, scoring, training settings, and analysis |
| `slurm/` | Runners for the original GPU cluster |
| `paper/` | Downloaded manuscript source and reproduced figures |

[REPRODUCING.txt](REPRODUCING.txt) gives the full analysis and GPU commands.
The [Overleaf project](https://www.overleaf.com/project/6ada6357f0e9637a95c1b3eb)
is the working paper. The local source was downloaded on 30 October 2026;
changes here do not sync to Overleaf.

## Paper results

| Question | Main analysis | Figure |
|---|---|---|
| H1: Does frozen Centaur fit US responses better? | Frozen-model NLL by population | 2 |
| H2: Does fine-tuning population change predictions? | US, China, and pooled adapters on held-out items | 3 |
| H3: Do these differences transfer? | Giving-trained adapters evaluated on Taking | 3 |
| H4: Does fine-tuning beat response adjustment? | Paired NLL differences on the same test responses | 4 |

There are 45 allocation fits, 45 categorization fits, and 36 risky-choice fits.
All use the final checkpoint after 25 updates. The validation split was unused.
Prediction shifts and predictive gains are separate outcomes. Bootstrap
intervals are conditional on the evaluated items and saved fits.

## Package

```bash
make submission
```

This creates `build/culture-bias-centaur-submission.zip` with code, data,
results, and the local paper sources. It excludes environments, caches,
model weights, and local build files. Nothing is uploaded or submitted.
Use `make paper` to compile the local manuscript if LaTeX and `latexmk` are
installed.
