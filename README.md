# Sampling Bias in Foundation Models of Human Cognition

Code and data for comparing Centaur predictions after fine-tuning on US,
Chinese, or equally pooled responses. The tasks are social allocation
(Giving and Taking), categorization, and risky choice.

## Reproduce the results

Use Python 3.10–3.12. From this folder:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
make results figures test
```

This checks the saved results, rebuilds the six paper figures, and runs the tests
without a GPU or model download. Outputs go to `build/results/` and `paper/figures/`.

To repeat the analyses from participant responses and saved model predictions:

```bash
make data
make analysis
```

The included data contain prepared responses, participant splits, prompts, and
model predictions. Model weights are not included. See [data/README.md](data/README.md)
for data sources and formats.

## Training and other commands

[REPRODUCING.txt](REPRODUCING.txt) contains the GPU training commands, environment
details, and instructions for preparing the submission ZIP.
