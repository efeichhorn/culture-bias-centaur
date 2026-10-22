# Population bias in Centaur

Code and draft paper on how the population used for fine-tuning changes Centaur's behavioral predictions.

## Contents

| Folder | Contents |
|---|---|
| `paper/` | ACL-format draft, references and figures |
| `scripts/`, `src/` | Data preparation, fine-tuning and analysis |
| `results/` | Aggregate results for Giving/Taking, Cao and Anlló; supporting Ruggeri audit |
| `designs/` | Study protocols |
| `tests/` | Checks for the analysis and data preparation |
| `slurm/` | Configurable cluster runners |
| `provenance/` | Source hashes and recorded training environments |

Raw participant data, model weights and unrelated experiments are excluded.

## Start here

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
make results figures test
```

These commands use committed aggregates and need no GPU. For raw-data analysis and fine-tuning, see [REPRODUCING.txt](REPRODUCING.txt) and [data/README.txt](data/README.txt).

## Paper

[Open the Overleaf draft](https://www.overleaf.com/project/6ada6357f0e9637a95c1b3eb).

```bash
make paper       # requires latexmk and a LaTeX installation
make overleaf    # creates build/culture-bias-centaur-overleaf.zip
```

Edit `paper/sections/` and `paper/references.bib`. The paper is a working draft. The newer datasets still need uncertainty estimates, stronger calibration controls and the planned sensitivity checks. No additional training is started by these commands.

The Overleaf project contains `paper/` only. Sync is manual: copy edited sources back to `paper/` before committing, or upload local changes to Overleaf.

## Before submission

- Finish the analyses above and review every claim and reference.
- Confirm authors, course details and the paper-length limit.
- Review the AI-use declaration.
- Review the repository before making it public. See [COURSE.txt](COURSE.txt).

This repository is private during drafting. Code and data licensing must be settled before public release; third-party materials retain their original terms.
