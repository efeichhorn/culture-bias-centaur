DATA AND MODELS

Raw responses, participant IDs and weights stay outside Git. Source access terms
still apply. Preparation scripts check the recorded source hashes.

Giving/Taking — Wang et al., https://doi.org/10.1093/pnasnexus/pgag172
Project: https://osf.io/syta3/
Behavior.xlsx: https://osf.io/download/wtmpu/
GroupNo.xlsx: https://osf.io/download/684051ee2434947ecd85b513/
Codebook: https://osf.io/download/684051f8ee16d7e551ab83f8/
The released column labels are rotated. Use the exact download, conversion and
repair commands in REPRODUCING.txt. The two reports in provenance/wang_source_audit/
contain aggregate raw-survey checks for all eight samples, with URLs and hashes.
They contain no participant records. The repair guard requires these reports.

Cao — https://doi.org/10.1037/xge0001559
Data: https://github.com/anjiecao/CCRR_writeups
Stimuli: https://github.com/anjiecao/uschinareplication.github.io
prepare_cao_culture_extension.py downloads the six required files at fixed
commits. It selects Experiment 2, 15 critical category/relation items. Exact
file URLs and hashes are in the script's SOURCES table.

Anllo — https://doi.org/10.1038/s41562-024-01894-9
CSV: https://osf.io/download/56gyd/
prepare_anllo_culture_extension.py downloads it and selects explicit lotteries,
phase 3, conditions 5–12. It retains all 561 people for the frozen audit.

Ruggeri — https://doi.org/10.1038/s41562-022-01392-w
Original test prompts (access required):
https://huggingface.co/datasets/marcelbinz/Psych-101-test
Download prompts_testing_t1.jsonl; preserve its original row order.
Country metadata: https://huggingface.co/datasets/marcelbinz/Psych-201-discrete
Download all four data/train-0000N-of-00004.parquet files (N=0..3), and name
local copies psych201-0.parquet through psych201-3.parquet in --private-dir.
Tokenizer: https://huggingface.co/unsloth/Meta-Llama-3.1-70B-bnb-4bit/resolve/main/tokenizer.json
Save tokenizer.json in the same private directory. Released score arrays:
https://media.githubusercontent.com/media/marcelbinz/Llama-3.1-Centaur-70B/main/results/custom_metrics_full_log_likelihoods_marcelbinz-Llama-3.1-Centaur-70B-adapter.pth
https://media.githubusercontent.com/media/marcelbinz/Llama-3.1-Centaur-70B/main/results/custom_metrics_full_log_likelihoods_unsloth-Meta-Llama-3.1-70B-bnb-4bit.pth
Save these as centaur70b.pth and llama70b.pth in --score-dir.
The supplied analysis matches 1,195 of 1,295 participants across 61 countries.
The remaining metadata need Psych-201-discrete-test access and an explicit
extension of the join; they are not silently included in the current results.

Models used for new training and frozen evaluations:
https://huggingface.co/unsloth/Meta-Llama-3.1-70B-bnb-4bit
https://huggingface.co/marcelbinz/Llama-3.1-Centaur-70B-adapter
After obtaining any required access, cache both with Hugging Face's hf download
command using these repository IDs. The Slurm runners are offline and expect
this cache in HF_HOME. Base Llama evaluation disables the Centaur adapter.
