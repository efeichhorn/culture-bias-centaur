#!/usr/bin/env python3
"""Package the paper's code, prepared data, results, and manuscript sources."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

from prepare_paper_data import prepare

ROOT = Path(__file__).resolve().parents[1]
FOLDERS = {
    "scripts": {".py", ".sh"},
    "src": {".py"},
    "tests": {".py"},
    "slurm": {".sh"},
    "designs": {".json", ".csv"},
    "results": {".csv", ".json", ".txt", ".md"},
    "provenance": {".csv", ".json", ".txt", ".md"},
    "paper": {".tex", ".bib", ".sty", ".bst", ".pdf", ".png", ".svg", ".csv", ".json", ".txt"},
}


def submission_files(root: Path) -> list[Path]:
    files = [root / name for name in (
        "README.md", "REPRODUCING.txt", "Makefile", "requirements.txt",
        "requirements-training.txt", ".gitignore", ".gitattributes",
        ".github/workflows/check.yml", "data/README.md", "data/paper_inputs.zip",
        "data/paper_inputs.manifest.json",
    )]
    for folder, suffixes in FOLDERS.items():
        for path in (root / folder).rglob("*"):
            if not path.is_file() or path.suffix not in suffixes:
                continue
            if path.name in {"main.pdf", "submission_details.tex"}:
                continue
            if any(part.startswith(".") or part == "__pycache__" for part in path.relative_to(root).parts):
                continue
            if path.is_symlink():
                raise ValueError(f"Unexpected symlink: {path}")
            files.append(path)
    for path in files:
        if not path.is_file():
            raise FileNotFoundError(path)
    return sorted(set(files))


def package(root: Path, output: Path) -> int:
    prepare(root / "data/paper_inputs.zip", root / "data/paper_inputs.manifest.json",
            root / "data/workspace", verify_only=True)
    files = submission_files(root)
    for stem in ("finetuning_pipeline", "figure2_distributions_nll", "population_effects_h2_h3",
                 "response_adjustment_comparison", "frozen_sample_fit", "answer_position_control"):
        if root / f"paper/figures/{stem}.pdf" not in files:
            raise FileNotFoundError(f"Missing {stem}.pdf; run make figures first")
    output.parent.mkdir(parents=True, exist_ok=True)
    sums = []
    with ZipFile(output, "w", compression=ZIP_DEFLATED) as archive:
        for path in files:
            name = path.relative_to(root).as_posix()
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            archive.write(path, f"culture-bias-centaur/{name}")
            sums.append(f"{digest}  {name}\n")
        archive.writestr("culture-bias-centaur/SHA256SUMS", "".join(sums))
    with ZipFile(output) as archive:
        for line in sums:
            digest, name = line.rstrip("\n").split("  ", 1)
            if hashlib.sha256(archive.read(f"culture-bias-centaur/{name}")).hexdigest() != digest:
                raise ValueError(f"Package verification failed: {name}")
    output.with_suffix(".zip.sha256").write_text(
        f"{hashlib.sha256(output.read_bytes()).hexdigest()}  {output.name}\n")
    return len(files)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "build/culture-bias-centaur-submission.zip")
    args = parser.parse_args()
    count = package(ROOT, args.output)
    print(f"Verified {count} files in {args.output} ({args.output.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()
