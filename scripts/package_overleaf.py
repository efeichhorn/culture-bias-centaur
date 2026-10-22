#!/usr/bin/env python3
"""Package only the paper sources and figures for Overleaf."""
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

root = Path(__file__).resolve().parents[1]
paper = root / "paper"
destination = root / "build/culture-bias-centaur-overleaf.zip"
destination.parent.mkdir(parents=True, exist_ok=True)
files = [paper / name for name in (
    "main.tex", "references.bib", "acl2023.sty", "acl_natbib.bst",
    "TEMPLATE.txt",
)]
files += sorted((paper / "sections").glob("*.tex"))
files += sorted((paper / "figures").glob("*.pdf"))
if not all(path.is_file() for path in files):
    raise FileNotFoundError("Paper sources are incomplete")
with ZipFile(destination, "w", compression=ZIP_DEFLATED) as archive:
    for path in files:
        archive.write(path, path.relative_to(paper))
print(destination)
