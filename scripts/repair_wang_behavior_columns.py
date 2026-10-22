#!/usr/bin/env python3
"""Repair verified column-label rotation in the specific Wang OSF release.

The source release labels allocation/warmth/competence as Warmth/Competence/Token.
Correction is gated on its known SHA256 and exact raw-questionnaire matches in
all eight samples. Original source files are never modified. No value-based
guessing or automatic schema detection is used for correction.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import subprocess
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

KNOWN_CSV_SHA256 = "071d30ced6b923cae8b1953fdb1bb17c0b2403a7896031a578d4ebf76c8d8be9"
KNOWN_XLSX_SHA256 = "db44bc4940b7860c1188f525c167555530222e7311bbe17770f08411a18723d6"
EXPECTED_SAMPLES = {
    "US-G1": (1, 0, 543), "CN-G1": (1, 1, 501),
    "US-G2": (2, 0, 828), "CN-G2": (2, 1, 800),
    "US-T1": (3, 0, 546), "CN-T1": (3, 1, 529),
    "US-T2": (4, 0, 825), "CN-T2": (4, 1, 800),
}
GIVING_RAW_URLS = {
    "US-G1": "https://osf.io/download/6840513aa24576f776a4dd52/",
    "CN-G1": "https://osf.io/download/fstpb/",
    "US-G2": "https://osf.io/download/68405140e049b869fd85b4b1/",
    "CN-G2": "https://osf.io/download/6840512219d4bd897ba4db2b/",
}
CORRECTED_FROM_SOURCE = {"Token": "Warmth", "Warmth": "Competence", "Competence": "Token"}


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def verified_source(path: Path) -> pd.DataFrame:
    data = path.read_bytes()
    expected = KNOWN_CSV_SHA256 if path.suffix.lower() == ".csv" else KNOWN_XLSX_SHA256
    if sha256(data) != expected:
        raise ValueError("Source SHA256 is not the independently verified affected release; refusing rotation.")
    return pd.read_csv(io.BytesIO(data)) if path.suffix.lower() == ".csv" else pd.read_excel(io.BytesIO(data))


def ensure_not_mislabeled_source(path: str | Path) -> None:
    """Reject the known bad release before ordinary training or evaluation.

    An unknown hash is not certified by this guard; callers still need their
    normal schema/provenance checks. This guard only blocks confirmed bad files.
    """
    if sha256(Path(path).read_bytes()) in {KNOWN_CSV_SHA256, KNOWN_XLSX_SHA256}:
        raise ValueError("Known mislabeled Wang source: Token contains competence ratings. Use the verified corrected data artifact.")


def audit_raw_sample(primary: pd.DataFrame, raw_bytes: bytes, sample: str, url: str,
                     groups: pd.DataFrame) -> dict:
    """Compare all 75 values per cleaned person with released raw questionnaires."""
    dataset, nation, expected_count = EXPECTED_SAMPLES[sample]
    raw = pd.read_excel(io.BytesIO(raw_bytes))
    raw.columns = [str(c).replace(" ", "") for c in raw.columns]
    if sample == "CN-G2":
        # This specific survey exports native token counts and randomized rating
        # positions. Reconstruct semantic target order using the Chinese labels.
        columns = [f"dictator_{i}" for i in range(1, 26)]
        allocation = raw[columns].apply(pd.to_numeric, errors="coerce").to_numpy() * 5
        target_labels = groups.set_index("No.").Chinese.to_dict()
        warmth, competence = [], []
        for _, row in raw.iterrows():
            w = {str(row[f"hidQ1_{i}"]).strip(): pd.to_numeric(row[f"Q{i}"], errors="coerce") for i in range(1, 26)}
            c = {str(row[f"hidQ2_{i}"]).strip(): pd.to_numeric(row[f"Q{i}x1"], errors="coerce") for i in range(1, 26)}
            warmth.append([w.get(str(target_labels[i]).strip(), np.nan) for i in range(1, 26)])
            competence.append([c.get(str(target_labels[i]).strip(), np.nan) for i in range(1, 26)])
        raw_values = np.column_stack([allocation, np.asarray(warmth), np.asarray(competence)])
        raw_layout = "dictator_1..25 * 5; Q1..25 and Q1x1..25 reordered by hidQ Chinese target labels"
    else:
        columns = ([f"{i}_dictator" for i in range(1, 26)]
                   + [f"Q{i}_1" for i in range(1, 26)]
                   + [f"Q{i}-1_1" for i in range(1, 26)])
        raw_values = raw[columns].apply(pd.to_numeric, errors="coerce").to_numpy()
        raw_layout = "1_dictator..25_dictator; Q1_1..Q25_1; Q1-1_1..Q25-1_1"
    lookup = Counter(tuple(row) for row in raw_values if np.isfinite(row).all())
    literal_lookup = lookup.copy()
    subset = primary.loc[(primary.Dataset == dataset) & (primary.Nation == nation)]
    shifted_matches = 0
    literal_matches = 0
    people = 0
    for _, person in subset.groupby("SubNo"):
        person = person.sort_values("GroupNo")
        if list(person.GroupNo) != list(range(1, 26)):
            raise ValueError("A primary participant lacks exactly the 25 expected targets.")
        shifted = tuple(person[["Warmth", "Competence", "Token"]].to_numpy().T.ravel())
        literal = tuple(person[["Token", "Warmth", "Competence"]].to_numpy().T.ravel())
        if lookup[shifted] > 0:
            shifted_matches += 1
            lookup[shifted] -= 1
        if literal_lookup[literal] > 0:
            literal_matches += 1
            literal_lookup[literal] -= 1
        people += 1
    return dict(sample=sample, dataset=dataset, nation=nation, primary_participants=people,
                expected_participants=expected_count, matched_all_75_values=shifted_matches,
                matched_all_75_literal_header_values=literal_matches, source_url=url,
                source_sha256=sha256(raw_bytes), raw_field_layout=raw_layout,
                matching_preserves_raw_vector_multiplicity=True,
                inference="Released Warmth is allocation percent; Competence is warmth; Token is competence.")


def audit_giving(workspace: Path, audit_dir: Path) -> None:
    primary = verified_source(workspace / "data_cache/dictator_Behavior.csv")
    groups = pd.read_csv(workspace / "data_cache/dictator_GroupNo.csv")
    records = []
    for sample, url in GIVING_RAW_URLS.items():
        raw_bytes = subprocess.run(["curl", "--fail", "--silent", "--show-error", "--location",
                                    "--max-time", "120", url], check=True, capture_output=True).stdout
        record = audit_raw_sample(primary, raw_bytes, sample, url, groups)
        records.append(record)
        print(json.dumps(record), flush=True)
    audit_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "source_primary_csv_sha256": KNOWN_CSV_SHA256,
        "source_primary_xlsx_sha256": KNOWN_XLSX_SHA256,
        "method": "Exact multiset matching of complete 25-allocation + 25-warmth + 25-competence vectors; raw source headers identify constructs.",
        "samples": records,
        "privacy": "Only aggregate matching counts are saved; no raw personal identifiers are retained.",
    }
    (audit_dir / "source_audit_giving.json").write_text(json.dumps(payload, indent=2) + "\n")


def validate_audits(records: list[dict]) -> None:
    by_sample = {r["sample"]: r for r in records}
    if len(records) != len(by_sample) or set(by_sample) != set(EXPECTED_SAMPLES):
        raise ValueError("Exactly one raw source audit is required for each of the eight samples.")
    for sample, (dataset, nation, count) in EXPECTED_SAMPLES.items():
        row = by_sample[sample]
        if (row["dataset"], row["nation"], row["primary_participants"], row["matched_all_75_values"]) != (dataset, nation, count, count):
            raise ValueError(f"Incomplete or mismatched raw-source proof for {sample}.")
        if len(row.get("source_sha256", "")) != 64 or not row.get("source_url", "").startswith("https://osf.io/download/"):
            raise ValueError(f"Missing public source provenance for {sample}.")


def corrected_frame(source: pd.DataFrame) -> pd.DataFrame:
    """Apply the explicit verified rotation to a copy; never infer a schema."""
    result = source.copy()
    for target, original in CORRECTED_FROM_SOURCE.items():
        result[target] = source[original].copy()
    allocation = result.Token.to_numpy(dtype=float)
    if not np.isfinite(allocation).all() or np.any((allocation < 0) | (allocation > 100)) or np.any(allocation % 5):
        raise ValueError("Corrected allocation percentages must lie on the 0,5,...100 lattice.")
    return result


def repair(source_path: Path, output_path: Path, audit_paths: list[Path]) -> dict:
    if source_path.resolve() == output_path.resolve():
        raise ValueError("Correction must not overwrite the original source file.")
    source = verified_source(source_path)
    audit_payloads = [json.loads(p.read_text()) for p in audit_paths]
    records = [row for payload in audit_payloads for row in payload["samples"]]
    validate_audits(records)
    for sample, (dataset, nation, expected_count) in EXPECTED_SAMPLES.items():
        subset = source.loc[(source.Dataset == dataset) & (source.Nation == nation)]
        if len(subset) != 25 * expected_count or subset.SubNo.nunique() != expected_count:
            raise ValueError(f"Unexpected source sample size for {sample}.")
    corrected = corrected_frame(source)
    unmodified = [c for c in source.columns if c not in CORRECTED_FROM_SOURCE]
    pd.testing.assert_frame_equal(source[unmodified], corrected[unmodified])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    corrected.to_csv(output_path, index=False)
    # Re-read the artifact to check actual serialization, not just the in-memory frame.
    restored = pd.read_csv(output_path)
    pd.testing.assert_frame_equal(corrected, restored)
    provenance = {
        "status": "corrected_from_independently_verified_mislabeled_source",
        "source_path": str(source_path.resolve()), "source_sha256": sha256(source_path.read_bytes()),
        "source_xlsx_sha256": KNOWN_XLSX_SHA256,
        "source_download": "https://osf.io/download/wtmpu/",
        "output_path": str(output_path.resolve()), "output_sha256": sha256(output_path.read_bytes()),
        "corrected_column_from_source_column": CORRECTED_FROM_SOURCE,
        "corrected_Token_definition": "Percentage GIVEN for Dataset1,2; percentage TAKEN for Dataset3,4. Divide by5 exactly for response tokens0..20.",
        "source_other_columns_preserved": unmodified,
        "rows": len(corrected), "participants": int(corrected[["Dataset", "Nation", "SubNo"]].drop_duplicates().shape[0]),
        "all_eight_raw_samples_verified": True,
        "source_audits": [{"path": str(p.resolve()), "sha256": sha256(p.read_bytes())} for p in audit_paths],
        "sample_verification": records,
        "prior_results": "Archived LoRA corpora used source Token, which is competence. Training corpora and behavioral results must be regenerated; frozen model prompt probabilities remain reusable.",
    }
    output_path.with_suffix(".provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    return provenance


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--audit-giving", action="store_true")
    args = parser.parse_args()
    audit_dir = args.workspace / "model_results/wang_source_audit_20260930"
    if args.audit_giving:
        audit_giving(args.workspace, audit_dir)
    else:
        provenance = repair(args.workspace / "data_cache/dictator_Behavior.csv",
                            args.workspace / "data_cache/dictator_Behavior_corrected_20260930.csv",
                            [audit_dir / "source_audit_giving.json", audit_dir / "taking_raw_alignment.json"])
        print(json.dumps({k: v for k, v in provenance.items() if k != "sample_verification"}, indent=2))
