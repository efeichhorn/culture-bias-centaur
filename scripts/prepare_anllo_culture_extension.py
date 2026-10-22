"""Freeze and validate Anllo's explicit-lottery responses for culture adaptation.

No model fitting is performed here. Each response is an original human choice;
the mirrored model prompts in items.json do not create extra observations.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
from pathlib import Path
import shutil
import subprocess


ROOT = Path(__file__).resolve().parents[1]
RAW_URL = "https://osf.io/download/56gyd/"
RAW_SHA256 = "6ad9ef4a88b81dcb8d97863fbd80ba7fd567609dea6812ef91e6ae44582291ca"
COUNTRY_COUNTS = {
    "argentina": 50, "chile": 54, "china": 49, "france": 54,
    "india": 54, "iran": 49, "israel": 50, "japan": 50,
    "morocco": 51, "russia": 50, "us": 50,
}


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def integer(value: str, field: str) -> int:
    number = float(value)
    require(number.is_integer(), f"Noninteger {field}: {value!r}")
    return int(number)


def normalize_country(country: str) -> str:
    normalized = country.strip().lower()
    return "us" if normalized == "usa" else normalized


def decode_response(row: dict[str, str], item: dict) -> dict:
    """Map raw displayed left/right options to the 10-point semantic choice.

    In the explicit phase, Option1 and Magnitude1 refer to the actual left
    option; Option2 and Magnitude2 refer to the actual right option. This is
    independently checked against the serialized Option1/Option2 arrays.
    """
    require(integer(row["WhichPhase"], "WhichPhase") == 3, "Not an explicit-lottery row")
    condition = integer(row["Condition"], "Condition")
    require(condition == item["condition"], "Condition does not match item")
    require(integer(row["InvertedPosition"], "InvertedPosition") == 0,
            "Unexpected explicit-phase InvertedPosition; re-audit side mapping")
    magnitudes = [integer(row[f"Magnitude{i}"], f"Magnitude{i}") for i in (1, 2)]
    probabilities = [float(row[f"Proba{i}"]) for i in (1, 2)]
    require(sorted(magnitudes) == [1, 10], f"Unexpected magnitudes: {magnitudes}")
    for index in (0, 1):
        require(0 <= probabilities[index] <= 1, "Probability outside [0, 1]")
        option = json.loads(row[f"Option{index + 1}"])
        require(option == [probabilities[index], magnitudes[index]],
                "Serialized lottery disagrees with magnitude/probability columns")
        expected = item["p_10"] if magnitudes[index] == 10 else item["p_1"]
        require(probabilities[index] == expected,
                f"Condition {condition} probability disagrees with frozen protocol")
    choice = integer(row["ChoseLorR"], "ChoseLorR")
    require(choice in (-1, 1), f"Invalid side choice: {choice}")
    selected_index = 0 if choice == -1 else 1
    positive = int(magnitudes[selected_index] == 10)
    # In all eight pairs, the 10-point lottery has greater expected value.
    require(integer(row["ChoseGoodorBad"], "ChoseGoodorBad") == positive,
            "Reconstructed choice disagrees with released ChoseGoodorBad")
    repeat = integer(row["TrialPerCond"], "TrialPerCond") - 1
    require(repeat in range(4), f"Invalid within-pair repetition: {repeat}")
    return {
        "participant_id": row["Participant.Private.ID"],
        "country": normalize_country(row["Country"]),
        "item_id": str(condition),
        "response_positive": positive,
        "repeat_id": repeat,
        "positive_on_left": int(magnitudes[0] == 10),
        "trialnum": integer(row["TrialNumber"], "TrialNumber"),
    }


def lottery_description(magnitude: int, probability: float) -> str:
    unit = "point" if magnitude == 1 else "points"
    return (f"{magnitude} {unit} with a {100 * probability:g}% chance; "
            f"0 points with a {100 * (1 - probability):g}% chance")


def make_prompt(item: dict, positive_is_a: bool) -> str:
    high = lottery_description(10, item["p_10"])
    low = lottery_description(1, item["p_1"])
    option_a, option_b = (high, low) if positive_is_a else (low, high)
    return (
        "You are choosing between two lotteries. Each lottery pays the stated "
        "number of points with the stated probability and otherwise pays 0 points.\n"
        f"A: {option_a}.\n"
        f"B: {option_b}.\n"
        "Which lottery do you choose? Respond with A or B.\n"
        "You choose <<"
    )


def prepare(raw_path: Path, output_dir: Path, protocol_path: Path) -> dict:
    protocol = json.loads(protocol_path.read_text())
    spec = protocol["datasets"]["anllo"]
    require(spec["raw_url"] == RAW_URL and spec["raw_sha256"] == RAW_SHA256,
            "Protocol source changed; explicitly audit before preparing")
    raw_hash = file_sha256(raw_path)
    require(raw_hash == RAW_SHA256, "Raw source hash differs from frozen protocol")
    item_specs = {item["condition"]: item for item in spec["lottery_items"]}
    require(set(item_specs) == set(range(5, 13)), "Expected eight complete item families")
    responses: list[dict] = []
    participant_country: dict[str, str] = {}
    total_rows = 0
    with raw_path.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            total_rows += 1
            person = row["Participant.Private.ID"]
            country = normalize_country(row["Country"])
            require(person != "", "Missing Participant.Private.ID")
            require(participant_country.setdefault(person, country) == country,
                    f"Country changes within person {person}")
            if integer(row["WhichPhase"], "WhichPhase") != 3:
                continue
            condition = integer(row["Condition"], "Condition")
            require(condition in item_specs, "Unexpected explicit-lottery condition")
            responses.append(decode_response(row, item_specs[condition]))
    require(total_rows == 160255, f"Unexpected raw row count: {total_rows}")
    require(len(responses) == 17952, f"Unexpected lottery row count: {len(responses)}")
    require(Counter(participant_country.values()) == COUNTRY_COUNTS,
            "Participant country counts differ from frozen main sample")
    per_pair: dict[tuple[str, str], list[dict]] = defaultdict(list)
    per_person_trials: dict[str, list[int]] = defaultdict(list)
    strict_failures = set()
    for response in responses:
        person = response["participant_id"]
        per_pair[(person, response["item_id"])].append(response)
        per_person_trials[person].append(response["trialnum"])
        if response["item_id"] in ("7", "12") and not response["response_positive"]:
            strict_failures.add(person)
    require(set(per_person_trials) == set(participant_country),
            "A source participant lacks explicit-lottery responses")
    for person in participant_country:
        # Israel uses source trial numbers 1..109 with gaps. Completeness is
        # determined by the eight item families and their four repeats below.
        require(len(per_person_trials[person]) == 32
                and len(set(per_person_trials[person])) == 32
                and min(per_person_trials[person]) >= 1,
                f"Invalid or duplicated source trial numbers for {person}")
        for condition in item_specs:
            pair = per_pair[(person, str(condition))]
            require(sorted(r["repeat_id"] for r in pair) == list(range(4)),
                    f"Incomplete/duplicated repetitions for {person}, condition {condition}")
            require(sum(r["positive_on_left"] for r in pair) == 2,
                    f"Unbalanced source side order for {person}, condition {condition}")
    require(len(strict_failures) == 124, "Strict-filter exclusion count disagrees with paper")
    responses.sort(key=lambda r: (r["country"], r["participant_id"], int(r["item_id"]), r["repeat_id"]))
    participants = [
        {"participant_id": person, "country": country,
         "strict_filter_pass": int(person not in strict_failures)}
        for person, country in sorted(participant_country.items(), key=lambda p: (p[1], p[0]))
    ]
    items = [
        {
            "item_id": str(condition), "stratum": item["type"],
            "prompt_ab": make_prompt(item, True), "prompt_ba": make_prompt(item, False),
            "positive_response": "10_point",
            "source_details": {
                "WhichPhase": 3, "Condition": condition, "p_10": item["p_10"],
                "p_1": item["p_1"], "positive_magnitude": 10, "negative_magnitude": 1,
                "otherwise_points": 0, "repetitions_per_person": 4,
                "heldout_fold": next(i for i, fold in enumerate(spec["heldout_condition_folds"])
                                     if condition in fold),
            },
        }
        for condition, item in sorted(item_specs.items())
    ]
    output_dir.mkdir(parents=True, exist_ok=True)
    archived_raw = output_dir / "raw" / "WEIRD__CDep_NoExclusions.csv"
    archived_raw.parent.mkdir(parents=True, exist_ok=True)
    if archived_raw.resolve() != raw_path.resolve():
        shutil.copyfile(raw_path, archived_raw)
    write_json(output_dir / "items.json", items)
    write_csv(output_dir / "responses.csv", responses)
    write_csv(output_dir / "participants.csv", participants)
    country_summary = {}
    for country in sorted(COUNTRY_COUNTS):
        subset = [r for r in responses if r["country"] == country]
        country_summary[country] = {
            "participants": COUNTRY_COUNTS[country], "responses": len(subset),
            "strict_filter_pass": sum(p["strict_filter_pass"] for p in participants if p["country"] == country),
            "mean_10_point_choice": sum(r["response_positive"] for r in subset) / len(subset),
            "mean_10_point_choice_tradeoff": mean_in_stratum(subset, item_specs, "tradeoff"),
            "mean_10_point_choice_dominance": mean_in_stratum(subset, item_specs, "dominance"),
        }
    manifest = {
        "dataset": "anllo", "citation": spec["citation"], "raw_url": RAW_URL,
        "raw_sha256": raw_hash, "raw_file": str(archived_raw.relative_to(output_dir)),
        "protocol_sha256": file_sha256(protocol_path),
        "preparation_script_sha256": file_sha256(Path(__file__)),
        "raw_rows": total_rows, "participants": len(participants),
        "response_rows": len(responses), "items": len(items),
        "countries": country_summary, "strict_filter_excluded": len(strict_failures),
        "strict_filter_pass": len(participants) - len(strict_failures),
        "primary_filter": "All 561 people retained; strict_filter_pass is evaluation-only sensitivity.",
        "strict_filter_rule": "Exclude person if any response on condition 7 or 12 chooses the 1-point option.",
        "reconstruction": "ChoseLorR -1 selects Magnitude1 (left), +1 Magnitude2 (right); validate Option1/2 arrays and ChoseGoodorBad.",
        "participant_id_source": "Participant.Private.ID; never ParticipantID",
        "boolean_csv_encoding": "0=false, 1=true",
        "repeat_id": "TrialPerCond minus 1; retain all repeats together",
        "source_trial_number_note": "Israel source TrialNumber values have gaps (range 1..109); all 32 responses are present. Completeness is checked using condition and TrialPerCond.",
        "prompt_mapping": "ab: A=10-point, B=1-point; ba reverses; no history, feedback or country labels",
        "human_response_budget": "Mirrored prompts do not duplicate responses; 32 genuine choices per person.",
        "heldout_condition_folds": spec["heldout_condition_folds"],
        "output_sha256": {name: file_sha256(output_dir / name)
                          for name in ("items.json", "responses.csv", "participants.csv")},
    }
    write_json(output_dir / "manifest.json", manifest)
    return manifest


def mean_in_stratum(responses: list[dict], item_specs: dict, stratum: str) -> float:
    subset = [r["response_positive"] for r in responses if item_specs[int(r["item_id"])]["type"] == stratum]
    return sum(subset) / len(subset)


def write_json(path: Path, obj: object) -> None:
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n")


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, help="Existing exact raw CSV; otherwise download the frozen public source")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "data_cache/culture_extensions_20261021/anllo")
    parser.add_argument("--protocol", type=Path, default=ROOT / "designs/culture_extension_protocol_20261021.json")
    args = parser.parse_args()
    raw_path = args.raw
    if raw_path is None:
        raw_path = args.output_dir / "raw" / "WEIRD__CDep_NoExclusions.csv"
        raw_path.parent.mkdir(parents=True, exist_ok=True)
        if not raw_path.exists():
            temporary = raw_path.with_suffix(".download")
            subprocess.run(["curl", "--fail", "--location", "--silent", "--show-error",
                            "--max-time", "120", RAW_URL, "--output", str(temporary)], check=True)
            require(file_sha256(temporary) == RAW_SHA256, "Downloaded raw source hash differs from protocol")
            temporary.replace(raw_path)
    manifest = prepare(raw_path, args.output_dir, args.protocol)
    print(json.dumps({"output_dir": str(args.output_dir), **{
        key: manifest[key] for key in ("participants", "response_rows", "items", "strict_filter_excluded", "raw_sha256")
    }}, indent=2))


if __name__ == "__main__":
    main()
