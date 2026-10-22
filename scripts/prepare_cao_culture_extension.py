#!/usr/bin/env python3
"""Reconstruct Cao experiment-2 triad choices for the culture extension.

The processed file supplies *membership only*. Binary labels are reconstructed
from raw JSON and checked against the displayed options and original bilingual
stimuli. All source revisions and bytes are pinned; no network is needed after
the first successful preparation. This script does not make training splits.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import html
import io
import json
import re
import subprocess
from collections import Counter
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_COMMIT = "2361e91188684831b666ead902a502df181a3524"
STIMULUS_COMMIT = "c3bf6176ee040e3d090affece67419fda9d0ded2"
DATA_BASE = f"https://raw.githubusercontent.com/anjiecao/CCRR_writeups/{DATA_COMMIT}/"
STIMULUS_BASE = (
    "https://raw.githubusercontent.com/anjiecao/uschinareplication.github.io/"
    f"{STIMULUS_COMMIT}/projects/exp2/js/"
)
SOURCES = {
    "merged_data.csv": {
        "url": DATA_BASE + "data/01_merged_data/exp2/merged_data.csv",
        "sha256": "511c5a683cf4074bf062b8f0a36d2d446264d71499273538881c8db2745f9926",
    },
    "tidy_main.csv": {
        "url": DATA_BASE + "data/03_processed_data/exp2/tidy_main.csv",
        "sha256": "5c2719913f91cdc1b5346be3e7597d63620c0d6b2a5f1046cc86a8f0bcf6a89c",
    },
    "triads_answer.csv": {
        "url": DATA_BASE + "processing_pipelines/helper/exp2_preprocessing/task/triads_answer.csv",
        "sha256": "78cac509ecb6e45a5dfa49733ad5b28e15995382536fe4a9d492b8afde475286",
    },
    "english.js": {
        "url": STIMULUS_BASE + "languages/english.js",
        "sha256": "fe1ebc2409bd6c27d523ba76315402755cdb55d4cd283822389c82895f81dfd2",
    },
    "mandarin.js": {
        "url": STIMULUS_BASE + "languages/mandarin.js",
        "sha256": "2f6c171d1616f13f3f66117700bf42d173178debe459db245e82cfbe7a6335d9",
    },
    "triads.js": {
        "url": STIMULUS_BASE + "tasks/triads.js",
        "sha256": "c012e3c7ca68f4789f4e723fd72e068f2ef82a34b284328eaee0379970026028",
    },
}
# Explicit translation families, checked against both source stimulus arrays.
# A translation is an item variant, never an independent item in a split.
TRANSLATIONS = {
    "teacup": "茶杯", "teacher": "老师", "bowl": "碗", "pigeon": "鸽子",
    "boot": "靴子", "legs": "腿", "goat": "山羊", "bee": "蜜蜂",
    "saw": "锯子", "dress": "连衣裙", "elephant": "大象",
    "notepad": "草稿本", "spoon": "勺子", "airplane": "飞机", "ring": "婚戒",
}
COUNTRIES = {"US": "us", "CN": "china"}
EXPECTED_PARTICIPANTS = {"us": 295, "china": 178}
EXPECTED_RESPONSES = {"us": 4425, "china": 2670}
EXPECTED_POSITIVE = {"us": 3266, "china": 1240}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def ensure_sources(raw_dir: Path, offline: bool = False) -> None:
    raw_dir.mkdir(parents=True, exist_ok=True)
    for filename, source in SOURCES.items():
        path = raw_dir / filename
        if not path.exists():
            require(not offline, f"Missing cached source: {path}")
            temporary = path.with_suffix(path.suffix + ".download")
            try:
                subprocess.run(
                    ["curl", "--fail", "--location", "--silent", "--show-error",
                     "--retry", "2", "--max-time", "120", source["url"],
                     "--output", str(temporary)], check=True,
                )
                require(sha256(temporary) == source["sha256"], f"Hash mismatch: {filename}")
                temporary.replace(path)
            finally:
                temporary.unlink(missing_ok=True)
        require(sha256(path) == source["sha256"], f"Hash mismatch: {filename}")


def retained_participants(path: Path) -> dict[str, str]:
    """Ignore every field except subject/culture/task_name in this artifact.

    The public processed file contains malformed Chinese text and NUL bytes.
    Latin-1 is deliberately used only to preserve the ASCII membership columns;
    neither its item strings nor derived response columns are consulted.
    """
    text = path.read_bytes().decode("latin-1").replace("\x00", "")
    participants: dict[str, str] = {}
    for row in csv.DictReader(io.StringIO(text)):
        if row["task_name"] != "TD":
            continue
        country = COUNTRIES[row["culture"]]
        subject = row["subject"]
        require(subject not in participants or participants[subject] == country,
                f"Inconsistent retained country for {subject}")
        participants[subject] = country
    return participants


def stimulus_array(source: str) -> list[tuple[str, str, str]]:
    """Parse only the literal critical-item array, without executing JavaScript."""
    match = re.search(r"\btriads_items\s*=\s*shuffle\((\[.*?\])\s*\);", source, re.S)
    require(match is not None, "Original triads_items array not found")
    array_text = re.sub(r",\s*]", "]", match.group(1))
    values = json.loads(array_text)
    return [(cue.strip(), options[0].strip(), options[1].strip())
            for cue, options in values]


def load_item_mapping(raw_dir: Path) -> dict[tuple[str, str], dict]:
    with (raw_dir / "triads_answer.csv").open(encoding="utf-8-sig", newline="") as handle:
        answer_rows = list(csv.DictReader(handle))
    keyed = {(row["culture"], row["cue"].strip()): {
        "positive": row["tax_match"].strip(),
        "negative": row["them_match"].strip(),
        "source_positive_untrimmed": row["tax_match"],
    } for row in answer_rows}
    require(len(keyed) == 30, "Expected 30 bilingual answer-key entries")
    english = stimulus_array((raw_dir / "english.js").read_text())
    mandarin = stimulus_array((raw_dir / "mandarin.js").read_text())
    require(len(english) == len(mandarin) == 15, "Expected 15 items per language")
    require({row[0] for row in english} == set(TRANSLATIONS), "Unexpected English items")
    require({row[0] for row in mandarin} == set(TRANSLATIONS.values()), "Unexpected Mandarin items")
    for en, zh in zip(english, mandarin):
        require(TRANSLATIONS[en[0]] == zh[0], "Source translation order disagrees with frozen mapping")
        for culture, entry in (("US", en), ("CN", zh)):
            cue, positive, negative = entry
            key = keyed[(culture, cue)]
            require((positive, negative) == (key["positive"], key["negative"]),
                    f"Original stimuli and answer key disagree for {culture}/{cue}")
            key.update(item_id=en[0], cue=cue, culture=culture)
    return keyed


def plain_prompt(value: str) -> str:
    return html.unescape(re.sub(r"<[^>]+>", "", value)).strip()


def parse_raw_page(row: dict, mapping: dict) -> tuple[list[dict], list[dict]]:
    """Reconstruct a retained participant's page using the recorded question list.

    question_order permutes question *display order*, not the stored mapping of
    prompts to their two options. Options were vertical in the original task.
    """
    cues = [plain_prompt(value) for value in row["prompt"].split(",")]
    options_raw = row["options"].split(",")
    options = [value.strip() for value in options_raw]
    types = row["stimulus_type"].split(",")
    answers = json.loads(row["response"])
    require(len(cues) == len(types) == len(answers), "Question/response counts differ")
    require(len(options) == 2 * len(cues), "Each question must show two options")
    require(len(set(cues)) == len(cues), "Duplicate cue in raw page")
    question_order = [int(value) for value in row["question_order"].split(",")]
    require(sorted(question_order) == list(range(len(cues))), "Invalid question-order permutation")
    critical, excluded = [], []
    for index, (cue, stimulus_type) in enumerate(zip(cues, types)):
        suffix = {"triads": "triads", "filler": "fillers", "catch": "catch"}[stimulus_type]
        response_key = f"{cue}_{suffix}"
        require(response_key in answers, f"Response key missing: {response_key}")
        answer_raw = answers[response_key]
        require(isinstance(answer_raw, str), "Expected textual raw response")
        answer = answer_raw.strip()
        pair = options[2 * index:2 * index + 2]
        require(len(set(pair)) == 2, f"Identical displayed alternatives: {cue}")
        require(answer in pair, f"Response was not a displayed alternative: {cue}/{answer}")
        common = {
            "participant_id": row["subject"], "country": COUNTRIES[row["culture"]],
            "source_culture": row["culture"], "source_cue": cue,
            "source_response_raw": answer_raw, "source_response_trimmed": answer,
            "source_option_first": pair[0], "source_option_second": pair[1],
            "source_trial_index": row["trial_index"],
            "source_question_position": question_order.index(index),
        }
        if stimulus_type != "triads":
            excluded.append({**common, "stimulus_type": stimulus_type})
            continue
        key = mapping[(row["culture"], cue)]
        require(set(pair) == {key["positive"], key["negative"]},
                f"Displayed choices disagree with validated key: {cue}")
        critical.append({
            "participant_id": row["subject"], "country": COUNTRIES[row["culture"]],
            "item_id": key["item_id"], "response_positive": int(answer == key["positive"]),
            "repeat_id": 0,
            # The original task used horizontal:false, so left/right is unknown
            # and not silently equated with the known first/second vertical order.
            "positive_on_left": "",
            "source_positive_first": int(pair[0] == key["positive"]),
            **{k: v for k, v in common.items() if k not in {"participant_id", "country"}},
            "source_response_whitespace_trimmed": int(answer_raw != answer),
            "source_untrimmed_key_match": int(answer_raw == key["source_positive_untrimmed"]),
        })
    return critical, excluded


def reconstruct(raw_dir: Path, participants: dict, mapping: dict) -> tuple[list[dict], list[dict]]:
    critical, excluded = [], []
    with (raw_dir / "merged_data.csv").open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if row["trial_type"] != "triads" or row["subject"] not in participants:
                continue
            require(COUNTRIES[row["culture"]] == participants[row["subject"]],
                    f"Raw and retained countries differ: {row['subject']}")
            page_critical, page_excluded = parse_raw_page(row, mapping)
            critical.extend(page_critical)
            excluded.extend(page_excluded)
    critical.sort(key=lambda row: (row["country"], row["participant_id"], row["item_id"]))
    excluded.sort(key=lambda row: (row["country"], row["participant_id"], row["source_cue"]))
    return critical, excluded


def make_prompt(cue: str, option_a: str, option_b: str) -> str:
    return (
        "Choose the option that is most closely related to the bolded item.\n\n"
        f"**{cue}**\nA: {option_a}\nB: {option_b}\n\nYou choose <<"
    )


def make_items(mapping: dict) -> list[dict]:
    result = []
    for item_id, chinese_cue in TRANSLATIONS.items():
        en, zh = mapping[("US", item_id)], mapping[("CN", chinese_cue)]
        result.append({
            "item_id": item_id, "stratum": "critical", "positive_response": "category",
            "prompt_ab": make_prompt(item_id, en["positive"], en["negative"]),
            "prompt_ba": make_prompt(item_id, en["negative"], en["positive"]),
            "source_details": {
                "english_cue": item_id, "mandarin_cue": chinese_cue,
                "english_category_option": en["positive"],
                "english_relation_option": en["negative"],
                "mandarin_category_option": zh["positive"],
                "mandarin_relation_option": zh["negative"],
                "data_repository_commit": DATA_COMMIT,
                "stimulus_repository_commit": STIMULUS_COMMIT,
                "human_display": "vertical alternatives; randomized order",
                "prompt_mapping": "ab: A category, B relation; ba: A relation, B category",
                "prompt_note": "Primary prompts are shared English with no population labels.",
            },
        })
    return result


def audit(participants: dict, responses: list[dict], excluded: list[dict], protocol: dict) -> dict:
    require(dict(Counter(participants.values())) == EXPECTED_PARTICIPANTS,
            "Retained participant counts differ from frozen protocol")
    require(dict(Counter(row["country"] for row in responses)) == EXPECTED_RESPONSES,
            "Critical response counts differ from frozen protocol")
    positive = Counter()
    for row in responses:
        positive[row["country"]] += row["response_positive"]
    require(dict(positive) == EXPECTED_POSITIVE, "Reconstructed category totals differ from protocol")
    unique = {(row["participant_id"], row["item_id"]) for row in responses}
    require(len(unique) == len(responses), "Repeated participant/item detected")
    require(set(row["participant_id"] for row in responses) == set(participants), "Participant loss")
    for subject in participants:
        require({item for person, item in unique if person == subject} == set(TRANSLATIONS),
                f"Incomplete critical response set: {subject}")
    excluded_counts = Counter((row["participant_id"], row["stimulus_type"]) for row in excluded)
    require(all(excluded_counts[(subject, "filler")] == 15 and
                excluded_counts[(subject, "catch")] == 2 for subject in participants),
            "Expected 15 fillers and two attention checks per retained participant")
    require(set(protocol["datasets"]["cao"]["item_ids"]) == set(TRANSLATIONS), "Protocol item mismatch")
    require(protocol["datasets"]["cao"]["repository_commit"] == DATA_COMMIT, "Protocol revision mismatch")
    changes = [row for row in responses
               if row["response_positive"] != row["source_untrimmed_key_match"]]
    return {
        "participants": dict(Counter(participants.values())),
        "critical_responses": dict(Counter(row["country"] for row in responses)),
        "category_choices": dict(positive),
        "category_proportion": {country: positive[country] / EXPECTED_RESPONSES[country]
                                for country in COUNTRIES.values()},
        "critical_items": len(TRANSLATIONS), "total_critical_responses": len(responses),
        "excluded_response_counts": dict(Counter(row["stimulus_type"] for row in excluded)),
        "whitespace_trimmed_responses": sum(row["source_response_whitespace_trimmed"] for row in responses),
        "classification_changes_vs_untrimmed_raw_key_comparison": len(changes),
        "changed_item_counts": dict(Counter(row["item_id"] for row in changes)),
        "changed_country_counts": dict(Counter(row["country"] for row in changes)),
    }


def write_csv(path: Path, rows: list[dict]) -> None:
    require(bool(rows), f"No rows to save: {path}")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def prepare(output_dir: Path, protocol_path: Path, offline: bool = False) -> dict:
    raw_dir = output_dir / "raw"
    ensure_sources(raw_dir, offline=offline)
    protocol = json.loads(protocol_path.read_text())
    participants = retained_participants(raw_dir / "tidy_main.csv")
    mapping = load_item_mapping(raw_dir)
    responses, excluded = reconstruct(raw_dir, participants, mapping)
    counts = audit(participants, responses, excluded, protocol)
    items = make_items(mapping)
    require("most closely related" in (raw_dir / "english.js").read_text(),
            "Prompt wording is absent from original instructions")
    (output_dir / "items.json").write_text(json.dumps(items, ensure_ascii=False, indent=2) + "\n")
    write_csv(output_dir / "responses.csv", responses)
    write_csv(output_dir / "participants.csv", [
        {"participant_id": subject, "country": country}
        for subject, country in sorted(participants.items(), key=lambda pair: (pair[1], pair[0]))
    ])
    write_csv(output_dir / "excluded_responses.csv", excluded)
    outputs = ["items.json", "responses.csv", "participants.csv", "excluded_responses.csv"]
    manifest = {
        "dataset": "cao", "schema_version": 1,
        "status": "raw reconstruction and validation complete; no training or splits performed",
        "citation": protocol["datasets"]["cao"]["citation"],
        "protocol_sha256": sha256(protocol_path),
        "preparation_script_sha256": sha256(Path(__file__)),
        "sources": SOURCES, "counts": counts,
        "inclusion": "Only authors' retained experiment-2 TD subject IDs, read from processed membership fields.",
        "exclusions": "All 15 fillers and 2 attention checks per person excluded from training; preserved separately.",
        "coding_correction": {
            "method": "Strip surrounding whitespace from raw JSON choices, displayed options, and answer key before exact comparison.",
            "processed_labels_used": False,
            "processed_encoding": "Latin-1 with NUL removal solely for ASCII subject/culture/task membership fields.",
            "note": "The original saw/锯子 category alternatives include a trailing space. The pigeon answer key also has trailing whitespace. Compare normalized responses and keys, never processed binary labels.",
        },
        "option_order": {
            "ab_positive_token": "A", "ba_positive_token": "B",
            "model_order_weight": 0.5,
            "human_orientation": "vertical",
            "positive_on_left": "Empty because original alternatives were vertical; source_positive_first records top/first option.",
        },
        "translation_map": TRANSLATIONS,
        "cautions": [
            "Primary English prompts differ from the Mandarin language of Chinese human trials.",
            "Country labels are metadata and never included in the model prompts.",
            "All translations and option orders belong to the same canonical item family.",
            "Country contrasts do not identify culture separately from language or recruitment.",
        ],
        "output_sha256": {filename: sha256(output_dir / filename) for filename in outputs},
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path,
                        default=REPO_ROOT / "data_cache/culture_extensions_20261021/cao")
    parser.add_argument("--protocol", type=Path,
                        default=REPO_ROOT / "designs/culture_extension_protocol_20261021.json")
    parser.add_argument("--offline", action="store_true")
    args = parser.parse_args()
    manifest = prepare(args.output_dir, args.protocol, args.offline)
    print(json.dumps({"output_dir": str(args.output_dir.resolve()), **manifest["counts"]}, indent=2))


if __name__ == "__main__":
    main()
