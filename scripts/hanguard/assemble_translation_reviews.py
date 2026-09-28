"""Assemble exact-text review overrides and a fixed-sample quality comparison.

This only formats recorded assistant reviews; it does not judge translations,
change labels, run inference, or certify the unreviewed corpus.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path


GRADES = {"faithful", "minor_issue", "substantive_issue", "untranslated_or_incomplete", "unclear"}
FIELDS = {"original_english_sha256", "translation_sha256", "decision", "corrected_translation",
          "reviewer", "reason", "basis", "label_blind", "reviewed_at"}


def sha(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def read(path):
    # JSON strings may legally contain U+2028/U+2029. str.splitlines() treats
    # them as record boundaries; a JSONL record ends at a physical file newline.
    with Path(path).open() as stream:
        records = [json.loads(line) for line in stream if line.strip()]
    require(all(isinstance(row, dict) for row in records), f"Expected JSON objects in {path}")
    return records


def require(condition, message):
    if not condition:
        raise ValueError(message)


def assemble(plan_path, review_paths, output):
    output = Path(output)
    require(not output.exists(), f"Refusing to overwrite {output}")
    paths = [Path(plan_path), *map(Path, review_paths)]
    input_hashes = {str(p.resolve()): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    plan = read(plan_path)
    require(len(plan) == 72 and Counter(r["split"] for r in plan) == {"train": 24, "validation": 24, "test": 24},
            "Expected the fixed 72-row, three-split sample")
    require(all(r["qa_seed"] == 20260928 and r["qa_selection_reason"] == "fixed_hash_random" for r in plan),
            "Unexpected QA sampling protocol")
    require(len({r["base_id"] for r in plan}) == 72, "Repeated base_id in fixed sample")
    selected = {r["original_english_sha256"]: r for r in plan}
    require(len(selected) == 72, "Repeated source identities in fixed sample")
    reviews = {}
    overrides = []
    comparisons = []
    for path in review_paths:
        for record in read(path):
            source = record["original_english"]
            key = record["original_english_sha256"]
            model_text = record.get("model_translation", record.get("translation"))
            model_hash = record.get("translation_sha256", record.get("model_translation_sha256"))
            require(isinstance(source, str) and sha(source) == key and isinstance(model_text, str) and sha(model_text) == model_hash,
                    f"Stale source/model text hash in {path}: {key}")
            require(all(record[field] == model_text for field in ("model_translation", "translation") if field in record)
                    and all(record[field] == model_hash for field in ("translation_sha256", "model_translation_sha256") if field in record),
                    f"Conflicting model-text/hash aliases: {key}")
            require(key not in reviews, f"Repeated review: {key}")
            require(record.get("basis") == "bilingual_text_review" and record.get("label_blind") is True,
                    f"Missing text-review attestation: {key}")
            require(all(isinstance(record.get(field), str) and record[field].strip() for field in ("reviewer", "reason")),
                    f"Missing reviewer or review reason: {key}")
            reviews[key] = record
            override = {k: v for k, v in record.items() if k in FIELDS}
            override["translation_sha256"] = model_hash
            require(override["decision"] in {"approve", "correct", "hold", "reject"}, "Unsupported review decision")
            if override["decision"] == "correct":
                require(isinstance(override.get("corrected_translation"), str)
                        and override["corrected_translation"].strip()
                        and override["corrected_translation"] != model_text, "Correction must provide a revised full text")
                if "corrected_translation_sha256" in record:
                    require(record["corrected_translation_sha256"] == sha(override["corrected_translation"]),
                            f"Stale corrected translation hash: {key}")
            else:
                require("corrected_translation" not in override, "Only a correction may supply revised text")
            overrides.append(override)
            if key not in selected:
                continue
            expected = selected[key]
            require(record.get("base_id") == expected["base_id"] and record.get("split") == expected["split"],
                    f"Random sample identity changed: {key}")
            require(source == expected["original_english"] and record.get("old_translation") == expected["old_translation"],
                    f"Random source/old text changed: {key}")
            require(record.get("old_translation_sha256") == sha(expected["old_translation"]),
                    f"Old text hash differs: {key}")
            grades = {
                "old": record.get("old_translation_status", record.get("old_translation_assessment")),
                "model": record.get("model_translation_status", record.get("original_model_translation_assessment")),
                "reviewed": record.get("reviewed_translation_status", record.get("reviewed_translation_assessment")),
            }
            require(all(v in GRADES for v in grades.values()), f"Missing/unknown comparison grade: {key}: {grades}")
            comparisons.append(dict(base_id=expected["base_id"], split=expected["split"],
                original_english_sha256=key, original_source=source, old_translation=expected["old_translation"],
                model_translation=model_text, reviewed_translation=override.get("corrected_translation", model_text),
                decision=override["decision"], reviewer=override["reviewer"], reason=override["reason"],
                quality=grades, evidence_file=str(Path(path).resolve())))
    require(len(comparisons) == 72, f"Fixed bilingual sample is incomplete: {len(comparisons)}/72")
    summary = dict(fixed_random_rows=72, fixed_random_seed=20260928,
        fixed_random_by_split=dict(Counter(r["split"] for r in comparisons)),
        fixed_random_quality={kind: dict(Counter(r["quality"][kind] for r in comparisons)) for kind in ("old", "model", "reviewed")},
        fixed_random_old_to_model_transitions=dict(Counter(f'{r["quality"]["old"]} -> {r["quality"]["model"]}' for r in comparisons)),
        fixed_random_decisions=dict(Counter(r["decision"] for r in comparisons)),
        all_reviewed_unique_sources=len(reviews), all_review_decisions=dict(Counter(r["decision"] for r in reviews.values())),
        additional_nonrandom_reviewed_sources=len(reviews)-72, input_sha256=input_hashes,
        assembler_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        limitations=["Codex assistant bilingual review, not human expert gold annotation.",
                     "The fixed sample is small and review was not blinded to old/new identity.",
                     "Additional regression and flagged examples are not a population sample.",
                     "Corrections include minor wording changes; correction count is not a count of substantive errors.",
                     "No classification accuracy, semantic certification, or model-comparison significance is inferred."])
    require(all(hashlib.sha256(Path(p).read_bytes()).hexdigest() == h for p, h in input_hashes.items()),
            "Review files changed during assembly")
    output.mkdir(parents=True)
    for name, rows in (("review_overrides.jsonl", overrides), ("qa_comparison.jsonl", comparisons)):
        (output / name).write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--review", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(assemble(args.plan, args.review, args.output), ensure_ascii=False, indent=2))
