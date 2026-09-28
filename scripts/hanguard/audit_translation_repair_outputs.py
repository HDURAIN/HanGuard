"""Independent CPU readback of production literal restoration and repaired data.

This does not import the merge or dataset-builder verification routines. It
loads no model weights, executes no source text, changes no labels, and writes
only a new audit directory after all readback checks pass.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import sys
import unicodedata

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.hanguard.audit_translation_repair import AUDIT_VERSION, CONTROL_MARKERS, audit_record
from scripts.hanguard.translation_literal_guard import (
    DEFAULT_CONTROL_LITERALS, MARKER_RE, protect_literals, restore_literals,
    suspected_unprotected_code_changes,
)

SPLITS = ("train", "validation", "test")
COMPLETE_STOPS = {"stop", "eos", "eos_token", "end_of_text"}


def sha(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def file_sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def normalize(text):
    text = unicodedata.normalize("NFKC", text).casefold()
    return re.sub(r"\s+", "", "".join(c for c in text if unicodedata.category(c) != "Cf"))


def require(condition, message):
    if not condition:
        raise ValueError(message)


def jsonl(path):
    records = []
    with Path(path).open() as stream:
        for number, line in enumerate(stream, 1):
            if line.strip():
                item = json.loads(line)
                require(isinstance(item, dict), f"Non-object JSONL record: {path}:{number}")
                records.append(item)
    return records


def _validate_manifest(source, spans):
    require(isinstance(spans, list), "Missing literal manifest")
    previous_end = 0
    for span in spans:
        require(isinstance(span, dict), "Non-object literal span")
        start, end, text = span.get("start"), span.get("end"), span.get("text")
        require(type(start) is int and type(end) is int and previous_end <= start < end <= len(source),
                "Literal manifest positions are invalid or overlap")
        require(isinstance(text, str) and source[start:end] == text and sha(text) == span.get("sha256"),
                "Literal manifest text/hash differs from the exact original English")
        previous_end = end


def verify_raw(mapping: pd.DataFrame, raw_records) -> tuple[dict, dict, set]:
    """Replay each full raw translation; retained hard failures are not successes."""
    require(mapping.base_id.is_unique, "Duplicate source-mapping base_id")
    require(mapping.original_prompt.map(sha).eq(mapping.english_text_sha256).all(), "Mapping English hash mismatch")
    expected = mapping.drop_duplicates("english_text_sha256").set_index("english_text_sha256")
    indexed, hard, literal_hard, flags = {}, set(), set(), Counter()
    for raw in raw_records:
        key, source = raw.get("original_english_sha256"), raw.get("original_english")
        require(isinstance(source, str) and sha(source) == key, "Raw English text/hash mismatch")
        require(key in expected.index and source == expected.at[key, "original_prompt"], "Unknown or mismatched raw English")
        require(key not in indexed, "Duplicate raw English identity")
        _validate_manifest(source, raw.get("literal_spans"))
        require(isinstance(raw.get("raw_translation"), str), "Missing full raw model output")
        require(isinstance(raw.get("translation"), str), "Missing restored translation")
        restored, recomputed = restore_literals(raw["raw_translation"], raw["literal_spans"])
        require(restored.encode("utf-8") == raw["translation"].encode("utf-8"),
                f"Literal-restoration replay differs from recorded translation: {key}")
        require(recomputed == raw.get("literal_issues"), f"Recorded/replayed literal issues disagree: {key}")
        code_issues = suspected_unprotected_code_changes(source, restored, raw["literal_spans"])
        require(code_issues == raw.get("unprotected_code_issues"), f"Recorded/replayed unprotected code issues disagree: {key}")
        initial = audit_record(dict(raw, base_id=expected.at[key, "base_id"], split=expected.at[key, "split"]))
        hard_here = initial["status"] == "hard_fail" or str(raw.get("finish_reason", "")).casefold() not in COMPLETE_STOPS
        if "status" in raw and str(raw["status"]).casefold() not in {"ok", "success", "complete", "completed"}:
            hard_here = True
        for field in ("literal_issues", "unprotected_code_issues", "generation_issues"):
            require(isinstance(raw.get(field), list), f"Missing explicit {field} list")
            for issue in raw[field]:
                require(isinstance(issue, dict) and issue.get("severity") in {"review", "hard_fail"}
                        and isinstance(issue.get("code"), str), f"Malformed {field} entry")
                flags[field + ":" + issue["code"]] += 1
                if issue["severity"] == "hard_fail":
                    hard_here = True
                    if field == "literal_issues":
                        literal_hard.add(key)
        if hard_here:
            hard.add(key)
        indexed[key] = raw
    require(set(indexed) == set(expected.index), "Production raw records do not exactly cover mapped English")
    summary = dict(unique_english=len(indexed), expanded_mapping_rows=len(mapping),
                   literal_restoration_byte_exact=True, literal_issue_lists_exact=True,
                   unprotected_code_issue_lists_exact=True, source_manifest_and_hash_verified=True,
                   raw_hard_fail_unique_english=len(hard), literal_hard_fail_unique_english=len(literal_hard),
                   raw_issue_occurrences=dict(flags), audit_version=AUDIT_VERSION)
    return summary, indexed, hard


def _check_corrected_literals(source, corrected, spans, special_literals):
    controls = set(DEFAULT_CONTROL_LITERALS) | set(CONTROL_MARKERS) | set(special_literals)
    controls.update(s["text"] for s in spans if s.get("kind") == "tokenizer_control_literal")
    _, source_spans = protect_literals(source, extra_literals=controls)
    _, target_spans = protect_literals(corrected, extra_literals=controls)
    require([s["text"] for s in source_spans] == [s["text"] for s in target_spans],
            "Corrected protected literals differ in content, count or order")
    for literal in {s["text"] for s in spans} | {s["text"] for s in source_spans} | controls:
        require(source.count(literal) == corrected.count(literal), "Corrected literal/control occurrence count changed")
    patterns = [MARKER_RE, re.compile(r"\[\[HG_LITERAL_[^\s\]\r\n]*(?:\]\]?)?"),
                re.compile(r"(?<![A-Za-z0-9_])(?:NAME|PERSON|USER|TARGET)_[0-9]+(?![A-Za-z0-9_])|\{\{[^{}\n]+\}\}|(?<!\{)\{[A-Z][A-Z_0-9]*\}(?!\})|\[[A-Z][A-Z_0-9]{1,30}\]"),
                re.compile(r"<[|｜][^<>\r\n]*[|｜]>|\[/?INST\]")]
    for pattern in patterns:
        require(Counter(pattern.findall(source)) == Counter(pattern.findall(corrected)), "Correction changed/added a placeholder/control string")


def verify_merged(mapping, merged_records, raw_index, raw_hard, *, special_literals=()):
    indexed = {}
    mapping_by_id = mapping.set_index("base_id")
    corrections, effective_by_english = {}, {}
    for row in merged_records:
        key = row.get("base_id")
        require(key in mapping_by_id.index and key not in indexed, "Unknown or duplicate merged base_id")
        original = mapping_by_id.loc[key]
        english_hash = original.english_text_sha256
        raw = raw_index[english_hash]
        for name, expected in (("split", original.split), ("source_row", int(original.source_row)),
                               ("old_prompt", original.prompt), ("original_english", original.original_prompt),
                               ("original_english_sha256", english_hash)):
            require(row.get(name) == expected, f"Merged immutable lineage differs: {key}:{name}")
        require(row.get("raw_generation_record") == raw, f"Archived raw generation record differs: {key}")
        require(row.get("model_translation") == raw["translation"], f"Original model translation lost: {key}")
        require(row.get("model_translation_sha256") == sha(raw["translation"]), "Model translation SHA mismatch")
        require(isinstance(row.get("translation"), str) and row.get("translation_sha256") == sha(row["translation"]), "Effective translation SHA mismatch")
        require(row.get("audit_version") == AUDIT_VERSION, "Merged/current QA versions differ")
        original_audit = row.get("original_model_audit") or {}
        require(original_audit.get("translation_sha256") == sha(raw["translation"])
                and original_audit.get("source_sha256") == english_hash, "Original model audit bound to wrong text")
        require(row.get("status") in {"ok", "review", "error"}, "Invalid merged translation status")
        override = row.get("review_override")
        is_corrected = bool(override and override.get("decision") == "correct")
        if override:
            require(override.get("original_english_sha256") == english_hash
                    and override.get("translation_sha256") == sha(raw["translation"]), "Review is not bound to source and model text")
            require(override.get("basis") == "bilingual_text_review" and override.get("label_blind") is True
                    and isinstance(override.get("reviewer"), str) and override["reviewer"].strip()
                    and isinstance(override.get("reason"), str) and override["reason"].strip(), "Review provenance is incomplete")
        if english_hash in raw_hard:
            require(row["status"] == "error" and not is_corrected, "A raw hard failure bypassed quarantine")
        if is_corrected:
            require(english_hash not in raw_hard and row["status"] == "ok", "Correction bypassed an incomplete or hard-failed raw output")
            require(override.get("corrected_translation") == row["translation"] and row["translation"] != raw["translation"], "Correction text/source mismatch")
            require(row.get("translation_source") == "recorded_bilingual_correction"
                    and row.get("correction_source") == "recorded_bilingual_review", "Correction is disguised as model output")
            _check_corrected_literals(original.original_prompt, row["translation"], raw["literal_spans"], special_literals)
            corrected_audit = row.get("corrected_translation_audit") or {}
            require(corrected_audit.get("translation_sha256") == sha(row["translation"])
                    and corrected_audit.get("source_sha256") == english_hash
                    and corrected_audit.get("status") in {"clean", "review"}
                    and corrected_audit.get("literal_preservation_verified") is True,
                    "Corrected audit missing or bound to the wrong text")
            replay = audit_record(dict(raw, base_id=key, split=original.split, translation=row["translation"]))
            require(replay["status"] != "hard_fail", "Corrected text fails independent mandatory QA")
            corrections[english_hash] = override
        else:
            require(row["translation"] == raw["translation"] and row.get("translation_source") == "model"
                    and row.get("correction_source") == "none", "Unrecorded translation modification")
        identity = (row["translation"], row["status"], json.dumps(override, sort_keys=True))
        require(english_hash not in effective_by_english or effective_by_english[english_hash] == identity,
                "Same English identity received inconsistent translation/review decisions")
        effective_by_english[english_hash] = identity
        indexed[key] = row
    require(set(indexed) == set(mapping_by_id.index), "Merged fanout does not cover each mapped base_id once")
    return dict(merged_rows=len(indexed), recorded_correction_unique_english=len(corrections),
                recorded_correction_rows=sum(bool(r.get("review_override") and r["review_override"]["decision"] == "correct") for r in indexed.values()),
                same_english_review_decisions_consistent=True, original_model_and_effective_text_hashes_verified=True), indexed


def verify_dataset(originals, archive, splits, quarantine, merged_index, tokenizer, max_tokens=None):
    require(max_tokens is None or (type(max_tokens) is int and max_tokens > 0), "Invalid explicit length-cap policy")
    original = pd.concat(originals.values(), ignore_index=True).set_index("base_id").sort_index()
    require(original.index.is_unique and archive.base_id.is_unique, "Duplicate original/archive base_id")
    saved = archive.set_index("base_id").sort_index()
    require(set(saved.index) == set(original.index), "Full archive does not exactly preserve original population")
    stable = [c for c in ("split", "group_id", "source", "source_row", "prompt_harm_label", "category_id",
                          "source_file", "source_unit", "base_normalized", "category_label", "text_form", "template_id") if c in original]
    for column in stable:
        require(saved[column].eq(original[column]).all(), "Original stable field changed: " + column)
    require(saved.old_prompt.eq(original.prompt).all(), "Archived original Chinese prompt changed")
    target_ids = set(original[original.source.eq("wildguard_zh")].index)
    require(target_ids == set(merged_index), "WildGuard translation coverage differs from source data")
    for key, row in merged_index.items():
        stored = saved.loc[key]
        require(stored.prompt == row["translation"] and stored.original_english == row["original_english"], "Archive full effective text differs from merged record")
        require(stored.old_prompt == row["old_prompt"] and stored.source_row == row["source_row"]
                and stored.split == row["split"], "Merged/source mapping lineage differs from original archive")
        require(json.loads(stored.translation_metadata_json) == row, "Archive lost merged revision/generation evidence")
        require(stored.model_translation == row["model_translation"]
                and stored.model_translation_sha256 == row["model_translation_sha256"]
                and stored.translation_source == row["translation_source"]
                and stored.correction_source == row["correction_source"]
                and stored.reviewer == (row.get("review_override") or {}).get("reviewer", ""), "Archive correction provenance differs")
    other = original.source.ne("wildguard_zh")
    require(saved.loc[other, "prompt"].eq(original.loc[other, "prompt"]).all(), "Non-WildGuard source text changed")
    require(saved.normalized_prompt.eq(saved.prompt.map(normalize)).all(), "Stale repaired normalization")
    require(saved.sample_id.eq(pd.Series({key: sha(key + ":" + text) for key, text in saved.prompt.items()})).all(), "Stale repaired sample ID")
    tokens = []
    for start in range(0, len(archive), 512):
        batch = archive.prompt.iloc[start:start + 512].tolist()
        tokens.extend(len(ids) for ids in tokenizer(batch, add_special_tokens=False, truncation=False)["input_ids"])
    require(tokens == archive.prompt_tokens.tolist(), "Stored token counts differ from full untruncated tokenizer readback")
    require(archive.prompt_over_legacy_limit.eq(archive.prompt_tokens.gt(370)).all(), "Legacy 370-token annotation is stale")
    accepted = pd.concat(splits.values(), ignore_index=True)
    require(accepted.base_id.is_unique and quarantine.base_id.is_unique, "Candidate/quarantine duplicate base_id")
    kept, removed = set(accepted.base_id), set(quarantine.base_id)
    require(not kept & removed and kept | removed == set(original.index), "Candidate and quarantine do not exactly partition the full archive")
    for split, frame in splits.items():
        require(frame.split.eq(split).all(), "A record moved to a different split")
    for frame in [accepted, quarantine]:
        pd.testing.assert_frame_equal(frame.set_index("base_id").sort_index(), saved.loc[sorted(frame.base_id)], check_dtype=False)
    require(set(saved[saved.quarantine_reason.eq("")].index) == kept, "Saved quarantine reasons disagree with actual partition")
    require(accepted.normalized_prompt.ne("").all(), "Empty text was accepted")
    if max_tokens is not None:
        require(accepted.prompt_tokens.le(max_tokens).all(), "Text exceeds the explicitly configured length cap")
    require(accepted.loc[accepted.source.eq("wildguard_zh"), "translation_status"].eq("ok").all(), "Unapproved translation entered a split")
    over_cap = archive.prompt_tokens.gt(max_tokens) if max_tokens is not None else pd.Series(False, index=archive.index)
    expected_removed = set(archive.loc[archive.normalized_prompt.eq("") | over_cap, "base_id"])
    for key, row in merged_index.items():
        if (row["status"] != "ok" or str(row.get("finish_reason", "")).casefold() not in COMPLETE_STOPS
                or row.get("input_truncated") is True):
            expected_removed.add(key)
            require(key in removed, "A review/error translation bypassed quarantine")
    collisions = archive[archive.normalized_prompt.ne("")].groupby("normalized_prompt", sort=False)
    for _, group in collisions:
        if len(group) > 1:
            expected_removed.update(group.base_id)
            require(set(group.base_id) <= removed, "A repaired Chinese duplicate group was not wholly quarantined")
    english = archive[archive.source.eq("wildguard_zh")].copy()
    english["identity"] = english.original_english.map(normalize)
    for _, group in english.groupby("identity", sort=False):
        if group.split.nunique() > 1 or group.prompt_harm_label.nunique() > 1 or group.category_id.nunique() > 1:
            expected_removed.update(group.base_id)
            require(set(group.base_id) <= removed, "A recovered English cross-split/conflict group was not wholly quarantined")
    require(removed == expected_removed, "Actual quarantine does not exactly match independently recomputed exclusion rules")
    for column in ("base_id", "group_id", "normalized_prompt"):
        require(accepted.groupby(column).split.nunique().le(1).all(), "Remaining cross-split overlap: " + column)
    return dict(original_rows=len(original), full_archive_rows=len(archive), candidate_rows=len(accepted),
                quarantine_rows=len(quarantine), full_population_partition_verified=True,
                quarantine_matches_independent_rules=True,
                max_prompt_tokens=max_tokens, legacy_prompt_limit=370,
                archive_over_legacy_limit=int(archive.prompt_tokens.gt(370).sum()),
                candidate_over_legacy_limit=int(accepted.prompt_tokens.gt(370).sum()),
                quarantine_over_legacy_limit=int(quarantine.prompt_tokens.gt(370).sum()),
                legacy_length_is_annotation_only=max_tokens is None,
                stable_fields_verified=stable, full_untruncated_token_counts_verified=True,
                chinese_duplicate_groups_wholly_quarantined=True, english_cross_split_conflict_groups_wholly_quarantined=True,
                splits={s: len(f) for s, f in splits.items()})


def run(mapping_path, raw_dir, merged_path, source_dir, dataset_dir, tokenizer, out):
    mapping_path, raw_dir, merged_path = Path(mapping_path), Path(raw_dir), Path(merged_path)
    source_dir, dataset_dir, out = Path(source_dir), Path(dataset_dir), Path(out)
    require(not out.exists(), "Independent audit output must be a new directory")
    raw_paths = sorted(raw_dir.glob("raw_*.jsonl"))
    require(bool(raw_paths), "No production raw files")
    paths = [mapping_path, merged_path] + raw_paths
    paths += [source_dir / f"{s}.parquet" for s in SPLITS]
    paths += [dataset_dir / f"{s}.parquet" for s in SPLITS]
    paths += [dataset_dir / "full_repaired_archive.parquet", dataset_dir / "quarantine.parquet",
              dataset_dir / "manifest.json"]
    for raw in raw_paths:
        suffix = raw.stem.removeprefix("raw_")
        paths += [raw.with_name(f"protocol_{suffix}.json"), raw.with_name(f"status_{suffix}.json")]
    snapshots = {str(p.resolve()): file_sha(p) for p in paths}
    guard_path = ROOT / "scripts/hanguard/translation_literal_guard.py"
    guard_hash = file_sha(guard_path)
    raw_records, special_literals, shard_ids, shard_counts = [], set(), set(), set()
    for raw_path in raw_paths:
        suffix = raw_path.stem.removeprefix("raw_")
        protocol = json.loads(raw_path.with_name(f"protocol_{suffix}.json").read_text())
        state = json.loads(raw_path.with_name(f"status_{suffix}.json").read_text())
        require(state.get("state") == "complete", "Production shard is not complete; postpone final audit")
        require(type(protocol.get("pilot")) is int and protocol["pilot"] == 0, "Pilot output is not production data")
        require(protocol.get("mapping_sha256") == file_sha(mapping_path), "Production source mapping hash differs")
        require(protocol.get("literal_guard_sha256") == guard_hash, "Current guard differs from frozen production guard")
        require(str(protocol.get("shard")) == suffix, "Production shard filename/index mismatch")
        shard_ids.add(protocol["shard"])
        shard_counts.add(protocol["shards"])
        special_literals.update(protocol.get("special_literals", []))
        raw_records.extend(jsonl(raw_path))
    require(len(shard_counts) == 1 and shard_ids == set(range(next(iter(shard_counts)))), "Incomplete or inconsistent production shards")
    mapping = pd.read_parquet(mapping_path)
    raw_summary, raw_index, raw_hard = verify_raw(mapping, raw_records)
    merged_summary, merged_index = verify_merged(mapping, jsonl(merged_path), raw_index, raw_hard,
                                                  special_literals=special_literals)
    originals = {s: pd.read_parquet(source_dir / f"{s}.parquet") for s in SPLITS}
    splits = {s: pd.read_parquet(dataset_dir / f"{s}.parquet") for s in SPLITS}
    dataset_manifest = json.loads((dataset_dir / "manifest.json").read_text())
    require(dataset_manifest.get("legacy_prompt_limit") == 370, "Unknown legacy-length annotation protocol")
    dataset_summary = verify_dataset(originals, pd.read_parquet(dataset_dir / "full_repaired_archive.parquet"),
                                     splits, pd.read_parquet(dataset_dir / "quarantine.parquet"), merged_index, tokenizer,
                                     max_tokens=dataset_manifest["max_prompt_tokens"])
    for subset in ("archive", "candidate", "quarantine"):
        require(dataset_manifest["length_statistics"][subset]["over_legacy_limit"] == dataset_summary[subset + "_over_legacy_limit"],
                "Manifest legacy-length counts differ from independent readback")
    for name, digest in snapshots.items():
        require(file_sha(name) == digest, "Input changed during independent audit: " + name)
    result = dict(passed=True, raw=raw_summary, merged=merged_summary, dataset=dataset_summary,
                  paths={"mapping": str(mapping_path.resolve()), "merged": str(merged_path.resolve()),
                         "dataset": str(dataset_dir.resolve()), "source": str(source_dir.resolve()),
                         "production": str(raw_dir.resolve())},
                  input_sha256=snapshots, frozen_guard_sha256=guard_hash, audit_script_sha256=file_sha(__file__),
                  limitations=["This validates text lineage, restoration, recorded review provenance and dataset partition; it does not certify semantic equivalence.",
                               "Recorded Codex bilingual reviews are not human expert gold labels.",
                               "No training ran; revised-data scores must not be subtracted from historical scores as algorithmic gains."])
    out.mkdir(parents=True)
    (out / "verification.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    lines = ["# hanguard 翻译修复独立读回审计", "", "结果：通过。此脚本未调用merge/builder内部验证函数。", "",
             f"逐条重放 {raw_summary['unique_english']:,} 个英文身份的literal恢复；恢复译文字节、literal_issues和unprotected_code_issues均与生产记录一致。",
             f"原生产hard_fail {raw_summary['raw_hard_fail_unique_english']:,} 个英文身份；失败保留原始证据并隔离，未算作翻译成功。",
             f"双语复核纠译 {merged_summary['recorded_correction_unique_english']} 个英文身份/{merged_summary['recorded_correction_rows']} 个base_id；模型原译、修订文本、双哈希及来源标识全部核对。",
             f"完整档案 {dataset_summary['full_archive_rows']:,} 条，候选三集 {dataset_summary['candidate_rows']:,} 条，隔离 {dataset_summary['quarantine_rows']:,} 条；构成互斥且完备的原始记录分区。",
             f"候选划分：{dataset_summary['splits']}。原split/标签/来源/身份保留，全文token计数重算一致。", "",
             f"显式长度上限：{dataset_summary['max_prompt_tokens']}（None表示不按长度隔离）；候选保留超过旧370限制的完整文本{dataset_summary['candidate_over_legacy_limit']}条。",
             "本检查不认证所有译文的语义正确性，Codex双语复核不是真实人工专家金标。未训练或修改历史训练入口。", ""]
    (out / "report.md").write_text("\n".join(lines))
    return result


def write_release_qa(dataset_dir, verification_path, sample_plan_path, *, sample_plan_sha256,
                     qa_evidence_sha256: dict, decision_by: str, reason: str,
                     per_split=24, seed=20260928):
    """Explicit root decision sidecar; normal ``run`` never invokes this.

    Reference hashes and the complete fixed sample's applied, text-bound review
    decisions must match. Existing dataset manifests and statuses stay intact.
    """
    dataset_dir, verification_path, sample_plan_path = map(Path, (dataset_dir, verification_path, sample_plan_path))
    target = dataset_dir / "release_qa.json"
    require(not target.exists(), "Release QA sidecar already exists; do not overwrite a decision")
    require(isinstance(decision_by, str) and decision_by.strip() and isinstance(reason, str) and reason.strip(),
            "Release sidecar requires an explicit decision maker and reason")
    verification_hash = file_sha(verification_path)
    verification = json.loads(verification_path.read_text())
    require(verification.get("passed") is True and Path(verification["paths"]["dataset"]).resolve() == dataset_dir.resolve(),
            "Independent verification did not pass for this dataset")
    for filename, expected in verification["input_sha256"].items():
        require(file_sha(filename) == expected, "A verified input changed before release: " + filename)
    require(file_sha(sample_plan_path) == sample_plan_sha256, "Fixed sample-plan SHA differs")
    require(bool(qa_evidence_sha256), "Release requires hash-bound bilingual review evidence")
    evidence = []
    for filename, expected in qa_evidence_sha256.items():
        require(file_sha(filename) == expected, "Bilingual review evidence SHA differs: " + str(filename))
        if Path(filename).suffix == ".jsonl":
            evidence.extend(jsonl(filename))
        else:
            require(isinstance(json.loads(Path(filename).read_text()), dict), "Review summary must be a JSON object")
    merged = {r["base_id"]: r for r in jsonl(verification["paths"]["merged"])}
    plan = jsonl(sample_plan_path)
    counts = Counter(r["split"] for r in plan)
    require(len({r["base_id"] for r in plan}) == len(plan) and counts == Counter({s: per_split for s in SPLITS}),
            "Fixed random bilingual review sample does not cover the required per-split counts")
    mapping = pd.read_parquet(verification["paths"]["mapping"], columns=["base_id", "split"])
    chosen = set()
    for split in SPLITS:
        ids = mapping.loc[mapping.split.eq(split), "base_id"].tolist()
        chosen.update(sorted(ids, key=lambda value: sha(f"{seed}|{split}|{value}"))[:per_split])
    require(chosen == {r["base_id"] for r in plan}, "Release sample differs from the preregistered fixed-hash sample")
    decisions = Counter()
    for sample in plan:
        require(sample.get("qa_seed") == seed and sample.get("qa_selection_reason") == "fixed_hash_random"
                and sample["base_id"] in merged, "Sample seed/identity mismatch")
        row = merged[sample["base_id"]]
        require(sample["original_english"] == row["original_english"]
                and sample["original_english_sha256"] == row["original_english_sha256"]
                and sample["split"] == row["split"], "Sample's English lineage differs from the verified record")
        review = row.get("review_override")
        require(bool(review) and review.get("basis") == "bilingual_text_review" and review.get("label_blind") is True,
                "A fixed random sample lacks an applied bilingual review decision")
        matches = []
        for item in evidence:
            if item.get("original_english_sha256") != row["original_english_sha256"]:
                continue
            original_text = item.get("original_english", item.get("original_source"))
            if original_text is not None:
                require(original_text == row["original_english"], "Review evidence contains mismatched English text")
            model_text = item.get("model_translation", item.get("translation"))
            model_hash = item.get("translation_sha256", item.get("model_translation_sha256"))
            if model_text is not None:
                require(isinstance(model_text, str) and model_text == row["model_translation"], "Review evidence contains stale model text")
                if model_hash is None:
                    model_hash = sha(model_text)
            corrected = item.get("corrected_translation", item.get("reviewed_translation"))
            if (model_hash == row["model_translation_sha256"] and item.get("decision") == review.get("decision")
                    and (review.get("decision") != "correct" or corrected == row["translation"])):
                matches.append(item)
        require(bool(matches), "Fixed sample review lacks matching full-text-hash evidence")
        decisions[review["decision"]] += 1
    manifest_path = dataset_dir / "manifest.json"
    require(str(manifest_path.resolve()) in verification["input_sha256"], "Dataset manifest was not bound by independent verification")
    result = dict(release_status="qa_complete_with_documented_limitations",
                  decided_at=datetime.now(timezone.utc).isoformat(), decision_by=decision_by, reason=reason,
                  reviewer_type="codex_assistant", expert_human_annotation=False,
                  immutable_build_status_preserved=True, dataset_manifest_sha256=file_sha(manifest_path),
                  independent_verification={"path": str(verification_path.resolve()), "sha256": verification_hash},
                  sample_plan={"path": str(sample_plan_path.resolve()), "sha256": sample_plan_sha256,
                               "seed": seed, "rows": len(plan), "per_split": dict(counts), "decisions": dict(decisions)},
                  qa_evidence_sha256={str(Path(p).resolve()): v for p, v in qa_evidence_sha256.items()},
                  passed_scope=["structural and source-lineage readback", "full-corpus uniform rule screening",
                                "fixed random assistant bilingual review sample", "recorded correction provenance and quarantine"],
                  limitations=["No claim of full-corpus semantic certification or expert human gold labels.",
                               "Rule-flag probes are separate from the fixed random sample and cannot estimate population error rates.",
                               "Use the new data explicitly; rebuild caches and register a new training/evaluation protocol.",
                               "Do not interpret score differences from the historical dataset as algorithmic gains."])
    require(file_sha(verification_path) == verification_hash and file_sha(sample_plan_path) == sample_plan_sha256,
            "Audit/sample evidence changed while preparing the release sidecar")
    for filename, expected in qa_evidence_sha256.items():
        require(file_sha(filename) == expected, "Review evidence changed while preparing release")
    with target.open("x") as stream:
        stream.write(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mapping", type=Path, required=True)
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument("--merged", type=Path, required=True, help="Final translation_results.jsonl")
    parser.add_argument("--source", type=Path, default=ROOT / "data/three_source_original")
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, default=ROOT / "models/Qwen3.5-4B")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True)
    result = run(args.mapping, args.raw_dir, args.merged, args.source, args.dataset, tokenizer, args.output)
    print(json.dumps({key: value for key, value in result.items() if key != "input_sha256"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
