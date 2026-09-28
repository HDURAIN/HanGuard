"""Merge complete production translations into label-blind QA/build records.

No model is loaded. Pilot shards, missing/duplicate English identities, stale
text-review overrides, and attempts to approve a hard failure are rejected.
Heuristic review flags are pending review, not established semantic errors.
"""
from __future__ import annotations

import argparse
from collections import Counter
import copy
import hashlib
import json
from pathlib import Path
import re
import shutil
import sys
import tempfile

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.hanguard.audit_translation_repair import (
    AUDIT_VERSION, CONTROL_MARKERS, audit_record, deterministic_sample, sha256_text,
)
from scripts.hanguard.translation_literal_guard import (
    DEFAULT_CONTROL_LITERALS, MARKER_RE, protect_literals, suspected_unprotected_code_changes,
)

SPLITS = ("train", "validation", "test")
COMPLETE_STOPS = {"stop", "eos", "eos_token", "end_of_text"}
FORBIDDEN_KEYS = {"label", "labels", "category_id", "category_label", "prompt_harm_label",
                  "original_prompt_harm_label", "probability", "prediction", "predictions",
                  "baseline_probability", "prototype_probability", "classifier_score"}
OVERRIDE_FIELDS = {"original_english_sha256", "translation_sha256", "decision", "reviewer",
                   "reason", "basis", "label_blind", "reviewed_at", "corrected_translation"}
PROTOCOL_FIELDS = ("model", "prompt_sha256", "backend", "max_model_len", "batch_size",
                   "max_batch_tokens", "generator_sha256", "package_versions",
                   "model_metadata_sha256", "drafts_sha256", "shards")


def read_jsonl(path: str | Path) -> list[dict]:
    result = []
    with Path(path).open() as stream:
        for number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Malformed JSONL {path}:{number}") from exc
            if not isinstance(item, dict):
                raise ValueError(f"Expected JSON object at {path}:{number}")
            result.append(item)
    return result


def _file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _forbidden(value) -> set[str]:
    """Reject classifier fields in generation/review records, including nesting."""
    if isinstance(value, dict):
        return (set(value) & FORBIDDEN_KEYS) | set().union(*(_forbidden(v) for v in value.values()), set())
    if isinstance(value, list):
        return set().union(*(_forbidden(v) for v in value), set())
    return set()


def _mapping(mapping: pd.DataFrame) -> pd.DataFrame:
    columns = ["base_id", "split", "source_row", "prompt", "original_prompt", "english_text_sha256"]
    missing = set(columns) - set(mapping)
    if missing:
        raise ValueError(f"Source mapping missing fields {sorted(missing)}")
    if "source" in mapping and not mapping.source.eq("wildguard_zh").all():
        raise ValueError("Translation source mapping must contain only WildGuard rows")
    # Labels present in the lineage table never enter translation QA or samples.
    frame = mapping[columns].copy()
    if frame.empty or frame.isna().any().any():
        raise ValueError("Source mapping is empty or has missing identity/text fields")
    if not frame.base_id.is_unique or not frame.base_id.map(lambda x: isinstance(x, str) and bool(x)).all():
        raise ValueError("Mapping base_id must be unique, nonempty strings")
    if set(frame.split) != set(SPLITS):
        raise ValueError("Mapping must cover train, validation and test")
    if not frame.original_prompt.map(lambda x: isinstance(x, str) and bool(x.strip())).all():
        raise ValueError("Mapped English must be nonempty text")
    if not frame.prompt.map(lambda x: isinstance(x, str)).all():
        raise ValueError("Mapped old Chinese prompt must be text")
    if not frame.original_prompt.map(sha256_text).eq(frame.english_text_sha256).all():
        raise ValueError("Mapping English SHA256 does not match full source text")
    return frame.sort_values(["split", "base_id"]).reset_index(drop=True)


def _raw_index(raw_records, frame: pd.DataFrame) -> dict[str, dict]:
    expected = frame.drop_duplicates("english_text_sha256").set_index("english_text_sha256").original_prompt.to_dict()
    indexed = {}
    required = {"original_english", "original_english_sha256", "translation", "model", "prompt_sha256"}
    for raw in raw_records:
        if not isinstance(raw, dict) or not required.issubset(raw):
            raise ValueError(f"Raw generation record requires {sorted(required)}")
        if _forbidden(raw):
            raise ValueError(f"Classifier labels/predictions prohibited in raw QA input: {sorted(_forbidden(raw))}")
        english = raw["original_english"]
        key = raw["original_english_sha256"]
        if not isinstance(english, str) or sha256_text(english) != key:
            raise ValueError("Raw English source/hash mismatch")
        if key in indexed:
            raise ValueError(f"Duplicate raw English hash (no implicit last-wins): {key}")
        if key not in expected or expected[key] != english:
            raise ValueError(f"Unknown or mismatched raw English source: {key}")
        if not isinstance(raw["translation"], str):
            raise ValueError(f"Raw translation must be a string: {key}")
        if not isinstance(raw["model"], str) or not raw["model"]:
            raise ValueError("Raw model identifier must be a nonempty string")
        if not isinstance(raw["prompt_sha256"], str) or not raw["prompt_sha256"]:
            raise ValueError("Raw prompt SHA256 must be present")
        json.dumps(raw, allow_nan=False)
        indexed[key] = copy.deepcopy(raw)
    missing = set(expected) - set(indexed)
    if missing:
        raise ValueError(f"Incomplete production coverage: {len(missing)} English identities missing")
    panels = {(r["model"], r["prompt_sha256"], r.get("literal_guard_version"), r.get("drafts_sha256"))
              for r in indexed.values()}
    if len(panels) != 1:
        raise ValueError("Mixed model/prompt/literal-guard configurations in production results")
    return indexed


def _overrides(records, indexed: dict[str, dict]) -> dict[str, dict]:
    result = {}
    required = OVERRIDE_FIELDS - {"reviewed_at", "corrected_translation"}
    for item in records:
        if not isinstance(item, dict) or set(item) - OVERRIDE_FIELDS or not required.issubset(item):
            raise ValueError("Text-review override has missing/unsupported fields; no labels or predictions allowed")
        key = item["original_english_sha256"]
        if key not in indexed:
            raise ValueError(f"Override source hash absent from this production batch: {key}")
        if key in result:
            raise ValueError(f"Duplicate or conflicting override for English source: {key}")
        if item["translation_sha256"] != sha256_text(indexed[key]["translation"]):
            raise ValueError(f"Stale override: translation hash differs for {key}")
        if item["decision"] not in {"approve", "reject", "hold", "correct"}:
            raise ValueError("Override decision must be approve, reject, hold or correct")
        if item["decision"] == "correct":
            corrected = item.get("corrected_translation")
            if not isinstance(corrected, str) or not corrected.strip():
                raise ValueError("A correct decision requires the complete nonempty corrected_translation")
            if corrected == indexed[key]["translation"]:
                raise ValueError("A correct decision must change the text; use approve for unchanged text")
        elif "corrected_translation" in item:
            raise ValueError("corrected_translation is only permitted with decision=correct")
        if item["basis"] != "bilingual_text_review" or item["label_blind"] is not True:
            raise ValueError("Override must attest to label-blind bilingual text review")
        if any(not isinstance(item[k], str) or not item[k].strip() for k in ["reviewer", "reason"]):
            raise ValueError("Override requires a reviewer and a substantive text-review reason")
        result[key] = copy.deepcopy(item)
    return result


def _audit_correction(record: dict, corrected: str, special_literals=()) -> dict:
    """Recheck a recorded bilingual edit; never execute source code/instructions.

    Original generation issues remain in the original model audit. New text is
    inspected independently. Protected code/control spans and placeholders are
    immutable in semantic corrections, including their order and count.
    """
    source = record["original_english"]
    manifest = record.get("literal_spans")
    if not isinstance(manifest, list):
        raise ValueError("Recorded correction requires the original literal_spans manifest")
    controls = set(DEFAULT_CONTROL_LITERALS) | set(CONTROL_MARKERS) | set(special_literals)
    previous_end = 0
    for span in manifest:
        if not isinstance(span, dict):
            raise ValueError("Invalid original literal manifest for recorded correction")
        left, right, literal = span.get("start"), span.get("end"), span.get("text")
        if (type(left) is not int or type(right) is not int or not previous_end <= left < right <= len(source)
                or not isinstance(literal, str) or source[left:right] != literal
                or sha256_text(literal) != span.get("sha256")):
            raise ValueError("Original literal manifest does not match full English source")
        previous_end = right
        if span.get("kind") == "tokenizer_control_literal":
            controls.add(literal)
    _, source_spans = protect_literals(source, extra_literals=controls)
    _, corrected_spans = protect_literals(corrected, extra_literals=controls)
    if [s["text"] for s in source_spans] != [s["text"] for s in corrected_spans]:
        raise ValueError("Recorded correction changed, added, removed or reordered protected literals")
    for literal in {s["text"] for s in manifest} | {s["text"] for s in source_spans}:
        if source.count(literal) != corrected.count(literal):
            raise ValueError("Recorded correction changed the occurrence count of a protected literal")
    placeholders = re.compile(r"(?<![A-Za-z0-9_])(?:NAME|PERSON|USER|TARGET)_[0-9]+(?![A-Za-z0-9_])|\{\{[^{}\n]+\}\}|"
                              r"(?<!\{)\{[A-Z][A-Z_0-9]*\}(?!\})|\[[A-Z][A-Z_0-9]{1,30}\]")
    marker_fragments = re.compile(r"\[\[HG_LITERAL_[^\s\]\r\n]*(?:\]\]?)?")
    for pattern in (MARKER_RE, marker_fragments, placeholders):
        if Counter(pattern.findall(source)) != Counter(pattern.findall(corrected)):
            raise ValueError("Recorded correction changed or introduced a placeholder")
    # An incomplete generated marker must not evade the complete-marker regex.
    if source.count("[[HG_LITERAL_") != corrected.count("[[HG_LITERAL_"):
        raise ValueError("Recorded correction introduced or removed a literal placeholder prefix")
    control_pattern = re.compile(r"<[|｜][^<>\r\n]*[|｜]>|\[/?INST\]")
    if Counter(control_pattern.findall(source)) != Counter(control_pattern.findall(corrected)):
        raise ValueError("Recorded correction changed or introduced a control string")
    if any(source.count(control) != corrected.count(control) for control in controls):
        raise ValueError("Recorded correction changed or introduced a tokenizer/control string")
    edited = copy.deepcopy(record)
    edited.update(translation=corrected, literal_issues=[], generation_issues=[],
                  unprotected_code_issues=suspected_unprotected_code_changes(source, corrected, source_spans))
    # The record's original generation status is already verified before this
    # function; original finish metadata does not describe review token generation.
    edited.pop("status", None)
    result = _audit_with_literals(edited)
    result["audit_subject"] = "recorded_bilingual_correction"
    result["literal_preservation_verified"] = True
    if result["status"] == "hard_fail":
        raise ValueError("Corrected translation failed mandatory QA: " +
                         ", ".join(f["code"] for f in result["flags"] if f["severity"] == "hard_fail"))
    return result


def _audit_with_literals(record: dict) -> dict:
    result = audit_record(record)
    for field, prefix in (("literal_issues", "literal"),
                          ("unprotected_code_issues", "unprotected_code"),
                          ("generation_issues", "generation")):
        issues = record.get(field)
        if not isinstance(issues, list):
            issues = [dict(code="invalid_issues", severity="hard_fail",
                           evidence=f"{field}: expected an explicitly recorded list")]
        for issue in issues:
            if (not isinstance(issue, dict) or issue.get("severity") not in {"review", "hard_fail"}
                    or not isinstance(issue.get("code"), str) or not issue["code"]):
                issue = dict(code="invalid_issue", severity="hard_fail", evidence=repr(issue))
            result["flags"].append(dict(code=prefix + ":" + issue["code"], severity=issue["severity"],
                                         evidence=copy.deepcopy(issue.get("evidence")), origin=field))
    if str(record.get("finish_reason", "")).casefold() not in COMPLETE_STOPS and not any(
            f["code"] == "generation_limit_reached" for f in result["flags"]):
        result["flags"].append(dict(code="unverified_generation_finish", severity="hard_fail",
                                     evidence="No supported complete-generation stop reason"))
    # A successful stop does not erase parse errors or unrecognized generation
    # statuses. The generator normally omits status; final QA sets it below.
    if "status" in record and str(record["status"]).casefold() not in {"ok", "success", "complete", "completed"}:
        if not any(f["code"] == "generation_failed" for f in result["flags"]):
            result["flags"].append(dict(code="unverified_generation_status", severity="hard_fail",
                                         evidence=f"Unsupported generation status: {record['status']!r}"))
    result["status"] = ("hard_fail" if any(f["severity"] == "hard_fail" for f in result["flags"])
                        else "review" if result["flags"] else "clean")
    return result


def prepare_records(mapping: pd.DataFrame, raw_records, *, overrides=(),
                    per_split: int = 24, seed: int = 20260928, special_literals=()) -> dict:
    """Pure CPU preparation; file publication additionally checks production protocols."""
    if type(per_split) is not int or per_split < 1 or type(seed) is not int:
        raise ValueError("per_split must be a positive integer and seed must be an integer")
    frame = _mapping(mapping)
    indexed = _raw_index(raw_records, frame)
    reviewed = _overrides(overrides, indexed)
    results, audits = [], []
    unique_audits, corrected_audits = {}, {}
    for row in frame.itertuples():
        key = row.english_text_sha256
        raw = indexed[key]
        record = dict(copy.deepcopy(raw), base_id=row.base_id, split=row.split,
                      source_row=int(row.source_row), old_prompt=row.prompt,
                      translation_sha256=sha256_text(raw["translation"]))
        # Preserve the exact generator dictionary even if contract fields below
        # replace a generation-level status/reason with the final QA decision.
        record["raw_generation_record"] = copy.deepcopy(raw)
        if key not in unique_audits:
            unique_audits[key] = _audit_with_literals(record)
        quality = dict(copy.deepcopy(unique_audits[key]), base_id=row.base_id, split=row.split)
        original_quality = copy.deepcopy(quality)
        effective_quality = quality
        automatic = {"clean": "ok", "review": "review", "hard_fail": "error"}[quality["status"]]
        final = automatic
        reason = ("Uniform full-English retranslation; no heuristic flag; semantic QA still required"
                  if automatic == "ok" else "Translation QA " + quality["status"] + ": " +
                  ", ".join(f["code"] for f in quality["flags"]))
        override = reviewed.get(key)
        correction_source = "none"
        if override:
            if quality["status"] == "hard_fail" and override["decision"] in {"approve", "correct"}:
                raise ValueError(f"Cannot approve or correct a hard-failed translation: {key}")
            final = {"approve": "ok", "reject": "error", "hold": "review", "correct": "ok"}[override["decision"]]
            if quality["status"] == "hard_fail":
                final = "error"  # A hold also cannot soften mandatory quarantine.
            if override["decision"] == "correct":
                if key not in corrected_audits:
                    corrected_audits[key] = _audit_correction(record, override["corrected_translation"], special_literals)
                effective_quality = dict(copy.deepcopy(corrected_audits[key]), base_id=row.base_id, split=row.split)
                record["translation"] = override["corrected_translation"]
                record["translation_sha256"] = sha256_text(record["translation"])
                correction_source = "recorded_bilingual_review"
            reason = f"Recorded bilingual text {override['decision']} by {override['reviewer']}: {override['reason']}"
        record.update(status=final, reason=reason, automatic_qa_status=quality["status"],
                      effective_qa_status=effective_quality["status"],
                      qa_flags=copy.deepcopy(effective_quality["flags"]), audit_version=AUDIT_VERSION,
                      review_override=copy.deepcopy(override), model_translation=raw["translation"],
                      model_translation_sha256=sha256_text(raw["translation"]),
                      translation_source="recorded_bilingual_correction" if correction_source != "none" else "model",
                      correction_source=correction_source, original_model_audit=original_quality,
                      corrected_translation_audit=copy.deepcopy(effective_quality) if correction_source != "none" else None)
        quality.update(final_status=final, override_applied=bool(override),
                       review_override=copy.deepcopy(override), reason=reason,
                       effective_qa_status=effective_quality["status"],
                       effective_translation_sha256=record["translation_sha256"],
                       correction_source=correction_source,
                       corrected_translation_audit=copy.deepcopy(record["corrected_translation_audit"]))
        results.append(record)
        audits.append(quality)
    sampled = deterministic_sample(results, per_split=per_split, seed=seed)
    results_by_id = {r["base_id"]: r for r in results}
    for row in sampled:
        row["translation_sha256"] = sha256_text(row["translation"])
        result = results_by_id[row["base_id"]]
        for field in ("model_translation", "model_translation_sha256", "translation_source", "correction_source",
                      "review_override", "original_model_audit", "corrected_translation_audit"):
            row[field] = copy.deepcopy(result[field])
    templates = {}
    for row in sampled:
        key = row["original_english_sha256"]
        if key in reviewed:
            continue  # Do not present an already adjudicated edit as an unreviewed model translation.
        templates.setdefault(key, dict(original_english_sha256=key,
            translation_sha256=row["model_translation_sha256"], decision="", reviewer="", reason="",
            basis="bilingual_text_review", label_blind=True))
    unique_final = {r["original_english_sha256"]: r["status"] for r in results}
    by_split = {split: dict(Counter(r["status"] for r in results if r["split"] == split)) for split in SPLITS}
    flags_rows = Counter(f["code"] for a in audits for f in a["flags"])
    flags_unique = Counter(f["code"] for a in unique_audits.values() for f in a["flags"])
    summary = dict(rows=len(results), unique_english=len(indexed), full_coverage=True,
                   automatic_status_rows=dict(Counter(a["status"] for a in audits)),
                   automatic_status_unique_english=dict(Counter(a["status"] for a in unique_audits.values())),
                   final_status_rows=dict(Counter(r["status"] for r in results)),
                   final_status_unique_english=dict(Counter(unique_final.values())),
                   final_status_by_split=by_split, flag_occurrences_rows=dict(flags_rows),
                   flag_occurrences_unique_english=dict(flags_unique),
                   literal_hard_fail_unique_english=sum(any(f.get("origin") == "literal_issues" and
                       f["severity"] == "hard_fail" for f in a["flags"]) for a in unique_audits.values()),
                   unprotected_code_hard_fail_unique_english=sum(any(f.get("origin") == "unprotected_code_issues" and
                       f["severity"] == "hard_fail" for f in a["flags"]) for a in unique_audits.values()),
                   generation_hard_fail_unique_english=sum(any(f.get("origin") == "generation_issues" and
                       f["severity"] == "hard_fail" for f in a["flags"]) for a in unique_audits.values()),
                   override_unique_english=len(reviewed),
                   override_decisions=dict(Counter(r["decision"] for r in reviewed.values())),
                   corrected_unique_english=len(corrected_audits),
                   corrected_rows=sum(r["correction_source"] != "none" for r in results),
                   effective_qa_status_rows=dict(Counter(r["effective_qa_status"] for r in results)),
                   sample_seed=seed, requested_sample_per_split=per_split,
                   sampled_rows=len(sampled), sample_by_split=dict(Counter(r["split"] for r in sampled)),
                   audit_version=AUDIT_VERSION, release_status="candidate_pending_bilingual_qa",
                   limitations=["Heuristic review is not a confirmed semantic translation error.",
                                "Clean/ok does not certify semantic equivalence.",
                                "Review rows remain pending and are quarantined by the dataset builder.",
                                "Recorded bilingual overrides are exact-text, label-blind attestations, not label corrections or human gold labels.",
                                "Every hard failure remains quarantined, including after an override hold."])
    return dict(translations=results, audits=audits, summary=summary, sample=sampled,
                override_template=list(templates.values()))


def _production_inputs(mapping_path: Path, raw_directory: Path) -> tuple[list[dict], dict]:
    paths = sorted(raw_directory.glob("raw_*.jsonl"))  # Never recurse into pilot directories.
    if not paths:
        raise ValueError(f"No production raw_*.jsonl files directly under {raw_directory}")
    mapping_sha = _file_hash(mapping_path)
    records, protocols = [], {}
    configuration, shard_ids = None, set()
    hashes = {str(mapping_path.resolve()): mapping_sha}
    for raw_path in paths:
        suffix = raw_path.stem.removeprefix("raw_")
        protocol_path = raw_path.with_name(f"protocol_{suffix}.json")
        if not protocol_path.exists():
            raise ValueError(f"Production protocol missing for {raw_path.name}")
        raw_sha, protocol_sha = _file_hash(raw_path), _file_hash(protocol_path)
        protocol = json.loads(protocol_path.read_text())
        if type(protocol.get("pilot")) is not int or protocol["pilot"] != 0:
            raise ValueError(f"Pilot output is not eligible for production merge: {raw_path}")
        if protocol.get("mapping_sha256") != mapping_sha:
            raise ValueError(f"Protocol belongs to a different source mapping: {protocol_path}")
        missing = set(PROTOCOL_FIELDS) - set(protocol)
        if missing:
            raise ValueError(f"Production protocol missing provenance fields {sorted(missing)}: {protocol_path}")
        for field in ("max_model_len", "batch_size", "max_batch_tokens", "shards"):
            if type(protocol[field]) is not int or protocol[field] < 1:
                raise ValueError(f"Invalid positive protocol integer {field}: {protocol_path}")
        for field in ("model", "prompt_sha256", "backend", "generator_sha256"):
            if not isinstance(protocol[field], str) or not protocol[field].strip():
                raise ValueError(f"Invalid protocol provenance {field}: {protocol_path}")
        for field in ("package_versions", "model_metadata_sha256"):
            if not isinstance(protocol[field], dict) or not protocol[field] or any(
                    not isinstance(v, str) or not v for v in protocol[field].values()):
                raise ValueError(f"Invalid protocol provenance {field}: {protocol_path}")
        drafts_sha = protocol["drafts_sha256"]
        if drafts_sha is not None and (not isinstance(drafts_sha, str) or len(drafts_sha) != 64
                                      or any(c not in "0123456789abcdef" for c in drafts_sha)):
            raise ValueError(f"Invalid draft-batch provenance hash: {protocol_path}")
        shard_id = protocol.get("shard")
        if type(shard_id) is not int or not 0 <= shard_id < protocol["shards"] or str(shard_id) != suffix:
            raise ValueError(f"Protocol shard range/filename mismatch: {protocol_path}")
        if shard_id in shard_ids:
            raise ValueError(f"Duplicate production shard {shard_id}")
        shard_ids.add(shard_id)
        current = {field: protocol[field] for field in PROTOCOL_FIELDS}
        if configuration is None:
            configuration = current
        elif configuration != current:
            differences = [field for field in PROTOCOL_FIELDS if configuration[field] != current[field]]
            raise ValueError(f"Mixed production shard configurations {differences}: {protocol_path}")
        shard = read_jsonl(raw_path)
        for record in shard:
            if any(record.get(k) != protocol[k] for k in ("model", "prompt_sha256", "backend")):
                raise ValueError(f"Raw model/prompt/backend differs from production protocol: {raw_path}")
            if record.get("drafts_sha256") != drafts_sha:
                raise ValueError(f"Raw draft-batch hash differs from production protocol: {raw_path}")
            if drafts_sha is not None and not isinstance(record.get("draft_translation"), str):
                raise ValueError(f"Post-edit record is missing the complete draft text: {raw_path}")
            if drafts_sha is None and "draft_translation" in record:
                raise ValueError(f"Draft text provided without draft-batch provenance: {raw_path}")
            key = record.get("original_english_sha256")
            if not isinstance(key, str) or len(key) != 64 or any(c not in "0123456789abcdef" for c in key):
                raise ValueError(f"Invalid exact-English SHA256 in shard: {raw_path}")
            if int(key[:12], 16) % protocol["shards"] != shard_id:
                raise ValueError(f"English source placed in the wrong production shard: {raw_path}")
        records.extend(shard)
        protocols[str(protocol_path.resolve())] = protocol
        hashes[str(raw_path.resolve())] = raw_sha
        hashes[str(protocol_path.resolve())] = protocol_sha
    if shard_ids != set(range(configuration["shards"])):
        missing = sorted(set(range(configuration["shards"])) - shard_ids)
        raise ValueError(f"Incomplete production shard coverage: missing {missing}")
    return records, dict(input_sha256=hashes, protocols=protocols,
                         production_configuration=configuration,
                         raw_directory=str(raw_directory.resolve()), pilot_outputs_used=False)


def _write_jsonl(path: Path, records) -> None:
    with path.open("x") as stream:
        for record in records:
            stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")


def merge(mapping_path: str | Path, raw_directory: str | Path, out: str | Path,
          *, overrides_path: str | Path | None = None, per_split: int = 24,
          seed: int = 20260928) -> dict:
    mapping_path, raw_directory, out = Path(mapping_path), Path(raw_directory), Path(out)
    names = ("translation_results.jsonl", "qa_results.jsonl", "qa_counts.json", "qa_report.md",
             "qa_sample.jsonl", "review_overrides_template.jsonl", "merge_manifest.json")
    if any((out / name).exists() for name in names):
        raise FileExistsError("Refusing to overwrite existing merged outputs; use a new output directory")
    raw, provenance = _production_inputs(mapping_path, raw_directory)
    if overrides_path:
        provenance["input_sha256"][str(Path(overrides_path).resolve())] = _file_hash(Path(overrides_path))
    overrides = read_jsonl(overrides_path) if overrides_path else []
    # Protocols record every tokenizer special string used by the generator,
    # including added special tokens outside tokenizer.all_special_tokens.
    special_literals = sorted({literal for protocol in provenance["protocols"].values()
                               for literal in protocol.get("special_literals", [])})
    prepared = prepare_records(pd.read_parquet(mapping_path), raw, overrides=overrides,
                               per_split=per_split, seed=seed, special_literals=special_literals)
    summary = prepared["summary"]
    out.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".translation-merge-", dir=out.parent))
    try:
        for name, records in [("translation_results.jsonl", prepared["translations"]),
                              ("qa_results.jsonl", prepared["audits"]),
                              ("qa_sample.jsonl", prepared["sample"]),
                              ("review_overrides_template.jsonl", prepared["override_template"])]:
            _write_jsonl(staging / name, records)
        (staging / "qa_counts.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
        lines = ["# hanguard 翻译修复合并与盲化质量检查", "",
                 f"生产英文身份 {summary['unique_english']:,} 条，展开为 {summary['rows']:,} 个稳定base_id；覆盖所有三集。",
                 "未使用标签或分类预测选译文、挑抽检样本或豁免质量问题；未运行训练或发布数据集。", "",
                 "## 自动检查与最终状态", "",
                 "hard_fail 表示必须隔离的流水线缺陷；review 是待复核的启发式疑点，不等于语义错误。",
                 "clean/ok 也不保证语义等价。builder会隔离review/error；有记录的双语复核仅能解除非hard_fail的待复核状态。", "",
                 "| 统计层次 | 自动检查 | 最终builder状态 |", "|---|---|---|",
                 f"| 唯一英文 | {summary['automatic_status_unique_english']} | {summary['final_status_unique_english']} |",
                 f"| 展开base_id | {summary['automatic_status_rows']} | {summary['final_status_rows']} |", "",
                 "| 原划分 | 最终状态计数 |", "|---|---|"]
        for split in SPLITS:
            lines.append(f"| {split} | {summary['final_status_by_split'][split]} |")
        lines += ["", f"literal_issues中含hard_fail的唯一英文：{summary['literal_hard_fail_unique_english']}。",
                  f"unprotected_code_issues中含hard_fail的唯一英文：{summary['unprotected_code_hard_fail_unique_english']}。",
                  f"generation_issues中含hard_fail的唯一英文：{summary['generation_hard_fail_unique_english']}。",
                  f"已应用有记录的双语复核：{summary['override_unique_english']}个唯一英文，决定分布{summary['override_decisions']}。",
                  f"其中双语纠译{summary['corrected_unique_english']}个唯一英文/{summary['corrected_rows']}个base_id；原模型译文和原自动审计完整保留。", "",
                  "## 固定抽检和有记录的双语复核接口", "",
                  f"固定seed={seed}，按split/base_id的哈希次序每集抽{per_split}条；实际{summary['sample_by_split']}。",
                  "抽检在全部展开记录中进行，不按质量状态过滤；qa_sample.jsonl不含原标签、类别或分类器预测。",
                  "完整原文、旧译文、新译文和双哈希随抽检记录保留。review_overrides_template.jsonl按英文身份去重，空白decision/reviewer/reason必须由审核者填写。", "",
                  "Override字段：original_english_sha256、translation_sha256、decision(approve/reject/hold/correct)、reviewer、reason、basis=bilingual_text_review、label_blind=true；reviewed_at可选，correct必须加完整corrected_translation。",
                  "仅匹配当前英文与当前完整译文的双哈希；失配、重复决定或额外标签/预测字段会拒绝。相同译文在全部base_id上应用同一决定，不能逐split选择性豁免。",
                  "审核者必须仅根据原文和译文解释决定；label_blind是审核者声明，程序不能验证其真实审阅过程。Codex助手复核必须在reviewer如实标注，不能称为人工专家验证或人工金标。",
                  "approve/correct不能豁免原译文任何hard_fail。correct只适用于生成完整的原译文；绑定原英文和原模型译文双哈希，保留model_translation、original_model_audit、复核理由和新译文哈希。",
                  "纠译重新检查全文与保护字面量，不允许增删改代码、占位符、控制串或改变保护片段顺序。新审计hard_fail拒绝；剩余heuristic review保留在corrected_translation_audit，correct决定表明审核者基于双语文本接受新译文，不是规则证明正确。",
                  "修订记录translation_source=recorded_bilingual_correction、correction_source=recorded_bilingual_review，固定抽检同时给出模型原译与修订译文；不把修订伪装成模型原输出。", "",
                  "## 产物与边界", "",
                  "translation_results.jsonl符合builder接口，保留完整原文和译文、原始生成record、literal_issues、unprotected_code_issues、generation_issues、模型元数据、自动QA及有记录的复核决定。",
                  "qa_results.jsonl与qa_counts.json分别保留逐条标记和唯一英文/展开行两种计数；不把重复展开当作独立翻译成功次数。",
                  "merge_manifest.json最后写入，记录全部生产协议、输入及输出哈希。旧pilot不会被递归搜集，pilot协议和不完整覆盖均拒绝。",
                  "若生产协议使用drafts_sha256，两阶段初译内容由generator记录；本脚本只验证其声明和跨分片一致性，不独立读取初译文件认证逐条初译文本。直接翻译不涉及该边界。",
                  "所有产物只是待独立双语QA的候选；本脚本不改动原三集、不切换训练入口。", ""]
        (staging / "qa_report.md").write_text("\n".join(lines))
        manifest = dict(provenance, summary=summary,
                        merge_script_sha256=_file_hash(Path(__file__)),
                        audit_script_sha256=_file_hash(ROOT / "scripts/hanguard/audit_translation_repair.py"),
                        artifact_sha256={p.name: _file_hash(p) for p in sorted(staging.iterdir())})
        (staging / "merge_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
        for name, expected in provenance["input_sha256"].items():
            if _file_hash(Path(name)) != expected:
                raise ValueError(f"Input changed while merging; no outputs committed: {name}")
        out.mkdir(parents=True, exist_ok=True)
        if any((out / name).exists() for name in names):
            raise FileExistsError("Output appeared while merging; refusing overwrite")
        for name in names:  # Manifest last is the completion marker.
            (staging / name).replace(out / name)
    finally:
        shutil.rmtree(staging)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mapping", type=Path, required=True)
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--overrides", type=Path)
    parser.add_argument("--per-split", type=int, default=24)
    parser.add_argument("--seed", type=int, default=20260928)
    args = parser.parse_args()
    print(json.dumps(merge(args.mapping, args.raw_dir, args.output, overrides_path=args.overrides,
                           per_split=args.per_split, seed=args.seed), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
