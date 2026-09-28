"""Build a separate, quarantined translation-repair candidate; never retrain.

Every WildGuard row is treated by source membership, independently of labels or
model predictions. Stable identities and split assignments refer to the OLD
records; sample_id identifies the NEW full text. Existing inputs are immutable.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import re
import shutil
import tempfile
from typing import Iterable, Mapping
import unicodedata

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
SPLITS = ("train", "validation", "test")
TARGET_SOURCE = "wildguard_zh"
STABLE_COLUMNS = ("base_id", "group_id", "split", "source", "source_row",
                  "prompt_harm_label", "category_id")
TRUNCATED_FINISH_REASONS = {"length", "max_length", "max_tokens", "max_new_tokens", "token_limit"}
COMPLETE_FINISH_REASONS = {"stop", "eos", "eos_token", "end_of_text"}
LEGACY_PROMPT_LIMIT = 370


def digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def normalize(text: str) -> str:
    """Exactly the normalization used by the original three-source builder."""
    text = unicodedata.normalize("NFKC", text).casefold()
    text = "".join(c for c in text if unicodedata.category(c) != "Cf")
    return re.sub(r"\s+", "", text)


def _json(path: Path, value) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def load_records(directory: str | Path) -> dict[str, pd.DataFrame]:
    directory = Path(directory)
    return {name: pd.read_parquet(directory / f"{name}.parquet") for name in SPLITS}


def read_jsonl(path: str | Path) -> list[dict]:
    result = []
    with Path(path).open() as stream:
        for number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid translation JSON at line {number}") from exc
            if not isinstance(value, dict):
                raise ValueError(f"Translation line {number} must be an object")
            result.append(value)
    return result


def _originals(records: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
    if set(records) != set(SPLITS):
        raise ValueError(f"Exactly these original splits are required: {SPLITS}")
    parts = []
    for split in SPLITS:
        frame = records[split].copy(deep=True)
        missing = set(STABLE_COLUMNS + ("prompt",)) - set(frame)
        if missing:
            raise ValueError(f"{split}: missing columns {sorted(missing)}")
        if not frame["split"].eq(split).all():
            raise ValueError(f"{split}: stored split disagrees with its file")
        if frame[list(STABLE_COLUMNS) + ["prompt"]].isna().any().any():
            raise ValueError(f"{split}: null identity, label, source or prompt")
        if not frame.prompt.map(lambda x: isinstance(x, str) and bool(normalize(x))).all():
            raise ValueError(f"{split}: original text must be a nonempty string")
        if not frame.base_id.map(lambda x: isinstance(x, str) and bool(x)).all():
            raise ValueError(f"{split}: base_id must be a nonempty string")
        if not frame.prompt_harm_label.isin(["harmful", "unharmful"]).all():
            raise ValueError(f"{split}: invalid original binary label")
        parts.append(frame)
    original = pd.concat(parts, ignore_index=True)
    if not original.base_id.is_unique:
        raise ValueError("Original base_id must be globally unique")
    if original.groupby("group_id").split.nunique().gt(1).any():
        raise ValueError("Original group_id already crosses split boundaries")
    return original


def _mapping(frame: pd.DataFrame, original: pd.DataFrame) -> pd.DataFrame:
    required = {"base_id", "source_row", "original_prompt"}
    if not required.issubset(frame):
        raise ValueError(f"English mapping requires {sorted(required)}")
    if not {"prompt", "translated_prompt"} & set(frame):
        raise ValueError("English mapping must include old Chinese prompt or translated_prompt for alignment")
    if frame.base_id.duplicated().any():
        raise ValueError("Duplicate base_id in English mapping")
    target = original[original.source.eq(TARGET_SOURCE)].set_index("base_id")
    missing = set(target.index) - set(frame.base_id)
    if missing:
        raise ValueError(f"English mapping missing {len(missing)} target IDs")
    result = frame.set_index("base_id").loc[target.index].copy()
    if not result.original_prompt.map(lambda x: isinstance(x, str) and bool(x.strip())).all():
        raise ValueError("Mapped English must be a nonempty string")
    if not result.source_row.eq(target.source_row).all():
        raise ValueError("English mapping source_row disagrees with original records")
    if "translated_prompt" in result and not result.translated_prompt.eq(target.prompt).all():
        raise ValueError("English mapping old Chinese text disagrees with original records")
    if "prompt" in result and not result.prompt.eq(target.prompt).all():
        raise ValueError("English mapping old Chinese text disagrees with original records")
    if "english_text_sha256" in result and not result.english_text_sha256.eq(
            result.original_prompt.map(digest)).all():
        raise ValueError("English mapping text SHA256 is invalid")
    return result


def _translation_index(translations: Iterable[dict], mapped: pd.DataFrame,
                       require_complete: bool) -> dict[str, dict]:
    indexed = {}
    for item in translations:
        if not isinstance(item, dict):
            raise ValueError("Each translation must be an object")
        required = {"base_id", "original_english", "original_english_sha256",
                    "translation", "status", "reason"}
        if not required.issubset(item):
            raise ValueError(f"Translation missing fields: {sorted(required - set(item))}")
        key = item["base_id"]
        if key in indexed:
            raise ValueError(f"Duplicate translation base_id: {key}")
        if key not in mapped.index:
            raise ValueError(f"Translation for unknown/non-WildGuard base_id: {key}")
        english = item["original_english"]
        if english != mapped.at[key, "original_prompt"]:
            raise ValueError(f"English text disagrees with mapping: {key}")
        if item["original_english_sha256"] != digest(english):
            raise ValueError(f"English SHA256 mismatch: {key}")
        if item["status"] not in {"ok", "review", "error"}:
            raise ValueError(f"Invalid translation status: {key}")
        if not isinstance(item["translation"], str):
            raise ValueError(f"Translation must be a string, including on error: {key}")
        if not isinstance(item["reason"], str) or not item["reason"].strip():
            raise ValueError(f"A per-row revision/review reason is required: {key}")
        if "generation_tokens" in item and (type(item["generation_tokens"]) is not int
                                              or item["generation_tokens"] < 0):
            raise ValueError(f"generation_tokens must be a nonnegative integer: {key}")
        model_text = item.get("model_translation", item["translation"])
        if not isinstance(model_text, str):
            raise ValueError(f"Archived model_translation must be text: {key}")
        for field, text in (("translation_sha256", item["translation"]),
                            ("model_translation_sha256", model_text)):
            if field in item and item[field] != digest(text):
                raise ValueError(f"{field} does not match full recorded text: {key}")
        if item.get("translation_source") == "recorded_bilingual_correction":
            review = item.get("review_override") or {}
            if (review.get("decision") != "correct" or review.get("corrected_translation") != item["translation"]
                    or review.get("translation_sha256") != digest(model_text)
                    or review.get("original_english_sha256") != digest(english)
                    or not item.get("original_model_audit") or not item.get("corrected_translation_audit")):
                raise ValueError(f"Recorded correction lacks consistent original-text/review evidence: {key}")
        # All generator metadata is retained, including optional future fields.
        json.dumps(item, ensure_ascii=False, allow_nan=False)
        indexed[key] = dict(item)
    missing = set(mapped.index) - set(indexed)
    if missing and require_complete:
        raise ValueError(f"Missing {len(missing)} WildGuard translations; no candidate written")
    return indexed


def token_lengths(tokenizer, texts: list[str], batch_size: int = 512) -> list[int]:
    lengths = []
    for start in range(0, len(texts), batch_size):
        encoded = tokenizer(texts[start:start + batch_size], add_special_tokens=False,
                            truncation=False)["input_ids"]
        if len(encoded) != len(texts[start:start + batch_size]):
            raise ValueError("Tokenizer changed the batch length")
        lengths.extend(len(tokens) for tokens in encoded)
    return lengths


def length_distribution(frame: pd.DataFrame) -> dict:
    values = frame.prompt_tokens
    boundaries = [(0, 128), (129, 256), (257, 370), (371, 512), (513, 1024),
                  (1025, 2048), (2049, 4096)]
    bins = {f"{low}-{high}": int(values.between(low, high).sum()) for low, high in boundaries}
    bins[">4096"] = int(values.gt(4096).sum())
    return dict(rows=len(frame), min_tokens=int(values.min()) if len(frame) else None,
                max_tokens=int(values.max()) if len(frame) else None,
                mean_tokens=float(values.mean()) if len(frame) else None,
                p50_tokens=float(values.quantile(.5)) if len(frame) else None,
                p95_tokens=float(values.quantile(.95)) if len(frame) else None,
                p99_tokens=float(values.quantile(.99)) if len(frame) else None,
                over_legacy_limit=int(values.gt(LEGACY_PROMPT_LIMIT).sum()), bins=bins)


def length_statistics(archive, accepted, quarantine) -> dict:
    frames = {"archive": archive, "candidate": accepted, "quarantine": quarantine}
    return dict(**{name: length_distribution(frame) for name, frame in frames.items()},
                by_split={split: {name: length_distribution(frame[frame.split.eq(split)])
                                  for name, frame in frames.items()} for split in SPLITS},
                by_source={source: {name: length_distribution(frame[frame.source.eq(source)])
                                    for name, frame in frames.items()} for source in sorted(archive.source.unique())})


def collision_groups(frame: pd.DataFrame) -> list[dict]:
    groups = []
    nonempty = frame[frame.normalized_prompt.ne("")]
    duplicates = nonempty[nonempty.normalized_prompt.duplicated(keep=False)]
    for norm, group in duplicates.groupby("normalized_prompt", sort=True):
        labels = set(zip(group.prompt_harm_label, group.category_id.astype(str)))
        cross_split = group.split.nunique() > 1
        reasons = ["cross_split_normalized_duplicate" if cross_split
                   else "within_split_normalized_duplicate"]
        if len(labels) > 1:
            reasons.append("conflicting_labels_for_same_normalized_text")
        groups.append(dict(normalized_sha256=digest(norm), base_ids=sorted(group.base_id),
                           splits=sorted(group.split.unique()), rows=len(group),
                           binary_label_conflict=group.prompt_harm_label.nunique() > 1,
                           category_label_conflict=group.category_id.astype(str).nunique() > 1,
                           label_pairs=[list(x) for x in sorted(labels)], reasons=reasons))
    return groups


def english_collision_groups(frame: pd.DataFrame) -> list[dict]:
    """Recover exact English lineage without guessing semantic families."""
    english = frame.loc[frame.source.eq(TARGET_SOURCE)].copy()
    english["normalized_english"] = english.original_english.map(normalize)
    duplicated = english[english.normalized_english.duplicated(keep=False)]
    groups = []
    for norm, group in duplicated.groupby("normalized_english", sort=True):
        labels = set(zip(group.prompt_harm_label, group.category_id.astype(str)))
        reasons = []
        if group.split.nunique() > 1:
            reasons.append("cross_split_normalized_english_lineage")
        if len(labels) > 1:
            reasons.append("conflicting_labels_for_same_normalized_english")
        groups.append(dict(normalized_english_sha256=digest(norm),
                           base_ids=sorted(group.base_id), splits=sorted(group.split.unique()),
                           rows=len(group), binary_label_conflict=group.prompt_harm_label.nunique() > 1,
                           category_label_conflict=group.category_id.astype(str).nunique() > 1,
                           label_pairs=[list(x) for x in sorted(labels)],
                           reasons=reasons))
    return groups


def _pair_overlaps(frame: pd.DataFrame) -> dict[str, int]:
    result = {}
    for i, left in enumerate(SPLITS):
        for right in SPLITS[i + 1:]:
            for column in ("base_id", "group_id", "normalized_prompt"):
                a = set(frame.loc[frame.split.eq(left), column])
                b = set(frame.loc[frame.split.eq(right), column])
                result[f"{left}_{right}_{column}_overlap"] = len(a & b)
    return result


def audit(directory: str | Path, records: Mapping[str, pd.DataFrame], tokenizer,
          *, mapping: pd.DataFrame | None = None) -> dict:
    """Read saved files back and verify identities, exclusion accounting and text."""
    directory = Path(directory)
    original = _originals(records).set_index("base_id").sort_index()
    archive = pd.read_parquet(directory / "full_repaired_archive.parquet")
    quarantine = pd.read_parquet(directory / "quarantine.parquet")
    accepted = pd.concat([pd.read_parquet(directory / f"{s}.parquet") for s in SPLITS],
                         ignore_index=True)
    manifest = json.loads((directory / "manifest.json").read_text())
    for name, expected in manifest["artifact_sha256"].items():
        if hashlib.sha256((directory / name).read_bytes()).hexdigest() != expected:
            raise ValueError(f"Artifact hash mismatch: {name}")
    if not archive.base_id.is_unique or set(archive.base_id) != set(original.index):
        raise ValueError("Full archive does not preserve every original base_id exactly once")
    saved = archive.set_index("base_id").sort_index()
    preserved_columns = list(STABLE_COLUMNS) + [c for c in ("source_file", "source_unit",
        "base_normalized", "category_label", "text_form", "template_id", "exclusion_reason") if c in original]
    for col in preserved_columns:
        if col == "base_id":
            continue
        if not saved[col].eq(original[col]).all():
            raise ValueError(f"Stable original field changed: {col}")
    if not saved.old_prompt.eq(original.prompt).all():
        raise ValueError("Old prompt archive differs from input")
    target = saved.source.eq(TARGET_SOURCE)
    if not saved.loc[target, "original_english_sha256"].eq(
            saved.loc[target, "original_english"].map(digest)).all():
        raise ValueError("Archived English text and SHA256 disagree")
    for row in saved[target & saved.translation_status.ne("missing")].itertuples():
        metadata = json.loads(row.translation_metadata_json)
        if (row.prompt != metadata["translation"] or row.model_translation != metadata.get("model_translation", metadata["translation"])
                or row.model_translation_sha256 != digest(row.model_translation)
                or row.translation_source != metadata.get("translation_source", "model")
                or row.correction_source != metadata.get("correction_source", "none")):
            raise ValueError("Archived effective/model text or correction provenance disagrees with metadata")
    if mapping is not None:
        mapped = _mapping(mapping, original.reset_index())
        if not saved.loc[mapped.index, "original_english"].eq(mapped.original_prompt).all():
            raise ValueError("Archived English differs from original source mapping")
    untouched = original.source.ne(TARGET_SOURCE)
    if not saved.loc[untouched, "prompt"].eq(original.loc[untouched, "prompt"]).all():
        raise ValueError("A non-WildGuard prompt was modified")
    if set(accepted.base_id) & set(quarantine.base_id):
        raise ValueError("Quarantined records leaked into candidate splits")
    if (not accepted.base_id.is_unique or not quarantine.base_id.is_unique
            or set(accepted.base_id) | set(quarantine.base_id) != set(original.index)):
        raise ValueError("Accepted and quarantine records do not exactly partition the archive")
    expected = archive.loc[archive.quarantine_reason.eq(""), "base_id"]
    if set(accepted.base_id) != set(expected):
        raise ValueError("Candidate membership disagrees with archive quarantine decisions")
    if not archive.repair_eligible.eq(archive.quarantine_reason.eq("")).all():
        raise ValueError("Eligibility flag disagrees with quarantine reasons")
    if not archive.translation_changed.eq(archive.prompt.ne(archive.old_prompt)).all():
        raise ValueError("Translation-change flag is incorrect")
    for frame, is_accepted in [(accepted, True), (quarantine, False)]:
        if not frame.empty:
            indexed = frame.set_index("base_id").sort_index()
            pd.testing.assert_frame_equal(indexed, saved.loc[indexed.index], check_dtype=False)
        if not frame.quarantine_reason.eq("").eq(is_accepted).all():
            raise ValueError("Candidate/quarantine reasons are inconsistent")
    if not archive.normalized_prompt.eq(archive.prompt.map(normalize)).all():
        raise ValueError("Repaired normalized_prompt is stale")
    sample_ids = [digest(f"{row.base_id}:{row.prompt}") for row in archive.itertuples()]
    if sample_ids != archive.sample_id.tolist() or not archive.sample_id.is_unique:
        raise ValueError("Repaired sample_id is stale or duplicated")
    if token_lengths(tokenizer, archive.prompt.tolist()) != archive.prompt_tokens.tolist():
        raise ValueError("Saved token lengths differ from full, untruncated tokenization")
    if not archive.prompt_over_legacy_limit.eq(archive.prompt_tokens.gt(LEGACY_PROMPT_LIMIT)).all():
        raise ValueError("Legacy prompt-length annotation is stale")
    if manifest["length_statistics"] != length_statistics(archive, accepted, quarantine):
        raise ValueError("Saved length distributions disagree with complete records")
    if collision_groups(accepted):
        raise ValueError("Unquarantined normalized duplicate/label-conflict records")
    if any(group["reasons"] for group in english_collision_groups(accepted)):
        raise ValueError("Unquarantined English lineage crosses splits or has conflicting labels")
    overlaps = _pair_overlaps(accepted)
    if any(overlaps.values()):
        raise ValueError(f"Candidate cross-split overlap: {overlaps}")
    limit = manifest["max_prompt_tokens"]
    if limit is not None and accepted.prompt_tokens.gt(limit).any():
        raise ValueError("Overlength record in candidate splits")
    if accepted.normalized_prompt.eq("").any():
        raise ValueError("Empty normalized prompt in candidate splits")
    if accepted.loc[accepted.source.eq(TARGET_SOURCE), "translation_status"].ne("ok").any():
        raise ValueError("Unapproved translation status in candidate splits")
    for metadata in accepted.loc[accepted.source.eq(TARGET_SOURCE), "translation_metadata_json"]:
        item = json.loads(metadata)
        if str(item.get("finish_reason", "")).casefold() not in COMPLETE_FINISH_REASONS:
            raise ValueError("A translation without a verified generation stop entered candidate splits")
        if item.get("input_truncated") is True:
            raise ValueError("A translation with truncated source input entered candidate splits")
    for split in SPLITS:
        saved_split = pd.read_parquet(directory / f"{split}.parquet")
        if not saved_split.split.eq(split).all():
            raise ValueError(f"A record was moved to another split file: {split}")
    return dict(passed=True, release_status="candidate_pending_root_qa",
                input_rows=len(original), archived_rows=len(archive),
                candidate_rows=len(accepted), quarantined_rows=len(quarantine),
                stable_fields_verified=preserved_columns,
                full_untruncated_token_lengths_verified=True,
                exact_duplicate_groups_before_quarantine=len(collision_groups(archive)),
                exact_duplicate_groups_after_quarantine=0,
                english_lineage_groups_requiring_quarantine=sum(bool(g["reasons"])
                    for g in english_collision_groups(archive)),
                english_cross_split_or_conflict_groups_after_quarantine=0, **overlaps)


def build(records: Mapping[str, pd.DataFrame], out: str | Path,
          translations: Iterable[dict], *, mapping: pd.DataFrame, tokenizer,
          max_tokens: int | None = None, require_complete: bool = True,
          provenance: dict | None = None) -> dict:
    """Write a separate candidate and all quarantined full texts atomically.

    Missing translations fail by default. A deliberately incomplete preview may
    use require_complete=False; its missing rows are explicitly quarantined.
    Full quality-approved texts are retained by default, regardless of length.
    The historical 370-token setting is an annotation, not a quality criterion.
    Every member of a duplicate/conflict group is quarantined, without choosing
    a preferred split, label, source, or model outcome.
    """
    out = Path(out)
    if out.exists() and (not out.is_dir() or any(out.iterdir())):
        raise ValueError(f"Refusing to overwrite nonempty output: {out}")
    if max_tokens is not None and (type(max_tokens) is not int or max_tokens < 1):
        raise ValueError("max_tokens must be a positive integer or None")
    original = _originals(records)
    mapped = _mapping(mapping, original)
    indexed = _translation_index(translations, mapped, require_complete)
    archive = original.copy(deep=True)
    archive["old_prompt"] = archive.prompt
    if "sample_id" in archive:
        archive["old_sample_id"] = archive.sample_id
    if "normalized_prompt" in archive:
        archive["old_normalized_prompt"] = archive.normalized_prompt
    archive["original_english"] = ""
    archive["original_english_sha256"] = ""
    archive["translation_status"] = "unchanged_source"
    archive["revision_reason"] = "Non-WildGuard source retained without text changes"
    archive["translation_metadata_json"] = "{}"
    archive["model_translation"] = ""
    archive["model_translation_sha256"] = ""
    archive["translation_source"] = "unchanged_source"
    archive["correction_source"] = "none"
    archive["reviewer"] = ""
    reasons: dict[str, list[str]] = defaultdict(list)
    for i, row in archive[archive.source.eq(TARGET_SOURCE)].iterrows():
        english = mapped.at[row.base_id, "original_prompt"]
        archive.at[i, "original_english"] = english
        archive.at[i, "original_english_sha256"] = digest(english)
        item = indexed.get(row.base_id)
        if item is None:
            # The old text remains only in the full archive, never as a fallback
            # training example. Its missing status and quarantine are explicit.
            archive.at[i, "translation_status"] = "missing"
            archive.at[i, "revision_reason"] = "Missing translation; old text archived only"
            reasons[row.base_id].append("missing_translation")
            continue
        archive.at[i, "prompt"] = item["translation"]
        archive.at[i, "translation_status"] = item["status"]
        archive.at[i, "revision_reason"] = item["reason"]
        model_text = item.get("model_translation", item["translation"])
        archive.at[i, "model_translation"] = model_text
        archive.at[i, "model_translation_sha256"] = digest(model_text)
        archive.at[i, "translation_source"] = item.get("translation_source", "model")
        archive.at[i, "correction_source"] = item.get("correction_source", "none")
        archive.at[i, "reviewer"] = (item.get("review_override") or {}).get("reviewer", "")
        archive.at[i, "translation_metadata_json"] = json.dumps(item, ensure_ascii=False,
                                                                  sort_keys=True, allow_nan=False)
        if item["status"] != "ok":
            reasons[row.base_id].append(f"translation_status_{item['status']}")
        finish_reason = str(item.get("finish_reason", "")).casefold()
        if finish_reason in TRUNCATED_FINISH_REASONS:
            reasons[row.base_id].append("translation_generation_length_limit")
        elif finish_reason not in COMPLETE_FINISH_REASONS:
            reasons[row.base_id].append("unverified_generation_finish")
        if item.get("input_truncated") is True:
            reasons[row.base_id].append("translation_source_input_truncated")
    archive["translation_changed"] = archive.prompt.ne(archive.old_prompt)
    archive["normalized_prompt"] = archive.prompt.map(normalize)
    archive["sample_id"] = [digest(f"{r.base_id}:{r.prompt}") for r in archive.itertuples()]
    archive["prompt_tokens"] = token_lengths(tokenizer, archive.prompt.tolist())
    archive["prompt_over_legacy_limit"] = archive.prompt_tokens.gt(LEGACY_PROMPT_LIMIT)
    for row in archive.itertuples():
        if not row.normalized_prompt:
            reasons[row.base_id].append("empty_repaired_text")
        if max_tokens is not None and row.prompt_tokens > max_tokens:
            reasons[row.base_id].append("overlength_full_text_retained")
    collisions = collision_groups(archive)
    for group in collisions:
        for key in group["base_ids"]:
            reasons[key].extend(group["reasons"])
    english_collisions = english_collision_groups(archive)
    for group in english_collisions:
        for key in group["base_ids"]:
            reasons[key].extend(group["reasons"])
    archive["quarantine_reason"] = [";".join(sorted(set(reasons[key]))) for key in archive.base_id]
    archive["repair_eligible"] = archive.quarantine_reason.eq("")
    accepted = archive[archive.repair_eligible].copy()
    quarantine = archive[~archive.repair_eligible].copy()
    reason_counts = Counter(reason for key in archive.base_id for reason in set(reasons[key]))
    # Restored English lineage is checked even if two new translations differ.
    # All members are quarantined; original split membership remains unchanged.
    english = archive[archive.source.eq(TARGET_SOURCE)].copy()
    english["normalized_english"] = english.original_english.map(normalize)
    english_cross = english.groupby("normalized_english").filter(lambda x: x.split.nunique() > 1)
    chinese_collision_ids = {key for g in collisions for key in g["base_ids"]}
    english_collision_ids = {key for g in english_collisions if g["reasons"] for key in g["base_ids"]}
    manifest = dict(protocol="three_source_uniform_wildguard_retranslation",
                    release_status="candidate_pending_root_qa", source_selection=TARGET_SOURCE,
                    translation_selection_uses_predictions_or_labels=False,
                    quarantine_checks_label_conflicts=True, max_prompt_tokens=max_tokens,
                    legacy_prompt_limit=LEGACY_PROMPT_LIMIT,
                    length_policy="annotation_only_no_cap" if max_tokens is None else "explicit_optional_cap_quarantine_no_truncation",
                    length_statistics=length_statistics(archive, accepted, quarantine),
                    input_rows=len(original), archived_rows=len(archive),
                    translated_rows=len(indexed), expected_translation_rows=len(mapped),
                    candidate_rows=len(accepted), quarantined_rows=len(quarantine),
                    quarantine_reason_counts=dict(sorted(reason_counts.items())),
                    translation_status_counts=archive.translation_status.value_counts().to_dict(),
                    translation_source_counts=archive.translation_source.value_counts().to_dict(),
                    recorded_bilingual_correction_rows=int(archive.translation_source.eq("recorded_bilingual_correction").sum()),
                    collision_policy="quarantine every member; preserve original split and labels",
                    english_cross_split_rows=len(english_cross),
                    english_lineage_quarantine_rows=len(english_collision_ids),
                    english_binary_label_conflict_groups=sum(g["binary_label_conflict"] for g in english_collisions),
                    english_category_label_conflict_groups=sum(g["category_label_conflict"] for g in english_collisions),
                    chinese_collision_quarantine_rows=len(chinese_collision_ids),
                    english_and_chinese_quarantine_overlap=len(english_collision_ids & chinese_collision_ids),
                    english_lineage_additional_rows=len(english_collision_ids - chinese_collision_ids),
                    normalization="NFKC, casefold, drop Unicode Cf, remove whitespace",
                    sample_id_formula="SHA256(base_id + ':' + full repaired prompt)",
                    stable_identity_note="base_id/group_id retain old lineage; not hashes of repaired text",
                    provenance=provenance or {},
                    builder_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                    limitations=["Candidate requires independent translation/semantic QA before use.",
                                 "Exact deduplication is not semantic-near-duplicate isolation.",
                                 "Recovered normalized English cross-split/conflicting-label identities are quarantined.",
                                 "Source labels are preserved, not certified by retranslation."],
                    splits={split: dict(original=int(original.split.eq(split).sum()),
                                        retained=int(accepted.split.eq(split).sum()),
                                        quarantined=int(quarantine.split.eq(split).sum())) for split in SPLITS})
    out.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{out.name}.building-", dir=out.parent))
    try:
        archive.to_parquet(staging / "full_repaired_archive.parquet", index=False)
        quarantine.to_parquet(staging / "quarantine.parquet", index=False)
        quarantine.to_csv(staging / "quarantine.csv", index=False)
        for split in SPLITS:
            accepted[accepted.split.eq(split)].to_parquet(staging / f"{split}.parquet", index=False)
        revision_columns = ["base_id", "group_id", "split", "source", "source_row", "old_prompt",
                            "original_english", "original_english_sha256", "prompt", "prompt_harm_label",
                            "category_id", "sample_id", "prompt_tokens", "prompt_over_legacy_limit", "translation_status",
                            "translation_changed", "revision_reason", "quarantine_reason", "model_translation",
                            "model_translation_sha256", "translation_source", "correction_source", "reviewer"]
        archive[revision_columns].to_csv(staging / "revision_log.csv", index=False)
        english_cross.to_csv(staging / "english_cross_split_review.csv", index=False)
        _json(staging / "collision_groups.json", collisions)
        _json(staging / "english_collision_groups.json", english_collisions)
        train_counts = accepted.loc[accepted.split.eq("train"), "category_id"].value_counts()
        _json(staging / "class_weights.json", {str(k): round(math.sqrt(train_counts.max() / v), 4)
                                               for k, v in sorted(train_counts.items())})
        manifest["artifact_sha256"] = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                                        for p in sorted(staging.iterdir()) if p.is_file()}
        _json(staging / "manifest.json", manifest)
        result = audit(staging, records, tokenizer, mapping=mapping)
        _json(staging / "audit.json", result)
        length_note = ("不设统一token长度上限；通过质量/重复检查的长文本完整保留。"
                       if max_tokens is None else f"此显式兼容导出采用{max_tokens} tokens上限，超限全文归档并隔离。")
        card = ("# hanguard 中文翻译修复候选数据\n\n"
                "构建时状态：candidate_pending_root_qa，不自动切换训练入口。最终当前QA状态以单独的release_qa.json为准（若存在）；"
                "侧录必须绑定本manifest和独立verification的SHA，不改变构建时manifest/audit状态，也不代表全量语义认证。\n\n"
                "全部 WildGuard 原有三集记录按统一规则重新翻译，其他来源保留原文本。"
                "原 split/base_id/group_id/标签不变；base_id 表示历史归属，不再要求等于新文本哈希。\n\n"
                f"原始 {len(original)} 条；完整档案 {len(archive)} 条；候选三集 {len(accepted)} 条；"
                f"明确隔离 {len(quarantine)} 条。{length_note}不截断，不回退旧译文。\n\n"
                f"prompt_over_legacy_limit仅标记是否超过旧实验{LEGACY_PROMPT_LIMIT} tokens，长度不是翻译质量判据。"
                f"完整档案超旧限{manifest['length_statistics']['archive']['over_legacy_limit']}条，候选三集保留超旧限"
                f"{manifest['length_statistics']['candidate']['over_legacy_limit']}条；长度分布及各来源/划分明细见manifest.length_statistics。\n\n"
                "full_repaired_archive.parquet 保存所有记录及完整新旧文本、英文和修订理由；"
                "quarantine.parquet/.csv 保存所有隔离记录。normalized_prompt/sample_id/token数已重算。\n\n"
                "修复中文的跨集重复、同集重复及标签冲突均整组隔离，明细见collision_groups.json。"
                "恢复英文同源后发现的跨集精确重复或标签冲突也整组隔离，不依据模型错误选删。"
                "明细另见english_collision_groups.json及english_cross_split_review.csv。\n\n"
                "model_translation/model_translation_sha256保留模型原译；translation_source/correction_source及reviewer区分模型输出与有记录的双语复核修订，完整原/新审计保存在translation_metadata_json。"
                "Codex助手复核不是人工专家金标，保留原标签也不代表标签已验证；精确去重不保证语义近重复隔离。\n\n"
                "后续实验必须显式指定本新目录、重新生成token/特征缓存，并另行登记训练与评估协议。"
                "历史feature_fusion.DATA仍指向original，旧缓存含80709条断言，classification_study继承旧LoRA父检查点；本任务不改这些入口、不训练、不复用旧缓存。"
                "旧370长度配置/断言不可直接用于这份完整主数据，后续训练必须另行适配长文本策略并验证，不能静默截断主数据。"
                "译文和隔离后的测试样本已发生变化，新分数不能直接减去旧数据上的分数并宣称算法增益。\n")
        (staging / "DATASET_CARD.md").write_text(card)
        (staging / "README.md").write_text(card)
        if out.exists():
            if any(out.iterdir()):
                raise ValueError(f"Output became nonempty during build: {out}")
            out.rmdir()
        staging.replace(out)
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    return dict(result, output=str(out))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=ROOT / "data/three_source_original")
    parser.add_argument("--translations", type=Path)
    parser.add_argument("--mapping", type=Path, required=True,
                        help="CSV/parquet: base_id, source_row, original_prompt English, prompt/translated_prompt old Chinese")
    parser.add_argument("--output", type=Path, default=ROOT / "data/three_source_translation_repaired")
    parser.add_argument("--tokenizer", type=Path, default=ROOT / "models/Qwen3.5-4B")
    parser.add_argument("--max-tokens", type=int, default=0, help="Default 0: retain full long texts; positive values explicitly quarantine over-cap texts, never truncate")
    parser.add_argument("--allow-incomplete-preview", action="store_true")
    parser.add_argument("--audit-only", action="store_true", help="Read and verify existing candidate without writing")
    args = parser.parse_args()
    if args.source.resolve() == args.output.resolve():
        raise ValueError("Output must differ from the immutable source directory")
    if args.max_tokens < 0:
        raise ValueError("--max-tokens cannot be negative")
    if not args.audit_only and args.translations is None:
        parser.error("--translations is required when building")
    files = [args.source / f"{s}.parquet" for s in SPLITS] + [args.mapping]
    if args.translations is not None:
        files.append(args.translations)
    hashes = {str(p.resolve()): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    mapping = (pd.read_parquet(args.mapping) if args.mapping.suffix == ".parquet"
               else pd.read_csv(args.mapping, float_precision="round_trip", keep_default_na=False))
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True)
    tokenizer_hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in
                        [args.tokenizer / name for name in ["tokenizer.json", "tokenizer_config.json",
                                                          "special_tokens_map.json"]] if p.exists()}
    records = load_records(args.source)
    if args.audit_only:
        result = audit(args.output, records, tokenizer, mapping=mapping)
    else:
        result = build(records, args.output, read_jsonl(args.translations),
                       mapping=mapping, tokenizer=tokenizer, max_tokens=args.max_tokens or None,
                       require_complete=not args.allow_incomplete_preview,
                       provenance=dict(input_sha256=hashes, tokenizer=str(args.tokenizer.resolve()),
                                       tokenizer_sha256=tokenizer_hashes))
    for name, expected in hashes.items():
        if hashlib.sha256(Path(name).read_bytes()).hexdigest() != expected:
            raise ValueError(f"An input changed during the build; candidate must not be used: {name}")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
