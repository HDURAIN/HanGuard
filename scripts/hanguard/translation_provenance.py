"""Recover translation lineage and screen old translations without model scores.

Only the new output directory is written. Existing source/split files, labels,
identities and split assignments are never changed. Risk flags are review
candidates, not translation-error verdicts or instructions to remove examples.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import re
import unicodedata

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = ROOT / "outputs/hanguard_translation_repair_20260928"
DEFAULT_TOKENIZER = Path("/mnt/data1/zhouhanyu/models/nllb-200-3.3B")
SPLITS = ("train", "validation", "test")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def text_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def write_new_json(path: Path, value: dict) -> None:
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def write_new_parquet(path: Path, frame: pd.DataFrame) -> None:
    # Complete the temporary artifact before exposing it to translation workers.
    # Exclusive destination creation refuses to overwrite previous artifacts.
    temp = path.with_suffix(path.suffix + ".partial")
    if path.exists() or temp.exists():
        raise FileExistsError(path)
    frame.to_parquet(temp, index=False)
    path.hardlink_to(temp)
    temp.unlink()


def load_current_splits(root: Path) -> tuple[pd.DataFrame, dict]:
    parts, files = [], {}
    for split in SPLITS:
        path = root / "data/three_source_original" / f"{split}.parquet"
        frame = pd.read_parquet(path)
        assert frame["split"].eq(split).all(), path
        frame = frame.copy()
        frame["split_row"] = range(len(frame))
        parts.append(frame)
        files[str(path.relative_to(root))] = {"rows": len(frame), "sha256": sha256(path)}
    all_rows = pd.concat(parts, ignore_index=True)
    assert all_rows.base_id.is_unique and all_rows.sample_id.is_unique
    assert len(all_rows) == 80709, "Unexpected source dataset; inspect before using this protocol."
    return all_rows, files


def build_mapping(root: Path, output: Path) -> pd.DataFrame:
    all_rows, files = load_current_splits(root)
    names = {
        "english": "data/sources/wildguard_en/train.parquet",
        "translation_input": "legacy/data/train_wildguard.parquet",
        "chinese": "data/sources/wildguard_zh.parquet",
    }
    frames = {}
    for key, name in names.items():
        path = root / name
        frames[key] = pd.read_parquet(path)
        files[name] = {"rows": len(frames[key]), "sha256": sha256(path)}
    chunks = []
    chunk_paths = sorted((root / "legacy/data/train_wildguard_checkpoints").glob("chunk_*.parquet"))
    assert len(chunk_paths) == 6
    for path in chunk_paths:
        frame = pd.read_parquet(path)
        chunks.append(frame)
        files[str(path.relative_to(root))] = {"rows": len(frame), "sha256": sha256(path)}
    en, legacy, zh = (frames[key] for key in ("english", "translation_input", "chinese"))
    combined = pd.concat(chunks, ignore_index=True)
    assert en.prompt.equals(legacy.prompt)
    assert zh.prompt.equals(combined.prompt)
    assert zh.prompt_harm_label.equals(combined.prompt_harm_label)
    shuffled = en.sample(frac=1, random_state=42)
    assert len(shuffled) == len(zh) == 86759
    assert shuffled.prompt_harm_label.reset_index(drop=True).equals(zh.prompt_harm_label)
    mapping = all_rows.loc[all_rows.source.eq("wildguard_zh")].copy().reset_index(drop=True)
    indexes = mapping.source_row.astype(int).tolist()
    assert len(mapping) == 46187
    assert mapping.prompt.tolist() == zh.iloc[indexes].prompt.tolist()
    assert mapping.prompt_harm_label.tolist() == zh.iloc[indexes].prompt_harm_label.tolist()
    originals = shuffled.iloc[indexes]
    mapping["original_file"] = names["english"]
    mapping["original_row"] = originals.index.to_numpy()
    mapping["original_prompt"] = originals.prompt.to_numpy()
    mapping["original_prompt_harm_label"] = originals.prompt_harm_label.to_numpy()
    mapping["original_adversarial"] = originals.adversarial.to_numpy()
    mapping["original_subcategory"] = originals.subcategory.to_numpy()
    mapping["english_text_sha256"] = mapping.original_prompt.map(text_hash)
    mapping["lineage_evidence"] = "verified_shuffle42_translation_chunks_and_current_source_row"
    assert mapping.prompt_harm_label.equals(mapping.original_prompt_harm_label)
    output.mkdir(parents=True, exist_ok=True)
    write_new_parquet(output / "source_mapping.parquet", mapping)
    write_new_json(output / "mapping_manifest.json", {
        "protocol": "recover_existing_split_wildguard_translation_lineage",
        "input_rows": len(all_rows), "mapped_rows": len(mapping),
        "mapped_split_counts": mapping.split.value_counts().to_dict(),
        "unique_english_texts": int(mapping.english_text_sha256.nunique()),
        "no_predictions_used": True, "no_labels_or_split_assignments_changed": True,
        "source_files": files, "code_sha256_at_mapping": sha256(Path(__file__)),
        "mapping_sha256": sha256(output / "source_mapping.parquet"),
        "limitations": ["Source row numbers require the recorded file versions.",
                        "Original annotation reasoning and historical tokenizer hashes were not saved."],
    })
    print(json.dumps({"mapped_rows": len(mapping), "split_counts": mapping.split.value_counts().to_dict(),
                      "path": str(output / "source_mapping.parquet")}, ensure_ascii=False), flush=True)
    return mapping


# This transparent, incomplete lexicon finds possible omitted concepts, not
# equivalent meaning. Legitimate paraphrases produce false positives; words
# absent from this lexicon are never assumed to be preserved.
CONCEPTS = {
    "child_or_minor": (r"\b(?:child(?:ren)?|kids?|minors?|underage)\b", r"儿童|孩子|小孩|小朋友|未成年|幼童|婴儿|少年|青少年"),
    "pregnancy": (r"\b(?:pregnan\w*|fetus|foetus)\b", r"孕|胎|妊娠"),
    "password": (r"\b(?:passwords?|passcodes?|credentials)\b", r"密码|口令|凭证|凭据|证书|password|passcode|credential"),
    "private_information": (r"\b(?:personal|private|sensitive)\s+(?:information|data)\b", r"个人|私人|私密|隐私|敏感|personal|private|sensitive"),
    "without_permission": (r"\b(?:without\s+(?:their\s+)?(?:permission|consent)|unauthori[sz]ed)\b", r"未经|未获|无权|未授权|未经授权|不经|没有.*(?:许可|允许|同意)|unauthori[sz]ed|without"),
    "slur": (r"\bslurs?\b", r"侮辱|辱骂|蔑称|贬称|贬损|贬义|歧视性.*(?:词|语言)|冒犯性.*(?:词|语言)|slur"),
    "alcohol": (r"\b(?:alcohol|alcoholic|liquor|vodka|whisk(?:e)?y)\b", r"酒|乙醇|伏特加|威士忌|alcohol|liquor|vodka|whisk"),
    "weapon": (r"\b(?:weapons?|firearms?|guns?|rifles?)\b", r"武器|枪|火器|兵器|weapon|firearm|gun|rifle"),
    "explosive": (r"\b(?:explosives?|bombs?|detonators?)\b", r"爆|炸|雷管|explosive|bomb|detonator"),
    "self_harm": (r"\b(?:suicid\w*|self[- ]harm)\b", r"自杀|自残|自伤|轻生|伤害自己|suicid|self[- ]harm"),
}
COMPILED_CONCEPTS = {
    name: (re.compile(en, re.I), re.compile(zh, re.I))
    for name, (en, zh) in CONCEPTS.items()
}
RISK_RULES = {
    "source_over_512": "Current local NLLB tokenization with special tokens exceeds historical input max_length=512.",
    "target_reencoded_content_ge_254": "Current target content re-encodes to >=254 tokens, near historical max_new_tokens=256; generation IDs/stopping reason unavailable.",
    "replacement_or_nul": "Old Chinese text contains U+FFFD or U+0000; may reflect an intentional source character.",
    "unexpected_unicode_control": "Old text contains Cc controls other than tab/newline/carriage return, or Cf characters; may be intentional injection text.",
    "no_cjk_despite_latin_source": "Source has >=20 ASCII letters and old target has no CJK; code, names and deliberate English output can be valid.",
    "low_target_token_ratio": "Source content >=40 tokens and target/source content token ratio <0.35; not a language-independent quality test.",
    "high_target_token_ratio": "Source content >=16 tokens and target/source content token ratio >3.0; possible expansion or repetition, not proof.",
    "long_identical_character_run": "At least 12 identical non-whitespace characters in a row; may faithfully reproduce attack formatting.",
    "missing_concept_lexicon_candidate": "An explicit English concept matches the declared lexicon but no listed Chinese/English equivalent occurs; incomplete synonyms create false positives.",
    "missing_numeric_literal_candidate": "At least one source Arabic numeric literal is absent from target; Chinese numeral conversion and renumbering can be valid.",
    "missing_url_literal_candidate": "At least one source HTTP(S) URL is absent literally from target; punctuation/normalization can explain differences.",
}
NUMBER = re.compile(r"(?<![\w.])\d+(?:[.,]\d+)*")
URL = re.compile(r"https?://[^\s<>\"']+", re.I)
CJK = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\U00020000-\U0003134f]")


def normalize_identity(text: str) -> str:
    """Use the existing dataset's textual normalization, never semantic matching."""
    text = unicodedata.normalize("NFKC", text).casefold()
    return "".join(c for c in text if not c.isspace() and unicodedata.category(c) != "Cf")


def identity_audit(mapping: pd.DataFrame) -> tuple[dict, pd.DataFrame, pd.Series]:
    normalized = mapping.original_prompt.map(normalize_identity).map(text_hash)
    summary, records = {}, []
    for kind, keys in (("exact", mapping.english_text_sha256), ("normalized", normalized)):
        grouped = mapping.assign(identity_hash=keys).groupby("identity_hash", sort=True)
        stats = grouped.agg(rows=("base_id", "size"), splits=("split", "nunique"),
                            labels=("prompt_harm_label", "nunique"))
        duplicate = stats.rows.gt(1)
        cross = stats.splits.gt(1)
        conflict = stats.labels.gt(1)
        summary[kind] = {
            "unique_groups": len(stats), "duplicate_extra_rows": len(mapping) - len(stats),
            "duplicate_groups": int(duplicate.sum()), "rows_in_duplicate_groups": int(stats[duplicate].rows.sum()),
            "same_split_only_duplicate_groups": int((duplicate & ~cross).sum()),
            "same_split_only_duplicate_rows": int(stats[duplicate & ~cross].rows.sum()),
            "cross_split_groups": int(cross.sum()), "cross_split_rows": int(stats[cross].rows.sum()),
            "binary_conflict_groups": int(conflict.sum()), "binary_conflict_rows": int(stats[conflict].rows.sum()),
        }
        for key in stats.index[duplicate | conflict]:
            frame = grouped.get_group(key)
            records.append({
                "identity_kind": kind, "identity_hash": key, "rows": len(frame),
                "split_count": int(frame.split.nunique()), "splits": sorted(frame.split.unique().tolist()),
                "binary_labels": sorted(frame.prompt_harm_label.unique().tolist()),
                "base_ids": frame.base_id.tolist(), "source_rows": frame.source_row.astype(int).tolist(),
                "member_splits": frame.split.tolist(), "cross_split": bool(frame.split.nunique() > 1),
                "binary_conflict": bool(frame.prompt_harm_label.nunique() > 1),
            })
    return summary, pd.DataFrame(records), normalized


def curated_lineage_audit(root: Path) -> tuple[dict, pd.DataFrame]:
    all_rows, _ = load_current_splits(root)
    curated = all_rows[all_rows.source.eq("chinese_curated")]
    en = pd.read_parquet(root / "data/sources/wildguard_en/train.parquet").sample(frac=1, random_state=42)
    zh = pd.read_parquet(root / "data/sources/wildguard_zh.parquet")
    candidates = defaultdict(set)
    for translated, original in zip(zh.prompt, en.prompt):
        if isinstance(translated, str) and isinstance(original, str):
            candidates[translated].add(original)
    retranslation_path = root / "legacy/data/wildguard_injection_retrans_ckpt.parquet"
    ckpt = pd.read_parquet(retranslation_path)
    for row in ckpt.itertuples():
        if pd.notna(row.prompt_zh_new) and pd.notna(row.prompt):
            # Only the actual new-translation pair is valid evidence. The old
            # checkpoint's purported previous-translation field had a row bug.
            candidates[str(row.prompt_zh_new)].add(str(row.prompt))
    records = []
    for row in curated.itertuples():
        parents = sorted(candidates.get(row.prompt, set()))
        if not parents:
            continue
        records.append({"base_id": row.base_id, "split": row.split, "source_row": int(row.source_row),
                        "prompt": row.prompt, "english_parent_candidates": parents,
                        "candidate_count": len(parents), "all_matches_are_text_identity": all(p == row.prompt for p in parents),
                        "status": "exact_text_correspondence_not_proof_of_translated_origin"})
    columns = ["base_id", "split", "source_row", "prompt", "english_parent_candidates", "candidate_count",
               "all_matches_are_text_identity", "status"]
    matches = pd.DataFrame(records, columns=columns)
    return {
        "current_curated_rows": len(curated), "exact_known_translation_text_matches": len(matches),
        "identity_only_matches": sum(r["all_matches_are_text_identity"] for r in records),
        "nonidentity_correspondence_candidates": sum(not r["all_matches_are_text_identity"] for r in records),
        "unmatched_rows": len(curated) - len(matches),
        "checked_pairs": ["verified wildguard_zh shuffle42 lineage", str(retranslation_path.relative_to(root))],
        "limitations": ["The current curated source does not retain bilingual pairs or upstream per-row dataset IDs.",
                        "Exact shared Chinese text alone does not prove the curated record was produced by translation.",
                        "Unmatched records were not assigned English parents using semantic similarity or model judgments."],
    }, matches


def text_risks(original: str, translated: str, source_tokens: int, source_content_tokens: int,
               target_content_tokens: int) -> dict:
    missing_concepts = [name for name, (en, zh) in COMPILED_CONCEPTS.items()
                        if en.search(original) and not zh.search(translated)]
    missing_numbers = sorted(set(NUMBER.findall(original)) - set(NUMBER.findall(translated)))
    source_urls = {s.rstrip(".,;:!?)]}，。；：！？）】") for s in URL.findall(original)}
    missing_urls = sorted(url for url in source_urls if url not in translated)
    ratio = target_content_tokens / max(1, source_content_tokens)
    flags = {
        "source_over_512": source_tokens > 512,
        "target_reencoded_content_ge_254": target_content_tokens >= 254,
        "replacement_or_nul": "\ufffd" in translated or "\0" in translated,
        "unexpected_unicode_control": any(unicodedata.category(c) == "Cf" or
            (unicodedata.category(c) == "Cc" and c not in "\t\r\n") for c in translated),
        "no_cjk_despite_latin_source": len(re.findall(r"[A-Za-z]", original)) >= 20 and not CJK.search(translated),
        "low_target_token_ratio": source_content_tokens >= 40 and ratio < 0.35,
        "high_target_token_ratio": source_content_tokens >= 16 and ratio > 3.0,
        "long_identical_character_run": bool(re.search(r"([^\s])\1{11,}", translated)),
        "missing_concept_lexicon_candidate": bool(missing_concepts),
        "missing_numeric_literal_candidate": bool(missing_numbers),
        "missing_url_literal_candidate": bool(missing_urls),
    }
    assert set(flags) == set(RISK_RULES)
    return {**{f"risk_{name}": bool(value) for name, value in flags.items()},
            "risk_any": any(flags.values()), "risk_count": sum(bool(v) for v in flags.values()),
            "missing_concept_candidates": missing_concepts, "missing_numeric_literal_candidates": missing_numbers,
            "missing_url_literal_candidates": missing_urls, "target_source_content_token_ratio": ratio,
            "source_contains_cjk": bool(CJK.search(original))}


def run_diagnostics(root: Path, output: Path, mapping: pd.DataFrame, tokenizer_path: Path) -> None:
    output_names = ("source_diagnostics.parquet", "english_identity_groups.parquet", "curated_translation_matches.parquet",
                    "diagnostics.json")
    if any((output / name).exists() for name in output_names):
        raise FileExistsError("Diagnostic artifacts already exist; use a fresh output directory.")
    manifest = json.loads((output / "mapping_manifest.json").read_text())
    assert sha256(output / "source_mapping.parquet") == manifest["mapping_sha256"]
    for name, info in manifest["source_files"].items():
        assert sha256(root / name) == info["sha256"], f"Input changed since mapping: {name}"
    identity_summary, identity_groups, normalized_hash = identity_audit(mapping)
    curated_summary, curated_matches = curated_lineage_audit(root)
    from transformers import AutoTokenizer
    import transformers
    tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_path), local_files_only=True)
    tokenizer.src_lang = "eng_Latn"
    rows = []
    batch_size = 512
    for start in range(0, len(mapping), batch_size):
        batch = mapping.iloc[start:start + batch_size]
        originals, translated = batch.original_prompt.tolist(), batch.prompt.tolist()
        src = tokenizer(originals, add_special_tokens=True, truncation=False)["input_ids"]
        src_content = tokenizer(originals, add_special_tokens=False, truncation=False)["input_ids"]
        tgt = tokenizer(translated, add_special_tokens=False, truncation=False)["input_ids"]
        for original, target, a, b, c in zip(originals, translated, src, src_content, tgt):
            rows.append({"source_nllb_tokens": len(a), "source_nllb_content_tokens": len(b),
                         "old_target_nllb_content_tokens": len(c),
                         **text_risks(original, target, len(a), len(b), len(c))})
        if start % 4096 == 0 or start + batch_size >= len(mapping):
            print(f"Quality screening {min(start + batch_size, len(mapping))}/{len(mapping)}", flush=True)
    metadata = mapping[["base_id", "split", "group_id", "source_row", "original_row", "english_text_sha256"]].copy()
    metadata["english_normalized_sha256"] = normalized_hash
    diagnostics = pd.concat([metadata.reset_index(drop=True), pd.DataFrame(rows)], axis=1)
    rule_counts = {name: int(diagnostics[f"risk_{name}"].sum()) for name in RISK_RULES}
    counts_by_split = {split: {name: int(frame[f"risk_{name}"].sum()) for name in RISK_RULES}
                       for split, frame in diagnostics.groupby("split")}
    tokenizer_hashes = {p.name: sha256(p) for p in tokenizer_path.iterdir() if p.name in
                       {"tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", "sentencepiece.bpe.model"}}
    summary = {
        "protocol": "uniform_text_only_translation_candidate_screen",
        "mapped_rows": len(mapping), "rules": RISK_RULES,
        "missing_concept_lexicon": {name: {"english_regex": en, "target_equivalent_regex": zh}
                                    for name, (en, zh) in CONCEPTS.items()},
        "candidate_counts": rule_counts, "any_candidate_rows": int(diagnostics.risk_any.sum()),
        "candidate_counts_by_split": counts_by_split, "english_identity": identity_summary,
        "curated_source_review": curated_summary,
        "tokenizer": {"path": str(tokenizer_path), "file_sha256": tokenizer_hashes,
                      "class": type(tokenizer).__name__, "transformers_version": transformers.__version__,
                      "source_language": "eng_Latn", "truncation": False,
                      "historical_tokenizer_hash_available": False},
        "historical_generation": {"input_limit": 512, "max_new_tokens_at_call": 256,
                                  "source": "legacy/scripts/translate_wildguard.py:91-105",
                                  "generation_ids_or_stopping_reasons_available": False},
        "code_sha256": sha256(Path(__file__)), "mapping_sha256": sha256(output / "source_mapping.parquet"),
        "no_predictions_or_model_error_sets_used": True, "no_data_or_labels_changed": True,
        "limitations": ["All flags are candidates; no automatic deletion, label change, or semantic-error adjudication.",
                        "Historical tokenizer byte versions and generation traces are unavailable.",
                        "Re-encoding a decoded target cannot prove the historical generation limit was hit.",
                        "Concept/number/URL rules have false positives and false negatives; missing synonyms are not proof of missing meaning.",
                        "English normalization only uses NFKC, casefold, whitespace and Cf removal; no semantic family expansion.",
                        "Repairing translations can expose cross-split exact duplicates; the builder must handle these before release."],
    }
    for name, frame in (("source_diagnostics.parquet", diagnostics), ("english_identity_groups.parquet", identity_groups),
                        ("curated_translation_matches.parquet", curated_matches)):
        write_new_parquet(output / name, frame)
    summary["artifact_sha256"] = {name: sha256(output / name) for name in output_names if name.endswith(".parquet")}
    write_new_json(output / "diagnostics.json", summary)
    print(json.dumps({"candidate_counts": rule_counts, "english_identity": identity_summary,
                      "curated_source_review": curated_summary}, ensure_ascii=False), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--tokenizer", type=Path, default=DEFAULT_TOKENIZER)
    parser.add_argument("--mapping-only", action="store_true")
    parser.add_argument("--diagnostics-only", action="store_true")
    args = parser.parse_args()
    if args.mapping_only and args.diagnostics_only:
        parser.error("Choose at most one stage-only option.")
    if args.diagnostics_only:
        mapping = pd.read_parquet(args.output_dir / "source_mapping.parquet")
    else:
        mapping = build_mapping(args.root, args.output_dir)
    if not args.mapping_only:
        run_diagnostics(args.root, args.output_dir, mapping, args.tokenizer)


if __name__ == "__main__":
    main()
