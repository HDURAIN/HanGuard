"""Tamper and failure-retention boundaries for the independent readback audit."""
import copy
import json
from pathlib import Path
import tempfile
import unittest

import pandas as pd

from scripts.hanguard.audit_translation_repair_outputs import (
    sha, file_sha, run, verify_raw, verify_merged, verify_dataset, write_release_qa,
)
from scripts.hanguard.build_translation_repaired_dataset import build
from scripts.hanguard.merge_translation_repair import prepare_records
from scripts.hanguard.translation_literal_guard import protect_literals, restore_literals
from test_merge_translation_repair import fixtures, override


class CharacterTokenizer:
    def __call__(self, texts, *, add_special_tokens, truncation):
        assert add_special_tokens is False and truncation is False
        return {"input_ids": [list(text) for text in texts]}


class IndependentTranslationReadbackTest(unittest.TestCase):
    def setUp(self):
        self.mapping, self.raw = fixtures()
        self.tokenizer = CharacterTokenizer()

    def literal_record(self, missing=False):
        source = "Explain `x()`."
        original_hash = self.raw[0]["original_english_sha256"]
        selected = self.mapping.english_text_sha256.eq(original_hash)
        self.mapping.loc[selected, "original_prompt"] = source
        self.mapping.loc[selected, "english_text_sha256"] = sha(source)
        _, spans = protect_literals(source)
        raw_translation = "解释。" if missing else "解释 " + spans[0]["placeholder"] + "。"
        translation, issues = restore_literals(raw_translation, spans)
        self.raw[0].update(original_english=source, original_english_sha256=sha(source),
                           raw_translation=raw_translation, translation=translation,
                           literal_spans=spans, literal_issues=issues)

    def bundle(self, overrides=()):
        raw_summary, raw_index, raw_hard = verify_raw(self.mapping, self.raw)
        prepared = prepare_records(self.mapping, self.raw, overrides=overrides)
        merged_summary, merged_index = verify_merged(self.mapping, prepared["translations"], raw_index, raw_hard)
        return raw_summary, prepared, merged_summary, merged_index

    def test_restoration_is_independently_replayed_not_only_flag_trusted(self):
        self.literal_record()
        summary, _, hard = verify_raw(self.mapping, self.raw)
        self.assertTrue(summary["literal_restoration_byte_exact"])
        self.assertFalse(hard)
        self.raw[0]["translation"] += "changed"
        with self.assertRaisesRegex(ValueError, "replay differs"):
            verify_raw(self.mapping, self.raw)

    def test_recorded_literal_failure_is_retained_and_requires_quarantine(self):
        self.literal_record(missing=True)
        raw_summary, prepared, _, _ = self.bundle()
        self.assertEqual(raw_summary["literal_hard_fail_unique_english"], 1)
        affected = [r for r in prepared["translations"] if r["original_english"] == "Explain `x()`."]
        self.assertEqual([r["status"] for r in affected], ["error", "error"])
        _, raw_index, hard = verify_raw(self.mapping, self.raw)
        affected[0]["status"] = "ok"
        with self.assertRaisesRegex(ValueError, "hard failure bypassed"):
            verify_merged(self.mapping, prepared["translations"], raw_index, hard)

    def test_changed_failure_list_source_hash_and_manifest_do_not_pass(self):
        self.literal_record(missing=True)
        original = copy.deepcopy(self.raw)
        self.raw[0]["literal_issues"] = []
        with self.assertRaisesRegex(ValueError, "literal issues disagree"):
            verify_raw(self.mapping, self.raw)
        self.raw = copy.deepcopy(original)
        self.raw[0]["original_english"] += "changed"
        with self.assertRaisesRegex(ValueError, "hash mismatch"):
            verify_raw(self.mapping, self.raw)
        self.raw = copy.deepcopy(original)
        self.raw[0]["literal_spans"][0]["text"] = "`z()`"
        with self.assertRaisesRegex(ValueError, "manifest text/hash"):
            verify_raw(self.mapping, self.raw)

    def test_raw_coverage_duplicate_and_wrong_code_issue_list_rejected(self):
        with self.assertRaisesRegex(ValueError, "exactly cover"):
            verify_raw(self.mapping, self.raw[:-1])
        with self.assertRaisesRegex(ValueError, "Duplicate raw"):
            verify_raw(self.mapping, self.raw + [self.raw[0]])
        self.raw[0]["unprotected_code_issues"] = [{"code": "invented", "severity": "review"}]
        with self.assertRaisesRegex(ValueError, "code issues disagree"):
            verify_raw(self.mapping, self.raw)

    def test_corrected_translation_original_and_new_text_evidence_are_verified(self):
        review = override(self.raw[0], "correct", corrected_translation="您好。")
        _, prepared, summary, _ = self.bundle(overrides=[review])
        self.assertEqual(summary["recorded_correction_rows"], 2)
        _, raw_index, hard = verify_raw(self.mapping, self.raw)
        row = next(r for r in prepared["translations"] if r["original_english"] == "Hello.")
        row["translation_source"] = "model"
        with self.assertRaisesRegex(ValueError, "disguised as model"):
            verify_merged(self.mapping, prepared["translations"], raw_index, hard)

    def test_unrecorded_text_edits_and_forged_original_generation_fail(self):
        _, prepared, _, _ = self.bundle()
        _, raw_index, hard = verify_raw(self.mapping, self.raw)
        original = copy.deepcopy(prepared["translations"])
        row = prepared["translations"][0]
        row["translation"] = "changed"
        row["translation_sha256"] = sha("changed")
        with self.assertRaisesRegex(ValueError, "Unrecorded translation"):
            verify_merged(self.mapping, prepared["translations"], raw_index, hard)
        original[0]["raw_generation_record"]["generation_tokens"] += 1
        with self.assertRaisesRegex(ValueError, "raw generation record differs"):
            verify_merged(self.mapping, original, raw_index, hard)

    def test_dataset_exact_partition_labels_correction_provenance_and_extra_exclusion(self):
        review = override(self.raw[0], "correct", corrected_translation="您好。")
        _, prepared, _, merged = self.bundle(overrides=[review])
        originals = self.mapping.assign(group_id=self.mapping.base_id.map(lambda key: "group_" + key))
        originals = originals.drop(columns=["original_prompt", "english_text_sha256"])
        originals = {split: frame.reset_index(drop=True) for split, frame in originals.groupby("split")}
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "dataset"
            build(originals, target, prepared["translations"], mapping=self.mapping, tokenizer=self.tokenizer)
            archive = pd.read_parquet(target / "full_repaired_archive.parquet")
            quarantine = pd.read_parquet(target / "quarantine.parquet")
            splits = {s: pd.read_parquet(target / f"{s}.parquet") for s in originals}
            result = verify_dataset(originals, archive, splits, quarantine, merged, self.tokenizer)
            self.assertEqual((result["full_archive_rows"], result["candidate_rows"], result["quarantine_rows"]), (6, 2, 4))
            wrong_labels = archive.copy()
            wrong_labels.loc[0, "prompt_harm_label"] = "harmful"
            with self.assertRaisesRegex(ValueError, "stable field changed"):
                verify_dataset(originals, wrong_labels, splits, quarantine, merged, self.tokenizer)
            wrong_source = archive.copy()
            wrong_source.loc[wrong_source.translation_source.eq("recorded_bilingual_correction"), "translation_source"] = "model"
            with self.assertRaisesRegex(ValueError, "correction provenance differs"):
                verify_dataset(originals, wrong_source, splits, quarantine, merged, self.tokenizer)
            with self.assertRaisesRegex(ValueError, "exactly partition"):
                verify_dataset(originals, archive, splits, quarantine.iloc[:-1], merged, self.tokenizer)
            # A tamperer updates all three artifacts consistently to hide an
            # unnecessary exclusion: independent rule recomputation still fails.
            moved = splits["validation"].copy()
            extra = archive.copy()
            extra.loc[extra.base_id.isin(moved.base_id), "quarantine_reason"] = "arbitrary_exclusion"
            extra.loc[extra.base_id.isin(moved.base_id), "repair_eligible"] = False
            moved = extra[extra.base_id.isin(moved.base_id)]
            wrong_splits = dict(splits, validation=splits["validation"].iloc[:0])
            wrong_quarantine = pd.concat([quarantine, moved], ignore_index=True)
            with self.assertRaisesRegex(ValueError, "independently recomputed"):
                verify_dataset(originals, extra, wrong_splits, wrong_quarantine, merged, self.tokenizer)

    def test_full_readback_and_explicit_release_sidecar_preserve_manifests_and_require_sample_coverage(self):
        reviews = [override(row) for row in self.raw]
        prepared = prepare_records(self.mapping, self.raw, overrides=reviews)
        originals = self.mapping.assign(group_id=self.mapping.base_id.map(lambda key: "group_" + key))
        originals = originals.drop(columns=["original_prompt", "english_text_sha256"])
        originals = {split: frame.reset_index(drop=True) for split, frame in originals.groupby("split")}
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source, production, dataset, output = [root / name for name in ("original", "production", "dataset", "readback")]
            source.mkdir()
            production.mkdir()
            for split, frame in originals.items():
                frame.to_parquet(source / f"{split}.parquet", index=False)
            mapping = root / "mapping.parquet"
            self.mapping.to_parquet(mapping, index=False)
            merged = root / "translation_results.jsonl"
            merged.write_text("".join(json.dumps(r) + "\n" for r in prepared["translations"]))
            (production / "raw_0.jsonl").write_text("".join(json.dumps(r) + "\n" for r in self.raw))
            guard = Path(__file__).resolve().parents[1] / "scripts/hanguard/translation_literal_guard.py"
            (production / "protocol_0.json").write_text(json.dumps(dict(pilot=0, shard=0, shards=1,
                literal_guard_sha256=file_sha(guard), mapping_sha256=file_sha(mapping), special_literals=[])))
            (production / "status_0.json").write_text(json.dumps({"state": "translating"}))
            build(originals, dataset, prepared["translations"], mapping=self.mapping, tokenizer=self.tokenizer)
            with self.assertRaisesRegex(ValueError, "not complete"):
                run(mapping, production, merged, source, dataset, self.tokenizer, output)
            self.assertFalse(output.exists())
            (production / "status_0.json").write_text(json.dumps({"state": "complete"}))
            result = run(mapping, production, merged, source, dataset, self.tokenizer, output)
            self.assertTrue(result["passed"])
            self.assertFalse((dataset / "release_qa.json").exists())
            immutable = {p: p.read_bytes() for p in [dataset / "manifest.json", dataset / "audit.json"]}
            plan = root / "sample_plan.jsonl"
            samples = [dict(base_id=r.base_id, split=r.split, original_english=r.original_prompt,
                            original_english_sha256=r.english_text_sha256, qa_seed=20260928,
                            qa_selection_reason="fixed_hash_random") for r in self.mapping.itertuples()]
            plan.write_text("".join(json.dumps(r) + "\n" for r in samples))
            evidence = root / "bilingual_reviews.jsonl"
            evidence.write_text("".join(json.dumps(r) + "\n" for r in reviews[:-1]))
            kwargs = dict(sample_plan_sha256=file_sha(plan), qa_evidence_sha256={str(evidence): file_sha(evidence)},
                          decision_by="Codex root agent", reason="Toy verification and fixed sample reviewed.", per_split=2)
            with self.assertRaisesRegex(ValueError, "lacks matching"):
                write_release_qa(dataset, output / "verification.json", plan, **kwargs)
            self.assertFalse((dataset / "release_qa.json").exists())
            evidence.write_text("".join(json.dumps(r) + "\n" for r in reviews))
            with self.assertRaisesRegex(ValueError, "evidence SHA differs"):
                write_release_qa(dataset, output / "verification.json", plan, **kwargs)
            # The root assembler's comparison artifact binds complete text,
            # while its summary is JSON rather than line-delimited reviews.
            comparisons = [dict(original_english_sha256=r["original_english_sha256"],
                                original_source=r["original_english"], model_translation=r["translation"],
                                reviewed_translation=r["translation"], decision="approve") for r in self.raw]
            evidence.write_text("".join(json.dumps(r) + "\n" for r in comparisons))
            summary_path = root / "review_summary.json"
            summary_path.write_text(json.dumps({"fixed_random_rows": 6}, indent=2))
            kwargs["qa_evidence_sha256"] = {str(evidence): file_sha(evidence), str(summary_path): file_sha(summary_path)}
            released = write_release_qa(dataset, output / "verification.json", plan, **kwargs)
            self.assertEqual(released["release_status"], "qa_complete_with_documented_limitations")
            self.assertEqual(released["sample_plan"]["rows"], 6)
            self.assertFalse(released["expert_human_annotation"])
            for path, expected in immutable.items():
                self.assertEqual(path.read_bytes(), expected)
            with self.assertRaisesRegex(ValueError, "already exists"):
                write_release_qa(dataset, output / "verification.json", plan, **kwargs)

    def test_independent_no_cap_audit_retains_long_text_and_rejects_length_only_exclusion(self):
        full = "长" * 500
        self.raw[2].update(translation=full, raw_translation=full, generation_tokens=500, max_new_tokens=1024)
        reviews = [override(row) for row in self.raw]
        _, prepared, _, merged = self.bundle(overrides=reviews)
        originals = self.mapping.assign(group_id=self.mapping.base_id.map(lambda key: "group_" + key))
        originals = originals.drop(columns=["original_prompt", "english_text_sha256"])
        originals = {split: frame.reset_index(drop=True) for split, frame in originals.groupby("split")}
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "dataset"
            build(originals, target, prepared["translations"], mapping=self.mapping, tokenizer=self.tokenizer)
            archive = pd.read_parquet(target / "full_repaired_archive.parquet")
            quarantine = pd.read_parquet(target / "quarantine.parquet")
            splits = {s: pd.read_parquet(target / f"{s}.parquet") for s in originals}
            result = verify_dataset(originals, archive, splits, quarantine, merged, self.tokenizer)
            self.assertIsNone(result["max_prompt_tokens"])
            self.assertEqual(result["candidate_over_legacy_limit"], 1)
            self.assertTrue(result["legacy_length_is_annotation_only"])
            self.assertEqual(splits["validation"].iloc[0].prompt, full)
            with self.assertRaisesRegex(ValueError, "explicitly configured length cap"):
                verify_dataset(originals, archive, splits, quarantine, merged, self.tokenizer, max_tokens=370)
            archive.loc[archive.prompt.eq(full), "quarantine_reason"] = "overlength_full_text_retained"
            archive.loc[archive.prompt.eq(full), "repair_eligible"] = False
            moved = archive[archive.prompt.eq(full)]
            changed_splits = dict(splits, validation=splits["validation"].iloc[:0])
            changed_quarantine = pd.concat([quarantine, moved], ignore_index=True)
            with self.assertRaisesRegex(ValueError, "independently recomputed"):
                verify_dataset(originals, archive, changed_splits, changed_quarantine, merged, self.tokenizer, max_tokens=None)


if __name__ == "__main__":
    unittest.main()
