"""CPU boundaries for complete, label-blind production translation merging."""
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import pandas as pd

from scripts.hanguard.merge_translation_repair import (
    FORBIDDEN_KEYS, merge, prepare_records, read_jsonl, sha256_text,
)


def fixtures():
    sources = {"Hello.": "你好。", "Good morning.": "早上好。",
               "Thank you.": "谢谢。", "Good night.": "晚安。"}
    allocation = [("train", "Hello."), ("train", "Good morning."),
                  ("validation", "Good morning."), ("validation", "Thank you."),
                  ("test", "Hello."), ("test", "Good night.")]
    mapping = pd.DataFrame([dict(base_id=f"base_{i}", split=split, source_row=i,
                                source="wildguard_zh", prompt=f"旧中文{i}",
                                original_prompt=english, english_text_sha256=sha256_text(english),
                                prompt_harm_label="unharmful", category_id="0")
                            for i, (split, english) in enumerate(allocation)])
    raw = [dict(original_english=english, original_english_sha256=sha256_text(english),
                translation=translation, finish_reason="stop", generation_tokens=3,
                max_new_tokens=256, eos_reached=True, input_truncated=False,
                model="toy-translator", prompt_sha256=sha256_text("prompt"),
                rendered_prompt_sha256=sha256_text("rendered " + english),
                raw_translation=translation, literal_spans=[], literal_issues=[],
                unprotected_code_issues=[], generation_issues=[], backend="toy")
           for english, translation in sources.items()]
    return mapping, raw


def override(raw, decision="approve", **changes):
    record = dict(original_english_sha256=raw["original_english_sha256"],
                  translation_sha256=sha256_text(raw["translation"]), decision=decision,
                  reviewer="Codex agent (synthetic test)", reason="Compared the complete source and target texts.",
                  basis="bilingual_text_review", label_blind=True)
    record.update(changes)
    return record


class TranslationMergeTest(unittest.TestCase):
    def setUp(self):
        self.mapping, self.raw = fixtures()

    def prepare(self, **kwargs):
        return prepare_records(self.mapping, self.raw, **kwargs)

    def replace_first_source(self, english, translation):
        from scripts.hanguard.translation_literal_guard import protect_literals
        old_hash = self.raw[0]["original_english_sha256"]
        affected = self.mapping.english_text_sha256.eq(old_hash)
        self.mapping.loc[affected, "original_prompt"] = english
        self.mapping.loc[affected, "english_text_sha256"] = sha256_text(english)
        _, spans = protect_literals(english)
        self.raw[0].update(original_english=english, original_english_sha256=sha256_text(english),
                           translation=translation, raw_translation=translation, literal_spans=spans)

    def test_fanout_preserves_generation_and_immutable_lineage_without_labels(self):
        self.raw[0]["raw_translation"] = "MASKED MODEL OUTPUT"
        before_mapping, before_raw = self.mapping.copy(deep=True), copy.deepcopy(self.raw)
        result = self.prepare()
        self.assertEqual(result["summary"]["rows"], 6)
        self.assertEqual(result["summary"]["unique_english"], 4)
        self.assertEqual(result["summary"]["final_status_rows"], {"ok": 6})
        by_id = {row["base_id"]: row for row in result["translations"]}
        indexed_raw = {r["original_english_sha256"]: r for r in self.raw}
        for original in self.mapping.itertuples():
            new = by_id[original.base_id]
            self.assertEqual(new["split"], original.split)
            self.assertEqual(new["source_row"], original.source_row)
            self.assertEqual(new["old_prompt"], original.prompt)
            self.assertEqual(new["raw_generation_record"], indexed_raw[original.english_text_sha256])
            self.assertEqual(new["translation_sha256"], sha256_text(new["translation"]))
            self.assertFalse(set(new) & FORBIDDEN_KEYS)
        self.assertEqual(by_id["base_0"]["translation"], "你好。")
        self.assertEqual(by_id["base_0"]["automatic_qa_status"], "clean")
        self.assertEqual(by_id["base_0"]["translation"], by_id["base_4"]["translation"])
        self.assertFalse(any(set(row) & FORBIDDEN_KEYS for row in result["sample"]))
        pd.testing.assert_frame_equal(self.mapping, before_mapping)
        self.assertEqual(self.raw, before_raw)

    def test_labels_and_predictions_in_mapping_do_not_change_qa_or_sampling(self):
        expected = self.prepare()
        self.mapping["prompt_harm_label"] = "harmful"
        self.mapping["category_id"] = "arbitrary"
        self.mapping["probability"] = .99
        self.mapping["prediction"] = 1
        self.assertEqual(self.prepare(), expected)

    def test_coverage_unknown_duplicate_and_hash_mismatch_are_rejected(self):
        original = copy.deepcopy(self.raw)
        cases = [original[:-1], original + [copy.deepcopy(original[0])]]
        unknown = copy.deepcopy(original[0])
        unknown.update(original_english="Unknown source", original_english_sha256=sha256_text("Unknown source"))
        cases.append(original + [unknown])
        mismatch = copy.deepcopy(original)
        mismatch[0]["original_english"] += " changed"
        cases.append(mismatch)
        for raw in cases:
            with self.subTest(raw_count=len(raw)), self.assertRaises(ValueError):
                prepare_records(self.mapping, raw)

    def test_mapping_identity_text_hash_and_split_validation(self):
        for column, value in (("base_id", "base_1"), ("english_text_sha256", "wrong"),
                              ("original_prompt", ""), ("source", "other"), ("split", "unused")):
            altered = self.mapping.copy()
            altered.at[0, column] = value
            with self.subTest(column=column), self.assertRaises(ValueError):
                prepare_records(altered, self.raw)

    def test_mixed_generator_configuration_and_classifier_fields_rejected(self):
        for key in ("model", "prompt_sha256", "literal_guard_version"):
            changed = copy.deepcopy(self.raw)
            changed[0][key] = "different"
            with self.subTest(key=key), self.assertRaises(ValueError):
                prepare_records(self.mapping, changed)
        self.raw[0]["metadata"] = {"prediction": 1}
        with self.assertRaisesRegex(ValueError, "Classifier labels/predictions"):
            self.prepare()

    def test_all_three_issue_sources_hard_fail_and_review_remain_distinct(self):
        for field in ("literal_issues", "unprotected_code_issues", "generation_issues"):
            for severity, expected in (("hard_fail", "error"), ("review", "review")):
                raw = copy.deepcopy(self.raw)
                raw[0][field] = [dict(code="test_quality_flag", severity=severity, evidence={"detail": "full evidence"})]
                result = prepare_records(self.mapping, raw)
                affected = [r for r in result["translations"] if r["original_english"] == "Hello."]
                with self.subTest(field=field, severity=severity):
                    self.assertEqual([r["status"] for r in affected], [expected, expected])
                    self.assertTrue(all(any(f["origin"] == field for f in r["qa_flags"] if "origin" in f)
                                        for r in affected))
                    self.assertEqual(result["summary"]["final_status_unique_english"][expected], 1)

    def test_missing_malformed_issue_evidence_is_quarantined(self):
        for field in ("literal_issues", "unprotected_code_issues", "generation_issues"):
            for bad in (None, {}, ["bad"], [{"code": "bad", "severity": "clean"}]):
                raw = copy.deepcopy(self.raw)
                if bad is None:
                    raw[0].pop(field)
                else:
                    raw[0][field] = bad
                with self.subTest(field=field, bad=bad):
                    result = prepare_records(self.mapping, raw)
                    self.assertEqual(result["summary"]["final_status_rows"]["error"], 2)

    def test_finish_error_status_and_empty_text_cannot_be_approved(self):
        changes = [{"finish_reason": None}, {"finish_reason": "unknown"}, {"finish_reason": "length"},
                   {"status": "parse_failure"}, {"status": "error"}, {"translation": ""},
                   {"input_truncated": True}, {"eos_reached": False, "generation_tokens": 256}]
        for change in changes:
            raw = copy.deepcopy(self.raw)
            raw[0].update(change)
            with self.subTest(change=change):
                result = prepare_records(self.mapping, raw)
                self.assertEqual(result["summary"]["final_status_rows"]["error"], 2)
                with self.assertRaisesRegex(ValueError, "Cannot approve"):
                    prepare_records(self.mapping, raw, overrides=[override(raw[0])])
                held = prepare_records(self.mapping, raw, overrides=[override(raw[0], "hold")])
                self.assertEqual(held["summary"]["final_status_rows"]["error"], 2)

    def test_bilingual_approval_applies_all_ids_and_keeps_original_flags(self):
        self.raw[0]["generation_issues"] = [dict(code="possible_new_control", severity="review", evidence="review only")]
        automatic = self.prepare()
        self.assertEqual(automatic["summary"]["final_status_rows"]["review"], 2)
        manual = override(self.raw[0])
        approved = self.prepare(overrides=[manual])
        self.assertEqual(approved["summary"]["final_status_rows"], {"ok": 6})
        rows = [r for r in approved["translations"] if r["original_english"] == "Hello."]
        self.assertEqual({r["split"] for r in rows}, {"train", "test"})
        for row in rows:
            self.assertEqual(row["automatic_qa_status"], "review")
            self.assertEqual(row["review_override"], manual)
            self.assertTrue(row["qa_flags"])
        self.assertEqual(approved["summary"]["override_unique_english"], 1)

    def test_stale_unmatched_conflicting_and_nonblind_overrides_rejected(self):
        changes = [{"translation_sha256": sha256_text("one character changed")},
                   {"original_english_sha256": sha256_text("unknown")},
                   {"reviewer": ""}, {"reason": " "}, {"label_blind": False},
                   {"label_blind": 1}, {"basis": "classifier_error"},
                   {"base_id": "base_0"}, {"label": "harmful"}, {"decision": ""}]
        for change in changes:
            with self.subTest(change=change), self.assertRaises(ValueError):
                self.prepare(overrides=[override(self.raw[0], **change)])
        for second in ("approve", "reject"):
            with self.assertRaisesRegex(ValueError, "Duplicate or conflicting"):
                self.prepare(overrides=[override(self.raw[0]), override(self.raw[0], second)])

    def test_reject_clean_is_explicit_error_without_losing_automatic_clean(self):
        result = self.prepare(overrides=[override(self.raw[0], "reject")])
        self.assertEqual(result["summary"]["final_status_rows"]["error"], 2)
        self.assertEqual(result["summary"]["automatic_status_rows"], {"clean": 6})

    def test_recorded_correction_keeps_model_text_original_audit_and_all_lineages(self):
        self.raw[0]["translation"] = "译文：你好。"
        original = copy.deepcopy(self.raw[0])
        review = override(self.raw[0], "correct", corrected_translation="您好。")
        result = self.prepare(overrides=[review])
        self.assertEqual(result["summary"]["corrected_unique_english"], 1)
        self.assertEqual(result["summary"]["corrected_rows"], 2)
        rows = [r for r in result["translations"] if r["original_english"] == "Hello."]
        self.assertEqual({r["split"] for r in rows}, {"train", "test"})
        for row in rows:
            self.assertEqual(row["translation"], "您好。")
            self.assertEqual(row["translation_sha256"], sha256_text("您好。"))
            self.assertEqual(row["model_translation"], "译文：你好。")
            self.assertEqual(row["model_translation_sha256"], sha256_text("译文：你好。"))
            self.assertEqual(row["raw_generation_record"], original)
            self.assertEqual(row["automatic_qa_status"], "review")
            self.assertEqual(row["original_model_audit"]["status"], "review")
            self.assertEqual(row["corrected_translation_audit"]["status"], "clean")
            self.assertTrue(row["corrected_translation_audit"]["literal_preservation_verified"])
            self.assertEqual(row["translation_source"], "recorded_bilingual_correction")
            self.assertEqual(row["correction_source"], "recorded_bilingual_review")
            self.assertEqual(row["status"], "ok")
        sample = next(r for r in result["sample"] if r["base_id"] == rows[0]["base_id"])
        self.assertEqual(sample["translation"], "您好。")
        self.assertEqual(sample["model_translation"], "译文：你好。")
        self.assertEqual(sample["review_override"], review)
        self.assertFalse(any(r["original_english_sha256"] == review["original_english_sha256"]
                             for r in result["override_template"]))
        self.assertEqual(self.raw[0], original)

    def test_correction_must_bind_original_translation_and_not_bypass_hard_fail(self):
        for field, value in (("finish_reason", "length"), ("translation", ""), ("input_truncated", True)):
            raw = copy.deepcopy(self.raw)
            raw[0][field] = value
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, "Cannot approve or correct"):
                prepare_records(self.mapping, raw, overrides=[override(raw[0], "correct", corrected_translation="您好。")])
        review = override(self.raw[0], "correct", corrected_translation="您好。",
                          translation_sha256=sha256_text("您好。"))
        with self.assertRaisesRegex(ValueError, "Stale override"):
            self.prepare(overrides=[review])

    def test_correction_schema_requires_changed_complete_text_and_correct_decision(self):
        for corrected in (None, "", " ", 3, "你好。"):
            with self.subTest(corrected=corrected), self.assertRaises(ValueError):
                self.prepare(overrides=[override(self.raw[0], "correct", corrected_translation=corrected)])
        with self.assertRaises(ValueError):
            self.prepare(overrides=[override(self.raw[0], "correct")])
        with self.assertRaisesRegex(ValueError, "only permitted"):
            self.prepare(overrides=[override(self.raw[0], "approve", corrected_translation="您好。")])

    def test_correction_keeps_heuristic_flags_as_review_evidence_without_claiming_gold(self):
        review = override(self.raw[0], "correct", corrected_translation="译文：您好。")
        result = self.prepare(overrides=[review])
        row = next(r for r in result["translations"] if r["original_english"] == "Hello.")
        self.assertEqual(row["status"], "ok")  # Explicit text-review decision, not a clean-rule claim.
        self.assertEqual(row["effective_qa_status"], "review")
        self.assertTrue(row["corrected_translation_audit"]["flags"])
        self.assertIn("Codex agent", row["review_override"]["reviewer"])

    def test_correction_rejects_code_changes_reordering_omission_repetition_and_additions(self):
        self.replace_first_source("Explain `x()` and `y()`.", "解释 `x()` 和 `y()`。")
        valid = "请说明 `x()` 和 `y()`。"
        result = self.prepare(overrides=[override(self.raw[0], "correct", corrected_translation=valid)])
        self.assertEqual(result["summary"]["corrected_rows"], 2)
        bad = ["说明 `z()` 和 `y()`。", "说明 `y()` 和 `x()`。", "说明 `x()`。",
               "说明 `x()` 和 `x()` 和 `y()`。", "说明 `x()` 和 `y()` 和 `z()`。"]
        for corrected in bad:
            with self.subTest(corrected=corrected), self.assertRaisesRegex(ValueError, "protected literals"):
                self.prepare(overrides=[override(self.raw[0], "correct", corrected_translation=corrected)])
        self.raw[0]["literal_spans"][0]["sha256"] = "tampered"
        with self.assertRaisesRegex(ValueError, "manifest does not match"):
            self.prepare(overrides=[override(self.raw[0], "correct", corrected_translation=valid)])

    def test_existing_duplicate_code_control_inside_code_and_unclosed_fence_are_preserved(self):
        cases = [("Explain `x()` then `x()`.", "解释 `x()` 然后 `x()`。", "请说明 `x()` 然后 `x()`。"),
                 ("Explain:\n```text\n<|im_start|>\n```", "解释：\n```text\n<|im_start|>\n```", "说明如下：\n```text\n<|im_start|>\n```"),
                 ("Explain:\n```python\nprint(1)", "解释：\n```python\nprint(1)", "说明如下：\n```python\nprint(1)")]
        for english, original, corrected in cases:
            self.mapping, self.raw = fixtures()
            self.replace_first_source(english, original)
            with self.subTest(english=english):
                result = self.prepare(overrides=[override(self.raw[0], "correct", corrected_translation=corrected)])
                self.assertEqual(result["summary"]["corrected_rows"], 2)
        with self.assertRaisesRegex(ValueError, "protected literals"):
            self.prepare(overrides=[override(self.raw[0], "correct", corrected_translation=cases[-1][2] + "\n```")])

    def test_new_placeholder_or_control_string_cannot_enter_via_correction(self):
        additions = ("[[HG_LITERAL_fake_0000]]", "[[HG_LITERAL_broken", "{{NAME}}", "[PERSON]",
                     "{NAME}", "NAME_1", "<|new_control|>", "<｜hy_begin▁of▁sentence｜>",
                     "<think>", "[INST]", "<EXTRA_MODEL_CONTROL>")
        for addition in additions:
            with self.subTest(addition=addition), self.assertRaises(ValueError):
                self.prepare(overrides=[override(self.raw[0], "correct", corrected_translation="您好。" + addition)],
                             special_literals=["<EXTRA_MODEL_CONTROL>"])

    def test_source_placeholders_and_preexisting_marker_text_are_not_misclassified_as_new(self):
        for text in ("{{NAME}}", "[[HG_LITERAL_existing_0000]]", "[[HG_LITERAL_broken"):
            self.mapping, self.raw = fixtures()
            self.replace_first_source("Say " + text, "说 " + text)
            result = self.prepare(overrides=[override(self.raw[0], "correct", corrected_translation="请说 " + text)])
            self.assertEqual(result["summary"]["corrected_rows"], 2)
        with self.assertRaisesRegex(ValueError, "placeholder"):
            self.prepare(overrides=[override(self.raw[0], "correct", corrected_translation="请说 [[HG_LITERAL_changed")])

    def test_numbered_placeholders_allow_chinese_neighbors_but_cannot_change_ascii_identifier(self):
        from scripts.hanguard.audit_translation_repair_outputs import _check_corrected_literals
        source = "Teach my neighbour NAME_1 to play guitar."
        self.replace_first_source(source, "教我的邻居 NAME_1 学吉他。")
        corrected = "教邻居NAME_1学吉他。"
        result = self.prepare(overrides=[override(self.raw[0], "correct", corrected_translation=corrected)])
        self.assertEqual(result["summary"]["corrected_rows"], 2)
        _check_corrected_literals(source, corrected, [], ())
        for altered in ("NAME_2", "NAME_12", "prefixNAME_1", "NAME_1suffix", "NAME_1_suffix"):
            bad = "教邻居" + altered + "学吉他。"
            with self.subTest(altered=altered):
                with self.assertRaisesRegex(ValueError, "placeholder"):
                    self.prepare(overrides=[override(self.raw[0], "correct", corrected_translation=bad)])
                with self.assertRaisesRegex(ValueError, "placeholder"):
                    _check_corrected_literals(source, bad, [], ())

    def test_sampling_is_fixed_order_invariant_and_not_filtered_by_qa_status(self):
        expected = self.prepare(per_split=1)
        changed = prepare_records(self.mapping.iloc[::-1], list(reversed(self.raw)), per_split=1)
        self.assertEqual(expected, changed)
        self.raw[0]["finish_reason"] = "length"
        after = self.prepare(per_split=1)
        self.assertEqual([r["base_id"] for r in expected["sample"]], [r["base_id"] for r in after["sample"]])
        self.assertEqual(after["summary"]["sample_by_split"], {"train": 1, "validation": 1, "test": 1})
        self.assertEqual(self.prepare()["summary"]["sampled_rows"], 6)
        with self.assertRaises(ValueError):
            self.prepare(per_split=0)


class TranslationMergeFilesTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.mapping, self.raw = fixtures()
        self.mapping_path = self.root / "source_mapping.parquet"
        self.mapping.to_parquet(self.mapping_path, index=False)
        self.raw_dir = self.root / "production"
        self.raw_dir.mkdir()
        self.raw_path = self.raw_dir / "raw_0.jsonl"
        self.raw_path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in self.raw))
        self.protocol_path = self.raw_dir / "protocol_0.json"
        self.protocol = dict(pilot=0, model=self.raw[0]["model"], prompt_sha256=self.raw[0]["prompt_sha256"],
                             mapping_sha256=hashlib.sha256(self.mapping_path.read_bytes()).hexdigest(),
                             backend="toy", max_model_len=16384, batch_size=32, max_batch_tokens=32768,
                             generator_sha256=sha256_text("generator source"),
                             package_versions={"torch": "toy", "transformers": "toy", "tokenizers": "toy"},
                             model_metadata_sha256={"config.json": sha256_text("model config")},
                             drafts_sha256=None, shard=0, shards=1)
        self.write_protocol()
        self.out = self.root / "merged"

    def tearDown(self):
        self.temp.cleanup()

    def write_protocol(self):
        self.protocol_path.write_text(json.dumps(self.protocol))

    def write_two_shards(self):
        self.protocol["shards"] = 2
        for shard_id in range(2):
            records = [row for row in self.raw if int(row["original_english_sha256"][:12], 16) % 2 == shard_id]
            (self.raw_dir / f"raw_{shard_id}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in records))
            (self.raw_dir / f"protocol_{shard_id}.json").write_text(json.dumps(dict(self.protocol, shard=shard_id)))

    def test_production_files_have_hash_manifest_and_refuse_overwrite(self):
        # A nested pilot is not searched recursively or accidentally concatenated.
        pilot = self.raw_dir / "old_pilot"
        pilot.mkdir()
        (pilot / "raw_0.jsonl").write_text("invalid pilot data")
        originals = {p: p.read_bytes() for p in (self.mapping_path, self.raw_path, self.protocol_path)}
        summary = merge(self.mapping_path, self.raw_dir, self.out)
        self.assertEqual(summary["rows"], 6)
        self.assertEqual(len(read_jsonl(self.out / "translation_results.jsonl")), 6)
        manifest = json.loads((self.out / "merge_manifest.json").read_text())
        self.assertFalse(manifest["pilot_outputs_used"])
        for name, expected in manifest["artifact_sha256"].items():
            self.assertEqual(hashlib.sha256((self.out / name).read_bytes()).hexdigest(), expected)
        for path, expected in originals.items():
            self.assertEqual(path.read_bytes(), expected)
        with self.assertRaises(FileExistsError):
            merge(self.mapping_path, self.raw_dir, self.out)

    def test_pilot_or_mismatched_provenance_is_rejected_before_publish(self):
        original = self.protocol.copy()
        for changes in ({"pilot": 4}, {"pilot": False}, {"mapping_sha256": "other"},
                        {"model": "other"}, {"prompt_sha256": "other"}):
            self.protocol = dict(original, **changes)
            self.write_protocol()
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                merge(self.mapping_path, self.raw_dir, self.out)
            self.assertFalse(self.out.exists())
        self.protocol_path.unlink()
        with self.assertRaisesRegex(ValueError, "protocol missing"):
            merge(self.mapping_path, self.raw_dir, self.out)

    def test_incomplete_raw_or_invalid_override_does_not_publish(self):
        self.raw_path.write_text(json.dumps(self.raw[0]) + "\n")
        with self.assertRaisesRegex(ValueError, "Incomplete production coverage"):
            merge(self.mapping_path, self.raw_dir, self.out)
        self.assertFalse(self.out.exists())
        self.raw_path.write_text("".join(json.dumps(r) + "\n" for r in self.raw))
        stale = self.root / "reviews.jsonl"
        stale.write_text(json.dumps(override(self.raw[0], translation_sha256="stale")) + "\n")
        with self.assertRaisesRegex(ValueError, "Stale override"):
            merge(self.mapping_path, self.raw_dir, self.out, overrides_path=stale)
        self.assertFalse(self.out.exists())

    def test_two_complete_shards_merge_under_identical_execution_protocol(self):
        self.write_two_shards()
        summary = merge(self.mapping_path, self.raw_dir, self.out)
        self.assertEqual(summary["unique_english"], 4)
        manifest = json.loads((self.out / "merge_manifest.json").read_text())
        self.assertEqual(len(manifest["protocols"]), 2)
        self.assertEqual(manifest["production_configuration"]["shards"], 2)

    def test_differing_execution_provenance_or_missing_fields_fails(self):
        self.write_two_shards()
        path = self.raw_dir / "protocol_1.json"
        original = json.loads(path.read_text())
        for field in ("backend", "max_model_len", "batch_size", "max_batch_tokens", "generator_sha256",
                      "package_versions", "model_metadata_sha256", "drafts_sha256", "shards"):
            changed = copy.deepcopy(original)
            value = changed[field]
            changed[field] = (value + 1 if isinstance(value, int) else dict(value, extra="other")
                              if isinstance(value, dict) else sha256_text("draft batch") if value is None else "other")
            path.write_text(json.dumps(changed))
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, "Mixed production shard"):
                merge(self.mapping_path, self.raw_dir, self.out)
            changed = copy.deepcopy(original)
            del changed[field]
            path.write_text(json.dumps(changed))
            with self.subTest(missing=field), self.assertRaisesRegex(ValueError, "missing provenance fields"):
                merge(self.mapping_path, self.raw_dir, self.out)
        self.assertFalse(self.out.exists())

    def test_postedit_keeps_draft_batch_and_requires_matching_raw_metadata(self):
        self.protocol["drafts_sha256"] = sha256_text("complete draft batch file")
        self.write_protocol()
        for row in self.raw:
            row["drafts_sha256"] = self.protocol["drafts_sha256"]
            row["draft_translation"] = "完整初译：" + row["translation"]
        self.raw_path.write_text("".join(json.dumps(r) + "\n" for r in self.raw))
        merge(self.mapping_path, self.raw_dir, self.out)
        rows = read_jsonl(self.out / "translation_results.jsonl")
        self.assertTrue(all(r["drafts_sha256"] == self.protocol["drafts_sha256"] for r in rows))
        self.assertTrue(all(r["draft_translation"].startswith("完整初译：") for r in rows))
        self.raw[0]["drafts_sha256"] = sha256_text("different draft batch")
        self.raw_path.write_text("".join(json.dumps(r) + "\n" for r in self.raw))
        with self.assertRaisesRegex(ValueError, "draft-batch hash differs"):
            merge(self.mapping_path, self.raw_dir, self.root / "other")
        self.raw[0]["drafts_sha256"] = self.protocol["drafts_sha256"]
        self.raw[0].pop("draft_translation")
        self.raw_path.write_text("".join(json.dumps(r) + "\n" for r in self.raw))
        with self.assertRaisesRegex(ValueError, "missing the complete draft text"):
            merge(self.mapping_path, self.raw_dir, self.root / "other")

    def test_missing_shard_or_wrong_shard_identity_rejected(self):
        self.write_two_shards()
        shard1 = self.raw_dir / "raw_1.jsonl"
        shard1.unlink()
        with self.assertRaisesRegex(ValueError, "Incomplete production shard coverage"):
            merge(self.mapping_path, self.raw_dir, self.out)
        self.write_two_shards()
        wrong = self.raw[0]
        wrong_shard = 1 - int(wrong["original_english_sha256"][:12], 16) % 2
        with (self.raw_dir / f"raw_{wrong_shard}.jsonl").open("a") as stream:
            stream.write(json.dumps(wrong) + "\n")
        with self.assertRaisesRegex(ValueError, "wrong production shard"):
            merge(self.mapping_path, self.raw_dir, self.out)
        self.write_two_shards()
        self.protocol["shard"] = 1
        self.write_protocol()
        with self.assertRaisesRegex(ValueError, "range/filename mismatch"):
            merge(self.mapping_path, self.raw_dir, self.out)

    def test_builder_contract_keeps_all_full_text_and_quarantines_shared_sources(self):
        from scripts.hanguard.build_translation_repaired_dataset import build

        class CharacterTokenizer:
            def __call__(self, texts, *, add_special_tokens, truncation):
                assert add_special_tokens is False and truncation is False
                return {"input_ids": [list(text) for text in texts]}

        prepared = prepare_records(self.mapping, self.raw,
            overrides=[override(self.raw[0], "correct", corrected_translation="您好。")])
        originals = self.mapping.assign(group_id=self.mapping.base_id.map(lambda x: "group_" + x))
        originals = originals.drop(columns=["original_prompt", "english_text_sha256"])
        records = {split: group.reset_index(drop=True) for split, group in originals.groupby("split")}
        target = self.root / "repaired"
        audit = build(records, target, prepared["translations"], mapping=self.mapping,
                      tokenizer=CharacterTokenizer())
        self.assertTrue(audit["passed"])
        self.assertEqual(audit["archived_rows"], 6)
        self.assertEqual(audit["quarantined_rows"], 4)
        archive = pd.read_parquet(target / "full_repaired_archive.parquet").set_index("base_id")
        for row in prepared["translations"]:
            self.assertEqual(archive.loc[row["base_id"], "prompt"], row["translation"])
            self.assertEqual(archive.loc[row["base_id"], "split"], row["split"])
        self.assertTrue(archive.prompt_harm_label.eq("unharmful").all())
        corrected = archive[archive.translation_source.eq("recorded_bilingual_correction")]
        self.assertEqual(len(corrected), 2)
        self.assertTrue(corrected.model_translation.eq("你好。").all())
        self.assertTrue(corrected.prompt.eq("您好。").all())
        self.assertTrue(corrected.reviewer.str.contains("Codex agent").all())
        self.assertIn("不复用旧缓存", (target / "README.md").read_text())


if __name__ == "__main__":
    unittest.main()
