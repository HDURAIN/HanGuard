"""Boundary checks for immutable, fully accounted translation repair outputs."""
import copy
import json
from pathlib import Path
import tempfile
import unittest

import pandas as pd

from scripts.hanguard.build_translation_repaired_dataset import (
    SPLITS, audit, build, digest, normalize, read_jsonl,
)


class CharacterTokenizer:
    def __call__(self, texts, *, add_special_tokens, truncation):
        assert add_special_tokens is False
        assert truncation is False
        return {"input_ids": [list(text) for text in texts]}


def fixtures():
    records = {}
    for i, split in enumerate(SPLITS):
        rows = []
        for j, source in enumerate(["wildguard_zh", "jailbench" if split == "test" else "chinese_curated"]):
            key = f"{split}_{source}"
            prompt = f"原文{key}"
            harmless = split == "validation" or (source != "wildguard_zh" and source != "jailbench")
            rows.append(dict(base_id=key, group_id=f"group_{key}", split=split, source=source,
                             source_row=i * 10 + j, prompt=prompt,
                             prompt_harm_label="unharmful" if harmless else "harmful",
                             category_id="0" if harmless else "1", base_normalized=normalize(prompt),
                             normalized_prompt=normalize(prompt), sample_id=digest(f"{key}:{prompt}"),
                             prompt_tokens=len(prompt), source_unit=f"source_{key}",
                             text_form="existing_wrapper" if source == "jailbench" else "source_text"))
        records[split] = pd.DataFrame(rows)
    all_rows = pd.concat(records.values(), ignore_index=True)
    mapping = all_rows[all_rows.source.eq("wildguard_zh")][["base_id", "source_row", "prompt"]].copy()
    mapping["original_prompt"] = mapping.base_id.map(lambda x: "English source " + x)
    return records, mapping


def translated(mapping):
    return [dict(base_id=r.base_id, original_english=r.original_prompt,
                 original_english_sha256=digest(r.original_prompt),
                 translation="完整新译文" + r.base_id, status="ok", reason="按完整英文重新翻译",
                 model="test-translator", finish_reason="stop", generation_tokens=20,
                 prompt_sha256="opaque-generator-request-hash") for r in mapping.itertuples()]


class TranslationRepairTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.out = Path(self.temp.name) / "candidate"
        self.records, self.mapping = fixtures()
        self.translations = translated(self.mapping)
        self.tokenizer = CharacterTokenizer()

    def tearDown(self):
        self.temp.cleanup()

    def run_build(self, **kwargs):
        return build(self.records, self.out, self.translations, mapping=self.mapping,
                     tokenizer=self.tokenizer, **kwargs)

    def archive(self):
        return pd.read_parquet(self.out / "full_repaired_archive.parquet").set_index("base_id")

    def test_full_uniform_build_preserves_inputs_split_lineage_labels_and_other_sources(self):
        before = {k: v.copy(deep=True) for k, v in self.records.items()}
        result = self.run_build()
        self.assertTrue(result["passed"])
        self.assertEqual(result["archived_rows"], 6)
        self.assertEqual(result["quarantined_rows"], 0)
        self.assertEqual(result["release_status"], "candidate_pending_root_qa")
        saved = self.archive()
        for split, frame in before.items():
            pd.testing.assert_frame_equal(frame, self.records[split])
            for row in frame.itertuples():
                new = saved.loc[row.base_id]
                self.assertEqual(new.old_prompt, row.prompt)
                self.assertEqual(new.group_id, row.group_id)
                self.assertEqual(new.split, split)
                self.assertEqual(new.prompt_harm_label, row.prompt_harm_label)
                self.assertEqual(new.category_id, row.category_id)
                self.assertEqual(new.base_normalized, row.base_normalized)
                self.assertEqual(new.sample_id, digest(f"{row.base_id}:{new.prompt}"))
                self.assertEqual(new.prompt_tokens, len(new.prompt))
                if row.source != "wildguard_zh":
                    self.assertEqual(new.prompt, row.prompt)
                    self.assertEqual(new.sample_id, row.sample_id)
        self.assertIn("prompt_sha256", saved.loc["train_wildguard_zh", "translation_metadata_json"])

    def test_overlength_keeps_full_translation_never_old_fallback(self):
        full = "完整中文" * 110
        self.translations[0]["translation"] = full
        result = self.run_build(max_tokens=370)
        self.assertEqual(result["quarantined_rows"], 1)
        new = self.archive().loc["train_wildguard_zh"]
        self.assertEqual(new.prompt, full)
        self.assertEqual(new.prompt_tokens, len(full))
        self.assertEqual(new.quarantine_reason, "overlength_full_text_retained")
        self.assertFalse(pd.read_parquet(self.out / "train.parquet").base_id.eq(new.name).any())

    def test_explicit_no_length_limit_still_recomputes_full_length(self):
        self.translations[0]["translation"] = "长" * 500
        self.assertEqual(self.run_build(max_tokens=None)["quarantined_rows"], 0)
        self.assertEqual(self.archive().loc["train_wildguard_zh", "prompt_tokens"], 500)

    def test_default_full_text_dataset_retains_long_text_and_records_length_distributions(self):
        full = "完整长译文" * 180
        self.translations[0]["translation"] = full
        result = self.run_build()
        self.assertEqual(result["quarantined_rows"], 0)
        archive = self.archive()
        self.assertEqual(archive.loc["train_wildguard_zh", "prompt"], full)
        self.assertTrue(archive.loc["train_wildguard_zh", "prompt_over_legacy_limit"])
        train = pd.read_parquet(self.out / "train.parquet").set_index("base_id")
        self.assertEqual(train.loc["train_wildguard_zh", "prompt"], full)
        manifest = json.loads((self.out / "manifest.json").read_text())
        self.assertIsNone(manifest["max_prompt_tokens"])
        self.assertEqual(manifest["length_policy"], "annotation_only_no_cap")
        self.assertEqual(manifest["length_statistics"]["candidate"]["over_legacy_limit"], 1)
        self.assertEqual(manifest["length_statistics"]["candidate"]["max_tokens"], len(full))
        self.assertEqual(manifest["length_statistics"]["by_source"]["wildguard_zh"]["candidate"]["over_legacy_limit"], 1)
        self.assertEqual(manifest["length_statistics"]["by_split"]["train"]["candidate"]["over_legacy_limit"], 1)
        self.assertEqual(sum(manifest["length_statistics"]["candidate"]["bins"].values()), 6)
        self.assertIn("旧370长度配置/断言不可直接", (self.out / "README.md").read_text())

    def test_legacy_boundary_is_annotation_but_quality_review_still_quarantines(self):
        self.translations[0]["translation"] = "甲" * 370
        self.translations[1]["translation"] = "乙" * 371
        self.translations[1]["status"] = "review"
        result = self.run_build()
        self.assertEqual(result["quarantined_rows"], 1)
        archive = self.archive()
        self.assertFalse(archive.loc["train_wildguard_zh", "prompt_over_legacy_limit"])
        self.assertTrue(archive.loc["validation_wildguard_zh", "prompt_over_legacy_limit"])
        self.assertEqual(archive.loc["validation_wildguard_zh", "quarantine_reason"], "translation_status_review")
        self.assertEqual(archive.loc["validation_wildguard_zh", "prompt"], "乙" * 371)

    def test_cross_split_normalized_chinese_conflict_quarantines_both_sides(self):
        self.translations[0]["translation"] = "重 复Ａ\u200b"
        self.translations[1]["translation"] = "重复a"
        result = self.run_build()
        self.assertEqual(result["quarantined_rows"], 2)
        q = pd.read_parquet(self.out / "quarantine.parquet")
        self.assertEqual(set(q.split), {"train", "validation"})
        self.assertTrue(q.quarantine_reason.str.contains("cross_split_normalized_duplicate").all())
        self.assertTrue(q.quarantine_reason.str.contains("conflicting_labels").all())
        self.assertEqual(set(q.prompt), {"重 复Ａ\u200b", "重复a"})

    def test_collision_with_untouched_source_is_also_explicitly_quarantined(self):
        original = self.records["train"].iloc[1].prompt
        self.translations[0]["translation"] = original
        self.assertEqual(self.run_build()["quarantined_rows"], 2)
        saved = self.archive()
        self.assertEqual(saved.loc["train_chinese_curated", "prompt"], original)
        self.assertIn("within_split_normalized_duplicate", saved.loc["train_chinese_curated", "quarantine_reason"])

    def test_english_cross_split_lineage_detected_even_if_new_chinese_differs(self):
        self.mapping.loc[self.mapping.index[0], "original_prompt"] = "The same ENGLISH"
        self.mapping.loc[self.mapping.index[1], "original_prompt"] = "the same english\u200b "
        self.translations = translated(self.mapping)
        result = self.run_build()
        self.assertEqual(result["quarantined_rows"], 2)
        self.assertEqual(result["exact_duplicate_groups_before_quarantine"], 0)
        self.assertTrue(self.archive().loc[["train_wildguard_zh", "validation_wildguard_zh"],
                        "quarantine_reason"].str.contains("cross_split_normalized_english_lineage").all())
        manifest = json.loads((self.out / "manifest.json").read_text())
        self.assertEqual(manifest["english_lineage_additional_rows"], 2)
        self.assertEqual(manifest["english_binary_label_conflict_groups"], 1)

    def test_english_category_conflict_without_binary_conflict_is_recorded(self):
        second = self.records["train"].iloc[0].copy()
        second["base_id"] = "train_second_wildguard"
        second["group_id"] = "group_second"
        second["source_row"] = 200
        second["category_id"] = "2"
        second["prompt"] = "另一条原文"
        self.records["train"] = pd.concat([self.records["train"], pd.DataFrame([second])], ignore_index=True)
        mapping_row = self.mapping.iloc[0].copy()
        mapping_row["base_id"] = second.base_id
        mapping_row["source_row"] = 200
        mapping_row["prompt"] = second.prompt
        self.mapping = pd.concat([self.mapping, pd.DataFrame([mapping_row])], ignore_index=True)
        self.translations = translated(self.mapping)
        self.assertEqual(self.run_build()["quarantined_rows"], 2)
        manifest = json.loads((self.out / "manifest.json").read_text())
        self.assertEqual(manifest["english_binary_label_conflict_groups"], 0)
        self.assertEqual(manifest["english_category_label_conflict_groups"], 1)

    def test_status_and_generation_truncation_cannot_enter_training(self):
        self.translations[0]["status"] = "review"
        self.translations[1]["status"] = "error"
        self.translations[1]["translation"] = ""
        self.translations[2]["finish_reason"] = "length"
        result = self.run_build()
        self.assertEqual(result["quarantined_rows"], 3)
        q = self.archive().loc["test_wildguard_zh"]
        self.assertIn("translation_generation_length_limit", q.quarantine_reason)
        self.assertNotEqual(q.prompt, q.old_prompt)

    def test_missing_stop_evidence_and_truncated_source_are_quarantined(self):
        del self.translations[0]["finish_reason"]
        self.translations[1]["finish_reason"] = "unknown"
        self.translations[2]["input_truncated"] = True
        self.assertEqual(self.run_build()["quarantined_rows"], 3)
        saved = self.archive()
        self.assertEqual(saved.loc["train_wildguard_zh", "quarantine_reason"], "unverified_generation_finish")
        self.assertEqual(saved.loc["validation_wildguard_zh", "quarantine_reason"], "unverified_generation_finish")
        self.assertEqual(saved.loc["test_wildguard_zh", "quarantine_reason"], "translation_source_input_truncated")

    def test_mapping_must_include_original_chinese_alignment(self):
        self.mapping = self.mapping.drop(columns="prompt")
        with self.assertRaisesRegex(ValueError, "old Chinese"):
            self.run_build()
        self.assertFalse(self.out.exists())

    def test_missing_fails_by_default_and_explicit_preview_quarantines_missing(self):
        self.translations.pop()
        with self.assertRaisesRegex(ValueError, "Missing 1 WildGuard"):
            self.run_build()
        self.assertFalse(self.out.exists())
        self.assertEqual(self.run_build(require_complete=False)["quarantined_rows"], 1)
        row = self.archive().loc["test_wildguard_zh"]
        self.assertEqual(row.translation_status, "missing")
        self.assertEqual(row.quarantine_reason, "missing_translation")

    def test_duplicate_and_unknown_translation_ids_are_rejected(self):
        clean = copy.deepcopy(self.translations)
        for mutate, message in [(lambda x: x.append(copy.deepcopy(x[0])), "Duplicate translation"),
                                (lambda x: x[0].update(base_id="train_chinese_curated"), "non-WildGuard")]:
            self.translations = copy.deepcopy(clean)
            mutate(self.translations)
            with self.assertRaisesRegex(ValueError, message):
                self.run_build()
            self.assertFalse(self.out.exists())

    def test_wrong_english_mapping_hash_source_row_and_text_are_rejected(self):
        clean = copy.deepcopy(self.translations)
        for field, value, message in [("original_english", "Different source", "English text disagrees"),
                                      ("original_english_sha256", "invalid", "SHA256 mismatch")]:
            self.translations = copy.deepcopy(clean)
            self.translations[0][field] = value
            with self.assertRaisesRegex(ValueError, message):
                self.run_build()
        self.translations = clean
        self.mapping.loc[self.mapping.index[0], "source_row"] = 999
        with self.assertRaisesRegex(ValueError, "source_row disagrees"):
            self.run_build()
        self.assertFalse(self.out.exists())

    def test_normalized_empty_output_is_quarantined(self):
        self.translations[0]["translation"] = " \n\u200b"
        self.assertEqual(self.run_build()["quarantined_rows"], 1)
        self.assertIn("empty_repaired_text", self.archive().loc["train_wildguard_zh", "quarantine_reason"])

    def test_refuses_existing_output_without_touching_it(self):
        self.out.mkdir()
        marker = self.out / "train.parquet"
        marker.write_bytes(b"old dataset must stay untouched")
        with self.assertRaisesRegex(ValueError, "Refusing to overwrite"):
            self.run_build()
        self.assertEqual(marker.read_bytes(), b"old dataset must stay untouched")

    def test_audit_detects_modified_artifact(self):
        self.run_build()
        p = self.out / "validation.parquet"
        frame = pd.read_parquet(p)
        frame.loc[0, "prompt"] = "tampered"
        frame.to_parquet(p, index=False)
        with self.assertRaisesRegex(ValueError, "Artifact hash mismatch"):
            audit(self.out, self.records, self.tokenizer)

    def test_jsonl_rejects_partial_or_nonobject_records(self):
        f = Path(self.temp.name) / "bad.jsonl"
        f.write_text('{"base_id":')
        with self.assertRaisesRegex(ValueError, "line 1"):
            read_jsonl(f)
        f.write_text('[]\n')
        with self.assertRaisesRegex(ValueError, "must be an object"):
            read_jsonl(f)


if __name__ == "__main__":
    unittest.main()
