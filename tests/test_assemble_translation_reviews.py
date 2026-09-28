"""Boundary tests for recorded bilingual review assembly, with no model calls."""
import copy
import json
from pathlib import Path
import tempfile
import unittest

from scripts.hanguard.assemble_translation_reviews import FIELDS, assemble, sha


class AssembleTranslationReviewsTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.plan_path, self.review_path = self.root / "plan.jsonl", self.root / "reviews.jsonl"
        self.out = self.root / "assembled"
        self.plan, self.reviews = [], []
        for i in range(72):
            split = ("train", "validation", "test")[i // 24]
            source, old, model = f"Synthetic source {i}", f"旧译{i}", f"新译{i}"
            planned = dict(base_id=f"id_{i}", split=split, original_english=source, old_translation=old,
                           original_english_sha256=sha(source), qa_seed=20260928, qa_selection_reason="fixed_hash_random")
            self.plan.append(planned)
            review = dict(base_id=planned["base_id"], split=split, original_english=source,
                          original_english_sha256=sha(source), old_translation=old, old_translation_sha256=sha(old),
                          decision="approve", reviewer="Codex synthetic-test agent", reason="Compared source and complete target.",
                          basis="bilingual_text_review", label_blind=True)
            if i % 2:
                review.update(model_translation=model, model_translation_sha256=sha(model),
                              old_translation_assessment="minor_issue", original_model_translation_assessment="faithful",
                              reviewed_translation_assessment="faithful")
            else:
                review.update(translation=model, translation_sha256=sha(model), old_translation_status="minor_issue",
                              model_translation_status="faithful", reviewed_translation_status="faithful")
            self.reviews.append(review)

    def tearDown(self):
        self.temp.cleanup()

    def run_assemble(self):
        self.plan_path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in self.plan))
        self.review_path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in self.reviews))
        return assemble(self.plan_path, [self.review_path], self.out)

    def test_two_agent_schemas_and_extra_rule_probes_remain_separate(self):
        self.reviews[0].update(decision="correct", corrected_translation="修订正文0", corrected_translation_sha256=sha("修订正文0"))
        extra = copy.deepcopy(self.reviews[1])
        extra.update(original_english="Additional rule probe", original_english_sha256=sha("Additional rule probe"))
        self.reviews.append(extra)
        result = self.run_assemble()
        self.assertEqual(result["fixed_random_rows"], 72)
        self.assertEqual(result["all_reviewed_unique_sources"], 73)
        self.assertEqual(result["additional_nonrandom_reviewed_sources"], 1)
        self.assertEqual(result["fixed_random_decisions"], {"correct": 1, "approve": 71})
        rows = [json.loads(line) for line in (self.out / "review_overrides.jsonl").read_text().splitlines()]
        self.assertTrue(all(set(row) <= FIELDS for row in rows))
        self.assertEqual(rows[1]["translation_sha256"], self.reviews[1]["model_translation_sha256"])
        self.assertTrue(all(row["translation_sha256"] for row in rows))

    def test_conflicting_model_text_or_hash_aliases_cannot_be_silently_ignored(self):
        original = copy.deepcopy(self.reviews[0])
        for updates in ({"model_translation": "different text"}, {"model_translation_sha256": sha("different text")}):
            self.reviews[0] = dict(original, **updates)
            with self.subTest(updates=updates), self.assertRaises(ValueError):
                self.run_assemble()
            self.assertFalse(self.out.exists())

    def test_stale_model_or_corrected_hashes_are_rejected(self):
        original = copy.deepcopy(self.reviews[0])
        self.reviews[0]["translation_sha256"] = sha("stale")
        with self.assertRaisesRegex(ValueError, "Stale source/model"):
            self.run_assemble()
        self.reviews[0] = dict(original, decision="correct", corrected_translation="修订正文0", corrected_translation_sha256=sha("stale"))
        with self.assertRaisesRegex(ValueError, "Stale corrected"):
            self.run_assemble()

    def test_incomplete_or_duplicate_sample_and_duplicate_reviews_are_rejected(self):
        original_plan, original_reviews = copy.deepcopy(self.plan), copy.deepcopy(self.reviews)
        self.reviews = self.reviews[:-1]
        with self.assertRaisesRegex(ValueError, "incomplete"):
            self.run_assemble()
        self.reviews = original_reviews + [original_reviews[0]]
        with self.assertRaisesRegex(ValueError, "Repeated review"):
            self.run_assemble()
        self.reviews = original_reviews
        self.plan[0]["base_id"] = self.plan[1]["base_id"]
        with self.assertRaisesRegex(ValueError, "Repeated base_id"):
            self.run_assemble()
        self.plan = original_plan[:-1]
        with self.assertRaisesRegex(ValueError, "72-row"):
            self.run_assemble()

    def test_review_reason_and_old_text_identity_are_required(self):
        original = copy.deepcopy(self.reviews[0])
        for updates in ({"reason": ""}, {"reviewer": " "}, {"old_translation_sha256": sha("stale")},
                        {"base_id": "another"}, {"split": "test"}):
            self.reviews[0] = dict(original, **updates)
            with self.subTest(updates=updates), self.assertRaises(ValueError):
                self.run_assemble()

    def test_unicode_line_separators_and_vertical_tab_roundtrip_inside_json_strings(self):
        original_plan, original_review = copy.deepcopy(self.plan[0]), copy.deepcopy(self.reviews[0])
        for i, separator in enumerate(("\u2028", "\u2029", "\v")):
            self.out = self.root / f"assembled_separator_{i}"
            source, old = "Source first" + separator + "second", "旧译首段" + separator + "末段"
            model, corrected = "新译首段" + separator + "末段", "修订首段" + separator + "末段"
            self.plan[0] = dict(original_plan, original_english=source,
                                original_english_sha256=sha(source), old_translation=old)
            self.reviews[0] = dict(original_review, original_english=source, original_english_sha256=sha(source),
                old_translation=old, old_translation_sha256=sha(old), translation=model, translation_sha256=sha(model),
                decision="correct", corrected_translation=corrected, corrected_translation_sha256=sha(corrected),
                reason="完整对照" + separator + "含义修订")
            with self.subTest(separator=repr(separator)):
                result = self.run_assemble()
                self.assertEqual(result["fixed_random_rows"], 72)
                with (self.out / "qa_comparison.jsonl").open() as stream:
                    comparisons = [json.loads(line) for line in stream if line.strip()]
                row = next(r for r in comparisons if r["base_id"] == original_plan["base_id"])
                self.assertEqual(row["original_source"], source)
                self.assertEqual(row["old_translation"], old)
                self.assertEqual(row["model_translation"], model)
                self.assertEqual(row["reviewed_translation"], corrected)
                with (self.out / "review_overrides.jsonl").open() as stream:
                    overrides = [json.loads(line) for line in stream if line.strip()]
                self.assertEqual(overrides[0]["corrected_translation"], corrected)
                self.assertEqual(overrides[0]["translation_sha256"], sha(model))
                if separator in ("\u2028", "\u2029"):
                    self.assertIn(separator, self.review_path.read_text())


if __name__ == "__main__":
    unittest.main()
