import csv
import json
from pathlib import Path
import tempfile
import unittest

from scripts.hanguard import package_fusion_ablation_results as bundle


class FusionArchiveTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)

    def write_predictions(self, path, rows):
        path.parent.mkdir(parents=True, exist_ok=True)
        fields = ["base_id", "label", "source", "original_english", "normalized_prompt", "unseen_text"]
        for method in bundle.METHODS:
            fields.extend([f"prediction_{method}", f"probability_{method}", f"threshold_{method}"])
        with path.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            for identity, label, predicted in rows:
                row = {"base_id": identity, "label": label, "source": "wildguard_zh",
                       "original_english": "DO NOT EXPORT", "normalized_prompt": "完整原文", "unseen_text": "PRIVATE"}
                for method in bundle.METHODS:
                    row.update({f"prediction_{method}": predicted, f"probability_{method}": 0.9 if predicted else 0.1,
                                f"threshold_{method}": 0.5})
                writer.writerow(row)

    def test_text_is_stripped_and_subset_metrics_recomputed(self):
        source, target = self.root / "predictions.csv", self.root / "export.csv"
        self.write_predictions(source, [("keep1", 1, 1), ("excluded", 0, 1), ("keep2", 0, 0)])
        details = bundle.export_predictions(source, target, {"excluded"})
        self.assertEqual(details["rows"], 2)
        text = target.read_text()
        for private in ("DO NOT EXPORT", "完整原文", "PRIVATE", "excluded", "original_english"):
            self.assertNotIn(private, text)
        for value in bundle.prediction_metrics(target).values():
            self.assertEqual(value["n"], 2)
            self.assertEqual(value["accuracy"], 1.0)

    def test_duplicate_identity_is_rejected(self):
        source = self.root / "predictions.csv"
        self.write_predictions(source, [("same", 1, 1), ("same", 1, 1)])
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            bundle.export_predictions(source, self.root / "export.csv")

    def test_nested_ablation_scopes_and_pending_refresh(self):
        study = self.root / bundle.ABLATION
        target = study / "seed_43/analysis/results.json"
        metrics = {"best_single": {"n": 8350, "accuracy": 0.9},
                   "fusion_fixed_best_anchor": {"n": 8350, "accuracy": 0.91}}
        bundle.write_json(target, {"complete": True, "full_test": {"metrics": metrics},
                                   "selected_subset": {"metrics": metrics}})
        archive = self.root / bundle.DEFAULT_ARCHIVE
        bundle.package(self.root, archive)
        index = bundle.read_json(archive / "index.json")
        self.assertEqual(next(item for item in index["studies"] if item["study"] == "nolora_additional_ablation")["status"], "pending")
        with (archive / "results_summary.csv").open() as stream:
            rows = list(csv.DictReader(stream))
        self.assertEqual(len(rows), 4)
        self.assertEqual({row["seed"] for row in rows}, {"43"})
        self.assertEqual({row["evaluation_scope"] for row in rows}, {"full_test_8350", "selected_test_8228"})
        bundle.write_json(study / "results.json", {"complete": True, "metrics": metrics})
        bundle.package(self.root, archive)
        self.assertEqual(next(item for item in bundle.read_json(archive / "index.json")["studies"]
                              if item["study"] == "nolora_additional_ablation")["status"], "complete")
        with (archive / "results_summary.csv").open() as stream:
            self.assertEqual(len(list(csv.DictReader(stream))), 4)  # aggregate not duplicated

    def test_archive_verification_detects_tampering(self):
        archive = self.root / bundle.DEFAULT_ARCHIVE
        bundle.package(self.root, archive)
        self.assertTrue(bundle.verify_archive(archive)["verified"])
        (archive / "results_summary.csv").write_text("changed")
        with self.assertRaisesRegex(ValueError, "Missing or changed"):
            bundle.verify_archive(archive)

    def test_archive_cannot_write_outside_repository(self):
        with self.assertRaisesRegex(ValueError, "subdirectory"):
            bundle.package(self.root, self.root.parent / "outside")

    def test_source_closure_resolves_project_imports_without_execution(self):
        source = self.root / "scripts/hanguard/runner.py"
        dependency = self.root / "scripts/hanguard/helper.py"
        nested = self.root / "shared.py"
        source.parent.mkdir(parents=True)
        source.write_text("from scripts.hanguard import helper\nraise RuntimeError('must not execute')\n")
        dependency.write_text("from shared import VALUE\n")
        nested.write_text("VALUE = 1\n")
        # Protocol paths cannot broaden the archive into data or outside root.
        forbidden = self.root / "data/private.py"
        forbidden.parent.mkdir()
        forbidden.write_text("PRIVATE = 'text'\n")
        value = {"code_sha256": {str(source): bundle.sha256(source), str(forbidden): bundle.sha256(forbidden)},
                 "nested": {"bindings": {"/etc/os.py": "0" * 64}}}
        paths = bundle.bound_source_paths(self.root, value)
        self.assertEqual(set(paths), {source})
        self.assertEqual(set(bundle.source_dependency_closure(self.root, list(paths))), {source, dependency, nested})

    def test_presentation_uses_same_device_replay_and_sample_seed_std(self):
        tasks = {}
        for seed, best, anchor in [(42, .90, .92), (43, .91, .94), (44, .92, .93)]:
            metrics = {method: {"accuracy": best} for method in bundle.METHODS}
            metrics["fusion_fixed_best_anchor"] = {"accuracy": anchor}
            tasks["seed42_imported" if seed == 42 else f"seed{seed}"] = {
                "complete": True, "full_test": {"metrics": metrics}, "selected_subset": {"metrics": metrics},
            }
        components = {arm: {"accuracy": .95} for arm in bundle.COMPONENT_ARMS}
        components["uniform_weights"] = {"accuracy": .93}
        tasks["components"] = {
            "complete": True, "full_test": {"metrics": components}, "selected_subset": {"metrics": components},
            "parameters": {arm: {"total": 100, "trainable": 90} for arm in bundle.COMPONENT_ARMS},
            "specs": {arm: {"layers": [8, 16, 24, 32], "anchor": 19} for arm in bundle.COMPONENT_ARMS},
            "execution": {"training_device": "cpu"},
        }
        bundle.write_json(self.root / bundle.ABLATION / "results.json", {"complete": True, "tasks": tasks})
        tables = bundle.presentation_tables(self.root)
        uniform = next(row for row in tables["component_ablation.csv"] if row["method"] == "uniform_weights")
        self.assertEqual(uniform["status"], "completed")
        self.assertAlmostEqual(uniform["gain_vs_anchor_replay_pp"], -2.0)
        self.assertEqual(uniform["training_device"], "cpu")
        best = next(row for row in tables["seed_summary.csv"] if row["method"] == "best_single")
        self.assertEqual(best["status"], "completed")
        self.assertAlmostEqual(best["mean_accuracy_percent"], 91.0)
        self.assertAlmostEqual(best["sample_std_accuracy_pp"], 1.0)
        text = bundle.presentation_readme(tables)
        self.assertIn("91.0000 ± 1.0000", text)
        self.assertIn("-2.0000", text)
        self.assertIn("+2.0000 ± 1.0000", text)

    def test_presentation_does_not_treat_plans_or_partial_seeds_as_results(self):
        study = self.root / bundle.ABLATION
        bundle.write_json(study / "protocol.json", {"component_specs": {"anchor_replay": {"layers": [8, 16], "anchor": 19}}})
        metrics = {method: {"accuracy": .99} for method in bundle.METHODS}
        bundle.write_json(study / "seed_43/analysis/results.json", {"complete": False, "full_test": {"metrics": metrics}})
        bundle.write_json(study / "results.json", {"complete": False, "tasks": {"seed44": {
            "complete": True, "full_test": {"metrics": metrics}, "selected_subset": {"metrics": metrics}}}})
        tables = bundle.presentation_tables(self.root)
        row = next(row for row in tables["seed_summary.csv"] if row["method"] == "best_single")
        self.assertEqual(row["status"], "pending")
        self.assertNotIn("mean_accuracy_percent", row)
        self.assertEqual(row["completed_seeds"], "44")
        replay = tables["component_ablation.csv"][0]
        self.assertEqual(replay["layers"], "8/16")
        self.assertEqual(replay["status"], "pending")
        self.assertNotIn("accuracy_percent", replay)
        seed43 = next(row for row in tables["ablation_coverage.csv"] if row["ablation"] == "head_seed_43")
        self.assertEqual(seed43["status"], "pending")


if __name__ == "__main__":
    unittest.main()
