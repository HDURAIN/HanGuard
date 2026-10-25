"""CPU checks of the factorial trainer's data/selection/resume contracts."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np
import pandas as pd
import torch

from scripts.hanguard import four_arm_experiment_training as training


class SyntheticTokens:
    def __init__(self, values):
        self.values = values

    def get_row(self, row):
        return self.values[row]


def synthetic_split(split, count=12, hidden=6):
    rng = np.random.default_rng({"train": 1, "validation": 2, "test": 3}[split])
    lengths = np.asarray([1 + i % 4 for i in range(count)], dtype=np.int64)
    binary = np.asarray([int(i % 3 != 0) for i in range(count)])
    categories = np.where(binary, np.arange(count) % 5 + 1, 0)
    sources = np.asarray(["wildguard_zh" if i % 4 == 0 else "chinese_curated" if i % 2 else "jailbench" for i in range(count)])
    frame = pd.DataFrame(dict(base_id=[f"{split}_{i}" for i in range(count)],
                              group_id=[f"{split}_group{i // 2}" for i in range(count)],
                              source=sources, category_id=categories,
                              prompt_harm_label=np.where(binary, "harmful", "unharmful"),
                              type_label_mask=binary.astype(bool)))
    meta = frame[["base_id", "group_id", "source"]].copy()
    meta["label"], meta["prompt_tokens"] = binary, lengths
    return dict(last=rng.normal(size=(count, 32, hidden)).astype(np.float16),
                tokens=SyntheticTokens([rng.normal(size=(int(n), hidden)).astype(np.float16) for n in lengths]),
                meta=meta, frame=frame, lengths=lengths,
                labels=training.label_arrays(frame), hidden_size=hidden,
                identity=dict(dataset_sha256=f"synthetic_{split}", last_manifest_sha256=f"last_{split}", rows=count))


class FourArmTrainingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.study = Path(self.temp.name)
        self.data = {split: synthetic_split(split) for split in ("train", "validation", "test")}
        excluded = self.study / "excluded.txt"
        excluded.write_text("test_0\ntest_1\n")
        self.protocol = dict(tasks=["binary", "category"], seeds=[42],
                             data_dir=str(self.study / "data"),
                             data_sha256={s: f"synthetic_{s}" for s in self.data},
                             last_token_cache_dir=str(self.study / "cache"),
                             excluded_test_ids_path=str(excluded), excluded_test_ids_sha256=training.sha(excluded),
                             training=dict(width=4, epochs=2, max_epochs=2, dropout=0., lr=.001,
                                           weight_decay=.001, token_budget=20, max_micro=4,
                                           effective_batch=5, grad_clip=5., bootstrap_draws=12))
        training.dump(self.study / "protocol.json", self.protocol)

    def loader(self, study, split):
        return self.data[split]

    def lock(self):
        selections, artifacts = {}, {}
        for task in training.TASKS:
            folder = training.task_folder(self.study, task, 42)
            selections[f"{task}/s42"] = training.sha(folder / "selection.json")
            for filename in ("selection.json", "selected_heads.pt", "validation_predictions.parquet"):
                path = folder / filename
                artifacts[str(path)] = training.sha(path)
        training.dump(self.study / "selection_lock.json", dict(
            complete=True, protocol_sha256=training.sha(self.study / "protocol.json"),
            selections=selections, artifacts=artifacts, test_opened=False))

    def test_native_type_mask_preserves_binary_and_excludes_wildguard(self):
        data = self.data["train"]
        arrays = data["labels"]
        self.assertEqual(len(arrays["binary"]), 12)
        self.assertFalse(arrays["typed"][data["frame"].source.eq("wildguard_zh")].any())
        self.assertTrue((arrays["categories"][arrays["typed"]] > 0).all())
        frame = data["frame"].copy()
        index = frame.index[(frame.source != "wildguard_zh") & frame.prompt_harm_label.eq("harmful")][0]
        frame.loc[index, "type_label_mask"] = False
        changed = training.label_arrays(frame)
        self.assertFalse(changed["typed"][index])
        self.assertFalse(changed["known_native"][index])
        np.testing.assert_array_equal(changed["binary"], arrays["binary"])

    def test_microbatches_full_input_budget_and_order(self):
        rows = np.asarray([7, 2, 6, 0, 3, 1])
        lengths = self.data["train"]["lengths"]
        batches = list(training.micro_batches(rows, lengths, token_budget=8, max_micro=3, pad_multiple=2))
        np.testing.assert_array_equal(np.concatenate(batches), rows)
        for indices in batches:
            maximum = int(lengths[indices].max())
            self.assertLessEqual(len(indices) * ((maximum + 1) // 2 * 2), 8)
        with self.assertRaisesRegex(ValueError, "full input"):
            list(training.micro_batches([3], lengths, 2, 3))

    def test_sample_weighted_microaccumulation_matches_whole_macro(self):
        data = self.data["train"]
        cfg = training.settings(self.protocol)
        cfg["token_budget"], cfg["max_micro"] = 100, 100
        heads_a = training.build_heads(6, "binary", 42, cfg, "cpu")
        heads_b = training.build_heads(6, "binary", 42, cfg, "cpu")
        opt_a = {a: torch.optim.SGD(h.parameters(), lr=.01) for a, h in heads_a.items()}
        opt_b = {a: torch.optim.SGD(h.parameters(), lr=.01) for a, h in heads_b.items()}
        rows = np.arange(5)
        losses_a, _ = training.train_macro(heads_a, opt_a, data, rows, "binary", cfg, "cpu", seed=42, epoch=0, step=0)
        cfg["max_micro"] = 2
        losses_b, _ = training.train_macro(heads_b, opt_b, data, rows, "binary", cfg, "cpu", seed=42, epoch=0, step=0)
        for arm in training.ARMS:
            self.assertAlmostEqual(losses_a[arm], losses_b[arm], places=6)
            for key, value in heads_a[arm].state_dict().items():
                torch.testing.assert_close(value, heads_b[arm].state_dict()[key], atol=1e-6, rtol=1e-5)

    def test_extension_considers_only_four_task_heads(self):
        history = [dict(epoch=i + 1, validation_ce={a: .5 - i * .01 for a in training.ARMS}) for i in range(12)]
        choice = training.extension_rule(history, {a: 12 for a in training.ARMS}, 12, 40)
        self.assertEqual(choice["new_budget"], 24)
        self.assertTrue(choice["all_four_arms_extended"])
        self.assertIsNone(training.extension_rule(history, {a: 1 for a in training.ARMS}, 12, 40))
        self.assertIsNone(training.extension_rule(history, {a: 12 for a in training.ARMS}, 40, 40))

    def test_end_to_end_two_tasks_lock_evaluate_and_summarize(self):
        with mock.patch.object(training, "_load_split", side_effect=self.loader) as loader:
            for task in training.TASKS:
                selection = training.fit_task(self.study, task, 42)
                self.assertEqual(set(selection["arms"]), set(training.ARMS))
                self.assertEqual(selection["training_rows"], len(training._task_indices(self.data["train"], task)))
            self.assertNotIn("test", [c.args[1] for c in loader.call_args_list])
            self.lock()
            training.verify_selection_lock(self.study)
            for task in training.TASKS:
                result = training.evaluate_task(self.study, task, 42)
                self.assertEqual(set(result["arms"]), set(training.ARMS))
                predictions = pd.read_parquet(result["predictions_path"])
                self.assertEqual(len(predictions), 12)
                self.assertFalse(set(predictions) & {"prompt", "original_english", "original_prompt"})
            result = training.summarize(self.study)
            self.assertTrue(result["complete"])
            self.assertIn("classic_gate_type_factorial", result["per_seed"]["42"]["systems"])
            self.assertIn("fusion_in_both_stages_factorial", result["per_seed"]["42"]["systems"])
            self.assertTrue((self.study / "analysis" / "metrics_by_seed.csv").is_file())

    def test_missing_global_lock_blocks_test_before_cache_load(self):
        with mock.patch.object(training, "_load_split") as loader:
            with self.assertRaises(FileNotFoundError):
                training.evaluate_task(self.study, "binary", 42)
            loader.assert_not_called()

    def test_partial_task_lock_is_rejected(self):
        training.dump(self.study / "selection_lock.json", dict(
            complete=True, protocol_sha256=training.sha(self.study / "protocol.json"),
            selections={"binary/s42": "dummy"}, artifacts={}))
        with self.assertRaisesRegex(ValueError, "Every task"):
            training.verify_selection_lock(self.study)

    def test_exact_epoch_resume_matches_uninterrupted_fit(self):
        real_train_macro = training.train_macro
        interrupted = False

        def stop_after_epoch(*args, **kwargs):
            nonlocal interrupted
            if kwargs["epoch"] == 1 and not interrupted:
                interrupted = True
                raise RuntimeError("synthetic interruption")
            return real_train_macro(*args, **kwargs)

        with mock.patch.object(training, "_load_split", side_effect=self.loader):
            with mock.patch.object(training, "train_macro", side_effect=stop_after_epoch):
                with self.assertRaisesRegex(RuntimeError, "synthetic interruption"):
                    training.fit_task(self.study, "binary", 42)
            saved = torch.load(training.task_folder(self.study, "binary", 42) / "epoch_resume.pt", weights_only=True)
            self.assertEqual(saved["completed_epochs"], 1)
            resumed = training.fit_task(self.study, "binary", 42)
            result_state = torch.load(resumed["checkpoint_path"], weights_only=True)["states"]
            another = self.study / "another"
            another.mkdir()
            training.dump(another / "protocol.json", self.protocol)
            full = training.fit_task(another, "binary", 42)
            full_state = torch.load(full["checkpoint_path"], weights_only=True)["states"]
        for arm in training.ARMS:
            self.assertEqual(resumed["arms"][arm]["best_epoch"], full["arms"][arm]["best_epoch"])
            for key, value in result_state[arm].items():
                torch.testing.assert_close(value, full_state[arm][key], atol=0, rtol=0)

    def test_selection_idempotence_and_tamper_detection(self):
        with mock.patch.object(training, "_load_split", side_effect=self.loader):
            first = training.fit_task(self.study, "binary", 42)
        with mock.patch.object(training, "_load_split") as loader:
            second = training.fit_task(self.study, "binary", 42)
            loader.assert_not_called()
        self.assertEqual(first, second)
        Path(first["checkpoint_path"]).write_bytes(b"altered")
        with self.assertRaisesRegex(ValueError, "Selected artifact"):
            training.fit_task(self.study, "binary", 42)

    def test_paired_counts_and_multiclass_bootstrap(self):
        labels, baseline, candidate = np.asarray([0, 1, 2, 3]), np.asarray([0, 0, 2, 0]), np.asarray([1, 1, 2, 3])
        changes = training.paired_changes(labels, baseline, candidate)
        self.assertEqual((changes["rescued"], changes["harmed"], changes["net_correct"]), (2, 1, 1))
        result = training.multiclass_bootstrap(labels, candidate, baseline, [0, 0, 1, 1], draws=20)
        self.assertEqual(result["accuracy"]["delta"], .25)
        metrics = training.multiclass_metrics(labels, candidate, range(4))
        self.assertEqual(metrics["accuracy"], .75)

    def test_fixed_subset_uses_registered_original_ids(self):
        mask = training._subset_mask(self.data["test"]["meta"], self.protocol)
        self.assertEqual(int(mask.sum()), 10)
        Path(self.protocol["excluded_test_ids_path"]).write_text("test_2\n")
        with self.assertRaisesRegex(ValueError, "changed"):
            training._subset_mask(self.data["test"]["meta"], self.protocol)


if __name__ == "__main__":
    unittest.main()
