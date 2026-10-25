"""CPU-only checks for the separate diagnostic, never production payloads."""
import argparse
import copy
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from scripts.hanguard import four_arm_feature_mismatch_diagnostic as diag


class FakeBackbone(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = torch.nn.ModuleList([torch.nn.Identity(), torch.nn.Identity()])

    def forward(self, input_ids, attention_mask, position_ids, use_cache=False):
        hidden = torch.stack((input_ids, input_ids + 1), -1).float()
        for layer in self.layers:
            hidden = layer(hidden)
        return hidden


class DiagnosticTests(unittest.TestCase):
    def test_comparison_exact_and_numeric_metrics(self):
        a = np.array([[0, 3], [4, 0]], np.float16)
        self.assertTrue(diag.compare_values(a, a.copy())["equal"])
        record = diag.compare_values(a, np.array([[0, 0], [4, 0]], np.float16))
        self.assertFalse(record["equal"])
        self.assertEqual(record["different_count"], 1)
        self.assertEqual(record["max_abs"], 3)
        self.assertAlmostEqual(record["mean_abs"], .75)
        self.assertAlmostEqual(record["rmse"], 1.5)
        self.assertAlmostEqual(record["cosine"], .8)
        self.assertEqual(record["atol"], 0)
        self.assertEqual(record["rtol"], 0)

    def test_comparison_zero_vectors_and_reject_nonfinite(self):
        self.assertEqual(diag.compare_values(np.zeros((2,)), np.zeros((2,)))["cosine"], 1)
        self.assertEqual(diag.compare_values(np.zeros((2,)), np.ones((2,)))["cosine"], 0)
        for a, b in ((np.zeros(2), np.zeros(3)), (np.array([np.nan]), np.zeros(1))):
            with self.subTest(a=a, b=b), self.assertRaises(ValueError):
                diag.compare_values(a, b)

    def test_ragged_row_boundaries(self):
        a, b = np.ones((5, 2)), np.ones((5, 2))
        b[4, 1] = 0
        result = diag.comparison(a, b, [2, 3])
        self.assertEqual(len(result["rows"]), 2)
        self.assertTrue(result["rows"][0]["equal"])
        self.assertFalse(result["rows"][1]["equal"])
        with self.assertRaises(ValueError):
            diag.comparison(a, b, [2, 2])

    def test_group_targets_include_boundary_and_calibration(self):
        self.assertEqual(diag.target_groups(1179), [0, 589, 860, 861, 862, 863, 864, 865, 1178])
        with self.assertRaises(ValueError):
            diag.target_groups(800)

    def test_missing_busy_low_memory_and_unknown_telemetry_block(self):
        for inventory in ([], [dict(index=2, pids=[123], free_gib=40)],
                          [dict(index=2, pids=[], free_gib=19)], [dict(index=2, pids=[])]):
            with self.subTest(inventory=inventory), patch.object(diag, "gpu_inventory", return_value=inventory):
                self.assertIsNone(diag.empty_gpu(2, 20))
        with patch.object(diag, "gpu_inventory", side_effect=RuntimeError("telemetry")):
            self.assertIsNone(diag.empty_gpu(2, 20))
        free = dict(index=2, pids=[], free_gib=24)
        with patch.object(diag, "gpu_inventory", return_value=[free]):
            self.assertEqual(diag.empty_gpu(2, 20), free)

    def _ledger_fixture(self, folder):
        root = Path(folder)
        args = argparse.Namespace(study=root / "study", output=root / "separate",
            gpu=2, run_id="r01", admission_ledger=root / "ledger.json")
        args.study.mkdir()
        (args.study / "protocol.json").write_text("{}")
        protocol = dict(parent_study=str(root / "parent"), model_path=str(root / "model"),
                        data_dir=str(root / "data"))
        exact = diag.process_identity(os.getpid())
        ledger = dict(run_id=args.run_id, study=str(args.study), output=str(args.output), gpu=2,
            identity=exact, protocol_sha256=diag.base.sha(args.study / "protocol.json"),
            diagnostic_script_sha256=diag.base.sha(diag.__file__))
        args.admission_ledger.write_text(json.dumps(ledger))
        return args, protocol, ledger

    def test_exact_child_ledger_and_command_source_binding(self):
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, CUDA_VISIBLE_DEVICES="2"):
            args, protocol, ledger = self._ledger_fixture(tmp)
            self.assertEqual(diag.verify_ledger(args, protocol), ledger["identity"])
            for field in diag.IDENTITY_FIELDS:
                changed = copy.deepcopy(ledger)
                changed["identity"][field] = "wrong"
                args.admission_ledger.write_text(json.dumps(changed))
                with self.subTest(field=field), self.assertRaises(ValueError):
                    diag.verify_ledger(args, protocol)
            for field in ("run_id", "study", "output", "gpu", "protocol_sha256", "diagnostic_script_sha256"):
                changed = copy.deepcopy(ledger)
                changed[field] = "wrong"
                args.admission_ledger.write_text(json.dumps(changed))
                with self.subTest(field=field), self.assertRaises(ValueError):
                    diag.verify_ledger(args, protocol)

    def test_output_cannot_write_original_study(self):
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, CUDA_VISIBLE_DEVICES="2"):
            args, protocol, ledger = self._ledger_fixture(tmp)
            args.output = args.study / "diagnostic"
            ledger["output"] = str(args.output)
            args.admission_ledger.write_text(json.dumps(ledger))
            with self.assertRaisesRegex(ValueError, "separate"):
                diag.verify_ledger(args, protocol)

    def test_scoped_hashing_never_reads_test_payload(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            model, data, parent = root / "model", root / "data", root / "parent"
            filenames = [model / "weight.safetensors", data / "train.parquet", data / "validation.parquet",
                         data / "test.parquet", parent / "caches/train.npy", parent / "caches/validation.npy",
                         parent / "caches/test.npy", parent / "parallel_features/test/lengths.npy",
                         parent / "protocol.json"]
            protocol = dict(model_path=str(model), data_dir=str(data), parent_study=str(parent),
                            inputs={str(p): "digest" for p in filenames})
            visited = []
            def digest(path):
                visited.append(str(path))
                return "digest"
            with patch.object(diag.base, "sha", side_effect=digest):
                checked = diag.verify_train_inputs(protocol)
            self.assertEqual(set(checked), set(visited))
            self.assertEqual(len(visited), 6)
            self.assertFalse(any("test" in Path(p).name or "/test/" in p for p in visited))

    def test_blocked_run_never_initializes_cuda(self):
        args = argparse.Namespace(study=Path("/study"), output=Path("/output"), gpu=2,
                                  run_id="r01", admission_ledger=Path("/ledger"))
        protocol = dict(execution=dict(minimum_free_gib=20))
        with patch.object(diag.study_module, "verify", return_value=protocol) as verify, \
                patch.object(diag, "verify_ledger", return_value={}), \
                patch.object(diag.original, "read", return_value={}), \
                patch.object(diag.base, "sha", return_value="digest"), \
                patch.object(diag, "empty_gpu", return_value=None), \
                patch.object(torch.cuda, "init") as cuda:
            self.assertEqual(diag.run(args), 75)
            cuda.assert_not_called()
            verify.assert_called_once_with(args.study, full_hash=False)

    def test_cpu_hook_replay_all_exact_and_changed_parent_fails(self):
        backbone = FakeBackbone()
        batch = dict(input_ids=torch.tensor([[1, 2, 0], [3, 4, 5]]),
                     attention_mask=torch.tensor([[1, 1, 0], [1, 1, 1]]))
        last = np.array([[2, 3], [5, 6]], np.float16)
        expected = np.stack((last, last), axis=1)
        report = diag.replay_group(backbone, batch, np.array([2, 3]), expected)
        self.assertTrue(report["passed"])
        self.assertEqual(len(report["comparisons"]), 12)
        self.assertEqual(len(backbone.layers[-1]._forward_hooks), 0)
        changed = expected.copy()
        changed[0, -1, 0] += 1
        report = diag.replay_group(backbone, batch, np.array([2, 3]), changed)
        self.assertFalse(report["passed"])
        self.assertTrue(report["comparisons"]["simultaneous_raw_last_vs_old_last"]["overall"]["equal"])
        self.assertFalse(report["comparisons"]["raw_last_vs_parent"]["overall"]["equal"])
        self.assertEqual(len(backbone.layers[-1]._forward_hooks), 0)

    def test_parent_raw_chunk_compares_correct_global_row_order_and_rejects_tamper(self):
        with tempfile.TemporaryDirectory() as tmp:
            parent = Path(tmp)
            caches = parent / "caches"
            caches.mkdir()
            folder = parent / "parallel_features/train/chunks/000000"
            attempt = folder / "attempt_001"
            attempt.mkdir(parents=True)
            merged = np.arange(12, dtype=np.float16).reshape(2, 2, 3)
            feature_path, rows_path = attempt / "features.npy", attempt / "rows.npy"
            np.save(feature_path, merged[[1, 0]])
            np.save(rows_path, np.array([1, 0], np.int64))
            receipt_path = folder / "receipt.json"
            receipt = dict(attempt=1, path=str(feature_path), sha256=diag.base.sha(feature_path),
                row_indices_path=str(rows_path), row_indices_sha256=diag.base.sha(rows_path))
            receipt_path.write_text(json.dumps(receipt))
            audit_path = caches / "train.replay_audit.json"
            audit_path.write_text(json.dumps(dict(complete_chunk_receipts=[dict(chunk_id=0,
                receipt_path=str(receipt_path), receipt_sha256=diag.base.sha(receipt_path))])))
            (caches / "train.manifest.json").write_text(json.dumps(dict(replay_audit_sha256=diag.base.sha(audit_path))))
            plan = dict(chunks=[dict(chunk_id=0, group_start=0, group_stop=2)],
                        groups=[[1], [0]], shape=[2, 2, 3])
            with patch.object(diag.original, "_quick_receipt", return_value=receipt):
                result = diag.parent_chunks(parent, {}, plan, merged, [0, 1])
                self.assertTrue(all(r["comparison"]["overall"]["equal"] for r in result.values()))
                changed = merged.copy()
                changed[1, 0, 0] += 1
                result = diag.parent_chunks(parent, {}, plan, changed, [0, 1])
                self.assertFalse(result[0]["comparison"]["overall"]["equal"])
                self.assertTrue(result[1]["comparison"]["overall"]["equal"])
                with feature_path.open("ab") as stream:
                    stream.write(b"tamper")
                with self.assertRaisesRegex(ValueError, "artifact changed"):
                    diag.parent_chunks(parent, {}, plan, merged, [0, 1])


if __name__ == "__main__":
    unittest.main()
