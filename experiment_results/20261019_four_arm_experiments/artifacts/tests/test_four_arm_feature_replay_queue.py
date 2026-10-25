"""CPU-only recovery tests: occupied cards and unproven replays never resume."""
from pathlib import Path
import os
import tempfile
import unittest
from unittest.mock import Mock, patch

import torch

from scripts.hanguard import four_arm_feature_replay_queue as queue
from scripts.hanguard import four_arm_experiment_study as study


class RecoveryQueueTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.study = self.root / "study"
        self.output = self.root / "recovery"
        self.study.mkdir()
        self.output.mkdir()
        self.audit = self.root / "audit.json"
        queue.write(self.study / "protocol.json", {"immutable": True})
        queue.write(self.study / "state.json", {"state": "requires_review"})
        self.protocol = dict(schema=queue.SCHEMA, study=str(self.study), output=str(self.output),
            original_study_protocol_sha256=queue.sha(self.study / "protocol.json"),
            audit_path=str(self.audit), minimum_free_gib=20., poll_seconds=0.)
        self.audit_value = dict(complete=True, passed=True, checked_rows=55296,
            expected_completed_rows=55296, different_rows=0, different_elements=0,
            original_study_protocol_sha256=self.protocol["original_study_protocol_sha256"])
        queue.write(self.audit, self.audit_value)
        self.protocol["audit_sha256"] = queue.sha(self.audit)
        queue.write(self.output / "protocol.json", self.protocol)
        self.exact = dict(pid=12345, uid=os.getuid(), start_time_ticks=6789,
            command_sha256="exact-argv-hash", argv=["python", "diagnostic.py"], cwd=str(queue.ROOT))
        self.record = dict(run_id="exact_run", gpu=3, identity=self.exact,
            output=str(self.output / "runs" / "exact_run"),
            protocol_sha256=self.protocol["original_study_protocol_sha256"],
            diagnostic_script_sha256="diagnostic-hash", recovery_protocol_sha256="recovery-hash")
        self.admitted = dict(admitted=True, run_id="exact_run", physical_gpu=3, gpu=3,
            schema="hanguard_four_arm_feature_mismatch_diagnostic_v1", study=str(self.study),
            output=self.record["output"],
            identity=self.exact, protocol_sha256=self.record["protocol_sha256"],
            diagnostic_script_sha256="diagnostic-hash", recovery_protocol_sha256="recovery-hash")
        self.result = dict(self.admitted, complete=True, passed=True, study=str(self.study),
            split="train", diagnostic_only=True, production_cache_written=False,
            test_payloads_read=False, zero_tolerance_unchanged=True)
        self.launches = []
        cuda_patch = patch.object(torch.cuda, "init")
        self.cuda = cuda_patch.start()
        self.addCleanup(cuda_patch.stop)

    def tearDown(self):
        self.cuda.assert_not_called()

    def _run(self, outcomes, cards=None):
        outcomes = iter(outcomes)

        def launch(command, log, gpu=None):
            self.launches.append((command, log, gpu))
            if "--admission-ledger" in command:
                outcome = next(outcomes)
                folder = Path(command[command.index("--output") + 1])
                folder.mkdir(parents=True, exist_ok=True)
                run_id = command[command.index("--run-id") + 1]
                common = dict(run_id=run_id, physical_gpu=gpu, gpu=gpu, identity=self.exact,
                    schema="hanguard_four_arm_feature_mismatch_diagnostic_v1", study=str(self.study),
                    output=str(folder),
                    protocol_sha256=self.protocol["original_study_protocol_sha256"],
                    diagnostic_script_sha256=queue.sha(queue.DIAGNOSTIC),
                    recovery_protocol_sha256=queue.sha(self.output / "protocol.json"))
                if outcome.get("admitted", True):
                    queue.write(folder / "admitted.json", dict(common, admitted=True))
                if outcome.get("result", True):
                    result = dict(common, complete=True, passed=True, study=str(self.study),
                        split="train", diagnostic_only=True, production_cache_written=False,
                        test_payloads_read=False, zero_tolerance_unchanged=True)
                    result.update(outcome.get("changes", {}))
                    queue.write(folder / "result.json", result)
                code = outcome.get("code", 0)
            else:
                queue.write(self.study / "state.json", {"state": "completed"})
                code = 0
            child = Mock(returncode=code)
            child.poll.return_value = code
            return child, self.exact

        inventory = cards or [[dict(index=3, pids=[], free_gib=47)]]
        with patch.object(queue, "verify", return_value=self.protocol), \
                patch.object(queue, "inventory", side_effect=inventory), \
                patch.object(queue, "launch_owned", side_effect=launch), \
                patch.object(queue.time, "sleep"), patch.object(study, "verify_retired"):
            queue.run(self.output)
        return queue.read(self.output / "state.json")

    def test_every_completed_row_must_match_and_audit_identity_must_match(self):
        self.assertTrue(queue.audit_allows_resume(self.protocol))
        changes = [dict(complete=False), dict(passed=False), dict(checked_rows=55295),
                   dict(different_rows=1), dict(different_elements=1),
                   dict(original_study_protocol_sha256="other")]
        for change in changes:
            with self.subTest(change=change):
                queue.write(self.audit, dict(self.audit_value, **change))
                self.assertFalse(queue.audit_allows_resume(self.protocol))

    def test_diagnostic_requires_exact_child_run_gpu_sources_and_flags(self):
        self.assertTrue(queue.diagnostic_allows_resume(
            self.protocol, self.result, self.admitted, self.record))
        changes = [dict(complete=False), dict(passed=False), dict(study="other"),
            dict(split="test"), dict(run_id="previous_run"), dict(physical_gpu=4),
            dict(protocol_sha256="other"), dict(diagnostic_script_sha256="other"),
            dict(recovery_protocol_sha256="other"), dict(diagnostic_only=False),
            dict(production_cache_written=True), dict(test_payloads_read=True),
            dict(zero_tolerance_unchanged=False), dict(identity=dict(self.exact, start_time_ticks=6790)),
            dict(schema="other"), dict(output="other"), dict(gpu=4)]
        for change in changes:
            with self.subTest(change=change):
                self.assertFalse(queue.diagnostic_allows_resume(
                    self.protocol, dict(self.result, **change), self.admitted, self.record))
        for change in [dict(admitted=False), dict(run_id="other"),
                       dict(identity=dict(self.exact, uid=self.exact["uid"] + 1)),
                       dict(physical_gpu=4), dict(diagnostic_script_sha256="other"),
                       dict(schema="other"), dict(study="other"), dict(output="other"), dict(gpu=4)]:
            with self.subTest(admission=change):
                self.assertFalse(queue.diagnostic_allows_resume(
                    self.protocol, self.result, dict(self.admitted, **change), self.record))

    def test_empty_card_policy_excludes_any_pid_and_low_memory(self):
        cards = [dict(index=0, pids=[os.getpid()], free_gib=47),
            dict(index=1, pids=[999999999], free_gib=47),
            dict(index=2, pids=[], free_gib=19.99), dict(index=3, pids=[], free_gib=20)]
        self.assertEqual(queue.empty_cards(cards, list(range(8)), 20), [3])

    def test_occupied_cards_wait_then_launch_only_empty_gpu(self):
        cards = [[dict(index=3, pids=[555], free_gib=47)],
                 [dict(index=3, pids=[], free_gib=47)]]
        state = self._run([dict(code=1)], cards)
        self.assertEqual(len(self.launches), 1)
        self.assertEqual(self.launches[0][2], 3)
        self.assertEqual(state["state"], "diagnostic_requires_review")
        self.assertFalse(state["automatic_run"])

    def test_inventory_failure_does_not_admit_a_gpu(self):
        cards = [RuntimeError("inventory unavailable"), [dict(index=3, pids=[], free_gib=47)]]
        self._run([dict(code=1)], cards)
        self.assertEqual(len(self.launches), 1)

    def test_lost_admission_exit_75_retries_without_production_resume(self):
        self.audit_value["passed"] = False
        queue.write(self.audit, self.audit_value)
        cards = [[dict(index=3, pids=[], free_gib=47)]] * 2
        state = self._run([dict(code=75, admitted=False, result=False), {}], cards)
        self.assertEqual(len(self.launches), 2)
        self.assertEqual(state["state"], "completed_cache_requires_review")
        first = self.launches[0][1].parent
        self.assertEqual(queue.read(first / "exit.json")["exit_code"], 75)

    def test_exit_75_after_admission_requires_review(self):
        state = self._run([dict(code=75)])
        self.assertEqual(len(self.launches), 1)
        self.assertEqual(state["state"], "diagnostic_requires_review")

    def test_missing_result_never_resumes(self):
        with self.assertRaisesRegex(RuntimeError, "identity-bound"):
            self._run([dict(result=False)])
        self.assertEqual(len(self.launches), 1)
        self.assertEqual(queue.read(self.output / "state.json")["state"], "requires_review")

    def test_incomplete_or_other_run_result_never_resumes(self):
        with self.assertRaisesRegex(RuntimeError, "identity-bound"):
            self._run([dict(changes=dict(complete=False, run_id="other_run"))])
        self.assertEqual(len(self.launches), 1)

    def test_failed_all_row_audit_blocks_a_successful_diagnostic(self):
        queue.write(self.audit, dict(self.audit_value, different_rows=1))
        state = self._run([{}])
        self.assertEqual(state["state"], "completed_cache_requires_review")
        self.assertEqual(len(self.launches), 1)

    def test_only_both_exact_gates_resume_original_command_with_identity_ledger(self):
        state = self._run([{}])
        self.assertEqual(state["state"], "completed")
        self.assertEqual(len(self.launches), 2)
        command, _, gpu = self.launches[1]
        self.assertEqual(command, [queue.sys.executable, str(queue.STUDY_SCRIPT),
                                   "run", "--output", str(self.study)])
        self.assertIsNone(gpu)
        launch = queue.read(self.output / "production_resume_launch.json")
        self.assertEqual(launch["identity"], self.exact)
        self.assertEqual(launch["audit_sha256"], self.protocol.get("audit_sha256"))

    def test_prior_worker_pid_reuse_is_distinguished_by_exact_identity(self):
        queue.write(self.output / "runs" / "previous" / "admission.json", {"identity": self.exact})
        with patch.object(queue, "identity", return_value=self.exact):
            with self.assertRaisesRegex(RuntimeError, "still alive"):
                queue._prior_child_is_live(self.output)
        reused = dict(self.exact, start_time_ticks=self.exact["start_time_ticks"] + 1)
        with patch.object(queue, "identity", return_value=reused):
            queue._prior_child_is_live(self.output)

    def test_existing_production_child_blocks_duplicate_recovery_coordinator(self):
        queue.write(self.output / "production_resume_launch.json", {"identity": self.exact})
        with patch.object(queue, "identity", return_value=self.exact), \
                patch.object(queue, "verify", return_value=self.protocol), \
                patch.object(queue, "inventory") as inv, patch.object(queue, "launch_owned") as launch:
            with self.assertRaisesRegex(RuntimeError, "recovery child is still alive"):
                queue.run(self.output)
        inv.assert_not_called()
        launch.assert_not_called()
        reused = dict(self.exact, start_time_ticks=self.exact["start_time_ticks"] + 1)
        with patch.object(queue, "identity", return_value=reused):
            queue._prior_child_is_live(self.output)
        with patch.object(queue, "identity", side_effect=FileNotFoundError):
            queue._prior_child_is_live(self.output)

    def test_owned_launch_binds_exact_argv_uid_and_cwd_before_ledger(self):
        command = [queue.sys.executable, "diagnostic.py"]
        exact = dict(self.exact, argv=command)
        child = Mock(pid=12345)
        child.poll.return_value = None
        with patch.object(queue.subprocess, "Popen", return_value=child) as popen, \
                patch.object(queue, "identity", side_effect=[dict(exact, uid=exact["uid"] + 1), exact]), \
                patch.object(queue.time, "sleep"):
            actual_child, actual_identity = queue.launch_owned(command, self.root / "launch.log", gpu=3)
        self.assertIs(actual_child, child)
        self.assertEqual(actual_identity, exact)
        self.assertEqual(popen.call_args.kwargs["env"]["CUDA_VISIBLE_DEVICES"], "3")
        child.terminate.assert_not_called()

    def test_unbindable_launch_stops_only_the_owned_popen_child(self):
        child = Mock(pid=12345)
        child.poll.return_value = None
        with patch.object(queue.subprocess, "Popen", return_value=child), \
                patch.object(queue, "identity", return_value=dict(self.exact, cwd="/other")), \
                patch.object(queue.time, "sleep"):
            with self.assertRaisesRegex(RuntimeError, "bind owned child"):
                queue.launch_owned(["python", "diagnostic.py"], self.root / "launch.log")
        child.terminate.assert_called_once()
        child.wait.assert_called_once_with(timeout=20)

    def test_source_and_audit_mutations_are_rejected_before_runtime_work(self):
        source = self.root / "source.py"
        source.write_text("frozen = True\n")
        p = dict(self.protocol, audit_sha256=queue.sha(self.audit), sources={str(source): queue.sha(source)})
        queue.write(self.output / "protocol.json", p)
        queue.write(self.output / "registration.json", {"protocol_sha256": queue.sha(self.output / "protocol.json")})
        with patch.object(study, "verify", return_value={}):
            self.assertEqual(queue.verify(self.output), p)
            source.write_text("frozen = False\n")
            with self.assertRaisesRegex(ValueError, "source changed"):
                queue.verify(self.output)
            source.write_text("frozen = True\n")
            queue.write(self.audit, dict(self.audit_value, different_rows=1))
            with self.assertRaisesRegex(ValueError, "audit changed"):
                queue.verify(self.output)

    def test_stop_file_blocks_work_before_gpu_inventory(self):
        (self.study / "STOP").write_text("hold")
        with patch.object(queue, "verify", return_value=self.protocol), patch.object(queue, "inventory") as inv:
            with self.assertRaisesRegex(RuntimeError, "Stopped by"):
                queue.run(self.output)
        inv.assert_not_called()


if __name__ == "__main__":
    unittest.main()
