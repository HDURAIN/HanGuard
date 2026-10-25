"""CPU-only deterministic configuration selection and unchanged pruning tests."""
import types
import unittest

from scripts.hanguard import clean_ablation_runtime as runtime


class Config:
    def __init__(self, block, warps=4):
        self.kwargs = dict(BLOCK=block)
        self.num_warps, self.num_stages, self.num_ctas = warps, 2, 1
        self.maxnreg = None
        self.pre_hook = None


class Tuner:
    def __init__(self, configs):
        self.configs = configs
        self.cache = {'stale': configs[0]}
        self.nargs = dict(T=130)
        self.pruned_calls = 0

    def prune_configs(self, kwargs):
        self.pruned_calls += 1
        return [config for config in self.configs if config.kwargs['BLOCK'] <= kwargs['limit']]


class RuntimeTests(unittest.TestCase):
    def _modules(self, tuner):
        module = types.ModuleType('fla.ops.synthetic')
        module.kernel = types.SimpleNamespace(fn=tuner)
        module.alias = tuner
        return {'fla.ops.synthetic': module}

    def test_deterministic_singleton_preserves_original_pruning_and_clears_cache(self):
        tuner = Tuner([Config(256), Config(64), Config(128)])
        path = runtime.DeterministicKernelPath(modules=self._modules(tuner), autotuner_type=Tuner)
        self.assertEqual(tuner.cache, {})
        selected = tuner.prune_configs(dict(limit=128))
        self.assertEqual(len(selected), 1)
        self.assertLessEqual(selected[0].kwargs['BLOCK'], 128)
        self.assertEqual(tuner.pruned_calls, 1)
        self.assertEqual(selected, tuner.prune_configs(dict(limit=128)))
        snapshot = path.snapshot()
        self.assertEqual(snapshot['autotuners_discovered'], 1)
        self.assertTrue(snapshot['real_benchmarks_unchanged'])
        self.assertFalse(snapshot['installed_files_modified'])
        self.assertEqual(len(snapshot['entries'][0]['selections']), 1)

    def test_candidate_input_order_cannot_change_choice(self):
        choices = []
        for configs in ([Config(64), Config(128)], [Config(128), Config(64)]):
            tuner = Tuner(configs)
            runtime.DeterministicKernelPath(modules=self._modules(tuner), autotuner_type=Tuner)
            choices.append(runtime.canonical_config(tuner.prune_configs(dict(limit=256))[0]))
        self.assertEqual(*choices)

    def test_no_valid_configuration_is_a_hard_failure(self):
        tuner = Tuner([Config(128)])
        runtime.DeterministicKernelPath(modules=self._modules(tuner), autotuner_type=Tuner)
        with self.assertRaisesRegex(RuntimeError, 'no configurations'):
            tuner.prune_configs(dict(limit=1))

    def test_missing_fla_does_not_silently_fallback(self):
        with self.assertRaisesRegex(RuntimeError, 'No FLA'):
            runtime.DeterministicKernelPath(modules={}, autotuner_type=Tuner)

    def test_math_sdpa_context_is_cpu_safe(self):
        with runtime.math_sdpa():
            pass


if __name__ == '__main__':
    unittest.main()
