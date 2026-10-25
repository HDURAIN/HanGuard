"""Process-local deterministic choices for the clean feature study.

No installed file is modified.  Original Triton validity/performance pruning
still runs; its valid configurations are sorted canonically and exactly one
is passed to the real autotuner benchmark.  Benchmarks and kernel execution
are unchanged.  Each independent process records all candidates and choices.
"""
from __future__ import annotations

from contextlib import contextmanager
import hashlib
import json
import sys
import types

import torch


def _canonical(value):
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(k): _canonical(v) for k, v in sorted(value.items(), key=lambda x: str(x[0]))}
    if isinstance(value, (list, tuple)):
        return [_canonical(v) for v in value]
    if isinstance(value, torch.Tensor):
        return dict(shape=list(value.shape), stride=list(value.stride()), dtype=str(value.dtype))
    return str(value)


def canonical_config(config):
    result = dict(kwargs=_canonical(config.kwargs))
    for key in ('num_warps', 'num_stages', 'num_ctas', 'num_buffers_warp_spec',
                'num_consumer_groups', 'reg_dec_producer', 'reg_inc_consumer',
                'maxnreg', 'enable_fp_fusion'):
        if hasattr(config, key):
            result[key] = _canonical(getattr(config, key))
    hook = getattr(config, 'pre_hook', None)
    result['pre_hook'] = None if hook is None else f'{hook.__module__}.{hook.__qualname__}'
    return result


def _key(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)


def _sha(value):
    return hashlib.sha256(_key(value).encode()).hexdigest()


def discover_autotuners(modules=None, autotuner_type=None):
    if autotuner_type is None:
        from triton.runtime.autotuner import Autotuner
        autotuner_type = Autotuner
    modules = sys.modules if modules is None else modules
    found = {}
    for module_name, module in sorted(list(modules.items())):
        if not module_name.startswith('fla.') or module is None:
            continue
        for name, candidate in sorted(list(vars(module).items())):
            seen = set()
            depth = 0
            while candidate is not None and id(candidate) not in seen and depth < 12:
                seen.add(id(candidate))
                if isinstance(candidate, autotuner_type):
                    if id(candidate) not in found:
                        found[id(candidate)] = (f'{module_name}.{name}' + '.fn' * depth, candidate)
                candidate = getattr(candidate, 'fn', None)
                depth += 1
    return sorted(found.values(), key=lambda pair: pair[0])


class DeterministicKernelPath:
    def __init__(self, *, require_fla=True, modules=None, autotuner_type=None):
        self.entries = []
        self.objects = discover_autotuners(modules=modules, autotuner_type=autotuner_type)
        if require_fla and not self.objects:
            raise RuntimeError('No FLA Triton autotuners discovered; canonical-valid path unavailable')
        for name, tuner in self.objects:
            if getattr(tuner, '_clean_ablation_patched', False):
                raise ValueError('Clean numerical path has already been installed in this process')
            original_prune = tuner.prune_configs
            entry = dict(kernel=name, candidate_configs=sorted([canonical_config(c) for c in tuner.configs], key=_key),
                         selections={})
            entry['candidate_configs_sha256'] = _sha(entry['candidate_configs'])
            self.entries.append(entry)

            def prune(this, kwargs, _original=original_prune, _entry=entry):
                valid = list(_original(kwargs))
                if not valid:
                    raise RuntimeError(f'Original FLA validity pruning returned no configurations: {_entry["kernel"]}')
                selected = min(valid, key=lambda config: _key(canonical_config(config)))
                invocation = _canonical(dict(arguments=getattr(this, 'nargs', {}), keywords=kwargs))
                signature = _sha(invocation)
                record = dict(invocation=invocation, selected=canonical_config(selected),
                              valid_configs_sha256=_sha(sorted([canonical_config(c) for c in valid], key=_key)),
                              valid_config_count=len(valid))
                prior = _entry['selections'].get(signature)
                if prior is not None and prior != record:
                    raise RuntimeError('Canonical-valid FLA kernel choice changed within one process')
                _entry['selections'][signature] = record
                return [selected]

            tuner.cache.clear()
            tuner.prune_configs = types.MethodType(prune, tuner)
            tuner._clean_ablation_patched = True
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False

    def snapshot(self):
        entries = []
        for (name, tuner), entry in zip(self.objects, self.entries):
            current = dict(entry, selections=[dict(signature=k, **v) for k, v in sorted(entry['selections'].items())])
            current['cached_config_choices'] = sorted([dict(key=_canonical(key), config=canonical_config(config))
                for key, config in tuner.cache.items()], key=_key)
            best = getattr(tuner, 'best_config', None)
            current['last_executed_config'] = None if best is None else canonical_config(best)
            entries.append(current)
        return dict(schema='hanguard_clean_process_local_kernel_path_v1',
                    sdpa='math', fla_autotune='canonical_valid_config',
                    autotuners_discovered=len(entries), real_benchmarks_unchanged=True,
                    installed_files_modified=False, entries=entries)


@contextmanager
def math_sdpa():
    from torch.nn.attention import SDPBackend, sdpa_kernel
    with sdpa_kernel(SDPBackend.MATH):
        yield
