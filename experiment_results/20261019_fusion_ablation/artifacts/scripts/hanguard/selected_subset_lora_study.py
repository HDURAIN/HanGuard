"""Fresh LoRA and five readouts on a post hoc, error-selected test subset.

Train and validation are unchanged. The test excludes 122 WildGuard rows where
the previous frozen-base best single layer was correct and its anchor fusion
was wrong. This is a conditional exploratory comparison, not quality cleaning
or a blind estimate. The old +2.10 percentage-point result is not a prediction
of the fresh LoRA result. ``init`` is CPU-only; ``run`` waits for empty cards.
"""
from __future__ import annotations

from contextlib import contextmanager
import importlib.util
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
SCHEMA = 'hanguard_selected122_fresh_lora_v1'
READOUT_SCHEMA = 'hanguard_selected122_frozen_lora_readouts_v1'
SELECTION_POLICY = 'posthoc_error_selected_test_subset'
DEFAULT_DATA = ROOT / 'data/three_source_wildguard122_subset_20261018'
IMPLEMENTATION = ROOT / 'scripts/hanguard/fragment_clean_lora_study.py'
SCOPE = ('Post hoc error-selected test subset: unchanged train and validation; '
         '122 WildGuard test rows excluded using previous frozen-base model errors. '
         'Fresh E03 LoRA shared by all five frozen readouts; validation-only model '
         'selection within this exploratory study; not blind or quality-cleaned.')
LIMITATIONS = [
    'Test membership was selected post hoc using labels and two previous models: '
    'best single layer correct and anchor fusion wrong on 122 WildGuard rows.',
    'This subset comparison is conditional and exploratory, not a blind estimate '
    'or evidence that excluded rows have bad labels or defective text.',
    'Train and validation bytes are unchanged; no outcome-dependent filtering '
    'is applied inside the training loop.',
    'The previous +2.10 percentage-point gain belongs only to the old fixed '
    'no-LoRA models on this subset; a fresh LoRA gain is unknown.',
    'One fresh E03 LoRA seed; all five readouts share exactly that selected frozen adapter.',
    'Complete original microbatch groups are distributed; different batch shapes '
    'are not claimed equivalent.',
]

# A private module keeps all proven worker/resume mathematics intact and avoids
# changing globals in an imported historical implementation. Its __file__ must
# point to this entrypoint so every subprocess reloads this scientific contract.
_spec = importlib.util.spec_from_file_location(
    'scripts.hanguard._selected122_lora_runtime', IMPLEMENTATION)
_runtime = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_runtime)
_runtime.__file__ = __file__
_runtime.__doc__ = __doc__
_runtime.SCHEMA = SCHEMA
_runtime.READOUT_SCHEMA = READOUT_SCHEMA
_base_write, _base_sources = _runtime.write, _runtime.source_paths
_base_parser, _base_initialize = _runtime.parser, _runtime.initialize


def scoped(value):
    """Attach the selection caveat before any protocol/artifact is hash locked."""
    if isinstance(value, dict):
        value.update(scientific_scope=SELECTION_POLICY,
                     subset_selection_policy=SELECTION_POLICY,
                     is_quality_cleanup=False, blind_test=False)
        if value.get('schema') in (SCHEMA, READOUT_SCHEMA) or 'no_lora' in value:
            value.update(scope=SCOPE, limitations=list(LIMITATIONS))
    return value


def write(path, value):
    if isinstance(value, dict) and value.get('schema') == SCHEMA and 'data_dir' in value:
        # Bind the exact excluded IDs and their selection policy in addition to
        # the implementation's original split/model/metadata hash contracts.
        data = Path(value['data_dir'])
        for source in sorted(p for p in data.iterdir() if p.is_file()):
            name = str(source.resolve())
            value['inputs'][name] = _runtime.sha(source)
            stat = source.stat()
            value['input_stats'][name] = dict(size=stat.st_size, mtime_ns=stat.st_mtime_ns)
    # Internal resume markers compare exact dictionaries constructed by the
    # historical runner; keep their shape unchanged. Annotate the scientific
    # protocols/selections/results and visible coordinator state instead.
    if isinstance(value, dict) and (value.get('schema') in (SCHEMA, READOUT_SCHEMA) or
                                   'no_lora' in value or Path(path).name == 'state.json'):
        scoped(value)
    _base_write(path, value)


def source_paths():
    return sorted(set(_base_sources()) | {Path(__file__).resolve(), IMPLEMENTATION})


def parser():
    result = _base_parser()
    result.set_defaults(data=DEFAULT_DATA)
    return result


def validate_selection_policy(data):
    data = Path(data)
    manifest = _runtime.read(data / 'manifest.json')
    policy = _runtime.read(data / 'filter_policy.json')
    if (manifest.get('selection_policy') != SELECTION_POLICY or
            policy.get('selection_policy') != SELECTION_POLICY or
            manifest.get('excluded_test_rows') != 122 or policy.get('selected_rows') != 122 or
            manifest.get('train_validation_byte_identical') is not True or
            policy.get('train_validation_byte_identical') is not True or
            manifest.get('is_quality_cleanup') is not False or
            policy.get('is_quality_cleanup') is not False):
        raise ValueError('Require the registered post hoc 122-row test subset; unchanged train/validation')
    if not (data / 'excluded_base_ids.txt').is_file():
        raise FileNotFoundError(data / 'excluded_base_ids.txt')


@contextmanager
def scoped_child_registration():
    from scripts.hanguard import current_training as training
    build, files = training.build_protocol, training.code_files

    def build_protocol(args):
        protocol, audit = build(args)
        protocol.update(scope=SCOPE, limitations=list(LIMITATIONS))
        return scoped(protocol), audit

    def code_files(stage):
        return sorted(set(files(stage)) | {
            str(Path(__file__).resolve().relative_to(ROOT)),
            str(IMPLEMENTATION.relative_to(ROOT)),
        })

    training.build_protocol, training.code_files = build_protocol, code_files
    try:
        yield
    finally:
        training.build_protocol, training.code_files = build, files


def initialize(args):
    validate_selection_policy(args.data)
    with scoped_child_registration():
        return _base_initialize(args)


_runtime.write = write
_runtime.source_paths = source_paths
_runtime.parser = parser
_runtime.initialize = initialize


def __getattr__(name):
    return getattr(_runtime, name)


def main(argv=None):
    return _runtime.main(argv)


if __name__ == '__main__':
    main()
