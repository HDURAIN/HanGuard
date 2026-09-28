"""Plan, archive, verify, then explicitly remove a fixed list of retired artifacts.

Default: generate outputs/repository_cleanup_20260928/plan.json only.
Use --apply only after reviewing that plan. Current data, source data, active
experiments, environments, and unlisted new directories are never selected.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import stat
import tarfile
import tempfile

ROOT = Path(__file__).resolve().parents[2]
EXPECTED_ROOT = Path('/mnt/data1/zhouhanyu/projects/jailbreak-defense/hanguard')
_SELF_TEST_ROOTS = set()
RECORD_DIR = 'outputs/repository_cleanup_20260928'
ARCHIVE_LIMIT = 16 * 1024 * 1024
OLD_DATA = (
    'data/academic', 'data/three_source', 'data/three_source_original',
    'data/hanguard_multilabel_pilot_20260928', 'data/hanguard_primary_intent_20260928',
)
OLD_OUTPUT_NAMES = (
    'hanguard_token_20260922_204648', 'hanguard_classification_shape_diagnostic_20260928',
    'hanguard_lora_fusion_pilot_20260923', 'hanguard_error_analysis_20260928',
    'hanguard_classification_study_20260928', 'hanguard_position_control_smoke',
    'hanguard_repaired_study_20260928', 'hanguard_classification_preflight_20260928',
    'hanguard_v5_new', 'hanguard_fusion_smoke', 'hanguard_head_improvements_smoke_20260923',
    'hanguard_primary_intent_20260928', 'hanguard_speed_smoke', 'hanguard_token_smoke',
    'hanguard_zero_shot_20260923', 'hanguard_joint_readout_20260924',
    'hanguard_qwen35_4b_smoke', 'hanguard_position_control_20260922_213522',
    'hanguard_shared_concat_20260923_162415', 'hanguard_shared_concat_smoke_20260923_162415',
    'hanguard_local_pooling_20260924', 'hanguard', 'hanguard_fusion', 'comparison',
    'hanguard_residual_fusion_20260924', 'eval_v5',
    'hanguard_classification_preflight_micro64_20260928',
    'hanguard_classification_preflight_pad32_20260928', 'injection_review',
    'hanguard_optimization_report_20260924', 'hanguard_qwen35_4b_tiny', 'hanguard_v5',
    'hanguard_mlp_20260922_195850', 'hanguard_head_improvements_20260923',
    'hanguard_multilabel_study_20260928', 'hanguard_mlp_smoke',
    'hanguard_depth_local_factorial_20260924', 'hanguard_primary_category_study_20260928',
)
OLD_MODELS = ('models/HY-MT1.5-1.8B', 'models/qwen35-tiny-test',
              'models/Qwen3.5-4B-modelscope', 'legacy/models', 'legacy/data')
TARGETS = OLD_DATA + tuple('outputs/' + name for name in OLD_OUTPUT_NAMES) + OLD_MODELS
KEEP = (
    'data/sources', 'data/three_source_translation_repaired',
    'data/hanguard_two_source_primary_20260928',
    'outputs/hanguard_repaired_core_20260928',
    'outputs/hanguard_translation_repair_20260928',
    'outputs/hanguard_two_source_primary_20260928',
    'outputs/storage_cleanup_20260928', RECORD_DIR,
    'models/Qwen3.5-4B', 'models/Qwen3Guard-Gen-4B', 'models/hanguard',
    'scripts', 'docs', 'tests', 'archive', '.venv-hanguard',
)
CACHE_SOURCE = 'outputs/hanguard_repaired_study_20260928/kernel_cache'
CACHE_DEST = 'outputs/hanguard_repaired_core_20260928/kernel_cache'
BEFORE_ARCHIVE = 'archive/data_before_repair_20260928.tar.gz'
OLD_ARCHIVE = 'archive/old_experiments_20260928.tar.gz'
PROVENANCE_DIR = 'archive/source_provenance'


def digest_stream(stream):
    value = hashlib.sha256()
    for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
        value.update(block)
    return value.hexdigest()


def file_sha(path):
    with Path(path).open('rb') as stream:
        return digest_stream(stream)


def canonical_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                    separators=(',', ':')).encode()).hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.writing')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    temporary.replace(path)


def validate_root(root):
    root = Path(root).resolve()
    if root != EXPECTED_ROOT or root != EXPECTED_ROOT.resolve():
        if root not in _SELF_TEST_ROOTS:
            raise ValueError(f'Cleanup is restricted to the fixed repository root: {EXPECTED_ROOT}')
    if root == Path('/') or len(root.parts) < 3:
        raise ValueError('Unsafe repository root')
    if not (root / '.git').exists() or not (root / 'data/sources').is_dir():
        raise ValueError('Expected repository markers .git and data/sources')
    return root


def safe_path(root, relative):
    relative = Path(relative)
    if relative.is_absolute() or not relative.parts or any(x in {'.', '..'} for x in relative.parts):
        raise ValueError(f'Unsafe relative path: {relative}')
    path = root / relative
    resolved = path.resolve(strict=False)
    if resolved == root or root not in resolved.parents:
        raise ValueError(f'Outside-repository path or symlink: {relative}')
    # Internal leaf symlinks may be inventoried/unlinked, but parent symlinks
    # must not turn a directory deletion into a different tree operation.
    for parent in path.parents:
        if parent == root:
            break
        if parent.is_symlink():
            raise ValueError(f'Symlink parent is not allowed: {parent}')
    return path


def describe(root, path):
    relative = path.relative_to(root).as_posix()
    safe_path(root, relative)
    before = path.lstat()
    if stat.S_ISLNK(before.st_mode):
        target = os.readlink(path)
        return dict(path=relative, kind='symlink', bytes=before.st_size,
                    sha256=hashlib.sha256(target.encode()).hexdigest(), target=target)
    if stat.S_ISDIR(before.st_mode):
        return dict(path=relative, kind='directory', bytes=0, sha256=None)
    if not stat.S_ISREG(before.st_mode):
        raise ValueError(f'Refusing special filesystem object: {relative}')
    digest = file_sha(path)
    after = path.stat()
    if (before.st_size, before.st_mtime_ns, before.st_ino) != (after.st_size, after.st_mtime_ns, after.st_ino):
        raise ValueError(f'File changed while hashing: {relative}')
    return dict(path=relative, kind='file', bytes=before.st_size, sha256=digest)


def scan(root, relative, exclusions=()):
    path = safe_path(root, relative)
    if not os.path.lexists(path):
        return []
    result = []
    excluded = set(exclusions)

    def visit(current):
        rel = current.relative_to(root).as_posix()
        if rel in excluded:
            return
        item = describe(root, current)
        result.append(item)
        if item['kind'] == 'directory':
            for child in sorted(current.iterdir()):
                visit(child)

    visit(path)
    return result


def verify_rows(root, rows, allow_missing=False, remap=None):
    remap = remap or {}
    for original in rows:
        relative = original['path']
        for old, new in remap.items():
            if relative == old or relative.startswith(old + '/'):
                relative = new + relative[len(old):]
                break
        path = safe_path(root, relative)
        if not os.path.lexists(path):
            if allow_missing:
                continue
            raise ValueError(f'Required artifact missing: {relative}')
        actual = describe(root, path)
        for field in ('kind', 'bytes', 'sha256', 'target'):
            if actual.get(field) != original.get(field):
                raise ValueError(f'Artifact changed since plan: {relative} ({field})')


def metadata_candidate(item):
    if item['kind'] != 'file' or item['bytes'] > ARCHIVE_LIMIT:
        return False
    path = Path(item['path'])
    if 'kernel_cache' in path.parts or 'inductor_cache' in path.parts:
        return False
    if path.name in {'tokenizer.json', 'tokenizer_config.json', 'vocab.json', 'merges.txt',
                     'added_tokens.json', 'raw_responses.jsonl', 'annotations.jsonl'}:
        return False
    if path.suffix.lower() in {'.md', '.json', '.py'}:
        return True
    if path.suffix.lower() == '.txt':
        return any(term in path.name.lower() for term in ('report', 'summary', 'metric', 'readme', 'license'))
    if path.suffix.lower() in {'.csv', '.parquet', '.npz', '.jsonl'}:
        return any(term in path.name.lower() for term in (
            'prediction', 'result', 'metric', 'review', 'audit', 'summary')) or path.stem in {'test', 'validation'}
    return False


def make_plan(root):
    root = validate_root(root)
    if safe_path(root, RECORD_DIR).is_symlink():
        raise ValueError('Cleanup record directory cannot be a symlink')
    if (root / RECORD_DIR / 'result.json').exists():
        result = json.loads((root / RECORD_DIR / 'result.json').read_text())
        if result.get('status') == 'completed':
            return json.loads((root / RECORD_DIR / 'plan.json').read_text())
    targets = []
    for relative in TARGETS:
        for kept in KEEP:
            if relative == kept or relative.startswith(kept + '/') or kept.startswith(relative + '/'):
                raise ValueError(f'Cleanup target conflicts with preserved tree: {relative}/{kept}')
        rows = scan(root, relative, exclusions=(CACHE_SOURCE,))
        if rows:
            targets.append(dict(path=relative, absolute_path=str(root / relative),
                                bytes=sum(row['bytes'] for row in rows),
                                files=sum(row['kind'] != 'directory' for row in rows), entries=rows))
    all_rows = [row for target in targets for row in target['entries']]
    lookup = {row['path']: row for row in all_rows}
    before = [row for row in all_rows if row['kind'] == 'file'
              and row['path'].startswith('data/three_source_original/')
              and (Path(row['path']).name in {'train.parquet', 'validation.parquet', 'test.parquet'}
                   or Path(row['path']).suffix in {'.json', '.md'})]
    if any(target['path'] == 'data/three_source_original' for target in targets):
        for split in ('train', 'validation', 'test'):
            if f'data/three_source_original/{split}.parquet' not in {row['path'] for row in before}:
                raise ValueError('All three before-repair split files must be archived')
    old_records = [row for row in all_rows if metadata_candidate(row)
                   and not row['path'].startswith('data/three_source_original/')]
    provenance = []
    mapping_path = root / 'outputs/hanguard_translation_repair_20260928/mapping_manifest.json'
    if mapping_path.exists():
        mapping = json.loads(mapping_path.read_text())
        for relative, entry in mapping.get('source_files', {}).items():
            if not relative.startswith('legacy/data/'):
                continue
            if relative not in lookup or lookup[relative]['sha256'] != entry['sha256']:
                raise ValueError(f'Legacy provenance missing or hash mismatch: {relative}')
            provenance.append(lookup[relative])
    elif (root / 'legacy/data').exists():
        raise ValueError('Missing translation mapping manifest for legacy-data preservation')
    cache_rows = scan(root, CACHE_SOURCE)
    cache_dest = safe_path(root, CACHE_DEST)
    if cache_rows:
        if not cache_dest.is_symlink() or cache_dest.resolve() != (root / CACHE_SOURCE).resolve():
            raise ValueError('Expected core compile-cache symlink to retired study')
    elif cache_dest.is_symlink():
        raise ValueError('Core compile-cache symlink has no available target')
    plan = dict(schema_version=1, status='dry_run', root=str(root),
        created_at=datetime.now(timezone.utc).isoformat(), script_sha256=file_sha(__file__),
        explicit_target_allowlist=list(TARGETS), retained=list(KEEP), targets=targets,
        bytes_to_remove=sum(t['bytes'] for t in targets),
        file_count_to_remove=sum(t['files'] for t in targets),
        archives=[dict(path=BEFORE_ARCHIVE, members=before), dict(path=OLD_ARCHIVE, members=old_records)],
        metadata_archive_per_file_limit_bytes=ARCHIVE_LIMIT,
        source_provenance=dict(directory=PROVENANCE_DIR, members=provenance),
        cache_relocation=dict(source=CACHE_SOURCE, destination=CACHE_DEST, entries=cache_rows,
                              bytes=sum(row['bytes'] for row in cache_rows)),
        reconstruction_notes=[
            'Current repaired data, full archive/quarantine, all raw data/sources, both active studies and their model dependencies stay in place.',
            'Before-repair three splits are archived byte-for-byte with metadata. Restore these paths from data_before_repair_20260928.tar.gz before using historical rebuild/audit commands.',
            'Legacy source-lineage inputs are copied under archive/source_provenance with their original relative paths and hashes. Restore those paths to rerun original source recovery.',
            'The frozen source_mapping and production/merged translation outputs remain available, so subsequent translation merge can be replayed without rerunning historical source recovery.',
            'Retired split generators, intermediate data, checkpoints and caches are removed. Metadata preserves reported results, not executable model weights; old experiments require regeneration/retraining.',
            'Unlisted new output/model/data directories, scripts/docs/tests, Git history and virtual environments are untouched.',
        ])
    plan['plan_sha256'] = canonical_sha(plan)
    write_json(root / RECORD_DIR / 'plan.json', plan)
    return plan


def verify_tar(path, members):
    expected = {row['path']: row for row in members}
    with tarfile.open(path, 'r:gz') as archive:
        actual = archive.getmembers()
        if len(actual) != len(expected) or {entry.name for entry in actual} != set(expected):
            raise ValueError(f'Archive inventory mismatch: {path}')
        for entry in actual:
            if not entry.isfile() or entry.size != expected[entry.name]['bytes']:
                raise ValueError(f'Archive contains unexpected member type/size: {entry.name}')
            stream = archive.extractfile(entry)
            if stream is None or digest_stream(stream) != expected[entry.name]['sha256']:
                raise ValueError(f'Archive content hash mismatch: {entry.name}')


def ensure_tar(root, specification):
    path = safe_path(root, specification['path'])
    if path.is_symlink():
        raise ValueError('Archive destination cannot be a symlink')
    members = specification['members']
    if path.exists():
        verify_tar(path, members)
        return dict(path=specification['path'], sha256=file_sha(path), bytes=path.stat().st_size,
                    member_count=len(members), verified=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + '.partial')
    if partial.exists():
        partial.unlink()  # Owned incomplete archive; it was never a verified backup.
    verify_rows(root, members)
    with tarfile.open(partial, 'w:gz', compresslevel=6) as archive:
        for row in members:
            source = safe_path(root, row['path'])
            archive.add(source, arcname=row['path'], recursive=False)
    verify_tar(partial, members)
    partial.replace(path)
    return dict(path=specification['path'], sha256=file_sha(path), bytes=path.stat().st_size,
                member_count=len(members), verified=True)


def ensure_provenance(root, specification):
    directory = safe_path(root, specification['directory'])
    if directory.is_symlink():
        raise ValueError('Provenance directory cannot be a symlink')
    directory.mkdir(parents=True, exist_ok=True)
    records = []
    for row in specification['members']:
        destination_relative = specification['directory'] + '/' + row['path']
        destination = safe_path(root, destination_relative)
        if destination.exists():
            if destination.is_symlink() or file_sha(destination) != row['sha256']:
                raise ValueError(f'Existing provenance copy differs: {destination}')
        else:
            verify_rows(root, [row])
            destination.parent.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile(prefix=destination.name + '.', suffix='.partial',
                                             dir=destination.parent, delete=False) as stream:
                partial = Path(stream.name)
                with safe_path(root, row['path']).open('rb') as source:
                    shutil.copyfileobj(source, stream)
            if file_sha(partial) != row['sha256']:
                raise ValueError('Provenance copy failed hash verification')
            partial.replace(destination)
        records.append(dict(original_path=row['path'], archived_path=destination_relative,
                            bytes=row['bytes'], sha256=row['sha256']))
    value = dict(records=records, verified=True, restore_original_paths_for_source_recovery=True)
    if safe_path(root, specification['directory'] + '/manifest.json').is_symlink():
        raise ValueError('Provenance manifest cannot be a symlink')
    write_json(directory / 'manifest.json', value)
    return value


def relocate_cache(root, specification):
    rows = specification['entries']
    if not rows:
        return dict(status='not_needed')
    source = safe_path(root, specification['source'])
    destination = safe_path(root, specification['destination'])
    if not source.exists() and destination.is_dir() and not destination.is_symlink():
        verify_rows(root, rows, remap={specification['source']: specification['destination']})
        return dict(status='already_relocated', bytes=specification['bytes'])
    verify_rows(root, rows)
    if destination.is_symlink():
        if destination.resolve() != source.resolve():
            raise ValueError('Unexpected compile-cache link target')
        destination.unlink()
    elif destination.exists():
        raise FileExistsError('Refusing to replace an existing local compile-cache directory')
    source.rename(destination)
    verify_rows(root, rows, remap={specification['source']: specification['destination']})
    return dict(status='relocated', bytes=specification['bytes'])


def verify_target(root, target, allow_missing):
    actual = scan(root, target['path'], exclusions=(CACHE_SOURCE,))
    expected_paths = {row['path'] for row in target['entries']}
    actual_paths = {row['path'] for row in actual}
    if actual_paths - expected_paths or (not allow_missing and actual_paths != expected_paths):
        raise ValueError(f'Target tree changed after planning: {target["path"]}')
    verify_rows(root, target['entries'], allow_missing=allow_missing)


def remove_planned_entries(root, target):
    # Unlink only inventoried files. rmdir refuses unexpected new contents;
    # no recursive wildcard removal is used.
    for row in sorted(target['entries'], key=lambda item: len(Path(item['path']).parts), reverse=True):
        path = safe_path(root, row['path'])
        if not os.path.lexists(path):
            continue
        verify_rows(root, [row])
        if row['kind'] == 'directory':
            path.rmdir()
        else:
            path.unlink()


def apply_plan(root):
    root = validate_root(root)
    if safe_path(root, RECORD_DIR).is_symlink():
        raise ValueError('Cleanup record directory cannot be a symlink')
    plan_path = root / RECORD_DIR / 'plan.json'
    if not plan_path.exists():
        raise FileNotFoundError('Generate and review the dry-run plan first')
    plan = json.loads(plan_path.read_text())
    expected_digest = plan.pop('plan_sha256')
    if canonical_sha(plan) != expected_digest:
        raise ValueError('Plan content hash mismatch')
    plan['plan_sha256'] = expected_digest
    if plan['root'] != str(root) or plan['script_sha256'] != file_sha(__file__):
        raise ValueError('Plan belongs to another root or script; regenerate it before applying')
    if any(target['path'] not in TARGETS for target in plan['targets']):
        raise ValueError('Plan includes a target outside the explicit cleanup allowlist')
    planned_files = {}
    for target in plan['targets']:
        if target.get('absolute_path') != str(root / target['path']):
            raise ValueError('Plan absolute target differs from the fixed repository path')
        for row in target['entries']:
            if row['path'] != target['path'] and not row['path'].startswith(target['path'] + '/'):
                raise ValueError('Inventory member escapes its allowlisted target')
            safe_path(root, row['path'])
            planned_files[row['path']] = row
    if [entry['path'] for entry in plan['archives']] != [BEFORE_ARCHIVE, OLD_ARCHIVE]:
        raise ValueError('Unexpected archive destination')
    for archive in plan['archives']:
        for row in archive['members']:
            if row != planned_files.get(row['path']) or row['kind'] != 'file':
                raise ValueError('Archive member is not an inventoried regular file')
    if plan['source_provenance']['directory'] != PROVENANCE_DIR:
        raise ValueError('Unexpected provenance destination')
    for row in plan['source_provenance']['members']:
        if not row['path'].startswith('legacy/data/') or row != planned_files.get(row['path']):
            raise ValueError('Provenance member is not an inventoried legacy source')
    relocation = plan['cache_relocation']
    if (relocation['source'], relocation['destination']) != (CACHE_SOURCE, CACHE_DEST):
        raise ValueError('Unexpected cache relocation paths')
    for row in relocation['entries']:
        if row['path'] != CACHE_SOURCE and not row['path'].startswith(CACHE_SOURCE + '/'):
            raise ValueError('Cache inventory escapes its fixed source')
    result_path = root / RECORD_DIR / 'result.json'
    result = json.loads(result_path.read_text()) if result_path.exists() else {}
    if result.get('plan_sha256', expected_digest) != expected_digest:
        raise ValueError('Existing cleanup result belongs to a different plan')
    if result.get('status') == 'completed':
        for archive in plan['archives']:
            verify_tar(safe_path(root, archive['path']), archive['members'])
        for row in result['source_provenance']['records']:
            if file_sha(safe_path(root, row['archived_path'])) != row['sha256']:
                raise ValueError('Preserved provenance changed after cleanup')
        return result  # Never delete files newly created after a completed run.
    allow_missing = result.get('deletion_started', False)
    for target in plan['targets']:
        verify_target(root, target, allow_missing)
    archives = [ensure_tar(root, archive) for archive in plan['archives']]
    provenance = ensure_provenance(root, plan['source_provenance'])
    relocation = relocate_cache(root, plan['cache_relocation'])
    result.update(status='in_progress', plan_sha256=expected_digest, archives=archives,
                  source_provenance=provenance, cache_relocation=relocation,
                  deletion_started=True, completed_targets=result.get('completed_targets', []))
    write_json(result_path, result)
    for target in plan['targets']:
        if target['path'] in result['completed_targets']:
            if os.path.lexists(safe_path(root, target['path'])):
                raise ValueError('A retired directory was recreated during cleanup; refusing to remove it')
            continue
        verify_target(root, target, allow_missing=True)
        remove_planned_entries(root, target)
        result['completed_targets'].append(target['path'])
        write_json(result_path, result)
    result.update(status='completed', completed_at=datetime.now(timezone.utc).isoformat(),
                  removed_bytes=plan['bytes_to_remove'], removed_file_count=plan['file_count_to_remove'],
                  backup_bytes=sum(item['bytes'] for item in archives)
                    + sum(item['bytes'] for item in provenance['records']),
                  preserved=list(KEEP), reconstruction_notes=plan['reconstruction_notes'])
    write_json(result_path, result)
    return result


def self_test():
    with tempfile.TemporaryDirectory(prefix='hanguard-cleanup-test-') as temporary:
        root = Path(temporary)
        _SELF_TEST_ROOTS.add(root)
        (root / '.git').mkdir()
        (root / 'data/sources').mkdir(parents=True)
        source = root / 'data/three_source_original'
        source.mkdir()
        for split in ('train', 'validation', 'test'):
            (source / f'{split}.parquet').write_bytes(f'fixture-{split}'.encode())
        (source / 'manifest.json').write_text('{}')
        cache = root / CACHE_SOURCE
        cache.mkdir(parents=True)
        (cache / 'kernel.bin').write_bytes(b'compiled-fixture')
        core_link = root / CACHE_DEST
        core_link.parent.mkdir(parents=True)
        core_link.symlink_to(cache, target_is_directory=True)
        retired = root / 'outputs/hanguard_v5'
        retired.mkdir()
        (retired / 'report.md').write_text('fixture result')
        (retired / 'head.pt').write_bytes(b'old-model')
        (root / 'outputs/new_unlisted_experiment').mkdir()
        (root / 'outputs/new_unlisted_experiment/keep.txt').write_text('untouched')
        plan = make_plan(root)
        assert source.exists() and not (root / BEFORE_ARCHIVE).exists()
        (retired / 'new_after_plan.txt').write_text('must not silently delete')
        try:
            apply_plan(root)
        except ValueError:
            pass
        else:
            raise AssertionError('Mutation after dry-run was not rejected')
        (retired / 'new_after_plan.txt').unlink()
        external = retired / 'external'
        external.symlink_to('/etc/passwd')
        try:
            make_plan(root)
        except ValueError:
            pass
        else:
            raise AssertionError('External symlink was not rejected')
        external.unlink()
        result = apply_plan(root)
        assert result['status'] == 'completed' and not source.exists() and not retired.exists()
        assert core_link.is_dir() and not core_link.is_symlink()
        assert (root / 'outputs/new_unlisted_experiment/keep.txt').read_text() == 'untouched'
        assert apply_plan(root)['status'] == 'completed'
        verify_tar(root / BEFORE_ARCHIVE, plan['archives'][0]['members'])
        _SELF_TEST_ROOTS.remove(root)
    return dict(status='passed', checks=['default_is_plan_only', 'reject_changed_tree',
        'reject_external_symlink', 'verify_archives_before_delete', 'preserve_unlisted_outputs',
        'relocate_core_cache', 'idempotent_completed_apply'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--apply', action='store_true', help='Apply only the previously generated and verified plan')
    parser.add_argument('--self-test', action='store_true', help='Exercise destructive operations only inside a temporary fixture')
    args = parser.parse_args()
    if ROOT != EXPECTED_ROOT or ROOT != EXPECTED_ROOT.resolve():
        parser.error(f'This cleanup script must remain inside {EXPECTED_ROOT}')
    if args.self_test:
        if args.apply:
            parser.error('--self-test and --apply are mutually exclusive')
        value = self_test()
    elif args.apply:
        value = apply_plan(ROOT)
    else:
        plan = make_plan(ROOT)
        value = dict(status='dry_run', plan=str(ROOT / RECORD_DIR / 'plan.json'),
                     target_directories=len(plan['targets']), files=plan['file_count_to_remove'],
                     bytes=plan['bytes_to_remove'], GiB=plan['bytes_to_remove'] / 2**30,
                     cache_bytes_preserved=plan['cache_relocation']['bytes'],
                     archive_members={entry['path']: len(entry['members']) for entry in plan['archives']},
                     source_provenance_files=len(plan['source_provenance']['members']))
    print(json.dumps(value, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
