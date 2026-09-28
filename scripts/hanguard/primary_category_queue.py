"""Run the three existing-label category heads, then unlock their common test."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.hanguard import primary_category_contract as contract
PYTHON = str(ROOT/'.venv-hanguard/bin/python')
ARMS = ('last_mlp', 'learned_queries', 'description_queries')


def sha(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            value.update(block)
    return value.hexdigest()


def dump(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n')
    temporary.replace(path)


class Queue:
    def __init__(self, study):
        self.study = Path(study).resolve()
        self.protocol_path = self.study/'protocol.json'
        self.protocol = json.loads(self.protocol_path.read_text())
        self.protocol_sha = sha(self.protocol_path)
        self.logs = self.study/'logs'
        self.logs.mkdir(exist_ok=True)
        self.code_hashes = {}
        self.audit_hash = None
        self.began = time.time()

    def status(self, stage, **details):
        value = dict(stage=stage, pid=os.getpid(), started=self.began, updated=time.time(),
                     seconds=time.time()-self.began, protocol_sha256=self.protocol_sha, **details)
        dump(self.study/'queue_status.json', value)
        print(json.dumps(value, ensure_ascii=False), flush=True)

    def check(self):
        if sha(self.protocol_path) != self.protocol_sha:
            raise ValueError('Primary-category protocol changed during execution')
        seeds = self.protocol['seeds']
        if self.protocol['all_arms'] != list(ARMS) or not seeds or len(set(seeds)) != len(seeds) or any(type(seed) is not int or seed < 0 for seed in seeds):
            raise ValueError('Register three arms and unique nonnegative integer seeds')
        if self.protocol.get('new_multilabel_annotations_used') is not False:
            raise ValueError('This study must use only the existing primary labels')
        if self.protocol.get('task', 'existing_primary_category_classification') not in ('existing_primary_category_classification', contract.TASK):
            raise ValueError('Unknown primary-category protocol task')
        if contract.is_intent(self.protocol):
            contract.validate_protocol(self.protocol)
            for split in contract.SPLITS:
                contract.original_frame(self.protocol, split)
        for split, digest in self.protocol['data_sha256'].items():
            if sha(Path(self.protocol['data_dir'])/f'{split}.parquet') != digest:
                raise ValueError(f'Existing source data changed: {split}')
        if sha(self.protocol['description_file']) != self.protocol['description_sha256']:
            raise ValueError('Category descriptions changed')
        if self.audit_hash is not None and sha(self.study/'data_audit.json') != self.audit_hash:
            raise ValueError('Primary-category data audit changed during execution')
        for name, digest in self.code_hashes.items():
            if sha(ROOT/'scripts/hanguard'/name) != digest:
                raise ValueError(f'Primary-category experiment code changed: {name}')

    def run(self, action, arm=None, gpu=None, seed=42):
        self.check()
        name = f'{action}_{arm}_s{seed}' if arm else action
        if action == 'report':
            argv = [PYTHON, 'scripts/hanguard/primary_category_report.py', '--output', str(self.study)]
        else:
            argv = [PYTHON, 'scripts/hanguard/primary_category_study.py', action, '--output', str(self.study),
                    '--arm', arm, '--seed', str(seed)]
        env = os.environ.copy()
        env.update(PYTHONUNBUFFERED='1', OMP_NUM_THREADS='4', TOKENIZERS_PARALLELISM='false')
        if gpu is not None:
            env['CUDA_VISIBLE_DEVICES'] = str(gpu)
        with (self.logs/f'{name}.log').open('a') as log:
            process = subprocess.Popen(argv, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
            dump(self.logs/f'{name}.process.json', dict(pid=process.pid, gpu=gpu, seed=seed, started=time.time()))
            result = process.wait()
        if result:
            raise RuntimeError(f'{name} exited {result}; inspect logs/{name}.log')

    def execute(self):
        self.check()
        audit = json.loads((self.study/'data_audit.json').read_text())
        if audit.get('passed') is not True:
            raise ValueError('Existing-primary-label data audit must pass before training')
        self.audit_hash = sha(self.study/'data_audit.json')
        files = ['primary_category_queue.py', 'primary_category_study.py', 'primary_category_report.py',
                 'multilabel_heads.py', 'multilabel_study.py']
        if contract.is_intent(self.protocol) or self.protocol.get('experiment_scope') == 'two_source_full_primary_category':
            files.append('primary_category_contract.py')
        self.code_hashes = {name: sha(ROOT/'scripts/hanguard'/name) for name in files}
        registration = dict(protocol_sha256=self.protocol_sha,
                            data_audit_sha256=self.audit_hash, code_sha256=self.code_hashes)
        path = self.study/'execution_registration.json'
        if path.exists() and json.loads(path.read_text()) != registration:
            raise ValueError('A different execution is already registered in this study')
        dump(path, registration)
        snapshot = self.study/'code_snapshot'
        snapshot.mkdir(exist_ok=True)
        for name, digest in self.code_hashes.items():
            path = snapshot/name
            if path.exists() and sha(path) != digest:
                raise ValueError(f'Existing code snapshot changed: {name}')
            if not path.exists():
                shutil.copyfile(ROOT/'scripts/hanguard'/name, path)

        gpus = self.protocol['gpu_ids']
        if not gpus or len(set(gpus)) != len(gpus):
            raise ValueError('Distinct nonempty GPU assignments are required')
        tasks = [(arm, seed) for seed in self.protocol['seeds'] for arm in ARMS]
        self.status('preflight')
        for arm in ARMS:
            self.run('preflight', arm, gpus[0], self.protocol['seeds'][0])
        self.status('training', arms=list(ARMS), seeds=self.protocol['seeds'], runs=len(tasks), epochs=self.protocol['epochs'])
        def worker(gpu, assigned):
            for arm, seed in assigned:
                self.run('train', arm, gpu, seed)
        with ThreadPoolExecutor(max_workers=len(gpus)) as executor:
            jobs = [executor.submit(worker, gpu, tasks[index::len(gpus)]) for index, gpu in enumerate(gpus)]
            for job in jobs:
                job.result()

        self.check()
        selections = {}
        for arm, seed in tasks:
            path = self.study/'runs'/f'{arm}_s{seed}'/'selection.json'
            selection = json.loads(path.read_text())
            if selection['protocol_sha256'] != self.protocol_sha:
                raise ValueError('A head selection belongs to another protocol')
            if selection['checkpoint_sha256'] != sha(path.parent/'head.pt'):
                raise ValueError('A selected checkpoint changed before test')
            selections[str(path.relative_to(self.study))] = sha(path)
        dump(self.study/'test_barrier.json', dict(locked=True, protocol_sha256=self.protocol_sha,
             selection_sha256=selections, test_dataset_sha256=self.protocol['data_sha256']['test'],
             locked_at=datetime.now(timezone.utc).isoformat()))
        self.status('evaluating')
        for arm, seed in tasks:
            self.run('evaluate', arm, gpus[0], seed)
        self.status('reporting')
        self.run('report')
        self.status('complete', report=str(self.study/'report.md'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    queue = Queue(args.output)
    with (queue.study/'queue.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            queue.execute()
        except Exception as error:
            queue.status('failed', error_type=type(error).__name__, error=str(error))
            raise


if __name__ == '__main__':
    main()
