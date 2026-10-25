"""Run the existing binary trainer with the remediated release data contract.

The historical repaired_study module is kept byte-for-byte unchanged so locked
experiments retain their source identity. Only this new entrypoint wraps its
loader with independent binary/type validation and completed-release QA.
"""
from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.hanguard import remediated_dataset_contract as contract
from scripts.hanguard import repaired_study as legacy

_legacy_load_split = legacy.load_split


def load_split(split, tokenizer, protocol):
    if split not in contract.SPLITS:
        raise ValueError(split)
    directory = Path(protocol['data_dir'])
    release = contract.release_contract(directory)
    if release.get('dataset_contract') != contract.CONTRACT:
        raise ValueError('The remediated binary entrypoint requires the explicit remediated release contract')
    for key in ('dataset_contract', 'dataset_version'):
        if protocol.get(key) != release[key]:
            raise ValueError(f'Registered {key} differs from the data release')
    frame = pd.read_parquet(directory / f'{split}.parquet')
    contract.label_arrays(frame, require_type_mask=True)
    return _legacy_load_split(split, tokenizer, protocol)


def main():
    previous = legacy.load_split
    legacy.load_split = load_split
    try:
        legacy.main()
    finally:
        legacy.load_split = previous


if __name__ == '__main__':
    main()
