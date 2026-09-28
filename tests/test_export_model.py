import json

import pytest

from scripts.hanguard.export_model import export, sha


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


@pytest.fixture
def artifacts(tmp_path):
    base = tmp_path / 'models' / 'base'
    write(base / 'config.json', {'hidden_size': 4})
    write(base / 'tokenizer.json', {'test': True})
    binary_study = tmp_path / 'outputs' / 'binary'
    binary = binary_study / 'runs' / 'E04_s42'
    binary.mkdir(parents=True)
    (binary / 'head.pt').write_bytes(b'binary head')
    (binary / 'adapter.pt').write_bytes(b'adapter')
    binary_protocol = dict(model_path=str(base), max_tokens=4096, head_width=128,
        layers=[8, 16, 24, 32], lora_rank=8, lora_alpha=16, lora_dropout=.05,
        model_config_sha256=sha(base / 'config.json'), tokenizer_sha256=sha(base / 'tokenizer.json'))
    write(binary_study / 'protocol.json', binary_protocol)
    write(binary_study / 'arms.json', {'E04': dict(lora=True, mode='fusion', readout='mlp')})
    hashes = {name: sha(binary / name) for name in ('head.pt', 'adapter.pt')}
    write(binary / 'selection.json', dict(arm='E04', threshold=.51, checkpoint_hashes=hashes,
        protocol_sha256=sha(binary_study / 'protocol.json'), arms_sha256=sha(binary_study / 'arms.json')))
    category = tmp_path / 'outputs' / 'category'
    write(category / 'protocol.json', dict(parent_run=str(binary), seeds=[42, 43, 44],
        max_tokens=4096, pad_multiple=32, token_budget=16384, dropout=.1))
    for seed, ce in [(42, .52), (43, .47), (44, .54)]:
        run = category / 'runs' / f'learned_queries_s{seed}'
        run.mkdir(parents=True)
        (run / 'head.pt').write_bytes(str(seed).encode())
        write(run / 'selection.json', dict(arm='learned_queries', seed=seed, validation_ce=ce,
            head_width=128, checkpoint_sha256=sha(run / 'head.pt'),
            protocol_sha256=sha(category / 'protocol.json'),
            binary_parent=dict(checkpoint_hashes=hashes, binary_threshold=.51,
                selection_sha256=sha(binary / 'selection.json'),
                protocol_sha256=sha(binary_study / 'protocol.json'))))
        # Test ranking deliberately contradicts validation ranking.
        write(run / 'test_results.json', {'accuracy': 1.0 if seed == 44 else 0.1})
    return binary, category, tmp_path / 'models' / 'hanguard'


def test_selects_validation_and_copies_verified_weights(artifacts):
    binary, category, output = artifacts
    manifest = export(binary, category, output)
    assert manifest['category']['seed'] == 43
    assert manifest['provenance']['seed_selection_uses_test'] is False
    assert (output / 'category_head.pt').read_bytes() == b'43'
    assert (output / manifest['base_model']).resolve() == output.parent / 'base'
    assert sha(output / 'binary_adapter.pt') == sha(binary / 'adapter.pt')
    with pytest.raises(ValueError, match='already exists'):
        export(binary, category, output)


def test_rejects_modified_checkpoint_before_export(artifacts):
    binary, category, output = artifacts
    (category / 'runs' / 'learned_queries_s42' / 'head.pt').write_bytes(b'changed')
    with pytest.raises(ValueError, match='hash mismatch'):
        export(binary, category, output)
    assert not output.exists()


def test_rejects_category_from_another_parent(artifacts):
    binary, category, output = artifacts
    selection = category / 'runs' / 'learned_queries_s43' / 'selection.json'
    value = json.loads(selection.read_text())
    value['binary_parent']['binary_threshold'] = .9
    write(selection, value)
    with pytest.raises(ValueError, match='provenance disagree'):
        export(binary, category, output)
    assert not output.exists()
