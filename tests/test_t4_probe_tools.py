"""CPU checks of probe evidence comparisons and real-byte subset construction."""
import json
import numpy as np
import pytest
import torch

from scripts.build_probe_bundle import build
from scripts.check_gdn_t4 import comparison, probe
from scripts.prepare_corpus import filehash, write_manifest
from scripts.train_diffusion import bundle_provenance, load_manifest


def corpus(root):
    root.mkdir()
    (root / 'tokenizer.json').write_text('{}')
    (root / 'tokenizer_manifest.json').write_text('{}')
    splits = {}
    for split in ('train', 'validation'):
        shards = []
        for index in range(2):
            path = root / f'{split}-{index}.bin'
            np.arange(5 + index * 16, 21 + index * 16, dtype='<u2').tofile(path)
            shards.append({'path': path.name, 'tokens': 16, 'sequences': 2,
                           'sha256': filehash(path)})
        splits[split] = {'shards': shards, 'tokens': 32, 'sequences': 4}
    write_manifest(root / 'manifest.json', {
        'schema': 'training-mixture/v1', 'dtype': '<u2', 'seq_len': 8,
        'special_ids': {'[MASK]': 0, '[EOS]': 1, '[PAD]': 2, '[UNK]': 3, '[BOS]': 4},
        'splits': splits, 'tokenizer_sha256': filehash(root / 'tokenizer.json'),
        'tokenizer_manifest_sha256': filehash(root / 'tokenizer_manifest.json')})
    return root


def test_probe_subset_crosses_shards_and_preserves_source(tmp_path):
    source = corpus(tmp_path / 'source')
    before = {p.name: p.read_bytes() for p in source.iterdir()}
    # A shorter context deliberately repacks the same token prefix.
    out = tmp_path / 'subset'
    derived = build(source, out, 6, 2, 'test subset', seq_len=4)
    assert (out / 'train.bin').read_bytes() == (before['train-0.bin'] + before['train-1.bin'])[:48]
    assert (out / 'validation.bin').read_bytes() == before['validation-0.bin'][:16]
    assert {p.name: p.read_bytes() for p in source.iterdir()} == before
    assert load_manifest(out, verify_hashes=True) == derived
    assert derived['derived_from'] == bundle_provenance(source, load_manifest(source))
    assert derived['derived_from']['seq_len'] == 8 and derived['seq_len'] == 4
    assert derived['smoke_only']
    with pytest.raises(FileExistsError):
        build(source, out, 1, 1, 'test')


@pytest.mark.parametrize('name', ['validation-1.bin', 'tokenizer.json'])
def test_probe_builder_rejects_corrupt_source_even_outside_subset(tmp_path, name):
    source = corpus(tmp_path / 'source')
    path = source / name
    data = bytearray(path.read_bytes()); data[0] ^= 1; path.write_bytes(data)
    with pytest.raises(ValueError, match='hash mismatch'):
        build(source, tmp_path / 'subset', 1, 1, 'test')
    assert not (tmp_path / 'subset').exists()


@pytest.mark.parametrize('count', [0, -1, 5])
def test_probe_builder_rejects_bad_counts(tmp_path, count):
    source = corpus(tmp_path / 'source')
    with pytest.raises(ValueError):
        build(source, tmp_path / 'subset', count, 1, 'test')
    assert not (tmp_path / 'subset').exists()


@pytest.mark.parametrize('side', [0, 1])
@pytest.mark.parametrize('value', [float('nan'), float('inf')])
def test_probe_comparison_rejects_nonfinite_reference_and_output(side, value):
    pair = [torch.ones(2), torch.ones(2)]
    pair[side][0] = value
    with pytest.raises(AssertionError, match='non-finite'):
        comparison(*pair, 0.01, 1e-6, 'test')


def test_probe_comparison_enforces_error_budget():
    with pytest.raises(AssertionError, match='relative_l2'):
        comparison(torch.ones(2), torch.ones(2) * 1.1, 0.01, 1e-6, 'test')


@pytest.mark.parametrize('caps', [{'wall_cap': 601}, {'wall_cap': 0},
                                  {'update_cap': 25}, {'forced_micro': 3}])
def test_probe_rejects_unbounded_or_invalid_settings(tmp_path, caps):
    with pytest.raises(ValueError):
        probe(tmp_path, **caps)


@pytest.mark.skipif(torch.cuda.is_available(), reason='CPU failure-mode check')
def test_p0_refuses_to_claim_gpu_validation_on_cpu(tmp_path, monkeypatch):
    from scripts.check_d1_p0 import main

    monkeypatch.setattr('sys.argv', ['check_d1_p0.py', '--phase', 'p0', '--root', str(tmp_path)])
    assert main() == 1
    report = json.loads((tmp_path / 'p0.json').read_text())
    assert report['status'] == 'FAIL' and 'real T4' in report['error']


def test_kaggle_package_is_private_bounded_and_excludes_unlisted_files(tmp_path):
    import ast
    import base64
    import io
    import zipfile
    from scripts.package_kaggle_probe import package
    source = corpus(tmp_path / 'source')
    for split in ('train', 'validation'):
        path = source / f'{split}-0.bin'
        np.arange(5, 1029, dtype='<u2').tofile(path)
    meta = json.loads((source / 'manifest.json').read_text())
    meta.pop('fingerprint')
    meta['seq_len'] = 1024
    for split in ('train', 'validation'):
        path = source / f'{split}-0.bin'
        meta['splits'][split] = {'shards': [{'path': path.name, 'tokens': 1024,
            'sequences': 1, 'sha256': filehash(path)}], 'tokens': 1024, 'sequences': 1}
    write_manifest(source / 'tokenizer_manifest.json', {'schema': 'tokenizer/v1',
        'vocab_size': 32000, 'special_ids': meta['special_ids']})
    meta['tokenizer_manifest_sha256'] = filehash(source / 'tokenizer_manifest.json')
    write_manifest(source / 'manifest.json', meta)
    (source / 'do-not-upload.secret').write_text('private neighboring file')
    out = package(source, tmp_path / 'package', 'example-owner')
    config = json.loads((out / 'kernel-metadata.json').read_text())
    assert config['is_private'] and config['machine_shape'] == 'NvidiaTeslaT4'
    tree = ast.parse((out / 'benchmark.py').read_text())
    call = next(n for n in ast.walk(tree) if isinstance(n, ast.Call)
                and isinstance(n.func, ast.Attribute) and n.func.attr == 'b64decode')
    archive = zipfile.ZipFile(io.BytesIO(base64.b64decode(ast.literal_eval(call.args[0]))))
    assert 'probe-bundle/do-not-upload.secret' not in archive.namelist()
    assert archive.read('probe-bundle/train-0.bin') == (source / 'train-0.bin').read_bytes()
    assert 'scripts/distributed_training.py' in archive.namelist()
    with pytest.raises(FileExistsError):
        package(source, out, 'example-owner')
    with pytest.raises(ValueError, match='owner'):
        package(source, tmp_path / 'invalid', '../wrong')
