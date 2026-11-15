"""CPU checks of probe evidence comparisons and real-byte subset construction."""
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
