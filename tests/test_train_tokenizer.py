import importlib.util
import json
from pathlib import Path

import pytest

from scripts import train_tokenizer as m

def fixture(path):
    rows=[{"text":s,"bucket":b} for b,s in [("code","def f():\n\treturn 'λ'\n"),("web","Hello 世界\n"),("math","x = α + β\n")]*12]
    path.write_text("".join(json.dumps(x,ensure_ascii=False)+"\n" for x in rows),encoding="utf-8")

def test_artifacts_specials_and_roundtrip(tmp_path):
    src=tmp_path/"train.jsonl"; fixture(src)
    a,_=m.train([src],tmp_path/"a",280,seed=9); b,_=m.train([src],tmp_path/"b",280,seed=9)
    assert (tmp_path/"a/tokenizer.json").read_bytes()==(tmp_path/"b/tokenizer.json").read_bytes()
    for i,t in enumerate(m.SPECIAL): assert a.token_to_id(t)==i
    for s in ["a\t  b\n", "λ 世界", "def f():\n    return 1"]: assert a.decode(a.encode(s).ids)==s

def test_invalid_jsonl(tmp_path):
    p=tmp_path/"bad"; p.write_text('{"text":1,"bucket":"web"}\n')
    with pytest.raises(ValueError): m.load([p])
    p.write_text('{bad}\n')
    with pytest.raises(ValueError): m.load([p])
    with pytest.raises(ValueError): m.load([tmp_path/"absent"])

@pytest.mark.parametrize('kwargs', [{'max_docs': 0}, {'max_bytes': -1}, {'vocab_size': 260}, {'seed': -1}, {'vocab_size': 32000}])
def test_invalid_limits_and_insufficient_sample_do_not_publish(tmp_path, kwargs):
    src = tmp_path / 'train.jsonl'
    fixture(src)
    with pytest.raises(ValueError):
        m.train([src], tmp_path / 'out', **kwargs)
    assert not (tmp_path / 'out').exists()


def test_manifest_and_reports_are_reproducible(tmp_path):
    src = tmp_path / 'train.jsonl'
    fixture(src)
    _, a = m.train([src], tmp_path / 'a', 280, seed=9)
    _, b = m.train([src], tmp_path / 'b', 280, seed=9)
    assert a == b
    assert a['smoke_only']
    assert all(not v['roundtrip_failures'] and not v['unknown_tokens'] for v in a['sample_report'].values())
    with pytest.raises(ValueError, match='already exists'):
        m.train([src], tmp_path / 'a', 280)


def test_full_prepare_tokenizer_pack_workflow(tmp_path):
    from scripts import prepare_corpus as prep
    from types import SimpleNamespace
    rows = [{'text': f"def function_{i}(value):\n    return value + {i}  # λ 世界\n", 'bucket': 'code'} for i in range(100)]
    raw = tmp_path / 'raw.jsonl'
    raw.write_text(''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in rows))
    cfg = tmp_path / 'sources.json'
    cfg.write_text(json.dumps({'sources': [{'bucket': 'code', 'dataset_repo': 'fixture', 'revision': 'unresolved', 'license': 'fixture', 'source_file': raw.name}]}))
    prepared = tmp_path / 'prepared'
    prep.prepare(cfg, prepared, limit=100, smoke=True, validation_fraction=0.2)
    tokenizer_dir = tmp_path / 'tokenizer'
    tok, meta = m.train([prepared / 'train.jsonl'], tokenizer_dir, 280, prepared=prepared / 'manifest.json')
    assert meta['prepared_manifest_sha256'] == prep.filehash(prepared / 'manifest.json')
    assert tok.get_vocab_size() == 280
    args = SimpleNamespace(tokenizer=tokenizer_dir / 'tokenizer.json', train=prepared / 'train.jsonl',
                           validation=prepared / 'validation.jsonl', seq_len=32, output=tmp_path / 'packed',
                           prepared_manifest=prepared / 'manifest.json', tokenizer_manifest=tokenizer_dir / 'config/manifest.json')
    prep.pack(args)
    packed = json.loads((args.output / 'manifest.json').read_text())
    assert packed['smoke_only'] and packed['splits']['train']['sequences'] > 0
    assert packed['splits']['validation']['sequences'] > 0
    # The provenance check stops accidental tokenizer training on validation.
    with pytest.raises(ValueError, match='train file does not match'):
        m.train([prepared / 'validation.jsonl'], tmp_path / 'leak', 280, prepared=prepared / 'manifest.json')
    args.tokenizer.write_text(args.tokenizer.read_text() + '\n')
    args.output = tmp_path / 'tampered'
    with pytest.raises(ValueError, match='tokenizer manifest/hash mismatch'):
        prep.pack(args)
    assert not args.output.exists()
