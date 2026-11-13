"""Offline corpus reproducibility and leakage/accounting regressions."""
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from tokenizers import Tokenizer, models, pre_tokenizers

from scripts import prepare_corpus as prep


def jsonl(path, rows):
    path.write_text(''.join(json.dumps(row) + '\n' for row in rows))
    return path


def config(tmp_path, texts, benchmarks=None):
    source = jsonl(tmp_path / 'input.jsonl', [{'text': t} for t in texts])
    cfg = {'sources': [{'bucket': 'web', 'dataset_repo': 'test/repo', 'revision': 'a' * 40,
                        'license': 'test-license', 'source_file': source.name}]}
    if benchmarks is not None:
        cfg['benchmarks'] = benchmarks
    path = tmp_path / 'config.json'
    path.write_text(json.dumps(cfg))
    return path


def test_reproducible_sampling_dedup_hashes_and_immutable_split(tmp_path):
    texts = [f'document {i} has useful educational text' for i in range(100)]
    cfg = config(tmp_path, texts + [texts[0].upper()])
    for output in ('a', 'b'):
        prep.prepare(cfg, tmp_path / output, seed=42, limit=30, smoke=True, validation_fraction=0.3)
    for name in ('train.jsonl', 'validation.jsonl', 'manifest.json'):
        assert (tmp_path / 'a' / name).read_bytes() == (tmp_path / 'b' / name).read_bytes()
    meta = json.loads((tmp_path / 'a/manifest.json').read_text())
    assert meta['counts']['duplicate'] == 1
    assert meta['counts']['selected'] == 30
    assert meta['counts']['sampled_out'] == 70
    for split in ('train', 'validation'):
        path = tmp_path / 'a' / (split + '.jsonl')
        assert meta['outputs'][split]['sha256'] == prep.filehash(path)
        for row in prep.records(path):
            assert prep.split_for(prep.document_hash(row['text']), 42, 0.3) == split
    # Adding a document cannot move an existing document to another split.
    assignments = {prep.document_hash(t): prep.split_for(prep.document_hash(t), 42, .3) for t in texts}
    assert all(prep.split_for(h, 42, .3) == split for h, split in assignments.items())
    with pytest.raises(ValueError, match='already exists'):
        prep.prepare(cfg, tmp_path / 'a', smoke=True)


def test_humaneval_and_mbpp_fields_and_normalized_substring(tmp_path):
    human = ' '.join(f'word{i}' for i in range(20))
    mbpp = 'ASSERT very_long_python_function_name(value) == expected_result'
    jsonl(tmp_path / 'he.jsonl', [{'prompt': human, 'canonical_solution': '', 'test': ''}])
    jsonl(tmp_path / 'mbpp.jsonl', [{'text': 'task', 'code': '', 'test_list': [mbpp]}])
    cfg = config(tmp_path, [human, 'prefix ' + mbpp.lower() + ' suffix', 'unrelated clean prose'],
                 {'human_eval': 'he.jsonl', 'mbpp': 'mbpp.jsonl'})
    prep.prepare(cfg, tmp_path / 'out', smoke=True)
    meta = json.loads((tmp_path / 'out/manifest.json').read_text())
    assert meta['counts']['benchmark_ngram'] == 1
    assert meta['counts']['benchmark_substring'] == 1
    assert meta['counts']['selected'] == 1
    assert set(meta['benchmarks']) == {'human_eval', 'mbpp'}
    assert meta['smoke_only']


def test_fail_closed_benchmarks_and_bad_json(tmp_path):
    cfg = config(tmp_path, ['text'])
    with pytest.raises(ValueError, match='both human_eval and mbpp'):
        prep.prepare(cfg, tmp_path / 'out')
    jsonl(tmp_path / 'he.jsonl', [{'unknown': 'silently skipped before'}])
    cfg = config(tmp_path, ['text'], {'human_eval': 'he.jsonl', 'mbpp': 'he.jsonl'})
    with pytest.raises(ValueError, match='no supported'):
        prep.prepare(cfg, tmp_path / 'out', smoke=True)
    (tmp_path / 'input.jsonl').write_text('null\n')
    with pytest.raises(ValueError, match='expected JSON object'):
        list(prep.records(tmp_path / 'input.jsonl'))


def tokenizer(tmp_path):
    tok = Tokenizer(models.WordLevel(vocab={**{token: i for i, token in enumerate(prep.SPECIAL)},
                                          'alpha': 5, 'beta': 6, 'gamma': 7, 'delta': 8}, unk_token='[UNK]'))
    tok.pre_tokenizer = pre_tokenizers.Whitespace()
    path = tmp_path / 'tokenizer.json'
    tok.save(str(path))
    return path


def test_packing_eos_tail_accounting_and_cross_split_leaks(tmp_path):
    tok = tokenizer(tmp_path)
    train = jsonl(tmp_path / 'train.jsonl', [{'text': 'alpha beta', 'bucket': 'code'}, {'text': 'beta gamma', 'bucket': 'web'}])
    val = jsonl(tmp_path / 'val.jsonl', [{'text': 'gamma delta', 'bucket': 'math'}])
    args = SimpleNamespace(tokenizer=tok, train=train, validation=val, output=tmp_path / 'packed', seq_len=4)
    prep.pack(args)
    rows = list(prep.records(args.output / 'train.jsonl'))
    assert rows == [{'input_ids': [5, 6, 1, 6], 'attention_mask': [1] * 4}]
    meta = json.loads((args.output / 'manifest.json').read_text())
    for split in ('train', 'validation'):
        counts = meta['splits'][split]
        assert counts['encoded_tokens_including_eos'] == counts['packed_tokens'] + counts['tail_dropped']
        assert counts['shard_sha256'] == prep.filehash(args.output / (split + '.jsonl'))
    jsonl(val, [{'text': 'ALPHA   BETA', 'bucket': 'math'}])
    args.output = tmp_path / 'bad'
    with pytest.raises(ValueError, match='duplicate/leaked'):
        prep.pack(args)
    assert not args.output.exists()  # no partial publish or stale manifest
    args.seq_len = 0
    with pytest.raises(ValueError, match='positive'):
        prep.pack(args)


def test_benchmark_malformed_fields_and_suite_without_signatures_fail_closed(tmp_path):
    human = ' '.join(f'word{i}' for i in range(20))
    jsonl(tmp_path / 'he.jsonl', [{'prompt': human}])
    jsonl(tmp_path / 'mbpp.jsonl', [{'text': 'task', 'code': 123, 'test_list': 'not a list'}])
    cfg = config(tmp_path, ['ordinary prose'], {'human_eval':'he.jsonl', 'mbpp':'mbpp.jsonl'})
    with pytest.raises(ValueError, match='malformed'):
        prep.prepare(cfg, tmp_path / 'bad', smoke=True)
    jsonl(tmp_path / 'mbpp.jsonl', [{'text': 'task', 'code': '', 'test_list': []}])
    with pytest.raises(ValueError, match='mbpp: benchmark has no usable'):
        prep.prepare(cfg, tmp_path / 'bad', smoke=True)


def test_external_tokenizer_truncation_and_reserved_literals(tmp_path):
    path=tokenizer(tmp_path)
    tok=Tokenizer.from_file(str(path));tok.enable_truncation(max_length=2);tok.save(str(path))
    train=jsonl(tmp_path/'train.jsonl',[{'text':'alpha beta gamma delta','bucket':'code'}])
    val=jsonl(tmp_path/'val.jsonl',[{'text':'beta gamma','bucket':'web'}])
    args=SimpleNamespace(tokenizer=path,train=train,validation=val,seq_len=2,output=tmp_path/'out')
    with pytest.raises(ValueError,match='truncation'):
        prep.pack(args)
    tok.no_truncation();tok.save(str(path))
    jsonl(train,[{'text':'alpha [MASK] beta','bucket':'code'}])
    with pytest.raises(ValueError,match='reserved special-token literal'):
        prep.pack(args)


def test_near_duplicate_signature_is_order_stable():
    from scripts.near_dedup import NearDedup,signature
    text=' '.join(f'word{i}' for i in range(500))
    a=signature(text)
    assert a == signature(text)
    seen=NearDedup()
    assert not seen.duplicate(text)
    assert seen.duplicate(text+' final')
    assert not seen.duplicate('short text')


def test_production_benchmarks_require_complete_distinct_task_ids(tmp_path):
    long_text=' '.join(f'word{i}' for i in range(20))
    jsonl(tmp_path/'he.jsonl',[{'task_id':f'HumanEval/{i}','prompt':long_text} for i in range(164)])
    jsonl(tmp_path/'mbpp.jsonl',[{'task_id':i,'text':long_text,'code':'','test_list':[]} for i in range(1,975)])
    cfg={'benchmarks':{name:{'path':file,'dataset_repo':'fixture','revision':'a'*40} for name,file in [('human_eval','he.jsonl'),('mbpp','mbpp.jsonl')]}}
    _,_,meta=prep.benchmark_index(cfg,tmp_path,False)
    assert meta['human_eval']['records']==164 and meta['mbpp']['records']==974
    jsonl(tmp_path/'mbpp.jsonl',[{'task_id':1,'text':long_text}])
    with pytest.raises(ValueError,match='incomplete benchmark'):
        prep.benchmark_index(cfg,tmp_path,False)


def test_export_provenance_rejects_changed_source_bytes(tmp_path):
    raw=jsonl(tmp_path/'raw.jsonl',[{'text':'source'}])
    src={'bucket':'web','dataset_repo':'fixture','revision':'a'*40,'license':'fixture','source_file':raw.name,'export_manifest':'export.json'}
    prep.write_manifest(tmp_path/'export.json',{'schema':'source-export/v1','source':src,'documents_sha256':prep.filehash(raw)})
    assert prep.verify_source_export(src,tmp_path)==prep.filehash(tmp_path/'export.json')
    raw.write_text('{}\n')
    with pytest.raises(ValueError,match='document hash'):
        prep.verify_source_export(src,tmp_path)
