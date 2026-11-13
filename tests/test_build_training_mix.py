import json
from pathlib import Path
import numpy as np
import pytest
from scripts.prepare_corpus import filehash,write_manifest,SPECIAL
from scripts.train_tokenizer import train
from scripts.build_training_mix import build


def fixture(tmp_path):
    data=tmp_path/'prepared';data.mkdir()
    for split in ('train','validation'):
        rows=[{'text':f'{split} {bucket} document {i} '+('some ordinary Unicode λ text ' * 4),'bucket':bucket,'source':{'fixture':True}} for bucket in ('code','web','math') for i in range(2)]
        (data/f'{split}.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
    write_manifest(data/'manifest.json',{'schema':'prepared-corpus/v1','smoke_only':False,'benchmarks':{'human_eval':{'records':164},'mbpp':{'records':974}},'outputs':{s:{'sha256':filehash(data/f'{s}.jsonl')} for s in ('train','validation')}})
    tokens=tmp_path/'tokenizer'
    train([data/'train.jsonl'],tokens,vocab_size=261,prepared=data/'manifest.json')
    return data,tokens


def test_exact_mixture_and_binary_hashes_and_reproducibility(tmp_path):
    data,tokens=fixture(tmp_path)
    for name in ('one','two'):
        build(data,tokens,tmp_path/name,train_tokens=337,validation_tokens=161,seq_len=8,shard_sequences=7)
    a=json.loads((tmp_path/'one/manifest.json').read_text())
    b=json.loads((tmp_path/'two/manifest.json').read_text())
    assert a==b
    assert not a['smoke_only'] and a['token_mix_enforced']
    assert a['splits']['train']['tokens']==320
    for split in ('train','validation'):
        info=a['splits'][split]
        assert info['bucket_fractions']=={'code':.5,'web':.35,'math':.15}
        assert sum(info['bucket_tokens'].values())==info['tokens']
        for shard in info['shards']:
            path=tmp_path/'one'/shard['path']
            ids=np.fromfile(path,dtype='<u2')
            assert len(ids)==shard['tokens']
            assert filehash(path)==shard['sha256']
            assert not any(i in (0,2,3,4) for i in ids)
        assert not (tmp_path/'one/.pool').exists()


def test_insufficient_unique_tokens_never_repeats_or_publishes(tmp_path):
    data,tokens=fixture(tmp_path)
    with pytest.raises(ValueError,match='insufficient unique'):
        build(data,tokens,tmp_path/'out',train_tokens=100000,validation_tokens=160,seq_len=8)
    assert not (tmp_path/'out').exists()


def test_bundle_audit_rejects_modified_shard(tmp_path):
    from scripts.audit_training_data import audit
    data,tokens=fixture(tmp_path)
    build(data,tokens,tmp_path/'bundle',train_tokens=160,validation_tokens=160,seq_len=8)
    assert audit(tmp_path/'bundle')['passed']
    shard=next((tmp_path/'bundle').glob('train-*.bin'))
    contents=shard.read_bytes();shard.write_bytes(b'\0\0'+contents[2:])
    with pytest.raises(ValueError,match='shard hash'):
        audit(tmp_path/'bundle')
