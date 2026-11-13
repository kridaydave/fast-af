#!/usr/bin/env python3
"""Build an exact 50/35/15 token mixture as memory-mappable uint16 shards."""
from __future__ import annotations
import argparse
from collections import Counter
import json
from pathlib import Path
import random
import shutil
import numpy as np

if __package__:
    from .prepare_corpus import (BUCKETS, SPECIAL, canonical, destination, document_hash,
                                filehash, prepared_manifest, records, write_manifest, digest)
else:
    from prepare_corpus import (BUCKETS, SPECIAL, canonical, destination, document_hash,
                               filehash, prepared_manifest, records, write_manifest, digest)

WEIGHTS = {'code': 10, 'web': 7, 'math': 3}


def build(prepared, tokenizer_dir, output, train_tokens=100_000_000,
          validation_tokens=1_000_000, seq_len=1024, seed=17, shard_sequences=10240):
    from tokenizers import Tokenizer
    if any(type(v) is not int or v < 1 for v in (train_tokens, validation_tokens, seq_len, shard_sequences)) or seed < 0:
        raise ValueError('positive token budgets, seq_len, shard size and nonnegative seed required')
    prepared, tokenizer_dir = Path(prepared), Path(tokenizer_dir)
    corpus_path, token_path = prepared/'manifest.json', tokenizer_dir/'tokenizer.json'
    corpus = prepared_manifest(corpus_path, prepared/'train.jsonl', prepared/'validation.jsonl')
    if corpus['smoke_only']:
        raise ValueError('actual training mixture requires non-smoke prepared data')
    token_meta_path = tokenizer_dir/'config/manifest.json'
    token_meta = json.loads(token_meta_path.read_text())
    claimed=token_meta.pop('fingerprint',None)
    if claimed != digest(canonical(token_meta).encode()) or token_meta.get('smoke_only') is not False:
        raise ValueError('invalid or smoke-only tokenizer manifest')
    if token_meta.get('tokenizer_sha256') != filehash(token_path) or token_meta.get('prepared_manifest_sha256') != filehash(corpus_path):
        raise ValueError('tokenizer/corpus lineage mismatch')
    tok=Tokenizer.from_file(str(token_path))
    if tok.truncation is not None or tok.padding is not None or tok.get_vocab_size()>65536:
        raise ValueError('uint16 packing needs <=65536 vocabulary and no truncation/padding')
    if [tok.token_to_id(t) for t in SPECIAL] != list(range(5)):
        raise ValueError('special ID mismatch')
    rng=random.Random(seed)
    seen=set(); split_metadata={}
    with destination(output) as target:
        pool=target/'.pool';pool.mkdir()
        for split,requested in [('train',train_tokens),('validation',validation_tokens)]:
            cycles=requested//(20*seq_len)
            if not cycles: raise ValueError('budget must cover at least one 20-sequence mixture cycle')
            lengths=Counter();doc_counts=Counter()
            streams={b:(pool/f'{split}-{b}.bin').open('wb') for b in BUCKETS}
            doc_index=target/f'{split}-documents.jsonl'
            try:
                with doc_index.open('w',encoding='utf-8') as index:
                    for row in records(prepared/f'{split}.jsonl'):
                        text=row['text'];bucket=row['bucket'];h=document_hash(text)
                        if h in seen:raise ValueError('duplicate/cross-split document leakage')
                        seen.add(h)
                        if any(t in text for t in SPECIAL):raise ValueError('reserved token literal')
                        ids=tok.encode(text,add_special_tokens=False).ids
                        if any(i<5 for i in ids):raise ValueError('unexpected control/unknown token in text')
                        ids.append(1)
                        np.asarray(ids,dtype='<u2').tofile(streams[bucket])
                        index.write(canonical({'bucket':bucket,'start':lengths[bucket],'length':len(ids),
                                               'document_sha256':h,'source':row['source']})+'\n')
                        lengths[bucket]+=len(ids);doc_counts[bucket]+=1
            finally:
                for stream in streams.values():stream.close()
            quotas={b:cycles*w*seq_len for b,w in WEIGHTS.items()}
            shortages={b:{'available':lengths[b],'required':quotas[b]} for b in BUCKETS if lengths[b]<quotas[b]}
            if shortages:raise ValueError(f'insufficient unique source tokens (no repetition allowed): {shortages}')
            arrays={b:np.memmap(pool/f'{split}-{b}.bin',dtype='<u2',mode='r') for b in BUCKETS}
            positions=Counter();shards=[];current=None;shard_count=0;total_sequences=0
            index_path=target/f'{split}-sequences.jsonl'
            try:
                with index_path.open('w',encoding='utf-8') as index:
                    for _ in range(cycles):
                        order=[b for b,w in WEIGHTS.items() for _ in range(w)]
                        rng.shuffle(order)
                        for bucket in order:
                            if current is None or shard_count>=shard_sequences:
                                if current:current.close()
                                name=f'{split}-{len(shards):05d}.bin'
                                shards.append({'path':name,'sequences':0})
                                current=(target/name).open('wb');shard_count=0
                            start=positions[bucket]
                            arrays[bucket][start:start+seq_len].tofile(current)
                            index.write(canonical({'sequence':total_sequences,'bucket':bucket,'pool_offset':start})+'\n')
                            positions[bucket]+=seq_len;shard_count+=1;total_sequences+=1;shards[-1]['sequences']+=1
            finally:
                if current:current.close()
                arrays.clear()
            for shard in shards:
                shard['sha256']=filehash(target/shard['path'])
                shard['tokens']=shard['sequences']*seq_len
                if (target/shard['path']).stat().st_size != shard['tokens']*2:raise ValueError('shard size mismatch')
            included=Counter();partial=Counter()
            for row in records(doc_index):
                q=quotas[row['bucket']]
                if row['start']<q:
                    included[row['bucket']]+=1
                    if row['start']+row['length']>q:partial[row['bucket']]+=1
            total=sum(quotas.values())
            split_metadata[split]={'requested_tokens':requested,'tokens':total,'sequences':total_sequences,
                'bucket_tokens':quotas,'bucket_fractions':{b:quotas[b]/total for b in BUCKETS},
                'candidate_tokens':dict(lengths),'unused_candidate_tokens':{b:lengths[b]-quotas[b] for b in BUCKETS},
                'candidate_documents':dict(doc_counts),'included_documents':dict(included),'partially_used_documents':dict(partial),
                'shards':shards,'document_index':{'path':doc_index.name,'sha256':filehash(doc_index)},
                'sequence_index':{'path':index_path.name,'sha256':filehash(index_path)}}
            for bucket in BUCKETS:(pool/f'{split}-{bucket}.bin').unlink()
            print(split,total,'tokens',flush=True)
        pool.rmdir()
        for src,name in [(corpus_path,'prepared_manifest.json'),(token_path,'tokenizer.json'),(token_meta_path,'tokenizer_manifest.json')]:
            shutil.copyfile(src,target/name)
        evidence=[]
        if corpus.get('sources'):
            directory=target/'source-provenance';directory.mkdir()
            for index,source in enumerate(corpus['sources']):
                export=Path(source['export_manifest'])
                if filehash(export)!=source['export_manifest_sha256']:
                    raise ValueError('source export evidence changed')
                metadata=json.loads(export.read_text())
                for path,suffix,expected in [(export,'export.json',source['export_manifest_sha256']),
                    (export.parent/'SOURCE_README.md','README.md',metadata['source_card_sha256'])]:
                    name=f"{index}-{source['bucket']}-{suffix}"
                    if filehash(path)!=expected:raise ValueError('source attribution evidence changed')
                    shutil.copyfile(path,directory/name)
                    evidence.append({'path':str(Path('source-provenance')/name),'sha256':expected})
        write_manifest(target/'manifest.json',{'schema':'training-mixture/v1','smoke_only':False,
            'dtype':'<u2','seq_len':seq_len,'attention_mask':'all ones; construct in loader',
            'special_ids':dict(zip(SPECIAL,range(5))),'mask_corruption':'apply dynamically in training; stored tokens are clean targets',
            'seed':seed,'token_mix_enforced':True,'token_mix':{b:w/20 for b,w in WEIGHTS.items()},
            'budget_rounding':'floor to whole 20-sequence cycles','oversampling':False,
            'tokenizer_sha256':filehash(token_path),'prepared_manifest_sha256':filehash(corpus_path),
            'tokenizer_manifest_sha256':filehash(token_meta_path),'implementation_sha256':filehash(__file__),
            'source_evidence':evidence,
            'splits':split_metadata})
    return Path(output)/'manifest.json'


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--prepared',required=True);p.add_argument('--tokenizer-dir',required=True);p.add_argument('--output',required=True)
    p.add_argument('--train-tokens',type=int,default=100_000_000);p.add_argument('--validation-tokens',type=int,default=1_000_000)
    p.add_argument('--seq-len',type=int,default=1024);p.add_argument('--seed',type=int,default=17)
    a=p.parse_args()
    try:print(build(a.prepared,a.tokenizer_dir,a.output,a.train_tokens,a.validation_tokens,a.seq_len,a.seed))
    except (ValueError,OSError) as error:p.exit(2,f'error: {error}\n')
if __name__=='__main__':main()
