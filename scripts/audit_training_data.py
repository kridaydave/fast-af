#!/usr/bin/env python3
"""Verify training bundle bytes, token quotas, document splits, and residual overlaps."""
from __future__ import annotations
import argparse
from collections import Counter
import json
from pathlib import Path
import numpy as np
if __package__:
    from .prepare_corpus import (benchmark_index,canonical,digest,document_hash,filehash,norm,records,split_for,SubstringIndex,grams,write_manifest)
    from .near_dedup import NearDedup
else:
    from prepare_corpus import (benchmark_index,canonical,digest,document_hash,filehash,norm,records,split_for,SubstringIndex,grams,write_manifest)
    from near_dedup import NearDedup


def checked(path,schema):
    doc=json.loads(Path(path).read_text())
    if not isinstance(doc,dict) or doc.get('schema')!=schema:raise ValueError('invalid manifest schema')
    signature=doc.pop('fingerprint',None)
    if signature!=digest(canonical(doc).encode()):raise ValueError('manifest fingerprint mismatch')
    return doc


def audit(bundle,prepared=None,source_config=None,corpus_audit=None):
    from tokenizers import Tokenizer
    bundle=Path(bundle);meta=checked(bundle/'manifest.json','training-mixture/v1')
    if meta.get('smoke_only') is not False or meta.get('token_mix_enforced') is not True:raise ValueError('bundle is not an actual enforced training mixture')
    for filename,key in [('tokenizer.json','tokenizer_sha256'),('tokenizer_manifest.json','tokenizer_manifest_sha256'),('prepared_manifest.json','prepared_manifest_sha256')]:
        if filehash(bundle/filename)!=meta[key]:raise ValueError('bundle lineage hash mismatch: '+filename)
    corpus=checked(bundle/'prepared_manifest.json','prepared-corpus/v1')
    tokenizer=checked(bundle/'tokenizer_manifest.json','tokenizer/v1')
    if corpus['smoke_only'] or tokenizer['smoke_only']:raise ValueError('smoke lineage')
    if tokenizer['prepared_manifest_sha256']!=meta['prepared_manifest_sha256'] or tokenizer['tokenizer_sha256']!=meta['tokenizer_sha256']:raise ValueError('tokenizer lineage mismatch')
    runtime_tokenizer=Tokenizer.from_file(str(bundle/'tokenizer.json'))
    if runtime_tokenizer.get_vocab_size()!=tokenizer['vocab_size']:
        raise ValueError('tokenizer vocabulary count mismatch')
    if {k:v['records'] for k,v in corpus['benchmarks'].items()}!={'human_eval':164,'mbpp':974}:raise ValueError('incomplete benchmark provenance')
    result={'schema':'training-data-audit/v1','manifest_sha256':filehash(bundle/'manifest.json'),'implementation_sha256':filehash(__file__),'splits':{}}
    for evidence in meta.get('source_evidence',[]):
        if filehash(bundle/evidence['path'])!=evidence['sha256']:
            raise ValueError('source attribution evidence hash mismatch')
    for split,info in meta['splits'].items():
        count=0
        for shard in info['shards']:
            path=bundle/shard['path']
            if filehash(path)!=shard['sha256'] or path.stat().st_size!=shard['tokens']*2:raise ValueError('shard hash/size mismatch')
            tokens=np.memmap(path,dtype='<u2',mode='r')
            if np.any(tokens>=tokenizer['vocab_size']) or np.any(np.isin(tokens,[0,2,3,4])):raise ValueError('invalid/control token in shard')
            if len(tokens)%meta['seq_len']:raise ValueError('non-full sequence')
            count+=len(tokens)
            del tokens
        if count!=info['tokens'] or sum(info['bucket_tokens'].values())!=count:raise ValueError('token accounting mismatch')
        for bucket,weight in [('code',10),('web',7),('math',3)]:
            if info['bucket_tokens'][bucket]*20!=count*weight:raise ValueError('incorrect token mixture')
        idx=info['sequence_index']
        if filehash(bundle/idx['path'])!=idx['sha256']:raise ValueError('sequence index hash mismatch')
        offsets=Counter();sequence=0
        for row in records(bundle/idx['path']):
            if row['sequence']!=sequence or row['pool_offset']!=offsets[row['bucket']]:raise ValueError('invalid sequence provenance offsets')
            offsets[row['bucket']]+=meta['seq_len'];sequence+=1
        if dict(offsets)!=info['bucket_tokens'] or sequence!=info['sequences']:raise ValueError('sequence provenance/quota mismatch')
        idx=info['document_index']
        if filehash(bundle/idx['path'])!=idx['sha256']:raise ValueError('document index hash mismatch')
        result['splits'][split]={'tokens':count,'sequences':sequence,'bucket_tokens':info['bucket_tokens'],'hashes_verified':True}
    if prepared is not None:
        if filehash(Path(prepared)/'manifest.json')!=meta['prepared_manifest_sha256']:
            raise ValueError('prepared manifest changed since build')
        for split in ('train','validation'):
            if filehash(Path(prepared)/(split+'.jsonl'))!=corpus['outputs'][split]['sha256']:
                raise ValueError('prepared corpus changed since audit/build')
        if corpus_audit is not None:
            proof=checked(corpus_audit,'prepared-data-audit/v1')
            if proof.get('passed') is not True or proof['prepared_manifest_sha256']!=meta['prepared_manifest_sha256']:
                raise ValueError('corpus audit lineage mismatch')
            if source_config is None or proof['source_config_sha256']!=filehash(source_config):
                raise ValueError('corpus audit source config mismatch')
            result['corpus_audit_sha256']=filehash(corpus_audit)
        else:
            proof=audit_documents(prepared,source_config)
        for key in ('documents_audited','detected_benchmark_overlaps','detected_exact_or_near_duplicates','scope'):
            result[key]=proof[key]
        path=Path(prepared)/'validation.jsonl'
        if filehash(path)!=corpus['outputs']['validation']['sha256']:raise ValueError('validation byte mismatch')
        validation_documents=0
        for row in records(path):
            ids=runtime_tokenizer.encode(row['text'],add_special_tokens=False).ids
            if 3 in ids or runtime_tokenizer.decode(ids,skip_special_tokens=False)!=row['text']:
                raise ValueError('held-out tokenizer coverage/roundtrip failure')
            validation_documents+=1
        result['validation_tokenizer_documents_checked']=validation_documents
        result['validation_unknown_tokens']=0
        result['validation_roundtrip_failures']=0
    result['passed']=True
    return result

def audit_documents(prepared,source_config):
    if source_config is None:raise ValueError('source config required for residual audit')
    prepared=Path(prepared);corpus=checked(prepared/'manifest.json','prepared-corpus/v1')
    if corpus.get('smoke_only') is not False or corpus['config_sha256']!=filehash(source_config):
        raise ValueError('smoke corpus or source config mismatch')
    config=json.loads(Path(source_config).read_text())
    bg,fragments,_=benchmark_index(config,Path(source_config).parent,False)
    matcher=SubstringIndex(fragments);seen=set();near=NearDedup();documents=0
    for split in ('train','validation'):
        path=prepared/(split+'.jsonl')
        if filehash(path)!=corpus['outputs'][split]['sha256']:raise ValueError('prepared corpus byte mismatch')
        for row in records(path):
            normalized=norm(row['text']);h=document_hash(row['text'])
            if h in seen:raise ValueError('duplicate/residual split leakage')
            seen.add(h)
            if split_for(h,corpus['seed'],corpus['validation_fraction'])!=split:raise ValueError('hash split mismatch')
            if grams(row['text'])&bg or matcher.matches(normalized):raise ValueError('residual detected benchmark overlap')
            if near.duplicate(normalized):raise ValueError('residual near duplicate')
            documents+=1
            if documents%20000==0:print('audited',documents,'documents',flush=True)
    return {'schema':'prepared-data-audit/v1','passed':True,
            'prepared_manifest_sha256':filehash(prepared/'manifest.json'),
            'source_config_sha256':filehash(source_config),'implementation_sha256':filehash(__file__),
            'documents_audited':documents,'detected_benchmark_overlaps':0,'detected_exact_or_near_duplicates':0,
            'scope':'full prepared candidate corpus; 13-word ngrams, normalized substrings >=32 characters, conservative 64-bit SimHash'}


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--bundle');p.add_argument('--prepared');p.add_argument('--source-config');p.add_argument('--out',required=True)
    p.add_argument('--documents-only',action='store_true');p.add_argument('--corpus-audit')
    a=p.parse_args()
    try:
        if a.documents_only:
            if not a.prepared:p.error('--documents-only requires --prepared')
            result=audit_documents(a.prepared,a.source_config)
        else:
            if not a.bundle:p.error('--bundle is required')
            result=audit(a.bundle,a.prepared,a.source_config,a.corpus_audit)
        write_manifest(Path(a.out),result);print(json.dumps(result,indent=2))
    except (ValueError,OSError) as error:p.exit(2,f'error: {error}\n')
