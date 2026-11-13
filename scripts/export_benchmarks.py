#!/usr/bin/env python3
"""Export full HumanEval and MBPP decontamination inputs from pinned parquet files."""
import argparse
import json
from pathlib import Path
import pyarrow.parquet as pq
from huggingface_hub import HfApi,hf_hub_download
if __package__:
    from .prepare_corpus import canonical,destination,filehash,write_manifest
else:
    from prepare_corpus import canonical,destination,filehash,write_manifest


def export(lock,output):
    cfg=json.loads(Path(lock).read_text())
    with destination(output) as out:
        metadata={}
        for key,source in cfg['benchmarks'].items():
            info=HfApi().dataset_info(source['dataset_repo'],revision=source['revision'])
            if info.sha!=source['revision']:raise ValueError('benchmark revision mismatch')
            files=sorted(s.rfilename for s in info.siblings if s.rfilename.startswith(source['prefix']) and s.rfilename.endswith('.parquet'))
            rows=[];assets=[]
            for name in files:
                path=Path(hf_hub_download(source['dataset_repo'],name,repo_type='dataset',revision=source['revision']))
                rows.extend(pq.read_table(path).to_pylist());assets.append({'path':name,'sha256':filehash(path)})
            if len(rows)!=source['records']:raise ValueError(f'{key}: expected complete benchmark set')
            rows.sort(key=lambda row:str(row['task_id']))
            path=out/(key+'.jsonl');path.write_text(''.join(canonical(row)+'\n' for row in rows),encoding='utf-8')
            metadata[key]={**source,'upstream_files':assets,'sha256':filehash(path),'path':path.name}
        write_manifest(out/'manifest.json',{'schema':'benchmark-exports/v1','benchmarks':metadata,'lock_sha256':filehash(lock),'implementation_sha256':filehash(__file__)})
    return Path(output)/'manifest.json'

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--lock',default='configs/training_sources.lock.json');parser.add_argument('--output',required=True)
    args=parser.parse_args();print(export(args.lock,args.output))
