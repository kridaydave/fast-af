#!/usr/bin/env python3
"""Export a bounded corpus from immutable HF parquet sources, retaining provenance."""
from __future__ import annotations
import argparse
import json
import os
import sys
import traceback
from importlib.metadata import version
from pathlib import Path
import re

if __package__:
    from .prepare_corpus import canonical, destination, filehash, write_manifest
else:
    from prepare_corpus import canonical, destination, filehash, write_manifest


def export_source(lock, bucket, output, token=None):
    from datasets import load_dataset
    from huggingface_hub import HfApi, hf_hub_url, hf_hub_download
    config = json.loads(Path(lock).read_text())
    source = next(s for s in config['sources'] if s['bucket'] == bucket)
    if not re.fullmatch(r'[0-9a-f]{40}', source['revision']):
        raise ValueError('immutable source revision required')
    api = HfApi(token=token)
    info = api.dataset_info(source['dataset_repo'], revision=source['revision'], files_metadata=True)
    if info.sha != source['revision']:
        raise ValueError('source revision did not resolve exactly')
    files = sorted(s.rfilename for s in info.siblings if s.rfilename.startswith(source['prefix']) and s.rfilename.endswith('.parquet'))[:source['max_shards']]
    if not files:
        raise ValueError('no pinned parquet files found')
    seen_files, counts = [], {'read':0,'exported':0,'text_bytes':0,'length_filtered':0}
    with destination(output) as out:
        card = Path(hf_hub_download(source['dataset_repo'], 'README.md', repo_type='dataset', revision=source['revision'], token=token))
        (out/'SOURCE_README.md').write_bytes(card.read_bytes())
        with (out/'documents.jsonl').open('w',encoding='utf-8') as stream:
            for filename in files:
                url=hf_hub_url(source['dataset_repo'],filename,repo_type='dataset',revision=source['revision'])
                dataset=load_dataset('parquet',data_files={'train':[url]},split='train',streaming=True,token=token)
                seen_files.append(filename)
                for row_index,row in enumerate(dataset):
                    counts['read']+=1
                    text=row[source['text_field']]
                    if not isinstance(text,str): raise ValueError('source has non-string text')
                    size=len(text.encode('utf-8'))
                    if size<200 or size>200000:
                        counts['length_filtered']+=1
                        continue
                    # Retain source attribution fields, without duplicating content.
                    source_record={k:v for k,v in row.items() if k!=source['text_field']}
                    stream.write(canonical({'text':text,'source_record':source_record,
                                            'upstream_file':filename,'upstream_row':row_index})+'\n')
                    counts['exported']+=1;counts['text_bytes']+=size
                    if counts['exported']%5000==0:
                        print(bucket,counts['exported'],counts['text_bytes'],flush=True)
                    if counts['text_bytes']>=source['min_text_bytes']:break
                if counts['text_bytes']>=source['min_text_bytes']:break
        if counts['text_bytes']<source['min_text_bytes']:
            raise ValueError(f'insufficient source shards: {counts}')
        write_manifest(out/'export.json',{'schema':'source-export/v1','source':source,'upstream_files':seen_files,
                      'selection':'ordered pinned shard prefix; stop at min_text_bytes; retain 200..200000 UTF-8 bytes per document',
                      'counts':counts,'documents_sha256':filehash(out/'documents.jsonl'),
                      'source_card_sha256':filehash(out/'SOURCE_README.md'),
                      'lock_sha256':filehash(lock),'implementation_sha256':filehash(__file__),
                      'packages':{name:version(name) for name in ('datasets','huggingface_hub','pyarrow')}})
    print(output,flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--lock',default='configs/training_sources.lock.json')
    p.add_argument('--bucket',choices=('code','web','math'),required=True)
    p.add_argument('--output',required=True)
    p.add_argument('--token-file',help='secure local HF token file; value is never logged')
    a=p.parse_args()
    token=Path(a.token_file).read_text().strip() if a.token_file else None
    export_source(a.lock,a.bucket,a.output,token)

if __name__=='__main__':
    status=0
    try:
        main()
    except SystemExit as error:
        status=error.code or 0
    except KeyboardInterrupt:
        status=130
    except Exception:
        traceback.print_exc();status=1
    finally:
        # datasets streaming workers can hang during interpreter shutdown.
        # main has already finalized or discarded its atomic output directory.
        sys.stdout.flush();sys.stderr.flush();os._exit(status)
