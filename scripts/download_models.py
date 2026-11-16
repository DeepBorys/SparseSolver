#!/usr/bin/env python3
# SparseSolver — Copyright (C) 2026 Jerzy Slaby (DeepBorys).
# Uses the postflop-solver engine by Wataru Inariba; see NOTICE.
# SPDX-License-Identifier: AGPL-3.0-or-later
# Modified for SparseSolver inference and benchmarks; see NOTICE and LICENSE.

"""Install the single versioned eight-flop inference archive, with SHA-256 checks."""
import argparse,hashlib,json,os,shutil,tarfile,tempfile,urllib.request,urllib.error
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]

def digest(path):
 h=hashlib.sha256()
 with path.open('rb')as f:
  for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
 return h.hexdigest()

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--archive',type=Path);p.add_argument('--url');a=p.parse_args()
 spec=json.loads((ROOT/'models/ALL_ASSETS.json').read_text());dest=ROOT/'.model-assets'
 if (dest/'MODEL_MANIFEST.json').is_file():
  if digest(dest/'MODEL_MANIFEST.json')!=spec['inner_manifest_sha256']:raise SystemExit('Installed model manifest differs from the pinned version.')
  if a.archive and (not a.archive.is_file()or a.archive.stat().st_size!=spec['bytes']or digest(a.archive)!=spec['sha256']):raise SystemExit('Archive size or SHA-256 mismatch.')
  print('Eight-flop model archive already unpacked; verification follows.');return
 if a.archive:archive=a.archive.resolve()
 else:
  cache=ROOT/'models/cache';cache.mkdir(parents=True,exist_ok=True);archive=cache/spec['filename']
  if not archive.is_file() or digest(archive)!=spec['sha256']:
   url=a.url or os.environ.get('SPARSE_MODEL_ARCHIVE_URL') or spec['url'];headers={}
   token=os.environ.get('GH_TOKEN')or os.environ.get('GITHUB_TOKEN')
   if token and not a.url and not os.environ.get('SPARSE_MODEL_ARCHIVE_URL'):
    headers={'Authorization':'Bearer '+token,'Accept':'application/vnd.github+json','User-Agent':'SparseSolver-model-downloader'}
    api='https://api.github.com/repos/'+spec['repository']+'/releases/tags/'+spec['release_tag']
    try:
     with urllib.request.urlopen(urllib.request.Request(api,headers=headers))as response:release=json.load(response)
    except urllib.error.HTTPError as e:raise SystemExit(f'Cannot access model release (HTTP{e.code}). Supply --archive FILE or check GitHub release access.')
    assets=[x for x in release['assets']if x['name']==spec['filename']]
    if len(assets)!=1:raise SystemExit('Model archive is not attached to the release yet.')
    url=assets[0]['url'];headers['Accept']='application/octet-stream'
   print('Downloading',spec['filename'],f'({spec["bytes"]/1e9:.2f}GB)',flush=True)
   partial=archive.with_suffix('.part')
   try:
    with urllib.request.urlopen(urllib.request.Request(url,headers=headers))as response,partial.open('wb')as f:
     count=0;milestone=0
     while True:
      chunk=response.read(8*1024**2)
      if not chunk:break
      f.write(chunk);count+=len(chunk)
      if count//(100*1024**2)>milestone:milestone=count//(100*1024**2);print(f'Downloaded {count/1024**2:.0f}MiB',flush=True)
   except urllib.error.HTTPError as e:raise SystemExit(f'Model download HTTP{e.code}: release may be unpublished/private. Use --archive FILE or GH_TOKEN for private releases.')
   if digest(partial)!=spec['sha256']:raise SystemExit('Downloaded archive SHA-256 mismatch.')
   partial.replace(archive)
 if not archive.is_file()or archive.stat().st_size!=spec['bytes']or digest(archive)!=spec['sha256']:raise SystemExit('Archive size or SHA-256 mismatch.')
 with tempfile.TemporaryDirectory(prefix='.unpack-models-',dir=ROOT)as temp:
  stage=Path(temp)
  with tarfile.open(archive,'r|gz')as tar:
   for member in tar:
    target=(stage/member.name).resolve();target.relative_to(stage.resolve())
    if member.isdir():target.mkdir(parents=True,exist_ok=True);continue
    if not member.isfile()or member.size>10*1024**2 or not(member.name=='MODEL_MANIFEST.json'or member.name.startswith('models/')):raise ValueError('Unexpected archive member')
    target.parent.mkdir(parents=True,exist_ok=True)
    with tar.extractfile(member)as source,target.open('wb')as out:shutil.copyfileobj(source,out)
  if digest(stage/'MODEL_MANIFEST.json')!=spec['inner_manifest_sha256']:raise ValueError('Archive manifest SHA-256 mismatch')
  manifest=json.loads((stage/'MODEL_MANIFEST.json').read_text())
  if len(manifest['files'])!=272:raise ValueError('Incomplete eight-flop manifest')
  for name,expected in manifest['assets'].items():
   path=(stage/name).resolve();path.relative_to(stage.resolve());assert digest(path)==expected,name
  for x in manifest['files'].values():
   path=(stage/x['blob']).resolve();path.relative_to(stage.resolve());assert digest(path)==x['blob_sha256'],x['blob']
  stage.rename(dest)
 print('Eight-flop archive installed and verified.',flush=True)
if __name__=='__main__':main()
