#!/usr/bin/env python3
# SparseSolver — Copyright (C) 2026 Jerzy Slaby (DeepBorys).
# Uses the postflop-solver engine by Wataru Inariba; see NOTICE.
# SPDX-License-Identifier: AGPL-3.0-or-later
# Modified for SparseSolver inference and benchmarks; see NOTICE and LICENSE.

"""Verify and unpack frozen CPU inference assets. No training dependencies."""
import argparse,hashlib,json,os,zlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def inside(name):
 p=(ROOT/name).resolve();p.relative_to(ROOT.resolve());return p

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--all',action='store_true');a=p.parse_args()
 base=ROOT/'.model-assets' if a.all else ROOT
 manifest=base/'MODEL_MANIFEST.json' if a.all else ROOT/'models/MANIFEST.json'
 if not manifest.exists():raise SystemExit('Run bash download_models.sh first.')
 spec=json.loads(manifest.read_text());checked=set()
 for name,expected in spec['assets'].items():
  source=(base/name).resolve();source.relative_to(base.resolve())
  if digest(source)!=expected:raise ValueError('Asset hash mismatch: '+name)
  if a.all:
   target=inside(name);target.parent.mkdir(parents=True,exist_ok=True)
   if target.exists() and digest(target)!=expected:raise ValueError('Existing interface differs: '+name)
   if not target.exists():target.write_bytes(source.read_bytes())
 for name,x in spec['files'].items():
  dst=inside(name);packed=(base/x['blob']).resolve();packed.relative_to(base.resolve());key=x['raw_sha256'];cache=inside('models/cache/'+key+'.cpu.bin')
  if key not in checked:
   if digest(packed)!=x['blob_sha256']:raise ValueError('Compressed model hash mismatch: '+name)
   if not cache.exists() or cache.stat().st_size!=x['raw_bytes'] or digest(cache)!=key:
    if not 0<x['raw_bytes']<64*1024**2:raise ValueError('Invalid model size')
    decoder=zlib.decompressobj();raw=decoder.decompress(packed.read_bytes(),x['raw_bytes']+1)
    if not decoder.eof or decoder.unused_data or len(raw)!=x['raw_bytes'] or hashlib.sha256(raw).hexdigest()!=key:raise ValueError('Native model verification failed: '+name)
    cache.parent.mkdir(parents=True,exist_ok=True);tmp=cache.with_suffix('.tmp');tmp.write_bytes(raw);tmp.replace(cache)
   checked.add(key)
  dst.parent.mkdir(parents=True,exist_ok=True)
  if dst.exists():
   if digest(dst)!=key:raise ValueError('Existing native model differs: '+name)
  else:
   try:os.link(cache,dst)
   except OSError:dst.write_bytes(cache.read_bytes())
 print(f'MODELS_READY files={len(spec["files"])} hash_verified=true native_cpu=true',flush=True)
if __name__=='__main__':main()
