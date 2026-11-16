#!/usr/bin/env python3
# SparseSolver — Copyright (C) 2026 Jerzy Slaby (DeepBorys).
# Uses the postflop-solver engine by Wataru Inariba; see NOTICE.
# SPDX-License-Identifier: AGPL-3.0-or-later
# Modified for SparseSolver inference and benchmarks; see NOTICE and LICENSE.

"""Quick QJ2 model benchmark; no reference downloads or locked-response solves."""
import argparse,csv,hashlib,json,os,platform,statistics,subprocess,time
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"scripts"))
from check_results import check_policy
from cpu_threads import available_threads,validate_threads

def main():
 root=Path(__file__).resolve().parents[1]
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('--models',nargs='+',choices=['49x48','6x6','4x4','2x2','1x1'],default=['1x1','2x2','4x4','6x6'])
 p.add_argument('--threads',type=int,default=available_threads(),help='CFR workers; defaults to available CPUs (nproc)')
 p.add_argument('--repeats',type=int,default=3)
 p.add_argument('--assets',type=Path,default=root if (root/'models/banks').is_dir() else root/'ec2/public-benchmark')
 default_binary=next((q for q in [root/'target-ec2/release/examples/qj2_3bet_sparse',root/'target/release/examples/qj2_3bet_sparse'] if q.is_file()),root/'target/release/examples/qj2_3bet_sparse')
 p.add_argument('--binary',type=Path,default=default_binary)
 p.add_argument('--output',type=Path,default=root/'results'/('basic_models_'+time.strftime('%Y%m%d_%H%M%S')+'_'+str(os.getpid())))
 a=p.parse_args();validate_threads(a.threads);assert a.repeats>0
 a.output=a.output.resolve();a.output.mkdir(parents=True,exist_ok=False)
 env=os.environ.copy();env.update(OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
 system={'platform':platform.platform(),'threads':a.threads,'binary_sha256':hashlib.sha256(a.binary.read_bytes()).hexdigest(),'timing_scope':'CFR iterations only; setup, finalization, evaluation and exports excluded','warmups_per_model':1,'repeats':a.repeats}
 if Path('/proc/cpuinfo').exists():system['cpuinfo']=Path('/proc/cpuinfo').read_text()
 if hasattr(os,'sched_getaffinity'):system['cpu_affinity']=sorted(os.sched_getaffinity(0))
 (a.output/'SYSTEM.json').write_text(json.dumps(system,indent=2)+'\n')
 results=[]
 print('Model  Iters  Median ms  Peak MiB  Flop strategy log',flush=True)
 for model in a.models:
  samples=[];hashes=[];checks=[]
  for rep in range(-1,a.repeats):
   dest=a.output/model/('warmup' if rep<0 else f'r{rep}');dest.mkdir(parents=True)
   cmd=[str(a.binary.resolve()),'--model',model,'--threads',str(a.threads),'--assets',str(a.assets.resolve()),'--output',str(dest),'--allin-cache',str(a.output/'allin.bin')]
   (dest/'command.json').write_text(json.dumps(cmd)+'\n')
   with (dest/'solve.log').open('w') as log:subprocess.run(cmd,cwd=root,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
   if rep>=0:
    samples.append(json.loads((dest/'summary.json').read_text()));hashes.append(hashlib.sha256((dest/'flop_strategy.csv').read_bytes()).hexdigest())
    if model!='49x48':checks.append(check_policy(dest/'flop_strategy.csv',model,samples[-1]['iterations']))
  if len(set(hashes))!=1:raise RuntimeError(f'{model}: repeated flop strategies differ')
  peak=max(s['peak_rss_bytes'] for s in samples) if all(s['peak_rss_bytes'] is not None for s in samples) else None
  row={'model':model,'iterations':samples[0]['iterations'],'median_ms':statistics.median(s['solve_ms'] for s in samples),'peak_MiB':peak/2**20 if peak is not None else None,'flop_strategy_log':str(a.output/model/'r0/flop_strategy.csv'),'strategy_sha256':hashes[0],'samples':samples,'static_checks':checks}
  results.append(row)
  (a.output/model/'STATIC_CHECKS.json').write_text(json.dumps(checks,indent=2)+'\n')
  if checks:print(f'STATIC_CHECK_PASS {model}: {checks[0]["entries"]} strategy entries; max difference {max(x["max_probability_difference"]for x in checks):.3g}',flush=True)
  (a.output/'RESULTS.json').write_text(json.dumps(results,indent=2)+'\n')
  with (a.output/'SUMMARY.csv').open('w',newline='') as f:
   w=csv.DictWriter(f,fieldnames=['model','iterations','median_ms','peak_MiB','flop_strategy_log','strategy_sha256'],extrasaction='ignore');w.writeheader();w.writerows(results)
  peak_text=f"{row['peak_MiB']:.1f}" if peak is not None else 'unavailable'
  print(f"{model:6} {row['iterations']:5} {row['median_ms']:10.3f} {peak_text:>9} {row['flop_strategy_log']}",flush=True)
 print('Results:',a.output)
if __name__=='__main__':main()
