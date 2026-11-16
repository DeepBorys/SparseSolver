#!/usr/bin/env python3
# SparseSolver — Copyright (C) 2026 Jerzy Slaby (DeepBorys).
# Uses the postflop-solver engine by Wataru Inariba; see NOTICE.
# SPDX-License-Identifier: AGPL-3.0-or-later
# Modified for SparseSolver inference and benchmarks; see NOTICE and LICENSE.

"""Forty public situations x four models: timing, process RAM and flop policies only."""
import argparse,csv,hashlib,json,math,os,platform,shutil,statistics,subprocess,time
from pathlib import Path
from check_results import check_policy
from cpu_threads import available_threads,validate_threads
ROOT=Path(__file__).resolve().parents[1]

def sha(p):
 h=hashlib.sha256()
 with p.open('rb')as f:
  for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
 return h.hexdigest()

def save(p,x):
 p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n');tmp.replace(p)

def validate_policy(path):
 groups={};keys=set();nodes=set();rows=0
 with path.open(newline='')as f:
  for r in csv.DictReader(f):
   key=(r['path'],r['player'],r['action_index'],r['hand'])
   if key in keys:raise ValueError('Duplicate strategy row')
   keys.add(key);q=float(r['probability']);w=float(r['compatible_weight'])
   if not math.isfinite(q)or not -1e-7<=q<=1.000001 or not math.isfinite(w)or w<0:raise ValueError('Invalid strategy probability/weight')
   group=(r['path'],r['player'],r['hand']);groups[group]=groups.get(group,0)+q;nodes.add((r['path'],r['player']));rows+=1
 if not rows or any(abs(v-1)>2e-6 for v in groups.values()):raise ValueError('Strategy normalization failed')
 return {'strategy_entries':rows,'flop_nodes':len(nodes)}

def main():
 spec=ROOT/'configs/speed_cases.json';allcases=json.loads(spec.read_text())['cases'];trees=sorted({v['tree']for v in allcases.values()});flops=sorted({v['flop']for v in allcases.values()})
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--threads',type=int,default=available_threads(),help='CFR workers; defaults to available CPUs (nproc)');p.add_argument('--repeats',type=int,default=3);p.add_argument('--warmups',type=int,default=1)
 p.add_argument('--models',nargs='+',choices=['1x1','2x2','4x4','6x6'],default=['1x1','2x2','4x4','6x6']);p.add_argument('--flops',nargs='+',choices=flops);p.add_argument('--trees',nargs='+',choices=trees);p.add_argument('--cases',nargs='+',choices=sorted({v['case_id']for v in allcases.values()}));p.add_argument('--list',action='store_true');p.add_argument('--resume',action='store_true');p.add_argument('--output',type=Path,default=ROOT/'results'/('speed40_'+time.strftime('%Y%m%d_%H%M%S')+'_'+str(os.getpid())))
 a=p.parse_args();selected={k:v for k,v in allcases.items()if v['model']in a.models and (not a.flops or v['flop']in a.flops)and(not a.trees or v['tree']in a.trees)and(not a.cases or v['case_id']in a.cases)}
 if a.list:
  for k,v in selected.items():print(k,v['iterations'])
  print('Situations:',len({v['case_id']for v in selected.values()}),'Configurations:',len(selected));return
 if not selected or a.threads<1 or a.repeats<1 or a.warmups<0:raise ValueError('Invalid selection/worker/repeat count')
 validate_threads(a.threads)
 binary=ROOT/'target/release/examples/speed_case'
 if not binary.is_file():raise SystemExit('Run bash setup.sh first.')
 if not (ROOT/'.model-assets/MODEL_MANIFEST.json').is_file():raise SystemExit('Run bash download_models.sh first (or --archive FILE).')
 output=a.output.resolve();signature={'binary_sha256':sha(binary),'cases_sha256':sha(spec),'configuration_sha256':{k:sha(ROOT/v['config'])for k,v in selected.items()},'threads':a.threads,'repeats':a.repeats,'warmups':a.warmups,'selection':list(selected),'model_archive_sha256':json.loads((ROOT/'models/ALL_ASSETS.json').read_text())['sha256']}
 if a.resume:
  if json.loads((output/'SIGNATURE.json').read_text())!=signature:raise ValueError('Resume settings, binary or configuration changed')
 else:
  output.mkdir(parents=True,exist_ok=False);save(output/'SIGNATURE.json',signature);save(output/'SYSTEM.json',{'platform':platform.platform(),'cpuinfo':Path('/proc/cpuinfo').read_text()if Path('/proc/cpuinfo').exists()else None,'timing_scope':'CFR iterations only; setup, finalization, exports and checks excluded','peak_RAM_scope':'Peak process RSS including setup and exports','reference_quality_evaluation':False})
 env=os.environ.copy();env.update(OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1');results=[]
 print(f'Running {len(selected)} configurations across {len({v["case_id"]for v in selected.values()})} situations; {a.warmups}warmup +{a.repeats} measured runs',flush=True)
 print('Case / model / iters / median ms / peak MiB',flush=True)
 for key,v in selected.items():
  folder=output/key;folder.mkdir(parents=True,exist_ok=True);samples=[];hashes=[];shape={}
  for kind,count in [('warmup',a.warmups),('r',a.repeats)]:
   for rep in range(count):
    dest=folder/(kind+str(rep));done=dest/'summary.json'
    cmd=[str(binary),'--model',v['model'],'--case-config',str(ROOT/v['config']),'--threads',str(a.threads),'--assets',str(ROOT),'--output',str(dest),'--allin-cache',str(output/'cache'/v['case_id']/'allin.bin')]
    if not(a.resume and done.exists()):
     if dest.exists():shutil.rmtree(dest)
     dest.mkdir(parents=True);(output/'cache'/v['case_id']).mkdir(parents=True,exist_ok=True);save(dest/'command.json',cmd)
     with (dest/'solve.log').open('w')as log:subprocess.run(cmd,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
    row=json.loads(done.read_text())
    if row['model']!=v['model']or row['iterations']!=v['iterations']or row['threads']!=a.threads:raise ValueError('Run summary does not match specification')
    shape=validate_policy(dest/'flop_strategy.csv')
    if v['case_id']=='QsJh2h_normal':shape['static_check']=check_policy(dest/'flop_strategy.csv',v['model'],v['iterations'])
    if kind=='r':samples.append(row);hashes.append(sha(dest/'flop_strategy.csv'))
    save(output/'PROGRESS.json',{'state':'running','configuration':key,'sample':kind+str(rep),'completed_configurations':len(results),'total':len(selected)})
  if len(set(hashes))!=1:raise ValueError('Repeated policies differ: '+key)
  ms=statistics.median(x['solve_ms']for x in samples)
  if not math.isfinite(ms)or ms<=0:raise ValueError('Invalid solve time')
  peak=max(x['peak_rss_bytes']for x in samples)/2**20
  row={'case':v['case_id'],'flop':v['flop'],'tree':v['tree'],'model':v['model'],'iterations':v['iterations'],'median_ms':ms,'peak_MiB':peak,'strategy_sha256':hashes[0],'flop_strategy_log':str(folder/'r0/flop_strategy.csv'),**shape,'samples':samples};results.append(row)
  save(output/'RESULTS.json',results)
  with (output/'SUMMARY.csv').open('w',newline='')as f:
   w=csv.DictWriter(f,fieldnames=['case','flop','tree','model','iterations','median_ms','peak_MiB','flop_nodes','strategy_entries','strategy_sha256','flop_strategy_log'],extrasaction='ignore');w.writeheader();w.writerows(results)
  print(f'{v["case_id"]:25} {v["model"]:3} {v["iterations"]:4} {ms:10.3f}ms {peak:8.1f}MiB',flush=True)
 save(output/'PROGRESS.json',{'state':'complete','completed_configurations':len(results),'total':len(selected)})
 print('Results:',output,flush=True)
if __name__=='__main__':main()
