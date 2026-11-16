#!/usr/bin/env python3
# SparseSolver — Copyright (C) 2026 Jerzy Slaby (DeepBorys).
# Uses the postflop-solver engine by Wataru Inariba; see NOTICE.
# SPDX-License-Identifier: AGPL-3.0-or-later
# Modified for SparseSolver inference and benchmarks; see NOTICE and LICENSE.

"""Check QJ2 single-size outputs against small frozen numerical fixtures."""
import argparse,csv,gzip,hashlib,json,math
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
FIXTURES=ROOT/'tests/fixtures/qj2_single_size'

def fail(condition,message):
 if not condition:raise ValueError(message)

def finite(value,label):
 x=float(value);fail(math.isfinite(x),label+': non-finite value');return x

def policy_rows(path,model,iterations):
 opener=gzip.open if str(path).endswith('.gz') else open
 result={};sums={};nodes=set()
 with opener(path,'rt',newline='') as f:
  for r in csv.DictReader(f):
   fail(r['model']==model and int(r['iterations'])==iterations,'Policy model/iteration mismatch')
   h=r['hand'];fail(len(h)==4,'Invalid hand name');hand=''.join(sorted([h[:2],h[2:]]))
   node=(r['path'],int(r['player']));key=(*node,int(r['action_index']),hand)
   fail(key not in result,'Duplicate policy entry')
   q=finite(r['probability'],'probability');w=finite(r['compatible_weight'],'compatible_weight')
   fail(-1e-7<=q<=1.0000001 and w>=0,'Invalid probability or compatible weight')
   result[key]=(r['action'],q);nodes.add(node);group=(*node,hand);sums[group]=sums.get(group,0.0)+q
 return result,sums,nodes

def check_policy(path,model,iterations):
 expected=json.loads((FIXTURES/'expected.json').read_text());spec=expected['models'].get(model)
 fail(spec is not None,'No static policy fixture for '+model)
 fail(iterations==spec['iterations'],'No fixture for this iteration budget')
 fixture=FIXTURES/spec['file'];fail(hashlib.sha256(fixture.read_bytes()).hexdigest()==spec['sha256'],'Static fixture hash mismatch')
 truth,_,_=policy_rows(fixture,model,iterations);actual,sums,nodes=policy_rows(path,model,iterations)
 fail(actual.keys()==truth.keys(),'Policy node/hand/action coverage mismatch')
 fail(len(actual)==spec['entries'] and len(nodes)==spec['flop_nodes'],'Unexpected entry/node count')
 fail(all(abs(v-1.0)<=expected['probability_sum_tolerance']for v in sums.values()),'Action probabilities do not sum to one')
 delta=0.0
 for key,(action,q)in actual.items():
  fail(action==truth[key][0],'Action label differs from fixture');delta=max(delta,abs(q-truth[key][1]))
 fail(delta<=expected['probability_absolute_tolerance'],f'{model}: maximum probability difference {delta:.9g} exceeds {expected["probability_absolute_tolerance"]}')
 return {'model':model,'iterations':iterations,'entries':len(actual),'flop_nodes':len(nodes),'max_probability_difference':delta,'absolute_tolerance':expected['probability_absolute_tolerance'],'state':'pass'}

def check_directory(folder):
 folder=Path(folder).resolve();expected=json.loads((FIXTURES/'expected.json').read_text());reports=[]
 if (folder/'RESULTS.json').is_file():
  rows=json.loads((folder/'RESULTS.json').read_text());fail(bool(rows),'Empty benchmark results')
  for row in rows:
   # Relative to this result directory, so saved outputs can be moved.
   samples=sorted((folder/row['model']).glob('r[0-9]*'));fail(bool(samples),'No policies for '+row['model'])
   if 'samples'in row:fail(len(samples)==len(row['samples']),'Missing benchmark sample directory')
   for sample in samples:
    reports.append(check_policy(sample/'flop_strategy.csv',row['model'],row['iterations']))
  fail(bool(reports),'No benchmark policies found')
 else:
  s=json.loads((folder/'summary.json').read_text())
  if 'seed_model' not in s:
   reports.append(check_policy(folder/'flop_strategy.csv',s['model'],s['iterations']))
  else:
   fail(s['seed_model']=='2x2' and s['seed_iterations']==120,'No static fixture for this seed')
   reports.append(check_policy(folder/'seed_flop_strategy.csv','2x2',120))
   tolerance=expected['full_exploitability_tolerance_pct_pot']
   if 'refine_iterations'in s:
    target=expected['assembly']
    for key in ['refine_iterations','frontiers','flop_frozen','full49x48']:fail(s.get(key)==target[key],'Assembly configuration/coverage mismatch: '+key)
    value=finite(s['full_tree_exploitability_pct_pot'],'full exploitability');delta=abs(value-target['full_tree_exploitability_pct_pot'])
    fail(delta<=tolerance,f'Assembly exploitability differs by {delta}% pot')
    reports.append(check_policy(folder/'flop_strategy.csv','2x2',120))
    fail((folder/'seed_flop_strategy.csv').read_bytes()==(folder/'flop_strategy.csv').read_bytes(),'Flop changed during refinement')
    reports.append({'state':'pass','assembly_full_exploitability_pct_pot':value,'expected':target['full_tree_exploitability_pct_pot'],'tolerance_pct_pot':tolerance})
   else:
    with (folder/'full_curve.csv').open(newline='')as f:rows=list(csv.DictReader(f))
    curve={int(r['full_iteration']):r for r in rows}
    fail(len(curve)==len(rows),'Duplicate full checkpoints')
    fail(s['full_iterations']>=10,'Need at least10 full updates to check staged continuation')
    for step in range(0,s['full_iterations']+1,10):fail(step in curve,'Missing full checkpoint: '+str(step))
    for text,target in expected['staged']['checkpoints'].items():
     step=int(text)
     if step>s['full_iterations']:continue
     fail(step in curve and int(curve[step]['cfr_clock'])==120+step,'Staged checkpoint/clock mismatch')
     value=finite(curve[step]['exploitability_pct_pot'],'staged exploitability');fail(abs(value-target)<=tolerance,'Staged exploitability differs at '+text)
     reports.append({'state':'pass','full_iteration':step,'exploitability_pct_pot':value,'expected':target,'tolerance_pct_pot':tolerance})
 return reports

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('results',type=Path);a=p.parse_args()
 reports=check_directory(a.results)
 (a.results/'STATIC_CHECKS.json').write_text(json.dumps(reports,indent=2)+'\n')
 for r in reports:print('STATIC_CHECK_PASS',json.dumps(r,sort_keys=True))
 print('No wall-clock speed assertion; locked-BR losses were not recomputed.')
if __name__=='__main__':main()
