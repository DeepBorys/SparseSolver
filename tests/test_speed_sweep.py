import csv,json,subprocess,sys,tempfile,unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'scripts'))
from speed_sweep import validate_policy

class SweepChecks(unittest.TestCase):
 def test_frozen_coverage_and_budgets(self):
  cases=json.loads((ROOT/'configs/speed_cases.json').read_text())['cases']
  self.assertEqual(len(cases),160);self.assertEqual(len({x['case_id']for x in cases.values()}),40);self.assertEqual(len({x['flop']for x in cases.values()}),8)
  for v in cases.values():
   budgets=dict(zip(['1x1','2x2','4x4','6x6'],[120,200,250,300]if v['tree']=='SRP_deep'else[60,120,150,200]));self.assertEqual(v['iterations'],budgets[v['model']])
   rows=dict(line.split('\t',1)for line in (ROOT/v['config']).read_text().splitlines());self.assertEqual(rows['model'],v['model']);self.assertEqual(rows['flop'],v['flop'])
 def test_downstream_lock_cases_keep_starting_ranges(self):
  for flop in ['QsJh2h','As9h4d','Js7h3d','Js8s3s','Js9s4h','Ks4h4d','QsTs7h','Td9d6h']:
   read=lambda tree:dict(line.split('\t',1)for line in (ROOT/f'configs/speed/{flop}_{tree}_2x2.tsv').read_text().splitlines())
   normal=read('normal')
   for tree,forced in [('forced_ip_checks','check'),('forced_ip_bets','bet')]:
    row=read(tree);self.assertEqual((row['oop'],row['ip']),(normal['oop'],normal['ip']));self.assertEqual(row['force_ip_opening'],forced)
 def test_invalid_policy_is_rejected(self):
  with tempfile.TemporaryDirectory()as folder:
   p=Path(folder)/'strategy.csv';p.write_text('path,player,action_index,hand,probability,compatible_weight\n,0,0,AcAd,0.7,1\n,0,1,AcAd,0.7,1\n')
   with self.assertRaisesRegex(ValueError,'normalization'):validate_policy(p)
 def test_wrong_archive_is_rejected_without_downloading(self):
  with tempfile.TemporaryDirectory()as folder:
   p=Path(folder)/'wrong.tar.gz';p.write_bytes(b'not the model archive')
   result=subprocess.run([sys.executable,str(ROOT/'scripts/download_models.py'),'--archive',str(p)],capture_output=True,text=True)
   self.assertNotEqual(result.returncode,0);self.assertIn('SHA-256 mismatch',result.stdout+result.stderr)
if __name__=='__main__':unittest.main()
