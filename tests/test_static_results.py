import csv,gzip,sys,tempfile,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from check_results import FIXTURES,check_policy

class StaticPolicyChecks(unittest.TestCase):
 def setUp(self):
  self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup);self.path=Path(self.temp.name)/'flop_strategy.csv'
  with gzip.open(FIXTURES/'2x2.csv.gz','rb')as f:self.path.write_bytes(f.read())
 def test_matching_fixture_passes(self):
  x=check_policy(self.path,'2x2',120);self.assertEqual(x['entries'],11024);self.assertEqual(x['max_probability_difference'],0)
 def test_changed_policy_rejected_even_if_normalized(self):
  with self.path.open(newline='')as f:r=list(csv.DictReader(f));fields=list(r[0])
  group=[x for x in r if x['path']==''and x['player']=='0'and x['hand']==r[0]['hand']]
  low=min(group,key=lambda x:float(x['probability']));high=max(group,key=lambda x:float(x['probability']))
  low['probability']=str(float(low['probability'])+.01);high['probability']=str(float(high['probability'])-.01)
  with self.path.open('w',newline='')as f:w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(r)
  with self.assertRaisesRegex(ValueError,'maximum probability difference'):check_policy(self.path,'2x2',120)
 def test_missing_hand_action_rejected(self):
  rows=self.path.read_text().splitlines();self.path.write_text('\n'.join(rows[:-1])+'\n')
  with self.assertRaisesRegex(ValueError,'coverage mismatch'):check_policy(self.path,'2x2',120)
 def test_wrong_budget_rejected(self):
  with self.assertRaisesRegex(ValueError,'iteration budget'):check_policy(self.path,'2x2',121)
if __name__=='__main__':unittest.main()
