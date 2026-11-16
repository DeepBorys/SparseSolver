import os,sys,unittest
from pathlib import Path
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from cpu_threads import available_threads,validate_threads

class WorkerDetection(unittest.TestCase):
 def test_omp_single_thread_does_not_limit_cfr(self):
  def nproc(*args,**kwargs):
   self.assertNotIn('OMP_NUM_THREADS',kwargs['env']);self.assertNotIn('OMP_THREAD_LIMIT',kwargs['env']);return '4\n'
  with patch.dict(os.environ,{'OMP_NUM_THREADS':'1','OMP_THREAD_LIMIT':'1'}),patch('cpu_threads.subprocess.check_output',side_effect=nproc),patch('cpu_threads.os.sched_getaffinity',return_value={0,1,2,3}):self.assertEqual(available_threads(),4)
 def test_affinity_caps_detected_count(self):
  with patch('cpu_threads.subprocess.check_output',return_value='16\n'),patch('cpu_threads.os.sched_getaffinity',return_value={0,1,2,3}):self.assertEqual(available_threads(),4)
 def test_explicit_oversubscription_rejected(self):
  with patch('cpu_threads.available_threads',return_value=4):
   validate_threads(1);validate_threads(4)
   with self.assertRaisesRegex(ValueError,'available CPUs: 4'):validate_threads(16)
 def test_missing_nproc_uses_affinity(self):
  with patch('cpu_threads.subprocess.check_output',side_effect=FileNotFoundError),patch('cpu_threads.os.sched_getaffinity',return_value={0,1,2,3}):self.assertEqual(available_threads(),4)
if __name__=='__main__':unittest.main()
