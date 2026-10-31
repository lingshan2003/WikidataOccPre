import unittest,random
import numpy as np
from training.graphmask.window_schedule import window_order,batch_seed
class ScheduleTest(unittest.TestCase):
 def test_permutation_and_replay(self):
  for e in range(6):
   for k in range(20):
    a=window_order(10,43,e,k);self.assertEqual(sorted(a),list(range(10)));self.assertEqual(a,window_order(10,43,e,k))
 def test_no_global_rng_changes(self):
  random.seed(123);np.random.seed(123);r=random.getstate();n=np.random.get_state()
  window_order(10,42,1,1);batch_seed(42,1,1,1)
  self.assertEqual(r,random.getstate());self.assertTrue(np.array_equal(n[1],np.random.get_state()[1]))
 def test_order_does_not_change_window_draws(self):
  def draws(order):
   return {i:np.random.RandomState(batch_seed(42,1,3,i)).rand(8).tolist() for i in order}
  self.assertEqual(draws(window_order(10,None,1,3)),draws(window_order(10,44,1,3)))
  self.assertEqual(window_order(10,None,1,1),list(range(10)))
if __name__=='__main__':unittest.main()
