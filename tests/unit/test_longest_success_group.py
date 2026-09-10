import unittest
import numpy as np
from epa.core.evaluation import _longest_successful_group
from epa.compat.ov_eval.evaluate import _compute_valid_rpe_segments

class LongestGroupTests(unittest.TestCase):
 def test_short_gap_connects_without_filling(self):
  t=np.array([0,20,26,46,66,96.]);m=np.array([1,0,1,0,1],bool)
  out,info=_longest_successful_group(t,m)
  np.testing.assert_array_equal(out,[1,0,1,0,0]);self.assertEqual(info['selected_group']['successful_duration_s'],40)
 def test_exactly_ten_does_not_connect(self):
  out,_=_longest_successful_group([0,8,18,27],[1,0,1]);np.testing.assert_array_equal(out,[0,0,1])
 def test_duration_not_sample_count(self):
  out,_=_longest_successful_group([0,1,2,3,23,33],[1,1,1,0,1]);np.testing.assert_array_equal(out,[0,0,0,0,1])
 def test_failed_time_does_not_increase_group_score(self):
  out,_=_longest_successful_group([0,1,10,11,31,36],[1,0,1,0,1]);np.testing.assert_array_equal(out,[0,0,0,0,1])
 def test_empty_all_failed_and_tie(self):
  self.assertEqual(len(_longest_successful_group([],[])[0]),0)
  np.testing.assert_array_equal(_longest_successful_group([0,1,2],[0,0])[0],[0,0])
  np.testing.assert_array_equal(_longest_successful_group([0,10,30,40],[1,0,1])[0],[1,0,0])
 def test_no_extra_failures_for_one_group(self):
  np.testing.assert_array_equal(_longest_successful_group([0,1,2,3,4],[0,1,0,1])[0],[0,1,0,1])
 def test_rpe_does_not_cross_connected_gap(self):
  t=np.arange(31.);m=np.ones(30,bool);m[12:16]=False
  selected,_=_longest_successful_group(t,m)
  p=np.column_stack((t,t*0,t*0));q=np.tile([0,0,0,1],(31,1))
  out=_compute_valid_rpe_segments(p,q,p,q,selected,[10.])
  pairs=out[10.]['pair_ids'];self.assertGreater(len(pairs),0)
  for i,j in pairs:self.assertTrue(np.all(selected[i:j]))
if __name__=='__main__':unittest.main()
