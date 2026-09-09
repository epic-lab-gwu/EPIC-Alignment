import unittest
import numpy as np
from scipy.spatial.transform import Rotation
from epa.ov_align import associate_est_gt

class ShortGapTests(unittest.TestCase):
 def evaluate(self, est, gt, enabled=True, offset=0):
  est=np.asarray(est,float);gt=np.asarray(gt,float)
  p=np.column_stack((est,est*0,est*0)); q=Rotation.from_euler('z',est).as_quat()
  return associate_est_gt(est,p,q,gt,np.column_stack((gt,gt*0,gt*0)),Rotation.from_euler('z',gt).as_quat(),.02,offset,enabled)
 def test_short_sampling_and_rotation(self):
  out=self.evaluate(np.arange(0,1,.1),[.035,.135,.235,.335])
  np.testing.assert_allclose(out[1][:,0],out[0]);np.testing.assert_allclose(Rotation.from_quat(out[2]).as_rotvec()[:,2],out[0])
 def test_disabled(self):
  with self.assertRaises(ValueError):self.evaluate(np.arange(0,1,.1),[.035,.135,.235],False)
 def test_long_gap_and_no_extrapolation(self):
  out=self.evaluate([0,.1,.2,10,10.1,10.2],[-1,.035,.135,5,10.035,11])
  np.testing.assert_allclose(out[0],[.035,.135,10.035])
 def test_sparse_direct_matches_preserved(self):
  out=self.evaluate([0,10,20],[0,5,10,15,20]);np.testing.assert_allclose(out[0],[0,10,20])
 def test_offset(self):
  out=self.evaluate(np.arange(0,1,.1),[1.035,1.135,1.235],offset=1)
  np.testing.assert_allclose(out[1][:,0],out[0]-1)
 def test_direct_pose_not_replaced(self):
  out=self.evaluate([0,.1,.2,.3],[.01,.11,.21]);np.testing.assert_allclose(out[1][:,0],[0,.1,.2])
if __name__=='__main__':unittest.main()
