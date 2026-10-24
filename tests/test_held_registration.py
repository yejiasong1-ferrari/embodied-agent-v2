import unittest
import numpy as np
from perception.held_registration import register_translation
from test_adaptive_geometry import surface

class HeldRegistrationTests(unittest.TestCase):
 def test_partial_stem_and_cap_register_without_uniform_cylinder_assumption(self):
  shape=surface(True);visible=shape[shape[:,0]>.40];shift=np.array([.003,-.002,.001])
  recovered,evidence=register_translation(visible+shift,shape)
  self.assertIsNotNone(recovered);self.assertTrue(evidence['consistent'])
  np.testing.assert_allclose(recovered,shift,atol=.002)
 def test_unrelated_points_do_not_confirm_attachment(self):
  shape=surface(True);recovered,evidence=register_translation(shape+np.array([.15,0,0]),shape)
  self.assertIsNone(recovered)
 def test_missing_pixels_cannot_confirm_attachment(self):
  self.assertIsNone(register_translation(np.empty((0,3)),surface(True))[0])

if __name__=='__main__':unittest.main()
