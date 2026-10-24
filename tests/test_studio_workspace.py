"""Camera clamps and evidence reducers: UI must never manufacture acceptance."""
import unittest
import numpy as np
from studio.camera import CameraState,project
from studio.bridge import MissionState

class WorkspaceContracts(unittest.TestCase):
    def test_spectator_camera_is_immutable_and_bounded(self):
        c=CameraState();d=c.orbit(1e6,1e6).zoom(1e6).pan(1e6,1e6,0)
        self.assertEqual(c,CameraState());self.assertGreaterEqual(d.elevation,-89)
        self.assertEqual(d.distance,.55);self.assertLessEqual(max(abs(v) for v in d.lookat[:2]),1.2)
        self.assertEqual(c.focus([float('nan'),0,0]),c)
        self.assertEqual(c.zoom(-1e6).zoom(-1e6).distance,4.)
    def test_projection_uses_only_camera_calibration(self):
        p={'pos':[0,0,1],'forward':[0,0,-1],'up':[0,1,0],'near':.01,'tan_half_fov':1.,'aspect':2.}
        self.assertEqual(project([0,0,0],p),(.5,.5,1.))
        self.assertIsNone(project([0,0,2],p));self.assertIsNone(project([3,0,0],p))
    def test_success_needs_explicit_validator_acceptance(self):
        s=MissionState();s.apply({'type':'state','data':{'state':'SUCCESS'}});self.assertFalse(s.complete)
        s.apply({'type':'final_validation','data':{'accepted':False}});self.assertFalse(s.complete)
        s.apply({'type':'final_validation','data':{'accepted':True}});self.assertTrue(s.complete)
    def test_recovery_rolls_back_plan_and_preserves_real_attempts(self):
        s=MissionState()
        def event(kind,**d):s.apply({'type':kind,'data':d})
        event('plan',steps=[{'skill':x} for x in ['APPROACH_OBJECT','GRASP_OBJECT','VERIFY_TASK']])
        for skill in ['APPROACH_OBJECT','GRASP_OBJECT']:
            event('state',state='EXECUTE',skill=skill);event('skill',name=skill,success=True)
        event('state',state='RECOVER');event('recovery',failure='GRASP_FAILED',candidate='top_grasp:0',action='CHANGE_GRASP',attempt=1)
        event('state',state='EXECUTE',skill='APPROACH_OBJECT')
        self.assertEqual(s.completed,set());self.assertTrue(s.recovering);self.assertFalse(s.recovery_resolved)
        event('skill',name='APPROACH_OBJECT',success=True)
        self.assertTrue(s.recovery_resolved);self.assertFalse(s.complete);self.assertEqual(len(s.recoveries),1)
        event('state',state='RECOVER');self.assertFalse(s.recovery_resolved)
    def test_new_subplan_clears_prior_acceptance(self):
        s=MissionState(validation={'accepted':True});s.apply({'type':'plan','data':{'steps':[{'skill':'FIND_OBJECT'}]}})
        self.assertEqual(s.validation,{})

if __name__=='__main__':unittest.main()
