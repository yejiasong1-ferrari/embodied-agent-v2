import unittest

import numpy as np

from config import AgentConfig
from spatial.constraint_parser import parse_local
from agent.task_interpreter import reference
from validation.task_validator import TaskValidator
from world.world_model import WorldObject


class UprightResultEvidence(unittest.TestCase):
    def test_explicit_upright_requires_a_measured_surface_normal(self):
        goal=parse_local('把红杯放到黄盘左边10cm，保持竖直',reference)
        source=WorldObject('cup','red cup','red','cup',np.array([-.3,.2,.04]),
            np.array([0,0,0,1]),np.array([.06,.06,.08]),.95,1.)
        target=WorldObject('plate','yellow plate','yellow','plate',np.array([.25,.2,.02]),
            np.array([0,0,0,1]),np.array([.20,.20,.04]),.95,1.)
        validator=TaskValidator(AgentConfig())
        self.assertFalse(validator.validate(goal,source,target,source.position.copy(),False)[0])
        source.manipulation_geometry['top_surface_normal']=[0.,0.,1.]
        self.assertTrue(validator.validate(goal,source,target,source.position.copy(),False)[0])
        source.manipulation_geometry['top_surface_normal']=[.5,0.,.866]
        self.assertFalse(validator.validate(goal,source,target,source.position.copy(),False)[0])


if __name__=='__main__':unittest.main()
