import unittest
from types import SimpleNamespace

import mujoco

from evaluation.ground_truth import SimulationGroundTruth


class GroundTruthGeometry(unittest.TestCase):
    def test_yawed_cylinder_keeps_its_circular_horizontal_radius(self):
        model=mujoco.MjModel.from_xml_string('''<mujoco><worldbody>
            <body name="cap" euler="0 0 45" pos=".2 -.1 .08">
                <geom type="cylinder" size=".055 .01" mass=".02"/>
            </body></worldbody></mujoco>''')
        data=mujoco.MjData(model);mujoco.mj_forward(model,data)
        low,high=SimulationGroundTruth(SimpleNamespace(model=model,data=data,execution_finished=True)).bounds('cap')
        self.assertAlmostEqual(high[0]-low[0],.11,places=6)
        self.assertAlmostEqual(high[1]-low[1],.11,places=6)
        self.assertAlmostEqual(high[2]-low[2],.02,places=6)


if __name__=='__main__':unittest.main()
