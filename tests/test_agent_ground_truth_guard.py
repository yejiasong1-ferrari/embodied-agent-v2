"""Enforce the runtime/evaluation boundary and pixel provenance."""
import ast
from pathlib import Path
import unittest
import numpy as np
from perception.object_detector import ColorComponentDetector


ROOT=Path(__file__).resolve().parents[1]/"src"
PURE_PACKAGES=("agent","perception","world","skills","manipulation","validation","recovery","llm","spatial")


class GroundTruthGuardTests(unittest.TestCase):
    def test_runtime_packages_cannot_import_simulation_state_or_legacy_perception(self):
        forbidden={"mujoco","simulation","evaluation","object_registry","scene_layout","vision_detector","camera_demo","embodied_v2"}
        for package in PURE_PACKAGES:
            for path in (ROOT/package).glob("*.py"):
                if path.name=="cli.py": continue  # composition root only
                tree=ast.parse(path.read_text())
                for node in ast.walk(tree):
                    if isinstance(node,ast.Import): names=[a.name for a in node.names]
                    elif isinstance(node,ast.ImportFrom): names=[node.module or ""]
                    else: names=[]
                    self.assertFalse({n.split('.')[0] for n in names}&forbidden,f"{path}:{getattr(node,'lineno',0)}")
                    if isinstance(node,ast.Attribute):
                        self.assertNotIn(node.attr,{"xpos","body_pos","geom_xpos","qpos","_sim"},str(path))
                    if isinstance(node,ast.Name): self.assertNotEqual(node.id,"SimulationGroundTruth",str(path))

    def test_robot_has_no_object_lookup_api(self):
        path=ROOT/"robot"/"mujoco_robot.py"
        tree=ast.parse(path.read_text())
        methods={n.name for n in ast.walk(tree) if isinstance(n,ast.FunctionDef)}
        self.assertFalse(methods&{"body_pose","body_bounds","movable_body_names","object_position"})

    def test_black_camera_cannot_reveal_objects(self):
        detector=ColorComponentDetector()
        self.assertEqual(detector.detect(np.zeros((200,200,3),np.uint8),["red cup"]),[])
        image=np.zeros((200,200,3),np.uint8); image[20:80,50:110]=(220,25,20)
        detections=detector.detect(image,["red cup"])
        self.assertEqual(len(detections),1)
        self.assertEqual(detections[0].color,"red")
        # A text query alone never supplies a semantic category or a pose.
        self.assertEqual(detections[0].label,"object")


if __name__=="__main__": unittest.main()
