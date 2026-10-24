import unittest
import numpy as np
from config import AgentConfig
from perception.camera import CameraFrame
from perception.pointcloud import pixel_to_camera,pixel_to_world,world_to_pixel,mask_pointcloud
from world.world_model import WorldObject,WorldModel
from world.scene_graph import relation_satisfied
from llm.schemas import Goal,Reference,RecoveryDecision
from pydantic import ValidationError


def obj(name="red_cup",position=(.3,0,.05),size=(.06,.06,.1),color="red",category="cup",stamp=1):
    return WorldObject(name,f"{color} {category}",color,category,np.array(position,float),np.array([0,0,0,1.]),np.array(size,float),.9,stamp)


class ProjectionTests(unittest.TestCase):
    def test_calibrated_projection_rotated_camera_and_round_trip(self):
        k=np.array([[100.,0,30],[0,120.,20],[0,0,1]])
        t=np.array([[0,-1,0,.4],[1,0,0,-.2],[0,0,1,1],[0,0,0,1.]])
        uv=np.array([[30.,20.],[40.,32.]])
        depth=np.array([.5,1.])
        np.testing.assert_allclose(pixel_to_camera(uv,depth,k),[[0,0,.5],[.1,.1,1]])
        points=pixel_to_world(uv,depth,k,t)
        np.testing.assert_allclose(points,[[.4,-.2,1.5],[.3,-.1,2]])
        f=CameraFrame(np.zeros((40,60,3),np.uint8),np.ones((40,60)),k,t,1,"replay")
        result,z=world_to_pixel(points,f)
        np.testing.assert_allclose(result,uv); np.testing.assert_allclose(z,depth)

    def test_invalid_depth_is_removed_not_replaced_by_a_pose(self):
        depth=np.array([[0,np.nan],[np.inf,1.]])
        frame=CameraFrame(np.zeros((2,2,3),np.uint8),depth,np.eye(3),np.eye(4),1,"replay")
        cloud=mask_pointcloud(frame,np.ones((2,2),bool))
        np.testing.assert_allclose(cloud,[[1,1,1]])
        with self.assertRaises(ValueError): pixel_to_camera((0,0),0,np.eye(3))


class WorldTests(unittest.TestCase):
    def test_occlusion_remembers_then_expires_and_reacquires_same_identity(self):
        cfg=AgentConfig().perception; cfg.memory_ttl_s=5
        world=WorldModel(cfg); world.update([obj()],1)
        identity=next(iter(world.objects))
        world.update([],2); self.assertFalse(world.objects[identity].visible)
        world.update([obj(position=(.305,0,.05),stamp=3)],3)
        self.assertEqual(list(world.objects),[identity])
        world.update([],10); self.assertFalse(world.objects)

    def test_held_object_pose_uses_measured_top_and_previous_size(self):
        world=WorldModel(AgentConfig().perception); world.update([obj()],1)
        identity=next(iter(world.objects)); world.memory.held_id=identity
        # Top is now 0.15 m; neighbouring table remains at zero.
        world.update([obj(position=(.3,0,.075),size=(.06,.06,.15),stamp=2)],2)
        self.assertAlmostEqual(world.objects[identity].position[2],.1)
        self.assertAlmostEqual(world.objects[identity].size[2],.1)

    def test_relations_require_measured_geometry(self):
        plate=obj("plate",(.3,0,.02),(.24,.24,.04),"yellow","plate")
        cup=obj(position=(.3,0,.09))
        self.assertTrue(relation_satisfied(cup,plate,"on"))
        cup.position[0]=.65
        self.assertFalse(relation_satisfied(cup,plate,"on"))
        self.assertTrue(relation_satisfied(cup,plate,"right_of"))
        self.assertFalse(relation_satisfied(cup,plate,"inside"))

    def test_reference_does_not_silently_choose_between_two_cups(self):
        world=WorldModel(AgentConfig().perception)
        world.update([obj(),obj(position=(-.3,0,.05),color="blue")],1)
        with self.assertRaises(LookupError): world.resolve(Reference(description="cup",category="cup"))


class LLMBoundaryTests(unittest.TestCase):
    def test_motor_commands_and_unapproved_recovery_are_rejected(self):
        valid={"goal":"place_on","object":{"description":"red cup"},"target":{"description":"plate"}}
        self.assertEqual(Goal.model_validate(valid).relation,"on")
        with self.assertRaises(ValidationError): Goal.model_validate({**valid,"joint_angles":[0]*7})
        with self.assertRaises(ValidationError): RecoveryDecision(action="DISABLE_SAFETY",reason="go")
        with self.assertRaises(ValidationError): Goal.model_validate_json("I suggest grabbing the cup")


if __name__=="__main__": unittest.main()
