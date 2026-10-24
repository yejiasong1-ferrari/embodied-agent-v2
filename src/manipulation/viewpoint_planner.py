"""Safe sensor poses generated locally from measured targets and robot calibration."""
import numpy as np
from robot.robot_interface import Pose


class ViewpointPlanner:
    def __init__(self,config):self.config=config
    def candidates(self,target,camera,side=False):
        distance=self.config.localization.observation_distance_m
        # Top-oblique followed by opposing sides. The object geometry, never
        # a named simulator fixture, is the pivot for every view.
        offsets=[np.array([0,0,distance]),np.array([.28,0,.22]),np.array([-.28,0,.22]),np.array([0,-.28,.22]),np.array([0,.28,.22])]
        if side:
            offsets=[np.array([.28,0,.10]),np.array([-.28,0,.10]),np.array([0,-.28,.10]),np.array([0,.28,.10])]+offsets[1:]
        for offset in offsets:
            location=target.position+offset;forward=target.position-location;forward/=np.linalg.norm(forward)
            up=np.array([0.,1.,0.]) if abs(forward[2])>.98 else np.array([0.,0.,1.])
            right=np.cross(forward,up);right/=np.linalg.norm(right);down=np.cross(forward,right)
            transform=np.eye(4);transform[:3,:3]=np.column_stack([right,down,forward]);transform[:3,3]=location
            hand=transform@np.linalg.inv(camera.ee_to_camera)
            yield Pose(hand[:3,3],hand[:3,:3])
