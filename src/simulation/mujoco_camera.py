import time
import mujoco
import numpy as np
from perception.camera import CameraFrame


class MujocoCamera:
    """Only camera calibration and rendered RGB/depth leave this boundary."""
    def __init__(self, simulation, robot=None):
        self._sim, self._robot = simulation, robot
        self.config = simulation.config.camera
        self.ee_to_camera=np.eye(4)
        cid=simulation.model.camera("gripper_camera").id
        rotation=np.empty(9);mujoco.mju_quat2Mat(rotation,simulation.model.cam_quat[cid])
        self.ee_to_camera[:3,:3]=rotation.reshape(3,3)@np.diag([1,-1,-1])
        self.ee_to_camera[:3,3]=simulation.model.cam_pos[cid]
        self.index = 0
        self.manipulation_active=False
        if self.config.mode!="wrist": self.move_viewpoint(0)

    @property
    def viewpoint_count(self):
        return len(self.config.wrist_scan_angles if self.config.mode == "wrist" else self.config.viewpoints)

    def move_viewpoint(self,index):
        if not 0 <= index < self.viewpoint_count:
            raise ValueError("viewpoint out of range")
        self.index = index
        if self.config.mode == "wrist":
            if self.manipulation_active: return False
            if self._robot is None:
                raise RuntimeError("wrist scanning requires a robot")
            q = np.array([self.config.wrist_scan_angles[index],-.2575,0,-2.0464,0,1.7889,-.7853])
            from validation.safety_monitor import SafetyMonitor
            return self._robot.move_joint(q,lambda:SafetyMonitor(self._robot,self._sim.config).check())
        position = np.asarray(self.config.viewpoints[index])
        target=self.config.view_targets[index] if index<len(self.config.view_targets) else self.config.look_at
        forward = np.asarray(target)-position
        forward /= np.linalg.norm(forward)
        up = np.array([0.,1.,0.]) if abs(forward[2])>.99 else np.array([0.,0.,1.])
        right = np.cross(forward,up); right /= np.linalg.norm(right)
        up = np.cross(right,forward)
        rotation = np.column_stack((right,up,-forward))
        quat = np.empty(4); mujoco.mju_mat2Quat(quat,rotation.ravel())
        with self._sim.lock:
            body = self._sim.model.body("agent_camera_mount")
            mocap = int(self._sim.model.body_mocapid[body.id])
            self._sim.data.mocap_pos[mocap] = position
            self._sim.data.mocap_quat[mocap] = quat
            mujoco.mj_forward(self._sim.model,self._sim.data)
        return True

    def move_observation(self,position,target):
        if self.config.mode=='wrist':raise RuntimeError('wrist viewpoint must be moved through robot planner')
        forward=np.asarray(target)-np.asarray(position);forward/=np.linalg.norm(forward)
        up=np.array([0.,1.,0.]) if abs(forward[2])>.99 else np.array([0.,0.,1.])
        right=np.cross(forward,up);right/=np.linalg.norm(right);up=np.cross(right,forward)
        quat=np.empty(4);mujoco.mju_mat2Quat(quat,np.column_stack((right,up,-forward)).ravel())
        with self._sim.lock:
            body=self._sim.model.body('agent_camera_mount');mocap=int(self._sim.model.body_mocapid[body.id])
            self._sim.data.mocap_pos[mocap]=position;self._sim.data.mocap_quat[mocap]=quat
            mujoco.mj_forward(self._sim.model,self._sim.data)

    def set_manipulation_active(self,active):
        self.manipulation_active=active

    def update_collision_world(self,world):
        if self.config.mode=="wrist" and self._robot is not None:
            self._robot.configure_collision_world(world)

    def capture(self,detail=False):
        name = "gripper_camera" if self.config.mode=="wrist" else "agent_rgbd"
        with self._sim.lock:
            model,data = self._sim.model,self._sim.data
            renderer=self._sim.get_detail_renderer() if detail else self._sim.renderer
            cid = model.camera(name).id
            renderer.update_scene(data,camera=name)
            rgb = renderer.render().copy()
            renderer.enable_depth_rendering()
            depth = renderer.render().copy()
            renderer.disable_depth_rendering()
            height,width=depth.shape
            fy = height/(2*np.tan(np.deg2rad(model.cam_fovy[cid])/2))
            k = np.array([[fy,0,(width-1)/2],[0,fy,(height-1)/2],[0,0,1.]])
            transform = np.eye(4)
            transform[:3,:3] = data.cam_xmat[cid].reshape(3,3) @ np.diag([1,-1,-1])
            transform[:3,3] = data.cam_xpos[cid]
            if self.config.mode=='wrist' and self._robot is not None:
                ee=self._robot.get_ee_pose();base=np.eye(4);base[:3,:3]=ee.rotation;base[:3,3]=ee.position
                transform=base@self.ee_to_camera
        frame=CameraFrame(rgb,depth,k,transform,time.monotonic(),f"{name}:{self.index}")
        # Detail frames are used for semantic review only. The 640x480
        # observation remains the source of depth geometry and robot masks.
        if self._robot is not None and not detail:
            from perception.pointcloud import pixel_to_world
            v,u=np.indices(depth.shape)
            valid=np.isfinite(depth)&(depth>.025)&(depth<3)
            robot_mask=np.zeros(depth.shape,bool)
            points=pixel_to_world(np.column_stack((u[valid],v[valid])),depth[valid],k,transform)
            # The perception foreground excludes the calibrated table plane.
            # Avoid testing hundreds of thousands of floor/background pixels
            # against every robot mesh on each closed-loop observation.
            relevant=(points[:,2]>.009)&(points[:,2]<self._sim.config.safety.workspace[2][1])
            own=np.zeros(len(points),bool)
            own[relevant]=self._robot.self_mask(points[relevant])
            robot_mask[valid]=own
            frame=CameraFrame(rgb,depth,k,transform,frame.timestamp,frame.view_id,robot_mask)
        return frame

    def capture_detail(self):
        return self.capture(detail=True)
