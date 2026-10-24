"""Proprioception/actuation boundary; never exposes an object pose or name."""
import mujoco
import numpy as np
from joint_control import PandaController, step_with_contact_friction
from simulation.planning_backend import PlanningBackend
from .robot_interface import Pose,ContactState,RobotInterface


class MuJoCoRobot(RobotInterface):
    def __init__(self, simulation, step_observer=None):
        self._sim=simulation
        self.config=simulation.config
        self._control=PandaController(simulation.model,simulation.data,data_lock=simulation.lock,one_sided_friction=True)
        self._planning=PlanningBackend(self.config.safety.collision_margin_m)
        self.joint_limits=(self._planning.limits[:,0].copy(),self._planning.limits[:,1].copy())
        self._addresses=[int(simulation.model.joint(f"joint{i}").qposadr[0]) for i in range(1,8)]
        self._hand=simulation.model.body("hand").id
        self._fingers={simulation.model.body("left_finger").id:"left",simulation.model.body("right_finger").id:"right"}
        self._robot_ids={i for i in range(1,simulation.model.nbody)
                         if (mujoco.mj_id2name(simulation.model,mujoco.mjtObj.mjOBJ_BODY,i) or "").startswith("link") or i==self._hand or i in self._fingers}
        self.last_error=None
        self._collision_active=False
        self._step_observer=step_observer
        self._bilateral_grasp_verified=False

    def get_joint_positions(self): return self._sim.data.qpos[self._addresses].copy()
    def get_ee_pose(self):
        return Pose(self._sim.data.xpos[self._hand].copy(),self._sim.data.xmat[self._hand].reshape(3,3).copy())
    def get_gripper_state(self):
        gap=sum(float(self._sim.data.qpos[int(self._sim.model.joint(n).qposadr[0])]) for n in ("finger_joint1","finger_joint2"))
        return {"width":gap,"command":float(self._sim.data.ctrl[-1])}

    def get_contact_state(self):
        model,data=self._sim.model,self._sim.data
        forces={"left":0.,"right":0.}; unexpected=False
        for i in range(data.ncon):
            c=data.contact[i]
            if c.dist>.001: continue
            a,b=(int(model.geom_bodyid[int(g)]) for g in (c.geom1,c.geom2))
            if a!=b and a in self._robot_ids and b in self._robot_ids and c.dist<-.001:
                unexpected=True
            for robot,other in ((a,b),(b,a)):
                if robot in self._fingers and other not in self._robot_ids:
                    force=np.zeros(6); mujoco.mj_contactForce(model,data,i,force)
                    forces[self._fingers[robot]]+=abs(float(force[0]))
                elif robot in self._robot_ids and other not in self._robot_ids and c.dist<-.001:
                    if not (other==0 and robot==model.body("link0").id): unexpected=True
        if unexpected and not self._collision_active: self._sim.collision_count+=1
        self._collision_active=unexpected
        return ContactState(forces["left"],forces["right"],unexpected)

    def _monitor(self,monitor):
        if self._sim.stopped: return "user_stop"
        if monitor:
            reason=monitor()
            if reason: self.last_error=reason; return reason
        if self._step_observer is not None:
            self._step_observer()
        return None

    def execute_trajectory(self,positions,times,monitor=None):
        if self._sim.stopped: return False
        model,data=self._sim.model,self._sim.data
        for i,q in enumerate(positions):
            duration=model.opt.timestep if i==0 else times[i]-times[i-1]
            for _ in range(max(1,int(np.ceil(duration/model.opt.timestep)))):
                with self._sim.lock:
                    data.ctrl[:7]=q
                    if self._bilateral_grasp_verified:mujoco.mj_step(model,data)
                    else:step_with_contact_friction(model, data)
                reason=self._monitor(monitor)
                if reason is None and np.max(np.abs(self.get_joint_positions()-np.asarray(q)))>self.config.safety.joint_tracking_tolerance_rad:
                    reason="trajectory_deviation"
                if reason:
                    self.last_error=reason; self.stop(); return False
        return self.hold(self.config.robot.physics_settle_s,monitor)

    def move_joint(self,joints,monitor=None):
        from manipulation.trajectory_executor import retime
        q=np.asarray(joints)
        if self.check_collision(q): return False
        positions,times=retime([self.get_joint_positions(),q],self.config.safety.max_velocity)
        if any(self.check_collision(s) for s in positions): return False
        return self.execute_trajectory(positions,times,monitor)

    def move_cartesian(self,pose,monitor=None):
        for solution in self.solve_ik(pose,self.get_joint_positions()):
            if self.move_joint(solution,monitor): return True
        return False

    def open_gripper(self,monitor=None):
        self._bilateral_grasp_verified=False
        self._control.one_sided_friction=True
        return self._control.grasp_to(255,duration=1.,monitor=lambda:self._monitor(monitor))
    def confirm_grasp(self):
        # A grasp must first pass bilateral force validation. During carry,
        # ordinary physical friction can bridge brief one-pad contact without
        # turning every such step into a zero-friction drop. Visual slip and
        # contact loss monitors remain active; no rigid attachment is added.
        self._bilateral_grasp_verified=True
        self._control.one_sided_friction=False
    def close_gripper(self,width,monitor=None,target_force_n=None):
        # Pad thickness is robot calibration, independent of the target type.
        desired=np.clip(width-self.config.robot.gripper_squeeze_m,0,.08)
        command=float(np.clip(desired/.08*255,0,255))
        if not self._control.grasp_to(command,duration=.6,monitor=lambda:self._monitor(monitor)): return False
        # Regulate against measured finger forces, not a scene object's mass
        # or a fixed close command. Stop after a bounded closure search.
        target_force_n=self.config.robot.gripper_force_target_n if target_force_n is None else target_force_n
        for _ in range(16):
            contact=self.get_contact_state()
            if min(contact.left_force,contact.right_force)>=target_force_n: return True
            if command<=0: break
            command=max(0,command-3)
            if not self._control.grasp_to(command,duration=.12,monitor=lambda:self._monitor(monitor)): return False
        return True
    def hold(self,duration,monitor=None):
        return self._control.hold(duration,monitor=lambda:self._monitor(monitor))
    def stop(self):
        self._sim.stopped=True
        with self._sim.lock:
            self._sim.data.ctrl[:7]=self.get_joint_positions()
    def resume(self): self._sim.stopped=False
    def solve_ik(self,pose,seed): return self._planning.solve(pose,seed)
    def forward_kinematics(self,joints): return self._planning.forward(joints)
    def configure_collision_world(self,world,target_id=None,payload=None,support_id=None):
        self._planning.gripper_width=self.get_gripper_state()["width"]
        self._planning.configure(world,target_id,payload,support_id)
    def check_collision(self,joints): return self._planning.collision(joints)
    def self_mask(self,points): return self._planning.self_mask(points,self.get_joint_positions())
