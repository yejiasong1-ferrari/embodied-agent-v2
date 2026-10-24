import numpy as np


class SafetyMonitor:
    def __init__(self,robot,config,cancelled=lambda:False):
        self.robot,self.config=robot,config
        self.last_failure=None; self.allow_finger_contact=False
        self._lost_count=0
        self.cancelled=cancelled
        self.collision_count=0
        self._contact_violation=False
        self.carry_reference_rotation=None
        self.force_limit_override=None

    def check(self,carrying=False):
        if self.cancelled():
            self.last_failure="USER_STOP"
            return self.last_failure
        q=self.robot.get_joint_positions(); lower,upper=self.robot.joint_limits
        contact=self.robot.get_contact_state(); ee=self.robot.get_ee_pose().position
        bounds=np.asarray(self.config.safety.workspace)
        reason=None
        if np.any(q<lower-.005) or np.any(q>upper+.005): reason="joint_limit"
        elif np.any(ee<bounds[:,0]) or np.any(ee>bounds[:,1]): reason="workspace_limit"
        elif contact.unexpected: reason="unexpected_contact"
        elif max(contact.left_force,contact.right_force)>min(self.config.safety.contact_force_limit_n,self.force_limit_override if self.force_limit_override is not None else float("inf")): reason="force_limit"
        elif not self.allow_finger_contact and max(contact.left_force,contact.right_force)>self.config.safety.min_grasp_force_n: reason="unexpected_finger_contact"
        if carrying:
            rotation=self.robot.get_ee_pose().rotation
            # Physics can settle a fraction of a degree past a planned 12°
            # limit. Keep the planner at 12° and allow measurement hysteresis
            # so the next safe corrective motion is not blocked at its start.
            upright=rotation[2,2]<-np.cos(np.deg2rad(12.25)) if self.carry_reference_rotation is None else (rotation@self.carry_reference_rotation.T)[2,2]>np.cos(np.deg2rad(12.25))
            if not upright and reason is None:reason="carrying_tilt_limit"
            self._lost_count=0 if contact.bilateral else self._lost_count+1
            # Contact can alternate between pads for a few physics frames
            # during a joint transition; the next RGB-D waypoint also checks
            # attachment displacement before carrying on.
            if self._lost_count>40 and reason is None: reason="OBJECT_SLIPPED"
        else: self._lost_count=0
        contact_violation=reason in {"unexpected_contact","unexpected_finger_contact"}
        if contact_violation and not self._contact_violation: self.collision_count+=1
        self._contact_violation=contact_violation
        if reason: self.last_failure=reason
        return reason
