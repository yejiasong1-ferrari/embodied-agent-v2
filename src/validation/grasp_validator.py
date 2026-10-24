import numpy as np


class GraspValidator:
    def __init__(self,config): self.config=config
    def contact(self,robot):
        c=robot.get_contact_state(); gap=robot.get_gripper_state()["width"]
        return min(c.left_force,c.right_force)>=self.config.safety.min_grasp_force_n and gap>.002, {"left_force":c.left_force,"right_force":c.right_force,"width":gap}
    def stable_contact(self,samples):
        threshold=self.config.safety.min_grasp_force_n
        bilateral=[min(row['left_force'],row['right_force'])>=threshold for row in samples]
        widths=[row['width'] for row in samples]
        stable=bool(len(samples)>=3 and all(bilateral) and min(widths)>.002
                    and max(widths)-min(widths)<.002)
        return stable,{"samples":samples,"bilateral_samples":sum(bilateral),
                       "jaw_width_drift_m":max(widths)-min(widths) if widths else None,
                       "stable":stable}
    def probe_lifted(self,before,after,ee_delta):
        moved=after.position-before
        slip=float(np.linalg.norm(moved-ee_delta))
        ok=bool(after.visible and moved[2]>.002 and ee_delta[2]>.003
                and slip<min(self.config.safety.slip_distance_m,.008))
        return ok,{"object_motion":moved.tolist(),"gripper_motion":ee_delta.tolist(),
                   "relative_slip_m":slip,"visible":after.visible}
    def lifted(self,before,after,ee_delta):
        moved=after.position-before
        ok=after.visible and moved[2]>.008 and np.linalg.norm(moved-ee_delta)<self.config.safety.slip_distance_m
        return bool(ok),{"object_motion":moved.tolist(),"gripper_motion":ee_delta.tolist(),"visible":after.visible}
