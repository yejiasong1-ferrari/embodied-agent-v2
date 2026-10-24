class IKSolver:
    def __init__(self,robot): self.robot=robot
    def solve(self,pose,seed=None):
        return self.robot.solve_ik(pose,self.robot.get_joint_positions() if seed is None else seed)
