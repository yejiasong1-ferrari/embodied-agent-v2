class CollisionChecker:
    """Robot kinematics plus the perceived world, never a simulator scene clone."""
    def __init__(self,robot): self.robot=robot
    def check_collision(self,joints): return self.robot.check_collision(joints)
    def validate_trajectory(self,positions): return all(self.check_collision(q) is None for q in positions)
