from .observation import ObserveSkill,ScanSceneSkill,MoveCameraSkill,FindObjectSkill
from .manipulation import ApproachSkill,GraspSkill,VerifyGraspSkill,LiftSkill,MoveObjectSkill,PlaceSkill,RetreatSkill,VerifyTaskSkill


class SkillRegistry:
    def __init__(self):
        self.skills={}
        for name,cls in (("OBSERVE_SCENE",ObserveSkill),("SCAN_SCENE",ScanSceneSkill),("MOVE_CAMERA",MoveCameraSkill),
                         ("FIND_OBJECT",FindObjectSkill),("APPROACH_OBJECT",ApproachSkill),("GRASP_OBJECT",GraspSkill),
                         ("VERIFY_GRASP",VerifyGraspSkill),("LIFT_OBJECT",LiftSkill),("MOVE_TO_TARGET",MoveObjectSkill),
                         ("PLACE_OBJECT",PlaceSkill),("RETREAT",RetreatSkill),("VERIFY_TASK",VerifyTaskSkill)):
            self.register(name,cls())
    def register(self,name,skill):
        if name in self.skills: raise ValueError(f"duplicate skill {name}")
        self.skills[name]=skill
    def execute(self,name,context):
        if name not in self.skills: raise ValueError(f"unauthorized skill {name}")
        skill=self.skills[name]; context.current_skill=name
        skill.check_preconditions(context)
        result=skill.execute(context)
        skill.monitor(context)
        return skill.validate(context,result)
