import re
from llm.schemas import Goal,Reference


COLORS={"红":"red","red":"red","蓝":"blue","blue":"blue","绿":"green","green":"green","黄":"yellow","yellow":"yellow","白":"white","white":"white","紫":"purple","purple":"purple","粉":"pink","pink":"pink","橙":"orange","orange":"orange","青":"cyan","cyan":"cyan"}
CATEGORIES={"杯":"cup","cup":"cup","mug":"cup","瓶":"bottle","bottle":"bottle","盘":"plate","plate":"plate","方块":"box","盒":"box","积木":"box","block":"box","box":"box","cube":"box"}
GRASP_REGION_WORDS={
    "handle":("把手","手柄","提手","握柄","handle"),
    "top":("顶部","顶面","上端","top"),
    "middle_body":("中部","中间","杯身","瓶身","middle"),
    "lower_body":("下部","底部","下端","lower","bottom"),
}


def user_requested_grasp_region(command,region):
    text=command.lower()
    return bool(re.search(r"抓|夹|握|grasp|grab|hold",text) and
                any(word in text for word in GRASP_REGION_WORDS[region]))


def reference(text):
    text=text.strip()
    def first(vocab):
        hits=[(text.lower().find(k),v) for k,v in vocab.items() if k in text.lower()]
        return min(hits)[1] if hits else None
    return Reference(description=text or "object",color=first(COLORS),category=first(CATEGORIES))


class TaskInterpreter:
    def __init__(self,client,log): self.client,self.log=client,log
    def understand(self,command):
        if self.client.online:
            try:
                goal=self.client.complete(Goal,"interpret",{"command":command})
                region=goal.manipulation.grasp_specific_region
                if region and not user_requested_grasp_region(command,region):
                    goal.manipulation.grasp_specific_region=None
                    self.log.event("unrequested_grasp_region_ignored",{"region":region})
                from spatial.constraint_parser import explicit_distances
                requested=explicit_distances(command)
                accepted=[]
                for constraint in goal.spatial_constraints:
                    value=constraint.distance_m
                    if value is not None and not any(abs(value-item)<.001 for item in requested):
                        self.log.event("unrequested_distance_ignored",{
                            "distance_m":value,"constraint_type":constraint.type})
                        if constraint.type!="relative_position":continue
                        constraint.distance_m=None
                    accepted.append(constraint)
                goal.spatial_constraints=accepted
                return goal
            except RuntimeError:
                if self.client.config.mode=="required": raise
                self.log.event("llm_fallback",{"purpose":"interpret","backend":"local_grammar"})
        elif self.client.config.mode=="required": raise RuntimeError("required LLM mode needs an API key")
        self.log.event("interpret_backend",{"backend":"local_grammar","llm_used":False})
        lowered=command.lower()
        if ("挡住" in command or "blocking" in lowered) and ("拿出来" in command or "extract" in lowered):
            target=reference(command)
            return Goal(goal="extract",object=target)
        from spatial.constraint_parser import parse_local
        return parse_local(command,reference)
