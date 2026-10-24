from enum import Enum
import re
from typing import Literal
from pydantic import BaseModel,ConfigDict,Field,model_validator,field_validator


class StrictSchema(BaseModel):
    model_config=ConfigDict(extra="forbid")


class Reference(StrictSchema):
    description: str = Field(min_length=1,max_length=200)
    color: str | None = None
    category: str | None = None

    @field_validator("color", "category", mode="before")
    @classmethod
    def normalize_vocabulary(cls,value):
        if value is None: return None
        aliases={"红色":"red","红":"red","蓝色":"blue","蓝":"blue","绿色":"green","绿":"green",
                 "黄色":"yellow","黄":"yellow","白色":"white","紫色":"purple","橙色":"orange",
                 "青色":"cyan","灰色":"gray","粉色":"pink","杯子":"cup","杯":"cup","mug":"cup",
                 "盘子":"plate","盘":"plate","盒子":"box","盒":"box","方块":"box","积木":"box","block":"box",
                 "cube":"box","瓶子":"bottle","瓶":"bottle","物体":"object","障碍物":"obstacle"}
        value=value.strip().lower()
        return aliases.get(value,value)

    @model_validator(mode="after")
    def resolve_description_facets(self):
        # Structured LLM output may put "red cup" only in description.
        # Resolve vocabulary facets here; identity still comes solely from
        # visual WorldModel candidates and remains ambiguous when appropriate.
        phrase=self.description.lower()
        def contains(word):
            return word in phrase if re.search(r"[\u4e00-\u9fff]",word) else bool(re.search(r"\b"+re.escape(word)+r"\b",phrase))
        colors=(("red","红色","红"),("blue","蓝色","蓝"),("green","绿色","绿"),
                ("yellow","黄色","黄"),("white","白色","白"),("purple","紫色","紫"),
                ("orange","橙色","橙"),("cyan","青色","青"),("pink","粉色","粉"),("gray","灰色","灰"))
        categories=(("cup","杯子","杯","mug"),("plate","盘子","盘"),("box","盒子","盒","积木","方块","block","cube"),
                    ("bottle","瓶子","瓶"),("region","空桌面","free table region"))
        if self.color is None:
            self.color=next((words[0] for words in colors if any(contains(w) for w in words)),None)
        if self.category is None:
            self.category=next((words[0] for words in categories if any(contains(w) for w in words)),None)
        return self


Relation = Literal["left_of","right_of","upper_of","lower_of","front_of","behind",
                   "above","below","on","inside","near","between","centered_with"]
ManipulationConstraint = Literal["avoid_collision","keep_object_upright","keep_upright","fragile",
    "avoid_top_contact","prefer_handle","avoid_squeezing","orientation_preservation","no_sliding"]


class ManipulationConstraints(StrictSchema):
    explicit: list[ManipulationConstraint] = Field(default_factory=list)
    inferred: list[ManipulationConstraint] = Field(default_factory=list)
    grasp_specific_region: Literal["top","middle_body","lower_body","handle"] | None = None


class SpatialConstraint(StrictSchema):
    type: Literal["relative_position","min_distance","max_distance","distance_to"] = "relative_position"
    relation: Relation | None = None
    reference: Reference | None = None
    references: list[Reference] = Field(default_factory=list,max_length=2)
    distance_m: float | None = Field(default=None,ge=0,le=2)
    distance_semantics: Literal["edge_gap","center_distance","surface_offset"] = "edge_gap"
    reference_frame: Literal["operator","robot_base","world","object"] = "operator"
    tolerance_m: float | None = Field(default=None,gt=0,le=.05)

    @model_validator(mode="after")
    def consistency(self):
        if self.type=="relative_position" and self.relation is None:raise ValueError("relation required")
        if self.type!="relative_position" and self.distance_m is None:raise ValueError("distance required")
        if self.relation=="between" and len(self.references)!=2:raise ValueError("between requires two references")
        return self


class Goal(StrictSchema):
    goal: Literal["place_on","place_near","place_left","place_right","place_front","place_behind","place_inside","place_spatial","clear_then_place","extract"]
    object: Reference
    target: Reference | None = None
    constraints: list[ManipulationConstraint] = Field(default_factory=lambda:["avoid_collision"])
    manipulation: ManipulationConstraints = Field(default_factory=ManipulationConstraints)
    spatial_constraints: list[SpatialConstraint] = Field(default_factory=list,max_length=8)

    @model_validator(mode="after")
    def check_target(self):
        if self.goal not in {"extract"} and self.target is None:
            raise ValueError("placement requires an explicit destination")
        if "avoid_collision" not in self.constraints:raise ValueError("collision constraint is mandatory")
        if self.goal=="place_spatial" and not self.spatial_constraints:raise ValueError("spatial constraints required")
        return self

    @property
    def relation(self):
        primary=next((s for s in self.spatial_constraints if s.type=="relative_position"),None)
        if primary:return primary.relation
        return {"place_on":"on","place_near":"near","place_left":"left_of","place_right":"right_of",
                "place_front":"front_of","place_behind":"behind","place_inside":"inside","clear_then_place":"on","extract":"near"}.get(self.goal,"near")

    @property
    def hard_constraints(self):
        return set(self.constraints)|set(self.manipulation.explicit)

    @property
    def references(self):
        rows=([self.target] if self.target else [])
        for constraint in self.spatial_constraints:
            rows+=([constraint.reference] if constraint.reference else [])+constraint.references
        unique={r.description:r for r in rows}
        return list(unique.values())


class PlanStep(StrictSchema):
    skill: Literal["OBSERVE_SCENE","FIND_OBJECT","APPROACH_OBJECT","GRASP_OBJECT","VERIFY_GRASP","LIFT_OBJECT","MOVE_TO_TARGET","PLACE_OBJECT","RETREAT","VERIFY_TASK","SCAN_SCENE","MOVE_CAMERA"]


class SymbolicPlan(StrictSchema):
    steps: list[PlanStep] = Field(min_length=1,max_length=30)
    reason: str = ""


class RecoveryDecision(StrictSchema):
    action: Literal["REOBSERVE","CHANGE_GRASP","CHANGE_APPROACH","REPLAN","ABORT"]
    reason: str = Field(min_length=1,max_length=600)


class VisualBox(StrictSchema):
    label: str
    color: str
    box: tuple[float,float,float,float]
    confidence: float = Field(ge=0,le=1)
    @model_validator(mode="after")
    def bounds(self):
        x1,y1,x2,y2=self.box
        if not (0<=x1<x2<=1 and 0<=y1<y2<=1): raise ValueError("invalid normalized box")
        return self


class VisualDetections(StrictSchema):
    objects: list[VisualBox]


class GroundedTarget(StrictSchema):
    found: bool
    label: str
    color: str | None = None
    category: str | None = None
    bbox: tuple[float,float,float,float] | None = None
    coordinate_space: Literal["normalized","pixel"] = "normalized"
    confidence: float = Field(ge=0,le=1)
    description: str = ""

    @model_validator(mode="after")
    def bounds(self):
        if self.found:
            if self.bbox is None:raise ValueError("found requires a box")
            x1,y1,x2,y2=self.bbox
            if not (0<=x1<x2 and 0<=y1<y2):raise ValueError("invalid bbox")
            if self.coordinate_space=="normalized" and (x2>1 or y2>1):raise ValueError("normalized bbox out of range")
        elif self.bbox is not None:raise ValueError("absent target must not have bbox")
        return self


class CandidateJudgement(StrictSchema):
    candidate_id: str
    verdict: Literal["yes", "no", "uncertain"]
    confidence: float = Field(ge=0, le=1)
    evidence: str = Field(min_length=1, max_length=300)


class VisualDisambiguation(StrictSchema):
    candidates: list[CandidateJudgement] = Field(min_length=1, max_length=8)

    @model_validator(mode="after")
    def unique_ids(self):
        ids=[row.candidate_id for row in self.candidates]
        if len(ids)!=len(set(ids)):raise ValueError("duplicate candidate IDs")
        return self


class GraspStrategyDecision(StrictSchema):
    selected_candidate_id: str
    strategy: Literal["top_grasp","side_grasp","diagonal_grasp"]
    preferred_region: Literal["top","middle_body","lower_body","handle"]
    approach_direction: Literal["top","left","right","front","back"]
    constraints: list[ManipulationConstraint] = Field(default_factory=list)
    reason_code: Literal["TOP_FEASIBLE","TOP_TOO_WIDE_LOWER_REGION_GRASPABLE","CLEARANCE","TASK_CONSTRAINT","ALTERNATE_AFTER_FAILURE","SIMILAR_CANDIDATES"]
