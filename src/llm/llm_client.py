import base64
import io
import json
import os
from pathlib import Path
from urllib.request import Request,urlopen
from PIL import Image


PROMPTS={
    "visual_grounding":"Identify the queried target only from the supplied RGB pixels. Use a tight xyxy bbox, explicitly normalized or pixel. Return found=false for absent/ambiguous targets. No world poses or hidden objects.",
    "grasp_strategy":"Select only an offered grasp candidate by ID using task semantics and geometry summary. Never invent coordinates, forces, poses or candidates. Safety and feasibility remain local. Return structured strategy and reason_code.",
    "interpret":"Interpret the user's manipulation goal. For relative placement fill spatial_constraints: explicit numeric distances in metres; default edge_gap and operator frame. right/left are +/-X, 上边/下边 are +/-Y (upper_of/lower_of), 上方/下方 are +/-Z (above/below). on means physical support. between requires two references; preserve every min/max distance constraint. Manipulation explicit constraints only if user asked; distinguish inferred. Ambiguous object-front uses object frame. No unspecified numeric distances. Resolve pronouns and descriptions. Use symbolic object descriptions, colours and categories. Do not output coordinates, joint angles, motor commands or forces. For 'move things blocking the cup then take it out', use extract.",
    "plan":"Propose only the authorized skills needed to reach the symbolic goal. Include, in this exact mandatory relative order: OBSERVE_SCENE, FIND_OBJECT, APPROACH_OBJECT, GRASP_OBJECT, VERIFY_GRASP, LIFT_OBJECT, MOVE_TO_TARGET, PLACE_OBJECT, RETREAT, VERIFY_TASK. You may insert SCAN_SCENE or MOVE_CAMERA where observations require it. Never output poses or controller parameters.",
    "recover":"Diagnose the supplied failure evidence and history. Choose only REOBSERVE, CHANGE_GRASP, CHANGE_APPROACH, REPLAN or ABORT. Never override a safety rejection or emit motion values.",
    "visual_detection":"Identify objects from image pixels. Output normalized boxes and visible semantic labels. Never infer simulator object names or hidden poses.",
    "visual_disambiguation":"Compare every numbered crop in the supplied real high-resolution camera image with the user's full instruction and target phrase. Determine which physical object the user refers to; everyday names can differ from an internal provisional category, which is not ground truth. Return one verdict per candidate_id. Use yes only when visible distinguishing evidence supports that referent. Colour alone, a roughly similar shape, or the absence of alternatives is insufficient. Mark unclear or occluded candidates uncertain. Do not infer simulator names, world coordinates or hidden geometry. Multiple yes and all uncertain are allowed when the image cannot decide. Return concrete visual evidence for each verdict.",
}
PROMPTS['interpret'] += ' For min_distance/max_distance/distance_to constraints, relation MUST be null (not min_distance or a phrase). Example additional constraint: {"type":"min_distance","relation":null,"reference":{"description":"blue box","color":"blue","category":"box"},"distance_m":0.08,"distance_semantics":"edge_gap","reference_frame":"operator"}.'
PROMPTS['interpret'] += ' Always include "avoid_collision" in constraints: it is mandatory local policy even when not explicitly requested. Never put it into manipulation.explicit unless the user said it.'
PROMPTS['interpret'] += ' Set manipulation.grasp_specific_region only when the user explicitly instructs a grasp at that part. A cup or mug does not imply a handle grasp; otherwise use null.'
PROMPTS['visual_grounding'] += ' Prefer coordinate_space="normalized" with decimal bbox coordinates between 0 and 1. Pixel coordinates MUST fit the supplied image_width and image_height; do not use 0..1000 coordinates as pixels.'


class LLMClient:
    def __init__(self,config,event_sink=None):
        self.config=config; self.calls=0; self.event_sink=event_sink or (lambda *args:None)
        env=Path(__file__).resolve().parents[2]/".env"
        if env.exists():
            for line in env.read_text().splitlines():
                if "=" in line and not line.strip().startswith("#"):
                    key,value=line.split("=",1)
                    os.environ.setdefault(key.strip(),value.strip().strip("\"'"))
        self.api_key=os.getenv("DASHSCOPE_API_KEY") or os.getenv("OPENAI_API_KEY")
        self.base_url=os.getenv("VLM_BASE_URL",config.base_url)
        self.model=os.getenv("VLM_MODEL",config.model)

    @property
    def online(self): return self.config.mode!="offline" and bool(self.api_key)

    def complete(self,schema,purpose,payload,image=None):
        if not self.online: raise RuntimeError("LLM is not configured/enabled")
        if self.calls>=self.config.max_calls: raise RuntimeError("LLM call budget exhausted")
        system=PROMPTS[purpose]+" Return only JSON matching this schema: "+json.dumps(schema.model_json_schema())
        content=[{"type":"text","text":json.dumps(payload,ensure_ascii=False)}]
        if image is not None:
            buffer=io.BytesIO(); picture=Image.fromarray(image)
            detail=purpose=='visual_disambiguation'
            picture.thumbnail((1280,960) if detail else (640,480))
            picture.save(buffer,format="JPEG",quality=88 if detail else 75)
            content.append({"type":"image_url","image_url":{"url":"data:image/jpeg;base64,"+base64.b64encode(buffer.getvalue()).decode()}})
        self.calls+=1
        self.event_sink("llm_request",{"purpose":purpose,"model":self.model,"call":self.calls,"payload":payload,"has_image":image is not None,
                                       "image_size":None if image is None else list(picture.size),
                                       "endpoint":self.base_url.rstrip("/")+"/chat/completions"})
        request=Request(self.base_url.rstrip("/")+"/chat/completions",data=json.dumps({"model":self.model,"messages":[{"role":"system","content":system},{"role":"user","content":content}],"response_format":{"type":"json_object"},"temperature":0}).encode(),headers={"Authorization":"Bearer "+self.api_key,"Content-Type":"application/json"})
        try:
            with urlopen(request,timeout=self.config.timeout_s) as response:
                raw=json.loads(response.read())["choices"][0]["message"]["content"]
            result=schema.model_validate_json(raw)
        except Exception as exc:
            self.event_sink("llm_error",{"purpose":purpose,"error_type":type(exc).__name__})
            if hasattr(exc,'errors'):
                self.event_sink('llm_schema_error',{'purpose':purpose,'errors':[
                    {'location':list(e['loc']),'type':e['type'],'message':e['msg']} for e in exc.errors()]})
            raise RuntimeError(f"{purpose} LLM call failed: {type(exc).__name__}") from exc
        self.event_sink("llm",{"purpose":purpose,"model":self.model,"response":result.model_dump()})
        return result
