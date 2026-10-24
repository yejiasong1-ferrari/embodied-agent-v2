from dataclasses import asdict,is_dataclass
from datetime import datetime,timezone
import json
from pathlib import Path
import time
import uuid
import numpy as np


def serializable(value):
    if isinstance(value,np.ndarray): return value.tolist()
    if isinstance(value,np.generic): return value.item()
    if hasattr(value,"model_dump"): return value.model_dump()
    if is_dataclass(value): return asdict(value)
    if isinstance(value,set): return sorted(value)
    raise TypeError(type(value).__name__)


class TaskLog:
    def __init__(self,root,command,config,event_sink=None):
        self.task_id=datetime.now(timezone.utc).strftime("task_%Y%m%dT%H%M%S_")+uuid.uuid4().hex[:8]
        self.path=Path(root)/self.task_id
        self.path.mkdir(parents=True,exist_ok=False)
        (self.path/"images").mkdir(); (self.path/"world_states").mkdir()
        self.started=time.monotonic()
        self.events=[]
        self.event_sink=event_sink
        self.write("task.json",{"task_id":self.task_id,"user_command":command,"config":config.model_dump()})

    def write(self,name,value):
        (self.path/name).write_text(json.dumps(value,ensure_ascii=False,indent=2,default=serializable))

    def event(self,kind,data):
        row={"sequence":len(self.events),"elapsed_s":time.monotonic()-self.started,"type":kind,"data":data}
        self.events.append(row)
        with (self.path/"events.jsonl").open("a") as stream:
            stream.write(json.dumps(row,ensure_ascii=False,default=serializable)+"\n")
        if self.event_sink is not None:
            self.event_sink(row)

    def world(self,world):
        self.write(f"world_states/{world.revision:05d}.json",world.to_dict())
