"""Event-sourced UI state; no estimated task success or fabricated attempts."""
from dataclasses import dataclass, field

@dataclass
class MissionState:
    state: str = 'IDLE'
    plan: list = field(default_factory=list)
    completed: set = field(default_factory=set)
    current: int | None = None
    goal: dict = field(default_factory=dict)
    context: dict = field(default_factory=dict)
    recoveries: list = field(default_factory=list)
    recovering: bool = False
    recovery_resolved: bool = False
    retry_skill: str | None = None
    validation: dict = field(default_factory=dict)
    result: dict = field(default_factory=dict)
    llm_calls: int = 0
    backend: str = 'Unavailable'
    localization: dict = field(default_factory=dict)
    grasp_strategy: dict = field(default_factory=dict)
    spatial: dict = field(default_factory=dict)
    corrections: list = field(default_factory=list)
    seen_states: set = field(default_factory=set)

    @property
    def complete(self):
        return self.state=='SUCCESS' and self.validation.get('accepted') is True

    def apply(self,event):
        kind,d=event['type'],event['data']
        if kind=='interpretation':self.goal=d
        elif kind=='plan':
            self.plan=[x['skill'] for x in d['steps']]
            self.completed.clear();self.current=None;self.validation={}
        elif kind=='state':
            self.state=d['state'];skill=d.get('skill')
            self.seen_states.add(self.state)
            if self.state in {'EXECUTE','VERIFY'} and skill in self.plan:
                matching=[i for i,s in enumerate(self.plan) if s==skill]
                self.current=next((i for i in matching if i not in self.completed),matching[0])
                self.completed={i for i in self.completed if i<self.current}
                if self.recovering and self.retry_skill is None:self.retry_skill=skill
            elif self.state=='RECOVER':
                self.recovering=True;self.recovery_resolved=False;self.retry_skill=None;self.validation={}
            elif self.state=='FAILED':self.recovering=False
        elif kind=='skill' and d.get('success'):
            if self.current is not None and self.plan[self.current]==d.get('name'):
                self.completed.add(self.current)
            if self.recovering and d.get('name')==self.retry_skill:
                self.recovering=False;self.recovery_resolved=True
        elif kind=='recovery':self.recoveries.append(d)
        elif kind=='final_validation':self.validation=d
        elif kind=='result':self.result=d
        elif kind=='llm_request':self.llm_calls=d['call'];self.backend=d['model']
        elif kind=='llm':self.backend=d['model']
        elif kind in {'llm_fallback','interpret_backend'}:self.backend='本地规则（当前步骤）'
        elif kind=='context':self.context=d
        elif kind=='localization':self.localization[d['object_id']]=d
        elif kind=='adaptive_grasp':self.grasp_strategy=d
        elif kind=='relation_validation':self.spatial=d
        elif kind=='placement_correction':self.corrections.append(d)
