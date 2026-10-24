from llm.schemas import RecoveryDecision


class Replanner:
    def __init__(self,client): self.client=client
    def reason(self,goal,world,failure,history,evidence,safety_state):
        return self.client.complete(RecoveryDecision,"recover",{"task":goal.model_dump(),"world":world.to_dict(),
                 "failure":failure,"attempt_history":history,"evidence":evidence,"safety_state":safety_state})
