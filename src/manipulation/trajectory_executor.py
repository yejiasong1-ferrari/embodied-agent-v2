import numpy as np


def retime(waypoints,max_velocity,sample_period=.02):
    """Same quintic sampling used by V2; validate exactly what is executed."""
    positions=[]; times=[]; elapsed=0.
    for start,end in zip(waypoints,waypoints[1:]):
        start,end=np.asarray(start),np.asarray(end)
        duration=max(.15,1.875*float(np.max(np.abs(end-start)))/max_velocity)
        count=max(2,int(np.ceil(duration/sample_period))+1)
        for i,alpha in enumerate(np.linspace(0,1,count)):
            if positions and i==0: continue
            smooth=alpha**3*(alpha*(6*alpha-15)+10)
            positions.append(start+smooth*(end-start)); times.append(elapsed+alpha*duration)
        elapsed+=duration
    return positions,times


class TrajectoryExecutor:
    def __init__(self,robot,safety,config,log):
        self.robot,self.safety,self.config,self.log=robot,safety,config,log

    def execute(self,waypoints,carrying=False,cartesian_speed_limit=None):
        velocity=self.config.safety.carrying_velocity if carrying else self.config.safety.max_velocity
        positions,times=retime(waypoints,velocity)
        if cartesian_speed_limit is not None:
            points=np.asarray([self.robot.forward_kinematics(q).position for q in positions])
            speeds=np.linalg.norm(np.diff(points,axis=0),axis=1)/np.maximum(np.diff(times),1e-9)
            scale=max(1.,float(speeds.max(initial=0))/cartesian_speed_limit)
            times=[t*scale for t in times]
        for q in positions:
            reason=self.robot.check_collision(q)
            if reason: return False,reason
        self.log.event("trajectory",{"joints":[q.tolist() for q in waypoints],"cartesian":[self.robot.forward_kinematics(q).position.tolist() for q in waypoints],"duration_s":times[-1],"samples":len(positions),"validated":True})
        ok=self.robot.execute_trajectory(positions,times,lambda:self.safety.check(carrying=carrying))
        return ok,None if ok else self.safety.last_failure or getattr(self.robot,"last_error",None) or "trajectory_stopped"
