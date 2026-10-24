#!/usr/bin/env python3
"""Randomized actual wrist observations; score static targets only after stopping.
No simulator labels or positions are supplied to perception or local motion.
"""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import argparse,json,time
import numpy as np
from config import AgentConfig
from simulation.environment import Simulation
from simulation.mujoco_camera import MujocoCamera
from robot.mujoco_robot import MuJoCoRobot
from evaluation.ground_truth import SimulationGroundTruth
from llm.llm_client import LLMClient
from llm.schemas import Goal,Reference
from agent.task_log import TaskLog
from perception.active_perception import ActivePerception
from world.world_model import WorldModel
from validation.safety_monitor import SafetyMonitor
from manipulation.motion_planner import MotionPlanner
from manipulation.trajectory_executor import TrajectoryExecutor
from manipulation.grasp_planner import GraspPlanner
from skills.skill_base import SkillContext
from skills.localization import localize_target


def main():
 parser=argparse.ArgumentParser();parser.add_argument('--seeds',type=int,nargs='+',default=[23,41,67]);parser.add_argument('--online',action='store_true');parser.add_argument('--output',type=Path,default=Path('outputs/upgrade_checks/localization_benchmark.json'));args=parser.parse_args()
 rows=[]
 for seed in args.seeds:
  cfg=AgentConfig();cfg.llm.mode='required' if args.online else 'offline';cfg.localization.grounding_mode='auto' if args.online else 'local'
  cfg.camera.wrist_scan_angles=[x+float(np.random.default_rng(seed).uniform(-.08,.08)) for x in cfg.camera.wrist_scan_angles]
  log=TaskLog(cfg.log_root,'randomized localization benchmark',cfg);world=WorldModel(cfg.perception)
  goal=Goal(goal='place_left',object=Reference(description='蓝色盒子'),target=Reference(description='绿色积木'))
  row={'seed':seed,'log_path':str(log.path),'coarse_found':False,'precise_found':False,'failure':None}
  coarse=precise=None
  with Simulation(cfg,seed,'randomized') as sim:
   robot=MuJoCoRobot(sim);camera=MujocoCamera(sim,robot);client=LLMClient(cfg.llm,log.event)
   perception=ActivePerception(camera,world,cfg,client,log);safety=SafetyMonitor(robot,cfg)
   c=SkillContext(robot,perception,world,goal,cfg,MotionPlanner(robot,cfg),TrajectoryExecutor(robot,safety,cfg,log),GraspPlanner(cfg),safety,log,None)
   try:
    perception.scan([goal.object,goal.target]);source=world.resolve(goal.object);target=world.resolve(goal.target)
    c.source,c.target,c.source_id,c.target_id=source,target,source.id,target.id
    world.memory.mission_active=True;coarse=source.position.copy();row['coarse_found']=True
    camera.set_manipulation_active(True);obj=localize_target(c,source.id);precise=obj.position.copy();row['precise_found']=True
    row['bbox']=list(obj.bbox);row['bbox_in_image']=bool(0<=obj.bbox[0]<obj.bbox[2]<=cfg.camera.width and 0<=obj.bbox[1]<obj.bbox[3]<=cfg.camera.height)
   except Exception as exc:row['failure']=str(exc)
   robot.stop();sim.execution_finished=True;truth=SimulationGroundTruth(sim);low,high=truth.bounds('blue_box');center=(low+high)/2
   row.update(coarse_error_m=None if coarse is None else float(np.linalg.norm(coarse-center)),precise_error_m=None if precise is None else float(np.linalg.norm(precise-center)),
              observations=perception.observation_count,vlm_calls=perception.grounder.calls,unexpected_contacts=max(safety.collision_count,sim.collision_count))
  rows.append(row);print(json.dumps(row,ensure_ascii=False),flush=True)
  def mean(key):
   values=[r[key] for r in rows if r.get(key) is not None];return float(np.mean(values)) if values else None
  summary={'trials':len(rows),'visual_detection_success_rate':mean('coarse_found'),'precise_lock_success_rate':mean('precise_found'),
    'coarse_localization_error_m':mean('coarse_error_m'),'precise_localization_error_m':mean('precise_error_m'),'scope':'actual randomized wrist perception and approach; no grasp executed'}
  args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps({'summary':summary,'trials':rows},ensure_ascii=False,indent=2))
if __name__=='__main__':main()
