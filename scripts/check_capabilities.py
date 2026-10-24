#!/usr/bin/env python3
"""Actual wrist RGB-D execution; ground truth is scored only after the run.

The optional release-bias fixture changes the placement request only inside
PlaceSkill, restores the real goal before validation, and never edits object
poses. It exercises physical push/regrasp recovery after a real wrong release.
"""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import argparse,json
from unittest.mock import patch
import numpy as np
import mujoco
from config import AgentConfig
from simulation.environment import Simulation
from simulation.mujoco_camera import MujocoCamera
from robot.mujoco_robot import MuJoCoRobot
from agent.embodied_agent import EmbodiedAgent
from evaluation.ground_truth import SimulationGroundTruth
from skills.manipulation import PlaceSkill


def main():
 p=argparse.ArgumentParser();p.add_argument('--scenario',default='default');p.add_argument('--seed',type=int)
 p.add_argument('--instruction',default='把蓝色盒子放到绿色积木左边10cm');p.add_argument('--online',action='store_true')
 p.add_argument('--release-bias',type=float,default=0.);p.add_argument('--output',type=Path,required=True);args=p.parse_args()
 cfg=AgentConfig();cfg.camera.mode='wrist';cfg.llm.mode='required' if args.online else 'offline'
 cfg.localization.grounding_mode='auto' if args.online else 'local'
 original=PlaceSkill.execute;injected=False
 def biased(skill,c):
  nonlocal injected
  goal=c.goal
  if args.release_bias and not injected:
   c.goal=goal.model_copy(deep=True);constraint=c.goal.spatial_constraints[0]
   constraint.distance_m=max(0.,constraint.distance_m+args.release_bias);injected=True
   c.log.event('benchmark_release_bias',{'distance_bias_m':args.release_bias,'fixture_only':True,'method':'physical placement to biased symbolic gap'})
  try:return original(skill,c)
  finally:c.goal=goal
 with Simulation(cfg,args.seed,args.scenario) as sim:
  robot=MuJoCoRobot(sim);camera=MujocoCamera(sim,robot);agent=EmbodiedAgent(robot,camera,cfg,
   event_sink=lambda state,data:print(state,json.dumps(data,ensure_ascii=False),flush=True))
  with patch.object(PlaceSkill,'execute',biased):result=agent.execute(args.instruction)
  sim.execution_finished=True;truth=SimulationGroundTruth(sim)
  low,high=truth.bounds('blue_box');rlo,rhi=truth.bounds('green_block')
  constraint=agent.context.goal.spatial_constraints[0] if agent.context and agent.context.goal.spatial_constraints else None
  relation=constraint.relation if constraint is not None else None
  actual_gap=float(rlo[0]-high[0] if relation=='left_of' else low[0]-rhi[0] if relation=='right_of' else float('nan'))
  requested_gap=None if constraint is None else constraint.distance_m
  tolerance=cfg.spatial.default_tolerance_m if constraint is None or constraint.tolerance_m is None else constraint.tolerance_m
  truth_pass=bool(requested_gap is not None and np.isfinite(actual_gap) and abs(actual_gap-requested_gap)<=tolerance)
  visual_constraints=result.metrics.get('final_spatial_constraints') or []
  visual_gap=next((x.get('actual_gap_m') for x in visual_constraints if x.get('relation')==relation),None)
  source=agent.context.source if agent.context else None
  contacts=[]
  for i,contact in enumerate(sim.data.contact[:sim.data.ncon]):
   force=np.zeros(6);mujoco.mj_contactForce(sim.model,sim.data,i,force)
   if force[0]<.1:continue
   contacts.append({'bodies':[sim.model.body(int(sim.model.geom_bodyid[g])).name for g in [contact.geom1,contact.geom2]],
                    'force_n':float(force[0]),'position':contact.pos.tolist()})
  row={'result':result.__dict__,'benchmark':{'scenario':args.scenario,'seed':args.seed,'release_bias_m':args.release_bias},
       'runtime_snapshot':{'joints':robot.get_joint_positions().tolist(),'ee_position':robot.get_ee_pose().position.tolist(),
          'ee_rotation':robot.get_ee_pose().rotation.tolist(),'gripper':robot.get_gripper_state(),
          'collision_at_measured_joints':robot.check_collision(robot.get_joint_positions()),
          'payload':agent.context.payload if agent.context else None,'world':agent.world.to_dict(),
          'source_id':agent.context.source_id if agent.context else None,'target_id':agent.context.target_id if agent.context else None},
       'after_execution_ground_truth':{'source_bounds':[low.tolist(),high.tolist()],'reference_bounds':[rlo.tolist(),rhi.tolist()],
         'left_edge_gap_m':float(rlo[0]-high[0]),'right_edge_gap_m':float(low[0]-rhi[0]),
         'requested_gap_m':requested_gap,'tolerance_m':tolerance,'relation':relation,'actual_gap_m':actual_gap,
         'within_tolerance':truth_pass,'visual_ground_truth_disagreement_m':None if visual_gap is None else abs(visual_gap-actual_gap),
         'source_position_error_m':None if source is None else float(np.linalg.norm(source.position-(low+high)/2)),
         'contacts':contacts}}
  encode=lambda value:value.tolist() if isinstance(value,np.ndarray) else value.item() if isinstance(value,np.generic) else str(value)
  args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(row,ensure_ascii=False,indent=2,default=encode))
  print(json.dumps(row,ensure_ascii=False,indent=2,default=encode))
 return 0 if result.success and truth_pass else 1

if __name__=='__main__':raise SystemExit(main())
