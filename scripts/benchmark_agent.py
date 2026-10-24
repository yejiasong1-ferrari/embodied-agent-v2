#!/usr/bin/env python3
"""Randomized vision-only execution; privileged scoring happens afterwards."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
import argparse
import json
import numpy as np
from config import AgentConfig
from simulation.environment import Simulation
from simulation.mujoco_camera import MujocoCamera
from robot.mujoco_robot import MuJoCoRobot
from agent.embodied_agent import EmbodiedAgent
from evaluation.ground_truth import SimulationGroundTruth


CASES=(
    ("red_cup","plate","on","把红色杯子放到黄色盘子上","randomized"),
    ("blue_box","green_block","right_of","把蓝色盒子移到绿色积木右边","randomized"),
    ("red_cup","plate","on","把红色杯子放到黄色盘子上","near_obstacle"),
    ("blue_box","plate","on","Put the blue box on the yellow plate","outside_view"),
    ("red_cup","plate","on","Put the red cup on the yellow plate","occluded"),
)


def main():
    parser=argparse.ArgumentParser(); parser.add_argument("--trials",type=int,default=5); parser.add_argument("--seed",type=int,default=23)
    parser.add_argument("--output",type=Path,default=Path("outputs/agent_benchmark.json")); parser.add_argument("--online",action="store_true")
    args=parser.parse_args(); rows=[]
    if args.trials<1: parser.error("--trials must be positive")
    rng=np.random.default_rng(args.seed)
    order=rng.permutation(len(CASES))
    for i in range(args.trials):
        source,target,relation,command,scenario=CASES[int(order[i%len(CASES)])]
        seed=int(rng.integers(1,100000)); config=AgentConfig(); config.llm.mode="auto" if args.online else "offline"
        if scenario=="outside_view": config.camera.view_targets=[(1.8,-1.8,0),config.camera.look_at,config.camera.look_at]
        with Simulation(config,seed,scenario) as sim:
            robot=MuJoCoRobot(sim); camera=MujocoCamera(sim,robot); agent=EmbodiedAgent(robot,camera,config)
            result=agent.execute(command)
            sim.execution_finished=True
            truth=SimulationGroundTruth(sim)
            error=None
            if agent.context and agent.context.source:
                error=float(np.linalg.norm(agent.context.source.position-truth.pose(source)))
            row={"trial":i,"seed":seed,"scenario":scenario,"command":command,"agent_success":result.success,
                 "ground_truth_success":bool(truth.relation(source,target,relation)),"perception_error_m":error,
                 "collision_count":max(sim.collision_count,result.metrics.get("unexpected_contact_events",0)),"log_path":result.log_path,**result.metrics}
            rows.append(row); print(json.dumps(row,ensure_ascii=False),flush=True)
        args.output.parent.mkdir(parents=True,exist_ok=True)
        errors=[r["perception_error_m"] for r in rows if r["perception_error_m"] is not None]
        summary={"trials":len(rows),"task_success_rate":np.mean([r["agent_success"] and r["ground_truth_success"] for r in rows]),
                 "grasp_success_rate":np.mean([r["grasp_success"] for r in rows]),"placement_success_rate":np.mean([r["placement_success"] for r in rows]),
                 "first_grasp_verified_rate":np.mean([r["first_grasp_verified"] for r in rows]),
                 "first_lift_success_rate":np.mean([r["first_lift_success"] for r in rows]),
                 "no_retry_task_success_rate":np.mean([r["agent_success"] and r["grasp_attempts"]==1 and r["recovery_attempts"]==0 for r in rows]),
                 "perception_error_m":float(np.mean(errors)) if errors else None,"average_recovery_attempts":np.mean([r["recovery_attempts"] for r in rows]),
                 "collision_count":sum(r["collision_count"] for r in rows),"llm_calls_per_task":np.mean([r["llm_calls"] for r in rows]),
                 "execution_time_s":np.mean([r["execution_time_s"] for r in rows])}
        args.output.write_text(json.dumps({"summary":summary,"trials":rows},ensure_ascii=False,indent=2))
    print(json.dumps(summary,ensure_ascii=False,indent=2))


if __name__=="__main__": main()
