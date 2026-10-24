import argparse
import json
from pathlib import Path
from config import AgentConfig
from .embodied_agent import EmbodiedAgent


def main(argv=None):
    parser=argparse.ArgumentParser(description="RGB-D embodied manipulation agent")
    parser.add_argument("instruction",nargs="?")
    parser.add_argument("--config",type=Path)
    parser.add_argument("--offline",action="store_true")
    parser.add_argument("--required-llm",action="store_true")
    parser.add_argument("--seed",type=int)
    parser.add_argument("--scenario",choices=["default","randomized","near_obstacle","occluded","outside_view","blocked","wide_top","wide_top_short"],default="default")
    parser.add_argument("--report",type=Path)
    parser.add_argument("--snapshot",type=Path)
    parser.add_argument("--live-view",action="store_true")
    parser.add_argument("--online-manager",action="store_true")
    parser.add_argument("--perception-only",action="store_true")
    args=parser.parse_args(argv)
    config=AgentConfig.load(args.config)
    if args.offline: config.llm.mode="offline"
    if args.required_llm or args.online_manager: config.llm.mode="required"
    if args.scenario=="outside_view": config.camera.view_targets=[(1.8,-1.8,0),config.camera.look_at,config.camera.look_at]
    from simulation.environment import Simulation
    from simulation.mujoco_camera import MujocoCamera
    from robot.mujoco_robot import MuJoCoRobot
    with Simulation(config,seed=args.seed,scenario=args.scenario) as sim:
        robot=MuJoCoRobot(sim); camera=MujocoCamera(sim,robot)
        def output(state,data):
            print(f"[{state}] "+json.dumps(data,ensure_ascii=False),flush=True)
            if state=="OBSERVE" and data.get("objects"):
                print("Detected: "+", ".join(data["objects"]),flush=True)
            if state=="PLAN" and data.get("skills"):
                print("Plan:\n"+"\n".join(f"{i} {skill}" for i,skill in enumerate(data["skills"],1)),flush=True)
            phase={"UNDERSTAND":"understand_task","OBSERVE":"scan_scene","SUCCESS":"done","FAILED":"stopped","RECOVER":"recovery"}.get(state)
            phase=phase or {"APPROACH_OBJECT":"approach","GRASP_OBJECT":"contact_search","VERIFY_GRASP":"stable_grasp","LIFT_OBJECT":"micro_lift","MOVE_TO_TARGET":"transport","PLACE_OBJECT":"descent","RETREAT":"gradual_release","VERIFY_TASK":"verify_result"}.get(data.get("skill"))
            if phase: print("V2_EVENT "+json.dumps({"phase":phase,"evidence":data}),flush=True)
        def image_sink(rgb,overlay,world):
            if args.snapshot:
                from PIL import Image
                args.snapshot.parent.mkdir(parents=True,exist_ok=True)
                temporary=args.snapshot.with_name(args.snapshot.stem+".tmp.png")
                Image.fromarray(overlay).save(temporary); temporary.replace(args.snapshot)
        agent=EmbodiedAgent(robot,camera,config,event_sink=output,image_sink=image_sink)
        if args.perception_only:
            from .task_log import TaskLog
            from llm.llm_client import LLMClient
            from perception.active_perception import ActivePerception
            from perception.debug_visualizer import write_report
            log=TaskLog(config.log_root,"perception only",config)
            perception=ActivePerception(camera,agent.world,config,LLMClient(config.llm,log.event),log)
            perception.image_sink=image_sink; perception.scan(force_all=True); write_report(log)
            print(json.dumps(agent.world.to_dict(),ensure_ascii=False,indent=2)); print("Evidence:",log.path)
            return 0
        while True:  # User input session; every task has independent budgets.
            command=args.instruction or input("Enter task (输入 exit 退出):\n> ").strip()
            if command.lower() in {"exit","quit","退出"}: break
            if not command: continue
            sim.execution_finished=False
            result=agent.execute(command)
            sim.execution_finished=True
            print(result.message,flush=True); print("Evidence:",result.log_path,flush=True)
            if args.report:
                args.report.parent.mkdir(parents=True,exist_ok=True)
                args.report.write_text(json.dumps(result.__dict__,ensure_ascii=False,indent=2))
            if args.snapshot:
                from PIL import Image
                args.snapshot.parent.mkdir(parents=True,exist_ok=True)
                Image.fromarray(camera.capture().rgb).save(args.snapshot)
            if args.instruction: return 0 if result.success else 1
    return 0
