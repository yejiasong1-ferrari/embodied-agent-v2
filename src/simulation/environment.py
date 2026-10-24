from pathlib import Path
import threading
import mujoco
import numpy as np
from joint_control import create_model
from scene_layout import place_movable


class Simulation:
    def __init__(self, config, seed=None, scenario="default"):
        self.config = config
        self.lock = threading.RLock()
        self.stopped = False
        self.collision_count = 0
        self.execution_finished = False
        self.seed, self.scenario = seed, scenario
        def sensors(spec):
            body = spec.worldbody.add_body(name="agent_camera_mount", mocap=True, pos=(0,0,1.65))
            body.add_camera(name="agent_rgbd", fovy=config.camera.fovy_deg)
            if scenario in {'wide_top','wide_top_short'}:
                # Evaluation fixture only: an 11 cm cap over a 5 cm stem.
                # Runtime perception receives rendered RGB-D, never dimensions.
                stem=spec.geom('blue_box_geom')
                stem.type=mujoco.mjtGeom.mjGEOM_CYLINDER
                stem_height=.14 if scenario=='wide_top' else .09
                stem.size=(.025,stem_height/2,0)
                stem.pos=(0,0,stem_height/2-.04)
                spec.body('blue_box').add_geom(name='benchmark_wide_cap',
                    type=mujoco.mjtGeom.mjGEOM_CYLINDER,size=(.055,.010),pos=(0,0,stem_height+.01-.04),
                    mass=.015,friction=(1.1,.01,.001),rgba=(.08,.25,.90,1))
            if seed is not None:
                rng = np.random.default_rng(seed)
                # Benchmark-only scene construction. These values are never
                # provided to the agent, camera detector, planner or validator.
                for name in ("plate","clutter_books","clutter_thermos","clutter_can","clutter_box"):
                    body = spec.body(name)
                    body.pos[:2] += rng.uniform(-.025,.025,2)
            if scenario in {"near_obstacle","occluded"}:
                # Evaluation fixtures are created here, outside perception.
                block=spec.worldbody.add_body(name="benchmark_occluder",pos=(.46,-.255,.065 if scenario=="near_obstacle" else .12))
                block.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX,size=(.022,.032,.065 if scenario=="near_obstacle" else .12),rgba=(.35,.35,.35,1))
        self.model = create_model(spec_hook=sensors)
        self.data = mujoco.MjData(self.model)
        mujoco.mj_resetDataKeyframe(self.model,self.data,0)
        place_movable(self.model,self.data)
        if seed is not None:
            rng = np.random.default_rng(seed)
            for jid in range(self.model.njnt):
                if self.model.jnt_type[jid] == mujoco.mjtJoint.mjJNT_FREE:
                    address = self.model.jnt_qposadr[jid]
                    self.data.qpos[address:address+2] += rng.uniform(-.025,.025,2)
        if scenario=="blocked":
            # A two-step evaluation fixture, confined to scene construction.
            cup=self.model.body("red_cup").id; box=self.model.body("cyan_mug").id
            cup_address=int(self.model.jnt_qposadr[self.model.body_jntadr[cup]])
            box_address=int(self.model.jnt_qposadr[self.model.body_jntadr[box]])
            self.data.qpos[box_address:box_address+3]=self.data.qpos[cup_address:cup_address+3]+np.array([0,.095,-.01])
        # Make the existing invisible proxy pads visible without changing their
        # established contact geometry or dynamics.
        for i in range(self.model.ngeom):
            name = mujoco.mj_id2name(self.model,mujoco.mjtObj.mjOBJ_GEOM,i) or ""
            if name.endswith("_grip_pad"):
                self.model.geom_group[i] = 0
                self.model.geom_rgba[i] = (.09,.09,.09,1)
        mujoco.mj_forward(self.model,self.data)
        # Keep routine captures small. MuJoCo's offscreen framebuffer must
        # still be large enough for the lazily created detail renderer.
        self.model.vis.global_.offwidth=max(config.camera.width,config.camera.detail_width)
        self.model.vis.global_.offheight=max(config.camera.height,config.camera.detail_height)
        self.renderer = mujoco.Renderer(self.model,height=config.camera.height,width=config.camera.width)
        self.detail_renderer = None

    def get_detail_renderer(self):
        if self.detail_renderer is None:
            self.detail_renderer=mujoco.Renderer(self.model,height=self.config.camera.detail_height,
                                                 width=self.config.camera.detail_width)
        return self.detail_renderer

    def close(self):
        if self.detail_renderer is not None:self.detail_renderer.close()
        self.renderer.close()

    def __enter__(self): return self
    def __exit__(self,*args): self.close()
