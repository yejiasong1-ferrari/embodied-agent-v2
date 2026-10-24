"""Task and renderer threads exchange immutable copies through a tiny mailbox."""
import copy
import json
import queue
import threading
import time
import traceback
import numpy as np
import mujoco
from PySide6.QtCore import QThread, Signal
from PySide6.QtGui import QImage
from agent.embodied_agent import EmbodiedAgent
from agent.task_log import serializable
from simulation.environment import Simulation
from simulation.mujoco_camera import MujocoCamera
from robot.mujoco_robot import MuJoCoRobot
from .camera import CameraState

STATE_SPEC=mujoco.mjtState.mjSTATE_INTEGRATION

def frozen(value):return json.loads(json.dumps(value,ensure_ascii=False,default=serializable))
def qimage(array):
    a=np.ascontiguousarray(array,dtype=np.uint8)
    return QImage(a.data,a.shape[1],a.shape[0],a.strides[0],QImage.Format.Format_RGB888).copy()
def depth_image(depth):
    if depth is None:return QImage()
    valid=np.isfinite(depth)&(depth>0)&(depth<3.)
    scaled=(np.clip((3.-np.nan_to_num(depth,nan=3.,posinf=3.))/3.,0,1)*255).astype(np.uint8)
    rgb=np.repeat(scaled[...,None],3,axis=2);rgb[~valid]=0
    return qimage(rgb)

class RenderMailbox:
    """Never contains the live MjData. Camera requests cannot reach the task."""
    def __init__(self):
        self.lock=threading.Lock();self.model=None;self.generation=0;self.snapshot=None
        self.revision=0;self.camera=CameraState();self.camera_revision=0;self.size=(1200,760)
        self.output=None;self.output_revision=0
    def set_model(self,model):
        with self.lock:self.model=model;self.generation+=1;self.snapshot=None
    def publish(self,snapshot,telemetry):
        with self.lock:self.snapshot=(snapshot,telemetry);self.revision+=1
    def set_camera(self,camera):
        with self.lock:self.camera=camera;self.camera_revision+=1
    def set_size(self,size):
        with self.lock:self.size=size
    def read(self):
        with self.lock:return self.model,self.generation,self.snapshot,self.revision,self.camera,self.camera_revision,self.size
    def display(self,packet):
        with self.lock:self.output=packet;self.output_revision+=1
    def take(self,last):
        with self.lock:return (self.output_revision,self.output) if last!=self.output_revision else (last,None)

class RenderWorker(QThread):
    failed=Signal(str)
    def __init__(self,mailbox):
        super().__init__();self.mailbox=mailbox;self.quit_event=threading.Event()
    def run(self):
        renderer=None;generation=-1;previous=None;size=None;data=None
        frames=0;fps=0.;window=time.monotonic();quality=1.;last_request=None;last_change=window
        try:
            while not self.quit_event.is_set():
                begin=time.monotonic()
                model,gen,snapshot,revision,state,camrev,requested=self.mailbox.read()
                if model is None or snapshot is None:
                    self.quit_event.wait(.02);continue
                width,height=requested
                request=(gen,revision,camrev,requested)
                if request!=last_request:last_request=request;last_change=begin
                display_quality=quality if begin-last_change<.5 else 1.
                # The mailbox size is in physical display pixels. Cap only the
                # spectator image; the task's RGB-D camera is untouched.
                scale=min(1.,1920/max(width,1),1200/max(height,1))*display_quality
                target=(max(160,int(width*scale)),max(120,int(height*scale)))
                if gen!=generation:
                    # This independent model belongs exclusively to rendering.
                    if renderer:renderer.close();renderer=None
                    generation=gen;data=mujoco.MjData(model);previous=None
                    model.vis.global_.offwidth=2048;model.vis.global_.offheight=1280
                    plane=model.geom_type==mujoco.mjtGeom.mjGEOM_PLANE
                    model.geom_matid[plane]=-1;model.geom_rgba[plane]=(.10,.12,.135,1)
                if renderer is None or target!=size:
                    if renderer:renderer.close()
                    size=target;renderer=mujoco.Renderer(model,width=size[0],height=size[1]);previous=None
                key=(gen,revision,camrev,size)
                if key==previous:
                    self.quit_event.wait(.015);continue
                previous=key
                values,telemetry=snapshot
                mujoco.mj_setState(model,data,values,STATE_SPEC)
                mujoco.mj_forward(model,data)
                camera=mujoco.MjvCamera();mujoco.mjv_defaultCamera(camera)
                camera.azimuth=state.azimuth;camera.elevation=state.elevation
                camera.distance=state.distance;camera.lookat[:]=state.lookat
                renderer.update_scene(data,camera=camera)
                self.grid(renderer.scene)
                c=renderer.scene.camera[0];other=renderer.scene.camera[1]
                projection={'pos':((c.pos+other.pos)/2).copy().tolist(),'forward':c.forward.copy().tolist(),'up':c.up.copy().tolist(),
                            'near':float(c.frustum_near),'tan_half_fov':float((c.frustum_top-c.frustum_bottom)/(2*c.frustum_near)),
                            'aspect':size[0]/size[1]}
                im=qimage(renderer.render())
                frames+=1;now=time.monotonic()
                if now-window>=1:
                    fps=frames/(now-window);frames=0;window=now
                    # Resolution adapts on concrete rendering cost, not task data.
                    if now-begin>.055 and quality>.65:quality=max(.65,quality-.1)
                    elif now-begin<.028 and quality<1.:quality=min(1.,quality+.05)
                self.mailbox.display({'image':im,'projection':projection,'camera':state,'fps':fps,
                    'telemetry':telemetry,'rendered_at':now,'generation':gen})
                self.quit_event.wait(max(0,1/30-(time.monotonic()-begin)))
        except Exception:self.failed.emit(traceback.format_exc())
        finally:
            if renderer:renderer.close()
    @staticmethod
    def grid(scene):
        for i in range(-10,11):
            for start,end in [([i*.1,-1,.001],[i*.1,1,.001]),([-1,i*.1,.001],[1,i*.1,.001])]:
                if scene.ngeom>=scene.maxgeom:return
                geom=scene.geoms[scene.ngeom]
                mujoco.mjv_initGeom(geom,mujoco.mjtGeom.mjGEOM_LINE,np.zeros(3),np.zeros(3),np.eye(3).ravel(),np.array([.115,.135,.15,.38]))
                mujoco.mjv_connector(geom,mujoco.mjtGeom.mjGEOM_LINE,1.,np.array(start),np.array(end))
                scene.ngeom+=1

class AgentWorker(QThread):
    ready=Signal();trace=Signal(dict);world=Signal(dict);vision=Signal(dict)
    finished_task=Signal(dict);fatal=Signal(str);task_started=Signal(dict)
    def __init__(self,config,mailbox):
        super().__init__();self.config=config;self.mailbox=mailbox;self.commands=queue.Queue()
        self.quit_event=threading.Event();self.cancel_requested=threading.Event()
        self.robot=self.agent=self.sim=None;self.last_snapshot=0.;self.last_world_revision=-1;self.last_candidate=None;self.force_sample=None
    def submit(self,command,mode):
        self.cancel_requested.clear();self.commands.put((command,mode))
    def stop(self):
        # Direct high-priority cancellation; never queued behind camera/log/LLM work.
        self.cancel_requested.set()
        if self.agent:self.agent.cancel()
    def reset(self):self.commands.put(None)
    def publish(self,force=False):
        now=time.monotonic()
        if not force and now-self.last_snapshot<1/25:return
        self.last_snapshot=now
        sim=self.sim
        values=np.empty(mujoco.mj_stateSize(sim.model,STATE_SPEC))
        with sim.lock:mujoco.mj_getState(sim.model,sim.data,values,STATE_SPEC)
        values.setflags(write=False)
        telemetry={'simulation_time':float(sim.data.time),'joints':np.rad2deg(self.robot.get_joint_positions()).tolist(),
                   'width_mm':self.robot.get_gripper_state()['width']*1000,'ee':self.robot.get_ee_pose().position.tolist(),
                   'grasp_force_sample':self.force_sample}
        self.mailbox.publish(values,telemetry)
        c=self.agent.context if self.agent else None
        candidate=c.candidate.id if c and c.candidate else None
        if candidate!=self.last_candidate:
            self.last_candidate=candidate
            self.context_event(0.)
    def context_event(self,elapsed):
        c=self.agent.context
        if c:
            d={'skill':c.current_skill,'source_id':c.source_id,'target_id':c.target_id,
               'source':None if c.source is None else c.source.to_dict(),
               'candidate':None if c.candidate is None else {'id':c.candidate.id,'type':c.candidate.grasp_type,'width':c.candidate.grasp_width}}
            self.trace.emit({'type':'context','data':frozen(d),'elapsed_s':elapsed})
    def _trace(self,event):
        if self.cancel_requested.is_set():self.agent.cancel()
        # Keep full on-disk evidence; don't enqueue huge per-step arrays in Qt.
        if event['type'] not in {'trajectory','visual_attachment','observation','attachment_hypothesis'}:
            row=frozen(event);row['task_id']=self.agent.log.task_id;row['log_path']=str(self.agent.log.path)
            row['wall_time']=time.time();self.trace.emit(row)
        if event['type'] in {'state','skill','grasp_candidates','multiview_fusion'}:
            self.context_event(event['elapsed_s'])
        if event['type']=='grasp_validation':self.force_sample={**frozen(event['data']),'sampled_at':time.monotonic()}
        if event['type']=='multiview_fusion':self.world.emit(frozen(self.agent.world.to_dict()))
        self.publish()
    def _images(self,rgb,overlay,world):
        c=self.agent.context;frame=c.perception.last_frame if c else self.initial_frame
        payload={'RGB':qimage(rgb),'Detection':qimage(overlay),'Depth':depth_image(frame.depth if frame else None),
                 'timestamp':frame.timestamp if frame else time.monotonic(),'view':frame.view_id if frame else '',
                 'depth_range':'0–3 m · 近亮远暗 · 无效值为黑色','revision':world.get('revision',0)}
        self.vision.emit(payload);self.world.emit(frozen(world));self.publish(True)
    def run(self):
        try:
            while not self.quit_event.is_set():
                with Simulation(self.config) as sim:
                    self.sim=sim;self.force_sample=None;self.last_candidate=None;self.robot=MuJoCoRobot(sim,step_observer=self.publish)
                    camera=MujocoCamera(sim,self.robot)
                    self.mailbox.set_model(copy.copy(sim.model))
                    self.agent=EmbodiedAgent(self.robot,camera,self.config,log_sink=self._trace,image_sink=self._images)
                    self.initial_frame=camera.capture()
                    self._images(self.initial_frame.rgb,self.initial_frame.rgb,self.agent.world.to_dict())
                    self.ready.emit()
                    while not self.quit_event.is_set():
                        try:request=self.commands.get(timeout=.1)
                        except queue.Empty:continue
                        if request is None:break
                        command,mode=request
                        if self.cancel_requested.is_set():
                            self.finished_task.emit({'success':False,'message':'任务开始前已停止','metrics':{'failure':'USER_STOP'}});continue
                        self.config.llm.mode=mode;sim.execution_finished=False
                        self.task_started.emit({'command':command,'mode':mode})
                        result=self.agent.execute(command)
                        sim.execution_finished=True
                        self.publish(True);self.finished_task.emit(frozen(result.__dict__))
        except Exception:self.fatal.emit(traceback.format_exc())
        finally:self.sim=None
