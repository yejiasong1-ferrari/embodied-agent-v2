"""Robot-only kinematics plus observed collision boxes. No physics state copy."""
import mujoco
import numpy as np
from joint_control import SCENE_PATH
from kinematics import solve_hand_ik
from robot.robot_interface import Pose


class PlanningBackend:
    CAPACITY = 64
    def __init__(self,margin=.003):
        self.margin=margin
        spec = mujoco.MjSpec.from_file(str(SCENE_PATH))
        for side in ("left_finger","right_finger"):
            spec.body(side).add_geom(name=side+"_grip_pad",type=mujoco.mjtGeom.mjGEOM_BOX,
                size=(.0085,.006,.029),pos=(0,.008,.05))
        for i in range(self.CAPACITY):
            body = spec.worldbody.add_body(name=f"observed_{i}",pos=(0,0,-10))
            body.add_geom(name=f"observed_geom_{i}",type=mujoco.mjtGeom.mjGEOM_BOX,size=(.001,.001,.001))
        body = spec.worldbody.add_body(name="perceived_payload",mocap=True,pos=(0,0,-10))
        body.add_geom(name="perceived_payload_geom",type=mujoco.mjtGeom.mjGEOM_BOX,size=(.001,.001,.001))
        self.model = spec.compile()
        # Reserve clearance for measured servo tracking error near the table.
        # This is a conservative robot workspace plane, not an object pose.
        self.model.geom_pos[self.model.geom('floor').id,2]=.008
        self.data = mujoco.MjData(self.model)
        self.addresses = [int(self.model.joint(f"joint{i}").qposadr[0]) for i in range(1,8)]
        self.limits = np.array([self.model.jnt_range[self.model.joint(f"joint{i}").id] for i in range(1,8)])
        self.hand_id = self.model.body("hand").id
        self.robot_bodies = {self.model.body(name).id for name in [*(f"link{i}" for i in range(9)),"hand","left_finger","right_finger"] if self._has_body(name)}
        self.fingers = {self.model.body(n).id for n in ("left_finger","right_finger")}
        self.obstacles = {}
        self.target_id = self.support_id = None
        self.payload = None
        self.gripper_width = .08

    def _has_body(self,name):
        try: self.model.body(name); return True
        except KeyError: return False

    def configure(self,world,target_id,payload,support_id):
        fragments=set(payload.get('merged_fragment_ids',())) if payload else set()
        self.target_id,self.payload,self.support_id=target_id,payload,support_id
        items = [o for o in world.objects.values() if not payload or o.id not in fragments|{payload["id"]}]
        if len(items)>self.CAPACITY: raise ValueError("collision-world capacity exceeded")
        self.obstacles.clear()
        for i in range(self.CAPACITY):
            body = self.model.body(f"observed_{i}").id
            geom = self.model.geom(f"observed_geom_{i}").id
            if i<len(items):
                obj=items[i]
                geometry=getattr(obj,'manipulation_geometry',{})
                self.model.body_pos[body]=obj.position+np.asarray(geometry.get('collision_center_offset',[0,0,0]))
                self.model.geom_size[geom]=np.maximum(np.asarray(geometry.get('collision_size',obj.size))/2,.003)+self.margin
                self._update_box_bounds(body,geom)
                self.obstacles[body]=obj.id
            else: self.model.body_pos[body]=(0,0,-10)
        pgeom=self.model.geom("perceived_payload_geom").id
        self.model.geom_size[pgeom]=np.maximum(np.asarray(payload.get('collision_size',payload["size"]))/2,.003) if payload else (.001,.001,.001)
        self._update_box_bounds(self.model.body('perceived_payload').id,pgeom)
        mujoco.mj_setConst(self.model,self.data)

    def _update_box_bounds(self,body,geom):
        # MuJoCo does not rebuild these broad-phase bounds when geom_size is
        # edited. Leaving the original 1 mm bounds silently misses collisions
        # against larger perceived boxes. Each observed body has one box.
        half=self.model.geom_size[geom]
        self.model.geom_rbound[geom]=float(np.linalg.norm(half))
        self.model.geom_aabb[geom,:3]=0
        self.model.geom_aabb[geom,3:]=half
        address=int(self.model.body_bvhadr[body])
        if address>=0:
            self.model.bvh_aabb[address,:3]=0
            self.model.bvh_aabb[address,3:]=half

    def forward(self,q):
        self.data.qpos[self.addresses]=q
        for name in ("finger_joint1","finger_joint2"):
            self.data.qpos[int(self.model.joint(name).qposadr[0])]=self.gripper_width/2
        mujoco.mj_forward(self.model,self.data)
        return Pose(self.data.xpos[self.hand_id].copy(),self.data.xmat[self.hand_id].reshape(3,3).copy())

    def solve(self,pose,seed):
        results=[]
        seeds=[]
        for delta in (0,.45,-.45):
            start=np.asarray(seed).copy(); start[2]+=delta
            seeds.append(start)
        # A side observation can leave the arm on the opposite IK branch.
        # Seed additional solves from calibrated robot postures, never scene
        # object states; all returned configurations still require collision checks.
        if self.model.nkey:
            home=self.model.key_qpos[0,self.addresses].copy()
            for angle in (np.arctan2(pose.position[1],pose.position[0]),0.):
                for elbow in (0.,.7,-.7):
                    start=home.copy();start[0]=angle;start[2]=elbow;seeds.append(start)
        for index,start in enumerate(seeds):
            if index==3 and any(self.collision(q) is None for q in results):break
            start=np.clip(start,self.limits[:,0]+1e-4,self.limits[:,1]-1e-4)
            try:
                q=solve_hand_ik(self.model,pose.position,start,max_iterations=180,target_rotation=pose.rotation)
            except RuntimeError: continue
            if not any(np.linalg.norm(q-old)<.05 for old in results): results.append(q)
        return results

    def self_mask(self,points,q):
        self.forward(q)
        mask=np.zeros(len(points),bool)
        for gid in range(self.model.ngeom):
            bid=int(self.model.geom_bodyid[gid])
            if bid not in self.robot_bodies: continue
            rotation=self.data.geom_xmat[gid].reshape(3,3)
            local=(points-self.data.geom_xpos[gid])@rotation
            radius=float(self.model.geom_rbound[gid])
            # Robot geometry is proprioceptive calibration. It contains no
            # environment object pose and only masks the robot's own pixels.
            if self.model.geom_type[gid]==mujoco.mjtGeom.mjGEOM_BOX:
                hit=np.all(np.abs(local)<=self.model.geom_size[gid]+.003,axis=1)
            elif self.model.geom_type[gid]==mujoco.mjtGeom.mjGEOM_MESH:
                mesh=int(self.model.geom_dataid[gid])
                start=int(self.model.mesh_vertadr[mesh])
                vertices=self.model.mesh_vert[start:start+int(self.model.mesh_vertnum[mesh])]
                # A bounding sphere around the long hand mesh wrongly masks
                # the object between the fingers. Use the calibrated mesh's
                # local bounds, which do not expand into that empty space.
                hit=np.all((local>=vertices.min(axis=0)-.001)&(local<=vertices.max(axis=0)+.001),axis=1)
            else:
                hit=np.linalg.norm(local,axis=1)<radius+.003
            mask|=hit
        return mask

    def collision(self,q):
        q=np.asarray(q)
        if np.any(q<self.limits[:,0]-.001) or np.any(q>self.limits[:,1]+.001): return "joint_limit"
        pose=self.forward(q)
        pbody=self.model.body("perceived_payload").id
        mocap=int(self.model.body_mocapid[pbody])
        if self.payload:
            if (pose.rotation@np.asarray(self.payload["relative_rotation"]))[2,2]<np.cos(np.deg2rad(12)):return "carrying_tilt_limit"
            self.data.mocap_pos[mocap]=pose.position+pose.rotation@np.asarray(self.payload.get("collision_offset",self.payload["offset"]))
            rotation=pose.rotation@np.asarray(self.payload["relative_rotation"])
            quat=np.empty(4); mujoco.mju_mat2Quat(quat,rotation.ravel())
            self.data.mocap_quat[mocap]=quat
        else: self.data.mocap_pos[mocap]=(0,0,-10)
        mujoco.mj_forward(self.model,self.data)
        for contact in self.data.contact[:self.data.ncon]:
            if contact.dist >= -.0005: continue
            a,b=(int(self.model.geom_bodyid[int(g)]) for g in (contact.geom1,contact.geom2))
            if a==b: continue
            if a in self.robot_bodies and b in self.robot_bodies:
                # Model contact exclusions already remove adjacent links.
                return "self_collision"
            robot=a if a in self.robot_bodies else b if b in self.robot_bodies else None
            other=b if robot==a else a
            if robot is not None:
                if other==pbody and robot in self.fingers|{self.hand_id}: continue
                if self.obstacles.get(other)==self.target_id and (robot in self.fingers or
                        (robot==self.hand_id and self.support_id in {'side_release','pregrasp'})): continue
                # The robot base is bolted to the calibrated table.
                if other==0 and robot==self.model.body("link0").id: continue
                return f"robot_collision:{self.obstacles.get(other,'table')}"
            if self.payload and pbody in (a,b):
                other=b if a==pbody else a
                if other==0 and self.support_id=="observed_table" and contact.dist>-.012: continue
                if self.obstacles.get(other)==self.support_id: continue
                return f"payload_collision:{self.obstacles.get(other,'table')}"
        return None
