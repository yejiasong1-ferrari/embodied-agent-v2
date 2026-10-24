import numpy as np
import mujoco


class SimulationGroundTruth:
    def __init__(self,simulation):
        if not simulation.execution_finished:
            raise PermissionError("Ground truth is unavailable until agent execution finishes")
        self._sim=simulation

    def pose(self,name):
        self._require_finished()
        bid=self._sim.model.body(name).id
        return self._sim.data.xpos[bid].copy()

    def bounds(self,name):
        self._require_finished()
        model,data=self._sim.model,self._sim.data
        bid=model.body(name).id
        lows=[]; highs=[]
        for gid in range(model.ngeom):
            if model.geom_bodyid[gid]!=bid or model.geom_contype[gid]==0: continue
            kind=model.geom_type[gid]; size=model.geom_size[gid]
            rotation=data.geom_xmat[gid].reshape(3,3)
            if kind==mujoco.mjtGeom.mjGEOM_CYLINDER:
                # A yawed circle is still a circle. Treating its local radial
                # axes as a box overstates the gap by r*(sqrt(2)-1).
                extent=size[0]*np.linalg.norm(rotation[:,:2],axis=1)+size[1]*np.abs(rotation[:,2])
            elif kind==mujoco.mjtGeom.mjGEOM_CAPSULE:
                extent=size[0]+size[1]*np.abs(rotation[:,2])
            elif kind==mujoco.mjtGeom.mjGEOM_SPHERE:
                extent=np.full(3,size[0])
            elif kind==mujoco.mjtGeom.mjGEOM_ELLIPSOID:
                extent=np.linalg.norm(rotation*size[None,:],axis=1)
            elif kind==mujoco.mjtGeom.mjGEOM_BOX:
                extent=np.abs(rotation)@size
            else:extent=np.full(3,model.geom_rbound[gid])
            lows.append(data.geom_xpos[gid]-extent); highs.append(data.geom_xpos[gid]+extent)
        return np.min(lows,axis=0),np.max(highs,axis=0)

    def _require_finished(self):
        if not self._sim.execution_finished:
            raise PermissionError("Ground truth cannot be read during execution")

    def relation(self,source,target,relation):
        from world.world_model import WorldObject
        from world.scene_graph import relation_satisfied
        def item(name):
            low,high=self.bounds(name)
            return WorldObject(name,name,"","",(low+high)/2,np.array([0,0,0,1]),high-low,1,0)
        return relation_satisfied(item(source),item(target),relation)
