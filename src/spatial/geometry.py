"""Footprints are reconstructed surface hulls; no simulator inputs."""
import cv2
import numpy as np


def footprint(obj,position=None):
    points=getattr(obj,'points',np.empty((0,3)))
    verified_world=getattr(obj,'manipulation_geometry',{}).get('verified_footprint_world')
    verified=getattr(obj,'manipulation_geometry',{}).get('verified_footprint_offsets')
    carried=getattr(obj,'manipulation_geometry',{}).get('carried_footprint_offsets')
    if verified_world is not None:
        poly=np.asarray(verified_world,dtype=float)
        if position is not None:
            anchor=np.asarray(obj.manipulation_geometry['verified_footprint_anchor'],dtype=float)
            poly=poly+np.asarray(position[:2])-anchor
        return poly
    if verified is not None:
        poly=np.asarray(verified)+obj.position[:2]
    elif carried is not None:
        poly=np.asarray(carried)+obj.position[:2]
    elif len(points)>=8:
        poly=cv2.convexHull(np.asarray(points[:,:2],np.float32)).reshape(-1,2).astype(float)
        # A partial face is not the object's whole footprint. Box dimensions
        # retained from a complete RGB-D view bound its unseen opposite face.
        if obj.category=='box' and np.any(np.ptp(poly,axis=0)<np.asarray(obj.size[:2])*.85):
            corners=obj.position[:2]+np.array([[-1,-1],[1,-1],[1,1],[-1,1]])*obj.size[:2]/2
            poly=cv2.convexHull(np.concatenate([poly,corners]).astype(np.float32)).reshape(-1,2).astype(float)
        elif obj.category=='plate' and np.any(np.ptp(poly,axis=0)<np.asarray(obj.size[:2])*.85):
            # Keep the complete plate extent measured from an earlier view
            # when the carried object or fingers hide most of its surface.
            angles=np.linspace(0,2*np.pi,48,endpoint=False)
            poly=obj.position[:2]+np.column_stack((np.cos(angles),np.sin(angles)))*np.asarray(obj.size[:2])/2
    else:
        # WorldModel dimensions are already RGB-D estimates.
        half=np.asarray(obj.size[:2])/2
        poly=np.asarray(obj.position[:2])+np.array([[-1,-1],[1,-1],[1,1],[-1,1]])*half
    if position is not None:poly=poly+np.asarray(position[:2])-obj.position[:2]
    return poly


def overlap(a,b):
    for polygon in (a,b):
        edges=np.roll(polygon,-1,axis=0)-polygon
        for edge in edges:
            axis=np.array([-edge[1],edge[0]])
            if np.linalg.norm(axis)<1e-8:continue
            av,bv=a@axis,b@axis
            if av.max()<bv.min() or bv.max()<av.min():return False
    return True


def ray_exit_distance(polygon,center,outward):
    """Measured footprint boundary on a ray through the object's centre."""
    center=np.asarray(center,float);outward=np.asarray(outward,float)
    hits=[]
    for start,end in zip(polygon,np.roll(polygon,-1,axis=0)):
        edge=end-start;matrix=np.column_stack([outward,-edge])
        if abs(np.linalg.det(matrix))<1e-10:continue
        distance,segment=np.linalg.solve(matrix,start-center)
        if distance>=0 and -.0001<=segment<=1.0001:hits.append(distance)
    return min(hits) if hits else None


def polygon_gap(a,b):
    if overlap(a,b):return 0.
    def distance(points,polygon):
        start=polygon;delta=np.roll(polygon,-1,axis=0)-polygon
        offset=points[:,None,:]-start[None,:,:]
        t=np.clip(np.sum(offset*delta,axis=2)/np.maximum(np.sum(delta**2,axis=1),1e-12),0,1)
        return np.linalg.norm(offset-t[...,None]*delta,axis=2).min()
    return float(min(distance(a,b),distance(b,a)))


def projected_edges(obj,axis,position=None):
    measured_position=obj.position if position is None else np.asarray(position)
    if abs(axis[2])>.9:
        values=(measured_position[2]+np.array([-.5,.5])*obj.size[2])*axis[2]
    else:values=footprint(obj,position)@axis[:2]
    return float(values.min()),float(values.max())


def support_covered(world,obj,position,margin=.025):
    """All footprint corners and edge midpoints need nearby observed support."""
    points=world.support_points
    if len(points)<9:return False
    half=np.max(np.abs(footprint(obj,position)-np.asarray(position[:2])),axis=0)+margin
    sample=np.asarray(position[:2])+np.array([(x,y) for x in (-1,0,1) for y in (-1,0,1)])*half
    distances=np.sqrt(np.min(np.sum((sample[:,None,:]-points[None,:,:2])**2,axis=2),axis=1))
    return bool(np.all(distances<.025))


def supported_by(obj,support,position,margin=0.):
    """Require the entire measured footprint inside this support's surface."""
    polygon=footprint(support).astype(np.float32)
    if abs(position[2]-obj.size[2]/2-support.high[2])>.01:return False
    return all(cv2.pointPolygonTest(polygon,tuple(map(float,p)),True)>=margin
               for p in footprint(obj,position))
