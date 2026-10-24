"""Separate equal-colour silhouettes that touch in RGB but are apart in 3D."""
from itertools import product
import numpy as np
from .pointcloud import pixel_to_world


def split_mask_3d(frame, mask, min_pixels=30, voxel=.012):
    valid=mask & np.isfinite(frame.depth) & (frame.depth>.025) & (frame.depth<3)
    v,u=np.nonzero(valid)
    if len(u)<min_pixels: return []
    points=pixel_to_world(np.column_stack((u,v)),frame.depth[v,u],frame.intrinsics,frame.camera_to_world)
    if np.max(np.ptp(points,axis=0))<.14: return [valid]
    voxel=max(voxel,1.5*float(np.max(frame.depth[v,u]))/min(frame.intrinsics[0,0],frame.intrinsics[1,1]))
    cells,inverse=np.unique(np.floor(points/voxel).astype(int),axis=0,return_inverse=True)
    lookup={tuple(cell):i for i,cell in enumerate(cells)}
    parent=np.arange(len(cells))
    def find(i):
        while parent[i]!=i:
            parent[i]=parent[parent[i]]; i=parent[i]
        return i
    offsets=[p for p in product((-1,0,1),repeat=3) if p>(0,0,0)]
    for i,cell in enumerate(cells):
        x,y,z=map(int,cell)
        for dx,dy,dz in offsets:
            j=lookup.get((x+dx,y+dy,z+dz))
            if j is not None: parent[find(j)]=find(i)
    labels=np.array([find(i) for i in range(len(cells))])[inverse]
    result=[]
    for identity in np.unique(labels):
        selected=labels==identity
        if np.count_nonzero(selected)<min_pixels: continue
        component=np.zeros_like(mask); component[v[selected],u[selected]]=True
        result.append(component)
    return result
