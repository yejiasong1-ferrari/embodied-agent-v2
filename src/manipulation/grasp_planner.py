import numpy as np
from .grasp_candidates import GraspCandidate


def down_rotation(yaw):
    closing=np.array([-np.sin(yaw),-np.cos(yaw),0.])
    down=np.array([0.,0.,-1.])
    return np.column_stack((np.cross(closing,down),closing,down))


class GraspPlanner:
    def __init__(self,config): self.config=config
    def candidates(self,obj,world):
        cfg=self.config.planner
        results=[]
        for kind in ("top_grasp",):
            for index,yaw in enumerate((0.,np.pi/2,np.pi/4,-np.pi/4,np.pi,-np.pi/2)):
                closing=np.array([-np.sin(yaw),-np.cos(yaw)])
                # Project actual segmented cloud to estimate width.
                body_points=obj.points[obj.points[:,2]>obj.high[2]-.012] if len(obj.points) else obj.points
                values=body_points[:,:2]@closing if len(body_points)>12 else np.array([0.,min(obj.size[:2])])
                # Near the maximum jaw opening, ignoring the outer 2% of
                # measured surface points can make a rotated box look as if
                # it fits while the actual finger path hits its corner.
                width=float(np.percentile(values,99.5)-np.percentile(values,.5))
                if width>cfg.max_gripper_width_m-.003: continue
                height=obj.high[2]-min(cfg.grasp_insertion_m,obj.size[2]*.42)
                palm=obj.position.copy(); palm[2]=height+cfg.palm_to_contact_m
                approach=np.array([0.,0.,-1.]) if kind=="top_grasp" else np.array([-np.cos(yaw),np.sin(yaw),0.])
                # Reserve both finger lanes from fully open to contact. A
                # neighbour on one side can obstruct one yaw but not another.
                along=np.array([-closing[1],closing[0]])
                clearance=.3
                for other in world.objects.values():
                    if other.id==obj.id or other.high[2]<height: continue
                    offset=other.position[:2]-obj.position[:2]
                    extent=other.size[:2]/2
                    for sign in (-1,1):
                        lane=sign*(width+cfg.max_gripper_width_m)/4
                        half_lane=(cfg.max_gripper_width_m-width)/4+.012
                        gaps=np.array([abs(offset@closing-lane)-extent@np.abs(closing)-half_lane,
                                       abs(offset@along)-extent@np.abs(along)-.012])
                        distance=float(np.linalg.norm(np.maximum(gaps,0))+min(max(gaps),0))
                        clearance=min(clearance,distance)
                results.append(GraspCandidate(f"{kind}:{index}",palm,down_rotation(yaw),approach,width,
                               float(.95-.035*index-(.08 if kind=="side_grasp" else 0)-max(0,.06-clearance)),clearance,kind))
        model=getattr(obj,'manipulation_geometry',{})
        if model and not model.get('gripper_fit_top',True):results=[]
        for index,region in enumerate(model.get('graspable_regions',[])):
            if region['name']=='top':continue
            for direction,outward in {'left':[-1.,0.,0.],'right':[1.,0.,0.],'front':[0.,-1.,0.],'back':[0.,1.,0.]}.items():
                if region['clearance'][direction]<max(.05,self.config.grasp.finger_clearance_m):continue
                forward=-np.asarray(outward);closing=np.array([-forward[1],forward[0],0.])
                rotation=np.column_stack([np.cross(closing,forward),closing,forward])
                contact=obj.position.copy();contact[2]=(region['z_min']+region['z_max'])/2
                palm=contact-forward*cfg.palm_to_contact_m
                points=obj.points[(obj.points[:,2]>=region['z_min'])&(obj.points[:,2]<=region['z_max'])]
                width=region['width']
                if len(points)>8:
                    values=points@closing;width=max(width,float(np.percentile(values,98)-np.percentile(values,2)))
                if width>cfg.max_gripper_width_m-self.config.grasp.finger_clearance_m:continue
                score=region['graspability']+min(.1,region['clearance'][direction]/3)
                # Prefer the middle of a usable stem to very low contacts
                # that put the palm close to the supporting surface.
                score-=.6*abs(contact[2]-(obj.low[2]+.6*obj.size[2]))
                results.append(GraspCandidate(f"side:{index}:{direction}",palm,rotation,forward,width,score,
                    region['clearance'][direction],'side_grasp',region['name'],direction,contact,geometry_score=score,
                    stability_score=region['graspability'],source_position=obj.position.copy()))
                # Opposite wrist roll keeps the same horizontal closing axis
                # and contact region, but can lift the bulky hand above table.
                results.append(GraspCandidate(f"side:{index}:{direction}:roll",palm,rotation@np.diag([-1.,-1.,1.]),forward,width,score,
                    region['clearance'][direction],'side_grasp',region['name'],direction,contact,geometry_score=score,
                    stability_score=region['graspability'],source_position=obj.position.copy()))
        return sorted(results,key=lambda c:c.score,reverse=True)
