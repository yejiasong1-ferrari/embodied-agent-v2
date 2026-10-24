"""Manipulation geometry from bounded, measured object point clouds."""
import time
import cv2
import numpy as np
from .pointcloud import voxel_downsample


class ObjectCloudFusion:
    def __init__(self,config):self.config=config;self.cache={}
    def add(self,obj):
        previous=self.cache.get(obj.id);points=obj.points
        if previous is not None and np.linalg.norm(previous['position']-obj.position)>=.02:previous=None
        if previous is not None:
            points=np.concatenate([previous['points'],points])
        points=voxel_downsample(points,self.config.perception.voxel_m)
        if len(points)>30:
            data=np.asarray(points,np.float32)
            tree=cv2.flann_Index(data,dict(algorithm=1,trees=4))
            _,square=tree.knnSearch(data,min(8,len(points)),params={'checks':32})
            mean=np.sqrt(square[:,1:]).mean(axis=1);points=points[mean<=np.median(mean)+3*np.std(mean)]
        limit=self.config.grasp.max_cloud_points
        if len(points)>limit:points=points[np.linspace(0,len(points)-1,limit).astype(int)]
        pose=getattr(obj,'observation_camera_pose',None)
        view=str(tuple(np.round(pose.ravel(),3))) if pose is not None else next(iter(obj.view_ids),'unknown')
        views={view}|(previous['views'] if previous else set())
        self.cache[obj.id]={'position':obj.position.copy(),'points':points,'views':views}
        return points,views


class GeometryAnalyzer:
    def __init__(self,config):self.config=config
    def analyze(self,obj,world,points=None,views=None):
        points=obj.points if points is None else points;cfg=self.config.grasp;maximum=self.config.planner.max_gripper_width_m
        if len(points)<cfg.min_slice_points:
            return {'height':float(obj.size[2]),'complexity_score':1.,'geometry_completeness':0.,'width_profile':[],
                    'graspable_regions':[],'top_grasp_score':0.,'side_grasp_score':0.,'views_used':len(views or obj.view_ids)}
        low,high=np.percentile(points,[1,99],axis=0);height=float(max(high[2]-obj.low[2],.005))
        center=np.median(points[:,:2],axis=0);cov=np.cov((points[:,:2]-center).T);eigen,axes=np.linalg.eigh(cov);basis=axes[:,::-1]
        if np.linalg.det(basis)<0:basis[:,1]*=-1
        # Gravity stays world +Z; PCA only resolves horizontal principal axes.
        horizontal=(points[:,:2]-center)@basis;profile=[]
        edges=np.linspace(obj.low[2],high[2]+1e-6,cfg.profile_slices+1)
        for index,(a,b) in enumerate(zip(edges,edges[1:])):
            mask=(points[:,2]>=a)&(points[:,2]<b);count=int(mask.sum())
            width=None;extents=None
            if count>=cfg.min_slice_points:
                bounds=np.percentile(horizontal[mask],[2,98],axis=0);extents=np.maximum(bounds[1]-bounds[0],.002)
                width=float(min(extents))
                # Visible arcs of a round stem need not span its diameter.
                # Fit only when measured side samples support a circle over
                # at least half its circumference; do not infer from a label.
                xy=points[mask,:2];design=np.column_stack([2*xy,np.ones(len(xy))])
                fit=np.linalg.lstsq(design,np.sum(xy*xy,axis=1),rcond=None)[0]
                square=fit[2]+np.dot(fit[:2],fit[:2])
                if square>0:
                    radius=np.sqrt(square);residual=np.median(np.abs(np.linalg.norm(xy-fit[:2],axis=1)-radius))
                    angles=np.sort(np.arctan2(xy[:,1]-fit[1],xy[:,0]-fit[0]));coverage=2*np.pi-np.diff(np.r_[angles,angles[0]+2*np.pi]).max()
                    if .01<radius<.10 and residual<.0015 and coverage>np.pi:
                        width=max(width,float(2*radius));extents=np.maximum(extents,2*radius)
            profile.append({'z_min':float(a),'z_max':float(b),'width':width,'extents':None if extents is None else extents.tolist(),'points':count})
        valid=[p for p in profile if p['width'] is not None];completeness=len(valid)/len(profile)
        top_mask=points[:,2]>=high[2]-.012;top=horizontal[top_mask]
        top_extents=np.diff(np.percentile(top,[2,98],axis=0),axis=0)[0] if len(top)>=3 else high[:2]-low[:2]
        top_width=float(min(top_extents));top_area=float(cv2.contourArea(cv2.convexHull(top.astype(np.float32)))) if len(top)>2 else 0.
        normal=np.array([0.,0.,1.])
        if top_mask.sum()>8:
            _,_,v=np.linalg.svd(points[top_mask]-points[top_mask].mean(axis=0),full_matrices=False);normal=v[-1]
            if normal[2]<0:normal=-normal
        widths=[row['width'] for row in valid];variation=(max(widths)-min(widths))/max(max(widths),.005) if widths else 1.
        half=(high[:2]-low[:2])/2;asymmetry=float(np.clip(np.linalg.norm(center-(high[:2]+low[:2])/2)/max(np.mean(half),.005),0,1))
        clearances={}
        for name,axis in {'left':[-1,0],'right':[1,0],'front':[0,-1],'back':[0,1]}.items():
            axis=np.array(axis);perp=np.array([-axis[1],axis[0]]);clearance=.3
            for other in world.objects.values():
                if other.id==obj.id or other.high[2]<obj.low[2]+.01:continue
                offset=other.position[:2]-obj.position[:2]
                if offset@axis<=0:continue
                if abs(offset@perp)<float((other.size[:2]+obj.size[:2])@np.abs(perp)/2)+.04:
                    clearance=min(clearance,float(offset@axis-(other.size[:2]+obj.size[:2])@np.abs(axis)/2))
            clearances[name]=clearance
        top_clear=all(not (other.id!=obj.id and other.low[2]>=obj.high[2]-.01 and np.all(np.abs(other.position[:2]-obj.position[:2])<(other.size[:2]+obj.size[:2])/2+.02)) for other in world.objects.values())
        fit_top=bool(top_width<=maximum-cfg.finger_clearance_m)
        uncertainty=float(np.clip(obj.uncertainty_m/self.config.perception.max_uncertainty_m,0,1))
        top_score=float(np.clip(.98-.35*variation-.25*asymmetry-.20*(1-abs(normal[2]))-.1*uncertainty,0,1)) if fit_top and top_clear else 0.
        regions=[]
        for i,row in enumerate(profile):
            width=row['width']
            if width is None or width>maximum-cfg.finger_clearance_m or row['z_max']<obj.low[2]+.025:continue
            name='lower_body' if i<cfg.profile_slices/3 else 'middle_body' if i<2*cfg.profile_slices/3 else 'top'
            if width<top_width*.75 and row['z_max']<high[2]-.015:name='lower_body'
            score=float(np.clip(.9-.15*uncertainty-.15*asymmetry,0,1))
            region_clearances={}
            for direction,vector in {'left':[-1,0],'right':[1,0],'front':[0,-1],'back':[0,1]}.items():
                axis=np.asarray(vector);perp=np.array([-axis[1],axis[0]]);available=.3
                contact_height=(row['z_min']+row['z_max'])/2
                for other in world.objects.values():
                    if other.id==obj.id or other.high[2]<contact_height-.04:continue
                    offset=other.position[:2]-obj.position[:2]
                    if offset@axis>0 and abs(offset@perp)<float((other.size[:2]+obj.size[:2])@np.abs(perp)/2)+.04:
                        available=min(available,float(offset@axis-(other.size[:2]+obj.size[:2])@np.abs(axis)/2))
                region_clearances[direction]=available
            regions.append({'name':name,'z_min':row['z_min'],'z_max':row['z_max'],'width':width,'graspability':score,'clearance':region_clearances})
        side_score=max((r['graspability'] for r in regions),default=0.) if max(clearances.values())>.05 else 0.
        complexity=float(np.clip(max(0.8 if not fit_top else 0.,variation,1-completeness,asymmetry,.8 if not top_clear else 0.),0,1))
        xy_bounds=np.percentile(points[:,:2],[1,99],axis=0)
        footprint_points=points[np.all((points[:,:2]>=xy_bounds[0])&(points[:,:2]<=xy_bounds[1]),axis=1)]
        if len(footprint_points)<8:footprint_points=points
        def nearest(fraction):
            return min(valid,key=lambda r:abs((r['z_min']+r['z_max'])/2-(obj.low[2]+height*fraction)))['width'] if valid else None
        model={'height':height,'top_width':top_width,'middle_width':nearest(.5),'lower_width':nearest(.2),
               'top_surface_area':top_area,'top_surface_normal':normal.tolist(),'width_profile':profile,
               'principal_axes_xy':basis.tolist(),'symmetry':1-asymmetry,'shape_irregularity':variation,
               'clearance':clearances,'gripper_fit_top':fit_top,'gripper_fit_side':bool(regions),
               'top_approach_collision_free':top_clear,'side_approach_collision_free':max(clearances.values())>.05,
               'top_grasp_score':top_score,'side_grasp_score':side_score,'complexity_score':complexity,
               'geometry_completeness':completeness,'graspable_regions':regions,'views_used':len(views or obj.view_ids),
               'collision_size':(np.maximum(high,obj.high)-np.minimum(low,obj.low)).tolist(),
               'collision_center_offset':((np.maximum(high,obj.high)+np.minimum(low,obj.low))/2-obj.position).tolist(),
               'footprint_offsets':(cv2.convexHull(footprint_points[:,:2].astype(np.float32)).reshape(-1,2)-obj.position[:2]).tolist(),
               'observed_bounds':[low.tolist(),high.tolist()],'support_contact_region':{'z':float(obj.low[2]),'extent':(high[:2]-low[:2]).tolist()}}
        obj.manipulation_geometry=model;return model
