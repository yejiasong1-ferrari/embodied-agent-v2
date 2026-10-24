import cv2
import numpy as np
from world.world_model import WorldObject
from .pointcloud import mask_pointcloud, voxel_downsample


def fit_visible_circle(points,expected_radius=None):
    """RANSAC on the visible outer hull tolerates a hand/cup cutting the disk."""
    if len(points)<12: return None
    hull=cv2.convexHull(np.asarray(points,np.float32)).reshape(-1,2).astype(float)
    if len(hull)<6: return None
    rng=np.random.default_rng(0); best=None
    for _ in range(60):
        sample=hull[rng.choice(len(hull),3,replace=False)]
        design=np.column_stack((2*sample,np.ones(3)))
        if abs(np.linalg.det(design))<1e-8: continue
        fit=np.linalg.solve(design,np.sum(sample**2,axis=1))
        square=fit[2]+np.sum(fit[:2]**2)
        if square<=0: continue
        radius=np.sqrt(square)
        if not .015<radius<.20: continue
        if expected_radius is not None and not .8*expected_radius<radius<1.3*expected_radius: continue
        errors=np.abs(np.linalg.norm(hull-fit[:2],axis=1)-radius)
        inliers=errors<.003
        score=int(inliers.sum())
        if score>=6 and (best is None or score>best[0]): best=(score,fit[:2],radius)
    if best is None or best[0]<.55*len(hull): return None
    return best[1],best[2]


class PoseEstimator:
    def __init__(self, config, workspace):
        self.config, self.workspace = config, np.asarray(workspace)

    def estimate(self, frame, detection, mask, priors=()):
        points = mask_pointcloud(frame, mask, self.config.min_depth_m, self.config.max_depth_m)
        points = points[np.all((points >= self.workspace[:,0]) & (points <= self.workspace[:,1]),axis=1)]
        if len(points) < self.config.min_pixels:
            return None
        low, high = np.percentile(points, [2,98], axis=0)
        if high[2] < .012 or min(high[:2]-low[:2]) < .012:
            return None
        # Estimate supporting surface from neighbouring depth pixels, not a
        # catalogue height. The lower quantile rejects adjacent tall clutter.
        ring = cv2.dilate(mask.astype(np.uint8), np.ones((13,13),np.uint8)).astype(bool) & ~mask
        surroundings = mask_pointcloud(frame, ring, self.config.min_depth_m, self.config.max_depth_m)
        surroundings = surroundings[(surroundings[:,2] >= -.01) & (surroundings[:,2] < high[2]-.008)]
        if len(surroundings):
            support=max(0,float(np.percentile(surroundings[:,2],25)))
            if low[2]-support<max(.18,1.5*max(high[:2]-low[:2])):
                low[2]=min(low[2],support)
        size = np.maximum(high-low, .008)
        center = (low+high)/2
        nearby=[o for o in priors if o.color==detection.color and o.view_quality>.95
                and np.linalg.norm(o.position[:2]-center[:2])<.10]
        prior=min(nearby,key=lambda o:np.linalg.norm(o.position-center)) if nearby else None
        if max(size[:2]) > .34 or size[2] > .36:
            return None
        top = points[points[:,2] > high[2]-.012,:2].astype(np.float32)
        yaw = 0.0
        circular = False
        if len(top) >= 12:
            hull = cv2.convexHull(top)
            (_, _), (w,h), angle = cv2.minAreaRect(hull)
            circular = cv2.contourArea(hull)/max(w*h,1e-8) < .88
            yaw = np.deg2rad(angle)
            # Highest surface excludes lower handles. Its robust centre is
            # the useful body pose rather than the centre of a handle's box.
            top_low,top_high=np.percentile(top,[2,98],axis=0)
            if min(top_high-top_low)>.015:
                center[:2]=(top_low+top_high)/2
                size[:2]=top_high-top_low
        width = max(size[:2])
        if size[2] < .05 and width > .13:
            category = "plate"
        elif circular:
            category = "bottle" if size[2]/max(min(size[:2]), .01)>1.9 else "cup"
        else:
            category = "box"
        x0,y0,x1,y1=detection.bbox
        clipped=x0<=1 or y0<=1 or x1>=frame.rgb.shape[1]-1 or y1>=frame.rgb.shape[0]-1
        bbox_roundness=min(x1-x0,y1-y0)/max(x1-x0,y1-y0,1)
        if category=="plate" and detection.color!='gray' and not clipped and (bbox_roundness>=.70 or
                (prior is not None and prior.category=='plate')):
            circle=fit_visible_circle(top,expected_radius=prior.size[0]/2 if prior is not None and prior.category=='plate' else None)
            if circle is not None:
                # A narrow rectangular face can supply a few points on a
                # huge fitted circle. Without an existing plate observation,
                # never extrapolate its diameter beyond the visible span.
                if prior is not None or 2*circle[1]<=1.25*max(size[:2]):
                    center[:2]=circle[0]; size[:2]=2*circle[1]
        wall=points[(points[:,2]>low[2]+.012)&(points[:,2]<low[2]+.5*(high[2]-low[2])),:2]
        if len(wall)>=20 and (prior is None or prior.category in {"cup","bottle"}):
            design=np.column_stack((2*wall[:,0],2*wall[:,1],np.ones(len(wall))))
            fit=np.linalg.lstsq(design,np.sum(wall**2,axis=1),rcond=None)[0]
            radius_sq=fit[2]+np.sum(fit[:2]**2)
            if radius_sq>0:
                radius=np.sqrt(radius_sq)
                residual=np.median(np.abs(np.linalg.norm(wall-fit[:2],axis=1)-radius))
                if .015<radius<min(.06,.65*max(size[:2])) and residual<.0025:
                    center[:2]=fit[:2]
                    size[:2]=2*radius
                    category="bottle" if size[2]/(2*radius)>1.9 else "cup"
        geometric_category=category
        category_scores={geometric_category:.25 if clipped else .6}
        if detection.label not in {"object",""}:
            if detection.label in {"cup","bottle","box","plate","obstacle"}:
                category_scores[detection.label]=category_scores.get(detection.label,0.)+.9*float(detection.confidence)
            # A vision model can confidently call a tiny, top-down cylinder
            # a cup. Measured RGB-D aspect ratio is stronger evidence against
            # that label when the body is clearly bottle-shaped.
            if detection.label=="cup" and size[2]/max(min(size[:2]),.01)>1.9:
                category="bottle"
            else:
                category=detection.label
        movable = max(size[:2]) <= .13 and size[2] <= .24 and category != "plate"
        if detection.color == "gray" or not movable and category != "plate":
            category = "obstacle"
            movable = False
        # Large white components are robot links, identified geometrically.
        if detection.color == "white" and max(size[:2]) > .13:
            return None
        valid_fraction = len(points)/max(1,np.count_nonzero(mask))
        confidence = float(min(detection.confidence, .7+.25*valid_fraction))
        return WorldObject("", f"{detection.color} {category}", detection.color, category, center,
                           np.array([0,0,np.sin(yaw/2),np.cos(yaw/2)]), size, confidence, frame.timestamp,
                           movable=movable, uncertainty_m=max(.003, .018*(1-valid_fraction)),
                           points=voxel_downsample(points,self.config.voxel_m), view_ids={frame.view_id},
                           view_quality=float(abs(frame.camera_to_world[2,2])),
                           category_observations={frame.view_id:category_scores})
