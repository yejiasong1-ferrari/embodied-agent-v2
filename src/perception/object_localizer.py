import numpy as np


class ObjectLocalizer:
    def __init__(self,config,log):self.config=config;self.log=log;self.previous={}
    def update(self,obj,frame,detection,mask,query,phase='coarse'):
        cfg=self.config.localization;valid=mask&np.isfinite(frame.depth)&(frame.depth>self.config.perception.min_depth_m)&(frame.depth<self.config.perception.max_depth_m)
        ratio=float(np.count_nonzero(valid)/max(1,np.count_nonzero(mask)))
        depths=frame.depth[valid];median=float(np.median(depths)) if len(depths) else None
        obj.query=query;obj.bbox=tuple(detection.bbox);obj.semantic_confidence=float(detection.confidence)
        obj.valid_depth_ratio=ratio;obj.observation_camera_pose=frame.camera_to_world.copy();obj.observation_range_m=median
        obj.localization_state='COARSE_LOCKED';obj.visual_state='DISCOVERED'
        old=self.previous.get(obj.id);motion=None if old is None else float(np.linalg.norm(obj.position-old[0]))
        obj.stability_displacement_m=motion
        if phase=='local' and old is not None:
            # Two fresh close views must agree. A label alone never raises precision.
            uncertainty=max(obj.uncertainty_m,motion/2)
            obj.uncertainty_m=uncertainty
            if (not obj.observation_clipped and ratio>=cfg.min_valid_depth_ratio and median is not None and median<=cfg.precise_max_range_m
                and uncertainty<=cfg.precise_uncertainty_m and 0<frame.timestamp-old[1]<=cfg.max_lock_age_s
                and old[2]=='local' and old[3] is not None and old[3]<=cfg.precise_max_range_m
                and old[4]>=cfg.min_valid_depth_ratio and detection.confidence>=self.config.perception.confidence_threshold):
                obj.localization_state='PRECISE_LOCKED'
        self.previous[obj.id]=(obj.position.copy(),frame.timestamp,phase,median,ratio)
        self.log.event('localization',{'object_id':obj.id,'query':query,'state':obj.localization_state,'phase':phase,
            'position':obj.position.tolist(),'uncertainty_m':obj.uncertainty_m,'semantic_confidence':obj.semantic_confidence,
            'valid_depth_ratio':ratio,'range_m':median,'bbox':obj.bbox,'camera':frame.view_id,'timestamp':frame.timestamp})
        return obj
    def precise(self,obj,now):
        cfg=self.config.localization
        return (obj.visible and not obj.observation_clipped and obj.localization_state=='PRECISE_LOCKED' and now-obj.last_seen<=cfg.max_lock_age_s
                and obj.uncertainty_m<=cfg.precise_uncertainty_m and obj.valid_depth_ratio>=cfg.min_valid_depth_ratio)
