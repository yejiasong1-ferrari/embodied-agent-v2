"""Appearance/flow tracking has no access to object positions or simulator IDs."""
import cv2
import numpy as np
from .visual_grounder import VisualDetection
from .object_detector import ColorComponentDetector


class TargetTracker:
    def __init__(self,iou_min=.15):self.previous=None;self.target=None;self.local=ColorComponentDetector();self.iou_min=iou_min
    def initialize(self,image,target):self.previous=cv2.cvtColor(image,cv2.COLOR_RGB2GRAY);self.target=target
    def update(self,image,local_image=None):
        if self.target is None or self.target.bbox is None:return None
        gray=cv2.cvtColor(image,cv2.COLOR_RGB2GRAY);h,w=gray.shape;x1,y1,x2,y2=self.target.bbox
        mask=np.zeros((h,w),np.uint8);mask[max(0,y1):min(h,y2),max(0,x1):min(w,x2)]=255
        points=cv2.goodFeaturesToTrack(self.previous,60,.02,3,mask=mask)
        predicted=np.array([x1,y1,x2,y2],float)
        if points is not None and len(points)>=3:
            following,status,_=cv2.calcOpticalFlowPyrLK(self.previous,gray,points,None)
            if following is not None and status.sum()>=3:
                delta=np.median((following-points)[status.ravel()==1].reshape(-1,2),axis=0);predicted+=np.tile(delta,2)
        detections=[d for d in self.local.detect(image if local_image is None else local_image) if d.color==self.target.color]
        scores=[]
        for d in detections:
            box=np.array(d.bbox);lo=np.maximum(predicted[:2],box[:2]);hi=np.minimum(predicted[2:],box[2:])
            intersection=float(np.prod(np.maximum(0,hi-lo)));union=np.prod(predicted[2:]-predicted[:2])+np.prod(box[2:]-box[:2])-intersection
            scores.append(intersection/max(union,1.))
        if not scores or max(scores)<self.iou_min:self.previous=gray;return None
        d=detections[int(np.argmax(scores))]
        result=VisualDetection(True,self.target.query,self.target.label,d.bbox,min(self.target.confidence,d.confidence),d.mask,d.color,self.target.category,'rgb_tracker')
        self.initialize(image,result);return result
