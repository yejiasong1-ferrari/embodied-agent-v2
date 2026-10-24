from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class OperatorFrame:
    origin: tuple = (0.,0.,0.)
    yaw_deg: float = 0.

    @property
    def transform(self):
        a=np.deg2rad(self.yaw_deg);t=np.eye(4)
        t[:3,:3]=[[np.cos(a),-np.sin(a),0],[np.sin(a),np.cos(a),0],[0,0,1]]
        t[:3,3]=self.origin;return t

    def to_world(self,point):return self.transform[:3,:3]@np.asarray(point)+self.transform[:3,3]
    def to_operator(self,point):return self.transform[:3,:3].T@(np.asarray(point)-self.transform[:3,3])


def relation_axis(relation,frame,config,reference=None):
    axis={'right_of':(1,0,0),'left_of':(-1,0,0),'upper_of':(0,1,0),'lower_of':(0,-1,0),
          'front_of':(0,-1,0),'behind':(0,1,0),'above':(0,0,1),'below':(0,0,-1)}.get(relation)
    if axis is None:return None
    rotation=np.eye(3)
    if frame=='operator':rotation=OperatorFrame(config.operator_origin,config.operator_yaw_deg).transform[:3,:3]
    elif frame=='object':
        if reference is None or not getattr(reference,'orientation_known',False):raise ValueError('AMBIGUOUS_REFERENCE_FRAME')
        x,y,z,w=np.asarray(reference.orientation)/np.linalg.norm(reference.orientation)
        rotation=np.array([[1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w)],
                           [2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w)],
                           [2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)]])
    return rotation@np.asarray(axis,float)
