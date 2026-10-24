"""Pure spectator-camera math. Has no robot, simulation or perception handles."""
from dataclasses import dataclass, replace
import math
import numpy as np

@dataclass(frozen=True)
class CameraState:
    azimuth: float = 135.
    elevation: float = -32.
    distance: float = 2.05
    lookat: tuple = (0., 0., .20)

    def orbit(self, dx, dy):
        return replace(self, azimuth=(self.azimuth-dx*.35)%360,
                       elevation=float(np.clip(self.elevation-dy*.3,-89.,-5.)))
    def zoom(self, delta):
        return replace(self,distance=float(np.clip(self.distance*math.exp(float(np.clip(-delta*.0015,-2,2))),.55,4.)))
    def pan(self, dx, dy, height):
        a=math.radians(self.azimuth);scale=self.distance/max(height,1)*.7
        x,y,z=self.lookat
        x+=(-dx*math.sin(a)-dy*math.cos(a))*scale
        y+=(dx*math.cos(a)-dy*math.sin(a))*scale
        return replace(self,lookat=(float(np.clip(x,-1.2,1.2)),float(np.clip(y,-1.2,1.2)),z))
    def focus(self, position):
        p=np.asarray(position,float)
        if p.shape!=(3,) or not np.isfinite(p).all():return self
        return replace(self,lookat=tuple(np.clip(p,[-1.2,-1.2,0],[1.2,1.2,1.2])))
    @classmethod
    def preset(cls,name):
        return {'Top':cls(90,-89,2.1,(0,0,.03)), 'Front':cls(90,-25),
                'Left':cls(180,-25),'Right':cls(0,-25)}.get(name,cls())


def project(position, projection):
    """Project WorldModel estimates using the renderer's camera calibration."""
    if not projection:return None
    v=np.asarray(position)-np.asarray(projection['pos'])
    forward=np.asarray(projection['forward']);up=np.asarray(projection['up'])
    right=np.cross(forward,up)
    depth=float(v@forward)
    if depth<=projection['near']:return None
    half=projection['tan_half_fov']*depth
    u=.5+float(v@right)/(2*half*projection['aspect'])
    w=.5-float(v@up)/(2*half)
    return (u,w,depth) if 0<=u<=1 and 0<=w<=1 else None
