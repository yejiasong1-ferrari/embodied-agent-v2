"""Low-frequency semantic grounding of the *supplied* RGB image."""
from dataclasses import dataclass
import numpy as np
from llm.schemas import GroundedTarget,Reference
from .object_detector import Detection,ColorComponentDetector


@dataclass
class VisualDetection:
    found: bool
    query: str
    label: str = ''
    bbox: tuple | None = None
    confidence: float = 0.
    mask: np.ndarray | None = None
    color: str = ''
    category: str = 'object'
    backend: str = 'unavailable'

    def detection(self):return Detection(self.bbox,self.category,self.color,self.confidence,self.mask)


class VisualGrounder:
    def __init__(self,client,config,log):
        self.client,self.config,self.log=client,config,log;self.calls=0
        self.local=ColorComponentDetector(config.perception.min_pixels)
    def ground(self,image,query,local_image=None):
        cfg=self.config.localization;reference=Reference(description=query)
        # Keep model calls available for a later high-resolution review if
        # multiple visually plausible objects survive the coarse scan.
        coarse_limit=max(0,cfg.max_vlm_calls-cfg.detail_vlm_reserve)
        if cfg.grounding_mode!='local' and self.client.online and self.calls<coarse_limit:
            for attempt in range(2):
                if self.calls>=coarse_limit:break
                self.calls+=1
                try:
                    result=self.client.complete(GroundedTarget,'visual_grounding',{
                        'query':query,'image_width':image.shape[1],'image_height':image.shape[0],
                        'instruction':('Retry with coordinate_space="normalized" and decimal xyxy values from 0 to 1. '
                                       'The previous pixel box exceeded the image bounds.' if attempt else
                                       'Locate only this target in this current RGB. Return found=false if absent or ambiguous. '
                                       'bbox xyxy; state normalized or pixel coordinates. Never world coordinates.')},image=image)
                    box=None
                    if result.found:
                        box=np.asarray(result.bbox,float)
                        if result.coordinate_space=='normalized':box*=np.array([image.shape[1],image.shape[0]]*2)
                        box=np.rint(box).astype(int)
                        if np.any(box<0) or box[2]>image.shape[1] or box[3]>image.shape[0]:
                            raise ValueError('pixel bbox outside image')
                        box=tuple(map(int,box))
                    facets=Reference(description=result.label,color=result.color,category=result.category)
                    found=VisualDetection(result.found,query,result.label,box,result.confidence,color=facets.color or '',category=facets.category or 'object',backend='vlm')
                    self.log.event('visual_grounding',{'query':query,'found':found.found,'bbox':box,'coordinate_space':'pixel','confidence':found.confidence,'backend':'vlm'})
                    return found
                except ValueError:
                    if attempt==0 and self.calls<coarse_limit:
                        self.log.event('visual_grounding_retry',{'query':query,'reason':'box outside camera image','next':'normalized coordinates'})
                        continue
                    self.log.event('visual_fallback',{'query':query,'reason':'ValueError','backend':'local_rgb_components'})
                except RuntimeError as exc:
                    self.log.event('visual_fallback',{'query':query,'reason':type(exc).__name__,'backend':'local_rgb_components'})
                    break
        else:
            self.log.event('visual_fallback',{'query':query,'reason':'VLM unavailable or visual call budget reached','backend':'local_rgb_components'})
        # Local detection supplies color/shape pixels; the requested category is
        # NOT stamped onto a colored component as a fabricated recognition.
        candidates=[d for d in self.local.detect(image if local_image is None else local_image)
                    if reference.color is None or d.color==reference.color]
        if len(candidates)!=1:return VisualDetection(False,query,backend='local_rgb_components')
        d=candidates[0]
        result=VisualDetection(True,query,d.label,d.bbox,d.confidence,d.mask,d.color,'object','local_rgb_components')
        self.log.event('visual_grounding',{'query':query,'found':True,'bbox':d.bbox,'coordinate_space':'pixel','confidence':d.confidence,'backend':result.backend})
        return result
