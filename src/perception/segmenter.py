from typing import Protocol
import numpy as np
from .object_detector import color_masks


class Segmenter(Protocol):
    def segment(self, image, boxes): ...


class ColorSegmenter:
    def segment(self, image, boxes):
        palettes = color_masks(image)
        results = []
        for item in boxes:
            if item.mask is not None:
                results.append(item.mask.copy())
                continue
            x1,y1,x2,y2 = item.bbox
            gate = np.zeros(image.shape[:2], bool)
            gate[max(0,y1):min(image.shape[0],y2),max(0,x1):min(image.shape[1],x2)] = True
            results.append(gate & palettes.get(item.color, np.ones_like(gate)))
        return results


class SAM2Adapter:
    def __init__(self, config_file, checkpoint, device="cpu"):
        try:
            from sam2.build_sam import build_sam2
            from sam2.sam2_image_predictor import SAM2ImagePredictor
        except ImportError as exc:
            raise RuntimeError("SAM2 requires the official sam2 package, torch and a checkpoint") from exc
        self.predictor = SAM2ImagePredictor(build_sam2(config_file, checkpoint, device=device))

    def segment(self, image, boxes):
        self.predictor.set_image(image)
        result = []
        for box in boxes:
            masks, scores, _ = self.predictor.predict(box=np.asarray(box.bbox), multimask_output=True)
            result.append(masks[int(np.argmax(scores))].astype(bool))
        return result
