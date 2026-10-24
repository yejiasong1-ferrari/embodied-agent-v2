from dataclasses import dataclass
from typing import Protocol
import cv2
import numpy as np


@dataclass
class Detection:
    bbox: tuple[int, int, int, int]
    label: str
    color: str
    confidence: float
    mask: np.ndarray | None = None


class ObjectDetector(Protocol):
    def detect(self, image: np.ndarray, text_queries: list[str]) -> list[Detection]: ...


# Generic colour vocabulary, independent of the scene's object catalogue.
HUE_BANDS = {"red": [(0, 9), (174, 179)], "orange": [(10, 24)], "yellow": [(25, 38)],
             "green": [(39, 82)], "cyan": [(83, 99)], "blue": [(100, 128)],
             "purple": [(129, 149)], "pink": [(150, 173)]}


def color_masks(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    h, s, v = cv2.split(hsv)
    masks = {color: (np.logical_or.reduce([(h >= lo) & (h <= hi) for lo, hi in bands]) & (s > 75) & (v > 45))
             for color, bands in HUE_BANDS.items()}
    masks["white"] = (s < 45) & (v > 145)
    masks["gray"] = (s < 75) & (v > 35) & (v <= 145)
    return masks


class ColorComponentDetector:
    """Honest closed-vocabulary fallback: pixels + components, no object lookup."""
    def __init__(self, min_pixels=30):
        self.min_pixels = min_pixels

    def detect(self, image, text_queries=()):
        result = []
        for color, raw in color_masks(image).items():
            mask = cv2.morphologyEx(raw.astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
            count, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
            for i in range(1, count):
                x, y, w, h, area = stats[i]
                if area >= self.min_pixels:
                    result.append(Detection((int(x), int(y), int(x+w), int(y+h)), "object", color,
                                            min(.95, .65 + area / 4000), labels == i))
        return result


class GroundingDINOAdapter:
    """Real HF inference adapter. Weights/packages are optional, never mocked."""
    def __init__(self, model_id, device="cpu"):
        try:
            import torch
            from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
        except ImportError as exc:
            raise RuntimeError("Grounding DINO requires torch + transformers and downloaded model weights") from exc
        self.torch = torch
        self.device = device
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(device).eval()

    def detect(self, image, text_queries):
        from PIL import Image
        text = ". ".join(text_queries or ["cup", "bottle", "box", "plate", "obstacle"]) + "."
        inputs = self.processor(images=Image.fromarray(image), text=text, return_tensors="pt").to(self.device)
        with self.torch.no_grad():
            output = self.model(**inputs)
        result = self.processor.post_process_grounded_object_detection(output, inputs.input_ids,
                    threshold=.3, text_threshold=.25, target_sizes=[image.shape[:2]])[0]
        masks = color_masks(image)
        detections = []
        for box, score, label in zip(result["boxes"], result["scores"], result.get("text_labels",result.get("labels",[]))):
            x1,y1,x2,y2 = map(int, box.tolist())
            color = max(masks, key=lambda k: np.count_nonzero(masks[k][max(0,y1):y2,max(0,x1):x2]))
            detections.append(Detection((x1,y1,x2,y2), label, color, float(score)))
        return detections


class VLMDetector:
    def __init__(self, client):
        self.client = client
        self.labels={}
        self.local=ColorComponentDetector()
        self.frames=0

    def detect(self, image, text_queries):
        from llm.schemas import VisualDetections,Reference
        self.frames+=1
        # VLM supplies low-frequency semantics; each intervening frame still
        # redetects pixel components and depth. No cached position is returned.
        if self.labels and (self.frames%40 or self.client.calls>=self.client.config.max_calls-2):
            detections=self.local.detect(image,text_queries)
            for detection in detections: detection.label=self.labels.get(detection.color,"object")
            return detections
        response = self.client.complete(VisualDetections, "visual_detection",
            {"queries": list(text_queries), "instruction": "List visible objects with normalized xyxy boxes. Do not invent occluded objects."}, image=image)
        height, width = image.shape[:2]
        groups={}
        for item in response.objects:
            item.color=Reference.normalize_vocabulary(item.color)
            item.label=Reference.normalize_vocabulary(item.label)
            groups.setdefault(item.color,set()).add(item.label)
        self.labels={color:next(iter(labels)) for color,labels in groups.items() if len(labels)==1}
        return [Detection((int(d.box[0]*width), int(d.box[1]*height), int(d.box[2]*width), int(d.box[3]*height)),
                          d.label, d.color, d.confidence) for d in response.objects]
