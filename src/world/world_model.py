from dataclasses import dataclass, field
import numpy as np


class ReferenceNotFoundError(LookupError): pass
class AmbiguousReferenceError(LookupError): pass


@dataclass
class WorldObject:
    id: str
    semantic_label: str
    color: str
    category: str
    position: np.ndarray
    orientation: np.ndarray
    size: np.ndarray
    confidence: float
    last_seen: float
    visible: bool = True
    movable: bool = True
    uncertainty_m: float = .01
    points: np.ndarray = field(default_factory=lambda: np.empty((0,3)), repr=False)
    view_ids: set[str] = field(default_factory=set)
    view_quality: float = .0
    observation_clipped: bool = False
    evidence_source: str = "rgbd_pixels"
    category_observations: dict[str,dict[str,float]] = field(default_factory=dict)

    query: str = ""
    bbox: tuple | None = None
    mask_ref: str | None = None
    visual_state: str = "DISCOVERED"
    localization_state: str = "COARSE_LOCKED"
    semantic_confidence: float = 0.
    valid_depth_ratio: float = 0.
    observation_camera_pose: np.ndarray | None = field(default=None,repr=False)
    observation_range_m: float | None = None
    manipulation_geometry: dict = field(default_factory=dict)
    orientation_known: bool = False
    stability_displacement_m: float | None = None

    @property
    def low(self): return self.position - self.size / 2
    @property
    def high(self): return self.position + self.size / 2

    def to_dict(self):
        return {"id":self.id, "semantic_label":self.semantic_label,"color":self.color,"category":self.category,
                "position":self.position.tolist(),"orientation":self.orientation.tolist(),"size":self.size.tolist(),
                "confidence":float(self.confidence),"last_seen":float(self.last_seen),"visible":bool(self.visible),"movable":bool(self.movable),
                "query":self.query,"bbox":self.bbox,"mask_ref":self.mask_ref,"visual_state":self.visual_state,
                "localization_state":self.localization_state,"semantic_confidence":self.semantic_confidence,
                "valid_depth_ratio":self.valid_depth_ratio,"observation_range_m":self.observation_range_m,
                "observation_camera_pose":None if self.observation_camera_pose is None else self.observation_camera_pose.tolist(),
                "manipulation_geometry":self.manipulation_geometry,"orientation_known":self.orientation_known,
                "stability_displacement_m":self.stability_displacement_m,
                "observation_clipped":self.observation_clipped,
                "uncertainty_m":float(self.uncertainty_m),"view_ids":sorted(self.view_ids),"source":self.evidence_source,
                "category_observations":self.category_observations}


class WorldModel:
    def __init__(self, config):
        from .object_memory import ObjectMemory
        self.memory = ObjectMemory(config)
        self.objects = self.memory.objects
        self.revision = 0
        self.timestamp = 0.0
        self.relationships = []
        self.regions = {}
        self.reference_bindings = {}
        self.reference_binding_evidence = {}
        self.support_points = np.empty((0,3))
        self._support_cells = {}

    def update_support(self,points,timestamp):
        """Measured table coverage; absence of detections is not free space."""
        points=points[(np.abs(points[:,2])<.008)]
        if len(points):
            cells=np.floor(points[:,:2]/.012).astype(int)
            _,indices=np.unique(cells,axis=0,return_index=True)
            for i in indices: self._support_cells[tuple(cells[i])]=(points[i],timestamp)
        if not self.memory.mission_active:
            self._support_cells={k:v for k,v in self._support_cells.items() if timestamp-v[1]<self.memory.config.memory_ttl_s}
        self.support_points=np.array([v[0] for v in self._support_cells.values()]).reshape(-1,3)
        for region in self.regions.values():
            # Validate the exposed rim, even after an object covers the centre.
            local=points[np.all(np.abs(points[:,:2]-region.position[:2])<region.size[:2]/2,axis=1)]
            region.visible=len(local)>12
            if region.visible: region.last_seen=timestamp

    def update(self, observations, timestamp):
        from .scene_graph import SceneGraph
        self.memory.update(observations, timestamp)
        self.timestamp = timestamp
        self.revision += 1
        self.relationships = SceneGraph().build(self.objects.values())
        return self

    def resolve(self, reference, require_visible=False):
        key=(reference.description,reference.color,reference.category)
        bound_id=self.reference_bindings.get(key)
        if bound_id is not None:
            bound=self.objects.get(bound_id) or self.regions.get(bound_id)
            # A visually verified reference names a tracked physical object.
            # Its provisional colour/category may change in later observations.
            if bound is not None and (not require_visible or bound.visible):
                return bound
            self.reference_bindings.pop(key,None)
            self.reference_binding_evidence.pop(key,None)
        matches = [o for o in [*self.objects.values(),*self.regions.values()] if (not require_visible or o.visible)
                   and (not reference.color or o.color == reference.color)
                   and (not reference.category or o.category == reference.category
                        or reference.category == "object")]
        if not matches:
            raise ReferenceNotFoundError(f"reference {reference.description!r}: 0 candidates")
        if len(matches)>1:
            raise AmbiguousReferenceError(f"reference {reference.description!r}: {len(matches)} candidates")
        return matches[0]

    def related_candidates(self,reference,require_visible=True):
        """Recall measured objects for visual review without requiring an exact label."""
        return [o for o in [*self.objects.values(),*self.regions.values()]
                if (not require_visible or o.visible) and
                ((reference.color is None or o.color==reference.color) or
                 (o.query==reference.description and o.semantic_confidence>=.85))]

    def bind_reference(self,reference,object_id,evidence=None):
        if object_id not in self.objects and object_id not in self.regions:
            raise ReferenceNotFoundError(f"cannot bind missing object {object_id!r}")
        key=(reference.description,reference.color,reference.category)
        self.reference_bindings[key]=object_id
        self.reference_binding_evidence[key]=evidence or {}

    def to_dict(self):
        return {"revision":self.revision,"timestamp":self.timestamp,"objects":[o.to_dict() for o in self.objects.values()],
                "relationships":self.relationships,"regions":[r.to_dict() for r in self.regions.values()],
                "reference_bindings":[{"reference":key[0],"object_id":identity,"evidence":self.reference_binding_evidence.get(key,{})}
                                      for key,identity in self.reference_bindings.items()]}
