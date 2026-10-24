import numpy as np


class ObjectMemory:
    def __init__(self, config):
        self.config = config
        self.objects = {}
        self.counter = 0
        self.held_id = None
        self.held_expected_position = None
        self.held_fragments = set()
        self._fragment_sightings = {}
        self._fragment_held_id = None
        self.grasped_heights = {}
        self.retained_ids = set()
        self.mission_active = False
        self.last_update = None

    def update(self, observations, timestamp):
        if self.held_id!=self._fragment_held_id:
            self.held_fragments.clear()
            self._fragment_sightings.clear()
            self._fragment_held_id=self.held_id
        elapsed=0 if self.last_update is None else max(0,timestamp-self.last_update)
        self.last_update=timestamp
        for obj in self.objects.values():
            obj.visible = False
        used = set()
        for observed in observations:
            candidates = [o for o in self.objects.values() if o.id not in used
                          and ((o.color == observed.color and np.linalg.norm(o.position[:2]-observed.position[:2]) < self.config.association_m)
                               or (not o.movable and not observed.movable and o.color==observed.color
                                   and np.all(np.minimum(o.high,observed.high)-np.maximum(o.low,observed.low)>-.003)
                                   and np.prod(np.maximum(0,np.minimum(o.high[:2],observed.high[:2])-np.maximum(o.low[:2],observed.low[:2])))>.0001)
                               or (o.id==self.held_id and self.held_expected_position is not None and o.color==observed.color
                                   and np.linalg.norm(np.asarray(self.held_expected_position)-observed.position)<max(.08,float(np.linalg.norm(o.size)/2)))
                               or (o.movable and observed.movable and max(o.size[:2])<.13 and max(observed.size[:2])<.13
                                   and np.linalg.norm(o.position[:2]-observed.position[:2])<.02
                                   and abs(o.high[2]-observed.high[2])<.015
                                   and np.all(observed.size[:2]>o.size[:2]*.5)))]
            if candidates:
                old = min(candidates, key=lambda o:np.linalg.norm((self.held_expected_position if o.id==self.held_id and self.held_expected_position is not None else o.position)-observed.position))
                observed.id = old.id
                observed.query=old.query
                observed.semantic_confidence=max(observed.semantic_confidence,old.semantic_confidence)
                observed.manipulation_geometry=old.manipulation_geometry
                observed.stability_displacement_m=float(np.linalg.norm(observed.position-old.position))
                if old.localization_state=='PRECISE_LOCKED' and observed.stability_displacement_m<.012:
                    observed.localization_state=old.localization_state
                observed.view_ids |= old.view_ids
                if (old.movable and old.id!=self.held_id
                        and min(observed.size[:2]/old.size[:2])<.8
                        and np.linalg.norm(observed.position[:2]-old.position[:2])<.04):
                    # A closing gripper reveals only one face. Its centroid
                    # must not redefine the attachment offset before lifting.
                    observed.position[:2]=old.position[:2]
                    observed.uncertainty_m=max(observed.uncertainty_m,.012)
                stable_plate=(old.category=='plate' and not old.observation_clipped and
                    not observed.observation_clipped and
                    np.max(np.abs(observed.size[:2]-old.size[:2]))<.015 and
                    np.linalg.norm(observed.position[:2]-old.position[:2])<.015)
                if old.category=="plate" and (min(observed.size[:2]/old.size[:2])<.85 or stable_plate):
                    # A cup/arm can split one disk into visible crescents.
                    # Keep its previously measured complete footprint. This
                    # observation confirms an exposed part, not a new centre.
                    observed.position[:2]=old.position[:2]
                    observed.size[:2]=old.size[:2]
                    if not stable_plate:observed.uncertainty_m=max(observed.uncertainty_m,.012)
                geometric_bottle=(old.category=='cup' and observed.category=='bottle'
                    and observed.size[2]/max(min(observed.size[:2]),.01)>1.9)
                if observed.view_quality<=old.view_quality+.02 and not geometric_bottle:
                    observed.color,observed.category,observed.movable=old.color,old.category,old.movable
                    observed.view_quality=old.view_quality
                votes={view:dict(scores) for view,scores in old.category_observations.items()}
                for view,scores in observed.category_observations.items():
                    previous=votes.setdefault(view,{})
                    for name,score in scores.items():previous[name]=max(previous.get(name,0.),score)
                observed.category_observations=votes
                if old.movable and observed.movable and len(votes)>=2:
                    totals={name:sum(row.get(name,0.) for row in votes.values())
                            for name in {name for row in votes.values() for name in row}}
                    ranked=sorted(totals,key=totals.get,reverse=True)
                    winner=ranked[0] if ranked else None
                    runner=totals[ranked[1]] if len(ranked)>1 else 0.
                    # Require independent viewpoints; repeat scans from one
                    # angle cannot overturn a previously tracked category.
                    support=sum(row.get(winner,0.)>0 for row in votes.values()) if winner else 0
                    if (winner in {'cup','bottle','box'} and support>=2
                            and totals[winner]>=runner+.45):
                        observed.category=winner
                # A top/oblique partial view may shrink the measured extent.
                # Retain already observed dimensions, never hidden simulator data.
                if old.category != "obstacle" and observed.category == "obstacle":
                    observed.category, observed.movable = old.category, old.movable
                if observed.movable and old.movable:
                    # Once measured on its support, upright object height is
                    # retained through a lift. The visible top updates pose;
                    # the surrounding table is no longer its bottom.
                    measured_top=observed.high[2]
                    if (observed.observation_clipped and old.id!=self.held_id
                            and np.linalg.norm(observed.position[:2]-old.position[:2])<.04):
                        # A cropped side face does not measure the top. Keep
                        # the last observed volume until a complete view can
                        # establish a new height; never sink an obstacle.
                        measured_top=max(measured_top,old.high[2])
                        observed.uncertainty_m=max(observed.uncertainty_m,.012)
                    observed.size[:2]=np.maximum(observed.size[:2],old.size[:2])
                    measured_height=old.size[2]
                    if old.id in self.grasped_heights:
                        # A released object partly hidden by its support can
                        # have its top seen while RGB-D support inference
                        # treats the underlying table as its bottom. Retain
                        # the complete height measured before the grasp.
                        measured_height=self.grasped_heights[old.id]
                    elif old.id!=self.held_id and not observed.observation_clipped:
                        # Later side views can reveal a stem hidden by a wide
                        # cap in the initial view. Grow from measured RGB-D
                        # support extent before grasping, then retain it in air.
                        measured_height=max(measured_height,observed.size[2])
                    observed.size[2]=measured_height
                    if (old.id!=self.held_id and old.low[2]<.012 and
                            np.linalg.norm(observed.position[:2]-old.position[:2])<.04 and
                            measured_top<old.high[2]-.012):
                        # An occluded top face cannot make a rigid upright
                        # object shrink while its measured support is still
                        # at the same place. Keep the previously seen top and
                        # report larger uncertainty until another view sees it.
                        measured_top=old.high[2]
                        observed.uncertainty_m=max(observed.uncertainty_m,.012)
                    observed.position[2]=measured_top-measured_height/2
                elif not stable_plate:
                    # Seeing just the top/front of a fixed obstacle must not
                    # move its remembered volume away from the unseen half.
                    low=np.minimum(observed.low,old.low);high=np.maximum(observed.high,old.high)
                    observed.position=(low+high)/2;observed.size=high-low
            else:
                self.counter += 1
                observed.id = f"{observed.color}_{observed.category}_{self.counter:03d}"
            observed.visible = True
            observed.semantic_label=f"{observed.color} {observed.category}"
            used.add(observed.id)
            self.objects[observed.id] = observed
        # A finger or cap can split one RGB component into two detections in
        # the same frame. Keep a single physical obstacle and a stable task ID.
        # Nested, equal-height footprints are stronger evidence than colour
        # alone; distinct nearby objects remain separate.
        objects=list(self.objects.values())
        for i,first in enumerate(objects):
            if first.id not in self.objects or not first.movable:continue
            for second in objects[i+1:]:
                if second.id not in self.objects or not second.movable or first.color!=second.color:continue
                overlap=np.maximum(0,np.minimum(first.high[:2],second.high[:2])-np.maximum(first.low[:2],second.low[:2]))
                small_area=min(float(np.prod(first.size[:2])),float(np.prod(second.size[:2])))
                if (small_area<=0 or float(np.prod(overlap))/small_area<.70 or
                        abs(first.size[2]-second.size[2])>.025 or abs(first.high[2]-second.high[2])>.04):continue
                def priority(obj):
                    suffix=obj.id.rsplit('_',1)[-1]
                    return (obj.id==self.held_id,obj.id in self.retained_ids,float(np.prod(obj.size[:2])),-int(suffix) if suffix.isdigit() else 0)
                keep,drop=(first,second) if priority(first)>=priority(second) else (second,first)
                if not keep.visible and drop.visible and np.prod(drop.size[:2])>=.8*np.prod(keep.size[:2]):
                    keep.position=drop.position.copy()
                keep.visible=keep.visible or drop.visible
                keep.last_seen=max(keep.last_seen,drop.last_seen)
                keep.confidence=max(keep.confidence,drop.confidence)
                measured_top=max(keep.high[2],drop.high[2])
                keep.size=np.maximum(keep.size,drop.size)
                keep.position[2]=measured_top-keep.size[2]/2
                if len(keep.points)==0 and len(drop.points):keep.points=drop.points
                del self.objects[drop.id]
                if drop is first:break
        # A fixed obstacle can appear as a full RGB component plus a narrow
        # visible strip in the same frame. The strip must not become a second
        # collision box: repeated partial views can otherwise inflate its
        # remembered union until it blocks an unrelated grasp.
        fixed=list(self.objects.values())
        for i,first in enumerate(fixed):
            if first.id not in self.objects:continue
            for second in fixed[i+1:]:
                if (second.id not in self.objects or (first.movable and second.movable)
                        or first.color!=second.color):continue
                shared_height=max(0.,min(first.high[2],second.high[2])-max(first.low[2],second.low[2]))
                if (abs(first.high[2]-second.high[2])>.025 or
                        shared_height/min(first.size[2],second.size[2])<.70):continue
                overlap=np.maximum(0,np.minimum(first.high[:2],second.high[:2])-np.maximum(first.low[:2],second.low[:2]))
                small=min(float(np.prod(first.size[:2])),float(np.prod(second.size[:2])))
                if small<=0 or float(np.prod(overlap))/small<.70:continue
                def fixed_priority(obj):
                    suffix=obj.id.rsplit('_',1)[-1]
                    return (not obj.movable,obj.id in self.retained_ids,float(np.prod(obj.size)),-int(suffix) if suffix.isdigit() else 0)
                keep,drop=(first,second) if fixed_priority(first)>=fixed_priority(second) else (second,first)
                keep.visible=keep.visible or drop.visible
                keep.last_seen=max(keep.last_seen,drop.last_seen)
                keep.confidence=max(keep.confidence,drop.confidence)
                keep.view_ids |= drop.view_ids
                del self.objects[drop.id]
                if drop is first:break
        for key, obj in list(self.objects.items()):
            age = max(0, timestamp-obj.last_seen)
            if not obj.visible:
                obj.confidence*=float(np.exp(-elapsed/max(self.config.memory_ttl_s,1e-6)))
                obj.uncertainty_m+=elapsed*.001
            if not self.mission_active and not obj.visible and age > self.config.memory_ttl_s and key != self.held_id and key not in self.retained_ids:
                del self.objects[key]
        held=self.objects.get(self.held_id)
        if held is not None and held.visible:
            for obj in self.objects.values():
                if obj.id==held.id or not obj.visible or not obj.movable or obj.color!=held.color:continue
                near_xy=np.linalg.norm(obj.position[:2]-held.position[:2])<(
                    .5*np.linalg.norm(obj.size[:2])+.5*np.linalg.norm(held.size[:2])+.025)
                shared_height=min(obj.high[2],held.high[2])-max(obj.low[2],held.low[2])
                if not near_xy or shared_height<.015:
                    self.held_fragments.discard(obj.id)
                    self._fragment_sightings.pop(obj.id,None)
                    continue
                previous=self._fragment_sightings.get(obj.id)
                if previous is not None:
                    old_held,old_fragment=previous
                    source_motion=held.position-old_held
                    fragment_motion=obj.position-old_fragment
                    if (np.linalg.norm(source_motion)>.03 and np.linalg.norm(fragment_motion)>.03
                            and np.linalg.norm(source_motion-fragment_motion)<.025):
                        self.held_fragments.add(obj.id)
                    if np.linalg.norm(source_motion)<=.03:continue
                self._fragment_sightings[obj.id]=(held.position.copy(),obj.position.copy())

