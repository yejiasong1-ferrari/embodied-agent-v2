import numpy as np
import cv2
from .object_detector import ColorComponentDetector,GroundingDINOAdapter,VLMDetector
from .segmenter import ColorSegmenter,SAM2Adapter
from .pose_estimator import PoseEstimator
from .debug_visualizer import save_observation,draw_observation
from .depth_components import split_mask_3d
from .object_detector import Detection


class ActivePerception:
    def __init__(self,camera,world,config,client,log):
        self.camera,self.world,self.config,self.client,self.log=camera,world,config,client,log
        p=config.perception
        self.detector=(GroundingDINOAdapter(p.dino_model) if p.detector=="grounding_dino" else
                       VLMDetector(client) if p.detector=="vlm" else ColorComponentDetector(p.min_pixels))
        self.segmenter=SAM2Adapter(p.sam_config,p.sam_checkpoint) if p.segmenter=="sam2" else ColorSegmenter()
        self.estimator=PoseEstimator(p,config.safety.workspace)
        self.observation_count=0; self.last_frame=None; self.last_failure=None
        self.image_sink=None
        self.annotations={}
        from .visual_grounder import VisualGrounder
        from .object_localizer import ObjectLocalizer
        self.grounder=VisualGrounder(client,config,log);self.localizer=ObjectLocalizer(config,log)
        self.trackers={};self.references=[];self.phase='coarse';self.last_ground_view={}
        self.last_ground_time={}
        self.local_detector=ColorComponentDetector(p.min_pixels)
        self.reviewed_references={}
        self.task_description=""

    def set_references(self,references):self.references=list(references)

    def grounded_detections(self,frame,image,queries,reground):
        from .target_tracker import TargetTracker
        from llm.schemas import Reference
        detections=self.local_detector.detect(image,queries)
        for query in queries:
            tracker=self.trackers.get(query);grounded=None
            changed_view=self.last_ground_view.get(query)!=frame.view_id
            if tracker is not None and not reground and not changed_view:grounded=tracker.update(frame.rgb,image)
            retry_due=frame.timestamp-self.last_ground_time.get(query,-float('inf'))>=self.config.localization.reground_interval_s
            if grounded is None and (reground or changed_view or tracker is None or retry_due):
                grounded=self.grounder.ground(frame.rgb,query,image);self.last_ground_view[query]=frame.view_id
                self.last_ground_time[query]=frame.timestamp
                if grounded.found:
                    tracker=TargetTracker(self.config.localization.tracking_iou_min);tracker.initialize(frame.rgb,grounded);self.trackers[query]=tracker
            if grounded is None or not grounded.found or grounded.confidence<self.config.perception.confidence_threshold:continue
            box=np.asarray(grounded.bbox)
            def same_component(d):
                lo=np.maximum(box[:2],d.bbox[:2]);hi=np.minimum(box[2:],d.bbox[2:])
                area=np.prod(np.maximum(hi-lo,0));den=max(1,np.prod(np.asarray(d.bbox[2:])-d.bbox[:2]))
                return d.color==grounded.color and area/den>.4
            detections=[d for d in detections if not same_component(d)]
            detection=grounded.detection();detection.query=query;detections.append(detection)
        return detections

    def observe(self,queries=(),reground=False):
        queries=list(dict.fromkeys([r.description for r in self.references]+list(queries)))
        frame=self.camera.capture(); self.last_frame=frame
        image=frame.rgb.copy()
        if frame.robot_mask is not None: image[frame.robot_mask]=0
        # Remove the table/background using measured depth before connected
        # components. This prevents equal-colour objects merging with floor.
        from .pointcloud import pixel_to_world
        v,u=np.indices(frame.depth.shape)
        valid=np.isfinite(frame.depth)&(frame.depth>self.config.perception.min_depth_m)&(frame.depth<self.config.perception.max_depth_m)
        foreground=np.zeros_like(valid)
        pts=pixel_to_world(np.column_stack((u[valid],v[valid])),frame.depth[valid],frame.intrinsics,frame.camera_to_world)
        support_valid=np.ones(len(pts),bool) if frame.robot_mask is None else ~frame.robot_mask[valid]
        self.world.update_support(pts[support_valid],frame.timestamp)
        foreground[valid]=(pts[:,2]>.009)&(pts[:,2]<self.config.safety.workspace[2][1])
        image[~foreground]=0
        detections=(self.grounded_detections(frame,image,queries,reground) if self.config.localization.enabled and queries
                    else self.detector.detect(image,list(queries)))
        masks=self.segmenter.segment(image,detections)
        # A tracked crop can overlap the detector's complete component. Union
        # those measured pixels before 3-D splitting, so one physical surface
        # cannot create a second identity and later a phantom obstacle.
        unique=[]
        for det,mask in zip(detections,masks):
            match=next((i for i,(old,oldmask) in enumerate(unique) if old.color==det.color
                and np.count_nonzero(oldmask&mask)>.5*min(np.count_nonzero(oldmask),np.count_nonzero(mask))),None)
            if match is None:unique.append((det,mask));continue
            old,oldmask=unique[match];merged_mask=oldmask|mask
            semantic=det if getattr(det,'query','') else old
            unique[match]=(semantic,merged_mask)
        separated=[]
        for det,mask in unique:
            for component in split_mask_3d(frame,mask,self.config.perception.min_pixels):
                ys,xs=np.nonzero(component)
                part=Detection((int(xs.min()),int(ys.min()),int(xs.max()+1),int(ys.max()+1)),det.label,det.color,det.confidence,component)
                part.query=getattr(det,'query','');separated.append((part,component))
        detections=[item[0] for item in separated]; masks=[item[1] for item in separated]
        objects=[];records={}
        for det,mask in zip(detections,masks):
            obj=self.estimator.estimate(frame,det,mask,self.world.objects.values())
            if obj is not None:
                objects.append(obj);records[id(obj)]=(det,mask)
                obj.bbox=det.bbox
                x0,y0,x1,y1=det.bbox
                if x0<=1 or y0<=1 or x1>=frame.rgb.shape[1]-1 or y1>=frame.rgb.shape[0]-1:
                    obj.view_quality=0.
                    obj.observation_clipped=True
        # Merge lighting-induced colour fragments of the same surface; use
        # the larger visible component's colour, never a scene catalogue.
        merged=[]
        for obj in sorted(objects,key=lambda o:len(o.points),reverse=True):
            connected=None
            for index,other in enumerate(merged):
                if obj.color!=other.color or not len(obj.points) or not len(other.points):continue
                gap=np.maximum(np.maximum(obj.points.min(0)-other.points.max(0),other.points.min(0)-obj.points.max(0)),0)
                held=self.world.objects.get(self.world.memory.held_id)
                predicted=self.world.memory.held_expected_position
                shift=np.zeros(3) if held is None or predicted is None else predicted-held.position
                held_fragments=(held is not None and held.color==obj.color and all(
                    np.all(part.points>=held.low+shift-.03) and np.all(part.points<=held.high+shift+.03) for part in (obj,other)))
                if np.linalg.norm(gap)<=self.config.perception.voxel_m*3 or held_fragments:
                    connected=index;break
            if connected is not None:
                other=merged[connected];a,ma=records[id(other)];b,mb=records[id(obj)];mask=ma|mb
                ys,xs=np.nonzero(mask);semantic=a if getattr(a,'query','') else b
                det=Detection((int(xs.min()),int(ys.min()),int(xs.max()+1),int(ys.max()+1)),semantic.label,obj.color,max(a.confidence,b.confidence),mask)
                det.query=getattr(semantic,'query','')
                combined=self.estimator.estimate(frame,det,mask,self.world.objects.values())
                if combined is not None:
                    combined.observation_clipped=other.observation_clipped or obj.observation_clipped
                    if combined.observation_clipped:combined.view_quality=0.
                    combined.bbox=det.bbox;records[id(combined)]=(det,mask);merged[connected]=combined;continue
            if any((np.linalg.norm(obj.position[:2]-other.position[:2])<.035
                   and (obj.color==other.color or abs(obj.high[2]-other.high[2])<.025))
                   or (np.all(obj.position[:2]>=other.low[:2]-.01) and np.all(obj.position[:2]<=other.high[:2]+.01)
                       and abs(obj.high[2]-other.high[2])<.015) for other in merged): continue
            merged.append(obj)
        objects=merged
        self.world.update(objects,frame.timestamp)
        self.observation_count+=1
        for obj in objects:
            det,mask=records[id(obj)]
            refs=[r for r in self.references if (not r.color or r.color==obj.color) and (not r.category or r.category==obj.category)]
            query=getattr(det,'query','') or (refs[0].description if len(refs)==1 else '')
            if query:
                self.localizer.update(obj,frame,det,mask,query,self.phase)
                obj.mask_ref=f'images/{self.observation_count:05d}_{obj.id}_geometry.npz'
                np.savez_compressed(self.log.path/obj.mask_ref,mask=mask,points=obj.points,camera_to_world=frame.camera_to_world,
                                    intrinsics=frame.intrinsics,timestamp=frame.timestamp)
        self.last_failure=None if objects else "DEPTH_INVALID" if not np.any(np.isfinite(frame.depth)&(frame.depth<self.config.perception.max_depth_m)) else "TARGET_NOT_FOUND"
        self.log.event("observation",{"view":frame.view_id,"objects":[o.to_dict() for o in objects],"failure":self.last_failure})
        self.log.world(self.world)
        save_observation(self.log,self.observation_count,frame,detections,masks,objects,self.annotations)
        if self.image_sink:
            self.image_sink(frame.rgb,draw_observation(frame,detections,masks,objects,self.annotations),self.world.to_dict())
        return objects

    def scan(self,references=(),force_all=False):
        self.set_references(references or self.references)
        queries=[r.description for r in references]
        samples={};views_used=0
        # A lone round silhouette is not enough to finish a search for a cup
        # or bottle. Complete the low-resolution scan so a later viewpoint
        # can reveal the actual vessel and competing lookalikes.
        semantic_search=any(r.category in {'cup','bottle'} and
            (r.description,r.color,r.category) not in self.world.reference_bindings for r in references)
        # Establish observed obstacles before moving a robot-mounted camera.
        if self.last_frame is None: self.observe(queries)
        for index in range(min(self.camera.viewpoint_count,self.config.localization.max_search_views)):
            self.camera.update_collision_world(self.world)
            if not self.camera.move_viewpoint(index):
                self.log.event("viewpoint_rejected",{"index":index}); continue
            observed=self.observe(queries,reground=self.config.localization.enabled)
            views_used+=1
            for obj in observed:
                samples.setdefault(obj.id,[]).append((obj.position.copy(),obj.size.copy()))
            if not force_all and not semantic_search and references and all(self.ready(r) for r in references): return True
        # Independent views vote on the pose. A single hand-occluded view
        # cannot drag the target centre away from the other two observations.
        for identity,values in samples.items():
            if identity not in self.world.objects: continue
            obj=self.world.objects[identity]
            if len(values)>1:
                obj.position=np.median([p for p,_ in values],axis=0)
                obj.size=np.median([s for _,s in values],axis=0)
            obj.visible=True
        self.world.revision+=1
        from world.scene_graph import SceneGraph
        self.world.relationships=SceneGraph().build(self.world.objects.values())
        self.log.world(self.world)
        self.log.event("multiview_fusion",{"views":views_used,"objects":len(samples)})
        return all(self.ready(r) for r in references) if references else bool(self.world.objects)

    def ready(self,reference):
        try: obj=self.world.resolve(reference)
        except LookupError: return False
        return obj.visible and obj.confidence>=self.config.perception.confidence_threshold and obj.uncertainty_m<=self.config.perception.max_uncertainty_m

    def identify(self,reference):
        """Resolve the user's description to a measured, tracked object."""
        from world.world_model import AmbiguousReferenceError,ReferenceNotFoundError

        key=(reference.description,reference.color,reference.category)
        if key in self.world.reference_bindings:
            try:return self.world.resolve(reference,require_visible=True)
            except ReferenceNotFoundError:pass
        exact=[o for o in [*self.world.objects.values(),*self.world.regions.values()]
               if o.visible and (not reference.color or o.color==reference.color)
               and (not reference.category or reference.category=='object' or o.category==reference.category)]
        if reference.category=='region':
            return self.world.resolve(reference,require_visible=True)
        related=self.world.related_candidates(reference)
        # An explicit image grounding of the user's phrase can disagree with
        # the provisional geometric category. That disagreement needs review
        # even when another object is an exact category match.
        exact_ids={o.id for o in exact}
        def category_support(obj):
            return sum(scores.get(reference.category,0.) for scores in obj.category_observations.values())
        competing=[o for o in related if o.id not in exact_ids and
                   ((o.query==reference.description and o.semantic_confidence>=.85)
                    or (reference.category is not None and category_support(o)>=.85))]
        if len(exact)==1 and not competing:
            obj=exact[0]
            if (reference.category=='cup' and obj.size[2]/max(min(obj.size[:2]),.01)<.65):
                self.log.event('target_candidates',{'query':reference.description,
                    'exact_ids':[obj.id],'related_ids':[obj.id],'reason':'shallow_shape'})
                if self.disambiguate(reference,[obj]):
                    return self.world.resolve(reference,require_visible=True)
                raise AmbiguousReferenceError(f"reference {reference.description!r}: shallow shape unconfirmed")
            self.world.bind_reference(reference,obj.id,{'source':'unique_coarse_candidate',
                'confidence':obj.semantic_confidence or obj.confidence})
            self.log.event('target_binding',{'query':reference.description,'object_id':obj.id,
                'requested_category':reference.category,'provisional_category':obj.category,
                'source':'unique_coarse_candidate'})
            return obj
        candidates=related if related else exact
        self.log.event('target_candidates',{'query':reference.description,
            'exact_ids':[o.id for o in exact],
            'related_ids':[o.id for o in candidates],
            'reason':'category_conflict' if not exact or competing else 'multiple_exact_matches'})
        if not candidates:
            raise ReferenceNotFoundError(f"reference {reference.description!r}: 0 candidates")
        if self.disambiguate(reference,candidates):
            return self.world.resolve(reference,require_visible=True)
        raise AmbiguousReferenceError(f"reference {reference.description!r}: visual identity unconfirmed among "
                                      f"{[o.id for o in candidates]}")

    def disambiguate(self,reference,candidates):
        """Review measured candidates from real 1280x960 camera frames once.

        The world-model IDs only connect pixels across views; they contain no
        simulator identity or privileged object label.
        """
        from itertools import product
        from pathlib import Path
        from PIL import Image,ImageDraw,ImageOps
        from .pointcloud import world_to_pixel
        from llm.schemas import VisualDisambiguation

        key=(reference.description,reference.color,reference.category)
        candidates=sorted(candidates,key=lambda o:o.id)
        signature=tuple((o.id,tuple(sorted(o.view_ids)),tuple(round(float(s),2) for s in o.size)) for o in candidates)
        if self.reviewed_references.get(key)==signature:return False
        self.reviewed_references[key]=signature
        labels={o.id:chr(ord('A')+i) for i,o in enumerate(candidates)}
        by_label={label:object_id for object_id,label in labels.items()}
        cfg=self.config.localization
        if (len(candidates)>4 or not self.client.online or cfg.grounding_mode=='local'
                or self.grounder.calls>=cfg.max_vlm_calls or not hasattr(self.camera,'capture_detail')):
            self.log.event('visual_disambiguation',{'query':reference.description,'candidate_ids':[o.id for o in candidates],
                'selected':None,'reason':'detail capture or model unavailable'})
            return False

        # Select previously productive scan viewpoints, covering all candidates
        # with as few moves and detail captures as possible.
        available={i:{o.id for o in candidates if f'{self.camera.config.mode == "wrist" and "gripper_camera" or "agent_rgbd"}:{i}' in o.view_ids}
                   for i in range(min(self.camera.viewpoint_count,cfg.max_search_views))}
        uncovered={o.id for o in candidates};views=[]
        while uncovered and len(views)<3:
            remaining=[i for i in available if i not in views]
            if not remaining:break
            index=max(remaining,key=lambda i:len(available[i]&uncovered))
            if not available[index]&uncovered:break
            views.append(index);uncovered-=available[index]
        if uncovered:
            self.log.event('visual_disambiguation',{'query':reference.description,'candidate_ids':[o.id for o in candidates],
                'selected':None,'reason':'no observed viewpoint covers every candidate'})
            return False
        # One extra angle can expose a handle or rim that is hidden by the
        # wrist in the first covering view. It is still a bounded detail scan.
        if len(views)<3:
            remaining=[i for i in available if i not in views and available[i]]
            if remaining:
                views.append(max(remaining,key=lambda i:len(available[i])))

        patches={o.id:[] for o in candidates};captured=[]
        for index in views:
            self.camera.update_collision_world(self.world)
            if not self.camera.move_viewpoint(index):continue
            frame=self.camera.capture_detail()
            if frame.rgb.shape[:2]!=(self.config.camera.detail_height,self.config.camera.detail_width):
                raise ValueError('detail camera did not return calibrated high-resolution pixels')
            path=Path(self.log.path)/'images'/f'detail_{self.observation_count:05d}_view_{index}.png'
            Image.fromarray(frame.rgb).save(path)
            captured.append({'view':index,'path':str(path.relative_to(self.log.path)),
                             'size':[frame.rgb.shape[1],frame.rgb.shape[0]]})
            for obj in candidates:
                if obj.id not in available[index]:continue
                corners=np.asarray(list(product(*zip(obj.low,obj.high))),float)
                pixels,z=world_to_pixel(corners,frame)
                if not np.all(z>.01) or not np.all(np.isfinite(pixels)):continue
                x0,y0=np.min(pixels,axis=0);x1,y1=np.max(pixels,axis=0)
                cx,cy=(x0+x1)/2,(y0+y1)/2
                side=max(110.,1.9*max(x1-x0,y1-y0))
                left=max(0,int(cx-side/2));top=max(0,int(cy-side/2))
                right=min(frame.rgb.shape[1],int(cx+side/2));bottom=min(frame.rgb.shape[0],int(cy+side/2))
                if right-left<45 or bottom-top<45:continue
                # A foreground surface near the projected measured object
                # must still be present in this frame; do not review a crop
                # containing only table or an intervening robot link.
                depth=frame.depth[top:bottom,left:right]
                if not np.any(np.isfinite(depth)&(np.abs(depth-np.median(z))<.12)):continue
                patches[obj.id].append((index,Image.fromarray(frame.rgb[top:bottom,left:right])))

        if any(not row for row in patches.values()):
            self.log.event('visual_disambiguation',{'query':reference.description,'candidate_ids':[o.id for o in candidates],
                'selected':None,'reason':'a candidate was not visible in detail frames','captures':captured})
            return False
        columns=len(candidates);width=1280; height=960; margin=16
        tile_w=(width-(columns+1)*margin)//columns;tile_h=(height-3*margin)//2
        sheet=Image.new('RGB',(width,height),(20,24,29));draw=ImageDraw.Draw(sheet)
        for col,obj in enumerate(candidates):
            for row,(index,patch) in enumerate(patches[obj.id][:2]):
                x=margin+col*(tile_w+margin);y=margin+row*(tile_h+margin)
                fitted=ImageOps.contain(patch,(tile_w-12,tile_h-36),method=Image.Resampling.LANCZOS)
                sheet.paste(fitted,(x+(tile_w-fitted.width)//2,y+28))
                draw.text((x+8,y+7),f'Candidate {labels[obj.id]} / view {index}',fill=(255,230,120))
        review_path=Path(self.log.path)/'images'/f'detail_{self.observation_count:05d}_candidates.jpg'
        sheet.save(review_path,quality=90)
        self.log.event('detail_capture',{'query':reference.description,'candidate_ids':[o.id for o in candidates],
            'captures':captured,'review_image':str(review_path.relative_to(self.log.path)),
            'image_size':[width,height]})
        try:
            self.grounder.calls+=1
            answer=self.client.complete(VisualDisambiguation,'visual_disambiguation',{
                'user_command':self.task_description,'query':reference.description,
                'candidate_ids':[labels[o.id] for o in candidates],
                'candidate_observations':[
                    {'candidate_id':labels[o.id],'measured_color':o.color,'provisional_category':o.category,
                     'height_m':round(float(o.size[2]),3),'width_m':round(float(max(o.size[:2])),3),
                     'previously_grounded_as_query':o.query==reference.description}
                    for o in candidates],
                'instruction':'Identify which visible physical object the user means. Internal category labels are provisional; '
                              'different everyday nouns can refer to the same object. Compare all candidates and use visible '
                              'distinguishing evidence. Reject clear alternatives, mark unclear views uncertain. Do not guess.'},
                image=np.asarray(sheet))
        except RuntimeError as exc:
            self.log.event('visual_disambiguation',{'query':reference.description,'selected':None,
                'reason':str(exc),'captures':captured})
            return False
        reviews={row.candidate_id:row for row in answer.candidates}
        ids=set(by_label)
        yes=[row for row in reviews.values() if row.verdict=='yes' and row.confidence>=.85]
        # Every other measured candidate must be rejected, so an occluded
        # second cup cannot be silently replaced by a guess.
        selected=(by_label.get(yes[0].candidate_id) if len(yes)==1 and set(reviews)==ids and
                  all(row.verdict=='no' and row.confidence>=.65
                      for row in reviews.values() if row.candidate_id!=yes[0].candidate_id)
                  else None)
        self.log.event('visual_disambiguation',{'query':reference.description,'candidate_ids':[o.id for o in candidates],
            'selected':selected,'reviews':[row.model_dump() for row in answer.candidates],
            'label_to_object_id':by_label,
            'review_image':str(review_path.relative_to(self.log.path))})
        if selected:
            chosen=reviews[labels[selected]]
            self.world.bind_reference(reference,selected,{'source':'high_resolution_visual_review',
                'confidence':chosen.confidence,'reason':chosen.evidence,
                'review_image':str(review_path.relative_to(self.log.path))})
            self.log.event('target_binding',{'query':reference.description,'object_id':selected,
                'requested_category':reference.category,
                'provisional_category':self.world.objects[selected].category if selected in self.world.objects else None,
                'source':'high_resolution_visual_review','evidence':chosen.evidence})
            return True
        return False
