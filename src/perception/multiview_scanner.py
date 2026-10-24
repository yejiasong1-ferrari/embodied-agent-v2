import numpy as np
from manipulation.viewpoint_planner import ViewpointPlanner
from .geometry_analyzer import ObjectCloudFusion,GeometryAnalyzer
from recovery.failure_classifier import SkillFailure


class MultiViewObjectScanner:
    def __init__(self,config):self.config=config;self.fusion=ObjectCloudFusion(config);self.analyzer=GeometryAnalyzer(config)
    def scan(self,c):
        if c.held:raise SkillFailure('GRASP_FAILED',{'reason':'object reconstruction cannot move a held object'})
        views=0;points,used=self.fusion.add(c.source);model=self.analyzer.analyze(c.source,c.world,points,used)
        camera=c.perception.camera
        if getattr(camera.config,'mode','')!='wrist':
            # Explicit external sensor mode remains available for regression;
            # production wrist mode never swaps to a spectator/gantry camera.
            poses=range(camera.viewpoint_count)
        else:poses=ViewpointPlanner(self.config).candidates(c.source,camera,side=True)
        for pose in poses:
            if views>=self.config.grasp.max_multiview_observations:break
            c.budget();c.log.event('stage',{'stage':'MULTIVIEW_SCAN','view':views+1})
            try:
                if isinstance(pose,int):
                    camera.update_collision_world(c.world)
                    if not camera.move_viewpoint(pose):continue
                else:
                    # Observation poses may need a large wrist rotation. Plan
                    # its whole joint-space route against the observed world;
                    # forcing that rotation at an unrelated high waypoint can
                    # be unreachable even when the sensor pose itself is safe.
                    c.robot.configure_collision_world(c.world,None,None,None)
                    path=c.planner.plan_to_pose(pose)
                    if path is None:
                        c.log.event('viewpoint_rejected',{'reason':c.planner.last_failure,'position':pose.position.tolist()});continue
                    ok,reason=c.executor.execute(path)
                    if not ok:
                        from recovery.failure_classifier import FailureClassifier
                        raise SkillFailure(FailureClassifier.classify(reason),{'reason':reason})
                c.perception.phase='local';c.perception.observe(reground=True);c.refresh()
            except SkillFailure as exc:
                if exc.failure.value in {'COLLISION_RISK','USER_STOP'}:raise
                c.log.event('viewpoint_rejected',{'reason':exc.failure.value});continue
            if not c.source.visible:continue
            views+=1;points,used=self.fusion.add(c.source);model=self.analyzer.analyze(c.source,c.world,points,used)
            c.log.event('manipulation_geometry',{'object_id':c.source.id,'model':model,'scan_views':views})
            if views>=2 and model.get('gripper_fit_top') and model['geometry_completeness']>=self.config.grasp.min_geometry_completeness and model['graspable_regions']:break
        path=c.log.path/f'fused_object_cloud_{c.perception.observation_count:05d}.npz'
        np.savez_compressed(path,points=points,views=sorted(used))
        c.log.event('object_multiview_complete',{'object_id':c.source.id,'scan_views':views,'cloud_ref':path.name,'points':len(points)})
        return model,points,views
