from world.scene_graph import relation_satisfied


class PlacementValidator:
    def __init__(self,config): self.config=config
    def validate(self,source,target,relation):
        visible=source.visible and target.visible
        relation_ok=relation_satisfied(source,target,relation,self.config.safety.placement_z_tolerance_m)
        return bool(visible and relation_ok),{"source":source.to_dict(),"target":target.to_dict(),"relation":relation,"relation_satisfied":relation_ok,"fresh_visual_evidence":visible}
