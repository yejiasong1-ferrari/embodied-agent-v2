import numpy as np


def merge_observed_fragment(payload,hand_pose,fragment):
    """Grow the carried collision box using a confirmed rigid visual fragment."""
    if fragment.id in payload.get('merged_fragment_ids',()):return payload
    relative_rotation=np.asarray(payload['relative_rotation'])
    rotation=hand_pose.rotation@relative_rotation
    offset=np.asarray(payload.get('collision_offset',payload['offset']))
    center=hand_pose.position+hand_pose.rotation@offset
    half=np.asarray(payload.get('collision_size',payload['size']))/2
    corners=np.array([[x,y,z] for x in (fragment.low[0],fragment.high[0])
        for y in (fragment.low[1],fragment.high[1]) for z in (fragment.low[2],fragment.high[2])])
    local=(corners-center)@rotation
    low=np.minimum(-half,local.min(axis=0))
    high=np.maximum(half,local.max(axis=0))
    merged=dict(payload)
    merged['collision_size']=high-low
    merged['collision_offset']=offset+relative_rotation@((low+high)/2)
    merged['merged_fragment_ids']=[*payload.get('merged_fragment_ids',()),fragment.id]
    return merged
