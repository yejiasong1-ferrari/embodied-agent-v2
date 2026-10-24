"""Register actual visible depth surfaces against the previously measured shape."""
import cv2
import numpy as np


def register_translation(observed,reference,max_shift=.04):
    observed=np.asarray(observed,float);reference=np.asarray(reference,float)
    if len(observed)<30 or len(reference)<30:return None,{}
    tree=cv2.flann_Index(np.asarray(reference,np.float32),dict(algorithm=1,trees=4))
    shift=np.zeros(3)
    for _ in range(8):
        indices,square=tree.knnSearch(np.asarray(observed-shift,np.float32),1,params={'checks':64})
        distances=np.sqrt(square[:,0]);valid=distances<.025
        if np.mean(valid)<.7:return None,{'reason':'insufficient_surface_overlap'}
        difference=observed[valid]-reference[indices[valid,0]]
        new=np.median(difference,axis=0)
        if np.linalg.norm(new)>max_shift:return None,{'reason':'surface_displacement_limit'}
        if np.linalg.norm(new-shift)<.0001:shift=new;break
        shift=new
    _,square=tree.knnSearch(np.asarray(observed-shift,np.float32),1,params={'checks':64})
    distances=np.sqrt(square[:,0]);residual=float(np.median(distances));fraction=float(np.mean(distances<.008))
    matched=observed[distances<.008]
    extent=np.ptp(matched,axis=0) if len(matched) else np.zeros(3)
    # Both clouds are partial: newly exposed surfaces need not have a prior
    # neighbour. Require a majority of actual correspondences spanning two
    # spatial axes, rather than pretending the old scan was a complete mesh.
    supported=len(matched)>=100 and np.count_nonzero(extent>.025)>=2
    evidence={'surface_residual_m':residual,'inlier_fraction':fraction,'matched_points':len(matched),
              'matched_extent_m':extent.tolist(),'translation_m':shift.tolist(),
              'consistent':bool(residual<.006 and fraction>.7 and supported),'source':'measured_multiview_cloud'}
    return (shift if evidence['consistent'] else None),evidence
