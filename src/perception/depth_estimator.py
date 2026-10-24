import numpy as np


class DepthEstimator:
    """Validate measured depth; never manufacture depth for missing pixels."""
    def __init__(self, config):
        self.config = config

    def valid_mask(self, frame, mask):
        return (np.asarray(mask, bool) & np.isfinite(frame.depth)
                & (frame.depth > self.config.min_depth_m) & (frame.depth < self.config.max_depth_m))
