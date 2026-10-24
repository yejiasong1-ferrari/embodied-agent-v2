"""Optical convention: x right, y down, z forward; depth is axial metres."""
import numpy as np
from .camera import CameraFrame


def pixel_to_camera(uv, depth, intrinsics):
    uv = np.asarray(uv, dtype=float)
    depth = np.asarray(depth, dtype=float)
    if np.any(~np.isfinite(depth)) or np.any(depth <= 0):
        raise ValueError("depth must be finite and positive")
    return np.stack(((uv[..., 0] - intrinsics[0, 2]) * depth / intrinsics[0, 0],
                     (uv[..., 1] - intrinsics[1, 2]) * depth / intrinsics[1, 1], depth), axis=-1)


def camera_to_world(points, transform):
    return np.asarray(points) @ transform[:3, :3].T + transform[:3, 3]


def pixel_to_world(uv, depth, intrinsics, transform):
    return camera_to_world(pixel_to_camera(uv, depth, intrinsics), transform)


def mask_pointcloud(frame: CameraFrame, mask, min_depth=.025, max_depth=3.0):
    valid = np.asarray(mask, bool) & np.isfinite(frame.depth) & (frame.depth > min_depth) & (frame.depth < max_depth)
    v, u = np.nonzero(valid)
    return pixel_to_world(np.column_stack((u, v)), frame.depth[v, u], frame.intrinsics, frame.camera_to_world) if len(v) else np.empty((0, 3))


def voxel_downsample(points, size=.004):
    if not len(points):
        return np.empty((0, 3))
    _, indices = np.unique(np.floor(np.asarray(points) / size).astype(int), axis=0, return_index=True)
    return np.asarray(points)[indices]


def world_to_pixel(points, frame):
    camera = (np.asarray(points) - frame.camera_to_world[:3, 3]) @ frame.camera_to_world[:3, :3]
    z = camera[..., 2]
    return np.column_stack((camera[..., 0] / z * frame.intrinsics[0, 0] + frame.intrinsics[0, 2],
                            camera[..., 1] / z * frame.intrinsics[1, 1] + frame.intrinsics[1, 2])), z
