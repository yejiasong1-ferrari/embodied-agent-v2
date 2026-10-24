"""Tabletop layout: cluttered kitchen props and a fixed pickable arrangement."""

from __future__ import annotations

import mujoco
import numpy as np

from object_registry import record_for_body

# First-look base angles matching the fixed layout below.
OBJECT_BEARING_DEG: dict[str, float] = {
    "orange": -90.0,
    "red": -25.0,
    "yellow": 41.4,
    "blue": 147.0,
    "green": -142.0,
    "cyan": 45.0,
    "purple": 180.0,
    "white": 120.0,
}

# x, y, z, yaw.  The yellow plate stays at (0.34, 0.30).
FIXED_POSES: dict[str, tuple[float, float, float, float]] = {
    "cube": (0.00, -0.48, 0.276, 0.0),
    "red_cup": (0.38, -0.18, 0.050, 0.0),
    "blue_box": (-0.34, 0.22, 0.040, 0.0),
    "green_block": (-0.32, -0.25, 0.040, 0.0),
    "cyan_mug": (0.16, 0.16, 0.040, 0.0),
    "purple_bottle": (-0.42, 0.02, 0.060, 0.0),
    "white_mug": (-0.24, 0.42, 0.040, 0.0),
    # H acceptance fixture. Color and category are intentionally absent from
    # the object registry; perception has to inventory it.
    "pink_bottle": (0.22, -0.30, 0.040, 0.0),
}

PLATE_XY = np.array((0.34, 0.30))


def _yaw_quat(yaw: float) -> tuple[float, float, float, float]:
    half = 0.5 * float(yaw)
    return (float(np.cos(half)), 0.0, 0.0, float(np.sin(half)))


def add_scene_bodies(spec: mujoco.MjSpec) -> None:
    """Pedestals, destination plate, kitchen clutter, and pickable objects."""
    for name, y_position in (("left_pedestal", -0.48), ("right_pedestal", 0.48)):
        pedestal = spec.worldbody.add_body(name=name, pos=(0.0, y_position, 0.125))
        pedestal.add_geom(
            type=mujoco.mjtGeom.mjGEOM_BOX,
            size=(0.14, 0.14, 0.125),
            rgba=(0.22, 0.25, 0.29, 1.0),
        )

    plate = spec.worldbody.add_body(name="plate", pos=(0.34, 0.30, 0.018))
    plate.add_geom(
        name="plate_geom",
        type=mujoco.mjtGeom.mjGEOM_CYLINDER,
        size=(0.12, 0.018),
        rgba=(0.92, 0.88, 0.25, 1.0),
    )

    books = spec.worldbody.add_body(name="clutter_books", pos=(0.50, -0.40, 0.035))
    books.add_geom(
        type=mujoco.mjtGeom.mjGEOM_BOX,
        size=(0.085, 0.11, 0.018),
        rgba=(0.42, 0.24, 0.14, 1.0),
    )
    books.add_geom(
        type=mujoco.mjtGeom.mjGEOM_BOX,
        pos=(0.0, 0.01, 0.028),
        size=(0.075, 0.10, 0.012),
        rgba=(0.55, 0.18, 0.16, 1.0),
    )

    thermos = spec.worldbody.add_body(name="clutter_thermos", pos=(0.50, 0.10, 0.09))
    thermos.add_geom(
        type=mujoco.mjtGeom.mjGEOM_CYLINDER,
        size=(0.028, 0.09),
        rgba=(0.42, 0.44, 0.46, 1.0),
    )
    thermos.add_geom(
        type=mujoco.mjtGeom.mjGEOM_CYLINDER,
        pos=(0.0, 0.0, 0.095),
        size=(0.022, 0.012),
        rgba=(0.28, 0.30, 0.32, 1.0),
    )

    can = spec.worldbody.add_body(name="clutter_can", pos=(0.24, -0.44, 0.055))
    can.add_geom(
        type=mujoco.mjtGeom.mjGEOM_CYLINDER,
        size=(0.033, 0.055),
        rgba=(0.62, 0.64, 0.66, 1.0),
    )

    bowl = spec.worldbody.add_body(name="clutter_bowl", pos=(-0.46, -0.44, 0.022))
    bowl.add_geom(
        type=mujoco.mjtGeom.mjGEOM_CYLINDER,
        size=(0.055, 0.022),
        rgba=(0.32, 0.30, 0.28, 1.0),
    )

    tray = spec.worldbody.add_body(name="clutter_tray", pos=(-0.50, 0.38, 0.012))
    tray.add_geom(
        type=mujoco.mjtGeom.mjGEOM_BOX,
        size=(0.12, 0.16, 0.012),
        rgba=(0.58, 0.42, 0.26, 1.0),
    )

    jar = spec.worldbody.add_body(name="clutter_jar", pos=(0.12, 0.54, 0.04))
    jar.add_geom(
        type=mujoco.mjtGeom.mjGEOM_CYLINDER,
        size=(0.022, 0.04),
        rgba=(0.20, 0.22, 0.24, 1.0),
    )

    cereal = spec.worldbody.add_body(name="clutter_box", pos=(0.52, -0.16, 0.08))
    cereal.add_geom(
        type=mujoco.mjtGeom.mjGEOM_BOX,
        size=(0.035, 0.055, 0.08),
        rgba=(0.72, 0.48, 0.18, 1.0),
    )

    cube = spec.worldbody.add_body(name="cube", pos=(0.0, -0.48, 0.276))
    cube.add_freejoint(name="cube_joint")
    cube.add_geom(
        name="cube_geom",
        type=mujoco.mjtGeom.mjGEOM_BOX,
        size=(0.025, 0.025, 0.025),
        mass=0.08,
        friction=(1.0, 0.01, 0.001),
        rgba=(0.95, 0.35, 0.08, 1.0),
    )

    red_cup = spec.worldbody.add_body(name="red_cup", pos=(0.38, -0.18, 0.050))
    red_cup.add_freejoint(name="red_cup_joint")
    red_cup.add_geom(
        name="red_cup_geom",
        type=mujoco.mjtGeom.mjGEOM_CYLINDER,
        size=(0.035, 0.050),
        mass=0.020,
        friction=(1.0, 0.01, 0.001),
        rgba=(0.85, 0.08, 0.06, 1.0),
    )
    red_cup.add_geom(
        name="red_cup_handle",
        type=mujoco.mjtGeom.mjGEOM_CAPSULE,
        fromto=(0.04, 0.0, 0.025, 0.09, 0.0, 0.025),
        size=(0.012,),
        mass=0.001,
        friction=(1.0, 0.01, 0.001),
        contype=0,
        conaffinity=0,
        rgba=(0.85, 0.08, 0.06, 1.0),
    )

    blue_box = spec.worldbody.add_body(name="blue_box", pos=(-0.34, 0.22, 0.04))
    blue_box.add_freejoint(name="blue_box_joint")
    blue_box.add_geom(
        name="blue_box_geom",
        type=mujoco.mjtGeom.mjGEOM_BOX,
        size=(0.035, 0.030, 0.040),
        mass=0.09,
        friction=(1.1, 0.01, 0.001),
        rgba=(0.08, 0.25, 0.90, 1.0),
    )

    green_block = spec.worldbody.add_body(name="green_block", pos=(-0.32, -0.25, 0.04))
    green_block.add_freejoint(name="green_block_joint")
    green_block.add_geom(
        name="green_block_geom",
        type=mujoco.mjtGeom.mjGEOM_BOX,
        size=(0.035, 0.030, 0.040),
        mass=0.08,
        friction=(1.1, 0.01, 0.001),
        rgba=(0.10, 0.70, 0.20, 1.0),
    )

    cyan_mug = spec.worldbody.add_body(name="cyan_mug", pos=(0.16, 0.16, 0.040))
    cyan_mug.add_freejoint(name="cyan_mug_joint")
    cyan_mug.add_geom(
        name="cyan_mug_geom",
        type=mujoco.mjtGeom.mjGEOM_CYLINDER,
        size=(0.032, 0.040),
        mass=0.018,
        friction=(1.0, 0.01, 0.001),
        rgba=(0.05, 0.70, 0.75, 1.0),
    )
    cyan_mug.add_geom(
        name="cyan_mug_handle",
        type=mujoco.mjtGeom.mjGEOM_CAPSULE,
        fromto=(0.035, 0.0, 0.012, 0.078, 0.0, 0.012),
        size=(0.010,),
        mass=0.001,
        contype=0,
        conaffinity=0,
        rgba=(0.05, 0.70, 0.75, 1.0),
    )

    bottle = spec.worldbody.add_body(name="purple_bottle", pos=(-0.42, 0.02, 0.060))
    bottle.add_freejoint(name="purple_bottle_joint")
    bottle.add_geom(
        name="purple_bottle_geom",
        type=mujoco.mjtGeom.mjGEOM_CYLINDER,
        size=(0.026, 0.060),
        mass=0.035,
        friction=(1.0, 0.01, 0.001),
        rgba=(0.52, 0.16, 0.72, 1.0),
    )
    bottle.add_geom(
        name="purple_bottle_neck",
        type=mujoco.mjtGeom.mjGEOM_CYLINDER,
        pos=(0.0, 0.0, 0.072),
        size=(0.014, 0.016),
        mass=0.004,
        friction=(1.0, 0.01, 0.001),
        rgba=(0.42, 0.12, 0.58, 1.0),
    )

    white_mug = spec.worldbody.add_body(name="white_mug", pos=(-0.24, 0.42, 0.040))
    white_mug.add_freejoint(name="white_mug_joint")
    white_mug.add_geom(
        name="white_mug_geom",
        type=mujoco.mjtGeom.mjGEOM_CYLINDER,
        size=(0.032, 0.040),
        mass=0.018,
        friction=(1.0, 0.01, 0.001),
        rgba=(0.94, 0.94, 0.92, 1.0),
    )
    white_mug.add_geom(
        name="white_mug_handle",
        type=mujoco.mjtGeom.mjGEOM_CAPSULE,
        fromto=(0.035, 0.0, 0.012, 0.078, 0.0, 0.012),
        size=(0.010,),
        mass=0.001,
        contype=0,
        conaffinity=0,
        rgba=(0.94, 0.94, 0.92, 1.0),
    )

    # Unregistered fixture: a short pink cylinder in the open foreground
    # at (0.22, -0.30). No registry entry. A colliding neck made the
    # existing carry clearance reject the place, so the body is a plain cylinder.
    pink_bottle = spec.worldbody.add_body(name="pink_bottle", pos=(0.22, -0.30, 0.040))
    pink_bottle.add_freejoint(name="pink_bottle_joint")
    pink_bottle.add_geom(
        name="pink_bottle_geom",
        type=mujoco.mjtGeom.mjGEOM_CYLINDER,
        size=(0.028, 0.040),
        mass=0.022,
        friction=(1.0, 0.01, 0.001),
        rgba=(0.93, 0.15, 0.45, 1.0),
    )


def place_movable(model: mujoco.MjModel, data: mujoco.MjData) -> None:
    """Put pickable objects back on the designed tabletop.  The plate stays put."""
    for name, (x, y, z, yaw) in FIXED_POSES.items():
        qpos = int(model.joint(f"{name}_joint").qposadr[0])
        qw, qx, qy, qz = _yaw_quat(yaw)
        data.qpos[qpos : qpos + 7] = (x, y, z, qw, qx, qy, qz)
        record = record_for_body(name)
        if record is not None:
            OBJECT_BEARING_DEG[record.primary_color] = float(
                np.rad2deg(np.arctan2(y, x))
            )
        elif name == "pink_bottle":
            OBJECT_BEARING_DEG["pink"] = float(np.rad2deg(np.arctan2(y, x)))
    OBJECT_BEARING_DEG["yellow"] = float(
        np.rad2deg(np.arctan2(PLATE_XY[1], PLATE_XY[0]))
    )
