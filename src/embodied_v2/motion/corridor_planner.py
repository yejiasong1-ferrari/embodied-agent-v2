"""Translate symbolic route intent into metric Cartesian guide points."""

from __future__ import annotations

import numpy as np

from ..domain import PayloadState, RouteIntent, SceneModel, Vec3, vec3


class CorridorPlanner:
    def waypoints(
        self,
        start_m: Vec3,
        destination_m: Vec3,
        intent: RouteIntent,
        payload: PayloadState,
        scene: SceneModel,
    ) -> tuple[Vec3, ...]:
        start = np.asarray(start_m, dtype=float)
        goal = np.asarray(destination_m, dtype=float)
        corridor = intent.corridor.lower()
        if "right" in corridor:
            lateral = max(start[1], goal[1], 0.48)
            xy_nodes = (
                start[:2], np.array((start[0], lateral)),
                np.array((goal[0], lateral)), goal[:2],
            )
        elif "left" in corridor:
            lateral = min(start[1], goal[1], -0.48)
            xy_nodes = (
                start[:2], np.array((start[0], lateral)),
                np.array((goal[0], lateral)), goal[:2],
            )
        elif "rear" in corridor or "back" in corridor:
            # The table's rear is the side farther from the destination in x.
            x = min(start[0], goal[0], -0.48)
            xy_nodes = (
                start[:2], np.array((x, start[1])),
                np.array((x, goal[1])), goal[:2],
            )
        else:
            lateral = 0.5 * (start[1] + goal[1])
            # A nominal "center" line can cut through the robot base's inner
            # unreachable disk when start and goal lie on opposite X sides.
            # Detour around that disk on the shorter Y side; exact collision
            # and IK validation still decide whether the route is usable.
            x_low, x_high = sorted((float(start[0]), float(goal[0])))
            # Preserve the pickup-side IK branch. A 24 cm tangent stays inside
            # the pedestal gap for the measured hand volume; the exact robot
            # collision checker remains authoritative.
            base_keepout_m = 0.24
            crosses_base_x = x_low <= 0.0 <= x_high
            if crosses_base_x and abs(float(lateral)) < 0.30:
                side = float(start[1]) if abs(float(start[1])) > 0.04 else float(goal[1])
                lateral = float(np.copysign(base_keepout_m, side or 1.0))
            xy_nodes = (
                start[:2], np.array((start[0], lateral)),
                np.array((goal[0], lateral)), goal[:2],
            )

        # Only objects intersecting the swept XY corridor dictate its height.
        # This avoids lifting over the tallest object on the entire table.
        horizontal_margin = (
            0.5 * max(payload.geometry.size[:2])
            + payload.envelope_margin_m
            + 0.015
        )

        def intersects_swept_path(entity) -> bool:
            for first, second in zip(xy_nodes, xy_nodes[1:]):
                low = np.minimum(first, second) - horizontal_margin
                high = np.maximum(first, second) + horizontal_margin
                bounds = entity.bounds
                if not (
                    bounds.maximum[0] < low[0]
                    or bounds.minimum[0] > high[0]
                    or bounds.maximum[1] < low[1]
                    or bounds.minimum[1] > high[1]
                ):
                    return True
            return False

        transport_obstacles = tuple(
            item for item in scene.entities
            if item.entity_id != payload.entity_id and intersects_swept_path(item)
        )
        obstacle_top = max(
            (item.bounds.maximum[2] for item in transport_obstacles), default=0.0
        )
        required_hand_z = (
            obstacle_top
            + payload.bottom_offset_m
            + payload.envelope_margin_m
            + 0.04
        )
        transit_z = max(start[2], goal[2], required_hand_z)
        if corridor == "center":
            # Downward-palm Panda IK has a low inner singular band near the
            # base.  A moderate transit height stays on one continuous wrist
            # branch; lower paths branch-switch, while pedestal-height outer
            # routes can be unnecessarily high.
            branch_height = 0.35 if float(start[1]) < 0.0 else 0.30
            transit_z = max(transit_z, branch_height)
        if not intent.high_clearance:
            transit_z = max(start[2], goal[2], min(transit_z, 0.52))
        return tuple(
            vec3((node[0], node[1], transit_z)) for node in xy_nodes
        )
