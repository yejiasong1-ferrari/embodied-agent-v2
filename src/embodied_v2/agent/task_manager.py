"""Natural-language task semantics; never emits coordinates or joints."""

from __future__ import annotations

import re

from ..domain import Relation, SceneEntity, SceneModel, TaskSemantics
from .manager_client import ManagerClient
from .scene_memory import SceneMemory


_COLORS = {
    "红": "red", "红色": "red", "red": "red",
    "黄": "yellow", "黄色": "yellow", "yellow": "yellow",
    "绿": "green", "绿色": "green", "green": "green",
    "蓝": "blue", "蓝色": "blue", "blue": "blue",
    "白": "white", "白色": "white", "white": "white",
    "紫": "purple", "紫色": "purple", "purple": "purple",
    "青": "cyan", "cyan": "cyan", "粉": "pink", "pink": "pink",
    "橙": "orange", "orange": "orange",
}
_CATEGORIES = {
    "杯子": "cup", "杯": "cup", "cup": "cup", "mug": "cup",
    "瓶子": "bottle", "瓶": "bottle", "bottle": "bottle",
    "方块": "box", "盒子": "box", "box": "box", "block": "box",
    "盘子": "plate", "盘": "plate", "plate": "plate",
    "圆柱": "cylinder", "cylinder": "cylinder",
}


def _ordered_hits(text: str, vocabulary: dict[str, str]) -> list[tuple[int, str]]:
    hits: list[tuple[int, str]] = []
    lowered = text.lower()
    for token, value in vocabulary.items():
        for match in re.finditer(re.escape(token), lowered):
            hits.append((match.start(), value))
    output: list[tuple[int, str]] = []
    for hit in sorted(hits):
        if not output or hit[0] != output[-1][0]:
            output.append(hit)
    return output


_RELATION_SPLIT = re.compile(
    r"(?:拿到|放到|搬到|移到|放在|放进|置于|移动到|next\s+to|onto|\bto\b)",
    re.IGNORECASE,
)


def _task_category(entity: SceneEntity) -> str:
    """Convert measured geometry to the task vocabulary without object names."""
    if entity.geometry == "plate":
        return "plate"
    if entity.geometry == "box":
        return "box"
    if entity.geometry == "cylinder":
        width = max(float(entity.bounds.size[0]), float(entity.bounds.size[1]), 1e-6)
        return "bottle" if float(entity.bounds.size[2]) > 2.2 * width else "cup"
    return entity.geometry


def _first_value(text: str, vocabulary: dict[str, str]) -> str | None:
    hits = _ordered_hits(text, vocabulary)
    return hits[0][1] if hits else None


def _inventory_description(scene: SceneModel) -> str:
    return ", ".join(
        f"{item.entity_id}={item.color} {_task_category(item)}"
        for item in scene.entities
        if item.movable or item.support_candidate
    )


class NaturalLanguageTaskManager:
    def __init__(
        self,
        client: ManagerClient | None = None,
        use_online: bool = False,
        scene_memory: SceneMemory | None = None,
    ):
        self.client = client or ManagerClient()
        self.use_online = bool(use_online)
        self.scene_memory = scene_memory

    @staticmethod
    def _ground_one(
        scene: SceneModel,
        *,
        color: str | None,
        category: str | None,
        source: bool,
    ) -> SceneEntity:
        pool = (
            [item for item in scene.entities if item.movable]
            if source else list(scene.supports)
        )
        candidates = [
            item for item in pool
            if (color is None or item.color == color)
            and (category is None or _task_category(item) == category)
        ]
        role = "source" if source else "destination"
        if len(candidates) == 1:
            return candidates[0]
        details = _inventory_description(scene)
        if not candidates:
            raise ValueError(
                f"no observed {role} matches color={color}, category={category}; "
                f"camera inventory: {details}"
            )
        names = ", ".join(item.entity_id for item in candidates)
        raise ValueError(
            f"ambiguous {role}: {names}; add a color or category. "
            f"camera inventory: {details}"
        )

    def _ground_locally(self, instruction: str, scene: SceneModel) -> TaskSemantics:
        match = _RELATION_SPLIT.search(instruction)
        if match is not None:
            source_text = instruction[:match.start()]
            destination_text = instruction[match.end():]
        else:
            source_text = instruction
            destination_text = ""

        source_color = _first_value(source_text, _COLORS)
        source_category = _first_value(source_text, _CATEGORIES)
        destination_color = _first_value(destination_text, _COLORS)
        destination_category = _first_value(destination_text, _CATEGORIES)

        # For terse commands without a clear relation verb, preserve ordered
        # first/last semantics, then let the camera inventory fill omissions.
        colors = _ordered_hits(instruction, _COLORS)
        categories = _ordered_hits(instruction, _CATEGORIES)
        if match is None and source_color is None and colors:
            source_color = colors[0][1]
        if destination_color is None and len(colors) >= 2:
            destination_color = colors[-1][1]
        if match is None and source_category is None and categories:
            source_category = categories[0][1]
        if destination_category is None and len(categories) >= 2:
            destination_category = categories[-1][1]

        source = self._ground_one(
            scene, color=source_color, category=source_category, source=True
        )
        destination = self._ground_one(
            scene,
            color=destination_color,
            category=destination_category,
            source=False,
        )
        relation = Relation.NEXT_TO if any(
            token in instruction.lower() for token in ("旁边", "next to", "next_to")
        ) else Relation.ON
        return TaskSemantics(
            source.color,
            _task_category(source),
            destination.color,
            _task_category(destination),
            relation,
            any(token in instruction.lower() for token in ("易碎", "轻拿", "fragile")),
            instruction,
        )

    def understand(self, instruction: str, scene: SceneModel | None = None) -> TaskSemantics:
        if self.use_online and self.client.available:
            try:
                inventory = (
                    self.scene_memory.prompt_context()
                    if self.scene_memory is not None else (
                        _inventory_description(scene) if scene is not None else "[]"
                    )
                )
                result = self.client.json_completion(
                    "Extract only symbolic manipulation semantics as JSON: "
                    "object_color, object_category, destination_color, "
                    "destination_category, relation(on|next_to), fragile. "
                    "Ground every noun against the camera inventory supplied by the user. "
                    "Never invent an object and never output coordinates or joint angles.",
                    f"instruction={instruction}\ncamera_inventory={inventory}",
                )
                parsed = TaskSemantics(
                    str(result["object_color"]), str(result["object_category"]),
                    str(result["destination_color"]), str(result["destination_category"]),
                    Relation(str(result.get("relation", "on"))),
                    bool(result.get("fragile", False)), instruction,
                )
                if scene is not None:
                    self._ground_one(
                        scene,
                        color=parsed.object_color,
                        category=parsed.object_category,
                        source=True,
                    )
                    self._ground_one(
                        scene,
                        color=parsed.destination_color,
                        category=parsed.destination_category,
                        source=False,
                    )
                return parsed
            except (KeyError, ValueError, RuntimeError, OSError):
                pass
        if scene is not None:
            return self._ground_locally(instruction, scene)
        colors = _ordered_hits(instruction, _COLORS)
        categories = _ordered_hits(instruction, _CATEGORIES)
        if len(colors) < 2 or len(categories) < 2:
            raise ValueError("task must identify source and destination color/category")
        relation = Relation.NEXT_TO if any(
            token in instruction.lower() for token in ("旁边", "next to", "next_to")
        ) else Relation.ON
        return TaskSemantics(
            colors[0][1], categories[0][1], colors[-1][1], categories[-1][1],
            relation, any(token in instruction for token in ("易碎", "轻拿", "fragile")),
            instruction,
        )
