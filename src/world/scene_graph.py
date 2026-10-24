import numpy as np


def relation_satisfied(a, b, relation, z_tolerance=.025, gap=.04):
    overlap = np.minimum(a.high[:2], b.high[:2]) - np.maximum(a.low[:2], b.low[:2])
    delta = a.position-b.position
    if relation == "on":
        return bool(np.all(overlap > 0) and np.all(a.position[:2] >= b.low[:2])
                    and np.all(a.position[:2] <= b.high[:2]) and abs(a.low[2]-b.high[2]) <= z_tolerance)
    if relation == "inside":
        return bool(np.all(a.low >= b.low-z_tolerance) and np.all(a.high <= b.high+z_tolerance))
    if relation == "left_of": return bool(a.high[0] < b.low[0] and abs(delta[1]) < max(a.size[1],b.size[1])+.08)
    if relation == "right_of": return bool(a.low[0] > b.high[0] and abs(delta[1]) < max(a.size[1],b.size[1])+.08)
    if relation == "front_of": return bool(a.high[1] < b.low[1] and abs(delta[0]) < max(a.size[0],b.size[0])+.08)
    if relation == "behind": return bool(a.low[1] > b.high[1] and abs(delta[0]) < max(a.size[0],b.size[0])+.08)
    if relation == "above": return bool(a.low[2] > b.high[2]+z_tolerance)
    if relation == "below": return bool(a.high[2] < b.low[2]-z_tolerance)
    if relation == "overlapping": return bool(np.all(overlap > 0) and min(a.high[2],b.high[2]) > max(a.low[2],b.low[2]))
    if relation in {"near","next_to"}:
        separation = np.maximum(0, np.maximum(a.low[:2]-b.high[:2], b.low[:2]-a.high[:2]))
        return bool(np.linalg.norm(separation) <= .12 and not np.all(overlap>0))
    raise ValueError(f"unknown relation {relation}")


class SceneGraph:
    RELATIONS = ("left_of","right_of","above","below","on","inside","near","overlapping","front_of","behind")
    def build(self, objects):
        items = list(objects)
        return [{"source":a.id,"relation":r,"target":b.id} for a in items for b in items if a.id != b.id
                for r in self.RELATIONS if relation_satisfied(a,b,r)]
