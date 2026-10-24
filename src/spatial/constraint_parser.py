"""Explicit offline grammar; online interpretation uses the same strict schema."""
import re
from llm.schemas import Goal,Reference,SpatialConstraint,ManipulationConstraints

RELATIONS=[(('上方','above'),'above'),(('下方','below'),'below'),(('上边','上侧','upper'),'upper_of'),
           (('下边','下侧','lower'),'lower_of'),(('右边','右侧','右手边','right of'),'right_of'),
           (('左边','左侧','left of'),'left_of'),(('前面','前方','front of'),'front_of'),
           (('后面','后方','behind'),'behind'),(('旁边','附近','next to','near'),'near'),
           (('里面','放进','inside','into'),'inside')]


def explicit_distances(text):
    matches=re.finditer(r'(\d+(?:\.\d+)?)\s*(厘米|公分|cm|毫米|mm|米|m)(?![a-z])',text,re.I)
    scales={'厘米':.01,'公分':.01,'cm':.01,'毫米':.001,'mm':.001}
    return [float(match[1])*scales.get(match[2].lower(),1.) for match in matches]


def distance(text):
    values=explicit_distances(text)
    return values[0] if values else None


def constraints_from_text(command):
    mappings={'keep_upright':['保持竖直','杯口朝上','不要弄倒','keep upright'],
              'avoid_top_contact':['不要碰顶部','avoid top'],'prefer_handle':['抓住把手','抓把手','by the handle'],
              'fragile':['易碎','fragile'],'avoid_squeezing':['不要挤压','avoid squeezing'],
              'orientation_preservation':['保持朝向','preserve orientation'],'no_sliding':['不滑动','no sliding','不要推']}
    return ManipulationConstraints(explicit=[key for key,phrases in mappings.items() if any(p in command.lower() for p in phrases)])


def parse_local(command,reference):
    text=command.lower();split=re.search(r'拿到|放到|放在|移到|移动到|搬到|放进|\b(?:onto|into|to|on)\b',text)
    if split is None:raise ValueError('请说明要移动什么，以及放在哪里')
    source=reference(command[:split.start()]);destination=command[split.end():]
    clauses=re.split(r'[,，。;；]|并且|且',destination)
    primary=clauses[0];target=reference(primary);relation=next((r for words,r in RELATIONS if any(w in primary.lower() for w in words)),'on')
    frame='object' if '正前方' in primary else 'robot_base' if '机械臂' in primary else 'world' if '世界坐标' in primary else 'operator'
    semantics='center_distance' if any(w in text for w in ['中心距离','center distance']) else 'edge_gap'
    if '之间' in primary or 'between' in primary.lower():
        terms=re.split(r'和|与|\band\b',re.sub(r'之间|between','',primary,flags=re.I))
        if len(terms)!=2:raise ValueError('between requires two explicit references')
        refs=[reference(t) for t in terms];target=refs[0]
        spatial=[SpatialConstraint(relation='between',references=refs,reference_frame=frame)]
    else:spatial=[SpatialConstraint(relation=relation,distance_m=distance(primary),distance_semantics=semantics,reference_frame=frame)]
    for clause in clauses[1:]:
        if any(word in clause.lower() for word in ['至少','at least','不小于','最多','at most','不超过']):
            kind='max_distance' if any(w in clause.lower() for w in ['最多','at most','不超过']) else 'min_distance'
            spatial.append(SpatialConstraint(type=kind,reference=reference(clause),distance_m=distance(clause),distance_semantics=semantics))
    if not (source.color or source.category) or not (target.color or target.category):raise ValueError('请提供物体颜色或类别')
    return Goal(goal='place_spatial',object=source,target=target,spatial_constraints=spatial,manipulation=constraints_from_text(command))
