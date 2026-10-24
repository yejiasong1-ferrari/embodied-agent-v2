"""Sensor evidence, overlays, planned paths and a local HTML inspection page."""
import html
import json
import cv2
import numpy as np
from .pointcloud import world_to_pixel


def draw_observation(frame,detections,masks,objects,annotations=None):
    output=frame.rgb.copy()
    for detection,mask in zip(detections,masks):
        output[mask]=(.65*output[mask]+.35*np.array([0,255,180])).astype(np.uint8)
        x1,y1,x2,y2=detection.bbox
        cv2.rectangle(output,(x1,y1),(x2,y2),(255,210,0),1)
    for obj in objects:
        pixels,z=world_to_pixel([obj.position],frame)
        if z[0]<=0: continue
        u,v=map(int,pixels[0]); cv2.circle(output,(u,v),4,(255,0,255),-1)
        caption=f"{obj.color} {obj.category} {np.round(obj.position,3)} {obj.confidence:.2f}"
        cv2.putText(output,caption,(max(0,u-40),max(15,v-9)),cv2.FONT_HERSHEY_SIMPLEX,.32,(255,255,255),1)
    annotations=annotations or {}
    for label,color in (("target",(255,220,20)),("grasp",(255,80,210))):
        if annotations.get(label) is not None:
            pixels,z=world_to_pixel([annotations[label]],frame)
            if z[0]>0:
                uv=tuple(map(int,pixels[0])); cv2.drawMarker(output,uv,color,cv2.MARKER_CROSS,18,2)
                cv2.putText(output,label,uv,cv2.FONT_HERSHEY_SIMPLEX,.5,color,1)
    path=annotations.get("trajectory",[])
    if len(path)>1:
        pixels,z=world_to_pixel(path,frame)
        for a,b,za,zb in zip(pixels,pixels[1:],z,z[1:]):
            if za>0 and zb>0: cv2.line(output,tuple(a.astype(int)),tuple(b.astype(int)),(20,220,255),2)
    return output


def save_observation(log,index,frame,detections,masks,objects,annotations=None):
    prefix=f"{index:05d}"
    cv2.imwrite(str(log.path/"images"/(prefix+"_rgb.png")),cv2.cvtColor(frame.rgb,cv2.COLOR_RGB2BGR))
    cv2.imwrite(str(log.path/"images"/(prefix+"_vision.png")),cv2.cvtColor(draw_observation(frame,detections,masks,objects,annotations),cv2.COLOR_RGB2BGR))
    np.savez_compressed(log.path/"images"/(prefix+"_depth.npz"),depth=frame.depth,intrinsics=frame.intrinsics,camera_to_world=frame.camera_to_world,timestamp=frame.timestamp)


def write_report(log):
    rows=[]
    images=sorted((log.path/"images").glob("*_vision.png"))+sorted((log.path/"images").glob("detail_*_candidates.jpg"))
    for image in images:
        rows.append(f'<figure><img loading="lazy" src="images/{image.name}"><figcaption>{image.stem}</figcaption></figure>')
    states=sorted((log.path/"world_states").glob("*.json"))
    world=states[-1].read_text() if states else "{}"
    events="\n".join(json.dumps(e,ensure_ascii=False,default=str) for e in log.events if e["type"] in {"plan","trajectory","grasp_candidates","recovery","final_validation","result"})
    page='<!doctype html><meta charset="utf-8"><title>Robot perception evidence</title><style>body{background:#102026;color:#e5eded;font:15px system-ui;margin:30px}main{display:flex;flex-wrap:wrap}figure{margin:8px}img{width:600px;max-width:90vw}pre{white-space:pre-wrap;max-height:600px;overflow:auto;background:#071317;padding:18px}h1{font-size:24px}</style>'
    page+=f'<h1>{html.escape(log.task_id)}</h1><p>RGB-D perception only · estimated positions · masks and centres</p><main>{"".join(rows)}</main><h2>World model</h2><pre>{html.escape(world)}</pre><h2>Plan / grasp / trajectory / validation</h2><pre>{html.escape(events)}</pre>'
    (log.path/"debug.html").write_text(page)
