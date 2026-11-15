

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter

ON_TARGET = "on target"
OVERFLICK = "overflick"
UNDERFLICK = "underflick"
TOO_HIGH = "high"        
TOO_LOW = "low"          
OFF_LINE = "off-line"    


@dataclass
class Flick:
    track_id: int
    start_frame: int
    end_frame: int
    label: str
    start_err: tuple       
    end_err: tuple        
    along: float          
    cross: float          
    miss: float           
    start_dist: float     
    duration_ms: float
    end_box: tuple        
    path: list = field(default_factory=list)  


def _smooth(arr, window=5, order=2):
    n = len(arr)
    w = min(window, n if n % 2 == 1 else n - 1)
    if w <= order:
        return arr
    return savgol_filter(arr, w, order, axis=0, mode="interp")


def _split_runs(frames, max_gap):
    runs, start = [], 0
    for i in range(1, len(frames)):
        if frames[i] - frames[i - 1] > max_gap + 1:
            runs.append((start, i))
            start = i
    runs.append((start, len(frames)))
    return runs


def _analyze_run(track_id, frames, boxes, width, height, fps,
                 speed_thr, min_start, min_flick_ms):
    full = np.arange(frames[0], frames[-1] + 1)

    x1 = np.interp(full, frames, boxes[:, 0])
    y1 = np.interp(full, frames, boxes[:, 1])
    x2 = np.interp(full, frames, boxes[:, 2])
    y2 = np.interp(full, frames, boxes[:, 3])

    hx, hy = (x1 + x2) / 2, (y1 + y2) / 2
    hw = np.maximum(x2 - x1, 1.0)
    hh = np.maximum(y2 - y1, 1.0)

    sm = _smooth(np.stack([hx, hy, hw, hh], axis=1))
    hx, hy, hw, hh = sm[:, 0], sm[:, 1], np.maximum(sm[:, 2], 1.0), np.maximum(sm[:, 3], 1.0)

    cx, cy = width / 2, height / 2
    e = np.stack([(hx - cx) / hw, (hy - cy) / hw], axis=1)   

    n = len(e)
    speed = np.linalg.norm(np.gradient(e, axis=0), axis=1) * fps   
    moving = speed > speed_thr

    
    for i in range(1, n - 1):
        if not moving[i] and moving[i - 1] and moving[i + 1]:
            moving[i] = True

    min_frames = max(2, int(np.ceil(min_flick_ms / 1000 * fps)))
    settle = max(1, int(round(0.04 * fps)))   

    flicks = []
    i = 0
    while i < n:
        if not moving[i]:
            i += 1
            continue
        j = i
        while j < n and moving[j]:
            j += 1
        
        if (j - i) >= min_frames and j < n:
            s = max(i - 1, 0)
            k = min(j + settle, n - 1)
            e0, e1 = e[s], e[k]
            d0 = float(np.linalg.norm(e0))

            if d0 >= min_start:
                u = e0 / d0
                along = float(e1 @ u)
                cross = float(np.linalg.norm(e1 - along * u))
                aspect = float(hh[k] / hw[k])

                inside = abs(e1[0]) <= 0.5 and abs(e1[1]) <= 0.5 * aspect
                if inside:
                    label = ON_TARGET
                elif abs(along) >= cross:
                    label = UNDERFLICK if along > 0 else OVERFLICK
                elif abs(e1[1]) >= abs(e1[0]):
                    
                    label = TOO_HIGH if e1[1] > 0 else TOO_LOW
                else:
                    label = OFF_LINE

                flicks.append(Flick(
                    track_id=int(track_id),
                    start_frame=int(full[s]),
                    end_frame=int(full[k]),
                    label=label,
                    start_err=(float(e0[0]), float(e0[1])),
                    end_err=(float(e1[0]), float(e1[1])),
                    along=along,
                    cross=cross,
                    miss=float(np.linalg.norm(e1)),
                    start_dist=d0,
                    duration_ms=(j - s) / fps * 1000,
                    end_box=(float(hx[k] - hw[k] / 2), float(hy[k] - hh[k] / 2),
                             float(hx[k] + hw[k] / 2), float(hy[k] + hh[k] / 2)),
                    path=[(float(hx[t]), float(hy[t])) for t in range(s, k + 1)],
                ))
        i = j

    return flicks


def analyze_flicks(detections, head_names, width, height, fps,
                   speed_thr=12.0, min_start=1.0, max_gap=5,
                   min_track_frames=6, min_flick_ms=25):
    
    by_track = {}
    for d in detections:
        if d["name"] in head_names and d.get("track_id") is not None:
            by_track.setdefault(d["track_id"], {})[d["frame"]] = (
                d["x"], d["y"], d["x"] + d["width"], d["y"] + d["height"]
            )

    flicks = []
    for track_id, per_frame in by_track.items():
        frames = np.array(sorted(per_frame), dtype=int)
        boxes = np.array([per_frame[f] for f in frames], dtype=float)

        for a, b in _split_runs(frames, max_gap):
            if b - a < min_track_frames:
                continue
            flicks += _analyze_run(track_id, frames[a:b], boxes[a:b],
                                   width, height, fps,
                                   speed_thr, min_start, min_flick_ms)

    flicks.sort(key=lambda f: f.end_frame)
    return flicks


def flicks_to_dataframe(flicks, fps):
    rows = []
    for f in flicks:
        rows.append({
            "time (s)": round(f.end_frame / fps, 2),
            "enemy id": f.track_id,
            "result": f.label,
            "miss (heads)": round(f.miss, 2),
            "along (heads)": round(f.along, 2),     
            "cross (heads)": round(f.cross, 2),
            "flick (ms)": round(f.duration_ms),
            "start dist (heads)": round(f.start_dist, 1),
            "land_x": -f.end_err[0],
            "land_y": f.end_err[1],
        })
    return pd.DataFrame(rows)