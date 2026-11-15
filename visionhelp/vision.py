import os
import shutil
import time
from collections import defaultdict

import cv2
from huggingface_hub import hf_hub_download
from ultralytics import YOLO

from flicks import (
    OFF_LINE, ON_TARGET, OVERFLICK, TOO_HIGH, TOO_LOW, UNDERFLICK,
)

HF_REPO = "Vombit/yolo12n_cs2"
HF_FILE = "yolo12n_cs2.pt"

FONT = cv2.FONT_HERSHEY_SIMPLEX

# BGR colours
BODY_COLOR = (0, 200, 0)
HEAD_COLOR = (255, 200, 0)

FLICK_COLORS = {
    ON_TARGET: (0, 220, 0),
    OVERFLICK: (0, 0, 255),
    UNDERFLICK: (0, 165, 255),
    TOO_HIGH: (255, 0, 255),
    TOO_LOW: (255, 0, 255),
    OFF_LINE: (180, 180, 180),
}


def download_cs2_weights(target="models/yolo12n_cs2.pt"):
    """Download the CS2 weights from Hugging Face into `target`."""
    folder = os.path.dirname(target) or "."
    os.makedirs(folder, exist_ok=True)
    downloaded = hf_hub_download(
        repo_id=HF_REPO,
        filename=HF_FILE,
        local_dir=folder,
    )
    if os.path.abspath(downloaded) != os.path.abspath(target):
        shutil.copyfile(downloaded, target)
    return target


# ----------------------------------------------------------------- drawing

def _label(img, text, org, color, scale=0.55, thickness=1):
    """Text with a dark background so it stays readable on any frame."""
    (w, h), base = cv2.getTextSize(text, FONT, scale, thickness)
    x, y = int(org[0]), int(org[1])
    y = max(y, h + base + 2)
    cv2.rectangle(img, (x, y - h - base), (x + w + 4, y + base), (0, 0, 0), -1)
    cv2.putText(img, text, (x + 2, y), FONT, scale, color, thickness, cv2.LINE_AA)


def draw_detections(frame, detections, show_labels=True, head_names=()):
    out = frame.copy()

    for d in detections:
        x1, y1 = d["x"], d["y"]
        x2, y2 = x1 + d["width"], y1 + d["height"]

        color = HEAD_COLOR if d["name"] in head_names else BODY_COLOR
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)

        if show_labels:
            text = f'{d["name"]} {d["confidence"]:.2f}'
            if d.get("track_id") is not None:
                text += f' #{d["track_id"]}'
            _label(out, text, (x1, max(20, y1 - 8)), color)

    return out


def draw_flick(frame, flick):
    """Draw one finished flick on `frame` (in place)."""
    h, w = frame.shape[:2]
    cx, cy = w // 2, h // 2
    color = FLICK_COLORS.get(flick.label, (255, 255, 255))

    x1, y1, x2, y2 = (int(v) for v in flick.end_box)
    hx, hy = (x1 + x2) // 2, (y1 + y2) // 2

    # Path the head took on screen during the flick (shows the overshoot).
    pts = [(int(x), int(y)) for x, y in flick.path]
    for a, b in zip(pts, pts[1:]):
        cv2.line(frame, a, b, color, 2, cv2.LINE_AA)

    # Head box at the moment the flick ended + line from crosshair to it.
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    cv2.line(frame, (cx, cy), (hx, hy), color, 1, cv2.LINE_AA)
    cv2.drawMarker(frame, (cx, cy), color, cv2.MARKER_CROSS, 16, 2)

    text = f"{flick.label.upper()}  {flick.miss:.1f} heads"
    _label(frame, text, (x1, y1 - 8), color, scale=0.6, thickness=2)


def render_video(video_path, output_path, detections, flicks,
                 head_names=(), show_labels=True, hold_seconds=0.8,
                 on_progress=None):
    """Second pass: re-read the video and draw boxes + flick overlays."""
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    by_frame = defaultdict(list)
    for d in detections:
        by_frame[d["frame"]].append(d)

    # Show every flick result for `hold_seconds` after it ended.
    hold = max(1, int(hold_seconds * fps))
    active = defaultdict(list)
    for fl in flicks:
        for f in range(fl.end_frame, fl.end_frame + hold + 1):
            active[f].append(fl)

    writer = None
    i = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        frame = draw_detections(frame, by_frame.get(i, []), show_labels, head_names)
        for fl in active.get(i, []):
            draw_flick(frame, fl)

        if writer is None:
            height, width = frame.shape[:2]
            writer = cv2.VideoWriter(
                output_path,
                cv2.VideoWriter_fourcc(*"mp4v"),
                fps,
                (width, height),
            )
        writer.write(frame)

        i += 1
        if on_progress and i % 10 == 0:
            on_progress(min(i / max(total, 1), 1.0))

    cap.release()
    if writer:
        writer.release()
    if on_progress:
        on_progress(1.0)

    return output_path


# ------------------------------------------------------------------ engine

class VisionEngine:

    def __init__(self, model_path):
        self.model = YOLO(model_path)

    @property
    def class_names(self):
        return list(self.model.names.values())

    def reset_tracker(self):
        """Call before a new video so enemy IDs start fresh."""
        predictor = getattr(self.model, "predictor", None)
        trackers = getattr(predictor, "trackers", None)
        if not trackers:
            return
        for tracker in trackers:
            if hasattr(tracker, "reset"):
                tracker.reset()
            else:
                self.model.predictor = None
                break

    def detect(
        self,
        frame,
        confidence=0.5,
        image_size=640,
        track=False,
        show_labels=True,
        head_names=(),
        draw=True
    ):

        start = time.perf_counter()

        if track:
            # persist=True keeps enemy IDs stable from frame to frame
            results = self.model.track(
                frame,
                persist=True,
                tracker="bytetrack.yaml",
                conf=confidence,
                imgsz=image_size,
                verbose=False
            )
        else:
            results = self.model.predict(
                frame,
                conf=confidence,
                imgsz=image_size,
                verbose=False
            )

        boxes = results[0].boxes

        if boxes.id is not None:
            ids = boxes.id.int().tolist()
        else:
            ids = [None] * len(boxes)

        detections = []

        for box, track_id in zip(boxes, ids):

            x1, y1, x2, y2 = map(
                int,
                box.xyxy[0].tolist()
            )

            class_id = int(box.cls[0])

            detections.append({
                "name": self.model.names[class_id],
                "confidence": float(box.conf[0]),
                "track_id": track_id,
                "x": x1,
                "y": y1,
                "width": x2 - x1,
                "height": y2 - y1
            })

        if draw:
            annotated = draw_detections(
                frame,
                detections,
                show_labels,
                head_names
            )
        else:
            annotated = frame

        elapsed = time.perf_counter() - start
        fps = 1 / max(elapsed, 0.00001)

        return annotated, detections, fps