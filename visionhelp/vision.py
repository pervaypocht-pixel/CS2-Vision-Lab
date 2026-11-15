from ultralytics import YOLO
import cv2
import time
from huggingface_hub import hf_hub_download
from ultralytics import YOLO

weights = hf_hub_download(
    repo_id="Vombit/yolo12n_cs2",
    filename="yolo12n_cs2.pt"
)
model = YOLO(weights)
source = 'http://images.cocodataset.org/val2017/000000039769.jpg'
model.predict(source=source, save=True)


class VisionEngine:

    def __init__(self, model_path):
        self.model = YOLO(model_path)

    def detect(
        self,
        frame,
        confidence=0.5,
        image_size=640
    ):

        start = time.perf_counter()

        results = self.model.predict(
            frame,
            conf=confidence,
            imgsz=image_size,
            verbose=False
        )

        result = results[0]
        annotated = frame.copy()

        detections = []

        for box in result.boxes:

            x1, y1, x2, y2 = map(
                int,
                box.xyxy[0].tolist()
            )

            class_id = int(box.cls[0])
            score = float(box.conf[0])

            name = self.model.names[class_id]

            detections.append({
                "name": name,
                "confidence": score,
                "x": x1,
                "y": y1,
                "width": x2 - x1,
                "height": y2 - y1
            })

            cv2.rectangle(
                annotated,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                2
            )

            cv2.putText(
                annotated,
                f"{name} {score:.2f}",
                (x1, max(20, y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2
            )

        elapsed = time.perf_counter() - start

        fps = 1 / max(elapsed, 0.00001)

        return annotated, detections, fps
