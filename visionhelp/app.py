
import streamlit as st
import cv2
import tempfile
import os
import pandas as pd

from vision import VisionEngine


st.set_page_config(
    page_title="CS2 Vision Lab",
    page_icon="🎯",
    layout="wide"
)

st.title("CS2 Vision Lab")
st.caption("Computer vision and video analysis")

st.sidebar.header("Model settings")

model_path = st.sidebar.text_input(
    "YOLO model path",
    "models/yolo11n.pt"
)

confidence = st.sidebar.slider(
    "Confidence",
    0.10,
    1.00,
    0.50,
    0.05
)

image_size = st.sidebar.select_slider(
    "Inference resolution",
    options=[320, 480, 640, 960],
    value=640
)

st.sidebar.divider()

show_labels = st.sidebar.checkbox(
    "Show detection labels",
    value=True
)

@st.cache_resource
def load_model(path):
    return VisionEngine(path)


if not os.path.exists(model_path):
    st.error(f"Model not found: {model_path}")
    st.info(
        "Place a compatible YOLO model in the models folder."
    )
    st.stop()

engine = load_model(model_path)

uploaded = st.file_uploader(
    "Upload a video",
    type=["mp4", "avi", "mov", "mkv"]
)

if uploaded:

    with tempfile.NamedTemporaryFile(
        delete=False,
        suffix=".mp4"
    ) as temp:
        temp.write(uploaded.read())
        video_path = temp.name

    cap = cv2.VideoCapture(video_path)

    total_frames = int(
        cap.get(cv2.CAP_PROP_FRAME_COUNT)
    )

    original_fps = cap.get(
        cv2.CAP_PROP_FPS
    )

    st.subheader("Video analysis")

    video_display = st.empty()

    progress = st.progress(0)

    stats = st.columns(3)

    frame_number = 0
    all_detections = []
    output_path = None
    writer = None

    if st.button("Start analysis", type="primary"):

        output_path = tempfile.NamedTemporaryFile(
            delete=False,
            suffix=".mp4"
        ).name

        while cap.isOpened():

            success, frame = cap.read()

            if not success:
                break

            annotated, detections, processing_fps = (
                engine.detect(
                    frame,
                    confidence,
                    image_size
                )
            )

            if not show_labels:
                # Draw boxes without text
                annotated = frame.copy()

                for d in detections:
                    cv2.rectangle(
                        annotated,
                        (d["x"], d["y"]),
                        (
                            d["x"] + d["width"],
                            d["y"] + d["height"]
                        ),
                        (0, 255, 0),
                        2
                    )

            if writer is None:
                height, width = annotated.shape[:2]

                writer = cv2.VideoWriter(
                    output_path,
                    cv2.VideoWriter_fourcc(*"mp4v"),
                    original_fps if original_fps > 0 else 30,
                    (width, height)
                )

            writer.write(annotated)

            for d in detections:
                all_detections.append({
                    "frame": frame_number,
                    **d
                })

            frame_number += 1

            video_display.image(
                cv2.cvtColor(
                    annotated,
                    cv2.COLOR_BGR2RGB
                ),
                use_container_width=True
            )

            progress.progress(
                min(frame_number / max(total_frames, 1), 1.0)
            )

            stats[0].metric(
                "Frames processed",
                frame_number
            )

            stats[1].metric(
                "Objects detected",
                len(all_detections)
            )

            stats[2].metric(
                "Processing FPS",
                f"{processing_fps:.1f}"
            )

        cap.release()

        if writer:
            writer.release()

        st.success("Video analysis completed!")

        if os.path.exists(output_path):

            with open(output_path, "rb") as file:
                st.download_button(
                    "Download annotated video",
                    file,
                    "cs2_analysis.mp4",
                    "video/mp4"
                )

        if all_detections:

            df = pd.DataFrame(all_detections)

            st.subheader("Detection results")

            st.dataframe(
                df,
                use_container_width=True
            )

            st.download_button(
                "Download detection CSV",
                df.to_csv(index=False),
                "detections.csv",
                "text/csv"
            )

        else:
            st.warning("No objects detected.")

    cap.release()

    if os.path.exists(video_path):
        os.remove(video_path)

else:
    st.info("Upload a video to begin.")
