import os
import tempfile

import cv2
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from flicks import (
    OFF_LINE, ON_TARGET, OVERFLICK, TOO_HIGH, TOO_LOW, UNDERFLICK,
    analyze_flicks, flicks_to_dataframe,
)
from vision import VisionEngine, download_cs2_weights, render_video


st.set_page_config(
    page_title="CS2 Vision Lab",
    page_icon="🎯",
    layout="wide"
)

st.title("CS2 Vision Lab")
st.caption("Computer vision and video analysis")

# ------------------------------------------------------------------ sidebar

st.sidebar.header("Model settings")

model_path = st.sidebar.text_input(
    "YOLO model path",
    "models/yolo12n_cs2.pt"
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
        "Place a compatible YOLO model in the models folder, "
        "or download the CS2 model below."
    )
    if st.button("Download yolo12n_cs2 from Hugging Face"):
        try:
            with st.spinner("Downloading..."):
                download_cs2_weights(model_path)
            st.rerun()
        except Exception as error:
            st.error(f"Download failed: {error}")
    st.stop()

engine = load_model(model_path)
class_names = engine.class_names
default_heads = [
    n for n in class_names
    if n.lower().endswith("h") or "head" in n.lower()
]

st.sidebar.divider()
st.sidebar.header("Flick analysis")

head_names = st.sidebar.multiselect(
    "Enemy head classes",
    class_names,
    default=default_heads,
    help="Pick only the team you are playing against, "
         "e.g. only 'th' if you play CT."
)

speed_thr = st.sidebar.slider(
    "Flick speed threshold (heads/sec)",
    4.0,
    60.0,
    12.0,
    1.0,
    help="Lower = more small movements count as flicks."
)

min_start = st.sidebar.slider(
    "Minimum flick distance (heads)",
    0.5,
    5.0,
    1.0,
    0.25,
    help="Ignore movements that start closer to the head than this."
)

hold_seconds = st.sidebar.slider(
    "Overlay duration (sec)",
    0.3,
    2.0,
    0.8,
    0.1
)

# ------------------------------------------------------------------- upload

uploaded = st.file_uploader(
    "Upload a video",
    type=["mp4", "avi", "mov", "mkv"]
)

if uploaded:

    file_key = (uploaded.name, uploaded.size)

    if st.session_state.get("file_key") != file_key:
        old_path = st.session_state.get("video_path")
        if old_path and os.path.exists(old_path):
            os.remove(old_path)

        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=".mp4"
        ) as temp:
            temp.write(uploaded.getbuffer())

        st.session_state.file_key = file_key
        st.session_state.video_path = temp.name
        st.session_state.detections = None
        st.session_state.meta = None
        st.session_state.rendered = None

    video_path = st.session_state.video_path

    st.subheader("Video analysis")

    if st.button("Start analysis", type="primary"):

        cap = cv2.VideoCapture(video_path)

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        original_fps = cap.get(cv2.CAP_PROP_FPS) or 30
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        progress = st.progress(0, text="Analyzing video...")

        engine.reset_tracker()

        frame_number = 0
        all_detections = []

        while cap.isOpened():

            success, frame = cap.read()

            if not success:
                break

            _, detections, _ = engine.detect(
                frame,
                confidence,
                image_size,
                track=True,
                draw=False
            )

            for d in detections:
                all_detections.append({
                    "frame": frame_number,
                    **d
                })

            frame_number += 1

            if frame_number % 5 == 0 or frame_number == total_frames:
                progress.progress(
                    min(frame_number / max(total_frames, 1), 1.0),
                    text=f"Analyzing video... {frame_number} / {total_frames} frames"
                )

        cap.release()
        progress.empty()

        st.session_state.detections = all_detections
        st.session_state.meta = {
            "width": width,
            "height": height,
            "fps": original_fps,
        }
        st.session_state.rendered = None

        st.success("Video analysis completed!")

    detections = st.session_state.get("detections")
    meta = st.session_state.get("meta")

    if detections is not None and meta:

        if not detections:
            st.warning("No objects detected.")

        st.subheader("Flick analysis")

        if not head_names:
            st.warning("Select at least one head class in the sidebar.")
            flicks = []
        else:
            flicks = analyze_flicks(
                detections,
                set(head_names),
                meta["width"],
                meta["height"],
                meta["fps"],
                speed_thr=speed_thr,
                min_start=min_start
            )

        if head_names and not flicks:
            st.info(
                "No flicks found. Try lowering the speed threshold or the "
                "minimum flick distance, or the detection confidence."
            )

        if flicks:

            df = flicks_to_dataframe(flicks, meta["fps"])
            counts = df["result"].value_counts()
            total = len(df)

            cols = st.columns(5)
            cols[0].metric("Flicks", total)
            cols[1].metric(
                "On target",
                f"{counts.get(ON_TARGET, 0) / total:.0%}"
            )
            cols[2].metric("Overflicks", counts.get(OVERFLICK, 0))
            cols[3].metric("Underflicks", counts.get(UNDERFLICK, 0))
            cols[4].metric(
                "High / low",
                counts.get(TOO_HIGH, 0) + counts.get(TOO_LOW, 0)
            )

            bias = df["along (heads)"].mean()
            if bias > 0.1:
                st.caption(
                    f"On average you stop {bias:.2f} heads short of the "
                    "target (underflick tendency)."
                )
            elif bias < -0.1:
                st.caption(
                    f"On average you go {-bias:.2f} heads past the "
                    "target (overflick tendency)."
                )
            else:
                st.caption("No clear over/underflick tendency.")

            left, right = st.columns([1, 2])

            with left:
                fig, ax = plt.subplots(figsize=(4.5, 4.5))

                ax.add_patch(
                    plt.Rectangle((-0.5, -0.5), 1, 1, fill=False, linewidth=2)
                )
                ax.axhline(0, color="gray", linewidth=0.5)
                ax.axvline(0, color="gray", linewidth=0.5)

                for label, group in df.groupby("result"):
                    ax.scatter(
                        group["land_x"],
                        group["land_y"],
                        label=label,
                        s=45
                    )

                ax.set_aspect("equal")
                ax.set_xlabel("crosshair left / right of head (heads)")
                ax.set_ylabel("crosshair below / above head (heads)")
                ax.set_title("Where your crosshair stopped")
                ax.legend(fontsize=8)

                st.pyplot(fig)
                plt.close(fig)

                st.caption(
                    "Each dot is where the crosshair ended a flick, "
                    "relative to the enemy head (the square)."
                )

            with right:
                st.dataframe(
                    df.drop(columns=["land_x", "land_y"]),
                    use_container_width=True
                )


            if st.button("Render video with flick overlay"):

                output_path = tempfile.NamedTemporaryFile(
                    delete=False,
                    suffix=".mp4"
                ).name

                bar = st.progress(0, text="Rendering video...")

                render_video(
                    video_path,
                    output_path,
                    detections,
                    flicks,
                    head_names=set(head_names),
                    show_labels=show_labels,
                    hold_seconds=hold_seconds,
                    on_progress=lambda p: bar.progress(p, text="Rendering video...")
                )

                with open(output_path, "rb") as file:
                    st.session_state.rendered = file.read()

                os.remove(output_path)
                bar.empty()

            if st.session_state.get("rendered"):
                st.download_button(
                    "Download video with flick overlay",
                    st.session_state.rendered,
                    "cs2_flicks.mp4",
                    "video/mp4"
                )

        if detections:
            with st.expander("Raw detections"):
                raw = pd.DataFrame(detections)

                st.dataframe(
                    raw,
                    use_container_width=True
                )

                st.download_button(
                    "Download detection CSV",
                    raw.to_csv(index=False),
                    "detections.csv",
                    "text/csv"
                )

else:
    st.info("Upload a video to begin.")