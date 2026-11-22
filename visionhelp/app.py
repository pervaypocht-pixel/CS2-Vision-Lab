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
from vision import VisionEngine, class_label, download_cs2_weights, render_video


# ------------------------------------------------------ video requirements
# Flick analysis measures how fast a head moves across the screen, so the video
# needs enough pixels to see heads and enough frames to catch a ~100 ms flick.

MIN_WIDTH, MIN_HEIGHT = 1280, 720    # below this, distant heads are missed
REC_WIDTH, REC_HEIGHT = 1920, 1080   # recommended
MIN_FPS = 60                         # a flick is only 4-9 frames long at 60 FPS
FPS_TOLERANCE = 1.0                  # so that 59.94 FPS counts as 60
MAX_DURATION_S = 60                  # keeps analysis time and memory reasonable
LONG_DURATION_S = 30                 # warn above this


def read_video_info(path):
    cap = cv2.VideoCapture(path)
    info = None

    if cap.isOpened():
        fps = cap.get(cv2.CAP_PROP_FPS) or 0
        frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        info = {
            "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            "fps": fps,
            "frames": frames,
            "duration": frames / fps if fps > 0 else 0,
        }

    cap.release()
    return info


def check_video(info):
    """Returns (errors, warnings). Errors block the analysis, warnings do not."""
    if not info or info["fps"] <= 0 or info["frames"] <= 0:
        return [
            "Could not read this video. Try exporting it again as MP4 (H.264)."
        ], []

    errors, warnings = [], []
    w, h = info["width"], info["height"]

    if w < MIN_WIDTH or h < MIN_HEIGHT:
        errors.append(
            f"Resolution is too low ({w}×{h}). The minimum is "
            f"{MIN_WIDTH}×{MIN_HEIGHT}; enemy heads are too small to "
            "detect reliably below that."
        )
    elif w < REC_WIDTH or h < REC_HEIGHT:
        warnings.append(
            f"Resolution is {w}×{h}. It works, but far-away heads may be "
            f"missed. {REC_WIDTH}×{REC_HEIGHT} is recommended."
        )

    if info["fps"] < MIN_FPS - FPS_TOLERANCE:
        errors.append(
            f"Frame rate is too low ({info['fps']:.0f} FPS). Use at least "
            f"{MIN_FPS} FPS: a flick lasts only about a tenth of a second, "
            "so lower frame rates cannot capture where it ended."
        )

    if info["duration"] > MAX_DURATION_S:
        errors.append(
            f"Video is too long ({info['duration']:.0f} s). The maximum is "
            f"{MAX_DURATION_S} seconds. Trim it to the rounds you want to "
            "analyze."
        )
    elif info["duration"] > LONG_DURATION_S:
        warnings.append(
            f"Video is {info['duration']:.0f} s long. Analysis may take "
            f"several minutes. Clips under {LONG_DURATION_S} s are fastest."
        )

    return errors, warnings


st.set_page_config(
    page_title="CS2 Vision Lab",
    page_icon="🎯",
    layout="wide"
)

st.title("CS2 Vision Lab")
st.caption("AI aim analysis for your Counter-Strike 2 gameplay videos")

st.markdown(
    "Upload a clip of your own CS2 gameplay and CS2 Vision Lab shows you "
    "**where your crosshair stops when you flick to an enemy**: right on the "
    "head, too far past it, too short, or too high or low."
)

tab_about, tab_use, tab_results = st.tabs(
    ["What is this?", "How to use it", "Reading the results"]
)

with tab_about:
    st.markdown(
        "**CS2 Vision Lab** is an aim analysis tool for Counter-Strike 2. "
        "An AI model (YOLO) watches your recorded gameplay, finds every "
        "enemy and enemy head in each frame, and follows them through the "
        "video. The app then measures how your crosshair moves toward each "
        "head during a flick and checks where the flick ended.\n\n"
        "**What it is for**\n"
        "- Seeing your aim mistakes objectively instead of guessing\n"
        "- Finding your habits: do you usually fly past the head "
        "(overflick) or stop short of it (underflick)? Are you often too "
        "high or too low?\n"
        "- Checking whether your aim improves over time by comparing clips\n\n"
        "**What it is not**\n"
        "- It does not run during the game and it never touches "
        "Counter-Strike. It only analyzes video files that you upload after "
        "you have recorded them."
    )
    st.caption(
        "Detection model: yolo12n_cs2 by Vombit (Hugging Face). "
        "This project is not affiliated with Valve or Counter-Strike."
    )

with tab_use:
    st.markdown(
        "1. **Record** your gameplay in first person at 60 FPS or higher "
        "(1920×1080 is best). The exact requirements are listed further "
        "down this page.\n"
        "2. **Pick the enemy team** in the sidebar under *Flick analysis*, "
        "in *Enemy heads to analyze*. If you play Counter-Terrorist, select "
        "only *Terrorist head*. The other settings have good defaults; "
        "hover over the **?** next to any setting to see what it does.\n"
        "3. **Upload** your video. It is checked right away, and you are "
        "told if something does not meet the requirements.\n"
        "4. Click **Start analysis** and wait for the progress bar to "
        "finish. The results appear automatically.\n"
        "5. **Read the results** (see the next tab).\n"
        "6. Click **Render video with flick overlay** to get a video with "
        "every flick marked on screen, then download it.\n"
        "7. If too many or too few flicks are found, change **Flick speed "
        "threshold** in the sidebar. The results update instantly, without "
        "analyzing the video again."
    )

with tab_results:
    st.markdown(
        "**Result of each flick**\n"
        "- **On target:** the crosshair ended inside the head box.\n"
        "- **Overflick:** the crosshair moved past the head.\n"
        "- **Underflick:** the crosshair stopped before reaching the head.\n"
        "- **High / Low:** the crosshair ended above / below the head "
        "(a vertical miss).\n"
        "- **Off-line:** the crosshair missed sideways, off the path of the "
        "flick.\n\n"
        "**Numbers and charts**\n"
        "- Distances are measured in *heads*: one head is the width of the "
        "enemy's head on screen, so 1.0 means you ended one head width away "
        "from its center. This makes results comparable at any distance.\n"
        "- In the chart, each dot is where your crosshair ended a flick, "
        "relative to the enemy head (the square in the middle). Dots "
        "gathered on one side of the square show a consistent habit.\n"
        "- The line under the metrics tells you if you tend to stop short "
        "or go past the target on average.\n\n"
        "**Good to know**\n"
        "- One flick says little. Look for patterns across 20 or more "
        "flicks, ideally from several clips.\n"
        "- It measures where a flick ends, not the moment you click.\n"
        "- A moving enemy adds some error that is not your aim.\n"
        "- Far away or partly hidden heads can be missed. If a result looks "
        "wrong, check it in the rendered video."
    )

# ------------------------------------------------------------------ sidebar

st.sidebar.header("Model settings")
st.sidebar.caption("Control how players and heads are found in the video.")

model_path = st.sidebar.text_input(
    "YOLO model path",
    "models/yolo12n_cs2.pt",
    help="Location of the detection model file. The default is the CS2 "
         "player detector, which finds Counter-Terrorists, Terrorists and "
         "their heads. Only change this if you want to use a different model."
)

confidence = st.sidebar.slider(
    "Detection confidence",
    0.10,
    1.00,
    0.50,
    0.05,
    help="How sure the model must be before it counts something as a player "
         "or head. Higher = fewer false detections, but far away or partly "
         "hidden enemies may be missed. Lower = finds more enemies, but may "
         "box things that are not players. 0.50 is a good starting point."
)

image_size = st.sidebar.select_slider(
    "Detection resolution (px)",
    options=[320, 480, 640, 960],
    value=640,
    help="The size frames are shrunk to before detection. Bigger (960) "
         "spots small, far away heads better but analysis is slower. "
         "Smaller (320) is fast but misses distant enemies. 640 is a good "
         "balance."
)

st.sidebar.divider()

show_labels = st.sidebar.checkbox(
    "Show detection labels",
    value=True,
    help="Writes the class, confidence and enemy number on every box in the "
         "rendered video. Turn it off for a cleaner picture with only the "
         "boxes. Only affects the rendered video."
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

# Flick settings live after the model loads, because the head class list
# comes from the model itself (for this model: c, ch, t, th).
class_names = engine.class_names
default_heads = [
    n for n in class_names
    if n.lower().endswith("h") or "head" in n.lower()
]
head_options = default_heads or class_names

st.sidebar.divider()
st.sidebar.header("Flick analysis")
st.sidebar.caption(
    "A flick is a fast aim movement toward an enemy head. "
    "These settings decide what counts as a flick and how it is shown."
)

head_names = st.sidebar.multiselect(
    "Enemy heads to analyze",
    head_options,
    default=default_heads,
    format_func=class_label,
    help="Choose the team you are playing against. If you play as "
         "Counter-Terrorist, select only 'Terrorist head'. Selecting both "
         "also analyzes your teammates' heads, which gives wrong results."
)

speed_thr = st.sidebar.slider(
    "Flick speed threshold (heads/sec)",
    4.0,
    60.0,
    12.0,
    1.0,
    help="How fast a head must move across the screen to count as a flick. "
         "Distance is measured in head widths ('heads'), so it works the "
         "same at any range. Lower = slower, smaller movements also count "
         "as flicks. Higher = only fast flicks count. If too many or too "
         "few flicks are found, change this one first."
)

min_start = st.sidebar.slider(
    "Minimum flick distance (heads)",
    0.5,
    5.0,
    1.0,
    0.25,
    help="Ignore aim movements that start closer to the head than this "
         "(in head widths). Raise it to skip tiny adjustments and keep only "
         "real flicks. Lower it to include small corrections."
)

hold_seconds = st.sidebar.slider(
    "Overlay duration (sec)",
    0.3,
    2.0,
    0.8,
    0.1,
    help="How long each flick result stays on screen in the rendered video "
         "after the flick ends. Only affects the video, not the numbers."
)

# ------------------------------------------------------------------- upload

st.info(
    "**Video requirements for accurate analysis**\n\n"
    f"- **Resolution:** at least {MIN_WIDTH}×{MIN_HEIGHT} "
    f"({REC_WIDTH}×{REC_HEIGHT} recommended)\n"
    f"- **Frame rate:** {MIN_FPS} FPS or higher\n"
    f"- **Length:** up to {MAX_DURATION_S} seconds "
    f"(under {LONG_DURATION_S} s is fastest)\n"
    "- **Content:** your own first-person view with the crosshair in the "
    "exact center of the screen\n"
    "- **No editing:** no black bars, cropping, zoom, slow motion or speed-up"
)

with st.expander("Why these requirements?"):
    st.markdown(
        "- **Resolution:** enemy heads are only a few pixels wide at a "
        "distance. Low resolution makes them undetectable or the boxes too "
        "imprecise to judge a miss.\n"
        "- **Frame rate:** a flick lasts about 60-150 ms. At 60 FPS that is "
        "4-9 frames, enough to see where it ended. At 30 FPS it is only "
        "2-4 frames.\n"
        "- **Length:** every frame is analyzed, so long videos take a long "
        "time and a lot of memory. Short clips of the fights you want to "
        "review work best.\n"
        "- **Crosshair in the center:** the analysis assumes the crosshair is "
        "at the exact center of the picture. Black bars, cropping or a "
        "different screen layout shift that center and make every result "
        "wrong.\n"
        "- **No slow motion or speed-up:** flick speed is measured in real "
        "time, so changing the video speed changes the results."
    )

uploaded = st.file_uploader(
    "Upload a video",
    type=["mp4", "avi", "mov", "mkv"],
    help="Supported formats: MP4, AVI, MOV, MKV (MP4 with H.264 is the "
         "safest). The maximum file size is 200 MB. Your video must meet "
         "the requirements above."
)

if uploaded:

    # Save the video once per upload (Streamlit reruns the script on every click).
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
        st.session_state.video_info = read_video_info(temp.name)
        st.session_state.detections = None
        st.session_state.meta = None
        st.session_state.rendered = None

    video_path = st.session_state.video_path

    # ------------------------------------------------- check requirements

    info = st.session_state.video_info
    errors, warnings = check_video(info)

    if info and info["fps"] > 0 and info["frames"] > 0:
        st.caption(
            f"{info['width']}×{info['height']} · "
            f"{info['fps']:.0f} FPS · {info['duration']:.1f} s"
        )

    for message in errors:
        st.error(message)

    for message in warnings:
        st.warning(message)

    if errors:
        st.stop()

    # ------------------------------------------------- pass 1: detect + track

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

    # --------------------------------------------------- pass 2: flick stats
    # Runs on stored detections, so moving the sliders is instant (no YOLO rerun).

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

            # ------------------------------------------ pass 3: render video

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
                raw["name"] = raw["name"].map(class_label)
                raw = raw.rename(columns={"name": "class"})

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