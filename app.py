from __future__ import annotations

import sys
import tempfile
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd
import streamlit as st
from streamlit.runtime.scriptrunner import get_script_run_ctx

import config
from backend import IMAGE_EXTS, Annotation, PromptSpec, SAM3TextBackend


def inject_css():
    st.markdown(
        """
        <style>
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        header {visibility: hidden;}
        div[data-testid="stToolbar"] {display: none;}
        div[data-testid="stDecoration"] {display: none;}
        [data-testid="collapsedControl"] {display: none;}
        section[data-testid="stSidebar"] {display: none;}
        .block-container {
            padding-top: 1.2rem;
            padding-bottom: 2rem;
            max-width: 1320px;
        }
        .hero-wrap {
            display: flex;
            align-items: center;
            gap: 1rem;
            padding: 1rem 1.2rem;
            border-radius: 20px;
            background: linear-gradient(135deg, #eef6ff 0%, #fffdf6 100%);
            border: 1px solid #dbe5f0;
            margin-bottom: 1rem;
        }
        .hero-title {
            margin: 0;
            font-size: 2rem;
            line-height: 1.05;
            color: #12324a;
        }
        .hero-subtitle {
            margin: 0.28rem 0 0 0;
            color: #496276;
            font-size: 0.98rem;
        }
        .panel-card {
            padding: 1rem 1.05rem;
            border: 1px solid #dbe5f0;
            border-radius: 18px;
            background: #fbfdff;
        }
        .count-chip {
            display: inline-block;
            margin: 0.18rem 0.35rem 0.18rem 0;
            padding: 0.35rem 0.7rem;
            border-radius: 999px;
            background: #12324a;
            color: white;
            font-size: 0.9rem;
        }
        .verifygn-overlay {
            position: fixed;
            inset: 0;
            background: rgba(18, 50, 74, 0.28);
            display: flex;
            align-items: center;
            justify-content: center;
            z-index: 999999;
            padding: 1.2rem;
        }
        .verifygn-popup {
            width: min(560px, 92vw);
            background: white;
            border-radius: 22px;
            border: 1px solid #dbe5f0;
            box-shadow: 0 24px 70px rgba(18, 50, 74, 0.22);
            padding: 1.3rem 1.35rem;
            text-align: center;
        }
        .verifygn-spinner {
            width: 54px;
            height: 54px;
            margin: 0 auto 0.9rem auto;
            border-radius: 999px;
            border: 5px solid #dbe5f0;
            border-top-color: #12324a;
            animation: verifygn-spin 0.9s linear infinite;
        }
        .verifygn-popup h3 {
            margin: 0 0 0.55rem 0;
            color: #12324a;
            font-size: 1.15rem;
        }
        .verifygn-popup p {
            margin: 0;
            color: #3e5a6e;
            font-size: 1rem;
            line-height: 1.55;
        }
        @keyframes verifygn-spin {
            from { transform: rotate(0deg); }
            to { transform: rotate(360deg); }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_header():
    st.markdown('<div class="hero-wrap">', unsafe_allow_html=True)
    cols = st.columns([1.1, 7.9])
    with cols[0]:
        if config.LOGO_PATH.exists():
            st.image(str(config.LOGO_PATH), width=104)
    with cols[1]:
        st.markdown(f"<h1 class='hero-title'>{config.APP_TITLE}</h1>", unsafe_allow_html=True)
        st.markdown(f"<p class='hero-subtitle'>{config.APP_SUBTITLE}</p>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)


def is_running_in_streamlit() -> bool:
    try:
        return get_script_run_ctx() is not None
    except Exception:
        return False


def launch_with_streamlit_cli():
    from streamlit.web import cli as stcli

    sys.argv = ["streamlit", "run", str(Path(__file__).resolve())]
    raise SystemExit(stcli.main())


def build_prompt_specs() -> List[PromptSpec]:
    specs: List[PromptSpec] = []
    for prompt, label in config.PROMPT_CLASS_MAP:
        prompt_text = str(prompt).strip()
        label_text = str(label).strip() or prompt_text
        if prompt_text:
            specs.append(PromptSpec(prompt=prompt_text, label=label_text))
    return specs


@st.cache_resource(show_spinner=False)
def get_backend(model_path: str, conf: float, iou: float, half: bool, imgsz: int) -> SAM3TextBackend:
    backend = SAM3TextBackend(model_path=model_path, conf=conf, iou=iou, half=half, imgsz=imgsz)
    backend.load()
    return backend


def color_for_label(label: str) -> Tuple[int, int, int]:
    base = abs(hash(label))
    return (
        40 + (base % 170),
        50 + ((base // 170) % 160),
        60 + ((base // (170 * 160)) % 150),
    )


def decode_uploaded_image(uploaded_file) -> Optional[np.ndarray]:
    if uploaded_file is None:
        return None
    data = np.frombuffer(uploaded_file.getvalue(), dtype=np.uint8)
    if data.size == 0:
        return None
    image_bgr = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if image_bgr is None:
        return None
    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)


def save_rgb_to_temp(image_rgb: np.ndarray, suffix: str = ".png") -> Path:
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
        cv2.imwrite(tmp.name, image_bgr)
        return Path(tmp.name)


def save_uploaded_to_temp(uploaded_file) -> Path:
    suffix = Path(getattr(uploaded_file, "name", "image.png")).suffix or ".png"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded_file.getbuffer())
        return Path(tmp.name)


def capture_system_webcam_frame() -> Optional[np.ndarray]:
    backend = None
    if sys.platform.startswith("win"):
        backend = cv2.CAP_DSHOW

    cap = cv2.VideoCapture(config.SYSTEM_CAMERA_INDEX, backend) if backend is not None else cv2.VideoCapture(config.SYSTEM_CAMERA_INDEX)
    if not cap or not cap.isOpened():
        if cap:
            cap.release()
        cap = cv2.VideoCapture(config.SYSTEM_CAMERA_INDEX)
        if not cap or not cap.isOpened():
            return None

    frame = None
    try:
        for _ in range(max(1, int(config.SYSTEM_CAMERA_WARMUP_FRAMES))):
            ok, current = cap.read()
            if ok and current is not None:
                frame = current
    finally:
        cap.release()

    if frame is None:
        return None
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


def draw_detections(image_path: str, anns: List[Annotation]) -> np.ndarray:
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")

    rendered = image.copy()
    box_thickness = max(2, int(config.BBOX_THICKNESS))
    font_scale = max(0.5, float(config.LABEL_FONT_SCALE))
    font_thickness = max(1, int(config.LABEL_FONT_THICKNESS))
    pad_x = max(6, int(config.LABEL_BOX_PADDING_X))
    pad_y = max(4, int(config.LABEL_BOX_PADDING_Y))
    for ann in anns:
        x1, y1, x2, y2 = ann.bbox
        color = color_for_label(ann.label)
        cv2.rectangle(rendered, (x1, y1), (x2, y2), color, box_thickness)
        text = ann.label
        (text_w, text_h), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)
        bg_x2 = min(rendered.shape[1] - 1, x1 + text_w + pad_x)
        bg_y1 = max(0, y1 - text_h - pad_y - baseline)
        cv2.rectangle(rendered, (x1, bg_y1), (bg_x2, y1), color, -1)
        cv2.putText(
            rendered,
            text,
            (x1 + max(4, pad_x // 3), max(text_h + 2, y1 - max(5, pad_y // 2))),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (255, 255, 255),
            font_thickness,
        )
    return cv2.cvtColor(rendered, cv2.COLOR_BGR2RGB)


def show_image_compat(target, image, width: Optional[int] = None):
    try:
        target.image(image, use_container_width=True)
    except TypeError:
        if width is not None:
            target.image(image, width=width)
        else:
            target.image(image)


def show_dataframe_compat(target, dataframe: pd.DataFrame):
    try:
        target.dataframe(dataframe, hide_index=True, use_container_width=True)
    except TypeError:
        try:
            target.dataframe(dataframe, hide_index=True)
        except TypeError:
            target.dataframe(dataframe)


def show_processing_popup():
    popup = st.empty()
    popup.markdown(
        """
        <div class="verifygn-overlay">
            <div class="verifygn-popup">
                <div class="verifygn-spinner"></div>
                <h3>VERIFYGN DETECTION SOFTWARE</h3>
                <p>VERIFYGN DETECTION SOFTWARE IS PROCESS YOUR UPLOADED FILE SO PLEASE WAIT A MINUTE WILL BE SHOWN THE RESULT</p>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    return popup


def counts_dataframe(anns: List[Annotation]) -> pd.DataFrame:
    counts = Counter(ann.label for ann in anns)
    rows = [{"Class": label, "Count": count} for label, count in counts.most_common(config.SHOW_CLASS_TABLE_MAX_ROWS)]
    return pd.DataFrame(rows, columns=["Class", "Count"])


def reset_app():
    st.session_state.pop("sam3_result", None)
    st.session_state.pop("system_camera_frame_rgb", None)
    st.session_state.pop("system_camera_frame_name", None)
    st.session_state["input_nonce"] = st.session_state.get("input_nonce", 0) + 1


def render_input_panel():
    nonce = st.session_state.get("input_nonce", 0)
    st.markdown("### Inputs")
    st.markdown('<div class="panel-card">', unsafe_allow_html=True)
    source_options = ["Upload Image File"]
    if config.ENABLE_BROWSER_CAMERA:
        source_options.append("Browser Camera")
    if config.ENABLE_SYSTEM_WEBCAM:
        source_options.append("System Webcam")

    input_mode = st.radio("Input source", source_options, horizontal=True, key=f"input_mode_{nonce}")

    payload: Dict[str, object] = {"kind": None, "raw_rgb": None, "name": ""}
    if input_mode == "Upload Image File":
        selected_item = st.file_uploader(
            "Choose an image file",
            type=[ext.lstrip(".") for ext in sorted(IMAGE_EXTS)],
            accept_multiple_files=False,
            key=f"upload_file_{nonce}",
        )
        if selected_item is not None:
            payload = {
                "kind": "uploaded_file",
                "file": selected_item,
                "raw_rgb": decode_uploaded_image(selected_item),
                "name": getattr(selected_item, "name", "uploaded_image"),
            }
    elif input_mode == "Browser Camera":
        st.info(
            "If the browser camera area stays blank, allow Camera permission in the browser and Windows privacy settings. "
            "If you opened the app with a network IP, try localhost or use the System Webcam option below."
        )
        selected_item = st.camera_input("Capture image from browser camera", key=f"camera_file_{nonce}")
        if selected_item is not None:
            payload = {
                "kind": "uploaded_file",
                "file": selected_item,
                "raw_rgb": decode_uploaded_image(selected_item),
                "name": getattr(selected_item, "name", "browser_camera.png"),
            }
    else:
        st.caption("Capture a frame directly from the PC webcam without relying on browser camera permission.")
        if st.button("Capture From System Webcam", use_container_width=True, key=f"capture_system_camera_{nonce}"):
            frame_rgb = capture_system_webcam_frame()
            if frame_rgb is None:
                st.error(
                    "Could not read from the system webcam. Close other apps using the camera and check Windows camera privacy permission."
                )
            else:
                st.session_state["system_camera_frame_rgb"] = frame_rgb
                st.session_state["system_camera_frame_name"] = f"system_webcam_{config.SYSTEM_CAMERA_INDEX}.png"
        saved_frame = st.session_state.get("system_camera_frame_rgb")
        if saved_frame is not None:
            payload = {
                "kind": "rgb_array",
                "raw_rgb": saved_frame,
                "name": st.session_state.get("system_camera_frame_name", "system_webcam.png"),
            }

    button_cols = st.columns([2.3, 1])
    detect_clicked = button_cols[0].button("Detect", type="primary", use_container_width=True)
    reset_clicked = button_cols[1].button("Reset", use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)
    return input_mode, payload, detect_clicked, reset_clicked


def run_detection(payload: Dict[str, object]):
    if not config.MODEL_PATH.exists():
        st.error(f"Model file not found. Place your SAM3 checkpoint here: {config.MODEL_PATH}")
        return

    prompt_specs = build_prompt_specs()
    if not prompt_specs:
        st.error("No prompt/class mappings are configured. Update PROMPT_CLASS_MAP in config.py.")
        return

    raw_rgb = payload.get("raw_rgb")
    if not payload.get("kind") or raw_rgb is None:
        st.error("Add an image file or camera capture before detection.")
        return

    backend = get_backend(
        str(config.MODEL_PATH),
        config.HARD_CODED_CONFIDENCE,
        config.HARD_CODED_IOU,
        config.USE_FP16,
        config.HARD_CODED_IMGSZ,
    )

    temp_path = None
    popup = show_processing_popup()
    try:
        if payload["kind"] == "uploaded_file":
            temp_path = save_uploaded_to_temp(payload["file"])
        else:
            temp_path = save_rgb_to_temp(raw_rgb)
        anns = backend.run_text(str(temp_path), prompt_specs)
        detected_rgb = draw_detections(str(temp_path), anns)
        st.session_state["sam3_result"] = {
            "name": str(payload.get("name") or "input_image"),
            "raw_image": raw_rgb,
            "detected_image": detected_rgb,
            "counts_df": counts_dataframe(anns),
            "total_count": len(anns),
            "unique_classes": len({ann.label for ann in anns}),
        }
    except Exception as exc:
        st.error(str(exc))
    finally:
        popup.empty()
        if temp_path is not None and temp_path.exists():
            temp_path.unlink(missing_ok=True)


def render_result_panels(current_raw_rgb: Optional[np.ndarray]):
    result = st.session_state.get("sam3_result")
    raw_for_display = current_raw_rgb
    if raw_for_display is None and result is not None:
        raw_for_display = result.get("raw_image")

    preview_cols = st.columns(2)
    with preview_cols[0]:
        st.markdown("### Raw Upload Image")
        if raw_for_display is not None:
            show_image_compat(st, raw_for_display)
        else:
            st.info("Upload an image or capture from camera to preview the raw input here.")

    with preview_cols[1]:
        st.markdown("### Detect Image")
        if result is not None:
            show_image_compat(st, result["detected_image"])
            st.caption(config.RESULT_IMAGE_CAPTION)
        else:
            st.info("Detected image with bounding boxes will appear here after clicking Detect.")


def render_results_table():
    result = st.session_state.get("sam3_result")
    st.markdown("### Result of Class and Counts")
    if result is None:
        st.info("Detection results will appear here after running Detect.")
        return

    metric_cols = st.columns(2)
    metric_cols[0].metric("Detected Boxes", result["total_count"])
    metric_cols[1].metric("Detected Classes", result["unique_classes"])

    counts_df = result["counts_df"]
    if counts_df.empty:
        st.warning("No detections found for the configured prompt/class mapping.")
        return

    chips = "".join(
        f"<span class='count-chip'>{row['Class']}: {row['Count']}</span>"
        for _, row in counts_df.iterrows()
    )
    st.markdown(chips, unsafe_allow_html=True)
    show_dataframe_compat(st, counts_df)


def main():
    st.set_page_config(
        page_title=config.APP_TITLE,
        layout="wide",
        initial_sidebar_state="collapsed",
    )
    inject_css()
    render_header()

    input_mode, payload, detect_clicked, reset_clicked = render_input_panel()

    if reset_clicked:
        reset_app()
        st.rerun()

    if detect_clicked:
        run_detection(payload)

    render_result_panels(payload.get("raw_rgb"))
    render_results_table()


if __name__ == "__main__":
    if is_running_in_streamlit():
        main()
    else:
        launch_with_streamlit_cli()
