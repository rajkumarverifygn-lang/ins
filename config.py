from __future__ import annotations

from pathlib import Path

APP_DIR = Path(__file__).resolve().parent

# Branding: edit these values to customize the operator UI.
APP_TITLE = "INSERT NUT DETECTOR"
APP_SUBTITLE = "Upload or capture an image, detect configured classes, and review per-class counts."
LOGO_PATH = APP_DIR / "assets" / "VFN_logo.png"

# Script-side model placement.
MODEL_PATH = APP_DIR / "model" / "sam3.pt"

# Hardcoded detection settings: intentionally not exposed in the UI.
HARD_CODED_CONFIDENCE = 0.65
HARD_CODED_IOU = 0.50
HARD_CODED_IMGSZ = 644
USE_FP16 = True

# Prompt/class mappings: also script-side only.
PROMPT_CLASS_MAP = [
    ("GOLD", "INSERT_NUT"),
    #("person wearing helmet", "helmet_person"),
    #("car", "vehicle"),
]

# UI behavior.
ENABLE_BROWSER_CAMERA = True
ENABLE_SYSTEM_WEBCAM = True
SYSTEM_CAMERA_INDEX = 0
SYSTEM_CAMERA_WARMUP_FRAMES = 4
SHOW_CLASS_TABLE_MAX_ROWS = 50
RESULT_IMAGE_CAPTION = "Bounding boxes and class labels are shown. Confidence is hidden by design."

# Detection drawing style.
BBOX_THICKNESS = 4
LABEL_FONT_SCALE = 1.05
LABEL_FONT_THICKNESS = 3
LABEL_BOX_PADDING_X = 14
LABEL_BOX_PADDING_Y = 12

# Backward-compatible aliases for older Streamlit app revisions.
ENABLE_CAMERA_INPUT = ENABLE_BROWSER_CAMERA
DEFAULT_LOCAL_IMAGE_DIR = APP_DIR / "input_images"
DEFAULT_PROMPT_ROWS = [{"Prompt": prompt, "Class": label} for prompt, label in PROMPT_CLASS_MAP]
