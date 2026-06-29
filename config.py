from __future__ import annotations

import os
from pathlib import Path

APP_DIR = Path(__file__).resolve().parent


def env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() not in {"0", "false", "no", "off"}


def env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return default


def env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return default


def legacy_name(name: str) -> str:
    return name.replace("VERIFYGNCLOUD", "ROBOFLOW")


def env_text(name: str, default: str = "") -> str:
    value = os.getenv(name)
    if value is None:
        value = os.getenv(legacy_name(name), default)
    return str(value or default).strip()


# Branding: edit these values to customize the operator UI.
APP_TITLE = "VERIFYGN-DEFECT DETECTOR"
APP_SUBTITLE = "Upload or capture an image, detect configured classes, and review per-class counts."
LOGO_PATH = APP_DIR / "assets" / "VFN_logo.png"

# Inference backend:
# - "verifygncloud" calls your cloud workflow API and does not download/load a model
#   on Streamlit Cloud.
# - "local" uses the local model settings below.
INFERENCE_BACKEND = os.getenv("INFERENCE_BACKEND", "verifygncloud").strip().lower() or "verifygncloud"

# VERIFYGNCLOUD settings. Keep the API key in Streamlit Secrets:
# VERIFYGNCLOUD_API_KEY = "your_key_here"
# As a last-resort deployment fallback, you can hardcode the key here.
# Avoid doing that in a public GitHub repo.
VERIFYGNCLOUD_API_KEY = env_text("VERIFYGNCLOUD_API_KEY", "")
VERIFYGNCLOUD_API_URL = env_text("VERIFYGNCLOUD_API_URL", "https://serverless.roboflow.com")
VERIFYGNCLOUD_API_KEY_ENV = env_text("VERIFYGNCLOUD_API_KEY_ENV", "VERIFYGNCLOUD_API_KEY") or "VERIFYGNCLOUD_API_KEY"
VERIFYGNCLOUD_WORKSPACE = env_text("VERIFYGNCLOUD_WORKSPACE", "rajkumarm")
VERIFYGNCLOUD_WORKFLOW_ID = env_text("VERIFYGNCLOUD_WORKFLOW_ID", "general-segmentation-api")
VERIFYGNCLOUD_IMAGE_INPUT = env_text("VERIFYGNCLOUD_IMAGE_INPUT", "image") or "image"
VERIFYGNCLOUD_CLASSES_INPUT = env_text("VERIFYGNCLOUD_CLASSES_INPUT", "classes") or "classes"
VERIFYGNCLOUD_ANNOTATED_OUTPUT = env_text("VERIFYGNCLOUD_ANNOTATED_OUTPUT", "annotated_image") or "annotated_image"
VERIFYGNCLOUD_PREDICTIONS_OUTPUT = env_text("VERIFYGNCLOUD_PREDICTIONS_OUTPUT", "predictions") or "predictions"
VERIFYGNCLOUD_TIMEOUT_SEC = env_int("VERIFYGNCLOUD_TIMEOUT_SEC", env_int(legacy_name("VERIFYGNCLOUD_TIMEOUT_SEC"), 180))

# Script-side model placement.
# Lightweight YOLOE segmentation model (~28 MB) with text prompts, instead of
# the ~3.2 GB SAM3 checkpoint that could not fit in Streamlit Cloud RAM.
MODEL_PATH = APP_DIR / "model" / "yoloe-11s-seg.pt"
# Cloud model source. Hardcoded to the Ultralytics YOLOE release so a stale
# Streamlit Secret cannot pull the old SAM3 model by accident.
MODEL_DOWNLOAD_URL = "https://github.com/ultralytics/assets/releases/download/v8.3.0/yoloe-11s-seg.pt"
MODEL_HF_REPO_ID = os.getenv("MODEL_HF_REPO_ID", "").strip()
MODEL_HF_FILENAME = os.getenv("MODEL_HF_FILENAME", "yoloe-11s-seg.pt").strip()
MODEL_HF_REVISION = os.getenv("MODEL_HF_REVISION", "main").strip() or "main"
MODEL_HF_TOKEN_ENV = os.getenv("MODEL_HF_TOKEN_ENV", "HF_TOKEN").strip() or "HF_TOKEN"
MODEL_DOWNLOAD_TIMEOUT_SEC = env_int("MODEL_DOWNLOAD_TIMEOUT_SEC", 3600)

# Prevent Streamlit Cloud from being killed by very large model downloads/loads.
MODEL_RESOURCE_GUARD_ENABLED = env_bool("MODEL_RESOURCE_GUARD_ENABLED", True)
MODEL_LOAD_RAM_MULTIPLIER = env_float("MODEL_LOAD_RAM_MULTIPLIER", 2.5)
MODEL_MIN_FREE_DISK_GB = env_float("MODEL_MIN_FREE_DISK_GB", 1.0)

# Hardcoded detection settings: intentionally not exposed in the UI.
HARD_CODED_CONFIDENCE = 0.01
HARD_CODED_IOU = 0.50
HARD_CODED_IMGSZ = 2560
USE_FP16 = True

# Prompt/class mappings: also script-side only.
PROMPT_CLASS_MAP = [
    ("concave surface deformation, depressed area on the surface, circular indentation on metal, light distortion from a dent on a painted surface, crushed inward section, shadowed depression indicating a dent, irregular surface concavity,dent on metal surface, shallow depression in metal, deformed dented spot dust, dent", "DENT"),
    ("structural deformity, broken or missing material, compromised surface integrity, chipped edge, broken contour, fractured corner,gouge in the material, mangled surface texture, abrasion area, damaged area on surface, broken deformed region,surface defect, damage, crush", "DAMAGE"),
    ("thin linear surface abrasion, scuff mark, contrasting scratch line, discolored hairline scratch, deep linear gouge, white scratch on clear coat, reflective scratch on brushed metal, surface scraping, scratch mark on surface, thin linear scratch, scratch line, scratch", "SCRATCH"),
    ("hairline fracture, structural fissure, branching material separation,dark jagged line indicating a crack, spiderweb fracture pattern, split in the metallic housing, shattered pattern in glass/plastic, deep surface split, crack on surface, hairline fracture line, crack line in metal, crack", "CRACK"),
    ("crumpled structural zone, massive inward deformation, crushed and folded material, severe impact damage, shattered and displaced components, buckled surface area, crushed metal area, impact damage region, deformed crushed surface, crash", "CRASH"),
    #("", "SCRATCH"),
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
